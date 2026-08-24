"""Pure continuous-motion steering prototype with no dispatch capability.

This module deliberately contains no Home Assistant, coordinator, BLE, or
service imports. It answers one question only: given a bounded route and one
telemetry observation, what command would a future continuous controller ask
an integration-owned executor to hold until the next observation?

The executor does not exist yet. A stop decision here is data, not a mower
command. Keeping that boundary explicit lets replay and fault tests mature
before any physical-motion service is designed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

ControllerAction = Literal["drive", "stop"]

# 🚨 **THE SIGN THAT LOST THE FIRST PHYSICAL PHASE 2 RUN, 2026-08-24.**
#
# A POSITIVE commanded `angular_speed` makes `course_heading_degrees` go DOWN.
# Until this constant existed the steering law assumed the opposite and was
# therefore POSITIVE feedback: a positive heading error commanded a positive
# angular correction, which grew the very error it was correcting.
#
# Two separately established facts compose to give this, and neither one is
# enough on its own -- which is exactly why the bug survived review:
#
#   1. `angular_speed > 0` INCREASES `position.toward`. Measured in place with
#      VIO off (`docs/toward-tracks-in-place-rotation-20260812.md`): angular
#      +500 moved `toward` +99.55 deg, angular -500 moved it -61.43 deg.
#   2. `course_heading_degrees` is a REFLECTION of `toward`, not an offset:
#      `_continuous_course_heading()` in `services.py` returns
#      `(90.13 - toward) % 360`. A reflection REVERSES the sense of rotation.
#
# So d(course)/d(angular) = d(course)/d(toward) x d(toward)/d(angular)
#                         = (-1) x (+ve) = NEGATIVE.
#
# Verified offline against six banked captures, on BOTH command signs, using
# the mirror-derived course AND a travel bearing computed independently from
# raw position deltas (`atan2(dy, dx)`), which never touches `toward` at all:
#
#   capture                              linear angular d(toward) d(course)
#   Phase 2 first physical run 20260824    400    +180    +14.42    -14.42
#   Phase 1b certified arc     20260823    400    +180    +64.18    -64.18
#   arc120 out-of-sample       20260823    400    +120    +47.06    -47.06
#   arc sweep a300             20260812    400    +300    +43.42    -43.42
#   arc sweep a500             20260812    400    +500    +66.85    -66.85
#   night pivot (reverse sign) 20260812      0    -500    -61.43    +61.43
#
# The last row is the one that makes this a measured law rather than a
# one-directional coincidence: flipping the command sign flips the response.
#
# ⚠️ **Do NOT "fix" a future divergence by flipping this back.** It is pinned by
# `test_positive_angular_lowers_map_course_so_the_gain_is_applied_negatively`
# and by the 2026-08-24 divergence regression test, both of which encode the
# measurement above rather than the code's own behaviour.
# ⚠️ This says nothing about the MAGNITUDE of the response, which is measured
# and non-proportional -- see `angular_speed_per_heading_degree` below.
_ANGULAR_COMMAND_SIGN_PER_COURSE_DEGREE = -1.0


@dataclass(frozen=True)
class Point:
    """One map-local point in metres."""

    x: float
    y: float


@dataclass(frozen=True)
class ContinuousRoute:
    """One straight, prevalidated route used by the feasibility controller.

    `contained` is caller-supplied and never re-derived here: this module does
    no live keep-out or area-edge geometry check of its own. The caller is
    responsible for scanning and freezing a corridor before the first decision
    -- the same discipline as `scripts/freeze_phase1_corridors.py` and
    `scripts/scan_contained_bearings.py` already use for Phase 1 captures.
    Confirmed 2026-08-23 while reconciling this module against this week's
    measurements: no such check exists anywhere in this file.
    """

    start: Point
    target: Point
    contained: bool = True


@dataclass(frozen=True)
class ContinuousControllerConfig:
    """Provisional bounds for offline replay, not hardware-accepted settings."""

    linear_speed: int = 400
    max_abs_angular_speed: int = 180
    # A STEERING GAIN, not a yaw-RATE model. This converts a heading error into
    # a commanded angular_speed for the NEXT pulse -- "how hard do I turn to
    # fix this" -- which is a different question from "how fast will the mower
    # actually rotate at a given angular_speed command", the relationship
    # measured and found NON-proportional this week
    # (`docs/prediction-model-holds-out-of-sample-20260823.md`: a 33% cut in
    # commanded angular moved the observed yaw rate by only 3%). The two are
    # easy to conflate because both involve "angular speed" and "degrees".
    # This constant is unaffected by that finding; it is not the thing that was
    # refuted.
    # ⚠️ **MAGNITUDE ONLY -- it carries no sign.** It stays positive and is
    # validated positive in `__post_init__`; the physical direction of the
    # correction lives in `_ANGULAR_COMMAND_SIGN_PER_COURSE_DEGREE` and nowhere
    # else. Do not encode a direction here by making it negative: a negative
    # gain would be rejected by the positivity check, and burying the sign in a
    # tuning knob is how the 2026-08-24 inversion stayed invisible.
    angular_speed_per_heading_degree: float = 12.0
    heading_deadband_degrees: float = 1.5
    # 🚨 **Added 2026-08-24 after the FIRST physical Phase 2 run**
    # (`docs/evidence-phase2-first-physical-run-20260824.json`). That run's very
    # first decision fired at `elapsed_s: 0.458` with `distance_travelled_m:
    # 0.0` -- the mower had not yet moved a millimetre inside this window -- and
    # immediately commanded `angular_speed: 180`, the saturated maximum, off a
    # 46.6 deg heading error.
    #
    # `course_heading_degrees` is derived from `toward`, which is
    # COURSE-OVER-GROUND, not a body heading a stationary machine can report.
    # Two separately measured properties make an opening sample untrustworthy:
    # `toward` is bit-identical through a bounded pulse and arrives as one
    # post-hoc step (`docs/night-toward-latency-20260813.md`), and it LATCHES on
    # straight motion, updating once in 8 s
    # (`docs/corrections-to-the-20260822-analysis-20260823.md`). So the value
    # present when a window opens describes whatever the mower last did -- which
    # may be a previous session -- and the controller cannot tell that apart
    # from a live measurement. It saturated on it anyway.
    #
    # Below this much confirmed displacement THIS window the controller drives
    # straight instead of steering, which is exactly the convention the executor
    # already uses for its opening command ("Opening command is straight --
    # angular_speed=0", `services.py`). It is a gate on the CORRECTION only; it
    # never suppresses a stop, and every fail-closed check still runs first.
    #
    # 0.15 m is this project's own registered "a chord this short cannot test a
    # bearing" floor -- `MIN_MOVING_STEP_M` in
    # `scripts/analyze_phase1_capture.py`, raised 0.01 -> 0.15 on 2026-08-23
    # because at the measured sigma = 0.0031 m the bearing noise bound
    # `atan(sigma*sqrt(2)/chord)` is +-12.2 deg at 0.046 m and +-7.4 deg at
    # 0.076 m, at or above the criterion it was being tested against, while at
    # 0.15 m it falls to 1.7 deg (`docs/mirror-criterion-repaired-20260823.md`).
    # The same arithmetic governs here: steering on a bearing whose own noise
    # bound is larger than the error being corrected is not feedback.
    # ⚠️ It is a FLOOR ON INFORMATIVENESS, not a proof of freshness. Nothing in
    # a single observation can prove `toward` was recomputed this window; see
    # the replay note in the 2026-08-24 report.
    min_travel_for_heading_trust_m: float = 0.15
    lookahead_m: float = 0.80
    waypoint_tolerance_m: float = 0.15
    max_cross_track_m: float = 0.30
    max_telemetry_age_s: float = 2.0
    # How long since the most recent refresh write completed, sampled at THIS
    # decision instant. This is a basic "is the refresh loop still alive at
    # all" bound, not a stall detector -- see `max_refresh_gap_s` below for why
    # those are different questions and both are needed.
    max_refresh_age_s: float = 1.20
    # 🔑 **The BLE-stall detector, added 2026-08-23.** `max_refresh_age_s`
    # alone is NOT sufficient: replaying this controller against
    # `docs/evidence-8s-continuous-window-20260822T233000Z.json`
    # (`scripts/replay_continuous_controller_against_capture.py`) showed a real
    # 810 ms stall -- the one that produced the largest prediction error in the
    # whole corpus, 0.1418 m -- go COMPLETELY UNDETECTED by refresh_age_s,
    # because a fast recovery write (106 ms) happened to complete essentially
    # simultaneously with the next ~1 Hz decision, so "time since most recent
    # completion, sampled now" read ~0 s at the exact instant checked. A stall
    # that resolves between two decision instants is invisible to a point
    # sample. `max_refresh_gap_s` instead bounds the WORST gap between any two
    # consecutive completions since the last decision -- the caller must track
    # a running max, not just the most recent timestamp. 0.60 s matches the
    # registered `3R` cadence-stall rule at the 200 ms app refresh interval
    # (`docs/phase1b-arc-protocol-20260823.md`).
    max_refresh_gap_s: float = 0.60
    max_window_s: float = 4.0
    max_distance_m: float = 1.50
    # v@400 measured from 16 steady-state in-window steps across three straight
    # captures on 2026-08-22 (`docs/frozen-prediction-constants-20260822.json`,
    # k_lin = 6.204299e-04). Was 0.28, a provisional Phase 0 guess that predates
    # the measurement.
    nominal_speed_mps: float = 0.2482
    # 🚨 **The SECOND defect the 2026-08-24 run exposed** -- independent of the
    # sign inversion, and NOT fixed by it. That run opened its window 46.639 deg
    # misaligned. At every plausible turn rate this project has measured, nulling
    # that error while driving forward overruns the 0.30 m cross-track bound
    # before the mower finishes turning, so the window was geometrically lost
    # before the first command went out. See `alignment_feasibility()`.
    #
    # A CONSERVATIVE LOWER BOUND on the map-frame course rate at
    # `max_abs_angular_speed`, re-derived here from raw in-window position fixes
    # rather than quoted:
    #
    #   angular +180, Phase 1b certified arc 20260823:
    #       8.091 deg/s including the spin-up step, 8.964 deg/s excluding it
    #   angular +120, out-of-sample arc       20260823:
    #       6.936 deg/s including the spin-up step, 7.808 deg/s excluding it
    #
    # (The excluding-spin-up figures reproduce the published 9.386 / 7.813 in
    # `docs/phase2-gate-readiness-20260823.md` -- the 120 case to 0.06% -- which
    # is the cross-check that says this re-derivation is reading the captures
    # the same way that analysis did.)
    #
    # 8.0 sits below the slowest of those. SLOWER IS THE SAFE DIRECTION for a
    # refusal gate: a lower rate means more turn time, more turn distance, and a
    # LARGER predicted arc excursion, so it refuses more, never fewer.
    # ⚠️ Measured only across angular 120-180. This project has REFUTED
    # proportionality between commanded angular and observed yaw rate
    # (`docs/prediction-model-holds-out-of-sample-20260823.md`: a 33% cut in
    # command moved the rate 3%), so do NOT scale this by
    # `max_abs_angular_speed`; outside that band it is simply unmeasured.
    min_turn_rate_deg_per_s: float = 8.0
    max_prediction_horizon_s: float = 1.25

    def __post_init__(self) -> None:
        """Reject nonsensical bounds before replay begins."""

        positive = {
            "linear_speed": self.linear_speed,
            "max_abs_angular_speed": self.max_abs_angular_speed,
            "angular_speed_per_heading_degree": (self.angular_speed_per_heading_degree),
            # Rejected at zero on purpose: a 0.0 here silently restores the
            # steer-before-you-have-moved behaviour of the 2026-08-24 run.
            "min_travel_for_heading_trust_m": self.min_travel_for_heading_trust_m,
            "lookahead_m": self.lookahead_m,
            "waypoint_tolerance_m": self.waypoint_tolerance_m,
            "max_cross_track_m": self.max_cross_track_m,
            "max_telemetry_age_s": self.max_telemetry_age_s,
            "max_refresh_age_s": self.max_refresh_age_s,
            "max_refresh_gap_s": self.max_refresh_gap_s,
            "max_window_s": self.max_window_s,
            "max_distance_m": self.max_distance_m,
            "nominal_speed_mps": self.nominal_speed_mps,
            # Rejected at zero on purpose: a 0.0 rate makes every turn take
            # forever, which would make `alignment_feasibility()` refuse
            # everything, and a negative one would silently invert its
            # arithmetic into passing everything.
            "min_turn_rate_deg_per_s": self.min_turn_rate_deg_per_s,
            "max_prediction_horizon_s": self.max_prediction_horizon_s,
        }
        invalid = [
            name
            for name, value in positive.items()
            if not math.isfinite(float(value)) or value <= 0
        ]
        if invalid:
            raise ValueError(f"controller bounds must be positive: {invalid}")
        if (
            not math.isfinite(self.heading_deadband_degrees)
            or self.heading_deadband_degrees < 0
        ):
            raise ValueError("heading_deadband_degrees must be non-negative")
        if self.linear_speed > 1000 or self.max_abs_angular_speed > 1000:
            raise ValueError("controller commands must stay within mower schema bounds")


@dataclass(frozen=True)
class ContinuousObservation:
    """One executor-independent snapshot presented to the pure controller."""

    position: Point
    course_heading_degrees: float
    telemetry_age_s: float
    refresh_age_s: float
    elapsed_s: float
    distance_travelled_m: float
    # The worst gap between any two CONSECUTIVE refresh completions since the
    # previous decision -- a running max the caller must track, not a
    # point-sample. See `ContinuousControllerConfig.max_refresh_gap_s` for why
    # this is a separate field from `refresh_age_s` rather than a replacement
    # for it. Defaults to 0.0 (no stall observed) so existing callers that do
    # not yet track this are unaffected -- unlike `elapsed_s` and
    # `distance_travelled_m` above, which stay required: those two directly
    # gate `window_limit_reached` and `distance_limit_reached`, and a silent
    # default would let a caller bypass both without noticing.
    refresh_max_gap_since_last_decision_s: float = 0.0
    stop_available: bool = True
    ble_live: bool = True
    refresh_healthy: bool = True
    position_valid: bool = True
    inside_area: bool = True
    rtk_fixed: bool = True
    blades_off: bool = True
    work_mode_safe: bool = True
    cancelled: bool = False


@dataclass(frozen=True)
class ContinuousDecision:
    """A requested command or fail-closed stop returned by the prototype."""

    action: ControllerAction
    reason: str
    linear_speed: int
    angular_speed: int
    predicted_position: Point
    aim_point: Point | None = None
    target_distance_m: float | None = None
    along_track_m: float | None = None
    cross_track_m: float | None = None
    observed_cross_track_m: float | None = None
    desired_course_degrees: float | None = None
    heading_error_degrees: float | None = None
    # None on every stop path (nothing was steered), matching the convention the
    # optional fields above already use. On a drive decision, False means the
    # heading error WAS computed and deliberately not acted on because this
    # window has not yet observed
    # `ContinuousControllerConfig.min_travel_for_heading_trust_m` of travel.
    # ⚠️ A future reader will otherwise see `angular_speed: 0` next to a large
    # `heading_error_degrees` in an evidence file and read it as a bug.
    heading_confirmed_by_motion: bool | None = None
    prediction_horizon_s: float = 0.0


@dataclass(frozen=True)
class AlignmentFeasibility:
    """Whether the OPENING heading error can be nulled inside the window budget.

    A preflight verdict, computed once before a window opens. It is NOT a
    runtime check and deliberately never runs inside the decision loop: the
    controller's job mid-window is to steer and to hard-abort on cross-track,
    not to re-litigate whether the window should have started.
    """

    heading_error_degrees: float
    # PHASE 1 -- the blind run. The controller withholds every correction until
    # `min_travel_for_heading_trust_m` of travel is confirmed, so the window
    # opens by driving STRAIGHT at the full opening error for that distance.
    blind_travel_m: float
    blind_time_s: float
    blind_cross_track_m: float
    # PHASE 2 -- the turn that actually nulls the error.
    turn_time_s: float
    turn_distance_m: float
    turn_cross_track_m: float
    # Both phases together: what nulling the error is PREDICTED to cost.
    total_time_s: float
    total_distance_m: float
    total_cross_track_m: float
    window_budget_s: float
    distance_budget_m: float
    cross_track_budget_m: float
    feasible: bool
    # Which budget the correction breaks first, or None when it fits.
    limiting_factor: str | None


def normalize_degrees(angle: float) -> float:
    """Normalize an angle to [-180, 180)."""

    return (angle + 180.0) % 360.0 - 180.0


def alignment_feasibility(
    route: ContinuousRoute,
    course_heading_degrees: float,
    config: ContinuousControllerConfig | None = None,
) -> AlignmentFeasibility:
    """Predict whether the opening heading error fits the window's own budgets.

    🚨 **Why this exists.** The 2026-08-24 run had TWO independent defects. The
    steering sign was one (see `_ANGULAR_COMMAND_SIGN_PER_COURSE_DEGREE`). The
    other is that it opened 46.639 deg misaligned, which no correctly-signed
    controller could have rescued inside a 4 s / 0.30 m window -- the mower
    drives forward the whole time it is turning, so the correction itself sweeps
    the machine off the line.

    The cost is modelled in the TWO phases the controller actually executes.

    **Phase 1, the blind run.** `min_travel_for_heading_trust_m` (0.15 m) of
    travel must be confirmed before any correction is issued at all, so the
    window opens driving straight at the full opening error::

        blind_cross_track = blind_travel * sin(theta0)

    **Phase 2, the turn.** Driving at speed `v` while the course rotates at a
    constant `w` toward the route, cross-track accumulates at `v * sin(theta)`
    where `theta` is the remaining error, and `dt = dtheta / w`, so::

        turn_cross_track = integral of (v/w) * sin(theta) dtheta, 0 -> theta0
                         = (v / w) * (1 - cos(theta0))

    with `w` in radians per second. At the conservative `w` = 8.0 deg/s and
    `v` = 0.2482 m/s that second term is `1.778 m * (1 - cos theta0)`.

    🔑 **Both phases are needed and modelling only the turn is not safe.** The
    turn term alone reaches 0.30 m at theta0 ~= 34 deg, but adding the blind
    run's own excursion brings that down to **29.25 deg** -- so a gate built on
    the turn term alone would admit a 32 deg opening that really costs 0.3496 m
    against a 0.30 m bound. On the default 4 s window the time limb is tighter
    still (**27.17 deg**), because the blind run spends 0.604 s before the turn
    can even start.

    ⚠️ Today's 46.639 deg breaches 0.30 m at EVERY rate in the disputed
    turn-rate range, the optimistic 11.224 deg/s single-pulse fit included. The
    verdict does not depend on resolving that discrepancy.

    Three budgets are tested against the PHASE TOTALS and the first breached is
    reported: `window_s` (blind run plus turn must fit the window at all),
    `distance_m` (their travel must fit the distance guard), and
    `cross_track_m` (their combined excursion).

    ⚠️ **Assumptions, stated rather than hidden.** `w` is a lower bound, not a
    model -- real rate varies and spin-up is not instantaneous. The desired
    course is the ROUTE bearing; the caller's `start_drift_within_bound` gate
    already holds live position within 0.30 m of the frozen start, so the
    difference is small but nonzero. Both phases are assumed to steer perfectly
    once they begin, and the excursion is assumed to start from the centreline.
    Passing here is NECESSARY, not sufficient -- the runtime cross-track abort
    remains the real protection.
    """

    settings = config or ContinuousControllerConfig()
    route_dx = route.target.x - route.start.x
    route_dy = route.target.y - route.start.y
    finite = all(
        math.isfinite(float(value))
        for value in (route_dx, route_dy, course_heading_degrees)
    )
    route_length = math.hypot(route_dx, route_dy) if finite else 0.0
    if not finite or route_length <= 1e-9:
        # Fail closed: an unusable route or heading is never "feasible".
        infinite = float("inf")
        return AlignmentFeasibility(
            heading_error_degrees=float("nan"),
            blind_travel_m=settings.min_travel_for_heading_trust_m,
            blind_time_s=infinite,
            blind_cross_track_m=infinite,
            turn_time_s=infinite,
            turn_distance_m=infinite,
            turn_cross_track_m=infinite,
            total_time_s=infinite,
            total_distance_m=infinite,
            total_cross_track_m=infinite,
            window_budget_s=settings.max_window_s,
            distance_budget_m=settings.max_distance_m,
            cross_track_budget_m=settings.max_cross_track_m,
            feasible=False,
            limiting_factor="route_or_heading_unusable",
        )

    desired_course = math.degrees(math.atan2(route_dy, route_dx))
    heading_error = normalize_degrees(desired_course - course_heading_degrees)
    error_radians = math.radians(abs(heading_error))

    # Phase 1: driving straight at the full error until the heading is trusted.
    blind_travel_m = settings.min_travel_for_heading_trust_m
    blind_time_s = blind_travel_m / settings.nominal_speed_mps
    blind_cross_track_m = blind_travel_m * math.sin(error_radians)

    # Phase 2: the turn that nulls it, still driving forward throughout.
    turn_time_s = abs(heading_error) / settings.min_turn_rate_deg_per_s
    turn_distance_m = settings.nominal_speed_mps * turn_time_s
    turn_radius_m = settings.nominal_speed_mps / math.radians(
        settings.min_turn_rate_deg_per_s
    )
    turn_cross_track_m = turn_radius_m * (1.0 - math.cos(error_radians))

    total_time_s = blind_time_s + turn_time_s
    total_distance_m = blind_travel_m + turn_distance_m
    total_cross_track_m = blind_cross_track_m + turn_cross_track_m

    breaches = (
        (total_time_s > settings.max_window_s, "window_s"),
        (total_distance_m > settings.max_distance_m, "distance_m"),
        (total_cross_track_m > settings.max_cross_track_m, "cross_track_m"),
    )
    limiting_factor = next((name for failed, name in breaches if failed), None)
    return AlignmentFeasibility(
        heading_error_degrees=heading_error,
        blind_travel_m=blind_travel_m,
        blind_time_s=blind_time_s,
        blind_cross_track_m=blind_cross_track_m,
        turn_time_s=turn_time_s,
        turn_distance_m=turn_distance_m,
        turn_cross_track_m=turn_cross_track_m,
        total_time_s=total_time_s,
        total_distance_m=total_distance_m,
        total_cross_track_m=total_cross_track_m,
        window_budget_s=settings.max_window_s,
        distance_budget_m=settings.max_distance_m,
        cross_track_budget_m=settings.max_cross_track_m,
        feasible=limiting_factor is None,
        limiting_factor=limiting_factor,
    )


def _predict_position(
    observation: ContinuousObservation,
    config: ContinuousControllerConfig,
) -> tuple[Point, float]:
    """Project a stale fix along the reported course within a hard horizon."""

    horizon = min(
        max(float(observation.telemetry_age_s), 0.0),
        config.max_prediction_horizon_s,
    )
    radians = math.radians(observation.course_heading_degrees)
    distance = config.nominal_speed_mps * horizon
    return (
        Point(
            observation.position.x + math.cos(radians) * distance,
            observation.position.y + math.sin(radians) * distance,
        ),
        horizon,
    )


def _stop(
    reason: str,
    predicted_position: Point,
    *,
    prediction_horizon_s: float,
    aim_point: Point | None = None,
    target_distance_m: float | None = None,
    along_track_m: float | None = None,
    cross_track_m: float | None = None,
    observed_cross_track_m: float | None = None,
    desired_course_degrees: float | None = None,
    heading_error_degrees: float | None = None,
) -> ContinuousDecision:
    return ContinuousDecision(
        action="stop",
        reason=reason,
        linear_speed=0,
        angular_speed=0,
        predicted_position=predicted_position,
        aim_point=aim_point,
        target_distance_m=target_distance_m,
        along_track_m=along_track_m,
        cross_track_m=cross_track_m,
        observed_cross_track_m=observed_cross_track_m,
        desired_course_degrees=desired_course_degrees,
        heading_error_degrees=heading_error_degrees,
        prediction_horizon_s=prediction_horizon_s,
    )


def _input_failure_reason(
    route: ContinuousRoute,
    observation: ContinuousObservation,
    settings: ContinuousControllerConfig,
) -> str | None:
    """Return the first fail-closed input reason, or None when inputs are safe."""

    numeric_inputs = (
        route.start.x,
        route.start.y,
        route.target.x,
        route.target.y,
        observation.position.x,
        observation.position.y,
        observation.course_heading_degrees,
        observation.telemetry_age_s,
        observation.refresh_age_s,
        observation.refresh_max_gap_since_last_decision_s,
        observation.elapsed_s,
        observation.distance_travelled_m,
    )
    if not all(math.isfinite(float(value)) for value in numeric_inputs):
        return "inputs_not_finite"
    if any(
        value < 0
        for value in (
            observation.telemetry_age_s,
            observation.refresh_age_s,
            observation.refresh_max_gap_since_last_decision_s,
            observation.elapsed_s,
            observation.distance_travelled_m,
        )
    ):
        return "observation_values_invalid"
    fail_closed_checks = (
        (observation.cancelled, "operator_cancelled"),
        (not observation.stop_available, "stop_primitive_unavailable"),
        (not route.contained, "route_not_prevalidated_contained"),
        (not observation.ble_live, "ble_link_not_live"),
        (not observation.refresh_healthy, "refresh_cadence_unhealthy"),
        (
            observation.refresh_age_s > settings.max_refresh_age_s,
            "refresh_age_exceeded",
        ),
        (
            observation.refresh_max_gap_since_last_decision_s
            > settings.max_refresh_gap_s,
            "refresh_cadence_stalled",
        ),
        (
            observation.telemetry_age_s > settings.max_telemetry_age_s,
            "telemetry_stale",
        ),
        (not observation.position_valid, "position_invalid"),
        (not observation.inside_area, "position_outside_area"),
        (not observation.rtk_fixed, "rtk_not_fixed"),
        (not observation.blades_off, "blades_not_off"),
        (not observation.work_mode_safe, "work_mode_not_safe"),
        (observation.elapsed_s >= settings.max_window_s, "window_limit_reached"),
        (
            observation.distance_travelled_m >= settings.max_distance_m,
            "distance_limit_reached",
        ),
    )
    return next((reason for failed, reason in fail_closed_checks if failed), None)


def continuous_control_decision(
    route: ContinuousRoute,
    observation: ContinuousObservation,
    config: ContinuousControllerConfig | None = None,
) -> ContinuousDecision:
    """Return one bounded steering decision without sending anything."""

    settings = config or ContinuousControllerConfig()
    input_failure = _input_failure_reason(route, observation, settings)
    if input_failure in {"inputs_not_finite", "observation_values_invalid"}:
        return _stop(
            input_failure,
            observation.position,
            prediction_horizon_s=0.0,
        )

    predicted, horizon = _predict_position(observation, settings)
    if input_failure is not None:
        return _stop(input_failure, predicted, prediction_horizon_s=horizon)

    route_dx = route.target.x - route.start.x
    route_dy = route.target.y - route.start.y
    route_length = math.hypot(route_dx, route_dy)
    if route_length <= 1e-9:
        return _stop("route_has_zero_length", predicted, prediction_horizon_s=horizon)

    ux, uy = route_dx / route_length, route_dy / route_length
    observed_rel_x = observation.position.x - route.start.x
    observed_rel_y = observation.position.y - route.start.y
    observed_cross_track = ux * observed_rel_y - uy * observed_rel_x
    rel_x = predicted.x - route.start.x
    rel_y = predicted.y - route.start.y
    along_track = rel_x * ux + rel_y * uy
    cross_track = ux * rel_y - uy * rel_x
    target_distance = math.hypot(
        route.target.x - predicted.x,
        route.target.y - predicted.y,
    )

    if max(abs(observed_cross_track), abs(cross_track)) > settings.max_cross_track_m:
        return _stop(
            "cross_track_limit_reached",
            predicted,
            prediction_horizon_s=horizon,
            target_distance_m=target_distance,
            along_track_m=along_track,
            cross_track_m=cross_track,
            observed_cross_track_m=observed_cross_track,
        )
    if target_distance <= settings.waypoint_tolerance_m:
        return _stop(
            "target_reached",
            predicted,
            prediction_horizon_s=horizon,
            target_distance_m=target_distance,
            along_track_m=along_track,
            cross_track_m=cross_track,
            observed_cross_track_m=observed_cross_track,
        )
    if along_track > route_length + settings.waypoint_tolerance_m:
        return _stop(
            "target_passed",
            predicted,
            prediction_horizon_s=horizon,
            target_distance_m=target_distance,
            along_track_m=along_track,
            cross_track_m=cross_track,
            observed_cross_track_m=observed_cross_track,
        )

    lookahead_along = min(
        max(along_track, 0.0) + settings.lookahead_m,
        route_length,
    )
    aim_point = Point(
        route.start.x + ux * lookahead_along,
        route.start.y + uy * lookahead_along,
    )
    aim_dx = aim_point.x - predicted.x
    aim_dy = aim_point.y - predicted.y
    if math.hypot(aim_dx, aim_dy) <= 1e-9:
        aim_dx = route.target.x - predicted.x
        aim_dy = route.target.y - predicted.y
    desired_course = math.degrees(math.atan2(aim_dy, aim_dx))
    heading_error = normalize_degrees(
        desired_course - observation.course_heading_degrees
    )
    # 🚨 **The 2026-08-24 gate: never steer on a heading no motion has
    # confirmed inside THIS window.** Compared against the caller's displacement
    # from the window's own first fix (`services.py` seeds `origin` there, not
    # from `route.start`), so a stale heading carried in from a previous session
    # cannot be laundered by a route that happens to start where the mower sits.
    # This deliberately does NOT gate any fail-closed check above -- a stop is
    # still a stop when the heading is unconfirmed.
    heading_confirmed = (
        observation.distance_travelled_m >= settings.min_travel_for_heading_trust_m
    )
    if not heading_confirmed or abs(heading_error) <= settings.heading_deadband_degrees:
        angular_speed = 0
    else:
        # `heading_error > 0` means the desired course sits COUNTER-CLOCKWISE of
        # the actual course, i.e. `course_heading_degrees` must RISE to null it.
        # Positive angular lowers that course, so nulling a positive error takes
        # a NEGATIVE command. See `_ANGULAR_COMMAND_SIGN_PER_COURSE_DEGREE`.
        angular_speed = round(
            _ANGULAR_COMMAND_SIGN_PER_COURSE_DEGREE
            * heading_error
            * settings.angular_speed_per_heading_degree
        )
        angular_speed = max(
            -settings.max_abs_angular_speed,
            min(settings.max_abs_angular_speed, angular_speed),
        )

    return ContinuousDecision(
        action="drive",
        reason="tracking_route",
        linear_speed=settings.linear_speed,
        angular_speed=angular_speed,
        predicted_position=predicted,
        aim_point=aim_point,
        target_distance_m=target_distance,
        along_track_m=along_track,
        cross_track_m=cross_track,
        observed_cross_track_m=observed_cross_track,
        desired_course_degrees=desired_course,
        heading_error_degrees=heading_error,
        heading_confirmed_by_motion=heading_confirmed,
        prediction_horizon_s=horizon,
    )
