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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

ControllerAction = Literal["drive", "stop"]
HeadingSource = Literal["position_chord"]

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
    # A position chord this short cannot support a useful map bearing. Unlike
    # the superseded 2026-08-24 gate, reaching this distance does not itself
    # confirm a separate `toward` value: the executor must derive the heading
    # from the chord and provide explicit `HeadingEvidence`.
    min_travel_for_heading_trust_m: float = 0.15
    # A missing opening chord or an old rolling chord stops the window.
    #
    # 🚨 **RAISED 2.0 -> 3.0 s on 2026-08-27, and it is a SAFETY TRADE, not a
    # tuning tweak** -- see `docs/phase2-acquisition-budget-decision-20260827.md`.
    # Acquiring heading needs a >= `min_travel_for_heading_trust_m` (0.15 m)
    # chord, which from a standstill takes about TWO position samples. The feed's
    # median interval is 1016 ms (measured over 117 intervals, 2026-08-27,
    # `docs/evidence-position-cadence-post-reconnect-20260827.json`), so a 2.0 s
    # budget bought about two samples and losing either one failed acquisition.
    # That is exactly what separated Phase 2 attempt 2 (acquired at 1.95 s) from
    # attempt 3 (timed out at 2.03 s) -- 57% of intervals exceed 1010 ms, so it
    # was close to a coin flip, and neither outcome was a fault.
    #
    # 3.0 s buys a third sample, so acquisition survives losing one interval.
    # ⚠️ The cost is real and deliberate: blind travel grows from ~0.51 m to
    # ~0.75 m and the required clear disk grows 26%, from 1.06 m to 1.34 m.
    # 🗑️ Do NOT instead lower `min_travel_for_heading_trust_m` -- it is the
    # registered informativeness floor at sigma = 0.0031 m position noise, and
    # lowering it to make a test pass is what the 2026-08-23 repair prevents.
    max_heading_acquisition_s: float = 3.0
    # Unchanged: this bounds how OLD a heading may be when steering on it, a
    # different question from how long acquiring one may take.
    max_heading_age_s: float = 2.0
    # Blind acquisition is admitted only when a complete worst-case disk fits
    # inside the frozen corridor. At the validated v1 command this is
    # 0.28 m/s * 3.0 s + the banked 0.50 m guard/stop overshoot = **1.34 m**
    # (it was 1.06 m while the budget was 2.0 s). The disk is COMPUTED from the
    # budget, so raising one raises the clearance a run must prove.
    max_safety_speed_mps: float = 0.28
    stop_overshoot_m: float = 0.50
    lookahead_m: float = 0.80
    waypoint_tolerance_m: float = 0.15
    max_cross_track_m: float = 0.30
    # The pre-registered Phase 2 criterion is 0.20 m; 0.30 m remains the hard
    # runtime abort. Admission uses whichever is smaller.
    max_admission_cross_track_m: float = 0.20
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
    # ⚠️ 2026-09-03: a DIRECT post-ramp measurement puts linear 400 at 0.295 m/s,
    # so this 0.2482 -- derived from 4 s windows -- is ~16% low as a SUSTAINED
    # speed. It is deliberately left alone: Phase 2 is parked (standing decision
    # 5), this controller is unreachable from any shipped path, and the banked
    # replays that validate it were scored against this value. If Phase 2 ever
    # resumes, re-derive this from post-ramp samples BEFORE trusting a
    # prediction, and re-run the replays rather than editing the number alone.
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
            "max_heading_acquisition_s": self.max_heading_acquisition_s,
            "max_heading_age_s": self.max_heading_age_s,
            "max_safety_speed_mps": self.max_safety_speed_mps,
            "stop_overshoot_m": self.stop_overshoot_m,
            "lookahead_m": self.lookahead_m,
            "waypoint_tolerance_m": self.waypoint_tolerance_m,
            "max_cross_track_m": self.max_cross_track_m,
            "max_admission_cross_track_m": self.max_admission_cross_track_m,
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
class HeadingEvidence:
    """A map-local course derived from fresh displacement in this window."""

    course_heading_degrees: float
    measured_at_s: float
    chord_m: float
    uncertainty_degrees: float
    source: HeadingSource = "position_chord"


@dataclass(frozen=True)
class ContinuousObservation:
    """One executor-independent snapshot presented to the pure controller."""

    position: Point
    # Retained as an additive evidence/report compatibility field only. The
    # controller never steers from this scalar; `heading_evidence` is the sole
    # control input. The service sets this from the same position chord and
    # never from `toward`.
    course_heading_degrees: float | None
    telemetry_age_s: float
    refresh_age_s: float
    elapsed_s: float
    distance_travelled_m: float
    heading_evidence: HeadingEvidence | None = None
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
    # controller is still acquiring an explicit position-chord heading, so no
    # heading error exists yet and angular speed must remain zero.
    # ⚠️ A future reader will otherwise see `angular_speed: 0` next to a large
    # `heading_error_degrees` in an evidence file and read it as a bug.
    heading_confirmed_by_motion: bool | None = None
    heading_source: HeadingSource | None = None
    heading_age_s: float | None = None
    heading_chord_m: float | None = None
    heading_uncertainty_degrees: float | None = None
    prediction_horizon_s: float = 0.0


@dataclass(frozen=True)
class AlignmentFeasibility:
    """Conservative post-acquisition admission against remaining budgets."""

    desired_course_degrees: float
    heading_error_degrees: float
    opening_cross_track_m: float
    current_cross_track_m: float
    blind_travel_m: float
    blind_time_s: float
    blind_cross_track_m: float
    turn_time_s: float
    turn_distance_m: float
    turn_cross_track_m: float
    predicted_end_cross_track_m: float
    max_abs_cross_track_m: float
    total_time_s: float
    total_distance_m: float
    total_cross_track_m: float
    window_budget_s: float
    distance_budget_m: float
    cross_track_budget_m: float
    remaining_window_s: float
    remaining_distance_m: float
    model_turn_rate_deg_per_s: float
    model_abs_angular_command: int
    model_assumption: str
    feasible: bool
    limiting_factor: str | None


@dataclass(frozen=True)
class BlindAcquisitionFeasibility:
    """Whether every possible straight acquisition ray fits the corridor."""

    required_radius_m: float
    boundary_clearance_m: float | None
    live_position_inside: bool
    feasible: bool


def course_from_position_chord(
    start: Point,
    end: Point,
    *,
    measured_at_s: float,
    min_chord_m: float = 0.15,
    position_sigma_m: float = 0.0031,
) -> HeadingEvidence | None:
    """Return fresh map-course evidence only when the chord is informative."""

    numeric = (
        *vars(start).values(),
        *vars(end).values(),
        measured_at_s,
        min_chord_m,
        position_sigma_m,
    )
    if (
        not all(math.isfinite(float(value)) for value in numeric)
        or measured_at_s < 0
        or min_chord_m <= 0
        or position_sigma_m < 0
    ):
        return None
    chord_m = math.hypot(end.x - start.x, end.y - start.y)
    if chord_m < min_chord_m:
        return None
    course = math.degrees(math.atan2(end.y - start.y, end.x - start.x))
    uncertainty = math.degrees(math.atan(position_sigma_m * math.sqrt(2.0) / chord_m))
    return HeadingEvidence(
        course_heading_degrees=course,
        measured_at_s=measured_at_s,
        chord_m=chord_m,
        uncertainty_degrees=uncertainty,
    )


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    squared_length = (end.x - start.x) ** 2 + (end.y - start.y) ** 2
    if squared_length <= 1e-18:
        return math.hypot(point.x - start.x, point.y - start.y) <= 1e-9
    cross = (point.y - start.y) * (end.x - start.x) - (point.x - start.x) * (
        end.y - start.y
    )
    if abs(cross) > 1e-9:
        return False
    dot = (point.x - start.x) * (end.x - start.x) + (point.y - start.y) * (
        end.y - start.y
    )
    return -1e-9 <= dot <= squared_length + 1e-9


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Return whether two closed line segments intersect."""

    def _orientation(p: Point, q: Point, r: Point) -> float:
        return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)

    orientations = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    if orientations[0] * orientations[1] < 0 and orientations[2] * orientations[3] < 0:
        return True
    return any(
        abs(orientation) <= 1e-9 and _point_on_segment(point, start, end)
        for orientation, point, start, end in (
            (orientations[0], c, a, b),
            (orientations[1], d, a, b),
            (orientations[2], a, c, d),
            (orientations[3], b, c, d),
        )
    )


def polygon_is_valid(polygon: Sequence[Point]) -> bool:
    """Return whether vertices form one finite, non-self-intersecting polygon."""

    if len(polygon) < 3 or not all(
        math.isfinite(value) for vertex in polygon for value in (vertex.x, vertex.y)
    ):
        return False
    edges = [
        (polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ]
    if any(math.hypot(b.x - a.x, b.y - a.y) <= 1e-9 for a, b in edges):
        return False
    signed_double_area = sum(a.x * b.y - b.x * a.y for a, b in edges)
    if abs(signed_double_area) <= 1e-12:
        return False
    for first_index, (a, b) in enumerate(edges):
        for second_index in range(first_index + 1, len(edges)):
            if second_index in {
                first_index,
                first_index + 1,
            } or (first_index == 0 and second_index == len(edges) - 1):
                continue
            c, d = edges[second_index]
            if _segments_intersect(a, b, c, d):
                return False
    return True


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return whether a point is inside or on a simple polygon boundary."""

    if not polygon_is_valid(polygon) or not all(
        math.isfinite(value) for value in (point.x, point.y)
    ):
        return False
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        crosses = (current.y > point.y) != (previous.y > point.y)
        if crosses:
            x_at_y = (previous.x - current.x) * (point.y - current.y) / (
                previous.y - current.y
            ) + current.x
            if point.x <= x_at_y:
                inside = not inside
        previous = current
    return inside


def polygon_boundary_clearance(point: Point, polygon: Sequence[Point]) -> float | None:
    """Return the shortest Euclidean distance from a point to polygon edges."""

    if not polygon_is_valid(polygon) or not all(
        math.isfinite(value) for value in (point.x, point.y)
    ):
        return None
    best = float("inf")
    previous = polygon[-1]
    for current in polygon:
        dx, dy = current.x - previous.x, current.y - previous.y
        length_squared = dx * dx + dy * dy
        projection = (
            0.0
            if length_squared <= 1e-18
            else max(
                0.0,
                min(
                    1.0,
                    ((point.x - previous.x) * dx + (point.y - previous.y) * dy)
                    / length_squared,
                ),
            )
        )
        closest = Point(previous.x + projection * dx, previous.y + projection * dy)
        best = min(best, math.hypot(point.x - closest.x, point.y - closest.y))
        previous = current
    return best


def blind_acquisition_feasibility(
    position: Point,
    corridor_polygon: Sequence[Point],
    config: ContinuousControllerConfig | None = None,
) -> BlindAcquisitionFeasibility:
    """Require the complete unknown-heading acquisition disk to be contained."""

    settings = config or ContinuousControllerConfig()
    required = (
        settings.max_safety_speed_mps * settings.max_heading_acquisition_s
        + settings.stop_overshoot_m
    )
    inside = point_in_polygon(position, corridor_polygon)
    clearance = polygon_boundary_clearance(position, corridor_polygon)
    return BlindAcquisitionFeasibility(
        required_radius_m=required,
        boundary_clearance_m=clearance,
        live_position_inside=inside,
        feasible=inside and clearance is not None and clearance >= required,
    )


def normalize_degrees(angle: float) -> float:
    """Normalize an angle to [-180, 180)."""

    return (angle + 180.0) % 360.0 - 180.0


def _route_geometry(route: ContinuousRoute) -> tuple[float, float, float, float]:
    dx, dy = route.target.x - route.start.x, route.target.y - route.start.y
    length = math.hypot(dx, dy)
    return dx, dy, length, math.degrees(math.atan2(dy, dx))


def _signed_cross_track(route: ContinuousRoute, position: Point) -> float:
    dx, dy, length, _ = _route_geometry(route)
    if length <= 1e-9:
        return float("nan")
    ux, uy = dx / length, dy / length
    rel_x, rel_y = position.x - route.start.x, position.y - route.start.y
    return ux * rel_y - uy * rel_x


def _desired_course_from_position(
    route: ContinuousRoute,
    position: Point,
    settings: ContinuousControllerConfig,
) -> float:
    dx, dy, length, _ = _route_geometry(route)
    ux, uy = dx / length, dy / length
    rel_x, rel_y = position.x - route.start.x, position.y - route.start.y
    along = rel_x * ux + rel_y * uy
    lookahead = min(max(along, 0.0) + settings.lookahead_m, length)
    aim = Point(route.start.x + ux * lookahead, route.start.y + uy * lookahead)
    return math.degrees(math.atan2(aim.y - position.y, aim.x - position.x))


def alignment_feasibility(
    route: ContinuousRoute,
    *,
    opening_position: Point,
    position: Point,
    heading_evidence: HeadingEvidence,
    elapsed_s: float,
    cumulative_distance_m: float,
    config: ContinuousControllerConfig | None = None,
) -> AlignmentFeasibility:
    """Estimate whether alignment fits the live remaining safety budgets.

    The opening blind phase is measured, not invented: its signed cross-track
    and travel come from the fresh pre-dispatch origin and the qualifying chord.
    The turn term retains the measured 8 deg/s model only for refusal/admission
    estimation at the validated +/-180 envelope. Passing is necessary, never a
    guarantee; runtime corridor and 0.30 m hard aborts remain authoritative.
    """

    settings = config or ContinuousControllerConfig()
    model_assumption = (
        f"{settings.min_turn_rate_deg_per_s:g} deg/s refusal/admission estimate "
        f"only at the validated +/-{settings.max_abs_angular_speed} command "
        "envelope; passing does not prove successful heading nulling"
    )
    numeric = (
        route.start.x,
        route.start.y,
        route.target.x,
        route.target.y,
        opening_position.x,
        opening_position.y,
        position.x,
        position.y,
        heading_evidence.course_heading_degrees,
        heading_evidence.measured_at_s,
        heading_evidence.chord_m,
        heading_evidence.uncertainty_degrees,
        elapsed_s,
        cumulative_distance_m,
    )
    _, _, route_length, route_course = _route_geometry(route)
    unusable = (
        not all(math.isfinite(float(value)) for value in numeric)
        or route_length <= 1e-9
        or elapsed_s < 0
        or cumulative_distance_m < 0
        or heading_evidence.source != "position_chord"
        or heading_evidence.measured_at_s < 0
        or heading_evidence.measured_at_s > elapsed_s
        or heading_evidence.chord_m < settings.min_travel_for_heading_trust_m
        or heading_evidence.uncertainty_degrees < 0
    )
    if unusable:
        infinite = float("inf")
        return AlignmentFeasibility(
            desired_course_degrees=float("nan"),
            heading_error_degrees=float("nan"),
            opening_cross_track_m=infinite,
            current_cross_track_m=infinite,
            blind_travel_m=infinite,
            blind_time_s=infinite,
            blind_cross_track_m=infinite,
            turn_time_s=infinite,
            turn_distance_m=infinite,
            turn_cross_track_m=infinite,
            predicted_end_cross_track_m=infinite,
            max_abs_cross_track_m=infinite,
            total_time_s=infinite,
            total_distance_m=infinite,
            total_cross_track_m=infinite,
            window_budget_s=settings.max_window_s,
            distance_budget_m=settings.max_distance_m,
            cross_track_budget_m=min(
                settings.max_cross_track_m, settings.max_admission_cross_track_m
            ),
            remaining_window_s=0.0,
            remaining_distance_m=0.0,
            model_turn_rate_deg_per_s=settings.min_turn_rate_deg_per_s,
            model_abs_angular_command=settings.max_abs_angular_speed,
            model_assumption=model_assumption,
            feasible=False,
            limiting_factor="route_or_heading_unusable",
        )

    opening_cross = _signed_cross_track(route, opening_position)
    current_cross = _signed_cross_track(route, position)
    desired_course = _desired_course_from_position(route, position, settings)
    course = heading_evidence.course_heading_degrees
    heading_error = normalize_degrees(desired_course - course)
    blind_cross = current_cross - opening_cross

    turn_time = abs(heading_error) / settings.min_turn_rate_deg_per_s
    turn_distance = settings.nominal_speed_mps * turn_time
    route_relative_actual = math.radians(normalize_degrees(course - route_course))
    route_relative_desired = math.radians(
        normalize_degrees(desired_course - route_course)
    )
    direction = 0.0 if heading_error == 0 else math.copysign(1.0, heading_error)
    radius = settings.nominal_speed_mps / math.radians(settings.min_turn_rate_deg_per_s)
    turn_cross = (
        direction
        * radius
        * (math.cos(route_relative_actual) - math.cos(route_relative_desired))
    )
    end_cross = current_cross + turn_cross

    cross_candidates = [opening_cross, current_cross, end_cross]
    # The signed arc can have an interior cross-track extremum whenever it
    # crosses either the route direction (sin(theta)=0 at 0 deg) or its
    # antipode (sin(theta)=0 at 180 deg). Include both, rather than checking
    # only the end point or assuming every admitted turn is monotonic.
    for critical_course, critical_cosine in (
        (route_course, 1.0),
        (route_course + 180.0, -1.0),
    ):
        crossing = normalize_degrees(critical_course - course)
        lies_inside_turn = (heading_error > 0 and 0 < crossing < heading_error) or (
            heading_error < 0 and heading_error < crossing < 0
        )
        if lies_inside_turn:
            crossing_delta = (
                direction * radius * (math.cos(route_relative_actual) - critical_cosine)
            )
            cross_candidates.append(current_cross + crossing_delta)

    admission_cross = min(
        settings.max_cross_track_m, settings.max_admission_cross_track_m
    )
    max_abs_cross = max(abs(value) for value in cross_candidates)
    total_time = elapsed_s + turn_time
    total_distance = cumulative_distance_m + turn_distance
    remaining_window = max(settings.max_window_s - elapsed_s, 0.0)
    remaining_distance = max(settings.max_distance_m - cumulative_distance_m, 0.0)
    breaches = (
        (turn_time > remaining_window, "window_s"),
        (turn_distance > remaining_distance, "distance_m"),
        (max_abs_cross > admission_cross, "cross_track_m"),
    )
    limiting = next((name for failed, name in breaches if failed), None)
    return AlignmentFeasibility(
        desired_course_degrees=desired_course,
        heading_error_degrees=heading_error,
        opening_cross_track_m=opening_cross,
        current_cross_track_m=current_cross,
        blind_travel_m=cumulative_distance_m,
        blind_time_s=elapsed_s,
        blind_cross_track_m=blind_cross,
        turn_time_s=turn_time,
        turn_distance_m=turn_distance,
        turn_cross_track_m=turn_cross,
        predicted_end_cross_track_m=end_cross,
        max_abs_cross_track_m=max_abs_cross,
        total_time_s=total_time,
        total_distance_m=total_distance,
        total_cross_track_m=max_abs_cross,
        window_budget_s=settings.max_window_s,
        distance_budget_m=settings.max_distance_m,
        cross_track_budget_m=admission_cross,
        remaining_window_s=remaining_window,
        remaining_distance_m=remaining_distance,
        model_turn_rate_deg_per_s=settings.min_turn_rate_deg_per_s,
        model_abs_angular_command=settings.max_abs_angular_speed,
        model_assumption=model_assumption,
        feasible=limiting is None,
        limiting_factor=limiting,
    )


def _predict_position(
    observation: ContinuousObservation,
    config: ContinuousControllerConfig,
) -> tuple[Point, float]:
    """Project a stale fix only along an explicitly evidenced course."""

    evidence = observation.heading_evidence
    if evidence is None:
        return observation.position, 0.0

    horizon = min(
        max(float(observation.telemetry_age_s), 0.0),
        config.max_prediction_horizon_s,
    )
    radians = math.radians(evidence.course_heading_degrees)
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
    heading_evidence: HeadingEvidence | None = None,
    heading_age_s: float | None = None,
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
        heading_source=(heading_evidence.source if heading_evidence else None),
        heading_age_s=heading_age_s,
        heading_chord_m=(heading_evidence.chord_m if heading_evidence else None),
        heading_uncertainty_degrees=(
            heading_evidence.uncertainty_degrees if heading_evidence else None
        ),
        prediction_horizon_s=prediction_horizon_s,
    )


def _input_failure_reason(
    route: ContinuousRoute,
    observation: ContinuousObservation,
    settings: ContinuousControllerConfig,
) -> str | None:
    """Return the first fail-closed input reason, or None when inputs are safe."""

    numeric_inputs: tuple[float | int, ...] = (
        route.start.x,
        route.start.y,
        route.target.x,
        route.target.y,
        observation.position.x,
        observation.position.y,
        observation.telemetry_age_s,
        observation.refresh_age_s,
        observation.refresh_max_gap_since_last_decision_s,
        observation.elapsed_s,
        observation.distance_travelled_m,
    )
    if observation.course_heading_degrees is not None:
        numeric_inputs += (observation.course_heading_degrees,)
    if observation.heading_evidence is not None:
        numeric_inputs += (
            observation.heading_evidence.course_heading_degrees,
            observation.heading_evidence.measured_at_s,
            observation.heading_evidence.chord_m,
            observation.heading_evidence.uncertainty_degrees,
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
    evidence = observation.heading_evidence
    if evidence is not None and (
        evidence.source != "position_chord"
        or evidence.measured_at_s < 0
        or evidence.chord_m < settings.min_travel_for_heading_trust_m
        or evidence.uncertainty_degrees < 0
        or evidence.measured_at_s > observation.elapsed_s
    ):
        return "heading_evidence_invalid"
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


def continuous_control_decision(  # noqa: C901
    route: ContinuousRoute,
    observation: ContinuousObservation,
    config: ContinuousControllerConfig | None = None,
) -> ContinuousDecision:
    """Return one bounded steering decision without sending anything."""

    settings = config or ContinuousControllerConfig()
    input_failure = _input_failure_reason(route, observation, settings)
    if input_failure in {
        "inputs_not_finite",
        "observation_values_invalid",
        "heading_evidence_invalid",
    }:
        return _stop(
            input_failure,
            observation.position,
            prediction_horizon_s=0.0,
        )

    predicted, horizon = _predict_position(observation, settings)
    if input_failure is not None:
        return _stop(input_failure, predicted, prediction_horizon_s=horizon)

    evidence = observation.heading_evidence
    heading_age_s = (
        observation.elapsed_s - evidence.measured_at_s if evidence is not None else None
    )
    if evidence is None and observation.elapsed_s >= settings.max_heading_acquisition_s:
        return _stop(
            "heading_acquisition_timeout",
            predicted,
            prediction_horizon_s=horizon,
        )
    if heading_age_s is not None and heading_age_s > settings.max_heading_age_s:
        return _stop(
            "heading_evidence_stale",
            predicted,
            prediction_horizon_s=horizon,
            heading_evidence=evidence,
            heading_age_s=heading_age_s,
        )

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
    if evidence is None:
        return ContinuousDecision(
            action="drive",
            reason="acquiring_heading",
            linear_speed=settings.linear_speed,
            angular_speed=0,
            predicted_position=predicted,
            aim_point=aim_point,
            target_distance_m=target_distance,
            along_track_m=along_track,
            cross_track_m=cross_track,
            observed_cross_track_m=observed_cross_track,
            desired_course_degrees=desired_course,
            heading_error_degrees=None,
            heading_confirmed_by_motion=False,
            prediction_horizon_s=horizon,
        )

    heading_error = normalize_degrees(desired_course - evidence.course_heading_degrees)
    if abs(heading_error) <= settings.heading_deadband_degrees:
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
        heading_confirmed_by_motion=True,
        heading_source=evidence.source,
        heading_age_s=heading_age_s,
        heading_chord_m=evidence.chord_m,
        heading_uncertainty_degrees=evidence.uncertainty_degrees,
        prediction_horizon_s=horizon,
    )
