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
    angular_speed_per_heading_degree: float = 12.0
    heading_deadband_degrees: float = 1.5
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
    max_prediction_horizon_s: float = 1.25

    def __post_init__(self) -> None:
        """Reject nonsensical bounds before replay begins."""

        positive = {
            "linear_speed": self.linear_speed,
            "max_abs_angular_speed": self.max_abs_angular_speed,
            "angular_speed_per_heading_degree": (self.angular_speed_per_heading_degree),
            "lookahead_m": self.lookahead_m,
            "waypoint_tolerance_m": self.waypoint_tolerance_m,
            "max_cross_track_m": self.max_cross_track_m,
            "max_telemetry_age_s": self.max_telemetry_age_s,
            "max_refresh_age_s": self.max_refresh_age_s,
            "max_refresh_gap_s": self.max_refresh_gap_s,
            "max_window_s": self.max_window_s,
            "max_distance_m": self.max_distance_m,
            "nominal_speed_mps": self.nominal_speed_mps,
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
    prediction_horizon_s: float = 0.0


def normalize_degrees(angle: float) -> float:
    """Normalize an angle to [-180, 180)."""

    return (angle + 180.0) % 360.0 - 180.0


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
    if abs(heading_error) <= settings.heading_deadband_degrees:
        angular_speed = 0
    else:
        angular_speed = round(heading_error * settings.angular_speed_per_heading_degree)
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
        prediction_horizon_s=horizon,
    )
