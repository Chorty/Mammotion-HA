"""Offline tests for the non-dispatching continuous controller prototype."""

from __future__ import annotations

import ast
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from custom_components.mammotion.continuous_controller import (
    ContinuousControllerConfig,
    ContinuousObservation,
    ContinuousRoute,
    HeadingEvidence,
    Point,
    alignment_feasibility,
    blind_acquisition_feasibility,
    continuous_control_decision,
    course_from_position_chord,
    normalize_degrees,
)


def _route(*, contained: bool = True) -> ContinuousRoute:
    return ContinuousRoute(
        start=Point(0.0, 0.0),
        target=Point(3.0, 0.0),
        contained=contained,
    )


def _observation(**overrides: object) -> ContinuousObservation:
    defaults: dict[str, object] = {
        "position": Point(0.0, 0.0),
        "course_heading_degrees": 0.0,
        "telemetry_age_s": 0.0,
        "refresh_age_s": 0.0,
        "elapsed_s": 0.0,
        "distance_travelled_m": 0.0,
    }
    data = defaults | overrides
    if "heading_evidence" not in overrides:
        course = data["course_heading_degrees"]
        elapsed = float(data["elapsed_s"])
        data["heading_evidence"] = HeadingEvidence(
            course_heading_degrees=float(course),
            measured_at_s=elapsed,
            chord_m=0.20,
            uncertainty_degrees=1.0,
        )
    return ContinuousObservation(**data)  # type: ignore[arg-type]


def test_healthy_aligned_observation_requests_continuous_drive() -> None:
    """An aligned, healthy route asks to hold forward with no steering.

    `distance_travelled_m` is set past `min_travel_for_heading_trust_m` so this
    exercises the DEADBAND, not the 2026-08-24 unconfirmed-heading gate -- with
    the default 0.0 both suppress steering and the test could not tell which.
    """

    decision = continuous_control_decision(
        _route(), _observation(distance_travelled_m=0.20)
    )

    assert decision.action == "drive"
    assert decision.reason == "tracking_route"
    assert decision.linear_speed == 400
    assert decision.angular_speed == 0
    assert decision.heading_confirmed_by_motion is True
    assert decision.aim_point == Point(0.8, 0.0)


def test_heading_error_steers_in_the_correct_direction_and_clamps() -> None:
    """Heading corrections oppose the error's sign and never exceed the bound.

    🚨 **The sign flipped on 2026-08-24 and that is the fix, not a regression.**
    A POSITIVE `heading_error_degrees` means the desired course sits
    counter-clockwise of the actual course, so `course_heading_degrees` must
    RISE to null it -- and a positive commanded `angular_speed` LOWERS that
    course on this hardware. Nulling a positive error therefore takes a NEGATIVE
    command. Until this date the module asserted the opposite, which is what
    made the first physical Phase 2 run positive feedback. The measurement this
    encodes lives in `_ANGULAR_COMMAND_SIGN_PER_COURSE_DEGREE`.
    """

    error_positive = continuous_control_decision(
        _route(),
        _observation(course_heading_degrees=-30.0, distance_travelled_m=0.20),
    )
    error_negative = continuous_control_decision(
        _route(),
        _observation(course_heading_degrees=30.0, distance_travelled_m=0.20),
    )

    assert error_positive.heading_error_degrees is not None
    assert error_negative.heading_error_degrees is not None
    assert error_positive.heading_error_degrees > 0
    assert error_negative.heading_error_degrees < 0
    # Opposed, and clamped to the configured bound rather than 30 * 12 = 360.
    assert error_positive.angular_speed == -180
    assert error_negative.angular_speed == 180


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"cancelled": True}, "operator_cancelled"),
        ({"stop_available": False}, "stop_primitive_unavailable"),
        ({"ble_live": False}, "ble_link_not_live"),
        ({"refresh_healthy": False}, "refresh_cadence_unhealthy"),
        ({"refresh_age_s": 1.21}, "refresh_age_exceeded"),
        (
            {"refresh_max_gap_since_last_decision_s": 0.61},
            "refresh_cadence_stalled",
        ),
        ({"telemetry_age_s": 2.01}, "telemetry_stale"),
        ({"position_valid": False}, "position_invalid"),
        ({"inside_area": False}, "position_outside_area"),
        ({"rtk_fixed": False}, "rtk_not_fixed"),
        ({"blades_off": False}, "blades_not_off"),
        ({"work_mode_safe": False}, "work_mode_not_safe"),
        ({"elapsed_s": 4.0}, "window_limit_reached"),
        ({"distance_travelled_m": 1.5}, "distance_limit_reached"),
    ],
)
def test_every_runtime_fault_returns_zero_speed_stop(
    overrides: dict[str, object], reason: str
) -> None:
    """Every runtime fault must fail closed with a literal zero command."""

    decision = continuous_control_decision(_route(), _observation(**overrides))

    assert decision.action == "stop"
    assert decision.reason == reason
    assert decision.linear_speed == 0
    assert decision.angular_speed == 0


def test_uncontained_route_fails_closed_before_geometry() -> None:
    """A caller cannot use steering to bypass route containment."""

    decision = continuous_control_decision(_route(contained=False), _observation())

    assert decision.action == "stop"
    assert decision.reason == "route_not_prevalidated_contained"


def test_zero_length_route_fails_closed() -> None:
    """A degenerate target cannot reach heading or command calculations."""

    route = ContinuousRoute(start=Point(1.0, 1.0), target=Point(1.0, 1.0))
    decision = continuous_control_decision(
        route, _observation(position=Point(1.0, 1.0))
    )

    assert decision.action == "stop"
    assert decision.reason == "route_has_zero_length"


def test_cross_track_limit_stops_instead_of_steering_back_from_anywhere() -> None:
    """A corridor breach stops instead of attempting an unbounded recovery."""

    decision = continuous_control_decision(
        _route(), _observation(position=Point(0.5, 0.31))
    )

    assert decision.action == "stop"
    assert decision.reason == "cross_track_limit_reached"
    assert decision.cross_track_m == pytest.approx(0.31)
    assert decision.observed_cross_track_m == pytest.approx(0.31)


def test_prediction_cannot_hide_an_observed_corridor_breach() -> None:
    """A projected return toward the line cannot erase an unsafe measured fix."""

    decision = continuous_control_decision(
        _route(),
        _observation(
            position=Point(0.5, 0.31),
            course_heading_degrees=-90.0,
            telemetry_age_s=1.0,
        ),
    )

    assert decision.action == "stop"
    assert decision.reason == "cross_track_limit_reached"
    assert decision.observed_cross_track_m == pytest.approx(0.31)
    # 0.31 - nominal_speed_mps * 1.0 s, reconciled 2026-08-23 to the measured
    # k_lin-derived speed (0.2482 m/s at linear 400), was 0.03 at the old 0.28.
    assert decision.cross_track_m == pytest.approx(0.0618)


def test_target_tolerance_stops_without_issuing_a_final_drive() -> None:
    """Arrival returns stop immediately rather than another forward command."""

    decision = continuous_control_decision(
        _route(), _observation(position=Point(2.87, 0.0))
    )

    assert decision.action == "stop"
    assert decision.reason == "target_reached"
    assert decision.target_distance_m == pytest.approx(0.13)


def test_target_passed_stops_instead_of_turning_back() -> None:
    """Continuous mode never attempts reverse recovery after an overshoot."""

    decision = continuous_control_decision(
        _route(), _observation(position=Point(3.2, 0.0))
    )

    assert decision.action == "stop"
    assert decision.reason == "target_passed"


def test_bounded_prediction_projects_a_stale_fix_along_course() -> None:
    """A recent stale fix is projected using the measured nominal speed."""

    decision = continuous_control_decision(
        _route(),
        _observation(
            position=Point(1.0, 0.0),
            course_heading_degrees=0.0,
            telemetry_age_s=1.0,
        ),
    )

    assert decision.prediction_horizon_s == 1.0
    # 1.0 + nominal_speed_mps * 1.0 s, reconciled 2026-08-23 (was 1.28 at 0.28).
    assert decision.predicted_position == Point(1.2482, 0.0)


def test_prediction_horizon_never_extends_past_its_bound() -> None:
    """Prediction remains bounded even when telemetry is older but still valid."""

    config = replace(
        ContinuousControllerConfig(),
        max_telemetry_age_s=3.0,
        max_prediction_horizon_s=1.25,
    )
    decision = continuous_control_decision(
        _route(),
        _observation(telemetry_age_s=2.0),
        config,
    )

    assert decision.action == "drive"
    assert decision.prediction_horizon_s == 1.25
    # nominal_speed_mps * 1.25 s, reconciled 2026-08-23 (was 0.35 at 0.28).
    assert decision.predicted_position.x == pytest.approx(0.31025)


def test_lookahead_corrects_cross_track_before_the_endpoint() -> None:
    """Lookahead requests an early correction toward the route centerline."""

    decision = continuous_control_decision(
        _route(),
        _observation(
            position=Point(0.5, 0.10),
            course_heading_degrees=0.0,
            # Past `min_travel_for_heading_trust_m`: this test is about the
            # lookahead geometry, not the 2026-08-24 gate.
            distance_travelled_m=0.51,
        ),
    )

    assert decision.action == "drive"
    # Sitting +0.10 m left of a due-east route, the aim point is BELOW the
    # mower, so the desired course is negative and the error is negative --
    # which takes a POSITIVE command to null (sign corrected 2026-08-24).
    assert decision.heading_error_degrees is not None
    assert decision.heading_error_degrees < 0
    assert decision.angular_speed > 0
    assert decision.aim_point == Point(1.3, 0.0)


def test_invalid_configuration_is_rejected_offline() -> None:
    """Zero or negative safety bounds cannot enter a replay."""

    with pytest.raises(ValueError, match="controller bounds must be positive"):
        ContinuousControllerConfig(max_window_s=0.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"position": Point(float("nan"), 0.0)},
        {"course_heading_degrees": float("inf")},
        {"telemetry_age_s": float("nan")},
    ],
)
def test_non_finite_observations_fail_closed(overrides: dict[str, object]) -> None:
    """NaN and infinity cannot bypass comparisons or reach command rounding."""

    decision = continuous_control_decision(_route(), _observation(**overrides))

    assert decision.action == "stop"
    assert decision.reason == "inputs_not_finite"
    assert decision.linear_speed == 0
    assert decision.angular_speed == 0


def test_negative_observation_ages_fail_closed() -> None:
    """Invalid negative clock values cannot extend a controller window."""

    decision = continuous_control_decision(_route(), _observation(elapsed_s=-0.1))

    assert decision.action == "stop"
    assert decision.reason == "observation_values_invalid"


def test_command_bounds_cannot_exceed_the_mower_schema() -> None:
    """Offline configuration cannot request speeds outside the service schema."""

    with pytest.raises(ValueError, match="mower schema bounds"):
        ContinuousControllerConfig(linear_speed=1001)


def test_controller_module_imports_only_standard_library_calculation_tools() -> None:
    """The pure decision module cannot quietly acquire a dispatch dependency."""

    source = (
        Path(__file__).resolve().parents[3]
        / "custom_components"
        / "mammotion"
        / "continuous_controller.py"
    ).read_text()
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    assert imports <= {"__future__", "collections", "dataclasses", "math", "typing"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0.0, 0.0), (180.0, -180.0), (181.0, -179.0), (-181.0, 179.0)],
)
def test_heading_normalization(raw: float, expected: float) -> None:
    """Heading differences always take the signed shortest rotation."""

    assert normalize_degrees(raw) == expected


def test_refresh_age_alone_misses_a_stall_that_resolves_before_the_next_arrival() -> (
    None
):
    """The bug this field exists to fix, reproduced directly.

    Found 2026-08-23 replaying this module against a REAL capture
    (`docs/evidence-8s-continuous-window-20260822T233000Z.json`,
    `scripts/replay_continuous_controller_against_capture.py`): an 810 ms
    refresh stall -- the source of the largest prediction error in the whole
    corpus, 0.1418 m -- produced `refresh_age_s ~= 0` at the very next decision,
    because a fast recovery write happened to complete essentially
    simultaneously with that decision. A point sample of "time since the most
    recent completion" cannot see a stall that already resolved.
    """
    # refresh_age_s reads healthy: the most recent write completed moments ago.
    # But a stall happened WITHIN this window and has already recovered.
    decision = continuous_control_decision(
        _route(),
        _observation(
            refresh_age_s=0.0,
            refresh_max_gap_since_last_decision_s=0.81,
        ),
    )

    assert decision.action == "stop"
    assert decision.reason == "refresh_cadence_stalled"


def test_a_recovered_stall_inside_the_registered_tolerance_does_not_stop() -> None:
    """A gap under the registered 3R = 600 ms rule is not a stall."""
    decision = continuous_control_decision(
        _route(),
        _observation(
            refresh_age_s=0.0,
            refresh_max_gap_since_last_decision_s=0.59,
            distance_travelled_m=0.20,
        ),
    )

    assert decision.action == "drive"


# --- the 2026-08-24 unconfirmed-heading gate ----------------------------------
#
# Every observation below is copied from the FIRST physical Phase 2 run,
# `docs/evidence-phase2-first-physical-run-20260824.json`. The route, the
# reported course, and the heading error are that run's literal recorded
# numbers, not a constructed scenario.

_RUN_20260824_ROUTE = ContinuousRoute(
    start=Point(5.0628, -4.2926),
    target=Point(5.5602, -4.6281),
    contained=True,
)
# `course_heading_degrees` at decision 0, i.e. `90.13 - toward`.
_RUN_20260824_OPENING_COURSE = 279.3608


def test_the_20260824_opening_decision_no_longer_saturates_a_standing_mower() -> None:
    """The exact decision that lost the first physical Phase 2 run.

    Decision 0 fired at `elapsed_s: 0.458` with `distance_travelled_m: 0.0` --
    the mower had not moved inside this window at all -- computed a 46.639 deg
    heading error from a `toward` that no motion in this window had produced,
    and commanded `angular_speed: 180`, the saturated maximum. Cross-track then
    diverged 0 -> -0.016 -> -0.187 -> -0.389 m and the 0.30 m abort fired.
    """

    decision = continuous_control_decision(
        _RUN_20260824_ROUTE,
        _observation(
            position=Point(5.0628, -4.2926),
            course_heading_degrees=_RUN_20260824_OPENING_COURSE,
            distance_travelled_m=0.0,
            heading_evidence=None,
        ),
    )

    assert decision.action == "drive"
    assert decision.reason == "acquiring_heading"
    assert decision.linear_speed == 400
    # The whole fix: straight, not the saturated 180 that actually went out.
    assert decision.angular_speed == 0
    assert decision.heading_confirmed_by_motion is False
    # The stale scalar is diagnostic only and cannot become a control error.
    assert decision.heading_error_degrees is None


def test_correction_resumes_once_real_displacement_is_observed() -> None:
    """Past the trust floor the identical geometry steers exactly as before."""

    decision = continuous_control_decision(
        _RUN_20260824_ROUTE,
        _observation(
            position=Point(5.0628, -4.2926),
            course_heading_degrees=_RUN_20260824_OPENING_COURSE,
            distance_travelled_m=0.15,
        ),
    )

    assert decision.action == "drive"
    assert decision.heading_confirmed_by_motion is True
    # -180, not the +180 that actually went out: the error is positive, so the
    # course must RISE, so the command must be negative (fixed 2026-08-24).
    assert decision.angular_speed == -180
    assert decision.heading_error_degrees == pytest.approx(46.639242, abs=1e-5)


def test_distance_alone_never_confirms_a_heading() -> None:
    """Even kilometres of travel cannot launder a stale scalar course."""

    decision = continuous_control_decision(
        _RUN_20260824_ROUTE,
        _observation(
            position=Point(5.0628, -4.2926),
            course_heading_degrees=_RUN_20260824_OPENING_COURSE,
            distance_travelled_m=10.0,
            heading_evidence=None,
        ),
        ContinuousControllerConfig(max_distance_m=20.0),
    )

    assert decision.reason == "acquiring_heading"
    assert decision.angular_speed == 0
    assert decision.heading_confirmed_by_motion is False


def test_missing_heading_evidence_times_out_after_two_seconds() -> None:
    """Acquisition cannot continue past its shared two-second bound."""
    decision = continuous_control_decision(
        _route(),
        _observation(
            course_heading_degrees=123.0,
            elapsed_s=2.0,
            heading_evidence=None,
        ),
    )

    assert decision.action == "stop"
    assert decision.reason == "heading_acquisition_timeout"
    assert decision.angular_speed == 0


def test_stale_heading_evidence_is_never_reused() -> None:
    """A rolling course older than two seconds requests a stop."""
    decision = continuous_control_decision(
        _route(),
        _observation(
            elapsed_s=3.0,
            heading_evidence=HeadingEvidence(
                course_heading_degrees=0.0,
                measured_at_s=0.9,
                chord_m=0.15,
                uncertainty_degrees=1.7,
            ),
        ),
    )

    assert decision.action == "stop"
    assert decision.reason == "heading_evidence_stale"
    assert decision.heading_age_s == pytest.approx(2.1)


def test_an_unconfirmed_heading_never_suppresses_a_stop() -> None:
    """The gate withholds a CORRECTION only; every abort still fires.

    Uses that run's final recorded position, which breached the 0.30 m
    cross-track bound at -0.3887 m, but with `distance_travelled_m: 0.0` so the
    heading is unconfirmed at the same moment.
    """

    decision = continuous_control_decision(
        _RUN_20260824_ROUTE,
        _observation(
            position=Point(5.1275, -4.8051),
            course_heading_degrees=_RUN_20260824_OPENING_COURSE,
            distance_travelled_m=0.0,
            heading_evidence=None,
        ),
    )

    assert decision.action == "stop"
    assert decision.reason == "cross_track_limit_reached"
    assert decision.linear_speed == 0
    assert decision.angular_speed == 0
    assert decision.observed_cross_track_m == pytest.approx(-0.388702, abs=1e-5)
    # Nothing was steered, so the flag is not applicable -- the same convention
    # `aim_point` and `heading_error_degrees` already use on a stop path.
    assert decision.heading_confirmed_by_motion is None


def test_the_heading_trust_floor_cannot_be_configured_away() -> None:
    """A 0.0 floor would silently restore the 2026-08-24 behaviour."""

    with pytest.raises(ValueError, match="controller bounds must be positive"):
        ContinuousControllerConfig(min_travel_for_heading_trust_m=0.0)


# --- the 2026-08-24 STEERING SIGN --------------------------------------------
#
# 🚨 These tests encode a MEASUREMENT, not the code's own behaviour. If a change
# makes them fail, the change inverted the steering law -- do not "fix" them by
# flipping the expected sign.
#
# The measurement, re-derived offline from banked captures on both command
# signs (mirror-derived course AND an independent `atan2(dy, dx)` travel bearing
# that never reads `toward` at all):
#
#   capture                              linear angular d(toward) d(course)
#   Phase 2 first physical run 20260824    400    +180    +14.42    -14.42
#   Phase 1b certified arc     20260823    400    +180    +64.18    -64.18
#   arc120 out-of-sample       20260823    400    +120    +47.06    -47.06
#   arc sweep a300             20260812    400    +300    +43.42    -43.42
#   arc sweep a500             20260812    400    +500    +66.85    -66.85
#   night pivot (reverse sign) 20260812      0    -500    -61.43    +61.43
#
# i.e. POSITIVE `angular_speed` LOWERS `course_heading_degrees`, because
# `angular > 0` raises `toward` and `course = 90.13 - toward` is a REFLECTION.

# Measured map-frame course rate at angular 180 with linear 400: the Phase 1b
# certified arc, excluding its spin-up step (8.964 deg/s; including spin-up it
# is 8.091). Used ONLY to integrate a predicted next course inside these tests.
_MEASURED_COURSE_RATE_DEG_PER_S = 8.964


def _course_after(course_degrees: float, angular_speed: int, seconds: float) -> float:
    """Rotate a map course the way the MOWER does, per the table above.

    Positive `angular_speed` DECREASES `course_heading_degrees`.
    """

    if angular_speed == 0:
        return course_degrees
    direction = -1.0 if angular_speed > 0 else 1.0
    return course_degrees + direction * _MEASURED_COURSE_RATE_DEG_PER_S * seconds


def test_positive_angular_lowers_map_course_so_the_gain_is_applied_negatively() -> None:
    """The named pin referenced by `_ANGULAR_COMMAND_SIGN_PER_COURSE_DEGREE`.

    A positive heading error needs the course to RISE; the course rises only
    under a NEGATIVE command; therefore a positive error must produce a negative
    `angular_speed`. Asserted on an unsaturated error so the sign is carried by
    the law and not by a clamp.
    """

    # 5 deg of error * gain 12 = 60, well inside the +-180 clamp.
    decision = continuous_control_decision(
        _route(),
        _observation(course_heading_degrees=-5.0, distance_travelled_m=0.20),
    )

    assert decision.heading_error_degrees == pytest.approx(5.0)
    assert decision.angular_speed == -60
    # And the measured law then moves the course the way the error needs.
    assert _course_after(-5.0, decision.angular_speed, 1.0) > -5.0


@pytest.mark.parametrize("course_degrees", [-30.0, -12.0, -5.0, 5.0, 12.0, 30.0])
def test_one_correction_step_shrinks_the_error_under_the_measured_law(
    course_degrees: float,
) -> None:
    """Closed loop, not just a sign: integrate the command and re-measure.

    This is the property the 2026-08-24 run violated. It is asserted against the
    measured rotation law, so it fails if either the command sign or the error
    convention is inverted.
    """

    before = continuous_control_decision(
        _route(),
        _observation(course_heading_degrees=course_degrees, distance_travelled_m=0.20),
    )
    assert before.heading_error_degrees is not None
    assert before.desired_course_degrees is not None

    stepped = _course_after(course_degrees, before.angular_speed, 1.0)
    error_after = normalize_degrees(before.desired_course_degrees - stepped)

    assert abs(error_after) < abs(before.heading_error_degrees)


@pytest.mark.parametrize("course_degrees", [-30.0, -12.0, -5.0, 5.0, 12.0, 30.0])
def test_the_law_that_actually_shipped_would_have_grown_the_error(
    course_degrees: float,
) -> None:
    """The counterfactual, so the bug is documented and not just deleted.

    Feeding the SAME geometry through the pre-fix arithmetic (`+gain * error`)
    and integrating with the same measured rotation law grows the error every
    time. That is positive feedback, and it is exactly what the first physical
    Phase 2 run did.
    """

    decision = continuous_control_decision(
        _route(),
        _observation(course_heading_degrees=course_degrees, distance_travelled_m=0.20),
    )
    assert decision.heading_error_degrees is not None
    assert decision.desired_course_degrees is not None

    inverted_command = -decision.angular_speed  # what the module used to emit
    stepped = _course_after(course_degrees, inverted_command, 1.0)
    error_after = normalize_degrees(decision.desired_course_degrees - stepped)

    assert abs(error_after) > abs(decision.heading_error_degrees)


# The three distinct position fixes of the 2026-08-24 run, with the course each
# reported and the heading error the executor recorded at that instant. Copied
# from `docs/evidence-phase2-first-physical-run-20260824.json`; the errors are
# the literal `decisions[*].decision.heading_error_degrees` values.
_RUN_20260824_DIVERGENCE = [
    (Point(5.0628, -4.2926), 279.3608, 46.639242),
    (Point(5.0702, -4.3173), 279.3608, 48.252827),
    (Point(5.1138, -4.5522), 272.9549, 77.395558),
]


@pytest.mark.parametrize(
    ("position", "course_degrees", "recorded_error"), _RUN_20260824_DIVERGENCE
)
def test_the_20260824_divergence_sequence_now_steers_back_instead_of_deeper(
    position: Point, course_degrees: float, recorded_error: float
) -> None:
    """Every step of the real divergence, replayed against the corrected law.

    The run recorded 46.639 -> 48.253 -> 77.396 deg of heading error while
    holding a saturated `angular_speed: +180` throughout, until the 0.30 m
    cross-track abort fired. At each of those three fixes the corrected
    controller must command the OPPOSITE sign, and integrating that command
    under the measured rotation law must shrink the error rather than grow it.
    """

    decision = continuous_control_decision(
        _RUN_20260824_ROUTE,
        _observation(
            position=position,
            course_heading_degrees=course_degrees,
            # Past the trust floor so this isolates the SIGN, not the
            # unconfirmed-heading gate that already suppresses these.
            distance_travelled_m=0.20,
        ),
    )

    assert decision.action == "drive"
    assert decision.heading_error_degrees == pytest.approx(recorded_error, abs=1e-5)
    # The run held +180. The fix holds -180.
    assert decision.angular_speed == -180

    assert decision.desired_course_degrees is not None
    stepped = _course_after(course_degrees, decision.angular_speed, 1.0)
    error_after = normalize_degrees(decision.desired_course_degrees - stepped)
    assert abs(error_after) < abs(recorded_error)


def test_the_20260824_run_converges_instead_of_diverging_over_its_whole_window() -> (
    None
):
    """End to end: the same opening state, integrated for the same 4 s window.

    Not a per-step check -- this runs the corrected controller in closed loop
    against the measured rotation law from the run's real opening course, and
    asserts the error is monotonically falling where the real run's rose.
    """

    course_degrees = _RUN_20260824_OPENING_COURSE
    errors: list[float] = []
    # ~1 Hz, the measured position/heading bundle rate, for the 4 s window.
    for _ in range(4):
        decision = continuous_control_decision(
            _RUN_20260824_ROUTE,
            _observation(
                position=Point(5.0628, -4.2926),
                course_heading_degrees=course_degrees,
                distance_travelled_m=0.20,
            ),
        )
        assert decision.action == "drive"
        assert decision.heading_error_degrees is not None
        errors.append(abs(decision.heading_error_degrees))
        course_degrees = _course_after(course_degrees, decision.angular_speed, 1.0)

    assert errors == sorted(errors, reverse=True), errors
    # The real run went 46.6 -> 48.3 -> 77.4. This one must end well below where
    # it started, though 4 s at ~9 deg/s cannot null 46.6 -- see
    # `alignment_feasibility`, which is why that error must be refused up front.
    assert errors[0] == pytest.approx(46.639242, abs=1e-5)
    assert errors[-1] < errors[0] - 25.0


# --- the 2026-08-24 opening-alignment preflight -------------------------------
#
# The SECOND defect that run exposed, independent of the sign inversion: an
# opening heading error large enough that nulling it, while driving forward the
# whole time, overruns the window's own budgets before the mower is aligned.


def _alignment(
    course_degrees: float,
    *,
    route: ContinuousRoute | None = None,
    opening: Point = Point(0.0, 0.0),
    position: Point = Point(0.15, 0.0),
    elapsed_s: float = 0.6044,
    cumulative_distance_m: float = 0.15,
    config: ContinuousControllerConfig | None = None,
):
    return alignment_feasibility(
        route or _route(),
        opening_position=opening,
        position=position,
        heading_evidence=HeadingEvidence(
            course_heading_degrees=course_degrees,
            measured_at_s=elapsed_s,
            chord_m=0.15,
            uncertainty_degrees=1.7,
        ),
        elapsed_s=elapsed_s,
        cumulative_distance_m=cumulative_distance_m,
        config=config,
    )


def test_the_20260824_opening_error_is_refused_as_geometrically_infeasible() -> None:
    """46.639 deg cannot be nulled in that run's 4 s / 1.0 m / 0.30 m budget."""

    verdict = _alignment(
        _RUN_20260824_OPENING_COURSE,
        route=_RUN_20260824_ROUTE,
        opening=_RUN_20260824_ROUTE.start,
        position=_RUN_20260824_ROUTE.start,
        config=ContinuousControllerConfig(max_window_s=4.0, max_distance_m=1.0),
    )

    assert verdict.feasible is False
    assert verdict.heading_error_degrees == pytest.approx(46.639242, abs=1e-5)
    # 0.604 s blind + 46.639 / 8.0 = 5.830 s turning, against a 4.0 s window --
    # time breaks first.
    assert verdict.blind_time_s == pytest.approx(0.6044, abs=1e-3)
    assert verdict.turn_time_s == pytest.approx(5.8299, abs=1e-3)
    assert verdict.total_time_s == pytest.approx(6.4343, abs=1e-3)
    assert verdict.limiting_factor == "window_s"
    # Blind excursion is now measured from live positions, not guessed from a
    # stale heading. This constructed replay starts on-line, while the turn
    # alone still exceeds the stricter 0.20 m admission bound.
    assert verdict.blind_cross_track_m == pytest.approx(0.0)
    assert abs(verdict.turn_cross_track_m) == pytest.approx(0.5571, abs=1e-3)
    assert verdict.total_cross_track_m == pytest.approx(0.5571, abs=1e-3)
    assert verdict.total_cross_track_m > verdict.cross_track_budget_m


def test_the_20260824_excursion_breaches_at_every_disputed_turn_rate() -> None:
    """The verdict does not depend on resolving the open turn-rate discrepancy.

    `CLAUDE.md` records an unresolved disagreement between a 2026-08-12
    single-pulse fit (11.224 deg/s at angular 180) and this week's steady-state
    measurement (9.386). A faster rate is the FAVOURABLE assumption -- less time
    turning, smaller excursion -- and even the fastest of them breaches 0.30 m
    on this geometry, so the refusal stands either way.
    """

    for rate in (8.0, 8.091, 8.964, 9.386, 11.224):
        verdict = _alignment(
            _RUN_20260824_OPENING_COURSE,
            route=_RUN_20260824_ROUTE,
            opening=_RUN_20260824_ROUTE.start,
            position=_RUN_20260824_ROUTE.start,
            config=ContinuousControllerConfig(
                max_window_s=4.0, max_distance_m=1.0, min_turn_rate_deg_per_s=rate
            ),
        )
        assert verdict.total_cross_track_m > 0.20, (rate, verdict.total_cross_track_m)
        assert abs(verdict.turn_cross_track_m) > 0.20, (
            rate,
            verdict.turn_cross_track_m,
        )
        assert verdict.feasible is False, rate


def test_a_well_aimed_opening_is_admitted() -> None:
    """The gate must not refuse the geometry the executor is FOR."""

    # Route runs due east from the origin; the mower is already on that course.
    verdict = _alignment(0.0)

    assert verdict.feasible is True
    assert verdict.limiting_factor is None
    assert verdict.heading_error_degrees == pytest.approx(0.0)
    assert verdict.turn_time_s == pytest.approx(0.0)
    assert verdict.total_cross_track_m == pytest.approx(0.0)
    # A perfectly aimed mower still spends the blind run, but straight down the
    # line it costs no cross-track at all.
    assert verdict.blind_cross_track_m == pytest.approx(0.0)
    assert verdict.total_time_s == pytest.approx(0.6044, abs=1e-3)


def test_a_modest_opening_error_still_fits_the_default_budget() -> None:
    """20 deg costs 3.10 s, 0.77 m and 0.159 m of excursion -- all inside."""

    verdict = _alignment(-20.0)

    assert verdict.heading_error_degrees == pytest.approx(20.0)
    assert verdict.turn_time_s == pytest.approx(2.5)
    assert verdict.turn_distance_m == pytest.approx(0.6205, abs=1e-3)
    assert verdict.turn_cross_track_m == pytest.approx(-0.1072, abs=1e-3)
    assert verdict.total_time_s == pytest.approx(3.1044, abs=1e-3)
    assert verdict.total_distance_m == pytest.approx(0.7705, abs=1e-3)
    assert verdict.total_cross_track_m == pytest.approx(0.1072, abs=1e-3)
    assert verdict.feasible is True


def test_live_signed_cross_track_distinguishes_inward_and_outward_turns() -> None:
    """Admission uses the live signed path, not an unsigned opening estimate."""

    inward = _alignment(
        -20.0,
        opening=Point(0.0, 0.19),
        position=Point(0.15, 0.19),
        config=ContinuousControllerConfig(max_window_s=8.0),
    )
    outward = _alignment(
        20.0,
        opening=Point(0.0, 0.19),
        position=Point(0.15, 0.19),
        config=ContinuousControllerConfig(max_window_s=8.0),
    )

    assert inward.opening_cross_track_m == pytest.approx(0.19)
    assert inward.current_cross_track_m == pytest.approx(0.19)
    assert inward.predicted_end_cross_track_m < inward.current_cross_track_m
    assert inward.feasible is True
    assert outward.predicted_end_cross_track_m > outward.current_cross_track_m
    assert outward.feasible is False
    assert outward.limiting_factor == "cross_track_m"


@pytest.mark.parametrize("signed_cross_track", [0.19, -0.19])
def test_signed_019_m_live_starts_fit_the_admission_bound(
    signed_cross_track: float,
) -> None:
    """Both sides of the route fit immediately inside the 0.20 m bound."""
    verdict = _alignment(
        0.0,
        opening=Point(0.0, signed_cross_track),
        position=Point(0.15, signed_cross_track),
        config=ContinuousControllerConfig(max_window_s=8.0),
    )

    assert verdict.feasible is True
    assert verdict.max_abs_cross_track_m == pytest.approx(0.19)


@pytest.mark.parametrize("signed_cross_track", [0.29, -0.29])
def test_signed_029_m_live_starts_are_refused_before_steering(
    signed_cross_track: float,
) -> None:
    """Both signs outside the admission limit are rejected symmetrically."""
    verdict = _alignment(
        0.0,
        opening=Point(0.0, signed_cross_track),
        position=Point(0.15, signed_cross_track),
        config=ContinuousControllerConfig(max_window_s=8.0),
    )

    assert verdict.feasible is False
    assert verdict.limiting_factor == "cross_track_m"
    assert verdict.cross_track_budget_m == pytest.approx(0.20)


@pytest.mark.parametrize(
    ("course_degrees", "config", "expected"),
    [
        (-28.0, ContinuousControllerConfig(), "window_s"),
        (-30.0, ContinuousControllerConfig(max_window_s=8.0), "cross_track_m"),
        # Squeeze the distance guard and it binds before either.
        (
            -25.0,
            ContinuousControllerConfig(max_window_s=8.0, max_distance_m=0.60),
            "distance_m",
        ),
    ],
)
def test_each_budget_limb_can_be_the_binding_one(
    course_degrees: float, config: ContinuousControllerConfig, expected: str
) -> None:
    """All three limbs do real work; none is decorative."""

    verdict = _alignment(course_degrees, config=config)

    assert verdict.feasible is False
    assert verdict.limiting_factor == expected


@pytest.mark.parametrize(
    ("route", "course_degrees"),
    [
        (ContinuousRoute(start=Point(1.0, 1.0), target=Point(1.0, 1.0)), 0.0),
        (ContinuousRoute(start=Point(0.0, 0.0), target=Point(3.0, 0.0)), float("nan")),
        (
            ContinuousRoute(start=Point(0.0, 0.0), target=Point(float("inf"), 0.0)),
            0.0,
        ),
    ],
)
def test_an_unusable_route_or_heading_fails_closed(
    route: ContinuousRoute, course_degrees: float
) -> None:
    """A degenerate input is never reported as feasible."""

    verdict = alignment_feasibility(
        route,
        opening_position=route.start,
        position=Point(route.start.x + 0.15, route.start.y),
        heading_evidence=HeadingEvidence(
            course_heading_degrees=course_degrees,
            measured_at_s=0.5,
            chord_m=0.15,
            uncertainty_degrees=1.7,
        ),
        elapsed_s=0.5,
        cumulative_distance_m=0.15,
    )

    assert verdict.feasible is False
    assert verdict.limiting_factor == "route_or_heading_unusable"


def test_alignment_refuses_heading_evidence_below_the_chord_floor() -> None:
    """The admission helper cannot launder an uninformative chord."""
    verdict = alignment_feasibility(
        _route(),
        opening_position=Point(0.0, 0.0),
        position=Point(0.14, 0.0),
        heading_evidence=HeadingEvidence(
            course_heading_degrees=0.0,
            measured_at_s=0.5,
            chord_m=0.14,
            uncertainty_degrees=1.8,
        ),
        elapsed_s=0.5,
        cumulative_distance_m=0.14,
    )

    assert verdict.feasible is False
    assert verdict.limiting_factor == "route_or_heading_unusable"


def test_the_turn_rate_bound_cannot_be_configured_away() -> None:
    """A zero or negative rate would break the feasibility arithmetic."""

    for bad in (0.0, -8.0):
        with pytest.raises(ValueError, match="controller bounds must be positive"):
            ContinuousControllerConfig(min_turn_rate_deg_per_s=bad)


def test_the_feasibility_verdict_is_serialisable_for_an_evidence_file() -> None:
    """It is reported as gate diagnostics, so it must survive `asdict`."""

    verdict = _alignment(
        _RUN_20260824_OPENING_COURSE,
        route=_RUN_20260824_ROUTE,
        opening=_RUN_20260824_ROUTE.start,
        position=_RUN_20260824_ROUTE.start,
    )
    payload = asdict(verdict)

    assert payload["feasible"] is False
    assert payload["limiting_factor"] == "window_s"
    assert set(payload) == {
        "desired_course_degrees",
        "heading_error_degrees",
        "opening_cross_track_m",
        "current_cross_track_m",
        "blind_travel_m",
        "blind_time_s",
        "blind_cross_track_m",
        "turn_time_s",
        "turn_distance_m",
        "turn_cross_track_m",
        "predicted_end_cross_track_m",
        "max_abs_cross_track_m",
        "total_time_s",
        "total_distance_m",
        "total_cross_track_m",
        "window_budget_s",
        "distance_budget_m",
        "cross_track_budget_m",
        "remaining_window_s",
        "remaining_distance_m",
        "model_turn_rate_deg_per_s",
        "model_abs_angular_command",
        "model_assumption",
        "feasible",
        "limiting_factor",
    }


def test_position_chord_threshold_and_course_signs() -> None:
    """Exactly 0.15 m qualifies and atan2 preserves both course signs."""
    assert (
        course_from_position_chord(
            Point(0.0, 0.0), Point(0.149, 0.0), measured_at_s=0.5
        )
        is None
    )
    east = course_from_position_chord(
        Point(0.0, 0.0), Point(0.15, 0.0), measured_at_s=0.5
    )
    north = course_from_position_chord(
        Point(0.0, 0.0), Point(0.0, 0.15), measured_at_s=0.5
    )
    south = course_from_position_chord(
        Point(0.0, 0.0), Point(0.0, -0.15), measured_at_s=0.5
    )

    assert east is not None and east.course_heading_degrees == pytest.approx(0.0)
    assert north is not None and north.course_heading_degrees == pytest.approx(90.0)
    assert south is not None and south.course_heading_degrees == pytest.approx(-90.0)


def test_blind_acquisition_requires_the_complete_106_m_disk() -> None:
    """Clearance must cover acquisition travel plus stopping overshoot."""
    narrow = [
        Point(-0.30, -0.30),
        Point(0.30, -0.30),
        Point(0.30, 0.30),
        Point(-0.30, 0.30),
    ]
    wide = [
        Point(-1.06, -1.06),
        Point(1.06, -1.06),
        Point(1.06, 1.06),
        Point(-1.06, 1.06),
    ]

    refused = blind_acquisition_feasibility(Point(0.0, 0.0), narrow)
    admitted = blind_acquisition_feasibility(Point(0.0, 0.0), wide)

    assert refused.required_radius_m == pytest.approx(1.06)
    assert refused.boundary_clearance_m == pytest.approx(0.30)
    assert refused.feasible is False
    assert admitted.boundary_clearance_m == pytest.approx(1.06)
    assert admitted.feasible is True


def test_invalid_polygon_refuses_blind_acquisition() -> None:
    """Malformed and non-finite corridor geometry fails closed."""
    invalid = blind_acquisition_feasibility(
        Point(0.0, 0.0), [Point(0.0, 0.0), Point(float("nan"), 1.0)]
    )

    assert invalid.feasible is False
    assert invalid.boundary_clearance_m is None


def test_self_intersecting_polygon_refuses_blind_acquisition() -> None:
    """A bow-tie corridor cannot be treated as a contained safety envelope."""
    bow_tie = [
        Point(-2.0, -2.0),
        Point(2.0, 2.0),
        Point(-2.0, 2.0),
        Point(2.0, -2.0),
    ]

    invalid = blind_acquisition_feasibility(Point(0.0, 0.0), bow_tie)

    assert invalid.feasible is False
    assert invalid.boundary_clearance_m is None
