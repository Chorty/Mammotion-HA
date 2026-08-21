"""Offline tests for the non-dispatching continuous controller prototype."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from custom_components.mammotion.continuous_controller import (
    ContinuousControllerConfig,
    ContinuousObservation,
    ContinuousRoute,
    Point,
    continuous_control_decision,
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
    return ContinuousObservation(**(defaults | overrides))  # type: ignore[arg-type]


def test_healthy_aligned_observation_requests_continuous_drive() -> None:
    """An aligned, healthy route asks to hold forward with no steering."""

    decision = continuous_control_decision(_route(), _observation())

    assert decision.action == "drive"
    assert decision.reason == "tracking_route"
    assert decision.linear_speed == 400
    assert decision.angular_speed == 0
    assert decision.aim_point == Point(0.8, 0.0)


def test_heading_error_steers_in_the_correct_direction_and_clamps() -> None:
    """Heading corrections use the shortest sign and never exceed the bound."""

    right = continuous_control_decision(
        _route(), _observation(course_heading_degrees=-30.0)
    )
    left = continuous_control_decision(
        _route(), _observation(course_heading_degrees=30.0)
    )

    assert right.angular_speed == 180
    assert left.angular_speed == -180
    assert right.heading_error_degrees is not None
    assert left.heading_error_degrees is not None
    assert right.heading_error_degrees > 0
    assert left.heading_error_degrees < 0


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"cancelled": True}, "operator_cancelled"),
        ({"stop_available": False}, "stop_primitive_unavailable"),
        ({"ble_live": False}, "ble_link_not_live"),
        ({"refresh_healthy": False}, "refresh_cadence_unhealthy"),
        ({"refresh_age_s": 1.21}, "refresh_age_exceeded"),
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
    assert decision.cross_track_m == pytest.approx(0.03)


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
    assert decision.predicted_position == Point(1.28, 0.0)


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
    assert decision.predicted_position.x == pytest.approx(0.35)


def test_lookahead_corrects_cross_track_before_the_endpoint() -> None:
    """Lookahead requests an early correction toward the route centerline."""

    decision = continuous_control_decision(
        _route(),
        _observation(position=Point(0.5, 0.10), course_heading_degrees=0.0),
    )

    assert decision.action == "drive"
    assert decision.angular_speed < 0
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

    assert imports <= {"__future__", "dataclasses", "math", "typing"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0.0, 0.0), (180.0, -180.0), (181.0, -179.0), (-181.0, 179.0)],
)
def test_heading_normalization(raw: float, expected: float) -> None:
    """Heading differences always take the signed shortest rotation."""

    assert normalize_degrees(raw) == expected
