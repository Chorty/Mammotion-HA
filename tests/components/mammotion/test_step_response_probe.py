"""Offline tests for `raw_pymammotion_step_response_probe`.

The open-loop dead-time probe from
`docs/phase2-dead-time-step-test-design-20260828.md`. It answers whether the
control lag measured on steering attempt 5 sits in the ACTUATOR or the OBSERVER,
and how large it is.

🔑 This probe deliberately has **no controller** -- no route, no aim point, no
steering law, no corridor-breach override. The question is about the plant, so
nothing in a loop may influence the answer.

Every real-motion path is exercised through `dry_run=True` or a blocked gate
here: no coordinator I/O, no BLE, no mower command. `would_send` must never be
`True` in this file.
"""

from __future__ import annotations

import asyncio
import time

import pytest
import voluptuous as vol

from custom_components.mammotion.services import (
    STEP_RESPONSE_PROBE_SCHEMA,
    _step_response_analysis,
    _step_response_course_series,
    _step_response_gates,
    _step_response_phase_scheduler,
    _step_response_probe,
)

from .test_continuous_motion_window import _FakeCoordinator

ENTITY = "lawn_mower.test"
START = {"x": 0.0, "y": 0.0}
# Half-width 3.5 m clears the default 2.50 m budget plus 0.50 m of stop
# overshoot with margin.
WIDE_CORRIDOR = [
    {"x": -3.5, "y": -3.5},
    {"x": 3.5, "y": -3.5},
    {"x": 3.5, "y": 3.5},
    {"x": -3.5, "y": 3.5},
]
# Half-width 1.0 m cannot contain a 3.00 m disk.
NARROW_CORRIDOR = [
    {"x": -1.0, "y": -1.0},
    {"x": 1.0, "y": -1.0},
    {"x": 1.0, "y": 1.0},
    {"x": -1.0, "y": 1.0},
]


def _validated(**overrides: object) -> dict:
    return STEP_RESPONSE_PROBE_SCHEMA(
        {
            "entity_id": ENTITY,
            "route_start": START,
            "corridor_polygon": WIDE_CORRIDOR,
            **overrides,
        }
    )


# --- schema -------------------------------------------------------------------


def test_schema_defaults_match_the_design_document() -> None:
    """Defaults are the design document's numbers, not convenience values."""
    data = _validated()
    assert data["dry_run"] is True
    assert data["linear_speed"] == 400
    assert data["step_angular_speed"] == 120
    assert data["baseline_ms"] == 3000
    assert data["step_ms"] == 3000
    assert data["settle_ms"] == 4000
    assert data["max_travel_m"] == 2.50
    assert data["confirm_step_response_run"] is False


@pytest.mark.parametrize("value", [0, 60, 100, 500])
def test_schema_rejects_unmeasured_angular_commands(value: int) -> None:
    """Only the measured 120-180 band, either sign.

    A smaller value is not "safer": it is unmeasured, and may sit in the
    actuation deadband where the mower turns less than commanded or not at all.
    """
    with pytest.raises(vol.Invalid):
        _validated(step_angular_speed=value)


@pytest.mark.parametrize("value", [-180, -120, 120, 180])
def test_schema_accepts_both_signs_of_the_measured_band(value: int) -> None:
    """Accept both signs of the measured band.

    A one-sided step cannot separate carryover from a direction-dependent
    drivetrain asymmetry.
    """
    assert _validated(step_angular_speed=value)["step_angular_speed"] == value


def test_schema_cannot_relax_the_travel_guard_above_three_metres() -> None:
    """A caller may tighten the distance guard but never relax it past 3.0 m."""
    assert _validated(max_travel_m=1.0)["max_travel_m"] == 1.0
    with pytest.raises(vol.Invalid):
        _validated(max_travel_m=3.01)


# --- gates --------------------------------------------------------------------


def _gates(corridor: list[dict[str, float]], **kwargs: object) -> list[dict]:
    return _step_response_gates(
        _FakeCoordinator(),
        {"position": {"x": 0.0, "y": 0.0}},
        route_start=START,
        corridor_polygon=corridor,
        max_travel_m=2.50,
        dry_run=True,
        confirm_blades_off=True,
        confirm_clear_area=True,
        **kwargs,
    )


def _gate(gates: list[dict], name: str) -> dict:
    return next(gate for gate in gates if gate["name"] == name)


def test_containment_requires_travel_budget_plus_stop_overshoot() -> None:
    """The path curves and its shape is the unknown, so the whole disk is required."""
    gate = _gate(_gates(WIDE_CORRIDOR), "step_path_contained")
    assert gate["passed"] is True
    assert gate["diagnostics"]["required_radius_m"] == pytest.approx(3.0)
    assert gate["diagnostics"]["boundary_clearance_m"] == pytest.approx(3.5)


def test_narrow_corridor_refuses_the_open_loop_path() -> None:
    """A corridor that cannot hold the whole disk refuses before any command."""
    gate = _gate(_gates(NARROW_CORRIDOR), "step_path_contained")
    assert gate["passed"] is False
    assert gate["diagnostics"]["boundary_clearance_m"] == pytest.approx(1.0)


def test_required_radius_tracks_the_travel_budget() -> None:
    """Shrinking the budget shrinks the disk -- containment is COMPUTED, not fixed."""
    gates = _step_response_gates(
        _FakeCoordinator(),
        {"position": {"x": 0.0, "y": 0.0}},
        route_start=START,
        corridor_polygon=NARROW_CORRIDOR,
        max_travel_m=0.25,
        dry_run=True,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )
    gate = _gate(gates, "step_path_contained")
    assert gate["diagnostics"]["required_radius_m"] == pytest.approx(0.75)
    assert gate["passed"] is True


# --- refusals -----------------------------------------------------------------


async def test_dry_run_sends_nothing() -> None:
    """The default path plans everything and dispatches nothing."""
    result = await _step_response_probe(
        _FakeCoordinator(),
        route_start=START,
        corridor_polygon=WIDE_CORRIDOR,
        dry_run=True,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )
    assert result["would_send"] is False
    assert result["reason"] == "dry_run"
    assert result["command_result"]["attempted"] is False


async def test_real_run_refuses_without_the_per_call_opt_in() -> None:
    """Arming the motion gate is deliberately not sufficient to drive a curve."""
    result = await _step_response_probe(
        _FakeCoordinator(),
        route_start=START,
        corridor_polygon=WIDE_CORRIDOR,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        confirm_step_response_run=False,
    )
    assert result["would_send"] is False
    assert "step_response_run_not_confirmed" in result["blockers"]
    assert result["command_result"]["attempted"] is False


async def test_total_window_is_capped_even_when_every_phase_is_legal() -> None:
    """Each phase may be at its own maximum while the SUM is still refused."""
    result = await _step_response_probe(
        _FakeCoordinator(),
        route_start=START,
        corridor_polygon=WIDE_CORRIDOR,
        baseline_ms=5000,
        step_ms=5000,
        settle_ms=6000,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        confirm_step_response_run=True,
    )
    assert result["would_send"] is False
    assert "step_window_too_long" in result["blockers"]


# --- phase scheduler ----------------------------------------------------------


async def test_scheduler_walks_baseline_step_settle() -> None:
    """The shared command walks 0 -> step -> 0 and ends back at zero."""
    command_state = {"linear_speed": 400, "angular_speed": 0}
    # A REAL monotonic base, so the scheduler actually waits out each phase
    # rather than finding every deadline already in the past.
    started = time.monotonic()
    transitions = await _step_response_phase_scheduler(
        command_state,
        window_started=started,
        baseline_ms=20,
        step_ms=20,
        step_angular_speed=120,
        abort_event=asyncio.Event(),
    )
    assert [row["phase"] for row in transitions] == ["baseline", "step", "settle"]
    assert [row["angular_speed"] for row in transitions] == [0, 120, 0]
    # The settle phase must leave the shared command at zero: it is what the
    # serialized writer keeps resending until the mandatory stop.
    assert command_state["angular_speed"] == 0
    # Ordered in time, and the step really did begin after the baseline.
    assert transitions[1]["elapsed_ms"] >= 20
    assert transitions[2]["elapsed_ms"] >= 40


async def test_scheduler_stops_advancing_once_the_guard_trips() -> None:
    """An aborted window must never be handed a fresh turn command."""
    command_state = {"linear_speed": 400, "angular_speed": 0}
    abort = asyncio.Event()
    abort.set()
    transitions = await _step_response_phase_scheduler(
        command_state,
        window_started=0.0,
        baseline_ms=20,
        step_ms=20,
        step_angular_speed=120,
        abort_event=abort,
    )
    assert [row["phase"] for row in transitions] == ["baseline"]
    assert command_state["angular_speed"] == 0


# --- analysis -----------------------------------------------------------------


def _samples(points: list[tuple[float, float, float]]) -> list[dict]:
    return [{"elapsed_ms": t, "position": {"x": x, "y": y}} for t, x, y in points]


def test_course_series_labels_phases_by_interval_midpoint() -> None:
    """A chord is an interval AVERAGE, so it belongs to its midpoint."""
    series = _step_response_course_series(
        _samples([(0, 0.0, 0.0), (1000, 1.0, 0.0), (2000, 2.0, 0.0)]),
        baseline_ms=1000,
        step_ms=1000,
        min_chord_m=0.15,
    )
    assert [row["phase"] for row in series] == ["baseline", "step"]
    assert series[0]["midpoint_elapsed_ms"] == pytest.approx(500)
    assert series[0]["course_degrees"] == pytest.approx(0.0)


def test_sub_floor_chords_are_marked_uninformative_and_carry_no_course() -> None:
    """At sigma = 0.0031 m a short chord's bearing is noise, not a measurement."""
    series = _step_response_course_series(
        _samples([(0, 0.0, 0.0), (1000, 0.02, 0.0)]),
        baseline_ms=1000,
        step_ms=1000,
        min_chord_m=0.15,
    )
    assert series[0]["informative"] is False
    assert series[0]["course_degrees"] is None


def test_analysis_divides_post_zero_rotation_by_the_steady_rate() -> None:
    """Tau is |rotation after commanding zero| / steady rate during the step."""
    series = [
        {
            "phase": "step",
            "informative": True,
            "midpoint_elapsed_ms": 1000,
            "course_degrees": 0.0,
        },
        {
            "phase": "step",
            "informative": True,
            "midpoint_elapsed_ms": 3000,
            "course_degrees": 20.0,
        },
        {
            "phase": "settle",
            "informative": True,
            "midpoint_elapsed_ms": 4000,
            "course_degrees": 30.0,
        },
    ]
    analysis = _step_response_analysis(series, baseline_ms=0, step_ms=3500)
    assert analysis["omega_step_deg_per_s"] == pytest.approx(10.0)
    assert analysis["rotation_after_zero_deg"] == pytest.approx(10.0)
    assert analysis["tau_actuator_s"] == pytest.approx(1.0)


def test_analysis_is_none_rather_than_guessing_when_evidence_is_thin() -> None:
    """One informative interval cannot yield a rate; it must not invent one."""
    series = [
        {
            "phase": "step",
            "informative": True,
            "midpoint_elapsed_ms": 1000,
            "course_degrees": 0.0,
        },
    ]
    analysis = _step_response_analysis(series, baseline_ms=0, step_ms=2000)
    assert analysis["omega_step_deg_per_s"] is None
    assert analysis["tau_actuator_s"] is None


def test_rotation_through_the_wrap_is_not_read_as_a_full_turn() -> None:
    """normalize_degrees keeps the signed short way round across +/-180."""
    series = [
        {
            "phase": "step",
            "informative": True,
            "midpoint_elapsed_ms": 0,
            "course_degrees": 170.0,
        },
        {
            "phase": "step",
            "informative": True,
            "midpoint_elapsed_ms": 1000,
            "course_degrees": -170.0,
        },
        {
            "phase": "settle",
            "informative": True,
            "midpoint_elapsed_ms": 2000,
            "course_degrees": -160.0,
        },
    ]
    analysis = _step_response_analysis(series, baseline_ms=0, step_ms=1500)
    assert analysis["omega_step_deg_per_s"] == pytest.approx(20.0)
    assert analysis["rotation_after_zero_deg"] == pytest.approx(10.0)
