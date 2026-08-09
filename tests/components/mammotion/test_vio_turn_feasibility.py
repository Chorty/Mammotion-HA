"""Tests for the fail-closed VIO turn-budget feasibility guard.

Anchor: the 2026-08-03 Gate 4 retry
(docs/evidence-gate4-beta19-retry-real-result-20260803.json). Segment 1
dispatched a 167.413 deg turn against a 4-command budget whose observed
rotation rate (16.5-21.3 deg/s at refresh 200 / angular 500 / 1500 ms pulses)
could never reach the 18 deg tolerance. The executor burned the budget and
0.185 m of translation before stopping `max_commands_reached`. The guard must
refuse that turn BEFORE the first turn command, keep genuinely feasible turns
running, and keep the refusal distinguishable from mid-turn budget exhaustion
and from `max_linear_commands_reached`.

Second anchor: the 2026-08-04 daylight turn characterization
(docs/evidence-turnchar-beta19-analysis-20260804.json), four supervised turns
at +45/-90/+135/-170 deg that ALL reached target. Pooled with Gate 4 it gives
13 refresh-200 pulses across two geometries, and it is the reason translation
is bounded per DEGREE rather than per second: rotation rate varied 16.5-49.6
deg/s, so a per-second bound is not a fixed quantity, and the old per-second
figure both understated the worst case and compounded with a command count
derived from the pessimistic rotation floor. Those runs are ground truth --
the guard must admit all four and still refuse Gate 4.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.services import (
    _VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND,
    _VIO_TURN_CONSERVATIVE_TRANSLATION_M_PER_DEGREE,
    _VIO_TURN_MEASURED_MINIMUM_DEGREES_PER_SECOND,
    _raw_pymammotion_execute_multi_segment,
    _raw_pymammotion_execute_vector_segment,
    _vio_turn_budget_feasibility,
    _vio_turn_commands_at_rate,
    _vio_turn_to_heading,
)

from .test_map_task_visibility import _pulse_coordinator

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import diagnose_motion_result  # noqa: E402

GATE4_EVIDENCE = (
    REPO_ROOT / "docs" / "evidence-gate4-beta19-retry-real-result-20260803.json"
)

# The recorded Gate 4 segment-1 turn, exactly as retained.
GATE4_INITIAL_VISION_HEADING = 6.479514644483135
GATE4_TARGET_VISION_HEADING = 173.8920323680674
GATE4_INITIAL_ERROR_DEGREES = 167.413


def test_helper_refuses_the_recorded_gate4_near_180_turn() -> None:
    """The retained 167.4 deg / 4-command case is judged infeasible up front."""
    feasibility = _vio_turn_budget_feasibility(
        initial_error_degrees=GATE4_INITIAL_ERROR_DEGREES,
        heading_tolerance_degrees=18.0,
        max_commands=4,
        pulse_duration_ms=1500.0,
        motion_refresh_interval_ms=200,
        max_displacement_m=0.25,
    )

    assert feasibility["feasible"] is False
    assert feasibility["reason"] == "turn_budget"
    # 16.5 deg/s (minimum observed) * 1.5 s = 24.75 deg/command;
    # ceil(149.413 / 24.75) = 7 commands needed against a budget of 4.
    assert feasibility["per_command_rotation_bound_degrees"] == pytest.approx(24.75)
    assert feasibility["rotation_bound_source"] == (
        "conservative_observed_rate_with_refresh"
    )
    assert feasibility["estimated_commands_needed"] == 7
    assert feasibility["max_commands"] == 4


def test_helper_keeps_a_90_degree_junction_turn_feasible() -> None:
    """A 90 deg L-path junction still fits the accepted 4-command budget."""
    feasibility = _vio_turn_budget_feasibility(
        initial_error_degrees=90.0,
        heading_tolerance_degrees=18.0,
        max_commands=4,
        pulse_duration_ms=1500.0,
        motion_refresh_interval_ms=200,
        max_displacement_m=0.25,
    )

    assert feasibility["feasible"] is True
    assert feasibility["reason"] == "within_budget"
    # 4 commands, not the 3 the pre-beta32 closed form reported. The closed form
    # assumed every command ran the full 1500 ms; the beta31 overshoot ceiling
    # shortens them to 1500/1394/1048/787 ms as the error closes, and the model
    # now replays that. The verdict is unchanged but the margin is gone: an
    # L-path junction is feasible at EXACTLY the 4-command budget, so any further
    # shortening -- or any rate below the 16.5 deg/s floor, which
    # `test_conservative_rate_floor_is_known_optimistic` shows has already been
    # measured -- makes the canonical junction infeasible.
    assert feasibility["estimated_commands_needed"] == 4
    assert feasibility["max_commands"] == 4
    assert feasibility["command_count_model"] == "executor_pulse_policy_replay"
    assert feasibility["modelled_pulse_durations_ms"] == [1500.0, 1387.5, 1005.9, 729.3]
    # 90 deg * 0.0026 m/deg = 0.234 m stays under the 0.25 m cap. This headroom
    # is what bounds the per-degree constant from above: anything over
    # 0.25/90 = 0.002778 would refuse an L-path junction, which is the exact
    # geometry Gate 4 needs.
    assert feasibility["estimated_translation_m"] == pytest.approx(0.234, abs=0.001)


def test_helper_refuses_when_estimated_translation_exceeds_the_cap() -> None:
    """Enough commands may exist while their translation would breach the cap."""
    feasibility = _vio_turn_budget_feasibility(
        initial_error_degrees=130.0,
        heading_tolerance_degrees=18.0,
        max_commands=8,
        pulse_duration_ms=1500.0,
        motion_refresh_interval_ms=200,
        max_displacement_m=0.25,
    )

    # 6 ceiling-aware commands fit the 8-command budget, but sweeping 130 deg at
    # 0.0026 m/deg = 0.338 m would exceed the 0.25 m cap the run enforces anyway.
    # The two criteria are independent and the budget one must not mask this.
    assert feasibility["feasible"] is False
    assert feasibility["reason"] == "translation_cap"
    assert feasibility["estimated_commands_needed"] == 6
    assert feasibility["estimated_translation_m"] == pytest.approx(0.338, abs=0.001)


def test_feasibility_model_replays_the_executors_own_pulse_policy() -> None:
    """The planning model and the executor must not be able to disagree.

    beta31 added an overshoot ceiling that shortens turn pulses as the error
    closes, but left the preflight assuming every command ran the full pulse.
    The preflight therefore promised turns the executor could not finish -- the
    Gate 4 failure mode, reintroduced from the planning side. The model now
    calls the same `_turn_final_approach_pulse_ms` the turn loop calls, so the
    two cannot drift again.

    The regression guard: at the accepted 4-command budget the two models
    disagree over 100-117 deg. A closed form over full-length pulses says
    ceil((105 - 18) / 24.75) = 4 commands and admits the turn; the real policy
    needs 5, because the ceiling shortens every pulse after the first. Anything
    in that band would have been dispatched and then aborted mid-turn after four
    pulses and real translation. 117 deg is where the closed form itself gives
    up, so the whole band was previously invisible.
    """
    error = 105.0
    closed_form = math.ceil(
        (error - 18.0) / (_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND * 1.5)
    )
    modelled, pulses = _vio_turn_commands_at_rate(
        initial_error_degrees=error,
        heading_tolerance_degrees=18.0,
        degrees_per_second=_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND,
        turn_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        slow_pulse_duration_ms=700.0,
        slow_threshold_degrees=15.0,
        refresh_interval_ms=200,
    )

    assert closed_form == 4
    assert modelled == 5
    # Full pulses while the error is above the 72 deg binding threshold, then
    # monotonically shorter ones once the ceiling takes over. It is the tail the
    # closed form cannot see.
    assert pulses == [1500.0, 1500.0, 1225.0, 888.1, 643.9]
    assert pulses == sorted(pulses, reverse=True)

    # And the guard now refuses it at the accepted 4-command budget instead of
    # dispatching it.
    feasibility = _vio_turn_budget_feasibility(
        initial_error_degrees=error,
        heading_tolerance_degrees=18.0,
        max_commands=4,
        pulse_duration_ms=1500.0,
        motion_refresh_interval_ms=200,
        max_displacement_m=0.3,
    )

    assert feasibility["feasible"] is False
    assert feasibility["reason"] == "turn_budget"
    assert feasibility["command_count_model"] == "executor_pulse_policy_replay"


def test_feasibility_estimator_seed_must_be_the_configured_rate() -> None:
    """Seeding the model's estimator with the floor would bias it fail-OPEN.

    `_vio_turn_commands_at_rate` takes two rates: what the mower does
    physically (the conservative floor) and what the executor's estimator
    believes before it has measured anything (the configured rate). Passing the
    floor for both looks harmless and is not -- a lower believed rate makes the
    modelled `needed_ms` larger, which keeps pulses on the full-length
    "cruising" branch and yields a LOWER command count, i.e. the model would
    under-count exactly the way the closed form did.
    """
    kwargs = {
        "initial_error_degrees": 60.0,
        "heading_tolerance_degrees": 18.0,
        "degrees_per_second": _VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND,
        "pulse_duration_ms": 1500.0,
        "slow_pulse_duration_ms": 700.0,
        "slow_threshold_degrees": 15.0,
        "refresh_interval_ms": 200,
    }
    correct, _ = _vio_turn_commands_at_rate(turn_degrees_per_second=37.0, **kwargs)
    seeded_with_floor, _ = _vio_turn_commands_at_rate(
        turn_degrees_per_second=_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND, **kwargs
    )

    assert correct is not None and seeded_with_floor is not None
    assert seeded_with_floor <= correct


def test_conservative_rate_floor_is_known_optimistic() -> None:
    """Pins a live inaccuracy so it cannot quietly become load-bearing.

    `_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND` claims to be the minimum
    rotation rate ever observed. It is not, and has not been moved: Gate 5
    attempt 5 measured 14.490 deg/s against a delivered window, ~12% below it.
    Lowering the constant is not a free correction -- at 14.4 deg/s a 90 deg
    junction needs 5 commands against the accepted budget of 4, so a truthful
    floor and L-path junctions cannot both hold at the current
    `vio_turn_max_commands`. This test exists so that trade is a decision
    somebody makes, not a comment somebody misses.
    """
    assert (
        _VIO_TURN_MEASURED_MINIMUM_DEGREES_PER_SECOND
        < _VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND
    )

    at_measured_minimum, _ = _vio_turn_commands_at_rate(
        initial_error_degrees=90.0,
        heading_tolerance_degrees=18.0,
        degrees_per_second=_VIO_TURN_MEASURED_MINIMUM_DEGREES_PER_SECOND,
        turn_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        slow_pulse_duration_ms=700.0,
        slow_threshold_degrees=15.0,
        refresh_interval_ms=200,
    )

    assert at_measured_minimum == 5
    assert at_measured_minimum > 4


def test_helper_uses_the_single_shot_quantum_without_refresh() -> None:
    """Refresh 0 rotation is a duration-independent quantum, not a rate."""
    refused = _vio_turn_budget_feasibility(
        initial_error_degrees=100.0,
        heading_tolerance_degrees=18.0,
        max_commands=8,
        pulse_duration_ms=1500.0,
        motion_refresh_interval_ms=0,
        max_displacement_m=0.5,
    )
    allowed = _vio_turn_budget_feasibility(
        initial_error_degrees=40.0,
        heading_tolerance_degrees=8.0,
        max_commands=8,
        pulse_duration_ms=1500.0,
        motion_refresh_interval_ms=0,
        max_displacement_m=0.5,
    )

    assert refused["rotation_bound_source"] == "single_shot_rotation_quantum_floor"
    assert refused["per_command_rotation_bound_degrees"] == pytest.approx(8.0)
    assert refused["feasible"] is False
    assert refused["reason"] == "turn_budget"
    assert allowed["feasible"] is True
    # No trustworthy single-shot translation figure exists (the 0.0026 m/deg
    # bound is a refresh-regime measurement); the runtime displacement cap
    # bounds translation during execution instead of a preflight estimate.
    assert refused["translation_bound_m_per_degree"] is None
    assert refused["translation_bound_source"] is None
    assert refused["estimated_translation_m"] is None


#: The 2026-08-04 daylight characterization, as retained. Each entry is
#: (label, initial error deg, commands actually sent, actual displacement m).
#: All four reached target, so the guard must judge every one feasible.
CHARACTERIZATION_RUNS = [
    ("run1 +45", 45.0, 1, 0.012912009913255463),
    ("run2 -90", 90.0, 2, 0.13031937691686574),
    ("run3 +135", 135.0, 2, 0.02875065216651603),
    ("run4 -170", 170.0, 4, 0.2955293555638759),
]


@pytest.mark.parametrize(
    ("label", "initial_error_degrees", "actual_commands", "actual_displacement_m"),
    CHARACTERIZATION_RUNS,
)
def test_helper_admits_every_characterization_turn_that_succeeded(
    label: str,
    initial_error_degrees: float,
    actual_commands: int,
    actual_displacement_m: float,
) -> None:
    """All four 2026-08-04 turns reached target, so none may be refused.

    This is the regression that rejected simply raising the old per-second
    bound: at 0.0720 m/s the +135 and -170 deg runs were refused with estimates
    of 0.540 m and 0.756 m against actuals of 0.029 m and 0.296 m.
    """
    feasibility = _vio_turn_budget_feasibility(
        initial_error_degrees=initial_error_degrees,
        heading_tolerance_degrees=18.0,
        max_commands=8,
        pulse_duration_ms=1500.0,
        motion_refresh_interval_ms=200,
        max_displacement_m=0.5,
    )

    assert feasibility["feasible"] is True, label
    assert feasibility["reason"] == "within_budget"
    # The bound must stay on the conservative side of what the hardware did...
    assert feasibility["estimated_translation_m"] >= actual_displacement_m
    # ...while the pessimistic rotation floor still over-counts real commands.
    assert feasibility["estimated_commands_needed"] >= actual_commands


def test_translation_estimate_scales_with_angle_not_command_budget() -> None:
    """Translation is a geometry question, so the command budget cannot move it.

    The superseded model multiplied a per-command translation by a count
    derived from the pessimistic rotation floor, compounding two anti-correlated
    worst cases: a slow pulse sweeps fewer degrees and therefore drags less.
    """
    same_angle = [
        _vio_turn_budget_feasibility(
            initial_error_degrees=120.0,
            heading_tolerance_degrees=18.0,
            max_commands=budget,
            pulse_duration_ms=1500.0,
            motion_refresh_interval_ms=200,
            max_displacement_m=1.0,
        )["estimated_translation_m"]
        for budget in (5, 6, 8, 12)
    ]

    assert same_angle == [pytest.approx(0.312, abs=0.001)] * 4

    # Halving the pulse doubles the commands needed but sweeps the same arc.
    long_pulse = _vio_turn_budget_feasibility(
        initial_error_degrees=120.0,
        heading_tolerance_degrees=18.0,
        max_commands=16,
        pulse_duration_ms=1500.0,
        motion_refresh_interval_ms=200,
        max_displacement_m=1.0,
    )
    short_pulse = _vio_turn_budget_feasibility(
        initial_error_degrees=120.0,
        heading_tolerance_degrees=18.0,
        max_commands=16,
        pulse_duration_ms=750.0,
        motion_refresh_interval_ms=200,
        max_displacement_m=1.0,
    )

    assert (
        short_pulse["estimated_commands_needed"]
        > (long_pulse["estimated_commands_needed"])
    )
    assert short_pulse["estimated_translation_m"] == pytest.approx(
        long_pulse["estimated_translation_m"]
    )


def test_translation_bound_stays_within_its_evidence_and_refusal_limits() -> None:
    """The per-degree constant is boxed in from both sides; pin both walls."""
    bound = _VIO_TURN_CONSERVATIVE_TRANSLATION_M_PER_DEGREE

    # Floor: at or above the pooled maximum over 13 pulses / two geometries.
    assert bound >= 0.002410
    # Ceiling: a 90 deg junction at a 0.25 m cap must stay feasible...
    assert bound <= 0.25 / 90
    # ...and the proven -170 deg turn at the schema's 0.5 m default likewise.
    assert bound <= 0.5 / 170


def test_helper_reports_an_already_in_tolerance_error_as_feasible() -> None:
    """No commands needed means trivially feasible, with zero estimates."""
    feasibility = _vio_turn_budget_feasibility(
        initial_error_degrees=5.0,
        heading_tolerance_degrees=18.0,
        max_commands=4,
        pulse_duration_ms=1500.0,
        motion_refresh_interval_ms=200,
        max_displacement_m=0.25,
    )

    assert feasibility["feasible"] is True
    assert feasibility["reason"] == "already_within_tolerance"
    assert feasibility["estimated_commands_needed"] == 0
    assert feasibility["estimated_translation_m"] == 0.0


@pytest.mark.asyncio
async def test_real_turn_refuses_the_recorded_case_before_any_command() -> None:
    """The Gate 4 turn is refused pre-dispatch: zero commands, zero translation."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=GATE4_INITIAL_VISION_HEADING, vio_state=2
    )

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=GATE4_TARGET_VISION_HEADING,
        heading_tolerance_degrees=18.0,
        pulse_duration_ms=1500,
        max_commands=4,
        max_displacement_m=0.25,
        motion_refresh_interval_ms=200,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "turn_budget_infeasible"
    assert result["commands_sent"] == 0
    assert result["final_displacement_m"] == 0.0
    assert result["would_send"] is False
    assert result["turn_feasibility"]["feasible"] is False
    assert result["turn_feasibility"]["reason"] == "turn_budget"
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_a_feasible_real_turn_still_runs_to_its_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard reports its math on the feasible path and does not block it."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        vi = coordinator.data.report_data.vision_info
        vi.heading = min(30.0, vi.heading + 10.0)

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=30.0,
        heading_tolerance_degrees=8.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "target_heading_reached"
    assert result["turn_feasibility"]["feasible"] is True
    assert result["commands_sent"] > 0


@pytest.mark.asyncio
async def test_dry_run_previews_the_same_feasibility_math() -> None:
    """A dry run exposes the refusal math without sending or refusing."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=GATE4_INITIAL_VISION_HEADING, vio_state=2
    )

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=GATE4_TARGET_VISION_HEADING,
        heading_tolerance_degrees=18.0,
        pulse_duration_ms=1500,
        max_commands=4,
        max_displacement_m=0.25,
        motion_refresh_interval_ms=200,
    )

    assert result["dry_run"] is True
    assert result["stop_reason"] == "dry_run"
    assert result["turn_feasibility"]["feasible"] is False
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vector_segment_surfaces_the_refusal_stop_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-dispatch refusal is not collapsed into `turn_phase_incomplete`."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )
    refusal_feasibility = {
        "feasible": False,
        "reason": "turn_budget",
        "estimated_commands_needed": 7,
        "max_commands": 4,
    }

    async def fake_calibration(
        coordinator_arg: object, **kwargs: object
    ) -> dict[str, object]:
        return {
            "passed": True,
            "reason": "calibrated",
            "offset_degrees": -90.0,
            "map_motion_heading_degrees": 280.0,
            "vision_heading": 10.0,
            "vio_state": 2,
            "distance_m": 0.06,
            "pulses_sent": 1,
            "command_results": [],
        }

    async def fake_vio_turn(
        coordinator_arg: object, **kwargs: object
    ) -> dict[str, object]:
        return {
            "stop_reason": "turn_budget_infeasible",
            "turn_feasibility": refusal_feasibility,
            "commands_sent": 0,
            "command_results": [],
        }

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_commands=1,
        vio_max_realignments=0,
        motion_refresh_interval_ms=200,
        heading_tolerance_degrees=18.0,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "turn_budget_infeasible"
    assert result["turn_feasibility"] == refusal_feasibility
    assert result["linear_commands_sent"] == 0
    # The refusal itself dispatched nothing; only the calibration drive ran.
    assert result["turn_commands_sent"] == 0


@pytest.mark.asyncio
async def test_multi_segment_real_run_refuses_an_infeasible_junction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A near-reversal junction refuses the whole path before any motion."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def fail_if_called(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("no segment may execute on an infeasible path")

    monkeypatch.setattr(
        mammotion_services,
        "_raw_pymammotion_execute_vector_segment",
        fail_if_called,
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [
            {"x": 1.0, "y": 1.0},
            {"x": 2.0, "y": 1.0},
            # Doubles back: the junction turn is ~179 degrees.
            {"x": 1.0, "y": 1.02},
        ],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "path_turn_infeasible"
    assert result["segments_executed"] == 0
    junctions = result["junction_turn_feasibility"]
    assert len(junctions) == 1
    assert junctions[0]["segment_index"] == 2
    assert abs(junctions[0]["turn_degrees"]) > 170
    assert junctions[0]["feasibility"]["feasible"] is False
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_multi_segment_dry_run_reports_junctions_without_refusing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run keeps the advisory pattern: same math, no refusal."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def fake_vector(
        coordinator_arg: object,
        points: list[dict[str, float]],
        **kwargs: object,
    ) -> dict[str, object]:
        return {
            "valid": True,
            "stop_reason": "dry_run",
            "blockers": [],
            "phases": [{"passed": True}, {"passed": True}],
            "final_telemetry": mammotion_services._custom_path_telemetry_snapshot(  # noqa: SLF001
                coordinator
            ),
            "progress_diagnostics": [],
        }

    monkeypatch.setattr(
        mammotion_services,
        "_raw_pymammotion_execute_vector_segment",
        fake_vector,
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [
            {"x": 1.0, "y": 1.0},
            {"x": 2.0, "y": 1.0},
            {"x": 1.0, "y": 1.02},
        ],
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "dry_run"
    assert result["junction_turn_feasibility"][0]["feasibility"]["feasible"] is False
    coordinator.manager.send_command_with_args.assert_not_called()


def test_diagnose_still_classifies_the_retained_gate4_budget_exhaustion() -> None:
    """The retained evidence keeps its mid-turn budget-exhaustion classification."""
    report = diagnose_motion_result.diagnose(json.loads(GATE4_EVIDENCE.read_text()))

    assert report["classification"] == "vio_turn_budget_exhausted_before_linear_phase"
    assert report["segment_stop_reason"] == "turn_phase_incomplete"
    assert report["turn"]["stop_reason"] == "max_commands_reached"
    assert report["linear"]["started"] is False


def test_diagnose_classifies_a_preflight_refusal_distinctly() -> None:
    """A refused turn is not reported as budget exhaustion or a linear failure."""
    refusal_feasibility = {
        "feasible": False,
        "reason": "turn_budget",
        "estimated_commands_needed": 7,
        "max_commands": 4,
    }
    document = {
        "result": {
            "stop_reason": "segment_failed",
            "failed_segment_index": 1,
            "segments": [
                {
                    "index": 1,
                    "result": {
                        "stop_reason": "turn_budget_infeasible",
                        "turn_feasibility": refusal_feasibility,
                        "turn_commands_sent": 0,
                        "linear_commands_sent": 0,
                        "phases": [
                            {
                                "name": "turn_to_target_heading",
                                "result": {
                                    "stop_reason": "turn_budget_infeasible",
                                    "turn_feasibility": refusal_feasibility,
                                    "commands_sent": 0,
                                    "command_results": [],
                                },
                            }
                        ],
                        "vio": {"target_vision_heading": 173.892},
                    },
                }
            ],
        }
    }

    report = diagnose_motion_result.diagnose(document)

    assert report["classification"] == "vio_turn_refused_infeasible_preflight"
    assert report["turn"]["commands_sent"] == 0
    assert report["turn"]["turn_feasibility"] == refusal_feasibility


def test_diagnose_keeps_linear_budget_exhaustion_distinct() -> None:
    """`max_linear_commands_reached` stays a linear classification, not a turn one."""
    document = {
        "result": {
            "stop_reason": "segment_failed",
            "failed_segment_index": 1,
            "segments": [
                {
                    "index": 1,
                    "result": {
                        "stop_reason": "max_linear_commands_reached",
                        "turn_commands_sent": 2,
                        "linear_commands_sent": 1,
                        "phases": [
                            {
                                "name": "turn_to_target_heading",
                                "result": {
                                    "stop_reason": "target_heading_reached",
                                    "commands_sent": 2,
                                    "command_results": [],
                                },
                            },
                            {
                                "name": "linear_forward_to_target",
                                "result": {
                                    "stop_reason": "max_linear_commands_reached",
                                },
                            },
                        ],
                        "vio": {"target_vision_heading": 90.0},
                    },
                }
            ],
        }
    }

    report = diagnose_motion_result.diagnose(document)

    assert report["classification"] == "linear_budget_exhausted"
    assert report["turn"]["stop_reason"] == "target_heading_reached"


def test_diagnose_names_a_refused_reverse_recovery() -> None:
    """A refused U-turn must name itself, not fall through to 'inspect'."""
    guard = {
        "after_linear_pulse": 1,
        "facing_degrees": 120.0,
        "bearing_degrees": 0.0,
        "aim_error_degrees": -120.0,
        "max_forward_realignment_degrees": 90.0,
        "reason": "target_requires_reverse_recovery",
    }
    document = {
        "result": {
            "stop_reason": "segment_failed",
            "failed_segment_index": 1,
            "segments": [
                {
                    "index": 1,
                    "result": {
                        "stop_reason": "target_requires_reverse_recovery",
                        "reverse_recovery_guard": guard,
                        "turn_commands_sent": 2,
                        "linear_commands_sent": 1,
                        "phases": [
                            {
                                "name": "turn_to_target_heading",
                                "result": {
                                    "stop_reason": "target_heading_reached",
                                    "commands_sent": 2,
                                    "command_results": [],
                                },
                            },
                        ],
                        "vio": {"target_vision_heading": 90.0},
                    },
                }
            ],
        }
    }

    report = diagnose_motion_result.diagnose(document)

    assert report["classification"] == "forward_only_segment_refused_reverse_recovery"
    assert report["reverse_recovery_guard"] == guard
    assert report["linear"]["started"] is True


def test_diagnose_names_an_exhausted_realignment_budget() -> None:
    """The off-bearing stop is distinct from a linear budget exhaustion."""
    document = {
        "result": {
            "stop_reason": "segment_failed",
            "failed_segment_index": 1,
            "segments": [
                {
                    "index": 1,
                    "result": {
                        "stop_reason": "vio_realign_budget_exhausted",
                        "turn_commands_sent": 4,
                        "linear_commands_sent": 1,
                        "realignments": [],
                        "phases": [
                            {
                                "name": "turn_to_target_heading",
                                "result": {
                                    "stop_reason": "target_heading_reached",
                                    "commands_sent": 2,
                                    "command_results": [],
                                },
                            },
                        ],
                        "vio": {"target_vision_heading": 90.0},
                    },
                }
            ],
        }
    }

    report = diagnose_motion_result.diagnose(document)

    assert report["classification"] == "vio_realign_budget_exhausted_before_target"
    assert report["reverse_recovery_guard"] is None
