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
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import voluptuous as vol
import yaml

from custom_components.mammotion import services
from custom_components.mammotion.services import (
    _STEP_RESPONSE_MAX_TOTAL_MS,
    _STEP_RESPONSE_MIN_SPEED_BY_LINEAR,
    STEP_RESPONSE_PROBE_SCHEMA,
    _in_window_ble_snapshot,
    _in_window_telemetry_sample,
    _step_response_analysis,
    _step_response_completion_reason,
    _step_response_course_series,
    _step_response_gates,
    _step_response_phase_scheduler,
    _step_response_probe,
)

from .test_continuous_motion_window import _FakeCoordinator, _position_stream

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


def test_schema_cannot_relax_the_travel_guard_above_four_point_five_metres() -> None:
    """A caller may tighten the distance guard but never relax it past 4.5 m.

    Raised from 3.0 m by docs/phase2-route1-predeclared-20260830.md -- a
    deliberate safety-bound increase, not a tightening.
    """
    assert _validated(max_travel_m=1.0)["max_travel_m"] == 1.0
    with pytest.raises(vol.Invalid):
        _validated(max_travel_m=4.51)


def test_schema_cannot_relax_the_step_phase_above_fifteen_seconds() -> None:
    """A caller may shorten the step phase but never relax it past 15000 ms.

    5000 -> 7000 by docs/phase2-route1-step-extension-predeclared-20260830.md.
    Then 7000 -> 15000 on 2026-09-01: criterion 2a's half-phase split gives the
    single onset-contaminated interval ~1/k of the first half's weight, so
    half_diff ~= |steady - onset| / k, and the worst observed contamination
    (10.43 deg/s) needs k >= 7 -- about 14 informative step intervals at the
    ~1 Hz VIO cadence. See
    docs/findings-plus180-split-is-onset-sampling-phase-20260901.md.

    ⚠️ This raises the CLOCK, not the distance: `max_travel_m` is untouched and
    the pairing that would overrun it is refused before dispatch.
    """
    assert _validated(step_ms=1000)["step_ms"] == 1000
    assert _validated(step_ms=15000)["step_ms"] == 15000
    with pytest.raises(vol.Invalid):
        _validated(step_ms=15001)


def test_max_total_window_is_pinned_at_twenty_three_seconds() -> None:
    """Pin the raised total-window cap so a future edit cannot drift it silently.

    docs/phase2-route1-predeclared-20260830.md moved this 12000 -> 14000 ms so
    baseline 3000 + step 5000 + settle 5000 = 13000 ms fits. Then
    docs/phase2-route1-step-extension-predeclared-20260830.md moved it
    14000 -> 16000 ms for baseline 3000 + step 7000 + settle 5000 = 15000 ms.
    Then 2026-09-01 moved it 16000 -> 23000 ms so baseline 3000 + step 15000 +
    settle 5000 = 23000 ms fits.
    """
    assert _STEP_RESPONSE_MAX_TOTAL_MS == 23000


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
        step_ms=15000,
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


# --- completion reason ---------------------------------------------------------
#
# Found 2026-08-30 on route-1 run 1: the window completed its full 13000 ms
# schedule (phase transitions on time, zero tripped samples, cumulative travel
# 2.71 m of a 4.00 m budget) and the service still reported
# "travel_guard_tripped". The old logic read `travel_abort.is_set()`, which the
# caller's own `finally` block sets UNCONDITIONALLY as mandatory-stop teardown
# -- so it was always true, on every real run, trip or no trip.


def test_reason_is_window_complete_when_the_refresh_loop_never_saw_the_abort() -> None:
    """A normal end of window must not read back as a safety trip."""
    assert (
        _step_response_completion_reason({"aborted_early": False}) == "window_complete"
    )
    assert _step_response_completion_reason({}) == "window_complete"


def test_reason_is_travel_guard_tripped_only_when_the_loop_observed_it() -> None:
    """The one signal that actually distinguishes a real trip from teardown."""
    assert (
        _step_response_completion_reason({"aborted_early": True})
        == "travel_guard_tripped"
    )


# --- the discriminator the 2026-08-28 abort could not provide ------------------


def _sample(handle: object) -> dict:
    class _Coordinator(_FakeCoordinator):
        class _Manager:
            @staticmethod
            def mower(_name: str) -> object:
                return handle

        manager = _Manager()

    return _in_window_telemetry_sample(
        _Coordinator(),
        index=0,
        window_started=time.monotonic(),
        command="send_movement",
        command_args={"linear_speed": 400, "angular_speed": 0},
    )


def test_sample_records_position_sequence_and_epoch() -> None:
    """The fields that separate a stale payload from an absent one.

    pymammotion bumps `_position_sequence` inside `_publish_position_sample`,
    which the handle calls ONLY when the decoded frame actually carried a
    position payload. So across a window where x/y never change, an ADVANCING
    sequence means payloads arrived carrying stale coordinates (observer lag)
    while a FROZEN one means no position payloads arrived at all (a feed stall).
    """
    handle = SimpleNamespace(
        last_report_at=123.0,
        position_epoch=4,
        latest_position_sample=SimpleNamespace(sequence=57),
    )
    sample = _sample(handle)
    assert sample["position_sequence"] == 57
    assert sample["position_epoch"] == 4
    assert sample["last_report_at_monotonic"] == 123.0


def test_sample_is_none_safe_when_no_position_has_ever_been_published() -> None:
    """`latest_position_sample` is legitimately None before the first payload."""
    handle = SimpleNamespace(
        last_report_at=0.0, position_epoch=1, latest_position_sample=None
    )
    sample = _sample(handle)
    assert sample["position_sequence"] is None
    assert sample["position_epoch"] == 1


def test_last_report_at_alone_cannot_answer_the_2026_08_28_question() -> None:
    """Pin why the extra fields exist, so nobody removes them as redundant.

    On 2026-08-28 `last_report_at` advanced three times across a 2.088 s window
    in which the mower travelled 0.4375 m and x/y never changed. It stamps every
    LubaMsg, so it proved frames arrived and nothing about position payloads.
    """
    frames_arriving = SimpleNamespace(
        last_report_at=200.0,
        position_epoch=2,
        latest_position_sample=SimpleNamespace(sequence=17),
    )
    first = _sample(frames_arriving)
    # A later frame arrives, but no position payload came with it.
    frames_arriving.last_report_at = 201.0
    second = _sample(frames_arriving)
    assert second["last_report_at_monotonic"] > first["last_report_at_monotonic"]
    assert second["position_sequence"] == first["position_sequence"]


# --- outbound BLE facts, recorded in-window -----------------------------------


def _handle_with_queue(
    *, connected: bool, depth: int, gate_set: bool, saga: bool
) -> object:
    class _Gate:
        @staticmethod
        def is_set() -> bool:
            return gate_set

    class _Q:
        @staticmethod
        def qsize() -> int:
            return depth

    return SimpleNamespace(
        last_report_at=1.0,
        position_epoch=1,
        latest_position_sample=SimpleNamespace(sequence=3),
        get_transport=lambda _t: SimpleNamespace(is_connected=connected),
        queue=SimpleNamespace(
            is_saga_active=saga, _transport_gate=_Gate(), _queue=_Q()
        ),
    )


def test_ble_snapshot_reports_the_four_discriminating_fields() -> None:
    """Connection, backlog, gating and saga are recorded separately.

    They separate different causes of a stalled position stream, so folding
    them into one verdict would lose exactly the information wanted.
    """

    class _C(_FakeCoordinator):
        class _Manager:
            @staticmethod
            def mower(_name: str) -> object:
                return _handle_with_queue(
                    connected=True, depth=7, gate_set=False, saga=True
                )

        manager = _Manager()

    snap = _in_window_ble_snapshot(_C())
    assert snap["is_connected"] is True
    assert snap["queue_depth"] == 7
    assert snap["queue_dispatch_paused"] is True
    assert snap["saga_active"] is True


def test_ble_snapshot_reads_none_rather_than_healthy_when_introspection_fails() -> None:
    """Absence must never read as healthy.

    `_ble_link_liveness` degrades to "not live" for the same reason: a gate that
    silently passes when it cannot see is the vacuously-true failure this
    project has been bitten by before.
    """

    class _C(_FakeCoordinator):
        class _Manager:
            @staticmethod
            def mower(_name: str) -> object:
                return SimpleNamespace(last_report_at=1.0)

        manager = _Manager()

    snap = _in_window_ble_snapshot(_C())
    assert snap == {
        "is_connected": None,
        "queue_depth": None,
        "queue_dispatch_paused": None,
        "saga_active": None,
    }


def test_telemetry_sample_carries_the_ble_snapshot() -> None:
    """The BLE facts ride the same 100 ms sample as position_sequence."""

    class _C(_FakeCoordinator):
        class _Manager:
            @staticmethod
            def mower(_name: str) -> object:
                return _handle_with_queue(
                    connected=False, depth=0, gate_set=True, saga=False
                )

        manager = _Manager()

    sample = _in_window_telemetry_sample(
        _C(),
        index=0,
        window_started=time.monotonic(),
        command="send_movement",
        command_args={"linear_speed": 400, "angular_speed": 0},
    )
    assert sample["ble"]["is_connected"] is False
    assert sample["ble"]["queue_dispatch_paused"] is False
    assert sample["position_sequence"] == 3


# --- the lease stops the report stream; the probe must restart it -------------
#
# 🐛 The probe's first bug, 2026-08-29. `exclusive_report_subscription` enqueues
# RPT_STOP and clears `_ble_stream_active` as its first act, and blocks the
# background loop from starting a new configuration for the life of the lease.
# The probe took the lease and drove without restarting the stream, so four runs
# across three builds recorded zero position payloads and were mis-read as a
# device- or backend-side feed stall.


def _real_run_snapshot() -> dict[str, Any]:
    return {
        "work_mode_label": "MODE_READY",
        "charge_state_label": "not_charging",
        "position": {
            "source": "test",
            "x": 0.0,
            "y": 0.0,
            "toward": 0.0,
            "pos_type_label": "AREA_INSIDE",
            "zone_hash": "1",
        },
        "blade": {"reported_state": 0, "current_cutter_rpm": 0},
    }


class _ReportingCoordinator(_FakeCoordinator):
    """Records the report-start calls the probe must make under its lease."""

    def __init__(self, position_stream: Any) -> None:
        super().__init__(position_stream)
        self.report_stream_starts = 0
        self.continuous_report_starts = 0

    async def async_start_report_stream(self, **_kwargs: Any) -> None:
        self.report_stream_starts += 1

    async def async_start_continuous_reports(self, **_kwargs: Any) -> None:
        self.continuous_report_starts += 1


async def _real_run(
    monkeypatch: pytest.MonkeyPatch, coordinator: Any, **overrides: Any
) -> dict[str, Any]:
    monkeypatch.setattr(
        services, "_custom_path_telemetry_snapshot", lambda _c: _real_run_snapshot()
    )
    monkeypatch.setattr(services, "_settle_ble_command_queue", _settle_stub)
    # These two tests isolate the report-stream handoff, so the environmental
    # gates (BLE preflight, map position) are stubbed passing. The gates
    # themselves are covered separately above.
    monkeypatch.setattr(
        services,
        "_step_response_gates",
        lambda *_a, **_k: [{"name": "stubbed", "passed": True, "detail": ""}],
    )
    return await _step_response_probe(
        coordinator,
        route_start={"x": 0.0, "y": 0.0},
        corridor_polygon=WIDE_CORRIDOR,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        confirm_step_response_run=True,
        **overrides,
    )


async def _settle_stub(_coordinator: Any) -> dict[str, Any]:
    return {"live": True, "queue_depth": 0}


async def test_probe_starts_the_report_stream_it_took_the_lease_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both start calls must be made, under a fresh generation."""
    coordinator = _ReportingCoordinator(_position_stream([]))
    result = await _real_run(monkeypatch, coordinator)
    assert coordinator.report_stream_starts == 1
    assert coordinator.continuous_report_starts == 1
    stream = result["report_stream"]
    assert stream["started"] is True
    assert stream["continuous_started"] is True
    assert stream["subscription_generation"]["generation"] >= 1


async def test_probe_refuses_to_drive_when_no_position_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FAIL CLOSED. An empty stream must refuse, not drive blind.

    This is the regression that would have caught the 2026-08-29 bug: without
    it the probe drives, records nothing, and the null reads as a device fault.
    """
    coordinator = _ReportingCoordinator(_position_stream([]))
    monkeypatch.setattr(services, "_STEP_RESPONSE_READINESS_TIMEOUT_S", 0.05)
    result = await _real_run(monkeypatch, coordinator)
    assert result["would_send"] is False
    assert result["blockers"] == ["position_subscription_not_ready"]
    assert result["command_result"]["attempted"] is False
    assert result["report_stream"]["ready"] is False


# --- the maxsize=1 defect, pinned so it cannot come back ----------------------


def test_no_position_stream_is_opened_one_deep() -> None:
    """Every position stream must use `_SAFETY_POSITION_STREAM_MAXSIZE`.

    `maxsize=1` structurally guarantees a false `position_sequence_gap`:
    `PositionSampleStream._offer` is latest-wins, and a safety consumer opens
    the stream then runs gates and starts reports before its first `get()`, so
    at ~1 Hz every sample in that gap is dropped.

    The beta80 fix reached the two lease wrappers and MISSED
    `_capture_in_window_telemetry` and the distance-guarded probe wrapper. That
    cost a linear-300 speed check on 2026-08-30, which aborted at 413 ms with
    `trip_reason: position_sequence_gap` and `travel_at_trip_m: 0.0` on a
    perfectly healthy feed. A source scan is used rather than a behavioural
    test because the failure is a *literal*: a new call site would reintroduce
    it silently and no existing test would notice.
    """
    source = Path(services.__file__).read_text(encoding="utf-8").replace(" ", "")
    assert "open_position_stream(maxsize=1)" not in source
    assert "open_position_sample_stream(maxsize=1)" not in source


# --- the 2026-09-01 long-step / slow-speed change ------------------------------


def test_linear_speed_300_is_admissible_and_400_stays_the_default() -> None:
    """E-VIO imposes no travel floor, so the slow speed is back on the menu.

    300 was eliminated on 2026-08-30 because the RTK course statistic needed a
    0.15 m chord and a 0.116 m/s mower does not produce one. E-VIO reads VIO
    heading between consecutive DISTINCT readings instead, so that objection
    does not transfer -- and driving slower is what lets a long step phase fit
    inside an unchanged `max_travel_m`.
    """
    assert _validated()["linear_speed"] == 400
    assert _validated(linear_speed=300)["linear_speed"] == 300
    for rejected in (0, 200, 350, 500):
        with pytest.raises(vol.Invalid):
            _validated(linear_speed=rejected)


def test_step_ms_reaches_15000_and_refuses_more() -> None:
    """2a needs ~14 informative step intervals at the worst observed onset."""
    assert _validated(step_ms=15000)["step_ms"] == 15000
    with pytest.raises(vol.Invalid):
        _validated(step_ms=15100)


async def test_a_long_window_at_the_FAST_speed_is_flagged_but_not_refused() -> None:
    """A LIKELY guard trip is surfaced, not blocked -- the bound refuses only the impossible.

    ⚠️ Corrected 2026-09-01. This test originally asserted a refusal, which only
    held because the table's 400 entry was 0.24 -- above four of the five banked
    observations, i.e. an upper bound mislabelled as a lower one. With the
    corrected 0.17 floor, 0.17 x 23 = 3.91 m clears the 4.5 m budget, so the
    refusal correctly does NOT fire: at the slowest speed the mower has shown,
    this window genuinely fits.

    The risk is real but probabilistic (typical 0.216 m/s x 23 s = 4.97 m would
    trip), so it is reported rather than enforced. Over-refusing feasible
    configurations is the failure this replaced.
    """
    result = await _step_response_probe(
        _FakeCoordinator(),
        route_start=START,
        corridor_polygon=WIDE_CORRIDOR,
        linear_speed=400,
        baseline_ms=3000,
        step_ms=15000,
        settle_ms=5000,
        max_travel_m=4.5,
        dry_run=True,
    )
    assert "step_window_travel_exceeds_budget" not in result["blockers"]
    projection = result["travel_projection"]
    assert projection["likely_guard_trip"] is True
    assert projection["floor_travel_m"] < 4.5 < projection["typical_travel_m"]


async def test_the_proposed_long_slow_run_IS_flagged_as_a_likely_trip() -> None:
    """The long/slow window does NOT fit, and the projection now says so.

    ⚠️ INVERTED 2026-09-03 by measurement. This previously asserted the opposite,
    because the projection used an EXTRAPOLATED 0.16 m/s for linear 300. Phase A
    measured the SUSTAINED speed at 0.223 m/s -- 39% higher -- so a 23 s window
    projects 5.13 m against a 4.5 m budget rather than the 3.7 m once claimed,
    and the 28 s window the cap-raise proposal wanted projects 5.9 m.

    The flag stays non-blocking by design: the lower bound still clears, so
    nothing is refused and the travel guard keeps carrying the safety. What
    changed is that the operator is now WARNED instead of reassured.
    """
    result = await _step_response_probe(
        _FakeCoordinator(),
        route_start=START,
        corridor_polygon=WIDE_CORRIDOR,
        linear_speed=300,
        baseline_ms=3000,
        step_ms=15000,
        settle_ms=5000,
        max_travel_m=4.5,
        dry_run=True,
    )
    projection = result["travel_projection"]
    assert projection["typical_travel_m"] == pytest.approx(5.129, abs=0.01)
    assert projection["likely_guard_trip"] is True
    assert "step_window_travel_exceeds_budget" not in result["blockers"]


async def test_the_same_long_window_at_linear_300_fits_the_unchanged_budget() -> None:
    """Slower driving is what makes the long step admissible at all."""
    result = await _step_response_probe(
        _FakeCoordinator(),
        route_start=START,
        corridor_polygon=WIDE_CORRIDOR,
        linear_speed=300,
        baseline_ms=3000,
        step_ms=15000,
        settle_ms=5000,
        max_travel_m=4.5,
        dry_run=True,
    )
    assert "step_window_travel_exceeds_budget" not in result["blockers"]


async def test_the_travel_refusal_does_not_tighten_the_existing_defaults() -> None:
    """A MARGINAL config still dispatches; only an impossible one is refused.

    Regression. The first version of this gate used a rounded-UP speed and so
    refused the schema's own defaults (10 s at 0.28 m/s = 2.8 m against the
    2.50 m default budget) -- tightening long-standing behaviour as a side
    effect of raising the step ceiling. The bound is a LOWER one precisely so
    that the travel guard, not a projection, keeps carrying the safety.
    """
    result = await _step_response_probe(
        _FakeCoordinator(),
        route_start=START,
        corridor_polygon=WIDE_CORRIDOR,
        dry_run=True,
    )
    assert "step_window_travel_exceeds_budget" not in result["blockers"]


def test_min_speed_table_is_below_every_banked_full_window_rate() -> None:
    """The table must be a LOWER bound, checked against real runs, not asserted.

    Regression, 2026-09-01. The 400 entry shipped as 0.24 -- taken from the single
    FASTEST banked run while its own comment called it the slowest -- so it sat
    above four of the five banked observations and over-refused. A replay of
    banked route-1 run 1 (3000/5000/5000 at max_travel_m 3.0) was refused
    pre-dispatch although both banked runs of it travelled 2.71 m and 2.77 m.

    Pinning it at the single default point (as the first version of this suite
    did) cannot catch that. This measures the invariant against the evidence.
    """

    raw_dir = Path(__file__).resolve().parents[3] / "docs" / "raw-samples"
    rates = []
    for path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("linear_speed") != 400:
            continue
        points: list[tuple[float, float]] = []
        for sample in payload["samples"]:
            position = sample.get("position") or {}
            if position.get("x") is None:
                continue
            point = (position["x"], position["y"])
            if not points or point != points[-1]:
                points.append(point)
        travel = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(points, points[1:], strict=False)
        )
        rates.append(travel / (payload["motion_refresh"]["elapsed_ms"] / 1000))

    assert rates, "no banked linear-400 runs found to check the bound against"
    assert _STEP_RESPONSE_MIN_SPEED_BY_LINEAR[400] < min(rates), (
        f"table entry {_STEP_RESPONSE_MIN_SPEED_BY_LINEAR[400]} is not "
        f"below the slowest banked run ({min(rates):.4f} m/s) -- it would "
        "over-refuse feasible configurations"
    )


async def test_replaying_a_banked_config_is_not_refused_by_the_travel_gate() -> None:
    """The exact banked route-1 run-1 config must still be dispatchable.

    Both banked runs of 3000/5000/5000 at linear 400 travelled 2.71 m and 2.77 m,
    so a tightened max_travel_m of 3.0 genuinely fits. The pre-dispatch gate must
    not refuse a configuration the mower has already completed.
    """
    result = await _step_response_probe(
        _FakeCoordinator(),
        route_start=START,
        corridor_polygon=WIDE_CORRIDOR,
        linear_speed=400,
        baseline_ms=3000,
        step_ms=5000,
        settle_ms=5000,
        max_travel_m=3.0,
        dry_run=True,
    )
    assert "step_window_travel_exceeds_budget" not in result["blockers"]


def test_services_yaml_and_strings_agree_with_the_schema_and_the_code() -> None:
    """services.yaml, strings.json and the schema must not drift apart.

    Every sibling service has a parity test of this shape and this one did not,
    so on 2026-09-01 a fix to strings.json left services.yaml behind: the same
    field quoted 0.16 m/s in one file and 0.12 m/s in the other, and 0.26 m/s
    for linear 400 where the code's own table says 0.216. All 1003 tests were
    green throughout, because nothing read the two files together.
    """
    root = Path(__file__).resolve().parents[3] / "custom_components" / "mammotion"
    fields = yaml.safe_load((root / "services.yaml").read_text())[
        "raw_pymammotion_step_response_probe"
    ]["fields"]
    strings = json.loads((root / "strings.json").read_text())["services"][
        "raw_pymammotion_step_response_probe"
    ]["fields"]

    # Same field set in both files.
    assert set(fields) == set(strings), (
        f"services.yaml and strings.json disagree on fields: "
        f"{set(fields) ^ set(strings)}"
    )

    # A select selector's default must be the SAME TYPE as its options, or the
    # HA dropdown matches nothing and renders with no preselected value.
    for name, spec in fields.items():
        select = (spec.get("selector") or {}).get("select")
        if not select or "default" not in spec:
            continue
        assert spec["default"] in select["options"], (
            f"{name}: default {spec['default']!r} is not among its select "
            f"options {select['options']!r}"
        )

    # Numeric bounds in the YAML must match the voluptuous schema's real limits.
    validated = STEP_RESPONSE_PROBE_SCHEMA(
        {
            "entity_id": ENTITY,
            "route_start": START,
            "corridor_polygon": WIDE_CORRIDOR,
            "step_ms": fields["step_ms"]["selector"]["number"]["max"],
        }
    )
    assert validated["step_ms"] == fields["step_ms"]["selector"]["number"]["max"]

    # No speed figure may be quoted in one file and contradicted in the other.
    for name in ("linear_speed", "step_ms"):
        assert strings[name]["description"], f"{name} has no strings.json description"
    assert "0.12" not in fields["linear_speed"]["description"], (
        "services.yaml quotes a ramp-inclusive 0.12 m/s for linear 300 while "
        "strings.json and _STEP_RESPONSE_TYPICAL_SPEED_BY_LINEAR use ~0.16"
    )


def test_containment_uses_the_WORST_of_travel_budget_and_wall_clock() -> None:
    """The corridor must hold the path even if the distance guard does nothing.

    🚨 `max_travel_m + overshoot` assumes the guard WORKS. This project has a
    documented mode where it silently does not: position payloads keep arriving
    with an advancing sequence and a fresh timestamp while x/y stay latched
    (2026-08-28, 21 bit-identical samples across 0.4375 m of real travel). Then
    `cumulative_distance_m` stays ~0, nothing trips, and the window runs to the
    wall clock.

    `raw_pymammotion_motion_probe` was corrected for this on 2026-08-23; this
    probe was missed for four cap raises. Found by adversarial review 2026-09-02.
    """
    # 23 s at linear 400: clock bound 0.30 * 23 = 6.90 m beats 4.5 + 0.5 = 5.0 m.
    gates = _step_response_gates(
        _FakeCoordinator(),
        {"position": {"x": 0.0, "y": 0.0}},
        route_start=START,
        corridor_polygon=WIDE_CORRIDOR,
        max_travel_m=4.5,
        linear_speed=400,
        total_ms=23000,
        dry_run=True,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )
    diagnostics = _gate(gates, "step_path_contained")["diagnostics"]
    assert diagnostics["clock_bound_m"] == pytest.approx(6.90, abs=0.01)
    assert diagnostics["travel_budget_bound_m"] == pytest.approx(5.0)
    assert diagnostics["required_radius_m"] == pytest.approx(6.90, abs=0.01)
    assert diagnostics["bound_that_binds"] == "clock"


def test_short_windows_still_bind_on_the_travel_budget() -> None:
    """The clock bound must not quietly inflate every ordinary run's corridor."""
    gates = _step_response_gates(
        _FakeCoordinator(),
        {"position": {"x": 0.0, "y": 0.0}},
        route_start=START,
        corridor_polygon=WIDE_CORRIDOR,
        max_travel_m=2.5,
        linear_speed=300,
        total_ms=8000,
        dry_run=True,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )
    diagnostics = _gate(gates, "step_path_contained")["diagnostics"]
    # Phase A: clock bound 0.225 * 8 = 1.80 m, under the 3.0 m travel bound.
    assert diagnostics["clock_bound_m"] == pytest.approx(1.80, abs=0.01)
    assert diagnostics["required_radius_m"] == pytest.approx(3.0)
    assert diagnostics["bound_that_binds"] == "travel_budget"
