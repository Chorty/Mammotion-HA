"""Tests for the vector/manual-velocity segment executor: pulses, re-aim, post-turn correction."""

import asyncio
import math
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.coordinator import (
    MammotionReportUpdateCoordinator,
)
from custom_components.mammotion.manual_motion import ManualMotionCancelledError
from custom_components.mammotion.services import (
    _app_scale_speeds,
    _app_speed_scale_report,
    _custom_path_telemetry_snapshot,
    _experimental_execute_segment_burst,
    _forward_two_pulse_latency_test,
    _is_zero_motion_stop_nudge,
    _manual_velocity_best_heading_decision,
    _manual_velocity_controller_decision,
    _manual_velocity_cumulative_pulse_test,
    _manual_velocity_heading_calibration,
    _manual_velocity_path_progress_diagnostic,
    _manual_velocity_pulse_test,
    _manual_velocity_segment_test,
    _motion_open_sleep,
    _motion_refresh_window,
    _position_feedback_diagnostic,
    _position_source_comparison,
    _raw_motion_readiness_test,
    _raw_multi_segment_phase_passed,
    _raw_pymammotion_execute_multi_segment,
    _raw_pymammotion_execute_segment,
    _raw_pymammotion_execute_vector_segment,
    _raw_pymammotion_motion_probe,
    _raw_vector_readiness_phase_passed,
    _raw_vector_readiness_test,
    _realign_cannot_improve_the_landing,
    _settle_linear_position_feed,
    _vio_motion_probe,
    _vio_segment_calibration_drive,
    _wrap_exclusive_manual_motion,
)

from .conftest import _patch_services_monotonic, _pulse_coordinator


def test_manual_velocity_controller_simulates_forward_when_heading_aligned() -> None:
    """Controller chooses a forward pulse when heading already faces target."""
    decision = _manual_velocity_controller_decision(
        [{"x": 1.0, "y": 1.0}, {"x": 4.0, "y": 1.0}],
        {
            "position": {
                "x": 1.0,
                "y": 1.0,
                "toward": 0.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
    )

    assert decision["mode"] == "simulated"
    assert decision["would_send"] is False
    assert decision["action"] == "forward"
    assert decision["reason"] == "heading_aligned"
    assert decision["target_index"] == 1
    assert decision["distance_to_target"] == 3.0
    assert decision["command_not_sent"] == {
        "service": "mammotion.move_forward",
        "data": {"speed": 0.2, "use_wifi": False},
    }


def test_manual_velocity_controller_simulates_turn_left_or_right() -> None:
    """Controller turns toward the next waypoint before moving forward."""
    points = [{"x": 1.0, "y": 1.0}, {"x": 4.0, "y": 1.0}]

    left = _manual_velocity_controller_decision(
        points,
        {
            "position": {
                "x": 1.0,
                "y": 1.0,
                "toward": 270.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
    )
    right = _manual_velocity_controller_decision(
        points,
        {
            "position": {
                "x": 1.0,
                "y": 1.0,
                "toward": 90.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
    )

    assert left["action"] == "turn_left"
    assert left["heading_error_degrees"] == 90.0
    assert left["command_not_sent"]["service"] == "mammotion.move_left"
    assert right["action"] == "turn_right"
    assert right["heading_error_degrees"] == -90.0
    assert right["command_not_sent"]["service"] == "mammotion.move_right"


def test_manual_velocity_controller_applies_heading_offset() -> None:
    """Heading offset corrects reported heading before choosing an action."""
    decision = _manual_velocity_controller_decision(
        [{"x": 1.0, "y": 1.0}, {"x": 4.0, "y": 1.0}],
        {
            "position": {
                "x": 1.0,
                "y": 1.0,
                "toward": 250.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
        heading_offset_degrees=110.0,
    )

    assert decision["current_heading_degrees"] == 250.0
    assert decision["corrected_heading_degrees"] == 0.0
    assert decision["heading_offset_degrees"] == 110.0
    assert decision["action"] == "forward"
    assert decision["reason"] == "heading_aligned"


def test_manual_velocity_best_heading_decision_selects_forward_candidate() -> None:
    """Candidate selection prefers an aligned forward command over a turn."""
    decision = _manual_velocity_best_heading_decision(
        [{"x": 1.0, "y": 1.0}, {"x": 4.0, "y": 1.0}],
        {
            "position": {
                "x": 1.0,
                "y": 1.0,
                "toward": 0.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
        heading_offset_degrees=110.0,
        heading_offset_candidates=[110.0, 0.0, 90.0],
    )

    assert decision["action"] == "forward"
    assert decision["reason"] == "heading_aligned"
    assert decision["selected_heading_offset_degrees"] == 0.0
    assert decision["heading_offset_candidates"] == [110.0, 0.0, 90.0]
    assert [
        item["heading_offset_degrees"]
        for item in decision["heading_offset_diagnostics"]
    ] == [
        110.0,
        0.0,
        90.0,
    ]


def test_manual_velocity_controller_skips_stale_start_waypoint() -> None:
    """Controller does not turn back to an obsolete drawn start point."""
    decision = _manual_velocity_controller_decision(
        [{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 0.08}],
        {
            "position": {
                "x": 0.0,
                "y": 0.04,
                "toward": 90.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
        waypoint_tolerance=0.03,
    )

    assert decision["target_index"] == 1
    assert decision["action"] == "forward"
    assert decision["reason"] == "heading_aligned"


def test_manual_velocity_controller_skips_start_after_segment_progress() -> None:
    """Controller skips start once mower has positive projection on first segment."""
    decision = _manual_velocity_controller_decision(
        [{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": -0.16}],
        {
            "position": {
                "x": 0.002,
                "y": -0.053,
                "toward": 270.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
        waypoint_tolerance=0.03,
    )

    assert decision["target_index"] == 1
    assert decision["action"] == "forward"


def test_manual_velocity_controller_uses_later_segment_projection() -> None:
    """Controller targets the next point on the closest later path segment."""
    decision = _manual_velocity_controller_decision(
        [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 0.0},
            {"x": 1.0, "y": 1.0},
        ],
        {
            "position": {
                "x": 1.02,
                "y": 0.45,
                "toward": 90.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
        waypoint_tolerance=0.03,
    )

    assert decision["target_index"] == 2
    assert decision["action"] == "forward"


def test_manual_velocity_controller_keeps_target_after_start_progress() -> None:
    """Controller keeps targeting endpoint after forward progress along segment."""
    decision = _manual_velocity_controller_decision(
        [
            {"x": 4.5424, "y": -0.9319},
            {"x": 4.587795864039179, "y": -1.0853249508005016},
        ],
        {
            "position": {
                "x": 4.5447,
                "y": -0.9849,
                "toward": 176.4826,
                "source": "mowing_state",
            }
        },
        speed=0.4,
        waypoint_tolerance=0.03,
        heading_offset_degrees=110.0,
    )

    assert decision["target_index"] == 1
    assert decision["action"] == "forward"
    assert abs(decision["heading_error_degrees"]) < 15


def test_manual_velocity_path_progress_requires_target_direction() -> None:
    """Forward progress must project toward the active target."""
    before = {
        "position": {
            "x": 0.0,
            "y": 0.0,
            "toward": 0.0,
            "source": "mowing_state",
        }
    }
    decision = {
        "action": "forward",
        "target": {"x": 1.0, "y": 0.0},
    }

    toward = _manual_velocity_path_progress_diagnostic(
        before,
        {
            "position": {
                "x": 0.1,
                "y": 0.0,
                "toward": 0.0,
                "source": "mowing_state",
            }
        },
        decision,
        min_progress_distance=0.02,
        min_heading_change_degrees=1.0,
    )
    away = _manual_velocity_path_progress_diagnostic(
        before,
        {
            "position": {
                "x": -0.1,
                "y": 0.0,
                "toward": 0.0,
                "source": "mowing_state",
            }
        },
        decision,
        min_progress_distance=0.02,
        min_heading_change_degrees=1.0,
    )

    assert toward["passed"] is True
    assert toward["status"] == "path_progress"
    assert away["passed"] is False
    assert away["status"] == "no_path_progress"


def test_manual_velocity_controller_stops_without_live_position() -> None:
    """Controller refuses to plan movement without live map-local position."""
    decision = _manual_velocity_controller_decision(
        [{"x": 1.0, "y": 1.0}, {"x": 4.0, "y": 1.0}],
        {"position": {"x": None, "y": None, "toward": None, "source": "unavailable"}},
        speed=0.2,
    )

    assert decision["action"] == "stop"
    assert decision["reason"] == "live_position_unavailable"
    assert decision["command_not_sent"] is None


@pytest.mark.asyncio
async def test_raw_pymammotion_motion_probe_defaults_to_dry_run() -> None:
    """Raw pymammotion probe default sends no command and reports exact call."""
    coordinator = _pulse_coordinator()

    result = await _raw_pymammotion_motion_probe(coordinator, sample_delays=())

    assert result["service"] == "raw_pymammotion_motion_probe"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["reason"] == "dry_run"
    assert result["in_window_telemetry"] == {
        "enabled": False,
        "sample_interval_ms": 0,
        "source": "coordinator_cache_only",
        "extra_ble_report_requests_during_window": 0,
        "planned_max_samples": 0,
        "report_stream_plan": [],
        "samples": [],
        "summary": None,
    }
    assert result["command_not_sent"] == {
        "manager_method": "send_command_with_args",
        "device_name": "Luba-Test",
        "command": "send_movement",
        "prefer_ble": True,
        "kwargs": {"linear_speed": 400, "angular_speed": 0},
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_probe_in_window_instrumentation_dry_run_is_complete_and_inert() -> (
    None
):
    """The Phase-1 plan is inspectable without starting a stream or motion."""
    coordinator = _pulse_coordinator()

    result = await _raw_pymammotion_motion_probe(
        coordinator,
        linear_speed=400,
        angular_speed=180,
        motion_refresh_interval_ms=200,
        in_window_sample_interval_ms=100,
        duration_ms=4000,
        sample_delays=(),
    )

    instrumentation = result["in_window_telemetry"]
    assert result["reason"] == "dry_run"
    assert result["would_send"] is False
    assert instrumentation["enabled"] is True
    assert instrumentation["planned_max_samples"] == 41
    assert instrumentation["source"] == "coordinator_cache_only"
    assert instrumentation["extra_ble_report_requests_during_window"] == 0
    assert instrumentation["report_stream_plan"] == [
        "async_start_report_stream",
        "async_start_continuous_reports",
    ]
    coordinator.async_start_report_stream.assert_not_awaited()
    coordinator.manager.request_iot_sync.assert_not_awaited()
    coordinator.manager.request_iot_sync_continuous.assert_not_awaited()
    coordinator.manager.send_command_with_args.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_probe_in_window_instrumentation_requires_refreshed_motion() -> None:
    """Instrumentation cannot silently produce an empty single-shot capture."""
    coordinator = _pulse_coordinator()

    result = await _raw_pymammotion_motion_probe(
        coordinator,
        in_window_sample_interval_ms=100,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(),
    )

    assert result["reason"] == "safety_gates_failed"
    assert "in_window_sampling_requires_motion_refresh" in result["blockers"]
    coordinator.async_start_report_stream.assert_not_awaited()
    coordinator.manager.send_command_with_args.assert_not_awaited()


@pytest.mark.asyncio
async def test_in_window_sampler_reads_cache_at_bounded_cadence_without_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sampler polls local state only and stops at the planned sample bound."""
    coordinator = _pulse_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay
        coordinator.data.mowing_state.pos_x += 0.1
        coordinator.data.mowing_state.toward += 1.0
        handle.last_report_at = clock["now"]

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)

    samples = await mammotion_services._capture_in_window_telemetry(  # noqa: SLF001
        coordinator,
        sample_interval_ms=100,
        duration_ms=300,
        window_started=100.0,
        stop_event=asyncio.Event(),
        command="send_movement",
        command_args={"linear_speed": 400, "angular_speed": 180},
    )

    assert [sample["elapsed_ms"] for sample in samples] == pytest.approx(
        [0.0, 100.0, 200.0, 300.0]
    )
    assert samples[-1]["position"]["x"] == pytest.approx(1.3)
    assert samples[-1]["last_report_at_monotonic"] == pytest.approx(100.3)
    assert samples[-1]["active_command"]["kwargs"] == {
        "linear_speed": 400,
        "angular_speed": 180,
    }
    coordinator.manager.request_iot_sync.assert_not_awaited()
    coordinator.manager.request_iot_sync_continuous.assert_not_awaited()
    coordinator.async_get_reports.assert_not_awaited()


def test_in_window_telemetry_summary_measures_freshness_and_course_before_stop() -> (
    None
):
    """Phase-1 go/no-go inputs come from changes observed inside the window."""
    samples = [
        {
            "elapsed_ms": elapsed,
            "last_report_at_monotonic": stamp,
            "position": {"x": x, "y": 2.0, "toward": toward},
        }
        for elapsed, stamp, x, toward in (
            (0.0, 10.0, 1.0, 90.0),
            (100.0, 10.0, 1.0, 90.0),
            (900.0, 10.9, 1.2, 91.0),
            (1900.0, 11.9, 1.4, 94.0),
            (3000.0, 13.0, 1.6, 98.0),
        )
    ]

    summary = mammotion_services._summarize_in_window_telemetry(  # noqa: SLF001
        samples,
        window_duration_ms=4000,
    )

    assert summary["fresh_report_arrival_count"] == 3
    assert summary["fresh_position_arrival_count"] == 3
    assert summary["max_position_arrival_gap_ms"] == 1100.0
    assert summary["toward_change_count"] == 3
    assert summary["toward_changed_before_stop"] is True


@pytest.mark.asyncio
async def test_forward_two_pulse_latency_test_defaults_to_dry_run() -> None:
    """Two-pulse latency test default sends no raw movement commands."""
    coordinator = _pulse_coordinator()

    result = await _forward_two_pulse_latency_test(coordinator)

    assert result["service"] == "forward_two_pulse_latency_test"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["reason"] == "dry_run"
    assert result["command_not_sent"] == {
        "manager_method": "send_command_with_args",
        "device_name": "Luba-Test",
        "command": "send_movement",
        "prefer_ble": True,
        "kwargs": {"linear_speed": 200, "angular_speed": 0},
    }
    assert len(result["commands"]) == 2
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_position_feedback_diagnostic_defaults_to_dry_run() -> None:
    """Position feedback diagnostic default captures sources but sends nothing."""
    coordinator = _pulse_coordinator()

    result = await _position_feedback_diagnostic(coordinator)

    assert result["service"] == "position_feedback_diagnostic"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["reason"] == "dry_run"
    assert result["snapshots"][0]["raw_sources"]["paths"]["mowing_state.pos_x"] == 1.0
    assert result["snapshots"][0]["raw_sources"]["handle"]["active_transport"] == "ble"
    assert result["refresh_attempts"] == []
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_forward_two_pulse_latency_test_rejects_missing_confirmations() -> None:
    """Real two-pulse latency test requires explicit operator confirmations."""
    coordinator = _pulse_coordinator()

    result = await _forward_two_pulse_latency_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=False,
    )

    assert result["would_send"] is False
    assert result["reason"] == "safety_gates_failed"
    assert "operator_confirmed_clear_area" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_position_feedback_diagnostic_rejects_missing_confirmations() -> None:
    """Position feedback diagnostic requires confirmations before real pulses."""
    coordinator = _pulse_coordinator()

    result = await _position_feedback_diagnostic(
        coordinator,
        dry_run=False,
        pulse_count=1,
        confirm_blades_off=True,
        confirm_clear_area=False,
    )

    assert result["would_send"] is False
    assert result["reason"] == "safety_gates_failed"
    assert "operator_confirmed_clear_area" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vio_motion_probe_defaults_to_dry_run() -> None:
    """VIO motion probe default captures a baseline but sends nothing."""
    coordinator = _pulse_coordinator()

    result = await _vio_motion_probe(coordinator)

    assert result["service"] == "vio_motion_probe"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["reason"] == "dry_run"
    assert result["command"]["kwargs"] == {"linear_speed": 200, "angular_speed": 0}
    assert result["samples"] == []
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_vio_motion_probe_rejects_missing_confirmations() -> None:
    """Real VIO motion probe requires explicit operator confirmations."""
    coordinator = _pulse_coordinator()

    result = await _vio_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=False,
    )

    assert result["would_send"] is False
    assert result["reason"] == "safety_gates_failed"
    assert "operator_confirmed_clear_area" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_fake_pulse_clock_never_moves_the_event_loop_deadline_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A test clock that jumps a pulse must not expire real dispatch deadlines.

    The motion tests advance a fake monotonic clock by the whole pulse duration
    on every simulated sleep. asyncio computes ``wait_for`` deadlines from
    ``loop.time()``, so if that fake clock is installed globally, the production
    dispatch guard's real 4.0s write deadline and 2.0s queue-start deadline can
    expire mid-test at random -- which is exactly what made six motion tests
    fail on one CI attempt and pass on the next with identical code.
    """
    clock = {"now": 100.0}
    _patch_services_monotonic(monkeypatch, lambda: clock["now"])
    loop = asyncio.get_running_loop()
    loop_time_before = loop.time()

    clock["now"] += 60.0

    assert mammotion_services.time.monotonic() == 160.0
    assert loop.time() - loop_time_before < 1.0
    assert time.monotonic() - loop_time_before < 60.0


@pytest.mark.asyncio
async def test_vio_motion_probe_drives_samples_vio_and_always_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real VIO probe sends one continuous command, samples VIO, and always stops."""
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    original_snapshot = mammotion_services._custom_path_telemetry_snapshot  # noqa: SLF001

    def fake_snapshot(
        coordinator_arg: MammotionReportUpdateCoordinator,
    ) -> dict:
        telemetry = original_snapshot(coordinator_arg)
        # Simulate forward motion: map-local y drifts as the clock advances.
        moved = (clock["now"] - 100.0) * 0.05
        telemetry["position"]["y"] = float(telemetry["position"]["y"]) - moved
        return telemetry

    async def fake_get_reports(count: int = 5) -> None:
        # VIO initializes once the mower has been moving for a moment.
        if clock["now"] >= 101.0:
            coordinator.data.report_data.vision_info = SimpleNamespace(
                heading=42.0,
                vio_state=2,
            )

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        mammotion_services,
        "_custom_path_telemetry_snapshot",
        fake_snapshot,
    )
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        drive_seconds=3.0,
        sample_interval_seconds=1.0,
        post_stop_samples=1,
    )

    # A single continuous velocity command, not one command per sample.
    handle = coordinator.manager.mower(coordinator.device_name)
    assert handle._send_marked.await_count == 2  # noqa: SLF001
    # The explicit stop is mandatory even on the happy path.
    assert handle.commands.send_movement.call_args_list[-1].kwargs == {
        "linear_speed": 0,
        "angular_speed": 0,
    }
    assert result["command_ok"] is True
    assert result["reason"] == "vio_initialized_during_motion"
    assert result["verdict"]["motion_confirmed"] is True
    assert result["verdict"]["vio_activated_while_moving"] is True
    assert 42.0 in result["verdict"]["heading_series"]
    assert result["samples"]


@pytest.mark.asyncio
async def test_vio_motion_probe_reports_settled_post_stop_displacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Motion that only lands post-stop is still confirmed, not mislabelled no-motion.

    The position feed lags ~4s: during-drive samples stay frozen and the real move
    only registers after the stop (live 2026-07-15, a 4in 6s pulse). The verdict
    must read the post-stop samples for displacement + motion_confirmed.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    clock = {"now": 100.0}
    phase = {"stopped": False}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    original_snapshot = mammotion_services._custom_path_telemetry_snapshot  # noqa: SLF001

    def fake_snapshot(
        coordinator_arg: MammotionReportUpdateCoordinator,
    ) -> dict:
        telemetry = original_snapshot(coordinator_arg)
        # Frozen while driving; the whole ~10cm move only appears after the stop.
        if phase["stopped"]:
            telemetry["position"]["y"] = float(telemetry["position"]["y"]) - 0.10
        return telemetry

    async def fake_stop(_coordinator: object) -> dict:
        phase["stopped"] = True
        return {"movement_ok": True}

    async def fake_get_reports(count: int = 5) -> None:
        return None  # VIO stays cold the whole time

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        mammotion_services, "_custom_path_telemetry_snapshot", fake_snapshot
    )
    monkeypatch.setattr(mammotion_services, "_stop_manual_motion_confirmed", fake_stop)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        drive_seconds=2.0,
        sample_interval_seconds=1.0,
        post_stop_samples=2,
    )

    # During-drive samples never registered motion...
    assert all(not sample["moving"] for sample in result["samples"])
    # ...but the settled post-stop position is used for the verdict.
    assert result["displacement_source"] == "post_stop"
    assert result["final_displacement_m"] == pytest.approx(0.10, abs=0.02)
    assert result["verdict"]["motion_confirmed"] is True
    assert result["reason"] != "no_motion_detected"


@pytest.mark.asyncio
async def test_vio_motion_probe_active_vio_lagged_motion_not_mislabelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VIO active during a lagged pulse is NOT reported as 'never initialized'.

    Regression: motion_confirmed reads post_stop (where the lagged move lands), so
    vio_activated_while_moving must judge VIO over the drive window rather than
    require the frozen per-sample `moving` flag -- otherwise a VIO-active pulse
    whose motion only registers after the stop is mislabelled
    vio_never_initialized_despite_motion.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=42.0, vio_state=2
    )
    clock = {"now": 100.0}
    phase = {"stopped": False}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    original_snapshot = mammotion_services._custom_path_telemetry_snapshot  # noqa: SLF001

    def fake_snapshot(
        coordinator_arg: MammotionReportUpdateCoordinator,
    ) -> dict:
        telemetry = original_snapshot(coordinator_arg)
        # Frozen during the drive; the whole move lands only after the stop.
        if phase["stopped"]:
            telemetry["position"]["y"] = float(telemetry["position"]["y"]) - 0.10
        return telemetry

    async def fake_stop(_coordinator: object) -> dict:
        phase["stopped"] = True
        return {"movement_ok": True}

    async def fake_get_reports(count: int = 5) -> None:
        return None  # VIO stays active (state 2) the whole time

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        mammotion_services, "_custom_path_telemetry_snapshot", fake_snapshot
    )
    monkeypatch.setattr(mammotion_services, "_stop_manual_motion_confirmed", fake_stop)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        drive_seconds=2.0,
        sample_interval_seconds=1.0,
        post_stop_samples=2,
    )

    # Frozen feed during the drive -> no drive sample registered `moving`...
    assert all(not sample["moving"] for sample in result["samples"])
    # ...but VIO was active throughout, so the verdict credits it instead of
    # claiming VIO never initialized.
    assert result["verdict"]["motion_confirmed"] is True
    assert result["verdict"]["vio_activated_while_moving"] is True
    assert result["reason"] == "vio_initialized_during_motion"


@pytest.mark.asyncio
async def test_vio_motion_probe_static_reports_no_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mower that never moves (drive or post-stop) reports no motion, ~0 displacement."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    async def fake_get_reports(count: int = 5) -> None:
        return None

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        drive_seconds=2.0,
        sample_interval_seconds=1.0,
        post_stop_samples=2,
    )

    assert result["verdict"]["motion_confirmed"] is False
    assert result["reason"] == "no_motion_detected"
    assert result["final_displacement_m"] == pytest.approx(0.0, abs=1e-6)


def test_position_source_comparison_reports_both_sources_and_agreement() -> None:
    """Both position sources + RTK quality are captured, with their divergence."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    # report_data.locations[0] is scaled by 1/10000; place it 0.2m off mowing_state.
    coordinator.data.report_data.locations = [
        SimpleNamespace(
            real_pos_x=12000, real_pos_y=10000, real_toward=0, pos_type=1, bol_hash=123
        )
    ]

    present = _position_source_comparison(coordinator)
    assert present["locations_xy"] == pytest.approx((1.2, 1.0))
    assert present["locations_stale_zero"] is False
    assert present["mowing_state_xy"] == pytest.approx((1.0, 1.0))
    assert present["agreement_m"] == pytest.approx(0.2, abs=1e-4)
    assert present["rtk_status"] == 4
    assert present["pos_level"] == 0
    assert present["pos_type"] == 1

    # With no report_data.locations, the location source degrades to None.
    coordinator.data.report_data.locations = []
    missing = _position_source_comparison(coordinator)
    assert missing["locations_xy"] is None
    assert missing["mowing_state_xy"] == pytest.approx((1.0, 1.0))
    assert missing["agreement_m"] is None

    # The post-restart stale (0,0)/AREA_OUT pose is filtered out so it can't
    # masquerade as a huge source divergence in agreement_m.
    coordinator.data.report_data.locations = [
        SimpleNamespace(
            real_pos_x=0, real_pos_y=0, real_toward=0, pos_type=0, bol_hash=0
        )
    ]
    stale = _position_source_comparison(coordinator)
    assert stale["locations_xy"] is None
    assert stale["locations_stale_zero"] is True
    assert stale["agreement_m"] is None


@pytest.mark.asyncio
async def test_forward_two_pulse_latency_test_sends_pulses_and_detects_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real two-pulse latency test sends exactly two forward pulses."""
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    original_snapshot = mammotion_services._custom_path_telemetry_snapshot  # noqa: SLF001

    def fake_snapshot(
        coordinator_arg: MammotionReportUpdateCoordinator,
    ) -> dict:
        telemetry = original_snapshot(coordinator_arg)
        if clock["now"] >= 112.0:
            telemetry["position"]["y"] = float(telemetry["position"]["y"]) - 0.02
        return telemetry

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        mammotion_services,
        "_custom_path_telemetry_snapshot",
        fake_snapshot,
    )

    result = await _forward_two_pulse_latency_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        pulse_count=3,
        telemetry_timeout_seconds=10,
        telemetry_sample_interval_seconds=1,
    )

    assert result["reason"] == "telemetry_position_change_detected"
    assert len(result["commands"]) == 3
    assert result["commands"][0]["kwargs"] == {
        "linear_speed": 200,
        "angular_speed": 0,
    }
    assert result["telemetry"]["first_position_change_after_command_1_seconds"] == 12.0
    assert result["telemetry"]["first_position_change_after_command_2_seconds"] == 7.0
    assert result["telemetry"][
        "first_position_change_after_final_command_seconds"
    ] == pytest.approx(2.0)
    assert result["telemetry"]["final_delta"]["distance"] == pytest.approx(0.02)
    assert coordinator.manager.send_command_with_args.await_count == 3


@pytest.mark.asyncio
async def test_position_feedback_diagnostic_runs_refresh_attempts_and_detects_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Position feedback diagnostic compares all snapshots after refresh paths."""
    coordinator = _pulse_coordinator()

    async def fake_sleep(_delay: float) -> None:
        return None

    async def mutate_report_snapshot() -> None:
        coordinator.data.mowing_state.pos_y = 1.25

    coordinator.async_request_report_snapshot.side_effect = mutate_report_snapshot
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)

    result = await _position_feedback_diagnostic(
        coordinator,
        dry_run=False,
        pulse_count=0,
        refresh_wait_seconds=0.1,
    )

    assert result["reason"] == "position_source_changed"
    assert result["position_source_changed"] is True
    assert "telemetry.position" in result["changed_sources"]
    assert "raw_sources.paths" in result["changed_sources"]
    assert "telemetry.position" in result["position_changed_sources"]
    assert result["metadata_changed_sources"] == []
    assert [attempt["name"] for attempt in result["refresh_attempts"]] == [
        "request_report_snapshot",
        "request_reports_count_5",
        "start_report_stream",
        "request_iot_sync_one_shot",
        "request_iot_sync_continuous_window",
        "ensure_fresh_state_forced",
        "ble_sync_type_3",
        "ha_request_refresh",
    ]
    assert all(attempt["ok"] for attempt in result["refresh_attempts"])
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_get_reports.assert_awaited_once_with(count=5)
    coordinator.async_start_report_stream.assert_awaited_once_with(duration_ms=60_000)
    coordinator.manager.request_iot_sync.assert_awaited_once_with("Luba-Test")
    coordinator.manager.request_iot_sync_continuous.assert_awaited_once_with(
        "Luba-Test",
        period=1000,
        no_change_period=4000,
    )
    coordinator.manager.request_iot_sync_continuous_stop.assert_awaited_once_with(
        "Luba-Test"
    )
    coordinator.manager.ensure_fresh_state.assert_awaited_once_with(
        "Luba-Test",
        max_age_s=0.0,
    )
    coordinator.async_send_command.assert_awaited_once_with(
        "send_todev_ble_sync",
        prefer_ble=True,
        sync_type=3,
    )
    coordinator.async_request_refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_position_feedback_diagnostic_handle_only_change_is_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handle timestamp changes are metadata, not proof of position feedback."""
    coordinator = _pulse_coordinator()

    async def fake_sleep(_delay: float) -> None:
        return None

    calls = {"count": 0}

    def mower(_device_name: str) -> SimpleNamespace:
        calls["count"] += 1
        return SimpleNamespace(
            last_report_at=float(calls["count"]),
            availability=SimpleNamespace(
                mqtt_reported_offline=False,
            ),
            get_transport=lambda _transport_type: SimpleNamespace(
                _connect_cooldown_until=0.0
            ),
            active_transport=lambda: "ble",
        )

    coordinator.manager.mower = mower
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)

    result = await _position_feedback_diagnostic(
        coordinator,
        dry_run=False,
        pulse_count=0,
        refresh_wait_seconds=0.1,
    )

    assert result["reason"] == "metadata_source_changed"
    assert result["position_source_changed"] is False
    assert result["position_changed_sources"] == []
    assert result["metadata_changed_sources"] == ["raw_sources.handle"]


@pytest.mark.asyncio
async def test_position_feedback_diagnostic_sends_optional_pulses() -> None:
    """Position feedback diagnostic can send a bounded pulse burst when approved."""
    coordinator = _pulse_coordinator()

    result = await _position_feedback_diagnostic(
        coordinator,
        dry_run=False,
        pulse_count=1,
        refresh_wait_seconds=0,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["commands"][0]["ok"] is True
    assert result["commands"][0]["kwargs"] == {
        "linear_speed": 200,
        "angular_speed": 0,
    }
    coordinator.manager.send_command_with_args.assert_awaited_once_with(
        "Luba-Test",
        "send_movement",
        prefer_ble=True,
        linear_speed=200,
        angular_speed=0,
    )


@pytest.mark.asyncio
async def test_raw_pymammotion_motion_probe_rejects_missing_confirmations() -> None:
    """Real raw probe rejects missing operator confirmations before command."""
    coordinator = _pulse_coordinator()

    result = await _raw_pymammotion_motion_probe(
        coordinator,
        dry_run=False,
        sample_delays=(),
    )

    assert result["would_send"] is False
    assert result["reason"] == "safety_gates_failed"
    assert result["blockers"] == [
        "operator_confirmed_blades_off",
        "operator_confirmed_clear_area",
    ]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_motion_probe_rejects_unsafe_blades() -> None:
    """Real raw probe rejects unsafe blade telemetry before command."""
    coordinator = _pulse_coordinator(blade_state=1, cutter_rpm=0)

    result = await _raw_pymammotion_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(),
    )

    assert result["would_send"] is False
    assert result["blockers"] == ["mower_reports_blades_off"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_motion_probe_sends_raw_movement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw send_movement passes integer speeds through to pymammotion."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_motion_probe(
        coordinator,
        command="send_movement",
        linear_speed=-400,
        angular_speed=180,
        prefer_ble=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["command_result"]["ok"] is True
    coordinator.manager.send_command_with_args.assert_awaited_once_with(
        "Luba-Test",
        "send_movement",
        prefer_ble=True,
        linear_speed=-400,
        angular_speed=180,
    )


@pytest.mark.asyncio
async def test_raw_pymammotion_motion_probe_sends_wrapper_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrapper commands pass through to pymammotion without HA motion wrappers."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_motion_probe(
        coordinator,
        command="move_left",
        speed=0.4,
        prefer_ble=False,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["command_result"]["ok"] is True
    coordinator.manager.send_command_with_args.assert_awaited_once_with(
        "Luba-Test",
        "move_left",
        prefer_ble=False,
        angular=0.4,
    )


@pytest.mark.asyncio
async def test_raw_pymammotion_motion_probe_reports_telemetry_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw probe reports movement interpretation from sampled telemetry."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        coordinator.data.mowing_state.pos_x = 1.0
        coordinator.data.mowing_state.pos_y = 0.5

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["motion_interpretation"]["status"] == "translation_detected"
    assert result["motion_interpretation"]["delta"]["distance"] == pytest.approx(0.5)
    assert result["motion_interpretation"]["movement_heading_degrees"] == 270.0


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_dry_run_negative_y() -> None:
    """Negative-Y segment selects positive raw linear speed and sends nothing."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.7}],
        sample_delays=(),
    )

    assert result["service"] == "raw_pymammotion_execute_segment"
    assert result["dry_run"] is True
    assert result["stop_reason"] == "dry_run"
    assert result["would_send"] is False
    assert result["selected_axis"] == "map_y"
    assert result["initial_command_selection"]["linear_speed"] == 400
    assert result["command_not_sent"]["kwargs"] == {
        "linear_speed": 400,
        "angular_speed": 0,
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_dry_run_positive_y() -> None:
    """Positive-Y segment selects negative raw linear speed."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 1.3}],
        sample_delays=(),
    )

    assert result["stop_reason"] == "dry_run"
    assert result["initial_command_selection"]["linear_speed"] == -400


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_uses_slow_speed_near_target() -> None:
    """Remaining Y distance below threshold selects slow raw speed."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.88}],
        sample_delays=(),
    )

    assert result["stop_reason"] == "dry_run"
    assert result["initial_command_selection"]["linear_speed"] == 200
    assert result["initial_command_selection"]["speed_tier"] == "slow"


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_rejects_lateral_segment() -> None:
    """Part 1 rejects segments that need unproven lateral/turning motion."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.5, "y": 0.95}],
        sample_delays=(),
    )

    assert result["stop_reason"] == "segment_requires_lateral_or_turning_motion"
    assert result["lateral_diagnostic"]["passed"] is False
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_rejects_missing_confirmations() -> None:
    """Real raw segment rejects missing operator confirmations."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.7}],
        dry_run=False,
        sample_delays=(),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert result["blockers"] == [
        "operator_confirmed_blades_off",
        "operator_confirmed_clear_area",
    ]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_sends_one_raw_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real raw segment sends one send_movement command and accepts progress."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        coordinator.data.mowing_state.pos_y = 0.9

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.9}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 1
    assert result["stop_reason"] == "target_reached"
    assert result["completion_status"]["complete"] is True
    assert result["progress_diagnostics"][0]["passed"] is True
    coordinator.manager.send_command_with_args.assert_awaited_once_with(
        "Luba-Test",
        "send_movement",
        prefer_ble=True,
        linear_speed=200,
        angular_speed=0,
    )


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_stops_on_no_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real raw segment stops when delayed telemetry shows no target progress."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.7}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 1
    assert result["stop_reason"] == "no_target_progress"
    assert result["progress_diagnostics"][0]["status"] == "no_path_progress"
    # With the explicit software stop in place this executor now also runs the
    # position-settle poll; a no-op pulse never registers movement.
    assert result["command_results"][0]["position_settled"] is False
    assert result["command_results"][0]["position_moved"] is False
    assert "position_source_comparison" in result["command_results"][0]


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_stops_after_max_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real raw segment stops after capped commands when progress is insufficient."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def advance_on_pulse(*_args: object, **_kwargs: object) -> None:
        # Each real pulse nudges the mower ~1.5cm toward the target -- enough to
        # pass min_progress but never reach it. Tie movement to the pulse itself,
        # not to asyncio.sleep count (the settle poll adds sleep calls).
        coordinator.data.mowing_state.pos_y = round(
            coordinator.data.mowing_state.pos_y - 0.015, 4
        )

    coordinator.manager.send_command_with_args.side_effect = advance_on_pulse

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.5}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_commands=2,
        min_progress_distance=0.01,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 2
    assert result["stop_reason"] == "max_commands_reached"
    assert result["progress_diagnostics"][-1]["passed"] is True


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_sends_explicit_stop_after_pulse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each real raw-segment pulse is followed by an explicit software stop.

    This executor used to rely on firmware auto-stop only (the reason the
    position-settle poll was reverted from it); it now mirrors the vector
    executor: bounded pulse -> stop primitive -> settle poll.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.7}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        prefer_ble=True,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 1
    coordinator.async_stop_manual_motion.assert_awaited_once()
    command_result = result["command_results"][0]
    assert command_result["stop_result"]["ok"] is True
    assert command_result["position_settled"] is False


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_aborts_when_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undeliverable stop aborts the raw segment run immediately."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.async_stop_manual_motion.side_effect = RuntimeError("stop write failed")

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.7}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_commands=3,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "stop_failed_aborting"
    assert result["commands_sent"] == 1
    assert result["command_results"][0]["stop_result"]["ok"] is False
    coordinator.manager.send_command_with_args.assert_awaited_once()


@pytest.mark.asyncio
async def test_raw_motion_readiness_test_dry_run_selects_expected_commands() -> None:
    """Readiness dry-run runs all non-moving phases and selects expected commands."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_motion_readiness_test(coordinator, sample_delays=())

    assert result["ready_for_vector_segment"] is True
    assert result["ready_for_multi_point"] is False
    assert result["linear_y_ready"] is True
    assert result["turn_to_heading_ready"] is True
    assert result["real_steps_run"] == 0
    assert result["failed_phase"] is None
    phase_by_name = {phase["name"]: phase for phase in result["phases"]}
    assert list(phase_by_name) == [
        "safety_snapshot",
        "dry_run_negative_y_segment",
        "dry_run_positive_y_segment",
        "dry_run_positive_turn_to_heading",
        "dry_run_negative_turn_to_heading",
    ]
    assert phase_by_name["dry_run_negative_y_segment"]["result"]["command_not_sent"][
        "kwargs"
    ] == {"linear_speed": 200, "angular_speed": 0}
    assert phase_by_name["dry_run_positive_y_segment"]["result"]["command_not_sent"][
        "kwargs"
    ] == {"linear_speed": -200, "angular_speed": 0}
    assert phase_by_name["dry_run_positive_turn_to_heading"]["result"][
        "command_not_sent"
    ]["kwargs"] == {"linear_speed": 0, "angular_speed": 180}
    assert phase_by_name["dry_run_negative_turn_to_heading"]["result"][
        "command_not_sent"
    ]["kwargs"] == {"linear_speed": 0, "angular_speed": -180}
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_motion_readiness_test_fails_on_unsafe_snapshot() -> None:
    """Readiness stops immediately on unsafe runtime state."""
    coordinator = _pulse_coordinator(blade_state=1, position=(1.0, 1.0, 0.0))

    result = await _raw_motion_readiness_test(coordinator, sample_delays=())

    assert result["ready_for_vector_segment"] is False
    assert result["failed_phase"] == "safety_snapshot"
    assert result["blockers"] == ["blade_reported_on"]
    assert [phase["name"] for phase in result["phases"]] == ["safety_snapshot"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_motion_readiness_test_real_rejects_missing_confirmations() -> None:
    """Real readiness rejects real phases without operator confirmations."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_motion_readiness_test(
        coordinator,
        dry_run=False,
        max_real_steps=1,
        sample_delays=(),
    )

    assert result["failed_phase"] == "real_preflight"
    assert result["blockers"] == [
        "operator_confirmed_blades_off",
        "operator_confirmed_clear_area",
    ]
    assert result["real_steps_run"] == 0
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_motion_readiness_test_max_real_steps_limits_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness only runs the requested number of real phases."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    headings = [4.0, 8.0, 4.0, 0.0]

    async def no_sleep(_: float) -> None:
        if headings:
            coordinator.data.mowing_state.toward = headings.pop(0)

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_motion_readiness_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_real_steps=2,
        sample_delays=(0,),
    )

    assert result["ready_for_vector_segment"] is True
    assert result["real_steps_run"] == 2
    assert [phase["name"] for phase in result["phases"]][-2:] == [
        "real_positive_turn_to_heading",
        "real_negative_turn_to_heading",
    ]
    assert coordinator.manager.send_command_with_args.await_count == 2


@pytest.mark.asyncio
async def test_raw_motion_readiness_test_stops_on_first_failed_real_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness stops on the first failed real phase."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_motion_readiness_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_real_steps=4,
        sample_delays=(0,),
    )

    assert result["ready_for_vector_segment"] is False
    assert result["failed_phase"] == "real_positive_turn_to_heading"
    assert result["real_steps_run"] == 1
    assert [phase["name"] for phase in result["phases"]][-1] == (
        "real_positive_turn_to_heading"
    )
    assert coordinator.manager.send_command_with_args.await_count == 1


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_vector_segment_dry_run_with_zero_offset() -> (
    None
):
    """Vector dry-run can use an explicit zero heading offset."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        sample_delays=(0,),
    )

    assert result["service"] == "raw_pymammotion_execute_vector_segment"
    assert result["dry_run"] is True
    assert result["stop_reason"] == "dry_run"
    assert result["target_map_heading_degrees"] == 0.0
    assert result["target_reported_heading_degrees"] == 0.0
    assert result["target_heading_degrees"] == 0.0
    assert result["ready_for_multi_point"] is False
    assert [phase["name"] for phase in result["phases"]] == [
        "turn_to_target_heading",
        "linear_forward_to_target",
    ]
    assert result["command_not_sent"]["kwargs"] == {
        "linear_speed": 200,
        "angular_speed": 0,
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_vector_segment_dry_run_applies_offset() -> None:
    """Vector dry-run converts map target heading into reported mower heading."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=116.5,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "dry_run"
    assert result["target_map_heading_degrees"] == 0.0
    assert result["target_reported_heading_degrees"] == pytest.approx(243.5)
    assert result["heading_calibration"] == {
        "formula": (
            "target_reported_heading = "
            "target_map_heading - calibrated_forward_heading_offset"
        ),
        "target_map_heading_degrees": 0.0,
        "calibrated_forward_heading_offset_degrees": 116.5,
        "target_reported_heading_degrees": pytest.approx(243.5),
    }
    turn_phase = result["phases"][0]["result"]
    assert turn_phase["command_not_sent"]["kwargs"] == {
        "linear_speed": 0,
        "angular_speed": -180,
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_vector_segment_rejects_missing_confirmations() -> (
    None
):
    """Real vector execution requires explicit operator confirmations."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=False,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "operator_confirmed_clear_area" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_vector_segment_sends_forward_after_heading_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real vector execution sends one raw forward command after heading is aligned."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        max_turn_commands=1,
        max_linear_commands=1,
        sample_delays=(0,),
    )

    assert result["turn_commands_sent"] == 0
    assert result["linear_commands_sent"] == 1
    assert result["stop_reason"] == "no_target_progress"
    coordinator.manager.send_command_with_args.assert_awaited_once_with(
        "Luba-Test",
        "send_movement",
        prefer_ble=True,
        linear_speed=200,
        angular_speed=0,
    )


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_vector_segment_sends_explicit_stop_after_pulse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each real linear pulse is followed by an explicit stop, not left to firmware.

    ``send_movement`` is a continuous-velocity command with no protocol-level
    duration bound -- live testing showed a single "pulse" travel ~7x the
    expected distance because nothing ever called the stop primitive.
    Regression guard: assert async_stop_manual_motion fires after the pulse.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        max_turn_commands=1,
        max_linear_commands=1,
        prefer_ble=True,
        sample_delays=(0,),
    )

    coordinator.async_stop_manual_motion.assert_awaited_once()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_vector_segment_halts_before_linear_on_incomplete_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segment requiring a large turn halts in the turn phase; no linear command is sent.

    Unlike the pure-linear `_raw_pymammotion_execute_segment`, the vector segment has no
    explicit pre-flight rejection for out-of-calibrated-window turns -- it attempts the turn
    and relies on the turn-command budget (`max_turn_commands`) and heading-progress checks
    to halt safely before any forward motion is attempted. This is the actual safety
    mechanism the multi-segment chain (used by the multi-waypoint click/go path builder)
    relies on for segments requiring more than a small heading correction.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 2.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        max_turn_commands=1,
        max_linear_commands=1,
        sample_delays=(0,),
    )

    assert result["target_map_heading_degrees"] == pytest.approx(90.0)
    assert result["stop_reason"] == "turn_phase_incomplete"
    assert result["turn_commands_sent"] == 1
    assert result["linear_commands_sent"] == 0


@pytest.mark.asyncio
async def test_vector_segment_loop_to_tolerance_stops_on_consecutive_no_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loop-to-tolerance mode keeps pulsing past the legacy budget, then stops on stall.

    With ``max_linear_pulse_ceiling`` set the linear phase no longer quits at the tiny
    ``max_linear_commands`` budget (max 3); it pulses until the waypoint is reached or
    ``max_no_progress_pulses`` consecutive pulses make no target-directed progress. Here
    the mocked mower never moves, so it must stop after exactly that many pulses.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    # Target due +X with a zero offset and toward 0.0 => turn phase needs 0 commands,
    # so the linear loop runs; a stationary mower makes no progress.
    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        max_linear_commands=1,
        max_linear_pulse_ceiling=20,
        max_no_progress_pulses=3,
        sample_delays=(0,),
    )

    assert result["linear_execution_mode"] == "loop_to_tolerance"
    assert result["turn_commands_sent"] == 0
    assert result["stop_reason"] == "no_target_progress"
    # Pulsed 3 times (max_no_progress_pulses) -- well past the legacy budget of 1.
    assert result["linear_commands_sent"] == 3
    # Each linear pulse ran the position-settle poll; a stationary mower never
    # registers motion, so every pulse records it did not settle/move.
    settle_flags = [
        (c.get("position_moved"), c.get("position_settled"))
        for c in result["command_results"]
        if "position_settled" in c
    ]
    assert settle_flags == [(False, False)] * 3


@pytest.mark.asyncio
async def test_vector_segment_vio_real_run_blocked_when_feed_degraded() -> None:
    """VIO real run is refused when vio_state reads active but the feed is blind."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=2, track_feature_num=0, brightness=10
    )

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="vio",
        vio_heading_offset_degrees=0.0,
        sample_delays=(),
    )

    assert "vio_feed_live" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"
    assert result["vio"]["initial_vio_feed"]["live"] is False
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_multi_segment_vio_real_run_blocked_when_feed_degraded() -> None:
    """Multi-segment refuses a real VIO run when the feed is blind at chain entry."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=2, track_feature_num=0, brightness=10
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}, {"x": 1.2, "y": 1.1}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="vio",
        vio_heading_offset_degrees=0.0,
        sample_delays=(),
    )

    assert "vio_feed_live" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"
    assert result["initial_vio_feed"]["live"] is False
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_multi_segment_dry_run_chains_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-segment dry-run calls the vector primitive for each segment only."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    calls: list[tuple[list[dict[str, float]], bool]] = []

    async def fake_vector(
        coordinator_arg: MammotionReportUpdateCoordinator,
        points: list[dict[str, float]],
        **kwargs: object,
    ) -> dict:
        assert coordinator_arg is coordinator
        calls.append((points, bool(kwargs["dry_run"])))
        return {
            "valid": True,
            "stop_reason": "dry_run",
            "blockers": [],
            "phases": [{"passed": True}, {"passed": True}],
            "final_telemetry": _custom_path_telemetry_snapshot(coordinator),
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
            {"x": 1.1, "y": 1.0},
            {"x": 1.2, "y": 1.1},
        ],
        sample_delays=(0,),
    )

    assert result["service"] == "raw_pymammotion_execute_multi_segment"
    assert result["dry_run"] is True
    assert result["stop_reason"] == "dry_run"
    assert result["ready_for_multi_segment"] is True
    assert result["ready_for_multi_point"] is False
    assert result["segments_executed"] == 2
    assert calls == [
        ([{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}], True),
        ([{"x": 1.1, "y": 1.0}, {"x": 1.2, "y": 1.1}], True),
    ]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_multi_segment_forwards_approach_and_turn_rate_to_every_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card profile constants reach the vector executor instead of being ignored."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    forwarded: list[tuple[float, float]] = []

    async def fake_vector(
        coordinator_arg: MammotionReportUpdateCoordinator,
        points: list[dict[str, float]],
        **kwargs: object,
    ) -> dict:
        forwarded.append(
            (
                float(kwargs["final_approach_metres_per_pulse"]),
                float(kwargs["turn_degrees_per_second"]),
            )
        )
        return {
            "valid": True,
            "stop_reason": "dry_run",
            "blockers": [],
            "phases": [{"passed": True}, {"passed": True}],
            "final_telemetry": _custom_path_telemetry_snapshot(coordinator_arg),
            "progress_diagnostics": [],
        }

    monkeypatch.setattr(
        mammotion_services, "_raw_pymammotion_execute_vector_segment", fake_vector
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [
            {"x": 1.0, "y": 1.0},
            {"x": 1.1, "y": 1.0},
            {"x": 1.2, "y": 1.1},
        ],
        final_approach_metres_per_pulse=1.23,
        turn_degrees_per_second=41.5,
        sample_delays=(0,),
    )

    assert result["final_approach_metres_per_pulse"] == 1.23
    assert result["turn_degrees_per_second"] == 41.5
    assert forwarded == [(1.23, 41.5), (1.23, 41.5)]


@pytest.mark.asyncio
async def test_vector_segment_vio_dry_run_plans_calibration_and_turn() -> None:
    """Default VIO turn mode dry-runs with a planned calibration drive + turn."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        sample_delays=(0,),
    )

    assert result["turn_mode"] == "vio"
    assert result["stop_reason"] == "dry_run"
    assert [phase["name"] for phase in result["phases"]] == [
        "turn_to_target_heading",
        "linear_forward_to_target",
    ]
    turn_phase = result["phases"][0]
    assert turn_phase["turn_mode"] == "vio"
    assert turn_phase["passed"] is True
    planned = turn_phase["result"]["planned"]
    assert planned["turn_primitive"] == "vio_turn_to_heading"
    assert planned["angular_speed"] == 500
    assert planned["calibration_drive"]["kwargs"] == {
        "linear_speed": 400,
        "angular_speed": 0,
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vector_segment_vio_real_blocked_when_vio_cold() -> None:
    """Real VIO-mode segment refuses to move unless VIO is actively tracking."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "vio_active" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vector_segment_vio_cold_start_allowed_when_scene_bright(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold VIO + bright scene: the calibration drive doubles as the warm-up."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    # vio_state 0 (cold) but brightness 100 -> "Light" scene.
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=0, brightness=100
    )

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        # The drive woke VIO and calibrated.
        return {
            "passed": True,
            "reason": "calibrated",
            "offset_degrees": -90.0,
            "map_motion_heading_degrees": 280.0,
            "vision_heading": 10.0,
            "vio_state": 2,
            "distance_m": 0.08,
            "pulses_sent": 1,
            "command_results": [],
        }

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 2,
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
        sample_delays=(0,),
    )

    # Not blocked: the run proceeded through calibration and the turn.
    assert "vio_active" not in result["blockers"]
    assert result["calibration_commands_sent"] == 1
    assert result["phases"][0]["passed"] is True


@pytest.mark.asyncio
async def test_vector_segment_vio_cold_start_blocked_when_offset_skips_warmup() -> None:
    """Cold VIO stays blocked when a provided offset would skip the warm-up drive."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=0, brightness=100
    )

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        vio_heading_offset_degrees=100.0,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "vio_active" in result["blockers"]


@pytest.mark.asyncio
async def test_vector_segment_reports_stale_stream_after_feed_dies_mid_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feed that works then freezes aborts telemetry_stream_stale, not no_progress.

    Replays the 2026-07-19 card run: linear pulses advanced normally, then the
    position feed froze bit-identical for three pulses. The run aborted
    no_target_progress -- but the mower had actually driven 25.4cm during those
    pulses (~8.5cm each, the proven step), which only surfaced after the feed
    caught up post-run. The executor spent ~30s issuing motion against a stale
    coordinate, so it must stop and say so rather than blame progress.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )
    state = {"y": 1.0, "frozen": False, "pulses": 0}

    async def no_sleep(_: float) -> None:
        return None

    async def advancing_then_frozen(count: int = 5) -> None:
        # Feed advances ~9cm per refresh until it dies, then repeats verbatim.
        if not state["frozen"]:
            state["y"] += 0.09
            coordinator.data.mowing_state.pos_y = state["y"]

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    coordinator.async_get_reports.side_effect = advancing_then_frozen

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
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
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
            "final_vision_heading": 90.0,
            "final_heading_error_degrees": 0.5,
        }

    async def freeze_after_two(*args: object, **kwargs: object) -> None:
        state["pulses"] += 1
        if state["pulses"] >= 3:
            state["frozen"] = True

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)
    monkeypatch.setattr(mammotion_services, "_motion_open_sleep", freeze_after_two)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 4.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        # A ceiling switches the executor into loop-to-tolerance mode.
        max_linear_pulse_ceiling=12,
        vio_max_realignments=0,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "telemetry_stream_stale"
    assert "bit-identical" in result["telemetry_stream_stale_hint"]
    # It must have driven successfully first -- otherwise "the stream died" is
    # not a supportable claim.
    assert any(c.get("position_moved") for c in result["command_results"])
    assert any(c.get("position_feed_stale") for c in result["command_results"])


@pytest.mark.asyncio
async def test_vector_segment_echoes_pulse_geometry_params() -> None:
    """The vector executor reports the pulse-geometry params it was given.

    Regression for 2026-07-25: ``max_linear_pulse_ceiling`` was honoured but
    echoed as absent, so a stalled run gave no way to confirm the numbers that
    matter most. The multi-segment executor got this fixed on 2026-07-19; the
    vector one was missed. A dry run is enough -- the echo must not depend on
    the mower moving.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 4.0}],
        dry_run=True,
        max_linear_pulse_ceiling=12,
        turn_pulse_duration_ms=1500.0,
        linear_pulse_duration_ms=3500.0,
        vio_turn_max_commands=16,
        vio_angular_speed=500,
        vio_heading_offset_degrees=42.5,
    )

    assert result["max_linear_pulse_ceiling"] == 12
    assert result["turn_pulse_duration_ms"] == 1500.0
    assert result["linear_pulse_duration_ms"] == 3500.0
    assert result["vio_turn_max_commands"] == 16
    assert result["vio_angular_speed"] == 500
    assert result["vio_heading_offset_degrees"] == 42.5


def test_a_re_aim_is_skipped_only_when_it_cannot_improve_the_landing() -> None:
    """The test is the perpendicular miss, not the distance to the waypoint.

    Landing anywhere inside `waypoint_tolerance` ends the segment, so a re-aim is
    worth spending turn commands on exactly when driving straight on would MISS
    that disc. The perpendicular miss -- distance * sin(aim) -- answers that
    directly and needs no tuned constant.

    ⚠️ This replaces a wrong guard shipped in beta36, which compared distance
    alone against `tolerance / tan(trigger)` = 0.4617 m. On the 0.7 m legs
    introduced the same day that disabled re-aim for the last 66% of every leg
    however badly the mower was pointed. It suppressed corrections at 40.099 and
    77.922 deg of aim error and segment 3 landed 0.2548 m out, the worst of the
    day (docs/evidence-beta32-4segment-20260810T002506Z.json). Those errors were
    real, not bearing noise: RTK measured +40.73 and +72.92 deg independently of
    VIO.

    All four mid-drive re-aims on record, with the outcome each actually had.
    """
    tolerance = 0.15
    cases = [
        # distance, aim, suppress?, what happened
        (0.540, 23.301, False, "allowed and legitimate"),
        (0.210, 34.837, True, "allowed, then oscillated -- should have been skipped"),
        (0.3104, 40.099, False, "beta36 suppressed it and the segment missed"),
        (0.2072, 77.922, False, "beta36 suppressed it and the segment missed"),
    ]
    for distance, aim, expected, note in cases:
        assert (
            _realign_cannot_improve_the_landing(
                distance_to_target_m=distance,
                aim_error_degrees=aim,
                waypoint_tolerance=tolerance,
                metres_per_pulse=1.06,
            )
            is expected
        ), note

    # The boundary sits where the geometry puts it. beta42 MOVED it: the mower
    # drives a whole pulse rather than stopping at the closest approach, so with
    # the remaining distance `d` the landing is the chord `2*d*sin(aim/2)` and
    # the edge is at `2*asin(tolerance / (2*d))` -- 17.254 deg at d = 0.5,
    # against 17.458 for the old closest-approach rule.
    on_edge = 2.0 * math.degrees(math.asin(tolerance / (2 * 0.5)))
    assert on_edge == pytest.approx(17.2544, abs=1e-3)
    for aim, expected in ((on_edge - 0.5, True), (on_edge + 0.5, False)):
        assert (
            _realign_cannot_improve_the_landing(
                distance_to_target_m=0.5,
                aim_error_degrees=aim,
                waypoint_tolerance=tolerance,
                metres_per_pulse=1.06,
            )
            is expected
        )


@pytest.mark.parametrize(
    "override",
    [
        {"distance_to_target_m": 0.0},
        {"waypoint_tolerance": 0.0},
        # At or past 90 deg the target is abeam or behind, so driving on can only
        # make it worse; `_requires_reverse_recovery` owns that case.
        {"aim_error_degrees": 90.0},
        {"aim_error_degrees": 120.0},
    ],
)
def test_the_re_aim_guard_fails_open_on_degenerate_input(
    override: dict[str, float],
) -> None:
    """A guard that cannot judge must never suppress.

    Suppressing a re-aim is the dangerous direction: a mower that stops
    correcting its aim keeps driving. Every degenerate input resolves to "do not
    suppress" and lets the existing trigger decide, which is the pre-guard
    behaviour.
    """
    kwargs = {
        "distance_to_target_m": 0.05,
        "aim_error_degrees": 30.0,
        "waypoint_tolerance": 0.15,
        "metres_per_pulse": 1.06,
    }
    kwargs.update(override)

    assert _realign_cannot_improve_the_landing(**kwargs) is False


@pytest.mark.asyncio
async def test_vector_segment_forwards_refresh_and_rate_into_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The executor must hand its refresh cadence to the turn phase.

    Regression for 2026-07-27: the executor accepted
    ``motion_refresh_interval_ms`` and used it for linear pulses but did not
    forward it to ``_vio_turn_to_heading``, so its turns always ran single-shot
    at ~13 deg/command. A 176 deg turn then exhausted the 8-command budget live
    and the segment never reached its linear phase.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )
    turn_calls: list[dict[str, object]] = []

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
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
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        turn_calls.append(kwargs)
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
            "final_vision_heading": 90.0,
            "final_heading_error_degrees": 0.5,
        }

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_commands=1,
        vio_max_realignments=0,
        motion_refresh_interval_ms=200,
        heading_tolerance_degrees=18.0,
        turn_degrees_per_second=37.0,
        sample_delays=(0,),
    )

    assert len(turn_calls) == 1
    assert turn_calls[0]["motion_refresh_interval_ms"] == 200
    assert turn_calls[0]["turn_degrees_per_second"] == 37.0
    # The tolerance must come from the segment call, not the executor default.
    assert turn_calls[0]["heading_tolerance_degrees"] == 18.0


@pytest.mark.asyncio
async def test_vector_segment_shortens_the_final_pulse_instead_of_overshooting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replays the 2026-07-27 return run, which overshot by 0.8 m and gave up.

    That run drove 4.06 m for a 3.26 m target: with ~0.2 m to go the executor
    had no move available except another full ~1.06 m pulse, so it stepped past
    the waypoint and then failed trying to re-aim at a target now 176 deg behind
    it. Raising ``waypoint_tolerance`` does not fix that -- pulse granularity is
    the limit. The fix bounds the final pulse by confirmed refresh writes rather
    than nominal duration.

    Here the mower starts 3.0 m out and covers 1.06 m per full pulse: two full
    pulses leave 0.88 m, which receives nine refreshes rather than another full
    ten-refresh pulse.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )
    state = {"y": 1.0}
    windows: list[float] = []
    refresh_limits: list[int | None] = []

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
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
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
            "final_vision_heading": 90.0,
            "final_heading_error_degrees": 0.5,
        }

    async def fake_refresh_window(
        coordinator_arg: MammotionReportUpdateCoordinator,
        *,
        resend: object,
        duration_seconds: float,
        refresh_interval_ms: int,
        max_refresh_commands: int | None = None,
    ) -> dict:
        windows.append(duration_seconds)
        refresh_limits.append(max_refresh_commands)
        nonzero_writes = (
            11 if max_refresh_commands is None else max_refresh_commands + 1
        )
        state["y"] += (nonzero_writes / 11) * 1.06
        coordinator.data.mowing_state.pos_y = state["y"]
        return {
            "refresh_enabled": True,
            "refresh_interval_ms": refresh_interval_ms,
            "refresh_commands_sent": nonzero_writes - 1,
        }

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)
    monkeypatch.setattr(
        mammotion_services, "_motion_refresh_window", fake_refresh_window
    )

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 4.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_pulse_ceiling=12,
        vio_max_realignments=0,
        linear_pulse_duration_ms=3500.0,
        motion_refresh_interval_ms=200,
        waypoint_tolerance=0.15,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "target_reached"
    # Two full pulses, then one bounded to the 0.88 m that was left.
    assert len(windows) == 3
    assert windows == pytest.approx([3.5, 3.5, 3.5])
    assert refresh_limits == [None, None, 9]

    approaches = [
        c["final_approach"] for c in result["command_results"] if "final_approach" in c
    ]
    assert [a["applied"] for a in approaches] == [False, False, True]
    # The scale factor came from the two pulses this run actually measured.
    assert approaches[-1]["metres_per_pulse_source"] == "observed"
    assert approaches[-1]["metres_per_pulse"] == pytest.approx(1.06, abs=0.01)


@pytest.mark.asyncio
async def test_vector_segment_vio_real_calibrates_turns_then_drives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real VIO segment: calibration offset -> VIO turn on mapped heading -> linear."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        assert coordinator_arg is coordinator
        return {
            "passed": True,
            "reason": "calibrated",
            "offset_degrees": -90.0,
            "map_motion_heading_degrees": 280.0,
            "vision_heading": 10.0,
            "vio_state": 2,
            "distance_m": 0.06,
            "pulses_sent": 1,
            "command_results": [{"phase": "vio_calibration_drive", "ok": True}],
        }

    turn_calls: list[dict[str, object]] = []

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        turn_calls.append(kwargs)
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 2,
            "command_results": [],
            "final_vision_heading": 90.0,
            "final_heading_error_degrees": 0.5,
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
        sample_delays=(0,),
    )

    # Map heading to target is 0 deg; offset -90 -> target_vision_heading 90.
    assert len(turn_calls) == 1
    assert turn_calls[0]["target_vision_heading"] == pytest.approx(90.0)
    assert turn_calls[0]["angular_speed"] == 500
    assert result["vio"]["offset_degrees"] == pytest.approx(-90.0)
    assert result["vio"]["offset_source"] == "calibration_drive"
    assert result["vio"]["target_vision_heading"] == pytest.approx(90.0)
    assert result["calibration_commands_sent"] == 1
    assert result["turn_commands_sent"] == 2
    assert result["phases"][0]["passed"] is True
    # Static test position never reaches the waypoint: linear stops on progress.
    assert result["linear_commands_sent"] == 1
    assert result["stop_reason"] == "no_target_progress"


@pytest.mark.asyncio
async def test_vector_segment_realigns_after_turn_translation_before_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A translating pivot recomputes its bearing before the forward command.

    Gate 4 (2026-07-31) reached its pre-turn VIO target but drifted 14.4 cm
    during the pivot. That changed the bearing to a 30 cm waypoint enough that
    the subsequent forward pulse was 23 degrees off course. The fresh-position
    correction must happen before, not after, that linear dispatch.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)
    turn_calls: list[dict[str, object]] = []

    async def no_sleep(_: float) -> None:
        return None

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict[str, object]:
        turn_calls.append(kwargs)
        if len(turn_calls) == 1:
            # Reach the original north-facing target while drifting 20 cm east.
            coordinator.data.mowing_state.pos_x = 1.2
            coordinator.data.report_data.vision_info.heading = 90.0
            displacement = 0.2
        else:
            # The fresh bearing from (1.2, 1.0) to (1.0, 1.5) is ~111.8 deg.
            coordinator.data.report_data.vision_info.heading = float(
                kwargs["target_vision_heading"]
            )
            displacement = 0.0
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
            "final_displacement_m": displacement,
        }

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 1.5}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        vio_heading_offset_degrees=0.0,
        vio_max_realignments=1,
        max_turn_translation_distance=0.25,
        max_linear_commands=1,
        sample_delays=(0,),
    )

    assert len(turn_calls) == 2
    assert turn_calls[0]["max_displacement_m"] == pytest.approx(0.25)
    assert turn_calls[1]["target_vision_heading"] == pytest.approx(111.801, abs=0.01)
    assert turn_calls[1]["max_displacement_m"] == pytest.approx(0.25)
    alignment = result["post_turn_alignment"]
    assert alignment["correction_attempted"] is True
    assert alignment["before"]["aim_error_degrees"] == pytest.approx(21.801, abs=0.01)
    assert alignment["after"]["aim_error_degrees"] == pytest.approx(0.0)
    assert alignment["passed"] is True
    assert result["realignments"][0]["before_linear"] is True
    # The static fixture does not model forward travel, but only after alignment
    # is proven may the single linear command be attempted.
    assert result["linear_commands_sent"] == 1


@pytest.mark.asyncio
async def test_vector_segment_blocks_linear_when_post_turn_alignment_stays_bad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claimed correction that remains off-bearing must fail before driving."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)
    turn_count = 0

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict[str, object]:
        nonlocal turn_count
        turn_count += 1
        if turn_count == 1:
            coordinator.data.mowing_state.pos_x = 1.2
            coordinator.data.report_data.vision_info.heading = 90.0
            displacement = 0.2
        else:
            # Backend claims success but the live heading did not change.
            displacement = 0.0
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
            "final_displacement_m": displacement,
        }

    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 1.5}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        vio_heading_offset_degrees=0.0,
        vio_max_realignments=1,
        max_linear_commands=1,
        sample_delays=(0,),
    )

    assert turn_count == 2
    assert result["stop_reason"] == "post_turn_alignment_incomplete"
    assert result["post_turn_alignment"]["passed"] is False
    assert result["linear_commands_sent"] == 0


@pytest.mark.asyncio
async def test_vector_segment_vio_real_stops_on_failed_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed calibration drive halts the segment before any turn/linear motion."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "passed": False,
            "reason": "vio_not_active_after_drive",
            "offset_degrees": None,
            "vision_heading": 0.0,
            "vio_state": 0,
            "distance_m": 0.05,
            "pulses_sent": 2,
            "command_results": [],
        }

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "vio_calibration_failed"
    assert result["turn_commands_sent"] == 0
    assert result["linear_commands_sent"] == 0
    assert result["phases"][0]["passed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_linear_commands", "expected_realignments"), [(1, 0), (2, 1)]
)
async def test_vector_segment_vio_realigns_when_facing_drifts_off_bearing(
    monkeypatch: pytest.MonkeyPatch,
    max_linear_commands: int,
    expected_realignments: int,
) -> None:
    """Mid-drive re-aim runs only when another forward command can follow."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    # Facing estimate = vision_heading + offset = 10 + (-90) = -80 deg map,
    # bearing to target is 0 deg -> 80 deg aim error > 15 deg threshold.
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
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

    turn_calls: list[dict[str, object]] = []

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        turn_calls.append(kwargs)
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
        }

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        # 1.0 m leg, not 0.1 m: a leg shorter than
        # `waypoint_tolerance / tan(trigger)` sits entirely inside the
        # radius where `_realign_bearing_is_ill_conditioned` suppresses
        # re-aim -- rightly, since at 0.1 m against an 0.08 m tolerance the
        # mower needs to move 2 cm and its aim cannot matter. A realistic
        # leg keeps this test exercising the re-aim path it was written for.
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_commands=max_linear_commands,
        sample_delays=(0,),
    )

    # The initial turn always runs. A mid-drive re-aim runs only after the first
    # of two linear commands; after the final budget it could add drift but no
    # forward command could benefit from it.
    assert len(turn_calls) == 1 + expected_realignments
    assert len(result["realignments"]) == expected_realignments
    if expected_realignments:
        realign = result["realignments"][0]
        assert realign["stop_reason"] == "target_heading_reached"
        assert abs(realign["aim_error_degrees"]) > 15.0
    # This fixture intentionally leaves position static, so after exercising
    # the re-alignment decision the executor fails closed on no progress.
    assert result["stop_reason"] == "no_target_progress"


@pytest.mark.asyncio
async def test_mid_drive_re_aim_follows_the_effective_linear_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With loop-to-tolerance on, re-aim must not stop at `max_linear_commands`.

    The re-aim guard asks "can another forward command follow this?", and the
    answer is `effective_linear_ceiling`. Those two numbers are equal only while
    `max_linear_pulse_ceiling` is None. Enable loop-to-tolerance and the ceiling
    becomes the pulse ceiling, so a guard testing `max_linear_commands` would
    silently stop cross-track correction after that many pulses while the linear
    loop kept driving -- a mower correcting nothing for the rest of a long
    segment, which is precisely the failure mode the whole re-aim mechanism
    exists to prevent.

    Recorded as a prerequisite in docs/HANDOVER-beta31-20260809.md section 5:
    harmless while the mode is off, and to be fixed BEFORE anyone turns it on.
    This is that fix, with the mode on.

    `max_linear_commands=1` against `max_linear_pulse_ceiling=4` separates them:
    after the first linear pulse the old comparison is 1 < 1 and refuses, the
    correct one is 1 < 4 and proceeds.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
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

    turn_calls: list[dict[str, object]] = []

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        turn_calls.append(kwargs)
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
        }

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_commands=1,
        max_linear_pulse_ceiling=4,
        sample_delays=(0,),
    )

    assert result["linear_execution_mode"] == "loop_to_tolerance"
    assert result["effective_linear_ceiling"] == 4
    # The initial turn plus at least one mid-drive re-aim. Against
    # `max_linear_commands` this would be the initial turn alone.
    assert len(result["realignments"]) >= 1, (
        "re-aim stopped at max_linear_commands while loop-to-tolerance kept "
        "driving -- cross-track correction would be silently dead"
    )
    assert len(turn_calls) >= 2


@pytest.mark.asyncio
async def test_vector_segment_refuses_a_u_turn_after_passing_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forward-only segment stops instead of recovering to a target behind it."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def no_sleep(_: float) -> None:
        return None

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "passed": True,
            "reason": "calibrated",
            "offset_degrees": 110.0,
            "map_motion_heading_degrees": 120.0,
            "vision_heading": 10.0,
            "vio_state": 2,
            "distance_m": 0.06,
            "pulses_sent": 1,
            "command_results": [],
        }

    turn_calls: list[dict[str, object]] = []

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        turn_calls.append(kwargs)
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
        }

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
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
        max_linear_commands=2,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "target_requires_reverse_recovery"
    assert abs(result["reverse_recovery_guard"]["aim_error_degrees"]) == 120.0
    assert result["linear_commands_sent"] == 1
    # Only the initial heading turn ran; no U-turn recovery was dispatched.
    assert len(turn_calls) == 1


@pytest.mark.asyncio
async def test_vector_segment_stops_when_the_realign_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An off-bearing segment with no correction left stops instead of driving on."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    # Facing estimate = vision_heading + offset = 10 + (-90) = -80 deg map,
    # bearing to target is 0 deg -> 80 deg aim error: over the 15 deg realign
    # threshold but under the 90 deg reverse-recovery boundary, so this is the
    # budget decision and not the U-turn guard.
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def no_sleep(_: float) -> None:
        return None

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
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

    turn_calls: list[dict[str, object]] = []

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        turn_calls.append(kwargs)
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
        }

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        # 1.0 m leg, not 0.1 m: a leg shorter than
        # `waypoint_tolerance / tan(trigger)` sits entirely inside the
        # radius where `_realign_bearing_is_ill_conditioned` suppresses
        # re-aim -- rightly, since at 0.1 m against an 0.08 m tolerance the
        # mower needs to move 2 cm and its aim cannot matter. A realistic
        # leg keeps this test exercising the re-aim path it was written for.
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_commands=2,
        vio_max_realignments=0,
        sample_delays=(0,),
    )

    # Before this guard the exhausted budget silently skipped the correction and
    # spent the remaining forward budget driving 80 deg off the bearing.
    assert result["stop_reason"] == "vio_realign_budget_exhausted"
    assert result["linear_commands_sent"] == 1
    assert result["realignments"] == []
    # Only the initial heading turn ran; no correction was dispatched.
    assert len(turn_calls) == 1


@pytest.mark.asyncio
async def test_vio_calibration_drive_aborts_when_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed stop (e.g. BLE cooldown) aborts the drive before another pulse."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=2, brightness=100
    )

    async def no_sleep(_: float) -> None:
        return None

    async def failing_stop(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "attempted": True,
            "ok": False,
            "error": "BLEUnavailableError: cooldown",
        }

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        mammotion_services, "_manual_velocity_stop_attempt", failing_stop
    )

    result = await _vio_segment_calibration_drive(coordinator, max_pulses=3)

    assert result["passed"] is False
    assert result["reason"] == "stop_failed_aborting"
    # Only the first pulse fired; no further motion after the failed stop.
    assert result["pulses_sent"] == 1


@pytest.mark.asyncio
async def test_vio_segment_calibration_drive_computes_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The calibration drive derives offset = map motion heading - vision heading."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=0)

    async def no_sleep(_: float) -> None:
        return None

    refresh_settle_seconds: list[float] = []

    async def fake_refresh(
        coordinator_arg: MammotionReportUpdateCoordinator,
        *,
        settle_seconds: float = 2.0,
    ) -> dict:
        refresh_settle_seconds.append(settle_seconds)
        # Simulate the post-drive feedback refresh: mower moved +x/+y and the
        # motion woke VIO with a fresh body heading.
        coordinator.data.mowing_state.pos_x += 0.03
        coordinator.data.mowing_state.pos_y += 0.03
        coordinator.data.report_data.vision_info = SimpleNamespace(
            heading=15.0, vio_state=2
        )
        return {"refreshed": True}

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        mammotion_services, "_refresh_position_after_raw_motion", fake_refresh
    )

    result = await _vio_segment_calibration_drive(coordinator, max_pulses=2)

    assert result["passed"] is True
    # 6 cm minimum baseline: one 4.2 cm pulse is not enough, two (8.5 cm) are.
    assert result["pulses_sent"] == 2
    # Motion vector (+0.06, +0.06) -> 45 deg map heading; offset = 45 - 15 = 30.
    assert result["map_motion_heading_degrees"] == pytest.approx(45.0)
    assert result["offset_degrees"] == pytest.approx(30.0)
    assert refresh_settle_seconds == [0.0, 0.0]
    coordinator.async_stop_manual_motion.assert_awaited()


@pytest.mark.asyncio
async def test_vio_segment_calibration_drive_rejects_offset_on_degraded_feed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blind feed (0 features) yields no offset even though vio_state reads active."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=0)

    async def no_sleep(_: float) -> None:
        return None

    async def fake_refresh(
        coordinator_arg: MammotionReportUpdateCoordinator,
        *,
        settle_seconds: float = 2.0,
    ) -> dict:
        # Motion moved the mower, but VIO woke blind: vio_state latched active
        # with 0 tracked features, so the vision heading is untrustworthy and the
        # offset it would produce would be silently wrong.
        coordinator.data.mowing_state.pos_x += 0.03
        coordinator.data.mowing_state.pos_y += 0.03
        coordinator.data.report_data.vision_info = SimpleNamespace(
            heading=15.0, vio_state=2, track_feature_num=0, brightness=10
        )
        return {"refreshed": True}

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        mammotion_services, "_refresh_position_after_raw_motion", fake_refresh
    )

    result = await _vio_segment_calibration_drive(coordinator, max_pulses=2)

    assert result["passed"] is False
    assert result["reason"] == "vio_feed_degraded"
    assert result["offset_degrees"] is None
    assert result["vio_feed"]["live"] is False


@pytest.mark.asyncio
async def test_vio_linear_pulse_reuses_settled_position_without_sample_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real Go does not wait through samples after position already settled."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=2, track_feature_num=100, brightness=100
    )
    sleep_delays: list[float] = []
    refresh_settle_seconds: list[float] = []
    queue_settles: list[int] = []
    queue_live = {"value": True}

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    async def fake_turn(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 0,
            "command_results": [],
        }

    async def fake_refresh(
        *_args: object, settle_seconds: float = 2.0, **_kwargs: object
    ) -> dict[str, object]:
        refresh_settle_seconds.append(settle_seconds)
        return {"ok": True, "settle_seconds": settle_seconds}

    async def fake_motion_refresh(
        *_args: object, **_kwargs: object
    ) -> dict[str, object]:
        return {
            "refresh_enabled": True,
            "refresh_interval_ms": 200,
            "refresh_commands_sent": 1,
        }

    async def fake_settle(*_args: object, **_kwargs: object) -> dict[str, object]:
        coordinator.data.mowing_state.pos_x = 1.45
        telemetry = _custom_path_telemetry_snapshot(coordinator)
        return {
            "telemetry": telemetry,
            "settled": True,
            "moved": True,
            "feed_stale": False,
            "settle_polls": 2,
            "wait_seconds": 2.0,
        }

    async def fake_queue_settle(*_args: object) -> dict[str, object]:
        queue_settles.append(1)
        return {
            "live": queue_live["value"],
            "reason": None if queue_live["value"] else "command_queue_backlogged",
            "queue_depth": 0 if queue_live["value"] else 1,
        }

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading_staged", fake_turn)
    monkeypatch.setattr(
        mammotion_services, "_refresh_position_after_raw_motion", fake_refresh
    )
    monkeypatch.setattr(
        mammotion_services, "_motion_refresh_window", fake_motion_refresh
    )
    monkeypatch.setattr(mammotion_services, "_settle_linear_position_feed", fake_settle)
    monkeypatch.setattr(
        mammotion_services, "_settle_ble_command_queue", fake_queue_settle
    )

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.5, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="vio",
        vio_heading_offset_degrees=0.0,
        max_linear_commands=1,
        motion_refresh_interval_ms=200,
        sample_delays=(0, 3),
    )

    assert result["stop_reason"] == "target_reached"
    assert refresh_settle_seconds == [0.0]
    assert sleep_delays == []
    # One settle at executor entry, one after report-driven position settling.
    assert queue_settles == [1, 1]
    linear = next(
        command
        for command in result["command_results"]
        if command.get("phase") == "linear_forward_to_target"
    )
    assert linear["post_settle_feedback"] == {
        "source": "position_settle",
        "additional_wait_seconds": 0.0,
        "requested_sample_delays_skipped": [0, 3],
    }
    assert linear["post_feedback_queue_settle"]["live"] is True
    assert result["samples"][-1]["source"] == "position_settle"
    assert result["samples"][-1]["telemetry"]["position"]["x"] == pytest.approx(1.45)

    # A persistent queue is named and refused before a second pulse, rather
    # than surfacing as the hardware run's generic command_failed.
    coordinator.data.mowing_state.pos_x = 1.0
    queue_live["value"] = False
    blocked = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.8, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="vio",
        vio_heading_offset_degrees=0.0,
        max_linear_commands=2,
        motion_refresh_interval_ms=200,
        sample_delays=(0, 3),
    )
    assert blocked["stop_reason"] == "ble_link_not_ready_after_feedback"
    assert blocked["linear_commands_sent"] == 1
    assert blocked["post_feedback_queue_settle"]["reason"] == (
        "command_queue_backlogged"
    )


@pytest.mark.asyncio
async def test_settle_linear_position_feed_waits_for_lagged_jump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The settle poll waits through a lagged/frozen feed until the jump registers.

    The map-local feed sits at the pre-pulse value for a couple of samples then
    jumps (live 2026-07-15). Settling must require the feed to actually move off
    the pre-pulse position AND then stop changing, so the pulse's motion is not
    missed as a false "already settled".
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    before = _custom_path_telemetry_snapshot(coordinator)
    calls = {"n": 0}

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        calls["n"] += 1
        # Frozen for two polls (feed lag), then a single jump on the third, then
        # holds steady.
        if calls["n"] == 3:
            coordinator.data.mowing_state.pos_x += 0.10

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    res = await _settle_linear_position_feed(coordinator, before)

    assert res["moved"] is True
    assert res["settled"] is True
    # Registered the jump (poll 3) then confirmed it held (poll 4); did not run to
    # the full poll budget.
    assert calls["n"] == 4


@pytest.mark.asyncio
async def test_settle_linear_position_feed_times_out_when_no_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked pulse never registers motion, so the poll times out un-settled."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    before = _custom_path_telemetry_snapshot(coordinator)

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        return None  # position never changes

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    res = await _settle_linear_position_feed(
        coordinator, before, timeout_seconds=4.0, poll_interval_seconds=1.0
    )

    assert res["moved"] is False
    assert res["settled"] is False
    # Ran the full bounded budget (4s / 1s = 4 polls) without settling.
    assert coordinator.async_get_reports.await_count == 4


@pytest.mark.asyncio
async def test_multi_segment_vio_carries_offset_between_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segment 1's derived VIO offset is passed to segment 2 (no recalibration)."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    received_offsets: list[object] = []

    async def fake_vector(
        coordinator_arg: MammotionReportUpdateCoordinator,
        points: list[dict[str, float]],
        **kwargs: object,
    ) -> dict:
        received_offsets.append(kwargs["vio_heading_offset_degrees"])
        return {
            "valid": True,
            "stop_reason": "dry_run",
            "blockers": [],
            "phases": [{"passed": True}, {"passed": True}],
            "final_telemetry": _custom_path_telemetry_snapshot(coordinator),
            "progress_diagnostics": [],
            "vio": {"offset_degrees": 42.0},
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
            {"x": 1.1, "y": 1.0},
            {"x": 1.2, "y": 1.1},
        ],
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "dry_run"
    assert received_offsets == [None, 42.0]


@pytest.mark.asyncio
async def test_vector_segment_refetches_runtime_context_after_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-recovery gates judge refetched HA state, not the pre-recovery snapshot.

    BLE recovery can wait ~90s. Here the mower starts mowing during that wait:
    the handler-captured ha_state says "idle" (would pass), but the refetched
    context says "mowing" and must block the run.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud"

    async def fake_recover(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        coordinator.active_transport_state = "ble"
        return {"attempted": True, "ok": True, "reason": "promoted", "steps": []}

    refetches = {"count": 0}

    def refetch() -> tuple[str | None, dict | None]:
        refetches["count"] += 1
        return ("mowing", None)

    monkeypatch.setattr(mammotion_services, "_attempt_ble_recovery", fake_recover)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        ha_state="idle",
        sample_delays=(),
        refetch_runtime_context=refetch,
    )

    assert refetches["count"] == 1
    assert "runtime_not_mowing" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_multi_segment_real_rejects_missing_confirmations() -> (
    None
):
    """Real multi-segment execution requires explicit operator confirmations."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=False,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "operator_confirmed_clear_area" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_multi_segment_limits_real_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_real_segments limits real chained segment execution."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    calls: list[bool] = []

    async def fake_vector(
        coordinator_arg: MammotionReportUpdateCoordinator,
        points: list[dict[str, float]],
        **kwargs: object,
    ) -> dict:
        assert coordinator_arg is coordinator
        calls.append(bool(kwargs["dry_run"]))
        return {
            "valid": True,
            "stop_reason": "target_reached",
            "blockers": [],
            "phases": [{"passed": True}, {"passed": True}],
            "final_telemetry": _custom_path_telemetry_snapshot(coordinator),
            "progress_diagnostics": [{"passed": True}],
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
            {"x": 1.1, "y": 1.0},
            {"x": 1.2, "y": 1.1},
        ],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_real_segments=1,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "max_real_segments_reached"
    assert result["real_segments_executed"] == 1
    assert result["segments"][1]["skipped_reason"] == "max_real_segments_reached"
    assert calls == [False]


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_multi_segment_stops_on_first_failed_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-segment wrapper stops on the first failed vector segment."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    call_count = 0

    async def fake_vector(
        coordinator_arg: MammotionReportUpdateCoordinator,
        points: list[dict[str, float]],
        **kwargs: object,
    ) -> dict:
        nonlocal call_count
        assert coordinator_arg is coordinator
        call_count += 1
        return {
            "valid": True,
            "stop_reason": "no_target_progress" if call_count == 1 else "dry_run",
            "blockers": [],
            "phases": [{"passed": True}, {"passed": True}],
            "final_telemetry": _custom_path_telemetry_snapshot(coordinator),
            "progress_diagnostics": [{"passed": False}],
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
            {"x": 1.1, "y": 1.0},
            {"x": 1.2, "y": 1.1},
        ],
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "segment_failed"
    assert result["failed_segment_index"] == 1
    assert result["segments_executed"] == 1
    assert call_count == 1


@pytest.mark.asyncio
async def test_raw_vector_readiness_test_dry_run_selects_expected_phases() -> None:
    """Vector readiness dry-run covers aligned and both turn directions."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_vector_readiness_test(coordinator, sample_delays=(0,))

    assert result["dry_run"] is True
    assert result["ready_for_multi_segment"] is True
    assert result["ready_for_multi_point"] is False
    assert result["real_steps_run"] == 0
    assert result["failed_phase"] is None
    assert [phase["name"] for phase in result["phases"]] == [
        "safety_snapshot",
        "dry_run_aligned_vector",
        "dry_run_positive_turn_vector",
        "dry_run_negative_turn_vector",
    ]
    aligned = result["phases"][1]["result"]
    positive = result["phases"][2]["result"]
    negative = result["phases"][3]["result"]
    assert aligned["phases"][0]["result"]["stop_reason"] == "target_heading_reached"
    assert positive["phases"][0]["result"]["command_not_sent"]["kwargs"] == {
        "linear_speed": 0,
        "angular_speed": 180,
    }
    assert negative["phases"][0]["result"]["command_not_sent"]["kwargs"] == {
        "linear_speed": 0,
        "angular_speed": -180,
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_vector_readiness_test_real_rejects_missing_confirmations() -> None:
    """Real vector readiness rejects missing operator confirmations."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_vector_readiness_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=False,
        max_real_steps=1,
        sample_delays=(0,),
    )

    assert result["ready_for_multi_segment"] is False
    assert result["failed_phase"] == "real_preflight"
    assert result["blockers"] == ["operator_confirmed_clear_area"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_vector_readiness_test_max_real_steps_limits_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vector readiness max_real_steps limits real movement phases."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_vector_readiness_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_real_steps=1,
        calibrated_forward_heading_offset_degrees=0.0,
        sample_delays=(0,),
    )

    assert result["real_steps_run"] == 1
    assert [phase["name"] for phase in result["phases"]][-1] == "real_aligned_vector"
    assert coordinator.manager.send_command_with_args.await_count == 2


def test_raw_vector_readiness_phase_passed_accepts_proven_progress() -> None:
    """Real phase passes when at least one path-progress pulse is demonstrated."""
    assert (
        _raw_vector_readiness_phase_passed(
            "real_aligned_vector",
            {
                "stop_reason": "no_target_progress",
                "valid": True,
                "blockers": [],
                "progress_diagnostics": [
                    {"status": "path_progress", "passed": True},
                    {"status": "no_path_progress", "passed": False},
                ],
            },
        )
        is True
    )


def test_raw_vector_readiness_phase_passed_rejects_no_progress() -> None:
    """Real phase fails if no path-progress pulse is demonstrated."""
    assert (
        _raw_vector_readiness_phase_passed(
            "real_aligned_vector",
            {
                "stop_reason": "no_target_progress",
                "valid": True,
                "blockers": [],
                "progress_diagnostics": [
                    {"status": "no_path_progress", "passed": False},
                ],
            },
        )
        is False
    )


def test_raw_vector_readiness_phase_passed_accepts_aligned_translation_signal() -> None:
    """Aligned real phase accepts measured translation with heading progress."""
    assert (
        _raw_vector_readiness_phase_passed(
            "real_aligned_vector",
            {
                "stop_reason": "no_target_progress",
                "valid": True,
                "blockers": [],
                "progress_diagnostics": [
                    {
                        "status": "no_path_progress",
                        "passed": False,
                        "heading_progress": True,
                        "min_progress_distance": 0.005,
                        "measured_delta": {"distance": 0.0048},
                        "path_progress_distance": -0.0043,
                    }
                ],
            },
        )
        is True
    )


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_defaults_to_dry_run() -> None:
    """Pulse test default sends no command and reports the command not sent."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_pulse_test(coordinator, followup_samples=0)

    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["real_pulse_allowed"] is False
    assert result["reason"] == "dry_run"
    assert result["command_not_sent"] == {
        "service": "mammotion.move_forward",
        "data": {"speed": 0.55, "use_wifi": False},
    }
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_firmware_mode_skips_explicit_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Firmware nudge mode sends movement but does not issue zero-speed stop."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _manual_velocity_pulse_test(
        coordinator,
        action="forward",
        speed=0.4,
        duration_ms=750,
        stop_mode="firmware",
        post_command_sample_delays=(0.0, 2.0),
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["would_send"] is True
    assert result["stop_mode"] == "firmware"
    assert result["stop_result"]["attempted"] is False
    assert result["stop_result"]["reason"] == "firmware_nudge_mode_no_explicit_stop"
    assert result["real_pulse_completed"] is True
    assert coordinator.async_move_forward.await_count == 1
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_missing_confirmations() -> None:
    """Real pulse rejects missing operator confirmations before movement."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["reason"] == "safety_gates_failed"
    assert result["blockers"] == [
        "operator_confirmed_blades_off",
        "operator_confirmed_clear_area",
    ]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_unsafe_blade_telemetry() -> None:
    """Real pulse rejects nonzero blade/RPM telemetry before movement."""
    coordinator = _pulse_coordinator(blade_state=1, cutter_rpm=1200)

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["blockers"] == ["mower_reports_blades_off"]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_active_work_mode() -> None:
    """Real pulse rejects active mowing/working mode before movement."""
    coordinator = _pulse_coordinator(work_mode=13)

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["blockers"] == ["mower_ready"]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_allows_paused_work_mode() -> None:
    """Real pulse allows MODE_PAUSE after a canceled job when other gates pass."""
    coordinator = _pulse_coordinator(work_mode=19)

    result = await _manual_velocity_pulse_test(
        coordinator,
        duration_ms=50,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is True
    assert result["blockers"] == []
    coordinator.async_move_forward.assert_awaited_once()
    coordinator.async_stop_manual_motion.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_unavailable_position() -> None:
    """Real pulse rejects missing live map-local position before movement."""
    coordinator = _pulse_coordinator(
        position=(None, None, None), pos_type=0, zone_hash=0
    )

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["blockers"] == [
        "live_map_position_available",
        "position_area_inside",
    ]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_charging_state() -> None:
    """Real pulse rejects docked/charging state before movement."""
    coordinator = _pulse_coordinator(charge_state=2)

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["blockers"] == ["not_docked_or_charging"]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_area_out_zero_zone() -> None:
    """Real pulse rejects AREA_OUT and unknown zone hash before movement."""
    coordinator = _pulse_coordinator(pos_type=0, zone_hash=0)

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["blockers"] == ["position_area_inside"]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_allows_turn_area_inside() -> None:
    """Real pulse allows TURN_AREA_INSIDE when position and zone are known."""
    coordinator = _pulse_coordinator(pos_type=4, zone_hash=123)

    result = await _manual_velocity_pulse_test(
        coordinator,
        duration_ms=50,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is True
    assert result["blockers"] == []
    before_position = result["samples"][0]["telemetry"]["position"]
    assert before_position["pos_type_label"] == ("TURN_AREA_INSIDE")
    assert before_position["valid_for_motion"] is True
    coordinator.async_move_forward.assert_awaited_once()
    coordinator.async_stop_manual_motion.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_allows_channel_area_overlap() -> None:
    """Real pulse allows CHANNEL_AREA_OVERLAP when position and zone are known."""
    coordinator = _pulse_coordinator(pos_type=9, zone_hash=123)

    result = await _manual_velocity_pulse_test(
        coordinator,
        duration_ms=50,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    before_position = result["samples"][0]["telemetry"]["position"]

    assert result["would_send"] is True
    assert result["blockers"] == []
    assert before_position["pos_type_label"] == "CHANNEL_AREA_OVERLAP"
    assert before_position["valid_for_motion"] is True
    coordinator.async_move_forward.assert_awaited_once()
    coordinator.async_stop_manual_motion.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_zero_map_position() -> None:
    """Real pulse rejects zero map-local x/y even with known area metadata."""
    coordinator = _pulse_coordinator(position=(0.0, 0.0, 0.0))

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["blockers"] == ["map_position_nonzero"]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_real_probe_calls_move_then_stop() -> None:
    """Allowed real probe sends one tiny pulse and then the stop primitive."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_pulse_test(
        coordinator,
        action="turn_left",
        speed=0.1,
        duration_ms=50,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is True
    assert result["real_pulse_allowed"] is True
    assert result["command_result"]["attempted"] is True
    assert result["command_result"]["ok"] is True
    assert result["command_result"]["error"] is None
    assert result["command_result"]["action"] == "turn_left"
    assert result["command_result"]["coordinator_method"] == "async_move_left"
    assert result["command_result"]["transport_preference"] == "ble_preferred"
    assert result["command_result"]["duration_ms"] >= 0
    assert result["stop_result"]["attempted"] is True
    assert result["stop_result"]["ok"] is True
    assert result["stop_result"]["error"] is None
    assert result["stop_result"]["coordinator_method"] == "async_stop_manual_motion"
    assert result["real_pulse_completed"] is True
    coordinator.async_move_left.assert_awaited_once_with(speed=0.1, use_wifi=False)
    coordinator.async_stop_manual_motion.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_reports_false_command_ack() -> None:
    """A false coordinator command return is reported as an unsuccessful attempt."""
    coordinator = _pulse_coordinator()
    coordinator.async_move_forward.side_effect = RuntimeError("motion write failed")

    result = await _manual_velocity_pulse_test(
        coordinator,
        action="forward",
        speed=0.1,
        duration_ms=50,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["command_result"]["attempted"] is True
    assert result["command_result"]["ok"] is False
    assert result["command_result"]["ack"] is None
    assert result["command_result"]["error"] == "RuntimeError: motion write failed"
    assert result["real_pulse_completed"] is False
    assert result["stop_result"]["ok"] is True


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_defaults_to_dry_run() -> None:
    """Segment probe default plans the next command but sends nothing."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
    )

    assert result["service"] == "manual_velocity_segment_test"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["real_segment_allowed"] is False
    assert result["stop_reason"] == "dry_run"
    assert result["initial_controller_decision"]["action"] == "forward"
    assert result["command_not_sent"] == {
        "service": "mammotion.move_forward",
        "data": {"speed": 0.4, "use_wifi": False},
    }
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_rejects_charging_state() -> None:
    """Real segment probe blocks before movement when pulse gates fail."""
    coordinator = _pulse_coordinator(charge_state=2)

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["would_send"] is False
    assert result["stop_reason"] == "safety_gates_failed"
    assert result["blockers"] == ["not_docked_or_charging"]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_real_probe_calls_move_then_stop() -> None:
    """Allowed segment probe sends one capped pulse and then stops."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=1,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
    )

    assert result["would_send"] is True
    assert result["real_segment_allowed"] is True
    assert result["stop_reason"] == "path_progress_lost"
    assert result["pulses_sent"] == 1
    assert result["iterations"][0]["controller_decision"]["action"] == "forward"
    assert result["iterations"][0]["command_result"]["attempted"] is True
    assert result["iterations"][0]["command_result"]["ok"] is True
    assert result["iterations"][0]["command_result"]["error"] is None
    assert result["iterations"][0]["command_result"]["action"] == "forward"
    assert (
        result["iterations"][0]["command_result"]["coordinator_method"]
        == "async_move_forward"
    )
    assert result["iterations"][0]["command_result"]["transport_preference"] == "wifi"
    assert result["iterations"][0]["command_result"]["duration_ms"] >= 0
    assert result["iterations"][0]["stop_result"]["attempted"] is True
    assert result["iterations"][0]["stop_result"]["ok"] is True
    assert result["iterations"][0]["stop_result"]["error"] is None
    assert (
        result["iterations"][0]["stop_result"]["coordinator_method"]
        == "async_stop_manual_motion"
    )
    assert result["iterations"][0]["movement_diagnostic"]["status"] == (
        "visual_motion_possible_but_telemetry_unchanged"
    )
    assert result["progress_summary"]["no_progress_count"] == 1
    coordinator.async_move_forward.assert_awaited_once_with(speed=0.4, use_wifi=True)
    coordinator.async_stop_manual_motion.assert_awaited_once_with(use_wifi=True)


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_force_action_overrides_controller() -> None:
    """Force action lets diagnostics test a specific low-level movement command."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 2.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=1,
        force_action="forward",
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
    )

    decision = result["iterations"][0]["controller_decision"]
    assert decision["action"] == "forward"
    assert decision["forced"] is True
    assert decision["original_action"] == "turn_left"
    coordinator.async_move_forward.assert_awaited_once_with(speed=0.4, use_wifi=True)
    coordinator.async_move_left.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_stops_after_no_progress_limit() -> None:
    """Multi-pulse probes stop after consecutive low-progress telemetry samples."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=5,
        no_progress_limit=2,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
        require_progress_each_pulse=False,
    )

    assert result["stop_reason"] == "no_progress_limit_reached"
    assert result["pulses_sent"] == 2
    assert result["progress_summary"]["no_progress_count"] == 2
    assert coordinator.async_move_forward.await_count == 2
    assert coordinator.async_stop_manual_motion.await_count == 2


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_reports_partial_progress_timeout() -> None:
    """Max pulses after target-directed progress is a partial-progress timeout."""
    coordinator = _pulse_coordinator()

    async def move_forward_progress(*_: object, **__: object) -> None:
        coordinator.data.mowing_state.pos_x = 1.2

    coordinator.async_stop_manual_motion.side_effect = move_forward_progress

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=1,
        no_progress_limit=2,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
    )

    assert result["stop_reason"] == "partial_progress_timeout"
    assert result["completion_status"]["complete"] is False
    assert result["progress_summary"]["cumulative_path_progress"] == pytest.approx(0.2)
    assert result["iterations"][0]["path_progress_diagnostic"]["passed"] is True


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_accepts_delayed_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delayed target-directed telemetry prevents premature path_progress_lost."""
    coordinator = _pulse_coordinator()
    original_snapshot = mammotion_services._custom_path_telemetry_snapshot  # noqa: SLF001
    snapshot_count = 0

    def delayed_snapshot(coordinator_arg: object) -> dict[str, object]:
        nonlocal snapshot_count
        snapshot_count += 1
        if snapshot_count >= 5:
            coordinator.data.mowing_state.pos_x = 1.2
        return original_snapshot(coordinator_arg)

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(
        mammotion_services, "_custom_path_telemetry_snapshot", delayed_snapshot
    )
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=1,
        no_progress_limit=1,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(30.0, 45.0, 60.0),
    )

    iteration = result["iterations"][0]
    assert result["stop_reason"] == "partial_progress_timeout"
    assert iteration["late_telemetry_check"] is True
    assert iteration["late_progress_detected"] is True
    assert iteration["telemetry_latency_seconds"] == 45.0
    assert iteration["late_path_progress_diagnostic"]["passed"] is True
    assert iteration["path_progress_diagnostic"]["passed"] is True
    assert result["progress_summary"]["cumulative_path_progress"] == pytest.approx(0.2)
    coordinator.async_move_forward.assert_awaited_once_with(speed=0.4, use_wifi=True)


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_reports_lost_after_late_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No delayed telemetry progress through the late window remains progress lost."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=1,
        no_progress_limit=1,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(30.0, 45.0, 60.0),
    )

    iteration = result["iterations"][0]
    assert result["stop_reason"] == "path_progress_lost"
    assert iteration["late_telemetry_check"] is True
    assert iteration["late_progress_detected"] is False
    assert iteration["telemetry_latency_seconds"] is None
    assert iteration["late_path_progress_diagnostic"]["passed"] is False


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_reports_path_complete_at_max_pulses() -> (
    None
):
    """Max pulses at target reports path_complete, not a timeout."""
    coordinator = _pulse_coordinator()

    async def move_to_target(*_: object, **__: object) -> None:
        coordinator.data.mowing_state.pos_x = 2.0

    coordinator.async_stop_manual_motion.side_effect = move_to_target

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=1,
        waypoint_tolerance=0.1,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
    )

    assert result["stop_reason"] == "path_complete"
    assert result["completion_status"]["complete"] is True
    assert result["iterations"][0]["path_progress_diagnostic"]["passed"] is True


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_stops_when_quality_degrades() -> None:
    """Multi-pulse probes stop when telemetry quality degrades after a pulse."""
    coordinator = _pulse_coordinator()

    async def degrade_position_quality(*_: object, **__: object) -> None:
        coordinator.data.mowing_state.pos_level = 2
        coordinator.data.report_data.rtk.pos_level = 2

    coordinator.async_stop_manual_motion.side_effect = degrade_position_quality

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=5,
        no_progress_limit=5,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
        service_name="manual_velocity_multi_pulse_test",
    )

    assert result["stop_reason"] == "telemetry_quality_degraded"
    assert result["blockers"] == ["pos_level_degraded"]
    assert result["pulses_sent"] == 1
    assert result["iterations"][0]["quality_degradation"]["degraded"] is True


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_can_report_multi_pulse_service_name() -> (
    None
):
    """The same guarded engine can back the explicit multi-pulse service."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        service_name="manual_velocity_multi_pulse_test",
    )

    assert result["service"] == "manual_velocity_multi_pulse_test"
    coordinator.async_move_forward.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_cumulative_pulse_test_defaults_to_dry_run() -> None:
    """Cumulative pulse probe default plans the burst but sends nothing."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_cumulative_pulse_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        heading_offset_degrees=110.0,
        heading_offset_candidates=[110.0, 0.0],
    )

    assert result["service"] == "manual_velocity_cumulative_pulse_test"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["real_probe_allowed"] is False
    assert result["stop_reason"] == "dry_run"
    assert result["heading_offset_candidates"] == [110.0, 0.0]
    assert (
        result["initial_controller_decision"]["selected_heading_offset_degrees"] == 0.0
    )
    assert len(result["initial_controller_decision"]["heading_offset_diagnostics"]) == 2
    assert result["command_not_sent"] == {
        "service": "mammotion.move_forward",
        "data": {"speed": 0.4, "use_wifi": False},
    }
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_cumulative_pulse_test_firmware_mode_skips_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cumulative firmware nudge mode sends pulses without explicit zero-stop."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _manual_velocity_cumulative_pulse_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=2,
        force_action="forward",
        stop_mode="firmware",
        use_wifi=False,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        cumulative_sample_delays=(0.0,),
    )

    assert result["stop_mode"] == "firmware"
    assert result["pulses_sent"] == 2
    assert result["pulse_results"][0]["stop_result"]["attempted"] is False
    assert result["pulse_results"][0]["stop_result"]["reason"] == (
        "firmware_nudge_mode_no_explicit_stop"
    )
    assert coordinator.async_move_forward.await_count == 2
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_cumulative_pulse_test_detects_delayed_cumulative_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cumulative probe sends a pulse burst, then accepts delayed total progress."""
    coordinator = _pulse_coordinator()
    original_snapshot = mammotion_services._custom_path_telemetry_snapshot  # noqa: SLF001
    snapshot_count = 0

    def delayed_snapshot(coordinator_arg: object) -> dict[str, object]:
        nonlocal snapshot_count
        snapshot_count += 1
        if snapshot_count >= 10:
            coordinator.data.mowing_state.pos_x = 1.2
        return original_snapshot(coordinator_arg)

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(
        mammotion_services,
        "_custom_path_telemetry_snapshot",
        delayed_snapshot,
    )
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _manual_velocity_cumulative_pulse_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=3,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        cumulative_sample_delays=(0.0, 30.0, 60.0),
    )

    assert result["stop_reason"] == "cumulative_progress_detected"
    assert result["result_status"] == "cumulative_progress_detected"
    assert result["pulses_sent"] == 3
    assert result["cumulative_progress_detected"] is True
    assert result["telemetry_latency_seconds"] == 60.0
    assert result["cumulative_delta"]["distance"] == pytest.approx(0.2)
    assert result["cumulative_path_progress_diagnostic"]["passed"] is True
    assert coordinator.async_move_forward.await_count == 3
    assert coordinator.async_stop_manual_motion.await_count == 3


@pytest.mark.asyncio
async def test_experimental_execute_segment_burst_stops_after_no_cumulative_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Burst execution stops after a burst that never gets telemetry progress."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _experimental_execute_segment_burst(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        pulses_per_burst=2,
        max_bursts=3,
        stop_mode="immediate",
        calibrated_forward_heading_degrees=0,
        use_wifi=True,
        cumulative_sample_delays=(0.0,),
    )

    assert result["service"] == "experimental_execute_segment_burst"
    assert result["stop_reason"] == "no_cumulative_progress"
    assert result["bursts_sent"] == 1
    assert result["pulses_sent"] == 2
    assert result["bursts"][0]["cumulative_progress_detected"] is False
    assert coordinator.async_move_forward.await_count == 2
    assert coordinator.async_stop_manual_motion.await_count == 2


@pytest.mark.asyncio
async def test_experimental_execute_segment_burst_continues_after_cumulative_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Burst execution can send another burst after delayed target progress."""
    coordinator = _pulse_coordinator()
    stop_count = 0

    async def move_after_stop(*_: object, **__: object) -> None:
        nonlocal stop_count
        stop_count += 1
        coordinator.data.mowing_state.pos_x = 1.2 if stop_count == 1 else 2.0

    async def no_sleep(_: float) -> None:
        return None

    coordinator.async_stop_manual_motion.side_effect = move_after_stop
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _experimental_execute_segment_burst(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        pulses_per_burst=1,
        max_bursts=3,
        waypoint_tolerance=0.05,
        stop_mode="immediate",
        calibrated_forward_heading_degrees=0,
        use_wifi=True,
        cumulative_sample_delays=(0.0, 60.0),
    )

    assert result["stop_reason"] == "path_complete"
    assert result["completion_status"]["complete"] is True
    assert result["bursts_sent"] == 2
    assert result["pulses_sent"] == 2
    assert result["bursts"][0]["cumulative_progress_detected"] is True
    assert result["bursts"][1]["cumulative_progress_detected"] is True
    assert result["cumulative_path_progress"] == pytest.approx(1.0)
    assert coordinator.async_move_forward.await_count == 2
    assert coordinator.async_stop_manual_motion.await_count == 2


@pytest.mark.asyncio
async def test_experimental_execute_segment_burst_blocks_unproven_turn_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default experimental execution only allows calibrated forward segments."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _experimental_execute_segment_burst(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        cumulative_sample_delays=(0.0,),
    )

    assert result["stop_reason"] == (
        "segment_heading_outside_calibrated_forward_window"
    )
    assert result["blockers"] == ["unproven_turn_or_lateral_motion_required"]
    assert (
        result["manual_motion_execution_policy"]["experimental_segment_scope"]
        == "one_segment_calibrated_forward_only"
    )
    assert result["calibrated_forward_heading_diagnostic"] == {
        "segment_heading_degrees": 0.0,
        "calibrated_forward_heading_degrees": 270.0,
        "heading_error_degrees": 90.0,
        "tolerance_degrees": 45.0,
        "within_calibrated_forward_window": False,
        "allow_unproven_turns": False,
    }
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_stops_when_path_complete() -> None:
    """Segment probe sends nothing when current position is already at target."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.01, "y": 1.01}],
        waypoint_tolerance=0.1,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
    )

    assert result["would_send"] is True
    assert result["stop_reason"] == "path_complete"
    assert result["pulses_sent"] == 0
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


def test_manual_velocity_heading_calibration_reports_vector_offset() -> None:
    """Heading calibration compares reported heading to movement vector heading."""
    before = {
        "position": {
            "x": 1.0,
            "y": 1.0,
            "toward": 0.0,
            "source": "report_data.locations[0]",
        }
    }
    after = {
        "position": {
            "x": 1.0,
            "y": 2.0,
            "toward": 0.0,
            "source": "report_data.locations[0]",
        }
    }

    result = _manual_velocity_heading_calibration(
        action="forward",
        before=before,
        after=after,
        min_progress_distance=0.003,
        min_heading_change_degrees=1.0,
    )

    assert result["movement_vector_heading"] == 90.0
    assert result["heading_error_degrees"] == 90.0
    assert result["recommended_heading_offset_degrees"] == 90.0
    assert result["interpretation"] == "movement_vector_available"


async def test_motion_open_sleep_normal_path_sends_no_stop() -> None:
    """An uncancelled pulse sleep must not add an extra stop of its own."""
    coordinator = SimpleNamespace(async_stop_manual_motion=AsyncMock())
    await _motion_open_sleep(coordinator, 0)
    coordinator.async_stop_manual_motion.assert_not_called()


async def test_motion_open_sleep_delivers_stop_on_cancellation() -> None:
    """Cancellation mid-pulse delivers the mandatory stop before re-raising."""
    coordinator = SimpleNamespace(async_stop_manual_motion=AsyncMock())
    task = asyncio.create_task(_motion_open_sleep(coordinator, 30.0))
    await asyncio.sleep(0)  # let the task enter the pulse sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    coordinator.async_stop_manual_motion.assert_awaited_once()


async def test_motion_open_sleep_swallows_stop_errors_on_cancellation() -> None:
    """A failing stop must not mask the CancelledError during teardown."""
    coordinator = SimpleNamespace(
        async_stop_manual_motion=AsyncMock(side_effect=RuntimeError("ble gone"))
    )
    task = asyncio.create_task(_motion_open_sleep(coordinator, 30.0))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    coordinator.async_stop_manual_motion.assert_awaited_once()


def _fake_motion_mower(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Route _get_mower_by_entity_id to a fixed fake mower/coordinator."""
    coordinator = SimpleNamespace(async_stop_manual_motion=AsyncMock())
    mower = SimpleNamespace(reporting_coordinator=coordinator)
    monkeypatch.setattr(
        mammotion_services,
        "_get_mower_by_entity_id",
        lambda _hass, _entity_id: mower,
    )
    monkeypatch.setattr(
        mammotion_services,
        "_manual_motion_authorization",
        lambda *_args, **_kwargs: {
            "real_motion_allowed": True,
            "blockers": [],
        },
    )
    return mower


def _motion_call(**overrides: object) -> SimpleNamespace:
    """Minimal ServiceCall stand-in for the exclusivity wrapper."""
    data: dict[str, object] = {"entity_id": "lawn_mower.test", "dry_run": False}
    data.update(overrides)
    return SimpleNamespace(data=data)


async def test_exclusive_motion_wrapper_rejects_concurrent_real_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second real motion run is strictly rejected while the first owns the mower."""
    _fake_motion_mower(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_handler(call: object) -> dict[str, object]:
        started.set()
        await release.wait()
        return {"ran": "slow"}

    async def fast_handler(call: object) -> dict[str, object]:
        return {"ran": "fast"}

    slow = _wrap_exclusive_manual_motion(object(), "svc_slow", slow_handler)
    fast = _wrap_exclusive_manual_motion(object(), "svc_fast", fast_handler)
    task = asyncio.create_task(slow(_motion_call()))
    await started.wait()

    busy = await fast(_motion_call())
    assert busy["stop_reason"] == "manual_motion_in_progress"
    assert busy["blockers"] == ["manual_motion_in_progress"]
    assert busy["busy_owner"] == "svc_slow"
    assert busy["would_send"] is False

    # Dry runs never move and pass straight through while the owner runs.
    dry = await fast(_motion_call(dry_run=True))
    assert dry == {"ran": "fast"}

    release.set()
    assert await task == {"ran": "slow"}

    # Owner finished: the mower is claimable again.
    again = await fast(_motion_call())
    assert again == {"ran": "fast"}


def _saga_motion_mower(
    monkeypatch: pytest.MonkeyPatch, *, saga_active: bool | None
) -> SimpleNamespace:
    """Motion mower whose command queue reports an exclusive saga state.

    ``saga_active=None`` omits the queue entirely, standing in for pymammotion
    API drift.
    """
    handle = SimpleNamespace()
    if saga_active is not None:
        handle.queue = SimpleNamespace(is_saga_active=saga_active)
    coordinator = SimpleNamespace(
        async_stop_manual_motion=AsyncMock(),
        device_name="Luba-Test",
        manager=SimpleNamespace(mower=lambda _name: handle),
    )
    mower = SimpleNamespace(reporting_coordinator=coordinator)
    monkeypatch.setattr(
        mammotion_services,
        "_get_mower_by_entity_id",
        lambda _hass, _entity_id: mower,
    )
    monkeypatch.setattr(
        mammotion_services,
        "_manual_motion_authorization",
        lambda *_args, **_kwargs: {
            "real_motion_allowed": True,
            "blockers": [],
        },
    )
    return mower


async def test_exclusive_motion_wrapper_rejects_motion_during_a_saga(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Motion is refused *by name* while an exclusive map saga holds the queue.

    The reverse ordering of the guard that already exists: `manual_motion_owner`
    stops a saga starting while motion runs, but nothing stopped motion when a
    saga was already running. Motion is `Priority.NORMAL` with
    `skip_if_saga_active=False`, so it blocks on the exclusive slot while the
    executor sleeps out its pulse on local timing -- separating a movement from
    its stop, or letting either be dropped at the 120 s `_COMMAND_TTL`.

    The original design deliberately had no gate here ("refusing a command the
    operator just issued is worse than the wait"), which assumed sagas were
    rare and operator-triggered. They now also fire automatically, and the wait
    was never benign. A *named* refusal answers the original objection.

    Raised by adversarial review 2026-07-26.
    """
    _saga_motion_mower(monkeypatch, saga_active=True)

    async def handler(call: object) -> dict[str, object]:
        return {"ran": "yes"}

    wrapped = _wrap_exclusive_manual_motion(object(), "svc_motion", handler)

    busy = await wrapped(_motion_call())
    assert busy["stop_reason"] == "manual_motion_in_progress"
    assert busy["busy_owner"] == "map_sync_saga"
    assert busy["would_send"] is False

    # A dry run reads telemetry only and still passes through.
    assert await wrapped(_motion_call(dry_run=True)) == {"ran": "yes"}


async def test_exclusive_motion_wrapper_allows_motion_once_saga_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the exclusive slot is free the mower is claimable again."""
    _saga_motion_mower(monkeypatch, saga_active=False)

    async def handler(call: object) -> dict[str, object]:
        return {"ran": "yes"}

    wrapped = _wrap_exclusive_manual_motion(object(), "svc_motion", handler)
    assert await wrapped(_motion_call()) == {"ran": "yes"}


async def test_exclusive_saga_probe_degrades_to_allowing_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable queue must not block motion.

    Positively-True-only: pymammotion API drift should lose the guard, never
    refuse every motion command. Mirrors `_ble_transport_usable`'s convention.
    """
    _saga_motion_mower(monkeypatch, saga_active=None)

    async def handler(call: object) -> dict[str, object]:
        return {"ran": "yes"}

    wrapped = _wrap_exclusive_manual_motion(object(), "svc_motion", handler)
    assert await wrapped(_motion_call()) == {"ran": "yes"}


async def test_exclusive_motion_wrapper_releases_on_handler_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashing run must release the mower for the next start."""
    _fake_motion_mower(monkeypatch)

    async def crashing_handler(call: object) -> dict[str, object]:
        raise RuntimeError("boom")

    async def ok_handler(call: object) -> dict[str, object]:
        return {"ran": True}

    crashing = _wrap_exclusive_manual_motion(object(), "svc_crash", crashing_handler)
    ok = _wrap_exclusive_manual_motion(object(), "svc_ok", ok_handler)
    with pytest.raises(RuntimeError):
        await crashing(_motion_call())
    assert await ok(_motion_call()) == {"ran": True}


async def test_exclusive_motion_wrapper_exempts_zero_motion_stop_nudge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The card's Abort (send_movement 0/0) preempts a running loop; real probes don't."""
    _fake_motion_mower(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_handler(call: object) -> dict[str, object]:
        started.set()
        await release.wait()
        return {"ran": "slow"}

    async def probe_handler(call: object) -> dict[str, object]:
        return {"ran": "probe"}

    slow = _wrap_exclusive_manual_motion(object(), "svc_slow", slow_handler)
    probe = _wrap_exclusive_manual_motion(
        object(), "svc_probe", probe_handler, allow_stop_nudge=True
    )
    task = asyncio.create_task(slow(_motion_call()))
    await started.wait()

    # Zero-motion stop nudge: passes through even while the mower is owned.
    nudge = await probe(
        _motion_call(command="send_movement", linear_speed=0, angular_speed=0)
    )
    assert nudge == {"ran": "probe"}

    # A real (nonzero) probe is still rejected as busy.
    real_probe = await probe(
        _motion_call(command="send_movement", linear_speed=400, angular_speed=0)
    )
    assert real_probe["stop_reason"] == "manual_motion_in_progress"

    release.set()
    await task


def test_is_zero_motion_stop_nudge_truth_table() -> None:
    """Only send_movement with both speeds zero counts as a stop nudge."""
    is_nudge = _is_zero_motion_stop_nudge
    assert is_nudge("send_movement", 0, 0)
    assert not is_nudge("send_movement", 400, 0)
    assert not is_nudge("send_movement", 0, 500)
    assert not is_nudge("move_forward", 0, 0)


def _install_virtual_clock(
    monkeypatch: pytest.MonkeyPatch, start: float = 100.0
) -> dict[str, float]:
    """Replace monotonic/sleep with a virtual clock that advances on sleep."""
    clock = {"now": start}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    return clock


def test_motion_refresh_interval_matches_decompiled_app_constant() -> None:
    """Our app-parity default is the app's own timer period, not a guess.

    `CarRemoteControlManage2.frequency = 0.2f` -> 200 ms (Mammotion 2.3.8.19).
    """
    assert mammotion_services._MOTION_REFRESH_INTERVAL_MS_APP == 200  # noqa: SLF001


@pytest.mark.asyncio
async def test_motion_refresh_window_disabled_keeps_single_shot_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interval 0 is the proven path: wait out the pulse, never re-send."""
    coordinator = _pulse_coordinator()
    clock = _install_virtual_clock(monkeypatch)
    sends: list[float] = []

    async def resend() -> None:
        sends.append(clock["now"])

    report = await _motion_refresh_window(
        coordinator,
        resend=resend,
        duration_seconds=4.0,
        refresh_interval_ms=0,
    )

    assert report["refresh_enabled"] is False
    assert report["refresh_commands_sent"] == 0
    assert report["refresh_write_completions_elapsed_ms"] == []
    assert sends == []
    # The full pulse window is still waited out.
    assert clock["now"] == pytest.approx(104.0)


@pytest.mark.asyncio
async def test_motion_refresh_window_resends_for_whole_pulse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A positive interval re-sends across the pulse, never past its end."""
    coordinator = _pulse_coordinator()
    clock = _install_virtual_clock(monkeypatch)
    sends: list[float] = []

    async def resend() -> None:
        sends.append(clock["now"])

    # 250 ms divides 4.0 s exactly in binary floating point, so the count is
    # deterministic rather than sensitive to accumulated rounding.
    report = await _motion_refresh_window(
        coordinator,
        resend=resend,
        duration_seconds=4.0,
        refresh_interval_ms=250,
    )

    assert report["refresh_enabled"] is True
    assert report["refresh_interval_ms"] == 250
    assert report["max_refresh_commands"] == 16
    # 16 slots fit the window; the one landing exactly on the deadline is
    # skipped so the last command is always followed by real motion time.
    assert report["refresh_commands_sent"] == 15
    assert len(sends) == 15
    assert clock["now"] == pytest.approx(104.0)
    assert max(sends) < 104.0
    # Evenly spaced at the requested cadence.
    assert sends[0] == pytest.approx(100.25)
    assert sends[1] - sends[0] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_motion_refresh_window_stops_at_confirmed_refresh_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final approach stops immediately after its discrete resend budget."""
    coordinator = _pulse_coordinator()
    clock = _install_virtual_clock(monkeypatch)
    sends: list[float] = []

    async def resend() -> None:
        sends.append(clock["now"])

    report = await _motion_refresh_window(
        coordinator,
        resend=resend,
        duration_seconds=3.5,
        refresh_interval_ms=200,
        max_refresh_commands=3,
    )

    assert report["refresh_command_limit"] == 3
    assert report["max_refresh_commands"] == 3
    assert report["refresh_commands_sent"] == 3
    assert len(report["refresh_write_durations_ms"]) == 3
    assert report["refresh_write_completions_elapsed_ms"] == pytest.approx(
        [200.0, 400.0, 600.0]
    )
    assert sends == pytest.approx([100.2, 100.4, 100.6])
    assert clock["now"] == pytest.approx(100.6)
    assert report["elapsed_ms"] == pytest.approx(600.0)


@pytest.mark.asyncio
async def test_motion_refresh_window_stops_when_cancelled_inside_resend() -> None:
    """Cancellation while a refresh send is in flight must still stop the mower.

    ``_motion_open_sleep`` was added (commit abf65696) because
    ``CancelledError`` is a ``BaseException``, so ``except Exception`` handlers
    never see it and a cancel during the pulse sleep exited without the
    mandatory stop. The refresh window reintroduced exactly that trap one await
    further in: a cancel landing inside ``resend()`` fell straight past the
    caller's stop. A movement command is already open on the mower at that
    point, so it would keep driving until its own device-side timeout.

    Found by adversarial review 2026-07-26 and reproduced against the real
    helper before fixing: zero stop calls.
    """
    stops: list[str] = []
    coordinator = SimpleNamespace(
        async_stop_manual_motion=AsyncMock(
            side_effect=lambda *a, **k: stops.append("stop")
        )
    )
    resend_started = asyncio.Event()

    async def blocking_resend() -> None:
        resend_started.set()
        await asyncio.sleep(3600)  # cancellation lands here, inside resend()

    task = asyncio.create_task(
        mammotion_services._motion_refresh_window(  # noqa: SLF001
            coordinator,
            resend=blocking_resend,
            duration_seconds=4.0,
            refresh_interval_ms=200,
        )
    )
    await resend_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # The stop must have been delivered, and the cancellation must still
    # propagate so the caller's own teardown runs.
    assert stops == ["stop"]


@pytest.mark.asyncio
async def test_motion_refresh_window_propagates_operator_session_stop() -> None:
    """An operator-cancelled session stops and releases its owner immediately.

    ``ManualMotionCancelledError`` is raised by the confirmed-dispatch guard
    before a post-abort nonzero refresh can be queued.  Treating it as an
    ordinary failed resend kept the service handler alive in feedback/sample
    waits, so ``stop_manual_motion`` could time out waiting for the owner even
    though motion itself was safely blocked.
    """
    coordinator = SimpleNamespace(async_stop_manual_motion=AsyncMock())

    async def cancelled_resend() -> None:
        raise ManualMotionCancelledError("operator stop")

    with pytest.raises(ManualMotionCancelledError, match="operator stop"):
        await _motion_refresh_window(
            coordinator,
            resend=cancelled_resend,
            duration_seconds=1.0,
            refresh_interval_ms=200,
        )

    coordinator.async_stop_manual_motion.assert_awaited_once()


@pytest.mark.asyncio
async def test_motion_refresh_window_clamps_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absurd intervals are clamped, so a bad param cannot flood the BLE queue."""
    coordinator = _pulse_coordinator()
    _install_virtual_clock(monkeypatch)

    async def resend() -> None:
        return None

    too_fast = await _motion_refresh_window(
        coordinator, resend=resend, duration_seconds=1.0, refresh_interval_ms=1
    )
    too_slow = await _motion_refresh_window(
        coordinator, resend=resend, duration_seconds=1.0, refresh_interval_ms=99999
    )

    assert too_fast["refresh_interval_ms"] == 50
    assert too_slow["refresh_interval_ms"] == 1000


@pytest.mark.asyncio
async def test_motion_refresh_window_survives_a_failed_resend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed refresh stops refreshing and reports why, but never raises.

    The caller still owns the mandatory stop, so a half-refreshed window is a
    shorter drive -- never a runaway one.
    """
    coordinator = _pulse_coordinator()
    _install_virtual_clock(monkeypatch)
    attempts: list[int] = []

    async def resend() -> None:
        attempts.append(1)
        if len(attempts) == 2:
            raise RuntimeError("ble dropped")

    report = await _motion_refresh_window(
        coordinator,
        resend=resend,
        duration_seconds=4.0,
        refresh_interval_ms=250,
    )

    assert report["refresh_commands_sent"] == 1
    assert "ble dropped" in report["refresh_error"]
    assert len(attempts) == 2


def test_app_scale_speeds_applies_deadband_and_ceilings() -> None:
    """Stick fractions convert exactly as the app's rocker transform does."""
    # 15% deadband: anything at or below it is a dead stick.
    assert _app_scale_speeds(0.15, 0.0) == (0, 0)
    assert _app_scale_speeds(0.10, 0.0) == (0, 0)
    # Full deflection lands on the real ceilings, not the round numbers in the
    # app's own comments (1000/450), because the deadband is applied first.
    assert _app_scale_speeds(1.0, 0.0) == (850, 0)
    assert _app_scale_speeds(0.0, 1.0) == (0, 382)
    # Sign selects direction: negative is backward / left.
    assert _app_scale_speeds(-1.0, 0.0) == (-850, 0)
    assert _app_scale_speeds(0.0, -1.0) == (0, -382)


def test_app_speed_scale_report_flags_our_angular_default() -> None:
    """Our long-standing angular 500 is above anything the app can produce."""
    report = _app_speed_scale_report(400, 500)

    assert report["available"] is True
    assert report["app_max_linear_speed"] == 850
    assert report["app_max_angular_speed"] == 382
    assert report["linear_above_app_max"] is False
    assert report["angular_above_app_max"] is True
    # Our linear default is under half the app's full-scale throttle.
    assert report["linear_fraction_of_app_max"] == pytest.approx(0.471, abs=0.001)


def test_app_speed_scale_report_handles_missing_values() -> None:
    """A non-numeric speed degrades to unavailable instead of raising."""
    assert _app_speed_scale_report(None, 0) == {"available": False}


def test_app_speed_scale_matches_pymammotion() -> None:
    """Our local transform agrees with pymammotion's, which mirrors the app.

    Implemented locally so the numbers stay deterministic across dependency
    bumps; this pins the two together so a silent upstream change is caught.
    """
    movement = pytest.importorskip("pymammotion.utility.movement")

    for fraction in (0.0, 0.1, 0.15, 0.4, 0.55, 1.0):
        percent = movement.get_percent(abs(fraction) * 100)
        # 90 degrees == forward (linear axis), 0 degrees == right (angular axis).
        upstream_linear, _ = movement.transform_both_speeds(90.0, 0.0, percent, 0.0)
        _, upstream_angular = movement.transform_both_speeds(0.0, 0.0, 0.0, percent)
        ours_linear, _ = _app_scale_speeds(fraction, 0.0)
        _, ours_angular = _app_scale_speeds(0.0, fraction)
        assert ours_linear == upstream_linear
        assert ours_angular == upstream_angular


def test_multi_segment_segment_that_reached_target_passes() -> None:
    """Reaching the target is success even if the final pulse crept.

    Regression: requiring every per-pulse diagnostic to clear
    `min_progress_distance` marked arrived segments as failed, because the
    final approach necessarily moves less than a full pulse -- which stopped
    the run and meant later segments never executed.
    """
    segment_result = {
        "stop_reason": "target_reached",
        "valid": True,
        "blockers": [],
        "progress_diagnostics": [
            {"passed": True, "path_progress_distance": 0.31},
            # Short final approach: real motion, below the per-pulse threshold.
            {"passed": False, "path_progress_distance": 0.02},
        ],
    }

    assert _raw_multi_segment_phase_passed(segment_result, real_segment=True) is True


def test_multi_segment_segment_that_did_not_arrive_fails() -> None:
    """A run that aborted short of the target is still a failure."""
    aborted = {
        "stop_reason": "no_target_progress",
        "valid": True,
        "blockers": [],
        "progress_diagnostics": [{"passed": True}],
    }
    blocked = {
        "stop_reason": "target_reached",
        "valid": True,
        "blockers": ["ble_transport_required"],
        "progress_diagnostics": [{"passed": True}],
    }
    invalid = {
        "stop_reason": "target_reached",
        "valid": False,
        "blockers": [],
        "progress_diagnostics": [{"passed": True}],
    }

    assert _raw_multi_segment_phase_passed(aborted, real_segment=True) is False
    assert _raw_multi_segment_phase_passed(blocked, real_segment=True) is False
    assert _raw_multi_segment_phase_passed(invalid, real_segment=True) is False
