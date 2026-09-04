"""Tests for BLE transport health: link liveness, queue settle, recovery, cooldown."""

import asyncio
import time
from collections.abc import Callable, Coroutine
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pymammotion.messaging.command_queue import DeviceCommandQueue, Priority
from pymammotion.transport.base import NoTransportAvailableError, TransportType
from pymammotion.transport.ble import BLETransport, BLETransportConfig

from custom_components.mammotion import coordinator as mammotion_coordinator
from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.coordinator import (
    MammotionReportUpdateCoordinator,
)
from custom_components.mammotion.services import (
    _BLE_SEND_STALL_SECONDS,
    _ble_connect_cooldown_active,
    _ble_link_liveness,
    _ble_ready_for_motion,
    _ble_transport_usable,
    _custom_path_telemetry_snapshot,
    _raw_pymammotion_execute_multi_segment,
    _raw_pymammotion_execute_vector_segment,
    _settle_linear_position_feed,
    _streak_shows_dead_telemetry,
    _streak_shows_no_actuation,
    _transport_is_ble,
    _vio_turn_probe,
)

from .conftest import _patch_services_monotonic, _pulse_coordinator


@pytest.mark.asyncio
async def test_vector_segment_real_run_requires_ble_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real motion is refused when the active transport is not BLE.

    ⚠️ `_attempt_ble_recovery` is stubbed to a FAILED recovery, which is what
    production does here: `ble_auto_recover` defaults to True, so a non-BLE
    transport runs recovery BEFORE the gates. Unstubbed, this test drove the
    real 90 s timeout at a 5 s poll and took **90 seconds -- 65% of the entire
    suite's runtime** -- to assert a refusal that needs none of it. The stub
    keeps the path under test (the gates judge POST-recovery state, and the
    recovery failed) and every assertion is unchanged.
    🔑 The recovery routine's own behaviour is covered directly, and fast, by
    `test_ble_recovery_gives_up_when_ble_never_promotes` below.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    # Flip the coordinator's normalized live transport to cloud.
    coordinator.active_transport_state = "cloud_aliyun"

    async def failed_recovery(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "attempted": True,
            "ok": False,
            "reason": "timeout",
            "steps": ["reasserted_ble_preference"],
        }

    monkeypatch.setattr(mammotion_services, "_attempt_ble_recovery", failed_recovery)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        sample_delays=(),
    )

    assert "ble_transport_required" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"
    coordinator.manager.send_command_with_args.assert_not_called()
    # The gate refused on POST-recovery state, and the report says recovery ran.
    assert result["ble_recovery"]["ok"] is False


@pytest.mark.asyncio
async def test_ble_recovery_gives_up_when_ble_never_promotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🔑 Direct coverage for `_attempt_ble_recovery`, which every other test stubs.

    It is patched out at all six existing call sites and at the transport-gate
    test above, so the routine itself had no test of its own -- the 90 s it used
    to burn there bought no coverage of it either, because that test asserts
    only the gate. Driven here with a tiny budget: it must reassert the
    preference, spend its one full off->on toggle at the halfway point, and
    report failure rather than claiming success.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud_aliyun"
    toggles: list[bool] = []

    async def record_toggle(value: bool) -> None:
        toggles.append(value)

    coordinator.async_set_bluetooth_enabled = record_toggle

    # The off->on toggle has a hardcoded `asyncio.sleep(3)` that no argument
    # reaches, so the budget alone cannot make this fast. The loop is still
    # bounded by the real `timeout_seconds` below, not by the patched sleep.
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    report = await mammotion_services._attempt_ble_recovery(  # noqa: SLF001
        coordinator, timeout_seconds=0.05, poll_interval_seconds=0.01
    )

    assert report["attempted"] is True
    assert report["ok"] is False
    # Reasserted the preference, then spent its single off->on toggle.
    assert toggles == [True, False, True]
    assert "reasserted_ble_preference" in report["steps"]


@pytest.mark.asyncio
async def test_vector_segment_dry_run_allowed_off_ble() -> None:
    """Dry-run stays valid over a non-BLE transport (the BLE gate only guards real motion)."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud_aliyun"

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        sample_delays=(),
    )

    assert result["stop_reason"] == "dry_run"
    assert "ble_transport_required" not in result["blockers"]


def test_active_transport_state_normalizes_real_ble_enum() -> None:
    """The coordinator normalizes the real TransportType.BLE enum to 'ble'.

    Regression guard for the BLE gate: ``str(TransportType.BLE)`` is
    ``'TransportType.BLE'`` (not ``'ble'``), so any exact-string match against
    ``str(active_transport())`` silently fails and blocks every real run. The
    services BLE gate must go through this normalized property, and this test
    exercises the property against the genuine enum -- not a stand-in string.
    """
    # Document the trap: the raw stringified enum is not "ble".
    assert str(TransportType.BLE).lower() != "ble"

    handle = SimpleNamespace(active_transport=lambda: TransportType.BLE)
    fake_self = SimpleNamespace(
        device_name="Luba-Test",
        manager=SimpleNamespace(mower=lambda _name: handle),
    )

    normalized = MammotionReportUpdateCoordinator.active_transport_state.fget(fake_self)

    assert normalized == "ble"
    assert _transport_is_ble(SimpleNamespace(active_transport_state=normalized))


def test_active_transport_state_handles_no_transport_available() -> None:
    """A temporarily offline mower reports ``none`` without breaking setup."""
    handle = SimpleNamespace(
        active_transport=MagicMock(
            side_effect=NoTransportAvailableError("all transports unavailable")
        )
    )
    fake_self = SimpleNamespace(
        device_name="Luba-Test",
        manager=SimpleNamespace(mower=lambda _name: handle),
    )

    normalized = MammotionReportUpdateCoordinator.active_transport_state.fget(fake_self)

    assert normalized == "none"


@pytest.mark.asyncio
async def test_vector_segment_ble_auto_recovers_then_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful BLE auto-recovery lets ble_transport_required pass and the run proceed."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud"  # not BLE at entry
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def no_sleep(_: float) -> None:
        return None

    recovery_calls: list[object] = []

    async def fake_recover(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        recovery_calls.append(coordinator_arg)
        coordinator.active_transport_state = "ble"  # promoted
        return {"attempted": True, "ok": True, "reason": "promoted", "steps": []}

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
            "commands_sent": 2,
            "command_results": [],
        }

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(mammotion_services, "_attempt_ble_recovery", fake_recover)
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

    # Recovery ran once, promoted BLE, and the transport gate is no longer a blocker.
    assert recovery_calls == [coordinator]
    assert result["ble_recovery"]["ok"] is True
    assert "ble_transport_required" not in result["blockers"]
    assert result["blockers"] == []


@pytest.mark.asyncio
async def test_vector_segment_ble_auto_recovery_failure_fails_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed BLE auto-recovery leaves ble_transport_required blocking the real run."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud"  # not BLE, and recovery can't fix it
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def fake_recover(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        # Present-but-not-promoted: recovery could not win the slot (phone app).
        return {
            "attempted": True,
            "ok": False,
            "reason": "ble_promotion_timeout_check_phone_app",
            "steps": ["reasserted_ble_preference", "ble_toggled"],
        }

    monkeypatch.setattr(mammotion_services, "_attempt_ble_recovery", fake_recover)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["ble_recovery"]["ok"] is False
    assert result["ble_recovery"]["reason"] == "ble_promotion_timeout_check_phone_app"
    assert "ble_transport_required" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vector_segment_ble_auto_recover_disabled_skips_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ble_auto_recover=false skips recovery entirely; the transport gate blocks as before."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud"
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def fail_if_called(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        raise AssertionError("recovery must not run when ble_auto_recover is False")

    monkeypatch.setattr(mammotion_services, "_attempt_ble_recovery", fail_if_called)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        ble_auto_recover=False,
        sample_delays=(0,),
    )

    assert result["ble_recovery"] is None
    assert "ble_transport_required" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"


@pytest.mark.asyncio
async def test_multi_segment_ble_auto_recovers_then_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-segment BLE auto-recovery promotes BLE so gates pass and segments run."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud"

    recovery_calls: list[object] = []

    async def fake_recover(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        recovery_calls.append(coordinator_arg)
        coordinator.active_transport_state = "ble"
        return {"attempted": True, "ok": True, "reason": "promoted", "steps": []}

    async def fake_vector(
        coordinator_arg: MammotionReportUpdateCoordinator,
        points: list[dict[str, float]],
        **kwargs: object,
    ) -> dict:
        return {
            "valid": True,
            "stop_reason": "target_reached",
            "blockers": [],
            "progress_diagnostics": [],
            "final_telemetry": _custom_path_telemetry_snapshot(coordinator),
            "vio": {"offset_degrees": 42.0},
        }

    monkeypatch.setattr(mammotion_services, "_attempt_ble_recovery", fake_recover)
    monkeypatch.setattr(
        mammotion_services, "_raw_pymammotion_execute_vector_segment", fake_vector
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_real_segments=1,
        sample_delays=(0,),
    )

    assert recovery_calls == [coordinator]
    assert result["ble_recovery"]["ok"] is True
    assert "ble_transport_required" not in result["blockers"]
    assert result["blockers"] == []
    assert result["stop_reason"] == "target_reached"


@pytest.mark.asyncio
async def test_multi_segment_ble_auto_recovery_failure_fails_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed multi-segment BLE recovery keeps ble_transport_required blocking."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud"

    async def fake_recover(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "attempted": True,
            "ok": False,
            "reason": "mower_not_advertising_needs_wake",
            "steps": ["ble_cooldown_active_waiting"],
        }

    monkeypatch.setattr(mammotion_services, "_attempt_ble_recovery", fake_recover)

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["ble_recovery"]["ok"] is False
    assert "ble_transport_required" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"
    coordinator.manager.send_command_with_args.assert_not_called()


def test_ble_connect_cooldown_active_reads_transport_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cooldown guard reflects the BLE transport's _connect_cooldown_until deadline."""
    coordinator = _pulse_coordinator()
    deadline = {"value": 0.0}

    def get_transport(_transport_type: object) -> SimpleNamespace:
        return SimpleNamespace(_connect_cooldown_until=deadline["value"])

    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        get_transport=get_transport
    )
    _patch_services_monotonic(monkeypatch, lambda: 1000.0)

    # No cooldown armed (0.0 deadline is in the past).
    assert _ble_connect_cooldown_active(coordinator) is False
    # Deadline in the future -> cooldown active.
    deadline["value"] = 1005.0
    assert _ble_connect_cooldown_active(coordinator) is True
    # Deadline already elapsed -> inactive again.
    deadline["value"] = 995.0
    assert _ble_connect_cooldown_active(coordinator) is False


@pytest.mark.asyncio
async def test_settle_feed_flags_stale_when_coordinates_bit_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bit-identical coordinates across polls read as a stale feed, not stillness."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def fake_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    before = _custom_path_telemetry_snapshot(coordinator)

    result = await _settle_linear_position_feed(coordinator, before)

    assert result["feed_stale"] is True
    assert result["observed_jitter"] is False
    assert result["settle_polls"] >= 3
    assert result["moved"] is False


@pytest.mark.asyncio
async def test_settle_feed_not_stale_when_live_feed_jitters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live feed's mm-level noise on a stationary mower is NOT staleness.

    This is the case that must stay distinguishable: during the 2026-07-19 e-stop
    the mower genuinely did not move, but the feed still jittered 2-4mm, which is
    what tells us the link is alive and the mower is the thing that stopped.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    tick = {"n": 0}

    async def fake_sleep(delay: float) -> None:
        return None

    async def jitter(count: int = 5) -> None:
        tick["n"] += 1
        # ~2mm of sensor noise, well under the 1cm settle epsilon.
        coordinator.data.mowing_state.pos_x = 1.0 + (0.002 if tick["n"] % 2 else 0.0)

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = jitter
    before = _custom_path_telemetry_snapshot(coordinator)

    result = await _settle_linear_position_feed(coordinator, before)

    assert result["observed_jitter"] is True
    assert result["feed_stale"] is False


def test_streak_shows_no_actuation_requires_both_sensors_flat() -> None:
    """Only a fully flat streak (no heading AND no position change) counts.

    Models the 2026-07-19 e-stop session: five commands returned ok with a
    dual-axis stop ACK, heading stayed bit-identical and displacement read
    0.25-0.43cm.
    """
    frozen = 91.38829636391407
    dead = [
        {
            "before_vision_heading": frozen,
            "after_vision_heading": frozen,
            "displacement_m": 0.0043,
        },
        {
            "before_vision_heading": frozen,
            "after_vision_heading": frozen,
            "displacement_m": 0.0025,
        },
    ]
    assert _streak_shows_no_actuation(dead, 2) is True

    # Dusk-latched VIO on a mower that may well be rotating: the feed still emits
    # sub-epsilon noise, so it is NOT bit-identical. Must stay no_heading_progress.
    latched_with_noise = [
        {
            "before_vision_heading": -49.836495,
            "after_vision_heading": -49.8382935,
            "displacement_m": 0.0,
        },
        {
            "before_vision_heading": -49.8382935,
            "after_vision_heading": -49.836495,
            "displacement_m": 0.0,
        },
    ]
    assert _streak_shows_no_actuation(latched_with_noise, 2) is False

    # Heading frozen but the mower demonstrably translated -> not "no actuation".
    latched_but_moving = [
        {
            "before_vision_heading": frozen,
            "after_vision_heading": frozen,
            "displacement_m": 0.081,
        },
        {
            "before_vision_heading": frozen,
            "after_vision_heading": frozen,
            "displacement_m": 0.074,
        },
    ]
    assert _streak_shows_no_actuation(latched_but_moving, 2) is False

    # Real rotation that simply isn't converging -> no_heading_progress.
    turning = [
        {
            "before_vision_heading": 0.0,
            "after_vision_heading": 8.2,
            "displacement_m": 0.001,
        },
        {
            "before_vision_heading": 8.2,
            "after_vision_heading": -7.3,
            "displacement_m": 0.002,
        },
    ]
    assert _streak_shows_no_actuation(turning, 2) is False

    # Unknown displacement cannot prove the mower stayed put.
    unknown = [
        {
            "before_vision_heading": frozen,
            "after_vision_heading": frozen,
            "displacement_m": None,
        },
        {
            "before_vision_heading": frozen,
            "after_vision_heading": frozen,
            "displacement_m": None,
        },
    ]
    assert _streak_shows_no_actuation(unknown, 2) is False

    # Not enough pulses yet.
    assert _streak_shows_no_actuation(dead[:1], 2) is False
    assert _streak_shows_no_actuation(dead, 0) is False


def test_streak_shows_dead_telemetry_requires_no_jitter_across_enough_polls() -> None:
    """Only bit-identical position across enough polls counts as a dead feed."""
    dead = [
        {"heading_poll_count": 16, "heading_poll_feed_alive": False},
        {"heading_poll_count": 16, "heading_poll_feed_alive": False},
    ]
    assert _streak_shows_dead_telemetry(dead, 2) is True

    # A live feed moves in some channel -- the mower may be stopped, but we can
    # see it, so the run must not be blamed on a dead link.
    alive = [
        {"heading_poll_count": 16, "heading_poll_feed_alive": False},
        {"heading_poll_count": 16, "heading_poll_feed_alive": True},
    ]
    assert _streak_shows_dead_telemetry(alive, 2) is False

    # One or two unchanged reads prove nothing (mirrors _STALE_FEED_MIN_POLLS).
    too_few = [
        {"heading_poll_count": 2, "heading_poll_feed_alive": False},
        {"heading_poll_count": 2, "heading_poll_feed_alive": False},
    ]
    assert _streak_shows_dead_telemetry(too_few, 2) is False

    # Pulses recorded before this instrumentation existed must not be read as
    # evidence of a dead feed.
    missing = [{"heading_poll_count": None}, {"heading_poll_count": None}]
    assert _streak_shows_dead_telemetry(missing, 2) is False

    assert _streak_shows_dead_telemetry(dead[:1], 2) is False
    assert _streak_shows_dead_telemetry(dead, 0) is False


def test_ble_transport_usable_reflects_transport_flag() -> None:
    """The usability probe mirrors BLETransport.is_usable."""
    coordinator = _pulse_coordinator()
    usable = {"value": True}

    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        get_transport=lambda _transport_type: SimpleNamespace(is_usable=usable["value"])
    )

    assert _ble_transport_usable(coordinator) is True
    usable["value"] = False
    assert _ble_transport_usable(coordinator) is False


def test_ble_transport_usable_defends_against_api_drift() -> None:
    """Anything unreadable degrades to "usable" (the old label-only behaviour)."""
    coordinator = _pulse_coordinator()

    # Handle without get_transport at all.
    coordinator.manager.mower = lambda _device_name: SimpleNamespace()
    assert _ble_transport_usable(coordinator) is True

    # Transport without the is_usable property (older pymammotion).
    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        get_transport=lambda _transport_type: SimpleNamespace()
    )
    assert _ble_transport_usable(coordinator) is True

    def raising_mower(_device_name: str) -> object:
        raise RuntimeError("handle unavailable")

    coordinator.manager.mower = raising_mower
    assert _ble_transport_usable(coordinator) is True


def test_ble_ready_for_motion_requires_label_and_usability() -> None:
    """BLE must be BOTH the active transport and actually usable.

    Regression for 2026-07-19: the mower reported active_transport "ble" while
    BLETransport.is_usable was False (advertisement lost). Motion commands
    returned command_ok with a dual-axis stop ACK and the mower never moved.
    """
    coordinator = _pulse_coordinator()
    usable = {"value": True}
    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        get_transport=lambda _transport_type: SimpleNamespace(is_usable=usable["value"])
    )

    coordinator.active_transport_state = "ble"
    assert _ble_ready_for_motion(coordinator) is True

    # The live failure: selected for routing, but cannot carry a command.
    usable["value"] = False
    assert _ble_ready_for_motion(coordinator) is False

    # Usable but not the active transport is still not ready.
    usable["value"] = True
    coordinator.active_transport_state = "cloud"
    assert _ble_ready_for_motion(coordinator) is False


@pytest.mark.asyncio
async def test_motion_gate_blocks_when_ble_selected_but_unusable() -> None:
    """ble_transport_required must fail on an unusable BLE transport.

    End-to-end regression for the 2026-07-19 live bug: before the fix the gate
    only compared the transport label, so real motion was dispatched onto a dead
    link -- five commands in a row silently no-opped while every health
    indicator read green.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "ble"
    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        last_report_at=123.0,
        availability=SimpleNamespace(mqtt_reported_offline=False),
        get_transport=lambda _transport_type: SimpleNamespace(
            _connect_cooldown_until=0.0,
            is_usable=False,
        ),
        active_transport=lambda: "ble",
    )

    result = await _vio_turn_probe(
        coordinator,
        angular_speed=500,
        drive_seconds=1.5,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert "ble_transport_required" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


def test_ble_connect_cooldown_active_defends_against_api_drift() -> None:
    """A handle without get_transport (or a raising one) reads as "no cooldown"."""
    coordinator = _pulse_coordinator()

    coordinator.manager.mower = lambda _device_name: SimpleNamespace()
    assert _ble_connect_cooldown_active(coordinator) is False

    def raising_mower(_device_name: str) -> object:
        raise RuntimeError("handle unavailable")

    coordinator.manager.mower = raising_mower
    assert _ble_connect_cooldown_active(coordinator) is False


def test_pinned_pymammotion_ble_transport_exposes_connect_cooldown_until() -> None:
    """Guard against pymammotion drift: the cooldown attr the guard reads must exist.

    _ble_connect_cooldown_active reads BLETransport._connect_cooldown_until; if a
    pymammotion bump renames or drops it the guard silently degrades to "never in
    cooldown", so pin the contract here.
    """
    transport = BLETransport(BLETransportConfig(device_id="test"))
    cooldown_attr = "_connect_cooldown_until"
    assert hasattr(transport, cooldown_attr)
    assert getattr(transport, cooldown_attr) == 0.0


def test_pinned_pymammotion_exposes_ble_liveness_fields() -> None:
    """Pin every field _ble_link_liveness reads against pymammotion drift.

    Two of them are private. If a bump renames them the helper would report
    "cannot read" -- which refuses motion rather than passing it, so drift is
    loud rather than silent. This test makes it loud at CI time instead.
    """
    transport = BLETransport(BLETransportConfig(device_id="test"))
    assert transport.is_connected is False
    assert transport.is_usable is False
    # Public on the Transport ABC; BLETransport.send() is its only writer.
    # It is an attempt timestamp (stamped before the awaited write), so the
    # confirmed-dispatch tests below—not this field—prove completion.
    assert transport.last_send_monotonic == 0.0

    queue = DeviceCommandQueue("test")
    assert queue.is_saga_active is False
    # Private on purpose -- pinning them here is the point of this test.
    assert queue._transport_gate.is_set() is True  # noqa: SLF001
    assert queue._queue.qsize() == 0  # noqa: SLF001


def test_ble_link_liveness_passes_on_a_healthy_link() -> None:
    """A connected transport with a drained queue and a recent send is live."""
    report = _ble_link_liveness(_pulse_coordinator())

    assert report["live"] is True
    assert report["reason"] is None
    assert report["queue_depth"] == 0
    assert report["queue_dispatch_paused"] is False
    assert report["last_send_age_seconds"] < _BLE_SEND_STALL_SECONDS


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"ble_connected": False}, "ble_client_not_connected"),
        ({"ble_usable": False}, "ble_transport_not_usable"),
        ({"ble_queue_paused": True}, "command_queue_dispatch_paused"),
        ({"ble_queue_depth": 21}, "command_queue_backlogged"),
        ({"ble_last_send_age": 30.0}, "ble_send_stalled"),
        ({"ble_last_send_age": None}, "no_ble_send_observed"),
    ],
)
def test_ble_link_liveness_names_the_failing_check(
    kwargs: dict[str, object], reason: str
) -> None:
    """Each stall signature is refused and reported by name."""
    report = _ble_link_liveness(_pulse_coordinator(**kwargs))

    assert report["live"] is False
    assert report["reason"] == reason


def test_ble_link_liveness_refuses_when_it_cannot_see() -> None:
    """Unreadable introspection must refuse, not pass.

    The inverse of _ble_transport_usable's deliberate permissiveness. A liveness
    gate that degrades to "live" is vacuously true -- the exact failure mode that
    has already bitten this project twice (zone_hash/bol_hash, _point_in_polygon).
    """
    coordinator = _pulse_coordinator()

    coordinator.manager.mower = lambda _device_name: None
    assert _ble_link_liveness(coordinator)["reason"] == "device_handle_unavailable"

    coordinator.manager.mower = lambda _device_name: SimpleNamespace()
    assert _ble_link_liveness(coordinator)["reason"] == "get_transport_unavailable"

    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        get_transport=lambda _transport_type: None
    )
    assert _ble_link_liveness(coordinator)["reason"] == "ble_transport_not_registered"

    def raising_mower(_device_name: str) -> object:
        raise RuntimeError("handle unavailable")

    coordinator.manager.mower = raising_mower
    assert _ble_link_liveness(coordinator)["live"] is False

    # A handle whose queue introspection is gone entirely: depth unknown, refuse.
    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        get_transport=lambda _transport_type: SimpleNamespace(
            _connect_cooldown_until=0.0,
            is_usable=True,
            is_connected=True,
            last_send_monotonic=time.monotonic(),
        ),
    )
    report = _ble_link_liveness(coordinator)
    assert report["live"] is False
    # Both queue reads are unavailable; reason names the first one checked.
    assert report["reason"] == "command_queue_dispatch_paused"
    assert report["queue_dispatch_paused"] is None
    assert report["queue_depth"] is None


@pytest.mark.asyncio
async def test_confirmed_ble_motion_waits_for_gatt_write() -> None:
    """A motion send does not complete merely because queue insertion succeeded."""
    coordinator = _pulse_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    release_write = asyncio.Event()

    async def blocked_write(_transport: object, _payload: bytes) -> None:
        await release_write.wait()

    handle._send_marked.side_effect = blocked_write  # noqa: SLF001
    send_task = asyncio.create_task(
        mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
            coordinator,
            "send_movement",
            command_kwargs={"linear_speed": 200, "angular_speed": 0},
        )
    )

    await asyncio.sleep(0)
    assert send_task.done() is False

    release_write.set()
    await send_task
    handle._send_marked.assert_awaited_once()  # noqa: SLF001


@pytest.mark.asyncio
async def test_normal_motion_teardown_stop_uses_emergency_queue_priority() -> None:
    """A bounded pulse's zero write bypasses normal queue work."""
    coordinator = _pulse_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    priorities: list[object] = []

    async def capture_priority(
        work: Callable[[], Coroutine[object, object, None]],
        priority: object = None,
        **_kwargs: object,
    ) -> None:
        priorities.append(priority)
        await work()

    handle.queue.enqueue = capture_priority

    result = await mammotion_services._manual_velocity_stop_attempt(  # noqa: SLF001
        coordinator, use_wifi=False
    )

    assert result["ok"] is True
    assert priorities == [Priority.EMERGENCY]


@pytest.mark.asyncio
async def test_confirmed_ble_motion_disarms_an_item_that_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-preflight queue stall cannot replay the motion item later."""
    coordinator = _pulse_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    queued_work: list[Callable[[], Coroutine[object, object, None]]] = []

    async def hold_in_queue(
        work: Callable[[], Coroutine[object, object, None]],
        **_kwargs: object,
    ) -> None:
        queued_work.append(work)

    handle.queue.enqueue = hold_in_queue
    monkeypatch.setattr(
        mammotion_services,
        "_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS",
        0.01,
    )

    with pytest.raises(RuntimeError, match="queued item was disarmed"):
        await mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
            coordinator,
            "send_movement",
            command_kwargs={"linear_speed": 200, "angular_speed": 0},
        )

    assert len(queued_work) == 1
    await queued_work[0]()
    handle._send_marked.assert_not_awaited()  # noqa: SLF001


@pytest.mark.asyncio
async def test_confirmed_ble_motion_disarms_on_cancellation_before_start() -> None:
    """Cancellation while queued cannot leave a command that executes later."""
    coordinator = _pulse_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    queued_work: list[Callable[[], Coroutine[object, object, None]]] = []

    async def hold_in_queue(
        work: Callable[[], Coroutine[object, object, None]],
        **_kwargs: object,
    ) -> None:
        queued_work.append(work)

    handle.queue.enqueue = hold_in_queue
    send_task = asyncio.create_task(
        mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
            coordinator,
            "send_movement",
            command_kwargs={"linear_speed": 200, "angular_speed": 0},
        )
    )
    await asyncio.sleep(0)
    send_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await send_task

    await queued_work[0]()
    handle._send_marked.assert_not_awaited()  # noqa: SLF001


@pytest.mark.asyncio
async def test_confirmed_ble_motion_stops_on_cancellation_during_write() -> None:
    """Cancellation after dispatch waits for the write and confirms a stop."""
    coordinator = _pulse_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    write_started = asyncio.Event()
    release_write = asyncio.Event()
    payloads: list[tuple[str, dict[str, object]]] = []
    queue_tasks: list[asyncio.Task[None]] = []

    async def enqueue_in_background(
        work: Callable[[], Coroutine[object, object, None]],
        **_kwargs: object,
    ) -> None:
        queue_tasks.append(asyncio.create_task(work()))

    async def block_first_write(
        _transport: object,
        payload: tuple[str, dict[str, object]],
    ) -> None:
        payloads.append(payload)
        if len(payloads) == 1:
            write_started.set()
            await release_write.wait()

    handle._send_marked.side_effect = block_first_write  # noqa: SLF001
    handle.queue.enqueue = enqueue_in_background
    send_task = asyncio.create_task(
        mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
            coordinator,
            "send_movement",
            command_kwargs={"linear_speed": 200, "angular_speed": 0},
        )
    )
    await write_started.wait()
    send_task.cancel()
    release_write.set()

    with pytest.raises(asyncio.CancelledError):
        await send_task

    assert payloads == [
        ("send_movement", {"linear_speed": 200, "angular_speed": 0}),
        ("send_movement", {"linear_speed": 0, "angular_speed": 0}),
    ]
    await asyncio.gather(*queue_tasks)


@pytest.mark.asyncio
async def test_motion_gate_blocks_a_stalled_command_queue() -> None:
    """End-to-end regression for the 2026-07-28 late-burst incident.

    A gated DeviceCommandQueue accumulates commands -- including the mandatory
    stop that bounds a pulse -- and flushes them as a burst tens of seconds
    later. On 2026-07-28 a command issued at 21:06:20 was reported as no
    actuation; the queue flushed at 21:06:41-43 and the mower drove 1.0778 m at
    21:07:16, unattended.

    Every pre-existing indicator read healthy through this: active_transport
    "ble", is_usable True, RSSI -64 dBm, command_result.ok True. Only the queue
    depth and the age of the last send discriminate.
    """
    coordinator = _pulse_coordinator(
        position=(1.0, 1.0, 0.0),
        ble_queue_depth=21,
        ble_last_send_age=21.0,
    )

    # The old gate is satisfied -- this is exactly why it did not catch it.
    assert _ble_ready_for_motion(coordinator) is True

    result = await _vio_turn_probe(
        coordinator,
        angular_speed=500,
        drive_seconds=1.5,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert "ble_link_live" in result["blockers"]
    assert "ble_transport_required" not in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_motion_gate_ble_link_live_is_skipped_on_dry_runs() -> None:
    """Dry runs send nothing, so they must not be blocked by link liveness."""
    result = await _vio_turn_probe(
        _pulse_coordinator(
            position=(1.0, 1.0, 0.0),
            ble_connected=False,
            ble_queue_depth=21,
            ble_last_send_age=None,
        ),
        angular_speed=500,
        drive_seconds=1.5,
        dry_run=True,
    )

    assert "ble_link_live" not in result["blockers"]


_OPPORTUNISTIC_BLE_RECONNECT = (
    mammotion_coordinator.MammotionReportUpdateCoordinator._async_opportunistic_ble_reconnect  # noqa: SLF001
)


def _reconnect_self(
    *,
    is_usable: bool = True,
    is_connected: bool = False,
    prefer_ble: bool = True,
    bluetooth_enabled: bool = True,
    connect: object = None,
    handle_present: bool = True,
) -> SimpleNamespace:
    """Build a minimal stand-in for the report coordinator."""
    if connect is None:

        async def connect() -> None:  # noqa: RUF029
            return None

    ble = SimpleNamespace(
        is_usable=is_usable, is_connected=is_connected, connect=connect
    )
    handle = SimpleNamespace(prefer_ble=prefer_ble, get_transport=lambda _t: ble)
    return SimpleNamespace(
        manager=SimpleNamespace(mower=lambda _name: handle if handle_present else None),
        device_name="Luba-Test",
        _bluetooth_enabled=bluetooth_enabled,
    )


@pytest.mark.asyncio
async def test_opportunistic_ble_reconnect_is_bounded_and_best_effort() -> None:
    """A hanging BLE reconnect must not stall the coordinator update.

    ``BLETransport.connect()`` is unbounded overall (self-managed scan at
    scan_timeout 10s, then establish_connection retries up to 4 times with
    backoff), and it runs inline in ``_async_update_data``. An unbounded call
    blocks the tick and every entity update behind it. The update is already
    served by cloud transport, so reconnecting is best-effort.
    """
    started = asyncio.Event()

    async def hanging_connect() -> None:
        started.set()
        await asyncio.sleep(3600)

    fake_self = _reconnect_self(connect=hanging_connect)

    with patch.object(mammotion_coordinator, "_BLE_RECONNECT_TIMEOUT_SECONDS", 0.05):
        # Must return rather than hang, and must not raise.
        await asyncio.wait_for(_OPPORTUNISTIC_BLE_RECONNECT(fake_self), timeout=5)

    assert started.is_set()


@pytest.mark.asyncio
async def test_opportunistic_ble_reconnect_skips_when_not_applicable() -> None:
    """No connect attempt when BLE is unusable, already connected, or disabled."""
    calls: list[str] = []

    async def record_connect() -> None:
        calls.append("connect")

    # Unusable transport (e.g. armed connect cooldown) -> leave it alone.
    await _OPPORTUNISTIC_BLE_RECONNECT(
        _reconnect_self(is_usable=False, connect=record_connect)
    )
    # Already connected -> nothing to do.
    await _OPPORTUNISTIC_BLE_RECONNECT(
        _reconnect_self(is_connected=True, connect=record_connect)
    )
    # Bluetooth switched off by the user -> respect it.
    await _OPPORTUNISTIC_BLE_RECONNECT(
        _reconnect_self(bluetooth_enabled=False, connect=record_connect)
    )
    # prefer_ble off -> user opted out of BLE routing.
    await _OPPORTUNISTIC_BLE_RECONNECT(
        _reconnect_self(prefer_ble=False, connect=record_connect)
    )
    # No handle at all.
    await _OPPORTUNISTIC_BLE_RECONNECT(
        _reconnect_self(handle_present=False, connect=record_connect)
    )

    assert calls == []


@pytest.mark.asyncio
async def test_opportunistic_ble_reconnect_connects_when_usable() -> None:
    """The whole point: a usable-but-disconnected transport gets reconnected."""
    calls: list[str] = []

    async def record_connect() -> None:
        calls.append("connect")

    await _OPPORTUNISTIC_BLE_RECONNECT(_reconnect_self(connect=record_connect))

    assert calls == ["connect"]
