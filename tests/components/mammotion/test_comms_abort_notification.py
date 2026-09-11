"""Tests for the comms-loss abort notification (option B) and its diagnostics.

Two changes are covered here, both from 2026-09-11:

* ``_notify_motion_comms_abort`` -- surfacing a comms abort to the operator
  instead of returning it as a string nobody reads
  (``docs/design-comms-loss-recovery-20260910.md``).
* the ``queue_diagnostics`` capture extended to the helper *phases* of
  ``_raw_pymammotion_execute_vector_segment``. The 2026-09-10 edit covered the
  linear phase only, so a leg dying in the calibration drive or a turn -- which
  is where legs 4, 7 and 8 died -- still produced no queue snapshot.
"""

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.services import (
    EVENT_MOTION_COMMS_ABORT,
    _comms_abort_reason,
    _describe_abort_position,
    _last_queue_diagnostics,
    _notify_motion_comms_abort,
    _vio_segment_calibration_drive,
)

from .conftest import _pulse_coordinator


class _RecordingHass:
    """A hass double that records what the notifier did, without a real loop."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.service_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.bus = SimpleNamespace(async_fire=self._async_fire)
        self.services = SimpleNamespace(async_call=self._async_call)

    def _async_fire(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))

    async def _async_call(
        self,
        domain: str,
        service: str,
        data: dict[str, Any],
        **_kwargs: object,
    ) -> None:
        self.service_calls.append((domain, service, data))

    def async_create_task(self, coro: Any) -> None:
        # Run the coroutine to completion synchronously; it only appends to a
        # list, so there is nothing to await on. Closing instead of running
        # would leave the recorded call missing.
        with contextlib.suppress(StopIteration):
            coro.send(None)


def _call(entity_id: str = "lawn_mower.test") -> SimpleNamespace:
    return SimpleNamespace(data={"entity_id": entity_id})


# --------------------------------------------------------------------------
# _comms_abort_reason
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"stop_reason": "command_failed"}, "command_failed"),
        ({"stop_reason": "stop_failed_aborting"}, "stop_failed_aborting"),
        # The VIO calibration drive reports `reason`, not `stop_reason`, for the
        # identical condition -- reading only one key misses a whole phase.
        ({"reason": "command_failed"}, "command_failed"),
        ({"reason": "stop_failed_aborting"}, "stop_failed_aborting"),
        ({"stop_reason": "target_reached"}, None),
        # An operator-commanded stop is not a comms failure.
        ({"stop_reason": "operator_stop"}, None),
        ({"reason": "position_unavailable"}, None),
        ({}, None),
        ({"stop_reason": None}, None),
    ],
)
def test_comms_abort_reason_reads_both_keys(
    result: dict[str, Any], expected: str | None
) -> None:
    """Both reason keys count, and only the two comms reasons match."""
    assert _comms_abort_reason(result) == expected


def test_command_failed_is_in_scope_because_legs_7_and_8_returned_it() -> None:
    """🔑 The two legs that aborted the 2026-09-10 series were `command_failed`.

    Scoping the notification to `stop_failed_aborting` alone -- as the design
    doc's own wording did -- would have stayed silent for exactly the pair that
    stopped the series, and fired only for leg 4.
    """
    assert "command_failed" in mammotion_services._COMMS_ABORT_REASONS  # noqa: SLF001
    assert (
        "stop_failed_aborting" in mammotion_services._COMMS_ABORT_REASONS  # noqa: SLF001
    )


# --------------------------------------------------------------------------
# _last_queue_diagnostics / _describe_abort_position
# --------------------------------------------------------------------------


def test_last_queue_diagnostics_takes_the_most_recent_capture() -> None:
    """The snapshot that matters is the one taken at the refusal, not earlier."""
    result = {
        "command_results": [
            {"index": 1, "queue_diagnostics": {"queue_depth": 0}},
            {"index": 2},
            {"index": 3, "queue_diagnostics": {"queue_depth": 4}},
        ]
    }
    assert _last_queue_diagnostics(result) == {"queue_depth": 4}


@pytest.mark.parametrize(
    "result",
    [{}, {"command_results": []}, {"command_results": "not-a-list"}, {"x": 1}],
)
def test_last_queue_diagnostics_is_none_when_nothing_was_captured(
    result: dict[str, Any],
) -> None:
    """A result with no snapshot must not raise; the notification just says so."""
    assert _last_queue_diagnostics(result) is None


def test_describe_abort_position_renders_coordinates() -> None:
    """The operator needs somewhere to walk to, so x/y lead the description."""
    described = _describe_abort_position(
        {"final_telemetry": {"position": {"x": 6.62, "y": -9.18, "toward": 1234}}}
    )
    assert "x=6.62" in described
    assert "y=-9.18" in described


@pytest.mark.parametrize(
    "result",
    [{}, {"final_telemetry": None}, {"final_telemetry": {"position": None}}],
)
def test_describe_abort_position_degrades_to_unknown(result: dict[str, Any]) -> None:
    """Missing telemetry is reported as unknown rather than raising."""
    assert "unknown" in _describe_abort_position(result)


# --------------------------------------------------------------------------
# _notify_motion_comms_abort
# --------------------------------------------------------------------------


def test_notify_fires_event_and_creates_notification() -> None:
    """An abort reaches the operator two ways: an event and a notification."""
    hass = _RecordingHass()
    result = {
        "stop_reason": "command_failed",
        "final_telemetry": {"position": {"x": 1.5, "y": 2.5, "toward": 900}},
        "command_results": [
            {"index": 1, "queue_diagnostics": {"queue_depth": 2, "is_connected": True}}
        ],
    }

    _notify_motion_comms_abort(
        hass, "raw_pymammotion_execute_vector_segment", _call(), result
    )  # type: ignore[arg-type]

    assert len(hass.events) == 1
    event_type, payload = hass.events[0]
    assert event_type == EVENT_MOTION_COMMS_ABORT
    assert payload["reason"] == "command_failed"
    assert payload["entity_id"] == "lawn_mower.test"
    assert payload["queue_diagnostics"] == {"queue_depth": 2, "is_connected": True}

    assert len(hass.service_calls) == 1
    domain, service, data = hass.service_calls[0]
    assert (domain, service) == ("persistent_notification", "create")
    assert "command_failed" in data["message"]
    assert "x=1.5" in data["message"]
    assert "queue_depth=2" in data["message"]
    # Notify-only: the operator is told, never driven at.
    assert "No further command was sent" in data["message"]


@pytest.mark.parametrize(
    "result",
    [
        {"stop_reason": "target_reached"},
        {"stop_reason": "operator_stop"},
        {"reason": "position_unavailable"},
        {},
    ],
)
def test_notify_is_silent_for_anything_that_is_not_a_comms_abort(
    result: dict[str, Any],
) -> None:
    """A normal landing must not notify, or the signal becomes noise."""
    hass = _RecordingHass()
    _notify_motion_comms_abort(hass, "svc", _call(), result)  # type: ignore[arg-type]
    assert hass.events == []
    assert hass.service_calls == []


def test_notify_never_lets_a_notification_failure_break_the_result() -> None:
    """🚨 A completed motion result must survive a broken notification path.

    The notifier runs on the wrapper's return path, so an exception here would
    convert a finished run's result into a raised error -- losing the telemetry
    the operator needs precisely when something has already gone wrong.
    """
    broken = SimpleNamespace(
        bus=SimpleNamespace(
            async_fire=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no bus"))
        )
    )
    _notify_motion_comms_abort(
        broken, "svc", _call(), {"stop_reason": "command_failed"}
    )  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the instrumentation extension
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calibration_drive_captures_queue_diagnostics_on_command_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🔑 A leg dying in the calibration drive now carries a queue snapshot.

    This is the phase the 2026-09-10 instrumentation did not reach: it is the
    vector executor's FIRST real motion, so it is the earliest point a leg can
    die on a queue-start timeout, and before this change it produced
    `reason: command_failed` with nothing to diagnose it from.
    """
    coordinator = _pulse_coordinator(ble_queue_depth=3)

    async def _refuse(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(
            "BLE motion command did not start before the queue deadline; "
            "the queued item was disarmed"
        )

    monkeypatch.setattr(mammotion_services, "_send_manager_command_with_args", _refuse)

    result = await _vio_segment_calibration_drive(coordinator)

    assert result["reason"] == "command_failed"
    diagnostics = result["command_results"][-1]["queue_diagnostics"]
    assert diagnostics is not None
    assert "queue_depth" in diagnostics
    # And the notifier can find it through the same accessor the wrapper uses.
    assert _last_queue_diagnostics(result) is diagnostics
