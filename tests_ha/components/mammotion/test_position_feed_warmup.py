"""The vector executor must not move until the position feed is demonstrably live.

On 2026-10-03 the first pulse of a scored click-to-go leg (the 2.0 s VIO
calibration drive) ran with no position report at all: an idle mower does not
stream, and the executor only requested reports AFTER that pulse stopped. These
are the falsifiers predeclared in Amendment 6 of
docs/predeclared-backend-096-clicktogo-leg-20260928.md.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.services import (
    _raw_pymammotion_execute_vector_segment,
    _warm_position_feed,
)

from .conftest import _pulse_coordinator

_REAL_RUN = {
    "dry_run": False,
    "confirm_blades_off": True,
    "confirm_clear_area": True,
    "sample_delays": (0,),
}
_LEG = [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 2.0}]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)


def _vio_coordinator() -> SimpleNamespace:
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )
    return coordinator


def _motion_sends(coordinator: SimpleNamespace) -> list[dict]:
    """Every non-zero send_movement that reached the transport."""
    return [
        call.kwargs
        for call in coordinator.manager.send_command_with_args.await_args_list
        if call.args[1:2] == ("send_movement",)
    ]


async def test_first_motion_command_waits_for_a_new_position_report() -> None:
    """(a) No motion command goes out before a report newer than the baseline."""
    coordinator = _vio_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    # A cached sample from long before the run: re-reading it is not evidence.
    handle.latest_position_sample = SimpleNamespace(
        sequence=478, epoch=1, received_at_monotonic=0.0, valid_for_motion=True
    )
    seen_at_first_send: list[int | None] = []

    async def record_send(*_args: object, **_kwargs: object) -> None:
        if not seen_at_first_send:
            sample = handle.latest_position_sample
            seen_at_first_send.append(getattr(sample, "sequence", None))

    coordinator.manager.send_command_with_args.side_effect = record_send

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator, _LEG, **_REAL_RUN
    )

    assert seen_at_first_send, "the run never sent a motion command"
    assert seen_at_first_send[0] is not None
    assert seen_at_first_send[0] > 478
    warmup = result["position_feed_warmup"]
    assert warmup["ok"] is True
    assert warmup["baseline_sequence"] == 478
    assert warmup["fresh_sequence"] == 479
    assert warmup["fresh_epoch"] == warmup["baseline_epoch"] == 1


@pytest.mark.parametrize("cached", [None, 478])
async def test_no_new_report_refuses_with_nothing_sent(cached: int | None) -> None:
    """(b) A silent feed refuses `position_feed_not_live`; nothing moves or stops."""
    coordinator = _vio_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    if cached is not None:
        handle.latest_position_sample = SimpleNamespace(
            sequence=cached, epoch=1, received_at_monotonic=0.0, valid_for_motion=True
        )
    coordinator.async_get_reports.side_effect = None  # the request goes unanswered

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator, _LEG, **_REAL_RUN
    )

    assert result["stop_reason"] == "position_feed_not_live"
    assert result["commands_sent"] == 0
    assert result["position_feed_warmup"]["reason"] == "no_fresh_position_report"
    assert _motion_sends(coordinator) == []
    coordinator.async_stop_manual_motion.assert_not_awaited()
    coordinator.async_get_reports.assert_awaited_once_with(count=5)


async def test_report_from_a_replaced_link_is_not_fresh() -> None:
    """(c) A newer sample in a different transport epoch does not count."""
    coordinator = _vio_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)

    async def reconnect_then_report(*_args: object, **_kwargs: object) -> None:
        handle.position_epoch += 1  # BLE dropped and came back
        handle.latest_position_sample = SimpleNamespace(
            sequence=1,
            epoch=handle.position_epoch,
            received_at_monotonic=0.0,
            valid_for_motion=True,
        )

    coordinator.async_get_reports.side_effect = reconnect_then_report

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator, _LEG, **_REAL_RUN
    )

    assert result["stop_reason"] == "position_feed_not_live"
    assert result["position_feed_warmup"]["reason"] == "position_epoch_changed"
    assert result["commands_sent"] == 0
    assert _motion_sends(coordinator) == []


async def test_dry_run_requests_no_reports_and_sends_nothing() -> None:
    """(d) A dry run stays send-free: no report request, no motion."""
    coordinator = _vio_coordinator()

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        _LEG,
        dry_run=True,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert "position_feed_warmup" not in result
    coordinator.async_get_reports.assert_not_awaited()
    assert _motion_sends(coordinator) == []


async def test_warmup_reports_unavailable_stream_without_requesting() -> None:
    """No handle epoch means no way to judge freshness: refuse before asking."""
    coordinator = _vio_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    handle.position_epoch = None

    warmup = await _warm_position_feed(coordinator)

    assert warmup["ok"] is False
    assert warmup["reason"] == "position_stream_unavailable"
    coordinator.async_get_reports.assert_not_awaited()


async def test_warmup_reports_a_failed_request() -> None:
    """A report request that raises is a refusal with the error kept."""
    coordinator = _vio_coordinator()
    coordinator.async_get_reports = AsyncMock(side_effect=TimeoutError("queue"))

    warmup = await _warm_position_feed(coordinator)

    assert warmup["ok"] is False
    assert warmup["reason"] == "report_request_failed"
    assert warmup["error"] == "TimeoutError: queue"
