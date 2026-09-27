"""Motion dispatch against pymammotion's real DeviceCommandQueue, not a stand-in.

On 2026-09-27 beta118 (pymammotion 0.9.6.post1) sent a real forward pulse and
then failed its stop with ``EMERGENCY is a direct-send priority and must not be
queued``. The mower halted only on its own refresh watchdog. Every test here had
passed, because the fixture queue accepted any priority. These pin the contract
against the real queue: an emergency stop is written without ever entering it,
and ordinary motion still goes through it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pymammotion.messaging.command_queue import DeviceCommandQueue, Priority

from custom_components.mammotion import services as mammotion_services

from .conftest import _pulse_coordinator

_STOP = {"linear_speed": 0, "angular_speed": 0}


def _with_real_queue() -> tuple[Any, Any, DeviceCommandQueue]:
    coordinator = _pulse_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    queue = DeviceCommandQueue(coordinator.device_name)
    handle.queue = queue
    return coordinator, handle, queue


@pytest.mark.asyncio
async def test_the_real_queue_refuses_emergency() -> None:
    """The upstream contract this integration must route around."""
    queue = DeviceCommandQueue("Luba-Test")

    async def work() -> None:
        return None

    with pytest.raises(ValueError, match="direct-send priority"):
        await queue.enqueue(work, priority=Priority.EMERGENCY)


@pytest.mark.asyncio
async def test_an_emergency_stop_is_written_without_the_queue_running() -> None:
    """The queue processor is never started, so anything queued would never run."""
    coordinator, handle, queue = _with_real_queue()

    await mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
        coordinator, "send_movement", command_kwargs=_STOP, emergency_stop=True
    )

    handle._send_marked.assert_awaited_once()  # noqa: SLF001
    assert queue._queue.qsize() == 0  # noqa: SLF001


@pytest.mark.asyncio
async def test_the_operator_stop_path_reaches_the_mower() -> None:
    """The stop behind the card's Stop, a pulse's own stop and executor aborts."""
    coordinator, handle, _queue = _with_real_queue()

    result = await mammotion_services._manual_velocity_stop_attempt(  # noqa: SLF001
        coordinator, use_wifi=False
    )

    assert result["ok"] is True, result
    handle._send_marked.assert_awaited_once()  # noqa: SLF001


@pytest.mark.asyncio
async def test_ordinary_motion_still_goes_through_the_real_queue() -> None:
    """Only the stop bypasses it; a movement keeps its queue ordering."""
    coordinator, handle, queue = _with_real_queue()
    queue.start()
    try:
        await mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
            coordinator,
            "send_movement",
            command_kwargs={"linear_speed": 200, "angular_speed": 0},
        )
    finally:
        await queue.stop()

    handle._send_marked.assert_awaited_once()  # noqa: SLF001


@pytest.mark.asyncio
async def test_cancelling_the_caller_does_not_abort_a_stop_mid_write() -> None:
    """The stop is its own task, so a cancelled caller cannot cut the write short."""
    coordinator, handle, _queue = _with_real_queue()
    write_started = asyncio.Event()
    release_write = asyncio.Event()
    writes_finished: list[bool] = []

    async def slow_write(*_args: object, **_kwargs: object) -> None:
        write_started.set()
        await release_write.wait()
        writes_finished.append(True)

    handle._send_marked = slow_write  # noqa: SLF001
    caller = asyncio.create_task(
        mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
            coordinator, "send_movement", command_kwargs=_STOP, emergency_stop=True
        )
    )
    # Bounded: on a regression the write never starts, and that must fail, not hang.
    await asyncio.wait_for(write_started.wait(), timeout=2)
    caller.cancel()
    release_write.set()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert writes_finished == [True]
