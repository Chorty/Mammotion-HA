"""Tests for the Agora WebSocket reconnect machinery.

These require the full dev environment (``uv sync``) because importing the
handler pulls in Home Assistant and the websocket stack; they skip cleanly
where those aren't installed.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("homeassistant")
pytest.importorskip("websockets")
pytest.importorskip("sdp_transform")

from custom_components.mammotion.agora_websocket import (  # noqa: E402
    AgoraWebSocketHandler,
)


def _make_handler() -> AgoraWebSocketHandler:
    """Create a handler with a hass stub that schedules real asyncio tasks."""
    loop = asyncio.get_running_loop()
    hass = SimpleNamespace(async_create_task=loop.create_task)
    return AgoraWebSocketHandler(hass)


async def test_schedule_reconnect_is_single_flight(monkeypatch: Any) -> None:
    """A second trigger while a reconnect is running must not stack tasks."""
    handler = _make_handler()
    started = 0
    release = asyncio.Event()

    async def fake_loop() -> None:
        nonlocal started
        started += 1
        await release.wait()

    monkeypatch.setattr(handler, "_reconnect_loop", fake_loop)
    handler._schedule_reconnect()
    handler._schedule_reconnect()
    await asyncio.sleep(0)
    assert started == 1
    release.set()
    await handler._reconnect_task


async def test_schedule_reconnect_noop_when_stopped(monkeypatch: Any) -> None:
    """After an intentional disconnect no reconnect may be scheduled."""
    handler = _make_handler()
    handler._stopped = True
    handler._schedule_reconnect()
    assert handler._reconnect_task is None


async def test_reconnect_backoff_and_success(monkeypatch: Any) -> None:
    """Delays grow exponentially (capped) and success ends the loop."""
    handler = _make_handler()
    delays: list[float] = []
    attempts = 0

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def fake_restart() -> bool:
        nonlocal attempts
        attempts += 1
        return attempts == 4  # succeed on the 4th attempt

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        "custom_components.mammotion.agora_websocket.random.uniform",
        lambda _a, _b: 1.0,
    )
    monkeypatch.setattr(handler, "_restart_websocket", fake_restart)

    await handler._reconnect_loop()

    assert attempts == 4
    assert delays == [1.0, 2.0, 4.0, 8.0]


async def test_reconnect_gives_up_after_max_attempts(monkeypatch: Any) -> None:
    """The loop stops after RECONNECT_MAX_ATTEMPTS failures."""
    handler = _make_handler()
    delays: list[float] = []
    attempts = 0

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def fake_restart() -> bool:
        nonlocal attempts
        attempts += 1
        return False

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        "custom_components.mammotion.agora_websocket.random.uniform",
        lambda _a, _b: 1.0,
    )
    monkeypatch.setattr(handler, "_restart_websocket", fake_restart)

    await handler._reconnect_loop()

    assert attempts == handler.RECONNECT_MAX_ATTEMPTS
    # Exponential growth capped at RECONNECT_MAX_BACKOFF_SECS.
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]


async def test_failed_attempt_clears_rejoin_token(monkeypatch: Any) -> None:
    """A failed rejoin falls back to a full join on the next attempt."""
    handler = _make_handler()
    handler._rejoin_token = "stale-token"
    attempts = 0

    async def fake_sleep(_delay: float) -> None:
        return

    async def fake_restart() -> bool:
        nonlocal attempts
        attempts += 1
        return attempts == 2

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(handler, "_restart_websocket", fake_restart)

    await handler._reconnect_loop()
    assert handler._rejoin_token is None


async def test_disconnect_cancels_reconnect_and_latches(monkeypatch: Any) -> None:
    """disconnect() stops an in-flight reconnect and prevents new ones."""
    handler = _make_handler()
    release = asyncio.Event()

    async def fake_loop() -> None:
        await release.wait()

    monkeypatch.setattr(handler, "_reconnect_loop", fake_loop)
    handler._schedule_reconnect()
    task = handler._reconnect_task
    assert task is not None

    await handler.disconnect()
    assert handler._stopped is True
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()

    handler._schedule_reconnect()
    assert handler._reconnect_task is None


async def test_stopped_checked_between_sleep_and_attempt(monkeypatch: Any) -> None:
    """The loop aborts without a restart attempt if stopped mid-sleep."""
    handler = _make_handler()
    attempts = 0

    async def fake_sleep(_delay: float) -> None:
        handler._stopped = True

    async def fake_restart() -> bool:
        nonlocal attempts
        attempts += 1
        return True

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(handler, "_restart_websocket", fake_restart)

    await handler._reconnect_loop()
    assert attempts == 0
