"""Tests for the experimental manual-motion safety boundary."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.mammotion import manual_motion
from custom_components.mammotion.manual_motion import (
    ManualMotionCancelledError,
    ManualMotionSession,
    assert_session_can_dispatch,
    experimental_motion_status,
    motion_backend_verified,
)
from custom_components.mammotion.services import _stop_active_manual_motion


def _coordinator(*, enabled: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        config_entry=SimpleNamespace(options={"enable_experimental_motion": enabled}),
        manual_motion_session=None,
        last_manual_motion_session=None,
    )


def test_backend_floor_keeps_pymammotion_0812_locked() -> None:
    """The currently pinned backend must not unlock real motion."""
    assert motion_backend_verified("0.8.12") is False
    assert motion_backend_verified("not-installed") is False
    assert motion_backend_verified("0.8.13") is True


def test_status_is_fail_closed_and_reports_actionable_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing opt-in, backend audit, BLE, and runtime safety all block."""
    monkeypatch.setattr(
        manual_motion,
        "installed_pymammotion_version",
        lambda: "0.8.12",
    )
    status = experimental_motion_status(
        _coordinator(),
        ble_liveness={"live": False, "reason": "command_queue_backlogged"},
        safety={
            "allowed_for_manual_motion": False,
            "blockers": ["blade_reported_on"],
        },
    )

    assert status["real_motion_allowed"] is False
    assert status["blockers"] == [
        "experimental_motion_disabled",
        "pymammotion_backend_unverified",
        "command_queue_backlogged",
        "blade_reported_on",
    ]
    assert status["real_click_to_go_segment_limit"] == 2


def test_cancelled_session_blocks_nonzero_but_never_stop() -> None:
    """An abort becomes visible before any later nonzero dispatch."""
    coordinator = _coordinator(enabled=True)
    session = ManualMotionSession(owner="test")
    session.cancelled = True
    coordinator.manual_motion_session = session

    with pytest.raises(ManualMotionCancelledError):
        assert_session_can_dispatch(coordinator, is_stop=False)
    assert_session_can_dispatch(coordinator, is_stop=True)


@pytest.mark.asyncio
async def test_stop_marks_cancelled_before_three_confirmed_writes_and_waits_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public stop contract aborts first, then confirms a bounded sequence."""
    coordinator = _coordinator(enabled=True)
    session = ManualMotionSession(owner="test")
    session.phase = "running"
    coordinator.manual_motion_session = session
    observations: list[tuple[bool, bool]] = []

    async def fake_stop(_coordinator: object, *, emergency: bool = False) -> dict:
        observations.append((session.cancelled, emergency))
        return {"movement_ok": True}

    monkeypatch.setattr(
        "custom_components.mammotion.services._stop_manual_motion_confirmed",
        fake_stop,
    )

    async def finish_owner() -> None:
        await asyncio.sleep(0)
        session.owner_done.set()

    owner = asyncio.create_task(finish_owner())
    result = await _stop_active_manual_motion(coordinator)
    await owner

    assert observations == [(True, True), (True, True), (True, True)]
    assert result["stop_confirmed"] is True
    assert result["all_stop_writes_confirmed"] is True
    assert result["owner_exited"] is True
    assert session.cancel_reason == "operator_stop"
    assert session.stop_result is result


@pytest.mark.asyncio
async def test_stop_reports_partial_failure_without_hiding_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One successful write stops the mower but the degraded sequence is visible."""
    coordinator = _coordinator()
    stop = AsyncMock(
        side_effect=[
            TimeoutError("first"),
            {"movement_ok": True},
            TimeoutError("third"),
        ]
    )
    monkeypatch.setattr(
        "custom_components.mammotion.services._stop_manual_motion_confirmed",
        stop,
    )

    result = await _stop_active_manual_motion(coordinator)

    assert result["stop_confirmed"] is True
    assert result["all_stop_writes_confirmed"] is False
    assert [attempt["ok"] for attempt in result["attempts"]] == [
        False,
        True,
        False,
    ]
