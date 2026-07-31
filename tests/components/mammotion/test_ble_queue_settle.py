"""Tests for the BLE command-queue settle wait in the motion preflight.

Regression cover for a deterministic self-block found on hardware 2026-07-30:
every motion executor starts the dense report stream first, that start *is* a
BLE command, and the ``ble_link_live`` gate then demanded an empty command
queue -- so the executor refused its own dispatch. Two consecutive real
``experimental_execute_segment`` calls were rejected with
``command_queue_backlogged`` / ``queue_depth: 1`` on a fully healthy link,
while twenty idle samples of the same gate reported live.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from custom_components.mammotion import services
from custom_components.mammotion.coordinator import MammotionBaseUpdateCoordinator


def _report(reason: str | None, *, live: bool) -> dict[str, Any]:
    """Build a liveness report shaped like ``_ble_link_liveness`` returns."""
    return {"live": live, "reason": reason, "queue_depth": 0 if live else 1}


@pytest.mark.asyncio
async def test_idle_manual_motion_requests_a_bounded_continuous_report_stream() -> None:
    """Manual motion must explicitly request count=0 reports while idle."""
    coordinator = type(
        "CoordinatorStub",
        (),
        {"manager": AsyncMock(), "device_name": "test-mower"},
    )()

    await MammotionBaseUpdateCoordinator.async_start_continuous_reports(
        coordinator, duration_ms=12_345
    )

    coordinator.manager.request_reports.assert_awaited_once_with(
        "test-mower", count=0, timeout=12_345
    )


@pytest.mark.asyncio
async def test_a_transient_backlog_is_waited_out(monkeypatch) -> None:
    """A queue that drains shortly after the stream start must not refuse."""
    reports = [
        _report("command_queue_backlogged", live=False),
        _report("command_queue_backlogged", live=False),
        _report(None, live=True),
    ]
    seen: list[int] = []

    def fake_liveness(_coordinator: Any) -> dict[str, Any]:
        seen.append(len(seen))
        return reports[min(len(seen) - 1, len(reports) - 1)]

    monkeypatch.setattr(services, "_ble_link_liveness", fake_liveness)
    monkeypatch.setattr(services, "_BLE_QUEUE_SETTLE_POLL_SECONDS", 0.0)

    result = await services._settle_ble_command_queue(object())  # noqa: SLF001

    assert result["live"] is True
    assert len(seen) == 3


@pytest.mark.asyncio
async def test_a_persistent_backlog_still_fails_the_gate(monkeypatch) -> None:
    """Waiting must never turn a real backlog into a pass."""
    monkeypatch.setattr(
        services,
        "_ble_link_liveness",
        lambda _c: _report("command_queue_backlogged", live=False),
    )
    monkeypatch.setattr(services, "_BLE_QUEUE_SETTLE_POLL_SECONDS", 0.0)
    monkeypatch.setattr(services, "_BLE_QUEUE_SETTLE_TIMEOUT_SECONDS", 0.05)

    result = await services._settle_ble_command_queue(object())  # noqa: SLF001

    assert result["live"] is False
    assert result["reason"] == "command_queue_backlogged"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "ble_client_not_connected",
        "ble_transport_not_usable",
        "ble_connect_cooldown_armed",
        "exclusive_saga_active",
        "ble_send_stalled",
    ],
)
async def test_standing_failures_return_immediately(monkeypatch, reason) -> None:
    """Only queue states are transient; waiting cannot fix a dead link.

    A standing condition must not burn the settle timeout before refusing --
    the operator gets the real reason straight away.
    """
    calls: list[int] = []

    def fake_liveness(_coordinator: Any) -> dict[str, Any]:
        calls.append(1)
        return _report(reason, live=False)

    monkeypatch.setattr(services, "_ble_link_liveness", fake_liveness)

    result = await services._settle_ble_command_queue(object())  # noqa: SLF001

    assert result["live"] is False
    assert result["reason"] == reason
    # Sampled once and returned -- no polling loop entered.
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_an_already_live_link_is_not_delayed(monkeypatch) -> None:
    """The common case costs exactly one read and no sleeping."""
    calls: list[int] = []

    def fake_liveness(_coordinator: Any) -> dict[str, Any]:
        calls.append(1)
        return _report(None, live=True)

    monkeypatch.setattr(services, "_ble_link_liveness", fake_liveness)

    result = await services._settle_ble_command_queue(object())  # noqa: SLF001

    assert result["live"] is True
    assert len(calls) == 1
