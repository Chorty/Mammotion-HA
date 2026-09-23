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

from .conftest import _pulse_coordinator


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


@pytest.mark.asyncio
async def test_the_vector_segment_executor_settles_before_it_gates(
    monkeypatch,
) -> None:
    """The multi-segment reach regression, pinned at the executor.

    Live 2026-08-09, the first four-segment run: segment 1 reached its waypoint,
    and segment 2 was refused PRE-DISPATCH with ``command_queue_backlogged`` /
    ``queue_depth: 1`` just 1.5 s after segment 1's last send -- on a link that
    was connected, usable, uncooled and dispatching. The offending queue entry
    was segment 1's own trailing stop, still draining. Segments 3 and 4 never
    ran, so the run answered nothing it was designed to answer. Evidence:
    ``docs/evidence-beta32-4segment-20260809T170941Z.json``.

    Five other executors already awaited ``_settle_ble_command_queue`` before
    evaluating their gates; the vector segment executor did not. The old
    2-segment reach limit made the boundary rare enough to hide it.

    This asserts the ORDER -- settle strictly before the gate is judged --
    because that ordering is the whole fix. A settle placed after the gate would
    leave every assertion about liveness true and still refuse the segment.
    """
    order: list[str] = []

    async def fake_settle(_coordinator: Any) -> dict[str, Any]:
        order.append("settle")
        return _report(None, live=True)

    def fake_gates(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        order.append("gates")
        return [{"name": "ble_link_live", "passed": True, "detail": "stub"}]

    monkeypatch.setattr(services, "_settle_ble_command_queue", fake_settle)
    monkeypatch.setattr(services, "_manual_velocity_pulse_gates", fake_gates)

    result = await services._raw_pymammotion_execute_vector_segment(  # noqa: SLF001
        _pulse_coordinator(position=(1.0, 1.0, 0.0)),
        [{"x": 0.0, "y": 0.0}, {"x": 0.9, "y": 0.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert order[:2] == ["settle", "gates"], (
        "the queue must drain BEFORE ble_link_live is judged, or a segment can "
        "be refused on a command the previous segment queued"
    )
    assert result["queue_settle"] == _report(None, live=True)


@pytest.mark.asyncio
async def test_a_dry_run_segment_never_waits_on_the_queue(monkeypatch) -> None:
    """A dry run enqueues nothing, so it must stay instant and not settle."""
    called: list[int] = []

    async def fake_settle(_coordinator: Any) -> dict[str, Any]:
        called.append(1)
        return _report(None, live=True)

    monkeypatch.setattr(services, "_settle_ble_command_queue", fake_settle)

    result = await services._raw_pymammotion_execute_vector_segment(  # noqa: SLF001
        _pulse_coordinator(position=(1.0, 1.0, 0.0)),
        [{"x": 0.0, "y": 0.0}, {"x": 0.9, "y": 0.0}],
        dry_run=True,
    )

    assert called == []
    assert result["queue_settle"] is None


@pytest.mark.asyncio
async def test_refresh_cadence_absorbs_a_slow_write_instead_of_adding_to_it(
    monkeypatch,
) -> None:
    """Refreshes must fire on a fixed cadence, not one interval after each write.

    Motion continues only while refresh writes keep arriving, so the gap between
    commands REACHING the mower is what matters. Sleeping a full interval after
    awaiting each write makes that gap `interval + write_duration`.

    That is not academic on this link. Across all 98 refresh writes of the five
    real runs of 2026-08-09 the latency was p50 225.6 / p95 1029.2 / max 2014.0
    ms, with 59% of writes over the 200 ms interval -- so sleep-then-await put a
    median ~426 ms between commands against a 200 ms design, and the tail
    starved the device watchdog outright (a 1303.972 ms write consumed an entire
    1303.7 ms pulse).

    Simulated here with a write that costs a full interval: on a fixed schedule
    the second fire is already due when the first write returns, so the wait is
    zero and the cadence holds. The old behaviour would have slept another full
    interval on top.
    """
    clock = {"t": 0.0}
    interval, write_cost = 0.2, 0.2
    fires: list[float] = []

    async def fake_sleep(_coordinator: Any, seconds: float) -> None:
        clock["t"] += seconds

    async def fake_resend() -> None:
        fires.append(clock["t"])
        clock["t"] += write_cost

    monkeypatch.setattr(services.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(services, "_motion_open_sleep", fake_sleep)

    report = await services._motion_refresh_window(  # noqa: SLF001
        object(),
        resend=fake_resend,
        duration_seconds=1.0,
        refresh_interval_ms=int(interval * 1000),
    )

    assert report["refresh_commands_sent"] == len(fires)
    assert fires, "no refresh ever fired"
    # Fires track the fixed schedule (0.2, 0.4, ...) rather than drifting out by
    # the write cost each time (which would give 0.2, 0.6, 1.0, ...).
    for index, fired_at in enumerate(fires):
        assert fired_at == pytest.approx(interval * (index + 1), abs=1e-6)
