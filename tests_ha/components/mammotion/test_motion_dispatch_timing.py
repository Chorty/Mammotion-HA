"""Tests for the queue-start timing instrument (issue 1 step 2, 2026-09-11).

The bound these measure -- ``_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS = 2.0`` --
was **unfalsifiable from telemetry** before this: a refusal recorded only that
the wait exceeded 2.0 s, and a success recorded nothing at all. That censoring
is why `docs/plan-post-20260910-session-issues.md` issue 1 forbids changing the
number until it is measured. These tests pin the measurement, not the number:
nothing here asserts what the constant should be.
"""

from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.services import (
    _record_motion_dispatch_timing,
    _summarise_motion_dispatch_timings,
)

from .conftest import _pulse_coordinator


def _timing_coordinator(**kwargs: Any) -> SimpleNamespace:
    """Build a pulse coordinator that carries the rolling timing history."""
    coordinator = _pulse_coordinator(**kwargs)
    coordinator.motion_dispatch_timings = deque(maxlen=500)
    return coordinator


# --------------------------------------------------------------------------
# the recorder
# --------------------------------------------------------------------------


def test_recorder_appends_and_stamps_the_sample() -> None:
    """Each dispatch appends one timestamped sample."""
    coordinator = _timing_coordinator()
    sample = _record_motion_dispatch_timing(
        coordinator, outcome="completed", queue_wait_ms=12.5
    )
    assert sample["outcome"] == "completed"
    assert "recorded_at_utc" in sample
    assert list(coordinator.motion_dispatch_timings) == [sample]


def test_recorder_never_raises_without_the_history_attribute() -> None:
    """🚨 Instrumentation must not be able to fail a dispatch.

    A coordinator predating this field (or a test double) must be skipped
    silently -- recording a measurement is never worth aborting real motion for.
    """
    bare = SimpleNamespace()
    sample = _record_motion_dispatch_timing(bare, outcome="completed")
    assert sample["outcome"] == "completed"
    assert not hasattr(bare, "motion_dispatch_timings")


def test_history_is_bounded() -> None:
    """A long session cannot grow the history without limit."""
    coordinator = _pulse_coordinator()
    coordinator.motion_dispatch_timings = deque(maxlen=3)
    for index in range(10):
        _record_motion_dispatch_timing(coordinator, outcome="completed", index=index)
    assert [s["index"] for s in coordinator.motion_dispatch_timings] == [7, 8, 9]


# --------------------------------------------------------------------------
# the summary
# --------------------------------------------------------------------------


def test_summary_reports_the_distribution_and_headroom() -> None:
    """🔑 The whole point: how much headroom the 2.0 s bound actually has."""
    coordinator = _timing_coordinator()
    for wait in (10.0, 20.0, 30.0, 40.0, 1900.0):
        _record_motion_dispatch_timing(
            coordinator,
            outcome="completed",
            queue_wait_ms=wait,
            queue_budget_seconds=2.0,
        )
    report = _summarise_motion_dispatch_timings(coordinator)
    assert report["sample_count"] == 5
    assert report["queue_wait_ms"]["min"] == 10.0
    assert report["queue_wait_ms"]["max"] == 1900.0
    assert report["queue_wait_ms"]["p50"] == 30.0
    assert report["outcomes"] == {"completed": 5}
    # 1900 ms against a 2000 ms budget -- 95% consumed, i.e. marginal.
    assert report["worst_wait_fraction_of_budget"] == pytest.approx(0.95)
    # The constant is reported, never asserted as a target.
    assert report["queue_start_timeout_seconds"] == (
        mammotion_services._BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS  # noqa: SLF001
    )


def test_summary_is_empty_but_valid_before_any_dispatch() -> None:
    """A fresh coordinator reports zero samples rather than raising."""
    report = _summarise_motion_dispatch_timings(_timing_coordinator())
    assert report["sample_count"] == 0
    assert report["queue_wait_ms"]["p95"] is None
    assert report["worst_wait_fraction_of_budget"] is None


def test_summary_tolerates_a_coordinator_without_history() -> None:
    """Read-only diagnostics must degrade, not fail."""
    assert _summarise_motion_dispatch_timings(SimpleNamespace())["sample_count"] == 0


# --------------------------------------------------------------------------
# the dispatch path -- the part that removes the censoring
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_successful_dispatch_records_its_queue_wait() -> None:
    """🔑 This is the half that did not exist before.

    A pulse that starts fine now records how long it waited, so the population
    of normal waits is observable instead of invisible.
    """
    coordinator = _timing_coordinator()

    await mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
        coordinator,
        "send_movement",
        command_kwargs={"linear_speed": 200, "angular_speed": 0},
    )

    samples = list(coordinator.motion_dispatch_timings)
    assert len(samples) == 1
    sample = samples[0]
    assert sample["outcome"] == "completed"
    assert sample["started"] is True
    assert sample["queue_wait_ms"] >= 0.0
    assert sample["write_ms"] >= 0.0
    assert sample["queue_budget_seconds"] == (
        mammotion_services._BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS  # noqa: SLF001
    )


@pytest.mark.asyncio
async def test_a_queue_start_timeout_records_the_wait_it_gave_up_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal is recorded as a measurement, not just an exception."""
    coordinator = _timing_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)

    async def hold_in_queue(work: object, **_kwargs: object) -> None:
        del work  # never started: models a queue occupied by other traffic

    handle.queue.enqueue = hold_in_queue
    monkeypatch.setattr(
        mammotion_services, "_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS", 0.01
    )

    with pytest.raises(RuntimeError, match="queued item was disarmed"):
        await mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
            coordinator,
            "send_movement",
            command_kwargs={"linear_speed": 200, "angular_speed": 0},
        )

    samples = list(coordinator.motion_dispatch_timings)
    assert len(samples) == 1
    sample = samples[0]
    assert sample["outcome"] == "queue_start_timeout"
    assert sample["started"] is False
    assert sample["write_ms"] is None
    # It waited at least the (patched) budget before giving up.
    assert sample["queue_wait_ms"] >= 10.0
    assert sample["queue_budget_seconds"] == 0.01


@pytest.mark.asyncio
async def test_both_outcomes_land_in_one_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Success and refusal share a history, which is what makes it a rate."""
    coordinator = _timing_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    good_enqueue = handle.queue.enqueue

    await mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
        coordinator,
        "send_movement",
        command_kwargs={"linear_speed": 200, "angular_speed": 0},
    )

    async def hold_in_queue(work: object, **_kwargs: object) -> None:
        del work

    handle.queue.enqueue = hold_in_queue
    monkeypatch.setattr(
        mammotion_services, "_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS", 0.01
    )
    with pytest.raises(RuntimeError):
        await mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
            coordinator,
            "send_movement",
            command_kwargs={"linear_speed": 200, "angular_speed": 0},
        )
    handle.queue.enqueue = good_enqueue

    report = _summarise_motion_dispatch_timings(coordinator)
    assert report["sample_count"] == 2
    assert report["outcomes"] == {"completed": 1, "queue_start_timeout": 1}
    assert report["queue_wait_ms"]["count"] == 2
