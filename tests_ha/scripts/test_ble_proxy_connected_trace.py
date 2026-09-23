"""Tests for attributing the mower's connection to a proxy."""

from __future__ import annotations

from scripts.ble_proxy_connected_trace import (
    MOWER_MAC,
    CurrentProxy,
    _handle_allocation_event,
)

SCANNERS = {
    "DC:54:75:C8:C8:86": "atom-fireplace",
    "78:21:84:9B:C1:D6": "garage-m5stack",
}


def _event(allocations: list[dict]) -> dict:
    return {"type": "event", "id": 2, "event": allocations}


def test_an_unrelated_scanner_update_does_not_disconnect_the_mower() -> None:
    """Captured shape, 2026-09-16: a full snapshot, then one-scanner deltas."""
    current = CurrentProxy()
    _handle_allocation_event(
        _event(
            [
                {"source": "DC:54:75:C8:C8:86", "allocated": [MOWER_MAC]},
                {"source": "78:21:84:9B:C1:D6", "allocated": ["5C:02:72:9E:1B:32"]},
            ]
        ),
        current,
        SCANNERS,
    )
    assert current.name == "atom-fireplace"
    _handle_allocation_event(
        _event([{"source": "78:21:84:9B:C1:D6", "allocated": []}]), current, SCANNERS
    )
    assert current.name == "atom-fireplace"


def test_a_real_release_and_a_handoff_are_both_seen() -> None:
    """The mower leaving its own proxy is a disconnect; appearing elsewhere is a handoff."""
    current = CurrentProxy()
    _handle_allocation_event(
        _event([{"source": "DC:54:75:C8:C8:86", "allocated": [MOWER_MAC]}]),
        current,
        SCANNERS,
    )
    _handle_allocation_event(
        _event([{"source": "DC:54:75:C8:C8:86", "allocated": []}]), current, SCANNERS
    )
    assert current.name == "disconnected"
    _handle_allocation_event(
        _event([{"source": "78:21:84:9B:C1:D6", "allocated": [MOWER_MAC]}]),
        current,
        SCANNERS,
    )
    assert current.name == "garage-m5stack"
