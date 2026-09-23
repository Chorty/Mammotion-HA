"""Tests for turning connected-trace rows into per-proxy coverage samples."""

from __future__ import annotations

from scripts.build_ble_coverage_map import bin_samples, normalize_trace_rows


def _row(
    x: float | None, y: float | None, rssi: int | None, proxy: str = "p1s-printer"
) -> dict:
    """One row in the exact shape ble_proxy_connected_trace.py writes."""
    return {
        "t_utc": "2026-09-16T19:00:00Z",
        "x": x,
        "y": y,
        "proxy": proxy,
        "ble_rssi": rssi,
    }


def test_trace_rows_reach_the_map() -> None:
    """The tracer writes ble_rssi; before the fix every row was dropped."""
    kept, dropped = normalize_trace_rows([_row(4.9, -3.8, -72), _row(4.9, -3.6, -74)])
    cells = bin_samples(kept, "trace")
    assert sum(cell["n"] for cell in cells) == 2
    assert cells[0]["proxy"] == "p1s-printer"
    assert dropped == {"unattributed": 0, "no_rssi": 0, "repeat": 0, "no_position": 0}


def test_unattributed_dozed_and_unpositioned_rows_are_dropped_and_counted() -> None:
    """Every drop is attributed to a reason, never silently discarded."""
    rows = [
        _row(4.9, -3.8, -72, proxy="?"),
        _row(4.9, -3.7, -72, proxy="disconnected"),
        _row(4.9, -3.6, 0),
        _row(4.9, -3.5, None),
        _row(None, -3.4, -70),
        _row(4.9, -3.3, -70),
    ]
    kept, dropped = normalize_trace_rows(rows)
    assert len(kept) == 1
    assert dropped == {"unattributed": 2, "no_rssi": 2, "repeat": 0, "no_position": 1}


def test_a_re_read_of_the_same_report_does_not_inflate_n() -> None:
    """Polling faster than the ~1 Hz bundle must not count one report twice."""
    rows = [
        _row(4.9, -3.8, -72),
        _row(4.9, -3.8, -72),
        _row(4.9, -3.8, -72),
        _row(4.9, -3.8, -74),
    ]
    kept, dropped = normalize_trace_rows(rows)
    assert len(kept) == 2
    assert dropped["repeat"] == 2
    # Stationary but a changed RSSI is a fresh report and counts.
    assert [row["rssi"] for row in kept] == [-72, -74]
