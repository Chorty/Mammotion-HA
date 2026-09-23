"""Offline tests for isolated position-cadence matrix classification."""

from scripts.position_cadence_matrix import PERIODS_MS, REPEATS, classify_matrix


def _cell(period: int, *, p95_ms: float, gaps: int = 0) -> dict:
    """Return one complete isolated position-payload cell."""
    return {
        "isolated": True,
        "period_ms": period,
        "no_change_period_ms": period,
        "position_payloads": {
            "observed": 100,
            "dropped_samples": 0,
            "sequence_gaps": gaps,
            "p95_interval_ms": p95_ms,
        },
    }


def test_three_complete_repeats_classify_each_period() -> None:
    """Classification uses all three complete cells for every requested period."""
    runs = [
        _cell(period, p95_ms=period * 1.4)
        for period in PERIODS_MS
        for _ in range(REPEATS)
    ]
    result = classify_matrix(runs)
    assert result["complete"] is True
    assert all(cell["honoured"] is True for cell in result["periods"].values())


def test_gap_or_missing_repeat_refuses_a_period_classification() -> None:
    """A dropped sample or incomplete repeat set leaves classification unknown."""
    runs = [_cell(1000, p95_ms=1000.0), _cell(1000, p95_ms=1000.0, gaps=1)]
    result = classify_matrix(runs)
    period = result["periods"]["1000"]
    assert period["honoured"] is None
    assert "three_repeats_required" in period["blockers"]
    assert "repeat_2_evidence_gap" in period["blockers"]
