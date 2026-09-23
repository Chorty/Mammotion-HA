"""Pins for the Phase 1 repeat scorer."""  # noqa: INP001

from __future__ import annotations

import datetime
import json
from pathlib import Path

from scripts.score_queue_measurement import (
    axis1,
    classify_leg,
    expected_segment_sizes,
    load_runner_session,
    pct,
    score,
    stop_delimited_segments,
    verdict,
)

_T0 = datetime.datetime(2026, 9, 14, 12, 0, tzinfo=datetime.UTC)


def _sample(
    ms: int, *, stop: bool = False, budget: float = 2.0, wait: float = 10.0
) -> dict:
    """Build a timing sample recorded ``ms`` milliseconds after a fixed origin."""
    return {
        "recorded_at_utc": (_T0 + datetime.timedelta(milliseconds=ms)).isoformat(),
        "command": "send_movement",
        "is_stop": stop,
        "emergency_stop": stop and budget == 5.0,
        "outcome": "completed",
        "queue_wait_ms": wait,
        "queue_budget_seconds": budget,
        "started": True,
        "write_ms": 150.0,
    }


def _linear(refreshes: int) -> dict:
    """Build a linear pulse record as the executor reports it."""
    return {
        "phase": "linear_forward_to_target",
        "motion_refresh": {"refresh_commands_sent": refreshes},
    }


def _turn(refreshes: int) -> dict:
    """Build a turn pulse record as the executor reports it."""
    return {
        "heading_error_before": -60.0,
        "motion_refresh": {"refresh_commands_sent": refreshes},
    }


_CALIBRATION = {"phase": "vio_calibration_drive", "feedback_refresh": {"ok": True}}


def _leg_samples(
    sizes: list[int], *, stop_budgets: list[float] | None = None, start_ms: int = 0
) -> list[dict]:
    """Build motion segments of the given sizes, each followed by one stop sample."""
    samples, t = [], start_ms
    for index, size in enumerate(sizes):
        for _ in range(size):
            samples.append(_sample(t))
            t += 900  # deliberately wider than the withdrawn 500 ms rule
        budget = stop_budgets[index] if stop_budgets else 5.0
        samples.append(_sample(t, stop=True, budget=budget))
        t += 1500
    return samples


def _write_session(
    out: Path, *, drop_oldest_after_first_leg: bool = False
) -> list[dict]:
    """Write a synthetic runner output directory and return the scored leg's samples.

    S1 is a scored leg, S2 a refusal that sent nothing, S3 a setup leg. The
    history also holds one sample recorded before the session started.
    """
    before = _sample(-10_000)
    s1 = _leg_samples([1, 3])
    s3 = _leg_samples([1, 3], start_ms=100_000)
    after_s1 = [before, *s1]
    after_s2 = s1 if drop_oldest_after_first_leg else after_s1
    files = {
        "meta_S1.json": {
            "label": "S1",
            "role": "scored",
            "dispatched_utc": _T0.isoformat(),
        },
        "meta_S2.json": {
            "label": "S2",
            "role": "scored",
            "dispatched_utc": (_T0 + datetime.timedelta(seconds=30)).isoformat(),
        },
        "meta_S3.json": {
            "label": "S3",
            "role": "setup",
            "dispatched_utc": (_T0 + datetime.timedelta(seconds=90)).isoformat(),
        },
        "leg_S1.json": {"command_results": [_CALIBRATION, _linear(2)]},
        "leg_S2.json": {"would_send": False, "blockers": ["ble_client_not_connected"]},
        "leg_S3.json": {"command_results": [_CALIBRATION, _linear(2)]},
        "timing_after_S1.json": {"samples": after_s1},
        "timing_after_S2.json": {"samples": after_s2},
        "timing_after_S3.json": {"samples": [*after_s2, *s3]},
    }
    for name, content in files.items():
        (out / name).write_text(json.dumps(content))
    return s1


def test_reproduces_67_of_67_on_the_20260913_evidence() -> None:
    """Repeat section 2.7/2.8: the committed rule reproduces the design evidence.

    Classification only. The 2026-09-13 session is INCONCLUSIVE and is never
    rescored, so no axis is evaluated and no verdict is drawn here.
    """
    evidence = json.loads(
        Path("docs/evidence-queue-timeout-measurement-20260913.json").read_text()
    )
    raw = json.loads(Path("docs/evidence-phase1-legs-20260913.json").read_text())[
        "executor_results"
    ]

    legs = [
        classify_leg(leg["label"], leg["samples"], raw[leg["label"]]["command_results"])
        for leg in evidence["legs"]
    ]

    assert len(legs) == 13
    assert sum(leg["expected_pulses"] for leg in legs) == 67
    assert sum(len(leg["pulse_open"]) for leg in legs) == 67
    assert sum(leg["unclassifiable_segments"] for leg in legs) == 0
    assert all(leg["aligned"] and leg["reconciled"] for leg in legs)


def test_turn_stops_with_budget_2_are_delimiters_not_samples() -> None:
    """Section 2.2: filtering stops on budget 5.0 alone was wrong."""
    samples = _leg_samples([1, 5, 8], stop_budgets=[5.0, 2.0, 2.0])

    leg = classify_leg("T", samples, [_CALIBRATION, _turn(4), _turn(7)])

    assert leg["segment_sizes"] == [1, 5, 8]
    assert leg["unclassifiable_segments"] == 0
    assert not any(s["is_stop"] for s in leg["pulse_open"] + leg["refresh"])


def test_calibration_drive_expects_zero_refreshes() -> None:
    """The calibration drive carries no motion_refresh; its segment is one sample."""
    assert expected_segment_sizes([_CALIBRATION, _linear(6), _turn(3)]) == [1, 7, 4]


def test_wide_gaps_inside_a_pulse_do_not_split_it() -> None:
    """Section 2.1: no timing gap is used, so 900 ms within a pulse is fine."""
    segments, trailing = stop_delimited_segments(_leg_samples([1, 7]))

    assert [len(s) for s in segments] == [1, 7]
    assert trailing == 0


def test_segment_size_mismatch_is_unclassifiable_and_never_redrawn() -> None:
    """Section 2.4: a segment that differs from the executor count is excluded."""
    leg = classify_leg(
        "M", _leg_samples([1, 6, 4]), [_CALIBRATION, _linear(6), _linear(3)]
    )

    assert [s["classifiable"] for s in leg["segments"]] == [True, False, True]
    assert leg["unclassifiable_segments"] == 1
    assert len(leg["pulse_open"]) == 2


def test_segment_count_mismatch_makes_every_segment_unclassifiable() -> None:
    """Section 2.5: a missing stop merges pulses; the leg is not re-paired by hand."""
    samples = _leg_samples([1, 7, 4])
    second_stop = [s for s in samples if s["is_stop"]][1]
    merged = [s for s in samples if s is not second_stop]

    leg = classify_leg("C", merged, [_CALIBRATION, _linear(6), _linear(3)])

    assert leg["segment_sizes"] == [1, 11]
    assert not leg["aligned"]
    assert leg["reconciled"]
    assert leg["unclassifiable_segments"] == 3
    assert leg["pulse_open"] == []


def test_motion_after_the_last_stop_makes_the_leg_unclassifiable() -> None:
    """Section 2.5: trailing motion means a stop never recorded."""
    samples = _leg_samples([1, 7])
    samples.append(_sample(60_000))

    leg = classify_leg("R", samples, [_CALIBRATION, _linear(6)])

    assert leg["motion_after_last_stop"] == 1
    assert not leg["aligned"]
    assert not leg["reconciled"]
    assert leg["unclassifiable_segments"] == 2


def test_unreconcilable_is_defined_by_leg_totals_and_fails_axis_one() -> None:
    """Section 2.5 defines parent 11.1 clause 4 by the motion-sample total."""
    good = classify_leg(
        "G", _leg_samples([1, 7, 7, 7, 7, 7, 7]), [_CALIBRATION, *([_linear(6)] * 6)]
    )
    bad = classify_leg("B", _leg_samples([1, 6]), [_CALIBRATION, _linear(6)])

    result = axis1(
        [good, bad], truncated=False, rf_changed=False, history_dropped=False
    )

    assert good["reconciled"]
    assert not bad["reconciled"]
    assert result["checks"]["no_unreconcilable_leg"] == {"value": ["B"], "pass": False}
    assert not result["valid"]


def test_pct_matches_the_integration_index_rule() -> None:
    """Same order statistic as _summarise_motion_dispatch_timings."""
    values = [float(v) for v in range(1, 41)]

    assert pct(values, 0.95) == 38.0  # round(0.95 * 39) = 37 -> the 38th value
    assert pct([5.0, 1.0, 3.0], 0.5) == 3.0
    assert pct([], 0.95) is None


def test_verdict_follows_the_fixed_evaluation_order() -> None:
    """Parent 13.2: 1, 1b, 2, 3, then everything else is inconclusive."""
    assert verdict(80.0, 0.15, 1.2, 0, 0) == "1_constant_fine"
    assert verdict(80.0, 1.0, 1.2, 0, 1) == "1b_constant_fine_strongest_form"
    assert verdict(80.0, 0.55, 1.2, 0, 0) == "4_inconclusive"
    assert verdict(1200.0, 0.9, 1.4, 3, 1) == "2_write_latency_binds"
    assert verdict(1200.0, 0.9, 2.0, 2, 1) == "3_raise_defensible"
    assert verdict(1200.0, 0.9, 2.0, 1, 1) == "4_inconclusive"
    assert verdict(1200.0, 0.9, 2.0, 2, 0) == "4_inconclusive"
    assert verdict(600.0, 0.3, 1.1, 0, 0) == "4_inconclusive"


def test_axis_two_is_not_evaluated_when_axis_one_fails() -> None:
    """Parent section 5: an inconclusive session's numbers are not interpreted."""
    leg = classify_leg("S", _leg_samples([1, 7]), [_CALIBRATION, _linear(6)])

    result = score([leg], [], truncated=False, rf_changed=False, history_dropped=False)

    assert not result["axis1"]["valid"]
    assert result["axis2"] is None
    assert result["result"] == "INCONCLUSIVE (axis 1)"


def test_rf_change_or_truncation_alone_makes_the_session_inconclusive() -> None:
    """Repeat section 3 and parent 13.2: facts the operator asserts still gate."""
    legs = [
        classify_leg(
            f"L{i}",
            _leg_samples([1, 7, 7, 7, 7, 7, 7, 7, 7, 7]),
            [_CALIBRATION, *([_linear(6)] * 9)],
        )
        for i in range(5)
    ]

    clean = axis1(legs, truncated=False, rf_changed=False, history_dropped=False)
    moved = axis1(legs, truncated=False, rf_changed=True, history_dropped=False)
    cut = axis1(legs, truncated=True, rf_changed=False, history_dropped=False)

    assert clean["valid"]
    assert not moved["valid"]
    assert not cut["valid"]


def test_loader_excludes_pre_session_refusals_and_setup_legs(tmp_path: Path) -> None:
    """Repeat section 4: population is scored legs' samples from the session only."""
    s1 = _write_session(tmp_path)

    legs, session, dropped, notes = load_runner_session(tmp_path, _T0.isoformat())

    assert [leg["label"] for leg in legs] == ["S1"]
    assert sorted(s["recorded_at_utc"] for s in session) == sorted(
        s["recorded_at_utc"] for s in s1
    )
    assert legs[0]["aligned"]
    assert legs[0]["unclassifiable_segments"] == 0
    assert not dropped
    assert [note.split(":")[0] for note in notes] == ["S2", "S3"]


def test_loader_detects_a_history_capacity_drop(tmp_path: Path) -> None:
    """Parent section 5: a sample vanishing between snapshots is a detected drop."""
    _write_session(tmp_path, drop_oldest_after_first_leg=True)

    _, _, dropped, _ = load_runner_session(tmp_path, _T0.isoformat())

    assert dropped
