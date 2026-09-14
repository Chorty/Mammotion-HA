#!/usr/bin/env python3
"""Score a Phase 1 queue-timing session under the repeat predeclaration.

Implements ``docs/predeclared-queue-timeout-measurement-repeat-20260913.md``:
the stop-delimited classifier of its section 2, and the axis 1 / axis 2
evaluation it inherits unchanged from
``docs/predeclared-queue-timeout-measurement-20260911.md``.

🔑 **Committed before the repeat's first dispatch (section 2.8).** Scoring the
repeat must use this file as committed. A change made after repeat data exists
is exactly the failure the predeclaration forbids.

🛑 **The 2026-09-13 session is INCONCLUSIVE and is never rescored with this
code.** The test suite uses its evidence only to reproduce the classification
(67/67 pulses), never to evaluate an axis or draw a verdict.

Reads the runner's output directory (``leg_*.json``, ``meta_*.json``,
``timing_after_*.json``). Sends nothing; standard library only.

Usage:
    scripts/score_queue_measurement.py RUNNER_OUT --session-start ISO --truncated {yes,no} --rf-changed {yes,no} [--json OUT]
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

Sample = dict[str, Any]

#: Parent section 13.1 / 11.4: verdict bar, surviving exclusion.
MIN_PULSE_OPEN = 40
MIN_LEGS = 4
MIN_TURN_OR_CALIBRATION_LEGS = 2
#: Parent section 11.1 clause 1.
PULSE_OPEN_SHARE_MIN = 0.08
PULSE_OPEN_SHARE_MAX = 0.35
#: Repeat section 2.6 (denominator: expected pulses).
MAX_UNCLASSIFIABLE_SHARE = 0.20
#: Parent sections 3, 4 and 12.1, restated over the pulse_open class (10.2).
Q_FINE_MS = 250.0
Q_RAISE_MS = 1000.0
FRACTION_FINE = 0.40
FRACTION_RAISE = 0.75
RATIO_WRITE_BINDS = 1.5
ORDINARY_BUDGET_SECONDS = 2.0
#: Parent section 10.4(B).
RATE_CLAIM_N = 120


def pct(values: list[float], fraction: float) -> float | None:
    """Percentile exactly as ``_summarise_motion_dispatch_timings`` computes it."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return round(ordered[index], 3)


def _recorded_at(sample: Sample) -> datetime.datetime:
    """Parse a sample's microsecond-resolution UTC stamp."""
    return datetime.datetime.fromisoformat(sample["recorded_at_utc"])


def _number(value: Any) -> float | None:
    """Return a float for a real number, else None (booleans excluded)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def expected_segment_sizes(command_results: list[dict[str, Any]]) -> list[int]:
    """Section 2.4: one pulse-open plus the executor's own refresh count, per pulse.

    The calibration drive carries no ``motion_refresh``; its count is 0.
    """
    return [
        1 + int((result.get("motion_refresh") or {}).get("refresh_commands_sent") or 0)
        for result in command_results
    ]


def has_turn_or_calibration(command_results: list[dict[str, Any]]) -> bool:
    """Whether a leg ran a turn or calibration-drive phase (parent section 2)."""
    return any(
        result.get("phase") == "vio_calibration_drive"
        or "heading_error_before" in result
        for result in command_results
    )


def stop_delimited_segments(samples: list[Sample]) -> tuple[list[list[Sample]], int]:
    """Section 2.2-2.3: split a leg's samples at ``is_stop`` samples.

    Returns (segments, motion samples left after the final stop). Stop samples
    are delimiters and belong to neither class, whatever their budget.
    """
    segments: list[list[Sample]] = []
    current: list[Sample] = []
    for sample in sorted(samples, key=_recorded_at):
        if sample.get("is_stop"):
            segments.append(current)
            current = []
        else:
            current.append(sample)
    return segments, len(current)


def classify_leg(
    label: str, samples: list[Sample], command_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Classify one dispatched leg under repeat section 2."""
    expected = expected_segment_sizes(command_results)
    segments, trailing = stop_delimited_segments(samples)
    motion_total = sum(len(segment) for segment in segments) + trailing
    aligned = len(segments) == len(expected) and trailing == 0

    records: list[dict[str, Any]] = []
    pulse_open: list[Sample] = []
    refresh: list[Sample] = []
    for index, segment in enumerate(segments):
        want = expected[index] if index < len(expected) else None
        classifiable = aligned and len(segment) == want
        records.append(
            {"segment": index, "size": len(segment), "expected": want, "classifiable": classifiable}
        )
        if classifiable:
            pulse_open.append(segment[0])
            refresh.extend(segment[1:])

    unclassifiable_segments = len(expected) - len(pulse_open)
    return {
        "label": label,
        "expected_pulses": len(expected),
        "expected_segment_sizes": expected,
        "segment_sizes": [len(segment) for segment in segments],
        "motion_after_last_stop": trailing,
        "aligned": aligned,
        "reconciled": motion_total == sum(expected),
        "has_turn_or_calibration": has_turn_or_calibration(command_results),
        "segments": records,
        "pulse_open": pulse_open,
        "refresh": refresh,
        "unclassifiable_segments": unclassifiable_segments,
        "unclassifiable_samples": motion_total - len(pulse_open) - len(refresh),
    }


def axis1(
    legs: list[dict[str, Any]],
    *,
    truncated: bool,
    rf_changed: bool,
    history_dropped: bool,
) -> dict[str, Any]:
    """Parent section 13.2 axis 1, with repeat sections 2.5, 2.6 and 3."""
    expected = sum(leg["expected_pulses"] for leg in legs)
    unclassifiable = sum(leg["unclassifiable_segments"] for leg in legs)
    n_pulse_open = sum(len(leg["pulse_open"]) for leg in legs)
    classified = sum(len(leg["pulse_open"]) + len(leg["refresh"]) for leg in legs)
    contributing = [leg for leg in legs if leg["pulse_open"]]
    turn_legs = sum(1 for leg in contributing if leg["has_turn_or_calibration"])
    share = n_pulse_open / classified if classified else 0.0
    unclassifiable_share = unclassifiable / expected if expected else 1.0
    unreconciled = [leg["label"] for leg in legs if not leg["reconciled"]]

    checks = {
        "pulse_open_surviving_exclusion_ge_40": (n_pulse_open, n_pulse_open >= MIN_PULSE_OPEN),
        "contributing_legs_ge_4": (len(contributing), len(contributing) >= MIN_LEGS),
        "legs_with_turn_or_calibration_ge_2": (turn_legs, turn_legs >= MIN_TURN_OR_CALIBRATION_LEGS),
        "pulse_open_share_8_to_35_pct": (
            round(share, 4),
            PULSE_OPEN_SHARE_MIN <= share <= PULSE_OPEN_SHARE_MAX,
        ),
        "unclassifiable_segments_le_20_pct_of_expected_pulses": (
            f"{unclassifiable}/{expected}",
            unclassifiable_share <= MAX_UNCLASSIFIABLE_SHARE,
        ),
        "no_unreconcilable_leg": (unreconciled, not unreconciled),
        "not_truncated_by_unrelated_cause": (truncated, not truncated),
        "rf_set_unchanged": (not rf_changed, not rf_changed),
        "no_history_capacity_drop": (not history_dropped, not history_dropped),
    }
    return {
        "checks": {name: {"value": value, "pass": ok} for name, (value, ok) in checks.items()},
        "valid": all(ok for _, ok in checks.values()),
        "excluded_segments": unclassifiable,
        "excluded_samples": sum(leg["unclassifiable_samples"] for leg in legs),
    }


def verdict(
    q_ms: float | None,
    fraction: float | None,
    ratio: float | None,
    legs_at_raise_fraction: int,
    timeouts: int,
) -> str:
    """Parent section 13.2 axis 2, in its fixed evaluation order (and 12.1)."""
    if q_ms is None:
        return "4_inconclusive"
    if q_ms <= Q_FINE_MS:
        if timeouts >= 1:
            return "1b_constant_fine_strongest_form"
        if fraction is not None and fraction <= FRACTION_FINE:
            return "1_constant_fine"
        return "4_inconclusive"
    if q_ms >= Q_RAISE_MS and ratio is not None:
        if ratio <= RATIO_WRITE_BINDS:
            return "2_write_latency_binds"
        if legs_at_raise_fraction >= 2 and timeouts >= 1:
            return "3_raise_defensible"
    return "4_inconclusive"


def axis2(legs: list[dict[str, Any]], session_samples: list[Sample]) -> dict[str, Any]:
    """Compute Q, W, the recomputed fraction and the verdict (only after axis 1)."""
    budget_ms = ORDINARY_BUDGET_SECONDS * 1000.0
    per_leg_max: dict[str, float] = {}
    waits: list[float] = []
    for leg in legs:
        leg_waits = [
            wait
            for sample in leg["pulse_open"]
            if _number(sample.get("queue_budget_seconds")) == ORDINARY_BUDGET_SECONDS
            and (wait := _number(sample.get("queue_wait_ms"))) is not None
        ]
        waits.extend(leg_waits)
        if leg_waits:
            per_leg_max[leg["label"]] = max(leg_waits) / budget_ms

    writes = [
        write
        for leg in legs
        for sample in leg["pulse_open"] + leg["refresh"]
        if sample.get("outcome") == "completed"
        and (write := _number(sample.get("write_ms"))) is not None
    ]
    q_ms, w_ms = pct(waits, 0.95), pct(writes, 0.95)
    fraction = round(max(waits) / budget_ms, 4) if waits else None
    ratio = round(q_ms / w_ms, 4) if q_ms is not None and w_ms else None
    legs_at_raise = sum(1 for value in per_leg_max.values() if value >= FRACTION_RAISE)
    timeouts = sum(1 for sample in session_samples if sample.get("outcome") == "queue_start_timeout")
    n = len(waits)
    return {
        "n_pulse_open_budget_2": n,
        "q_p95_queue_wait_ms_pulse_open": q_ms,
        "w_p95_write_ms_classified_completed": w_ms,
        "write_inheritance_ratio": ratio,
        "worst_wait_fraction_recomputed": fraction,
        "per_leg_worst_wait_fraction": {k: round(v, 4) for k, v in per_leg_max.items()},
        "legs_at_or_above_raise_fraction": legs_at_raise,
        "queue_start_timeouts_in_session": timeouts,
        "rate_statement": (
            f"95% upper bound on the timeout rate ~ 3/{n}"
            if n >= RATE_CLAIM_N
            else f"no rate claim below {RATE_CLAIM_N}; the honest bound is 3/{n}"
        ),
        "verdict": verdict(q_ms, fraction, ratio, legs_at_raise, timeouts),
    }


def score(
    legs: list[dict[str, Any]],
    session_samples: list[Sample],
    *,
    truncated: bool,
    rf_changed: bool,
    history_dropped: bool,
) -> dict[str, Any]:
    """Axis 1 always; axis 2 only when axis 1 is valid (parent section 13.2)."""
    first = axis1(legs, truncated=truncated, rf_changed=rf_changed, history_dropped=history_dropped)
    return {
        "axis1": first,
        "axis2": axis2(legs, session_samples) if first["valid"] else None,
        "result": "scored" if first["valid"] else "INCONCLUSIVE (axis 1)",
    }


def _snapshot_samples(path: Path) -> list[Sample]:
    """Return the samples in a runner timing snapshot, with or without the HA wrapper."""
    data = json.loads(path.read_text())
    return list(data.get("service_response", data)["samples"])


def load_runner_session(
    out_dir: Path, session_start: str
) -> tuple[list[dict[str, Any]], list[Sample], bool, list[str]]:
    """Build classified legs from runner output, in dispatch order.

    Returns (legs, session samples, history dropped, notes). A leg's samples are
    the difference between its snapshot and the previous one; the first leg's
    baseline is everything recorded before ``session_start``. Setup legs and
    dispatches that sent nothing are recorded in ``notes`` and contribute nothing.
    """
    metas = sorted(
        (json.loads(path.read_text()) for path in out_dir.glob("meta_*.json")),
        key=lambda meta: meta["dispatched_utc"],
    )
    start = datetime.datetime.fromisoformat(session_start)
    legs: list[dict[str, Any]] = []
    session: list[Sample] = []
    notes: list[str] = []
    previous: set[str] | None = None
    dropped = False
    for meta in metas:
        label = meta["label"]
        snapshot = _snapshot_samples(out_dir / f"timing_after_{label}.json")
        keys = {sample["recorded_at_utc"] for sample in snapshot}
        if previous is None:
            previous = {s["recorded_at_utc"] for s in snapshot if _recorded_at(s) < start}
        dropped = dropped or not previous <= keys
        new = [sample for sample in snapshot if sample["recorded_at_utc"] not in previous]
        previous = keys
        results = json.loads((out_dir / f"leg_{label}.json").read_text()).get("command_results") or []
        if meta.get("role") != "scored" or not results:
            notes.append(f"{label}: role={meta.get('role')} command_results={len(results)} -- excluded")
            continue
        session.extend(new)
        legs.append(classify_leg(label, new, results))
    return legs, session, dropped, notes


def _without_samples(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-leg records for printing, with the raw sample lists dropped."""
    return [
        {k: v for k, v in leg.items() if k not in ("pulse_open", "refresh")} for leg in legs
    ]


def main() -> int:
    """Score a runner output directory and print (and optionally save) the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner_out", type=Path)
    parser.add_argument("--session-start", required=True, help="UTC ISO time of the first dispatch")
    parser.add_argument("--truncated", required=True, choices=("yes", "no"))
    parser.add_argument("--rf-changed", required=True, choices=("yes", "no"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    legs, session, dropped, notes = load_runner_session(args.runner_out, args.session_start)
    result = score(
        legs,
        session,
        truncated=args.truncated == "yes",
        rf_changed=args.rf_changed == "yes",
        history_dropped=dropped,
    )
    for note in notes:
        print(note)
    for name, check in result["axis1"]["checks"].items():
        print(f"{'PASS' if check['pass'] else 'FAIL'}  {name}: {check['value']}")
    print(f"excluded: {result['axis1']['excluded_segments']} segments, "
          f"{result['axis1']['excluded_samples']} samples")
    print("RESULT:", result["result"])
    if result["axis2"] is not None:
        print(json.dumps(result["axis2"], indent=1))
    if args.json is not None:
        args.json.write_text(
            json.dumps({**result, "legs": _without_samples(legs), "notes": notes}, indent=1, default=str)
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
