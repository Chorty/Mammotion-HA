#!/usr/bin/env python3
"""Score candidate rules for criterion 2a against every banked step-response run.

Offline. Reads `docs/raw-samples/*.json` and nothing else: no Home Assistant
client, no token, no coordinator, no BLE, no service call, no dispatch path.
**No verdict this script prints ever authorizes motion.**

Why this exists, and how it differs from `scripts/rescore_course_rate_rules.py`:

1. 🔑 **It IMPORTS the shipped scoring functions instead of reimplementing them.**
   The older script reimplemented `_step_response_course_series` byte-for-byte to
   stay stdlib-only. That is exactly the drift class that let a published figure
   (`1.727 / 0.503`) describe a statistic the shipped rule does not compute —
   the shipped `_step_response_half_phase_agreement` uses a time-weighted
   ENDPOINT DIFFERENCE per half, not a mean of interval rates, and the correct
   pair is `1.625 / 0.107`. Importing removes the possibility.
   Precedent for importing production code offline:
   `scripts/replay_continuous_controller_against_capture.py`.

2. **Rules live in a registry**, so a candidate is added without editing the
   scorer or the printer. The older script inlined rules A-E inside `score_run`.

3. **The roster is globbed**, not typed. A hand-typed roster is why the
   2026-09-01 repeat sat unscored for two days.

⚠️ **Every candidate rule here was registered by
`docs/predeclared-2a-replacement-20260903.md` (commit `88b7fddb`) BEFORE any of
its verdicts were computed.** That commit's git timestamp is the evidence. Adding
a rule after seeing verdicts, or retuning a registered constant to improve one,
defeats the entire method — see
`docs/predeclared-rtk-vio-course-rate-scoring-20260831.md` for the precedent.

Usage:
    .venv/bin/python scripts/rescore_2a_candidates.py
    .venv/bin/python scripts/rescore_2a_candidates.py --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from custom_components.mammotion.continuous_controller import (  # noqa: E402
    normalize_degrees,
)
from custom_components.mammotion.services import (  # noqa: E402
    _STEP_RESPONSE_MIN_INFORMATIVE_INTERVALS,
    _STEP_RESPONSE_RATE_AGREEMENT_BOUND_DEG_PER_S,
    _step_response_half_phase_agreement,
    _step_response_vio_intervals,
)

RAW = REPO / "docs" / "raw-samples"

# Runs excused from scoring, with the reason. Mirrors the excuse list in
# tests/components/mammotion/test_step_response_vio_scoring.py -- a file is
# never silently skipped.
EXCUSED: dict[str, str] = {
    "raw-phaseA-linear300-speed-20260903.json": (
        "1000 ms step yields ~1 informative step interval against the rule's "
        ">=3, so 2a is unscoreable BY DESIGN. It measured speed, not 2a."
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------
# A rule is `callable(step_intervals) -> dict` returning at least
# {"statistic": float | None, "passed": bool}. Extra keys are printed as detail.
Rule = Callable[[list[dict[str, Any]]], dict[str, Any]]


def rule_a_status_quo(step_intervals: list[dict[str, Any]]) -> dict[str, Any]:
    """Rule A -- the shipped E-VIO statistic, called directly, not re-derived.

    Half-phase agreement on a time-weighted ENDPOINT difference per half, with
    the boundary reading shared between halves.
    """
    verdict = _step_response_half_phase_agreement(step_intervals, "step")
    return {
        "statistic": verdict["half_diff_deg_per_s"],
        "passed": verdict["passed"],
        "half_rates": verdict["half_rates_deg_per_s"],
        "n": verdict["informative_intervals"],
    }


# --- candidates, registered by docs/predeclared-2a-replacement-20260903.md ---
# Commit 88b7fddb, BEFORE any verdict below was computed.

ONSET_ALLOWANCE_MS = 2000.0
"""Rule C's onset window. NOT tuned here -- it is this project's own measured
onset lag (2026-08-29: rotation does not start for ~1-2 s). Declaring a physical
TIME window rather than "the first interval" is what makes the rule independent
of where sample boundaries fall, which is the defect that split the two +180
runs."""

PLATEAU_WINDOW_MS = 3000.0
"""Rule D's comparison window."""

SLOPE_BOUND_DEG_PER_S2 = 0.30
"""Rule E's bound: 1.5 deg/s of drift across a nominal 5 s of post-onset step."""


def _endpoint_rate(intervals: list[dict[str, Any]]) -> float | None:
    """Time-weighted endpoint rate across a run of intervals, as Rule A defines it."""
    if len(intervals) < 1:
        return None
    t0 = intervals[0]["from_elapsed_ms"]
    t1 = intervals[-1]["to_elapsed_ms"]
    if t1 <= t0:
        return None
    h0 = intervals[0]["from_heading_degrees"]
    h1 = intervals[-1]["to_heading_degrees"]
    return normalize_degrees(h1 - h0) / ((t1 - t0) / 1000.0)


def _after_onset(
    step_intervals: list[dict[str, Any]], allowance_ms: float
) -> list[dict[str, Any]]:
    """Intervals whose MIDPOINT clears the onset window, measured from step start."""
    if not step_intervals:
        return []
    start = step_intervals[0]["from_elapsed_ms"]
    return [
        iv
        for iv in step_intervals
        if (iv["from_elapsed_ms"] + iv["to_elapsed_ms"]) / 2 - start >= allowance_ms
    ]


def rule_b_drop_first(step_intervals: list[dict[str, Any]]) -> dict[str, Any]:
    """Rule B -- drop step_intervals[0], then Rule A unchanged."""
    kept = step_intervals[1:]
    verdict = _step_response_half_phase_agreement(kept, "step")
    return {
        "statistic": verdict["half_diff_deg_per_s"],
        "passed": verdict["passed"],
        "half_rates": verdict["half_rates_deg_per_s"],
        "n": verdict["informative_intervals"],
    }


def rule_c_onset_window(step_intervals: list[dict[str, Any]]) -> dict[str, Any]:
    """Rule C -- exclude a declared 2000 ms onset window, then Rule A."""
    kept = _after_onset(step_intervals, ONSET_ALLOWANCE_MS)
    verdict = _step_response_half_phase_agreement(kept, "step")
    return {
        "statistic": verdict["half_diff_deg_per_s"],
        "passed": verdict["passed"],
        "half_rates": verdict["half_rates_deg_per_s"],
        "n": verdict["informative_intervals"],
        "excluded": len(step_intervals) - len(kept),
    }


def rule_d_plateau(step_intervals: list[dict[str, Any]]) -> dict[str, Any]:
    """Rule D -- final 3000 ms of the step vs the 3000 ms before it."""
    if not step_intervals:
        return {"statistic": None, "passed": False, "n": 0}
    end = step_intervals[-1]["to_elapsed_ms"]
    late = [
        iv for iv in step_intervals if iv["from_elapsed_ms"] >= end - PLATEAU_WINDOW_MS
    ]
    early = [
        iv
        for iv in step_intervals
        if end - 2 * PLATEAU_WINDOW_MS
        <= iv["from_elapsed_ms"]
        < end - PLATEAU_WINDOW_MS
    ]
    r_late, r_early = _endpoint_rate(late), _endpoint_rate(early)
    if r_late is None or r_early is None:
        return {
            "statistic": None,
            "passed": False,
            "n": len(late) + len(early),
            "note": "a window held fewer than 2 distinct readings",
        }
    diff = round(abs(r_late - r_early), 4)
    return {
        "statistic": diff,
        "passed": diff <= _STEP_RESPONSE_RATE_AGREEMENT_BOUND_DEG_PER_S,
        "window_rates": [round(r_early, 4), round(r_late, 4)],
        "n": len(late) + len(early),
    }


def rule_e_slope(step_intervals: list[dict[str, Any]]) -> dict[str, Any]:
    """Rule E -- least-squares slope of interval rate vs midpoint, ex onset window."""
    kept = _after_onset(step_intervals, ONSET_ALLOWANCE_MS)
    if len(kept) < _STEP_RESPONSE_MIN_INFORMATIVE_INTERVALS:
        return {"statistic": None, "passed": False, "n": len(kept)}
    xs = [iv["midpoint_elapsed_ms"] / 1000.0 for iv in kept]
    ys = [iv["rate_deg_per_s"] for iv in kept]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return {"statistic": None, "passed": False, "n": len(kept)}
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False)) / denom
    return {
        "statistic": round(abs(slope), 4),
        "passed": abs(slope) <= SLOPE_BOUND_DEG_PER_S2,
        "slope_deg_per_s2": round(slope, 4),
        "n": len(kept),
    }


RULES: dict[str, Rule] = {
    "A (status quo, shipped)": rule_a_status_quo,
    "B (drop first interval)": rule_b_drop_first,
    "C (2000 ms onset window)": rule_c_onset_window,
    "D (plateau, last 3 s vs prior 3 s)": rule_d_plateau,
    "E (residual slope)": rule_e_slope,
}


def score_run(path: Path) -> dict[str, Any]:
    """Score one banked run under every registered rule."""
    payload = _load(path)
    phases = payload.get("phases") or {}
    baseline_ms = int(phases.get("baseline_ms", 3000))
    step_ms = int(phases.get("step_ms", 7000))
    samples = payload["samples"]

    intervals = _step_response_vio_intervals(
        samples, baseline_ms=baseline_ms, step_ms=step_ms
    )
    step_intervals = [iv for iv in intervals if iv["phase"] == "step"]

    record: dict[str, Any] = {
        "file": path.name,
        "sha256": _sha256(path),
        "baseline_ms": baseline_ms,
        "step_ms": step_ms,
        "linear_speed": payload.get("linear_speed"),
        "step_angular_speed": payload.get("step_angular_speed"),
        "step_intervals": len(step_intervals),
        # ⚠️ Completion comes from motion_refresh.aborted_early, NEVER from the
        # `reason` field: it reads "travel_guard_tripped" wrongly on the two
        # 5000 ms files. See docs/raw-samples/README.md.
        "aborted_early": (payload.get("motion_refresh") or {}).get("aborted_early"),
        # The two post-E-VIO files carry the shipped rule's own verdict, which
        # is the reproduction anchor for rule A.
        "shipped_vio_analysis": (payload.get("vio_analysis") or {}).get(
            "step_steady_rotation_2a"
        ),
        "rules": {},
    }
    for name, rule in RULES.items():
        record["rules"][name] = rule(step_intervals)
    return record


def reproduction_gate(records: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Rule A must reproduce every shipped verdict a raw file carries.

    A predeclared STOP condition, in the spirit of §1 of the 2026-08-31 study:
    if the status quo cannot be reproduced, no preference judgement built on it
    is trustworthy, so stop and report rather than proceeding.
    """
    problems: list[str] = []
    for rec in records:
        shipped = rec.get("shipped_vio_analysis")
        if not shipped:
            continue  # pre-E-VIO file, carries no shipped verdict
        got = rec["rules"]["A (status quo, shipped)"]
        want_stat = shipped.get("half_diff_deg_per_s")
        if got["passed"] is not shipped.get("passed"):
            problems.append(
                f"{rec['file']}: verdict {got['passed']} != shipped {shipped.get('passed')}"
            )
        if (
            want_stat is not None
            and got["statistic"] is not None
            and abs(got["statistic"] - want_stat) > 0.01
        ):
            problems.append(
                f"{rec['file']}: statistic {got['statistic']} != shipped {want_stat}"
            )
    return (not problems), problems


def main() -> int:
    """Score every banked run under every registered rule and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the full record here")
    args = parser.parse_args()

    on_disk = sorted(RAW.glob("*.json"))
    scored = [p for p in on_disk if p.name not in EXCUSED]

    print(
        f"bound {_STEP_RESPONSE_RATE_AGREEMENT_BOUND_DEG_PER_S} deg/s, "
        f"minimum informative intervals {_STEP_RESPONSE_MIN_INFORMATIVE_INTERVALS}"
    )
    print(f"\nroster: {len(scored)} scored, {len(EXCUSED)} excused, from {RAW}")
    for name, why in EXCUSED.items():
        print(f"  excused: {name}\n           {why}")

    records = [score_run(p) for p in scored]

    ok, problems = reproduction_gate(records)
    print("\n--- reproduction gate (rule A vs each file's own shipped verdict) ---")
    if ok:
        checked = sum(1 for r in records if r.get("shipped_vio_analysis"))
        print(f"  PASS -- reproduced {checked} shipped verdict(s) to 0.01 deg/s")
        if checked < len(records):
            print(
                f"  ⚠️  THIN: only {checked} of {len(records)} raw files carry a "
                "`vio_analysis` block, because the four 2026-08-30 captures\n"
                "      predate E-VIO. The fuller anchor is the pinned roster in\n"
                "      tests/components/mammotion/test_step_response_vio_scoring.py,\n"
                "      which fixes all five expected statistics. Run it alongside."
            )
    else:
        print("  🚨 FAIL -- do NOT proceed to preference judgements:")
        for p in problems:
            print(f"     {p}")

    print("\n--- per-run, per-rule ---")
    for rec in records:
        print(
            f"\n{rec['file']}"
            f"\n  step {rec['step_ms']} ms, angular {rec['step_angular_speed']}, "
            f"linear {rec['linear_speed']}, {rec['step_intervals']} step intervals, "
            f"aborted_early={rec['aborted_early']}"
        )
        for name, verdict in rec["rules"].items():
            stat = verdict.get("statistic")
            stat_s = "None" if stat is None else f"{stat:.4f}"
            print(
                f"    {name:<26} {'PASS' if verdict['passed'] else 'FAIL'}  "
                f"statistic {stat_s}"
            )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "bound_deg_per_s": _STEP_RESPONSE_RATE_AGREEMENT_BOUND_DEG_PER_S,
                    "min_informative_intervals": _STEP_RESPONSE_MIN_INFORMATIVE_INTERVALS,
                    "rules": sorted(RULES),
                    "excused": EXCUSED,
                    "reproduction_gate_passed": ok,
                    "reproduction_gate_problems": problems,
                    "runs": records,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
