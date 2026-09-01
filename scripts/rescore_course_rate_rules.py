#!/usr/bin/env python3
"""Offline re-scoring of criterion 2a/2b under the predeclared candidate rules.

Reads the banked raw per-sample records in ``docs/raw-samples/`` and scores all
four route-1 step-response runs (2026-08-30) under every rule registered in
``docs/predeclared-rtk-vio-course-rate-scoring-20260831.md``. Runs entirely
offline: stdlib only, no Home Assistant import, no network, no mower access.

Sanity gates (predeclared):
  1. The rebuilt RTK course series must match each raw file's own
     ``course_series`` (the deployed build's output at run time).
  2. Rule A (status quo) must reproduce the published step/settle rate
     sequences in the four ``docs/evidence-route1-*.json`` files to 0.01 deg/s
     and the same verdicts. A mismatch is a finding; the script reports it
     loudly rather than proceeding silently.

Usage:
    .venv/bin/python scripts/rescore_course_rate_rules.py [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

DOCS = Path(__file__).resolve().parent.parent / "docs"
RAW = DOCS / "raw-samples"

MIN_CHORD_M = 0.15
BOUND_DEG_PER_S = 1.5
MIN_INTERVALS = 3
POSITION_SIGMA_M = 0.0031  # registered per-axis position noise

RUNS = {
    "R1 (+120, step 5000)": (
        "raw-route1-run1-plus120-step5000-20260830.json",
        "evidence-route1-run1-fail-20260830.json",
    ),
    "R1r (+120, step 5000)": (
        "raw-route1-run1repeat-plus120-step5000-20260830.json",
        "evidence-route1-run1-repeat-fail-20260830.json",
    ),
    "SX (+120, step 7000)": (
        "raw-route1-stepext-plus120-step7000-20260830.json",
        "evidence-route1-step-extension-pass-20260830.json",
    ),
    "R2 (+180, step 7000)": (
        "raw-route1-run2-plus180-step7000-20260830.json",
        "evidence-route1-run2-plus180-fail-20260830.json",
    ),
}


def normalize_degrees(angle: float) -> float:
    """Byte-for-byte the codebase's own normalization to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# RTK course series (reimplementation of _step_response_course_series)
# ---------------------------------------------------------------------------


def build_course_series(
    samples: list[dict[str, Any]], *, baseline_ms: float, step_ms: float
) -> list[dict[str, Any]]:
    """Reimplement the shipped _step_response_course_series, byte-compatible."""
    distinct: list[tuple[float, float, float]] = []
    for sample in samples:
        position = sample.get("position") or {}
        x, y = position.get("x"), position.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        elapsed_ms = sample.get("elapsed_ms")
        if not isinstance(elapsed_ms, (int, float)):
            continue
        if distinct and distinct[-1][1] == float(x) and distinct[-1][2] == float(y):
            continue
        distinct.append((float(elapsed_ms), float(x), float(y)))

    series: list[dict[str, Any]] = []
    for (t0, x0, y0), (t1, x1, y1) in zip(distinct, distinct[1:], strict=False):
        chord = math.hypot(x1 - x0, y1 - y0)
        midpoint_ms = (t0 + t1) / 2
        phase = (
            "baseline"
            if midpoint_ms < baseline_ms
            else "step"
            if midpoint_ms < baseline_ms + step_ms
            else "settle"
        )
        series.append(
            {
                "midpoint_elapsed_ms": round(midpoint_ms, 3),
                "phase": phase,
                "chord_m": round(chord, 6),
                "informative": chord >= MIN_CHORD_M,
                "course_degrees": (
                    round(math.degrees(math.atan2(y1 - y0, x1 - x0)), 4)
                    if chord >= MIN_CHORD_M
                    else None
                ),
            }
        )
    return series


def phase_rates(
    rows: list[dict[str, Any]], phase: str, *, carryover_from: str | None = None
) -> list[float]:
    """Consecutive-pair rates over informative rows of one phase.

    With carryover_from set, the previous phase's last informative row is
    prepended, matching how the published evidence scored settle (2b).
    """
    informative = [r for r in rows if r["informative"]]
    seq = [r for r in informative if r["phase"] == phase]
    if carryover_from is not None:
        prior = [r for r in informative if r["phase"] == carryover_from]
        if prior:
            seq = [prior[-1], *seq]
    rates: list[float] = []
    for a, b in zip(seq, seq[1:], strict=False):
        dt = (b["midpoint_elapsed_ms"] - a["midpoint_elapsed_ms"]) / 1000
        if dt <= 0:
            continue
        rates.append(
            round(normalize_degrees(b["course_degrees"] - a["course_degrees"]) / dt, 3)
        )
    return rates


# ---------------------------------------------------------------------------
# VIO track
# ---------------------------------------------------------------------------


def vio_track(samples: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """(elapsed_ms, heading) at each sample where the VIO reading changed."""
    track: list[tuple[float, float]] = []
    for sample in samples:
        vio = sample.get("vio") or {}
        heading = vio.get("heading")
        elapsed_ms = sample.get("elapsed_ms")
        if not isinstance(heading, (int, float)) or not isinstance(
            elapsed_ms, (int, float)
        ):
            continue
        if track and track[-1][1] == float(heading):
            continue
        track.append((float(elapsed_ms), float(heading)))
    return track


def vio_intervals(
    track: list[tuple[float, float]], *, baseline_ms: float, step_ms: float
) -> list[dict[str, Any]]:
    """Per-interval VIO rates between consecutive distinct readings."""
    intervals: list[dict[str, Any]] = []
    for (t0, h0), (t1, h1) in zip(track, track[1:], strict=False):
        midpoint_ms = (t0 + t1) / 2
        phase = (
            "baseline"
            if midpoint_ms < baseline_ms
            else "step"
            if midpoint_ms < baseline_ms + step_ms
            else "settle"
        )
        dt = (t1 - t0) / 1000
        intervals.append(
            {
                "from_ms": t0,
                "to_ms": t1,
                "midpoint_elapsed_ms": round(midpoint_ms, 3),
                "phase": phase,
                "dt_s": round(dt, 3),
                "rate": round(normalize_degrees(h1 - h0) / dt, 3) if dt > 0 else None,
                "from_heading": h0,
                "to_heading": h1,
            }
        )
    return intervals


def vio_phase_rates(
    intervals: list[dict[str, Any]], phase: str, *, carryover_from: str | None = None
) -> list[float]:
    """VIO rates for one phase, optionally prepending the prior phase's last."""
    seq = [iv for iv in intervals if iv["phase"] == phase and iv["rate"] is not None]
    if carryover_from is not None:
        prior = [
            iv
            for iv in intervals
            if iv["phase"] == carryover_from and iv["rate"] is not None
        ]
        # Carryover for VIO: the interval straddling the boundary is already in
        # one phase or the other by midpoint; prepending the prior phase's last
        # interval mirrors the RTK carryover-pair convention.
        if prior:
            seq = [prior[-1], *seq]
    return [iv["rate"] for iv in seq]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def last_two_verdict(rates: list[float], n_informative: int) -> dict[str, Any]:
    """Score the predeclared last-two-rates-within-bound criterion."""
    enough = n_informative >= MIN_INTERVALS
    diff = round(abs(rates[-1] - rates[-2]), 3) if len(rates) >= 2 else None
    ok = enough and diff is not None and diff <= BOUND_DEG_PER_S
    return {
        "informative_intervals": n_informative,
        "rates": rates,
        "last_two_diff": diff,
        "pass": bool(ok),
    }


def half_phase_verdict(
    readings: list[tuple[float, float]], n_informative: int
) -> dict[str, Any]:
    """Rule E: agreement of the two half-phase mean rates.

    readings: (elapsed_ms, angle) pairs spanning the phase, in order.
    """
    k = len(readings)
    if k < 3 or n_informative < MIN_INTERVALS:
        return {
            "informative_intervals": n_informative,
            "half_rates": None,
            "diff": None,
            "pass": False,
        }
    boundary = k // 2
    halves = [readings[: boundary + 1], readings[boundary:]]
    half_rates: list[float] = []
    for half in halves:
        (t0, a0), (t1, a1) = half[0], half[-1]
        dt = (t1 - t0) / 1000
        if dt <= 0:
            return {
                "informative_intervals": n_informative,
                "half_rates": None,
                "diff": None,
                "pass": False,
            }
        half_rates.append(round(normalize_degrees(a1 - a0) / dt, 3))
    diff = round(abs(half_rates[1] - half_rates[0]), 3)
    return {
        "informative_intervals": n_informative,
        "half_rates": half_rates,
        "diff": diff,
        "pass": diff <= BOUND_DEG_PER_S,
    }


def rtk_noise_sigma_last_two(rows: list[dict[str, Any]], phase: str) -> float | None:
    """Analytic sigma (deg/s) of the last-two-rate difference, step phase.

    Bearing noise per chord: sqrt(2)*sigma/chord (independent-bearing
    approximation, declared in the predeclaration). The middle bearing is
    shared between the two rates.
    """
    seq = [r for r in rows if r["informative"] and r["phase"] == phase]
    if len(seq) < 3:
        return None
    b_prev, b_mid, b_last = seq[-3], seq[-2], seq[-1]
    dt1 = (b_mid["midpoint_elapsed_ms"] - b_prev["midpoint_elapsed_ms"]) / 1000
    dt2 = (b_last["midpoint_elapsed_ms"] - b_mid["midpoint_elapsed_ms"]) / 1000
    if dt1 <= 0 or dt2 <= 0:
        return None

    def sigma_bearing_deg(row: dict[str, Any]) -> float:
        return math.degrees(math.sqrt(2) * POSITION_SIGMA_M / row["chord_m"])

    s_prev, s_mid, s_last = (
        sigma_bearing_deg(b_prev),
        sigma_bearing_deg(b_mid),
        sigma_bearing_deg(b_last),
    )
    var = (s_last / dt2) ** 2 + (s_mid * (1 / dt2 + 1 / dt1)) ** 2 + (s_prev / dt1) ** 2
    return round(math.sqrt(var), 3)


def rtk_noise_sigma_half_diff(rows: list[dict[str, Any]], phase: str) -> float | None:
    """Analytic sigma (deg/s) of rule E's half-rate difference, same channel.

    The boundary bearing is shared by both halves with opposite signs.
    """
    seq = [r for r in rows if r["informative"] and r["phase"] == phase]
    k = len(seq)
    if k < 3:
        return None
    boundary = k // 2
    first, mid, last = seq[0], seq[boundary], seq[-1]
    dt1 = (mid["midpoint_elapsed_ms"] - first["midpoint_elapsed_ms"]) / 1000
    dt2 = (last["midpoint_elapsed_ms"] - mid["midpoint_elapsed_ms"]) / 1000
    if dt1 <= 0 or dt2 <= 0:
        return None

    def sigma_bearing_deg(row: dict[str, Any]) -> float:
        return math.degrees(math.sqrt(2) * POSITION_SIGMA_M / row["chord_m"])

    s_first, s_mid, s_last = (
        sigma_bearing_deg(first),
        sigma_bearing_deg(mid),
        sigma_bearing_deg(last),
    )
    var = (
        (s_last / dt2) ** 2 + (s_mid * (1 / dt2 + 1 / dt1)) ** 2 + (s_first / dt1) ** 2
    )
    return round(math.sqrt(var), 3)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def crosschannel_rms(
    rows: list[dict[str, Any]], intervals: list[dict[str, Any]], phase: str
) -> tuple[float | None, int]:
    """RMS of (RTK rate - VIO rate) at RTK interval midpoints, one phase."""
    rtk_seq = [r for r in rows if r["informative"] and r["phase"] == phase]
    diffs: list[float] = []
    for a, b in zip(rtk_seq, rtk_seq[1:], strict=False):
        dt = (b["midpoint_elapsed_ms"] - a["midpoint_elapsed_ms"]) / 1000
        if dt <= 0:
            continue
        rtk_rate = normalize_degrees(b["course_degrees"] - a["course_degrees"]) / dt
        mid = (a["midpoint_elapsed_ms"] + b["midpoint_elapsed_ms"]) / 2
        vio_rate = next(
            (
                iv["rate"]
                for iv in intervals
                if iv["rate"] is not None and iv["from_ms"] <= mid < iv["to_ms"]
            ),
            None,
        )
        if vio_rate is not None:
            diffs.append(rtk_rate - vio_rate)
    if not diffs:
        return None, 0
    return round(math.sqrt(sum(d * d for d in diffs) / len(diffs)), 3), len(diffs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def score_run(  # noqa: C901
    name: str, raw_path: Path, evidence_path: Path
) -> dict[str, Any]:
    """Score one run under every predeclared rule, sanity gates first."""
    raw = json.loads(raw_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    phases = raw["phases"]
    baseline_ms, step_ms = float(phases["baseline_ms"]), float(phases["step_ms"])
    samples = raw["samples"]

    rows = build_course_series(samples, baseline_ms=baseline_ms, step_ms=step_ms)

    # Sanity gate 1: rebuilt series matches the deployed build's own output.
    recorded = raw["course_series"]
    series_mismatches: list[str] = []
    if len(rows) != len(recorded):
        series_mismatches.append(f"row count {len(rows)} != recorded {len(recorded)}")
    else:
        for i, (mine, theirs) in enumerate(zip(rows, recorded, strict=False)):
            for key in ("phase", "informative", "chord_m", "course_degrees"):
                a, b = mine.get(key), theirs.get(key)
                if isinstance(a, float) and isinstance(b, (int, float)):
                    if abs(a - float(b)) > 1e-3:
                        series_mismatches.append(f"row {i} {key}: {a} != {b}")
                elif a != b:
                    series_mismatches.append(f"row {i} {key}: {a} != {b}")

    informative = [r for r in rows if r["informative"]]
    n_step = sum(1 for r in informative if r["phase"] == "step")
    n_settle = sum(1 for r in informative if r["phase"] == "settle")

    rtk_step_rates = phase_rates(rows, "step")
    rtk_settle_rates = phase_rates(rows, "settle", carryover_from="step")

    # Sanity gate 2: reproduce the published rate sequences and verdicts.
    cs = evidence["criteria_scoring"]
    pub_2a = cs["2_step_reaches_steady_rotation_2a"]
    pub_2b = cs["3_settle_goes_flat_2b"]
    repro_issues: list[str] = []

    def compare(label: str, mine: list[float], theirs: list[float]) -> None:
        if len(mine) != len(theirs):
            repro_issues.append(f"{label}: count {len(mine)} != {len(theirs)}")
            return
        for i, (a, b) in enumerate(zip(mine, theirs, strict=False)):
            if abs(a - b) > 0.01:
                repro_issues.append(f"{label}[{i}]: {a} != {b}")

    compare("step_rates", rtk_step_rates, pub_2a["step_rates_deg_per_s"])
    compare(
        "settle_rates",
        rtk_settle_rates,
        pub_2b["settle_rates_deg_per_s_including_carryover_from_step"],
    )

    rule_a_2a = last_two_verdict(rtk_step_rates, n_step)
    rule_a_2b = last_two_verdict(rtk_settle_rates, n_settle)
    if rule_a_2a["pass"] != (pub_2a["result"] == "PASS"):
        repro_issues.append("2a verdict mismatch vs published")
    if rule_a_2b["pass"] != (pub_2b["result"] == "PASS"):
        repro_issues.append("2b verdict mismatch vs published")

    # Rule B — VIO.
    track = vio_track(samples)
    intervals = vio_intervals(track, baseline_ms=baseline_ms, step_ms=step_ms)
    vio_step = [iv for iv in intervals if iv["phase"] == "step"]
    vio_settle = [iv for iv in intervals if iv["phase"] == "settle"]
    vio_step_rates = vio_phase_rates(intervals, "step")
    vio_settle_rates = vio_phase_rates(intervals, "settle", carryover_from="step")
    rule_b_2a = last_two_verdict(vio_step_rates, len(vio_step))
    rule_b_2b = last_two_verdict(vio_settle_rates, len(vio_settle))

    # Rule C — agreement.
    rule_c_2a = {"pass": rule_a_2a["pass"] and rule_b_2a["pass"]}
    rule_c_2b = {"pass": rule_a_2b["pass"] and rule_b_2b["pass"]}

    # Rule D — noise floors (RTK analytic; VIO pooled later, per run here we
    # bank the baseline/settle-tail VIO rates for pooling).
    rtk_sigma = rtk_noise_sigma_last_two(rows, "step")
    rtk_sigma_half = rtk_noise_sigma_half_diff(rows, "step")
    window_end = max(s["elapsed_ms"] for s in samples)
    vio_baseline_rates = [
        iv["rate"]
        for iv in intervals
        if iv["phase"] == "baseline" and iv["rate"] is not None
    ]
    vio_settle_tail_rates = [
        iv["rate"]
        for iv in intervals
        if iv["phase"] == "settle"
        and iv["rate"] is not None
        and iv["midpoint_elapsed_ms"] >= window_end - 2000
    ]

    # Rule E — half-phase mean rates.
    rtk_step_readings = [
        (r["midpoint_elapsed_ms"], r["course_degrees"])
        for r in informative
        if r["phase"] == "step"
    ]
    rule_e_rtk = half_phase_verdict(rtk_step_readings, n_step)
    vio_step_readings: list[tuple[float, float]] = []
    if vio_step:
        vio_step_readings = [(vio_step[0]["from_ms"], vio_step[0]["from_heading"])]
        vio_step_readings += [(iv["to_ms"], iv["to_heading"]) for iv in vio_step]
    rule_e_vio = half_phase_verdict(vio_step_readings, len(vio_step))

    # Diagnostics.
    rms_step, n_rms_step = crosschannel_rms(rows, intervals, "step")
    rms_settle, n_rms_settle = crosschannel_rms(rows, intervals, "settle")
    short_vio = [iv for iv in intervals if iv["dt_s"] < 0.5]
    step_chords = [r["chord_m"] for r in informative if r["phase"] == "step"]

    return {
        "name": name,
        "series_mismatches": series_mismatches,
        "repro_issues": repro_issues,
        "published_2a": pub_2a["result"],
        "published_2b": pub_2b["result"],
        "rule_a_2a": rule_a_2a,
        "rule_a_2b": rule_a_2b,
        "rule_b_2a": rule_b_2a,
        "rule_b_2b": rule_b_2b,
        "rule_c_2a": rule_c_2a,
        "rule_c_2b": rule_c_2b,
        "rule_d_rtk_sigma_last_two_step": rtk_sigma,
        "rule_d_rtk_sigma_half_diff_step": rtk_sigma_half,
        "vio_final_step_interval": (
            {
                "from_ms": vio_step[-1]["from_ms"],
                "to_ms": vio_step[-1]["to_ms"],
                "rate": vio_step[-1]["rate"],
            }
            if vio_step
            else None
        ),
        "vio_baseline_rates": vio_baseline_rates,
        "vio_settle_tail_rates": vio_settle_tail_rates,
        "rule_e_rtk_2a": rule_e_rtk,
        "rule_e_vio_2a": rule_e_vio,
        "diag_rms_step": {"rms": rms_step, "n": n_rms_step},
        "diag_rms_settle": {"rms": rms_settle, "n": n_rms_settle},
        "diag_short_vio_intervals": [
            {"dt_s": iv["dt_s"], "rate": iv["rate"], "phase": iv["phase"]}
            for iv in short_vio
        ],
        "diag_step_chords_m": step_chords,
        "diag_vio_interval_dts_step": [iv["dt_s"] for iv in vio_step],
    }


def main() -> int:  # noqa: C901
    """Run all sanity gates, rules and diagnostics; print the full table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="optional path to write results")
    args = parser.parse_args()

    results = []
    for name, (raw_name, evidence_name) in RUNS.items():
        results.append(score_run(name, RAW / raw_name, DOCS / evidence_name))

    # Pooled VIO noise (rule D): scatter about each run's own mean.
    def pooled_std(key: str) -> tuple[float | None, int]:
        deviations: list[float] = []
        for r in results:
            rates = r[key]
            if len(rates) >= 2:
                mean = statistics.fmean(rates)
                deviations.extend(rate - mean for rate in rates)
        if len(deviations) < 3:
            return None, len(deviations)
        return round(statistics.stdev(deviations), 3), len(deviations)

    vio_baseline_std, n_base = pooled_std("vio_baseline_rates")
    vio_settle_std, n_tail = pooled_std("vio_settle_tail_rates")
    candidates = [s for s in (vio_baseline_std, vio_settle_std) if s is not None]
    vio_sigma_rate = max(candidates) if candidates else None
    vio_sigma_last_two = (
        round(math.sqrt(2) * vio_sigma_rate, 3) if vio_sigma_rate is not None else None
    )

    print("=" * 78)
    print("SANITY GATES")
    print("=" * 78)
    clean = True
    for r in results:
        flags = []
        if r["series_mismatches"]:
            flags.append(f"course_series mismatches: {r['series_mismatches'][:5]}")
        if r["repro_issues"]:
            flags.append(f"published-number mismatches: {r['repro_issues'][:5]}")
        status = "OK" if not flags else "MISMATCH"
        if flags:
            clean = False
        print(f"  {r['name']}: {status}")
        for f in flags:
            print(f"    - {f}")
    if not clean:
        print("\n  ** Reproduction gate FAILED — findings above are a result in ")
        print("     themselves; interpret rule comparisons with caution. **")

    print()
    print("=" * 78)
    print("PER-RUN, PER-RULE RESULTS (bound 1.5 deg/s, >=3 informative intervals)")
    print("=" * 78)
    for r in results:
        print(
            f"\n--- {r['name']} (published 2a={r['published_2a']},"
            f" 2b={r['published_2b']})"
        )
        a2a, b2a = r["rule_a_2a"], r["rule_b_2a"]
        print(
            f"  A  RTK 2a: n={a2a['informative_intervals']}"
            f" diff={a2a['last_two_diff']} -> {'PASS' if a2a['pass'] else 'FAIL'}"
        )
        print(f"     rates: {a2a['rates']}")
        print(
            f"  B  VIO 2a: n={b2a['informative_intervals']}"
            f" diff={b2a['last_two_diff']} -> {'PASS' if b2a['pass'] else 'FAIL'}"
        )
        print(f"     rates: {b2a['rates']}")
        print(f"  C  BOTH 2a -> {'PASS' if r['rule_c_2a']['pass'] else 'FAIL'}")
        e_rtk, e_vio = r["rule_e_rtk_2a"], r["rule_e_vio_2a"]
        print(
            f"  E  RTK halves: {e_rtk['half_rates']} diff={e_rtk['diff']}"
            f" -> {'PASS' if e_rtk['pass'] else 'FAIL'}"
        )
        print(
            f"  E  VIO halves: {e_vio['half_rates']} diff={e_vio['diff']}"
            f" -> {'PASS' if e_vio['pass'] else 'FAIL'}"
        )
        a2b, b2b = r["rule_a_2b"], r["rule_b_2b"]
        print(
            f"  A  RTK 2b: n={a2b['informative_intervals']}"
            f" diff={a2b['last_two_diff']} -> {'PASS' if a2b['pass'] else 'FAIL'}"
        )
        print(
            f"  B  VIO 2b: n={b2b['informative_intervals']}"
            f" diff={b2b['last_two_diff']} -> {'PASS' if b2b['pass'] else 'FAIL'}"
        )
        print(f"     settle rates: {b2b['rates']}")
        sigma = r["rule_d_rtk_sigma_last_two_step"]
        flag = ""
        if sigma is not None and 2 * sigma > BOUND_DEG_PER_S:
            flag = "  ** 2-sigma exceeds the 1.5 bound -> INDISTINGUISHABLE **"
        print(f"  D  RTK analytic sigma(last-two-diff, step): {sigma} deg/s{flag}")
        sigma_half = r["rule_d_rtk_sigma_half_diff_step"]
        flag_half = ""
        if sigma_half is not None and 2 * sigma_half > BOUND_DEG_PER_S:
            flag_half = "  ** 2-sigma exceeds the 1.5 bound **"
        print(
            f"  D  RTK analytic sigma(E half-diff, step): {sigma_half} deg/s{flag_half}"
        )
        print(f"  diag VIO final step interval: {r['vio_final_step_interval']}")
        print(
            f"  diag step chords (m): {[round(c, 3) for c in r['diag_step_chords_m']]}"
        )
        print(
            f"  diag RTK-vs-VIO RMS: step {r['diag_rms_step']},"
            f" settle {r['diag_rms_settle']}"
        )
        if r["diag_short_vio_intervals"]:
            print(f"  diag VIO intervals <0.5s: {r['diag_short_vio_intervals']}")

    print()
    print("=" * 78)
    print("POOLED VIO NOISE (rule D)")
    print("=" * 78)
    print(f"  baseline-phase rate scatter: std={vio_baseline_std} deg/s (n={n_base})")
    print(
        f"  settle-tail (last 2 s) rate scatter: std={vio_settle_std} deg/s"
        f" (n={n_tail})"
    )
    print(
        f"  sigma_rate = {vio_sigma_rate}; sigma(last-two-diff) ="
        f" {vio_sigma_last_two} deg/s"
    )
    if vio_sigma_last_two is not None:
        verdict = (
            "INDISTINGUISHABLE at n=1"
            if 2 * vio_sigma_last_two > BOUND_DEG_PER_S
            else "bound is resolvable"
        )
        print(
            f"  2-sigma = {round(2 * vio_sigma_last_two, 3)} vs bound"
            f" {BOUND_DEG_PER_S} -> {verdict}"
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "results": results,
                    "pooled_vio": {
                        "baseline_std": vio_baseline_std,
                        "settle_tail_std": vio_settle_std,
                        "sigma_last_two": vio_sigma_last_two,
                    },
                },
                indent=1,
            )
        )
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
