#!/usr/bin/env python3
"""Replay banked runs to test one-step position predictability.

How well is the next position predicted from the last fix plus the commanded
velocity?

Read-only. Commands no motion, touches no host -- it reads `docs/evidence-*.json`
only. Run it to reproduce `docs/evidence-position-predictability-20260821.json`.

**Why this question matters.** A continuous-motion controller never stops to
measure, so it must decide where to stop from a prediction rather than from a
settled reading. The prediction error IS the achievable landing tolerance for
such a design. If it lands inside `waypoint_tolerance` (0.15 m), continuous
motion is viable on the telemetry we already have; if not, it is not.

**The model.** For each linear pulse: ``travel = k * linear_speed *
delivered_window_seconds``. Both inputs are known BEFORE the pulse (the window
is what we ask for), so this is a genuine forward prediction, not a fit after
the fact. ``k`` is calibrated on the healthiest 20% of pulses by refresh cadence
and then scored against ALL of them, so the reported error is out-of-sample for
79% of the data.

🔑 **The delivered window is not the commanded one, and the difference is the
whole story.** The mower runs an H-watchdog: it moves only while refresh writes
keep arriving, so a pulse whose writes stall is a pulse that stopped early. The
replay therefore reports error against ``cadence = refresh_commands_sent /
(elapsed_ms / refresh_interval_ms)``.

⚠️ **`nonzero_writes` and `refresh_commands_sent` are close but not identical;**
this uses the latter, which is the executor's own count.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import Any

TOLERANCE_M = 0.15


def _walk_results(node: Any) -> Iterator[dict[str, Any]]:
    """Yield every segment result carrying a command_results list."""
    if isinstance(node, dict):
        if isinstance(node.get("command_results"), list):
            yield node
        for value in node.values():
            yield from _walk_results(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_results(value)


def harvest(docs: str) -> list[dict[str, Any]]:
    """Pull every linear pulse that recorded both a command and a travel."""
    pulses: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(docs, "evidence-*.json"))):
        try:
            payload = json.load(open(path))
        except ValueError, OSError:
            continue
        for result in _walk_results(payload):
            for command in result["command_results"]:
                if command.get("phase") != "linear_forward_to_target":
                    continue
                observation = command.get("final_approach_observation") or {}
                refresh = command.get("motion_refresh") or {}
                kwargs = command.get("kwargs") or {}
                travel = observation.get("measured_distance")
                elapsed = refresh.get("elapsed_ms")
                speed = kwargs.get("linear_speed")
                if travel is None or not elapsed or speed is None:
                    continue
                interval = refresh.get("refresh_interval_ms") or 200
                expected = float(elapsed) / float(interval)
                pulses.append(
                    {
                        "file": os.path.basename(path),
                        "speed": abs(int(speed)),
                        "elapsed_ms": float(elapsed),
                        "travel_m": float(travel),
                        "refresh_sent": refresh.get("refresh_commands_sent") or 0,
                        "refresh_interval_ms": interval,
                        "cadence": (
                            (refresh.get("refresh_commands_sent") or 0) / expected
                            if expected
                            else 0.0
                        ),
                        "predictor": abs(int(speed)) * float(elapsed) / 1000.0,
                    }
                )
    return pulses


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "median": round(ordered[n // 2], 4),
        "p75": round(ordered[int(n * 0.75)], 4),
        "p90": round(ordered[int(n * 0.90)], 4),
        "p95": round(ordered[int(n * 0.95)], 4),
        "max": round(ordered[-1], 4),
        "fraction_under_tolerance": round(
            sum(1 for v in ordered if v < TOLERANCE_M) / n, 4
        ),
    }


def main() -> int:
    """Replay every banked linear pulse and report prediction error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    pulses = harvest(args.docs)
    if len(pulses) < 20:
        print(f"only {len(pulses)} pulses found -- nothing to conclude")
        return 1

    # Calibrate k on the healthiest fifth, then score everything against it.
    pulses.sort(key=lambda r: -r["cadence"])
    calibration = pulses[: len(pulses) // 5]
    k = sum(r["predictor"] * r["travel_m"] for r in calibration) / sum(
        r["predictor"] ** 2 for r in calibration
    )
    for r in pulses:
        r["predicted_m"] = k * r["predictor"]
        r["error_m"] = abs(r["travel_m"] - r["predicted_m"])

    cadences = [r["cadence"] for r in pulses]
    errors = [r["error_m"] for r in pulses]
    mean_c = sum(cadences) / len(cadences)
    mean_e = sum(errors) / len(errors)
    covariance = sum(
        (c - mean_c) * (e - mean_e) for c, e in zip(cadences, errors, strict=False)
    ) / len(pulses)
    correlation = covariance / (statistics.pstdev(cadences) * statistics.pstdev(errors))

    bands = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 2.0)]
    by_band = []
    for low, high in bands:
        band = [r["error_m"] for r in pulses if low <= r["cadence"] < high]
        if band:
            by_band.append(
                {"cadence_from": low, "cadence_to": high, **_quantiles(band)}
            )

    healthy = [r["error_m"] for r in pulses if r["cadence"] >= 0.5]

    print(
        f"pulses {len(pulses)} from {len({r['file'] for r in pulses})} evidence files"
    )
    print(
        f"k calibrated on healthiest 20% (cadence >= {calibration[-1]['cadence']:.2f})"
    )
    print(f"  => {k * 400:.3f} m/s at commanded speed 400")
    print(f"correlation(cadence, |error|) = {correlation:+.3f}")
    print()
    print(f"{'cadence band':>14} {'n':>4} {'median':>8} {'p90':>8} {'<0.15 m':>9}")
    for row in by_band:
        print(
            f"{row['cadence_from']:5.2f}-{row['cadence_to']:<5.2f}  {row['n']:>4} "
            f"{row['median']:8.3f} {row['p90']:8.3f} "
            f"{row['fraction_under_tolerance'] * 100:8.1f}%"
        )
    print()
    print(f"cadence >= 0.5 (the workable regime): {_quantiles(healthy)}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "date": "2026-08-21",
                    "test": "Is the next position predictable from the last fix "
                    "plus commanded velocity?",
                    "motion_commanded": False,
                    "gate_armed": False,
                    "source": "replay of banked docs/evidence-*.json, read-only",
                    "model": "travel = k * linear_speed * delivered_window_seconds",
                    "calibration": {
                        "fitted_on": "healthiest 20% of pulses by refresh cadence",
                        "k": k,
                        "metres_per_second_at_speed_400": round(k * 400, 4),
                        "out_of_sample_fraction": 0.8,
                    },
                    "pulses": len(pulses),
                    "evidence_files": sorted({r["file"] for r in pulses}),
                    "correlation_cadence_vs_error": round(correlation, 4),
                    "all_pulses": _quantiles(errors),
                    "by_cadence_band": by_band,
                    "workable_regime_cadence_ge_0_5": _quantiles(healthy),
                    "waypoint_tolerance_m": TOLERANCE_M,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
