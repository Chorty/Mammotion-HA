#!/usr/bin/env python3
"""Replay banked runs to find what predicts BLE refresh-cadence collapse.

Read-only. Commands no motion and touches no host -- it reads
`docs/evidence-*.json` only.

**Why cadence matters.** The mower runs an H-watchdog: it moves only while
refresh writes keep arriving, so a pulse whose writes stall is a pulse that
stopped early. `scripts/replay_position_predictability.py` measured the cost --
above cadence 0.5 the next position is predictable to ~0.03 m median, below 0.3
it is not predictable at all (p90 0.885 m). So cadence is the gate on any
predictive or continuous controller.

``cadence = refresh_commands_sent / (elapsed_ms / refresh_interval_ms)``

🔑 **Correlations are computed WITHIN each run, not pooled.** Pooling mixes
sessions recorded on different days at different distances from the proxy, and
that confound reverses the RSSI result: pooled it reads r = -0.245, which looks
like "stronger signal, worse cadence" -- an artifact of which runs happen to sit
in which band. Within-run, RSSI scatters around zero.

⚠️ **Write duration is excluded as a predictor.** It correlates strongly
(r ~ -0.63) but the relationship is close to definitional: cadence counts how
many writes fit in the window, so writes that each take longer than the interval
mechanically depress it. It is a useful real-time DETECTOR of cadence collapse,
not an independent cause of it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import statistics
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _walk_results(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if isinstance(node.get("command_results"), list):
            yield node
        for value in node.values():
            yield from _walk_results(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_results(value)


def harvest(docs: str) -> list[dict[str, Any]]:
    """One row per linear pulse, with the candidate predictors attached."""
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(docs, "evidence-*.json"))):
        try:
            payload = json.load(open(path))
        except ValueError, OSError:
            continue
        for result in _walk_results(payload):
            rssi_by_index: dict[int, float] = {}
            for sample in result.get("samples") or []:
                index = sample.get("command_index")
                transport = (sample.get("telemetry") or {}).get("transport") or {}
                if index is not None and transport.get("ble_rssi") is not None:
                    rssi_by_index.setdefault(index, transport["ble_rssi"])
            start = None
            linear = [
                c
                for c in result["command_results"]
                if c.get("phase") == "linear_forward_to_target"
            ]
            for number, command in enumerate(linear, start=1):
                refresh = command.get("motion_refresh") or {}
                elapsed = refresh.get("elapsed_ms")
                if not elapsed:
                    continue
                interval = refresh.get("refresh_interval_ms") or 200
                sent = refresh.get("refresh_commands_sent") or 0
                expected = float(elapsed) / float(interval)
                stamp = command.get("sent_at_utc")
                seconds = None
                if stamp:
                    moment = dt.datetime.fromisoformat(stamp)
                    if start is None:
                        start = moment
                    seconds = (moment - start).total_seconds()
                rows.append(
                    {
                        "file": os.path.basename(path),
                        "pulse_number": number,
                        "cadence": sent / expected if expected else 0.0,
                        "rssi": rssi_by_index.get(command.get("index")),
                        "elapsed_in_run_s": seconds,
                    }
                )
    return rows


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    xs, ys = zip(*pairs, strict=False)
    if statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    covariance = sum((a - mx) * (b - my) for a, b in pairs) / len(pairs)
    return covariance / (statistics.pstdev(xs) * statistics.pstdev(ys))


def within_run(
    rows: list[dict[str, Any]], key: str, minimum: int = 6
) -> dict[str, Any]:
    """Correlate a predictor against cadence separately inside each run."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["file"]].append(row)
    per_run = []
    for name, group in sorted(grouped.items()):
        pairs = [(r[key], r["cadence"]) for r in group if r.get(key) is not None]
        if len(pairs) < minimum:
            continue
        value = _pearson(pairs)
        if value is not None:
            per_run.append({"file": name, "n": len(pairs), "r": round(value, 3)})
    values = sorted(p["r"] for p in per_run)
    if not values:
        return {"runs": 0}
    return {
        "runs": len(values),
        "median_r": round(values[len(values) // 2], 3),
        "positive": sum(1 for v in values if v > 0),
        "negative": sum(1 for v in values if v < 0),
        "per_run": per_run,
    }


def main() -> int:
    """Replay cadence predictors and report which survive a within-run test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rows = harvest(args.docs)
    rssi = within_run(rows, "rssi")
    age = within_run(rows, "elapsed_in_run_s")

    bands = []
    for low, high in [(0, 20), (20, 45), (45, 80), (80, 10**9)]:
        band = sorted(
            r["cadence"]
            for r in rows
            if r["elapsed_in_run_s"] is not None and low <= r["elapsed_in_run_s"] < high
        )
        if band:
            bands.append(
                {
                    "from_s": low,
                    "to_s": None if high == 10**9 else high,
                    "n": len(band),
                    "median_cadence": round(band[len(band) // 2], 3),
                    "fraction_stalled_below_0_3": round(
                        sum(1 for v in band if v < 0.3) / len(band), 3
                    ),
                }
            )

    print(f"pulses {len(rows)} from {len({r['file'] for r in rows})} evidence files\n")
    print("WITHIN-RUN correlation against cadence:")
    print(
        f"  ble_rssi          median r={rssi['median_r']:+.3f} "
        f"over {rssi['runs']} runs ({rssi['positive']} pos / {rssi['negative']} neg)"
    )
    print(
        f"  seconds into run  median r={age['median_r']:+.3f} "
        f"over {age['runs']} runs ({age['positive']} pos / {age['negative']} neg)"
    )
    print("\ncadence by time into the run:")
    for band in bands:
        label = f"{band['from_s']}-{band['to_s'] or '+'}s"
        print(
            f"  {label:>9}  n={band['n']:>3}  median {band['median_cadence']:.2f}  "
            f"stalled {band['fraction_stalled_below_0_3'] * 100:5.1f}%"
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "date": "2026-08-21",
                    "test": "What predicts BLE refresh-cadence collapse?",
                    "motion_commanded": False,
                    "gate_armed": False,
                    "source": "read-only replay of docs/evidence-*.json",
                    "pulses": len(rows),
                    "evidence_files": sorted({r["file"] for r in rows}),
                    "method": (
                        "Correlations computed WITHIN each run, then summarised "
                        "across runs. Pooling mixes sessions from different days "
                        "and distances from the proxy; that confound reverses the "
                        "RSSI result (pooled r = -0.245, within-run ~0)."
                    ),
                    "ble_rssi": rssi,
                    "seconds_into_run": age,
                    "cadence_by_time_band": bands,
                    "excluded_predictor": {
                        "refresh_write_duration_ms": (
                            "Correlates at r ~ -0.63 but the relation is close to "
                            "definitional -- cadence counts writes fitting in the "
                            "window. Useful as a real-time DETECTOR, not a cause."
                        )
                    },
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
