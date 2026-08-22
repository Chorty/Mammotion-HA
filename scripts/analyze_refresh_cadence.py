#!/usr/bin/env python3
"""Bound the mower's motion watchdog from refresh-write timing already recorded.

WHY. Every movement pulse is held alive by refresh writes on a fixed cadence --
`motion_refresh_interval_ms: 200`. That 200 comes from the vendor app (APK
decompile 2026-07-20, corroborated by upstream pymammotion's
`examples/pyjoystick_example.py`, which drives `PeriodicThread(0.2, ...)`).
Nobody has ever measured what the mower's own watchdog actually tolerates.

It matters because BLE write latency is the documented failure mode, not the
cadence: a 1303 ms write dropped a turn to 9.23 deg/s against 23-43 deg/s
cadence-intact, and `refresh_cadence_broken` is a recorded abort cause. If the
watchdog tolerates far more than 200 ms, we are sending several times more
writes than needed and paying for it in queue pressure.

METHOD. Every recorded linear pulse carries `motion_refresh.refresh_write_durations_ms`
and a measured travel distance. A long write is a long stretch with no refresh
reaching the mower, so `max(write_durations)` is a proxy for the longest gap.
Bucket pulses by that proxy and compare median speed (travel / elapsed).

⚠️ CRUISING PULSES ONLY by default. The final-approach planner deliberately
shortens pulses near the target, so including them depresses the short-gap
buckets and manufactures a fake trend. Filter on
`final_approach.applied is False`.

⚠️ WHAT THIS IS NOT. Correlational, on a proxy, with small tail buckets. It
BOUNDS the watchdog above 400 ms; it does not measure it. The clean experiment
is one fixed-duration pulse at 200 / 400 / 600 / 800 ms refresh, measuring
travel -- the same single-variable shape as the 2026-07-22 B1 tape A/B that
found the 4in -> 44in result.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics

BANDS = [(0, 150), (150, 250), (250, 400), (400, 700), (700, 1200), (1200, 10**9)]


def collect(
    paths: list[str], *, cruising_only: bool
) -> list[tuple[float, float, float]]:
    """Return (longest_write_ms, travel_m, elapsed_ms) per qualifying linear pulse."""
    rows: list[tuple[float, float, float]] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("phase") == "linear_forward_to_target":
                refresh = node.get("motion_refresh") or {}
                approach = node.get("final_approach") or {}
                observed = node.get("final_approach_observation") or {}
                travel = observed.get("measured_distance")
                writes = refresh.get("refresh_write_durations_ms") or []
                elapsed = refresh.get("elapsed_ms")
                shortened = approach.get("applied") is not False
                if travel and writes and elapsed and not (cruising_only and shortened):
                    rows.append((max(writes), float(travel), float(elapsed)))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for path in paths:
        try:
            walk(json.load(open(path)))
        except OSError, json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    """Print median speed bucketed by longest inter-write gap."""
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", default=None)
    parser.add_argument(
        "--all-pulses", action="store_true", help="include final-approach pulses"
    )
    args = parser.parse_args()

    paths = args.files or sorted(glob.glob("docs/evidence-*.json"))
    rows = collect(paths, cruising_only=not args.all_pulses)
    scope = "all pulses" if args.all_pulses else "cruising pulses only"
    print(f"{scope}: {len(rows)} linear pulses with per-write timing and travel\n")
    print(f"{'longest write / gap':>22} {'n':>5} {'median m/s':>12} {'vs 150-250':>12}")

    baseline = None
    for low, high in BANDS:
        speeds = [t / (e / 1000.0) for mx, t, e in rows if low <= mx < high]
        if not speeds:
            continue
        median = statistics.median(speeds)
        if low == 150:
            baseline = median
        relative = f"{median / baseline * 100:.0f}%" if baseline else "-"
        label = f"{low}-{high if high < 10**8 else 9999}"
        print(f"{label:>19} ms {len(speeds):>5} {median:>12.3f} {relative:>12}")

    print(
        "\nSpeed flat to ~700 ms of gap => the watchdog tolerates far more than the"
        "\n200 ms we refresh at. Correlational, on a proxy. Confirm with a"
        "\nsingle-variable pulse A/B before changing motion_refresh_interval_ms."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
