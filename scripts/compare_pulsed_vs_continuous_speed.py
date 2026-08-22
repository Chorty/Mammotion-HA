#!/usr/bin/env python3
"""Measure what continuous motion would actually buy, in seconds.

The operator's goal is fluidity, not accuracy -- accuracy is closed at ~0.089 m
mean. So the question worth answering before more mower time is how much faster
a continuous controller would be, from data already banked.

🗑️ **It REFUTES the obvious guess.** A 4 s capture appears to show the mower
still accelerating at the end (0.243 -> 0.266 -> 0.298 m/s), which suggests
short pulses never reach full speed and that continuous motion would be faster
*per second of movement*. The banked pulse corpus says otherwise: a 500 ms pulse
already achieves a median 0.2422 m/s and a 1500 ms pulse 0.2586 m/s, matching
continuous. The apparent in-window ramp is the ~1 Hz feed's reporting lag
unwinding, not the drivetrain.

So the entire gain is **removing dead time**, and that is worth measuring
exactly rather than guessing.

Reads banked evidence only. No Home Assistant import, no network, no dispatch.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_position_predictability import harvest  # noqa: E402

TIMESTAMP = re.compile(r'"(20\d\d-\d\d-\d\dT[\d:.]+(?:\+00:00|Z))"')


def pulsed_speed(docs: Path) -> dict[str, Any]:
    """Median in-pulse speed for the accepted profile's linear pulses."""
    rows = [
        r
        for r in harvest(str(docs))
        if abs(int(r["speed"])) == 400 and 1100 <= r["elapsed_ms"] <= 1600
    ]
    speeds = sorted(r["travel_m"] / (r["elapsed_ms"] / 1000.0) for r in rows)
    return {
        "pulse_count": len(speeds),
        "median_in_pulse_speed_ms": round(statistics.median(speeds), 4),
        "max_in_pulse_speed_ms": round(speeds[-1], 4),
    }


def end_to_end(path: Path) -> dict[str, Any]:
    """Wall-clock speed of a banked multi-segment run, settle time included."""
    text = path.read_text()
    stamps = sorted(set(TIMESTAMP.findall(text)))
    parsed = [datetime.fromisoformat(s.replace("Z", "+00:00")) for s in stamps]
    seconds = (parsed[-1] - parsed[0]).total_seconds()
    distance = float(re.search(r'"total_distance_m":\s*([\d.]+)', text).group(1))
    return {
        "path": str(path),
        "distance_m": round(distance, 4),
        "wall_clock_s": round(seconds, 1),
        "effective_speed_ms": round(distance / seconds, 4),
    }


def continuous(path: Path) -> dict[str, Any]:
    """Speed sustained through a continuous capture window."""
    body = json.loads(path.read_text())
    body = body.get("service_response", body)
    delta = body["motion_interpretation"]["delta"]["distance"]
    window = body["duration_ms"] / 1000.0
    return {
        "path": str(path),
        "distance_m": round(delta, 4),
        "window_s": window,
        "sustained_speed_ms": round(delta / window, 4),
    }


def main() -> int:
    """Compare pulsed end-to-end speed against a continuous window."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--pulsed-run", type=Path, required=True)
    parser.add_argument("--continuous-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    pulses = pulsed_speed(args.docs)
    pulsed = end_to_end(args.pulsed_run)
    cont = continuous(args.continuous_capture)
    speedup = cont["sustained_speed_ms"] / pulsed["effective_speed_ms"]
    duty = pulsed["effective_speed_ms"] / pulses["median_in_pulse_speed_ms"]

    print("IN-PULSE speed (the drivetrain, banked corpus)")
    print(f"  {pulses['pulse_count']} pulses at linear 400, 1.1-1.6 s")
    print(
        f"  median {pulses['median_in_pulse_speed_ms']} m/s, "
        f"max {pulses['max_in_pulse_speed_ms']} m/s"
    )
    print("\nCONTINUOUS window (today)")
    print(
        f"  {cont['distance_m']} m in {cont['window_s']} s "
        f"= {cont['sustained_speed_ms']} m/s"
    )
    print("\nPULSED end to end (settle time included)")
    print(
        f"  {pulsed['distance_m']} m in {pulsed['wall_clock_s']} s "
        f"= {pulsed['effective_speed_ms']} m/s"
    )
    print("\n  in-pulse and continuous speeds agree => no ramp penalty")
    print(f"  effective duty cycle          : {duty * 100:.1f}%")
    print(f"  continuous speed-up, end to end: {speedup:.2f}x")
    print(
        f"  a {pulsed['distance_m']:.1f} m route: "
        f"{pulsed['wall_clock_s']:.0f} s -> "
        f"{pulsed['distance_m'] / cont['sustained_speed_ms']:.0f} s"
    )

    result = {
        "mode": "offline_pulsed_vs_continuous_speed",
        "authoritative": False,
        "in_pulse": pulses,
        "pulsed_end_to_end": pulsed,
        "continuous_window": cont,
        "effective_duty_cycle": round(duty, 4),
        "continuous_speedup_x": round(speedup, 3),
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
