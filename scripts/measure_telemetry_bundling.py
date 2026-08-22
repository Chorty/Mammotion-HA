#!/usr/bin/env python3
"""Measure what actually updates, and how often, inside a capture window.

THE QUESTION THIS SETTLES. A continuous controller needs feedback. If heading
arrived faster than position, a heading loop could run faster than a position
loop and the whole feasibility picture would change. If everything arrives in
one bundle, the bundle rate is a hard ceiling on any loop, and no criterion
design can get around it.

Counts distinct-value transitions per field across the 100 ms cache samples,
and counts how often a heading field changed WITHOUT a new position -- the
decoupling that a faster heading loop would need.

⚠️ Report stamps are not position arrivals. `last_report_at` moves for every
frame; only some frames carry new `sys.toapp_report_data`. Counting stamps
overstates the usable feedback rate, which is the trap this script exists to
avoid.

Reads banked capture files. No Home Assistant import, no network, no dispatch.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

Sample = dict[str, Any]

FIELDS: tuple[tuple[str, Callable[[Sample], Any]], ...] = (
    ("report_stamp", lambda s: s["last_report_at_monotonic"]),
    ("position_xy", lambda s: (s["position"]["x"], s["position"]["y"])),
    ("toward", lambda s: s["position"]["toward"]),
    ("vio_heading", lambda s: s["vio"]["heading"]),
    ("vio_state", lambda s: s["vio"]["state"]),
)


def _samples(path: Path) -> list[Sample]:
    """Load a capture's in-window samples, wrapper or bare response."""
    raw = json.loads(path.read_text())
    return raw.get("service_response", raw)["in_window_telemetry"]["samples"]


def analyse(samples: list[Sample]) -> dict[str, Any]:
    """Count update transitions per field and heading/position decoupling."""
    span_ms = samples[-1]["elapsed_ms"] - samples[0]["elapsed_ms"]
    seconds = span_ms / 1000.0

    fields: dict[str, Any] = {}
    for name, getter in FIELDS:
        times, previous = [], None
        for index, sample in enumerate(samples):
            value = getter(sample)
            if index and value != previous:
                times.append(round(sample["elapsed_ms"], 1))
            previous = value
        fields[name] = {
            "update_count": len(times),
            "rate_hz": round(len(times) / seconds, 3) if seconds else None,
            "elapsed_ms": times,
        }

    decoupled = 0
    previous_key = None
    for sample in samples:
        position = (sample["position"]["x"], sample["position"]["y"])
        heading = (sample["position"]["toward"], sample["vio"]["heading"])
        if previous_key is not None:
            same_position = position == previous_key[0]
            new_heading = heading != previous_key[1]
            if same_position and new_heading:
                decoupled += 1
        previous_key = (position, heading)

    return {
        "window_ms": round(span_ms, 1),
        "cache_sample_count": len(samples),
        "fields": fields,
        "heading_updates_without_new_position": decoupled,
        "heading_is_bundled_with_position": decoupled == 0,
    }


def main() -> int:
    """Report bundling for each capture given."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result: dict[str, Any] = {"mode": "offline_telemetry_bundling", "captures": {}}
    for path in args.captures:
        found = analyse(_samples(path))
        result["captures"][str(path)] = found
        print(f"\n=== {path.name} ===")
        print(
            f"  window {found['window_ms']:.0f} ms, "
            f"{found['cache_sample_count']} cache samples"
        )
        for name, entry in found["fields"].items():
            print(
                f"  {name:14s} {entry['update_count']:2d} updates  "
                f"{entry['rate_hz']:5.2f} Hz  at {entry['elapsed_ms']}"
            )
        print(
            f"  heading updated without a new position: "
            f"{found['heading_updates_without_new_position']}"
        )
        print(
            f"  => heading bundled with position: "
            f"{found['heading_is_bundled_with_position']}"
        )

    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
