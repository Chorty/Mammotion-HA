#!/usr/bin/env python3
"""Re-pose the Phase 1 compass-mirror check as a like-with-like comparison.

⚠️ **This does NOT change any verdict.** `analyze_phase1_capture.py` is the
authority and its 2026-08-22 result is `no_go`; a criterion that is ill-posed
gets fixed deliberately in the plan, never by a script that reports a friendlier
number. This tool exists to supply the evidence for that plan decision.

THE PROBLEM. The shipped criterion compares a **chord bearing** -- an interval
average between position fixes about a second apart -- against a **single**
`toward` sample. On a body rotating ~10 deg per interval those are different
quantities, and the answer swings by the whole rotation depending on which end
of the interval supplies `toward`.

WHAT THIS COMPUTES. For every moving step, the mirror error under three
pairings -- `toward` at the interval start, at its midpoint, and at its end --
plus the bearing uncertainty that position noise alone buys at that chord
length. `toward` is bit-identical between arrivals, so the midpoint is the only
average available: there are no intermediate samples to integrate.

Reads banked capture files. No Home Assistant import, no network, no dispatch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

MIRROR_SUM_DEGREES = 90.0
THRESHOLD_DEGREES = 10.0

# During continuous motion the position feed measured 0.70 cm cross-track RMS,
# far better than the 2-6 cm seen in pulsed measurement, which that same work
# attributes to the pulsed method rather than the sensor. Two independent
# endpoints, hence the sqrt(2).
MOVING_POSITION_SIGMA_M = 0.007


def _wrap(degrees: float) -> float:
    """Fold an angle into (-180, 180]."""
    return (degrees + 180.0) % 360.0 - 180.0


def _samples(path: Path) -> list[dict[str, Any]]:
    """Load a capture's in-window samples, wrapper or bare response."""
    raw = json.loads(path.read_text())
    body = raw.get("service_response", raw)
    return body["in_window_telemetry"]["samples"]


def _arrivals(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one sample per distinct position/toward arrival."""
    out: list[dict[str, Any]] = []
    seen = None
    for sample in samples:
        position = sample["position"]
        key = (position["x"], position["y"], position["toward"])
        if key != seen:
            out.append(sample)
            seen = key
    return out


def steps(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one record per moving step, with all three pairings."""
    arrivals = _arrivals(samples)
    records = []
    for first, second in zip(arrivals, arrivals[1:], strict=False):
        start, end = first["position"], second["position"]
        dx, dy = end["x"] - start["x"], end["y"] - start["y"]
        chord = math.hypot(dx, dy)
        if chord <= 0.0:
            continue
        bearing = math.degrees(math.atan2(dy, dx)) % 360.0
        rotation = _wrap(end["toward"] - start["toward"])
        midpoint = start["toward"] + rotation / 2.0
        records.append(
            {
                "elapsed_ms": second["elapsed_ms"],
                "chord_m": chord,
                "bearing_degrees": bearing,
                "toward_rotation_degrees": rotation,
                "error_at_start": _wrap(bearing + start["toward"] - MIRROR_SUM_DEGREES),
                "error_at_midpoint": _wrap(bearing + midpoint - MIRROR_SUM_DEGREES),
                "error_at_end": _wrap(bearing + end["toward"] - MIRROR_SUM_DEGREES),
                # What position noise alone buys at this chord length.
                "noise_bearing_uncertainty_degrees": math.degrees(
                    math.atan2(MOVING_POSITION_SIGMA_M * math.sqrt(2.0), chord)
                ),
            }
        )
    return records


def report(
    name: str, records: list[dict[str, Any]], min_chord: float
) -> dict[str, Any]:
    """Print the per-step table and return the summary for both filters."""
    print(f"\n=== {name} ===")
    header = (
        f"{'t_ms':>8} {'chord_m':>8} {'bearing':>9} {'rot':>7} "
        f"{'err@start':>10} {'err@mid':>9} {'err@end':>9} {'noise+-':>8}"
    )
    print(header)
    for record in records:
        print(
            f"{record['elapsed_ms']:8.1f} {record['chord_m']:8.4f} "
            f"{record['bearing_degrees']:9.3f} {record['toward_rotation_degrees']:7.2f} "
            f"{record['error_at_start']:10.3f} {record['error_at_midpoint']:9.3f} "
            f"{record['error_at_end']:9.3f} "
            f"{record['noise_bearing_uncertainty_degrees']:8.1f}"
        )

    summary: dict[str, Any] = {"step_count": len(records)}
    for label, subset in (
        ("all_steps", records),
        (f"chord_ge_{min_chord}m", [r for r in records if r["chord_m"] >= min_chord]),
    ):
        entry = {"step_count": len(subset)}
        for pairing in ("error_at_start", "error_at_midpoint", "error_at_end"):
            worst = max((abs(r[pairing]) for r in subset), default=None)
            entry[pairing] = {
                "max_abs_degrees": None if worst is None else round(worst, 3),
                "passes_10_deg": None if worst is None else worst <= THRESHOLD_DEGREES,
            }
        summary[label] = entry
        print(f"  {label}: {len(subset)} steps")
        for pairing in ("error_at_start", "error_at_midpoint", "error_at_end"):
            value = entry[pairing]["max_abs_degrees"]
            verdict = entry[pairing]["passes_10_deg"]
            shown = (
                "n/a"
                if value is None
                else f"{value:6.3f} deg  {'PASS' if verdict else 'FAIL'}"
            )
            print(f"    max |{pairing:17s}| = {shown}")
    return summary


def main() -> int:
    """Re-pose the mirror check across both captures and both filters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--straight", type=Path, required=True)
    parser.add_argument("--arc", type=Path, required=True)
    parser.add_argument(
        "--min-chord",
        type=float,
        default=0.15,
        help="drop steps shorter than this as position-noise dominated",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = {
        "mode": "offline_mirror_pairing_reanalysis",
        "authoritative": False,
        "note": (
            "Exploratory. analyze_phase1_capture.py holds the verdict and it is "
            "no_go. This changes no threshold and authorizes nothing."
        ),
        "moving_position_sigma_m": MOVING_POSITION_SIGMA_M,
        "min_chord_m": args.min_chord,
        "captures": {},
    }
    for name, path in (("straight", args.straight), ("shallow_arc", args.arc)):
        records = steps(_samples(path))
        result["captures"][name] = {
            "path": str(path),
            "steps": records,
            "summary": report(name, records, args.min_chord),
        }

    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
