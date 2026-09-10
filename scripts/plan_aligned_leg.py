#!/usr/bin/env python3
"""Pick the next aligned test leg, with runway lookahead. Sends no movement.

🚨 **Why this exists.** On 2026-09-10 the 4.0 m repeat series walked itself into
a corner of "Backyard Right". Each leg's target had been chosen reactively --
"continue near whatever the mower is currently facing" -- and validated only as
`is this one target inside the polygon`. That is an unmanaged random walk in a
~15x16 m area with 4.0 m hops, and it ran out of room on leg 4: no heading
within the +/-10 deg alignment tolerance stayed in bounds, so the only way out
was a 156.8 deg turn, which by definition cannot score.

This planner fixes the three things that caused it:

1. **Runway lookahead.** A candidate is judged on the reach still available
   *from its own target*, not just on whether the target is inside. A leg that
   lands somewhere with no room for the next leg is a trap, and this reports it
   one leg early instead of one leg late.
2. **Steer inside the tolerance.** `aligned_start_confirmed` needs the target
   bearing within `--tolerance` of the mower's measured facing -- it does not
   need the exact facing. So search that whole window rather than testing only
   the raw `map_facing_degrees` value. ⚠️ Among headings that clear
   `--min-runway`, the winner is the one CLOSEST to the measured facing, not
   the one with the most runway: runway is a threshold (4 m and 10 m are
   equally sufficient for the next leg) while alignment margin is not -- a
   candidate at the tolerance edge fails scoring on any drift between this
   reading and the run's own calibration drive.
3. **Name the reset leg before it is forced.** When no heading in the window
   clears `--min-runway`, say so and point at the most open bearing overall, so
   a reset is a planned unscored leg rather than a dead end discovered mid-run.

🔑 **Bearing convention is the integration's own**: heading = ``atan2(dy, dx)``,
CCW from +x. Verified against a real run: leg 1's due-south target reported
`target_map_heading_degrees` 269.785 for dx=0, dy=-4.0.

⚠️ **This plans; it does not dispatch.** Every leg still needs its own dry-run,
physical corridor confirmation and explicit operator go/no-go.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mammotion_ha_helpers import load_dotenv, post_service  # noqa: E402
from scan_contained_bearings import (  # noqa: E402
    ENTITY,
    Point,
    _polygon,
    probe,
)


def _candidate(
    start: Point,
    heading: float,
    area: list[Point],
    keep_outs: dict[str, list[Point]],
    *,
    leg: float,
    margin_area: float,
    margin_keepout: float,
    step: float,
    cap: float,
) -> dict[str, Any] | None:
    """Score one heading: does the leg fit, and what is left after it."""
    here = probe(
        start,
        heading,
        area,
        keep_outs,
        margin_area=margin_area,
        margin_keepout=margin_keepout,
        step=step,
        cap=cap,
    )
    if here["reach_m"] + 1e-9 < leg:
        return None

    radians = math.radians(heading)
    target: Point = (
        start[0] + math.cos(radians) * leg,
        start[1] + math.sin(radians) * leg,
    )

    # Fix 1 -- lookahead. From the target, how far could a NEXT leg go? Scan
    # every bearing, not just this one: the next leg is free to steer too.
    best_next = 0.0
    best_next_heading = None
    for i in range(72):  # 5 deg resolution
        nxt_heading = i * 5.0
        got = probe(
            target,
            nxt_heading,
            area,
            keep_outs,
            margin_area=margin_area,
            margin_keepout=margin_keepout,
            step=step,
            cap=cap,
        )
        if got["reach_m"] > best_next:
            best_next = got["reach_m"]
            best_next_heading = nxt_heading

    return {
        "heading_degrees": round(heading, 3),
        "target": {"x": round(target[0], 4), "y": round(target[1], 4)},
        "reach_this_leg_m": round(here["reach_m"], 3),
        "min_area_edge_clearance_m": round(here["min_area_edge_clearance_m"], 3),
        "min_keepout_clearance_m": (
            round(here["min_keepout_clearance_m"], 3)
            if here["min_keepout_clearance_m"] != float("inf")
            else None
        ),
        "runway_after_m": round(best_next, 3),
        "runway_after_heading_degrees": best_next_heading,
    }


def main() -> int:  # noqa: C901
    """Plan the next leg and report whether it has runway."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--leg", type=float, default=4.0, help="leg length (m)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="alignment window, must match the predeclared "
        "_ALIGNED_START_TOLERANCE_DEGREES",
    )
    parser.add_argument(
        "--min-runway",
        type=float,
        default=4.0,
        help="reach that must remain from the target for the NEXT leg",
    )
    parser.add_argument("--margin-area", type=float, default=0.60)
    parser.add_argument("--margin-keepout", type=float, default=1.00)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--cap", type=float, default=25.0)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    url, token = os.environ["HA_URL"], os.environ["HA_TOKEN"]

    state = post_service(
        url, token, "mammotion", "export_runtime_state", {"entity_id": args.entity}, 30
    )
    position = state["position"]
    start: Point = (float(position["x"]), float(position["y"]))
    zone_hash = str(position.get("zone_hash") or "")

    facing = (state.get("map_facing") or {}).get("map_facing_degrees")
    if facing is None:
        print("ERROR: map_facing_degrees is null -- the two heading sources do not")
        print("       corroborate, so there is no facing to align a leg against.")
        print("       Drive a short leg to re-anchor, or wait for corroboration.")
        return 2
    facing = float(facing)

    payload = post_service(
        url, token, "mammotion", "export_map", {"entity_id": args.entity}, 60
    )
    area_polygons = payload.get("area_polygons") or {}
    if zone_hash not in area_polygons:
        print(f"ERROR: mower zone_hash {zone_hash!r} has no polygon in export_map.")
        print(f"       available: {sorted(area_polygons)}")
        return 2
    area = _polygon(area_polygons[zone_hash])
    keep_outs = {
        k: _polygon(v) for k, v in (payload.get("keep_out_polygons") or {}).items()
    }

    print(f"start            : ({start[0]:.4f}, {start[1]:.4f})")
    print(f"area             : {zone_hash}  ({position.get('area_name')})")
    print(
        f"map_facing       : {facing:.3f} deg  "
        f"(confidence {(state.get('map_facing') or {}).get('confidence')})"
    )
    print(f"keep-out zones   : {len(keep_outs)}  {sorted(keep_outs)}")
    print(f"leg / tolerance  : {args.leg} m / +/-{args.tolerance} deg")
    print(f"min runway after : {args.min_runway} m")
    print("convention       : heading = atan2(dy,dx), CCW from +x")
    print()

    # Fix 2 -- steer inside the alignment window rather than taking `facing` raw.
    aligned: list[dict[str, Any]] = []
    offset = -args.tolerance
    while offset <= args.tolerance + 1e-9:
        got = _candidate(
            start,
            (facing + offset) % 360.0,
            area,
            keep_outs,
            leg=args.leg,
            margin_area=args.margin_area,
            margin_keepout=args.margin_keepout,
            step=args.step,
            cap=args.cap,
        )
        if got is not None:
            got["offset_from_facing_degrees"] = round(offset, 3)
            aligned.append(got)
        offset += args.resolution

    result: dict[str, Any] = {
        "start": {"x": start[0], "y": start[1]},
        "map_facing_degrees": facing,
        "leg_m": args.leg,
        "tolerance_degrees": args.tolerance,
        "min_runway_m": args.min_runway,
        "aligned_candidates": len(aligned),
    }

    with_runway = [c for c in aligned if c["runway_after_m"] >= args.min_runway]

    if with_runway:
        # 🚨 Among candidates that clear `--min-runway`, prefer the one closest
        # to the measured facing -- NOT the one with the most runway. Runway is
        # a threshold, not a score: 10 m and 5 m are equally sufficient for the
        # next 4 m leg. Alignment margin is not a threshold -- a candidate at
        # the -10.0 deg edge of the window fails `aligned_start_confirmed` on
        # any drift between this reading and the run's own calibration drive,
        # and an unscored leg is the exact cost this planner exists to avoid.
        best = min(with_runway, key=lambda c: abs(c["offset_from_facing_degrees"]))
        result["verdict"] = "aligned_leg_available"
        result["recommended"] = best
        print("VERDICT: aligned leg available, with runway for the next one.")
        print()
        print(
            f"  heading        : {best['heading_degrees']} deg  "
            f"(offset {best['offset_from_facing_degrees']:+} from facing)"
        )
        print(f"  target         : ({best['target']['x']}, {best['target']['y']})")
        print(f"  edge clearance : {best['min_area_edge_clearance_m']} m")
        print(
            f"  runway after   : {best['runway_after_m']} m "
            f"(best next heading {best['runway_after_heading_degrees']} deg)"
        )
    elif aligned:
        best = max(aligned, key=lambda c: c["runway_after_m"])
        result["verdict"] = "aligned_leg_fits_but_no_runway"
        result["recommended"] = best
        result["best_runway_after_m"] = best["runway_after_m"]
        print("⚠️  VERDICT: an aligned leg FITS, but leaves too little runway.")
        print(
            f"    Best runway after any aligned heading: "
            f"{best['runway_after_m']} m (< {args.min_runway} m)."
        )
        print("    Taking it means the NEXT leg is likely a forced reset turn.")
        print()
        print(
            f"  heading        : {best['heading_degrees']} deg  "
            f"(offset {best['offset_from_facing_degrees']:+} from facing)"
        )
        print(f"  target         : ({best['target']['x']}, {best['target']['y']})")
        print(f"  runway after   : {best['runway_after_m']} m")
    else:
        result["verdict"] = "no_aligned_leg_reset_required"
        print("🛑 VERDICT: NO aligned leg fits. A reset turn is required.")
        print(
            "    Every heading within the tolerance leaves the area before "
            f"{args.leg} m."
        )
        print("    A reset leg CANNOT score (aligned_start_confirmed will be false).")
        print("    Plan it deliberately rather than discovering it mid-run.")

    # Fix 3 -- always report the most open bearing overall, so a reset has a
    # destination whether or not an aligned leg was found.
    steps = int(round(360.0 / args.resolution))
    open_best = None
    for i in range(steps):
        heading = i * args.resolution
        got = probe(
            start,
            heading,
            area,
            keep_outs,
            margin_area=args.margin_area,
            margin_keepout=args.margin_keepout,
            step=args.step,
            cap=args.cap,
        )
        if open_best is None or got["reach_m"] > open_best[1]:
            open_best = (heading, got["reach_m"])

    if open_best is not None:
        turn = abs((open_best[0] - facing + 180.0) % 360.0 - 180.0)
        result["most_open"] = {
            "heading_degrees": open_best[0],
            "reach_m": round(open_best[1], 3),
            "turn_from_facing_degrees": round(turn, 1),
            "would_be_aligned": turn <= args.tolerance,
        }
        print()
        print(
            f"most open bearing: {open_best[0]} deg, reach "
            f"{open_best[1]:.2f} m, turn {turn:.1f} deg from facing"
            f"{'  (within tolerance)' if turn <= args.tolerance else ''}"
        )

    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
