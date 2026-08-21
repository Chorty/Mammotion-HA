#!/usr/bin/env python3
"""Find the longest contained click from the mower, keep-outs included.

The longest ray from the mower that stays inside its area AND clear of every
keep-out zone.

⚠️ **This exists because a scan against area geometry alone drove the mower into
a trampoline on 2026-08-20.** `export_map` exposes `keep_out_polygons` (beta63);
any bearing scan that ignores them is the same bug with a different number.

One way this is deliberately STRICTER than the shipped pre-dispatch check:

1. **It holds a clearance margin**, not just containment, so a landing error of
   up to the margin still cannot put the mower in a zone.

The backend now checks complete segments with `_keep_out_leg_violations`, so
sampling is no longer a stricter *containment* rule. This scanner still samples
at `--step` to measure clearance along the ray.

🔑 **Bearing convention is the integration's own**: heading = ``atan2(dy, dx)``,
degrees CCW from +x (`services.py:4256`). An earlier version of this scan used a
compass convention (0 = +y, clockwise); the reach numbers were right and every
LABEL was wrong. Reported headings here are directly comparable to
`target_map_heading_degrees` in an evidence file.

Usage:
    scripts/scan_contained_bearings.py                     # scan from live position
    scripts/scan_contained_bearings.py --distance 9.0      # can this click fit?
    scripts/scan_contained_bearings.py --margin-area 0.4 --margin-keepout 0.8

Requires HA_URL and HA_TOKEN:  set -a && source .env && set +a
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

ENTITY = "lawn_mower.back_yard_clip_skywalker"

Point = tuple[float, float]


def _polygon(raw: list[dict[str, Any]]) -> list[Point]:
    return [(float(p["x"]), float(p["y"])) for p in raw]


def _inside(poly: list[Point], x: float, y: float) -> bool:
    """Ray-cast point-in-polygon."""
    crossings = False
    count = len(poly)
    for i in range(count):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % count]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            crossings = not crossings
    return crossings


def _distance_to_edges(poly: list[Point], x: float, y: float) -> float:
    """Shortest distance from a point to the polygon boundary."""
    best = float("inf")
    count = len(poly)
    for i in range(count):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % count]
        dx, dy = x2 - x1, y2 - y1
        length_squared = dx * dx + dy * dy
        t = (
            0.0
            if length_squared == 0
            else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_squared))
        )
        best = min(best, math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
    return best


def probe(
    start: Point,
    heading_degrees: float,
    area: list[Point],
    keep_outs: dict[str, list[Point]],
    *,
    margin_area: float,
    margin_keepout: float,
    step: float,
    cap: float,
) -> dict[str, float]:
    """Walk one ray until it leaves the area or nears a keep-out."""
    radians = math.radians(heading_degrees)
    ux, uy = math.cos(radians), math.sin(radians)
    reach = 0.0
    closest_keepout = float("inf")
    closest_edge = float("inf")
    while reach + step <= cap:
        nxt = reach + step
        x, y = start[0] + ux * nxt, start[1] + uy * nxt
        if not _inside(area, x, y):
            break
        edge = _distance_to_edges(area, x, y)
        if edge < margin_area:
            break
        if any(_inside(poly, x, y) for poly in keep_outs.values()):
            break
        keepout = min(
            (_distance_to_edges(poly, x, y) for poly in keep_outs.values()),
            default=float("inf"),
        )
        if keepout < margin_keepout:
            break
        reach = nxt
        closest_keepout = min(closest_keepout, keepout)
        closest_edge = min(closest_edge, edge)
    return {
        "reach_m": reach,
        "min_keepout_clearance_m": closest_keepout,
        "min_area_edge_clearance_m": closest_edge,
    }


def main() -> int:
    """Scan every bearing and report the ones that fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="report whether a click of this length fits, and where",
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
    print(f"keep-out zones   : {len(keep_outs)}  {sorted(keep_outs)}")
    print(
        f"margins          : area {args.margin_area} m, keep-out "
        f"{args.margin_keepout} m, sampled every {args.step} m"
    )
    print("convention       : heading = atan2(dy,dx), CCW from +x")
    print()

    steps = int(round(360.0 / args.resolution))
    results = []
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
        results.append({"heading_degrees": heading, **got})

    ranked = sorted(results, key=lambda r: -r["reach_m"])
    print("best bearings (>= 8 deg apart):")
    shown: list[float] = []
    for row in ranked:
        heading = row["heading_degrees"]
        if any(abs(heading - s) < 8 or abs(heading - s) > 352 for s in shown):
            continue
        shown.append(heading)
        print(f"  heading {heading:6.1f}deg   reach {row['reach_m']:6.2f} m")
        if len(shown) >= 6:
            break

    if args.distance is not None:
        want = args.distance
        fits = [r for r in ranked if r["reach_m"] >= want]
        print()
        print(f"click of {want:.2f} m: {len(fits)} of {len(results)} bearings fit")
        if not fits:
            print(
                f"  DOES NOT FIT -- best available reach is {ranked[0]['reach_m']:.2f} m"
            )
        else:
            best = fits[0]
            heading = best["heading_degrees"]
            radians = math.radians(heading)
            end = (
                start[0] + math.cos(radians) * want,
                start[1] + math.sin(radians) * want,
            )
            print(
                f"  best heading {heading:.1f}deg, reach {best['reach_m']:.2f} m "
                f"(spare {best['reach_m'] - want:.2f} m)"
            )
            print(f"  end point ({end[0]:.4f}, {end[1]:.4f})")
            print(
                f"  min keep-out clearance along leg {best['min_keepout_clearance_m']:.2f} m"
            )
            print(
                f"  min area-edge clearance along leg {best['min_area_edge_clearance_m']:.2f} m"
            )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "start": {"x": start[0], "y": start[1]},
                    "area_hash": zone_hash,
                    "keep_out_zones": sorted(keep_outs),
                    "margins": {
                        "area_m": args.margin_area,
                        "keepout_m": args.margin_keepout,
                    },
                    "step_m": args.step,
                    "convention": "atan2(dy,dx) CCW from +x",
                    "results": results,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
