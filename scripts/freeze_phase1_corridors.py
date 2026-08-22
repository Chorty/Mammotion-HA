#!/usr/bin/env python3
"""Freeze and verify the two Phase 1 corridors, straight and shallow arc.

The Phase 1 analyzer takes a corridor file whose margins are *declared*. It
cannot check that the polygon really came from a fresh scan, so
`prevalidated: true` is operator-supplied evidence. This script is what earns
that claim: it builds each corridor from the mower's live position and then
verifies, over a dense grid of points **inside** the polygon, that every one of
them holds the declared area and keep-out margins.

⚠️ A ray scan is not enough for the arc. `linear 400 + angular 180` curves at an
implied radius of ~1.512 m (`docs/arcs-work-20260812.md`), so the arc leaves the
initial bearing almost immediately. The arc corridor is therefore widened on the
turning side and verified as an area, not as a line.

Geometry conventions and the point-in-polygon and edge-distance helpers are
imported from `scan_contained_bearings` so the two tools cannot drift apart.
Heading is `atan2(dy, dx)`, degrees CCW from +x.

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
from scan_contained_bearings import (  # noqa: E402
    ENTITY,
    _distance_to_edges,
    _inside,
    _polygon,
)

Point = tuple[float, float]

# `linear 400 + angular 180` measured 0.5823 m of travel for +22.20 deg of
# course rotation, an implied radius of 1.512 m. Measured once, so the corridor
# below allows for the arc being considerably tighter than that.
ARC_RADIUS_M = 1.512


def _corridor(
    start: Point,
    heading_degrees: float,
    *,
    length_m: float,
    left_m: float,
    right_m: float,
) -> list[Point]:
    """Build a rectangle along `heading`, asymmetric about the centreline."""
    radians = math.radians(heading_degrees)
    ux, uy = math.cos(radians), math.sin(radians)
    # Left of travel is +90 deg in this CCW convention.
    lx, ly = -uy, ux
    back = 0.30  # cover start drift backwards as well
    corners = [
        (-back, -right_m),
        (length_m, -right_m),
        (length_m, left_m),
        (-back, left_m),
    ]
    return [
        (start[0] + ux * a + lx * b, start[1] + uy * a + ly * b) for a, b in corners
    ]


def verify(
    polygon: list[Point],
    area: list[Point],
    keep_outs: dict[str, list[Point]],
    *,
    step: float,
) -> dict[str, Any]:
    """Check every point inside `polygon` against the area and keep-outs."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_edge = float("inf")
    min_keepout = float("inf")
    sampled = 0
    outside_area = 0
    inside_keepout = 0

    x = min(xs)
    while x <= max(xs):
        y = min(ys)
        while y <= max(ys):
            if _inside(polygon, x, y):
                sampled += 1
                if not _inside(area, x, y):
                    outside_area += 1
                else:
                    min_edge = min(min_edge, _distance_to_edges(area, x, y))
                if any(_inside(poly, x, y) for poly in keep_outs.values()):
                    inside_keepout += 1
                for poly in keep_outs.values():
                    min_keepout = min(min_keepout, _distance_to_edges(poly, x, y))
            y += step
        x += step

    return {
        "grid_step_m": step,
        "points_sampled": sampled,
        "points_outside_area": outside_area,
        "points_inside_keep_out": inside_keepout,
        "min_area_edge_clearance_m": None
        if min_edge == float("inf")
        else round(min_edge, 4),
        "min_keep_out_clearance_m": None
        if min_keepout == float("inf")
        else round(min_keepout, 4),
    }


def main() -> int:
    """Build both corridors, verify them, and write the analyzer's input file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--heading", type=float, required=True)
    parser.add_argument("--length", type=float, default=2.50)
    parser.add_argument("--margin-area", type=float, default=1.20)
    parser.add_argument("--margin-keepout", type=float, default=1.50)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    url, token = os.environ["HA_URL"], os.environ["HA_TOKEN"]

    state = post_service(
        url, token, "mammotion", "export_runtime_state", {"entity_id": args.entity}, 30
    )
    position = state["position"]
    if not state["safety"]["position_valid_for_motion"]:
        print("REFUSED: position is not valid for motion", file=sys.stderr)
        return 1
    start: Point = (float(position["x"]), float(position["y"]))
    zone_hash = str(position.get("zone_hash") or "")

    mapping = post_service(
        url, token, "mammotion", "export_map", {"entity_id": args.entity}, 60
    )
    area_polygons = mapping.get("area_polygons") or {}
    if zone_hash not in area_polygons:
        print(
            f"REFUSED: zone {zone_hash!r} has no polygon in export_map", file=sys.stderr
        )
        print(f"         available: {sorted(area_polygons)}", file=sys.stderr)
        return 1
    area = _polygon(area_polygons[zone_hash])
    keep_outs = {
        k: _polygon(v) for k, v in (mapping.get("keep_out_polygons") or {}).items()
    }

    # The straight run holds its bearing, so a narrow symmetric corridor covers
    # it. The arc curves to the LEFT (positive angular measured +22.20 deg of
    # course rotation), so its corridor is widened that way by the full arc
    # sagitta plus slack, and kept wide on the right for the same-cost reason
    # that a wrong-signed arc must still be contained.
    swept = args.length / ARC_RADIUS_M
    lateral = ARC_RADIUS_M * (1.0 - math.cos(swept)) + 0.50

    routes = {
        "straight": _corridor(
            start, args.heading, length_m=args.length, left_m=0.80, right_m=0.80
        ),
        "shallow_arc": _corridor(
            start, args.heading, length_m=args.length, left_m=lateral, right_m=lateral
        ),
    }

    out: dict[str, Any] = {}
    failed = False
    for name, polygon in routes.items():
        checks = verify(polygon, area, keep_outs, step=args.step)
        ok = (
            checks["points_outside_area"] == 0
            and checks["points_inside_keep_out"] == 0
            and (checks["min_area_edge_clearance_m"] or 0.0) >= args.margin_area
            and (checks["min_keep_out_clearance_m"] or 0.0) >= args.margin_keepout
        )
        failed = failed or not ok
        radians = math.radians(args.heading)
        out[name] = {
            "prevalidated": ok,
            "area_margin_m": args.margin_area,
            "keepout_margin_m": args.margin_keepout,
            "frozen_start": {"x": start[0], "y": start[1]},
            "frozen_endpoint": {
                "x": start[0] + math.cos(radians) * args.length,
                "y": start[1] + math.sin(radians) * args.length,
            },
            "polygon": [{"x": round(px, 4), "y": round(py, 4)} for px, py in polygon],
            "verification": checks,
            "heading_degrees": args.heading,
            "corridor_length_m": args.length,
            "area_hash": zone_hash,
            "area_name": position.get("area_name"),
            "keep_out_zones": sorted(keep_outs),
        }
        verdict = "OK" if ok else "FAILED"
        print(
            f"{name:12s} {verdict:6s} "
            f"area edge {checks['min_area_edge_clearance_m']} m, "
            f"keep-out {checks['min_keep_out_clearance_m']} m, "
            f"{checks['points_sampled']} points"
        )

    args.json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
