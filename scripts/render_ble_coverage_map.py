#!/usr/bin/env python3
"""Render the banked BLE coverage samples as a map of the yard you can look at.

Reads `scripts/ble_coverage_map.json` (written by `scripts/ble_coverage_map.py`,
which is what derives and verifies the lat/lon -> `mower_map_xy` transform) plus
the map geometry from `export_map`, and writes ONE self-contained HTML file.
Sends no movement and commands nothing; the only live calls are the two
read-only diagnostic services, and `--map-json` skips even those.

🔑 **The primary layer is not a heatmap I invented -- it is the planner's own
gate, drawn.** `scripts/plan_aligned_leg.py` rejects a candidate target whose
coverage samples within `--coverage-radius-m` (2.0 m) average below
`--min-rssi-dbm` (-76). This script evaluates that identical field on a grid by
calling `plan_aligned_leg._estimate_rssi` itself rather than reimplementing it,
so the picture cannot drift from the decision it depicts. Every blue cell is a
place the planner would accept; every cell in the bottom bin is a place it would
refuse; every hatched cell is a place it has no evidence about at all.

🚨 **WHAT THIS MAP IS NOT.** Four limits, none of them cosmetic:

1. **It is a record of where the link HAS been sampled, not a propagation
   model.** The mower drives paths and parks on the dock, so coverage is dense
   along routes and absent everywhere else. Hatched ground means *unverified*,
   never *fine* -- optimistically reading blank space as clear is the assumption
   that walked a leg into a dead zone.
2. **`ble_rssi` is self-reported and stale.** It read -60 dBm straight through a
   total outage. A strong-looking cell is therefore weak evidence of a healthy
   link, while a weak-looking cell is strong evidence of a bad one. The map is
   worth more for the dark patches than the light ones.
3. **RSSI does not predict cadence** (within-run median r = +0.042 over 24
   runs), so this map cannot be read as a throughput or reliability map.
4. 🚨 **It cannot see the failure mode that actually bit.** Legs 7 and 8 of the
   2026-09-10 repeat series died on the 2.0 s BLE command-queue timeout with
   good RSSI (-60 to -67 dBm) moments before and after. RSSI cannot see queue
   occupancy. A leg can run entirely through the strongest cells on this map and
   still refuse. Use `motion_dispatch_timing_report` for that question.

Samples reading exactly 0 dBm are already dropped upstream (`ble_rssi 0` means
the mower has dozed, not a 0 dBm link), so they cannot brighten a cell here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mammotion_ha_helpers import load_dotenv, post_service  # noqa: E402
from plan_aligned_leg import _estimate_rssi  # noqa: E402
from scan_contained_bearings import ENTITY  # noqa: E402

Point = tuple[float, float]
Sample = dict[str, float]

#: The measured BLE regime for this mower: the link works above ~-70 dBm and
#: dies below ~-76. These are observations, not settings -- do not "tune" them.
WORKS_ABOVE_DBM = -70.0
DIES_BELOW_DBM = -76.0

#: 🚨 These MUST equal `plan_aligned_leg`'s `--coverage-radius-m` and
#: `--min-rssi-dbm` defaults. They are the gate this map draws; if the planner's
#: defaults move and these do not, the map depicts a decision nothing makes.
#: `tests/scripts/test_render_ble_coverage_map.py` pins the pair together.
PLANNER_RADIUS_M = 2.0
PLANNER_MIN_RSSI_DBM = -76.0

#: Ordinal ramp, one hue (blue), monotone in lightness. It encodes link
#: WEAKNESS, so the darkest step on a light surface (and the brightest on a
#: dark one) is the worst spot -- the thing an operator opens this map to find.
#: Bin edges are the measured regime above, not round numbers. Each entry is
#: (INCLUSIVE lower bound in dBm, label, light hex, dark hex), ordered strongest
#: first; the last entry's floor is a sentinel that catches everything left.
RSSI_BINS: tuple[tuple[float, str, str, str], ...] = (
    (-60.0, "−60 dBm and stronger", "#86b6ef", "#184f95"),
    (-65.0, "−60 to −65", "#5598e7", "#256abf"),
    (WORKS_ABOVE_DBM, "−65 to −70", "#2a78d6", "#3987e5"),
    (DIES_BELOW_DBM, "−70 to −76 — marginal", "#1c5cab", "#86b6ef"),
    (-999.0, "below −76 — refused", "#0d366b", "#cde2fb"),
)

_TEMPLATE_PATH = Path(__file__).resolve().parent / "render_ble_coverage_map.html"


def load_samples(path: Path) -> tuple[list[Sample], dict[str, Any]]:
    """Load the banked coverage file, returning (samples, its metadata)."""
    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} does not exist. Run scripts/ble_coverage_map.py "
            "first -- it is what derives and verifies the lat/lon -> map_xy "
            "transform these samples are expressed in."
        )
    data = json.loads(path.read_text())
    samples = [
        {"x": float(s["x"]), "y": float(s["y"]), "rssi": float(s["rssi"])}
        for s in data.get("samples") or []
    ]
    if not samples:
        raise SystemExit(
            f"ERROR: {path} holds no samples. Nothing to draw -- widen "
            "scripts/ble_coverage_map.py's --hours window and re-bank."
        )
    meta = {k: v for k, v in data.items() if k != "samples"}
    return samples, meta


def _polygons(raw: dict[str, Any], names: dict[str, str]) -> list[dict[str, Any]]:
    """Normalize an `export_map` polygon dict into name/points records."""
    out: list[dict[str, Any]] = []
    for key, points in (raw or {}).items():
        pts = [[float(p["x"]), float(p["y"])] for p in points if "x" in p and "y" in p]
        if len(pts) >= 3:
            out.append({"name": names.get(str(key), str(key)), "points": pts})
    return out


def load_geometry(
    map_json: Path | None, entity: str, save_map: Path | None
) -> dict[str, Any]:
    """Return mowing areas and keep-outs in `mower_map_xy`, cached or live."""
    if map_json is not None:
        payload = json.loads(map_json.read_text())
    else:
        load_dotenv(Path(".env"))
        url, token = os.environ["HA_URL"], os.environ["HA_TOKEN"]
        payload = post_service(
            url, token, "mammotion", "export_map", {"entity_id": entity}, 60
        )
        if save_map is not None:
            save_map.parent.mkdir(parents=True, exist_ok=True)
            save_map.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"cached export_map payload to {save_map}")

    names = {
        str(area.get("area_hash")): str(area.get("name") or area.get("area_hash"))
        for area in payload.get("areas") or []
    }
    areas = _polygons(payload.get("area_polygons") or {}, names)
    if not areas:
        raise SystemExit(
            "ERROR: export_map returned no area_polygons -- there is no yard "
            "outline to draw the coverage against."
        )
    return {
        "areas": areas,
        "keep_outs": _polygons(payload.get("keep_out_polygons") or {}, {}),
    }


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """Ray-casting containment test for a closed polygon in map_xy."""
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        if (y1 > y) != (y2 > y):
            crossing = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if crossing > x:
                inside = not inside
    return inside


def bounds_of(areas: list[dict[str, Any]], samples: list[Sample]) -> tuple[float, ...]:
    """Bounding box covering every area vertex and every banked sample."""
    xs = [p[0] for area in areas for p in area["points"]] + [s["x"] for s in samples]
    ys = [p[1] for area in areas for p in area["points"]] + [s["y"] for s in samples]
    return (min(xs), min(ys), max(xs), max(ys))


def _bucket_index(
    samples: list[Sample], radius: float
) -> dict[tuple[int, int], list[Sample]]:
    """Bucket samples on a `radius`-sized grid so lookups stay local."""
    index: dict[tuple[int, int], list[Sample]] = defaultdict(list)
    for sample in samples:
        index[(int(sample["x"] // radius), int(sample["y"] // radius))].append(sample)
    return index


def _nearby(
    index: dict[tuple[int, int], list[Sample]], x: float, y: float, radius: float
) -> list[Sample]:
    """Every sample that could lie within `radius` of (x, y) -- a superset.

    A superset is sufficient and exact for the caller: `_estimate_rssi` applies
    its own radius filter, so handing it the 3x3 bucket neighbourhood gives a
    bit-identical answer to handing it the whole list, at a fraction of the
    work. Anything outside those buckets is further than `radius` by
    construction and cannot change the result.
    """
    bx, by = int(x // radius), int(y // radius)
    out: list[Sample] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            out.extend(index.get((bx + dx, by + dy), ()))
    return out


def build_cells(
    samples: list[Sample],
    areas: list[dict[str, Any]],
    *,
    cell_m: float,
    radius: float,
) -> tuple[list[list[float]], int]:
    """Evaluate the planner's coverage field on a grid over the mowing areas.

    Returns (cells, in_area_cell_count) where each cell is
    `[x, y, mean_dbm, worst_dbm, sample_count]` at the cell's lower-left corner.
    Only cells whose centre falls inside a mowing area are considered, and only
    those with at least one sample in range are returned -- a cell with no
    evidence is deliberately absent so the renderer can show bare ground
    through it rather than paint a guess.
    """
    min_x, min_y, max_x, max_y = bounds_of(areas, [])
    index = _bucket_index(samples, radius)
    cells: list[list[float]] = []
    in_area = 0

    steps_x = int(math.ceil((max_x - min_x) / cell_m))
    steps_y = int(math.ceil((max_y - min_y) / cell_m))
    for ix in range(steps_x):
        for iy in range(steps_y):
            x = min_x + ix * cell_m
            y = min_y + iy * cell_m
            cx, cy = x + cell_m / 2, y + cell_m / 2
            if not any(point_in_polygon(cx, cy, a["points"]) for a in areas):
                continue
            in_area += 1
            candidates = _nearby(index, cx, cy, radius)
            mean, count = _estimate_rssi(cx, cy, candidates, radius=radius)
            if mean is None:
                continue
            worst = min(
                s["rssi"]
                for s in candidates
                if math.hypot(s["x"] - cx, s["y"] - cy) <= radius
            )
            cells.append([round(x, 3), round(y, 3), round(mean, 1), worst, count])
    return cells, in_area


def summarize(
    cells: list[list[float]], in_area: int, samples: list[Sample]
) -> dict[str, Any]:
    """Headline numbers for the page: evidence, refusals and the worst spot."""
    refused = [c for c in cells if c[2] < PLANNER_MIN_RSSI_DBM]
    marginal = [c for c in cells if PLANNER_MIN_RSSI_DBM <= c[2] < WORKS_ABOVE_DBM]
    worst_cell = min(cells, key=lambda c: c[2]) if cells else None
    rssis = [s["rssi"] for s in samples]
    return {
        "sample_count": len(samples),
        "rssi_median": round(statistics.median(rssis), 1),
        "rssi_min": min(rssis),
        "rssi_max": max(rssis),
        "cells_in_area": in_area,
        "cells_with_evidence": len(cells),
        "evidence_fraction": round(len(cells) / in_area, 4) if in_area else 0.0,
        "cells_refused": len(refused),
        "cells_marginal": len(marginal),
        "worst_cell": worst_cell,
    }


def render_html(payload: dict[str, Any]) -> str:
    """Inject the payload into the standalone HTML template."""
    template = _TEMPLATE_PATH.read_text()
    # Area names come off the device, so escape any sequence that could close
    # the host <script> element early. `\/` is a legal JSON string escape.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return template.replace("__COVERAGE_PAYLOAD__", blob)


def _parse_args() -> argparse.Namespace:
    """Build the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("scripts/ble_coverage_map.json"),
        help="banked samples from scripts/ble_coverage_map.py",
    )
    parser.add_argument(
        "--map-json",
        type=Path,
        default=None,
        help="a saved export_map payload; omit to fetch it live from HA",
    )
    parser.add_argument(
        "--save-map",
        type=Path,
        default=Path("scripts/ble_coverage_map.export.json"),
        help="cache the fetched export_map payload here, so later re-renders "
        "can run fully offline via --map-json (gitignored)",
    )
    parser.add_argument(
        "--cell-m",
        type=float,
        default=0.5,
        help="grid resolution of the rendered field",
    )
    parser.add_argument(
        "--radius-m",
        type=float,
        default=PLANNER_RADIUS_M,
        help="sample pooling radius; the default is the planner's own, and "
        "changing it means the map no longer depicts the planner's gate",
    )
    parser.add_argument("--out", type=Path, default=Path("scripts/ble_coverage_map.html"))
    return parser.parse_args()


def main() -> int:
    """Build the coverage field and write the standalone map page."""
    args = _parse_args()

    samples, meta = load_samples(args.coverage)
    geometry = load_geometry(args.map_json, args.entity, args.save_map)
    print(
        f"samples: {len(samples)}  areas: {len(geometry['areas'])}  "
        f"keep-outs: {len(geometry['keep_outs'])}"
    )

    cells, in_area = build_cells(
        samples, geometry["areas"], cell_m=args.cell_m, radius=args.radius_m
    )
    summary = summarize(cells, in_area, samples)
    print(
        f"grid: {summary['cells_with_evidence']}/{summary['cells_in_area']} cells "
        f"have evidence ({summary['evidence_fraction'] * 100:.1f}%); "
        f"{summary['cells_refused']} would be refused at {PLANNER_MIN_RSSI_DBM} dBm"
    )
    if args.radius_m != PLANNER_RADIUS_M:
        print(
            f"⚠️  --radius-m {args.radius_m} differs from the planner's "
            f"{PLANNER_RADIUS_M} m -- this map no longer depicts its gate."
        )

    payload = {
        "meta": meta,
        "summary": summary,
        "bounds": bounds_of(geometry["areas"], samples),
        "cell_m": args.cell_m,
        "radius_m": args.radius_m,
        "planner_min_rssi_dbm": PLANNER_MIN_RSSI_DBM,
        "works_above_dbm": WORKS_ABOVE_DBM,
        "dies_below_dbm": DIES_BELOW_DBM,
        "bins": [
            {"floor": floor, "label": label, "light": light, "dark": dark}
            for floor, label, light, dark in RSSI_BINS
        ],
        "areas": geometry["areas"],
        "keep_outs": geometry["keep_outs"],
        "cells": cells,
        "samples": [[s["x"], s["y"], s["rssi"]] for s in samples],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(payload))
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
