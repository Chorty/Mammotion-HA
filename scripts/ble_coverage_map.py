#!/usr/bin/env python3
"""Build a BLE RSSI coverage map in the executor's own map_xy frame.

Sends no movement. Read-only over banked HA history plus two diagnostic
service calls (`export_map`, `get_geojson`).

🔑 **Why this needed a coordinate transform first.** `device_tracker.*` records
position as lat/lon (WGS84); the click-to-go executor's `points` and every
area/keep-out polygon are in `mower_map_xy`, a local Cartesian frame with no
documented relationship to lat/lon. Nothing else in this repo bridges them.

**Derived here, verified against real geometry, not assumed:** `get_geojson`
exports the SAME area polygons as `export_map`, in lat/lon instead of
`mower_map_xy`. Both have identical vertex counts (72/61/60/33 on the 2026-09
map). Fitting a 2D affine transform between them at the correct
correspondence (found by searching every rotation and both winding directions
-- the two exports do not start at the same vertex or wind the same way) gave
an RMS residual of **0.0000 m** on 72 points: `mower_map_xy` is plain ENU
meters (scale ~1.0011, zero rotation) offset from the geojson polygon's first
vertex. This script re-derives and verifies that fit every run rather than
hardcoding it, because a different or re-surveyed map would silently break a
cached constant.

🚨 **The script refuses to produce a coverage map on a bad fit.** See
`--max-fit-rms-m`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mammotion_ha_helpers import get_history, load_dotenv, post_service  # noqa: E402
from scan_contained_bearings import ENTITY  # noqa: E402

Point = tuple[float, float]

DEVICE_TRACKER = "device_tracker.back_yard_clip_skywalker_luba_vsplv397"
RSSI_ENTITY = "sensor.back_yard_clip_skywalker_ble_rssi"


_EARTH_RADIUS_M = 6371000.0


def _to_enu(lon: float, lat: float, lon0: float, lat0: float) -> Point:
    """Equirectangular local-tangent-plane approximation, meters from (lon0, lat0)."""
    e = math.radians(lon - lon0) * _EARTH_RADIUS_M * math.cos(math.radians(lat0))
    n = math.radians(lat - lat0) * _EARTH_RADIUS_M
    return e, n


def _best_fit_for_polygon(
    xy: np.ndarray, match: list[list[float]]
) -> tuple[float, np.ndarray, float, float]:
    """Best (rms, affine, lon0, lat0) over every winding direction and start vertex."""
    lon0, lat0 = match[0]
    enu = np.array([_to_enu(lon, lat, lon0, lat0) for lon, lat in match])

    best: tuple[float, np.ndarray, float, float] | None = None
    for reverse in (False, True):
        xy_use = xy[::-1] if reverse else xy
        for shift in range(len(xy_use)):
            xy_shifted = np.roll(xy_use, shift, axis=0)
            mat = np.hstack([enu, np.ones((len(enu), 1))])
            sol, _, _, _ = np.linalg.lstsq(mat, xy_shifted, rcond=None)
            pred = mat @ sol
            rms = float(np.sqrt(((pred - xy_shifted) ** 2).sum(axis=1).mean()))
            if best is None or rms < best[0]:
                best = (rms, sol, lon0, lat0)
    assert best is not None  # noqa: S101 -- loop always runs at least once
    return best


def _matching_polygon(
    features: list[dict[str, Any]], n_points: int
) -> list[list[float]] | None:
    """First geojson Polygon feature with exactly `n_points` vertices, or None."""
    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        coords = geom["coordinates"][0]
        if len(coords) == n_points:
            return coords
    return None


def _derive_transform(
    geojson_payload: dict[str, Any],
    area_polygons: dict[str, list[dict[str, float]]],
    *,
    max_fit_rms_m: float,
) -> tuple[np.ndarray, float, float, float]:
    """Fit lat/lon -> mower_map_xy, searching vertex correspondence.

    Returns (3x2 affine matrix, lon0, lat0, rms_m). Raises SystemExit on a
    fit worse than `max_fit_rms_m` -- a bad fit here silently mislocates
    every coverage sample, so it must not produce output quietly.
    """
    features = geojson_payload.get("features") or []
    best: tuple[float, np.ndarray, float, float] | None = None

    for polygon in area_polygons.values():
        xy = np.array([(p["x"], p["y"]) for p in polygon])
        match = _matching_polygon(features, len(xy))
        if match is None:
            continue
        candidate = _best_fit_for_polygon(xy, match)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise SystemExit(
            "ERROR: no area polygon in export_map has a matching vertex count "
            "in get_geojson -- cannot derive a lat/lon -> map_xy transform."
        )
    rms, sol, lon0, lat0 = best
    if rms > max_fit_rms_m:
        raise SystemExit(
            f"ERROR: best-fit transform RMS is {rms:.3f} m, over "
            f"--max-fit-rms-m {max_fit_rms_m}. Refusing to build a coverage "
            "map on an unverified transform -- the map may have changed "
            "shape (re-synced, edited) since the polygons were captured."
        )
    return sol, lon0, lat0, rms


def _latlon_to_xy(
    lon: float, lat: float, sol: np.ndarray, lon0: float, lat0: float
) -> Point:
    e, n = _to_enu(lon, lat, lon0, lat0)
    row = np.array([e, n, 1.0])
    x, y = row @ sol
    return float(x), float(y)


def main() -> int:  # noqa: C901
    """Build and write an RSSI-vs-map_xy coverage dataset from banked history."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--hours", type=float, default=48.0, help="history window")
    parser.add_argument(
        "--max-fit-rms-m",
        type=float,
        default=0.5,
        help="refuse the transform if its fit is worse than this",
    )
    parser.add_argument(
        "--max-match-age-s",
        type=float,
        default=30.0,
        help="max gap between a position sample and an RSSI sample to pair them",
    )
    parser.add_argument("--out", type=Path, default=Path("scripts/ble_coverage_map.json"))
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    url, token = os.environ["HA_URL"], os.environ["HA_TOKEN"]

    geojson_payload = post_service(
        url, token, "mammotion", "get_geojson", {"entity_id": args.entity}, 60
    )
    map_payload = post_service(
        url, token, "mammotion", "export_map", {"entity_id": args.entity}, 60
    )
    area_polygons = map_payload.get("area_polygons") or {}
    if not area_polygons:
        print("ERROR: export_map returned no area_polygons.")
        return 2

    sol, lon0, lat0, rms = _derive_transform(
        geojson_payload, area_polygons, max_fit_rms_m=args.max_fit_rms_m
    )
    print(
        f"transform fit RMS: {rms:.4f} m  (reference lon0={lon0:.8f} lat0={lat0:.8f})"
    )

    end = datetime.datetime.now(datetime.UTC)
    start = end - datetime.timedelta(hours=args.hours)
    history = get_history(
        url,
        token,
        [DEVICE_TRACKER, RSSI_ENTITY],
        start_iso=start.isoformat(),
        end_iso=end.isoformat(),
    )
    positions = history.get(DEVICE_TRACKER, [])
    rssi_points = history.get(RSSI_ENTITY, [])
    print(
        f"history: {len(positions)} position points, {len(rssi_points)} rssi points "
        f"over {args.hours:.1f} h"
    )

    def ts(point: dict[str, Any]) -> float:
        return datetime.datetime.fromisoformat(point["last_changed"]).timestamp()

    pos_sorted = sorted(positions, key=ts)
    pos_times = [ts(p) for p in pos_sorted]

    samples: list[dict[str, Any]] = []
    for rp in rssi_points:
        state = rp.get("state")
        if state is None or state in ("unknown", "unavailable"):
            continue
        try:
            rssi = float(state)
        except TypeError, ValueError:
            continue
        if rssi == 0.0:
            # `ble_rssi 0` means the mower has DOZED -- no live session, not a
            # real (and implausibly strong) 0 dBm reading. Per the project's
            # own standing finding, this is self-reported and stale; including
            # it here would corrupt a coverage lookup with a value that looks
            # like "excellent signal" and means the opposite.
            continue
        rt = ts(rp)
        # nearest position sample by timestamp
        idx = min(
            range(len(pos_times)), key=lambda i: abs(pos_times[i] - rt), default=None
        )
        if idx is None:
            continue
        if abs(pos_times[idx] - rt) > args.max_match_age_s:
            continue
        attrs = pos_sorted[idx].get("attributes", {})
        lat, lon = attrs.get("latitude"), attrs.get("longitude")
        if lat is None or lon is None:
            continue
        x, y = _latlon_to_xy(lon, lat, sol, lon0, lat0)
        samples.append({"x": round(x, 3), "y": round(y, 3), "rssi": rssi})

    print(f"matched samples: {len(samples)}")

    result = {
        "generated_at_utc": end.isoformat(),
        "window_hours": args.hours,
        "transform_fit_rms_m": rms,
        "max_match_age_s": args.max_match_age_s,
        "sample_count": len(samples),
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
