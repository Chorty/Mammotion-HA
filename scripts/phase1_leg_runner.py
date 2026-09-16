#!/usr/bin/env python3
"""Run ONE guarded Phase 1 leg, halting on any anomaly.

Per leg: live state -> daylight/VIO halt -> corridor + excursion + band check ->
dry run (accepted-profile echo, safety gates) -> real dispatch -> timing snapshot.
Exits 2 (HALT) on any failed check; a halt is never overridden in-session
(docs/predeclared-queue-timeout-measurement-20260911.md section 16.3).

🚨 **Why the daylight/VIO halt exists.** On 2026-09-12 a setup leg was armed at
23:55:07Z, ~6 min after sunset, because the sun was only checked hours earlier.
VIO collapsed through the leg and the final-approach realign overshot. The
point-in-time `visual_positioning_status` read `signal_good` at dispatch and
throughout the preceding 90 s -- a status check alone would have PASSED it.
What separated it from the daylight leg was the tracked-feature minimum over the
look-back window (14 vs 80) and the sun elevation (-1.95 vs 59.2 deg), so all
three clauses are enforced and any one halts. `vio_brightness` and
`camera_brightness` are deliberately NOT used: they read "good" through the
collapse.

Usage: scripts/phase1_leg_runner.py LABEL TX TY {setup,scored} --out DIR
Requires HA_URL and HA_TOKEN (set -a && source .env && set +a).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mammotion_ha_helpers import post_service  # noqa: E402

ENTITY = "lawn_mower.back_yard_clip_skywalker"
SENSOR_PREFIX = "sensor.back_yard_clip_skywalker_"
ZONE = "1343645155037768237"
ANCHOR = (4.94, -3.82)
SITING = Path("docs/evidence-phase1-siting-20260912.json")
PROFILE = Path("docs/accepted-profile.json")

YARD_LAT = 34.0247
YARD_LON = -84.7698
#: About an hour of margin before sunset at this latitude in September.
SUN_MIN_ELEVATION_DEG = 10.0
VIO_WINDOW_SECONDS = 60
#: Daylight driving dipped to 73-75; the dusk leg's 60 s window reached 14.
VIO_MIN_TRACKED_FEATURES = 70
VIO_REQUIRED_STATUS = "signal_good"

MAX_SEGMENT_M = 6.1
MIN_CORRIDOR_CLEARANCE_M = 1.0
MAX_EXCURSION_M = 3.0
STRONG_MIN_DBM = -68.0
MODERATE_MIN_DBM = -76.0


def solar_elevation_degrees(
    when: datetime.datetime, lat: float = YARD_LAT, lon: float = YARD_LON
) -> float:
    """Approximate solar elevation (NOAA low-precision), degrees above horizon."""
    n = when.timestamp() / 86400 + 2440587.5 - 2451545.0
    mean_long = (280.46 + 0.9856474 * n) % 360
    anomaly = math.radians((357.528 + 0.9856003 * n) % 360)
    ecl_long = math.radians(
        mean_long + 1.915 * math.sin(anomaly) + 0.02 * math.sin(2 * anomaly)
    )
    obliquity = math.radians(23.439 - 4e-7 * n)
    dec = math.asin(math.sin(obliquity) * math.sin(ecl_long))
    ra = math.atan2(math.cos(obliquity) * math.sin(ecl_long), math.cos(ecl_long))
    gmst = (280.46061837 + 360.98564736629 * n) % 360
    hour_angle = math.radians((gmst + lon) % 360) - ra
    lat_r = math.radians(lat)
    return math.degrees(
        math.asin(
            math.sin(lat_r) * math.sin(dec)
            + math.cos(lat_r) * math.cos(dec) * math.cos(hour_angle)
        )
    )


def window_values(
    series: list[tuple[datetime.datetime, str]],
    dispatch: datetime.datetime,
    seconds: float,
) -> list[str]:
    """States in force during ``(dispatch - seconds, dispatch]``.

    Includes the value already in force at the window start: HA history reports
    changes, so a state that did not change inside the window is still the
    reading for the whole of it.
    """
    ordered = sorted(series)
    start = dispatch - datetime.timedelta(seconds=seconds)
    before = [state for ts, state in ordered if ts <= start]
    in_force = [before[-1]] if before else ([ordered[0][1]] if ordered else [])
    return in_force + [state for ts, state in ordered if start < ts <= dispatch]


def daylight_vio_verdict(
    *,
    sun_elevation: float,
    features_window: list[str],
    status_window: list[str],
    allow_low_sun: bool = False,
    allow_recovered_vio_dip: bool = False,
) -> list[str]:
    """Return halt reasons; an empty list means daylight and VIO are acceptable.

    ``allow_low_sun`` is a narrow, explicit operator override of the sun-elevation
    clause ONLY (docs/findings-phase1-repeat-20260914.md, dusk retest 2026-09-14,
    operator go on that specific named risk). It must never also relax the
    vio_tracked_features or visual_positioning_status clauses below: those, not
    the point-in-time status read, are what actually caught the 2026-09-12
    collapse (14 features in the 60s window vs the 70 floor, while status read
    signal_good throughout). Default False; every other caller is unaffected.
    """
    reasons: list[str] = []
    if sun_elevation < SUN_MIN_ELEVATION_DEG and not allow_low_sun:
        reasons.append(
            f"sun elevation {sun_elevation:.2f} deg below {SUN_MIN_ELEVATION_DEG}"
        )
    features = [int(v) for v in features_window if str(v).lstrip("-").isdigit()]
    if not features:
        reasons.append("no numeric vio_tracked_features in the look-back window")
    elif allow_recovered_vio_dip:
        # Operator rule, 2026-09-16 (predeclaration
        # docs/predeclared-ble-connected-trace-collection-20260916.md §12): a
        # dip inside the window is acceptable once the reading IN FORCE at
        # dispatch is back at the floor. "Stays below" is the driver's job --
        # it re-checks after 60 s a bounded number of times, then stops. The
        # floor itself does not move.
        if features[-1] < VIO_MIN_TRACKED_FEATURES:
            reasons.append(
                f"vio_tracked_features {features[-1]} at dispatch below "
                f"{VIO_MIN_TRACKED_FEATURES} (not recovered; window min "
                f"{min(features)})"
            )
    elif min(features) < VIO_MIN_TRACKED_FEATURES:
        reasons.append(
            f"vio_tracked_features min {min(features)} below "
            f"{VIO_MIN_TRACKED_FEATURES} in the last {VIO_WINDOW_SECONDS}s"
        )
    if not status_window:
        reasons.append("no visual_positioning_status in the look-back window")
    elif any(s != VIO_REQUIRED_STATUS for s in status_window):
        reasons.append(
            f"visual_positioning_status not all {VIO_REQUIRED_STATUS}: "
            f"{sorted(set(status_window))}"
        )
    return reasons


def _points(poly: list[Any]) -> list[tuple[float, float]]:
    return [
        (p[0], p[1]) if isinstance(p, list | tuple) else (p["x"], p["y"]) for p in poly
    ]


def _inside(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y = pt
    hit = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def _dist_to_poly(pt: tuple[float, float], poly: list[tuple[float, float]]) -> float:
    best = math.inf
    for i in range(len(poly)):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % len(poly)]
        dx, dy = bx - ax, by - ay
        length = dx * dx + dy * dy
        t = (
            0.0
            if length == 0
            else max(0.0, min(1.0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / length))
        )
        best = min(best, math.hypot(pt[0] - (ax + t * dx), pt[1] - (ay + t * dy)))
    return best


def _history(
    url: str, token: str, entity: str, seconds: int
) -> list[tuple[datetime.datetime, str]]:
    now = datetime.datetime.now(datetime.UTC)
    start = (now - datetime.timedelta(seconds=seconds + 5)).isoformat()
    query = (
        f"{url}/api/history/period/{urllib.parse.quote(start)}"
        f"?minimal_response&filter_entity_id={entity}"
    )
    request = urllib.request.Request(
        query, headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(request, timeout=40) as response:  # noqa: S310
        data = json.load(response)
    return [
        (datetime.datetime.fromisoformat(p["last_changed"]), p["state"])
        for series in data
        for p in series
    ]


def halt(message: str) -> None:
    """Print a halt reason and exit 2."""
    print("HALT:", message)
    sys.exit(2)


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    """Run one leg."""
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("tx", type=float)
    parser.add_argument("ty", type=float)
    parser.add_argument("role", choices=["setup", "scored"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-low-sun",
        action="store_true",
        help=(
            "Operator override of the sun-elevation clause ONLY (2026-09-14 dusk "
            "retest). vio_tracked_features and visual_positioning_status stay "
            "enforced -- never pass this without a fresh, explicit operator go on "
            "that specific risk."
        ),
    )
    parser.add_argument(
        "--allow-recovered-vio-dip",
        action="store_true",
        help=(
            "Operator rule 2026-09-16: a vio_tracked_features dip inside the "
            "look-back window is accepted if the reading at dispatch is back at "
            "the floor. The floor and the status clause are unchanged."
        ),
    )
    args = parser.parse_args()
    url, token = os.environ["HA_URL"].rstrip("/"), os.environ["HA_TOKEN"]
    args.out.mkdir(parents=True, exist_ok=True)

    siting = json.loads(SITING.read_text())
    area = _points(siting["area_polygon"])
    keep_outs = [_points(v) for v in siting["keep_out_polygons"].values()]
    grid = siting["grid"]

    now = datetime.datetime.now(datetime.UTC)
    reasons = daylight_vio_verdict(
        sun_elevation=solar_elevation_degrees(now),
        features_window=window_values(
            _history(
                url, token, SENSOR_PREFIX + "vio_tracked_features", VIO_WINDOW_SECONDS
            ),
            now,
            VIO_WINDOW_SECONDS,
        ),
        status_window=window_values(
            _history(
                url,
                token,
                SENSOR_PREFIX + "visual_positioning_status",
                VIO_WINDOW_SECONDS,
            ),
            now,
            VIO_WINDOW_SECONDS,
        ),
        allow_low_sun=args.allow_low_sun,
        allow_recovered_vio_dip=args.allow_recovered_vio_dip,
    )
    if reasons:
        halt("daylight/VIO: " + "; ".join(reasons))
    if args.allow_low_sun:
        print(
            f"OVERRIDE: sun-elevation clause bypassed by operator go "
            f"(actual elevation {solar_elevation_degrees(now):.2f} deg); "
            f"VIO tracked-features and status clauses still enforced and clean."
        )

    target = (args.tx, args.ty)
    cell = min(grid, key=lambda r: math.hypot(r["x"] - target[0], r["y"] - target[1]))
    runtime = post_service(
        url, token, "mammotion", "export_runtime_state", {"entity_id": ENTITY}, 90
    )
    pos, facing, blade = runtime["position"], runtime["map_facing"], runtime["blade"]
    start = (pos["x"], pos["y"])
    leg = math.hypot(target[0] - start[0], target[1] - start[1])
    if (
        not pos["valid_for_motion"]
        or pos["pos_type_label"] != "AREA_INSIDE"
        or str(pos["zone_hash"]) != ZONE
    ):
        halt("position not valid / not AREA_INSIDE / wrong zone")
    if pos["rtk_status_label"] != "Fix":
        halt(f"RTK not Fix: {pos['rtk_status_label']}")
    if (
        blade["reported_state_label"] != "OFF"
        or blade["blade_rpm_looks_latched"]
        or blade["safety_blockers"]
    ):
        halt("blade not safe")
    if facing["confidence"] == "unknown":
        halt("facing unknown")
    if leg > MAX_SEGMENT_M:
        halt(f"leg {leg:.2f} m > {MAX_SEGMENT_M}")
    samples = [
        (start[0] + f * (target[0] - start[0]), start[1] + f * (target[1] - start[1]))
        for f in (i / 40 for i in range(41))
    ]
    if any(not _inside(p, area) for p in samples):
        halt("corridor leaves the area")
    corridor_min = min(
        min([_dist_to_poly(p, area)] + [_dist_to_poly(p, k) for k in keep_outs])
        for p in samples
    )
    if corridor_min < MIN_CORRIDOR_CLEARANCE_M:
        halt(
            f"corridor min clearance {corridor_min:.2f} < {MIN_CORRIDOR_CLEARANCE_M} m"
        )
    excursion = math.hypot(target[0] - ANCHOR[0], target[1] - ANCHOR[1])
    if args.role == "scored" and excursion > MAX_EXCURSION_M + 1e-6:
        halt(f"target excursion {excursion:.2f} > {MAX_EXCURSION_M}")
    rssi = cell["rssi"]
    if rssi is None or rssi < MODERATE_MIN_DBM:
        halt(f"target band rejected (rssi {rssi})")
    band = "strong" if rssi >= STRONG_MIN_DBM else "moderate"

    profile = json.loads(PROFILE.read_text())["profile"]

    def call(dry_run: bool) -> dict[str, Any]:
        payload = {
            "entity_id": ENTITY,
            "points": [
                {"x": start[0], "y": start[1]},
                {"x": target[0], "y": target[1]},
            ],
            "area_hash": ZONE,
            "dry_run": dry_run,
            "confirm_blades_off": True,
            "confirm_clear_area": True,
            **profile,
        }
        return post_service(
            url,
            token,
            "mammotion",
            "raw_pymammotion_execute_vector_segment",
            payload,
            400,
        )

    def echoed(key: str, want: Any, got: Any) -> bool:
        if isinstance(want, list):
            return isinstance(got, list) and [float(x) for x in want] == [
                float(x) for x in got
            ]
        if isinstance(want, bool) or not isinstance(want, int | float):
            return want == got
        return isinstance(got, int | float) and abs(float(got) - float(want)) < 1e-9

    dry = call(True)
    mismatched = [k for k, v in profile.items() if not echoed(k, v, dry.get(k))]
    failing = [g["name"] for g in dry.get("safety_gates", []) if not g.get("passed")]
    if (
        not dry.get("valid")
        or dry.get("errors")
        or dry.get("keep_out_violations")
        or dry.get("blockers")
        or mismatched
        or failing
    ):
        halt(
            f"dry run: valid={dry.get('valid')} errors={dry.get('errors')} blockers={dry.get('blockers')} mismatched={mismatched} gates={failing}"
        )

    real = call(False)
    (args.out / f"leg_{args.label}.json").write_text(json.dumps(real, indent=1))
    timing = post_service(
        url,
        token,
        "mammotion",
        "motion_dispatch_timing_report",
        {"entity_id": ENTITY},
        60,
    )
    (args.out / f"timing_after_{args.label}.json").write_text(
        json.dumps(timing, indent=1)
    )
    after = post_service(
        url, token, "mammotion", "export_runtime_state", {"entity_id": ENTITY}, 90
    )["position"]
    meta = {
        "label": args.label,
        "role": args.role,
        "dispatched_utc": now.isoformat(),
        "start": start,
        "target": target,
        "leg_m": round(leg, 3),
        "corridor_min_clearance_m": round(corridor_min, 2),
        "excursion_m": round(excursion, 2),
        "band": band,
        "target_rssi": rssi,
        "facing_confidence": facing["confidence"],
        "stop_reason": real.get("stop_reason"),
        "landing_error_m": round(
            math.hypot(after["x"] - target[0], after["y"] - target[1]), 3
        ),
        "commands_sent": real.get("commands_sent"),
        "motion_refresh_commands_sent": real.get("motion_refresh_commands_sent"),
        "samples_total": timing.get("sample_count"),
        "outcomes": timing.get("outcomes"),
    }
    (args.out / f"meta_{args.label}.json").write_text(json.dumps(meta))
    print(json.dumps(meta))
    if real.get("stop_reason") != "target_reached":
        halt(f"stop_reason {real.get('stop_reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
