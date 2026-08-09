#!/usr/bin/env python3
"""Morning harness for the beta32 four-segment reach validation run.

Everything the run needs, in one command, so no step is improvised at the mower.
It is SAFE BY DEFAULT: without ``--arm`` it never enables the motion gate and
never sends a movement command, so the preflight and the path preview can be run
as many times as you like.

    scripts/beta32_validation_run.py                 # preflight + dry run only
    scripts/beta32_validation_run.py --arm           # the real run

Design rules this encodes, each of which was learned the expensive way:

* **The response JSON is written to disk before anything parses it.** Gate 5
  attempt 5's per-command record existed only in a browser pane and had to be
  reconstructed by hand days later. The raw payload is saved first, always,
  including on failure.
* **Disarm runs in a ``finally``.** A crash, a timeout or a Ctrl-C must not
  leave the motion gate open.
* **Junctions are held to 45-70 degrees.** Below 72 the beta31 overshoot ceiling
  is the active bound from the first pulse of every turn, so this is maximum
  exposure to the new code while staying clear of the 86-100 degree band where
  the 4-command turn budget and the rate floor are both in doubt. See
  docs/HANDOVER-beta31-20260809.md sections 2.2 and 2.6.
* **The path is built from the LIVE position** and every point is checked inside
  the area polygon before anything is dispatched, because a path that fails
  validation at the mower wastes the daylight window.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os  # noqa: E402

from mammotion_ha_helpers import load_dotenv, post_service  # noqa: E402

ENTITY = "lawn_mower.back_yard_clip_skywalker"
REPO = Path(__file__).resolve().parents[1]

#: Junction turn band. Lower bound keeps the turn worth measuring; upper bound
#: stays under the 72 deg ceiling-binding threshold with margin.
JUNCTION_MIN_DEGREES = 45.0
JUNCTION_MAX_DEGREES = 70.0
#: Leg length. Long enough that the linear phase runs multiple pulses and the
#: VIO forward-heading offset gets refreshed (it needs >= 0.05 m of travel),
#: short enough that four of them stay inside a mapped area.
LEG_METRES = 0.9
#: Turn pattern across the three junctions: right, left, right. Alternating
#: keeps the path compact and exercises both rotation directions, which a
#: single-handed zigzag would not.
JUNCTION_PATTERN = (60.0, -60.0, 60.0)

#: The accepted profile, sent explicitly. Mirrors LUBA_ACCEPTANCE_PROFILE in
#: www/mammotion-custom-path-card.js -- if these drift the run is NOT the
#: hardware-accepted profile and the result does not compare to Gate 5.
ACCEPTANCE_PROFILE: dict[str, Any] = {
    "prefer_ble": True,
    "turn_mode": "vio",
    "max_turn_commands": 4,
    "vio_turn_max_commands": 4,
    "max_linear_commands": 3,
    "max_no_progress_pulses": 3,
    "heading_tolerance_degrees": 18,
    "waypoint_tolerance": 0.15,
    "min_progress_distance": 0.0025,
    "max_turn_translation_distance": 0.3,
    "calibrated_forward_heading_offset_degrees": 102.4,
    "turn_pulse_duration_ms": 1500,
    "linear_pulse_duration_ms": 1300,
    "motion_refresh_interval_ms": 200,
    "final_approach_metres_per_pulse": 1.06,
    "turn_degrees_per_second": 37,
    "ble_auto_recover": False,
    "sample_delays": [0, 3],
}

#: Hard preflight gates. Each is a (label, predicate, detail) triple evaluated
#: against the runtime state; every one must pass before --arm proceeds.
MIN_TRACKED_FEATURES = 70
MIN_BATTERY_PERCENT = 30


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _call(service: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    return post_service(
        os.environ["HA_URL"],
        os.environ["HA_TOKEN"],
        "mammotion",
        service,
        {"entity_id": ENTITY, **payload},
        timeout,
    )


def _state(entity: str) -> str | None:
    """Read one HA entity state, or None if it cannot be read."""
    req = urllib.request.Request(
        f"{os.environ['HA_URL'].rstrip('/')}/api/states/{entity}",
        headers={"Authorization": f"Bearer {os.environ['HA_TOKEN']}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response).get("state")
    except Exception:  # noqa: BLE001
        return None


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _load_area_polygons() -> dict[str, list[tuple[float, float]]]:
    raw = _call("get_map_data", {}, timeout=180)
    names = {n["hash"]: n["name"] for n in raw.get("area_name", [])}
    polygons: dict[str, list[tuple[float, float]]] = {}
    for area_hash, blob in (raw.get("area") or {}).items():
        points = [
            (couple["x"], couple["y"])
            for frame in blob.get("data", [])
            for couple in frame.get("data_couple", [])
            if isinstance(couple, dict) and "x" in couple and "y" in couple
        ]
        if points:
            polygons[names.get(area_hash, area_hash)] = points
    return polygons


def build_path(
    start: tuple[float, float], polygons: dict[str, list[tuple[float, float]]]
) -> tuple[list[dict[str, float]], str, float]:
    """Lay a 4-segment path from ``start`` that stays inside one mapped area.

    Sweeps the initial heading and keeps the first orientation whose five points
    all sit inside the same polygon, with a margin check on the midpoints too so
    a leg cannot clip a concave boundary between its endpoints.
    """
    containing = [
        (name, poly)
        for name, poly in polygons.items()
        if _point_in_polygon(start[0], start[1], poly)
    ]
    if not containing:
        raise SystemExit(
            f"start point {start} is not inside any mapped area -- drive the "
            f"mower into a mapped area before running (it is probably still docked)"
        )
    area_name, poly = containing[0]

    for initial in range(0, 360, 5):
        heading = float(initial)
        x, y = start
        points = [{"x": round(x, 4), "y": round(y, 4)}]
        samples: list[tuple[float, float]] = [start]
        for index in range(4):
            # Sample the leg's midpoint as well as its end, so a leg cannot cut
            # a corner across a concave boundary between two inside endpoints.
            samples.extend(
                (
                    x + LEG_METRES * step * math.cos(math.radians(heading)),
                    y + LEG_METRES * step * math.sin(math.radians(heading)),
                )
                for step in (0.5, 1.0)
            )
            x += LEG_METRES * math.cos(math.radians(heading))
            y += LEG_METRES * math.sin(math.radians(heading))
            points.append({"x": round(x, 4), "y": round(y, 4)})
            if index < len(JUNCTION_PATTERN):
                heading += JUNCTION_PATTERN[index]
        if all(_point_in_polygon(px, py, poly) for px, py in samples):
            return points, area_name, float(initial)

    raise SystemExit(
        "no initial heading keeps all four legs inside the area -- move the "
        "mower further from the boundary and retry"
    )


def preflight() -> dict[str, Any]:
    """Evaluate every hard gate the run needs and print a pass/fail table."""
    runtime = _call("export_runtime_state", {}, timeout=120)
    motion = runtime.get("experimental_motion", {})
    safety = runtime.get("safety", {})
    position = runtime.get("position", {})
    feed = runtime.get("vio", {}) or {}

    tracked = _state("sensor.back_yard_clip_skywalker_vio_tracked_features")
    battery = _state("sensor.back_yard_clip_skywalker_battery")
    tracked_n = float(tracked) if tracked not in (None, "unknown", "unavailable") else 0
    battery_n = float(battery) if battery not in (None, "unknown", "unavailable") else 0

    checks = [
        (
            "daylight / VIO feed",
            tracked_n >= MIN_TRACKED_FEATURES,
            f"tracked_features={tracked_n:.0f} (need >= {MIN_TRACKED_FEATURES}); "
            f"brightness={_state('sensor.back_yard_clip_skywalker_camera_brightness')}",
        ),
        (
            # `rtk_status_label` lives under `safety`/`position`, NOT at the top
            # level -- reading it from the root silently yields None and would
            # have failed this gate on a healthy Fix.
            "RTK precise",
            safety.get("rtk_status_label") == "Fix",
            f"rtk_status_label={safety.get('rtk_status_label')} "
            f"(entity: {_state('sensor.back_yard_clip_skywalker_rtk_position')})",
        ),
        (
            "position valid for motion",
            bool(safety.get("position_valid_for_motion")),
            f"pos_type={position.get('pos_type_label')} "
            f"x={position.get('x')} y={position.get('y')}",
        ),
        (
            "blade safe",
            _state("binary_sensor.back_yard_clip_skywalker_blade_safe_for_motion")
            == "on",
            "blade_safe_for_motion",
        ),
        (
            "BLE link live",
            _state("binary_sensor.back_yard_clip_skywalker_ble_link_live") == "on",
            f"ble_rssi={_state('sensor.back_yard_clip_skywalker_ble_rssi')}",
        ),
        (
            "work mode ready",
            runtime.get("work_mode_label") in {"MODE_READY", "MODE_PAUSE"},
            f"work_mode_label={runtime.get('work_mode_label')}",
        ),
        (
            "battery",
            battery_n >= MIN_BATTERY_PERCENT,
            f"battery={battery_n:.0f}% (need >= {MIN_BATTERY_PERCENT}%)",
        ),
        (
            "no active session",
            not motion.get("active_session"),
            f"active_session={motion.get('active_session')}",
        ),
    ]

    print("\n== PREFLIGHT ==")
    failed = []
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:26s} {detail}")
        if not ok:
            failed.append(label)
    print(
        f"\n  segment limit={motion.get('real_click_to_go_segment_limit')} "
        f"real_motion_allowed={motion.get('real_motion_allowed')} "
        f"blockers={motion.get('blockers')}"
    )
    print(f"  vio feed: {feed if feed else runtime.get('initial_vio_feed')}")
    print(f"  work_mode={runtime.get('work_mode_label')} battery={battery_n:.0f}%")
    return {"runtime": runtime, "failed": failed, "position": position}


def _summarise(result: dict[str, Any]) -> None:
    """Print the section 6 record. Reads per-item records, never aggregates."""
    print("\n== RESULT ==")
    print(f"  stop_reason        : {result.get('stop_reason')}")
    print(f"  segments executed  : {result.get('real_segments_executed')}")

    print("\n  per-segment landing error (tolerance 0.15):")
    for segment in result.get("segments") or []:
        seg = segment.get("result") or {}
        landing = seg.get("landing_error_m", segment.get("landing_error_m"))
        print(
            f"    seg{segment.get('index')}  passed={segment.get('passed')}  "
            f"landing={landing}  stop={seg.get('stop_reason')}"
        )

    print("\n  turn commands BROKEN DOWN BY PHASE (never as one number):")
    for segment in result.get("segments") or []:
        seg = segment.get("result") or {}
        realigns = seg.get("realignments") or []
        print(
            f"    seg{segment.get('index')}  turn_commands_sent="
            f"{seg.get('turn_commands_sent')}  linear={seg.get('linear_commands_sent')}"
            f"  realignments={len(realigns)}  "
            f"post_turn={bool(seg.get('post_turn_alignment'))}"
        )

    print("\n  every turn pulse: heading_error_after and the ceiling's verdict")
    for segment in result.get("segments") or []:
        seg = segment.get("result") or {}
        for command in seg.get("command_results") or []:
            approach = command.get("final_approach")
            if not isinstance(approach, dict):
                continue
            refresh = command.get("motion_refresh") or {}
            print(
                f"    seg{segment.get('index')} cmd{command.get('index')}  "
                f"err_before={command.get('heading_error_before')} -> "
                f"err_after={command.get('heading_error_after')}  "
                f"pulse={command.get('pulse_duration_ms')} "
                f"elapsed={refresh.get('elapsed_ms')}  "
                f"reason={approach.get('reason')}"
            )

    stops = {
        (segment.get("result") or {}).get("stop_reason")
        for segment in result.get("segments") or []
    }
    for bad in ("target_requires_reverse_recovery", "vio_realign_budget_exhausted"):
        print(f"\n  {bad}: {'PRESENT -- investigate' if bad in stops else 'none'}")


def main() -> int:  # noqa: C901
    """Preflight, preview, and -- only with ``--arm`` -- execute and disarm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="store_true",
        help="actually run it: enables the motion gate, executes, then disarms",
    )
    parser.add_argument("--skip-preflight-gate", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    for required in ("HA_URL", "HA_TOKEN"):
        if not os.environ.get(required):
            raise SystemExit(f"{required} missing -- `set -a && source .env && set +a`")

    state = preflight()
    if state["failed"] and not args.skip_preflight_gate:
        print(f"\nPREFLIGHT FAILED: {', '.join(state['failed'])}")
        if args.arm:
            print("refusing to arm.")
            return 1

    position = state["position"]
    start = (float(position["x"]), float(position["y"]))
    points, area_name, initial_heading = build_path(start, _load_area_polygons())
    print(f"\n== PATH ==  area={area_name}  initial heading={initial_heading:.0f} deg")
    for index, point in enumerate(points):
        print(f"    p{index}: ({point['x']:.3f}, {point['y']:.3f})")
    print(f"  junction pattern: {JUNCTION_PATTERN} (all inside 45-70 deg band)")

    payload = {
        "points": points,
        "max_real_segments": 4,
        **ACCEPTANCE_PROFILE,
    }

    print("\n== DRY RUN (zero motion) ==")
    dry = _call(
        "raw_pymammotion_execute_multi_segment", {**payload, "dry_run": True}, 180
    )
    print(f"  valid={dry.get('valid')} errors={dry.get('errors')}")
    print(f"  stop_reason={dry.get('stop_reason')} would_send={dry.get('would_send')}")
    for junction in dry.get("junction_turn_feasibility") or []:
        feasibility = junction["feasibility"]
        print(
            f"    seg{junction['segment_index']} turn={junction['turn_degrees']:>8}  "
            f"feasible={feasibility['feasible']}  "
            f"needed={feasibility['estimated_commands_needed']}/"
            f"{feasibility['max_commands']}  "
            f"pulses={feasibility.get('modelled_pulse_durations_ms')}"
        )
    if not dry.get("valid"):
        print("\nDRY RUN INVALID -- not proceeding.")
        return 1

    if not args.arm:
        print("\nPreview only. Re-run with --arm to execute (this is the safe exit).")
        return 0

    stamp = _now()
    out = REPO / "docs" / f"evidence-beta32-4segment-{stamp}.json"
    armed = False
    try:
        print("\n== ARM ==")
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/ha_set_experimental_motion.py"),
                "on",
                "--yes",
            ],
            check=True,
        )
        verify = _call("export_runtime_state", {}, timeout=120)
        allowed = verify.get("experimental_motion", {}).get("real_motion_allowed")
        print(f"  real_motion_allowed = {allowed}")
        if not allowed:
            print("  gate did not open -- aborting without sending anything.")
            return 1
        armed = True

        print("\n== EXECUTE ==")
        started = time.monotonic()
        result = _call(
            "raw_pymammotion_execute_multi_segment",
            {
                **payload,
                "dry_run": False,
                "confirm_blades_off": True,
                "confirm_clear_area": True,
            },
            600,
        )
        # SAVE FIRST. Nothing above may parse this before it is on disk.
        out.write_text(json.dumps(result, indent=1))
        print(f"  wall clock {time.monotonic() - started:.1f} s")
        print(f"  COMPLETE RESPONSE SAVED -> {out.relative_to(REPO)}")
        _summarise(result)
    except BaseException as err:  # noqa: BLE001
        print(f"\n!! {type(err).__name__}: {err}")
        raise
    finally:
        if armed:
            print("\n== DISARM ==")
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts/ha_set_experimental_motion.py"),
                    "off",
                    "--yes",
                ],
                check=False,
            )
            final = _call("export_runtime_state", {}, timeout=120)
            motion = final.get("experimental_motion", {})
            print(
                f"  enabled={motion.get('enabled')} "
                f"real_motion_allowed={motion.get('real_motion_allowed')}"
            )
            if motion.get("real_motion_allowed"):
                print("  !! GATE STILL OPEN -- disarm by hand immediately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
