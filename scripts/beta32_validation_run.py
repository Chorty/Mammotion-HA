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

from mammotion_ha_helpers import (  # noqa: E402
    load_dotenv,
    post_service,
    warm_ble_link,
)

ENTITY = "lawn_mower.back_yard_clip_skywalker"
REPO = Path(__file__).resolve().parents[1]

#: Junction turn band. Lower bound keeps the turn worth measuring; upper bound
#: stays under the 72 deg ceiling-binding threshold with margin.
JUNCTION_MIN_DEGREES = 45.0
JUNCTION_MAX_DEGREES = 70.0
#: Leg length. Long enough that the linear phase runs multiple pulses and the
#: VIO forward-heading offset gets refreshed (it needs >= 0.05 m of travel),
#: short enough that four of them stay inside a mapped area.
#:
#: 0.9 -> 0.7 on 2026-08-09. `max_linear_commands: 3` at ~0.35 m per pulse gives
#: about 1.05 m of travel budget, and cross-track error makes the real path
#: longer than the straight-line leg -- so a 0.9 m leg leaves roughly 0.15 m of
#: margin and any single bad pulse exhausts it. That is how the 21:02 run ended:
#: segment 3 stopped on `max_linear_commands_reached` 0.164 m from target,
#: 1.4 cm outside tolerance, having driven 102% of the way along the leg.
#:
#: This is a TEST-GEOMETRY change, not a fix. The underlying limit is
#: `max_linear_commands`, which is a frozen `LUBA_ACCEPTANCE_PROFILE` key;
#: shortening the leg buys the same margin without un-accepting the profile.
LEG_METRES = 0.7
#: Turn pattern across the three junctions: right, left, right. Alternating
#: keeps the path compact and exercises both rotation directions, which a
#: single-handed zigzag would not.
JUNCTION_PATTERN = (60.0, -60.0, 60.0)

#: `--reposition`: a U-turn built out of three same-direction 60 deg junctions.
#:
#: A single 180 deg turn is refused pre-dispatch on the accepted profile, twice
#: over: `_vio_turn_budget_feasibility` needs 8 commands against
#: `vio_turn_max_commands: 4`, and the estimated 0.468 m of turn drift exceeds
#: `max_turn_translation_distance: 0.30`. The largest single turn that dispatches
#: is ~114 deg. Both limits are `LUBA_ACCEPTANCE_PROFILE` keys, so raising either
#: would un-accept the profile and owe a fresh Gate 5 -- absurd for a
#: repositioning move. Three 60 deg junctions accumulate the same 180 deg inside
#: the band already validated twice on hardware.
#:
#: Each entry is (leg metres, degrees to turn BEFORE that leg). The two short
#: legs exist only to carry rotation; the long one does the travelling.
REPOSITION_PLAN = ((0.8, 60.0), (0.8, 60.0), (2.0, 60.0))

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
    start: tuple[float, float],
    polygons: dict[str, list[tuple[float, float]]],
    pattern: tuple[float, ...] = JUNCTION_PATTERN,
    prefer_heading: float | None = None,
    leg_metres: float = LEG_METRES,
    segments: int = 4,
) -> tuple[list[dict[str, float]], str, float]:
    """Lay a path of ``segments`` legs from ``start``, all inside one mapped area.

    Sweeps the initial heading and keeps the first orientation whose five points
    all sit inside the same polygon, with a margin check on the midpoints too so
    a leg cannot clip a concave boundary between its endpoints.

    ``pattern`` is the signed turn at each junction. It alternates so the path
    stays compact and both rotation directions get exercised; a same-signed
    pattern would spiral out of the area.

    ``prefer_heading`` orders that sweep by angular distance from where the mower
    is already pointing, instead of always starting at 0 deg. Segment 1's turn is
    NOT a junction -- it runs from the mower's real facing to the first leg's
    bearing, and nothing in the junction preflight covers it. Ignoring the
    current facing let this script demand a pivot the profile refuses: live
    2026-08-09 the mower was left facing ~225 deg after a U-turn, the sweep
    picked 0 deg because it always did, and segment 1 needed **135.017 deg** --
    6 commands against a budget of 4, and 0.351 m of drift against a 0.30 m cap.
    It was refused pre-dispatch (correctly) and the three 90 deg junctions under
    test, all admitted, never ran. Evidence:
    docs/evidence-beta32-4segment-20260809T192923Z.json.

    Containment still wins: a preferred heading that leaves the area is skipped,
    exactly as before. This only reorders which candidate is tried first.

    ``leg_metres`` and ``segments`` exist for the REACH test. Per-segment reach is
    ~1 m on the accepted profile (`max_linear_commands: 3` at ~0.35-0.42 m per
    pulse), so a 2 m leg is not dispatchable and stops on
    `max_linear_commands_reached` -- measured 2026-08-09. Testing whether
    loop-to-tolerance lifts that needs a longer leg, and a longer leg needs FEWER
    of them to stay inside the polygon: four 2 m legs sweep roughly 8 m of path,
    which no area here holds. Both are TEST GEOMETRY and touch no profile key.
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

    candidates = [float(d) for d in range(0, 360, 5)]
    if prefer_heading is not None:
        candidates.sort(key=lambda d: abs((d - prefer_heading + 180) % 360 - 180))
    for initial in candidates:
        heading = float(initial)
        x, y = start
        points = [{"x": round(x, 4), "y": round(y, 4)}]
        samples: list[tuple[float, float]] = [start]
        for index in range(segments):
            # Sample the leg's midpoint as well as its end, so a leg cannot cut
            # a corner across a concave boundary between two inside endpoints.
            # A long leg needs more than a midpoint: at 2 m the gap between
            # samples is itself wider than the 0.7 m legs this was written for,
            # so step the check along the leg instead of assuming three points.
            checks = max(2, int(math.ceil(leg_metres / 0.5)))
            samples.extend(
                (
                    x + leg_metres * (step / checks) * math.cos(math.radians(heading)),
                    y + leg_metres * (step / checks) * math.sin(math.radians(heading)),
                )
                for step in range(1, checks + 1)
            )
            x += leg_metres * math.cos(math.radians(heading))
            y += leg_metres * math.sin(math.radians(heading))
            points.append({"x": round(x, 4), "y": round(y, 4)})
            if index < len(pattern):
                heading += pattern[index]
        if all(_point_in_polygon(px, py, poly) for px, py in samples):
            return points, area_name, float(initial)

    raise SystemExit(
        f"no initial heading keeps all {segments} legs of {leg_metres:.2f} m inside "
        "the area -- move the mower further from the boundary, shorten --leg, or "
        "reduce --segments"
    )


def last_travel_heading() -> float | None:
    """Bearing of the most recent leg this project actually drove, in degrees.

    Used only to lay out `--reposition`, and deliberately NOT read from
    `position.toward`: that field is course-over-ground and latches while the
    mower is stationary, which is why the card refuses to draw it as current
    orientation at all. The bearing of the last leg we ourselves commanded and
    reached is the honest stand-in.

    Being wrong here is cheap. Only SEGMENT 1's turn depends on the starting
    heading; every junction after it is fixed by the waypoint geometry. If the
    real facing differs enough that segment 1's turn exceeds ~114 deg, the turn
    primitive refuses it pre-dispatch with `turn_budget_infeasible` and nothing
    moves.
    """
    # Both run shapes count: whichever we drove most recently is the one whose
    # last leg the mower is actually sitting on. Sorted by mtime, not by name --
    # the two filename prefixes differ, so a name sort would group by prefix and
    # silently return an older run's heading.
    #
    # ⚠️ Only segments that actually DROVE count. Reading the newest file's last
    # segment unconditionally returns the PLANNED bearing of a leg that may never
    # have executed: live 2026-08-09 the 90 deg attempt was refused pre-dispatch
    # with zero linear commands, and this function duly reported the mower facing
    # 0 deg -- the direction of the leg it had just declined to drive -- when it
    # was really facing ~225 deg. `linear_commands_sent > 0` is the test, because
    # a segment that sent no linear command cannot have changed which way the
    # mower points.
    files = sorted(
        [
            *Path(REPO / "docs").glob("evidence-beta32-4segment-*.json"),
            *Path(REPO / "docs").glob("evidence-beta33-reposition-*.json"),
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in files:
        try:
            segments = json.loads(path.read_text())["segments"]
        except KeyError, ValueError:
            continue
        for segment in reversed(segments):
            result = segment.get("result") or {}
            if int(result.get("linear_commands_sent") or 0) <= 0:
                continue
            try:
                start, target = result["true_start"], result["target"]
            except KeyError:
                continue
            return (
                math.degrees(
                    math.atan2(target["y"] - start["y"], target["x"] - start["x"])
                )
                % 360
            )
    return None


def build_reposition_path(
    start: tuple[float, float],
    heading: float,
    polygons: dict[str, list[tuple[float, float]]],
) -> tuple[list[dict[str, float]], str, float]:
    """Lay out the three-junction U-turn from a known start and heading."""
    containing = [
        (name, poly)
        for name, poly in polygons.items()
        if _point_in_polygon(start[0], start[1], poly)
    ]
    if not containing:
        raise SystemExit(f"start point {start} is not inside any mapped area")
    area_name, poly = containing[0]

    x, y = start
    points = [{"x": round(x, 4), "y": round(y, 4)}]
    samples = [start]
    for leg_metres, turn in REPOSITION_PLAN:
        heading = (heading + turn) % 360
        samples.extend(
            (
                x + leg_metres * step * math.cos(math.radians(heading)),
                y + leg_metres * step * math.sin(math.radians(heading)),
            )
            for step in (0.5, 1.0)
        )
        x += leg_metres * math.cos(math.radians(heading))
        y += leg_metres * math.sin(math.radians(heading))
        points.append({"x": round(x, 4), "y": round(y, 4)})

    outside = [p for p in samples if not _point_in_polygon(p[0], p[1], poly)]
    if outside:
        raise SystemExit(
            f"the reposition arc leaves {area_name} at {len(outside)} sampled "
            f"point(s), first {outside[0]} -- drive the mower somewhere with "
            f"more room and retry"
        )
    return points, area_name, heading


def preflight() -> dict[str, Any]:
    """Evaluate every hard gate the run needs and print a pass/fail table."""
    # Wake the link before judging it. `ble_link_live` needs a RECENT outbound
    # send and fails `ble_send_stalled` after 15 s of quiet, so a preflight on a
    # rested link reports the staleness of its own idleness rather than anything
    # about the link. The motion executors never hit this because they start the
    # dense report stream first; a preflight run before any of that has no such
    # luck. Read-only, sends no movement command. Live 2026-08-09 this turned a
    # spurious FAIL on a healthy -62 dBm link into a PASS within 3 seconds.
    warm_ble_link(os.environ["HA_URL"], os.environ["HA_TOKEN"], ENTITY)
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


def _segment_landing_error_m(seg: dict[str, Any]) -> float | None:
    """Distance from where the segment stopped to the waypoint it aimed at.

    ⚠️ This used to read a key named ``landing_error_m`` that THE BACKEND HAS
    NEVER EMITTED, so every run this project has ever done printed
    ``landing=None`` and the single most-watched number in the whole effort had to
    be recomputed by hand from the saved JSON afterwards. Compute it here instead:
    ``target`` is the waypoint the executor drove at, and ``final_telemetry`` is
    the position it came to rest at, both already in every result.

    Returns None when the segment never drove at the waypoint. A pre-dispatch
    refusal still carries both a ``target`` and a ``final_telemetry``, so the
    arithmetic succeeds and yields the UNTOUCHED leg length -- 0.8347 m for the
    turn_budget_infeasible segment of 2026-08-10, which is not a landing error and
    would poison any mean it was averaged into. The discriminator is whether a
    linear command ever ran: no forward pulse, no landing.
    """
    if not int(seg.get("linear_commands_sent") or 0):
        return None
    target = seg.get("target")
    position = (seg.get("final_telemetry") or {}).get("position")
    if not isinstance(target, dict) or not isinstance(position, dict):
        return None
    try:
        return math.dist(
            (float(target["x"]), float(target["y"])),
            (float(position["x"]), float(position["y"])),
        )
    except KeyError, TypeError, ValueError:
        return None


def _summarise(result: dict[str, Any]) -> None:
    """Print the section 6 record. Reads per-item records, never aggregates."""
    print("\n== RESULT ==")
    print(f"  stop_reason        : {result.get('stop_reason')}")
    print(f"  segments executed  : {result.get('real_segments_executed')}")

    print("\n  per-segment landing error (tolerance 0.15):")
    landings: list[float] = []
    for segment in result.get("segments") or []:
        seg = segment.get("result") or {}
        landing = _segment_landing_error_m(seg)
        if landing is not None:
            landings.append(landing)
        shown = f"{landing:.4f}" if landing is not None else "n/a"
        print(
            f"    seg{segment.get('index')}  passed={segment.get('passed')}  "
            f"landing={shown}  stop={seg.get('stop_reason')}"
        )
    if landings:
        print(
            f"    -> max {max(landings):.4f}  mean "
            f"{sum(landings) / len(landings):.4f}  over {len(landings)} landing(s)"
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
    parser.add_argument(
        "--reposition",
        action="store_true",
        help=(
            "U-turn and drive back instead of the straight validation path: "
            "three same-direction 60 deg junctions, because a single 180 deg "
            "turn is refused pre-dispatch on the accepted profile"
        ),
    )
    parser.add_argument(
        "--junction",
        type=float,
        default=None,
        help=(
            "junction turn magnitude in degrees (default 60). Use 90 to test "
            "whether an L-path junction completes: at the rates actually "
            "measured the shipped ceiling reaches 169 deg, so the 45-70 deg "
            "band may be needlessly conservative -- a measurement, not a fix"
        ),
    )
    parser.add_argument(
        "--leg",
        type=float,
        default=LEG_METRES,
        help=(
            f"leg length in metres (default {LEG_METRES}). Per-segment reach is "
            "~1 m on the accepted profile, so anything past that needs "
            "--pulse-ceiling as well or it stops on max_linear_commands_reached"
        ),
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=4,
        help=(
            "number of legs (default 4). Long legs need fewer of them to stay "
            "inside a mapped area"
        ),
    )
    parser.add_argument(
        "--pulse-ceiling",
        type=int,
        default=None,
        help=(
            "enable loop-to-tolerance with this many linear pulses per segment. "
            "⚠️ NOT the accepted profile: max_linear_pulse_ceiling is a frozen "
            "LUBA_ACCEPTANCE_PROFILE key and the card sends null. Use this to "
            "MEASURE whether longer reach works before anyone changes the profile"
        ),
    )
    parser.add_argument(
        "--heading",
        type=float,
        default=None,
        help=(
            "starting heading in map degrees for --reposition; defaults to the "
            "bearing of the last leg this project drove"
        ),
    )
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
    polygons = _load_area_polygons()

    if args.reposition:
        heading = args.heading if args.heading is not None else last_travel_heading()
        if heading is None:
            raise SystemExit("no starting heading available -- pass --heading")
        points, area_name, final_heading = build_reposition_path(
            start, float(heading), polygons
        )
        turned = (final_heading - float(heading)) % 360
        print(f"\n== REPOSITION ==  area={area_name}")
        print(f"  assumed start heading {float(heading):.1f} deg (last leg driven)")
        for index, point in enumerate(points):
            print(f"    p{index}: ({point['x']:.3f}, {point['y']:.3f})")
        net = math.hypot(points[-1]["x"] - start[0], points[-1]["y"] - start[1])
        bearing = (
            math.degrees(
                math.atan2(points[-1]["y"] - start[1], points[-1]["x"] - start[0])
            )
            % 360
        )
        print(
            f"  plan {REPOSITION_PLAN} -> {turned:.0f} deg turned, "
            f"final heading {final_heading:.0f} deg"
        )
        print(f"  net displacement {net:.2f} m at bearing {bearing:.0f} deg")
    else:
        if args.junction is None:
            pattern = JUNCTION_PATTERN[: max(0, args.segments - 1)]
            band = "all inside the validated 45-70 deg band"
        else:
            magnitude = abs(args.junction)
            # Alternate the sign: a same-signed pattern spirals out of the area.
            pattern = tuple(
                magnitude * (-1.0) ** index for index in range(args.segments - 1)
            )
            band = (
                "inside the validated band"
                if JUNCTION_MIN_DEGREES <= magnitude <= JUNCTION_MAX_DEGREES
                else f"OUTSIDE the validated {JUNCTION_MIN_DEGREES:.0f}-"
                f"{JUNCTION_MAX_DEGREES:.0f} deg band -- this IS the experiment"
            )
        facing = args.heading if args.heading is not None else last_travel_heading()
        points, area_name, initial_heading = build_path(
            start,
            polygons,
            pattern,
            prefer_heading=facing,
            leg_metres=args.leg,
            segments=args.segments,
        )
        print(
            f"\n== PATH ==  area={area_name}  initial heading={initial_heading:.0f} deg"
        )
        if facing is not None:
            opening = abs((initial_heading - facing + 180) % 360 - 180)
            print(
                f"  mower faces ~{facing:.0f} deg, so segment 1's opening turn is "
                f"~{opening:.0f} deg (must stay under ~114 to dispatch)"
            )
        for index, point in enumerate(points):
            print(f"    p{index}: ({point['x']:.3f}, {point['y']:.3f})")
        print(f"  junction pattern: {pattern} ({band})")

    payload = {
        "points": points,
        # Cap at what this path actually has, not a constant: an honest cap
        # means `max_real_segments_reached` can only ever mean the backend
        # limit, never a stale number in this script.
        "max_real_segments": len(points) - 1,
        **ACCEPTANCE_PROFILE,
    }
    if args.pulse_ceiling is not None:
        # Loop-to-tolerance. The backend branches on this key being present at
        # all (`loop_to_tolerance = max_linear_pulse_ceiling is not None`), so
        # sending it changes the linear phase from a fixed budget to a loop --
        # and `max_linear_commands` stops being the binding limit.
        payload["max_linear_pulse_ceiling"] = int(args.pulse_ceiling)
        print(
            "\n⚠️  CUSTOMISED PROFILE -- this run is NOT the hardware-accepted "
            "profile.\n"
            f"    max_linear_pulse_ceiling={args.pulse_ceiling} enables "
            "loop-to-tolerance;\n"
            "    the card sends null. Results do NOT compare to Gate 5, and "
            "adopting this\n"
            "    would un-accept the profile and owe a fresh Gate 5."
        )
    if abs(args.leg - LEG_METRES) > 1e-9 or args.segments != 4:
        print(
            f"    test geometry: {args.segments} x {args.leg:.2f} m legs "
            f"(default 4 x {LEG_METRES:.2f}) -- geometry only, no profile key."
        )
    if args.leg > 1.0 and args.pulse_ceiling is None:
        print(
            f"\n⚠️  A {args.leg:.2f} m leg exceeds the ~1 m per-segment reach of "
            "`max_linear_commands: 3`.\n"
            "    Expect `max_linear_commands_reached` short of the waypoint. Pass "
            "--pulse-ceiling to test the loop."
        )

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
    name = "beta33-reposition" if args.reposition else "beta32-4segment"
    out = REPO / "docs" / f"evidence-{name}-{stamp}.json"
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
