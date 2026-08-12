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
* **The mower's facing is derived twice and the two must agree**, or the script
  refuses to lay out a path at all. Getting this wrong built a backwards path
  twice on 2026-08-10 and cost two daylight runs. See ``resolve_start_facing``.
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
    # Adopted 2026-08-12. Loop-to-tolerance is now part of the accepted profile,
    # so a default run here is a reach-enabled run. ⚠️ The Gate 5 re-pass on this
    # profile has NOT been done.
    "max_linear_pulse_ceiling": 14,
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

#: `toward` is a COMPASS bearing (clockwise from north) while map headings are
#: math angles (counter-clockwise from +x), so the map facing is the MIRROR of
#: `toward` about this value -- `(90.13 - toward) % 360` -- and NOT any additive
#: offset. No constant added to `toward` can ever work; that is why the legacy
#: path's `+102.4` mis-aimed by ~10 deg and why this is written as a subtraction.
#:
#: Checked against every calibration drive this project has recorded, comparing
#: the mirror of the pre-run `toward` against the facing the drive then measured:
#:
#:     20260810T002506  toward  176.0868  measured 274.160  mirror 274.0432  0.117
#:     20260810T185433  toward  174.0572  measured 278.811  mirror 276.0728  2.738
#:     20260810T193833  toward   33.5651  measured  55.099  mirror  56.5649  1.466
#:     20260810T205514  toward -173.9049  measured 266.712  mirror 264.0349  2.677
#:     20260810T205937  toward -173.9049  measured 263.856  mirror 264.0349  0.179
#:     20260810T232848  toward  173.2761  measured 277.416  mirror 276.8539  0.562
#:     20260811T001250  toward  122.6853  measured 326.772  mirror 327.4447  0.673
#:
#: Seven for seven, worst residual **2.738 deg**. Pinned by a test.
TOWARD_MIRROR_DEGREES = 90.13

#: How far the two independent facing estimates -- the live mirror and the
#: bearing of the last leg we ourselves drove -- may disagree before this script
#: refuses to lay out a path.
#:
#: The failure this guards against is not subtle. On 2026-08-10 the operator
#: repositioned the mower from the app three times; `last_travel_heading()`
#: cannot see that and kept reporting the bearing of OUR last leg, so the path
#: was built backwards and twice the run was refused pre-dispatch with
#: `turn_budget_infeasible` at a ~177 deg opening turn. Two daylight runs were
#: spent finding that out by hand.
#:
#: 15 sits >5x above the worst agreement ever measured (2.738) and an order of
#: magnitude below the failure mode (~177), so it separates the two cleanly
#: without being a tuned number.
FACING_DISAGREEMENT_LIMIT_DEGREES = 15.0


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

    One of the two facing estimates `resolve_start_facing()` holds against each
    other; never used alone if the mirror is available. Its blind spot is
    specific and was expensive: it can only see motion THIS PROJECT commanded,
    so an app-driven reposition leaves it confidently reporting a bearing the
    mower abandoned minutes ago.

    ⚠️ An earlier version of this docstring claimed being wrong here is cheap,
    on the reasoning that only segment 1's turn depends on it and an impossible
    turn gets refused pre-dispatch. **Both halves have since failed.** The
    refusal did fire -- twice on 2026-08-10, at ~177 deg openings -- but each
    one cost a preflight, an arming cycle and a slice of the daylight window,
    and the operator had no way to tell a bad path from a bad build. And since
    beta41 the refusal is no longer a backstop at all: an opening turn that the
    primitive declines is now DECOMPOSED into <=60 deg stages and driven, so a
    backwards path executes instead of being caught.
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


def mirror_facing(position: dict[str, Any]) -> float | None:
    """Map facing implied by the live `toward`, or None if it is unreadable.

    See ``TOWARD_MIRROR_DEGREES`` for the relation and its validation. This is
    only as good as `toward` is fresh: the field is course-over-ground and
    LATCHES while the mower is stationary, so it reports the direction of the
    last travel, whoever commanded it. That is exactly what makes it useful
    here -- an app-driven reposition updates it while `last_travel_heading()`
    goes silently stale -- and exactly what makes it useless if the mower was
    carried rather than driven, which nothing on this device can detect.
    """
    toward = position.get("toward")
    if toward is None:
        return None
    try:
        return (TOWARD_MIRROR_DEGREES - float(toward)) % 360
    except TypeError, ValueError:
        return None


def resolve_start_facing(
    position: dict[str, Any], override: float | None
) -> tuple[float, str]:
    """Agree two independent facing estimates, or refuse to build a path.

    Returns ``(facing_degrees, provenance)``. Raises ``SystemExit`` rather than
    guessing, because a wrong facing here does not fail safe in any useful
    sense: it lays out a plausible-looking path pointing the wrong way, burns
    the arming step and the daylight, and is only caught -- if we are lucky --
    by the turn primitive refusing a huge opening turn pre-dispatch.

    The two estimates fail in different ways, which is the whole point of
    holding them against each other:

    * ``last_travel_heading()`` reads the last leg THIS PROJECT drove out of its
      own evidence files. It is blind to any movement commanded elsewhere.
    * the mirror of ``toward`` is live telemetry. It sees an app-driven move,
      but latches when the mower is stationary and cannot see a carried mower.

    So when they agree, nothing has moved the mower since our last run and both
    are right. When they disagree, something moved it: the mirror saw the move
    and the evidence files did not, so the MIRROR is the one to trust -- but
    only if the mower was driven. That is a judgement about the physical world
    this script cannot make, so it stops and puts the choice in front of the
    operator with both numbers in hand.
    """
    mirror = mirror_facing(position)
    driven = last_travel_heading()

    print("\n== FACING ==")
    print(
        f"  mirror of live `toward`   : "
        f"{'unavailable' if mirror is None else f'{mirror:7.2f} deg'}"
        f"   (toward={position.get('toward')})"
    )
    print(
        f"  last leg we actually drove: "
        f"{'unavailable' if driven is None else f'{driven:7.2f} deg'}"
    )

    if override is not None:
        facing = override % 360
        print(f"  -> USING --heading {facing:.2f} deg (operator override)")
        for label, estimate in (("mirror", mirror), ("last leg", driven)):
            if estimate is not None:
                gap = abs((estimate - facing + 180) % 360 - 180)
                print(f"     {label} disagrees with it by {gap:.2f} deg")
        return facing, "operator_override"

    if mirror is not None and driven is not None:
        disagreement = abs((mirror - driven + 180) % 360 - 180)
        if disagreement > FACING_DISAGREEMENT_LIMIT_DEGREES:
            raise SystemExit(
                f"\nREFUSING TO BUILD A PATH: the two facing estimates disagree by "
                f"{disagreement:.1f} deg,\nwhich is past the "
                f"{FACING_DISAGREEMENT_LIMIT_DEGREES:.0f} deg limit (worst honest "
                f"agreement ever measured: 2.7 deg).\n\n"
                f"Something moved the mower since our last run -- almost certainly "
                f"the app.\n"
                f"  if it was DRIVEN (app, joystick, its own dock return), `toward` "
                f"is fresh and the\n"
                f"  mirror is right:            --heading {mirror:.2f}\n"
                f"  if it was CARRIED or nudged in place, BOTH are stale. Drive it a "
                f"metre in a\n"
                f"  straight line first, then re-run this with no --heading at all.\n\n"
                f"Do not guess. A backwards path cost two daylight runs on 2026-08-10."
            )
        print(
            f"  -> agree within {disagreement:.2f} deg; using the mirror "
            f"{mirror:.2f} deg (live)"
        )
        return mirror, "mirror_corroborated_by_last_leg"

    if mirror is not None:
        print(
            f"  -> using the mirror {mirror:.2f} deg, UNCORROBORATED "
            "(no driven leg on record to check it against)"
        )
        return mirror, "mirror_uncorroborated"

    if driven is not None:
        print(
            f"  -> using the last driven leg {driven:.2f} deg, UNCORROBORATED "
            "(`toward` unreadable, so the live check is unavailable)"
        )
        return driven, "last_leg_uncorroborated"

    raise SystemExit(
        "\nREFUSING TO BUILD A PATH: no facing is available from either source.\n"
        "`toward` is unreadable and no evidence file records a leg that drove.\n"
        "Drive the mower a metre in a straight line and re-run, or pass --heading "
        "with a facing you have measured."
    )


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


def blocking_reasons(motion: dict[str, Any]) -> list[str]:
    """Backend blockers that a preflight should fail on.

    Everything the gate reports EXCEPT `experimental_motion_disabled`, which is
    the normal resting posture and is precisely what `--arm` is about to clear.
    """
    return [
        blocker
        for blocker in (motion.get("blockers") or [])
        if blocker != "experimental_motion_disabled"
    ]


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
        (
            # Ask the BACKEND what would stop it, rather than only re-deriving
            # the answer from entities. On 2026-08-11 every check above passed
            # -- including "BLE link live", whose binary_sensor still read `on`
            # -- while `blockers` already carried `ble_client_not_connected` and
            # `ble_rssi` read 0. The run armed, was refused by the gate, and the
            # only warning had been a line of informational print. The gate's
            # own blocker list is the authoritative answer; anything in it other
            # than the disarmed-gate entry is a real preflight failure.
            "no backend blockers",
            not blocking_reasons(motion),
            f"blockers={motion.get('blockers')}",
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
            "the mower's true facing in map degrees, overriding both automatic "
            "estimates. Without it the script derives the facing from the live "
            "`toward` mirror, cross-checks it against the last leg it drove, and "
            "REFUSES to build a path if the two disagree -- pass this to settle it"
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

    facing, facing_source = resolve_start_facing(position, args.heading)

    if args.reposition:
        heading = facing
        points, area_name, final_heading = build_reposition_path(
            start, float(heading), polygons
        )
        turned = (final_heading - float(heading)) % 360
        print(f"\n== REPOSITION ==  area={area_name}")
        print(f"  assumed start heading {float(heading):.1f} deg ({facing_source})")
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
        opening = abs((initial_heading - facing + 180) % 360 - 180)
        print(
            f"  mower faces ~{facing:.0f} deg ({facing_source}), so segment 1's "
            f"opening turn is ~{opening:.0f} deg"
        )
        # beta41 decomposes an opening turn into <=60 deg stages instead of
        # refusing it, so the old ~114 deg dispatch ceiling no longer applies to
        # segment 1 -- a 165.048 deg opening turn completed on hardware
        # 2026-08-10. Still worth seeing: a staged turn spends its whole
        # translation budget across the stages, so a large opening turn starts
        # the first leg further off its bearing than a small one.
        if opening > 114.0:
            print(
                "    (over ~114 deg: the direct turn will be refused and beta41 "
                "will stage it)"
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
    accepted_ceiling = ACCEPTANCE_PROFILE["max_linear_pulse_ceiling"]
    if args.pulse_ceiling is not None and int(args.pulse_ceiling) != accepted_ceiling:
        # Loop-to-tolerance is now part of the accepted profile, so overriding
        # the ceiling is what leaves it -- the opposite of before 2026-08-12.
        payload["max_linear_pulse_ceiling"] = int(args.pulse_ceiling)
        print(
            "\n⚠️  CUSTOMISED PROFILE -- this run is NOT the hardware-accepted "
            "profile.\n"
            f"    max_linear_pulse_ceiling={args.pulse_ceiling} overrides the "
            f"accepted {accepted_ceiling};\n"
            "    results do not compare to a Gate 5 run on the accepted profile."
        )
    if abs(args.leg - LEG_METRES) > 1e-9 or args.segments != 4:
        print(
            f"    test geometry: {args.segments} x {args.leg:.2f} m legs "
            f"(default 4 x {LEG_METRES:.2f}) -- geometry only, no profile key."
        )
    print(
        f"\n  loop-to-tolerance ON at max_linear_pulse_ceiling="
        f"{payload['max_linear_pulse_ceiling']} (accepted 2026-08-12). "
        "⚠️ Gate 5 re-pass on this profile is PENDING."
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
    # ⚠️ Set BEFORE the enable, never after. This flag does not mean "the gate
    # opened", it means "this script may have touched the gate and therefore
    # owes a disarm". Keying it off the readback -- which is what it used to do
    # -- leaves the gate ENABLED on every path where the enable succeeds but
    # `real_motion_allowed` comes back false, because the `finally` then
    # declines to clean up. That fired for real on 2026-08-11: BLE dropped
    # between the preflight and the arm, `real_motion_allowed` read false on
    # `ble_client_not_connected`, the script aborted "without sending anything"
    # -- and left `enabled: true` behind it, one BLE reconnect away from an
    # unattended open gate. Setting it first also covers a crash or a Ctrl-C
    # landing mid-enable.
    armed = True
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
        motion_now = verify.get("experimental_motion", {})
        allowed = motion_now.get("real_motion_allowed")
        print(f"  real_motion_allowed = {allowed}")
        if not allowed:
            print(
                "  gate did not open -- aborting without sending anything.\n"
                f"  blockers: {motion_now.get('blockers')}"
            )
            return 1

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
        # Trailing newline because these files are committed and `pre-commit`'s
        # end-of-file-fixer rewrites them otherwise -- which would mean the repo
        # hook editing an evidence file after the fact, and evidence files are
        # meant to be exactly what the mower returned.
        out.write_text(json.dumps(result, indent=1) + "\n")
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
