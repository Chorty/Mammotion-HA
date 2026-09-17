#!/usr/bin/env python3
"""Drive a ~2 m x 4 left-turn square on raw timed pulses; measure RTK closure.

Predeclared in ``docs/predeclared-rtk-square-return-20260916.md`` -- read that
first; it fixes the parameters and the scoring before any data existed.

🚨 **This commands REAL MOTION.** It arms the experimental-motion gate, drives,
and **always disarms in a finally block**, including on an abort or an
exception. Every pulse is individually bounded by its own immediate stop.

Why raw pulses and not the vector executor: with ``vio_tracked_features`` at 0
(fully dark), the executor and both VIO turn paths refuse -- correctly.
``manual_velocity_pulse_test`` has no VIO gate and needs no heading, so a
square can still be driven open-loop and measured by RTK.

🔑 **Heading comes from RTK displacement, not a compass.** The bearing actually
driven on a leg is ``atan2(dy, dx)`` across that leg, so the angle a turn
really achieved is only known once the *next* leg has been driven. Turn 1 is an
open-loop guess; turns 2-4 use the rate measured from the turn before.

🚨 ``motion_refresh_interval_ms`` is 200 on every pulse. At 0 a pulse travels
~4 in regardless of duration (2026-07-22 tape test, ~11x difference) -- a run
without refresh is void.

Usage:  .venv/bin/python scripts/rtk_square_return.py --out DIR [--dry-run]
"""  # noqa: INP001

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mammotion_ha_helpers import load_dotenv, post_service  # noqa: E402

ENTITY = "lawn_mower.back_yard_clip_skywalker"

#: Predeclaration section 3 -- fixed, do not tune mid-run.
PULSE_SPEED = 0.6
PULSE_MS = 4000
REFRESH_MS = 200
LEG_TARGET_M = 2.0
TURN_TARGET_DEG = 90.0

#: Safety bounds. A leg that overshoots this far has lost the plot; stop.
LEG_ABORT_M = 3.5
MAX_PULSES_PER_LEG = 8
MAX_PULSES_PER_TURN = 4

#: Opening guess for pure rotation at angular 202. Banked points are angular
#: 120 -> 9.175 deg/s and 180 -> 13.431 deg/s, both at linear 300 and n = 1 --
#: NOT a law, just a starting duration. Turn 1's measured result replaces it.
INITIAL_TURN_RATE_DEG_S = 15.0


def emit(message: str) -> None:
    """Print one line of progress output."""
    print(message, flush=True)  # noqa: T201


def now_iso() -> str:
    """UTC timestamp for the record."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def read_state(url: str, token: str) -> dict[str, Any]:
    """Return the mower's runtime state, retrying an empty service response."""
    for _ in range(5):
        state = post_service(
            url, token, "mammotion", "export_runtime_state", {"entity_id": ENTITY}, 90
        )
        if state.get("position"):
            return state
        time.sleep(4)
    raise RuntimeError("export_runtime_state returned no position five times")


def preflight(state: dict[str, Any]) -> list[str]:
    """Return abort reasons; empty means the run may start."""
    position = state["position"]
    blade = state["blade"]
    reasons: list[str] = []
    if position["rtk_status_label"] != "Fix":
        reasons.append(f"RTK not Fix: {position['rtk_status_label']}")
    if not position["valid_for_motion"]:
        reasons.append("position not valid for motion")
    if position["pos_type_label"] not in {"AREA_INSIDE", "TURN_AREA_INSIDE"}:
        reasons.append(
            f"position type {position['pos_type_label']} (drive it off the dock first)"
        )
    if blade["safety_blockers"] or blade["reported_state_label"] != "OFF":
        reasons.append(f"blade not safe: {blade['safety_blockers']}")
    if state.get("charge_state_label") != "not_charging":
        reasons.append(f"charge state {state.get('charge_state_label')}")
    return reasons


def set_gate(url: str, token: str, *, on: bool) -> None:
    """Arm or disarm the experimental-motion gate via the options flow helper."""
    repo = Path(__file__).resolve().parent.parent
    subprocess.run(
        [
            str(repo / ".venv/bin/python"),
            str(repo / "scripts/ha_set_experimental_motion.py"),
            "on" if on else "off",
            "--yes",
        ],
        check=False,
        capture_output=True,
    )


def pulse(
    url: str, token: str, action: str, duration_ms: int, *, dry_run: bool
) -> dict[str, Any]:
    """Dispatch one bounded velocity pulse and return its result."""
    return post_service(
        url,
        token,
        "mammotion",
        "manual_velocity_pulse_test",
        {
            "entity_id": ENTITY,
            "action": action,
            "speed": PULSE_SPEED,
            "duration_ms": duration_ms,
            "motion_refresh_interval_ms": REFRESH_MS,
            "stop_mode": "immediate",
            "dry_run": dry_run,
            "confirm_blades_off": not dry_run,
            "confirm_clear_area": not dry_run,
        },
        180,
    )


def pulse_ok(result: dict[str, Any], *, dry_run: bool) -> tuple[bool, str]:
    """Judge one pulse -- did it send, and did its stop confirm."""
    if dry_run:
        failing = [
            g["name"] for g in result.get("safety_gates", []) if not g.get("passed")
        ]
        return (
            not failing,
            f"dry-run gates failing: {failing}" if failing else "dry run ok",
        )
    if result.get("blockers"):
        return False, f"blockers: {result['blockers']}"
    if result.get("real_pulse_completed") is not True:
        return (
            False,
            f"pulse did not complete: command={result.get('command_result')} stop={result.get('stop_result')}",
        )
    return True, "ok"


def bearing_deg(start: tuple[float, float], end: tuple[float, float]) -> float:
    """Bearing of the displacement from ``start`` to ``end``, degrees 0-360."""
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 360.0


def angle_delta(from_deg: float, to_deg: float) -> float:
    """Signed smallest angle from one bearing to another, -180..180."""
    return (to_deg - from_deg + 180.0) % 360.0 - 180.0


def drive_leg(
    url: str, token: str, label: str, record: dict[str, Any], *, dry_run: bool
) -> dict[str, Any]:
    """Pulse forward until RTK reports LEG_TARGET_M, or an abort bound trips."""
    state = read_state(url, token)
    start = (state["position"]["x"], state["position"]["y"])
    leg: dict[str, Any] = {
        "label": label,
        "start": start,
        "started_utc": now_iso(),
        "pulses": [],
    }
    travelled = 0.0
    for index in range(MAX_PULSES_PER_LEG):
        result = pulse(url, token, "forward", PULSE_MS, dry_run=dry_run)
        ok, why = pulse_ok(result, dry_run=dry_run)
        state = read_state(url, token)
        here = (state["position"]["x"], state["position"]["y"])
        travelled = math.hypot(here[0] - start[0], here[1] - start[1])
        leg["pulses"].append(
            {
                "index": index + 1,
                "utc": now_iso(),
                "ok": ok,
                "why": why,
                "position": here,
                "travelled_m": round(travelled, 4),
                "rtk": state["position"]["rtk_status_label"],
            }
        )
        emit(
            f"  [{label}] pulse {index + 1}: {why}; travelled {travelled:.3f} m; rtk {state['position']['rtk_status_label']}"
        )
        if not ok:
            leg["abort"] = why
            break
        if state["position"]["rtk_status_label"] != "Fix":
            leg["abort"] = "RTK left Fix"
            break
        if travelled >= LEG_ABORT_M:
            leg["abort"] = f"overshot {travelled:.2f} m"
            break
        if travelled >= LEG_TARGET_M or dry_run:
            break
    leg["end"] = (state["position"]["x"], state["position"]["y"])
    leg["length_m"] = round(travelled, 4)
    leg["bearing_deg"] = (
        round(bearing_deg(start, leg["end"]), 3) if travelled > 0.05 else None
    )
    record["legs"].append(leg)
    return leg


def drive_turn(
    url: str,
    token: str,
    label: str,
    rate_deg_s: float,
    record: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Pulse left for the duration that ``rate_deg_s`` predicts gives 90 deg."""
    total_ms = int(
        min(
            TURN_TARGET_DEG / max(rate_deg_s, 1.0) * 1000,
            PULSE_MS * MAX_PULSES_PER_TURN,
        )
    )
    turn: dict[str, Any] = {
        "label": label,
        "started_utc": now_iso(),
        "assumed_rate_deg_s": round(rate_deg_s, 3),
        "commanded_ms": total_ms,
        "pulses": [],
    }
    remaining = total_ms
    while remaining > 0:
        this_ms = min(PULSE_MS, remaining)
        result = pulse(url, token, "turn_left", this_ms, dry_run=dry_run)
        ok, why = pulse_ok(result, dry_run=dry_run)
        turn["pulses"].append({"utc": now_iso(), "ms": this_ms, "ok": ok, "why": why})
        emit(f"  [{label}] turn pulse {this_ms} ms: {why}")
        if not ok:
            turn["abort"] = why
            break
        remaining -= this_ms
        if dry_run:
            break
    record["turns"].append(turn)
    return turn


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    """Run the square, always disarming the gate on the way out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    load_dotenv(Path(".env"))
    url, token = os.environ["HA_URL"].rstrip("/"), os.environ["HA_TOKEN"]

    state = read_state(url, token)
    reasons = preflight(state)
    if reasons and not args.dry_run:
        emit("PREFLIGHT FAILED, nothing armed:")
        for reason in reasons:
            emit(f"  - {reason}")
        return 2
    if reasons:
        emit(f"(dry run; preflight would have blocked: {reasons})")

    start = (state["position"]["x"], state["position"]["y"])
    record: dict[str, Any] = {
        "predeclaration": "docs/predeclared-rtk-square-return-20260916.md",
        "started_utc": now_iso(),
        "dry_run": args.dry_run,
        "start_position": start,
        "parameters": {
            "speed": PULSE_SPEED,
            "pulse_ms": PULSE_MS,
            "refresh_ms": REFRESH_MS,
            "leg_target_m": LEG_TARGET_M,
            "turn_target_deg": TURN_TARGET_DEG,
        },
        "legs": [],
        "turns": [],
    }
    emit(f"start {start}  rtk {state['position']['rtk_status_label']}")

    if not args.dry_run:
        set_gate(url, token, on=True)
        emit(f"ARMED {now_iso()}")
    try:
        rate = INITIAL_TURN_RATE_DEG_S
        for index in range(4):
            leg = drive_leg(url, token, f"leg{index + 1}", record, dry_run=args.dry_run)
            if leg.get("abort"):
                emit(f"ABORT on {leg['label']}: {leg['abort']}")
                break
            # Turn 1's true angle is only knowable once leg 2 exists; from then
            # on, re-derive the rate from what the previous turn actually did.
            if index >= 1 and len(record["turns"]) >= 1:
                previous, current = record["legs"][index - 1], record["legs"][index]
                if (
                    previous["bearing_deg"] is not None
                    and current["bearing_deg"] is not None
                ):
                    achieved = angle_delta(
                        previous["bearing_deg"], current["bearing_deg"]
                    )
                    turn = record["turns"][-1]
                    turn["achieved_deg"] = round(achieved, 3)
                    if abs(achieved) > 5.0 and turn["commanded_ms"]:
                        rate = abs(achieved) / (turn["commanded_ms"] / 1000)
                        turn["measured_rate_deg_s"] = round(rate, 3)
                    emit(
                        f"  turn {index} achieved {achieved:+.1f} deg -> rate {rate:.2f} deg/s"
                    )
            # 🚨 A turn after EVERY leg, including the last. The 2026-09-16 run
            # stopped after turn 3, so the mower closed on its start *position*
            # ~100 deg off its start *heading*: every body point displaced by a
            # different amount and the tape-vs-RTK comparison inherited the
            # unknown antenna offset. With heading restored, one tape
            # measurement equals RTK's closure directly. The predeclaration
            # (§0) always said four turns; only the runner disagreed.
            turn = drive_turn(
                url, token, f"turn{index + 1}", rate, record, dry_run=args.dry_run
            )
            if turn.get("abort"):
                emit(f"ABORT on {turn['label']}: {turn['abort']}")
                break
    finally:
        if not args.dry_run:
            set_gate(url, token, on=False)
            emit(f"DISARMED {now_iso()}")
        final = read_state(url, token)
        end = (final["position"]["x"], final["position"]["y"])
        record["end_position"] = end
        record["rtk_closure_m"] = round(
            math.hypot(end[0] - start[0], end[1] - start[1]), 4
        )
        record["ended_utc"] = now_iso()
        record["final_rtk"] = final["position"]["rtk_status_label"]
        path = (
            args.out
            / f"rtk_square_{'dry' if args.dry_run else 'real'}_{int(time.time())}.json"
        )
        path.write_text(json.dumps(record, indent=1))
        emit("=" * 68)
        emit(f"start {start}   end {end}")
        emit(
            f"RTK-claimed closure: {record['rtk_closure_m']:.3f} m   (final rtk {record['final_rtk']})"
        )
        for leg in record["legs"]:
            emit(
                f"  {leg['label']}: {leg['length_m']:.3f} m  bearing {leg['bearing_deg']}  pulses {len(leg['pulses'])}"
            )
        for turn in record["turns"]:
            emit(
                f"  {turn['label']}: commanded {turn['commanded_ms']} ms  achieved {turn.get('achieved_deg')}"
            )
        emit(f"record: {path}")
        emit(
            "NOW TAPE-MEASURE the real distance from the start mark -- that is the headline number."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
