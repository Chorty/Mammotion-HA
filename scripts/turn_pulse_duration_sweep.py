#!/usr/bin/env python3
"""Measure rotation against turn-pulse duration, and settle the rate model.

    scripts/turn_pulse_duration_sweep.py            # preflight + plan, no motion
    scripts/turn_pulse_duration_sweep.py --arm      # the real sweep

`_MIN_SCALED_TURN_PULSE_MS`'s docstring has asked for this run since it was
written and it has never happened: "run ``vio_turn_to_heading`` at refresh 200 /
angular 500 with ``pulse_duration_ms`` stepped down 700 -> 500 -> 400 -> 300 and
find where measured rotation stops tracking duration."

A three-call method validation on 2026-08-09
(docs/evidence-turn-pulse-duration-smoketest-20260809.json) made it urgent. It
found that apparent rate FALLS as the window grows -- 50.37 deg/s at 253 ms,
39.83 at 401 ms, 30.42 at 1240 ms -- fitting

    rotation = 25.51 deg/s * window + 6.04 deg      residuals 0.25 / -0.29 / 0.04

i.e. a constant ~6 deg per pulse that does not depend on duration, plus a real
~25.5 deg/s while driven. If that holds it explains the entire "rotation rate
variance" this project has chased for weeks: every figure on record is
rotation/window, so short windows read fast and long ones read slow, and the
9.23 and 69.41 deg/s extremes are the same mower measured over different
windows. It would also mean no pulse can land closer than ~6 deg however short,
which is a floor on turn PRECISION and a different thing from the actuation
floor `_MIN_SCALED_TURN_PULSE_MS` was written to guard.

Three points cannot carry that. This sweep is built to confirm or kill it.

Method notes that are load-bearing:

* ``turn_degrees_per_second: 1.0`` makes the estimator's ``needed_ms`` enormous,
  so pulse 1 takes the ``cruising_full_pulse_fits`` branch -- the one path that
  applies NO ``_MIN_SCALED_TURN_PULSE_MS`` floor. Without it a 300 ms request is
  silently rewritten to 400 ms and the sweep measures nothing. Verified live.
* The heading error must be large enough that the overshoot ceiling also permits
  a full-length pulse: ``|error| >= 60 * D/1000 - tolerance``.
* Fit against the DELIVERED window, never the commanded duration. They differ:
  ``max_refreshes = int(duration / interval)`` truncated a 300 ms command to a
  253 ms window, and BLE latency stretched a 700 ms command to 1240 ms.
* Samples flagged ``refresh_cadence_broken`` (beta34) are excluded from the fit:
  a pulse whose refresh cadence collapsed did not rotate for the window it was
  billed for.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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

TOLERANCE_DEGREES = 18.0
#: Commanded durations. Weighted toward the short end, which is where the
#: constant-offset hypothesis and the real actuation floor both live.
DURATIONS_MS = (200, 250, 300, 400, 500, 700, 1000, 1500)
REPEATS = 1
#: Each direction is swept as a CONTIGUOUS BLOCK, not alternated.
#:
#: The first sweep alternated the sign every call, which confounds turn direction
#: with call parity: whichever sign leads is also always the first of its pair.
#: It measured one direction rotating further in 8 pairs of 8, and could not say
#: whether that was direction or order. A second run with the opposite lead
#: settled it -- the same direction won from the second position, so the effect
#: is DIRECTIONAL, 1.44x at matched 200 ms windows
#: (docs/evidence-turn-direction-vs-order-20260809T234653Z.json).
#:
#: Blocks remove the confound entirely: every duration is measured at one sign
#: before the sign changes, so a per-direction fit is clean and the additive term
#: can be separated from the slope instead of being an artefact of interleaving.
SIGN_BLOCKS = (1, -1)
#: Generous, so the feasibility preflight never refuses. The turn stops on
#: tolerance long before this, so it does not change what is measured.
MAX_COMMANDS = 8
MIN_BATTERY_PERCENT = 30
MIN_TRACKED_FEATURES = 70


def _heading_error_for(duration_ms: int) -> float:
    """Smallest error that still lets the ceiling permit a full-length pulse."""
    required = 60.0 * duration_ms / 1000.0 - TOLERANCE_DEGREES
    # Sit clear of both the ceiling bound and the tolerance, which the turn's
    # entry check would otherwise satisfy immediately.
    return max(required + 6.0, TOLERANCE_DEGREES + 4.0)


def _state(entity: str) -> str | None:
    req = urllib.request.Request(
        f"{os.environ['HA_URL'].rstrip('/')}/api/states/{entity}",
        headers={"Authorization": f"Bearer {os.environ['HA_TOKEN']}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response).get("state")
    except Exception:  # noqa: BLE001
        return None


def _call(service: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    return post_service(
        os.environ["HA_URL"],
        os.environ["HA_TOKEN"],
        "mammotion",
        service,
        {"entity_id": ENTITY, **payload},
        timeout,
    )


def preflight() -> list[str]:
    """Check the same hard gates the segment runs use, minus path gates."""
    # Wake the link before judging it. `ble_link_live` needs a recent
    # outbound send and fails `ble_send_stalled` after 15 s of quiet, so a
    # preflight on a rested link reports the staleness of its own idleness.
    # Read-only; sends no movement command. Live 2026-08-09 this turned a
    # spurious FAIL on a healthy -62 dBm link into a PASS in 3 seconds.
    warm_ble_link(os.environ["HA_URL"], os.environ["HA_TOKEN"], ENTITY)
    runtime = _call("export_runtime_state", {}, 120)
    safety = runtime.get("safety", {})
    tracked = _state("sensor.back_yard_clip_skywalker_vio_tracked_features")
    battery = _state("sensor.back_yard_clip_skywalker_battery")
    tracked_n = float(tracked) if tracked not in (None, "unknown", "unavailable") else 0
    battery_n = float(battery) if battery not in (None, "unknown", "unavailable") else 0
    checks = [
        ("daylight / VIO feed", tracked_n >= MIN_TRACKED_FEATURES, f"{tracked_n:.0f}"),
        ("RTK precise", safety.get("rtk_status_label") == "Fix", "Fix required"),
        ("blade safe", bool(safety.get("blade_safe_for_motion")), ""),
        (
            "BLE link live",
            _state("binary_sensor.back_yard_clip_skywalker_ble_link_live") == "on",
            f"rssi={_state('sensor.back_yard_clip_skywalker_ble_rssi')}",
        ),
        (
            "work mode ready",
            runtime.get("work_mode_label") in {"MODE_READY", "MODE_PAUSE"},
            str(runtime.get("work_mode_label")),
        ),
        ("battery", battery_n >= MIN_BATTERY_PERCENT, f"{battery_n:.0f}%"),
    ]
    print("\n== PREFLIGHT ==")
    failed = [label for label, ok, _ in checks if not ok]
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:22s} {detail}")
    # The mower turns in place here, so it needs room but not a mapped path.
    print(f"  position: {runtime.get('position', {}).get('pos_type_label')}")
    return failed


def _fit(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Least-squares rotation = slope * window + intercept."""
    if len(points) < 2:
        return None
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    if sxx == 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sxy / sxx
    return slope, mean_y - slope * mean_x


def _report_group(label: str, samples: list[dict[str, Any]]) -> None:
    """Fit one direction and say which model its own data supports."""
    usable = [s for s in samples if s.get("window_ms") and not s.get("cadence_broken")]
    excluded = len(samples) - len(usable)
    print(f"\n  --- {label} ---")
    print(f"  {'cmd ms':>7} {'window ms':>10} {'rotation':>9} {'apparent':>9}  flags")
    for s in sorted(samples, key=lambda s: s.get("window_ms") or 0):
        window, rotation = s.get("window_ms"), s.get("rotation_degrees")
        rate = (
            f"{rotation / (window / 1000):8.2f}" if window and rotation else "     n/a"
        )
        print(
            f"  {s['commanded_ms']:7d} {window or 0:10.1f} {rotation or 0:9.3f} "
            f"{rate}  {'CADENCE-BROKEN' if s.get('cadence_broken') else ''}"
        )
    if excluded:
        print(f"  ({excluded} excluded as cadence-broken)")
    points = [(s["window_ms"] / 1000.0, s["rotation_degrees"]) for s in usable]
    fit = _fit(points)
    if fit is None:
        print("  too few usable samples to fit")
        return
    slope, intercept = fit
    through_origin = sum(y for _, y in points) / sum(x for x, _ in points)
    resid_prop = sum((y - through_origin * x) ** 2 for x, y in points)
    resid_off = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    print(
        f"  proportional : {through_origin:6.2f} deg/s * window"
        f"                 residual {resid_prop:8.3f}"
    )
    print(
        f"  offset       : {slope:6.2f} deg/s * window {intercept:+6.2f} deg"
        f"   residual {resid_off:8.3f}"
    )
    # A two-point "fit" has zero residual by construction, and the whole reason
    # this sweep exists is that three points could not carry a conclusion.
    if len(points) < 5:
        print(f"  >>> NO VERDICT for {label}: {len(points)} usable sample(s), need 5")
        return
    if resid_off < resid_prop * 0.5:
        print(
            f"  >>> {label}: OFFSET model wins -- a constant "
            f"{intercept:+.2f} deg per pulse"
        )
    else:
        print(f"  >>> {label}: proportional model is adequate")


def analyse(samples: list[dict[str, Any]]) -> None:
    """Fit each turn direction separately, then compare them.

    Pooling the two directions is what made the first sweep unreadable: the
    asymmetry is directional and 1.44x at matched windows, so a pooled fit
    averages two different behaviours and its residuals hide both.
    """
    print("\n== RESULT ==")
    groups = {
        sign: [s for s in samples if s.get("sign") == sign]
        for sign in ("+", "-")
        if any(s.get("sign") == sign for s in samples)
    }
    if not groups:
        _report_group("all samples", samples)
        return
    for sign, group in groups.items():
        _report_group(f"sign {sign}", group)

    print("\n  --- direction comparison, at matched commanded durations ---")
    by_duration: dict[int, dict[str, float]] = {}
    for s in samples:
        if s.get("rotation_degrees") and not s.get("cadence_broken"):
            by_duration.setdefault(s["commanded_ms"], {})[s.get("sign", "?")] = s[
                "rotation_degrees"
            ]
    ratios = []
    for duration in sorted(by_duration):
        pair = by_duration[duration]
        if "+" in pair and "-" in pair and pair["-"]:
            ratio = pair["+"] / pair["-"]
            ratios.append(ratio)
            print(
                f"  {duration:5d} ms  + {pair['+']:7.3f}  - {pair['-']:7.3f}  "
                f"ratio {ratio:5.2f}"
            )
    if ratios:
        print(
            f"\n  '+' rotated further in {sum(1 for r in ratios if r > 1)}"
            f"/{len(ratios)} matched pairs, ratio "
            f"{min(ratios):.2f}-{max(ratios):.2f}"
        )


def main() -> int:  # noqa: C901
    """Preflight, print the plan, and -- only with ``--arm`` -- run the sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="store_true", help="actually run the sweep")
    args = parser.parse_args()
    load_dotenv()
    for required in ("HA_URL", "HA_TOKEN"):
        if not os.environ.get(required):
            raise SystemExit(f"{required} missing -- `set -a && source .env && set +a`")

    failed = preflight()
    plan = [
        (d, sign, i)
        for sign in SIGN_BLOCKS
        for d in DURATIONS_MS
        for i in range(REPEATS)
    ]
    print(f"\n== PLAN ==  {len(plan)} calls, in-place turns, blades off")
    print(f"  {len(DURATIONS_MS) * REPEATS} calls at sign +, then the same at sign -")
    for duration_ms in DURATIONS_MS:
        print(
            f"  {duration_ms:5d} ms  x{REPEATS}  heading error "
            f"+/-{_heading_error_for(duration_ms):.0f} deg"
        )
    if failed:
        print(f"\nPREFLIGHT FAILED: {', '.join(failed)}")
        if args.arm:
            print("refusing to arm.")
            return 1
    if not args.arm:
        print("\nPreview only. Re-run with --arm to execute (the safe exit).")
        return 0

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "docs" / f"evidence-turn-pulse-duration-sweep-{stamp}.json"
    samples: list[dict[str, Any]] = []
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
        if (
            not _call("export_runtime_state", {}, 120)
            .get("experimental_motion", {})
            .get("real_motion_allowed")
        ):
            print("  gate did not open -- sending nothing")
            return 1
        armed = True

        for index, (duration_ms, sign, repeat) in enumerate(plan):
            heading = _state("sensor.back_yard_clip_skywalker_vio_heading")
            if heading is None:
                print("  vio_heading unreadable -- stopping early")
                break
            # Fixed sign within a block. The mower therefore walks around its
            # own heading over a block instead of oscillating, which is the
            # point: interleaving is what made the first sweep unreadable.
            error = _heading_error_for(duration_ms) * sign
            response = _call(
                "vio_turn_to_heading",
                {
                    "target_vision_heading": (float(heading) + error) % 360,
                    "heading_tolerance_degrees": TOLERANCE_DEGREES,
                    "angular_speed": 500,
                    "pulse_duration_ms": duration_ms,
                    "slow_threshold_degrees": 1.0,
                    "max_commands": MAX_COMMANDS,
                    "motion_refresh_interval_ms": 200,
                    "turn_degrees_per_second": 1.0,
                    "max_displacement_m": 0.5,
                    "dry_run": False,
                    "confirm_blades_off": True,
                    "confirm_clear_area": True,
                },
                240,
            )
            command = (response.get("command_results") or [{}])[0]
            refresh = command.get("motion_refresh") or {}
            sample = {
                "commanded_ms": duration_ms,
                "repeat": repeat,
                "sign": "+" if sign > 0 else "-",
                "block_position": index,
                "heading_error_degrees": error,
                "delivered_pulse_ms": command.get("pulse_duration_ms"),
                "final_approach_reason": (command.get("final_approach") or {}).get(
                    "reason"
                ),
                "window_ms": refresh.get("elapsed_ms"),
                "rotation_degrees": abs(command.get("measured_change_degrees") or 0.0)
                or None,
                "cadence_broken": command.get("refresh_cadence_broken"),
                "longest_write_ms": command.get("longest_refresh_write_ms"),
                "stop_reason": response.get("stop_reason"),
                "response": response,
            }
            samples.append(sample)
            out.write_text(json.dumps(samples, indent=1))  # save before parsing
            print(
                f"  [{index + 1:2d}/{len(plan)}] sign {'+' if sign > 0 else '-'} "
                f"{duration_ms:5d} ms -> "
                f"window {sample['window_ms']} ms, "
                f"rotation {sample['rotation_degrees']} deg"
                + ("  CADENCE-BROKEN" if sample["cadence_broken"] else "")
            )
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
            motion = _call("export_runtime_state", {}, 120).get(
                "experimental_motion", {}
            )
            print(
                f"  enabled={motion.get('enabled')} "
                f"real_motion_allowed={motion.get('real_motion_allowed')}"
            )
            if motion.get("real_motion_allowed"):
                print("  !! GATE STILL OPEN -- disarm by hand immediately.")
        if samples:
            print(f"\n  COMPLETE RESPONSES SAVED -> {out.relative_to(REPO)}")
            analyse(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
