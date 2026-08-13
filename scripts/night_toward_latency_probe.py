#!/usr/bin/env python3
"""Capture live ``toward`` through exactly one bounded angular pulse.

Preview is the default. ``--arm`` performs one blades-off, in-place pulse after
the operator has separately authorized it. No linear or reverse command exists
in this harness. The complete service response and every concurrent read-only
runtime sample are saved before any result is summarized.
"""  # noqa: INP001

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mammotion_ha_helpers import load_dotenv, post_service  # noqa: E402

ENTITY = "lawn_mower.back_yard_clip_skywalker"
REPO = Path(__file__).resolve().parents[1]
TURN_DELTA_DEGREES = 60.0
PULSE_DURATION_MS = 1500.0
REFRESH_INTERVAL_MS = 200
ANGULAR_SPEED = 500
POLL_INTERVAL_SECONDS = 0.10


def _call(service: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    return post_service(
        os.environ["HA_URL"],
        os.environ["HA_TOKEN"],
        "mammotion",
        service,
        {"entity_id": ENTITY, **payload},
        timeout,
    )


def _runtime() -> dict[str, Any]:
    return _call("export_runtime_state", {}, 30)


def _preflight(runtime: dict[str, Any]) -> list[str]:
    motion = runtime.get("experimental_motion") or {}
    position = runtime.get("position") or {}
    blade = runtime.get("blade") or {}
    blockers = list(motion.get("blockers") or [])
    checks = [
        ("backend verified", bool(motion.get("backend_verified"))),
        ("BLE active", runtime.get("active_transport") == "ble"),
        ("RTK Fix", position.get("rtk_status_label") == "Fix"),
        ("mapped position", bool(position.get("valid_for_motion"))),
        ("live toward", position.get("toward") is not None),
        ("mower ready", runtime.get("work_mode_label") in {"MODE_READY", "MODE_PAUSE"}),
        ("not charging", runtime.get("charge_state_label") == "not_charging"),
        ("blades zero", blade.get("current_cutter_rpm") in {0, "0", None}),
        ("no active session", not motion.get("active_session")),
        ("only disabled-gate blocker", blockers == ["experimental_motion_disabled"]),
    ]
    print("\n== PREFLIGHT ==")
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    return [label for label, passed in checks if not passed]


def _payload(target: float, *, dry_run: bool) -> dict[str, Any]:
    return {
        "target_heading_degrees": target,
        "heading_tolerance_degrees": 0.5,
        "angular_speed_fast": ANGULAR_SPEED,
        "angular_speed_slow": ANGULAR_SPEED,
        "slow_turn_threshold_degrees": 1.0,
        "motion_refresh_interval_ms": REFRESH_INTERVAL_MS,
        "max_commands": 1,
        "min_heading_change_degrees": 0.5,
        "max_translation_distance": 0.30,
        "pulse_duration_ms": PULSE_DURATION_MS,
        "prefer_ble": True,
        # These are cumulative post-stop delays. Concurrent polling below is
        # what records the pulse itself; this keeps the service open through the
        # required three-second post-stop window.
        "sample_delays": [0.0, 0.25, 0.5, 1.0, 2.0, 3.0],
        "dry_run": dry_run,
        "confirm_blades_off": not dry_run,
        "confirm_clear_area": not dry_run,
    }


def _capture(
    samples: list[dict[str, Any]],
    stop: threading.Event,
    origin: float,
) -> None:
    """Poll the cached runtime snapshot until the service's post-stop window ends."""
    while not stop.is_set():
        began = time.monotonic()
        try:
            runtime = _runtime()
            position = runtime.get("position") or {}
            motion = runtime.get("experimental_motion") or {}
            session = motion.get("active_session") or {}
            samples.append(
                {
                    "elapsed_seconds": round(time.monotonic() - origin, 6),
                    "utc": datetime.now(UTC).isoformat(),
                    "request_seconds": round(time.monotonic() - began, 6),
                    "x": position.get("x"),
                    "y": position.get("y"),
                    "toward": position.get("toward"),
                    "rtk_status_label": position.get("rtk_status_label"),
                    "work_mode_label": runtime.get("work_mode_label"),
                    "active_transport": runtime.get("active_transport"),
                    "session_id": session.get("session_id"),
                    "session_phase": session.get("phase"),
                }
            )
        except BaseException as err:  # noqa: BLE001
            samples.append(
                {
                    "elapsed_seconds": round(time.monotonic() - origin, 6),
                    "utc": datetime.now(UTC).isoformat(),
                    "capture_error": f"{type(err).__name__}: {err}",
                }
            )
        stop.wait(max(0.0, POLL_INTERVAL_SECONDS - (time.monotonic() - began)))


def main() -> int:  # noqa: C901
    """Preview or execute the single authorized item-16 pulse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    for key in ("HA_URL", "HA_TOKEN"):
        if not os.environ.get(key):
            raise SystemExit(f"{key} missing")

    runtime = _runtime()
    failed = _preflight(runtime)
    toward = float((runtime.get("position") or {})["toward"])
    target = (toward + TURN_DELTA_DEGREES) % 360.0
    payload = _payload(target, dry_run=True)
    print(
        f"\n== PLAN == one angular-only pulse: toward {toward:.4f} -> target "
        f"{target:.4f} (+{TURN_DELTA_DEGREES:.1f} deg), angular +{ANGULAR_SPEED}, "
        f"{PULSE_DURATION_MS:.0f} ms, refresh {REFRESH_INTERVAL_MS} ms"
    )
    dry = _call("raw_pymammotion_turn_to_heading", payload)
    print(
        f"  dry-run stop={dry.get('stop_reason')} would_send={dry.get('would_send')} "
        f"blockers={dry.get('blockers')}"
    )
    selected_angular = (dry.get("initial_command_selection") or {}).get("angular_speed")
    if (
        failed
        or dry.get("stop_reason") != "dry_run"
        or dry.get("blockers")
        or selected_angular != ANGULAR_SPEED
    ):
        print(f"REFUSING: preflight={failed}, dry_run={dry.get('stop_reason')}")
        return 1
    if not args.arm:
        print("Preview only; no motion command sent.")
        return 0

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "docs" / f"evidence-night-toward-latency-{stamp}.json"
    samples: list[dict[str, Any]] = []
    response: dict[str, Any] | None = None
    error: str | None = None
    origin = time.monotonic()
    stop = threading.Event()
    capture = threading.Thread(
        target=_capture,
        args=(samples, stop, origin),
        name="toward-runtime-capture",
        daemon=True,
    )
    # Set before enable: calling enable is what creates the disarm obligation.
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
        if not (_runtime().get("experimental_motion") or {}).get("real_motion_allowed"):
            raise RuntimeError("gate did not reach real_motion_allowed")
        capture.start()
        time.sleep(0.5)
        print("\n== ONE PULSE + CAPTURE ==")
        response = _call(
            "raw_pymammotion_turn_to_heading",
            _payload(target, dry_run=False),
            180,
        )
    except BaseException as err:  # noqa: BLE001
        error = f"{type(err).__name__}: {err}"
        raise
    finally:
        stop.set()
        if capture.is_alive():
            capture.join(timeout=35)
        evidence = {
            "test": "night_toward_latency_item_16",
            "parameters": {
                "initial_toward": toward,
                "target_toward": target,
                "turn_delta_degrees": TURN_DELTA_DEGREES,
                "angular_speed": ANGULAR_SPEED,
                "pulse_duration_ms": PULSE_DURATION_MS,
                "motion_refresh_interval_ms": REFRESH_INTERVAL_MS,
                "poll_interval_seconds": POLL_INTERVAL_SECONDS,
            },
            "service_response": response,
            "capture_samples": samples,
            "error": error,
        }
        out.write_text(json.dumps(evidence, indent=1) + "\n")
        print(f"  COMPLETE RESPONSE AND CAPTURE SAVED -> {out.relative_to(REPO)}")
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
            final_motion = _runtime().get("experimental_motion") or {}
            print(
                f"  enabled={final_motion.get('enabled')} "
                f"real_motion_allowed={final_motion.get('real_motion_allowed')} "
                f"active_session={final_motion.get('active_session')}"
            )
    if response is None:
        return 1
    print(
        f"\nRESULT: stop={response.get('stop_reason')} "
        f"commands={response.get('commands_sent')} samples={len(samples)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
