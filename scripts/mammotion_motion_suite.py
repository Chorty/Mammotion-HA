#!/usr/bin/env python3
"""Run the consolidated Mammotion motion test suite through Home Assistant."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from mammotion_ha_helpers import (
    call_readiness_with_runtime_retry,
    load_dotenv,
    wait_for_runtime,
)

DEFAULT_SAMPLE_DELAYS = [0, 5, 10, 20, 30, 45, 60]
QUICK_SAMPLE_DELAYS = [0, 4, 8, 15, 25]


def _write_json(path: Path, payload: dict) -> None:
    """Write formatted JSON."""
    path.write_text(json.dumps(payload, indent=2))


def _vector_payload(args: argparse.Namespace) -> dict:
    """Build raw_vector_readiness_test payload."""
    return {
        "entity_id": args.entity_id,
        "dry_run": args.dry_run,
        "confirm_blades_off": args.confirm_blades_off,
        "confirm_clear_area": args.confirm_clear_area,
        "prefer_ble": not args.use_wifi,
        "max_real_steps": args.max_real_steps,
        "target_distance": args.target_distance,
        "turn_delta_degrees": args.turn_delta_degrees,
        "calibrated_forward_heading_offset_degrees": (
            args.calibrated_forward_heading_offset_degrees
        ),
        "max_turn_commands": args.max_turn_commands,
        "max_linear_commands": args.max_linear_commands,
        "sample_delays": args.sample_delays,
    }


def _multi_segment_points(runtime_state: dict, args: argparse.Namespace) -> list[dict]:
    """Generate a small two-segment path from live map-local mower telemetry."""
    position = runtime_state.get("position") or {}
    x = float(position["x"])
    y = float(position["y"])
    reported_heading = float(position["toward"])
    first_heading = (
        reported_heading + args.calibrated_forward_heading_offset_degrees
    ) % 360
    second_heading = (first_heading + args.turn_delta_degrees) % 360
    distance = float(args.target_distance)

    def advance(point_x: float, point_y: float, heading: float) -> dict:
        return {
            "x": round(point_x + math.cos(math.radians(heading)) * distance, 3),
            "y": round(point_y + math.sin(math.radians(heading)) * distance, 3),
        }

    start = {"x": round(x, 3), "y": round(y, 3)}
    first = advance(x, y, first_heading)
    second = advance(first["x"], first["y"], second_heading)
    return [start, first, second]


def _multi_segment_payload(
    args: argparse.Namespace,
    runtime_state: dict,
    *,
    dry_run: bool,
) -> dict:
    """Build raw_pymammotion_execute_multi_segment payload."""
    return {
        "entity_id": args.entity_id,
        "points": _multi_segment_points(runtime_state, args),
        "dry_run": dry_run,
        "confirm_blades_off": args.confirm_blades_off,
        "confirm_clear_area": args.confirm_clear_area,
        "prefer_ble": not args.use_wifi,
        "max_real_segments": args.max_real_segments,
        "linear_speed_fast": 400,
        "linear_speed_slow": 200,
        "max_turn_commands": args.max_turn_commands,
        "max_linear_commands": args.max_linear_commands,
        "calibrated_forward_heading_offset_degrees": (
            args.calibrated_forward_heading_offset_degrees
        ),
        "sample_delays": args.sample_delays,
    }


def _summary(
    result: dict,
    output_dir: Path,
    *,
    multi_dry_run: dict | None = None,
    multi_real: dict | None = None,
) -> dict:
    """Return compact suite summary."""
    vector_passed = (
        bool(result)
        and result.get("ready_for_multi_segment") is True
        and result.get("failed_phase") is None
    )
    multi_dry_run_passed = (
        None
        if multi_dry_run is None
        else (
            bool(multi_dry_run)
            and multi_dry_run.get("stop_reason") == "dry_run"
            and multi_dry_run.get("ready_for_multi_segment") is True
        )
    )
    multi_real_passed = (
        None
        if multi_real is None
        else (
            bool(multi_real)
            and multi_real.get("stop_reason")
            in {"target_reached", "max_real_segments_reached"}
            and multi_real.get("failed_segment_index") is None
            and int(multi_real.get("real_segments_executed") or 0) > 0
        )
    )
    passed = vector_passed
    if multi_dry_run_passed is not None:
        passed = passed and multi_dry_run_passed
    if multi_real_passed is not None:
        passed = passed and multi_real_passed
    return {
        "output_dir": str(output_dir),
        "suite": "motion_readiness",
        "integration_not_ready": not bool(result),
        "dry_run": result.get("dry_run"),
        "ready_for_multi_segment": result.get("ready_for_multi_segment"),
        "ready_for_multi_point": result.get("ready_for_multi_point"),
        "aligned_vector_ready": result.get("aligned_vector_ready"),
        "positive_turn_vector_ready": result.get("positive_turn_vector_ready"),
        "negative_turn_vector_ready": result.get("negative_turn_vector_ready"),
        "real_steps_run": result.get("real_steps_run"),
        "failed_phase": result.get("failed_phase"),
        "blockers": result.get("blockers"),
        "multi_segment_dry_run_included": multi_dry_run is not None,
        "multi_segment_dry_run_passed": multi_dry_run_passed,
        "multi_segment_real_included": multi_real is not None,
        "multi_segment_real_passed": multi_real_passed,
        "multi_segment_real_segments_executed": (
            None
            if multi_real is None
            else multi_real.get("real_segments_executed")
        ),
        "multi_segment_failed_segment_index": (
            None
            if multi_real is None
            else multi_real.get("failed_segment_index")
        ),
        "recommended_next_step": result.get("recommended_next_step"),
        "passed": passed,
    }


def _apply_quick_profile(args: argparse.Namespace) -> None:
    """Apply fast-turnaround defaults unless the user already overrode them."""
    if args.sample_delays == DEFAULT_SAMPLE_DELAYS:
        args.sample_delays = list(QUICK_SAMPLE_DELAYS)
    if args.runtime_timeout == 180:
        args.runtime_timeout = 90
    if args.timeout == 720:
        args.timeout = 240


def main() -> int:  # noqa: C901
    """Run the consolidated motion suite."""
    load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity_id", help="Mammotion mower entity_id")
    parser.add_argument("--ha-url", default=os.environ.get("HA_URL"))
    parser.add_argument("--ha-token", default=os.environ.get("HA_TOKEN"))
    parser.add_argument(
        "--output-dir",
        default="/tmp/mammotion_motion_suite",  # noqa: S108
    )
    parser.add_argument(
        "--sample-delays",
        type=float,
        nargs="+",
        default=list(DEFAULT_SAMPLE_DELAYS),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use shorter delays and timeouts for faster troubleshooting runs.",
    )
    parser.add_argument("--use-wifi", action="store_true", help="Use non-BLE transport preference.")
    parser.add_argument("--max-real-steps", type=int, default=0)
    parser.add_argument("--target-distance", type=float, default=0.10)
    parser.add_argument("--turn-delta-degrees", type=float, default=10.0)
    parser.add_argument("--calibrated-forward-heading-offset-degrees", type=float, default=116.5)
    parser.add_argument("--max-turn-commands", type=int, default=4)
    parser.add_argument("--max-linear-commands", type=int, default=2)
    parser.add_argument("--include-multi-segment-dry-run", action="store_true")
    parser.add_argument("--include-multi-segment-real", action="store_true")
    parser.add_argument("--max-real-segments", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--real", dest="dry_run", action="store_false")
    parser.add_argument("--confirm-blades-off", action="store_true")
    parser.add_argument("--confirm-clear-area", action="store_true")
    parser.add_argument("--no-wait-runtime", action="store_true")
    parser.add_argument("--runtime-timeout", type=int, default=180)
    parser.add_argument("--timeout", type=int, default=720)
    args = parser.parse_args()

    if args.quick:
        _apply_quick_profile(args)

    if not args.ha_url or not args.ha_token:
        parser.error("Provide --ha-url/--ha-token or set HA_URL/HA_TOKEN in .env or environment")
    if args.max_real_steps < 0 or args.max_real_steps > 3:
        parser.error("--max-real-steps must be between 0 and 3")
    if args.max_real_segments < 1 or args.max_real_segments > 3:
        parser.error("--max-real-segments must be between 1 and 3")
    if not args.dry_run and args.max_real_steps > 0 and (
        not args.confirm_blades_off or not args.confirm_clear_area
    ):
        parser.error("Real suite steps require --confirm-blades-off and --confirm-clear-area")
    if args.include_multi_segment_real and (
        not args.confirm_blades_off or not args.confirm_clear_area
    ):
        parser.error("Real multi-segment steps require --confirm-blades-off and --confirm-clear-area")
    if args.include_multi_segment_real and args.dry_run:
        parser.error("Real multi-segment steps require --real")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _vector_payload(args)
    result = call_readiness_with_runtime_retry(
        args.ha_url,
        args.ha_token,
        "raw_vector_readiness_test",
        payload,
        timeout=args.timeout,
        wait_runtime=not args.no_wait_runtime,
        runtime_timeout=args.runtime_timeout,
    )
    _write_json(output_dir / "raw_vector_readiness.json", {"payload": payload, "result": result})
    multi_dry_run: dict | None = None
    multi_real: dict | None = None
    runtime_state: dict | None = None
    if args.include_multi_segment_dry_run or args.include_multi_segment_real:
        runtime_state = wait_for_runtime(
            args.ha_url,
            args.ha_token,
            args.entity_id,
            timeout=args.runtime_timeout,
        )

    if args.include_multi_segment_dry_run and runtime_state is not None:
        multi_payload = _multi_segment_payload(args, runtime_state, dry_run=True)
        multi_dry_run = call_readiness_with_runtime_retry(
            args.ha_url,
            args.ha_token,
            "raw_pymammotion_execute_multi_segment",
            multi_payload,
            timeout=args.timeout,
            wait_runtime=not args.no_wait_runtime,
            runtime_timeout=args.runtime_timeout,
        )
        _write_json(
            output_dir / "multi_segment_dry_run.json",
            {"payload": multi_payload, "result": multi_dry_run},
        )

    if args.include_multi_segment_real and runtime_state is not None:
        multi_payload = _multi_segment_payload(args, runtime_state, dry_run=False)
        multi_real = call_readiness_with_runtime_retry(
            args.ha_url,
            args.ha_token,
            "raw_pymammotion_execute_multi_segment",
            multi_payload,
            timeout=args.timeout,
            wait_runtime=not args.no_wait_runtime,
            runtime_timeout=args.runtime_timeout,
        )
        _write_json(
            output_dir / "multi_segment_real.json",
            {"payload": multi_payload, "result": multi_real},
        )

    summary = _summary(
        result,
        output_dir,
        multi_dry_run=multi_dry_run,
        multi_real=multi_real,
    )
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))  # noqa: T201
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
