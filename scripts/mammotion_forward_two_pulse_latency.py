#!/usr/bin/env python3
"""Run the Mammotion two-forward-pulse telemetry latency test through HA."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from mammotion_ha_helpers import load_dotenv, post_service, wait_for_runtime


def _apply_quick_profile(args: argparse.Namespace) -> None:
    """Apply fast-turnaround defaults unless user provided custom values."""
    if args.pulse_gap_seconds == 5.0:
        args.pulse_gap_seconds = 3.0
    if args.telemetry_timeout_seconds == 60.0:
        args.telemetry_timeout_seconds = 35.0
    if args.telemetry_sample_interval_seconds == 1.0:
        args.telemetry_sample_interval_seconds = 0.5
    if args.runtime_timeout == 180:
        args.runtime_timeout = 90
    if args.timeout == 180:
        args.timeout = 120


def _write_json(path: Path, payload: dict) -> None:
    """Write formatted JSON."""
    path.write_text(json.dumps(payload, indent=2))


def _payload(args: argparse.Namespace) -> dict:
    """Build forward_two_pulse_latency_test payload."""
    return {
        "entity_id": args.entity_id,
        "linear_speed": args.linear_speed,
        "pulse_count": args.pulse_count,
        "pulse_gap_seconds": args.pulse_gap_seconds,
        "telemetry_timeout_seconds": args.telemetry_timeout_seconds,
        "telemetry_sample_interval_seconds": args.telemetry_sample_interval_seconds,
        "min_position_change_distance": args.min_position_change_distance,
        "prefer_ble": not args.use_wifi,
        "dry_run": args.dry_run,
        "confirm_blades_off": args.confirm_blades_off,
        "confirm_clear_area": args.confirm_clear_area,
    }


def _summary(result: dict, output_dir: Path) -> dict:
    """Return compact latency summary."""
    telemetry = result.get("telemetry") or {}
    commands = result.get("commands") or []
    return {
        "output_dir": str(output_dir),
        "dry_run": result.get("dry_run"),
        "reason": result.get("reason"),
        "passed": result.get("reason")
        in {"dry_run", "telemetry_position_change_detected"},
        "commands_sent": len([command for command in commands if command.get("ok")]),
        "pulse_count": result.get("pulse_count"),
        "first_position_change_at": telemetry.get("first_position_change_at"),
        "first_position_change_after_command_1_seconds": telemetry.get(
            "first_position_change_after_command_1_seconds"
        ),
        "first_position_change_after_command_2_seconds": telemetry.get(
            "first_position_change_after_command_2_seconds"
        ),
        "first_position_change_after_final_command_seconds": telemetry.get(
            "first_position_change_after_final_command_seconds"
        ),
        "final_delta": telemetry.get("final_delta"),
        "blockers": result.get("blockers"),
    }


def main() -> int:
    """Run the two-pulse latency script."""
    load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity_id", help="Mammotion mower entity_id")
    parser.add_argument("--ha-url", default=os.environ.get("HA_URL"))
    parser.add_argument("--ha-token", default=os.environ.get("HA_TOKEN"))
    parser.add_argument(
        "--output-dir",
        default="/tmp/mammotion_forward_two_pulse_latency",  # noqa: S108
    )
    parser.add_argument("--linear-speed", type=int, default=200)
    parser.add_argument("--pulse-count", type=int, default=2)
    parser.add_argument("--pulse-gap-seconds", type=float, default=5.0)
    parser.add_argument("--telemetry-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--telemetry-sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--min-position-change-distance", type=float, default=0.003)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use shorter pulse/telemetry windows for faster troubleshooting runs.",
    )
    parser.add_argument("--use-wifi", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--real", dest="dry_run", action="store_false")
    parser.add_argument("--confirm-blades-off", action="store_true")
    parser.add_argument("--confirm-clear-area", action="store_true")
    parser.add_argument("--no-wait-runtime", action="store_true")
    parser.add_argument("--runtime-timeout", type=int, default=180)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if args.quick:
        _apply_quick_profile(args)

    if not args.ha_url or not args.ha_token:
        parser.error("Provide --ha-url/--ha-token or set HA_URL/HA_TOKEN in .env")
    if not args.dry_run and (
        not args.confirm_blades_off or not args.confirm_clear_area
    ):
        parser.error("Real two-pulse test requires --confirm-blades-off and --confirm-clear-area")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_wait_runtime:
        wait_for_runtime(
            args.ha_url,
            args.ha_token,
            args.entity_id,
            timeout=args.runtime_timeout,
        )
    payload = _payload(args)
    result = post_service(
        args.ha_url,
        args.ha_token,
        "mammotion",
        "forward_two_pulse_latency_test",
        payload,
        args.timeout,
    )
    _write_json(output_dir / "forward_two_pulse_latency.json", {"payload": payload, "result": result})
    summary = _summary(result, output_dir)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))  # noqa: T201
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
