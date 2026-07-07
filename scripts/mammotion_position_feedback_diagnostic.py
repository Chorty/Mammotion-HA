#!/usr/bin/env python3
"""Run Mammotion position feedback refresh diagnostics through Home Assistant."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from mammotion_ha_helpers import load_dotenv, post_service, wait_for_runtime


def _write_json(path: Path, payload: dict) -> None:
    """Write formatted JSON."""
    path.write_text(json.dumps(payload, indent=2))


def _payload(args: argparse.Namespace) -> dict:
    """Build position_feedback_diagnostic payload."""
    return {
        "entity_id": args.entity_id,
        "linear_speed": args.linear_speed,
        "pulse_count": args.pulse_count,
        "pulse_gap_seconds": args.pulse_gap_seconds,
        "refresh_wait_seconds": args.refresh_wait_seconds,
        "prefer_ble": not args.use_wifi,
        "dry_run": args.dry_run,
        "confirm_blades_off": args.confirm_blades_off,
        "confirm_clear_area": args.confirm_clear_area,
    }


def _summary(result: dict, output_dir: Path) -> dict:
    """Return compact diagnostic summary."""
    return {
        "output_dir": str(output_dir),
        "dry_run": result.get("dry_run"),
        "reason": result.get("reason"),
        "passed": result.get("reason") in {
            "dry_run",
            "position_source_changed",
            "metadata_source_changed",
            "position_source_unchanged",
        },
        "pulse_count": result.get("pulse_count"),
        "commands_sent": len([cmd for cmd in result.get("commands", []) if cmd.get("ok")]),
        "refresh_attempts": [
            {
                "name": attempt.get("name"),
                "ok": attempt.get("ok"),
                "error": attempt.get("error"),
            }
            for attempt in result.get("refresh_attempts", [])
        ],
        "position_source_changed": result.get("position_source_changed"),
        "changed_sources": result.get("changed_sources"),
        "position_changed_sources": result.get("position_changed_sources"),
        "metadata_changed_sources": result.get("metadata_changed_sources"),
        "blockers": result.get("blockers"),
        "snapshot_count": len(result.get("snapshots", [])),
    }


def main() -> int:
    """Run the position feedback diagnostic script."""
    load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity_id", help="Mammotion mower entity_id")
    parser.add_argument("--ha-url", default=os.environ.get("HA_URL"))
    parser.add_argument("--ha-token", default=os.environ.get("HA_TOKEN"))
    parser.add_argument(
        "--output-dir",
        default="/tmp/mammotion_position_feedback_diagnostic",  # noqa: S108
    )
    parser.add_argument("--linear-speed", type=int, default=200)
    parser.add_argument("--pulse-count", type=int, default=0)
    parser.add_argument("--pulse-gap-seconds", type=float, default=5.0)
    parser.add_argument("--refresh-wait-seconds", type=float, default=2.0)
    parser.add_argument("--use-wifi", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--real", dest="dry_run", action="store_false")
    parser.add_argument("--confirm-blades-off", action="store_true")
    parser.add_argument("--confirm-clear-area", action="store_true")
    parser.add_argument("--no-wait-runtime", action="store_true")
    parser.add_argument("--runtime-timeout", type=int, default=180)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    if not args.ha_url or not args.ha_token:
        parser.error("Provide --ha-url/--ha-token or set HA_URL/HA_TOKEN in .env")
    if args.pulse_count < 0 or args.pulse_count > 5:
        parser.error("--pulse-count must be between 0 and 5")
    if not args.dry_run and args.pulse_count > 0 and (
        not args.confirm_blades_off or not args.confirm_clear_area
    ):
        parser.error("Real movement requires --confirm-blades-off and --confirm-clear-area")

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
        "position_feedback_diagnostic",
        payload,
        args.timeout,
    )
    _write_json(
        output_dir / "position_feedback_diagnostic.json",
        {"payload": payload, "result": result},
    )
    summary = _summary(result, output_dir)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))  # noqa: T201
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
