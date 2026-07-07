#!/usr/bin/env python3
"""Run Mammotion raw vector readiness through Home Assistant."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from mammotion_ha_helpers import call_readiness_with_runtime_retry, load_dotenv


def main() -> int:
    """Run the vector readiness service and save the full response."""
    load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity_id", help="Mammotion mower entity_id")
    parser.add_argument("--ha-url", default=os.environ.get("HA_URL"))
    parser.add_argument("--ha-token", default=os.environ.get("HA_TOKEN"))
    parser.add_argument(
        "--output-dir",
        default="/tmp/mammotion_vector_readiness",  # noqa: S108
    )
    parser.add_argument(
        "--sample-delays",
        type=float,
        nargs="+",
        default=[0, 5, 10, 20, 30, 45, 60],
    )
    parser.add_argument("--use-wifi", action="store_true", help="Use non-BLE transport preference.")
    parser.add_argument("--max-real-steps", type=int, default=0)
    parser.add_argument("--target-distance", type=float, default=0.10)
    parser.add_argument("--turn-delta-degrees", type=float, default=10.0)
    parser.add_argument("--calibrated-forward-heading-offset-degrees", type=float, default=116.5)
    parser.add_argument("--max-turn-commands", type=int, default=4)
    parser.add_argument("--max-linear-commands", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--real", dest="dry_run", action="store_false")
    parser.add_argument("--confirm-blades-off", action="store_true")
    parser.add_argument("--confirm-clear-area", action="store_true")
    parser.add_argument("--no-wait-runtime", action="store_true")
    parser.add_argument("--runtime-timeout", type=int, default=180)
    parser.add_argument("--timeout", type=int, default=720)
    args = parser.parse_args()

    if not args.ha_url or not args.ha_token:
        parser.error("Provide --ha-url/--ha-token or set HA_URL/HA_TOKEN in .env or environment")
    if args.max_real_steps < 0 or args.max_real_steps > 3:
        parser.error("--max-real-steps must be between 0 and 3")
    if not args.dry_run and args.max_real_steps > 0 and (
        not args.confirm_blades_off or not args.confirm_clear_area
    ):
        parser.error("Real vector readiness steps require --confirm-blades-off and --confirm-clear-area")

    payload = {
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
    result = call_readiness_with_runtime_retry(
        args.ha_url,
        args.ha_token,
        "raw_vector_readiness_test",
        payload,
        timeout=args.timeout,
        wait_runtime=not args.no_wait_runtime,
        runtime_timeout=args.runtime_timeout,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"{timestamp}-raw-vector-readiness.json"
    output_path.write_text(json.dumps({"payload": payload, "result": result}, indent=2))

    print(  # noqa: T201
        json.dumps(
            {
                "output": str(output_path),
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
                "recommended_next_step": result.get("recommended_next_step"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
