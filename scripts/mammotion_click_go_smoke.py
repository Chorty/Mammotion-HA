#!/usr/bin/env python3
"""Run a guarded click/go smoke flow through Home Assistant services."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from mammotion_ha_helpers import (
    build_one_segment_vector_payload,
    load_dotenv,
    post_service,
    wait_for_runtime,
)

DEFAULT_SAMPLE_DELAYS = [0, 5, 10]


def _preview_payload(
    entity_id: str,
    points: list[dict[str, float]],
    *,
    speed: float,
    area_hash: str | None,
) -> dict[str, Any]:
    """Build preview_custom_path payload for one-segment click/go."""
    payload: dict[str, Any] = {
        "entity_id": entity_id,
        "points": points,
        "speed": speed,
        "blade_mode": "off",
    }
    if area_hash:
        payload["area_hash"] = area_hash
    return payload


def _resolve_target(
    runtime_state: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float]:
    """Resolve explicit or runtime-relative target point in map-local coordinates."""
    if args.target_x is not None and args.target_y is not None:
        return {
            "x": round(float(args.target_x), 3),
            "y": round(float(args.target_y), 3),
        }

    position = runtime_state.get("position") or {}
    return {
        "x": round(float(position["x"]) + float(args.offset_x), 3),
        "y": round(float(position["y"]) + float(args.offset_y), 3),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write formatted JSON output."""
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:  # noqa: C901
    """Execute runtime preflight -> preview -> dry-run -> optional guarded real run."""
    load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity_id", help="Mammotion mower entity_id")
    parser.add_argument("--ha-url", default=os.environ.get("HA_URL"))
    parser.add_argument("--ha-token", default=os.environ.get("HA_TOKEN"))
    parser.add_argument(
        "--output-dir",
        default="/tmp/mammotion_click_go_smoke",  # noqa: S108
    )
    parser.add_argument("--target-x", type=float)
    parser.add_argument("--target-y", type=float)
    parser.add_argument(
        "--offset-x",
        type=float,
        default=0.1,
        help="Runtime-relative x offset when explicit target is not provided.",
    )
    parser.add_argument(
        "--offset-y",
        type=float,
        default=0.0,
        help="Runtime-relative y offset when explicit target is not provided.",
    )
    parser.add_argument("--area-hash")
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--use-wifi", action="store_true")
    parser.add_argument("--max-turn-commands", type=int, default=1)
    parser.add_argument("--max-linear-commands", type=int, default=1)
    parser.add_argument(
        "--sample-delays",
        type=float,
        nargs="+",
        default=list(DEFAULT_SAMPLE_DELAYS),
    )
    parser.add_argument("--run-real", action="store_true")
    parser.add_argument("--confirm-blades-off", action="store_true")
    parser.add_argument("--confirm-clear-area", action="store_true")
    parser.add_argument("--runtime-timeout", type=int, default=180)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    if not args.ha_url or not args.ha_token:
        parser.error("Provide --ha-url/--ha-token or set HA_URL/HA_TOKEN in .env or environment")
    if (args.target_x is None) != (args.target_y is None):
        parser.error("Provide both --target-x and --target-y together, or neither to use offsets")
    if args.run_real and (not args.confirm_blades_off or not args.confirm_clear_area):
        parser.error("--run-real requires both --confirm-blades-off and --confirm-clear-area")

    runtime_state = wait_for_runtime(
        args.ha_url,
        args.ha_token,
        args.entity_id,
        timeout=args.runtime_timeout,
    )
    target = _resolve_target(runtime_state, args)
    dry_payload = build_one_segment_vector_payload(
        args.entity_id,
        runtime_state,
        target,
        dry_run=True,
        prefer_ble=not args.use_wifi,
        max_turn_commands=args.max_turn_commands,
        max_linear_commands=args.max_linear_commands,
        sample_delays=args.sample_delays,
    )
    if args.area_hash:
        dry_payload["area_hash"] = args.area_hash

    preview_payload = _preview_payload(
        args.entity_id,
        dry_payload["points"],
        speed=float(args.speed),
        area_hash=args.area_hash,
    )
    preview_result = post_service(
        args.ha_url,
        args.ha_token,
        "mammotion",
        "preview_custom_path",
        preview_payload,
        args.timeout,
    )
    dry_result = post_service(
        args.ha_url,
        args.ha_token,
        "mammotion",
        "raw_pymammotion_execute_vector_segment",
        dry_payload,
        args.timeout,
    )

    real_payload: dict[str, Any] | None = None
    real_result: dict[str, Any] | None = None
    if args.run_real:
        real_payload = build_one_segment_vector_payload(
            args.entity_id,
            runtime_state,
            target,
            dry_run=False,
            prefer_ble=not args.use_wifi,
            max_turn_commands=args.max_turn_commands,
            max_linear_commands=args.max_linear_commands,
            sample_delays=args.sample_delays,
            confirm_blades_off=True,
            confirm_clear_area=True,
        )
        if args.area_hash:
            real_payload["area_hash"] = args.area_hash
        real_result = post_service(
            args.ha_url,
            args.ha_token,
            "mammotion",
            "raw_pymammotion_execute_vector_segment",
            real_payload,
            args.timeout,
        )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "runtime_state.json", runtime_state)
    _write_json(output_dir / "preview.json", {"payload": preview_payload, "result": preview_result})
    _write_json(output_dir / "dry_run.json", {"payload": dry_payload, "result": dry_result})
    if real_payload is not None and real_result is not None:
        _write_json(output_dir / "real_run.json", {"payload": real_payload, "result": real_result})

    summary = {
        "output_dir": str(output_dir),
        "target": target,
        "preview_valid": bool(preview_result.get("valid")),
        "preview_errors": preview_result.get("errors"),
        "dry_run_stop_reason": dry_result.get("stop_reason"),
        "dry_run_blockers": dry_result.get("blockers"),
        "real_run_requested": args.run_real,
        "real_run_stop_reason": None if real_result is None else real_result.get("stop_reason"),
        "real_run_blockers": None if real_result is None else real_result.get("blockers"),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))  # noqa: T201

    if not summary["preview_valid"]:
        return 2
    if summary["dry_run_stop_reason"] == "integration_not_ready":
        return 3
    if args.run_real and summary["real_run_stop_reason"] in (None, "integration_not_ready"):
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
