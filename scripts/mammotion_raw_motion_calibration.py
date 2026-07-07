#!/usr/bin/env python3
"""Run raw Mammotion motion calibration probes through Home Assistant."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROBES: tuple[dict[str, Any], ...] = (
    {"name": "linear_plus_400", "command": "send_movement", "linear_speed": 400, "angular_speed": 0},
    {"name": "linear_minus_400", "command": "send_movement", "linear_speed": -400, "angular_speed": 0},
    {"name": "angular_plus_180", "command": "send_movement", "linear_speed": 0, "angular_speed": 180},
    {"name": "angular_minus_180", "command": "send_movement", "linear_speed": 0, "angular_speed": -180},
    {"name": "linear_plus_200", "command": "send_movement", "linear_speed": 200, "angular_speed": 0},
    {"name": "linear_minus_200", "command": "send_movement", "linear_speed": -200, "angular_speed": 0},
    {"name": "linear_plus_100", "command": "send_movement", "linear_speed": 100, "angular_speed": 0},
    {"name": "linear_minus_100", "command": "send_movement", "linear_speed": -100, "angular_speed": 0},
)


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE lines from .env without external dependencies."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _post_service(ha_url: str, token: str, service: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Call an HA service and return service_response."""
    url = f"{ha_url.rstrip('/')}/api/services/mammotion/{service}?return_response"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response).get("service_response", {})
    except urllib.error.HTTPError as err:
        detail = err.read().decode(errors="replace")
        raise SystemExit(f"HA service call failed: HTTP {err.code}: {detail}") from err


def _probe_payload(args: argparse.Namespace, probe: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "entity_id": args.entity_id,
        "command": probe["command"],
        "prefer_ble": not args.use_wifi,
        "sample_delays": args.sample_delays,
        "dry_run": args.dry_run,
        "confirm_blades_off": args.confirm_blades_off,
        "confirm_clear_area": args.confirm_clear_area,
    }
    if probe["command"] == "send_movement":
        payload["linear_speed"] = probe["linear_speed"]
        payload["angular_speed"] = probe["angular_speed"]
    else:
        payload["speed"] = probe["speed"]
    return payload


def main() -> int:
    """Run selected calibration probes."""
    _load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity_id", help="Mammotion mower entity_id")
    parser.add_argument("--ha-url", default=os.environ.get("HA_URL"))
    parser.add_argument("--ha-token", default=os.environ.get("HA_TOKEN"))
    parser.add_argument(
        "--output-dir",
        default="/tmp/mammotion_motion_calibration",  # noqa: S108
    )
    parser.add_argument("--probe", choices=[probe["name"] for probe in PROBES])
    parser.add_argument("--sample-delays", type=float, nargs="+", default=[0, 5, 10, 20, 30, 45, 60])
    parser.add_argument("--use-wifi", action="store_true", help="Use non-BLE transport preference.")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--real", dest="dry_run", action="store_false", help="Send real one-command probes.")
    parser.add_argument("--confirm-blades-off", action="store_true")
    parser.add_argument("--confirm-clear-area", action="store_true")
    parser.add_argument("--yes-run-all", action="store_true", help="Run all selected real probes without prompting between probes.")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if not args.ha_url or not args.ha_token:
        parser.error("Provide --ha-url/--ha-token or set HA_URL/HA_TOKEN in .env or environment")
    if not args.dry_run and (not args.confirm_blades_off or not args.confirm_clear_area):
        parser.error("Real probes require --confirm-blades-off and --confirm-clear-area")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probes = [probe for probe in PROBES if args.probe in (None, probe["name"])]

    for index, probe in enumerate(probes, start=1):
        if not args.dry_run and not args.yes_run_all:
            answer = input(f"Run real probe {index}/{len(probes)} {probe['name']}? Type RUN: ")
            if answer != "RUN":
                print("Stopped before probe.")  # noqa: T201
                return 1
        payload = _probe_payload(args, probe)
        result = _post_service(
            args.ha_url,
            args.ha_token,
            "raw_pymammotion_motion_probe",
            payload,
            args.timeout,
        )
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = output_dir / f"{timestamp}-{probe['name']}.json"
        output_path.write_text(json.dumps({"probe": probe, "payload": payload, "result": result}, indent=2))
        summary = result.get("motion_interpretation", {})
        print(  # noqa: T201
            json.dumps(
                {
                    "probe": probe["name"],
                    "output": str(output_path),
                    "dry_run": result.get("dry_run"),
                    "command_ok": (result.get("command_result") or {}).get("ok"),
                    "status": summary.get("status"),
                    "delta": summary.get("delta"),
                    "movement_heading_degrees": summary.get(
                        "movement_heading_degrees"
                    ),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
