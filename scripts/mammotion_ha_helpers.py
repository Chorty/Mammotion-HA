"""Small Home Assistant helpers for Mammotion diagnostic scripts."""  # noqa: INP001

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE lines from .env without external dependencies."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def post_service(
    ha_url: str,
    token: str,
    domain: str,
    service: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    """Call a Home Assistant response service and return service_response."""
    url = f"{ha_url.rstrip('/')}/api/services/{domain}/{service}?return_response"
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
        raise SystemExit(
            f"HA service call failed: HTTP {err.code}: {detail}"
        ) from err


def runtime_ready(runtime_state: dict[str, Any]) -> bool:
    """Return true when Mammotion runtime has live position and heading."""
    position = runtime_state.get("position") or {}
    return (
        position.get("x") is not None
        and position.get("y") is not None
        and position.get("toward") is not None
    )


def wait_for_runtime(
    ha_url: str,
    token: str,
    entity_id: str,
    *,
    timeout: int,
    interval: float = 5.0,
) -> dict[str, Any]:
    """Wait until Mammotion export_runtime_state includes live x/y/toward."""
    deadline = time.monotonic() + timeout
    payload = {"entity_id": entity_id}
    last_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_state = post_service(
            ha_url,
            token,
            "mammotion",
            "export_runtime_state",
            payload,
            min(30, max(1, timeout)),
        )
        if runtime_ready(last_state):
            return last_state
        time.sleep(interval)
    raise SystemExit(
        "Mammotion runtime did not become ready before timeout. "
        f"Last response: {json.dumps(last_state)[:1000]}"
    )


def call_readiness_with_runtime_retry(
    ha_url: str,
    token: str,
    service: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    wait_runtime: bool,
    runtime_timeout: int,
) -> dict[str, Any]:
    """Call readiness service, retrying empty integration-not-ready responses."""
    entity_id = str(payload["entity_id"])
    if wait_runtime:
        wait_for_runtime(
            ha_url,
            token,
            entity_id,
            timeout=runtime_timeout,
        )
    result = post_service(ha_url, token, "mammotion", service, payload, timeout)
    if result or not wait_runtime:
        return result
    wait_for_runtime(
        ha_url,
        token,
        entity_id,
        timeout=runtime_timeout,
    )
    return post_service(ha_url, token, "mammotion", service, payload, timeout)


def build_one_segment_vector_payload(
    entity_id: str,
    runtime_state: dict[str, Any],
    target_point: dict[str, Any],
    *,
    dry_run: bool = True,
    prefer_ble: bool = True,
    max_turn_commands: int = 1,
    max_linear_commands: int = 1,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10),
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
) -> dict[str, Any]:
    """Build guarded one-segment vector payload from live runtime position + target."""
    position = runtime_state.get("position") if isinstance(runtime_state, dict) else None
    if not isinstance(position, dict):
        raise TypeError("runtime_state.position is required")
    if position.get("x") is None or position.get("y") is None:
        raise ValueError("runtime_state.position must include x and y")
    if not isinstance(target_point, dict):
        raise TypeError("target_point must be an object with x and y")
    if target_point.get("x") is None or target_point.get("y") is None:
        raise ValueError("target_point must include x and y")

    start = {
        "x": round(float(position["x"]), 3),
        "y": round(float(position["y"]), 3),
    }
    target = {
        "x": round(float(target_point["x"]), 3),
        "y": round(float(target_point["y"]), 3),
    }
    dry_run_bool = bool(dry_run)
    return {
        "entity_id": entity_id,
        "points": [start, target],
        "dry_run": dry_run_bool,
        "confirm_blades_off": False if dry_run_bool else bool(confirm_blades_off),
        "confirm_clear_area": False if dry_run_bool else bool(confirm_clear_area),
        "prefer_ble": bool(prefer_ble),
        "max_turn_commands": int(max_turn_commands),
        "max_linear_commands": int(max_linear_commands),
        "sample_delays": [float(value) for value in sample_delays],
    }
