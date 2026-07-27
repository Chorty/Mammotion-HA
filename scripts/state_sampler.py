#!/usr/bin/env python3
"""Sample the mower's self-reported BLE/transport state on a fixed cadence.

Pairs with ``ble_advert_monitor.py``: that records when the radio was
demonstrably on air (observed by HA's own scanners), this records what the mower
*claims* about itself. Comparing the two quantifies how stale ``ble_rssi`` gets
once the radio goes quiet -- the ``@`` columns are each entity's
``last_updated``, so a frozen timestamp beside a healthy-looking value is the
tell.

Read-only.

Usage:  .venv/bin/python scripts/state_sampler.py [seconds] [interval_seconds]
"""  # noqa: INP001

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

from ble_advert_monitor import load_env

ENTITIES = (
    "sensor.back_yard_clip_skywalker_ble_rssi",
    "sensor.back_yard_clip_skywalker_active_transport",
    "sensor.back_yard_clip_skywalker_activity_mode",
    "sensor.back_yard_clip_skywalker_wi_fi_rssi",
    "sensor.back_yard_clip_skywalker_connection",
    "switch.back_yard_clip_skywalker_bluetooth",
)
OUT = Path(__file__).with_name("state_log.jsonl")


def main(duration_s: float = 1800.0, every_s: float = 30.0) -> None:
    """Poll the REST API and append one JSON row per sample."""
    ws_url, token = load_env()
    base = ws_url.replace("wss://", "https://").replace("ws://", "http://")
    base = base[: -len("/api/websocket")]
    deadline = time.time() + duration_s

    with OUT.open("w") as handle:
        while time.time() < deadline:
            row: dict[str, object] = {
                "iso": time.strftime("%H:%M:%S", time.localtime())
            }
            for entity in ENTITIES:
                short = entity.split("skywalker_")[1]
                request = urllib.request.Request(  # noqa: S310
                    f"{base}/api/states/{entity}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                try:
                    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
                        payload = json.load(response)
                except (OSError, ValueError) as exc:
                    row[short] = f"ERR {type(exc).__name__}"
                    continue
                row[short] = payload.get("state")
                row[f"{short}@"] = (payload.get("last_updated") or "")[11:19]
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            time.sleep(every_s)


if __name__ == "__main__":
    main(
        float(sys.argv[1]) if len(sys.argv) > 1 else 1800.0,
        float(sys.argv[2]) if len(sys.argv) > 2 else 30.0,
    )
