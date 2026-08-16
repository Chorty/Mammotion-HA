"""One-shot OTA URL-probe burst watcher.

Companion to ota_safety_net.py, run alongside it (not instead of it) --
this watches the same trigger and fires a *different* action: a burst of
read-only ota_info_probe service calls, over the mower's own BLE
connection, hoping to catch fota_sub_info.sub_img_url populated while a
real OTA is actually active (it was empty/null when queried idle).

Uses the exact same trigger-detection logic as ota_safety_net.py: tails
ota_mqtt_capture.jsonl for the first message containing an otaProgress
item (proven twice now to arrive right at the true start of a real
download). On trigger, fires ota_info_probe immediately, then every ~2s
for ~10s (6 calls total), logging the full raw response of each.

This is independent of and does not interfere with:
  - ota_safety_net.py's block-on-timeout (different transport: BLE here,
    UniFi/WiFi there)
  - a manual UniFi block of the mower's network, which only affects the
    WiFi download path, never the BLE probe calls here.

Read-only: ota_info_probe sends only MctlOta.todev_get_info_req(type=IT_OTA)
-- never fw_download_ctrl, never device/upgrade. Cannot trigger or affect
an install either way.
"""  # noqa: INP001

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/Users/mattjoslin/Documents/Git Projects/Mammotion-HA")
CAPTURE = REPO / "scripts" / "ota_mqtt_capture.jsonl"
ENTITY_ID = "lawn_mower.back_yard_clip_skywalker"
BURST_OFFSETS_SECONDS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
SEND_TIMEOUT = 5.0
POLL_INTERVAL = 0.1
LOG = Path(__file__).parent / "ota_url_probe_burst.log"


def log(msg: str) -> None:
    """Print *msg* with a millisecond timestamp and append it to LOG."""
    line = f"{time.strftime('%H:%M:%S')}.{int((time.time() % 1) * 1000):03d} {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def load_dotenv() -> None:
    """Load KEY=VALUE lines from .env into the environment."""
    path = REPO / ".env"
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def fire_probe(attempt: int) -> None:
    """Call the ota_info_probe HA service once and log the raw response."""
    load_dotenv()
    ha_url = os.environ["HA_URL"]
    ha_token = os.environ["HA_TOKEN"]
    url = f"{ha_url}/api/services/mammotion/ota_info_probe?return_response"
    body = json.dumps({"entity_id": ENTITY_ID, "send_timeout": SEND_TIMEOUT})
    log(f"PROBE #{attempt}: POST {url}")
    result = subprocess.run(
        [
            "curl",
            "-sk",
            "-X",
            "POST",
            url,
            "-H",
            f"Authorization: Bearer {ha_token}",
            "-H",
            "Content-Type: application/json",
            "-d",
            body,
        ],
        capture_output=True,
        text=True,
        timeout=SEND_TIMEOUT + 5,
        check=False,
    )
    log(f"PROBE #{attempt} RESPONSE: {result.stdout.strip()}")
    if result.stderr.strip():
        log(f"PROBE #{attempt} STDERR: {result.stderr.strip()}")


def main() -> None:  # noqa: C901 -- one-off diagnostic watcher, not worth splitting up
    """Watch for the otaProgress trigger, then fire a timed burst of probes."""
    log("burst watcher armed. waiting for otaProgress trigger.")
    baseline = len(CAPTURE.read_text().splitlines()) if CAPTURE.exists() else 0
    log(f"baseline capture line count: {baseline}")

    triggered_at: float | None = None
    seen = baseline
    fired: set[int] = set()

    while True:
        time.sleep(POLL_INTERVAL)

        if triggered_at is None and CAPTURE.exists():
            lines = CAPTURE.read_text().splitlines()
            if len(lines) > seen:
                for line in lines[seen:]:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = rec.get("text")
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    ota = payload.get("params", {}).get("items", {}).get("otaProgress")
                    if ota is not None:
                        triggered_at = time.time()
                        value = ota.get("value", {})
                        log(
                            "TRIGGER DETECTED: otaProgress "
                            f"progress={value.get('progress')} message={value.get('message')!r}"
                        )
                        break
                seen = len(lines)

        if triggered_at is not None:
            elapsed = time.time() - triggered_at
            for i, offset in enumerate(BURST_OFFSETS_SECONDS):
                if i not in fired and elapsed >= offset:
                    fired.add(i)
                    fire_probe(i + 1)
            if len(fired) == len(BURST_OFFSETS_SECONDS):
                log("burst complete. exiting.")
                return


if __name__ == "__main__":
    main()
