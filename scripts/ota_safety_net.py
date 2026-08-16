"""One-shot OTA safety-net watcher.

2026-08-02 REWIRED: the 1.30.29.8 attempt proved the real OTA trigger is
invisible over HTTP (it never appeared in mitm_ota_capture.jsonl — likely
routed through api-iot-business.cloud-us.mammotion.com, which is
deliberately excluded from TLS interception, or sent over BLE). The channel
that DID see the event, from the true start, was MQTT: the mower publishes
a `thing.properties` message with an `otaProgress` item the instant it
begins (first captured event that run was progress:0). So this watcher now
tails ota_mqtt_capture.jsonl (written by listen_ota_mqtt.py) for the first
message containing an otaProgress item, instead of watching for an
"upgrade" URL over HTTP.

Once triggered, starts a countdown:
  - if scripts/firmware/DOWNLOAD_STARTED appears before the countdown ends,
    our own auto-download got the firmware URL first -> do nothing.
  - otherwise, fire a UniFi block-sta against the mower's MAC to interrupt
    an in-progress download, on the theory an incomplete download can't be
    flashed, preserving a second attempt.

Logs every step with timestamps for after-the-fact review.
"""  # noqa: INP001

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/Users/mattjoslin/Documents/Git Projects/Mammotion-HA")
CAPTURE = REPO / "scripts" / "ota_mqtt_capture.jsonl"
DOWNLOAD_STARTED = REPO / "scripts" / "firmware" / "DOWNLOAD_STARTED"
MOWER_MAC = "a8:b5:8e:2c:52:3f"
COUNTDOWN_SECONDS = 3.0
POLL_INTERVAL = 0.1
LOG = Path(__file__).parent / "ota_safety_net.log"


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


def fire_block() -> None:
    """POST a UniFi block-sta command against MOWER_MAC and log the response."""
    load_dotenv()
    gateway = os.environ["UNIF_GATEWAY_URL"]
    api_key = os.environ["UNIFI_API_KEY"]
    scheme_url = gateway if gateway.startswith("http") else f"https://{gateway}"
    url = f"{scheme_url}/proxy/network/api/s/default/cmd/stamgr"
    body = json.dumps({"cmd": "block-sta", "mac": MOWER_MAC})
    log(f"FIRING BLOCK: POST {url} body={body}")
    result = subprocess.run(
        [
            "curl",
            "-sk",
            "-L",
            "-X",
            "POST",
            url,
            "-H",
            f"X-API-KEY: {api_key}",
            "-H",
            "Content-Type: application/json",
            "-d",
            body,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    log(f"BLOCK RESPONSE: {result.stdout.strip()} stderr={result.stderr.strip()}")


def main() -> None:  # noqa: C901 -- one-off diagnostic watcher, not worth splitting up
    """Watch for the otaProgress trigger, then race DOWNLOAD_STARTED against a block."""
    log(f"safety net armed. countdown={COUNTDOWN_SECONDS}s mower_mac={MOWER_MAC}")
    baseline = len(CAPTURE.read_text().splitlines()) if CAPTURE.exists() else 0
    log(f"baseline capture line count: {baseline}")

    triggered_at: float | None = None
    seen = baseline

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
            if DOWNLOAD_STARTED.exists():
                log(
                    "SUCCESS: DOWNLOAD_STARTED appeared before countdown expired. No block fired."
                )
                return
            elapsed = time.time() - triggered_at
            if elapsed >= COUNTDOWN_SECONDS:
                log(f"COUNTDOWN EXPIRED ({elapsed:.2f}s) with no DOWNLOAD_STARTED.")
                fire_block()
                return


if __name__ == "__main__":
    main()
