#!/usr/bin/env python3
"""Read-only RTK watch: log mower fix state alongside the base station's own.

Why this exists: on 2026-08-07 RTK sat in Float for three hours while the rover
looked healthy -- 24 co-viewed satellites, corrections flowing, no
reference-station error -- and only power-cycling the base cleared it. One
episode is not a diagnosis, so this records the next one with **both ends of the
correction chain** captured at the same moment.

What the 2026-08-07 investigation established, and why this logs what it logs:

* The chain is **internet source -> base station (Wi-Fi) -> LoRa E22 -> mower**.
  The base reports ``position_mode: rtk_over_internet``; the mower reports
  ``rtk_over_datalink``. So a failure *upstream of the base* can degrade the
  corrections it relays while changing nothing the base says about itself.
  ``base position_mode`` and ``wifi_rssi`` are therefore first-class signals.
* ⚠️ **Do not add ``base_moved`` / ``base_moving``.** The query reply on this
  hardware returns ``score_info: null`` -- they are never populated. The field
  exists in the proto and pymammotion reduces it when present; the base simply
  does not send it.
* RTK payload quiet legitimately reaches ~1 h, so a stale-looking feed is not a
  fault and nothing here should treat it as one.
* Base telemetry is only recorded ~hourly by HA's recorder, which is why this
  samples it directly rather than relying on history.

Everything is read-only: it reads ``/api/states`` and sends no command. It also
never exits on a transient error -- an HA restart mid-run must not end a watch
that exists to catch a rare event.

Usage:
    set -a && source .env && set +a
    .venv/bin/python scripts/rtk_watch.py --out rtk_watch.jsonl --interval 30
"""  # noqa: INP001

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mammotion_ha_helpers import load_dotenv  # noqa: E402

#: Mower-side entity suffixes worth recording, mapped to short JSONL keys.
MOWER_FIELDS = {
    "rtk_position": "rtk",
    "position_level": "level",
    "position_mode": "mode",
    "device_position_type": "pos_type",
    "satellites_robot": "sats",
    "l1_satellites_co_viewing": "l1_cov",
    "l2_satellites_co_viewing": "l2_cov",
    "l1_signal_quality": "l1_sq",
    "l2_signal_quality": "l2_sq",
    "rtk_correction_age": "corr_age",
    "last_error": "last_error",
}

#: Base-station entity suffixes. position_mode and wi_fi_rssi are the two that
#: the 2026-08-07 analysis says matter most.
BASE_FIELDS = {
    "position_mode": "base_mode",
    "wi_fi_rssi": "base_rssi",
    "satellites": "base_sats",
    "latitude": "base_lat",
    "longitude": "base_lon",
    "connection_type": "base_conn",
}


def fetch_states(ha_url: str, token: str, timeout: int) -> list[dict[str, Any]] | None:
    """Return all HA states, or None on any failure.

    Deliberately returns None rather than raising: this runs for hours and must
    survive HA restarts, transient 502s and network blips. (``post_service`` in
    the shared helpers raises ``SystemExit``, a BaseException, which silently
    killed an earlier version of this watch on the first restart.)
    """
    request = urllib.request.Request(
        f"{ha_url.rstrip('/')}/api/states",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except BaseException:  # noqa: BLE001 - a watch must never die on a blip
        return None


def discover(states: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Find the mower and base-station entity prefixes on this installation."""
    mower_prefix: str | None = None
    base_prefix: str | None = None
    for state in states:
        entity_id = state["entity_id"]
        if mower_prefix is None and entity_id.startswith("lawn_mower."):
            mower_prefix = f"sensor.{entity_id.split('.', 1)[1]}"
        if base_prefix is None and entity_id.startswith("sensor.rtk"):
            # RTK base entities look like sensor.rtkbna235279309_<field>.
            name = entity_id.split(".", 1)[1]
            for suffix in BASE_FIELDS:
                if name.endswith(f"_{suffix}"):
                    base_prefix = f"sensor.{name[: -len(suffix) - 1]}"
                    break
    return mower_prefix, base_prefix


def sample(
    states: list[dict[str, Any]], mower_prefix: str | None, base_prefix: str | None
) -> dict[str, Any]:
    """Reduce a full state dump to one JSONL record."""
    by_id = {state["entity_id"]: state for state in states}
    record: dict[str, Any] = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epoch": round(time.time(), 1),
    }
    for prefix, fields in ((mower_prefix, MOWER_FIELDS), (base_prefix, BASE_FIELDS)):
        if prefix is None:
            continue
        for suffix, key in fields.items():
            entry = by_id.get(f"{prefix}_{suffix}")
            if entry is not None:
                record[key] = entry.get("state")
    return record


def main() -> int:
    """Run the watch until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="JSONL output path")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 = forever")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    load_dotenv()
    ha_url = os.environ.get("HA_URL")
    token = os.environ.get("HA_TOKEN")
    if not ha_url or not token:
        print("HA_URL and HA_TOKEN must be set (set -a && source .env)")
        return 2

    out = Path(args.out)
    deadline = time.monotonic() + args.duration if args.duration else None
    mower_prefix: str | None = None
    base_prefix: str | None = None
    written = 0
    failures = 0

    while deadline is None or time.monotonic() < deadline:
        states = fetch_states(ha_url, token, args.timeout)
        if states is None:
            # Recorded as a gap rather than dropped, so a Float episode is never
            # confused with a stretch where the logger simply could not see.
            failures += 1
            with out.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "epoch": round(time.time(), 1),
                            "fetch_failed": True,
                        }
                    )
                    + "\n"
                )
        else:
            if mower_prefix is None or base_prefix is None:
                mower_prefix, base_prefix = discover(states)
            record = sample(states, mower_prefix, base_prefix)
            with out.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            written += 1
        time.sleep(args.interval)

    print(f"{written} records, {failures} fetch failures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
