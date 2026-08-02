#!/usr/bin/env python3
"""Sample mower telemetry through a supervised motion run and log every change.

Read-only. Never commands motion. Exists as a repo script rather than an ad-hoc
one because a scratchpad copy was wiped by /tmp cleanup twice mid-session, and
losing the capture means losing the only per-sample record of a run that cannot
be cheaply repeated.

Captures the fields needed to re-derive the forward-heading offset:
position (RTK, valid in darkness), ``toward`` (course-over-ground), and
``vio_heading`` plus feature count (daylight only). Sampling both heading
sources at once is the point -- on 2026-08-01 ``toward`` did not update at all
across a 1.36 m drive, and only a dense capture makes that visible.

Usage:
    set -a && source .env && set +a
    scripts/motion_capture.py --seconds 300 --out run.jsonl

Then summarise travel bearing vs the configured offset:
    scripts/motion_capture.py --summarise run.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time
import urllib.error
import urllib.request

ENTITY = "lawn_mower.back_yard_clip_skywalker"
SENSORS = ("vio_heading", "vio_tracked_features", "camera_brightness", "ble_rssi")
CONFIGURED_OFFSET = 102.4
# Below this, a travel bearing is dominated by RTK noise rather than motion.
MIN_TRAVEL_FOR_BEARING = 0.20


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ['HA_TOKEN']}",
        "Content-Type": "application/json",
    }


def _post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        os.environ["HA_URL"].rstrip("/") + path,
        headers=_headers(),
        data=json.dumps(payload).encode(),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read() or "{}")


def _sensor(name: str) -> str | None:
    try:
        request = urllib.request.Request(
            f"{os.environ['HA_URL'].rstrip('/')}/api/states/sensor.back_yard_clip_skywalker_{name}",
            headers=_headers(),
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())["state"]
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
        return None


def sample() -> dict:
    """Take one read-only telemetry sample."""
    state = _post(
        "/api/services/mammotion/export_runtime_state?return_response",
        {"entity_id": ENTITY},
    ).get("service_response", {})
    motion = state.get("experimental_motion", {}) or {}
    session = motion.get("active_session") or {}
    position = state.get("position", {}) or {}
    return {
        "t": time.strftime("%H:%M:%S"),
        "epoch": time.time(),
        "x": position.get("x"),
        "y": position.get("y"),
        "toward": position.get("toward"),
        "pos_type": position.get("pos_type_label"),
        "work_mode": state.get("work_mode_label"),
        "session": session.get("session_id"),
        "phase": session.get("phase"),
        **{name: _sensor(name) for name in SENSORS},
    }


def capture(seconds: float, out: pathlib.Path, interval: float) -> int:
    """Sample until the window closes, printing only when something changes."""
    deadline = time.time() + seconds
    previous: dict | None = None
    with out.open("w") as handle:
        while time.time() < deadline:
            try:
                snap = sample()
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
                print(f"[{time.strftime('%H:%M:%S')}] poll error: {type(err).__name__}", flush=True)
                time.sleep(interval)
                continue
            handle.write(json.dumps(snap) + "\n")
            handle.flush()
            moved = previous is not None and (
                abs((snap["x"] or 0) - (previous["x"] or 0)) > 0.01
                or abs((snap["y"] or 0) - (previous["y"] or 0)) > 0.01
            )
            changed = previous is None or moved or any(
                snap[k] != previous[k] for k in ("session", "phase", "work_mode")
            )
            if changed:
                print(
                    f"[{snap['t']}] {snap['work_mode']} session={snap['session']} "
                    f"phase={snap['phase']} x={snap['x']} y={snap['y']} "
                    f"toward={snap['toward']} vio_hdg={snap['vio_heading']}",
                    flush=True,
                )
            previous = snap
            time.sleep(interval)
    print(f"capture ended, wrote {out}", flush=True)
    return 0


def summarise(path: pathlib.Path) -> int:
    """Report travel bearing and the offset it implies, per motion burst."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    moving = [r for r in rows if r.get("x") is not None]
    if len(moving) < 2:
        print("not enough samples")
        return 1
    first, last = moving[0], moving[-1]
    travelled = math.dist((first["x"], first["y"]), (last["x"], last["y"]))
    bearing = math.degrees(math.atan2(last["y"] - first["y"], last["x"] - first["x"])) % 360
    print(f"samples        : {len(moving)}  {first['t']} -> {last['t']}")
    print(f"net travel     : {travelled:.4f} m")
    # .get(), not [] -- captures written by earlier ad-hoc scripts lack some of
    # these keys, and a summariser that cannot read the historical evidence is
    # useless exactly when comparing a new run against an old one.
    for label, row in (("start", first), ("end", last)):
        print(
            f"  {label:5s} toward={row.get('toward')} "
            f"vio_heading={row.get('vio_heading', 'n/a')}"
        )
    # A bearing derived from a near-zero displacement is noise, not a heading.
    # Without this guard a stationary capture happily reports a precise-looking
    # offset (a 0.0000 m capture produced "181.89 deg"), which is exactly how a
    # confident wrong constant gets into a document.
    if travelled < MIN_TRAVEL_FOR_BEARING:
        print(
            f"travel bearing : NOT COMPUTED -- net travel {travelled:.4f} m is under "
            f"{MIN_TRAVEL_FOR_BEARING} m, so the bearing would be noise."
        )
        return 0
    print(f"travel bearing : {bearing:.2f} deg")
    if first.get("toward") is not None:
        implied = (bearing - float(first["toward"])) % 360
        print(f"implied offset : {implied:.2f} deg  (configured {CONFIGURED_OFFSET})")
        print(f"discrepancy    : {implied - CONFIGURED_OFFSET:+.2f} deg")
    if first.get("toward") == last.get("toward"):
        print("⚠️  `toward` did NOT change across the run -- treat it as stale.")
    return 0


def main() -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("motion_capture.jsonl"))
    parser.add_argument("--summarise", type=pathlib.Path, help="summarise an existing capture")
    args = parser.parse_args()
    if args.summarise:
        return summarise(args.summarise)
    return capture(args.seconds, args.out, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
