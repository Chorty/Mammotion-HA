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
# `vio_detected_features` and `visual_positioning_status` are captured alongside
# the tracked count because tracked alone cannot say WHY a feed is degrading.
# detected-vs-tracked separates "the scene has no texture" (detected also low)
# from "tracking is failing" (detected fine, tracked collapsing). And
# `visual_positioning_status` is the documented dusk-latch discriminator: the
# latch signature is vio_state holding at 2 while the tracked count collapses to
# 0 and the heading freezes bit-identical, so the state must be sampled next to
# the count to see it. On 2026-08-04 a Gate 4 attempt was read as "light
# degradation" purely from tracked falling 71 -> 58, with no way to confirm which
# failure it was.
SENSORS = (
    "vio_heading",
    "vio_tracked_features",
    "vio_detected_features",
    "visual_positioning_status",
    "camera_brightness",
    "ble_rssi",
)
#: `toward` is a compass bearing and the travel bearing computed here is a math
#: angle, so a correct reading satisfies `bearing + toward == 90`. This replaces
#: the former additive `CONFIGURED_OFFSET = 102.4`, which compared the two
#: conventions directly and could therefore only ever be right at one heading.
COMPASS_MIRROR_DEGREES = 90.0
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
    except urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError:
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
                print(
                    f"[{time.strftime('%H:%M:%S')}] poll error: {type(err).__name__}",
                    flush=True,
                )
                time.sleep(interval)
                continue
            handle.write(json.dumps(snap) + "\n")
            handle.flush()
            moved = previous is not None and (
                abs((snap["x"] or 0) - (previous["x"] or 0)) > 0.01
                or abs((snap["y"] or 0) - (previous["y"] or 0)) > 0.01
            )
            changed = (
                previous is None
                or moved
                or any(
                    snap[k] != previous[k] for k in ("session", "phase", "work_mode")
                )
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


def _signed_delta(value: float, reference: float) -> float:
    """Return the wrap-aware signed difference between two angles, in (-180, 180]."""
    return (value - reference + 180) % 360 - 180


#: A per-step displacement below this is dominated by position noise rather than
#: motion, so its bearing is meaningless. Larger than the whole-run threshold
#: because a single step has far less baseline to average the noise out.
MIN_STEP_FOR_BEARING = 0.25


def _report_compass_mirror(moving: list[dict]) -> None:
    """Check `bearing + toward == 90` PER STEP, not across the whole capture.

    `toward` is a COMPASS bearing (clockwise from north) while the bearing from
    ``atan2(dy, dx)`` is a MATH angle (counter-clockwise from +x), so a correct
    reading satisfies ``bearing + toward == 90``. The former code compared them
    with a subtraction against a 102.4 "configured offset"; since
    ``bearing - toward == 90 - 2*toward``, no constant could ever fit and the
    value appeared to drift with heading.

    This is deliberately per-step. A first-vs-last comparison across a capture
    spanning several legs measures a net vector that is not a travel direction at
    all -- on the 2026-08-04 night mow that produced a 101 deg "deviation" from a
    dataset whose per-step quadrant means were 88.43 / 90.88 / 89.71 / 90.36.
    Same aggregate-vs-per-item error the rest of this file guards against.

    Steps whose `toward` did not change are skipped: `toward` only tracks during
    CONTINUOUS motion. In pulsed motion it stays frozen for a whole leg (measured
    over 0.54 m and 0.66 m legs), which silently poisons any average.
    """
    sin_sum = cos_sum = 0.0
    used = 0
    stale = 0
    for previous, current in zip(moving, moving[1:], strict=False):
        if previous.get("toward") is None or current.get("toward") is None:
            continue
        step = math.dist((previous["x"], previous["y"]), (current["x"], current["y"]))
        if step < MIN_STEP_FOR_BEARING:
            continue
        if previous["toward"] == current["toward"]:
            stale += 1
            continue
        step_bearing = math.degrees(
            math.atan2(current["y"] - previous["y"], current["x"] - previous["x"])
        )
        value = math.radians((step_bearing + float(current["toward"])) % 360)
        sin_sum += math.sin(value)
        cos_sum += math.cos(value)
        used += 1
    if not used:
        print(
            "bearing+toward : NOT COMPUTED -- no step cleared "
            f"{MIN_STEP_FOR_BEARING} m with a fresh `toward` "
            f"({stale} stale step(s) skipped)."
        )
        return
    mean = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
    resultant = math.hypot(sin_sum, cos_sum) / used
    spread = math.degrees(math.sqrt(max(0.0, -2 * math.log(resultant))))
    print(
        f"bearing+toward : {mean:.2f} deg over {used} step(s), circular sd "
        f"{spread:.2f} deg  (expected ~{COMPASS_MIRROR_DEGREES})"
    )
    print(f"deviation      : {_signed_delta(mean, COMPASS_MIRROR_DEGREES):+.2f} deg")
    if stale:
        print(
            f"  note: skipped {stale} step(s) whose `toward` was stale (pulsed motion)"
        )


def summarise(path: pathlib.Path) -> int:
    """Report travel bearing and the compass-mirror check, per motion step."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    moving = [r for r in rows if r.get("x") is not None]
    if len(moving) < 2:
        print("not enough samples")
        return 1
    first, last = moving[0], moving[-1]
    travelled = math.dist((first["x"], first["y"]), (last["x"], last["y"]))
    bearing = (
        math.degrees(math.atan2(last["y"] - first["y"], last["x"] - first["x"])) % 360
    )
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
    print(f"travel bearing : {bearing:.2f} deg  (net first->last; see per-step below)")
    _report_compass_mirror(moving)
    if first.get("toward") == last.get("toward"):
        # Only trustworthy during CONTINUOUS motion: in pulsed motion `toward`
        # stays frozen across an entire leg (measured 2026-08-04 over 0.54 m and
        # 0.66 m legs), while a continuous mow updated it 35 times in 20.7 m.
        print("⚠️  `toward` did NOT change across the run -- treat it as stale.")
    return 0


def main() -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("motion_capture.jsonl")
    )
    parser.add_argument(
        "--summarise", type=pathlib.Path, help="summarise an existing capture"
    )
    args = parser.parse_args()
    if args.summarise:
        return summarise(args.summarise)
    return capture(args.seconds, args.out, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
