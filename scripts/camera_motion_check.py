#!/usr/bin/env python3
"""Capture backyard-camera frames across a supervised run as durable evidence.

**Primary use: path quality.** The operator's 2026-08-06 fixed-camera recording
showed repeated pivots, reversals and partial backtracking that the run's own
telemetry described only as two `target_reached` segments. That gap -- a nominal
pass hiding a badly-tracked path -- is what this automates, so a run's visual
record is captured rather than remembered.

Secondary: a crude did-it-move-at-all check. Note this was NOT built as an
e-stop detector; that failure mode (2026-07-19, five commands silently no-op'd)
was reviewed on 2026-08-07 and deliberately deprioritised, since a mower that
does nothing is self-announcing and harmless.

Read-only with respect to the mower: this only fetches camera snapshots from
Home Assistant. It sends no command and cannot move anything.

**Deliberately numeric.** Frame comparison happens in numpy and the output is a
handful of numbers. Reviewing frames by eye costs roughly 1,800 tokens each, so
a 90-second run inspected frame-by-frame would cost hundreds of thousands of
tokens to answer what arithmetic answers for almost nothing. Inspect the few
frames the report flags, not the whole capture.

⚠️ **It detects SCENE motion, not MOWER motion.** This is the limitation that
bites. On the very first 90-second stationary capture it reported movement --
correctly, because a dog walked through the frame while the mower sat still.
Anything that moves in view (person, animal, vehicle, a wind-blown toy) trips
it. Attribution is the caller's job, and the report exists to make that
possible: ``peak_block_row_col`` locates the change on a coarse grid, so a
flagged frame can be checked against where the mower actually is before
concluding anything.

⚠️ **What this cannot do.** The camera is not calibrated to mower map
coordinates, so pixel motion does not convert to metres and this can never
substitute for RTK. It supports qualitative claims -- moved / did not move,
turned in place, reversed -- not metric distance or map bearing. It is also not
a safety device: snapshot polling is far too slow and too late to intervene, and
the supervising operator remains the safety control.

Usage:
    set -a && source .env && set +a
    scripts/camera_motion_check.py --seconds 90 --out ~/run-evidence
    scripts/camera_motion_check.py --baseline --out ~/baseline
"""  # noqa: INP001

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mammotion_ha_helpers import load_dotenv  # noqa: E402

#: The backyard UniFi G6 Turret, not the mower's onboard camera. The mower's own
#: entity (`camera.back_yard_clip_skywalker`, model Luba-VSPLV397) looks forward
#: from the machine and cannot show the machine, so it is useless as an
#: actuation detector.
DEFAULT_CAMERA = "camera.g6_turret_high_resolution_channel"

#: Frames are compared at this size, not native 4K. Downscaling suppresses
#: sensor noise and JPEG artefacts, and the mower is a large pale object on
#: grass -- detecting whether it moved needs nothing like 4K. Native frames are
#: ~1.7 MB each; a 90 s run at full resolution is ~300 MB, and this project has
#: already lost a run's durable evidence to a full disk (day2b, 2026-08-05).
COMPARE_SIZE = (960, 540)

#: Per-pixel absolute luminance change counted as "different". Below roughly 25
#: ordinary sensor noise and JPEG ringing dominate; grass moving in wind sits
#: well under this at the compare size.
PIXEL_DELTA_THRESHOLD = 35

#: Whole-frame changed fraction. Retained as a diagnostic only -- it is NOT the
#: movement criterion, because it cannot work: at this camera distance the mower
#: covers roughly 0.1% of the frame, so even driving 1.6 m moved only 0.20-0.32%
#: of pixels (measured 2026-08-07) against a stationary lighting floor that
#: itself reaches 0.295%. Signal and noise overlap, and an early version of this
#: script used exactly this statistic and reported "no movement" for a mower
#: that had demonstrably driven 1.6 m.
MOVEMENT_FRACTION_THRESHOLD = 0.005

#: The real criterion: the most-changed block. Localised change separates
#: cleanly where whole-frame change does not. Same 2026-08-07 run -- the three
#: linear pulses lit block [2,8] to 17.4%, 41.3% and 29.8%, while every other
#: step in the capture stayed at or below 3.4% and sat in a different block.
#: 10% is placed in that gap, ~1.7x above the observed non-mower maximum and
#: ~1.7x below the weakest true detection.
MOVEMENT_PEAK_BLOCK_THRESHOLD = 0.10

#: ⚠️ SENSITIVITY FLOOR. Detection needs the mower to vacate enough of one block
#: to clear the threshold, so this is validated for moves of roughly a metre and
#: up (1.6 m detected 2026-08-07). A few centimetres shifts only edge pixels and
#: will NOT register: it cannot confirm a small bounded move such as Gate 2's
#: 9 cm, and must never be used to argue a waypoint was or was not reached.


def _fetch(ha_url: str, token: str, camera: str, dest: Path) -> tuple[bool, str]:
    """Fetch one camera snapshot to *dest*. Returns (ok, http_status).

    The proxy answers 500 when polled too hard -- measured 2026-08-07, a burst
    of ~19 snapshots inside a minute pushed it into 10-second timeouts returning
    a 26-byte error body. That body lands in the destination file, so size alone
    is not proof of an image; the status code is captured and reported instead
    of failing silently.
    """
    url = f"{ha_url.rstrip('/')}/api/camera_proxy/{urllib.parse.quote(camera)}"
    completed = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "curl",
            "-s",
            "-o",
            str(dest),
            "-w",
            "%{http_code}",
            "--max-time",
            "20",
            "-H",
            f"Authorization: Bearer {token}",
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    status = (completed.stdout or "").strip() or "000"
    ok = status == "200" and dest.exists() and dest.stat().st_size > 1024
    if not ok:
        dest.unlink(missing_ok=True)
    return ok, status


def _load(path: Path) -> np.ndarray:
    """Load a frame as a downscaled greyscale array for comparison."""
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L").resize(COMPARE_SIZE), dtype=np.int16)


#: Side of the square blocks the change mask is pooled into, in compare-size
#: pixels. Pooling separates *concentrated* change from *diffuse* change:
#: measured 2026-08-07, a dog crossing the frame lit one block to 42.9% while
#: the whole-frame mean was 0.58%, whereas drifting daylight raises many blocks
#: slightly and no single one much. Reporting where the change is, is what makes
#: attribution possible at all.
BLOCK_PIXELS = 60


def compare(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Compare two frames and report whether, and where, the scene changed."""
    delta = np.abs(second - first)
    mask = delta > PIXEL_DELTA_THRESHOLD
    fraction = float(mask.mean())

    height, width = mask.shape
    rows, cols = height // BLOCK_PIXELS, width // BLOCK_PIXELS
    blocks = (
        mask[: rows * BLOCK_PIXELS, : cols * BLOCK_PIXELS]
        .reshape(rows, BLOCK_PIXELS, cols, BLOCK_PIXELS)
        .mean(axis=(1, 3))
    )
    hottest = np.unravel_index(int(blocks.argmax()), blocks.shape)

    return {
        "mean_abs_delta": round(float(delta.mean()), 3),
        "changed_pixels": int(mask.sum()),
        "changed_fraction": round(fraction, 6),
        "peak_block_fraction": round(float(blocks.max()), 6),
        # Grid coordinates of the most-changed block, so a flagged frame can be
        # attributed to the mower or to something else in the scene without
        # viewing every frame.
        "peak_block_row_col": [int(hottest[0]), int(hottest[1])],
        "block_grid": [rows, cols],
        "blocks_over_2pct": int((blocks > 0.02).sum()),
        "movement": bool(blocks.max() > MOVEMENT_PEAK_BLOCK_THRESHOLD),
        "movement_by_whole_frame": bool(fraction > MOVEMENT_FRACTION_THRESHOLD),
    }


def capture(
    ha_url: str,
    token: str,
    camera: str,
    seconds: float,
    out_dir: Path,
    interval: float,
) -> dict[str, Any]:
    """Capture frames for *seconds* and report per-step and overall motion."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[float, Path]] = []
    failures: dict[str, int] = {}
    started = time.monotonic()
    index = 0
    backoff = 0.0
    while time.monotonic() - started < seconds:
        dest = out_dir / f"frame_{index:04d}.jpg"
        ok, status = _fetch(ha_url, token, camera, dest)
        if ok:
            frames.append((round(time.monotonic() - started, 2), dest))
            index += 1
            backoff = 0.0
        else:
            failures[status] = failures.get(status, 0) + 1
            # Back off on a throttled proxy instead of hammering it harder,
            # which is what produced the 500s in the first place.
            backoff = min(backoff * 2 if backoff else 2.0, 15.0)
            time.sleep(backoff)
        time.sleep(max(0.0, interval - (time.monotonic() - started) % interval))

    if len(frames) < 2:
        return {
            "frames": len(frames),
            "fetch_failures": failures,
            "error": (
                "need at least two frames; "
                + (
                    f"all fetches failed with HTTP {max(failures, key=failures.get)} "
                    "-- the camera proxy throttles under rapid polling, so raise "
                    "--interval and retry"
                    if failures
                    else "no fetches were attempted"
                )
            ),
        }

    arrays = [(stamp, _load(path)) for stamp, path in frames]
    steps = [
        {"from_s": arrays[i][0], "to_s": arrays[i + 1][0]}
        | compare(arrays[i][1], arrays[i + 1][1])
        for i in range(len(arrays) - 1)
    ]
    overall = compare(arrays[0][1], arrays[-1][1])
    moving = [s for s in steps if s["movement"]]
    return {
        "camera": camera,
        "frames": len(frames),
        "fetch_failures": failures,
        "duration_s": frames[-1][0],
        "effective_fps": round(len(frames) / max(frames[-1][0], 1e-6), 3),
        "compare_size": list(COMPARE_SIZE),
        "pixel_delta_threshold": PIXEL_DELTA_THRESHOLD,
        "movement_fraction_threshold": MOVEMENT_FRACTION_THRESHOLD,
        "steps_with_movement": len(moving),
        "first_movement_at_s": moving[0]["from_s"] if moving else None,
        "last_movement_at_s": moving[-1]["to_s"] if moving else None,
        "peak_changed_fraction": max(s["changed_fraction"] for s in steps),
        "first_to_last": overall,
        # The headline. False after a run that reported dispatched commands is
        # the 2026-07-19 signature: commands accepted, nothing actuated.
        "any_movement_detected": bool(moving) or overall["movement"],
        "steps": steps,
        "limitation": (
            "Uncalibrated fixed camera: supports moved/did-not-move and gross "
            "path shape only, never metric distance or map bearing."
        ),
    }


def main() -> int:
    """Parse arguments and run the capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument(
        "--interval",
        type=float,
        default=4.0,
        help=(
            "Seconds between snapshot fetches. The HA camera proxy returns 500 "
            "under rapid polling (measured 2026-08-07), and each 4K frame takes "
            "~2.2 s to transfer, so the practical ceiling is ~0.3 fps."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Directory for captured frames and the JSON report. Required "
            "rather than defaulted: a 90 s capture is tens of megabytes and a "
            "full disk has already destroyed one run's durable evidence "
            "(day2b, 2026-08-05), so the caller picks the volume."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Short stationary capture to measure the scene's noise floor.",
    )
    args = parser.parse_args()

    load_dotenv()
    ha_url = os.environ.get("HA_URL")
    token = os.environ.get("HA_TOKEN")
    if not ha_url or not token:
        print("HA_URL and HA_TOKEN must be set (set -a && source .env)")
        return 2

    seconds = 20.0 if args.baseline else args.seconds
    report = capture(ha_url, token, args.camera, seconds, args.out, args.interval)
    summary = {k: v for k, v in report.items() if k != "steps"}
    print(json.dumps(summary, indent=2))
    (args.out / "camera_motion_check.json").write_text(json.dumps(report, indent=2))
    print(f"\nfull report -> {args.out / 'camera_motion_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
