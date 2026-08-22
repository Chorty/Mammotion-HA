#!/usr/bin/env python3
"""One-step pose prediction on a Phase 1 capture, curvature included.

WHY THIS AND NOT THE MIRROR CHECK. A continuous controller does not consume an
identity between a bearing and a heading. It consumes a **prediction**: from the
last fix, the last heading, and what it just commanded, where will the mower be
when the next fix arrives? That is what this measures, and it is immune to the
pairing convention that made the compass-mirror criterion ill-posed -- a
constant lag is absorbed into the fitted rate rather than deciding the verdict.

The 2-D extension of `scripts/replay_position_predictability.py`, which models
travel distance only (`travel = k * linear_speed * window`) and therefore cannot
score a turn.

**The model.** Speed `v = k_lin * linear_speed`, yaw rate
`w = k_ang * angular_speed`. Over `dt` the mower follows a circular arc of
radius `v / w`, starting from the heading the mirror gives at the interval
start. With `w` at zero it degenerates to a straight line.

⚠️ **Say which constant was fitted where.** `k_lin` fitted on a straight capture
and applied to an arc is genuinely held out. `k_ang` fitted on the only arc
available and scored against that same arc is **in-sample and optimistic**. The
verdict-grade test is a SECOND arc at a different `angular_speed`, scored with
both constants frozen beforehand.

⚠️ **The first interval is an acceleration transient**, not steady state: the
mower is still spinning up, so a constant-velocity model necessarily overshoots
it. `--skip-first` reports both with and without it.

Reads banked capture files. No Home Assistant import, no network, no dispatch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# `map_bearing = 90.13 - toward`; the mirror, not an additive offset.
MIRROR_SUM_DEGREES = 90.13


def _arrivals(path: Path) -> list[dict[str, Any]]:
    """Load a capture and keep one sample per distinct position/toward arrival."""
    raw = json.loads(path.read_text())
    samples = raw.get("service_response", raw)["in_window_telemetry"]["samples"]
    out: list[dict[str, Any]] = []
    seen = None
    for sample in samples:
        position = sample["position"]
        key = (position["x"], position["y"], position["toward"])
        if key != seen:
            out.append(sample)
            seen = key
    return out


def _facing_radians(toward: float) -> float:
    """Map-frame facing, CCW from +x, from the compass mirror."""
    return math.radians((MIRROR_SUM_DEGREES - toward) % 360.0)


def _intervals(arrivals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair consecutive arrivals into intervals."""
    out = []
    for first, second in zip(arrivals, arrivals[1:], strict=False):
        start, end = first["position"], second["position"]
        out.append(
            {
                "dt_s": (second["elapsed_ms"] - first["elapsed_ms"]) / 1000.0,
                "start": start,
                "observed_dx": end["x"] - start["x"],
                "observed_dy": end["y"] - start["y"],
                "toward_rotation_degrees": (end["toward"] - start["toward"] + 180.0)
                % 360.0
                - 180.0,
            }
        )
    return out


def fit_k_lin(intervals: list[dict[str, Any]], linear_speed: int) -> float:
    """Least-squares speed constant from observed travel over commanded window."""
    travelled = sum(math.hypot(i["observed_dx"], i["observed_dy"]) for i in intervals)
    commanded = sum(abs(linear_speed) * i["dt_s"] for i in intervals)
    return travelled / commanded


def fit_k_ang(intervals: list[dict[str, Any]], angular_speed: int) -> float:
    """Least-squares yaw-rate constant from observed `toward` rotation."""
    if not angular_speed:
        return 0.0
    rotated = sum(i["toward_rotation_degrees"] for i in intervals)
    commanded = sum(angular_speed * i["dt_s"] for i in intervals)
    return rotated / commanded


def predict(
    intervals: list[dict[str, Any]],
    *,
    linear_speed: int,
    angular_speed: int,
    k_lin: float,
    k_ang: float,
) -> list[dict[str, Any]]:
    """Score each interval's one-step prediction error in metres."""
    speed = k_lin * linear_speed
    yaw = math.radians(k_ang * angular_speed)
    out = []
    for interval in intervals:
        dt = interval["dt_s"]
        theta = _facing_radians(interval["start"]["toward"])
        if abs(yaw) < 1e-9:
            dx, dy = speed * dt * math.cos(theta), speed * dt * math.sin(theta)
        else:
            radius = speed / yaw
            dx = radius * (math.sin(theta + yaw * dt) - math.sin(theta))
            dy = -radius * (math.cos(theta + yaw * dt) - math.cos(theta))
        error = math.hypot(dx - interval["observed_dx"], dy - interval["observed_dy"])
        out.append(
            {**interval, "predicted_dx": dx, "predicted_dy": dy, "error_m": error}
        )
    return out


def _summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Median and max prediction error."""
    errors = sorted(s["error_m"] for s in scored)
    if not errors:
        return {"count": 0}
    return {
        "count": len(errors),
        "median_m": round(errors[len(errors) // 2], 4),
        "max_m": round(errors[-1], 4),
    }


def main() -> int:
    """Fit on the calibration capture, score the target capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate-straight", type=Path, required=True)
    parser.add_argument("--calibrate-arc", type=Path, default=None)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--score-linear", type=int, default=400)
    parser.add_argument("--score-angular", type=int, default=180)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    straight = _intervals(_arrivals(args.calibrate_straight))
    k_lin = fit_k_lin(straight, 400)
    k_ang = 0.0
    k_ang_source = "none (no arc supplied)"
    if args.calibrate_arc:
        arc = _intervals(_arrivals(args.calibrate_arc))
        k_ang = fit_k_ang(arc, 180)
        same = args.calibrate_arc.resolve() == args.score.resolve()
        k_ang_source = "IN-SAMPLE (optimistic)" if same else "held out"

    scored = predict(
        _intervals(_arrivals(args.score)),
        linear_speed=args.score_linear,
        angular_speed=args.score_angular,
        k_lin=k_lin,
        k_ang=k_ang,
    )

    print(
        f"k_lin = {k_lin:.6e}  (v@400 = {k_lin * 400:.4f} m/s), held out from straight"
    )
    print(f"k_ang = {k_ang:.6e}  (w@180 = {k_ang * 180:.3f} deg/s), {k_ang_source}")
    print(
        f"\n{'dt_s':>6} {'pred_dx':>9} {'pred_dy':>9} {'obs_dx':>9} {'obs_dy':>9} {'err_m':>8}"
    )
    for step in scored:
        print(
            f"{step['dt_s']:6.3f} {step['predicted_dx']:9.4f} {step['predicted_dy']:9.4f} "
            f"{step['observed_dx']:9.4f} {step['observed_dy']:9.4f} {step['error_m']:8.4f}"
        )

    everything = _summary(scored)
    steady = _summary(scored[1:])
    print(f"\n  all intervals      : {everything}")
    print(f"  excluding spin-up  : {steady}")

    result = {
        "mode": "offline_arc_predictability",
        "authoritative": False,
        "k_lin": k_lin,
        "k_ang": k_ang,
        "k_ang_source": k_ang_source,
        "scored_path": str(args.score),
        "steps": scored,
        "all_intervals": everything,
        "excluding_first_interval": steady,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
