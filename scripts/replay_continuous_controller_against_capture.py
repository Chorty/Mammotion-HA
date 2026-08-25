#!/usr/bin/env python3
"""Replay the pure Phase 0 controller against a REAL banked capture.

`scripts/replay_continuous_controller.py` replays hand-built synthetic
scenarios. This script builds a scenario from an actual `raw_pymammotion_probe`
capture's `in_window_telemetry.samples`, so the reconciled constants
(`nominal_speed_mps`, `max_refresh_age_s`) are checked against real telemetry
cadence and real BLE stall behaviour rather than invented numbers.

Still no dispatch. It imports `continuous_controller.py` directly (no Home
Assistant, no coordinator, no BLE) and calls `continuous_control_decision`
exactly as `replay_continuous_controller.py` does; it only differs in where the
scenario comes from.

**The route target is placed far beyond the observed path on purpose.** The
question this replay answers is "do the reconciled fail-closed thresholds fire
correctly against real telemetry", not "does route-completion logic work" --
that would confound a target_reached/target_passed stop with a
refresh/telemetry stop. A distant target keeps every stop in the replay
attributable to the constant being tested.

Convention: `course_heading_degrees` is map-local, `atan2(dy, dx)` CCW from
+x -- the same frame as `Point.x/y` -- derived only from fresh >=0.15 m
position chords. The capture's `toward` field is deliberately ignored.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = REPO / "custom_components" / "mammotion" / "continuous_controller.py"
_SPEC = importlib.util.spec_from_file_location(
    "mammotion_continuous_controller_offline_capture", CONTROLLER_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load controller module from {CONTROLLER_PATH}")
_CONTROLLER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CONTROLLER
_SPEC.loader.exec_module(_CONTROLLER)

ContinuousControllerConfig = _CONTROLLER.ContinuousControllerConfig
ContinuousObservation = _CONTROLLER.ContinuousObservation
ContinuousRoute = _CONTROLLER.ContinuousRoute
HeadingEvidence = _CONTROLLER.HeadingEvidence
Point = _CONTROLLER.Point
course_from_position_chord = _CONTROLLER.course_from_position_chord
continuous_control_decision = _CONTROLLER.continuous_control_decision

# How far beyond the observed path to place the target, so target-completion
# logic never fires during the replay window.
TARGET_OVERSHOOT_M = 10.0


def _arrivals(capture_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a capture and keep one sample per distinct position arrival."""
    raw = json.loads(capture_path.read_text())
    body = raw.get("service_response", raw)
    body = body.get("run", body)
    if "in_window_telemetry" in body:
        samples = body["in_window_telemetry"]["samples"]
    else:
        samples = [
            {
                "elapsed_ms": float(decision["elapsed_s"]) * 1000.0,
                "position": decision["observation"]["position"],
            }
            for decision in body.get("decisions", [])
        ]
    out: list[dict[str, Any]] = []
    seen = None
    for sample in samples:
        position = sample["position"]
        key = (position["x"], position["y"])
        if key != seen:
            out.append(sample)
            seen = key
    return out, body


def _refresh_age_s(elapsed_ms: float, completions_ms: list[float]) -> float:
    """Time since the most recent refresh write completed, at `elapsed_ms`."""
    prior = [c for c in completions_ms if c <= elapsed_ms]
    if not prior:
        return elapsed_ms / 1000.0
    return (elapsed_ms - max(prior)) / 1000.0


def _refresh_max_gap_s(
    window_start_ms: float, window_end_ms: float, completions_ms: list[float]
) -> float:
    """Worst gap between consecutive completions inside one decision window.

    This is the field a real executor would have to track as a running max --
    `_refresh_age_s` above is a POINT SAMPLE and, as this replay demonstrated
    against `evidence-8s-continuous-window-20260822T233000Z.json`, can read
    healthy immediately after a stall that already resolved.
    """
    in_window = [c for c in completions_ms if window_start_ms <= c <= window_end_ms]
    bounded = [window_start_ms, *in_window, window_end_ms]
    if len(bounded) < 2:
        return 0.0
    return max(b - a for a, b in zip(bounded, bounded[1:], strict=False)) / 1000.0


def build_scenario(capture_path: Path) -> dict[str, Any]:
    """Build a controller scenario from one banked capture's real telemetry."""
    arrivals, body = _arrivals(capture_path)
    completions_ms = body.get("motion_refresh", {}).get(
        "refresh_write_completions_elapsed_ms", []
    )
    if not completions_ms:
        raise ValueError(f"{capture_path}: no refresh_write_completions_elapsed_ms")

    origin = arrivals[0]["position"]
    origin_point = Point(float(origin["x"]), float(origin["y"]))
    first_evidence = None
    for sample in arrivals[1:]:
        first_evidence = course_from_position_chord(
            origin_point,
            Point(float(sample["position"]["x"]), float(sample["position"]["y"])),
            measured_at_s=float(sample["elapsed_ms"]) / 1000.0,
        )
        if first_evidence is not None:
            break
    if first_evidence is None:
        raise ValueError(f"{capture_path}: no informative >=0.15 m position chord")
    radians0 = math.radians(first_evidence.course_heading_degrees)
    target = {
        "x": origin["x"] + math.cos(radians0) * TARGET_OVERSHOOT_M,
        "y": origin["y"] + math.sin(radians0) * TARGET_OVERSHOOT_M,
    }

    observations = []
    previous_elapsed_ms = arrivals[0]["elapsed_ms"]
    previous_position = origin_point
    heading_anchor = origin_point
    heading_evidence = None
    cumulative_distance_m = 0.0
    for arrival in arrivals[1:]:
        position = arrival["position"]
        current = Point(float(position["x"]), float(position["y"]))
        cumulative_distance_m += math.hypot(
            current.x - previous_position.x, current.y - previous_position.y
        )
        previous_position = current
        candidate = course_from_position_chord(
            heading_anchor,
            current,
            measured_at_s=float(arrival["elapsed_ms"]) / 1000.0,
        )
        if candidate is not None:
            heading_evidence = candidate
            heading_anchor = current
        observations.append(
            {
                "position": {"x": position["x"], "y": position["y"]},
                "course_heading_degrees": (
                    heading_evidence.course_heading_degrees
                    if heading_evidence is not None
                    else None
                ),
                # Decisions are made AT each fresh arrival, so the position fix
                # backing this decision is fresh by construction.
                "telemetry_age_s": 0.0,
                "refresh_age_s": _refresh_age_s(arrival["elapsed_ms"], completions_ms),
                "refresh_max_gap_since_last_decision_s": _refresh_max_gap_s(
                    previous_elapsed_ms, arrival["elapsed_ms"], completions_ms
                ),
                "elapsed_s": arrival["elapsed_ms"] / 1000.0,
                "distance_travelled_m": cumulative_distance_m,
                "heading_evidence": (
                    asdict(heading_evidence) if heading_evidence is not None else None
                ),
            }
        )
        previous_elapsed_ms = arrival["elapsed_ms"]

    return {
        "route": {
            "start": {"x": origin["x"], "y": origin["y"]},
            "target": target,
            "contained": True,
        },
        "observations": observations,
    }


def replay(scenario: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Replay one scenario and return JSON-compatible step-by-step output."""
    route = ContinuousRoute(
        start=Point(**scenario["route"]["start"]),
        target=Point(**scenario["route"]["target"]),
        contained=scenario["route"]["contained"],
    )
    settings = ContinuousControllerConfig(**config)
    steps = []
    for index, raw_observation in enumerate(scenario["observations"], start=1):
        observation_data = dict(raw_observation)
        observation_data["position"] = Point(**observation_data["position"])
        if observation_data.get("heading_evidence") is not None:
            observation_data["heading_evidence"] = HeadingEvidence(
                **observation_data["heading_evidence"]
            )
        observation = ContinuousObservation(**observation_data)
        decision = continuous_control_decision(route, observation, settings)
        steps.append(
            {
                "index": index,
                "elapsed_s": observation.elapsed_s,
                "refresh_age_s": round(observation.refresh_age_s, 3),
                "refresh_max_gap_since_last_decision_s": round(
                    observation.refresh_max_gap_since_last_decision_s, 3
                ),
                "action": decision.action,
                "reason": decision.reason,
                "angular_speed": decision.angular_speed,
                "heading_error_degrees": (
                    round(decision.heading_error_degrees, 3)
                    if decision.heading_error_degrees is not None
                    else None
                ),
            }
        )
    return {
        "mode": "offline_continuous_controller_capture_replay",
        "dispatch_capable": False,
        "commands_sent": 0,
        "config": asdict(settings),
        "steps": steps,
        "first_stop": next((s for s in steps if s["action"] == "stop"), None),
    }


def main() -> int:
    """Replay a banked capture under both the old and reconciled constants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    scenario = build_scenario(args.capture)

    # max_window_s / max_distance_m / max_telemetry_age_s are widened past
    # anything this capture can reach, so ONLY the constant under test --
    # max_refresh_age_s -- can produce a stop. Without this, the default
    # max_window_s=4.0 masks whether refresh-age reconciliation would have
    # caught a stall LATER than 4 s, which is exactly what happened on the
    # first run of this script against the 8 s capture.
    isolate = {
        "max_window_s": 30.0,
        "max_distance_m": 30.0,
        "max_telemetry_age_s": 30.0,
        "max_cross_track_m": 30.0,
        "waypoint_tolerance_m": 0.001,
    }
    # OLD: the Phase 0 stub's provisional defaults. Notably has NO gap check at
    # all -- refresh_max_gap_since_last_decision_s defaults to 0.0 when the
    # caller does not supply it, so it can never trip regardless of
    # max_refresh_gap_s. That is deliberate: the point of this comparison is to
    # show that a point-sampled refresh_age_s check alone, even tightened, is
    # not what caught the real stall below.
    old_config = {
        **isolate,
        "nominal_speed_mps": 0.28,
        "max_refresh_age_s": 1.20,
        # The gap FIELD did not exist before today, so nothing could check it.
        # Setting the bound absurdly high is how "this check did not exist" is
        # represented in a dataclass that always carries the field now.
        "max_refresh_gap_s": 999.0,
    }
    # NEW: reconciled 2026-08-23, plus the gap detector this replay motivated.
    new_config = {
        **isolate,
        "nominal_speed_mps": 0.2482,
        "max_refresh_age_s": 0.60,
        "max_refresh_gap_s": 0.60,
    }

    old_result = replay(scenario, old_config)
    new_result = replay(scenario, new_config)

    print(f"=== {args.capture.name} ===")
    print(f"  {len(scenario['observations'])} decision points")
    print("  OLD (0.28 m/s, refresh_age<=1.20s, NO gap check):")
    print(f"    first stop: {old_result['first_stop']}")
    print("  NEW (0.2482 m/s, refresh_age<=0.60s, refresh_gap<=0.60s):")
    print(f"    first stop: {new_result['first_stop']}")
    for step in new_result["steps"]:
        flag = (
            f" <-- {step['reason']}"
            if step["reason"] in {"refresh_age_exceeded", "refresh_cadence_stalled"}
            else ""
        )
        print(
            f"    t={step['elapsed_s']:6.3f}s  age={step['refresh_age_s']:6.3f}s"
            f"  gap={step['refresh_max_gap_since_last_decision_s']:6.3f}s"
            f"  {step['action']:5s} {step['reason']}{flag}"
        )

    result = {"capture": str(args.capture), "old": old_result, "new": new_result}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
