#!/usr/bin/env python3
"""Replay the pure continuous controller without Home Assistant or motion.

The output is diagnostic data only. This script has no HA client, token access,
service call, coordinator, BLE import, or dispatch path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = REPO / "custom_components" / "mammotion" / "continuous_controller.py"
_SPEC = importlib.util.spec_from_file_location(
    "mammotion_continuous_controller_offline", CONTROLLER_PATH
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
continuous_control_decision = _CONTROLLER.continuous_control_decision


def _default_scenario() -> dict[str, Any]:
    """Return a shallow-error replay followed by a deliberate stale frame."""

    return {
        "route": {
            "start": {"x": 0.0, "y": 0.0},
            "target": {"x": 3.0, "y": 0.0},
            "contained": True,
        },
        "observations": [
            {
                "position": {"x": 0.0, "y": 0.0},
                "course_heading_degrees": None,
                "telemetry_age_s": 0.0,
                "refresh_age_s": 0.0,
                "elapsed_s": 0.0,
                "distance_travelled_m": 0.0,
                "heading_evidence": None,
            },
            {
                "position": {"x": 0.27, "y": 0.04},
                "course_heading_degrees": 8.426969,
                "telemetry_age_s": 0.1,
                "refresh_age_s": 0.2,
                "elapsed_s": 1.0,
                "distance_travelled_m": 0.27,
                "heading_evidence": {
                    "course_heading_degrees": 8.426969,
                    "measured_at_s": 1.0,
                    "chord_m": 0.273,
                    "uncertainty_degrees": 0.92,
                },
            },
            {
                "position": {"x": 0.55, "y": 0.05},
                "course_heading_degrees": 2.045408,
                "telemetry_age_s": 0.1,
                "refresh_age_s": 0.2,
                "elapsed_s": 2.0,
                "distance_travelled_m": 0.55,
                "heading_evidence": {
                    "course_heading_degrees": 2.045408,
                    "measured_at_s": 2.0,
                    "chord_m": 0.280,
                    "uncertainty_degrees": 0.90,
                },
            },
            {
                "position": {"x": 0.82, "y": 0.03},
                "course_heading_degrees": -4.236395,
                "telemetry_age_s": 2.1,
                "refresh_age_s": 0.2,
                "elapsed_s": 3.0,
                "distance_travelled_m": 0.82,
                "heading_evidence": {
                    "course_heading_degrees": -4.236395,
                    "measured_at_s": 3.0,
                    "chord_m": 0.271,
                    "uncertainty_degrees": 0.93,
                },
            },
        ],
    }


def _point(raw: dict[str, Any]) -> Any:
    return Point(x=float(raw["x"]), y=float(raw["y"]))


def replay_scenario(raw: dict[str, Any]) -> dict[str, Any]:
    """Replay one JSON-compatible scenario and return JSON-compatible output."""

    route_raw = raw["route"]
    route = ContinuousRoute(
        start=_point(route_raw["start"]),
        target=_point(route_raw["target"]),
        contained=bool(route_raw.get("contained", True)),
    )
    config = ContinuousControllerConfig(**raw.get("config", {}))
    steps = []
    for index, observation_raw in enumerate(raw["observations"], start=1):
        observation_data = dict(observation_raw)
        observation_data["position"] = _point(observation_data["position"])
        if observation_data.get("heading_evidence") is not None:
            observation_data["heading_evidence"] = HeadingEvidence(
                **observation_data["heading_evidence"]
            )
        observation = ContinuousObservation(**observation_data)
        decision = continuous_control_decision(route, observation, config)
        steps.append(
            {
                "index": index,
                "observation": asdict(observation),
                "decision": asdict(decision),
            }
        )
    return {
        "mode": "offline_continuous_controller_replay",
        "dispatch_capable": False,
        "commands_sent": 0,
        "route": asdict(route),
        "config": asdict(config),
        "steps": steps,
    }


def main() -> int:
    """Load a scenario, replay it offline, and print diagnostics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="optional JSON scenario; defaults to a built-in synthetic replay",
    )
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    raw = json.loads(args.input.read_text()) if args.input else _default_scenario()
    print(json.dumps(replay_scenario(raw), indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
