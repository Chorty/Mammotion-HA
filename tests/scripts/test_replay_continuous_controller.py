"""Tests for the offline continuous-controller replay command."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.replay_continuous_controller import replay_scenario

ROOT = Path(__file__).resolve().parents[2]


def test_replay_is_explicitly_non_dispatching_and_stops_on_stale_data() -> None:
    """Replay labels itself non-dispatching and preserves fail-closed output."""

    scenario = {
        "route": {
            "start": {"x": 0.0, "y": 0.0},
            "target": {"x": 3.0, "y": 0.0},
            "contained": True,
        },
        "observations": [
            {
                "position": {"x": 0.0, "y": 0.0},
                "course_heading_degrees": 8.0,
                "telemetry_age_s": 0.0,
                "refresh_age_s": 0.0,
                "elapsed_s": 0.0,
                "distance_travelled_m": 0.0,
            },
            {
                "position": {"x": 0.3, "y": 0.03},
                "course_heading_degrees": 5.0,
                "telemetry_age_s": 2.1,
                "refresh_age_s": 0.2,
                "elapsed_s": 1.0,
                "distance_travelled_m": 0.3,
            },
        ],
    }

    output = replay_scenario(scenario)

    assert output["dispatch_capable"] is False
    assert output["commands_sent"] == 0
    assert output["steps"][0]["decision"]["action"] == "drive"
    assert output["steps"][1]["decision"]["reason"] == "telemetry_stale"
    assert output["steps"][1]["decision"]["linear_speed"] == 0


def test_default_cli_output_is_valid_non_dispatching_json() -> None:
    """The standalone command runs without importing the HA package."""

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/replay_continuous_controller.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(completed.stdout)

    assert output["mode"] == "offline_continuous_controller_replay"
    assert output["dispatch_capable"] is False
    assert output["commands_sent"] == 0
    assert output["steps"][-1]["decision"]["reason"] == "telemetry_stale"
