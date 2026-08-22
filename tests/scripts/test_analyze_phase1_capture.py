"""Tests for the non-dispatching Phase 1 capture analyzer."""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.analyze_phase1_capture import analyze_phase1_pair

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyze_phase1_capture.py"


def _sample(
    index: int,
    elapsed_ms: float,
    x: float,
    y: float,
    toward: float,
) -> dict[str, Any]:
    return {
        "index": index,
        "elapsed_ms": elapsed_ms,
        "captured_at": f"2026-08-22T00:00:0{index}+00:00",
        "last_report_at_monotonic": 100.0 + elapsed_ms / 1000,
        "position": {
            "source": "mowing_state",
            "x": x,
            "y": y,
            "toward": toward,
            "pos_type": 1,
            "zone_hash": 123,
        },
        "vio": {"heading": 10.0 + index, "state": 2},
        "active_command": {
            "command": "send_movement",
            "kwargs": {"linear_speed": 400, "angular_speed": 0},
        },
    }


def _capture(*, arc: bool) -> dict[str, Any]:
    if arc:
        # Each course bearing plus the later sample's `toward` is 90 degrees.
        positions = (
            (0.0, 0.0, 90.0),
            (0.20, 0.02, 84.289407),
            (0.40, 0.06, 78.690068),
            (0.58, 0.13, 68.749494),
        )
    else:
        positions = (
            (0.0, 0.0, 90.0),
            (0.20, 0.0, 90.0),
            (0.40, 0.0, 90.0),
            (0.60, 0.0, 90.0),
        )
    samples = [
        _sample(index, elapsed, x, y, toward)
        for index, (elapsed, (x, y, toward)) in enumerate(
            zip((0.0, 900.0, 1900.0, 3000.0), positions, strict=True)
        )
    ]
    angular_speed = 180 if arc else 0
    for sample in samples:
        sample["active_command"]["kwargs"]["angular_speed"] = angular_speed
    completions = [float(index * 200) for index in range(1, 20)]
    return {
        "service": "raw_pymammotion_motion_probe",
        "mode": "real_raw_pymammotion_probe",
        "dry_run": False,
        "reason": "completed",
        "would_send": True,
        "command": "send_movement",
        "command_args": {
            "linear_speed": 400,
            "angular_speed": angular_speed,
        },
        "motion_axes": "arc" if arc else "linear",
        "motion_refresh_interval_ms": 200,
        "duration_ms": 4000,
        "command_result": {
            "attempted": True,
            "ok": True,
            "ack": None,
            "error": None,
        },
        "in_window_telemetry": {
            "enabled": True,
            "sample_interval_ms": 100,
            "source": "coordinator_cache_only",
            "extra_ble_report_requests_during_window": 0,
            "report_stream": {
                "attempted": True,
                "started": True,
                "continuous_started": True,
                "error": None,
                "queue_settle": {"live": True, "reason": "live"},
            },
            "samples": samples,
        },
        "motion_refresh": {
            "refresh_enabled": True,
            "refresh_interval_ms": 200,
            "refresh_commands_sent": 19,
            "refresh_write_completions_elapsed_ms": completions,
            "refresh_write_durations_ms": [10.0] * 19,
            "elapsed_ms": 4000.0,
        },
        "stop_result": {"attempted": True, "ok": True, "error": None},
    }


def _corridor() -> dict[str, Any]:
    return {
        "prevalidated": True,
        "area_margin_m": 1.2,
        "keepout_margin_m": 1.5,
        "frozen_start": {"x": 0.0, "y": 0.0},
        "frozen_endpoint": {"x": 1.0, "y": 0.0},
        "polygon": [
            {"x": -0.2, "y": -0.2},
            {"x": 1.2, "y": -0.2},
            {"x": 1.2, "y": 0.5},
            {"x": -0.2, "y": 0.5},
        ],
    }


def _pair() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    corridor = _corridor()
    return (
        _capture(arc=False),
        _capture(arc=True),
        {"straight": corridor, "shallow_arc": copy.deepcopy(corridor)},
    )


def test_passing_pair_returns_go_without_dispatch_capability() -> None:
    """All written Phase 1 criteria produce a scoped, offline go verdict."""
    straight, arc, corridors = _pair()

    result = analyze_phase1_pair(straight, arc, corridors)

    assert result["verdict"] == "go"
    assert result["failed_criteria"] == []
    assert result["dispatch_capable"] is False
    assert result["network_access"] is False
    assert result["commands_sent"] == 0
    assert result["physical_run_authorized"] is False
    assert result["captures"]["straight"]["passed"] is True
    assert result["captures"]["shallow_arc"]["passed"] is True
    assert (
        result["captures"]["straight"]["position_diagnostics"][
            "fresh_position_arrival_count"
        ]
        == 3
    )
    assert (
        result["captures"]["straight"]["position_diagnostics"][
            "max_position_arrival_gap_ms"
        ]
        == 1100.0
    )
    assert (
        result["captures"]["shallow_arc"]["position_diagnostics"][
            "toward_changed_before_stop"
        ]
        is True
    )


def test_failures_are_named_and_pair_fails_closed() -> None:
    """Stop, refresh, cadence, toward, and corridor failures remain distinct."""
    straight, arc, corridors = _pair()
    arc["stop_result"]["ok"] = False
    arc["motion_refresh"]["refresh_error"] = "RuntimeError: write failed"
    arc["in_window_telemetry"]["samples"] = [
        _sample(0, 0.0, 0.0, 0.0, 90.0),
        _sample(1, 2500.0, 0.2, 0.0, 90.0),
        _sample(2, 3000.0, 2.0, 0.0, 90.0),
    ]

    result = analyze_phase1_pair(straight, arc, corridors)

    assert result["verdict"] == "no_go"
    failed = result["captures"]["shallow_arc"]["failed_criteria"]
    assert "refresh_writes_confirmed" in failed
    assert "stop_confirmed" in failed
    assert "fresh_position_arrivals" in failed
    assert "maximum_position_arrival_gap" in failed
    assert "bearing_toward_compass_mirror" in failed
    assert "observed_positions_inside_corridor" in failed
    assert "toward_changed_before_stop" in failed


def test_missing_corridor_metadata_is_no_go_not_an_exception() -> None:
    """A capture without frozen containment evidence cannot pass by omission."""
    straight, arc, _corridors = _pair()

    result = analyze_phase1_pair(straight, arc, {})

    assert result["verdict"] == "no_go"
    assert "straight.corridor_prevalidated" in result["failed_criteria"]
    assert "straight.frozen_route_present" in result["failed_criteria"]
    assert "straight.corridor_polygon_valid" in result["failed_criteria"]
    assert "shallow_arc.observed_positions_inside_corridor" in result["failed_criteria"]


def test_malformed_samples_and_rest_wrapper_fail_closed() -> None:
    """REST wrapping is accepted, while malformed samples remain visible."""
    straight, arc, corridors = _pair()
    straight["in_window_telemetry"]["samples"].append(
        {"elapsed_ms": "nan", "position": {"x": 1.0}}
    )

    result = analyze_phase1_pair(
        {"service_response": straight},
        {"service_response": arc},
        corridors,
    )

    assert result["verdict"] == "no_go"
    assert "straight.samples_valid" in result["failed_criteria"]


def test_post_window_samples_do_not_earn_arrival_credit() -> None:
    """Positions arriving after four seconds cannot satisfy in-window cadence."""
    straight, arc, corridors = _pair()
    straight["in_window_telemetry"]["samples"] = [
        _sample(0, 0.0, 0.0, 0.0, 90.0),
        _sample(1, 4100.0, 0.2, 0.0, 90.0),
        _sample(2, 4200.0, 0.4, 0.0, 90.0),
        _sample(3, 4300.0, 0.6, 0.0, 90.0),
    ]

    result = analyze_phase1_pair(straight, arc, corridors)

    diagnostics = result["captures"]["straight"]["position_diagnostics"]
    assert diagnostics["sample_count"] == 1
    assert diagnostics["samples_after_window_count"] == 3
    assert "straight.fresh_position_arrivals" in result["failed_criteria"]


def test_non_cache_only_instrumentation_fails_control_profile() -> None:
    """Phase 1 must not hide extra BLE work inside the measurement window."""
    straight, arc, corridors = _pair()
    straight["in_window_telemetry"]["source"] = "direct_ble"
    straight["in_window_telemetry"]["extra_ble_report_requests_during_window"] = 1

    result = analyze_phase1_pair(straight, arc, corridors)

    assert "straight.control_profile" in result["failed_criteria"]


def test_cli_writes_bankable_json_and_returns_success_for_go(
    tmp_path: Path,
) -> None:
    """The standalone CLI reads files only and writes the same scoped verdict."""
    straight, arc, corridors = _pair()
    straight_path = tmp_path / "straight.json"
    arc_path = tmp_path / "arc.json"
    corridors_path = tmp_path / "corridors.json"
    output_path = tmp_path / "analysis.json"
    straight_path.write_text(json.dumps({"service_response": straight}))
    arc_path.write_text(json.dumps(arc))
    corridors_path.write_text(json.dumps(corridors))

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--straight",
            str(straight_path),
            "--arc",
            str(arc_path),
            "--corridors",
            str(corridors_path),
            "--output",
            str(output_path),
            "--compact",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = json.loads(completed.stdout)
    banked = json.loads(output_path.read_text())
    assert stdout == banked
    assert banked["verdict"] == "go"
    assert banked["commands_sent"] == 0
    assert banked["physical_run_authorized"] is False
    assert set(banked["inputs"]) == {"straight", "shallow_arc", "corridors"}
    assert len(banked["inputs"]["straight"]["sha256"]) == 64


def test_script_imports_only_standard_library_modules() -> None:
    """Keep the analyzer structurally disconnected from runtime/dispatch code."""
    tree = ast.parse(SCRIPT.read_text())
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert imported <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "typing",
    }
