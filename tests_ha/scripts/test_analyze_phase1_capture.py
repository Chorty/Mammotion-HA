"""Tests for the non-dispatching Phase 1 capture analyzer."""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.analyze_phase1_capture import (
    EXPECTED_CONTROLS,
    MAX_COMPASS_MIRROR_ERROR_DEGREES,
    MIN_MOVING_STEP_M,
    _position_diagnostics,
    _valid_samples,
    analyze_phase1_pair,
)

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
    """Build a passing capture at the duration its control actually requires.

    ⚠️ The arc runs **8000 ms** (`docs/phase1b-arc-protocol-20260823.md`). At 4 s
    it cannot reliably clear the repaired criterion's 3-informative-chord bar
    once the short spin-up chord is dropped, which is exactly how the
    2026-08-22 arc failed. Only the duration differs; the command is unchanged.
    """
    if arc:
        # Each chord's bearing plus the `toward` at the START of that interval
        # is 90 degrees -- the repaired START pairing, not the end.
        positions = (
            (0.0, 0.0, 90.0),
            (0.2, 0.0, 84.0),
            (0.3989, 0.0209, 78.0),
            (0.5945, 0.0625, 72.0),
            (0.7847, 0.1243, 66.0),
            (0.9675, 0.2056, 60.0),
            (1.1407, 0.3056, 54.0),
        )
        elapsed_ms = (0.0, 900.0, 1900.0, 3000.0, 4100.0, 5200.0, 6300.0)
    else:
        positions = (
            (0.0, 0.0, 90.0),
            (0.20, 0.0, 90.0),
            (0.40, 0.0, 90.0),
            (0.60, 0.0, 90.0),
        )
        elapsed_ms = (0.0, 900.0, 1900.0, 3000.0)
    duration_ms = 8000 if arc else 4000
    refresh_count = duration_ms // 200 - 1
    samples = [
        _sample(index, elapsed, x, y, toward)
        for index, (elapsed, (x, y, toward)) in enumerate(
            zip(elapsed_ms, positions, strict=True)
        )
    ]
    angular_speed = 180 if arc else 0
    for sample in samples:
        sample["active_command"]["kwargs"]["angular_speed"] = angular_speed
    completions = [float(index * 200) for index in range(1, refresh_count + 1)]
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
        "duration_ms": duration_ms,
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
            "refresh_commands_sent": refresh_count,
            "refresh_write_completions_elapsed_ms": completions,
            "refresh_write_durations_ms": [10.0] * refresh_count,
            "elapsed_ms": float(duration_ms),
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
            {"x": 1.4, "y": -0.2},
            {"x": 1.4, "y": 0.6},
            {"x": -0.2, "y": 0.6},
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


# --- the 2026-08-23 repair to bearing_toward_compass_mirror -------------------


def test_the_mirror_check_pairs_toward_at_the_START_of_the_interval() -> None:
    """The pairing decides the verdict, so it must be pinned.

    `bearing` is a chord between fixes ~1 s apart, an interval AVERAGE; `toward`
    is one instant. On a body rotating ~10 deg per interval the two ends differ
    by the whole rotation -- the 2026-08-22 arc scored 2.5 deg paired at the
    start and 12.6 deg paired at the end, and the end pairing is what produced
    that day's `no_go`.

    The start is the only heading a controller HAS when predicting the chord it
    is about to travel.
    """
    # A body rotating 20 deg per interval whose every chord follows the heading
    # it held at the START of that interval. Under START pairing each step reads
    # 0 deg; under END pairing each reads the full 20 deg of rotation.
    samples = [
        _sample(0, 0.0, 0.0, 0.0, 90.0),
        _sample(1, 1000.0, 1.0, 0.0, 110.0),
        _sample(2, 2000.0, 1.939693, -0.34202, 130.0),
        _sample(3, 3000.0, 2.705737, -0.984808, 150.0),
    ]
    diagnostics = _position_diagnostics(_valid_samples(samples)[0], duration_ms=4000)
    steps = diagnostics["moving_steps"]
    assert len(steps) == 3

    for step in steps:
        assert abs(step["bearing_plus_toward_error_degrees"]) < 0.001
    assert diagnostics["max_bearing_plus_toward_error_degrees"] < 0.001


def test_steps_too_short_to_carry_a_bearing_are_not_scored() -> None:
    """A step whose noise bound exceeds the threshold cannot test anything.

    The floor was 0.01 m, three orders below the position noise floor, so it
    excluded only exactly-zero steps. At the measured sigma = 0.0031 m a 0.076 m
    chord carries +-7.4 deg of bearing uncertainty against a 10 deg threshold.
    """
    assert MIN_MOVING_STEP_M >= 0.15

    samples = [
        _sample(0, 0.0, 0.0, 0.0, 90.0),
        # 5 cm: noise-dominated, must be ignored however badly it scores.
        _sample(1, 1000.0, 0.05, 0.0, 90.0),
        _sample(2, 2000.0, 1.05, 0.0, 90.0),
        _sample(3, 3000.0, 2.05, 0.0, 90.0),
    ]
    diagnostics = _position_diagnostics(_valid_samples(samples)[0], duration_ms=4000)
    distances = [step["distance_m"] for step in diagnostics["moving_steps"]]
    assert all(d >= MIN_MOVING_STEP_M for d in distances)
    assert len(distances) == 2


def test_the_banked_arc_now_fails_on_STEP_COUNT_not_on_error() -> None:
    """The repair changes the failure mode, and does NOT manufacture a pass.

    Paired at the interval start the 2026-08-22 arc's mirror error is 2.52 deg
    against a 10 deg threshold -- fine. But a correctly sized minimum chord
    leaves it only 2 informative steps against the 3 required, because a 4 s
    window at ~1 Hz yields 3 arrivals and one of them is too short.

    That is a defect in the CAPTURE DESIGN, not in the mower, and it must not be
    fixed by lowering the required step count.
    """
    arc = json.loads(
        (
            ROOT / "docs" / "evidence-phase1-shallow-arc-20260822T203400Z.json"
        ).read_text()
    )
    straight = json.loads(
        (ROOT / "docs" / "evidence-phase1-straight-20260822T202600Z.json").read_text()
    )
    corridors = json.loads(
        (ROOT / "docs" / "evidence-phase1-corridors-20260822T203400Z.json").read_text()
    )
    result = analyze_phase1_pair(straight, arc, corridors)

    assert result["verdict"] == "no_go"
    mirror = next(
        c
        for c in result["captures"]["shallow_arc"]["criteria"]
        if c["name"] == "bearing_toward_compass_mirror"
    )
    assert mirror["passed"] is False
    # The ERROR is now comfortably inside the threshold...
    assert mirror["observed"]["max_error_degrees"] <= MAX_COMPASS_MIRROR_ERROR_DEGREES
    # ...and only the step count is short.
    assert mirror["observed"]["moving_step_count"] == 2


def test_duration_is_fixed_PER_CONTROL_and_is_not_a_menu() -> None:
    """The arc runs longer than the straight run, and each is a single value.

    🚨 The anti-gaming rule from `docs/phase1b-arc-protocol-20260823.md`: a
    capture must match ONE duration for its control. If this ever became a list
    or range of accepted durations, a capture that failed at one length could be
    re-scored at another, which is the whole failure mode the Phase 1b
    registration exists to prevent.
    """
    assert EXPECTED_CONTROLS["straight"]["duration_ms"] == 4000
    assert EXPECTED_CONTROLS["shallow_arc"]["duration_ms"] == 8000
    for spec in EXPECTED_CONTROLS.values():
        assert isinstance(spec["duration_ms"], int), "one value, never a menu"

    # The command itself is unchanged from the original plan -- only duration.
    assert EXPECTED_CONTROLS["shallow_arc"]["command_args"] == {
        "linear_speed": 400,
        "angular_speed": 180,
    }


def test_a_capture_at_the_wrong_duration_is_refused() -> None:
    """An 8 s arc is required; a 4 s one fails the control profile."""
    straight, arc, corridors = _pair()
    arc["duration_ms"] = 4000
    result = analyze_phase1_pair(straight, arc, corridors)

    assert result["verdict"] == "no_go"
    profile = next(
        c
        for c in result["captures"]["shallow_arc"]["criteria"]
        if c["name"] == "control_profile"
    )
    assert profile["passed"] is False
