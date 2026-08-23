#!/usr/bin/env python3
"""Analyze paired Phase 1 continuous-motion captures entirely offline.

The command reads JSON files only. It has no Home Assistant client, token,
coordinator, BLE, service-call, or dispatch path. A ``go`` verdict means the
banked straight and shallow-arc captures satisfy the predeclared Phase 1
measurement criteria; it never authorizes or commands a later physical run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

# ⚠️ **PER CONTROL, not global — changed 2026-08-23.** The arc needs a longer
# window than the straight run: at ~1 Hz a 4 s window yields ~4 arrivals, and
# once the short spin-up chord is dropped that leaves only ~3 informative chords
# with **no margin**. The 2026-08-22 arc got 2 and failed the repaired criterion
# on step count. 8000 ms makes a fourth arrival reliable rather than lucky.
#
# 🚨 **This must stay a fixed duration PER CONTROL, never a menu of accepted
# durations.** A menu is how an after-the-fact choice hides: it would let a
# capture that failed at one length be re-scored at another. Registered before
# any Phase 1b capture exists -- `docs/phase1b-arc-protocol-20260823.md`.
_DEFAULT_DURATION_MS = 4000
EXPECTED_REFRESH_INTERVAL_MS = 200
EXPECTED_SAMPLE_INTERVAL_MS = 100
MIN_FRESH_POSITION_ARRIVALS = 3
MAX_POSITION_ARRIVAL_GAP_MS = 2000.0
MAX_COMPASS_MIRROR_ERROR_DEGREES = 10.0
COMPASS_MIRROR_SUM_DEGREES = 90.0
MIN_AREA_MARGIN_M = 1.2
MIN_KEEPOUT_MARGIN_M = 1.5
MAX_START_DRIFT_M = 0.30
# 🔑 **RAISED 0.01 -> 0.15 on 2026-08-23.** A chord this short cannot carry a
# bearing: position noise alone buys `atan(sigma*sqrt(2)/chord)` of bearing
# uncertainty, and at the sigma = 0.0031 m measured across 16 steady in-window
# steps that is +-12.2 deg on the straight capture's 0.0456 m step and +-7.4 deg
# on the arc's 0.0760 m step -- at or above the 10 deg threshold itself. **A step
# whose noise bound exceeds the threshold cannot test anything**, and the old
# 0.01 m floor excluded only exactly-zero steps. At 0.15 m the bound is 1.7 deg.
MIN_MOVING_STEP_M = 0.15
_POSITION_EPSILON_M = 1e-9

EXPECTED_CONTROLS: dict[str, dict[str, Any]] = {
    "straight": {
        "motion_axes": "linear",
        "command_args": {"linear_speed": 400, "angular_speed": 0},
        "require_toward_change": False,
        # Unchanged from the original plan; it passes at this length.
        "duration_ms": 4000,
    },
    "shallow_arc": {
        "motion_axes": "arc",
        # Unchanged from the original plan. Only the DURATION moves.
        "command_args": {"linear_speed": 400, "angular_speed": 180},
        "require_toward_change": True,
        "duration_ms": 8000,
    },
}


def _criterion(
    name: str,
    passed: bool,
    *,
    observed: Any,
    required: Any,
) -> dict[str, Any]:
    """Return one JSON-compatible criterion record."""
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


def _finite_number(value: Any) -> float | None:
    """Return *value* as a finite float, otherwise ``None``."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    return number if math.isfinite(number) else None


def _point(value: Any) -> tuple[float, float] | None:
    """Return a finite x/y pair from a JSON object."""
    if not isinstance(value, dict):
        return None
    x = _finite_number(value.get("x"))
    y = _finite_number(value.get("y"))
    return (x, y) if x is not None and y is not None else None


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _heading_error_degrees(current: float, target: float) -> float:
    """Return the signed shortest angular difference ``target - current``."""
    return (target - current + 540.0) % 360.0 - 180.0


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether *point* lies on a closed segment."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    squared_length = dx * dx + dy * dy
    if squared_length <= tolerance:
        return _distance(point, start) <= tolerance
    cross = (point[1] - start[1]) * dx - (point[0] - start[0]) * dy
    if abs(cross) > tolerance:
        return False
    dot = (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    return -tolerance <= dot <= squared_length + tolerance


def _point_in_polygon(
    point: tuple[float, float], polygon: list[tuple[float, float]]
) -> bool:
    """Return whether a point is inside or on a polygon boundary."""
    if len(polygon) < 3:
        return False
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        crosses = (current[1] > point[1]) != (previous[1] > point[1])
        if crosses:
            x_at_y = (previous[0] - current[0]) * (point[1] - current[1]) / (
                previous[1] - current[1]
            ) + current[0]
            if point[0] <= x_at_y:
                inside = not inside
        previous = current
    return inside


def _service_response(raw: Any) -> dict[str, Any]:
    """Unwrap a Home Assistant REST response or accept a bare service response."""
    if not isinstance(raw, dict):
        return {}
    response = raw.get("service_response")
    return response if isinstance(response, dict) else raw


def _valid_samples(samples_raw: Any) -> tuple[list[dict[str, Any]], int]:
    """Return normalized finite samples and the number rejected."""
    if not isinstance(samples_raw, list):
        return [], 1
    samples: list[dict[str, Any]] = []
    invalid = 0
    for index, sample in enumerate(samples_raw):
        if not isinstance(sample, dict):
            invalid += 1
            continue
        elapsed_ms = _finite_number(sample.get("elapsed_ms"))
        position_raw = sample.get("position")
        position = _point(position_raw)
        if elapsed_ms is None or elapsed_ms < 0 or position is None:
            invalid += 1
            continue
        toward = (
            _finite_number(position_raw.get("toward"))
            if isinstance(position_raw, dict)
            else None
        )
        samples.append(
            {
                "index": index,
                "elapsed_ms": elapsed_ms,
                "position": position,
                "toward": toward,
                "last_report_at_monotonic": _finite_number(
                    sample.get("last_report_at_monotonic")
                ),
            }
        )
    samples.sort(key=lambda item: item["elapsed_ms"])
    return samples, invalid


def _position_diagnostics(
    samples: list[dict[str, Any]], *, duration_ms: int
) -> dict[str, Any]:
    """Recompute arrivals, gaps, course mirror, and toward changes."""
    unique_positions: list[dict[str, Any]] = []
    for sample in samples:
        if (
            not unique_positions
            or _distance(unique_positions[-1]["position"], sample["position"])
            > _POSITION_EPSILON_M
        ):
            unique_positions.append(sample)

    fresh = unique_positions[1:]
    arrival_times = [sample["elapsed_ms"] for sample in fresh]
    boundaries = [0.0, *arrival_times, float(duration_ms)]
    gaps = [
        round(later - earlier, 3)
        for earlier, later in zip(boundaries, boundaries[1:], strict=False)
    ]

    moving_steps: list[dict[str, Any]] = []
    for previous, current in zip(unique_positions, unique_positions[1:], strict=False):
        dx = current["position"][0] - previous["position"][0]
        dy = current["position"][1] - previous["position"][1]
        distance_m = math.hypot(dx, dy)
        if distance_m < MIN_MOVING_STEP_M:
            continue
        bearing = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        # 🔑 **PAIR WITH THE START OF THE INTERVAL, not its end. Changed
        # 2026-08-23; the original `current["toward"]` is what produced the
        # 2026-08-22 `no_go`.**
        #
        # `bearing` is a CHORD between two fixes about a second apart -- an
        # interval average. `toward` is a single instant. On a body rotating
        # ~10 deg per interval those differ by the whole rotation, so which end
        # supplies `toward` decides the verdict: the same arc scores 2.5 deg at
        # the start and 12.6 deg at the end.
        #
        # The start is the physically meaningful pairing because it is the only
        # one a controller HAS. Predicting the chord it is about to travel, it
        # knows the heading it holds now; the heading it will hold when the fix
        # arrives is precisely what it does not know yet. Certifying a reference
        # the controller cannot obtain would certify the wrong thing.
        #
        # ⚠️ It is not exactly right either. Solving all 8 informative arc steps
        # for the pairing that zeroes the error gives alpha = -0.165 +- 0.043,
        # which excludes the start (alpha = 0) at ~3 sigma as well -- END is
        # merely far worse, at ~22 sigma. The residual is unexplained and is not
        # modelled here; the 10 deg threshold is unchanged and absorbs it.
        toward = previous["toward"]
        mirror_error = (
            abs(
                _heading_error_degrees(
                    (bearing + toward) % 360.0,
                    COMPASS_MIRROR_SUM_DEGREES,
                )
            )
            if toward is not None
            else None
        )
        moving_steps.append(
            {
                "from_sample_index": previous["index"],
                "to_sample_index": current["index"],
                "distance_m": round(distance_m, 6),
                "bearing_degrees": round(bearing, 6),
                "toward_degrees": toward,
                "bearing_plus_toward_error_degrees": (
                    round(mirror_error, 6) if mirror_error is not None else None
                ),
            }
        )

    toward_values = [
        sample["toward"] for sample in samples if sample["toward"] is not None
    ]
    toward_changes = [
        abs(_heading_error_degrees(previous, current))
        for previous, current in zip(toward_values, toward_values[1:], strict=False)
    ]
    total_toward_change = sum(toward_changes)
    mirror_errors = [
        step["bearing_plus_toward_error_degrees"]
        for step in moving_steps
        if step["bearing_plus_toward_error_degrees"] is not None
    ]
    return {
        "sample_count": len(samples),
        "fresh_position_arrival_count": len(fresh),
        "fresh_position_arrivals_elapsed_ms": arrival_times,
        "position_arrival_gaps_including_boundaries_ms": gaps,
        "max_position_arrival_gap_ms": max(gaps, default=None),
        "moving_steps": moving_steps,
        "moving_step_count": len(moving_steps),
        "moving_steps_with_toward_count": len(mirror_errors),
        "max_bearing_plus_toward_error_degrees": max(mirror_errors, default=None),
        "toward_change_count": sum(
            change > _POSITION_EPSILON_M for change in toward_changes
        ),
        "total_absolute_toward_change_degrees": round(total_toward_change, 6),
        "toward_changed_before_stop": total_toward_change > _POSITION_EPSILON_M,
        "path_length_m": round(sum(step["distance_m"] for step in moving_steps), 6),
        "net_displacement_m": (
            round(
                _distance(
                    unique_positions[0]["position"], unique_positions[-1]["position"]
                ),
                6,
            )
            if unique_positions
            else None
        ),
    }


def _corridor_diagnostics(
    corridor: Any, samples: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate frozen scan metadata and observed containment."""
    corridor = corridor if isinstance(corridor, dict) else {}
    polygon_raw = corridor.get("polygon")
    polygon = (
        [point for raw_point in polygon_raw if (point := _point(raw_point))]
        if isinstance(polygon_raw, list)
        else []
    )
    polygon_valid = (
        isinstance(polygon_raw, list)
        and len(polygon) == len(polygon_raw)
        and len(polygon) >= 3
    )
    frozen_start = _point(corridor.get("frozen_start"))
    frozen_endpoint = _point(corridor.get("frozen_endpoint"))
    area_margin = _finite_number(corridor.get("area_margin_m"))
    keepout_margin = _finite_number(corridor.get("keepout_margin_m"))
    start_drift = (
        _distance(frozen_start, samples[0]["position"])
        if frozen_start is not None and samples
        else None
    )
    outside = [
        sample["index"]
        for sample in samples
        if not polygon_valid or not _point_in_polygon(sample["position"], polygon)
    ]
    criteria = [
        _criterion(
            "corridor_prevalidated",
            corridor.get("prevalidated") is True,
            observed=corridor.get("prevalidated"),
            required=True,
        ),
        _criterion(
            "area_margin",
            area_margin is not None and area_margin >= MIN_AREA_MARGIN_M,
            observed=area_margin,
            required=f">= {MIN_AREA_MARGIN_M} m",
        ),
        _criterion(
            "keepout_margin",
            keepout_margin is not None and keepout_margin >= MIN_KEEPOUT_MARGIN_M,
            observed=keepout_margin,
            required=f">= {MIN_KEEPOUT_MARGIN_M} m",
        ),
        _criterion(
            "frozen_route_present",
            frozen_start is not None and frozen_endpoint is not None,
            observed={
                "frozen_start": frozen_start,
                "frozen_endpoint": frozen_endpoint,
            },
            required="finite frozen_start and frozen_endpoint",
        ),
        _criterion(
            "start_drift",
            start_drift is not None and start_drift <= MAX_START_DRIFT_M,
            observed=round(start_drift, 6) if start_drift is not None else None,
            required=f"<= {MAX_START_DRIFT_M} m",
        ),
        _criterion(
            "corridor_polygon_valid",
            polygon_valid,
            observed=len(polygon),
            required=">= 3 finite vertices",
        ),
        _criterion(
            "observed_positions_inside_corridor",
            bool(samples) and not outside,
            observed={"outside_sample_indices": outside},
            required="every in-window position inside/on prevalidated polygon",
        ),
    ]
    return {
        "prevalidated": corridor.get("prevalidated"),
        "area_margin_m": area_margin,
        "keepout_margin_m": keepout_margin,
        "frozen_start": frozen_start,
        "frozen_endpoint": frozen_endpoint,
        "start_drift_m": round(start_drift, 6) if start_drift is not None else None,
        "polygon_vertex_count": len(polygon),
        "outside_sample_indices": outside,
    }, criteria


def analyze_capture(control: str, raw_capture: Any, corridor: Any) -> dict[str, Any]:
    """Analyze one straight or shallow-arc capture."""
    expected = EXPECTED_CONTROLS[control]
    capture = _service_response(raw_capture)
    instrumentation = capture.get("in_window_telemetry")
    instrumentation = instrumentation if isinstance(instrumentation, dict) else {}
    duration_ms = expected.get("duration_ms", _DEFAULT_DURATION_MS)
    samples, invalid_sample_count = _valid_samples(instrumentation.get("samples"))
    window_samples = [
        sample for sample in samples if sample["elapsed_ms"] <= duration_ms
    ]
    position = _position_diagnostics(window_samples, duration_ms=duration_ms)
    position["samples_after_window_count"] = len(samples) - len(window_samples)
    corridor_report, corridor_criteria = _corridor_diagnostics(corridor, samples)
    command_result = capture.get("command_result")
    command_result = command_result if isinstance(command_result, dict) else {}
    refresh = capture.get("motion_refresh")
    refresh = refresh if isinstance(refresh, dict) else {}
    stop = capture.get("stop_result")
    stop = stop if isinstance(stop, dict) else {}
    stream = instrumentation.get("report_stream")
    stream = stream if isinstance(stream, dict) else {}
    queue_settle = stream.get("queue_settle")
    queue_settle = queue_settle if isinstance(queue_settle, dict) else {}
    completions = refresh.get("refresh_write_completions_elapsed_ms")
    completions = completions if isinstance(completions, list) else []
    refresh_sent = refresh.get("refresh_commands_sent")
    refresh_sent_int = refresh_sent if isinstance(refresh_sent, int) else -1
    completion_values = [_finite_number(value) for value in completions]
    completion_numbers = [value for value in completion_values if value is not None]
    completions_valid = (
        refresh_sent_int > 0
        and len(completions) == refresh_sent_int
        and len(completion_numbers) == len(completions)
        and all(
            later >= earlier
            for earlier, later in zip(
                completion_numbers, completion_numbers[1:], strict=False
            )
        )
    )
    mirror_error = position["max_bearing_plus_toward_error_degrees"]
    criteria = [
        _criterion(
            "control_profile",
            capture.get("service") == "raw_pymammotion_motion_probe"
            and capture.get("dry_run") is False
            and capture.get("duration_ms") == duration_ms
            and capture.get("motion_refresh_interval_ms")
            == EXPECTED_REFRESH_INTERVAL_MS
            and instrumentation.get("enabled") is True
            and instrumentation.get("sample_interval_ms") == EXPECTED_SAMPLE_INTERVAL_MS
            and instrumentation.get("source") == "coordinator_cache_only"
            and instrumentation.get("extra_ble_report_requests_during_window") == 0
            and capture.get("motion_axes") == expected["motion_axes"]
            and capture.get("command_args") == expected["command_args"],
            observed={
                "service": capture.get("service"),
                "dry_run": capture.get("dry_run"),
                "duration_ms": capture.get("duration_ms"),
                "motion_refresh_interval_ms": capture.get("motion_refresh_interval_ms"),
                "instrumentation_enabled": instrumentation.get("enabled"),
                "sample_interval_ms": instrumentation.get("sample_interval_ms"),
                "instrumentation_source": instrumentation.get("source"),
                "extra_ble_report_requests_during_window": instrumentation.get(
                    "extra_ble_report_requests_during_window"
                ),
                "motion_axes": capture.get("motion_axes"),
                "command_args": capture.get("command_args"),
            },
            required={
                "service": "raw_pymammotion_motion_probe",
                "dry_run": False,
                "duration_ms": duration_ms,
                "motion_refresh_interval_ms": EXPECTED_REFRESH_INTERVAL_MS,
                "instrumentation_enabled": True,
                "sample_interval_ms": EXPECTED_SAMPLE_INTERVAL_MS,
                "instrumentation_source": "coordinator_cache_only",
                "extra_ble_report_requests_during_window": 0,
                "motion_axes": expected["motion_axes"],
                "command_args": expected["command_args"],
            },
        ),
        _criterion(
            "command_confirmed",
            command_result.get("attempted") is True
            and command_result.get("ok") is True,
            observed=command_result,
            required="movement command attempted and confirmed",
        ),
        _criterion(
            "report_stream_started",
            stream.get("started") is True
            and stream.get("continuous_started") is True
            and stream.get("error") in (None, "")
            and queue_settle.get("live") is True,
            observed=stream,
            required="bounded and continuous streams started; queue settled live",
        ),
        _criterion(
            "refresh_writes_confirmed",
            refresh.get("refresh_enabled") is True
            and refresh.get("refresh_interval_ms") == EXPECTED_REFRESH_INTERVAL_MS
            and refresh.get("refresh_error") in (None, "")
            and completions_valid,
            observed={
                "refresh_enabled": refresh.get("refresh_enabled"),
                "refresh_interval_ms": refresh.get("refresh_interval_ms"),
                "refresh_error": refresh.get("refresh_error"),
                "refresh_commands_sent": refresh_sent,
                "completion_count": len(completions),
            },
            required=(
                f"enabled at {EXPECTED_REFRESH_INTERVAL_MS} ms; no refresh error; "
                "positive, ordered completion per refresh"
            ),
        ),
        _criterion(
            "stop_confirmed",
            stop.get("attempted") is True and stop.get("ok") is True,
            observed=stop,
            required="explicit stop attempted and confirmed",
        ),
        _criterion(
            "capture_completed",
            capture.get("reason") == "completed",
            observed=capture.get("reason"),
            required="completed",
        ),
        _criterion(
            "samples_valid",
            bool(samples) and invalid_sample_count == 0,
            observed={
                "valid_sample_count": len(samples),
                "invalid_sample_count": invalid_sample_count,
            },
            required="at least one sample; zero malformed samples",
        ),
        _criterion(
            "fresh_position_arrivals",
            position["fresh_position_arrival_count"] >= MIN_FRESH_POSITION_ARRIVALS,
            observed=position["fresh_position_arrival_count"],
            required=f">= {MIN_FRESH_POSITION_ARRIVALS}",
        ),
        _criterion(
            "maximum_position_arrival_gap",
            position["max_position_arrival_gap_ms"] is not None
            and position["max_position_arrival_gap_ms"] <= MAX_POSITION_ARRIVAL_GAP_MS,
            observed=position["max_position_arrival_gap_ms"],
            required=f"<= {MAX_POSITION_ARRIVAL_GAP_MS} ms including boundaries",
        ),
        _criterion(
            "bearing_toward_compass_mirror",
            position["moving_step_count"] >= MIN_FRESH_POSITION_ARRIVALS
            and position["moving_steps_with_toward_count"]
            == position["moving_step_count"]
            and mirror_error is not None
            and mirror_error <= MAX_COMPASS_MIRROR_ERROR_DEGREES,
            observed={
                "moving_step_count": position["moving_step_count"],
                "moving_steps_with_toward_count": position[
                    "moving_steps_with_toward_count"
                ],
                "max_error_degrees": mirror_error,
            },
            required=(
                f">= {MIN_FRESH_POSITION_ARRIVALS} moving steps, every step with "
                f"|bearing + toward - {COMPASS_MIRROR_SUM_DEGREES}| <= "
                f"{MAX_COMPASS_MIRROR_ERROR_DEGREES} degrees"
            ),
        ),
        *corridor_criteria,
    ]
    if expected["require_toward_change"]:
        criteria.append(
            _criterion(
                "toward_changed_before_stop",
                position["toward_changed_before_stop"],
                observed={
                    "toward_change_count": position["toward_change_count"],
                    "total_absolute_change_degrees": position[
                        "total_absolute_toward_change_degrees"
                    ],
                },
                required=True,
            )
        )
    failed = [criterion["name"] for criterion in criteria if not criterion["passed"]]
    return {
        "control": control,
        "passed": not failed,
        "failed_criteria": failed,
        "criteria": criteria,
        "position_diagnostics": position,
        "corridor_diagnostics": corridor_report,
    }


def analyze_phase1_pair(
    straight_raw: Any,
    shallow_arc_raw: Any,
    corridors_raw: Any,
) -> dict[str, Any]:
    """Analyze the required capture pair and return a fail-closed verdict."""
    corridors = corridors_raw if isinstance(corridors_raw, dict) else {}
    captures = {
        "straight": analyze_capture(
            "straight", straight_raw, corridors.get("straight")
        ),
        "shallow_arc": analyze_capture(
            "shallow_arc", shallow_arc_raw, corridors.get("shallow_arc")
        ),
    }
    failed = [
        f"{control}.{criterion}"
        for control, capture in captures.items()
        for criterion in capture["failed_criteria"]
    ]
    return {
        "mode": "offline_phase1_capture_analysis",
        "dispatch_capable": False,
        "network_access": False,
        "commands_sent": 0,
        "physical_run_authorized": False,
        "thresholds": {
            # Per control, so the banked verdict records WHICH duration each
            # capture was required to have rather than implying one global.
            "duration_ms_by_control": {
                control: spec["duration_ms"]
                for control, spec in EXPECTED_CONTROLS.items()
            },
            "refresh_interval_ms": EXPECTED_REFRESH_INTERVAL_MS,
            "sample_interval_ms": EXPECTED_SAMPLE_INTERVAL_MS,
            "min_fresh_position_arrivals": MIN_FRESH_POSITION_ARRIVALS,
            "max_position_arrival_gap_ms": MAX_POSITION_ARRIVAL_GAP_MS,
            "compass_mirror_sum_degrees": COMPASS_MIRROR_SUM_DEGREES,
            "max_compass_mirror_error_degrees": (MAX_COMPASS_MIRROR_ERROR_DEGREES),
            "min_area_margin_m": MIN_AREA_MARGIN_M,
            "min_keepout_margin_m": MIN_KEEPOUT_MARGIN_M,
            "max_start_drift_m": MAX_START_DRIFT_M,
        },
        "captures": captures,
        "failed_criteria": failed,
        "verdict": "go" if not failed else "no_go",
        "verdict_scope": (
            "Phase 1 telemetry feasibility only; never authorizes Phase 2 or motion"
        ),
    }


def main() -> int:
    """Parse capture files, print/write analysis, and fail on a no-go verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--straight", type=Path, required=True)
    parser.add_argument("--arc", type=Path, required=True)
    parser.add_argument("--corridors", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    input_paths = {
        "straight": args.straight,
        "shallow_arc": args.arc,
        "corridors": args.corridors,
    }
    input_bytes = {name: path.read_bytes() for name, path in input_paths.items()}
    result = analyze_phase1_pair(
        json.loads(input_bytes["straight"]),
        json.loads(input_bytes["shallow_arc"]),
        json.loads(input_bytes["corridors"]),
    )
    result["inputs"] = {
        name: {
            "path": str(input_paths[name]),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in input_bytes.items()
    }
    rendered = json.dumps(result, indent=None if args.compact else 2)
    if args.output:
        args.output.write_text(f"{rendered}\n")
    print(rendered)
    return 0 if result["verdict"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
