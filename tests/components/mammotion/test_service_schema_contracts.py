"""Tests for Mammotion service schema defaults and entity-service-schema contracts."""

import ast
import json
import pathlib

import pytest
import voluptuous as vol
import yaml
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.service import _validate_entity_service_schema

from custom_components.mammotion import lawn_mower as mammotion_lawn_mower
from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.services import (
    DEFAULT_HEADING_OFFSET_CANDIDATES,
    EXPERIMENTAL_EXECUTE_SEGMENT_BURST_SCHEMA,
    EXPERIMENTAL_EXECUTE_SEGMENT_SCHEMA,
    FORWARD_TWO_PULSE_LATENCY_TEST_SCHEMA,
    MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST_SCHEMA,
    MANUAL_VELOCITY_HEADING_CALIBRATION_TEST_SCHEMA,
    MANUAL_VELOCITY_MULTI_PULSE_TEST_SCHEMA,
    MANUAL_VELOCITY_PULSE_TEST_SCHEMA,
    MANUAL_VELOCITY_SEGMENT_TEST_SCHEMA,
    POSITION_FEEDBACK_DIAGNOSTIC_SCHEMA,
    RAW_MOTION_READINESS_TEST_SCHEMA,
    RAW_PYMAMMOTION_ANGULAR_CALIBRATION_SCHEMA,
    RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA,
    RAW_PYMAMMOTION_EXECUTE_SEGMENT_SCHEMA,
    RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA,
    RAW_PYMAMMOTION_MOTION_PROBE_SCHEMA,
    RAW_PYMAMMOTION_TURN_TO_HEADING_SCHEMA,
    RAW_VECTOR_READINESS_TEST_SCHEMA,
    VIO_TURN_PROBE_SCHEMA,
    VIO_TURN_TO_HEADING_SCHEMA,
    _app_scale_speeds,
)


def test_manual_velocity_pulse_schema_allows_up_to_executor_pulse_speed() -> None:
    """Pulse probe allows up to the click-to-path executor pulse (raw 400) but caps at 0.6.

    The default 0.55 resolves to raw linear 400 -- the exact speed the vector and
    multi-segment executors send -- so the B1 A/B measures the pulse click-to-path
    actually drives. 0.6 (raw 450) is the ceiling; anything above is rejected.
    """
    parsed = MANUAL_VELOCITY_PULSE_TEST_SCHEMA(
        {
            "entity_id": "lawn_mower.test",
            "action": "forward",
            "speed": 0.55,
        }
    )
    assert parsed["speed"] == 0.55
    assert parsed["stop_mode"] == "immediate"
    assert parsed["post_command_sample_delays"] == [0.0, 2.0, 10.0, 30.0, 60.0]
    # The old emergency-nudge-tied ceiling of 0.4 is lifted; 0.45 now passes.
    assert (
        MANUAL_VELOCITY_PULSE_TEST_SCHEMA(
            {"entity_id": "lawn_mower.test", "action": "forward", "speed": 0.45}
        )["speed"]
        == 0.45
    )
    with pytest.raises(Exception):  # noqa: B017
        MANUAL_VELOCITY_PULSE_TEST_SCHEMA(
            {
                "entity_id": "lawn_mower.test",
                "action": "forward",
                "speed": 0.65,
            }
        )


def test_manual_velocity_pulse_defaults_match_executor_pulse() -> None:
    """Schema defaults reproduce the proven executor pulse (raw 400, ~4s window).

    Regression guard for the B1 harness bug: the old defaults (speed 0.1 -> raw 0,
    duration 250ms -> below the ~3s move threshold) made every default pulse a
    physical no-op, and the 750ms cap rejected the documented 4000ms B1 call.
    """
    parsed = MANUAL_VELOCITY_PULSE_TEST_SCHEMA({"entity_id": "lawn_mower.test"})
    assert parsed["speed"] == 0.55
    assert parsed["duration_ms"] == 3500
    # Default speed resolves to the raw linear speed the executors send.
    linear_speed, angular_speed = _app_scale_speeds(parsed["speed"], 0.0)
    assert linear_speed == 400
    assert angular_speed == 0
    # The documented B1 call (4000ms) is now accepted rather than HTTP 400.
    assert (
        MANUAL_VELOCITY_PULSE_TEST_SCHEMA(
            {"entity_id": "lawn_mower.test", "duration_ms": 4000}
        )["duration_ms"]
        == 4000
    )


@pytest.mark.parametrize(
    ("schema", "payload", "expected"),
    [
        (
            RAW_PYMAMMOTION_MOTION_PROBE_SCHEMA,
            {"entity_id": "lawn_mower.test"},
            {
                "command": "send_movement",
                "linear_speed": 400,
                "angular_speed": 0,
                "speed": 0.4,
                "prefer_ble": True,
                "sample_delays": [0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0],
                "dry_run": True,
            },
        ),
        (
            FORWARD_TWO_PULSE_LATENCY_TEST_SCHEMA,
            {"entity_id": "lawn_mower.test"},
            {
                "linear_speed": 200,
                "pulse_count": 2,
                "pulse_gap_seconds": 5.0,
                "telemetry_timeout_seconds": 60.0,
                "telemetry_sample_interval_seconds": 1.0,
                "min_position_change_distance": 0.003,
                "prefer_ble": True,
                "dry_run": True,
            },
        ),
        (
            POSITION_FEEDBACK_DIAGNOSTIC_SCHEMA,
            {"entity_id": "lawn_mower.test"},
            {
                "linear_speed": 200,
                "pulse_count": 0,
                "pulse_gap_seconds": 5.0,
                "refresh_wait_seconds": 2.0,
                "prefer_ble": True,
                "dry_run": True,
            },
        ),
        (
            RAW_PYMAMMOTION_EXECUTE_SEGMENT_SCHEMA,
            {
                "entity_id": "lawn_mower.test",
                "points": [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.8}],
            },
            {
                "dry_run": True,
                "prefer_ble": True,
                "linear_speed_fast": 400,
                "linear_speed_slow": 200,
                "max_commands": 3,
                "waypoint_tolerance": 0.08,
                "min_progress_distance": 0.06,
                # The handler reads this for the software-stop pulse bound;
                # covered by the generic parity sweep too.
                "linear_pulse_duration_ms": 3500.0,
                "sample_delays": [0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0],
            },
        ),
        (
            RAW_PYMAMMOTION_ANGULAR_CALIBRATION_SCHEMA,
            {"entity_id": "lawn_mower.test"},
            {
                "direction": "positive_heading",
                "angular_speed": 180,
                "target_heading_delta_degrees": 10.0,
                "max_commands": 3,
                "min_heading_change_degrees": 1.0,
                "max_translation_distance": 0.25,
                "prefer_ble": True,
                "sample_delays": [0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0],
                "dry_run": True,
            },
        ),
        (
            RAW_PYMAMMOTION_TURN_TO_HEADING_SCHEMA,
            {"entity_id": "lawn_mower.test", "target_heading_degrees": 20},
            {
                "target_heading_degrees": 20.0,
                "heading_tolerance_degrees": 18.0,
                "angular_speed_fast": 180,
                "angular_speed_slow": 90,
                "slow_turn_threshold_degrees": 8.0,
                "max_commands": 3,
                "min_heading_change_degrees": 0.5,
                "max_translation_distance": 0.25,
                "prefer_ble": True,
                "dry_run": True,
            },
        ),
        (
            RAW_MOTION_READINESS_TEST_SCHEMA,
            {"entity_id": "lawn_mower.test"},
            {
                "dry_run": True,
                "confirm_blades_off": False,
                "confirm_clear_area": False,
                "prefer_ble": True,
                "max_real_steps": 0,
                "sample_delays": [0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0],
            },
        ),
        (
            RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA,
            {
                "entity_id": "lawn_mower.test",
                "points": [{"x": 1, "y": 1}, {"x": 1.1, "y": 1}],
            },
            {
                "dry_run": True,
                "prefer_ble": True,
                # Handler reads call.data["ble_auto_recover"]; schema must default it.
                "ble_auto_recover": True,
                "linear_speed_fast": 400,
                "linear_speed_slow": 200,
                "angular_speed_fast": 180,
                "angular_speed_slow": 180,
                "calibrated_forward_heading_offset_degrees": 116.5,
                "max_turn_commands": 3,
                "max_linear_commands": 1,
                # App-parity refresh defaults ON for the executors (B1 2026-07-22);
                # the manual-pulse / turn-probe harnesses stay single-shot (0).
                "motion_refresh_interval_ms": 200,
            },
        ),
        (
            RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA,
            {
                "entity_id": "lawn_mower.test",
                "points": [
                    {"x": 1, "y": 1},
                    {"x": 1.1, "y": 1},
                    {"x": 1.2, "y": 1.1},
                ],
            },
            {
                "dry_run": True,
                "prefer_ble": True,
                # Regression: the multi handler reads call.data["ble_auto_recover"];
                # a missing schema default made every card multi-segment call 500.
                "ble_auto_recover": True,
                "max_real_segments": 1,
                "max_turn_commands": 4,
                "max_linear_commands": 2,
                "final_approach_metres_per_pulse": 1.06,
                "turn_degrees_per_second": 37.0,
                "calibrated_forward_heading_offset_degrees": 116.5,
                # App-parity refresh defaults ON for the executors (B1 2026-07-22).
                "motion_refresh_interval_ms": 200,
            },
        ),
        (
            RAW_VECTOR_READINESS_TEST_SCHEMA,
            {"entity_id": "lawn_mower.test"},
            {
                "dry_run": True,
                "prefer_ble": True,
                "max_real_steps": 0,
                "target_distance": 0.10,
                "turn_delta_degrees": 10.0,
                "calibrated_forward_heading_offset_degrees": 116.5,
                "max_turn_commands": 4,
                "max_linear_commands": 2,
            },
        ),
    ],
)
def test_motion_and_vector_schema_defaults_parameterized(
    schema: object,
    payload: dict[str, object],
    expected: dict[str, object],
) -> None:
    """Schema defaults remain stable across motion and vector service families."""
    parsed = schema(payload)
    for key, value in expected.items():
        assert parsed[key] == value


def test_motion_refresh_default_split_executors_on_harnesses_off() -> None:
    """The path executors default to app-parity refresh; the diagnostic harnesses stay single-shot.

    B1 (2026-07-22) proved re-sending the movement command every 200 ms drives
    ~11x further than a single shot, so the vector and multi-segment executors --
    the services the click-to-path card drives -- now default
    ``motion_refresh_interval_ms`` to 200. The bare-pulse A/B harness
    (``manual_velocity_pulse_test``) and the turn probe (``vio_turn_probe``) must
    stay at 0: they exist to compare 0 vs 200 explicitly, and refresh was proven
    speed-gated (it did nothing for the under-powered turn), so a defaulted-on
    turn would silently change the very experiment they run.
    """
    minimal_points = [{"x": 1, "y": 1}, {"x": 1.1, "y": 1}]
    vector = RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA(
        {"entity_id": "lawn_mower.test", "points": minimal_points}
    )
    multi = RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA(
        {
            "entity_id": "lawn_mower.test",
            "points": [*minimal_points, {"x": 1.2, "y": 1.1}],
        }
    )
    assert vector["motion_refresh_interval_ms"] == 200
    assert multi["motion_refresh_interval_ms"] == 200

    pulse = MANUAL_VELOCITY_PULSE_TEST_SCHEMA({"entity_id": "lawn_mower.test"})
    turn = VIO_TURN_PROBE_SCHEMA({"entity_id": "lawn_mower.test"})
    assert pulse["motion_refresh_interval_ms"] == 0
    assert turn["motion_refresh_interval_ms"] == 0

    # The closed-loop turn accepts refresh (proven ~7x at angular 500 on
    # 2026-07-25) but stays OPT-IN: `heading_tolerance_degrees` is still 18,
    # derived from the single-shot ~8-15 deg quantum, so defaulting refresh on
    # before re-deriving it would drive continuous rotation into a deadband
    # sized for discrete steps.
    closed_loop = VIO_TURN_TO_HEADING_SCHEMA(
        {"entity_id": "lawn_mower.test", "target_vision_heading": 90.0}
    )
    assert closed_loop["motion_refresh_interval_ms"] == 0
    assert closed_loop["heading_tolerance_degrees"] == 18.0
    assert (
        VIO_TURN_TO_HEADING_SCHEMA(
            {
                "entity_id": "lawn_mower.test",
                "target_vision_heading": 90.0,
                "motion_refresh_interval_ms": 200,
            }
        )["motion_refresh_interval_ms"]
        == 200
    )


_SERVICES_AST = ast.parse(
    pathlib.Path(mammotion_services.__file__).read_text(encoding="utf-8")
)


def _collect_function_defs() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Index every function definition in services.py by name, nested included."""
    defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(_SERVICES_AST):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.setdefault(node.name, node)
    return defs


_FUNCTION_DEFS = _collect_function_defs()


def _resolve_key_node(node: ast.expr) -> str | None:
    """Resolve an AST key node (string literal or module constant) to a string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        value = getattr(mammotion_services, node.id, None)
        return value if isinstance(value, str) else None
    return None


def _is_call_data_attribute(node: ast.expr) -> bool:
    """Return True when the node is the ``call.data`` attribute access."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "data"
        and isinstance(node.value, ast.Name)
        and node.value.id == "call"
    )


def _direct_call_data_reads(fn: ast.AST) -> set[str]:
    """Collect keys read via ``call.data[...]`` subscripts inside a function."""
    reads: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript) and _is_call_data_attribute(node.value):
            key = _resolve_key_node(node.slice)
            if key is not None:
                reads.add(key)
    return reads


def _membership_guarded_keys(fn: ast.AST) -> set[str]:
    """Collect keys the function membership-tests via ``"key" in call.data``."""
    guarded: set[str] = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.In)
            and _is_call_data_attribute(node.comparators[0])
        ):
            key = _resolve_key_node(node.left)
            if key is not None:
                guarded.add(key)
    return guarded


def _callees_passed_call(fn: ast.AST) -> set[str]:
    """Collect names of functions this function calls with ``call`` as an arg."""
    callees: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        passes_call = any(
            isinstance(arg, ast.Name) and arg.id == "call" for arg in node.args
        ) or any(
            isinstance(kw.value, ast.Name) and kw.value.id == "call"
            for kw in node.keywords
        )
        if not passes_call:
            continue
        if isinstance(node.func, ast.Name):
            callees.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            callees.add(node.func.attr)
    return callees


def _handler_call_data_keys(handler_name: str, _depth: int = 0) -> set[str]:
    """Unguarded call.data keys a handler reads, following call-passing helpers."""
    fn = _FUNCTION_DEFS.get(handler_name)
    if fn is None or _depth > 3:
        return set()
    keys = _direct_call_data_reads(fn) - _membership_guarded_keys(fn)
    for callee in _callees_passed_call(fn):
        if callee != handler_name:
            keys |= _handler_call_data_keys(callee, _depth + 1)
    return keys


def _service_schema_registrations() -> list[tuple[str, str, str | None]]:
    """Map every async_register call to (service label, handler, schema name)."""
    registrations: list[tuple[str, str, str | None]] = []
    for node in ast.walk(_SERVICES_AST):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "async_register"
        ):
            continue
        args = node.args
        handler_node = args[2] if len(args) > 2 else None
        schema_node: ast.expr | None = args[3] if len(args) > 3 else None
        for keyword in node.keywords:
            if keyword.arg == "schema":
                schema_node = keyword.value
        if (
            isinstance(handler_node, ast.Call)
            and isinstance(handler_node.func, ast.Name)
            and handler_node.func.id == "_wrap_exclusive_manual_motion"
            and len(handler_node.args) > 2
        ):
            # The per-mower exclusivity guard wraps the real handler at
            # registration time; parity must keep checking the handler's
            # call.data reads (the guard itself only uses .get()).
            handler_node = handler_node.args[2]
        if not isinstance(handler_node, ast.Name):
            continue
        schema_name = schema_node.id if isinstance(schema_node, ast.Name) else None
        service_label = (
            _resolve_key_node(args[1]) if len(args) > 1 else None
        ) or handler_node.id
        registrations.append((service_label, handler_node.id, schema_name))
    return registrations


_SERVICE_REGISTRATIONS = _service_schema_registrations()

_MINIMAL_REQUIRED_SAMPLES: dict[str, object] = {
    "entity_id": "lawn_mower.test",
    "periods_ms": [1000],
    "name": "Test task",
    "enabled": True,
    "points": [{"x": 1.0, "y": 1.0}, {"x": 1.2, "y": 1.0}],
    "route_start": {"x": 0.0, "y": 0.0},
    "route_target": {"x": 3.0, "y": 0.0},
    "corridor_polygon": [
        {"x": -0.3, "y": -0.3},
        {"x": 3.3, "y": -0.3},
        {"x": 3.3, "y": 0.3},
        {"x": -0.3, "y": 0.3},
    ],
    "target_vision_heading": 90.0,
    "target_heading_degrees": 90.0,
    "dry_run": False,
    "confirm_blades_off": True,
    "confirm_clear_area": True,
    "area_hash": 12345,
    "device_hash": 67890,
    "svg_data": "<svg></svg>",
}


def _minimal_valid_payload(schema: vol.Schema) -> dict[str, object]:
    """Build the smallest payload that satisfies the schema's required keys."""
    payload: dict[str, object] = {}
    for marker in schema.schema:
        if not isinstance(marker, vol.Required):
            continue
        key = marker.schema
        assert key in _MINIMAL_REQUIRED_SAMPLES, (
            f"add a sample value for new required schema key {key!r} to"
            " _MINIMAL_REQUIRED_SAMPLES"
        )
        payload[key] = _MINIMAL_REQUIRED_SAMPLES[key]
    return payload


def test_service_registration_discovery_is_complete() -> None:
    """The AST sweep keeps seeing the full service surface and known reads."""
    assert len(_SERVICE_REGISTRATIONS) >= 45
    assert "ble_auto_recover" in _handler_call_data_keys(
        "handle_raw_pymammotion_execute_multi_segment"
    )
    assert "stop_mode" in _handler_call_data_keys("handle_manual_velocity_segment_test")
    # Indirection: the movement services read speed/use_wifi via handle_movement.
    assert "speed" in _handler_call_data_keys("handle_directional_movement")


@pytest.mark.parametrize(
    ("service_label", "handler_name", "schema_name"),
    _SERVICE_REGISTRATIONS,
    ids=[registration[0] for registration in _SERVICE_REGISTRATIONS],
)
def test_handler_read_keys_resolve_from_schema_defaults(
    service_label: str, handler_name: str, schema_name: str | None
) -> None:
    """Every unguarded call.data[...] read resolves on a minimal service call."""
    read_keys = _handler_call_data_keys(handler_name)
    if schema_name is None:
        assert not read_keys, (
            f"{service_label}: handler {handler_name} reads call.data keys"
            f" {sorted(read_keys)} but is registered without a schema"
        )
        return
    schema = getattr(mammotion_services, schema_name)
    parsed = schema(_minimal_valid_payload(schema))
    missing = sorted(key for key in read_keys if key not in parsed)
    assert not missing, (
        f"{service_label}: handler {handler_name} reads call.data[...] for"
        f" {missing} but {schema_name} does not guarantee them on a minimal"
        " call — declare vol.Required or vol.Optional with a default"
        " (KeyError -> HTTP 500 regression class)"
    )


def test_manual_velocity_segment_schema_caps_probe_values() -> None:
    """Segment probe allows proven pulse caps and rejects larger values."""
    parsed = MANUAL_VELOCITY_SEGMENT_TEST_SCHEMA(
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1, "y": 1}, {"x": 1, "y": 2}],
            "speed": 0.4,
            "pulse_duration_ms": 750,
            "max_pulses": 5,
            "force_action": "forward",
            "heading_offset_degrees": 110,
        }
    )

    assert parsed["speed"] == 0.4
    assert parsed["pulse_duration_ms"] == 750
    assert parsed["max_pulses"] == 5
    assert parsed["force_action"] == "forward"
    assert parsed["heading_offset_degrees"] == 110
    assert parsed["min_progress_distance"] == 0.003
    assert parsed["no_progress_limit"] == 2
    with pytest.raises(Exception):  # noqa: B017
        MANUAL_VELOCITY_SEGMENT_TEST_SCHEMA(
            {
                "entity_id": "lawn_mower.test",
                "points": [{"x": 1, "y": 1}, {"x": 1, "y": 2}],
                "speed": 0.45,
            }
        )
    with pytest.raises(Exception):  # noqa: B017
        MANUAL_VELOCITY_SEGMENT_TEST_SCHEMA(
            {
                "entity_id": "lawn_mower.test",
                "points": [{"x": 1, "y": 1}, {"x": 1, "y": 2}],
                "pulse_duration_ms": 800,
            }
        )


def test_manual_velocity_multi_pulse_schema_requires_at_least_two_pulses() -> None:
    """Explicit multi-pulse service requires a multi-pulse cap range."""
    parsed = MANUAL_VELOCITY_MULTI_PULSE_TEST_SCHEMA(
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1, "y": 1}, {"x": 1, "y": 2}],
            "max_pulses": 2,
        }
    )

    assert parsed["max_pulses"] == 2
    with pytest.raises(Exception):  # noqa: B017
        MANUAL_VELOCITY_MULTI_PULSE_TEST_SCHEMA(
            {
                "entity_id": "lawn_mower.test",
                "points": [{"x": 1, "y": 1}, {"x": 1, "y": 2}],
                "max_pulses": 1,
            }
        )


def test_experimental_execute_segment_schema_is_real_two_point_only() -> None:
    """Experimental segment execution requires explicit real two-point execution."""
    parsed = EXPERIMENTAL_EXECUTE_SEGMENT_SCHEMA(
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1, "y": 1}, {"x": 2, "y": 1}],
            "dry_run": False,
            "confirm_blades_off": True,
            "confirm_clear_area": True,
            "max_pulses": 3,
        }
    )

    assert parsed["dry_run"] is False
    assert parsed["confirm_blades_off"] is True
    assert parsed["confirm_clear_area"] is True
    assert parsed["max_pulses"] == 3

    invalid_cases = [
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1, "y": 1}],
            "dry_run": False,
            "confirm_blades_off": True,
            "confirm_clear_area": True,
        },
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}],
            "dry_run": False,
            "confirm_blades_off": True,
            "confirm_clear_area": True,
        },
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1, "y": 1}, {"x": 2, "y": 1}],
            "dry_run": True,
            "confirm_blades_off": True,
            "confirm_clear_area": True,
        },
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1, "y": 1}, {"x": 2, "y": 1}],
            "dry_run": False,
            "confirm_blades_off": False,
            "confirm_clear_area": True,
        },
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1, "y": 1}, {"x": 2, "y": 1}],
            "dry_run": False,
            "confirm_blades_off": True,
            "confirm_clear_area": True,
            "max_pulses": 4,
        },
    ]
    for invalid in invalid_cases:
        with pytest.raises(Exception):  # noqa: B017
            EXPERIMENTAL_EXECUTE_SEGMENT_SCHEMA(invalid)


def test_experimental_execute_segment_burst_schema_is_real_two_point_only() -> None:
    """Burst execution requires explicit real two-point execution."""
    parsed = EXPERIMENTAL_EXECUTE_SEGMENT_BURST_SCHEMA(
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1, "y": 1}, {"x": 2, "y": 1}],
            "dry_run": False,
            "confirm_blades_off": True,
            "confirm_clear_area": True,
            "pulses_per_burst": 3,
            "max_bursts": 3,
            "heading_offset_candidates": [110, 0, 90],
        }
    )

    assert parsed["dry_run"] is False
    assert parsed["confirm_blades_off"] is True
    assert parsed["confirm_clear_area"] is True
    assert parsed["pulses_per_burst"] == 3
    assert parsed["max_bursts"] == 3
    assert parsed["heading_offset_candidates"] == [110.0, 0.0, 90.0]
    assert parsed["allow_unproven_turns"] is False
    assert parsed["calibrated_forward_heading_degrees"] == 270.0
    assert parsed["calibrated_forward_heading_tolerance_degrees"] == 45.0

    invalid_cases = [
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1, "y": 1}],
            "dry_run": False,
            "confirm_blades_off": True,
            "confirm_clear_area": True,
        },
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1, "y": 1}, {"x": 2, "y": 1}],
            "dry_run": True,
            "confirm_blades_off": True,
            "confirm_clear_area": True,
        },
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1, "y": 1}, {"x": 2, "y": 1}],
            "dry_run": False,
            "confirm_blades_off": True,
            "confirm_clear_area": True,
            "pulses_per_burst": 4,
        },
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1, "y": 1}, {"x": 2, "y": 1}],
            "dry_run": False,
            "confirm_blades_off": True,
            "confirm_clear_area": True,
            "max_bursts": 4,
        },
    ]
    for invalid in invalid_cases:
        with pytest.raises(Exception):  # noqa: B017
            EXPERIMENTAL_EXECUTE_SEGMENT_BURST_SCHEMA(invalid)


def test_manual_velocity_heading_calibration_schema_defaults() -> None:
    """Heading calibration schema defaults to a dry-run forward pulse."""
    parsed = MANUAL_VELOCITY_HEADING_CALIBRATION_TEST_SCHEMA(
        {"entity_id": "lawn_mower.test"}
    )

    assert parsed["action"] == "forward"
    assert parsed["dry_run"] is True
    assert parsed["speed"] == 0.4
    assert parsed["duration_ms"] == 750
    assert parsed["use_wifi"] is False
    assert parsed["stop_mode"] == "firmware"
    assert parsed["post_command_sample_delays"] == [0, 10, 20, 30, 45, 60]


def test_manual_velocity_cumulative_pulse_schema_defaults() -> None:
    """Cumulative pulse schema defaults to dry-run delayed telemetry sampling."""
    parsed = MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST_SCHEMA(
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        }
    )

    assert parsed["dry_run"] is True
    assert parsed["max_pulses"] == 3
    assert parsed["stop_mode"] == "immediate"
    assert parsed["stop_delay_ms"] == 0
    assert parsed["cumulative_sample_delays"][-1] == 120
    assert parsed["heading_offset_candidates"] == list(
        DEFAULT_HEADING_OFFSET_CANDIDATES
    )


def test_manual_velocity_heading_offset_candidate_schema_rejects_invalid_values() -> (
    None
):
    """Heading offset candidates are bounded to valid degrees."""
    with pytest.raises(Exception):  # noqa: B017
        MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST_SCHEMA(
            {
                "entity_id": "lawn_mower.test",
                "points": [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
                "heading_offset_candidates": [0, 181],
            }
        )


def test_services_yaml_has_matching_strings_entries() -> None:
    """New services must be documented in strings.json; known pre-existing gaps are allowlisted.

    raw_pymammotion_execute_multi_segment was fully implemented and registered but had no
    strings.json entry until it was added alongside this test. A broader, older set of
    services is intentionally allowlisted below rather than fixed here (unrelated pre-existing
    debt), so this test only guards against *new* undocumented services going forward.
    """
    package_dir = pathlib.Path(mammotion_services.__file__).parent
    services_yaml = yaml.safe_load((package_dir / "services.yaml").read_text())
    strings_json = json.loads((package_dir / "strings.json").read_text())
    yaml_keys = set(services_yaml.keys())
    strings_keys = set(strings_json["services"].keys())

    known_undocumented = {
        "force_map_resync",
        "forward_two_pulse_latency_test",
        "get_geojson",
        "get_mow_path_geojson",
        "get_mow_progress_geojson",
        "get_tokens",
        "move_backward",
        "move_forward",
        "move_left",
        "move_right",
        "position_feedback_diagnostic",
        "raw_motion_readiness_test",
        "raw_pymammotion_angular_calibration",
        "raw_pymammotion_execute_segment",
        "raw_pymammotion_turn_to_heading",
        "raw_vector_readiness_test",
        "refresh_stream",
        "start_video",
        "stop_video",
        "vio_motion_probe",
        "vio_turn_probe",
        "vio_turn_to_heading",
    }

    missing = yaml_keys - strings_keys - known_undocumented
    assert not missing, (
        f"Service(s) {sorted(missing)} are registered in services.yaml but missing a "
        "strings.json entry. Add documentation, or add to known_undocumented in this test "
        "if the gap is intentional pre-existing debt."
    )

    now_documented = known_undocumented & strings_keys
    assert not now_documented, (
        f"Service(s) {sorted(now_documented)} now have strings.json entries -- remove them "
        "from known_undocumented in this test to keep the allowlist honest."
    )


def test_motion_services_registered_with_exclusive_guard() -> None:
    """Every manual-motion service registration goes through the guard."""
    guarded: set[str] = set()
    for node in ast.walk(_SERVICES_AST):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "async_register"
        ):
            continue
        args = node.args
        handler_node = args[2] if len(args) > 2 else None
        if (
            isinstance(handler_node, ast.Call)
            and isinstance(handler_node.func, ast.Name)
            and handler_node.func.id == "_wrap_exclusive_manual_motion"
            and len(args) > 1
        ):
            label = _resolve_key_node(args[1])
            if label is not None:
                guarded.add(label)
    expected = {
        getattr(mammotion_services, name)
        for name in (
            "SERVICE_MANUAL_VELOCITY_PULSE_TEST",
            "SERVICE_MANUAL_VELOCITY_SEGMENT_TEST",
            "SERVICE_MANUAL_VELOCITY_MULTI_PULSE_TEST",
            "SERVICE_MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST",
            "SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT",
            "SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT_BURST",
            "SERVICE_MANUAL_VELOCITY_HEADING_CALIBRATION_TEST",
            "SERVICE_RAW_PYMAMMOTION_MOTION_PROBE",
            "SERVICE_RAW_PYMAMMOTION_EXECUTE_SEGMENT",
            "SERVICE_RAW_PYMAMMOTION_ANGULAR_CALIBRATION",
            "SERVICE_RAW_PYMAMMOTION_TURN_TO_HEADING",
            "SERVICE_RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT",
            "SERVICE_RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT",
            "SERVICE_FORWARD_TWO_PULSE_LATENCY_TEST",
            "SERVICE_POSITION_FEEDBACK_DIAGNOSTIC",
            "SERVICE_VIO_MOTION_PROBE",
            "SERVICE_VIO_TURN_PROBE",
            "SERVICE_VIO_TURN_TO_HEADING",
            "SERVICE_RAW_MOTION_READINESS_TEST",
            "SERVICE_RAW_VECTOR_READINESS_TEST",
        )
    }
    missing = expected - guarded
    assert not missing, (
        f"Manual-motion service(s) {sorted(missing)} are registered without "
        "_wrap_exclusive_manual_motion -- concurrent motion loops can interleave."
    )


@pytest.mark.parametrize(
    "schema_name",
    [
        "START_MOW_SCHEMA",
        "START_STOP_BLADES_SCHEMA",
        "SET_NON_WORK_HOURS_SCHEMA",
        "SET_BLADE_WARNING_TIME_SCHEMA",
    ],
)
def test_lawn_mower_platform_schemas_are_entity_service_schemas(
    schema_name: str,
) -> None:
    """Platform entity-service schemas must survive HA's own validator.

    Regression for 2026-07-19: these were registered as ``vol.Schema(DICT)``.
    HA wraps a plain dict with ``cv.make_entity_service_schema`` but requires an
    already-built schema to BE an entity service schema, so setup raised "The
    mammotion.start_mow service registers an entity service with a non entity
    service schema" on every restart. Because that raised out of
    ``async_setup_entry`` AFTER ``async_add_entities``, the entities loaded but
    all six lawn_mower platform services were silently missing.
    """
    schema = getattr(mammotion_lawn_mower, schema_name)
    validated = _validate_entity_service_schema(schema, f"mammotion.{schema_name}")
    assert cv.is_entity_service_schema(validated)
