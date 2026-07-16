"""Tests for Mammotion read-only map/task visibility helpers."""

import datetime
import json
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from homeassistant.exceptions import HomeAssistantError
from pymammotion.data.model.hash_list import Plan
from pymammotion.transport.base import TransportType
from pymammotion.transport.ble import BLETransport, BLETransportConfig

from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.button import BUTTON_LUBA_PRO_YUKA
from custom_components.mammotion.coordinator import (
    MammotionBaseUpdateCoordinator,
    MammotionReportUpdateCoordinator,
)
from custom_components.mammotion.sensor import WORK_SENSOR_TYPES
from custom_components.mammotion.services import (
    _VIO_HEADING_FRESH_EPSILON_DEGREES,
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
    _ble_connect_cooldown_active,
    _custom_path_telemetry_snapshot,
    _dry_run_custom_path,
    _execute_custom_path,
    _experimental_execute_segment_burst,
    _export_active_route,
    _export_mower_map,
    _export_mower_tasks,
    _export_runtime_state,
    _forward_two_pulse_latency_test,
    _manual_velocity_best_heading_decision,
    _manual_velocity_controller_decision,
    _manual_velocity_cumulative_pulse_test,
    _manual_velocity_heading_calibration,
    _manual_velocity_path_progress_diagnostic,
    _manual_velocity_pulse_test,
    _manual_velocity_segment_test,
    _normalize_mower_areas,
    _normalize_mower_tasks,
    _position_feedback_diagnostic,
    _preview_custom_path,
    _raw_motion_readiness_test,
    _raw_pymammotion_angular_calibration,
    _raw_pymammotion_execute_multi_segment,
    _raw_pymammotion_execute_segment,
    _raw_pymammotion_execute_vector_segment,
    _raw_pymammotion_motion_probe,
    _raw_pymammotion_turn_to_heading,
    _raw_vector_readiness_phase_passed,
    _raw_vector_readiness_test,
    _settle_linear_position_feed,
    _transport_is_ble,
    _validate_custom_path,
    _vio_feed_liveness,
    _vio_motion_probe,
    _vio_segment_calibration_drive,
    _vio_turn_probe,
    _vio_turn_to_heading,
)

LARGE_HASH = 9_223_372_036_854_775_000


def _plan(
    plan_id: str = "plan-1",
    *,
    name: str = "Front yard",
    zone_hashs: list[int] | None = None,
    enabled: bool = True,
) -> Plan:
    """Build a mower plan fixture."""
    plan = Plan(
        plan_id=plan_id,
        task_name=name,
        weeks=[1, 3, 5],
        start_time="07:30",
        end_time="09:00",
        start_date="2026-06-01",
        end_date="2026-08-31",
        knife_height=60,
        speed=0.4,
        edge_mode=1,
        route_angle=15,
        route_spacing=25,
        zone_hashs=zone_hashs or [LARGE_HASH],
    )
    return plan.with_enabled(enabled)


def _coordinator(plan: Plan | None = None) -> SimpleNamespace:
    """Build a minimal coordinator-like fixture for pure helpers."""
    plan = plan or _plan()
    mower_map = SimpleNamespace(
        plan={plan.plan_id: plan},
        area={LARGE_HASH: SimpleNamespace(data=[object(), object()])},
        area_name=[SimpleNamespace(hash=LARGE_HASH, name="Front Main")],
    )
    data = SimpleNamespace(map=mower_map)
    return SimpleNamespace(
        data=data,
        last_map_sync=None,
        last_task_sync=None,
        last_map_task_error=None,
        get_area_entity_name=lambda area_hash: (
            "Front Main" if area_hash == LARGE_HASH else f"area {area_hash}"
        ),
    )


def _pulse_coordinator(
    *,
    blade_state: int | None = 0,
    cutter_rpm: int | None = 0,
    work_mode: int = 11,
    charge_state: int = 0,
    pos_type: int = 1,
    zone_hash: int = 123,
    pos_level: int = 0,
    rtk_status: int = 4,
    position: tuple[float | None, float | None, float | None] = (1.0, 1.0, 0.0),
) -> SimpleNamespace:
    """Build a coordinator fixture for manual velocity pulse tests."""
    pos_x, pos_y, toward = position
    handle = SimpleNamespace(
        last_report_at=123.0,
        availability=SimpleNamespace(
            mqtt_reported_offline=False,
        ),
        # The real BLE connect cooldown lives on the transport, not on
        # availability; expose it the way pymammotion's DeviceHandle does. 0.0 =
        # no cooldown armed.
        get_transport=lambda _transport_type: SimpleNamespace(
            _connect_cooldown_until=0.0
        ),
        active_transport=lambda: "ble",
    )
    manager = SimpleNamespace(
        send_command_with_args=AsyncMock(),
        ensure_fresh_state=AsyncMock(),
        request_iot_sync=AsyncMock(),
        request_iot_sync_continuous=AsyncMock(),
        request_iot_sync_continuous_stop=AsyncMock(),
        mower=lambda _device_name: handle,
    )
    return SimpleNamespace(
        async_move_forward=AsyncMock(),
        async_move_back=AsyncMock(),
        async_move_left=AsyncMock(),
        async_move_right=AsyncMock(),
        async_stop_manual_motion=AsyncMock(),
        async_request_report_snapshot=AsyncMock(),
        async_get_reports=AsyncMock(),
        async_start_report_stream=AsyncMock(),
        async_send_command=AsyncMock(),
        async_request_refresh=AsyncMock(),
        device_name="Luba-Test",
        manager=manager,
        active_transport_state="ble",
        is_online=lambda: True,
        data=SimpleNamespace(
            map=SimpleNamespace(
                plan={},
                area={
                    123: SimpleNamespace(
                        data=[
                            SimpleNamespace(
                                current_frame=0,
                                data_couple=[
                                    SimpleNamespace(x=-10, y=-10),
                                    SimpleNamespace(x=10, y=-10),
                                    SimpleNamespace(x=10, y=10),
                                    SimpleNamespace(x=-10, y=10),
                                ],
                            )
                        ]
                    )
                },
                area_name=[SimpleNamespace(hash=123, name="Backyard Right")],
            ),
            mowing_state=SimpleNamespace(
                pos_x=pos_x,
                pos_y=pos_y,
                toward=toward,
                pos_level=pos_level,
                rtk_status=rtk_status,
                zone_hash=zone_hash,
                pos_type=pos_type,
            ),
            location=SimpleNamespace(
                orientation=toward,
                position_type=pos_type,
                work_zone=zone_hash,
            ),
            report_data=SimpleNamespace(
                dev=SimpleNamespace(
                    sys_status=work_mode,
                    charge_state=charge_state,
                    blade_state=blade_state,
                ),
                rtk=SimpleNamespace(status=rtk_status, pos_level=pos_level),
                locations=[],
                cutter_work_mode_info=SimpleNamespace(
                    current_cutter_mode=0,
                    current_cutter_rpm=cutter_rpm,
                ),
                connect=None,
            ),
        ),
        get_area_entity_name=lambda area_hash: (
            "Backyard Right" if area_hash == 123 else f"area {area_hash}"
        ),
    )


def test_get_tasks_normalizes_plan_fields_and_stringifies_raw_hashes() -> None:
    """Task response contains normalized fields plus precision-safe raw data."""
    coordinator = _coordinator()

    tasks = _normalize_mower_tasks(coordinator)

    assert tasks == [
        {
            "plan_id": "plan-1",
            "name": "Front yard",
            "enabled": True,
            "weeks": [1, 3, 5],
            "start_time": "07:30",
            "end_time": "09:00",
            "start_date": "2026-06-01",
            "end_date": "2026-08-31",
            "knife_height": 60,
            "speed": 0.4,
            "edge_mode": 1,
            "route_angle": 15,
            "route_spacing": 25,
            "zone_hashs": [str(LARGE_HASH)],
            "zone_names": ["Front Main"],
            "raw": tasks[0]["raw"],
        }
    ]
    assert tasks[0]["raw"]["zone_hashs"] == [str(LARGE_HASH)]


def test_get_areas_includes_names_geometry_and_task_references() -> None:
    """Area response links area metadata to referencing tasks."""
    coordinator = _coordinator()

    areas = _normalize_mower_areas(coordinator)

    assert areas == [
        {
            "area_hash": str(LARGE_HASH),
            "name": "Front Main",
            "has_geometry": True,
            "frame_count": 2,
            "referenced_by_tasks": [
                {"plan_id": "plan-1", "name": "Front yard"},
            ],
        }
    ]


def test_get_areas_handles_unnamed_geometry() -> None:
    """Unnamed areas fall back to coordinator area naming."""
    plan = _plan(zone_hashs=[123])
    coordinator = _coordinator(plan)
    coordinator.data.map.area = {123: SimpleNamespace(data=[])}
    coordinator.data.map.area_name = []

    areas = _normalize_mower_areas(coordinator)

    assert areas == [
        {
            "area_hash": 123,
            "name": "area 123",
            "has_geometry": False,
            "frame_count": 0,
            "referenced_by_tasks": [{"plan_id": "plan-1", "name": "Front yard"}],
        }
    ]


def test_export_map_includes_area_polygons_and_raw_map_data() -> None:
    """Map export includes normalized areas, polygons, and raw map data."""
    coordinator = _coordinator()
    coordinator.data.map.area = {
        LARGE_HASH: SimpleNamespace(
            data=[
                SimpleNamespace(
                    current_frame=0,
                    data_couple=[
                        SimpleNamespace(x=0.0, y=0.0),
                        SimpleNamespace(x=10.0, y=0.0),
                        SimpleNamespace(x=10.0, y=10.0),
                        SimpleNamespace(x=0.0, y=10.0),
                    ],
                )
            ]
        )
    }

    export = _export_mower_map(coordinator)

    assert export["coordinate_system"] == "mower_map_xy"
    assert export["areas"][0]["area_hash"] == str(LARGE_HASH)
    assert export["area_polygons"][str(LARGE_HASH)] == [
        {"x": 0.0, "y": 0.0},
        {"x": 10.0, "y": 0.0},
        {"x": 10.0, "y": 10.0},
        {"x": 0.0, "y": 10.0},
    ]
    assert "area" in export["raw"]
    assert "area_name" in export["raw"]


def test_export_tasks_includes_counts_and_sync_metadata() -> None:
    """Task export wraps normalized tasks with diagnostic metadata."""
    coordinator = _coordinator(_plan(enabled=True))
    coordinator.last_task_sync = "2026-06-28T12:00:00+00:00"

    export = _export_mower_tasks(coordinator)

    assert export["task_count"] == 1
    assert export["enabled_task_count"] == 1
    assert export["tasks"][0]["plan_id"] == "plan-1"
    assert export["last_task_sync"] == "2026-06-28T12:00:00+00:00"
    assert export["last_map_task_error"] is None


def test_export_runtime_state_reports_blade_on_as_unsafe() -> None:
    """Reported blade ON blocks motion even when cutter RPM is zero."""
    coordinator = _pulse_coordinator(blade_state=1, cutter_rpm=0, work_mode=11)
    coordinator.active_transport_state = "ble"
    coordinator.ble_only_fallback_mode = False
    coordinator.last_cloud_login_success = None
    coordinator.last_token_refresh = None
    coordinator.last_command_failure_reason = "set_car_wiper:GatewayTimeoutException"
    coordinator.last_camera_stream_failure_code = "401"
    active_route = {
        "mow_path_feature_count": 0,
        "mow_progress_feature_count": 0,
        "active_progress": None,
    }

    exported = _export_runtime_state(
        coordinator,
        ha_state="paused",
        active_route=active_route,
    )

    assert exported["blade"]["reported_state"] == 1
    assert exported["blade"]["current_cutter_rpm"] == 0
    assert exported["blade"]["blade_safe_for_motion"] is False
    assert exported["active_transport"] == "ble"
    assert exported["ble_only_fallback_mode"] is False
    assert exported["last_command_failure_reason"] == "set_car_wiper:GatewayTimeoutException"
    assert exported["last_camera_stream_failure_code"] == "401"
    assert "blade_reported_on" in exported["safety"]["blockers"]


def test_export_runtime_state_reports_nonzero_rpm_as_unsafe() -> None:
    """Nonzero cutter RPM blocks motion even if reported blade state is off."""
    coordinator = _pulse_coordinator(blade_state=0, cutter_rpm=2995, work_mode=11)
    active_route = {
        "mow_path_feature_count": 0,
        "mow_progress_feature_count": 0,
        "active_progress": None,
    }

    exported = _export_runtime_state(
        coordinator,
        ha_state="paused",
        active_route=active_route,
    )

    assert exported["blade"]["reported_state"] == 0
    assert exported["blade"]["current_cutter_rpm"] == 2995
    assert exported["blade"]["blade_safe_for_motion"] is False
    assert "blade_rpm_nonzero" in exported["safety"]["blockers"]


def test_export_runtime_state_reports_active_mowing_and_route_blockers() -> None:
    """Active mowing state and active route/progress both block manual motion."""
    coordinator = _pulse_coordinator(blade_state=0, cutter_rpm=0, work_mode=13)
    active_route = {
        "mow_path_feature_count": 1,
        "mow_progress_feature_count": 1,
        "active_progress": {"is_active": True},
    }

    exported = _export_runtime_state(
        coordinator,
        ha_state="mowing",
        active_route=active_route,
    )

    assert exported["work_mode_label"] == "MODE_WORKING"
    assert exported["safety"]["active_mowing_detected"] is True
    assert exported["safety"]["active_route_detected"] is True
    assert exported["safety"]["active_route_status"]["blocks_motion"] is True
    assert (
        exported["safety"]["active_route_status"]["reason"]
        == "live_route_while_mowing"
    )
    assert "active_mowing_detected" in exported["safety"]["blockers"]
    assert "active_route_detected" in exported["safety"]["blockers"]


def test_export_runtime_state_allows_stale_route_when_paused_ready() -> None:
    """Residual active route data does not block a paused/ready mower by itself."""
    coordinator = _pulse_coordinator(blade_state=0, cutter_rpm=0, work_mode=11)
    active_route = {
        "mow_path_feature_count": 6,
        "mow_progress_feature_count": 5,
        "active_progress": {"is_active": True},
    }

    exported = _export_runtime_state(
        coordinator,
        ha_state="paused",
        active_route=active_route,
    )

    assert exported["safety"]["active_route_detected"] is True
    assert exported["safety"]["active_route_status"]["blocks_motion"] is False
    assert (
        exported["safety"]["active_route_status"]["reason"]
        == "stale_route_while_ready"
    )
    assert "active_route_detected" not in exported["safety"]["blockers"]
    assert exported["safety"]["allowed_for_manual_motion"] is True
    assert exported["manual_motion_execution_policy"] == {
        "arbitrary_path_execution_allowed": False,
        "full_path_execution_allowed": False,
        "experimental_segment_execution_allowed": True,
        "experimental_segment_scope": "one_segment_calibrated_forward_only",
        "turn_primitive_proven": False,
        "reverse_primitive_proven": False,
        "lateral_motion_proven": False,
        "raw_pymammotion_primitives": {
            "linear_positive": {
                "status": "partially_calibrated",
                "command": "send_movement",
                "linear_speed": 400,
                "angular_speed": 0,
                "observed_effect": "translation toward map-local negative Y",
            },
            "linear_negative": {
                "status": "partially_calibrated",
                "command": "send_movement",
                "linear_speed": -400,
                "angular_speed": 0,
                "observed_effect": "translation toward map-local positive Y",
            },
            "angular_positive": {
                "status": "weak_heading_change",
                "command": "send_movement",
                "linear_speed": 0,
                "angular_speed": 180,
                "observed_effect": "small positive heading change with drift",
            },
            "angular_negative": {
                "status": "weak_heading_change",
                "command": "send_movement",
                "linear_speed": 0,
                "angular_speed": -180,
                "observed_effect": "small negative heading change with minimal translation",
            },
        },
        "default_transport": "ble_preferred",
        "default_stop_mode": "firmware",
        "default_pulses_per_burst": 1,
        "default_max_bursts": 3,
        "calibrated_forward_heading_degrees": 270.0,
        "calibrated_forward_heading_tolerance_degrees": 45.0,
        "blocked_without_override": [
            "segments_outside_calibrated_forward_window",
            "turn_left",
            "turn_right",
            "multi_segment_paths",
            "arbitrary_drawn_path_execution",
        ],
        "reason": (
            "Raw pymammotion linear movement is partially calibrated; angular "
            "movement is weak but measurable. Arbitrary path execution remains "
            "blocked until closed-loop raw movement is implemented and tested."
        ),
    }


def test_export_runtime_state_blocks_active_route_when_state_ambiguous() -> None:
    """Residual route data blocks motion if mower runtime state is ambiguous."""
    coordinator = _pulse_coordinator(blade_state=0, cutter_rpm=0, work_mode=99)
    active_route = {
        "mow_path_feature_count": 1,
        "mow_progress_feature_count": 1,
        "active_progress": {"is_active": True},
    }

    exported = _export_runtime_state(
        coordinator,
        ha_state="unknown",
        active_route=active_route,
    )

    assert exported["safety"]["active_route_status"]["blocks_motion"] is True
    assert (
        exported["safety"]["active_route_status"]["reason"] == "route_state_ambiguous"
    )
    assert "active_route_detected" in exported["safety"]["blockers"]


def test_export_active_route_normalizes_path_and_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active route export summarizes mow-path and progress GeoJSON features."""
    large_path_hash = 9_223_372_036_854_775_000
    coordinator = SimpleNamespace(
        device_name="Luba-Test",
        map_offset_lat=0.0,
        map_offset_lon=0.0,
        data=SimpleNamespace(
            device_firmwares=SimpleNamespace(main_controller="1.0.0"),
            map=SimpleNamespace(
                generated_mow_path_geojson={
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {
                                "transaction_id": large_path_hash,
                                "type_name": "mow_path",
                                "path_type": 0,
                                "total_path_num": 86,
                                "length": 12.5,
                            },
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[1.0, 2.0], [3.0, 4.0]],
                            },
                        }
                    ],
                },
                generated_mow_progress_geojson={
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {
                                "type_name": "mow_progress",
                                "path_hash": large_path_hash,
                                "is_active": True,
                                "now_index": 15,
                                "total_points": 28,
                            },
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
                            },
                        }
                    ],
                },
                generated_dynamics_line_geojson={
                    "type": "FeatureCollection",
                    "features": [],
                },
            ),
        ),
    )

    class FakeDeviceType:
        def is_support_dynamics_line(self, _: object) -> bool:
            return False

    monkeypatch.setattr(
        mammotion_services.DeviceType,
        "value_of_str",
        lambda _: FakeDeviceType(),
    )

    exported = _export_active_route(coordinator)

    assert exported["mow_path_feature_count"] == 1
    assert exported["mow_progress_feature_count"] == 1
    assert exported["mow_path_features"][0]["type_name"] == "mow_path"
    assert exported["mow_path_features"][0]["point_count"] == 2
    assert exported["mow_path_features"][0]["transaction_id"] == str(large_path_hash)
    assert exported["active_progress"]["type_name"] == "mow_progress"
    assert exported["active_progress"]["path_hash"] == str(large_path_hash)
    assert exported["active_progress"]["point_count"] == 3


def test_validate_custom_path_accepts_inside_map_xy_path() -> None:
    """Custom path validation accepts points inside known area geometry."""
    coordinator = _coordinator()
    coordinator.data.map.area = {
        123: SimpleNamespace(
            data=[
                SimpleNamespace(
                    current_frame=0,
                    data_couple=[
                        SimpleNamespace(x=0.0, y=0.0),
                        SimpleNamespace(x=10.0, y=0.0),
                        SimpleNamespace(x=10.0, y=10.0),
                        SimpleNamespace(x=0.0, y=10.0),
                    ],
                )
            ]
        )
    }

    result = _validate_custom_path(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 9.0, "y": 9.0}],
        area_hash=123,
        speed=0.2,
        blade_mode="off",
    )

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["coordinate_system"] == "mower_map_xy"
    assert result["distance"] > 0


def test_validate_custom_path_rejects_unsafe_or_outside_path() -> None:
    """Custom path validation reports outside geometry and blade-mode errors."""
    coordinator = _coordinator()
    coordinator.data.map.area = {
        123: SimpleNamespace(
            data=[
                SimpleNamespace(
                    current_frame=0,
                    data_couple=[
                        SimpleNamespace(x=0.0, y=0.0),
                        SimpleNamespace(x=10.0, y=0.0),
                        SimpleNamespace(x=10.0, y=10.0),
                        SimpleNamespace(x=0.0, y=10.0),
                    ],
                )
            ]
        )
    }

    result = _validate_custom_path(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 20.0, "y": 20.0}],
        area_hash=123,
        speed=0.5,
        blade_mode="on",
    )

    assert result["valid"] is False
    assert "blade_mode_must_be_off" in result["errors"]
    assert "path_points_outside_known_area_geometry" in result["errors"]
    assert "speed_above_recommended_validation_default" in result["warnings"]


def test_validate_custom_path_accepts_point_on_polygon_boundary() -> None:
    """Boundary points are valid because mower map areas are closed polygons."""
    coordinator = _coordinator()
    coordinator.data.map.area = {
        123: SimpleNamespace(
            data=[
                SimpleNamespace(
                    current_frame=0,
                    data_couple=[
                        SimpleNamespace(x=0.0, y=0.0),
                        SimpleNamespace(x=10.0, y=0.0),
                        SimpleNamespace(x=10.0, y=10.0),
                        SimpleNamespace(x=0.0, y=10.0),
                    ],
                )
            ]
        )
    }

    result = _validate_custom_path(
        coordinator,
        [{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 0.0}],
        area_hash=123,
    )

    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_custom_path_rejects_unknown_area_hash() -> None:
    """Unknown area hashes are hard failures."""
    coordinator = _coordinator()

    result = _validate_custom_path(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 2.0}],
        area_hash=123,
    )

    assert result["valid"] is False
    assert "area_hash_not_found" in result["errors"]


def test_validate_custom_path_warns_when_area_has_no_geometry() -> None:
    """Known areas without geometry warn instead of blocking validation."""
    coordinator = _coordinator()
    coordinator.data.map.area = {123: SimpleNamespace(data=[])}

    result = _validate_custom_path(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 2.0}],
        area_hash=123,
    )

    assert result["valid"] is True
    assert result["errors"] == []
    assert "no_area_geometry_available_for_containment_check" in result["warnings"]


def test_preview_custom_path_returns_geojson_line_and_markers() -> None:
    """Custom path preview returns validation plus display-ready GeoJSON."""
    coordinator = _coordinator()
    coordinator.data.map.area = {
        123: SimpleNamespace(
            data=[
                SimpleNamespace(
                    current_frame=0,
                    data_couple=[
                        SimpleNamespace(x=0.0, y=0.0),
                        SimpleNamespace(x=10.0, y=0.0),
                        SimpleNamespace(x=10.0, y=10.0),
                        SimpleNamespace(x=0.0, y=10.0),
                    ],
                )
            ]
        )
    }

    result = _preview_custom_path(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 9.0, "y": 9.0}],
        area_hash=123,
    )

    assert result["valid"] is True
    assert result["path"]["coordinate_system"] == "mower_map_xy"
    assert result["geojson"]["type"] == "FeatureCollection"
    assert [
        feature["geometry"]["type"] for feature in result["geojson"]["features"]
    ] == ["Point", "LineString", "Point"]
    assert result["geojson"]["features"][1]["geometry"]["coordinates"] == [
        [1.0, 1.0],
        [9.0, 9.0],
    ]


def test_preview_custom_path_includes_errors_for_invalid_path() -> None:
    """Preview still returns normalized metadata when validation fails."""
    coordinator = _coordinator()

    result = _preview_custom_path(coordinator, [{"x": 1.0, "y": 1.0}])

    assert result["valid"] is False
    assert "path_requires_at_least_two_points" in result["errors"]
    assert result["geojson"]["features"][0]["properties"]["marker"] == "start"


def test_dry_run_custom_path_builds_segments_without_allowing_execution() -> None:
    """Dry-run describes a possible controller plan but never allows movement."""
    coordinator = _coordinator()
    coordinator.is_online = lambda: True
    coordinator.data.map.area = {
        123: SimpleNamespace(
            data=[
                SimpleNamespace(
                    current_frame=0,
                    data_couple=[
                        SimpleNamespace(x=0.0, y=0.0),
                        SimpleNamespace(x=10.0, y=0.0),
                        SimpleNamespace(x=10.0, y=10.0),
                        SimpleNamespace(x=0.0, y=10.0),
                    ],
                )
            ]
        )
    }

    result = _dry_run_custom_path(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 4.0, "y": 1.0}, {"x": 4.0, "y": 5.0}],
        area_hash=123,
        speed=0.2,
    )

    assert result["valid"] is True
    assert result["dry_run"] is True
    assert result["real_execution_allowed"] is False
    assert result["reason_real_execution_blocked"] == (
        "firmware_waypoint_api_with_blades_off_not_proven"
    )
    assert result["segments"] == [
        {
            "index": 1,
            "start": {"x": 1.0, "y": 1.0},
            "end": {"x": 4.0, "y": 1.0},
            "distance": 3.0,
            "heading_degrees": 0.0,
            "estimated_seconds": 15.0,
        },
        {
            "index": 2,
            "start": {"x": 4.0, "y": 1.0},
            "end": {"x": 4.0, "y": 5.0},
            "distance": 4.0,
            "heading_degrees": 90.0,
            "estimated_seconds": 20.0,
        },
    ]
    assert result["estimated_total_seconds"] == 35.0
    assert result["candidate_existing_feature_plan"]["would_send"] is False
    assert result["safety_gates"][-1]["passed"] is False


def test_execute_custom_path_remains_blocked_even_when_requested_real() -> None:
    """Execution envelope performs readiness checks but still sends nothing."""
    coordinator = _coordinator()
    coordinator.is_online = lambda: True
    coordinator.data.map.area = {
        123: SimpleNamespace(
            data=[
                SimpleNamespace(
                    current_frame=0,
                    data_couple=[
                        SimpleNamespace(x=0.0, y=0.0),
                        SimpleNamespace(x=10.0, y=0.0),
                        SimpleNamespace(x=10.0, y=10.0),
                        SimpleNamespace(x=0.0, y=10.0),
                    ],
                )
            ]
        )
    }
    coordinator.data.mowing_state = SimpleNamespace(
        pos_x=1.0,
        pos_y=1.0,
        toward=0.0,
        pos_level=0,
        rtk_status=4,
        zone_hash=123,
        pos_type=1,
    )
    coordinator.data.report_data = SimpleNamespace(
        dev=SimpleNamespace(sys_status=11, charge_state=2, blade_state=0),
        rtk=SimpleNamespace(status=4, pos_level=0),
        locations=[],
        cutter_work_mode_info=SimpleNamespace(
            current_cutter_mode=0,
            current_cutter_rpm=0,
        ),
        connect=None,
    )

    result = _execute_custom_path(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 4.0, "y": 1.0}],
        area_hash=123,
        speed=0.2,
        dry_run=False,
        confirm_blades_off=True,
        allow_manual_velocity=True,
    )

    readiness = result["execution_readiness"]
    assert result["dry_run"] is False
    assert result["real_execution_allowed"] is False
    assert readiness["requested_real_execution"] is True
    assert readiness["can_execute_now"] is False
    assert readiness["start_distance"] == 0.0
    assert readiness["blockers"] == [
        "firmware_waypoint_api_proven",
        "dry_run_guard",
    ]
    assert result["manual_velocity_command_plan"]["would_send"] is False


def test_execute_custom_path_reports_operator_and_blade_blockers() -> None:
    """Readiness output shows the missing safety requirements."""
    coordinator = _coordinator()
    coordinator.is_online = lambda: True
    coordinator.data.map.area = {
        123: SimpleNamespace(
            data=[
                SimpleNamespace(
                    current_frame=0,
                    data_couple=[
                        SimpleNamespace(x=0.0, y=0.0),
                        SimpleNamespace(x=10.0, y=0.0),
                        SimpleNamespace(x=10.0, y=10.0),
                        SimpleNamespace(x=0.0, y=10.0),
                    ],
                )
            ]
        )
    }
    coordinator.data.report_data = SimpleNamespace(
        dev=SimpleNamespace(sys_status=11, charge_state=2, blade_state=1),
        rtk=SimpleNamespace(status=4, pos_level=0),
        locations=[],
        cutter_work_mode_info=SimpleNamespace(
            current_cutter_mode=0,
            current_cutter_rpm=1200,
        ),
        connect=None,
    )

    result = _execute_custom_path(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 4.0, "y": 1.0}],
        area_hash=123,
        speed=0.2,
    )

    assert result["real_execution_allowed"] is False
    assert result["execution_readiness"]["blockers"] == [
        "operator_confirmed_blades_off",
        "mower_reports_blades_off",
        "live_map_position_available",
        "manual_velocity_opt_in",
        "firmware_waypoint_api_proven",
    ]


def test_manual_velocity_controller_simulates_forward_when_heading_aligned() -> None:
    """Controller chooses a forward pulse when heading already faces target."""
    decision = _manual_velocity_controller_decision(
        [{"x": 1.0, "y": 1.0}, {"x": 4.0, "y": 1.0}],
        {
            "position": {
                "x": 1.0,
                "y": 1.0,
                "toward": 0.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
    )

    assert decision["mode"] == "simulated"
    assert decision["would_send"] is False
    assert decision["action"] == "forward"
    assert decision["reason"] == "heading_aligned"
    assert decision["target_index"] == 1
    assert decision["distance_to_target"] == 3.0
    assert decision["command_not_sent"] == {
        "service": "mammotion.move_forward",
        "data": {"speed": 0.2, "use_wifi": False},
    }


def test_manual_velocity_controller_simulates_turn_left_or_right() -> None:
    """Controller turns toward the next waypoint before moving forward."""
    points = [{"x": 1.0, "y": 1.0}, {"x": 4.0, "y": 1.0}]

    left = _manual_velocity_controller_decision(
        points,
        {
            "position": {
                "x": 1.0,
                "y": 1.0,
                "toward": 270.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
    )
    right = _manual_velocity_controller_decision(
        points,
        {
            "position": {
                "x": 1.0,
                "y": 1.0,
                "toward": 90.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
    )

    assert left["action"] == "turn_left"
    assert left["heading_error_degrees"] == 90.0
    assert left["command_not_sent"]["service"] == "mammotion.move_left"
    assert right["action"] == "turn_right"
    assert right["heading_error_degrees"] == -90.0
    assert right["command_not_sent"]["service"] == "mammotion.move_right"


def test_manual_velocity_controller_applies_heading_offset() -> None:
    """Heading offset corrects reported heading before choosing an action."""
    decision = _manual_velocity_controller_decision(
        [{"x": 1.0, "y": 1.0}, {"x": 4.0, "y": 1.0}],
        {
            "position": {
                "x": 1.0,
                "y": 1.0,
                "toward": 250.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
        heading_offset_degrees=110.0,
    )

    assert decision["current_heading_degrees"] == 250.0
    assert decision["corrected_heading_degrees"] == 0.0
    assert decision["heading_offset_degrees"] == 110.0
    assert decision["action"] == "forward"
    assert decision["reason"] == "heading_aligned"


def test_manual_velocity_best_heading_decision_selects_forward_candidate() -> None:
    """Candidate selection prefers an aligned forward command over a turn."""
    decision = _manual_velocity_best_heading_decision(
        [{"x": 1.0, "y": 1.0}, {"x": 4.0, "y": 1.0}],
        {
            "position": {
                "x": 1.0,
                "y": 1.0,
                "toward": 0.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
        heading_offset_degrees=110.0,
        heading_offset_candidates=[110.0, 0.0, 90.0],
    )

    assert decision["action"] == "forward"
    assert decision["reason"] == "heading_aligned"
    assert decision["selected_heading_offset_degrees"] == 0.0
    assert decision["heading_offset_candidates"] == [110.0, 0.0, 90.0]
    assert [item["heading_offset_degrees"] for item in decision["heading_offset_diagnostics"]] == [
        110.0,
        0.0,
        90.0,
    ]


def test_manual_velocity_controller_skips_stale_start_waypoint() -> None:
    """Controller does not turn back to an obsolete drawn start point."""
    decision = _manual_velocity_controller_decision(
        [{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 0.08}],
        {
            "position": {
                "x": 0.0,
                "y": 0.04,
                "toward": 90.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
        waypoint_tolerance=0.03,
    )

    assert decision["target_index"] == 1
    assert decision["action"] == "forward"
    assert decision["reason"] == "heading_aligned"


def test_manual_velocity_controller_skips_start_after_segment_progress() -> None:
    """Controller skips start once mower has positive projection on first segment."""
    decision = _manual_velocity_controller_decision(
        [{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": -0.16}],
        {
            "position": {
                "x": 0.002,
                "y": -0.053,
                "toward": 270.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
        waypoint_tolerance=0.03,
    )

    assert decision["target_index"] == 1
    assert decision["action"] == "forward"


def test_manual_velocity_controller_uses_later_segment_projection() -> None:
    """Controller targets the next point on the closest later path segment."""
    decision = _manual_velocity_controller_decision(
        [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 0.0},
            {"x": 1.0, "y": 1.0},
        ],
        {
            "position": {
                "x": 1.02,
                "y": 0.45,
                "toward": 90.0,
                "source": "mowing_state",
            }
        },
        speed=0.2,
        waypoint_tolerance=0.03,
    )

    assert decision["target_index"] == 2
    assert decision["action"] == "forward"


def test_manual_velocity_controller_keeps_target_after_start_progress() -> None:
    """Controller keeps targeting endpoint after forward progress along segment."""
    decision = _manual_velocity_controller_decision(
        [
            {"x": 4.5424, "y": -0.9319},
            {"x": 4.587795864039179, "y": -1.0853249508005016},
        ],
        {
            "position": {
                "x": 4.5447,
                "y": -0.9849,
                "toward": 176.4826,
                "source": "mowing_state",
            }
        },
        speed=0.4,
        waypoint_tolerance=0.03,
        heading_offset_degrees=110.0,
    )

    assert decision["target_index"] == 1
    assert decision["action"] == "forward"
    assert abs(decision["heading_error_degrees"]) < 15


def test_manual_velocity_path_progress_requires_target_direction() -> None:
    """Forward progress must project toward the active target."""
    before = {
        "position": {
            "x": 0.0,
            "y": 0.0,
            "toward": 0.0,
            "source": "mowing_state",
        }
    }
    decision = {
        "action": "forward",
        "target": {"x": 1.0, "y": 0.0},
    }

    toward = _manual_velocity_path_progress_diagnostic(
        before,
        {
            "position": {
                "x": 0.1,
                "y": 0.0,
                "toward": 0.0,
                "source": "mowing_state",
            }
        },
        decision,
        min_progress_distance=0.02,
        min_heading_change_degrees=1.0,
    )
    away = _manual_velocity_path_progress_diagnostic(
        before,
        {
            "position": {
                "x": -0.1,
                "y": 0.0,
                "toward": 0.0,
                "source": "mowing_state",
            }
        },
        decision,
        min_progress_distance=0.02,
        min_heading_change_degrees=1.0,
    )

    assert toward["passed"] is True
    assert toward["status"] == "path_progress"
    assert away["passed"] is False
    assert away["status"] == "no_path_progress"


def test_manual_velocity_controller_stops_without_live_position() -> None:
    """Controller refuses to plan movement without live map-local position."""
    decision = _manual_velocity_controller_decision(
        [{"x": 1.0, "y": 1.0}, {"x": 4.0, "y": 1.0}],
        {"position": {"x": None, "y": None, "toward": None, "source": "unavailable"}},
        speed=0.2,
    )

    assert decision["action"] == "stop"
    assert decision["reason"] == "live_position_unavailable"
    assert decision["command_not_sent"] is None


def test_manual_velocity_pulse_schema_allows_emergency_nudge_speed() -> None:
    """Pulse probe allows the existing emergency nudge speed but no higher."""
    parsed = MANUAL_VELOCITY_PULSE_TEST_SCHEMA(
        {
            "entity_id": "lawn_mower.test",
            "action": "forward",
            "speed": 0.4,
        }
    )
    assert parsed["speed"] == 0.4
    assert parsed["stop_mode"] == "immediate"
    assert parsed["post_command_sample_delays"] == [0.0, 2.0, 10.0, 30.0, 60.0]
    with pytest.raises(Exception):  # noqa: B017
        MANUAL_VELOCITY_PULSE_TEST_SCHEMA(
            {
                "entity_id": "lawn_mower.test",
                "action": "forward",
                "speed": 0.45,
            }
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
                "min_progress_distance": 0.01,
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
                "heading_tolerance_degrees": 3.0,
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
                "linear_speed_fast": 400,
                "linear_speed_slow": 200,
                "angular_speed_fast": 180,
                "angular_speed_slow": 180,
                "calibrated_forward_heading_offset_degrees": 116.5,
                "max_turn_commands": 3,
                "max_linear_commands": 1,
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
                "max_real_segments": 1,
                "max_turn_commands": 4,
                "max_linear_commands": 2,
                "calibrated_forward_heading_offset_degrees": 116.5,
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


@pytest.mark.asyncio
async def test_raw_pymammotion_motion_probe_defaults_to_dry_run() -> None:
    """Raw pymammotion probe default sends no command and reports exact call."""
    coordinator = _pulse_coordinator()

    result = await _raw_pymammotion_motion_probe(coordinator, sample_delays=())

    assert result["service"] == "raw_pymammotion_motion_probe"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["reason"] == "dry_run"
    assert result["command_not_sent"] == {
        "manager_method": "send_command_with_args",
        "device_name": "Luba-Test",
        "command": "send_movement",
        "prefer_ble": True,
        "kwargs": {"linear_speed": 400, "angular_speed": 0},
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_forward_two_pulse_latency_test_defaults_to_dry_run() -> None:
    """Two-pulse latency test default sends no raw movement commands."""
    coordinator = _pulse_coordinator()

    result = await _forward_two_pulse_latency_test(coordinator)

    assert result["service"] == "forward_two_pulse_latency_test"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["reason"] == "dry_run"
    assert result["command_not_sent"] == {
        "manager_method": "send_command_with_args",
        "device_name": "Luba-Test",
        "command": "send_movement",
        "prefer_ble": True,
        "kwargs": {"linear_speed": 200, "angular_speed": 0},
    }
    assert len(result["commands"]) == 2
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_position_feedback_diagnostic_defaults_to_dry_run() -> None:
    """Position feedback diagnostic default captures sources but sends nothing."""
    coordinator = _pulse_coordinator()

    result = await _position_feedback_diagnostic(coordinator)

    assert result["service"] == "position_feedback_diagnostic"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["reason"] == "dry_run"
    assert result["snapshots"][0]["raw_sources"]["paths"]["mowing_state.pos_x"] == 1.0
    assert result["snapshots"][0]["raw_sources"]["handle"]["active_transport"] == "ble"
    assert result["refresh_attempts"] == []
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_forward_two_pulse_latency_test_rejects_missing_confirmations() -> None:
    """Real two-pulse latency test requires explicit operator confirmations."""
    coordinator = _pulse_coordinator()

    result = await _forward_two_pulse_latency_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=False,
    )

    assert result["would_send"] is False
    assert result["reason"] == "safety_gates_failed"
    assert "operator_confirmed_clear_area" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_position_feedback_diagnostic_rejects_missing_confirmations() -> None:
    """Position feedback diagnostic requires confirmations before real pulses."""
    coordinator = _pulse_coordinator()

    result = await _position_feedback_diagnostic(
        coordinator,
        dry_run=False,
        pulse_count=1,
        confirm_blades_off=True,
        confirm_clear_area=False,
    )

    assert result["would_send"] is False
    assert result["reason"] == "safety_gates_failed"
    assert "operator_confirmed_clear_area" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vio_motion_probe_defaults_to_dry_run() -> None:
    """VIO motion probe default captures a baseline but sends nothing."""
    coordinator = _pulse_coordinator()

    result = await _vio_motion_probe(coordinator)

    assert result["service"] == "vio_motion_probe"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["reason"] == "dry_run"
    assert result["command"]["kwargs"] == {"linear_speed": 200, "angular_speed": 0}
    assert result["samples"] == []
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_vio_motion_probe_rejects_missing_confirmations() -> None:
    """Real VIO motion probe requires explicit operator confirmations."""
    coordinator = _pulse_coordinator()

    result = await _vio_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=False,
    )

    assert result["would_send"] is False
    assert result["reason"] == "safety_gates_failed"
    assert "operator_confirmed_clear_area" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_vio_motion_probe_drives_samples_vio_and_always_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real VIO probe sends one continuous command, samples VIO, and always stops."""
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    original_snapshot = mammotion_services._custom_path_telemetry_snapshot  # noqa: SLF001

    def fake_snapshot(
        coordinator_arg: MammotionReportUpdateCoordinator,
    ) -> dict:
        telemetry = original_snapshot(coordinator_arg)
        # Simulate forward motion: map-local y drifts as the clock advances.
        moved = (clock["now"] - 100.0) * 0.05
        telemetry["position"]["y"] = float(telemetry["position"]["y"]) - moved
        return telemetry

    async def fake_get_reports(count: int = 5) -> None:
        # VIO initializes once the mower has been moving for a moment.
        if clock["now"] >= 101.0:
            coordinator.data.report_data.vision_info = SimpleNamespace(
                heading=42.0,
                vio_state=2,
            )

    monkeypatch.setattr(mammotion_services.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        mammotion_services,
        "_custom_path_telemetry_snapshot",
        fake_snapshot,
    )
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        drive_seconds=3.0,
        sample_interval_seconds=1.0,
        post_stop_samples=1,
    )

    # A single continuous velocity command, not one command per sample.
    assert coordinator.manager.send_command_with_args.await_count == 1
    # The explicit stop is mandatory even on the happy path.
    coordinator.async_stop_manual_motion.assert_awaited_once()
    assert result["command_ok"] is True
    assert result["reason"] == "vio_initialized_during_motion"
    assert result["verdict"]["motion_confirmed"] is True
    assert result["verdict"]["vio_activated_while_moving"] is True
    assert 42.0 in result["verdict"]["heading_series"]
    assert result["samples"]


@pytest.mark.asyncio
async def test_vio_turn_probe_defaults_to_dry_run() -> None:
    """VIO turn probe default plans an in-place rotation but sends nothing."""
    coordinator = _pulse_coordinator()

    result = await _vio_turn_probe(coordinator)

    assert result["service"] == "vio_turn_probe"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["reason"] == "dry_run"
    assert result["command"]["kwargs"] == {"linear_speed": 0, "angular_speed": 180}
    assert result["samples"] == []
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_vio_turn_probe_detects_heading_tracking_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vision heading moving while course-over-ground is frozen tracks rotation."""
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    async def fake_get_reports(count: int = 5) -> None:
        # VIO heading rotates 10 deg/s; position and course-over-ground stay put.
        heading = (clock["now"] - 100.0) * 10.0
        coordinator.data.report_data.vision_info = SimpleNamespace(
            heading=heading,
            vio_state=2,
        )

    monkeypatch.setattr(mammotion_services.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        angular_speed=180,
        drive_seconds=3.0,
        sample_interval_seconds=1.0,
        post_stop_samples=0,
    )

    # A single continuous angular command, then a mandatory explicit stop.
    assert coordinator.manager.send_command_with_args.await_count == 1
    assert result["command"]["kwargs"] == {"linear_speed": 0, "angular_speed": 180}
    coordinator.async_stop_manual_motion.assert_awaited_once()
    assert result["reason"] == "vision_heading_tracks_rotation"
    assert result["verdict"]["vision_heading_change"]["total_abs_degrees"] >= 3.0
    assert result["verdict"]["course_over_ground_change"]["total_abs_degrees"] == 0.0


@pytest.mark.asyncio
async def test_vio_turn_to_heading_defaults_to_dry_run() -> None:
    """VIO turn-to-heading default plans a turn (opposite sign of error), no send."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    result = await _vio_turn_to_heading(coordinator, target_vision_heading=40.0)

    assert result["service"] == "vio_turn_to_heading"
    assert result["dry_run"] is True
    assert result["stop_reason"] == "dry_run"
    assert result["initial_heading_error_degrees"] == 40.0
    # Positive error -> negative angular (calibrated: -angular increases heading).
    assert result["planned_command"]["kwargs"]["angular_speed"] == -500
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_vio_turn_to_heading_rejects_missing_confirmations() -> None:
    """Real VIO turn-to-heading requires explicit operator confirmations."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=False,
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "operator_confirmed_clear_area" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vio_turn_to_heading_cold_vio_still_allows_dry_run() -> None:
    """A cold VIO (vio_state != 2) still plans in dry-run without sending."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=0)

    result = await _vio_turn_to_heading(coordinator, target_vision_heading=40.0)

    assert result["stop_reason"] == "dry_run"
    assert result["initial_vio_state"] == 0
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vio_turn_to_heading_refuses_real_turn_when_vio_cold() -> None:
    """Real VIO turn-to-heading refuses to move unless VIO is actively tracking."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=0)

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "vio_active" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


def test_vio_feed_liveness_gates_on_tracked_features() -> None:
    """The feed reads degraded only when a reported feature count is below the floor."""
    coordinator = _pulse_coordinator()

    # Healthy daylight feed.
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=2, track_feature_num=80, brightness=200
    )
    healthy = _vio_feed_liveness(coordinator)
    assert healthy["live"] is True
    assert healthy["tracked_features"] == 80
    assert healthy["brightness_label"] == "Light"

    # Dusk latch: vio_state stays active but the track collapsed to 0 features.
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=2, track_feature_num=0, brightness=10
    )
    degraded = _vio_feed_liveness(coordinator)
    assert degraded["live"] is False
    assert degraded["tracked_features"] == 0
    assert degraded["brightness_label"] == "Dark"

    # Devices that never report a feature count must not be blocked.
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)
    assert _vio_feed_liveness(coordinator)["live"] is True


@pytest.mark.asyncio
async def test_vio_turn_to_heading_blocks_real_turn_when_feed_degraded() -> None:
    """vio_state==2 with a collapsed feature track blocks a real turn (dusk latch)."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=2, track_feature_num=0, brightness=10
    )

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "vio_feed_live" in result["blockers"]
    assert result["initial_vio_feed"]["live"] is False
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_vio_turn_to_heading_stops_when_feed_degrades_mid_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-turn feature-track collapse stops distinctly from vio_state dropping out."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=2, track_feature_num=80, brightness=200
    )

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        # Pulse one makes real progress, then the feed goes blind (sunset): the
        # track drops to 0 features while vio_state stays active and the heading
        # would otherwise latch. The next iteration must bail on the blind feed.
        vi = coordinator.data.report_data.vision_info
        vi.heading = vi.heading + 10.0
        vi.track_feature_num = 0
        vi.brightness = 10

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "vio_feed_degraded"
    assert result["final_vio_feed"]["live"] is False
    assert result["commands_sent"] == 1
    assert coordinator.async_stop_manual_motion.await_count == 1


@pytest.mark.asyncio
async def test_vio_turn_to_heading_stops_if_vio_drops_out_mid_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If VIO deactivates during the loop, stop instead of chasing a stale heading."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        # Pulse one makes real progress, but VIO drops out (enters shadow) so the
        # next iteration must bail rather than trust the now-stale heading.
        vi = coordinator.data.report_data.vision_info
        vi.heading = vi.heading + 10.0
        vi.vio_state = 0

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "vio_inactive"
    # Exactly one pulse fired before VIO dropped out and the loop bailed.
    assert result["commands_sent"] == 1
    assert coordinator.async_stop_manual_motion.await_count == 1


@pytest.mark.asyncio
async def test_vio_turn_to_heading_closed_loop_reaches_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded pulses converge vision_heading to the target and stop each pulse."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        vi = coordinator.data.report_data.vision_info
        vi.heading = min(30.0, vi.heading + 10.0)

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=30.0,
        heading_tolerance_degrees=8.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "target_heading_reached"
    assert result["commands_sent"] == 3
    # First pulse: error +30 -> negative angular per calibration.
    assert result["command_results"][0]["angular_speed"] == -500
    # A bounded pulse + explicit stop per command.
    assert coordinator.manager.send_command_with_args.await_count == 3
    assert coordinator.async_stop_manual_motion.await_count == 3
    assert abs(result["final_heading_error_degrees"]) <= 8.0


@pytest.mark.asyncio
async def test_vio_turn_to_heading_polls_through_stale_heading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale first sample is re-polled to a fresh heading, not judged as progress-less."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    # The VIO feed lags ~4s: the first request_reports after a pulse returns the
    # pre-pulse heading jittered only by sub-epsilon sensor noise (stale); only the
    # second poll reflects the real rotation. The loop must poll through the stale
    # sample rather than treat the noise wiggle as fresh movement.
    calls = {"n": 0}

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        calls["n"] += 1
        vi = coordinator.data.report_data.vision_info
        if calls["n"] % 2 == 0:  # advance only on the second poll of each pulse
            vi.heading = min(30.0, round(vi.heading) + 10.0)
        else:  # first poll: latched value plus sub-epsilon noise
            vi.heading = round(vi.heading) + 0.002

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=30.0,
        heading_tolerance_degrees=8.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "target_heading_reached"
    assert result["commands_sent"] == 3
    # Each pulse polled twice (stale then fresh) before judging progress.
    assert all(cmd["heading_went_fresh"] for cmd in result["command_results"])
    assert coordinator.async_get_reports.await_count == 6


@pytest.mark.asyncio
async def test_vio_turn_to_heading_tolerates_one_stale_pulse_before_no_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-progress only aborts after max_no_progress_pulses consecutive stale pulses."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)
    clock = {"now": 100.0}
    calls = {"flip": False}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    async def fake_get_reports(count: int = 5) -> None:
        # Heading is permanently latched but the feed still emits sub-epsilon sensor
        # noise (run 2, dusk: ~0.0018 deg jitter). The fresh-heading poll must treat
        # that as still-stale, time out, and keep progress at zero.
        vi = coordinator.data.report_data.vision_info
        vi.heading = round(vi.heading, 3) + (0.0018 if calls["flip"] else -0.0018)
        calls["flip"] = not calls["flip"]

    monkeypatch.setattr(mammotion_services.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "no_heading_progress"
    # One stale pulse is tolerated; the second consecutive no-progress pulse aborts.
    assert result["commands_sent"] == 2
    assert coordinator.async_stop_manual_motion.await_count == 2
    assert all(not cmd["heading_went_fresh"] for cmd in result["command_results"])
    assert result["command_results"][-1]["consecutive_no_progress"] == 2
    # First pulse runs full-length; the second, fired after a *stale* no-progress
    # sample, is capped to the slow duration to bound blind rotation on a latched
    # feed.
    assert result["command_results"][0]["pulse_duration_ms"] == 1500
    assert result["command_results"][1]["pulse_duration_ms"] == 700


@pytest.mark.asyncio
async def test_vio_turn_to_heading_streak_keeps_full_pulse_when_sample_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh-but-stalled no-progress streak keeps the full pulse (not the slow cap).

    The slow-duration cap exists to bound blind rotation on a latched feed, so it
    only applies when the streak's last sample was stale. Here the heading moves
    the wrong way each pulse (fresh reading, negative progress): the feed is not
    blind, so the full pulse -- the faster correction -- must be kept.
    """
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        # Each poll returns a genuinely fresh reading that drifts *away* from the
        # +40 target, so heading_went_fresh is True but progress is negative.
        vi = coordinator.data.report_data.vision_info
        vi.heading = vi.heading - 10.0

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "no_heading_progress"
    assert all(cmd["heading_went_fresh"] for cmd in result["command_results"])
    # Fresh feed -> the streak never forces the slow cap; every pulse is full-length.
    assert all(
        cmd["pulse_duration_ms"] == 1500 for cmd in result["command_results"]
    )


@pytest.mark.asyncio
async def test_vio_turn_to_heading_sub_epsilon_wiggle_is_not_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 0.002 deg feed wiggle must not pass the freshness gate (run 2 regression).

    Run 2 (dusk) latched the heading bit-identical while the feed still jittered by
    ~0.0018 deg; the old float-inequality check read that noise as movement. With the
    epsilon gate the poll must treat a 0.002 deg wiggle as stale, time out, and abort
    on no progress instead of trusting the blind feed.
    """
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)
    clock = {"now": 100.0}
    flip = {"v": False}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    async def fake_get_reports(count: int = 5) -> None:
        vi = coordinator.data.report_data.vision_info
        # Oscillate by +/-0.002 deg around the latched value: never clears the
        # 0.1 deg freshness epsilon.
        vi.heading = 0.002 if flip["v"] else 0.0
        flip["v"] = not flip["v"]

    monkeypatch.setattr(mammotion_services.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "no_heading_progress"
    assert all(not cmd["heading_went_fresh"] for cmd in result["command_results"])
    assert all(
        abs(cmd["measured_change_degrees"]) <= _VIO_HEADING_FRESH_EPSILON_DEGREES
        for cmd in result["command_results"]
    )


@pytest.mark.asyncio
async def test_forward_two_pulse_latency_test_sends_pulses_and_detects_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real two-pulse latency test sends exactly two forward pulses."""
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    original_snapshot = mammotion_services._custom_path_telemetry_snapshot  # noqa: SLF001

    def fake_snapshot(
        coordinator_arg: MammotionReportUpdateCoordinator,
    ) -> dict:
        telemetry = original_snapshot(coordinator_arg)
        if clock["now"] >= 112.0:
            telemetry["position"]["y"] = float(telemetry["position"]["y"]) - 0.02
        return telemetry

    monkeypatch.setattr(mammotion_services.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        mammotion_services,
        "_custom_path_telemetry_snapshot",
        fake_snapshot,
    )

    result = await _forward_two_pulse_latency_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        pulse_count=3,
        telemetry_timeout_seconds=10,
        telemetry_sample_interval_seconds=1,
    )

    assert result["reason"] == "telemetry_position_change_detected"
    assert len(result["commands"]) == 3
    assert result["commands"][0]["kwargs"] == {
        "linear_speed": 200,
        "angular_speed": 0,
    }
    assert result["telemetry"]["first_position_change_after_command_1_seconds"] == 12.0
    assert result["telemetry"]["first_position_change_after_command_2_seconds"] == 7.0
    assert (
        result["telemetry"]["first_position_change_after_final_command_seconds"]
        == pytest.approx(2.0)
    )
    assert result["telemetry"]["final_delta"]["distance"] == pytest.approx(0.02)
    assert coordinator.manager.send_command_with_args.await_count == 3


@pytest.mark.asyncio
async def test_position_feedback_diagnostic_runs_refresh_attempts_and_detects_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Position feedback diagnostic compares all snapshots after refresh paths."""
    coordinator = _pulse_coordinator()

    async def fake_sleep(_delay: float) -> None:
        return None

    async def mutate_report_snapshot() -> None:
        coordinator.data.mowing_state.pos_y = 1.25

    coordinator.async_request_report_snapshot.side_effect = mutate_report_snapshot
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)

    result = await _position_feedback_diagnostic(
        coordinator,
        dry_run=False,
        pulse_count=0,
        refresh_wait_seconds=0.1,
    )

    assert result["reason"] == "position_source_changed"
    assert result["position_source_changed"] is True
    assert "telemetry.position" in result["changed_sources"]
    assert "raw_sources.paths" in result["changed_sources"]
    assert "telemetry.position" in result["position_changed_sources"]
    assert result["metadata_changed_sources"] == []
    assert [attempt["name"] for attempt in result["refresh_attempts"]] == [
        "request_report_snapshot",
        "request_reports_count_5",
        "start_report_stream",
        "request_iot_sync_one_shot",
        "request_iot_sync_continuous_window",
        "ensure_fresh_state_forced",
        "ble_sync_type_3",
        "ha_request_refresh",
    ]
    assert all(attempt["ok"] for attempt in result["refresh_attempts"])
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_get_reports.assert_awaited_once_with(count=5)
    coordinator.async_start_report_stream.assert_awaited_once_with(duration_ms=60_000)
    coordinator.manager.request_iot_sync.assert_awaited_once_with("Luba-Test")
    coordinator.manager.request_iot_sync_continuous.assert_awaited_once_with(
        "Luba-Test",
        period=1000,
        no_change_period=4000,
    )
    coordinator.manager.request_iot_sync_continuous_stop.assert_awaited_once_with(
        "Luba-Test"
    )
    coordinator.manager.ensure_fresh_state.assert_awaited_once_with(
        "Luba-Test",
        max_age_s=0.0,
    )
    coordinator.async_send_command.assert_awaited_once_with(
        "send_todev_ble_sync",
        prefer_ble=True,
        sync_type=3,
    )
    coordinator.async_request_refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_position_feedback_diagnostic_handle_only_change_is_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handle timestamp changes are metadata, not proof of position feedback."""
    coordinator = _pulse_coordinator()

    async def fake_sleep(_delay: float) -> None:
        return None

    calls = {"count": 0}

    def mower(_device_name: str) -> SimpleNamespace:
        calls["count"] += 1
        return SimpleNamespace(
            last_report_at=float(calls["count"]),
            availability=SimpleNamespace(
                mqtt_reported_offline=False,
            ),
            get_transport=lambda _transport_type: SimpleNamespace(
                _connect_cooldown_until=0.0
            ),
            active_transport=lambda: "ble",
        )

    coordinator.manager.mower = mower
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)

    result = await _position_feedback_diagnostic(
        coordinator,
        dry_run=False,
        pulse_count=0,
        refresh_wait_seconds=0.1,
    )

    assert result["reason"] == "metadata_source_changed"
    assert result["position_source_changed"] is False
    assert result["position_changed_sources"] == []
    assert result["metadata_changed_sources"] == ["raw_sources.handle"]


@pytest.mark.asyncio
async def test_position_feedback_diagnostic_sends_optional_pulses() -> None:
    """Position feedback diagnostic can send a bounded pulse burst when approved."""
    coordinator = _pulse_coordinator()

    result = await _position_feedback_diagnostic(
        coordinator,
        dry_run=False,
        pulse_count=1,
        refresh_wait_seconds=0,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["commands"][0]["ok"] is True
    assert result["commands"][0]["kwargs"] == {
        "linear_speed": 200,
        "angular_speed": 0,
    }
    coordinator.manager.send_command_with_args.assert_awaited_once_with(
        "Luba-Test",
        "send_movement",
        prefer_ble=True,
        linear_speed=200,
        angular_speed=0,
    )


@pytest.mark.asyncio
async def test_raw_pymammotion_motion_probe_rejects_missing_confirmations() -> None:
    """Real raw probe rejects missing operator confirmations before command."""
    coordinator = _pulse_coordinator()

    result = await _raw_pymammotion_motion_probe(
        coordinator,
        dry_run=False,
        sample_delays=(),
    )

    assert result["would_send"] is False
    assert result["reason"] == "safety_gates_failed"
    assert result["blockers"] == [
        "operator_confirmed_blades_off",
        "operator_confirmed_clear_area",
    ]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_motion_probe_rejects_unsafe_blades() -> None:
    """Real raw probe rejects unsafe blade telemetry before command."""
    coordinator = _pulse_coordinator(blade_state=1, cutter_rpm=0)

    result = await _raw_pymammotion_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(),
    )

    assert result["would_send"] is False
    assert result["blockers"] == ["mower_reports_blades_off"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_motion_probe_sends_raw_movement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw send_movement passes integer speeds through to pymammotion."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_motion_probe(
        coordinator,
        command="send_movement",
        linear_speed=-400,
        angular_speed=180,
        prefer_ble=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["command_result"]["ok"] is True
    coordinator.manager.send_command_with_args.assert_awaited_once_with(
        "Luba-Test",
        "send_movement",
        prefer_ble=True,
        linear_speed=-400,
        angular_speed=180,
    )


@pytest.mark.asyncio
async def test_raw_pymammotion_motion_probe_sends_wrapper_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrapper commands pass through to pymammotion without HA motion wrappers."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_motion_probe(
        coordinator,
        command="move_left",
        speed=0.4,
        prefer_ble=False,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["command_result"]["ok"] is True
    coordinator.manager.send_command_with_args.assert_awaited_once_with(
        "Luba-Test",
        "move_left",
        prefer_ble=False,
        angular=0.4,
    )


@pytest.mark.asyncio
async def test_raw_pymammotion_motion_probe_reports_telemetry_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw probe reports movement interpretation from sampled telemetry."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        coordinator.data.mowing_state.pos_x = 1.0
        coordinator.data.mowing_state.pos_y = 0.5

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["motion_interpretation"]["status"] == "translation_detected"
    assert result["motion_interpretation"]["delta"]["distance"] == pytest.approx(0.5)
    assert result["motion_interpretation"]["movement_heading_degrees"] == 270.0


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_dry_run_negative_y() -> None:
    """Negative-Y segment selects positive raw linear speed and sends nothing."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.7}],
        sample_delays=(),
    )

    assert result["service"] == "raw_pymammotion_execute_segment"
    assert result["dry_run"] is True
    assert result["stop_reason"] == "dry_run"
    assert result["would_send"] is False
    assert result["selected_axis"] == "map_y"
    assert result["initial_command_selection"]["linear_speed"] == 400
    assert result["command_not_sent"]["kwargs"] == {
        "linear_speed": 400,
        "angular_speed": 0,
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_dry_run_positive_y() -> None:
    """Positive-Y segment selects negative raw linear speed."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 1.3}],
        sample_delays=(),
    )

    assert result["stop_reason"] == "dry_run"
    assert result["initial_command_selection"]["linear_speed"] == -400


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_uses_slow_speed_near_target() -> None:
    """Remaining Y distance below threshold selects slow raw speed."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.88}],
        sample_delays=(),
    )

    assert result["stop_reason"] == "dry_run"
    assert result["initial_command_selection"]["linear_speed"] == 200
    assert result["initial_command_selection"]["speed_tier"] == "slow"


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_rejects_lateral_segment() -> None:
    """Part 1 rejects segments that need unproven lateral/turning motion."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.5, "y": 0.95}],
        sample_delays=(),
    )

    assert result["stop_reason"] == "segment_requires_lateral_or_turning_motion"
    assert result["lateral_diagnostic"]["passed"] is False
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_rejects_missing_confirmations() -> None:
    """Real raw segment rejects missing operator confirmations."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.7}],
        dry_run=False,
        sample_delays=(),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert result["blockers"] == [
        "operator_confirmed_blades_off",
        "operator_confirmed_clear_area",
    ]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_sends_one_raw_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real raw segment sends one send_movement command and accepts progress."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        coordinator.data.mowing_state.pos_y = 0.9

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.9}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 1
    assert result["stop_reason"] == "target_reached"
    assert result["completion_status"]["complete"] is True
    assert result["progress_diagnostics"][0]["passed"] is True
    coordinator.manager.send_command_with_args.assert_awaited_once_with(
        "Luba-Test",
        "send_movement",
        prefer_ble=True,
        linear_speed=200,
        angular_speed=0,
    )


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_stops_on_no_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real raw segment stops when delayed telemetry shows no target progress."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.7}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 1
    assert result["stop_reason"] == "no_target_progress"
    assert result["progress_diagnostics"][0]["status"] == "no_path_progress"


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_stops_after_max_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real raw segment stops after capped commands when progress is insufficient."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    positions = [0.985, 0.97, 0.955, 0.94]

    async def no_sleep(_: float) -> None:
        if positions:
            coordinator.data.mowing_state.pos_y = positions.pop(0)

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.5}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_commands=2,
        min_progress_distance=0.01,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 2
    assert result["stop_reason"] == "max_commands_reached"
    assert result["progress_diagnostics"][-1]["passed"] is True


@pytest.mark.asyncio
async def test_raw_pymammotion_angular_calibration_defaults_to_dry_run() -> None:
    """Raw angular calibration dry-run reports the exact command not sent."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_angular_calibration(
        coordinator,
        sample_delays=(),
    )

    assert result["service"] == "raw_pymammotion_angular_calibration"
    assert result["stop_reason"] == "dry_run"
    assert result["would_send"] is False
    assert result["command_not_sent"] == {
        "manager_method": "send_command_with_args",
        "device_name": "Luba-Test",
        "command": "send_movement",
        "prefer_ble": True,
        "kwargs": {"linear_speed": 0, "angular_speed": 180},
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_angular_calibration_negative_direction_dry_run() -> None:
    """Negative heading direction selects negative raw angular speed."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_angular_calibration(
        coordinator,
        direction="negative_heading",
        sample_delays=(),
    )

    assert result["stop_reason"] == "dry_run"
    assert result["initial_command_selection"]["angular_speed"] == -180


@pytest.mark.asyncio
async def test_raw_pymammotion_angular_calibration_rejects_missing_confirmations() -> None:
    """Real raw angular calibration rejects missing operator confirmations."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_angular_calibration(
        coordinator,
        dry_run=False,
        sample_delays=(),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert result["blockers"] == [
        "operator_confirmed_blades_off",
        "operator_confirmed_clear_area",
    ]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_angular_calibration_sends_raw_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real raw angular calibration sends one angular command and reaches target."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        coordinator.data.mowing_state.toward = 12.0

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_angular_calibration(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        target_heading_delta_degrees=10.0,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 1
    assert result["stop_reason"] == "target_heading_reached"
    assert result["target_status"]["complete"] is True
    assert result["heading_diagnostics"][0]["passed"] is True
    coordinator.manager.send_command_with_args.assert_awaited_once_with(
        "Luba-Test",
        "send_movement",
        prefer_ble=True,
        linear_speed=0,
        angular_speed=180,
    )


@pytest.mark.asyncio
async def test_raw_pymammotion_angular_calibration_stops_on_no_heading_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real raw angular calibration stops when heading telemetry does not move."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_angular_calibration(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 1
    assert result["stop_reason"] == "no_heading_progress"
    assert result["heading_diagnostics"][0]["status"] == "wrong_heading_direction"


@pytest.mark.asyncio
async def test_raw_pymammotion_angular_calibration_stops_after_max_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real raw angular calibration stops at the command cap with progress."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    headings = [2.0, 4.0]

    async def no_sleep(_: float) -> None:
        if headings:
            coordinator.data.mowing_state.toward = headings.pop(0)

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_angular_calibration(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        target_heading_delta_degrees=10.0,
        max_commands=2,
        min_heading_change_degrees=1.0,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 2
    assert result["stop_reason"] == "max_commands_reached"
    assert result["heading_diagnostics"][-1]["passed"] is True
    assert result["target_status"]["target_direction_progress_degrees"] == 4.0


@pytest.mark.asyncio
async def test_raw_pymammotion_turn_to_heading_dry_run_positive_direction() -> None:
    """Dry-run chooses positive angular speed for positive shortest error."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_turn_to_heading(
        coordinator,
        target_heading_degrees=20.0,
        sample_delays=(),
    )

    assert result["service"] == "raw_pymammotion_turn_to_heading"
    assert result["stop_reason"] == "dry_run"
    assert result["would_send"] is False
    assert result["heading_status"]["heading_error_degrees"] == 20.0
    assert result["initial_command_selection"]["angular_speed"] == 180
    assert result["command_not_sent"]["kwargs"] == {
        "linear_speed": 0,
        "angular_speed": 180,
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_turn_to_heading_dry_run_negative_direction() -> None:
    """Dry-run chooses negative angular speed for negative shortest error."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_turn_to_heading(
        coordinator,
        target_heading_degrees=350.0,
        sample_delays=(),
    )

    assert result["stop_reason"] == "dry_run"
    assert result["heading_status"]["heading_error_degrees"] == -10.0
    assert result["initial_command_selection"]["angular_speed"] == -180


@pytest.mark.asyncio
async def test_raw_pymammotion_turn_to_heading_uses_slow_speed_near_target() -> None:
    """Dry-run selects slow angular speed inside the slow threshold."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_turn_to_heading(
        coordinator,
        target_heading_degrees=6.0,
        heading_tolerance_degrees=1.0,
        sample_delays=(),
    )

    assert result["stop_reason"] == "dry_run"
    assert result["initial_command_selection"]["angular_speed"] == 90
    assert result["initial_command_selection"]["speed_tier"] == "slow"


@pytest.mark.asyncio
async def test_raw_pymammotion_turn_to_heading_returns_reached_without_command() -> None:
    """Already-at-target heading returns reached and sends nothing."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 1.0))

    result = await _raw_pymammotion_turn_to_heading(
        coordinator,
        target_heading_degrees=2.0,
        heading_tolerance_degrees=3.0,
        sample_delays=(),
    )

    assert result["stop_reason"] == "target_heading_reached"
    assert result["commands_sent"] == 0
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_turn_to_heading_rejects_missing_confirmations() -> None:
    """Real turn-to-heading rejects missing confirmations."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_turn_to_heading(
        coordinator,
        target_heading_degrees=20.0,
        dry_run=False,
        sample_delays=(),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert result["blockers"] == [
        "operator_confirmed_blades_off",
        "operator_confirmed_clear_area",
    ]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_turn_to_heading_sends_raw_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real turn-to-heading sends raw angular commands until target is reached."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        coordinator.data.mowing_state.toward = 18.0

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_turn_to_heading(
        coordinator,
        target_heading_degrees=20.0,
        heading_tolerance_degrees=3.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 1
    assert result["stop_reason"] == "target_heading_reached"
    assert result["heading_status"]["complete"] is True
    assert result["heading_diagnostics"][0]["passed"] is True
    coordinator.manager.send_command_with_args.assert_awaited_once_with(
        "Luba-Test",
        "send_movement",
        prefer_ble=True,
        linear_speed=0,
        angular_speed=180,
    )


@pytest.mark.asyncio
async def test_raw_pymammotion_turn_to_heading_sends_explicit_stop_after_pulse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each real angular pulse is followed by an explicit stop, not left to firmware.

    ``send_movement`` is a continuous-velocity command with no protocol-level
    duration bound -- live testing showed the mower can travel/turn far past
    the intended pulse when nothing ever calls the stop primitive. Regression
    guard: assert async_stop_manual_motion fires after the pulse.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        coordinator.data.mowing_state.toward = 18.0

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    await _raw_pymammotion_turn_to_heading(
        coordinator,
        target_heading_degrees=20.0,
        heading_tolerance_degrees=3.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        prefer_ble=True,
        sample_delays=(0,),
    )

    coordinator.async_stop_manual_motion.assert_awaited_once_with(use_wifi=False)


@pytest.mark.asyncio
async def test_raw_pymammotion_turn_to_heading_stops_on_no_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real turn-to-heading stops when heading telemetry does not progress."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_turn_to_heading(
        coordinator,
        target_heading_degrees=20.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 1
    assert result["stop_reason"] == "no_heading_progress"


@pytest.mark.asyncio
async def test_raw_pymammotion_turn_to_heading_stops_after_max_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real turn-to-heading stops at cap after valid progress."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    headings = [4.0, 8.0, 12.0, 16.0]

    async def no_sleep(_: float) -> None:
        if headings:
            coordinator.data.mowing_state.toward = headings.pop(0)

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_turn_to_heading(
        coordinator,
        target_heading_degrees=20.0,
        heading_tolerance_degrees=1.0,
        max_commands=2,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 2
    assert result["stop_reason"] == "max_commands_reached"
    assert result["heading_diagnostics"][-1]["passed"] is True
    assert result["heading_status"]["heading_error_degrees"] == 4.0


@pytest.mark.asyncio
async def test_raw_motion_readiness_test_dry_run_selects_expected_commands() -> None:
    """Readiness dry-run runs all non-moving phases and selects expected commands."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_motion_readiness_test(coordinator, sample_delays=())

    assert result["ready_for_vector_segment"] is True
    assert result["ready_for_multi_point"] is False
    assert result["linear_y_ready"] is True
    assert result["turn_to_heading_ready"] is True
    assert result["real_steps_run"] == 0
    assert result["failed_phase"] is None
    phase_by_name = {phase["name"]: phase for phase in result["phases"]}
    assert list(phase_by_name) == [
        "safety_snapshot",
        "dry_run_negative_y_segment",
        "dry_run_positive_y_segment",
        "dry_run_positive_turn_to_heading",
        "dry_run_negative_turn_to_heading",
    ]
    assert phase_by_name["dry_run_negative_y_segment"]["result"]["command_not_sent"][
        "kwargs"
    ] == {"linear_speed": 200, "angular_speed": 0}
    assert phase_by_name["dry_run_positive_y_segment"]["result"]["command_not_sent"][
        "kwargs"
    ] == {"linear_speed": -200, "angular_speed": 0}
    assert phase_by_name["dry_run_positive_turn_to_heading"]["result"][
        "command_not_sent"
    ]["kwargs"] == {"linear_speed": 0, "angular_speed": 180}
    assert phase_by_name["dry_run_negative_turn_to_heading"]["result"][
        "command_not_sent"
    ]["kwargs"] == {"linear_speed": 0, "angular_speed": -180}
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_motion_readiness_test_fails_on_unsafe_snapshot() -> None:
    """Readiness stops immediately on unsafe runtime state."""
    coordinator = _pulse_coordinator(blade_state=1, position=(1.0, 1.0, 0.0))

    result = await _raw_motion_readiness_test(coordinator, sample_delays=())

    assert result["ready_for_vector_segment"] is False
    assert result["failed_phase"] == "safety_snapshot"
    assert result["blockers"] == ["blade_reported_on"]
    assert [phase["name"] for phase in result["phases"]] == ["safety_snapshot"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_motion_readiness_test_real_rejects_missing_confirmations() -> None:
    """Real readiness rejects real phases without operator confirmations."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_motion_readiness_test(
        coordinator,
        dry_run=False,
        max_real_steps=1,
        sample_delays=(),
    )

    assert result["failed_phase"] == "real_preflight"
    assert result["blockers"] == [
        "operator_confirmed_blades_off",
        "operator_confirmed_clear_area",
    ]
    assert result["real_steps_run"] == 0
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_motion_readiness_test_max_real_steps_limits_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness only runs the requested number of real phases."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    headings = [4.0, 8.0, 4.0, 0.0]

    async def no_sleep(_: float) -> None:
        if headings:
            coordinator.data.mowing_state.toward = headings.pop(0)

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_motion_readiness_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_real_steps=2,
        sample_delays=(0,),
    )

    assert result["ready_for_vector_segment"] is True
    assert result["real_steps_run"] == 2
    assert [phase["name"] for phase in result["phases"]][-2:] == [
        "real_positive_turn_to_heading",
        "real_negative_turn_to_heading",
    ]
    assert coordinator.manager.send_command_with_args.await_count == 2


@pytest.mark.asyncio
async def test_raw_motion_readiness_test_stops_on_first_failed_real_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness stops on the first failed real phase."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_motion_readiness_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_real_steps=4,
        sample_delays=(0,),
    )

    assert result["ready_for_vector_segment"] is False
    assert result["failed_phase"] == "real_positive_turn_to_heading"
    assert result["real_steps_run"] == 1
    assert [phase["name"] for phase in result["phases"]][-1] == (
        "real_positive_turn_to_heading"
    )
    assert coordinator.manager.send_command_with_args.await_count == 1


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_vector_segment_dry_run_with_zero_offset() -> None:
    """Vector dry-run can use an explicit zero heading offset."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        sample_delays=(0,),
    )

    assert result["service"] == "raw_pymammotion_execute_vector_segment"
    assert result["dry_run"] is True
    assert result["stop_reason"] == "dry_run"
    assert result["target_map_heading_degrees"] == 0.0
    assert result["target_reported_heading_degrees"] == 0.0
    assert result["target_heading_degrees"] == 0.0
    assert result["ready_for_multi_point"] is False
    assert [phase["name"] for phase in result["phases"]] == [
        "turn_to_target_heading",
        "linear_forward_to_target",
    ]
    assert result["command_not_sent"]["kwargs"] == {
        "linear_speed": 200,
        "angular_speed": 0,
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_vector_segment_dry_run_applies_offset() -> None:
    """Vector dry-run converts map target heading into reported mower heading."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=116.5,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "dry_run"
    assert result["target_map_heading_degrees"] == 0.0
    assert result["target_reported_heading_degrees"] == pytest.approx(243.5)
    assert result["heading_calibration"] == {
        "formula": (
            "target_reported_heading = "
            "target_map_heading - calibrated_forward_heading_offset"
        ),
        "target_map_heading_degrees": 0.0,
        "calibrated_forward_heading_offset_degrees": 116.5,
        "target_reported_heading_degrees": pytest.approx(243.5),
    }
    turn_phase = result["phases"][0]["result"]
    assert turn_phase["command_not_sent"]["kwargs"] == {
        "linear_speed": 0,
        "angular_speed": -180,
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_vector_segment_rejects_missing_confirmations() -> None:
    """Real vector execution requires explicit operator confirmations."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=False,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "operator_confirmed_clear_area" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_vector_segment_sends_forward_after_heading_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real vector execution sends one raw forward command after heading is aligned."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        max_turn_commands=1,
        max_linear_commands=1,
        sample_delays=(0,),
    )

    assert result["turn_commands_sent"] == 0
    assert result["linear_commands_sent"] == 1
    assert result["stop_reason"] == "no_target_progress"
    coordinator.manager.send_command_with_args.assert_awaited_once_with(
        "Luba-Test",
        "send_movement",
        prefer_ble=True,
        linear_speed=200,
        angular_speed=0,
    )


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_vector_segment_sends_explicit_stop_after_pulse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each real linear pulse is followed by an explicit stop, not left to firmware.

    ``send_movement`` is a continuous-velocity command with no protocol-level
    duration bound -- live testing showed a single "pulse" travel ~7x the
    expected distance because nothing ever called the stop primitive.
    Regression guard: assert async_stop_manual_motion fires after the pulse.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        max_turn_commands=1,
        max_linear_commands=1,
        prefer_ble=True,
        sample_delays=(0,),
    )

    coordinator.async_stop_manual_motion.assert_awaited_once_with(use_wifi=False)


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_vector_segment_halts_before_linear_on_incomplete_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segment requiring a large turn halts in the turn phase; no linear command is sent.

    Unlike the pure-linear `_raw_pymammotion_execute_segment`, the vector segment has no
    explicit pre-flight rejection for out-of-calibrated-window turns -- it attempts the turn
    and relies on the turn-command budget (`max_turn_commands`) and heading-progress checks
    to halt safely before any forward motion is attempted. This is the actual safety
    mechanism the multi-segment chain (used by the multi-waypoint click/go path builder)
    relies on for segments requiring more than a small heading correction.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 2.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        max_turn_commands=1,
        max_linear_commands=1,
        sample_delays=(0,),
    )

    assert result["target_map_heading_degrees"] == pytest.approx(90.0)
    assert result["stop_reason"] == "turn_phase_incomplete"
    assert result["turn_commands_sent"] == 1
    assert result["linear_commands_sent"] == 0


@pytest.mark.asyncio
async def test_vector_segment_loop_to_tolerance_stops_on_consecutive_no_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loop-to-tolerance mode keeps pulsing past the legacy budget, then stops on stall.

    With ``max_linear_pulse_ceiling`` set the linear phase no longer quits at the tiny
    ``max_linear_commands`` budget (max 3); it pulses until the waypoint is reached or
    ``max_no_progress_pulses`` consecutive pulses make no target-directed progress. Here
    the mocked mower never moves, so it must stop after exactly that many pulses.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    # Target due +X with a zero offset and toward 0.0 => turn phase needs 0 commands,
    # so the linear loop runs; a stationary mower makes no progress.
    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        max_linear_commands=1,
        max_linear_pulse_ceiling=20,
        max_no_progress_pulses=3,
        sample_delays=(0,),
    )

    assert result["linear_execution_mode"] == "loop_to_tolerance"
    assert result["turn_commands_sent"] == 0
    assert result["stop_reason"] == "no_target_progress"
    # Pulsed 3 times (max_no_progress_pulses) -- well past the legacy budget of 1.
    assert result["linear_commands_sent"] == 3
    # Each linear pulse ran the position-settle poll; a stationary mower never
    # registers motion, so every pulse records it did not settle/move.
    settle_flags = [
        (c.get("position_moved"), c.get("position_settled"))
        for c in result["command_results"]
        if "position_settled" in c
    ]
    assert settle_flags == [(False, False)] * 3


@pytest.mark.asyncio
async def test_vector_segment_real_run_requires_ble_transport() -> None:
    """Real motion is refused when the active transport is not BLE."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    # Flip the coordinator's normalized live transport to cloud.
    coordinator.active_transport_state = "cloud_aliyun"

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        sample_delays=(),
    )

    assert "ble_transport_required" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vector_segment_vio_real_run_blocked_when_feed_degraded() -> None:
    """VIO real run is refused when vio_state reads active but the feed is blind."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=2, track_feature_num=0, brightness=10
    )

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="vio",
        vio_heading_offset_degrees=0.0,
        sample_delays=(),
    )

    assert "vio_feed_live" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"
    assert result["vio"]["initial_vio_feed"]["live"] is False
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vector_segment_dry_run_allowed_off_ble() -> None:
    """Dry-run stays valid over a non-BLE transport (the BLE gate only guards real motion)."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud_aliyun"

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        sample_delays=(),
    )

    assert result["stop_reason"] == "dry_run"
    assert "ble_transport_required" not in result["blockers"]


def test_active_transport_state_normalizes_real_ble_enum() -> None:
    """The coordinator normalizes the real TransportType.BLE enum to 'ble'.

    Regression guard for the BLE gate: ``str(TransportType.BLE)`` is
    ``'TransportType.BLE'`` (not ``'ble'``), so any exact-string match against
    ``str(active_transport())`` silently fails and blocks every real run. The
    services BLE gate must go through this normalized property, and this test
    exercises the property against the genuine enum -- not a stand-in string.
    """
    # Document the trap: the raw stringified enum is not "ble".
    assert str(TransportType.BLE).lower() != "ble"

    handle = SimpleNamespace(active_transport=lambda: TransportType.BLE)
    fake_self = SimpleNamespace(
        device_name="Luba-Test",
        manager=SimpleNamespace(mower=lambda _name: handle),
    )

    normalized = MammotionReportUpdateCoordinator.active_transport_state.fget(
        fake_self
    )

    assert normalized == "ble"
    assert _transport_is_ble(SimpleNamespace(active_transport_state=normalized))


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_multi_segment_dry_run_chains_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-segment dry-run calls the vector primitive for each segment only."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    calls: list[tuple[list[dict[str, float]], bool]] = []

    async def fake_vector(
        coordinator_arg: MammotionReportUpdateCoordinator,
        points: list[dict[str, float]],
        **kwargs: object,
    ) -> dict:
        assert coordinator_arg is coordinator
        calls.append((points, bool(kwargs["dry_run"])))
        return {
            "valid": True,
            "stop_reason": "dry_run",
            "blockers": [],
            "phases": [{"passed": True}, {"passed": True}],
            "final_telemetry": _custom_path_telemetry_snapshot(coordinator),
            "progress_diagnostics": [],
        }

    monkeypatch.setattr(
        mammotion_services,
        "_raw_pymammotion_execute_vector_segment",
        fake_vector,
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [
            {"x": 1.0, "y": 1.0},
            {"x": 1.1, "y": 1.0},
            {"x": 1.2, "y": 1.1},
        ],
        sample_delays=(0,),
    )

    assert result["service"] == "raw_pymammotion_execute_multi_segment"
    assert result["dry_run"] is True
    assert result["stop_reason"] == "dry_run"
    assert result["ready_for_multi_segment"] is True
    assert result["ready_for_multi_point"] is False
    assert result["segments_executed"] == 2
    assert calls == [
        ([{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}], True),
        ([{"x": 1.1, "y": 1.0}, {"x": 1.2, "y": 1.1}], True),
    ]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vector_segment_vio_dry_run_plans_calibration_and_turn() -> None:
    """Default VIO turn mode dry-runs with a planned calibration drive + turn."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        sample_delays=(0,),
    )

    assert result["turn_mode"] == "vio"
    assert result["stop_reason"] == "dry_run"
    assert [phase["name"] for phase in result["phases"]] == [
        "turn_to_target_heading",
        "linear_forward_to_target",
    ]
    turn_phase = result["phases"][0]
    assert turn_phase["turn_mode"] == "vio"
    assert turn_phase["passed"] is True
    planned = turn_phase["result"]["planned"]
    assert planned["turn_primitive"] == "vio_turn_to_heading"
    assert planned["angular_speed"] == 500
    assert planned["calibration_drive"]["kwargs"] == {
        "linear_speed": 400,
        "angular_speed": 0,
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vector_segment_vio_real_blocked_when_vio_cold() -> None:
    """Real VIO-mode segment refuses to move unless VIO is actively tracking."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "vio_active" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vector_segment_vio_cold_start_allowed_when_scene_bright(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold VIO + bright scene: the calibration drive doubles as the warm-up."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    # vio_state 0 (cold) but brightness 100 -> "Light" scene.
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=0, brightness=100
    )

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        # The drive woke VIO and calibrated.
        return {
            "passed": True,
            "reason": "calibrated",
            "offset_degrees": -90.0,
            "map_motion_heading_degrees": 280.0,
            "vision_heading": 10.0,
            "vio_state": 2,
            "distance_m": 0.08,
            "pulses_sent": 1,
            "command_results": [],
        }

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 2,
            "command_results": [],
        }

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_commands=1,
        vio_max_realignments=0,
        sample_delays=(0,),
    )

    # Not blocked: the run proceeded through calibration and the turn.
    assert "vio_active" not in result["blockers"]
    assert result["calibration_commands_sent"] == 1
    assert result["phases"][0]["passed"] is True


@pytest.mark.asyncio
async def test_vector_segment_vio_cold_start_blocked_when_offset_skips_warmup() -> None:
    """Cold VIO stays blocked when a provided offset would skip the warm-up drive."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=0, brightness=100
    )

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        vio_heading_offset_degrees=100.0,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "vio_active" in result["blockers"]


@pytest.mark.asyncio
async def test_vector_segment_vio_real_calibrates_turns_then_drives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real VIO segment: calibration offset -> VIO turn on mapped heading -> linear."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        assert coordinator_arg is coordinator
        return {
            "passed": True,
            "reason": "calibrated",
            "offset_degrees": -90.0,
            "map_motion_heading_degrees": 280.0,
            "vision_heading": 10.0,
            "vio_state": 2,
            "distance_m": 0.06,
            "pulses_sent": 1,
            "command_results": [{"phase": "vio_calibration_drive", "ok": True}],
        }

    turn_calls: list[dict[str, object]] = []

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        turn_calls.append(kwargs)
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 2,
            "command_results": [],
            "final_vision_heading": 90.0,
            "final_heading_error_degrees": 0.5,
        }

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_commands=1,
        vio_max_realignments=0,
        sample_delays=(0,),
    )

    # Map heading to target is 0 deg; offset -90 -> target_vision_heading 90.
    assert len(turn_calls) == 1
    assert turn_calls[0]["target_vision_heading"] == pytest.approx(90.0)
    assert turn_calls[0]["angular_speed"] == 500
    assert result["vio"]["offset_degrees"] == pytest.approx(-90.0)
    assert result["vio"]["offset_source"] == "calibration_drive"
    assert result["vio"]["target_vision_heading"] == pytest.approx(90.0)
    assert result["calibration_commands_sent"] == 1
    assert result["turn_commands_sent"] == 2
    assert result["phases"][0]["passed"] is True
    # Static test position never reaches the waypoint: linear stops on progress.
    assert result["linear_commands_sent"] == 1
    assert result["stop_reason"] == "no_target_progress"


@pytest.mark.asyncio
async def test_vector_segment_vio_real_stops_on_failed_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed calibration drive halts the segment before any turn/linear motion."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "passed": False,
            "reason": "vio_not_active_after_drive",
            "offset_degrees": None,
            "vision_heading": 0.0,
            "vio_state": 0,
            "distance_m": 0.05,
            "pulses_sent": 2,
            "command_results": [],
        }

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "vio_calibration_failed"
    assert result["turn_commands_sent"] == 0
    assert result["linear_commands_sent"] == 0
    assert result["phases"][0]["passed"] is False


@pytest.mark.asyncio
async def test_vector_segment_vio_realigns_when_facing_drifts_off_bearing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-drive re-aim fires a bounded VIO turn when facing drifts off bearing."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    # Facing estimate = vision_heading + offset = 10 + (-90) = -80 deg map,
    # bearing to target is 0 deg -> 80 deg aim error > 15 deg threshold.
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "passed": True,
            "reason": "calibrated",
            "offset_degrees": -90.0,
            "map_motion_heading_degrees": 280.0,
            "vision_heading": 10.0,
            "vio_state": 2,
            "distance_m": 0.06,
            "pulses_sent": 1,
            "command_results": [],
        }

    turn_calls: list[dict[str, object]] = []

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        turn_calls.append(kwargs)
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
        }

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_commands=1,
        sample_delays=(0,),
    )

    # One initial turn + one mid-drive re-aim, both via the VIO primitive.
    assert len(turn_calls) == 2
    assert len(result["realignments"]) == 1
    realign = result["realignments"][0]
    assert realign["stop_reason"] == "target_heading_reached"
    assert abs(realign["aim_error_degrees"]) > 15.0
    # The corrected off-bearing pulse does not count as no-progress.
    assert result["stop_reason"] == "max_linear_commands_reached"


@pytest.mark.asyncio
async def test_vio_calibration_drive_aborts_when_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed stop (e.g. BLE cooldown) aborts the drive before another pulse."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=2, brightness=100
    )

    async def no_sleep(_: float) -> None:
        return None

    async def failing_stop(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {"attempted": True, "ok": False, "error": "BLEUnavailableError: cooldown"}

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        mammotion_services, "_manual_velocity_stop_attempt", failing_stop
    )

    result = await _vio_segment_calibration_drive(coordinator, max_pulses=3)

    assert result["passed"] is False
    assert result["reason"] == "stop_failed_aborting"
    # Only the first pulse fired; no further motion after the failed stop.
    assert result["pulses_sent"] == 1


@pytest.mark.asyncio
async def test_vio_turn_to_heading_aborts_when_stop_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stop exception mid-turn aborts instead of sending more turn pulses."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)
    coordinator.async_stop_manual_motion.side_effect = RuntimeError("BLE cooldown")

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "stop_failed_aborting"
    assert result["commands_sent"] == 1


@pytest.mark.asyncio
async def test_vio_segment_calibration_drive_computes_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The calibration drive derives offset = map motion heading - vision heading."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=0
    )

    async def no_sleep(_: float) -> None:
        return None

    async def fake_refresh(
        coordinator_arg: MammotionReportUpdateCoordinator,
    ) -> dict:
        # Simulate the post-drive feedback refresh: mower moved +x/+y and the
        # motion woke VIO with a fresh body heading.
        coordinator.data.mowing_state.pos_x += 0.03
        coordinator.data.mowing_state.pos_y += 0.03
        coordinator.data.report_data.vision_info = SimpleNamespace(
            heading=15.0, vio_state=2
        )
        return {"refreshed": True}

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        mammotion_services, "_refresh_position_after_raw_motion", fake_refresh
    )

    result = await _vio_segment_calibration_drive(coordinator, max_pulses=2)

    assert result["passed"] is True
    # 6 cm minimum baseline: one 4.2 cm pulse is not enough, two (8.5 cm) are.
    assert result["pulses_sent"] == 2
    # Motion vector (+0.06, +0.06) -> 45 deg map heading; offset = 45 - 15 = 30.
    assert result["map_motion_heading_degrees"] == pytest.approx(45.0)
    assert result["offset_degrees"] == pytest.approx(30.0)
    coordinator.async_stop_manual_motion.assert_awaited()


@pytest.mark.asyncio
async def test_vio_segment_calibration_drive_rejects_offset_on_degraded_feed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blind feed (0 features) yields no offset even though vio_state reads active."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=0
    )

    async def no_sleep(_: float) -> None:
        return None

    async def fake_refresh(
        coordinator_arg: MammotionReportUpdateCoordinator,
    ) -> dict:
        # Motion moved the mower, but VIO woke blind: vio_state latched active
        # with 0 tracked features, so the vision heading is untrustworthy and the
        # offset it would produce would be silently wrong.
        coordinator.data.mowing_state.pos_x += 0.03
        coordinator.data.mowing_state.pos_y += 0.03
        coordinator.data.report_data.vision_info = SimpleNamespace(
            heading=15.0, vio_state=2, track_feature_num=0, brightness=10
        )
        return {"refreshed": True}

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        mammotion_services, "_refresh_position_after_raw_motion", fake_refresh
    )

    result = await _vio_segment_calibration_drive(coordinator, max_pulses=2)

    assert result["passed"] is False
    assert result["reason"] == "vio_feed_degraded"
    assert result["offset_degrees"] is None
    assert result["vio_feed"]["live"] is False


@pytest.mark.asyncio
async def test_settle_linear_position_feed_waits_for_lagged_jump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The settle poll waits through a lagged/frozen feed until the jump registers.

    The map-local feed sits at the pre-pulse value for a couple of samples then
    jumps (live 2026-07-15). Settling must require the feed to actually move off
    the pre-pulse position AND then stop changing, so the pulse's motion is not
    missed as a false "already settled".
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    before = _custom_path_telemetry_snapshot(coordinator)
    calls = {"n": 0}

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        calls["n"] += 1
        # Frozen for two polls (feed lag), then a single jump on the third, then
        # holds steady.
        if calls["n"] == 3:
            coordinator.data.mowing_state.pos_x += 0.10

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    res = await _settle_linear_position_feed(coordinator, before)

    assert res["moved"] is True
    assert res["settled"] is True
    # Registered the jump (poll 3) then confirmed it held (poll 4); did not run to
    # the full poll budget.
    assert calls["n"] == 4


@pytest.mark.asyncio
async def test_settle_linear_position_feed_times_out_when_no_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked pulse never registers motion, so the poll times out un-settled."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    before = _custom_path_telemetry_snapshot(coordinator)

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        return None  # position never changes

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    res = await _settle_linear_position_feed(
        coordinator, before, timeout_seconds=4.0, poll_interval_seconds=1.0
    )

    assert res["moved"] is False
    assert res["settled"] is False
    # Ran the full bounded budget (4s / 1s = 4 polls) without settling.
    assert coordinator.async_get_reports.await_count == 4


@pytest.mark.asyncio
async def test_multi_segment_vio_carries_offset_between_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segment 1's derived VIO offset is passed to segment 2 (no recalibration)."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    received_offsets: list[object] = []

    async def fake_vector(
        coordinator_arg: MammotionReportUpdateCoordinator,
        points: list[dict[str, float]],
        **kwargs: object,
    ) -> dict:
        received_offsets.append(kwargs["vio_heading_offset_degrees"])
        return {
            "valid": True,
            "stop_reason": "dry_run",
            "blockers": [],
            "phases": [{"passed": True}, {"passed": True}],
            "final_telemetry": _custom_path_telemetry_snapshot(coordinator),
            "progress_diagnostics": [],
            "vio": {"offset_degrees": 42.0},
        }

    monkeypatch.setattr(
        mammotion_services,
        "_raw_pymammotion_execute_vector_segment",
        fake_vector,
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [
            {"x": 1.0, "y": 1.0},
            {"x": 1.1, "y": 1.0},
            {"x": 1.2, "y": 1.1},
        ],
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "dry_run"
    assert received_offsets == [None, 42.0]


@pytest.mark.asyncio
async def test_vector_segment_ble_auto_recovers_then_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful BLE auto-recovery lets ble_transport_required pass and the run proceed."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud"  # not BLE at entry
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=10.0, vio_state=2)

    async def no_sleep(_: float) -> None:
        return None

    recovery_calls: list[object] = []

    async def fake_recover(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        recovery_calls.append(coordinator_arg)
        coordinator.active_transport_state = "ble"  # promoted
        return {"attempted": True, "ok": True, "reason": "promoted", "steps": []}

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "passed": True,
            "reason": "calibrated",
            "offset_degrees": -90.0,
            "map_motion_heading_degrees": 280.0,
            "vision_heading": 10.0,
            "vio_state": 2,
            "distance_m": 0.06,
            "pulses_sent": 1,
            "command_results": [],
        }

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 2,
            "command_results": [],
        }

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(mammotion_services, "_attempt_ble_recovery", fake_recover)
    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_commands=1,
        vio_max_realignments=0,
        sample_delays=(0,),
    )

    # Recovery ran once, promoted BLE, and the transport gate is no longer a blocker.
    assert recovery_calls == [coordinator]
    assert result["ble_recovery"]["ok"] is True
    assert "ble_transport_required" not in result["blockers"]
    assert result["blockers"] == []


@pytest.mark.asyncio
async def test_vector_segment_ble_auto_recovery_failure_fails_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed BLE auto-recovery leaves ble_transport_required blocking the real run."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud"  # not BLE, and recovery can't fix it
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=10.0, vio_state=2)

    async def fake_recover(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        # Present-but-not-promoted: recovery could not win the slot (phone app).
        return {
            "attempted": True,
            "ok": False,
            "reason": "ble_promotion_timeout_check_phone_app",
            "steps": ["reasserted_ble_preference", "ble_toggled"],
        }

    monkeypatch.setattr(mammotion_services, "_attempt_ble_recovery", fake_recover)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["ble_recovery"]["ok"] is False
    assert result["ble_recovery"]["reason"] == "ble_promotion_timeout_check_phone_app"
    assert "ble_transport_required" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vector_segment_ble_auto_recover_disabled_skips_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ble_auto_recover=false skips recovery entirely; the transport gate blocks as before."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud"
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=10.0, vio_state=2)

    async def fail_if_called(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        raise AssertionError("recovery must not run when ble_auto_recover is False")

    monkeypatch.setattr(mammotion_services, "_attempt_ble_recovery", fail_if_called)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        ble_auto_recover=False,
        sample_delays=(0,),
    )

    assert result["ble_recovery"] is None
    assert "ble_transport_required" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"


@pytest.mark.asyncio
async def test_vector_segment_refetches_runtime_context_after_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-recovery gates judge refetched HA state, not the pre-recovery snapshot.

    BLE recovery can wait ~90s. Here the mower starts mowing during that wait:
    the handler-captured ha_state says "idle" (would pass), but the refetched
    context says "mowing" and must block the run.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud"

    async def fake_recover(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        coordinator.active_transport_state = "ble"
        return {"attempted": True, "ok": True, "reason": "promoted", "steps": []}

    refetches = {"count": 0}

    def refetch() -> tuple[str | None, dict | None]:
        refetches["count"] += 1
        return ("mowing", None)

    monkeypatch.setattr(mammotion_services, "_attempt_ble_recovery", fake_recover)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="legacy",
        calibrated_forward_heading_offset_degrees=0.0,
        ha_state="idle",
        sample_delays=(),
        refetch_runtime_context=refetch,
    )

    assert refetches["count"] == 1
    assert "runtime_not_mowing" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_multi_segment_ble_auto_recovers_then_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-segment BLE auto-recovery promotes BLE so gates pass and segments run."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud"

    recovery_calls: list[object] = []

    async def fake_recover(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        recovery_calls.append(coordinator_arg)
        coordinator.active_transport_state = "ble"
        return {"attempted": True, "ok": True, "reason": "promoted", "steps": []}

    async def fake_vector(
        coordinator_arg: MammotionReportUpdateCoordinator,
        points: list[dict[str, float]],
        **kwargs: object,
    ) -> dict:
        return {
            "valid": True,
            "stop_reason": "target_reached",
            "blockers": [],
            "progress_diagnostics": [],
            "final_telemetry": _custom_path_telemetry_snapshot(coordinator),
            "vio": {"offset_degrees": 42.0},
        }

    monkeypatch.setattr(mammotion_services, "_attempt_ble_recovery", fake_recover)
    monkeypatch.setattr(
        mammotion_services, "_raw_pymammotion_execute_vector_segment", fake_vector
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_real_segments=1,
        sample_delays=(0,),
    )

    assert recovery_calls == [coordinator]
    assert result["ble_recovery"]["ok"] is True
    assert "ble_transport_required" not in result["blockers"]
    assert result["blockers"] == []
    assert result["stop_reason"] == "target_reached"


@pytest.mark.asyncio
async def test_multi_segment_ble_auto_recovery_failure_fails_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed multi-segment BLE recovery keeps ble_transport_required blocking."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "cloud"

    async def fake_recover(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "attempted": True,
            "ok": False,
            "reason": "mower_not_advertising_needs_wake",
            "steps": ["ble_cooldown_active_waiting"],
        }

    monkeypatch.setattr(mammotion_services, "_attempt_ble_recovery", fake_recover)

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(0,),
    )

    assert result["ble_recovery"]["ok"] is False
    assert "ble_transport_required" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"
    coordinator.manager.send_command_with_args.assert_not_called()


def test_ble_connect_cooldown_active_reads_transport_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cooldown guard reflects the BLE transport's _connect_cooldown_until deadline."""
    coordinator = _pulse_coordinator()
    deadline = {"value": 0.0}

    def get_transport(_transport_type: object) -> SimpleNamespace:
        return SimpleNamespace(_connect_cooldown_until=deadline["value"])

    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        get_transport=get_transport
    )
    monkeypatch.setattr(mammotion_services.time, "monotonic", lambda: 1000.0)

    # No cooldown armed (0.0 deadline is in the past).
    assert _ble_connect_cooldown_active(coordinator) is False
    # Deadline in the future -> cooldown active.
    deadline["value"] = 1005.0
    assert _ble_connect_cooldown_active(coordinator) is True
    # Deadline already elapsed -> inactive again.
    deadline["value"] = 995.0
    assert _ble_connect_cooldown_active(coordinator) is False


def test_ble_connect_cooldown_active_defends_against_api_drift() -> None:
    """A handle without get_transport (or a raising one) reads as "no cooldown"."""
    coordinator = _pulse_coordinator()

    coordinator.manager.mower = lambda _device_name: SimpleNamespace()
    assert _ble_connect_cooldown_active(coordinator) is False

    def raising_mower(_device_name: str) -> object:
        raise RuntimeError("handle unavailable")

    coordinator.manager.mower = raising_mower
    assert _ble_connect_cooldown_active(coordinator) is False


def test_pinned_pymammotion_ble_transport_exposes_connect_cooldown_until() -> None:
    """Guard against pymammotion drift: the cooldown attr the guard reads must exist.

    _ble_connect_cooldown_active reads BLETransport._connect_cooldown_until; if a
    pymammotion bump renames or drops it the guard silently degrades to "never in
    cooldown", so pin the contract here.
    """
    transport = BLETransport(BLETransportConfig(device_id="test"))
    cooldown_attr = "_connect_cooldown_until"
    assert hasattr(transport, cooldown_attr)
    assert getattr(transport, cooldown_attr) == 0.0


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_multi_segment_real_rejects_missing_confirmations() -> None:
    """Real multi-segment execution requires explicit operator confirmations."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=False,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "operator_confirmed_clear_area" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_multi_segment_limits_real_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_real_segments limits real chained segment execution."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    calls: list[bool] = []

    async def fake_vector(
        coordinator_arg: MammotionReportUpdateCoordinator,
        points: list[dict[str, float]],
        **kwargs: object,
    ) -> dict:
        assert coordinator_arg is coordinator
        calls.append(bool(kwargs["dry_run"]))
        return {
            "valid": True,
            "stop_reason": "target_reached",
            "blockers": [],
            "phases": [{"passed": True}, {"passed": True}],
            "final_telemetry": _custom_path_telemetry_snapshot(coordinator),
            "progress_diagnostics": [{"passed": True}],
        }

    monkeypatch.setattr(
        mammotion_services,
        "_raw_pymammotion_execute_vector_segment",
        fake_vector,
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [
            {"x": 1.0, "y": 1.0},
            {"x": 1.1, "y": 1.0},
            {"x": 1.2, "y": 1.1},
        ],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_real_segments=1,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "max_real_segments_reached"
    assert result["real_segments_executed"] == 1
    assert result["segments"][1]["skipped_reason"] == "max_real_segments_reached"
    assert calls == [False]


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_multi_segment_stops_on_first_failed_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-segment wrapper stops on the first failed vector segment."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    call_count = 0

    async def fake_vector(
        coordinator_arg: MammotionReportUpdateCoordinator,
        points: list[dict[str, float]],
        **kwargs: object,
    ) -> dict:
        nonlocal call_count
        assert coordinator_arg is coordinator
        call_count += 1
        return {
            "valid": True,
            "stop_reason": "no_target_progress" if call_count == 1 else "dry_run",
            "blockers": [],
            "phases": [{"passed": True}, {"passed": True}],
            "final_telemetry": _custom_path_telemetry_snapshot(coordinator),
            "progress_diagnostics": [{"passed": False}],
        }

    monkeypatch.setattr(
        mammotion_services,
        "_raw_pymammotion_execute_vector_segment",
        fake_vector,
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [
            {"x": 1.0, "y": 1.0},
            {"x": 1.1, "y": 1.0},
            {"x": 1.2, "y": 1.1},
        ],
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "segment_failed"
    assert result["failed_segment_index"] == 1
    assert result["segments_executed"] == 1
    assert call_count == 1


@pytest.mark.asyncio
async def test_raw_vector_readiness_test_dry_run_selects_expected_phases() -> None:
    """Vector readiness dry-run covers aligned and both turn directions."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_vector_readiness_test(coordinator, sample_delays=(0,))

    assert result["dry_run"] is True
    assert result["ready_for_multi_segment"] is True
    assert result["ready_for_multi_point"] is False
    assert result["real_steps_run"] == 0
    assert result["failed_phase"] is None
    assert [phase["name"] for phase in result["phases"]] == [
        "safety_snapshot",
        "dry_run_aligned_vector",
        "dry_run_positive_turn_vector",
        "dry_run_negative_turn_vector",
    ]
    aligned = result["phases"][1]["result"]
    positive = result["phases"][2]["result"]
    negative = result["phases"][3]["result"]
    assert aligned["phases"][0]["result"]["stop_reason"] == "target_heading_reached"
    assert positive["phases"][0]["result"]["command_not_sent"]["kwargs"] == {
        "linear_speed": 0,
        "angular_speed": 180,
    }
    assert negative["phases"][0]["result"]["command_not_sent"]["kwargs"] == {
        "linear_speed": 0,
        "angular_speed": -180,
    }
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_vector_readiness_test_real_rejects_missing_confirmations() -> None:
    """Real vector readiness rejects missing operator confirmations."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_vector_readiness_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=False,
        max_real_steps=1,
        sample_delays=(0,),
    )

    assert result["ready_for_multi_segment"] is False
    assert result["failed_phase"] == "real_preflight"
    assert result["blockers"] == ["operator_confirmed_clear_area"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_raw_vector_readiness_test_max_real_steps_limits_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vector readiness max_real_steps limits real movement phases."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_vector_readiness_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_real_steps=1,
        calibrated_forward_heading_offset_degrees=0.0,
        sample_delays=(0,),
    )

    assert result["real_steps_run"] == 1
    assert [phase["name"] for phase in result["phases"]][-1] == "real_aligned_vector"
    assert coordinator.manager.send_command_with_args.await_count == 2


def test_raw_vector_readiness_phase_passed_accepts_proven_progress() -> None:
    """Real phase passes when at least one path-progress pulse is demonstrated."""
    assert (
        _raw_vector_readiness_phase_passed(
            "real_aligned_vector",
            {
                "stop_reason": "no_target_progress",
                "valid": True,
                "blockers": [],
                "progress_diagnostics": [
                    {"status": "path_progress", "passed": True},
                    {"status": "no_path_progress", "passed": False},
                ],
            },
        )
        is True
    )


def test_raw_vector_readiness_phase_passed_rejects_no_progress() -> None:
    """Real phase fails if no path-progress pulse is demonstrated."""
    assert (
        _raw_vector_readiness_phase_passed(
            "real_aligned_vector",
            {
                "stop_reason": "no_target_progress",
                "valid": True,
                "blockers": [],
                "progress_diagnostics": [
                    {"status": "no_path_progress", "passed": False},
                ],
            },
        )
        is False
    )


def test_raw_vector_readiness_phase_passed_accepts_aligned_translation_signal() -> None:
    """Aligned real phase accepts measured translation with heading progress."""
    assert (
        _raw_vector_readiness_phase_passed(
            "real_aligned_vector",
            {
                "stop_reason": "no_target_progress",
                "valid": True,
                "blockers": [],
                "progress_diagnostics": [
                    {
                        "status": "no_path_progress",
                        "passed": False,
                        "heading_progress": True,
                        "min_progress_distance": 0.005,
                        "measured_delta": {"distance": 0.0048},
                        "path_progress_distance": -0.0043,
                    }
                ],
            },
        )
        is True
    )


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_defaults_to_dry_run() -> None:
    """Pulse test default sends no command and reports the command not sent."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_pulse_test(coordinator, followup_samples=0)

    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["real_pulse_allowed"] is False
    assert result["reason"] == "dry_run"
    assert result["command_not_sent"] == {
        "service": "mammotion.move_forward",
        "data": {"speed": 0.1, "use_wifi": False},
    }
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_firmware_mode_skips_explicit_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Firmware nudge mode sends movement but does not issue zero-speed stop."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _manual_velocity_pulse_test(
        coordinator,
        action="forward",
        speed=0.4,
        duration_ms=750,
        stop_mode="firmware",
        post_command_sample_delays=(0.0, 2.0),
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["would_send"] is True
    assert result["stop_mode"] == "firmware"
    assert result["stop_result"]["attempted"] is False
    assert result["stop_result"]["reason"] == "firmware_nudge_mode_no_explicit_stop"
    assert result["real_pulse_completed"] is True
    assert coordinator.async_move_forward.await_count == 1
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_missing_confirmations() -> None:
    """Real pulse rejects missing operator confirmations before movement."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["reason"] == "safety_gates_failed"
    assert result["blockers"] == [
        "operator_confirmed_blades_off",
        "operator_confirmed_clear_area",
    ]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_unsafe_blade_telemetry() -> None:
    """Real pulse rejects nonzero blade/RPM telemetry before movement."""
    coordinator = _pulse_coordinator(blade_state=1, cutter_rpm=1200)

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["blockers"] == ["mower_reports_blades_off"]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_active_work_mode() -> None:
    """Real pulse rejects active mowing/working mode before movement."""
    coordinator = _pulse_coordinator(work_mode=13)

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["blockers"] == ["mower_ready"]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_allows_paused_work_mode() -> None:
    """Real pulse allows MODE_PAUSE after a canceled job when other gates pass."""
    coordinator = _pulse_coordinator(work_mode=19)

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is True
    assert result["blockers"] == []
    coordinator.async_move_forward.assert_awaited_once()
    coordinator.async_stop_manual_motion.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_unavailable_position() -> None:
    """Real pulse rejects missing live map-local position before movement."""
    coordinator = _pulse_coordinator(position=(None, None, None), pos_type=0, zone_hash=0)

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["blockers"] == [
        "live_map_position_available",
        "position_area_inside",
    ]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_charging_state() -> None:
    """Real pulse rejects docked/charging state before movement."""
    coordinator = _pulse_coordinator(charge_state=2)

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["blockers"] == ["not_docked_or_charging"]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_area_out_zero_zone() -> None:
    """Real pulse rejects AREA_OUT and unknown zone hash before movement."""
    coordinator = _pulse_coordinator(pos_type=0, zone_hash=0)

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["blockers"] == ["position_area_inside"]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_allows_turn_area_inside() -> None:
    """Real pulse allows TURN_AREA_INSIDE when position and zone are known."""
    coordinator = _pulse_coordinator(pos_type=4, zone_hash=123)

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is True
    assert result["blockers"] == []
    before_position = result["samples"][0]["telemetry"]["position"]
    assert before_position["pos_type_label"] == (
        "TURN_AREA_INSIDE"
    )
    assert before_position["valid_for_motion"] is True
    coordinator.async_move_forward.assert_awaited_once()
    coordinator.async_stop_manual_motion.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_allows_channel_area_overlap() -> None:
    """Real pulse allows CHANNEL_AREA_OVERLAP when position and zone are known."""
    coordinator = _pulse_coordinator(pos_type=9, zone_hash=123)

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    before_position = result["samples"][0]["telemetry"]["position"]

    assert result["would_send"] is True
    assert result["blockers"] == []
    assert before_position["pos_type_label"] == "CHANNEL_AREA_OVERLAP"
    assert before_position["valid_for_motion"] is True
    coordinator.async_move_forward.assert_awaited_once()
    coordinator.async_stop_manual_motion.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_rejects_zero_map_position() -> None:
    """Real pulse rejects zero map-local x/y even with known area metadata."""
    coordinator = _pulse_coordinator(position=(0.0, 0.0, 0.0))

    result = await _manual_velocity_pulse_test(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is False
    assert result["blockers"] == ["map_position_nonzero"]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_real_probe_calls_move_then_stop() -> None:
    """Allowed real probe sends one tiny pulse and then the stop primitive."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_pulse_test(
        coordinator,
        action="turn_left",
        speed=0.1,
        duration_ms=50,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is True
    assert result["real_pulse_allowed"] is True
    assert result["command_result"]["attempted"] is True
    assert result["command_result"]["ok"] is True
    assert result["command_result"]["error"] is None
    assert result["command_result"]["action"] == "turn_left"
    assert result["command_result"]["coordinator_method"] == "async_move_left"
    assert result["command_result"]["transport_preference"] == "ble_preferred"
    assert result["command_result"]["duration_ms"] >= 0
    assert result["stop_result"]["attempted"] is True
    assert result["stop_result"]["ok"] is True
    assert result["stop_result"]["error"] is None
    assert result["stop_result"]["coordinator_method"] == "async_stop_manual_motion"
    assert result["real_pulse_completed"] is True
    coordinator.async_move_left.assert_awaited_once_with(speed=0.1, use_wifi=False)
    coordinator.async_stop_manual_motion.assert_awaited_once_with(use_wifi=False)


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_reports_false_command_ack() -> None:
    """A false coordinator command return is reported as an unsuccessful attempt."""
    coordinator = _pulse_coordinator()
    coordinator.async_move_forward.return_value = False

    result = await _manual_velocity_pulse_test(
        coordinator,
        action="forward",
        speed=0.1,
        duration_ms=50,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["command_result"]["attempted"] is True
    assert result["command_result"]["ok"] is False
    assert result["command_result"]["ack"] is False
    assert result["real_pulse_completed"] is False
    assert result["stop_result"]["ok"] is True


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
    assert parsed["heading_offset_candidates"] == list(DEFAULT_HEADING_OFFSET_CANDIDATES)


def test_manual_velocity_heading_offset_candidate_schema_rejects_invalid_values() -> None:
    """Heading offset candidates are bounded to valid degrees."""
    with pytest.raises(Exception):  # noqa: B017
        MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST_SCHEMA(
            {
                "entity_id": "lawn_mower.test",
                "points": [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
                "heading_offset_candidates": [0, 181],
            }
        )


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_defaults_to_dry_run() -> None:
    """Segment probe default plans the next command but sends nothing."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
    )

    assert result["service"] == "manual_velocity_segment_test"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["real_segment_allowed"] is False
    assert result["stop_reason"] == "dry_run"
    assert result["initial_controller_decision"]["action"] == "forward"
    assert result["command_not_sent"] == {
        "service": "mammotion.move_forward",
        "data": {"speed": 0.4, "use_wifi": False},
    }
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_rejects_charging_state() -> None:
    """Real segment probe blocks before movement when pulse gates fail."""
    coordinator = _pulse_coordinator(charge_state=2)

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["would_send"] is False
    assert result["stop_reason"] == "safety_gates_failed"
    assert result["blockers"] == ["not_docked_or_charging"]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_real_probe_calls_move_then_stop() -> None:
    """Allowed segment probe sends one capped pulse and then stops."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=1,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
    )

    assert result["would_send"] is True
    assert result["real_segment_allowed"] is True
    assert result["stop_reason"] == "path_progress_lost"
    assert result["pulses_sent"] == 1
    assert result["iterations"][0]["controller_decision"]["action"] == "forward"
    assert result["iterations"][0]["command_result"]["attempted"] is True
    assert result["iterations"][0]["command_result"]["ok"] is True
    assert result["iterations"][0]["command_result"]["error"] is None
    assert result["iterations"][0]["command_result"]["action"] == "forward"
    assert (
        result["iterations"][0]["command_result"]["coordinator_method"]
        == "async_move_forward"
    )
    assert result["iterations"][0]["command_result"]["transport_preference"] == "wifi"
    assert result["iterations"][0]["command_result"]["duration_ms"] >= 0
    assert result["iterations"][0]["stop_result"]["attempted"] is True
    assert result["iterations"][0]["stop_result"]["ok"] is True
    assert result["iterations"][0]["stop_result"]["error"] is None
    assert (
        result["iterations"][0]["stop_result"]["coordinator_method"]
        == "async_stop_manual_motion"
    )
    assert result["iterations"][0]["movement_diagnostic"]["status"] == (
        "visual_motion_possible_but_telemetry_unchanged"
    )
    assert result["progress_summary"]["no_progress_count"] == 1
    coordinator.async_move_forward.assert_awaited_once_with(speed=0.4, use_wifi=True)
    coordinator.async_stop_manual_motion.assert_awaited_once_with(use_wifi=True)


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_force_action_overrides_controller() -> None:
    """Force action lets diagnostics test a specific low-level movement command."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 2.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=1,
        force_action="forward",
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
    )

    decision = result["iterations"][0]["controller_decision"]
    assert decision["action"] == "forward"
    assert decision["forced"] is True
    assert decision["original_action"] == "turn_left"
    coordinator.async_move_forward.assert_awaited_once_with(speed=0.4, use_wifi=True)
    coordinator.async_move_left.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_stops_after_no_progress_limit() -> None:
    """Multi-pulse probes stop after consecutive low-progress telemetry samples."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=5,
        no_progress_limit=2,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
        require_progress_each_pulse=False,
    )

    assert result["stop_reason"] == "no_progress_limit_reached"
    assert result["pulses_sent"] == 2
    assert result["progress_summary"]["no_progress_count"] == 2
    assert coordinator.async_move_forward.await_count == 2
    assert coordinator.async_stop_manual_motion.await_count == 2


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_reports_partial_progress_timeout() -> None:
    """Max pulses after target-directed progress is a partial-progress timeout."""
    coordinator = _pulse_coordinator()

    async def move_forward_progress(*_: object, **__: object) -> None:
        coordinator.data.mowing_state.pos_x = 1.2

    coordinator.async_stop_manual_motion.side_effect = move_forward_progress

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=1,
        no_progress_limit=2,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
    )

    assert result["stop_reason"] == "partial_progress_timeout"
    assert result["completion_status"]["complete"] is False
    assert result["progress_summary"]["cumulative_path_progress"] == pytest.approx(0.2)
    assert result["iterations"][0]["path_progress_diagnostic"]["passed"] is True


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_accepts_delayed_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delayed target-directed telemetry prevents premature path_progress_lost."""
    coordinator = _pulse_coordinator()
    original_snapshot = mammotion_services._custom_path_telemetry_snapshot  # noqa: SLF001
    snapshot_count = 0

    def delayed_snapshot(coordinator_arg: object) -> dict[str, object]:
        nonlocal snapshot_count
        snapshot_count += 1
        if snapshot_count >= 5:
            coordinator.data.mowing_state.pos_x = 1.2
        return original_snapshot(coordinator_arg)

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services, "_custom_path_telemetry_snapshot", delayed_snapshot)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=1,
        no_progress_limit=1,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(30.0, 45.0, 60.0),
    )

    iteration = result["iterations"][0]
    assert result["stop_reason"] == "partial_progress_timeout"
    assert iteration["late_telemetry_check"] is True
    assert iteration["late_progress_detected"] is True
    assert iteration["telemetry_latency_seconds"] == 45.0
    assert iteration["late_path_progress_diagnostic"]["passed"] is True
    assert iteration["path_progress_diagnostic"]["passed"] is True
    assert result["progress_summary"]["cumulative_path_progress"] == pytest.approx(0.2)
    coordinator.async_move_forward.assert_awaited_once_with(speed=0.4, use_wifi=True)


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_reports_lost_after_late_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No delayed telemetry progress through the late window remains progress lost."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=1,
        no_progress_limit=1,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(30.0, 45.0, 60.0),
    )

    iteration = result["iterations"][0]
    assert result["stop_reason"] == "path_progress_lost"
    assert iteration["late_telemetry_check"] is True
    assert iteration["late_progress_detected"] is False
    assert iteration["telemetry_latency_seconds"] is None
    assert iteration["late_path_progress_diagnostic"]["passed"] is False


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_reports_path_complete_at_max_pulses() -> None:
    """Max pulses at target reports path_complete, not a timeout."""
    coordinator = _pulse_coordinator()

    async def move_to_target(*_: object, **__: object) -> None:
        coordinator.data.mowing_state.pos_x = 2.0

    coordinator.async_stop_manual_motion.side_effect = move_to_target

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=1,
        waypoint_tolerance=0.1,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
    )

    assert result["stop_reason"] == "path_complete"
    assert result["completion_status"]["complete"] is True
    assert result["iterations"][0]["path_progress_diagnostic"]["passed"] is True


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_stops_when_quality_degrades() -> None:
    """Multi-pulse probes stop when telemetry quality degrades after a pulse."""
    coordinator = _pulse_coordinator()

    async def degrade_position_quality(*_: object, **__: object) -> None:
        coordinator.data.mowing_state.pos_level = 2
        coordinator.data.report_data.rtk.pos_level = 2

    coordinator.async_stop_manual_motion.side_effect = degrade_position_quality

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=5,
        no_progress_limit=5,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
        service_name="manual_velocity_multi_pulse_test",
    )

    assert result["stop_reason"] == "telemetry_quality_degraded"
    assert result["blockers"] == ["pos_level_degraded"]
    assert result["pulses_sent"] == 1
    assert result["iterations"][0]["quality_degradation"]["degraded"] is True


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_can_report_multi_pulse_service_name() -> None:
    """The same guarded engine can back the explicit multi-pulse service."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        service_name="manual_velocity_multi_pulse_test",
    )

    assert result["service"] == "manual_velocity_multi_pulse_test"
    coordinator.async_move_forward.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_cumulative_pulse_test_defaults_to_dry_run() -> None:
    """Cumulative pulse probe default plans the burst but sends nothing."""
    coordinator = _pulse_coordinator()

    result = await _manual_velocity_cumulative_pulse_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        heading_offset_degrees=110.0,
        heading_offset_candidates=[110.0, 0.0],
    )

    assert result["service"] == "manual_velocity_cumulative_pulse_test"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["real_probe_allowed"] is False
    assert result["stop_reason"] == "dry_run"
    assert result["heading_offset_candidates"] == [110.0, 0.0]
    assert result["initial_controller_decision"]["selected_heading_offset_degrees"] == 0.0
    assert len(result["initial_controller_decision"]["heading_offset_diagnostics"]) == 2
    assert result["command_not_sent"] == {
        "service": "mammotion.move_forward",
        "data": {"speed": 0.4, "use_wifi": False},
    }
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_cumulative_pulse_test_firmware_mode_skips_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cumulative firmware nudge mode sends pulses without explicit zero-stop."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _manual_velocity_cumulative_pulse_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=2,
        force_action="forward",
        stop_mode="firmware",
        use_wifi=False,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        cumulative_sample_delays=(0.0,),
    )

    assert result["stop_mode"] == "firmware"
    assert result["pulses_sent"] == 2
    assert result["pulse_results"][0]["stop_result"]["attempted"] is False
    assert result["pulse_results"][0]["stop_result"]["reason"] == (
        "firmware_nudge_mode_no_explicit_stop"
    )
    assert coordinator.async_move_forward.await_count == 2
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_cumulative_pulse_test_detects_delayed_cumulative_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cumulative probe sends a pulse burst, then accepts delayed total progress."""
    coordinator = _pulse_coordinator()
    original_snapshot = mammotion_services._custom_path_telemetry_snapshot  # noqa: SLF001
    snapshot_count = 0

    def delayed_snapshot(coordinator_arg: object) -> dict[str, object]:
        nonlocal snapshot_count
        snapshot_count += 1
        if snapshot_count >= 10:
            coordinator.data.mowing_state.pos_x = 1.2
        return original_snapshot(coordinator_arg)

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(
        mammotion_services,
        "_custom_path_telemetry_snapshot",
        delayed_snapshot,
    )
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _manual_velocity_cumulative_pulse_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        max_pulses=3,
        use_wifi=True,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        cumulative_sample_delays=(0.0, 30.0, 60.0),
    )

    assert result["stop_reason"] == "cumulative_progress_detected"
    assert result["result_status"] == "cumulative_progress_detected"
    assert result["pulses_sent"] == 3
    assert result["cumulative_progress_detected"] is True
    assert result["telemetry_latency_seconds"] == 60.0
    assert result["cumulative_delta"]["distance"] == pytest.approx(0.2)
    assert result["cumulative_path_progress_diagnostic"]["passed"] is True
    assert coordinator.async_move_forward.await_count == 3
    assert coordinator.async_stop_manual_motion.await_count == 3


@pytest.mark.asyncio
async def test_experimental_execute_segment_burst_stops_after_no_cumulative_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Burst execution stops after a burst that never gets telemetry progress."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _experimental_execute_segment_burst(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        pulses_per_burst=2,
        max_bursts=3,
        stop_mode="immediate",
        calibrated_forward_heading_degrees=0,
        use_wifi=True,
        cumulative_sample_delays=(0.0,),
    )

    assert result["service"] == "experimental_execute_segment_burst"
    assert result["stop_reason"] == "no_cumulative_progress"
    assert result["bursts_sent"] == 1
    assert result["pulses_sent"] == 2
    assert result["bursts"][0]["cumulative_progress_detected"] is False
    assert coordinator.async_move_forward.await_count == 2
    assert coordinator.async_stop_manual_motion.await_count == 2


@pytest.mark.asyncio
async def test_experimental_execute_segment_burst_continues_after_cumulative_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Burst execution can send another burst after delayed target progress."""
    coordinator = _pulse_coordinator()
    stop_count = 0

    async def move_after_stop(*_: object, **__: object) -> None:
        nonlocal stop_count
        stop_count += 1
        coordinator.data.mowing_state.pos_x = 1.2 if stop_count == 1 else 2.0

    async def no_sleep(_: float) -> None:
        return None

    coordinator.async_stop_manual_motion.side_effect = move_after_stop
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _experimental_execute_segment_burst(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        pulses_per_burst=1,
        max_bursts=3,
        waypoint_tolerance=0.05,
        stop_mode="immediate",
        calibrated_forward_heading_degrees=0,
        use_wifi=True,
        cumulative_sample_delays=(0.0, 60.0),
    )

    assert result["stop_reason"] == "path_complete"
    assert result["completion_status"]["complete"] is True
    assert result["bursts_sent"] == 2
    assert result["pulses_sent"] == 2
    assert result["bursts"][0]["cumulative_progress_detected"] is True
    assert result["bursts"][1]["cumulative_progress_detected"] is True
    assert result["cumulative_path_progress"] == pytest.approx(1.0)
    assert coordinator.async_move_forward.await_count == 2
    assert coordinator.async_stop_manual_motion.await_count == 2


@pytest.mark.asyncio
async def test_experimental_execute_segment_burst_blocks_unproven_turn_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default experimental execution only allows calibrated forward segments."""
    coordinator = _pulse_coordinator()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _experimental_execute_segment_burst(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        speed=0.4,
        pulse_duration_ms=50,
        cumulative_sample_delays=(0.0,),
    )

    assert result["stop_reason"] == (
        "segment_heading_outside_calibrated_forward_window"
    )
    assert result["blockers"] == ["unproven_turn_or_lateral_motion_required"]
    assert result["manual_motion_execution_policy"][
        "experimental_segment_scope"
    ] == "one_segment_calibrated_forward_only"
    assert result["calibrated_forward_heading_diagnostic"] == {
        "segment_heading_degrees": 0.0,
        "calibrated_forward_heading_degrees": 270.0,
        "heading_error_degrees": 90.0,
        "tolerance_degrees": 45.0,
        "within_calibrated_forward_window": False,
        "allow_unproven_turns": False,
    }
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_manual_velocity_segment_test_stops_when_path_complete() -> None:
    """Segment probe sends nothing when current position is already at target."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _manual_velocity_segment_test(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.01, "y": 1.01}],
        waypoint_tolerance=0.1,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        post_stop_sample_delays=(0,),
    )

    assert result["would_send"] is True
    assert result["stop_reason"] == "path_complete"
    assert result["pulses_sent"] == 0
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


def test_manual_velocity_heading_calibration_reports_vector_offset() -> None:
    """Heading calibration compares reported heading to movement vector heading."""
    before = {
        "position": {
            "x": 1.0,
            "y": 1.0,
            "toward": 0.0,
            "source": "report_data.locations[0]",
        }
    }
    after = {
        "position": {
            "x": 1.0,
            "y": 2.0,
            "toward": 0.0,
            "source": "report_data.locations[0]",
        }
    }

    result = _manual_velocity_heading_calibration(
        action="forward",
        before=before,
        after=after,
        min_progress_distance=0.003,
        min_heading_change_degrees=1.0,
    )

    assert result["movement_vector_heading"] == 90.0
    assert result["heading_error_degrees"] == 90.0
    assert result["recommended_heading_offset_degrees"] == 90.0
    assert result["interpretation"] == "movement_vector_available"


def test_custom_path_telemetry_uses_mowing_state_position() -> None:
    """Telemetry prefers live mowing_state map-local position values."""
    coordinator = SimpleNamespace(
        is_online=lambda: True,
        data=SimpleNamespace(
            mowing_state=SimpleNamespace(
                pos_x=1.25,
                pos_y=-2.5,
                toward=91.5,
                pos_level=0,
                rtk_status=SimpleNamespace(value=4, name="FINE"),
                zone_hash=LARGE_HASH,
                pos_type=1,
            ),
            location=SimpleNamespace(orientation=45, position_type=7, work_zone=123),
            report_data=SimpleNamespace(
                dev=SimpleNamespace(sys_status=11, charge_state=2, blade_state=0),
                rtk=SimpleNamespace(status=4, pos_level=2),
                locations=[],
                cutter_work_mode_info=SimpleNamespace(
                    current_cutter_mode=0,
                    current_cutter_rpm=0,
                ),
                connect=SimpleNamespace(
                    ble_rssi=-70,
                    wifi_rssi=-69,
                    connect_type=0,
                    used_net="",
                    wifi_connect_status=None,
                    iot_connect_status=None,
                ),
            ),
        ),
    )

    snapshot = _custom_path_telemetry_snapshot(coordinator)

    assert snapshot["work_mode"] == 11
    assert snapshot["work_mode_label"] == "MODE_READY"
    assert snapshot["charge_state"] == 2
    assert snapshot["charge_state_label"] == "docked_or_charging"
    assert snapshot["position"] == {
        "x": 1.25,
        "y": -2.5,
        "toward": 91.5,
        "source": "mowing_state",
        "pos_level": 0,
        "pos_level_label": "FIX",
        "rtk_status": 4,
        "rtk_status_label": "Fix",
        "pos_type": 1,
        "pos_type_label": "AREA_INSIDE",
        "zone_hash": str(LARGE_HASH),
        "area_name": None,
        "valid_for_motion": True,
    }
    assert snapshot["blade"]["reported_state"] == 0
    assert snapshot["blade"]["current_cutter_rpm"] == 0
    assert snapshot["transport"]["connection_label"] == "WIFI/BLE"


def test_custom_path_telemetry_overlays_location_metadata_on_stale_zero_pose() -> None:
    """Stale mowing_state zero/AREA_OUT does not hide valid area metadata."""
    coordinator = SimpleNamespace(
        is_online=lambda: True,
        get_area_entity_name=lambda area_hash: (
            "Backyard Right" if area_hash == 123 else None
        ),
        data=SimpleNamespace(
            mowing_state=SimpleNamespace(
                pos_x=0.0,
                pos_y=0.0,
                toward=0.0,
                pos_level=0,
                rtk_status=0,
                zone_hash=0,
                pos_type=0,
            ),
            location=SimpleNamespace(orientation=45, position_type=1, work_zone=123),
            report_data=SimpleNamespace(
                dev=SimpleNamespace(sys_status=11, charge_state=0, blade_state=0),
                rtk=SimpleNamespace(status=4, pos_level=0),
                locations=[],
                cutter_work_mode_info=SimpleNamespace(
                    current_cutter_mode=0,
                    current_cutter_rpm=0,
                ),
                connect=None,
            ),
        ),
    )

    position = _custom_path_telemetry_snapshot(coordinator)["position"]

    assert position["source"] == "location_metadata"
    assert position["x"] is None
    assert position["y"] is None
    assert position["toward"] == 45
    assert position["pos_type"] == 1
    assert position["pos_type_label"] == "AREA_INSIDE"
    assert position["zone_hash"] == 123
    assert position["area_name"] == "Backyard Right"
    assert position["valid_for_motion"] is False


def test_custom_path_telemetry_falls_back_to_report_location() -> None:
    """Telemetry falls back to report_data.locations[0] and scales raw fields."""
    coordinator = SimpleNamespace(
        is_online=lambda: True,
        data=SimpleNamespace(
            location=SimpleNamespace(orientation=45, position_type=7, work_zone=123),
            report_data=SimpleNamespace(
                dev=SimpleNamespace(sys_status=11, charge_state=2, blade_state=0),
                rtk=SimpleNamespace(status=4, pos_level=0),
                locations=[
                    SimpleNamespace(
                        real_pos_x=12_345,
                        real_pos_y=-67_890,
                        real_toward=900_000,
                        pos_type=5,
                        bol_hash=123,
                    )
                ],
                cutter_work_mode_info=SimpleNamespace(
                    current_cutter_mode=0,
                    current_cutter_rpm=0,
                ),
                connect=None,
            ),
        ),
    )

    position = _custom_path_telemetry_snapshot(coordinator)["position"]

    assert position["source"] == "report_data.locations[0]"
    assert position["x"] == 1.2345
    assert position["y"] == -6.789
    assert position["toward"] == 90.0
    assert position["pos_level"] == 0
    assert position["pos_level_label"] == "FIX"
    assert position["rtk_status"] == 4
    assert position["rtk_status_label"] == "Fix"
    assert position["pos_type"] == 5
    assert position["pos_type_label"] == "CHARGE_ON"
    assert position["zone_hash"] == 123
    assert position["area_name"] is None
    assert position["valid_for_motion"] is False


def test_custom_path_telemetry_reports_position_candidates() -> None:
    """Telemetry exposes all candidate position sources for diagnostics."""
    coordinator = SimpleNamespace(
        is_online=lambda: True,
        data=SimpleNamespace(
            mowing_state=SimpleNamespace(
                pos_x=1.0,
                pos_y=2.0,
                toward=30.0,
                pos_level=0,
                rtk_status=4,
                zone_hash=456,
                pos_type=1,
            ),
            location=SimpleNamespace(orientation=45, position_type=1, work_zone=456),
            report_data=SimpleNamespace(
                dev=SimpleNamespace(sys_status=11, charge_state=0, blade_state=0),
                rtk=SimpleNamespace(status=4, pos_level=0),
                locations=[
                    SimpleNamespace(
                        real_pos_x=30_000,
                        real_pos_y=40_000,
                        real_toward=500_000,
                        pos_type=1,
                        bol_hash=456,
                    )
                ],
                cutter_work_mode_info=SimpleNamespace(
                    current_cutter_mode=0,
                    current_cutter_rpm=0,
                ),
                connect=None,
            ),
        ),
    )

    candidates = _custom_path_telemetry_snapshot(coordinator)["position_candidates"]
    sources = {candidate["source"]: candidate for candidate in candidates}

    assert sources["mowing_state"]["x"] == 1.0
    assert sources["mowing_state"]["valid_for_motion"] is True
    assert sources["report_data.locations[0]"]["x"] == 3.0
    assert sources["report_data.locations[0]"]["toward"] == 50.0
    assert sources["location_metadata"]["pos_type_label"] == "AREA_INSIDE"
    assert sources["report_data.rtk"]["rtk_status_label"] == "Fix"


def test_custom_path_telemetry_reports_unavailable_position_safely() -> None:
    """Missing position data returns an unavailable source without raising."""
    coordinator = SimpleNamespace(
        is_online=lambda: False,
        data=SimpleNamespace(
            report_data=SimpleNamespace(
                dev=SimpleNamespace(sys_status=99, charge_state=99, blade_state=None),
                rtk=SimpleNamespace(status=99, pos_level=99),
                locations=[],
                cutter_work_mode_info=SimpleNamespace(
                    current_cutter_mode=0,
                    current_cutter_rpm=0,
                ),
                connect=None,
            ),
        ),
    )

    snapshot = _custom_path_telemetry_snapshot(coordinator)

    assert snapshot["online"] is False
    assert snapshot["work_mode_label"] == "Invalid mode"
    assert snapshot["charge_state_label"] == "unknown"
    assert snapshot["position"]["source"] == "unavailable"
    assert snapshot["position"]["x"] is None
    assert snapshot["position"]["y"] is None
    assert snapshot["position"]["toward"] is None
    assert snapshot["position"]["pos_level"] == 99
    assert snapshot["position"]["pos_level_label"] == "UNKNOWN"
    assert snapshot["position"]["rtk_status"] == 99
    assert snapshot["position"]["rtk_status_label"] == "Unknown"


def test_diagnostic_sensor_values_match_map_and_task_data() -> None:
    """Diagnostic count/error sensors expose current coordinator data."""
    coordinator = _coordinator(_plan(enabled=False))
    coordinator.last_map_task_error = "task_sync: RuntimeError"
    coordinator.active_transport_state = "ble"
    coordinator.ble_only_fallback_mode = True
    coordinator.last_cloud_login_success = datetime.datetime.now(datetime.UTC)
    coordinator.last_token_refresh = datetime.datetime.now(datetime.UTC)
    coordinator.last_command_failure_reason = "set_car_wiper:GatewayTimeoutException"
    coordinator.last_camera_stream_failure_code = "401"
    descriptions = {description.key: description for description in WORK_SENSOR_TYPES}

    assert descriptions["task_count"].value_fn(coordinator, coordinator.data) == 1
    assert descriptions["enabled_task_count"].value_fn(coordinator, coordinator.data) == 0
    assert descriptions["area_count"].value_fn(coordinator, coordinator.data) == 1
    assert descriptions["map_area_name_count"].value_fn(coordinator, coordinator.data) == 1
    assert descriptions["last_map_sync"].value_fn(coordinator, coordinator.data) is None
    assert descriptions["last_task_sync"].value_fn(coordinator, coordinator.data) is None
    assert (
        descriptions["last_map_task_error"].value_fn(coordinator, coordinator.data)
        == "task_sync: RuntimeError"
    )
    assert descriptions["active_transport"].value_fn(coordinator, coordinator.data) == "ble"
    assert (
        descriptions["ble_only_fallback_mode"].value_fn(coordinator, coordinator.data)
        == "fallback_active"
    )
    assert (
        descriptions["last_cloud_login_success"].value_fn(coordinator, coordinator.data)
        == coordinator.last_cloud_login_success
    )
    assert (
        descriptions["last_token_refresh"].value_fn(coordinator, coordinator.data)
        == coordinator.last_token_refresh
    )
    assert (
        descriptions["last_command_failure_reason"].value_fn(coordinator, coordinator.data)
        == "set_car_wiper:GatewayTimeoutException"
    )
    assert (
        descriptions["last_camera_stream_failure_code"].value_fn(coordinator, coordinator.data)
        == "401"
    )


def test_camera_recovery_buttons_present() -> None:
    """Camera/cloud recovery buttons are exposed for camera-capable mowers."""
    keys = {description.key for description in BUTTON_LUBA_PRO_YUKA}

    assert "refresh_camera_stream" in keys
    assert "refresh_cloud_session" in keys


@pytest.mark.asyncio
async def test_refresh_camera_stream_raises_when_unavailable() -> None:
    """Camera refresh button surfaces a translated HA error if refresh fails."""
    coordinator = SimpleNamespace(
        async_check_stream_expiry=AsyncMock(return_value=(None, None)),
    )

    with pytest.raises(HomeAssistantError):
        await MammotionBaseUpdateCoordinator.async_refresh_camera_stream(coordinator)


@pytest.mark.asyncio
async def test_refresh_camera_stream_succeeds_when_available() -> None:
    """Camera refresh helper returns without error when stream data is available."""
    coordinator = SimpleNamespace(
        async_check_stream_expiry=AsyncMock(return_value=(SimpleNamespace(), None)),
    )

    await MammotionBaseUpdateCoordinator.async_refresh_camera_stream(coordinator)


@pytest.mark.asyncio
async def test_refresh_cloud_session_requires_cloud_account() -> None:
    """Cloud refresh helper rejects devices without cloud account configuration."""
    coordinator = SimpleNamespace(
        has_cloud_account=False,
        async_refresh_login=AsyncMock(),
    )

    with pytest.raises(HomeAssistantError):
        await MammotionBaseUpdateCoordinator.async_refresh_cloud_session(coordinator)


@pytest.mark.asyncio
async def test_refresh_cloud_session_calls_refresh_login() -> None:
    """Cloud refresh helper runs account refresh for cloud-enabled entries."""
    coordinator = SimpleNamespace(
        has_cloud_account=True,
        async_refresh_login=AsyncMock(),
    )

    await MammotionBaseUpdateCoordinator.async_refresh_cloud_session(coordinator)

    coordinator.async_refresh_login.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_sync_success_updates_last_sync_metadata() -> None:
    """Map/task sync success records timestamps and clears stale errors."""
    coordinator = SimpleNamespace(
        manager=SimpleNamespace(
            start_map_sync=AsyncMock(),
            start_plan_sync=AsyncMock(),
        ),
        device_name="Luba-Test",
        last_map_sync=None,
        last_task_sync=None,
        last_map_task_error="old error",
    )

    await MammotionReportUpdateCoordinator.async_sync_maps(coordinator)
    await MammotionReportUpdateCoordinator.async_sync_schedule(coordinator)

    assert coordinator.last_map_sync is not None
    assert coordinator.last_task_sync is not None
    assert coordinator.last_map_task_error is None


@pytest.mark.asyncio
async def test_sync_failure_updates_last_error() -> None:
    """Unexpected sync failures are recorded and re-raised."""
    coordinator = SimpleNamespace(
        manager=SimpleNamespace(start_map_sync=AsyncMock(side_effect=RuntimeError())),
        device_name="Luba-Test",
        last_map_sync=None,
        last_map_task_error=None,
    )

    with pytest.raises(RuntimeError):
        await MammotionReportUpdateCoordinator.async_sync_maps(coordinator)

    assert coordinator.last_map_task_error == "map_sync: RuntimeError"


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
        "set_non_work_hours",
        "start_stop_blades",
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
