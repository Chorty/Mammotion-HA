"""Tests for Mammotion read-only map/task visibility helpers."""

import ast
import asyncio
import datetime
import json
import pathlib
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.exceptions import HomeAssistantError
from pymammotion.transport.base import NoTransportAvailableError
from pymammotion.utility.constant import WorkMode

from custom_components.mammotion import coordinator as mammotion_coordinator
from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.button import BUTTON_LUBA_PRO_YUKA
from custom_components.mammotion.coordinator import (
    MammotionBaseUpdateCoordinator,
    MammotionReportUpdateCoordinator,
)
from custom_components.mammotion.sensor import WORK_SENSOR_TYPES
from custom_components.mammotion.services import (
    _basestation_has_query_fields,
    _basestation_info_probe,
    _current_orientation,
    _custom_path_telemetry_snapshot,
    _dry_run_custom_path,
    _export_active_route,
    _export_mower_map,
    _export_mower_tasks,
    _export_runtime_state,
    _manual_velocity_quality_degradation,
    _normalize_mower_areas,
    _normalize_mower_tasks,
    _point_on_segment,
    _position_stage_latency_summary,
    _preview_custom_path,
    _raw_pymammotion_angular_calibration,
    _raw_pymammotion_execute_vector_segment,
    _raw_pymammotion_motion_probe,
    _raw_pymammotion_turn_to_heading,
    _report_stream_probe,
    _report_stream_sequence_probe,
    _requires_reverse_recovery,
    _rtk_report_age_seconds,
    _runtime_motion_safety_summary,
    _validate_custom_path,
    _vio_feed_liveness,
    _wait_for_position_subscription_ready,
)

from .conftest import LARGE_HASH, _coordinator, _plan, _pulse_coordinator


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
    assert (
        exported["last_command_failure_reason"]
        == "set_car_wiper:GatewayTimeoutException"
    )
    assert exported["last_camera_stream_failure_code"] == "401"
    assert "blade_reported_on" in exported["safety"]["blockers"]


def test_export_runtime_state_reports_rapid_fusion_only_in_runtime_export() -> None:
    """Export the item-17 source without changing shared VIO telemetry records."""
    coordinator = _pulse_coordinator()
    coordinator.data.mowing_state.fuse_status = 2
    coordinator.data.mowing_state.vision_state_raw = 7
    coordinator.data.report_data.dev.fuse_status = 5

    telemetry = _custom_path_telemetry_snapshot(coordinator)
    exported = _export_runtime_state(
        coordinator,
        ha_state="paused",
        active_route={
            "mow_path_feature_count": 0,
            "mow_progress_feature_count": 0,
            "active_progress": None,
        },
    )

    assert "rapid_state_fusion" not in telemetry
    assert exported["rapid_state_fusion"] == {
        "source": "mowing_state.fuse_status (tard_state_data[16] bits 8-15)",
        "available": True,
        "fuse_status": 2,
        "fuse_status_label": "RTK_EXTENDED_VISION",
        "vision_state_raw": 7,
        "device_vslam_fuse_status": 5,
        "device_vslam_source": "report_data.dev.fuse_status (distinct 0-5 field)",
    }


def test_export_runtime_state_labels_unknown_or_missing_rapid_fusion() -> None:
    """Unknown codes remain visible and missing old-backend fields stay null."""
    coordinator = _pulse_coordinator()
    coordinator.data.mowing_state.fuse_status = 99
    coordinator.data.mowing_state.vision_state_raw = 4
    exported = _export_runtime_state(
        coordinator,
        ha_state="paused",
        active_route={
            "mow_path_feature_count": 0,
            "mow_progress_feature_count": 0,
            "active_progress": None,
        },
    )
    assert exported["rapid_state_fusion"]["fuse_status"] == 99
    assert exported["rapid_state_fusion"]["fuse_status_label"] == "UNKNOWN"

    del coordinator.data.mowing_state.fuse_status
    del coordinator.data.mowing_state.vision_state_raw
    exported = _export_runtime_state(
        coordinator,
        ha_state="paused",
        active_route={
            "mow_path_feature_count": 0,
            "mow_progress_feature_count": 0,
            "active_progress": None,
        },
    )
    assert exported["rapid_state_fusion"]["available"] is False
    assert exported["rapid_state_fusion"]["fuse_status"] is None
    assert exported["rapid_state_fusion"]["fuse_status_label"] is None


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
        exported["safety"]["active_route_status"]["reason"] == "live_route_while_mowing"
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
        exported["safety"]["active_route_status"]["reason"] == "stale_route_while_ready"
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


def test_validate_custom_path_rejects_outside_point_on_a_closed_ring() -> None:
    """A CLOSED polygon ring must still reject points outside it.

    Regression for 2026-07-28, found on the mower. Every existing containment
    test used an OPEN polygon (first vertex != last), but the device sends
    CLOSED RINGS. With a closed ring, ``_point_in_polygon`` starts at
    ``previous = polygon[-1]``, ``current = polygon[0]`` -- the same point, a
    zero-length segment. ``_point_on_segment`` then had cross == 0 and dot == 0
    identically, and its final check reduced to ``0 <= 0``, so it returned True
    for ANY point and containment short-circuited to True before casting a
    single ray.

    Live effect: a target 2 m outside the lawn tested as inside all four mapped
    areas at once, including two that were 19 m and 28 m away, and
    ``validate_custom_path`` reported ``valid: True`` with no errors. Every
    containment check in the integration was inert.
    """
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
                        # The closing vertex the real device sends.
                        SimpleNamespace(x=0.0, y=0.0),
                    ],
                )
            ]
        )
    }

    outside = _validate_custom_path(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 40.0, "y": 40.0}],
        area_hash=123,
        speed=0.2,
        blade_mode="off",
    )

    assert outside["valid"] is False
    assert "path_points_outside_known_area_geometry" in outside["errors"]

    # ...and the same ring must still ACCEPT genuinely interior points, so the
    # fix cannot be "reject everything".
    inside = _validate_custom_path(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 9.0, "y": 9.0}],
        area_hash=123,
        speed=0.2,
        blade_mode="off",
    )

    assert inside["valid"] is True
    assert inside["errors"] == []


def test_point_on_segment_rejects_far_points_on_a_degenerate_segment() -> None:
    """A zero-length segment is a point: only that point lies on it."""
    origin = {"x": 3.0, "y": -4.0}

    assert _point_on_segment(origin, origin, origin) is True
    assert _point_on_segment({"x": 3.0, "y": -4.0}, origin, origin) is True
    assert _point_on_segment({"x": 100.0, "y": 100.0}, origin, origin) is False
    assert _point_on_segment({"x": 3.1, "y": -4.0}, origin, origin) is False


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


@pytest.mark.asyncio
async def test_raw_probe_stream_start_failure_sends_no_motion() -> None:
    """A missing dense report feed fails closed before the movement command."""
    coordinator = _pulse_coordinator()
    coordinator.async_start_report_stream.side_effect = RuntimeError("stream failed")

    result = await _raw_pymammotion_motion_probe(
        coordinator,
        motion_refresh_interval_ms=200,
        in_window_sample_interval_ms=100,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=(),
    )

    assert result["reason"] == "report_stream_failed"
    assert result["command_result"]["attempted"] is False
    assert result["in_window_telemetry"]["report_stream"]["error"] == (
        "RuntimeError: stream failed"
    )
    coordinator.manager.send_command_with_args.assert_not_awaited()


def _orientation_coordinator(
    *, vio_heading: float, toward: float, features: int = 80
) -> tuple[Any, dict[str, Any]]:
    """Build a coordinator whose VIO and compass headings are set independently."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=vio_heading, vio_state=2, track_feature_num=features, brightness=200
    )
    return coordinator, {"position": {"toward": toward}}


def test_current_orientation_publishes_only_when_two_sources_corroborate() -> None:
    """The card's arrow needs a heading no single sensor can vouch for alone.

    Live values from the 2026-08-26 6 m run: toward -1.800 gives a compass mirror
    of 91.93 deg and VIO read 91.81, agreeing to 0.12 deg.
    """
    coordinator, telemetry = _orientation_coordinator(vio_heading=91.81, toward=-1.800)

    result = _current_orientation(coordinator, telemetry)

    assert result["trustworthy"] is True
    assert result["map_heading_degrees"] == pytest.approx(91.81, abs=0.01)
    assert result["reason"] == "corroborated"
    assert result["disagreement_degrees"] == pytest.approx(0.12, abs=0.02)
    # VIO is the published value; the mirror only corroborates it.
    assert result["source"] == "vio_heading corroborated by compass mirror"


def test_current_orientation_refuses_when_the_two_sources_disagree() -> None:
    """Disagreement means one source is wrong and we cannot tell which.

    Drawing a confidently wrong arrow is the failure that matters -- it is the
    exact reason beta19 stopped rendering the last-travel projection -- so the
    honest output is no arrow.
    """
    # VIO says 91.8; the compass mirror says ~0. Both cannot be right.
    coordinator, telemetry = _orientation_coordinator(vio_heading=91.81, toward=90.13)

    result = _current_orientation(coordinator, telemetry)

    assert result["trustworthy"] is False
    assert result["map_heading_degrees"] is None
    assert result["reason"] == "heading_sources_disagree"
    assert result["disagreement_degrees"] > 15.0


def test_current_orientation_refuses_a_dusk_latched_vio_feed() -> None:
    """`vio_state` stays 2 at dusk while the track goes blind and heading freezes.

    Gating on the feature count rather than the state is what catches it.
    """
    coordinator, telemetry = _orientation_coordinator(
        vio_heading=91.81, toward=-1.800, features=0
    )

    result = _current_orientation(coordinator, telemetry)

    assert result["trustworthy"] is False
    assert result["reason"] == "vio_feed_degraded"
    assert result["vio_feed_live"] is False


def test_current_orientation_is_absent_not_wrong_without_vio() -> None:
    """No VIO heading at all must read as unavailable, never as a default."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(
        vio_state=0, track_feature_num=0
    )

    result = _current_orientation(coordinator, {"position": {"toward": -1.800}})

    assert result["trustworthy"] is False
    assert result["map_heading_degrees"] is None
    assert result["reason"] == "vio_heading_unavailable"


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
async def test_raw_pymammotion_angular_calibration_rejects_missing_confirmations() -> (
    None
):
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
async def test_raw_pymammotion_turn_to_heading_returns_reached_without_command() -> (
    None
):
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

    coordinator.async_stop_manual_motion.assert_awaited_once()


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
async def test_vio_active_still_refuses_in_dark() -> None:
    """Adding night mode does not weaken the daylight-only VIO gate."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=0, brightness=0, track_feature_num=0
    )
    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.8, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="vio",
    )
    assert "vio_active" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


def test_reverse_recovery_guard_replays_both_gate4_passes() -> None:
    """Both nominal Gate 4 passes contain a U-turn that forward-only must refuse."""
    docs = pathlib.Path(__file__).parents[3] / "docs"
    day2j = json.loads(
        (docs / "evidence-gate4-beta20-day2j-real-result-20260805.json").read_text()
    )
    beta21 = json.loads(
        (
            docs / "evidence-gate4-beta21-second-geometry-summary-20260806.json"
        ).read_text()
    )

    day2j_errors = [
        item["aim_error_degrees"]
        for segment in day2j["result"]["segments"]
        for item in segment["result"]["realignments"]
    ]
    beta21_errors = [
        error
        for segment in beta21["real_result"]["segments"]
        for error in segment["realignment_errors_degrees"]
    ]

    assert any(_requires_reverse_recovery(error) for error in day2j_errors)
    assert any(_requires_reverse_recovery(error) for error in beta21_errors)
    assert not _requires_reverse_recovery(89.999)
    assert _requires_reverse_recovery(90.0)


def _safety_telemetry() -> dict[str, object]:
    """Minimal telemetry that clears every guard except the one under test."""
    return {
        "online": True,
        "work_mode": 11,
        "work_mode_label": "MODE_READY",
        "blade": {"reported_state": 0, "current_cutter_rpm": 0},
        "position": {
            "x": 1.0,
            "y": 1.0,
            "source": "report",
            "pos_type_label": "AREA_INSIDE",
            "zone_hash": 123,
        },
    }


def test_rtk_payload_age_is_reported_but_never_blocks() -> None:
    """Age is advisory. It must not gate motion at ANY value.

    Two thresholds were tried and both false-blocked: 300 s (disproved by a
    582 s legitimate quiet period) and 1800 s (disproved by 3573 s, measured on
    a healthy Fix-locked stationary mower). A stationary mower's RTK payload
    changes about hourly while the one observed fault lasted three hours, so no
    threshold can separate quiet from dead. An active probe cannot either: a
    forced burst on healthy RTK produced 49 messages and zero RTK updates.
    """
    for age in (12.0, 600.0, 3573.0, 3 * 60 * 60.0, 24 * 60 * 60.0):
        summary = _runtime_motion_safety_summary(
            _safety_telemetry(), rtk_report_age_seconds=age
        )
        assert "rtk_telemetry_stale" not in summary["blockers"], age
        assert summary["blockers"] == [], age
        assert summary["allowed_for_manual_motion"] is True, age
        assert summary["rtk_report_age_seconds"] == age

    # Still annotated past the advisory threshold, so a suspicious run can be
    # audited against it -- it simply carries no authority to refuse.
    quiet = _runtime_motion_safety_summary(
        _safety_telemetry(), rtk_report_age_seconds=4000.0
    )
    assert quiet["rtk_report_quiet"] is True
    assert quiet["allowed_for_manual_motion"] is True


def _rtk_telemetry(label: str | None) -> dict[str, object]:
    """Safety telemetry carrying a specific RTK solution type."""
    telemetry = _safety_telemetry()
    position = dict(telemetry["position"])  # type: ignore[arg-type]
    position["rtk_status_label"] = label
    telemetry["position"] = position
    return telemetry


def test_rtk_quality_blocks_a_precision_run_on_a_degraded_fix() -> None:
    """Float is decimetre-grade; a precision run on it steers on noise.

    Measured 2026-08-07: Fix jitters 0.55 cm at worst while stationary, Float
    produced a 13.9 cm jump with no command sent -- larger than the entire
    0.08 m waypoint tolerance.
    """
    fix = _runtime_motion_safety_summary(_rtk_telemetry("Fix"))
    assert "rtk_not_precise" not in fix["blockers"]
    assert fix["rtk_degraded"] is False

    for degraded in ("Float", "Single", "None"):
        summary = _runtime_motion_safety_summary(_rtk_telemetry(degraded))
        assert "rtk_not_precise" in summary["blockers"], degraded
        assert summary["allowed_for_manual_motion"] is False
        assert summary["rtk_status_label"] == degraded


def test_rtk_quality_override_permits_a_deliberate_degraded_run() -> None:
    """Relocations legitimately do not need centimetre accuracy.

    Both cases occurred on 2026-08-07: a 1.6 m relocation on Float was entirely
    reasonable, while a precision measurement on Float would have been
    meaningless. The override makes the caller say which kind of run it is, and
    the choice is recorded either way.
    """
    summary = _runtime_motion_safety_summary(
        _rtk_telemetry("Float"), allow_degraded_rtk=True
    )

    assert "rtk_not_precise" not in summary["blockers"]
    # The override suppresses the refusal but never hides the condition.
    assert summary["rtk_degraded"] is True
    assert summary["rtk_degraded_override"] is True
    assert summary["rtk_status_label"] == "Float"


def test_rtk_quality_unknown_status_does_not_block() -> None:
    """Unknown is unmeasured, not degraded -- same rule as freshness."""
    summary = _runtime_motion_safety_summary(_rtk_telemetry(None))

    assert "rtk_not_precise" not in summary["blockers"]
    assert summary["rtk_degraded"] is False


def test_rtk_freshness_is_advisory_when_it_cannot_be_measured() -> None:
    """`None` means unmeasured, not unsafe -- diagnostic callers have no tracker."""
    summary = _runtime_motion_safety_summary(
        _safety_telemetry(), rtk_report_age_seconds=None
    )

    assert "rtk_telemetry_stale" not in summary["blockers"]
    assert summary["rtk_report_age_seconds"] is None


def test_rtk_tracker_refreshes_only_when_the_payload_actually_changes() -> None:
    """The tracker must distinguish a steady link from a latched one.

    Bound to a stand-in carrying only the two attributes the method touches, so
    this exercises the real coordinator logic without building a coordinator.
    Asserted through the public age property rather than the private timestamp.
    """
    holder = SimpleNamespace(
        _rtk_fingerprint=None, _rtk_fingerprint_changed_at=time.monotonic() - 60.0
    )
    cls = mammotion_coordinator.MammotionReportUpdateCoordinator
    note = cls.note_rtk_report_seen.__get__(holder, SimpleNamespace)
    age = cls.rtk_report_age_seconds.fget  # type: ignore[attr-defined]

    def device(status: int, stars: int) -> SimpleNamespace:
        return SimpleNamespace(
            report_data=SimpleNamespace(
                rtk=SimpleNamespace(
                    status=status,
                    gps_stars=stars,
                    age=0,
                    lat_std=0.01,
                    lon_std=0.01,
                    l1_satellites=[1, 2],
                    l2_satellites=[3],
                )
            )
        )

    note(device(4, 26))
    assert age(holder) < 1.0  # first sighting resets the clock

    # A byte-identical payload is the latch signature: it must NOT count as
    # fresh, or a frozen feed would look perpetually healthy. Age keeps growing.
    holder._rtk_fingerprint_changed_at -= 120.0  # noqa: SLF001 - test stand-in
    note(device(4, 26))
    assert age(holder) >= 120.0

    # Satellite counts drift continuously on a live link (26 -> 23 was the real
    # transition when the base station rejoined on 2026-08-07).
    note(device(4, 23))
    assert age(holder) < 1.0

    # Absent data must never raise inside the update path.
    holder._rtk_fingerprint_changed_at -= 120.0  # noqa: SLF001 - test stand-in
    note(SimpleNamespace(report_data=None))
    note(None)
    assert age(holder) >= 120.0  # and must not be mistaken for freshness


def test_rtk_age_accessor_tolerates_a_coordinator_without_the_tracker() -> None:
    """Older builds and test doubles must degrade to None, never raise."""
    assert _rtk_report_age_seconds(SimpleNamespace()) is None
    assert _rtk_report_age_seconds(SimpleNamespace(rtk_report_age_seconds=4.5)) == 4.5
    assert (
        _rtk_report_age_seconds(SimpleNamespace(rtk_report_age_seconds="nope")) is None
    )


def _drive_report_probe_clock(
    monkeypatch: pytest.MonkeyPatch,
    coordinator: SimpleNamespace,
    arrival_times: list[float],
) -> dict[str, float]:
    """Run the probe against a fake clock that stamps reports at fixed times.

    The probe polls a monotonic timestamp, so a real clock would make the test
    both slow and flaky. Advance a fake one from inside ``asyncio.sleep`` and
    stamp ``last_report_at`` as each scheduled arrival time is crossed.
    """
    clock = {"t": 0.0}
    handle = coordinator.manager.mower(coordinator.device_name)
    handle.last_report_at = 0.0

    async def fake_sleep(seconds: float) -> None:
        clock["t"] += seconds
        due = [at for at in arrival_times if at <= clock["t"]]
        if due:
            handle.last_report_at = due[-1]

    monkeypatch.setattr(mammotion_services.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    return clock


@pytest.mark.asyncio
async def test_basestation_probe_reports_the_survey_discriminator() -> None:
    """base_moved / base_moving must reach the caller.

    These are the fields that would say whether the base believes its own
    position changed -- the leading explanation for the 2026-08-07 Float
    episode, where corrections flowed and the rover stayed healthy but could
    never resolve.
    """
    coordinator = _pulse_coordinator()
    info = SimpleNamespace(
        rtk_status=1,
        sats_num=28,
        rtk_channel=3,
        rtk_switch=1,
        mqtt_rtk_status=0,
        lora_channel=7,
        wifi_rssi=-55,
        app_connect_type=1,
        basestation_status=2,
        connect_status_since_poweron=99,
        score_info=SimpleNamespace(
            base_score=88, base_leve=2, base_moved=0, base_moving=0
        ),
    )
    coordinator.data.report_data.basestation_info = info

    async def send(command: str, **_kw: object) -> bool:
        # Simulate the base answering by mutating the reduced state, which is
        # what pymammotion's _update_base_data does on base.to_app.
        assert command == "basestation_info"
        info.score_info.base_moved = 1
        return True

    coordinator.async_send_command = send

    result = await _basestation_info_probe(coordinator, wait_seconds=0.0)

    assert result["command_sent"] is True
    assert result["answered"] is True
    assert result["reason"] == "answered"
    assert result["motion_commanded"] is False
    assert result["before"]["score_info"]["base_moved"] == 0
    assert result["best"]["score_info"]["base_moved"] == 1
    assert result["best"]["sats_num"] == 28


@pytest.mark.asyncio
async def test_basestation_probe_does_not_claim_the_base_is_dead() -> None:
    """A payload with no query-only fields is ambiguous, not proof of death.

    The struct is still ``available`` -- pymammotion default-constructs it --
    and the report channel keeps ``connect_status_since_poweron`` populated, so
    "we saw nothing from the base" must never be reported as "the base is dead".
    """
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.basestation_info = SimpleNamespace(
        # Report-channel fields only: exactly what rpt_basestation_info carries.
        basestation_status=0,
        connect_status_since_poweron=2,
        rtk_status=0,
        sats_num=0,
        score_info=SimpleNamespace(
            base_score=0, base_leve=0, base_moved=0, base_moving=0
        ),
    )

    async def send(command: str, **_kw: object) -> bool:
        return True

    coordinator.async_send_command = send

    result = await _basestation_info_probe(coordinator, wait_seconds=0.0)

    assert result["command_sent"] is True
    assert result["answered"] is False
    assert result["reason"] == "no_query_fields_observed"
    # Still reports what it did see, rather than discarding it.
    assert result["final"]["connect_status_since_poweron"] == 2


@pytest.mark.asyncio
async def test_basestation_probe_survives_the_report_channel_clobber() -> None:
    """A reply that arrives and is then overwritten must still be captured.

    ``report_info.py:646`` replaces ``report_data.basestation_info`` wholesale
    from ``rpt_basestation_info``, which carries none of the query-only fields.
    So every report resets ``sats_num`` / ``score_info`` to defaults and wipes a
    reply that arrived first. The 2026-08-07 beta27 run hit exactly this and
    returned an uninterpretable "no change". Polling must keep the reply.
    """
    coordinator = _pulse_coordinator()
    answered = SimpleNamespace(
        rtk_status=4,
        sats_num=30,
        rtk_channel=3,
        rtk_switch=1,
        mqtt_rtk_status=0,
        lora_channel=7,
        wifi_rssi=-55,
        app_connect_type=1,
        basestation_status=2,
        connect_status_since_poweron=2,
        score_info=SimpleNamespace(
            base_score=91, base_leve=2, base_moved=0, base_moving=0
        ),
    )
    clobbered = SimpleNamespace(
        rtk_status=0,
        sats_num=0,
        rtk_channel=0,
        rtk_switch=0,
        mqtt_rtk_status=0,
        lora_channel=0,
        wifi_rssi=0,
        app_connect_type=0,
        basestation_status=0,
        connect_status_since_poweron=2,
        score_info=SimpleNamespace(
            base_score=0, base_leve=0, base_moved=0, base_moving=0
        ),
    )
    coordinator.data.report_data.basestation_info = answered

    async def send(command: str, **_kw: object) -> bool:
        return True

    coordinator.async_send_command = send

    async def clobber() -> None:
        await asyncio.sleep(0.2)
        coordinator.data.report_data.basestation_info = clobbered

    task = asyncio.create_task(clobber())
    result = await _basestation_info_probe(coordinator, wait_seconds=0.5)
    await task

    assert result["answered"] is True
    assert result["reason"] == "answered"
    assert result["best"]["sats_num"] == 30
    assert result["best"]["score_info"]["base_score"] == 91
    # The clobber is reported rather than hidden -- it is the reason a
    # before/after comparison cannot be trusted here.
    assert result["clobbered_after_answer"] is True
    assert result["final"]["sats_num"] == 0
    assert result["samples"] > 1


@pytest.mark.asyncio
async def test_basestation_probe_reads_the_rtk_device_not_only_the_mower() -> None:
    """A reply on the base's own iot_id must still count as an answer.

    ``base.to_app`` frames carrying the RTK device's iot_id are reduced onto
    ``RTKBaseStationDevice`` and never touch the mower's ``report_data``. The
    first corrected live run saw nothing on the mower path while the
    installation had a live base station reporting 26 satellites, so watching
    only the mower would call a healthy base silent.
    """
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.basestation_info = SimpleNamespace(
        basestation_status=0,
        connect_status_since_poweron=2,
        rtk_status=0,
        sats_num=0,
        score_info=SimpleNamespace(
            base_score=0, base_leve=0, base_moved=0, base_moving=0
        ),
    )
    rtk_coordinator = SimpleNamespace(
        data=SimpleNamespace(
            sats_num=26,
            rtk_status=4,
            lora_channel=7,
            wifi_rssi=-72,
            lat=0.5938,
            lon=-1.4795,
            online=True,
            score_info=SimpleNamespace(
                base_score=91, base_leve=2, base_moved=0, base_moving=0
            ),
        )
    )

    async def send(command: str, **_kw: object) -> bool:
        return True

    coordinator.async_send_command = send

    result = await _basestation_info_probe(
        coordinator,
        wait_seconds=0.0,
        rtk_sources=[("rtkbna235279309", rtk_coordinator)],
    )

    assert result["answered"] is True
    assert result["answered_via"] == ["rtk_device:rtkbna235279309"]
    assert result["reason"] == "answered"
    # The mower path genuinely saw nothing; that is reported, not papered over.
    assert _basestation_has_query_fields(result["final"]) is False
    rtk = result["rtk_devices"][0]
    assert rtk["name"] == "rtkbna235279309"
    assert rtk["answered"] is True
    assert rtk["best"]["sats_num"] == 26
    assert rtk["best"]["score_info"]["base_score"] == 91
    assert result["motion_commanded"] is False


@pytest.mark.asyncio
async def test_basestation_probe_reports_a_refused_command() -> None:
    """An offline device must not look like a silent base station."""
    coordinator = _pulse_coordinator()

    async def send(command: str, **_kw: object) -> bool:
        return False

    coordinator.async_send_command = send

    result = await _basestation_info_probe(coordinator, wait_seconds=0.0)

    assert result["command_sent"] is False
    assert result["reason"] == "command_refused_device_offline_or_unavailable"
    assert result["answered"] is False


@pytest.mark.asyncio
async def test_basestation_probe_commands_no_motion() -> None:
    """The probe is read-only with respect to motion."""
    coordinator = _pulse_coordinator()
    sent: list[str] = []

    async def send(command: str, **_kw: object) -> bool:
        sent.append(command)
        return True

    coordinator.async_send_command = send

    await _basestation_info_probe(coordinator, wait_seconds=0.0)

    assert sent == ["basestation_info"]
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()
    handle = coordinator.manager.mower(coordinator.device_name)
    handle.commands.send_movement.assert_not_called()


@pytest.mark.asyncio
async def test_report_stream_probe_measures_arrival_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe reports the feed's own cadence, not its polling cadence."""
    coordinator = _pulse_coordinator()
    # Reports every 200 ms. A 1.1 s sampler could never resolve this, which is
    # the whole reason the probe exists.
    _drive_report_probe_clock(monkeypatch, coordinator, [0.2, 0.4, 0.6, 0.8, 1.0])

    result = await _report_stream_probe(
        coordinator,
        period_ms=200,
        no_change_period_ms=200,
        duration_seconds=1.5,
    )

    assert result["reason"] == "completed"
    assert result["reports_observed"] == 5
    assert result["summary"]["median_ms"] == pytest.approx(200.0, abs=1.0)
    assert result["aggregate_period_heuristic"] is True
    assert result["honoured_requested_period"] is None
    assert result["period_classification_reason"] == "isolated_subscription_required"
    # The requested period must reach the device as a protocol field.
    coordinator.manager.request_iot_sync_continuous.assert_awaited_once()
    kwargs = coordinator.manager.request_iot_sync_continuous.await_args.kwargs
    assert kwargs["period"] == 200
    assert kwargs["no_change_period"] == 200


@pytest.mark.asyncio
async def test_report_stream_probe_detects_a_device_ignoring_the_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device that clamps to 1 s must not be recorded as honouring 200 ms."""
    coordinator = _pulse_coordinator()
    _drive_report_probe_clock(monkeypatch, coordinator, [1.0, 2.0, 3.0])

    result = await _report_stream_probe(
        coordinator,
        period_ms=200,
        no_change_period_ms=200,
        duration_seconds=3.5,
    )

    assert result["summary"]["median_ms"] == pytest.approx(1000.0, abs=1.0)
    assert result["aggregate_period_heuristic"] is False
    assert result["honoured_requested_period"] is None


@pytest.mark.asyncio
async def test_isolated_probe_classifies_only_position_payload_p95(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Firmware-period classification requires 100 ordered position payloads."""
    coordinator = _pulse_coordinator()
    queue: asyncio.Queue[SimpleNamespace] = asyncio.Queue()
    baseline = time.monotonic()
    for sequence in range(1, 101):
        received_at = baseline + sequence * 0.2
        queue.put_nowait(
            SimpleNamespace(
                sequence=sequence,
                epoch=1,
                received_at_monotonic=received_at,
                decoded_at_monotonic=received_at + 0.0002,
                broker_completed_at_monotonic=received_at + 0.0004,
                reducer_completed_at_monotonic=received_at + 0.0006,
                state_applied_at_monotonic=received_at + 0.0008,
                published_at_monotonic=received_at + 0.001,
                valid_for_motion=True,
            )
        )
    stream = SimpleNamespace(queue=queue, dropped_samples=0, close=lambda: None)
    coordinator.open_position_sample_stream = lambda **_kwargs: stream
    handle = coordinator.manager.mower(coordinator.device_name)

    lease = SimpleNamespace(
        owner="report_stream_probe",
        lease_id=1,
        acquired_at_monotonic=baseline,
        background_stop_enqueued=True,
        background_stop_enqueued_at_monotonic=baseline,
    )

    class _ExclusiveContext:
        async def __aenter__(self) -> SimpleNamespace:
            return lease

        async def __aexit__(self, *_args: object) -> None:
            return None

    handle.exclusive_report_subscription = lambda _owner: _ExclusiveContext()
    handle.report_subscription_generation = 0
    handle.report_subscription_lease_is_current = lambda candidate: candidate is lease

    def _begin_generation(candidate: object) -> SimpleNamespace:
        assert candidate is lease
        handle.report_subscription_generation = 1
        return SimpleNamespace(
            owner=lease.owner,
            lease_id=lease.lease_id,
            generation=1,
            requested_at_monotonic=baseline,
            baseline_position_sequence=0,
            baseline_position_epoch=1,
            baseline_last_report_at=handle.last_report_at,
        )

    handle.begin_report_subscription_generation = _begin_generation

    async def _observed(*_args: object) -> tuple[list[float], dict[str, list[float]]]:
        return [], {"position": [], "rtk": [], "vio": []}

    async def _collected(
        *_args: object, **_kwargs: object
    ) -> list[tuple[object, float]]:
        records: list[tuple[object, float]] = []
        while not queue.empty():
            sample = queue.get_nowait()
            records.append((sample, sample.published_at_monotonic + 0.0002))
        return records

    monkeypatch.setattr(mammotion_services, "_observe_report_arrivals", _observed)
    monkeypatch.setattr(mammotion_services, "_collect_position_records", _collected)
    monkeypatch.setattr(
        mammotion_services,
        "_settle_ble_command_queue",
        AsyncMock(return_value={"settled": True}),
    )

    result = await _report_stream_probe(
        coordinator,
        period_ms=200,
        no_change_period_ms=200,
        duration_seconds=20.0,
        isolated=True,
    )

    assert result["position_payloads"]["observed"] == 100
    assert result["position_payloads"]["p95_interval_ms"] == pytest.approx(200.0)
    assert result["position_payload_cell_meets_period_criterion"] is True
    assert result["honoured_requested_period"] is None
    assert result["period_classification_reason"] == (
        "three_randomized_repeats_required"
    )


@pytest.mark.asyncio
async def test_isolated_probe_readiness_starts_at_the_start_flush_not_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame still in flight from the previous configuration is not readiness.

    `MammotionClient.request_iot_sync_continuous` returns when the command has
    been ENQUEUED on `DeviceCommandQueue`, not when the device acknowledged it.
    Anything received before that queue drains may still belong to the
    configuration this generation replaces, so the evidence boundary is the
    post-settle flush -- otherwise a stale payload silently certifies a
    transition that never happened, which is the exact failure the 30-transition
    acceptance run is meant to detect.
    """
    coordinator = _pulse_coordinator()
    queue: asyncio.Queue[SimpleNamespace] = asyncio.Queue()
    stream = SimpleNamespace(queue=queue, dropped_samples=0, close=lambda: None)
    coordinator.open_position_sample_stream = lambda **_kwargs: stream
    handle = coordinator.manager.mower(coordinator.device_name)

    lease = SimpleNamespace(
        owner="report_stream_probe",
        lease_id=1,
        acquired_at_monotonic=time.monotonic(),
        background_stop_enqueued=True,
        background_stop_enqueued_at_monotonic=time.monotonic(),
    )

    class _ExclusiveContext:
        async def __aenter__(self) -> SimpleNamespace:
            return lease

        async def __aexit__(self, *_args: object) -> None:
            return None

    handle.exclusive_report_subscription = lambda _owner: _ExclusiveContext()
    handle.report_subscription_generation = 0
    handle.report_subscription_lease_is_current = lambda candidate: candidate is lease

    def _begin_generation(candidate: object) -> SimpleNamespace:
        assert candidate is lease
        handle.report_subscription_generation = 1
        return SimpleNamespace(
            owner=lease.owner,
            lease_id=lease.lease_id,
            generation=1,
            requested_at_monotonic=time.monotonic(),
            baseline_position_sequence=0,
            baseline_position_epoch=1,
            baseline_last_report_at=handle.last_report_at,
        )

    handle.begin_report_subscription_generation = _begin_generation

    def _sample(sequence: int) -> SimpleNamespace:
        received = time.monotonic()
        return SimpleNamespace(
            sequence=sequence,
            epoch=1,
            received_at_monotonic=received,
            decoded_at_monotonic=received,
            broker_completed_at_monotonic=received,
            reducer_completed_at_monotonic=received,
            state_applied_at_monotonic=received,
            published_at_monotonic=received,
            valid_for_motion=True,
        )

    async def _late_arrival() -> None:
        await asyncio.sleep(0.05)
        queue.put_nowait(_sample(2))

    async def _settled(_coordinator: object) -> dict[str, object]:
        # Ordered, valid, and in this generation's sequence -- but received while
        # the START was still queued, so it proves nothing about the new config.
        queue.put_nowait(_sample(1))
        asyncio.get_running_loop().create_task(_late_arrival())
        await asyncio.sleep(0.02)
        return {"settled": True}

    async def _observed(*_args: object) -> tuple[list[float], dict[str, list[float]]]:
        return [], {"position": [], "rtk": [], "vio": []}

    async def _collected(
        *_args: object, **_kwargs: object
    ) -> list[tuple[object, float]]:
        return []

    monkeypatch.setattr(mammotion_services, "_observe_report_arrivals", _observed)
    monkeypatch.setattr(mammotion_services, "_collect_position_records", _collected)
    monkeypatch.setattr(mammotion_services, "_settle_ble_command_queue", _settled)

    result = await _report_stream_probe(
        coordinator,
        period_ms=1000,
        no_change_period_ms=1000,
        duration_seconds=0.0,
        isolated=True,
        readiness_timeout_seconds=1.0,
    )

    assert result["position_readiness"]["ready"] is True
    # Sequence 1 was consumed and rejected as pre-flush; only 2 can certify.
    assert result["position_readiness"]["first_position_sequence"] == 2
    assert (
        result["subscription_command_flushed_at_monotonic"]
        > result["subscription_generation"]["requested_at_monotonic"]
    )


@pytest.mark.asyncio
async def test_position_subscription_readiness_distinguishes_generic_stall() -> None:
    """Generic traffic cannot substitute for a current-generation position."""
    lease = object()
    generation = SimpleNamespace(
        generation=4,
        baseline_position_sequence=10,
        baseline_position_epoch=2,
        baseline_last_report_at=100.0,
        requested_at_monotonic=time.monotonic(),
    )
    handle = SimpleNamespace(
        report_subscription_generation=4,
        last_report_at=101.0,
        position_epoch=2,
        report_subscription_lease_is_current=lambda candidate: candidate is lease,
    )
    stream = SimpleNamespace(queue=asyncio.Queue(), dropped_samples=0)

    sample, consumed_at, reason = await _wait_for_position_subscription_ready(
        handle,
        stream,
        generation,
        lease=lease,
        timeout_seconds=0.01,
    )

    assert sample is None
    assert consumed_at is None
    assert reason == "position_channel_stalled"


@pytest.mark.asyncio
async def test_position_subscription_readiness_refuses_generation_change() -> None:
    """Evidence cannot cross a report-configuration generation boundary."""
    lease = object()
    generation = SimpleNamespace(
        generation=4,
        baseline_position_sequence=10,
        baseline_position_epoch=2,
        baseline_last_report_at=100.0,
        requested_at_monotonic=time.monotonic(),
    )
    handle = SimpleNamespace(
        report_subscription_generation=5,
        last_report_at=100.0,
        position_epoch=2,
        report_subscription_lease_is_current=lambda candidate: candidate is lease,
    )
    stream = SimpleNamespace(queue=asyncio.Queue(), dropped_samples=0)

    sample, consumed_at, reason = await _wait_for_position_subscription_ready(
        handle,
        stream,
        generation,
        lease=lease,
        timeout_seconds=1.0,
    )

    assert sample is None
    assert consumed_at is None
    assert reason == "report_subscription_generation_changed"


@pytest.mark.asyncio
async def test_position_subscription_readiness_requires_post_ack_evidence() -> None:
    """A late frame from the old config cannot make the new START ready."""
    lease = object()
    generation = SimpleNamespace(
        generation=4,
        baseline_position_sequence=10,
        baseline_position_epoch=2,
        baseline_last_report_at=100.0,
        requested_at_monotonic=10.0,
    )
    handle = SimpleNamespace(
        report_subscription_generation=4,
        last_report_at=100.0,
        position_epoch=2,
        report_subscription_lease_is_current=lambda candidate: candidate is lease,
    )
    queue: asyncio.Queue[SimpleNamespace] = asyncio.Queue()
    queue.put_nowait(
        SimpleNamespace(
            sequence=11,
            epoch=2,
            received_at_monotonic=10.1,
            valid_for_motion=True,
        )
    )
    expected = SimpleNamespace(
        sequence=12,
        epoch=2,
        received_at_monotonic=10.3,
        valid_for_motion=True,
    )
    queue.put_nowait(expected)
    stream = SimpleNamespace(queue=queue, dropped_samples=0)

    sample, consumed_at, reason = await _wait_for_position_subscription_ready(
        handle,
        stream,
        generation,
        lease=lease,
        timeout_seconds=1.0,
        not_before_monotonic=10.2,
    )

    assert sample is expected
    assert consumed_at is not None
    assert reason is None


@pytest.mark.asyncio
async def test_position_subscription_readiness_refuses_epoch_change() -> None:
    """A reconnect invalidates the generation even when no sample follows it."""
    lease = object()
    generation = SimpleNamespace(
        generation=4,
        baseline_position_sequence=10,
        baseline_position_epoch=2,
        baseline_last_report_at=100.0,
        requested_at_monotonic=time.monotonic(),
    )
    handle = SimpleNamespace(
        report_subscription_generation=4,
        last_report_at=100.0,
        position_epoch=3,
        report_subscription_lease_is_current=lambda candidate: candidate is lease,
    )
    stream = SimpleNamespace(queue=asyncio.Queue(), dropped_samples=0)

    sample, consumed_at, reason = await _wait_for_position_subscription_ready(
        handle,
        stream,
        generation,
        lease=lease,
        timeout_seconds=1.0,
    )

    assert sample is None
    assert consumed_at is None
    assert reason == "position_epoch_changed"


@pytest.mark.asyncio
async def test_position_subscription_readiness_refuses_pre_wait_queue_drop() -> None:
    """A replacement during command dispatch invalidates later readiness."""
    lease = object()
    generation = SimpleNamespace(
        generation=4,
        baseline_position_sequence=10,
        baseline_position_epoch=2,
        baseline_last_report_at=100.0,
        requested_at_monotonic=time.monotonic(),
    )
    handle = SimpleNamespace(
        report_subscription_generation=4,
        last_report_at=100.0,
        position_epoch=2,
        report_subscription_lease_is_current=lambda candidate: candidate is lease,
    )
    stream = SimpleNamespace(queue=asyncio.Queue(), dropped_samples=1)

    sample, consumed_at, reason = await _wait_for_position_subscription_ready(
        handle,
        stream,
        generation,
        lease=lease,
        timeout_seconds=1.0,
        baseline_dropped_samples=0,
    )

    assert sample is None
    assert consumed_at is None
    assert reason == "position_evidence_gap"


def test_position_stage_latency_summary_attributes_every_boundary() -> None:
    """The diagnostic attributes transport, reducer, and consumer latency."""
    sample = SimpleNamespace(
        received_at_monotonic=10.000,
        decoded_at_monotonic=10.001,
        broker_completed_at_monotonic=10.003,
        reducer_completed_at_monotonic=10.006,
        state_applied_at_monotonic=10.010,
        published_at_monotonic=10.015,
    )

    summary = _position_stage_latency_summary([(sample, 10.021)])

    assert summary["receipt_to_decode"]["p50"] == pytest.approx(1.0)
    assert summary["decode_to_broker"]["p50"] == pytest.approx(2.0)
    assert summary["broker_to_reducer"]["p50"] == pytest.approx(3.0)
    assert summary["reducer_to_state_apply"]["p50"] == pytest.approx(4.0)
    assert summary["state_apply_to_publication"]["p50"] == pytest.approx(5.0)
    assert summary["publication_to_consumption"]["p50"] == pytest.approx(6.0)
    assert summary["receipt_to_consumption"]["p50"] == pytest.approx(21.0)


@pytest.mark.asyncio
async def test_report_stream_sequence_reuses_one_lease_for_every_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adjacent stationary transitions do not release and rearm ownership."""
    lease = SimpleNamespace(
        owner="report_stream_sequence_probe",
        lease_id=9,
        acquired_at_monotonic=1.0,
        background_stop_enqueued=True,
        background_stop_enqueued_at_monotonic=1.1,
    )
    lifecycle: list[str] = []

    class _ExclusiveContext:
        async def __aenter__(self) -> SimpleNamespace:
            lifecycle.append("enter")
            return lease

        async def __aexit__(self, *_args: object) -> None:
            lifecycle.append("exit")

    handle = SimpleNamespace(
        exclusive_report_subscription=lambda _owner: _ExclusiveContext()
    )
    coordinator = SimpleNamespace(
        device_name="Luba-Test",
        manager=SimpleNamespace(mower=lambda _name: handle),
        manual_motion_owner=None,
    )
    used_leases: list[object] = []

    async def _cell(*_args: object, **kwargs: object) -> dict[str, object]:
        used_leases.append(kwargs["report_lease"])
        return {
            "reason": "completed",
            "subscription_started": True,
            "subscription_stopped": True,
            "position_readiness": {"ready": True},
            "position_payloads": {"dropped_samples": 0, "sequence_gaps": 0},
        }

    monkeypatch.setattr(mammotion_services, "_report_stream_probe", _cell)

    result = await _report_stream_sequence_probe(
        coordinator,
        periods_ms=[1000, 500, 1000],
        observation_seconds=0.0,
        readiness_timeout_seconds=3.5,
    )

    assert result["complete"] is True
    assert result["failed_cells"] == []
    assert used_leases == [lease, lease, lease]
    assert lifecycle == ["enter", "exit"]


@pytest.mark.asyncio
async def test_report_stream_probe_sees_through_unrelated_traffic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fast median must not read as 'honoured' when the worst gap is a clamp.

    This is the live 2026-08-07 shape: requesting 200 ms produced a median well
    under 200 ms because other inbound LubaMsg traffic lands between the
    periodic reports, while the largest gap stayed near 1 s -- the device
    clamping. Judging on the median alone called that a success.
    """
    coordinator = _pulse_coordinator()
    # Bursty traffic, then a ~1 s hole: the clamp the median hides.
    _drive_report_probe_clock(
        monkeypatch, coordinator, [0.05, 0.10, 0.15, 1.10, 1.15, 1.20]
    )

    result = await _report_stream_probe(
        coordinator,
        period_ms=200,
        no_change_period_ms=200,
        duration_seconds=1.6,
    )

    assert result["summary"]["median_ms"] < 200
    assert result["summary"]["max_ms"] > 500
    assert result["aggregate_period_heuristic"] is False
    assert result["honoured_requested_period"] is None


@pytest.mark.asyncio
async def test_report_stream_probe_separates_channels_from_total_traffic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent channel must be visible even while total traffic looks healthy.

    This is the 2026-08-07 shape exactly: messages kept arriving at ~2 Hz while
    the RTK channel had been frozen for three hours. Counting LubaMsgs cannot
    see that; attributing arrivals per channel can.
    """
    coordinator = _pulse_coordinator()
    clock = _drive_report_probe_clock(
        monkeypatch, coordinator, [0.2, 0.4, 0.6, 0.8, 1.0]
    )

    # Position advances on every poll; RTK is latched and never changes.
    location = SimpleNamespace(real_pos_x=1.0, real_pos_y=1.0, real_toward=0.0)
    coordinator.data.report_data.locations = [location]
    real_sleep = mammotion_services.asyncio.sleep

    async def moving_sleep(seconds: float) -> None:
        await real_sleep(seconds)
        location.real_pos_x = 1.0 + clock["t"]

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", moving_sleep)

    result = await _report_stream_probe(
        coordinator,
        period_ms=200,
        no_change_period_ms=200,
        duration_seconds=1.2,
    )

    channels = result["channels"]
    assert channels["position"]["updates"] > 0
    assert channels["rtk"]["updates"] == 0
    assert channels["rtk"]["note"] == "no updates observed in the window"
    # Total traffic was healthy the whole time, which is precisely why the
    # per-channel split is needed.
    assert result["reports_observed"] > 0


@pytest.mark.asyncio
async def test_report_stream_probe_commands_no_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe is read-only: it must never dispatch a movement command."""
    coordinator = _pulse_coordinator()
    _drive_report_probe_clock(monkeypatch, coordinator, [0.5, 1.0])

    await _report_stream_probe(
        coordinator,
        period_ms=1000,
        no_change_period_ms=1000,
        duration_seconds=1.5,
    )

    handle = coordinator.manager.mower(coordinator.device_name)
    coordinator.async_move_forward.assert_not_called()
    coordinator.async_move_back.assert_not_called()
    coordinator.manager.send_command_with_args.assert_not_called()
    handle.commands.send_movement.assert_not_called()


@pytest.mark.asyncio
async def test_report_stream_probe_always_stops_its_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short-period stream left running is a standing BLE load."""
    coordinator = _pulse_coordinator()
    _drive_report_probe_clock(monkeypatch, coordinator, [0.5])

    ok = await _report_stream_probe(
        coordinator,
        period_ms=500,
        no_change_period_ms=500,
        duration_seconds=1.0,
    )
    assert ok["subscription_stopped"] is True
    coordinator.manager.request_iot_sync_continuous_stop.assert_awaited_once()

    # ... and on the failure path too, which is where a leak would actually hurt.
    coordinator.manager.request_iot_sync_continuous_stop.reset_mock()

    async def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("BLE link is not ready for motion")

    monkeypatch.setattr(
        mammotion_services, "_settle_ble_command_queue", boom, raising=True
    )
    failed = await _report_stream_probe(
        coordinator,
        period_ms=500,
        no_change_period_ms=500,
        duration_seconds=1.0,
    )
    assert failed["reason"].startswith("RuntimeError")
    assert failed["subscription_stopped"] is True
    coordinator.manager.request_iot_sync_continuous_stop.assert_awaited_once()

    # A send may reach the mower and still raise before the caller sees an ACK.
    # Teardown therefore follows the attempt boundary, not only acknowledged STARTs.
    coordinator.manager.request_iot_sync_continuous_stop.reset_mock()
    coordinator.manager.request_iot_sync_continuous.side_effect = RuntimeError(
        "ACK lost"
    )
    uncertain = await _report_stream_probe(
        coordinator,
        period_ms=500,
        no_change_period_ms=500,
        duration_seconds=1.0,
    )
    assert uncertain["subscription_started"] is False
    assert uncertain["subscription_stopped"] is True
    coordinator.manager.request_iot_sync_continuous_stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_report_stream_probe_refuses_during_a_motion_session() -> None:
    """Never reconfigure the report feed underneath a live motion run."""
    coordinator = _pulse_coordinator()
    coordinator.manual_motion_owner = "raw_pymammotion_execute_multi_segment"

    result = await _report_stream_probe(
        coordinator,
        period_ms=200,
        no_change_period_ms=200,
        duration_seconds=5.0,
    )

    assert result["reason"] == "manual_motion_session_active"
    assert result["subscription_started"] is False
    coordinator.manager.request_iot_sync_continuous.assert_not_called()


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
        # No report_data.locations here, so there is no map checksum to report.
        "map_bol_hash": None,
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


def _report_location_coordinator(
    *, zone_hash: int, bol_hash: int, pos_type: int = 1
) -> SimpleNamespace:
    """Build a coordinator whose only position source is report_data.locations[0]."""
    return SimpleNamespace(
        is_online=lambda: True,
        data=SimpleNamespace(
            # location.* is unset so nothing can mask which field the snapshot
            # actually read off the report message.
            location=SimpleNamespace(
                orientation=None, position_type=None, work_zone=None
            ),
            report_data=SimpleNamespace(
                dev=SimpleNamespace(sys_status=11, charge_state=2, blade_state=0),
                rtk=SimpleNamespace(status=4, pos_level=0),
                locations=[
                    SimpleNamespace(
                        real_pos_x=12_345,
                        real_pos_y=-67_890,
                        real_toward=900_000,
                        pos_type=pos_type,
                        zone_hash=zone_hash,
                        bol_hash=bol_hash,
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


def test_report_location_zone_hash_is_field_5_not_the_map_checksum() -> None:
    """``zone_hash`` comes from proto field 5, never from ``bol_hash`` (field 6).

    ``rpt_dev_location`` reports both.  Reading the checksum as the zone made
    every zone-based guard inert, because a map checksum is non-zero whenever
    the device has any map at all.  Live proof (2026-07-24, docked mower):
    ``zone_hash`` 0 and ``bol_hash`` 8311072749804434520 in the same message.
    """
    position = _custom_path_telemetry_snapshot(
        _report_location_coordinator(zone_hash=456, bol_hash=8311072749804434520)
    )["position"]

    assert position["source"] == "report_data.locations[0]"
    assert position["zone_hash"] == 456
    # The checksum stays visible for map-sync forensics, under its own name.
    assert position["map_bol_hash"] == "8311072749804434520"


def test_zone_hash_zero_is_not_masked_by_a_nonzero_map_checksum() -> None:
    """A mower outside any zone reads zone_hash 0 even with a map loaded.

    This is the live docked case.  While the snapshot read ``bol_hash`` the
    position looked zone-tagged, so ``_is_valid_motion_position`` and the
    ``zone_hash_unavailable`` degradation reason could never trip.
    """
    position = _custom_path_telemetry_snapshot(
        _report_location_coordinator(zone_hash=0, bol_hash=8311072749804434520)
    )["position"]

    assert position["zone_hash"] == 0
    assert position["map_bol_hash"] == "8311072749804434520"
    # AREA_INSIDE alone must not be enough once the zone is genuinely unknown.
    assert position["valid_for_motion"] is False

    degradation = _manual_velocity_quality_degradation(
        baseline={"position": {"zone_hash": 789, "pos_type_label": "AREA_INSIDE"}},
        current={"position": position},
    )
    assert "zone_hash_unavailable" in degradation["reasons"]
    assert degradation["degraded"] is True


def test_zone_hash_change_mid_run_is_detected() -> None:
    """Leaving one zone for another is a real degradation signal.

    The map checksum is constant across a run, so this could never fire before.
    """
    baseline = _custom_path_telemetry_snapshot(
        _report_location_coordinator(zone_hash=111, bol_hash=8311072749804434520)
    )
    current = _custom_path_telemetry_snapshot(
        _report_location_coordinator(zone_hash=222, bol_hash=8311072749804434520)
    )

    degradation = _manual_velocity_quality_degradation(
        baseline=baseline, current=current
    )
    assert "zone_hash_changed" in degradation["reasons"]


def test_stale_dock_pose_is_rejected_when_zone_hash_is_zero() -> None:
    """The (0,0)/AREA_OUT stale pose must not be accepted as a real position.

    ``_is_stale_zero_area_out_pose`` needs pos_type 0 *and* zone_hash 0.  Fed
    the map checksum it never saw a zero, so the guard was dead code.
    """
    coordinator = _report_location_coordinator(
        zone_hash=0, bol_hash=8311072749804434520, pos_type=0
    )
    coordinator.data.report_data.locations[0].real_pos_x = 0
    coordinator.data.report_data.locations[0].real_pos_y = 0

    position = _custom_path_telemetry_snapshot(coordinator)["position"]

    assert position["source"] != "report_data.locations[0]"
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
    assert (
        descriptions["enabled_task_count"].value_fn(coordinator, coordinator.data) == 0
    )
    assert descriptions["area_count"].value_fn(coordinator, coordinator.data) == 1
    assert (
        descriptions["map_area_name_count"].value_fn(coordinator, coordinator.data) == 1
    )
    assert descriptions["last_map_sync"].value_fn(coordinator, coordinator.data) is None
    assert (
        descriptions["last_task_sync"].value_fn(coordinator, coordinator.data) is None
    )
    assert (
        descriptions["last_map_task_error"].value_fn(coordinator, coordinator.data)
        == "task_sync: RuntimeError"
    )
    assert (
        descriptions["active_transport"].value_fn(coordinator, coordinator.data)
        == "ble"
    )
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
        descriptions["last_command_failure_reason"].value_fn(
            coordinator, coordinator.data
        )
        == "set_car_wiper:GatewayTimeoutException"
    )
    assert (
        descriptions["last_camera_stream_failure_code"].value_fn(
            coordinator, coordinator.data
        )
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
async def test_sync_maps_refuses_during_a_guarded_motion_run() -> None:
    """The sync_maps button must not start a saga mid-run.

    This is the path that actually fires sagas: ``button.<mower>_sync_maps`` is
    pressable at any moment, including during a run (live history shows a press
    at 2026-07-22T23:49:09 followed by ``syncing`` one second later).  A
    ``MapFetchSaga`` holds the command queue exclusively, and motion commands
    are ``Priority.NORMAL`` — they block on that slot — so a press mid-run
    stalls the run's pulses.  Refuse loudly rather than queue behind it.
    """
    coordinator = SimpleNamespace(
        manager=SimpleNamespace(start_map_sync=AsyncMock()),
        device_name="Luba-Test",
        last_map_sync=None,
        last_map_task_error=None,
        manual_motion_owner="raw_pymammotion_execute_vector_segment",
    )
    coordinator._raise_if_manual_motion_in_progress = (  # noqa: SLF001
        lambda action: (
            MammotionBaseUpdateCoordinator._raise_if_manual_motion_in_progress(  # noqa: SLF001
                coordinator, action
            )
        )
    )

    with pytest.raises(HomeAssistantError) as excinfo:
        await MammotionReportUpdateCoordinator.async_sync_maps(coordinator)

    # The message must name the owner so the operator knows what to wait for.
    assert "raw_pymammotion_execute_vector_segment" in str(excinfo.value)
    coordinator.manager.start_map_sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_force_map_resync_refuses_during_a_guarded_motion_run() -> None:
    """force_map_resync refuses up front and sends nothing.

    Every step it runs (RTK/dock refresh, area-name fetch, the saga itself)
    enqueues device commands, so it has to bail before the first one rather
    than partway through.  Reported as a result, not raised — this is a
    response service whose caller wants the diagnostics.
    """
    fake = _force_resync_self(
        manual_motion_owner="raw_pymammotion_execute_multi_segment"
    )

    result = await MammotionBaseUpdateCoordinator.async_force_map_resync(fake)

    assert result["error"] == "manual_motion_in_progress"
    assert result["busy_owner"] == "raw_pymammotion_execute_multi_segment"
    assert result["steps"] == ["refused_manual_motion_in_progress"]
    # Nothing was sent to the device on any of the three command paths.
    fake.async_rtk_dock_location.assert_not_awaited()
    fake.async_get_area_list.assert_not_awaited()
    fake.async_sync_maps.assert_not_awaited()
    fake.manager.regenerate_stale_geojson.assert_not_called()


def _map_sync_gate_self(**overrides: object) -> SimpleNamespace:
    """Build a coordinator-shaped self for the _should_start_map_sync gate."""
    fake = SimpleNamespace(
        manual_motion_owner=None,
        last_map_sync=None,
        last_map_sync_bol_hash=None,
    )
    for key, value in overrides.items():
        setattr(fake, key, value)
    return fake


def _should_sync(fake: SimpleNamespace, bol_hash: int) -> bool:
    """Call the real gate against a coordinator-shaped stub."""
    return MammotionBaseUpdateCoordinator._should_start_map_sync(fake, bol_hash)  # noqa: SLF001


def test_map_sync_gate_allows_the_first_attempt() -> None:
    """With no prior attempt recorded, a needed sync runs."""
    assert _should_sync(_map_sync_gate_self(), 8311072749804434520) is True


def test_map_sync_gate_backs_off_a_non_converging_retry() -> None:
    """A repeat attempt against the same bol_hash waits out MAP_INTERVAL.

    ``is_map_synced()`` can stay False indefinitely on a map that is complete
    and containment-usable (live 2026-07-24), which otherwise re-ran the
    exclusive saga every REPORT_INTERVAL (5 min) forever.
    """
    now = datetime.datetime.now(datetime.UTC)
    recent = _map_sync_gate_self(
        last_map_sync=now - datetime.timedelta(minutes=5),
        last_map_sync_bol_hash=8311072749804434520,
    )
    assert _should_sync(recent, 8311072749804434520) is False

    stale = _map_sync_gate_self(
        last_map_sync=now - mammotion_coordinator.MAP_INTERVAL,
        last_map_sync_bol_hash=8311072749804434520,
    )
    assert _should_sync(stale, 8311072749804434520) is True


def test_map_sync_gate_never_delays_a_real_map_edit() -> None:
    """A changed bol_hash means a device-side map edit — sync immediately."""
    fake = _map_sync_gate_self(
        last_map_sync=datetime.datetime.now(datetime.UTC),
        last_map_sync_bol_hash=8311072749804434520,
    )
    assert _should_sync(fake, 1234567890123456789) is True


def test_map_sync_gate_yields_to_a_guarded_motion_run() -> None:
    """No exclusive saga while a motion run owns the mower.

    ``MapFetchSaga`` holds the command queue exclusively and regular commands
    are ``Priority.NORMAL``, so a saga starting mid-run blocks the run's pulses
    and collapses the 200 ms refresh cadence the mower needs to keep driving.
    """
    fake = _map_sync_gate_self(
        manual_motion_owner="raw_pymammotion_execute_vector_segment"
    )
    # Even the otherwise-unconditional first-attempt case must yield.
    assert _should_sync(fake, 8311072749804434520) is False

    fake.manual_motion_owner = None
    assert _should_sync(fake, 8311072749804434520) is True


def test_update_loop_only_starts_a_map_sync_through_the_gate() -> None:
    """The per-tick sync in _async_update_data must go through the gate.

    ``_async_update_data`` needs a full HA instance to exercise, so the gate
    logic lives in ``_should_start_map_sync`` (unit-tested above) and this
    pins the wiring — mirroring how ``_async_opportunistic_ble_reconnect`` was
    extracted for the same reason.  Without the guard the exclusive saga runs
    every REPORT_INTERVAL forever whenever ``is_map_synced()`` stays False.
    """
    source = pathlib.Path(mammotion_coordinator.__file__).read_text()
    tree = ast.parse(source)

    update_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_async_update_data"
    ]
    assert update_loops, "no _async_update_data found"

    guarded = 0
    for loop in update_loops:
        for node in ast.walk(loop):
            if not isinstance(node, ast.If):
                continue
            calls_sync = any(
                isinstance(inner, ast.Attribute) and inner.attr == "start_map_sync"
                for inner in ast.walk(node)
            )
            if not calls_sync:
                continue
            gated = any(
                isinstance(inner, ast.Attribute)
                and inner.attr == "_should_start_map_sync"
                for inner in ast.walk(node.test)
            )
            assert gated, "start_map_sync in _async_update_data is not gated"
            guarded += 1

    assert guarded == 1, f"expected exactly one gated map sync, found {guarded}"


def _short_circuit_self(
    *,
    online: bool = True,
    enabled: bool = True,
    sys_status: object = 11,
    update_failures: int = 0,
) -> SimpleNamespace:
    """Duck-typed ``self`` for ``_async_short_circuit_update``.

    ``ble_mac`` is deliberately empty so the bluetooth-scanner lookup is skipped
    and the helper needs no HA instance.
    """
    device = SimpleNamespace(
        enabled=enabled,
        mower_state=SimpleNamespace(ble_mac=""),
        report_data=SimpleNamespace(dev=SimpleNamespace(sys_status=sys_status)),
    )
    return SimpleNamespace(
        manager=SimpleNamespace(
            get_device_by_name=lambda _name: device,
            mower=lambda _name: SimpleNamespace(),
            update_ble_device=AsyncMock(),
        ),
        device_name="Luba-Test",
        data=device,
        update_failures=update_failures,
        is_online=lambda: online,
        get_coordinator_data=lambda dev: dev,
        hass=SimpleNamespace(),
        clear_update_failures=lambda: None,
    )


@pytest.mark.asyncio
async def test_short_circuit_update_returns_none_on_the_healthy_path() -> None:
    """A healthy tick must signal "carry on" with None, not truthy data.

    Regression for 2026-07-25. This helper used to be ``_async_update_data`` and
    ended with ``return self.data``; every subclass opened with
    ``if data := await super()._async_update_data(): return data``. Since
    MowingDevice/MowerInfo/Maintain define neither ``__bool__`` nor ``__len__``,
    that value was always truthy, so the early return fired on every healthy
    tick and everything after it in all five subclasses was unreachable in
    steady state -- it only ran once per HA start, while ``self.data`` was still
    None.

    That silently disabled the per-tick map-sync check (a device-side map edit
    was never picked up until a restart) and the opportunistic BLE reconnect
    (the mower could sit on cloud transport at healthy RSSI indefinitely).
    Confirmed live with DEBUG on: HA logged "Finished fetching mammotion data in
    0.000 seconds (success: True)" every tick while the LOGGER.debug three lines
    past the early return never appeared once.
    """
    healthy = _short_circuit_self()

    result = await MammotionBaseUpdateCoordinator._async_short_circuit_update(  # noqa: SLF001
        healthy
    )

    assert result is None, (
        "healthy tick returned data, so every subclass early-returns and the "
        "map-sync / BLE-reconnect work after it never runs"
    )
    # The trap that made the bug invisible: the value it used to return is a
    # perfectly ordinary truthy object, so `if data := ...` could never tell
    # "carry on" from "stop here".
    assert bool(healthy.data) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"enabled": False}, "device disabled"),
        ({"online": False}, "device offline"),
        ({"sys_status": WorkMode.MODE_UPDATING}, "mid firmware update"),
        ({"update_failures": 6}, "failing repeatedly"),
    ],
)
async def test_short_circuit_update_still_stops_on_every_guard(
    kwargs: dict[str, object], reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each real short-circuit still returns data so the caller stops there."""
    # The repeated-failure guard schedules a real HA timer; this test is about
    # the return contract, not the scheduler.
    monkeypatch.setattr(
        mammotion_coordinator, "async_call_later", lambda *_args, **_kw: None
    )

    result = await MammotionBaseUpdateCoordinator._async_short_circuit_update(  # noqa: SLF001
        _short_circuit_self(**kwargs)  # type: ignore[arg-type]
    )

    assert result is not None, f"{reason} must stop the update"


def test_every_coordinator_tests_short_circuit_with_is_not_none() -> None:
    """No coordinator may go back to testing the short-circuit for truthiness.

    Truthiness is exactly what broke this for months: a real payload that
    happens to be falsy would be read as "carry on", and an always-truthy one
    as "stop here". Pin the explicit ``is not None`` comparison at every call
    site so the regression cannot be reintroduced by a later edit.
    """
    source = pathlib.Path(mammotion_coordinator.__file__).read_text()
    tree = ast.parse(source)

    call_sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_async_short_circuit_update"
    ]
    assert len(call_sites) == 5, f"expected 5 call sites, found {len(call_sites)}"

    compares = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, ast.IsNot) for op in node.ops)
        and any(
            isinstance(inner, ast.Attribute)
            and inner.attr == "_async_short_circuit_update"
            for inner in ast.walk(node.left)
        )
        and any(
            isinstance(cmp, ast.Constant) and cmp.value is None
            for cmp in node.comparators
        )
    ]
    assert len(compares) == 5, (
        f"expected all 5 call sites to use `is not None`, found {len(compares)}"
    )


@pytest.mark.asyncio
async def test_sync_success_updates_last_sync_metadata() -> None:
    """Map/task sync success records timestamps and clears stale errors."""
    coordinator = SimpleNamespace(
        manager=SimpleNamespace(
            start_map_sync=AsyncMock(),
            start_plan_sync=AsyncMock(),
            get_device_by_name=MagicMock(return_value=None),
        ),
        device_name="Luba-Test",
        last_map_sync=None,
        last_map_sync_bol_hash=None,
        last_task_sync=None,
        last_map_task_error="old error",
        manual_motion_owner=None,
    )
    coordinator._raise_if_manual_motion_in_progress = (  # noqa: SLF001
        lambda action: (
            MammotionBaseUpdateCoordinator._raise_if_manual_motion_in_progress(  # noqa: SLF001
                coordinator, action
            )
        )
    )
    coordinator._reported_bol_hash = (  # noqa: SLF001
        lambda: MammotionBaseUpdateCoordinator._reported_bol_hash(coordinator)  # noqa: SLF001
    )
    coordinator._record_map_sync_attempt = (  # noqa: SLF001
        lambda bol_hash=None: MammotionBaseUpdateCoordinator._record_map_sync_attempt(  # noqa: SLF001
            coordinator, bol_hash
        )
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
        manual_motion_owner=None,
    )
    coordinator._raise_if_manual_motion_in_progress = (  # noqa: SLF001
        lambda action: (
            MammotionBaseUpdateCoordinator._raise_if_manual_motion_in_progress(  # noqa: SLF001
                coordinator, action
            )
        )
    )

    with pytest.raises(RuntimeError):
        await MammotionReportUpdateCoordinator.async_sync_maps(coordinator)

    assert coordinator.last_map_task_error == "map_sync: RuntimeError"


def _force_resync_self(**overrides: object) -> SimpleNamespace:
    """Build a minimal coordinator-shaped self for async_force_map_resync."""
    fake = SimpleNamespace(
        device_name="Test mower",
        manager=SimpleNamespace(regenerate_stale_geojson=MagicMock()),
        map_sync_status="out_of_sync",
        map_sync_diagnostics=lambda: {"status": "out_of_sync"},
        manual_motion_owner=None,
        last_map_task_error=None,
        async_rtk_dock_location=AsyncMock(),
        async_get_area_list=AsyncMock(),
        async_sync_maps=AsyncMock(),
    )
    for key, value in overrides.items():
        setattr(fake, key, value)
    return fake


def test_map_sync_diagnostics_explains_a_usable_but_out_of_sync_map() -> None:
    """A complete, containment-usable map can still report ``out_of_sync``.

    Live 2026-07-24: four areas with full polygon frames and containment
    passing, yet ``map_sync_status`` read ``out_of_sync`` — which also re-runs
    the sync saga on every coordinator tick.  ``is_map_synced()`` folds three
    conditions into one boolean; this breaks them back out so the failing one
    is identifiable from a read-only call.
    """
    device_map = SimpleNamespace(
        computed_bol_hash=5553341678865748256,
        find_incomplete_hashes=lambda _sub_cmd: [],
        area_name=[SimpleNamespace(hash=1343645155037768237, name="Backyard Right")],
        area_root_hashlist=[1343645155037768237],
        area={1343645155037768237: SimpleNamespace(data=[SimpleNamespace()])},
    )
    fake = SimpleNamespace(
        map_sync_status="out_of_sync",
        data=SimpleNamespace(
            map=device_map,
            report_data=SimpleNamespace(
                locations=[SimpleNamespace(bol_hash=8311072749804434520)]
            ),
        ),
    )

    diagnostics = MammotionBaseUpdateCoordinator.map_sync_diagnostics(fake)

    assert diagnostics["status"] == "out_of_sync"
    assert diagnostics["reported_bol_hash"] == "8311072749804434520"
    assert diagnostics["computed_bol_hash"] == "5553341678865748256"
    # The isolated culprit: the two other conditions are healthy.
    assert diagnostics["bol_hash_matches"] is False
    assert diagnostics["incomplete_area_hashes"] == []
    assert diagnostics["area_names_covered"] is True
    assert diagnostics["area_frame_counts"] == {"1343645155037768237": 1}
    assert "error" not in diagnostics


def test_map_sync_diagnostics_never_raises_on_a_broken_map() -> None:
    """Diagnostics degrade to an ``error`` field rather than breaking callers."""
    fake = SimpleNamespace(
        map_sync_status="out_of_sync",
        data=SimpleNamespace(
            map=SimpleNamespace(),  # missing every hash-list attribute
            report_data=SimpleNamespace(locations=[]),
        ),
    )

    diagnostics = MammotionBaseUpdateCoordinator.map_sync_diagnostics(fake)

    assert diagnostics["status"] == "out_of_sync"
    assert "error" in diagnostics


@pytest.mark.asyncio
async def test_force_map_resync_happy_path_runs_full_sequence() -> None:
    """Recovery runs RTK/dock -> area names -> saga -> regen and reports steps."""
    fake = _force_resync_self()

    async def _sync() -> None:
        fake.map_sync_status = "synced"

    fake.async_sync_maps = AsyncMock(side_effect=_sync)

    result = await MammotionBaseUpdateCoordinator.async_force_map_resync(fake)

    assert result["error"] is None
    assert result["steps"] == [
        "rtk_dock_refreshed",
        "area_names_fetched",
        "map_synced",
        "geojson_regenerated",
    ]
    assert result["map_sync_status_before"] == "out_of_sync"
    assert result["map_sync_status_after"] == "synced"
    fake.async_rtk_dock_location.assert_awaited_once()
    fake.async_sync_maps.assert_awaited_once()
    fake.manager.regenerate_stale_geojson.assert_called_once_with("Test mower")


@pytest.mark.asyncio
async def test_force_map_resync_tolerates_missing_area_names() -> None:
    """A transient area-name fetch failure is non-fatal; the saga still runs.

    Mirrors the cloud-session case where ``toapp_all_hash_name`` never arrives.
    """
    fake = _force_resync_self(
        async_get_area_list=AsyncMock(side_effect=NoTransportAvailableError())
    )

    result = await MammotionBaseUpdateCoordinator.async_force_map_resync(fake)

    assert result["error"] is None
    assert "area_names_skipped" in result["steps"]
    assert "area_names_fetched" not in result["steps"]
    assert result["steps"][-1] == "geojson_regenerated"
    fake.async_sync_maps.assert_awaited_once()
    fake.manager.regenerate_stale_geojson.assert_called_once()


@pytest.mark.asyncio
async def test_force_map_resync_reports_sync_failure_without_regen() -> None:
    """A saga failure surfaces as error and never regenerates against a dead sync."""
    fake = _force_resync_self(
        async_sync_maps=AsyncMock(side_effect=RuntimeError("boom")),
        last_map_task_error="map_sync: RuntimeError",
    )

    result = await MammotionBaseUpdateCoordinator.async_force_map_resync(fake)

    assert result["error"] is not None
    assert "RuntimeError" in result["error"]
    assert "geojson_regenerated" not in result["steps"]
    fake.manager.regenerate_stale_geojson.assert_not_called()
    assert result["last_map_task_error"] == "map_sync: RuntimeError"
