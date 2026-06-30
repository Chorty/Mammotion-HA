"""Tests for Mammotion read-only map/task visibility helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pymammotion.data.model.hash_list import Plan

from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.coordinator import MammotionReportUpdateCoordinator
from custom_components.mammotion.sensor import WORK_SENSOR_TYPES
from custom_components.mammotion.services import (
    EXPERIMENTAL_EXECUTE_SEGMENT_SCHEMA,
    MANUAL_VELOCITY_HEADING_CALIBRATION_TEST_SCHEMA,
    MANUAL_VELOCITY_MULTI_PULSE_TEST_SCHEMA,
    MANUAL_VELOCITY_PULSE_TEST_SCHEMA,
    MANUAL_VELOCITY_SEGMENT_TEST_SCHEMA,
    _custom_path_telemetry_snapshot,
    _dry_run_custom_path,
    _execute_custom_path,
    _export_mower_map,
    _export_mower_tasks,
    _manual_velocity_controller_decision,
    _manual_velocity_heading_calibration,
    _manual_velocity_path_progress_diagnostic,
    _manual_velocity_pulse_test,
    _manual_velocity_segment_test,
    _normalize_mower_areas,
    _normalize_mower_tasks,
    _preview_custom_path,
    _validate_custom_path,
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
    return SimpleNamespace(
        async_move_forward=AsyncMock(),
        async_move_back=AsyncMock(),
        async_move_left=AsyncMock(),
        async_move_right=AsyncMock(),
        async_stop_manual_motion=AsyncMock(),
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
    assert (
        MANUAL_VELOCITY_PULSE_TEST_SCHEMA(
            {
                "entity_id": "lawn_mower.test",
                "action": "forward",
                "speed": 0.4,
            }
        )["speed"]
        == 0.4
    )
    with pytest.raises(Exception):  # noqa: B017
        MANUAL_VELOCITY_PULSE_TEST_SCHEMA(
            {
                "entity_id": "lawn_mower.test",
                "action": "forward",
                "speed": 0.45,
            }
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


def test_manual_velocity_heading_calibration_schema_defaults() -> None:
    """Heading calibration schema defaults to a dry-run forward pulse."""
    parsed = MANUAL_VELOCITY_HEADING_CALIBRATION_TEST_SCHEMA(
        {"entity_id": "lawn_mower.test"}
    )

    assert parsed["action"] == "forward"
    assert parsed["dry_run"] is True
    assert parsed["speed"] == 0.4
    assert parsed["duration_ms"] == 750


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
