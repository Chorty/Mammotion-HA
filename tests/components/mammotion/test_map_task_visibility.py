"""Tests for Mammotion read-only map/task visibility helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pymammotion.data.model.hash_list import Plan

from custom_components.mammotion.coordinator import MammotionReportUpdateCoordinator
from custom_components.mammotion.sensor import WORK_SENSOR_TYPES
from custom_components.mammotion.services import (
    _export_mower_map,
    _export_mower_tasks,
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
