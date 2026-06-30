"""Mammotion services."""

from __future__ import annotations

import asyncio
import dataclasses
import math
from typing import TYPE_CHECKING, Any, cast

import voluptuous as vol
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity_registry as er
from pymammotion.data.model.hash_list import CommDataCouple, Plan
from pymammotion.data.model.pool_state import PoolPlan
from pymammotion.utility.constant.device_constant import (
    PosType,
    device_connection,
    device_mode,
)
from pymammotion.utility.device_type import DeviceType

from .const import DOMAIN, LOGGER
from .coordinator import MammotionReportUpdateCoordinator, MammotionSpinoCoordinator

if TYPE_CHECKING:
    from . import MammotionConfigEntry
from .geojson_utils import apply_geojson_offset
from .models import MammotionMowerData

SERVICE_GET_GEOJSON = "get_geojson"
SERVICE_GET_MOW_PATH_GEOJSON = "get_mow_path_geojson"
SERVICE_GET_MOW_PROGRESS_GEOJSON = "get_mow_progress_geojson"
SERVICE_GET_MAP_DATA = "get_map_data"
SERVICE_GET_TASKS = "get_tasks"
SERVICE_GET_AREAS = "get_areas"
SERVICE_EXPORT_MAP = "export_map"
SERVICE_EXPORT_TASKS = "export_tasks"
SERVICE_VALIDATE_CUSTOM_PATH = "validate_custom_path"
SERVICE_PREVIEW_CUSTOM_PATH = "preview_custom_path"
SERVICE_DRY_RUN_CUSTOM_PATH = "dry_run_custom_path"
SERVICE_EXECUTE_CUSTOM_PATH = "execute_custom_path"
SERVICE_MANUAL_VELOCITY_PULSE_TEST = "manual_velocity_pulse_test"
SERVICE_MANUAL_VELOCITY_SEGMENT_TEST = "manual_velocity_segment_test"
SERVICE_MANUAL_VELOCITY_MULTI_PULSE_TEST = "manual_velocity_multi_pulse_test"
SERVICE_MANUAL_VELOCITY_HEADING_CALIBRATION_TEST = (
    "manual_velocity_heading_calibration_test"
)
SERVICE_SVG_ADD = "svg_add"
SERVICE_SVG_UPDATE = "svg_update"
SERVICE_SVG_DELETE = "svg_delete"
SERVICE_REFRESH_STREAM = "refresh_stream"
SERVICE_START_VIDEO = "start_video"
SERVICE_STOP_VIDEO = "stop_video"
SERVICE_GET_TOKENS = "get_tokens"
SERVICE_MOVE_FORWARD = "move_forward"
SERVICE_MOVE_LEFT = "move_left"
SERVICE_MOVE_RIGHT = "move_right"
SERVICE_MOVE_BACKWARD = "move_backward"

# --- Task / schedule CRUD services ---------------------------------------
# Modify ops target a task button entity (entity_id).  Create / refresh
# target the device's lawn_mower or vacuum entity.  See
# ``docs/tasks_and_schedules.md`` in pymammotion for the wire protocol
# every one of these wraps.
SERVICE_CREATE_TASK = "create_task"
SERVICE_EDIT_TASK = "edit_task"
SERVICE_RENAME_TASK = "rename_task"
SERVICE_SET_TASK_ENABLED = "set_task_enabled"
SERVICE_DELETE_TASK = "delete_task"
SERVICE_COPY_TASK = "copy_task"
SERVICE_REFRESH_TASKS = "refresh_tasks"
# "start task" === "start schedule" — runs a stored mower schedule now.
# Backed by ``NavPlanTaskExecute(sub_cmd=1, id=plan_id)`` on the wire (see
# APK ``MACommandHelper.singleSchedule`` / docs/tasks_and_schedules.md § 1.6).
# Spino has no equivalent in the proto — the service rejects Spino targets
# with a translated error.
SERVICE_START_TASK = "start_task"

# Optional schedule fields shared by both device kinds.  The HA service
# layer normalises them into the per-kind Plan / PoolPlan dataclass.
_SCHEDULE_FIELDS = {
    vol.Optional("enabled", default=True): cv.boolean,
    vol.Optional("weeks"): vol.All(
        cv.ensure_list, [vol.All(vol.Coerce(int), vol.Range(min=0, max=6))]
    ),
    vol.Optional("start_time"): cv.string,  # "HH:MM"
    vol.Optional("end_time"): cv.string,
    vol.Optional("start_date"): cv.string,
    vol.Optional("end_date"): cv.string,
    vol.Optional("trigger_type"): vol.All(vol.Coerce(int), vol.Range(min=0, max=3)),
    vol.Optional("day"): vol.All(vol.Coerce(int), vol.Range(min=0)),
}

# Mower-only fields keyed by the names used on ``pymammotion.Plan``.
_MOWER_ONLY_FIELDS = {
    vol.Optional("knife_height"): vol.All(vol.Coerce(int), vol.Range(min=20, max=100)),
    vol.Optional("speed"): vol.Coerce(float),
    vol.Optional("edge_mode"): vol.All(vol.Coerce(int), vol.Range(min=0, max=2)),
    vol.Optional("route_angle"): vol.All(vol.Coerce(int), vol.Range(min=0, max=179)),
    vol.Optional("route_spacing"): vol.All(vol.Coerce(int), vol.Range(min=0)),
    vol.Optional("zone_hashs"): vol.All(cv.ensure_list, [vol.Coerce(int)]),
}

# Spino-only fields keyed by names on ``pymammotion.PoolPlan``.
_SPINO_ONLY_FIELDS = {
    vol.Optional("work_mode"): vol.All(vol.Coerce(int), vol.Range(min=0, max=6)),
    vol.Optional("sub_mode"): vol.All(cv.ensure_list, [vol.Coerce(int)]),
    vol.Optional("speed"): vol.All(vol.Coerce(int), vol.Range(min=0)),
    vol.Optional("operating_power"): vol.All(vol.Coerce(int), vol.Range(min=0)),
    vol.Optional("starttime"): vol.All(vol.Coerce(int), vol.Range(min=0)),
}


# Task services declare ``target:`` in services.yaml, so the HA UI delivers
# the selected task button(s) under ``entity_id`` as a list — one element per
# selected entity — even when only one is picked.  Plain ``cv.entity_id``
# rejected that list outright, which is why enable/disable (and every other
# task service) failed when invoked from the UI.
#
# Two flavours of validator handle this:
#   * ``cv.entity_ids`` — used by the bulk operations (enable/disable, delete)
#     where applying the same action to many tasks is meaningful.  Always
#     normalises to a list so the handler can iterate.
#   * ``_single_entity_id`` — used by operations that carry per-task identity
#     (edit/rename/copy/create) or target a single device (refresh/start).
#     Accepts the one-element target list and returns the lone entity_id,
#     rejecting ambiguous multi-entity input.
def _single_entity_id(value: Any) -> str:
    """Validate a single entity_id, tolerating the target list form."""
    ids = cv.entity_ids(value)
    if len(ids) != 1:
        raise vol.Invalid("expected exactly one entity_id")
    return ids[0]


CREATE_TASK_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): _single_entity_id,
        vol.Required("name"): cv.string,
        **_SCHEDULE_FIELDS,
        **_MOWER_ONLY_FIELDS,
        **_SPINO_ONLY_FIELDS,
    },
    extra=vol.ALLOW_EXTRA,
)

EDIT_TASK_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): _single_entity_id,
        vol.Optional("name"): cv.string,
        **_SCHEDULE_FIELDS,
        **_MOWER_ONLY_FIELDS,
        **_SPINO_ONLY_FIELDS,
    },
    extra=vol.ALLOW_EXTRA,
)

RENAME_TASK_SCHEMA = vol.Schema(
    {vol.Required(ATTR_ENTITY_ID): _single_entity_id, vol.Required("name"): cv.string},
    extra=vol.ALLOW_EXTRA,
)

SET_TASK_ENABLED_SCHEMA = vol.Schema(
    {vol.Required(ATTR_ENTITY_ID): cv.entity_ids, vol.Required("enabled"): cv.boolean},
    extra=vol.ALLOW_EXTRA,
)

DELETE_TASK_SCHEMA = vol.Schema(
    {vol.Required(ATTR_ENTITY_ID): cv.entity_ids}, extra=vol.ALLOW_EXTRA
)

COPY_TASK_SCHEMA = vol.Schema(
    {vol.Required(ATTR_ENTITY_ID): _single_entity_id, vol.Optional("name"): cv.string},
    extra=vol.ALLOW_EXTRA,
)

REFRESH_TASKS_SCHEMA = vol.Schema(
    {vol.Required(ATTR_ENTITY_ID): _single_entity_id}, extra=vol.ALLOW_EXTRA
)

START_TASK_SCHEMA = vol.Schema(
    {vol.Required(ATTR_ENTITY_ID): _single_entity_id}, extra=vol.ALLOW_EXTRA
)

GEOJSON_SCHEMA = vol.Schema(
    {vol.Required(ATTR_ENTITY_ID): cv.entity_id}, extra=vol.ALLOW_EXTRA
)

_CUSTOM_PATH_POINT_SCHEMA = vol.Schema(
    {
        vol.Required("x"): vol.Coerce(float),
        vol.Required("y"): vol.Coerce(float),
    },
    extra=vol.ALLOW_EXTRA,
)

VALIDATE_CUSTOM_PATH_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("points"): vol.All(
            cv.ensure_list, [_CUSTOM_PATH_POINT_SCHEMA]
        ),
        vol.Optional("area_hash"): vol.Coerce(int),
        vol.Optional("speed", default=0.2): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.6)
        ),
        vol.Optional("blade_mode", default="off"): vol.In(["off"]),
    },
    extra=vol.ALLOW_EXTRA,
)

DRY_RUN_CUSTOM_PATH_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("points"): vol.All(
            cv.ensure_list, [_CUSTOM_PATH_POINT_SCHEMA]
        ),
        vol.Optional("area_hash"): vol.Coerce(int),
        vol.Optional("speed", default=0.2): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.6)
        ),
        vol.Optional("blade_mode", default="off"): vol.In(["off"]),
        vol.Optional("heading_offset_degrees", default=0.0): vol.All(
            vol.Coerce(float), vol.Range(min=-180.0, max=180.0)
        ),
        vol.Optional("dry_run", default=True): vol.All(cv.boolean, vol.Equal(True)),
    },
    extra=vol.ALLOW_EXTRA,
)

EXECUTE_CUSTOM_PATH_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("points"): vol.All(
            cv.ensure_list, [_CUSTOM_PATH_POINT_SCHEMA]
        ),
        vol.Optional("area_hash"): vol.Coerce(int),
        vol.Optional("speed", default=0.2): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.3)
        ),
        vol.Optional("blade_mode", default="off"): vol.In(["off"]),
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("allow_manual_velocity", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

MANUAL_VELOCITY_PULSE_TEST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("action", default="forward"): vol.In(
            ["forward", "backward", "turn_left", "turn_right"]
        ),
        vol.Optional("speed", default=0.1): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.4)
        ),
        vol.Optional("duration_ms", default=250): vol.All(
            vol.Coerce(int), vol.Range(min=50, max=750)
        ),
        vol.Optional("use_wifi", default=False): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

MANUAL_VELOCITY_SEGMENT_TEST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("points"): vol.All(
            cv.ensure_list, [_CUSTOM_PATH_POINT_SCHEMA]
        ),
        vol.Optional("area_hash"): vol.Coerce(int),
        vol.Optional("speed", default=0.4): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.4)
        ),
        vol.Optional("pulse_duration_ms", default=750): vol.All(
            vol.Coerce(int), vol.Range(min=50, max=750)
        ),
        vol.Optional("max_pulses", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=5)
        ),
        vol.Optional("waypoint_tolerance", default=0.1): vol.All(
            vol.Coerce(float), vol.Range(min=0.02, max=0.5)
        ),
        vol.Optional("force_action", default="auto"): vol.In(
            ["auto", "forward", "backward", "turn_left", "turn_right"]
        ),
        vol.Optional("heading_offset_degrees", default=0.0): vol.All(
            vol.Coerce(float), vol.Range(min=-180.0, max=180.0)
        ),
        vol.Optional("min_progress_distance", default=0.003): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)
        ),
        vol.Optional("no_progress_limit", default=2): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=5)
        ),
        vol.Optional("min_heading_change_degrees", default=1.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=45.0)
        ),
        vol.Optional("use_wifi", default=True): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

MANUAL_VELOCITY_MULTI_PULSE_TEST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("points"): vol.All(
            cv.ensure_list, [_CUSTOM_PATH_POINT_SCHEMA]
        ),
        vol.Optional("area_hash"): vol.Coerce(int),
        vol.Optional("speed", default=0.4): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.4)
        ),
        vol.Optional("pulse_duration_ms", default=750): vol.All(
            vol.Coerce(int), vol.Range(min=50, max=750)
        ),
        vol.Optional("max_pulses", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=2, max=5)
        ),
        vol.Optional("waypoint_tolerance", default=0.1): vol.All(
            vol.Coerce(float), vol.Range(min=0.02, max=0.5)
        ),
        vol.Optional("force_action", default="auto"): vol.In(
            ["auto", "forward", "backward", "turn_left", "turn_right"]
        ),
        vol.Optional("heading_offset_degrees", default=0.0): vol.All(
            vol.Coerce(float), vol.Range(min=-180.0, max=180.0)
        ),
        vol.Optional("min_progress_distance", default=0.003): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)
        ),
        vol.Optional("no_progress_limit", default=2): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=5)
        ),
        vol.Optional("min_heading_change_degrees", default=1.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=45.0)
        ),
        vol.Optional("use_wifi", default=True): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

MANUAL_VELOCITY_HEADING_CALIBRATION_TEST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("action", default="forward"): vol.In(
            ["forward", "turn_left", "turn_right"]
        ),
        vol.Optional("speed", default=0.4): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.4)
        ),
        vol.Optional("duration_ms", default=750): vol.All(
            vol.Coerce(int), vol.Range(min=50, max=750)
        ),
        vol.Optional("use_wifi", default=True): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
        vol.Optional("min_progress_distance", default=0.003): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)
        ),
        vol.Optional("min_heading_change_degrees", default=1.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=45.0)
        ),
    },
    extra=vol.ALLOW_EXTRA,
)

CAMERA_SCHEMA = vol.Schema(
    {vol.Required(ATTR_ENTITY_ID): cv.entity_id}, extra=vol.ALLOW_EXTRA
)

MOVEMENT_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("speed", default=0.4): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=1.0)
        ),
        vol.Optional("use_wifi", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

_SVG_COMMON_FIELDS = {
    vol.Optional("svg_file_name", default="pattern.svg"): str,
    vol.Optional("scale", default=1.0): vol.Coerce(float),
    vol.Optional("rotate", default=0.0): vol.Coerce(float),
    vol.Optional("base_width_m", default=2.5): vol.Coerce(float),
    vol.Optional("base_height_m", default=2.5): vol.Coerce(float),
    vol.Optional("x_move"): vol.Coerce(float),
    vol.Optional("y_move"): vol.Coerce(float),
}

SVG_ADD_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("area_hash"): vol.Coerce(int),
        vol.Required("svg_data"): str,
        **_SVG_COMMON_FIELDS,
    },
    extra=vol.ALLOW_EXTRA,
)

SVG_UPDATE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("device_hash"): vol.Coerce(int),
        vol.Required("area_hash"): vol.Coerce(int),
        vol.Required("svg_data"): str,
        **_SVG_COMMON_FIELDS,
    },
    extra=vol.ALLOW_EXTRA,
)

SVG_DELETE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("device_hash"): vol.Coerce(int),
        vol.Required("area_hash"): vol.Coerce(int),
    },
    extra=vol.ALLOW_EXTRA,
)


_JS_MAX_SAFE_INT = (1 << 53) - 1


def _stringify_large_ints(obj: Any) -> Any:
    """Recursively convert integers beyond JS Number.MAX_SAFE_INTEGER to strings.

    JavaScript's JSON.parse silently loses precision on integers > 2**53-1.
    Converting them to strings before sending over the WebSocket preserves the
    full hash value; Python's vol.Coerce(int) can convert them back on ingress.
    """
    if isinstance(obj, dict):
        return {k: _stringify_large_ints(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_large_ints(v) for v in obj]
    if (
        isinstance(obj, int)
        and not isinstance(obj, bool)
        and abs(obj) > _JS_MAX_SAFE_INT
    ):
        return str(obj)
    return obj


def _json_safe_int(value: int) -> int | str:
    """Return *value* as a JSON-safe int, stringifying values JS cannot preserve."""
    return str(value) if abs(value) > _JS_MAX_SAFE_INT else value


def _safe_asdict(obj: Any) -> Any:
    """Return a JSON-ish representation for dataclass or plain test objects."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {key: _safe_asdict(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_safe_asdict(value) for value in obj]
    if hasattr(obj, "__dict__"):
        return {key: _safe_asdict(value) for key, value in vars(obj).items()}
    return obj


def _plan_area_names(
    coordinator: MammotionReportUpdateCoordinator, zone_hashs: list[int]
) -> list[str | None]:
    """Resolve mower plan zone hashes to area names."""
    return [coordinator.get_area_entity_name(zone_hash) for zone_hash in zone_hashs]


def _normalize_mower_tasks(
    coordinator: MammotionReportUpdateCoordinator,
) -> list[dict[str, Any]]:
    """Return normalized read-only task data for a mower coordinator."""
    from pymammotion.data.model.device import MowingDevice  # noqa: PLC0415

    device_data = cast(MowingDevice, coordinator.data)
    tasks: list[dict[str, Any]] = []
    for plan_id, plan in sorted(device_data.map.plan.items()):
        zone_hashs = list(getattr(plan, "zone_hashs", []) or [])
        tasks.append(
            {
                "plan_id": plan.plan_id or str(plan_id),
                "name": plan.task_name,
                "enabled": plan.is_enabled(),
                "weeks": list(getattr(plan, "weeks", []) or []),
                "start_time": plan.start_time,
                "end_time": plan.end_time,
                "start_date": plan.start_date,
                "end_date": plan.end_date,
                "knife_height": plan.knife_height,
                "speed": plan.speed,
                "edge_mode": plan.edge_mode,
                "route_angle": plan.route_angle,
                "route_spacing": plan.route_spacing,
                "zone_hashs": [_json_safe_int(zone_hash) for zone_hash in zone_hashs],
                "zone_names": _plan_area_names(coordinator, zone_hashs),
                "raw": _stringify_large_ints(dataclasses.asdict(plan)),
            }
        )
    return tasks


def _normalize_mower_areas(
    coordinator: MammotionReportUpdateCoordinator,
) -> list[dict[str, Any]]:
    """Return normalized read-only area data for a mower coordinator."""
    from pymammotion.data.model.device import MowingDevice  # noqa: PLC0415

    device_data = cast(MowingDevice, coordinator.data)
    area_names: dict[int, str] = {
        area_name.hash: area_name.name for area_name in device_data.map.area_name
    }
    known_hashes = set(device_data.map.area.keys()) | set(area_names.keys())

    task_refs_by_area: dict[int, list[dict[str, str]]] = {}
    for plan_id, plan in device_data.map.plan.items():
        for zone_hash in getattr(plan, "zone_hashs", []) or []:
            task_refs_by_area.setdefault(zone_hash, []).append(
                {"plan_id": plan.plan_id or str(plan_id), "name": plan.task_name}
            )

    areas: list[dict[str, Any]] = []
    for area_hash in sorted(known_hashes):
        frame_list = device_data.map.area.get(area_hash)
        frame_count = len(getattr(frame_list, "data", []) or []) if frame_list else 0
        areas.append(
            {
                "area_hash": _json_safe_int(area_hash),
                "name": area_names.get(area_hash)
                or coordinator.get_area_entity_name(area_hash),
                "has_geometry": frame_count > 0,
                "frame_count": frame_count,
                "referenced_by_tasks": task_refs_by_area.get(area_hash, []),
            }
        )
    return areas


def _area_polygons(
    coordinator: MammotionReportUpdateCoordinator, area_hash: int | None = None
) -> dict[int, list[dict[str, float]]]:
    """Return known map area polygons as map-local x/y point lists."""
    from pymammotion.data.model.device import MowingDevice  # noqa: PLC0415

    device_data = cast(MowingDevice, coordinator.data)
    polygons: dict[int, list[dict[str, float]]] = {}
    for current_hash, frame_list in device_data.map.area.items():
        if area_hash is not None and current_hash != area_hash:
            continue
        points: list[dict[str, float]] = []
        for frame in sorted(
            getattr(frame_list, "data", []) or [],
            key=lambda f: getattr(f, "current_frame", 0),
        ):
            points.extend(
                {"x": float(point.x), "y": float(point.y)}
                for point in getattr(frame, "data_couple", []) or []
                if hasattr(point, "x") and hasattr(point, "y")
            )
        polygons[current_hash] = points
    return polygons


def _point_on_segment(
    point: dict[str, float],
    start: dict[str, float],
    end: dict[str, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return True if *point* lies on the line segment from *start* to *end*."""
    cross = (point["y"] - start["y"]) * (end["x"] - start["x"]) - (
        point["x"] - start["x"]
    ) * (end["y"] - start["y"])
    if abs(cross) > tolerance:
        return False
    dot = (point["x"] - start["x"]) * (end["x"] - start["x"]) + (
        point["y"] - start["y"]
    ) * (end["y"] - start["y"])
    if dot < -tolerance:
        return False
    squared_len = (end["x"] - start["x"]) ** 2 + (end["y"] - start["y"]) ** 2
    return dot <= squared_len + tolerance


def _point_in_polygon(
    point: dict[str, float], polygon: list[dict[str, float]]
) -> bool:
    """Return True when a map-local point is inside or on a polygon boundary."""
    if len(polygon) < 3:
        return False
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        crosses = (current["y"] > point["y"]) != (previous["y"] > point["y"])
        if crosses:
            x_at_y = (previous["x"] - current["x"]) * (
                point["y"] - current["y"]
            ) / (previous["y"] - current["y"]) + current["x"]
            if point["x"] <= x_at_y:
                inside = not inside
        previous = current
    return inside


def _path_distance(points: list[dict[str, float]]) -> float:
    """Return total map-local path distance in mower map units."""
    return sum(
        math.hypot(end["x"] - start["x"], end["y"] - start["y"])
        for start, end in zip(points, points[1:], strict=False)
    )


def _path_heading_degrees(
    start: dict[str, float], end: dict[str, float]
) -> float:
    """Return a map-local heading in degrees for a segment."""
    return (
        math.degrees(math.atan2(end["y"] - start["y"], end["x"] - start["x"])) + 360
    ) % 360


def _heading_error_degrees(current: float, target: float) -> float:
    """Return signed shortest heading error in degrees."""
    return (target - current + 540) % 360 - 180


def _isoformat_or_none(value: Any) -> str | None:
    """Return datetime-like values as ISO strings for HA service responses."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _export_mower_map(coordinator: MammotionReportUpdateCoordinator) -> dict[str, Any]:
    """Return read-only map export data for route planning/debugging."""
    from pymammotion.data.model.device import MowingDevice  # noqa: PLC0415

    device_data = cast(MowingDevice, coordinator.data)
    map_dict = _safe_asdict(device_data.map)
    polygons = _area_polygons(coordinator)
    return cast(
        dict[str, Any],
        _stringify_large_ints(
            {
                "coordinate_system": "mower_map_xy",
                "areas": _normalize_mower_areas(coordinator),
                "area_polygons": {
                    str(_json_safe_int(area_hash)): points
                    for area_hash, points in polygons.items()
                },
                "raw": {
                    "area": map_dict.get("area", {}),
                    "svg": map_dict.get("svg", {}),
                    "area_name": map_dict.get("area_name", []),
                },
            }
        ),
    )


def _export_mower_tasks(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Return read-only task export data for route planning/debugging."""
    tasks = _normalize_mower_tasks(coordinator)
    return {
        "tasks": tasks,
        "task_count": len(tasks),
        "enabled_task_count": sum(1 for task in tasks if task["enabled"]),
        "last_task_sync": _isoformat_or_none(coordinator.last_task_sync),
        "last_map_task_error": coordinator.last_map_task_error,
    }


def _validate_custom_path(  # noqa: C901
    coordinator: MammotionReportUpdateCoordinator,
    points: list[dict[str, float]],
    *,
    area_hash: int | None = None,
    speed: float = 0.2,
    blade_mode: str = "off",
) -> dict[str, Any]:
    """Validate a proposed custom path without sending movement commands."""
    errors: list[str] = []
    warnings: list[str] = []

    normalized_points = [
        {"x": float(point["x"]), "y": float(point["y"])} for point in points
    ]

    if len(normalized_points) < 2:
        errors.append("path_requires_at_least_two_points")
    if len(normalized_points) > 500:
        errors.append("path_has_too_many_points")
    if blade_mode != "off":
        errors.append("blade_mode_must_be_off")
    if speed > 0.4:
        warnings.append("speed_above_recommended_validation_default")

    polygons = _area_polygons(coordinator, area_hash)
    if area_hash is not None and not polygons:
        errors.append("area_hash_not_found")
    valid_polygons = {
        current_hash: polygon
        for current_hash, polygon in polygons.items()
        if len(polygon) >= 3
    }
    if not valid_polygons:
        warnings.append("no_area_geometry_available_for_containment_check")
    else:
        outside: list[int] = []
        for index, point in enumerate(normalized_points):
            if not any(
                _point_in_polygon(point, polygon)
                for polygon in valid_polygons.values()
            ):
                outside.append(index)
        if outside:
            errors.append("path_points_outside_known_area_geometry")

    distance = _path_distance(normalized_points)
    if distance == 0 and len(normalized_points) >= 2:
        errors.append("path_distance_must_be_greater_than_zero")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "coordinate_system": "mower_map_xy",
        "blade_mode": blade_mode,
        "speed": speed,
        "area_hash": _json_safe_int(area_hash) if area_hash is not None else None,
        "point_count": len(normalized_points),
        "distance": distance,
        "points": normalized_points,
    }


def _custom_path_preview_geojson(
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Build GeoJSON preview data for a validated custom path response."""
    points = validation["points"]
    features: list[dict[str, Any]] = []
    if points:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "type_name": "custom_path_start",
                    "Name": "Start",
                    "marker": "start",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [points[0]["x"], points[0]["y"]],
                },
            }
        )
    if len(points) >= 2:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "type_name": "custom_path",
                    "Name": "Custom path",
                    "valid": validation["valid"],
                    "distance": validation["distance"],
                    "color": "#22c55e" if validation["valid"] else "#ef4444",
                    "weight": 3,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[point["x"], point["y"]] for point in points],
                },
            }
        )
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "type_name": "custom_path_end",
                    "Name": "End",
                    "marker": "end",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [points[-1]["x"], points[-1]["y"]],
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _preview_custom_path(
    coordinator: MammotionReportUpdateCoordinator,
    points: list[dict[str, float]],
    *,
    area_hash: int | None = None,
    speed: float = 0.2,
    blade_mode: str = "off",
) -> dict[str, Any]:
    """Return a read-only custom path validation plus display preview."""
    validation = _validate_custom_path(
        coordinator,
        points,
        area_hash=area_hash,
        speed=speed,
        blade_mode=blade_mode,
    )
    return {
        **validation,
        "geojson": _custom_path_preview_geojson(validation),
        "path": {
            "coordinate_system": validation["coordinate_system"],
            "points": validation["points"],
            "distance": validation["distance"],
        },
    }


def _safe_attr_path(obj: Any, path: str) -> Any:
    """Return a nested attribute value or None when any hop is missing."""
    current = obj
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _first_not_none(*values: Any) -> Any:
    """Return the first value that is not None, preserving falsey telemetry values."""
    for value in values:
        if value is not None:
            return value
    return None


def _enum_value(value: Any) -> Any:
    """Return the primitive value for enum-like values."""
    return getattr(value, "value", value)


def _enum_label(value: Any) -> str | None:
    """Return a readable label for enum-like values."""
    if value is None:
        return None
    return getattr(value, "name", str(value))


def _scale_report_position(value: Any) -> float | None:
    """Scale raw report map-local integer position fields to mower-map units."""
    if value is None:
        return None
    try:
        return float(value) / 10_000
    except (TypeError, ValueError):
        return None


def _position_mode_label(pos_level: Any) -> str | None:
    """Return a readable position quality label from a pos_level value."""
    if pos_level is None:
        return None
    try:
        from pymammotion.data.model.enums import PositionMode  # noqa: PLC0415

        return PositionMode.from_value(int(pos_level)).name
    except (TypeError, ValueError):
        return "UNKNOWN"


def _rtk_status_label(value: Any) -> str | None:
    """Return a readable RTK status label from enum or numeric values."""
    if value is None:
        return None
    if hasattr(value, "name"):
        return {
            "FINE": "Fix",
            "BAD": "Single",
            "NONE": "None",
        }.get(str(value.name), str(value.name).title())
    try:
        from pymammotion.data.model.enums import RTKStatus  # noqa: PLC0415

        return str(RTKStatus.from_value(int(value)))
    except (TypeError, ValueError):
        return "Unknown"


def _pos_type_label(value: Any) -> str | None:
    """Return a readable position type label."""
    if value is None:
        return None
    try:
        return PosType(int(value)).name
    except ValueError:
        return "UNKNOWN"


def _charge_state_label(value: Any) -> str:
    """Return a readable charge-state label."""
    try:
        charge_state = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return {
        0: "not_charging",
        1: "charging",
        2: "docked_or_charging",
    }.get(charge_state, "unknown")


def _blade_state_label(value: Any) -> str | None:
    """Return a readable blade state label."""
    if value is None:
        return None
    return _enum_label(value)


def _is_zero_pose(x: Any, y: Any) -> bool:
    """Return true when x/y are both exactly zero-like values."""
    try:
        return float(x) == 0.0 and float(y) == 0.0
    except (TypeError, ValueError):
        return False


def _is_area_out(pos_type: Any, zone_hash: Any) -> bool:
    """Return true when position metadata indicates outside/no mapped area."""
    try:
        pos_type_int = int(pos_type)
    except (TypeError, ValueError):
        pos_type_int = None
    try:
        zone_hash_int = int(zone_hash)
    except (TypeError, ValueError):
        zone_hash_int = None
    return pos_type_int == 0 and zone_hash_int == 0


def _is_stale_zero_area_out_pose(
    x: Any, y: Any, pos_type: Any, zone_hash: Any
) -> bool:
    """Return true for the common stale dock/default pose."""
    return _is_zero_pose(x, y) and _is_area_out(pos_type, zone_hash)


def _is_valid_motion_position(position: dict[str, Any]) -> bool:
    """Return true when position is usable for real manual motion probing."""
    return (
        position.get("source") != "unavailable"
        and position.get("x") is not None
        and position.get("y") is not None
        and not _is_zero_pose(position.get("x"), position.get("y"))
        and position.get("pos_type_label") == "AREA_INSIDE"
        and position.get("zone_hash") not in (None, 0, "0")
    )


def _latest_location(data: Any) -> Any:
    """Return the first reported location entry, if present."""
    locations = _safe_attr_path(data, "report_data.locations")
    if not locations:
        return None
    return locations[0]


def _custom_path_position_snapshot(
    data: Any, coordinator: MammotionReportUpdateCoordinator | None = None
) -> dict[str, Any]:
    """Return normalized map-local position diagnostics for custom-path dry runs."""
    mowing_state = _safe_attr_path(data, "mowing_state")
    report_location = _latest_location(data)
    rtk = _safe_attr_path(data, "report_data.rtk")
    location_pos_type = _safe_attr_path(data, "location.position_type")
    location_zone_hash = _safe_attr_path(data, "location.work_zone")
    location_toward = _safe_attr_path(data, "location.orientation")

    source = "unavailable"
    x = y = toward = None
    pos_level = rtk_status = pos_type = zone_hash = None

    if report_location is not None and (
        _safe_attr_path(report_location, "real_pos_x") is not None
        or _safe_attr_path(report_location, "real_pos_y") is not None
    ):
        candidate_x = _scale_report_position(
            _safe_attr_path(report_location, "real_pos_x")
        )
        candidate_y = _scale_report_position(
            _safe_attr_path(report_location, "real_pos_y")
        )
        candidate_pos_type = _safe_attr_path(report_location, "pos_type")
        candidate_zone_hash = _safe_attr_path(report_location, "bol_hash")
        if not _is_stale_zero_area_out_pose(
            candidate_x, candidate_y, candidate_pos_type, candidate_zone_hash
        ):
            source = "report_data.locations[0]"
            x = candidate_x
            y = candidate_y
            toward = _scale_report_position(
                _safe_attr_path(report_location, "real_toward")
            )
            pos_type = candidate_pos_type
            zone_hash = candidate_zone_hash

    if source == "unavailable" and mowing_state is not None and (
        _safe_attr_path(mowing_state, "pos_x") is not None
        or _safe_attr_path(mowing_state, "pos_y") is not None
    ):
        candidate_x = _safe_attr_path(mowing_state, "pos_x")
        candidate_y = _safe_attr_path(mowing_state, "pos_y")
        candidate_pos_type = _safe_attr_path(mowing_state, "pos_type")
        candidate_zone_hash = _safe_attr_path(mowing_state, "zone_hash")
        if not _is_stale_zero_area_out_pose(
            candidate_x, candidate_y, candidate_pos_type, candidate_zone_hash
        ):
            source = "mowing_state"
            x = candidate_x
            y = candidate_y
            toward = _safe_attr_path(mowing_state, "toward")
            pos_level = _safe_attr_path(mowing_state, "pos_level")
            rtk_status = _safe_attr_path(mowing_state, "rtk_status")
            pos_type = candidate_pos_type
            zone_hash = candidate_zone_hash

    if source == "unavailable" and (
        location_pos_type is not None or location_zone_hash is not None
    ):
        source = "location_metadata"
        pos_type = location_pos_type
        zone_hash = location_zone_hash
        toward = location_toward
    if source == "unavailable" and mowing_state is not None and (
        _safe_attr_path(mowing_state, "pos_x") is not None
        or _safe_attr_path(mowing_state, "pos_y") is not None
    ):
        # Keep raw zero-pose diagnostics visible on dry-runs, but overlay
        # known-good location metadata below so real-pulse gates can reject it
        # precisely instead of treating AREA_OUT/zone_hash=0 as authoritative.
        source = "mowing_state"
        x = _safe_attr_path(mowing_state, "pos_x")
        y = _safe_attr_path(mowing_state, "pos_y")
        toward = _safe_attr_path(mowing_state, "toward")
        pos_level = _safe_attr_path(mowing_state, "pos_level")
        rtk_status = _safe_attr_path(mowing_state, "rtk_status")
        pos_type = _safe_attr_path(mowing_state, "pos_type")
        zone_hash = _safe_attr_path(mowing_state, "zone_hash")

    pos_level = _first_not_none(pos_level, _safe_attr_path(rtk, "pos_level"))
    rtk_status = _first_not_none(rtk_status, _safe_attr_path(rtk, "status"))
    if _is_area_out(pos_type, zone_hash) and (
        location_pos_type is not None or location_zone_hash is not None
    ):
        pos_type = location_pos_type
        zone_hash = location_zone_hash
    else:
        pos_type = _first_not_none(pos_type, location_pos_type)
        zone_hash = _first_not_none(zone_hash, location_zone_hash)
    toward = _first_not_none(toward, location_toward)
    safe_zone_hash = _json_safe_int(zone_hash) if zone_hash is not None else None
    area_name = (
        coordinator.get_area_entity_name(int(zone_hash))
        if coordinator is not None
        and hasattr(coordinator, "get_area_entity_name")
        and zone_hash not in (None, 0, "0")
        else None
    )

    return {
        "x": x,
        "y": y,
        "toward": toward,
        "source": source,
        "pos_level": pos_level,
        "pos_level_label": _position_mode_label(pos_level),
        "rtk_status": _enum_value(rtk_status),
        "rtk_status_label": _rtk_status_label(rtk_status),
        "pos_type": pos_type,
        "pos_type_label": _pos_type_label(pos_type),
        "zone_hash": safe_zone_hash,
        "area_name": area_name,
        "valid_for_motion": _is_valid_motion_position(
            {
                "source": source,
                "x": x,
                "y": y,
                "pos_type_label": _pos_type_label(pos_type),
                "zone_hash": safe_zone_hash,
            }
        ),
    }


def _custom_path_telemetry_snapshot(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Return local cached telemetry useful for a custom-path dry run."""
    data = coordinator.data
    work_mode = _safe_attr_path(data, "report_data.dev.sys_status")
    charge_state = _safe_attr_path(data, "report_data.dev.charge_state")
    blade_state = _safe_attr_path(data, "report_data.dev.blade_state")
    connect = _safe_attr_path(data, "report_data.connect")
    return {
        "online": coordinator.is_online() if hasattr(coordinator, "is_online") else None,
        "work_mode": work_mode,
        "work_mode_label": device_mode(work_mode) if work_mode is not None else None,
        "charge_state": charge_state,
        "charge_state_label": _charge_state_label(charge_state),
        "position": _custom_path_position_snapshot(data, coordinator),
        "blade": {
            "reported_state": _enum_value(blade_state),
            "reported_state_label": _blade_state_label(blade_state),
            "knife_status": _safe_attr_path(data, "report_data.knife_status.knife_status"),
            "current_cutter_mode": _safe_attr_path(
                data, "report_data.cutter_work_mode_info.current_cutter_mode"
            ),
            "current_cutter_rpm": _safe_attr_path(
                data, "report_data.cutter_work_mode_info.current_cutter_rpm"
            ),
        },
        "transport": {
            "ble_rssi": _safe_attr_path(data, "report_data.connect.ble_rssi"),
            "wifi_rssi": _safe_attr_path(data, "report_data.connect.wifi_rssi"),
            "wifi_connect_status": _safe_attr_path(
                data, "report_data.connect.wifi_connect_status"
            ),
            "iot_connect_status": _safe_attr_path(
                data, "report_data.connect.iot_connect_status"
            ),
            "connection_label": device_connection(connect) if connect is not None else None,
        },
    }


def _manual_velocity_controller_decision(
    path_points: list[dict[str, float]],
    telemetry: dict[str, Any],
    *,
    speed: float,
    waypoint_tolerance: float = 0.4,
    heading_tolerance_degrees: float = 15.0,
    heading_offset_degrees: float = 0.0,
    max_pulse_seconds: float = 0.5,
) -> dict[str, Any]:
    """Return the next simulated manual-velocity action without sending it."""
    position = telemetry.get("position", {})
    current_x = position.get("x")
    current_y = position.get("y")
    current_heading = position.get("toward")

    base_response: dict[str, Any] = {
        "mode": "simulated",
        "would_send": False,
        "coordinate_system": "mower_map_xy",
        "waypoint_tolerance": waypoint_tolerance,
        "heading_tolerance_degrees": heading_tolerance_degrees,
        "heading_offset_degrees": heading_offset_degrees,
        "max_pulse_seconds": max_pulse_seconds,
        "speed": speed,
        "use_wifi": False,
    }

    if not path_points:
        return {
            **base_response,
            "action": "stop",
            "reason": "path_has_no_points",
            "command_not_sent": None,
        }
    if current_x is None or current_y is None:
        return {
            **base_response,
            "action": "stop",
            "reason": "live_position_unavailable",
            "command_not_sent": None,
        }
    if current_heading is None:
        return {
            **base_response,
            "action": "stop",
            "reason": "live_heading_unavailable",
            "command_not_sent": None,
        }

    current = {"x": float(current_x), "y": float(current_y)}
    target_index = None
    target = None
    distance_to_target = None
    for index, point in enumerate(path_points):
        distance = _path_distance([current, point])
        if distance > waypoint_tolerance:
            target_index = index
            target = point
            distance_to_target = distance
            break

    if target is None or target_index is None or distance_to_target is None:
        return {
            **base_response,
            "action": "stop",
            "reason": "path_complete",
            "target_index": None,
            "distance_to_target": 0.0,
            "command_not_sent": None,
        }

    reported_heading = float(current_heading)
    corrected_heading = (reported_heading + heading_offset_degrees) % 360
    target_heading = _path_heading_degrees(current, target)
    heading_error = _heading_error_degrees(corrected_heading, target_heading)
    if abs(heading_error) > heading_tolerance_degrees:
        action = "turn_left" if heading_error > 0 else "turn_right"
        service = SERVICE_MOVE_LEFT if heading_error > 0 else SERVICE_MOVE_RIGHT
        command = {
            "service": f"{DOMAIN}.{service}",
            "data": {"speed": speed, "use_wifi": False},
        }
        reason = "heading_error_exceeds_tolerance"
    else:
        action = "forward"
        command = {
            "service": f"{DOMAIN}.{SERVICE_MOVE_FORWARD}",
            "data": {"speed": speed, "use_wifi": False},
        }
        reason = "heading_aligned"

    return {
        **base_response,
        "action": action,
        "reason": reason,
        "current": current,
        "current_heading_degrees": reported_heading,
        "corrected_heading_degrees": corrected_heading,
        "target_index": target_index,
        "target": target,
        "target_heading_degrees": target_heading,
        "heading_error_degrees": heading_error,
        "distance_to_target": distance_to_target,
        "command_not_sent": command,
    }


def _manual_velocity_action_method(action: str) -> str:
    """Return the coordinator method name for a manual velocity action."""
    return {
        "forward": "async_move_forward",
        "backward": "async_move_back",
        "turn_left": "async_move_left",
        "turn_right": "async_move_right",
    }[action]


def _manual_velocity_action_service(action: str) -> str:
    """Return the HA service name for a manual velocity action."""
    return {
        "forward": SERVICE_MOVE_FORWARD,
        "backward": SERVICE_MOVE_BACKWARD,
        "turn_left": SERVICE_MOVE_LEFT,
        "turn_right": SERVICE_MOVE_RIGHT,
    }[action]


def _position_available(telemetry: dict[str, Any]) -> bool:
    """Return true when telemetry contains a map-local mower position."""
    position = telemetry.get("position", {})
    return (
        position.get("source") != "unavailable"
        and position.get("x") is not None
        and position.get("y") is not None
    )


def _position_has_known_area(telemetry: dict[str, Any]) -> bool:
    """Return true when telemetry ties the mower to a known mowing area."""
    position = telemetry.get("position", {})
    return (
        position.get("pos_type_label") == "AREA_INSIDE"
        and position.get("zone_hash") not in (None, 0, "0")
    )


def _blade_reported_safe(telemetry: dict[str, Any]) -> bool:
    """Return true when telemetry reports blades off and cutter RPM zero/unknown."""
    blade = telemetry.get("blade", {})
    return blade.get("reported_state") == 0 and blade.get("current_cutter_rpm") in (
        None,
        0,
    )


def _telemetry_position_delta(
    start: dict[str, Any], end: dict[str, Any]
) -> dict[str, Any]:
    """Return measured movement delta between two telemetry samples."""
    start_position = start.get("position", {})
    end_position = end.get("position", {})
    if not _position_available(start) or not _position_available(end):
        return {
            "distance": None,
            "dx": None,
            "dy": None,
            "heading_change_degrees": None,
        }
    dx = float(end_position["x"]) - float(start_position["x"])
    dy = float(end_position["y"]) - float(start_position["y"])
    start_heading = start_position.get("toward")
    end_heading = end_position.get("toward")
    return {
        "distance": math.hypot(dx, dy),
        "dx": dx,
        "dy": dy,
        "heading_change_degrees": (
            _heading_error_degrees(float(start_heading), float(end_heading))
            if start_heading is not None and end_heading is not None
            else None
        ),
    }


def _manual_velocity_forced_decision(
    decision: dict[str, Any], *, force_action: str, speed: float
) -> dict[str, Any]:
    """Return a controller decision with an explicit test action if requested."""
    if force_action == "auto":
        return decision
    return {
        **decision,
        "action": force_action,
        "reason": "force_action_requested",
        "forced": True,
        "original_action": decision.get("action"),
        "original_reason": decision.get("reason"),
        "command_not_sent": {
            "service": f"{DOMAIN}.{_manual_velocity_action_service(force_action)}",
            "data": {"speed": speed, "use_wifi": False},
        },
    }


def _manual_velocity_motion_diagnostic(
    delta: dict[str, Any],
    *,
    command_ok: bool,
    min_progress_distance: float,
    min_heading_change_degrees: float,
) -> dict[str, Any]:
    """Classify whether telemetry confirms movement after a pulse."""
    distance = delta.get("distance")
    heading_change = delta.get("heading_change_degrees")
    distance_detected = (
        distance is not None and abs(float(distance)) >= min_progress_distance
    )
    heading_detected = (
        heading_change is not None
        and abs(float(heading_change)) >= min_heading_change_degrees
    )
    detected = distance_detected or heading_detected
    if detected:
        status = "telemetry_motion_detected"
    elif command_ok:
        status = "visual_motion_possible_but_telemetry_unchanged"
    else:
        status = "command_not_confirmed"
    return {
        "status": status,
        "telemetry_motion_detected": detected,
        "distance_detected": distance_detected,
        "heading_detected": heading_detected,
        "min_progress_distance": min_progress_distance,
        "min_heading_change_degrees": min_heading_change_degrees,
    }


def _quality_rank(value: Any) -> int | None:
    """Return a coarse quality rank where larger means better."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _manual_velocity_quality_degradation(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Return explicit telemetry-quality degradation between samples."""
    baseline_position = baseline.get("position", {})
    current_position = current.get("position", {})
    reasons: list[str] = []

    if not _position_available(current):
        reasons.append("position_unavailable")
    if current_position.get("toward") is None:
        reasons.append("heading_unavailable")
    if current_position.get("pos_type_label") != "AREA_INSIDE":
        reasons.append("pos_type_not_area_inside")
    if current_position.get("zone_hash") in (None, 0, "0"):
        reasons.append("zone_hash_unavailable")
    elif (
        baseline_position.get("zone_hash") not in (None, 0, "0")
        and current_position.get("zone_hash") != baseline_position.get("zone_hash")
    ):
        reasons.append("zone_hash_changed")

    baseline_pos_level = _quality_rank(baseline_position.get("pos_level"))
    current_pos_level = _quality_rank(current_position.get("pos_level"))
    if (
        baseline_pos_level is not None
        and current_pos_level is not None
        and current_pos_level > baseline_pos_level
    ):
        reasons.append("pos_level_degraded")

    baseline_rtk = _quality_rank(baseline_position.get("rtk_status"))
    current_rtk = _quality_rank(current_position.get("rtk_status"))
    if baseline_rtk is not None and current_rtk is not None and current_rtk < baseline_rtk:
        reasons.append("rtk_status_degraded")

    return {
        "degraded": bool(reasons),
        "reasons": reasons,
        "baseline": {
            "source": baseline_position.get("source"),
            "pos_level": baseline_position.get("pos_level"),
            "rtk_status": baseline_position.get("rtk_status"),
            "pos_type_label": baseline_position.get("pos_type_label"),
            "zone_hash": baseline_position.get("zone_hash"),
            "toward": baseline_position.get("toward"),
        },
        "current": {
            "source": current_position.get("source"),
            "pos_level": current_position.get("pos_level"),
            "rtk_status": current_position.get("rtk_status"),
            "pos_type_label": current_position.get("pos_type_label"),
            "zone_hash": current_position.get("zone_hash"),
            "toward": current_position.get("toward"),
        },
    }


def _manual_velocity_heading_calibration(
    *,
    action: str,
    before: dict[str, Any],
    after: dict[str, Any],
    min_progress_distance: float,
    min_heading_change_degrees: float,
) -> dict[str, Any]:
    """Return heading calibration data from a before/after telemetry pair."""
    delta = _telemetry_position_delta(before, after)
    movement_vector_heading = None
    if delta["dx"] is not None and delta["dy"] is not None:
        distance = delta.get("distance")
        if distance is not None and float(distance) >= min_progress_distance:
            movement_vector_heading = (
                math.degrees(math.atan2(float(delta["dy"]), float(delta["dx"]))) + 360
            ) % 360

    reported_heading = before.get("position", {}).get("toward")
    heading_delta = delta.get("heading_change_degrees")
    heading_error = (
        _heading_error_degrees(float(reported_heading), movement_vector_heading)
        if reported_heading is not None and movement_vector_heading is not None
        else None
    )
    return {
        "action": action,
        "reported_heading": reported_heading,
        "movement_vector_heading": movement_vector_heading,
        "heading_delta_degrees": heading_delta,
        "heading_error_degrees": heading_error,
        "recommended_heading_offset_degrees": heading_error,
        "movement_delta": delta,
        "movement_diagnostic": _manual_velocity_motion_diagnostic(
            delta,
            command_ok=True,
            min_progress_distance=min_progress_distance,
            min_heading_change_degrees=min_heading_change_degrees,
        ),
        "interpretation": (
            "movement_vector_available"
            if movement_vector_heading is not None
            else "insufficient_translation_for_heading_calibration"
        ),
    }


def _manual_velocity_pulse_gates(
    coordinator: MammotionReportUpdateCoordinator,
    before: dict[str, Any],
    *,
    dry_run: bool,
    confirm_blades_off: bool,
    confirm_clear_area: bool,
) -> list[dict[str, Any]]:
    """Return safety gates for a manual velocity pulse probe."""
    return [
        {
            "name": "stop_primitive_available",
            "passed": hasattr(coordinator, "async_stop_manual_motion"),
            "detail": "Coordinator must expose async_stop_manual_motion().",
        },
        {
            "name": "operator_confirmed_blades_off",
            "passed": dry_run or confirm_blades_off,
            "detail": "Real pulse requires confirm_blades_off=true.",
        },
        {
            "name": "operator_confirmed_clear_area",
            "passed": dry_run or confirm_clear_area,
            "detail": "Real pulse requires confirm_clear_area=true.",
        },
        {
            "name": "mower_reports_blades_off",
            "passed": dry_run or _blade_reported_safe(before),
            "detail": "Real pulse requires blade state off and cutter RPM zero/unknown.",
        },
        {
            "name": "mower_ready",
            "passed": dry_run or before.get("work_mode_label") == "MODE_READY",
            "detail": "Real pulse requires the mower to be idle/ready, not mowing or charging.",
        },
        {
            "name": "not_docked_or_charging",
            "passed": dry_run or before.get("charge_state_label") == "not_charging",
            "detail": "Real pulse requires the mower to be off the dock and not charging.",
        },
        {
            "name": "live_map_position_available",
            "passed": dry_run or _position_available(before),
            "detail": "Real pulse requires live map-local mower position.",
        },
        {
            "name": "position_area_inside",
            "passed": dry_run or _position_has_known_area(before),
            "detail": "Real pulse requires AREA_INSIDE and a nonzero known zone hash.",
        },
        {
            "name": "map_position_nonzero",
            "passed": dry_run
            or not _is_zero_pose(
                before.get("position", {}).get("x"),
                before.get("position", {}).get("y"),
            ),
            "detail": "Real pulse requires nonzero map-local x/y coordinates.",
        },
    ]


async def _manual_velocity_pulse_test(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    action: str = "forward",
    speed: float = 0.1,
    duration_ms: int = 250,
    use_wifi: bool = False,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    followup_samples: int = 4,
    followup_interval_seconds: float = 0.5,
) -> dict[str, Any]:
    """Run or simulate one tiny manual-velocity pulse with telemetry sampling."""
    if hasattr(coordinator, "async_start_report_stream"):
        await coordinator.async_start_report_stream(duration_ms=10_000)

    before = _custom_path_telemetry_snapshot(coordinator)
    gates = _manual_velocity_pulse_gates(
        coordinator,
        before,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    service = _manual_velocity_action_service(action)
    command = {
        "service": f"{DOMAIN}.{service}",
        "data": {"speed": speed, "use_wifi": use_wifi},
    }
    result: dict[str, Any] = {
        "service": SERVICE_MANUAL_VELOCITY_PULSE_TEST,
        "mode": "dry_run" if dry_run else "real_probe",
        "dry_run": dry_run,
        "action": action,
        "speed": speed,
        "duration_ms": duration_ms,
        "use_wifi": use_wifi,
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "would_send": not dry_run and not blockers,
        "command": command if not dry_run and not blockers else None,
        "command_not_sent": command if dry_run or blockers else None,
        "real_pulse_allowed": not dry_run and not blockers,
        "blockers": blockers,
        "safety_gates": gates,
        "samples": [{"label": "before", "telemetry": before}],
        "stop_result": {"attempted": False, "ok": None, "error": None},
        "command_result": {"attempted": False, "ok": None, "error": None},
        "measured_delta": {
            "distance": None,
            "dx": None,
            "dy": None,
            "heading_change_degrees": None,
        },
    }

    if dry_run or blockers:
        result["reason"] = "dry_run" if dry_run else "safety_gates_failed"
        return result

    command_ok = False
    try:
        method = getattr(coordinator, _manual_velocity_action_method(action))
        result["command_result"]["attempted"] = True
        await method(speed=speed, use_wifi=use_wifi)
        command_ok = True
        result["command_result"]["ok"] = True
        await asyncio.sleep(duration_ms / 1000)
    except Exception as err:  # noqa: BLE001
        result["command_result"]["ok"] = False
        result["command_result"]["error"] = f"{type(err).__name__}: {err}"
    finally:
        try:
            result["stop_result"]["attempted"] = True
            await coordinator.async_stop_manual_motion(use_wifi=use_wifi)
            result["stop_result"]["ok"] = True
        except Exception as err:  # noqa: BLE001
            result["stop_result"]["ok"] = False
            result["stop_result"]["error"] = f"{type(err).__name__}: {err}"

    after_stop = _custom_path_telemetry_snapshot(coordinator)
    result["samples"].append({"label": "after_stop", "telemetry": after_stop})
    for index in range(followup_samples):
        await asyncio.sleep(followup_interval_seconds)
        result["samples"].append(
            {
                "label": f"followup_{index + 1}",
                "telemetry": _custom_path_telemetry_snapshot(coordinator),
            }
        )
    result["measured_delta"] = _telemetry_position_delta(before, after_stop)
    result["real_pulse_completed"] = command_ok and result["stop_result"]["ok"] is True
    return result


async def _manual_velocity_heading_calibration_test(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    action: str = "forward",
    speed: float = 0.4,
    duration_ms: int = 750,
    use_wifi: bool = True,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    min_progress_distance: float = 0.003,
    min_heading_change_degrees: float = 1.0,
) -> dict[str, Any]:
    """Run or simulate a tiny movement pulse and report heading calibration data."""
    pulse_result = await _manual_velocity_pulse_test(
        coordinator,
        action=action,
        speed=speed,
        duration_ms=duration_ms,
        use_wifi=use_wifi,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
        followup_samples=6,
        followup_interval_seconds=0.5,
    )
    samples = pulse_result.get("samples", [])
    before = samples[0]["telemetry"] if samples else _custom_path_telemetry_snapshot(coordinator)
    after = samples[-1]["telemetry"] if samples else before
    command_ok = pulse_result.get("command_result", {}).get("ok") is True
    calibration = _manual_velocity_heading_calibration(
        action=action,
        before=before,
        after=after,
        min_progress_distance=min_progress_distance,
        min_heading_change_degrees=min_heading_change_degrees,
    )
    if not command_ok and not dry_run:
        calibration["interpretation"] = "command_not_confirmed"
    return {
        "service": SERVICE_MANUAL_VELOCITY_HEADING_CALIBRATION_TEST,
        "mode": "dry_run" if dry_run else "real_heading_calibration_probe",
        "dry_run": dry_run,
        "action": action,
        "speed": speed,
        "duration_ms": duration_ms,
        "use_wifi": use_wifi,
        "min_progress_distance": min_progress_distance,
        "min_heading_change_degrees": min_heading_change_degrees,
        "pulse_result": pulse_result,
        "calibration": calibration,
        "full_path_execution_allowed": False,
    }


async def _manual_velocity_segment_test(  # noqa: C901
    coordinator: MammotionReportUpdateCoordinator,
    points: list[dict[str, float]],
    *,
    area_hash: int | None = None,
    speed: float = 0.4,
    pulse_duration_ms: int = 750,
    max_pulses: int = 3,
    waypoint_tolerance: float = 0.1,
    force_action: str = "auto",
    heading_offset_degrees: float = 0.0,
    min_progress_distance: float = 0.003,
    no_progress_limit: int = 2,
    min_heading_change_degrees: float = 1.0,
    use_wifi: bool = True,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    post_stop_sample_delays: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
    service_name: str = SERVICE_MANUAL_VELOCITY_SEGMENT_TEST,
) -> dict[str, Any]:
    """Run or simulate a guarded one-segment closed-loop movement probe.

    This intentionally remains a probe, not full path execution.  Real mode
    only sends repeated capped manual-velocity pulses and stops after each one.
    """
    if hasattr(coordinator, "async_start_report_stream"):
        await coordinator.async_start_report_stream(
            duration_ms=max(10_000, (pulse_duration_ms + 750) * max_pulses)
        )

    preview = _preview_custom_path(
        coordinator,
        points,
        area_hash=area_hash,
        speed=speed,
        blade_mode="off",
    )
    normalized_points = preview["points"]
    telemetry = _custom_path_telemetry_snapshot(coordinator)
    initial_decision = _manual_velocity_controller_decision(
        normalized_points,
        telemetry,
        speed=speed,
        waypoint_tolerance=waypoint_tolerance,
        heading_offset_degrees=heading_offset_degrees,
        max_pulse_seconds=pulse_duration_ms / 1000,
    )
    initial_decision = _manual_velocity_forced_decision(
        initial_decision,
        force_action=force_action,
        speed=speed,
    )
    gates = _manual_velocity_pulse_gates(
        coordinator,
        telemetry,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    if not preview["valid"]:
        gates.append(
            {
                "name": "path_validation",
                "passed": False,
                "detail": "Path must pass preview validation before real motion.",
            }
        )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    no_progress_count = 0
    cumulative_distance = 0.0
    cumulative_heading_change = 0.0

    result: dict[str, Any] = {
        **preview,
        "service": service_name,
        "mode": "dry_run" if dry_run else "real_segment_probe",
        "dry_run": dry_run,
        "speed": speed,
        "pulse_duration_ms": pulse_duration_ms,
        "max_pulses": max_pulses,
        "waypoint_tolerance": waypoint_tolerance,
        "force_action": force_action,
        "heading_offset_degrees": heading_offset_degrees,
        "min_progress_distance": min_progress_distance,
        "no_progress_limit": no_progress_limit,
        "min_heading_change_degrees": min_heading_change_degrees,
        "post_stop_sample_delays": list(post_stop_sample_delays),
        "use_wifi": use_wifi,
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "would_send": not dry_run and not blockers,
        "real_segment_allowed": not dry_run and not blockers,
        "blockers": blockers,
        "safety_gates": gates,
        "initial_telemetry": telemetry,
        "initial_controller_decision": initial_decision,
        "iterations": [],
        "final_telemetry": telemetry,
        "stop_reason": "dry_run" if dry_run else None,
        "real_execution_scope": "manual_velocity_probe_only",
        "progress_policy": {
            "min_progress_distance": min_progress_distance,
            "min_heading_change_degrees": min_heading_change_degrees,
            "no_progress_limit": no_progress_limit,
            "decision_sample": "last_post_stop_sample",
        },
        "progress_summary": {
            "no_progress_count": no_progress_count,
            "cumulative_distance": cumulative_distance,
            "cumulative_heading_change_degrees": cumulative_heading_change,
        },
        "full_path_execution_allowed": False,
    }
    if dry_run or blockers:
        result["stop_reason"] = "dry_run" if dry_run else "safety_gates_failed"
        result["command_not_sent"] = initial_decision.get("command_not_sent")
        return result

    baseline_quality_telemetry = telemetry
    for index in range(1, max_pulses + 1):
        before = _custom_path_telemetry_snapshot(coordinator)
        gates = _manual_velocity_pulse_gates(
            coordinator,
            before,
            dry_run=False,
            confirm_blades_off=confirm_blades_off,
            confirm_clear_area=confirm_clear_area,
        )
        blockers = [gate["name"] for gate in gates if not gate["passed"]]
        if blockers:
            result["stop_reason"] = "safety_gates_failed"
            result["blockers"] = blockers
            result["iterations"].append(
                {
                    "index": index,
                    "before": before,
                    "safety_gates": gates,
                    "blockers": blockers,
                    "command_result": {"attempted": False, "ok": None, "error": None},
                    "stop_result": {"attempted": False, "ok": None, "error": None},
                    "measured_delta": _telemetry_position_delta(before, before),
                }
            )
            break

        quality_degradation = _manual_velocity_quality_degradation(
            baseline_quality_telemetry, before
        )
        if quality_degradation["degraded"]:
            result["stop_reason"] = "telemetry_quality_degraded"
            result["blockers"] = quality_degradation["reasons"]
            result["iterations"].append(
                {
                    "index": index,
                    "before": before,
                    "quality_degradation": quality_degradation,
                    "command_result": {"attempted": False, "ok": None, "error": None},
                    "stop_result": {"attempted": False, "ok": None, "error": None},
                    "measured_delta": _telemetry_position_delta(before, before),
                }
            )
            break

        decision = _manual_velocity_controller_decision(
            normalized_points,
            before,
            speed=speed,
            waypoint_tolerance=waypoint_tolerance,
            heading_offset_degrees=heading_offset_degrees,
            max_pulse_seconds=pulse_duration_ms / 1000,
        )
        decision = _manual_velocity_forced_decision(
            decision,
            force_action=force_action,
            speed=speed,
        )
        action = decision["action"]
        if action == "stop":
            result["stop_reason"] = decision["reason"]
            result["iterations"].append(
                {
                    "index": index,
                    "before": before,
                    "controller_decision": decision,
                    "command_result": {"attempted": False, "ok": None, "error": None},
                    "stop_result": {"attempted": False, "ok": None, "error": None},
                    "measured_delta": _telemetry_position_delta(before, before),
                }
            )
            break

        command_result = {"attempted": False, "ok": None, "error": None}
        stop_result = {"attempted": False, "ok": None, "error": None}
        try:
            method = getattr(coordinator, _manual_velocity_action_method(action))
            command_result["attempted"] = True
            await method(speed=speed, use_wifi=use_wifi)
            command_result["ok"] = True
            await asyncio.sleep(pulse_duration_ms / 1000)
        except Exception as err:  # noqa: BLE001
            command_result["ok"] = False
            command_result["error"] = f"{type(err).__name__}: {err}"
        finally:
            try:
                stop_result["attempted"] = True
                await coordinator.async_stop_manual_motion(use_wifi=use_wifi)
                stop_result["ok"] = True
            except Exception as err:  # noqa: BLE001
                stop_result["ok"] = False
                stop_result["error"] = f"{type(err).__name__}: {err}"

        immediate_after_stop = _custom_path_telemetry_snapshot(coordinator)
        post_stop_samples = [
            {"delay_seconds": 0.0, "telemetry": immediate_after_stop}
        ]
        previous_delay = 0.0
        for delay in post_stop_sample_delays:
            await asyncio.sleep(max(0.0, delay - previous_delay))
            previous_delay = delay
            post_stop_samples.append(
                {
                    "delay_seconds": delay,
                    "telemetry": _custom_path_telemetry_snapshot(coordinator),
                }
            )
        after = post_stop_samples[-1]["telemetry"]
        measured_delta = _telemetry_position_delta(before, after)
        immediate_delta = _telemetry_position_delta(before, immediate_after_stop)
        quality_degradation = _manual_velocity_quality_degradation(
            baseline_quality_telemetry, after
        )
        movement_diagnostic = _manual_velocity_motion_diagnostic(
            measured_delta,
            command_ok=command_result["ok"] is True,
            min_progress_distance=min_progress_distance,
            min_heading_change_degrees=min_heading_change_degrees,
        )
        if movement_diagnostic["telemetry_motion_detected"]:
            no_progress_count = 0
        else:
            no_progress_count += 1
        if measured_delta["distance"] is not None:
            cumulative_distance += float(measured_delta["distance"])
        if measured_delta["heading_change_degrees"] is not None:
            cumulative_heading_change += abs(
                float(measured_delta["heading_change_degrees"])
            )
        result["iterations"].append(
            {
                "index": index,
                "before": before,
                "after": after,
                "immediate_after_stop": immediate_after_stop,
                "post_stop_samples": post_stop_samples,
                "controller_decision": decision,
                "command": {
                    "service": f"{DOMAIN}.{_manual_velocity_action_service(action)}",
                    "data": {"speed": speed, "use_wifi": use_wifi},
                },
                "command_result": command_result,
                "stop_result": stop_result,
                "immediate_delta": immediate_delta,
                "measured_delta": measured_delta,
                "movement_diagnostic": movement_diagnostic,
                "quality_degradation": quality_degradation,
                "no_progress_count": no_progress_count,
                "cumulative_distance": cumulative_distance,
                "cumulative_heading_change_degrees": cumulative_heading_change,
            }
        )
        result["final_telemetry"] = after
        result["progress_summary"] = {
            "no_progress_count": no_progress_count,
            "cumulative_distance": cumulative_distance,
            "cumulative_heading_change_degrees": cumulative_heading_change,
        }

        if command_result["ok"] is not True:
            result["stop_reason"] = "command_failed"
            break
        if stop_result["ok"] is not True:
            result["stop_reason"] = "stop_failed"
            break
        if quality_degradation["degraded"]:
            result["stop_reason"] = "telemetry_quality_degraded"
            result["blockers"] = quality_degradation["reasons"]
            break
        if no_progress_count >= no_progress_limit:
            result["stop_reason"] = "no_progress_limit_reached"
            break

    if result["stop_reason"] is None:
        result["stop_reason"] = "max_pulses_reached"
    result["pulses_sent"] = sum(
        1
        for iteration in result["iterations"]
        if iteration.get("command_result", {}).get("attempted")
    )
    return result


def _dry_run_custom_path(
    coordinator: MammotionReportUpdateCoordinator,
    points: list[dict[str, float]],
    *,
    area_hash: int | None = None,
    speed: float = 0.2,
    blade_mode: str = "off",
    heading_offset_degrees: float = 0.0,
) -> dict[str, Any]:
    """Plan a non-moving custom-path dry run.

    This intentionally does not call any coordinator method that can move the
    mower, start a task, or change blade state.
    """
    preview = _preview_custom_path(
        coordinator,
        points,
        area_hash=area_hash,
        speed=speed,
        blade_mode=blade_mode,
    )
    normalized_points = preview["points"]
    segments: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(
        zip(normalized_points, normalized_points[1:], strict=False), start=1
    ):
        distance = _path_distance([start, end])
        segments.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "distance": distance,
                "heading_degrees": _path_heading_degrees(start, end),
                "estimated_seconds": distance / speed if speed > 0 else None,
            }
        )

    safety_gates = [
        {
            "name": "dry_run_only",
            "passed": True,
            "detail": "This service never sends mower movement, task, or blade commands.",
        },
        {
            "name": "path_validation",
            "passed": bool(preview["valid"]),
            "detail": "Path must pass preview/containment validation before any future execution research.",
        },
        {
            "name": "blade_mode_off_requested",
            "passed": blade_mode == "off",
            "detail": "Only blade_mode=off is accepted.",
        },
        {
            "name": "firmware_waypoint_api_proven",
            "passed": False,
            "detail": "No proven Mammotion/pymammotion arbitrary waypoint API with guaranteed blades-off behavior has been found.",
        },
    ]
    telemetry = _custom_path_telemetry_snapshot(coordinator)
    controller_decision = _manual_velocity_controller_decision(
        normalized_points,
        telemetry,
        speed=speed,
        heading_offset_degrees=heading_offset_degrees,
    )

    return {
        **preview,
        "dry_run": True,
        "real_execution_allowed": False,
        "reason_real_execution_blocked": "firmware_waypoint_api_with_blades_off_not_proven",
        "segments": segments,
        "estimated_total_seconds": (
            preview["distance"] / speed if speed > 0 and preview["distance"] else 0
        ),
        "safety_gates": safety_gates,
        "telemetry_snapshot": telemetry,
        "manual_velocity_controller": controller_decision,
        "candidate_existing_feature_plan": {
            "strategy": "manual_velocity_controller",
            "would_send": False,
            "commands_not_sent": [
                "start_stop_blades(false)",
                "move_forward/move_left/move_right/move_backward",
                "stop/cancel_job safety fallback",
            ],
            "risk": "Existing movement commands are low-level velocity controls, not firmware waypoint following.",
        },
    }


def _custom_path_execution_readiness(
    dry_run_plan: dict[str, Any],
    *,
    dry_run: bool,
    confirm_blades_off: bool,
    allow_manual_velocity: bool,
) -> dict[str, Any]:
    """Return safety/readiness gates for a future custom-path executor."""
    telemetry = dry_run_plan["telemetry_snapshot"]
    position = telemetry.get("position", {})
    blade = telemetry.get("blade", {})
    reported_blade_state = blade.get("reported_state")
    cutter_rpm = blade.get("current_cutter_rpm")
    blade_reported_off = reported_blade_state == 0 and cutter_rpm in (None, 0)

    start_distance = None
    path_points = dry_run_plan.get("points") or []
    if path_points and position.get("x") is not None and position.get("y") is not None:
        start_distance = _path_distance(
            [
                {"x": float(position["x"]), "y": float(position["y"])},
                path_points[0],
            ]
        )

    gates = [
        {
            "name": "path_validation",
            "passed": bool(dry_run_plan.get("valid")),
            "detail": "Custom path must pass map containment and blade_mode validation.",
        },
        {
            "name": "blade_mode_off_requested",
            "passed": dry_run_plan.get("blade_mode") == "off",
            "detail": "Only blade_mode=off is accepted.",
        },
        {
            "name": "operator_confirmed_blades_off",
            "passed": bool(confirm_blades_off),
            "detail": "Future real movement requires an explicit confirm_blades_off=true request.",
        },
        {
            "name": "mower_reports_blades_off",
            "passed": blade_reported_off,
            "detail": "Telemetry must report blade state off and cutter RPM zero/unknown.",
        },
        {
            "name": "live_map_position_available",
            "passed": position.get("source") != "unavailable"
            and position.get("x") is not None
            and position.get("y") is not None,
            "detail": "Manual path following requires a live map-local mower position.",
        },
        {
            "name": "manual_velocity_opt_in",
            "passed": bool(allow_manual_velocity),
            "detail": "Existing commands are low-level velocity controls and require explicit opt-in.",
        },
        {
            "name": "firmware_waypoint_api_proven",
            "passed": False,
            "detail": "No proven Mammotion/pymammotion arbitrary waypoint API with guaranteed blades-off behavior has been found.",
        },
        {
            "name": "dry_run_guard",
            "passed": bool(dry_run),
            "detail": "This implementation never sends mower movement, task, blade, or stop commands.",
        },
    ]

    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    return {
        "can_execute_now": False,
        "real_execution_allowed": False,
        "reason_real_execution_blocked": (
            "firmware_waypoint_api_with_blades_off_not_proven"
        ),
        "requested_real_execution": not dry_run,
        "confirm_blades_off": confirm_blades_off,
        "allow_manual_velocity": allow_manual_velocity,
        "start_distance": start_distance,
        "blockers": blockers,
        "gates": gates,
    }


def _execute_custom_path(
    coordinator: MammotionReportUpdateCoordinator,
    points: list[dict[str, float]],
    *,
    area_hash: int | None = None,
    speed: float = 0.2,
    blade_mode: str = "off",
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    allow_manual_velocity: bool = False,
) -> dict[str, Any]:
    """Build a guarded custom-path execution response without moving the mower."""
    dry_run_plan = _dry_run_custom_path(
        coordinator,
        points,
        area_hash=area_hash,
        speed=speed,
        blade_mode=blade_mode,
    )
    readiness = _custom_path_execution_readiness(
        dry_run_plan,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        allow_manual_velocity=allow_manual_velocity,
    )

    return {
        **dry_run_plan,
        "service": SERVICE_EXECUTE_CUSTOM_PATH,
        "dry_run": dry_run,
        "execution_readiness": readiness,
        "real_execution_allowed": False,
        "reason_real_execution_blocked": readiness["reason_real_execution_blocked"],
        "manual_velocity_command_plan": {
            "would_send": False,
            "transport_preference": "BLE",
            "strategy": "closed_loop_manual_velocity_controller",
            "commands_not_sent": [
                "start_stop_blades(false)",
                "move_left/move_right to heading",
                "move_forward by short timed pulses",
                "position re-check after each pulse",
                "cancel_job/stop safety fallback",
            ],
            "why_not_sent": (
                "The integration has manual velocity commands, but no proven "
                "closed-loop waypoint executor with guaranteed blades-off behavior."
            ),
        },
    }


def _get_mower_by_entity_id(
    hass: HomeAssistant, entity_id: str
) -> MammotionMowerData | None:
    """Find the MammotionMowerData for the given entity_id across all config entries."""

    entity_reg = er.async_get(hass)
    entity_entry = entity_reg.async_get(entity_id)
    if entity_entry is None:
        LOGGER.error("Could not find entity %s", entity_id)
        return None

    entries: list[MammotionConfigEntry] = hass.config_entries.async_entries(DOMAIN)
    for entry in entries:
        runtime_data = getattr(entry, "runtime_data", None)
        if not runtime_data:
            continue
        mower = next(
            (
                m
                for m in runtime_data.mowers
                if entity_entry.unique_id.startswith(
                    m.reporting_coordinator.unique_name
                )
            ),
            None,
        )
        if mower is not None:
            return mower
    return None


def _get_camera_mower(hass: HomeAssistant, entity_id: str) -> MammotionMowerData | None:
    """Resolve a Mammotion camera entity across all config entries."""
    entity_entry = er.async_get(hass).async_get(entity_id)
    if (
        entity_entry is None
        or entity_entry.domain != "camera"
        or entity_entry.platform != DOMAIN
    ):
        return None
    return _get_mower_by_entity_id(hass, entity_id)


def _require_camera_mower(hass: HomeAssistant, entity_id: str) -> MammotionMowerData:
    """Resolve a camera target or raise a translated service error."""
    if mower := _get_camera_mower(hass, entity_id):
        return mower
    raise HomeAssistantError(
        translation_domain=DOMAIN,
        translation_key="camera_target_not_found",
    )


def _resolve_mower_task(
    hass: HomeAssistant, entity_id: str
) -> tuple[MammotionReportUpdateCoordinator, str] | None:
    """Resolve a task button entity_id to (coordinator, plan_id) for a mower.

    Returns ``None`` when the entity_id doesn't belong to any mower
    coordinator, or when the suffix isn't a known plan in
    ``coordinator.data.map.plan``.
    """
    entity_reg = er.async_get(hass)
    entry = entity_reg.async_get(entity_id)
    if entry is None:
        return None

    for cfg in hass.config_entries.async_entries(DOMAIN):
        if not cfg.runtime_data:
            continue
        for mower in cfg.runtime_data.mowers:
            prefix = mower.reporting_coordinator.unique_name + "_"
            if not entry.unique_id.startswith(prefix):
                continue
            plan_id = entry.unique_id[len(prefix) :]
            if plan_id in mower.reporting_coordinator.data.map.plan:
                return mower.reporting_coordinator, plan_id
    return None


def _resolve_spino_task(
    hass: HomeAssistant, entity_id: str
) -> tuple[MammotionSpinoCoordinator, int] | None:
    """Resolve a task button entity_id to (coordinator, jobid) for a Spino.

    Returns ``None`` when the entity_id doesn't belong to any Spino
    coordinator, or when the suffix isn't a known jobid in
    ``coordinator.data.plans``.
    """
    entity_reg = er.async_get(hass)
    entry = entity_reg.async_get(entity_id)
    if entry is None:
        return None

    for cfg in hass.config_entries.async_entries(DOMAIN):
        if not cfg.runtime_data:
            continue
        for spino in cfg.runtime_data.spino:
            prefix = spino.coordinator.unique_name + "_"
            if not entry.unique_id.startswith(prefix):
                continue
            suffix = entry.unique_id[len(prefix) :]
            try:
                jobid = int(suffix)
            except ValueError:
                continue
            if jobid in spino.coordinator.data.plans:
                return spino.coordinator, jobid
    return None


def _resolve_device(
    hass: HomeAssistant, entity_id: str
) -> tuple[MammotionReportUpdateCoordinator | MammotionSpinoCoordinator, str] | None:
    """Resolve any entity_id to (coordinator, kind) — used by create / refresh.

    ``kind`` is ``"mower"`` or ``"spino"``.  Returns the *device's* primary
    coordinator regardless of which of the device's entities was targeted.
    """
    entity_reg = er.async_get(hass)
    entry = entity_reg.async_get(entity_id)
    if entry is None:
        return None

    for cfg in hass.config_entries.async_entries(DOMAIN):
        if not cfg.runtime_data:
            continue
        for mower in cfg.runtime_data.mowers:
            if entry.unique_id.startswith(mower.reporting_coordinator.unique_name):
                return mower.reporting_coordinator, "mower"
        for spino in cfg.runtime_data.spino:
            if entry.unique_id.startswith(spino.coordinator.unique_name):
                return spino.coordinator, "spino"
    return None


def _raise_task_not_found(entity_id: str) -> None:
    """Raise a translated HomeAssistantError when no task matches."""
    raise HomeAssistantError(
        translation_domain=DOMAIN,
        translation_key="task_not_found",
        translation_placeholders={"plan_id": entity_id},
    )


def _build_mower_plan(data: dict[str, Any], base: Plan | None = None) -> Plan:
    """Map service kwargs onto a ``Plan`` dataclass (mower side).

    When ``base`` is given the unspecified fields come from it (edit
    path); otherwise defaults from ``Plan()`` apply (create path).
    """
    plan = dataclasses.replace(base) if base is not None else Plan()
    if name := data.get("name"):
        plan = plan.with_renamed(name)
    if "enabled" in data:
        plan = plan.with_enabled(bool(data["enabled"]))
    for key in (
        "weeks",
        "start_time",
        "end_time",
        "start_date",
        "end_date",
        "trigger_type",
        "day",
        "knife_height",
        "speed",
        "edge_mode",
        "route_angle",
        "route_spacing",
        "zone_hashs",
    ):
        if key in data:
            plan = dataclasses.replace(plan, **{key: data[key]})
    return plan


def _build_spino_plan(data: dict[str, Any], base: PoolPlan | None = None) -> PoolPlan:
    """Map service kwargs onto a ``PoolPlan`` dataclass (spino side)."""
    plan = dataclasses.replace(base) if base is not None else PoolPlan()
    if name := data.get("name"):
        plan = plan.with_renamed(name)
    if "enabled" in data:
        plan = plan.with_enabled(bool(data["enabled"]))
    if "weeks" in data:
        plan = dataclasses.replace(plan, weeks=list(data["weeks"]))
    if "sub_mode" in data:
        plan = dataclasses.replace(plan, sub_mode=list(data["sub_mode"]))
    for key, target in (
        ("trigger_type", "triggertype"),
        ("start_date", "startdate"),
        ("end_date", "enddate"),
    ):
        if key in data:
            plan = dataclasses.replace(plan, **{target: data[key]})
    for key in ("day", "work_mode", "speed", "operating_power", "starttime"):
        if key in data:
            plan = dataclasses.replace(plan, **{key: data[key]})
    return plan


@callback
def async_setup_services(hass: HomeAssistant) -> None:  # noqa: C901
    """Register Mammotion services."""

    async def handle_refresh_stream(call: ServiceCall) -> None:
        mower = _require_camera_mower(hass, call.data[ATTR_ENTITY_ID])
        (
            stream_data,
            agora_response,
        ) = await mower.reporting_coordinator.async_check_stream_expiry(force=True)
        if stream_data is None or agora_response is None:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="camera_temporarily_unavailable",
            )

    async def handle_start_video(call: ServiceCall) -> None:
        mower = _require_camera_mower(hass, call.data[ATTR_ENTITY_ID])
        try:
            await mower.reporting_coordinator.join_webrtc_channel()
        except HomeAssistantError as err:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="camera_temporarily_unavailable",
            ) from err

    async def handle_stop_video(call: ServiceCall) -> None:
        mower = _require_camera_mower(hass, call.data[ATTR_ENTITY_ID])
        try:
            await mower.reporting_coordinator.leave_webrtc_channel()
        except HomeAssistantError as err:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="camera_temporarily_unavailable",
            ) from err

    async def handle_get_tokens(call: ServiceCall) -> dict[str, Any]:
        mower = _require_camera_mower(hass, call.data[ATTR_ENTITY_ID])
        coordinator = mower.reporting_coordinator
        cached = coordinator.get_stream_data()
        if cached is None or cached.data is None:
            stream_data, agora_response = await coordinator.async_check_stream_expiry(
                force=True
            )
            if stream_data is None or agora_response is None:
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="camera_temporarily_unavailable",
                )
            return stream_data.to_dict()
        return cached.data.to_dict()

    async def handle_movement(call: ServiceCall, direction: str) -> None:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="mower_target_not_found",
            )
        method = getattr(mower.reporting_coordinator, direction)
        await method(speed=call.data["speed"], use_wifi=call.data["use_wifi"])

    hass.services.async_register(
        DOMAIN, SERVICE_REFRESH_STREAM, handle_refresh_stream, schema=CAMERA_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_START_VIDEO, handle_start_video, schema=CAMERA_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_STOP_VIDEO, handle_stop_video, schema=CAMERA_SCHEMA
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_TOKENS,
        handle_get_tokens,
        schema=CAMERA_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    for service_name, method_name in (
        (SERVICE_MOVE_FORWARD, "async_move_forward"),
        (SERVICE_MOVE_LEFT, "async_move_left"),
        (SERVICE_MOVE_RIGHT, "async_move_right"),
        (SERVICE_MOVE_BACKWARD, "async_move_back"),
    ):
        hass.services.async_register(
            DOMAIN,
            service_name,
            lambda call, method_name=method_name: handle_movement(call, method_name),
            schema=MOVEMENT_SCHEMA,
        )

    async def handle_get_geojson(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        coordinator = mower.reporting_coordinator
        if coordinator.is_online():
            await coordinator.async_start_report_stream(duration_ms=300_000)
        return apply_geojson_offset(
            coordinator.data.map.generated_geojson,
            coordinator.map_offset_lat,
            coordinator.map_offset_lon,
        )

    async def handle_get_mow_path_geojson(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        coordinator = mower.reporting_coordinator
        return apply_geojson_offset(
            coordinator.data.map.generated_mow_path_geojson,
            coordinator.map_offset_lat,
            coordinator.map_offset_lon,
        )

    async def handle_get_mow_progress_geojson(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        coordinator = mower.reporting_coordinator
        device_type = DeviceType.value_of_str(coordinator.device_name)
        firmware = coordinator.data.device_firmwares.main_controller
        if device_type.is_support_dynamics_line(firmware):
            geojson = coordinator.data.map.generated_dynamics_line_geojson
        else:
            geojson = coordinator.data.map.generated_mow_progress_geojson
        return apply_geojson_offset(
            geojson, coordinator.map_offset_lat, coordinator.map_offset_lon
        )

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_GEOJSON,
        handle_get_geojson,
        schema=GEOJSON_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_MOW_PATH_GEOJSON,
        handle_get_mow_path_geojson,
        schema=GEOJSON_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_MOW_PROGRESS_GEOJSON,
        handle_get_mow_progress_geojson,
        schema=GEOJSON_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    async def handle_get_map_data(call: ServiceCall) -> dict[str, Any]:
        from pymammotion.data.model.device import MowingDevice  # noqa: PLC0415

        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        device_data = cast(MowingDevice, mower.reporting_coordinator.data)
        map_dict = dataclasses.asdict(device_data.map)
        return cast(
            dict[str, Any],
            _stringify_large_ints(
                {
                    "area": map_dict.get("area", {}),
                    "svg": map_dict.get("svg", {}),
                    "area_name": map_dict.get("area_name", []),
                }
            ),
        )

    async def handle_get_tasks(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return {"tasks": _normalize_mower_tasks(mower.reporting_coordinator)}

    async def handle_get_areas(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return {"areas": _normalize_mower_areas(mower.reporting_coordinator)}

    async def handle_export_map(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return _export_mower_map(mower.reporting_coordinator)

    async def handle_export_tasks(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return _export_mower_tasks(mower.reporting_coordinator)

    async def handle_validate_custom_path(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return _validate_custom_path(
            mower.reporting_coordinator,
            cast(list[dict[str, float]], call.data["points"]),
            area_hash=call.data.get("area_hash"),
            speed=call.data["speed"],
            blade_mode=call.data["blade_mode"],
        )

    async def handle_preview_custom_path(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return _preview_custom_path(
            mower.reporting_coordinator,
            cast(list[dict[str, float]], call.data["points"]),
            area_hash=call.data.get("area_hash"),
            speed=call.data["speed"],
            blade_mode=call.data["blade_mode"],
        )

    async def handle_dry_run_custom_path(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return _dry_run_custom_path(
            mower.reporting_coordinator,
            cast(list[dict[str, float]], call.data["points"]),
            area_hash=call.data.get("area_hash"),
            speed=call.data["speed"],
            blade_mode=call.data["blade_mode"],
            heading_offset_degrees=call.data["heading_offset_degrees"],
        )

    async def handle_execute_custom_path(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return _execute_custom_path(
            mower.reporting_coordinator,
            cast(list[dict[str, float]], call.data["points"]),
            area_hash=call.data.get("area_hash"),
            speed=call.data["speed"],
            blade_mode=call.data["blade_mode"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            allow_manual_velocity=call.data["allow_manual_velocity"],
        )

    async def handle_manual_velocity_pulse_test(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _manual_velocity_pulse_test(
            mower.reporting_coordinator,
            action=call.data["action"],
            speed=call.data["speed"],
            duration_ms=call.data["duration_ms"],
            use_wifi=call.data["use_wifi"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
        )

    async def handle_manual_velocity_segment_test(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _manual_velocity_segment_test(
            mower.reporting_coordinator,
            cast(list[dict[str, float]], call.data["points"]),
            area_hash=call.data.get("area_hash"),
            speed=call.data["speed"],
            pulse_duration_ms=call.data["pulse_duration_ms"],
            max_pulses=call.data["max_pulses"],
            waypoint_tolerance=call.data["waypoint_tolerance"],
            force_action=call.data["force_action"],
            heading_offset_degrees=call.data["heading_offset_degrees"],
            min_progress_distance=call.data["min_progress_distance"],
            no_progress_limit=call.data["no_progress_limit"],
            min_heading_change_degrees=call.data["min_heading_change_degrees"],
            use_wifi=call.data["use_wifi"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
        )

    async def handle_manual_velocity_multi_pulse_test(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _manual_velocity_segment_test(
            mower.reporting_coordinator,
            cast(list[dict[str, float]], call.data["points"]),
            area_hash=call.data.get("area_hash"),
            speed=call.data["speed"],
            pulse_duration_ms=call.data["pulse_duration_ms"],
            max_pulses=call.data["max_pulses"],
            waypoint_tolerance=call.data["waypoint_tolerance"],
            force_action=call.data["force_action"],
            heading_offset_degrees=call.data["heading_offset_degrees"],
            min_progress_distance=call.data["min_progress_distance"],
            no_progress_limit=call.data["no_progress_limit"],
            min_heading_change_degrees=call.data["min_heading_change_degrees"],
            use_wifi=call.data["use_wifi"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            service_name=SERVICE_MANUAL_VELOCITY_MULTI_PULSE_TEST,
        )

    async def handle_manual_velocity_heading_calibration_test(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _manual_velocity_heading_calibration_test(
            mower.reporting_coordinator,
            action=call.data["action"],
            speed=call.data["speed"],
            duration_ms=call.data["duration_ms"],
            use_wifi=call.data["use_wifi"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            min_progress_distance=call.data["min_progress_distance"],
            min_heading_change_degrees=call.data["min_heading_change_degrees"],
        )

    async def handle_svg_add(call: ServiceCall) -> dict[str, Any]:
        from pymammotion.data.model.device import MowingDevice  # noqa: PLC0415
        from pymammotion.utility.svg import build_svg_for_area  # noqa: PLC0415

        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        coordinator = mower.reporting_coordinator
        device_data = cast(MowingDevice, coordinator.data)
        area_hash: int = call.data["area_hash"]
        frame_list = device_data.map.area.get(area_hash)
        boundary: list[CommDataCouple] = []
        if frame_list:
            for frame in sorted(
                frame_list.data, key=lambda f: getattr(f, "current_frame", 0)
            ):
                boundary.extend(getattr(frame, "data_couple", []))
        msg = build_svg_for_area(
            area_hash=area_hash,
            boundary=boundary,
            svg_file_data=call.data["svg_data"],
            svg_file_name=call.data["svg_file_name"],
            scale=call.data["scale"],
            rotate=call.data["rotate"],
            base_width_m=call.data["base_width_m"],
            base_height_m=call.data["base_height_m"],
        )
        if "x_move" in call.data:
            msg.svg_message.x_move = call.data["x_move"]
        if "y_move" in call.data:
            msg.svg_message.y_move = call.data["y_move"]
        result = await coordinator.send_svg_command(msg)
        return {"device_hash": str(result)}

    async def handle_svg_update(call: ServiceCall) -> dict[str, Any]:
        from pymammotion.data.model.device import MowingDevice  # noqa: PLC0415
        from pymammotion.utility.svg import build_svg_update  # noqa: PLC0415

        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        coordinator = mower.reporting_coordinator
        device_data = cast(MowingDevice, coordinator.data)
        area_hash: int = call.data["area_hash"]
        frame_list = device_data.map.area.get(area_hash)
        boundary: list[CommDataCouple] = []
        if frame_list:
            for frame in sorted(
                frame_list.data, key=lambda f: getattr(f, "current_frame", 0)
            ):
                boundary.extend(getattr(frame, "data_couple", []))
        msg = build_svg_update(
            device_hash=call.data["device_hash"],
            area_hash=area_hash,
            boundary=boundary,
            svg_file_data=call.data["svg_data"],
            svg_file_name=call.data["svg_file_name"],
            scale=call.data["scale"],
            rotate=call.data["rotate"],
            base_width_m=call.data["base_width_m"],
            base_height_m=call.data["base_height_m"],
        )
        if "x_move" in call.data:
            msg.svg_message.x_move = call.data["x_move"]
        if "y_move" in call.data:
            msg.svg_message.y_move = call.data["y_move"]
        result = await coordinator.send_svg_command(msg)
        return {"device_hash": str(result)}

    async def handle_svg_delete(call: ServiceCall) -> dict[str, Any]:
        from pymammotion.utility.svg import build_svg_delete  # noqa: PLC0415

        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        msg = build_svg_delete(
            device_hash=call.data["device_hash"],
            area_hash=call.data["area_hash"],
        )
        await mower.reporting_coordinator.send_svg_command(msg)
        return {}

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_MAP_DATA,
        handle_get_map_data,
        schema=GEOJSON_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_TASKS,
        handle_get_tasks,
        schema=GEOJSON_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_AREAS,
        handle_get_areas,
        schema=GEOJSON_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_EXPORT_MAP,
        handle_export_map,
        schema=GEOJSON_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_EXPORT_TASKS,
        handle_export_tasks,
        schema=GEOJSON_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_VALIDATE_CUSTOM_PATH,
        handle_validate_custom_path,
        schema=VALIDATE_CUSTOM_PATH_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_PREVIEW_CUSTOM_PATH,
        handle_preview_custom_path,
        schema=VALIDATE_CUSTOM_PATH_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_DRY_RUN_CUSTOM_PATH,
        handle_dry_run_custom_path,
        schema=DRY_RUN_CUSTOM_PATH_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_EXECUTE_CUSTOM_PATH,
        handle_execute_custom_path,
        schema=EXECUTE_CUSTOM_PATH_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MANUAL_VELOCITY_PULSE_TEST,
        handle_manual_velocity_pulse_test,
        schema=MANUAL_VELOCITY_PULSE_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MANUAL_VELOCITY_SEGMENT_TEST,
        handle_manual_velocity_segment_test,
        schema=MANUAL_VELOCITY_SEGMENT_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MANUAL_VELOCITY_MULTI_PULSE_TEST,
        handle_manual_velocity_multi_pulse_test,
        schema=MANUAL_VELOCITY_MULTI_PULSE_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MANUAL_VELOCITY_HEADING_CALIBRATION_TEST,
        handle_manual_velocity_heading_calibration_test,
        schema=MANUAL_VELOCITY_HEADING_CALIBRATION_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_SVG_ADD,
        handle_svg_add,
        schema=SVG_ADD_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_SVG_UPDATE,
        handle_svg_update,
        schema=SVG_UPDATE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_SVG_DELETE,
        handle_svg_delete,
        schema=SVG_DELETE_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )

    # === Task / schedule services =====================================
    #
    # Modify ops (rename / enable / delete / copy / edit) target a task
    # button entity_id; we resolve to the mower or Spino path by checking
    # the entity's owning coordinator.  Create / refresh target *any*
    # entity that belongs to the device (typically the lawn_mower or
    # vacuum entity).

    async def handle_rename_task(call: ServiceCall) -> None:
        entity_id = call.data[ATTR_ENTITY_ID]
        if (mower := _resolve_mower_task(hass, entity_id)) is not None:
            await mower[0].async_rename_mower_task(mower[1], call.data["name"])
            return
        if (spino := _resolve_spino_task(hass, entity_id)) is not None:
            await spino[0].async_rename_spino_task(spino[1], call.data["name"])
            return
        _raise_task_not_found(entity_id)

    async def handle_set_task_enabled(call: ServiceCall) -> None:
        enabled = bool(call.data["enabled"])
        for entity_id in call.data[ATTR_ENTITY_ID]:
            if (mower := _resolve_mower_task(hass, entity_id)) is not None:
                await mower[0].async_set_mower_task_enabled(mower[1], enabled)
                continue
            if (spino := _resolve_spino_task(hass, entity_id)) is not None:
                await spino[0].async_set_spino_task_enabled(spino[1], enabled)
                continue
            _raise_task_not_found(entity_id)

    async def handle_delete_task(call: ServiceCall) -> None:
        for entity_id in call.data[ATTR_ENTITY_ID]:
            if (mower := _resolve_mower_task(hass, entity_id)) is not None:
                await mower[0].async_delete_mower_task(mower[1])
                continue
            if (spino := _resolve_spino_task(hass, entity_id)) is not None:
                await spino[0].async_delete_spino_task(spino[1])
                continue
            _raise_task_not_found(entity_id)

    async def handle_copy_task(call: ServiceCall) -> None:
        entity_id = call.data[ATTR_ENTITY_ID]
        new_name: str | None = call.data.get("name")
        if (mower := _resolve_mower_task(hass, entity_id)) is not None:
            await mower[0].async_copy_mower_task(mower[1], new_name=new_name)
            return
        if (spino := _resolve_spino_task(hass, entity_id)) is not None:
            await spino[0].async_copy_spino_task(spino[1], new_name=new_name)
            return
        _raise_task_not_found(entity_id)

    async def handle_edit_task(call: ServiceCall) -> None:
        entity_id = call.data[ATTR_ENTITY_ID]
        if (mower := _resolve_mower_task(hass, entity_id)) is not None:
            base = mower[0].data.map.plan[mower[1]]
            await mower[0].async_edit_mower_task(
                _build_mower_plan(dict(call.data), base)
            )
            return
        if (spino := _resolve_spino_task(hass, entity_id)) is not None:
            base = spino[0].data.plans[spino[1]]
            await spino[0].async_edit_spino_task(
                _build_spino_plan(dict(call.data), base)
            )
            return
        _raise_task_not_found(entity_id)

    async def handle_create_task(call: ServiceCall) -> None:
        entity_id = call.data[ATTR_ENTITY_ID]
        resolved = _resolve_device(hass, entity_id)
        if resolved is None:
            _raise_task_not_found(entity_id)
            return  # pragma: no cover — unreachable after raise above
        coord, kind = resolved
        if kind == "mower":
            await cast(MammotionReportUpdateCoordinator, coord).async_create_mower_task(
                _build_mower_plan(dict(call.data))
            )
        else:
            await cast(MammotionSpinoCoordinator, coord).async_create_spino_task(
                _build_spino_plan(dict(call.data))
            )

    async def handle_refresh_tasks(call: ServiceCall) -> None:
        entity_id = call.data[ATTR_ENTITY_ID]
        resolved = _resolve_device(hass, entity_id)
        if resolved is None:
            _raise_task_not_found(entity_id)
            return
        coord, kind = resolved
        if kind == "mower":
            await cast(
                MammotionReportUpdateCoordinator, coord
            ).async_refresh_mower_tasks()
        else:
            await cast(MammotionSpinoCoordinator, coord).async_refresh_spino_tasks()

    async def handle_start_task(call: ServiceCall) -> None:
        """Run a stored mower schedule immediately ("start task" / "start schedule").

        Backed by the APK's ``singleSchedule(planId)`` →
        ``NavPlanTaskExecute(sub_cmd=1, id=plan_id)`` (file MACommandHelper.java,
        line 1673). Spino has no equivalent in the wire protocol — we raise a
        translated error rather than silently doing nothing so users see why
        the press / service call didn't take effect.
        """
        entity_id = call.data[ATTR_ENTITY_ID]
        if (mower := _resolve_mower_task(hass, entity_id)) is not None:
            await mower[0].start_task(mower[1])
            return
        if _resolve_spino_task(hass, entity_id) is not None:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="start_task_unsupported_on_spino",
            )
        _raise_task_not_found(entity_id)

    hass.services.async_register(
        DOMAIN, SERVICE_RENAME_TASK, handle_rename_task, schema=RENAME_TASK_SCHEMA
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_SET_TASK_ENABLED,
        handle_set_task_enabled,
        schema=SET_TASK_ENABLED_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN, SERVICE_DELETE_TASK, handle_delete_task, schema=DELETE_TASK_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_COPY_TASK, handle_copy_task, schema=COPY_TASK_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_EDIT_TASK, handle_edit_task, schema=EDIT_TASK_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_CREATE_TASK, handle_create_task, schema=CREATE_TASK_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_REFRESH_TASKS, handle_refresh_tasks, schema=REFRESH_TASKS_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_START_TASK, handle_start_task, schema=START_TASK_SCHEMA
    )
