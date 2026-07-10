"""Mammotion services."""

from __future__ import annotations

import asyncio
import dataclasses
import math
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypedDict, cast

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
SERVICE_EXPORT_RUNTIME_STATE = "export_runtime_state"
SERVICE_EXPORT_ACTIVE_ROUTE = "export_active_route"
SERVICE_VALIDATE_CUSTOM_PATH = "validate_custom_path"
SERVICE_PREVIEW_CUSTOM_PATH = "preview_custom_path"
SERVICE_DRY_RUN_CUSTOM_PATH = "dry_run_custom_path"
SERVICE_EXECUTE_CUSTOM_PATH = "execute_custom_path"
SERVICE_MANUAL_VELOCITY_PULSE_TEST = "manual_velocity_pulse_test"
SERVICE_MANUAL_VELOCITY_SEGMENT_TEST = "manual_velocity_segment_test"
SERVICE_MANUAL_VELOCITY_MULTI_PULSE_TEST = "manual_velocity_multi_pulse_test"
SERVICE_MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST = (
    "manual_velocity_cumulative_pulse_test"
)
SERVICE_MANUAL_VELOCITY_HEADING_CALIBRATION_TEST = (
    "manual_velocity_heading_calibration_test"
)
SERVICE_RAW_PYMAMMOTION_MOTION_PROBE = "raw_pymammotion_motion_probe"
SERVICE_RAW_PYMAMMOTION_EXECUTE_SEGMENT = "raw_pymammotion_execute_segment"
SERVICE_RAW_PYMAMMOTION_ANGULAR_CALIBRATION = (
    "raw_pymammotion_angular_calibration"
)
SERVICE_RAW_PYMAMMOTION_TURN_TO_HEADING = "raw_pymammotion_turn_to_heading"
SERVICE_RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT = (
    "raw_pymammotion_execute_vector_segment"
)
SERVICE_RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT = (
    "raw_pymammotion_execute_multi_segment"
)
SERVICE_FORWARD_TWO_PULSE_LATENCY_TEST = "forward_two_pulse_latency_test"
SERVICE_POSITION_FEEDBACK_DIAGNOSTIC = "position_feedback_diagnostic"
SERVICE_VIO_MOTION_PROBE = "vio_motion_probe"
SERVICE_VIO_TURN_PROBE = "vio_turn_probe"
SERVICE_VIO_TURN_TO_HEADING = "vio_turn_to_heading"
SERVICE_RAW_MOTION_READINESS_TEST = "raw_motion_readiness_test"
SERVICE_RAW_VECTOR_READINESS_TEST = "raw_vector_readiness_test"
SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT = "experimental_execute_segment"
SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT_BURST = "experimental_execute_segment_burst"
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

DEFAULT_CALIBRATED_FORWARD_HEADING_DEGREES = 270.0
DEFAULT_CALIBRATED_FORWARD_HEADING_TOLERANCE_DEGREES = 45.0
DEFAULT_EXPERIMENTAL_SEGMENT_PULSES_PER_BURST = 1
DEFAULT_EXPERIMENTAL_SEGMENT_MAX_BURSTS = 3
DEFAULT_EXPERIMENTAL_SEGMENT_STOP_MODE = "firmware"
DEFAULT_EXPERIMENTAL_SEGMENT_USE_WIFI = False
RAW_PYMAMMOTION_MOTION_COMMANDS = (
    "send_movement",
    "move_forward",
    "move_back",
    "move_left",
    "move_right",
)

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


class _TelemetryDelaySample(TypedDict):
    """Telemetry sample captured after an optional delay."""

    delay_seconds: float
    telemetry: dict[str, Any]

# Optional schedule fields shared by both device kinds.  The HA service
# layer normalises them into the per-kind Plan / PoolPlan dataclass.
_SCHEDULE_FIELDS: dict[Any, Any] = {
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
_MOWER_ONLY_FIELDS: dict[Any, Any] = {
    vol.Optional("knife_height"): vol.All(vol.Coerce(int), vol.Range(min=20, max=100)),
    vol.Optional("speed"): vol.Coerce(float),
    vol.Optional("edge_mode"): vol.All(vol.Coerce(int), vol.Range(min=0, max=2)),
    vol.Optional("route_angle"): vol.All(vol.Coerce(int), vol.Range(min=0, max=179)),
    vol.Optional("route_spacing"): vol.All(vol.Coerce(int), vol.Range(min=0)),
    vol.Optional("zone_hashs"): vol.All(cv.ensure_list, [vol.Coerce(int)]),
}

# Spino-only fields keyed by names on ``pymammotion.PoolPlan``.
_SPINO_ONLY_FIELDS: dict[Any, Any] = {
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

DEFAULT_HEADING_OFFSET_CANDIDATES = (110.0, 0.0, 90.0, -90.0, 180.0)

_HEADING_OFFSET_CANDIDATES_SCHEMA = vol.All(
    cv.ensure_list,
    vol.Length(min=1, max=13),
    [
        vol.All(
            vol.Coerce(float),
            vol.Range(min=-180.0, max=180.0),
        )
    ],
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
        vol.Optional("stop_mode", default="immediate"): vol.In(
            ["immediate", "delayed", "firmware"]
        ),
        vol.Optional("stop_delay_ms", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=5000)
        ),
        vol.Optional("post_command_sample_delays", default=[0, 2, 10, 30, 60]): vol.All(
            cv.ensure_list,
            [vol.All(vol.Coerce(float), vol.Range(min=0.0, max=120.0))],
        ),
        vol.Optional("use_wifi", default=DEFAULT_EXPERIMENTAL_SEGMENT_USE_WIFI): cv.boolean,
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

MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("points"): vol.All(
            cv.ensure_list,
            vol.Length(min=2, max=2),
            [_CUSTOM_PATH_POINT_SCHEMA],
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
        vol.Optional("stop_mode", default="immediate"): vol.In(
            ["immediate", "delayed", "firmware"]
        ),
        vol.Optional("stop_delay_ms", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=5000)
        ),
        vol.Optional("heading_offset_degrees", default=0.0): vol.All(
            vol.Coerce(float), vol.Range(min=-180.0, max=180.0)
        ),
        vol.Optional("min_progress_distance", default=0.003): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)
        ),
        vol.Optional("min_heading_change_degrees", default=1.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=45.0)
        ),
        vol.Optional("use_wifi", default=True): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
        vol.Optional("cumulative_sample_delays", default=[0, 10, 20, 30, 45, 60, 90, 120]): vol.All(
            cv.ensure_list,
            [
                vol.All(
                    vol.Coerce(float),
                    vol.Range(min=0.0, max=120.0),
                )
            ],
        ),
        vol.Optional(
            "heading_offset_candidates",
            default=list(DEFAULT_HEADING_OFFSET_CANDIDATES),
        ): _HEADING_OFFSET_CANDIDATES_SCHEMA,
    },
    extra=vol.ALLOW_EXTRA,
)

MANUAL_VELOCITY_HEADING_CALIBRATION_TEST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("action", default="forward"): vol.In(
            ["forward", "backward", "turn_left", "turn_right"]
        ),
        vol.Optional("speed", default=0.4): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.4)
        ),
        vol.Optional("duration_ms", default=750): vol.All(
            vol.Coerce(int), vol.Range(min=50, max=750)
        ),
        vol.Optional("stop_mode", default="firmware"): vol.In(
            ["immediate", "delayed", "firmware"]
        ),
        vol.Optional("stop_delay_ms", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=5000)
        ),
        vol.Optional(
            "post_command_sample_delays",
            default=[0, 10, 20, 30, 45, 60],
        ): vol.All(
            cv.ensure_list,
            [
                vol.All(
                    vol.Coerce(float),
                    vol.Range(min=0.0, max=120.0),
                )
            ],
        ),
        vol.Optional("use_wifi", default=False): cv.boolean,
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

RAW_PYMAMMOTION_MOTION_PROBE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("command", default="send_movement"): vol.In(
            RAW_PYMAMMOTION_MOTION_COMMANDS
        ),
        vol.Optional("linear_speed", default=400): vol.All(
            vol.Coerce(int), vol.Range(min=-1000, max=1000)
        ),
        vol.Optional("angular_speed", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=-1000, max=1000)
        ),
        vol.Optional("speed", default=0.4): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.4)
        ),
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("sample_delays", default=[0, 5, 10, 20, 30, 45, 60]): vol.All(
            cv.ensure_list,
            [vol.All(vol.Coerce(float), vol.Range(min=0.0, max=120.0))],
        ),
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

FORWARD_TWO_PULSE_LATENCY_TEST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("linear_speed", default=200): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("pulse_count", default=2): vol.All(
            vol.Coerce(int), vol.Range(min=2, max=5)
        ),
        vol.Optional("pulse_gap_seconds", default=5.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.5, max=30.0)
        ),
        vol.Optional("telemetry_timeout_seconds", default=60.0): vol.All(
            vol.Coerce(float), vol.Range(min=5.0, max=300.0)
        ),
        vol.Optional("telemetry_sample_interval_seconds", default=1.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.5, max=10.0)
        ),
        vol.Optional("min_position_change_distance", default=0.003): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)
        ),
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

POSITION_FEEDBACK_DIAGNOSTIC_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("linear_speed", default=200): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("pulse_count", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=5)
        ),
        vol.Optional("pulse_gap_seconds", default=5.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.5, max=30.0)
        ),
        vol.Optional("refresh_wait_seconds", default=2.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=30.0)
        ),
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

VIO_MOTION_PROBE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("linear_speed", default=200): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("drive_seconds", default=6.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=12.0)
        ),
        vol.Optional("sample_interval_seconds", default=1.5): vol.All(
            vol.Coerce(float), vol.Range(min=0.5, max=5.0)
        ),
        vol.Optional("post_stop_samples", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=6)
        ),
        vol.Optional("max_displacement_m", default=1.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=2.0)
        ),
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

VIO_TURN_PROBE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("angular_speed", default=180): vol.All(
            vol.Coerce(int), vol.Range(min=-1000, max=1000)
        ),
        vol.Optional("linear_speed", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=-1000, max=1000)
        ),
        vol.Optional("drive_seconds", default=6.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=12.0)
        ),
        vol.Optional("sample_interval_seconds", default=1.5): vol.All(
            vol.Coerce(float), vol.Range(min=0.5, max=5.0)
        ),
        vol.Optional("post_stop_samples", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=6)
        ),
        vol.Optional("max_displacement_m", default=0.5): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=2.0)
        ),
        vol.Optional("min_heading_change_degrees", default=3.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.5, max=45.0)
        ),
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

VIO_TURN_TO_HEADING_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("target_vision_heading"): vol.Coerce(float),
        vol.Optional("heading_tolerance_degrees", default=8.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=45.0)
        ),
        vol.Optional("angular_speed", default=500): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("pulse_duration_ms", default=1500): vol.All(
            vol.Coerce(int), vol.Range(min=200, max=4000)
        ),
        vol.Optional("slow_pulse_duration_ms", default=700): vol.All(
            vol.Coerce(int), vol.Range(min=200, max=4000)
        ),
        vol.Optional("slow_threshold_degrees", default=15.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=90.0)
        ),
        vol.Optional("refresh_wait_seconds", default=2.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=10.0)
        ),
        vol.Optional("max_commands", default=8): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=20)
        ),
        vol.Optional("min_progress_degrees", default=2.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=20.0)
        ),
        vol.Optional("max_displacement_m", default=0.5): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=2.0)
        ),
        vol.Optional("invert_direction", default=False): cv.boolean,
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

RAW_PYMAMMOTION_EXECUTE_SEGMENT_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("points"): vol.All(
            cv.ensure_list,
            vol.Length(min=2, max=2),
            [_CUSTOM_PATH_POINT_SCHEMA],
        ),
        vol.Optional("area_hash"): vol.Any(vol.Coerce(int), str),
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("linear_speed_fast", default=400): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("linear_speed_slow", default=200): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("max_commands", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=3)
        ),
        vol.Optional("waypoint_tolerance", default=0.08): vol.All(
            vol.Coerce(float), vol.Range(min=0.02, max=0.5)
        ),
        vol.Optional("min_progress_distance", default=0.01): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)
        ),
        vol.Optional("sample_delays", default=[0, 5, 10, 20, 30, 45, 60]): vol.All(
            cv.ensure_list,
            [vol.All(vol.Coerce(float), vol.Range(min=0.0, max=120.0))],
        ),
    },
    extra=vol.ALLOW_EXTRA,
)

RAW_PYMAMMOTION_ANGULAR_CALIBRATION_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("direction", default="positive_heading"): vol.In(
            ["positive_heading", "negative_heading"]
        ),
        vol.Optional("angular_speed", default=180): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("target_heading_delta_degrees", default=10.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=90.0)
        ),
        vol.Optional("max_commands", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=5)
        ),
        vol.Optional("min_heading_change_degrees", default=1.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=45.0)
        ),
        vol.Optional("max_translation_distance", default=0.25): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=2.0)
        ),
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("sample_delays", default=[0, 5, 10, 20, 30, 45, 60]): vol.All(
            cv.ensure_list,
            [vol.All(vol.Coerce(float), vol.Range(min=0.0, max=120.0))],
        ),
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

RAW_PYMAMMOTION_TURN_TO_HEADING_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("target_heading_degrees"): vol.All(
            vol.Coerce(float), vol.Range(min=-360.0, max=360.0)
        ),
        vol.Optional("heading_tolerance_degrees", default=3.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.5, max=30.0)
        ),
        vol.Optional("angular_speed_fast", default=180): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("angular_speed_slow", default=90): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("slow_turn_threshold_degrees", default=8.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=45.0)
        ),
        vol.Optional("max_commands", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=5)
        ),
        vol.Optional("min_heading_change_degrees", default=0.5): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=45.0)
        ),
        vol.Optional("max_translation_distance", default=0.25): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=2.0)
        ),
        vol.Optional("pulse_duration_ms", default=300.0): vol.All(
            vol.Coerce(float), vol.Range(min=50.0, max=2000.0)
        ),
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("sample_delays", default=[0, 5, 10, 20, 30, 45, 60]): vol.All(
            cv.ensure_list,
            [vol.All(vol.Coerce(float), vol.Range(min=0.0, max=120.0))],
        ),
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("points"): vol.All(
            cv.ensure_list,
            vol.Length(min=2, max=2),
            [_CUSTOM_PATH_POINT_SCHEMA],
        ),
        vol.Optional("area_hash"): vol.Any(vol.Coerce(int), str),
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("linear_speed_fast", default=400): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("linear_speed_slow", default=200): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("slow_linear_threshold", default=0.15): vol.All(
            vol.Coerce(float), vol.Range(min=0.02, max=1.0)
        ),
        vol.Optional("max_turn_commands", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=5)
        ),
        vol.Optional("max_linear_commands", default=1): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=3)
        ),
        vol.Optional("max_linear_pulse_ceiling"): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=200)
        ),
        vol.Optional("max_no_progress_pulses", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=20)
        ),
        vol.Optional("linear_distance_ceiling_factor", default=2.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=10.0)
        ),
        vol.Optional("heading_tolerance_degrees", default=3.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.5, max=30.0)
        ),
        vol.Optional("angular_speed_fast", default=180): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("angular_speed_slow", default=180): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("slow_turn_threshold_degrees", default=8.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=45.0)
        ),
        vol.Optional("waypoint_tolerance", default=0.08): vol.All(
            vol.Coerce(float), vol.Range(min=0.02, max=0.5)
        ),
        vol.Optional("min_progress_distance", default=0.005): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)
        ),
        vol.Optional("min_heading_change_degrees", default=0.5): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=45.0)
        ),
        vol.Optional("max_turn_translation_distance", default=0.25): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=2.0)
        ),
        vol.Optional("calibrated_forward_heading_offset_degrees", default=116.5): vol.All(
            vol.Coerce(float), vol.Range(min=-180.0, max=180.0)
        ),
        vol.Optional("turn_pulse_duration_ms", default=300.0): vol.All(
            vol.Coerce(float), vol.Range(min=50.0, max=2000.0)
        ),
        vol.Optional("linear_pulse_duration_ms", default=300.0): vol.All(
            vol.Coerce(float), vol.Range(min=50.0, max=2000.0)
        ),
        vol.Optional("sample_delays", default=[0, 5, 10, 20, 30, 45, 60]): vol.All(
            cv.ensure_list,
            [vol.All(vol.Coerce(float), vol.Range(min=0.0, max=120.0))],
        ),
    },
    extra=vol.ALLOW_EXTRA,
)

RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("points"): vol.All(
            cv.ensure_list,
            vol.Length(min=2, max=4),
            [_CUSTOM_PATH_POINT_SCHEMA],
        ),
        vol.Optional("area_hash"): vol.Any(vol.Coerce(int), str),
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("max_real_segments", default=1): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=3)
        ),
        vol.Optional("linear_speed_fast", default=400): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("linear_speed_slow", default=200): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("slow_linear_threshold", default=0.15): vol.All(
            vol.Coerce(float), vol.Range(min=0.02, max=1.0)
        ),
        vol.Optional("max_turn_commands", default=4): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=5)
        ),
        vol.Optional("max_linear_commands", default=2): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=3)
        ),
        vol.Optional("max_linear_pulse_ceiling"): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=200)
        ),
        vol.Optional("max_no_progress_pulses", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=20)
        ),
        vol.Optional("linear_distance_ceiling_factor", default=2.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=10.0)
        ),
        vol.Optional("heading_tolerance_degrees", default=3.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.5, max=30.0)
        ),
        vol.Optional("angular_speed_fast", default=180): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("angular_speed_slow", default=180): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("slow_turn_threshold_degrees", default=8.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=45.0)
        ),
        vol.Optional("waypoint_tolerance", default=0.08): vol.All(
            vol.Coerce(float), vol.Range(min=0.02, max=0.5)
        ),
        vol.Optional("min_progress_distance", default=0.01): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)
        ),
        vol.Optional("min_heading_change_degrees", default=0.5): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=45.0)
        ),
        vol.Optional("max_turn_translation_distance", default=0.25): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=2.0)
        ),
        vol.Optional("calibrated_forward_heading_offset_degrees", default=116.5): vol.All(
            vol.Coerce(float), vol.Range(min=-180.0, max=180.0)
        ),
        vol.Optional("turn_pulse_duration_ms", default=300.0): vol.All(
            vol.Coerce(float), vol.Range(min=50.0, max=2000.0)
        ),
        vol.Optional("linear_pulse_duration_ms", default=300.0): vol.All(
            vol.Coerce(float), vol.Range(min=50.0, max=2000.0)
        ),
        vol.Optional("sample_delays", default=[0, 5, 10, 20, 30, 45, 60]): vol.All(
            cv.ensure_list,
            [vol.All(vol.Coerce(float), vol.Range(min=0.0, max=120.0))],
        ),
    },
    extra=vol.ALLOW_EXTRA,
)

RAW_MOTION_READINESS_TEST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("max_real_steps", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=4)
        ),
        vol.Optional("sample_delays", default=[0, 5, 10, 20, 30, 45, 60]): vol.All(
            cv.ensure_list,
            [vol.All(vol.Coerce(float), vol.Range(min=0.0, max=120.0))],
        ),
    },
    extra=vol.ALLOW_EXTRA,
)

RAW_VECTOR_READINESS_TEST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("max_real_steps", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=3)
        ),
        vol.Optional("target_distance", default=0.10): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.3)
        ),
        vol.Optional("turn_delta_degrees", default=10.0): vol.All(
            vol.Coerce(float), vol.Range(min=3.0, max=45.0)
        ),
        vol.Optional("calibrated_forward_heading_offset_degrees", default=116.5): vol.All(
            vol.Coerce(float), vol.Range(min=-180.0, max=180.0)
        ),
        vol.Optional("max_turn_commands", default=4): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=5)
        ),
        vol.Optional("max_linear_commands", default=2): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=3)
        ),
        vol.Optional("sample_delays", default=[0, 5, 10, 20, 30, 45, 60]): vol.All(
            cv.ensure_list,
            [vol.All(vol.Coerce(float), vol.Range(min=0.0, max=120.0))],
        ),
    },
    extra=vol.ALLOW_EXTRA,
)

EXPERIMENTAL_EXECUTE_SEGMENT_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("points"): vol.All(
            cv.ensure_list,
            vol.Length(min=2, max=2),
            [_CUSTOM_PATH_POINT_SCHEMA],
        ),
        vol.Optional("area_hash"): vol.Coerce(int),
        vol.Optional("speed", default=0.4): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.4)
        ),
        vol.Optional("pulse_duration_ms", default=750): vol.All(
            vol.Coerce(int), vol.Range(min=50, max=750)
        ),
        vol.Optional("max_pulses", default=1): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=3)
        ),
        vol.Optional("waypoint_tolerance", default=0.1): vol.All(
            vol.Coerce(float), vol.Range(min=0.02, max=0.5)
        ),
        vol.Optional("heading_offset_degrees", default=0.0): vol.All(
            vol.Coerce(float), vol.Range(min=-180.0, max=180.0)
        ),
        vol.Optional("min_progress_distance", default=0.003): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)
        ),
        vol.Optional("min_heading_change_degrees", default=1.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=45.0)
        ),
        vol.Optional("use_wifi", default=True): cv.boolean,
        vol.Required("dry_run"): vol.All(cv.boolean, vol.Equal(False)),
        vol.Required("confirm_blades_off"): vol.All(cv.boolean, vol.Equal(True)),
        vol.Required("confirm_clear_area"): vol.All(cv.boolean, vol.Equal(True)),
    },
    extra=vol.ALLOW_EXTRA,
)

EXPERIMENTAL_EXECUTE_SEGMENT_BURST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("points"): vol.All(
            cv.ensure_list,
            vol.Length(min=2, max=2),
            [_CUSTOM_PATH_POINT_SCHEMA],
        ),
        vol.Optional("area_hash"): vol.Coerce(int),
        vol.Optional("speed", default=0.4): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.4)
        ),
        vol.Optional("pulse_duration_ms", default=750): vol.All(
            vol.Coerce(int), vol.Range(min=50, max=750)
        ),
        vol.Optional("pulses_per_burst", default=DEFAULT_EXPERIMENTAL_SEGMENT_PULSES_PER_BURST): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=3)
        ),
        vol.Optional("max_bursts", default=DEFAULT_EXPERIMENTAL_SEGMENT_MAX_BURSTS): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=3)
        ),
        vol.Optional("waypoint_tolerance", default=0.1): vol.All(
            vol.Coerce(float), vol.Range(min=0.02, max=0.5)
        ),
        vol.Optional("heading_offset_degrees", default=0.0): vol.All(
            vol.Coerce(float), vol.Range(min=-180.0, max=180.0)
        ),
        vol.Optional(
            "heading_offset_candidates",
            default=list(DEFAULT_HEADING_OFFSET_CANDIDATES),
        ): _HEADING_OFFSET_CANDIDATES_SCHEMA,
        vol.Optional("stop_mode", default=DEFAULT_EXPERIMENTAL_SEGMENT_STOP_MODE): vol.In(
            ["immediate", "delayed", "firmware"]
        ),
        vol.Optional("stop_delay_ms", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=5000)
        ),
        vol.Optional("min_progress_distance", default=0.003): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)
        ),
        vol.Optional("min_heading_change_degrees", default=1.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=45.0)
        ),
        vol.Optional("allow_unproven_turns", default=False): cv.boolean,
        vol.Optional("calibrated_forward_heading_degrees", default=DEFAULT_CALIBRATED_FORWARD_HEADING_DEGREES): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=360.0)
        ),
        vol.Optional(
            "calibrated_forward_heading_tolerance_degrees",
            default=DEFAULT_CALIBRATED_FORWARD_HEADING_TOLERANCE_DEGREES,
        ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=180.0)),
        vol.Optional(
            "cumulative_sample_delays",
            default=[0, 10, 20, 30, 45, 60, 90, 120],
        ): vol.All(
            cv.ensure_list,
            [
                vol.All(
                    vol.Coerce(float),
                    vol.Range(min=0.0, max=120.0),
                )
            ],
        ),
        vol.Optional("use_wifi", default=False): cv.boolean,
        vol.Required("dry_run"): vol.All(cv.boolean, vol.Equal(False)),
        vol.Required("confirm_blades_off"): vol.All(cv.boolean, vol.Equal(True)),
        vol.Required("confirm_clear_area"): vol.All(cv.boolean, vol.Equal(True)),
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

_SVG_COMMON_FIELDS: dict[Any, Any] = {
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


def _coerce_optional_int(value: int | str | None) -> int | None:
    """Return an optional integer parsed from service/API values."""
    if value is None:
        return None
    return int(value)


def _safe_asdict(obj: Any) -> Any:
    """Return a JSON-ish representation for dataclass or plain test objects."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
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
    device_data = coordinator.data
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
    device_data = coordinator.data
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
    device_data = coordinator.data
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


def _manual_velocity_next_waypoint(  # noqa: C901
    path_points: list[dict[str, float]],
    current: dict[str, float],
    *,
    waypoint_tolerance: float,
) -> tuple[int | None, dict[str, float] | None, float | None, list[dict[str, Any]]]:
    """Return the next useful waypoint for closed-loop manual velocity control."""
    distances: list[dict[str, Any]] = [
        {"index": index, "distance": _path_distance([current, point])}
        for index, point in enumerate(path_points)
    ]
    if not distances:
        return None, None, None, distances
    if distances[-1]["distance"] <= waypoint_tolerance:
        return None, None, None, distances

    if len(path_points) > 1:
        segment_projections: list[dict[str, Any]] = []
        for index, (start, end) in enumerate(
            zip(path_points, path_points[1:], strict=False)
        ):
            segment_dx = end["x"] - start["x"]
            segment_dy = end["y"] - start["y"]
            segment_len_sq = segment_dx**2 + segment_dy**2
            if segment_len_sq <= 0:
                continue
            progress = (
                (current["x"] - start["x"]) * segment_dx
                + (current["y"] - start["y"]) * segment_dy
            ) / segment_len_sq
            clamped_progress = min(1.0, max(0.0, progress))
            closest = {
                "x": start["x"] + segment_dx * clamped_progress,
                "y": start["y"] + segment_dy * clamped_progress,
            }
            segment_projections.append(
                {
                    "segment_index": index,
                    "target_index": index + 1,
                    "progress": progress,
                    "clamped_progress": clamped_progress,
                    "distance_to_segment": _path_distance([current, closest]),
                }
            )
        if segment_projections:
            closest_segment = min(
                segment_projections,
                key=lambda item: item["distance_to_segment"],
            )
            for item in distances:
                item["segment_projections"] = segment_projections
            if closest_segment["distance_to_segment"] <= max(
                waypoint_tolerance * 2, 0.02
            ):
                target_index = int(closest_segment["target_index"])
                if (
                    closest_segment["clamped_progress"] >= 1.0
                    and target_index + 1 < len(path_points)
                    and distances[target_index]["distance"] <= waypoint_tolerance
                ):
                    target_index += 1
                distance_to_target = distances[target_index]["distance"]
                if distance_to_target > waypoint_tolerance:
                    return (
                        target_index,
                        path_points[target_index],
                        float(distance_to_target),
                        distances,
                    )

    active = next(
        (
            item
            for item in distances
            if item["distance"] > waypoint_tolerance
        ),
        None,
    )
    if active is None:
        return None, None, None, distances

    # A drawn path usually includes the mower's start point as point 0.  During
    # live testing the mower can drift past that point, making point 0 a stale
    # target behind the mower.  If the next waypoint is no farther away, prefer
    # it so the controller continues down the path instead of turning back to
    # the original start marker.
    if active["index"] == 0 and len(distances) > 1:
        start = path_points[0]
        next_point = path_points[1]
        segment_dx = next_point["x"] - start["x"]
        segment_dy = next_point["y"] - start["y"]
        segment_len_sq = segment_dx**2 + segment_dy**2
        progress_along_first_segment = None
        if segment_len_sq > 0:
            progress_along_first_segment = (
                (current["x"] - start["x"]) * segment_dx
                + (current["y"] - start["y"]) * segment_dy
            ) / segment_len_sq
        next_distance = distances[1]["distance"]
        if (
            progress_along_first_segment is not None
            and progress_along_first_segment > 0
        ) or next_distance <= active["distance"] + waypoint_tolerance:
            active = distances[1]

    index = int(active["index"])
    return index, path_points[index], float(active["distance"]), distances


def _isoformat_or_none(value: Any) -> str | None:
    """Return datetime-like values as ISO strings for HA service responses."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _export_mower_map(coordinator: MammotionReportUpdateCoordinator) -> dict[str, Any]:
    """Return read-only map export data for route planning/debugging."""
    device_data = coordinator.data
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


def _active_mowing_detected(telemetry: dict[str, Any], ha_state: str | None) -> bool:
    """Return true when runtime state indicates firmware-managed mowing is active."""
    return ha_state == "mowing" or telemetry.get("work_mode_label") == "MODE_WORKING"


def _runtime_blade_diagnostics(telemetry: dict[str, Any]) -> dict[str, Any]:
    """Return blade diagnostics plus a conservative motion-safety decision."""
    blade = dict(telemetry.get("blade", {}) or {})
    reported_state = blade.get("reported_state")
    rpm = blade.get("current_cutter_rpm")
    reported_on = reported_state not in (None, 0, "0")
    rpm_nonzero = rpm not in (None, 0, "0")
    blade_safe = not reported_on and not rpm_nonzero
    blockers = []
    if reported_on:
        blockers.append("blade_reported_on")
    if rpm_nonzero:
        blockers.append("blade_rpm_nonzero")
    return {
        **blade,
        "blade_safe_for_motion": blade_safe,
        "safety_blockers": blockers,
    }


def _runtime_motion_safety_summary(
    telemetry: dict[str, Any],
    *,
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return conservative safety summary for diagnostics and future motion gates."""
    blade = _runtime_blade_diagnostics(telemetry)
    route_status = _runtime_route_status(
        telemetry,
        ha_state=ha_state,
        active_route=active_route,
    )
    active_mowing = _active_mowing_detected(telemetry, ha_state)
    position_valid = _position_available(telemetry) and _position_has_known_area(
        telemetry
    )
    blockers = list(blade["safety_blockers"])
    if active_mowing:
        blockers.append("active_mowing_detected")
    if route_status["blocks_motion"]:
        blockers.append("active_route_detected")
    if not position_valid:
        blockers.append("position_not_valid_for_motion")
    return {
        "allowed_for_manual_motion": not blockers,
        "blockers": blockers,
        "blade_safe_for_motion": blade["blade_safe_for_motion"],
        "active_mowing_detected": active_mowing,
        "active_route_detected": route_status["route_present"],
        "active_route_status": route_status,
        "position_valid_for_motion": position_valid,
    }


def _runtime_route_status(
    telemetry: dict[str, Any],
    *,
    ha_state: str | None,
    active_route: dict[str, Any] | None,
) -> dict[str, Any]:
    """Classify active route/progress data as live, stale, absent, or ambiguous."""
    route_present = bool(
        active_route
        and (
            active_route.get("mow_path_feature_count", 0) > 0
            or active_route.get("mow_progress_feature_count", 0) > 0
        )
    )
    active_progress = active_route.get("active_progress") if active_route else None
    progress_present = bool(
        active_route and active_route.get("mow_progress_feature_count", 0) > 0
    )
    progress_is_active = bool(
        isinstance(active_progress, dict) and active_progress.get("is_active") is True
    )
    work_mode_label = telemetry.get("work_mode_label")
    active_mowing = _active_mowing_detected(telemetry, ha_state)
    ready_or_paused = ha_state in ("paused", "idle") or work_mode_label == "MODE_READY"
    if not route_present:
        reason = "no_route"
        blocks_motion = False
    elif active_mowing:
        reason = "live_route_while_mowing"
        blocks_motion = True
    elif ready_or_paused:
        reason = "stale_route_while_ready"
        blocks_motion = False
    else:
        reason = "route_state_ambiguous"
        blocks_motion = True
    return {
        "route_present": route_present,
        "progress_present": progress_present,
        "progress_is_active": progress_is_active,
        "blocks_motion": blocks_motion,
        "reason": reason,
        "ha_state": ha_state,
        "work_mode_label": work_mode_label,
    }


def _geojson_features(geojson: Any) -> list[dict[str, Any]]:
    """Return GeoJSON features from a feature collection-like dict."""
    if not isinstance(geojson, dict):
        return []
    features = geojson.get("features")
    return [feature for feature in features if isinstance(feature, dict)] if isinstance(features, list) else []


def _feature_coordinate_count(geometry: dict[str, Any]) -> int | None:
    """Return coordinate count for common GeoJSON geometry types."""
    coordinates = geometry.get("coordinates")
    geometry_type = geometry.get("type")
    if not isinstance(coordinates, list):
        return None
    if geometry_type == "LineString":
        return len(coordinates)
    if geometry_type in ("MultiLineString", "Polygon"):
        return sum(len(line) for line in coordinates if isinstance(line, list))
    if geometry_type == "Point":
        return 1
    return None


def _normalize_route_feature(feature: dict[str, Any]) -> dict[str, Any]:
    """Normalize a GeoJSON route/progress feature while preserving coordinates."""
    raw_geometry = feature.get("geometry")
    geometry: dict[str, Any] = raw_geometry if isinstance(raw_geometry, dict) else {}
    raw_properties = feature.get("properties")
    properties: dict[str, Any] = (
        raw_properties if isinstance(raw_properties, dict) else {}
    )
    return cast(
        dict[str, Any],
        _stringify_large_ints(
            {
                "type_name": properties.get("type_name"),
                "path_hash": properties.get("path_hash"),
                "transaction_id": properties.get("transaction_id"),
                "path_type": properties.get("path_type"),
                "total_path_num": properties.get("total_path_num"),
                "is_active": properties.get("is_active"),
                "now_index": properties.get("now_index"),
                "total_points": properties.get("total_points"),
                "point_count": properties.get("point_count")
                or _feature_coordinate_count(geometry),
                "length": properties.get("length"),
                "area": properties.get("area"),
                "time": properties.get("time"),
                "geometry_type": geometry.get("type"),
                "coordinates": geometry.get("coordinates"),
                "raw_properties": properties,
            }
        ),
    )


def _export_active_route(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Return read-only active firmware route/progress diagnostics."""
    mow_path_geojson = apply_geojson_offset(
        coordinator.data.map.generated_mow_path_geojson,
        coordinator.map_offset_lat,
        coordinator.map_offset_lon,
    )
    device_type = DeviceType.value_of_str(coordinator.device_name)
    firmware = coordinator.data.device_firmwares.main_controller
    if device_type.is_support_dynamics_line(firmware):
        progress_geojson = coordinator.data.map.generated_dynamics_line_geojson
        progress_source = "generated_dynamics_line_geojson"
    else:
        progress_geojson = coordinator.data.map.generated_mow_progress_geojson
        progress_source = "generated_mow_progress_geojson"
    progress_geojson = apply_geojson_offset(
        progress_geojson,
        coordinator.map_offset_lat,
        coordinator.map_offset_lon,
    )
    mow_path_features = [
        _normalize_route_feature(feature) for feature in _geojson_features(mow_path_geojson)
    ]
    progress_features = [
        _normalize_route_feature(feature) for feature in _geojson_features(progress_geojson)
    ]
    active_progress = next(
        (feature for feature in progress_features if feature.get("is_active") is True),
        progress_features[0] if progress_features else None,
    )
    return cast(
        dict[str, Any],
        _stringify_large_ints(
            {
                "coordinate_system": "mower_map_geojson",
                "mow_path_feature_count": len(mow_path_features),
                "mow_progress_feature_count": len(progress_features),
                "progress_source": progress_source,
                "mow_path_features": mow_path_features,
                "mow_progress_features": progress_features,
                "active_progress": active_progress,
                "raw": {
                    "mow_path_geojson": mow_path_geojson,
                    "mow_progress_geojson": progress_geojson,
                },
            }
        ),
    )


def _export_runtime_state(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return read-only runtime diagnostics for active path/motion work."""
    telemetry = _custom_path_telemetry_snapshot(coordinator)
    route = active_route if active_route is not None else _export_active_route(coordinator)
    blade = _runtime_blade_diagnostics(telemetry)
    safety = _runtime_motion_safety_summary(
        telemetry,
        ha_state=ha_state,
        active_route=route,
    )
    route_status = safety["active_route_status"]
    return {
        "ha_state": ha_state,
        "online": telemetry.get("online"),
        "work_mode": telemetry.get("work_mode"),
        "work_mode_label": telemetry.get("work_mode_label"),
        "charge_state": telemetry.get("charge_state"),
        "charge_state_label": telemetry.get("charge_state_label"),
        "position": telemetry.get("position"),
        "position_candidates": telemetry.get("position_candidates"),
        "blade": blade,
        "transport": telemetry.get("transport"),
        "active_route_summary": {
            "mow_path_feature_count": route.get("mow_path_feature_count", 0),
            "mow_progress_feature_count": route.get("mow_progress_feature_count", 0),
            "active_progress": route.get("active_progress"),
            "status": route_status,
        },
        "safety": safety,
        "manual_motion_execution_policy": _manual_motion_execution_policy(),
        "last_task_sync": _isoformat_or_none(
            getattr(coordinator, "last_task_sync", None)
        ),
        "last_map_sync": _isoformat_or_none(
            getattr(coordinator, "last_map_sync", None)
        ),
        "last_map_task_error": getattr(coordinator, "last_map_task_error", None),
        "active_transport": getattr(coordinator, "active_transport_state", None),
        "ble_only_fallback_mode": getattr(
            coordinator, "ble_only_fallback_mode", None
        ),
        "last_cloud_login_success": _isoformat_or_none(
            getattr(coordinator, "last_cloud_login_success", None)
        ),
        "last_token_refresh": _isoformat_or_none(
            getattr(coordinator, "last_token_refresh", None)
        ),
        "last_command_failure_reason": getattr(
            coordinator, "last_command_failure_reason", None
        ),
        "last_camera_stream_failure_code": getattr(
            coordinator, "last_camera_stream_failure_code", None
        ),
    }


def _manual_motion_execution_policy() -> dict[str, Any]:
    """Return the current conservative manual-motion execution policy."""
    return {
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
        "default_transport": (
            "wifi"
            if DEFAULT_EXPERIMENTAL_SEGMENT_USE_WIFI
            else "ble_preferred"
        ),
        "default_stop_mode": DEFAULT_EXPERIMENTAL_SEGMENT_STOP_MODE,
        "default_pulses_per_burst": (
            DEFAULT_EXPERIMENTAL_SEGMENT_PULSES_PER_BURST
        ),
        "default_max_bursts": DEFAULT_EXPERIMENTAL_SEGMENT_MAX_BURSTS,
        "calibrated_forward_heading_degrees": (
            DEFAULT_CALIBRATED_FORWARD_HEADING_DEGREES
        ),
        "calibrated_forward_heading_tolerance_degrees": (
            DEFAULT_CALIBRATED_FORWARD_HEADING_TOLERANCE_DEGREES
        ),
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
        and _is_manual_motion_area_label(position.get("pos_type_label"))
        and position.get("zone_hash") not in (None, 0, "0")
    )


def _is_manual_motion_area_label(label: Any) -> bool:
    """Return true when position type is acceptable for guarded manual motion."""
    return label in {"AREA_INSIDE", "TURN_AREA_INSIDE", "CHANNEL_AREA_OVERLAP"}


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


def _custom_path_position_candidates(
    data: Any, coordinator: MammotionReportUpdateCoordinator | None = None
) -> list[dict[str, Any]]:
    """Return all known candidate map-position sources for diagnostics."""

    def build_candidate(
        source: str,
        *,
        x: Any = None,
        y: Any = None,
        toward: Any = None,
        pos_level: Any = None,
        rtk_status: Any = None,
        pos_type: Any = None,
        zone_hash: Any = None,
    ) -> dict[str, Any]:
        safe_zone_hash = _json_safe_int(zone_hash) if zone_hash is not None else None
        area_name = (
            coordinator.get_area_entity_name(int(zone_hash))
            if coordinator is not None
            and hasattr(coordinator, "get_area_entity_name")
            and zone_hash not in (None, 0, "0")
            else None
        )
        return {
            "source": source,
            "x": x,
            "y": y,
            "toward": toward,
            "pos_level": pos_level,
            "pos_level_label": _position_mode_label(pos_level),
            "rtk_status": _enum_value(rtk_status),
            "rtk_status_label": _rtk_status_label(rtk_status),
            "pos_type": pos_type,
            "pos_type_label": _pos_type_label(pos_type),
            "zone_hash": safe_zone_hash,
            "area_name": area_name,
            "stale_zero_area_out": _is_stale_zero_area_out_pose(
                x, y, pos_type, zone_hash
            ),
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

    candidates: list[dict[str, Any]] = []
    mowing_state = _safe_attr_path(data, "mowing_state")
    if mowing_state is not None and (
        _safe_attr_path(mowing_state, "pos_x") is not None
        or _safe_attr_path(mowing_state, "pos_y") is not None
    ):
        candidates.append(
            build_candidate(
                "mowing_state",
                x=_safe_attr_path(mowing_state, "pos_x"),
                y=_safe_attr_path(mowing_state, "pos_y"),
                toward=_safe_attr_path(mowing_state, "toward"),
                pos_level=_safe_attr_path(mowing_state, "pos_level"),
                rtk_status=_safe_attr_path(mowing_state, "rtk_status"),
                pos_type=_safe_attr_path(mowing_state, "pos_type"),
                zone_hash=_safe_attr_path(mowing_state, "zone_hash"),
            )
        )

    report_location = _latest_location(data)
    if report_location is not None and (
        _safe_attr_path(report_location, "real_pos_x") is not None
        or _safe_attr_path(report_location, "real_pos_y") is not None
    ):
        candidates.append(
            build_candidate(
                "report_data.locations[0]",
                x=_scale_report_position(
                    _safe_attr_path(report_location, "real_pos_x")
                ),
                y=_scale_report_position(
                    _safe_attr_path(report_location, "real_pos_y")
                ),
                toward=_scale_report_position(
                    _safe_attr_path(report_location, "real_toward")
                ),
                pos_type=_safe_attr_path(report_location, "pos_type"),
                zone_hash=_safe_attr_path(report_location, "bol_hash"),
            )
        )

    location_pos_type = _safe_attr_path(data, "location.position_type")
    location_zone_hash = _safe_attr_path(data, "location.work_zone")
    location_toward = _safe_attr_path(data, "location.orientation")
    if location_pos_type is not None or location_zone_hash is not None:
        candidates.append(
            build_candidate(
                "location_metadata",
                toward=location_toward,
                pos_type=location_pos_type,
                zone_hash=location_zone_hash,
            )
        )

    rtk = _safe_attr_path(data, "report_data.rtk")
    if rtk is not None:
        candidates.append(
            build_candidate(
                "report_data.rtk",
                pos_level=_safe_attr_path(rtk, "pos_level"),
                rtk_status=_safe_attr_path(rtk, "status"),
            )
        )
    return candidates


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
        "position_candidates": _custom_path_position_candidates(data, coordinator),
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
    target_index, target, distance_to_target, waypoint_distances = (
        _manual_velocity_next_waypoint(
            path_points,
            current,
            waypoint_tolerance=waypoint_tolerance,
        )
    )

    if target is None or target_index is None or distance_to_target is None:
        return {
            **base_response,
            "action": "stop",
            "reason": "path_complete",
            "target_index": None,
            "distance_to_target": 0.0,
            "waypoint_distances": waypoint_distances,
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
        "waypoint_distances": waypoint_distances,
        "target_heading_degrees": target_heading,
        "heading_error_degrees": heading_error,
        "distance_to_target": distance_to_target,
        "command_not_sent": command,
    }


def _manual_velocity_heading_offset_candidates(
    heading_offset_degrees: float,
    heading_offset_candidates: list[float] | tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """Return de-duplicated heading-offset candidates, preserving order."""
    raw_candidates = (
        tuple(heading_offset_candidates)
        if heading_offset_candidates
        else DEFAULT_HEADING_OFFSET_CANDIDATES
    )
    candidates: list[float] = []
    for raw_candidate in (*raw_candidates, heading_offset_degrees):
        candidate = float(raw_candidate)
        if candidate < -180.0 or candidate > 180.0:
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates) or (float(heading_offset_degrees),)


def _manual_velocity_decision_rank(decision: dict[str, Any]) -> tuple[int, float]:
    """Rank decisions for safe heading candidate selection."""
    heading_error = decision.get("heading_error_degrees")
    abs_heading_error = (
        abs(float(heading_error)) if heading_error is not None else float("inf")
    )
    if decision.get("action") == "forward" and decision.get("reason") == "heading_aligned":
        return (0, abs_heading_error)
    if decision.get("action") in {"turn_left", "turn_right"}:
        return (1, abs_heading_error)
    return (2, abs_heading_error)


def _manual_velocity_best_heading_decision(
    path_points: list[dict[str, float]],
    telemetry: dict[str, Any],
    *,
    speed: float,
    waypoint_tolerance: float = 0.4,
    heading_tolerance_degrees: float = 15.0,
    heading_offset_degrees: float = 0.0,
    heading_offset_candidates: list[float] | tuple[float, ...] | None = None,
    max_pulse_seconds: float = 0.5,
) -> dict[str, Any]:
    """Choose the safest controller decision across heading-offset candidates."""
    candidates = _manual_velocity_heading_offset_candidates(
        heading_offset_degrees,
        heading_offset_candidates,
    )
    decisions = [
        _manual_velocity_controller_decision(
            path_points,
            telemetry,
            speed=speed,
            waypoint_tolerance=waypoint_tolerance,
            heading_tolerance_degrees=heading_tolerance_degrees,
            heading_offset_degrees=candidate,
            max_pulse_seconds=max_pulse_seconds,
        )
        for candidate in candidates
    ]
    selected = min(decisions, key=_manual_velocity_decision_rank)
    return {
        **selected,
        "selected_heading_offset_degrees": selected["heading_offset_degrees"],
        "heading_offset_candidates": list(candidates),
        "heading_offset_selection": {
            "strategy": "prefer_forward_then_lowest_heading_error",
            "candidate_count": len(candidates),
        },
        "heading_offset_diagnostics": [
            {
                "heading_offset_degrees": decision.get("heading_offset_degrees"),
                "action": decision.get("action"),
                "reason": decision.get("reason"),
                "current_heading_degrees": decision.get("current_heading_degrees"),
                "corrected_heading_degrees": decision.get(
                    "corrected_heading_degrees"
                ),
                "target_heading_degrees": decision.get("target_heading_degrees"),
                "heading_error_degrees": decision.get("heading_error_degrees"),
                "distance_to_target": decision.get("distance_to_target"),
            }
            for decision in decisions
        ],
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


async def _manual_velocity_command_attempt(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    action: str,
    speed: float,
    use_wifi: bool,
) -> dict[str, Any]:
    """Run one low-level manual motion command and return command diagnostics."""
    method_name = _manual_velocity_action_method(action)
    method = getattr(coordinator, method_name)
    transport_preference = "wifi" if use_wifi else "ble_preferred"
    started = time.monotonic()
    result: dict[str, Any] = {
        "attempted": True,
        "ok": None,
        "error": None,
        "action": action,
        "coordinator_method": method_name,
        "service": f"{DOMAIN}.{_manual_velocity_action_service(action)}",
        "speed": speed,
        "use_wifi": use_wifi,
        "transport_preference": transport_preference,
        "ack": None,
        "duration_ms": None,
    }
    try:
        ack = await method(speed=speed, use_wifi=use_wifi)
        result["ack"] = ack
        result["ok"] = ack is not False
    except Exception as err:  # noqa: BLE001
        result["ok"] = False
        result["error"] = f"{type(err).__name__}: {err}"
    finally:
        result["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
    return result


async def _manual_velocity_stop_attempt(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    use_wifi: bool,
) -> dict[str, Any]:
    """Run the manual-motion stop primitive and return command diagnostics."""
    transport_preference = "wifi" if use_wifi else "ble_preferred"
    started = time.monotonic()
    result: dict[str, Any] = {
        "attempted": True,
        "ok": None,
        "error": None,
        "coordinator_method": "async_stop_manual_motion",
        "use_wifi": use_wifi,
        "transport_preference": transport_preference,
        "ack": None,
        "duration_ms": None,
    }
    try:
        ack = await coordinator.async_stop_manual_motion(use_wifi=use_wifi)
        result["ack"] = ack
        if isinstance(ack, dict):
            result["ok"] = all(value is not False for value in ack.values())
        else:
            result["ok"] = ack is not False
    except Exception as err:  # noqa: BLE001
        result["ok"] = False
        result["error"] = f"{type(err).__name__}: {err}"
    finally:
        result["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
    return result


def _manual_velocity_delayed_progress_diagnostics(
    before: dict[str, Any],
    samples: Sequence[Mapping[str, object]],
    decision: dict[str, Any],
    *,
    min_progress_distance: float,
    min_heading_change_degrees: float,
) -> dict[str, Any]:
    """Return progress diagnostics across delayed post-stop telemetry samples."""
    sample_diagnostics = []
    telemetry_latency_seconds = None
    for sample in samples:
        telemetry = cast(dict[str, Any], sample["telemetry"])
        path_progress = _manual_velocity_path_progress_diagnostic(
            before,
            telemetry,
            decision,
            min_progress_distance=min_progress_distance,
            min_heading_change_degrees=min_heading_change_degrees,
        )
        measured_delta = _telemetry_position_delta(before, telemetry)
        sample_diagnostics.append(
            {
                "delay_seconds": sample["delay_seconds"],
                "path_progress_diagnostic": path_progress,
                "measured_delta": measured_delta,
            }
        )
        if telemetry_latency_seconds is None and path_progress["passed"]:
            telemetry_latency_seconds = sample["delay_seconds"]

    final_sample = sample_diagnostics[-1] if sample_diagnostics else None
    return {
        "late_telemetry_check": True,
        "late_progress_detected": telemetry_latency_seconds is not None,
        "telemetry_latency_seconds": telemetry_latency_seconds,
        "late_path_progress_diagnostic": (
            final_sample["path_progress_diagnostic"] if final_sample else None
        ),
        "late_measured_delta": final_sample["measured_delta"] if final_sample else None,
        "post_stop_sample_diagnostics": sample_diagnostics,
    }


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
        _is_manual_motion_area_label(position.get("pos_type_label"))
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


def _manual_velocity_path_progress_diagnostic(
    before: dict[str, Any],
    after: dict[str, Any],
    decision: dict[str, Any],
    *,
    min_progress_distance: float,
    min_heading_change_degrees: float,
) -> dict[str, Any]:
    """Classify whether telemetry moved in the direction the controller intended."""
    action = decision.get("action")
    delta = _telemetry_position_delta(before, after)
    heading_change = delta.get("heading_change_degrees")
    heading_progress = (
        heading_change is not None
        and abs(float(heading_change)) >= min_heading_change_degrees
    )
    if action in {"turn_left", "turn_right"}:
        return {
            "status": (
                "heading_progress"
                if heading_progress
                else "heading_progress_not_detected"
            ),
            "passed": heading_progress,
            "action": action,
            "path_progress_distance": None,
            "expected_target_heading_degrees": None,
            "movement_vector_heading_degrees": None,
            "heading_progress": heading_progress,
            "min_progress_distance": min_progress_distance,
            "min_heading_change_degrees": min_heading_change_degrees,
        }

    target = decision.get("target")
    current = before.get("position", {})
    if (
        action != "forward"
        or not isinstance(target, dict)
        or current.get("x") is None
        or current.get("y") is None
        or delta.get("dx") is None
        or delta.get("dy") is None
    ):
        return {
            "status": "path_progress_unavailable",
            "passed": False,
            "action": action,
            "path_progress_distance": None,
            "expected_target_heading_degrees": None,
            "movement_vector_heading_degrees": None,
            "heading_progress": heading_progress,
            "min_progress_distance": min_progress_distance,
            "min_heading_change_degrees": min_heading_change_degrees,
        }

    target_dx = float(target["x"]) - float(current["x"])
    target_dy = float(target["y"]) - float(current["y"])
    target_distance = math.hypot(target_dx, target_dy)
    if target_distance <= 0:
        path_progress_distance = 0.0
        target_heading = None
    else:
        unit_x = target_dx / target_distance
        unit_y = target_dy / target_distance
        path_progress_distance = float(delta["dx"]) * unit_x + float(delta["dy"]) * unit_y
        target_heading = (
            math.degrees(math.atan2(target_dy, target_dx)) + 360
        ) % 360

    movement_vector_heading = None
    if delta.get("distance") is not None and float(delta["distance"]) > 0:
        movement_vector_heading = (
            math.degrees(math.atan2(float(delta["dy"]), float(delta["dx"]))) + 360
        ) % 360

    passed = path_progress_distance >= min_progress_distance
    if passed:
        status = "path_progress"
    elif path_progress_distance > 0:
        status = "path_progress_below_threshold"
    else:
        status = "no_path_progress"
    return {
        "status": status,
        "passed": passed,
        "action": action,
        "path_progress_distance": path_progress_distance,
        "expected_target_heading_degrees": target_heading,
        "movement_vector_heading_degrees": movement_vector_heading,
        "heading_progress": heading_progress,
        "min_progress_distance": min_progress_distance,
        "min_heading_change_degrees": min_heading_change_degrees,
    }


def _manual_velocity_completion_status(
    path_points: list[dict[str, float]],
    telemetry: dict[str, Any],
    *,
    waypoint_tolerance: float,
) -> dict[str, Any]:
    """Return whether the path target is currently complete."""
    position = telemetry.get("position", {})
    current_x = position.get("x")
    current_y = position.get("y")
    if current_x is None or current_y is None:
        return {
            "complete": False,
            "target_index": None,
            "distance_to_target": None,
            "reason": "live_position_unavailable",
        }
    current = {"x": float(current_x), "y": float(current_y)}
    target_index, _target, distance_to_target, waypoint_distances = (
        _manual_velocity_next_waypoint(
            path_points,
            current,
            waypoint_tolerance=waypoint_tolerance,
        )
    )
    return {
        "complete": target_index is None,
        "target_index": target_index,
        "distance_to_target": distance_to_target,
        "waypoint_distances": waypoint_distances,
        "reason": "path_complete" if target_index is None else "target_remaining",
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
    if not _is_manual_motion_area_label(current_position.get("pos_type_label")):
        reasons.append("pos_type_not_valid_manual_motion_area")
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


def _active_transport_label(
    coordinator: MammotionReportUpdateCoordinator,
) -> str | None:
    """Return the mower's normalized active transport label (e.g. 'ble').

    Reuses the coordinator's ``active_transport_state`` property so this always
    agrees with what ``export_runtime_state`` reports. Note that
    ``str(TransportType.BLE)`` is ``'TransportType.BLE'`` (not ``'ble'``), so the
    enum must be normalized rather than stringified directly.
    """
    try:
        return coordinator.active_transport_state
    except Exception:  # noqa: BLE001
        return None


def _transport_is_ble(coordinator: MammotionReportUpdateCoordinator) -> bool:
    """Return True when the mower's active transport is BLE.

    Real guarded closed-loop motion requires BLE: cloud/Wi-Fi telemetry lags by
    minutes, so the guard would be driving partly blind. Dry-runs are exempt.
    """
    label = _active_transport_label(coordinator)
    return label is not None and label.lower() == "ble"


def _manual_velocity_pulse_gates(
    coordinator: MammotionReportUpdateCoordinator,
    before: dict[str, Any],
    *,
    dry_run: bool,
    confirm_blades_off: bool,
    confirm_clear_area: bool,
) -> list[dict[str, Any]]:
    """Return safety gates for a manual velocity pulse probe."""
    work_mode_label = before.get("work_mode_label")
    return [
        {
            "name": "stop_primitive_available",
            "passed": hasattr(coordinator, "async_stop_manual_motion"),
            "detail": "Coordinator must expose async_stop_manual_motion().",
        },
        {
            "name": "ble_transport_required",
            "passed": dry_run or _transport_is_ble(coordinator),
            "detail": (
                "Real closed-loop motion requires the BLE transport for "
                "responsive telemetry; cloud/Wi-Fi lag is unsafe for guarded "
                "path execution."
            ),
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
            "passed": dry_run or work_mode_label in {"MODE_READY", "MODE_PAUSE"},
            "detail": (
                "Real pulse requires the mower to be ready/paused, not mowing "
                "or charging."
            ),
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
            "detail": (
                "Real pulse requires AREA_INSIDE, TURN_AREA_INSIDE, or "
                "CHANNEL_AREA_OVERLAP and a nonzero known zone hash."
            ),
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
    stop_mode: str = "immediate",
    stop_delay_ms: int = 0,
    post_command_sample_delays: list[float] | tuple[float, ...] | None = None,
    use_wifi: bool = DEFAULT_EXPERIMENTAL_SEGMENT_USE_WIFI,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    followup_samples: int = 4,
    followup_interval_seconds: float = 0.5,
) -> dict[str, Any]:
    """Run or simulate one tiny manual-velocity pulse with telemetry sampling."""
    if post_command_sample_delays is None:
        post_command_sample_delays = tuple(
            followup_interval_seconds * (index + 1)
            for index in range(followup_samples)
        )
    if hasattr(coordinator, "async_start_report_stream"):
        stream_duration_ms = int((max(post_command_sample_delays, default=0.0) + 10) * 1000)
        await coordinator.async_start_report_stream(duration_ms=max(10_000, stream_duration_ms))

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
        "stop_mode": stop_mode,
        "stop_delay_ms": stop_delay_ms,
        "post_command_sample_delays": list(post_command_sample_delays),
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

    result["command_result"] = await _manual_velocity_command_attempt(
        coordinator,
        action=action,
        speed=speed,
        use_wifi=use_wifi,
    )
    command_ok = result["command_result"]["ok"] is True
    await asyncio.sleep(duration_ms / 1000)
    after_command = _custom_path_telemetry_snapshot(coordinator)
    result["samples"].append({"label": "after_command_window", "telemetry": after_command})
    if stop_mode == "delayed" and stop_delay_ms > 0:
        await asyncio.sleep(stop_delay_ms / 1000)
    if stop_mode in {"immediate", "delayed"}:
        result["stop_result"] = await _manual_velocity_stop_attempt(
            coordinator,
            use_wifi=use_wifi,
        )
        after_stop = _custom_path_telemetry_snapshot(coordinator)
        result["samples"].append({"label": "after_stop", "telemetry": after_stop})
    else:
        result["stop_result"] = {
            "attempted": False,
            "ok": None,
            "error": None,
            "reason": "firmware_nudge_mode_no_explicit_stop",
        }
        after_stop = after_command

    previous_delay = 0.0
    for index, delay in enumerate(post_command_sample_delays):
        await asyncio.sleep(max(0.0, delay - previous_delay))
        previous_delay = delay
        result["samples"].append(
            {
                "label": f"post_command_{index + 1}_{delay:g}s",
                "telemetry": _custom_path_telemetry_snapshot(coordinator),
            }
        )
    final_telemetry = result["samples"][-1]["telemetry"]
    result["measured_delta"] = _telemetry_position_delta(before, final_telemetry)
    result["immediate_delta"] = _telemetry_position_delta(before, after_stop)
    stop_ok = result["stop_result"]["ok"] is True or stop_mode == "firmware"
    result["real_pulse_completed"] = command_ok and stop_ok
    return result


def _raw_pymammotion_command_args(
    command: str,
    *,
    linear_speed: int,
    angular_speed: int,
    speed: float,
) -> dict[str, Any]:
    """Return pymammotion command kwargs for a raw motion probe."""
    if command == "send_movement":
        return {"linear_speed": int(linear_speed), "angular_speed": int(angular_speed)}
    if command in {"move_forward", "move_back"}:
        return {"linear": float(speed)}
    if command in {"move_left", "move_right"}:
        return {"angular": float(speed)}
    raise ValueError(f"unsupported raw pymammotion command: {command}")


def _raw_pymammotion_motion_interpretation(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    min_translation_distance: float = 0.003,
    min_heading_change_degrees: float = 1.0,
) -> dict[str, Any]:
    """Return a compact interpretation of raw command movement telemetry."""
    delta = _telemetry_position_delta(before, after)
    distance = delta.get("distance")
    heading_change = delta.get("heading_change_degrees")
    translated = distance is not None and distance >= min_translation_distance
    rotated = (
        heading_change is not None
        and abs(float(heading_change)) >= min_heading_change_degrees
    )
    status = "no_motion_detected"
    if translated and rotated:
        status = "translation_and_heading_change"
    elif translated:
        status = "translation_detected"
    elif rotated:
        status = "heading_change_detected"
    return {
        "status": status,
        "translation_detected": translated,
        "heading_change_detected": rotated,
        "movement_heading_degrees": (
            _path_heading_degrees(
                {
                    "x": float(before["position"]["x"]),
                    "y": float(before["position"]["y"]),
                },
                {
                    "x": float(after["position"]["x"]),
                    "y": float(after["position"]["y"]),
                },
            )
            if translated
            else None
        ),
        "delta": delta,
        "min_translation_distance": min_translation_distance,
        "min_heading_change_degrees": min_heading_change_degrees,
    }


async def _send_manager_command_with_args(
    coordinator: MammotionReportUpdateCoordinator,
    command: str,
    *,
    prefer_ble: bool,
    command_kwargs: Mapping[str, Any],
) -> None:
    """Send a raw manager command with dynamic kwargs."""
    await cast(Any, coordinator.manager.send_command_with_args)(
        coordinator.device_name,
        command,
        prefer_ble=prefer_ble,
        **dict(command_kwargs),
    )


async def _raw_pymammotion_motion_probe(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    command: str = "send_movement",
    linear_speed: int = 400,
    angular_speed: int = 0,
    speed: float = 0.4,
    prefer_ble: bool = True,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60),
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
) -> dict[str, Any]:
    """Run or simulate one raw pymammotion movement command with telemetry."""
    before = _custom_path_telemetry_snapshot(coordinator)
    gates = _manual_velocity_pulse_gates(
        coordinator,
        before,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    command_args = _raw_pymammotion_command_args(
        command,
        linear_speed=linear_speed,
        angular_speed=angular_speed,
        speed=speed,
    )
    result: dict[str, Any] = {
        "service": SERVICE_RAW_PYMAMMOTION_MOTION_PROBE,
        "mode": "dry_run" if dry_run else "real_raw_pymammotion_probe",
        "dry_run": dry_run,
        "command": command,
        "command_args": command_args,
        "prefer_ble": prefer_ble,
        "transport_preference": "ble_preferred" if prefer_ble else "default",
        "sample_delays": list(sample_delays),
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "would_send": not dry_run and not blockers,
        "real_probe_allowed": not dry_run and not blockers,
        "blockers": blockers,
        "safety_gates": gates,
        "samples": [{"label": "before", "telemetry": before}],
        "command_result": {
            "attempted": False,
            "ok": None,
            "ack": None,
            "error": None,
            "duration_ms": None,
        },
        "motion_interpretation": _raw_pymammotion_motion_interpretation(
            before,
            before,
        ),
    }
    if dry_run or blockers:
        result["reason"] = "dry_run" if dry_run else "safety_gates_failed"
        result["command_not_sent"] = {
            "manager_method": "send_command_with_args",
            "device_name": getattr(coordinator, "device_name", None),
            "command": command,
            "prefer_ble": prefer_ble,
            "kwargs": command_args,
        }
        return result

    started = time.monotonic()
    result["command_result"]["attempted"] = True
    try:
        await _send_manager_command_with_args(
            coordinator,
            command,
            prefer_ble=prefer_ble,
            command_kwargs=command_args,
        )
        result["command_result"]["ack"] = None
        result["command_result"]["ok"] = True
    except Exception as err:  # noqa: BLE001
        result["command_result"]["ok"] = False
        result["command_result"]["error"] = f"{type(err).__name__}: {err}"
        result["reason"] = "command_failed"
        return result
    finally:
        result["command_result"]["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )

    previous_delay = 0.0
    for index, delay in enumerate(sample_delays):
        await asyncio.sleep(max(0.0, float(delay) - previous_delay))
        previous_delay = float(delay)
        result["samples"].append(
            {
                "label": f"sample_{index + 1}_{delay:g}s",
                "telemetry": _custom_path_telemetry_snapshot(coordinator),
            }
        )
    final_telemetry = result["samples"][-1]["telemetry"]
    result["motion_interpretation"] = _raw_pymammotion_motion_interpretation(
        before,
        final_telemetry,
    )
    result["final_telemetry"] = final_telemetry
    result["reason"] = "completed"
    return result


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _position_change_detected(
    before: dict[str, Any],
    current: dict[str, Any],
    *,
    min_position_change_distance: float,
) -> bool:
    """Return true when telemetry position changed by the configured threshold."""
    delta = _telemetry_position_delta(before, current)
    distance = delta.get("distance")
    return distance is not None and float(distance) >= min_position_change_distance


_RAW_POSITION_PATHS = (
    "mowing_state.pos_x",
    "mowing_state.pos_y",
    "mowing_state.toward",
    "mowing_state.pos_level",
    "mowing_state.rtk_status",
    "mowing_state.zone_hash",
    "mowing_state.pos_type",
    "location.orientation",
    "location.position_type",
    "location.work_zone",
    "location.RTK.latitude",
    "location.RTK.longitude",
    "location.RTK.yaw",
    "location.device.latitude",
    "location.device.longitude",
    "report_data.work.path_pos_x",
    "report_data.work.path_pos_y",
    "report_data.work.area",
    "report_data.work.progress",
    "report_data.work.nav_heading_state.heading_state",
    "report_data.vision_info.heading",
    "report_data.vision_info.vio_state",
    "report_data.dev.sys_status",
    "report_data.dev.charge_state",
    "report_data.dev.blade_state",
    "report_data.rtk.status",
    "report_data.rtk.pos_level",
)


def _position_feedback_raw_sources(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Return compact raw position/status fields that may move independently."""
    data = coordinator.data
    sources: dict[str, Any] = {
        "paths": {
            path: _enum_value(_safe_attr_path(data, path))
            for path in _RAW_POSITION_PATHS
        },
        "report_data.locations": [],
        "handle": {},
    }
    for index, location in enumerate(_safe_attr_path(data, "report_data.locations") or []):
        if index >= 3:
            break
        bol_hash = _safe_attr_path(location, "bol_hash")
        try:
            safe_bol_hash: Any = _json_safe_int(int(bol_hash or 0))
        except (TypeError, ValueError):
            safe_bol_hash = str(bol_hash) if bol_hash is not None else None
        sources["report_data.locations"].append(
            {
                "index": index,
                "real_pos_x": _safe_attr_path(location, "real_pos_x"),
                "real_pos_y": _safe_attr_path(location, "real_pos_y"),
                "real_toward": _safe_attr_path(location, "real_toward"),
                "pos_type": _enum_value(_safe_attr_path(location, "pos_type")),
                "bol_hash": safe_bol_hash,
            }
        )
    try:
        handle = coordinator.manager.mower(coordinator.device_name)
    except Exception:  # noqa: BLE001
        handle = None
    if handle is not None:
        active_transport = None
        if hasattr(handle, "active_transport"):
            try:
                active_transport = str(handle.active_transport())
            except Exception as err:  # noqa: BLE001
                active_transport = f"{type(err).__name__}: {err}"
        sources["handle"] = {
            "last_report_at": _safe_attr_path(handle, "last_report_at"),
            "availability": _stringify_large_ints(
                {
                    "mqtt_reported_offline": _safe_attr_path(
                        handle,
                        "availability.mqtt_reported_offline",
                    ),
                    "ble_in_cooldown": _safe_attr_path(
                        handle,
                        "availability.ble_in_cooldown",
                    ),
                }
            ),
            "active_transport": active_transport,
        }
    return _stringify_large_ints(sources)


def _position_feedback_snapshot(
    coordinator: MammotionReportUpdateCoordinator,
    label: str,
    initial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture normalized and raw position diagnostics."""
    telemetry = _custom_path_telemetry_snapshot(coordinator)
    snapshot = {
        "label": label,
        "captured_at": _utc_timestamp(),
        "telemetry": telemetry,
        "raw_sources": _position_feedback_raw_sources(coordinator),
    }
    if initial is not None:
        snapshot["delta_from_initial"] = _telemetry_position_delta(initial, telemetry)
    return snapshot


def _position_feedback_changed_sources(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    """Return high-level position source groups whose captured value changed."""
    changed = [
        f"telemetry.{key}"
        for key in ("position", "position_candidates")
        if before.get("telemetry", {}).get(key) != after.get("telemetry", {}).get(key)
    ]
    before_raw = before.get("raw_sources", {})
    after_raw = after.get("raw_sources", {})
    changed.extend(
        f"raw_sources.{key}"
        for key in ("paths", "report_data.locations", "handle")
        if before_raw.get(key) != after_raw.get(key)
    )
    return changed


async def _forward_two_pulse_latency_test(  # noqa: C901, PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    *,
    linear_speed: int = 200,
    pulse_count: int = 2,
    pulse_gap_seconds: float = 5.0,
    telemetry_timeout_seconds: float = 60.0,
    telemetry_sample_interval_seconds: float = 1.0,
    min_position_change_distance: float = 0.003,
    prefer_ble: bool = True,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run or simulate two forward raw movement pulses and measure telemetry latency."""
    before = _custom_path_telemetry_snapshot(coordinator)
    gates = _manual_velocity_pulse_gates(
        coordinator,
        before,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    runtime_safety = _runtime_motion_safety_summary(
        before,
        ha_state=ha_state,
        active_route=active_route,
    )
    if runtime_safety["active_mowing_detected"]:
        gates.append(
            {
                "name": "runtime_not_mowing",
                "passed": False,
                "detail": "Forward latency test is blocked while mowing is active.",
            }
        )
    if runtime_safety["active_route_status"]["blocks_motion"]:
        gates.append(
            {
                "name": "runtime_route_not_blocking",
                "passed": False,
                "detail": "Forward latency test is blocked by live/ambiguous route data.",
            }
        )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    command_args = {"linear_speed": int(linear_speed), "angular_speed": 0}
    command_not_sent = {
        "manager_method": "send_command_with_args",
        "device_name": getattr(coordinator, "device_name", None),
        "command": "send_movement",
        "prefer_ble": prefer_ble,
        "kwargs": command_args,
    }
    result: dict[str, Any] = {
        "service": SERVICE_FORWARD_TWO_PULSE_LATENCY_TEST,
        "mode": "dry_run" if dry_run else "real_forward_two_pulse_latency_test",
        "dry_run": dry_run,
        "would_send": not dry_run and not blockers,
        "real_test_allowed": not dry_run and not blockers,
        "linear_speed": linear_speed,
        "pulse_count": pulse_count,
        "pulse_gap_seconds": pulse_gap_seconds,
        "telemetry_timeout_seconds": telemetry_timeout_seconds,
        "telemetry_sample_interval_seconds": telemetry_sample_interval_seconds,
        "min_position_change_distance": min_position_change_distance,
        "prefer_ble": prefer_ble,
        "transport_preference": "ble_preferred" if prefer_ble else "default",
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "safety_gates": gates,
        "runtime_safety": runtime_safety,
        "blockers": blockers,
        "commands": [],
        "command_not_sent": command_not_sent if dry_run or blockers else None,
        "telemetry": {
            "before": before,
            "samples": [],
            "first_position_change_at": None,
            "first_position_change_after_command_1_seconds": None,
            "first_position_change_after_command_2_seconds": None,
            "first_position_change_after_final_command_seconds": None,
            "final_delta": _telemetry_position_delta(before, before),
        },
        "operator_observation": {
            "visual_time": None,
            "note": "User reports visual movement time manually after the run.",
        },
        "reason": None,
    }

    if dry_run or blockers:
        result["reason"] = "dry_run" if dry_run else "safety_gates_failed"
        result["commands"] = [
            {
                "index": index,
                "planned": True,
                "sent_at": None,
                "monotonic_seconds": None,
                "ack": None,
                "ok": None,
                "error": None,
                "duration_ms": None,
                "command": "send_movement",
                "kwargs": command_args,
                "planned_after_gap_seconds": (
                    None if index == 1 else pulse_gap_seconds
                ),
            }
            for index in range(1, pulse_count + 1)
        ]
        return result

    async def send_pulse(index: int) -> dict[str, Any]:
        command_started = time.monotonic()
        command_result: dict[str, Any] = {
            "index": index,
            "planned": True,
            "sent_at": _utc_timestamp(),
            "monotonic_seconds": command_started,
            "ack": None,
            "ok": None,
            "error": None,
            "duration_ms": None,
            "command": "send_movement",
            "kwargs": command_args,
        }
        try:
            await _send_manager_command_with_args(
                coordinator,
                "send_movement",
                prefer_ble=prefer_ble,
                command_kwargs=command_args,
            )
            command_result["ack"] = None
            command_result["ok"] = True
        except Exception as err:  # noqa: BLE001
            command_result["ok"] = False
            command_result["error"] = f"{type(err).__name__}: {err}"
        finally:
            command_result["duration_ms"] = round(
                (time.monotonic() - command_started) * 1000,
                3,
            )
        return command_result

    for command_index in range(1, pulse_count + 1):
        if command_index > 1:
            await asyncio.sleep(pulse_gap_seconds)
        command_result = await send_pulse(command_index)
        command_result["planned_after_gap_seconds"] = (
            None if command_index == 1 else pulse_gap_seconds
        )
        result["commands"].append(command_result)
        if command_result["ok"] is not True:
            result["reason"] = f"command_{command_index}_failed"
            return result

    command_1_monotonic = float(result["commands"][0]["monotonic_seconds"])
    command_2_monotonic = float(result["commands"][1]["monotonic_seconds"])
    final_command_monotonic = float(result["commands"][-1]["monotonic_seconds"])
    deadline = final_command_monotonic + telemetry_timeout_seconds
    first_change_sample: dict[str, Any] | None = None
    sample_index = 0
    while time.monotonic() <= deadline:
        now = time.monotonic()
        telemetry = _custom_path_telemetry_snapshot(coordinator)
        delta = _telemetry_position_delta(before, telemetry)
        changed = _position_change_detected(
            before,
            telemetry,
            min_position_change_distance=min_position_change_distance,
        )
        sample = {
            "index": sample_index + 1,
            "sampled_at": _utc_timestamp(),
            "seconds_after_command_1": round(now - command_1_monotonic, 3),
            "seconds_after_command_2": round(now - command_2_monotonic, 3),
            "seconds_after_final_command": round(now - final_command_monotonic, 3),
            "position_change_detected": changed,
            "delta": delta,
            "telemetry": telemetry,
        }
        result["telemetry"]["samples"].append(sample)
        if changed:
            first_change_sample = sample
            break
        sample_index += 1
        await asyncio.sleep(telemetry_sample_interval_seconds)

    final_telemetry = (
        result["telemetry"]["samples"][-1]["telemetry"]
        if result["telemetry"]["samples"]
        else _custom_path_telemetry_snapshot(coordinator)
    )
    result["telemetry"]["final_delta"] = _telemetry_position_delta(
        before,
        final_telemetry,
    )
    result["telemetry"]["final"] = final_telemetry
    if first_change_sample is not None:
        result["telemetry"]["first_position_change_at"] = first_change_sample[
            "sampled_at"
        ]
        result["telemetry"]["first_position_change_after_command_1_seconds"] = (
            first_change_sample["seconds_after_command_1"]
        )
        result["telemetry"]["first_position_change_after_command_2_seconds"] = (
            first_change_sample["seconds_after_command_2"]
        )
        result["telemetry"]["first_position_change_after_final_command_seconds"] = (
            first_change_sample["seconds_after_final_command"]
        )
        result["reason"] = "telemetry_position_change_detected"
    else:
        result["reason"] = "telemetry_position_change_timeout"
    return result


async def _position_feedback_refresh_attempt(  # noqa: C901
    coordinator: MammotionReportUpdateCoordinator,
    name: str,
    *,
    refresh_wait_seconds: float,
) -> dict[str, Any]:
    """Run one safe refresh/status path and report its result."""
    attempt: dict[str, Any] = {
        "name": name,
        "attempted_at": _utc_timestamp(),
        "ok": None,
        "error": None,
        "duration_ms": None,
    }
    started = time.monotonic()
    try:
        if name == "request_report_snapshot":
            await coordinator.async_request_report_snapshot()
        elif name == "request_reports_count_5":
            await coordinator.async_get_reports(count=5)
        elif name == "start_report_stream":
            await coordinator.async_start_report_stream(duration_ms=60_000)
        elif name == "request_iot_sync_one_shot":
            await coordinator.manager.request_iot_sync(coordinator.device_name)
        elif name == "request_iot_sync_continuous_window":
            await coordinator.manager.request_iot_sync_continuous(
                coordinator.device_name,
                period=1000,
                no_change_period=4000,
            )
            if refresh_wait_seconds > 0:
                await asyncio.sleep(refresh_wait_seconds)
            await coordinator.manager.request_iot_sync_continuous_stop(
                coordinator.device_name,
            )
            attempt["ok"] = True
            return attempt
        elif name == "ensure_fresh_state_forced":
            await coordinator.manager.ensure_fresh_state(
                coordinator.device_name,
                max_age_s=0.0,
            )
        elif name == "ble_sync_type_3":
            await coordinator.async_send_command(
                "send_todev_ble_sync",
                prefer_ble=True,
                sync_type=3,
            )
        elif name == "ha_request_refresh":
            await coordinator.async_request_refresh()
        else:
            raise ValueError(f"unknown refresh attempt: {name}")
        if refresh_wait_seconds > 0:
            await asyncio.sleep(refresh_wait_seconds)
        attempt["ok"] = True
    except Exception as err:  # noqa: BLE001
        attempt["ok"] = False
        attempt["error"] = f"{type(err).__name__}: {err}"
    finally:
        attempt["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
    return attempt


async def _position_feedback_diagnostic(  # noqa: C901, PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    *,
    linear_speed: int = 200,
    pulse_count: int = 0,
    pulse_gap_seconds: float = 5.0,
    refresh_wait_seconds: float = 2.0,
    prefer_ble: bool = True,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dump and refresh possible position sources around optional raw movement."""
    initial = _position_feedback_snapshot(coordinator, "initial")
    initial_telemetry = initial["telemetry"]
    gates = _manual_velocity_pulse_gates(
        coordinator,
        initial_telemetry,
        dry_run=dry_run or pulse_count == 0,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    runtime_safety = _runtime_motion_safety_summary(
        initial_telemetry,
        ha_state=ha_state,
        active_route=active_route,
    )
    if runtime_safety["active_mowing_detected"]:
        gates.append(
            {
                "name": "runtime_not_mowing",
                "passed": False,
                "detail": "Position feedback diagnostic is blocked while mowing is active.",
            }
        )
    if runtime_safety["active_route_status"]["blocks_motion"]:
        gates.append(
            {
                "name": "runtime_route_not_blocking",
                "passed": False,
                "detail": "Position feedback diagnostic is blocked by live/ambiguous route data.",
            }
        )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    command_args = {"linear_speed": int(linear_speed), "angular_speed": 0}
    result: dict[str, Any] = {
        "service": SERVICE_POSITION_FEEDBACK_DIAGNOSTIC,
        "mode": "dry_run" if dry_run else "real_position_feedback_diagnostic",
        "dry_run": dry_run,
        "would_send": not dry_run and pulse_count > 0 and not blockers,
        "linear_speed": linear_speed,
        "pulse_count": pulse_count,
        "pulse_gap_seconds": pulse_gap_seconds,
        "refresh_wait_seconds": refresh_wait_seconds,
        "prefer_ble": prefer_ble,
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "safety_gates": gates,
        "runtime_safety": runtime_safety,
        "blockers": blockers,
        "commands": [],
        "refresh_attempts": [],
        "snapshots": [initial],
        "changed_sources": [],
        "position_source_changed": False,
        "reason": None,
    }
    if dry_run or blockers:
        result["reason"] = "dry_run" if dry_run else "safety_gates_failed"
        result["commands"] = [
            {
                "index": index,
                "planned": True,
                "sent_at": None,
                "ok": None,
                "command": "send_movement",
                "kwargs": command_args,
            }
            for index in range(1, pulse_count + 1)
        ]
        return result

    for command_index in range(1, pulse_count + 1):
        if command_index > 1:
            await asyncio.sleep(pulse_gap_seconds)
        started = time.monotonic()
        command_result: dict[str, Any] = {
            "index": command_index,
            "sent_at": _utc_timestamp(),
            "ok": None,
            "ack": None,
            "error": None,
            "duration_ms": None,
            "command": "send_movement",
            "kwargs": command_args,
        }
        try:
            await _send_manager_command_with_args(
                coordinator,
                "send_movement",
                prefer_ble=prefer_ble,
                command_kwargs=command_args,
            )
            command_result["ack"] = None
            command_result["ok"] = True
        except Exception as err:  # noqa: BLE001
            command_result["ok"] = False
            command_result["error"] = f"{type(err).__name__}: {err}"
        finally:
            command_result["duration_ms"] = round(
                (time.monotonic() - started) * 1000,
                3,
            )
        result["commands"].append(command_result)
        if command_result["ok"] is not True:
            result["reason"] = f"command_{command_index}_failed"
            return result

    result["snapshots"].append(
        _position_feedback_snapshot(
            coordinator,
            "after_commands_before_refresh",
            initial_telemetry,
        )
    )
    refresh_steps = (
        "request_report_snapshot",
        "request_reports_count_5",
        "start_report_stream",
        "request_iot_sync_one_shot",
        "request_iot_sync_continuous_window",
        "ensure_fresh_state_forced",
        "ble_sync_type_3",
        "ha_request_refresh",
    )
    for refresh_step in refresh_steps:
        attempt = await _position_feedback_refresh_attempt(
            coordinator,
            refresh_step,
            refresh_wait_seconds=refresh_wait_seconds,
        )
        result["refresh_attempts"].append(attempt)
        result["snapshots"].append(
            _position_feedback_snapshot(
                coordinator,
                f"after_{refresh_step}",
                initial_telemetry,
            )
        )
    changed: list[str] = []
    for snapshot in result["snapshots"][1:]:
        for source in _position_feedback_changed_sources(initial, snapshot):
            if source not in changed:
                changed.append(source)
    result["changed_sources"] = changed
    position_changed_sources = [
        source for source in changed if source != "raw_sources.handle"
    ]
    result["position_changed_sources"] = position_changed_sources
    result["metadata_changed_sources"] = [
        source for source in changed if source not in position_changed_sources
    ]
    result["position_source_changed"] = bool(position_changed_sources)
    result["reason"] = (
        "position_source_changed"
        if position_changed_sources
        else "metadata_source_changed"
        if changed
        else "position_source_unchanged"
    )
    return result


def _vio_sample_from_snapshot(
    snapshot: dict[str, Any],
    prev_telemetry: dict[str, Any],
    initial_telemetry: dict[str, Any],
) -> dict[str, Any]:
    """Extract VIO/heading/position fields plus motion deltas from one snapshot."""
    paths = snapshot.get("raw_sources", {}).get("paths", {})
    telemetry = snapshot.get("telemetry", {})
    position = telemetry.get("position", {}) or {}
    delta_prev = _telemetry_position_delta(prev_telemetry, telemetry)
    delta_init = snapshot.get("delta_from_initial") or _telemetry_position_delta(
        initial_telemetry, telemetry
    )
    dist_prev = delta_prev.get("distance")
    return {
        "label": snapshot.get("label"),
        "captured_at": snapshot.get("captured_at"),
        "vio_state": paths.get("report_data.vision_info.vio_state"),
        "vision_heading": paths.get("report_data.vision_info.heading"),
        "orientation": paths.get("location.orientation"),
        "rtk_yaw": paths.get("location.RTK.yaw"),
        "x": position.get("x"),
        "y": position.get("y"),
        "toward": position.get("toward"),
        "delta_from_prev_m": dist_prev,
        "delta_from_initial_m": delta_init.get("distance"),
        "moving": bool(dist_prev is not None and dist_prev > 0.01),
    }


async def _vio_motion_probe(  # noqa: C901, PLR0912, PLR0913, PLR0915
    coordinator: MammotionReportUpdateCoordinator,
    *,
    linear_speed: int = 200,
    drive_seconds: float = 6.0,
    sample_interval_seconds: float = 1.5,
    post_stop_samples: int = 3,
    max_displacement_m: float = 1.0,
    prefer_ble: bool = True,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Drive one bounded continuous forward motion while sampling VIO fields.

    Answers whether VIO (``report_data.vision_info.heading`` / ``vio_state``)
    initializes and produces a heading during motion, so it could serve as a
    rotation-feedback signal for turning. Bounded by time (``drive_seconds``),
    distance (``max_displacement_m``), and a mandatory explicit stop.
    """
    initial = _position_feedback_snapshot(coordinator, "initial")
    initial_telemetry = initial["telemetry"]
    gates = _manual_velocity_pulse_gates(
        coordinator,
        initial_telemetry,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    runtime_safety = _runtime_motion_safety_summary(
        initial_telemetry,
        ha_state=ha_state,
        active_route=active_route,
    )
    if runtime_safety["active_mowing_detected"]:
        gates.append(
            {
                "name": "runtime_not_mowing",
                "passed": False,
                "detail": "VIO motion probe is blocked while mowing is active.",
            }
        )
    if runtime_safety["active_route_status"]["blocks_motion"]:
        gates.append(
            {
                "name": "runtime_route_not_blocking",
                "passed": False,
                "detail": "VIO motion probe is blocked by live/ambiguous route data.",
            }
        )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    command_args = {"linear_speed": int(linear_speed), "angular_speed": 0}
    result: dict[str, Any] = {
        "service": SERVICE_VIO_MOTION_PROBE,
        "mode": "dry_run" if dry_run else "real_vio_motion_probe",
        "dry_run": dry_run,
        "would_send": not dry_run and not blockers,
        "linear_speed": linear_speed,
        "drive_seconds": drive_seconds,
        "sample_interval_seconds": sample_interval_seconds,
        "post_stop_samples": post_stop_samples,
        "max_displacement_m": max_displacement_m,
        "prefer_ble": prefer_ble,
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "active_transport": _active_transport_label(coordinator),
        "safety_gates": gates,
        "runtime_safety": runtime_safety,
        "blockers": blockers,
        "baseline": _vio_sample_from_snapshot(
            initial, initial_telemetry, initial_telemetry
        ),
        "command": {"command": "send_movement", "kwargs": command_args},
        "command_ok": None,
        "command_error": None,
        "stop_ack": None,
        "samples": [],
        "post_stop": [],
        "final_displacement_m": None,
        "verdict": {},
        "reason": None,
    }
    if dry_run or blockers:
        result["reason"] = "dry_run" if dry_run else "safety_gates_failed"
        return result

    # BLE pre-flight: refuse to fire into cloud even if the gate momentarily passed.
    if not _transport_is_ble(coordinator):
        result["reason"] = "ble_not_active_at_fire"
        return result

    samples: list[dict[str, Any]] = []
    aborted_reason: str | None = None
    prev_telemetry = initial_telemetry
    command_started = False
    try:
        await _send_manager_command_with_args(
            coordinator,
            "send_movement",
            prefer_ble=prefer_ble,
            command_kwargs=command_args,
        )
        command_started = True
        result["command_ok"] = True
        drive_start = time.monotonic()
        sample_index = 0
        while (time.monotonic() - drive_start) < drive_seconds:
            sample_index += 1
            try:
                await coordinator.async_get_reports(count=5)
            except Exception as err:  # noqa: BLE001
                LOGGER.debug("vio_motion_probe drive refresh failed: %s", err)
            snapshot = _position_feedback_snapshot(
                coordinator, f"drive_{sample_index}", initial_telemetry
            )
            sample = _vio_sample_from_snapshot(
                snapshot, prev_telemetry, initial_telemetry
            )
            sample["elapsed_seconds"] = round(time.monotonic() - drive_start, 3)
            samples.append(sample)
            prev_telemetry = snapshot["telemetry"]
            telemetry = snapshot["telemetry"]
            if not _blade_reported_safe(telemetry):
                aborted_reason = "aborted_unsafe_blade"
                break
            if telemetry.get("work_mode_label") not in {"MODE_READY", "MODE_PAUSE"}:
                aborted_reason = "aborted_unsafe_mode"
                break
            displacement = sample["delta_from_initial_m"]
            if displacement is not None and displacement > max_displacement_m:
                aborted_reason = "aborted_displacement_cap"
                break
            await asyncio.sleep(sample_interval_seconds)
    except Exception as err:  # noqa: BLE001
        aborted_reason = "command_failed"
        result["command_ok"] = command_started
        result["command_error"] = f"{type(err).__name__}: {err}"
    finally:
        if command_started:
            try:
                result["stop_ack"] = await coordinator.async_stop_manual_motion()
            except Exception as err:  # noqa: BLE001
                result["stop_ack"] = {"error": f"{type(err).__name__}: {err}"}

    post_stop: list[dict[str, Any]] = []
    for post_index in range(1, post_stop_samples + 1):
        await asyncio.sleep(sample_interval_seconds)
        try:
            await coordinator.async_get_reports(count=5)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("vio_motion_probe post-stop refresh failed: %s", err)
        snapshot = _position_feedback_snapshot(
            coordinator, f"post_stop_{post_index}", initial_telemetry
        )
        sample = _vio_sample_from_snapshot(snapshot, prev_telemetry, initial_telemetry)
        post_stop.append(sample)
        prev_telemetry = snapshot["telemetry"]

    result["samples"] = samples
    result["post_stop"] = post_stop

    def _vio_active(value: Any) -> bool:
        return value is not None and value != 0

    all_samples = samples + post_stop
    motion_confirmed = any(sample["moving"] for sample in samples)
    vio_activated_while_moving = any(
        _vio_active(sample["vio_state"]) and sample["moving"] for sample in samples
    )
    vio_activated_any = any(_vio_active(sample["vio_state"]) for sample in all_samples)
    heading_series = [
        sample["vision_heading"]
        for sample in samples
        if _vio_active(sample["vio_state"])
    ]
    final_displacement: float | None = None
    for sample in reversed(samples):
        if sample["delta_from_initial_m"] is not None:
            final_displacement = sample["delta_from_initial_m"]
            break
    result["final_displacement_m"] = final_displacement
    result["verdict"] = {
        "motion_confirmed": motion_confirmed,
        "vio_activated_while_moving": vio_activated_while_moving,
        "vio_activated_any": vio_activated_any,
        "heading_series": heading_series,
        "max_vio_state": max(
            (
                sample["vio_state"]
                for sample in all_samples
                if sample["vio_state"] is not None
            ),
            default=None,
        ),
    }
    if aborted_reason:
        result["reason"] = aborted_reason
    elif not motion_confirmed:
        result["reason"] = "no_motion_detected"
    elif vio_activated_while_moving:
        result["reason"] = "vio_initialized_during_motion"
    else:
        result["reason"] = "vio_never_initialized_despite_motion"
    return result


def _angle_series_change(values: list[Any]) -> dict[str, Any]:
    """Return net and cumulative absolute change (deg) over an angle series."""
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if len(nums) < 2:
        return {
            "net_degrees": None,
            "total_abs_degrees": None,
            "samples": len(nums),
        }
    net = _heading_error_degrees(nums[0], nums[-1])
    total = 0.0
    for prev, curr in zip(nums, nums[1:], strict=False):
        total += abs(_heading_error_degrees(prev, curr))
    return {
        "net_degrees": round(net, 3),
        "total_abs_degrees": round(total, 3),
        "samples": len(nums),
    }


async def _vio_turn_probe(  # noqa: C901, PLR0912, PLR0913, PLR0915
    coordinator: MammotionReportUpdateCoordinator,
    *,
    angular_speed: int = 180,
    linear_speed: int = 0,
    drive_seconds: float = 6.0,
    sample_interval_seconds: float = 1.5,
    post_stop_samples: int = 3,
    max_displacement_m: float = 0.5,
    min_heading_change_degrees: float = 3.0,
    prefer_ble: bool = True,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rotate in place while sampling VIO heading vs course-over-ground.

    Directly tests whether ``report_data.vision_info.heading`` tracks rotation:
    during an in-place pivot a true body heading changes while course-over-ground
    (``toward`` / ``orientation``) stays frozen. The operator must visually
    confirm the mower physically pivots; this probe only measures whether the
    heading signal follows. Bounded by time (``drive_seconds``), distance
    (``max_displacement_m`` — a pivot should barely translate), and a mandatory
    explicit stop.
    """
    initial = _position_feedback_snapshot(coordinator, "initial")
    initial_telemetry = initial["telemetry"]
    gates = _manual_velocity_pulse_gates(
        coordinator,
        initial_telemetry,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    runtime_safety = _runtime_motion_safety_summary(
        initial_telemetry,
        ha_state=ha_state,
        active_route=active_route,
    )
    if runtime_safety["active_mowing_detected"]:
        gates.append(
            {
                "name": "runtime_not_mowing",
                "passed": False,
                "detail": "VIO turn probe is blocked while mowing is active.",
            }
        )
    if runtime_safety["active_route_status"]["blocks_motion"]:
        gates.append(
            {
                "name": "runtime_route_not_blocking",
                "passed": False,
                "detail": "VIO turn probe is blocked by live/ambiguous route data.",
            }
        )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    command_args = {
        "linear_speed": int(linear_speed),
        "angular_speed": int(angular_speed),
    }
    baseline = _vio_sample_from_snapshot(initial, initial_telemetry, initial_telemetry)
    result: dict[str, Any] = {
        "service": SERVICE_VIO_TURN_PROBE,
        "mode": "dry_run" if dry_run else "real_vio_turn_probe",
        "dry_run": dry_run,
        "would_send": not dry_run and not blockers,
        "angular_speed": angular_speed,
        "linear_speed": linear_speed,
        "drive_seconds": drive_seconds,
        "sample_interval_seconds": sample_interval_seconds,
        "post_stop_samples": post_stop_samples,
        "max_displacement_m": max_displacement_m,
        "min_heading_change_degrees": min_heading_change_degrees,
        "prefer_ble": prefer_ble,
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "active_transport": _active_transport_label(coordinator),
        "operator_note": (
            "Visually confirm the mower physically pivots. This probe only "
            "measures whether vision_info.heading follows the rotation."
        ),
        "safety_gates": gates,
        "runtime_safety": runtime_safety,
        "blockers": blockers,
        "baseline": baseline,
        "command": {"command": "send_movement", "kwargs": command_args},
        "command_ok": None,
        "command_error": None,
        "stop_ack": None,
        "samples": [],
        "post_stop": [],
        "final_displacement_m": None,
        "verdict": {},
        "reason": None,
    }
    if dry_run or blockers:
        result["reason"] = "dry_run" if dry_run else "safety_gates_failed"
        return result

    # BLE pre-flight: refuse to fire into cloud even if the gate momentarily passed.
    if not _transport_is_ble(coordinator):
        result["reason"] = "ble_not_active_at_fire"
        return result

    samples: list[dict[str, Any]] = []
    aborted_reason: str | None = None
    prev_telemetry = initial_telemetry
    command_started = False
    try:
        await _send_manager_command_with_args(
            coordinator,
            "send_movement",
            prefer_ble=prefer_ble,
            command_kwargs=command_args,
        )
        command_started = True
        result["command_ok"] = True
        drive_start = time.monotonic()
        sample_index = 0
        while (time.monotonic() - drive_start) < drive_seconds:
            sample_index += 1
            try:
                await coordinator.async_get_reports(count=5)
            except Exception as err:  # noqa: BLE001
                LOGGER.debug("vio_turn_probe drive refresh failed: %s", err)
            snapshot = _position_feedback_snapshot(
                coordinator, f"turn_{sample_index}", initial_telemetry
            )
            sample = _vio_sample_from_snapshot(
                snapshot, prev_telemetry, initial_telemetry
            )
            sample["elapsed_seconds"] = round(time.monotonic() - drive_start, 3)
            samples.append(sample)
            prev_telemetry = snapshot["telemetry"]
            telemetry = snapshot["telemetry"]
            if not _blade_reported_safe(telemetry):
                aborted_reason = "aborted_unsafe_blade"
                break
            if telemetry.get("work_mode_label") not in {"MODE_READY", "MODE_PAUSE"}:
                aborted_reason = "aborted_unsafe_mode"
                break
            displacement = sample["delta_from_initial_m"]
            if displacement is not None and displacement > max_displacement_m:
                aborted_reason = "aborted_displacement_cap"
                break
            await asyncio.sleep(sample_interval_seconds)
    except Exception as err:  # noqa: BLE001
        aborted_reason = "command_failed"
        result["command_ok"] = command_started
        result["command_error"] = f"{type(err).__name__}: {err}"
    finally:
        if command_started:
            try:
                result["stop_ack"] = await coordinator.async_stop_manual_motion()
            except Exception as err:  # noqa: BLE001
                result["stop_ack"] = {"error": f"{type(err).__name__}: {err}"}

    post_stop: list[dict[str, Any]] = []
    for post_index in range(1, post_stop_samples + 1):
        await asyncio.sleep(sample_interval_seconds)
        try:
            await coordinator.async_get_reports(count=5)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("vio_turn_probe post-stop refresh failed: %s", err)
        snapshot = _position_feedback_snapshot(
            coordinator, f"post_stop_{post_index}", initial_telemetry
        )
        sample = _vio_sample_from_snapshot(snapshot, prev_telemetry, initial_telemetry)
        post_stop.append(sample)
        prev_telemetry = snapshot["telemetry"]

    result["samples"] = samples
    result["post_stop"] = post_stop

    heading_seq = [baseline["vision_heading"]] + [s["vision_heading"] for s in samples]
    cog_seq = [baseline["toward"]] + [s["toward"] for s in samples]
    vision_change = _angle_series_change(heading_seq)
    cog_change = _angle_series_change(cog_seq)
    vio_states = [baseline["vio_state"]] + [s["vio_state"] for s in samples]
    vio_active_throughout = bool(vio_states) and all(
        state is not None and state != 0 for state in vio_states
    )
    max_disp: float | None = None
    for sample in samples:
        disp = sample["delta_from_initial_m"]
        if disp is not None and (max_disp is None or disp > max_disp):
            max_disp = disp
    result["final_displacement_m"] = max_disp
    result["verdict"] = {
        "vision_heading_change": vision_change,
        "course_over_ground_change": cog_change,
        "vio_active_throughout": vio_active_throughout,
        "max_displacement_m": max_disp,
    }
    vision_total = vision_change["total_abs_degrees"]
    cog_total = cog_change["total_abs_degrees"] or 0.0
    if aborted_reason:
        result["reason"] = aborted_reason
    elif vision_total is None:
        result["reason"] = "no_vision_heading_data"
    elif vision_total >= min_heading_change_degrees:
        if cog_total < min_heading_change_degrees:
            result["reason"] = "vision_heading_tracks_rotation"
        else:
            result["reason"] = "vision_heading_and_cog_both_moved"
    else:
        result["reason"] = "vision_heading_static_during_command"
    return result


def _vio_reading(coordinator: MammotionReportUpdateCoordinator) -> dict[str, Any]:
    """Return the current VIO heading and state from live telemetry."""
    paths = _position_feedback_raw_sources(coordinator).get("paths", {})
    return {
        "vision_heading": paths.get("report_data.vision_info.heading"),
        "vio_state": paths.get("report_data.vision_info.vio_state"),
    }


async def _vio_turn_to_heading(  # noqa: C901, PLR0912, PLR0913, PLR0915
    coordinator: MammotionReportUpdateCoordinator,
    *,
    target_vision_heading: float,
    heading_tolerance_degrees: float = 8.0,
    angular_speed: int = 500,
    pulse_duration_ms: int = 1500,
    slow_pulse_duration_ms: int = 700,
    slow_threshold_degrees: float = 15.0,
    refresh_wait_seconds: float = 2.0,
    max_commands: int = 8,
    min_progress_degrees: float = 2.0,
    max_displacement_m: float = 0.5,
    invert_direction: bool = False,
    prefer_ble: bool = True,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Closed-loop turn to an absolute ``vision_info.heading`` via bounded pulses.

    Uses the VIO body heading (``report_data.vision_info.heading``) as feedback --
    proven 2026-07-10 (via ``vio_turn_probe``) to track rotation directionally.
    Calibration baked in: ``angular_speed`` must be strong (~500; 180 does not
    rotate this unit), and positive angular speed DECREASES ``vision_heading``
    while negative INCREASES it, so the loop turns the opposite sign of the
    heading error (flip with ``invert_direction`` if a session's convention
    differs). Because VIO refreshes ~1.5s into a command then latches, each
    iteration is a bounded pulse + explicit stop + ``request_reports`` refresh,
    then measures ``vision_heading`` and repeats until within tolerance.
    """
    initial_telemetry = _custom_path_telemetry_snapshot(coordinator)
    initial_reading = _vio_reading(coordinator)
    initial_heading = initial_reading["vision_heading"]
    target = float(target_vision_heading)
    gates = _manual_velocity_pulse_gates(
        coordinator,
        initial_telemetry,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    runtime_safety = _runtime_motion_safety_summary(
        initial_telemetry,
        ha_state=ha_state,
        active_route=active_route,
    )
    if initial_heading is None:
        gates.append(
            {
                "name": "vio_heading_available",
                "passed": dry_run,
                "detail": "VIO turn-to-heading requires live vision_info.heading.",
            }
        )
    if runtime_safety["active_mowing_detected"]:
        gates.append(
            {
                "name": "runtime_not_mowing",
                "passed": False,
                "detail": "VIO turn-to-heading is blocked while mowing is active.",
            }
        )
    if runtime_safety["active_route_status"]["blocks_motion"]:
        gates.append(
            {
                "name": "runtime_route_not_blocking",
                "passed": False,
                "detail": "VIO turn-to-heading is blocked by live/ambiguous route data.",
            }
        )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]

    def _planned_angular(error: float) -> int:
        # Observed: +angular decreases vision_heading, -angular increases it.
        base = -angular_speed if error > 0 else angular_speed
        return -base if invert_direction else base

    initial_error: float | None = None
    if initial_heading is not None:
        initial_error = _heading_error_degrees(float(initial_heading), target)
    planned_angular = _planned_angular(initial_error) if initial_error is not None else None
    result: dict[str, Any] = {
        "service": SERVICE_VIO_TURN_TO_HEADING,
        "mode": "dry_run" if dry_run else "real_vio_turn_to_heading",
        "dry_run": dry_run,
        "real_execution_scope": "vio_turn_to_heading_only",
        "path_execution_allowed": False,
        "target_vision_heading": target,
        "heading_tolerance_degrees": heading_tolerance_degrees,
        "angular_speed": angular_speed,
        "pulse_duration_ms": pulse_duration_ms,
        "slow_pulse_duration_ms": slow_pulse_duration_ms,
        "slow_threshold_degrees": slow_threshold_degrees,
        "max_commands": max_commands,
        "min_progress_degrees": min_progress_degrees,
        "max_displacement_m": max_displacement_m,
        "invert_direction": invert_direction,
        "prefer_ble": prefer_ble,
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "active_transport": _active_transport_label(coordinator),
        "initial_vision_heading": initial_heading,
        "initial_vio_state": initial_reading["vio_state"],
        "initial_heading_error_degrees": (
            round(initial_error, 3) if initial_error is not None else None
        ),
        "safety_gates": gates,
        "runtime_safety": runtime_safety,
        "blockers": blockers,
        "would_send": (
            not dry_run
            and not blockers
            and initial_error is not None
            and abs(initial_error) > heading_tolerance_degrees
        ),
        "planned_command": {
            "command": "send_movement",
            "kwargs": {"linear_speed": 0, "angular_speed": planned_angular},
        },
        "commands_sent": 0,
        "command_results": [],
        "final_vision_heading": initial_heading,
        "final_heading_error_degrees": (
            round(initial_error, 3) if initial_error is not None else None
        ),
        "stop_reason": None,
    }
    if initial_error is not None and abs(initial_error) <= heading_tolerance_degrees:
        result["stop_reason"] = "target_heading_reached"
        return result
    if dry_run:
        result["stop_reason"] = "dry_run"
        return result
    if blockers:
        result["stop_reason"] = "safety_gates_failed"
        return result
    if not _transport_is_ble(coordinator):
        result["stop_reason"] = "ble_not_active_at_fire"
        return result

    for command_index in range(1, max_commands + 1):
        before_telemetry = _custom_path_telemetry_snapshot(coordinator)
        before_heading = _vio_reading(coordinator)["vision_heading"]
        if before_heading is None:
            result["stop_reason"] = "vio_heading_unavailable"
            return result
        if not _blade_reported_safe(before_telemetry):
            result["stop_reason"] = "aborted_unsafe_blade"
            return result
        if before_telemetry.get("work_mode_label") not in {"MODE_READY", "MODE_PAUSE"}:
            result["stop_reason"] = "aborted_unsafe_mode"
            return result
        error = _heading_error_degrees(float(before_heading), target)
        if abs(error) <= heading_tolerance_degrees:
            result["final_vision_heading"] = before_heading
            result["final_heading_error_degrees"] = round(error, 3)
            result["stop_reason"] = "target_heading_reached"
            return result
        pulse_ms = (
            slow_pulse_duration_ms
            if abs(error) <= slow_threshold_degrees
            else pulse_duration_ms
        )
        angular = _planned_angular(error)
        command_result: dict[str, Any] = {
            "index": command_index,
            "angular_speed": angular,
            "pulse_duration_ms": pulse_ms,
            "before_vision_heading": before_heading,
            "heading_error_before": round(error, 3),
            "command": "send_movement",
            "ok": None,
            "error": None,
            "stop_ack": None,
            "after_vision_heading": None,
            "measured_change_degrees": None,
            "heading_error_after": None,
            "progress_degrees": None,
            "displacement_m": None,
        }
        try:
            await _send_manager_command_with_args(
                coordinator,
                "send_movement",
                prefer_ble=prefer_ble,
                command_kwargs={"linear_speed": 0, "angular_speed": angular},
            )
            command_result["ok"] = True
        except Exception as err:  # noqa: BLE001
            command_result["ok"] = False
            command_result["error"] = f"{type(err).__name__}: {err}"
            result["command_results"].append(command_result)
            result["commands_sent"] += 1
            result["stop_reason"] = "command_failed"
            return result
        result["commands_sent"] += 1
        # Bounded pulse, then a mandatory explicit stop before sampling.
        await asyncio.sleep(pulse_ms / 1000)
        try:
            command_result["stop_ack"] = await coordinator.async_stop_manual_motion()
        except Exception as err:  # noqa: BLE001
            command_result["stop_ack"] = {"error": f"{type(err).__name__}: {err}"}
        try:
            await coordinator.async_get_reports(count=5)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("vio_turn_to_heading refresh failed: %s", err)
        if refresh_wait_seconds > 0:
            await asyncio.sleep(refresh_wait_seconds)
        after_telemetry = _custom_path_telemetry_snapshot(coordinator)
        after_heading = _vio_reading(coordinator)["vision_heading"]
        command_result["after_vision_heading"] = after_heading
        if after_heading is None:
            result["command_results"].append(command_result)
            result["stop_reason"] = "vio_heading_unavailable"
            return result
        measured_change = _heading_error_degrees(float(before_heading), float(after_heading))
        new_error = _heading_error_degrees(float(after_heading), target)
        displacement = _telemetry_position_delta(
            initial_telemetry, after_telemetry
        ).get("distance")
        command_result["measured_change_degrees"] = round(measured_change, 3)
        command_result["heading_error_after"] = round(new_error, 3)
        command_result["progress_degrees"] = round(abs(error) - abs(new_error), 3)
        command_result["displacement_m"] = displacement
        result["command_results"].append(command_result)
        result["final_vision_heading"] = after_heading
        result["final_heading_error_degrees"] = round(new_error, 3)
        if displacement is not None and displacement > max_displacement_m:
            result["stop_reason"] = "aborted_displacement_cap"
            return result
        if not _blade_reported_safe(after_telemetry):
            result["stop_reason"] = "aborted_unsafe_blade"
            return result
        if abs(new_error) <= heading_tolerance_degrees:
            result["stop_reason"] = "target_heading_reached"
            return result
        if (abs(error) - abs(new_error)) < min_progress_degrees:
            result["stop_reason"] = "no_heading_progress"
            return result

    result["stop_reason"] = "max_commands_reached"
    return result


def _raw_segment_current_point(telemetry: dict[str, Any]) -> dict[str, float] | None:
    """Return the current map-local point from telemetry, if available."""
    position = telemetry.get("position", {})
    if not _position_available(telemetry):
        return None
    return {"x": float(position["x"]), "y": float(position["y"])}


def _raw_segment_lateral_diagnostic(
    current: dict[str, float] | None,
    target: dict[str, float],
) -> dict[str, Any]:
    """Return whether a target is compatible with calibrated Y-axis nudges."""
    if current is None:
        return {
            "passed": False,
            "reason": "position_unavailable",
            "dx": None,
            "dy": None,
            "lateral_limit": None,
        }
    dx = float(target["x"]) - float(current["x"])
    dy = float(target["y"]) - float(current["y"])
    lateral_limit = max(0.10, abs(dy) * 0.35)
    return {
        "passed": abs(dx) <= lateral_limit,
        "reason": (
            "mostly_y_axis_segment"
            if abs(dx) <= lateral_limit
            else "segment_requires_lateral_or_turning_motion"
        ),
        "dx": dx,
        "dy": dy,
        "abs_dx": abs(dx),
        "abs_dy": abs(dy),
        "lateral_limit": lateral_limit,
        "rule": "abs(dx) <= max(0.10, abs(dy) * 0.35)",
    }


def _raw_segment_command_selection(
    current: dict[str, float] | None,
    target: dict[str, float],
    *,
    linear_speed_fast: int,
    linear_speed_slow: int,
    slow_distance_threshold: float = 0.15,
) -> dict[str, Any]:
    """Return the calibrated raw send_movement command for a Y-axis segment."""
    if current is None:
        return {
            "command": "send_movement",
            "linear_speed": None,
            "angular_speed": 0,
            "remaining_y": None,
            "remaining_distance": None,
            "speed_tier": None,
            "reason": "position_unavailable",
        }
    remaining_y = float(target["y"]) - float(current["y"])
    remaining_x = float(target["x"]) - float(current["x"])
    remaining_distance = math.hypot(remaining_x, remaining_y)
    speed_magnitude = (
        int(linear_speed_slow)
        if abs(remaining_y) < slow_distance_threshold
        else int(linear_speed_fast)
    )
    linear_speed = speed_magnitude if remaining_y < 0 else -speed_magnitude
    return {
        "command": "send_movement",
        "linear_speed": linear_speed,
        "angular_speed": 0,
        "remaining_x": remaining_x,
        "remaining_y": remaining_y,
        "remaining_distance": remaining_distance,
        "speed_tier": "slow" if abs(remaining_y) < slow_distance_threshold else "fast",
        "negative_y_uses_positive_linear_speed": True,
        "positive_y_uses_negative_linear_speed": True,
    }


async def _refresh_position_after_raw_motion(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    settle_seconds: float = 2.0,
) -> dict[str, Any]:
    """Run the best proven native feedback refresh after raw movement."""
    started = time.monotonic()
    result: dict[str, Any] = {
        "method": "request_reports_count_5",
        "settle_seconds": settle_seconds,
        "ok": None,
        "error": None,
        "duration_ms": None,
    }
    try:
        await coordinator.async_get_reports(count=5)
        if settle_seconds > 0:
            await asyncio.sleep(settle_seconds)
        result["ok"] = True
    except Exception as err:  # noqa: BLE001
        result["ok"] = False
        result["error"] = f"{type(err).__name__}: {err}"
    finally:
        result["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
    return result


async def _raw_pymammotion_execute_segment(  # noqa: C901, PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    points: list[dict[str, float]],
    *,
    area_hash: int | None = None,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    prefer_ble: bool = True,
    linear_speed_fast: int = 400,
    linear_speed_slow: int = 200,
    max_commands: int = 3,
    waypoint_tolerance: float = 0.08,
    min_progress_distance: float = 0.01,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60),
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute or dry-run one calibrated raw Y-axis segment."""
    preview = _preview_custom_path(
        coordinator,
        points,
        area_hash=area_hash,
        speed=0.2,
        blade_mode="off",
    )
    normalized_points = preview["points"]
    initial_telemetry = _custom_path_telemetry_snapshot(coordinator)
    current_point = _raw_segment_current_point(initial_telemetry)
    target = normalized_points[-1] if normalized_points else None
    command_selection = (
        _raw_segment_command_selection(
            current_point,
            target,
            linear_speed_fast=linear_speed_fast,
            linear_speed_slow=linear_speed_slow,
        )
        if target is not None
        else {}
    )
    lateral_diagnostic = (
        _raw_segment_lateral_diagnostic(current_point, target)
        if target is not None
        else {
            "passed": False,
            "reason": "path_requires_exactly_two_points",
            "dx": None,
            "dy": None,
            "lateral_limit": None,
        }
    )
    gates = _manual_velocity_pulse_gates(
        coordinator,
        initial_telemetry,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    runtime_safety = _runtime_motion_safety_summary(
        initial_telemetry,
        ha_state=ha_state,
        active_route=active_route,
    )
    if not preview["valid"]:
        gates.append(
            {
                "name": "path_validation",
                "passed": False,
                "detail": "Path must pass containment validation before motion.",
            }
        )
    if not lateral_diagnostic["passed"]:
        gates.append(
            {
                "name": "linear_y_axis_segment_only",
                "passed": False,
                "detail": "Part 1 accepts mostly Y-axis segments only.",
            }
        )
    if runtime_safety["active_mowing_detected"]:
        gates.append(
            {
                "name": "runtime_not_mowing",
                "passed": False,
                "detail": "Raw segment execution is blocked while mowing is active.",
            }
        )
    if runtime_safety["active_route_status"]["blocks_motion"]:
        gates.append(
            {
                "name": "runtime_route_not_blocking",
                "passed": False,
                "detail": "Raw segment execution is blocked by live/ambiguous route data.",
            }
        )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    completion_status = _manual_velocity_completion_status(
        normalized_points,
        initial_telemetry,
        waypoint_tolerance=waypoint_tolerance,
    )
    result: dict[str, Any] = {
        **preview,
        "service": SERVICE_RAW_PYMAMMOTION_EXECUTE_SEGMENT,
        "mode": "dry_run" if dry_run else "real_raw_linear_segment",
        "dry_run": dry_run,
        "would_send": not dry_run and not blockers,
        "real_execution_scope": "one_segment_raw_y_axis_only",
        "full_path_execution_allowed": False,
        "prefer_ble": prefer_ble,
        "transport_preference": "ble_preferred" if prefer_ble else "default",
        "linear_speed_fast": linear_speed_fast,
        "linear_speed_slow": linear_speed_slow,
        "max_commands": max_commands,
        "waypoint_tolerance": waypoint_tolerance,
        "min_progress_distance": min_progress_distance,
        "sample_delays": list(sample_delays),
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "points": normalized_points,
        "advisory_start": normalized_points[0] if normalized_points else None,
        "true_start": current_point,
        "target": target,
        "selected_axis": "map_y",
        "lateral_diagnostic": lateral_diagnostic,
        "initial_command_selection": command_selection,
        "initial_telemetry": initial_telemetry,
        "final_telemetry": initial_telemetry,
        "runtime_safety": runtime_safety,
        "safety_gates": gates,
        "blockers": blockers,
        "commands_sent": 0,
        "command_results": [],
        "samples": [{"label": "initial", "telemetry": initial_telemetry}],
        "progress_diagnostics": [],
        "completion_status": completion_status,
        "stop_reason": None,
    }

    if not preview["valid"]:
        result["stop_reason"] = "path_validation_failed"
        return result
    if not _position_available(initial_telemetry):
        result["stop_reason"] = "position_unavailable"
        return result
    if completion_status["complete"]:
        result["stop_reason"] = "target_reached"
        return result
    if not lateral_diagnostic["passed"]:
        result["stop_reason"] = "segment_requires_lateral_or_turning_motion"
        return result
    if dry_run:
        result["stop_reason"] = "dry_run"
        result["command_not_sent"] = {
            "manager_method": "send_command_with_args",
            "device_name": getattr(coordinator, "device_name", None),
            "command": "send_movement",
            "prefer_ble": prefer_ble,
            "kwargs": {
                "linear_speed": command_selection.get("linear_speed"),
                "angular_speed": 0,
            },
        }
        return result
    if blockers:
        result["stop_reason"] = "safety_gates_failed"
        return result

    baseline_telemetry = initial_telemetry
    for command_index in range(1, max_commands + 1):
        before = _custom_path_telemetry_snapshot(coordinator)
        result["final_telemetry"] = before
        if not _position_available(before):
            result["stop_reason"] = "position_unavailable"
            return result
        if not _blade_reported_safe(before):
            result["stop_reason"] = "blade_unsafe"
            return result
        current_runtime_safety = _runtime_motion_safety_summary(
            before,
            ha_state=ha_state,
            active_route=active_route,
        )
        if (
            current_runtime_safety["active_mowing_detected"]
            or current_runtime_safety["active_route_status"]["blocks_motion"]
            or before.get("charge_state_label") != "not_charging"
            or before.get("work_mode_label") not in {"MODE_READY", "MODE_PAUSE"}
        ):
            result["runtime_safety"] = current_runtime_safety
            result["stop_reason"] = "mower_state_unsafe"
            return result
        quality = _manual_velocity_quality_degradation(baseline_telemetry, before)
        if quality["degraded"]:
            result["quality_degradation"] = quality
            result["stop_reason"] = "telemetry_quality_degraded"
            return result
        completion_status = _manual_velocity_completion_status(
            normalized_points,
            before,
            waypoint_tolerance=waypoint_tolerance,
        )
        result["completion_status"] = completion_status
        if completion_status["complete"]:
            result["stop_reason"] = "target_reached"
            result["final_telemetry"] = before
            return result
        if target is None:
            result["stop_reason"] = "path_validation_failed"
            return result
        current_point = _raw_segment_current_point(before)
        lateral_diagnostic = _raw_segment_lateral_diagnostic(current_point, target)
        if not lateral_diagnostic["passed"]:
            result["lateral_diagnostic"] = lateral_diagnostic
            result["stop_reason"] = "segment_requires_lateral_or_turning_motion"
            return result
        selection = _raw_segment_command_selection(
            current_point,
            target,
            linear_speed_fast=linear_speed_fast,
            linear_speed_slow=linear_speed_slow,
        )
        command_result: dict[str, Any] = {
            "index": command_index,
            "attempted": True,
            "ok": None,
            "ack": None,
            "error": None,
            "duration_ms": None,
            "command": "send_movement",
            "prefer_ble": prefer_ble,
            "kwargs": {
                "linear_speed": selection["linear_speed"],
                "angular_speed": 0,
            },
            "selection": selection,
        }
        started = time.monotonic()
        try:
            await _send_manager_command_with_args(
                coordinator,
                "send_movement",
                prefer_ble=prefer_ble,
                command_kwargs=command_result["kwargs"],
            )
            command_result["ack"] = None
            command_result["ok"] = True
        except Exception as err:  # noqa: BLE001
            command_result["ok"] = False
            command_result["error"] = f"{type(err).__name__}: {err}"
        finally:
            command_result["duration_ms"] = round(
                (time.monotonic() - started) * 1000,
                3,
            )
        result["command_results"].append(command_result)
        result["commands_sent"] += 1
        if command_result["ok"] is not True:
            result["stop_reason"] = "command_failed"
            return result
        command_result["post_command_feedback_refresh"] = (
            await _refresh_position_after_raw_motion(coordinator)
        )

        command_samples: list[dict[str, Any]] = []
        previous_delay = 0.0
        for sample_index, delay in enumerate(sample_delays):
            await asyncio.sleep(max(0.0, float(delay) - previous_delay))
            previous_delay = float(delay)
            sample_telemetry = _custom_path_telemetry_snapshot(coordinator)
            sample = {
                "label": f"command_{command_index}_sample_{sample_index + 1}_{delay:g}s",
                "command_index": command_index,
                "delay_seconds": float(delay),
                "telemetry": sample_telemetry,
            }
            result["samples"].append(sample)
            command_samples.append(sample)

        after = (
            command_samples[-1]["telemetry"]
            if command_samples
            else _custom_path_telemetry_snapshot(coordinator)
        )
        result["final_telemetry"] = after
        progress = _manual_velocity_path_progress_diagnostic(
            before,
            after,
            {"action": "forward", "target": target},
            min_progress_distance=min_progress_distance,
            min_heading_change_degrees=0.0,
        )
        progress.update(
            {
                "command_index": command_index,
                "measured_delta": _telemetry_position_delta(before, after),
            }
        )
        result["progress_diagnostics"].append(progress)
        completion_status = _manual_velocity_completion_status(
            normalized_points,
            after,
            waypoint_tolerance=waypoint_tolerance,
        )
        result["completion_status"] = completion_status
        if completion_status["complete"]:
            result["stop_reason"] = "target_reached"
            return result
        quality = _manual_velocity_quality_degradation(baseline_telemetry, after)
        if quality["degraded"]:
            result["quality_degradation"] = quality
            result["stop_reason"] = "telemetry_quality_degraded"
            return result
        if not _blade_reported_safe(after):
            result["stop_reason"] = "blade_unsafe"
            return result
        if not progress["passed"]:
            result["stop_reason"] = "no_target_progress"
            return result

    result["stop_reason"] = "max_commands_reached"
    return result


def _raw_readiness_target_points(
    telemetry: dict[str, Any],
    *,
    y_delta: float,
) -> list[dict[str, float]] | None:
    """Return two-point path from current telemetry and a Y offset."""
    current = _raw_segment_current_point(telemetry)
    if current is None:
        return None
    return [
        {"x": current["x"], "y": current["y"]},
        {"x": current["x"], "y": current["y"] + y_delta},
    ]


def _raw_readiness_target_heading(
    telemetry: dict[str, Any],
    *,
    heading_delta: float,
) -> float | None:
    """Return absolute target heading from current telemetry and a heading offset."""
    current = _normalized_heading_degrees(telemetry.get("position", {}).get("toward"))
    return None if current is None else (current + heading_delta) % 360


def _raw_readiness_phase_passed(name: str, result: dict[str, Any]) -> bool:
    """Return whether a readiness phase passed."""
    if name == "safety_snapshot":
        safety = result.get("safety", {})
        position = result.get("position", {})
        return (
            safety.get("allowed_for_manual_motion") is True
            and position.get("toward") is not None
        )
    if name in {"dry_run_negative_y_segment", "dry_run_positive_y_segment"}:
        return (
            result.get("stop_reason") == "dry_run"
            and result.get("command_not_sent", {}).get("kwargs", {}).get("angular_speed")
            == 0
        )
    if name in {"dry_run_positive_turn_to_heading", "dry_run_negative_turn_to_heading"}:
        return (
            result.get("stop_reason") == "dry_run"
            and result.get("command_not_sent", {}).get("kwargs", {}).get("linear_speed")
            == 0
        )
    if name in {"real_negative_y_segment", "real_positive_y_segment"}:
        return result.get("stop_reason") in {"target_reached", "max_commands_reached"} and all(
            diagnostic.get("passed")
            for diagnostic in result.get("progress_diagnostics", [])
        )
    if name in {"real_positive_turn_to_heading", "real_negative_turn_to_heading"}:
        return result.get("stop_reason") in {
            "target_heading_reached",
            "max_commands_reached",
        } and all(
            diagnostic.get("passed")
            for diagnostic in result.get("heading_diagnostics", [])
        )
    return False


def _raw_readiness_summary(
    phases: list[dict[str, Any]],
    *,
    failed_phase: str | None,
    real_steps_run: int,
) -> dict[str, Any]:
    """Return compact readiness summary from phase results."""
    passed_names = {phase["name"] for phase in phases if phase.get("passed")}
    dry_linear_ready = {
        "dry_run_negative_y_segment",
        "dry_run_positive_y_segment",
    }.issubset(passed_names)
    dry_turn_ready = {
        "dry_run_positive_turn_to_heading",
        "dry_run_negative_turn_to_heading",
    }.issubset(passed_names)
    real_phase_names = {
        phase["name"] for phase in phases if phase["name"].startswith("real_")
    }
    real_phases_ready = all(
        phase.get("passed") for phase in phases if phase["name"].startswith("real_")
    )
    linear_y_ready = dry_linear_ready and all(
        name in passed_names
        for name in real_phase_names
        if name in {"real_negative_y_segment", "real_positive_y_segment"}
    )
    turn_to_heading_ready = dry_turn_ready and all(
        name in passed_names
        for name in real_phase_names
        if name in {"real_positive_turn_to_heading", "real_negative_turn_to_heading"}
    )
    ready_for_vector_segment = (
        failed_phase is None
        and dry_linear_ready
        and dry_turn_ready
        and real_phases_ready
    )
    return {
        "ready_for_vector_segment": ready_for_vector_segment,
        "ready_for_multi_point": False,
        "linear_y_ready": linear_y_ready,
        "angular_ready": turn_to_heading_ready,
        "turn_to_heading_ready": turn_to_heading_ready,
        "real_steps_run": real_steps_run,
        "failed_phase": failed_phase,
        "recommended_next_step": (
            "implement_vector_segment"
            if ready_for_vector_segment
            else "fix_failed_readiness_phase"
        ),
    }


def _raw_readiness_response(
    *,
    dry_run: bool,
    confirm_blades_off: bool,
    confirm_clear_area: bool,
    prefer_ble: bool,
    max_real_steps: int,
    sample_delays: list[float] | tuple[float, ...],
    blockers: list[str],
    phases: list[dict[str, Any]],
    failed_phase: str | None,
    real_steps_run: int,
) -> dict[str, Any]:
    """Build a raw readiness service response."""
    return {
        "service": SERVICE_RAW_MOTION_READINESS_TEST,
        "mode": "dry_run" if dry_run else "real_readiness",
        "dry_run": dry_run,
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "prefer_ble": prefer_ble,
        "max_real_steps": max_real_steps,
        "sample_delays": list(sample_delays),
        "blockers": blockers,
        "phases": phases,
        **_raw_readiness_summary(
            phases,
            failed_phase=failed_phase,
            real_steps_run=real_steps_run,
        ),
    }


def _raw_vector_readiness_target_points(
    telemetry: dict[str, Any],
    *,
    reported_heading_delta: float,
    target_distance: float,
    calibrated_forward_heading_offset_degrees: float,
) -> list[dict[str, float]] | None:
    """Return a two-point vector target from live telemetry and heading delta."""
    current = _raw_segment_current_point(telemetry)
    current_heading = _normalized_heading_degrees(
        telemetry.get("position", {}).get("toward")
    )
    if current is None or current_heading is None:
        return None
    target_reported_heading = (current_heading + reported_heading_delta) % 360
    target_map_heading = (
        target_reported_heading + calibrated_forward_heading_offset_degrees
    ) % 360
    return [
        {"x": current["x"], "y": current["y"]},
        {
            "x": current["x"]
            + math.cos(math.radians(target_map_heading)) * target_distance,
            "y": current["y"]
            + math.sin(math.radians(target_map_heading)) * target_distance,
        },
    ]


def _raw_vector_readiness_phase_passed(name: str, result: dict[str, Any]) -> bool:
    """Return whether a vector readiness phase passed."""
    if name == "safety_snapshot":
        safety = result.get("safety", {})
        position = result.get("position", {})
        return (
            safety.get("allowed_for_manual_motion") is True
            and position.get("toward") is not None
        )
    if name.startswith("dry_run_"):
        phases = result.get("phases") or []
        return (
            result.get("stop_reason") == "dry_run"
            and result.get("valid") is True
            and not result.get("blockers")
            and len(phases) == 2
            and all(phase.get("passed") for phase in phases)
        )
    if name.startswith("real_"):
        progress_diagnostics = result.get("progress_diagnostics", [])
        showed_path_progress = any(
            diagnostic.get("status") == "path_progress" and diagnostic.get("passed")
            for diagnostic in progress_diagnostics
        )
        showed_translation_signal = False
        for diagnostic in progress_diagnostics:
            if diagnostic.get("status") != "no_path_progress":
                continue
            if diagnostic.get("heading_progress") is not True:
                continue
            min_progress_distance = float(diagnostic.get("min_progress_distance") or 0.0)
            measured_delta = diagnostic.get("measured_delta") or {}
            distance_value = measured_delta.get("distance")
            measured_distance = (
                float(cast(float, distance_value))
                if distance_value is not None
                else abs(float(diagnostic.get("path_progress_distance") or 0.0))
            )
            if measured_distance >= max(0.002, min_progress_distance * 0.8):
                showed_translation_signal = True
                break
        return (
            result.get("stop_reason") in {"target_reached", "no_target_progress"}
            and result.get("valid") is True
            and not result.get("blockers")
            and (
                showed_path_progress
                or (name == "real_aligned_vector" and showed_translation_signal)
            )
        )
    return False


def _raw_vector_readiness_summary(
    phases: list[dict[str, Any]],
    *,
    failed_phase: str | None,
    real_steps_run: int,
) -> dict[str, Any]:
    """Return compact vector readiness summary."""
    passed_names = {phase["name"] for phase in phases if phase.get("passed")}
    aligned_ready = "dry_run_aligned_vector" in passed_names and (
        "real_aligned_vector" not in {phase["name"] for phase in phases}
        or "real_aligned_vector" in passed_names
    )
    positive_ready = "dry_run_positive_turn_vector" in passed_names and (
        "real_positive_turn_vector" not in {phase["name"] for phase in phases}
        or "real_positive_turn_vector" in passed_names
    )
    negative_ready = "dry_run_negative_turn_vector" in passed_names and (
        "real_negative_turn_vector" not in {phase["name"] for phase in phases}
        or "real_negative_turn_vector" in passed_names
    )
    ready_for_multi_segment = (
        failed_phase is None and aligned_ready and positive_ready and negative_ready
    )
    return {
        "aligned_vector_ready": aligned_ready,
        "positive_turn_vector_ready": positive_ready,
        "negative_turn_vector_ready": negative_ready,
        "ready_for_multi_segment": ready_for_multi_segment,
        "ready_for_multi_point": False,
        "real_steps_run": real_steps_run,
        "failed_phase": failed_phase,
        "recommended_next_step": (
            "implement_guarded_multi_segment_wrapper"
            if ready_for_multi_segment
            else "fix_failed_vector_readiness_phase"
        ),
    }


def _raw_vector_readiness_response(
    *,
    dry_run: bool,
    confirm_blades_off: bool,
    confirm_clear_area: bool,
    prefer_ble: bool,
    max_real_steps: int,
    target_distance: float,
    turn_delta_degrees: float,
    calibrated_forward_heading_offset_degrees: float,
    max_turn_commands: int,
    max_linear_commands: int,
    sample_delays: list[float] | tuple[float, ...],
    blockers: list[str],
    phases: list[dict[str, Any]],
    failed_phase: str | None,
    real_steps_run: int,
) -> dict[str, Any]:
    """Build a raw vector readiness service response."""
    return {
        "service": SERVICE_RAW_VECTOR_READINESS_TEST,
        "mode": "dry_run" if dry_run else "real_vector_readiness",
        "dry_run": dry_run,
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "prefer_ble": prefer_ble,
        "max_real_steps": max_real_steps,
        "target_distance": target_distance,
        "turn_delta_degrees": turn_delta_degrees,
        "calibrated_forward_heading_offset_degrees": (
            calibrated_forward_heading_offset_degrees
        ),
        "max_turn_commands": max_turn_commands,
        "max_linear_commands": max_linear_commands,
        "sample_delays": list(sample_delays),
        "blockers": blockers,
        "phases": phases,
        **_raw_vector_readiness_summary(
            phases,
            failed_phase=failed_phase,
            real_steps_run=real_steps_run,
        ),
    }


async def _raw_vector_readiness_test(  # noqa: C901, PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    *,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    prefer_ble: bool = True,
    max_real_steps: int = 0,
    target_distance: float = 0.10,
    turn_delta_degrees: float = 10.0,
    calibrated_forward_heading_offset_degrees: float = 116.5,
    max_turn_commands: int = 4,
    max_linear_commands: int = 2,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60),
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run consolidated vector readiness checks."""
    real_min_progress_distance = 0.0025
    phases: list[dict[str, Any]] = []
    blockers: list[str] = []
    failed_phase: str | None = None
    real_steps_run = 0
    route = active_route or {}

    def add_phase(name: str, result: dict[str, Any]) -> bool:
        nonlocal failed_phase
        passed = _raw_vector_readiness_phase_passed(name, result)
        phases.append({"name": name, "passed": passed, "result": result})
        if not passed and failed_phase is None:
            failed_phase = name
        return passed

    def response() -> dict[str, Any]:
        return _raw_vector_readiness_response(
            dry_run=dry_run,
            confirm_blades_off=confirm_blades_off,
            confirm_clear_area=confirm_clear_area,
            prefer_ble=prefer_ble,
            max_real_steps=max_real_steps,
            target_distance=target_distance,
            turn_delta_degrees=turn_delta_degrees,
            calibrated_forward_heading_offset_degrees=(
                calibrated_forward_heading_offset_degrees
            ),
            max_turn_commands=max_turn_commands,
            max_linear_commands=max_linear_commands,
            sample_delays=sample_delays,
            blockers=blockers,
            phases=phases,
            failed_phase=failed_phase,
            real_steps_run=real_steps_run,
        )

    runtime_state = _export_runtime_state(
        coordinator,
        ha_state=ha_state,
        active_route=route,
    )
    if not add_phase("safety_snapshot", runtime_state):
        blockers = list(runtime_state.get("safety", {}).get("blockers") or [])
        if runtime_state.get("position", {}).get("toward") is None:
            blockers.append("heading_unavailable")
        return response()

    async def vector_phase(
        name: str,
        *,
        reported_heading_delta: float,
        real: bool,
    ) -> bool:
        telemetry = _custom_path_telemetry_snapshot(coordinator)
        points = _raw_vector_readiness_target_points(
            telemetry,
            reported_heading_delta=reported_heading_delta,
            target_distance=target_distance,
            calibrated_forward_heading_offset_degrees=(
                calibrated_forward_heading_offset_degrees
            ),
        )
        if points is None:
            return add_phase(
                name,
                {
                    "stop_reason": "position_unavailable",
                    "blockers": ["position_unavailable"],
                },
            )
        result = await _raw_pymammotion_execute_vector_segment(
            coordinator,
            points,
            dry_run=not real,
            confirm_blades_off=confirm_blades_off if real else False,
            confirm_clear_area=confirm_clear_area if real else False,
            prefer_ble=prefer_ble,
            linear_speed_fast=400,
            linear_speed_slow=200,
            slow_linear_threshold=0.15,
            max_turn_commands=max_turn_commands,
            max_linear_commands=max_linear_commands,
            heading_tolerance_degrees=3.0,
            angular_speed_fast=180,
            angular_speed_slow=180,
            slow_turn_threshold_degrees=8.0,
            waypoint_tolerance=0.08,
            min_progress_distance=real_min_progress_distance,
            min_heading_change_degrees=0.5,
            max_turn_translation_distance=0.25,
            calibrated_forward_heading_offset_degrees=(
                calibrated_forward_heading_offset_degrees
            ),
            sample_delays=tuple(sample_delays),
            ha_state=ha_state,
            active_route=route,
        )
        return add_phase(name, result)

    dry_phase_specs: tuple[tuple[str, float], ...] = (
        ("dry_run_aligned_vector", 0.0),
        ("dry_run_positive_turn_vector", turn_delta_degrees),
        ("dry_run_negative_turn_vector", -turn_delta_degrees),
    )
    for phase_name, heading_delta in dry_phase_specs:
        if not await vector_phase(
            phase_name,
            reported_heading_delta=heading_delta,
            real=False,
        ):
            blockers = list(phases[-1]["result"].get("blockers") or [])
            return response()

    if not dry_run and max_real_steps > 0 and (
        not confirm_blades_off or not confirm_clear_area
    ):
        failed_phase = "real_preflight"
        blockers = [
            blocker
            for blocker, passed in (
                ("operator_confirmed_blades_off", confirm_blades_off),
                ("operator_confirmed_clear_area", confirm_clear_area),
            )
            if not passed
        ]
        phases.append(
            {
                "name": "real_preflight",
                "passed": False,
                "result": {"stop_reason": "safety_gates_failed", "blockers": blockers},
            }
        )
        return response()

    real_phase_specs: tuple[tuple[str, float], ...] = (
        ("real_aligned_vector", 0.0),
        ("real_positive_turn_vector", turn_delta_degrees),
        ("real_negative_turn_vector", -turn_delta_degrees),
    )
    for phase_name, heading_delta in real_phase_specs[:max_real_steps]:
        if dry_run:
            break
        real_steps_run += 1
        if not await vector_phase(
            phase_name,
            reported_heading_delta=heading_delta,
            real=True,
        ):
            blockers = list(phases[-1]["result"].get("blockers") or [])
            break

    return response()


async def _raw_motion_readiness_test(  # noqa: C901, PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    *,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    prefer_ble: bool = True,
    max_real_steps: int = 0,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60),
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run consolidated raw motion readiness checks."""
    phases: list[dict[str, Any]] = []
    blockers: list[str] = []
    failed_phase: str | None = None
    real_steps_run = 0
    route = active_route or {}

    def add_phase(name: str, result: dict[str, Any]) -> bool:
        nonlocal failed_phase
        passed = _raw_readiness_phase_passed(name, result)
        phases.append({"name": name, "passed": passed, "result": result})
        if not passed and failed_phase is None:
            failed_phase = name
        return passed

    runtime_state = _export_runtime_state(
        coordinator,
        ha_state=ha_state,
        active_route=route,
    )
    if not add_phase("safety_snapshot", runtime_state):
        blockers = list(runtime_state.get("safety", {}).get("blockers") or [])
        if runtime_state.get("position", {}).get("toward") is None:
            blockers.append("heading_unavailable")
        return _raw_readiness_response(
            dry_run=dry_run,
            confirm_blades_off=confirm_blades_off,
            confirm_clear_area=confirm_clear_area,
            prefer_ble=prefer_ble,
            max_real_steps=max_real_steps,
            sample_delays=sample_delays,
            blockers=blockers,
            phases=phases,
            failed_phase=failed_phase,
            real_steps_run=real_steps_run,
        )

    async def segment_phase(name: str, *, y_delta: float, real: bool) -> bool:
        telemetry = _custom_path_telemetry_snapshot(coordinator)
        points = _raw_readiness_target_points(telemetry, y_delta=y_delta)
        if points is None:
            return add_phase(
                name,
                {"stop_reason": "position_unavailable", "blockers": ["position_unavailable"]},
            )
        result = await _raw_pymammotion_execute_segment(
            coordinator,
            points,
            dry_run=not real,
            confirm_blades_off=confirm_blades_off if real else False,
            confirm_clear_area=confirm_clear_area if real else False,
            prefer_ble=prefer_ble,
            linear_speed_fast=400,
            linear_speed_slow=200,
            max_commands=1,
            waypoint_tolerance=0.08,
            min_progress_distance=0.01,
            sample_delays=tuple(sample_delays),
            ha_state=ha_state,
            active_route=route,
        )
        return add_phase(name, result)

    async def heading_phase(name: str, *, heading_delta: float, real: bool) -> bool:
        telemetry = _custom_path_telemetry_snapshot(coordinator)
        target_heading = _raw_readiness_target_heading(
            telemetry,
            heading_delta=heading_delta,
        )
        if target_heading is None:
            return add_phase(
                name,
                {"stop_reason": "heading_unavailable", "blockers": ["heading_unavailable"]},
            )
        result = await _raw_pymammotion_turn_to_heading(
            coordinator,
            target_heading_degrees=target_heading,
            heading_tolerance_degrees=3.0,
            angular_speed_fast=180,
            angular_speed_slow=180,
            slow_turn_threshold_degrees=8.0,
            max_commands=1,
            min_heading_change_degrees=0.5,
            max_translation_distance=0.25,
            prefer_ble=prefer_ble,
            sample_delays=tuple(sample_delays),
            dry_run=not real,
            confirm_blades_off=confirm_blades_off if real else False,
            confirm_clear_area=confirm_clear_area if real else False,
            ha_state=ha_state,
            active_route=route,
        )
        return add_phase(name, result)

    for phase_name, y_delta in (
        ("dry_run_negative_y_segment", -0.10),
        ("dry_run_positive_y_segment", 0.10),
    ):
        if not await segment_phase(phase_name, y_delta=y_delta, real=False):
            blockers = list(phases[-1]["result"].get("blockers") or [])
            return _raw_readiness_response(
                dry_run=dry_run,
                confirm_blades_off=confirm_blades_off,
                confirm_clear_area=confirm_clear_area,
                prefer_ble=prefer_ble,
                max_real_steps=max_real_steps,
                sample_delays=sample_delays,
                blockers=blockers,
                phases=phases,
                failed_phase=failed_phase,
                real_steps_run=real_steps_run,
            )

    for phase_name, heading_delta in (
        ("dry_run_positive_turn_to_heading", 8.0),
        ("dry_run_negative_turn_to_heading", -8.0),
    ):
        if not await heading_phase(phase_name, heading_delta=heading_delta, real=False):
            blockers = list(phases[-1]["result"].get("blockers") or [])
            return _raw_readiness_response(
                dry_run=dry_run,
                confirm_blades_off=confirm_blades_off,
                confirm_clear_area=confirm_clear_area,
                prefer_ble=prefer_ble,
                max_real_steps=max_real_steps,
                sample_delays=sample_delays,
                blockers=blockers,
                phases=phases,
                failed_phase=failed_phase,
                real_steps_run=real_steps_run,
            )

    if not dry_run and max_real_steps > 0 and (
        not confirm_blades_off or not confirm_clear_area
    ):
        failed_phase = "real_preflight"
        blockers = [
            blocker
            for blocker, passed in (
                ("operator_confirmed_blades_off", confirm_blades_off),
                ("operator_confirmed_clear_area", confirm_clear_area),
            )
            if not passed
        ]
        phases.append(
            {
                "name": "real_preflight",
                "passed": False,
                "result": {"stop_reason": "safety_gates_failed", "blockers": blockers},
            }
        )
        return _raw_readiness_response(
            dry_run=dry_run,
            confirm_blades_off=confirm_blades_off,
            confirm_clear_area=confirm_clear_area,
            prefer_ble=prefer_ble,
            max_real_steps=max_real_steps,
            sample_delays=sample_delays,
            blockers=blockers,
            phases=phases,
            failed_phase=failed_phase,
            real_steps_run=real_steps_run,
        )

    real_phase_specs: tuple[tuple[str, str, float], ...] = (
        ("real_positive_turn_to_heading", "heading", 8.0),
        ("real_negative_turn_to_heading", "heading", -8.0),
        ("real_negative_y_segment", "segment", -0.10),
        ("real_positive_y_segment", "segment", 0.10),
    )
    for phase_name, phase_type, delta in real_phase_specs[:max_real_steps]:
        if dry_run:
            break
        real_steps_run += 1
        passed = (
            await heading_phase(phase_name, heading_delta=delta, real=True)
            if phase_type == "heading"
            else await segment_phase(phase_name, y_delta=delta, real=True)
        )
        if not passed:
            blockers = list(phases[-1]["result"].get("blockers") or [])
            break

    return _raw_readiness_response(
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
        prefer_ble=prefer_ble,
        max_real_steps=max_real_steps,
        sample_delays=sample_delays,
        blockers=blockers,
        phases=phases,
        failed_phase=failed_phase,
        real_steps_run=real_steps_run,
    )


def _raw_angular_command_selection(
    *,
    direction: str,
    angular_speed: int,
) -> dict[str, Any]:
    """Return raw angular send_movement command selection."""
    signed_speed = int(angular_speed)
    if direction == "negative_heading":
        signed_speed = -signed_speed
    return {
        "command": "send_movement",
        "linear_speed": 0,
        "angular_speed": signed_speed,
        "direction": direction,
        "positive_heading_uses_positive_angular_speed": True,
        "negative_heading_uses_negative_angular_speed": True,
    }


def _raw_angular_heading_diagnostic(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    direction: str,
    min_heading_change_degrees: float,
    max_translation_distance: float,
) -> dict[str, Any]:
    """Return whether a raw angular command produced useful heading progress."""
    delta = _telemetry_position_delta(before, after)
    heading_change = delta.get("heading_change_degrees")
    if heading_change is None:
        signed_progress = None
    elif direction == "positive_heading":
        signed_progress = float(heading_change)
    else:
        signed_progress = -float(heading_change)
    heading_progress = (
        signed_progress is not None
        and signed_progress >= min_heading_change_degrees
    )
    excessive_translation = (
        delta.get("distance") is not None
        and float(delta["distance"]) > max_translation_distance
    )
    if heading_progress and not excessive_translation:
        status = "heading_progress"
    elif excessive_translation:
        status = "translation_exceeded_limit"
    elif signed_progress is not None and signed_progress > 0:
        status = "heading_progress_below_threshold"
    elif signed_progress is not None:
        status = "wrong_heading_direction"
    else:
        status = "heading_unavailable"
    return {
        "status": status,
        "passed": heading_progress and not excessive_translation,
        "direction": direction,
        "heading_change_degrees": heading_change,
        "target_direction_progress_degrees": signed_progress,
        "measured_delta": delta,
        "min_heading_change_degrees": min_heading_change_degrees,
        "max_translation_distance": max_translation_distance,
        "excessive_translation": excessive_translation,
    }


def _raw_angular_target_status(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    direction: str,
    target_heading_delta_degrees: float,
) -> dict[str, Any]:
    """Return whether cumulative heading change reached the target."""
    delta = _telemetry_position_delta(baseline, current)
    heading_change = delta.get("heading_change_degrees")
    if heading_change is None:
        progress = None
    elif direction == "positive_heading":
        progress = float(heading_change)
    else:
        progress = -float(heading_change)
    return {
        "complete": progress is not None and progress >= target_heading_delta_degrees,
        "heading_change_degrees": heading_change,
        "target_direction_progress_degrees": progress,
        "target_heading_delta_degrees": target_heading_delta_degrees,
        "measured_delta": delta,
        "reason": (
            "target_heading_reached"
            if progress is not None and progress >= target_heading_delta_degrees
            else "target_heading_remaining"
        ),
    }


def _normalized_heading_degrees(value: Any) -> float | None:
    """Return a heading normalized to [0, 360), or None."""
    if value is None:
        return None
    try:
        return float(value) % 360
    except (TypeError, ValueError):
        return None


def _raw_turn_to_heading_status(
    telemetry: dict[str, Any],
    *,
    target_heading_degrees: float,
    heading_tolerance_degrees: float,
) -> dict[str, Any]:
    """Return absolute target-heading status from current telemetry."""
    current_heading = _normalized_heading_degrees(
        telemetry.get("position", {}).get("toward")
    )
    target_heading = _normalized_heading_degrees(target_heading_degrees)
    if current_heading is None or target_heading is None:
        return {
            "complete": False,
            "current_heading_degrees": current_heading,
            "target_heading_degrees": target_heading,
            "heading_error_degrees": None,
            "absolute_heading_error_degrees": None,
            "heading_tolerance_degrees": heading_tolerance_degrees,
            "reason": "heading_unavailable",
        }
    heading_error = _heading_error_degrees(current_heading, target_heading)
    absolute_error = abs(heading_error)
    return {
        "complete": absolute_error <= heading_tolerance_degrees,
        "current_heading_degrees": current_heading,
        "target_heading_degrees": target_heading,
        "heading_error_degrees": heading_error,
        "absolute_heading_error_degrees": absolute_error,
        "heading_tolerance_degrees": heading_tolerance_degrees,
        "reason": (
            "target_heading_reached"
            if absolute_error <= heading_tolerance_degrees
            else "target_heading_remaining"
        ),
    }


def _raw_turn_to_heading_command_selection(
    status: dict[str, Any],
    *,
    angular_speed_fast: int,
    angular_speed_slow: int,
    slow_turn_threshold_degrees: float,
) -> dict[str, Any]:
    """Return the raw angular command for the current heading error."""
    heading_error = status.get("heading_error_degrees")
    if heading_error is None:
        return {
            "command": "send_movement",
            "linear_speed": 0,
            "angular_speed": None,
            "direction": None,
            "speed_tier": None,
            "reason": "heading_unavailable",
        }
    direction = "positive_heading" if float(heading_error) > 0 else "negative_heading"
    magnitude = (
        int(angular_speed_slow)
        if abs(float(heading_error)) <= slow_turn_threshold_degrees
        else int(angular_speed_fast)
    )
    selection = _raw_angular_command_selection(
        direction=direction,
        angular_speed=magnitude,
    )
    return {
        **selection,
        "heading_error_degrees": heading_error,
        "absolute_heading_error_degrees": abs(float(heading_error)),
        "speed_tier": (
            "slow"
            if abs(float(heading_error)) <= slow_turn_threshold_degrees
            else "fast"
        ),
        "slow_turn_threshold_degrees": slow_turn_threshold_degrees,
    }


async def _raw_pymammotion_turn_to_heading(  # noqa: C901, PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    *,
    target_heading_degrees: float,
    heading_tolerance_degrees: float = 3.0,
    angular_speed_fast: int = 180,
    angular_speed_slow: int = 90,
    slow_turn_threshold_degrees: float = 8.0,
    max_commands: int = 3,
    min_heading_change_degrees: float = 0.5,
    max_translation_distance: float = 0.25,
    pulse_duration_ms: float = 300.0,
    prefer_ble: bool = True,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60),
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run or simulate a guarded absolute heading turn using raw angular commands.

    ``send_movement`` is a continuous-velocity command with no protocol-level
    duration bound -- the mower keeps turning until something explicitly stops
    it. Each pulse therefore sleeps ``pulse_duration_ms`` and then sends an
    explicit stop before sampling telemetry, rather than trusting the mower's
    own (empirically inconsistent) firmware auto-stop timing.
    """
    initial_telemetry = _custom_path_telemetry_snapshot(coordinator)
    heading_status = _raw_turn_to_heading_status(
        initial_telemetry,
        target_heading_degrees=target_heading_degrees,
        heading_tolerance_degrees=heading_tolerance_degrees,
    )
    selection = _raw_turn_to_heading_command_selection(
        heading_status,
        angular_speed_fast=angular_speed_fast,
        angular_speed_slow=angular_speed_slow,
        slow_turn_threshold_degrees=slow_turn_threshold_degrees,
    )
    gates = _manual_velocity_pulse_gates(
        coordinator,
        initial_telemetry,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    runtime_safety = _runtime_motion_safety_summary(
        initial_telemetry,
        ha_state=ha_state,
        active_route=active_route,
    )
    if heading_status["heading_error_degrees"] is None:
        gates.append(
            {
                "name": "live_heading_available",
                "passed": False,
                "detail": "Turn-to-heading requires live heading/toward telemetry.",
            }
        )
    if runtime_safety["active_mowing_detected"]:
        gates.append(
            {
                "name": "runtime_not_mowing",
                "passed": False,
                "detail": "Turn-to-heading is blocked while mowing is active.",
            }
        )
    if runtime_safety["active_route_status"]["blocks_motion"]:
        gates.append(
            {
                "name": "runtime_route_not_blocking",
                "passed": False,
                "detail": "Turn-to-heading is blocked by live/ambiguous route data.",
            }
        )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    result: dict[str, Any] = {
        "service": SERVICE_RAW_PYMAMMOTION_TURN_TO_HEADING,
        "mode": "dry_run" if dry_run else "real_raw_turn_to_heading",
        "dry_run": dry_run,
        "would_send": not dry_run and not blockers and not heading_status["complete"],
        "real_execution_scope": "turn_to_heading_only",
        "path_execution_allowed": False,
        "target_heading_degrees": _normalized_heading_degrees(target_heading_degrees),
        "heading_tolerance_degrees": heading_tolerance_degrees,
        "angular_speed_fast": angular_speed_fast,
        "angular_speed_slow": angular_speed_slow,
        "slow_turn_threshold_degrees": slow_turn_threshold_degrees,
        "max_commands": max_commands,
        "min_heading_change_degrees": min_heading_change_degrees,
        "max_translation_distance": max_translation_distance,
        "prefer_ble": prefer_ble,
        "transport_preference": "ble_preferred" if prefer_ble else "default",
        "sample_delays": list(sample_delays),
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "initial_command_selection": selection,
        "initial_heading_status": heading_status,
        "heading_status": heading_status,
        "initial_telemetry": initial_telemetry,
        "final_telemetry": initial_telemetry,
        "runtime_safety": runtime_safety,
        "safety_gates": gates,
        "blockers": blockers,
        "commands_sent": 0,
        "command_results": [],
        "samples": [{"label": "initial", "telemetry": initial_telemetry}],
        "heading_diagnostics": [],
        "stop_reason": None,
    }
    if not _position_available(initial_telemetry):
        result["stop_reason"] = "position_unavailable"
        return result
    if heading_status["heading_error_degrees"] is None:
        result["stop_reason"] = "heading_unavailable"
        return result
    if heading_status["complete"]:
        result["stop_reason"] = "target_heading_reached"
        return result
    if dry_run:
        result["stop_reason"] = "dry_run"
        result["command_not_sent"] = {
            "manager_method": "send_command_with_args",
            "device_name": getattr(coordinator, "device_name", None),
            "command": "send_movement",
            "prefer_ble": prefer_ble,
            "kwargs": {
                "linear_speed": 0,
                "angular_speed": selection["angular_speed"],
            },
        }
        return result
    if blockers:
        result["stop_reason"] = "safety_gates_failed"
        return result

    baseline_telemetry = initial_telemetry
    for command_index in range(1, max_commands + 1):
        before = _custom_path_telemetry_snapshot(coordinator)
        result["final_telemetry"] = before
        if not _position_available(before):
            result["stop_reason"] = "position_unavailable"
            return result
        if before.get("position", {}).get("toward") is None:
            result["stop_reason"] = "heading_unavailable"
            return result
        if not _blade_reported_safe(before):
            result["stop_reason"] = "blade_unsafe"
            return result
        current_runtime_safety = _runtime_motion_safety_summary(
            before,
            ha_state=ha_state,
            active_route=active_route,
        )
        if (
            current_runtime_safety["active_mowing_detected"]
            or current_runtime_safety["active_route_status"]["blocks_motion"]
            or before.get("charge_state_label") != "not_charging"
            or before.get("work_mode_label") not in {"MODE_READY", "MODE_PAUSE"}
        ):
            result["runtime_safety"] = current_runtime_safety
            result["stop_reason"] = "mower_state_unsafe"
            return result
        quality = _manual_velocity_quality_degradation(baseline_telemetry, before)
        if quality["degraded"]:
            result["quality_degradation"] = quality
            result["stop_reason"] = "telemetry_quality_degraded"
            return result
        heading_status = _raw_turn_to_heading_status(
            before,
            target_heading_degrees=target_heading_degrees,
            heading_tolerance_degrees=heading_tolerance_degrees,
        )
        result["heading_status"] = heading_status
        if heading_status["complete"]:
            result["stop_reason"] = "target_heading_reached"
            return result
        selection = _raw_turn_to_heading_command_selection(
            heading_status,
            angular_speed_fast=angular_speed_fast,
            angular_speed_slow=angular_speed_slow,
            slow_turn_threshold_degrees=slow_turn_threshold_degrees,
        )
        direction = selection["direction"]
        if direction is None or selection["angular_speed"] is None:
            result["stop_reason"] = "heading_unavailable"
            return result

        command_result: dict[str, Any] = {
            "index": command_index,
            "attempted": True,
            "ok": None,
            "ack": None,
            "error": None,
            "duration_ms": None,
            "command": "send_movement",
            "prefer_ble": prefer_ble,
            "kwargs": {
                "linear_speed": 0,
                "angular_speed": selection["angular_speed"],
            },
            "selection": selection,
        }
        started = time.monotonic()
        try:
            await _send_manager_command_with_args(
                coordinator,
                "send_movement",
                prefer_ble=prefer_ble,
                command_kwargs=command_result["kwargs"],
            )
            command_result["ack"] = None
            command_result["ok"] = True
        except Exception as err:  # noqa: BLE001
            command_result["ok"] = False
            command_result["error"] = f"{type(err).__name__}: {err}"
        finally:
            command_result["duration_ms"] = round(
                (time.monotonic() - started) * 1000,
                3,
            )
        result["command_results"].append(command_result)
        result["commands_sent"] += 1
        if command_result["ok"] is not True:
            result["stop_reason"] = "command_failed"
            return result
        await asyncio.sleep(pulse_duration_ms / 1000)
        command_result["stop_result"] = await _manual_velocity_stop_attempt(
            coordinator, use_wifi=not prefer_ble
        )
        command_result["post_command_feedback_refresh"] = (
            await _refresh_position_after_raw_motion(coordinator)
        )

        command_samples: list[dict[str, Any]] = []
        previous_delay = 0.0
        for sample_index, delay in enumerate(sample_delays):
            await asyncio.sleep(max(0.0, float(delay) - previous_delay))
            previous_delay = float(delay)
            sample_telemetry = _custom_path_telemetry_snapshot(coordinator)
            sample = {
                "label": f"command_{command_index}_sample_{sample_index + 1}_{delay:g}s",
                "command_index": command_index,
                "delay_seconds": float(delay),
                "telemetry": sample_telemetry,
            }
            result["samples"].append(sample)
            command_samples.append(sample)

        after = (
            command_samples[-1]["telemetry"]
            if command_samples
            else _custom_path_telemetry_snapshot(coordinator)
        )
        result["final_telemetry"] = after
        heading_diagnostic = _raw_angular_heading_diagnostic(
            before,
            after,
            direction=direction,
            min_heading_change_degrees=min_heading_change_degrees,
            max_translation_distance=max_translation_distance,
        )
        heading_diagnostic["command_index"] = command_index
        result["heading_diagnostics"].append(heading_diagnostic)
        heading_status = _raw_turn_to_heading_status(
            after,
            target_heading_degrees=target_heading_degrees,
            heading_tolerance_degrees=heading_tolerance_degrees,
        )
        result["heading_status"] = heading_status
        if heading_status["complete"]:
            result["stop_reason"] = "target_heading_reached"
            return result
        quality = _manual_velocity_quality_degradation(baseline_telemetry, after)
        if quality["degraded"]:
            result["quality_degradation"] = quality
            result["stop_reason"] = "telemetry_quality_degraded"
            return result
        if not _blade_reported_safe(after):
            result["stop_reason"] = "blade_unsafe"
            return result
        if heading_diagnostic["excessive_translation"]:
            result["stop_reason"] = "translation_exceeded_limit"
            return result
        if not heading_diagnostic["passed"]:
            result["stop_reason"] = "no_heading_progress"
            return result

    result["stop_reason"] = "max_commands_reached"
    return result


def _raw_vector_linear_command_selection(
    telemetry: dict[str, Any],
    target: dict[str, float],
    *,
    linear_speed_fast: int,
    linear_speed_slow: int,
    slow_linear_threshold: float,
) -> dict[str, Any]:
    """Return raw forward command selection for a vector target."""
    position = telemetry.get("position", {})
    if position.get("x") is None or position.get("y") is None:
        return {
            "command": "send_movement",
            "linear_speed": None,
            "angular_speed": 0,
            "distance_to_target": None,
            "speed_tier": None,
            "reason": "position_unavailable",
        }
    distance = math.hypot(
        float(target["x"]) - float(position["x"]),
        float(target["y"]) - float(position["y"]),
    )
    speed = (
        int(linear_speed_slow)
        if distance <= slow_linear_threshold
        else int(linear_speed_fast)
    )
    return {
        "command": "send_movement",
        "linear_speed": speed,
        "angular_speed": 0,
        "distance_to_target": distance,
        "speed_tier": "slow" if speed == int(linear_speed_slow) else "fast",
        "slow_linear_threshold": slow_linear_threshold,
        "reason": "target_remaining",
    }


async def _raw_pymammotion_execute_vector_segment(  # noqa: C901, PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    points: list[dict[str, float]],
    *,
    area_hash: int | None = None,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    prefer_ble: bool = True,
    linear_speed_fast: int = 400,
    linear_speed_slow: int = 200,
    slow_linear_threshold: float = 0.15,
    max_turn_commands: int = 3,
    max_linear_commands: int = 1,
    max_linear_pulse_ceiling: int | None = None,
    max_no_progress_pulses: int = 3,
    linear_distance_ceiling_factor: float = 2.0,
    heading_tolerance_degrees: float = 3.0,
    angular_speed_fast: int = 180,
    angular_speed_slow: int = 180,
    slow_turn_threshold_degrees: float = 8.0,
    waypoint_tolerance: float = 0.08,
    min_progress_distance: float = 0.005,
    min_heading_change_degrees: float = 0.5,
    max_turn_translation_distance: float = 0.25,
    calibrated_forward_heading_offset_degrees: float = 116.5,
    turn_pulse_duration_ms: float = 300.0,
    linear_pulse_duration_ms: float = 300.0,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60),
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute or dry-run one vector segment using raw turn then forward motion.

    ``send_movement`` is a continuous-velocity command with no protocol-level
    duration bound -- the mower keeps moving until something explicitly stops
    it. Each turn/linear pulse therefore sleeps its ``*_pulse_duration_ms`` and
    then sends an explicit stop before sampling telemetry, rather than
    trusting the mower's own (empirically inconsistent) firmware auto-stop
    timing.

    When ``max_linear_pulse_ceiling`` is provided the linear phase runs in
    loop-to-tolerance mode: it keeps pulsing forward until the waypoint is
    reached, stopping only on ``max_no_progress_pulses`` consecutive
    non-progressing pulses, a cumulative-distance ceiling
    (segment length * ``linear_distance_ceiling_factor``), the pulse ceiling,
    or a safety gate. When it is ``None`` the legacy fixed
    ``max_linear_commands`` budget is used unchanged.
    """
    preview = _preview_custom_path(
        coordinator,
        points,
        area_hash=area_hash,
        speed=0.2,
        blade_mode="off",
    )
    normalized_points = preview["points"]
    initial_telemetry = _custom_path_telemetry_snapshot(coordinator)
    current_point = _raw_segment_current_point(initial_telemetry)
    target = normalized_points[-1] if normalized_points else None
    target_heading = (
        _path_heading_degrees(current_point, target)
        if current_point is not None and target is not None
        else None
    )
    target_reported_heading = (
        _normalized_heading_degrees(
            float(target_heading) - float(calibrated_forward_heading_offset_degrees)
        )
        if target_heading is not None
        else None
    )
    gates = _manual_velocity_pulse_gates(
        coordinator,
        initial_telemetry,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    runtime_safety = _runtime_motion_safety_summary(
        initial_telemetry,
        ha_state=ha_state,
        active_route=active_route,
    )
    if len(normalized_points) != 2:
        gates.append(
            {
                "name": "one_segment_only",
                "passed": False,
                "detail": "Vector segment execution accepts exactly two points.",
            }
        )
    if not preview["valid"]:
        gates.append(
            {
                "name": "path_validation",
                "passed": False,
                "detail": "Path must pass containment validation before motion.",
            }
        )
    if target_heading is None:
        gates.append(
            {
                "name": "target_heading_available",
                "passed": False,
                "detail": "Vector segment execution requires live position and target heading.",
            }
        )
    if runtime_safety["active_mowing_detected"]:
        gates.append(
            {
                "name": "runtime_not_mowing",
                "passed": False,
                "detail": "Vector segment execution is blocked while mowing is active.",
            }
        )
    if runtime_safety["active_route_status"]["blocks_motion"]:
        gates.append(
            {
                "name": "runtime_route_not_blocking",
                "passed": False,
                "detail": "Vector segment execution is blocked by live/ambiguous route data.",
            }
        )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    completion_status = _manual_velocity_completion_status(
        normalized_points,
        initial_telemetry,
        waypoint_tolerance=waypoint_tolerance,
    )
    initial_selection = (
        _raw_vector_linear_command_selection(
            initial_telemetry,
            target,
            linear_speed_fast=linear_speed_fast,
            linear_speed_slow=linear_speed_slow,
            slow_linear_threshold=slow_linear_threshold,
        )
        if target is not None
        else {}
    )
    result: dict[str, Any] = {
        **preview,
        "service": SERVICE_RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT,
        "mode": "dry_run" if dry_run else "real_raw_vector_segment",
        "dry_run": dry_run,
        "would_send": not dry_run and not blockers and not completion_status["complete"],
        "real_execution_scope": "one_segment_turn_then_forward_only",
        "full_path_execution_allowed": False,
        "ready_for_multi_point": False,
        "prefer_ble": prefer_ble,
        "transport_preference": "ble_preferred" if prefer_ble else "default",
        "linear_speed_fast": linear_speed_fast,
        "linear_speed_slow": linear_speed_slow,
        "slow_linear_threshold": slow_linear_threshold,
        "max_turn_commands": max_turn_commands,
        "max_linear_commands": max_linear_commands,
        "heading_tolerance_degrees": heading_tolerance_degrees,
        "angular_speed_fast": angular_speed_fast,
        "angular_speed_slow": angular_speed_slow,
        "slow_turn_threshold_degrees": slow_turn_threshold_degrees,
        "waypoint_tolerance": waypoint_tolerance,
        "min_progress_distance": min_progress_distance,
        "min_heading_change_degrees": min_heading_change_degrees,
        "max_turn_translation_distance": max_turn_translation_distance,
        "calibrated_forward_heading_offset_degrees": (
            calibrated_forward_heading_offset_degrees
        ),
        "sample_delays": list(sample_delays),
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "points": normalized_points,
        "advisory_start": normalized_points[0] if normalized_points else None,
        "true_start": current_point,
        "target": target,
        "target_map_heading_degrees": target_heading,
        "target_reported_heading_degrees": target_reported_heading,
        "target_heading_degrees": target_reported_heading,
        "heading_calibration": {
            "formula": (
                "target_reported_heading = "
                "target_map_heading - calibrated_forward_heading_offset"
            ),
            "target_map_heading_degrees": target_heading,
            "calibrated_forward_heading_offset_degrees": (
                calibrated_forward_heading_offset_degrees
            ),
            "target_reported_heading_degrees": target_reported_heading,
        },
        "initial_linear_command_selection": initial_selection,
        "initial_telemetry": initial_telemetry,
        "final_telemetry": initial_telemetry,
        "runtime_safety": runtime_safety,
        "safety_gates": gates,
        "blockers": blockers,
        "commands_sent": 0,
        "turn_commands_sent": 0,
        "linear_commands_sent": 0,
        "command_results": [],
        "samples": [{"label": "initial", "telemetry": initial_telemetry}],
        "phases": [],
        "progress_diagnostics": [],
        "completion_status": completion_status,
        "stop_reason": None,
    }

    if not preview["valid"]:
        result["stop_reason"] = "path_validation_failed"
        return result
    if len(normalized_points) != 2:
        result["stop_reason"] = "path_requires_exactly_two_points"
        return result
    if not _position_available(initial_telemetry):
        result["stop_reason"] = "position_unavailable"
        return result
    if target_reported_heading is None:
        result["stop_reason"] = "target_heading_unavailable"
        return result
    if completion_status["complete"]:
        result["stop_reason"] = "target_reached"
        return result
    if blockers and not dry_run:
        result["stop_reason"] = "safety_gates_failed"
        return result

    turn_result = await _raw_pymammotion_turn_to_heading(
        coordinator,
        target_heading_degrees=target_reported_heading,
        heading_tolerance_degrees=heading_tolerance_degrees,
        angular_speed_fast=angular_speed_fast,
        angular_speed_slow=angular_speed_slow,
        slow_turn_threshold_degrees=slow_turn_threshold_degrees,
        max_commands=max_turn_commands,
        min_heading_change_degrees=min_heading_change_degrees,
        max_translation_distance=max_turn_translation_distance,
        pulse_duration_ms=turn_pulse_duration_ms,
        prefer_ble=prefer_ble,
        sample_delays=tuple(sample_delays),
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
        ha_state=ha_state,
        active_route=active_route,
    )
    result["phases"].append(
        {
            "name": "turn_to_target_heading",
            "passed": turn_result.get("stop_reason")
            in {"dry_run", "target_heading_reached"},
            "result": turn_result,
        }
    )
    result["turn_commands_sent"] = int(turn_result.get("commands_sent") or 0)
    result["commands_sent"] += result["turn_commands_sent"]
    result["command_results"].extend(turn_result.get("command_results") or [])
    result["samples"].extend(
        sample
        for sample in turn_result.get("samples", [])[1:]
        if isinstance(sample, dict)
    )
    result["final_telemetry"] = turn_result.get("final_telemetry", initial_telemetry)

    if dry_run:
        result["stop_reason"] = "dry_run"
        result["command_not_sent"] = {
            "phase": "linear_forward_to_target",
            "manager_method": "send_command_with_args",
            "device_name": getattr(coordinator, "device_name", None),
            "command": "send_movement",
            "prefer_ble": prefer_ble,
            "kwargs": {
                "linear_speed": initial_selection.get("linear_speed"),
                "angular_speed": 0,
            },
        }
        result["phases"].append(
            {
                "name": "linear_forward_to_target",
                "passed": True,
                "result": {
                    "dry_run": True,
                    "stop_reason": "dry_run",
                    "command_not_sent": result["command_not_sent"],
                },
            }
        )
        return result

    if turn_result.get("stop_reason") != "target_heading_reached":
        result["stop_reason"] = "turn_phase_incomplete"
        return result

    baseline_telemetry = result["final_telemetry"]
    loop_to_tolerance = max_linear_pulse_ceiling is not None
    effective_linear_ceiling = (
        max_linear_pulse_ceiling
        if max_linear_pulse_ceiling is not None
        else max_linear_commands
    )
    segment_length = (
        _path_distance([current_point, target])
        if current_point is not None and target is not None
        else None
    )
    linear_distance_ceiling = (
        segment_length * linear_distance_ceiling_factor
        if segment_length is not None
        else None
    )
    result["linear_execution_mode"] = (
        "loop_to_tolerance" if loop_to_tolerance else "fixed_budget"
    )
    result["effective_linear_ceiling"] = effective_linear_ceiling
    result["linear_distance_ceiling"] = linear_distance_ceiling
    consecutive_no_progress = 0
    cumulative_linear_distance = 0.0
    command_index = 0
    while command_index < effective_linear_ceiling:
        command_index += 1
        before = _custom_path_telemetry_snapshot(coordinator)
        result["final_telemetry"] = before
        if not _position_available(before):
            result["stop_reason"] = "position_unavailable"
            return result
        if not _blade_reported_safe(before):
            result["stop_reason"] = "blade_unsafe"
            return result
        current_runtime_safety = _runtime_motion_safety_summary(
            before,
            ha_state=ha_state,
            active_route=active_route,
        )
        if (
            current_runtime_safety["active_mowing_detected"]
            or current_runtime_safety["active_route_status"]["blocks_motion"]
            or before.get("charge_state_label") != "not_charging"
            or before.get("work_mode_label") not in {"MODE_READY", "MODE_PAUSE"}
        ):
            result["runtime_safety"] = current_runtime_safety
            result["stop_reason"] = "mower_state_unsafe"
            return result
        quality = _manual_velocity_quality_degradation(baseline_telemetry, before)
        if quality["degraded"]:
            result["quality_degradation"] = quality
            result["stop_reason"] = "telemetry_quality_degraded"
            return result
        completion_status = _manual_velocity_completion_status(
            normalized_points,
            before,
            waypoint_tolerance=waypoint_tolerance,
        )
        result["completion_status"] = completion_status
        if completion_status["complete"]:
            result["stop_reason"] = "target_reached"
            result["final_telemetry"] = before
            return result
        if target is None:
            result["stop_reason"] = "path_validation_failed"
            return result
        selection = _raw_vector_linear_command_selection(
            before,
            target,
            linear_speed_fast=linear_speed_fast,
            linear_speed_slow=linear_speed_slow,
            slow_linear_threshold=slow_linear_threshold,
        )
        if selection["linear_speed"] is None:
            result["stop_reason"] = "position_unavailable"
            return result

        command_result: dict[str, Any] = {
            "index": command_index,
            "phase": "linear_forward_to_target",
            "attempted": True,
            "ok": None,
            "ack": None,
            "error": None,
            "duration_ms": None,
            "command": "send_movement",
            "prefer_ble": prefer_ble,
            "kwargs": {
                "linear_speed": selection["linear_speed"],
                "angular_speed": 0,
            },
            "selection": selection,
        }
        started = time.monotonic()
        try:
            await _send_manager_command_with_args(
                coordinator,
                "send_movement",
                prefer_ble=prefer_ble,
                command_kwargs=command_result["kwargs"],
            )
            command_result["ack"] = None
            command_result["ok"] = True
        except Exception as err:  # noqa: BLE001
            command_result["ok"] = False
            command_result["error"] = f"{type(err).__name__}: {err}"
        finally:
            command_result["duration_ms"] = round(
                (time.monotonic() - started) * 1000,
                3,
            )
        result["command_results"].append(command_result)
        result["commands_sent"] += 1
        result["linear_commands_sent"] += 1
        if command_result["ok"] is not True:
            result["stop_reason"] = "command_failed"
            return result
        await asyncio.sleep(linear_pulse_duration_ms / 1000)
        command_result["stop_result"] = await _manual_velocity_stop_attempt(
            coordinator, use_wifi=not prefer_ble
        )
        command_result["post_command_feedback_refresh"] = (
            await _refresh_position_after_raw_motion(coordinator)
        )

        command_samples: list[dict[str, Any]] = []
        previous_delay = 0.0
        for sample_index, delay in enumerate(sample_delays):
            await asyncio.sleep(max(0.0, float(delay) - previous_delay))
            previous_delay = float(delay)
            sample_telemetry = _custom_path_telemetry_snapshot(coordinator)
            sample = {
                "label": f"linear_{command_index}_sample_{sample_index + 1}_{delay:g}s",
                "command_index": command_index,
                "delay_seconds": float(delay),
                "telemetry": sample_telemetry,
            }
            result["samples"].append(sample)
            command_samples.append(sample)

        after = (
            command_samples[-1]["telemetry"]
            if command_samples
            else _custom_path_telemetry_snapshot(coordinator)
        )
        result["final_telemetry"] = after
        progress = _manual_velocity_path_progress_diagnostic(
            before,
            after,
            {"action": "forward", "target": target},
            min_progress_distance=min_progress_distance,
            min_heading_change_degrees=0.0,
        )
        progress.update(
            {
                "command_index": command_index,
                "measured_delta": _telemetry_position_delta(before, after),
            }
        )
        result["progress_diagnostics"].append(progress)
        completion_status = _manual_velocity_completion_status(
            normalized_points,
            after,
            waypoint_tolerance=waypoint_tolerance,
        )
        result["completion_status"] = completion_status
        if completion_status["complete"]:
            result["stop_reason"] = "target_reached"
            result["phases"].append(
                {
                    "name": "linear_forward_to_target",
                    "passed": True,
                    "result": {
                        "commands_sent": result["linear_commands_sent"],
                        "stop_reason": result["stop_reason"],
                        "progress_diagnostics": result["progress_diagnostics"],
                    },
                }
            )
            return result
        quality = _manual_velocity_quality_degradation(baseline_telemetry, after)
        if quality["degraded"]:
            result["quality_degradation"] = quality
            result["stop_reason"] = "telemetry_quality_degraded"
            return result
        if not _blade_reported_safe(after):
            result["stop_reason"] = "blade_unsafe"
            return result
        if not progress["passed"]:
            consecutive_no_progress += 1
            if loop_to_tolerance:
                if consecutive_no_progress >= max_no_progress_pulses:
                    result["stop_reason"] = "no_target_progress"
                    return result
                continue
            if command_index < max_linear_commands:
                continue
            result["stop_reason"] = "no_target_progress"
            return result
        consecutive_no_progress = 0
        if loop_to_tolerance and linear_distance_ceiling is not None:
            measured = (progress.get("measured_delta") or {}).get("distance")
            if measured is not None:
                cumulative_linear_distance += float(measured)
                if cumulative_linear_distance > linear_distance_ceiling:
                    result["cumulative_linear_distance"] = cumulative_linear_distance
                    result["stop_reason"] = "linear_distance_ceiling_reached"
                    return result

    result["stop_reason"] = (
        "max_linear_pulse_ceiling_reached"
        if loop_to_tolerance
        else "max_linear_commands_reached"
    )
    result["phases"].append(
        {
            "name": "linear_forward_to_target",
            "passed": all(
                diagnostic.get("passed")
                for diagnostic in result["progress_diagnostics"]
            ),
            "result": {
                "commands_sent": result["linear_commands_sent"],
                "stop_reason": result["stop_reason"],
                "progress_diagnostics": result["progress_diagnostics"],
            },
        }
    )
    return result


def _raw_multi_segment_phase_passed(
    segment_result: dict[str, Any],
    *,
    real_segment: bool,
) -> bool:
    """Return whether a guarded multi-segment phase passed."""
    if real_segment:
        return (
            segment_result.get("stop_reason") == "target_reached"
            and segment_result.get("valid") is True
            and not segment_result.get("blockers")
            and all(
                diagnostic.get("passed")
                for diagnostic in segment_result.get("progress_diagnostics", [])
            )
        )
    phases = segment_result.get("phases") or []
    return (
        segment_result.get("stop_reason") == "dry_run"
        and segment_result.get("valid") is True
        and not segment_result.get("blockers")
        and len(phases) == 2
        and all(phase.get("passed") for phase in phases)
    )


async def _raw_pymammotion_execute_multi_segment(  # noqa: C901, PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    points: list[dict[str, float]],
    *,
    area_hash: int | str | None = None,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    prefer_ble: bool = True,
    max_real_segments: int = 1,
    linear_speed_fast: int = 400,
    linear_speed_slow: int = 200,
    slow_linear_threshold: float = 0.15,
    max_turn_commands: int = 4,
    max_linear_commands: int = 2,
    max_linear_pulse_ceiling: int | None = None,
    max_no_progress_pulses: int = 3,
    linear_distance_ceiling_factor: float = 2.0,
    heading_tolerance_degrees: float = 3.0,
    angular_speed_fast: int = 180,
    angular_speed_slow: int = 180,
    slow_turn_threshold_degrees: float = 8.0,
    waypoint_tolerance: float = 0.08,
    min_progress_distance: float = 0.005,
    min_heading_change_degrees: float = 0.5,
    max_turn_translation_distance: float = 0.25,
    calibrated_forward_heading_offset_degrees: float = 116.5,
    turn_pulse_duration_ms: float = 300.0,
    linear_pulse_duration_ms: float = 300.0,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60),
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute or dry-run a guarded chain of proven raw vector segments."""
    normalized_area_hash = _coerce_optional_int(area_hash)
    preview = _preview_custom_path(
        coordinator,
        points,
        area_hash=normalized_area_hash,
        speed=0.2,
        blade_mode="off",
    )
    normalized_points = preview["points"]
    initial_telemetry = _custom_path_telemetry_snapshot(coordinator)
    gates = _manual_velocity_pulse_gates(
        coordinator,
        initial_telemetry,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    runtime_safety = _runtime_motion_safety_summary(
        initial_telemetry,
        ha_state=ha_state,
        active_route=active_route,
    )
    if len(normalized_points) < 2 or len(normalized_points) > 4:
        gates.append(
            {
                "name": "point_count_2_to_4",
                "passed": False,
                "detail": "Multi-segment execution accepts 2 to 4 points.",
            }
        )
    if not dry_run and max_real_segments < 1:
        gates.append(
            {
                "name": "max_real_segments_positive",
                "passed": False,
                "detail": "Real multi-segment execution requires max_real_segments >= 1.",
            }
        )
    if not preview["valid"]:
        gates.append(
            {
                "name": "path_validation",
                "passed": False,
                "detail": "Path must pass containment validation before motion.",
            }
        )
    if runtime_safety["active_mowing_detected"]:
        gates.append(
            {
                "name": "runtime_not_mowing",
                "passed": False,
                "detail": "Multi-segment execution is blocked while mowing is active.",
            }
        )
    if runtime_safety["active_route_status"]["blocks_motion"]:
        gates.append(
            {
                "name": "runtime_route_not_blocking",
                "passed": False,
                "detail": "Multi-segment execution is blocked by live/ambiguous route data.",
            }
        )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    total_segments = max(0, len(normalized_points) - 1)
    result: dict[str, Any] = {
        **preview,
        "service": SERVICE_RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT,
        "mode": "dry_run" if dry_run else "real_raw_multi_segment",
        "dry_run": dry_run,
        "would_send": not dry_run and not blockers and total_segments > 0,
        "real_execution_scope": "guarded_multi_segment_vector_chain",
        "full_path_execution_allowed": False,
        "ready_for_multi_point": False,
        "ready_for_multi_segment": False,
        "prefer_ble": prefer_ble,
        "transport_preference": "ble_preferred" if prefer_ble else "default",
        "max_real_segments": max_real_segments,
        "max_turn_commands": max_turn_commands,
        "max_linear_commands": max_linear_commands,
        "linear_speed_fast": linear_speed_fast,
        "linear_speed_slow": linear_speed_slow,
        "slow_linear_threshold": slow_linear_threshold,
        "heading_tolerance_degrees": heading_tolerance_degrees,
        "angular_speed_fast": angular_speed_fast,
        "angular_speed_slow": angular_speed_slow,
        "slow_turn_threshold_degrees": slow_turn_threshold_degrees,
        "waypoint_tolerance": waypoint_tolerance,
        "min_progress_distance": min_progress_distance,
        "min_heading_change_degrees": min_heading_change_degrees,
        "max_turn_translation_distance": max_turn_translation_distance,
        "calibrated_forward_heading_offset_degrees": (
            calibrated_forward_heading_offset_degrees
        ),
        "sample_delays": list(sample_delays),
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "points": normalized_points,
        "total_segments": total_segments,
        "segments_planned": total_segments,
        "segments_executed": 0,
        "real_segments_executed": 0,
        "segments": [],
        "initial_telemetry": initial_telemetry,
        "final_telemetry": initial_telemetry,
        "runtime_safety": runtime_safety,
        "safety_gates": gates,
        "blockers": blockers,
        "stop_reason": None,
        "failed_segment_index": None,
    }

    if not preview["valid"]:
        result["stop_reason"] = "path_validation_failed"
        return result
    if total_segments < 1 or len(normalized_points) > 4:
        result["stop_reason"] = "invalid_point_count"
        return result
    if blockers and not dry_run:
        result["stop_reason"] = "safety_gates_failed"
        return result

    for segment_offset in range(total_segments):
        segment_index = segment_offset + 1
        segment_points = [
            normalized_points[segment_offset],
            normalized_points[segment_offset + 1],
        ]
        if not dry_run and segment_index > max_real_segments:
            result["segments"].append(
                {
                    "index": segment_index,
                    "points": segment_points,
                    "real_segment": False,
                    "passed": None,
                    "skipped_reason": "max_real_segments_reached",
                }
            )
            result["stop_reason"] = "max_real_segments_reached"
            return result

        pre_segment_telemetry = _custom_path_telemetry_snapshot(coordinator)
        pre_segment_runtime_safety = _runtime_motion_safety_summary(
            pre_segment_telemetry,
            ha_state=ha_state,
            active_route=active_route,
        )
        if not dry_run and (
            not _position_available(pre_segment_telemetry)
            or not _blade_reported_safe(pre_segment_telemetry)
            or pre_segment_runtime_safety["active_mowing_detected"]
            or pre_segment_runtime_safety["active_route_status"]["blocks_motion"]
            or pre_segment_telemetry.get("charge_state_label") != "not_charging"
            or pre_segment_telemetry.get("work_mode_label")
            not in {"MODE_READY", "MODE_PAUSE"}
        ):
            result["segments"].append(
                {
                    "index": segment_index,
                    "points": segment_points,
                    "real_segment": True,
                    "passed": False,
                    "pre_segment_telemetry": pre_segment_telemetry,
                    "runtime_safety": pre_segment_runtime_safety,
                    "stop_reason": "pre_segment_safety_failed",
                }
            )
            result["stop_reason"] = "pre_segment_safety_failed"
            result["failed_segment_index"] = segment_index
            result["final_telemetry"] = pre_segment_telemetry
            return result

        segment_result = await _raw_pymammotion_execute_vector_segment(
            coordinator,
            segment_points,
            area_hash=normalized_area_hash,
            dry_run=dry_run,
            confirm_blades_off=confirm_blades_off,
            confirm_clear_area=confirm_clear_area,
            prefer_ble=prefer_ble,
            linear_speed_fast=linear_speed_fast,
            linear_speed_slow=linear_speed_slow,
            slow_linear_threshold=slow_linear_threshold,
            max_turn_commands=max_turn_commands,
            max_linear_commands=max_linear_commands,
            max_linear_pulse_ceiling=max_linear_pulse_ceiling,
            max_no_progress_pulses=max_no_progress_pulses,
            linear_distance_ceiling_factor=linear_distance_ceiling_factor,
            heading_tolerance_degrees=heading_tolerance_degrees,
            angular_speed_fast=angular_speed_fast,
            angular_speed_slow=angular_speed_slow,
            slow_turn_threshold_degrees=slow_turn_threshold_degrees,
            waypoint_tolerance=waypoint_tolerance,
            min_progress_distance=min_progress_distance,
            min_heading_change_degrees=min_heading_change_degrees,
            max_turn_translation_distance=max_turn_translation_distance,
            calibrated_forward_heading_offset_degrees=(
                calibrated_forward_heading_offset_degrees
            ),
            turn_pulse_duration_ms=turn_pulse_duration_ms,
            linear_pulse_duration_ms=linear_pulse_duration_ms,
            sample_delays=tuple(sample_delays),
            ha_state=ha_state,
            active_route=active_route,
        )
        passed = _raw_multi_segment_phase_passed(
            segment_result,
            real_segment=not dry_run,
        )
        result["segments"].append(
            {
                "index": segment_index,
                "points": segment_points,
                "real_segment": not dry_run,
                "passed": passed,
                "result": segment_result,
            }
        )
        result["segments_executed"] += 1
        if not dry_run:
            result["real_segments_executed"] += 1
        result["final_telemetry"] = segment_result.get(
            "final_telemetry",
            result["final_telemetry"],
        )
        if not passed:
            result["stop_reason"] = "segment_failed"
            result["failed_segment_index"] = segment_index
            return result

    result["ready_for_multi_segment"] = dry_run or (
        result["real_segments_executed"] == total_segments
    )
    result["stop_reason"] = "dry_run" if dry_run else "target_reached"
    return result


async def _raw_pymammotion_angular_calibration(  # noqa: C901, PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    *,
    direction: str = "positive_heading",
    angular_speed: int = 180,
    target_heading_delta_degrees: float = 10.0,
    max_commands: int = 3,
    min_heading_change_degrees: float = 1.0,
    max_translation_distance: float = 0.25,
    prefer_ble: bool = True,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60),
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run or simulate repeated raw angular turn calibration pulses."""
    initial_telemetry = _custom_path_telemetry_snapshot(coordinator)
    selection = _raw_angular_command_selection(
        direction=direction,
        angular_speed=angular_speed,
    )
    gates = _manual_velocity_pulse_gates(
        coordinator,
        initial_telemetry,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    runtime_safety = _runtime_motion_safety_summary(
        initial_telemetry,
        ha_state=ha_state,
        active_route=active_route,
    )
    if initial_telemetry.get("position", {}).get("toward") is None:
        gates.append(
            {
                "name": "live_heading_available",
                "passed": False,
                "detail": "Angular calibration requires live heading/toward telemetry.",
            }
        )
    if runtime_safety["active_mowing_detected"]:
        gates.append(
            {
                "name": "runtime_not_mowing",
                "passed": False,
                "detail": "Raw angular calibration is blocked while mowing is active.",
            }
        )
    if runtime_safety["active_route_status"]["blocks_motion"]:
        gates.append(
            {
                "name": "runtime_route_not_blocking",
                "passed": False,
                "detail": "Raw angular calibration is blocked by live/ambiguous route data.",
            }
        )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    target_status = _raw_angular_target_status(
        initial_telemetry,
        initial_telemetry,
        direction=direction,
        target_heading_delta_degrees=target_heading_delta_degrees,
    )
    result: dict[str, Any] = {
        "service": SERVICE_RAW_PYMAMMOTION_ANGULAR_CALIBRATION,
        "mode": "dry_run" if dry_run else "real_raw_angular_calibration",
        "dry_run": dry_run,
        "would_send": not dry_run and not blockers,
        "real_execution_scope": "raw_angular_calibration_only",
        "path_execution_allowed": False,
        "direction": direction,
        "angular_speed": angular_speed,
        "target_heading_delta_degrees": target_heading_delta_degrees,
        "max_commands": max_commands,
        "min_heading_change_degrees": min_heading_change_degrees,
        "max_translation_distance": max_translation_distance,
        "prefer_ble": prefer_ble,
        "transport_preference": "ble_preferred" if prefer_ble else "default",
        "sample_delays": list(sample_delays),
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "initial_command_selection": selection,
        "initial_telemetry": initial_telemetry,
        "final_telemetry": initial_telemetry,
        "runtime_safety": runtime_safety,
        "safety_gates": gates,
        "blockers": blockers,
        "commands_sent": 0,
        "command_results": [],
        "samples": [{"label": "initial", "telemetry": initial_telemetry}],
        "heading_diagnostics": [],
        "target_status": target_status,
        "stop_reason": None,
    }
    if not _position_available(initial_telemetry):
        result["stop_reason"] = "position_unavailable"
        return result
    if initial_telemetry.get("position", {}).get("toward") is None:
        result["stop_reason"] = "heading_unavailable"
        return result
    if dry_run:
        result["stop_reason"] = "dry_run"
        result["command_not_sent"] = {
            "manager_method": "send_command_with_args",
            "device_name": getattr(coordinator, "device_name", None),
            "command": "send_movement",
            "prefer_ble": prefer_ble,
            "kwargs": {
                "linear_speed": 0,
                "angular_speed": selection["angular_speed"],
            },
        }
        return result
    if blockers:
        result["stop_reason"] = "safety_gates_failed"
        return result

    baseline_telemetry = initial_telemetry
    for command_index in range(1, max_commands + 1):
        before = _custom_path_telemetry_snapshot(coordinator)
        result["final_telemetry"] = before
        if not _position_available(before):
            result["stop_reason"] = "position_unavailable"
            return result
        if before.get("position", {}).get("toward") is None:
            result["stop_reason"] = "heading_unavailable"
            return result
        if not _blade_reported_safe(before):
            result["stop_reason"] = "blade_unsafe"
            return result
        current_runtime_safety = _runtime_motion_safety_summary(
            before,
            ha_state=ha_state,
            active_route=active_route,
        )
        if (
            current_runtime_safety["active_mowing_detected"]
            or current_runtime_safety["active_route_status"]["blocks_motion"]
            or before.get("charge_state_label") != "not_charging"
            or before.get("work_mode_label") not in {"MODE_READY", "MODE_PAUSE"}
        ):
            result["runtime_safety"] = current_runtime_safety
            result["stop_reason"] = "mower_state_unsafe"
            return result
        quality = _manual_velocity_quality_degradation(baseline_telemetry, before)
        if quality["degraded"]:
            result["quality_degradation"] = quality
            result["stop_reason"] = "telemetry_quality_degraded"
            return result
        target_status = _raw_angular_target_status(
            baseline_telemetry,
            before,
            direction=direction,
            target_heading_delta_degrees=target_heading_delta_degrees,
        )
        result["target_status"] = target_status
        if target_status["complete"]:
            result["stop_reason"] = "target_heading_reached"
            return result

        command_result: dict[str, Any] = {
            "index": command_index,
            "attempted": True,
            "ok": None,
            "ack": None,
            "error": None,
            "duration_ms": None,
            "command": "send_movement",
            "prefer_ble": prefer_ble,
            "kwargs": {
                "linear_speed": 0,
                "angular_speed": selection["angular_speed"],
            },
            "selection": selection,
        }
        started = time.monotonic()
        try:
            await _send_manager_command_with_args(
                coordinator,
                "send_movement",
                prefer_ble=prefer_ble,
                command_kwargs=command_result["kwargs"],
            )
            command_result["ack"] = None
            command_result["ok"] = True
        except Exception as err:  # noqa: BLE001
            command_result["ok"] = False
            command_result["error"] = f"{type(err).__name__}: {err}"
        finally:
            command_result["duration_ms"] = round(
                (time.monotonic() - started) * 1000,
                3,
            )
        result["command_results"].append(command_result)
        result["commands_sent"] += 1
        if command_result["ok"] is not True:
            result["stop_reason"] = "command_failed"
            return result

        command_samples: list[dict[str, Any]] = []
        previous_delay = 0.0
        for sample_index, delay in enumerate(sample_delays):
            await asyncio.sleep(max(0.0, float(delay) - previous_delay))
            previous_delay = float(delay)
            sample_telemetry = _custom_path_telemetry_snapshot(coordinator)
            sample = {
                "label": f"command_{command_index}_sample_{sample_index + 1}_{delay:g}s",
                "command_index": command_index,
                "delay_seconds": float(delay),
                "telemetry": sample_telemetry,
            }
            result["samples"].append(sample)
            command_samples.append(sample)

        after = (
            command_samples[-1]["telemetry"]
            if command_samples
            else _custom_path_telemetry_snapshot(coordinator)
        )
        result["final_telemetry"] = after
        heading_diagnostic = _raw_angular_heading_diagnostic(
            before,
            after,
            direction=direction,
            min_heading_change_degrees=min_heading_change_degrees,
            max_translation_distance=max_translation_distance,
        )
        heading_diagnostic["command_index"] = command_index
        result["heading_diagnostics"].append(heading_diagnostic)
        target_status = _raw_angular_target_status(
            baseline_telemetry,
            after,
            direction=direction,
            target_heading_delta_degrees=target_heading_delta_degrees,
        )
        result["target_status"] = target_status
        if target_status["complete"]:
            result["stop_reason"] = "target_heading_reached"
            return result
        quality = _manual_velocity_quality_degradation(baseline_telemetry, after)
        if quality["degraded"]:
            result["quality_degradation"] = quality
            result["stop_reason"] = "telemetry_quality_degraded"
            return result
        if not _blade_reported_safe(after):
            result["stop_reason"] = "blade_unsafe"
            return result
        if heading_diagnostic["excessive_translation"]:
            result["stop_reason"] = "translation_exceeded_limit"
            return result
        if not heading_diagnostic["passed"]:
            result["stop_reason"] = "no_heading_progress"
            return result

    result["stop_reason"] = "max_commands_reached"
    return result


async def _manual_velocity_cumulative_pulse_test(  # noqa: C901
    coordinator: MammotionReportUpdateCoordinator,
    points: list[dict[str, float]],
    *,
    area_hash: int | None = None,
    speed: float = 0.4,
    pulse_duration_ms: int = 750,
    max_pulses: int = 3,
    waypoint_tolerance: float = 0.1,
    force_action: str = "auto",
    stop_mode: str = "immediate",
    stop_delay_ms: int = 0,
    heading_offset_degrees: float = 0.0,
    heading_offset_candidates: list[float] | tuple[float, ...] | None = None,
    min_progress_distance: float = 0.003,
    min_heading_change_degrees: float = 1.0,
    use_wifi: bool = True,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    cumulative_sample_delays: tuple[float, ...] = (
        0.0,
        10.0,
        20.0,
        30.0,
        45.0,
        60.0,
        90.0,
        120.0,
    ),
) -> dict[str, Any]:
    """Run or simulate a bounded pulse burst, then measure cumulative telemetry.

    Unlike the segment probes, this deliberately does not require progress after
    every pulse.  It is diagnostic-only and exists to characterize delayed mower
    telemetry after several short acknowledged movement commands.
    """
    if hasattr(coordinator, "async_start_report_stream"):
        stream_duration_ms = int(
            (
                (pulse_duration_ms / 1000)
                + max(stop_delay_ms / 1000, 0.0)
                + max(cumulative_sample_delays, default=0.0)
                + 5.0
            )
            * max_pulses
            * 1000
        )
        await coordinator.async_start_report_stream(
            duration_ms=max(10_000, stream_duration_ms)
        )

    preview = _preview_custom_path(
        coordinator,
        points,
        area_hash=area_hash,
        speed=speed,
        blade_mode="off",
    )
    normalized_points = preview["points"]
    initial_telemetry = _custom_path_telemetry_snapshot(coordinator)
    heading_candidates = _manual_velocity_heading_offset_candidates(
        heading_offset_degrees,
        heading_offset_candidates,
    )
    initial_decision = _manual_velocity_best_heading_decision(
        normalized_points,
        initial_telemetry,
        speed=speed,
        waypoint_tolerance=waypoint_tolerance,
        heading_offset_degrees=heading_offset_degrees,
        heading_offset_candidates=heading_candidates,
        max_pulse_seconds=pulse_duration_ms / 1000,
    )
    initial_decision = _manual_velocity_forced_decision(
        initial_decision,
        force_action=force_action,
        speed=speed,
    )
    gates = _manual_velocity_pulse_gates(
        coordinator,
        initial_telemetry,
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
    result: dict[str, Any] = {
        **preview,
        "service": SERVICE_MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST,
        "mode": "dry_run" if dry_run else "real_cumulative_pulse_probe",
        "dry_run": dry_run,
        "speed": speed,
        "pulse_duration_ms": pulse_duration_ms,
        "max_pulses": max_pulses,
        "waypoint_tolerance": waypoint_tolerance,
        "force_action": force_action,
        "stop_mode": stop_mode,
        "stop_delay_ms": stop_delay_ms,
        "heading_offset_degrees": heading_offset_degrees,
        "heading_offset_candidates": list(heading_candidates),
        "min_progress_distance": min_progress_distance,
        "min_heading_change_degrees": min_heading_change_degrees,
        "use_wifi": use_wifi,
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "cumulative_sample_delays": list(cumulative_sample_delays),
        "would_send": not dry_run and not blockers,
        "real_probe_allowed": not dry_run and not blockers,
        "blockers": blockers,
        "safety_gates": gates,
        "initial_telemetry": initial_telemetry,
        "initial_controller_decision": initial_decision,
        "pulse_results": [],
        "cumulative_samples": [],
        "final_telemetry": initial_telemetry,
        "cumulative_delta": _telemetry_position_delta(
            initial_telemetry,
            initial_telemetry,
        ),
        "cumulative_path_progress_diagnostic": None,
        "telemetry_latency_seconds": None,
        "pulses_sent": 0,
        "stop_reason": "dry_run" if dry_run else None,
        "result_status": "dry_run" if dry_run else None,
        "real_execution_scope": "manual_velocity_cumulative_probe_only",
        "full_path_execution_allowed": False,
    }
    if dry_run or blockers:
        result["stop_reason"] = "dry_run" if dry_run else "safety_gates_failed"
        result["result_status"] = result["stop_reason"]
        result["command_not_sent"] = initial_decision.get("command_not_sent")
        return result

    baseline_quality_telemetry = initial_telemetry
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
            result["pulse_results"].append(
                {
                    "index": index,
                    "before": before,
                    "safety_gates": gates,
                    "blockers": blockers,
                    "command_result": {"attempted": False, "ok": None, "error": None},
                    "stop_result": {"attempted": False, "ok": None, "error": None},
                }
            )
            break

        quality_degradation = _manual_velocity_quality_degradation(
            baseline_quality_telemetry,
            before,
        )
        if quality_degradation["degraded"]:
            result["stop_reason"] = "telemetry_quality_degraded"
            result["blockers"] = quality_degradation["reasons"]
            result["pulse_results"].append(
                {
                    "index": index,
                    "before": before,
                    "quality_degradation": quality_degradation,
                    "command_result": {"attempted": False, "ok": None, "error": None},
                    "stop_result": {"attempted": False, "ok": None, "error": None},
                }
            )
            break

        decision = _manual_velocity_best_heading_decision(
            normalized_points,
            before,
            speed=speed,
            waypoint_tolerance=waypoint_tolerance,
            heading_offset_degrees=heading_offset_degrees,
            heading_offset_candidates=heading_candidates,
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
            result["pulse_results"].append(
                {
                    "index": index,
                    "before": before,
                    "controller_decision": decision,
                    "command_result": {"attempted": False, "ok": None, "error": None},
                    "stop_result": {"attempted": False, "ok": None, "error": None},
                }
            )
            break

        command_result = await _manual_velocity_command_attempt(
            coordinator,
            action=action,
            speed=speed,
            use_wifi=use_wifi,
        )
        await asyncio.sleep(pulse_duration_ms / 1000)
        if stop_mode == "delayed" and stop_delay_ms > 0:
            await asyncio.sleep(stop_delay_ms / 1000)
        if stop_mode in {"immediate", "delayed"}:
            stop_result = await _manual_velocity_stop_attempt(
                coordinator,
                use_wifi=use_wifi,
            )
        else:
            stop_result = {
                "attempted": False,
                "ok": None,
                "error": None,
                "reason": "firmware_nudge_mode_no_explicit_stop",
            }
        after_stop = _custom_path_telemetry_snapshot(coordinator)
        result["pulses_sent"] += 1
        result["pulse_results"].append(
            {
                "index": index,
                "before": before,
                "after_stop": after_stop,
                "controller_decision": decision,
                "command": {
                    "service": f"{DOMAIN}.{_manual_velocity_action_service(action)}",
                    "data": {"speed": speed, "use_wifi": use_wifi},
                },
                "command_result": command_result,
                "stop_result": stop_result,
                "immediate_delta_from_initial": _telemetry_position_delta(
                    initial_telemetry,
                    after_stop,
                ),
            }
        )
        if command_result["ok"] is not True:
            result["stop_reason"] = "command_failed"
            break
        if stop_result["ok"] is not True and stop_mode != "firmware":
            result["stop_reason"] = "stop_failed"
            break

    if result["pulses_sent"] > 0:
        samples: list[dict[str, Any]] = []
        previous_delay = 0.0
        for delay in cumulative_sample_delays:
            await asyncio.sleep(max(0.0, delay - previous_delay))
            previous_delay = delay
            samples.append(
                {
                    "delay_seconds": delay,
                    "telemetry": _custom_path_telemetry_snapshot(coordinator),
                }
            )
        result["cumulative_samples"] = samples
        final_telemetry = samples[-1]["telemetry"] if samples else initial_telemetry
        result["final_telemetry"] = final_telemetry
        result["cumulative_delta"] = _telemetry_position_delta(
            initial_telemetry,
            final_telemetry,
        )
        late_progress = _manual_velocity_delayed_progress_diagnostics(
            initial_telemetry,
            samples,
            initial_decision,
            min_progress_distance=min_progress_distance,
            min_heading_change_degrees=min_heading_change_degrees,
        )
        result["cumulative_path_progress_diagnostic"] = late_progress[
            "late_path_progress_diagnostic"
        ]
        result["telemetry_latency_seconds"] = late_progress[
            "telemetry_latency_seconds"
        ]
        result["cumulative_sample_diagnostics"] = late_progress[
            "post_stop_sample_diagnostics"
        ]
        result["cumulative_progress_detected"] = late_progress[
            "late_progress_detected"
        ]
        if result["stop_reason"] is None:
            result["stop_reason"] = (
                "cumulative_progress_detected"
                if late_progress["late_progress_detected"]
                else "no_cumulative_progress"
            )
    if result["stop_reason"] is None:
        result["stop_reason"] = "no_pulses_sent"
    result["result_status"] = result["stop_reason"]
    return result


async def _experimental_execute_segment_burst(  # noqa: C901
    coordinator: MammotionReportUpdateCoordinator,
    points: list[dict[str, float]],
    *,
    area_hash: int | None = None,
    speed: float = 0.4,
    pulse_duration_ms: int = 750,
    pulses_per_burst: int = DEFAULT_EXPERIMENTAL_SEGMENT_PULSES_PER_BURST,
    max_bursts: int = DEFAULT_EXPERIMENTAL_SEGMENT_MAX_BURSTS,
    waypoint_tolerance: float = 0.1,
    heading_offset_degrees: float = 0.0,
    heading_offset_candidates: list[float] | tuple[float, ...] | None = None,
    stop_mode: str = DEFAULT_EXPERIMENTAL_SEGMENT_STOP_MODE,
    stop_delay_ms: int = 0,
    min_progress_distance: float = 0.003,
    min_heading_change_degrees: float = 1.0,
    allow_unproven_turns: bool = False,
    calibrated_forward_heading_degrees: float = DEFAULT_CALIBRATED_FORWARD_HEADING_DEGREES,
    calibrated_forward_heading_tolerance_degrees: float = DEFAULT_CALIBRATED_FORWARD_HEADING_TOLERANCE_DEGREES,
    use_wifi: bool = False,
    confirm_blades_off: bool = True,
    confirm_clear_area: bool = True,
    cumulative_sample_delays: tuple[float, ...] = (
        0.0,
        10.0,
        20.0,
        30.0,
        45.0,
        60.0,
        90.0,
        120.0,
    ),
) -> dict[str, Any]:
    """Run a bounded experimental burst and then inspect cumulative telemetry."""
    if hasattr(coordinator, "async_start_report_stream"):
        stream_duration_ms = int(
            (
                (pulse_duration_ms / 1000) * pulses_per_burst
                + (stop_delay_ms / 1000)
                + max(cumulative_sample_delays, default=0.0)
                + 5.0
            )
            * max_bursts
            * 1000
        )
        await coordinator.async_start_report_stream(
            duration_ms=max(10_000, stream_duration_ms)
        )

    preview = _preview_custom_path(
        coordinator,
        points,
        area_hash=area_hash,
        speed=speed,
        blade_mode="off",
    )
    normalized_points = preview["points"]
    initial_telemetry = _custom_path_telemetry_snapshot(coordinator)
    heading_candidates = _manual_velocity_heading_offset_candidates(
        heading_offset_degrees,
        heading_offset_candidates,
    )
    result: dict[str, Any] = {
        **preview,
        "service": SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT_BURST,
        "mode": "real_experimental_segment_burst",
        "dry_run": False,
        "speed": speed,
        "pulse_duration_ms": pulse_duration_ms,
        "pulses_per_burst": pulses_per_burst,
        "max_bursts": max_bursts,
        "waypoint_tolerance": waypoint_tolerance,
        "heading_offset_degrees": heading_offset_degrees,
        "heading_offset_candidates": list(heading_candidates),
        "stop_mode": stop_mode,
        "stop_delay_ms": stop_delay_ms,
        "min_progress_distance": min_progress_distance,
        "min_heading_change_degrees": min_heading_change_degrees,
        "allow_unproven_turns": allow_unproven_turns,
        "calibrated_forward_heading_degrees": calibrated_forward_heading_degrees,
        "calibrated_forward_heading_tolerance_degrees": (
            calibrated_forward_heading_tolerance_degrees
        ),
        "calibrated_forward_heading_diagnostic": None,
        "cumulative_sample_delays": list(cumulative_sample_delays),
        "use_wifi": use_wifi,
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "initial_telemetry": initial_telemetry,
        "final_telemetry": initial_telemetry,
        "manual_motion_execution_policy": _manual_motion_execution_policy(),
        "bursts": [],
        "bursts_sent": 0,
        "pulses_sent": 0,
        "cumulative_distance": 0.0,
        "cumulative_path_progress": 0.0,
        "completion_status": _manual_velocity_completion_status(
            normalized_points,
            initial_telemetry,
            waypoint_tolerance=waypoint_tolerance,
        ),
        "stop_reason": None,
        "real_execution_scope": "one_segment_burst_only",
        "full_path_execution_allowed": False,
    }
    if not preview["valid"]:
        result["stop_reason"] = "path_validation_failed"
        return result
    if result["completion_status"]["complete"]:
        result["stop_reason"] = "path_complete"
        return result
    segment_heading = _path_heading_degrees(normalized_points[0], normalized_points[1])
    calibrated_heading_error = _heading_error_degrees(
        calibrated_forward_heading_degrees,
        segment_heading,
    )
    result["calibrated_forward_heading_diagnostic"] = {
        "segment_heading_degrees": segment_heading,
        "calibrated_forward_heading_degrees": calibrated_forward_heading_degrees,
        "heading_error_degrees": calibrated_heading_error,
        "tolerance_degrees": calibrated_forward_heading_tolerance_degrees,
        "within_calibrated_forward_window": (
            abs(calibrated_heading_error)
            <= calibrated_forward_heading_tolerance_degrees
        ),
        "allow_unproven_turns": allow_unproven_turns,
    }
    if (
        not allow_unproven_turns
        and abs(calibrated_heading_error)
        > calibrated_forward_heading_tolerance_degrees
    ):
        result["stop_reason"] = (
            "segment_heading_outside_calibrated_forward_window"
        )
        result["blockers"] = ["unproven_turn_or_lateral_motion_required"]
        return result

    for burst_index in range(1, max_bursts + 1):
        before = _custom_path_telemetry_snapshot(coordinator)
        result["final_telemetry"] = before
        completion_status = _manual_velocity_completion_status(
            normalized_points,
            before,
            waypoint_tolerance=waypoint_tolerance,
        )
        if completion_status["complete"]:
            result["completion_status"] = completion_status
            result["stop_reason"] = "path_complete"
            break

        decision = _manual_velocity_best_heading_decision(
            normalized_points,
            before,
            speed=speed,
            waypoint_tolerance=waypoint_tolerance,
            heading_offset_degrees=heading_offset_degrees,
            heading_offset_candidates=heading_candidates,
            max_pulse_seconds=pulse_duration_ms / 1000,
        )
        if not allow_unproven_turns and decision.get("action") in {
            "turn_left",
            "turn_right",
        }:
            result["stop_reason"] = "turn_required_unproven"
            result["blockers"] = ["turn_primitive_unproven"]
            result["turn_blocked_decision"] = decision
            break

        burst_result = await _manual_velocity_cumulative_pulse_test(
            coordinator,
            normalized_points,
            area_hash=area_hash,
            speed=speed,
            pulse_duration_ms=pulse_duration_ms,
            max_pulses=pulses_per_burst,
            waypoint_tolerance=waypoint_tolerance,
            force_action="auto",
            stop_mode=stop_mode,
            stop_delay_ms=stop_delay_ms,
            heading_offset_degrees=heading_offset_degrees,
            heading_offset_candidates=heading_candidates,
            min_progress_distance=min_progress_distance,
            min_heading_change_degrees=min_heading_change_degrees,
            use_wifi=use_wifi,
            dry_run=False,
            confirm_blades_off=confirm_blades_off,
            confirm_clear_area=confirm_clear_area,
            cumulative_sample_delays=cumulative_sample_delays,
        )
        after = burst_result.get("final_telemetry", before)
        result["bursts"].append(
            {
                "index": burst_index,
                "stop_reason": burst_result.get("stop_reason"),
                "pulses_sent": burst_result.get("pulses_sent"),
                "cumulative_progress_detected": burst_result.get(
                    "cumulative_progress_detected"
                ),
                "telemetry_latency_seconds": burst_result.get(
                    "telemetry_latency_seconds"
                ),
                "cumulative_delta": burst_result.get("cumulative_delta"),
                "cumulative_path_progress_diagnostic": burst_result.get(
                    "cumulative_path_progress_diagnostic"
                ),
                "initial_position": burst_result.get("initial_telemetry", {}).get(
                    "position"
                ),
                "final_position": burst_result.get("final_telemetry", {}).get(
                    "position"
                ),
                "pulse_results": burst_result.get("pulse_results"),
            }
        )
        result["bursts_sent"] += 1
        result["pulses_sent"] += int(burst_result.get("pulses_sent") or 0)
        result["final_telemetry"] = after

        delta = burst_result.get("cumulative_delta") or {}
        if delta.get("distance") is not None:
            result["cumulative_distance"] += float(delta["distance"])
        progress_diagnostic = burst_result.get(
            "cumulative_path_progress_diagnostic"
        ) or {}
        path_progress = progress_diagnostic.get("path_progress_distance")
        if path_progress is not None and path_progress > 0:
            result["cumulative_path_progress"] += float(path_progress)

        if burst_result.get("stop_reason") in (
            "command_failed",
            "stop_failed",
            "safety_gates_failed",
            "telemetry_quality_degraded",
        ):
            result["stop_reason"] = burst_result["stop_reason"]
            result["blockers"] = burst_result.get("blockers")
            break
        if not burst_result.get("cumulative_progress_detected"):
            result["stop_reason"] = "no_cumulative_progress"
            break

        completion_status = _manual_velocity_completion_status(
            normalized_points,
            after,
            waypoint_tolerance=waypoint_tolerance,
        )
        result["completion_status"] = completion_status
        if completion_status["complete"]:
            result["stop_reason"] = "path_complete"
            break

    if result["stop_reason"] is None:
        result["completion_status"] = _manual_velocity_completion_status(
            normalized_points,
            result["final_telemetry"],
            waypoint_tolerance=waypoint_tolerance,
        )
        result["stop_reason"] = (
            "path_complete"
            if result["completion_status"]["complete"]
            else "partial_progress_timeout"
        )
    return result


async def _manual_velocity_heading_calibration_test(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    action: str = "forward",
    speed: float = 0.4,
    duration_ms: int = 750,
    stop_mode: str = "firmware",
    stop_delay_ms: int = 0,
    post_command_sample_delays: list[float] | tuple[float, ...] | None = None,
    use_wifi: bool = False,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    min_progress_distance: float = 0.003,
    min_heading_change_degrees: float = 1.0,
) -> dict[str, Any]:
    """Run or simulate a tiny movement pulse and report heading calibration data."""
    if post_command_sample_delays is None:
        post_command_sample_delays = (0, 10, 20, 30, 45, 60)
    pulse_result = await _manual_velocity_pulse_test(
        coordinator,
        action=action,
        speed=speed,
        duration_ms=duration_ms,
        stop_mode=stop_mode,
        stop_delay_ms=stop_delay_ms,
        post_command_sample_delays=post_command_sample_delays,
        use_wifi=use_wifi,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
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
        "stop_mode": stop_mode,
        "stop_delay_ms": stop_delay_ms,
        "post_command_sample_delays": list(post_command_sample_delays or []),
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
    stop_mode: str = "immediate",
    stop_delay_ms: int = 0,
    heading_offset_degrees: float = 0.0,
    heading_offset_candidates: list[float] | tuple[float, ...] | None = None,
    min_progress_distance: float = 0.003,
    no_progress_limit: int = 2,
    min_heading_change_degrees: float = 1.0,
    use_wifi: bool = True,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    pre_command_sample_delays: tuple[float, ...] = (0.0,),
    post_stop_sample_delays: tuple[float, ...] = (
        0.5,
        1.0,
        2.0,
    ),
    require_progress_each_pulse: bool = True,
    service_name: str = SERVICE_MANUAL_VELOCITY_SEGMENT_TEST,
) -> dict[str, Any]:
    """Run or simulate a guarded one-segment closed-loop movement probe.

    This intentionally remains a probe, not full path execution.  Real mode
    only sends repeated capped manual-velocity pulses and stops after each one.
    """
    if hasattr(coordinator, "async_start_report_stream"):
        stream_duration_ms = int(
            (
                max(pre_command_sample_delays, default=0.0)
                + (pulse_duration_ms / 1000)
                + max(post_stop_sample_delays, default=0.0)
                + 5.0
            )
            * max_pulses
            * 1000
        )
        await coordinator.async_start_report_stream(
            duration_ms=max(10_000, stream_duration_ms)
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
    cumulative_path_progress = 0.0

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
        "pre_command_sample_delays": list(pre_command_sample_delays),
        "require_progress_each_pulse": require_progress_each_pulse,
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
            "cumulative_path_progress": cumulative_path_progress,
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
        pre_command_samples: list[_TelemetryDelaySample] = []
        previous_delay = 0.0
        for delay in pre_command_sample_delays:
            await asyncio.sleep(max(0.0, delay - previous_delay))
            previous_delay = delay
            pre_command_samples.append(
                {
                    "delay_seconds": delay,
                    "telemetry": _custom_path_telemetry_snapshot(coordinator),
                }
            )
        before: dict[str, Any] = (
            pre_command_samples[-1]["telemetry"]
            if pre_command_samples
            else _custom_path_telemetry_snapshot(coordinator)
        )
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
                    "pre_command_samples": pre_command_samples,
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
                    "pre_command_samples": pre_command_samples,
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
                    "pre_command_samples": pre_command_samples,
                    "before": before,
                    "controller_decision": decision,
                    "command_result": {"attempted": False, "ok": None, "error": None},
                    "stop_result": {"attempted": False, "ok": None, "error": None},
                    "measured_delta": _telemetry_position_delta(before, before),
                }
            )
            break

        command_result = await _manual_velocity_command_attempt(
            coordinator,
            action=action,
            speed=speed,
            use_wifi=use_wifi,
        )
        await asyncio.sleep(pulse_duration_ms / 1000)
        stop_result = await _manual_velocity_stop_attempt(
            coordinator,
            use_wifi=use_wifi,
        )

        immediate_after_stop: dict[str, Any] = _custom_path_telemetry_snapshot(
            coordinator
        )
        post_stop_samples: list[_TelemetryDelaySample] = [
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
        after: dict[str, Any] = post_stop_samples[-1]["telemetry"]
        late_progress = _manual_velocity_delayed_progress_diagnostics(
            before,
            post_stop_samples,
            decision,
            min_progress_distance=min_progress_distance,
            min_heading_change_degrees=min_heading_change_degrees,
        )
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
        path_progress_diagnostic = _manual_velocity_path_progress_diagnostic(
            before,
            after,
            decision,
            min_progress_distance=min_progress_distance,
            min_heading_change_degrees=min_heading_change_degrees,
        )
        if path_progress_diagnostic["passed"]:
            no_progress_count = 0
        else:
            no_progress_count += 1
        if measured_delta["distance"] is not None:
            cumulative_distance += float(measured_delta["distance"])
        path_progress_distance = path_progress_diagnostic.get(
            "path_progress_distance"
        )
        if path_progress_distance is not None and path_progress_distance > 0:
            cumulative_path_progress += float(path_progress_distance)
        if measured_delta["heading_change_degrees"] is not None:
            cumulative_heading_change += abs(
                float(measured_delta["heading_change_degrees"])
            )
        result["iterations"].append(
            {
                "index": index,
                "pre_command_samples": pre_command_samples,
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
                "path_progress_diagnostic": path_progress_diagnostic,
                "late_telemetry_check": late_progress["late_telemetry_check"],
                "late_progress_detected": late_progress["late_progress_detected"],
                "late_path_progress_diagnostic": late_progress[
                    "late_path_progress_diagnostic"
                ],
                "late_measured_delta": late_progress["late_measured_delta"],
                "telemetry_latency_seconds": late_progress[
                    "telemetry_latency_seconds"
                ],
                "post_stop_sample_diagnostics": late_progress[
                    "post_stop_sample_diagnostics"
                ],
                "quality_degradation": quality_degradation,
                "no_progress_count": no_progress_count,
                "cumulative_distance": cumulative_distance,
                "cumulative_path_progress": cumulative_path_progress,
                "cumulative_heading_change_degrees": cumulative_heading_change,
            }
        )
        result["final_telemetry"] = after
        result["progress_summary"] = {
            "no_progress_count": no_progress_count,
            "cumulative_distance": cumulative_distance,
            "cumulative_path_progress": cumulative_path_progress,
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
        if require_progress_each_pulse and not path_progress_diagnostic["passed"]:
            result["stop_reason"] = "path_progress_lost"
            break
        if no_progress_count >= no_progress_limit:
            result["stop_reason"] = "no_progress_limit_reached"
            break

    if result["stop_reason"] is None:
        completion_status = _manual_velocity_completion_status(
            normalized_points,
            result["final_telemetry"],
            waypoint_tolerance=waypoint_tolerance,
        )
        result["completion_status"] = completion_status
        if completion_status["complete"]:
            result["stop_reason"] = "path_complete"
        elif cumulative_path_progress > 0:
            result["stop_reason"] = "partial_progress_timeout"
        else:
            result["stop_reason"] = "no_progress_timeout"
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
    if entity_entry is None or entity_entry.domain != "camera":
        return None

    return _get_mower_by_entity_id(hass, entity_id)


def _require_camera_mower(hass: HomeAssistant, entity_id: str) -> MammotionMowerData:
    """Return the mower backing a camera entity or raise a translated error."""
    mower = _get_camera_mower(hass, entity_id)
    if mower is not None:
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
        async def handle_directional_movement(
            call: ServiceCall,
            method_name: str = method_name,
        ) -> None:
            await handle_movement(call, method_name)

        hass.services.async_register(
            DOMAIN,
            service_name,
            handle_directional_movement,
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
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        device_data = mower.reporting_coordinator.data
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

    async def handle_export_active_route(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return _export_active_route(mower.reporting_coordinator)

    async def handle_export_runtime_state(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        state = hass.states.get(call.data[ATTR_ENTITY_ID])
        active_route = _export_active_route(mower.reporting_coordinator)
        return _export_runtime_state(
            mower.reporting_coordinator,
            ha_state=state.state if state is not None else None,
            active_route=active_route,
        )

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
            stop_mode=call.data["stop_mode"],
            stop_delay_ms=call.data["stop_delay_ms"],
            post_command_sample_delays=tuple(call.data["post_command_sample_delays"]),
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
            stop_mode=call.data["stop_mode"],
            stop_delay_ms=call.data["stop_delay_ms"],
            heading_offset_degrees=call.data["heading_offset_degrees"],
            heading_offset_candidates=call.data.get("heading_offset_candidates"),
            min_progress_distance=call.data["min_progress_distance"],
            no_progress_limit=call.data["no_progress_limit"],
            min_heading_change_degrees=call.data["min_heading_change_degrees"],
            use_wifi=call.data["use_wifi"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
        )

    async def handle_experimental_execute_segment(
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
            force_action="auto",
            heading_offset_degrees=call.data["heading_offset_degrees"],
            min_progress_distance=call.data["min_progress_distance"],
            no_progress_limit=1,
            min_heading_change_degrees=call.data["min_heading_change_degrees"],
            use_wifi=call.data["use_wifi"],
            dry_run=False,
            confirm_blades_off=True,
            confirm_clear_area=True,
            pre_command_sample_delays=(0.0, 10.0, 20.0),
            require_progress_each_pulse=True,
            service_name=SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT,
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
            pre_command_sample_delays=(0.0, 10.0, 20.0),
            require_progress_each_pulse=False,
            service_name=SERVICE_MANUAL_VELOCITY_MULTI_PULSE_TEST,
        )

    async def handle_manual_velocity_cumulative_pulse_test(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _manual_velocity_cumulative_pulse_test(
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
            min_heading_change_degrees=call.data["min_heading_change_degrees"],
            use_wifi=call.data["use_wifi"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            cumulative_sample_delays=tuple(call.data["cumulative_sample_delays"]),
        )

    async def handle_experimental_execute_segment_burst(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _experimental_execute_segment_burst(
            mower.reporting_coordinator,
            cast(list[dict[str, float]], call.data["points"]),
            area_hash=call.data.get("area_hash"),
            speed=call.data["speed"],
            pulse_duration_ms=call.data["pulse_duration_ms"],
            pulses_per_burst=call.data["pulses_per_burst"],
            max_bursts=call.data["max_bursts"],
            waypoint_tolerance=call.data["waypoint_tolerance"],
            heading_offset_degrees=call.data["heading_offset_degrees"],
            heading_offset_candidates=call.data.get("heading_offset_candidates"),
            stop_mode=call.data["stop_mode"],
            stop_delay_ms=call.data["stop_delay_ms"],
            min_progress_distance=call.data["min_progress_distance"],
            min_heading_change_degrees=call.data["min_heading_change_degrees"],
            allow_unproven_turns=call.data["allow_unproven_turns"],
            calibrated_forward_heading_degrees=call.data[
                "calibrated_forward_heading_degrees"
            ],
            calibrated_forward_heading_tolerance_degrees=call.data[
                "calibrated_forward_heading_tolerance_degrees"
            ],
            use_wifi=call.data["use_wifi"],
            confirm_blades_off=True,
            confirm_clear_area=True,
            cumulative_sample_delays=tuple(call.data["cumulative_sample_delays"]),
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
            stop_mode=call.data["stop_mode"],
            stop_delay_ms=call.data["stop_delay_ms"],
            post_command_sample_delays=tuple(call.data["post_command_sample_delays"]),
            use_wifi=call.data["use_wifi"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            min_progress_distance=call.data["min_progress_distance"],
            min_heading_change_degrees=call.data["min_heading_change_degrees"],
        )

    async def handle_raw_pymammotion_motion_probe(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _raw_pymammotion_motion_probe(
            mower.reporting_coordinator,
            command=call.data["command"],
            linear_speed=call.data["linear_speed"],
            angular_speed=call.data["angular_speed"],
            speed=call.data["speed"],
            prefer_ble=call.data["prefer_ble"],
            sample_delays=tuple(call.data["sample_delays"]),
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
        )

    async def handle_raw_pymammotion_execute_segment(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        ha_state = hass.states.get(call.data[ATTR_ENTITY_ID])
        active_route: dict[str, Any] | None = None
        try:
            active_route = _export_active_route(mower.reporting_coordinator)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("Could not export active route for raw segment: %s", err)
        return await _raw_pymammotion_execute_segment(
            mower.reporting_coordinator,
            cast(list[dict[str, float]], call.data["points"]),
            area_hash=call.data.get("area_hash"),
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            prefer_ble=call.data["prefer_ble"],
            linear_speed_fast=call.data["linear_speed_fast"],
            linear_speed_slow=call.data["linear_speed_slow"],
            max_commands=call.data["max_commands"],
            waypoint_tolerance=call.data["waypoint_tolerance"],
            min_progress_distance=call.data["min_progress_distance"],
            sample_delays=tuple(call.data["sample_delays"]),
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
        )

    async def handle_raw_pymammotion_angular_calibration(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        ha_state = hass.states.get(call.data[ATTR_ENTITY_ID])
        active_route: dict[str, Any] | None = None
        try:
            active_route = _export_active_route(mower.reporting_coordinator)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("Could not export active route for angular calibration: %s", err)
        return await _raw_pymammotion_angular_calibration(
            mower.reporting_coordinator,
            direction=call.data["direction"],
            angular_speed=call.data["angular_speed"],
            target_heading_delta_degrees=call.data["target_heading_delta_degrees"],
            max_commands=call.data["max_commands"],
            min_heading_change_degrees=call.data["min_heading_change_degrees"],
            max_translation_distance=call.data["max_translation_distance"],
            prefer_ble=call.data["prefer_ble"],
            sample_delays=tuple(call.data["sample_delays"]),
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
        )

    async def handle_raw_pymammotion_turn_to_heading(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        ha_state = hass.states.get(call.data[ATTR_ENTITY_ID])
        active_route: dict[str, Any] | None = None
        try:
            active_route = _export_active_route(mower.reporting_coordinator)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("Could not export active route for turn-to-heading: %s", err)
        return await _raw_pymammotion_turn_to_heading(
            mower.reporting_coordinator,
            target_heading_degrees=call.data["target_heading_degrees"],
            heading_tolerance_degrees=call.data["heading_tolerance_degrees"],
            angular_speed_fast=call.data["angular_speed_fast"],
            angular_speed_slow=call.data["angular_speed_slow"],
            slow_turn_threshold_degrees=call.data["slow_turn_threshold_degrees"],
            max_commands=call.data["max_commands"],
            min_heading_change_degrees=call.data["min_heading_change_degrees"],
            max_translation_distance=call.data["max_translation_distance"],
            pulse_duration_ms=call.data["pulse_duration_ms"],
            prefer_ble=call.data["prefer_ble"],
            sample_delays=tuple(call.data["sample_delays"]),
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
        )

    async def handle_raw_pymammotion_execute_vector_segment(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        ha_state = hass.states.get(call.data[ATTR_ENTITY_ID])
        active_route: dict[str, Any] | None = None
        try:
            active_route = _export_active_route(mower.reporting_coordinator)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("Could not export active route for vector segment: %s", err)
        return await _raw_pymammotion_execute_vector_segment(
            mower.reporting_coordinator,
            cast(list[dict[str, float]], call.data["points"]),
            area_hash=call.data.get("area_hash"),
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            prefer_ble=call.data["prefer_ble"],
            linear_speed_fast=call.data["linear_speed_fast"],
            linear_speed_slow=call.data["linear_speed_slow"],
            slow_linear_threshold=call.data["slow_linear_threshold"],
            max_turn_commands=call.data["max_turn_commands"],
            max_linear_commands=call.data["max_linear_commands"],
            max_linear_pulse_ceiling=call.data.get("max_linear_pulse_ceiling"),
            max_no_progress_pulses=call.data["max_no_progress_pulses"],
            linear_distance_ceiling_factor=call.data["linear_distance_ceiling_factor"],
            heading_tolerance_degrees=call.data["heading_tolerance_degrees"],
            angular_speed_fast=call.data["angular_speed_fast"],
            angular_speed_slow=call.data["angular_speed_slow"],
            slow_turn_threshold_degrees=call.data["slow_turn_threshold_degrees"],
            waypoint_tolerance=call.data["waypoint_tolerance"],
            min_progress_distance=call.data["min_progress_distance"],
            min_heading_change_degrees=call.data["min_heading_change_degrees"],
            max_turn_translation_distance=call.data["max_turn_translation_distance"],
            calibrated_forward_heading_offset_degrees=call.data[
                "calibrated_forward_heading_offset_degrees"
            ],
            turn_pulse_duration_ms=call.data["turn_pulse_duration_ms"],
            linear_pulse_duration_ms=call.data["linear_pulse_duration_ms"],
            sample_delays=tuple(call.data["sample_delays"]),
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
        )

    async def handle_raw_pymammotion_execute_multi_segment(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        ha_state = hass.states.get(call.data[ATTR_ENTITY_ID])
        active_route: dict[str, Any] | None = None
        try:
            active_route = _export_active_route(mower.reporting_coordinator)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("Could not export active route for multi segment: %s", err)
        return await _raw_pymammotion_execute_multi_segment(
            mower.reporting_coordinator,
            cast(list[dict[str, float]], call.data["points"]),
            area_hash=call.data.get("area_hash"),
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            prefer_ble=call.data["prefer_ble"],
            max_real_segments=call.data["max_real_segments"],
            linear_speed_fast=call.data["linear_speed_fast"],
            linear_speed_slow=call.data["linear_speed_slow"],
            slow_linear_threshold=call.data["slow_linear_threshold"],
            max_turn_commands=call.data["max_turn_commands"],
            max_linear_commands=call.data["max_linear_commands"],
            max_linear_pulse_ceiling=call.data.get("max_linear_pulse_ceiling"),
            max_no_progress_pulses=call.data["max_no_progress_pulses"],
            linear_distance_ceiling_factor=call.data["linear_distance_ceiling_factor"],
            heading_tolerance_degrees=call.data["heading_tolerance_degrees"],
            angular_speed_fast=call.data["angular_speed_fast"],
            angular_speed_slow=call.data["angular_speed_slow"],
            slow_turn_threshold_degrees=call.data["slow_turn_threshold_degrees"],
            waypoint_tolerance=call.data["waypoint_tolerance"],
            min_progress_distance=call.data["min_progress_distance"],
            min_heading_change_degrees=call.data["min_heading_change_degrees"],
            max_turn_translation_distance=call.data["max_turn_translation_distance"],
            calibrated_forward_heading_offset_degrees=call.data[
                "calibrated_forward_heading_offset_degrees"
            ],
            turn_pulse_duration_ms=call.data["turn_pulse_duration_ms"],
            linear_pulse_duration_ms=call.data["linear_pulse_duration_ms"],
            sample_delays=tuple(call.data["sample_delays"]),
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
        )

    async def handle_forward_two_pulse_latency_test(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        ha_state = hass.states.get(call.data[ATTR_ENTITY_ID])
        active_route: dict[str, Any] | None = None
        try:
            active_route = _export_active_route(mower.reporting_coordinator)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("Could not export active route for latency test: %s", err)
        return await _forward_two_pulse_latency_test(
            mower.reporting_coordinator,
            linear_speed=call.data["linear_speed"],
            pulse_count=call.data["pulse_count"],
            pulse_gap_seconds=call.data["pulse_gap_seconds"],
            telemetry_timeout_seconds=call.data["telemetry_timeout_seconds"],
            telemetry_sample_interval_seconds=call.data[
                "telemetry_sample_interval_seconds"
            ],
            min_position_change_distance=call.data["min_position_change_distance"],
            prefer_ble=call.data["prefer_ble"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
        )

    async def handle_position_feedback_diagnostic(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        ha_state = hass.states.get(call.data[ATTR_ENTITY_ID])
        active_route: dict[str, Any] | None = None
        try:
            active_route = _export_active_route(mower.reporting_coordinator)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug(
                "Could not export active route for position feedback diagnostic: %s",
                err,
            )
        return await _position_feedback_diagnostic(
            mower.reporting_coordinator,
            linear_speed=call.data["linear_speed"],
            pulse_count=call.data["pulse_count"],
            pulse_gap_seconds=call.data["pulse_gap_seconds"],
            refresh_wait_seconds=call.data["refresh_wait_seconds"],
            prefer_ble=call.data["prefer_ble"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
        )

    async def handle_vio_motion_probe(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        ha_state = hass.states.get(call.data[ATTR_ENTITY_ID])
        active_route: dict[str, Any] | None = None
        try:
            active_route = _export_active_route(mower.reporting_coordinator)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug(
                "Could not export active route for VIO motion probe: %s",
                err,
            )
        return await _vio_motion_probe(
            mower.reporting_coordinator,
            linear_speed=call.data["linear_speed"],
            drive_seconds=call.data["drive_seconds"],
            sample_interval_seconds=call.data["sample_interval_seconds"],
            post_stop_samples=call.data["post_stop_samples"],
            max_displacement_m=call.data["max_displacement_m"],
            prefer_ble=call.data["prefer_ble"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
        )

    async def handle_vio_turn_probe(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        ha_state = hass.states.get(call.data[ATTR_ENTITY_ID])
        active_route: dict[str, Any] | None = None
        try:
            active_route = _export_active_route(mower.reporting_coordinator)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug(
                "Could not export active route for VIO turn probe: %s",
                err,
            )
        return await _vio_turn_probe(
            mower.reporting_coordinator,
            angular_speed=call.data["angular_speed"],
            linear_speed=call.data["linear_speed"],
            drive_seconds=call.data["drive_seconds"],
            sample_interval_seconds=call.data["sample_interval_seconds"],
            post_stop_samples=call.data["post_stop_samples"],
            max_displacement_m=call.data["max_displacement_m"],
            min_heading_change_degrees=call.data["min_heading_change_degrees"],
            prefer_ble=call.data["prefer_ble"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
        )

    async def handle_vio_turn_to_heading(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        ha_state = hass.states.get(call.data[ATTR_ENTITY_ID])
        active_route: dict[str, Any] | None = None
        try:
            active_route = _export_active_route(mower.reporting_coordinator)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug(
                "Could not export active route for VIO turn-to-heading: %s",
                err,
            )
        return await _vio_turn_to_heading(
            mower.reporting_coordinator,
            target_vision_heading=call.data["target_vision_heading"],
            heading_tolerance_degrees=call.data["heading_tolerance_degrees"],
            angular_speed=call.data["angular_speed"],
            pulse_duration_ms=call.data["pulse_duration_ms"],
            slow_pulse_duration_ms=call.data["slow_pulse_duration_ms"],
            slow_threshold_degrees=call.data["slow_threshold_degrees"],
            refresh_wait_seconds=call.data["refresh_wait_seconds"],
            max_commands=call.data["max_commands"],
            min_progress_degrees=call.data["min_progress_degrees"],
            max_displacement_m=call.data["max_displacement_m"],
            invert_direction=call.data["invert_direction"],
            prefer_ble=call.data["prefer_ble"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
        )

    async def handle_raw_motion_readiness_test(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        ha_state = hass.states.get(call.data[ATTR_ENTITY_ID])
        active_route: dict[str, Any] | None = None
        try:
            active_route = _export_active_route(mower.reporting_coordinator)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("Could not export active route for readiness test: %s", err)
        return await _raw_motion_readiness_test(
            mower.reporting_coordinator,
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            prefer_ble=call.data["prefer_ble"],
            max_real_steps=call.data["max_real_steps"],
            sample_delays=tuple(call.data["sample_delays"]),
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
        )

    async def handle_raw_vector_readiness_test(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        ha_state = hass.states.get(call.data[ATTR_ENTITY_ID])
        active_route: dict[str, Any] | None = None
        try:
            active_route = _export_active_route(mower.reporting_coordinator)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("Could not export active route for vector readiness: %s", err)
        return await _raw_vector_readiness_test(
            mower.reporting_coordinator,
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            prefer_ble=call.data["prefer_ble"],
            max_real_steps=call.data["max_real_steps"],
            target_distance=call.data["target_distance"],
            turn_delta_degrees=call.data["turn_delta_degrees"],
            calibrated_forward_heading_offset_degrees=call.data[
                "calibrated_forward_heading_offset_degrees"
            ],
            max_turn_commands=call.data["max_turn_commands"],
            max_linear_commands=call.data["max_linear_commands"],
            sample_delays=tuple(call.data["sample_delays"]),
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
        )

    async def handle_svg_add(call: ServiceCall) -> dict[str, Any]:
        from pymammotion.utility.svg import build_svg_for_area  # noqa: PLC0415

        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        coordinator = mower.reporting_coordinator
        device_data = coordinator.data
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
        from pymammotion.utility.svg import build_svg_update  # noqa: PLC0415

        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        coordinator = mower.reporting_coordinator
        device_data = coordinator.data
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
        SERVICE_EXPORT_RUNTIME_STATE,
        handle_export_runtime_state,
        schema=GEOJSON_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_EXPORT_ACTIVE_ROUTE,
        handle_export_active_route,
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
        SERVICE_MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST,
        handle_manual_velocity_cumulative_pulse_test,
        schema=MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT,
        handle_experimental_execute_segment,
        schema=EXPERIMENTAL_EXECUTE_SEGMENT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT_BURST,
        handle_experimental_execute_segment_burst,
        schema=EXPERIMENTAL_EXECUTE_SEGMENT_BURST_SCHEMA,
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
        SERVICE_RAW_PYMAMMOTION_MOTION_PROBE,
        handle_raw_pymammotion_motion_probe,
        schema=RAW_PYMAMMOTION_MOTION_PROBE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_PYMAMMOTION_EXECUTE_SEGMENT,
        handle_raw_pymammotion_execute_segment,
        schema=RAW_PYMAMMOTION_EXECUTE_SEGMENT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_PYMAMMOTION_ANGULAR_CALIBRATION,
        handle_raw_pymammotion_angular_calibration,
        schema=RAW_PYMAMMOTION_ANGULAR_CALIBRATION_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_PYMAMMOTION_TURN_TO_HEADING,
        handle_raw_pymammotion_turn_to_heading,
        schema=RAW_PYMAMMOTION_TURN_TO_HEADING_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT,
        handle_raw_pymammotion_execute_vector_segment,
        schema=RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT,
        handle_raw_pymammotion_execute_multi_segment,
        schema=RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_FORWARD_TWO_PULSE_LATENCY_TEST,
        handle_forward_two_pulse_latency_test,
        schema=FORWARD_TWO_PULSE_LATENCY_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_POSITION_FEEDBACK_DIAGNOSTIC,
        handle_position_feedback_diagnostic,
        schema=POSITION_FEEDBACK_DIAGNOSTIC_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_VIO_MOTION_PROBE,
        handle_vio_motion_probe,
        schema=VIO_MOTION_PROBE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_VIO_TURN_PROBE,
        handle_vio_turn_probe,
        schema=VIO_TURN_PROBE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_VIO_TURN_TO_HEADING,
        handle_vio_turn_to_heading,
        schema=VIO_TURN_TO_HEADING_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_MOTION_READINESS_TEST,
        handle_raw_motion_readiness_test,
        schema=RAW_MOTION_READINESS_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_VECTOR_READINESS_TEST,
        handle_raw_vector_readiness_test,
        schema=RAW_VECTOR_READINESS_TEST_SCHEMA,
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
            mower_base = mower[0].data.map.plan[mower[1]]
            await mower[0].async_edit_mower_task(
                _build_mower_plan(dict(call.data), mower_base)
            )
            return
        if (spino := _resolve_spino_task(hass, entity_id)) is not None:
            spino_base = spino[0].data.plans[spino[1]]
            await spino[0].async_edit_spino_task(
                _build_spino_plan(dict(call.data), spino_base)
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
