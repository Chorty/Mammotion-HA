"""Mammotion services."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import math
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
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
from pymammotion.messaging.command_queue import Priority
from pymammotion.transport.base import (
    CommandTimeoutError,
    ConcurrentRequestError,
    TransportType,
)
from pymammotion.utility.constant.device_constant import (
    PosType,
    camera_brightness,
    device_connection,
    device_mode,
)
from pymammotion.utility.device_type import DeviceType

from .backend_capability import async_probe_backend_capabilities
from .capabilities import capability_snapshot
from .const import CONF_ENABLE_EXPERIMENTAL_MOTION, DOMAIN, LOGGER
from .continuous_controller import (
    ContinuousControllerConfig,
    ContinuousObservation,
    ContinuousRoute,
    HeadingEvidence,
    alignment_feasibility,
    blind_acquisition_feasibility,
    continuous_control_decision,
    course_from_position_chord,
    normalize_degrees,
    point_in_polygon,
    polygon_boundary_clearance,
    polygon_is_valid,
)
from .continuous_controller import (
    Point as ContinuousPoint,
)
from .coordinator import (
    MammotionReportUpdateCoordinator,
    MammotionRTKCoordinator,
    MammotionSpinoCoordinator,
)
from .manual_motion import (
    REAL_CLICK_TO_GO_SEGMENT_LIMIT,
    ManualMotionCancelledError,
    ManualMotionSession,
    active_motion_session,
    assert_session_can_dispatch,
    experimental_motion_enabled,
    experimental_motion_status,
    record_completed_dispatch,
)

if TYPE_CHECKING:
    from . import MammotionConfigEntry
from .geojson_utils import apply_geojson_offset
from .models import MammotionMowerData

SERVICE_GET_GEOJSON = "get_geojson"
SERVICE_FORCE_MAP_RESYNC = "force_map_resync"
SERVICE_GET_MOW_PATH_GEOJSON = "get_mow_path_geojson"
SERVICE_GET_MOW_PROGRESS_GEOJSON = "get_mow_progress_geojson"
SERVICE_GET_MAP_DATA = "get_map_data"
SERVICE_GET_TASKS = "get_tasks"
SERVICE_GET_AREAS = "get_areas"
SERVICE_EXPORT_MAP = "export_map"
SERVICE_EXPORT_TASKS = "export_tasks"
SERVICE_EXPORT_RUNTIME_STATE = "export_runtime_state"
#: 🔒 ONE-WAY BY DESIGN. There is deliberately no matching "arm" service.
#:
#: Arming stays behind the options flow, which is a human sitting in front of
#: Settings. A service that could arm would let any automation, script, scene or
#: voice assistant open the motion gate, which is a strictly larger attack
#: surface than exists today. This one can only ever CLOSE it, so the worst a
#: bug or a stray call can do is refuse to move.
#:
#: It exists because the gate was found armed at rest three times on
#: 2026-08-18, once with zero blockers and the mower off its dock -- no docked
#: position to fall back on. Nothing in HA could close it: the gate is a config
#: entry option, so there was no entity to toggle and no service to call, only a
#: binary_sensor reporting the state.
SERVICE_DISARM_EXPERIMENTAL_MOTION = "disarm_experimental_motion"
SERVICE_EXPORT_ACTIVE_ROUTE = "export_active_route"
SERVICE_VALIDATE_CUSTOM_PATH = "validate_custom_path"
SERVICE_PREVIEW_CUSTOM_PATH = "preview_custom_path"
SERVICE_DRY_RUN_CUSTOM_PATH = "dry_run_custom_path"
SERVICE_MANUAL_VELOCITY_PULSE_TEST = "manual_velocity_pulse_test"
SERVICE_MANUAL_VELOCITY_SEGMENT_TEST = "manual_velocity_segment_test"
SERVICE_MANUAL_VELOCITY_MULTI_PULSE_TEST = "manual_velocity_multi_pulse_test"
SERVICE_MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST = "manual_velocity_cumulative_pulse_test"
SERVICE_MANUAL_VELOCITY_HEADING_CALIBRATION_TEST = (
    "manual_velocity_heading_calibration_test"
)
SERVICE_RAW_PYMAMMOTION_MOTION_PROBE = "raw_pymammotion_motion_probe"
SERVICE_CONTINUOUS_MOTION_WINDOW = "continuous_motion_window"
SERVICE_HEADING_ACQUISITION_WINDOW = "heading_acquisition_window"
SERVICE_STEP_RESPONSE_PROBE = "raw_pymammotion_step_response_probe"
SERVICE_RAW_PYMAMMOTION_EXECUTE_SEGMENT = "raw_pymammotion_execute_segment"
SERVICE_RAW_PYMAMMOTION_ANGULAR_CALIBRATION = "raw_pymammotion_angular_calibration"
SERVICE_RAW_PYMAMMOTION_TURN_TO_HEADING = "raw_pymammotion_turn_to_heading"
SERVICE_RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT = (
    "raw_pymammotion_execute_vector_segment"
)
SERVICE_RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT = "raw_pymammotion_execute_multi_segment"
SERVICE_FORWARD_TWO_PULSE_LATENCY_TEST = "forward_two_pulse_latency_test"
SERVICE_POSITION_FEEDBACK_DIAGNOSTIC = "position_feedback_diagnostic"
SERVICE_REPORT_STREAM_PROBE = "report_stream_probe"
SERVICE_REPORT_STREAM_SEQUENCE_PROBE = "report_stream_sequence_probe"
SERVICE_BASESTATION_INFO_PROBE = "basestation_info_probe"
SERVICE_OTA_INFO_PROBE = "ota_info_probe"
SERVICE_VIO_MOTION_PROBE = "vio_motion_probe"
SERVICE_VIO_TURN_PROBE = "vio_turn_probe"
SERVICE_VIO_TURN_TO_HEADING = "vio_turn_to_heading"
SERVICE_RAW_MOTION_READINESS_TEST = "raw_motion_readiness_test"
SERVICE_RAW_VECTOR_READINESS_TEST = "raw_vector_readiness_test"
SERVICE_STOP_MANUAL_MOTION = "stop_manual_motion"
SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT = "experimental_execute_segment"
SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT_BURST = "experimental_execute_segment_burst"
SERVICE_SVG_ADD = "svg_add"
SERVICE_SVG_UPDATE = "svg_update"
SERVICE_SVG_DELETE = "svg_delete"
SERVICE_REFRESH_STREAM = "refresh_stream"
SERVICE_START_VIDEO = "start_video"
SERVICE_STOP_VIDEO = "stop_video"
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

#: Safety gates the operator may deliberately override, name -> metadata.
#:
#: Added 2026-08-19 at the operator's explicit request: every blocker the card
#: can show gets a toggle, so a restriction can be lifted ON PURPOSE instead of
#: being worked around by editing constants. This is a bespoke tool for one
#: yard (standing decision 1), the operator supervises every real run, and an
#: override is recorded in the run JSON -- which is strictly better than the
#: previous options, which were "edit a constant and redeploy" or "do not run".
#:
#: ⚠️ MUST BE DEFINED ABOVE THE SCHEMAS. The service schemas ~700 lines below
#: validate against `sorted(_OVERRIDABLE_GATES)`, and a constant defined with
#: the other constants ~11,000 lines down is a NameError at import -- the exact
#: trap documented on `_MAX_SEGMENT_LENGTH_M` and the 6.10 literal.
#:
#: `tier` orders the card's presentation and nothing else; the backend treats
#: every entry identically. `why` is rendered to the operator at the moment of
#: flipping the toggle, because a gate's NAME never says what it was protecting.
_OVERRIDABLE_GATES: dict[str, dict[str, str]] = {
    # -- chosen authorization numbers: someone picked these, they are not
    #    measurements, and overriding one is a research decision -------------
    "segment_too_long": {
        "tier": "cap",
        "why": (
            "6.10 m is an authorization cap chosen 2026-08-17, not a measured "
            "limit. The longest segment ever executed is 4.0 m (n = 1)."
        ),
    },
    "real_segment_limit": {
        "tier": "cap",
        "why": (
            "4 segments per click. Segment 3+ was never executed until beta33; "
            "error does not compound with segment index (slope +0.017 m)."
        ),
    },
    "split_exceeds_real_segment_budget": {
        "tier": "cap",
        "why": (
            "Sub-legs after the collinear split exceed the 4-segment budget. "
            "Overriding runs more legs than a click has ever driven."
        ),
    },
    "linear_budget_insufficient_for_segment": {
        "tier": "cap",
        "why": (
            "The configured pulse ceiling cannot reach this leg at the "
            "conservative 0.30 m/pulse. Overriding risks stranding mid-leg on "
            "max_linear_pulse_ceiling_reached -- which stops safely."
        ),
    },
    "point_count_2_to_8": {
        "tier": "cap",
        "why": "The executor chain accepts 2 to 8 points. Untested beyond that.",
    },
    "max_real_segments_positive": {
        "tier": "cap",
        "why": "A real run with max_real_segments < 1 executes nothing.",
    },
    "one_segment_only": {
        "tier": "cap",
        "why": "This executor was validated on a single segment.",
    },
    # -- night: the tightest caps in the codebase, and the least evidenced ---
    "night_segment_too_long": {
        "tier": "night",
        "why": (
            "Night is capped at 1.0 m because the refreshed turn quantum is "
            "48.15 deg +/- 5.70 with NOTHING scaling it -- 4 of 5 converging "
            "night turns landed inside tolerance by luck (margins 1.72 / 1.09 "
            "/ 0.36 deg). The cap is what keeps a coarse controller's error "
            "bounded. No night landing-accuracy population exists."
        ),
    },
    "night_multi_segment_unsupported": {
        "tier": "night",
        "why": (
            "⚠️ NIGHT HAS NO JUNCTION FEASIBILITY MODEL AT ALL. The preflight "
            "that refuses an impossible turn before segment 1 does not exist "
            "for night, so an infeasible junction is discovered AFTER motion "
            "has started."
        ),
    },
    "night_linear_loop_unsupported": {
        "tier": "night",
        "why": (
            "Night runs a fixed pulse budget. Loop-to-tolerance at night has "
            "never been exercised."
        ),
    },
    "night_requires_precise_rtk": {
        "tier": "night",
        "why": (
            "Night steers on RTK position alone -- there is no VIO to fall "
            "back on. Float produced a 13.9 cm stationary jump on 2026-08-07."
        ),
    },
    # -- sensing: proceed on degraded, stale or absent measurement -----------
    "rtk_not_precise": {
        "tier": "sensing",
        "why": (
            "Non-Fix RTK. Float produced a 13.9 cm stationary jump against an "
            "0.08 m tolerance (2026-08-07). Equivalent to allow_degraded_rtk."
        ),
    },
    "path_validation": {
        "tier": "sensing",
        "why": (
            "⚠️ CONTAINMENT. The path leaves every known area polygon. "
            "Overriding lets the mower drive outside mapped geometry entirely."
        ),
    },
    "position_not_valid_for_motion": {
        "tier": "sensing",
        "why": (
            "Position is not valid for motion -- typically docked, CHARGE_ON, "
            "or zone_hash 0. The dock is outside every mowing area."
        ),
    },
    "live_map_position_available": {
        "tier": "sensing",
        "why": "No live map position. The controller steers on position.",
    },
    "map_position_nonzero": {
        "tier": "sensing",
        "why": "Position reads (0, 0) -- usually a dead or unstarted feed.",
    },
    "position_area_inside": {
        "tier": "sensing",
        "why": "The mower does not report itself inside a known area.",
    },
    "vio_feed_live": {
        "tier": "sensing",
        "why": (
            "⚠️ THE DUSK LATCH. vio_state reads active while tracked_features "
            "is 0 -- the state field lies and the feed is already blind. "
            "Overriding steers a VIO turn on a sensor reporting nothing."
        ),
    },
    "vio_active": {
        "tier": "sensing",
        "why": "VIO is not active. The vio turn mode closes on VIO heading.",
    },
    "live_heading_available": {
        "tier": "sensing",
        "why": "No trustworthy current heading. Frozen course-over-ground is not orientation.",
    },
    "vio_heading_available": {
        "tier": "sensing",
        "why": "No VIO heading available to close the turn loop on.",
    },
    "target_heading_available": {
        "tier": "sensing",
        "why": "No target heading could be derived for this segment.",
    },
    # -- link: the command path itself --------------------------------------
    "ble_transport_required": {
        "tier": "link",
        "why": (
            "Not on BLE. The position feed is BLE-only and stone dead on "
            "cloud; a 30-min cloud window produced exactly ONE report."
        ),
    },
    "ble_link_live": {
        "tier": "link",
        "why": (
            "⚠️ The BLE link is not live. is_usable is routing eligibility, "
            "NOT liveness -- it stays true while commands pile up undelivered "
            "and 'command ok' never proved delivery. Overriding dispatches "
            "into a link that may not carry the STOP either."
        ),
    },
    # -- physical: the mower is doing something else, or is unsafe -----------
    "mower_reports_blades_off": {
        "tier": "physical",
        "why": (
            "🚨 THE MOWER REPORTS ITS BLADES ARE NOT OFF. Note the blade RPM "
            "register latches after a mow, so this can be stale -- but it can "
            "also be true. Confirm the blades physically before overriding."
        ),
    },
    "mower_ready": {
        "tier": "physical",
        "why": "Work mode is not MODE_READY or MODE_PAUSE.",
    },
    "not_docked_or_charging": {
        "tier": "physical",
        "why": "The mower reports charging. Motion while docked can damage the dock.",
    },
    "runtime_not_mowing": {
        "tier": "physical",
        "why": (
            "🚨 AN AUTONOMOUS MOW IS ACTIVE. Overriding commands manual motion "
            "into a running vendor job."
        ),
    },
    "runtime_route_not_blocking": {
        "tier": "physical",
        "why": "Live or ambiguous route data indicates the mower is executing a route.",
    },
}

#: Gates deliberately absent from the registry above, and why. Not a safety
#: veto -- the operator asked for every blocker and got it. These three are
#: INCOHERENT to override rather than merely risky:
#:
#: * ``stop_primitive_available`` -- ``hasattr(coordinator,
#:   "async_stop_manual_motion")``. If it is False the code has no way to stop
#:   the mower; an override does not create the method, it just dispatches
#:   motion with no stop path.
#: * ``turn_mode_valid`` -- the turn mode is not one the executor implements,
#:   so there is no code path to run.
#: * ``operator_confirmed_blades_off`` / ``operator_confirmed_clear_area`` --
#:   these ARE the operator's deliberate act. The card already exposes them as
#:   checkboxes; an "override" of a confirmation is simply not confirming, and
#:   would remove the last human step rather than add one.
#:
#: The experimental-motion gate itself is likewise not here: it is the ARMING
#: control, not a blocker. Overriding it would mean motion without arming,
#: which is the one thing the whole gate exists to prevent.
_NON_OVERRIDABLE_GATES: frozenset[str] = frozenset(
    {
        "stop_primitive_available",
        "turn_mode_valid",
        "operator_confirmed_blades_off",
        "operator_confirmed_clear_area",
    }
)


def _apply_safety_overrides(
    gates: list[dict[str, Any]],
    overrides: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Force-pass the named gates, recording exactly what was overridden.

    An overridden gate keeps its original verdict in ``original_passed`` and is
    marked ``overridden: True``, so the run JSON shows both that the gate FIRED
    and that the operator chose to proceed. Silently flipping ``passed`` would
    make an overridden run indistinguishable from a clean one, which is the
    failure this project has been bitten by repeatedly.

    Returns a summary; ``gates`` is mutated in place.
    """
    requested = [str(name) for name in (overrides or [])]
    applied: list[dict[str, str]] = []
    unused: list[str] = []
    refused: list[str] = []

    firing = {gate["name"] for gate in gates if not gate.get("passed")}
    for name in requested:
        if name in _NON_OVERRIDABLE_GATES or name not in _OVERRIDABLE_GATES:
            # Fail closed on a name we do not recognise. A typo must not read
            # as a granted override.
            refused.append(name)
        elif name not in firing:
            # Requested but the gate passed anyway -- recorded so the run JSON
            # does not imply the override did something.
            unused.append(name)

    for gate in gates:
        name = gate["name"]
        if gate.get("passed"):
            continue
        if name not in requested:
            continue
        if name in refused:
            continue
        gate["original_passed"] = False
        gate["overridden"] = True
        gate["passed"] = True
        applied.append({"name": name, **_OVERRIDABLE_GATES[name]})

    return {
        "requested": requested,
        "applied": applied,
        "applied_names": [item["name"] for item in applied],
        "unused": unused,
        "refused": refused,
        "any_applied": bool(applied),
    }


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
        vol.Required("points"): vol.All(cv.ensure_list, [_CUSTOM_PATH_POINT_SCHEMA]),
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
        vol.Required("points"): vol.All(cv.ensure_list, [_CUSTOM_PATH_POINT_SCHEMA]),
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

MANUAL_VELOCITY_PULSE_TEST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("action", default="forward"): vol.In(
            ["forward", "backward", "turn_left", "turn_right"]
        ),
        # `speed` is an app-scale stick fraction; the coordinator runs it through
        # the app's rocker transform (15% deadband, x10), so a forward pulse
        # resolves to raw ``(speed*100 - 15) * 10``. The default 0.55 -> raw 400,
        # matching the linear speed the click-to-path executors send, so the B1
        # A/B measures the same pulse click-to-path actually drives. The cap 0.6
        # (raw 450) keeps a diagnostic service from commanding more than a small
        # margin over that. Below ~0.16 the deadband yields raw 0 (a no-op) --
        # the old 0.1 default silently produced zero motion.
        vol.Optional("speed", default=0.55): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.6)
        ),
        # Duration must reach the taped ~3s minimum that actually moves this
        # hardware (2s -> 0", 4s -> 4"); the max mirrors the vector/multi
        # executors (4000). The old 750 cap made every pulse a physical no-op and
        # rejected the documented 4000ms B1 call outright.
        vol.Optional("duration_ms", default=3500): vol.All(
            vol.Coerce(int), vol.Range(min=50, max=4000)
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
        # App-parity motion cadence (see RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA).
        # This service is the bare-pulse A/B harness for it: same action and
        # duration, 0 vs 200, tape-measured.
        vol.Optional("motion_refresh_interval_ms", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=1000)
        ),
        vol.Optional(
            "use_wifi", default=DEFAULT_EXPERIMENTAL_SEGMENT_USE_WIFI
        ): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

MANUAL_VELOCITY_SEGMENT_TEST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("points"): vol.All(cv.ensure_list, [_CUSTOM_PATH_POINT_SCHEMA]),
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
        # The handler reads call.data["stop_mode"]/["stop_delay_ms"]; without these
        # defaults any call omitting them raises KeyError -> HTTP 500 (same class
        # as the multi-segment ble_auto_recover bug).
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
        vol.Required("points"): vol.All(cv.ensure_list, [_CUSTOM_PATH_POINT_SCHEMA]),
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
        vol.Optional(
            "cumulative_sample_delays", default=[0, 10, 20, 30, 45, 60, 90, 120]
        ): vol.All(
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

# The historic open-loop window cap, still the ceiling whenever no distance
# guard is supplied. Measured: 4000 ms at linear 400 travels ~1.1 m.
_PROBE_DURATION_MS_WITHOUT_TRAVEL_GUARD_MAX = 4000
# With a distance guard the window may run longer, because distance -- not time
# -- is then what bounds it. Still finite so a wedged guard cannot drive forever.
_PROBE_DURATION_MS_MAX = 12_000
# Hard ceiling on the guard itself, so no caller can request an unbounded drive.
_PROBE_MAX_TRAVEL_M_CEILING = 3.0
# How far past `max_travel_m` the guard lets the mower run before the stop lands.
#
# ⚠️ **RAISED 0.35 -> 0.50 on 2026-08-23 after review.** Both firings overshot by
# 0.276 m and 0.307 m, which looked comfortably inside 0.35 -- but decomposing
# them shows why that was luck. Post-trip travel is a stable ~0.20 m; the rest is
# where the bound-crossing happens to fall inside a ~0.26 m position chord, which
# is roughly uniform on [0, 0.26]. Overshoot therefore exceeds 0.35 whenever that
# phase exceeds 0.15 -- about **42% of the time**. Both samples drew from the low
# half. 0.50 covers the whole chord.
_PROBE_TRAVEL_GUARD_OVERSHOOT_M = 0.50
# Trip the guard when the report stamp has not advanced for this long: a
# bit-identical feed is a DEAD FEED, not a stopped mower, and a distance guard
# reading a dead feed measures zero travel forever. Matches the Phase 1
# analyzer's own 2000 ms position-arrival gap limit.
_PROBE_FEED_STALE_ABORT_MS = 2000.0
# Most position payloads the travel guard will consume in a single sampler poll.
# The guard MUST drain rather than take one per poll, or `max_travel_m` goes soft
# whenever payloads outpace polls -- but the drain must be BOUNDED, because an
# unbounded loop hangs against any queue that never raises QueueEmpty (every
# mocked stream, and a live feed that refilled faster than we drain). 64 matches
# `_SAFETY_POSITION_STREAM_MAXSIZE`, so a full queue drains in one poll while the
# ~1 Hz feed never produces more than ~1 payload per poll at any legal
# `sample_interval_ms`.
_PROBE_MAX_DRAIN_PER_POLL = 64
# Metres per second per unit of `linear_speed`, for sizing clock-bound corridor
# clearance. It exists to make a corridor big enough, NOT to be accurate, so it
# is always rounded UP -- too low is the unsafe direction.
#
# 🚨 RAISED 7.0e-04 -> 7.5e-04 on 2026-09-03. The old value was fitted to
# ramp-INCLUSIVE window averages and understated SUSTAINED travel, which is what
# a long window actually accumulates. Post-ramp speeds, measured directly:
#     linear 300 -> 0.223 m/s  (Phase A, 2026-09-03)  => 7.43e-04
#     linear 400 -> 0.295 m/s  (2026-09-01 run)       => 7.38e-04
# and 0.295 independently matches the 0.280-0.293 m/s measured during arcs on
# 2026-08-12. At 7.0e-04 the constant sat ~6% BELOW the measured value at 300.
#
# 🔑 Those two points also show the command/speed relation is essentially LINEAR
# in the sustained regime: 0.223/0.295 = 0.756 against a command ratio of 0.750.
# ⚠️ The long-standing "a 25% command cut gave a 39% speed cut" was an artifact
# of comparing 4 s ramp-inclusive averages, where the slower run spends a larger
# fraction of its window ramping. Do not quote it as a property of the drivetrain.
_PROBE_SPEED_PER_LINEAR_UNIT_MS = 7.5e-04

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
        # beta45. Without a refresh the h-watchdog stops the motor almost
        # immediately, so a single-shot probe moves ~10 cm however long the
        # window is (measured 2026-07-22: 4 in single-shot vs 44 in at refresh
        # 200, same 4 s command). That is enough to prove a command actuates and
        # nothing more -- and it is specifically NOT enough to characterise an
        # ARC, which is the one motion this project has never sent: every one of
        # its 55 send_movement call sites is single-axis, though DrvMotionCtrl
        # has always taken both. See docs/night-motion-options-20260811.md.
        vol.Optional("motion_refresh_interval_ms", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=1000)
        ),
        # Phase-1 continuous-controller feasibility instrumentation. Zero keeps
        # every existing probe equivalent: no stream startup and no sampler.
        vol.Optional("in_window_sample_interval_ms", default=0): vol.All(
            vol.Coerce(int),
            vol.Any(0, vol.Range(min=50, max=1000)),
        ),
        # This probe has no closed loop and no waypoint, so historically the
        # only thing limiting travel was the window -- 4000 ms at the app-parity
        # cadence is roughly 1.1 m measured. Windows ABOVE 4000 ms are allowed
        # only together with `max_travel_m`, which bounds the real safety
        # property (distance) instead of a time proxy for it. See
        # `_raw_pymammotion_motion_probe` and the
        # `duration_over_4000ms_requires_max_travel_m` blocker.
        vol.Optional("duration_ms", default=1300): vol.All(
            vol.Coerce(int), vol.Range(min=50, max=_PROBE_DURATION_MS_MAX)
        ),
        # In-window distance guard. Zero disables it and pins duration_ms to the
        # historic 4000 ms ceiling. A positive value aborts the window as soon
        # as the sampled position has moved this far from where it started.
        vol.Optional("max_travel_m", default=0.0): vol.All(
            vol.Coerce(float),
            vol.Any(0.0, vol.Range(min=0.10, max=_PROBE_MAX_TRAVEL_M_CEILING)),
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

_CONTINUOUS_MOTION_POINT_SCHEMA = vol.Schema(
    {
        vol.Required("x"): vol.Coerce(float),
        vol.Required("y"): vol.Coerce(float),
    }
)

STEP_RESPONSE_PROBE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("route_start"): _CONTINUOUS_MOTION_POINT_SCHEMA,
        vol.Required("corridor_polygon"): [_CONTINUOUS_MOTION_POINT_SCHEMA],
        # 300 became admissible with rule E-VIO. It was eliminated on
        # 2026-08-30 because a 0.116 m/s mower covers less than the 0.15 m
        # chord the RTK-course statistic needed -- ⚠️ and that 0.116 was a 4 s
        # ramp-inclusive average, not a speed: 300 SUSTAINS 0.223 m/s
        # (2026-09-03), which clears the 0.15 m chord comfortably, so the
        # original elimination was unsound on its own terms too. E-VIO reads
        # VIO heading
        # between consecutive DISTINCT readings and imposes no travel floor at
        # all, so the objection does not transfer. Driving slower is what lets a
        # long step phase fit the yard: see
        # docs/findings-plus180-split-is-onset-sampling-phase-20260901.md.
        vol.Optional("linear_speed", default=400): vol.All(
            vol.Coerce(int), vol.In([300, 400])
        ),
        # 🚨 MEASURED VALUES ONLY, both signs. The turn rate is characterised
        # across angular 120-180; anything smaller is unmeasured rather than
        # safer and may sit in an actuation deadband. Both signs are offered
        # because a one-sided step cannot distinguish carryover from a
        # direction-dependent drivetrain asymmetry.
        vol.Optional("step_angular_speed", default=120): vol.All(
            vol.Coerce(int), vol.In([-180, -120, 120, 180])
        ),
        vol.Optional("baseline_ms", default=3000): vol.All(
            vol.Coerce(int), vol.Range(min=1000, max=5000)
        ),
        # Raised 7000 -> 15000 on 2026-09-01. 2a's half-phase split gives the
        # single onset-contaminated interval ~1/k of the first half's weight, so
        # half_diff ~= |steady - onset| / k. At the worst observed contamination
        # (10.43 deg/s) the bound needs k >= 7, i.e. ~14 informative step
        # intervals at the ~1 Hz VIO cadence. ⚠️ This buys TIME, not distance:
        # `max_travel_m` is unchanged and a window that cannot fit it is refused
        # up front by `step_window_travel_exceeds_budget`.
        vol.Optional("step_ms", default=3000): vol.All(
            vol.Coerce(int), vol.Range(min=1000, max=15000)
        ),
        # The settle phase IS the experiment: it must outlast the carryover it
        # is measuring, or tau is censored rather than measured.
        vol.Optional("settle_ms", default=4000): vol.All(
            vol.Coerce(int), vol.Range(min=1000, max=6000)
        ),
        vol.Optional("motion_refresh_interval_ms", default=200): vol.All(
            vol.Coerce(int), vol.Range(min=50, max=1000)
        ),
        vol.Optional("sample_interval_ms", default=100): vol.All(
            vol.Coerce(int), vol.Range(min=50, max=1000)
        ),
        vol.Optional("max_travel_m", default=2.50): vol.All(
            vol.Coerce(float), vol.Range(min=0.10, max=4.5)
        ),
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
        vol.Optional("confirm_step_response_run", default=False): cv.boolean,
    }
)

CONTINUOUS_MOTION_WINDOW_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        # A frozen, offline-scanned route -- see scripts/freeze_phase1_corridors.py
        # and scripts/scan_contained_bearings.py. Never re-derived server-side.
        vol.Required("route_start"): _CONTINUOUS_MOTION_POINT_SCHEMA,
        vol.Required("route_target"): _CONTINUOUS_MOTION_POINT_SCHEMA,
        # A closed or open ring of >= 3 points; the last edge back to the first
        # point is always implied (`_point_in_polygon`).
        vol.Required("corridor_polygon"): [_CONTINUOUS_MOTION_POINT_SCHEMA],
        # Phase 2 v1 is validated only at this exact command.
        vol.Optional("linear_speed", default=400): vol.All(
            vol.Coerce(int), vol.In([400])
        ),
        # v1 is straight-line only (docs/phase2-continuous-motion-design-20260823.md);
        # this bounds the STEERING correction the pure controller may request,
        # it does not request a turn on its own.
        # 🚨 **120 and 180 ONLY, and 180 stays the default.** The turn rate is
        # measured across angular 120-180 and MUST NOT be scaled outside that
        # band, so a "safer because smaller" value like 60 is NOT safer -- it is
        # unmeasured, and may sit in an actuation deadband where the mower
        # steers less than commanded or not at all. 120 was added 2026-08-27 as
        # the lowest MEASURED arc value, for the first steering validation run,
        # where reduced correction authority is wanted on measured ground.
        vol.Optional("max_abs_angular_speed", default=180): vol.All(
            vol.Coerce(int), vol.In([120, 180])
        ),
        # The plan's own v1 cap is 4 s / 1.5 m
        # (docs/continuous-motion-feasibility-plan-20260821.md); 8000 is the
        # longest window ever driven on this hardware
        # (docs/evidence-phase1b-arc-20260823T171500Z.json), kept as headroom.
        vol.Optional("duration_ms", default=4000): vol.All(
            vol.Coerce(int), vol.Range(min=1000, max=8000)
        ),
        vol.Optional("motion_refresh_interval_ms", default=200): vol.All(
            vol.Coerce(int), vol.Range(min=50, max=1000)
        ),
        vol.Optional("decision_sample_interval_ms", default=100): vol.All(
            vol.Coerce(int), vol.Range(min=50, max=1000)
        ),
        vol.Optional("max_distance_m", default=1.50): vol.All(
            vol.Coerce(float), vol.Range(min=0.10, max=3.0)
        ),
        vol.Optional("max_cross_track_m", default=0.30): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=0.30)
        ),
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
        # 🚨 **THE STEERING OPT-IN.** Until 2026-08-27 this service refused ALL
        # real steering unconditionally (`steering_not_motion_validated`).
        # Opening that refusal did not make steering the default: it is now a
        # deliberate per-call act, defaulting False, on top of the experimental
        # motion gate and every other confirmation. Steering has never completed
        # a physical run -- the one attempt (2026-08-24) diverged on an inverted
        # sign and hard-aborted -- so a caller must say so explicitly every time.
        vol.Optional("confirm_steering_validation_run", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

HEADING_ACQUISITION_WINDOW_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("route_start"): _CONTINUOUS_MOTION_POINT_SCHEMA,
        vol.Required("route_target"): _CONTINUOUS_MOTION_POINT_SCHEMA,
        vol.Required("corridor_polygon"): [_CONTINUOUS_MOTION_POINT_SCHEMA],
        vol.Optional("linear_speed", default=400): vol.All(
            vol.Coerce(int), vol.In([400])
        ),
        vol.Optional("duration_ms", default=2000): vol.All(
            vol.Coerce(int), vol.In([2000])
        ),
        vol.Optional("motion_refresh_interval_ms", default=200): vol.All(
            vol.Coerce(int), vol.In([200])
        ),
        vol.Optional("max_distance_m", default=1.0): vol.All(
            vol.Coerce(float), vol.In([1.0])
        ),
        vol.Optional("prefer_ble", default=True): cv.boolean,
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
    },
    extra=vol.PREVENT_EXTRA,
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

#: Read-only RTK base-station query. No movement parameters exist on purpose:
#: this service cannot command motion. It sends one request_basestation_info_t
#: and reads the reply.
BASESTATION_INFO_PROBE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("wait_seconds", default=3.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.5, max=30.0)
        ),
    },
    extra=vol.ALLOW_EXTRA,
)

#: Read-only OTA-info query. Sends one MctlOta.todev_get_info_req(type=IT_OTA)
#: over the device's own BLE/cloud connection and returns the raw
#: toapp_get_info_rsp. No install/trigger call is made — this is the same
#: request/response shape as basestation_info_probe, just for the OTA
#: message family (EMBED_OTA / SubOtaMsg), which nothing in this integration
#: or pymammotion currently sends. No movement parameters exist on purpose:
#: this service cannot command motion.
OTA_INFO_PROBE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("send_timeout", default=5.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=30.0)
        ),
    },
    extra=vol.ALLOW_EXTRA,
)

#: Read-only telemetry-rate probe. No movement parameters exist on purpose:
#: this service cannot command motion, so it takes no speeds, no pulse counts
#: and no blade/area confirmations. 100 ms is the floor under test because it
#: is well below the 1000 ms library default without asking the device for a
#: rate no evidence suggests it supports.
REPORT_STREAM_PROBE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional("period_ms", default=1000): vol.All(
            vol.Coerce(int), vol.Range(min=100, max=10000)
        ),
        vol.Optional("no_change_period_ms", default=1000): vol.All(
            vol.Coerce(int), vol.Range(min=100, max=10000)
        ),
        vol.Optional("duration_seconds", default=20.0): vol.All(
            vol.Coerce(float), vol.Range(min=2.0, max=120.0)
        ),
        vol.Optional("isolated", default=False): cv.boolean,
    },
    extra=vol.ALLOW_EXTRA,
)

REPORT_STREAM_SEQUENCE_PROBE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("periods_ms"): vol.All(
            [vol.All(vol.Coerce(int), vol.Range(min=100, max=10000))],
            vol.Length(min=1, max=64),
        ),
        vol.Optional("observation_seconds", default=0.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=120.0)
        ),
        vol.Optional("readiness_timeout_seconds", default=3.5): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=10.0)
        ),
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
        vol.Optional("angular_speed", default=500): vol.All(
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
        # App-parity motion cadence for the turn A/B. B1 (2026-07-22) proved
        # refresh is SPEED-GATED: re-sending every 200 ms gave linear an 11x
        # continuous drive but did nothing to a turn at angular 180, which is
        # below this mower's rotation threshold. This probe's ``angular_speed``
        # reaches 500 (unlike manual_velocity_pulse_test, capped ~202), so it is
        # the tool to answer whether refresh unlocks a *properly-powered* turn.
        vol.Optional("motion_refresh_interval_ms", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=1000)
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
        vol.Optional("heading_tolerance_degrees", default=18.0): vol.All(
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
        vol.Optional("fresh_heading_timeout_seconds", default=8.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=20.0)
        ),
        vol.Optional("max_commands", default=8): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=20)
        ),
        vol.Optional("min_progress_degrees", default=2.0): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=20.0)
        ),
        vol.Optional("max_no_progress_pulses", default=2): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=6)
        ),
        vol.Optional("max_displacement_m", default=0.5): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=2.0)
        ),
        vol.Optional("invert_direction", default=False): cv.boolean,
        # 0 == the proven single-shot path. 200 mirrors the app and gave ~7x
        # more rotation per pulse at angular 500 (live 2026-07-25); default is
        # left at 0 until `heading_tolerance_degrees` is re-derived against
        # continuous rotation.
        vol.Optional("motion_refresh_interval_ms", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=1000)
        ),
        # Fallback rotation rate for the scaled final approach, used only until
        # the run measures its own. 37 deg/s is the live 2026-07-27 figure at
        # angular 500 / refresh 200, biased high so the turn undershoots.
        vol.Optional("turn_degrees_per_second", default=37.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=180.0)
        ),
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
        vol.Optional("min_progress_distance", default=0.06): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)
        ),
        vol.Optional("linear_pulse_duration_ms", default=3500.0): vol.All(
            vol.Coerce(float), vol.Range(min=50.0, max=4000.0)
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
        vol.Optional("heading_tolerance_degrees", default=18.0): vol.All(
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
        vol.Optional("motion_refresh_interval_ms", default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=1000)
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
        # Opt-out of the RTK Fix requirement for runs whose result does not
        # depend on centimetre accuracy (relocations, characterisation). Must be
        # stated deliberately: a precision run on Float steers against a
        # position that jumps further than the tolerance it is aiming at.
        vol.Optional("allow_degraded_rtk", default=False): cv.boolean,
        # Deliberate, per-run safety-gate overrides. Each name must be a key of
        # `_OVERRIDABLE_GATES` (defined ABOVE this schema on purpose -- see the
        # NameError note there); an unrecognised name is refused by validation
        # rather than silently ignored, so a typo can never read as a granted
        # override. Empty by default: omitting it dispatches exactly as before.
        vol.Optional("safety_overrides", default=[]): vol.All(
            cv.ensure_list,
            [vol.In(sorted(_OVERRIDABLE_GATES))],
        ),
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
        vol.Optional("ble_auto_recover", default=True): cv.boolean,
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
        vol.Optional("max_linear_pulse_ceiling"): vol.Any(
            None, vol.All(vol.Coerce(int), vol.Range(min=1, max=200))
        ),
        vol.Optional("max_no_progress_pulses", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=20)
        ),
        vol.Optional("linear_distance_ceiling_factor", default=2.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=10.0)
        ),
        vol.Optional("heading_tolerance_degrees", default=18.0): vol.All(
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
        vol.Optional("min_progress_distance", default=0.06): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)
        ),
        vol.Optional("min_heading_change_degrees", default=0.5): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=45.0)
        ),
        vol.Optional("max_turn_translation_distance", default=0.25): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=2.0)
        ),
        vol.Optional(
            "calibrated_forward_heading_offset_degrees", default=116.5
        ): vol.All(vol.Coerce(float), vol.Range(min=-180.0, max=180.0)),
        vol.Optional("turn_pulse_duration_ms", default=1500.0): vol.All(
            vol.Coerce(float), vol.Range(min=50.0, max=2000.0)
        ),
        vol.Optional("linear_pulse_duration_ms", default=3500.0): vol.All(
            vol.Coerce(float), vol.Range(min=50.0, max=4000.0)
        ),
        # Fallback eleven-write distance for final-approach scaling, used only
        # until the run has measured a pulse of its own. 1.06 m is the live
        # 2026-07-27 figure at linear 400: one initial write plus ten refreshes.
        vol.Optional("final_approach_metres_per_pulse", default=1.06): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=5.0)
        ),
        # Same idea for the turn phase; see VIO_TURN_TO_HEADING_SCHEMA.
        vol.Optional("turn_degrees_per_second", default=37.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=180.0)
        ),
        # App-parity motion cadence. Defaults to 200 ms, the app's own timer:
        # B1 (2026-07-22) proved re-sending the movement command every 200 ms
        # for the pulse duration drives ~11x further than a single shot (the
        # mower self-halts after ~one ~4 in step otherwise). 0 restores the
        # legacy single-shot. Used by linear motion and the VIO turn primitive;
        # the short calibration drive remains single-shot.
        vol.Optional("motion_refresh_interval_ms", default=200): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=1000)
        ),
        vol.Optional("turn_mode", default="vio"): vol.In(["vio", "legacy", "night"]),
        vol.Optional("night_angular_speed", default=500): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("toward_mirror_degrees", default=90.13): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=360.0)
        ),
        vol.Optional("vio_heading_offset_degrees"): vol.All(
            vol.Coerce(float), vol.Range(min=-180.0, max=360.0)
        ),
        vol.Optional("vio_turn_max_commands", default=8): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=20)
        ),
        vol.Optional("vio_angular_speed", default=500): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("vio_calibration_pulse_count", default=2): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=5)
        ),
        vol.Optional("vio_realign_threshold_degrees", default=15.0): vol.All(
            vol.Coerce(float), vol.Range(min=5.0, max=90.0)
        ),
        vol.Optional("vio_max_realignments", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=10)
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
        # Opt-out of the RTK Fix requirement for runs whose result does not
        # depend on centimetre accuracy (relocations, characterisation). Must be
        # stated deliberately: a precision run on Float steers against a
        # position that jumps further than the tolerance it is aiming at.
        vol.Optional("allow_degraded_rtk", default=False): cv.boolean,
        # Deliberate, per-run safety-gate overrides. Each name must be a key of
        # `_OVERRIDABLE_GATES` (defined ABOVE this schema on purpose -- see the
        # NameError note there); an unrecognised name is refused by validation
        # rather than silently ignored, so a typo can never read as a granted
        # override. Empty by default: omitting it dispatches exactly as before.
        vol.Optional("safety_overrides", default=[]): vol.All(
            cv.ensure_list,
            [vol.In(sorted(_OVERRIDABLE_GATES))],
        ),
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required("points"): vol.All(
            cv.ensure_list,
            vol.Length(min=2, max=8),
            [_CUSTOM_PATH_POINT_SCHEMA],
        ),
        vol.Optional("area_hash"): vol.Any(vol.Coerce(int), str),
        vol.Optional("dry_run", default=True): cv.boolean,
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
        vol.Optional("prefer_ble", default=True): cv.boolean,
        # The handler reads call.data["ble_auto_recover"]; without this default a
        # call that omits it (the card never sends it) KeyErrors -> HTTP 500. Kept
        # in sync with RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA.
        vol.Optional("ble_auto_recover", default=True): cv.boolean,
        vol.Optional("max_real_segments", default=1): vol.All(
            vol.Coerce(int),
            vol.Range(min=0, max=REAL_CLICK_TO_GO_SEGMENT_LIMIT),
        ),
        # Route B: split any leg longer than this into collinear sub-legs, so a
        # distant click is driven with only geometry that has been measured.
        # None (the default) is off and dispatches exactly as before.
        #
        # ⚠️ The 6.10 upper bound is a LITERAL on purpose. It mirrors
        # `_MAX_SEGMENT_LENGTH_M`, which is defined ~10,000 lines below this
        # schema -- referencing the constant here is a NameError at import.
        # `test_collinear_leg_split.py` asserts the literal still equals it.
        vol.Optional("split_leg_target_length_m"): vol.Any(
            None, vol.All(vol.Coerce(float), vol.Range(min=0.5, max=6.10))
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
        vol.Optional("max_linear_pulse_ceiling"): vol.Any(
            None, vol.All(vol.Coerce(int), vol.Range(min=1, max=200))
        ),
        vol.Optional("max_no_progress_pulses", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=20)
        ),
        vol.Optional("linear_distance_ceiling_factor", default=2.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=10.0)
        ),
        vol.Optional("heading_tolerance_degrees", default=18.0): vol.All(
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
        vol.Optional("min_progress_distance", default=0.06): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)
        ),
        vol.Optional("min_heading_change_degrees", default=0.5): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=45.0)
        ),
        vol.Optional("max_turn_translation_distance", default=0.25): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=2.0)
        ),
        vol.Optional(
            "calibrated_forward_heading_offset_degrees", default=116.5
        ): vol.All(vol.Coerce(float), vol.Range(min=-180.0, max=180.0)),
        vol.Optional("turn_pulse_duration_ms", default=1500.0): vol.All(
            vol.Coerce(float), vol.Range(min=50.0, max=2000.0)
        ),
        vol.Optional("linear_pulse_duration_ms", default=3500.0): vol.All(
            vol.Coerce(float), vol.Range(min=50.0, max=4000.0)
        ),
        vol.Optional("final_approach_metres_per_pulse", default=1.06): vol.All(
            vol.Coerce(float), vol.Range(min=0.05, max=5.0)
        ),
        vol.Optional("turn_degrees_per_second", default=37.0): vol.All(
            vol.Coerce(float), vol.Range(min=1.0, max=180.0)
        ),
        # App-parity motion cadence. Defaults to 200 ms, the app's own timer:
        # B1 (2026-07-22) proved re-sending the movement command every 200 ms
        # for the pulse duration drives ~11x further than a single shot (the
        # mower self-halts after ~one ~4 in step otherwise). 0 restores the
        # legacy single-shot. Scoped to the linear phase only; the calibration
        # drive and VIO turn phase never consult it.
        vol.Optional("motion_refresh_interval_ms", default=200): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=1000)
        ),
        vol.Optional("turn_mode", default="vio"): vol.In(["vio", "legacy", "night"]),
        vol.Optional("night_angular_speed", default=500): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("toward_mirror_degrees", default=90.13): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=360.0)
        ),
        vol.Optional("vio_heading_offset_degrees"): vol.All(
            vol.Coerce(float), vol.Range(min=-180.0, max=360.0)
        ),
        vol.Optional("vio_turn_max_commands", default=8): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=20)
        ),
        vol.Optional("vio_angular_speed", default=500): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=1000)
        ),
        vol.Optional("vio_calibration_pulse_count", default=2): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=5)
        ),
        vol.Optional("vio_realign_threshold_degrees", default=15.0): vol.All(
            vol.Coerce(float), vol.Range(min=5.0, max=90.0)
        ),
        vol.Optional("vio_max_realignments", default=3): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=10)
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
        vol.Optional(
            "calibrated_forward_heading_offset_degrees", default=116.5
        ): vol.All(vol.Coerce(float), vol.Range(min=-180.0, max=180.0)),
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
        vol.Optional(
            "pulses_per_burst", default=DEFAULT_EXPERIMENTAL_SEGMENT_PULSES_PER_BURST
        ): vol.All(vol.Coerce(int), vol.Range(min=1, max=3)),
        vol.Optional(
            "max_bursts", default=DEFAULT_EXPERIMENTAL_SEGMENT_MAX_BURSTS
        ): vol.All(vol.Coerce(int), vol.Range(min=1, max=3)),
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
        vol.Optional(
            "stop_mode", default=DEFAULT_EXPERIMENTAL_SEGMENT_STOP_MODE
        ): vol.In(["immediate", "delayed", "firmware"]),
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
        vol.Optional(
            "calibrated_forward_heading_degrees",
            default=DEFAULT_CALIBRATED_FORWARD_HEADING_DEGREES,
        ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=360.0)),
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
        vol.Optional("confirm_blades_off", default=False): cv.boolean,
        vol.Optional("confirm_clear_area", default=False): cv.boolean,
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


#: `HashList` fields holding KEEP-OUT geometry, in the same map-local x/y frame
#: as `area`. Every one is a polygon the mower must not enter.
#:
#: 🚨 Added 2026-08-20 after a supervised run drove into a no-go zone and pushed
#: a trampoline. `_area_polygons` read `map.area` and nothing else, so
#: containment validated INCLUSION in a mowing zone and never tested EXCLUSION
#: from a keep-out. The geometry was not missing and not unreachable -- it sits
#: in sibling dicts on the same object, and `get_geojson` has always exposed it
#: (the mower reported obstacle hash 1529607395159402290; the geojson names it
#: "Obstacle 1", "Obstacle in Backyard Right", ~4.0 x 4.1 m).
#:
#: ⚠️ Only `obstacle` is CONFIRMED populated on this hardware. The rest are
#: included because a keep-out we cannot see is exactly the failure this fixes,
#: and reading an empty dict costs nothing. `no_go_zone_variant` is documented
#: upstream as a sibling of `no_go_zone` -- both encode as `(shape=0, type=1)`.
_KEEP_OUT_MAP_FIELDS: tuple[str, ...] = (
    "obstacle",  # type 1  -- keep-out / no-go obstacle boundary (CONFIRMED live)
    "no_go_zone",  # type 23 -- user-drawn rectangular no-go zone
    "no_go_zone_variant",  # type 22 -- sibling of 23
    "virtual_wall",  # type 21 -- user-drawn virtual fence / keep-out line
    "visual_obstacle_zone",  # type 26 -- vision-detected obstacle (Vision/Pro)
)


def _keep_out_polygons(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, list[dict[str, float]]]:
    """Return keep-out polygons as map-local x/y, keyed ``"<field>:<hash>"``.

    Same extraction as `_area_polygons`, over the keep-out dicts instead of
    `map.area`. No coordinate conversion: these are already in the frame the
    path planner uses.

    Keyed by field AND hash because two different keep-out types could in
    principle carry the same hash, and a diagnostics blob that silently
    collapsed them would hide one.
    """
    device_data = coordinator.data
    polygons: dict[str, list[dict[str, float]]] = {}
    for field_name in _KEEP_OUT_MAP_FIELDS:
        frame_lists = getattr(device_data.map, field_name, None) or {}
        if not isinstance(frame_lists, dict):
            continue
        for current_hash, frame_list in frame_lists.items():
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
            # A keep-out needs 3 points to bound anything. A 2-point virtual
            # wall is a LINE, which this polygon test cannot represent -- it is
            # reported in diagnostics rather than silently treated as empty.
            if points:
                polygons[f"{field_name}:{current_hash}"] = points
    return polygons


def _keep_out_violations(
    points: list[dict[str, float]],
    keep_outs: dict[str, list[dict[str, float]]],
) -> list[dict[str, Any]]:
    """Return every (point index, keep-out) pair where a waypoint is forbidden."""
    violations: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        for key, polygon in keep_outs.items():
            if len(polygon) < 3:
                continue
            if _point_in_polygon(point, polygon):
                field_name, _, hash_text = key.partition(":")
                violations.append(
                    {
                        "point_index": index,
                        "point": point,
                        "keep_out_type": field_name,
                        "keep_out_hash": hash_text,
                    }
                )
    return violations


def _segments_intersect(
    first_start: dict[str, float],
    first_end: dict[str, float],
    second_start: dict[str, float],
    second_end: dict[str, float],
) -> bool:
    """Return whether two closed line segments intersect, including touching."""

    def orientation(
        start: dict[str, float],
        end: dict[str, float],
        point: dict[str, float],
    ) -> float:
        return (end["x"] - start["x"]) * (point["y"] - start["y"]) - (
            end["y"] - start["y"]
        ) * (point["x"] - start["x"])

    first_side_start = orientation(first_start, first_end, second_start)
    first_side_end = orientation(first_start, first_end, second_end)
    second_side_start = orientation(second_start, second_end, first_start)
    second_side_end = orientation(second_start, second_end, first_end)

    if (
        first_side_start > 0 > first_side_end or first_side_start < 0 < first_side_end
    ) and (
        second_side_start > 0 > second_side_end
        or second_side_start < 0 < second_side_end
    ):
        return True

    return (
        (
            first_side_start == 0
            and _point_on_segment(second_start, first_start, first_end)
        )
        or (
            first_side_end == 0
            and _point_on_segment(second_end, first_start, first_end)
        )
        or (
            second_side_start == 0
            and _point_on_segment(first_start, second_start, second_end)
        )
        or (
            second_side_end == 0
            and _point_on_segment(first_end, second_start, second_end)
        )
    )


def _keep_out_leg_violations(
    points: list[dict[str, float]],
    keep_outs: dict[str, list[dict[str, float]]],
) -> list[dict[str, Any]]:
    """Return legal-endpoint legs that cross or touch a keep-out boundary.

    Endpoint containment remains in `_keep_out_violations`; this closes the gap
    where both endpoints are legal but the segment between them is not. Keeping
    the diagnostics separate makes the refusal reason unambiguous.
    """
    violations: list[dict[str, Any]] = []
    for leg_index, (start, end) in enumerate(zip(points, points[1:], strict=False)):
        for key, polygon in keep_outs.items():
            if len(polygon) < 3:
                continue
            if _point_in_polygon(start, polygon) or _point_in_polygon(end, polygon):
                continue
            if any(
                _segments_intersect(
                    start,
                    end,
                    polygon[edge_index],
                    polygon[(edge_index + 1) % len(polygon)],
                )
                for edge_index in range(len(polygon))
            ):
                field_name, _, hash_text = key.partition(":")
                violations.append(
                    {
                        "leg_index": leg_index,
                        "start_point_index": leg_index,
                        "end_point_index": leg_index + 1,
                        "start": start,
                        "end": end,
                        "keep_out_type": field_name,
                        "keep_out_hash": hash_text,
                    }
                )
    return violations


def _point_on_segment(
    point: dict[str, float],
    start: dict[str, float],
    end: dict[str, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return True if *point* lies on the line segment from *start* to *end*."""
    squared_len = (end["x"] - start["x"]) ** 2 + (end["y"] - start["y"]) ** 2
    if squared_len <= tolerance:
        # Degenerate "segment": start == end, so it is a point, and only the
        # same point lies on it. Without this guard every test below passes for
        # ANY point -- cross and dot are both identically zero, and the final
        # comparison becomes `0 <= 0`. That made `_point_in_polygon` return True
        # for the entire plane, because the mower's area polygons are CLOSED
        # RINGS (first vertex == last), so the loop's very first segment is
        # `polygon[-1] -> polygon[0]` -- always degenerate. Every containment
        # check in the integration was therefore inert: live 2026-07-28, a point
        # 2 m outside the lawn tested as inside all four areas at once,
        # including two that were 19 m and 28 m away.
        return (point["x"] - start["x"]) ** 2 + (
            point["y"] - start["y"]
        ) ** 2 <= tolerance
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
    return dot <= squared_len + tolerance


def _point_in_polygon(point: dict[str, float], polygon: list[dict[str, float]]) -> bool:
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
            x_at_y = (previous["x"] - current["x"]) * (point["y"] - current["y"]) / (
                previous["y"] - current["y"]
            ) + current["x"]
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


def _path_heading_degrees(start: dict[str, float], end: dict[str, float]) -> float:
    """Return a map-local heading in degrees for a segment."""
    return (
        math.degrees(math.atan2(end["y"] - start["y"], end["x"] - start["x"])) + 360
    ) % 360


def _split_long_legs(
    points: list[dict[str, float]],
    *,
    target_length_m: float | None,
) -> dict[str, Any]:
    """Split each leg longer than ``target_length_m`` into collinear sub-legs.

    Route B (2026-08-19): reach a distant click using only geometry that has
    been measured. A leg of ``d`` metres becomes ``n = ceil(d / target)``
    sub-legs of ``d / n`` metres each, by linear interpolation between the two
    endpoints -- so every inserted point lies exactly on the operator's original
    line and every junction it creates is a 0 degree turn. A 0 degree turn costs
    zero turn commands and zero translation: ``_vio_turn_to_heading`` returns
    ``target_heading_reached`` immediately when the error is inside
    ``heading_tolerance_degrees``.

    ``target_length_m`` of ``None`` (the schema default) means off, and the
    points are returned unchanged -- a caller that omits the parameter dispatches
    byte-identically to before this existed.

    No rounding of the interpolated coordinates: rounding to 3 dp would inject
    up to ~1.4 mm of non-collinearity, and collinearity is the whole basis for
    treating these junctions as free.
    """
    requested = [dict(point) for point in points]
    result: dict[str, Any] = {
        "applied": False,
        "target_length_m": target_length_m,
        "requested_points": requested,
        "points": requested,
        "requested_leg_count": max(0, len(requested) - 1),
        "sub_leg_count": max(0, len(requested) - 1),
        "legs": [],
    }
    if target_length_m is None or target_length_m <= 0 or len(requested) < 2:
        return result

    split_points: list[dict[str, float]] = [requested[0]]
    legs: list[dict[str, Any]] = []
    for leg_index, (start, end) in enumerate(
        zip(requested, requested[1:], strict=False)
    ):
        length = _path_distance([start, end])
        # A non-finite coordinate reaches here: the point schema is a bare
        # `vol.Coerce(float)`, which accepts inf and nan. `math.ceil(inf)` raises
        # OverflowError -> HTTP 500. Leave such a leg unsplit and let the
        # existing length/containment gates refuse it by name instead.
        sub_legs = (
            1
            if not math.isfinite(length)
            else max(1, math.ceil(length / target_length_m))
        )
        legs.append(
            {
                "index": leg_index + 1,
                "length_m": length,
                "sub_legs": sub_legs,
                "sub_leg_length_m": length / sub_legs if sub_legs else length,
            }
        )
        for step in range(1, sub_legs):
            fraction = step / sub_legs
            split_points.append(
                {
                    "x": start["x"] + (end["x"] - start["x"]) * fraction,
                    "y": start["y"] + (end["y"] - start["y"]) * fraction,
                }
            )
        split_points.append(dict(end))

    result["points"] = split_points
    result["sub_leg_count"] = max(0, len(split_points) - 1)
    result["legs"] = legs
    result["applied"] = len(split_points) > len(requested)
    return result


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
        (item for item in distances if item["distance"] > waypoint_tolerance),
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
                # 🚨 Keep-outs, in the SAME map-local x/y frame as the areas, so
                # a consumer can test exclusion without any coordinate
                # conversion. `get_geojson` has always carried these in WGS84
                # lat/lon, which is why the card could draw them while
                # containment could not check them.
                "keep_out_polygons": _keep_out_polygons(coordinator),
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


#: The map frame is a math angle (CCW from +x) while ``toward`` is a compass
#: bearing (CW from north), so their relation is a reflection, not an offset.
#: The 0.13 degree map alignment is measured for this installation; callers can
#: override it per mower rather than treating it as a universal map constant.
#:
#: 🚨 A reflection cannot be emulated by adding a constant, so **no value of
#: ``calibrated_forward_heading_offset_degrees`` can ever be correct** for
#: turning ``toward`` into a map bearing. The additive form looks right only
#: near the two headings where the curves cross, which is how it survived to
#: 2026-09-04 and aimed a real dispatch 164 deg wrong.
_TOWARD_MIRROR_DEGREES = 90.13


#: How far the two independent heading sources may disagree and still be
#: published as a trustworthy orientation, in degrees.
#:
#: The two sources are genuinely independent -- VIO is visual odometry, the
#: mirror is derived from the RTK compass field -- so agreement is real evidence
#: rather than a self-check. Measured 2026-08-26 across two supervised runs, they
#: agreed to **0.91 deg** and **0.12 deg**, so 15 deg is loose by more than an
#: order of magnitude. It is deliberately NOT tight: this gates a display arrow,
#: and the failure that matters is drawing a confidently wrong heading, which a
#: disagreement of any real size already catches.
_ORIENTATION_AGREEMENT_TOLERANCE_DEGREES = 15.0


def _current_orientation(
    coordinator: MammotionReportUpdateCoordinator,
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    """Return a map-aligned body orientation, trustworthy only when corroborated.

    🚨 **This field did not exist until 2026-08-26 and the card had been reading
    it since beta19.** `mammotion-custom-path-card.js` gates its direction arrow
    on `current_orientation.trustworthy === true`, nothing in this integration
    ever emitted the key, so the arrow was unreachable by construction -- not a
    data problem, a missing producer.

    Two independent sources must agree before this is published as trustworthy:

    * **VIO** (`vision_info.heading`) is body orientation and tracks in-place
      rotation (established 2026-07-10). It is the reason an arrow is possible at
      all -- but it is a *relative* sensor whose zero can shift across a re-init,
      so it is never published alone.
    * **The compass mirror** (`90.13 - toward`). `toward` was long believed to be
      course-over-ground and blind to in-place rotation; that premise was
      REFUTED 2026-08-12 (a pivot moved `toward` 99.55 deg on 3.8 cm of travel).
      It does still freeze when the mower is genuinely stationary, which is
      harmless here -- a stationary mower's orientation is not changing either.

    Agreement between a visual and a magnetic source is real corroboration.
    Disagreement means at least one is wrong and we cannot tell which, so the
    honest output is no arrow -- the same conservative posture beta19 adopted
    when it stopped drawing the last-travel projection as if it were orientation.

    ⚠️ A degraded VIO feed is NOT trusted even though `vio_state` may still read
    2: at dusk the tracked-feature count falls to zero while the state latches
    and the heading freezes, so `_vio_feed_liveness` gates on the feature count.
    """
    feed = _vio_feed_liveness(coordinator)
    vio_raw = _safe_attr_path(coordinator.data, "report_data.vision_info.heading")
    toward = (telemetry.get("position") or {}).get("toward")

    evidence_source = getattr(coordinator, "facing_motion_evidence", None)
    result: dict[str, Any] = {
        "trustworthy": False,
        "map_heading_degrees": None,
        "source": None,
        "reason": None,
        "vio_feed_live": feed["live"],
        "vio_tracked_features": feed["tracked_features"],
        # 🚨 Say what the flag means, in the payload, because it was read as
        # something stronger and hardware moved the wrong way on 2026-09-04.
        "trustworthy_means": (
            "two independent estimates agree -- NOT that either is fresh; "
            "both can be stale together after a manual reposition. "
            "Use runtime_state.map_facing.safe_to_aim_dispatch to aim a dispatch."
        ),
        "motion_evidence": (
            evidence_source() if callable(evidence_source) else {"unavailable": True}
        ),
    }

    try:
        vio_map = float(vio_raw) % 360.0
    except TypeError, ValueError:
        result["reason"] = "vio_heading_unavailable"
        return result
    if not math.isfinite(vio_map):
        result["reason"] = "vio_heading_unavailable"
        return result
    if not feed["live"]:
        result["reason"] = "vio_feed_degraded"
        return result

    if toward is None:
        result["reason"] = "toward_unavailable"
        return result
    try:
        mirror = _continuous_course_heading(float(toward))
    except TypeError, ValueError:
        result["reason"] = "toward_unavailable"
        return result

    disagreement = abs(((mirror - vio_map + 180.0) % 360.0) - 180.0)
    result["vio_map_heading_degrees"] = round(vio_map, 3)
    result["mirror_map_heading_degrees"] = round(mirror, 3)
    result["disagreement_degrees"] = round(disagreement, 3)
    result["agreement_tolerance_degrees"] = _ORIENTATION_AGREEMENT_TOLERANCE_DEGREES

    if disagreement > _ORIENTATION_AGREEMENT_TOLERANCE_DEGREES:
        result["reason"] = "heading_sources_disagree"
        return result

    # VIO is the published value: it is body orientation by construction, while
    # the mirror only corroborates it.
    result["trustworthy"] = True
    result["map_heading_degrees"] = round(vio_map, 3)
    result["source"] = "vio_heading corroborated by compass mirror"
    result["reason"] = "corroborated"
    return result


#: How closely the last driven leg's bearing must match the mower's published
#: facing before that facing counts as confirmed against the ground.
#:
#: Loose on purpose. This does not measure anything -- it asks "did the machine
#: go roughly where its own estimate says it points?" The failure it exists to
#: catch is the ~166 deg class seen on 2026-09-04, not a few degrees of
#: cross-track. A tight bound here would demote perfectly good facings to
#: "ask the operator" on ordinary drift, and being needlessly blocked is a real
#: cost in this yard.
_FACING_MOTION_AGREEMENT_TOLERANCE_DEGREES = 20.0

#: How long a motion-confirmed facing stays motion-confirmed once the mower
#: stops.
#:
#: ⚠️ This is a POLICY choice, not a measurement, and it is the honest
#: expression of a gap: nothing in the telemetry can detect the mower being
#: picked up and turned by hand. Time at rest is the only proxy available --
#: the longer a machine sits, the more opportunity there was to move it. Five
#: minutes covers the seconds-scale gaps between segments of one run while
#: refusing anything after a real pause. It cannot be tightened into safety and
#: must not be read as one.
_FACING_MOTION_CONFIRMED_TTL_SECONDS = 300.0

#: 16-point compass names, indexed by (bearing + 11.25) // 22.5.
_COMPASS_POINTS: tuple[str, ...] = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)


def _compass_point(compass_bearing_degrees: float) -> str:
    """Return the 16-point compass name for a compass bearing."""
    index = int((float(compass_bearing_degrees) % 360.0) / 22.5 + 0.5) % 16
    return _COMPASS_POINTS[index]


def _map_facing_report(
    coordinator: MammotionReportUpdateCoordinator,
    telemetry: dict[str, Any],
    *,
    toward_mirror_degrees: float = _TOWARD_MIRROR_DEGREES,
) -> dict[str, Any]:
    """Answer "which way is this mower pointing, and is that answer fresh?".

    🔑 **This is the one place that question is answered.** It exists because on
    2026-09-04 nobody could answer it: not the integration, not the orchestrating
    session, not the operator through Home Assistant. The full record is
    ``docs/findings-clicktopath-reliability-4m-20260904.md``.

    Three things it deliberately does:

    1. **It uses the reflection and nothing else.** ``map_facing = mirror -
       toward``. The map frame is a math angle (CCW from +x) and ``toward`` is a
       compass bearing (CW from north), so the relation is a reflection. ⚠️ **No
       value of ``calibrated_forward_heading_offset_degrees`` can ever be
       correct here** -- an additive constant cannot emulate a reflection; it
       merely happens to look right near the two headings where the curves
       cross, which is how it survived. Measured on 43 real pulses that night:
       the mirror predicts the driven direction to a mean 1.000 deg, the
       additive offset to a mean 87.478 deg.
    2. **It separates corroboration from freshness.** ``current_orientation``
       publishes ``trustworthy`` when VIO and the mirror agree. That is real
       corroboration and it is NOT evidence of freshness: after a manual
       reposition both sources described the pre-reposition facing and agreed
       with each other to 0.079 deg. Only the third source here -- the bearing
       of the leg the mower actually drove -- can rule that out, because it is
       not an estimate.
    3. **It returns unknown rather than a number it cannot stand behind.**

    ``confidence`` is one of:

    * ``motion_confirmed`` -- the mower drove a real leg recently and its
      published facing agrees with where it went. This is the only value that
      sets ``safe_to_aim_dispatch``.
    * ``corroborated_not_motion_confirmed`` -- two independent estimates agree,
      but nothing has checked them against the ground. Usable for a display
      arrow; ⚠️ **not** for aiming a dispatch without operator confirmation.
    * ``unknown`` -- sources disagree, or one is unavailable.

    ⚠️ An in-place rotation after the last driven leg legitimately puts the
    driven bearing out of date, and this will demote such a facing to
    ``corroborated_not_motion_confirmed``. That is the conservative direction:
    it asks for a human, it never green-lights a wrong number.
    """
    orientation = _current_orientation(coordinator, telemetry)
    evidence_source = getattr(coordinator, "facing_motion_evidence", None)
    evidence: dict[str, Any] = (
        evidence_source() if callable(evidence_source) else {"unavailable": True}
    )

    report: dict[str, Any] = {
        "map_facing_degrees": None,
        "confidence": "unknown",
        "safe_to_aim_dispatch": False,
        "reason": orientation.get("reason"),
        "model": (
            "map_facing = toward_mirror_degrees - toward "
            "(a REFLECTION; no additive offset can emulate it)"
        ),
        "toward_mirror_degrees": toward_mirror_degrees,
        "sources": {
            "vio_heading": orientation.get("vio_map_heading_degrees"),
            "compass_mirror": orientation.get("mirror_map_heading_degrees"),
            "last_driven_leg": evidence.get("last_travel_bearing_degrees"),
        },
        "corroboration": {
            "corroborated": bool(orientation.get("trustworthy")),
            "disagreement_degrees": orientation.get("disagreement_degrees"),
            "agreement_tolerance_degrees": orientation.get(
                "agreement_tolerance_degrees"
            ),
            "means": (
                "two independent estimates agree; NOT evidence that either is fresh"
            ),
        },
        "motion_evidence": evidence,
        "operator_confirmation_required": True,
    }

    if not orientation.get("trustworthy"):
        return report

    facing = float(orientation["map_heading_degrees"])
    report["map_facing_degrees"] = round(facing, 3)
    compass = _map_heading_to_toward_degrees(
        facing, toward_mirror_degrees=toward_mirror_degrees
    )
    report["map_facing_compass_degrees"] = round(compass, 1)
    report["map_facing_compass_point"] = _compass_point(compass)
    report["confidence"] = "corroborated_not_motion_confirmed"
    report["reason"] = "corroborated_but_unverified_against_ground"

    driven = evidence.get("last_travel_bearing_degrees")
    age = evidence.get("last_travel_age_seconds")
    if driven is None or age is None:
        report["reason"] = "no_driven_leg_on_record"
        return report
    if float(age) > _FACING_MOTION_CONFIRMED_TTL_SECONDS:
        report["reason"] = "last_driven_leg_too_old"
        report["motion_confirmed_ttl_seconds"] = _FACING_MOTION_CONFIRMED_TTL_SECONDS
        return report

    drift = abs(((float(driven) - facing + 180.0) % 360.0) - 180.0)
    report["motion_agreement_degrees"] = round(drift, 3)
    report["motion_agreement_tolerance_degrees"] = (
        _FACING_MOTION_AGREEMENT_TOLERANCE_DEGREES
    )
    if drift > _FACING_MOTION_AGREEMENT_TOLERANCE_DEGREES:
        # Either the mower rotated in place since that leg (benign) or its
        # estimate is wrong (not benign). Nothing here can tell them apart, so
        # the answer is "ask", not "probably fine".
        report["reason"] = "facing_disagrees_with_last_driven_leg"
        return report

    report["confidence"] = "motion_confirmed"
    report["safe_to_aim_dispatch"] = True
    report["operator_confirmation_required"] = False
    report["reason"] = "confirmed_by_last_driven_leg"
    return report


def _runtime_blade_diagnostics(
    telemetry: dict[str, Any],
    *,
    rpm_stale_register: bool = False,
) -> dict[str, Any]:
    """Return blade diagnostics plus a conservative motion-safety decision.

    ``current_cutter_rpm`` is a LATCHED device register: it holds its last
    running value after a mow and is never reset by the firmware. Measured
    2026-07-30 on the dock, blade off, over a live BLE link -- the position feed
    jittered between reads (proving the report stream was fresh) while RPM stayed
    bit-identical at 3014 with ``current_cutter_mode`` 0 and state OFF. Since it
    re-latches after every mow, trusting it verbatim blocks every real run that
    follows one.

    ``rpm_stale_register`` is the verdict of :func:`_reconfirm_blade_rpm_stale`,
    which requires positive proof the feed is live before discounting the value.
    It defaults False so every non-dispatch caller stays conservative, and the
    single-sample ``blade_rpm_looks_latched`` hint explains a blocked gate
    without any polling.
    """
    blade = dict(telemetry.get("blade", {}) or {})
    reported_state = blade.get("reported_state")
    cutter_mode = blade.get("current_cutter_mode")
    rpm = blade.get("current_cutter_rpm")
    reported_on = reported_state not in (None, 0, "0")
    mode_on = cutter_mode not in (None, 0, "0")
    rpm_nonzero = rpm not in (None, 0, "0")
    # A nonzero RPM while BOTH state and mode say off is the latch signature.
    looks_latched = rpm_nonzero and not reported_on and not mode_on
    rpm_discounted = rpm_nonzero and looks_latched and rpm_stale_register
    blockers = []
    if reported_on:
        blockers.append("blade_reported_on")
    if mode_on:
        blockers.append("blade_cutter_mode_on")
    if rpm_nonzero and not rpm_discounted:
        blockers.append("blade_rpm_nonzero")
    return {
        **blade,
        "blade_safe_for_motion": not blockers,
        "safety_blockers": blockers,
        "blade_rpm_looks_latched": looks_latched,
        "blade_rpm_stale_register": rpm_discounted,
    }


#: Independent observations required before the RPM register may be called
#: latched. These are NOT polled on demand: they accumulate from the gate
#: snapshot that already runs on every coordinator tick, so the evidence spans
#: real time and the dispatch path adds no latency and no sleeping. An earlier
#: version polled three times with 1.5s gaps inside the authorization preamble,
#: which delayed every real run and deadlocked the concurrency test.
_BLADE_RPM_RECONFIRM_POLLS = 3
_BLADE_RPM_HISTORY_LIMIT = 8
_BLADE_RPM_HISTORY_ATTR = "_mammotion_blade_rpm_history"

#: Movement (metres) between polls that proves the report stream is genuinely
#: live. The position feed jitters ~2-4mm between reads on a live link, so this
#: sits just above the read-to-read noise while staying far below real motion.
_BLADE_RPM_FEED_LIVE_EPSILON_M = 0.0005


def _blade_rpm_stale_verdict(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Decide whether a nonzero cutter RPM is a latched register, from samples.

    Every condition must hold, and the last one is the important one: the feed
    must be PROVEN live by the position changing across the samples. A latched
    register cannot vary while the rest of the report demonstrably does, whereas
    a genuinely spinning blade reports a varying RPM. Without the liveness proof
    a frozen feed would look identical to a latch, so this refuses to decide on
    a dead feed rather than guessing.
    """
    reasons: list[str] = []
    if len(samples) < _BLADE_RPM_RECONFIRM_POLLS:
        reasons.append("insufficient_samples")
    blades = [s.get("blade") or {} for s in samples]
    positions = [s.get("position") or {} for s in samples]
    rpms = [b.get("current_cutter_rpm") for b in blades]
    if any(b.get("reported_state") not in (None, 0, "0") for b in blades):
        reasons.append("blade_reported_on_in_a_sample")
    if any(b.get("current_cutter_mode") not in (None, 0, "0") for b in blades):
        reasons.append("cutter_mode_on_in_a_sample")
    if any(r in (None, 0, "0") for r in rpms):
        reasons.append("rpm_not_consistently_nonzero")
    if len(set(rpms)) != 1:
        reasons.append("rpm_varied_so_it_may_be_live")

    def _moved(a: dict[str, Any], b: dict[str, Any]) -> bool:
        try:
            return (
                abs(float(a["x"]) - float(b["x"])) > _BLADE_RPM_FEED_LIVE_EPSILON_M
                or abs(float(a["y"]) - float(b["y"])) > _BLADE_RPM_FEED_LIVE_EPSILON_M
            )
        except KeyError, TypeError, ValueError:
            return False

    feed_live = any(
        _moved(positions[i], positions[j])
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    )
    if not feed_live:
        reasons.append("feed_not_proven_live")
    return {
        "stale_register": not reasons,
        "reasons": reasons,
        "samples": len(samples),
        "rpm_values": rpms,
        "feed_proven_live": feed_live,
    }


def _record_blade_rpm_sample(
    coordinator: MammotionReportUpdateCoordinator, telemetry: dict[str, Any]
) -> None:
    """Append one blade/position observation to the coordinator's short history.

    Called from the gate snapshot, which already runs once per coordinator tick,
    so the history accumulates on its own across real time. Never raises.
    """
    try:
        history = list(getattr(coordinator, _BLADE_RPM_HISTORY_ATTR, None) or [])
        history.append(
            {
                "blade": dict(telemetry.get("blade") or {}),
                "position": {
                    "x": (telemetry.get("position") or {}).get("x"),
                    "y": (telemetry.get("position") or {}).get("y"),
                },
            }
        )
        setattr(
            coordinator,
            _BLADE_RPM_HISTORY_ATTR,
            history[-_BLADE_RPM_HISTORY_LIMIT:],
        )
    except Exception:  # noqa: BLE001 - history is best-effort diagnostics
        LOGGER.debug("blade RPM history append failed", exc_info=True)


def blade_rpm_stale_register(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Judge whether a nonzero cutter RPM is a latched register, from history.

    Synchronous and allocation-cheap: it reads observations already gathered by
    the gate snapshot, so the dispatch path neither sleeps nor polls. If too few
    observations exist yet -- entities disabled, or a freshly started HA -- the
    verdict is not-stale and the blade guard stays closed.
    """
    history = list(getattr(coordinator, _BLADE_RPM_HISTORY_ATTR, None) or [])
    return _blade_rpm_stale_verdict(history[-_BLADE_RPM_RECONFIRM_POLLS:])


#: Age past which the RTK payload is *annotated* as suspiciously quiet. This is
#: **advisory only and must stay that way** -- see below before making it gate
#: anything again.
#:
#: ⚠️ DO NOT turn this back into a blocker. Two attempts, two false-blocks:
#:
#: * 300 s, justified as "far longer than any plausible quiet period", was
#:   disproved within the hour -- a healthy Fix-locked stationary mower went
#:   582 s without a payload change.
#: * 1800 s, chosen as ~3x that, was disproved the same evening -- the same
#:   mower reached **3573 s (59.5 min)** of unchanged payload while `Fix` and
#:   demonstrably healthy. Shipped in beta25, it would have refused motion after
#:   roughly 30 idle minutes with nothing wrong.
#:
#: The reason no value works: a stationary mower's RTK payload changes only
#: about once an hour, and the one observed genuine fault lasted three hours.
#: A 3x separation, from a single sample of each, cannot site a threshold.
#:
#: An *active* probe does not rescue it either. Forcing a report burst while RTK
#: was verifiably healthy produced 49 messages and **zero** RTK channel updates,
#: with the age still climbing -- indistinguishable from the latched case. So
#: neither a passive timeout nor an on-demand probe can tell "quiet" from "dead".
#:
#: What actually protects a precision run is `_RTK_PRECISE_STATUS_LABELS` below:
#: the fault observed on 2026-08-07 was a latched *Float*, which the quality gate
#: refuses outright without needing any freshness reasoning. Freshness only ever
#: addressed a latched *Fix* that silently stops being true, which has never been
#: observed on this hardware. Reporting the age is still worth it -- a run that
#: later looks wrong can be audited against it -- but it must not gate.
_RTK_REPORT_QUIET_SECONDS = 1800.0

#: RTK solution types precise enough for a run whose result depends on position.
#: Only a resolved carrier-phase fix qualifies. Measured 2026-08-07 on this
#: mower: **Fix** jitters 0.044 cm mean / 0.55 cm max while stationary, whereas
#: **Float** produced a single 13.9 cm jump with no command sent -- larger than
#: the entire 0.08 m waypoint tolerance. Float is decimetre-grade and Single is
#: metre-grade; steering a precision run on either means chasing a target that
#: moves further than the tolerance being aimed at.
_RTK_PRECISE_STATUS_LABELS = frozenset({"Fix"})


def _rtk_report_age_seconds(coordinator: Any) -> float | None:
    """Seconds since the RTK payload last changed, or None if unmeasurable.

    Tolerates coordinators without the tracker (older builds, test doubles) by
    returning None, which the safety summary treats as "not evaluated" rather
    than as a failure.
    """
    age = getattr(coordinator, "rtk_report_age_seconds", None)
    return float(age) if isinstance(age, int | float) else None


def _runtime_motion_safety_summary(
    telemetry: dict[str, Any],
    *,
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
    rpm_stale_register: bool = False,
    rtk_report_age_seconds: float | None = None,
    allow_degraded_rtk: bool = False,
) -> dict[str, Any]:
    """Return conservative safety summary for diagnostics and future motion gates.

    ``rtk_report_age_seconds`` is seconds since the RTK payload last changed, or
    ``None`` when the caller could not measure it. It is reported for auditing
    and **never blocks** -- a stationary mower goes ~an hour between RTK payload
    changes, so age cannot distinguish a quiet feed from a dead one. See
    ``_RTK_REPORT_QUIET_SECONDS``.

    ``allow_degraded_rtk`` opts a run out of the Fix requirement. It exists
    because both cases occurred on 2026-08-07: a 1.6 m relocation on Float was
    entirely reasonable (nothing about it depended on centimetre accuracy),
    while a precision measurement on Float would have been meaningless. A flat
    refusal would have blocked the first; permitting silently would have spoiled
    the second. The override makes the caller state which kind of run it is.
    """
    blade = _runtime_blade_diagnostics(telemetry, rpm_stale_register=rpm_stale_register)
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
    # Advisory only -- deliberately NOT appended to blockers. See
    # _RTK_REPORT_QUIET_SECONDS for why no threshold can gate on this.
    rtk_quiet = (
        rtk_report_age_seconds is not None
        and rtk_report_age_seconds > _RTK_REPORT_QUIET_SECONDS
    )

    rtk_label = (telemetry.get("position") or {}).get("rtk_status_label")
    # Unknown status does not block, matching the freshness rule: many diagnostic
    # and test callers build telemetry without it, and refusing on "unmeasured"
    # would break them. Only a positively-known non-Fix state refuses.
    rtk_degraded = rtk_label is not None and rtk_label not in _RTK_PRECISE_STATUS_LABELS
    if rtk_degraded and not allow_degraded_rtk:
        blockers.append("rtk_not_precise")
    return {
        "allowed_for_manual_motion": not blockers,
        "blockers": blockers,
        "blade_safe_for_motion": blade["blade_safe_for_motion"],
        "active_mowing_detected": active_mowing,
        "active_route_detected": route_status["route_present"],
        "active_route_status": route_status,
        "position_valid_for_motion": position_valid,
        # Diagnostic regardless of the verdict: a run that later looks wrong
        # can be checked against how fresh its positioning actually was.
        "rtk_report_age_seconds": rtk_report_age_seconds,
        "rtk_report_quiet_threshold_seconds": _RTK_REPORT_QUIET_SECONDS,
        # Advisory annotation, never a blocker: a stationary mower is legitimately
        # quiet for an hour at a time, so this cannot distinguish quiet from dead.
        "rtk_report_quiet": rtk_quiet,
        # Recorded whether or not it blocked, so a run's positioning quality is
        # always attributable after the fact.
        "rtk_status_label": rtk_label,
        "rtk_degraded": rtk_degraded,
        "rtk_degraded_override": bool(allow_degraded_rtk),
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
    return (
        [feature for feature in features if isinstance(feature, dict)]
        if isinstance(features, list)
        else []
    )


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
        _normalize_route_feature(feature)
        for feature in _geojson_features(mow_path_geojson)
    ]
    progress_features = [
        _normalize_route_feature(feature)
        for feature in _geojson_features(progress_geojson)
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
    route = (
        active_route if active_route is not None else _export_active_route(coordinator)
    )
    blade = _runtime_blade_diagnostics(telemetry)
    safety = _runtime_motion_safety_summary(
        telemetry,
        ha_state=ha_state,
        active_route=route,
        rtk_report_age_seconds=_rtk_report_age_seconds(coordinator),
    )
    ble_liveness = _ble_link_liveness(coordinator)
    capabilities = capability_snapshot(coordinator)
    route_status = safety["active_route_status"]
    pipeline_diagnostics = getattr(coordinator, "position_pipeline_diagnostics", None)
    return {
        "ha_state": ha_state,
        # The card's direction arrow reads this and nothing produced it until
        # 2026-08-26; see _current_orientation.
        "current_orientation": _current_orientation(coordinator, telemetry),
        # 🔑 The one place to ask "which way is it pointing, and is that fresh?".
        # `current_orientation.trustworthy` answers a weaker question -- see
        # _map_facing_report.
        "map_facing": _map_facing_report(coordinator, telemetry),
        "online": telemetry.get("online"),
        "work_mode": telemetry.get("work_mode"),
        "work_mode_label": telemetry.get("work_mode_label"),
        "charge_state": telemetry.get("charge_state"),
        "charge_state_label": telemetry.get("charge_state_label"),
        "position": telemetry.get("position"),
        "position_candidates": telemetry.get("position_candidates"),
        "position_pipeline": (
            pipeline_diagnostics()
            if callable(pipeline_diagnostics)
            else {"available": False, "reason": "position_stream_unavailable"}
        ),
        "rapid_state_fusion": _rapid_state_fusion_snapshot(coordinator),
        "blade": blade,
        "transport": telemetry.get("transport"),
        "active_route_summary": {
            "mow_path_feature_count": route.get("mow_path_feature_count", 0),
            "mow_progress_feature_count": route.get("mow_progress_feature_count", 0),
            "active_progress": route.get("active_progress"),
            "status": route_status,
        },
        "safety": safety,
        "capability_registry": capabilities,
        "experimental_motion": experimental_motion_status(
            coordinator,
            ble_liveness=ble_liveness,
            safety=safety,
        ),
        "manual_motion_execution_policy": _manual_motion_execution_policy(),
        "last_task_sync": _isoformat_or_none(
            getattr(coordinator, "last_task_sync", None)
        ),
        "last_map_sync": _isoformat_or_none(
            getattr(coordinator, "last_map_sync", None)
        ),
        "last_map_task_error": getattr(coordinator, "last_map_task_error", None),
        "active_transport": getattr(coordinator, "active_transport_state", None),
        "ble_only_fallback_mode": getattr(coordinator, "ble_only_fallback_mode", None),
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
            "wifi" if DEFAULT_EXPERIMENTAL_SEGMENT_USE_WIFI else "ble_preferred"
        ),
        "default_stop_mode": DEFAULT_EXPERIMENTAL_SEGMENT_STOP_MODE,
        "default_pulses_per_burst": (DEFAULT_EXPERIMENTAL_SEGMENT_PULSES_PER_BURST),
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
                _point_in_polygon(point, polygon) for polygon in valid_polygons.values()
            ):
                outside.append(index)
        if outside:
            errors.append("path_points_outside_known_area_geometry")

    # 🚨 EXCLUSION, added 2026-08-20. Being inside a mowing area says nothing
    # about keep-outs INSIDE that area: on a supervised run a 10.8 m leg stayed
    # within "Backyard Right" the whole way and drove into an obstacle zone
    # containing a trampoline. Inclusion and exclusion are separate questions
    # and this check only ever asked the first one.
    #
    # Deliberately OUTSIDE the `valid_polygons` branch above: a keep-out must be
    # honoured even when no area geometry is available to contain the path.
    keep_outs = _keep_out_polygons(coordinator)
    keep_out_violations = _keep_out_violations(normalized_points, keep_outs)
    keep_out_leg_violations = _keep_out_leg_violations(normalized_points, keep_outs)
    if keep_out_violations:
        errors.append("path_points_inside_keep_out_zone")
    if keep_out_leg_violations:
        errors.append("path_legs_cross_keep_out_zone")
    if not keep_outs:
        # Silence here would read as "no keep-outs on the path" when it means
        # "no keep-out geometry was loaded". Those are very different, and the
        # second one is how the trampoline run passed validation.
        warnings.append("no_keep_out_geometry_available_for_exclusion_check")

    distance = _path_distance(normalized_points)
    if distance == 0 and len(normalized_points) >= 2:
        errors.append("path_distance_must_be_greater_than_zero")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        # Echo it or it is unprovable: a run record must show HOW MANY keep-outs
        # were actually checked, so "no violation" can be distinguished from
        # "nothing was loaded to violate".
        "keep_out_zones_checked": len(keep_outs),
        "keep_out_violations": keep_out_violations,
        "keep_out_leg_violations": keep_out_leg_violations,
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


_RAPID_FUSE_STATUS_LABELS = {
    0: "NO_POSE",
    1: "RTK_FIXED",
    2: "RTK_EXTENDED_VISION",
    3: "VISION_EXTENDED",
    4: "VISION_EXTENDED_FAILED",
}


def _rapid_state_fusion_snapshot(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Return rapid-state fusion fields without changing motion responses.

    ``report_data.dev.fuse_status`` is a different, undocumented 0-5 sub-byte
    from ``vslam_status``. Item 17 needs ``mowing_state.fuse_status``, decoded
    by pymammotion from tard-state word 16 bits 8-15. Keep both values named by
    source so a numeric ``1`` cannot be silently interpreted as the wrong enum.

    This helper is consumed only by ``export_runtime_state``. Adding it to the
    shared custom-path telemetry snapshot would alter every VIO execution
    response, violating the frozen daylight-path evidence contract.
    """
    data = coordinator.data
    raw_fuse_status = _safe_attr_path(data, "mowing_state.fuse_status")
    try:
        fuse_status = int(raw_fuse_status) if raw_fuse_status is not None else None
    except TypeError, ValueError:
        fuse_status = None
    return {
        "source": "mowing_state.fuse_status (tard_state_data[16] bits 8-15)",
        "available": fuse_status is not None,
        "fuse_status": fuse_status,
        "fuse_status_label": (
            _RAPID_FUSE_STATUS_LABELS.get(fuse_status, "UNKNOWN")
            if fuse_status is not None
            else None
        ),
        "vision_state_raw": _safe_attr_path(data, "mowing_state.vision_state_raw"),
        "device_vslam_fuse_status": _safe_attr_path(
            data, "report_data.dev.fuse_status"
        ),
        "device_vslam_source": "report_data.dev.fuse_status (distinct 0-5 field)",
    }


def _scale_report_position(value: Any) -> float | None:
    """Scale raw report map-local integer position fields to mower-map units."""
    if value is None:
        return None
    try:
        return float(value) / 10_000
    except TypeError, ValueError:
        return None


def _position_mode_label(pos_level: Any) -> str | None:
    """Return a readable position quality label from a pos_level value."""
    if pos_level is None:
        return None
    try:
        from pymammotion.data.model.enums import PositionMode  # noqa: PLC0415

        return PositionMode.from_value(int(pos_level)).name
    except TypeError, ValueError:
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
    except TypeError, ValueError:
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
    except TypeError, ValueError:
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
    except TypeError, ValueError:
        return False


def _is_area_out(pos_type: Any, zone_hash: Any) -> bool:
    """Return true when position metadata indicates outside/no mapped area."""
    try:
        pos_type_int = int(pos_type)
    except TypeError, ValueError:
        pos_type_int = None
    try:
        zone_hash_int = int(zone_hash)
    except TypeError, ValueError:
        zone_hash_int = None
    return pos_type_int == 0 and zone_hash_int == 0


def _is_stale_zero_area_out_pose(x: Any, y: Any, pos_type: Any, zone_hash: Any) -> bool:
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
    map_bol_hash = _safe_attr_path(report_location, "bol_hash")

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
        # ``rpt_dev_location`` carries BOTH ``zone_hash`` (proto field 5, the
        # mowing zone the mower is currently inside) and ``bol_hash`` (field 6,
        # a MurMur checksum of the device's whole area set).  This used to read
        # ``bol_hash``, which silently disabled every zone-based guard: a map
        # checksum is non-zero whenever a map exists, so the stale-dock-pose
        # rejection, the ``location_metadata`` overlay and the
        # ``zone_hash_unavailable``/``zone_hash_changed`` degradation reasons
        # could never fire.  Field 5 is the zone; keep the checksum separate.
        candidate_zone_hash = _safe_attr_path(report_location, "zone_hash")
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

    if (
        source == "unavailable"
        and mowing_state is not None
        and (
            _safe_attr_path(mowing_state, "pos_x") is not None
            or _safe_attr_path(mowing_state, "pos_y") is not None
        )
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
    if (
        source == "unavailable"
        and mowing_state is not None
        and (
            _safe_attr_path(mowing_state, "pos_x") is not None
            or _safe_attr_path(mowing_state, "pos_y") is not None
        )
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
        # The device's whole-map checksum, reported alongside the zone in the
        # same message.  Not a position field — exposed only so map-sync
        # forensics can compare it against our locally computed bol_hash.
        "map_bol_hash": (
            _json_safe_int(map_bol_hash) if map_bol_hash is not None else None
        ),
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


def _position_source_comparison(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Capture both independent position sources + RTK quality for one instant.

    The map-local feed lags and can invent motion (live 2026-07-15: a physical
    no-op pulse reported ~9cm). To later tell a phantom feed-jump from real motion,
    log the two sources side by side -- ``report_data.locations[0]``
    (``real_pos_x/y``, scaled) vs ``mowing_state`` (``pos_x/pos_y``) -- plus the
    RTK/pos-level quality a re-anchor would move. Capture only: the phantom
    detector gets built once a daylight run shows how the two sources diverge.
    """
    data = coordinator.data
    report_location = _latest_location(data)
    mowing_state = _safe_attr_path(data, "mowing_state")

    locations_xy: tuple[float, float] | None = None
    locations_stale_zero = False
    if report_location is not None:
        lx = _scale_report_position(_safe_attr_path(report_location, "real_pos_x"))
        ly = _scale_report_position(_safe_attr_path(report_location, "real_pos_y"))
        if lx is not None and ly is not None:
            # Drop the known post-restart stale (0,0)/AREA_OUT pose (same guard as
            # _custom_path_position_snapshot); otherwise it reads as a huge source
            # divergence in agreement_m exactly in the window phantom analysis
            # cares about. Keep a flag so the raw fact stays visible.
            if _is_stale_zero_area_out_pose(
                lx,
                ly,
                _safe_attr_path(report_location, "pos_type"),
                _safe_attr_path(report_location, "bol_hash"),
            ):
                locations_stale_zero = True
            else:
                locations_xy = (lx, ly)

    mowing_xy: tuple[float, float] | None = None
    if mowing_state is not None:
        mx = _safe_attr_path(mowing_state, "pos_x")
        my = _safe_attr_path(mowing_state, "pos_y")
        if mx is not None and my is not None:
            mowing_xy = (float(mx), float(my))

    agreement_m: float | None = None
    if locations_xy is not None and mowing_xy is not None:
        agreement_m = round(
            math.hypot(locations_xy[0] - mowing_xy[0], locations_xy[1] - mowing_xy[1]),
            4,
        )

    return {
        "locations_xy": locations_xy,
        "locations_stale_zero": locations_stale_zero,
        "mowing_state_xy": mowing_xy,
        "agreement_m": agreement_m,
        "rtk_status": _safe_attr_path(mowing_state, "rtk_status"),
        "pos_level": _safe_attr_path(mowing_state, "pos_level"),
        "pos_type": _safe_attr_path(mowing_state, "pos_type"),
        "zone_hash": _safe_attr_path(mowing_state, "zone_hash"),
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
                # Field 5 (zone), not field 6 (map checksum) — see
                # _custom_path_position_snapshot.
                zone_hash=_safe_attr_path(report_location, "zone_hash"),
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
        "online": coordinator.is_online()
        if hasattr(coordinator, "is_online")
        else None,
        "work_mode": work_mode,
        "work_mode_label": device_mode(work_mode) if work_mode is not None else None,
        "charge_state": charge_state,
        "charge_state_label": _charge_state_label(charge_state),
        "position": _custom_path_position_snapshot(data, coordinator),
        "position_candidates": _custom_path_position_candidates(data, coordinator),
        "blade": {
            "reported_state": _enum_value(blade_state),
            "reported_state_label": _blade_state_label(blade_state),
            "knife_status": _safe_attr_path(
                data, "report_data.knife_status.knife_status"
            ),
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
            "connection_label": device_connection(connect)
            if connect is not None
            else None,
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
    if (
        decision.get("action") == "forward"
        and decision.get("reason") == "heading_aligned"
    ):
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
                "corrected_heading_degrees": decision.get("corrected_heading_degrees"),
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
        if use_wifi:
            ack = await method(speed=speed, use_wifi=True)
        else:
            command, command_kwargs = {
                "forward": ("move_forward", {"linear": speed}),
                "backward": ("move_back", {"linear": speed}),
                "turn_left": ("move_left", {"angular": speed}),
                "turn_right": ("move_right", {"angular": speed}),
            }[action]
            await _send_ble_motion_command_confirmed(
                coordinator,
                command,
                command_kwargs=command_kwargs,
            )
            ack = True
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
        if use_wifi:
            ack = await coordinator.async_stop_manual_motion(use_wifi=True)
        else:
            # A zero-speed write is always more urgent than queued telemetry or
            # another normal command. Live 2026-08-02 a normal-priority stop
            # took 1392.7 ms to confirm while the mower continued past its
            # target. Emergency priority preserves confirmed delivery while
            # putting the stop ahead of non-safety queue work.
            ack = await _stop_manual_motion_confirmed(coordinator, emergency=True)
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


async def _stop_manual_motion_confirmed(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    emergency: bool = False,
) -> dict[str, bool]:
    """Send one zero-velocity BLE command and await its GATT write."""
    await _send_ble_motion_command_confirmed(
        coordinator,
        "send_movement",
        command_kwargs={"linear_speed": 0, "angular_speed": 0},
        emergency_stop=emergency,
    )
    return {"movement_ok": True}


_OPERATOR_STOP_WRITE_ATTEMPTS = 3
_OPERATOR_STOP_OWNER_TIMEOUT_SECONDS = 8.0


async def _stop_active_manual_motion(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Abort the active session and deliver a bounded confirmed BLE stop."""
    started = time.monotonic()
    session = active_motion_session(coordinator)
    if session is not None:
        # Cancellation is visible before the first stop is queued. Every
        # nonzero confirmed-dispatch path checks this flag immediately before
        # building/queueing its write.
        session.cancelled = True
        session.cancel_reason = "operator_stop"
        session.phase = "stopping"

    attempts: list[dict[str, Any]] = []
    for attempt_number in range(1, _OPERATOR_STOP_WRITE_ATTEMPTS + 1):
        attempt_started = time.monotonic()
        try:
            await _stop_manual_motion_confirmed(coordinator, emergency=True)
        except Exception as err:  # noqa: BLE001 - all attempts are reported
            attempts.append(
                {
                    "attempt": attempt_number,
                    "ok": False,
                    "error": f"{type(err).__name__}: {err}",
                    "duration_ms": round(
                        (time.monotonic() - attempt_started) * 1000, 1
                    ),
                }
            )
        else:
            attempts.append(
                {
                    "attempt": attempt_number,
                    "ok": True,
                    "duration_ms": round(
                        (time.monotonic() - attempt_started) * 1000, 1
                    ),
                }
            )

    owner_exited = session is None
    if session is not None:
        try:
            await asyncio.wait_for(
                session.owner_done.wait(),
                timeout=_OPERATOR_STOP_OWNER_TIMEOUT_SECONDS,
            )
            owner_exited = True
        except TimeoutError:
            owner_exited = False

    result = {
        "service": SERVICE_STOP_MANUAL_MOTION,
        "session_id": session.session_id if session is not None else None,
        "session_was_active": session is not None,
        "aborted": session is not None,
        "stop_confirmed": any(attempt["ok"] for attempt in attempts),
        "all_stop_writes_confirmed": all(attempt["ok"] for attempt in attempts),
        "attempts": attempts,
        "owner_exited": owner_exited,
        "owner_wait_timeout_seconds": _OPERATOR_STOP_OWNER_TIMEOUT_SECONDS,
        "duration_ms": round((time.monotonic() - started) * 1000, 1),
    }
    if session is not None:
        session.stop_result = result
        if owner_exited:
            session.phase = "aborted"
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
    return _is_manual_motion_area_label(
        position.get("pos_type_label")
    ) and position.get("zone_hash") not in (None, 0, "0")


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
        path_progress_distance = (
            float(delta["dx"]) * unit_x + float(delta["dy"]) * unit_y
        )
        target_heading = (math.degrees(math.atan2(target_dy, target_dx)) + 360) % 360

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
    except TypeError, ValueError:
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
    elif baseline_position.get("zone_hash") not in (
        None,
        0,
        "0",
    ) and current_position.get("zone_hash") != baseline_position.get("zone_hash"):
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
    if (
        baseline_rtk is not None
        and current_rtk is not None
        and current_rtk < baseline_rtk
    ):
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


def _ble_connect_cooldown_active(
    coordinator: MammotionReportUpdateCoordinator,
) -> bool:
    """Return True while the BLE transport's connect-failure cooldown is armed.

    pymammotion exposes no public cooldown flag; ``BLETransport`` documents
    ``_connect_cooldown_until`` as the monotonic deadline (0.0 when inactive),
    reached via the public ``DeviceHandle.get_transport``. Any missing piece
    reads as "no cooldown" so this never blocks recovery on API drift.
    """
    try:
        handle = coordinator.manager.mower(coordinator.device_name)
    except Exception:  # noqa: BLE001
        return False
    get_transport = getattr(handle, "get_transport", None)
    if get_transport is None:
        return False
    try:
        transport = get_transport(TransportType.BLE)
        deadline = float(getattr(transport, "_connect_cooldown_until", 0.0))
    except Exception:  # noqa: BLE001
        return False
    return time.monotonic() < deadline


def _ble_transport_usable(
    coordinator: MammotionReportUpdateCoordinator,
) -> bool:
    """Return False only when the BLE transport positively reports itself unusable.

    ``DeviceHandle.active_transport()`` can hand back the BLE transport as the
    routing choice while ``BLETransport.is_usable`` is False -- that property
    additionally requires a live ``BLEDevice``, an advertisement RSSI at or above
    ``config.min_rssi``, and no armed connect cooldown. "Selected for routing" is
    therefore a weaker claim than "can carry a command".

    Live evidence (2026-07-19): a real motion command returned ``command_ok`` with
    a dual-axis stop ACK and the mower never moved, while HA's own
    ``emergency_nudge_*`` buttons -- which gate on ``is_usable`` via
    ``button.py::_nudge_available`` -- were correctly greyed out. The entity layer
    knew the link was dead; the service layer did not, because it only compared
    the transport label.

    Any missing piece reads as "usable" so this degrades to the pre-existing
    label-only behaviour rather than blocking all motion on pymammotion API drift.
    """
    try:
        handle = coordinator.manager.mower(coordinator.device_name)
    except Exception:  # noqa: BLE001
        return True
    get_transport = getattr(handle, "get_transport", None)
    if get_transport is None:
        return True
    try:
        transport = get_transport(TransportType.BLE)
        usable = getattr(transport, "is_usable", None)
    except Exception:  # noqa: BLE001
        return True
    return True if usable is None else bool(usable)


def _ble_ready_for_motion(coordinator: MammotionReportUpdateCoordinator) -> bool:
    """Return True when BLE is both the active transport and actually usable."""
    return _transport_is_ble(coordinator) and _ble_transport_usable(coordinator)


#: pymammotion's ``ble_loop._KEEP_ALIVE_BLE_INTERVAL`` -- a healthy BLE transport
#: writes a ``todev_ble_sync(2)`` heartbeat this often, so the age of the last
#: outbound send is bounded by it whenever the link is genuinely carrying traffic.
_BLE_KEEPALIVE_INTERVAL_SECONDS = 5.0
#: Age of the last BLE send beyond which the transport is treated as stalled.
#: Three missed heartbeats -- comfortably outside normal jitter, and far inside
#: the ~20-55s queue stalls observed on 2026-07-28.
_BLE_SEND_STALL_SECONDS = 3 * _BLE_KEEPALIVE_INTERVAL_SECONDS
#: Real motion is never allowed behind existing queue work. The command API
#: returns after enqueue, so even one predecessor makes the local pulse timer
#: diverge from the mower's actual execution window.
_BLE_QUEUE_DEPTH_LIMIT = 0
#: Maximum time a motion item may wait to start in the command queue. If this
#: expires the item is disarmed, so a later queue recovery cannot execute it.
_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS = 2.0
#: Maximum time allowed for the BLE GATT write itself. Motion timing begins only
#: after this awaited write completes.
_BLE_MOTION_WRITE_TIMEOUT_SECONDS = 4.0


def _ble_link_liveness(  # noqa: C901
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Report whether BLE is safe enough to enter confirmed motion dispatch.

    This exists because ``_ble_transport_usable`` answers a different question
    than it appears to. ``BLETransport.is_usable`` is a *routing-eligibility*
    flag -- a ``BLEDevice`` is cached, its advertisement RSSI clears
    ``min_rssi``, and no connect cooldown is armed. None of that requires a live
    GATT link, and none of it notices that commands are piling up undelivered.

    Live evidence (2026-07-28): pymammotion leaks proxy connection slots (see
    ``docs/pymammotion-ble-slot-leak-bug.md``), so the ESPHome proxy runs out,
    ``DeviceCommandQueue`` gates, and every command -- *including the mandatory
    stop that bounds a motion pulse* -- accumulates. A command issued at 21:06:20
    produced one send and silence; the queue flushed at 21:06:41-43 and the mower
    drove 1.0778 m at 21:07:16, long after the executor had sampled the window
    and reported it stationary. Throughout, ``active_transport`` read ``ble``,
    ``is_usable`` was True, RSSI was -64 dBm, and ``command_result.ok`` was True.

    This is a conservative preflight snapshot, not proof that the next write
    will complete. In pinned pymammotion ``last_send_monotonic`` is stamped
    before the GATT write is awaited, and queue state can change immediately
    after inspection. Real motion therefore also uses
    ``_send_ble_motion_command_confirmed``: it waits for queue start and GATT
    completion, disarms late queue items, and never falls back to MQTT.

    Unlike the other helpers here, missing introspection reads as **not live**
    rather than as live. Degrading permissive is what let unbounded motion
    through: a gate that silently passes when it cannot see is exactly the
    vacuously-true failure this project has already been bitten by twice. The
    cost of being wrong is asymmetric -- a false block wastes a run, a false pass
    puts an unstoppable mower in a yard.

    Returns:
        A diagnostic mapping. ``live`` is the gate verdict; ``reason`` names the
        first failing check (``None`` when live). The remaining keys are
        ``None`` when that field could not be read.

    """
    report: dict[str, Any] = {
        "live": False,
        "reason": None,
        "is_connected": None,
        "is_usable": None,
        "cooldown_active": None,
        "cooldown_remaining_seconds": None,
        "last_send_age_seconds": None,
        "queue_depth": None,
        "queue_dispatch_paused": None,
        "saga_active": None,
        "stall_threshold_seconds": _BLE_SEND_STALL_SECONDS,
        "queue_depth_limit": _BLE_QUEUE_DEPTH_LIMIT,
    }

    try:
        handle = coordinator.manager.mower(coordinator.device_name)
    except Exception:  # noqa: BLE001
        handle = None
    if handle is None:
        report["reason"] = "device_handle_unavailable"
        return report

    get_transport = getattr(handle, "get_transport", None)
    if get_transport is None:
        report["reason"] = "get_transport_unavailable"
        return report
    try:
        transport = get_transport(TransportType.BLE)
    except Exception:  # noqa: BLE001
        transport = None
    if transport is None:
        report["reason"] = "ble_transport_not_registered"
        return report

    # --- transport-level reads (public Transport API) ---------------------
    for key, attr in (
        ("is_connected", "is_connected"),
        ("is_usable", "is_usable"),
    ):
        try:
            report[key] = bool(getattr(transport, attr))
        except Exception:  # noqa: BLE001
            report[key] = None

    try:
        deadline = float(transport._connect_cooldown_until)  # noqa: SLF001
        remaining = deadline - time.monotonic()
        report["cooldown_active"] = remaining > 0
        report["cooldown_remaining_seconds"] = round(max(remaining, 0.0), 1)
    except Exception:  # noqa: BLE001
        pass

    try:
        last_send = float(transport.last_send_monotonic)
    except Exception:  # noqa: BLE001
        last_send = None
    if last_send is not None and last_send > 0.0:
        report["last_send_age_seconds"] = round(time.monotonic() - last_send, 1)

    # --- queue-level reads -------------------------------------------------
    queue = getattr(handle, "queue", None)
    if queue is not None:
        with contextlib.suppress(Exception):
            report["saga_active"] = bool(queue.is_saga_active)
        # Both are private in pymammotion 0.8.8; there is no public equivalent.
        # Absence is handled below by refusing, not by passing.
        gate = getattr(queue, "_transport_gate", None)
        if gate is not None:
            with contextlib.suppress(Exception):
                report["queue_dispatch_paused"] = not gate.is_set()
        pending = getattr(queue, "_queue", None)
        if pending is not None:
            with contextlib.suppress(Exception):
                report["queue_depth"] = int(pending.qsize())

    # --- verdict -----------------------------------------------------------
    if report["is_connected"] is not True:
        report["reason"] = "ble_client_not_connected"
        return report
    if report["is_usable"] is not True:
        report["reason"] = "ble_transport_not_usable"
        return report
    if report["cooldown_active"] is not False:
        report["reason"] = "ble_connect_cooldown_armed"
        return report
    if report["queue_dispatch_paused"] is not False:
        report["reason"] = "command_queue_dispatch_paused"
        return report
    if report["saga_active"] is not False:
        report["reason"] = "exclusive_saga_active"
        return report
    if report["queue_depth"] is None or report["queue_depth"] > _BLE_QUEUE_DEPTH_LIMIT:
        report["reason"] = "command_queue_backlogged"
        return report
    age = report["last_send_age_seconds"]
    if age is None:
        report["reason"] = "no_ble_send_observed"
        return report
    if age > _BLE_SEND_STALL_SECONDS:
        report["reason"] = "ble_send_stalled"
        return report

    report["live"] = True
    return report


#: Queue states that clear on their own within a moment. Anything else (no
#: client, transport unusable, connect cooldown, exclusive saga) is a standing
#: condition that waiting cannot fix, so the settle loop returns immediately.
_BLE_TRANSIENT_QUEUE_REASONS = (
    "command_queue_backlogged",
    "command_queue_dispatch_paused",
)
_BLE_QUEUE_SETTLE_TIMEOUT_SECONDS = 6.0
_BLE_QUEUE_SETTLE_POLL_SECONDS = 0.1


#: Poll cadence for `_report_stream_probe`. The whole point of the probe is to
#: resolve a report rate that a 1.1 s sampler could not (the 2026-08-06
#: direction review reported its own sampling interval as the feed's rate), so
#: this must stay far below the shortest period under test. At 20 ms it
#: oversamples a 100 ms period 5x. It reads an in-memory float; no I/O.
_REPORT_PROBE_POLL_SECONDS = 0.02


async def _observe_report_arrivals(
    coordinator: MammotionReportUpdateCoordinator,
    handle: Any,
    duration_seconds: float,
) -> tuple[list[float], dict[str, list[float]]]:
    """Poll for report arrivals, split by whole-traffic and by channel.

    Returns ``(message_arrivals, per_channel_arrivals)``. The first counts every
    inbound ``LubaMsg``; the second attributes arrivals to the position, RTK and
    VIO channels independently, because a channel that never changes is a
    channel that is not reporting -- which whole-traffic counting cannot reveal.
    """
    arrivals: list[float] = []
    last_seen = handle.last_report_at
    channel_last = _report_channel_fingerprints(coordinator)
    channel_arrivals: dict[str, list[float]] = {k: [] for k in channel_last}
    deadline = time.monotonic() + duration_seconds
    while time.monotonic() < deadline:
        await asyncio.sleep(_REPORT_PROBE_POLL_SECONDS)
        stamped = handle.last_report_at
        if stamped != last_seen:
            last_seen = stamped
            arrivals.append(stamped)
        now = time.monotonic()
        for name, fingerprint in _report_channel_fingerprints(coordinator).items():
            if fingerprint != channel_last.get(name):
                channel_last[name] = fingerprint
                channel_arrivals[name].append(now)
    return arrivals, channel_arrivals


def _basestation_snapshot(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Read whatever the mower currently holds about the RTK base station."""
    info = getattr(
        getattr(getattr(coordinator, "data", None), "report_data", None),
        "basestation_info",
        None,
    )
    if info is None:
        return {"available": False}
    score = getattr(info, "score_info", None)
    return {
        "available": True,
        # Relayed base state. rtk_status here is the BASE's own view, which is
        # not necessarily the rover's `position.rtk_status_label`.
        "rtk_status": getattr(info, "rtk_status", None),
        "sats_num": getattr(info, "sats_num", None),
        "rtk_channel": getattr(info, "rtk_channel", None),
        "rtk_switch": getattr(info, "rtk_switch", None),
        "mqtt_rtk_status": getattr(info, "mqtt_rtk_status", None),
        "lora_channel": getattr(info, "lora_channel", None),
        "wifi_rssi": getattr(info, "wifi_rssi", None),
        "app_connect_type": getattr(info, "app_connect_type", None),
        "basestation_status": getattr(info, "basestation_status", None),
        "connect_status_since_poweron": getattr(
            info, "connect_status_since_poweron", None
        ),
        # The survey-hypothesis discriminator. base_moved / base_moving say
        # whether the base believes its own position changed, which is what
        # would trigger (or invalidate) a survey and leave it transmitting
        # corrections against a wrong reference -- the leading explanation for
        # the 2026-08-07 Float episode.
        "score_info": None
        if score is None
        else {
            "base_score": getattr(score, "base_score", None),
            "base_leve": getattr(score, "base_leve", None),
            "base_moved": getattr(score, "base_moved", None),
            "base_moving": getattr(score, "base_moving", None),
        },
    }


def _rtk_base_station_snapshot(device: Any) -> dict[str, Any]:
    """Read an ``RTKBaseStationDevice`` -- the base's own ``iot_id`` state.

    This is the *other* place a ``base.to_app`` reply can land. Frames carrying
    the base station's own ``iot_id`` are reduced by pymammotion's
    ``RTKStateReducer`` onto ``RTKBaseStationDevice``; only frames carrying the
    **mower's** ``iot_id`` reach ``report_data.basestation_info``
    (:func:`_basestation_snapshot`). A probe that reads only the mower can
    therefore see nothing while the base is answering perfectly well.

    ``MammotionRTKCoordinator._async_update_data`` already issues
    ``async_send_and_wait("basestation_info", "to_app")`` every tick, so on an
    installation with a separate RTK device this state is normally already
    populated without anyone asking.
    """
    if device is None:
        return {"available": False}
    score = getattr(device, "score_info", None)
    return {
        "available": True,
        "sats_num": getattr(device, "sats_num", None),
        "rtk_status": getattr(device, "rtk_status", None),
        "rtk_channel": getattr(device, "rtk_channel", None),
        "rtk_switch": getattr(device, "rtk_switch", None),
        "mqtt_rtk_status": getattr(device, "mqtt_rtk_status", None),
        "app_connect_type": getattr(device, "app_connect_type", None),
        "lora_channel": getattr(device, "lora_channel", None),
        "lora_scan": getattr(device, "lora_scan", None),
        "lora_locid": getattr(device, "lora_locid", None),
        "lora_netid": getattr(device, "lora_netid", None),
        "lowpower_status": getattr(device, "lowpower_status", None),
        "ble_rssi": getattr(device, "ble_rssi", None),
        "wifi_rssi": getattr(device, "wifi_rssi", None),
        "basestation_status": getattr(device, "basestation_status", None),
        "connect_status_since_poweron": getattr(
            device, "connect_status_since_poweron", None
        ),
        # The base's own reference position, in radians on this model. On
        # 2026-08-07 this moved exactly once -- 4.7 m at the power cycle, then
        # back 47 min later -- which is a base re-deriving and converging, not
        # a base that has lost its position.
        "lat": getattr(device, "lat", None),
        "lon": getattr(device, "lon", None),
        "lora_version": getattr(device, "lora_version", None),
        "online": getattr(device, "online", None),
        "score_info": None
        if score is None
        else {
            "base_score": getattr(score, "base_score", None),
            "base_leve": getattr(score, "base_leve", None),
            "base_moved": getattr(score, "base_moved", None),
            "base_moving": getattr(score, "base_moving", None),
        },
    }


def _rtk_base_station_sources(
    hass: HomeAssistant,
) -> list[tuple[str, MammotionRTKCoordinator]]:
    """Return ``(name, coordinator)`` for every configured RTK base station."""
    sources: list[tuple[str, MammotionRTKCoordinator]] = []
    for entry in hass.config_entries.async_entries(DOMAIN):
        runtime_data = getattr(entry, "runtime_data", None)
        if not runtime_data:
            continue
        sources.extend(
            (rtk.name, rtk.coordinator)
            for rtk in getattr(runtime_data, "RTK", None) or []
        )
    return sources


#: Fields on ``report_data.basestation_info`` that **only** a
#: ``response_basestation_info_t`` reply can populate. The report channel's
#: ``rpt_basestation_info`` carries versions, ``basestation_status`` and
#: ``connect_status_since_poweron`` and nothing else, so any of these going
#: non-default is proof the base answered our query rather than the mower
#: simply reporting.
_BASESTATION_QUERY_ONLY_FIELDS = (
    "sats_num",
    "rtk_status",
    "rtk_channel",
    "rtk_switch",
    "wifi_rssi",
    "lora_channel",
    "mqtt_rtk_status",
    "app_connect_type",
)


def _basestation_has_query_fields(snapshot: dict[str, Any]) -> bool:
    """Report whether *snapshot* carries data only a query reply could set."""
    if not snapshot.get("available"):
        return False
    if any(snapshot.get(name) for name in _BASESTATION_QUERY_ONLY_FIELDS):
        return True
    score = snapshot.get("score_info") or {}
    return any(
        score.get(name)
        for name in ("base_score", "base_leve", "base_moved", "base_moving")
    )


async def _basestation_info_probe(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    wait_seconds: float,
    rtk_sources: Sequence[tuple[str, MammotionRTKCoordinator]] | None = None,
) -> dict[str, Any]:
    """Ask the RTK base station to report its own status.

    Read-only with respect to motion: this dispatches one ``basestation_info``
    query (``BaseStation.to_dev = request_basestation_info_t(request_type=1)``)
    and reads the reply pymammotion stores on
    ``report_data.basestation_info``. No movement command is sent.

    Why this exists: on 2026-08-07 RTK sat in Float for three hours while the
    rover's own reception was healthy -- 24 co-viewed satellites, corrections
    flowing, no reference-station error. Only power-cycling the base cleared it.
    The leading explanation is a base transmitting well-formed corrections
    against a **wrong reference position** (a survey that never converged), and
    ``score_info.base_moved`` / ``base_moving`` is the field that would say so.
    Nothing in this integration had ever asked the base anything.

    ⚠️ It is **unverified** that this hardware answers the query. pymammotion
    defines the message and reduces the response, but a message existing in the
    library is not the hardware replying to it. ``answered`` reports which.

    ⚠️ **Why this samples instead of comparing before/after.** A first live run
    on 2026-08-07 (beta27) returned ``no_change_observed`` -- which turned out to
    be uninterpretable, because a single ``BasestationInfo`` dataclass holds
    *both* field groups and the report channel **replaces the whole object**::

        # pymammotion/data/model/report_info.py:646
        if data.basestation_info is not None:
            self.basestation_info = BasestationInfo.from_dict(...)

    That ``from_dict`` is built from ``rpt_basestation_info``, which carries only
    versions, ``basestation_status`` and ``connect_status_since_poweron``. Every
    such report therefore resets ``sats_num`` / ``rtk_status`` / ``score_info``
    to defaults -- **wiping any query reply that arrived first.** Comparing a
    single ``after`` against ``before`` races that clobber and a longer wait
    makes it *more* likely to lose, not less.

    So this polls the struct through the wait window and keeps the
    most-informative sample it ever sees. A reply is recognised by any
    query-only field going non-default, which the report channel can never do.

    ⚠️ **And it watches two read paths, because the mower is often the wrong
    one.** A reply carrying the base station's own ``iot_id`` is reduced onto
    ``RTKBaseStationDevice``, never onto the mower's ``report_data``. The first
    corrected run (beta28, 101 samples over 15 s) saw nothing on the mower path
    while the installation had a perfectly live separate RTK device --
    ``rtk_over_internet``, 26 satellites, reporting its own coordinates. Passing
    ``rtk_sources`` (see :func:`_rtk_base_station_sources`) watches both, and
    ``answered_via`` records which path actually produced the answer.
    """
    poll_interval = 0.15
    rtk_sources = list(rtk_sources or [])
    result: dict[str, Any] = {
        "motion_commanded": False,
        "wait_seconds": wait_seconds,
        "poll_interval_seconds": poll_interval,
        "command_sent": False,
        "answered": False,
        "answered_via": None,
        "samples": 0,
        "before": _basestation_snapshot(coordinator),
        "best": None,
        "final": None,
        "clobbered_after_answer": False,
        "rtk_devices": [],
        "reason": None,
    }
    try:
        sent = await coordinator.async_send_command("basestation_info")
        result["command_sent"] = sent is not False
        if not result["command_sent"]:
            result["reason"] = "command_refused_device_offline_or_unavailable"
            return result

        deadline = time.monotonic() + max(wait_seconds, 0.0)
        best: dict[str, Any] | None = None
        latest: dict[str, Any] = result["before"]
        rtk_best: dict[str, dict[str, Any]] = {}
        rtk_latest: dict[str, dict[str, Any]] = {}
        samples = 0
        while True:
            latest = _basestation_snapshot(coordinator)
            samples += 1
            if _basestation_has_query_fields(latest) and best is None:
                best = latest
            for name, rtk_coordinator in rtk_sources:
                snapshot = _rtk_base_station_snapshot(
                    getattr(rtk_coordinator, "data", None)
                )
                rtk_latest[name] = snapshot
                if _basestation_has_query_fields(snapshot) and name not in rtk_best:
                    rtk_best[name] = snapshot
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(poll_interval)

        result["samples"] = samples
        result["final"] = latest
        result["best"] = best or latest
        result["rtk_devices"] = [
            {
                "name": name,
                "answered": name in rtk_best,
                "best": rtk_best.get(name) or rtk_latest.get(name),
                "final": rtk_latest.get(name),
            }
            for name, _ in rtk_sources
        ]

        # Either read path answering counts, and which one is recorded --
        # a probe that only watched the mower would have called a live base
        # silent on 2026-08-07.
        via: list[str] = []
        if best is not None:
            via.append("mower_report_data")
            result["clobbered_after_answer"] = not _basestation_has_query_fields(latest)
        via.extend(f"rtk_device:{name}" for name in rtk_best)
        result["answered"] = bool(via)
        result["answered_via"] = via or None
        if via:
            result["reason"] = "answered"
        else:
            # Never "the base is dead": a base that answers with all-default
            # values is indistinguishable from one that never answered, and an
            # installation with no separate RTK device has one fewer place to
            # look rather than a fault.
            result["reason"] = "no_query_fields_observed"
    except Exception as err:  # noqa: BLE001
        result["reason"] = f"{type(err).__name__}: {err}"
    return result


async def _ota_info_probe(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    send_timeout: float,
) -> dict[str, Any]:
    """Ask the device to report its own OTA info.

    Read-only: sends one ``MctlOta.todev_get_info_req(type=IT_OTA)`` and
    returns the raw ``toapp_get_info_rsp``. Only ever sends a get-info
    REQUEST -- never ``fw_download_ctrl``, never ``device/upgrade`` -- so
    this cannot trigger an install.

    Why this exists: pymammotion's ``MessageOta.get_device_ota_info``
    (EMBED_OTA / ``SubOtaMsg``) is a real, fully-defined request the
    device's own protocol supports, but it is never called anywhere in
    pymammotion or this integration -- nothing has ever exercised this
    request/response path. The broker already recognises the ``ota`` field
    group for correlation (``messaging/broker.py``: ``"ota": "SubOtaMsg"``),
    so ``send_command_and_wait`` resolves it the same way every other
    probe in this file does.

    ⚠️ The response shape (``GetInfoRsp.ota`` -> ``OtaInfo``:
    otaid/version/progress/result/message) is the same progress-only shape
    already seen over the cloud MQTT relay -- it carries no download URL.
    The URL-bearing variant (``fota_sub_info.sub_img_url``) is a sibling
    branch of the same oneof, and how pymammotion itself uses it
    (``send_swimming_pool_device_ota_second``) is to construct and SEND
    it, i.e. an app-to-device push of a URL the app already obtained via
    its own cloud fetch -- not something this probe would receive back.
    Kept as a genuine, unproven read rather than assumed useless: the
    full raw response is returned so that assumption can be checked
    against what the hardware actually answers.
    """
    result: dict[str, Any] = {
        "command_sent": False,
        "answered": False,
        "response": None,
        "reason": None,
    }
    device = coordinator.manager.get_device_by_name(coordinator.device_name)
    if device is None or not coordinator.is_online():
        result["reason"] = "device_offline_or_unavailable"
        return result
    try:
        response = await coordinator.manager.send_command_and_wait(
            coordinator.device_name,
            "get_device_ota_info",
            "toapp_get_info_rsp",
            send_timeout=send_timeout,
            prefer_ble=True,
            log_type=1,
        )
        result["command_sent"] = True
        result["answered"] = True
        result["response"] = response.to_dict(include_default_values=True)
        result["reason"] = "answered"
    except CommandTimeoutError:
        result["command_sent"] = True
        result["reason"] = "no_response_within_timeout"
    except ConcurrentRequestError as exc:
        result["reason"] = f"concurrent_request: {exc}"
    except Exception as err:  # noqa: BLE001
        result["reason"] = f"{type(err).__name__}: {err}"
    return result


def _summarise_channel_arrivals(
    name: str, stamps: list[float], duration_seconds: float
) -> dict[str, Any]:
    """Reduce one channel's arrival times to a cadence summary."""
    intervals = [
        round((later - earlier) * 1000, 1)
        for earlier, later in zip(stamps, stamps[1:], strict=False)
    ]
    summary: dict[str, Any] = {
        "channel": name,
        "updates": len(stamps),
        "updates_per_second": round(len(stamps) / max(duration_seconds, 1e-6), 3),
        "intervals_ms": intervals,
    }
    if intervals:
        ordered = sorted(intervals)
        summary |= {
            "min_ms": ordered[0],
            "median_ms": ordered[len(ordered) // 2],
            "max_ms": ordered[-1],
            "mean_ms": round(sum(ordered) / len(ordered), 1),
        }
    else:
        # Zero updates across the whole window is the finding, not a gap in the
        # data: a stationary mower's position genuinely may not change, but a
        # silent RTK channel is the 2026-08-07 latch.
        summary["note"] = "no updates observed in the window"
    return summary


def _report_channel_fingerprints(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, tuple[Any, ...]]:
    """Snapshot one fingerprint per report channel.

    ``last_report_at`` counts every inbound ``LubaMsg`` regardless of content,
    which is why the 2026-08-07 probe could measure total traffic but not the
    *position* cadence -- the number that actually governs overshoot. pymammotion
    mutates ``report_data`` in place as reports arrive, so fingerprinting each
    channel separately and watching which one changes attributes an arrival to
    its channel without needing per-message hooks.
    """
    data = getattr(coordinator, "data", None)
    report = getattr(data, "report_data", None)
    location = _latest_location(data)
    rtk = getattr(report, "rtk", None)
    vision = getattr(report, "vision_info", None)
    return {
        "position": (
            getattr(location, "real_pos_x", None),
            getattr(location, "real_pos_y", None),
            getattr(location, "real_toward", None),
        ),
        "rtk": (
            getattr(rtk, "status", None),
            getattr(rtk, "gps_stars", None),
            getattr(rtk, "lat_std", None),
            getattr(rtk, "lon_std", None),
        ),
        "vio": (
            getattr(vision, "heading", None),
            getattr(vision, "vio_state", None),
            getattr(vision, "tracked_feature_count", None),
        ),
    }


def _latency_distribution_ms(values: Sequence[float]) -> dict[str, Any]:
    """Return compact percentile diagnostics for monotonic stage durations."""
    ordered = sorted(max(float(value) * 1000.0, 0.0) for value in values)
    if not ordered:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}

    def percentile(fraction: float) -> float:
        index = max(math.ceil(fraction * len(ordered)) - 1, 0)
        return round(ordered[index], 3)

    return {
        "count": len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": round(ordered[-1], 3),
    }


def _position_stage_latency_summary(
    records: Sequence[tuple[Any, float]],
) -> dict[str, dict[str, Any]]:
    """Summarise every measured position pipeline stage."""
    stages: dict[str, list[float]] = {
        "receipt_to_decode": [],
        "decode_to_broker": [],
        "broker_to_reducer": [],
        "reducer_to_state_apply": [],
        "state_apply_to_publication": [],
        "receipt_to_publication": [],
        "publication_to_consumption": [],
        "receipt_to_consumption": [],
    }
    for sample, consumed_at in records:
        stages["receipt_to_decode"].append(
            sample.decoded_at_monotonic - sample.received_at_monotonic
        )
        stages["decode_to_broker"].append(
            sample.broker_completed_at_monotonic - sample.decoded_at_monotonic
        )
        stages["broker_to_reducer"].append(
            sample.reducer_completed_at_monotonic - sample.broker_completed_at_monotonic
        )
        stages["reducer_to_state_apply"].append(
            sample.state_applied_at_monotonic - sample.reducer_completed_at_monotonic
        )
        stages["state_apply_to_publication"].append(
            sample.published_at_monotonic - sample.state_applied_at_monotonic
        )
        stages["receipt_to_publication"].append(
            sample.published_at_monotonic - sample.received_at_monotonic
        )
        stages["publication_to_consumption"].append(
            consumed_at - sample.published_at_monotonic
        )
        stages["receipt_to_consumption"].append(
            consumed_at - sample.received_at_monotonic
        )
    return {name: _latency_distribution_ms(values) for name, values in stages.items()}


async def _wait_for_position_subscription_ready(  # noqa: C901
    handle: Any,
    position_stream: Any,
    generation: Any,
    *,
    lease: Any,
    timeout_seconds: float,
    not_before_monotonic: float | None = None,
    baseline_dropped_samples: int | None = None,
) -> tuple[Any | None, float | None, str | None]:
    """Require the first valid position in the current report generation."""
    deadline = time.monotonic() + timeout_seconds
    expected_sequence = generation.baseline_position_sequence + 1
    dropped_at_start = (
        position_stream.dropped_samples
        if baseline_dropped_samples is None
        else baseline_dropped_samples
    )
    while (remaining := deadline - time.monotonic()) > 0:
        if position_stream.dropped_samples != dropped_at_start:
            return None, None, "position_evidence_gap"
        if not handle.report_subscription_lease_is_current(lease):
            return None, None, "report_subscription_lease_lost"
        if handle.report_subscription_generation != generation.generation:
            return None, None, "report_subscription_generation_changed"
        if handle.position_epoch != generation.baseline_position_epoch:
            return None, None, "position_epoch_changed"
        try:
            sample = await asyncio.wait_for(position_stream.queue.get(), remaining)
        except TimeoutError:
            break
        consumed_at = time.monotonic()
        if position_stream.dropped_samples != dropped_at_start:
            return None, None, "position_evidence_gap"
        if sample.epoch != generation.baseline_position_epoch:
            return None, None, "position_epoch_changed"
        # A sample may have been queued between opening the stream and taking
        # the generation baseline. It is explicitly pre-generation evidence.
        if sample.sequence <= generation.baseline_position_sequence:
            continue
        if sample.sequence != expected_sequence:
            return None, None, "position_evidence_gap"
        expected_sequence += 1
        evidence_boundary = (
            generation.requested_at_monotonic
            if not_before_monotonic is None
            else max(generation.requested_at_monotonic, not_before_monotonic)
        )
        if sample.received_at_monotonic < evidence_boundary:
            continue
        if not sample.valid_for_motion:
            # Name the failing predicate. The sample already carries it, and
            # without it "position_invalid_for_motion" cannot distinguish an
            # environmental precondition (on the dock: zone_hash_unavailable)
            # from a telemetry fault (rtk_not_fixed, position_zero_pose) -- which
            # matters because the first is not a finding about report ownership
            # at all, and would otherwise fail all 30 transitions identically.
            return (
                None,
                None,
                f"position_invalid_for_motion: {sample.rejection_reason}",
            )
        return sample, consumed_at, None
    if not handle.report_subscription_lease_is_current(lease):
        return None, None, "report_subscription_lease_lost"
    if handle.report_subscription_generation != generation.generation:
        return None, None, "report_subscription_generation_changed"
    if handle.position_epoch != generation.baseline_position_epoch:
        return None, None, "position_epoch_changed"
    generic_advanced = handle.last_report_at > generation.baseline_last_report_at
    return (
        None,
        None,
        "position_channel_stalled" if generic_advanced else "position_channel_timeout",
    )


async def _collect_position_records(
    position_stream: Any,
    *,
    duration_seconds: float,
) -> list[tuple[Any, float]]:
    """Consume position samples live so controller latency remains measurable."""
    records: list[tuple[Any, float]] = []
    deadline = time.monotonic() + duration_seconds
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            sample = await asyncio.wait_for(position_stream.queue.get(), remaining)
        except TimeoutError:
            break
        records.append((sample, time.monotonic()))
    return records


async def _report_stream_probe(  # noqa: C901
    coordinator: MammotionReportUpdateCoordinator,
    *,
    period_ms: int,
    no_change_period_ms: int,
    duration_seconds: float,
    isolated: bool = False,
    report_lease: Any | None = None,
    readiness_timeout_seconds: float = 3.5,
) -> dict[str, Any]:
    """Measure how often the device actually reports at a requested period.

    Read-only with respect to motion: this dispatches the ``request_iot_sys``
    subscription config and nothing else. No movement command is sent, no
    motion confirmations are taken, and the experimental-motion gate is not
    consulted because nothing here can move the mower.

    ``period`` and ``no_change_period`` are *device-side* protocol fields
    (``ReportInfoCfg``), not client polling knobs, and both default to 1000 ms
    in pymammotion. Nothing in this integration has ever lowered them, so
    whether the device honours a shorter period is unmeasured. Aggregate report
    counts alone do not answer that question; the additive ``position_payloads``
    channel measures the relevant payloads. For a stationary mower the reported data is unchanging, so
    ``no_change_period`` (not ``period``) is what governs the cadence at rest;
    both are exposed so the caller can hold them equal.

    Aggregate arrivals are counted from ``DeviceHandle.last_report_at``, a monotonic
    timestamp stamped on every received ``LubaMsg``. It advances even when the
    mower is stationary and its coordinates never change, which is exactly why
    position-derived sampling could not measure this.
    """
    result: dict[str, Any] = {
        "period_ms": period_ms,
        "no_change_period_ms": no_change_period_ms,
        "duration_seconds": duration_seconds,
        "isolated": isolated,
        "poll_interval_ms": int(_REPORT_PROBE_POLL_SECONDS * 1000),
        "sampling_method": (
            "polled DeviceHandle.last_report_at every "
            f"{int(_REPORT_PROBE_POLL_SECONDS * 1000)} ms and recorded each "
            "distinct value; intervals are between successive report arrivals, "
            "not between polls"
        ),
        "motion_commanded": False,
        "subscription_attempted": False,
        "subscription_started": False,
        "subscription_stopped": False,
        "reports_observed": 0,
        "intervals_ms": [],
        "position_payloads": {
            "available": False,
            "observed": 0,
            "intervals_ms": [],
            "dropped_samples": 0,
        },
        "reason": None,
    }
    handle = coordinator.manager.mower(coordinator.device_name)
    if handle is None:
        result["reason"] = "device_handle_unavailable"
        return result
    # Never reconfigure the report stream underneath a live motion run: the
    # executor's own progress and completion checks read this feed.
    if getattr(coordinator, "manual_motion_owner", None) is not None:
        result["reason"] = "manual_motion_session_active"
        return result

    exclusive_context = None
    lease = report_lease
    if isolated and lease is None:
        exclusive_factory = getattr(handle, "exclusive_report_subscription", None)
        if not callable(exclusive_factory):
            result["reason"] = "exclusive_report_subscription_unavailable"
            return result
        exclusive_context = exclusive_factory("report_stream_probe")
        try:
            lease = await exclusive_context.__aenter__()
        except Exception as err:  # noqa: BLE001
            result["reason"] = (
                f"exclusive_subscription_failed: {type(err).__name__}: {err}"
            )
            return result

    if isolated and lease is None:
        result["reason"] = "report_subscription_lease_unavailable"
        if exclusive_context is not None:
            await exclusive_context.__aexit__(None, None, None)
        return result

    open_position_stream = getattr(coordinator, "open_position_sample_stream", None)
    position_stream = (
        open_position_stream(maxsize=2048) if callable(open_position_stream) else None
    )
    if isolated and position_stream is None:
        result["reason"] = "position_stream_unavailable"
        if exclusive_context is not None:
            await exclusive_context.__aexit__(None, None, None)
        return result
    begin_generation = getattr(handle, "begin_report_subscription_generation", None)
    lease_is_current = getattr(handle, "report_subscription_lease_is_current", None)
    if isolated and (
        not callable(begin_generation)
        or not callable(lease_is_current)
        or not isinstance(getattr(handle, "report_subscription_generation", None), int)
    ):
        result["reason"] = "report_subscription_generation_unavailable"
        if position_stream is not None:
            position_stream.close()
        if exclusive_context is not None:
            await exclusive_context.__aexit__(None, None, None)
        return result

    generation = (
        cast(Callable[[Any], Any], begin_generation)(lease) if isolated else None
    )
    position_drops_at_generation = (
        position_stream.dropped_samples if position_stream is not None else None
    )
    request_started_at = (
        generation.requested_at_monotonic
        if generation is not None
        else time.monotonic()
    )
    position_records: list[tuple[Any, float]] = []
    collector_task: asyncio.Task[list[tuple[Any, float]]] | None = None
    try:
        result["subscription_attempted"] = True
        await coordinator.manager.request_iot_sync_continuous(
            coordinator.device_name,
            period=period_ms,
            no_change_period=no_change_period_ms,
        )
        result["subscription_started"] = True
        if generation is not None:
            result["subscription_generation"] = {
                "owner": generation.owner,
                "lease_id": generation.lease_id,
                "generation": generation.generation,
                "requested_at_monotonic": generation.requested_at_monotonic,
                "baseline_position_sequence": generation.baseline_position_sequence,
                "baseline_position_epoch": generation.baseline_position_epoch,
                "baseline_last_report_at": generation.baseline_last_report_at,
            }
        result["queue_settle"] = await _settle_ble_command_queue(coordinator)
        # `request_iot_sync_continuous` returns when the command is QUEUED, not
        # when the device acknowledges it: pymammotion's client hands the send to
        # `DeviceCommandQueue.enqueue` and returns immediately. Draining that
        # queue is the first instant at which this generation's RPT_START is
        # known to have been written to the transport, so the flush -- not the
        # call's return -- is the evidence boundary. Stamping it at the return
        # instead would let a payload still arriving from the PREVIOUS
        # configuration satisfy this generation's readiness, which is exactly the
        # cross-generation acceptance this probe exists to rule out.
        subscription_command_flushed_at = time.monotonic()
        result["subscription_command_flushed_at_monotonic"] = (
            subscription_command_flushed_at
        )

        if isolated and position_stream is not None and generation is not None:
            (
                ready_sample,
                ready_consumed_at,
                readiness_reason,
            ) = await _wait_for_position_subscription_ready(
                handle,
                position_stream,
                generation,
                lease=lease,
                timeout_seconds=readiness_timeout_seconds,
                not_before_monotonic=subscription_command_flushed_at,
                baseline_dropped_samples=position_drops_at_generation,
            )
            result["position_readiness"] = {
                "ready": ready_sample is not None,
                "reason": readiness_reason,
                "timeout_seconds": readiness_timeout_seconds,
                "first_position_sequence": (
                    ready_sample.sequence if ready_sample is not None else None
                ),
                "first_position_after_request_ms": (
                    round(
                        (
                            ready_sample.received_at_monotonic
                            - generation.requested_at_monotonic
                        )
                        * 1000,
                        3,
                    )
                    if ready_sample is not None
                    else None
                ),
                "generic_report_advanced": (
                    handle.last_report_at > generation.baseline_last_report_at
                ),
            }
            if ready_sample is None or ready_consumed_at is None:
                result["reason"] = readiness_reason
                return result
            position_records.append((ready_sample, ready_consumed_at))

        collector_task = (
            asyncio.create_task(
                _collect_position_records(
                    position_stream, duration_seconds=duration_seconds
                )
            )
            if position_stream is not None
            else None
        )
        arrivals, channel_arrivals = await _observe_report_arrivals(
            coordinator, handle, duration_seconds
        )
        if collector_task is not None:
            position_records.extend(await collector_task)
        result["reports_observed"] = len(arrivals)
        result["channels"] = {
            name: _summarise_channel_arrivals(name, stamps, duration_seconds)
            for name, stamps in channel_arrivals.items()
        }
        result["intervals_ms"] = [
            round((later - earlier) * 1000, 1)
            for earlier, later in zip(arrivals, arrivals[1:], strict=False)
        ]
        if position_stream is not None:
            position_records = [
                record
                for record in position_records
                if record[0].received_at_monotonic >= request_started_at
            ]
            position_samples = [record[0] for record in position_records]
            position_arrivals = [
                position_sample.received_at_monotonic
                for position_sample in position_samples
            ]
            position_intervals = [
                round((later - earlier) * 1000, 1)
                for earlier, later in zip(
                    position_arrivals, position_arrivals[1:], strict=False
                )
            ]
            pipeline_latencies = [
                round(
                    (
                        position_sample.published_at_monotonic
                        - position_sample.received_at_monotonic
                    )
                    * 1000,
                    3,
                )
                for position_sample in position_samples
            ]
            result["position_payloads"] = {
                "available": True,
                "observed": len(position_samples),
                "valid_for_motion": sum(
                    position_sample.valid_for_motion
                    for position_sample in position_samples
                ),
                "intervals_ms": position_intervals,
                "pipeline_latencies_ms": pipeline_latencies,
                "dropped_samples": position_stream.dropped_samples,
                "sequence_gaps": sum(
                    later.sequence != earlier.sequence + 1
                    or later.epoch != earlier.epoch
                    for earlier, later in zip(
                        position_samples, position_samples[1:], strict=False
                    )
                ),
                "p95_interval_ms": (
                    sorted(position_intervals)[
                        max(math.ceil(0.95 * len(position_intervals)) - 1, 0)
                    ]
                    if position_intervals
                    else None
                ),
                "stage_latency_summary_ms": _position_stage_latency_summary(
                    position_records
                ),
            }
        if generation is not None and (
            not cast(Callable[[Any], bool], lease_is_current)(lease)
            or getattr(handle, "report_subscription_generation", None)
            != generation.generation
        ):
            result["reason"] = "report_subscription_generation_changed"
            return result
        result["reason"] = "completed"
    except Exception as err:  # noqa: BLE001
        result["reason"] = f"{type(err).__name__}: {err}"
    finally:
        if collector_task is not None and not collector_task.done():
            collector_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await collector_task
        if position_stream is not None:
            position_stream.close()
        # Always tear the subscription down, including on the error path: a
        # short-period stream left running is a standing BLE load.
        if result["subscription_attempted"]:
            try:
                await coordinator.manager.request_iot_sync_continuous_stop(
                    coordinator.device_name
                )
                result["subscription_stopped"] = True
                result["subscription_stop_ack_at_monotonic"] = time.monotonic()
            except Exception as err:  # noqa: BLE001
                result["stop_error"] = f"{type(err).__name__}: {err}"
        if exclusive_context is not None:
            await exclusive_context.__aexit__(None, None, None)

    intervals = result["intervals_ms"]
    if intervals:
        ordered = sorted(intervals)
        result["summary"] = {
            "count": len(ordered),
            "min_ms": ordered[0],
            "median_ms": ordered[len(ordered) // 2],
            "max_ms": ordered[-1],
            "mean_ms": round(sum(ordered) / len(ordered), 1),
            "observed_rate_hz": round(1000 / (sum(ordered) / len(ordered)), 3),
        }
        # A median alone is too weak: unrelated inbound traffic pulls it below
        # the requested period even when the device is clamping. Require the
        # *worst* gap to respect the request too, since a honoured period puts
        # a ceiling on the interval rather than merely a typical value.
        # Measured 2026-08-07: at every requested period from 100 to 1000 ms the
        # max gap stayed near 1.0-1.1 s, which is what a clamp looks like.
        result["aggregate_period_heuristic"] = (
            result["summary"]["median_ms"] <= no_change_period_ms * 1.5
            and result["summary"]["max_ms"] <= no_change_period_ms * 2.5
        )
    position_payloads = result["position_payloads"]
    result["honoured_requested_period"] = None
    if not isolated:
        result["period_classification_reason"] = "isolated_subscription_required"
    elif period_ms != no_change_period_ms:
        result["period_classification_reason"] = "periods_must_match"
    elif position_payloads["observed"] < 100:
        result["period_classification_reason"] = "fewer_than_100_position_payloads"
    elif position_payloads["dropped_samples"] or position_payloads["sequence_gaps"]:
        result["period_classification_reason"] = "position_evidence_gap"
    else:
        result["position_payload_cell_meets_period_criterion"] = (
            position_payloads["p95_interval_ms"] <= period_ms * 1.5
        )
        result["period_classification_reason"] = "three_randomized_repeats_required"
    return result


async def _report_stream_sequence_probe(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    periods_ms: Sequence[int],
    observation_seconds: float,
    readiness_timeout_seconds: float,
) -> dict[str, Any]:
    """Run multiple report transitions under one serialized ownership lease.

    Every cell keeps its per-sample position intervals and pipeline latencies.
    An earlier revision stripped them to bound artifact size, which inverted this
    project's own rule -- verify with per-item records, not aggregates. It left
    228 GENERIC report intervals per cell, the channel proven here to be unable
    to establish position freshness, while discarding the position intervals that
    are the actual evidence, so a reviewer could not re-derive the interval
    distribution the 3.5 s readiness bound is sized against.
    """
    result: dict[str, Any] = {
        "periods_ms": list(periods_ms),
        "observation_seconds": observation_seconds,
        "readiness_timeout_seconds": readiness_timeout_seconds,
        "isolated": True,
        "motion_commanded": False,
        "cells": [],
        "complete": False,
        "failed_cells": [],
        "reason": None,
    }
    handle = coordinator.manager.mower(coordinator.device_name)
    if handle is None:
        result["reason"] = "device_handle_unavailable"
        return result
    if getattr(coordinator, "manual_motion_owner", None) is not None:
        result["reason"] = "manual_motion_session_active"
        return result
    exclusive_factory = getattr(handle, "exclusive_report_subscription", None)
    if not callable(exclusive_factory):
        result["reason"] = "exclusive_report_subscription_unavailable"
        return result

    try:
        async with exclusive_factory("report_stream_sequence_probe") as lease:
            if lease is None:
                result["reason"] = "report_subscription_lease_unavailable"
                return result
            result["lease"] = {
                "owner": lease.owner,
                "lease_id": lease.lease_id,
                "acquired_at_monotonic": lease.acquired_at_monotonic,
                # Enqueue-time only. The library cannot confirm the device acted
                # on the quiescing STOP, so this is not evidence of quiescence --
                # per-generation position readiness below is.
                "background_stop_enqueued": lease.background_stop_enqueued,
                "background_stop_enqueued_at_monotonic": (
                    lease.background_stop_enqueued_at_monotonic
                ),
            }
            for index, period_ms in enumerate(periods_ms, start=1):
                cell = await _report_stream_probe(
                    coordinator,
                    period_ms=int(period_ms),
                    no_change_period_ms=int(period_ms),
                    duration_seconds=observation_seconds,
                    isolated=True,
                    report_lease=lease,
                    readiness_timeout_seconds=readiness_timeout_seconds,
                )
                cell["cell_index"] = index
                result["cells"].append(cell)
    except asyncio.CancelledError:
        raise
    except Exception as err:  # noqa: BLE001
        result["reason"] = f"{type(err).__name__}: {err}"
        return result

    result["failed_cells"] = [
        cell["cell_index"]
        for cell in result["cells"]
        if cell.get("reason") != "completed"
        or cell.get("subscription_started") is not True
        or cell.get("subscription_stopped") is not True
        or cell.get("position_readiness", {}).get("ready") is not True
        or cell.get("position_payloads", {}).get("dropped_samples", 0)
        or cell.get("position_payloads", {}).get("sequence_gaps", 0)
    ]
    result["complete"] = (
        len(result["cells"]) == len(periods_ms) and not result["failed_cells"]
    )
    result["reason"] = "completed" if result["complete"] else "transition_failed"
    return result


async def _settle_ble_command_queue(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Wait briefly for the BLE command queue to drain before gating on it.

    Every motion executor starts the dense report stream first, and that start
    *is* a BLE command. ``_ble_link_liveness`` then requires ``queue_depth`` at
    or below ``_BLE_QUEUE_DEPTH_LIMIT`` (zero), so an executor that starts the
    stream and immediately evaluates its gates blocks on its own command --
    deterministically, not as a race. Measured live 2026-07-30: two consecutive
    ``experimental_execute_segment`` dispatches were refused with
    ``command_queue_backlogged`` and ``queue_depth: 1`` while the link was
    healthy (connected, usable, no cooldown, last send 2-3s old), and twenty
    consecutive idle samples of the same gate reported live.

    This only ever *waits*. It never lowers the depth limit, never overrides a
    verdict, and returns the last report unchanged on timeout -- so a genuine
    backlog still fails the gate, exactly as before. The keepalive traffic that
    shares this queue is ~5s apart and clears in well under the timeout.
    """
    report = _ble_link_liveness(coordinator)
    deadline = time.monotonic() + _BLE_QUEUE_SETTLE_TIMEOUT_SECONDS
    while (
        not report["live"]
        and report["reason"] in _BLE_TRANSIENT_QUEUE_REASONS
        and time.monotonic() < deadline
    ):
        await asyncio.sleep(_BLE_QUEUE_SETTLE_POLL_SECONDS)
        report = _ble_link_liveness(coordinator)
    return report


async def _attempt_ble_recovery(  # noqa: C901
    coordinator: MammotionReportUpdateCoordinator,
    *,
    timeout_seconds: float = 90.0,
    poll_interval_seconds: float = 5.0,
) -> dict[str, Any]:
    """Try to promote BLE to the active transport without operator help.

    Automates the live-proven recovery recipe (2026-07-11/12 sessions): first
    gently re-assert the BLE preference (the switch's ON path, which also
    re-ensures the BLE client), then poll for promotion; halfway through the
    budget fall back to one full off->on toggle. Respects the BLE transport's
    connect cooldown by deferring the reassert/toggle while it is active.
    Cannot help when the mower has stopped advertising (idle sleep) or when a
    phone app holds the mower's single BLE connection slot -- those still need
    a physical/app-side wake, and the report says so.
    """
    report: dict[str, Any] = {
        "attempted": True,
        "ok": False,
        "reason": None,
        "steps": [],
        "ble_rssi": _safe_attr_path(coordinator.data, "report_data.connect.ble_rssi"),
        "timeout_seconds": timeout_seconds,
    }
    if _ble_ready_for_motion(coordinator):
        report.update(attempted=False, ok=True, reason="already_ble")
        return report
    if _ble_connect_cooldown_active(coordinator):
        report["steps"].append("ble_cooldown_active_waiting")
    else:
        try:
            await coordinator.async_set_bluetooth_enabled(True)
            report["steps"].append("reasserted_ble_preference")
        except Exception as err:  # noqa: BLE001
            report["steps"].append(f"reassert_failed: {type(err).__name__}: {err}")
    deadline = time.monotonic() + timeout_seconds
    half_budget = time.monotonic() + timeout_seconds / 2
    toggled = False
    while time.monotonic() < deadline:
        await asyncio.sleep(poll_interval_seconds)
        if _ble_ready_for_motion(coordinator):
            report.update(ok=True, reason="promoted")
            return report
        if not toggled and time.monotonic() >= half_budget:
            if _ble_connect_cooldown_active(coordinator):
                # Defer, don't skip: retry the toggle on a later pass once the
                # cooldown lapses instead of burning the one toggle mid-cooldown.
                if report["steps"][-1:] != ["ble_cooldown_active_waiting"]:
                    report["steps"].append("ble_cooldown_active_waiting")
                continue
            toggled = True
            try:
                await coordinator.async_set_bluetooth_enabled(False)
                await asyncio.sleep(3)
                await coordinator.async_set_bluetooth_enabled(True)
                report["steps"].append("ble_toggled")
            except Exception as err:  # noqa: BLE001
                report["steps"].append(f"toggle_failed: {type(err).__name__}: {err}")
    rssi = report["ble_rssi"]
    if rssi in (0, None):
        report["reason"] = "mower_not_advertising_needs_wake"
    elif "ble_cooldown_active_waiting" in report["steps"] and not toggled:
        # The connect cooldown (pymammotion default 120s) can outlast the whole
        # recovery budget; the remedy is waiting it out, not the wake/phone-app
        # checklist.
        report["reason"] = "ble_connect_cooldown_active_retry_later"
    else:
        report["reason"] = "ble_promotion_timeout_check_phone_app"
    return report


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
    ble_link = _ble_link_liveness(coordinator) if not dry_run else None
    rpm_verdict = blade_rpm_stale_register(coordinator)
    blade = _runtime_blade_diagnostics(
        before,
        rpm_stale_register=rpm_verdict.get("stale_register") is True,
    )
    return [
        {
            "name": "stop_primitive_available",
            "passed": hasattr(coordinator, "async_stop_manual_motion"),
            "detail": "Coordinator must expose async_stop_manual_motion().",
        },
        {
            "name": "ble_transport_required",
            "passed": dry_run or _ble_ready_for_motion(coordinator),
            "detail": (
                "Real closed-loop motion requires the BLE transport for "
                "responsive telemetry; cloud/Wi-Fi lag is unsafe for guarded "
                "path execution. BLE must also report itself usable -- being "
                "selected for routing does not mean a command can be delivered."
            ),
        },
        {
            "name": "ble_link_live",
            "passed": dry_run or bool(ble_link and ble_link["live"]),
            "detail": (
                "Real motion requires a conservative BLE preflight -- a live "
                "client, an open dispatch gate, an empty command queue, and a "
                "recent outbound attempt. 'Usable' only means BLE is "
                "eligible for routing; it stays True while the command queue is "
                "gated and commands (including the mandatory stop that bounds a "
                "pulse) accumulate undelivered. The send path separately waits "
                "for confirmed queue start and GATT-write completion. See "
                "docs/pymammotion-ble-slot-leak-bug.md."
            ),
            "diagnostics": ble_link,
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
            "passed": dry_run or blade["blade_safe_for_motion"],
            "detail": (
                "Real pulse requires blade state and cutter mode off. A nonzero "
                "latched RPM register is discounted only after three samples "
                "prove the position feed live and the RPM constant."
            ),
            "diagnostics": {
                "blade": blade,
                "rpm_stale_verdict": rpm_verdict,
            },
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


# Bounded ceiling for the best-effort stop delivered when a motion loop is
# cancelled mid-pulse (HA shutdown, task cancel, integration teardown).
_MOTION_CANCEL_STOP_TIMEOUT_SECONDS = 6.0


async def _deliver_stop_despite_cancellation(
    coordinator: MammotionReportUpdateCoordinator,
) -> None:
    """Best-effort motion stop that survives task cancellation.

    ``asyncio.shield`` keeps the stop running to completion even if this await
    is interrupted by a further cancellation; ``wait_for`` bounds it so a dead
    transport cannot hang teardown.
    """
    try:
        await asyncio.shield(
            asyncio.wait_for(
                _stop_manual_motion_confirmed(coordinator),
                timeout=_MOTION_CANCEL_STOP_TIMEOUT_SECONDS,
            )
        )
    except Exception:  # noqa: BLE001 - teardown keeps a final best-effort fallback
        with contextlib.suppress(Exception):
            await asyncio.shield(
                asyncio.wait_for(
                    coordinator.async_stop_manual_motion(),
                    timeout=_MOTION_CANCEL_STOP_TIMEOUT_SECONDS,
                )
            )


async def _motion_open_sleep(
    coordinator: MammotionReportUpdateCoordinator, seconds: float
) -> None:
    """Sleep out a window in which a movement command is open on the mower.

    ``CancelledError`` is a ``BaseException``, so the surrounding
    ``except Exception`` handlers never see it: a cancellation during the bare
    pulse sleep used to exit the loop without delivering the mandatory stop,
    leaving the mower moving. Deliver the stop first, then re-raise.
    """
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        await _deliver_stop_despite_cancellation(coordinator)
        raise


# --- App-parity motion cadence (2026-07-20 APK decompile) -------------------
#
# `com.agilexrobotics.command.CarRemoteControlManage2` (Mammotion 2.3.8.19)
# drives the mower from the on-screen stick like this:
#
#     public static float frequency = 0.2f;            // 200 ms
#     timer.schedule(countDownTask, 0L, 200);          // fire now, then every 200 ms
#         -> maCommandHelper.sendControl(linear, angular)
#     if (linearSpeed == 0 && angularSpeed == 0) cancelTimer();
#
# and `MACommandApiHelper.sendControl` builds exactly
# `DrvMotionCtrl(setLinearSpeed, setAngularSpeed)` -- byte-identical to
# pymammotion's `send_movement`, sent with needAck=false. So the app uses the
# same command we do; the only difference is that it RE-SENDS on a fixed timer
# for as long as motion should continue, and stops by sending zero speeds first.
#
# Our executors historically sent ONE command and slept out the pulse. That is
# the leading explanation for the tape-measured behaviour of 2026-07-15: a fixed
# ~4in step regardless of pulse duration (2s -> 0", 4s -> 4", 6s -> 4"). Kept
# OPT-IN (interval 0 == legacy single-shot) until a daylight tape A/B confirms
# it against the proven path -- see plan item B1.
_MOTION_REFRESH_INTERVAL_MS_APP = 200
_MOTION_REFRESH_INTERVAL_MS_MIN = 50
_MOTION_REFRESH_INTERVAL_MS_MAX = 1000


async def _motion_refresh_window(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    resend: Callable[[], Coroutine[Any, Any, Any]],
    duration_seconds: float,
    refresh_interval_ms: int,
    max_refresh_commands: int | None = None,
    abort_event: asyncio.Event | None = None,
) -> dict[str, Any]:
    """Hold a movement command open for ``duration_seconds``.

    With ``refresh_interval_ms <= 0`` this is exactly the legacy behaviour: sleep
    out the window while a single already-sent command is open. With a positive
    interval it mirrors the app, calling ``resend`` (which must re-issue the
    identical movement command) every interval until the window closes.

    The caller is responsible for the initial command (so a failed first send
    still aborts before any waiting) and for the explicit stop afterwards; this
    only owns the refreshes in between. A refresh is never sent when no window
    remains, so the last command is always followed by real motion time rather
    than an immediate stop.
    """
    window_started = time.monotonic()
    report: dict[str, Any] = {
        "refresh_enabled": refresh_interval_ms > 0,
        "refresh_interval_ms": max(int(refresh_interval_ms), 0),
        "refresh_commands_sent": 0,
        "refresh_command_limit": max_refresh_commands,
        "refresh_write_durations_ms": [],
        "refresh_write_completions_elapsed_ms": [],
    }
    if refresh_interval_ms <= 0:
        await _motion_open_sleep(coordinator, duration_seconds)
        report["elapsed_ms"] = round((time.monotonic() - window_started) * 1000, 3)
        return report

    interval_ms = min(
        max(int(refresh_interval_ms), _MOTION_REFRESH_INTERVAL_MS_MIN),
        _MOTION_REFRESH_INTERVAL_MS_MAX,
    )
    report["refresh_interval_ms"] = interval_ms
    interval_seconds = interval_ms / 1000
    deadline = time.monotonic() + duration_seconds
    # Bound the refresh count as well as the deadline. A wall-clock-only loop
    # spins forever if sleeps do not advance the clock (no-op sleeps in tests),
    # and this doubles as a hard ceiling on commands per pulse.
    max_refreshes = max(int(duration_seconds / interval_seconds), 0)
    if max_refresh_commands is not None:
        max_refreshes = min(max_refreshes, max(int(max_refresh_commands), 0))
    report["max_refresh_commands"] = max_refreshes
    while report["refresh_commands_sent"] < max_refreshes:
        # Stop refreshing the moment the distance guard trips. The caller's
        # mandatory stop follows immediately, exactly as it does when the
        # window runs to its deadline, so this shortens a drive and can never
        # extend one.
        if abort_event is not None and abort_event.is_set():
            report["aborted_early"] = True
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        # Fire on a FIXED cadence measured from the window start, not one
        # interval after the previous write *finished*. The app schedules on a
        # timer (`timer.schedule(task, 0L, 200)`) and does not await delivery;
        # sleeping a full interval after each await instead makes the real
        # cadence `interval + write_duration`.
        #
        # That is not a rounding error on this link. Across all 98 refresh
        # writes of the five real runs of 2026-08-09 the write latency was
        # p50 225.6 / p90 572.0 / p95 1029.2 / max 2014.0 ms, and 59% of writes
        # exceeded the 200 ms interval outright. Sleep-then-await therefore put
        # a MEDIAN of ~426 ms between commands reaching the mower against a
        # 200 ms design, and ~1229 ms at p95 -- long enough for the device-side
        # watchdog to stop the motor mid-pulse. That is the measured mechanism
        # behind the dead-time pulses (a 1303.972 ms write that starved a whole
        # 1303.7 ms pulse) and the +112%/+117% delivered-window overruns.
        #
        # Catching up is bounded and cannot burst: `max_refreshes` still caps
        # the count, the loop still awaits each write so only one is ever in
        # flight, and a write slower than the interval simply yields a zero
        # sleep rather than a queue. Behaviour is unchanged whenever writes are
        # faster than the interval, which is the case this was designed for.
        next_fire = (
            window_started + (report["refresh_commands_sent"] + 1) * interval_seconds
        )
        await _motion_open_sleep(
            coordinator, min(max(0.0, next_fire - time.monotonic()), remaining)
        )
        if time.monotonic() >= deadline:
            break
        resend_started = time.monotonic()
        try:
            await resend()
        except asyncio.CancelledError:
            # Same trap `_motion_open_sleep` exists for, one await further in:
            # CancelledError is a BaseException, so the `except Exception`
            # below never sees it and it propagates past the caller's mandatory
            # stop. A movement command is already open on the mower at this
            # point, so exiting without stopping leaves it driving until its
            # own device-side timeout. Deliver the stop first, then re-raise.
            await _deliver_stop_despite_cancellation(coordinator)
            raise
        except ManualMotionCancelledError:
            # ``stop_manual_motion`` marks the shared session cancelled before
            # queueing its zero writes.  The confirmed-dispatch guard raises
            # here on the next refresh.  This is control flow, not a transient
            # resend failure: propagate it so the exclusive owner exits
            # immediately and reports ``operator_stop`` instead of continuing
            # through feedback waits and post-command samples.
            await _deliver_stop_despite_cancellation(coordinator)
            raise
        except Exception as err:  # noqa: BLE001
            # Stop refreshing but let the caller run its mandatory stop: a
            # half-refreshed window is a shorter drive, never a runaway one.
            report["refresh_error"] = f"{type(err).__name__}: {err}"
            break
        report["refresh_write_durations_ms"].append(
            round((time.monotonic() - resend_started) * 1000, 3)
        )
        report["refresh_write_completions_elapsed_ms"].append(
            round((time.monotonic() - window_started) * 1000, 3)
        )
        report["refresh_commands_sent"] += 1
    report["elapsed_ms"] = round((time.monotonic() - window_started) * 1000, 3)
    return report


# --- App speed scale (2026-07-20 APK decompile) -----------------------------
#
# Mirrors `RockerControlUtil.transfrom3` + `getPercent` as shipped in the app
# (and re-implemented in `pymammotion.utility.movement.transform_both_speeds`).
# Implemented locally so the numbers are deterministic and testable regardless
# of the pinned pymammotion version; `test_app_speed_scale_matches_pymammotion`
# guards the two against drifting apart.
#
# For a pure axis the transform collapses to sin/cos of 90/270 (linear) or
# 0/180 (angular), i.e. +/-1, leaving magnitude scaling only:
#     linear  = int(percent) * 10
#     angular = int(int(percent) * 4.5)
# with a 15% stick deadband applied first. Full deflection therefore yields
# 85 percent -> 850 linear / 382 angular, which are the real ceilings.
_APP_SPEED_DEADBAND_PERCENT = 15.0
_APP_LINEAR_SPEED_SCALE = 10
_APP_ANGULAR_SPEED_SCALE = 4.5
_APP_MAX_LINEAR_SPEED = 850
_APP_MAX_ANGULAR_SPEED = 382


def _app_speed_percent(deflection_percent: float) -> float:
    """Apply the app's 15% stick deadband to a 0-100 deflection."""
    if deflection_percent <= _APP_SPEED_DEADBAND_PERCENT:
        return 0.0
    return deflection_percent - _APP_SPEED_DEADBAND_PERCENT


def _app_scale_speeds(
    linear_fraction: float = 0.0, angular_fraction: float = 0.0
) -> tuple[int, int]:
    """Convert app-scale stick fractions (-1.0..1.0) to raw movement speeds."""
    linear_component = int(_app_speed_percent(abs(linear_fraction) * 100))
    angular_component = int(_app_speed_percent(abs(angular_fraction) * 100))
    linear_speed = linear_component * _APP_LINEAR_SPEED_SCALE
    angular_speed = int(angular_component * _APP_ANGULAR_SPEED_SCALE)
    if linear_fraction < 0:
        linear_speed = -linear_speed
    if angular_fraction < 0:
        angular_speed = -angular_speed
    return linear_speed, angular_speed


def _app_speed_scale_report(linear_speed: Any, angular_speed: Any) -> dict[str, Any]:
    """Describe raw speeds against the app's own scale (read-only diagnostic).

    Flags values the on-screen control can never produce: our long-standing
    angular default of 500 is above the app's 382 ceiling and may simply be
    clamped by firmware, which would make the "angular is weak" calibration an
    artefact rather than a property of the mower.
    """
    try:
        linear = int(linear_speed)
        angular = int(angular_speed)
    except TypeError, ValueError:
        return {"available": False}
    return {
        "available": True,
        "linear_speed": linear,
        "angular_speed": angular,
        "app_max_linear_speed": _APP_MAX_LINEAR_SPEED,
        "app_max_angular_speed": _APP_MAX_ANGULAR_SPEED,
        "linear_fraction_of_app_max": (
            round(abs(linear) / _APP_MAX_LINEAR_SPEED, 3)
            if _APP_MAX_LINEAR_SPEED
            else None
        ),
        "angular_fraction_of_app_max": (
            round(abs(angular) / _APP_MAX_ANGULAR_SPEED, 3)
            if _APP_MAX_ANGULAR_SPEED
            else None
        ),
        "linear_above_app_max": abs(linear) > _APP_MAX_LINEAR_SPEED,
        "angular_above_app_max": abs(angular) > _APP_MAX_ANGULAR_SPEED,
    }


# The manual-motion claim lives on the coordinator itself
# (``MammotionBaseUpdateCoordinator.manual_motion_owner``) rather than in a
# module-level registry here, because the coordinator also has to read it: it
# must not start an exclusive map-fetch saga while a guarded motion run is in
# flight (the saga blocks the mower's command queue and stalls the run's
# pulses).  ``services`` imports ``coordinator``, so the flag cannot live here
# and still be visible there.
#
# HA service calls can overlap; two motion loops interleaving movement and stop
# commands would defeat every bounded-pulse guarantee, so a second start is
# strictly rejected (never queued -- a queued motion run would fire unattended
# after the owner finishes).


def _manual_motion_busy_result(service: str, owner: str) -> dict[str, Any]:
    """Structured rejection returned when another motion run owns the mower."""
    return {
        "service": service,
        "mode": "rejected_busy",
        "valid": False,
        "would_send": False,
        "blockers": ["manual_motion_in_progress"],
        "stop_reason": "manual_motion_in_progress",
        "busy_owner": owner,
    }


def _manual_motion_rejected_result(
    service: str,
    blockers: Sequence[str],
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a consistent fail-closed authorization rejection."""
    return {
        "service": service,
        "mode": "rejected_safety_gate",
        "valid": False,
        "would_send": False,
        "blockers": list(dict.fromkeys(blockers)),
        "stop_reason": blockers[0] if blockers else "manual_motion_not_authorized",
        "experimental_motion": dict(diagnostics or {}),
    }


def _manual_motion_authorization(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    ha_state: str | None,
    call_data: Mapping[str, Any],
    rpm_stale_register: bool = False,
) -> dict[str, Any]:
    """Evaluate the common authorization boundary for every nonzero run.

    ``rpm_stale_register`` comes from :func:`_reconfirm_blade_rpm_stale`, run by
    the caller before claiming the mower. It defaults False, so an authorization
    evaluated without that evidence keeps the blade guard closed.
    """
    try:
        route = _export_active_route(coordinator)
    except Exception:  # noqa: BLE001 - unreadable safety state must fail closed
        route = None
    telemetry = _custom_path_telemetry_snapshot(coordinator)
    safety = _runtime_motion_safety_summary(
        telemetry,
        ha_state=ha_state,
        active_route=route,
        rpm_stale_register=rpm_stale_register,
        rtk_report_age_seconds=_rtk_report_age_seconds(coordinator),
        allow_degraded_rtk=call_data.get("allow_degraded_rtk") is True,
    )
    liveness = _ble_link_liveness(coordinator)
    status = experimental_motion_status(
        coordinator,
        ble_liveness=liveness,
        safety=safety,
    )
    blockers = list(status["blockers"])

    if call_data.get("use_wifi") is True or call_data.get("prefer_ble") is False:
        blockers.append("manual_motion_requires_ble")
    if call_data.get("confirm_blades_off") is not True:
        blockers.append("operator_confirmation_blades_off_required")
    if call_data.get("confirm_clear_area") is not True:
        blockers.append("operator_confirmation_clear_area_required")

    capabilities = capability_snapshot(coordinator)
    if capabilities["capabilities"]["manual_motion"] != "yes":
        blockers.append("manual_motion_capability_unknown")
    if telemetry.get("online") is not True:
        blockers.append("mower_online_state_not_fresh")
    if telemetry.get("work_mode_label") not in {"MODE_READY", "MODE_PAUSE"}:
        blockers.append("mower_not_ready_or_paused")
    if telemetry.get("charge_state_label") != "not_charging":
        blockers.append("mower_charging_state_not_safe")

    status["blockers"] = list(dict.fromkeys(blockers))
    status["real_motion_allowed"] = not status["blockers"]
    return status


#: How long a gate snapshot is reused. Six diagnostic entities read the verdict
#: on the same coordinator tick, and ``_export_active_route`` rewrites every
#: GeoJSON coordinate through ``apply_geojson_offset``, so without this the map
#: would be re-projected once per entity per update.
_GATE_SNAPSHOT_TTL_SECONDS = 5.0

_GATE_SNAPSHOT_ATTR = "_mammotion_gate_snapshot"
_GATE_SNAPSHOT_STAMP_ATTR = "_mammotion_gate_snapshot_monotonic"


def _unknown_gate_snapshot(reason: str) -> dict[str, Any]:
    """Return a fail-closed snapshot when the gate state cannot be read."""
    return {
        "available": False,
        "reason": reason,
        "real_motion_ready": None,
        "blockers": [],
        "backend_verified": None,
        "backend_capabilities": {},
        "ble_link_live": None,
        "ble_link_reason": None,
        "blade_safe_for_motion": None,
        "blade_blockers": [],
        "position_valid_for_motion": None,
        "zone_hash": None,
        "pos_type_label": None,
    }


def motion_gate_snapshot(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Return the standing motion-gate verdict for diagnostic entities.

    This is a *display* of the standing gate state, not an authorization path.
    It deliberately omits the per-call operator confirmations
    (``confirm_blades_off``/``confirm_clear_area``) and the busy/saga checks,
    which only a service call can evaluate -- so ``real_motion_ready`` means
    "nothing standing in the way right now", never "the next dispatch will be
    accepted". :func:`_manual_motion_authorization` remains the only gate that
    authorizes a real write.

    Cached for ``_GATE_SNAPSHOT_TTL_SECONDS`` so the six entities that read it
    cost one computation per coordinator tick rather than six. Any failure
    degrades to an unavailable snapshot instead of raising into the entity.
    """
    now = time.monotonic()
    stamp = getattr(coordinator, _GATE_SNAPSHOT_STAMP_ATTR, None)
    cached = getattr(coordinator, _GATE_SNAPSHOT_ATTR, None)
    if (
        cached is not None
        and stamp is not None
        and now - float(stamp) < _GATE_SNAPSHOT_TTL_SECONDS
    ):
        return cached

    try:
        try:
            route = _export_active_route(coordinator)
        except Exception:  # noqa: BLE001 - an unreadable route must not blind the rest
            route = None
        telemetry = _custom_path_telemetry_snapshot(coordinator)
        # One observation per tick feeds the latched-RPM history the dispatch
        # path reads synchronously, so the evidence spans real time without
        # anything polling on demand.
        _record_blade_rpm_sample(coordinator, telemetry)
        safety = _runtime_motion_safety_summary(
            telemetry,
            ha_state=None,
            active_route=route,
            rtk_report_age_seconds=_rtk_report_age_seconds(coordinator),
        )
        liveness = _ble_link_liveness(coordinator)
        status = experimental_motion_status(
            coordinator, ble_liveness=liveness, safety=safety
        )
        blade = _runtime_blade_diagnostics(telemetry)
        position = telemetry.get("position") or {}
        snapshot = {
            "available": True,
            "reason": None,
            "real_motion_ready": status["real_motion_allowed"] is True,
            "blockers": list(status.get("blockers") or []),
            "backend_verified": status.get("backend_verified"),
            "backend_capabilities": status.get("backend_capabilities") or {},
            "ble_link_live": liveness.get("live") is True,
            "ble_link_reason": liveness.get("reason"),
            "blade_safe_for_motion": blade.get("blade_safe_for_motion"),
            "blade_blockers": list(blade.get("safety_blockers") or []),
            "blade_rpm_looks_latched": blade.get("blade_rpm_looks_latched"),
            "position_valid_for_motion": position.get("valid_for_motion"),
            "zone_hash": position.get("zone_hash"),
            "pos_type_label": position.get("pos_type_label"),
        }
    except Exception as err:  # noqa: BLE001 - diagnostics must never break setup
        LOGGER.debug("motion_gate_snapshot failed", exc_info=True)
        snapshot = _unknown_gate_snapshot(type(err).__name__)

    setattr(coordinator, _GATE_SNAPSHOT_ATTR, snapshot)
    setattr(coordinator, _GATE_SNAPSHOT_STAMP_ATTR, now)
    return snapshot


def _exclusive_saga_active(coordinator: MammotionReportUpdateCoordinator) -> bool:
    """Return True only when an exclusive saga demonstrably holds the queue.

    ``MapFetchSaga`` runs at ``Priority.EXCLUSIVE``; motion goes out at
    ``Priority.NORMAL`` with ``skip_if_saga_active=False``, so it *blocks* on
    the exclusive slot rather than being skipped, and ``_COMMAND_TTL`` (120 s)
    silently drops whatever is still undispatched.

    That is why a queued motion command is not merely late. The executor's
    guarantee is "send a bounded movement, sleep the pulse locally, then send
    an explicit stop" -- but the sleep is local timing and cannot see that the
    command is still queued. The pulse can therefore elapse before the mower
    moves at all, and the movement and its stop can be separated (or either one
    dropped at the TTL), which breaks the bounded-pulse guarantee the whole
    safety model rests on.

    ``coordinator.manual_motion_owner`` already stops a saga starting while
    motion holds the mower. This is the other direction, which was previously
    unguarded: the only ``is_saga_active`` consumer was the ``map_sync_status``
    sensor label.

    Positively-True-only: any unreadable piece degrades to False (motion
    allowed), so pymammotion API drift can never block all motion.
    """
    try:
        handle = coordinator.manager.mower(coordinator.device_name)
        if handle is None:
            return False
        return bool(handle.queue.is_saga_active)
    except Exception:  # noqa: BLE001 - never let a probe failure block motion
        return False


def _saga_active_for_diagnostics(handle: Any) -> bool | None:
    """Return saga state as a tri-state, where unreadable is ``None``.

    Deliberately NOT ``_exclusive_saga_active``: that one degrades to ``False``
    so pymammotion API drift can never block motion, which is right for a gate
    and wrong for a diagnostic. A record that says "no saga" when it actually
    means "could not tell" is exactly the mis-attribution this field exists to
    prevent, so unreadable stays unreadable here.
    """
    try:
        if handle is None:
            return None
        return bool(handle.queue.is_saga_active)
    except Exception:  # noqa: BLE001 - a diagnostic must never raise
        return None


def _is_zero_motion_stop_nudge(
    command: str, linear_speed: int, angular_speed: int
) -> bool:
    """Return True for the zero-motion stop nudge (the card's Abort path).

    A ``send_movement`` with both speeds zero is a stop, not motion -- it must
    be allowed to preempt a running motion loop, never be rejected as busy.
    """
    return (
        command == "send_movement"
        and int(linear_speed) == 0
        and int(angular_speed) == 0
    )


def _wrap_exclusive_manual_motion(  # noqa: C901
    hass: HomeAssistant,
    service: str,
    handler: Callable[[ServiceCall], Coroutine[Any, Any, dict[str, Any]]],
    *,
    allow_stop_nudge: bool = False,
    always_real: bool = False,
) -> Callable[[ServiceCall], Coroutine[Any, Any, dict[str, Any]]]:
    """Serialize a motion service's real runs per mower at registration time.

    Dry runs pass straight through (they read telemetry, never move). A real
    run atomically claims the mower on the event loop (no await between check
    and set) and releases it on every exit path, including cancellation.
    ``allow_stop_nudge`` exempts the zero-motion stop nudge (the card's Abort
    path) so a stop can always preempt a running loop.
    """

    async def wrapped(call: ServiceCall) -> dict[str, Any]:  # noqa: C901
        real = always_real or call.data.get("dry_run", True) is False
        if (
            real
            and allow_stop_nudge
            and _is_zero_motion_stop_nudge(
                str(call.data.get("command", "")),
                int(call.data.get("linear_speed", 0)),
                int(call.data.get("angular_speed", 0)),
            )
        ):
            real = False
        if not real:
            return await handler(call)
        # Prove the loaded backend carries the audited BLE fixes before the
        # claim block below, which must stay await-free. Cached after the first
        # call, so this costs nothing on later runs.
        await async_probe_backend_capabilities()
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            # The wrapped handler logs the unknown entity and returns {}.
            return await handler(call)
        coordinator = mower.reporting_coordinator
        # Synchronous read of observations the gate snapshot already collected --
        # no sleeping, no polling, and safe inside the await-free claim window
        # below. Not-stale unless the history positively proves the register is
        # latched, so the blade guard stays closed by default.
        rpm_verdict = blade_rpm_stale_register(coordinator)
        rpm_stale_register = rpm_verdict.get("stale_register") is True
        if rpm_stale_register:
            LOGGER.info(
                "cutter RPM %s judged a latched register (feed proven live across "
                "%s observations); blade guard not blocking on it",
                rpm_verdict.get("rpm_values"),
                rpm_verdict.get("samples"),
            )
        # Claim atomically on the event loop: no await between the read and the
        # write, so two overlapping calls cannot both see it free. The saga
        # probe is a synchronous read for the same reason -- check and claim
        # stay one uninterrupted block, so a saga cannot slip in between them.
        owner = getattr(coordinator, "manual_motion_owner", None)
        if owner is not None:
            return _manual_motion_busy_result(service, owner)
        if _exclusive_saga_active(coordinator):
            # Refuse with a reason instead of queueing behind the exclusive
            # slot. The earlier design deliberately had no gate here, on the
            # grounds that "refusing a command the operator just issued is
            # worse than the wait" -- but that assumed sagas were rare and
            # operator-triggered, so the operator would know why. Since the
            # per-tick map-sync path was made reachable (2026-07-25) they also
            # fire automatically, and the "wait" was never benign anyway: it
            # can separate a pulse from its stop (see _exclusive_saga_active).
            # A *named* refusal answers the original objection -- the operator
            # is told it is a map sync and can retry in seconds.
            return _manual_motion_busy_result(service, "map_sync_saga")
        states = getattr(hass, "states", None)
        state = states.get(call.data[ATTR_ENTITY_ID]) if states is not None else None
        authorization = _manual_motion_authorization(
            coordinator,
            ha_state=state.state if state is not None else None,
            call_data=call.data,
            rpm_stale_register=rpm_stale_register,
        )
        if authorization["real_motion_allowed"] is not True:
            return _manual_motion_rejected_result(
                service,
                authorization["blockers"],
                diagnostics=authorization,
            )
        session = ManualMotionSession(owner=service)
        coordinator.manual_motion_owner = service
        coordinator.manual_motion_session = session
        session.phase = "running"
        try:
            result = await handler(call)
        except ManualMotionCancelledError:
            session.phase = "aborted"
            session.cancelled = True
            session.cancel_reason = session.cancel_reason or "operator_stop"
            return {
                "service": service,
                "mode": "aborted",
                "valid": False,
                "would_send": False,
                "session": session.as_dict(),
                "stop_reason": "operator_stop",
            }
        except asyncio.CancelledError:
            session.phase = "cancelled"
            session.cancelled = True
            session.cancel_reason = session.cancel_reason or "task_cancelled"
            raise
        except Exception as err:
            session.phase = "failed"
            session.error = f"{type(err).__name__}: {err}"
            raise
        else:
            session.phase = "completed"
            return result
        finally:
            session.owner_done.set()
            coordinator.last_manual_motion_session = session
            if coordinator.manual_motion_session is session:
                coordinator.manual_motion_session = None
            coordinator.manual_motion_owner = None

    return wrapped


async def _manual_velocity_pulse_test(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    action: str = "forward",
    speed: float = 0.55,
    duration_ms: int = 3500,
    stop_mode: str = "immediate",
    stop_delay_ms: int = 0,
    post_command_sample_delays: list[float] | tuple[float, ...] | None = None,
    use_wifi: bool = DEFAULT_EXPERIMENTAL_SEGMENT_USE_WIFI,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    followup_samples: int = 4,
    followup_interval_seconds: float = 0.5,
    motion_refresh_interval_ms: int = 0,
) -> dict[str, Any]:
    """Run or simulate one tiny manual-velocity pulse with telemetry sampling."""
    if post_command_sample_delays is None:
        post_command_sample_delays = tuple(
            followup_interval_seconds * (index + 1) for index in range(followup_samples)
        )
    if hasattr(coordinator, "async_start_report_stream"):
        stream_duration_ms = int(
            (max(post_command_sample_delays, default=0.0) + 10) * 1000
        )
        await coordinator.async_start_report_stream(
            duration_ms=max(10_000, stream_duration_ms)
        )
        # start_report_stream degrades to a single snapshot outside ACTIVE mode,
        # which leaves a manually driven mower reporting one frozen position for
        # the whole run. Ask for the continuous subscription explicitly.
        if hasattr(coordinator, "async_start_continuous_reports"):
            await coordinator.async_start_continuous_reports(
                duration_ms=max(10_000, stream_duration_ms)
            )
        # The calls above enqueue BLE commands; let them clear before the
        # ble_link_live gate below demands an empty queue.
        await _settle_ble_command_queue(coordinator)

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
        "motion_refresh_interval_ms": motion_refresh_interval_ms,
        # `speed` here is the app-scale 0.0-1.0 stick fraction (the coordinator's
        # directional helpers run it through the same transform the app uses), so
        # report the raw speeds it resolves to. Turn actions drive the angular
        # axis, forward/backward the linear one.
        "app_speed_scale": _app_speed_scale_report(
            *(
                _app_scale_speeds(speed, 0.0)
                if action in {"forward", "backward"}
                else _app_scale_speeds(0.0, speed)
            )
        ),
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
    # Bare bounded pulse -- this is the A/B harness for the app-parity cadence
    # question (plan item B1): same action and duration, run once with
    # motion_refresh_interval_ms=0 and once with 200, tape-measure both.
    result["motion_refresh"] = await _motion_refresh_window(
        coordinator,
        resend=functools.partial(
            _manual_velocity_command_attempt,
            coordinator,
            action=action,
            speed=speed,
            use_wifi=use_wifi,
        ),
        duration_seconds=duration_ms / 1000,
        refresh_interval_ms=motion_refresh_interval_ms,
    )
    after_command = _custom_path_telemetry_snapshot(coordinator)
    result["samples"].append(
        {"label": "after_command_window", "telemetry": after_command}
    )
    if stop_mode == "delayed" and stop_delay_ms > 0:
        await _motion_open_sleep(coordinator, stop_delay_ms / 1000)
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


async def _send_ble_motion_command_confirmed(  # noqa: C901
    coordinator: MammotionReportUpdateCoordinator,
    command: str,
    *,
    command_kwargs: Mapping[str, Any],
    emergency_stop: bool = False,
) -> None:
    """Send one motion command over BLE and wait for the GATT write to finish.

    ``MammotionClient.send_command_with_args`` only waits for queue insertion.
    Starting a pulse timer after that call is unsafe: the command may still be
    behind a reconnect gate or an in-flight write, and its later stop can then
    be replayed on a different timeline.

    This helper keeps normal queue ordering but owns the queued work, allowing
    it to report actual completion. It also sends through the already-selected
    BLE transport directly, so a failed BLE write cannot silently fall back to
    cloud MQTT. If the item cannot start promptly it is disarmed before raising;
    a later queue recovery will consume it as a no-op instead of moving the
    mower late.
    """
    is_stop = command == "send_movement" and all(
        int(command_kwargs.get(key, 0)) == 0
        for key in ("linear_speed", "angular_speed")
    )
    assert_session_can_dispatch(coordinator, is_stop=is_stop)

    handle = coordinator.manager.mower(coordinator.device_name)
    if handle is None:
        raise RuntimeError("device handle unavailable for confirmed BLE motion")

    liveness = _ble_link_liveness(coordinator)
    if not emergency_stop and not liveness["live"]:
        raise RuntimeError(f"BLE link is not ready for motion: {liveness['reason']}")
    if emergency_stop and (
        liveness.get("is_connected") is not True
        or liveness.get("is_usable") is not True
    ):
        raise RuntimeError(
            "BLE link cannot deliver emergency stop: "
            f"{liveness.get('reason') or 'not_connected'}"
        )

    transport = handle.get_transport(TransportType.BLE)
    if transport is None:
        raise RuntimeError("BLE transport unavailable for confirmed motion")

    command_bytes: bytes = getattr(handle.commands, command)(**dict(command_kwargs))
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    completed: asyncio.Future[None] = loop.create_future()
    armed = True

    async def _dispatch() -> None:
        nonlocal armed
        if not armed:
            return
        started.set()
        dispatch_liveness = _ble_link_liveness(coordinator)
        dispatch_ready = (
            dispatch_liveness.get("is_connected") is True
            and dispatch_liveness.get("is_usable") is True
            if emergency_stop
            else dispatch_liveness["live"]
        )
        if not dispatch_ready:
            error = RuntimeError(
                "BLE link stopped being ready before motion dispatch: "
                f"{dispatch_liveness['reason']}"
            )
            if not completed.done():
                completed.set_exception(error)
            raise error
        try:
            await asyncio.wait_for(
                handle._send_marked(transport, command_bytes),  # noqa: SLF001
                timeout=_BLE_MOTION_WRITE_TIMEOUT_SECONDS,
            )
        except BaseException as err:
            if not completed.done():
                completed.set_exception(err)
            raise
        else:
            if not completed.done():
                completed.set_result(None)

    try:
        await handle.queue.enqueue(
            _dispatch,
            priority=Priority.EMERGENCY if emergency_stop else Priority.NORMAL,
        )
    except BaseException:
        # The real DeviceCommandQueue only inserts here, but eager test/dummy
        # queues may execute the work inline. Consume the mirrored future
        # exception before propagating so it cannot become an un-retrieved
        # event-loop warning.
        if completed.done() and not completed.cancelled():
            completed.exception()
        raise
    try:
        await asyncio.wait_for(
            started.wait(),
            timeout=(
                _BLE_MOTION_WRITE_TIMEOUT_SECONDS + 1.0
                if emergency_stop
                else _BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS
            ),
        )
    except TimeoutError:
        armed = False
        if not completed.done():
            completed.cancel()
        raise RuntimeError(
            "BLE motion command did not start before the queue deadline; "
            "the queued item was disarmed"
        ) from None
    except asyncio.CancelledError:
        armed = False
        if not completed.done():
            completed.cancel()
        raise

    try:
        await asyncio.shield(completed)
    except asyncio.CancelledError:
        # If cancellation lands after dispatch started, the queue task owns the
        # write and keeps running independently of this caller. Wait briefly for
        # it to settle, then enqueue a confirmed zero-velocity command before
        # propagating cancellation. Without this, the movement could arrive
        # after its caller has exited and no outer finally block would know it
        # needs to stop.
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(
                asyncio.shield(completed),
                timeout=_BLE_MOTION_WRITE_TIMEOUT_SECONDS + 0.5,
            )
        if not is_stop:
            with contextlib.suppress(BaseException):
                await _stop_manual_motion_confirmed(coordinator)
        raise
    else:
        record_completed_dispatch(
            coordinator,
            command=command,
            is_stop=is_stop,
        )


async def _send_manager_command_with_args(
    coordinator: MammotionReportUpdateCoordinator,
    command: str,
    *,
    prefer_ble: bool,
    command_kwargs: Mapping[str, Any],
) -> None:
    """Send a manager command, confirming BLE dispatch for motion commands."""
    if prefer_ble and command in RAW_PYMAMMOTION_MOTION_COMMANDS:
        await _send_ble_motion_command_confirmed(
            coordinator,
            command,
            command_kwargs=command_kwargs,
        )
        return
    await cast(Any, coordinator.manager.send_command_with_args)(
        coordinator.device_name,
        command,
        prefer_ble=prefer_ble,
        **dict(command_kwargs),
    )


def _in_window_ble_snapshot(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Read the four BLE dispatch facts that separate the causes of a stall.

    🔑 **Why not just call `_ble_link_liveness`.** That helper answers "is it
    safe to START dispatching", folding seven checks into one boolean plus a
    first-failing `reason`. Inside a window we do not want a verdict, we want
    the raw fields at 100 ms so a 2-second event is visible at all -- the
    `ble_link_live` ENTITY is a coordinator-tick sensor and is far too slow to
    resolve one. On 2026-08-29 it flapped three times in 67 s around a run and
    could neither confirm nor exclude a glitch inside the 2.031 s window itself.

    What each field discriminates, when position payloads stop arriving:
      * `is_connected` False        -> the GATT link actually dropped
      * `queue_dispatch_paused`     -> the command queue is gated
      * `queue_depth` climbing      -> outbound backlog, the documented proxy
                                       slot-leak signature where `active_transport`
                                       still reads `ble` and `is_usable` is True
                                       (`docs/pymammotion-ble-slot-leak-bug.md`)
      * `saga_active`               -> an exclusive saga is holding the queue

    ⚠️ **All four are OUTBOUND-side facts.** A stalled inbound position stream
    with all four healthy would mean the fault is not in our dispatch path at
    all -- which is itself the useful answer, and the reason to record them
    rather than assume.

    Every read is an in-memory attribute access. No I/O, matching
    `_in_window_telemetry_sample`'s contract.
    """
    snapshot: dict[str, Any] = {
        "is_connected": None,
        "queue_depth": None,
        "queue_dispatch_paused": None,
        "saga_active": None,
    }
    try:
        handle = coordinator.manager.mower(coordinator.device_name)
    except Exception:  # noqa: BLE001
        return snapshot
    get_transport = getattr(handle, "get_transport", None)
    if callable(get_transport):
        try:
            transport = get_transport(TransportType.BLE)
        except Exception:  # noqa: BLE001
            transport = None
        if transport is not None:
            with contextlib.suppress(Exception):
                snapshot["is_connected"] = bool(transport.is_connected)
    queue = getattr(handle, "queue", None)
    if queue is not None:
        with contextlib.suppress(Exception):
            snapshot["saga_active"] = bool(queue.is_saga_active)
        # Private in pinned pymammotion; there is no public equivalent, and
        # absence must read as None rather than as healthy.
        gate = getattr(queue, "_transport_gate", None)
        if gate is not None:
            with contextlib.suppress(Exception):
                snapshot["queue_dispatch_paused"] = not gate.is_set()
        pending = getattr(queue, "_queue", None)
        if pending is not None:
            with contextlib.suppress(Exception):
                snapshot["queue_depth"] = int(pending.qsize())
    return snapshot


def _in_window_telemetry_sample(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    index: int,
    window_started: float,
    command: str,
    command_args: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture one compact sample without causing coordinator or BLE I/O."""
    telemetry = _custom_path_telemetry_snapshot(coordinator)
    position = telemetry.get("position", {}) or {}
    try:
        handle = coordinator.manager.mower(coordinator.device_name)
    except Exception:  # noqa: BLE001
        handle = None
    return {
        "index": index,
        "elapsed_ms": round((time.monotonic() - window_started) * 1000, 3),
        "captured_at": _utc_timestamp(),
        # DeviceHandle stamps this monotonic value for every received LubaMsg.
        # Unlike x/y, it proves a fresh report even if position is unchanged.
        # ⚠️ **It proves a FRAME arrived, NOT a position payload** -- every
        # LubaMsg bumps it, and only some carry `sys.toapp_report_data`. On
        # 2026-08-28 that distinction was the whole question and this field
        # could not answer it: it advanced three times across a 2.088 s window
        # in which the mower travelled 0.4375 m and x/y never changed
        # (`docs/evidence-step-response-probe-aborted-20260828.json`).
        "last_report_at_monotonic": (
            _safe_attr_path(handle, "last_report_at") if handle is not None else None
        ),
        # 🔑 **THE DISCRIMINATOR `last_report_at` CANNOT BE.** pymammotion bumps
        # `_position_sequence` inside `_publish_position_sample`, which the
        # handle calls ONLY when the decoded frame actually carried a position
        # payload (`if position_source is not None and not self._stopping:`).
        # So across a window where x/y never change:
        #   sequence ADVANCING -> payloads arrived carrying STALE coordinates,
        #                         i.e. observer lag;
        #   sequence FROZEN    -> no position payloads arrived at all,
        #                         i.e. a feed stall.
        # Those are different faults with different owners, and nothing else
        # recorded in this sample separates them. `position_epoch` comes with it
        # because a BLE re-establishment bumps the epoch, and both known blind-
        # travel events followed a reconnect -- n = 2, a pattern to chase and
        # NOT a mechanism.
        "position_sequence": (
            _safe_attr_path(handle, "latest_position_sample.sequence")
            if handle is not None
            else None
        ),
        "position_epoch": (
            _safe_attr_path(handle, "position_epoch") if handle is not None else None
        ),
        "position": {
            "source": position.get("source"),
            "x": position.get("x"),
            "y": position.get("y"),
            "toward": position.get("toward"),
            "pos_type": position.get("pos_type"),
            "zone_hash": position.get("zone_hash"),
        },
        # Outbound BLE dispatch facts, recorded at the same 100 ms cadence as
        # position so a 2-second stall is attributable rather than inferred.
        "ble": _in_window_ble_snapshot(coordinator),
        "vio": {
            "heading": _safe_attr_path(
                coordinator.data, "report_data.vision_info.heading"
            ),
            "state": _safe_attr_path(
                coordinator.data, "report_data.vision_info.vio_state"
            ),
        },
        "active_command": {"command": command, "kwargs": dict(command_args)},
    }


def _apply_travel_guard(  # noqa: C901
    sample: dict[str, Any],
    *,
    position_sample: Any | None,
    guard_state: dict[str, Any],
    max_travel_m: float,
) -> bool:
    """Apply a cumulative-distance guard using only position evidence."""

    def _trip(reason: str) -> bool:
        sample["travel_guard_tripped"] = True
        sample["travel_guard_reason"] = reason
        return True

    stream = guard_state["stream"]
    if stream.dropped_samples != guard_state["dropped_samples"]:
        # ⚠️ Distinct from `position_sequence_gap` on purpose. Both used to
        # report the same string, so a live trip could not be told apart
        # without re-deriving it by hand -- and on 2026-09-05 that cost a
        # session two dispatches guessing which check had fired.
        return _trip("position_samples_dropped")
    if position_sample is None:
        stale_ms = (time.monotonic() - guard_state["last_receipt_at"]) * 1000.0
        if stale_ms >= _PROBE_FEED_STALE_ABORT_MS:
            sample["feed_stale_ms"] = round(stale_ms, 1)
            return _trip("feed_stale")
        return False
    if position_sample.epoch != guard_state["epoch"]:
        return _trip("position_epoch_changed")
    # 🐛 **The FIRST sample seeds the baseline; it cannot be a gap.**
    #
    # The baseline used to come from `handle.latest_position_sample`, read
    # AFTER the stream was opened -- so any payload landing in between (or
    # already queued on a caller-supplied stream) made the first drained sample
    # fail `sequence == baseline + 1`, and the guard tripped instantly at
    # `travel_at_trip_m: 0.0`. Reproduced deterministically on hardware
    # 2026-09-05, twice: the window died at 344 ms and 273 ms having sent 1 of
    # 11 refresh writes, so a `max_travel_m` run could not travel at all.
    #
    # 🔑 The inconsistency was internal: `last_position` was ALREADY seeded from
    # the first sample rather than checked against a prior value. Sequence now
    # does the same thing, which is the only reading under which the two agree.
    #
    # This weakens nothing. Contiguity is still enforced across every later
    # sample, `dropped_samples` still catches queue overruns during the window,
    # and the epoch check is untouched. What is given up is contiguity with
    # payloads that arrived BEFORE the guard began -- which carry no travel the
    # guard is accountable for, since `last_position` ignores them too.
    if guard_state["sequence"] is None:
        guard_state["sequence"] = position_sample.sequence
    elif position_sample.sequence != guard_state["sequence"] + 1:
        return _trip("position_sequence_gap")
    guard_state["sequence"] = position_sample.sequence
    guard_state["last_receipt_at"] = position_sample.received_at_monotonic
    age_ms = (time.monotonic() - position_sample.received_at_monotonic) * 1000.0
    if age_ms >= _PROBE_FEED_STALE_ABORT_MS:
        sample["position_age_ms"] = round(age_ms, 1)
        return _trip("feed_stale")
    if not position_sample.valid_for_motion:
        return _trip("position_unavailable")

    current = (float(position_sample.x), float(position_sample.y))
    previous = guard_state["last_position"]
    if previous is not None:
        guard_state["cumulative_distance_m"] += math.hypot(
            current[0] - previous[0], current[1] - previous[1]
        )
    guard_state["last_position"] = current
    sample["position"] = {
        "source": position_sample.source,
        "x": current[0],
        "y": current[1],
        "toward": position_sample.toward,
        "pos_type": position_sample.pos_type,
        "zone_hash": position_sample.zone_hash,
    }
    sample["position_sequence"] = position_sample.sequence
    sample["position_epoch"] = position_sample.epoch
    sample["cumulative_travel_m"] = round(guard_state["cumulative_distance_m"], 4)
    if guard_state["cumulative_distance_m"] >= max_travel_m:
        return _trip("max_travel_reached")
    return False


async def _capture_in_window_telemetry(  # noqa: C901
    coordinator: MammotionReportUpdateCoordinator,
    *,
    sample_interval_ms: int,
    duration_ms: int,
    window_started: float,
    stop_event: asyncio.Event,
    command: str,
    command_args: Mapping[str, Any],
    max_travel_m: float = 0.0,
    travel_abort: asyncio.Event | None = None,
    position_stream: Any | None = None,
) -> list[dict[str, Any]]:
    """Poll cached telemetry concurrently for one strictly bounded window.

    With ``max_travel_m`` positive this also acts as the window's **distance**
    guard: as soon as cumulative consecutive position samples reach the bound,
    ``travel_abort`` is set and the refresh loop stops
    refreshing, which brings the caller's mandatory stop forward.

    Position receipt age and queue continuity are checked directly. The
    conservative historic stop/guard overshoot remains part of corridor sizing.
    """
    interval_seconds = sample_interval_ms / 1000
    deadline = window_started + duration_ms / 1000
    # This count is a second bound when test clocks or sleeps are patched.
    max_samples = math.ceil(duration_ms / sample_interval_ms) + 1
    samples: list[dict[str, Any]] = []
    owns_position_stream = False
    if max_travel_m > 0 and position_stream is None:
        open_position_stream = getattr(coordinator, "open_position_sample_stream", None)
        if callable(open_position_stream):
            # 🐛 **This was `maxsize=1` until 2026-08-30, and the comment on
            # `_SAFETY_POSITION_STREAM_MAXSIZE` had warned since 2026-08-27 that
            # that value "structurally guarantees a false
            # `position_sequence_gap`".** The beta80 fix reached the two lease
            # wrappers and missed this shared sampler, so any caller that let
            # `_capture_in_window_telemetry` open its OWN stream still tripped.
            # It cost a linear-300 speed check on 2026-08-30, which aborted at
            # 413 ms with `trip_reason: position_sequence_gap` and
            # `travel_at_trip_m: 0.0` -- a healthy feed, read through a
            # one-deep queue.
            position_stream = open_position_stream(
                maxsize=_SAFETY_POSITION_STREAM_MAXSIZE
            )
            owns_position_stream = True
    if max_travel_m > 0 and position_stream is None:
        sample = _in_window_telemetry_sample(
            coordinator,
            index=0,
            window_started=window_started,
            command=command,
            command_args=command_args,
        )
        sample["travel_guard_tripped"] = True
        sample["travel_guard_reason"] = "position_stream_unavailable"
        if travel_abort is not None:
            travel_abort.set()
        return [sample]
    handle = coordinator.manager.mower(coordinator.device_name)
    guard_state = {
        "stream": position_stream,
        "epoch": getattr(handle, "position_epoch", 0),
        # None until the first drained sample seeds it -- see _apply_travel_guard.
        # Seeding from `handle.latest_position_sample` here is what made the
        # guard trip at zero travel on every real run.
        "sequence": None,
        "dropped_samples": getattr(position_stream, "dropped_samples", 0),
        "last_receipt_at": time.monotonic(),
        "last_position": None,
        "cumulative_distance_m": 0.0,
    }
    try:
        while len(samples) < max_samples and not stop_event.is_set():
            sample = _in_window_telemetry_sample(
                coordinator,
                index=len(samples),
                window_started=window_started,
                command=command,
                command_args=command_args,
            )
            samples.append(sample)

            if max_travel_m > 0:
                assert position_stream is not None
                # 🚨 DRAIN, do not take one. A single get_nowait() per poll makes
                # the guard advance at the SAMPLER's cadence rather than the
                # feed's, so `max_travel_m` goes silently soft whenever payloads
                # outpace polls -- and there is no catch-up path, the backlog
                # only grows. `sample_interval_ms` is schema-legal up to 1000 ms
                # against a measured payload cadence of 0.991 s mean / 0.711 s
                # minimum, so at 1000 ms a 23 s window can publish ~32 payloads
                # and consume ~23: the guard would not reach 4.5 m until the
                # mower had driven ~6.5-6.9 m. Draining makes the guard
                # independent of poll cadence.
                # ⚠️ BOUNDED. An unbounded `while True` here spins forever
                # against any queue that does not raise QueueEmpty -- which is
                # every mocked stream, and would also be a live hang if a feed
                # ever refilled faster than we drain. The cap is far above the
                # ~1 payload per poll the ~1 Hz feed can produce at any legal
                # `sample_interval_ms`, so it never truncates a real backlog.
                tripped = False
                drained = 0
                while drained < _PROBE_MAX_DRAIN_PER_POLL:
                    position_sample = None
                    with contextlib.suppress(asyncio.QueueEmpty):
                        position_sample = position_stream.queue.get_nowait()
                    if position_sample is None:
                        # Still call the guard once per poll with no payload, so
                        # the stale-feed detector keeps its own clock running.
                        if drained == 0:
                            tripped = _apply_travel_guard(
                                sample,
                                position_sample=None,
                                guard_state=guard_state,
                                max_travel_m=max_travel_m,
                            )
                        break
                    drained += 1
                    tripped = _apply_travel_guard(
                        sample,
                        position_sample=position_sample,
                        guard_state=guard_state,
                        max_travel_m=max_travel_m,
                    )
                    if tripped:
                        break
                if tripped:
                    if travel_abort is not None:
                        travel_abort.set()
                    break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(interval_seconds, remaining))
    finally:
        if owns_position_stream and position_stream is not None:
            position_stream.close()
    return samples


def _summarize_in_window_telemetry(
    samples: Sequence[Mapping[str, Any]], *, window_duration_ms: int
) -> dict[str, Any]:
    """Summarize fresh reports and position/course changes in cached samples."""
    report_arrivals: list[float] = []
    position_arrivals: list[float] = []
    toward_changes: list[float] = []
    unset = object()
    previous_report: Any = unset
    previous_position: tuple[Any, Any] | None = None
    previous_toward: Any = None
    for sample in samples:
        elapsed_ms = float(sample.get("elapsed_ms", 0.0))
        report_stamp = sample.get("last_report_at_monotonic")
        position = sample.get("position", {}) or {}
        xy = (position.get("x"), position.get("y"))
        toward = position.get("toward")
        if previous_report is not unset and report_stamp != previous_report:
            report_arrivals.append(elapsed_ms)
        if previous_position is not None and xy != previous_position:
            position_arrivals.append(elapsed_ms)
        if previous_toward is not None and toward != previous_toward:
            toward_changes.append(elapsed_ms)
        previous_report = report_stamp
        previous_position = xy
        previous_toward = toward

    position_boundaries = [0.0, *position_arrivals, float(window_duration_ms)]
    position_gaps = [
        round(later - earlier, 3)
        for earlier, later in zip(
            position_boundaries, position_boundaries[1:], strict=False
        )
    ]
    return {
        "sample_count": len(samples),
        "fresh_report_arrival_count": len(report_arrivals),
        "fresh_report_arrivals_elapsed_ms": report_arrivals,
        "fresh_position_arrival_count": len(position_arrivals),
        "fresh_position_arrivals_elapsed_ms": position_arrivals,
        "position_arrival_gaps_including_boundaries_ms": position_gaps,
        "max_position_arrival_gap_ms": max(position_gaps, default=None),
        "toward_change_count": len(toward_changes),
        "toward_changes_elapsed_ms": toward_changes,
        "toward_changed_before_stop": bool(toward_changes),
    }


async def _raw_pymammotion_motion_probe_impl(  # noqa: C901
    coordinator: MammotionReportUpdateCoordinator,
    *,
    position_stream: Any | None,
    command: str = "send_movement",
    linear_speed: int = 400,
    angular_speed: int = 0,
    speed: float = 0.4,
    prefer_ble: bool = True,
    motion_refresh_interval_ms: int = 0,
    in_window_sample_interval_ms: int = 0,
    duration_ms: int = 1300,
    max_travel_m: float = 0.0,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60),
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
) -> dict[str, Any]:
    """Run or simulate one raw pymammotion movement command with telemetry.

    With ``motion_refresh_interval_ms > 0`` the command is held open for
    ``duration_ms`` at the app's cadence and then explicitly stopped, instead of
    being fired once and left to the device watchdog. That is what makes this
    probe able to characterise an **arc** -- a command with both
    ``linear_speed`` and ``angular_speed`` non-zero, which this project has
    never sent despite ``DrvMotionCtrl`` accepting both since the beginning.

    An arc matters because it is the only route to night capability: translation
    keeps ``toward`` (course-over-ground) live, and a live ``toward`` closes a
    heading loop with no VIO at all. See
    `docs/night-motion-options-20260811.md`.
    """
    before = _custom_path_telemetry_snapshot(coordinator)
    gates = _manual_velocity_pulse_gates(
        coordinator,
        before,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    if in_window_sample_interval_ms > 0 and motion_refresh_interval_ms <= 0:
        blockers.append("in_window_sampling_requires_motion_refresh")
    # A window longer than the historic open-loop cap is only allowed when
    # something bounds DISTANCE, and the guard is the sampler, so it needs the
    # sampler running too. Fails closed: without both, the long window is
    # refused rather than silently clamped.
    if duration_ms > _PROBE_DURATION_MS_WITHOUT_TRAVEL_GUARD_MAX:
        if max_travel_m <= 0:
            blockers.append("duration_over_4000ms_requires_max_travel_m")
        if in_window_sample_interval_ms <= 0:
            blockers.append("duration_over_4000ms_requires_in_window_sampling")
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
        "motion_refresh_interval_ms": motion_refresh_interval_ms,
        "in_window_telemetry": {
            "enabled": in_window_sample_interval_ms > 0,
            "sample_interval_ms": in_window_sample_interval_ms,
            "source": "coordinator_cache_only",
            "extra_ble_report_requests_during_window": 0,
            "planned_max_samples": (
                math.ceil(duration_ms / in_window_sample_interval_ms) + 1
                if in_window_sample_interval_ms > 0
                else 0
            ),
            "report_stream_plan": (
                ["async_start_report_stream", "async_start_continuous_reports"]
                if in_window_sample_interval_ms > 0
                else []
            ),
            "samples": [],
            "summary": None,
        },
        "duration_ms": duration_ms,
        "travel_guard": {
            "enabled": max_travel_m > 0,
            "max_travel_m": max_travel_m,
            # The guard reads the ~1 Hz cache and the refresh loop notices
            # within one interval, so it trips late by a knowable amount.
            "expected_overshoot_m": (
                round(_PROBE_TRAVEL_GUARD_OVERSHOOT_M, 3) if max_travel_m > 0 else 0.0
            ),
            # ⚠️ **WORST CASE, not the nominal one — corrected 2026-08-23.**
            # This previously reported `max_travel_m + overshoot`, which assumes
            # the guard works. If the guard no-ops the wall clock still stops the
            # run, but at `duration_ms x speed`, which can be far further: 1.85 m
            # published against 2.21 m real on the arc120 configuration, and
            # 3.31 m at the schema maximum. A corridor must cover whichever bound
            # is larger, so this now reports that. Speed uses the schema's
            # `linear_speed`, not the tested 400, because the schema allows 1000.
            # 🚨 **THE OVERSHOOT APPLIES TO BOTH BRANCHES — corrected 2026-09-03.**
            # The clock branch shipped with NO overshoot term at all. That is the
            # same defect beta98 fixed in `step_path_contained`, which was never
            # propagated back to this sibling. The mandatory stop is issued AFTER
            # `_motion_refresh_window` returns, so when the guard no-ops (the
            # documented latched-position mode) the window runs to the wall clock
            # and the post-stop creep — measured 0.4544 m, attempt 5 — lands
            # entirely outside the number the operator was told covers the path.
            # It BINDS at the schema maximum: 12000 ms at linear 400 gives a
            # 3.60 m clock bound against a 3.50 m guard bound, so the larger
            # branch was the uncertified one. Both branches now carry it.
            "corridor_must_cover_m": (
                round(
                    max(
                        max_travel_m + _PROBE_TRAVEL_GUARD_OVERSHOOT_M,
                        _PROBE_SPEED_PER_LINEAR_UNIT_MS
                        * abs(linear_speed)
                        * duration_ms
                        / 1000.0
                        + _PROBE_TRAVEL_GUARD_OVERSHOOT_M,
                    ),
                    3,
                )
                if max_travel_m > 0
                else 0.0
            ),
            # Reported WITHOUT the overshoot, deliberately: this is the distance
            # the mower covers while driving, and `corridor_must_cover_m` above
            # is the number to size a corridor from.
            "clock_bound_m": round(
                _PROBE_SPEED_PER_LINEAR_UNIT_MS
                * abs(linear_speed)
                * duration_ms
                / 1000.0,
                3,
            ),
            "tripped": False,
        },
        # Named so the record says what kind of motion this was without anyone
        # having to re-read the speeds.
        "motion_axes": (
            "arc"
            if int(linear_speed) and int(angular_speed)
            else "linear"
            if int(linear_speed)
            else "angular"
            if int(angular_speed)
            else "none"
        ),
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

    if max_travel_m > 0 and position_stream is None:
        result["reason"] = "position_stream_unavailable"
        result["blockers"] = ["position_stream_unavailable"]
        result["would_send"] = False
        result["real_probe_allowed"] = False
        return result

    if in_window_sample_interval_ms > 0:
        stream_duration_ms = max(10_000, duration_ms + 5_000)
        stream_result: dict[str, Any] = {
            "attempted": True,
            "duration_ms": stream_duration_ms,
            "started": False,
            "continuous_started": False,
            "error": None,
        }
        result["in_window_telemetry"]["report_stream"] = stream_result
        try:
            if hasattr(coordinator, "async_start_report_stream"):
                await coordinator.async_start_report_stream(
                    duration_ms=stream_duration_ms
                )
                stream_result["started"] = True
            if hasattr(coordinator, "async_start_continuous_reports"):
                await coordinator.async_start_continuous_reports(
                    duration_ms=stream_duration_ms
                )
                stream_result["continuous_started"] = True
            stream_result["queue_settle"] = await _settle_ble_command_queue(coordinator)
        except Exception as err:  # noqa: BLE001
            stream_result["error"] = f"{type(err).__name__}: {err}"
            result["reason"] = "report_stream_failed"
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

    if motion_refresh_interval_ms > 0:
        # Hold the command open at the app's cadence, then STOP it explicitly.
        # Without the stop the window would end and the device watchdog would
        # coast the mower to a halt on its own timeline, which is exactly the
        # ambiguity this probe exists to remove.
        async def _resend() -> None:
            await _send_manager_command_with_args(
                coordinator,
                command,
                prefer_ble=prefer_ble,
                command_kwargs=command_args,
            )

        window_started = time.monotonic()
        sampler_stop = asyncio.Event()
        travel_abort = asyncio.Event()
        sampler_task: asyncio.Task[list[dict[str, Any]]] | None = None
        if in_window_sample_interval_ms > 0:
            sampler_task = asyncio.create_task(
                _capture_in_window_telemetry(
                    coordinator,
                    sample_interval_ms=in_window_sample_interval_ms,
                    duration_ms=duration_ms,
                    window_started=window_started,
                    stop_event=sampler_stop,
                    command=command,
                    command_args=command_args,
                    max_travel_m=max_travel_m,
                    travel_abort=travel_abort,
                    position_stream=position_stream,
                )
            )

            # 🚨 A DEAD SAMPLER MUST NOT LEAVE THE MOWER DRIVING ON THE CLOCK.
            # The sampler IS the distance guard, so if its task dies the guard
            # silently stops existing and the window falls back to `duration_ms`
            # -- the bound the guard was introduced to replace. Setting the abort
            # from a done-callback turns that into an early stop instead.
            if max_travel_m > 0:

                def _abort_if_sampler_died(task: asyncio.Task[Any]) -> None:
                    if task.cancelled() or task.exception() is not None:
                        travel_abort.set()

                sampler_task.add_done_callback(_abort_if_sampler_died)
        try:
            result["motion_refresh"] = await _motion_refresh_window(
                coordinator,
                resend=_resend,
                duration_seconds=duration_ms / 1000.0,
                refresh_interval_ms=motion_refresh_interval_ms,
                abort_event=travel_abort if max_travel_m > 0 else None,
            )
        except BaseException:
            sampler_stop.set()
            if sampler_task is not None:
                with contextlib.suppress(BaseException):
                    await sampler_task
            raise
        sampler_stop.set()
        result["stop_result"] = await _manual_velocity_stop_attempt(
            coordinator, use_wifi=not prefer_ble
        )
        if sampler_task is not None:
            in_window_samples = await sampler_task
            result["in_window_telemetry"]["samples"] = in_window_samples
            result["in_window_telemetry"]["summary"] = _summarize_in_window_telemetry(
                in_window_samples,
                window_duration_ms=duration_ms,
            )
            if max_travel_m > 0:
                tripped = any(
                    sample.get("travel_guard_tripped") for sample in in_window_samples
                )
                result["travel_guard"]["tripped"] = tripped
                result["travel_guard"]["trip_reason"] = next(
                    (
                        sample.get("travel_guard_reason")
                        for sample in in_window_samples
                        if sample.get("travel_guard_tripped")
                    ),
                    None,
                )
                # Renamed from `observed_travel_m` 2026-08-23: this is travel AT
                # THE TRIP, not travel of the run (1.6093 against 1.8074 on the
                # 2026-08-22 guard run), and the old name invited reading it as
                # the latter.
                result["travel_guard"]["travel_at_trip_m"] = max(
                    (
                        sample.get("cumulative_travel_m", 0.0)
                        for sample in in_window_samples
                    ),
                    default=0.0,
                )

    # 🚨 FORCE A REPORT REFRESH, THEN WAIT FOR THE FEED TO SETTLE. Without this
    # the probe reads whatever the coordinator last cached and is blind to its
    # own motion -- which is exactly what happened on 2026-08-12: an arc that
    # moved the mower 0.58 m and rotated its course 22.2 deg reported four
    # bit-identical samples, and the new position only appeared in the cache
    # about five minutes later. The device does not push position while
    # stationary, so after a pulse ends there is nothing to update the cache
    # until something asks. Every real executor already does this; the probe was
    # the one motion path that did not, and it produced a null result that was
    # nearly written up as "arcs do not actuate".
    result["post_command_feedback_refresh"] = await _refresh_position_after_raw_motion(
        coordinator
    )
    settle = await _settle_linear_position_feed(coordinator, before)
    result["position_settle"] = settle
    result["position_moved"] = bool(settle.get("moved"))
    result["position_feed_stale"] = bool(settle.get("feed_stale"))

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


async def _raw_pymammotion_motion_probe(  # noqa: PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    *,
    command: str = "send_movement",
    linear_speed: int = 400,
    angular_speed: int = 0,
    speed: float = 0.4,
    prefer_ble: bool = True,
    motion_refresh_interval_ms: int = 0,
    in_window_sample_interval_ms: int = 0,
    duration_ms: int = 1300,
    max_travel_m: float = 0.0,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60),
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
) -> dict[str, Any]:
    """Own position evidence before dispatch for a distance-guarded probe."""
    open_position_stream = getattr(coordinator, "open_position_sample_stream", None)
    # Same defect, same day, same fix as the sampler above: a one-deep queue is
    # latest-wins, so every sample that arrives while the caller is still
    # setting up is dropped and the contiguity check trips on a healthy feed.
    position_stream = (
        open_position_stream(maxsize=_SAFETY_POSITION_STREAM_MAXSIZE)
        if max_travel_m > 0 and not dry_run and callable(open_position_stream)
        else None
    )
    try:
        return await _raw_pymammotion_motion_probe_impl(
            coordinator,
            position_stream=position_stream,
            command=command,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            speed=speed,
            prefer_ble=prefer_ble,
            motion_refresh_interval_ms=motion_refresh_interval_ms,
            in_window_sample_interval_ms=in_window_sample_interval_ms,
            duration_ms=duration_ms,
            max_travel_m=max_travel_m,
            sample_delays=sample_delays,
            dry_run=dry_run,
            confirm_blades_off=confirm_blades_off,
            confirm_clear_area=confirm_clear_area,
        )
    finally:
        if position_stream is not None:
            position_stream.close()


# ============================================================================
# Continuous motion window -- the Phase 2 executor.
#
# Design decisions (operator-approved, 2026-08-23,
# docs/phase2-continuous-motion-design-20260823.md): straight-line segments
# only, no turns in v1; extends the bounded-window pattern below rather than a
# persistent velocity loop; corrects every ~1 Hz position arrival on MEASURED
# heading, never an integrated yaw-rate model; stops safely on a detected BLE
# stall. Gap reconciliation and the exact stall mechanism are documented in
# docs/phase2-gap-reconciliation-20260823.md. Pass criteria for a real run are
# pre-registered in docs/continuous-motion-feasibility-plan-20260821.md
# ("Phase 2 pass criteria") and confirmed still current in
# docs/phase2-gate-readiness-20260823.md -- this executor does not define new
# ones.
#
# `_motion_refresh_window` is deliberately NOT reused for the refresh loop
# below: the plan is explicit that its contract intentionally resends an
# IDENTICAL command, and retrofitting a variable one into it would be the
# wrong shape. `_continuous_refresh_window` is the "one serialized writer"
# the plan calls for: only it ever touches BLE for movement.
# `_continuous_decision_loop` only ever writes the shared `command_state` the
# writer reads. Python's single-threaded event loop makes that handoff safe
# with no lock, the same reasoning `_capture_in_window_telemetry` already
# relies on for `travel_abort`.
# ============================================================================

# Mirrors the Phase 1 corridor discipline exactly
# (scripts/freeze_phase1_corridors.py, scripts/scan_contained_bearings.py): the
# frozen route start is never re-derived from live position, only checked
# against it. On 2026-08-20 a dispatch script re-derived an endpoint "to be
# safe" and silently drove a path the scan never covered.
_CONTINUOUS_MAX_START_DRIFT_M = 0.30
_CONTINUOUS_MIRROR_SUM_DEGREES = _TOWARD_MIRROR_DEGREES
# The mower is already stopped during this wait, so it consumes no blind-motion
# budget and does not enlarge the acquisition disk (1.34 m since the 2026-08-27
# budget change; 1.06 m before it). The 3.5 s bound covers the stationary
# matrix's 2.909 s maximum position gap with measured margin.
_CONTINUOUS_POST_STOP_OBSERVATION_S = 3.5

#: Queue depth for a SAFETY position-evidence stream, in samples.
#:
#: 🚨 **`maxsize=1` structurally guarantees a false `position_sequence_gap`.**
#: `PositionSampleStream._offer` is latest-wins: when the queue is full it
#: discards the OLDEST sample and increments `dropped_samples`. A safety consumer
#: does not read continuously -- it opens the stream, then runs gates, takes a
#: generation baseline, starts the report stream and settles the BLE queue before
#: its first `get()`. At the measured ~1 Hz payload cadence that setup easily
#: spans one or more samples, so at depth 1 every one of them is dropped and the
#: contiguity check trips on a perfectly healthy feed.
#:
#: This refused Phase 2 steering run 1 before dispatch on 2026-08-27. It fails
#: CLOSED, so nothing unsafe happened -- but it is a RELIABILITY defect and must
#: not be read as evidence about the position channel.
#:
#: 64 covers a minute of setup at ~1 Hz with room to spare. Buffering weakens no
#: check: sequence contiguity, epoch, the evidence boundary and the drop counter
#: all still apply, and a drop now means a genuine overrun rather than "the
#: consumer was busy starting up".
_SAFETY_POSITION_STREAM_MAXSIZE = 64


def _continuous_course_heading(toward: float) -> float:
    """Return the map-local course heading from the compass mirror.

    `map_bearing = 90.13 - toward`, not an additive offset -- the same
    convention every Phase 1 capture used this week.
    """
    return (_CONTINUOUS_MIRROR_SUM_DEGREES - toward) % 360.0


def _continuous_motion_gates(
    coordinator: MammotionReportUpdateCoordinator,
    before: dict[str, Any],
    *,
    route_start: dict[str, float],
    route_target: dict[str, float],
    config: ContinuousControllerConfig,
    corridor_polygon: list[dict[str, float]],
    dry_run: bool,
    confirm_blades_off: bool,
    confirm_clear_area: bool,
) -> list[dict[str, Any]]:
    """Return the safety gates for one continuous-motion window.

    Extends `_manual_velocity_pulse_gates` with the checks specific to a
    continuous window. `ContinuousRoute.contained` in the pure controller is
    caller-supplied and never re-derived
    (docs/phase2-gap-reconciliation-20260823.md) -- these gates are what earns
    that trust before the window opens, not the pure controller itself.
    """
    gates = list(
        _manual_velocity_pulse_gates(
            coordinator,
            before,
            dry_run=dry_run,
            confirm_blades_off=confirm_blades_off,
            confirm_clear_area=confirm_clear_area,
        )
    )
    corridor_points = [ContinuousPoint(**point) for point in corridor_polygon]
    corridor_valid = polygon_is_valid(corridor_points)
    position = before.get("position", {}) or {}
    live_x, live_y = position.get("x"), position.get("y")
    drift = (
        math.hypot(live_x - route_start["x"], live_y - route_start["y"])
        if isinstance(live_x, (int, float)) and isinstance(live_y, (int, float))
        else None
    )
    gates.append(
        {
            "name": "corridor_polygon_valid",
            "passed": corridor_valid,
            "detail": "The frozen corridor must be a real polygon, >= 3 vertices.",
        }
    )
    gates.append(
        {
            "name": "frozen_start_inside_corridor",
            "passed": corridor_valid
            and _point_in_polygon(route_start, corridor_polygon),
            "detail": "The frozen route start itself must be inside the frozen corridor.",
        }
    )
    gates.append(
        {
            "name": "start_drift_within_bound",
            "passed": dry_run
            or (drift is not None and drift <= _CONTINUOUS_MAX_START_DRIFT_M),
            "detail": (
                f"Live position must be within {_CONTINUOUS_MAX_START_DRIFT_M} m "
                "of the frozen route start. The start is never re-derived from "
                "live position -- this gate aborts instead."
            ),
            "diagnostics": {"drift_m": drift},
        }
    )

    # No stationary telemetry value is accepted as an opening course. Before a
    # real window may move straight to acquire a position-chord bearing, EVERY
    # possible ray must fit inside the frozen corridor through the 2 s timeout
    # plus the banked 0.50 m stopping/guard overshoot.
    feasibility = (
        blind_acquisition_feasibility(
            ContinuousPoint(float(live_x), float(live_y)),
            corridor_points,
            config,
        )
        if isinstance(live_x, (int, float))
        and isinstance(live_y, (int, float))
        and corridor_valid
        else None
    )
    gates.append(
        {
            "name": "blind_heading_acquisition_contained",
            "passed": feasibility is not None and feasibility.feasible,
            "detail": (
                "No opening toward value is trusted. A complete disk around "
                "the live start must fit the frozen corridor before the mower "
                "may drive straight to derive a fresh position-chord course: "
                f"{config.max_safety_speed_mps} m/s for "
                f"{config.max_heading_acquisition_s} s plus "
                f"{config.stop_overshoot_m} m stopping/guard overshoot."
            ),
            "diagnostics": (
                dataclasses.asdict(feasibility) if feasibility is not None else None
            ),
        }
    )
    return gates


async def _wait_for_fresh_continuous_origin(  # noqa: C901
    position_stream: Any,
    *,
    request_started_at: float,
    baseline_sequence: int,
    baseline_epoch: int,
    timeout_s: float,
    handle: Any | None = None,
    report_lease: Any | None = None,
    report_generation: Any | None = None,
    baseline_dropped_samples: int | None = None,
) -> dict[str, Any]:
    """Wait for new position evidence, including an unchanged coordinate."""
    started = time.monotonic()
    expected_sequence = baseline_sequence + 1
    dropped_at_start = (
        position_stream.dropped_samples
        if baseline_dropped_samples is None
        else baseline_dropped_samples
    )
    while (remaining := timeout_s - (time.monotonic() - started)) > 0:
        if position_stream.dropped_samples != dropped_at_start:
            return {
                "ok": False,
                "reason": "position_sequence_gap",
                "sample": None,
                "elapsed_s": time.monotonic() - started,
            }
        if (
            handle is not None
            and report_lease is not None
            and not handle.report_subscription_lease_is_current(report_lease)
        ):
            return {
                "ok": False,
                "reason": "report_subscription_lease_lost",
                "sample": None,
                "elapsed_s": time.monotonic() - started,
            }
        if (
            handle is not None
            and report_generation is not None
            and handle.report_subscription_generation != report_generation.generation
        ):
            return {
                "ok": False,
                "reason": "report_subscription_generation_changed",
                "sample": None,
                "elapsed_s": time.monotonic() - started,
            }
        if handle is not None and handle.position_epoch != baseline_epoch:
            return {
                "ok": False,
                "reason": "position_epoch_changed",
                "sample": None,
                "elapsed_s": time.monotonic() - started,
            }
        try:
            sample = await asyncio.wait_for(position_stream.queue.get(), remaining)
        except TimeoutError:
            break
        if position_stream.dropped_samples != dropped_at_start:
            return {
                "ok": False,
                "reason": "position_sequence_gap",
                "sample": None,
                "elapsed_s": time.monotonic() - started,
            }
        if sample.epoch != baseline_epoch:
            return {
                "ok": False,
                "reason": "position_epoch_changed",
                "sample": None,
                "elapsed_s": time.monotonic() - started,
            }
        # A sample may have been queued between opening the stream and taking the
        # baseline. It is explicitly pre-generation evidence, so SKIP it rather
        # than calling it a gap.
        #
        # 🚨 This skip was missing until 2026-08-27 and
        # `_wait_for_position_subscription_ready` already had it. The asymmetry
        # refused Phase 2 steering run 1 before dispatch with
        # `position_sequence_gap` on a perfectly healthy feed. It fails CLOSED,
        # so nothing unsafe happened -- but it is a reliability defect, not
        # evidence about the position channel, and reading it as the latter would
        # have sent a future session chasing a telemetry ghost.
        if sample.sequence <= baseline_sequence:
            continue
        if sample.sequence != expected_sequence:
            return {
                "ok": False,
                "reason": "position_sequence_gap",
                "sample": None,
                "elapsed_s": time.monotonic() - started,
            }
        expected_sequence += 1
        if sample.received_at_monotonic < request_started_at:
            continue
        consumed_at = time.monotonic()
        sample_diagnostics = {
            "sequence": sample.sequence,
            "epoch": sample.epoch,
            "source": sample.source,
            "transport": sample.transport,
            "position": {
                "x": sample.x,
                "y": sample.y,
                "toward": sample.toward,
                "pos_type": sample.pos_type,
                "zone_hash": sample.zone_hash,
                "rtk_status": sample.rtk_status,
            },
            "valid_for_motion": sample.valid_for_motion,
            "rejection_reason": sample.rejection_reason,
            "receipt_to_consumption_s": max(
                consumed_at - sample.received_at_monotonic, 0.0
            ),
            "pipeline_latency_s": max(
                sample.published_at_monotonic - sample.received_at_monotonic, 0.0
            ),
        }
        if not sample.valid_for_motion:
            return {
                "ok": False,
                "reason": "position_invalid_for_motion",
                "sample": sample_diagnostics,
                "elapsed_s": consumed_at - started,
            }
        return {
            "ok": True,
            "reason": None,
            "sample": sample_diagnostics,
            "elapsed_s": consumed_at - started,
        }
    if handle is not None and handle.position_epoch != baseline_epoch:
        return {
            "ok": False,
            "reason": "position_epoch_changed",
            "sample": None,
            "elapsed_s": time.monotonic() - started,
        }
    # `generic_report_advanced` is reported as EVIDENCE, never promoted into the
    # reason. This wait is bounded by `max_heading_acquisition_s`, raised 2.0 ->
    # 3.0 s on 2026-08-27 (see `docs/phase2-acquisition-budget-decision-20260827.md`).
    # The argument for not promoting a timeout here to a channel-fault verdict is
    # unchanged by that raise: the beta76 stationary matrix showed 28 of 1434
    # healthy position intervals -- 1.95% -- exceeding 2.0 s while generic frames
    # arrived at roughly 2 Hz, so a tail-of-distribution gap is a NORMAL event on
    # this feed. Calling it `position_channel_stalled` would give that routine gap
    # the same name as cell 12's real outage (3 payloads, then ~119 s of silence
    # against 118 normal generic reports).
    # ⚠️ The 1.95% figure is measured against the OLD 2.0 s bound and is retained
    # as the historical measurement it is -- the fraction exceeding 3.0 s is
    # necessarily smaller but has NOT been re-derived, because the beta77 matrix
    # artifact was written with `include_raw_samples=False` and its per-sample
    # position intervals were stripped. Do not quote 1.95% as the rate for 3.0 s.
    # The readiness probe's 3.5 s budget is a different case: nothing in 1434
    # intervals exceeded it, so `stalled` is well founded there and is kept.
    generic_advanced = (
        handle is not None
        and report_generation is not None
        and handle.last_report_at > report_generation.baseline_last_report_at
    )
    return {
        "ok": False,
        "reason": "fresh_origin_timeout",
        "generic_report_advanced": generic_advanced,
        "sample": None,
        "elapsed_s": time.monotonic() - started,
    }


async def _wait_for_post_stop_position(
    position_stream: Any,
    *,
    after_sequence: int,
    epoch: int,
    timeout_s: float,
) -> tuple[Any | None, str | None]:
    """Observe the full stopped window and return its newest valid position."""
    dropped_at_start = position_stream.dropped_samples
    expected_sequence = after_sequence + 1
    deadline = time.monotonic() + timeout_s
    latest = None
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            sample = await asyncio.wait_for(position_stream.queue.get(), remaining)
        except TimeoutError:
            break
        if position_stream.dropped_samples != dropped_at_start:
            return None, "position_sequence_gap"
        if sample.epoch != epoch:
            return None, "position_epoch_changed"
        if sample.sequence != expected_sequence:
            return None, "position_sequence_gap"
        if not sample.valid_for_motion:
            return None, "position_invalid_for_motion"
        expected_sequence += 1
        latest = sample
    return (
        (latest, None) if latest is not None else (None, "post_stop_position_timeout")
    )


async def _continuous_decision_loop(  # noqa: C901
    position_stream: Any,
    *,
    route: ContinuousRoute,
    corridor_polygon: list[dict[str, float]],
    config: ContinuousControllerConfig,
    opening_position: ContinuousPoint,
    opening_sequence: int,
    opening_epoch: int,
    window_started: float,
    sample_interval_ms: int,
    refresh_state: dict[str, Any],
    command_state: dict[str, int],
    decision_abort: asyncio.Event,
    stop_event: asyncio.Event,
    acquisition_only: bool = False,
) -> list[dict[str, Any]]:
    """Consume ordered position evidence and compute the next command only."""
    del sample_interval_ms
    decisions: list[dict[str, Any]] = []
    deadline = window_started + config.max_window_s
    last_sequence = opening_sequence
    last_position = opening_position
    last_position_receipt = window_started
    heading_anchor = opening_position
    heading_evidence: HeadingEvidence | None = None
    cumulative_distance_m = 0.0
    consumed_completions = 0
    alignment_checked = False
    dropped_samples = position_stream.dropped_samples

    while not stop_event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            decision_abort.set()
            break
        sample = None
        with contextlib.suppress(TimeoutError):
            sample = await asyncio.wait_for(
                position_stream.queue.get(), min(0.1, remaining)
            )

        now = time.monotonic()
        elapsed_s = now - window_started
        timed_out = heading_evidence is None and (
            elapsed_s >= config.max_heading_acquisition_s
        )
        heading_stale = heading_evidence is not None and (
            now - last_position_receipt > config.max_heading_age_s
        )
        fault_reason: str | None = None
        telemetry_age_s = max(now - last_position_receipt, 0.0)
        if position_stream.dropped_samples != dropped_samples:
            fault_reason = "position_sequence_gap"

        current = last_position
        if sample is not None:
            telemetry_age_s = max(now - sample.received_at_monotonic, 0.0)
            if sample.epoch != opening_epoch:
                fault_reason = "position_epoch_changed"
            elif sample.sequence != last_sequence + 1:
                fault_reason = "position_sequence_gap"
            elif not sample.valid_for_motion:
                fault_reason = "position_invalid_for_motion"
            elif telemetry_age_s > config.max_heading_age_s:
                fault_reason = "telemetry_stale"
            if fault_reason is None:
                current = ContinuousPoint(float(sample.x), float(sample.y))
                cumulative_distance_m += math.hypot(
                    current.x - last_position.x, current.y - last_position.y
                )
                last_position = current
                last_sequence = sample.sequence
                last_position_receipt = sample.received_at_monotonic
                candidate = course_from_position_chord(
                    heading_anchor,
                    current,
                    measured_at_s=elapsed_s,
                    min_chord_m=config.min_travel_for_heading_trust_m,
                )
                if candidate is not None:
                    heading_evidence = candidate
                    heading_anchor = current

        if sample is None and not (timed_out or heading_stale or fault_reason):
            continue

        elapsed_ms = elapsed_s * 1000.0
        completions = refresh_state["completions_elapsed_ms"]
        new_completions = completions[consumed_completions:]
        consumed_completions = len(completions)
        previous_decision_ms = refresh_state.get("last_decision_elapsed_ms", 0.0)
        gap_s = 0.0
        if new_completions:
            bounds = [previous_decision_ms, *new_completions, elapsed_ms]
            gap_s = (
                max(b - a for a, b in zip(bounds, bounds[1:], strict=False)) / 1000.0
            )
        refresh_state["last_decision_elapsed_ms"] = elapsed_ms
        refresh_age_s = (
            max(elapsed_ms - completions[-1], 0.0) / 1000.0
            if completions
            else elapsed_s
        )
        observation = ContinuousObservation(
            position=current,
            course_heading_degrees=(
                heading_evidence.course_heading_degrees
                if heading_evidence is not None
                else None
            ),
            telemetry_age_s=telemetry_age_s,
            refresh_age_s=refresh_age_s,
            elapsed_s=elapsed_s,
            distance_travelled_m=cumulative_distance_m,
            heading_evidence=heading_evidence,
            refresh_max_gap_since_last_decision_s=gap_s,
        )
        decision = continuous_control_decision(route, observation, config)
        if acquisition_only and heading_evidence is not None:
            decision = dataclasses.replace(
                decision,
                action="stop",
                reason="heading_acquired",
                linear_speed=0,
                angular_speed=0,
            )
        if fault_reason is not None:
            decision = dataclasses.replace(
                decision,
                action="stop",
                reason=fault_reason,
                linear_speed=0,
                angular_speed=0,
            )
        inside_corridor = _point_in_polygon(
            {"x": current.x, "y": current.y}, corridor_polygon
        )
        feasibility = None
        if (
            heading_evidence is not None
            and not alignment_checked
            and decision.action != "stop"
        ):
            feasibility = alignment_feasibility(
                route,
                opening_position=opening_position,
                position=current,
                heading_evidence=heading_evidence,
                elapsed_s=elapsed_s,
                cumulative_distance_m=cumulative_distance_m,
                config=config,
            )
            alignment_checked = True
            if not feasibility.feasible:
                decision = dataclasses.replace(
                    decision,
                    action="stop",
                    reason="opening_alignment_infeasible",
                    linear_speed=0,
                    angular_speed=0,
                )
        if not inside_corridor:
            decision = dataclasses.replace(
                decision,
                action="stop",
                reason="corridor_breach",
                linear_speed=0,
                angular_speed=0,
            )

        decisions.append(
            {
                "index": len(decisions),
                "phase": (
                    "stopping"
                    if decision.action == "stop"
                    else "steering"
                    if heading_evidence is not None and alignment_checked
                    else "acquiring_heading"
                ),
                "elapsed_s": round(elapsed_s, 3),
                "position_sequence": last_sequence,
                "position_epoch": opening_epoch,
                "cumulative_distance_m": cumulative_distance_m,
                "observation": dataclasses.asdict(observation),
                "decision": dataclasses.asdict(decision),
                "alignment_feasibility": (
                    dataclasses.asdict(feasibility) if feasibility else None
                ),
                "inside_corridor": inside_corridor,
            }
        )
        if decision.action == "stop":
            decision_abort.set()
            break
        command_state["angular_speed"] = decision.angular_speed
    return decisions


async def _continuous_refresh_window(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    command_state: dict[str, int],
    prefer_ble: bool,
    duration_seconds: float,
    refresh_interval_ms: int,
    window_started: float,
    refresh_state: dict[str, Any],
    abort_event: asyncio.Event,
) -> dict[str, Any]:
    """Hold a continuous command open, resending whatever `command_state` holds.

    Adapted from `_motion_refresh_window`'s fixed-cadence-from-window-start
    design (measured 2026-08-09 to correct a sleep-then-await drift), but able
    to resend a CHANGING command -- `_motion_refresh_window` is deliberately
    left alone because its contract promises an identical one.
    """
    report: dict[str, Any] = {
        "refresh_interval_ms": refresh_interval_ms,
        "refresh_commands_sent": 0,
        "refresh_write_durations_ms": [],
        "refresh_write_completions_elapsed_ms": [],
        "commands_by_refresh": [],
        "refresh_error": None,
        "aborted_early": False,
    }
    interval_seconds = refresh_interval_ms / 1000.0
    deadline = window_started + duration_seconds
    max_refreshes = max(int(duration_seconds / interval_seconds), 0)

    while report["refresh_commands_sent"] < max_refreshes:
        if abort_event.is_set():
            report["aborted_early"] = True
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        next_fire = (
            window_started + (report["refresh_commands_sent"] + 1) * interval_seconds
        )
        await asyncio.sleep(min(max(0.0, next_fire - time.monotonic()), remaining))
        if abort_event.is_set():
            report["aborted_early"] = True
            break

        linear_speed = command_state["linear_speed"]
        angular_speed = command_state["angular_speed"]
        resend_started = time.monotonic()
        try:
            await _send_manager_command_with_args(
                coordinator,
                "send_movement",
                prefer_ble=prefer_ble,
                command_kwargs={
                    "linear_speed": linear_speed,
                    "angular_speed": angular_speed,
                },
            )
        except asyncio.CancelledError:
            # Same trap `_motion_open_sleep` exists for: a movement command is
            # already open on the mower, so exiting without stopping leaves it
            # driving until its own device-side timeout.
            await _deliver_stop_despite_cancellation(coordinator)
            raise
        except ManualMotionCancelledError:
            await _deliver_stop_despite_cancellation(coordinator)
            raise
        except Exception as err:  # noqa: BLE001
            report["refresh_error"] = f"{type(err).__name__}: {err}"
            break

        completion_ms = round((time.monotonic() - window_started) * 1000, 3)
        report["refresh_write_durations_ms"].append(
            round((time.monotonic() - resend_started) * 1000, 3)
        )
        report["refresh_write_completions_elapsed_ms"].append(completion_ms)
        report["commands_by_refresh"].append(
            {"linear_speed": linear_speed, "angular_speed": angular_speed}
        )
        refresh_state["completions_elapsed_ms"].append(completion_ms)
        report["refresh_commands_sent"] += 1

    report["elapsed_ms"] = round((time.monotonic() - window_started) * 1000, 3)
    return report


async def _continuous_motion_window_impl(  # noqa: C901, PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    *,
    position_stream: Any | None,
    report_lease: Any | None = None,
    acquisition_only: bool,
    route_start: dict[str, float],
    route_target: dict[str, float],
    corridor_polygon: list[dict[str, float]],
    linear_speed: int = 400,
    max_abs_angular_speed: int = 180,
    duration_ms: int = 4000,
    motion_refresh_interval_ms: int = 200,
    decision_sample_interval_ms: int = 100,
    max_distance_m: float = 1.50,
    max_cross_track_m: float = 0.30,
    prefer_ble: bool = True,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    confirm_steering_validation_run: bool = False,
) -> dict[str, Any]:
    """Run or simulate one continuous straight-line steering window.

    The Phase 2 executor. Straight-line only, per the operator's 2026-08-23
    design decision -- `route_target` is a point, not a path with turns.
    `corridor_polygon` must be pre-scanned and margin-verified OFFLINE, the
    same discipline `scripts/freeze_phase1_corridors.py` and
    `scripts/scan_contained_bearings.py` already use for Phase 1 captures.
    This function never scans or freezes one itself.

    With `dry_run=True` (the default) this sends nothing: it returns the full
    plan, every safety gate, and `would_send: false`. A real run additionally
    requires `confirm_blades_off` and `confirm_clear_area`, exactly like every
    other real-motion probe in this project.
    """
    before = _custom_path_telemetry_snapshot(coordinator)
    config = ContinuousControllerConfig(
        linear_speed=linear_speed,
        max_abs_angular_speed=max_abs_angular_speed,
        max_cross_track_m=max_cross_track_m,
        max_window_s=duration_ms / 1000.0,
        max_distance_m=max_distance_m,
    )
    gates = _continuous_motion_gates(
        coordinator,
        before,
        route_start=route_start,
        route_target=route_target,
        config=config,
        corridor_polygon=corridor_polygon,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    acquisition_diagnostics = next(
        (
            gate.get("diagnostics")
            for gate in gates
            if gate["name"] == "blind_heading_acquisition_contained"
        ),
        None,
    )

    route = ContinuousRoute(
        start=ContinuousPoint(**route_start),
        target=ContinuousPoint(**route_target),
        # Earned by the gates above (frozen_start_inside_corridor,
        # start_drift_within_bound), never trusted blindly.
        contained=True,
    )

    result: dict[str, Any] = {
        "service": (
            SERVICE_HEADING_ACQUISITION_WINDOW
            if acquisition_only
            else SERVICE_CONTINUOUS_MOTION_WINDOW
        ),
        "mode": (
            "dry_run"
            if dry_run
            else "real_heading_acquisition_window"
            if acquisition_only
            else "real_continuous_motion_window"
        ),
        "dry_run": dry_run,
        "route": {"start": route_start, "target": route_target},
        "corridor_polygon": corridor_polygon,
        "config": dataclasses.asdict(config),
        "duration_ms": duration_ms,
        "motion_refresh_interval_ms": motion_refresh_interval_ms,
        "decision_sample_interval_ms": decision_sample_interval_ms,
        "phase": "preflight",
        "heading_state": {
            "phase": "pending_position_chord",
            "source": None,
            "minimum_chord_m": config.min_travel_for_heading_trust_m,
            "maximum_age_s": config.max_heading_age_s,
        },
        "acquisition": acquisition_diagnostics,
        "remaining_budgets": {
            "acquisition_s": config.max_heading_acquisition_s,
            "window_s": config.max_window_s,
            "distance_m": config.max_distance_m,
        },
        "post_stop_observation_timeout_s": _CONTINUOUS_POST_STOP_OBSERVATION_S,
        "safety_gates": gates,
        "blockers": blockers,
        "would_send": False,
        "command_result": {
            "attempted": False,
            "ok": None,
            "error": None,
            "duration_ms": None,
        },
        "decisions": [],
        "fresh_origin": None,
        "motion_refresh": None,
        "stop_result": None,
        "reason": "dry_run" if dry_run else None,
    }
    if dry_run or blockers:
        if not dry_run:
            result["reason"] = "blocked"
        return result

    # 🚨 **THE STEERING BOUNDARY.** Until 2026-08-27 this refused all real
    # steering unconditionally, because position-event acquisition had never
    # been validated on hardware. It has been now
    # (`docs/evidence-phase2-acquisition-beta79-20260827.json`: `heading_acquired`,
    # both decisions at `angular_speed: 0`, chord 0.4667 m), which satisfies
    # condition A of `docs/phase2-steering-refusal-recommendation-20260826.md`.
    #
    # The refusal is therefore now conditional rather than absolute -- but
    # steering stays OFF unless a caller opts in per call. It is not enough to
    # arm the motion gate. ⚠️ The corrected steering sign has still never moved a
    # wheel: the sole physical attempt (2026-08-24) diverged monotonically
    # (46.64° → 48.25° → 77.40° while saturated at +180) and hard-aborted on the
    # 0.30 m cross-track bound at 0.517 m. Every guard held, which is why a second
    # attempt is reasonable -- and why the opt-in is explicit.
    if not acquisition_only and not confirm_steering_validation_run:
        result["reason"] = "steering_not_confirmed_for_validation_run"
        result["blockers"] = ["steering_not_confirmed_for_validation_run"]
        return result

    if position_stream is None:
        result["reason"] = "position_stream_unavailable"
        result["blockers"] = ["position_stream_unavailable"]
        return result

    handle = coordinator.manager.mower(coordinator.device_name)
    begin_generation = getattr(handle, "begin_report_subscription_generation", None)
    if report_lease is None or not callable(begin_generation):
        result["reason"] = "report_subscription_generation_unavailable"
        result["blockers"] = ["report_subscription_generation_unavailable"]
        return result
    report_generation = begin_generation(report_lease)
    baseline_sequence = report_generation.baseline_position_sequence
    baseline_epoch = report_generation.baseline_position_epoch
    baseline_dropped_samples = position_stream.dropped_samples

    stream_duration_ms = max(10_000, duration_ms + 5_000)
    stream_result: dict[str, Any] = {
        "attempted": True,
        "started": False,
        "continuous_started": False,
        "error": None,
        # Both start calls below reach the queue at Priority.BACKGROUND with
        # skip_if_saga_active=True, so a running saga DROPS them silently while
        # `started`/`continuous_started` still record True. Without this capture a
        # dispatch failure is indistinguishable from a telemetry stall, and the
        # subsequent timeout gets scored as evidence about the position channel.
        "saga_active_before_request": _saga_active_for_diagnostics(handle),
    }
    result["report_stream"] = stream_result
    stream_result["subscription_generation"] = {
        "owner": report_generation.owner,
        "lease_id": report_generation.lease_id,
        "generation": report_generation.generation,
        "requested_at_monotonic": report_generation.requested_at_monotonic,
        "baseline_position_sequence": baseline_sequence,
        "baseline_position_epoch": baseline_epoch,
    }
    try:
        if hasattr(coordinator, "async_start_report_stream"):
            await coordinator.async_start_report_stream(duration_ms=stream_duration_ms)
            stream_result["started"] = True
        if hasattr(coordinator, "async_start_continuous_reports"):
            await coordinator.async_start_continuous_reports(
                duration_ms=stream_duration_ms
            )
            stream_result["continuous_started"] = True
        stream_result["queue_settle"] = await _settle_ble_command_queue(coordinator)
        # Same reasoning as `_report_stream_probe`: both start calls return on
        # ENQUEUE, so only the post-settle instant proves the report START
        # reached the transport. Opening a blind-motion window on an origin fix
        # that predates it would reintroduce exactly the stale-origin class the
        # 2026-08-24 remediation closed.
        report_request_started_at = time.monotonic()
        stream_result["subscription_command_flushed_at_monotonic"] = (
            report_request_started_at
        )
    except Exception as err:  # noqa: BLE001
        stream_result["error"] = f"{type(err).__name__}: {err}"
        result["reason"] = "report_stream_failed"
        return result

    fresh_origin = await _wait_for_fresh_continuous_origin(
        position_stream,
        request_started_at=report_request_started_at,
        baseline_sequence=baseline_sequence,
        baseline_epoch=baseline_epoch,
        timeout_s=config.max_heading_acquisition_s,
        handle=handle,
        report_lease=report_lease,
        report_generation=report_generation,
        baseline_dropped_samples=baseline_dropped_samples,
    )
    result["fresh_origin"] = fresh_origin
    if not fresh_origin["ok"]:
        result["reason"] = fresh_origin["reason"]
        result["blockers"] = [fresh_origin["reason"]]
        return result

    origin_sample = fresh_origin["sample"]
    origin_position = origin_sample["position"]
    opening_position = ContinuousPoint(
        float(origin_position["x"]), float(origin_position["y"])
    )
    # Re-run every position-dependent gate against the post-stream origin that
    # will actually seed heading acquisition. Nothing has moved yet.
    refreshed_before = _custom_path_telemetry_snapshot(coordinator)
    refreshed_before["position"] = {
        **(refreshed_before.get("position") or {}),
        **origin_position,
    }
    gates = _continuous_motion_gates(
        coordinator,
        refreshed_before,
        route_start=route_start,
        route_target=route_target,
        config=config,
        corridor_polygon=corridor_polygon,
        dry_run=False,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    result["safety_gates"] = gates
    result["blockers"] = blockers
    result["acquisition"] = next(
        gate["diagnostics"]
        for gate in gates
        if gate["name"] == "blind_heading_acquisition_contained"
    )
    if blockers:
        result["reason"] = "blocked_after_fresh_origin"
        return result

    window_started = time.monotonic()
    result["phase"] = "acquiring_heading"
    # Opening command is straight. Steering remains locked at zero until a
    # fresh >=0.15 m position chord supplies HeadingEvidence.
    command_state = {"linear_speed": linear_speed, "angular_speed": 0}
    refresh_state: dict[str, Any] = {
        "completions_elapsed_ms": [],
        "last_decision_elapsed_ms": 0.0,
    }
    decision_abort = asyncio.Event()
    decision_stop = asyncio.Event()

    started = time.monotonic()
    result["command_result"]["attempted"] = True
    try:
        await _send_manager_command_with_args(
            coordinator,
            "send_movement",
            prefer_ble=prefer_ble,
            command_kwargs={
                "linear_speed": command_state["linear_speed"],
                "angular_speed": command_state["angular_speed"],
            },
        )
        result["command_result"]["ok"] = True
    except Exception as err:  # noqa: BLE001
        result["command_result"]["ok"] = False
        result["command_result"]["error"] = f"{type(err).__name__}: {err}"
        result["reason"] = "command_failed"
        return result
    finally:
        result["command_result"]["duration_ms"] = round(
            (time.monotonic() - started) * 1000, 3
        )
    # Charge dispatch latency to the same clock as motion and record the actual
    # completion instant for refresh-gap accounting.
    refresh_state["completions_elapsed_ms"].append(
        float(result["command_result"]["duration_ms"])
    )

    decision_task = asyncio.create_task(
        _continuous_decision_loop(
            position_stream,
            route=route,
            corridor_polygon=corridor_polygon,
            config=config,
            opening_position=opening_position,
            opening_sequence=origin_sample["sequence"],
            opening_epoch=origin_sample["epoch"],
            window_started=window_started,
            sample_interval_ms=decision_sample_interval_ms,
            refresh_state=refresh_state,
            command_state=command_state,
            decision_abort=decision_abort,
            stop_event=decision_stop,
            acquisition_only=acquisition_only,
        )
    )

    # 🚨 A DEAD DECISION LOOP MUST NOT LEAVE THE MOWER DRIVING ON THE CLOCK.
    # Same reasoning as beta72's `_abort_if_sampler_died`: if the task that
    # owns steering and stop decisions dies, the refresh loop must stop
    # refreshing rather than silently keep resending the last command it saw.
    def _abort_if_decision_loop_died(task: asyncio.Task[Any]) -> None:
        if task.cancelled() or task.exception() is not None:
            decision_abort.set()

    decision_task.add_done_callback(_abort_if_decision_loop_died)

    try:
        result["motion_refresh"] = await _continuous_refresh_window(
            coordinator,
            command_state=command_state,
            prefer_ble=prefer_ble,
            duration_seconds=duration_ms / 1000.0,
            refresh_interval_ms=motion_refresh_interval_ms,
            window_started=window_started,
            refresh_state=refresh_state,
            abort_event=decision_abort,
        )
    except BaseException:
        decision_stop.set()
        with contextlib.suppress(BaseException):
            await decision_task
        raise
    decision_stop.set()
    result["stop_result"] = await _manual_velocity_stop_attempt(
        coordinator, use_wifi=not prefer_ble
    )
    result["decisions"] = await decision_task
    result["phase"] = "stopping"
    result["would_send"] = True
    if acquisition_only:
        final_position = opening_position
        final_sequence = origin_sample["sequence"]
        if result["decisions"]:
            final_observation = result["decisions"][-1]["observation"]
            final_position = ContinuousPoint(**final_observation["position"])
            final_sequence = result["decisions"][-1]["position_sequence"]
        post_stop_sample, post_stop_reason = await _wait_for_post_stop_position(
            position_stream,
            after_sequence=final_sequence,
            epoch=origin_sample["epoch"],
            timeout_s=_CONTINUOUS_POST_STOP_OBSERVATION_S,
        )
        if post_stop_sample is not None:
            final_position = ContinuousPoint(
                float(post_stop_sample.x), float(post_stop_sample.y)
            )
            final_sequence = post_stop_sample.sequence
        final_heading = course_from_position_chord(
            opening_position,
            final_position,
            measured_at_s=time.monotonic() - window_started,
            min_chord_m=config.min_travel_for_heading_trust_m,
        )
        result["post_stop_position"] = {
            "sequence": final_sequence,
            "epoch": origin_sample["epoch"],
            "x": final_position.x,
            "y": final_position.y,
            "wait_reason": post_stop_reason,
        }
        result["heading_state"] = {
            "phase": "acquired" if final_heading is not None else "unconfirmed",
            "source": (final_heading.source if final_heading is not None else None),
            "minimum_chord_m": config.min_travel_for_heading_trust_m,
            "maximum_age_s": config.max_heading_age_s,
            # `maximum_age_s` is the steering freshness bound and is NOT applied
            # here: `course_from_position_chord` records `measured_at_s` without
            # enforcing it, and the stopped observation can add up to
            # `_CONTINUOUS_POST_STOP_OBSERVATION_S` on top of the motion window.
            # This evidence is diagnostic only and must never be read as a
            # freshness-validated heading -- it cannot steer anything, because
            # the invocation stops before it is computed.
            "age_bounded_by_maximum_age_s": False,
            "includes_post_stop_observation": post_stop_sample is not None,
            "post_stop_observation_timeout_s": _CONTINUOUS_POST_STOP_OBSERVATION_S,
            "evidence": (
                dataclasses.asdict(final_heading) if final_heading is not None else None
            ),
        }
    result["reason"] = (
        result["decisions"][-1]["decision"]["reason"]
        if result["decisions"]
        else "completed"
    )
    return result


async def _continuous_motion_window(  # noqa: PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    *,
    route_start: dict[str, float],
    route_target: dict[str, float],
    corridor_polygon: list[dict[str, float]],
    linear_speed: int = 400,
    max_abs_angular_speed: int = 180,
    duration_ms: int = 4000,
    motion_refresh_interval_ms: int = 200,
    decision_sample_interval_ms: int = 100,
    max_distance_m: float = 1.50,
    max_cross_track_m: float = 0.30,
    prefer_ble: bool = True,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    confirm_steering_validation_run: bool = False,
) -> dict[str, Any]:
    """Own and close the safety-consumer position stream for one window.

    ⚠️ **This wrapper passed `position_stream=None` and no report lease until
    2026-08-27**, which was harmless only because the executor refused all real
    steering before it ever reached them. With the refusal now conditional, a
    real run needs the same lease-and-stream ownership
    `_heading_acquisition_window` already had -- otherwise it fails on
    `position_stream_unavailable` and reads as a telemetry fault rather than a
    wiring gap.
    """
    common: dict[str, Any] = {
        "acquisition_only": False,
        "route_start": route_start,
        "route_target": route_target,
        "corridor_polygon": corridor_polygon,
        "linear_speed": linear_speed,
        "max_abs_angular_speed": max_abs_angular_speed,
        "duration_ms": duration_ms,
        "motion_refresh_interval_ms": motion_refresh_interval_ms,
        "decision_sample_interval_ms": decision_sample_interval_ms,
        "max_distance_m": max_distance_m,
        "max_cross_track_m": max_cross_track_m,
        "prefer_ble": prefer_ble,
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "confirm_steering_validation_run": confirm_steering_validation_run,
    }
    if dry_run:
        return await _continuous_motion_window_impl(
            coordinator,
            position_stream=None,
            report_lease=None,
            dry_run=True,
            **common,
        )

    handle = coordinator.manager.mower(coordinator.device_name)
    exclusive_factory = getattr(handle, "exclusive_report_subscription", None)
    open_position_stream = getattr(coordinator, "open_position_sample_stream", None)
    if not callable(exclusive_factory) or not callable(open_position_stream):
        return {
            "service": SERVICE_CONTINUOUS_MOTION_WINDOW,
            "mode": "real_continuous_motion_window",
            "dry_run": False,
            "would_send": False,
            "blockers": ["position_subscription_lease_unavailable"],
            "reason": "position_subscription_lease_unavailable",
        }

    result: dict[str, Any] | None = None
    async with exclusive_factory("continuous_motion_window") as report_lease:
        position_stream = open_position_stream(maxsize=_SAFETY_POSITION_STREAM_MAXSIZE)
        try:
            result = await _continuous_motion_window_impl(
                coordinator,
                position_stream=position_stream,
                report_lease=report_lease,
                dry_run=False,
                **common,
            )
            return result  # noqa: RET504 - finally annotates report-stream teardown
        finally:
            # The lease owns the report configuration; always issue its stop,
            # including when the executor raises before it can return a result.
            stop_reports = getattr(coordinator, "async_stop_continuous_reports", None)
            if callable(stop_reports):
                try:
                    await stop_reports()
                    if result is not None:
                        result.setdefault("report_stream", {})["stopped"] = True
                except Exception as err:  # noqa: BLE001
                    if result is not None:
                        result.setdefault("report_stream", {})["stop_error"] = (
                            f"{type(err).__name__}: {err}"
                        )
            # `open_position_sample_stream` may legitimately return None. The
            # executor already refuses that with `position_stream_unavailable`;
            # teardown must not crash on it and mask the real reason.
            if position_stream is not None:
                position_stream.close()


async def _heading_acquisition_window(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    route_start: dict[str, float],
    route_target: dict[str, float],
    corridor_polygon: list[dict[str, float]],
    prefer_ble: bool = True,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
) -> dict[str, Any]:
    """Acquire one position-chord heading without ever dispatching steering."""
    if dry_run:
        return await _continuous_motion_window_impl(
            coordinator,
            position_stream=None,
            report_lease=None,
            acquisition_only=True,
            route_start=route_start,
            route_target=route_target,
            corridor_polygon=corridor_polygon,
            linear_speed=400,
            max_abs_angular_speed=180,
            duration_ms=2000,
            motion_refresh_interval_ms=200,
            decision_sample_interval_ms=100,
            max_distance_m=1.0,
            max_cross_track_m=0.30,
            prefer_ble=prefer_ble,
            dry_run=True,
            confirm_blades_off=confirm_blades_off,
            confirm_clear_area=confirm_clear_area,
        )

    handle = coordinator.manager.mower(coordinator.device_name)
    exclusive_factory = getattr(handle, "exclusive_report_subscription", None)
    open_position_stream = getattr(coordinator, "open_position_sample_stream", None)
    if not callable(exclusive_factory) or not callable(open_position_stream):
        return {
            "service": SERVICE_HEADING_ACQUISITION_WINDOW,
            "mode": "real_heading_acquisition_window",
            "dry_run": False,
            "would_send": False,
            "blockers": ["position_subscription_lease_unavailable"],
            "reason": "position_subscription_lease_unavailable",
        }

    result: dict[str, Any] | None = None
    async with exclusive_factory("heading_acquisition_window") as report_lease:
        position_stream = open_position_stream(maxsize=_SAFETY_POSITION_STREAM_MAXSIZE)
        try:
            result = await _continuous_motion_window_impl(
                coordinator,
                position_stream=position_stream,
                report_lease=report_lease,
                acquisition_only=True,
                route_start=route_start,
                route_target=route_target,
                corridor_polygon=corridor_polygon,
                linear_speed=400,
                max_abs_angular_speed=180,
                duration_ms=2000,
                motion_refresh_interval_ms=200,
                decision_sample_interval_ms=100,
                max_distance_m=1.0,
                max_cross_track_m=0.30,
                prefer_ble=prefer_ble,
                dry_run=False,
                confirm_blades_off=confirm_blades_off,
                confirm_clear_area=confirm_clear_area,
            )
            return result  # noqa: RET504 - finally annotates report-stream teardown
        finally:
            # The lease owns the report configuration. Always issue its stop,
            # including when the executor raises before it can return a result;
            # otherwise an exception could leave a standing subscription behind.
            stop_reports = getattr(coordinator, "async_stop_continuous_reports", None)
            if callable(stop_reports):
                try:
                    await stop_reports()
                    if result is not None:
                        result.setdefault("report_stream", {})["stopped"] = True
                except Exception as err:  # noqa: BLE001
                    if result is not None:
                        result.setdefault("report_stream", {})["stop_error"] = (
                            f"{type(err).__name__}: {err}"
                        )
            position_stream.close()


# ============================================================================
# Open-loop step-response probe -- answers Q1 (is the dead time in the ACTUATOR
# or the OBSERVER?) and Q2 (how large is it?) from
# `docs/phase2-dead-time-step-test-design-20260828.md`.
#
# 🔑 WHY THIS EXISTS AS A SEPARATE SERVICE. `_motion_refresh_window`'s contract
# is explicitly to resend an IDENTICAL command, so it cannot express a step.
# `_continuous_refresh_window` CAN resend a changing one, but its only other
# caller is `continuous_motion_window` -- the closed-loop steering service that
# standing decision 5 parks. Assembling the step out of two bounded probe
# windows does not work either: the stop between them resets exactly the
# rotational carryover being measured.
#
# 🔑 THIS PROBE HAS NO CONTROLLER. No route, no aim point, no steering law, no
# corridor-breach override, no heading state machine. It commands a fixed
# sequence and records. That is what makes it cheaper to reason about than
# either steering attempt, and it is deliberate: the question is about the
# PLANT, so nothing in the loop should be able to influence the answer.
# ============================================================================

# Sized from the design document, not from convenience. The probe drives an
# OPEN LOOP curved path and nothing steers it back, so the only honest
# containment is a disk around the frozen start large enough for every path the
# command sequence can produce, plus the banked stop overshoot -- the same shape
# as `blind_acquisition_feasibility`.
# ⚠️ `stop_overshoot_m` is 0.50 m and attempt 5 measured 0.4544 m of post-stop
# creep (docs/evidence-phase2-steering-attempt5-20260828.json) -- a 9% margin.
# Budget the full constant and do not raise `linear_speed`.
_STEP_RESPONSE_MAX_TOTAL_MS = 23000
# LOWER-bound m/s for each admissible step-probe `linear_speed`. ⚠️ Rounded DOWN
# on purpose, and that direction is the whole point: this table refuses only a
# window that overruns `max_travel_m` even at the slowest speed the mower has
# ever shown, so a merely MARGINAL config still dispatches and only an
# impossible one is refused. Sizing a corridor needs the opposite rounding --
# use `_PROBE_SPEED_PER_LINEAR_UNIT_MS` for that, never this.
#
# 🚨 CORRECTED 2026-09-01, hours after this table was introduced. The 400 entry
# shipped as 0.24, taken from the single FASTEST banked run, while calling itself
# the slowest -- so it was roughly the population maximum wearing a lower-bound
# label. Cumulative PATH travel (the guard's own metric) over the full window of
# every banked linear-400 run:
#     0.1936  0.2000  0.2121  0.2124  0.2616 m/s   (min 0.1936, mean 0.2159)
# 0.24 sat ABOVE FOUR OF THE FIVE. The consequence was over-refusal, not danger:
# a replay of banked route-1 run 1 (3000/5000/5000 at max_travel_m 3.0) was
# refused pre-dispatch although both banked runs of it travelled 2.71 and 2.77 m.
# 400 is now 0.17, below the measured minimum with margin.
#
# 🗑️ The command/speed relation IS essentially linear -- do not reintroduce the
# old "a 25% command cut gives a 39% speed cut". That came from comparing 4 s
# ramp-INCLUSIVE averages on 2026-08-30 (400 -> 0.191, 300 -> 0.116 m/s), where
# the slower run spends a larger fraction of its window ramping. Sustained
# speeds measured 2026-09-03 scale with the command to within 1%; see
# `_PROBE_SPEED_PER_LINEAR_UNIT_MS`.
# ⚠️ The entries BELOW are still not a linear function of the command, and that
# is correct: they are whole-window floors, and a window's ramp fraction depends
# on its length, not only on its speed.
# ⚠️ These stay LOW on purpose. They are the floor a WHOLE window averages,
# including ramp-up from standstill, so they must sit under the shortest windows
# too: Phase A measured 0.157 m/s whole-window at 300 over 8 s against 0.223
# sustained, and a 3 s window averages far less again.
_STEP_RESPONSE_MIN_SPEED_BY_LINEAR: dict[int, float] = {300: 0.10, 400: 0.17}
# SUSTAINED (post-ramp) m/s, for a non-blocking projection only. A true lower
# bound refuses only the impossible, so it cannot flag a window that is merely
# LIKELY to trip the guard; the probe reports both numbers and lets the operator
# see the risk.
#
# 🚨 CORRECTED 2026-09-03 from ramp-INCLUSIVE window averages to SUSTAINED speeds,
# measured directly: 300 -> 0.223 (Phase A), 400 -> 0.295 (2026-09-01 run, and
# independently 0.280-0.293 during the 2026-08-12 arcs). The old {300: 0.16,
# 400: 0.216} understated a long window badly -- 0.16 at 300 was 39% low, and
# sizing a 28 s Phase B window with it predicted 4.5 m where the measured figure
# gives 5.9 m. Using sustained speed with no ramp credit slightly OVER-states
# travel on short windows; that is the safe direction for a warning.
_STEP_RESPONSE_TYPICAL_SPEED_BY_LINEAR: dict[int, float] = {300: 0.223, 400: 0.295}
# Matches the readiness budget the beta77 stationary work derived: the maximum
# healthy stationary publication interval measured 2910.1 ms across n=1434, so
# 3.5 s carries a 1.20x margin. It is a conservative stationary default, never
# a proven distribution and never motion validation.
_STEP_RESPONSE_READINESS_TIMEOUT_S = 3.5
_STEP_RESPONSE_DEFAULT_TRAVEL_M = 2.50
# Criterion 2a/2b agreement bound and interval floor, unchanged from
# docs/phase2-route1-predeclared-20260830.md §5. What changed on 2026-09-01 is
# the SIGNAL and the STATISTIC for 2a, not the numbers: scored from VIO heading
# via half-phase mean-rate agreement (rule E-VIO), adopted per
# docs/findings-rtk-vio-course-rate-scoring-20260831.md. The RTK chord rule's
# last-two-diff statistic carries ~2.7 deg/s of 1-sigma position noise against
# this 1.5 deg/s bound, so its verdicts were draws; VIO resolves the bound at
# ~11 sigma with this statistic.
_STEP_RESPONSE_RATE_AGREEMENT_BOUND_DEG_PER_S = 1.5
_STEP_RESPONSE_MIN_INFORMATIVE_INTERVALS = 3
# `vio_state == 2` is the only value ever observed while VIO was corroborated
# live (state: 2 across all 549 banked route-1 samples). Anything else refuses
# to score — never fall back silently to the noise-bound RTK chord rule.
_STEP_RESPONSE_VIO_STATE_LIVE = 2
# 🚨 A PLAUSIBILITY CEILING on |VIO heading rate| between consecutive distinct
# readings. Any interval above it, in ANY phase, refuses the whole run with
# `vio_heading_discontinuity`. Predeclared in
# docs/predeclared-vio-heading-continuity-guard-20260903.md.
#
# Why it exists: `vio_state` checks LIVENESS, not CONTINUITY. On 2026-09-03 a
# run returned `scoreable: true` while the mower drove straight, because VIO
# heading jumped -166.47 deg in one report (`toward` jumped +69.12 deg at the
# same instant -- different magnitudes, so no rotation explains it) while
# `vio_state` stayed 2 throughout. That jump was harmless only because the
# 1000 ms step yielded 1 interval against the rule's >=3; at the 5000-7000 ms
# steps the programme actually uses, the same jump lands INSIDE the half-phase
# statistic and becomes a certified rotation rate.
#
# 🗑️ NOT the commanded angular rate, which is the shape that suggests itself and
# is WRONG: commanded angular is ZERO in baseline and settle, yet rotation
# persists past the command going to zero -- that decay is the whole subject of
# the step-response programme (17 deg after zero, 2026-08-29). Measured across
# the banked corpus, a command-scaled bound refuses every clean run on its
# settle intervals alone (worst clean settle interval: 9.97 deg/s). The bound
# must be the PLANT's envelope, not the instantaneous command.
#
# Derivation, measurement only:
#     fastest steady rotation, any admissible command  13.431 deg/s  (300, 180)
#     fastest single clean interval, 8 banked runs     15.35  deg/s
#     the observed discontinuity                      149.79  deg/s
# 30.0 is ~2.0x the worst clean interval and 5.0x below the discontinuity. The
# ~10x clean/broken separation makes the value uncritical -- anything in roughly
# 20-70 deg/s scores the entire banked corpus identically -- so it is placed
# just above the physical envelope with a doubling of margin rather than tuned.
#
# ⚠️ NOT a claim about the plant. Do not quote it as a rotation capability and
# do not fit anything to it.
# 🚨 It is valid ONLY for this probe's admissible commands (linear 300/400,
# |angular| 120/180, always driving forward). Stationary pivots reach ~38 deg/s
# at angular 500 and would trip it. If the schema ever admits those, RE-DERIVE
# this first: `test_the_continuity_bound_is_tied_to_the_admissible_commands`
# fails when the schema widens without it.
_STEP_RESPONSE_VIO_MAX_PLAUSIBLE_RATE_DEG_PER_S = 30.0


def _step_response_gates(
    coordinator: MammotionReportUpdateCoordinator,
    before: dict[str, Any],
    *,
    route_start: dict[str, float],
    corridor_polygon: list[dict[str, float]],
    max_travel_m: float,
    linear_speed: int = 400,
    total_ms: int = 0,
    dry_run: bool,
    confirm_blades_off: bool,
    confirm_clear_area: bool,
) -> list[dict[str, Any]]:
    """Return the safety gates for one open-loop step-response window."""
    gates = list(
        _manual_velocity_pulse_gates(
            coordinator,
            before,
            dry_run=dry_run,
            confirm_blades_off=confirm_blades_off,
            confirm_clear_area=confirm_clear_area,
        )
    )
    corridor_points = [ContinuousPoint(**point) for point in corridor_polygon]
    corridor_valid = polygon_is_valid(corridor_points)
    position = before.get("position", {}) or {}
    live_x, live_y = position.get("x"), position.get("y")
    drift = (
        math.hypot(live_x - route_start["x"], live_y - route_start["y"])
        if isinstance(live_x, (int, float)) and isinstance(live_y, (int, float))
        else None
    )
    gates.append(
        {
            "name": "corridor_polygon_valid",
            "passed": corridor_valid,
            "detail": "The frozen corridor must be a real polygon, >= 3 vertices.",
        }
    )
    gates.append(
        {
            "name": "frozen_start_inside_corridor",
            "passed": corridor_valid
            and _point_in_polygon(route_start, corridor_polygon),
            "detail": "The frozen start must be inside the frozen corridor.",
        }
    )
    gates.append(
        {
            "name": "start_drift_within_bound",
            "passed": dry_run
            or (drift is not None and drift <= _CONTINUOUS_MAX_START_DRIFT_M),
            "detail": (
                f"Live position must be within {_CONTINUOUS_MAX_START_DRIFT_M} m "
                "of the frozen start. The start is never re-derived from live "
                "position -- this gate aborts instead."
            ),
            "diagnostics": {"drift_m": drift},
        }
    )
    # The open-loop path is curved and its exact shape is the UNKNOWN this probe
    # exists to measure, so no direction may be assumed: require the whole disk.
    # Computed directly rather than through `blind_acquisition_feasibility` --
    # that helper derives its radius from the heading-acquisition budget, and
    # bending this probe's budget into that shape would make the number look
    # like an acquisition disk when it is not one.
    stop_overshoot_m = ContinuousControllerConfig().stop_overshoot_m
    # 🚨 WORST CASE, not the nominal one. `max_travel_m + overshoot` assumes the
    # distance guard WORKS -- and this project has a documented mode where it
    # silently does not: position payloads keep arriving with an advancing
    # sequence and a fresh timestamp while x/y stay latched (2026-08-28: 21
    # bit-identical samples while the mower travelled 0.4375 m; attempt 3:
    # 0.5097 m observed as 0.021 m). In that mode `cumulative_distance_m` stays
    # ~0, nothing trips, and the window runs to the WALL CLOCK.
    #
    # `raw_pymammotion_motion_probe` was corrected for exactly this on
    # 2026-08-23 (`corridor_must_cover_m`); this probe was missed, and it is the
    # one whose window length keeps being raised. At the 23000 ms cap and linear
    # 400 the clock bound is 6.44 m against a 5.00 m requirement -- a ~1.4 m
    # breach of a corridor the operator was told holds the path.
    # ⚠️ The stop overshoot applies to BOTH branches. The mandatory stop is
    # issued after the window ENDS, so the post-stop creep (measured 0.4544 m,
    # attempt 5) sits outside whichever bound was reached -- including the clock
    # bound, which exists precisely for the case where the guard no-ops and the
    # window runs to the wall clock. Omitting it there left the corridor
    # uncertified for the creep, protected only by unmodelled ramp and curvature
    # margin this codebase elsewhere declines to credit. The sibling disk helper
    # `continuous_controller.blind_acquisition_feasibility` has always added it.
    clock_bound_m = (
        _PROBE_SPEED_PER_LINEAR_UNIT_MS * abs(int(linear_speed)) * (total_ms / 1000.0)
        + stop_overshoot_m
        if total_ms
        else 0.0
    )
    required = round(max(max_travel_m + stop_overshoot_m, clock_bound_m), 6)
    live_point = (
        ContinuousPoint(float(live_x), float(live_y))
        if isinstance(live_x, (int, float)) and isinstance(live_y, (int, float))
        else None
    )
    clearance = (
        polygon_boundary_clearance(live_point, corridor_points)
        if live_point is not None and corridor_valid
        else None
    )
    live_inside = (
        point_in_polygon(live_point, corridor_points)
        if live_point is not None and corridor_valid
        else False
    )
    gates.append(
        {
            "name": "step_path_contained",
            "passed": bool(
                live_inside and clearance is not None and clearance >= required
            ),
            "detail": (
                "The probe steers open loop on a curved path whose shape is the "
                "unknown under test, so EVERY possible ray must fit the frozen "
                f"corridor: {max_travel_m} m of commanded travel plus "
                f"{stop_overshoot_m} m stopping/guard overshoot = {required} m."
            ),
            "diagnostics": {
                "required_radius_m": required,
                "travel_budget_bound_m": round(max_travel_m + stop_overshoot_m, 6),
                "clock_bound_m": round(clock_bound_m, 6),
                "bound_that_binds": (
                    "clock"
                    if clock_bound_m > max_travel_m + stop_overshoot_m
                    else "travel_budget"
                ),
                "boundary_clearance_m": clearance,
                "live_position_inside": live_inside,
                "commanded_travel_m": max_travel_m,
                "stop_overshoot_m": stop_overshoot_m,
            },
        }
    )
    return gates


async def _step_response_phase_scheduler(
    command_state: dict[str, int],
    *,
    window_started: float,
    baseline_ms: int,
    step_ms: int,
    step_angular_speed: int,
    abort_event: asyncio.Event,
) -> list[dict[str, Any]]:
    """Flip the shared command through baseline -> step -> settle on the clock.

    Writes only `command_state`, which `_continuous_refresh_window` re-reads on
    every 200 ms write; it never touches BLE itself. That is the same
    one-serialized-writer handoff `_continuous_decision_loop` relies on, and the
    reason a phase change reaches the wire within one refresh interval -- proven
    on attempt 5, whose wire log shows angular changing within ~200 ms of each
    decision.
    """
    transitions: list[dict[str, Any]] = []

    async def _hold_until(elapsed_ms: int) -> bool:
        while True:
            remaining = window_started + elapsed_ms / 1000.0 - time.monotonic()
            if remaining <= 0:
                return not abort_event.is_set()
            if abort_event.is_set():
                return False
            await asyncio.sleep(min(remaining, 0.05))

    def _record(phase: str, angular: int) -> None:
        command_state["angular_speed"] = angular
        transitions.append(
            {
                "phase": phase,
                "angular_speed": angular,
                "elapsed_ms": round((time.monotonic() - window_started) * 1000, 3),
            }
        )

    _record("baseline", 0)
    if await _hold_until(baseline_ms):
        _record("step", step_angular_speed)
    if await _hold_until(baseline_ms + step_ms):
        # 🔑 THE WHOLE EXPERIMENT IS THIS INSTANT. Everything the mower rotates
        # after it is carryover, and its integral divided by the steady rate
        # measured during `step` is the actuator dead time.
        _record("settle", 0)
    return transitions


def _step_response_course_series(
    samples: list[dict[str, Any]],
    *,
    baseline_ms: int,
    step_ms: int,
    min_chord_m: float,
) -> list[dict[str, Any]]:
    """Turn 100 ms cache samples into per-interval chord courses.

    ⚠️ **Emitted whole, and every headline number below is derived from it.**
    This project's standing rule is to verify with per-item records rather than
    aggregates (a net figure once hid a 27 degree turn reversal), so the raw
    series is the deliverable and `_step_response_analysis` is a convenience.

    The 100 ms sampler reads a cache that only refreshes at ~1 Hz. That is not a
    defect here: what it buys is the ARRIVAL INSTANT of each new position to
    within 100 ms, which is what bounds when rotation started and stopped.
    """
    distinct: list[tuple[float, float, float]] = []
    for sample in samples:
        position = sample.get("position") or {}
        x, y = position.get("x"), position.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        elapsed_ms = sample.get("elapsed_ms")
        if not isinstance(elapsed_ms, (int, float)):
            continue
        if distinct and distinct[-1][1] == float(x) and distinct[-1][2] == float(y):
            continue
        distinct.append((float(elapsed_ms), float(x), float(y)))

    series: list[dict[str, Any]] = []
    for (t0, x0, y0), (t1, x1, y1) in zip(distinct, distinct[1:], strict=False):
        chord = math.hypot(x1 - x0, y1 - y0)
        midpoint_ms = (t0 + t1) / 2
        phase = (
            "baseline"
            if midpoint_ms < baseline_ms
            else "step"
            if midpoint_ms < baseline_ms + step_ms
            else "settle"
        )
        series.append(
            {
                "from_elapsed_ms": t0,
                "to_elapsed_ms": t1,
                # A chord is an interval AVERAGE, so its course describes the
                # midpoint, not either endpoint. Recording that explicitly is
                # what lets observer lag be separated from actuator lag offline.
                "midpoint_elapsed_ms": round(midpoint_ms, 3),
                "phase": phase,
                "chord_m": round(chord, 6),
                # Below the informativeness floor a chord's bearing is noise:
                # at sigma = 0.0031 m a 0.076 m chord carries +/-7.4 degrees.
                "informative": chord >= min_chord_m,
                "course_degrees": (
                    round(math.degrees(math.atan2(y1 - y0, x1 - x0)), 4)
                    if chord >= min_chord_m
                    else None
                ),
            }
        )
    return series


def _step_response_analysis(
    series: list[dict[str, Any]], *, baseline_ms: int, step_ms: int
) -> dict[str, Any]:
    """Derive the headline dead-time numbers from the per-interval series.

    🔑 Measures the INTEGRAL, not the rate. Total angle accumulated after the
    command goes to zero is a difference of two absolute headings, so it is
    accurate at ANY sample rate -- which is what makes a ~1 s lag measurable on
    a ~1 Hz feed. Per-sample rates are not, and must not be fitted here.
    """
    informative = [row for row in series if row["informative"]]
    step_rows = [row for row in informative if row["phase"] == "step"]
    settle_rows = [row for row in informative if row["phase"] == "settle"]

    def _rate(rows: list[dict[str, Any]]) -> float | None:
        if len(rows) < 2:
            return None
        span_s = (
            rows[-1]["midpoint_elapsed_ms"] - rows[0]["midpoint_elapsed_ms"]
        ) / 1000
        if span_s <= 0:
            return None
        # normalize_degrees keeps the signed short way round, so a rotation
        # through the +/-180 wrap is not read as a ~360 degree jump.
        delta = normalize_degrees(
            rows[-1]["course_degrees"] - rows[0]["course_degrees"]
        )
        return round(delta / span_s, 4)

    omega_step = _rate(step_rows)
    rotation_after_zero = (
        round(
            normalize_degrees(
                settle_rows[-1]["course_degrees"] - step_rows[-1]["course_degrees"]
            ),
            4,
        )
        if step_rows and settle_rows
        else None
    )
    tau: float | None = None
    if rotation_after_zero is not None and omega_step is not None and omega_step != 0:
        tau = round(abs(rotation_after_zero / omega_step), 4)
    return {
        "informative_intervals": {
            "baseline": sum(1 for r in informative if r["phase"] == "baseline"),
            "step": len(step_rows),
            "settle": len(settle_rows),
        },
        "omega_step_deg_per_s": omega_step,
        "rotation_after_zero_deg": rotation_after_zero,
        "tau_actuator_s": tau,
        # Stated so the number is never read as more than it is.
        "interpretation": (
            "tau_actuator_s = |rotation after commanding angular 0| / steady rate "
            "during the step. Compare it against the ~1 s decision period: well "
            "below means the dead time is dominated by the OBSERVER and damping "
            "cannot fix it; comparable or above means real actuator carryover."
        ),
        "caveats": (
            "n is small by construction at ~1 Hz. Do NOT fit a turn-rate law to "
            "this, and do NOT reuse omega as a calibration constant -- the "
            "2026-08-26 guarded-turn measurement showed a 2.6x spread on "
            "identical stationary parameters."
        ),
    }


def _step_response_vio_intervals(
    samples: list[dict[str, Any]], *, baseline_ms: int, step_ms: int
) -> list[dict[str, Any]]:
    """Per-interval VIO heading rates between consecutive DISTINCT readings.

    VIO latches, holding one value across several 100 ms samples, so naive
    per-sample differencing yields spurious zeros. A reading exists at the
    first sample whose heading differs from the previous distinct value; its
    timestamp is that sample's ``elapsed_ms``. Phase assignment matches the
    RTK series: interval midpoint against the nominal boundaries.
    """
    track: list[tuple[float, float]] = []
    for sample in samples:
        vio = sample.get("vio") or {}
        heading = vio.get("heading")
        elapsed_ms = sample.get("elapsed_ms")
        if not isinstance(heading, (int, float)) or not isinstance(
            elapsed_ms, (int, float)
        ):
            continue
        if track and track[-1][1] == float(heading):
            continue
        track.append((float(elapsed_ms), float(heading)))

    intervals: list[dict[str, Any]] = []
    for (t0, h0), (t1, h1) in zip(track, track[1:], strict=False):
        dt_s = (t1 - t0) / 1000
        if dt_s <= 0:
            continue
        midpoint_ms = (t0 + t1) / 2
        phase = (
            "baseline"
            if midpoint_ms < baseline_ms
            else "step"
            if midpoint_ms < baseline_ms + step_ms
            else "settle"
        )
        intervals.append(
            {
                "from_elapsed_ms": t0,
                "to_elapsed_ms": t1,
                "midpoint_elapsed_ms": round(midpoint_ms, 3),
                "phase": phase,
                "from_heading_degrees": h0,
                "to_heading_degrees": h1,
                "rate_deg_per_s": round(normalize_degrees(h1 - h0) / dt_s, 4),
            }
        )
    return intervals


def _step_response_half_phase_agreement(
    intervals: list[dict[str, Any]], phase: str
) -> dict[str, Any]:
    """Rule E: mean rate of the phase's first half vs its second half.

    Endpoint differences over each half average the per-reading noise down
    while still catching a ramp -- a still-accelerating phase has a faster
    second half. The last-two-diff statistic cannot do this on a low-noise
    channel: a smooth ramp has adjacent rates nearly equal long before it
    converges, which is exactly how VIO last-two wrongly passed run 1
    (docs/findings-rtk-vio-course-rate-scoring-20260831.md §3). Odd reading
    counts put the extra reading in the first half; the boundary reading ends
    the first half and starts the second.
    """
    seq = [iv for iv in intervals if iv["phase"] == phase]
    verdict: dict[str, Any] = {
        "informative_intervals": len(seq),
        "half_rates_deg_per_s": None,
        "half_diff_deg_per_s": None,
        "passed": False,
    }
    if len(seq) < _STEP_RESPONSE_MIN_INFORMATIVE_INTERVALS:
        return verdict
    readings: list[tuple[float, float]] = [
        (seq[0]["from_elapsed_ms"], seq[0]["from_heading_degrees"])
    ]
    readings.extend((iv["to_elapsed_ms"], iv["to_heading_degrees"]) for iv in seq)
    boundary = len(readings) // 2
    half_rates: list[float] = []
    for half in (readings[: boundary + 1], readings[boundary:]):
        (t0, a0), (t1, a1) = half[0], half[-1]
        dt_s = (t1 - t0) / 1000
        if dt_s <= 0:
            return verdict
        half_rates.append(round(normalize_degrees(a1 - a0) / dt_s, 4))
    diff = round(abs(half_rates[1] - half_rates[0]), 4)
    verdict["half_rates_deg_per_s"] = half_rates
    verdict["half_diff_deg_per_s"] = diff
    verdict["passed"] = diff <= _STEP_RESPONSE_RATE_AGREEMENT_BOUND_DEG_PER_S
    return verdict


def _step_response_vio_analysis(  # noqa: C901
    samples: list[dict[str, Any]], *, baseline_ms: int, step_ms: int
) -> dict[str, Any]:
    """Score criteria 2a/2b from VIO heading (rule E-VIO), failing closed.

    Adopted 2026-09-01 per docs/findings-rtk-vio-course-rate-scoring-20260831.md:
    2a is half-phase mean-rate agreement over the step; 2b keeps its published
    last-two-rates semantics ("goes flat at the END") but on the VIO channel --
    half-phase agreement cannot apply to settle, whose first half contains the
    decay transient by construction. The RTK ``course_series``/``analysis``
    stay emitted unchanged as the cross-check diagnostic; they are no longer
    the 2a instrument.

    🔑 omega and tau come from the SAME channel that certifies steadiness, and
    tau exists only when 2a passes -- a ramp-sampled omega is the exact failure
    2a exists to prevent. Dark or degraded VIO (any sample with
    ``vio_state != 2``) refuses to score rather than silently falling back to
    the noise-bound RTK chord rule; RTK chords remain the only night-capable
    course source, and a night run is UNSCOREABLE under this rule on purpose.
    """
    if not samples:
        return {"scoreable": False, "unscoreable_reason": "no_samples"}
    states = [(sample.get("vio") or {}).get("state") for sample in samples]
    if any(state != _STEP_RESPONSE_VIO_STATE_LIVE for state in states):
        return {
            "scoreable": False,
            "unscoreable_reason": "vio_not_live_throughout",
            "vio_states_observed": sorted({str(state) for state in states}),
        }

    intervals = _step_response_vio_intervals(
        samples, baseline_ms=baseline_ms, step_ms=step_ms
    )

    # 🚨 CONTINUITY, which `vio_state` does not check. A heading-frame jump
    # (VIO re-referencing after a restart) presents as a live channel reporting
    # an impossible rate, and the half-phase statistic converts it into a
    # certified rotation. Refuse the RUN, never just the interval: after a frame
    # jump every later heading is referenced to a shifted origin, so no
    # statistic over the track is meaningful -- and dropping the offending
    # interval and re-scoring the rest is the "just exclude the onset interval"
    # move rejected on 2026-09-01, choosing samples after seeing the verdicts.
    discontinuities = [
        interval
        for interval in intervals
        if abs(interval["rate_deg_per_s"])
        > _STEP_RESPONSE_VIO_MAX_PLAUSIBLE_RATE_DEG_PER_S
    ]
    if discontinuities:
        return {
            "scoreable": False,
            "unscoreable_reason": "vio_heading_discontinuity",
            "max_plausible_rate_deg_per_s": (
                _STEP_RESPONSE_VIO_MAX_PLAUSIBLE_RATE_DEG_PER_S
            ),
            "discontinuities": discontinuities,
            "interpretation": (
                "VIO heading moved faster between two consecutive distinct "
                "readings than the mower can rotate under any command this "
                "probe admits, so the heading track is not one continuous "
                "frame and no rate derived from it is a measurement. "
                "vio_state checks liveness, not continuity, so this can occur "
                "with vio_state 2 throughout. Cross-check the RTK "
                "course_series: if it shows no matching rotation, the VIO "
                "frame jumped (a mower restart re-references it). The run is "
                "unscoreable; it cannot be repaired by dropping the interval."
            ),
        }

    if len(intervals) < 2:
        return {
            "scoreable": False,
            "unscoreable_reason": "vio_track_insufficient",
            "distinct_reading_intervals": len(intervals),
        }

    step_2a = _step_response_half_phase_agreement(intervals, "step")

    # 2b: last-two settle rates within the bound, with the carryover pair
    # (the prior phase's final interval prepended) -- the published convention,
    # channel-switched to VIO.
    settle_seq = [iv for iv in intervals if iv["phase"] == "settle"]
    step_seq = [iv for iv in intervals if iv["phase"] == "step"]
    settle_rates = [iv["rate_deg_per_s"] for iv in settle_seq]
    if step_seq:
        settle_rates = [step_seq[-1]["rate_deg_per_s"], *settle_rates]
    settle_diff = (
        round(abs(settle_rates[-1] - settle_rates[-2]), 4)
        if len(settle_rates) >= 2
        else None
    )
    settle_2b = {
        "informative_intervals": len(settle_seq),
        "rates_deg_per_s_including_carryover_from_step": settle_rates,
        "last_two_diff_deg_per_s": settle_diff,
        "passed": (
            len(settle_seq) >= _STEP_RESPONSE_MIN_INFORMATIVE_INTERVALS
            and settle_diff is not None
            and settle_diff <= _STEP_RESPONSE_RATE_AGREEMENT_BOUND_DEG_PER_S
        ),
    }

    # Steady omega is the step's SECOND half -- the part 2a certifies steady.
    half_rates = step_2a["half_rates_deg_per_s"]
    omega_step = half_rates[1] if step_2a["passed"] and half_rates else None
    rotation_after_zero: float | None = None
    if step_seq and settle_seq:
        rotation_after_zero = round(
            normalize_degrees(
                settle_seq[-1]["to_heading_degrees"]
                - step_seq[-1]["to_heading_degrees"]
            ),
            4,
        )
    tau: float | None = None
    if rotation_after_zero is not None and omega_step:
        tau = round(abs(rotation_after_zero / omega_step), 4)

    return {
        "scoreable": True,
        "rule": (
            "E-VIO: 2a = half-phase mean-rate agreement (step), 2b = last-two "
            "settle rates with carryover; bound 1.5 deg/s, >=3 intervals per "
            "phase; VIO heading between consecutive distinct readings."
        ),
        "intervals": intervals,
        "informative_intervals": {
            "baseline": sum(1 for iv in intervals if iv["phase"] == "baseline"),
            "step": len(step_seq),
            "settle": len(settle_seq),
        },
        "step_steady_rotation_2a": step_2a,
        "settle_flat_2b": settle_2b,
        "omega_step_deg_per_s": omega_step,
        "rotation_after_zero_deg": rotation_after_zero,
        "tau_actuator_s": tau,
        "interpretation": (
            "tau_actuator_s exists only when 2a passes, so omega is never "
            "sampled off a ramp. omega is the step's second-half mean rate on "
            "the same VIO channel that certified steadiness. The RTK "
            "course_series/analysis alongside are diagnostics, not the 2a/2b "
            "instrument -- their last-two-diff statistic carries ~2.7 deg/s of "
            "1-sigma chord noise against the 1.5 deg/s bound."
        ),
    }


async def _step_response_probe_impl(  # noqa: C901, PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    *,
    position_stream: Any | None,
    report_lease: Any | None = None,
    route_start: dict[str, float],
    corridor_polygon: list[dict[str, float]],
    linear_speed: int = 400,
    step_angular_speed: int = 120,
    baseline_ms: int = 3000,
    step_ms: int = 3000,
    settle_ms: int = 4000,
    motion_refresh_interval_ms: int = 200,
    sample_interval_ms: int = 100,
    max_travel_m: float = _STEP_RESPONSE_DEFAULT_TRAVEL_M,
    prefer_ble: bool = True,
    dry_run: bool = True,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    confirm_step_response_run: bool = False,
) -> dict[str, Any]:
    """Run or simulate one open-loop baseline -> step -> settle window."""
    before = _custom_path_telemetry_snapshot(coordinator)
    total_ms = baseline_ms + step_ms + settle_ms
    gates = _step_response_gates(
        coordinator,
        before,
        route_start=route_start,
        corridor_polygon=corridor_polygon,
        max_travel_m=max_travel_m,
        linear_speed=linear_speed,
        total_ms=total_ms,
        dry_run=dry_run,
        confirm_blades_off=confirm_blades_off,
        confirm_clear_area=confirm_clear_area,
    )
    blockers = [gate["name"] for gate in gates if not gate["passed"]]
    # Opt-in PER CALL, exactly like `confirm_steering_validation_run`: arming the
    # motion gate is deliberately not sufficient to drive an open-loop curve.
    if not dry_run and not confirm_step_response_run:
        blockers.append("step_response_run_not_confirmed")
    if total_ms > _STEP_RESPONSE_MAX_TOTAL_MS:
        blockers.append("step_window_too_long")
    # A long window at the FAST speed overruns the distance guard mid-run, which
    # aborts the window and censors the measurement -- a wasted supervised run
    # rather than an unsafe one. Refuse that pairing before dispatch. The bound
    # is a LOWER one, so this fires only when the window cannot fit even at the
    # slowest speed measured; a marginal config is left to the guard, which is
    # what actually carries the safety.
    window_s = total_ms / 1000.0
    floor_travel_m = (
        _STEP_RESPONSE_MIN_SPEED_BY_LINEAR.get(int(linear_speed), 0.10) * window_s
    )
    typical_travel_m = (
        _STEP_RESPONSE_TYPICAL_SPEED_BY_LINEAR.get(int(linear_speed), 0.216) * window_s
    )
    if floor_travel_m > max_travel_m:
        blockers.append("step_window_travel_exceeds_budget")
    # Non-blocking: a window the floor clears but the TYPICAL speed does not is
    # likely to end on the guard, which censors the measurement. Surfaced so the
    # operator can retune before spending a supervised run, never enforced --
    # over-refusing feasible configurations is the failure this replaced.
    travel_projection = {
        "window_s": round(window_s, 3),
        "floor_speed_m_s": _STEP_RESPONSE_MIN_SPEED_BY_LINEAR.get(int(linear_speed)),
        "floor_travel_m": round(floor_travel_m, 4),
        "typical_speed_m_s": _STEP_RESPONSE_TYPICAL_SPEED_BY_LINEAR.get(
            int(linear_speed)
        ),
        "typical_travel_m": round(typical_travel_m, 4),
        "max_travel_m": max_travel_m,
        "likely_guard_trip": typical_travel_m > max_travel_m,
        "caveat": (
            "typical_speed_m_s is the SUSTAINED (post-ramp) speed, measured: "
            "linear 300 = 0.223 m/s (Phase A, 2026-09-03), linear 400 = 0.295. "
            "A short window averages LESS, because it spends a larger fraction "
            "of itself ramping, so this projection OVER-states travel there -- "
            "the safe direction for a warning."
        ),
    }

    config = ContinuousControllerConfig()
    result: dict[str, Any] = {
        "service": SERVICE_STEP_RESPONSE_PROBE,
        "mode": "dry_run" if dry_run else "real_step_response_probe",
        "dry_run": dry_run,
        "purpose": (
            "Open-loop dead-time measurement for Q1/Q2 of "
            "docs/phase2-dead-time-step-test-design-20260828.md. No controller, "
            "no route, no steering law."
        ),
        "route_start": route_start,
        "corridor_polygon": corridor_polygon,
        "phases": {
            "baseline_ms": baseline_ms,
            "step_ms": step_ms,
            "settle_ms": settle_ms,
            "total_ms": total_ms,
        },
        "linear_speed": linear_speed,
        "step_angular_speed": step_angular_speed,
        "motion_refresh_interval_ms": motion_refresh_interval_ms,
        "sample_interval_ms": sample_interval_ms,
        "max_travel_m": max_travel_m,
        "travel_projection": travel_projection,
        "safety_gates": gates,
        "blockers": blockers,
        "would_send": not dry_run and not blockers,
        "command_result": {
            "attempted": False,
            "ok": None,
            "error": None,
            "duration_ms": None,
        },
        "stop_result": {"attempted": False, "ok": None, "error": None},
        "phase_transitions": [],
        "samples": [],
        "course_series": [],
        "analysis": None,
        "report_lease": {"held": report_lease is not None},
    }
    if dry_run or blockers:
        result["reason"] = "dry_run" if dry_run else "safety_gates_failed"
        return result
    if position_stream is None:
        result["would_send"] = False
        result["blockers"] = ["position_stream_unavailable"]
        result["reason"] = "position_stream_unavailable"
        return result

    # 🚨 **THE LEASE STOPS THE REPORT STREAM. IT MUST BE RESTARTED HERE.**
    # `exclusive_report_subscription` enqueues `RPT_STOP` and clears
    # `_ble_stream_active` as its FIRST act -- its own docstring says it
    # "stop[s] background renewals" -- and it blocks the background loop from
    # starting a new configuration for the life of the lease. Taking the lease
    # and driving without restarting the stream means NO position payloads for
    # the whole window.
    # 🐛 That was this probe's first bug, 2026-08-29. Four runs across three
    # builds drove blind and were mis-read as a device- or backend-side feed
    # stall, until `continuous_motion_window` -- which does restart the stream --
    # turned out to be the only motion path whose feed worked. See
    # `docs/evidence-step-probe-stalled-on-its-own-lease-20260829.md`.
    handle = coordinator.manager.mower(coordinator.device_name)
    begin_generation = getattr(handle, "begin_report_subscription_generation", None)
    if report_lease is None or not callable(begin_generation):
        result["would_send"] = False
        result["blockers"] = ["report_subscription_generation_unavailable"]
        result["reason"] = "report_subscription_generation_unavailable"
        return result
    report_generation = begin_generation(report_lease)
    stream_result: dict[str, Any] = {
        "attempted": True,
        "started": False,
        "continuous_started": False,
        "error": None,
        # Both start calls reach the queue at Priority.BACKGROUND with
        # skip_if_saga_active=True, so a running saga drops them silently while
        # these flags still read True. Capturing it keeps a dispatch failure
        # distinguishable from a telemetry stall -- the exact confusion this
        # probe already caused once.
        "saga_active_before_request": _saga_active_for_diagnostics(handle),
        "subscription_generation": {
            "owner": report_generation.owner,
            "lease_id": report_generation.lease_id,
            "generation": report_generation.generation,
            "baseline_position_sequence": (
                report_generation.baseline_position_sequence
            ),
            "baseline_position_epoch": report_generation.baseline_position_epoch,
        },
    }
    result["report_stream"] = stream_result
    stream_duration_ms = max(10_000, total_ms + 5_000)
    baseline_dropped_samples = position_stream.dropped_samples
    try:
        if hasattr(coordinator, "async_start_report_stream"):
            await coordinator.async_start_report_stream(duration_ms=stream_duration_ms)
            stream_result["started"] = True
        if hasattr(coordinator, "async_start_continuous_reports"):
            await coordinator.async_start_continuous_reports(
                duration_ms=stream_duration_ms
            )
            stream_result["continuous_started"] = True
        # Both start calls return on ENQUEUE, so only the post-settle instant
        # proves the START reached the transport.
        stream_result["queue_settle"] = await _settle_ble_command_queue(coordinator)
        flushed_at = time.monotonic()
        stream_result["subscription_command_flushed_at_monotonic"] = flushed_at
    except Exception as err:  # noqa: BLE001
        stream_result["error"] = f"{type(err).__name__}: {err}"
        result["would_send"] = False
        result["blockers"] = ["report_stream_start_failed"]
        result["reason"] = "report_stream_start_failed"
        return result

    # 🔑 **FAIL CLOSED ON A FEED THAT IS NOT DELIVERING.** The only positive
    # evidence a configuration is live is a position payload inside its OWN
    # generation. Without this the probe would silently repeat 2026-08-29:
    # drive, record nothing, and look like a device fault.
    (
        origin_sample,
        _origin_elapsed,
        origin_reason,
    ) = await _wait_for_position_subscription_ready(
        handle,
        position_stream,
        report_generation,
        lease=report_lease,
        timeout_seconds=_STEP_RESPONSE_READINESS_TIMEOUT_S,
        not_before_monotonic=flushed_at,
        baseline_dropped_samples=baseline_dropped_samples,
    )
    stream_result["readiness_reason"] = origin_reason
    stream_result["ready"] = origin_sample is not None
    if origin_sample is None:
        result["would_send"] = False
        result["blockers"] = ["position_subscription_not_ready"]
        result["reason"] = origin_reason or "position_subscription_not_ready"
        return result
    stream_result["origin_position_sequence"] = origin_sample.sequence
    stream_result["origin_position_epoch"] = origin_sample.epoch

    command_state: dict[str, int] = {
        "linear_speed": linear_speed,
        "angular_speed": 0,
    }
    refresh_state: dict[str, Any] = {"completions_elapsed_ms": []}
    travel_abort = asyncio.Event()
    sampler_stop = asyncio.Event()
    window_started = time.monotonic()

    result["command_result"]["attempted"] = True
    started = time.monotonic()
    try:
        await _send_manager_command_with_args(
            coordinator,
            "send_movement",
            prefer_ble=prefer_ble,
            command_kwargs=dict(command_state),
        )
        result["command_result"]["ok"] = True
    except Exception as err:  # noqa: BLE001
        result["command_result"]["ok"] = False
        result["command_result"]["error"] = f"{type(err).__name__}: {err}"
        result["reason"] = "command_failed"
        return result
    finally:
        result["command_result"]["duration_ms"] = round(
            (time.monotonic() - started) * 1000, 3
        )
    refresh_state["completions_elapsed_ms"].append(
        float(result["command_result"]["duration_ms"])
    )

    sampler_task = asyncio.create_task(
        _capture_in_window_telemetry(
            coordinator,
            sample_interval_ms=sample_interval_ms,
            duration_ms=total_ms,
            window_started=window_started,
            stop_event=sampler_stop,
            command="send_movement",
            # The live dict, so every sample records the angular that was
            # actually in force when it was taken.
            command_args=command_state,
            max_travel_m=max_travel_m,
            travel_abort=travel_abort,
            position_stream=position_stream,
        )
    )
    phase_task = asyncio.create_task(
        _step_response_phase_scheduler(
            command_state,
            window_started=window_started,
            baseline_ms=baseline_ms,
            step_ms=step_ms,
            step_angular_speed=step_angular_speed,
            abort_event=travel_abort,
        )
    )

    # 🚨 NEITHER HELPER MAY DIE QUIETLY WHILE THE MOWER DRIVES. Same reasoning as
    # beta72's `_abort_if_sampler_died` and the continuous window's decision-loop
    # guard: a dead sampler means the distance guard is gone, and a phase
    # scheduler that dies mid-step leaves a turn command standing for the rest of
    # the window. Both must stop the refresh loop, which brings the mandatory
    # stop forward.
    def _abort_if_helper_died(task: asyncio.Task[Any]) -> None:
        if task.cancelled() or task.exception() is not None:
            travel_abort.set()

    sampler_task.add_done_callback(_abort_if_helper_died)
    phase_task.add_done_callback(_abort_if_helper_died)

    try:
        result["motion_refresh"] = await _continuous_refresh_window(
            coordinator,
            command_state=command_state,
            prefer_ble=prefer_ble,
            duration_seconds=total_ms / 1000.0,
            refresh_interval_ms=motion_refresh_interval_ms,
            window_started=window_started,
            refresh_state=refresh_state,
            abort_event=travel_abort,
        )
    finally:
        # The stop is mandatory and must not be skipped by an exception path.
        sampler_stop.set()
        travel_abort.set()
        with contextlib.suppress(BaseException):
            await phase_task
        result["stop_result"] = await _manual_velocity_stop_attempt(
            coordinator, use_wifi=not prefer_ble
        )

    result["phase_transitions"] = phase_task.result() if phase_task.done() else []
    result["samples"] = await sampler_task
    result["course_series"] = _step_response_course_series(
        result["samples"],
        baseline_ms=baseline_ms,
        step_ms=step_ms,
        min_chord_m=config.min_travel_for_heading_trust_m,
    )
    result["analysis"] = _step_response_analysis(
        result["course_series"], baseline_ms=baseline_ms, step_ms=step_ms
    )
    result["vio_analysis"] = _step_response_vio_analysis(
        result["samples"], baseline_ms=baseline_ms, step_ms=step_ms
    )
    result["reason"] = _step_response_completion_reason(result["motion_refresh"])
    result["after"] = _custom_path_telemetry_snapshot(coordinator)
    return result


def _step_response_completion_reason(motion_refresh: Mapping[str, Any]) -> str:
    """Say why the window ended, from evidence the loop itself observed.

    🐛 This used to read `travel_abort.is_set()` directly, which is ALWAYS true
    by this point: the caller's `finally` block sets that same event
    unconditionally as part of mandatory-stop teardown (to also unblock the
    phase/sampler tasks), whether `_continuous_refresh_window` returned because
    the guard fired mid-window or because the window simply finished on
    schedule. `motion_refresh["aborted_early"]` is set only when the refresh
    loop itself observed the abort event WHILE STILL RUNNING, so it is the one
    signal that actually distinguishes the two. Found 2026-08-30 when route-1
    run 1 completed its full 13000 ms window -- confirmed by phase_transitions
    landing on schedule and zero tripped samples -- while the old logic still
    reported "travel_guard_tripped".
    """
    return (
        "travel_guard_tripped"
        if motion_refresh.get("aborted_early")
        else "window_complete"
    )


async def _step_response_probe(
    coordinator: MammotionReportUpdateCoordinator,
    **kwargs: Any,
) -> dict[str, Any]:
    """Hold the report-subscription lease around one step-response window."""
    if kwargs.get("dry_run", True):
        return await _step_response_probe_impl(
            coordinator, position_stream=None, report_lease=None, **kwargs
        )

    handle = coordinator.manager.mower(coordinator.device_name)
    exclusive_factory = getattr(handle, "exclusive_report_subscription", None)
    open_position_stream = getattr(coordinator, "open_position_sample_stream", None)
    if not callable(exclusive_factory) or not callable(open_position_stream):
        return {
            "service": SERVICE_STEP_RESPONSE_PROBE,
            "mode": "real_step_response_probe",
            "dry_run": False,
            "would_send": False,
            "blockers": ["position_subscription_lease_unavailable"],
            "reason": "position_subscription_lease_unavailable",
        }

    result: dict[str, Any] | None = None
    async with exclusive_factory(SERVICE_STEP_RESPONSE_PROBE) as report_lease:
        position_stream = open_position_stream(maxsize=_SAFETY_POSITION_STREAM_MAXSIZE)
        try:
            result = await _step_response_probe_impl(
                coordinator,
                position_stream=position_stream,
                report_lease=report_lease,
                **kwargs,
            )
            return result  # noqa: RET504 - finally annotates report-stream teardown
        finally:
            stop_reports = getattr(coordinator, "async_stop_continuous_reports", None)
            if callable(stop_reports):
                try:
                    await stop_reports()
                    if result is not None:
                        result.setdefault("report_stream", {})["stopped"] = True
                except Exception as err:  # noqa: BLE001
                    if result is not None:
                        result.setdefault("report_stream", {})["stop_error"] = (
                            f"{type(err).__name__}: {err}"
                        )
            if position_stream is not None:
                position_stream.close()


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
    # --- feature-exposure shortlist probe (2026-07-11, read-only) ---
    # Obstacle/safety sensor health decoded from dev.sensor_status.
    "report_data.dev.sensor_status",
    "report_data.dev.bumper_state",
    "report_data.dev.ult_left",
    "report_data.dev.ult_left_front",
    "report_data.dev.ult_right_front",
    "report_data.dev.ult_right",
    "report_data.dev.fuse_status",
    # Device status / security / self-test.
    "report_data.dev.lock_state.lock_state",
    "report_data.dev.self_check_status",
    "report_data.dev.fpv_info.fpv_flag",
    "report_data.dev.collector_status.collector_installation_status",
    # 4G modem (non-sensitive fields only; IMEI/SIM/ICCID deliberately omitted).
    "report_data.dev.mnet_info.operator",
    "report_data.dev.mnet_info.model",
    "report_data.dev.mnet_info.inet",
    "report_data.dev.mnet_info.rssi",
    # RTK fix-quality detail.
    "report_data.rtk.lat_std",
    "report_data.rtk.lon_std",
    "report_data.rtk.top4_total_mean",
    "report_data.rtk.dis_status",
    # RTK base-station identity/health.
    "report_data.basestation_info.ver_major",
    "report_data.basestation_info.ver_minor",
    "report_data.basestation_info.ver_patch",
    "report_data.basestation_info.basestation_status",
    "report_data.basestation_info.sats_num",
    # Connectivity availability.
    "report_data.connect.mnet_inet",
    "report_data.connect.wifi_is_available",
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
    for index, location in enumerate(
        _safe_attr_path(data, "report_data.locations") or []
    ):
        if index >= 3:
            break
        bol_hash = _safe_attr_path(location, "bol_hash")
        try:
            safe_bol_hash: Any = _json_safe_int(int(bol_hash or 0))
        except TypeError, ValueError:
            safe_bol_hash = str(bol_hash) if bol_hash is not None else None
        zone_hash = _safe_attr_path(location, "zone_hash")
        try:
            safe_zone_hash: Any = _json_safe_int(int(zone_hash or 0))
        except TypeError, ValueError:
            safe_zone_hash = str(zone_hash) if zone_hash is not None else None
        sources["report_data.locations"].append(
            {
                "index": index,
                "real_pos_x": _safe_attr_path(location, "real_pos_x"),
                "real_pos_y": _safe_attr_path(location, "real_pos_y"),
                "real_toward": _safe_attr_path(location, "real_toward"),
                "pos_type": _enum_value(_safe_attr_path(location, "pos_type")),
                # Distinct proto fields: zone_hash is field 5 (current mowing
                # zone), bol_hash is field 6 (whole-map checksum).
                "zone_hash": safe_zone_hash,
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
                    # pymammotion has no availability.ble_in_cooldown flag; the
                    # real connect cooldown lives on the BLE transport, so read it
                    # there (this used to always report None).
                    "ble_connect_cooldown_active": _ble_connect_cooldown_active(
                        coordinator
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
            await _motion_open_sleep(coordinator, pulse_gap_seconds)
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
            # Same pacing floor as the vio_turn poll: never hammer the BLE
            # command queue when refresh_wait_seconds is 0.
            await asyncio.sleep(max(refresh_wait_seconds, 0.5))
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
        # Pacing floor: attempts run back-to-back per pulse over the same BLE
        # queue that delivers motion stops; refresh_wait_seconds=0 is schema-legal.
        await asyncio.sleep(max(refresh_wait_seconds, 0.5))
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
            await _motion_open_sleep(coordinator, pulse_gap_seconds)
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
        "displacement_source": None,
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
            # Phantom-motion investigation instrumentation (capture only).
            sample["position_source_comparison"] = _position_source_comparison(
                coordinator
            )
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
                result["stop_ack"] = await _stop_manual_motion_confirmed(coordinator)
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
        sample["position_source_comparison"] = _position_source_comparison(coordinator)
        post_stop.append(sample)
        prev_telemetry = snapshot["telemetry"]

    result["samples"] = samples
    result["post_stop"] = post_stop

    def _vio_active(value: Any) -> bool:
        return value is not None and value != 0

    all_samples = samples + post_stop
    # The position feed lags ~4s and only catches up AFTER the drive ends, so a
    # real move lands in the post-stop samples while the during-drive samples stay
    # frozen. Judge motion + displacement across all samples (not just the frozen
    # drive ones), otherwise a real move is mislabelled no_motion_detected and
    # final_displacement_m reads ~0 (live 2026-07-15: a 4in 6s pulse did exactly
    # this).
    motion_confirmed = any(sample["moving"] for sample in all_samples)
    # motion_confirmed (above) already establishes the mower moved; the per-sample
    # `moving` flag is unreliable DURING the drive because the position feed lags
    # ~4s and stays frozen, so a VIO-active pulse whose motion only registers
    # post-stop must NOT be judged "VIO never initialized". Ask only whether VIO
    # was active across the drive window -- VIO waking after the stop is not
    # "during motion", so post_stop samples are deliberately excluded here.
    vio_activated_while_moving = any(
        _vio_active(sample["vio_state"]) for sample in samples
    )
    vio_activated_any = any(_vio_active(sample["vio_state"]) for sample in all_samples)
    heading_series = [
        sample["vision_heading"]
        for sample in samples
        if _vio_active(sample["vio_state"])
    ]
    # Prefer the settled post-stop displacement; fall back to the drive samples.
    # NOTE: this is the feed's settled best estimate -- it still cannot tell a real
    # move from a phantom feed-jump on a no-op pulse (the position_source_comparison
    # capture is what the phantom detector will use).
    final_displacement: float | None = None
    displacement_source: str | None = None
    for sample in reversed(post_stop):
        if sample["delta_from_initial_m"] is not None:
            final_displacement = sample["delta_from_initial_m"]
            displacement_source = "post_stop"
            break
    if final_displacement is None:
        for sample in reversed(samples):
            if sample["delta_from_initial_m"] is not None:
                final_displacement = sample["delta_from_initial_m"]
                displacement_source = "drive"
                break
    result["final_displacement_m"] = final_displacement
    result["displacement_source"] = displacement_source
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
    angular_speed: int = 500,
    linear_speed: int = 0,
    drive_seconds: float = 6.0,
    sample_interval_seconds: float = 1.5,
    post_stop_samples: int = 3,
    max_displacement_m: float = 0.5,
    min_heading_change_degrees: float = 3.0,
    motion_refresh_interval_ms: int = 0,
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
        "motion_refresh_interval_ms": motion_refresh_interval_ms,
        "motion_refresh_commands_sent": 0,
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
        "displacement_source": None,
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
    refresh_commands_sent = 0
    # App-parity re-send: re-issue the identical rotation command during the gaps
    # between samples. Bounded by the same drive_seconds/displacement caps and the
    # mandatory stop below, so it can only ever mean "kept the turn going", never
    # "ran longer". motion_refresh_interval_ms=0 is the proven single-shot path.
    refresh_turn_command = functools.partial(
        _send_manager_command_with_args,
        coordinator,
        "send_movement",
        prefer_ble=prefer_ble,
        command_kwargs=command_args,
    )
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
            # Phantom-motion investigation instrumentation (capture only).
            sample["position_source_comparison"] = _position_source_comparison(
                coordinator
            )
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
            refresh_report = await _motion_refresh_window(
                coordinator,
                resend=refresh_turn_command,
                duration_seconds=sample_interval_seconds,
                refresh_interval_ms=motion_refresh_interval_ms,
            )
            refresh_commands_sent += refresh_report["refresh_commands_sent"]
    except Exception as err:  # noqa: BLE001
        aborted_reason = "command_failed"
        result["command_ok"] = command_started
        result["command_error"] = f"{type(err).__name__}: {err}"
    finally:
        if command_started:
            try:
                result["stop_ack"] = await _stop_manual_motion_confirmed(coordinator)
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
        sample["position_source_comparison"] = _position_source_comparison(coordinator)
        post_stop.append(sample)
        prev_telemetry = snapshot["telemetry"]

    result["samples"] = samples
    result["post_stop"] = post_stop
    result["motion_refresh_commands_sent"] = refresh_commands_sent

    # VIO heading refreshes ~1.5s into the command and the position feed lags ~4s,
    # so on a short pulse the ONLY sample taken during the command is the t=0 one
    # (bit-identical to baseline) and every real change lands in post_stop. Judging
    # the during-command samples alone therefore reports a real rotation as zero:
    # live 2026-07-19 a taped 13.18 deg pivot came back
    # `vision_heading_static_during_command` with `final_displacement_m: 0.0` while
    # this function's own post_stop samples held the answer. Same class of bug the
    # 2026-07-16 pass fixed in _vio_motion_probe; judge across all samples.
    turn_samples = samples + post_stop
    heading_seq = [baseline["vision_heading"]] + [
        s["vision_heading"] for s in turn_samples
    ]
    cog_seq = [baseline["toward"]] + [s["toward"] for s in turn_samples]
    vision_change = _angle_series_change(heading_seq)
    cog_change = _angle_series_change(cog_seq)
    # Liveness stays scoped to the command window: VIO waking after the stop is not
    # "active throughout the turn", so post_stop is deliberately excluded here.
    vio_states = [baseline["vio_state"]] + [s["vio_state"] for s in samples]
    vio_active_throughout = bool(vio_states) and all(
        state is not None and state != 0 for state in vio_states
    )
    # max_displacement_m is the safety bound (a pivot should barely translate), so
    # it stays a MAX across the whole window. final_displacement_m is the settled
    # post-stop estimate, mirroring _vio_motion_probe.
    max_disp: float | None = None
    for sample in turn_samples:
        disp = sample["delta_from_initial_m"]
        if disp is not None and (max_disp is None or disp > max_disp):
            max_disp = disp
    final_displacement: float | None = None
    displacement_source: str | None = None
    for sample in reversed(post_stop):
        if sample["delta_from_initial_m"] is not None:
            final_displacement = sample["delta_from_initial_m"]
            displacement_source = "post_stop"
            break
    if final_displacement is None:
        for sample in reversed(samples):
            if sample["delta_from_initial_m"] is not None:
                final_displacement = sample["delta_from_initial_m"]
                displacement_source = "drive"
                break
    result["final_displacement_m"] = final_displacement
    result["displacement_source"] = displacement_source
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


# report_data.vision_info.vio_state == 2 means the visual-odometry track is active
# and vision_heading is trustworthy. 0 is cold/uninitialised (e.g. dark scene).
_VIO_STATE_ACTIVE = 2

# vio_state alone lies: at dusk the tracked-feature count collapses to 0 while
# vio_state latches at 2 and the heading freezes bit-identical (live run 2,
# 2026-07-15). Require at least this many tracked features before trusting a VIO
# heading for real motion. Healthy runs report ~80; this margin above 0 catches
# a collapsing feed a beat before it is fully blind without risking false aborts.
_VIO_MIN_TRACKED_FEATURES = 5

# A single degraded feed read mid-turn can be a transient feature dip (a brief
# occlusion), not sustained dusk blindness. Before aborting the turn, re-poll the
# feed (read-only, no motion) up to this many times, returning as soon as it
# recovers -- symmetric with tolerating a single stale heading sample rather than
# ending the turn on it.
_VIO_FEED_RECONFIRM_POLLS = 2

# The map-local position feed lags ~4s and updates in JUMPS: after a linear pulse
# the reported x/y can sit at the pre-pulse value for a couple of samples and then
# jump (live 2026-07-15: two back-to-back 4s pulses reported ~10cm each yet the
# mower physically moved <6" total -- pulse 2's "displacement" was the delayed
# registration of pulse 1). So before judging a linear pulse's progress, poll the
# feed until it SETTLES (two consecutive snapshots agree within this epsilon) or a
# bounded timeout elapses, so each pulse's displacement is attributed to that pulse
# instead of leaking across pulses -- the position analogue of the turn phase's
# fresh-heading poll.
_LINEAR_POSITION_SETTLE_EPSILON_M = 0.01
_LINEAR_POSITION_SETTLE_TIMEOUT_SECONDS = 6.0

# Minimum wrap-aware heading change (degrees) that counts as a genuinely fresh
# reading in the vio_turn poll loop. A bare float-inequality check treated a
# latched feed's sub-0.01 sensor noise wiggle as movement (live run 2, dusk:
# a 0.0018 deg jitter passed while VIO was actually blind), so require the
# change to clear real noise before judging progress.
_VIO_HEADING_FRESH_EPSILON_DEGREES = 0.1

# Displacement (metres) below which a pulse is treated as having produced no
# physical motion at all. Deliberately conservative: the position feed has a
# ~2-6cm ABSOLUTE noise floor, and a real in-place pivot was measured at ~8cm of
# incidental translation (live 2026-07-19, tape-confirmed 13.18 deg), while
# genuinely dead pulses in the same session read 0.25-0.43cm. False negatives
# (missing a real no-actuation) are acceptable; false positives (calling a real
# turn "no actuation") are not, so this sits well under both the noise floor and
# the observed pivot translation.
_NO_ACTUATION_DISPLACEMENT_M = 0.02

# Consecutive settle polls that must all return bit-identical coordinates before
# the position feed is called stale. A live feed jitters ~2-4mm between reads, so
# zero movement across several polls means the feed stopped updating rather than
# the mower stopping. Live 2026-07-19: linear pulses 11-13 of a card run all read
# (4.6672, -1.4147) exactly and the run aborted no_target_progress -- but the
# mower had actually driven 25.4cm during them (~8.5cm/pulse, matching the proven
# step), which only showed up after the feed caught up post-run. The executor was
# issuing motion commands against a stale position for ~30s.
_STALE_FEED_MIN_POLLS = 3


def _streak_shows_no_actuation(
    command_results: list[dict[str, Any]], streak: int
) -> bool:
    """Return True when the last ``streak`` pulses show no actuation whatsoever.

    Distinguishes "the mower is moving but the turn isn't converging" (a control
    problem -- ``no_heading_progress``) from "the command path is dead" (an
    actuation problem). The latter has no HA-visible cause: a physical e-stop is
    completely invisible in telemetry (2026-07-19: raw report fields are
    byte-identical engaged vs cleared), and an unusable-but-selected BLE
    transport also accepts commands that never arrive. In both cases the mower
    reports ``command_ok`` with a sub-millisecond dual-axis stop ACK and does
    nothing, so the only evidence is that neither sensor moved.

    Requires the raw heading to be **bit-identical** across the pulse, not merely
    rounded-to-zero. That is the measured signature of a dead command path (live
    2026-07-19: 91.38829636391407 unchanged to 14 decimal places across five
    commands and 45 minutes). A VIO feed latched by dusk on a mower that IS
    rotating instead emits sub-epsilon sensor noise (~0.0018 deg, live run 2),
    which is NOT bit-identical -- so that case correctly stays
    ``no_heading_progress`` rather than being mislabelled a dead link.

    Position must ALSO be flat, since a latched feed alone proves nothing about
    whether the mower moved.
    """
    if streak <= 0 or len(command_results) < streak:
        return False
    for command in command_results[-streak:]:
        before = command.get("before_vision_heading")
        after = command.get("after_vision_heading")
        # Bit-identical raw heading: no sensor noise at all, not just "small".
        if before is None or after is None or float(before) != float(after):
            return False
        displacement = command.get("displacement_m")
        # Unknown displacement means we cannot prove the mower stayed put.
        if displacement is None:
            return False
        if float(displacement) > _NO_ACTUATION_DISPLACEMENT_M:
            return False
    return True


def _streak_shows_dead_telemetry(
    command_results: list[dict[str, Any]], streak: int
) -> bool:
    """Return True when the last ``streak`` pulses saw no live telemetry at all.

    ``_streak_shows_no_actuation`` asserts something about the *mower* -- that it
    accepted commands and did not move. That claim is only legitimate when the
    sensors were demonstrably alive; otherwise "nothing changed" means "we went
    blind", which needs the opposite response from the operator (fix the link,
    do not go looking for a physical e-stop).

    Liveness is judged on the same principle the linear phase uses
    (``_settle_linear_position_feed``): a live feed is never perfectly still.
    Position jitters ~2-4mm between consecutive reads even on a stationary
    mower, and a VIO heading latched by dusk still emits sub-epsilon noise
    (~0.0018 deg, live run 2 2026-07-15). A change in *either* channel proves
    reports are arriving, so only pulses whose heading and position were both
    bit-identical across every poll count as a dead stream. Requires at least
    ``_STALE_FEED_MIN_POLLS`` polls in each pulse -- one or two unchanged reads
    prove nothing.

    This deliberately keeps the dusk-latch case (``no_heading_progress``) and the
    live-link e-stop case (``no_actuation_detected``) out of the stale branch:
    both have a demonstrably live feed.

    Measured live 2026-07-25: two turn pulses reported bit-identical
    ``vision_heading`` (90.29915121519771) *and* bit-identical ``displacement_m``
    (0.006754257916307457) while the operator watched the mower turn ~4 inches.
    The server log for that window shows BLE frames being discarded outright
    (``dropping frame: malformed report data failed deserialization``), so the
    telemetry really was dead while actuation was fine.
    """
    if streak <= 0 or len(command_results) < streak:
        return False
    for command in command_results[-streak:]:
        polls = command.get("heading_poll_count")
        if polls is None or int(polls) < _STALE_FEED_MIN_POLLS:
            return False
        if command.get("heading_poll_feed_alive") is not False:
            return False
    return True


def _vio_reading(coordinator: MammotionReportUpdateCoordinator) -> dict[str, Any]:
    """Return the current VIO heading and state from live telemetry."""
    paths = _position_feedback_raw_sources(coordinator).get("paths", {})
    return {
        "vision_heading": paths.get("report_data.vision_info.heading"),
        "vio_state": paths.get("report_data.vision_info.vio_state"),
    }


def _vio_scene_brightness(
    coordinator: MammotionReportUpdateCoordinator,
) -> tuple[Any, str | None]:
    """Return the raw camera brightness value and its label ("Light"/"Dark"/...).

    One place for the (historically fragile) ``vision_info.brightness`` read plus
    the ``camera_brightness()`` mapping, shared by ``_vio_scene_is_bright`` and
    ``_vio_feed_liveness``. Label is None when brightness is absent or unparsable.
    """
    raw = _safe_attr_path(coordinator.data, "report_data.vision_info.brightness")
    label: str | None = None
    if raw is not None:
        try:
            label = camera_brightness(int(raw))
        except TypeError, ValueError:
            label = None
    return raw, label


def _vio_scene_is_bright(coordinator: MammotionReportUpdateCoordinator) -> bool:
    """Return whether the camera scene is bright enough for VIO to initialise.

    VIO is visual odometry: it cannot bootstrap a feature track in the dark
    (live-proven 2026-07-11), but in daylight it wakes during the first forward
    motion, so a bright scene means the calibration drive can double as the
    warm-up.
    """
    return _vio_scene_brightness(coordinator)[1] == "Light"


def _vio_feed_liveness(
    coordinator: MammotionReportUpdateCoordinator,
) -> dict[str, Any]:
    """Report whether the VIO feed is actually live, beyond the latch-prone vio_state.

    ``vio_state == 2`` can persist while the visual track is blind: at dusk the
    tracked-feature count falls to 0 but the state does not follow and the
    heading freezes (live run 2, 2026-07-15), so a heading read in that window is
    a stale latch. Gate real motion on ``track_feature_num`` (which does drop to
    0) and surface the scene brightness so an operator can tell a dark/blind feed
    from a mower that simply is not rotating. A missing feature field reads as
    "live" so devices that never report it are not blocked.
    """
    features = _safe_attr_path(
        coordinator.data, "report_data.vision_info.track_feature_num"
    )
    brightness_raw, brightness_label = _vio_scene_brightness(coordinator)
    degraded = False
    if features is not None:
        try:
            degraded = int(features) < _VIO_MIN_TRACKED_FEATURES
        except TypeError, ValueError:
            degraded = False
    return {
        "live": not degraded,
        "tracked_features": features,
        "brightness_raw": brightness_raw,
        "brightness_label": brightness_label,
    }


def _vio_feed_live_gate(feed: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    """Build the ``vio_feed_live`` safety gate for a degraded VIO feed.

    Shared by the turn and vector-segment executors so the blind-feed messaging
    stays consistent. Passes on dry runs (planning is allowed against a cold/blind
    feed); blocks a real run because the latched heading cannot be trusted.
    """
    return {
        "name": "vio_feed_live",
        "passed": dry_run,
        "detail": (
            "VIO feed degraded: only "
            f"{feed['tracked_features']} tracked features "
            f"(need >= {_VIO_MIN_TRACKED_FEATURES}), "
            f"brightness {feed['brightness_label']}. The track is blind despite "
            "vio_state; wait for daylight or warm VIO with forward motion."
        ),
    }


async def _reconfirm_vio_feed_degraded(
    coordinator: MammotionReportUpdateCoordinator,
    current: dict[str, Any],
    *,
    refresh_wait_seconds: float,
) -> dict[str, Any]:
    """Re-poll a degraded VIO feed (read-only) to tell a transient dip from blindness.

    ``current`` is the already-degraded reading. Poll ``request_reports`` up to
    ``_VIO_FEED_RECONFIRM_POLLS`` times, returning as soon as the feed recovers, so
    the caller only aborts on SUSTAINED degradation. No motion is issued -- the
    mower is already stopped, so this is purely a confirmation wait.
    """
    feed = current
    for _ in range(_VIO_FEED_RECONFIRM_POLLS):
        try:
            await coordinator.async_get_reports(count=5)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("vio_feed recheck refresh failed: %s", err)
        # Same pacing floor as the fresh-heading poll: never hammer request_reports.
        await asyncio.sleep(max(refresh_wait_seconds, 0.5))
        feed = _vio_feed_liveness(coordinator)
        if feed["live"]:
            break
    return feed


#: Rotation rate at the proven turn config (angular 500, refresh 200). Measured
#: live 2026-07-27: 1500 ms pulses gave 48.24 and 50.92 deg (~33 deg/s) and a
#: 700 ms pulse gave 25.94 deg (~37 deg/s). Biased to the HIGH end on purpose --
#: overestimating the rate shortens the pulse, so the turn undershoots and takes
#: another pulse instead of overshooting and having to reverse. Only the
#: fallback; the turn prefers the rate it measures during the run.
_DEFAULT_TURN_DEGREES_PER_SECOND = 37.0
#: Floor for a scaled turn pulse.
#:
#: MEASURED 2026-08-09, closing the "NOT PROVEN" this constant carried since it
#: was written. Its old docstring asked for exactly one experiment -- step
#: `pulse_duration_ms` down at refresh 200 / angular 500 and find where measured
#: rotation stops tracking duration -- and it was finally run: 35 cadence-intact
#: pulses over 200-1711 ms across three sessions
#: (docs/evidence-turn-pulse-duration-sweep-*.json).
#:
#: 200 ms ACTUATES. Ten pulses at ~200 ms windows rotated 5.44-15.20 deg, every
#: one of them real motion. There is no actuation threshold anywhere near 400 ms,
#: so the old value was costing precision for nothing: it forced a minimum pulse
#: that sweeps ~11 deg at the measured rate, which is most of an 18 deg tolerance
#: band and is why small corrections overshot and oscillated.
#:
#: 200 is the schema minimum for `pulse_duration_ms`, so it cannot go lower
#: without widening that too, and nothing below 200 ms has been measured.
_MIN_SCALED_TURN_PULSE_MS = 200.0
#: Upper envelope on how far ONE turn pulse can sweep, as an affine function of
#: its delivered window: ``sweep <= SLOPE * seconds + OFFSET``. Used only to cap
#: how long a pulse may run. Raising either value weakens the guard; lowering
#: them is the safe direction.
#:
#: WHY AFFINE, AND NOT A RATE. beta31 bounded sweep as ``C * seconds`` with
#: C = 60 deg/s, on the theory that rotation is proportional to duration. It is
#: not. Measured 2026-08-09 over 35 cadence-intact pulses spanning 200-1711 ms
#: (docs/evidence-turn-pulse-duration-sweep-*.json), the best fit is
#:
#:     rotation = 33.18 deg/s * seconds + 4.63 deg      residual sd 5.23 deg
#:
#: A pure rate cannot bound that shape: the C needed to cover the worst case is
#: ~110 deg/s at 0.2 s and ~43 deg/s at 1.5 s, so any single constant is either
#: useless at long pulses or ruinous at short ones. C = 60 was both -- it
#: under-bounded short pulses, which is where the overshoots actually happened,
#: while over-restricting long ones and costing turn budget.
#:
#: HOW THE NUMBERS WERE CHOSEN. The envelope must bound every sample, not fit
#: them. At slope 40 the smallest offset that covers all 35 is 9.09 deg; 12 deg
#: is used, which clears the worst observed sample by 2.9 deg and sits 1.7 sd
#: above the best fit at 0.2 s rising to 3.4 sd at 1.5 s. It is deliberately an
#: envelope with margin rather than a regression line.
#:
#: WHAT CHANGED IN PRACTICE. At the tolerance edge the two forms agree almost
#: exactly (an 18 deg error at 18 deg tolerance permits 600 ms either way), so
#: the well-tested case is untouched. Above that the new bound is more permissive
#: -- 1259 ms against 1040 for the Gate 5 geometry -- because 60 deg/s badly
#: over-estimated the slope, and that headroom goes straight back into the turn
#: budget. Below it the bound is tighter, which is correct: that is where the
#: constant term dominates and where beta31's ceiling silently under-bounded.
#:
#: ⚠️ The per-pulse scatter this envelope covers is large and irreducible: ten
#: pulses at matched ~200 ms windows spread 5.44-15.20 deg, 2.79x, with duration,
#: cadence and direction all held constant. Rotation cannot be predicted from
#: duration to better than ~40% at p90. That is why the ESTIMATE in
#: `_turn_final_approach_pulse_ms` can only ever get near the target and this
#: bound, which does not depend on the estimate being right, carries the safety.
_VIO_TURN_SWEEP_BOUND_DEGREES_PER_SECOND = 40.0
_VIO_TURN_SWEEP_BOUND_OFFSET_DEGREES = 12.0


def _max_turn_pulse_ms_for_sweep(allowed_sweep_degrees: float) -> float:
    """Longest pulse whose worst-case sweep stays within ``allowed_sweep``.

    Inverts ``sweep <= SLOPE * seconds + OFFSET``. Returns 0.0 when the allowance
    is smaller than the offset -- i.e. when even the shortest possible pulse can
    sweep past it, so no duration is safe and the caller must decide between
    overshooting and not moving.
    """
    usable = allowed_sweep_degrees - _VIO_TURN_SWEEP_BOUND_OFFSET_DEGREES
    if usable <= 0:
        return 0.0
    return (usable / _VIO_TURN_SWEEP_BOUND_DEGREES_PER_SECOND) * 1000


def _turn_final_approach_pulse_ms(
    *,
    heading_error_degrees: float | None,
    heading_tolerance_degrees: float,
    observed_rotation_degrees: float,
    observed_rotation_ms: float,
    default_degrees_per_second: float,
    pulse_duration_ms: float,
    refresh_interval_ms: int,
) -> dict[str, Any]:
    """Shorten the last turn pulse so it lands on the heading instead of past it.

    The turn phase has the same granularity defect the linear phase had, for the
    same reason. Live 2026-07-27: with a 23.7 deg error remaining, a full 1500 ms
    pulse turned 50.9 deg -- a 27 deg overshoot -- and the next pulse had to
    reverse direction to recover. ``slow_threshold_degrees`` does not catch this:
    it is 15 deg, so a 23.7 deg error is *above* the threshold and takes the full
    pulse, and even the 700 ms slow pulse is ~26 deg at this rate.

    **This only works because of the refresh cadence.** Single-shot rotation is a
    fixed ~8-15 deg quantum regardless of duration, so scaling would do nothing;
    with refresh, rotation became proportional to duration (1500 ms -> ~33 deg/s,
    700 ms -> ~37 deg/s), which is what makes a scaled pulse meaningful. Hence the
    hard guard on ``refresh_interval_ms > 0``.

    Self-calibrating on a *rate* rather than a per-pulse figure, so samples taken
    at different pulse lengths stay comparable -- and so a scaled pulse is still a
    valid sample, unlike the linear case where short-by-design pulses had to be
    excluded from a per-pulse average. The rate is measured against each pulse's
    *delivered* window (``motion_refresh.elapsed_ms``), not its commanded duration:
    live 2026-08-08 those differed by up to 36% on a nominal 1500 ms pulse, and
    dividing by the commanded figure alone made the estimator read 20.31 deg/s for
    a pulse that really ran at 14.91.

    Two independent bounds apply, and they pull in opposite directions on purpose:

    * the *estimated* rate shortens the pulse to the angle that remains, which is
      an accuracy optimisation and can be wrong in either direction;
    * the affine SWEEP BOUND (``_VIO_TURN_SWEEP_BOUND_*``) caps the pulse so that
      even at the worst sweep ever measured for that duration, it cannot pass the
      far edge of tolerance. That is a safety bound, it only ever shortens, and it
      applies even when the estimate says a full pulse fits -- which is exactly
      the case that overshot on 2026-08-08.

    The estimate is the weak half and is known to be. Rotation cannot be
    predicted from duration to better than ~40% at p90: ten pulses at matched
    ~200 ms windows spread 5.44-15.20 deg with duration, cadence and direction
    all held constant (2026-08-09, 35 samples). The bound therefore carries the
    safety and the estimate only improves the landing.
    """
    info: dict[str, Any] = {
        "applied": False,
        "reason": None,
        "heading_error_degrees": heading_error_degrees,
        "heading_tolerance_degrees": heading_tolerance_degrees,
        "degrees_per_second": None,
        "degrees_per_second_source": None,
        "sweep_bound_degrees_per_second": _VIO_TURN_SWEEP_BOUND_DEGREES_PER_SECOND,
        "sweep_bound_offset_degrees": _VIO_TURN_SWEEP_BOUND_OFFSET_DEGREES,
        "max_allowed_sweep_degrees": None,
        "ceiling_pulse_duration_ms": None,
        "pulse_duration_ms": pulse_duration_ms,
    }
    if refresh_interval_ms <= 0:
        info["reason"] = "refresh_disabled_rotation_not_proportional_to_duration"
        return info
    if heading_error_degrees is None:
        info["reason"] = "heading_error_unknown"
        return info

    if observed_rotation_ms > 0 and observed_rotation_degrees > 0:
        rate = observed_rotation_degrees / (observed_rotation_ms / 1000)
        info["degrees_per_second_source"] = "observed"
    else:
        rate = default_degrees_per_second
        info["degrees_per_second_source"] = "default"
    info["degrees_per_second"] = round(rate, 4)

    if rate <= 0:
        info["reason"] = "no_usable_rotation_rate"
        return info

    # Safety bound, independent of the estimate: even at the worst sweep ever
    # measured for a pulse of this length, it must not pass the far edge of
    # tolerance. Landing anywhere inside the tolerance band ends the turn, so the
    # worst acceptable sweep is the remaining error plus the band.
    max_allowed_sweep = abs(heading_error_degrees) + heading_tolerance_degrees
    ceiling_ms = _max_turn_pulse_ms_for_sweep(max_allowed_sweep)
    info["max_allowed_sweep_degrees"] = round(max_allowed_sweep, 3)
    info["ceiling_pulse_duration_ms"] = round(ceiling_ms, 1)
    # The two safety bounds still conflict at a tight tolerance, and now there is
    # a second, sharper way to fail. THE FLOOR WINS in both cases, deliberately:
    # an overshoot is recoverable by the next pulse, whereas a pulse too short to
    # actuate makes no progress at all and walks the turn into
    # `no_heading_progress` with the budget spent. Both are surfaced rather than
    # resolved silently, because they mean the anti-overshoot guarantee does NOT
    # hold there.
    #
    # `sweep_exceeds_any_pulse` is the sharp one: when the whole allowance is
    # smaller than the bound's constant term, NO pulse duration is safe, because
    # even the shortest can sweep past. Dropping the actuation floor to 200 ms
    # made this reachable in practice -- it needs a tolerance under ~6 deg -- and
    # it is a genuinely different condition from merely wanting a short pulse.
    info["ceiling_below_actuation_floor"] = ceiling_ms < _MIN_SCALED_TURN_PULSE_MS
    info["sweep_exceeds_any_pulse"] = ceiling_ms <= 0

    needed_ms = (abs(heading_error_degrees) / rate) * 1000
    if needed_ms >= pulse_duration_ms:
        # The estimate says the mower cannot reach the target within this pulse.
        # That judgement is only as good as the estimate, and on 2026-08-08 it was
        # wrong by 2.2x on exactly this branch, so the ceiling still applies.
        if ceiling_ms >= pulse_duration_ms:
            info["reason"] = "cruising_full_pulse_fits"
            return info
        bounded = max(_MIN_SCALED_TURN_PULSE_MS, ceiling_ms)
        info.update(
            {
                "applied": True,
                "reason": "bounded_by_max_rate_ceiling",
                "pulse_duration_ms": round(bounded, 1),
            }
        )
        return info

    # The estimate wants a shorter pulse than the ceiling would allow; take
    # whichever is shorter, then apply the actuation floor.
    scaled = max(
        _MIN_SCALED_TURN_PULSE_MS,
        min(needed_ms, ceiling_ms, pulse_duration_ms),
    )
    info.update(
        {
            "applied": True,
            "reason": (
                "final_approach_scaled_to_remaining_angle"
                if needed_ms <= ceiling_ms
                else "bounded_by_max_rate_ceiling"
            ),
            "pulse_duration_ms": round(scaled, 1),
        }
    )
    return info


#: Conservative per-command turn-progress bounds for the pre-dispatch
#: feasibility check. Each constant is chosen in the FAIL-CLOSED direction:
#: rotation is under-estimated (refuses more) and translation is over-estimated
#: (refuses more). Anchored to retained evidence, not tuning:
#:
#: - 16.5 deg/s is the minimum observed rotation rate across the four
#:   refresh-200 / angular-500 / 1500 ms pulses of the failed Gate 4 retry
#:   (16.54 / 16.54 / 20.41 / 21.33 deg/s,
#:   docs/evidence-gate4-beta19-retry-real-result-20260803.json). The
#:   configured `turn_degrees_per_second` (37) is an optimistic estimate used
#:   for final-approach shortening and would have judged that 167.4 deg turn
#:   feasible; the guard must use the floor the hardware actually delivered.
#: - 8.0 deg/command is the low end of the proven ~8-15 deg single-shot
#:   rotation quantum (live 2026-07-18); without refresh, rotation is NOT
#:   proportional to pulse duration, so a per-command quantum replaces the
#:   rate model.
#: - 0.0028 m/deg bounds translation by the ANGLE turned, not by elapsed time.
#:   During an in-place turn the tracked point sweeps an arc about the true
#:   rotation centre, so translation = r * theta; a per-second bound is only
#:   equivalent at a constant rotation rate, and the measured rate varies
#:   16.5-49.6 deg/s. The pooled maximum over 13 refresh-200 pulses across two
#:   geometries (the Gate 4 retry, 2026-08-03, plus the four-turn daylight
#:   characterization, 2026-08-04) is 0.002410 m/deg, implying a 13.8 cm offset
#:   between the drive centre and the tracked point. 0.0026 keeps +7.9% margin
#:   while staying under the two binding over-refusal limits: 0.25/90 =
#:   0.002778 keeps a 90 deg L-path junction feasible at a 0.25 m cap, and
#:   0.5/170 = 0.00294 keeps the proven -170 deg turn feasible at the schema's
#:   0.5 m default. Margin above 0.002778 would refuse Gate 4's own geometry.
#:   This replaced an earlier 0.0403 m/s per-second bound that was both invalid
#:   (the characterization measured 0.071959 m/s, 4 of 9 pulses over it) and
#:   structurally wrong: multiplying it by a command count derived from the
#:   PESSIMISTIC rotation floor compounded two anti-correlated worst cases, and
#:   correcting the rate alone would have refused two turns that demonstrably
#:   succeeded (the +135 deg run estimated 0.540 m against an actual 0.029 m).
#:   Evidence: docs/evidence-turnchar-beta19-analysis-20260804.json.
#:   It remains a refresh-regime figure -- no single-shot per-degree evidence
#:   exists -- so without refresh the translation criterion is still left to
#:   the runtime displacement cap instead of estimated.
#:
#: LOWERED 16.5 -> 14.4 on 2026-08-09, closing an item that had been open since
#: beta32. Gate 5 attempt 5 measured 14.905 and 14.490 deg/s against delivered
#: windows -- both BELOW the 16.5 this constant claimed as a minimum -- but
#: lowering it then would have cost the 90 deg L-path junction, because the
#: ceiling-aware model needed 5 commands against a budget of 4. beta32 recorded
#: that as "a truthful floor and L-path junctions are mutually exclusive".
#:
#: That is no longer true. Dropping `_MIN_SCALED_TURN_PULSE_MS` to its measured
#: 200 ms and replacing the pure-rate ceiling with the measured affine sweep
#: bound both lengthen the modelled pulses, so a 90 deg junction now completes in
#: 4 commands at any rate down to 14.0 deg/s. The trade that forced the
#: optimistic value is gone, and the constant can finally say something true.
#:
#: Verified against every retained real turn at the new value: the four
#: 2026-08-04 characterization turns still admit at their 8-command budget, and
#: the failed Gate 4 retry's 167.4 deg still refuses at 4. ⚠️ The -170 deg
#: characterization turn now needs 8 of 8 commands, so it sits exactly at its
#: budget with no margin.
_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND = 14.4
#: The lowest rotation rate ever measured against a delivered window. Not used
#: to plan anything -- it exists so the staleness of the constant above is a
#: tested fact rather than a comment.
_VIO_TURN_MEASURED_MINIMUM_DEGREES_PER_SECOND = 14.490
_VIO_TURN_SINGLE_SHOT_DEGREES_PER_COMMAND = 8.0
_VIO_TURN_CONSERVATIVE_TRANSLATION_M_PER_DEGREE = 0.0026


def _vio_turn_commands_at_rate(
    *,
    initial_error_degrees: float,
    heading_tolerance_degrees: float,
    degrees_per_second: float,
    turn_degrees_per_second: float,
    pulse_duration_ms: float,
    slow_pulse_duration_ms: float,
    slow_threshold_degrees: float,
    refresh_interval_ms: int,
    command_ceiling: int = 64,
) -> tuple[int | None, list[float]]:
    """Count the commands a turn needs if the mower rotates at a fixed rate.

    Replays the executor's OWN pulse-length policy by calling the same
    `_turn_final_approach_pulse_ms` the turn loop calls, so the planning model
    cannot drift away from the thing it predicts. That drift is exactly what
    beta31 introduced: the ceiling shortens pulses as the error closes, while
    the old closed-form `ceil(required / (rate * full_pulse))` still assumed
    every command ran the full `pulse_duration_ms`. It therefore under-counted
    commands and admitted turns the executor could not finish -- the failure the
    guard exists to prevent, reintroduced from the other side.

    Two rates, and they are not the same thing:

    * ``degrees_per_second`` is what the MOWER is assumed to do physically. Pass
      the conservative floor; it is the pessimistic input.
    * ``turn_degrees_per_second`` is what the executor's ESTIMATOR believes
      before it has measured anything (the configured rate, 37 by default). It
      must be the configured value, not the floor: substituting the floor makes
      the modelled `needed_ms` larger, which biases pulses LONGER and the
      command count LOWER, i.e. fail-open.

    Returns ``(commands, pulse_durations)``; ``commands`` is None when the turn
    does not converge within ``command_ceiling``.
    """
    error = abs(float(initial_error_degrees))
    observed_rotation_degrees = 0.0
    observed_rotation_ms = 0.0
    pulses: list[float] = []
    for command in range(1, command_ceiling + 1):
        if error <= heading_tolerance_degrees:
            return command - 1, pulses
        # Mirrors the turn loop's own base-duration choice. The no-progress /
        # blind-feed slow cap is NOT modelled: it only shortens pulses, so
        # omitting it under-states the command count in the fail-open
        # direction -- but it fires on telemetry conditions no preflight can
        # know, and modelling it as always-on would refuse nearly everything.
        base_ms = (
            slow_pulse_duration_ms
            if error <= slow_threshold_degrees
            else pulse_duration_ms
        )
        pulse_ms = float(
            _turn_final_approach_pulse_ms(
                heading_error_degrees=error,
                heading_tolerance_degrees=heading_tolerance_degrees,
                observed_rotation_degrees=observed_rotation_degrees,
                observed_rotation_ms=observed_rotation_ms,
                default_degrees_per_second=turn_degrees_per_second,
                pulse_duration_ms=base_ms,
                refresh_interval_ms=refresh_interval_ms,
            )["pulse_duration_ms"]
        )
        swept = degrees_per_second * (pulse_ms / 1000)
        pulses.append(round(pulse_ms, 1))
        observed_rotation_degrees += swept
        observed_rotation_ms += pulse_ms
        error -= swept
        if error <= heading_tolerance_degrees:
            return command, pulses
    return None, pulses


def _vio_turn_budget_feasibility(
    *,
    initial_error_degrees: float,
    heading_tolerance_degrees: float,
    max_commands: int,
    pulse_duration_ms: float,
    motion_refresh_interval_ms: int,
    max_displacement_m: float,
    turn_degrees_per_second: float = _DEFAULT_TURN_DEGREES_PER_SECOND,
    slow_pulse_duration_ms: float = 700.0,
    slow_threshold_degrees: float = 15.0,
) -> dict[str, Any]:
    """Judge whether a VIO turn can finish inside its configured budget.

    Pure planning math -- no I/O, no coordinator. Gate 4 failed 2026-08-03
    because a 167.4 deg turn was dispatched against a 4-command budget that the
    observed rotation rate could never satisfy; the executor burned the budget
    and 0.185 m of translation before stopping `max_commands_reached`. This
    helper refuses such a turn BEFORE the first turn command, using
    evidence-bounded per-command progress instead of the optimistic configured
    rate. Refusing is the safe direction: the caller dispatches no motion.

    Rotation is bounded per command (a budget question) and translation per
    degree (a geometry question). Validated against every real turn on record:
    it refuses the failed Gate 4 segment and admits all four turns of the
    2026-08-04 daylight characterization, which succeeded at +45/-90/+135/-170
    degrees.

    Under refresh the command count REPLAYS THE EXECUTOR'S OWN PULSE POLICY
    (`_vio_turn_commands_at_rate`) rather than assuming every command runs the
    full `pulse_duration_ms`. beta31's overshoot ceiling shortens pulses as the
    error closes, so the closed-form count silently under-estimated: a 90 deg
    junction reads 4 commands, not 3, and the admitted band narrows from ~117
    to ~100 deg. Without refresh, rotation is a duration-independent quantum
    rather than a rate, the ceiling is inert, and the closed form still applies.

    Still only as good as `_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND`, which is
    known to be ~12% optimistic -- see that constant.
    """
    pulse_seconds = float(pulse_duration_ms) / 1000
    translation_per_degree: float | None
    translation_bound_source: str | None
    if motion_refresh_interval_ms > 0:
        per_command_rotation = _VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND * pulse_seconds
        rotation_bound_source = "conservative_observed_rate_with_refresh"
        translation_per_degree = _VIO_TURN_CONSERVATIVE_TRANSLATION_M_PER_DEGREE
        translation_bound_source = "conservative_observed_translation_per_degree"
    else:
        per_command_rotation = _VIO_TURN_SINGLE_SHOT_DEGREES_PER_COMMAND
        rotation_bound_source = "single_shot_rotation_quantum_floor"
        # No trustworthy single-shot translation figure exists; the runtime
        # displacement cap bounds it during execution instead.
        translation_per_degree = None
        translation_bound_source = None
    info: dict[str, Any] = {
        "feasible": True,
        "reason": None,
        "initial_error_degrees": round(abs(initial_error_degrees), 3),
        "heading_tolerance_degrees": heading_tolerance_degrees,
        "required_rotation_degrees": round(
            max(0.0, abs(initial_error_degrees) - heading_tolerance_degrees), 3
        ),
        "per_command_rotation_bound_degrees": round(per_command_rotation, 3),
        "rotation_bound_source": rotation_bound_source,
        "estimated_commands_needed": 0,
        "max_commands": max_commands,
        "translation_bound_m_per_degree": translation_per_degree,
        "translation_bound_source": translation_bound_source,
        "estimated_translation_m": 0.0 if translation_per_degree is not None else None,
        "max_displacement_m": max_displacement_m,
        "command_count_model": None,
        "modelled_pulse_durations_ms": None,
    }
    required = info["required_rotation_degrees"]
    if required <= 0:
        info["reason"] = "already_within_tolerance"
        return info
    if per_command_rotation <= 0 or max_commands < 1:
        info["feasible"] = False
        info["reason"] = "turn_budget"
        return info
    if motion_refresh_interval_ms > 0:
        # Replay the real pulse policy, ceiling included.
        modelled, pulses = _vio_turn_commands_at_rate(
            initial_error_degrees=initial_error_degrees,
            heading_tolerance_degrees=heading_tolerance_degrees,
            degrees_per_second=_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND,
            turn_degrees_per_second=turn_degrees_per_second,
            pulse_duration_ms=pulse_duration_ms,
            slow_pulse_duration_ms=slow_pulse_duration_ms,
            slow_threshold_degrees=slow_threshold_degrees,
            refresh_interval_ms=motion_refresh_interval_ms,
        )
        info["command_count_model"] = "executor_pulse_policy_replay"
        info["modelled_pulse_durations_ms"] = pulses
        # Non-convergence within the replay ceiling is refusal, not a fallback:
        # a turn the policy cannot close is exactly what this guard is for.
        needed = modelled if modelled is not None else max_commands + 1
    else:
        info["command_count_model"] = "single_shot_quantum_closed_form"
        needed = math.ceil(required / per_command_rotation)
    info["estimated_commands_needed"] = needed
    estimated_translation: float | None = None
    if translation_per_degree is not None:
        # Scale by the angle actually swept -- the full initial error, since the
        # mower must rotate through it to land inside tolerance. Deliberately
        # NOT scaled by `needed`: that count comes from the pessimistic rotation
        # floor, and a slow pulse covers fewer degrees and so drags less, so
        # multiplying the two worst cases compounds anti-correlated pessimism.
        estimated_translation = round(
            abs(initial_error_degrees) * translation_per_degree, 3
        )
        info["estimated_translation_m"] = estimated_translation
    if needed > max_commands:
        info["feasible"] = False
        info["reason"] = "turn_budget"
    elif estimated_translation is not None and (
        estimated_translation > max_displacement_m
    ):
        info["feasible"] = False
        info["reason"] = "translation_cap"
    else:
        info["reason"] = "within_budget"
    return info


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
    fresh_heading_timeout_seconds: float = 8.0,
    max_commands: int = 8,
    min_progress_degrees: float = 2.0,
    max_no_progress_pulses: int = 2,
    max_displacement_m: float = 0.5,
    invert_direction: bool = False,
    motion_refresh_interval_ms: int = 0,
    turn_degrees_per_second: float = _DEFAULT_TURN_DEGREES_PER_SECOND,
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
    then measures ``vision_heading`` and repeats until within tolerance. The VIO
    feed lags the command by ~4s (longer than ``refresh_wait_seconds``), so after
    the stop the loop polls ``request_reports`` until the heading moves off the
    pre-pulse value (up to ``fresh_heading_timeout_seconds``) before judging
    progress, and only aborts on ``max_no_progress_pulses`` consecutive
    no-progress pulses -- a single stale sample no longer ends the turn. Pulses
    fired during a no-progress streak whose last sample was stale (the heading
    never went fresh) use ``slow_pulse_duration_ms`` so a latched feed cannot
    drive long full-power rotations blind; a streak with a fresh-but-stalled
    reading keeps the full pulse.
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
    initial_vio_state = initial_reading["vio_state"]
    initial_feed = _vio_feed_liveness(coordinator)
    if initial_heading is None:
        gates.append(
            {
                "name": "vio_heading_available",
                "passed": dry_run,
                "detail": "VIO turn-to-heading requires live vision_info.heading.",
            }
        )
    if initial_vio_state != _VIO_STATE_ACTIVE:
        # VIO is visual odometry: it will not initialise in a dark scene, and a
        # cold reading reports vision_heading=0.0 as a *valid* float. Refuse to
        # start a real turn unless VIO is actively tracking (vio_state == 2);
        # dry-run planning is still allowed so the command can be inspected cold.
        gates.append(
            {
                "name": "vio_active",
                "passed": dry_run,
                "detail": (
                    "VIO turn-to-heading requires an active VIO track "
                    f"(vio_state == {_VIO_STATE_ACTIVE}); saw {initial_vio_state}. "
                    "Warm VIO with forward motion in daylight "
                    "(camera_brightness must not be Dark) before turning."
                ),
            }
        )
    if not initial_feed["live"]:
        # vio_state can read active while the track is blind (feature count
        # collapsed at dusk); refuse the real turn and name the brightness so an
        # operator can tell a dark scene from a mower that just is not rotating.
        gates.append(_vio_feed_live_gate(initial_feed, dry_run=dry_run))
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
    planned_angular = (
        _planned_angular(initial_error) if initial_error is not None else None
    )
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
        "refresh_wait_seconds": refresh_wait_seconds,
        # The poll loop paces at this floored value; report it so iteration
        # math from the result matches actual behavior.
        "effective_poll_interval_seconds": max(refresh_wait_seconds, 0.5),
        "fresh_heading_timeout_seconds": fresh_heading_timeout_seconds,
        "max_commands": max_commands,
        "min_progress_degrees": min_progress_degrees,
        "max_no_progress_pulses": max_no_progress_pulses,
        "max_displacement_m": max_displacement_m,
        "invert_direction": invert_direction,
        # App-parity refresh for the TURN phase. Proven live 2026-07-25 at
        # angular 500: a 4s pulse turned +9 deg single-shot vs +62 deg at
        # interval 200 (~7x, compass ground truth). Refresh is SPEED-GATED --
        # it did nothing at angular 180, which is below this mower's rotation
        # threshold -- so it only helps when the angular speed already actuates.
        # Left opt-in (0 == the proven single-shot path) because
        # `heading_tolerance_degrees: 18` was derived from the ~8-15 deg
        # single-shot quantum and has NOT been re-derived against continuous
        # rotation; turning refresh on without lowering the tolerance risks
        # overshooting a deadband that is now far too wide.
        "motion_refresh_interval_ms": motion_refresh_interval_ms,
        "motion_refresh_commands_sent": 0,
        # Pulses whose refresh cadence collapsed, so they were EXCLUDED from the
        # rotation-rate estimate. A nonzero count means BLE write latency, not
        # the mower, is shaping this turn -- read it before concluding anything
        # about rotation speed from the run.
        "refresh_cadence_broken_pulses": 0,
        "prefer_ble": prefer_ble,
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        "active_transport": _active_transport_label(coordinator),
        "initial_vision_heading": initial_heading,
        "initial_vio_state": initial_reading["vio_state"],
        "initial_vio_feed": initial_feed,
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
        # Always present (like final_vision_heading); the vio_feed_degraded stop
        # path overwrites it. Avoids a KeyError for consumers on other stops.
        "final_vio_feed": initial_feed,
        # 0.0 rather than None: a turn that sends no commands genuinely did not
        # translate, and None reads as "not measured" (the 2026-07-19 bug).
        "final_displacement_m": 0.0,
        "turn_degrees_per_second": turn_degrees_per_second,
        # Populated whenever the initial heading error is known (including
        # dry-run) so previews expose the same budget math the real path
        # enforces. None only when there is no heading to judge.
        "turn_feasibility": None,
        "stop_reason": None,
    }
    if initial_error is not None and abs(initial_error) <= heading_tolerance_degrees:
        result["stop_reason"] = "target_heading_reached"
        return result
    if initial_error is not None:
        result["turn_feasibility"] = _vio_turn_budget_feasibility(
            initial_error_degrees=float(initial_error),
            heading_tolerance_degrees=heading_tolerance_degrees,
            max_commands=max_commands,
            pulse_duration_ms=float(pulse_duration_ms),
            motion_refresh_interval_ms=motion_refresh_interval_ms,
            max_displacement_m=max_displacement_m,
            # This loop's own pulse-policy inputs, so the model replays what
            # this call will actually do rather than the helper's defaults.
            turn_degrees_per_second=turn_degrees_per_second,
            slow_pulse_duration_ms=float(slow_pulse_duration_ms),
            slow_threshold_degrees=float(slow_threshold_degrees),
        )
    if dry_run:
        result["stop_reason"] = "dry_run"
        return result
    if blockers:
        result["stop_reason"] = "safety_gates_failed"
        return result
    if not _transport_is_ble(coordinator):
        result["stop_reason"] = "ble_not_active_at_fire"
        return result
    # Fail-closed budget preflight (Gate 4 retry, 2026-08-03): refuse a turn the
    # evidence-bounded per-command progress cannot finish within `max_commands`
    # and the displacement cap, instead of dispatching pulses that provably end
    # in `max_commands_reached` after real rotation and translation. Zero turn
    # commands are sent on this path.
    feasibility = result["turn_feasibility"]
    if feasibility is not None and not feasibility["feasible"]:
        result["would_send"] = False
        result["stop_reason"] = "turn_budget_infeasible"
        return result

    consecutive_no_progress = 0
    last_heading_went_fresh = True
    last_progress_degrees = 0.0
    # Rotation rate measured during this run, tracked as degrees and milliseconds
    # separately so pulses of different lengths stay comparable.
    observed_rotation_degrees = 0.0
    observed_rotation_ms = 0.0
    for command_index in range(1, max_commands + 1):
        before_telemetry = _custom_path_telemetry_snapshot(coordinator)
        before_reading = _vio_reading(coordinator)
        before_heading = before_reading["vision_heading"]
        if before_heading is None:
            result["stop_reason"] = "vio_heading_unavailable"
            return result
        if before_reading["vio_state"] != _VIO_STATE_ACTIVE:
            # VIO dropped out mid-turn (e.g. drove into shadow); its heading is no
            # longer trustworthy, so stop rather than chase a stale reading.
            result["stop_reason"] = "vio_inactive"
            return result
        # Pulse 1's feed was already proven live by the entry gate (a real run
        # can't get here otherwise, and nothing refreshes in between), so reuse it
        # rather than re-read; later pulses re-check live.
        before_feed = (
            initial_feed if command_index == 1 else _vio_feed_liveness(coordinator)
        )
        if not before_feed["live"]:
            # vio_state stayed active but the feature track collapsed (dusk): the
            # heading is a stale latch. A single degraded read may be a transient
            # dip, so re-poll (read-only) before aborting; only a SUSTAINED blind
            # feed stops the turn, with a distinct reason so the operator sees
            # "blind feed", not "not rotating".
            before_feed = await _reconfirm_vio_feed_degraded(
                coordinator, before_feed, refresh_wait_seconds=refresh_wait_seconds
            )
            if not before_feed["live"]:
                result["final_vio_feed"] = before_feed
                result["stop_reason"] = "vio_feed_degraded"
                return result
        if not _blade_reported_safe(before_telemetry):
            result["stop_reason"] = "aborted_unsafe_blade"
            return result
        if before_telemetry.get("work_mode_label") not in {"MODE_READY", "MODE_PAUSE"}:
            result["stop_reason"] = "aborted_unsafe_mode"
            return result
        if prefer_ble and not _transport_is_ble(coordinator):
            result["stop_reason"] = "ble_transport_lost"
            return result
        error = _heading_error_degrees(float(before_heading), target)
        if abs(error) <= heading_tolerance_degrees:
            result["final_vision_heading"] = before_heading
            result["final_heading_error_degrees"] = round(error, 3)
            result["stop_reason"] = "target_heading_reached"
            return result
        # Cap the pulse at the slow duration during a no-progress streak when the
        # last sample was stale (blind/latched feed) OR the last pulse moved AWAY
        # from the target (negative progress -- e.g. an angular sign miscalibration
        # or mechanical fault turning the wrong way): both are cases where a
        # full-length pulse risks a long wrong/blind rotation. A fresh streak that
        # was still moving toward the target (merely slowly) keeps the full pulse.
        base_pulse_ms = float(
            slow_pulse_duration_ms
            if (
                abs(error) <= slow_threshold_degrees
                or (
                    consecutive_no_progress > 0
                    and (not last_heading_went_fresh or last_progress_degrees < 0)
                )
            )
            else pulse_duration_ms
        )
        # Then scale to the angle that actually remains. This is applied on top of
        # the cap above, never instead of it: it can only shorten the pulse the
        # safety logic already chose, so a no-progress or blind-feed streak can
        # never be lengthened back to a full pulse.
        turn_approach = _turn_final_approach_pulse_ms(
            heading_error_degrees=error,
            heading_tolerance_degrees=heading_tolerance_degrees,
            observed_rotation_degrees=observed_rotation_degrees,
            observed_rotation_ms=observed_rotation_ms,
            default_degrees_per_second=turn_degrees_per_second,
            pulse_duration_ms=base_pulse_ms,
            refresh_interval_ms=motion_refresh_interval_ms,
        )
        pulse_ms = float(turn_approach["pulse_duration_ms"])
        angular = _planned_angular(error)
        command_result: dict[str, Any] = {
            "index": command_index,
            "angular_speed": angular,
            "pulse_duration_ms": pulse_ms,
            "before_vision_heading": before_heading,
            "heading_error_before": round(error, 3),
            "command": "send_movement",
            "sent_at_utc": _utc_timestamp(),
            "ok": None,
            "error": None,
            "stop_ack": None,
            "after_vision_heading": None,
            "measured_change_degrees": None,
            "heading_error_after": None,
            "progress_degrees": None,
            "displacement_m": None,
            "heading_poll_seconds": None,
            "heading_went_fresh": None,
            "motion_refresh": None,
            "final_approach": turn_approach,
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
        # Bounded pulse, then a mandatory explicit stop before sampling. With a
        # positive refresh interval the identical command is re-sent every
        # interval for the length of the pulse (app parity); at 0 this is
        # exactly the previous single-shot sleep. Refreshes are counted
        # separately so they never inflate `commands_sent`, which drives
        # `max_commands`.
        command_result["motion_refresh"] = await _motion_refresh_window(
            coordinator,
            resend=functools.partial(
                _send_manager_command_with_args,
                coordinator,
                "send_movement",
                prefer_ble=prefer_ble,
                command_kwargs={"linear_speed": 0, "angular_speed": angular},
            ),
            duration_seconds=pulse_ms / 1000,
            refresh_interval_ms=motion_refresh_interval_ms,
        )
        result["motion_refresh_commands_sent"] += command_result["motion_refresh"][
            "refresh_commands_sent"
        ]
        try:
            command_result["stop_ack"] = await _stop_manual_motion_confirmed(
                coordinator
            )
        except Exception as err:  # noqa: BLE001
            # Never keep turning when stops are not deliverable (live 2026-07-12:
            # BLE connect cooldown raised mid-run and motion continued unstopped).
            command_result["stop_ack"] = {"error": f"{type(err).__name__}: {err}"}
            result["command_results"].append(command_result)
            result["stop_reason"] = "stop_failed_aborting"
            return result
        # The VIO feed lags the command by ~4s -- longer than refresh_wait_seconds
        # -- so the first post-stop sample can be bit-identical to before_heading:
        # a stale reading, not a true absence of rotation (live 2026-07-12: a turn
        # aborted on no_heading_progress against an unchanged 78.93922636 heading).
        # Poll request_reports until the heading moves off the pre-pulse value or a
        # bounded timeout, so progress is judged against a fresh reading.
        poll_started = time.monotonic()
        after_telemetry = before_telemetry
        after_heading = before_heading
        heading_went_fresh = False
        # Liveness evidence gathered DURING the poll. `heading_went_fresh` cannot
        # serve as a stale-feed discriminator: it is True only when before/after
        # differ by more than the epsilon, which is exactly when
        # `_streak_shows_no_actuation` (bit-identical heading) is False -- the
        # two are perfectly correlated, so gating on it would delete the
        # no-actuation branch rather than refine it.
        #
        # The independent signal is "did ANY channel move at all". A live feed is
        # never perfectly still: position jitters ~2-4mm between consecutive
        # reads even on a stationary mower, and a VIO heading latched by dusk
        # still emits sub-epsilon sensor noise (~0.0018 deg, live run 2
        # 2026-07-15). Either one proves reports are still arriving. Only when
        # heading AND position are bit-identical across every poll has the
        # report stream itself stopped.
        poll_count = 0
        poll_feed_alive = False
        previous_poll_telemetry = before_telemetry
        previous_poll_heading = before_heading
        while True:
            try:
                await coordinator.async_get_reports(count=5)
            except Exception as err:  # noqa: BLE001
                LOGGER.debug("vio_turn_to_heading refresh failed: %s", err)
            # Floor the pacing even at refresh_wait_seconds=0: back-to-back
            # request_reports would flood the BLE command queue that also has
            # to deliver motion stops.
            await asyncio.sleep(max(refresh_wait_seconds, 0.5))
            after_telemetry = _custom_path_telemetry_snapshot(coordinator)
            poll_count += 1
            poll_step = _telemetry_position_delta(
                previous_poll_telemetry, after_telemetry
            ).get("distance")
            if poll_step is not None and float(poll_step) > 0.0:
                poll_feed_alive = True
            previous_poll_telemetry = after_telemetry
            after_heading = _vio_reading(coordinator)["vision_heading"]
            # Any change at all -- including one far below the freshness epsilon
            # -- means the heading channel is still being written to.
            if (
                after_heading is not None
                and previous_poll_heading is not None
                and float(after_heading) != float(previous_poll_heading)
            ):
                poll_feed_alive = True
            previous_poll_heading = after_heading
            if after_heading is None:
                break
            if (
                abs(_heading_error_degrees(float(before_heading), float(after_heading)))
                > _VIO_HEADING_FRESH_EPSILON_DEGREES
            ):
                heading_went_fresh = True
                break
            if (time.monotonic() - poll_started) >= fresh_heading_timeout_seconds:
                break
        command_result["heading_poll_seconds"] = round(
            time.monotonic() - poll_started, 2
        )
        command_result["heading_went_fresh"] = heading_went_fresh
        command_result["heading_poll_count"] = poll_count
        command_result["heading_poll_feed_alive"] = poll_feed_alive
        last_heading_went_fresh = heading_went_fresh
        command_result["after_vision_heading"] = after_heading
        if after_heading is None:
            result["command_results"].append(command_result)
            result["stop_reason"] = "vio_heading_unavailable"
            return result
        measured_change = _heading_error_degrees(
            float(before_heading), float(after_heading)
        )
        new_error = _heading_error_degrees(float(after_heading), target)
        progress = abs(error) - abs(new_error)
        # Remembered for the next pulse's slow-cap decision: a negative value means
        # this pulse moved away from the target (wrong direction).
        last_progress_degrees = progress
        displacement = _telemetry_position_delta(
            initial_telemetry, after_telemetry
        ).get("distance")
        command_result["measured_change_degrees"] = round(measured_change, 3)
        command_result["heading_error_after"] = round(new_error, 3)
        command_result["progress_degrees"] = round(progress, 3)
        command_result["displacement_m"] = displacement
        # A pulse whose refresh cadence collapsed did not rotate for the window
        # it was billed for, so it cannot be used to measure a rotation RATE.
        #
        # Motion continues only while refresh writes keep arriving -- that is the
        # entire reason the app re-sends every 200 ms and the reason
        # `_motion_refresh_window` exists. When a single BLE write blocks for as
        # long as the whole commanded pulse, no refresh reached the mower for
        # that span, its device-side watchdog stopped the motor, and the window
        # is mostly dead time.
        #
        # Live 2026-08-09 (docs/evidence-beta33-reposition-20260809T184618Z.json,
        # segment 3 pulse 1): a 1303.7 ms pulse at a 200 ms interval sent ONE of
        # a possible six refreshes, and that write took 1303.972 ms. It measured
        # 13.885 deg over a 1504 ms window -- "9.23 deg/s", which would have been
        # the slowest rotation ever recorded and 44% under
        # `_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND`. It is not a rotation rate
        # at all. Every other turn pulse that day, with an intact cadence,
        # measured 23-43 deg/s.
        #
        # This matters beyond one number, because a low estimate LENGTHENS later
        # pulses: the estimator learns "slow", judges that a full pulse cannot
        # reach the target, takes the `cruising_full_pulse_fits` branch, and then
        # a pulse with a healthy cadence rotates at the true rate and overshoots.
        # That is precisely the Gate 5 attempt 5 sequence -- two stall-degraded
        # pulses at ~14.7 followed by a clean one at 32.74 that overshot by
        # 13.258 deg -- so the outlier the overshoot ceiling was built to contain
        # is substantially an artefact of this accounting.
        #
        # The test is deliberately free of a tuned constant: one write lasting
        # the entire commanded pulse means the cadence definitionally did not
        # exist. Writes of 78-820 ms coexist with perfectly normal rates and are
        # NOT excluded. Narrow is the right bias -- discarding a good sample
        # shrinks an already small estimator population.
        refresh_report = command_result.get("motion_refresh") or {}
        write_durations = refresh_report.get("refresh_write_durations_ms") or []
        longest_write_ms = max(write_durations) if write_durations else None
        refresh_cadence_broken = (
            longest_write_ms is not None and longest_write_ms >= pulse_ms
        )
        command_result["longest_refresh_write_ms"] = longest_write_ms
        command_result["refresh_cadence_broken"] = refresh_cadence_broken
        if refresh_cadence_broken:
            result["refresh_cadence_broken_pulses"] += 1
        # Feed the rate estimate, but only from a pulse whose heading reading
        # actually went fresh. A stale/latched sample measures ~0 deg of rotation
        # for a pulse that really turned, and folding that in would collapse the
        # rate and over-lengthen every later pulse -- the exact failure the
        # scaling exists to prevent.
        if heading_went_fresh and not refresh_cadence_broken:
            observed_rotation_degrees += abs(measured_change)
            # Accumulate the DELIVERED window, not the commanded one. BLE write
            # latency routinely runs a nominal 1500 ms pulse long -- live
            # 2026-08-08 the three turn pulses of Gate 5 attempt 5 measured
            # 2043 / 1530 / 1760 ms -- and dividing by the nominal figure made the
            # estimator report 20.31 deg/s for a pulse that really ran at 14.91.
            # `motion_refresh` is populated just above, so elapsed_ms is in hand;
            # fall back to the commanded duration only if it is missing.
            elapsed_ms = (command_result.get("motion_refresh") or {}).get("elapsed_ms")
            observed_rotation_ms += (
                float(elapsed_ms) if elapsed_ms is not None else pulse_ms
            )
        result["command_results"].append(command_result)
        result["final_vision_heading"] = after_heading
        result["final_heading_error_degrees"] = round(new_error, 3)
        # Cumulative translation during the turn. Populated on every pulse so the
        # aggregate never reports None while the per-command values show real
        # movement (the 2026-07-19 honesty bug, still live on this path as of
        # 2026-07-27).
        if displacement is not None:
            result["final_displacement_m"] = displacement
        if displacement is not None and displacement > max_displacement_m:
            result["stop_reason"] = "aborted_displacement_cap"
            return result
        if not _blade_reported_safe(after_telemetry):
            result["stop_reason"] = "aborted_unsafe_blade"
            return result
        if abs(new_error) <= heading_tolerance_degrees:
            result["stop_reason"] = "target_heading_reached"
            return result
        if progress < min_progress_degrees:
            # Tolerate a single stale/latched sample; only abort once the turn has
            # made no measurable progress on max_no_progress_pulses in a row.
            consecutive_no_progress += 1
            command_result["consecutive_no_progress"] = consecutive_no_progress
            if consecutive_no_progress >= max_no_progress_pulses:
                if _streak_shows_dead_telemetry(
                    result["command_results"], max_no_progress_pulses
                ):
                    # No positive evidence the feed was alive, so we cannot say
                    # anything about actuation. Name the blindness instead of
                    # blaming the mower (live 2026-07-25: reported exactly this
                    # signature while physically turning ~4 inches).
                    result["stop_reason"] = "vio_telemetry_stream_stale"
                    result["vio_telemetry_stream_stale_hint"] = (
                        "Heading and position were bit-identical across every "
                        "refresh poll of the last "
                        f"{max_no_progress_pulses} pulses (a live feed jitters "
                        "~2-4mm and the heading poll goes fresh). The report "
                        "stream stopped updating, so whether the mower turned "
                        "is unknown -- check BLE transport health and the "
                        "server log for dropped/malformed frames rather than "
                        "retuning the turn."
                    )
                elif _streak_shows_no_actuation(
                    result["command_results"], max_no_progress_pulses
                ):
                    # Neither heading nor position moved: the commands were
                    # accepted but nothing actuated. Name the real failure so the
                    # operator checks the mower instead of retuning the turn.
                    result["stop_reason"] = "no_actuation_detected"
                    result["no_actuation_hint"] = (
                        "Commands were accepted (and the stop ACKed) but neither "
                        "heading nor position changed. Check the mower's physical "
                        "e-stop -- it is invisible in telemetry -- and whether the "
                        "BLE transport is actually usable."
                    )
                else:
                    result["stop_reason"] = "no_heading_progress"
                return result
            continue
        consecutive_no_progress = 0

    result["stop_reason"] = "max_commands_reached"
    return result


#: Largest rotation attempted in one stage of a staged turn, in degrees.
#: 60 is not a guess -- every validation run since 2026-08-09 has driven three
#: 60 deg junctions, and `--reposition` accumulated a full 180 deg out of three
#: of them on 2026-08-09. It is the most-exercised turn magnitude on this
#: hardware.
_STAGED_TURN_STAGE_DEGREES = 60.0
#: Stages allowed before giving up. A heading error is normalised to +/-180, so
#: at 60 deg a stage three always suffices; the fourth is slack for a stage that
#: under-rotates.
_MAX_TURN_STAGES = 4


async def _vio_turn_to_heading_staged(
    coordinator: MammotionReportUpdateCoordinator,
    *,
    target_vision_heading: float,
    heading_tolerance_degrees: float,
    max_displacement_m: float,
    **turn_kwargs: Any,
) -> dict[str, Any]:
    """Turn to a vision heading, splitting a rotation too large to dispatch.

    A single ``_vio_turn_to_heading`` call is refused pre-dispatch when the
    rotation needs more than ``max_commands`` pulses or more translation than
    ``max_displacement_m``. Measured 2026-08-09: a 180 deg turn wants 8 commands
    against a budget of 4 and 0.468 m of drift against a 0.30 m cap, and the
    largest single turn that dispatches is ~114 deg.

    **Chaining works where one call does not, and that is not a loophole.** Each
    call gets its own command budget and its own displacement allowance, which is
    exactly why three chained 60 deg junctions accumulate 180 deg on hardware
    (`--reposition`, 2026-08-09) while one 180 deg turn is refused. This makes the
    same decomposition available to a segment's OPENING turn, which is the one
    rotation nothing can preflight -- no feed reports a stationary mower's
    orientation, so a user clicking a point behind the mower could only discover
    the problem as a `turn_budget_infeasible` refusal.

    Deliberately tries the direct turn FIRST and only decomposes on a
    ``turn_budget_infeasible`` refusal, which sends zero commands. Any other
    failure (stale feed, lost transport, no progress) is returned untouched --
    those mean something is wrong, and retrying a broken turn in smaller pieces
    would just fail more slowly.

    The residual is left to the caller's post-turn alignment gate. Stages rotate
    toward a FIXED vision heading and do not re-derive the bearing to the target,
    so the translation they accumulate leaves a map-frame aim error behind. That
    is precisely what `post_turn_alignment` measures and corrects, so re-deriving
    here would duplicate it.

    ⚠️ Translation is budgeted across the WHOLE staged turn, not per stage. Each
    stage is given only what is left, so a staged turn may not translate further
    than a single turn was allowed to. Without that, four stages could drift four
    times the cap while every individual call looked compliant.
    """
    direct = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=target_vision_heading,
        heading_tolerance_degrees=heading_tolerance_degrees,
        max_displacement_m=max_displacement_m,
        **turn_kwargs,
    )
    if direct.get("stop_reason") != "turn_budget_infeasible":
        return direct

    staged: dict[str, Any] = {
        "staged_turn": True,
        "stage_degrees": _STAGED_TURN_STAGE_DEGREES,
        "max_stages": _MAX_TURN_STAGES,
        "direct_refusal": direct.get("turn_feasibility"),
        "stages": [],
        "commands_sent": 0,
        "motion_refresh_commands_sent": 0,
        "command_results": [],
        "samples": [],
        "final_displacement_m": 0.0,
        "turn_feasibility": direct.get("turn_feasibility"),
        "stop_reason": None,
    }
    displacement_budget = float(max_displacement_m)

    for _ in range(_MAX_TURN_STAGES):
        reading = _vio_reading(coordinator)
        current = reading.get("vision_heading")
        if reading.get("vio_state") != _VIO_STATE_ACTIVE or current is None:
            staged["stop_reason"] = "staged_turn_vio_unavailable"
            return staged
        remaining = _heading_error_degrees(float(current), target_vision_heading)
        if abs(remaining) <= heading_tolerance_degrees:
            staged["stop_reason"] = "target_heading_reached"
            return staged
        if displacement_budget <= 0:
            staged["stop_reason"] = "staged_turn_translation_budget_exhausted"
            return staged
        step = max(
            -_STAGED_TURN_STAGE_DEGREES,
            min(_STAGED_TURN_STAGE_DEGREES, remaining),
        )
        stage_target = _normalized_heading_degrees(float(current) + step)
        stage = await _vio_turn_to_heading(
            coordinator,
            target_vision_heading=float(stage_target or 0.0),
            heading_tolerance_degrees=heading_tolerance_degrees,
            max_displacement_m=displacement_budget,
            **turn_kwargs,
        )
        moved = float(stage.get("final_displacement_m") or 0.0)
        displacement_budget -= moved
        staged["commands_sent"] += int(stage.get("commands_sent") or 0)
        staged["motion_refresh_commands_sent"] += int(
            stage.get("motion_refresh_commands_sent") or 0
        )
        staged["command_results"].extend(stage.get("command_results") or [])
        staged["samples"].extend(stage.get("samples") or [])
        staged["final_displacement_m"] += moved
        staged["stages"].append(
            {
                "target_vision_heading": round(float(stage_target or 0.0), 3),
                "remaining_before_degrees": round(remaining, 3),
                "step_degrees": round(step, 3),
                "stop_reason": stage.get("stop_reason"),
                "commands_sent": int(stage.get("commands_sent") or 0),
                "displacement_m": round(moved, 4),
            }
        )
        if stage.get("stop_reason") == "turn_budget_infeasible":
            # A stage this small being refused too means the rotation was never
            # the problem -- the budget cannot dispatch ANY turn, so slicing it
            # finer cannot help. Report the original refusal rather than dressing
            # it up as a staging failure, and keep the feasibility math that says
            # why. Pinned by test_vector_segment_surfaces_the_refusal_stop_reason.
            staged["stop_reason"] = "turn_budget_infeasible"
            staged["turn_feasibility"] = stage.get("turn_feasibility") or direct.get(
                "turn_feasibility"
            )
            staged["staging_cannot_help"] = True
            return staged
        if stage.get("stop_reason") != "target_heading_reached":
            staged["stop_reason"] = "staged_turn_stage_failed"
            staged["failed_stage_reason"] = stage.get("stop_reason")
            return staged

    reading = _vio_reading(coordinator)
    current = reading.get("vision_heading")
    if current is not None and abs(
        _heading_error_degrees(float(current), target_vision_heading)
    ) <= float(heading_tolerance_degrees):
        staged["stop_reason"] = "target_heading_reached"
    else:
        staged["stop_reason"] = "staged_turn_stages_exhausted"
    return staged


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


async def _settle_linear_position_feed(
    coordinator: MammotionReportUpdateCoordinator,
    before_telemetry: dict[str, Any],
    *,
    timeout_seconds: float = _LINEAR_POSITION_SETTLE_TIMEOUT_SECONDS,
    poll_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    """Poll the position feed until this pulse's motion has settled.

    The map-local x/y feed lags ~4s and updates in jumps, so a reading taken right
    after a stop can either miss this pulse's motion or catch a prior pulse's
    delayed jump (live 2026-07-15). Settling requires BOTH that the feed has moved
    off ``before_telemetry`` (the motion registered -- a still-stale feed reads as
    unchanged and would otherwise look falsely "settled") AND that two consecutive
    snapshots agree within ``_LINEAR_POSITION_SETTLE_EPSILON_M`` (it stopped
    jumping). A pulse that truly produced no motion never registers movement and
    times out with ``settled=False`` so the caller can treat it as no-progress.

    Limitation: this fixes attribution for LAG (waiting out the delayed jump) but
    NOT for PHANTOM motion -- the feed can report a jump the mower did not make
    (live 2026-07-15: a physical no-op pulse showed ~9cm), which clears the epsilon
    and reads as settled=moved=True. Distinguishing phantom from real needs the
    dual-source data captured by ``_position_source_comparison``.
    """
    started = time.monotonic()
    # Bound by poll count, not wall clock: the pacing sleep does the timing, so a
    # test that stubs asyncio.sleep to a no-op still terminates in a fixed number
    # of iterations instead of spinning against a real monotonic clock.
    max_polls = max(1, round(timeout_seconds / poll_interval_seconds))
    previous = _custom_path_telemetry_snapshot(coordinator)
    moved = False
    settled = False
    # A LIVE position feed always carries sensor noise: consecutive reads of a
    # stationary mower differ by ~2-4mm. A feed that returns bit-identical
    # coordinates poll after poll is not reporting stillness, it has stopped
    # updating. Track that separately so the caller can tell "the mower stopped"
    # from "we went blind" -- they look identical in a single before/after
    # comparison but need opposite responses.
    polls = 0
    observed_jitter = False
    for _ in range(max_polls):
        await asyncio.sleep(poll_interval_seconds)
        try:
            await coordinator.async_get_reports(count=5)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("linear position settle refresh failed: %s", err)
        current = _custom_path_telemetry_snapshot(coordinator)
        step = _telemetry_position_delta(previous, current).get("distance")
        total = _telemetry_position_delta(before_telemetry, current).get("distance")
        previous = current
        polls += 1
        if step is not None and float(step) > 0.0:
            observed_jitter = True
        if total is not None and total > _LINEAR_POSITION_SETTLE_EPSILON_M:
            moved = True
        if moved and step is not None and step <= _LINEAR_POSITION_SETTLE_EPSILON_M:
            settled = True
            break
    # Only claim staleness on the evidence we actually have: several polls that
    # all returned the exact same coordinates. One poll proves nothing.
    feed_stale = polls >= _STALE_FEED_MIN_POLLS and not observed_jitter
    return {
        # The settled snapshot is also the VIO path's authoritative post-pulse
        # sample. Reusing it avoids waiting through `sample_delays` after this
        # loop has already established both movement and stillness.
        "telemetry": previous,
        "settled": settled,
        "moved": moved,
        "feed_stale": feed_stale,
        "observed_jitter": observed_jitter,
        "settle_polls": polls,
        "wait_seconds": round(time.monotonic() - started, 2),
    }


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
    linear_pulse_duration_ms: float = 300.0,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60),
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute or dry-run one calibrated raw Y-axis segment.

    Each real pulse is bounded by ``linear_pulse_duration_ms`` and followed by
    an explicit software stop (mirroring the vector executor) instead of
    relying on firmware auto-stop; an undeliverable stop aborts the run
    (``stop_failed_aborting``).
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
        "linear_pulse_duration_ms": linear_pulse_duration_ms,
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
        await _motion_open_sleep(coordinator, linear_pulse_duration_ms / 1000)
        command_result["stop_result"] = await _manual_velocity_stop_attempt(
            coordinator, use_wifi=not prefer_ble
        )
        if not (command_result["stop_result"] or {}).get("ok"):
            # Never keep driving when stops are not deliverable (BLE cooldown,
            # transport loss); abort immediately.
            result["stop_reason"] = "stop_failed_aborting"
            return result
        command_result[
            "post_command_feedback_refresh"
        ] = await _refresh_position_after_raw_motion(coordinator)
        # With the pulse now software-stopped (above), wait for the lagged
        # map-local feed to register and settle this pulse's motion before
        # sampling, mirroring the vector executor.
        position_settle = await _settle_linear_position_feed(coordinator, before)
        command_result["position_settled"] = position_settle["settled"]
        command_result["position_moved"] = position_settle["moved"]
        command_result["position_settle_wait_seconds"] = position_settle["wait_seconds"]
        command_result["position_feed_stale"] = position_settle["feed_stale"]
        command_result["position_settle_polls"] = position_settle["settle_polls"]
        # Phantom-motion investigation instrumentation (capture only): log both
        # position sources + RTK quality so a later run can tell a real move from a
        # feed-jump on a no-op pulse.
        command_result["position_source_comparison"] = _position_source_comparison(
            coordinator
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
            and result.get("command_not_sent", {})
            .get("kwargs", {})
            .get("angular_speed")
            == 0
        )
    if name in {"dry_run_positive_turn_to_heading", "dry_run_negative_turn_to_heading"}:
        return (
            result.get("stop_reason") == "dry_run"
            and result.get("command_not_sent", {}).get("kwargs", {}).get("linear_speed")
            == 0
        )
    if name in {"real_negative_y_segment", "real_positive_y_segment"}:
        return result.get("stop_reason") in {
            "target_reached",
            "max_commands_reached",
        } and all(
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
            min_progress_distance = float(
                diagnostic.get("min_progress_distance") or 0.0
            )
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
            # Readiness is a diagnostic probe: keep the pre-recovery fast-fail on
            # the BLE gate instead of blocking ~90s in transport recovery.
            ble_auto_recover=False,
            linear_speed_fast=400,
            linear_speed_slow=200,
            slow_linear_threshold=0.15,
            max_turn_commands=max_turn_commands,
            max_linear_commands=max_linear_commands,
            heading_tolerance_degrees=3.0,
            angular_speed_fast=180,
            angular_speed_slow=180,
            slow_turn_threshold_degrees=8.0,
            # Readiness validates the legacy course-over-ground pipeline.
            turn_mode="legacy",
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

    if (
        not dry_run
        and max_real_steps > 0
        and (not confirm_blades_off or not confirm_clear_area)
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
                {
                    "stop_reason": "position_unavailable",
                    "blockers": ["position_unavailable"],
                },
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
                {
                    "stop_reason": "heading_unavailable",
                    "blockers": ["heading_unavailable"],
                },
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

    if (
        not dry_run
        and max_real_steps > 0
        and (not confirm_blades_off or not confirm_clear_area)
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
        signed_progress is not None and signed_progress >= min_heading_change_degrees
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
    except TypeError, ValueError:
        return None


def _map_heading_to_toward_degrees(
    map_heading_degrees: float,
    *,
    toward_mirror_degrees: float = _TOWARD_MIRROR_DEGREES,
) -> float:
    """Convert a map-frame bearing into the ``toward`` frame."""
    return (float(toward_mirror_degrees) - float(map_heading_degrees)) % 360


def _toward_to_map_heading_degrees(
    toward_degrees: float,
    *,
    toward_mirror_degrees: float = _TOWARD_MIRROR_DEGREES,
) -> float:
    """Convert a ``toward`` reading into a map-frame bearing.

    The reflection is an involution; two names make the call-site direction
    explicit for review.
    """
    return (float(toward_mirror_degrees) - float(toward_degrees)) % 360


#: Above this the opening geometry is reported as a post-turn leg rather than an
#: aligned start. A reporting threshold only -- the measured error is always
#: present in the record, so a caller with a different bar can apply it.
_ALIGNED_START_TOLERANCE_DEGREES = 10.0

#: Why the obvious alignment check is not a check. Carried in every run record.
_CIRCULAR_ALIGNMENT_WARNING = (
    "target_reported_heading_degrees agrees with `toward` by construction "
    "whenever the target was placed along `toward`, so comparing them proves "
    "NOTHING. Alignment is only confirmed by a source that does not derive "
    "from the number being checked -- the VIO calibration drive's measured "
    "map_motion_heading_degrees, or the operator's eyes."
)


def _unmeasured_start_geometry(reason: str) -> dict[str, Any]:
    """Return a start-geometry record that asserts nothing."""
    return {
        "aligned_start_confirmed": None,
        "basis": None,
        "reason": reason,
        "measured_map_facing_degrees": None,
        "target_map_heading_degrees": None,
        "initial_heading_error_degrees": None,
        "circularity_warning": _CIRCULAR_ALIGNMENT_WARNING,
    }


def _start_alignment_evidence(
    *,
    measured_map_facing_degrees: float | None,
    target_map_heading_degrees: float | None,
    tolerance_degrees: float = _ALIGNED_START_TOLERANCE_DEGREES,
) -> dict[str, Any]:
    """Return a NON-CIRCULAR verdict on whether this segment starts aligned.

    🚨 On 2026-09-04 four runs were recorded as "aligned start confirmed"
    because the echoed ``target_reported_heading_degrees`` matched the live
    ``toward`` to ~0.0002 deg. It matched by construction -- the target had been
    placed along ``toward``. The check measured nothing, and runs 3 and 4 in
    fact opened with real ~120-135 deg turns, making them post-turn legs, the
    property that series had explicitly put out of scope.

    The only independent measurement of true facing this executor has is the VIO
    calibration drive's ``map_motion_heading_degrees``: the mower is told to go
    forward and the map-frame displacement says where forward was. It does not
    derive from ``toward`` at all, so comparing it against the bearing to the
    target is a real check. When it is unavailable the answer is ``None`` --
    never ``True``.
    """
    if measured_map_facing_degrees is None or target_map_heading_degrees is None:
        return _unmeasured_start_geometry("independent_facing_unavailable")
    error = abs(
        (
            (float(target_map_heading_degrees) - float(measured_map_facing_degrees))
            + 180.0
        )
        % 360.0
        - 180.0
    )
    return {
        "aligned_start_confirmed": error <= float(tolerance_degrees),
        "basis": "vio_calibration_drive.map_motion_heading_degrees",
        "reason": ("aligned" if error <= float(tolerance_degrees) else "post_turn_leg"),
        "measured_map_facing_degrees": round(float(measured_map_facing_degrees), 3),
        "target_map_heading_degrees": round(float(target_map_heading_degrees), 3),
        "initial_heading_error_degrees": round(error, 3),
        "tolerance_degrees": float(tolerance_degrees),
        "circularity_warning": _CIRCULAR_ALIGNMENT_WARNING,
    }


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
    motion_refresh_interval_ms: int = 0,
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
        "motion_refresh_interval_ms": motion_refresh_interval_ms,
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

        # beta47: hold the pulse open at the app's cadence instead of firing
        # once and sleeping. Without it the h-watchdog stops the motor almost
        # immediately and a single-shot angular pulse rotates only a few
        # degrees -- measured 2026-08-12 night, this loop closed CORRECTLY on
        # `toward` (+9.79 / +6.77 / +5.37 / +7.02 deg, every command in the
        # commanded direction, the per-command changes summing exactly to the
        # reported final heading) and still ran out of budget at 29 deg of a
        # 90 deg target. The loop was never the problem; the pulse was.
        #
        # `refresh_interval_ms <= 0` reproduces the exact legacy behaviour, so
        # every existing caller is unaffected and the default stays 0.
        # `kwargs` is bound at definition time on purpose: the closure is
        # created inside the command loop, and capturing the loop variable would
        # make a later iteration's kwargs visible to an earlier resend if the
        # helper ever deferred the call (ruff B023).
        async def _resend_turn(
            kwargs: dict[str, Any] = command_result["kwargs"],
        ) -> None:
            await _send_manager_command_with_args(
                coordinator,
                "send_movement",
                prefer_ble=prefer_ble,
                command_kwargs=kwargs,
            )

        command_result["motion_refresh"] = await _motion_refresh_window(
            coordinator,
            resend=_resend_turn,
            duration_seconds=pulse_duration_ms / 1000,
            refresh_interval_ms=motion_refresh_interval_ms,
        )
        command_result["stop_result"] = await _manual_velocity_stop_attempt(
            coordinator, use_wifi=not prefer_ble
        )
        command_result[
            "post_command_feedback_refresh"
        ] = await _refresh_position_after_raw_motion(coordinator)

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


#: Distance one full-length linear pulse covers at the proven config
#: (linear 400, 3500 ms, ``motion_refresh_interval_ms`` 200). Measured live
#: 2026-07-27 over six pulses across two runs: 2.125 m / 2 pulses and
#: 4.06 m / 4 pulses both give ~1.06 m. Only the fallback -- the executor
#: prefers what it actually observes during the run.
_DEFAULT_METRES_PER_LINEAR_PULSE = 1.06
#: Two isolated 3500 ms pulses delivered 10 and 11 refresh writes. Use the
#: lower measured count so a short approach never requests more actuation than
#: its distance fraction supports. The initial non-zero write is additional.
_DEFAULT_REFRESH_COMMANDS_PER_LINEAR_PULSE = 10
# A correction of 90 degrees or more cannot have a forward component toward
# the waypoint.  Calling that a "re-alignment" permits a U-turn after the mower
# has passed the target -- exactly the overshoot-and-recovery path captured on
# 2026-08-05 and independently filmed on 2026-08-06.  A forward-only segment
# must stop instead of silently becoming a reverse-recovery controller.
_MAX_FORWARD_REALIGNMENT_DEGREES = 90.0

#: Tolerance for the post-turn alignment gate, in degrees. Kept SEPARATE from
#: `vio_realign_threshold_degrees` (the mid-drive trigger), which the gate used to
#: borrow through a `min()`. The two answer different questions and only shared a
#: number by accident.
#:
#: WHY THE GATE MATTERS (measured 2026-08-10, see
#: docs/turn-translation-explains-the-landing-wall-20260810.md): a VIO turn does
#: not pivot in place. It displaced the mower 0.028-0.131 m across the five
#: completed segments of that day's two runs, and sideways displacement at the
#: start of a 0.6-0.7 m leg rotates the bearing to the target by
#: `atan(translation/leg)`. The turn primitive closes on VIO BODY HEADING, so it
#: cannot see this -- the heading did not change, the target's bearing moved. The
#: map-frame aim error after the turn ran 3.079-11.452 deg while the turn phase
#: reported 1.716-12.447 deg in the VIO frame, and the difference equals
#: `atan(translation/leg)` to within 0.02-1.25 deg. This gate is the only place
#: that error can be caught.
#:
#: WHY 10 AND NOT 5. A correction only fires when the error EXCEEDS the tolerance,
#: so the worst sweep the correction may be asked to make is `error + tolerance >
#: 2 x tolerance`. The anti-overshoot bound is affine --
#: `_VIO_TURN_SWEEP_BOUND_*`, i.e. `40 deg/s * t + 12 deg` -- and at the 200 ms
#: actuation floor the shortest safe pulse can still sweep 20 deg. So the
#: guarantee holds only while `error + tolerance >= 20`, which for a
#: trigger-on-exceed gate means **tolerance >= 10**. Below that, corrections land
#: in the `sweep_exceeds_any_pulse` regime where NO duration is safe, and a gate
#: meant to improve aim would start manufacturing overshoot with 2 commands to fix
#: it. 10 is the tightest value the turn primitive can actually honour.
#:
#: ⚠️ This therefore catches only 2 of the 5 observed cases (10.607 and 11.452
#: deg); 3.079, 7.360 and 7.304 still pass. Tightening further needs a shorter
#: actuation floor or a tighter sweep bound, NOT a smaller number here.
_POST_TURN_ALIGNMENT_TOLERANCE_DEGREES = 10.0


def _projected_landing_after_next_pulse(
    *,
    distance_to_target_m: float,
    aim_error_degrees: float,
    metres_per_pulse: float,
) -> float:
    """Where the mower ends up if it fires one more pulse on its current heading.

    ⚠️ **This is the correction beta42 exists for, and it is derived, not fitted.**
    The guard used to answer `distance * sin(aim)` -- the miss at the point of
    CLOSEST APPROACH. But the mower does not stop at the closest approach. It
    drives a whole pulse and sails past it, and the leftover along-track distance
    adds to the miss in quadrature.

    With the remaining distance ``d``, an aim error ``a``, and a pulse that
    travels ``t``:

        perpendicular  = d * sin(a)
        overshoot      = max(0, t - d * cos(a))
        landing        = hypot(perpendicular, overshoot)

    The executor's own final-approach planner aims the next pulse at the
    remaining distance and only fires a full pulse when more than one remains,
    so ``t = min(d, metres_per_pulse)``. In the ``t = d`` case -- every decision
    on record -- this collapses to the chord ``2 * d * sin(a / 2)``, which is
    just "you drove the right distance in the wrong direction".

    Measured on 2026-08-11: 0.3246 m out at 26.914 deg projected a 0.1469 m miss
    and was suppressed by **3.1 mm**; the next pulse ran 0.3771 m and it landed
    0.1797 m out, ending the segment on `target_requires_reverse_recovery`. This
    model projects 0.1511 m for that decision -- over tolerance, so it corrects.

    ⚠️ **That margin is 1.1 mm.** The model is directionally right and mechanistic,
    but it is not precise, because the next pulse's travel is not precisely
    predictable: measured next-pulse distance ran 0.30x to 1.16x of the remaining
    distance across the same runs. Do not read a projection as a landing
    prediction; read it as "which side of the tolerance is this on".
    """
    aim = math.radians(min(abs(float(aim_error_degrees)), 90.0))
    perpendicular = distance_to_target_m * math.sin(aim)
    travel = min(distance_to_target_m, max(0.0, metres_per_pulse))
    overshoot = max(0.0, travel - distance_to_target_m * math.cos(aim))
    return math.hypot(perpendicular, overshoot)


def _mid_drive_realign_decision(
    *,
    distance_to_target_m: float,
    aim_error_degrees: float,
    waypoint_tolerance: float,
    metres_per_pulse: float,
    realign_threshold_degrees: float,
) -> dict[str, Any]:
    """Decide whether a mid-drive re-aim should fire, and say why.

    🔑 **THE TRIGGER IS A DISTANCE GATED BY AN ANGLE, AND THAT ORDER MATTERS.**
    Through beta56 it was the other way round -- ``abs(aim) >
    vio_realign_threshold_degrees and abs(aim) > heading_tolerance_degrees``,
    an angle test whose effective value was 18 deg on the accepted profile. The
    projected-landing helpers existed (beta42) but could only ever SUPPRESS a
    correction, never fire one.

    That asymmetry is what limited leg length, and it is worth stating exactly,
    because "long legs are inaccurate" was the wrong diagnosis:

        aim error   range     miss      old trigger   new trigger
        17 deg      14.0 m    4.09 m    no  (17<18)   yes
        17 deg       0.8 m    0.23 m    no  (17<18)   yes
        12 deg       0.5 m    0.10 m    no             no (lands inside)

    The miss is ``range * sin(aim)``, so the SAME angle means wildly different
    things at different ranges. A controller triggering on the angle alone is
    tuned for exactly one range -- which is why ~0.8 m legs behaved and a 1.65 m
    leg after a turn did not.

    The angle survives only as a floor. A correction is an angle, and the turn
    primitive cannot make an arbitrarily small one: at the 200 ms actuation
    floor the affine sweep bound still permits 20 deg, so asking for a 3 deg
    correction leaves the mower worse aimed than it started. See
    ``_MIN_CORRECTABLE_AIM_ERROR_DEGREES``.

    ``max``, not ``min``, against the caller's threshold: an operator may make
    the controller less twitchy, but may not ask for a correction below what the
    hardware can deliver.
    """
    correctable_floor = max(
        float(realign_threshold_degrees), _MIN_CORRECTABLE_AIM_ERROR_DEGREES
    )
    past_correctable_floor = abs(float(aim_error_degrees)) >= correctable_floor
    already_lands_inside = _realign_cannot_improve_the_landing(
        distance_to_target_m=distance_to_target_m,
        aim_error_degrees=aim_error_degrees,
        waypoint_tolerance=waypoint_tolerance,
        metres_per_pulse=metres_per_pulse,
    )
    return {
        "correctable_floor_degrees": correctable_floor,
        "past_correctable_floor": past_correctable_floor,
        "already_lands_inside": already_lands_inside,
        "needs_correction": past_correctable_floor and not already_lands_inside,
        "projected_landing_m": _projected_landing_after_next_pulse(
            distance_to_target_m=distance_to_target_m,
            aim_error_degrees=aim_error_degrees,
            metres_per_pulse=metres_per_pulse,
        ),
    }


def _realign_cannot_improve_the_landing(
    *,
    distance_to_target_m: float,
    aim_error_degrees: float,
    waypoint_tolerance: float,
    metres_per_pulse: float,
) -> bool:
    """Return true when driving straight on still lands inside the tolerance disc.

    The question a mid-drive re-aim should ask is not "is the bearing numerically
    delicate here" but "will continuing as I am still put me inside the disc". If
    it will, correcting spends turn commands and adds turn translation to buy
    nothing, because landing anywhere inside `waypoint_tolerance` ends the
    segment.

    The perpendicular miss is ``distance * sin(aim_error)`` -- how far the target
    sits off the line the mower is currently travelling. Compare it to the
    tolerance and the answer follows; no tuned constant is involved.

    ⚠️ THIS REPLACES A WRONG GUARD I SHIPPED IN beta36, and the failure is worth
    keeping. That version compared DISTANCE alone against
    ``tolerance / tan(trigger)`` = 0.4617 m, on the theory that everything inside
    that radius was bearing noise. On the 0.7 m legs introduced the same day it
    disabled re-aim for the last 66% of every leg however badly the mower was
    pointed, and on 2026-08-09 it suppressed corrections at aim errors of 40.099
    and 77.922 deg. Segment 3 then landed 0.2548 m out, the worst of that day
    (docs/evidence-beta32-4segment-20260810T002506Z.json). Those errors were
    real, not noise: RTK measured +40.73 and +72.92 deg independently of VIO.

    Checked against all four mid-drive re-aims on record, this criterion agrees
    with the outcome every time where the distance rule got two wrong:

        d 0.540  aim 23.30  perp 0.214 m  -> correct   (was allowed, legitimate)
        d 0.210  aim 34.84  perp 0.120 m  -> suppress  (allowed, then oscillated)
        d 0.310  aim 40.10  perp 0.200 m  -> correct   (beta36 suppressed it)
        d 0.207  aim 77.92  perp 0.203 m  -> correct   (beta36 suppressed it)

    Fails OPEN on degenerate input, because suppressing a re-aim is the dangerous
    direction: a mower that stops correcting its aim keeps driving.

    🏁 **beta42 projects to the END OF THE NEXT PULSE, not to the closest
    approach** -- see `_projected_landing_after_next_pulse`. Replayed against all
    twelve beta38-era suppressions on record it changes three decisions: it
    corrects the one that actually missed (0.1797 m, reverse-recovery), and it
    also corrects two that had scraped inside at **52.0 and 54.2 deg** of aim
    error, landing 0.1393 and 0.1467 against a 0.150 tolerance. Those two are a
    deliberate accepted cost, not an oversight: declining to re-aim while pointed
    54 deg away from the target because the arithmetic says you would just clip
    the edge of the disc is not a trade worth defending, and both were within
    1 cm of missing. The measured price of a correction is ~0.97 deg of induced
    bearing error to buy ~10 deg, roughly 10:1 in our favour.
    """
    if distance_to_target_m <= 0 or waypoint_tolerance <= 0:
        return False
    aim = abs(float(aim_error_degrees))
    # At or past 90 deg the target is abeam or behind, so driving forward can
    # only make things worse and `_requires_reverse_recovery` should already have
    # stopped the segment. Never suppress there.
    if aim >= _MAX_FORWARD_REALIGNMENT_DEGREES:
        return False
    projected = _projected_landing_after_next_pulse(
        distance_to_target_m=distance_to_target_m,
        aim_error_degrees=aim,
        metres_per_pulse=metres_per_pulse,
    )
    return projected <= waypoint_tolerance


def _requires_reverse_recovery(aim_error_degrees: float) -> bool:
    """Return whether reaching the target now requires a non-forward recovery."""
    return abs(float(aim_error_degrees)) >= _MAX_FORWARD_REALIGNMENT_DEGREES


def _effective_metres_per_pulse(
    observed_pulse_distances: list[float], default_metres_per_pulse: float
) -> float:
    """How far the executor expects a FULL linear pulse to travel.

    Shared by the final-approach planner and the mid-drive re-aim guard so the
    two cannot disagree about the size of the next step -- the same reason the
    beta32 turn preflight was made to replay the executor's own pulse policy
    rather than keep a second model of it.

    Never let a slow or noisy observation shrink the figure: over-estimating
    stops short and spends another bounded pulse, under-estimating adds writes
    and overshoots.
    """
    if not observed_pulse_distances:
        return default_metres_per_pulse
    observed = sum(observed_pulse_distances) / len(observed_pulse_distances)
    return max(default_metres_per_pulse, observed)


def _final_approach_pulse_ms(
    *,
    distance_to_target: float | None,
    observed_pulse_distances: list[float],
    default_metres_per_pulse: float,
    pulse_duration_ms: float,
    refresh_interval_ms: int,
) -> dict[str, Any]:
    """Bound the last pulse by confirmed refresh-write count.

    A full pulse covers ~1 m. Whenever the remaining distance is less than that
    but more than ``waypoint_tolerance``, the executor previously had no move
    available except a full pulse, so it necessarily overshot -- live
    2026-07-27, a run with ~0.2 m to go fired a full pulse, overshot by 0.8 m,
    and then failed trying to re-aim at a target that was now 176 deg behind it.
    Raising the tolerance does not fix that; the granularity is the problem.

    **This only works because of the refresh cadence.** Single-shot motion moves
    a fixed ~4 in step regardless of duration. Duration-only refreshed scaling
    was also disproved live on 2026-08-02: 1012.5 ms delivered two refreshes and
    moved 0.1786 m, while 1191.8 ms delivered three and moved 0.4341 m. Confirmed
    BLE write and stop latency make nominal time a poor actuation unit. Bound
    the discrete non-zero writes instead and stop immediately after the budget.

    Self-calibrating: ``observed_pulse_distances`` holds each measured pulse
    normalised to the eleven non-zero writes represented by the fallback. This
    lets bounded final-approach pulses contribute instead of making the fallback
    permanently load-bearing on short segments. ``default_metres_per_pulse``
    only covers the first pulse at a given speed, before there is anything to
    observe.
    """
    info: dict[str, Any] = {
        "applied": False,
        "reason": None,
        "distance_to_target": distance_to_target,
        "metres_per_pulse": None,
        "metres_per_pulse_source": None,
        "pulse_duration_ms": pulse_duration_ms,
        "refresh_command_limit": None,
        "full_pulse_refresh_commands": _DEFAULT_REFRESH_COMMANDS_PER_LINEAR_PULSE,
    }
    if refresh_interval_ms <= 0:
        info["reason"] = "refresh_disabled_distance_not_proportional_to_duration"
        return info
    if distance_to_target is None:
        info["reason"] = "distance_unknown"
        return info

    metres_per_pulse = _effective_metres_per_pulse(
        observed_pulse_distances, default_metres_per_pulse
    )
    if observed_pulse_distances:
        observed_metres_per_pulse = sum(observed_pulse_distances) / len(
            observed_pulse_distances
        )
        info["observed_metres_per_pulse"] = round(observed_metres_per_pulse, 4)
        info["metres_per_pulse_source"] = (
            "observed"
            if observed_metres_per_pulse >= default_metres_per_pulse
            else "default_conservative_floor"
        )
    else:
        info["metres_per_pulse_source"] = "default"
    info["metres_per_pulse"] = round(metres_per_pulse, 4)

    if metres_per_pulse <= 0:
        info["reason"] = "no_usable_pulse_distance"
        return info
    if distance_to_target >= metres_per_pulse:
        info["reason"] = "cruising_full_pulse_fits"
        return info

    full_nonzero_writes = _DEFAULT_REFRESH_COMMANDS_PER_LINEAR_PULSE + 1
    target_nonzero_writes = max(
        1,
        math.ceil((distance_to_target / metres_per_pulse) * full_nonzero_writes),
    )
    refresh_command_limit = min(
        _DEFAULT_REFRESH_COMMANDS_PER_LINEAR_PULSE,
        max(target_nonzero_writes - 1, 0),
    )
    info.update(
        {
            "applied": True,
            "reason": "final_approach_bounded_by_refresh_count",
            "refresh_command_limit": refresh_command_limit,
            "target_nonzero_writes": target_nonzero_writes,
        }
    )
    return info


def _normalised_linear_pulse_distance(
    measured_distance: float, refresh_commands_sent: int
) -> float:
    """Normalise a measured pulse to the fallback's eleven non-zero writes."""
    actual_nonzero_writes = max(int(refresh_commands_sent), 0) + 1
    fallback_nonzero_writes = _DEFAULT_REFRESH_COMMANDS_PER_LINEAR_PULSE + 1
    return measured_distance * fallback_nonzero_writes / actual_nonzero_writes


_SEGMENT_TURN_MODES = ("vio", "legacy", "night")

#: Every measured stationary night turn that converged used +/-500 with refresh.
#: Both tiers deliberately use this value: 180 did not break stationary-pivot
#: friction, and the slow tier has no separate night measurement.
_NIGHT_TURN_ANGULAR_SPEED = 500

#: Chosen fixed-budget containment limit, not a measured night reach claim.
_NIGHT_MAX_SEGMENT_LENGTH_M = 1.0

#: Longest single segment the daylight/VIO path will dispatch, in metres.
#:
#: ⚠️ **This is an AUTHORIZATION cap, not a physics claim.** 6.10 m is 20 ft,
#: chosen by the operator on 2026-08-17. It is deliberately larger than anything
#: measured and smaller than anything asked for, and it exists so that the leg
#: length a run may dispatch is a decision on the record rather than whatever a
#: click happens to produce.
#:
#: WHAT THE MEASUREMENTS ACTUALLY SAY, so nobody reads 6.10 as evidence:
#:
#:   * 4.0 m is the longest segment ever executed -- 11 pulses, landing 0.1023 m,
#:     stopping on TOLERANCE with the ceiling never binding
#:     (docs/loop-to-tolerance-reach-20260811.md). n = 1, straight, starting
#:     aligned. "A demonstrated floor, not a limit."
#:   * 1.65 m DIVERGED after a 48.6 deg junction turn, stopping safely on
#:     `vio_realign_budget_exhausted` 0.2514 m out
#:     (docs/evidence-real-go-card-beta55-20260815T204747Z.json).
#:
#: Those two are not in conflict and the difference is not distance. The 4 m run
#: re-aimed while it still had range to spare; the 1.65 m run spent its whole
#: 3-correction budget on a final approach whose bearing was rotating faster
#: than an 18 deg correction could track. Leg length is only dangerous when the
#: mower STOPS correcting, which is what `vio_max_realignments: 3` and the old
#: angle-only re-aim trigger both caused.
#:
#: ⚠️ ONLY THE TRIGGER WAS FIXED. A raise of `vio_max_realignments` was attempted
#: on 2026-08-17 and reverted (see that parameter), so the budget is still the 3
#: that the 1.65 m divergence above exhausted. Whether 3 corrections carry a 6 m
#: leg is THE open question this cap exists to let us ask safely -- exhausting
#: the budget stops the segment, which is a measurement, not a hazard. 6.10 m is
#: beyond anything measured and OWES A GATE 5 before it is an accepted path.
_MAX_SEGMENT_LENGTH_M = 6.10

#: The sub-leg length the CARD sends as `split_leg_target_length_m`, in metres.
#:
#: Route B (2026-08-19). Not a schema default and NOT a `LUBA_ACCEPTANCE_PROFILE`
#: key -- putting it in the profile would un-accept the profile and owe another
#: Gate 5, which is the exact cost Route B exists to avoid. It is duplicated in
#: `www/mammotion-custom-path-card.js` as `SPLIT_LEG_TARGET_METRES`; keep the two
#: in step, the same way `MAX_REAL_SEGMENT_METRES` mirrors `_MAX_SEGMENT_LENGTH_M`.
#:
#: 3.85 rather than 3.81: `n = ceil(d / target)` is a step function and
#: 15.24 / 3.81 = 4.000 exactly, so a centimetre of drift between the card's
#: check and the backend's snapshot flips it to n=5 and the run is refused.
#: 3.85 buys 0.16 m of headroom before the count rounds up, and a true 50 ft
#: (15.24 m) click still splits into 4 sub-legs of 3.81 m.
#:
#: ⚠️ 3.81 m is 95% of the single longest straight leg ever executed (4.0 m,
#: landing 0.1023 m against 0.15 m, n = 1). It is not proven better than 4.0,
#: only shorter. 4 x 3.85 = 15.40 m HAS NEVER BEEN DRIVEN. If the first run
#: disappoints, 3.0 m (~39 ft) is the conservative fallback.
_SPLIT_LEG_TARGET_LENGTH_M = 3.85

#: Conservative per-pulse travel used ONLY to check a segment's pulse budget
#: before dispatch, in metres.
#:
#: Not the same question as `final_approach_metres_per_pulse` (1.06), which is
#: the planner's model of a full-length pulse. This is "what does a pulse travel
#: when the link is having a bad day", because a budget that only survives a
#: healthy link strands the mower mid-leg on `max_linear_pulse_ceiling_reached`.
#:
#: Measured 2026-08-11 on the reach runs: a healthy pulse travelled ~0.41 m and
#: a BLE-stalled one ~0.22 m, with 2 of 11 pulses stalled on the 4 m leg. At
#: double that stall rate the mean is ~0.34 m/pulse and at a 50% stall rate
#: ~0.315. 0.30 sits under all three.
_BUDGET_CHECK_METRES_PER_PULSE = 0.30

#: Smallest aim error a mid-drive correction can usefully act on, in degrees.
#:
#: 🔑 THIS IS WHY THE RE-AIM TRIGGER HAS AN ANGLE TERM AT ALL, and getting that
#: wrong is what limited leg length. The OBJECTIVE is a distance (land inside
#: `waypoint_tolerance`), but a correction is an ANGLE, and the turn primitive
#: cannot make an arbitrarily small one: the anti-overshoot bound is affine
#: (`40 deg/s * t + 12 deg`) and at the 200 ms actuation floor the shortest safe
#: pulse can still sweep 20 deg. Asking for a 3 deg correction gets a 11-20 deg
#: sweep and leaves the mower worse aimed than it started.
#:
#: So the angle term belongs, but it must be the SMALLEST CORRECTABLE ANGLE --
#: not, as it was through beta56, `heading_tolerance_degrees` (18). That test
#: asked "is the bearing far off" when the question is "will I miss the disc",
#: and the two diverge exactly as range grows: 17 deg of aim error with 14 m
#: still to run is a 4.1 m miss that fired NO correction, because 17 < 18. At
#: 0.8 m of range the same 17 deg is 0.23 m and the machinery worked, which is
#: the whole reason ~0.8 m legs behaved and longer ones did not.
#:
#: 🚨 IT IS DERIVED, NOT CHOSEN, AND THE DEADBAND TERM IS LOAD-BEARING. A
#: mid-drive correction closes to `min(heading_tolerance_degrees,
#: _POST_TURN_ALIGNMENT_TOLERANCE_DEGREES)` = 10 deg. A trigger floor EQUAL to
#: that tolerance is unstable in both directions: a correction ending at 9.95 deg
#: re-fires on the next pulse, and an error a hair past the floor makes the turn
#: primitive's entry check return `target_heading_reached` having sent nothing --
#: while the slot has already been charged. Three slots burn and the segment
#: aborts on `vio_realign_budget_exhausted` having corrected nothing.
#:
#: That configuration was actually tried on 2026-08-17 (threshold defaulted to
#: 10) and reverted. It was reverted by moving a DEFAULT, which left the hole
#: open for anyone passing `vio_realign_threshold_degrees: 5` -- the schema
#: still allows it. So the deadband now lives here, where no caller can collapse
#: it: `max(caller_threshold, this)` can only ever raise the floor.
_REALIGN_DEADBAND_DEGREES = 5.0
_MIN_CORRECTABLE_AIM_ERROR_DEGREES = (
    _POST_TURN_ALIGNMENT_TOLERANCE_DEGREES + _REALIGN_DEADBAND_DEGREES
)


#: Position-feed noise makes aim estimates from shorter pulses uninformative.
_NIGHT_MIN_AIM_BASELINE_M = 0.20


def _correctable_leg_length_limit_m(
    *,
    waypoint_tolerance: float,
    min_correctable_aim_degrees: float = _MIN_CORRECTABLE_AIM_ERROR_DEGREES,
) -> float:
    """Longest leg whose landing the mid-drive controller can still protect.

    🔑 **This number is implied by two values that were chosen independently, and
    until 2026-08-20 nothing computed it.** A mid-drive correction fires only
    once the aim error reaches ``_MIN_CORRECTABLE_AIM_ERROR_DEGREES`` -- below
    that the turn primitive cannot deliver the correction, so asking for one
    leaves the mower worse aimed (see ``_mid_drive_realign_decision``). An aim
    error sitting just under that floor is therefore NEVER corrected, however
    much it costs, and the miss it produces is ``distance * sin(floor)``.

    Setting that equal to the tolerance gives the longest protectable leg::

        limit = waypoint_tolerance / sin(min_correctable_aim_degrees)

    On the accepted profile -- tolerance 0.15 m, floor 15 deg (post-turn
    tolerance 10 + deadband 5) -- that is **0.580 m**. At 3.0 m the same floor
    permits an uncorrectable 0.776 m miss, over 5x the tolerance.

    🗑️ **CORRECTED 2026-09-04. The arithmetic stands; the operational reading
    attached to it was REFUTED and has been removed.** This docstring used to say
    0.580 m "is why the measured-good regime is ~0.8 m and why longer legs miss".
    Measurement contradicts both halves: **reach is CLOSED at 6.0 m and landing
    does not degrade with distance** -- 0.1023 m at 4 m, 0.1015 m at 5 m,
    0.1144 m at 6 m, every one inside the 0.15 m tolerance and roughly 10x better
    than this bound permits at that length. 3.0 m legs are 5 reached / 1 failed.
    ⚠️ **Do not re-attach a "~0.8 m operating rule" to this number.** That rule
    was an artifact of the pre-beta57 ANGLE-triggered re-aim, which never fired
    in the far field; beta57 made the trigger a projected miss.

    ⚠️ **This is an ADVISORY bound, not a hard limit, and it is deliberately
    pessimistic -- it is a GUARANTEE bound, not a prediction.** It asks what
    happens when aim error sits just under the floor for the whole leg and is
    never corrected. Real legs correct repeatedly and land far better. Read it as
    "beyond here the controller cannot GUARANTEE the landing", never as "beyond
    here the landing fails" -- the measured landings above are the direct
    counter-evidence to the second reading.

    🚨 **Do not respond to a breach by lowering the floor.** The floor is set by
    the turn primitive's actuation limit, not by preference: at the 200 ms floor
    the affine sweep bound still permits 20 deg, so a sub-floor correction
    manufactures the error it was meant to remove. Two attempts to loosen the
    re-aim path on thin evidence were reviewed and reverted. The lever is
    shorter legs, or less cross-track accumulation -- not more late corrections.
    """
    floor = abs(float(min_correctable_aim_degrees))
    if floor <= 0.0 or floor >= 90.0:
        return float("inf")
    return float(waypoint_tolerance) / math.sin(math.radians(floor))


async def _vio_segment_calibration_drive(  # noqa: C901, PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    *,
    prefer_ble: bool = True,
    linear_speed: int = 200,
    max_pulses: int = 2,
    pulse_duration_ms: float = 1500.0,
    refresh_wait_seconds: float = 2.0,
    min_calibration_distance_m: float = 0.06,
) -> dict[str, Any]:
    """Derive the map-frame -> VIO-frame heading offset from a short forward drive.

    The baseline requirement (default 6 cm) matters: live run 2026-07-11 showed
    a 2 cm baseline yields ~25 deg of offset error from cm-level position noise,
    sending the whole forward leg off-bearing.

    ``vision_info.heading`` is a body heading in the VIO's own frame, which is
    re-anchored whenever VIO (re)initialises -- it has no fixed relationship to
    map-local coordinates. Driving forward yields both frames at once: the
    telemetry x/y delta gives the motion heading in the map frame while the
    concurrent (motion-refreshed) ``vision_heading`` gives the body heading in
    the VIO frame, so ``offset = map_motion_heading - vision_heading``. The
    drive doubles as the VIO warm-up/refresh the latched heading needs.
    """
    start_telemetry = _custom_path_telemetry_snapshot(coordinator)
    result: dict[str, Any] = {
        "passed": False,
        "reason": None,
        "offset_degrees": None,
        "map_motion_heading_degrees": None,
        "vision_heading": None,
        "vio_state": None,
        "vio_feed": None,
        "distance_m": None,
        "pulses_sent": 0,
        "command_results": [],
    }
    if not _position_available(start_telemetry):
        result["reason"] = "position_unavailable"
        return result
    for pulse_index in range(1, max_pulses + 1):
        before = _custom_path_telemetry_snapshot(coordinator)
        if not _blade_reported_safe(before):
            result["reason"] = "aborted_unsafe_blade"
            return result
        if before.get("work_mode_label") not in {"MODE_READY", "MODE_PAUSE"}:
            result["reason"] = "aborted_unsafe_mode"
            return result
        if prefer_ble and not _transport_is_ble(coordinator):
            # Transport can flap between the gate check and each pulse; motion
            # over the laggy cloud fallback is neither observable nor stoppable
            # in time, so refuse to pulse without BLE.
            result["reason"] = "ble_transport_lost"
            return result
        command_result: dict[str, Any] = {
            "index": pulse_index,
            "phase": "vio_calibration_drive",
            "command": "send_movement",
            "kwargs": {"linear_speed": linear_speed, "angular_speed": 0},
            "sent_at_utc": _utc_timestamp(),
            "ok": None,
            "error": None,
        }
        try:
            await _send_manager_command_with_args(
                coordinator,
                "send_movement",
                prefer_ble=prefer_ble,
                command_kwargs=command_result["kwargs"],
            )
            command_result["ok"] = True
        except Exception as err:  # noqa: BLE001
            command_result["ok"] = False
            command_result["error"] = f"{type(err).__name__}: {err}"
            result["command_results"].append(command_result)
            result["pulses_sent"] += 1
            result["reason"] = "command_failed"
            return result
        result["pulses_sent"] += 1
        await _motion_open_sleep(coordinator, pulse_duration_ms / 1000)
        command_result["stop_result"] = await _manual_velocity_stop_attempt(
            coordinator, use_wifi=not prefer_ble
        )
        if not (command_result["stop_result"] or {}).get("ok"):
            # Live 2026-07-12: BLE dropped into its connect cooldown mid-run and
            # the stop could not be delivered — never keep pulsing motion when
            # stops are not deliverable.
            result["command_results"].append(command_result)
            result["reason"] = "stop_failed_aborting"
            return result
        # Request reports immediately, then give the asynchronous position feed
        # one total settling window. The refresh helper normally includes its
        # own two-second sleep; adding the four-second VIO calibration wait made
        # every cold-start pulse idle for six seconds. Real Go needs the measured
        # four-second feed-latency window once.
        command_result["feedback_refresh"] = await _refresh_position_after_raw_motion(
            coordinator, settle_seconds=0.0
        )
        await asyncio.sleep(max(refresh_wait_seconds, 0.5))
        result["command_results"].append(command_result)
        after = _custom_path_telemetry_snapshot(coordinator)
        reading = _vio_reading(coordinator)
        feed = _vio_feed_liveness(coordinator)
        delta = _telemetry_position_delta(start_telemetry, after)
        result["distance_m"] = delta.get("distance")
        result["vision_heading"] = reading["vision_heading"]
        result["vio_state"] = reading["vio_state"]
        result["vio_feed"] = feed
        if (
            delta.get("distance") is not None
            and float(delta["distance"]) >= min_calibration_distance_m
            and reading["vio_state"] == _VIO_STATE_ACTIVE
            and feed["live"]
            and reading["vision_heading"] is not None
        ):
            map_heading = (
                math.degrees(math.atan2(float(delta["dy"]), float(delta["dx"]))) + 360
            ) % 360
            result["map_motion_heading_degrees"] = round(map_heading, 3)
            result["offset_degrees"] = _normalized_heading_degrees(
                map_heading - float(reading["vision_heading"])
            )
            result["passed"] = True
            result["reason"] = "calibrated"
            return result
    if result["vio_state"] != _VIO_STATE_ACTIVE:
        result["reason"] = "vio_not_active_after_drive"
    elif not (result["vio_feed"] or {}).get("live", True):
        # vio_state active but the feature track collapsed: the heading is a
        # stale latch, so the offset it would yield is silently wrong. Report
        # the blind feed rather than a spurious calibration.
        result["reason"] = "vio_feed_degraded"
    else:
        result["reason"] = "insufficient_calibration_distance"
    return result


async def _raw_pymammotion_execute_vector_segment(  # noqa: C901, PLR0913
    coordinator: MammotionReportUpdateCoordinator,
    points: list[dict[str, float]],
    *,
    area_hash: int | None = None,
    dry_run: bool = True,
    safety_overrides: list[str] | tuple[str, ...] | None = None,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    prefer_ble: bool = True,
    ble_auto_recover: bool = True,
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
    final_approach_metres_per_pulse: float = _DEFAULT_METRES_PER_LINEAR_PULSE,
    turn_degrees_per_second: float = _DEFAULT_TURN_DEGREES_PER_SECOND,
    turn_mode: str = "vio",
    night_angular_speed: int = _NIGHT_TURN_ANGULAR_SPEED,
    toward_mirror_degrees: float = _TOWARD_MIRROR_DEGREES,
    vio_heading_offset_degrees: float | None = None,
    vio_turn_max_commands: int = 8,
    vio_angular_speed: int = 500,
    vio_calibration_pulse_count: int = 2,
    vio_realign_threshold_degrees: float = 15.0,
    vio_max_realignments: int = 3,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60),
    motion_refresh_interval_ms: int = 0,
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
    refetch_runtime_context: (
        Callable[[], tuple[str | None, dict[str, Any] | None]] | None
    ) = None,
) -> dict[str, Any]:
    """Execute or dry-run one vector segment using raw turn then forward motion.

    ``turn_mode="vio"`` (default) drives the turn phase on the live VIO body
    heading (``vision_info.heading``), which is the only signal proven
    (2026-07-10) to observe in-place rotation on this hardware. Because the
    VIO frame is re-anchored per initialisation, the map->VIO offset is
    derived live from a short forward calibration drive (or accepted via
    ``vio_heading_offset_degrees`` when already known, e.g. carried between
    multi-segment segments). ``turn_mode="legacy"`` keeps the original turn
    with its fixed additive calibrated-offset conversion.

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
    # When BLE is preferred for a real run but not yet the active transport, try to
    # promote it automatically BEFORE the telemetry snapshot and gates: a successful
    # recovery lets ble_transport_required pass without an operator toggle/reboot,
    # and the recovery wait can span ~90s during which blades/mode/position may
    # change -- the gates and target math below must judge post-recovery state. The
    # report says which cases still need a human (sleeping mower, phone on the slot).
    ble_recovery: dict[str, Any] | None = None
    if (
        not dry_run
        and prefer_ble
        and ble_auto_recover
        and not _transport_is_ble(coordinator)
    ):
        ble_recovery = await _attempt_ble_recovery(coordinator)
        if refetch_runtime_context is not None:
            # The recovery wait can span ~90s: the handler-captured HA state and
            # active-route snapshot are as stale as the telemetry would have
            # been, so re-capture them before the runtime gates judge them.
            ha_state, active_route = refetch_runtime_context()
    initial_telemetry = _custom_path_telemetry_snapshot(coordinator)
    current_point = _raw_segment_current_point(initial_telemetry)
    target = normalized_points[-1] if normalized_points else None
    target_heading = (
        _path_heading_degrees(current_point, target)
        if current_point is not None and target is not None
        else None
    )
    if target_heading is None:
        target_reported_heading = None
    elif turn_mode == "night":
        # Night only: map -> toward is a reflection. The frozen additive
        # calibration remains deliberately untouched for VIO and legacy paths.
        target_reported_heading = _map_heading_to_toward_degrees(
            target_heading, toward_mirror_degrees=toward_mirror_degrees
        )
    else:
        target_reported_heading = _normalized_heading_degrees(
            float(target_heading) - float(calibrated_forward_heading_offset_degrees)
        )
    # Let the BLE command queue drain before gating on it. This executor
    # enqueues commands of its own before it evaluates `ble_link_live`, which
    # demands `queue_depth == 0` -- so without this it can fail a fail-closed
    # gate on a command it queued itself. In a MULTI-SEGMENT run the offending
    # command is the previous segment's trailing stop, still draining when the
    # next segment begins: live 2026-08-09, segment 2 of the first four-segment
    # run was refused with `command_queue_backlogged` and `queue_depth: 1` just
    # 1.5 s after segment 1's last send, on a link that was connected, usable,
    # uncooled and dispatching (docs/evidence-beta32-4segment-20260809T170941Z.json).
    # Segments 3 and 4 never ran, so the run answered nothing.
    #
    # Five other executors already do exactly this; this one was missed, and the
    # 2-segment reach limit made it rare enough to go unnoticed until the limit
    # was raised to 4. `_settle_ble_command_queue` only ever WAITS -- it never
    # lowers the depth limit or overrides a verdict, and it returns the last
    # report unchanged on timeout, so a genuine backlog still fails the gate.
    # Skipped on a dry run, which enqueues nothing and must stay instant.
    queue_settle: dict[str, Any] | None = None
    if not dry_run:
        queue_settle = await _settle_ble_command_queue(coordinator)
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
    if turn_mode not in _SEGMENT_TURN_MODES:
        gates.append(
            {
                "name": "turn_mode_valid",
                "passed": False,
                "detail": f"turn_mode must be one of {_SEGMENT_TURN_MODES}.",
            }
        )
    if turn_mode == "night":
        # Night neither gates nor steers on VIO. Keep the response shape useful
        # without making an unavailable camera feed a hidden dependency.
        initial_vio_reading = {"vio_state": None, "vision_heading": None}
        initial_vio_feed = {
            "live": None,
            "tracked_features": None,
            "brightness_raw": None,
            "brightness_label": None,
        }
    else:
        initial_vio_reading = _vio_reading(coordinator)
        initial_vio_feed = _vio_feed_liveness(coordinator)
    if turn_mode == "vio" and initial_vio_reading["vio_state"] != _VIO_STATE_ACTIVE:
        # VIO won't initialise in the dark and a cold reading latches
        # heading=0.0 as a valid float. In a BRIGHT scene the calibration
        # drive doubles as the warm-up (VIO wakes during forward motion and
        # the drive fails safe with vio_not_active_after_drive otherwise), so
        # a cold start is allowed when the drive will actually run. In the
        # dark, or when a provided offset would skip the drive, refuse.
        calibration_will_warm = (
            vio_heading_offset_degrees is None and _vio_scene_is_bright(coordinator)
        )
        gates.append(
            {
                "name": "vio_active",
                "passed": dry_run or calibration_will_warm,
                "detail": (
                    "VIO is cold but the scene is bright: the calibration "
                    "drive will warm VIO and validate it after motion."
                    if calibration_will_warm
                    else (
                        "VIO turn mode requires an active VIO track "
                        f"(vio_state == {_VIO_STATE_ACTIVE}); saw "
                        f"{initial_vio_reading['vio_state']}. Warm it with "
                        "forward motion in daylight, or wait for daylight "
                        "(scene is dark)."
                    )
                ),
            }
        )
    if (
        turn_mode == "vio"
        and initial_vio_reading["vio_state"] == _VIO_STATE_ACTIVE
        and not initial_vio_feed["live"]
    ):
        # vio_state reads active but the feature track has collapsed (dusk latch);
        # the heading is stale. Block the real run distinctly from the cold-start
        # case above so the operator sees "blind feed", not "warm VIO".
        gates.append(_vio_feed_live_gate(initial_vio_feed, dry_run=dry_run))
    planned_segment_length = (
        _path_distance([current_point, target])
        if current_point is not None and target is not None
        else None
    )
    if turn_mode != "night":
        # Night owns a tighter cap of its own (`night_segment_too_long`, 1.0 m)
        # and a fixed 3-pulse budget, so these two would be dead weight there.
        #
        # ⚠️ Until 2026-08-17 the daylight/VIO path had NO length gate at all.
        # The ~0.8 m operating rule was documentation only, so a click could
        # dispatch a leg of any length the map allowed -- including the 1.65 m
        # geometry that had already been measured diverging. Nothing unsafe
        # happened (every failure stopped inside a bounded gate), but "the
        # operator remembers the rule" is not a gate.
        gates.append(
            {
                "name": "segment_too_long",
                "passed": dry_run
                or (
                    planned_segment_length is not None
                    and planned_segment_length <= _MAX_SEGMENT_LENGTH_M
                ),
                "detail": (
                    "Segment length is capped at "
                    f"{_MAX_SEGMENT_LENGTH_M} m by authorization, not by a "
                    "measured reach limit; the longest segment ever executed "
                    "is 4.0 m."
                ),
                "diagnostics": {
                    "segment_length_m": planned_segment_length,
                    "max_segment_length_m": _MAX_SEGMENT_LENGTH_M,
                },
            }
        )
        # A leg longer than its pulse budget does not fail safely-but-usefully;
        # it drives most of the way and stops on
        # `max_linear_pulse_ceiling_reached`, leaving the mower somewhere in the
        # middle of the yard with the run recorded as a failure. Catch it while
        # it is still arithmetic.
        #
        # ⚠️ LOOP-TO-TOLERANCE ONLY, and the distinction is not cosmetic. The two
        # linear modes have different per-pulse travel: fixed budget fires full
        # `linear_pulse_duration_ms` pulses measured at 1.0785 / 1.0449 m
        # (2026-08-01), while loop-to-tolerance shortens pulses on final
        # approach and averaged ~0.36 m across the reach runs. Applying the
        # conservative loop figure to the fixed-budget path refuses runs that
        # Gate 4 and Gate 5 both PASSED -- caught here by nine existing tests on
        # 2026-08-17, which is exactly what they are for. The fixed-budget path
        # keeps its accepted `max_linear_commands_reached` behaviour untouched.
        if max_linear_pulse_ceiling is not None:
            budget_reach = max_linear_pulse_ceiling * _BUDGET_CHECK_METRES_PER_PULSE
            gates.append(
                {
                    "name": "linear_budget_insufficient_for_segment",
                    "passed": dry_run
                    or (
                        planned_segment_length is not None
                        and planned_segment_length <= budget_reach
                    ),
                    "detail": (
                        f"{max_linear_pulse_ceiling} linear pulses reach about "
                        f"{budget_reach:.2f} m at a stall-tolerant "
                        f"{_BUDGET_CHECK_METRES_PER_PULSE} m/pulse, short of "
                        "this segment."
                    ),
                    "diagnostics": {
                        "segment_length_m": planned_segment_length,
                        "linear_pulse_ceiling": max_linear_pulse_ceiling,
                        "budget_check_metres_per_pulse": (
                            _BUDGET_CHECK_METRES_PER_PULSE
                        ),
                        "budget_reach_m": round(budget_reach, 4),
                    },
                }
            )
    if turn_mode == "night":
        # Night has no VIO witness: RTK is the sole source for position and the
        # `toward` heading loop, so Float is not acceptable here.
        gates.append(
            {
                "name": "night_requires_precise_rtk",
                "passed": dry_run or not runtime_safety["rtk_degraded"],
                "detail": (
                    "Night mode requires RTK Fix; saw "
                    f"{runtime_safety['rtk_status_label']}."
                ),
                "diagnostics": {
                    "rtk_status_label": runtime_safety["rtk_status_label"],
                    "rtk_degraded": runtime_safety["rtk_degraded"],
                },
            }
        )
        gates.append(
            {
                "name": "night_linear_loop_unsupported",
                "passed": dry_run or max_linear_pulse_ceiling is None,
                "detail": (
                    "Night runs the fixed max_linear_commands budget; pass "
                    "max_linear_pulse_ceiling: null."
                ),
            }
        )
        night_segment_length = planned_segment_length
        gates.append(
            {
                "name": "night_segment_too_long",
                "passed": dry_run
                or (
                    night_segment_length is not None
                    and night_segment_length <= _NIGHT_MAX_SEGMENT_LENGTH_M
                ),
                "detail": (
                    "Night runs a fixed 3-pulse budget with no mid-drive "
                    f"correction, so legs must be at most {_NIGHT_MAX_SEGMENT_LENGTH_M} m."
                ),
                "diagnostics": {
                    "segment_length_m": night_segment_length,
                    "max_segment_length_m": _NIGHT_MAX_SEGMENT_LENGTH_M,
                },
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
    # Apply the operator's deliberate overrides BEFORE blockers are computed --
    # one choke point, so no gate can be overridden anywhere else. An overridden
    # gate keeps `original_passed: False` and `overridden: True`, so the run JSON
    # can never present an overridden run as a clean one.
    safety_overrides_summary = _apply_safety_overrides(gates, safety_overrides)
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
        "would_send": not dry_run
        and not blockers
        and not completion_status["complete"],
        "real_execution_scope": "one_segment_turn_then_forward_only",
        "full_path_execution_allowed": False,
        "ready_for_multi_point": False,
        "prefer_ble": prefer_ble,
        "transport_preference": "ble_preferred" if prefer_ble else "default",
        "ble_auto_recover": ble_auto_recover,
        "ble_recovery": ble_recovery,
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
        # Echoed for post-run forensics, matching the multi-segment executor
        # (fixed there 2026-07-19, missed here). These are honoured but were
        # reported as absent, so a stalled run gave no way to confirm the pulse
        # geometry the mower was actually told to use -- `max_linear_pulse_ceiling`
        # especially, since without it `max_linear_commands` defaults to 1 and a
        # multi-metre segment stops after a single pulse (live 2026-07-25).
        "max_linear_pulse_ceiling": max_linear_pulse_ceiling,
        # beta44, same reason as the multi-segment echo: an accepted-profile key
        # the card sends must come back, or a gate cannot prove what ran.
        "max_no_progress_pulses": max_no_progress_pulses,
        "final_approach_metres_per_pulse": final_approach_metres_per_pulse,
        "turn_degrees_per_second": turn_degrees_per_second,
        "turn_pulse_duration_ms": turn_pulse_duration_ms,
        "linear_pulse_duration_ms": linear_pulse_duration_ms,
        "vio_turn_max_commands": vio_turn_max_commands,
        "vio_angular_speed": vio_angular_speed,
        **(
            {
                "night_angular_speed": night_angular_speed,
                "toward_mirror_degrees": toward_mirror_degrees,
            }
            if turn_mode == "night"
            else {}
        ),
        "vio_heading_offset_degrees": vio_heading_offset_degrees,
        "sample_delays": list(sample_delays),
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        # Echo it or it is unprovable. An override changes what the mower was
        # allowed to do, so the run record has to carry which gates were lifted
        # and why -- `applied` includes each gate's rationale verbatim.
        "safety_overrides": safety_overrides_summary,
        "points": normalized_points,
        "advisory_start": normalized_points[0] if normalized_points else None,
        "true_start": current_point,
        "target": target,
        "target_map_heading_degrees": target_heading,
        "target_reported_heading_degrees": target_reported_heading,
        "target_heading_degrees": target_reported_heading,
        # 🚨 Never asserted from `toward`. Filled in by the VIO calibration
        # drive, which measures true facing independently; stays None-with-a-
        # reason on every other path. See _start_alignment_evidence.
        "start_geometry": _unmeasured_start_geometry(
            "dry_run"
            if dry_run
            else (
                "not_measured_outside_vio_turn_mode"
                if turn_mode != "vio"
                else "calibration_drive_not_run"
            )
        ),
        "heading_calibration": (
            {
                "model": "mirror",
                "formula": (
                    "target_toward_heading = toward_mirror_degrees - target_map_heading"
                ),
                "toward_mirror_degrees": toward_mirror_degrees,
                "target_map_heading_degrees": target_heading,
                "calibrated_forward_heading_offset_degrees": (
                    calibrated_forward_heading_offset_degrees
                ),
                "calibrated_forward_heading_offset_applied": False,
                "target_reported_heading_degrees": target_reported_heading,
            }
            if turn_mode == "night"
            else {
                "formula": (
                    "target_reported_heading = "
                    "target_map_heading - calibrated_forward_heading_offset"
                ),
                "target_map_heading_degrees": target_heading,
                "calibrated_forward_heading_offset_degrees": (
                    calibrated_forward_heading_offset_degrees
                ),
                "target_reported_heading_degrees": target_reported_heading,
            }
        ),
        "initial_linear_command_selection": initial_selection,
        "turn_mode": turn_mode,
        "vio": {
            "initial_vio_state": initial_vio_reading["vio_state"],
            "initial_vio_feed": initial_vio_feed,
            "initial_vision_heading": initial_vio_reading["vision_heading"],
            "provided_offset_degrees": vio_heading_offset_degrees,
            "offset_degrees": vio_heading_offset_degrees,
            "offset_source": (
                "provided" if vio_heading_offset_degrees is not None else None
            ),
            "target_vision_heading": None,
            "calibration": None,
            "formula": (
                "target_vision_heading = target_map_heading - "
                "(map_motion_heading - vision_heading)"
            ),
        },
        "initial_telemetry": initial_telemetry,
        "final_telemetry": initial_telemetry,
        "runtime_safety": runtime_safety,
        # The liveness report the gate below actually judged, AFTER waiting for
        # the queue to drain. Recorded so a refusal can be told apart from a
        # settle that timed out -- without it, `command_queue_backlogged` looks
        # identical whether the run waited 6 s or never waited at all.
        "queue_settle": queue_settle,
        "safety_gates": gates,
        "blockers": blockers,
        "commands_sent": 0,
        "turn_commands_sent": 0,
        "linear_commands_sent": 0,
        # App-parity cadence (opt-in). Echoed so a run's numbers can be read back
        # without guessing which motion model produced them.
        "motion_refresh_interval_ms": motion_refresh_interval_ms,
        "motion_refresh_commands_sent": 0,
        "app_speed_scale": _app_speed_scale_report(
            linear_speed_fast, vio_angular_speed
        ),
        "calibration_commands_sent": 0,
        "realignments": [],
        **({"night_aim": []} if turn_mode == "night" else {}),
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

    # Shared budget for the post-turn and mid-drive VIO corrections. A turn can
    # translate the mower enough to change the bearing to a short waypoint; the
    # correction budget must cover that pre-linear case as well as later drift.
    realignments_used = 0
    turn_result: dict[str, Any]
    if turn_mode == "vio":
        vio_info = result["vio"]
        if dry_run:
            turn_result = {
                "mode": "vio_turn",
                "dry_run": True,
                "stop_reason": "dry_run",
                "commands_sent": 0,
                "command_results": [],
                "planned": {
                    "calibration_drive": (
                        "skipped_offset_provided"
                        if vio_heading_offset_degrees is not None
                        else {
                            "command": "send_movement",
                            "kwargs": {
                                "linear_speed": linear_speed_fast,
                                "angular_speed": 0,
                            },
                            "max_pulses": max(1, vio_calibration_pulse_count),
                        }
                    ),
                    "turn_primitive": "vio_turn_to_heading",
                    "angular_speed": vio_angular_speed,
                    "max_commands": vio_turn_max_commands,
                    "heading_tolerance_degrees": heading_tolerance_degrees,
                    "formula": vio_info["formula"],
                },
            }
        else:
            offset = vio_heading_offset_degrees
            calibration: dict[str, Any] | None = None
            if offset is None:
                calibration = await _vio_segment_calibration_drive(
                    coordinator,
                    prefer_ble=prefer_ble,
                    # Live 2026-07-11: speed-200 pulses barely move this unit
                    # (~1-2cm real per 2s pulse; firmware ramp eats them) while
                    # speed-400 pulses cover ~8cm -- calibrate at fast speed,
                    # with pulses long enough and a wait matching the ~4s
                    # position-feed latency.
                    linear_speed=linear_speed_fast,
                    max_pulses=max(1, vio_calibration_pulse_count),
                    pulse_duration_ms=2000.0,
                    refresh_wait_seconds=4.0,
                )
                vio_info["calibration"] = calibration
                result["calibration_commands_sent"] = calibration["pulses_sent"]
                result["commands_sent"] += calibration["pulses_sent"]
                result["command_results"].extend(calibration["command_results"])
                if not calibration["passed"]:
                    result["phases"].append(
                        {
                            "name": "turn_to_target_heading",
                            "turn_mode": "vio",
                            "passed": False,
                            "result": calibration,
                        }
                    )
                    result["final_telemetry"] = _custom_path_telemetry_snapshot(
                        coordinator
                    )
                    result["stop_reason"] = "vio_calibration_failed"
                    return result
                offset = calibration["offset_degrees"]
                vio_info["offset_source"] = "calibration_drive"
            vio_info["offset_degrees"] = offset
            # The calibration drive moved the mower: re-anchor position, target
            # heading, and completion on fresh telemetry before turning.
            post_calibration = _custom_path_telemetry_snapshot(coordinator)
            result["final_telemetry"] = post_calibration
            refreshed_point = _raw_segment_current_point(post_calibration)
            if refreshed_point is not None:
                current_point = refreshed_point
            completion_status = _manual_velocity_completion_status(
                normalized_points,
                post_calibration,
                waypoint_tolerance=waypoint_tolerance,
            )
            result["completion_status"] = completion_status
            if completion_status["complete"]:
                result["stop_reason"] = "target_reached"
                return result
            if current_point is None or target is None:
                result["stop_reason"] = "position_unavailable"
                return result
            target_heading = _path_heading_degrees(current_point, target)
            result["target_map_heading_degrees"] = target_heading
            # The calibration drive just measured true facing without going
            # anywhere near `toward`. That makes this the one non-circular
            # moment to record whether the segment actually starts aligned --
            # the check four runs were wrongly credited with on 2026-09-04.
            result["start_geometry"] = (
                _start_alignment_evidence(
                    measured_map_facing_degrees=calibration.get(
                        "map_motion_heading_degrees"
                    ),
                    target_map_heading_degrees=target_heading,
                )
                if calibration is not None
                else _unmeasured_start_geometry(
                    "vio_heading_offset_provided_no_calibration_drive"
                )
            )
            target_vision = _normalized_heading_degrees(
                float(target_heading) - float(offset)
            )
            vio_info["target_vision_heading"] = target_vision
            # STAGED, and only here. This is the segment's OPENING turn -- the one
            # rotation nothing can preflight, because no feed reports a stationary
            # mower's orientation (beta19), so an operator clicking a point behind
            # the mower can only discover it as a pre-dispatch refusal. The
            # mid-drive re-aim and the post-turn correction below deliberately
            # keep calling `_vio_turn_to_heading` directly: their rotations are
            # small by construction, so infeasibility there means something is
            # wrong rather than something is large.
            turn_result = await _vio_turn_to_heading_staged(
                coordinator,
                target_vision_heading=float(target_vision or 0.0),
                heading_tolerance_degrees=heading_tolerance_degrees,
                angular_speed=vio_angular_speed,
                max_commands=vio_turn_max_commands,
                # Forward the segment's refresh cadence into the turn. Without
                # this the executor's turns always ran single-shot at ~13 deg
                # per command even when the segment was given 200, which is why
                # a 176 deg turn exhausted an 8-command budget live 2026-07-27
                # and the segment never reached its linear phase.
                motion_refresh_interval_ms=motion_refresh_interval_ms,
                turn_degrees_per_second=turn_degrees_per_second,
                max_displacement_m=max_turn_translation_distance,
                prefer_ble=prefer_ble,
                dry_run=False,
                confirm_blades_off=confirm_blades_off,
                confirm_clear_area=confirm_clear_area,
                ha_state=ha_state,
                active_route=active_route,
            )
    elif turn_mode == "night":
        turn_result = await _raw_pymammotion_turn_to_heading(
            coordinator,
            target_heading_degrees=target_reported_heading,
            heading_tolerance_degrees=heading_tolerance_degrees,
            angular_speed_fast=night_angular_speed,
            angular_speed_slow=night_angular_speed,
            slow_turn_threshold_degrees=slow_turn_threshold_degrees,
            max_commands=max_turn_commands,
            min_heading_change_degrees=min_heading_change_degrees,
            max_translation_distance=max_turn_translation_distance,
            pulse_duration_ms=turn_pulse_duration_ms,
            prefer_ble=prefer_ble,
            motion_refresh_interval_ms=motion_refresh_interval_ms,
            sample_delays=tuple(sample_delays),
            dry_run=dry_run,
            confirm_blades_off=confirm_blades_off,
            confirm_clear_area=confirm_clear_area,
            ha_state=ha_state,
            active_route=active_route,
        )
    else:
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
            "turn_mode": turn_mode,
            "passed": turn_result.get("stop_reason")
            in {"dry_run", "target_heading_reached"},
            "result": turn_result,
        }
    )
    result["turn_commands_sent"] = int(turn_result.get("commands_sent") or 0)
    result["commands_sent"] += result["turn_commands_sent"]
    # Fold the turn primitive's refresh writes into the segment total. It keeps
    # its own counter, and until beta31 nothing rolled it up: the segment field
    # was accumulated ONLY inside the linear loop, so Gate 5 attempt 5 segment 1
    # reported 6 refresh writes against an actual 15.
    result["motion_refresh_commands_sent"] += int(
        turn_result.get("motion_refresh_commands_sent") or 0
    )
    result["command_results"].extend(turn_result.get("command_results") or [])
    result["samples"].extend(
        sample
        for sample in turn_result.get("samples", [])[1:]
        if isinstance(sample, dict)
    )
    if turn_mode == "vio" and not dry_run:
        # The VIO primitive reports headings, not telemetry snapshots; take a
        # fresh post-turn snapshot so the linear phase baselines correctly.
        result["final_telemetry"] = _custom_path_telemetry_snapshot(coordinator)
    else:
        result["final_telemetry"] = turn_result.get(
            "final_telemetry", result["final_telemetry"]
        )

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
        # Keep the pre-dispatch refusal distinguishable from a turn that ran and
        # fell short: `turn_budget_infeasible` sent zero turn commands, while
        # `turn_phase_incomplete` covers every mid-turn failure (budget
        # exhaustion, stale feed, lost transport, ...). Surface the refusal's
        # budget math on the segment result for offline diagnosis.
        if turn_result.get("stop_reason") == "turn_budget_infeasible":
            result["turn_feasibility"] = turn_result.get("turn_feasibility")
            result["stop_reason"] = "turn_budget_infeasible"
        else:
            result["stop_reason"] = "turn_phase_incomplete"
        # A staged opening turn fails with its own reasons, and flattening them
        # all into `turn_phase_incomplete` would hide WHICH stage broke and how
        # much translation it had already spent. The stage record is the only
        # place that says so.
        if turn_result.get("staged_turn"):
            result["staged_turn"] = {
                "stages": turn_result.get("stages"),
                "stop_reason": turn_result.get("stop_reason"),
                "failed_stage_reason": turn_result.get("failed_stage_reason"),
                "final_displacement_m": turn_result.get("final_displacement_m"),
                "direct_refusal": turn_result.get("direct_refusal"),
            }
        return result

    if turn_mode == "night" and int(turn_result.get("commands_sent") or 0) == 0:
        # A zero-command success can be a latched ``toward`` reading inside
        # tolerance. Do not begin the linear phase without an observed turn.
        result["stop_reason"] = "night_heading_unverified"
        return result

    # A nominally in-place VIO turn can translate materially. Gate 4 on
    # 2026-07-31 moved 14.4 cm while turning toward a 30 cm waypoint, changing
    # the required bearing by ~8 degrees. The original turn correctly reached
    # its PRE-turn target heading, but the executor then drove from the fresh
    # position without checking the fresh bearing and missed by 18 cm. Recompute
    # from post-turn telemetry and correct before any nonzero linear dispatch.
    if turn_mode == "vio":
        turn_displacement = turn_result.get("final_displacement_m")
        if turn_displacement is not None and float(turn_displacement) > 0.0025:
            post_turn = _custom_path_telemetry_snapshot(coordinator)
            result["final_telemetry"] = post_turn
            current_after_turn = _raw_segment_current_point(post_turn)
            reading_after_turn = _vio_reading(coordinator)
            offset_after_turn = result["vio"].get("offset_degrees")
            if (
                current_after_turn is None
                or target is None
                or offset_after_turn is None
                or reading_after_turn["vio_state"] != _VIO_STATE_ACTIVE
                or reading_after_turn["vision_heading"] is None
            ):
                result["stop_reason"] = "post_turn_alignment_unavailable"
                return result

            fresh_bearing = _path_heading_degrees(current_after_turn, target)
            fresh_facing = (
                float(reading_after_turn["vision_heading"]) + float(offset_after_turn)
            ) % 360
            fresh_aim_error = _heading_error_degrees(fresh_facing, fresh_bearing)
            # Still floored by the turn primitive's own tolerance: asking for a
            # landing tighter than the turn can deliver would spend the budget
            # chasing a target it cannot hit.
            alignment_tolerance = min(
                float(heading_tolerance_degrees),
                _POST_TURN_ALIGNMENT_TOLERANCE_DEGREES,
            )
            alignment: dict[str, Any] = {
                "turn_displacement_m": float(turn_displacement),
                "before": {
                    "facing_degrees": round(fresh_facing, 3),
                    "bearing_degrees": round(fresh_bearing, 3),
                    "aim_error_degrees": round(fresh_aim_error, 3),
                },
                "alignment_tolerance_degrees": alignment_tolerance,
                "correction_attempted": False,
                "passed": abs(fresh_aim_error) <= alignment_tolerance,
            }
            result["post_turn_alignment"] = alignment
            current_point = current_after_turn
            result["target_map_heading_degrees"] = fresh_bearing

            if not alignment["passed"]:
                if realignments_used >= vio_max_realignments:
                    result["stop_reason"] = "post_turn_realign_budget_exhausted"
                    return result
                realignments_used += 1
                correction_target = _normalized_heading_degrees(
                    fresh_bearing - float(offset_after_turn)
                )
                alignment["correction_attempted"] = True
                alignment["target_vision_heading"] = correction_target
                correction = await _vio_turn_to_heading(
                    coordinator,
                    target_vision_heading=float(correction_target or 0.0),
                    heading_tolerance_degrees=alignment_tolerance,
                    angular_speed=vio_angular_speed,
                    # 🏁 beta43: the SAME budget as any other turn. This was
                    # `min(2, vio_turn_max_commands)`, an uncommented cap that
                    # predates beta40 -- and beta40 tightened this correction's
                    # tolerance to 10 deg without revisiting it. That
                    # combination killed a Gate 5 attempt on 2026-08-12:
                    #
                    #   segment 3, post-turn aim error 29.647 deg, tolerance 10
                    #   -> required_rotation 19.647, estimated_commands_needed 3
                    #   -> max_commands 2 -> turn_budget_infeasible
                    #   -> post_turn_realign_incomplete, segment never drove
                    #
                    # A TIGHTER tolerance needs MORE rotation, and the pulse
                    # policy shortens each pulse as the error closes, so the
                    # modelled ladder was [691, 442, 283] ms -- three commands
                    # for what fits in two at the old 18 deg tolerance. The
                    # feasible envelope at 2 commands is 21.50 deg; at 4 it is
                    # 49.50, against a ~30 deg worst case ever observed.
                    #
                    # The command cap was never what bounded the cost --
                    # `max_displacement_m` is. Measured over 92 recorded turns,
                    # translation is 0.00098 m/deg median and 0.00487 worst
                    # ever, so the 19.647 deg above costs ~0.019 m typical and
                    # ~0.096 m worst against a 0.30 m allowance. Removing the
                    # cap spends translation we can afford to avoid losing the
                    # whole segment.
                    #
                    # ⚠️ beta40 validated this gate on a 16.551 deg correction,
                    # which fits in two commands. The budget was never stressed
                    # until a real gate ran.
                    max_commands=vio_turn_max_commands,
                    motion_refresh_interval_ms=motion_refresh_interval_ms,
                    turn_degrees_per_second=turn_degrees_per_second,
                    max_displacement_m=max_turn_translation_distance,
                    prefer_ble=prefer_ble,
                    dry_run=False,
                    confirm_blades_off=confirm_blades_off,
                    confirm_clear_area=confirm_clear_area,
                    ha_state=ha_state,
                    active_route=active_route,
                )
                correction_commands = int(correction.get("commands_sent") or 0)
                result["turn_commands_sent"] += correction_commands
                result["commands_sent"] += correction_commands
                result["motion_refresh_commands_sent"] += int(
                    correction.get("motion_refresh_commands_sent") or 0
                )
                result["command_results"].extend(
                    correction.get("command_results") or []
                )
                result["realignments"].append(
                    {
                        "before_linear": True,
                        "facing_degrees": round(fresh_facing, 3),
                        "bearing_degrees": round(fresh_bearing, 3),
                        "aim_error_degrees": round(fresh_aim_error, 3),
                        "stop_reason": correction.get("stop_reason"),
                        # A refusal must carry its own arithmetic. On 2026-08-12
                        # this record said only `turn_budget_infeasible`, and
                        # working out WHY meant replaying the shipped
                        # feasibility function by hand against a guessed set of
                        # inputs. The primitive already computes this dict;
                        # there is no reason to make the next reader re-derive
                        # it. Same principle as the per-command records: the
                        # evidence file should answer the question, not pose it.
                        "turn_feasibility": correction.get("turn_feasibility"),
                        "max_commands": vio_turn_max_commands,
                        "alignment_tolerance_degrees": alignment_tolerance,
                    }
                )
                if correction.get("stop_reason") != "target_heading_reached":
                    alignment["correction_stop_reason"] = correction.get("stop_reason")
                    alignment["correction_turn_feasibility"] = correction.get(
                        "turn_feasibility"
                    )
                    result["stop_reason"] = "post_turn_realign_incomplete"
                    return result

                # A correction can itself translate, so verify once more from
                # fresh position and VIO heading. Never begin a linear pulse on
                # the assumption that the correction pivot was perfectly fixed.
                after_correction = _custom_path_telemetry_snapshot(coordinator)
                result["final_telemetry"] = after_correction
                corrected_point = _raw_segment_current_point(after_correction)
                corrected_reading = _vio_reading(coordinator)
                if (
                    corrected_point is None
                    or corrected_reading["vio_state"] != _VIO_STATE_ACTIVE
                    or corrected_reading["vision_heading"] is None
                ):
                    result["stop_reason"] = "post_turn_alignment_unavailable"
                    return result
                corrected_bearing = _path_heading_degrees(corrected_point, target)
                corrected_facing = (
                    float(corrected_reading["vision_heading"])
                    + float(offset_after_turn)
                ) % 360
                corrected_error = _heading_error_degrees(
                    corrected_facing, corrected_bearing
                )
                alignment["after"] = {
                    "facing_degrees": round(corrected_facing, 3),
                    "bearing_degrees": round(corrected_bearing, 3),
                    "aim_error_degrees": round(corrected_error, 3),
                }
                alignment["passed"] = abs(corrected_error) <= alignment_tolerance
                current_point = corrected_point
                result["target_map_heading_degrees"] = corrected_bearing
                if not alignment["passed"]:
                    result["stop_reason"] = "post_turn_alignment_incomplete"
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
    # A dead report stream is only a credible diagnosis if the feed was
    # demonstrably alive earlier in THIS run. Bit-identical coordinates from the
    # very first pulse are better explained by a mower that never moved; the
    # 2026-07-19 signature was ten good pulses followed by three frozen ones.
    feed_moved_earlier = False
    # Measured pulse distances, normalised to the fallback's eleven non-zero
    # writes and kept separate by commanded speed. Bounded pulses are useful
    # samples once normalised; excluding them made the 1.06 m fallback
    # permanently load-bearing on every short Gate 4 segment.
    observed_pulse_distances_by_speed: dict[int, list[float]] = {}
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
        if prefer_ble and not _transport_is_ble(coordinator):
            result["stop_reason"] = "ble_transport_lost"
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
            "sent_at_utc": _utc_timestamp(),
            "prefer_ble": prefer_ble,
            "kwargs": {
                "linear_speed": selection["linear_speed"],
                "angular_speed": 0,
            },
            "selection": selection,
        }
        # Scale the pulse when less than one pulse of travel remains, so the
        # final approach lands on the target instead of stepping past it.
        final_approach = _final_approach_pulse_ms(
            distance_to_target=selection.get("distance_to_target"),
            observed_pulse_distances=observed_pulse_distances_by_speed.get(
                int(selection["linear_speed"]), []
            ),
            default_metres_per_pulse=final_approach_metres_per_pulse,
            pulse_duration_ms=linear_pulse_duration_ms,
            refresh_interval_ms=motion_refresh_interval_ms,
        )
        command_result["final_approach"] = final_approach
        pulse_ms = float(final_approach["pulse_duration_ms"])
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
        # Hold the pulse open the way the app does. With the default interval of
        # 0 this is the proven single-shot behaviour; with a positive interval the
        # movement command is re-sent every interval for the pulse duration.
        # Refreshes are counted separately from pulses on purpose: they must not
        # inflate `linear_commands_sent`, which drives the pulse ceilings.
        command_result["motion_refresh"] = await _motion_refresh_window(
            coordinator,
            resend=functools.partial(
                _send_manager_command_with_args,
                coordinator,
                "send_movement",
                prefer_ble=prefer_ble,
                command_kwargs=command_result["kwargs"],
            ),
            duration_seconds=pulse_ms / 1000,
            refresh_interval_ms=motion_refresh_interval_ms,
            max_refresh_commands=final_approach.get("refresh_command_limit"),
        )
        result["motion_refresh_commands_sent"] += command_result["motion_refresh"][
            "refresh_commands_sent"
        ]
        command_result["stop_result"] = await _manual_velocity_stop_attempt(
            coordinator, use_wifi=not prefer_ble
        )
        if not (command_result["stop_result"] or {}).get("ok"):
            # Never keep driving when stops are not deliverable (BLE cooldown,
            # transport loss); abort immediately.
            result["stop_reason"] = "stop_failed_aborting"
            return result
        command_result[
            "post_command_feedback_refresh"
        ] = await _refresh_position_after_raw_motion(
            coordinator,
            # The VIO path immediately enters the bounded position-settle loop,
            # which requests the same reports once per second. Do not sleep two
            # seconds here and then start that loop; night and legacy retain
            # their established timing byte-for-byte.
            settle_seconds=0.0 if turn_mode == "vio" else 2.0,
        )
        # The map-local feed lags ~4s and jumps: wait for this pulse's motion to
        # register and settle before sampling, so the samples below (and thus the
        # progress/completion checks) reflect THIS pulse instead of leaking a prior
        # pulse's delayed jump (live 2026-07-15). A blocked pulse never registers
        # movement and times out settled=False, which the existing progress check
        # then treats as no-progress.
        position_settle = await _settle_linear_position_feed(coordinator, before)
        command_result["position_settled"] = position_settle["settled"]
        command_result["position_moved"] = position_settle["moved"]
        command_result["position_settle_wait_seconds"] = position_settle["wait_seconds"]
        command_result["position_feed_stale"] = position_settle["feed_stale"]
        command_result["position_settle_polls"] = position_settle["settle_polls"]
        if position_settle["moved"]:
            feed_moved_earlier = True
        post_feedback_queue_settle: dict[str, Any] | None = None
        if turn_mode == "vio":
            # The position-settle loop requests five reports on every poll.
            # Those requests share the BLE command queue with motion. The first
            # hardware run without the old blind three-second wait settled its
            # position in 2.03 s but refused the next pulse on
            # `command_queue_backlogged`. Wait for the queue itself instead:
            # this returns immediately when empty, remains bounded at six
            # seconds, and never converts a persistent backlog into a pass.
            queue_started = time.monotonic()
            post_feedback_queue_settle = await _settle_ble_command_queue(coordinator)
            command_result["post_feedback_queue_settle"] = {
                **post_feedback_queue_settle,
                "duration_ms": round(
                    (time.monotonic() - queue_started) * 1000,
                    3,
                ),
            }
        # Phantom-motion investigation instrumentation (capture only): log both
        # position sources + RTK quality so a later run can tell a real move from a
        # feed-jump on a no-op pulse.
        command_result["position_source_comparison"] = _position_source_comparison(
            coordinator
        )

        command_samples: list[dict[str, Any]] = []
        settled_telemetry = position_settle.get("telemetry")
        reuse_settled_telemetry = (
            turn_mode == "vio"
            and position_settle["settled"] is True
            and isinstance(settled_telemetry, dict)
        )
        if reuse_settled_telemetry:
            # Real Go has already paid the bounded 1-6 second position-settle
            # wait. Waiting through [0, 3] afterward sampled the same stopped
            # mower and made the two feedback windows additive. Keep one
            # per-command record, but make its source and actual wait explicit.
            sample = {
                "label": f"linear_{command_index}_position_settled",
                "command_index": command_index,
                "delay_seconds": position_settle["wait_seconds"],
                "source": "position_settle",
                "telemetry": settled_telemetry,
            }
            result["samples"].append(sample)
            command_samples.append(sample)
            command_result["post_settle_feedback"] = {
                "source": "position_settle",
                "additional_wait_seconds": 0.0,
                "requested_sample_delays_skipped": list(sample_delays),
            }
        else:
            previous_delay = 0.0
            for sample_index, delay in enumerate(sample_delays):
                await asyncio.sleep(max(0.0, float(delay) - previous_delay))
                previous_delay = float(delay)
                sample_telemetry = _custom_path_telemetry_snapshot(coordinator)
                sample = {
                    "label": (
                        f"linear_{command_index}_sample_{sample_index + 1}_{delay:g}s"
                    ),
                    "command_index": command_index,
                    "delay_seconds": float(delay),
                    "telemetry": sample_telemetry,
                }
                result["samples"].append(sample)
                command_samples.append(sample)
            if turn_mode == "vio":
                command_result["post_settle_feedback"] = {
                    "source": "post_settle_samples",
                    "additional_wait_seconds": max(
                        (float(delay) for delay in sample_delays), default=0.0
                    ),
                    "requested_sample_delays_skipped": [],
                }

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
        # Feed every measured pulse back into the final-approach scale factor.
        # Normalising by its actual non-zero write count makes bounded pulses
        # comparable with the eleven-write fallback instead of excluding the
        # only samples short Gate 4 segments ever produce.
        measured_distance = (progress.get("measured_delta") or {}).get("distance")
        if (
            motion_refresh_interval_ms > 0
            and measured_distance is not None
            and float(measured_distance) > 0
        ):
            refresh_commands_sent = int(
                command_result["motion_refresh"]["refresh_commands_sent"]
            )
            normalised_distance = _normalised_linear_pulse_distance(
                float(measured_distance), refresh_commands_sent
            )
            observed_pulse_distances_by_speed.setdefault(
                int(selection["linear_speed"]), []
            ).append(normalised_distance)
            command_result["final_approach_observation"] = {
                "measured_distance": float(measured_distance),
                "nonzero_writes": refresh_commands_sent + 1,
                "normalised_metres_per_eleven_writes": round(normalised_distance, 4),
            }
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
        if (
            post_feedback_queue_settle is not None
            and not post_feedback_queue_settle.get("live")
        ):
            result["stop_reason"] = "ble_link_not_ready_after_feedback"
            result["post_feedback_queue_settle"] = post_feedback_queue_settle
            return result
        quality = _manual_velocity_quality_degradation(baseline_telemetry, after)
        if quality["degraded"]:
            result["quality_degradation"] = quality
            result["stop_reason"] = "telemetry_quality_degraded"
            return result
        if not _blade_reported_safe(after):
            result["stop_reason"] = "blade_unsafe"
            return result
        if turn_mode == "vio":
            reading = _vio_reading(coordinator)
            vio_info = result["vio"]
            measured_delta = progress.get("measured_delta") or {}
            # Continuous offset refresh: every real pulse is a fresh (and much
            # longer) calibration vector -- motion heading in the map frame plus
            # the concurrent VIO body heading -- so keep the map->VIO offset
            # current instead of trusting the short initial baseline forever.
            if (
                reading["vio_state"] == _VIO_STATE_ACTIVE
                and reading["vision_heading"] is not None
                and measured_delta.get("distance") is not None
                and float(measured_delta["distance"]) >= 0.05
                and measured_delta.get("dx") is not None
                and measured_delta.get("dy") is not None
            ):
                motion_heading = (
                    math.degrees(
                        math.atan2(
                            float(measured_delta["dy"]),
                            float(measured_delta["dx"]),
                        )
                    )
                    + 360
                ) % 360
                vio_info["offset_degrees"] = _normalized_heading_degrees(
                    motion_heading - float(reading["vision_heading"])
                )
                vio_info["offset_source"] = "linear_refresh"
            # Mid-drive re-aim: pure forward pulses cannot steer, so when the
            # estimated facing drifts off the bearing to the target (live run
            # 2026-07-11 drifted ~25 deg and sailed past the waypoint), run a
            # bounded VIO turn correction and resume driving.
            offset_now = vio_info.get("offset_degrees")
            current_now = _raw_segment_current_point(after)
            if (
                offset_now is not None
                and reading["vio_state"] == _VIO_STATE_ACTIVE
                and reading["vision_heading"] is not None
                and current_now is not None
                and target is not None
                # A re-alignment only has value when another forward command
                # can follow it. Live 2026-08-02 the sole linear command ended
                # 8.46 cm from target, then three otherwise-successful turns
                # added ~23 cm of drift even though the linear budget was
                # already exhausted. Stop boundedly instead.
                #
                # This must test `effective_linear_ceiling`, NOT
                # `max_linear_commands`. They are the same number only while
                # `max_linear_pulse_ceiling` is None; the moment loop-to-tolerance
                # is enabled the ceiling becomes the pulse ceiling and this
                # comparison would silently stop cross-track correction after
                # `max_linear_commands` pulses while the linear loop kept
                # driving -- a mower correcting nothing for the rest of a long
                # segment. Recorded as a prerequisite in
                # docs/HANDOVER-beta31-20260809.md section 5 and fixed here
                # BEFORE anyone enables the mode, which is the whole point.
                and command_index < effective_linear_ceiling
            ):
                facing = (float(reading["vision_heading"]) + float(offset_now)) % 360
                bearing = _path_heading_degrees(current_now, target)
                aim_error = _heading_error_degrees(facing, bearing)
                if _requires_reverse_recovery(aim_error):
                    result["reverse_recovery_guard"] = {
                        "after_linear_pulse": command_index,
                        "facing_degrees": round(facing, 3),
                        "bearing_degrees": round(bearing, 3),
                        "aim_error_degrees": round(aim_error, 3),
                        "max_forward_realignment_degrees": (
                            _MAX_FORWARD_REALIGNMENT_DEGREES
                        ),
                        "reason": "target_requires_reverse_recovery",
                    }
                    result["stop_reason"] = "target_requires_reverse_recovery"
                    return result
                # A re-aim is skipped when driving straight on would still land
                # inside the tolerance disc: correcting then spends turn commands
                # and adds turn translation to buy nothing. Recorded rather than
                # silent -- a suppressed re-aim is a decision worth seeing in the
                # run record.
                #
                # ⚠️ The suppression record is kept on its ORIGINAL condition
                # (aim error past the correctable floor, but the projection lands
                # inside) even though that can no longer coincide with a fired
                # correction. Since 2026-08-17 `needs_correction` REQUIRES
                # `not already_lands_inside`, so a naive `needs_correction and
                # already_lands_inside` would be dead code and every suppression
                # would silently vanish from the run record.
                distance_to_target = math.hypot(
                    float(target["x"]) - float(current_now["x"]),
                    float(target["y"]) - float(current_now["y"]),
                )
                perpendicular_miss = abs(
                    distance_to_target
                    * math.sin(math.radians(min(abs(aim_error), 90.0)))
                )
                # The SAME figure the final-approach planner will use for the
                # next pulse, from the same helper, so the guard cannot be
                # projecting a step the executor is not about to take.
                guard_metres_per_pulse = _effective_metres_per_pulse(
                    observed_pulse_distances_by_speed.get(
                        int(selection["linear_speed"]), []
                    ),
                    final_approach_metres_per_pulse,
                )
                # 🔑 THE TRIGGER IS A DISTANCE, NOT AN ANGLE (2026-08-17).
                #
                # This read `abs(aim_error) > vio_realign_threshold_degrees and
                # abs(aim_error) > heading_tolerance_degrees` -- pure angle,
                # effective threshold 18 deg. But the quantity that decides the
                # run is the MISS, `remaining_range * sin(aim_error)`, and the
                # two diverge with range. `already_lands_inside` (beta42) has
                # always reasoned in projected metres, yet it could only ever
                # SUPPRESS a correction, never fire one -- so the controller was
                # blind in precisely the long-leg regime: 17 deg of aim error
                # with 14 m to run is a 4.1 m miss and fired nothing, because
                # 17 < 18.
                #
                # Now the projection decides, and the angle term is only the
                # floor below which a correction cannot help
                # (`_MIN_CORRECTABLE_AIM_ERROR_DEGREES` -- read it before
                # touching this). Short legs are close to unaffected: inside
                # ~1 m the suppression guard was already the binding term and
                # still is. What changes is that a leg can now be corrected
                # while it still has the range to make correcting cheap.
                #
                # The decision itself lives in `_mid_drive_realign_decision` so
                # it can be inspected without driving a mower through a
                # 1,500-line executor. It was inline until 2026-08-17, and the
                # nine tests covering this branch all still passed when the
                # trigger changed meaning -- because none of them could reach a
                # long-range geometry.
                decision = _mid_drive_realign_decision(
                    distance_to_target_m=distance_to_target,
                    aim_error_degrees=aim_error,
                    waypoint_tolerance=waypoint_tolerance,
                    metres_per_pulse=guard_metres_per_pulse,
                    realign_threshold_degrees=float(vio_realign_threshold_degrees),
                )
                needs_correction = decision["needs_correction"]
                if (
                    decision["past_correctable_floor"]
                    and decision["already_lands_inside"]
                ):
                    result.setdefault("realignments_suppressed", []).append(
                        {
                            "after_linear_pulse": command_index,
                            # `facing`/`bearing` were missing here while the
                            # `realignments` record carried both, so every
                            # analysis of a suppression had to reconstruct them.
                            "facing_degrees": round(float(facing), 3),
                            "bearing_degrees": round(float(bearing), 3),
                            "aim_error_degrees": round(aim_error, 3),
                            "distance_to_target_m": round(distance_to_target, 4),
                            # Kept for continuity with every run before beta42:
                            # this is what the guard USED to decide on.
                            "perpendicular_miss_m": round(perpendicular_miss, 4),
                            # What it decides on now -- the miss at the end of
                            # the next pulse, not at the closest approach.
                            "projected_landing_m": round(
                                decision["projected_landing_m"], 4
                            ),
                            "metres_per_pulse": round(guard_metres_per_pulse, 4),
                            "waypoint_tolerance": waypoint_tolerance,
                            "reason": "already_lands_inside_tolerance",
                        }
                    )
                if needs_correction:
                    if realignments_used >= vio_max_realignments:
                        result["stop_reason"] = "vio_realign_budget_exhausted"
                        return result
                    realignments_used += 1
                    realign_target = _normalized_heading_degrees(
                        bearing - float(offset_now)
                    )
                    realign_result = await _vio_turn_to_heading(
                        coordinator,
                        target_vision_heading=float(realign_target or 0.0),
                        # 🔑 Corrects to the SAME floor the post-turn gate uses
                        # (10 deg), not to `heading_tolerance_degrees` (18 on the
                        # accepted profile). Changed 2026-08-17 with the
                        # projected-miss trigger above, and the two belong
                        # together.
                        #
                        # An 18 deg close is what let the 1.65 m segment diverge
                        # on 2026-08-15: every correction returned
                        # `target_heading_reached` while leaving 9.7 / 11.5 /
                        # 13.6 deg of residual against a bearing rotating
                        # -3.2 / -9.7 / -15.4 deg per pulse. The corrections were
                        # not failing -- they were succeeding at a tolerance too
                        # loose to converge.
                        #
                        # It also sets the terminal accuracy of a long leg. With
                        # a correction available every pulse the landing is
                        # ~`pulse_length * sin(residual)`, INDEPENDENT of leg
                        # length: 0.41 * sin(18 deg) = 0.127 m against an 0.15
                        # tolerance is 15% of margin, while 0.41 * sin(10 deg) =
                        # 0.071 m is comfortable. That is what makes a 6 m leg
                        # credible at all.
                        #
                        # 🔑 AND IT IS WHAT CREATES THE DEADBAND. The trigger
                        # floor is `max(vio_realign_threshold_degrees, 10)` = 15
                        # on the default; this closes to 10. That 5 deg gap is
                        # not slack, it is the reason a correction cannot
                        # immediately re-trigger itself. Setting the threshold to
                        # 10 was tried on 2026-08-17 and reverted: floor equal to
                        # tolerance means a correction ending at 9.9 deg fires
                        # again next pulse, and an error just past the floor
                        # makes the primitive's entry check return
                        # `target_heading_reached` having sent nothing.
                        #
                        # ⚠️ So the far-field improvement this change buys is
                        # 18 deg -> 15 deg, NOT 18 -> 10. The turn primitive
                        # cannot hold better than 10 deg, and a deadband is
                        # mandatory above that. Do not "finish the job" by
                        # lowering the threshold without first shortening the
                        # actuation floor.
                        heading_tolerance_degrees=min(
                            float(heading_tolerance_degrees),
                            _POST_TURN_ALIGNMENT_TOLERANCE_DEGREES,
                        ),
                        angular_speed=vio_angular_speed,
                        max_commands=min(6, vio_turn_max_commands),
                        motion_refresh_interval_ms=motion_refresh_interval_ms,
                        turn_degrees_per_second=turn_degrees_per_second,
                        max_displacement_m=max_turn_translation_distance,
                        prefer_ble=prefer_ble,
                        dry_run=False,
                        confirm_blades_off=confirm_blades_off,
                        confirm_clear_area=confirm_clear_area,
                        ha_state=ha_state,
                        active_route=active_route,
                    )
                    realign_commands = int(realign_result.get("commands_sent") or 0)
                    result["turn_commands_sent"] += realign_commands
                    result["commands_sent"] += realign_commands
                    result["motion_refresh_commands_sent"] += int(
                        realign_result.get("motion_refresh_commands_sent") or 0
                    )
                    result["command_results"].extend(
                        realign_result.get("command_results") or []
                    )
                    result["realignments"].append(
                        {
                            "after_linear_pulse": command_index,
                            "facing_degrees": round(facing, 3),
                            "bearing_degrees": round(bearing, 3),
                            "aim_error_degrees": round(aim_error, 3),
                            "stop_reason": realign_result.get("stop_reason"),
                        }
                    )
                    if realign_result.get("stop_reason") != "target_heading_reached":
                        result["stop_reason"] = "vio_realign_incomplete"
                        return result
                    # Course corrected: the off-bearing pulse should not count
                    # toward the no-progress abort.
                    consecutive_no_progress = 0
                    continue
        elif turn_mode == "night":
            # Both values are map-frame bearings derived from RTK position; VIO
            # and ``toward`` are not used to decide this per-pulse aim check.
            # The bearing must start at the SETTLED POST-PULSE position. Using
            # the shared progress diagnostic's pre-pulse bearing misses a target
            # that the pulse has just passed: measured 2026-08-14, pulse 2
            # settled 0.08266 m from an 0.08 m target with the residual bearing
            # at 155.64 deg, but its pre-pulse 9.26 deg bearing allowed a third
            # forward write that worsened the landing to 0.11708 m. Keep the
            # shared diagnostic unchanged; this correction is night-only.
            movement_heading = progress.get("movement_vector_heading_degrees")
            night_residual_point = _raw_segment_current_point(after)
            night_bearing = (
                _path_heading_degrees(night_residual_point, target)
                if night_residual_point is not None
                else None
            )
            measured_distance = (progress.get("measured_delta") or {}).get("distance")
            night_distance_to_target = completion_status.get("distance_to_target")
            after_toward = (after.get("position") or {}).get("toward")
            night_aim: dict[str, Any] = {
                "after_linear_pulse": command_index,
                "movement_vector_heading_degrees": movement_heading,
                "bearing_to_target_degrees": night_bearing,
                "measured_distance_m": measured_distance,
                "distance_to_target_m": night_distance_to_target,
                "observed_toward_mirror_degrees": (
                    (float(movement_heading) + float(after_toward)) % 360
                    if movement_heading is not None and after_toward is not None
                    else None
                ),
            }
            if (
                movement_heading is not None
                and night_bearing is not None
                and measured_distance is not None
                and float(measured_distance) >= _NIGHT_MIN_AIM_BASELINE_M
            ):
                aim_error = _heading_error_degrees(
                    float(movement_heading), float(night_bearing)
                )
                night_aim["aim_error_degrees"] = round(aim_error, 3)
                if _requires_reverse_recovery(aim_error):
                    night_aim["decision"] = "reverse_recovery_required"
                    result["night_aim"].append(night_aim)
                    result["stop_reason"] = "target_requires_reverse_recovery"
                    return result
                if night_distance_to_target is not None and not (
                    _realign_cannot_improve_the_landing(
                        distance_to_target_m=float(night_distance_to_target),
                        aim_error_degrees=aim_error,
                        waypoint_tolerance=waypoint_tolerance,
                        metres_per_pulse=_effective_metres_per_pulse(
                            observed_pulse_distances_by_speed.get(
                                int(selection["linear_speed"]), []
                            ),
                            final_approach_metres_per_pulse,
                        ),
                    )
                ):
                    # Night cannot safely correct this mis-aim yet, so it stops
                    # rather than suppressing a correction and driving onward.
                    night_aim["decision"] = "stop_reaim_unavailable"
                    result["night_aim"].append(night_aim)
                    result["stop_reason"] = "night_reaim_required_but_unavailable"
                    return result
                night_aim["decision"] = "drive_on_projects_inside_tolerance"
            else:
                night_aim["decision"] = "below_aim_baseline"
            result["night_aim"].append(night_aim)
        if not progress["passed"]:
            # A stale feed and a stopped mower both read as "no progress", but only
            # one of them means the mower is fine. Bit-identical coordinates across
            # several settle polls mean the feed stopped updating, so every
            # position-derived guard below (waypoint tolerance, displacement caps,
            # progress) is judging a coordinate that no longer describes reality.
            # Stop commanding motion rather than pulsing blind: live 2026-07-19 the
            # mower drove 25.4cm across three such pulses while the executor
            # believed it had not moved at all, then aborted a healthy run.
            if position_settle["feed_stale"] and feed_moved_earlier:
                result["stop_reason"] = "telemetry_stream_stale"
                result["telemetry_stream_stale_hint"] = (
                    "Position feed returned bit-identical coordinates across "
                    f"{position_settle['settle_polls']} settle polls (a live feed "
                    "jitters ~2-4mm). The mower may still have been moving; "
                    "position-derived guards were judging a stale coordinate, so "
                    "motion was stopped rather than continued blind."
                )
                return result
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
    """Return whether a guarded multi-segment phase passed.

    A real segment passes when it *arrived*: ``target_reached`` with a valid,
    unblocked run. It deliberately does NOT require every per-pulse progress
    diagnostic to have cleared ``min_progress_distance``.

    Requiring that was a real bug. As the mower closes on a waypoint the
    remaining distance shrinks below one pulse, so the final approach pulses
    legitimately move less than the per-pulse threshold. A segment could
    therefore land on its target and still be marked ``passed: False``, which
    stopped the multi-segment run and meant later segments never executed. The
    genuinely-stuck case is already handled inside the executor by the
    consecutive-no-progress abort, which never reports ``target_reached``.
    Per-pulse diagnostics remain in the result for forensics.
    """
    if real_segment:
        return (
            segment_result.get("stop_reason") == "target_reached"
            and segment_result.get("valid") is True
            and not segment_result.get("blockers")
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
    safety_overrides: list[str] | tuple[str, ...] | None = None,
    confirm_blades_off: bool = False,
    confirm_clear_area: bool = False,
    prefer_ble: bool = True,
    ble_auto_recover: bool = True,
    max_real_segments: int = 1,
    split_leg_target_length_m: float | None = None,
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
    final_approach_metres_per_pulse: float = _DEFAULT_METRES_PER_LINEAR_PULSE,
    turn_degrees_per_second: float = _DEFAULT_TURN_DEGREES_PER_SECOND,
    turn_mode: str = "vio",
    night_angular_speed: int = _NIGHT_TURN_ANGULAR_SPEED,
    toward_mirror_degrees: float = _TOWARD_MIRROR_DEGREES,
    vio_heading_offset_degrees: float | None = None,
    vio_turn_max_commands: int = 8,
    vio_angular_speed: int = 500,
    vio_calibration_pulse_count: int = 2,
    vio_realign_threshold_degrees: float = 15.0,
    vio_max_realignments: int = 3,
    sample_delays: list[float] | tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60),
    motion_refresh_interval_ms: int = 0,
    ha_state: str | None = None,
    active_route: dict[str, Any] | None = None,
    refetch_runtime_context: (
        Callable[[], tuple[str | None, dict[str, Any] | None]] | None
    ) = None,
) -> dict[str, Any]:
    """Execute or dry-run a guarded chain of proven raw vector segments.

    In ``turn_mode="vio"`` the map->VIO heading offset derived by segment 1's
    calibration drive is carried forward, so later segments skip their own
    calibration drives and turn immediately on the shared offset.
    """
    normalized_area_hash = _coerce_optional_int(area_hash)
    # Route B: split long legs BEFORE the preview, so `_validate_custom_path`
    # judges both the inserted points and every resulting sub-leg. An inserted
    # point can land outside a concave area even when both operator clicks are
    # inside it; segment containment is invariant under this collinear split.
    split = _split_long_legs(points, target_length_m=split_leg_target_length_m)
    # 🔑 Surface the leg length beyond which the mid-drive controller cannot
    # protect the landing. Advisory, not a refusal: a 3.0 m sub-leg DID reach
    # target at 0.094 m on 2026-08-20, while its sibling missed by 0.2594 m when
    # a 51 deg correction came due at 0.26 m to run and was refused
    # `turn_budget_infeasible`. The bound explains that spread; it does not
    # forbid the run. See `_correctable_leg_length_limit_m`.
    correctable_limit = _correctable_leg_length_limit_m(
        waypoint_tolerance=waypoint_tolerance
    )
    split["correctable_leg_length_limit_m"] = correctable_limit
    longest_sub_leg = max(
        (float(leg.get("sub_leg_length_m") or 0.0) for leg in split.get("legs") or []),
        default=0.0,
    )
    if not longest_sub_leg:
        # `legs` only lists legs that were actually split; a short path has none.
        pts = split["points"]
        longest_sub_leg = max(
            (
                math.dist(
                    (float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"]))
                )
                for a, b in zip(pts, pts[1:], strict=False)
            ),
            default=0.0,
        )
    split["longest_leg_length_m"] = longest_sub_leg
    split["exceeds_correctable_limit"] = longest_sub_leg > correctable_limit
    preview = _preview_custom_path(
        coordinator,
        split["points"],
        area_hash=normalized_area_hash,
        speed=0.2,
        blade_mode="off",
    )
    normalized_points = preview["points"]
    # Auto-promote BLE before the telemetry snapshot and gates for a real run (see
    # the vector-segment executor); a successful recovery lets ble_transport_required
    # pass without an operator toggle/reboot, and the snapshot below must reflect
    # post-recovery state since the wait can span ~90s.
    ble_recovery: dict[str, Any] | None = None
    if (
        not dry_run
        and prefer_ble
        and ble_auto_recover
        and not _transport_is_ble(coordinator)
    ):
        ble_recovery = await _attempt_ble_recovery(coordinator)
        if refetch_runtime_context is not None:
            # Recovery can take ~90s; re-capture the handler-side HA state and
            # active-route snapshot so the runtime gates judge fresh context.
            ha_state, active_route = refetch_runtime_context()
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
    initial_vio_feed = (
        {
            "live": None,
            "tracked_features": None,
            "brightness_raw": None,
            "brightness_label": None,
        }
        if turn_mode == "night"
        else _vio_feed_liveness(coordinator)
    )
    if (
        turn_mode == "vio"
        and _vio_reading(coordinator)["vio_state"] == _VIO_STATE_ACTIVE
        and not initial_vio_feed["live"]
    ):
        # Same dusk-latch guard as the per-segment vector executor, but at the
        # chain entry so a blind feed (vio_state active yet 0 tracked features) is
        # refused once up front rather than only when segment 1 happens to run.
        gates.append(_vio_feed_live_gate(initial_vio_feed, dry_run=dry_run))
    split_sub_leg_count = max(0, len(normalized_points) - 1)
    split_over_budget = (
        split["applied"] and split_sub_leg_count > REAL_CLICK_TO_GO_SEGMENT_LIMIT
    )
    if split_over_budget:
        # Fires on dry runs too. A dry run that passes while Real Go refuses is
        # the trap this gate exists to close -- the operator would plan against
        # a preview the real path will not accept.
        gates.append(
            {
                "name": "split_exceeds_real_segment_budget",
                "passed": False,
                "detail": (
                    f"{split['requested_leg_count']} destination(s) split into "
                    f"{split_sub_leg_count} sub-legs of at most "
                    f"{split['target_length_m']} m; at most "
                    f"{REAL_CLICK_TO_GO_SEGMENT_LIMIT} segments can run per click. "
                    "Click a nearer point, or fewer of them."
                ),
                "diagnostics": {
                    "requested_leg_count": split["requested_leg_count"],
                    "sub_leg_count": split_sub_leg_count,
                    "target_length_m": split["target_length_m"],
                    "max_segments": REAL_CLICK_TO_GO_SEGMENT_LIMIT,
                    "legs": split["legs"],
                },
            }
        )
    if len(normalized_points) < 2 or len(normalized_points) > 8:
        gates.append(
            {
                "name": "point_count_2_to_8",
                "passed": False,
                "detail": "Multi-segment execution accepts 2 to 8 points.",
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
    if not dry_run and max_real_segments > REAL_CLICK_TO_GO_SEGMENT_LIMIT:
        gates.append(
            {
                "name": "real_segment_limit",
                "passed": False,
                "detail": (
                    "Real click-to-go execution is limited to "
                    f"{REAL_CLICK_TO_GO_SEGMENT_LIMIT} segments."
                ),
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
    # Apply the operator's deliberate overrides BEFORE blockers are computed --
    # one choke point, so no gate can be overridden anywhere else. An overridden
    # gate keeps `original_passed: False` and `overridden: True`, so the run JSON
    # can never present an overridden run as a clean one.
    safety_overrides_summary = _apply_safety_overrides(gates, safety_overrides)
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
        "initial_vio_feed": initial_vio_feed,
        "prefer_ble": prefer_ble,
        "transport_preference": "ble_preferred" if prefer_ble else "default",
        "ble_auto_recover": ble_auto_recover,
        "ble_recovery": ble_recovery,
        "max_real_segments": max_real_segments,
        # Echo it or it is unprovable (the beta44 discipline): the run JSON is
        # the artifact every measurement is read out of, so it has to record
        # what the operator actually clicked as well as what was driven.
        "split_leg_target_length_m": split_leg_target_length_m,
        "split": split,
        "requested_points": split["requested_points"],
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
        "turn_mode": turn_mode,
        "vio_heading_offset_degrees": vio_heading_offset_degrees,
        # Echoed for post-run forensics. The handler has always forwarded these to
        # each segment, but the multi-segment result never reported them, so a
        # failed run gave no way to confirm what pulse lengths the mower was
        # actually told to use -- the first thing you want to know when a linear
        # phase stalls (live 2026-07-19).
        "turn_pulse_duration_ms": turn_pulse_duration_ms,
        "linear_pulse_duration_ms": linear_pulse_duration_ms,
        "final_approach_metres_per_pulse": final_approach_metres_per_pulse,
        "turn_degrees_per_second": turn_degrees_per_second,
        "vio_turn_max_commands": vio_turn_max_commands,
        "vio_angular_speed": vio_angular_speed,
        "max_linear_pulse_ceiling": max_linear_pulse_ceiling,
        # beta44: these two were the ONLY accepted-profile keys the multi-segment
        # response did not echo, and the Gate 5 pass of 2026-08-12 had to be
        # argued around the hole -- `motion_refresh_interval_ms` was provable
        # from the per-segment echo and the delivered writes, but
        # `max_no_progress_pulses` was unprovable from the record and had to be
        # dismissed as un-exercised instead. Gate 5's whole purpose is showing
        # the card sent the accepted profile, so every profile key it sends has
        # to come back.
        "max_no_progress_pulses": max_no_progress_pulses,
        "motion_refresh_interval_ms": motion_refresh_interval_ms,
        "sample_delays": list(sample_delays),
        "confirm_blades_off": confirm_blades_off,
        "confirm_clear_area": confirm_clear_area,
        # Echo it or it is unprovable. An override changes what the mower was
        # allowed to do, so the run record has to carry which gates were lifted
        # and why -- `applied` includes each gate's rationale verbatim.
        "safety_overrides": safety_overrides_summary,
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
    if split_over_budget:
        # Above `invalid_point_count` so a handful of long clicks reports the
        # real reason rather than surfacing as a bare point-count error.
        result["stop_reason"] = "split_exceeds_real_segment_budget"
        return result
    if total_segments < 1 or len(normalized_points) > 8:
        result["stop_reason"] = "invalid_point_count"
        return result
    if blockers and not dry_run:
        result["stop_reason"] = "safety_gates_failed"
        return result

    if turn_mode == "vio":
        # Zero-motion geometric preflight. The turn a segment must make at each
        # junction (segments 2..N) is pure path geometry, known before any
        # command is sent -- only segment 1's turn depends on the unknown
        # stationary orientation (beta19: no feed reports it) and is judged
        # post-calibration inside the turn primitive instead. A real run refuses
        # a path containing a junction the turn budget provably cannot finish
        # (Gate 4 retry, 2026-08-03); a dry run reports the same math without
        # refusing, matching the advisory dry-run gate pattern.
        junction_turn_feasibility: list[dict[str, Any]] = []
        for junction_index in range(1, total_segments):
            inbound_heading = _path_heading_degrees(
                normalized_points[junction_index - 1],
                normalized_points[junction_index],
            )
            outbound_heading = _path_heading_degrees(
                normalized_points[junction_index],
                normalized_points[junction_index + 1],
            )
            junction_turn_degrees = _heading_error_degrees(
                inbound_heading, outbound_heading
            )
            junction_turn_feasibility.append(
                {
                    "segment_index": junction_index + 1,
                    "turn_degrees": round(junction_turn_degrees, 3),
                    "feasibility": _vio_turn_budget_feasibility(
                        initial_error_degrees=junction_turn_degrees,
                        heading_tolerance_degrees=heading_tolerance_degrees,
                        max_commands=vio_turn_max_commands,
                        pulse_duration_ms=float(turn_pulse_duration_ms),
                        motion_refresh_interval_ms=motion_refresh_interval_ms,
                        max_displacement_m=max_turn_translation_distance,
                        # The segment executor lets the turn primitive default
                        # its slow-pulse pair, so the model must too; only the
                        # configured rate is forwarded.
                        turn_degrees_per_second=turn_degrees_per_second,
                    ),
                }
            )
        result["junction_turn_feasibility"] = junction_turn_feasibility
        if not dry_run and any(
            not junction["feasibility"]["feasible"]
            for junction in junction_turn_feasibility
        ):
            result["stop_reason"] = "path_turn_infeasible"
            return result
    elif turn_mode == "night" and total_segments > 1:
        # Night has no junction feasibility model. Refuse before segment one so
        # an impossible junction cannot be discovered after motion has begun.
        result["stop_reason"] = "night_multi_segment_unsupported"
        return result

    carried_vio_offset = vio_heading_offset_degrees
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
            ble_auto_recover=ble_auto_recover,
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
            final_approach_metres_per_pulse=final_approach_metres_per_pulse,
            turn_degrees_per_second=turn_degrees_per_second,
            motion_refresh_interval_ms=motion_refresh_interval_ms,
            turn_mode=turn_mode,
            night_angular_speed=night_angular_speed,
            toward_mirror_degrees=toward_mirror_degrees,
            vio_heading_offset_degrees=carried_vio_offset,
            vio_turn_max_commands=vio_turn_max_commands,
            vio_angular_speed=vio_angular_speed,
            vio_calibration_pulse_count=vio_calibration_pulse_count,
            vio_realign_threshold_degrees=vio_realign_threshold_degrees,
            vio_max_realignments=vio_max_realignments,
            sample_delays=tuple(sample_delays),
            ha_state=ha_state,
            active_route=active_route,
            # A segment can trigger its own BLE recovery mid-chain; let it
            # refresh the handler-side context too.
            refetch_runtime_context=refetch_runtime_context,
        )
        if turn_mode == "vio":
            segment_offset_degrees = (segment_result.get("vio") or {}).get(
                "offset_degrees"
            )
            if segment_offset_degrees is not None:
                carried_vio_offset = segment_offset_degrees
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
        # start_report_stream degrades to a single snapshot outside ACTIVE mode,
        # which leaves a manually driven mower reporting one frozen position for
        # the whole run. Ask for the continuous subscription explicitly.
        if hasattr(coordinator, "async_start_continuous_reports"):
            await coordinator.async_start_continuous_reports(
                duration_ms=max(10_000, stream_duration_ms)
            )
        # The calls above enqueue BLE commands; let them clear before the
        # ble_link_live gate below demands an empty queue.
        await _settle_ble_command_queue(coordinator)

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
        await _motion_open_sleep(coordinator, pulse_duration_ms / 1000)
        if stop_mode == "delayed" and stop_delay_ms > 0:
            await _motion_open_sleep(coordinator, stop_delay_ms / 1000)
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
        result["telemetry_latency_seconds"] = late_progress["telemetry_latency_seconds"]
        result["cumulative_sample_diagnostics"] = late_progress[
            "post_stop_sample_diagnostics"
        ]
        result["cumulative_progress_detected"] = late_progress["late_progress_detected"]
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
        # start_report_stream degrades to a single snapshot outside ACTIVE mode,
        # which leaves a manually driven mower reporting one frozen position for
        # the whole run. Ask for the continuous subscription explicitly.
        if hasattr(coordinator, "async_start_continuous_reports"):
            await coordinator.async_start_continuous_reports(
                duration_ms=max(10_000, stream_duration_ms)
            )
        # The calls above enqueue BLE commands; let them clear before the
        # ble_link_live gate below demands an empty queue.
        await _settle_ble_command_queue(coordinator)

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
        and abs(calibrated_heading_error) > calibrated_forward_heading_tolerance_degrees
    ):
        result["stop_reason"] = "segment_heading_outside_calibrated_forward_window"
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
        progress_diagnostic = (
            burst_result.get("cumulative_path_progress_diagnostic") or {}
        )
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
    before = (
        samples[0]["telemetry"]
        if samples
        else _custom_path_telemetry_snapshot(coordinator)
    )
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
        # start_report_stream degrades to a single snapshot outside ACTIVE mode,
        # which leaves a manually driven mower reporting one frozen position for
        # the whole run. Ask for the continuous subscription explicitly.
        if hasattr(coordinator, "async_start_continuous_reports"):
            await coordinator.async_start_continuous_reports(
                duration_ms=max(10_000, stream_duration_ms)
            )
        # The calls above enqueue BLE commands; let them clear before the
        # ble_link_live gate below demands an empty queue.
        await _settle_ble_command_queue(coordinator)

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
        await _motion_open_sleep(coordinator, pulse_duration_ms / 1000)
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
        path_progress_distance = path_progress_diagnostic.get("path_progress_distance")
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
                "telemetry_latency_seconds": late_progress["telemetry_latency_seconds"],
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


def _make_refetch_runtime_context(
    hass: HomeAssistant,
    entity_id: str,
    coordinator: MammotionReportUpdateCoordinator,
) -> Callable[[], tuple[str | None, dict[str, Any] | None]]:
    """Build the post-recovery HA-state + active-route re-capture callback.

    The vector/multi-segment executors call this after an in-executor BLE recovery
    wait (~90s) so the runtime gates judge fresh context instead of the handler's
    pre-recovery snapshot.
    """

    def _refetch() -> tuple[str | None, dict[str, Any] | None]:
        state = hass.states.get(entity_id)
        route: dict[str, Any] | None = None
        try:
            route = _export_active_route(coordinator)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("Could not refetch active route: %s", err)
        return (state.state if state is not None else None, route)

    return _refetch


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

    async def handle_movement(call: ServiceCall, direction: str) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="mower_target_not_found",
            )
        coordinator = mower.reporting_coordinator
        command, command_kwargs = {
            "async_move_forward": (
                "move_forward",
                {"linear": call.data["speed"]},
            ),
            "async_move_left": (
                "move_left",
                {"angular": call.data["speed"]},
            ),
            "async_move_right": (
                "move_right",
                {"angular": call.data["speed"]},
            ),
            "async_move_back": (
                "move_back",
                {"linear": call.data["speed"]},
            ),
        }[direction]
        await _send_ble_motion_command_confirmed(
            coordinator,
            command,
            command_kwargs=command_kwargs,
        )
        await _stop_manual_motion_confirmed(coordinator)
        session = active_motion_session(coordinator)
        return {
            "service": command,
            "ok": True,
            "stop_confirmed": True,
            "session": session.as_dict() if session is not None else None,
        }

    hass.services.async_register(
        DOMAIN, SERVICE_REFRESH_STREAM, handle_refresh_stream, schema=CAMERA_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_START_VIDEO, handle_start_video, schema=CAMERA_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_STOP_VIDEO, handle_stop_video, schema=CAMERA_SCHEMA
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
        ) -> dict[str, Any]:
            return await handle_movement(call, method_name)

        hass.services.async_register(
            DOMAIN,
            service_name,
            _wrap_exclusive_manual_motion(
                hass,
                service_name,
                handle_directional_movement,
                always_real=True,
            ),
            schema=MOVEMENT_SCHEMA,
            supports_response=SupportsResponse.OPTIONAL,
        )

    async def handle_stop_manual_motion(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="mower_target_not_found",
            )
        return await _stop_active_manual_motion(mower.reporting_coordinator)

    hass.services.async_register(
        DOMAIN,
        SERVICE_STOP_MANUAL_MOTION,
        handle_stop_manual_motion,
        schema=CAMERA_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    async def handle_force_map_resync(call: ServiceCall) -> dict[str, Any]:
        """Force a full map re-fetch + GeoJSON re-projection (recovery lever).

        For the "map stuck out_of_sync after a reload/restart" state where
        click-to-path containment fails ``area_hash_not_found`` and the GeoJSON
        has no Polygon features.  A config-entry reload does not fix it; this
        does. Returns the coordinator's step-by-step recovery result.
        """
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await mower.reporting_coordinator.async_force_map_resync()

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
        SERVICE_FORCE_MAP_RESYNC,
        handle_force_map_resync,
        schema=GEOJSON_SCHEMA,
        supports_response=SupportsResponse.ONLY,
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
                    # Why map_sync_status reads what it reads, without having
                    # to fire force_map_resync to find out.
                    "map_sync": mower.reporting_coordinator.map_sync_diagnostics(),
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
            motion_refresh_interval_ms=call.data["motion_refresh_interval_ms"],
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
            motion_refresh_interval_ms=call.data["motion_refresh_interval_ms"],
            in_window_sample_interval_ms=call.data["in_window_sample_interval_ms"],
            duration_ms=call.data["duration_ms"],
            max_travel_m=call.data["max_travel_m"],
            sample_delays=tuple(call.data["sample_delays"]),
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
        )

    async def handle_continuous_motion_window(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _continuous_motion_window(
            mower.reporting_coordinator,
            route_start=dict(call.data["route_start"]),
            route_target=dict(call.data["route_target"]),
            corridor_polygon=[dict(point) for point in call.data["corridor_polygon"]],
            linear_speed=call.data["linear_speed"],
            max_abs_angular_speed=call.data["max_abs_angular_speed"],
            duration_ms=call.data["duration_ms"],
            motion_refresh_interval_ms=call.data["motion_refresh_interval_ms"],
            decision_sample_interval_ms=call.data["decision_sample_interval_ms"],
            max_distance_m=call.data["max_distance_m"],
            max_cross_track_m=call.data["max_cross_track_m"],
            prefer_ble=call.data["prefer_ble"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            confirm_steering_validation_run=call.data[
                "confirm_steering_validation_run"
            ],
        )

    async def handle_step_response_probe(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _step_response_probe(
            mower.reporting_coordinator,
            route_start=dict(call.data["route_start"]),
            corridor_polygon=[dict(point) for point in call.data["corridor_polygon"]],
            linear_speed=call.data["linear_speed"],
            step_angular_speed=call.data["step_angular_speed"],
            baseline_ms=call.data["baseline_ms"],
            step_ms=call.data["step_ms"],
            settle_ms=call.data["settle_ms"],
            motion_refresh_interval_ms=call.data["motion_refresh_interval_ms"],
            sample_interval_ms=call.data["sample_interval_ms"],
            max_travel_m=call.data["max_travel_m"],
            prefer_ble=call.data["prefer_ble"],
            dry_run=call.data["dry_run"],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            confirm_step_response_run=call.data["confirm_step_response_run"],
        )

    async def handle_heading_acquisition_window(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _heading_acquisition_window(
            mower.reporting_coordinator,
            route_start=dict(call.data["route_start"]),
            route_target=dict(call.data["route_target"]),
            corridor_polygon=[dict(point) for point in call.data["corridor_polygon"]],
            prefer_ble=call.data["prefer_ble"],
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
            linear_pulse_duration_ms=call.data["linear_pulse_duration_ms"],
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
            LOGGER.debug(
                "Could not export active route for angular calibration: %s", err
            )
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
            motion_refresh_interval_ms=call.data["motion_refresh_interval_ms"],
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

        _refetch_runtime_context = _make_refetch_runtime_context(
            hass, call.data[ATTR_ENTITY_ID], mower.reporting_coordinator
        )

        return await _raw_pymammotion_execute_vector_segment(
            mower.reporting_coordinator,
            cast(list[dict[str, float]], call.data["points"]),
            area_hash=call.data.get("area_hash"),
            dry_run=call.data["dry_run"],
            safety_overrides=call.data.get("safety_overrides") or [],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            prefer_ble=call.data["prefer_ble"],
            ble_auto_recover=call.data["ble_auto_recover"],
            linear_speed_fast=call.data["linear_speed_fast"],
            linear_speed_slow=call.data["linear_speed_slow"],
            slow_linear_threshold=call.data["slow_linear_threshold"],
            max_turn_commands=call.data["max_turn_commands"],
            max_linear_commands=call.data["max_linear_commands"],
            max_linear_pulse_ceiling=call.data.get("max_linear_pulse_ceiling"),
            final_approach_metres_per_pulse=call.data[
                "final_approach_metres_per_pulse"
            ],
            turn_degrees_per_second=call.data["turn_degrees_per_second"],
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
            motion_refresh_interval_ms=call.data["motion_refresh_interval_ms"],
            turn_mode=call.data["turn_mode"],
            night_angular_speed=call.data["night_angular_speed"],
            toward_mirror_degrees=call.data["toward_mirror_degrees"],
            vio_heading_offset_degrees=call.data.get("vio_heading_offset_degrees"),
            vio_turn_max_commands=call.data["vio_turn_max_commands"],
            vio_angular_speed=call.data["vio_angular_speed"],
            vio_calibration_pulse_count=call.data["vio_calibration_pulse_count"],
            vio_realign_threshold_degrees=call.data["vio_realign_threshold_degrees"],
            vio_max_realignments=call.data["vio_max_realignments"],
            sample_delays=tuple(call.data["sample_delays"]),
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
            refetch_runtime_context=_refetch_runtime_context,
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

        _refetch_runtime_context = _make_refetch_runtime_context(
            hass, call.data[ATTR_ENTITY_ID], mower.reporting_coordinator
        )

        return await _raw_pymammotion_execute_multi_segment(
            mower.reporting_coordinator,
            cast(list[dict[str, float]], call.data["points"]),
            area_hash=call.data.get("area_hash"),
            dry_run=call.data["dry_run"],
            safety_overrides=call.data.get("safety_overrides") or [],
            confirm_blades_off=call.data["confirm_blades_off"],
            confirm_clear_area=call.data["confirm_clear_area"],
            prefer_ble=call.data["prefer_ble"],
            ble_auto_recover=call.data["ble_auto_recover"],
            max_real_segments=call.data["max_real_segments"],
            split_leg_target_length_m=call.data.get("split_leg_target_length_m"),
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
            final_approach_metres_per_pulse=call.data[
                "final_approach_metres_per_pulse"
            ],
            turn_degrees_per_second=call.data["turn_degrees_per_second"],
            motion_refresh_interval_ms=call.data["motion_refresh_interval_ms"],
            turn_mode=call.data["turn_mode"],
            night_angular_speed=call.data["night_angular_speed"],
            toward_mirror_degrees=call.data["toward_mirror_degrees"],
            vio_heading_offset_degrees=call.data.get("vio_heading_offset_degrees"),
            vio_turn_max_commands=call.data["vio_turn_max_commands"],
            vio_angular_speed=call.data["vio_angular_speed"],
            vio_calibration_pulse_count=call.data["vio_calibration_pulse_count"],
            vio_realign_threshold_degrees=call.data["vio_realign_threshold_degrees"],
            vio_max_realignments=call.data["vio_max_realignments"],
            sample_delays=tuple(call.data["sample_delays"]),
            ha_state=ha_state.state if ha_state is not None else None,
            active_route=active_route,
            refetch_runtime_context=_refetch_runtime_context,
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

    async def handle_basestation_info_probe(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _basestation_info_probe(
            mower.reporting_coordinator,
            wait_seconds=call.data["wait_seconds"],
            rtk_sources=_rtk_base_station_sources(hass),
        )

    async def handle_ota_info_probe(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _ota_info_probe(
            mower.reporting_coordinator,
            send_timeout=call.data["send_timeout"],
        )

    async def handle_report_stream_probe(call: ServiceCall) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _report_stream_probe(
            mower.reporting_coordinator,
            period_ms=call.data["period_ms"],
            no_change_period_ms=call.data["no_change_period_ms"],
            duration_seconds=call.data["duration_seconds"],
            isolated=call.data["isolated"],
        )

    async def handle_report_stream_sequence_probe(
        call: ServiceCall,
    ) -> dict[str, Any]:
        mower = _get_mower_by_entity_id(hass, call.data[ATTR_ENTITY_ID])
        if mower is None:
            LOGGER.error("Could not find entity %s", call.data[ATTR_ENTITY_ID])
            return {}
        return await _report_stream_sequence_probe(
            mower.reporting_coordinator,
            periods_ms=cast(list[int], call.data["periods_ms"]),
            observation_seconds=call.data["observation_seconds"],
            readiness_timeout_seconds=call.data["readiness_timeout_seconds"],
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
            motion_refresh_interval_ms=call.data["motion_refresh_interval_ms"],
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
            fresh_heading_timeout_seconds=call.data["fresh_heading_timeout_seconds"],
            max_commands=call.data["max_commands"],
            min_progress_degrees=call.data["min_progress_degrees"],
            max_no_progress_pulses=call.data["max_no_progress_pulses"],
            max_displacement_m=call.data["max_displacement_m"],
            invert_direction=call.data["invert_direction"],
            motion_refresh_interval_ms=call.data["motion_refresh_interval_ms"],
            turn_degrees_per_second=call.data["turn_degrees_per_second"],
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

    async def handle_disarm_experimental_motion(call: ServiceCall) -> dict[str, Any]:
        """Close the experimental-motion gate. Never opens it.

        Idempotent: disarming an already-disarmed gate is a success, not an
        error, so an automation can call it on a schedule without generating
        failures.

        ⚠️ It does NOT stop a run in progress -- that is `Abort / Stop`, a
        different and more urgent thing. This only removes the standing
        permission, so it refuses while a session is active rather than
        half-stopping a moving mower.
        """
        entity_id = call.data[ATTR_ENTITY_ID]
        mower = _get_mower_by_entity_id(hass, entity_id)
        if mower is None:
            LOGGER.error("Could not find entity %s", entity_id)
            return {"disarmed": False, "reason": "entity_not_found"}
        coordinator = mower.reporting_coordinator
        entry = getattr(coordinator, "config_entry", None)
        if entry is None:
            return {"disarmed": False, "reason": "config_entry_unavailable"}

        # ⚠️ Deliberately NOT experimental_motion_status(): that builds the full
        # blocker report and requires `ble_liveness` and `safety` keyword
        # arguments this handler has no reason to compute. Calling it without
        # them is a TypeError, which is exactly how the first version of this
        # service failed on the host at 500 while every unit test passed --
        # they exercised the constants around the handler, never the handler.
        #
        # Disarming needs two facts and nothing else: is it on, and is a run in
        # progress. Read both directly.
        was_enabled = experimental_motion_enabled(coordinator)
        if active_motion_session(coordinator) is not None:
            # Pulling the permission out from under a live run would leave the
            # session's own stop path to finish without it. Abort is the
            # correct tool and it is a separate service.
            LOGGER.warning(
                "Refusing to disarm %s while a manual-motion session is active; "
                "use the abort service to stop a run",
                entity_id,
            )
            return {
                "disarmed": False,
                "reason": "active_session",
                "was_enabled": was_enabled,
            }

        if was_enabled:
            options = dict(entry.options)
            options[CONF_ENABLE_EXPERIMENTAL_MOTION] = False
            hass.config_entries.async_update_entry(entry, options=options)
            LOGGER.info("Experimental motion gate disarmed for %s", entity_id)

        return {
            "disarmed": True,
            "was_enabled": was_enabled,
            "changed": was_enabled,
            "enabled": experimental_motion_enabled(coordinator),
        }

    hass.services.async_register(
        DOMAIN,
        SERVICE_EXPORT_RUNTIME_STATE,
        handle_export_runtime_state,
        schema=GEOJSON_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_DISARM_EXPERIMENTAL_MOTION,
        handle_disarm_experimental_motion,
        schema=GEOJSON_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
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
        SERVICE_MANUAL_VELOCITY_PULSE_TEST,
        _wrap_exclusive_manual_motion(
            hass, SERVICE_MANUAL_VELOCITY_PULSE_TEST, handle_manual_velocity_pulse_test
        ),
        schema=MANUAL_VELOCITY_PULSE_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MANUAL_VELOCITY_SEGMENT_TEST,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_MANUAL_VELOCITY_SEGMENT_TEST,
            handle_manual_velocity_segment_test,
        ),
        schema=MANUAL_VELOCITY_SEGMENT_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MANUAL_VELOCITY_MULTI_PULSE_TEST,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_MANUAL_VELOCITY_MULTI_PULSE_TEST,
            handle_manual_velocity_multi_pulse_test,
        ),
        schema=MANUAL_VELOCITY_MULTI_PULSE_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST,
            handle_manual_velocity_cumulative_pulse_test,
        ),
        schema=MANUAL_VELOCITY_CUMULATIVE_PULSE_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT,
            handle_experimental_execute_segment,
        ),
        schema=EXPERIMENTAL_EXECUTE_SEGMENT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT_BURST,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_EXPERIMENTAL_EXECUTE_SEGMENT_BURST,
            handle_experimental_execute_segment_burst,
        ),
        schema=EXPERIMENTAL_EXECUTE_SEGMENT_BURST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MANUAL_VELOCITY_HEADING_CALIBRATION_TEST,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_MANUAL_VELOCITY_HEADING_CALIBRATION_TEST,
            handle_manual_velocity_heading_calibration_test,
        ),
        schema=MANUAL_VELOCITY_HEADING_CALIBRATION_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_PYMAMMOTION_MOTION_PROBE,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_RAW_PYMAMMOTION_MOTION_PROBE,
            handle_raw_pymammotion_motion_probe,
            # The card's Abort is this service with linear/angular 0 -- a stop,
            # not motion; it must preempt a running loop, never be rejected.
            allow_stop_nudge=True,
        ),
        schema=RAW_PYMAMMOTION_MOTION_PROBE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_CONTINUOUS_MOTION_WINDOW,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_CONTINUOUS_MOTION_WINDOW,
            handle_continuous_motion_window,
        ),
        schema=CONTINUOUS_MOTION_WINDOW_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_STEP_RESPONSE_PROBE,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_STEP_RESPONSE_PROBE,
            handle_step_response_probe,
        ),
        schema=STEP_RESPONSE_PROBE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_HEADING_ACQUISITION_WINDOW,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_HEADING_ACQUISITION_WINDOW,
            handle_heading_acquisition_window,
        ),
        schema=HEADING_ACQUISITION_WINDOW_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_PYMAMMOTION_EXECUTE_SEGMENT,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_RAW_PYMAMMOTION_EXECUTE_SEGMENT,
            handle_raw_pymammotion_execute_segment,
        ),
        schema=RAW_PYMAMMOTION_EXECUTE_SEGMENT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_PYMAMMOTION_ANGULAR_CALIBRATION,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_RAW_PYMAMMOTION_ANGULAR_CALIBRATION,
            handle_raw_pymammotion_angular_calibration,
        ),
        schema=RAW_PYMAMMOTION_ANGULAR_CALIBRATION_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_PYMAMMOTION_TURN_TO_HEADING,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_RAW_PYMAMMOTION_TURN_TO_HEADING,
            handle_raw_pymammotion_turn_to_heading,
        ),
        schema=RAW_PYMAMMOTION_TURN_TO_HEADING_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT,
            handle_raw_pymammotion_execute_vector_segment,
        ),
        schema=RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT,
            handle_raw_pymammotion_execute_multi_segment,
        ),
        schema=RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_FORWARD_TWO_PULSE_LATENCY_TEST,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_FORWARD_TWO_PULSE_LATENCY_TEST,
            handle_forward_two_pulse_latency_test,
        ),
        schema=FORWARD_TWO_PULSE_LATENCY_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_POSITION_FEEDBACK_DIAGNOSTIC,
        _wrap_exclusive_manual_motion(
            hass,
            SERVICE_POSITION_FEEDBACK_DIAGNOSTIC,
            handle_position_feedback_diagnostic,
        ),
        schema=POSITION_FEEDBACK_DIAGNOSTIC_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    # Deliberately NOT wrapped in _wrap_exclusive_manual_motion: that wrapper
    # claims the mower only for real *motion* runs, keyed on `dry_run: false`,
    # and this service has no dry_run field and commands no motion. It guards
    # itself instead by refusing while `manual_motion_owner` is set, so it can
    # never reconfigure the report stream underneath a live run.
    hass.services.async_register(
        DOMAIN,
        SERVICE_REPORT_STREAM_PROBE,
        handle_report_stream_probe,
        schema=REPORT_STREAM_PROBE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_REPORT_STREAM_SEQUENCE_PROBE,
        handle_report_stream_sequence_probe,
        schema=REPORT_STREAM_SEQUENCE_PROBE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    # Not wrapped in _wrap_exclusive_manual_motion for the same reason as the
    # report probe: that wrapper claims the mower only for real motion runs, and
    # this query commands none.
    hass.services.async_register(
        DOMAIN,
        SERVICE_BASESTATION_INFO_PROBE,
        handle_basestation_info_probe,
        schema=BASESTATION_INFO_PROBE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    # Not wrapped in _wrap_exclusive_manual_motion for the same reason as the
    # basestation probe: this sends a read-only get-info REQUEST only, never
    # fw_download_ctrl or device/upgrade, so it commands no motion and cannot
    # trigger an install.
    hass.services.async_register(
        DOMAIN,
        SERVICE_OTA_INFO_PROBE,
        handle_ota_info_probe,
        schema=OTA_INFO_PROBE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_VIO_MOTION_PROBE,
        _wrap_exclusive_manual_motion(
            hass, SERVICE_VIO_MOTION_PROBE, handle_vio_motion_probe
        ),
        schema=VIO_MOTION_PROBE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_VIO_TURN_PROBE,
        _wrap_exclusive_manual_motion(
            hass, SERVICE_VIO_TURN_PROBE, handle_vio_turn_probe
        ),
        schema=VIO_TURN_PROBE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_VIO_TURN_TO_HEADING,
        _wrap_exclusive_manual_motion(
            hass, SERVICE_VIO_TURN_TO_HEADING, handle_vio_turn_to_heading
        ),
        schema=VIO_TURN_TO_HEADING_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_MOTION_READINESS_TEST,
        _wrap_exclusive_manual_motion(
            hass, SERVICE_RAW_MOTION_READINESS_TEST, handle_raw_motion_readiness_test
        ),
        schema=RAW_MOTION_READINESS_TEST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RAW_VECTOR_READINESS_TEST,
        _wrap_exclusive_manual_motion(
            hass, SERVICE_RAW_VECTOR_READINESS_TEST, handle_raw_vector_readiness_test
        ),
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
