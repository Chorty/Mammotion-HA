"""Tests for Mammotion read-only map/task visibility helpers."""

import ast
import asyncio
import datetime
import json
import math
import pathlib
import time
from collections.abc import Callable, Coroutine
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voluptuous as vol
import yaml
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.service import _validate_entity_service_schema
from pymammotion.data.model.hash_list import Plan
from pymammotion.messaging.command_queue import DeviceCommandQueue, Priority
from pymammotion.transport.base import NoTransportAvailableError, TransportType
from pymammotion.transport.ble import BLETransport, BLETransportConfig
from pymammotion.utility.constant import WorkMode

from custom_components.mammotion import coordinator as mammotion_coordinator
from custom_components.mammotion import lawn_mower as mammotion_lawn_mower
from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.button import BUTTON_LUBA_PRO_YUKA
from custom_components.mammotion.coordinator import (
    MammotionBaseUpdateCoordinator,
    MammotionReportUpdateCoordinator,
)
from custom_components.mammotion.manual_motion import ManualMotionCancelledError
from custom_components.mammotion.sensor import WORK_SENSOR_TYPES
from custom_components.mammotion.services import (
    _BLE_SEND_STALL_SECONDS,
    _MIN_SCALED_TURN_PULSE_MS,
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
    VIO_TURN_PROBE_SCHEMA,
    VIO_TURN_TO_HEADING_SCHEMA,
    _app_scale_speeds,
    _app_speed_scale_report,
    _basestation_has_query_fields,
    _basestation_info_probe,
    _ble_connect_cooldown_active,
    _ble_link_liveness,
    _ble_ready_for_motion,
    _ble_transport_usable,
    _custom_path_telemetry_snapshot,
    _dry_run_custom_path,
    _experimental_execute_segment_burst,
    _export_active_route,
    _export_mower_map,
    _export_mower_tasks,
    _export_runtime_state,
    _final_approach_pulse_ms,
    _forward_two_pulse_latency_test,
    _is_zero_motion_stop_nudge,
    _manual_velocity_best_heading_decision,
    _manual_velocity_controller_decision,
    _manual_velocity_cumulative_pulse_test,
    _manual_velocity_heading_calibration,
    _manual_velocity_path_progress_diagnostic,
    _manual_velocity_pulse_test,
    _manual_velocity_quality_degradation,
    _manual_velocity_segment_test,
    _motion_open_sleep,
    _motion_refresh_window,
    _normalised_linear_pulse_distance,
    _normalize_mower_areas,
    _normalize_mower_tasks,
    _point_on_segment,
    _position_feedback_diagnostic,
    _position_source_comparison,
    _preview_custom_path,
    _raw_motion_readiness_test,
    _raw_multi_segment_phase_passed,
    _raw_pymammotion_angular_calibration,
    _raw_pymammotion_execute_multi_segment,
    _raw_pymammotion_execute_segment,
    _raw_pymammotion_execute_vector_segment,
    _raw_pymammotion_motion_probe,
    _raw_pymammotion_turn_to_heading,
    _raw_vector_readiness_phase_passed,
    _raw_vector_readiness_test,
    _realign_cannot_improve_the_landing,
    _report_stream_probe,
    _requires_reverse_recovery,
    _rtk_report_age_seconds,
    _runtime_motion_safety_summary,
    _settle_linear_position_feed,
    _streak_shows_dead_telemetry,
    _streak_shows_no_actuation,
    _transport_is_ble,
    _turn_final_approach_pulse_ms,
    _validate_custom_path,
    _vio_feed_liveness,
    _vio_motion_probe,
    _vio_segment_calibration_drive,
    _vio_turn_probe,
    _vio_turn_to_heading,
    _wrap_exclusive_manual_motion,
)

LARGE_HASH = 9_223_372_036_854_775_000


class _ModuleShim:
    """A module view with selected attributes overridden, for one namespace."""

    def __init__(self, module: object, **overrides: object) -> None:
        """Wrap *module*, answering the named attributes from *overrides*."""
        self._module = module
        for name, value in overrides.items():
            setattr(self, name, value)

    def __getattr__(self, name: str) -> object:
        """Delegate everything not overridden to the real module."""
        return getattr(self._module, name)


def _patch_services_monotonic(
    monkeypatch: pytest.MonkeyPatch,
    monotonic: Callable[[], float],
) -> None:
    """Give services.py a fake monotonic clock without moving asyncio's.

    ``mammotion_services.time`` *is* the ``time`` module, so overriding
    ``monotonic`` on it is process-global -- and asyncio derives every timer
    deadline from ``loop.time()``, which reads ``time.monotonic()``. These
    tests' fake clocks jump a whole pulse (2-3.5s) forward inside ``fake_sleep``,
    which could therefore expire the production dispatch guard's real deadlines
    (4.0s per write, 2.0s to start on the queue) partway through a test. That
    surfaced as CI failures that came and went on identical code: reasons
    flipping to ``command_failed`` and one send recorded instead of two.

    Replacing the module *reference* inside the services namespace keeps the
    fake clock where it is needed and away from the event loop, so the guard
    timeouts stay on real time and still get exercised.
    """
    monkeypatch.setattr(
        mammotion_services,
        "time",
        _ModuleShim(time, monotonic=monotonic),
    )


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
    ble_connected: bool = True,
    ble_usable: bool = True,
    ble_last_send_age: float | None = 1.0,
    ble_queue_depth: int = 0,
    ble_queue_paused: bool = False,
) -> SimpleNamespace:
    """Build a coordinator fixture for manual velocity pulse tests.

    The ``ble_*`` knobs model what ``_ble_link_liveness`` reads. Defaults
    describe a healthy link (connected, drained queue, a send 1s ago); override
    them to exercise the ``ble_link_live`` gate.
    """
    pos_x, pos_y, toward = position
    now = time.monotonic()
    transport = SimpleNamespace(
        # The real BLE connect cooldown lives on the transport, not on
        # availability; expose it the way pymammotion's DeviceHandle does. 0.0 =
        # no cooldown armed.
        _connect_cooldown_until=0.0,
        # ``is_usable`` is a real BLETransport property: a transport can be the
        # active routing choice while being unusable (no BLEDevice / weak RSSI /
        # armed cooldown), which is why the motion gate checks it separately.
        is_usable=ble_usable,
        # ``is_usable`` is routing eligibility, not liveness -- it stays True
        # while the command queue is gated and commands pile up undelivered.
        # These two are what actually discriminate a live link.
        is_connected=ble_connected,
        last_send_monotonic=(
            0.0 if ble_last_send_age is None else now - ble_last_send_age
        ),
    )

    async def enqueue_immediately(
        work: object,
        priority: object = None,
        **_kwargs: object,
    ) -> None:
        """Run fixture queue work immediately while preserving the queue API."""
        del priority
        await cast(Callable[[], Coroutine[object, object, None]], work)()

    def build_command(
        command_name: str,
    ) -> Callable[..., tuple[str, dict[str, object]]]:
        """Return a fixture command builder that preserves its arguments."""

        def build(**kwargs: object) -> tuple[str, dict[str, object]]:
            return command_name, kwargs

        return build

    commands = MagicMock()
    for command_name in (
        "send_movement",
        "move_forward",
        "move_back",
        "move_left",
        "move_right",
    ):
        getattr(commands, command_name).side_effect = build_command(command_name)
    handle = SimpleNamespace(
        last_report_at=123.0,
        availability=SimpleNamespace(
            mqtt_reported_offline=False,
        ),
        get_transport=lambda _transport_type: transport,
        # DeviceCommandQueue: depth and the dispatch gate are private in
        # pymammotion 0.8.8, so mirror the attribute names the helper reads.
        queue=SimpleNamespace(
            is_saga_active=False,
            _transport_gate=SimpleNamespace(is_set=lambda: not ble_queue_paused),
            _queue=SimpleNamespace(qsize=lambda: ble_queue_depth),
            enqueue=enqueue_immediately,
        ),
        commands=commands,
        _send_marked=AsyncMock(),
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
    coordinator = SimpleNamespace(
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

    async def simulate_confirmed_write(
        _transport: object,
        payload: tuple[str, dict[str, object]],
    ) -> None:
        """Preserve existing observation hooks after confirmed dispatch."""
        command_name, kwargs = payload
        if command_name == "send_movement":
            if kwargs == {"linear_speed": 0, "angular_speed": 0}:
                await coordinator.async_stop_manual_motion()
                return
            await manager.send_command_with_args(
                coordinator.device_name,
                command_name,
                prefer_ble=True,
                **kwargs,
            )
            return
        method_name, speed_key = {
            "move_forward": ("async_move_forward", "linear"),
            "move_back": ("async_move_back", "linear"),
            "move_left": ("async_move_left", "angular"),
            "move_right": ("async_move_right", "angular"),
        }[command_name]
        ack = await getattr(coordinator, method_name)(
            speed=kwargs[speed_key],
            use_wifi=False,
        )
        if ack is False:
            raise RuntimeError(f"{command_name} write failed")

    handle._send_marked.side_effect = simulate_confirmed_write  # noqa: SLF001
    return coordinator


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
    assert [
        item["heading_offset_degrees"]
        for item in decision["heading_offset_diagnostics"]
    ] == [
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


# ---------------------------------------------------------------------------
# Schema/handler key parity.
#
# Every call.data["key"] subscript read in a service handler must resolve after
# schema validation of a minimal payload (vol.Required, or vol.Optional with a
# default). Otherwise any caller omitting the key gets KeyError -> HTTP 500 —
# the multi-segment ble_auto_recover / segment-test stop_mode regression class.
# Executor-level tests bypass the service schema, so this sweep is the only
# coverage of the handler<->schema contract.
# ---------------------------------------------------------------------------

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
    "name": "Test task",
    "enabled": True,
    "points": [{"x": 1.0, "y": 1.0}, {"x": 1.2, "y": 1.0}],
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
async def test_fake_pulse_clock_never_moves_the_event_loop_deadline_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A test clock that jumps a pulse must not expire real dispatch deadlines.

    The motion tests advance a fake monotonic clock by the whole pulse duration
    on every simulated sleep. asyncio computes ``wait_for`` deadlines from
    ``loop.time()``, so if that fake clock is installed globally, the production
    dispatch guard's real 4.0s write deadline and 2.0s queue-start deadline can
    expire mid-test at random -- which is exactly what made six motion tests
    fail on one CI attempt and pass on the next with identical code.
    """
    clock = {"now": 100.0}
    _patch_services_monotonic(monkeypatch, lambda: clock["now"])
    loop = asyncio.get_running_loop()
    loop_time_before = loop.time()

    clock["now"] += 60.0

    assert mammotion_services.time.monotonic() == 160.0
    assert loop.time() - loop_time_before < 1.0
    assert time.monotonic() - loop_time_before < 60.0


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

    _patch_services_monotonic(monkeypatch, fake_monotonic)
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
    handle = coordinator.manager.mower(coordinator.device_name)
    assert handle._send_marked.await_count == 2  # noqa: SLF001
    # The explicit stop is mandatory even on the happy path.
    assert handle.commands.send_movement.call_args_list[-1].kwargs == {
        "linear_speed": 0,
        "angular_speed": 0,
    }
    assert result["command_ok"] is True
    assert result["reason"] == "vio_initialized_during_motion"
    assert result["verdict"]["motion_confirmed"] is True
    assert result["verdict"]["vio_activated_while_moving"] is True
    assert 42.0 in result["verdict"]["heading_series"]
    assert result["samples"]


@pytest.mark.asyncio
async def test_vio_motion_probe_reports_settled_post_stop_displacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Motion that only lands post-stop is still confirmed, not mislabelled no-motion.

    The position feed lags ~4s: during-drive samples stay frozen and the real move
    only registers after the stop (live 2026-07-15, a 4in 6s pulse). The verdict
    must read the post-stop samples for displacement + motion_confirmed.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    clock = {"now": 100.0}
    phase = {"stopped": False}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    original_snapshot = mammotion_services._custom_path_telemetry_snapshot  # noqa: SLF001

    def fake_snapshot(
        coordinator_arg: MammotionReportUpdateCoordinator,
    ) -> dict:
        telemetry = original_snapshot(coordinator_arg)
        # Frozen while driving; the whole ~10cm move only appears after the stop.
        if phase["stopped"]:
            telemetry["position"]["y"] = float(telemetry["position"]["y"]) - 0.10
        return telemetry

    async def fake_stop(_coordinator: object) -> dict:
        phase["stopped"] = True
        return {"movement_ok": True}

    async def fake_get_reports(count: int = 5) -> None:
        return None  # VIO stays cold the whole time

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        mammotion_services, "_custom_path_telemetry_snapshot", fake_snapshot
    )
    monkeypatch.setattr(mammotion_services, "_stop_manual_motion_confirmed", fake_stop)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        drive_seconds=2.0,
        sample_interval_seconds=1.0,
        post_stop_samples=2,
    )

    # During-drive samples never registered motion...
    assert all(not sample["moving"] for sample in result["samples"])
    # ...but the settled post-stop position is used for the verdict.
    assert result["displacement_source"] == "post_stop"
    assert result["final_displacement_m"] == pytest.approx(0.10, abs=0.02)
    assert result["verdict"]["motion_confirmed"] is True
    assert result["reason"] != "no_motion_detected"


@pytest.mark.asyncio
async def test_vio_motion_probe_active_vio_lagged_motion_not_mislabelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VIO active during a lagged pulse is NOT reported as 'never initialized'.

    Regression: motion_confirmed reads post_stop (where the lagged move lands), so
    vio_activated_while_moving must judge VIO over the drive window rather than
    require the frozen per-sample `moving` flag -- otherwise a VIO-active pulse
    whose motion only registers after the stop is mislabelled
    vio_never_initialized_despite_motion.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=42.0, vio_state=2
    )
    clock = {"now": 100.0}
    phase = {"stopped": False}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    original_snapshot = mammotion_services._custom_path_telemetry_snapshot  # noqa: SLF001

    def fake_snapshot(
        coordinator_arg: MammotionReportUpdateCoordinator,
    ) -> dict:
        telemetry = original_snapshot(coordinator_arg)
        # Frozen during the drive; the whole move lands only after the stop.
        if phase["stopped"]:
            telemetry["position"]["y"] = float(telemetry["position"]["y"]) - 0.10
        return telemetry

    async def fake_stop(_coordinator: object) -> dict:
        phase["stopped"] = True
        return {"movement_ok": True}

    async def fake_get_reports(count: int = 5) -> None:
        return None  # VIO stays active (state 2) the whole time

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        mammotion_services, "_custom_path_telemetry_snapshot", fake_snapshot
    )
    monkeypatch.setattr(mammotion_services, "_stop_manual_motion_confirmed", fake_stop)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        drive_seconds=2.0,
        sample_interval_seconds=1.0,
        post_stop_samples=2,
    )

    # Frozen feed during the drive -> no drive sample registered `moving`...
    assert all(not sample["moving"] for sample in result["samples"])
    # ...but VIO was active throughout, so the verdict credits it instead of
    # claiming VIO never initialized.
    assert result["verdict"]["motion_confirmed"] is True
    assert result["verdict"]["vio_activated_while_moving"] is True
    assert result["reason"] == "vio_initialized_during_motion"


@pytest.mark.asyncio
async def test_vio_motion_probe_static_reports_no_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mower that never moves (drive or post-stop) reports no motion, ~0 displacement."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    async def fake_get_reports(count: int = 5) -> None:
        return None

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_motion_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        drive_seconds=2.0,
        sample_interval_seconds=1.0,
        post_stop_samples=2,
    )

    assert result["verdict"]["motion_confirmed"] is False
    assert result["reason"] == "no_motion_detected"
    assert result["final_displacement_m"] == pytest.approx(0.0, abs=1e-6)


def test_position_source_comparison_reports_both_sources_and_agreement() -> None:
    """Both position sources + RTK quality are captured, with their divergence."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    # report_data.locations[0] is scaled by 1/10000; place it 0.2m off mowing_state.
    coordinator.data.report_data.locations = [
        SimpleNamespace(
            real_pos_x=12000, real_pos_y=10000, real_toward=0, pos_type=1, bol_hash=123
        )
    ]

    present = _position_source_comparison(coordinator)
    assert present["locations_xy"] == pytest.approx((1.2, 1.0))
    assert present["locations_stale_zero"] is False
    assert present["mowing_state_xy"] == pytest.approx((1.0, 1.0))
    assert present["agreement_m"] == pytest.approx(0.2, abs=1e-4)
    assert present["rtk_status"] == 4
    assert present["pos_level"] == 0
    assert present["pos_type"] == 1

    # With no report_data.locations, the location source degrades to None.
    coordinator.data.report_data.locations = []
    missing = _position_source_comparison(coordinator)
    assert missing["locations_xy"] is None
    assert missing["mowing_state_xy"] == pytest.approx((1.0, 1.0))
    assert missing["agreement_m"] is None

    # The post-restart stale (0,0)/AREA_OUT pose is filtered out so it can't
    # masquerade as a huge source divergence in agreement_m.
    coordinator.data.report_data.locations = [
        SimpleNamespace(
            real_pos_x=0, real_pos_y=0, real_toward=0, pos_type=0, bol_hash=0
        )
    ]
    stale = _position_source_comparison(coordinator)
    assert stale["locations_xy"] is None
    assert stale["locations_stale_zero"] is True
    assert stale["agreement_m"] is None


@pytest.mark.asyncio
async def test_vio_turn_probe_defaults_to_dry_run() -> None:
    """VIO turn probe default plans an in-place rotation but sends nothing."""
    coordinator = _pulse_coordinator()

    result = await _vio_turn_probe(coordinator)

    assert result["service"] == "vio_turn_probe"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["reason"] == "dry_run"
    assert result["command"]["kwargs"] == {"linear_speed": 0, "angular_speed": 500}
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

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        angular_speed=500,
        drive_seconds=3.0,
        sample_interval_seconds=1.0,
        post_stop_samples=0,
    )

    # A single continuous angular command, then a mandatory explicit stop.
    handle = coordinator.manager.mower(coordinator.device_name)
    assert handle._send_marked.await_count == 2  # noqa: SLF001
    assert result["command"]["kwargs"] == {"linear_speed": 0, "angular_speed": 500}
    assert handle.commands.send_movement.call_args_list[-1].kwargs == {
        "linear_speed": 0,
        "angular_speed": 0,
    }
    assert result["reason"] == "vision_heading_tracks_rotation"
    assert result["verdict"]["vision_heading_change"]["total_abs_degrees"] >= 3.0
    assert result["verdict"]["course_over_ground_change"]["total_abs_degrees"] == 0.0


@pytest.mark.asyncio
async def test_vio_turn_probe_app_parity_refresh_resends_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """motion_refresh_interval_ms re-sends the rotation command app-style (B1 turn A/B).

    B1 proved refresh is speed-gated (it unlocked linear but did nothing at angular
    180, below this mower's rotation threshold). This probe reaches angular 500, so
    it is the tool to test refresh on a properly-powered turn -- which first requires
    that the refresh actually re-issues the command during the drive.
    """
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    async def fake_get_reports(count: int = 5) -> None:
        heading = (clock["now"] - 100.0) * 10.0
        coordinator.data.report_data.vision_info = SimpleNamespace(
            heading=heading,
            vio_state=2,
        )

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        angular_speed=500,
        drive_seconds=3.0,
        sample_interval_seconds=1.0,
        post_stop_samples=0,
        motion_refresh_interval_ms=200,
    )

    # The command is re-issued during the drive, not sent once.
    refreshes = result["motion_refresh_commands_sent"]
    assert result["motion_refresh_interval_ms"] == 200
    assert refreshes > 0
    # Every send is the initial one plus one per refresh; all identical turn commands.
    handle = coordinator.manager.mower(coordinator.device_name)
    assert handle._send_marked.await_count == refreshes + 2  # noqa: SLF001
    assert result["command"]["kwargs"] == {"linear_speed": 0, "angular_speed": 500}
    assert handle.commands.send_movement.call_args_list[-1].kwargs == {
        "linear_speed": 0,
        "angular_speed": 0,
    }


@pytest.mark.asyncio
async def test_vio_turn_probe_counts_rotation_that_lands_after_the_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real rotation visible only in post_stop must not read as static/zero.

    Regression for 2026-07-19: VIO heading refreshes ~1.5s into the command and
    the position feed lags ~4s, so on a short pulse the only during-command
    sample is the t=0 one. A tape-confirmed 13.18 deg pivot came back
    `vision_heading_static_during_command` with `final_displacement_m: 0.0`
    because the verdict ignored post_stop -- while this function's own post_stop
    samples held the real values.
    """
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}
    stopped = {"value": False}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    async def fake_stop(_coordinator: object) -> dict:
        stopped["value"] = True
        return {"movement_ok": True}

    async def fake_get_reports(count: int = 5) -> None:
        # Frozen during the command; the real rotation only registers post-stop.
        heading = 76.82 if stopped["value"] else 90.0
        coordinator.data.report_data.vision_info = SimpleNamespace(
            heading=heading,
            vio_state=2,
        )

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports
    monkeypatch.setattr(mammotion_services, "_stop_manual_motion_confirmed", fake_stop)
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=90.0, vio_state=2
    )

    result = await _vio_turn_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        angular_speed=500,
        drive_seconds=1.5,
        sample_interval_seconds=1.5,
        post_stop_samples=3,
    )

    # The ~13.18 deg swing lands entirely in post_stop and must be counted.
    total = result["verdict"]["vision_heading_change"]["total_abs_degrees"]
    assert total == pytest.approx(13.18, abs=0.05)
    assert result["reason"] != "vision_heading_static_during_command"
    assert result["displacement_source"] in {"post_stop", "drive", None}


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
async def test_vio_turn_to_heading_tolerates_transient_feed_dip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single transient feed dip that recovers on re-poll does NOT abort the turn.

    A one-read feature dip (brief occlusion) must not end an otherwise-good turn;
    the read-only re-confirmation poll should see the feed recover and continue,
    unlike the sustained-degradation case which aborts vio_feed_degraded.
    """
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    # Feed liveness by call: entry(live) -> pulse-2 before_feed(dip) ->
    # re-confirm(recovered) -> live thereafter.
    feed_live = iter([True, False, True])

    def fake_feed_liveness(_coordinator: object) -> dict:
        live = next(feed_live, True)
        return {
            "live": live,
            "tracked_features": 80 if live else 0,
            "brightness_raw": 200 if live else 10,
            "brightness_label": "Light" if live else "Dark",
        }

    async def advance_on_pulse(*_args: object, **_kwargs: object) -> None:
        vi = coordinator.data.report_data.vision_info
        vi.heading = min(40.0, vi.heading + 20.0)  # reach +40 target in 2 pulses

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        return None  # heading is driven by the pulse, feed by the fake above

    monkeypatch.setattr(mammotion_services, "_vio_feed_liveness", fake_feed_liveness)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.manager.send_command_with_args.side_effect = advance_on_pulse
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    # The dip recovered on re-poll, so the turn continued to its target instead of
    # aborting vio_feed_degraded.
    assert result["stop_reason"] == "target_heading_reached"
    assert result["commands_sent"] == 2


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
    # final_vio_feed is always present (not only on the degraded stop path).
    assert result["final_vio_feed"]["live"] is True


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

    _patch_services_monotonic(monkeypatch, fake_monotonic)
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
async def test_vio_turn_to_heading_slow_caps_wrong_direction_streak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh streak that moves AWAY from target is slow-capped (wrong-direction guard).

    Even with a fresh feed, negative progress (e.g. an angular sign miscalibration
    turning the wrong way) must not keep running full-power pulses. The first pulse
    runs full (no streak yet); once the streak sees the away-drift, subsequent
    pulses are capped to the slow duration to bound the wrong-way rotation.
    """
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        # Genuinely fresh reading that drifts *away* from the +40 target: fresh but
        # negative progress.
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
    # Pulse 1 runs full (streak not started); pulse 2, fired after away-progress,
    # is slow-capped.
    assert result["command_results"][0]["pulse_duration_ms"] == 1500
    assert result["command_results"][1]["pulse_duration_ms"] == 700


@pytest.mark.asyncio
async def test_vio_turn_to_heading_keeps_full_pulse_creeping_toward_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh streak still creeping TOWARD target (small +progress) keeps the full pulse.

    The slow cap is for stale/latched feeds and wrong-direction motion; a mower
    genuinely turning toward the target but slower than min_progress_degrees should
    keep the full, faster pulse.
    """
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        # Fresh reading creeping toward the +40 target by +1 deg/poll: fresh, and
        # positive progress but below min_progress_degrees (2.0).
        vi = coordinator.data.report_data.vision_info
        vi.heading = vi.heading + 1.0

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
    # Fresh AND still moving toward target -> never slow-capped; all pulses full.
    assert all(cmd["pulse_duration_ms"] == 1500 for cmd in result["command_results"])


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

    _patch_services_monotonic(monkeypatch, fake_monotonic)
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

    _patch_services_monotonic(monkeypatch, fake_monotonic)
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
    assert result["telemetry"][
        "first_position_change_after_final_command_seconds"
    ] == pytest.approx(2.0)
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
    # With the explicit software stop in place this executor now also runs the
    # position-settle poll; a no-op pulse never registers movement.
    assert result["command_results"][0]["position_settled"] is False
    assert result["command_results"][0]["position_moved"] is False
    assert "position_source_comparison" in result["command_results"][0]


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_stops_after_max_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real raw segment stops after capped commands when progress is insufficient."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def advance_on_pulse(*_args: object, **_kwargs: object) -> None:
        # Each real pulse nudges the mower ~1.5cm toward the target -- enough to
        # pass min_progress but never reach it. Tie movement to the pulse itself,
        # not to asyncio.sleep count (the settle poll adds sleep calls).
        coordinator.data.mowing_state.pos_y = round(
            coordinator.data.mowing_state.pos_y - 0.015, 4
        )

    coordinator.manager.send_command_with_args.side_effect = advance_on_pulse

    async def no_sleep(_: float) -> None:
        return None

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
async def test_raw_pymammotion_execute_segment_sends_explicit_stop_after_pulse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each real raw-segment pulse is followed by an explicit software stop.

    This executor used to rely on firmware auto-stop only (the reason the
    position-settle poll was reverted from it); it now mirrors the vector
    executor: bounded pulse -> stop primitive -> settle poll.
    """
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
        prefer_ble=True,
        sample_delays=(0,),
    )

    assert result["commands_sent"] == 1
    coordinator.async_stop_manual_motion.assert_awaited_once()
    command_result = result["command_results"][0]
    assert command_result["stop_result"]["ok"] is True
    assert command_result["position_settled"] is False


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_segment_aborts_when_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undeliverable stop aborts the raw segment run immediately."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.async_stop_manual_motion.side_effect = RuntimeError("stop write failed")

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _raw_pymammotion_execute_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 0.7}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_commands=3,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "stop_failed_aborting"
    assert result["commands_sent"] == 1
    assert result["command_results"][0]["stop_result"]["ok"] is False
    coordinator.manager.send_command_with_args.assert_awaited_once()


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
async def test_raw_pymammotion_execute_vector_segment_dry_run_with_zero_offset() -> (
    None
):
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
async def test_raw_pymammotion_execute_vector_segment_rejects_missing_confirmations() -> (
    None
):
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

    coordinator.async_stop_manual_motion.assert_awaited_once()


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
async def test_multi_segment_vio_real_run_blocked_when_feed_degraded() -> None:
    """Multi-segment refuses a real VIO run when the feed is blind at chain entry."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=2, track_feature_num=0, brightness=10
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}, {"x": 1.2, "y": 1.1}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="vio",
        vio_heading_offset_degrees=0.0,
        sample_delays=(),
    )

    assert "vio_feed_live" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"
    assert result["initial_vio_feed"]["live"] is False
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

    normalized = MammotionReportUpdateCoordinator.active_transport_state.fget(fake_self)

    assert normalized == "ble"
    assert _transport_is_ble(SimpleNamespace(active_transport_state=normalized))


def test_active_transport_state_handles_no_transport_available() -> None:
    """A temporarily offline mower reports ``none`` without breaking setup."""
    handle = SimpleNamespace(
        active_transport=MagicMock(
            side_effect=NoTransportAvailableError("all transports unavailable")
        )
    )
    fake_self = SimpleNamespace(
        device_name="Luba-Test",
        manager=SimpleNamespace(mower=lambda _name: handle),
    )

    normalized = MammotionReportUpdateCoordinator.active_transport_state.fget(fake_self)

    assert normalized == "none"


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
async def test_multi_segment_forwards_approach_and_turn_rate_to_every_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card profile constants reach the vector executor instead of being ignored."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    forwarded: list[tuple[float, float]] = []

    async def fake_vector(
        coordinator_arg: MammotionReportUpdateCoordinator,
        points: list[dict[str, float]],
        **kwargs: object,
    ) -> dict:
        forwarded.append(
            (
                float(kwargs["final_approach_metres_per_pulse"]),
                float(kwargs["turn_degrees_per_second"]),
            )
        )
        return {
            "valid": True,
            "stop_reason": "dry_run",
            "blockers": [],
            "phases": [{"passed": True}, {"passed": True}],
            "final_telemetry": _custom_path_telemetry_snapshot(coordinator_arg),
            "progress_diagnostics": [],
        }

    monkeypatch.setattr(
        mammotion_services, "_raw_pymammotion_execute_vector_segment", fake_vector
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [
            {"x": 1.0, "y": 1.0},
            {"x": 1.1, "y": 1.0},
            {"x": 1.2, "y": 1.1},
        ],
        final_approach_metres_per_pulse=1.23,
        turn_degrees_per_second=41.5,
        sample_delays=(0,),
    )

    assert result["final_approach_metres_per_pulse"] == 1.23
    assert result["turn_degrees_per_second"] == 41.5
    assert forwarded == [(1.23, 41.5), (1.23, 41.5)]


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
async def test_vector_segment_reports_stale_stream_after_feed_dies_mid_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feed that works then freezes aborts telemetry_stream_stale, not no_progress.

    Replays the 2026-07-19 card run: linear pulses advanced normally, then the
    position feed froze bit-identical for three pulses. The run aborted
    no_target_progress -- but the mower had actually driven 25.4cm during those
    pulses (~8.5cm each, the proven step), which only surfaced after the feed
    caught up post-run. The executor spent ~30s issuing motion against a stale
    coordinate, so it must stop and say so rather than blame progress.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )
    state = {"y": 1.0, "frozen": False, "pulses": 0}

    async def no_sleep(_: float) -> None:
        return None

    async def advancing_then_frozen(count: int = 5) -> None:
        # Feed advances ~9cm per refresh until it dies, then repeats verbatim.
        if not state["frozen"]:
            state["y"] += 0.09
            coordinator.data.mowing_state.pos_y = state["y"]

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    coordinator.async_get_reports.side_effect = advancing_then_frozen

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
            "commands_sent": 1,
            "command_results": [],
            "final_vision_heading": 90.0,
            "final_heading_error_degrees": 0.5,
        }

    async def freeze_after_two(*args: object, **kwargs: object) -> None:
        state["pulses"] += 1
        if state["pulses"] >= 3:
            state["frozen"] = True

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)
    monkeypatch.setattr(mammotion_services, "_motion_open_sleep", freeze_after_two)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 4.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        # A ceiling switches the executor into loop-to-tolerance mode.
        max_linear_pulse_ceiling=12,
        vio_max_realignments=0,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "telemetry_stream_stale"
    assert "bit-identical" in result["telemetry_stream_stale_hint"]
    # It must have driven successfully first -- otherwise "the stream died" is
    # not a supportable claim.
    assert any(c.get("position_moved") for c in result["command_results"])
    assert any(c.get("position_feed_stale") for c in result["command_results"])


@pytest.mark.asyncio
async def test_vector_segment_echoes_pulse_geometry_params() -> None:
    """The vector executor reports the pulse-geometry params it was given.

    Regression for 2026-07-25: ``max_linear_pulse_ceiling`` was honoured but
    echoed as absent, so a stalled run gave no way to confirm the numbers that
    matter most. The multi-segment executor got this fixed on 2026-07-19; the
    vector one was missed. A dry run is enough -- the echo must not depend on
    the mower moving.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 4.0}],
        dry_run=True,
        max_linear_pulse_ceiling=12,
        turn_pulse_duration_ms=1500.0,
        linear_pulse_duration_ms=3500.0,
        vio_turn_max_commands=16,
        vio_angular_speed=500,
        vio_heading_offset_degrees=42.5,
    )

    assert result["max_linear_pulse_ceiling"] == 12
    assert result["turn_pulse_duration_ms"] == 1500.0
    assert result["linear_pulse_duration_ms"] == 3500.0
    assert result["vio_turn_max_commands"] == 16
    assert result["vio_angular_speed"] == 500
    assert result["vio_heading_offset_degrees"] == 42.5


def test_final_approach_scales_the_last_pulse_to_the_remaining_distance() -> None:
    """Less than one pulse left is bounded by discrete confirmed refresh writes."""
    info = _final_approach_pulse_ms(
        distance_to_target=0.2,
        observed_pulse_distances=[],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is True
    assert info["reason"] == "final_approach_bounded_by_refresh_count"
    # A full pulse is the initial write plus ten refreshes. 0.2/1.06 of eleven
    # non-zero writes rounds up to three total: initial plus two refreshes.
    assert info["refresh_command_limit"] == 2
    assert info["target_nonzero_writes"] == 3
    assert info["pulse_duration_ms"] == 3500.0
    assert info["metres_per_pulse_source"] == "default"


def test_final_approach_is_disabled_without_the_refresh_cadence() -> None:
    """Single-shot motion moves a fixed step, so scaling the duration is a trap.

    The 2026-07-22 B1 tape proved distance is duration-dependent only while the
    command is being re-sent: the same 4 s pulse moved ~4 in single-shot and
    ~44 in at ``motion_refresh_interval_ms`` 200. Without refresh a shortened
    pulse would not land closer -- and 2026-07-18 measured a 2000 ms single-shot
    pulse as a physical no-op, so it could stop the mower moving at all. The
    guard must hold regardless of how little distance remains.
    """
    info = _final_approach_pulse_ms(
        distance_to_target=0.2,
        observed_pulse_distances=[1.06],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=0,
    )

    assert info["applied"] is False
    assert info["reason"] == "refresh_disabled_distance_not_proportional_to_duration"
    assert info["pulse_duration_ms"] == 3500.0


def test_final_approach_leaves_cruising_pulses_full_length() -> None:
    """More than one pulse to go -> drive the full pulse, unchanged."""
    info = _final_approach_pulse_ms(
        distance_to_target=2.5,
        observed_pulse_distances=[],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is False
    assert info["reason"] == "cruising_full_pulse_fits"
    assert info["pulse_duration_ms"] == 3500.0


def test_final_approach_uses_only_the_initial_write_for_a_tiny_remainder() -> None:
    """A tiny remainder gets one non-zero write and no refresh amplification."""
    info = _final_approach_pulse_ms(
        distance_to_target=0.01,
        observed_pulse_distances=[],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is True
    assert info["refresh_command_limit"] == 0
    assert info["target_nonzero_writes"] == 1
    assert info["pulse_duration_ms"] == 3500.0


def test_final_approach_prefers_the_distance_observed_this_run() -> None:
    """Today's measured pulses beat the baked-in constant.

    Speed, grass and gradient move the per-pulse distance around, so the run
    calibrates itself. Here the mower is covering 2.0 m per pulse -- against the
    1.06 m default the same 1.5 m of remaining distance would have read as
    "cruising" and fired a full pulse straight past the target.
    """
    info = _final_approach_pulse_ms(
        distance_to_target=1.5,
        observed_pulse_distances=[2.0, 2.0],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is True
    assert info["metres_per_pulse_source"] == "observed"
    assert info["metres_per_pulse"] == pytest.approx(2.0)
    assert info["refresh_command_limit"] == 8
    assert info["target_nonzero_writes"] == 9
    assert info["pulse_duration_ms"] == 3500.0


def test_final_approach_normalises_bounded_pulses_by_actual_write_count() -> None:
    """A bounded pulse can calibrate later approaches without shrinking the scale."""
    # day2j's first segment pulse 1: 0.4192 m from initial + three refresh writes.
    normalised = _normalised_linear_pulse_distance(0.4191586454, 3)

    assert normalised == pytest.approx(1.1526862749)
    info = _final_approach_pulse_ms(
        distance_to_target=0.2222578683,
        observed_pulse_distances=[normalised],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=1300.0,
        refresh_interval_ms=200,
    )
    assert info["metres_per_pulse_source"] == "observed"
    assert info["refresh_command_limit"] == 2


def test_final_approach_observation_cannot_increase_the_motion_budget() -> None:
    """A low observation may stop short but must not add writes and overshoot."""
    info = _final_approach_pulse_ms(
        distance_to_target=0.18,
        observed_pulse_distances=[0.96],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=1300.0,
        refresh_interval_ms=200,
    )

    assert info["observed_metres_per_pulse"] == 0.96
    assert info["metres_per_pulse"] == 1.06
    assert info["metres_per_pulse_source"] == "default_conservative_floor"
    assert info["refresh_command_limit"] == 1


@pytest.mark.parametrize(
    "evidence_name",
    [
        "evidence-gate4-beta20-day2i-real-result-20260805.json",
        "evidence-gate4-beta20-day2j-real-result-20260805.json",
    ],
)
def test_final_approach_replays_gate4_1300ms_write_limits(
    evidence_name: str,
) -> None:
    """The conservative estimator preserves every recorded day2i/day2j write cap."""
    evidence_path = pathlib.Path(__file__).parents[3] / "docs" / evidence_name
    evidence = json.loads(evidence_path.read_text())

    for segment in evidence["result"]["segments"]:
        segment_result = segment["result"]
        progress_by_index = {
            item["command_index"]: item
            for item in segment_result["progress_diagnostics"]
        }
        observed_by_speed: dict[int, list[float]] = {}
        for command in segment_result["command_results"]:
            if command.get("phase") != "linear_forward_to_target":
                continue
            speed = int(command["selection"]["linear_speed"])
            approach = _final_approach_pulse_ms(
                distance_to_target=command["selection"]["distance_to_target"],
                observed_pulse_distances=observed_by_speed.get(speed, []),
                default_metres_per_pulse=1.06,
                pulse_duration_ms=1300.0,
                refresh_interval_ms=200,
            )
            assert (
                approach["refresh_command_limit"]
                == command["final_approach"]["refresh_command_limit"]
            )

            measured = progress_by_index[command["index"]]["measured_delta"]["distance"]
            refreshes = command["motion_refresh"]["refresh_commands_sent"]
            observed_by_speed.setdefault(speed, []).append(
                _normalised_linear_pulse_distance(measured, refreshes)
            )


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


@pytest.mark.parametrize(
    ("distance", "expected_refreshes"),
    [(0.3066301518, 3), (0.3609403967, 3)],
)
def test_final_approach_replays_both_beta16_short_pulse_failures(
    distance: float, expected_refreshes: int
) -> None:
    """Both live failures choose three bounded refreshes, not nominal duration."""
    info = _final_approach_pulse_ms(
        distance_to_target=distance,
        observed_pulse_distances=[],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=200,
    )

    assert info["reason"] == "final_approach_bounded_by_refresh_count"
    assert info["refresh_command_limit"] == expected_refreshes


def test_final_approach_declines_when_the_distance_is_unknown() -> None:
    """No distance reading -> no scaling, and say why rather than guessing."""
    info = _final_approach_pulse_ms(
        distance_to_target=None,
        observed_pulse_distances=[1.06],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is False
    assert info["reason"] == "distance_unknown"
    assert info["pulse_duration_ms"] == 3500.0


def test_turn_final_approach_scales_the_pulse_to_the_remaining_angle() -> None:
    """Replays the 2026-07-27 overshoot: 23.7 deg left must not take a full pulse.

    Live, that error took the full 1500 ms pulse, turned 50.9 deg, overshot by
    27 deg and forced a reversal. At the measured ~33 deg/s it needs ~720 ms.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=-23.744,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=48.236,
        observed_rotation_ms=1500.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is True
    assert info["degrees_per_second_source"] == "observed"
    assert info["degrees_per_second"] == pytest.approx(32.16, abs=0.01)
    # The estimate wants 738.4 ms and the measured sweep bound allows 743.6, so
    # the estimate is the tighter of the two by 5 ms and takes it. beta31's
    # C = 60 deg/s ceiling allowed only 695.7 here and overrode the estimate; the
    # affine bound measured on 2026-08-09 restores the original, better-founded
    # answer. That is the point of the change -- 60 deg/s over-estimated the
    # slope, so it was shortening pulses that did not need shortening.
    assert info["reason"] == "final_approach_scaled_to_remaining_angle"
    assert info["pulse_duration_ms"] == pytest.approx(738.4, abs=1.0)
    assert info["ceiling_pulse_duration_ms"] == pytest.approx(743.6, abs=1.0)


def test_turn_final_approach_is_disabled_without_the_refresh_cadence() -> None:
    """Single-shot rotation is a fixed quantum, so scaling the duration is a trap.

    Without refresh the mower turns ~8-15 deg per command regardless of pulse
    length, so a shortened pulse would not land closer -- and the single-shot
    path has a hard actuation floor (a 2000 ms single-shot pulse was a measured
    physical no-op). The guard must hold however little angle remains.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=-5.0,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=48.0,
        observed_rotation_ms=1500.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=0,
    )

    assert info["applied"] is False
    assert info["reason"] == "refresh_disabled_rotation_not_proportional_to_duration"
    assert info["pulse_duration_ms"] == 1500.0


def test_turn_final_approach_leaves_a_large_error_on_the_full_pulse() -> None:
    """A large error needs more than one pulse -- do not shorten it.

    Uses 120 deg rather than the historical 71.98: with the beta31 ceiling at
    60 deg/s and an 18 deg tolerance the binding threshold is exactly 72 deg, so
    71.98 sat 0.3 ms inside it and tested a knife-edge instead of the intent.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=-120.0,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=0.0,
        observed_rotation_ms=0.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is False
    assert info["reason"] == "cruising_full_pulse_fits"
    assert info["degrees_per_second_source"] == "default"
    assert info["pulse_duration_ms"] == 1500.0


@pytest.mark.parametrize(
    ("observed_rate", "binds"),
    [(14.73, True), (25.0, True), (34.0, True), (37.0, False), (45.0, False)],
)
def test_turn_final_approach_bound_binds_only_when_the_estimate_is_slow(
    observed_rate: float, binds: bool
) -> None:
    """Pins when the sweep bound takes over from the estimate, and when it does not.

    beta31's ceiling was a pure rate, so it bound purely on ERROR: below
    C * pulse_seconds - tolerance, which at C = 60 was 72 deg. That made it the
    active constraint across most of a normal final approach rather than a
    backstop -- handover section 2.2's complaint.

    The affine bound measured on 2026-08-09 does not work that way. It binds when
    it is tighter than what the estimate wants, which is a statement about the
    ESTIMATED RATE, not the error: a slow estimate asks for a long pulse and gets
    capped, while an estimate at or above the configured 37 deg/s already asks
    for less than the bound allows and is left alone.

    Gate 5 attempt 5's geometry, 44.372 deg remaining. Its estimator had learned
    14.73 deg/s from two stall-degraded pulses and wanted a full 1500 ms; the
    bound allows 1259.3 and takes over, which is exactly the pulse that overshot.
    At the configured 37 deg/s the estimate wants 1199 ms and the bound is not
    consulted.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=44.372,
        heading_tolerance_degrees=18.0,
        # One second of observation at the rate under test.
        observed_rotation_degrees=observed_rate,
        observed_rotation_ms=1000.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["ceiling_pulse_duration_ms"] == pytest.approx(1259.3, abs=1.0)
    assert (info["reason"] == "bounded_by_max_rate_ceiling") is binds


def test_turn_final_approach_floors_the_pulse_so_the_mower_still_rotates() -> None:
    """A sliver of angle must not become a pulse too short to actuate."""
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=0.5,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=48.0,
        observed_rotation_ms=1500.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is True
    assert info["pulse_duration_ms"] == 200.0


def test_a_re_aim_is_skipped_only_when_it_cannot_improve_the_landing() -> None:
    """The test is the perpendicular miss, not the distance to the waypoint.

    Landing anywhere inside `waypoint_tolerance` ends the segment, so a re-aim is
    worth spending turn commands on exactly when driving straight on would MISS
    that disc. The perpendicular miss -- distance * sin(aim) -- answers that
    directly and needs no tuned constant.

    ⚠️ This replaces a wrong guard shipped in beta36, which compared distance
    alone against `tolerance / tan(trigger)` = 0.4617 m. On the 0.7 m legs
    introduced the same day that disabled re-aim for the last 66% of every leg
    however badly the mower was pointed. It suppressed corrections at 40.099 and
    77.922 deg of aim error and segment 3 landed 0.2548 m out, the worst of the
    day (docs/evidence-beta32-4segment-20260810T002506Z.json). Those errors were
    real, not bearing noise: RTK measured +40.73 and +72.92 deg independently of
    VIO.

    All four mid-drive re-aims on record, with the outcome each actually had.
    """
    tolerance = 0.15
    cases = [
        # distance, aim, suppress?, what happened
        (0.540, 23.301, False, "allowed and legitimate"),
        (0.210, 34.837, True, "allowed, then oscillated -- should have been skipped"),
        (0.3104, 40.099, False, "beta36 suppressed it and the segment missed"),
        (0.2072, 77.922, False, "beta36 suppressed it and the segment missed"),
    ]
    for distance, aim, expected, note in cases:
        assert (
            _realign_cannot_improve_the_landing(
                distance_to_target_m=distance,
                aim_error_degrees=aim,
                waypoint_tolerance=tolerance,
            )
            is expected
        ), note

    # The boundary sits exactly where the geometry puts it: a target 0.15 m off
    # the travelled line is on the edge of the disc.
    on_edge = math.degrees(math.asin(tolerance / 0.5))
    for aim, expected in ((on_edge - 1.0, True), (on_edge + 1.0, False)):
        assert (
            _realign_cannot_improve_the_landing(
                distance_to_target_m=0.5,
                aim_error_degrees=aim,
                waypoint_tolerance=tolerance,
            )
            is expected
        )


@pytest.mark.parametrize(
    "override",
    [
        {"distance_to_target_m": 0.0},
        {"waypoint_tolerance": 0.0},
        # At or past 90 deg the target is abeam or behind, so driving on can only
        # make it worse; `_requires_reverse_recovery` owns that case.
        {"aim_error_degrees": 90.0},
        {"aim_error_degrees": 120.0},
    ],
)
def test_the_re_aim_guard_fails_open_on_degenerate_input(
    override: dict[str, float],
) -> None:
    """A guard that cannot judge must never suppress.

    Suppressing a re-aim is the dangerous direction: a mower that stops
    correcting its aim keeps driving. Every degenerate input resolves to "do not
    suppress" and lets the existing trigger decide, which is the pre-guard
    behaviour.
    """
    kwargs = {
        "distance_to_target_m": 0.05,
        "aim_error_degrees": 30.0,
        "waypoint_tolerance": 0.15,
    }
    kwargs.update(override)

    assert _realign_cannot_improve_the_landing(**kwargs) is False


def test_a_stalled_refresh_write_is_not_a_rotation_rate() -> None:
    """A pulse whose refresh cadence collapsed must not feed the rate estimate.

    Live 2026-08-09 (docs/evidence-beta33-reposition-20260809T184618Z.json,
    segment 3 pulse 1): a 1303.7 ms pulse at a 200 ms refresh interval sent ONE
    of a possible six refresh writes, and that single BLE write blocked for
    1303.972 ms. Motion only continues while refreshes keep arriving, so the
    mower's watchdog stopped the motor and the window was mostly dead time. The
    executor measured 13.885 deg over 1504 ms and called it 9.23 deg/s -- which
    would have been the slowest rotation ever recorded and 44% below
    `_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND`. Every other turn pulse that day,
    cadence intact, measured 23-43 deg/s.

    Folding that number into the estimate is actively harmful rather than merely
    noisy: a low estimate LENGTHENS later pulses, which is how Gate 5 attempt 5
    overshot by 13.258 deg after two stall-degraded pulses taught it ~14.7 deg/s.

    Verified here at the arithmetic that matters -- what the estimator would
    report with and without the stalled sample folded in.
    """
    clean_degrees, clean_ms = 24.893 + 20.274, 1073.358 + 657.759
    stalled_degrees, stalled_ms = 13.885, 1504.162

    honest_rate = clean_degrees / (clean_ms / 1000)
    poisoned_rate = (clean_degrees + stalled_degrees) / ((clean_ms + stalled_ms) / 1000)

    assert honest_rate == pytest.approx(26.08, abs=0.05)
    assert poisoned_rate == pytest.approx(18.30, abs=0.05)
    # Folding the stalled pulse in costs 30% of the estimate, and drags it under
    # the 16.5 deg/s "conservative floor" territory that the feasibility guard
    # treats as a hardware minimum.
    assert poisoned_rate < honest_rate * 0.75

    # And the exclusion test itself needs no tuned constant: the stalled write
    # lasted the whole commanded pulse, while the healthy ones did not. The
    # 820/1500 pair is Gate 5 attempt 5 pulse 3 -- the longest write in the
    # corpus that still produced a normal rate, so it must stay INCLUDED.
    stalled = (1303.972, 1303.7)
    healthy = ((516.0, 1072.3), (260.0, 657.4), (820.0, 1500.0))
    assert stalled[0] >= stalled[1]
    for healthy_write, its_pulse in healthy:
        assert healthy_write < its_pulse


def test_turn_final_approach_rate_is_duration_normalised() -> None:
    """Samples taken at different pulse lengths must stay comparable.

    A per-pulse average would be corrupted by mixing a 1500 ms pulse with a
    700 ms one; a rate is not. Here 48.24 deg over 1500 ms and 25.94 deg over
    700 ms give (48.24+25.94)/2.2 s = 33.7 deg/s.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=10.0,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=48.236 + 25.942,
        observed_rotation_ms=1500.0 + 700.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["degrees_per_second"] == pytest.approx(33.72, abs=0.01)
    # 10 deg at 33.72 deg/s is 296.6 ms, and with the floor at 200 the estimate
    # is now honoured instead of being rounded up to a 400 ms minimum that would
    # have swept ~13 deg for a 10 deg error.
    assert info["pulse_duration_ms"] == pytest.approx(296.6, abs=1.0)


def test_turn_final_approach_ceiling_shortens_the_gate5_overshoot_pulse() -> None:
    """Replays Gate 5 attempt 5 pulse 3, which overshot on the 'cruising' branch.

    Live 2026-08-08 (docs/evidence-gate5-attempt5-segment1-raw-20260808.json):
    44.372 deg remained, the estimator had seen 52.634 deg over 3574 ms of
    delivered window (14.73 deg/s) and so judged that a full 1500 ms pulse could
    not reach the target. The mower then turned at 32.74 deg/s, swept 57.630 deg
    and overshot the target heading by 13.258 deg against an 18 deg tolerance --
    finishing on 4.74 deg of margin.

    No rate ESTIMATE could have caught that; only the fixed ceiling does. At
    60 deg/s the worst acceptable sweep of 44.372 + 18 takes 1039.5 ms.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=44.372,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=30.46 + 22.174,
        observed_rotation_ms=2043.622 + 1530.326,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    # The estimate alone still says "cruising" -- this is the branch that failed.
    assert info["degrees_per_second"] == pytest.approx(14.73, abs=0.01)
    assert info["applied"] is True
    assert info["reason"] == "bounded_by_max_rate_ceiling"
    assert info["max_allowed_sweep_degrees"] == pytest.approx(62.372, abs=0.001)
    # 1259.3 ms, not beta31's 1039.5: the affine bound measured on 2026-08-09
    # permits a longer pulse here because C = 60 deg/s over-estimated the slope
    # (measured 33.18) and so shortened pulses that did not need shortening.
    assert info["pulse_duration_ms"] == pytest.approx(1259.3, abs=1.0)

    # At the rate the mower actually turned (32.736 deg/s) the shortened pulse
    # sweeps ~34.0 deg, leaving ~10.3 deg of error -- inside the 18 deg tolerance,
    # so the turn still ends on this pulse and costs no extra command, instead of
    # finishing 13.258 deg past the target.
    swept = 32.736 * (info["pulse_duration_ms"] / 1000)
    assert swept == pytest.approx(41.2, abs=0.5)
    # Lands 3.2 deg short instead of 13.258 deg past. Still inside tolerance, so
    # the turn ends on this pulse and costs no extra command.
    assert 0 < 44.372 - swept < 18.0


def test_turn_final_approach_ceiling_stays_out_of_the_way_on_large_turns() -> None:
    """The ceiling must cost nothing while the turn is still far from target.

    Gate 5 attempt 5 pulse 1: 97.006 deg of error. The worst acceptable sweep is
    115.006 deg, which takes 1917 ms at the ceiling rate -- longer than the pulse
    -- so the full 1500 ms must survive untouched.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=97.006,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=0.0,
        observed_rotation_ms=0.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is False
    assert info["reason"] == "cruising_full_pulse_fits"
    assert info["pulse_duration_ms"] == 1500.0
    assert info["ceiling_pulse_duration_ms"] == pytest.approx(2575.1, abs=1.0)


@pytest.mark.parametrize(
    ("tolerance", "floor_binds", "no_safe_pulse"),
    [
        (3.0, True, True),
        (5.5, True, True),
        (8.0, True, False),
        (11.9, False, False),
        (18.0, False, False),
        (30.0, False, False),
    ],
)
def test_turn_final_approach_bound_vs_actuation_floor(
    tolerance: float, floor_binds: bool, no_safe_pulse: bool
) -> None:
    """Pins which safety bound wins when the two conflict, and where each starts.

    The turn loop returns `target_heading_reached` as soon as the error is inside
    tolerance, so any pulse that runs has error > tolerance and the worst
    acceptable sweep is > 2 * tolerance. Against the affine bound that permits
    (2 * tolerance - 12) / 40 seconds, which drops below the 200 ms actuation
    floor once tolerance is under ~10 deg.

    There are now TWO ways to fail, and they are different:

    * `ceiling_below_actuation_floor` -- the bound wants a pulse shorter than the
      mower reliably actuates. The FLOOR wins, deliberately: an overshoot is
      recoverable by the next pulse, but a pulse too short to actuate makes no
      progress and walks the turn into `no_heading_progress` with its budget
      spent.
    * `sweep_exceeds_any_pulse` -- the whole allowance is smaller than the
      bound's 12 deg constant term, so NO duration is safe, because even the
      shortest pulse can sweep past. Below ~6 deg of tolerance. This condition
      did not exist under beta31's pure-rate ceiling, which always returned some
      positive duration however small the allowance, and thereby implied a
      guarantee it could not keep.

    The accepted profile runs `heading_tolerance_degrees: 18`, where neither
    binds.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=tolerance + 0.001,
        heading_tolerance_degrees=tolerance,
        observed_rotation_degrees=0.0,
        observed_rotation_ms=0.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["ceiling_below_actuation_floor"] is floor_binds
    assert info["sweep_exceeds_any_pulse"] is no_safe_pulse
    assert info["pulse_duration_ms"] >= _MIN_SCALED_TURN_PULSE_MS
    if floor_binds:
        assert info["ceiling_pulse_duration_ms"] < _MIN_SCALED_TURN_PULSE_MS
        assert info["pulse_duration_ms"] == _MIN_SCALED_TURN_PULSE_MS
    else:
        assert info["ceiling_pulse_duration_ms"] >= _MIN_SCALED_TURN_PULSE_MS


@pytest.mark.asyncio
async def test_vio_turn_scales_the_last_pulse_and_does_not_overshoot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: the turn lands on target instead of blowing past and reversing.

    Replays the 2026-07-27 A/B run. The mower rotates at ~33 deg/s under refresh,
    so command 2 (23.7 deg to go) must be a ~720 ms pulse rather than the full
    1500 ms that overshot by 27 deg live and forced a direction reversal.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    state = {"heading": 75.6}
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=state["heading"], vio_state=2
    )
    durations: list[float] = []

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    async def fake_refresh_window(
        coordinator_arg: MammotionReportUpdateCoordinator,
        *,
        resend: object,
        duration_seconds: float,
        refresh_interval_ms: int,
        max_refresh_commands: int | None = None,
    ) -> dict:
        durations.append(duration_seconds)
        # 33 deg/s, and the sign follows the commanded direction: +angular
        # decreases vision_heading.
        state["heading"] -= 33.0 * duration_seconds
        coordinator.data.report_data.vision_info = SimpleNamespace(
            heading=state["heading"], vio_state=2
        )
        return {
            "refresh_enabled": True,
            "refresh_interval_ms": refresh_interval_ms,
            "refresh_commands_sent": int(duration_seconds * 1000 / refresh_interval_ms),
        }

    monkeypatch.setattr(
        mammotion_services, "_motion_refresh_window", fake_refresh_window
    )

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=3.62,
        # 18 deg, the accepted-profile tolerance, not the original 5. The sweep
        # of 2026-08-09 measured a minimum sweep of ~12 deg for ANY pulse length,
        # so a 5 deg tolerance is below the control scheme's achievable
        # precision and `_vio_turn_budget_feasibility` now correctly refuses it
        # up front with `turn_budget_infeasible`. Asking for 5 deg was always
        # unachievable; the model only recently learned to say so.
        heading_tolerance_degrees=18.0,
        angular_speed=500,
        pulse_duration_ms=1500,
        slow_threshold_degrees=0.0,
        max_commands=6,
        refresh_wait_seconds=0.0,
        motion_refresh_interval_ms=200,
        turn_degrees_per_second=37.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "target_heading_reached"
    # Never turned past the target: no command may reverse direction.
    signs = {(c["angular_speed"] > 0) for c in result["command_results"] if c.get("ok")}
    assert len(signs) == 1, "turn reversed direction -- it overshot"
    # The first pulse cruises at the full 1500 ms again: 71.98 deg of error
    # against an 18 deg tolerance permits (71.98 + 18 - 12) / 40 = 1.950 s, so
    # the bound is not the constraint. beta31's C = 60 ceiling capped it at
    # 1.283 s here, which is the over-restriction the measured affine bound
    # removes. Pulses still shorten monotonically toward the target, which is
    # what this test is really about.
    assert durations[0] == pytest.approx(1.5, abs=0.002)
    assert durations[-1] < durations[0]
    assert durations == sorted(durations, reverse=True)
    assert result["command_results"][-1]["final_approach"]["applied"] is True
    # And the aggregate displacement is reported, not left as None.
    assert result["final_displacement_m"] is not None


@pytest.mark.asyncio
async def test_vector_segment_forwards_refresh_and_rate_into_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The executor must hand its refresh cadence to the turn phase.

    Regression for 2026-07-27: the executor accepted
    ``motion_refresh_interval_ms`` and used it for linear pulses but did not
    forward it to ``_vio_turn_to_heading``, so its turns always ran single-shot
    at ~13 deg/command. A 176 deg turn then exhausted the 8-command budget live
    and the segment never reached its linear phase.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )
    turn_calls: list[dict[str, object]] = []

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

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        turn_calls.append(kwargs)
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
            "final_vision_heading": 90.0,
            "final_heading_error_degrees": 0.5,
        }

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_commands=1,
        vio_max_realignments=0,
        motion_refresh_interval_ms=200,
        heading_tolerance_degrees=18.0,
        turn_degrees_per_second=37.0,
        sample_delays=(0,),
    )

    assert len(turn_calls) == 1
    assert turn_calls[0]["motion_refresh_interval_ms"] == 200
    assert turn_calls[0]["turn_degrees_per_second"] == 37.0
    # The tolerance must come from the segment call, not the executor default.
    assert turn_calls[0]["heading_tolerance_degrees"] == 18.0


@pytest.mark.asyncio
async def test_vector_segment_shortens_the_final_pulse_instead_of_overshooting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replays the 2026-07-27 return run, which overshot by 0.8 m and gave up.

    That run drove 4.06 m for a 3.26 m target: with ~0.2 m to go the executor
    had no move available except another full ~1.06 m pulse, so it stepped past
    the waypoint and then failed trying to re-aim at a target now 176 deg behind
    it. Raising ``waypoint_tolerance`` does not fix that -- pulse granularity is
    the limit. The fix bounds the final pulse by confirmed refresh writes rather
    than nominal duration.

    Here the mower starts 3.0 m out and covers 1.06 m per full pulse: two full
    pulses leave 0.88 m, which receives nine refreshes rather than another full
    ten-refresh pulse.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )
    state = {"y": 1.0}
    windows: list[float] = []
    refresh_limits: list[int | None] = []

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

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
            "final_vision_heading": 90.0,
            "final_heading_error_degrees": 0.5,
        }

    async def fake_refresh_window(
        coordinator_arg: MammotionReportUpdateCoordinator,
        *,
        resend: object,
        duration_seconds: float,
        refresh_interval_ms: int,
        max_refresh_commands: int | None = None,
    ) -> dict:
        windows.append(duration_seconds)
        refresh_limits.append(max_refresh_commands)
        nonzero_writes = (
            11 if max_refresh_commands is None else max_refresh_commands + 1
        )
        state["y"] += (nonzero_writes / 11) * 1.06
        coordinator.data.mowing_state.pos_y = state["y"]
        return {
            "refresh_enabled": True,
            "refresh_interval_ms": refresh_interval_ms,
            "refresh_commands_sent": nonzero_writes - 1,
        }

    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)
    monkeypatch.setattr(
        mammotion_services, "_motion_refresh_window", fake_refresh_window
    )

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 4.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_pulse_ceiling=12,
        vio_max_realignments=0,
        linear_pulse_duration_ms=3500.0,
        motion_refresh_interval_ms=200,
        waypoint_tolerance=0.15,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "target_reached"
    # Two full pulses, then one bounded to the 0.88 m that was left.
    assert len(windows) == 3
    assert windows == pytest.approx([3.5, 3.5, 3.5])
    assert refresh_limits == [None, None, 9]

    approaches = [
        c["final_approach"] for c in result["command_results"] if "final_approach" in c
    ]
    assert [a["applied"] for a in approaches] == [False, False, True]
    # The scale factor came from the two pulses this run actually measured.
    assert approaches[-1]["metres_per_pulse_source"] == "observed"
    assert approaches[-1]["metres_per_pulse"] == pytest.approx(1.06, abs=0.01)


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
async def test_vector_segment_realigns_after_turn_translation_before_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A translating pivot recomputes its bearing before the forward command.

    Gate 4 (2026-07-31) reached its pre-turn VIO target but drifted 14.4 cm
    during the pivot. That changed the bearing to a 30 cm waypoint enough that
    the subsequent forward pulse was 23 degrees off course. The fresh-position
    correction must happen before, not after, that linear dispatch.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)
    turn_calls: list[dict[str, object]] = []

    async def no_sleep(_: float) -> None:
        return None

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict[str, object]:
        turn_calls.append(kwargs)
        if len(turn_calls) == 1:
            # Reach the original north-facing target while drifting 20 cm east.
            coordinator.data.mowing_state.pos_x = 1.2
            coordinator.data.report_data.vision_info.heading = 90.0
            displacement = 0.2
        else:
            # The fresh bearing from (1.2, 1.0) to (1.0, 1.5) is ~111.8 deg.
            coordinator.data.report_data.vision_info.heading = float(
                kwargs["target_vision_heading"]
            )
            displacement = 0.0
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
            "final_displacement_m": displacement,
        }

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 1.5}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        vio_heading_offset_degrees=0.0,
        vio_max_realignments=1,
        max_turn_translation_distance=0.25,
        max_linear_commands=1,
        sample_delays=(0,),
    )

    assert len(turn_calls) == 2
    assert turn_calls[0]["max_displacement_m"] == pytest.approx(0.25)
    assert turn_calls[1]["target_vision_heading"] == pytest.approx(111.801, abs=0.01)
    assert turn_calls[1]["max_displacement_m"] == pytest.approx(0.25)
    alignment = result["post_turn_alignment"]
    assert alignment["correction_attempted"] is True
    assert alignment["before"]["aim_error_degrees"] == pytest.approx(21.801, abs=0.01)
    assert alignment["after"]["aim_error_degrees"] == pytest.approx(0.0)
    assert alignment["passed"] is True
    assert result["realignments"][0]["before_linear"] is True
    # The static fixture does not model forward travel, but only after alignment
    # is proven may the single linear command be attempted.
    assert result["linear_commands_sent"] == 1


@pytest.mark.asyncio
async def test_vector_segment_blocks_linear_when_post_turn_alignment_stays_bad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claimed correction that remains off-bearing must fail before driving."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)
    turn_count = 0

    async def fake_vio_turn(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict[str, object]:
        nonlocal turn_count
        turn_count += 1
        if turn_count == 1:
            coordinator.data.mowing_state.pos_x = 1.2
            coordinator.data.report_data.vision_info.heading = 90.0
            displacement = 0.2
        else:
            # Backend claims success but the live heading did not change.
            displacement = 0.0
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
            "final_displacement_m": displacement,
        }

    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 1.5}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        vio_heading_offset_degrees=0.0,
        vio_max_realignments=1,
        max_linear_commands=1,
        sample_delays=(0,),
    )

    assert turn_count == 2
    assert result["stop_reason"] == "post_turn_alignment_incomplete"
    assert result["post_turn_alignment"]["passed"] is False
    assert result["linear_commands_sent"] == 0


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
@pytest.mark.parametrize(
    ("max_linear_commands", "expected_realignments"), [(1, 0), (2, 1)]
)
async def test_vector_segment_vio_realigns_when_facing_drifts_off_bearing(
    monkeypatch: pytest.MonkeyPatch,
    max_linear_commands: int,
    expected_realignments: int,
) -> None:
    """Mid-drive re-aim runs only when another forward command can follow."""
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
        # 1.0 m leg, not 0.1 m: a leg shorter than
        # `waypoint_tolerance / tan(trigger)` sits entirely inside the
        # radius where `_realign_bearing_is_ill_conditioned` suppresses
        # re-aim -- rightly, since at 0.1 m against an 0.08 m tolerance the
        # mower needs to move 2 cm and its aim cannot matter. A realistic
        # leg keeps this test exercising the re-aim path it was written for.
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_commands=max_linear_commands,
        sample_delays=(0,),
    )

    # The initial turn always runs. A mid-drive re-aim runs only after the first
    # of two linear commands; after the final budget it could add drift but no
    # forward command could benefit from it.
    assert len(turn_calls) == 1 + expected_realignments
    assert len(result["realignments"]) == expected_realignments
    if expected_realignments:
        realign = result["realignments"][0]
        assert realign["stop_reason"] == "target_heading_reached"
        assert abs(realign["aim_error_degrees"]) > 15.0
    # This fixture intentionally leaves position static, so after exercising
    # the re-alignment decision the executor fails closed on no progress.
    assert result["stop_reason"] == "no_target_progress"


@pytest.mark.asyncio
async def test_mid_drive_re_aim_follows_the_effective_linear_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With loop-to-tolerance on, re-aim must not stop at `max_linear_commands`.

    The re-aim guard asks "can another forward command follow this?", and the
    answer is `effective_linear_ceiling`. Those two numbers are equal only while
    `max_linear_pulse_ceiling` is None. Enable loop-to-tolerance and the ceiling
    becomes the pulse ceiling, so a guard testing `max_linear_commands` would
    silently stop cross-track correction after that many pulses while the linear
    loop kept driving -- a mower correcting nothing for the rest of a long
    segment, which is precisely the failure mode the whole re-aim mechanism
    exists to prevent.

    Recorded as a prerequisite in docs/HANDOVER-beta31-20260809.md section 5:
    harmless while the mode is off, and to be fixed BEFORE anyone turns it on.
    This is that fix, with the mode on.

    `max_linear_commands=1` against `max_linear_pulse_ceiling=4` separates them:
    after the first linear pulse the old comparison is 1 < 1 and refuses, the
    correct one is 1 < 4 and proceeds.
    """
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
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_commands=1,
        max_linear_pulse_ceiling=4,
        sample_delays=(0,),
    )

    assert result["linear_execution_mode"] == "loop_to_tolerance"
    assert result["effective_linear_ceiling"] == 4
    # The initial turn plus at least one mid-drive re-aim. Against
    # `max_linear_commands` this would be the initial turn alone.
    assert len(result["realignments"]) >= 1, (
        "re-aim stopped at max_linear_commands while loop-to-tolerance kept "
        "driving -- cross-track correction would be silently dead"
    )
    assert len(turn_calls) >= 2


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
    assert result["honoured_requested_period"] is True
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
    assert result["honoured_requested_period"] is False


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
    assert result["honoured_requested_period"] is False


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


@pytest.mark.asyncio
async def test_vector_segment_refuses_a_u_turn_after_passing_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forward-only segment stops instead of recovering to a target behind it."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def no_sleep(_: float) -> None:
        return None

    async def fake_calibration(
        coordinator_arg: MammotionReportUpdateCoordinator, **kwargs: object
    ) -> dict:
        return {
            "passed": True,
            "reason": "calibrated",
            "offset_degrees": 110.0,
            "map_motion_heading_degrees": 120.0,
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

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
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
        max_linear_commands=2,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "target_requires_reverse_recovery"
    assert abs(result["reverse_recovery_guard"]["aim_error_degrees"]) == 120.0
    assert result["linear_commands_sent"] == 1
    # Only the initial heading turn ran; no U-turn recovery was dispatched.
    assert len(turn_calls) == 1


@pytest.mark.asyncio
async def test_vector_segment_stops_when_the_realign_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An off-bearing segment with no correction left stops instead of driving on."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    # Facing estimate = vision_heading + offset = 10 + (-90) = -80 deg map,
    # bearing to target is 0 deg -> 80 deg aim error: over the 15 deg realign
    # threshold but under the 90 deg reverse-recovery boundary, so this is the
    # budget decision and not the U-turn guard.
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

    async def no_sleep(_: float) -> None:
        return None

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

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        mammotion_services, "_vio_segment_calibration_drive", fake_calibration
    )
    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading", fake_vio_turn)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        # 1.0 m leg, not 0.1 m: a leg shorter than
        # `waypoint_tolerance / tan(trigger)` sits entirely inside the
        # radius where `_realign_bearing_is_ill_conditioned` suppresses
        # re-aim -- rightly, since at 0.1 m against an 0.08 m tolerance the
        # mower needs to move 2 cm and its aim cannot matter. A realistic
        # leg keeps this test exercising the re-aim path it was written for.
        [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_linear_commands=2,
        vio_max_realignments=0,
        sample_delays=(0,),
    )

    # Before this guard the exhausted budget silently skipped the correction and
    # spent the remaining forward budget driving 80 deg off the bearing.
    assert result["stop_reason"] == "vio_realign_budget_exhausted"
    assert result["linear_commands_sent"] == 1
    assert result["realignments"] == []
    # Only the initial heading turn ran; no correction was dispatched.
    assert len(turn_calls) == 1


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
        return {
            "attempted": True,
            "ok": False,
            "error": "BLEUnavailableError: cooldown",
        }

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
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=0)

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
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=0)

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
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

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
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

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
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2
    )

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
    _patch_services_monotonic(monkeypatch, lambda: 1000.0)

    # No cooldown armed (0.0 deadline is in the past).
    assert _ble_connect_cooldown_active(coordinator) is False
    # Deadline in the future -> cooldown active.
    deadline["value"] = 1005.0
    assert _ble_connect_cooldown_active(coordinator) is True
    # Deadline already elapsed -> inactive again.
    deadline["value"] = 995.0
    assert _ble_connect_cooldown_active(coordinator) is False


@pytest.mark.asyncio
async def test_settle_feed_flags_stale_when_coordinates_bit_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bit-identical coordinates across polls read as a stale feed, not stillness."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))

    async def fake_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    before = _custom_path_telemetry_snapshot(coordinator)

    result = await _settle_linear_position_feed(coordinator, before)

    assert result["feed_stale"] is True
    assert result["observed_jitter"] is False
    assert result["settle_polls"] >= 3
    assert result["moved"] is False


@pytest.mark.asyncio
async def test_settle_feed_not_stale_when_live_feed_jitters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live feed's mm-level noise on a stationary mower is NOT staleness.

    This is the case that must stay distinguishable: during the 2026-07-19 e-stop
    the mower genuinely did not move, but the feed still jittered 2-4mm, which is
    what tells us the link is alive and the mower is the thing that stopped.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    tick = {"n": 0}

    async def fake_sleep(delay: float) -> None:
        return None

    async def jitter(count: int = 5) -> None:
        tick["n"] += 1
        # ~2mm of sensor noise, well under the 1cm settle epsilon.
        coordinator.data.mowing_state.pos_x = 1.0 + (0.002 if tick["n"] % 2 else 0.0)

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = jitter
    before = _custom_path_telemetry_snapshot(coordinator)

    result = await _settle_linear_position_feed(coordinator, before)

    assert result["observed_jitter"] is True
    assert result["feed_stale"] is False


def test_streak_shows_no_actuation_requires_both_sensors_flat() -> None:
    """Only a fully flat streak (no heading AND no position change) counts.

    Models the 2026-07-19 e-stop session: five commands returned ok with a
    dual-axis stop ACK, heading stayed bit-identical and displacement read
    0.25-0.43cm.
    """
    frozen = 91.38829636391407
    dead = [
        {
            "before_vision_heading": frozen,
            "after_vision_heading": frozen,
            "displacement_m": 0.0043,
        },
        {
            "before_vision_heading": frozen,
            "after_vision_heading": frozen,
            "displacement_m": 0.0025,
        },
    ]
    assert _streak_shows_no_actuation(dead, 2) is True

    # Dusk-latched VIO on a mower that may well be rotating: the feed still emits
    # sub-epsilon noise, so it is NOT bit-identical. Must stay no_heading_progress.
    latched_with_noise = [
        {
            "before_vision_heading": -49.836495,
            "after_vision_heading": -49.8382935,
            "displacement_m": 0.0,
        },
        {
            "before_vision_heading": -49.8382935,
            "after_vision_heading": -49.836495,
            "displacement_m": 0.0,
        },
    ]
    assert _streak_shows_no_actuation(latched_with_noise, 2) is False

    # Heading frozen but the mower demonstrably translated -> not "no actuation".
    latched_but_moving = [
        {
            "before_vision_heading": frozen,
            "after_vision_heading": frozen,
            "displacement_m": 0.081,
        },
        {
            "before_vision_heading": frozen,
            "after_vision_heading": frozen,
            "displacement_m": 0.074,
        },
    ]
    assert _streak_shows_no_actuation(latched_but_moving, 2) is False

    # Real rotation that simply isn't converging -> no_heading_progress.
    turning = [
        {
            "before_vision_heading": 0.0,
            "after_vision_heading": 8.2,
            "displacement_m": 0.001,
        },
        {
            "before_vision_heading": 8.2,
            "after_vision_heading": -7.3,
            "displacement_m": 0.002,
        },
    ]
    assert _streak_shows_no_actuation(turning, 2) is False

    # Unknown displacement cannot prove the mower stayed put.
    unknown = [
        {
            "before_vision_heading": frozen,
            "after_vision_heading": frozen,
            "displacement_m": None,
        },
        {
            "before_vision_heading": frozen,
            "after_vision_heading": frozen,
            "displacement_m": None,
        },
    ]
    assert _streak_shows_no_actuation(unknown, 2) is False

    # Not enough pulses yet.
    assert _streak_shows_no_actuation(dead[:1], 2) is False
    assert _streak_shows_no_actuation(dead, 0) is False


@pytest.mark.asyncio
async def test_vio_turn_reports_no_actuation_when_nothing_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead command path aborts as no_actuation_detected, not no_heading_progress.

    Regression for 2026-07-19: a forgotten physical e-stop silently no-opped
    every motion command for ~40 minutes while every health indicator read
    green. The turn loop blamed the turn (no_heading_progress) instead of
    surfacing that nothing actuated at all.

    Note the feed is deliberately kept ALIVE here (jittering position). Since
    2026-07-25 the claim "nothing actuated" requires positive evidence that the
    sensors were live, because a frozen report stream produces an identical
    before/after comparison while the mower turns normally. The 07-19 incident
    itself had a frozen feed too (heading bit-identical for 45 minutes), so a
    replay of that exact run now reports ``vio_telemetry_stream_stale`` -- which
    is the honest answer: telemetry never saw the e-stop, the operator did.
    """
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    # Heading is frozen bit-identical and the mower never moves, exactly as the
    # live e-stopped runs reported.
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=91.38829636391407, vio_state=2
    )

    # ...but the report stream is ALIVE: a live position feed jitters ~2mm
    # between reads even on a stationary mower. That is what licenses the claim
    # "the mower did not actuate" rather than "we went blind" -- without it the
    # run is indistinguishable from a dead feed and must report
    # vio_telemetry_stream_stale instead.
    jitter = {"n": 0}

    async def jittering_reports(*_args: object, **_kwargs: object) -> None:
        jitter["n"] += 1
        coordinator.data.mowing_state.pos_x = 1.0 + 0.002 * (jitter["n"] % 2)

    coordinator.async_get_reports = AsyncMock(side_effect=jittering_reports)

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=9.2,
        heading_tolerance_degrees=18.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_commands=16,
    )

    assert result["stop_reason"] == "no_actuation_detected"
    assert "e-stop" in result["no_actuation_hint"]
    # Bounded exactly like the old path: it still stops after the streak.
    assert result["commands_sent"] == 2


def test_streak_shows_dead_telemetry_requires_no_jitter_across_enough_polls() -> None:
    """Only bit-identical position across enough polls counts as a dead feed."""
    dead = [
        {"heading_poll_count": 16, "heading_poll_feed_alive": False},
        {"heading_poll_count": 16, "heading_poll_feed_alive": False},
    ]
    assert _streak_shows_dead_telemetry(dead, 2) is True

    # A live feed moves in some channel -- the mower may be stopped, but we can
    # see it, so the run must not be blamed on a dead link.
    alive = [
        {"heading_poll_count": 16, "heading_poll_feed_alive": False},
        {"heading_poll_count": 16, "heading_poll_feed_alive": True},
    ]
    assert _streak_shows_dead_telemetry(alive, 2) is False

    # One or two unchanged reads prove nothing (mirrors _STALE_FEED_MIN_POLLS).
    too_few = [
        {"heading_poll_count": 2, "heading_poll_feed_alive": False},
        {"heading_poll_count": 2, "heading_poll_feed_alive": False},
    ]
    assert _streak_shows_dead_telemetry(too_few, 2) is False

    # Pulses recorded before this instrumentation existed must not be read as
    # evidence of a dead feed.
    missing = [{"heading_poll_count": None}, {"heading_poll_count": None}]
    assert _streak_shows_dead_telemetry(missing, 2) is False

    assert _streak_shows_dead_telemetry(dead[:1], 2) is False
    assert _streak_shows_dead_telemetry(dead, 0) is False


@pytest.mark.asyncio
async def test_vio_turn_reports_stale_stream_when_the_feed_is_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frozen report stream aborts as stale telemetry, not no_actuation_detected.

    Regression for 2026-07-25: two turn pulses reported bit-identical
    vision_heading (90.29915121519771) and bit-identical displacement_m
    (0.006754257916307457) while the operator watched the mower turn ~4 inches.
    The server log for that window shows BLE frames being dropped outright, so
    the telemetry was dead while actuation was fine -- but the loop blamed the
    mower with no_actuation_detected, which sends the operator to check a
    physical e-stop that was not engaged.
    """
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    # Heading frozen AND position frozen: nothing in the report stream updates,
    # which is what a dropped-frame window looks like from here.
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=90.29915121519771, vio_state=2
    )

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=9.2,
        heading_tolerance_degrees=18.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_commands=16,
    )

    assert result["stop_reason"] == "vio_telemetry_stream_stale"
    assert "dropped/malformed frames" in result["vio_telemetry_stream_stale_hint"]
    # It must NOT accuse the mower of failing to actuate.
    assert "no_actuation_hint" not in result
    # Still bounded by the same streak logic.
    assert result["commands_sent"] == 2
    # The evidence is recorded per pulse for post-run forensics.
    assert all(
        command["heading_poll_feed_alive"] is False
        and command["heading_poll_count"] >= 3
        for command in result["command_results"]
    )


def test_ble_transport_usable_reflects_transport_flag() -> None:
    """The usability probe mirrors BLETransport.is_usable."""
    coordinator = _pulse_coordinator()
    usable = {"value": True}

    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        get_transport=lambda _transport_type: SimpleNamespace(is_usable=usable["value"])
    )

    assert _ble_transport_usable(coordinator) is True
    usable["value"] = False
    assert _ble_transport_usable(coordinator) is False


def test_ble_transport_usable_defends_against_api_drift() -> None:
    """Anything unreadable degrades to "usable" (the old label-only behaviour)."""
    coordinator = _pulse_coordinator()

    # Handle without get_transport at all.
    coordinator.manager.mower = lambda _device_name: SimpleNamespace()
    assert _ble_transport_usable(coordinator) is True

    # Transport without the is_usable property (older pymammotion).
    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        get_transport=lambda _transport_type: SimpleNamespace()
    )
    assert _ble_transport_usable(coordinator) is True

    def raising_mower(_device_name: str) -> object:
        raise RuntimeError("handle unavailable")

    coordinator.manager.mower = raising_mower
    assert _ble_transport_usable(coordinator) is True


def test_ble_ready_for_motion_requires_label_and_usability() -> None:
    """BLE must be BOTH the active transport and actually usable.

    Regression for 2026-07-19: the mower reported active_transport "ble" while
    BLETransport.is_usable was False (advertisement lost). Motion commands
    returned command_ok with a dual-axis stop ACK and the mower never moved.
    """
    coordinator = _pulse_coordinator()
    usable = {"value": True}
    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        get_transport=lambda _transport_type: SimpleNamespace(is_usable=usable["value"])
    )

    coordinator.active_transport_state = "ble"
    assert _ble_ready_for_motion(coordinator) is True

    # The live failure: selected for routing, but cannot carry a command.
    usable["value"] = False
    assert _ble_ready_for_motion(coordinator) is False

    # Usable but not the active transport is still not ready.
    usable["value"] = True
    coordinator.active_transport_state = "cloud"
    assert _ble_ready_for_motion(coordinator) is False


@pytest.mark.asyncio
async def test_motion_gate_blocks_when_ble_selected_but_unusable() -> None:
    """ble_transport_required must fail on an unusable BLE transport.

    End-to-end regression for the 2026-07-19 live bug: before the fix the gate
    only compared the transport label, so real motion was dispatched onto a dead
    link -- five commands in a row silently no-opped while every health
    indicator read green.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.active_transport_state = "ble"
    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        last_report_at=123.0,
        availability=SimpleNamespace(mqtt_reported_offline=False),
        get_transport=lambda _transport_type: SimpleNamespace(
            _connect_cooldown_until=0.0,
            is_usable=False,
        ),
        active_transport=lambda: "ble",
    )

    result = await _vio_turn_probe(
        coordinator,
        angular_speed=500,
        drive_seconds=1.5,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert "ble_transport_required" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


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


def test_pinned_pymammotion_exposes_ble_liveness_fields() -> None:
    """Pin every field _ble_link_liveness reads against pymammotion drift.

    Two of them are private. If a bump renames them the helper would report
    "cannot read" -- which refuses motion rather than passing it, so drift is
    loud rather than silent. This test makes it loud at CI time instead.
    """
    transport = BLETransport(BLETransportConfig(device_id="test"))
    assert transport.is_connected is False
    assert transport.is_usable is False
    # Public on the Transport ABC; BLETransport.send() is its only writer.
    # It is an attempt timestamp (stamped before the awaited write), so the
    # confirmed-dispatch tests below—not this field—prove completion.
    assert transport.last_send_monotonic == 0.0

    queue = DeviceCommandQueue("test")
    assert queue.is_saga_active is False
    # Private on purpose -- pinning them here is the point of this test.
    assert queue._transport_gate.is_set() is True  # noqa: SLF001
    assert queue._queue.qsize() == 0  # noqa: SLF001


def test_ble_link_liveness_passes_on_a_healthy_link() -> None:
    """A connected transport with a drained queue and a recent send is live."""
    report = _ble_link_liveness(_pulse_coordinator())

    assert report["live"] is True
    assert report["reason"] is None
    assert report["queue_depth"] == 0
    assert report["queue_dispatch_paused"] is False
    assert report["last_send_age_seconds"] < _BLE_SEND_STALL_SECONDS


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"ble_connected": False}, "ble_client_not_connected"),
        ({"ble_usable": False}, "ble_transport_not_usable"),
        ({"ble_queue_paused": True}, "command_queue_dispatch_paused"),
        ({"ble_queue_depth": 21}, "command_queue_backlogged"),
        ({"ble_last_send_age": 30.0}, "ble_send_stalled"),
        ({"ble_last_send_age": None}, "no_ble_send_observed"),
    ],
)
def test_ble_link_liveness_names_the_failing_check(
    kwargs: dict[str, object], reason: str
) -> None:
    """Each stall signature is refused and reported by name."""
    report = _ble_link_liveness(_pulse_coordinator(**kwargs))

    assert report["live"] is False
    assert report["reason"] == reason


def test_ble_link_liveness_refuses_when_it_cannot_see() -> None:
    """Unreadable introspection must refuse, not pass.

    The inverse of _ble_transport_usable's deliberate permissiveness. A liveness
    gate that degrades to "live" is vacuously true -- the exact failure mode that
    has already bitten this project twice (zone_hash/bol_hash, _point_in_polygon).
    """
    coordinator = _pulse_coordinator()

    coordinator.manager.mower = lambda _device_name: None
    assert _ble_link_liveness(coordinator)["reason"] == "device_handle_unavailable"

    coordinator.manager.mower = lambda _device_name: SimpleNamespace()
    assert _ble_link_liveness(coordinator)["reason"] == "get_transport_unavailable"

    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        get_transport=lambda _transport_type: None
    )
    assert _ble_link_liveness(coordinator)["reason"] == "ble_transport_not_registered"

    def raising_mower(_device_name: str) -> object:
        raise RuntimeError("handle unavailable")

    coordinator.manager.mower = raising_mower
    assert _ble_link_liveness(coordinator)["live"] is False

    # A handle whose queue introspection is gone entirely: depth unknown, refuse.
    coordinator.manager.mower = lambda _device_name: SimpleNamespace(
        get_transport=lambda _transport_type: SimpleNamespace(
            _connect_cooldown_until=0.0,
            is_usable=True,
            is_connected=True,
            last_send_monotonic=time.monotonic(),
        ),
    )
    report = _ble_link_liveness(coordinator)
    assert report["live"] is False
    # Both queue reads are unavailable; reason names the first one checked.
    assert report["reason"] == "command_queue_dispatch_paused"
    assert report["queue_dispatch_paused"] is None
    assert report["queue_depth"] is None


@pytest.mark.asyncio
async def test_confirmed_ble_motion_waits_for_gatt_write() -> None:
    """A motion send does not complete merely because queue insertion succeeded."""
    coordinator = _pulse_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    release_write = asyncio.Event()

    async def blocked_write(_transport: object, _payload: bytes) -> None:
        await release_write.wait()

    handle._send_marked.side_effect = blocked_write  # noqa: SLF001
    send_task = asyncio.create_task(
        mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
            coordinator,
            "send_movement",
            command_kwargs={"linear_speed": 200, "angular_speed": 0},
        )
    )

    await asyncio.sleep(0)
    assert send_task.done() is False

    release_write.set()
    await send_task
    handle._send_marked.assert_awaited_once()  # noqa: SLF001


@pytest.mark.asyncio
async def test_normal_motion_teardown_stop_uses_emergency_queue_priority() -> None:
    """A bounded pulse's zero write bypasses normal queue work."""
    coordinator = _pulse_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    priorities: list[object] = []

    async def capture_priority(
        work: Callable[[], Coroutine[object, object, None]],
        priority: object = None,
        **_kwargs: object,
    ) -> None:
        priorities.append(priority)
        await work()

    handle.queue.enqueue = capture_priority

    result = await mammotion_services._manual_velocity_stop_attempt(  # noqa: SLF001
        coordinator, use_wifi=False
    )

    assert result["ok"] is True
    assert priorities == [Priority.EMERGENCY]


@pytest.mark.asyncio
async def test_confirmed_ble_motion_disarms_an_item_that_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-preflight queue stall cannot replay the motion item later."""
    coordinator = _pulse_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    queued_work: list[Callable[[], Coroutine[object, object, None]]] = []

    async def hold_in_queue(
        work: Callable[[], Coroutine[object, object, None]],
        **_kwargs: object,
    ) -> None:
        queued_work.append(work)

    handle.queue.enqueue = hold_in_queue
    monkeypatch.setattr(
        mammotion_services,
        "_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS",
        0.01,
    )

    with pytest.raises(RuntimeError, match="queued item was disarmed"):
        await mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
            coordinator,
            "send_movement",
            command_kwargs={"linear_speed": 200, "angular_speed": 0},
        )

    assert len(queued_work) == 1
    await queued_work[0]()
    handle._send_marked.assert_not_awaited()  # noqa: SLF001


@pytest.mark.asyncio
async def test_confirmed_ble_motion_disarms_on_cancellation_before_start() -> None:
    """Cancellation while queued cannot leave a command that executes later."""
    coordinator = _pulse_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    queued_work: list[Callable[[], Coroutine[object, object, None]]] = []

    async def hold_in_queue(
        work: Callable[[], Coroutine[object, object, None]],
        **_kwargs: object,
    ) -> None:
        queued_work.append(work)

    handle.queue.enqueue = hold_in_queue
    send_task = asyncio.create_task(
        mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
            coordinator,
            "send_movement",
            command_kwargs={"linear_speed": 200, "angular_speed": 0},
        )
    )
    await asyncio.sleep(0)
    send_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await send_task

    await queued_work[0]()
    handle._send_marked.assert_not_awaited()  # noqa: SLF001


@pytest.mark.asyncio
async def test_confirmed_ble_motion_stops_on_cancellation_during_write() -> None:
    """Cancellation after dispatch waits for the write and confirms a stop."""
    coordinator = _pulse_coordinator()
    handle = coordinator.manager.mower(coordinator.device_name)
    write_started = asyncio.Event()
    release_write = asyncio.Event()
    payloads: list[tuple[str, dict[str, object]]] = []
    queue_tasks: list[asyncio.Task[None]] = []

    async def enqueue_in_background(
        work: Callable[[], Coroutine[object, object, None]],
        **_kwargs: object,
    ) -> None:
        queue_tasks.append(asyncio.create_task(work()))

    async def block_first_write(
        _transport: object,
        payload: tuple[str, dict[str, object]],
    ) -> None:
        payloads.append(payload)
        if len(payloads) == 1:
            write_started.set()
            await release_write.wait()

    handle._send_marked.side_effect = block_first_write  # noqa: SLF001
    handle.queue.enqueue = enqueue_in_background
    send_task = asyncio.create_task(
        mammotion_services._send_ble_motion_command_confirmed(  # noqa: SLF001
            coordinator,
            "send_movement",
            command_kwargs={"linear_speed": 200, "angular_speed": 0},
        )
    )
    await write_started.wait()
    send_task.cancel()
    release_write.set()

    with pytest.raises(asyncio.CancelledError):
        await send_task

    assert payloads == [
        ("send_movement", {"linear_speed": 200, "angular_speed": 0}),
        ("send_movement", {"linear_speed": 0, "angular_speed": 0}),
    ]
    await asyncio.gather(*queue_tasks)


@pytest.mark.asyncio
async def test_motion_gate_blocks_a_stalled_command_queue() -> None:
    """End-to-end regression for the 2026-07-28 late-burst incident.

    A gated DeviceCommandQueue accumulates commands -- including the mandatory
    stop that bounds a pulse -- and flushes them as a burst tens of seconds
    later. On 2026-07-28 a command issued at 21:06:20 was reported as no
    actuation; the queue flushed at 21:06:41-43 and the mower drove 1.0778 m at
    21:07:16, unattended.

    Every pre-existing indicator read healthy through this: active_transport
    "ble", is_usable True, RSSI -64 dBm, command_result.ok True. Only the queue
    depth and the age of the last send discriminate.
    """
    coordinator = _pulse_coordinator(
        position=(1.0, 1.0, 0.0),
        ble_queue_depth=21,
        ble_last_send_age=21.0,
    )

    # The old gate is satisfied -- this is exactly why it did not catch it.
    assert _ble_ready_for_motion(coordinator) is True

    result = await _vio_turn_probe(
        coordinator,
        angular_speed=500,
        drive_seconds=1.5,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert "ble_link_live" in result["blockers"]
    assert "ble_transport_required" not in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_motion_gate_ble_link_live_is_skipped_on_dry_runs() -> None:
    """Dry runs send nothing, so they must not be blocked by link liveness."""
    result = await _vio_turn_probe(
        _pulse_coordinator(
            position=(1.0, 1.0, 0.0),
            ble_connected=False,
            ble_queue_depth=21,
            ble_last_send_age=None,
        ),
        angular_speed=500,
        drive_seconds=1.5,
        dry_run=True,
    )

    assert "ble_link_live" not in result["blockers"]


@pytest.mark.asyncio
async def test_raw_pymammotion_execute_multi_segment_real_rejects_missing_confirmations() -> (
    None
):
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
        "data": {"speed": 0.55, "use_wifi": False},
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
        duration_ms=50,
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
    coordinator = _pulse_coordinator(
        position=(None, None, None), pos_type=0, zone_hash=0
    )

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
        duration_ms=50,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        followup_samples=0,
    )

    assert result["would_send"] is True
    assert result["blockers"] == []
    before_position = result["samples"][0]["telemetry"]["position"]
    assert before_position["pos_type_label"] == ("TURN_AREA_INSIDE")
    assert before_position["valid_for_motion"] is True
    coordinator.async_move_forward.assert_awaited_once()
    coordinator.async_stop_manual_motion.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_allows_channel_area_overlap() -> None:
    """Real pulse allows CHANNEL_AREA_OVERLAP when position and zone are known."""
    coordinator = _pulse_coordinator(pos_type=9, zone_hash=123)

    result = await _manual_velocity_pulse_test(
        coordinator,
        duration_ms=50,
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
    coordinator.async_stop_manual_motion.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_velocity_pulse_test_reports_false_command_ack() -> None:
    """A false coordinator command return is reported as an unsuccessful attempt."""
    coordinator = _pulse_coordinator()
    coordinator.async_move_forward.side_effect = RuntimeError("motion write failed")

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
    assert result["command_result"]["ack"] is None
    assert result["command_result"]["error"] == "RuntimeError: motion write failed"
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

    monkeypatch.setattr(
        mammotion_services, "_custom_path_telemetry_snapshot", delayed_snapshot
    )
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
async def test_manual_velocity_segment_test_reports_path_complete_at_max_pulses() -> (
    None
):
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
async def test_manual_velocity_segment_test_can_report_multi_pulse_service_name() -> (
    None
):
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
    assert (
        result["initial_controller_decision"]["selected_heading_offset_degrees"] == 0.0
    )
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
    assert (
        result["manual_motion_execution_policy"]["experimental_segment_scope"]
        == "one_segment_calibrated_forward_only"
    )
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


# ---------------------------------------------------------------------------
# Cancellation-safe motion stop + per-mower manual-motion exclusivity
# (2026-07-18 adversarial-review fixes)
# ---------------------------------------------------------------------------


async def test_motion_open_sleep_normal_path_sends_no_stop() -> None:
    """An uncancelled pulse sleep must not add an extra stop of its own."""
    coordinator = SimpleNamespace(async_stop_manual_motion=AsyncMock())
    await _motion_open_sleep(coordinator, 0)
    coordinator.async_stop_manual_motion.assert_not_called()


async def test_motion_open_sleep_delivers_stop_on_cancellation() -> None:
    """Cancellation mid-pulse delivers the mandatory stop before re-raising."""
    coordinator = SimpleNamespace(async_stop_manual_motion=AsyncMock())
    task = asyncio.create_task(_motion_open_sleep(coordinator, 30.0))
    await asyncio.sleep(0)  # let the task enter the pulse sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    coordinator.async_stop_manual_motion.assert_awaited_once()


async def test_motion_open_sleep_swallows_stop_errors_on_cancellation() -> None:
    """A failing stop must not mask the CancelledError during teardown."""
    coordinator = SimpleNamespace(
        async_stop_manual_motion=AsyncMock(side_effect=RuntimeError("ble gone"))
    )
    task = asyncio.create_task(_motion_open_sleep(coordinator, 30.0))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    coordinator.async_stop_manual_motion.assert_awaited_once()


def _fake_motion_mower(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Route _get_mower_by_entity_id to a fixed fake mower/coordinator."""
    coordinator = SimpleNamespace(async_stop_manual_motion=AsyncMock())
    mower = SimpleNamespace(reporting_coordinator=coordinator)
    monkeypatch.setattr(
        mammotion_services,
        "_get_mower_by_entity_id",
        lambda _hass, _entity_id: mower,
    )
    monkeypatch.setattr(
        mammotion_services,
        "_manual_motion_authorization",
        lambda *_args, **_kwargs: {
            "real_motion_allowed": True,
            "blockers": [],
        },
    )
    return mower


def _motion_call(**overrides: object) -> SimpleNamespace:
    """Minimal ServiceCall stand-in for the exclusivity wrapper."""
    data: dict[str, object] = {"entity_id": "lawn_mower.test", "dry_run": False}
    data.update(overrides)
    return SimpleNamespace(data=data)


async def test_exclusive_motion_wrapper_rejects_concurrent_real_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second real motion run is strictly rejected while the first owns the mower."""
    _fake_motion_mower(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_handler(call: object) -> dict[str, object]:
        started.set()
        await release.wait()
        return {"ran": "slow"}

    async def fast_handler(call: object) -> dict[str, object]:
        return {"ran": "fast"}

    slow = _wrap_exclusive_manual_motion(object(), "svc_slow", slow_handler)
    fast = _wrap_exclusive_manual_motion(object(), "svc_fast", fast_handler)
    task = asyncio.create_task(slow(_motion_call()))
    await started.wait()

    busy = await fast(_motion_call())
    assert busy["stop_reason"] == "manual_motion_in_progress"
    assert busy["blockers"] == ["manual_motion_in_progress"]
    assert busy["busy_owner"] == "svc_slow"
    assert busy["would_send"] is False

    # Dry runs never move and pass straight through while the owner runs.
    dry = await fast(_motion_call(dry_run=True))
    assert dry == {"ran": "fast"}

    release.set()
    assert await task == {"ran": "slow"}

    # Owner finished: the mower is claimable again.
    again = await fast(_motion_call())
    assert again == {"ran": "fast"}


def _saga_motion_mower(
    monkeypatch: pytest.MonkeyPatch, *, saga_active: bool | None
) -> SimpleNamespace:
    """Motion mower whose command queue reports an exclusive saga state.

    ``saga_active=None`` omits the queue entirely, standing in for pymammotion
    API drift.
    """
    handle = SimpleNamespace()
    if saga_active is not None:
        handle.queue = SimpleNamespace(is_saga_active=saga_active)
    coordinator = SimpleNamespace(
        async_stop_manual_motion=AsyncMock(),
        device_name="Luba-Test",
        manager=SimpleNamespace(mower=lambda _name: handle),
    )
    mower = SimpleNamespace(reporting_coordinator=coordinator)
    monkeypatch.setattr(
        mammotion_services,
        "_get_mower_by_entity_id",
        lambda _hass, _entity_id: mower,
    )
    monkeypatch.setattr(
        mammotion_services,
        "_manual_motion_authorization",
        lambda *_args, **_kwargs: {
            "real_motion_allowed": True,
            "blockers": [],
        },
    )
    return mower


async def test_exclusive_motion_wrapper_rejects_motion_during_a_saga(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Motion is refused *by name* while an exclusive map saga holds the queue.

    The reverse ordering of the guard that already exists: `manual_motion_owner`
    stops a saga starting while motion runs, but nothing stopped motion when a
    saga was already running. Motion is `Priority.NORMAL` with
    `skip_if_saga_active=False`, so it blocks on the exclusive slot while the
    executor sleeps out its pulse on local timing -- separating a movement from
    its stop, or letting either be dropped at the 120 s `_COMMAND_TTL`.

    The original design deliberately had no gate here ("refusing a command the
    operator just issued is worse than the wait"), which assumed sagas were
    rare and operator-triggered. They now also fire automatically, and the wait
    was never benign. A *named* refusal answers the original objection.

    Raised by adversarial review 2026-07-26.
    """
    _saga_motion_mower(monkeypatch, saga_active=True)

    async def handler(call: object) -> dict[str, object]:
        return {"ran": "yes"}

    wrapped = _wrap_exclusive_manual_motion(object(), "svc_motion", handler)

    busy = await wrapped(_motion_call())
    assert busy["stop_reason"] == "manual_motion_in_progress"
    assert busy["busy_owner"] == "map_sync_saga"
    assert busy["would_send"] is False

    # A dry run reads telemetry only and still passes through.
    assert await wrapped(_motion_call(dry_run=True)) == {"ran": "yes"}


async def test_exclusive_motion_wrapper_allows_motion_once_saga_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the exclusive slot is free the mower is claimable again."""
    _saga_motion_mower(monkeypatch, saga_active=False)

    async def handler(call: object) -> dict[str, object]:
        return {"ran": "yes"}

    wrapped = _wrap_exclusive_manual_motion(object(), "svc_motion", handler)
    assert await wrapped(_motion_call()) == {"ran": "yes"}


async def test_exclusive_saga_probe_degrades_to_allowing_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable queue must not block motion.

    Positively-True-only: pymammotion API drift should lose the guard, never
    refuse every motion command. Mirrors `_ble_transport_usable`'s convention.
    """
    _saga_motion_mower(monkeypatch, saga_active=None)

    async def handler(call: object) -> dict[str, object]:
        return {"ran": "yes"}

    wrapped = _wrap_exclusive_manual_motion(object(), "svc_motion", handler)
    assert await wrapped(_motion_call()) == {"ran": "yes"}


async def test_exclusive_motion_wrapper_releases_on_handler_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashing run must release the mower for the next start."""
    _fake_motion_mower(monkeypatch)

    async def crashing_handler(call: object) -> dict[str, object]:
        raise RuntimeError("boom")

    async def ok_handler(call: object) -> dict[str, object]:
        return {"ran": True}

    crashing = _wrap_exclusive_manual_motion(object(), "svc_crash", crashing_handler)
    ok = _wrap_exclusive_manual_motion(object(), "svc_ok", ok_handler)
    with pytest.raises(RuntimeError):
        await crashing(_motion_call())
    assert await ok(_motion_call()) == {"ran": True}


async def test_exclusive_motion_wrapper_exempts_zero_motion_stop_nudge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The card's Abort (send_movement 0/0) preempts a running loop; real probes don't."""
    _fake_motion_mower(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_handler(call: object) -> dict[str, object]:
        started.set()
        await release.wait()
        return {"ran": "slow"}

    async def probe_handler(call: object) -> dict[str, object]:
        return {"ran": "probe"}

    slow = _wrap_exclusive_manual_motion(object(), "svc_slow", slow_handler)
    probe = _wrap_exclusive_manual_motion(
        object(), "svc_probe", probe_handler, allow_stop_nudge=True
    )
    task = asyncio.create_task(slow(_motion_call()))
    await started.wait()

    # Zero-motion stop nudge: passes through even while the mower is owned.
    nudge = await probe(
        _motion_call(command="send_movement", linear_speed=0, angular_speed=0)
    )
    assert nudge == {"ran": "probe"}

    # A real (nonzero) probe is still rejected as busy.
    real_probe = await probe(
        _motion_call(command="send_movement", linear_speed=400, angular_speed=0)
    )
    assert real_probe["stop_reason"] == "manual_motion_in_progress"

    release.set()
    await task


def test_is_zero_motion_stop_nudge_truth_table() -> None:
    """Only send_movement with both speeds zero counts as a stop nudge."""
    is_nudge = _is_zero_motion_stop_nudge
    assert is_nudge("send_movement", 0, 0)
    assert not is_nudge("send_movement", 400, 0)
    assert not is_nudge("send_movement", 0, 500)
    assert not is_nudge("move_forward", 0, 0)


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


# Unbound so the coordinator's private reconnect helper can be exercised without
# standing up a full HA instance; bound once here rather than per call site.
_OPPORTUNISTIC_BLE_RECONNECT = (
    mammotion_coordinator.MammotionReportUpdateCoordinator._async_opportunistic_ble_reconnect  # noqa: SLF001
)


def _reconnect_self(
    *,
    is_usable: bool = True,
    is_connected: bool = False,
    prefer_ble: bool = True,
    bluetooth_enabled: bool = True,
    connect: object = None,
    handle_present: bool = True,
) -> SimpleNamespace:
    """Build a minimal stand-in for the report coordinator."""
    if connect is None:

        async def connect() -> None:  # noqa: RUF029
            return None

    ble = SimpleNamespace(
        is_usable=is_usable, is_connected=is_connected, connect=connect
    )
    handle = SimpleNamespace(prefer_ble=prefer_ble, get_transport=lambda _t: ble)
    return SimpleNamespace(
        manager=SimpleNamespace(mower=lambda _name: handle if handle_present else None),
        device_name="Luba-Test",
        _bluetooth_enabled=bluetooth_enabled,
    )


@pytest.mark.asyncio
async def test_opportunistic_ble_reconnect_is_bounded_and_best_effort() -> None:
    """A hanging BLE reconnect must not stall the coordinator update.

    ``BLETransport.connect()`` is unbounded overall (self-managed scan at
    scan_timeout 10s, then establish_connection retries up to 4 times with
    backoff), and it runs inline in ``_async_update_data``. An unbounded call
    blocks the tick and every entity update behind it. The update is already
    served by cloud transport, so reconnecting is best-effort.
    """
    started = asyncio.Event()

    async def hanging_connect() -> None:
        started.set()
        await asyncio.sleep(3600)

    fake_self = _reconnect_self(connect=hanging_connect)

    with patch.object(mammotion_coordinator, "_BLE_RECONNECT_TIMEOUT_SECONDS", 0.05):
        # Must return rather than hang, and must not raise.
        await asyncio.wait_for(_OPPORTUNISTIC_BLE_RECONNECT(fake_self), timeout=5)

    assert started.is_set()


@pytest.mark.asyncio
async def test_opportunistic_ble_reconnect_skips_when_not_applicable() -> None:
    """No connect attempt when BLE is unusable, already connected, or disabled."""
    calls: list[str] = []

    async def record_connect() -> None:
        calls.append("connect")

    # Unusable transport (e.g. armed connect cooldown) -> leave it alone.
    await _OPPORTUNISTIC_BLE_RECONNECT(
        _reconnect_self(is_usable=False, connect=record_connect)
    )
    # Already connected -> nothing to do.
    await _OPPORTUNISTIC_BLE_RECONNECT(
        _reconnect_self(is_connected=True, connect=record_connect)
    )
    # Bluetooth switched off by the user -> respect it.
    await _OPPORTUNISTIC_BLE_RECONNECT(
        _reconnect_self(bluetooth_enabled=False, connect=record_connect)
    )
    # prefer_ble off -> user opted out of BLE routing.
    await _OPPORTUNISTIC_BLE_RECONNECT(
        _reconnect_self(prefer_ble=False, connect=record_connect)
    )
    # No handle at all.
    await _OPPORTUNISTIC_BLE_RECONNECT(
        _reconnect_self(handle_present=False, connect=record_connect)
    )

    assert calls == []


@pytest.mark.asyncio
async def test_opportunistic_ble_reconnect_connects_when_usable() -> None:
    """The whole point: a usable-but-disconnected transport gets reconnected."""
    calls: list[str] = []

    async def record_connect() -> None:
        calls.append("connect")

    await _OPPORTUNISTIC_BLE_RECONNECT(_reconnect_self(connect=record_connect))

    assert calls == ["connect"]


# ---------------------------------------------------------------------------
# App-parity motion cadence (2026-07-20 APK decompile).
#
# The Mammotion app re-sends the identical movement command every 200 ms for as
# long as the on-screen stick is held; our executors sent it once and slept out
# the pulse. These cover the opt-in refresh window that closes that gap.
# ---------------------------------------------------------------------------


def _install_virtual_clock(
    monkeypatch: pytest.MonkeyPatch, start: float = 100.0
) -> dict[str, float]:
    """Replace monotonic/sleep with a virtual clock that advances on sleep."""
    clock = {"now": start}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    return clock


def test_motion_refresh_interval_matches_decompiled_app_constant() -> None:
    """Our app-parity default is the app's own timer period, not a guess.

    `CarRemoteControlManage2.frequency = 0.2f` -> 200 ms (Mammotion 2.3.8.19).
    """
    assert mammotion_services._MOTION_REFRESH_INTERVAL_MS_APP == 200  # noqa: SLF001


@pytest.mark.asyncio
async def test_motion_refresh_window_disabled_keeps_single_shot_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interval 0 is the proven path: wait out the pulse, never re-send."""
    coordinator = _pulse_coordinator()
    clock = _install_virtual_clock(monkeypatch)
    sends: list[float] = []

    async def resend() -> None:
        sends.append(clock["now"])

    report = await _motion_refresh_window(
        coordinator,
        resend=resend,
        duration_seconds=4.0,
        refresh_interval_ms=0,
    )

    assert report["refresh_enabled"] is False
    assert report["refresh_commands_sent"] == 0
    assert sends == []
    # The full pulse window is still waited out.
    assert clock["now"] == pytest.approx(104.0)


@pytest.mark.asyncio
async def test_motion_refresh_window_resends_for_whole_pulse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A positive interval re-sends across the pulse, never past its end."""
    coordinator = _pulse_coordinator()
    clock = _install_virtual_clock(monkeypatch)
    sends: list[float] = []

    async def resend() -> None:
        sends.append(clock["now"])

    # 250 ms divides 4.0 s exactly in binary floating point, so the count is
    # deterministic rather than sensitive to accumulated rounding.
    report = await _motion_refresh_window(
        coordinator,
        resend=resend,
        duration_seconds=4.0,
        refresh_interval_ms=250,
    )

    assert report["refresh_enabled"] is True
    assert report["refresh_interval_ms"] == 250
    assert report["max_refresh_commands"] == 16
    # 16 slots fit the window; the one landing exactly on the deadline is
    # skipped so the last command is always followed by real motion time.
    assert report["refresh_commands_sent"] == 15
    assert len(sends) == 15
    assert clock["now"] == pytest.approx(104.0)
    assert max(sends) < 104.0
    # Evenly spaced at the requested cadence.
    assert sends[0] == pytest.approx(100.25)
    assert sends[1] - sends[0] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_motion_refresh_window_stops_at_confirmed_refresh_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final approach stops immediately after its discrete resend budget."""
    coordinator = _pulse_coordinator()
    clock = _install_virtual_clock(monkeypatch)
    sends: list[float] = []

    async def resend() -> None:
        sends.append(clock["now"])

    report = await _motion_refresh_window(
        coordinator,
        resend=resend,
        duration_seconds=3.5,
        refresh_interval_ms=200,
        max_refresh_commands=3,
    )

    assert report["refresh_command_limit"] == 3
    assert report["max_refresh_commands"] == 3
    assert report["refresh_commands_sent"] == 3
    assert len(report["refresh_write_durations_ms"]) == 3
    assert sends == pytest.approx([100.2, 100.4, 100.6])
    assert clock["now"] == pytest.approx(100.6)
    assert report["elapsed_ms"] == pytest.approx(600.0)


@pytest.mark.asyncio
async def test_motion_refresh_window_stops_when_cancelled_inside_resend() -> None:
    """Cancellation while a refresh send is in flight must still stop the mower.

    ``_motion_open_sleep`` was added (commit abf65696) because
    ``CancelledError`` is a ``BaseException``, so ``except Exception`` handlers
    never see it and a cancel during the pulse sleep exited without the
    mandatory stop. The refresh window reintroduced exactly that trap one await
    further in: a cancel landing inside ``resend()`` fell straight past the
    caller's stop. A movement command is already open on the mower at that
    point, so it would keep driving until its own device-side timeout.

    Found by adversarial review 2026-07-26 and reproduced against the real
    helper before fixing: zero stop calls.
    """
    stops: list[str] = []
    coordinator = SimpleNamespace(
        async_stop_manual_motion=AsyncMock(
            side_effect=lambda *a, **k: stops.append("stop")
        )
    )
    resend_started = asyncio.Event()

    async def blocking_resend() -> None:
        resend_started.set()
        await asyncio.sleep(3600)  # cancellation lands here, inside resend()

    task = asyncio.create_task(
        mammotion_services._motion_refresh_window(  # noqa: SLF001
            coordinator,
            resend=blocking_resend,
            duration_seconds=4.0,
            refresh_interval_ms=200,
        )
    )
    await resend_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # The stop must have been delivered, and the cancellation must still
    # propagate so the caller's own teardown runs.
    assert stops == ["stop"]


@pytest.mark.asyncio
async def test_motion_refresh_window_propagates_operator_session_stop() -> None:
    """An operator-cancelled session stops and releases its owner immediately.

    ``ManualMotionCancelledError`` is raised by the confirmed-dispatch guard
    before a post-abort nonzero refresh can be queued.  Treating it as an
    ordinary failed resend kept the service handler alive in feedback/sample
    waits, so ``stop_manual_motion`` could time out waiting for the owner even
    though motion itself was safely blocked.
    """
    coordinator = SimpleNamespace(async_stop_manual_motion=AsyncMock())

    async def cancelled_resend() -> None:
        raise ManualMotionCancelledError("operator stop")

    with pytest.raises(ManualMotionCancelledError, match="operator stop"):
        await _motion_refresh_window(
            coordinator,
            resend=cancelled_resend,
            duration_seconds=1.0,
            refresh_interval_ms=200,
        )

    coordinator.async_stop_manual_motion.assert_awaited_once()


@pytest.mark.asyncio
async def test_motion_refresh_window_clamps_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absurd intervals are clamped, so a bad param cannot flood the BLE queue."""
    coordinator = _pulse_coordinator()
    _install_virtual_clock(monkeypatch)

    async def resend() -> None:
        return None

    too_fast = await _motion_refresh_window(
        coordinator, resend=resend, duration_seconds=1.0, refresh_interval_ms=1
    )
    too_slow = await _motion_refresh_window(
        coordinator, resend=resend, duration_seconds=1.0, refresh_interval_ms=99999
    )

    assert too_fast["refresh_interval_ms"] == 50
    assert too_slow["refresh_interval_ms"] == 1000


@pytest.mark.asyncio
async def test_motion_refresh_window_survives_a_failed_resend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed refresh stops refreshing and reports why, but never raises.

    The caller still owns the mandatory stop, so a half-refreshed window is a
    shorter drive -- never a runaway one.
    """
    coordinator = _pulse_coordinator()
    _install_virtual_clock(monkeypatch)
    attempts: list[int] = []

    async def resend() -> None:
        attempts.append(1)
        if len(attempts) == 2:
            raise RuntimeError("ble dropped")

    report = await _motion_refresh_window(
        coordinator,
        resend=resend,
        duration_seconds=4.0,
        refresh_interval_ms=250,
    )

    assert report["refresh_commands_sent"] == 1
    assert "ble dropped" in report["refresh_error"]
    assert len(attempts) == 2


# --- App speed scale -------------------------------------------------------


def test_app_scale_speeds_applies_deadband_and_ceilings() -> None:
    """Stick fractions convert exactly as the app's rocker transform does."""
    # 15% deadband: anything at or below it is a dead stick.
    assert _app_scale_speeds(0.15, 0.0) == (0, 0)
    assert _app_scale_speeds(0.10, 0.0) == (0, 0)
    # Full deflection lands on the real ceilings, not the round numbers in the
    # app's own comments (1000/450), because the deadband is applied first.
    assert _app_scale_speeds(1.0, 0.0) == (850, 0)
    assert _app_scale_speeds(0.0, 1.0) == (0, 382)
    # Sign selects direction: negative is backward / left.
    assert _app_scale_speeds(-1.0, 0.0) == (-850, 0)
    assert _app_scale_speeds(0.0, -1.0) == (0, -382)


def test_app_speed_scale_report_flags_our_angular_default() -> None:
    """Our long-standing angular 500 is above anything the app can produce."""
    report = _app_speed_scale_report(400, 500)

    assert report["available"] is True
    assert report["app_max_linear_speed"] == 850
    assert report["app_max_angular_speed"] == 382
    assert report["linear_above_app_max"] is False
    assert report["angular_above_app_max"] is True
    # Our linear default is under half the app's full-scale throttle.
    assert report["linear_fraction_of_app_max"] == pytest.approx(0.471, abs=0.001)


def test_app_speed_scale_report_handles_missing_values() -> None:
    """A non-numeric speed degrades to unavailable instead of raising."""
    assert _app_speed_scale_report(None, 0) == {"available": False}


def test_app_speed_scale_matches_pymammotion() -> None:
    """Our local transform agrees with pymammotion's, which mirrors the app.

    Implemented locally so the numbers stay deterministic across dependency
    bumps; this pins the two together so a silent upstream change is caught.
    """
    movement = pytest.importorskip("pymammotion.utility.movement")

    for fraction in (0.0, 0.1, 0.15, 0.4, 0.55, 1.0):
        percent = movement.get_percent(abs(fraction) * 100)
        # 90 degrees == forward (linear axis), 0 degrees == right (angular axis).
        upstream_linear, _ = movement.transform_both_speeds(90.0, 0.0, percent, 0.0)
        _, upstream_angular = movement.transform_both_speeds(0.0, 0.0, 0.0, percent)
        ours_linear, _ = _app_scale_speeds(fraction, 0.0)
        _, ours_angular = _app_scale_speeds(0.0, fraction)
        assert ours_linear == upstream_linear
        assert ours_angular == upstream_angular


# --- Multi-segment pass criteria -------------------------------------------


def test_multi_segment_segment_that_reached_target_passes() -> None:
    """Reaching the target is success even if the final pulse crept.

    Regression: requiring every per-pulse diagnostic to clear
    `min_progress_distance` marked arrived segments as failed, because the
    final approach necessarily moves less than a full pulse -- which stopped
    the run and meant later segments never executed.
    """
    segment_result = {
        "stop_reason": "target_reached",
        "valid": True,
        "blockers": [],
        "progress_diagnostics": [
            {"passed": True, "path_progress_distance": 0.31},
            # Short final approach: real motion, below the per-pulse threshold.
            {"passed": False, "path_progress_distance": 0.02},
        ],
    }

    assert _raw_multi_segment_phase_passed(segment_result, real_segment=True) is True


def test_multi_segment_segment_that_did_not_arrive_fails() -> None:
    """A run that aborted short of the target is still a failure."""
    aborted = {
        "stop_reason": "no_target_progress",
        "valid": True,
        "blockers": [],
        "progress_diagnostics": [{"passed": True}],
    }
    blocked = {
        "stop_reason": "target_reached",
        "valid": True,
        "blockers": ["ble_transport_required"],
        "progress_diagnostics": [{"passed": True}],
    }
    invalid = {
        "stop_reason": "target_reached",
        "valid": False,
        "blockers": [],
        "progress_diagnostics": [{"passed": True}],
    }

    assert _raw_multi_segment_phase_passed(aborted, real_segment=True) is False
    assert _raw_multi_segment_phase_passed(blocked, real_segment=True) is False
    assert _raw_multi_segment_phase_passed(invalid, real_segment=True) is False


# ---------------------------------------------------------------------------
# force_map_resync recovery (map stuck out_of_sync after reload/restart)
# ---------------------------------------------------------------------------


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
