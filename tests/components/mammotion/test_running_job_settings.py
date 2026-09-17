"""Mid-job mowing-setting changes must change only the edited setting.

Operator report 2026-09-17 ~02:34Z: a mow was started from the Mammotion app
(blade ~60 mm, speed 2 ft/s). HA's Working speed was changed 2.0 -> 1.6 ft/s
and the mower's reported blade height then fell 60 -> 35 -> 25 mm, because
``async_modify_plan_if_mowing`` re-sent HA's whole local ``OperationSettings``
(``NavReqCoverPath`` sub_cmd=3 carries every route field) with its stale local
blade height and path spacing.

The app's in-job editor (HomeMapFragment WorkingOptionView.onConfirm) re-sends
the full route too, but seeded from the running job, which it re-reads with
``queryGenerateRouteInformation`` (sub_cmd=2). These tests pin that shape:
query the running job, change one field, send the rest as the job has them —
and send nothing when the job cannot be read.

Every test drives the real number-entity descriptions and the real coordinator
methods; only the transport (``send_command_and_wait`` / ``async_send_command``)
is stubbed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from homeassistant.exceptions import HomeAssistantError
from pymammotion.data.model.device import MowingDevice
from pymammotion.data.model.device_config import OperationSettings
from pymammotion.data.model.device_limits import DeviceLimits, RangeLimit
from pymammotion.data.model.work import CurrentTaskSettings
from pymammotion.proto import LubaMsg, MctlNav, NavReqCoverPath
from pymammotion.transport.base import CommandTimeoutError
from pymammotion.utility.constant import WorkMode

from custom_components.mammotion.coordinator import MammotionReportUpdateCoordinator
from custom_components.mammotion.number import (
    LUBA_WORKING_ENTITIES,
    NUMBER_WORKING_ENTITIES,
    MammotionWorkingNumberEntity,
)

ZONE = 5_512_345_678_901_234_567
PATH_HASH = 777_001
FT_PER_S = 0.3048

# What the app started (the running job, as the mower would report it).
JOB_KNIFE_MM = 60
JOB_SPEED = 0.61
JOB_WIDTH_CM = 30

# HA's local planning values at the time (entity showed 1.0 in / 8.0 in).
LOCAL_BLADE_MM = 25
LOCAL_WIDTH_CM = 20


def _description(key: str) -> Any:
    for description in (*LUBA_WORKING_ENTITIES, *NUMBER_WORKING_ENTITIES):
        if description.key == key:
            return description
    raise AssertionError(key)


def _job_reply(**overrides: Any) -> LubaMsg:
    fields: dict[str, Any] = {
        "pver": 1,
        "sub_cmd": 2,
        "job_id": 42,
        "job_ver": 3,
        "job_mode": 4,
        "edge_mode": 1,
        "knife_height": JOB_KNIFE_MM,
        "speed": JOB_SPEED,
        "channel_width": JOB_WIDTH_CM,
        "ultra_wave": 2,
        "channel_mode": 0,
        "toward": 30,
        "toward_mode": 1,
        "zone_hashs": [ZONE],
        "path_hash": PATH_HASH,
    }
    fields.update(overrides)
    return LubaMsg(nav=MctlNav(bidire_reqconver_path=NavReqCoverPath(**fields)))


def _coordinator(
    *,
    sys_status: int = WorkMode.MODE_WORKING,
    reply: LubaMsg | Exception | None = None,
    stale_work: CurrentTaskSettings | None = None,
    device_name: str = "Luba-VSPLV397",
) -> Any:
    """Build a report coordinator with real methods and a stubbed transport."""
    data = MowingDevice()
    data.report_data.dev.sys_status = sys_status
    data.report_data.work.bp_hash = ZONE
    data.report_data.work.area = 40 << 16
    data.report_data.work.path_hash = PATH_HASH
    # HA's cached `work` is from an EARLIER job on the same zone — the gate
    # in the pre-fix code passes on it, and its values are not the app job's.
    data.work = stale_work or CurrentTaskSettings(
        job_id=41,
        knife_height=50,
        speed=0.4,
        channel_width=25,
        toward=90,
        toward_mode=1,
        zone_hashs=[ZONE],
    )

    coordinator = object.__new__(MammotionReportUpdateCoordinator)
    coordinator.data = data
    coordinator.device_name = device_name
    coordinator.unique_name = device_name
    coordinator._bluetooth_enabled = True  # noqa: SLF001
    coordinator._operation_settings = OperationSettings(  # noqa: SLF001
        blade_height=LOCAL_BLADE_MM, channel_width=LOCAL_WIDTH_CM, speed=0.2
    )
    coordinator._running_job_settings = None  # noqa: SLF001
    send_and_wait = AsyncMock()
    if isinstance(reply, Exception):
        send_and_wait.side_effect = reply
    else:
        send_and_wait.return_value = reply if reply is not None else _job_reply()
    coordinator.manager = SimpleNamespace(send_command_and_wait=send_and_wait)
    coordinator.async_send_command = AsyncMock()
    coordinator.async_send_and_wait = AsyncMock()
    return coordinator


async def _set(coordinator: Any, key: str, value: float) -> None:
    """Exactly what MammotionWorkingNumberEntity.async_set_native_value runs."""
    description = _description(key)
    description.set_fn(coordinator, value)
    if description.set_async_fn is not None:
        await description.set_async_fn(coordinator, value)


def _sent_route(coordinator: Any) -> Any:
    coordinator.async_send_command.assert_awaited_once()
    args, kwargs = coordinator.async_send_command.await_args
    assert args == ("modify_route_information",)
    return kwargs["generate_route_information"]


# --- the incident ---------------------------------------------------------


async def test_speed_change_mid_job_keeps_the_jobs_blade_height_and_spacing() -> None:
    """Replay of 2026-09-17 02:34Z: speed 1.6 ft/s must not move the blade."""
    coordinator = _coordinator()

    await _set(coordinator, "working_speed", 1.6 * FT_PER_S)

    route = _sent_route(coordinator)
    assert route.speed == pytest.approx(1.6 * FT_PER_S)
    assert route.blade_height == JOB_KNIFE_MM
    assert route.channel_width == JOB_WIDTH_CM
    assert route.one_hashs == [ZONE]


async def test_blade_height_change_mid_job_keeps_the_jobs_speed_and_spacing() -> None:
    """The mirror case: only the cut height moves."""
    coordinator = _coordinator()

    await _set(coordinator, "blade_height", 45)

    route = _sent_route(coordinator)
    assert route.blade_height == 45
    assert route.speed == pytest.approx(JOB_SPEED)
    assert route.channel_width == JOB_WIDTH_CM


async def test_the_running_job_is_read_fresh_not_from_the_cached_work() -> None:
    """HA's cached `work` can belong to an earlier job; it must not be trusted."""
    coordinator = _coordinator()

    await _set(coordinator, "working_speed", 0.5)

    coordinator.manager.send_command_and_wait.assert_awaited_once()
    args = coordinator.manager.send_command_and_wait.await_args.args
    assert args == (
        "Luba-VSPLV397",
        "query_generate_route_information",
        "bidire_reqconver_path",
    )
    route = _sent_route(coordinator)
    assert route.blade_height == JOB_KNIFE_MM  # not the stale 50
    assert route.toward == 30  # not the stale 90


async def test_a_mid_job_change_does_not_overwrite_the_other_local_plan_values() -> (
    None
):
    """Seeding uses a copy: HA's next-job plan keeps its own blade and spacing."""
    coordinator = _coordinator()

    await _set(coordinator, "working_speed", 0.5)

    assert coordinator.operation_settings.speed == pytest.approx(0.5)
    assert coordinator.operation_settings.blade_height == LOCAL_BLADE_MM
    assert coordinator.operation_settings.channel_width == LOCAL_WIDTH_CM


async def test_obstacle_detection_change_mid_job_keeps_the_jobs_blade_and_speed() -> (
    None
):
    """The bypass_mode select re-issued the same stale plan; it must not."""
    coordinator = _coordinator()
    coordinator.operation_settings.ultra_wave = 10

    await coordinator.async_apply_working_setting("ultra_wave")

    route = _sent_route(coordinator)
    assert route.ultra_wave == 10
    assert route.blade_height == JOB_KNIFE_MM
    assert route.speed == pytest.approx(JOB_SPEED)
    assert route.channel_width == JOB_WIDTH_CM


# --- fail closed ------------------------------------------------------------


async def test_nothing_is_sent_when_the_running_job_cannot_be_read() -> None:
    """A timed-out query must refuse, never fall back to local values."""
    coordinator = _coordinator(reply=CommandTimeoutError("bidire_reqconver_path", 1))

    with pytest.raises(HomeAssistantError):
        await _set(coordinator, "working_speed", 0.5)

    coordinator.async_send_command.assert_not_awaited()


@pytest.mark.parametrize(
    "overrides",
    [
        {"knife_height": 0},
        {"speed": 0.0},
        {"channel_width": 0},
        {"zone_hashs": []},
    ],
)
async def test_nothing_is_sent_when_the_job_reply_is_incomplete(
    overrides: dict[str, Any],
) -> None:
    """A zero field would be re-sent as a real setting (#856: knife 0 -> 1202)."""
    coordinator = _coordinator(reply=_job_reply(**overrides))

    with pytest.raises(HomeAssistantError):
        await _set(coordinator, "working_speed", 0.5)

    coordinator.async_send_command.assert_not_awaited()


async def test_idle_changes_only_update_the_plan() -> None:
    """Docked/ready: nothing goes to the mower, the value waits for the next job."""
    coordinator = _coordinator(sys_status=WorkMode.MODE_READY)

    await _set(coordinator, "working_speed", 0.5)
    await _set(coordinator, "blade_height", 45)

    coordinator.manager.send_command_and_wait.assert_not_awaited()
    coordinator.async_send_command.assert_not_awaited()
    assert coordinator.operation_settings.speed == pytest.approx(0.5)
    assert coordinator.operation_settings.blade_height == 45


async def test_luba1_mid_job_blade_change_uses_the_knife_command_like_the_app() -> None:
    """The app sends setKnifeHight on non-LubaPro models, never a route re-issue."""
    coordinator = _coordinator(device_name="Luba-A1B2C3")

    await _set(coordinator, "blade_height", 45)
    await _set(coordinator, "working_speed", 0.5)

    coordinator.async_send_and_wait.assert_awaited_once_with(
        "set_blade_height", "toapp_knife_status_change", height=45
    )
    coordinator.async_send_command.assert_not_awaited()


# --- what the entities show -------------------------------------------------


async def test_entities_show_the_running_job_once_it_has_been_read() -> None:
    """After a read, all three entities show the job, labelled as such."""
    coordinator = _coordinator()
    await _set(coordinator, "working_speed", 1.6 * FT_PER_S)

    blade = _description("blade_height").get_fn(coordinator)
    speed = _description("working_speed").get_fn(coordinator)
    spacing = _description("path_spacing").get_fn(coordinator)

    assert blade == JOB_KNIFE_MM
    assert spacing == JOB_WIDTH_CM
    assert speed == pytest.approx(1.6 * FT_PER_S, abs=0.005)
    assert coordinator.working_setting_source() == "running_job_after_ha_change"


def test_entities_show_the_plan_labelled_as_such_before_any_read() -> None:
    """No read yet: show what HA will send, and say so."""
    coordinator = _coordinator()

    assert _description("blade_height").get_fn(coordinator) == LOCAL_BLADE_MM
    assert _description("working_speed").get_fn(coordinator) == pytest.approx(0.2)
    assert _description("path_spacing").get_fn(coordinator) == LOCAL_WIDTH_CM
    assert coordinator.working_setting_source() == "next_job_plan"


async def test_a_new_job_invalidates_the_previous_jobs_values() -> None:
    """A different path hash means a different job: fall back to the plan."""
    coordinator = _coordinator()
    await _set(coordinator, "working_speed", 0.5)

    coordinator.data.report_data.work.path_hash = PATH_HASH + 1

    assert _description("blade_height").get_fn(coordinator) == LOCAL_BLADE_MM
    assert coordinator.working_setting_source() == "next_job_plan"


async def test_leaving_the_job_invalidates_its_values() -> None:
    """Docked again: the entities go back to the plan."""
    coordinator = _coordinator()
    await _set(coordinator, "working_speed", 0.5)

    coordinator.data.report_data.dev.sys_status = WorkMode.MODE_READY

    assert _description("path_spacing").get_fn(coordinator) == LOCAL_WIDTH_CM
    assert coordinator.working_setting_source() == "next_job_plan"


# --- the plan HA will send must be inside the model's limits ---------------


def test_out_of_range_plan_values_are_clamped_into_the_plan_not_just_the_display() -> (
    None
):
    """Pymammotion's default blade_height 0 was shown as 25 but sent as 0."""
    coordinator = _coordinator(sys_status=WorkMode.MODE_READY)
    coordinator._operation_settings = OperationSettings()  # noqa: SLF001
    coordinator.last_update_success = True
    limits = DeviceLimits(
        blade_height=RangeLimit(min=30, max=70),
        working_speed=RangeLimit(min=0.2, max=1.2),
        path_spacing=RangeLimit(min=8, max=14),
    )

    blade = MammotionWorkingNumberEntity(
        coordinator, _description("blade_height"), limits
    )
    spacing = MammotionWorkingNumberEntity(
        coordinator, _description("path_spacing"), limits
    )

    assert blade.native_value == 30
    assert coordinator.operation_settings.blade_height == 30
    assert spacing.native_value == 14
    assert coordinator.operation_settings.channel_width == 14
