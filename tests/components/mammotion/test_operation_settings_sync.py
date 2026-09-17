"""The mowing settings HA shows, and what a mid-mow change actually sends.

Reported by the operator 2026-09-17: a mow was started from the vendor app at
blade height 2.2" and 1.3 ft/s; Home Assistant's Working speed / Blade height /
Path spacing entities showed none of it, and changing *speed* in HA dropped the
mower's blade to 1" -- Home Assistant's own slider floor, a value nobody chose.
(The job's speed was first reported as 2 ft/s and corrected to 1.3 ft/s the same
day; the fixtures below carry the corrected figure.)

Root cause and per-field evidence:
``docs/findings-operation-settings-sync-20260917.md``.

Two properties are pinned here:

1. the entities show the running job's real values, and label which of the two
   sources the displayed number came from;
2. changing one setting mid-mow changes only that setting.

Both are written against the real ``MammotionBaseUpdateCoordinator`` methods,
called unbound with a stub ``self`` -- the pattern the rest of this suite uses
for coordinator helpers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pymammotion.data.model.device_config import OperationSettings
from pymammotion.data.model.work import CurrentTaskSettings

from custom_components.mammotion.coordinator import MammotionBaseUpdateCoordinator
from custom_components.mammotion.number import (
    LUBA_WORKING_ENTITIES,
    NUMBER_WORKING_ENTITIES,
)

#: The app-started job in the operator's report: 2.2" blade (55.88 mm), 1.3 ft/s
#: (0.3962 m/s), 25 cm spacing.  None of these equals a slider floor, so "read
#: from the job" cannot be confused with "fell back to a default" -- which is
#: the one property these fixtures have to have.
JOB_ZONE = 987_654_321
JOB_BLADE_MM = 56
JOB_SPEED_MS = 0.3962
JOB_SPACING_CM = 25


def _work(
    *,
    zone_hashs: list[int] | None = None,
    knife_height: int = JOB_BLADE_MM,
    speed: float = JOB_SPEED_MS,
    channel_width: int = JOB_SPACING_CM,
) -> CurrentTaskSettings:
    """Build the device's description of the job it is running."""
    return CurrentTaskSettings(
        zone_hashs=[JOB_ZONE] if zone_hashs is None else zone_hashs,
        knife_height=knife_height,
        speed=speed,
        channel_width=channel_width,
        job_id=7,
        job_ver=2,
        job_mode=4,
        edge_mode=1,
        toward=15,
        toward_mode=1,
        toward_included_angle=90,
        ultra_wave=2,
        channel_mode=1,
    )


def _coordinator(
    *,
    work: CurrentTaskSettings | None = None,
    mowing: bool = True,
    report_knife_height: int = JOB_BLADE_MM,
    device_name: str = "Luba-VS1234",  # Luba 2: DeviceType.is_luba_pro is True
    refresh_fills_work: bool = True,
) -> SimpleNamespace:
    """Build a coordinator stub with a real ``OperationSettings`` and device state.

    ``refresh_fills_work`` models the device answering (or not answering)
    ``query_generate_route_information``.  ``async_send_and_wait`` swallows its
    own transport timeouts and returns ``None`` either way, so a silent
    non-answer is the realistic failure -- not an exception.
    """
    work = _work() if work is None else work
    data = SimpleNamespace(
        work=work,
        report_data=SimpleNamespace(
            work=SimpleNamespace(
                # bp_hash inside the job's zone list + progress != 100 is the
                # running-job predicate.
                bp_hash=JOB_ZONE if mowing else 1,
                area=(50 << 16),
                knife_height=report_knife_height,
            )
        ),
    )
    coordinator = SimpleNamespace(
        data=data,
        device_name=device_name,
        _operation_settings=OperationSettings(),  # what the real methods mutate
        async_modify_plan_route=AsyncMock(),
        async_blade_height=AsyncMock(),
    )
    # The property the entities read; the same object the methods mutate.
    coordinator.operation_settings = coordinator._operation_settings  # noqa: SLF001

    async def _send_and_wait(command: str, expected_field: str, **kwargs: Any) -> None:
        coordinator.sent_queries.append((command, expected_field))
        if not refresh_fills_work:
            coordinator.data.work = CurrentTaskSettings(zone_hashs=[JOB_ZONE])

    coordinator.sent_queries = []
    coordinator.async_send_and_wait = _send_and_wait
    # The production methods under test call each other through ``self``, so
    # bind the real ones onto the stub rather than re-implementing any of them.
    #
    # Binding is deliberately tolerant of a method not existing: these tests are
    # also run against the pre-fix tree to show they catch the defect, and a
    # test that only ever failed with "no such attribute" would prove nothing
    # about the behaviour it claims to pin.  Every assertion below therefore
    # reaches a real value on both trees.
    for name in _REAL_METHODS:
        if hasattr(MammotionBaseUpdateCoordinator, name):
            setattr(coordinator, name, _bind(name, coordinator))
    return coordinator


#: Every coordinator method involved in reading or applying a job setting.
#: Bound onto the stub verbatim -- nothing here is a test double.
_REAL_METHODS = (
    "_is_route_job_running",
    "_running_job_settings_are_known",
    "_async_refresh_running_job_settings",
    "_seed_operation_settings_from_running_job",
    "_async_apply_route_field_if_working",
    "async_change_blade_height_if_working",
    "async_change_speed_if_working",
    "async_change_path_spacing_if_working",
    "async_modify_plan_if_mowing",
    "running_job_setting",
)


def _bind(name: str, coordinator: SimpleNamespace) -> Any:
    """Bind an unbound coordinator method to the stub."""
    return getattr(MammotionBaseUpdateCoordinator, name).__get__(coordinator)


def _description(entities: tuple[Any, ...], key: str) -> Any:
    for description in entities:
        if description.key == key:
            return description
    raise AssertionError(f"no {key} entity description")


BLADE = _description(LUBA_WORKING_ENTITIES, "blade_height")
SPEED = _description(NUMBER_WORKING_ENTITIES, "working_speed")
SPACING = _description(NUMBER_WORKING_ENTITIES, "path_spacing")


# --------------------------------------------------------------------------
# Property 1 -- the entities show the job's real values
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        (BLADE, JOB_BLADE_MM),
        (SPEED, JOB_SPEED_MS),
        (SPACING, JOB_SPACING_CM),
    ],
    ids=["blade_height", "working_speed", "path_spacing"],
)
def test_entities_show_the_running_jobs_values(
    description: Any, expected: float
) -> None:
    """A job started in the app is what the entities read, not local defaults.

    Before the fix ``working_speed`` and ``path_spacing`` had no ``get_fn`` at
    all and ``blade_height``'s read ``operation_settings``, so all three
    returned pymammotion's dataclass defaults (0.3 / 25 / 0) regardless of what
    the mower was doing.
    """
    coordinator = _coordinator()

    assert description.get_fn is not None
    assert description.get_fn(coordinator) == expected


def test_blade_height_prefers_the_live_report_over_the_cached_plan() -> None:
    """``RptWork.knife_height`` is in the ~1 Hz stream; ``work`` is cached.

    When the two disagree the live one wins -- the cached record is only ever
    refreshed by a ``bidire_reqconver_path`` message and can be a whole job old.
    """
    coordinator = _coordinator(report_knife_height=62)

    assert BLADE.get_fn(coordinator) == 62


def test_an_unreported_job_falls_back_to_the_staged_value_not_a_zero() -> None:
    """An untouched ``work`` record must never be presented as the mower's state.

    ``CurrentTaskSettings`` defaults every field to zero, so a job HA was never
    told about would otherwise read as 0 mm / 0 m/s -- indistinguishable from a
    real reading, and wrong.
    """
    coordinator = _coordinator(work=CurrentTaskSettings(zone_hashs=[JOB_ZONE]))
    coordinator.data.report_data.work.knife_height = 0
    coordinator.operation_settings.speed = 0.35

    assert SPEED.get_fn(coordinator) == 0.35
    assert SPACING.get_fn(coordinator) == 25  # the staged default, not work's 0


def test_value_source_distinguishes_a_reading_from_a_value_ha_would_send() -> None:
    """The operator must be able to tell the two apart without reading code."""
    running = _coordinator()
    idle = _coordinator(mowing=False)

    assert hasattr(running, "running_job_setting"), (
        "without a way to ask whether a value came from the mower, every "
        "displayed number is indistinguishable from a local default"
    )
    assert running.running_job_setting("speed") == JOB_SPEED_MS
    assert idle.running_job_setting("speed") is None


# --------------------------------------------------------------------------
# Property 2 -- changing one setting changes only that setting
# --------------------------------------------------------------------------


def _sent(coordinator: SimpleNamespace) -> OperationSettings:
    coordinator.async_modify_plan_route.assert_awaited_once()
    return coordinator.async_modify_plan_route.await_args.args[0]


async def test_a_midmow_speed_change_leaves_blade_height_alone() -> None:
    """The reported incident, as a test.

    Before the fix this sent ``blade_height`` straight out of
    ``operation_settings`` -- 25 mm, HA's slider floor -- which is how a speed
    change dropped the mower's blade to 1".
    """
    coordinator = _coordinator()
    # What HA had staged locally: the slider floors that caused the incident.
    coordinator.operation_settings.blade_height = 25
    coordinator.operation_settings.channel_width = 20

    SPEED.set_fn(coordinator, 0.5)
    await _bind("async_change_speed_if_working", coordinator)()

    sent = _sent(coordinator)
    assert sent.speed == 0.5, "the edited field must change"
    assert sent.blade_height == JOB_BLADE_MM, "the blade must stay where the job put it"
    assert sent.channel_width == JOB_SPACING_CM, "spacing must stay too"


async def test_a_midmow_blade_change_leaves_speed_and_spacing_alone() -> None:
    """The mirror image: editing height must not re-send a stale speed."""
    coordinator = _coordinator()
    coordinator.operation_settings.speed = 0.2
    coordinator.operation_settings.channel_width = 20

    BLADE.set_fn(coordinator, 45)
    await _bind("async_change_blade_height_if_working", coordinator)()

    sent = _sent(coordinator)
    assert sent.blade_height == 45
    assert sent.speed == JOB_SPEED_MS
    assert sent.channel_width == JOB_SPACING_CM


async def test_a_midmow_change_carries_the_jobs_identity_and_geometry() -> None:
    """Seeding must keep the eight fields the old code already preserved."""
    coordinator = _coordinator()

    SPEED.set_fn(coordinator, 0.5)
    await _bind("async_change_speed_if_working", coordinator)()

    sent = _sent(coordinator)
    assert sent.areas == [JOB_ZONE]
    assert (sent.job_id, sent.job_version, sent.job_mode) == (7, 2, 4)
    assert (sent.toward, sent.toward_mode, sent.toward_included_angle) == (15, 1, 90)
    assert sent.mowing_laps == 1
    assert (sent.ultra_wave, sent.channel_mode) == (2, 1)


async def test_the_running_jobs_settings_are_re_read_before_anything_is_sent() -> None:
    """A job started in the app is never described to HA unless HA asks.

    ``query_generate_route_information`` (``NavReqCoverPath sub_cmd=2``) is the
    ask; without it the seed would come from whatever ``work`` happened to hold.
    """
    coordinator = _coordinator()

    SPEED.set_fn(coordinator, 0.5)
    await _bind("async_change_speed_if_working", coordinator)()

    assert coordinator.sent_queries == [
        ("query_generate_route_information", "bidire_reqconver_path")
    ]


async def test_nothing_is_sent_when_the_device_will_not_describe_its_job() -> None:
    """Refusing to act beats guessing at a running mower's blade height.

    ``async_send_and_wait`` swallows transport timeouts, so the non-answer is
    silent; the guard is the state of ``work`` afterwards, not an exception.
    """
    coordinator = _coordinator(refresh_fills_work=False)
    coordinator.operation_settings.blade_height = 25

    SPEED.set_fn(coordinator, 0.5)
    await _bind("async_change_speed_if_working", coordinator)()

    coordinator.async_modify_plan_route.assert_not_awaited()


async def test_an_idle_mower_is_not_sent_anything() -> None:
    """An idle change is staged for the next job, not pushed at the device."""
    coordinator = _coordinator(mowing=False)

    SPEED.set_fn(coordinator, 0.5)
    await _bind("async_change_speed_if_working", coordinator)()

    coordinator.async_modify_plan_route.assert_not_awaited()
    assert coordinator.sent_queries == []
    assert coordinator.operation_settings.speed == 0.5


async def test_path_spacing_now_reaches_a_running_job_at_all() -> None:
    """``path_spacing`` had no ``set_async_fn``, so it silently never applied."""
    coordinator = _coordinator()

    assert SPACING.set_async_fn is not None
    SPACING.set_fn(coordinator, 30)
    await _bind("async_change_path_spacing_if_working", coordinator)()

    sent = _sent(coordinator)
    assert sent.channel_width == 30
    assert sent.speed == JOB_SPEED_MS
    assert sent.blade_height == JOB_BLADE_MM


async def test_luba_1_nudges_the_blade_motor_instead_of_reissuing_the_route() -> None:
    """The original Luba 1's in-job editor only moves the blade motor."""
    coordinator = _coordinator(device_name="Luba-AWD1000")

    BLADE.set_fn(coordinator, 45)
    await _bind("async_change_blade_height_if_working", coordinator)()

    coordinator.async_blade_height.assert_awaited_once_with(45)
    coordinator.async_modify_plan_route.assert_not_awaited()


async def test_luba_1_sends_nothing_for_a_midmow_speed_change() -> None:
    """Its in-job editor has no speed control, so there is nothing to mirror."""
    coordinator = _coordinator(device_name="Luba-AWD1000")

    SPEED.set_fn(coordinator, 0.5)
    await _bind("async_change_speed_if_working", coordinator)()

    coordinator.async_modify_plan_route.assert_not_awaited()
    assert coordinator.sent_queries == []


# --------------------------------------------------------------------------
# The defect itself, driven through the entity wiring rather than the new
# coordinator methods.  ``set_async_fn`` exists on both the pre-fix and the
# fixed tree, so these two fail on the pre-fix tree for the *reported* reason --
# a wrong blade height on the wire -- not because a helper is missing.
# --------------------------------------------------------------------------


async def test_the_reported_incident_does_not_reproduce_through_the_entity() -> None:
    """Operator, 2026-09-17: changing speed in HA dropped the blade to 1".

    25 mm is ``blade_height``'s slider floor and what ``RestoreNumber`` puts
    back into ``operation_settings`` on every HA start.  On the pre-fix tree
    this reaches the mower verbatim.
    """
    coordinator = _coordinator()
    coordinator.operation_settings.blade_height = 25  # HA's floor: 0.98"
    coordinator.operation_settings.channel_width = 20

    assert SPEED.set_async_fn is not None
    SPEED.set_fn(coordinator, 0.5)
    await SPEED.set_async_fn(coordinator, 0.5)

    sent = _sent(coordinator)
    assert sent.speed == 0.5
    assert sent.blade_height == JOB_BLADE_MM, (
        "a speed change re-sent HA's slider floor as the blade height -- "
        'this is the reported 1" cut'
    )
    assert sent.channel_width == JOB_SPACING_CM


async def test_the_entity_shows_the_job_not_a_local_default() -> None:
    """All three entities, through their own descriptions, on a running job."""
    coordinator = _coordinator()
    # Local staging holds the defaults that were being displayed as fact.
    coordinator.operation_settings.speed = 0.3
    coordinator.operation_settings.channel_width = 25
    coordinator.operation_settings.blade_height = 25

    for description, expected, label in (
        (BLADE, JOB_BLADE_MM, "blade height"),
        (SPEED, JOB_SPEED_MS, "working speed"),
        (SPACING, JOB_SPACING_CM, "path spacing"),
    ):
        assert description.get_fn is not None, f"{label} has no way to read the job"
        assert description.get_fn(coordinator) == expected, (
            f"{label} showed a local default instead of the running job's value"
        )
