"""Fixtures for Mammotion tests."""

import time
from collections.abc import Callable, Coroutine
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymammotion.data.model.hash_list import Plan

from custom_components.mammotion import services as mammotion_services


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations) -> None:
    """Enable loading Mammotion from custom_components."""


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
        position_epoch=1,
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
