"""Ports from upstream Mammotion-HA: BLE switch, keep-alive tolerance, dynamics poller."""
# ruff: noqa: SLF001

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymammotion.transport.base import AuthError, BLEUnavailableError

from custom_components import mammotion as mammotion_init
from custom_components.mammotion import coordinator as coordinator_module
from custom_components.mammotion.coordinator import (
    MammotionBaseUpdateCoordinator,
    MammotionMapUpdateCoordinator,
)


def _keep_alive_coordinator(
    handle: SimpleNamespace, **extra: object
) -> SimpleNamespace:
    coordinator = SimpleNamespace(
        device_name="Luba-Test",
        manager=SimpleNamespace(mower=lambda _name: handle),
        **extra,
    )
    coordinator._async_restart_keep_alive = lambda h: (
        MammotionBaseUpdateCoordinator._async_restart_keep_alive(coordinator, h)
    )
    return coordinator


@pytest.mark.asyncio
async def test_keep_alive_restart_tolerates_a_ble_miss() -> None:
    """A BLE cooldown during keep-alive restart must not escape."""
    handle = SimpleNamespace(
        restart_keep_alive=AsyncMock(side_effect=BLEUnavailableError("cooldown"))
    )
    coordinator = _keep_alive_coordinator(handle)

    await MammotionBaseUpdateCoordinator._async_restart_keep_alive(coordinator, handle)


@pytest.mark.asyncio
async def test_keep_alive_restart_still_raises_credential_errors() -> None:
    """AuthError subclasses TransportError here; re-auth must not be swallowed."""
    handle = SimpleNamespace(
        restart_keep_alive=AsyncMock(side_effect=AuthError("expired"))
    )
    coordinator = _keep_alive_coordinator(handle)

    with pytest.raises(AuthError):
        await MammotionBaseUpdateCoordinator._async_restart_keep_alive(
            coordinator, handle
        )


@pytest.mark.asyncio
async def test_enabling_scheduled_updates_survives_a_ble_miss() -> None:
    """set_scheduled_updates(True) completes when keep-alive cannot reach BLE."""
    handle = SimpleNamespace(
        restart_keep_alive=AsyncMock(side_effect=BLEUnavailableError("cooldown")),
        stop_polling=AsyncMock(),
    )
    device = SimpleNamespace(enabled=False, online=True)
    coordinator = _keep_alive_coordinator(handle, update_failures=3)
    coordinator.manager.get_device_by_name = lambda _name: device
    coordinator.manager.set_scheduled_updates = AsyncMock()

    await MammotionBaseUpdateCoordinator.set_scheduled_updates(coordinator, True)

    assert device.enabled is True
    handle.restart_keep_alive.assert_awaited_once()


@pytest.mark.asyncio
async def test_enabling_cloud_survives_a_ble_miss() -> None:
    """async_set_cloud_enabled(True) completes when keep-alive cannot reach BLE."""
    handle = SimpleNamespace(
        connect_transport=AsyncMock(),
        restart_keep_alive=AsyncMock(side_effect=BLEUnavailableError("cooldown")),
    )
    coordinator = _keep_alive_coordinator(handle, _cloud_enabled=False)

    await MammotionBaseUpdateCoordinator.async_set_cloud_enabled(coordinator, True)

    assert coordinator._cloud_enabled is True
    handle.restart_keep_alive.assert_awaited_once()


@pytest.mark.asyncio
async def test_cloud_reconnect_swallows_a_ble_miss_but_refreshes_login_on_auth() -> (
    None
):
    """The reconnect handler catches TransportError only after credential errors."""
    handle = SimpleNamespace(
        connect_transport=AsyncMock(),
        restart_keep_alive=AsyncMock(side_effect=BLEUnavailableError("cooldown")),
    )
    coordinator = _keep_alive_coordinator(handle, async_refresh_login=AsyncMock())
    await MammotionBaseUpdateCoordinator._async_reconnect_cloud(coordinator, "cloud")
    coordinator.async_refresh_login.assert_not_awaited()

    handle.restart_keep_alive = AsyncMock(side_effect=AuthError("expired"))
    await MammotionBaseUpdateCoordinator._async_reconnect_cloud(coordinator, "cloud")
    coordinator.async_refresh_login.assert_awaited_once()


@pytest.mark.asyncio
async def test_tick_does_not_reattach_ble_when_switch_is_off() -> None:
    """update_ble_device creates a transport, so the per-tick push must honour the switch."""
    device = SimpleNamespace(mower_state=SimpleNamespace(ble_mac="aa:bb:cc:dd:ee:ff"))
    manager = SimpleNamespace(update_ble_device=AsyncMock())
    off = SimpleNamespace(
        _bluetooth_enabled=False,
        device_name="Luba-Test",
        hass=MagicMock(),
        manager=manager,
    )
    on = SimpleNamespace(
        _bluetooth_enabled=True,
        device_name="Luba-Test",
        hass=MagicMock(),
        manager=manager,
    )

    with patch.object(
        coordinator_module.bluetooth,
        "async_ble_device_from_address",
        return_value="dev",
    ):
        await MammotionBaseUpdateCoordinator._async_push_ble_advertisement(
            off, device, object()
        )
        manager.update_ble_device.assert_not_awaited()
        await MammotionBaseUpdateCoordinator._async_push_ble_advertisement(
            on, device, object()
        )

    manager.update_ble_device.assert_awaited_once_with("Luba-Test", "dev")


def _captured_ble_seen(entry: object) -> tuple[object, MagicMock]:
    hass = MagicMock()
    mammotion = SimpleNamespace(
        mower=lambda _name: object(), add_ble_to_device=MagicMock()
    )
    with patch.object(mammotion_init.bluetooth, "async_register_callback") as register:
        mammotion_init._register_ble_reconnect_callback(
            hass, entry, mammotion, "Luba-Test", "AA:BB:CC:DD:EE:FF"
        )
    return register.call_args.args[1], hass


def _entry(*, bluetooth_enabled: bool) -> SimpleNamespace:
    coordinator = SimpleNamespace(
        device_name="Luba-Test", bluetooth_enabled=bluetooth_enabled
    )
    return SimpleNamespace(
        runtime_data=SimpleNamespace(
            mowers=[SimpleNamespace(reporting_coordinator=coordinator)]
        ),
        async_on_unload=MagicMock(),
    )


def test_advertisement_does_not_reattach_ble_when_switch_is_off() -> None:
    """_ble_seen must not re-create a transport the switch removed."""
    ble_seen, hass = _captured_ble_seen(_entry(bluetooth_enabled=False))
    ble_seen(SimpleNamespace(device="dev"), None)
    hass.async_create_task.assert_not_called()


def test_advertisement_attaches_ble_when_switch_is_on() -> None:
    """With the switch on, advertisements still attach BLE."""
    ble_seen, hass = _captured_ble_seen(_entry(bluetooth_enabled=True))
    ble_seen(SimpleNamespace(device="dev"), None)
    hass.async_create_task.assert_called_once()


def test_advertisement_attaches_ble_before_runtime_data_exists() -> None:
    """During setup there is no runtime data yet; BLE must still attach."""
    ble_seen, hass = _captured_ble_seen(SimpleNamespace(async_on_unload=MagicMock()))
    ble_seen(SimpleNamespace(device="dev"), None)
    hass.async_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_map_setup_does_not_register_the_ha_side_dynamics_poller() -> None:
    """Only pymammotion's guarded loop polls the dynamics line."""
    handle = SimpleNamespace(watch_field=MagicMock())
    coordinator = object.__new__(MammotionMapUpdateCoordinator)
    coordinator.device_name = "Luba-Test"
    coordinator.manager = SimpleNamespace(
        get_device_by_name=lambda _name: SimpleNamespace(enabled=False, online=False),
        mower=lambda _name: handle,
    )

    with patch.object(MammotionBaseUpdateCoordinator, "_async_setup", AsyncMock()):
        await MammotionMapUpdateCoordinator._async_setup(coordinator)

    handle.watch_field.assert_not_called()
