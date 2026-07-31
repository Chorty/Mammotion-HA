"""Tests for Mammotion config-entry lifecycle helpers."""
# ruff: noqa: SLF001

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymammotion.transport.base import BLEUnavailableError, TransportType

from custom_components import mammotion as mammotion_init
from custom_components.mammotion import async_remove_config_entry_device
from custom_components.mammotion.coordinator import MammotionBaseUpdateCoordinator


@pytest.mark.asyncio
async def test_active_mower_device_cannot_be_removed() -> None:
    """An active mower remains attached to its config entry."""
    entry = SimpleNamespace(
        runtime_data=SimpleNamespace(mowers=[SimpleNamespace(unique_name="Luba-123")])
    )
    device = SimpleNamespace(identifiers={("mammotion", "Luba-123")})

    assert not await async_remove_config_entry_device(None, entry, device)


@pytest.mark.asyncio
async def test_stale_mower_device_can_be_removed() -> None:
    """A device absent from runtime data may be removed."""
    entry = SimpleNamespace(runtime_data=SimpleNamespace(mowers=[]))
    device = SimpleNamespace(identifiers={("mammotion", "Luba-123")})

    assert await async_remove_config_entry_device(None, entry, device)


@pytest.mark.asyncio
async def test_cloud_mower_registers_late_ble_attachment_callback() -> None:
    """A proxy unavailable during setup attaches BLE on a later advertisement."""
    mower_state = SimpleNamespace(ble_mac="")
    mammotion = SimpleNamespace(
        get_device_by_name=MagicMock(
            return_value=SimpleNamespace(mower_state=mower_state)
        ),
        add_ble_to_device=AsyncMock(),
    )
    device = SimpleNamespace(device_name="Luba-Test")
    hass = MagicMock()
    entry = MagicMock()

    with (
        patch.object(
            mammotion_init.bluetooth,
            "async_ble_device_from_address",
            return_value=None,
        ),
        patch.object(
            mammotion_init, "_register_ble_reconnect_callback"
        ) as register_callback,
    ):
        await mammotion_init._attach_ble_to_mower(
            hass, entry, mammotion, device, "AA:BB:CC:DD:EE:FF"
        )

    assert mower_state.ble_mac == "AA:BB:CC:DD:EE:FF"
    mammotion.add_ble_to_device.assert_not_awaited()
    register_callback.assert_called_once_with(
        hass,
        entry,
        mammotion,
        "Luba-Test",
        "AA:BB:CC:DD:EE:FF",
    )


@pytest.mark.asyncio
async def test_bluetooth_toggle_off_refreshes_gate_entities() -> None:
    """A clean disconnect immediately refreshes ble_link_live."""
    handle = SimpleNamespace(
        set_prefer_ble=MagicMock(),
        disconnect_transport=AsyncMock(),
    )
    coordinator = SimpleNamespace(
        _bluetooth_enabled=True,
        device_name="Luba-Test",
        manager=SimpleNamespace(mower=lambda _name: handle),
        _async_refresh_motion_gate_entities=MagicMock(),
    )

    await MammotionBaseUpdateCoordinator.async_set_bluetooth_enabled(coordinator, False)

    assert coordinator._bluetooth_enabled is False
    handle.set_prefer_ble.assert_called_once_with(value=False)
    handle.disconnect_transport.assert_awaited_once_with(TransportType.BLE)
    coordinator._async_refresh_motion_gate_entities.assert_called_once_with()


@pytest.mark.asyncio
async def test_bluetooth_toggle_on_survives_temporarily_unavailable_link() -> None:
    """Enable stays on so a later advertisement can connect after a miss."""
    handle = SimpleNamespace(set_prefer_ble=MagicMock())
    coordinator = SimpleNamespace(
        _bluetooth_enabled=False,
        device_name="Luba-Test",
        manager=SimpleNamespace(mower=lambda _name: handle),
        _async_ensure_ble_client=AsyncMock(
            side_effect=BLEUnavailableError("not advertising")
        ),
        _async_refresh_motion_gate_entities=MagicMock(),
    )

    await MammotionBaseUpdateCoordinator.async_set_bluetooth_enabled(coordinator, True)

    assert coordinator._bluetooth_enabled is True
    handle.set_prefer_ble.assert_called_once_with(value=True)
    coordinator._async_refresh_motion_gate_entities.assert_called_once_with()


def test_bluetooth_toggle_invalidates_cached_gate_snapshot() -> None:
    """The listener sees a fresh BLE verdict instead of the five-second cache."""
    coordinator = SimpleNamespace(
        _mammotion_gate_snapshot_monotonic=123.0,
        async_update_listeners=MagicMock(),
    )

    MammotionBaseUpdateCoordinator._async_refresh_motion_gate_entities(coordinator)

    assert coordinator._mammotion_gate_snapshot_monotonic == float("-inf")
    coordinator.async_update_listeners.assert_called_once_with()
