"""The Bluetooth switch applies to every coordinator and survives restarts."""
# ruff: noqa: SLF001

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymammotion.transport.base import TransportType

from custom_components import mammotion as mammotion_init
from custom_components.mammotion import connectivity_store
from custom_components.mammotion.coordinator import MammotionBaseUpdateCoordinator


def test_unknown_switch_state_counts_as_on() -> None:
    """Nothing persisted must never switch Bluetooth off."""
    assert connectivity_store.get_bluetooth_enabled({}, "entry", "Luba-Test") is True


def test_persisting_one_mower_does_not_touch_others() -> None:
    """Switch state is keyed per entry and per mower, and inputs are not mutated."""
    original = {"entry": {"Luba-Other": {"bluetooth": True}}}
    updated = connectivity_store.set_bluetooth_enabled(
        original, "entry", "Luba-Test", False
    )
    assert (
        connectivity_store.get_bluetooth_enabled(updated, "entry", "Luba-Test") is False
    )
    assert (
        connectivity_store.get_bluetooth_enabled(updated, "entry", "Luba-Other") is True
    )
    assert "Luba-Test" not in original["entry"]


@pytest.mark.asyncio
async def test_store_round_trip_uses_the_dedicated_key() -> None:
    """Saving then reading returns the persisted value via its own Store key."""
    saved: dict = {}

    class FakeStore:
        def __init__(self, _hass, version, key) -> None:
            assert key == connectivity_store.STORAGE_KEY
            assert version == connectivity_store.STORAGE_VERSION

        async def async_load(self):
            return saved.get("data")

        async def async_save(self, data) -> None:
            saved["data"] = data

    hass = SimpleNamespace(data={})
    with patch.object(connectivity_store, "Store", FakeStore):
        await connectivity_store.async_set_bluetooth_enabled(
            hass, "entry", "Luba-Test", False
        )
        hass.data.clear()
        assert (
            await connectivity_store.async_get_bluetooth_enabled(
                hass, "entry", "Luba-Test"
            )
            is False
        )


def _mower(device_name: str) -> SimpleNamespace:
    def coordinator() -> SimpleNamespace:
        return SimpleNamespace(device_name=device_name, _bluetooth_enabled=True)

    return SimpleNamespace(
        reporting_coordinator=coordinator(),
        maintenance_coordinator=coordinator(),
        version_coordinator=coordinator(),
        map_coordinator=coordinator(),
        error_coordinator=coordinator(),
    )


def test_toggle_reaches_all_five_coordinators_of_this_mower_only() -> None:
    """Every coordinator runs the per-tick BLE push, so all must see the switch."""
    mine, other = _mower("Luba-Test"), _mower("Luba-Other")
    entry = SimpleNamespace(runtime_data=SimpleNamespace(mowers=[other, mine]))
    this = SimpleNamespace(device_name="Luba-Test", config_entry=entry)
    this._sibling_coordinators = lambda: (
        MammotionBaseUpdateCoordinator._sibling_coordinators(this)
    )

    MammotionBaseUpdateCoordinator._propagate_bluetooth_enabled(this, False)

    names = ("reporting", "maintenance", "version", "map", "error")
    assert all(
        getattr(mine, f"{n}_coordinator")._bluetooth_enabled is False for n in names
    )
    assert all(
        getattr(other, f"{n}_coordinator")._bluetooth_enabled is True for n in names
    )


@pytest.mark.asyncio
async def test_toggle_persists_the_switch() -> None:
    """Turning Bluetooth off records it for the next restart."""
    this = SimpleNamespace(
        device_name="Luba-Test",
        config_entry=SimpleNamespace(entry_id="entry"),
        hass=MagicMock(),
    )
    with patch.object(
        connectivity_store, "async_set_bluetooth_enabled", AsyncMock()
    ) as persist:
        await MammotionBaseUpdateCoordinator._async_persist_bluetooth_enabled(
            this, False
        )
    persist.assert_awaited_once_with(this.hass, "entry", "Luba-Test", False)


@pytest.mark.asyncio
async def test_persisted_off_is_applied_before_connecting() -> None:
    """A restart with the switch left off marks every coordinator off and removes BLE."""
    handle = SimpleNamespace(remove_transport=AsyncMock())
    mammotion = SimpleNamespace(set_prefer_ble=MagicMock(), mower=lambda _name: handle)
    coordinators = tuple(SimpleNamespace(_bluetooth_enabled=True) for _ in range(5))
    entry = SimpleNamespace(entry_id="entry")

    with patch.object(
        connectivity_store, "async_get_bluetooth_enabled", AsyncMock(return_value=False)
    ):
        result = await mammotion_init.async_apply_persisted_bluetooth_switch(
            MagicMock(), entry, mammotion, "Luba-Test", coordinators
        )

    assert result is False
    assert all(c._bluetooth_enabled is False for c in coordinators)
    mammotion.set_prefer_ble.assert_called_once_with("Luba-Test", prefer_ble=False)
    handle.remove_transport.assert_awaited_once_with(TransportType.BLE)


@pytest.mark.asyncio
async def test_persisted_on_changes_nothing() -> None:
    """The default path must not touch transports or preferences."""
    handle = SimpleNamespace(remove_transport=AsyncMock())
    mammotion = SimpleNamespace(set_prefer_ble=MagicMock(), mower=lambda _name: handle)
    coordinators = tuple(SimpleNamespace(_bluetooth_enabled=True) for _ in range(5))

    with patch.object(
        connectivity_store, "async_get_bluetooth_enabled", AsyncMock(return_value=True)
    ):
        result = await mammotion_init.async_apply_persisted_bluetooth_switch(
            MagicMock(),
            SimpleNamespace(entry_id="entry"),
            mammotion,
            "Luba-Test",
            coordinators,
        )

    assert result is True
    assert all(c._bluetooth_enabled is True for c in coordinators)
    mammotion.set_prefer_ble.assert_not_called()
    handle.remove_transport.assert_not_awaited()
