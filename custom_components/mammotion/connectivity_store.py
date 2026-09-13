"""Persist per-mower connectivity switch state across Home Assistant restarts.

The Bluetooth switch used to reset to on after every restart. A dedicated
``Store`` is used rather than config-entry options: the options flow replaces
options wholesale, so a switch persisted there would be silently wiped the next
time anyone saved the integration's options.
"""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN

STORAGE_KEY = f"{DOMAIN}.connectivity_switches"
STORAGE_VERSION = 1
_DATA_KEY = f"{DOMAIN}_connectivity_store"


def get_bluetooth_enabled(
    data: dict[str, Any], entry_id: str, device_name: str
) -> bool:
    """Return the persisted Bluetooth switch state; unknown counts as on."""
    return bool(data.get(entry_id, {}).get(device_name, {}).get("bluetooth", True))


def set_bluetooth_enabled(
    data: dict[str, Any], entry_id: str, device_name: str, enabled: bool
) -> dict[str, Any]:
    """Return a copy of *data* with this mower's Bluetooth switch recorded."""
    updated = {
        k: {d: dict(v) for d, v in devices.items()} for k, devices in data.items()
    }
    updated.setdefault(entry_id, {}).setdefault(device_name, {})["bluetooth"] = enabled
    return updated


def _store(hass: HomeAssistant) -> Store[dict[str, Any]]:
    return Store(hass, STORAGE_VERSION, STORAGE_KEY)


async def _async_load(hass: HomeAssistant) -> dict[str, Any]:
    cached = hass.data.get(_DATA_KEY)
    if cached is None:
        cached = await _store(hass).async_load() or {}
        hass.data[_DATA_KEY] = cached
    return cached


async def async_get_bluetooth_enabled(
    hass: HomeAssistant, entry_id: str, device_name: str
) -> bool:
    """Return the persisted Bluetooth switch state for one mower."""
    return get_bluetooth_enabled(await _async_load(hass), entry_id, device_name)


async def async_set_bluetooth_enabled(
    hass: HomeAssistant, entry_id: str, device_name: str, enabled: bool
) -> None:
    """Persist the Bluetooth switch state for one mower."""
    updated = set_bluetooth_enabled(
        await _async_load(hass), entry_id, device_name, enabled
    )
    hass.data[_DATA_KEY] = updated
    await _store(hass).async_save(updated)


async def async_remove(hass: HomeAssistant) -> None:
    """Delete the persisted switch state when the integration is removed."""
    hass.data.pop(_DATA_KEY, None)
    await _store(hass).async_remove()
