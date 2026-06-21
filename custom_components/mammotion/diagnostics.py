"""Privacy-preserving diagnostics support for Mammotion."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.loader import async_get_integration

from . import MammotionConfigEntry
from .const import CONF_BLE_DEVICES, CONF_HAS_CLOUD_ACCOUNT, CONF_USE_WIFI, DOMAIN

MAX_DIAGNOSTIC_DEVICES = 20


def _coordinator_status(coordinator: Any) -> dict[str, Any]:
    """Return bounded coordinator health without exposing device payloads."""
    interval = getattr(coordinator, "update_interval", None)
    return {
        "last_update_success": bool(getattr(coordinator, "last_update_success", False)),
        "update_interval_seconds": (
            interval.total_seconds() if interval is not None else None
        ),
    }


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant,
    entry: MammotionConfigEntry,
) -> dict[str, Any]:
    """Return sanitized diagnostics for a config entry."""
    integration = await async_get_integration(hass, DOMAIN)
    runtime = entry.runtime_data
    has_ble = bool(entry.data.get(CONF_BLE_DEVICES))
    has_cloud = bool(entry.data.get(CONF_HAS_CLOUD_ACCOUNT, False)) and bool(
        entry.data.get(CONF_USE_WIFI, True)
    )
    connection_mode = (
        "hybrid" if has_ble and has_cloud else "cloud" if has_cloud else "bluetooth"
    )

    mowers = [
        {
            "index": index,
            "reporting": _coordinator_status(device.reporting_coordinator),
            "maintenance": _coordinator_status(device.maintenance_coordinator),
            "firmware": _coordinator_status(device.version_coordinator),
            "map": _coordinator_status(device.map_coordinator),
            "errors": _coordinator_status(device.error_coordinator),
        }
        for index, device in enumerate(runtime.mowers[:MAX_DIAGNOSTIC_DEVICES], start=1)
    ]
    rtk_devices = [
        {"index": index, "coordinator": _coordinator_status(device.coordinator)}
        for index, device in enumerate(runtime.RTK[:MAX_DIAGNOSTIC_DEVICES], start=1)
    ]
    spino_devices = [
        {"index": index, "coordinator": _coordinator_status(device.coordinator)}
        for index, device in enumerate(runtime.spino[:MAX_DIAGNOSTIC_DEVICES], start=1)
    ]

    return {
        "integration": {
            "domain": DOMAIN,
            "version": integration.version,
            "entry_state": entry.state.value,
            "connection_mode": connection_mode,
        },
        "device_counts": {
            "mowers": len(runtime.mowers),
            "rtk": len(runtime.RTK),
            "spino": len(runtime.spino),
        },
        "devices_truncated": any(
            count > MAX_DIAGNOSTIC_DEVICES
            for count in (len(runtime.mowers), len(runtime.RTK), len(runtime.spino))
        ),
        "mowers": mowers,
        "rtk": rtk_devices,
        "spino": spino_devices,
    }
