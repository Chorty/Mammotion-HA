"""Diagnostics support for Mammotion."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from . import MammotionConfigEntry, MammotionMowerData


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant,
    entry: MammotionConfigEntry,
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    mammotion_devices: list[MammotionMowerData] = entry.runtime_data
    devices: list[dict[str, Any]] = []
    for mower_data in mammotion_devices:
        state = mower_data.reporting_coordinator.data
        mixed_device = mower_data.api.get_device_by_name(mower_data.name)
        devices.append(
            {
                "model": mower_data.device.productModel,
                "product_key": mower_data.device.productKey,
                "firmware_version": state.device_firmwares.device_version,
                "connection_preference": str(mixed_device.preference),
                "online": bool(state.online),
                "enabled": bool(state.enabled),
                "coordinators": {
                    "report": {
                        "last_update_success": mower_data.reporting_coordinator.last_update_success,
                        "update_failures": mower_data.reporting_coordinator.update_failures,
                    },
                    "maintenance": {
                        "last_update_success": mower_data.maintenance_coordinator.last_update_success,
                    },
                    "version": {
                        "last_update_success": mower_data.version_coordinator.last_update_success,
                    },
                    "map": {
                        "last_update_success": mower_data.map_coordinator.last_update_success,
                    },
                },
            }
        )

    return {
        "integration": {
            "domain": entry.domain,
            "version": "0.2.74",
            "device_count": len(devices),
        },
        "devices": devices,
    }
