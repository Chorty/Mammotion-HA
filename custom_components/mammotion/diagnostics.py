"""Privacy-preserving diagnostics support for Mammotion."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.loader import async_get_integration

from . import MammotionConfigEntry
from .const import CONF_BLE_DEVICES, CONF_HAS_CLOUD_ACCOUNT, CONF_USE_WIFI, DOMAIN

MAX_DIAGNOSTIC_DEVICES = 20

# Fields from the mower's ``deviceOtherInfo`` health payload that are safe and
# useful to surface in diagnostics. Deliberately a whitelist: the payload also
# carries values that are noisy, meaningless without device internals, or mild
# fingerprinting risks, and none of these name a location, network, account, or
# credential. Populated only once pymammotion retains the typed snapshot
# (Chorty/PyMammotion feat/retain-device-other-info); on an older wheel the
# attribute is absent and the whole section is simply omitted.
_DEVICE_HEALTH_FIELDS: tuple[str, ...] = (
    # crash / stability
    "soc_coredump",
    "embed_coredump",
    "nav_coredump",
    "perception_coredump",
    "location_coredump",
    "other_coredump",
    "process_restart_count",
    "usb_dis_cnt",
    # resource pressure
    "soc_tmp",
    "soc_mem_free",
    "soc_mem_total",
    "soc_mmc_life_time",
    "soc_up_time",
    "mcu_up_time",
    # subsystem health strings
    "nav",
    "ins_fusion",
    "perception",
    "vision_proxy",
    "vslam_vio",
    "chassis_state",
    # connectivity counters (counts only, never URLs/credentials)
    "iot_con",
    "iot_con_timeout",
    "mqtt_conn_cnt",
    "mqtt_disconn_cnt",
    "mqtt_rtk_status",
    "rtk_status",
    "lora_connect",
)


def _device_health(mowing_device: Any) -> dict[str, Any] | None:
    """Return a bounded, whitelisted view of the mower's deviceOtherInfo health.

    Returns ``None`` when the running pymammotion does not retain the typed
    ``device_other_info`` snapshot (older wheel), or when nothing has been
    reported yet, so the diagnostics output stays clean on both.
    """
    other_info = getattr(mowing_device, "device_other_info", None)
    if other_info is None:
        return None
    health = {
        field: value
        for field in _DEVICE_HEALTH_FIELDS
        if (value := getattr(other_info, field, None)) is not None
    }
    return health or None


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

    mowers = []
    for index, device in enumerate(runtime.mowers[:MAX_DIAGNOSTIC_DEVICES], start=1):
        entry_data: dict[str, Any] = {
            "index": index,
            "reporting": _coordinator_status(device.reporting_coordinator),
            "maintenance": _coordinator_status(device.maintenance_coordinator),
            "firmware": _coordinator_status(device.version_coordinator),
            "map": _coordinator_status(device.map_coordinator),
            "errors": _coordinator_status(device.error_coordinator),
        }
        health = _device_health(device.reporting_coordinator.data)
        if health is not None:
            entry_data["health"] = health
        mowers.append(entry_data)
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
