"""Update entity for Mammotion."""

import logging
from dataclasses import dataclass
from typing import Any

from homeassistant.components.update import (
    UpdateDeviceClass,
    UpdateEntity,
    UpdateEntityDescription,
    UpdateEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import MammotionRTKCoordinator
from .coordinator import MammotionBaseUpdateCoordinator, MammotionSpinoCoordinator
from .entity import (
    MammotionBaseEntity,
    MammotionBaseRTKEntity,
    MammotionBaseSpinoEntity,
)

LOGGER = logging.getLogger(__name__)
MIN_FIRMWARE_UPDATE_BATTERY_PERCENT = 30


def _enum_name(value: Any) -> str | None:
    """Return a stable enum name without inventing one for raw unknowns."""
    name = getattr(value, "name", None)
    return str(name) if name is not None else None


def _firmware_install_readiness(  # noqa: C901
    coordinator: Any,
    *,
    installed_version: str | None,
    target_version: str | None,
    hardware_kind: str = "mower",
) -> dict[str, Any]:
    """Return fail-closed firmware prerequisites from fresh coordinator data."""
    blockers: list[str] = []
    if getattr(coordinator, "last_update_success", None) is not True:
        blockers.append("fresh_state_unavailable")
    if hardware_kind != "mower":
        blockers.append(f"{hardware_kind}_firmware_acceptance_missing")
        return {
            "allowed": False,
            "blockers": blockers,
            "hardware_kind": hardware_kind,
        }

    device_name = str(getattr(coordinator, "device_name", "")).upper()
    if not device_name.startswith(("LUBA", "YUKA")):
        blockers.append("model_support_unknown")
    if getattr(coordinator, "has_cloud_account", None) is not True:
        blockers.append("cloud_account_required")

    try:
        online = coordinator.is_online()
    except AttributeError, TypeError:
        online = None
    if online is not True:
        blockers.append("device_online_state_unknown")

    data = getattr(coordinator, "data", None)
    report = getattr(data, "report_data", None)
    dev = getattr(report, "dev", None)
    connect = getattr(report, "connect", None)
    work_mode = getattr(dev, "sys_status", None)
    work_mode_name = _enum_name(work_mode)
    if work_mode_name not in {"MODE_READY", "MODE_PAUSE"}:
        blockers.append("device_not_idle")

    charge_state = getattr(dev, "charge_state", None)
    charge_name = (_enum_name(charge_state) or "").lower()
    charging = (
        "charging" in charge_name and "not_charging" not in charge_name
        if charge_name
        else None
    )
    if charging is not True:
        blockers.append("device_not_confirmed_charging")

    battery = getattr(dev, "battery_val", None)
    if (
        not isinstance(battery, int | float)
        or battery < MIN_FIRMWARE_UPDATE_BATTERY_PERCENT
    ):
        blockers.append("battery_below_or_unknown")

    wifi_rssi = getattr(connect, "wifi_rssi", None)
    if not isinstance(wifi_rssi, int | float) or wifi_rssi == 0:
        blockers.append("wifi_link_unknown")
    if not installed_version or not target_version:
        blockers.append("version_pair_incomplete")
    elif installed_version == target_version:
        blockers.append("target_version_already_installed")
    return {
        "allowed": not blockers,
        "blockers": blockers,
        "hardware_kind": hardware_kind,
        "installed_version": installed_version,
        "target_version": target_version,
        "work_mode": work_mode_name,
        "charging": charging,
        "battery_percent": battery,
        "wifi_rssi": wifi_rssi,
    }


def _raise_firmware_install_blocked(readiness: dict[str, Any]) -> None:
    """Raise an actionable HA error without starting an unsafe update."""
    if readiness["allowed"]:
        return
    raise HomeAssistantError(
        "Firmware install blocked: " + ", ".join(readiness["blockers"])
    )


@dataclass(frozen=True, kw_only=True)
class MammotionUpdateEntityDescription(UpdateEntityDescription):
    """Describes Mammotion switch entity."""

    key: str


MammotionUpdate = MammotionUpdateEntityDescription(
    key="update",
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up update entities for Netgear component."""
    mammotion_devices = entry.runtime_data.mowers
    entities: list[UpdateEntity] = []
    for mower in mammotion_devices:
        entities.append(
            MammotionUpdateEntity(mower.version_coordinator, MammotionUpdate)
        )

    mammotion_rtks = entry.runtime_data.RTK
    for rtk in mammotion_rtks:
        entities.append(MammotionRTKUpdateEntity(rtk.coordinator, MammotionUpdate))

    mammotion_spinos = entry.runtime_data.spino
    for spino in mammotion_spinos:
        entities.append(MammotionSpinoUpdateEntity(spino.coordinator, MammotionUpdate))

    async_add_entities(entities)


class MammotionUpdateEntity(MammotionBaseEntity, UpdateEntity):
    """Update entity for a Netgear device."""

    entity_description: MammotionUpdateEntityDescription

    _attr_device_class = UpdateDeviceClass.FIRMWARE
    _attr_supported_features = (
        UpdateEntityFeature.INSTALL
        | UpdateEntityFeature.RELEASE_NOTES
        | UpdateEntityFeature.PROGRESS
    )
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: MammotionBaseUpdateCoordinator,
        entity_description: MammotionUpdateEntityDescription,
    ) -> None:
        """Initialize a Netgear device."""
        super().__init__(coordinator, entity_description.key)
        self.coordinator = coordinator
        self.entity_description = entity_description
        self._attr_translation_key = entity_description.key

    @property
    def installed_version(self) -> str | None:
        """Version currently in use."""
        if self.coordinator.data is not None:
            return self.coordinator.data.device_firmwares.device_version
        return None

    @property
    def latest_version(self) -> str | None:
        """Latest version available for install."""
        if (
            self.coordinator.data.update_check.upgradeable
            and self.coordinator.data.update_check.product_version_info_vo is not None
        ):
            new_version = self.coordinator.data.update_check.product_version_info_vo
            return new_version.release_version
        return self.installed_version

    @property
    def release_summary(self) -> str | None:
        """Release summary."""
        if self.coordinator.data.update_check.product_version_info_vo is not None:
            return (
                self.coordinator.data.update_check.product_version_info_vo.release_note
            )
        return None

    def release_notes(self) -> str | None:
        """Release notes."""
        if self.coordinator.data.update_check.product_version_info_vo is not None:
            return (
                self.coordinator.data.update_check.product_version_info_vo.release_note
            )
        return None

    @property
    def in_progress(self) -> bool:
        """Update installation in progress."""
        return bool(self.coordinator.data.update_check.isupgrading)

    @property
    def update_percentage(self) -> int | float | None:
        """Update installation progress percentage."""
        if self.coordinator.data.update_check.isupgrading:
            return self.coordinator.data.update_check.progress
        return None

    async def async_install(
        self, version: str | None, backup: bool, **kwargs: Any
    ) -> None:
        """Install the latest firmware version."""
        await self.coordinator.async_refresh()
        if version is None:
            version = self.latest_version
        _raise_firmware_install_blocked(
            _firmware_install_readiness(
                self.coordinator,
                installed_version=self.installed_version,
                target_version=version,
            )
        )
        if version:
            await self.coordinator.update_firmware(version)
        await self.coordinator.async_refresh()

    @callback
    def async_update_device(self) -> None:
        """Update the Mammotion device."""


class MammotionRTKUpdateEntity(MammotionBaseRTKEntity, UpdateEntity):
    """Update entity for a Netgear device."""

    entity_description: MammotionUpdateEntityDescription

    _attr_device_class = UpdateDeviceClass.FIRMWARE
    _attr_supported_features = (
        UpdateEntityFeature.INSTALL
        | UpdateEntityFeature.RELEASE_NOTES
        | UpdateEntityFeature.PROGRESS
    )

    def __init__(
        self,
        coordinator: MammotionRTKCoordinator,
        entity_description: MammotionUpdateEntityDescription,
    ) -> None:
        """Initialize a Netgear device."""
        super().__init__(coordinator, entity_description.key)
        self.coordinator = coordinator
        self.entity_description = entity_description
        self._attr_translation_key = entity_description.key

    @property
    def installed_version(self) -> str | None:
        """Version currently in use."""
        if self.coordinator.data is not None:
            return self.coordinator.data.device_version
        return None

    @property
    def latest_version(self) -> str | None:
        """Latest version available for install."""
        if (
            self.coordinator.data.update_check.upgradeable
            and self.coordinator.data.update_check.product_version_info_vo is not None
        ):
            new_version = self.coordinator.data.update_check.product_version_info_vo
            return new_version.release_version
        return self.installed_version

    @property
    def release_summary(self) -> str | None:
        """Release summary."""
        if self.coordinator.data.update_check.product_version_info_vo is not None:
            return (
                self.coordinator.data.update_check.product_version_info_vo.release_note
            )
        return None

    def release_notes(self) -> str | None:
        """Release notes."""
        if self.coordinator.data.update_check.product_version_info_vo is not None:
            return (
                self.coordinator.data.update_check.product_version_info_vo.release_note
            )
        return None

    @property
    def in_progress(self) -> bool:
        """Update installation in progress."""
        return bool(self.coordinator.data.update_check.isupgrading)

    @property
    def update_percentage(self) -> int | float | None:
        """Update installation progress percentage."""
        if self.coordinator.data.update_check.isupgrading:
            return self.coordinator.data.update_check.progress
        return None

    async def async_install(
        self, version: str | None, backup: bool, **kwargs: Any
    ) -> None:
        """Install the latest firmware version."""
        await self.coordinator.async_refresh()
        if version is None:
            version = self.latest_version
        _raise_firmware_install_blocked(
            _firmware_install_readiness(
                self.coordinator,
                installed_version=self.installed_version,
                target_version=version,
                hardware_kind="rtk",
            )
        )
        if version:
            await self.coordinator.update_firmware(version)
        await self.coordinator.async_refresh()

    @callback
    def async_update_device(self) -> None:
        """Update the Mammotion device."""


class MammotionSpinoUpdateEntity(MammotionBaseSpinoEntity, UpdateEntity):
    """Update entity for a Mammotion Spino pool cleaner."""

    entity_description: MammotionUpdateEntityDescription

    _attr_device_class = UpdateDeviceClass.FIRMWARE
    _attr_supported_features = (
        UpdateEntityFeature.INSTALL
        | UpdateEntityFeature.RELEASE_NOTES
        | UpdateEntityFeature.PROGRESS
    )

    def __init__(
        self,
        coordinator: MammotionSpinoCoordinator,
        entity_description: MammotionUpdateEntityDescription,
    ) -> None:
        """Initialize a Spino update entity."""
        super().__init__(coordinator, entity_description.key)
        self.coordinator = coordinator
        self.entity_description = entity_description
        self._attr_translation_key = entity_description.key

    @property
    def installed_version(self) -> str | None:
        """Version currently in use."""
        if self.coordinator.data is not None:
            return self.coordinator.data.device_firmwares.device_version
        return None

    @property
    def latest_version(self) -> str | None:
        """Latest version available for install."""
        if (
            self.coordinator.data.update_check.upgradeable
            and self.coordinator.data.update_check.product_version_info_vo is not None
        ):
            new_version = self.coordinator.data.update_check.product_version_info_vo
            return new_version.release_version
        return self.installed_version

    @property
    def release_summary(self) -> str | None:
        """Release summary."""
        if self.coordinator.data.update_check.product_version_info_vo is not None:
            return (
                self.coordinator.data.update_check.product_version_info_vo.release_note
            )
        return None

    def release_notes(self) -> str | None:
        """Release notes."""
        if self.coordinator.data.update_check.product_version_info_vo is not None:
            return (
                self.coordinator.data.update_check.product_version_info_vo.release_note
            )
        return None

    @property
    def in_progress(self) -> bool:
        """Update installation in progress."""
        return bool(self.coordinator.data.update_check.isupgrading)

    @property
    def update_percentage(self) -> int | float | None:
        """Update installation progress percentage."""
        if self.coordinator.data.update_check.isupgrading:
            return self.coordinator.data.update_check.progress
        return None

    async def async_install(
        self, version: str | None, backup: bool, **kwargs: Any
    ) -> None:
        """Install the latest firmware version."""
        await self.coordinator.async_refresh()
        if version is None:
            version = self.latest_version
        _raise_firmware_install_blocked(
            _firmware_install_readiness(
                self.coordinator,
                installed_version=self.installed_version,
                target_version=version,
                hardware_kind="spino",
            )
        )
        if version:
            await self.coordinator.update_firmware(version)
        await self.coordinator.async_refresh()

    @callback
    def async_update_device(self) -> None:
        """Update the Mammotion device."""
