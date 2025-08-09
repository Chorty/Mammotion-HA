"""Expose mower as a vacuum entity for map card compatibility."""

from __future__ import annotations

from homeassistant.components.lawn_mower import LawnMowerActivity
from homeassistant.components.vacuum import (
    StateVacuumEntity,
    VacuumActivity,
    VacuumEntityFeature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import MammotionConfigEntry
from .coordinator import MammotionReportUpdateCoordinator
from .lawn_mower import MammotionLawnMowerEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: MammotionConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Mammotion vacuum entities."""
    mowers = entry.runtime_data
    entities = [MammotionVacuumEntity(mower.reporting_coordinator) for mower in mowers]
    async_add_entities(entities)


class MammotionVacuumEntity(MammotionLawnMowerEntity, StateVacuumEntity):
    """Representation of the mower as a vacuum."""

    _attr_supported_features = (
        VacuumEntityFeature.START
        | VacuumEntityFeature.PAUSE
        | VacuumEntityFeature.STOP
        | VacuumEntityFeature.RETURN_HOME
    )

    def __init__(self, coordinator: MammotionReportUpdateCoordinator) -> None:
        """Initialize the vacuum entity."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.device_name}_vacuum"
        self._attr_translation_key = "vacuum"
        self._attr_icon = "mdi:mower"

    @property
    def activity(self) -> VacuumActivity | None:
        """Map mower activity to vacuum activity."""
        mower_activity = super().activity
        if mower_activity is None:
            return None
        if mower_activity == LawnMowerActivity.MOWING:
            return VacuumActivity.CLEANING
        if mower_activity == LawnMowerActivity.RETURNING:
            return VacuumActivity.RETURNING
        if mower_activity == LawnMowerActivity.PAUSED:
            return VacuumActivity.PAUSED
        if mower_activity == LawnMowerActivity.DOCKED:
            return VacuumActivity.DOCKED
        if mower_activity == LawnMowerActivity.ERROR:
            return VacuumActivity.ERROR
        return VacuumActivity.IDLE

    async def async_start(self) -> None:
        """Start mowing."""
        await self.async_start_mowing()

    async def async_stop(self) -> None:
        """Cancel the current job."""
        await self.async_cancel()

    async def async_pause(self) -> None:
        """Pause the mower."""
        await super().async_pause()

    async def async_return_to_base(self) -> None:
        """Send mower to the dock."""
        await self.async_dock()
