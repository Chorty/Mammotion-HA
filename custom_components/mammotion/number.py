"""Number entities for the Mammotion integration."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from homeassistant.components.number import (
    NumberDeviceClass,
    NumberEntity,
    NumberEntityDescription,
    NumberMode,
    RestoreNumber,
)
from homeassistant.const import (
    DEGREE,
    PERCENTAGE,
    UnitOfArea,
    UnitOfLength,
    UnitOfSpeed,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from pymammotion.data.model.device import PoolCleanerDevice
from pymammotion.data.model.device_limits import DeviceLimits
from pymammotion.utility.device_config import DeviceConfig
from pymammotion.utility.device_type import DeviceType

from . import MammotionConfigEntry
from .coordinator import MammotionBaseUpdateCoordinator, MammotionSpinoCoordinator
from .entity import MammotionBaseEntity, MammotionBaseSpinoEntity


@dataclass(frozen=True, kw_only=True)
class MammotionConfigNumberEntityDescription(NumberEntityDescription):
    """Describes Mammotion number entity."""

    set_fn: Callable[[MammotionBaseUpdateCoordinator[Any], float], None] | None = None
    set_async_fn: (
        Callable[[MammotionBaseUpdateCoordinator[Any], float], Awaitable[None]] | None
    ) = None
    get_fn: Callable[[MammotionBaseUpdateCoordinator[Any]], float | None] | None = None


@dataclass(frozen=True, kw_only=True)
class MammotionSpinoNumberEntityDescription(NumberEntityDescription):
    """Describes a Mammotion Spino pool cleaner number entity."""

    value_fn: Callable[[PoolCleanerDevice], float]
    set_fn: Callable[[MammotionSpinoCoordinator, float], Awaitable[None]]


SPINO_NUMBER_ENTITIES: tuple[MammotionSpinoNumberEntityDescription, ...] = (
    MammotionSpinoNumberEntityDescription(
        key="spino_floor_speed",
        device_class=NumberDeviceClass.SPEED,
        native_unit_of_measurement=UnitOfSpeed.METERS_PER_SECOND,
        native_min_value=0.1,
        native_max_value=0.2,
        native_step=0.01,
        mode=NumberMode.SLIDER,
        entity_category=EntityCategory.CONFIG,
        value_fn=lambda spino_data: spino_data.pool_state.floor_speed,
        set_fn=lambda coordinator, value: coordinator.async_set_floor_speed(value),
    ),
)


MAP_OFFSET_ENTITIES: tuple[MammotionConfigNumberEntityDescription, ...] = (
    MammotionConfigNumberEntityDescription(
        key="map_offset_lat",
        device_class=NumberDeviceClass.DISTANCE,
        native_unit_of_measurement=UnitOfLength.METERS,
        native_step=0.1,
        native_min_value=-50,
        native_max_value=50,
        mode=NumberMode.BOX,
        set_fn=lambda coordinator, value: setattr(coordinator, "map_offset_lat", value),
        get_fn=lambda coordinator: coordinator.map_offset_lat,
    ),
    MammotionConfigNumberEntityDescription(
        key="map_offset_lon",
        device_class=NumberDeviceClass.DISTANCE,
        native_unit_of_measurement=UnitOfLength.METERS,
        native_step=0.1,
        native_min_value=-50,
        native_max_value=50,
        mode=NumberMode.BOX,
        set_fn=lambda coordinator, value: setattr(coordinator, "map_offset_lon", value),
        get_fn=lambda coordinator: coordinator.map_offset_lon,
    ),
)

AUDIO_NUMBER_ENTITIES: tuple[MammotionConfigNumberEntityDescription, ...] = (
    MammotionConfigNumberEntityDescription(
        key="prompt_volume",
        native_min_value=0,
        native_max_value=100,
        native_step=1,
        mode=NumberMode.SLIDER,
        native_unit_of_measurement=PERCENTAGE,
        set_async_fn=lambda coordinator, value: coordinator.async_set_prompt_volume(
            value
        ),
        get_fn=lambda coordinator: getattr(
            coordinator.data.mower_state.audio,
            "au_switch",
            getattr(coordinator.data.mower_state.audio, "volume", None),
        ),
    ),
    MammotionConfigNumberEntityDescription(
        key="voice_volume",
        native_min_value=0,
        native_max_value=100,
        native_step=1,
        mode=NumberMode.SLIDER,
        native_unit_of_measurement=PERCENTAGE,
        set_async_fn=lambda coordinator, value: coordinator.async_set_voice_volume(
            value
        ),
        get_fn=lambda coordinator: coordinator.data.mower_state.audio.volume,
    ),
)

NUMBER_ENTITIES: tuple[MammotionConfigNumberEntityDescription, ...] = (
    MammotionConfigNumberEntityDescription(
        key="start_progress",
        native_min_value=0,
        native_max_value=100,
        native_step=1,
        mode=NumberMode.SLIDER,
        native_unit_of_measurement=PERCENTAGE,
        set_fn=lambda coordinator, value: setattr(
            coordinator.operation_settings, "start_progress", value
        ),
    ),
    MammotionConfigNumberEntityDescription(
        key="cutting_angle",
        native_step=1,
        native_unit_of_measurement=DEGREE,
        native_min_value=-180,
        native_max_value=180,
        set_fn=lambda coordinator, value: setattr(
            coordinator.operation_settings, "toward", value
        ),
    ),
    MammotionConfigNumberEntityDescription(
        key="toward_included_angle",
        native_step=1,
        native_unit_of_measurement=DEGREE,
        native_min_value=-180,
        native_max_value=180,
        set_fn=lambda coordinator, value: setattr(
            coordinator.operation_settings, "toward_included_angle", value
        ),
    ),
)

YUKA_NUMBER_ENTITIES: tuple[MammotionConfigNumberEntityDescription, ...] = (
    MammotionConfigNumberEntityDescription(
        key="dumping_interval",
        native_min_value=5,
        native_max_value=100,
        native_step=1,
        mode=NumberMode.SLIDER,
        native_unit_of_measurement=UnitOfArea.SQUARE_METERS,
        set_fn=lambda coordinator, value: setattr(
            coordinator.operation_settings, "collect_grass_frequency", value
        ),
    ),
)

LUBA_WORKING_ENTITIES: tuple[MammotionConfigNumberEntityDescription, ...] = (
    MammotionConfigNumberEntityDescription(
        key="blade_height",
        device_class=NumberDeviceClass.DISTANCE,
        native_unit_of_measurement=UnitOfLength.MILLIMETERS,
        native_step=1,
        native_min_value=25,
        native_max_value=70,
        mode=NumberMode.SLIDER,
        set_fn=lambda coordinator, value: setattr(
            coordinator.operation_settings, "blade_height", int(value)
        ),
        set_async_fn=lambda coordinator, value: coordinator.async_apply_working_setting(
            "blade_height"
        ),
        get_fn=lambda coordinator: coordinator.working_setting_value("blade_height"),
    ),
)


NUMBER_WORKING_ENTITIES: tuple[MammotionConfigNumberEntityDescription, ...] = (
    MammotionConfigNumberEntityDescription(
        key="working_speed",
        device_class=NumberDeviceClass.SPEED,
        native_unit_of_measurement=UnitOfSpeed.METERS_PER_SECOND,
        native_step=0.1,
        native_min_value=0.2,
        native_max_value=0.6,
        set_async_fn=lambda coordinator, value: coordinator.async_apply_working_setting(
            "speed"
        ),
        set_fn=lambda coordinator, value: setattr(
            coordinator.operation_settings, "speed", value
        ),
        get_fn=lambda coordinator: round(coordinator.working_setting_value("speed"), 2),
    ),
    MammotionConfigNumberEntityDescription(
        key="path_spacing",
        native_step=1,
        device_class=NumberDeviceClass.DISTANCE,
        native_unit_of_measurement=UnitOfLength.CENTIMETERS,
        native_min_value=20,
        native_max_value=35,
        # Not applied to a running job: the app's in-job editor does not send
        # it either. A change here waits for the next job HA plans.
        set_fn=lambda coordinator, value: setattr(
            coordinator.operation_settings, "channel_width", value
        ),
        get_fn=lambda coordinator: coordinator.working_setting_value("channel_width"),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: MammotionConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Mammotion number entities."""
    mammotion_devices = entry.runtime_data.mowers

    for mower in mammotion_devices:
        limits: DeviceLimits | None = DeviceConfig().get_working_parameters(
            mower.device.product_key
        )
        if handle := mower.api.get_device_by_name(mower.name):
            limits = handle.device_limits
        entities: list[MammotionConfigNumberEntity] = []

        for entity_description in NUMBER_WORKING_ENTITIES:
            entities.append(
                MammotionWorkingNumberEntity(
                    mower.reporting_coordinator, entity_description, limits
                )
            )

        if DeviceType.is_luba_pro(mower.device.device_name):
            for entity_description in AUDIO_NUMBER_ENTITIES:
                entities.append(
                    MammotionConfigNumberEntity(
                        mower.reporting_coordinator, entity_description
                    )
                )

        for entity_description in MAP_OFFSET_ENTITIES:
            entities.append(
                MammotionConfigNumberEntity(
                    mower.reporting_coordinator, entity_description
                )
            )

        for entity_description in NUMBER_ENTITIES:
            entities.append(
                MammotionConfigNumberEntity(
                    mower.reporting_coordinator, entity_description
                )
            )

        if DeviceType.is_yuka(mower.device.device_name) and not DeviceType.is_yuka_mini(
            mower.device.device_name
        ):
            for entity_description in YUKA_NUMBER_ENTITIES:
                entities.append(
                    MammotionConfigNumberEntity(
                        mower.reporting_coordinator, entity_description
                    )
                )
        if not DeviceType.is_yuka(mower.device.device_name):
            for entity_description in LUBA_WORKING_ENTITIES:
                entities.append(
                    MammotionWorkingNumberEntity(
                        mower.reporting_coordinator, entity_description, limits
                    )
                )

        async_add_entities(entities)

    for spino in entry.runtime_data.spino:
        async_add_entities(
            MammotionSpinoNumberEntity(spino.coordinator, entity_description)
            for entity_description in SPINO_NUMBER_ENTITIES
        )


class MammotionConfigNumberEntity(MammotionBaseEntity, RestoreNumber):
    """Mammotion config number entity."""

    entity_description: MammotionConfigNumberEntityDescription
    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.CONFIG

    def __init__(
        self,
        coordinator: MammotionBaseUpdateCoordinator[Any],
        entity_description: MammotionConfigNumberEntityDescription,
    ) -> None:
        """Initialize the config number entity."""
        super().__init__(coordinator, entity_description.key)
        self.entity_description = entity_description
        self._attr_translation_key = entity_description.key
        if entity_description.native_min_value is not None:
            self._attr_native_min_value = entity_description.native_min_value
            self._attr_native_value = entity_description.native_min_value
        if entity_description.native_max_value is not None:
            self._attr_native_max_value = entity_description.native_max_value
        if entity_description.native_step is not None:
            self._attr_native_step = entity_description.native_step
        if self.entity_description.native_unit_of_measurement == DEGREE:
            self._attr_native_value = 0
        if self.entity_description.key == "toward_included_angle":
            self._attr_native_value = 90
        if self.entity_description.get_fn is not None:
            self._attr_native_value = self.entity_description.get_fn(self.coordinator)
        elif (
            self.entity_description.set_fn is not None
            and self._attr_native_value is not None
        ):
            self.entity_description.set_fn(self.coordinator, self._attr_native_value)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        if self.entity_description.get_fn is not None:
            self._attr_native_value = self.entity_description.get_fn(self.coordinator)
        super()._handle_coordinator_update()

    async def async_set_native_value(self, value: float) -> None:
        """Set native value for number."""
        self._attr_native_value = value
        if self.entity_description.set_fn is not None:
            self.entity_description.set_fn(self.coordinator, value)
        if self.entity_description.set_async_fn is not None:
            await self.entity_description.set_async_fn(self.coordinator, value)
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Restore last saved value when entity is added to hass."""
        await super().async_added_to_hass()
        last_number_data = await self.async_get_last_number_data()
        if (last_number_data is not None) and (
            last_number_data.native_value is not None
        ):
            self._attr_native_value = last_number_data.native_value
            if self.entity_description.set_fn is not None:
                self.entity_description.set_fn(
                    self.coordinator, self._attr_native_value
                )


class MammotionWorkingNumberEntity(MammotionConfigNumberEntity):
    """Mammotion working number entity.

    Shows the running job's value once HA has read it from the mower, and HA's
    own next-job plan otherwise; the ``value_source`` attribute says which.
    """

    def __init__(
        self,
        coordinator: MammotionBaseUpdateCoordinator[Any],
        entity_description: MammotionConfigNumberEntityDescription,
        limits: DeviceLimits | None,
    ) -> None:
        """Init MammotionWorkingNumberEntity."""
        super().__init__(coordinator, entity_description)

        if limits is not None and hasattr(limits, entity_description.key):
            # float(), deliberately: HA derives the precision of the DISPLAYED
            # min/max from the decimal count of the native value's string
            # (number/__init__.py `_convert_to_state_value`). An int 25 mm floors
            # to "0.0 in"; 25.0 floors to "0.9 in", which is the real limit.
            self._attr_native_min_value = float(
                getattr(limits, entity_description.key).min
            )
            self._attr_native_max_value = float(
                getattr(limits, entity_description.key).max
            )
        elif (
            entity_description.native_min_value is not None
            and entity_description.native_max_value is not None
        ):
            self._attr_native_min_value = entity_description.native_min_value
            self._attr_native_max_value = entity_description.native_max_value

        if self.entity_description.get_fn is not None:
            self._attr_native_value = self.entity_description.get_fn(self.coordinator)

        self._clamp_plan_value()

    def _clamped(self, value: float) -> float:
        """Return *value* held inside the model's native limits."""
        if (native_min := self._attr_native_min_value) is not None:
            value = max(value, native_min)
        if (native_max := self._attr_native_max_value) is not None:
            value = min(value, native_max)
        return value

    def _clamp_plan_value(self) -> None:
        """Keep HA's plan inside the model's limits, in the plan and not just shown.

        The plan is what HA sends when it plans a job, so a value clamped only
        for display (pymammotion's default blade_height 0 shown as 25) would
        still be sent as 0. A running job's value is never rewritten here.
        """
        value = self._attr_native_value
        if value is None:
            return
        clamped = self._clamped(value)
        if clamped == value:
            return
        self._attr_native_value = clamped
        if (
            self.entity_description.set_fn is not None
            and self.coordinator.working_setting_source() == "next_job_plan"
        ):
            self.entity_description.set_fn(self.coordinator, clamped)

    @property
    def native_step(self) -> float | None:
        """Return a step that makes sense in the unit actually being shown.

        🚨 HA never converts the step (`_calculate_step`), so this entity's 1 mm
        step reads as 1 INCH once the operator picks US units: a 25-70 mm blade
        range then offers 0, 1, 2 and 3 on the slider and nothing between.
        Reported by the operator 2026-09-18. In a converted unit, step by a
        tenth instead -- 0.1 in is 2.5 mm, finer than the 5 mm the device is
        observed to quantise to.
        """
        native_unit = self.entity_description.native_unit_of_measurement
        if native_unit is not None and self.unit_of_measurement != native_unit:
            return 0.1
        return self.entity_description.native_step

    @property
    def extra_state_attributes(self) -> dict[str, str]:
        """Say whether the value is the running job's or HA's next-job plan."""
        return {"value_source": self.coordinator.working_setting_source()}

    async def async_added_to_hass(self) -> None:
        """Restore the last plan value, then keep it inside the model's limits."""
        await super().async_added_to_hass()
        self._clamp_plan_value()

    @property
    def native_min_value(self) -> float:
        """Return the minimum value."""
        min_value = self._attr_native_min_value
        if min_value is None:
            raise ValueError("native min value is unavailable")
        return min_value

    @property
    def native_max_value(self) -> float:
        """Return the maximum value."""
        max_value = self._attr_native_max_value
        if max_value is None:
            raise ValueError("native max value is unavailable")
        return max_value

    async def async_set_native_value(self, value: float) -> None:
        """Set native value for number and call update_fn if defined.

        The value is clamped to the model's own limits first. HA validates the
        number the operator picked against the DISPLAYED range and converts it
        afterwards, and the displayed maximum is rounded outward: 2.8 in comes
        back as 71.1 mm against a 70 mm device limit.
        """
        value = self._clamped(value)
        if self._attr_native_value == value:
            return
        self._attr_native_value = value
        if self.entity_description.set_fn is not None:
            self.entity_description.set_fn(self.coordinator, value)
        if self.entity_description.set_async_fn is not None:
            await self.entity_description.set_async_fn(self.coordinator, value)
        self.async_write_ha_state()


class MammotionSpinoNumberEntity(MammotionBaseSpinoEntity, NumberEntity):
    """Mammotion Spino pool cleaner number entity."""

    entity_description: MammotionSpinoNumberEntityDescription

    def __init__(
        self,
        coordinator: MammotionSpinoCoordinator,
        entity_description: MammotionSpinoNumberEntityDescription,
    ) -> None:
        """Initialize the Spino number entity."""
        super().__init__(coordinator, entity_description.key)
        self.entity_description = entity_description
        self._attr_translation_key = entity_description.key

    @property
    def native_value(self) -> float:
        """Return the current value."""
        return self.entity_description.value_fn(self.coordinator.data)

    async def async_set_native_value(self, value: float) -> None:
        """Set a new value."""
        await self.entity_description.set_fn(self.coordinator, value)
        await self.coordinator.async_request_refresh()
