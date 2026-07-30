"""Mammotion binary sensor entities."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from pymammotion.data.model.device import MowingDevice

from . import MammotionConfigEntry
from .coordinator import MammotionBaseUpdateCoordinator
from .entity import MammotionBaseEntity
from .services import motion_gate_snapshot


@dataclass(frozen=True, kw_only=True)
class MammotionBinarySensorEntityDescription(
    BinarySensorEntityDescription,
):
    """Describes Mammotion binary sensor entity.

    Exactly one of ``is_on_fn`` (reads device data) or ``gate_key`` (reads the
    cached motion-gate snapshot) applies. ``attributes_keys`` copies extra keys
    from that snapshot onto the entity so a blocked gate says *why* without
    needing a service call.
    """

    is_on_fn: Callable[[MowingDevice], bool | None] | None = None
    gate_key: str | None = None
    attributes_keys: tuple[str, ...] = ()


BINARY_SENSORS: tuple[MammotionBinarySensorEntityDescription, ...] = (
    MammotionBinarySensorEntityDescription(
        key="charging",
        device_class=BinarySensorDeviceClass.BATTERY_CHARGING,
        is_on_fn=lambda mower_data: mower_data.report_data.dev.charge_state in (1, 2),
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    # The standing motion gate. "Ready" means nothing is currently in the way --
    # not that a dispatch would be accepted, since the per-call operator
    # confirmations and the busy/saga checks are only evaluated by the service.
    MammotionBinarySensorEntityDescription(
        key="real_motion_ready",
        gate_key="real_motion_ready",
        attributes_keys=("blockers",),
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    # ble_rssi is NOT a liveness signal -- it is self-reported by the mower and
    # has read -64 while no advertisement arrived for 10 minutes. Until now
    # nothing surfaced whether BLE was actually usable for a confirmed write.
    MammotionBinarySensorEntityDescription(
        key="ble_link_live",
        gate_key="ble_link_live",
        attributes_keys=("ble_link_reason",),
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    # Proof the loaded backend carries the audited BLE fixes. Rarely changes,
    # but it makes an accidental pin regression visible immediately.
    MammotionBinarySensorEntityDescription(
        key="motion_backend_verified",
        gate_key="backend_verified",
        attributes_keys=("backend_capabilities",),
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    # Combines blade state and cutter RPM. The two can disagree: RPM latches at
    # its last running value after a mow while the state reads OFF.
    MammotionBinarySensorEntityDescription(
        key="blade_safe_for_motion",
        gate_key="blade_safe_for_motion",
        attributes_keys=("blade_blockers",),
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    # Requires a real fix, a non-zero zone hash, and an in-area position type.
    MammotionBinarySensorEntityDescription(
        key="position_valid_for_motion",
        gate_key="position_valid_for_motion",
        attributes_keys=("zone_hash", "pos_type_label"),
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: MammotionConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Mammotion sensor entity."""
    mammotion_devices = entry.runtime_data.mowers

    for mower in mammotion_devices:
        async_add_entities(
            MammotionBinarySensorEntity(mower.reporting_coordinator, entity_description)
            for entity_description in BINARY_SENSORS
        )


class MammotionBinarySensorEntity(MammotionBaseEntity, BinarySensorEntity):
    """Mammotion sensor entity."""

    entity_description: MammotionBinarySensorEntityDescription

    def __init__(
        self,
        coordinator: MammotionBaseUpdateCoordinator,
        entity_description: MammotionBinarySensorEntityDescription,
    ) -> None:
        """Initialize the binary sensor entity."""
        super().__init__(coordinator, entity_description.key)
        self.entity_description = entity_description
        # Fall back to the key so a description without an explicit
        # translation_key still picks up its strings.json name. Without this,
        # entities that carry no device_class (every gate entity below) render
        # unnamed, since only `charging` gets a name from its device class.
        self._attr_translation_key = (
            entity_description.translation_key or entity_description.key
        )

    @property
    def is_on(self) -> bool | None:
        """Return true if the binary sensor is on."""
        if self.entity_description.gate_key is not None:
            return self._gate_snapshot().get(self.entity_description.gate_key)
        if self.entity_description.is_on_fn is not None:
            return self.entity_description.is_on_fn(self.coordinator.data)
        return None

    @property
    def available(self) -> bool:
        """Gate-backed entities go unavailable when the verdict is unreadable."""
        if self.entity_description.gate_key is None:
            return super().available
        return super().available and self._gate_snapshot()["available"] is True

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Expose why a gate is blocked, so the state is actionable on its own."""
        if not self.entity_description.attributes_keys:
            return None
        snapshot = self._gate_snapshot()
        return {
            key: snapshot.get(key) for key in self.entity_description.attributes_keys
        }

    def _gate_snapshot(self) -> dict[str, Any]:
        """Return the shared, briefly cached motion-gate verdict."""
        return motion_gate_snapshot(self.coordinator)
