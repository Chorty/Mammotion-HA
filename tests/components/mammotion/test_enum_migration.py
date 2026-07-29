"""Lowercase HA enum migration tests."""

from enum import Enum
from types import SimpleNamespace

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity

from custom_components.mammotion.select import _enum_member, _lower_options
from custom_components.mammotion.sensor import (
    _configure_enum_options,
    _normalize_enum_native_value,
    _raw_enum_attributes,
)


class ExampleEnum(Enum):
    """Representative uppercase vendor enum."""

    MODE_READY = 1


def test_select_options_are_lowercase_and_legacy_case_resolves() -> None:
    """HA sees lowercase while callers using the former case still resolve."""
    assert _lower_options(["MODE_READY"]) == ["mode_ready"]
    assert _enum_member(ExampleEnum, "mode_ready") is ExampleEnum.MODE_READY
    assert _enum_member(ExampleEnum, "MODE_READY") is ExampleEnum.MODE_READY


def test_sensor_enum_preserves_raw_protocol_label() -> None:
    """The breaking HA-state normalization does not discard vendor evidence."""
    description = SimpleNamespace(
        device_class=SensorDeviceClass.ENUM,
        options=["MODE_READY"],
    )
    entity = SensorEntity()

    _configure_enum_options(entity, description)

    assert entity.options == ["mode_ready"]
    assert _normalize_enum_native_value(description, "MODE_READY") == "mode_ready"
    assert _raw_enum_attributes(description, "MODE_READY") == {
        "raw_protocol_value": "MODE_READY"
    }
