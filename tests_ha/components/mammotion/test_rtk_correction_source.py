"""RTK correction-source diagnostic sensor tests."""

from pymammotion.data.model.device import MowingDevice
from pymammotion.proto import MqttRtkConnect, ReportInfoData, RptRtk, RtkUsedType

from custom_components.mammotion.sensor import SENSOR_TYPES, _rtk_correction_source


def _mower_with(info: MqttRtkConnect) -> MowingDevice:
    """Decode an RTK report through pymammotion's real update path."""
    mower = MowingDevice()
    mower.report_data.update(ReportInfoData(rtk=RptRtk(mqtt_rtk_info=info)))
    return mower


def test_never_reported_is_unknown_not_lora() -> None:
    """The dataclass default is LoRa, so absence must not read as LoRa."""
    assert _rtk_correction_source(MowingDevice()) is None


def test_bare_lora_report_is_indistinguishable_and_stays_unknown() -> None:
    """proto3 drops a zero enum, so a bare LoRa report decodes like no report."""
    mower = _mower_with(MqttRtkConnect(rtk_switch=RtkUsedType.RTK_USED_LORA))
    assert _rtk_correction_source(mower) is None


def test_lora_is_claimed_once_another_field_proves_the_message_arrived() -> None:
    """A LoRa channel or base id is evidence the report was received."""
    mower = _mower_with(
        MqttRtkConnect(rtk_switch=RtkUsedType.RTK_USED_LORA, rtk_channel=3)
    )
    assert _rtk_correction_source(mower) == "lora"


def test_non_default_sources_are_reported_directly() -> None:
    """Internet and NRTK are non-zero enums, so they cannot be a default."""
    assert (
        _rtk_correction_source(
            _mower_with(MqttRtkConnect(rtk_switch=RtkUsedType.RTK_USED_INTERNET))
        )
        == "internet"
    )
    assert (
        _rtk_correction_source(
            _mower_with(MqttRtkConnect(rtk_switch=RtkUsedType.RTK_USED_NRTK))
        )
        == "nrtk"
    )


def test_description_is_a_read_only_diagnostic_enum() -> None:
    """Every value the helper can return is a declared option."""
    description = next(d for d in SENSOR_TYPES if d.key == "rtk_correction_source")
    assert description.options == ["lora", "internet", "nrtk"]
    assert description.value_fn is _rtk_correction_source
