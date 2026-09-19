"""Tests for Mammotion diagnostics."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.mammotion.diagnostics import async_get_config_entry_diagnostics


@pytest.mark.asyncio
async def test_diagnostics_are_bounded_and_private() -> None:
    """Diagnostics exclude identifiers, locations, maps, and tokens."""
    coordinator = SimpleNamespace(
        last_update_success=True,
        update_interval=None,
        data={
            "coordinates": [42.123456, -71.123456],
            "token": "private-token",
            "agora": {
                "appid": "private-app-id",
                "channel": "private-channel",
                "uid": "private-uid",
            },
            "wifi": {"ssid": "private-ssid", "password": "private-wifi-password"},
            "pairing_id": "private-pairing-id",
            "mac": "AA:BB:CC:DD:EE:FF",
        },
    )
    mower = SimpleNamespace(
        name="serial-number",
        reporting_coordinator=coordinator,
        maintenance_coordinator=coordinator,
        version_coordinator=coordinator,
        map_coordinator=coordinator,
        error_coordinator=coordinator,
    )
    entry = SimpleNamespace(
        data={
            "account_name": "private@example.test",
            "mammotion_account_id": "private-account-id",
            "ble_devices": {"private-device": "AA:BB:CC:DD:EE:FF"},
        },
        state=SimpleNamespace(value="loaded"),
        runtime_data=SimpleNamespace(mowers=[mower], RTK=[], spino=[]),
    )

    with patch(
        "custom_components.mammotion.diagnostics.async_get_integration",
        AsyncMock(return_value=SimpleNamespace(version="0.6.4-beta7")),
    ):
        result = await async_get_config_entry_diagnostics(None, entry)
    serialized = str(result)

    assert result["device_counts"]["mowers"] == 1
    assert "serial-number" not in serialized
    assert "private-token" not in serialized
    assert "coordinates" not in serialized
    for secret in (
        "private@example.test",
        "private-account-id",
        "AA:BB:CC:DD:EE:FF",
        "42.123456",
        "-71.123456",
        "private-app-id",
        "private-channel",
        "private-uid",
        "private-ssid",
        "private-wifi-password",
        "private-pairing-id",
    ):
        assert secret not in serialized


def _entry_with(mower_data: object) -> SimpleNamespace:
    """Build a config entry whose single mower's reporting data is *mower_data*."""
    coordinator = SimpleNamespace(
        last_update_success=True, update_interval=None, data=mower_data
    )
    mower = SimpleNamespace(
        name="serial-number",
        reporting_coordinator=coordinator,
        maintenance_coordinator=coordinator,
        version_coordinator=coordinator,
        map_coordinator=coordinator,
        error_coordinator=coordinator,
    )
    return SimpleNamespace(
        data={"ble_devices": {"d": "AA:BB:CC:DD:EE:FF"}},
        state=SimpleNamespace(value="loaded"),
        runtime_data=SimpleNamespace(mowers=[mower], RTK=[], spino=[]),
    )


async def _run(entry: SimpleNamespace) -> dict:
    with patch(
        "custom_components.mammotion.diagnostics.async_get_integration",
        AsyncMock(return_value=SimpleNamespace(version="0.6.4")),
    ):
        return await async_get_config_entry_diagnostics(None, entry)


@pytest.mark.asyncio
async def test_health_absent_on_older_wheel() -> None:
    """A pymammotion without device_other_info yields no health section."""
    # Data object with no device_other_info attribute at all (current wheel).
    result = await _run(_entry_with(SimpleNamespace(mower_state=object())))
    assert "health" not in result["mowers"][0]


@pytest.mark.asyncio
async def test_health_absent_when_nothing_reported() -> None:
    """device_other_info present but empty -> no health section."""
    other = SimpleNamespace(
        **dict.fromkeys(("soc_coredump", "soc_tmp", "nav", "vslam_vio"))
    )
    result = await _run(_entry_with(SimpleNamespace(device_other_info=other)))
    assert "health" not in result["mowers"][0]


@pytest.mark.asyncio
async def test_health_whitelisted_fields_surface() -> None:
    """Reported whitelisted fields appear; a genuine 0 is kept."""
    other = SimpleNamespace(
        soc_coredump=2,
        nav_coredump=1,
        soc_tmp=78,
        usb_dis_cnt=0,  # a real 0 must survive
        nav="ok",
        vslam_vio="ok",
        # fields NOT on the whitelist must never appear, even if populated:
        nrtk_url="http://private.example/rtk",
        iot_url_max="http://private.example/iot",
        systemio_boot_time="2026-08-11T20:15:58",
    )
    result = await _run(_entry_with(SimpleNamespace(device_other_info=other)))
    health = result["mowers"][0]["health"]

    assert health["soc_coredump"] == 2
    assert health["nav_coredump"] == 1
    assert health["soc_tmp"] == 78
    assert health["usb_dis_cnt"] == 0
    assert health["nav"] == "ok"

    serialized = str(result)
    assert "private.example" not in serialized
    assert "nrtk_url" not in health
    assert "iot_url_max" not in health
    assert "systemio_boot_time" not in health
