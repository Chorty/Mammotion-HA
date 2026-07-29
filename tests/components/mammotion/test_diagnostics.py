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
