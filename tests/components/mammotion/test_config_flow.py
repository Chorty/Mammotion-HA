"""Tests for the Mammotion config flow."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant import config_entries
from homeassistant.const import CONF_PASSWORD
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mammotion.config_flow import MammotionConfigFlow
from custom_components.mammotion.const import (
    CONF_ACCOUNT_ID,
    CONF_ACCOUNTNAME,
    CONF_BLE_DEVICES,
    DOMAIN,
)


@pytest.mark.asyncio
async def test_validate_credentials_success() -> None:
    """Valid credentials return the authenticated HTTP client."""
    login_info = SimpleNamespace(
        userInformation=SimpleNamespace(userAccount="account-id")
    )
    with patch(
        "custom_components.mammotion.config_flow.MammotionHTTPCompat"
    ) as http_class:
        http_class.return_value.login = AsyncMock()
        http_class.return_value.login_info = login_info
        client, error = await MammotionConfigFlow()._async_validate_credentials(
            "user@example.com", "password"
        )

    assert client is http_class.return_value
    assert error is None


@pytest.mark.asyncio
async def test_validate_credentials_rejects_empty_values() -> None:
    """Empty cloud credentials are rejected without making a request."""
    client, error = await MammotionConfigFlow()._async_validate_credentials("", "")

    assert client is None
    assert error == "invalid_auth"


@pytest.mark.asyncio
async def test_reauth_preserves_entry_identity_and_bluetooth(hass) -> None:
    """Successful reauth updates credentials without replacing the entry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id="account-id",
        data={
            CONF_ACCOUNTNAME: "old@example.com",
            CONF_PASSWORD: "old-password",
            CONF_ACCOUNT_ID: "account-id",
            CONF_BLE_DEVICES: {"Luba-123": "AA:BB:CC:DD:EE:FF"},
        },
    )
    entry.add_to_hass(hass)
    login_info = SimpleNamespace(
        userInformation=SimpleNamespace(userAccount="account-id")
    )

    with patch(
        "custom_components.mammotion.config_flow.MammotionHTTPCompat"
    ) as http_class:
        http_class.return_value.login = AsyncMock()
        http_class.return_value.login_info = login_info
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={
                "source": config_entries.SOURCE_REAUTH,
                "entry_id": entry.entry_id,
            },
            data=entry.data,
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_ACCOUNTNAME: "new@example.com",
                CONF_PASSWORD: "new-password",
            },
        )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert entry.unique_id == "account-id"
    assert entry.data[CONF_ACCOUNTNAME] == "new@example.com"
    assert entry.data[CONF_BLE_DEVICES] == {"Luba-123": "AA:BB:CC:DD:EE:FF"}


@pytest.mark.asyncio
async def test_reauth_rejects_different_account(hass) -> None:
    """Reauth cannot silently transfer an entry to another account."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id="account-id",
        data={
            CONF_ACCOUNTNAME: "old@example.com",
            CONF_PASSWORD: "old-password",
            CONF_ACCOUNT_ID: "account-id",
        },
    )
    entry.add_to_hass(hass)
    login_info = SimpleNamespace(
        userInformation=SimpleNamespace(userAccount="different-account")
    )

    with patch(
        "custom_components.mammotion.config_flow.MammotionHTTPCompat"
    ) as http_class:
        http_class.return_value.login = AsyncMock()
        http_class.return_value.login_info = login_info
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={
                "source": config_entries.SOURCE_REAUTH,
                "entry_id": entry.entry_id,
            },
            data=entry.data,
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_ACCOUNTNAME: "other@example.com",
                CONF_PASSWORD: "password",
            },
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "wrong_account"}
    assert entry.data[CONF_ACCOUNTNAME] == "old@example.com"
