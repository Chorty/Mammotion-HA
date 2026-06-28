"""Tests for the Mammotion config flow."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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


def _authenticated_client(account_id: str) -> MagicMock:
    """Return a mocked authenticated PyMammotion client."""
    client = MagicMock()
    client.login_and_initiate_cloud = AsyncMock()
    client.stop = AsyncMock()
    client.to_cache.return_value = {}
    client.mammotion_http.login_info = SimpleNamespace(
        userInformation=SimpleNamespace(userAccount=account_id)
    )
    return client


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
    client = _authenticated_client("account-id")

    with (
        patch(
            "custom_components.mammotion.config_flow.MammotionClient",
            return_value=client,
        ),
        patch(
            "custom_components.mammotion.config_flow.async_get_integration",
            AsyncMock(return_value=SimpleNamespace(version="0.6.4-beta7")),
        ),
    ):
        flow = MammotionConfigFlow()
        flow.hass = hass
        flow.context = {"entry_id": entry.entry_id}
        result = await flow.async_step_reauth_confirm(
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
    client = _authenticated_client("different-account")

    with (
        patch(
            "custom_components.mammotion.config_flow.MammotionClient",
            return_value=client,
        ),
        patch(
            "custom_components.mammotion.config_flow.async_get_integration",
            AsyncMock(return_value=SimpleNamespace(version="0.6.4-beta7")),
        ),
    ):
        flow = MammotionConfigFlow()
        flow.hass = hass
        flow.context = {"entry_id": entry.entry_id}
        result = await flow.async_step_reauth_confirm(
            {
                CONF_ACCOUNTNAME: "other@example.com",
                CONF_PASSWORD: "password",
            },
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "wrong_account"}
    assert entry.data[CONF_ACCOUNTNAME] == "old@example.com"
