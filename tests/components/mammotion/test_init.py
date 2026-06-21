"""Tests for Mammotion config-entry lifecycle helpers."""

from types import SimpleNamespace

import pytest

from custom_components.mammotion import async_remove_config_entry_device


@pytest.mark.asyncio
async def test_active_mower_device_cannot_be_removed() -> None:
    """An active mower remains attached to its config entry."""
    entry = SimpleNamespace(
        runtime_data=SimpleNamespace(mowers=[SimpleNamespace(unique_name="Luba-123")])
    )
    device = SimpleNamespace(identifiers={("mammotion", "Luba-123")})

    assert not await async_remove_config_entry_device(None, entry, device)


@pytest.mark.asyncio
async def test_stale_mower_device_can_be_removed() -> None:
    """A device absent from runtime data may be removed."""
    entry = SimpleNamespace(runtime_data=SimpleNamespace(mowers=[]))
    device = SimpleNamespace(identifiers={("mammotion", "Luba-123")})

    assert await async_remove_config_entry_device(None, entry, device)
