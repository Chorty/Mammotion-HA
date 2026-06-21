"""Tests for Mammotion diagnostics."""

from types import SimpleNamespace

import pytest

from custom_components.mammotion.diagnostics import async_get_config_entry_diagnostics


@pytest.mark.asyncio
async def test_diagnostics_are_bounded_and_private() -> None:
    """Diagnostics exclude account, network, location, map, and token data."""
    state = SimpleNamespace(
        device_firmwares=SimpleNamespace(device_version="1.2.3"),
        online=True,
        enabled=True,
        account="private-account",
        map={"coordinates": [1, 2]},
        token="private-token",
    )
    coordinator = SimpleNamespace(
        data=state,
        last_update_success=True,
        update_failures=0,
    )
    mower = SimpleNamespace(
        name="serial-number",
        device=SimpleNamespace(productModel="Luba 2", productKey="luba2"),
        api=SimpleNamespace(
            get_device_by_name=lambda name: SimpleNamespace(preference="wifi")
        ),
        reporting_coordinator=coordinator,
        maintenance_coordinator=SimpleNamespace(last_update_success=True),
        version_coordinator=SimpleNamespace(last_update_success=True),
        map_coordinator=SimpleNamespace(last_update_success=True),
    )
    entry = SimpleNamespace(domain="mammotion", runtime_data=[mower])

    result = await async_get_config_entry_diagnostics(None, entry)
    serialized = str(result)

    assert result["integration"]["device_count"] == 1
    assert "serial-number" not in serialized
    assert "private-account" not in serialized
    assert "private-token" not in serialized
    assert "coordinates" not in serialized
