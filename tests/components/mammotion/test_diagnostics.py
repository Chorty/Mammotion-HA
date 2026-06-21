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
        data={"coordinates": [1, 2], "token": "private-token"},
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
        data={},
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
