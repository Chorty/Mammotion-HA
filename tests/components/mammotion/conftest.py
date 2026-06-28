"""Fixtures for Mammotion tests."""

import pytest


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations) -> None:
    """Enable loading Mammotion from custom_components."""
