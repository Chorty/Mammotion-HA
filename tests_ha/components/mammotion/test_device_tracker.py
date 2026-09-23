"""Tests for Mammotion device-tracker presentation."""

from __future__ import annotations

import math

import pytest

from custom_components.mammotion.device_tracker import _ha_map_direction


@pytest.mark.parametrize(
    ("orientation", "expected"),
    [
        (-29.0, 29.0),
        (29.0, 331.0),
        (0.0, 0.0),
        (360.0, 0.0),
    ],
)
def test_ha_map_direction_inverts_mammotion_rotation(
    orientation: float, expected: float
) -> None:
    """HA's clockwise compass marker receives the inverse Mammotion heading."""
    assert _ha_map_direction(orientation) == expected


@pytest.mark.parametrize("orientation", [None, "unknown", math.nan, math.inf])
def test_ha_map_direction_rejects_unusable_values(orientation: object) -> None:
    """Unavailable headings must not fabricate a map-marker direction."""
    assert _ha_map_direction(orientation) is None
