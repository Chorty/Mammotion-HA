"""Tests for conservative Mammotion capability classification."""

from types import SimpleNamespace

import pytest

from custom_components.mammotion.capabilities import capability_snapshot


@pytest.mark.parametrize(
    ("name", "family", "motion", "mower", "pool"),
    [
        ("LUBA-TEST", "luba", "yes", "yes", "no"),
        ("YUKA-TEST", "yuka", "unknown", "yes", "no"),
        ("RTK-TEST", "rtk", "no", "no", "no"),
        ("SPINO-TEST", "spino", "no", "no", "yes"),
        ("CHARGING STATION", "accessory", "no", "no", "no"),
        ("UNRECOGNIZED-TEST", "unknown", "unknown", "unknown", "unknown"),
    ],
)
def test_fixture_families_fail_closed(
    name: str,
    family: str,
    motion: str,
    mower: str,
    pool: str,
) -> None:
    """Only the live-accepted LUBA family receives manual-motion support."""
    coordinator = SimpleNamespace(
        device_name=name,
        device=SimpleNamespace(device_name=name, product_key=""),
        data=SimpleNamespace(),
        has_cloud_account=False,
    )

    result = capability_snapshot(coordinator)

    assert result["identity"]["parsed_family"] == family
    assert result["capabilities"]["manual_motion"] == motion
    assert result["capabilities"]["mower"] == mower
    assert result["capabilities"]["pool_cleaner"] == pool
