"""beta45: the motion probe can hold an ARC open at the app's cadence.

Every one of this project's `send_movement` call sites is single-axis -- linear
or angular, never both -- although `DrvMotionCtrl(set_linear_speed,
set_angular_speed)` has accepted both since the beginning. An arc is the only
route to night capability: translation keeps `toward` (course-over-ground) live,
and a live `toward` closes a heading loop with no VIO.
See `docs/night-motion-options-20260811.md`.

The probe could already send both axes, but had no refresh, so the h-watchdog
capped it at ~10 cm -- enough to prove a command actuates, not enough to
characterise an arc. These tests pin the plumbing that removes that limit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import voluptuous as vol
import yaml

from custom_components.mammotion.services import (
    RAW_PYMAMMOTION_MOTION_PROBE_SCHEMA,
)

ENTITY = "lawn_mower.test"


def _validated(**overrides: object) -> dict:
    return RAW_PYMAMMOTION_MOTION_PROBE_SCHEMA({"entity_id": ENTITY, **overrides})


def test_the_probe_defaults_to_single_shot() -> None:
    """Unchanged behaviour for every existing caller: refresh off."""
    data = _validated()
    assert data["motion_refresh_interval_ms"] == 0
    assert data["duration_ms"] == 1300


def test_an_arc_is_accepted_on_both_axes_at_the_app_cadence() -> None:
    """Linear AND angular together, held open at 200 ms."""
    data = _validated(
        linear_speed=400, angular_speed=180, motion_refresh_interval_ms=200
    )
    assert data["linear_speed"] == 400
    assert data["angular_speed"] == 180
    assert data["motion_refresh_interval_ms"] == 200


def test_the_window_is_bounded() -> None:
    """Cap the window, because nothing else limits how far this probe travels.

    The probe has no closed loop and no waypoint, so the window is capped where
    the other pulse services cap theirs.
    """
    assert _validated(duration_ms=4000)["duration_ms"] == 4000
    for bad in (49, 4001):
        with pytest.raises(vol.Invalid):
            _validated(duration_ms=bad)


def test_refresh_interval_is_bounded() -> None:
    """A cadence outside the app's range is refused rather than clamped."""
    for bad in (-1, 1001):
        with pytest.raises(vol.Invalid):
            _validated(motion_refresh_interval_ms=bad)


def test_services_yaml_exposes_both_fields() -> None:
    """A field the schema accepts but services.yaml hides is unusable from the UI."""
    root = Path(__file__).resolve().parents[3] / "custom_components" / "mammotion"
    doc = yaml.safe_load((root / "services.yaml").read_text(encoding="utf-8"))
    fields = doc["raw_pymammotion_motion_probe"]["fields"]
    assert fields["motion_refresh_interval_ms"]["default"] == 0
    assert fields["duration_ms"]["default"] == 1300


def test_the_probe_still_defaults_to_dry_run() -> None:
    """The safety default must survive the change.

    This service can now hold a real arc open for up to 4 s, so a caller that
    forgets `dry_run: false` must still move nothing.
    """
    data = _validated(linear_speed=400, angular_speed=180)
    assert data["dry_run"] is True
    assert data["confirm_blades_off"] is False
    assert data["confirm_clear_area"] is False
