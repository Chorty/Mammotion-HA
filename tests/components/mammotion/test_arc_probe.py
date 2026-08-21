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
    assert data["in_window_sample_interval_ms"] == 0
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


def test_in_window_sample_interval_is_opt_in_and_bounded() -> None:
    """Cache sampling is disabled by default and rejects unsafe polling values."""
    assert (
        _validated(in_window_sample_interval_ms="0")["in_window_sample_interval_ms"]
        == 0
    )
    assert (
        _validated(in_window_sample_interval_ms=100)["in_window_sample_interval_ms"]
        == 100
    )
    for bad in (-1, 1, 49, 1001):
        with pytest.raises(vol.Invalid):
            _validated(in_window_sample_interval_ms=bad)


def test_services_yaml_exposes_both_fields() -> None:
    """A field the schema accepts but services.yaml hides is unusable from the UI."""
    root = Path(__file__).resolve().parents[3] / "custom_components" / "mammotion"
    doc = yaml.safe_load((root / "services.yaml").read_text(encoding="utf-8"))
    fields = doc["raw_pymammotion_motion_probe"]["fields"]
    assert fields["motion_refresh_interval_ms"]["default"] == 0
    assert fields["in_window_sample_interval_ms"]["default"] == 0
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


def test_the_probe_forces_a_readback_after_real_motion() -> None:
    """It must not be blind to its own motion.

    On 2026-08-12 an arc moved the mower 0.5823 m and rotated its course 22.20
    deg, and the probe reported four bit-identical samples -- the new position
    only reached the coordinator cache about five minutes later. The device does
    not push position while stationary, so once a pulse ends nothing updates the
    cache until something asks. That null result was nearly written up as "arcs
    do not actuate".

    Every other motion path already called these two helpers. Asserting the
    source calls them keeps the probe from silently losing the readback again.
    """
    root = Path(__file__).resolve().parents[3] / "custom_components" / "mammotion"
    source = (root / "services.py").read_text(encoding="utf-8")
    start = source.index("async def _raw_pymammotion_motion_probe")
    # the next TOP-LEVEL def, so the nested _resend closure does not cut it short
    end = source.index("\nasync def ", start + 1)
    body = source[start:end]
    assert "_refresh_position_after_raw_motion" in body, (
        "probe does not force a report refresh after motion"
    )
    assert "_settle_linear_position_feed" in body, (
        "probe does not wait for the position feed to settle"
    )
