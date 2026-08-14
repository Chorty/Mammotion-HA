"""Pins for the one-pulse item-16 latency harness."""  # noqa: INP001

from __future__ import annotations

from scripts.night_toward_latency_probe import (
    ANGULAR_SPEED,
    PULSE_DURATION_MS,
    REFRESH_INTERVAL_MS,
    _payload,
)


def test_latency_probe_payload_is_one_angular_only_pulse() -> None:
    """The hardware probe must have no forward or reverse command surface."""
    payload = _payload(103.1856, dry_run=False)

    assert payload["target_heading_degrees"] == 103.1856
    assert payload["angular_speed_fast"] == ANGULAR_SPEED == 500
    assert payload["angular_speed_slow"] == ANGULAR_SPEED
    assert payload["max_commands"] == 1
    assert payload["pulse_duration_ms"] == PULSE_DURATION_MS == 1500.0
    assert payload["motion_refresh_interval_ms"] == REFRESH_INTERVAL_MS == 200
    assert payload["confirm_blades_off"] is True
    assert payload["confirm_clear_area"] is True
    assert not any("linear" in key or "reverse" in key for key in payload)


def test_latency_probe_preview_never_confirms_motion() -> None:
    """A preview cannot accidentally satisfy either real-motion confirmation."""
    payload = _payload(103.1856, dry_run=True)

    assert payload["dry_run"] is True
    assert payload["confirm_blades_off"] is False
    assert payload["confirm_clear_area"] is False
