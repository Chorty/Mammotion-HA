"""Safety pins for the one-pulse item-17 fusion harness."""  # noqa: INP001

from __future__ import annotations

from scripts.night_reverse_fusion_probe import (
    ANGULAR_SPEED,
    LINEAR_SPEED,
    PULSE_DURATION_MS,
    REFRESH_INTERVAL_MS,
    _payload,
)


def test_reverse_fusion_payload_is_one_backward_only_pulse() -> None:
    """The item-17 harness cannot dispatch forward or angular motion."""
    payload = _payload(dry_run=False)

    assert payload["command"] == "send_movement"
    assert payload["linear_speed"] == LINEAR_SPEED == -400
    assert payload["angular_speed"] == ANGULAR_SPEED == 0
    assert payload["duration_ms"] == PULSE_DURATION_MS == 1300
    assert payload["motion_refresh_interval_ms"] == REFRESH_INTERVAL_MS == 200
    assert payload["confirm_blades_off"] is True
    assert payload["confirm_clear_area"] is True


def test_reverse_fusion_preview_never_confirms_motion() -> None:
    """A preview cannot satisfy either confirmation for a real pulse."""
    payload = _payload(dry_run=True)

    assert payload["dry_run"] is True
    assert payload["confirm_blades_off"] is False
    assert payload["confirm_clear_area"] is False
