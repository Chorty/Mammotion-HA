"""Tests for the latched cutter-RPM discriminator.

``current_cutter_rpm`` is a device register that holds its last running value
after a mow (measured 2026-07-30: bit-identical 3014 on the dock, blade off,
while the position feed jittered on a live BLE link). Discounting it is a
deliberate narrowing of a blade-safety guard, so these tests pin the exact
conditions under which that is allowed -- and, more importantly, the ones under
which it must not be.
"""

from __future__ import annotations

from typing import Any

from custom_components.mammotion.services import (
    _blade_rpm_stale_verdict,
    _runtime_blade_diagnostics,
)


def _sample(
    rpm: Any, x: float, y: float, *, state: Any = 0, mode: Any = 0
) -> dict[str, Any]:
    """Build one telemetry sample for the discriminator."""
    return {
        "blade": {
            "reported_state": state,
            "current_cutter_mode": mode,
            "current_cutter_rpm": rpm,
        },
        "position": {"x": x, "y": y},
    }


def _live_feed(rpm: Any = 3014, **kw: Any) -> list[dict[str, Any]]:
    """Three samples with a constant RPM and a demonstrably moving position."""
    return [
        _sample(rpm, 4.3504, 3.3110, **kw),
        _sample(rpm, 4.3518, 3.3137, **kw),
        _sample(rpm, 4.3521, 3.3140, **kw),
    ]


def test_latched_rpm_on_a_proven_live_feed_is_discounted() -> None:
    """The measured real-world case: constant RPM, moving position, blade off."""
    verdict = _blade_rpm_stale_verdict(_live_feed())

    assert verdict["stale_register"] is True
    assert verdict["reasons"] == []
    assert verdict["feed_proven_live"] is True


def test_a_frozen_feed_is_never_treated_as_a_latch() -> None:
    """Without liveness proof a dead feed is indistinguishable from a latch."""
    frozen = [_sample(3014, 4.3504, 3.3110) for _ in range(3)]

    verdict = _blade_rpm_stale_verdict(frozen)

    assert verdict["stale_register"] is False
    assert "feed_not_proven_live" in verdict["reasons"]


def test_a_varying_rpm_is_treated_as_a_possibly_spinning_blade() -> None:
    """A real blade reports a varying RPM, so variation must block."""
    samples = [
        _sample(3014, 4.3504, 3.3110),
        _sample(2980, 4.3518, 3.3137),
        _sample(3002, 4.3521, 3.3140),
    ]

    verdict = _blade_rpm_stale_verdict(samples)

    assert verdict["stale_register"] is False
    assert "rpm_varied_so_it_may_be_live" in verdict["reasons"]


def test_blade_state_or_mode_on_in_any_sample_blocks() -> None:
    """Either field reporting the blade on must veto, even once."""
    on_state = _blade_rpm_stale_verdict(_live_feed(state=1))
    on_mode = _blade_rpm_stale_verdict(_live_feed(mode=1))

    assert on_state["stale_register"] is False
    assert "blade_reported_on_in_a_sample" in on_state["reasons"]
    assert on_mode["stale_register"] is False
    assert "cutter_mode_on_in_a_sample" in on_mode["reasons"]


def test_too_few_samples_blocks() -> None:
    """The verdict needs the full poll count before it may discount anything."""
    verdict = _blade_rpm_stale_verdict(_live_feed()[:2])

    assert verdict["stale_register"] is False
    assert "insufficient_samples" in verdict["reasons"]


def test_sub_noise_position_jitter_does_not_prove_liveness() -> None:
    """Movement must clear the read-to-read noise floor, not just differ."""
    samples = [
        _sample(3014, 4.35040000, 3.31100000),
        _sample(3014, 4.35040001, 3.31100001),
        _sample(3014, 4.35040002, 3.31100002),
    ]

    verdict = _blade_rpm_stale_verdict(samples)

    assert verdict["stale_register"] is False
    assert "feed_not_proven_live" in verdict["reasons"]


def test_guard_stays_closed_without_the_verdict() -> None:
    """The default keeps the blade guard conservative for every other caller."""
    telemetry = _sample(3014, 4.35, 3.31)

    default = _runtime_blade_diagnostics(telemetry)

    assert default["blade_safe_for_motion"] is False
    assert default["safety_blockers"] == ["blade_rpm_nonzero"]
    assert default["blade_rpm_looks_latched"] is True
    assert default["blade_rpm_stale_register"] is False


def test_guard_opens_only_for_the_latch_signature() -> None:
    """With the verdict supplied, the latched value stops blocking."""
    telemetry = _sample(3014, 4.35, 3.31)

    discounted = _runtime_blade_diagnostics(telemetry, rpm_stale_register=True)

    assert discounted["blade_safe_for_motion"] is True
    assert discounted["safety_blockers"] == []
    assert discounted["blade_rpm_stale_register"] is True


def test_the_verdict_cannot_override_a_blade_that_reports_on() -> None:
    """A stale-register verdict must never unblock a blade reporting itself on."""
    reported_on = _runtime_blade_diagnostics(
        _sample(3014, 4.35, 3.31, state=1), rpm_stale_register=True
    )
    mode_on = _runtime_blade_diagnostics(
        _sample(3014, 4.35, 3.31, mode=1), rpm_stale_register=True
    )

    assert reported_on["blade_safe_for_motion"] is False
    assert "blade_reported_on" in reported_on["safety_blockers"]
    assert mode_on["blade_safe_for_motion"] is False
    assert "blade_cutter_mode_on" in mode_on["safety_blockers"]
    # The latch signature requires BOTH off, so neither may be discounted.
    assert reported_on["blade_rpm_stale_register"] is False
    assert mode_on["blade_rpm_stale_register"] is False
