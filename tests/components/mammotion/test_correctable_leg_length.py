"""The leg length beyond which the mid-drive controller cannot protect a landing.

Derived, not fitted. A mid-drive correction only fires once the aim error
reaches ``_MIN_CORRECTABLE_AIM_ERROR_DEGREES``; an error just under that floor
is never corrected, so it costs ``distance * sin(floor)``. Setting that equal to
``waypoint_tolerance`` gives the longest protectable leg.

These pin the arithmetic and the two hardware cases that motivated it, both from
2026-08-20: a 3.0 m sub-leg that REACHED target at 0.094 m, and its sibling that
missed by 0.2594 m when a 51.025 deg correction came due at 0.2594 m to run and
was refused ``turn_budget_infeasible``.
"""

from __future__ import annotations

import math

import pytest

from custom_components.mammotion.services import (
    _MIN_CORRECTABLE_AIM_ERROR_DEGREES,
    _POST_TURN_ALIGNMENT_TOLERANCE_DEGREES,
    _correctable_leg_length_limit_m,
)

ACCEPTED_TOLERANCE = 0.15


def test_the_floor_is_still_the_post_turn_tolerance_plus_the_deadband() -> None:
    """If this drifts, every bound below moves with it."""
    assert _MIN_CORRECTABLE_AIM_ERROR_DEGREES == pytest.approx(
        _POST_TURN_ALIGNMENT_TOLERANCE_DEGREES + 5.0
    )
    assert _MIN_CORRECTABLE_AIM_ERROR_DEGREES == pytest.approx(15.0)


def test_the_accepted_profile_can_only_protect_a_58_cm_leg() -> None:
    """0.15 m tolerance against a 15 deg floor is 0.580 m, not 0.8 and not 3.0."""
    limit = _correctable_leg_length_limit_m(waypoint_tolerance=ACCEPTED_TOLERANCE)
    assert limit == pytest.approx(0.5796, abs=1e-3)


def test_it_is_exactly_tolerance_over_sine_of_the_floor() -> None:
    """State the identity, so a future edit cannot quietly become a fit."""
    for tolerance in (0.08, 0.15, 0.30):
        assert _correctable_leg_length_limit_m(
            waypoint_tolerance=tolerance
        ) == pytest.approx(tolerance / math.sin(math.radians(15.0)))


@pytest.mark.parametrize(
    ("leg_m", "uncorrectable_miss_m"),
    [(0.80, 0.207), (3.00, 0.776), (3.85, 0.996)],
)
def test_longer_legs_permit_an_uncorrectable_miss_over_tolerance(
    leg_m: float, uncorrectable_miss_m: float
) -> None:
    """The miss a sub-floor aim error buys, at the leg lengths actually driven."""
    miss = leg_m * math.sin(math.radians(_MIN_CORRECTABLE_AIM_ERROR_DEGREES))
    assert miss == pytest.approx(uncorrectable_miss_m, abs=1e-3)
    assert miss > ACCEPTED_TOLERANCE
    assert leg_m > _correctable_leg_length_limit_m(
        waypoint_tolerance=ACCEPTED_TOLERANCE
    )


def test_the_3m_sub_leg_that_REACHED_target_was_still_over_the_bound() -> None:
    """⚠️ The bound is advisory and pessimistic -- pin that, so nobody hardens it.

    Sub-leg 1 of the 2026-08-20 chain run was 3.000002 m, over the 0.580 m
    bound, and landed 0.094 m -- comfortably inside tolerance. Exceeding the
    bound means the landing is not GUARANTEED, not that it fails.
    """
    limit = _correctable_leg_length_limit_m(waypoint_tolerance=ACCEPTED_TOLERANCE)
    assert 3.000002 > limit
    assert 0.094 < ACCEPTED_TOLERANCE


def test_a_tighter_tolerance_shortens_the_protectable_leg() -> None:
    """The two profile keys move together; neither can be read alone."""
    tight = _correctable_leg_length_limit_m(waypoint_tolerance=0.08)
    accepted = _correctable_leg_length_limit_m(waypoint_tolerance=ACCEPTED_TOLERANCE)
    assert tight < accepted


def test_a_degenerate_floor_does_not_manufacture_a_limit() -> None:
    """A zero or absurd floor must not read as 'no leg is ever safe'."""
    assert _correctable_leg_length_limit_m(
        waypoint_tolerance=0.15, min_correctable_aim_degrees=0.0
    ) == float("inf")
    assert _correctable_leg_length_limit_m(
        waypoint_tolerance=0.15, min_correctable_aim_degrees=90.0
    ) == float("inf")


def test_lowering_the_floor_to_reach_3m_would_demand_an_undeliverable_turn() -> None:
    """🚨 Shows WHY the fix is not 'lower the floor'.

    To protect a 3.0 m leg at 0.15 m the floor would have to drop to ~2.9 deg.
    The turn primitive's affine sweep bound still permits 20 deg at its 200 ms
    actuation floor, so a 2.9 deg request manufactures the error it removes.
    """
    needed = math.degrees(math.asin(ACCEPTED_TOLERANCE / 3.0))
    assert needed == pytest.approx(2.866, abs=1e-3)
    assert needed < _MIN_CORRECTABLE_AIM_ERROR_DEGREES
    assert needed < 20.0  # the sweep bound at the actuation floor
