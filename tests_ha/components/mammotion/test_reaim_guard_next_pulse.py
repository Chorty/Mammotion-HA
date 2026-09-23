"""beta42: the re-aim guard projects to the end of the next pulse.

The guard used to answer `distance * sin(aim)` -- the miss at the point of
CLOSEST APPROACH. The mower does not stop there; it drives a whole pulse and
sails past it. On 2026-08-11 that cost a segment: a suppression with 3.1 mm of
margin landed 0.1797 m out and ended on `target_requires_reverse_recovery`.

Every case below is replayed from a committed evidence file, so this pins
behaviour against hardware rather than against the new arithmetic.
"""

from __future__ import annotations

import math

import pytest

from custom_components.mammotion.services import (
    _effective_metres_per_pulse,
    _projected_landing_after_next_pulse,
    _realign_cannot_improve_the_landing,
)

TOLERANCE = 0.15
#: `final_approach_metres_per_pulse` on the accepted profile. Every decision on
#: record happens with under half a metre remaining, so the next pulse is aimed
#: at the remaining distance and this figure never binds -- but it is the honest
#: input, and it is what the executor's own planner uses.
METRES_PER_PULSE = 1.06


def _suppresses(distance: float, aim: float) -> bool:
    return _realign_cannot_improve_the_landing(
        distance_to_target_m=distance,
        aim_error_degrees=aim,
        waypoint_tolerance=TOLERANCE,
        metres_per_pulse=METRES_PER_PULSE,
    )


# ------------------------------------------------------------------ the model


def test_the_projection_is_the_chord_when_one_pulse_covers_the_distance() -> None:
    """Driving the right distance in the wrong direction lands on a chord.

    With travel == remaining distance, `hypot(d sin a, d - d cos a)` reduces
    exactly to `2 d sin(a/2)`. Asserting the closed form keeps the derivation
    honest: if someone later "simplifies" the quadrature into something else,
    this fails.
    """
    for distance in (0.10, 0.2605, 0.3246, 0.75):
        for aim in (5.0, 18.719, 26.914, 54.246, 89.0):
            projected = _projected_landing_after_next_pulse(
                distance_to_target_m=distance,
                aim_error_degrees=aim,
                metres_per_pulse=METRES_PER_PULSE,
            )
            chord = 2 * distance * math.sin(math.radians(aim) / 2)
            assert projected == pytest.approx(chord, abs=1e-9)


def test_a_full_pulse_caps_the_overshoot_term() -> None:
    """Past one pulse of travel the mower cannot overshoot by more than a pulse."""
    projected = _projected_landing_after_next_pulse(
        distance_to_target_m=5.0, aim_error_degrees=10.0, metres_per_pulse=0.4
    )
    perpendicular = 5.0 * math.sin(math.radians(10.0))
    # 5 m out at 10 deg: the closest approach is 4.92 m ahead and one 0.4 m
    # pulse cannot reach it, so there is no overshoot term at all.
    assert projected == pytest.approx(perpendicular, abs=1e-9)


def test_zero_aim_projects_zero_miss() -> None:
    """Pointed straight at the target, one pulse of the remaining distance arrives."""
    assert _projected_landing_after_next_pulse(
        distance_to_target_m=0.4, aim_error_degrees=0.0, metres_per_pulse=1.06
    ) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------- the decision it changes


def test_the_2026_08_11_suppression_is_now_corrected() -> None:
    """THE regression. `docs/evidence-beta32-4segment-20260811T235133Z.json`.

    Segment 2, after linear pulse 4: 0.3246 m out at -26.914 deg. The old guard
    projected 0.1469 m and suppressed by 3.1 mm. The next pulse ran 0.3771 m and
    the segment landed 0.1797 m out, stopping on reverse-recovery.

    ⚠️ The new projection clears tolerance by 1.1 mm. This test pins the SIDE of
    the boundary, not a comfortable margin -- there isn't one.
    """
    projected = _projected_landing_after_next_pulse(
        distance_to_target_m=0.3246,
        aim_error_degrees=-26.914,
        metres_per_pulse=METRES_PER_PULSE,
    )
    assert projected == pytest.approx(0.1511, abs=5e-4)
    assert projected > TOLERANCE
    assert _suppresses(0.3246, -26.914) is False

    # And the old rule really did suppress it -- otherwise this fixes nothing.
    old_rule = 0.3246 * math.sin(math.radians(26.914))
    assert old_rule == pytest.approx(0.1469, abs=5e-4)
    assert old_rule <= TOLERANCE


@pytest.mark.parametrize(
    ("distance", "aim", "landed", "source"),
    [
        (0.2017, -34.733, 0.1424, "20260810T185433Z seg2"),
        (0.3035, 18.152, 0.1431, "20260810T193833Z seg2"),
        (0.2390, -34.655, 0.1447, "20260810T193833Z seg3"),
        (0.2606, 25.980, 0.1229, "20260810T193833Z seg4"),
        (0.1573, 34.692, 0.0867, "20260810T205937Z seg2"),
        (0.2624, -27.427, 0.1393, "20260810T205937Z seg3"),
        (0.2551, 24.496, 0.0960, "20260810T232848Z seg4"),
        (0.4063, 20.073, 0.1467, "20260811T001250Z seg4"),
        (0.2605, -18.719, 0.1023, "20260812T002804Z seg1"),
    ],
)
def test_suppressions_that_were_right_stay_suppressed(
    distance: float, aim: float, landed: float, source: str
) -> None:
    """Every beta38-era suppression whose segment then landed inside the disc.

    The failure mode of a tighter guard is spending turn commands and turn
    translation on corrections that buy nothing, so these matter as much as the
    one it fixes.
    """
    assert landed <= TOLERANCE, f"{source}: test data says it missed"
    assert _suppresses(distance, aim) is True, source


@pytest.mark.parametrize(
    ("distance", "aim", "landed", "source"),
    [
        (0.1764, -52.014, 0.1393, "20260810T205937Z seg3"),
        (0.1754, 54.246, 0.1467, "20260811T001250Z seg4"),
    ],
)
def test_two_large_aim_suppressions_now_correct_and_that_is_accepted(
    distance: float, aim: float, landed: float, source: str
) -> None:
    """The known, deliberate cost of beta42 -- recorded so it is not a surprise.

    Both scraped inside the disc (0.1393 and 0.1467 against 0.150) while pointed
    52.0 and 54.2 deg away from the target. Declining to re-aim at that angle
    because the arithmetic says you would just clip the edge is not a trade worth
    defending, and the measured price of a correction is ~0.97 deg of induced
    bearing error to buy ~10 deg. If a future run shows these corrections making
    landings WORSE, this test is the place that says the decision was knowing.
    """
    assert landed <= TOLERANCE, f"{source}: was inside the disc"
    assert _suppresses(distance, aim) is False, source


# -------------------------------------------------------- the shared helper


def test_the_guard_and_the_planner_share_one_pulse_estimate() -> None:
    """They must not keep separate models of the next step.

    The beta32 turn preflight had exactly this bug in the rotation axis: two
    models of the same pulse policy that disagreed over 100-117 deg.
    """
    assert _effective_metres_per_pulse([], 1.06) == 1.06
    # A slow observation never shrinks the estimate.
    assert _effective_metres_per_pulse([0.4, 0.5], 1.06) == 1.06
    # A fast one raises it.
    assert _effective_metres_per_pulse([1.2, 1.4], 1.06) == pytest.approx(1.3)
