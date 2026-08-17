"""Pins the 2026-08-17 long-segment reach work.

Three things ship together:

1. `_MAX_SEGMENT_LENGTH_M` -- a pre-dispatch length cap the daylight/VIO path
   never had. The ~0.8 m operating rule was documentation only.
2. `_mid_drive_realign_decision` -- the re-aim trigger, changed from an angle
   test to a projected-miss test gated by the smallest correctable angle.
3. `linear_budget_insufficient_for_segment`, loop-to-tolerance only.

⚠️ **SCOPE WAS CUT ON 2026-08-17, DELIBERATELY.** An earlier version of this
branch also raised `vio_max_realignments` 3 -> 10 and added a divergence
detector to make that safe. Two rounds of review found the detector wrong twice,
for two different reasons -- first it compared before-vs-after within one
correction and so measured the correction turn's own translation
(`atan(0.10/0.75)` = 7.6 deg against a 1.0 deg margin); then it compared
successive pre-correction errors and so measured the geometric inflation of aim
error as range closes (`atan(c/d)` grows as d shrinks), which happens on a
PERFECTLY HEALTHY leg. Both would have aborted good runs.

The budget is back at the accepted 3 and the detector is gone. If a 6 m leg
exhausts 3 corrections it stops safely on `vio_realign_budget_exhausted`, which
is a measurement, and a far better basis for raising the budget than either
geometry argument was.

⚠️ NONE OF THIS HAS RUN ON HARDWARE. These tests pin intent and arithmetic, not
behaviour of the mower. The 6.10 m cap is an authorization number; the longest
segment ever executed is 4.0 m.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from custom_components.mammotion.services import (
    _BUDGET_CHECK_METRES_PER_PULSE,
    _MAX_SEGMENT_LENGTH_M,
    _MIN_CORRECTABLE_AIM_ERROR_DEGREES,
    _POST_TURN_ALIGNMENT_TOLERANCE_DEGREES,
    _mid_drive_realign_decision,
    _raw_pymammotion_execute_vector_segment,
)

from .test_map_task_visibility import _pulse_coordinator

_MIN_FLOOR = _MIN_CORRECTABLE_AIM_ERROR_DEGREES
_MAX_SEGMENT = _MAX_SEGMENT_LENGTH_M
_PER_PULSE = _BUDGET_CHECK_METRES_PER_PULSE


def _decide(distance_m: float, aim_degrees: float, **kwargs: float) -> dict:
    return _mid_drive_realign_decision(
        distance_to_target_m=distance_m,
        aim_error_degrees=aim_degrees,
        waypoint_tolerance=kwargs.get("waypoint_tolerance", 0.15),
        metres_per_pulse=kwargs.get("metres_per_pulse", 0.41),
        realign_threshold_degrees=kwargs.get("realign_threshold_degrees", 15.0),
    )


def test_the_far_field_miss_that_the_old_angle_trigger_could_not_see() -> None:
    """The regression this whole change exists for.

    17 deg of aim error with 14 m still to run is a 4.09 m miss. The old trigger
    (`aim > 15 and aim > heading_tolerance_degrees`, effectively 18 on the
    accepted profile) fired NOTHING, because 17 < 18. This is the long-leg
    blind spot in one assertion.
    """
    decision = _decide(14.0, 17.0)

    assert decision["needs_correction"] is True
    # The miss it was blind to, stated so the number cannot quietly drift.
    assert 14.0 * math.sin(math.radians(17.0)) == pytest.approx(4.09, abs=0.01)
    assert decision["projected_landing_m"] > 4.0
    # And the old condition really would have declined it: the effective
    # threshold was `heading_tolerance_degrees`, 18 on the accepted profile.
    superseded_threshold = 18.0
    assert decision["correctable_floor_degrees"] < superseded_threshold


def test_short_range_behaviour_is_essentially_unchanged() -> None:
    """The accepted short-leg regime must not move.

    Inside ~1 m the suppression guard was already the binding term and still is,
    so a correction that used to be declined is still declined -- for the same
    reason, in metres.
    """
    decision = _decide(0.5, 12.0)

    assert decision["already_lands_inside"] is True
    assert decision["needs_correction"] is False
    assert decision["projected_landing_m"] < 0.15


def test_an_aim_error_below_the_correctable_floor_never_fires() -> None:
    """A correction is an angle, and the turn cannot make an arbitrarily small one.

    At 20 m even 5 deg is a 1.7 m miss, but the shortest safe turn pulse can
    still sweep ~20 deg, so correcting would leave the mower worse aimed. The
    floor must win over the projection.
    """
    decision = _decide(20.0, _MIN_FLOOR - 0.5)

    assert decision["already_lands_inside"] is False
    assert decision["past_correctable_floor"] is False
    assert decision["needs_correction"] is False


def test_the_operator_may_raise_the_threshold_but_not_lower_the_floor() -> None:
    """`max`, not `min` -- the hardware floor is not negotiable downward."""
    lowered = _decide(14.0, 7.0, realign_threshold_degrees=1.0)
    assert lowered["correctable_floor_degrees"] == _MIN_FLOOR
    assert lowered["needs_correction"] is False

    raised = _decide(14.0, 17.0, realign_threshold_degrees=25.0)
    assert raised["correctable_floor_degrees"] == 25.0
    assert raised["needs_correction"] is False


def test_the_suppression_record_survives_the_new_trigger() -> None:
    """`needs_correction` now REQUIRES `not already_lands_inside`.

    So the executor's suppression record cannot key off
    `needs_correction and already_lands_inside` -- that is dead code, and every
    suppression would silently vanish from the run record. It keys off
    `past_correctable_floor` instead; this pins that such a case exists.
    """
    # 0.35 m out at 16 deg: past the 15 deg floor, but the projection lands
    # 0.097 m out -- inside the 0.15 tolerance, so correcting buys nothing.
    decision = _decide(0.35, 16.0)

    assert decision["past_correctable_floor"] is True
    assert decision["already_lands_inside"] is True
    assert decision["needs_correction"] is False


@pytest.mark.parametrize(
    ("length_m", "allowed"),
    [
        (0.8, True),
        (4.0, True),
        (6.096, True),  # 20 ft, the authorized cap
        (6.2, False),
        (15.24, False),  # 50 ft -- asked for, deliberately not authorized
    ],
)
def test_the_segment_length_cap_is_20_feet(length_m: float, allowed: bool) -> None:
    """20 ft is authorized; 50 ft was asked for and deliberately is not."""
    assert (length_m <= _MAX_SEGMENT) is allowed


def test_the_shipped_pulse_ceiling_can_actually_reach_the_authorized_cap() -> None:
    """The two numbers must agree or every 20 ft run refuses itself.

    `linear_budget_insufficient_for_segment` refuses when the ceiling cannot
    cover the leg at a stall-tolerant 0.30 m/pulse. The card ships 22, so 22 *
    0.30 = 6.6 m must clear the 6.10 m cap -- with margin, not exactly.
    """
    shipped_ceiling = 22
    assert shipped_ceiling * _PER_PULSE >= _MAX_SEGMENT
    # The superseded ceiling of 14 could not, which is why it moved.
    assert 14 * _PER_PULSE < _MAX_SEGMENT


def test_the_budget_check_metres_per_pulse_stays_under_the_measured_stall_rate() -> (
    None
):
    """0.30 must remain conservative against the measurements that chose it.

    Healthy pulse ~0.41 m, BLE-stalled ~0.22 m, 2 of 11 stalled on the 4 m leg.
    Even at a 50% stall rate the mean is ~0.315 m/pulse.
    """
    healthy, stalled = 0.41, 0.22
    assert (healthy + stalled) / 2 > _PER_PULSE
    assert stalled < _PER_PULSE


def test_the_correctable_floor_matches_the_post_turn_gate() -> None:
    """Both are the same physical floor derived the same way; drift is a bug."""
    assert _MIN_FLOOR == _POST_TURN_ALIGNMENT_TOLERANCE_DEGREES


def test_terminal_accuracy_is_set_by_the_pulse_not_by_the_leg() -> None:
    """Why a 6 m leg is credible at all.

    With a correction available every pulse the landing is
    ~`pulse_length * sin(residual)`, INDEPENDENT of leg length. Correcting to
    18 deg leaves only 15% of margin against an 0.15 m tolerance; correcting to
    the 10 deg floor is comfortable. That is why the mid-drive correction
    tolerance moved with the trigger.
    """
    pulse = 0.41
    assert pulse * math.sin(math.radians(18.0)) == pytest.approx(0.127, abs=0.002)
    assert pulse * math.sin(math.radians(_MIN_FLOOR)) == pytest.approx(0.071, abs=0.002)


@pytest.mark.asyncio
async def test_an_over_long_segment_is_refused_before_any_command() -> None:
    """The gate must be WIRED IN, not merely defined.

    Every arithmetic test above would still pass if the gate were built and
    never appended to `gates`. This drives the real executor with a 7.0 m leg
    and asserts nothing was sent.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 8.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="vio",
        max_linear_pulse_ceiling=22,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "segment_too_long" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_a_leg_within_the_cap_is_not_refused_by_the_length_gates() -> None:
    """The 20 ft cap must actually admit a 20 ft leg.

    A cap that refuses the length it was built to allow is worse than no cap;
    this is the paired assertion to the refusal above.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 7.09, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="vio",
        max_linear_pulse_ceiling=22,
        sample_delays=(0,),
    )

    # Not merely "these codes are absent" -- ALL gates passed, so the run was
    # admitted. A vacuous pass here (refused for some other reason) would hide a
    # cap that refuses the very length it exists to allow.
    assert result["stop_reason"] != "safety_gates_failed"
    assert result.get("blockers") in (None, [])


@pytest.mark.asyncio
async def test_a_ceiling_too_small_for_the_leg_is_refused_up_front() -> None:
    """Better a refusal than a mower stranded mid-yard on a spent budget."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 6.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="vio",
        max_linear_pulse_ceiling=5,  # 5 * 0.30 = 1.5 m, well short of 5.0 m
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "linear_budget_insufficient_for_segment" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_the_fixed_budget_path_keeps_its_accepted_behaviour() -> None:
    """The budget gate is loop-to-tolerance only, and that is load-bearing.

    Fixed-budget pulses travel ~1.06 m (measured 1.0785 / 1.0449), not the
    conservative 0.30 the loop figure assumes. Applying the loop number here
    refused runs that Gate 4 and Gate 5 both passed.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 3.0, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="vio",
        max_linear_commands=1,
        sample_delays=(0,),
    )

    assert "linear_budget_insufficient_for_segment" not in (
        result.get("blockers") or []
    )
