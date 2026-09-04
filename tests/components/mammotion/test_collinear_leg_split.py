"""Pins Route B (2026-08-19): a distant click auto-splits into collinear sub-legs.

The ask from 2026-08-17 was to move the mower 50 ft (15.24 m) from one click.
Route A -- the 6.10 m per-segment cap and the projected-miss re-aim trigger --
shipped, passed a Gate 5, and is measured **inert**: replayed across 32 decision
points on three hardware runs, the old and new triggers made identical decisions
every time. The ask was still unmet.

Route B reaches the distance using only pieces that have been measured. A leg
longer than the card's 3.85 m target becomes `ceil(d / target)` equal sub-legs by
linear interpolation between the operator's two clicks, so every inserted point
sits exactly on the original line. A collinear junction is a 0 degree turn, and a
0 degree turn costs **zero turn commands and zero translation**:
`_vio_turn_to_heading` returns `target_heading_reached` before dispatching
anything when `|error| <= heading_tolerance_degrees`. Each sub-leg then gets its
own fresh pulse budget, which is what makes 15 m reachable inside a control law
validated at ~4 m.

WHAT THESE TESTS DO NOT PROVE
-----------------------------
* **That 15.40 m is drivable.** It has never been driven. The longest straight
  leg ever executed is 4.0 m (landing 0.1023 m against 0.15 m, **n = 1**), and
  3.81 m is 95% of that single datapoint. 3.85 is not proven better than 4.0,
  only shorter.
* **That splitting improves accuracy.** It does not. Cross-track error has
  UNITY GAIN across a collinear junction, not contraction: each sub-leg re-aims
  from the mower's live position to the next point on the original line, so
  `miss_{k+1} ~= miss_k` plus noise. See
  `test_an_intermediate_miss_is_below_every_correction_threshold` -- a 0.10 m
  junction miss opens the next 3.81 m leg at 1.50 degrees, below every
  correction threshold in the control law, so **nothing corrects it**. The fresh
  budget prevents ceiling exhaustion; it does not reduce lateral error.
* **Anything about the profile.** `split_leg_target_length_m` is deliberately
  NOT a `LUBA_ACCEPTANCE_PROFILE` key. Adding it would un-accept the
  hardware-accepted profile and owe another Gate 5 -- the exact cost Route B
  exists to avoid.

The failure mode this names: an intermediate miss near the 0.15 m tolerance puts
the next sub-leg on the tolerance edge and may end on
`target_requires_reverse_recovery`. That is bounded and self-announcing, and it
is the thing to watch per sub-leg index on the first hardware run.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import voluptuous as vol

from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.manual_motion import REAL_CLICK_TO_GO_SEGMENT_LIMIT
from custom_components.mammotion.services import (
    _MAX_SEGMENT_LENGTH_M,
    _SPLIT_LEG_TARGET_LENGTH_M,
    RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA,
    RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA,
    _path_distance,
    _path_heading_degrees,
    _raw_pymammotion_execute_multi_segment,
    _split_long_legs,
    _vio_turn_to_heading,
)

from .conftest import _pulse_coordinator

# A true 50 ft click. Placed inside the shared fixture's 20 x 20 m area
# (-10..10 on both axes) so containment is not what these tests are measuring.
_FIFTY_FEET_M = 15.24
_FIFTY_FEET_PATH = [{"x": -8.0, "y": 0.0}, {"x": -8.0 + _FIFTY_FEET_M, "y": 0.0}]
# The longest straight leg ever executed: 2026-08-11, landing 0.1023 m against a
# 0.15 m tolerance, stopping on tolerance rather than on the ceiling. n = 1.
_LONGEST_LEG_EVER_EXECUTED_M = 4.0

# Sentinel for "this schema has no such key at all".
_ABSENT = object()


def _headings(points: list[dict[str, float]]) -> list[float]:
    return [
        _path_heading_degrees(start, end)
        for start, end in zip(points, points[1:], strict=False)
    ]


# --------------------------------------------------------------------------
# The geometry the whole route rests on
# --------------------------------------------------------------------------


def test_fifty_feet_splits_into_four_sub_legs_of_3_81_m() -> None:
    """The ask, in one assertion: 15.24 m becomes 4 legs of 3.81 m."""
    split = _split_long_legs(
        _FIFTY_FEET_PATH, target_length_m=_SPLIT_LEG_TARGET_LENGTH_M
    )

    assert split["applied"] is True
    assert split["requested_leg_count"] == 1
    assert split["sub_leg_count"] == 4
    assert len(split["points"]) == 5
    for start, end in zip(split["points"], split["points"][1:], strict=False):
        assert _path_distance([start, end]) == pytest.approx(3.81, abs=1e-9)
    # Every sub-leg is inside the longest straight leg ever executed.
    assert (
        max(
            _path_distance([start, end])
            for start, end in zip(split["points"], split["points"][1:], strict=False)
        )
        < _LONGEST_LEG_EVER_EXECUTED_M
    )


def test_inserted_points_are_collinear_to_within_a_microdegree() -> None:
    """The load-bearing property. A non-collinear junction is not a free turn.

    Checked on an oblique bearing, where floating-point interpolation has the
    most room to drift -- an axis-aligned line would pass by construction.
    """
    bearing = math.radians(37.0)
    path = [
        {"x": 1.0, "y": 2.0},
        {
            "x": 1.0 + _FIFTY_FEET_M * math.cos(bearing),
            "y": 2.0 + _FIFTY_FEET_M * math.sin(bearing),
        },
    ]

    split = _split_long_legs(path, target_length_m=_SPLIT_LEG_TARGET_LENGTH_M)
    headings = _headings(split["points"])

    assert len(headings) == 4
    for heading in headings:
        assert abs(heading - headings[0]) < 1e-6


def test_interpolated_coordinates_are_not_rounded() -> None:
    """Rounding to 3 dp would inject ~1.4 mm of non-collinearity.

    1.4 mm sounds negligible and is not: it is the difference between a junction
    that dispatches nothing and one that spends a turn command and its
    translation. The guard is that the split emits full-precision floats.
    """
    bearing = math.radians(37.0)
    path = [
        {"x": 0.0, "y": 0.0},
        {
            "x": _FIFTY_FEET_M * math.cos(bearing),
            "y": _FIFTY_FEET_M * math.sin(bearing),
        },
    ]

    split = _split_long_legs(path, target_length_m=_SPLIT_LEG_TARGET_LENGTH_M)
    interior = split["points"][1:-1]

    assert interior
    assert any(point["x"] != round(point["x"], 3) for point in interior)

    rounded = [
        {"x": round(point["x"], 3), "y": round(point["y"], 3)}
        for point in split["points"]
    ]
    rounded_headings = _headings(rounded)
    # What rounding would have cost, stated so the choice cannot be undone by
    # accident: a junction error orders of magnitude worse than the exact one.
    assert max(abs(h - rounded_headings[0]) for h in rounded_headings) > 1e-4


@pytest.mark.asyncio
async def test_a_collinear_junction_costs_zero_turn_commands_and_zero_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route B's entire premise, exercised through the turn primitive itself.

    Not asserted around the handler -- the handler is called, and the mower's
    command channel is asserted untouched.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=90.0, vio_state=2
    )

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=90.0,
        heading_tolerance_degrees=18.0,
        max_commands=4,
        angular_speed=500,
        pulse_duration_ms=1500.0,
        max_displacement_m=0.25,
    )

    assert result["stop_reason"] == "target_heading_reached"
    assert result["commands_sent"] == 0
    assert result["final_displacement_m"] == 0.0
    coordinator.manager.send_command_with_args.assert_not_called()


# --------------------------------------------------------------------------
# Off by default -- the constraint the accepted profile depends on
# --------------------------------------------------------------------------


def test_off_by_default_returns_the_points_unchanged() -> None:
    """The most important test in the file.

    A caller that omits the parameter must dispatch byte-identically to before
    the splitter existed, so nothing about the hardware-accepted profile moves.
    """
    for target in (None, 0.0):
        split = _split_long_legs(_FIFTY_FEET_PATH, target_length_m=target)

        assert split["applied"] is False
        assert split["points"] == _FIFTY_FEET_PATH
        assert split["sub_leg_count"] == 1


@pytest.mark.asyncio
async def test_the_accepted_gate5_geometry_dispatches_identically_with_the_splitter_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0.8 m legs are far under the target, so the splitter is a no-op on them.

    This is the Gate 5 configuration -- four 0.8 m legs -- run with the splitter
    ON and OFF, comparing the points each segment was actually handed.
    """
    gate5_path = [{"x": 1.0 + 0.8 * index, "y": 1.0} for index in range(5)]

    async def run(target: float | None) -> list[list[dict[str, float]]]:
        coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
        coordinator.data.report_data.vision_info = SimpleNamespace(
            heading=10.0, vio_state=2
        )
        dispatched: list[list[dict[str, float]]] = []

        async def record(
            _coordinator: object,
            points: list[dict[str, float]],
            **_kwargs: object,
        ) -> dict[str, object]:
            dispatched.append([dict(point) for point in points])
            return {"stop_reason": "target_reached", "valid": True, "blockers": []}

        monkeypatch.setattr(
            mammotion_services,
            "_raw_pymammotion_execute_vector_segment",
            record,
        )
        await _raw_pymammotion_execute_multi_segment(
            coordinator,
            gate5_path,
            dry_run=False,
            confirm_blades_off=True,
            confirm_clear_area=True,
            max_real_segments=REAL_CLICK_TO_GO_SEGMENT_LIMIT,
            split_leg_target_length_m=target,
            sample_delays=(0,),
        )
        return dispatched

    without_splitter = await run(None)
    with_splitter = await run(_SPLIT_LEG_TARGET_LENGTH_M)

    assert len(without_splitter) == 4
    assert with_splitter == without_splitter


# --------------------------------------------------------------------------
# The handler: split before the preview, refuse by name, echo the provenance
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_fifty_foot_click_reaches_the_executor_as_four_sub_legs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through the handler: two clicks in, four driven legs out."""
    coordinator = _pulse_coordinator(position=(-8.0, 0.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)
    dispatched: list[list[dict[str, float]]] = []

    async def record(
        _coordinator: object,
        points: list[dict[str, float]],
        **_kwargs: object,
    ) -> dict[str, object]:
        dispatched.append([dict(point) for point in points])
        return {"stop_reason": "target_reached", "valid": True, "blockers": []}

    monkeypatch.setattr(
        mammotion_services, "_raw_pymammotion_execute_vector_segment", record
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        _FIFTY_FEET_PATH,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_real_segments=REAL_CLICK_TO_GO_SEGMENT_LIMIT,
        split_leg_target_length_m=_SPLIT_LEG_TARGET_LENGTH_M,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "target_reached"
    assert result["real_segments_executed"] == 4
    assert len(dispatched) == 4
    for points in dispatched:
        assert _path_distance(points) == pytest.approx(3.81, abs=1e-6)
    # Provenance: the response records what was CLICKED as well as what ran.
    assert result["split"]["applied"] is True
    assert result["split"]["sub_leg_count"] == 4
    assert len(result["requested_points"]) == 2
    assert len(result["points"]) == 5
    assert result["split_leg_target_length_m"] == _SPLIT_LEG_TARGET_LENGTH_M
    # Every inserted junction is free.
    for junction in result["junction_turn_feasibility"]:
        assert junction["turn_degrees"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_three_five_metre_clicks_are_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 x 5 m splits into 6 sub-legs against a budget of 4."""
    coordinator = _pulse_coordinator(position=(-8.0, 0.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    async def fail_if_called(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("no segment may execute on an over-budget split")

    monkeypatch.setattr(
        mammotion_services,
        "_raw_pymammotion_execute_vector_segment",
        fail_if_called,
    )

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [
            {"x": -8.0, "y": 0.0},
            {"x": -3.0, "y": 0.0},
            {"x": 2.0, "y": 0.0},
            {"x": 7.0, "y": 0.0},
        ],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_real_segments=REAL_CLICK_TO_GO_SEGMENT_LIMIT,
        split_leg_target_length_m=_SPLIT_LEG_TARGET_LENGTH_M,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "split_exceeds_real_segment_budget"
    assert "split_exceeds_real_segment_budget" in result["blockers"]
    gate = next(
        item
        for item in result["safety_gates"]
        if item["name"] == "split_exceeds_real_segment_budget"
    )
    # The detail names the real reason, not a bare point count.
    assert "3 destination(s) split into 6 sub-legs" in gate["detail"]
    assert gate["diagnostics"]["sub_leg_count"] == 6
    assert gate["diagnostics"]["max_segments"] == REAL_CLICK_TO_GO_SEGMENT_LIMIT
    assert result["segments_executed"] == 0


@pytest.mark.asyncio
async def test_the_identical_refusal_arrives_on_a_dry_run() -> None:
    """A dry run that passes while Real Go refuses is the trap this closes.

    The operator plans against the preview; if the preview is more permissive
    than the run, the refusal is discovered with the mower on the lawn.
    """
    coordinator = _pulse_coordinator(position=(-8.0, 0.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [
            {"x": -8.0, "y": 0.0},
            {"x": -3.0, "y": 0.0},
            {"x": 2.0, "y": 0.0},
            {"x": 7.0, "y": 0.0},
        ],
        dry_run=True,
        split_leg_target_length_m=_SPLIT_LEG_TARGET_LENGTH_M,
        sample_delays=(0,),
    )

    assert result["stop_reason"] == "split_exceeds_real_segment_budget"
    assert "split_exceeds_real_segment_budget" in result["blockers"]
    assert result["would_send"] is False
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_an_inserted_point_outside_the_area_fails_containment() -> None:
    """Proves the split runs BEFORE the preview.

    A concave area can contain both clicks while the straight line between them
    leaves it. `_validate_custom_path` checks containment per point, so it can
    only catch that if the inserted points exist by the time it runs. If the
    split moved after the preview this test would pass validation and drive the
    mower out of the area.
    """
    coordinator = _pulse_coordinator(position=(-8.0, 0.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)
    # A U-shaped area open at the bottom: both clicks sit in the arms at y = 0,
    # and the straight line between them crosses the gap.
    coordinator.data.map.area = {
        123: SimpleNamespace(
            data=[
                SimpleNamespace(
                    current_frame=0,
                    data_couple=[
                        SimpleNamespace(x=-9.0, y=-1.0),
                        SimpleNamespace(x=-3.0, y=-1.0),
                        SimpleNamespace(x=-3.0, y=4.0),
                        SimpleNamespace(x=3.0, y=4.0),
                        SimpleNamespace(x=3.0, y=-1.0),
                        SimpleNamespace(x=9.0, y=-1.0),
                        SimpleNamespace(x=9.0, y=5.0),
                        SimpleNamespace(x=-9.0, y=5.0),
                    ],
                )
            ]
        )
    }

    unsplit = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        _FIFTY_FEET_PATH,
        dry_run=True,
        area_hash=123,
        sample_delays=(0,),
    )
    # Both clicks are inside, so without the split there is nothing to catch.
    assert unsplit["valid"] is True

    split = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        _FIFTY_FEET_PATH,
        dry_run=True,
        area_hash=123,
        split_leg_target_length_m=_SPLIT_LEG_TARGET_LENGTH_M,
        sample_delays=(0,),
    )

    assert split["valid"] is False
    assert "path_points_outside_known_area_geometry" in split["errors"]
    assert split["stop_reason"] == "path_validation_failed"


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_the_schema_literal_still_mirrors_the_segment_length_cap() -> None:
    """The 6.10 bound in the schema is a LITERAL, and must not drift.

    `_MAX_SEGMENT_LENGTH_M` is defined ~10,000 lines BELOW the schema, so
    referencing it there is a `NameError` at import. This is the check that
    stands in for the reference.
    """
    assert _MAX_SEGMENT_LENGTH_M == 6.10

    base = {
        "entity_id": "lawn_mower.test",
        "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
    }

    at_cap = RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA(
        {**base, "split_leg_target_length_m": _MAX_SEGMENT_LENGTH_M}
    )
    assert at_cap["split_leg_target_length_m"] == _MAX_SEGMENT_LENGTH_M

    with pytest.raises(vol.Invalid):
        RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA(
            {**base, "split_leg_target_length_m": _MAX_SEGMENT_LENGTH_M + 0.01}
        )


def test_the_schema_omits_the_key_entirely_when_not_sent() -> None:
    """No default. An omitted parameter must not become an implicit `None` key.

    `gate4-repass` section 4 forbids changing schema DEFAULTS; this parameter
    has none, so a caller that omits it is unchanged by its existence.
    """
    validated = RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA(
        {
            "entity_id": "lawn_mower.test",
            "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
        }
    )

    assert "split_leg_target_length_m" not in validated


def test_the_card_target_is_reachable_within_the_segment_budget() -> None:
    """3.85 m x 4 segments must cover a 50 ft click with headroom.

    The headroom is the point: `n = ceil(d / target)` is a step function and
    15.24 / 3.81 = 4.000 exactly, so at 3.81 a centimetre of drift between the
    card's snapshot and the backend's flips the count to 5 and refuses the run.
    """
    assert _SPLIT_LEG_TARGET_LENGTH_M == 3.85
    assert math.ceil(_FIFTY_FEET_M / _SPLIT_LEG_TARGET_LENGTH_M) == 4
    assert _SPLIT_LEG_TARGET_LENGTH_M * REAL_CLICK_TO_GO_SEGMENT_LIMIT > _FIFTY_FEET_M
    # And the drift headroom itself, so shortening the target cannot silently
    # remove it.
    assert _FIFTY_FEET_M / 4 < _SPLIT_LEG_TARGET_LENGTH_M - 0.03


# --------------------------------------------------------------------------
# What splitting does NOT buy
# --------------------------------------------------------------------------


def test_an_intermediate_miss_is_below_every_correction_threshold() -> None:
    """Cross-track error has UNITY GAIN across a collinear junction.

    Landing 0.10 m off at an intermediate point opens the next 3.81 m sub-leg
    at atan(0.10 / 3.81) = 1.50 degrees. That is below the smallest correctable
    aim error (15 degrees) AND below the turn tolerance (18), so nothing
    corrects it -- the mower simply drives the next leg from where it is. The
    fresh budget prevents ceiling exhaustion; it does not reduce lateral error.

    Documenting the arithmetic, not claiming convergence.
    """
    miss_m = 0.10
    sub_leg_m = _FIFTY_FEET_M / 4
    opening_aim_degrees = math.degrees(math.atan2(miss_m, sub_leg_m))

    assert opening_aim_degrees == pytest.approx(1.50, abs=0.01)
    assert opening_aim_degrees < 15.0
    assert opening_aim_degrees < 18.0
    # And the honest consequence: a miss near the 0.15 m tolerance leaves the
    # next sub-leg opening on the tolerance edge, which is the thing to watch
    # per sub-leg index on the first hardware run.
    assert math.degrees(math.atan2(0.15, sub_leg_m)) < 15.0


def test_night_is_untouched_by_the_splitter() -> None:
    """Night caps at one 1.0 m segment and never sends the parameter."""
    split = _split_long_legs(
        [{"x": 0.0, "y": 0.0}, {"x": 0.9, "y": 0.0}],
        target_length_m=_SPLIT_LEG_TARGET_LENGTH_M,
    )

    assert split["applied"] is False
    assert split["points"] == [{"x": 0.0, "y": 0.0}, {"x": 0.9, "y": 0.0}]


def test_routing_a_long_two_point_leg_to_multi_segment_changes_no_dispatch() -> None:
    """The card now sends a 2-point long click to the MULTI-segment service.

    Before Route B a two-point path always went to the vector-segment service.
    The two schemas' defaults are not identical, so this is only safe because
    the card explicitly sends every key that differs. Verified, not assumed --
    if a future key diverges without the card sending it, a long click would
    silently dispatch different motion than a short one.
    """
    base = {
        "entity_id": "lawn_mower.test",
        "points": [{"x": 0.0, "y": 0.0}, {"x": 5.0, "y": 0.0}],
    }
    multi = RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA(dict(base))
    vector = RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA(dict(base))

    diverging = {
        key
        for key in set(multi) | set(vector)
        if multi.get(key, _ABSENT) != vector.get(key, _ABSENT)
    }

    # Every diverging key must be one the card sends explicitly on both branches
    # (`max_real_segments` exists only on the multi-segment service and is only
    # read behind `if not dry_run`).
    assert diverging == {
        "max_linear_commands",
        "max_turn_commands",
        "max_real_segments",
    }
