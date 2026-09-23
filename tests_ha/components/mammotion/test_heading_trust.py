"""The three heading failures of 2026-09-04, each pinned so it cannot return.

Full record: `docs/findings-clicktopath-reliability-4m-20260904.md`. In one
night nobody -- not the integration, not the orchestrating session, not the
operator through Home Assistant -- could reliably answer "which way is this
mower pointing?", and hardware moved in an unintended direction as a result.

Three distinct defects wore that one symptom:

1. **A model-form error.** Targets were placed along
   `toward + calibrated_forward_heading_offset_degrees`. The map frame is a math
   angle and `toward` is a compass bearing, so the relation is a REFLECTION;
   no additive constant can emulate one. Measured on 43 real pulses.
2. **A circular check.** "Aligned start confirmed" compared the echoed
   `target_reported_heading_degrees` against `toward` -- but the target had been
   placed along `toward`, so they agreed by construction and the check measured
   nothing.
3. **Corroboration read as freshness.** After a manual reposition both device
   heading sources still described the pre-reposition facing and agreed with
   each other to 0.079 deg, and `current_orientation` published
   `trustworthy: true` at the moment it was most wrong.

The first is replayed against the banked 43-pulse dataset, which is a real
regression test that needs no mower.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.mammotion.services import (
    _CONTINUOUS_MIRROR_SUM_DEGREES,
    _FACING_MOTION_CONFIRMED_TTL_SECONDS,
    _TOWARD_MIRROR_DEGREES,
    _current_orientation,
    _map_facing_report,
    _map_heading_to_toward_degrees,
    _start_alignment_evidence,
    _toward_to_map_heading_degrees,
    _unmeasured_start_geometry,
)

from .conftest import _pulse_coordinator

EVIDENCE = (
    Path(__file__).parents[3]
    / "docs"
    / "evidence-clicktopath-reliability-4m-20260904.json"
)

#: The accepted profile's additive offset, i.e. the model that was wrong.
ADDITIVE_OFFSET = 102.4


def _rows() -> list[dict[str, Any]]:
    with EVIDENCE.open(encoding="utf-8") as handle:
        return json.load(handle)["heading_model_check"]["per_pulse_rows"]


def _abs_error(predicted: float, measured: float) -> float:
    return abs(((predicted - measured + 180.0) % 360.0) - 180.0)


def _errors(rows: list[dict[str, Any]], model) -> list[float]:
    return [
        _abs_error(
            model(row["toward_at_pulse_start_degrees"]),
            row["measured_movement_heading_degrees"],
        )
        for row in rows
    ]


def _fresh_rows() -> list[dict[str, Any]]:
    """Pulses whose `toward` did not move during the pulse.

    Every excluded pulse is the first of a run or the one straight after a
    realignment turn: `toward` lags a rotation by a full ~5 s pulse cycle, which
    is the documented "only valid when `toward` is fresh" caveat, quantified.
    """
    return [row for row in _rows() if not row["toward_changed_during_pulse"]]


# ---------------------------------------------------------------------------
# 1. The model form, replayed against 43 real pulses
# ---------------------------------------------------------------------------


def test_mirror_predicts_the_driven_direction_on_banked_hardware_pulses() -> None:
    """The shipped reflection reproduces the 1.000 deg mean measured that night.

    Predictions come from `_toward_to_map_heading_degrees`, the function the
    integration actually ships, not from a formula rewritten in the test.
    """
    errors = _errors(_fresh_rows(), _toward_to_map_heading_degrees)

    assert len(errors) == 30
    assert statistics.mean(errors) == pytest.approx(1.000, abs=0.001)
    assert statistics.median(errors) == pytest.approx(0.944, abs=0.001)
    assert max(errors) == pytest.approx(3.003, abs=0.001)


def test_additive_offset_is_wrong_by_87_degrees_on_the_same_pulses() -> None:
    """The offset that placed every target is off by a mean 87.478 deg."""
    errors = _errors(_fresh_rows(), lambda toward: (toward + ADDITIVE_OFFSET) % 360.0)

    assert len(errors) == 30
    assert statistics.mean(errors) == pytest.approx(87.478, abs=0.001)
    assert max(errors) == pytest.approx(166.825, abs=0.001)


def test_no_additive_constant_can_replace_the_reflection() -> None:
    """🚨 This is a model-form bug, not a mistuned constant.

    Sweeping every offset at 0.5 deg resolution, the BEST achievable additive
    model is still an order of magnitude worse than the reflection. Retuning
    `calibrated_forward_heading_offset_degrees` therefore cannot fix this, which
    is the whole reason the remedy is not a new number.
    """
    rows = _fresh_rows()
    best = min(
        statistics.mean(_errors(rows, lambda t, o=offset: (t + o) % 360.0))
        for offset in [index * 0.5 for index in range(720)]
    )
    mirror = statistics.mean(_errors(rows, _toward_to_map_heading_degrees))

    assert mirror < 1.5
    assert best > 15 * mirror


def test_the_reflection_is_an_involution() -> None:
    """Map -> toward -> map must be the identity, or the two names lie."""
    for heading in (0.0, 12.5, 90.13, 180.0, 271.9, 359.9):
        assert _map_heading_to_toward_degrees(
            _toward_to_map_heading_degrees(heading)
        ) == pytest.approx(heading % 360.0, abs=1e-9)


def test_one_mirror_constant_serves_every_caller() -> None:
    """Three copies of 90.13 existed; a per-mower constant must not fork."""
    assert _CONTINUOUS_MIRROR_SUM_DEGREES is _TOWARD_MIRROR_DEGREES


# ---------------------------------------------------------------------------
# 2. The circular alignment check
# ---------------------------------------------------------------------------


def test_alignment_is_unknown_without_an_independent_facing() -> None:
    """No independent measurement must produce None, never True."""
    evidence = _start_alignment_evidence(
        measured_map_facing_degrees=None,
        target_map_heading_degrees=287.243,
    )

    assert evidence["aligned_start_confirmed"] is None
    assert evidence["reason"] == "independent_facing_unavailable"
    assert evidence["basis"] is None


def test_an_unmeasured_start_never_claims_alignment() -> None:
    """Every not-measured path returns the same honest shape."""
    evidence = _unmeasured_start_geometry("dry_run")

    assert evidence["aligned_start_confirmed"] is None
    assert evidence["initial_heading_error_degrees"] is None
    assert "by construction" in evidence["circularity_warning"]


@pytest.mark.parametrize(
    ("run", "measured_facing", "target_map_heading", "expected_error"),
    [
        # The four runs of the 4.0 m series, from findings section 4. The
        # calibration drive's measured facing is the independent source; the
        # target map heading is where the additive offset actually pointed.
        (1, 260.863, 287.243, 26.380),
        (2, 289.839, 262.953, 26.886),
        (3, 157.220, 35.399, 121.821),
        (4, 31.419, 165.975, 134.556),
    ],
)
def test_all_four_banked_runs_are_post_turn_legs_not_aligned_starts(
    run: int,
    measured_facing: float,
    target_map_heading: float,
    expected_error: float,
) -> None:
    """🚨 Every run the session recorded as "aligned" was nothing of the kind.

    Runs 3 and 4 opened with real ~120-135 deg turns, a property their own
    predeclaration put out of scope. This check, which the executor now runs
    itself, would have said so before the second dispatch.
    """
    evidence = _start_alignment_evidence(
        measured_map_facing_degrees=measured_facing,
        target_map_heading_degrees=target_map_heading,
    )

    assert evidence["aligned_start_confirmed"] is False, run
    assert evidence["reason"] == "post_turn_leg"
    assert evidence["initial_heading_error_degrees"] == pytest.approx(
        expected_error, abs=0.01
    )
    assert evidence["basis"] == "vio_calibration_drive.map_motion_heading_degrees"


def test_a_genuinely_aligned_start_is_confirmed() -> None:
    """The check must still be able to say yes, or it is just a refusal."""
    evidence = _start_alignment_evidence(
        measured_map_facing_degrees=260.863,
        target_map_heading_degrees=263.100,
    )

    assert evidence["aligned_start_confirmed"] is True
    assert evidence["reason"] == "aligned"


def test_target_reported_heading_matching_toward_is_never_the_basis() -> None:
    """A target placed along `toward` echoes `toward`; that is not evidence.

    Reconstructing what the session did -- place the target along
    `toward + 102.4`, then observe that the echoed reported heading equals
    `toward` -- and feeding the SAME number in as if it were a measurement must
    not be reachable: `_start_alignment_evidence` names its basis, and the only
    basis it will ever name is the calibration drive.
    """
    toward = -175.1567
    target_map_heading = (toward + ADDITIVE_OFFSET) % 360.0
    # The circular "check": the reported heading derived straight back out.
    echoed_reported_heading = (target_map_heading - ADDITIVE_OFFSET) % 360.0
    assert echoed_reported_heading == pytest.approx(toward % 360.0, abs=1e-6)

    evidence = _start_alignment_evidence(
        measured_map_facing_degrees=None,
        target_map_heading_degrees=target_map_heading,
    )

    assert evidence["aligned_start_confirmed"] is None
    assert evidence["basis"] is None


# ---------------------------------------------------------------------------
# 3. Corroboration is not freshness
# ---------------------------------------------------------------------------


def _facing_coordinator(
    *,
    vio_heading: float,
    toward: float,
    features: int = 80,
    motion: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build a coordinator with independently set heading and motion evidence."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=vio_heading, vio_state=2, track_feature_num=features, brightness=200
    )
    if motion is not None:
        coordinator.facing_motion_evidence = lambda: motion
    return coordinator, {"position": {"toward": toward}}


#: The exact pre-move raw values from findings section 6.3. The mower had just
#: been turned by hand; both sources still described the OLD facing.
REPOSITION_TOWARD = 91.8054
REPOSITION_VIO = -1.754


def test_two_stale_sources_agreeing_do_not_authorise_a_dispatch() -> None:
    """🚨 The 2026-09-04 incident state, replayed.

    Mirror `90.13 - 91.8054` = 358.325 and VIO `-1.754 % 360` = 358.246 agree to
    0.079 deg, so `current_orientation` publishes `trustworthy: true` -- and it
    is right to, because two independent estimates really do agree. What it
    cannot see is that both describe where the mower pointed BEFORE the operator
    picked it up. With no driven leg on record, aiming must be refused.
    """
    coordinator, telemetry = _facing_coordinator(
        vio_heading=REPOSITION_VIO,
        toward=REPOSITION_TOWARD,
        motion={
            "last_travel_bearing_degrees": None,
            "last_travel_distance_m": None,
            "last_travel_age_seconds": None,
            "seconds_since_position_change": 900.0,
            "min_travel_distance_m": 0.10,
        },
    )

    orientation = _current_orientation(coordinator, telemetry)
    report = _map_facing_report(coordinator, telemetry)

    assert orientation["trustworthy"] is True
    assert orientation["disagreement_degrees"] == pytest.approx(0.079, abs=0.005)
    assert report["confidence"] == "corroborated_not_motion_confirmed"
    assert report["safe_to_aim_dispatch"] is False
    assert report["operator_confirmation_required"] is True
    assert report["reason"] == "no_driven_leg_on_record"


def test_trustworthy_says_in_the_payload_what_it_does_not_mean() -> None:
    """The flag was read as freshness. It has to say that it is not."""
    coordinator, telemetry = _facing_coordinator(
        vio_heading=REPOSITION_VIO, toward=REPOSITION_TOWARD
    )

    orientation = _current_orientation(coordinator, telemetry)

    assert "NOT that either is fresh" in orientation["trustworthy_means"]


def test_a_driven_leg_that_agrees_confirms_the_facing() -> None:
    """After the mower drives, its estimate has been checked against the ground.

    Post-incident raw values: `toward` -101.8713 gives a mirror of 192.001 and
    VIO `-168.570 % 360` = 191.430. The mower had just driven that way.
    """
    coordinator, telemetry = _facing_coordinator(
        vio_heading=-168.570,
        toward=-101.8713,
        motion={
            "last_travel_bearing_degrees": 191.6,
            "last_travel_distance_m": 0.5,
            "last_travel_age_seconds": 8.0,
            "seconds_since_position_change": 8.0,
            "min_travel_distance_m": 0.10,
        },
    )

    report = _map_facing_report(coordinator, telemetry)

    assert report["confidence"] == "motion_confirmed"
    assert report["safe_to_aim_dispatch"] is True
    assert report["operator_confirmation_required"] is False
    assert report["map_facing_degrees"] == pytest.approx(191.430, abs=0.01)


def test_a_166_degree_jump_between_estimate_and_driven_leg_refuses() -> None:
    """The incident's signature: the mower went ~166 deg from where it "pointed".

    Both estimates still read the pre-reposition facing while the machine drove
    the other way. Whatever the explanation, the honest output is "ask", and it
    must not be `safe_to_aim_dispatch`.
    """
    coordinator, telemetry = _facing_coordinator(
        vio_heading=REPOSITION_VIO,
        toward=REPOSITION_TOWARD,
        motion={
            "last_travel_bearing_degrees": (358.325 - 166.3) % 360.0,
            "last_travel_distance_m": 0.5,
            "last_travel_age_seconds": 5.0,
            "seconds_since_position_change": 5.0,
            "min_travel_distance_m": 0.10,
        },
    )

    report = _map_facing_report(coordinator, telemetry)

    assert report["confidence"] == "corroborated_not_motion_confirmed"
    assert report["safe_to_aim_dispatch"] is False
    assert report["reason"] == "facing_disagrees_with_last_driven_leg"
    assert report["motion_agreement_degrees"] == pytest.approx(166.3, abs=0.5)


def test_a_stale_driven_leg_stops_confirming() -> None:
    """Sitting still is the window in which a mower gets picked up and turned."""
    coordinator, telemetry = _facing_coordinator(
        vio_heading=-168.570,
        toward=-101.8713,
        motion={
            "last_travel_bearing_degrees": 191.6,
            "last_travel_distance_m": 0.5,
            "last_travel_age_seconds": _FACING_MOTION_CONFIRMED_TTL_SECONDS + 1.0,
            "seconds_since_position_change": _FACING_MOTION_CONFIRMED_TTL_SECONDS + 1.0,
            "min_travel_distance_m": 0.10,
        },
    )

    report = _map_facing_report(coordinator, telemetry)

    assert report["safe_to_aim_dispatch"] is False
    assert report["reason"] == "last_driven_leg_too_old"


def test_facing_is_unknown_when_the_sources_disagree() -> None:
    """A disagreement returns no number at all, whatever the motion evidence."""
    coordinator, telemetry = _facing_coordinator(
        vio_heading=91.81,
        toward=90.13,
        motion={
            "last_travel_bearing_degrees": 91.8,
            "last_travel_distance_m": 4.0,
            "last_travel_age_seconds": 2.0,
            "seconds_since_position_change": 2.0,
            "min_travel_distance_m": 0.10,
        },
    )

    report = _map_facing_report(coordinator, telemetry)

    assert report["confidence"] == "unknown"
    assert report["map_facing_degrees"] is None
    assert report["safe_to_aim_dispatch"] is False


def test_facing_is_reported_in_compass_terms_for_the_operator() -> None:
    """Stating the destination in compass terms needs the number to exist.

    A map bearing is a math angle; the operator's yard is not. The compass
    bearing is the mirror applied the other way.
    """
    coordinator, telemetry = _facing_coordinator(
        vio_heading=-168.570,
        toward=-101.8713,
        motion={
            "last_travel_bearing_degrees": 191.6,
            "last_travel_distance_m": 0.5,
            "last_travel_age_seconds": 8.0,
            "seconds_since_position_change": 8.0,
            "min_travel_distance_m": 0.10,
        },
    )

    report = _map_facing_report(coordinator, telemetry)

    assert report["map_facing_compass_degrees"] == pytest.approx(258.7, abs=0.2)
    assert report["map_facing_compass_point"] == "WSW"


def test_facing_report_never_uses_the_additive_offset() -> None:
    """The model string is load-bearing: it is what a future caller copies."""
    coordinator, telemetry = _facing_coordinator(
        vio_heading=REPOSITION_VIO, toward=REPOSITION_TOWARD
    )

    report = _map_facing_report(coordinator, telemetry)

    assert "REFLECTION" in report["model"]
    assert report["toward_mirror_degrees"] == _TOWARD_MIRROR_DEGREES
