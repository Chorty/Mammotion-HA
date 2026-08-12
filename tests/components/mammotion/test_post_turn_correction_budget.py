"""beta43: the post-turn correction gets the same command budget as any turn.

It was capped at `min(2, vio_turn_max_commands)` by an uncommented line that
predated beta40 -- and beta40 tightened this correction's tolerance from an
effective 15 deg to 10 without revisiting the cap. A tighter tolerance requires
MORE rotation, and the executor's pulse policy shortens each pulse as the error
closes, so corrections that fit in two commands at 18 deg need three at 10.

That killed the Gate 5 attempt of 2026-08-12
(`docs/evidence-gate5-repass-20260812.json`): segment 3 turned, the turn
translated it 0.176 m, the post-turn gate correctly measured a -29.647 deg
map-frame aim error, and the correction was refused `turn_budget_infeasible`.
The segment never sent a linear command.

Every number here is replayed through the shipped
`_vio_turn_budget_feasibility`, not recomputed independently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from custom_components.mammotion.services import _vio_turn_budget_feasibility

#: beta40's `_POST_TURN_ALIGNMENT_TOLERANCE_DEGREES`.
POST_TURN_TOLERANCE = 10.0
#: `vio_turn_max_commands` on the accepted profile -- and, from beta43, what the
#: post-turn correction is given.
ACCEPTED_TURN_COMMANDS = 4
#: What the cap used to resolve to.
OLD_CAP = 2
#: The aim error that failed Gate 5.
GATE5_AIM_ERROR = 29.647


def _feasible(error: float, tolerance: float, max_commands: int) -> dict:
    return _vio_turn_budget_feasibility(
        initial_error_degrees=error,
        heading_tolerance_degrees=tolerance,
        max_commands=max_commands,
        pulse_duration_ms=1500.0,
        motion_refresh_interval_ms=200,
        max_displacement_m=0.30,
        turn_degrees_per_second=37.0,
    )


def test_the_gate5_correction_was_refused_by_the_old_cap() -> None:
    """The regression, exactly as it happened. This must keep failing at 2."""
    verdict = _feasible(GATE5_AIM_ERROR, POST_TURN_TOLERANCE, OLD_CAP)
    assert verdict["feasible"] is False
    assert verdict["reason"] == "turn_budget"
    assert verdict["required_rotation_degrees"] == pytest.approx(19.647, abs=1e-3)
    assert verdict["estimated_commands_needed"] == 3
    assert verdict["max_commands"] == 2


def test_the_same_correction_is_feasible_at_the_accepted_budget() -> None:
    """beta43: with the normal turn budget it goes ahead."""
    verdict = _feasible(GATE5_AIM_ERROR, POST_TURN_TOLERANCE, ACCEPTED_TURN_COMMANDS)
    assert verdict["feasible"] is True
    assert verdict["reason"] == "within_budget"


def test_the_cap_and_not_the_tolerance_was_the_problem() -> None:
    """The same error at the OLD tolerance fits in two commands.

    This is what proves beta40's tolerance change is not at fault and should not
    be reverted: it is correct, and the budget simply was never revisited.
    """
    assert _feasible(GATE5_AIM_ERROR, 18.0, OLD_CAP)["feasible"] is True


def test_translation_is_what_bounds_the_cost_not_the_command_count() -> None:
    """Raising the budget does not raise the modelled translation.

    The estimate derives from required rotation, so the displacement cap keeps
    doing its job unchanged -- which is why the command cap was redundant.
    """
    two = _feasible(GATE5_AIM_ERROR, POST_TURN_TOLERANCE, OLD_CAP)
    four = _feasible(GATE5_AIM_ERROR, POST_TURN_TOLERANCE, ACCEPTED_TURN_COMMANDS)
    assert two["estimated_translation_m"] == four["estimated_translation_m"]
    assert four["estimated_translation_m"] <= four["max_displacement_m"]


@pytest.mark.parametrize(
    ("max_commands", "envelope"),
    [(2, 21.50), (3, 32.50), (4, 49.50)],
)
def test_the_feasible_envelope_at_the_post_turn_tolerance(
    max_commands: int, envelope: float
) -> None:
    """Pin how large a post-turn correction each budget can accept.

    The worst post-turn aim error ever recorded is ~30 deg, so 4 commands leaves
    real margin where 2 did not even cover the observed range. If a future
    change to the pulse policy or the rate constants moves these numbers, this
    is where it shows up.
    """
    assert _feasible(envelope, POST_TURN_TOLERANCE, max_commands)["feasible"] is True
    assert (
        _feasible(envelope + 1.0, POST_TURN_TOLERANCE, max_commands)["feasible"]
        is False
    )


def test_every_post_turn_error_ever_recorded_is_now_feasible() -> None:
    """The observed range, from the committed evidence files.

    ⚠️ These are the errors the gate has actually SEEN, not a bound on what it
    could see. A correction larger than 49.5 deg would still be refused -- and
    should be, since it means the turn phase left the mower badly wrong.
    """
    observed = [16.551, 8.632, 9.733, 18.139, 22.742, 29.647]
    for error in observed:
        verdict = _feasible(error, POST_TURN_TOLERANCE, ACCEPTED_TURN_COMMANDS)
        assert verdict["feasible"] is True, f"{error} deg refused"


def test_a_hopeless_correction_is_still_refused() -> None:
    """The budget preflight must not become a rubber stamp.

    An enormous post-turn error means the turn phase failed; dispatching pulses
    that provably end in `max_commands_reached` after real rotation and
    translation is what this preflight exists to prevent.
    """
    assert _feasible(120.0, POST_TURN_TOLERANCE, ACCEPTED_TURN_COMMANDS)[
        "feasible"
    ] is (False)


def test_every_accepted_profile_key_is_echoed_by_the_multi_segment_result() -> None:
    """beta44: a gate cannot prove what ran if the response drops a key.

    The 2026-08-12 Gate 5 pass had to be argued around a hole:
    `motion_refresh_interval_ms` came back null at the top level and was only
    provable from the per-segment echo plus the delivered writes, and
    `max_no_progress_pulses` was unprovable from the record at all -- it had to
    be dismissed as un-exercised instead. Proving the card sent the accepted
    profile is the entire purpose of Gate 5, so every key it sends must return.

    Pinned against the card's own frozen profile so the two cannot drift.
    """
    root = Path(__file__).resolve().parents[3] / "custom_components" / "mammotion"
    card = (root / "www" / "mammotion-custom-path-card.js").read_text(encoding="utf-8")
    frozen = card.split("const LUBA_ACCEPTANCE_PROFILE = Object.freeze({")[1].split(
        "});"
    )[0]
    keys = set(re.findall(r"^\s{2}([a-z_]+):", frozen, re.MULTILINE))
    assert "max_no_progress_pulses" in keys and "motion_refresh_interval_ms" in keys

    source = (root / "services.py").read_text(encoding="utf-8")
    # Both executors that a card Real Go can reach must echo every profile key.
    missing = sorted(k for k in keys if f'"{k}": ' not in source)
    assert not missing, f"profile keys never echoed by any result: {missing}"

    # And specifically the two that were absent from the multi-segment echo.
    for key in ("max_no_progress_pulses", "motion_refresh_interval_ms"):
        assert source.count(f'"{key}": {key},') >= 2, (
            f"{key} is not echoed by both the vector and multi-segment results"
        )
