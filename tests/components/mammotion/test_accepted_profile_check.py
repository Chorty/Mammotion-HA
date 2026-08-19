"""Pins `scripts/check_accepted_profile.py`.

This check is the only thing that states, on the release page itself, whether a
build ships the hardware-accepted execution profile. Before it existed the claim
lived only in prose, and the workflow's `confirmed_luba_acceptance` boolean
could not supply it: that input gates the release job with an `if:`, so it is
true by construction whenever the job runs.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts"))

from check_accepted_profile import (  # noqa: E402
    ACCEPTED,
    CARD,
    compare,
    extract_profile,
)


def _accepted() -> dict:
    return json.loads(ACCEPTED.read_text())


def test_the_card_profile_parses() -> None:
    """The parser must cope with the real card, comments and all."""
    profile = extract_profile(CARD.read_text())
    assert profile["turn_mode"] == "vio"
    assert profile["waypoint_tolerance"] == 0.15
    assert profile["sample_delays"] == [0, 3]
    # Comment text must never leak into a value.
    assert all("//" not in str(v) for v in profile.values())


def test_the_accepted_snapshot_is_a_real_gate5_pass() -> None:
    """The snapshot records a real hardware pass, not a convenient value.

    Re-snapshotted 2026-08-18 after Gate 5 passed on the reach profile: four
    card-driven segments, 4/4 target_reached at 0.1038 / 0.0863 / 0.1261 /
    0.1129 m against a 0.15 m tolerance, and the dispatched payload carried
    every profile key byte-identically. The previous snapshot was the
    2026-08-12 re-pass at ceiling 14.
    """
    doc = _accepted()
    assert doc["accepted_on"] == "2026-08-18"
    assert doc["evidence"] == "docs/evidence-gate5-beta57-20260818.json"
    assert (_REPO / doc["evidence"]).exists()
    # The ceiling this Gate 5 actually ran on.
    assert doc["profile"]["max_linear_pulse_ceiling"] == 22


def test_the_shipped_profile_is_the_accepted_one() -> None:
    """The shipped card must match the snapshot exactly.

    ⚠️ INVERTED 2026-08-18, and the inversion is the point. This asserted the
    build was NOT accepted while beta57's ceiling change (14 -> 22) was
    outstanding; Gate 5 then passed on that profile and the snapshot was
    regenerated, so the same guard now asserts the opposite.

    If this starts failing, do NOT re-snapshot to silence it. A non-empty diff
    means the shipped profile has moved away from the last hardware-accepted
    one, which owes the section 4 re-pinning in docs/gate4-repass-20260805.md
    and another Gate 5. Regenerate only after that gate actually passes, with
    --write-accepted and the evidence file that proves it.
    """
    diffs = compare(extract_profile(CARD.read_text()), _accepted()["profile"])
    assert diffs == [], f"shipped profile has drifted from the accepted one: {diffs}"


def test_compare_detects_every_kind_of_drift() -> None:
    """Changed, added and removed keys must all surface."""
    accepted = {"a": 1, "b": 2, "gone": 3}
    current = {"a": 1, "b": 99, "added": 4}
    rows = {d["key"]: (d["accepted"], d["current"]) for d in compare(current, accepted)}
    assert rows == {
        "b": (2, 99),
        "gone": (3, "<absent>"),
        "added": ("<absent>", 4),
    }
    assert "a" not in rows


def test_an_identical_profile_reports_no_drift() -> None:
    """A profile compared to itself must be clean -- guards a false-alarm check."""
    profile = extract_profile(CARD.read_text())
    assert compare(profile, profile) == []


@pytest.mark.parametrize(
    "snippet",
    [
        "const LUBA_ACCEPTANCE_PROFILE = Object.freeze({\n  a: 1, // trailing\n});",
        "const LUBA_ACCEPTANCE_PROFILE = Object.freeze({\n  // leading\n  a: 1,\n});",
        'const LUBA_ACCEPTANCE_PROFILE = Object.freeze({\n  a: "http://x", \n});',
    ],
)
def test_comment_stripping_does_not_eat_strings(snippet: str) -> None:
    """`//` inside a string value must survive; a real comment must not."""
    out = extract_profile(snippet)
    assert out["a"] in (1, "http://x")
