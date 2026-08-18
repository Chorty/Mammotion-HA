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


def test_the_accepted_snapshot_is_the_gate5_repass_profile() -> None:
    """The snapshot records a real hardware pass, not a convenient value."""
    doc = _accepted()
    assert doc["accepted_on"] == "2026-08-12"
    assert doc["evidence"] == "docs/evidence-gate5-repass-2-20260812.json"
    assert (_REPO / doc["evidence"]).exists()
    # The ceiling Gate 5 actually ran on.
    assert doc["profile"]["max_linear_pulse_ceiling"] == 14


def test_the_current_build_is_correctly_reported_as_not_accepted() -> None:
    """beta57 raised the ceiling 14 -> 22, so it MUST report un-accepted.

    ⚠️ If this test starts failing because the diff is empty, do not delete it.
    Either a Gate 5 passed and the snapshot was legitimately regenerated -- in
    which case update this test in the same change -- or someone reverted the
    ceiling. Both are worth noticing.
    """
    diffs = compare(extract_profile(CARD.read_text()), _accepted()["profile"])
    assert diffs, "expected the shipped profile to diverge from the accepted one"
    keys = {d["key"] for d in diffs}
    assert keys == {"max_linear_pulse_ceiling"}
    row = next(d for d in diffs if d["key"] == "max_linear_pulse_ceiling")
    assert (row["accepted"], row["current"]) == (14, 22)


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
