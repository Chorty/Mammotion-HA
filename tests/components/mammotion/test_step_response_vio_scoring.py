"""Offline tests for the E-VIO scoring of step-response criteria 2a/2b.

Adopted 2026-09-01 per `docs/findings-rtk-vio-course-rate-scoring-20260831.md`
(predeclared in `docs/predeclared-rtk-vio-course-rate-scoring-20260831.md`):
2a is scored from VIO heading via half-phase mean-rate agreement, 2b keeps its
last-two-rates semantics on the VIO channel, omega/tau come from the same
channel and tau exists only when 2a passes, and dark VIO refuses to score
rather than falling back to the noise-bound RTK chord rule.

The banked runs in `docs/raw-samples/` are the regression fixtures — 775 real
samples across six supervised runs, five of them scored here, including the two
verdicts that FLIP versus the published RTK scoring AND the 2026-09-01 repeat
that refutes one of those flips. Pinning them is the point: nobody may quietly
restore the old instrument without these failing.

⚠️ The roster is `BANKED_RUNS`, and
`test_every_banked_run_is_pinned_or_explicitly_excused` enforces that every file
on disk is either in it or excused with a reason — because the 2026-09-01 repeat
sat unpinned for two days while this suite stayed green.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import voluptuous as vol

from custom_components.mammotion import services
from custom_components.mammotion.services import (
    _STEP_RESPONSE_VIO_MAX_PLAUSIBLE_RATE_DEG_PER_S,
    STEP_RESPONSE_PROBE_SCHEMA,
    _step_response_half_phase_agreement,
    _step_response_vio_analysis,
    _step_response_vio_intervals,
)

RAW_SAMPLES = Path(__file__).resolve().parents[3] / "docs" / "raw-samples"


def _load_samples(name: str) -> list[dict[str, Any]]:
    return json.loads((RAW_SAMPLES / name).read_text())["samples"]


def _synthetic_samples(headings: list[float | None], *, state: int = 2) -> list[dict]:
    """One sample per second, VIO latching handled by the extractor."""
    return [
        {"elapsed_ms": 1000.0 * i, "vio": {"heading": h, "state": state}}
        for i, h in enumerate(headings)
    ]


# ---------------------------------------------------------------------------
# The banked runs — the two flips, and the repeat that refutes one of them
# ---------------------------------------------------------------------------


BANKED_RUNS = [
    # Both 5000 ms runs were still accelerating when the step ended — the
    # predeclared ground-truth anchors. Any rule passing them is broken.
    ("raw-route1-run1-plus120-step5000-20260830.json", 5000, False, 2.156, True),
    (
        "raw-route1-run1repeat-plus120-step5000-20260830.json",
        5000,
        False,
        3.664,
        True,
    ),
    # 🚨 THE FLIP: the published RTK 2a PASS (0.11 deg/s, an 0.04-sigma
    # margin on a noise-bound statistic) does not survive — VIO's second
    # half is materially faster, so tau=2.038s stays demoted.
    ("raw-route1-stepext-plus120-step7000-20260830.json", 7000, False, 2.319, True),
    # 🚨 THE OTHER FLIP: the published RTK 2a FAIL (5.64 deg/s of chord
    # noise on a steady -11.3 deg/s plateau) becomes a clean PASS.
    ("raw-route1-run2-plus180-step7000-20260830.json", 7000, True, 0.130, True),
    # 🚨 AND IT DOES NOT REPRODUCE. Same configuration as the row above, run
    # 2026-09-01: 2a FAILS at 3.4049 -- 26x the banked run's margin. The two
    # runs' PLANT agrees to 0.195 deg/s; the entire verdict split comes from
    # the single interval straddling the command onset. See
    # docs/findings-plus180-split-is-onset-sampling-phase-20260901.md.
    # This run went unpinned for two days after the fact; that is why the
    # roster below is now derived from the directory rather than typed.
    (
        "raw-route1-run2repeat-plus180-step7000-20260901.json",
        7000,
        False,
        3.4049,
        True,
    ),
    # The 2026-09-03 (linear 300) pair. Both were banked unpinned and caught by
    # test_every_banked_run_is_pinned_or_explicitly_excused; see
    # docs/predeclared-vio-heading-continuity-guard-20260903.md §8. Neither
    # scores a 2a PASS -- a 5000 ms step is still ramping at both angular
    # commands, exactly as the 400-series 5000 ms runs were.
    (
        "raw-linear300-angular120-step5000-20260903.json",
        5000,
        False,
        2.8364,
        True,
    ),
    (
        "raw-linear300-angular180-step5000-20260903.json",
        5000,
        False,
        4.3966,
        True,
    ),
]


@pytest.mark.parametrize(
    ("filename", "step_ms", "pass_2a", "half_diff", "pass_2b"), BANKED_RUNS
)
def test_banked_runs_score_as_the_findings_document_published(
    filename: str, step_ms: int, pass_2a: bool, half_diff: float, pass_2b: bool
) -> None:
    """Every banked run scores exactly as the findings document published."""
    analysis = _step_response_vio_analysis(
        _load_samples(filename), baseline_ms=3000, step_ms=step_ms
    )
    assert analysis["scoreable"] is True
    verdict_2a = analysis["step_steady_rotation_2a"]
    assert verdict_2a["passed"] is pass_2a
    assert verdict_2a["half_diff_deg_per_s"] == pytest.approx(half_diff, abs=0.01)
    assert analysis["settle_flat_2b"]["passed"] is pass_2b


def test_tau_exists_only_when_2a_passes_and_omega_is_the_steady_half() -> None:
    """A ramp-sampled omega is the failure 2a exists to prevent."""
    failing = _step_response_vio_analysis(
        _load_samples("raw-route1-stepext-plus120-step7000-20260830.json"),
        baseline_ms=3000,
        step_ms=7000,
    )
    assert failing["omega_step_deg_per_s"] is None
    assert failing["tau_actuator_s"] is None
    # rotation_after_zero is a pure endpoint difference and stays reportable.
    assert failing["rotation_after_zero_deg"] is not None

    passing = _step_response_vio_analysis(
        _load_samples("raw-route1-run2-plus180-step7000-20260830.json"),
        baseline_ms=3000,
        step_ms=7000,
    )
    half_rates = passing["step_steady_rotation_2a"]["half_rates_deg_per_s"]
    assert passing["omega_step_deg_per_s"] == half_rates[1]
    assert passing["omega_step_deg_per_s"] == pytest.approx(-11.359, abs=0.01)
    assert passing["tau_actuator_s"] == pytest.approx(0.7995, abs=0.01)


def test_dark_vio_refuses_to_score_with_a_named_reason() -> None:
    """Never fall back silently to the noise-bound RTK chord rule."""
    samples = _load_samples("raw-route1-run2-plus180-step7000-20260830.json")
    samples[70]["vio"]["state"] = 1
    analysis = _step_response_vio_analysis(samples, baseline_ms=3000, step_ms=7000)
    assert analysis["scoreable"] is False
    assert analysis["unscoreable_reason"] == "vio_not_live_throughout"
    assert "step_steady_rotation_2a" not in analysis


def test_a_latched_vio_track_is_insufficient_not_a_pass() -> None:
    """A VIO heading that never changes cannot certify anything."""
    analysis = _step_response_vio_analysis(
        _synthetic_samples([10.0] * 20), baseline_ms=3000, step_ms=7000
    )
    assert analysis["scoreable"] is False
    assert analysis["unscoreable_reason"] == "vio_track_insufficient"


def test_half_phase_agreement_fails_a_smooth_ramp_that_last_two_would_pass() -> None:
    """The structural reason E-VIO won.

    Adjacent rates on a smooth ramp look equal long before the ramp converges
    (how VIO last-two wrongly passed R1).
    """
    # Rates 1, 2, 3, 4, 5, 6 deg/s — adjacent rates 1 deg/s apart (inside the
    # bound), halves 2 vs 5 deg/s apart (outside it).
    headings = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0]
    samples = _synthetic_samples(headings)
    intervals = _step_response_vio_intervals(samples, baseline_ms=0, step_ms=7000)
    rates = [iv["rate_deg_per_s"] for iv in intervals if iv["phase"] == "step"]
    assert abs(rates[-1] - rates[-2]) <= 1.5  # last-two would pass
    verdict = _step_response_half_phase_agreement(intervals, "step")
    assert verdict["passed"] is False  # the ramp is caught


def test_half_phase_agreement_passes_a_genuinely_steady_rotation() -> None:
    """A flat rate profile clears the bound comfortably."""
    headings = [0.0, -11.0, -22.5, -33.5, -45.0, -56.5, -67.5]
    intervals = _step_response_vio_intervals(
        _synthetic_samples(headings), baseline_ms=0, step_ms=7000
    )
    verdict = _step_response_half_phase_agreement(intervals, "step")
    assert verdict["passed"] is True
    assert verdict["half_diff_deg_per_s"] <= 0.5


def test_vio_intervals_skip_latched_samples_and_normalize_the_wrap() -> None:
    """Latched samples yield no interval; the +/-180 wrap goes the short way."""
    samples = [
        {"elapsed_ms": 0.0, "vio": {"heading": 175.0, "state": 2}},
        {"elapsed_ms": 100.0, "vio": {"heading": 175.0, "state": 2}},  # latched
        {"elapsed_ms": 1000.0, "vio": {"heading": -175.0, "state": 2}},  # wraps
    ]
    intervals = _step_response_vio_intervals(samples, baseline_ms=5000, step_ms=5000)
    assert len(intervals) == 1
    # +10 deg the short way round over 1 s, not -350.
    assert intervals[0]["rate_deg_per_s"] == pytest.approx(10.0, abs=0.01)


def test_the_probe_attaches_vio_analysis_to_its_result() -> None:
    """Source-level pin: the impl must emit vio_analysis alongside analysis."""
    source = inspect.getsource(
        services._step_response_probe_impl  # noqa: SLF001
    )
    assert 'result["vio_analysis"] = _step_response_vio_analysis(' in source
    # The RTK diagnostic stays emitted — cross-checkability is the point.
    assert 'result["course_series"] = _step_response_course_series(' in source


def test_every_banked_run_is_pinned_or_explicitly_excused() -> None:
    """No banked run may sit unscored just because nobody typed it into the table.

    🚨 Regression, 2026-09-03. `raw-route1-run2repeat-plus180-step7000-20260901.json`
    -- the run whose 2a FAIL refuted the +180 flip -- was absent from the
    parametrized roster for two days while the whole suite stayed green, because
    the roster was a hand-typed literal. A file appearing in docs/raw-samples/
    must now either be pinned above or be named here with a reason.

    Precedent for globbing the directory rather than listing it:
    tests/components/mammotion/test_step_response_probe.py.
    """
    pinned = {row[0] for row in BANKED_RUNS}
    # Excused, with reasons. Not "forgotten".
    excused = {
        # Phase A: a 1000 ms step yields ~1 informative step interval against the
        # rule's >=3, so 2a is unscoreable BY DESIGN. It measured speed, not 2a.
        "raw-phaseA-linear300-speed-20260903.json",
        # 🚨 The heading-discontinuity run. It has no 2a/2b verdict to pin
        # BECAUSE the continuity guard refuses it, which is the whole point --
        # it is pinned instead by
        # test_a_heading_frame_discontinuity_refuses_to_score_with_a_named_reason.
        "raw-linear300-angular180-20260903.json",
    }
    on_disk = {p.name for p in RAW_SAMPLES.glob("*.json")}
    unaccounted = on_disk - pinned - excused
    assert not unaccounted, (
        f"banked runs neither pinned nor excused: {sorted(unaccounted)} -- add "
        "them to the parametrized roster, or excuse them here with a reason"
    )


# ---------------------------------------------------------------------------
# The heading-continuity guard — predeclared in
# docs/predeclared-vio-heading-continuity-guard-20260903.md
# ---------------------------------------------------------------------------


def test_a_heading_frame_discontinuity_refuses_to_score_with_a_named_reason() -> None:
    """🚨 The 2026-09-03 defect, pinned as the fixture that must stay refused.

    VIO heading jumped -166.47 deg in one report while the mower drove STRAIGHT
    (operator-observed; RTK course moved ~6 deg across the whole window), and
    `vio_state` stayed 2 for all 79 samples -- so the liveness guard never
    fired and the run returned `scoreable: true` with a 2b PASS computed over a
    discontinuous track.
    """
    analysis = _step_response_vio_analysis(
        _load_samples("raw-linear300-angular180-20260903.json"),
        baseline_ms=3000,
        step_ms=1000,
    )
    assert analysis["scoreable"] is False
    assert analysis["unscoreable_reason"] == "vio_heading_discontinuity"
    # No verdict may survive alongside the refusal.
    assert "step_steady_rotation_2a" not in analysis
    assert "settle_flat_2b" not in analysis
    # The refusal must NAME what tripped it -- never silent (criterion 3).
    (offender,) = analysis["discontinuities"]
    assert offender["rate_deg_per_s"] == pytest.approx(-149.7945, abs=0.01)
    assert offender["from_heading_degrees"] == pytest.approx(89.769, abs=0.01)
    assert offender["to_heading_degrees"] == pytest.approx(-76.705, abs=0.01)
    assert offender["phase"] == "step"


def test_the_guard_is_state_independent_because_vio_state_never_sees_it() -> None:
    """The whole point: `vio_state` checks LIVENESS, not CONTINUITY.

    Every sample in the defective run carried the live state, so a guard keyed
    on state could never have caught this.
    """
    samples = _load_samples("raw-linear300-angular180-20260903.json")
    assert {(s["vio"] or {}).get("state") for s in samples} == {2}


def test_a_clean_fast_rotation_at_the_measured_envelope_still_scores() -> None:
    """Criterion 4 — the guard must not refuse the plant working normally.

    13.4 deg/s is the fastest steady rotation ever measured under an admissible
    command (linear 300, angular 180). A guard that refuses it is useless.
    """
    headings = [round(-13.4 * i, 4) for i in range(8)]
    analysis = _step_response_vio_analysis(
        _synthetic_samples(headings), baseline_ms=0, step_ms=7000
    )
    assert analysis["scoreable"] is True
    assert analysis["step_steady_rotation_2a"]["passed"] is True


def test_the_guard_fires_in_any_phase_not_only_the_step() -> None:
    """A frame jump anywhere shifts the origin for everything after it."""
    for jump_at in (1, 5):
        headings = [0.0, -11.0, -22.0, -33.0, -44.0, -55.0, -66.0]
        headings[jump_at:] = [h - 120.0 for h in headings[jump_at:]]
        analysis = _step_response_vio_analysis(
            _synthetic_samples(headings), baseline_ms=2000, step_ms=2000
        )
        assert analysis["scoreable"] is False, jump_at
        assert analysis["unscoreable_reason"] == "vio_heading_discontinuity"


def test_the_guard_refuses_the_run_rather_than_dropping_the_bad_interval() -> None:
    """Re-scoring the survivors is the move rejected on 2026-09-01.

    Choosing which samples to discard after seeing which verdicts it flips is
    the failure the 2026-08-23 mirror-criterion review registered against. A
    track with 6 clean intervals around one jump is still unscoreable.
    """
    headings = [0.0, -11.0, -22.0, 140.0, 129.0, 118.0, 107.0, 96.0]
    analysis = _step_response_vio_analysis(
        _synthetic_samples(headings), baseline_ms=0, step_ms=8000
    )
    assert analysis["scoreable"] is False
    assert analysis["unscoreable_reason"] == "vio_heading_discontinuity"


def test_the_continuity_bound_is_tied_to_the_admissible_commands() -> None:
    """🚨 Criterion 6 — widening the schema must force a re-derivation.

    The 30 deg/s bound is a plausibility ceiling for THIS probe's command
    envelope: linear 300/400, |angular| 120/180, always driving forward, where
    the fastest steady rotation measured is 13.431 deg/s and the fastest clean
    interval 15.35. Stationary pivots reach ~38 deg/s at angular 500 and would
    trip it. If either set widens, re-derive the bound before shipping.
    """
    assert _STEP_RESPONSE_VIO_MAX_PLAUSIBLE_RATE_DEG_PER_S == 30.0
    # ~2x the worst clean interval ever observed, 5x below the discontinuity.
    assert _STEP_RESPONSE_VIO_MAX_PLAUSIBLE_RATE_DEG_PER_S > 2 * 13.431
    assert _STEP_RESPONSE_VIO_MAX_PLAUSIBLE_RATE_DEG_PER_S < 149.79 / 3

    base = {
        "entity_id": "lawn_mower.mower",
        "route_start": {"x": 0.0, "y": 0.0},
        "corridor_polygon": [
            {"x": -50.0, "y": -50.0},
            {"x": 50.0, "y": -50.0},
            {"x": 50.0, "y": 50.0},
            {"x": -50.0, "y": 50.0},
        ],
    }

    def _admits(**overrides: object) -> bool:
        try:
            STEP_RESPONSE_PROBE_SCHEMA({**base, **overrides})
        except vol.Invalid:
            return False
        return True

    assert {s for s in (200, 300, 400, 500, 600) if _admits(linear_speed=s)} == {
        300,
        400,
    }
    admitted_angular = {
        a
        for a in (60, 90, 120, 180, 240, 300, 500, -60, -120, -180, -500)
        if _admits(step_angular_speed=a)
    }
    assert admitted_angular == {120, 180, -120, -180}
