"""Offline tests for the E-VIO scoring of step-response criteria 2a/2b.

Adopted 2026-09-01 per `docs/findings-rtk-vio-course-rate-scoring-20260831.md`
(predeclared in `docs/predeclared-rtk-vio-course-rate-scoring-20260831.md`):
2a is scored from VIO heading via half-phase mean-rate agreement, 2b keeps its
last-two-rates semantics on the VIO channel, omega/tau come from the same
channel and tau exists only when 2a passes, and dark VIO refuses to score
rather than falling back to the noise-bound RTK chord rule.

The four banked route-1 runs in `docs/raw-samples/` are the regression
fixtures — 549 real samples from supervised runs, including the two verdicts
that FLIP versus the published RTK scoring. Pinning the flips is the point:
nobody may quietly restore the old instrument without these failing.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from custom_components.mammotion import services
from custom_components.mammotion.services import (
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
# The four banked runs — including the two flips vs the published RTK verdicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "step_ms", "pass_2a", "half_diff", "pass_2b"),
    [
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
    ],
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
