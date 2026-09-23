"""Keep hardware-evidence regressions local when private docs are absent."""

from pathlib import Path

import pytest

_PRIVATE_EVIDENCE_TESTS = {
    "test_accepted_profile_check.py": {
        "test_the_accepted_snapshot_is_a_real_gate5_pass",
        "test_the_shipped_profile_is_the_accepted_one",
    },
    "test_heading_trust.py": {
        "test_mirror_predicts_the_driven_direction_on_banked_hardware_pulses",
        "test_additive_offset_is_wrong_by_87_degrees_on_the_same_pulses",
        "test_no_additive_constant_can_replace_the_reflection",
    },
    "test_map_task_visibility.py": {
        "test_reverse_recovery_guard_replays_both_gate4_passes",
    },
    "test_step_response_probe.py": {
        "test_min_speed_table_is_below_every_banked_full_window_rate",
    },
    "test_step_response_vio_scoring.py": {
        "test_banked_runs_score_as_the_findings_document_published",
        "test_tau_exists_only_when_2a_passes_and_omega_is_the_steady_half",
        "test_dark_vio_refuses_to_score_with_a_named_reason",
        "test_a_heading_frame_discontinuity_refuses_to_score_with_a_named_reason",
        "test_the_guard_is_state_independent_because_vio_state_never_sees_it",
    },
    "test_turn_primitives.py": {
        "test_final_approach_replays_gate4_1300ms_write_limits",
    },
    "test_vio_turn_feasibility.py": {
        "test_diagnose_still_classifies_the_retained_gate4_budget_exhaustion",
    },
    "test_analyze_phase1_capture.py": {
        "test_the_banked_arc_now_fails_on_STEP_COUNT_not_on_error",
    },
    "test_score_queue_measurement.py": {
        "test_reproduces_67_of_67_on_the_20260913_evidence",
    },
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip only tests that require evidence kept outside the public fork."""
    if (Path(__file__).resolve().parents[1] / "docs").is_dir():
        return

    private_evidence_missing = pytest.mark.skip(
        reason="private hardware evidence is stored locally"
    )
    for item in items:
        module_tests = _PRIVATE_EVIDENCE_TESTS.get(item.path.name, set())
        if item.name.split("[", 1)[0] in module_tests:
            item.add_marker(private_evidence_missing)
