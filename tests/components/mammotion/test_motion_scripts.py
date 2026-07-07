"""Tests for Mammotion motion readiness scripts."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import mammotion_click_go_smoke  # noqa: E402
import mammotion_forward_two_pulse_latency  # noqa: E402
import mammotion_ha_helpers  # noqa: E402
import mammotion_motion_suite  # noqa: E402


def test_call_readiness_with_runtime_retry_retries_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty readiness response is retried after waiting for runtime."""
    calls: list[tuple[str, str]] = []

    def fake_post_service(
        ha_url: str,
        token: str,
        domain: str,
        service: str,
        payload: dict,
        timeout: int,
    ) -> dict:
        calls.append((domain, service))
        if service == "export_runtime_state":
            return {"position": {"x": 1.0, "y": 2.0, "toward": 3.0}}
        if len([item for item in calls if item[1] == "raw_vector_readiness_test"]) == 1:
            return {}
        return {"ready_for_multi_segment": True}

    monkeypatch.setattr(mammotion_ha_helpers, "post_service", fake_post_service)

    result = mammotion_ha_helpers.call_readiness_with_runtime_retry(
        "http://ha",
        "token",
        "raw_vector_readiness_test",
        {"entity_id": "lawn_mower.test"},
        timeout=30,
        wait_runtime=True,
        runtime_timeout=30,
    )

    assert result == {"ready_for_multi_segment": True}
    assert calls == [
        ("mammotion", "export_runtime_state"),
        ("mammotion", "raw_vector_readiness_test"),
        ("mammotion", "export_runtime_state"),
        ("mammotion", "raw_vector_readiness_test"),
    ]


def test_call_readiness_with_runtime_retry_can_skip_runtime_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-wait-runtime style calls do not retry an empty response."""
    calls: list[tuple[str, str]] = []

    def fake_post_service(
        ha_url: str,
        token: str,
        domain: str,
        service: str,
        payload: dict,
        timeout: int,
    ) -> dict:
        calls.append((domain, service))
        return {}

    monkeypatch.setattr(mammotion_ha_helpers, "post_service", fake_post_service)

    result = mammotion_ha_helpers.call_readiness_with_runtime_retry(
        "http://ha",
        "token",
        "raw_vector_readiness_test",
        {"entity_id": "lawn_mower.test"},
        timeout=30,
        wait_runtime=False,
        runtime_timeout=30,
    )

    assert result == {}
    assert calls == [("mammotion", "raw_vector_readiness_test")]


def test_motion_suite_summary_includes_multi_segment_results(
    tmp_path: Path,
) -> None:
    """Suite summary keeps multi-segment readiness separate from vector readiness."""
    vector = {
        "dry_run": False,
        "ready_for_multi_segment": True,
        "ready_for_multi_point": False,
        "aligned_vector_ready": True,
        "positive_turn_vector_ready": True,
        "negative_turn_vector_ready": True,
        "real_steps_run": 3,
        "failed_phase": None,
        "blockers": [],
        "recommended_next_step": "implement_guarded_multi_segment_wrapper",
    }
    multi_dry_run = {
        "stop_reason": "dry_run",
        "ready_for_multi_segment": True,
    }
    multi_real = {
        "stop_reason": "max_real_segments_reached",
        "real_segments_executed": 1,
        "failed_segment_index": None,
    }

    summary = mammotion_motion_suite._summary(  # noqa: SLF001
        vector,
        tmp_path,
        multi_dry_run=multi_dry_run,
        multi_real=multi_real,
    )

    assert summary["passed"] is True
    assert summary["multi_segment_dry_run_passed"] is True
    assert summary["multi_segment_real_passed"] is True
    assert summary["multi_segment_real_segments_executed"] == 1


def test_motion_suite_multi_segment_points_use_live_position() -> None:
    """Generated multi-segment path starts at live map-local runtime position."""
    args = type(
        "Args",
        (),
        {
            "target_distance": 0.1,
            "turn_delta_degrees": 10.0,
            "calibrated_forward_heading_offset_degrees": 116.5,
        },
    )()

    points = mammotion_motion_suite._multi_segment_points(  # noqa: SLF001
        {"position": {"x": 1.234, "y": -2.345, "toward": 100.0}},
        args,
    )

    assert len(points) == 3
    assert points[0] == {"x": 1.234, "y": -2.345}


def test_motion_suite_quick_profile_overrides_defaults() -> None:
    """Quick profile shortens sampling and timeout defaults."""
    args = Namespace(
        sample_delays=list(mammotion_motion_suite.DEFAULT_SAMPLE_DELAYS),
        runtime_timeout=180,
        timeout=720,
    )

    mammotion_motion_suite._apply_quick_profile(args)  # noqa: SLF001

    assert args.sample_delays == mammotion_motion_suite.QUICK_SAMPLE_DELAYS
    assert args.runtime_timeout == 90
    assert args.timeout == 240


def test_motion_suite_quick_profile_respects_explicit_overrides() -> None:
    """Quick profile keeps user-provided non-default timings unchanged."""
    args = Namespace(
        sample_delays=[0.0, 12.0],
        runtime_timeout=45,
        timeout=99,
    )

    mammotion_motion_suite._apply_quick_profile(args)  # noqa: SLF001

    assert args.sample_delays == [0.0, 12.0]
    assert args.runtime_timeout == 45
    assert args.timeout == 99


def test_two_pulse_quick_profile_overrides_defaults() -> None:
    """Two-pulse quick profile shortens pulse and telemetry windows."""
    args = Namespace(
        pulse_gap_seconds=5.0,
        telemetry_timeout_seconds=60.0,
        telemetry_sample_interval_seconds=1.0,
        runtime_timeout=180,
        timeout=180,
    )

    mammotion_forward_two_pulse_latency._apply_quick_profile(args)  # noqa: SLF001

    assert args.pulse_gap_seconds == 3.0
    assert args.telemetry_timeout_seconds == 35.0
    assert args.telemetry_sample_interval_seconds == 0.5
    assert args.runtime_timeout == 90
    assert args.timeout == 120


def test_two_pulse_quick_profile_respects_explicit_overrides() -> None:
    """Two-pulse quick profile preserves explicit non-default values."""
    args = Namespace(
        pulse_gap_seconds=8.0,
        telemetry_timeout_seconds=20.0,
        telemetry_sample_interval_seconds=2.0,
        runtime_timeout=33,
        timeout=70,
    )

    mammotion_forward_two_pulse_latency._apply_quick_profile(args)  # noqa: SLF001

    assert args.pulse_gap_seconds == 8.0
    assert args.telemetry_timeout_seconds == 20.0
    assert args.telemetry_sample_interval_seconds == 2.0
    assert args.runtime_timeout == 33
    assert args.timeout == 70


def test_one_segment_payload_uses_runtime_start_and_target_end() -> None:
    """One-segment payload starts from live runtime position and ends at target."""
    payload = mammotion_ha_helpers.build_one_segment_vector_payload(
        "lawn_mower.test",
        {"position": {"x": 1.23456, "y": -2.34567, "toward": 90}},
        {"x": 7.89012, "y": 3.21098},
    )

    assert payload["points"] == [
        {"x": 1.235, "y": -2.346},
        {"x": 7.89, "y": 3.211},
    ]


def test_one_segment_payload_defaults_to_non_moving_dry_run() -> None:
    """Default payload is dry-run and keeps confirmation gates false."""
    payload = mammotion_ha_helpers.build_one_segment_vector_payload(
        "lawn_mower.test",
        {"position": {"x": 1.0, "y": 2.0, "toward": 30}},
        {"x": 1.1, "y": 2.2},
    )

    assert payload["dry_run"] is True
    assert payload["confirm_blades_off"] is False
    assert payload["confirm_clear_area"] is False
    assert payload["prefer_ble"] is True
    assert payload["max_turn_commands"] == 1
    assert payload["max_linear_commands"] == 1
    assert payload["sample_delays"] == [0.0, 5.0, 10.0]


def test_one_segment_payload_allows_real_mode_confirmation_flags() -> None:
    """Real-mode payload carries explicit confirmation flags when provided."""
    payload = mammotion_ha_helpers.build_one_segment_vector_payload(
        "lawn_mower.test",
        {"position": {"x": 1.0, "y": 2.0, "toward": 30}},
        {"x": 1.1, "y": 2.2},
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        sample_delays=[0, 4, 8],
    )

    assert payload["dry_run"] is False
    assert payload["confirm_blades_off"] is True
    assert payload["confirm_clear_area"] is True
    assert payload["sample_delays"] == [0.0, 4.0, 8.0]


def test_one_segment_payload_rejects_invalid_runtime_position_type() -> None:
    """Payload builder requires runtime_state.position to be an object."""
    with pytest.raises(TypeError, match="runtime_state.position is required"):
        mammotion_ha_helpers.build_one_segment_vector_payload(
            "lawn_mower.test",
            {"position": None},
            {"x": 1.1, "y": 2.2},
        )


def test_one_segment_payload_rejects_missing_target_coordinates() -> None:
    """Payload builder requires target x and y values."""
    with pytest.raises(ValueError, match="target_point must include x and y"):
        mammotion_ha_helpers.build_one_segment_vector_payload(
            "lawn_mower.test",
            {"position": {"x": 1.0, "y": 2.0, "toward": 30}},
            {"x": 1.1},
        )


def test_click_go_smoke_target_uses_explicit_coordinates() -> None:
    """Explicit target coordinates take precedence over runtime offsets."""
    args = Namespace(
        target_x=9.876,
        target_y=-1.234,
        offset_x=0.1,
        offset_y=0.0,
    )
    target = mammotion_click_go_smoke._resolve_target(  # noqa: SLF001
        {"position": {"x": 1.0, "y": 2.0}},
        args,
    )
    assert target == {"x": 9.876, "y": -1.234}


def test_click_go_smoke_target_uses_runtime_offset_when_unset() -> None:
    """Runtime-relative target derives from live position plus offsets."""
    args = Namespace(
        target_x=None,
        target_y=None,
        offset_x=0.1254,
        offset_y=-0.2559,
    )
    target = mammotion_click_go_smoke._resolve_target(  # noqa: SLF001
        {"position": {"x": 1.111, "y": 2.222}},
        args,
    )
    assert target == {"x": 1.236, "y": 1.966}


def test_click_go_smoke_preview_payload_shape() -> None:
    """Preview payload mirrors one-segment click/go card contract."""
    payload = mammotion_click_go_smoke._preview_payload(  # noqa: SLF001
        "lawn_mower.test",
        [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
        speed=0.2,
        area_hash="1234",
    )
    assert payload == {
        "entity_id": "lawn_mower.test",
        "points": [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
        "speed": 0.2,
        "blade_mode": "off",
        "area_hash": "1234",
    }
