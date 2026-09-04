"""Tests for night motion: night mirror, night refusals, night segment caps."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import voluptuous as vol

from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.services import (
    RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA,
    RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA,
    _custom_path_telemetry_snapshot,
    _map_heading_to_toward_degrees,
    _raw_pymammotion_execute_multi_segment,
    _raw_pymammotion_execute_vector_segment,
    _toward_to_map_heading_degrees,
)

from .conftest import _pulse_coordinator


def test_night_mirror_helpers_are_an_involution() -> None:
    """The night map/toward conversion is a reflection, never an offset."""
    for heading in range(0, 360, 10):
        assert _toward_to_map_heading_degrees(
            _map_heading_to_toward_degrees(heading)
        ) == pytest.approx(heading)
    assert _map_heading_to_toward_degrees(80.0) < _map_heading_to_toward_degrees(60.0)


@pytest.mark.parametrize(
    ("toward", "map_heading"),
    [
        (162.7649, 286.7914),
        (-85.9472, 175.9560),
        (172.1591, 278.86),
    ],
)
def test_night_mirror_reproduces_low_noise_forward_legs(
    toward: float, map_heading: float
) -> None:
    """Long 0.45-0.66 m baselines make these stronger pins than short drives."""
    assert _toward_to_map_heading_degrees(toward) == pytest.approx(map_heading, abs=1.0)


@pytest.mark.asyncio
async def test_night_dry_run_uses_mirror_and_preserves_vio_echo() -> None:
    """Night alone changes conversion; VIO's evidence dict remains exact."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    points = [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}]

    night = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        points,
        turn_mode="night",
        calibrated_forward_heading_offset_degrees=0.0,
        sample_delays=(0,),
    )
    vio = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        points,
        turn_mode="vio",
        calibrated_forward_heading_offset_degrees=116.5,
        sample_delays=(0,),
    )

    assert night["target_reported_heading_degrees"] == pytest.approx(90.13)
    assert night["heading_calibration"] == {
        "model": "mirror",
        "formula": "target_toward_heading = toward_mirror_degrees - target_map_heading",
        "toward_mirror_degrees": 90.13,
        "target_map_heading_degrees": 0.0,
        "calibrated_forward_heading_offset_degrees": 0.0,
        "calibrated_forward_heading_offset_applied": False,
        "target_reported_heading_degrees": pytest.approx(90.13),
    }
    assert vio["target_heading_degrees"] == pytest.approx(243.5)
    assert vio["heading_calibration"] == {
        "formula": "target_reported_heading = target_map_heading - calibrated_forward_heading_offset",
        "target_map_heading_degrees": 0.0,
        "calibrated_forward_heading_offset_degrees": 116.5,
        "target_reported_heading_degrees": pytest.approx(243.5),
    }


def test_night_schema_defaults_and_domain() -> None:
    """Night is opt-in; omitted turn_mode remains VIO."""
    vector = RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA(
        {"entity_id": "lawn_mower.test", "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}]}
    )
    multi = RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA(
        {"entity_id": "lawn_mower.test", "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}]}
    )
    assert vector["turn_mode"] == multi["turn_mode"] == "vio"
    assert vector["night_angular_speed"] == multi["night_angular_speed"] == 500
    assert vector["toward_mirror_degrees"] == multi["toward_mirror_degrees"] == 90.13
    assert (
        RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA(
            {
                "entity_id": "lawn_mower.test",
                "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
                "turn_mode": "night",
            }
        )["turn_mode"]
        == "night"
    )
    for schema in (
        RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA,
        RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA,
    ):
        explicit_fixed_budget = schema(
            {
                "entity_id": "lawn_mower.test",
                "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
                "turn_mode": "night",
                "max_linear_pulse_ceiling": None,
            }
        )
        assert explicit_fixed_budget["max_linear_pulse_ceiling"] is None
    with pytest.raises(vol.Invalid):
        RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA(
            {
                "entity_id": "lawn_mower.test",
                "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
                "turn_mode": "nite",
            }
        )


@pytest.mark.asyncio
async def test_night_conversion_ignores_offset_and_honours_mirror_override() -> None:
    """The frozen additive key cannot influence a night target."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    points = [{"x": 1.0, "y": 1.0}, {"x": 1.1, "y": 1.0}]
    zero = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        points,
        turn_mode="night",
        calibrated_forward_heading_offset_degrees=0.0,
        sample_delays=(0,),
    )
    frozen = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        points,
        turn_mode="night",
        calibrated_forward_heading_offset_degrees=116.5,
        sample_delays=(0,),
    )
    overridden = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        points,
        turn_mode="night",
        toward_mirror_degrees=90.205,
        sample_delays=(0,),
    )
    assert zero["target_heading_degrees"] == frozen["target_heading_degrees"]
    assert overridden["target_heading_degrees"] - frozen[
        "target_heading_degrees"
    ] == pytest.approx(0.075)


@pytest.mark.asyncio
async def test_night_has_no_vio_gates_and_vio_has_no_night_gates() -> None:
    """Mode-specific gates never leak into the other control spine."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=0, brightness=0, track_feature_num=0
    )
    points = [{"x": 1.0, "y": 1.0}, {"x": 5.0, "y": 1.0}]
    night = await _raw_pymammotion_execute_vector_segment(
        coordinator, [*points[:1], {"x": 1.8, "y": 1.0}], turn_mode="night"
    )
    vio = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        points,
        turn_mode="vio",
        max_linear_pulse_ceiling=14,
    )
    night_names = {gate["name"] for gate in night["safety_gates"]}
    vio_names = {gate["name"] for gate in vio["safety_gates"]}
    assert {"vio_active", "vio_feed_live"}.isdisjoint(night_names)
    assert {
        "night_requires_precise_rtk",
        "night_linear_loop_unsupported",
        "night_segment_too_long",
    }.isdisjoint(vio_names)


@pytest.mark.asyncio
async def test_night_dispatch_and_legacy_refresh_containment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Night gets 500+refresh while legacy keeps its omitted refresh kwarg."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    calls: list[dict[str, object]] = []

    async def fake_turn(*_args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"stop_reason": "max_commands_reached", "commands_sent": 1}

    monkeypatch.setattr(
        mammotion_services, "_raw_pymammotion_turn_to_heading", fake_turn
    )
    points = [{"x": 1.0, "y": 1.0}, {"x": 1.8, "y": 1.0}]
    await _raw_pymammotion_execute_vector_segment(
        coordinator,
        points,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="night",
        motion_refresh_interval_ms=200,
    )
    await _raw_pymammotion_execute_vector_segment(
        coordinator,
        points,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="legacy",
        motion_refresh_interval_ms=200,
    )
    assert calls[0]["angular_speed_fast"] == calls[0]["angular_speed_slow"] == 500
    assert calls[0]["motion_refresh_interval_ms"] == 200
    assert calls[1]["angular_speed_fast"] == calls[1]["angular_speed_slow"] == 180
    assert "motion_refresh_interval_ms" not in calls[1]


@pytest.mark.asyncio
async def test_night_real_gates_refuse_unsupported_inputs() -> None:
    """RTK, reach-loop and length refusals happen before any dispatch."""
    coordinator = _pulse_coordinator(
        position=(1.0, 1.0, 0.0), rtk_status=3, pos_level=1
    )
    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 2.5, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="night",
        max_linear_pulse_ceiling=14,
    )
    assert result["stop_reason"] == "safety_gates_failed"
    assert {
        "night_requires_precise_rtk",
        "night_linear_loop_unsupported",
        "night_segment_too_long",
    } <= set(result["blockers"])
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_night_refuses_degraded_rtk_despite_schema_override() -> None:
    """The generic degraded-RTK escape hatch is never forwarded into night."""
    points = [{"x": 1.0, "y": 1.0}, {"x": 1.8, "y": 1.0}]
    parsed = RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA(
        {
            "entity_id": "lawn_mower.test",
            "points": points,
            "turn_mode": "night",
            "allow_degraded_rtk": True,
            "dry_run": False,
            "confirm_blades_off": True,
            "confirm_clear_area": True,
        }
    )
    assert parsed["allow_degraded_rtk"] is True

    coordinator = _pulse_coordinator(
        position=(1.0, 1.0, 0.0), rtk_status=3, pos_level=1
    )
    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        points,
        turn_mode=parsed["turn_mode"],
        dry_run=parsed["dry_run"],
        confirm_blades_off=parsed["confirm_blades_off"],
        confirm_clear_area=parsed["confirm_clear_area"],
    )

    assert "night_requires_precise_rtk" in result["blockers"]
    assert result["stop_reason"] == "safety_gates_failed"
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_night_length_gate_handles_short_and_missing_position() -> None:
    """Short legs pass the night gate; missing position returns cleanly."""
    short = await _raw_pymammotion_execute_vector_segment(
        _pulse_coordinator(position=(1.0, 1.0, 0.0)),
        [{"x": 1.0, "y": 1.0}, {"x": 1.8, "y": 1.0}],
        turn_mode="night",
    )
    short_gate = next(
        gate
        for gate in short["safety_gates"]
        if gate["name"] == "night_segment_too_long"
    )
    assert short_gate["passed"] is True

    missing = await _raw_pymammotion_execute_vector_segment(
        _pulse_coordinator(position=(None, None, None)),
        [{"x": 1.0, "y": 1.0}, {"x": 1.8, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="night",
    )
    assert missing["stop_reason"] == "position_unavailable"


@pytest.mark.asyncio
async def test_night_refuses_zero_command_heading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A latched in-tolerance toward cannot authorize forward motion."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    fake_turn = AsyncMock(
        return_value={"stop_reason": "target_heading_reached", "commands_sent": 0}
    )
    monkeypatch.setattr(
        mammotion_services, "_raw_pymammotion_turn_to_heading", fake_turn
    )
    result = await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.8, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="night",
    )
    assert result["stop_reason"] == "night_heading_unverified"
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_night_multi_segment_refused_before_segments(dry_run: bool) -> None:
    """Night v1 never begins a multi-segment chain."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [
            {"x": 1.0, "y": 1.0},
            {"x": 1.5, "y": 1.0},
            {"x": 1.8, "y": 1.2},
        ],
        dry_run=dry_run,
        confirm_blades_off=not dry_run,
        confirm_clear_area=not dry_run,
        max_real_segments=2,
        turn_mode="night",
    )
    assert result["stop_reason"] == "night_multi_segment_unsupported"
    assert result["segments_executed"] == 0
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_night_single_segment_through_multi_forwards_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A two-point multi call forwards both night-specific parameters."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    forwarded: dict[str, object] = {}

    async def fake_vector(*_args: object, **kwargs: object) -> dict[str, object]:
        forwarded.update(kwargs)
        return {
            "valid": True,
            "stop_reason": "dry_run",
            "blockers": [],
            "phases": [{"passed": True}, {"passed": True}],
            "progress_diagnostics": [],
        }

    monkeypatch.setattr(
        mammotion_services, "_raw_pymammotion_execute_vector_segment", fake_vector
    )
    result = await _raw_pymammotion_execute_multi_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.8, "y": 1.0}],
        turn_mode="night",
        night_angular_speed=501,
        toward_mirror_degrees=90.205,
    )
    assert result["stop_reason"] == "dry_run"
    assert forwarded["turn_mode"] == "night"
    assert forwarded["night_angular_speed"] == 501
    assert forwarded["toward_mirror_degrees"] == pytest.approx(90.205)


async def _run_one_night_linear_pulse(
    monkeypatch: pytest.MonkeyPatch,
    *,
    after_position: tuple[float, float, float],
) -> dict[str, object]:
    """Run one synthetic night pulse using RTK position as the only aim input."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 90.13))

    async def fake_turn(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "stop_reason": "target_heading_reached",
            "commands_sent": 1,
            "command_results": [],
            "samples": [],
            "final_telemetry": _custom_path_telemetry_snapshot(coordinator),
        }

    async def fake_send(*_args: object, **kwargs: object) -> None:
        if int(kwargs.get("linear_speed", 0)) > 0:
            x, y, toward = after_position
            coordinator.data.mowing_state.pos_x = x
            coordinator.data.mowing_state.pos_y = y
            coordinator.data.mowing_state.toward = toward

    async def fake_refresh(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "refresh_enabled": True,
            "refresh_interval_ms": 200,
            "refresh_commands_sent": 10,
        }

    async def fake_settle(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "settled": True,
            "moved": True,
            "wait_seconds": 0.0,
            "feed_stale": False,
            "settle_polls": [],
        }

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(
        mammotion_services, "_raw_pymammotion_turn_to_heading", fake_turn
    )
    monkeypatch.setattr(mammotion_services, "_motion_refresh_window", fake_refresh)
    monkeypatch.setattr(mammotion_services, "_settle_linear_position_feed", fake_settle)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)
    coordinator.manager.send_command_with_args.side_effect = fake_send
    return await _raw_pymammotion_execute_vector_segment(
        coordinator,
        [{"x": 1.0, "y": 1.0}, {"x": 1.8, "y": 1.0}],
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        turn_mode="night",
        max_linear_commands=1,
        waypoint_tolerance=0.08,
        motion_refresh_interval_ms=200,
        sample_delays=(0,),
    )


@pytest.mark.asyncio
async def test_night_stops_when_reaim_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A projected miss outside the disc stops instead of driving onward."""
    result = await _run_one_night_linear_pulse(
        monkeypatch, after_position=(1.25, 1.25, 45.0)
    )
    assert result["stop_reason"] == "night_reaim_required_but_unavailable"
    assert result["night_aim"][-1]["decision"] == "stop_reaim_unavailable"


@pytest.mark.asyncio
async def test_night_drives_on_when_projection_lands_inside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared landing projection permits a small, harmless aim error."""
    result = await _run_one_night_linear_pulse(
        monkeypatch, after_position=(1.30, 1.01, 88.0)
    )
    assert result["night_aim"][-1]["decision"] == "drive_on_projects_inside_tolerance"
    assert result["stop_reason"] == "max_linear_commands_reached"


@pytest.mark.asyncio
async def test_night_refuses_reverse_recovery_after_pulse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pulse travelling away from the target triggers the reverse guard."""
    result = await _run_one_night_linear_pulse(
        monkeypatch, after_position=(0.75, 1.0, 270.0)
    )
    assert result["stop_reason"] == "target_requires_reverse_recovery"
    assert result["night_aim"][-1]["decision"] == "reverse_recovery_required"


@pytest.mark.asyncio
async def test_night_refuses_forward_after_pulse_passes_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The residual bearing catches the beta54 third-pulse overshoot.

    Measured 2026-08-14: pulse 2 crossed the waypoint and settled only 2.66 mm
    outside tolerance. The old night check reused the PRE-pulse bearing, called
    the correctly aimed pulse safe, then sent another forward write away from
    the now-behind target. Replaying that geometry must invoke the existing
    reverse-recovery refusal before another command can be considered.
    """
    result = await _run_one_night_linear_pulse(
        monkeypatch, after_position=(1.883, 1.0, 90.13)
    )
    assert result["stop_reason"] == "target_requires_reverse_recovery"
    assert result["linear_commands_sent"] == 1
    assert result["night_aim"][-1]["bearing_to_target_degrees"] == pytest.approx(180.0)
    assert result["night_aim"][-1]["decision"] == "reverse_recovery_required"


@pytest.mark.asyncio
async def test_night_does_not_claim_aim_below_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 5 cm pulse records evidence without claiming a noisy heading."""
    result = await _run_one_night_linear_pulse(
        monkeypatch, after_position=(1.05, 1.0, 90.13)
    )
    assert result["night_aim"][-1]["decision"] == "below_aim_baseline"
    assert "aim_error_degrees" not in result["night_aim"][-1]
    assert result["stop_reason"] == "max_linear_commands_reached"


@pytest.mark.asyncio
async def test_night_aim_uses_rtk_and_records_observed_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No VIO read is needed; the observed mirror is recorded, never gated."""

    def vio_must_not_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("night mode read VIO")

    monkeypatch.setattr(mammotion_services, "_vio_reading", vio_must_not_run)
    result = await _run_one_night_linear_pulse(
        monkeypatch, after_position=(1.30, 1.01, 88.0)
    )
    record = result["night_aim"][-1]
    assert record["aim_error_degrees"] is not None
    assert record["observed_toward_mirror_degrees"] == pytest.approx(
        (record["movement_vector_heading_degrees"] + 88.0) % 360
    )


@pytest.mark.asyncio
async def test_vio_turn_kwargs_ignore_night_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supplying night knobs cannot change the dispatched VIO turn payload."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=10.0, vio_state=2, brightness=100, track_feature_num=80
    )
    calls: list[dict[str, object]] = []

    async def fake_staged(*_args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"stop_reason": "max_commands_reached", "commands_sent": 1}

    monkeypatch.setattr(mammotion_services, "_vio_turn_to_heading_staged", fake_staged)
    raw_turn = AsyncMock()
    monkeypatch.setattr(
        mammotion_services, "_raw_pymammotion_turn_to_heading", raw_turn
    )
    common: dict[str, object] = {
        "dry_run": False,
        "confirm_blades_off": True,
        "confirm_clear_area": True,
        "turn_mode": "vio",
        "vio_heading_offset_degrees": 0.0,
        "motion_refresh_interval_ms": 200,
        "sample_delays": (0,),
    }
    points = [{"x": 1.0, "y": 1.0}, {"x": 1.8, "y": 1.0}]
    await _raw_pymammotion_execute_vector_segment(coordinator, points, **common)
    await _raw_pymammotion_execute_vector_segment(
        coordinator,
        points,
        **common,
        night_angular_speed=777,
        toward_mirror_degrees=180.0,
    )
    assert calls[0] == calls[1]
    assert calls[0]["target_vision_heading"] == pytest.approx(0.0)
    assert calls[0]["motion_refresh_interval_ms"] == 200
    raw_turn.assert_not_called()
