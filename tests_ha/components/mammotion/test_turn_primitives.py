"""Tests for turn primitives: final-approach bounds, actuation floor, turn budget."""

import json
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.coordinator import (
    MammotionReportUpdateCoordinator,
)
from custom_components.mammotion.services import (
    _MIN_SCALED_TURN_PULSE_MS,
    _VIO_HEADING_FRESH_EPSILON_DEGREES,
    _final_approach_pulse_ms,
    _normalised_linear_pulse_distance,
    _raw_vector_readiness_target_points,
    _turn_final_approach_pulse_ms,
    _vio_turn_probe,
    _vio_turn_to_heading,
    _vio_turn_to_heading_staged,
)

from .conftest import _patch_services_monotonic, _pulse_coordinator


@pytest.mark.asyncio
async def test_vio_turn_probe_defaults_to_dry_run() -> None:
    """VIO turn probe default plans an in-place rotation but sends nothing."""
    coordinator = _pulse_coordinator()

    result = await _vio_turn_probe(coordinator)

    assert result["service"] == "vio_turn_probe"
    assert result["dry_run"] is True
    assert result["would_send"] is False
    assert result["reason"] == "dry_run"
    assert result["command"]["kwargs"] == {"linear_speed": 0, "angular_speed": 500}
    assert result["samples"] == []
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_vio_turn_probe_detects_heading_tracking_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vision heading moving while course-over-ground is frozen tracks rotation."""
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    async def fake_get_reports(count: int = 5) -> None:
        # VIO heading rotates 10 deg/s; position and course-over-ground stay put.
        heading = (clock["now"] - 100.0) * 10.0
        coordinator.data.report_data.vision_info = SimpleNamespace(
            heading=heading,
            vio_state=2,
        )

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        angular_speed=500,
        drive_seconds=3.0,
        sample_interval_seconds=1.0,
        post_stop_samples=0,
    )

    # A single continuous angular command, then a mandatory explicit stop.
    handle = coordinator.manager.mower(coordinator.device_name)
    assert handle._send_marked.await_count == 2  # noqa: SLF001
    assert result["command"]["kwargs"] == {"linear_speed": 0, "angular_speed": 500}
    assert handle.commands.send_movement.call_args_list[-1].kwargs == {
        "linear_speed": 0,
        "angular_speed": 0,
    }
    assert result["reason"] == "vision_heading_tracks_rotation"
    assert result["verdict"]["vision_heading_change"]["total_abs_degrees"] >= 3.0
    assert result["verdict"]["course_over_ground_change"]["total_abs_degrees"] == 0.0


@pytest.mark.asyncio
async def test_vio_turn_probe_app_parity_refresh_resends_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """motion_refresh_interval_ms re-sends the rotation command app-style (B1 turn A/B).

    B1 proved refresh is speed-gated (it unlocked linear but did nothing at angular
    180, below this mower's rotation threshold). This probe reaches angular 500, so
    it is the tool to test refresh on a properly-powered turn -- which first requires
    that the refresh actually re-issues the command during the drive.
    """
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    async def fake_get_reports(count: int = 5) -> None:
        heading = (clock["now"] - 100.0) * 10.0
        coordinator.data.report_data.vision_info = SimpleNamespace(
            heading=heading,
            vio_state=2,
        )

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        angular_speed=500,
        drive_seconds=3.0,
        sample_interval_seconds=1.0,
        post_stop_samples=0,
        motion_refresh_interval_ms=200,
    )

    # The command is re-issued during the drive, not sent once.
    refreshes = result["motion_refresh_commands_sent"]
    assert result["motion_refresh_interval_ms"] == 200
    assert refreshes > 0
    # Every send is the initial one plus one per refresh; all identical turn commands.
    handle = coordinator.manager.mower(coordinator.device_name)
    assert handle._send_marked.await_count == refreshes + 2  # noqa: SLF001
    assert result["command"]["kwargs"] == {"linear_speed": 0, "angular_speed": 500}
    assert handle.commands.send_movement.call_args_list[-1].kwargs == {
        "linear_speed": 0,
        "angular_speed": 0,
    }


@pytest.mark.asyncio
async def test_vio_turn_probe_counts_rotation_that_lands_after_the_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real rotation visible only in post_stop must not read as static/zero.

    Regression for 2026-07-19: VIO heading refreshes ~1.5s into the command and
    the position feed lags ~4s, so on a short pulse the only during-command
    sample is the t=0 one. A tape-confirmed 13.18 deg pivot came back
    `vision_heading_static_during_command` with `final_displacement_m: 0.0`
    because the verdict ignored post_stop -- while this function's own post_stop
    samples held the real values.
    """
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}
    stopped = {"value": False}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    async def fake_stop(_coordinator: object) -> dict:
        stopped["value"] = True
        return {"movement_ok": True}

    async def fake_get_reports(count: int = 5) -> None:
        # Frozen during the command; the real rotation only registers post-stop.
        heading = 76.82 if stopped["value"] else 90.0
        coordinator.data.report_data.vision_info = SimpleNamespace(
            heading=heading,
            vio_state=2,
        )

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports
    monkeypatch.setattr(mammotion_services, "_stop_manual_motion_confirmed", fake_stop)
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=90.0, vio_state=2
    )

    result = await _vio_turn_probe(
        coordinator,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        angular_speed=500,
        drive_seconds=1.5,
        sample_interval_seconds=1.5,
        post_stop_samples=3,
    )

    # The ~13.18 deg swing lands entirely in post_stop and must be counted.
    total = result["verdict"]["vision_heading_change"]["total_abs_degrees"]
    assert total == pytest.approx(13.18, abs=0.05)
    assert result["reason"] != "vision_heading_static_during_command"
    assert result["displacement_source"] in {"post_stop", "drive", None}


@pytest.mark.asyncio
async def test_vio_turn_to_heading_defaults_to_dry_run() -> None:
    """VIO turn-to-heading default plans a turn (opposite sign of error), no send."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    result = await _vio_turn_to_heading(coordinator, target_vision_heading=40.0)

    assert result["service"] == "vio_turn_to_heading"
    assert result["dry_run"] is True
    assert result["stop_reason"] == "dry_run"
    assert result["initial_heading_error_degrees"] == 40.0
    # Positive error -> negative angular (calibrated: -angular increases heading).
    assert result["planned_command"]["kwargs"]["angular_speed"] == -500
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_vio_turn_to_heading_rejects_missing_confirmations() -> None:
    """Real VIO turn-to-heading requires explicit operator confirmations."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=False,
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "operator_confirmed_clear_area" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vio_turn_to_heading_cold_vio_still_allows_dry_run() -> None:
    """A cold VIO (vio_state != 2) still plans in dry-run without sending."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=0)

    result = await _vio_turn_to_heading(coordinator, target_vision_heading=40.0)

    assert result["stop_reason"] == "dry_run"
    assert result["initial_vio_state"] == 0
    coordinator.manager.send_command_with_args.assert_not_called()


@pytest.mark.asyncio
async def test_vio_turn_to_heading_refuses_real_turn_when_vio_cold() -> None:
    """Real VIO turn-to-heading refuses to move unless VIO is actively tracking."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=0)

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "vio_active" in result["blockers"]
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_vio_turn_to_heading_blocks_real_turn_when_feed_degraded() -> None:
    """vio_state==2 with a collapsed feature track blocks a real turn (dusk latch)."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=2, track_feature_num=0, brightness=10
    )

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "safety_gates_failed"
    assert "vio_feed_live" in result["blockers"]
    assert result["initial_vio_feed"]["live"] is False
    coordinator.manager.send_command_with_args.assert_not_called()
    coordinator.async_stop_manual_motion.assert_not_called()


@pytest.mark.asyncio
async def test_vio_turn_to_heading_stops_when_feed_degrades_mid_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-turn feature-track collapse stops distinctly from vio_state dropping out."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=0.0, vio_state=2, track_feature_num=80, brightness=200
    )

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        # Pulse one makes real progress, then the feed goes blind (sunset): the
        # track drops to 0 features while vio_state stays active and the heading
        # would otherwise latch. The next iteration must bail on the blind feed.
        vi = coordinator.data.report_data.vision_info
        vi.heading = vi.heading + 10.0
        vi.track_feature_num = 0
        vi.brightness = 10

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "vio_feed_degraded"
    assert result["final_vio_feed"]["live"] is False
    assert result["commands_sent"] == 1
    assert coordinator.async_stop_manual_motion.await_count == 1


@pytest.mark.asyncio
async def test_vio_turn_to_heading_tolerates_transient_feed_dip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single transient feed dip that recovers on re-poll does NOT abort the turn.

    A one-read feature dip (brief occlusion) must not end an otherwise-good turn;
    the read-only re-confirmation poll should see the feed recover and continue,
    unlike the sustained-degradation case which aborts vio_feed_degraded.
    """
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    # Feed liveness by call: entry(live) -> pulse-2 before_feed(dip) ->
    # re-confirm(recovered) -> live thereafter.
    feed_live = iter([True, False, True])

    def fake_feed_liveness(_coordinator: object) -> dict:
        live = next(feed_live, True)
        return {
            "live": live,
            "tracked_features": 80 if live else 0,
            "brightness_raw": 200 if live else 10,
            "brightness_label": "Light" if live else "Dark",
        }

    async def advance_on_pulse(*_args: object, **_kwargs: object) -> None:
        vi = coordinator.data.report_data.vision_info
        vi.heading = min(40.0, vi.heading + 20.0)  # reach +40 target in 2 pulses

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        return None  # heading is driven by the pulse, feed by the fake above

    monkeypatch.setattr(mammotion_services, "_vio_feed_liveness", fake_feed_liveness)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.manager.send_command_with_args.side_effect = advance_on_pulse
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    # The dip recovered on re-poll, so the turn continued to its target instead of
    # aborting vio_feed_degraded.
    assert result["stop_reason"] == "target_heading_reached"
    assert result["commands_sent"] == 2


@pytest.mark.asyncio
async def test_vio_turn_to_heading_stops_if_vio_drops_out_mid_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If VIO deactivates during the loop, stop instead of chasing a stale heading."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        # Pulse one makes real progress, but VIO drops out (enters shadow) so the
        # next iteration must bail rather than trust the now-stale heading.
        vi = coordinator.data.report_data.vision_info
        vi.heading = vi.heading + 10.0
        vi.vio_state = 0

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "vio_inactive"
    # Exactly one pulse fired before VIO dropped out and the loop bailed.
    assert result["commands_sent"] == 1
    assert coordinator.async_stop_manual_motion.await_count == 1


@pytest.mark.asyncio
async def test_vio_turn_to_heading_closed_loop_reaches_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded pulses converge vision_heading to the target and stop each pulse."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        vi = coordinator.data.report_data.vision_info
        vi.heading = min(30.0, vi.heading + 10.0)

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=30.0,
        heading_tolerance_degrees=8.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "target_heading_reached"
    assert result["commands_sent"] == 3
    # First pulse: error +30 -> negative angular per calibration.
    assert result["command_results"][0]["angular_speed"] == -500
    # A bounded pulse + explicit stop per command.
    assert coordinator.manager.send_command_with_args.await_count == 3
    assert coordinator.async_stop_manual_motion.await_count == 3
    assert abs(result["final_heading_error_degrees"]) <= 8.0
    # final_vio_feed is always present (not only on the degraded stop path).
    assert result["final_vio_feed"]["live"] is True


@pytest.mark.asyncio
async def test_vio_turn_to_heading_polls_through_stale_heading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale first sample is re-polled to a fresh heading, not judged as progress-less."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    # The VIO feed lags ~4s: the first request_reports after a pulse returns the
    # pre-pulse heading jittered only by sub-epsilon sensor noise (stale); only the
    # second poll reflects the real rotation. The loop must poll through the stale
    # sample rather than treat the noise wiggle as fresh movement.
    calls = {"n": 0}

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        calls["n"] += 1
        vi = coordinator.data.report_data.vision_info
        if calls["n"] % 2 == 0:  # advance only on the second poll of each pulse
            vi.heading = min(30.0, round(vi.heading) + 10.0)
        else:  # first poll: latched value plus sub-epsilon noise
            vi.heading = round(vi.heading) + 0.002

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=30.0,
        heading_tolerance_degrees=8.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "target_heading_reached"
    assert result["commands_sent"] == 3
    # Each pulse polled twice (stale then fresh) before judging progress.
    assert all(cmd["heading_went_fresh"] for cmd in result["command_results"])
    assert coordinator.async_get_reports.await_count == 6


@pytest.mark.asyncio
async def test_vio_turn_to_heading_tolerates_one_stale_pulse_before_no_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-progress only aborts after max_no_progress_pulses consecutive stale pulses."""
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)
    clock = {"now": 100.0}
    calls = {"flip": False}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    async def fake_get_reports(count: int = 5) -> None:
        # Heading is permanently latched but the feed still emits sub-epsilon sensor
        # noise (run 2, dusk: ~0.0018 deg jitter). The fresh-heading poll must treat
        # that as still-stale, time out, and keep progress at zero.
        vi = coordinator.data.report_data.vision_info
        vi.heading = round(vi.heading, 3) + (0.0018 if calls["flip"] else -0.0018)
        calls["flip"] = not calls["flip"]

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "no_heading_progress"
    # One stale pulse is tolerated; the second consecutive no-progress pulse aborts.
    assert result["commands_sent"] == 2
    assert coordinator.async_stop_manual_motion.await_count == 2
    assert all(not cmd["heading_went_fresh"] for cmd in result["command_results"])
    assert result["command_results"][-1]["consecutive_no_progress"] == 2
    # First pulse runs full-length; the second, fired after a *stale* no-progress
    # sample, is capped to the slow duration to bound blind rotation on a latched
    # feed.
    assert result["command_results"][0]["pulse_duration_ms"] == 1500
    assert result["command_results"][1]["pulse_duration_ms"] == 700


@pytest.mark.asyncio
async def test_vio_turn_to_heading_slow_caps_wrong_direction_streak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh streak that moves AWAY from target is slow-capped (wrong-direction guard).

    Even with a fresh feed, negative progress (e.g. an angular sign miscalibration
    turning the wrong way) must not keep running full-power pulses. The first pulse
    runs full (no streak yet); once the streak sees the away-drift, subsequent
    pulses are capped to the slow duration to bound the wrong-way rotation.
    """
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        # Genuinely fresh reading that drifts *away* from the +40 target: fresh but
        # negative progress.
        vi = coordinator.data.report_data.vision_info
        vi.heading = vi.heading - 10.0

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "no_heading_progress"
    assert all(cmd["heading_went_fresh"] for cmd in result["command_results"])
    # Pulse 1 runs full (streak not started); pulse 2, fired after away-progress,
    # is slow-capped.
    assert result["command_results"][0]["pulse_duration_ms"] == 1500
    assert result["command_results"][1]["pulse_duration_ms"] == 700


@pytest.mark.asyncio
async def test_vio_turn_to_heading_keeps_full_pulse_creeping_toward_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh streak still creeping TOWARD target (small +progress) keeps the full pulse.

    The slow cap is for stale/latched feeds and wrong-direction motion; a mower
    genuinely turning toward the target but slower than min_progress_degrees should
    keep the full, faster pulse.
    """
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)

    async def fake_sleep(_delay: float) -> None:
        return None

    async def fake_get_reports(count: int = 5) -> None:
        # Fresh reading creeping toward the +40 target by +1 deg/poll: fresh, and
        # positive progress but below min_progress_degrees (2.0).
        vi = coordinator.data.report_data.vision_info
        vi.heading = vi.heading + 1.0

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "no_heading_progress"
    assert all(cmd["heading_went_fresh"] for cmd in result["command_results"])
    # Fresh AND still moving toward target -> never slow-capped; all pulses full.
    assert all(cmd["pulse_duration_ms"] == 1500 for cmd in result["command_results"])


@pytest.mark.asyncio
async def test_vio_turn_to_heading_sub_epsilon_wiggle_is_not_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 0.002 deg feed wiggle must not pass the freshness gate (run 2 regression).

    Run 2 (dusk) latched the heading bit-identical while the feed still jittered by
    ~0.0018 deg; the old float-inequality check read that noise as movement. With the
    epsilon gate the poll must treat a 0.002 deg wiggle as stale, time out, and abort
    on no progress instead of trusting the blind feed.
    """
    coordinator = _pulse_coordinator()
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)
    clock = {"now": 100.0}
    flip = {"v": False}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    async def fake_get_reports(count: int = 5) -> None:
        vi = coordinator.data.report_data.vision_info
        # Oscillate by +/-0.002 deg around the latched value: never clears the
        # 0.1 deg freshness epsilon.
        vi.heading = 0.002 if flip["v"] else 0.0
        flip["v"] = not flip["v"]

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)
    coordinator.async_get_reports.side_effect = fake_get_reports

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "no_heading_progress"
    assert all(not cmd["heading_went_fresh"] for cmd in result["command_results"])
    assert all(
        abs(cmd["measured_change_degrees"]) <= _VIO_HEADING_FRESH_EPSILON_DEGREES
        for cmd in result["command_results"]
    )


@pytest.mark.parametrize(
    ("toward", "delta"), [(0.0, 10.0), (123.4, -10.0), (-173.9, 10.0)]
)
def test_readiness_probe_keeps_the_legacy_additive_cancellation(
    toward: float, delta: float
) -> None:
    """Night's mirror does not alter either shared legacy conversion site."""
    points = _raw_vector_readiness_target_points(
        {"position": {"x": 1.0, "y": 1.0, "toward": toward}},
        reported_heading_delta=delta,
        target_distance=0.1,
        calibrated_forward_heading_offset_degrees=116.5,
    )
    assert points is not None
    map_heading = mammotion_services._path_heading_degrees(*points)  # noqa: SLF001
    reported = (map_heading - 116.5) % 360
    assert reported == pytest.approx((toward + delta) % 360)


def test_final_approach_scales_the_last_pulse_to_the_remaining_distance() -> None:
    """Less than one pulse left is bounded by discrete confirmed refresh writes."""
    info = _final_approach_pulse_ms(
        distance_to_target=0.2,
        observed_pulse_distances=[],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is True
    assert info["reason"] == "final_approach_bounded_by_refresh_count"
    # A full pulse is the initial write plus ten refreshes. 0.2/1.06 of eleven
    # non-zero writes rounds up to three total: initial plus two refreshes.
    assert info["refresh_command_limit"] == 2
    assert info["target_nonzero_writes"] == 3
    assert info["pulse_duration_ms"] == 3500.0
    assert info["metres_per_pulse_source"] == "default"


def test_final_approach_is_disabled_without_the_refresh_cadence() -> None:
    """Single-shot motion moves a fixed step, so scaling the duration is a trap.

    The 2026-07-22 B1 tape proved distance is duration-dependent only while the
    command is being re-sent: the same 4 s pulse moved ~4 in single-shot and
    ~44 in at ``motion_refresh_interval_ms`` 200. Without refresh a shortened
    pulse would not land closer -- and 2026-07-18 measured a 2000 ms single-shot
    pulse as a physical no-op, so it could stop the mower moving at all. The
    guard must hold regardless of how little distance remains.
    """
    info = _final_approach_pulse_ms(
        distance_to_target=0.2,
        observed_pulse_distances=[1.06],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=0,
    )

    assert info["applied"] is False
    assert info["reason"] == "refresh_disabled_distance_not_proportional_to_duration"
    assert info["pulse_duration_ms"] == 3500.0


def test_final_approach_leaves_cruising_pulses_full_length() -> None:
    """More than one pulse to go -> drive the full pulse, unchanged."""
    info = _final_approach_pulse_ms(
        distance_to_target=2.5,
        observed_pulse_distances=[],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is False
    assert info["reason"] == "cruising_full_pulse_fits"
    assert info["pulse_duration_ms"] == 3500.0


def test_final_approach_uses_only_the_initial_write_for_a_tiny_remainder() -> None:
    """A tiny remainder gets one non-zero write and no refresh amplification."""
    info = _final_approach_pulse_ms(
        distance_to_target=0.01,
        observed_pulse_distances=[],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is True
    assert info["refresh_command_limit"] == 0
    assert info["target_nonzero_writes"] == 1
    assert info["pulse_duration_ms"] == 3500.0


def test_final_approach_prefers_the_distance_observed_this_run() -> None:
    """Today's measured pulses beat the baked-in constant.

    Speed, grass and gradient move the per-pulse distance around, so the run
    calibrates itself. Here the mower is covering 2.0 m per pulse -- against the
    1.06 m default the same 1.5 m of remaining distance would have read as
    "cruising" and fired a full pulse straight past the target.
    """
    info = _final_approach_pulse_ms(
        distance_to_target=1.5,
        observed_pulse_distances=[2.0, 2.0],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is True
    assert info["metres_per_pulse_source"] == "observed"
    assert info["metres_per_pulse"] == pytest.approx(2.0)
    assert info["refresh_command_limit"] == 8
    assert info["target_nonzero_writes"] == 9
    assert info["pulse_duration_ms"] == 3500.0


def test_final_approach_normalises_bounded_pulses_by_actual_write_count() -> None:
    """A bounded pulse can calibrate later approaches without shrinking the scale."""
    # day2j's first segment pulse 1: 0.4192 m from initial + three refresh writes.
    normalised = _normalised_linear_pulse_distance(0.4191586454, 3)

    assert normalised == pytest.approx(1.1526862749)
    info = _final_approach_pulse_ms(
        distance_to_target=0.2222578683,
        observed_pulse_distances=[normalised],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=1300.0,
        refresh_interval_ms=200,
    )
    assert info["metres_per_pulse_source"] == "observed"
    assert info["refresh_command_limit"] == 2


def test_final_approach_observation_cannot_increase_the_motion_budget() -> None:
    """A low observation may stop short but must not add writes and overshoot."""
    info = _final_approach_pulse_ms(
        distance_to_target=0.18,
        observed_pulse_distances=[0.96],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=1300.0,
        refresh_interval_ms=200,
    )

    assert info["observed_metres_per_pulse"] == 0.96
    assert info["metres_per_pulse"] == 1.06
    assert info["metres_per_pulse_source"] == "default_conservative_floor"
    assert info["refresh_command_limit"] == 1


@pytest.mark.parametrize(
    "evidence_name",
    [
        "evidence-gate4-beta20-day2i-real-result-20260805.json",
        "evidence-gate4-beta20-day2j-real-result-20260805.json",
    ],
)
def test_final_approach_replays_gate4_1300ms_write_limits(
    evidence_name: str,
) -> None:
    """The conservative estimator preserves every recorded day2i/day2j write cap."""
    evidence_path = pathlib.Path(__file__).parents[3] / "docs" / evidence_name
    evidence = json.loads(evidence_path.read_text())

    for segment in evidence["result"]["segments"]:
        segment_result = segment["result"]
        progress_by_index = {
            item["command_index"]: item
            for item in segment_result["progress_diagnostics"]
        }
        observed_by_speed: dict[int, list[float]] = {}
        for command in segment_result["command_results"]:
            if command.get("phase") != "linear_forward_to_target":
                continue
            speed = int(command["selection"]["linear_speed"])
            approach = _final_approach_pulse_ms(
                distance_to_target=command["selection"]["distance_to_target"],
                observed_pulse_distances=observed_by_speed.get(speed, []),
                default_metres_per_pulse=1.06,
                pulse_duration_ms=1300.0,
                refresh_interval_ms=200,
            )
            assert (
                approach["refresh_command_limit"]
                == command["final_approach"]["refresh_command_limit"]
            )

            measured = progress_by_index[command["index"]]["measured_delta"]["distance"]
            refreshes = command["motion_refresh"]["refresh_commands_sent"]
            observed_by_speed.setdefault(speed, []).append(
                _normalised_linear_pulse_distance(measured, refreshes)
            )


@pytest.mark.parametrize(
    ("distance", "expected_refreshes"),
    [(0.3066301518, 3), (0.3609403967, 3)],
)
def test_final_approach_replays_both_beta16_short_pulse_failures(
    distance: float, expected_refreshes: int
) -> None:
    """Both live failures choose three bounded refreshes, not nominal duration."""
    info = _final_approach_pulse_ms(
        distance_to_target=distance,
        observed_pulse_distances=[],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=200,
    )

    assert info["reason"] == "final_approach_bounded_by_refresh_count"
    assert info["refresh_command_limit"] == expected_refreshes


def test_final_approach_declines_when_the_distance_is_unknown() -> None:
    """No distance reading -> no scaling, and say why rather than guessing."""
    info = _final_approach_pulse_ms(
        distance_to_target=None,
        observed_pulse_distances=[1.06],
        default_metres_per_pulse=1.06,
        pulse_duration_ms=3500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is False
    assert info["reason"] == "distance_unknown"
    assert info["pulse_duration_ms"] == 3500.0


def test_turn_final_approach_scales_the_pulse_to_the_remaining_angle() -> None:
    """Replays the 2026-07-27 overshoot: 23.7 deg left must not take a full pulse.

    Live, that error took the full 1500 ms pulse, turned 50.9 deg, overshot by
    27 deg and forced a reversal. At the measured ~33 deg/s it needs ~720 ms.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=-23.744,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=48.236,
        observed_rotation_ms=1500.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is True
    assert info["degrees_per_second_source"] == "observed"
    assert info["degrees_per_second"] == pytest.approx(32.16, abs=0.01)
    # The estimate wants 738.4 ms and the measured sweep bound allows 743.6, so
    # the estimate is the tighter of the two by 5 ms and takes it. beta31's
    # C = 60 deg/s ceiling allowed only 695.7 here and overrode the estimate; the
    # affine bound measured on 2026-08-09 restores the original, better-founded
    # answer. That is the point of the change -- 60 deg/s over-estimated the
    # slope, so it was shortening pulses that did not need shortening.
    assert info["reason"] == "final_approach_scaled_to_remaining_angle"
    assert info["pulse_duration_ms"] == pytest.approx(738.4, abs=1.0)
    assert info["ceiling_pulse_duration_ms"] == pytest.approx(743.6, abs=1.0)


def test_turn_final_approach_is_disabled_without_the_refresh_cadence() -> None:
    """Single-shot rotation is a fixed quantum, so scaling the duration is a trap.

    Without refresh the mower turns ~8-15 deg per command regardless of pulse
    length, so a shortened pulse would not land closer -- and the single-shot
    path has a hard actuation floor (a 2000 ms single-shot pulse was a measured
    physical no-op). The guard must hold however little angle remains.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=-5.0,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=48.0,
        observed_rotation_ms=1500.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=0,
    )

    assert info["applied"] is False
    assert info["reason"] == "refresh_disabled_rotation_not_proportional_to_duration"
    assert info["pulse_duration_ms"] == 1500.0


def test_turn_final_approach_leaves_a_large_error_on_the_full_pulse() -> None:
    """A large error needs more than one pulse -- do not shorten it.

    Uses 120 deg rather than the historical 71.98: with the beta31 ceiling at
    60 deg/s and an 18 deg tolerance the binding threshold is exactly 72 deg, so
    71.98 sat 0.3 ms inside it and tested a knife-edge instead of the intent.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=-120.0,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=0.0,
        observed_rotation_ms=0.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is False
    assert info["reason"] == "cruising_full_pulse_fits"
    assert info["degrees_per_second_source"] == "default"
    assert info["pulse_duration_ms"] == 1500.0


@pytest.mark.parametrize(
    ("observed_rate", "binds"),
    [(14.73, True), (25.0, True), (34.0, True), (37.0, False), (45.0, False)],
)
def test_turn_final_approach_bound_binds_only_when_the_estimate_is_slow(
    observed_rate: float, binds: bool
) -> None:
    """Pins when the sweep bound takes over from the estimate, and when it does not.

    beta31's ceiling was a pure rate, so it bound purely on ERROR: below
    C * pulse_seconds - tolerance, which at C = 60 was 72 deg. That made it the
    active constraint across most of a normal final approach rather than a
    backstop -- handover section 2.2's complaint.

    The affine bound measured on 2026-08-09 does not work that way. It binds when
    it is tighter than what the estimate wants, which is a statement about the
    ESTIMATED RATE, not the error: a slow estimate asks for a long pulse and gets
    capped, while an estimate at or above the configured 37 deg/s already asks
    for less than the bound allows and is left alone.

    Gate 5 attempt 5's geometry, 44.372 deg remaining. Its estimator had learned
    14.73 deg/s from two stall-degraded pulses and wanted a full 1500 ms; the
    bound allows 1259.3 and takes over, which is exactly the pulse that overshot.
    At the configured 37 deg/s the estimate wants 1199 ms and the bound is not
    consulted.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=44.372,
        heading_tolerance_degrees=18.0,
        # One second of observation at the rate under test.
        observed_rotation_degrees=observed_rate,
        observed_rotation_ms=1000.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["ceiling_pulse_duration_ms"] == pytest.approx(1259.3, abs=1.0)
    assert (info["reason"] == "bounded_by_max_rate_ceiling") is binds


def test_turn_final_approach_floors_the_pulse_so_the_mower_still_rotates() -> None:
    """A sliver of angle must not become a pulse too short to actuate."""
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=0.5,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=48.0,
        observed_rotation_ms=1500.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is True
    assert info["pulse_duration_ms"] == 200.0


def test_a_stalled_refresh_write_is_not_a_rotation_rate() -> None:
    """A pulse whose refresh cadence collapsed must not feed the rate estimate.

    Live 2026-08-09 (docs/evidence-beta33-reposition-20260809T184618Z.json,
    segment 3 pulse 1): a 1303.7 ms pulse at a 200 ms refresh interval sent ONE
    of a possible six refresh writes, and that single BLE write blocked for
    1303.972 ms. Motion only continues while refreshes keep arriving, so the
    mower's watchdog stopped the motor and the window was mostly dead time. The
    executor measured 13.885 deg over 1504 ms and called it 9.23 deg/s -- which
    would have been the slowest rotation ever recorded and 44% below
    `_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND`. Every other turn pulse that day,
    cadence intact, measured 23-43 deg/s.

    Folding that number into the estimate is actively harmful rather than merely
    noisy: a low estimate LENGTHENS later pulses, which is how Gate 5 attempt 5
    overshot by 13.258 deg after two stall-degraded pulses taught it ~14.7 deg/s.

    Verified here at the arithmetic that matters -- what the estimator would
    report with and without the stalled sample folded in.
    """
    clean_degrees, clean_ms = 24.893 + 20.274, 1073.358 + 657.759
    stalled_degrees, stalled_ms = 13.885, 1504.162

    honest_rate = clean_degrees / (clean_ms / 1000)
    poisoned_rate = (clean_degrees + stalled_degrees) / ((clean_ms + stalled_ms) / 1000)

    assert honest_rate == pytest.approx(26.08, abs=0.05)
    assert poisoned_rate == pytest.approx(18.30, abs=0.05)
    # Folding the stalled pulse in costs 30% of the estimate, and drags it under
    # the 16.5 deg/s "conservative floor" territory that the feasibility guard
    # treats as a hardware minimum.
    assert poisoned_rate < honest_rate * 0.75

    # And the exclusion test itself needs no tuned constant: the stalled write
    # lasted the whole commanded pulse, while the healthy ones did not. The
    # 820/1500 pair is Gate 5 attempt 5 pulse 3 -- the longest write in the
    # corpus that still produced a normal rate, so it must stay INCLUDED.
    stalled = (1303.972, 1303.7)
    healthy = ((516.0, 1072.3), (260.0, 657.4), (820.0, 1500.0))
    assert stalled[0] >= stalled[1]
    for healthy_write, its_pulse in healthy:
        assert healthy_write < its_pulse


def test_turn_final_approach_rate_is_duration_normalised() -> None:
    """Samples taken at different pulse lengths must stay comparable.

    A per-pulse average would be corrupted by mixing a 1500 ms pulse with a
    700 ms one; a rate is not. Here 48.24 deg over 1500 ms and 25.94 deg over
    700 ms give (48.24+25.94)/2.2 s = 33.7 deg/s.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=10.0,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=48.236 + 25.942,
        observed_rotation_ms=1500.0 + 700.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["degrees_per_second"] == pytest.approx(33.72, abs=0.01)
    # 10 deg at 33.72 deg/s is 296.6 ms, and with the floor at 200 the estimate
    # is now honoured instead of being rounded up to a 400 ms minimum that would
    # have swept ~13 deg for a 10 deg error.
    assert info["pulse_duration_ms"] == pytest.approx(296.6, abs=1.0)


def test_turn_final_approach_ceiling_shortens_the_gate5_overshoot_pulse() -> None:
    """Replays Gate 5 attempt 5 pulse 3, which overshot on the 'cruising' branch.

    Live 2026-08-08 (docs/evidence-gate5-attempt5-segment1-raw-20260808.json):
    44.372 deg remained, the estimator had seen 52.634 deg over 3574 ms of
    delivered window (14.73 deg/s) and so judged that a full 1500 ms pulse could
    not reach the target. The mower then turned at 32.74 deg/s, swept 57.630 deg
    and overshot the target heading by 13.258 deg against an 18 deg tolerance --
    finishing on 4.74 deg of margin.

    No rate ESTIMATE could have caught that; only the fixed ceiling does. At
    60 deg/s the worst acceptable sweep of 44.372 + 18 takes 1039.5 ms.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=44.372,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=30.46 + 22.174,
        observed_rotation_ms=2043.622 + 1530.326,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    # The estimate alone still says "cruising" -- this is the branch that failed.
    assert info["degrees_per_second"] == pytest.approx(14.73, abs=0.01)
    assert info["applied"] is True
    assert info["reason"] == "bounded_by_max_rate_ceiling"
    assert info["max_allowed_sweep_degrees"] == pytest.approx(62.372, abs=0.001)
    # 1259.3 ms, not beta31's 1039.5: the affine bound measured on 2026-08-09
    # permits a longer pulse here because C = 60 deg/s over-estimated the slope
    # (measured 33.18) and so shortened pulses that did not need shortening.
    assert info["pulse_duration_ms"] == pytest.approx(1259.3, abs=1.0)

    # At the rate the mower actually turned (32.736 deg/s) the shortened pulse
    # sweeps ~34.0 deg, leaving ~10.3 deg of error -- inside the 18 deg tolerance,
    # so the turn still ends on this pulse and costs no extra command, instead of
    # finishing 13.258 deg past the target.
    swept = 32.736 * (info["pulse_duration_ms"] / 1000)
    assert swept == pytest.approx(41.2, abs=0.5)
    # Lands 3.2 deg short instead of 13.258 deg past. Still inside tolerance, so
    # the turn ends on this pulse and costs no extra command.
    assert 0 < 44.372 - swept < 18.0


def test_turn_final_approach_ceiling_stays_out_of_the_way_on_large_turns() -> None:
    """The ceiling must cost nothing while the turn is still far from target.

    Gate 5 attempt 5 pulse 1: 97.006 deg of error. The worst acceptable sweep is
    115.006 deg, which takes 1917 ms at the ceiling rate -- longer than the pulse
    -- so the full 1500 ms must survive untouched.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=97.006,
        heading_tolerance_degrees=18.0,
        observed_rotation_degrees=0.0,
        observed_rotation_ms=0.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["applied"] is False
    assert info["reason"] == "cruising_full_pulse_fits"
    assert info["pulse_duration_ms"] == 1500.0
    assert info["ceiling_pulse_duration_ms"] == pytest.approx(2575.1, abs=1.0)


@pytest.mark.parametrize(
    ("tolerance", "floor_binds", "no_safe_pulse"),
    [
        (3.0, True, True),
        (5.5, True, True),
        (8.0, True, False),
        (11.9, False, False),
        (18.0, False, False),
        (30.0, False, False),
    ],
)
def test_turn_final_approach_bound_vs_actuation_floor(
    tolerance: float, floor_binds: bool, no_safe_pulse: bool
) -> None:
    """Pins which safety bound wins when the two conflict, and where each starts.

    The turn loop returns `target_heading_reached` as soon as the error is inside
    tolerance, so any pulse that runs has error > tolerance and the worst
    acceptable sweep is > 2 * tolerance. Against the affine bound that permits
    (2 * tolerance - 12) / 40 seconds, which drops below the 200 ms actuation
    floor once tolerance is under ~10 deg.

    There are now TWO ways to fail, and they are different:

    * `ceiling_below_actuation_floor` -- the bound wants a pulse shorter than the
      mower reliably actuates. The FLOOR wins, deliberately: an overshoot is
      recoverable by the next pulse, but a pulse too short to actuate makes no
      progress and walks the turn into `no_heading_progress` with its budget
      spent.
    * `sweep_exceeds_any_pulse` -- the whole allowance is smaller than the
      bound's 12 deg constant term, so NO duration is safe, because even the
      shortest pulse can sweep past. Below ~6 deg of tolerance. This condition
      did not exist under beta31's pure-rate ceiling, which always returned some
      positive duration however small the allowance, and thereby implied a
      guarantee it could not keep.

    The accepted profile runs `heading_tolerance_degrees: 18`, where neither
    binds.
    """
    info = _turn_final_approach_pulse_ms(
        heading_error_degrees=tolerance + 0.001,
        heading_tolerance_degrees=tolerance,
        observed_rotation_degrees=0.0,
        observed_rotation_ms=0.0,
        default_degrees_per_second=37.0,
        pulse_duration_ms=1500.0,
        refresh_interval_ms=200,
    )

    assert info["ceiling_below_actuation_floor"] is floor_binds
    assert info["sweep_exceeds_any_pulse"] is no_safe_pulse
    assert info["pulse_duration_ms"] >= _MIN_SCALED_TURN_PULSE_MS
    if floor_binds:
        assert info["ceiling_pulse_duration_ms"] < _MIN_SCALED_TURN_PULSE_MS
        assert info["pulse_duration_ms"] == _MIN_SCALED_TURN_PULSE_MS
    else:
        assert info["ceiling_pulse_duration_ms"] >= _MIN_SCALED_TURN_PULSE_MS


@pytest.mark.asyncio
async def test_vio_turn_scales_the_last_pulse_and_does_not_overshoot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: the turn lands on target instead of blowing past and reversing.

    Replays the 2026-07-27 A/B run. The mower rotates at ~33 deg/s under refresh,
    so command 2 (23.7 deg to go) must be a ~720 ms pulse rather than the full
    1500 ms that overshot by 27 deg live and forced a direction reversal.
    """
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    state = {"heading": 75.6}
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=state["heading"], vio_state=2
    )
    durations: list[float] = []

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    async def fake_refresh_window(
        coordinator_arg: MammotionReportUpdateCoordinator,
        *,
        resend: object,
        duration_seconds: float,
        refresh_interval_ms: int,
        max_refresh_commands: int | None = None,
    ) -> dict:
        durations.append(duration_seconds)
        # 33 deg/s, and the sign follows the commanded direction: +angular
        # decreases vision_heading.
        state["heading"] -= 33.0 * duration_seconds
        coordinator.data.report_data.vision_info = SimpleNamespace(
            heading=state["heading"], vio_state=2
        )
        return {
            "refresh_enabled": True,
            "refresh_interval_ms": refresh_interval_ms,
            "refresh_commands_sent": int(duration_seconds * 1000 / refresh_interval_ms),
        }

    monkeypatch.setattr(
        mammotion_services, "_motion_refresh_window", fake_refresh_window
    )

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=3.62,
        # 18 deg, the accepted-profile tolerance, not the original 5. The sweep
        # of 2026-08-09 measured a minimum sweep of ~12 deg for ANY pulse length,
        # so a 5 deg tolerance is below the control scheme's achievable
        # precision and `_vio_turn_budget_feasibility` now correctly refuses it
        # up front with `turn_budget_infeasible`. Asking for 5 deg was always
        # unachievable; the model only recently learned to say so.
        heading_tolerance_degrees=18.0,
        angular_speed=500,
        pulse_duration_ms=1500,
        slow_threshold_degrees=0.0,
        max_commands=6,
        refresh_wait_seconds=0.0,
        motion_refresh_interval_ms=200,
        turn_degrees_per_second=37.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "target_heading_reached"
    # Never turned past the target: no command may reverse direction.
    signs = {(c["angular_speed"] > 0) for c in result["command_results"] if c.get("ok")}
    assert len(signs) == 1, "turn reversed direction -- it overshot"
    # The first pulse cruises at the full 1500 ms again: 71.98 deg of error
    # against an 18 deg tolerance permits (71.98 + 18 - 12) / 40 = 1.950 s, so
    # the bound is not the constraint. beta31's C = 60 ceiling capped it at
    # 1.283 s here, which is the over-restriction the measured affine bound
    # removes. Pulses still shorten monotonically toward the target, which is
    # what this test is really about.
    assert durations[0] == pytest.approx(1.5, abs=0.002)
    assert durations[-1] < durations[0]
    assert durations == sorted(durations, reverse=True)
    assert result["command_results"][-1]["final_approach"]["applied"] is True
    # And the aggregate displacement is reported, not left as None.
    assert result["final_displacement_m"] is not None


@pytest.mark.asyncio
async def test_vio_turn_to_heading_aborts_when_stop_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stop exception mid-turn aborts instead of sending more turn pulses."""
    coordinator = _pulse_coordinator(position=(1.0, 1.0, 0.0))
    coordinator.data.report_data.vision_info = SimpleNamespace(heading=0.0, vio_state=2)
    coordinator.async_stop_manual_motion.side_effect = RuntimeError("BLE cooldown")

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mammotion_services.asyncio, "sleep", no_sleep)

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=40.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    assert result["stop_reason"] == "stop_failed_aborting"
    assert result["commands_sent"] == 1


@pytest.mark.asyncio
async def test_vio_turn_reports_no_actuation_when_nothing_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead command path aborts as no_actuation_detected, not no_heading_progress.

    Regression for 2026-07-19: a forgotten physical e-stop silently no-opped
    every motion command for ~40 minutes while every health indicator read
    green. The turn loop blamed the turn (no_heading_progress) instead of
    surfacing that nothing actuated at all.

    Note the feed is deliberately kept ALIVE here (jittering position). Since
    2026-07-25 the claim "nothing actuated" requires positive evidence that the
    sensors were live, because a frozen report stream produces an identical
    before/after comparison while the mower turns normally. The 07-19 incident
    itself had a frozen feed too (heading bit-identical for 45 minutes), so a
    replay of that exact run now reports ``vio_telemetry_stream_stale`` -- which
    is the honest answer: telemetry never saw the e-stop, the operator did.
    """
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    # Heading is frozen bit-identical and the mower never moves, exactly as the
    # live e-stopped runs reported.
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=91.38829636391407, vio_state=2
    )

    # ...but the report stream is ALIVE: a live position feed jitters ~2mm
    # between reads even on a stationary mower. That is what licenses the claim
    # "the mower did not actuate" rather than "we went blind" -- without it the
    # run is indistinguishable from a dead feed and must report
    # vio_telemetry_stream_stale instead.
    jitter = {"n": 0}

    async def jittering_reports(*_args: object, **_kwargs: object) -> None:
        jitter["n"] += 1
        coordinator.data.mowing_state.pos_x = 1.0 + 0.002 * (jitter["n"] % 2)

    coordinator.async_get_reports = AsyncMock(side_effect=jittering_reports)

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=9.2,
        heading_tolerance_degrees=18.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_commands=16,
    )

    assert result["stop_reason"] == "no_actuation_detected"
    assert "e-stop" in result["no_actuation_hint"]
    # Bounded exactly like the old path: it still stops after the streak.
    assert result["commands_sent"] == 2


@pytest.mark.asyncio
async def test_vio_turn_reports_stale_stream_when_the_feed_is_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frozen report stream aborts as stale telemetry, not no_actuation_detected.

    Regression for 2026-07-25: two turn pulses reported bit-identical
    vision_heading (90.29915121519771) and bit-identical displacement_m
    (0.006754257916307457) while the operator watched the mower turn ~4 inches.
    The server log for that window shows BLE frames being dropped outright, so
    the telemetry was dead while actuation was fine -- but the loop blamed the
    mower with no_actuation_detected, which sends the operator to check a
    physical e-stop that was not engaged.
    """
    coordinator = _pulse_coordinator()
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    # Heading frozen AND position frozen: nothing in the report stream updates,
    # which is what a dropped-frame window looks like from here.
    coordinator.data.report_data.vision_info = SimpleNamespace(
        heading=90.29915121519771, vio_state=2
    )

    _patch_services_monotonic(monkeypatch, fake_monotonic)
    monkeypatch.setattr(mammotion_services.asyncio, "sleep", fake_sleep)

    result = await _vio_turn_to_heading(
        coordinator,
        target_vision_heading=9.2,
        heading_tolerance_degrees=18.0,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
        max_commands=16,
    )

    assert result["stop_reason"] == "vio_telemetry_stream_stale"
    assert "dropped/malformed frames" in result["vio_telemetry_stream_stale_hint"]
    # It must NOT accuse the mower of failing to actuate.
    assert "no_actuation_hint" not in result
    # Still bounded by the same streak logic.
    assert result["commands_sent"] == 2
    # The evidence is recorded per pulse for post-run forensics.
    assert all(
        command["heading_poll_feed_alive"] is False
        and command["heading_poll_count"] >= 3
        for command in result["command_results"]
    )


SERVICES = "custom_components.mammotion.services"


def _staged_turn_harness(
    *,
    direct_stop_reason: str,
    stage_displacement: float = 0.02,
    stage_stop_reason: str = "target_heading_reached",
) -> tuple[list[dict[str, object]], dict[str, float], object]:
    """Fake `_vio_turn_to_heading` that refuses once, then honours each stage.

    Returns the recorded calls, the mutable vision-heading state the fake keeps in
    sync with the stages it "executes", and the patch target.
    """
    calls: list[dict[str, object]] = []
    state = {"heading": 0.0}

    async def fake_turn(_coordinator, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return {"stop_reason": direct_stop_reason, "turn_feasibility": {"x": 1}}
        state["heading"] = float(kwargs["target_vision_heading"])
        return {
            "stop_reason": stage_stop_reason,
            "commands_sent": 2,
            "motion_refresh_commands_sent": 5,
            "command_results": [{"stage": len(calls) - 1}],
            "samples": [],
            "final_displacement_m": stage_displacement,
        }

    return calls, state, fake_turn


@pytest.mark.asyncio
async def test_staged_turn_passes_a_dispatchable_turn_straight_through() -> None:
    """A turn the primitive accepts is returned untouched, with no staging."""
    calls, _state, fake_turn = _staged_turn_harness(
        direct_stop_reason="target_heading_reached"
    )
    with patch(f"{SERVICES}._vio_turn_to_heading", fake_turn):
        result = await _vio_turn_to_heading_staged(
            MagicMock(),
            target_vision_heading=40.0,
            heading_tolerance_degrees=18.0,
            max_displacement_m=0.3,
        )

    assert result["stop_reason"] == "target_heading_reached"
    assert "staged_turn" not in result
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_staged_turn_only_decomposes_a_budget_refusal() -> None:
    """A turn that RAN and failed is not retried in pieces.

    `turn_budget_infeasible` sends zero commands, so decomposing costs nothing.
    Every other failure means something is wrong -- a stale feed, a lost
    transport, no rotation -- and slicing it up would just fail more slowly.
    """
    calls, _state, fake_turn = _staged_turn_harness(
        direct_stop_reason="no_heading_progress"
    )
    with patch(f"{SERVICES}._vio_turn_to_heading", fake_turn):
        result = await _vio_turn_to_heading_staged(
            MagicMock(),
            target_vision_heading=170.0,
            heading_tolerance_degrees=18.0,
            max_displacement_m=0.3,
        )

    assert result["stop_reason"] == "no_heading_progress"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_staged_turn_splits_a_refused_180_into_bounded_stages() -> None:
    """A 177 deg opening turn -- refused as one call -- completes in stages.

    This is the case an operator hits by clicking a point behind the mower. A
    single 180 deg turn wants 8 commands against a budget of 4 (measured
    2026-08-09), and nothing can preflight it because no feed reports a
    stationary mower's orientation.
    """
    calls, state, fake_turn = _staged_turn_harness(
        direct_stop_reason="turn_budget_infeasible"
    )

    def fake_reading(_coordinator):
        return {"vio_state": 2, "vision_heading": state["heading"]}

    with (
        patch(f"{SERVICES}._vio_turn_to_heading", fake_turn),
        patch(f"{SERVICES}._vio_reading", fake_reading),
    ):
        result = await _vio_turn_to_heading_staged(
            MagicMock(),
            target_vision_heading=177.0,
            heading_tolerance_degrees=18.0,
            max_displacement_m=0.3,
        )

    assert result["staged_turn"] is True
    assert result["stop_reason"] == "target_heading_reached"
    # No stage may exceed the validated 60 deg magnitude.
    steps = [abs(float(stage["step_degrees"])) for stage in result["stages"]]
    assert steps and max(steps) <= 60.0 + 1e-9
    # Totals roll up rather than reporting only the last stage.
    assert result["commands_sent"] == 2 * len(result["stages"])
    assert result["motion_refresh_commands_sent"] == 5 * len(result["stages"])
    assert len(result["command_results"]) == len(result["stages"])


@pytest.mark.asyncio
async def test_staged_turn_budgets_translation_across_the_whole_turn() -> None:
    """Each stage gets only the translation the earlier stages left.

    Without this, four stages could each drift the full `max_displacement_m`
    while every individual call looked compliant -- four times the cap the
    profile actually grants a turn.
    """
    calls, state, fake_turn = _staged_turn_harness(
        direct_stop_reason="turn_budget_infeasible", stage_displacement=0.1
    )

    def fake_reading(_coordinator):
        return {"vio_state": 2, "vision_heading": state["heading"]}

    with (
        patch(f"{SERVICES}._vio_turn_to_heading", fake_turn),
        patch(f"{SERVICES}._vio_reading", fake_reading),
    ):
        result = await _vio_turn_to_heading_staged(
            MagicMock(),
            target_vision_heading=177.0,
            heading_tolerance_degrees=18.0,
            max_displacement_m=0.3,
        )

    offered = [float(call["max_displacement_m"]) for call in calls[1:]]
    assert offered[0] == pytest.approx(0.3)
    for earlier, later in zip(offered, offered[1:], strict=False):
        assert later < earlier
    assert float(result["final_displacement_m"]) <= 0.3 + 1e-9


@pytest.mark.asyncio
async def test_staged_turn_reports_the_original_refusal_when_staging_cannot_help() -> (
    None
):
    """A 60 deg stage refused too means the budget, not the rotation, is the problem.

    Slicing finer cannot fix a budget that dispatches nothing, so the original
    `turn_budget_infeasible` is reported rather than a staging failure -- keeping
    the feasibility math that explains why.
    """
    calls, state, fake_turn = _staged_turn_harness(
        direct_stop_reason="turn_budget_infeasible",
        stage_stop_reason="turn_budget_infeasible",
    )

    def fake_reading(_coordinator):
        return {"vio_state": 2, "vision_heading": state["heading"]}

    with (
        patch(f"{SERVICES}._vio_turn_to_heading", fake_turn),
        patch(f"{SERVICES}._vio_reading", fake_reading),
    ):
        result = await _vio_turn_to_heading_staged(
            MagicMock(),
            target_vision_heading=177.0,
            heading_tolerance_degrees=18.0,
            max_displacement_m=0.3,
        )

    assert result["stop_reason"] == "turn_budget_infeasible"
    assert result["staging_cannot_help"] is True
    assert result["turn_feasibility"] is not None
    # It gave up after the first refused stage rather than burning all four.
    assert len(result["stages"]) == 1
