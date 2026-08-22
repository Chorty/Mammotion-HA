"""The long continuous window is bounded by DISTANCE, not by a time proxy.

Until 2026-08-22 `duration_ms` was capped at 4000 ms with the comment "the only
thing limiting travel is the window". That is a time proxy for the property
actually worth bounding, and it capped the longest continuous run this project
can perform at ~1.1 m -- while the whole case for continuous motion (4.88x, see
`docs/what-continuous-motion-is-worth-20260822.md`) rests on extrapolating a 4 s
window to a 159 s route.

Raising the cap alone would have traded a real safety bound for a bigger number.
Instead the in-window sampler -- which already runs at 100 ms and is proven on
two physical captures -- became the guard: it measures displacement from where
the window started and aborts as soon as `max_travel_m` is exceeded.

These tests pin the fail-closed part: a window longer than the historic cap is
REFUSED unless a distance guard is supplied, and the guard actually stops the
refresh loop.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
import voluptuous as vol
import yaml

from custom_components.mammotion import services
from custom_components.mammotion.services import (
    _PROBE_DURATION_MS_WITHOUT_TRAVEL_GUARD_MAX,
    _PROBE_MAX_TRAVEL_M_CEILING,
    _PROBE_TRAVEL_GUARD_OVERSHOOT_M,
    RAW_PYMAMMOTION_MOTION_PROBE_SCHEMA,
    _capture_in_window_telemetry,
    _motion_refresh_window,
    _raw_pymammotion_motion_probe,
)

ENTITY = "lawn_mower.test"


class _FakeCoordinator:
    """Only the attributes the sampler reaches for; position comes from a patch."""

    data = None
    device_name = "test"

    @property
    def manager(self) -> Any:
        raise RuntimeError("no manager in tests")


def _validated(**overrides: object) -> dict:
    return RAW_PYMAMMOTION_MOTION_PROBE_SCHEMA({"entity_id": ENTITY, **overrides})


def _dry_run(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> dict[str, Any]:
    """Run the probe's dry-run path without a real coordinator."""
    monkeypatch.setattr(
        services, "_custom_path_telemetry_snapshot", lambda _c: {"position": {}}
    )
    monkeypatch.setattr(services, "_manual_velocity_pulse_gates", lambda *a, **k: [])
    return asyncio.run(_raw_pymammotion_motion_probe(object(), dry_run=True, **kwargs))


def test_travel_guard_bounds_the_long_window() -> None:
    """A guard is expressible, opt-in, and itself bounded."""
    assert _validated()["max_travel_m"] == 0.0
    assert _validated(max_travel_m=1.5)["max_travel_m"] == 1.5
    assert (
        _validated(max_travel_m=_PROBE_MAX_TRAVEL_M_CEILING)["max_travel_m"]
        == _PROBE_MAX_TRAVEL_M_CEILING
    )
    # Neither an unbounded drive nor a guard so tight it is only noise.
    for bad in (0.05, _PROBE_MAX_TRAVEL_M_CEILING + 0.1):
        with pytest.raises(vol.Invalid):
            _validated(max_travel_m=bad)


def test_a_long_window_without_a_guard_is_refused_not_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed: the schema accepts it, the probe blocks it.

    Clamping silently would let a caller believe it asked for 8 s and got 4.
    """
    assert _PROBE_DURATION_MS_WITHOUT_TRAVEL_GUARD_MAX == 4000

    blocked = _dry_run(
        monkeypatch,
        duration_ms=8000,
        motion_refresh_interval_ms=200,
        in_window_sample_interval_ms=100,
    )
    assert "duration_over_4000ms_requires_max_travel_m" in blocked["blockers"]

    # The guard IS the sampler, so a long window needs the sampler running too.
    no_sampler = _dry_run(
        monkeypatch,
        duration_ms=8000,
        motion_refresh_interval_ms=200,
        max_travel_m=1.5,
    )
    assert "duration_over_4000ms_requires_in_window_sampling" in no_sampler["blockers"]

    allowed = _dry_run(
        monkeypatch,
        duration_ms=8000,
        motion_refresh_interval_ms=200,
        in_window_sample_interval_ms=100,
        max_travel_m=1.5,
    )
    assert not [b for b in allowed["blockers"] if b.startswith("duration_over_4000ms")]


def test_a_short_window_is_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every existing caller keeps working with no guard at all."""
    result = _dry_run(monkeypatch, duration_ms=4000, motion_refresh_interval_ms=200)
    assert not [b for b in result["blockers"] if b.startswith("duration_over_4000ms")]
    assert result["travel_guard"]["enabled"] is False


def test_the_response_states_the_real_bound_not_the_requested_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corridor sized to `max_travel_m` alone would be too small.

    The guard reads a ~1 Hz cache and the refresh loop notices within one
    interval, so it trips late. The response says by how much, so a corridor is
    sized against the real bound.
    """
    guard = _dry_run(
        monkeypatch,
        duration_ms=8000,
        motion_refresh_interval_ms=200,
        in_window_sample_interval_ms=100,
        max_travel_m=1.5,
    )["travel_guard"]

    assert guard["enabled"] is True
    assert guard["max_travel_m"] == 1.5
    assert guard["expected_overshoot_m"] == pytest.approx(
        _PROBE_TRAVEL_GUARD_OVERSHOOT_M
    )
    assert guard["corridor_must_cover_m"] == pytest.approx(
        1.5 + _PROBE_TRAVEL_GUARD_OVERSHOOT_M
    )
    assert guard["tripped"] is False


def _moving_snapshot(monkeypatch: pytest.MonkeyPatch, step_m: float) -> None:
    """Patch the snapshot so cached position advances a step per sampler read."""
    state = {"reads": 0}

    def _snapshot(_coordinator: Any) -> dict[str, Any]:
        state["reads"] += 1
        return {
            "position": {
                "source": "test",
                "x": step_m * (state["reads"] - 1),
                "y": 0.0,
                "toward": 0.0,
                "pos_type": 1,
                "zone_hash": "1",
            }
        }

    monkeypatch.setattr(services, "_custom_path_telemetry_snapshot", _snapshot)
    monkeypatch.setattr(services, "_safe_attr_path", lambda *a, **k: None)


def test_the_sampler_trips_the_guard_and_stops_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Displacement past the bound sets the abort and ends the sampler."""
    _moving_snapshot(monkeypatch, 0.20)
    abort = asyncio.Event()

    samples = asyncio.run(
        _capture_in_window_telemetry(
            _FakeCoordinator(),
            sample_interval_ms=100,
            duration_ms=8000,
            window_started=time.monotonic(),
            stop_event=asyncio.Event(),
            command="send_movement",
            command_args={"linear_speed": 400, "angular_speed": 0},
            max_travel_m=1.0,
            travel_abort=abort,
        )
    )

    assert abort.is_set()
    assert samples[-1]["travel_guard_tripped"] is True
    assert samples[-1]["travelled_from_origin_m"] >= 1.0
    # It stopped AT the breach rather than running the whole window out.
    assert len(samples) < 81


def test_the_guard_does_not_fire_when_travel_stays_inside_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A well-behaved run is untouched by the guard."""
    _moving_snapshot(monkeypatch, 0.001)
    abort = asyncio.Event()

    samples = asyncio.run(
        _capture_in_window_telemetry(
            _FakeCoordinator(),
            sample_interval_ms=100,
            duration_ms=500,
            window_started=time.monotonic(),
            stop_event=asyncio.Event(),
            command="send_movement",
            command_args={"linear_speed": 400, "angular_speed": 0},
            max_travel_m=2.0,
            travel_abort=abort,
        )
    )

    assert not abort.is_set()
    assert not any(s.get("travel_guard_tripped") for s in samples)


def test_the_refresh_window_stops_refreshing_once_the_guard_trips() -> None:
    """The abort shortens a drive; the caller's mandatory stop then lands."""
    sent = 0

    async def _resend() -> None:
        nonlocal sent
        sent += 1

    abort = asyncio.Event()
    abort.set()

    report = asyncio.run(
        _motion_refresh_window(
            object(),
            resend=_resend,
            duration_seconds=8.0,
            refresh_interval_ms=200,
            abort_event=abort,
        )
    )

    assert report["aborted_early"] is True
    assert sent == 0


def test_services_yaml_exposes_the_guard() -> None:
    """The UI must be able to set it, or nobody will."""
    path = Path("custom_components/mammotion/services.yaml")
    fields = yaml.safe_load(path.read_text())["raw_pymammotion_motion_probe"]["fields"]
    assert fields["max_travel_m"]["default"] == 0
    assert fields["max_travel_m"]["selector"]["number"]["max"] == 3.0
    assert fields["duration_ms"]["selector"]["number"]["max"] == 12000
