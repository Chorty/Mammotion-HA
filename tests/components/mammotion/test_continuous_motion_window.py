"""Offline tests for the Phase 2 executor, `continuous_motion_window`.

Design decisions this executor implements (operator-approved, 2026-08-23,
`docs/phase2-continuous-motion-design-20260823.md`): straight-line segments
only; extends the bounded-window pattern rather than a persistent velocity
loop; corrects on measured heading every ~1 Hz arrival, never an integrated
yaw-rate model; stops safely on a detected BLE stall. The stall mechanism and
the corridor-breach override are the two gaps found and closed in
`docs/phase2-gap-reconciliation-20260823.md`.

Every real-motion path is exercised through `dry_run=True` here -- no
coordinator I/O, no BLE, no mower command. `would_send` must stay `False` in
every single test in this file.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import voluptuous as vol
import yaml

from custom_components.mammotion import services
from custom_components.mammotion.services import (
    CONTINUOUS_MOTION_WINDOW_SCHEMA,
    HEADING_ACQUISITION_WINDOW_SCHEMA,
    ContinuousControllerConfig,
    ContinuousPoint,
    ContinuousRoute,
    _continuous_course_heading,
    _continuous_decision_loop,
    _continuous_motion_gates,
    _continuous_motion_window,
    _continuous_refresh_window,
    _heading_acquisition_window,
    _wait_for_fresh_continuous_origin,
    _wait_for_post_stop_position,
)

ENTITY = "lawn_mower.test"

STRAIGHT_ROUTE = {"x": 0.0, "y": 0.0}
STRAIGHT_TARGET = {"x": 3.0, "y": 0.0}
STRAIGHT_CORRIDOR = [
    {"x": -0.3, "y": -0.3},
    {"x": 3.3, "y": -0.3},
    {"x": 3.3, "y": 0.3},
    {"x": -0.3, "y": 0.3},
]
ACQUISITION_CORRIDOR = [
    {"x": -1.2, "y": -1.2},
    {"x": 4.2, "y": -1.2},
    {"x": 4.2, "y": 1.2},
    {"x": -1.2, "y": 1.2},
]


def _validated(**overrides: object) -> dict:
    return CONTINUOUS_MOTION_WINDOW_SCHEMA(
        {
            "entity_id": ENTITY,
            "route_start": STRAIGHT_ROUTE,
            "route_target": STRAIGHT_TARGET,
            "corridor_polygon": STRAIGHT_CORRIDOR,
            **overrides,
        }
    )


# --- schema ------------------------------------------------------------------


def test_schema_defaults_match_the_v1_design() -> None:
    """v1 is capped at the plan's own numbers unless a caller overrides them."""
    data = _validated()
    assert data["dry_run"] is True
    assert data["linear_speed"] == 400
    assert data["max_abs_angular_speed"] == 180
    assert data["duration_ms"] == 4000
    assert data["motion_refresh_interval_ms"] == 200
    assert data["decision_sample_interval_ms"] == 100
    assert data["max_distance_m"] == 1.50
    assert data["max_cross_track_m"] == 0.30
    assert data["confirm_blades_off"] is False
    assert data["confirm_clear_area"] is False


def test_heading_acquisition_schema_is_fixed_to_the_measured_envelope() -> None:
    """The acquisition action exposes no unmeasured speed/time configuration."""
    data = HEADING_ACQUISITION_WINDOW_SCHEMA(
        {
            "entity_id": ENTITY,
            "route_start": STRAIGHT_ROUTE,
            "route_target": STRAIGHT_TARGET,
            "corridor_polygon": ACQUISITION_CORRIDOR,
        }
    )
    assert data["linear_speed"] == 400
    assert data["duration_ms"] == 2000
    assert data["motion_refresh_interval_ms"] == 200
    assert data["max_distance_m"] == 1.0
    with pytest.raises(vol.Invalid):
        HEADING_ACQUISITION_WINDOW_SCHEMA({**data, "duration_ms": 2001})


@pytest.mark.parametrize(
    ("field", "value"), [("linear_speed", 399), ("max_abs_angular_speed", 179)]
)
def test_v1_schema_rejects_unmeasured_command_envelopes(field: str, value: int) -> None:
    """The public experimental schema accepts only the measured command pair."""
    with pytest.raises(vol.Invalid):
        _validated(**{field: value})


def test_schema_never_allows_the_runtime_hard_abort_above_030_m() -> None:
    """Callers may tighten the hard bound but cannot relax it above 0.30 m."""
    assert _validated(max_cross_track_m=0.20)["max_cross_track_m"] == 0.20
    with pytest.raises(vol.Invalid):
        _validated(max_cross_track_m=0.31)


def test_route_points_are_strict_xy_only() -> None:
    """A point schema with extra keys would break `ContinuousPoint(**point)`."""
    with pytest.raises(vol.Invalid):
        _validated(route_start={"x": 0.0, "y": 0.0, "z": 1.0})


def test_corridor_polygon_requires_at_least_the_declared_points() -> None:
    """An empty or malformed polygon is rejected at the schema, not the gate."""
    with pytest.raises(vol.Invalid):
        _validated(corridor_polygon=[{"x": 0.0}])


def test_duration_is_bounded_to_the_longest_window_ever_driven() -> None:
    """1000-8000 ms; 8000 is the longest continuous window run on hardware."""
    assert _validated(duration_ms=8000)["duration_ms"] == 8000
    for bad in (999, 8001):
        with pytest.raises(vol.Invalid):
            _validated(duration_ms=bad)


def test_services_yaml_and_translations_agree_with_the_schema() -> None:
    """A field present in one and not the others has bitten this project before."""
    fields_yaml = yaml.safe_load(
        Path("custom_components/mammotion/services.yaml").read_text()
    )["continuous_motion_window"]["fields"]
    assert set(fields_yaml) == {
        "entity_id",
        "route_start",
        "route_target",
        "corridor_polygon",
        "linear_speed",
        "max_abs_angular_speed",
        "duration_ms",
        "motion_refresh_interval_ms",
        "decision_sample_interval_ms",
        "max_distance_m",
        "max_cross_track_m",
        "prefer_ble",
        "dry_run",
        "confirm_blades_off",
        "confirm_clear_area",
    }
    assert fields_yaml["linear_speed"]["selector"]["number"] == {
        "min": 400,
        "max": 400,
        "step": 1,
    }
    assert fields_yaml["max_abs_angular_speed"]["selector"]["number"] == {
        "min": 180,
        "max": 180,
        "step": 1,
    }
    assert fields_yaml["max_cross_track_m"]["selector"]["number"]["max"] == 0.30


# --- heading convention -------------------------------------------------------


def test_course_heading_uses_the_compass_mirror_not_an_additive_offset() -> None:
    """Match `map_bearing = 90.13 - toward`, not an additive offset.

    The same convention every Phase 1 capture used this week -- not
    `toward + 102.4`, the legacy conversion this project has documented as
    wrong by construction.
    """
    assert _continuous_course_heading(0.0) == pytest.approx(90.13)
    assert _continuous_course_heading(90.13) == pytest.approx(0.0)


# --- dry-run gates -------------------------------------------------------------


class _FakeCoordinator:
    """Only the attributes the executor reaches for; overridden per test."""

    data = None
    device_name = "test"

    class _ReportLeaseContext:
        def __init__(self, handle: SimpleNamespace, owner: str) -> None:
            self._handle = handle
            self._lease = SimpleNamespace(
                owner=owner,
                lease_id=1,
                acquired_at_monotonic=time.monotonic(),
                background_stop_enqueued=True,
                background_stop_enqueued_at_monotonic=time.monotonic(),
            )

        async def __aenter__(self) -> SimpleNamespace:
            self._handle._active_lease = self._lease  # noqa: SLF001
            return self._lease

        async def __aexit__(self, *_args: object) -> None:
            self._handle._active_lease = None  # noqa: SLF001

    class _Manager:
        _handle = SimpleNamespace(
            latest_position_sample=None,
            position_epoch=1,
            position_sequence=0,
            last_report_at=0.0,
            report_subscription_generation=0,
            _active_lease=None,
        )

        @classmethod
        def mower(cls, _name: str) -> object:
            return cls._handle

    manager = _Manager()

    def __init__(self, position_stream: Any | None = None) -> None:
        self._position_stream = position_stream
        self.report_stop_calls = 0
        handle = self.manager.mower(self.device_name)

        def exclusive_report_subscription(owner: str) -> object:
            return self._ReportLeaseContext(handle, owner)

        def lease_is_current(lease: object) -> bool:
            return handle._active_lease is lease  # noqa: SLF001

        def begin_generation(lease: object) -> SimpleNamespace:
            assert lease_is_current(lease)
            handle.report_subscription_generation += 1
            return SimpleNamespace(
                owner=lease.owner,
                lease_id=lease.lease_id,
                generation=handle.report_subscription_generation,
                requested_at_monotonic=time.monotonic(),
                baseline_position_sequence=0,
                baseline_position_epoch=1,
                baseline_last_report_at=handle.last_report_at,
            )

        handle.exclusive_report_subscription = exclusive_report_subscription
        handle.report_subscription_lease_is_current = lease_is_current
        handle.begin_report_subscription_generation = begin_generation

    def open_position_sample_stream(self, *, maxsize: int = 1) -> Any | None:
        del maxsize
        return self._position_stream

    async def async_stop_manual_motion(self, **_kwargs: Any) -> None:
        """Satisfy the `stop_primitive_available` gate; never actually called."""

    async def async_stop_continuous_reports(self) -> None:
        """Record lease-owned report teardown without contacting a device."""
        self.report_stop_calls += 1


def _snapshot(**overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {
        "work_mode_label": "MODE_READY",
        "charge_state_label": "not_charging",
        "position": {
            "source": "test",
            "x": 0.0,
            "y": 0.0,
            "toward": 0.0,
            "pos_type_label": "AREA_INSIDE",
            "zone_hash": "1",
        },
        "blade": {"reported_state": 0, "current_cutter_rpm": 0},
    }
    base.update(overrides)
    return base


def _dry_run(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> dict[str, Any]:
    monkeypatch.setattr(
        services, "_custom_path_telemetry_snapshot", lambda _c: _snapshot()
    )
    return asyncio.run(
        _continuous_motion_window(
            _FakeCoordinator(),
            route_start=STRAIGHT_ROUTE,
            route_target=STRAIGHT_TARGET,
            corridor_polygon=ACQUISITION_CORRIDOR,
            dry_run=True,
            **kwargs,
        )
    )


def test_dry_run_sends_nothing_and_reports_the_full_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The universal contract every real-motion probe in this project keeps."""
    result = _dry_run(monkeypatch)

    assert result["would_send"] is False
    assert result["command_result"]["attempted"] is False
    assert result["reason"] == "dry_run"
    assert result["decisions"] == []
    assert result["heading_state"] == {
        "phase": "pending_position_chord",
        "source": None,
        "minimum_chord_m": 0.15,
        "maximum_age_s": 2.0,
    }
    assert result["acquisition"]["required_radius_m"] == pytest.approx(1.06)
    assert result["acquisition"]["boundary_clearance_m"] == pytest.approx(1.2)
    assert result["remaining_budgets"] == {
        "acquisition_s": 2.0,
        "window_s": 4.0,
        "distance_m": 1.5,
    }
    # Pulse gates plus the continuous route/acquisition gates.
    assert len(result["safety_gates"]) >= 11 + 4


def test_real_continuous_steering_is_blocked_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real steering remains unreachable while acquisition is validated."""
    sent = False

    async def _unexpected_send(*_args: Any, **_kwargs: Any) -> None:
        """Record an unsafe dispatch attempt."""
        nonlocal sent
        sent = True

    monkeypatch.setattr(
        services, "_custom_path_telemetry_snapshot", lambda _c: _snapshot()
    )
    monkeypatch.setattr(
        services,
        "_continuous_motion_gates",
        lambda *_a, **_k: [
            {
                "name": "blind_heading_acquisition_contained",
                "passed": True,
                "diagnostics": {"required_radius_m": 1.06},
            }
        ],
    )
    monkeypatch.setattr(services, "_send_manager_command_with_args", _unexpected_send)
    result = asyncio.run(
        _continuous_motion_window(
            _FakeCoordinator(),
            route_start=STRAIGHT_ROUTE,
            route_target=STRAIGHT_TARGET,
            corridor_polygon=ACQUISITION_CORRIDOR,
            dry_run=False,
            confirm_blades_off=True,
            confirm_clear_area=True,
        )
    )
    assert result["reason"] == "steering_not_motion_validated"
    assert result["would_send"] is False
    assert sent is False


def test_acquisition_dispatches_no_angular_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A qualifying chord stops acquisition without opening the steering path."""
    stream = _position_stream([(0.20, 0.0, 90.0)])
    queued = stream.queue.get_nowait()
    queued.sequence = 2
    stream.queue.put_nowait(queued)
    coordinator = _FakeCoordinator(stream)
    sent: list[dict[str, int]] = []
    post_stop_timeouts: list[float] = []

    async def _fresh_origin(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "reason": None,
            "elapsed_s": 0.01,
            "sample": {
                "sequence": 1,
                "epoch": 1,
                "source": "test",
                "transport": "ble",
                "position": {
                    "x": 0.0,
                    "y": 0.0,
                    "toward": 270.0,
                    "pos_type": 1,
                    "zone_hash": 1,
                    "rtk_status": 4,
                },
            },
        }

    async def _send(
        *_args: Any, command_kwargs: dict[str, int], **_kwargs: Any
    ) -> None:
        sent.append(dict(command_kwargs))

    async def _settled(_coordinator: Any) -> dict[str, Any]:
        return {"settled": True}

    async def _stop(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"attempted": True, "ok": True}

    async def _no_post_stop(*_args: Any, **kwargs: Any) -> tuple[None, str]:
        post_stop_timeouts.append(float(kwargs["timeout_s"]))
        return None, "post_stop_position_timeout"

    monkeypatch.setattr(
        services, "_custom_path_telemetry_snapshot", lambda _c: _snapshot()
    )
    monkeypatch.setattr(
        services,
        "_continuous_motion_gates",
        lambda *_a, **_k: [
            {
                "name": "blind_heading_acquisition_contained",
                "passed": True,
                "diagnostics": {"required_radius_m": 1.06},
            }
        ],
    )
    monkeypatch.setattr(services, "_wait_for_fresh_continuous_origin", _fresh_origin)
    monkeypatch.setattr(services, "_send_manager_command_with_args", _send)
    monkeypatch.setattr(services, "_settle_ble_command_queue", _settled)
    monkeypatch.setattr(services, "_manual_velocity_stop_attempt", _stop)
    monkeypatch.setattr(services, "_wait_for_post_stop_position", _no_post_stop)

    result = asyncio.run(
        _heading_acquisition_window(
            coordinator,
            route_start=STRAIGHT_ROUTE,
            route_target=STRAIGHT_TARGET,
            corridor_polygon=ACQUISITION_CORRIDOR,
            dry_run=False,
            confirm_blades_off=True,
            confirm_clear_area=True,
        )
    )

    assert result["reason"] == "heading_acquired"
    assert sent
    assert all(command["angular_speed"] == 0 for command in sent)
    assert result["heading_state"]["phase"] == "acquired"
    assert result["post_stop_observation_timeout_s"] == pytest.approx(3.5)
    assert post_stop_timeouts == [pytest.approx(3.5)]


def test_dry_run_passes_all_gates_on_a_healthy_frozen_corridor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean snapshot against its own frozen corridor blocks nothing."""
    result = _dry_run(monkeypatch)
    assert result["blockers"] == []


def test_a_degenerate_corridor_is_refused() -> None:
    """Fewer than 3 vertices cannot describe a polygon."""
    gates = _continuous_motion_gates(
        _FakeCoordinator(),
        _snapshot(),
        route_start=STRAIGHT_ROUTE,
        route_target=STRAIGHT_TARGET,
        config=ContinuousControllerConfig(),
        corridor_polygon=[{"x": 0.0, "y": 0.0}],
        dry_run=True,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )
    corridor_gate = next(g for g in gates if g["name"] == "corridor_polygon_valid")
    assert corridor_gate["passed"] is False


def test_a_frozen_start_outside_the_corridor_is_refused() -> None:
    """The frozen start itself must be inside the corridor it is paired with."""
    gates = _continuous_motion_gates(
        _FakeCoordinator(),
        _snapshot(),
        route_start={"x": 99.0, "y": 99.0},
        route_target=STRAIGHT_TARGET,
        config=ContinuousControllerConfig(),
        corridor_polygon=STRAIGHT_CORRIDOR,
        dry_run=True,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )
    gate = next(g for g in gates if g["name"] == "frozen_start_inside_corridor")
    assert gate["passed"] is False


def test_start_drift_beyond_the_bound_is_refused_for_a_real_run() -> None:
    """The start is never re-derived from live position -- it is refused."""
    telemetry = _snapshot(
        position={
            "source": "test",
            "x": 1.0,  # 1.0 m from the frozen (0, 0) start, over the 0.30 m bound
            "y": 0.0,
            "toward": 0.0,
            "pos_type_label": "AREA_INSIDE",
            "zone_hash": "1",
        }
    )
    gates = _continuous_motion_gates(
        _FakeCoordinator(),
        telemetry,
        route_start=STRAIGHT_ROUTE,
        route_target=STRAIGHT_TARGET,
        config=ContinuousControllerConfig(),
        corridor_polygon=STRAIGHT_CORRIDOR,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )
    gate = next(g for g in gates if g["name"] == "start_drift_within_bound")
    assert gate["passed"] is False
    assert gate["diagnostics"]["drift_m"] == pytest.approx(1.0)


def test_start_drift_gate_is_dry_run_exempt() -> None:
    """A dry run reports the plan even from a position far from the frozen start."""
    gates = _continuous_motion_gates(
        _FakeCoordinator(),
        _snapshot(position={"source": "test", "x": 50.0, "y": 50.0}),
        route_start=STRAIGHT_ROUTE,
        route_target=STRAIGHT_TARGET,
        config=ContinuousControllerConfig(),
        corridor_polygon=STRAIGHT_CORRIDOR,
        dry_run=True,
        confirm_blades_off=False,
        confirm_clear_area=False,
    )
    gate = next(g for g in gates if g["name"] == "start_drift_within_bound")
    assert gate["passed"] is True


def test_narrow_corridor_refuses_blind_heading_acquisition() -> None:
    """The frozen 0.30 m corridor cannot contain the required 1.06 m disk."""
    gates = _continuous_motion_gates(
        _FakeCoordinator(),
        _snapshot(),
        route_start=STRAIGHT_ROUTE,
        route_target=STRAIGHT_TARGET,
        config=ContinuousControllerConfig(),
        corridor_polygon=STRAIGHT_CORRIDOR,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )
    gate = next(g for g in gates if g["name"] == "blind_heading_acquisition_contained")

    assert gate["passed"] is False
    assert gate["diagnostics"]["required_radius_m"] == pytest.approx(1.06)
    assert gate["diagnostics"]["boundary_clearance_m"] == pytest.approx(0.30)
    assert gate["diagnostics"]["feasible"] is False


@pytest.mark.parametrize("toward", [None, 0.0, 90.13, 270.0])
def test_toward_never_controls_blind_acquisition(toward: float | None) -> None:
    """A wide corridor is admitted regardless of missing or misleading toward."""
    gates = _continuous_motion_gates(
        _FakeCoordinator(),
        _snapshot(
            position={
                "source": "test",
                "x": 0.0,
                "y": 0.0,
                "toward": toward,
                "pos_type_label": "AREA_INSIDE",
                "zone_hash": "1",
            }
        ),
        route_start=STRAIGHT_ROUTE,
        route_target=STRAIGHT_TARGET,
        config=ContinuousControllerConfig(),
        corridor_polygon=ACQUISITION_CORRIDOR,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )
    gate = next(g for g in gates if g["name"] == "blind_heading_acquisition_contained")

    assert gate["passed"] is True
    assert gate["diagnostics"]["boundary_clearance_m"] == pytest.approx(1.2)


def test_blind_acquisition_gate_is_not_dry_run_exempt() -> None:
    """Dry-run diagnostics show the exact blocker a real run would encounter."""
    gates = _continuous_motion_gates(
        _FakeCoordinator(),
        _snapshot(),
        route_start=STRAIGHT_ROUTE,
        route_target=STRAIGHT_TARGET,
        config=ContinuousControllerConfig(),
        corridor_polygon=STRAIGHT_CORRIDOR,
        dry_run=True,
        confirm_blades_off=False,
        confirm_clear_area=False,
    )
    gate = next(g for g in gates if g["name"] == "blind_heading_acquisition_contained")

    assert gate["passed"] is False
    assert gate["diagnostics"]["required_radius_m"] == pytest.approx(1.06)


@pytest.mark.parametrize(
    ("overrides", "blocked_gate"),
    [
        ({"work_mode_label": "MODE_MOWING"}, "mower_ready"),
        ({"charge_state_label": "charging"}, "not_docked_or_charging"),
        (
            {"blade": {"reported_state": 1, "current_cutter_rpm": 1200}},
            "mower_reports_blades_off",
        ),
    ],
)
def test_real_run_fails_closed_on_every_pulse_gate(
    overrides: dict[str, Any], blocked_gate: str
) -> None:
    """The executor inherits every existing pulse gate; spot-check a few."""
    telemetry = _snapshot(**overrides)
    gates = _continuous_motion_gates(
        _FakeCoordinator(),
        telemetry,
        route_start=STRAIGHT_ROUTE,
        route_target=STRAIGHT_TARGET,
        config=ContinuousControllerConfig(),
        corridor_polygon=STRAIGHT_CORRIDOR,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )
    gate = next(g for g in gates if g["name"] == blocked_gate)
    assert gate["passed"] is False


def test_real_run_reuses_proven_latched_rpm_verdict() -> None:
    """The inner pulse gate must agree with the common authorization boundary."""
    coordinator = _FakeCoordinator()
    setattr(
        coordinator,
        "_mammotion_blade_rpm_history",
        [
            {
                "blade": {
                    "reported_state": 0,
                    "current_cutter_mode": 0,
                    "current_cutter_rpm": 3004,
                },
                "position": {"x": 4.8524, "y": -1.5203},
            },
            {
                "blade": {
                    "reported_state": 0,
                    "current_cutter_mode": 0,
                    "current_cutter_rpm": 3004,
                },
                "position": {"x": 4.8527, "y": -1.5233},
            },
            {
                "blade": {
                    "reported_state": 0,
                    "current_cutter_mode": 0,
                    "current_cutter_rpm": 3004,
                },
                "position": {"x": 4.8525, "y": -1.5237},
            },
        ],
    )
    telemetry = _snapshot(
        blade={
            "reported_state": 0,
            "current_cutter_mode": 0,
            "current_cutter_rpm": 3004,
        }
    )

    gates = _continuous_motion_gates(
        coordinator,
        telemetry,
        route_start=STRAIGHT_ROUTE,
        route_target=STRAIGHT_TARGET,
        config=ContinuousControllerConfig(),
        corridor_polygon=ACQUISITION_CORRIDOR,
        dry_run=False,
        confirm_blades_off=True,
        confirm_clear_area=True,
    )

    gate = next(g for g in gates if g["name"] == "mower_reports_blades_off")
    assert gate["passed"] is True
    assert gate["diagnostics"]["blade"]["blade_rpm_stale_register"] is True
    assert gate["diagnostics"]["rpm_stale_verdict"]["feed_proven_live"] is True


# --- the decision loop, isolated from BLE entirely -----------------------------


def _position_stream(
    positions: list[tuple[float, float, float]], *, epoch: int = 1
) -> Any:
    """Return an ordered test stream of immutable-shaped position evidence."""
    queue: asyncio.Queue[Any] = asyncio.Queue()
    received_at = time.monotonic()
    for sequence, (x, y, toward) in enumerate(positions, start=1):
        queue.put_nowait(
            SimpleNamespace(
                sequence=sequence,
                epoch=epoch,
                x=x,
                y=y,
                toward=toward,
                pos_type=1,
                zone_hash=1,
                rtk_status=4,
                source="test",
                transport="ble",
                received_at_monotonic=received_at,
                published_at_monotonic=received_at,
                valid_for_motion=True,
                rejection_reason=None,
            )
        )
    return SimpleNamespace(queue=queue, dropped_samples=0, close=lambda: None)


def test_a_corridor_breach_forces_a_stop_the_pure_controller_did_not_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a breach the pure controller's own decision would miss.

    Gap 4's fix: `route.contained` is never re-checked by the pure module. The
    position here is far enough off the corridor's centreline to still be
    within `max_cross_track_m` (so the pure controller alone would say
    "drive"), but outside the frozen polygon itself.
    """
    stream = _position_stream([(0.5, 5.0, 0.0)])
    route = ContinuousRoute(
        start=ContinuousPoint(0.0, 0.0),
        target=ContinuousPoint(3.0, 0.0),
        contained=True,
    )
    config = ContinuousControllerConfig(max_cross_track_m=30.0, max_distance_m=30.0)
    decisions = asyncio.run(
        _continuous_decision_loop(
            stream,
            route=route,
            corridor_polygon=STRAIGHT_CORRIDOR,
            config=config,
            window_started=asyncio.get_event_loop().time(),
            opening_position=ContinuousPoint(0.0, 0.0),
            opening_sequence=0,
            opening_epoch=1,
            sample_interval_ms=10,
            refresh_state={
                "completions_elapsed_ms": [],
                "last_decision_elapsed_ms": 0.0,
            },
            command_state={"linear_speed": 400, "angular_speed": 0},
            decision_abort=asyncio.Event(),
            stop_event=asyncio.Event(),
        )
    )
    assert decisions
    assert decisions[-1]["decision"]["reason"] == "corridor_breach"
    assert decisions[-1]["decision"]["action"] == "stop"
    assert decisions[-1]["inside_corridor"] is False


@pytest.mark.parametrize(
    ("fault", "expected_reason"),
    [
        ("sequence", "position_sequence_gap"),
        ("epoch", "position_epoch_changed"),
        ("stale", "telemetry_stale"),
    ],
)
def test_position_evidence_faults_stop_the_decision_loop(
    fault: str, expected_reason: str
) -> None:
    """Ordered receipt evidence is mandatory throughout a moving window."""
    stream = _position_stream([(0.20, 0.0, 0.0)])
    sample = stream.queue.get_nowait()
    if fault == "sequence":
        sample.sequence = 2
    elif fault == "epoch":
        sample.epoch = 2
    else:
        sample.received_at_monotonic -= 3.0
    stream.queue.put_nowait(sample)
    decisions = asyncio.run(
        _continuous_decision_loop(
            stream,
            route=ContinuousRoute(
                start=ContinuousPoint(0.0, 0.0),
                target=ContinuousPoint(3.0, 0.0),
                contained=True,
            ),
            corridor_polygon=ACQUISITION_CORRIDOR,
            config=ContinuousControllerConfig(),
            opening_position=ContinuousPoint(0.0, 0.0),
            opening_sequence=0,
            opening_epoch=1,
            window_started=time.monotonic(),
            sample_interval_ms=10,
            refresh_state={
                "completions_elapsed_ms": [],
                "last_decision_elapsed_ms": 0.0,
            },
            command_state={"linear_speed": 400, "angular_speed": 0},
            decision_abort=asyncio.Event(),
            stop_event=asyncio.Event(),
        )
    )
    assert decisions[-1]["decision"]["reason"] == expected_reason
    assert decisions[-1]["decision"]["linear_speed"] == 0
    assert decisions[-1]["decision"]["angular_speed"] == 0


def test_position_queue_drop_stops_the_decision_loop() -> None:
    """Latest-wins replacement is a measurable path gap, never silent loss."""
    base = _position_stream([(0.20, 0.0, 0.0)])

    class _DroppedStream:
        queue = base.queue
        reads = 0

        @property
        def dropped_samples(self) -> int:
            self.reads += 1
            return 0 if self.reads == 1 else 1

    decisions = asyncio.run(
        _continuous_decision_loop(
            _DroppedStream(),
            route=ContinuousRoute(
                start=ContinuousPoint(0.0, 0.0),
                target=ContinuousPoint(3.0, 0.0),
                contained=True,
            ),
            corridor_polygon=ACQUISITION_CORRIDOR,
            config=ContinuousControllerConfig(),
            opening_position=ContinuousPoint(0.0, 0.0),
            opening_sequence=0,
            opening_epoch=1,
            window_started=time.monotonic(),
            sample_interval_ms=10,
            refresh_state={
                "completions_elapsed_ms": [],
                "last_decision_elapsed_ms": 0.0,
            },
            command_state={"linear_speed": 400, "angular_speed": 0},
            decision_abort=asyncio.Event(),
            stop_event=asyncio.Event(),
        )
    )
    assert decisions[-1]["decision"]["reason"] == "position_sequence_gap"


def test_a_stalled_refresh_gap_stops_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gap-2 fix, reproduced end to end through the decision loop.

    `refresh_max_gap_since_last_decision_s` is computed here from a REAL
    completions list with an 810 ms gap in it -- the exact class of stall that
    produced the corpus's largest prediction error
    (docs/phase2-gap-reconciliation-20260823.md), not an injected field.
    """
    stream = _position_stream([(0.2, 0.0, 0.0), (0.4, 0.0, 0.0)])
    route = ContinuousRoute(
        start=ContinuousPoint(0.0, 0.0),
        target=ContinuousPoint(3.0, 0.0),
        contained=True,
    )
    config = ContinuousControllerConfig(max_refresh_gap_s=0.60)
    window_started = asyncio.get_event_loop().time()
    # A refresh completed at 100 ms, then nothing until 910 ms -- an 810 ms gap,
    # matching the real stall this project measured.
    refresh_state = {
        "completions_elapsed_ms": [100.0, 910.0],
        "last_decision_elapsed_ms": 0.0,
    }
    decisions = asyncio.run(
        _continuous_decision_loop(
            stream,
            route=route,
            corridor_polygon=STRAIGHT_CORRIDOR,
            config=config,
            window_started=window_started,
            opening_position=ContinuousPoint(0.0, 0.0),
            opening_sequence=0,
            opening_epoch=1,
            sample_interval_ms=10,
            refresh_state=refresh_state,
            command_state={"linear_speed": 400, "angular_speed": 0},
            decision_abort=asyncio.Event(),
            stop_event=asyncio.Event(),
        )
    )
    assert decisions
    stalled = [
        d for d in decisions if d["decision"]["reason"] == "refresh_cadence_stalled"
    ]
    assert stalled, [d["decision"]["reason"] for d in decisions]


def test_fresh_origin_timeout_refuses_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No movement command is attempted without a post-stream origin."""
    sent = 0
    origin_wait_kwargs: list[dict[str, Any]] = []

    async def _no_origin(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        origin_wait_kwargs.append(dict(_kwargs))
        return {
            "ok": False,
            "reason": "fresh_origin_timeout",
            "sample": None,
            "elapsed_s": 2.0,
        }

    async def _settled(_coordinator: Any) -> dict[str, Any]:
        return {"settled": True}

    async def _unexpected_send(*_args: Any, **_kwargs: Any) -> None:
        nonlocal sent
        sent += 1

    monkeypatch.setattr(
        services, "_custom_path_telemetry_snapshot", lambda _c: _snapshot()
    )
    monkeypatch.setattr(services, "_continuous_motion_gates", lambda *_a, **_k: [])
    monkeypatch.setattr(services, "_wait_for_fresh_continuous_origin", _no_origin)
    monkeypatch.setattr(services, "_settle_ble_command_queue", _settled)
    monkeypatch.setattr(services, "_send_manager_command_with_args", _unexpected_send)

    coordinator = _FakeCoordinator(_position_stream([]))
    result = asyncio.run(
        _heading_acquisition_window(
            coordinator,
            route_start=STRAIGHT_ROUTE,
            route_target=STRAIGHT_TARGET,
            corridor_polygon=ACQUISITION_CORRIDOR,
            dry_run=False,
            confirm_blades_off=True,
            confirm_clear_area=True,
        )
    )

    assert result["reason"] == "fresh_origin_timeout"
    assert result["would_send"] is False
    assert result["command_result"]["attempted"] is False
    assert sent == 0
    assert coordinator.report_stop_calls == 1
    generation = result["report_stream"]["subscription_generation"]
    assert (
        origin_wait_kwargs[0]["request_started_at"]
        >= generation["requested_at_monotonic"]
    )


def test_fresh_origin_wait_times_out_without_a_position_payload() -> None:
    """Aggregate traffic cannot satisfy an empty position-evidence stream."""
    result = asyncio.run(
        _wait_for_fresh_continuous_origin(
            _position_stream([]),
            request_started_at=time.monotonic(),
            baseline_sequence=0,
            baseline_epoch=1,
            timeout_s=0.02,
        )
    )

    assert result["ok"] is False
    assert result["sample"] is None


def test_fresh_origin_timeout_does_not_claim_a_stalled_position_channel() -> None:
    """A routine tail gap must not be named like cell 12's real outage.

    This wait is bounded by `max_heading_acquisition_s` (2.0 s), and 28 of 1434
    healthy stationary position intervals in the beta76 matrix -- 1.95% -- already
    exceed 2.0 s, while generic frames arrive at roughly 2 Hz. So "generic
    advanced but no position arrived" is the NORMAL shape of a timeout here, not
    evidence of a channel fault. It is reported as a separate field, never
    promoted into the reason.
    """
    handle = SimpleNamespace(
        position_epoch=1,
        last_report_at=200.0,
        report_subscription_generation=7,
        report_subscription_lease_is_current=lambda _lease: True,
    )
    lease = object()
    generation = SimpleNamespace(generation=7, baseline_last_report_at=100.0)

    result = asyncio.run(
        _wait_for_fresh_continuous_origin(
            _position_stream([]),
            request_started_at=time.monotonic(),
            baseline_sequence=0,
            baseline_epoch=1,
            timeout_s=0.02,
            handle=handle,
            report_lease=lease,
            report_generation=generation,
        )
    )

    assert result["ok"] is False
    assert result["reason"] == "fresh_origin_timeout"
    assert result["generic_report_advanced"] is True


def test_fresh_origin_accepts_a_fresh_unchanged_coordinate() -> None:
    """Payload identity, not coordinate change, proves a fresh origin."""
    result = asyncio.run(
        _wait_for_fresh_continuous_origin(
            _position_stream([(0.0, 0.0, 180.0)]),
            request_started_at=time.monotonic() - 0.01,
            baseline_sequence=0,
            baseline_epoch=1,
            timeout_s=0.02,
        )
    )

    assert result["ok"] is True
    assert result["sample"]["position"]["x"] == 0.0


def test_post_stop_observation_returns_latest_consecutive_sample() -> None:
    """Stopped observation does not return the first potentially lagged fix."""
    stream = _position_stream([(0.02, 0.0, 0.0), (0.20, 0.0, 0.0)])
    first = stream.queue.get_nowait()
    second = stream.queue.get_nowait()
    first.sequence = 2
    second.sequence = 3
    stream.queue.put_nowait(first)
    stream.queue.put_nowait(second)

    sample, reason = asyncio.run(
        _wait_for_post_stop_position(
            stream,
            after_sequence=1,
            epoch=1,
            timeout_s=0.01,
        )
    )

    assert reason is None
    assert sample is second


def test_a_heading_error_updates_command_state_for_the_next_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decision loop's ONLY job besides safety: steer via `command_state`."""
    # Off the centreline, heading due east: the lookahead aims back toward the
    # line, so SOME nonzero correction must be requested.
    # The mower must first travel past `min_travel_for_heading_trust_m` (0.15 m)
    # for the heading to be actable at all -- added 2026-08-24, so 0.30 -> 0.50
    # is a real 0.20 m of confirmed displacement, not a held-still fix.
    stream = _position_stream([(0.30, 0.10, 0.0), (0.5, 0.10, 0.0)])
    route = ContinuousRoute(
        start=ContinuousPoint(0.0, 0.0),
        target=ContinuousPoint(3.0, 0.0),
        contained=True,
    )
    config = ContinuousControllerConfig(
        max_cross_track_m=30.0, heading_deadband_degrees=0.0
    )
    command_state = {"linear_speed": 400, "angular_speed": 0}
    decisions = asyncio.run(
        _continuous_decision_loop(
            stream,
            route=route,
            corridor_polygon=[
                {"x": -1.0, "y": -1.0},
                {"x": 4.0, "y": -1.0},
                {"x": 4.0, "y": 1.0},
                {"x": -1.0, "y": 1.0},
            ],
            config=config,
            window_started=asyncio.get_event_loop().time(),
            opening_position=ContinuousPoint(0.30, 0.10),
            opening_sequence=0,
            opening_epoch=1,
            sample_interval_ms=10,
            refresh_state={
                "completions_elapsed_ms": [],
                "last_decision_elapsed_ms": 0.0,
            },
            command_state=command_state,
            decision_abort=asyncio.Event(),
            stop_event=asyncio.Event(),
        )
    )
    assert decisions[0]["phase"] == "acquiring_heading"
    assert any(decision["phase"] == "steering" for decision in decisions)
    assert command_state["angular_speed"] != 0


def test_cumulative_arc_length_not_origin_chord_consumes_distance_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every consecutive segment consumes distance, including a bent path."""
    stream = _position_stream(
        [(0.20, 0.0, 0.0), (0.20, 0.20, 0.0), (0.40, 0.20, 0.0)],
    )
    route = ContinuousRoute(
        start=ContinuousPoint(0.0, 0.0),
        target=ContinuousPoint(3.0, 0.0),
        contained=True,
    )
    decisions = asyncio.run(
        _continuous_decision_loop(
            stream,
            route=route,
            corridor_polygon=ACQUISITION_CORRIDOR,
            config=ContinuousControllerConfig(
                max_distance_m=0.50, max_admission_cross_track_m=0.30
            ),
            window_started=asyncio.get_event_loop().time(),
            opening_position=ContinuousPoint(0.0, 0.0),
            opening_sequence=0,
            opening_epoch=1,
            sample_interval_ms=10,
            refresh_state={
                "completions_elapsed_ms": [],
                "last_decision_elapsed_ms": 0.0,
            },
            command_state={"linear_speed": 400, "angular_speed": 0},
            decision_abort=asyncio.Event(),
            stop_event=asyncio.Event(),
        )
    )

    assert decisions[-1]["decision"]["reason"] == "distance_limit_reached"
    assert decisions[-1]["cumulative_distance_m"] == pytest.approx(0.60)
    assert decisions[-1]["cumulative_distance_m"] > (0.40**2 + 0.20**2) ** 0.5


def test_heading_evidence_age_stops_when_position_refresh_stalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated coordinates cannot refresh the last qualifying course chord."""
    stream = _position_stream([(0.20, 0.0, 0.0)])
    route = ContinuousRoute(
        start=ContinuousPoint(0.0, 0.0),
        target=ContinuousPoint(3.0, 0.0),
        contained=True,
    )
    decisions = asyncio.run(
        _continuous_decision_loop(
            stream,
            route=route,
            corridor_polygon=ACQUISITION_CORRIDOR,
            config=ContinuousControllerConfig(max_heading_age_s=0.05),
            window_started=asyncio.get_event_loop().time(),
            opening_position=ContinuousPoint(0.0, 0.0),
            opening_sequence=0,
            opening_epoch=1,
            sample_interval_ms=10,
            refresh_state={
                "completions_elapsed_ms": [],
                "last_decision_elapsed_ms": 0.0,
            },
            command_state={"linear_speed": 400, "angular_speed": 0},
            decision_abort=asyncio.Event(),
            stop_event=asyncio.Event(),
        )
    )

    assert decisions[-1]["decision"]["reason"] == "heading_evidence_stale"
    assert decisions[-1]["decision"]["linear_speed"] == 0
    assert decisions[-1]["decision"]["angular_speed"] == 0


def test_a_standing_mower_never_has_a_correction_written_into_command_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-08-24 failure, reproduced end to end through the real loop.

    A held-still fix with a large heading error is exactly what decisions 0-3 of
    `docs/evidence-phase2-first-physical-run-20260824.json` looked like: the
    window's first fix never moves, so `distance_travelled_m` stays 0.0 while
    `toward` reports a course 90 deg off the route. Before the fix the loop wrote
    a saturated `angular_speed` into `command_state` on the very first arrival,
    and the refresh writer held it for the whole window.
    """
    # `origin` is seeded from the first fresh fix, so a repeated position is
    # zero displacement no matter how many arrivals occur.
    stream = _position_stream([(0.0, 0.0, 0.0)])
    route = ContinuousRoute(
        start=ContinuousPoint(0.0, 0.0),
        target=ContinuousPoint(3.0, 0.0),
        contained=True,
    )
    # toward=0.0 -> course 90.13 deg against a route running due east: a ~90 deg
    # error, which without the gate clamps straight to -max_abs_angular_speed.
    config = ContinuousControllerConfig(
        max_cross_track_m=30.0, max_heading_acquisition_s=0.05
    )
    command_state = {"linear_speed": 400, "angular_speed": 0}
    decisions = asyncio.run(
        _continuous_decision_loop(
            stream,
            route=route,
            corridor_polygon=STRAIGHT_CORRIDOR,
            config=config,
            window_started=asyncio.get_event_loop().time(),
            opening_position=ContinuousPoint(0.0, 0.0),
            opening_sequence=0,
            opening_epoch=1,
            sample_interval_ms=10,
            refresh_state={
                "completions_elapsed_ms": [],
                "last_decision_elapsed_ms": 0.0,
            },
            command_state=command_state,
            decision_abort=asyncio.Event(),
            stop_event=asyncio.Event(),
        )
    )

    driving = [d["decision"] for d in decisions if d["decision"]["action"] == "drive"]
    assert driving
    assert command_state["angular_speed"] == 0
    # No decision of ANY kind ever asks for a correction here.
    assert all(d["decision"]["angular_speed"] == 0 for d in decisions)
    assert all(d["heading_confirmed_by_motion"] is False for d in driving)
    assert all(d["heading_error_degrees"] is None for d in driving)
    assert decisions[-1]["decision"]["reason"] == "heading_acquisition_timeout"


# --- the refresh loop, isolated from the decision loop --------------------------


def test_refresh_loop_resends_whatever_command_state_currently_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one behaviour `_motion_refresh_window` cannot provide."""
    sent: list[tuple[int, int]] = []
    command_state = {"linear_speed": 400, "angular_speed": 0}

    async def _fake_send(
        _coordinator: Any, _command: str, *, prefer_ble: bool, command_kwargs: Any
    ) -> None:
        sent.append((command_kwargs["linear_speed"], command_kwargs["angular_speed"]))
        # Simulate the decision loop changing the command mid-window.
        command_state["angular_speed"] = 45

    monkeypatch.setattr(services, "_send_manager_command_with_args", _fake_send)

    report = asyncio.run(
        _continuous_refresh_window(
            _FakeCoordinator(),
            command_state=command_state,
            prefer_ble=True,
            duration_seconds=0.05,
            refresh_interval_ms=10,
            window_started=asyncio.get_event_loop().time(),
            refresh_state={"completions_elapsed_ms": []},
            abort_event=asyncio.Event(),
        )
    )

    assert report["refresh_commands_sent"] >= 2
    # The angular speed sent on later refreshes reflects the CHANGED command.
    assert any(angular == 45 for _linear, angular in sent[1:])


def test_refresh_loop_stops_refreshing_once_aborted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An abort shortens a window; it never extends one."""
    sent = 0

    async def _fake_send(*_a: Any, **_k: Any) -> None:
        nonlocal sent
        sent += 1

    monkeypatch.setattr(services, "_send_manager_command_with_args", _fake_send)

    abort = asyncio.Event()
    abort.set()
    report = asyncio.run(
        _continuous_refresh_window(
            _FakeCoordinator(),
            command_state={"linear_speed": 400, "angular_speed": 0},
            prefer_ble=True,
            duration_seconds=4.0,
            refresh_interval_ms=200,
            window_started=asyncio.get_event_loop().time(),
            refresh_state={"completions_elapsed_ms": []},
            abort_event=abort,
        )
    )

    assert report["aborted_early"] is True
    assert sent == 0
