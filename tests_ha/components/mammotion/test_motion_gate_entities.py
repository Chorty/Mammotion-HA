"""Tests for the motion-gate diagnostic entities and their shared snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.binary_sensor import BINARY_SENSORS
from custom_components.mammotion.sensor import SENSOR_TYPES
from custom_components.mammotion.services import (
    _GATE_SNAPSHOT_ATTR,
    _GATE_SNAPSHOT_STAMP_ATTR,
    motion_gate_snapshot,
)

GATE_BINARY_KEYS = {
    "real_motion_ready",
    "ble_link_live",
    "motion_backend_verified",
    "blade_safe_for_motion",
    "position_valid_for_motion",
}


def _coordinator() -> SimpleNamespace:
    """Return a bare object the snapshot helper can cache attributes on."""
    return SimpleNamespace()


def _integration_json(name: str) -> dict[str, Any]:
    """Load a JSON file that ships with the integration."""
    root = Path(__file__).resolve().parents[3] / "custom_components" / "mammotion"
    return json.loads((root / name).read_text())


def test_every_gate_entity_is_named_in_every_locale() -> None:
    """A gate entity carries no device_class, so a missing name renders blank."""
    files = ["strings.json"] + [
        f"translations/{loc}.json"
        for loc in (
            "cs",
            "da",
            "de",
            "en",
            "fr",
            "hu",
            "it",
            "nl",
            "pl",
            "ro",
            "sl",
            "sv",
        )
    ]
    for filename in files:
        doc = _integration_json(filename)
        binary = doc.get("entity", {}).get("binary_sensor", {})
        sensors = doc.get("entity", {}).get("sensor", {})
        for key in GATE_BINARY_KEYS:
            assert binary.get(key, {}).get("name"), f"{filename}: binary_sensor.{key}"
        for key in ("zone_hash", "cutter_rpm"):
            assert sensors.get(key, {}).get("name"), f"{filename}: sensor.{key}"


def test_every_gate_entity_has_an_icon() -> None:
    """Diagnostic entities without a device class need an explicit icon."""
    icons = _integration_json("icons.json")["entity"]
    for key in GATE_BINARY_KEYS:
        assert icons["binary_sensor"].get(key, {}).get("default"), key
    for key in ("zone_hash", "cutter_rpm"):
        assert icons["sensor"].get(key, {}).get("default"), key


def test_gate_binary_sensors_are_declared_with_a_gate_key() -> None:
    """The gate entities must read the snapshot, not device data."""
    declared = {d.key: d for d in BINARY_SENSORS if d.key in GATE_BINARY_KEYS}

    assert set(declared) == GATE_BINARY_KEYS
    for key, description in declared.items():
        assert description.gate_key is not None, key
        assert description.is_on_fn is None, key


def test_zone_hash_sensor_reads_the_gate_snapshot() -> None:
    """zone_hash is resolved by services.py, so it must not read raw device data."""
    zone = next(d for d in SENSOR_TYPES if d.key == "zone_hash")

    assert zone.gate_key == "zone_hash"
    assert zone.value_fn is None


def test_snapshot_failure_is_fail_closed_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable gate must mark entities unavailable, never raise."""

    def boom(_coordinator: Any) -> dict[str, Any]:
        raise RuntimeError("telemetry exploded")

    monkeypatch.setattr(mammotion_services, "_custom_path_telemetry_snapshot", boom)
    monkeypatch.setattr(mammotion_services, "_export_active_route", lambda _c: None)

    snapshot = motion_gate_snapshot(_coordinator())

    assert snapshot["available"] is False
    assert snapshot["reason"] == "RuntimeError"
    assert snapshot["real_motion_ready"] is None
    assert snapshot["zone_hash"] is None


def test_snapshot_is_cached_so_six_entities_cost_one_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route projection is expensive; entities must share one computation."""
    calls = 0

    def counting_route(_coordinator: Any) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(mammotion_services, "_export_active_route", counting_route)
    monkeypatch.setattr(
        mammotion_services,
        "_custom_path_telemetry_snapshot",
        lambda _c: {"blade": {}, "position": {}},
    )
    monkeypatch.setattr(
        mammotion_services,
        "_runtime_motion_safety_summary",
        lambda *_a, **_k: {"allowed_for_manual_motion": False, "blockers": []},
    )
    monkeypatch.setattr(
        mammotion_services, "_ble_link_liveness", lambda _c: {"live": False}
    )
    monkeypatch.setattr(
        mammotion_services,
        "experimental_motion_status",
        lambda *_a, **_k: {"real_motion_allowed": False, "blockers": ["x"]},
    )

    coordinator = _coordinator()
    for _ in range(6):
        motion_gate_snapshot(coordinator)

    assert calls == 1

    # Expiring the stamp forces exactly one recomputation.
    setattr(coordinator, _GATE_SNAPSHOT_STAMP_ATTR, -10_000.0)
    motion_gate_snapshot(coordinator)

    assert calls == 2
    assert getattr(coordinator, _GATE_SNAPSHOT_ATTR)["blockers"] == ["x"]


def test_snapshot_survives_an_unreadable_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing route export must not blind the rest of the verdict."""

    def boom(_coordinator: Any) -> dict[str, Any]:
        raise ValueError("no geojson")

    monkeypatch.setattr(mammotion_services, "_export_active_route", boom)
    monkeypatch.setattr(
        mammotion_services,
        "_custom_path_telemetry_snapshot",
        lambda _c: {"blade": {}, "position": {"valid_for_motion": True}},
    )
    monkeypatch.setattr(
        mammotion_services,
        "_runtime_motion_safety_summary",
        lambda *_a, **_k: {"allowed_for_manual_motion": True, "blockers": []},
    )
    monkeypatch.setattr(
        mammotion_services, "_ble_link_liveness", lambda _c: {"live": True}
    )
    monkeypatch.setattr(
        mammotion_services,
        "experimental_motion_status",
        lambda *_a, **_k: {"real_motion_allowed": True, "blockers": []},
    )

    snapshot = motion_gate_snapshot(_coordinator())

    assert snapshot["available"] is True
    assert snapshot["real_motion_ready"] is True
    assert snapshot["position_valid_for_motion"] is True
