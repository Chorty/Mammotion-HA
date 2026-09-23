"""The square must turn after EVERY leg, including the last one.

The 2026-09-16 run drove four legs but only three turns (`if index < 3`), so the
mower closed on its start *position* about 100 deg off its start *heading*.
Every body point then displaced by a different amount, the tape-vs-RTK
comparison needed a rigid-body reconstruction, and it inherited the unknown
antenna position — which is what cost that run its headline number. The
predeclaration always said four turns; only the runner disagreed.

These tests drive the real `main()` with the transport and the pulse dispatch
stubbed, so nothing is sent anywhere.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    """Import the runner by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "rtk_square_return", REPO / "scripts" / "rtk_square_return.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[ModuleType, dict[str, list[Any]]]:
    """Return the runner with its transport stubbed, plus a log of what it drove."""
    module = _load()
    calls: dict[str, list[Any]] = {"legs": [], "turns": [], "gate": []}

    monkeypatch.setenv("HA_URL", "http://stub")
    monkeypatch.setenv("HA_TOKEN", "stub")
    monkeypatch.setattr(module, "load_dotenv", lambda _path: None)
    monkeypatch.setattr(
        module,
        "read_state",
        lambda *_a, **_k: {"position": {"x": 0.0, "y": 0.0, "rtk_status_label": "fix"}},
    )
    monkeypatch.setattr(module, "preflight", lambda _state: [])
    monkeypatch.setattr(
        module, "set_gate", lambda *_a, on, **_k: calls["gate"].append(on)
    )

    def fake_leg(
        _url: str, _token: str, label: str, record: dict[str, Any], **_k: Any
    ) -> dict[str, Any]:
        # Each leg runs 90 deg off the previous one, as a driven square would.
        leg = {
            "label": label,
            "bearing_deg": 90.0 * len(record["legs"]),
            "length_m": 2.7,
            "pulses": [{"ms": 1300}, {"ms": 1300}],
        }
        record["legs"].append(leg)
        calls["legs"].append(label)
        return leg

    def fake_turn(
        _url: str,
        _token: str,
        label: str,
        rate: float,
        record: dict[str, Any],
        **_k: Any,
    ) -> dict[str, Any]:
        turn = {"label": label, "commanded_ms": 1000, "rate_used": rate}
        record["turns"].append(turn)
        calls["turns"].append(label)
        return turn

    monkeypatch.setattr(module, "drive_leg", fake_leg)
    monkeypatch.setattr(module, "drive_turn", fake_turn)
    monkeypatch.setattr("sys.argv", ["rtk_square_return.py", "--out", str(tmp_path)])
    return module, calls


def test_a_turn_is_driven_after_every_leg_including_the_last(
    runner: tuple[ModuleType, dict[str, list[Any]]],
) -> None:
    """Four legs, four turns — heading is restored, so one tape measure suffices."""
    module, calls = runner

    assert module.main() == 0
    assert calls["legs"] == ["leg1", "leg2", "leg3", "leg4"]
    assert calls["turns"] == ["turn1", "turn2", "turn3", "turn4"]


def test_the_gate_is_disarmed_even_though_the_run_added_a_turn(
    runner: tuple[ModuleType, dict[str, list[Any]]],
) -> None:
    """Arm once, disarm once — the extra turn must not escape the finally block."""
    module, calls = runner

    module.main()

    assert calls["gate"] == [True, False]


def test_an_aborted_final_turn_still_disarms(
    runner: tuple[ModuleType, dict[str, list[Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal on the new fourth turn is a stop, not a stranded armed gate."""
    module, calls = runner

    def aborting_turn(
        _url: str,
        _token: str,
        label: str,
        _rate: float,
        record: dict[str, Any],
        **_k: Any,
    ) -> dict[str, Any]:
        turn = {"label": label, "commanded_ms": 1000}
        if label == "turn4":
            turn["abort"] = "blockers: ['ble_client_not_connected']"
        record["turns"].append(turn)
        calls["turns"].append(label)
        return turn

    monkeypatch.setattr(module, "drive_turn", aborting_turn)

    module.main()

    assert calls["turns"] == ["turn1", "turn2", "turn3", "turn4"]
    assert calls["gate"] == [True, False]
