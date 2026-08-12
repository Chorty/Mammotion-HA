"""The validation harness must never lay out a path from a facing it cannot trust.

`scripts/` is not a package, so the module is loaded by path rather than
imported. Nothing here touches Home Assistant or the mower: every test drives
the pure functions with literal telemetry.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "beta32_validation_run.py"
)


def _load_harness() -> Any:
    spec = importlib.util.spec_from_file_location("beta32_validation_run", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


#: Every calibration drive this project has recorded: the `toward` reading before
#: the run, and the map facing the calibration drive then MEASURED. The mirror
#: relation is claimed to reproduce the second from the first, and this is the
#: pin on that claim -- if `TOWARD_MIRROR_DEGREES` is ever "tidied" into an
#: additive offset, or the sign flips, these fail.
RECORDED_CALIBRATION_DRIVES = [
    ("20260810T002506", 176.0868, 274.160),
    ("20260810T185433", 174.0572, 278.811),
    ("20260810T193833", 33.5651, 55.099),
    ("20260810T205514", -173.9049, 266.712),
    ("20260810T205937", -173.9049, 263.856),
    ("20260810T232848", 173.2761, 277.416),
    ("20260811T001250", 122.6853, 326.772),
]

#: Worst residual across those seven is 2.738 deg. 3.0 leaves the measurement
#: room to be re-derived without leaving room for a broken relation: an additive
#: constant, which is the mistake this guards against, misses by ~10 deg.
MAX_MIRROR_RESIDUAL_DEGREES = 3.0


def _gap(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


@pytest.mark.parametrize(("run", "toward", "measured"), RECORDED_CALIBRATION_DRIVES)
def test_mirror_reproduces_every_measured_facing(
    run: str, toward: float, measured: float
) -> None:
    """Seven for seven against hardware, worst case 2.738 deg."""
    mirror = harness.mirror_facing({"toward": toward})
    assert mirror is not None
    assert _gap(mirror, measured) <= MAX_MIRROR_RESIDUAL_DEGREES, run


def test_mirror_is_not_an_additive_offset() -> None:
    """Two `toward` values either side of the mirror axis map to opposite sides.

    The legacy path added a constant to `toward` and could never work, because
    `toward` runs clockwise and map headings run counter-clockwise. A mirror
    reverses that sense; an offset does not. Asserting the reversal directly
    means no amount of re-tuning a constant can make this pass.
    """
    lower = harness.mirror_facing({"toward": 60.0})
    higher = harness.mirror_facing({"toward": 80.0})
    assert lower is not None and higher is not None
    assert higher < lower


@pytest.mark.parametrize("toward", [None, "unknown"])
def test_mirror_is_none_when_toward_is_unreadable(toward: Any) -> None:
    """A missing or non-numeric `toward` yields no estimate, never a zero."""
    assert harness.mirror_facing({"toward": toward}) is None


def test_agreeing_estimates_yield_the_live_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-08-10 night: legs read 325.63/328.34, the mirror read 327.44."""
    monkeypatch.setattr(harness, "last_travel_heading", lambda: 328.34)
    facing, source = harness.resolve_start_facing({"toward": 122.6853}, None)
    assert _gap(facing, 327.44) < 0.01
    assert source == "mirror_corroborated_by_last_leg"


def test_the_2026_08_10_backwards_path_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression this whole guard exists for.

    The operator repositioned from the app; `last_travel_heading()` still
    reported ~88 deg from the leg we had driven, while the mower was really
    facing ~266.7 deg. The path was laid out backwards and the run died
    pre-dispatch on a ~177 deg opening turn. Twice.
    """
    monkeypatch.setattr(harness, "last_travel_heading", lambda: 88.0)
    with pytest.raises(SystemExit) as refusal:
        harness.resolve_start_facing({"toward": -173.9049}, None)
    message = str(refusal.value)
    assert "REFUSING TO BUILD A PATH" in message
    # It must hand the operator the number to act on, not just complain.
    assert "--heading 264.03" in message


def test_disagreement_just_inside_the_limit_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The limit is a cliff, so pin both sides of it."""
    mirror = harness.mirror_facing({"toward": 122.6853})
    assert mirror is not None
    inside = mirror - (harness.FACING_DISAGREEMENT_LIMIT_DEGREES - 0.5)
    monkeypatch.setattr(harness, "last_travel_heading", lambda: inside)
    facing, source = harness.resolve_start_facing({"toward": 122.6853}, None)
    assert source == "mirror_corroborated_by_last_leg"
    assert _gap(facing, mirror) < 0.01

    outside = mirror - (harness.FACING_DISAGREEMENT_LIMIT_DEGREES + 0.5)
    monkeypatch.setattr(harness, "last_travel_heading", lambda: outside)
    with pytest.raises(SystemExit):
        harness.resolve_start_facing({"toward": 122.6853}, None)


def test_operator_override_wins_over_both_estimates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--heading` is the escape hatch, so it must never itself be refused."""
    monkeypatch.setattr(harness, "last_travel_heading", lambda: 88.0)
    facing, source = harness.resolve_start_facing({"toward": -173.9049}, 266.712)
    assert facing == pytest.approx(266.712)
    assert source == "operator_override"


def test_override_is_normalised() -> None:
    """A negative override is wrapped, so -30 and 330 lay out the same path."""
    facing, _ = harness.resolve_start_facing({"toward": None}, -30.0)
    assert facing == pytest.approx(330.0)


def test_mirror_alone_is_used_but_flagged_uncorroborated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No evidence file to check against is a caveat, not a blocker."""
    monkeypatch.setattr(harness, "last_travel_heading", lambda: None)
    facing, source = harness.resolve_start_facing({"toward": 122.6853}, None)
    assert _gap(facing, 327.44) < 0.01
    assert source == "mirror_uncorroborated"


def test_driven_leg_alone_is_used_when_toward_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing the live check falls back rather than refusing outright."""
    monkeypatch.setattr(harness, "last_travel_heading", lambda: 214.0)
    facing, source = harness.resolve_start_facing({"toward": None}, None)
    assert facing == pytest.approx(214.0)
    assert source == "last_leg_uncorroborated"


@pytest.mark.parametrize(
    ("blockers", "expected"),
    [
        (None, []),
        ([], []),
        (["experimental_motion_disabled"], []),
        (["ble_client_not_connected"], ["ble_client_not_connected"]),
        (
            ["experimental_motion_disabled", "position_not_valid_for_motion"],
            ["position_not_valid_for_motion"],
        ),
    ],
)
def test_only_the_disarmed_gate_is_an_acceptable_blocker(
    blockers: list[str] | None, expected: list[str]
) -> None:
    """`experimental_motion_disabled` is the resting posture; nothing else is.

    On 2026-08-11 every entity-derived check passed -- `ble_link_live` still
    read `on` -- while the gate already knew the BLE client was gone. Asking the
    backend directly is the check that would have caught it.
    """
    assert harness.blocking_reasons({"blockers": blockers}) == expected


def _arm_abort_harness(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Drive main() to the arm step with a gate that refuses to open."""
    monkeypatch.setattr(harness, "load_dotenv", lambda: None)
    monkeypatch.setenv("HA_URL", "http://localhost:8123")
    monkeypatch.setenv("HA_TOKEN", "token")
    monkeypatch.setattr(
        harness,
        "preflight",
        lambda: {"runtime": {}, "failed": [], "position": {"x": 0.0, "y": 0.0}},
    )
    monkeypatch.setattr(
        harness,
        "_load_area_polygons",
        lambda: {"Test": [(-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0)]},
    )
    monkeypatch.setattr(harness, "last_travel_heading", lambda: 0.0)

    def fake_call(service: str, payload: dict[str, Any], timeout: int = 300) -> Any:
        calls.append(service)
        if service == "export_runtime_state":
            # The gate refuses: enabled, but not allowed, because BLE dropped.
            return {
                "experimental_motion": {
                    "enabled": True,
                    "real_motion_allowed": False,
                    "blockers": ["ble_client_not_connected"],
                }
            }
        return {"valid": True, "errors": [], "stop_reason": "dry_run"}

    def fake_subprocess(cmd: list[str], **kwargs: Any) -> Any:
        calls.append(f"gate:{cmd[-2]}")
        return None

    monkeypatch.setattr(harness, "_call", fake_call)
    monkeypatch.setattr(harness.subprocess, "run", fake_subprocess)


def test_a_gate_that_opens_but_refuses_is_still_disarmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE safety property: touching the gate obliges a disarm, always.

    This failed for real on 2026-08-11. BLE dropped between the preflight and
    the arm, so `real_motion_allowed` came back false, main() returned early
    reporting it had aborted "without sending anything" -- and left
    `enabled: true` behind, one BLE reconnect away from an unattended open gate.
    The `armed` flag had been set AFTER the readback, so the `finally` block
    declined to clean up something it had in fact already done.
    """
    calls: list[str] = []
    _arm_abort_harness(monkeypatch, calls)
    monkeypatch.setattr(
        sys, "argv", ["beta32_validation_run.py", "--arm", "--heading", "0"]
    )

    assert harness.main() == 1
    assert "gate:on" in calls, "test set-up wrong: the gate was never armed"
    assert "gate:off" in calls, "GATE LEFT OPEN after an aborted arm"
    assert calls.index("gate:off") > calls.index("gate:on")


def test_the_preview_path_never_touches_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counterpart: without --arm nothing may reach the gate at all."""
    calls: list[str] = []
    _arm_abort_harness(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", ["beta32_validation_run.py", "--heading", "0"])

    assert harness.main() == 0
    assert not [c for c in calls if c.startswith("gate:")]


def test_no_facing_at_all_refuses_rather_than_defaulting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusing beats defaulting to zero.

    Defaulting to 0 deg is what the heading sweep used to do, and on 2026-08-09
    it demanded a 135.017 deg opening turn against a 4-command budget.
    """
    monkeypatch.setattr(harness, "last_travel_heading", lambda: None)
    with pytest.raises(SystemExit) as refusal:
        harness.resolve_start_facing({"toward": None}, None)
    assert "REFUSING TO BUILD A PATH" in str(refusal.value)
