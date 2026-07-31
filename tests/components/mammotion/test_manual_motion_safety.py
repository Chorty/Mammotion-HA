"""Tests for the experimental manual-motion safety boundary."""

from __future__ import annotations

import asyncio
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.mammotion import manual_motion
from custom_components.mammotion.backend_capability import (
    CAPABILITY_BLE_TEARDOWN,
    CAPABILITY_BLUFI_REASSEMBLY,
    REQUIRED_MOTION_CAPABILITIES,
)
from custom_components.mammotion.manual_motion import (
    ManualMotionCancelledError,
    ManualMotionSession,
    assert_session_can_dispatch,
    experimental_motion_enabled,
    experimental_motion_status,
    motion_backend_verified,
)
from custom_components.mammotion.services import _stop_active_manual_motion


def _coordinator(*, enabled: bool = False) -> SimpleNamespace:
    # MappingProxyType, not dict: that is what Home Assistant actually puts on
    # ConfigEntry.options, and a plain dict here hid a bug that made the opt-in
    # impossible to turn on in production while every test passed.
    return SimpleNamespace(
        config_entry=SimpleNamespace(
            options=MappingProxyType({"enable_experimental_motion": enabled})
        ),
        manual_motion_session=None,
        last_manual_motion_session=None,
    )


def _capabilities(**present: bool) -> dict[str, object]:
    """Build a capability report with the named capabilities present."""
    capabilities = {
        name: bool(present.get(name, False)) for name in REQUIRED_MOTION_CAPABILITIES
    }
    missing = [name for name, ok in capabilities.items() if not ok]
    return {
        "probed": True,
        "capabilities": capabilities,
        "missing": missing,
        "verified": not missing,
        "reasons": [f"backend_missing_{name}" for name in missing],
    }


def test_the_opt_in_reads_a_mappingproxy_config_entry() -> None:
    """The opt-in must be readable from the mapping type HA actually supplies.

    ``ConfigEntry.options`` is a ``MappingProxyType``, which is a ``Mapping`` but
    not a ``dict``. An ``isinstance(options, dict)`` check therefore reported the
    opt-in as off for every real config entry: the operator could tick the box in
    the options flow, HA would store it, and motion stayed blocked with no error
    -- the failure was invisible precisely because the gate fails closed.
    """
    proxied_on = SimpleNamespace(
        config_entry=SimpleNamespace(
            options=MappingProxyType({"enable_experimental_motion": True})
        )
    )
    proxied_off = SimpleNamespace(
        config_entry=SimpleNamespace(
            options=MappingProxyType({"enable_experimental_motion": False})
        )
    )

    assert experimental_motion_enabled(proxied_on) is True
    assert experimental_motion_enabled(proxied_off) is False


def test_the_opt_in_stays_off_without_a_readable_config_entry() -> None:
    """Absence of options is never consent -- unreadable means disabled."""
    assert experimental_motion_enabled(SimpleNamespace()) is False
    assert experimental_motion_enabled(SimpleNamespace(config_entry=None)) is False
    assert (
        experimental_motion_enabled(SimpleNamespace(config_entry=SimpleNamespace()))
        is False
    )
    # A non-mapping options attribute must not be coerced into consent.
    assert (
        experimental_motion_enabled(
            SimpleNamespace(config_entry=SimpleNamespace(options="enabled"))
        )
        is False
    )


def test_a_version_number_alone_never_verifies_the_backend() -> None:
    """Any version with unproven capabilities stays locked, however new."""
    unproven = _capabilities()

    assert motion_backend_verified("0.8.12", capabilities=unproven) is False
    assert motion_backend_verified("0.8.13", capabilities=unproven) is False
    assert motion_backend_verified("99.0.0", capabilities=unproven) is False


def test_partial_backend_fixes_do_not_verify() -> None:
    """Both audited fixes are required; a cherry-pick of one is not enough."""
    proven = _capabilities(**dict.fromkeys(REQUIRED_MOTION_CAPABILITIES, True))

    assert (
        motion_backend_verified(
            "0.8.12",
            capabilities=_capabilities(**{CAPABILITY_BLE_TEARDOWN: True}),
        )
        is False
    )
    assert (
        motion_backend_verified(
            "0.8.12",
            capabilities=_capabilities(**{CAPABILITY_BLUFI_REASSEMBLY: True}),
        )
        is False
    )
    assert motion_backend_verified("0.8.12", capabilities=proven) is True


def test_backend_below_the_audited_base_never_verifies() -> None:
    """Proven capabilities cannot rescue a release nobody has read."""
    proven = _capabilities(**dict.fromkeys(REQUIRED_MOTION_CAPABILITIES, True))

    assert motion_backend_verified("0.8.11", capabilities=proven) is False
    assert motion_backend_verified("not-installed", capabilities=proven) is False


def test_unprobed_backend_is_never_verified() -> None:
    """Absence of evidence must read as absence of the fix."""
    never_probed = {
        "probed": False,
        "capabilities": dict.fromkeys(REQUIRED_MOTION_CAPABILITIES, False),
        "missing": list(REQUIRED_MOTION_CAPABILITIES),
        "verified": False,
        "reasons": ["backend_capability_probe_not_run"],
    }

    assert motion_backend_verified("0.8.12", capabilities=never_probed) is False
    assert motion_backend_verified("0.8.99", capabilities=never_probed) is False


def test_status_is_fail_closed_and_reports_actionable_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing opt-in, backend audit, BLE, and runtime safety all block."""
    monkeypatch.setattr(
        manual_motion,
        "installed_pymammotion_version",
        lambda: "0.8.12",
    )
    monkeypatch.setattr(
        manual_motion,
        "backend_capability_report",
        _capabilities,
    )
    status = experimental_motion_status(
        _coordinator(),
        ble_liveness={"live": False, "reason": "command_queue_backlogged"},
        safety={
            "allowed_for_manual_motion": False,
            "blockers": ["blade_reported_on"],
        },
    )

    assert status["real_motion_allowed"] is False
    assert status["blockers"] == [
        "experimental_motion_disabled",
        "pymammotion_backend_unverified",
        f"backend_missing_{CAPABILITY_BLE_TEARDOWN}",
        f"backend_missing_{CAPABILITY_BLUFI_REASSEMBLY}",
        "command_queue_backlogged",
        "blade_reported_on",
    ]
    assert status["real_click_to_go_segment_limit"] == 2
    assert status["backend_capabilities"]["verified"] is False


def test_cancelled_session_blocks_nonzero_but_never_stop() -> None:
    """An abort becomes visible before any later nonzero dispatch."""
    coordinator = _coordinator(enabled=True)
    session = ManualMotionSession(owner="test")
    session.cancelled = True
    coordinator.manual_motion_session = session

    with pytest.raises(ManualMotionCancelledError):
        assert_session_can_dispatch(coordinator, is_stop=False)
    assert_session_can_dispatch(coordinator, is_stop=True)


@pytest.mark.asyncio
async def test_stop_marks_cancelled_before_three_confirmed_writes_and_waits_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public stop contract aborts first, then confirms a bounded sequence."""
    coordinator = _coordinator(enabled=True)
    session = ManualMotionSession(owner="test")
    session.phase = "running"
    coordinator.manual_motion_session = session
    observations: list[tuple[bool, bool]] = []

    async def fake_stop(_coordinator: object, *, emergency: bool = False) -> dict:
        observations.append((session.cancelled, emergency))
        return {"movement_ok": True}

    monkeypatch.setattr(
        "custom_components.mammotion.services._stop_manual_motion_confirmed",
        fake_stop,
    )

    async def finish_owner() -> None:
        await asyncio.sleep(0)
        session.owner_done.set()

    owner = asyncio.create_task(finish_owner())
    result = await _stop_active_manual_motion(coordinator)
    await owner

    assert observations == [(True, True), (True, True), (True, True)]
    assert result["stop_confirmed"] is True
    assert result["all_stop_writes_confirmed"] is True
    assert result["owner_exited"] is True
    assert session.cancel_reason == "operator_stop"
    assert session.stop_result is result


@pytest.mark.asyncio
async def test_stop_reports_partial_failure_without_hiding_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One successful write stops the mower but the degraded sequence is visible."""
    coordinator = _coordinator()
    stop = AsyncMock(
        side_effect=[
            TimeoutError("first"),
            {"movement_ok": True},
            TimeoutError("third"),
        ]
    )
    monkeypatch.setattr(
        "custom_components.mammotion.services._stop_manual_motion_confirmed",
        stop,
    )

    result = await _stop_active_manual_motion(coordinator)

    assert result["stop_confirmed"] is True
    assert result["all_stop_writes_confirmed"] is False
    assert [attempt["ok"] for attempt in result["attempts"]] == [
        False,
        True,
        False,
    ]
