"""Pins the nudge buttons as RE-ENABLED and UNGATED, by operator decision.

🚨 This is deliberate, not a regression. The four `emergency_nudge_*` buttons
were hard-disabled -- `_nudge_available` returned a literal `False` and pressing
raised -- with the comment "cannot confirm safety". On 2026-08-20 the operator
re-enabled them, explicitly choosing BOTH the ungated press path and
always-available, after the trade-offs were stated.

The situation that prompted it: the mower came to rest INSIDE a no-go zone, so
`pos_type_label` read `OBS_ON`, so `position_not_valid_for_motion` fired, so
every guarded path refused -- including `move_backward`, verified live returning
`rejected_safety_gate` with `would_send: false`. The gate that exists to stop
motion from an invalid position was the only thing preventing the one movement
that would restore a valid position. An ungated primitive is the escape hatch.

⚠️ WHAT THIS COSTS, so nobody has to rediscover it: a press consults NOTHING.
Not the blades, not BLE liveness, not position validity, not whether an
autonomous mow is running, and there is no confirmation step. One press is one
movement command in every state. Motion is bounded only by the mower's own
H-watchdog -- these primitives send no refresh, so the device self-halts after
roughly one step. That bound belongs to the DEVICE, not to this code.

If a future session is tempted to "fix" this back to `return False`: read the
above first, then ask the operator. Do not revert it as a bug.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.mammotion.button import (
    BUTTON_SENSORS,
    _nudge_available,
    _unguarded_nudge,
)

_NUDGE_KEYS = {
    "emergency_nudge_forward": "async_move_forward",
    "emergency_nudge_left": "async_move_left",
    "emergency_nudge_right": "async_move_right",
    "emergency_nudge_back": "async_move_back",
}


def test_all_four_nudge_buttons_are_always_available() -> None:
    """`_nudge_available` consults nothing -- pressable is not safe."""
    assert _nudge_available(SimpleNamespace()) is True
    # And it must not depend on any coordinator state at all.
    assert _nudge_available(SimpleNamespace(data=None, online=False)) is True


def test_every_nudge_button_is_wired_to_its_own_direction() -> None:
    """A miswired direction would drive the mower the wrong way."""
    found = {d.key for d in BUTTON_SENSORS if d.key in _NUDGE_KEYS}
    assert found == set(_NUDGE_KEYS)
    for description in BUTTON_SENSORS:
        if description.key in _NUDGE_KEYS:
            assert description.available_fn is _nudge_available, description.key


@pytest.mark.asyncio
@pytest.mark.parametrize(("key", "method"), sorted(_NUDGE_KEYS.items()))
async def test_a_press_calls_the_primitive_directly_with_no_gates(
    key: str, method: str
) -> None:
    """🚨 Pins the bypass itself: press -> coordinator method, nothing between.

    If a gate is ever reintroduced on this path, this test fails and whoever
    added it has to say so out loud rather than quietly re-disabling the escape
    hatch the operator asked for.
    """
    coordinator = SimpleNamespace(
        **{name: AsyncMock() for name in _NUDGE_KEYS.values()}
    )

    await _unguarded_nudge(method)(coordinator)

    getattr(coordinator, method).assert_awaited_once_with()
    for other in _NUDGE_KEYS.values():
        if other != method:
            getattr(coordinator, other).assert_not_awaited()


@pytest.mark.asyncio
async def test_a_press_is_logged_as_ungated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The only trace an ungated press leaves. It must not be silent."""
    coordinator = SimpleNamespace(async_move_back=AsyncMock())

    with caplog.at_level(logging.WARNING, logger="custom_components.mammotion"):
        await _unguarded_nudge("async_move_back")(coordinator)

    assert "UNGATED NUDGE" in caplog.text
    assert "async_move_back" in caplog.text
