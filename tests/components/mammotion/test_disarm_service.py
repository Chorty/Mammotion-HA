"""Pins `mammotion.disarm_experimental_motion`.

🔒 The service is ONE-WAY on purpose. Arming stays behind the options flow --
a human in Settings -- because a service that could arm would let any
automation, script, scene or voice assistant open the motion gate. This one can
only ever close it, so the worst a bug or a stray call can do is refuse to move.

It exists because the gate was found armed at rest three times on 2026-08-18,
once with zero blockers and the mower off its dock. Nothing in HA could close
it: the gate is a config entry option, so there was no entity to toggle and no
service to call.
"""

from __future__ import annotations

import inspect
import pathlib
from types import SimpleNamespace

from custom_components.mammotion import manual_motion
from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.const import CONF_ENABLE_EXPERIMENTAL_MOTION
from custom_components.mammotion.manual_motion import (
    active_motion_session,
    experimental_motion_enabled,
    experimental_motion_status,
)


def test_there_is_no_arm_service() -> None:
    """The absence of an arming counterpart is the safety property.

    ⚠️ If a future change adds one, that is a deliberate widening of who can
    move the mower and it must be argued for explicitly -- not slipped in for
    symmetry with this one.
    """
    names = [
        getattr(mammotion_services, n)
        for n in dir(mammotion_services)
        if n.startswith("SERVICE_")
    ]
    for name in names:
        if not isinstance(name, str):
            continue
        assert "arm_experimental" not in name or name.startswith("disarm"), name
        assert name != "enable_experimental_motion", (
            "an arming service would let any automation open the motion gate"
        )
    assert (
        mammotion_services.SERVICE_DISARM_EXPERIMENTAL_MOTION
        == "disarm_experimental_motion"
    )


def test_the_option_key_is_the_one_the_gate_actually_reads() -> None:
    """The service writes the same key `experimental_motion_enabled` reads.

    A mismatch here would produce a service that reports success, changes an
    unrelated option, and leaves the gate wide open -- failing in the dangerous
    direction while looking fine.
    """
    assert manual_motion.CONF_ENABLE_EXPERIMENTAL_MOTION is (
        CONF_ENABLE_EXPERIMENTAL_MOTION
    )
    assert CONF_ENABLE_EXPERIMENTAL_MOTION == "enable_experimental_motion"


def test_the_gate_fails_closed_when_the_option_is_absent() -> None:
    """Absence must mean disabled, so disarming by removal is never ambiguous."""

    class _Entry:
        options: dict = {}

    coordinator = SimpleNamespace(config_entry=_Entry())
    assert experimental_motion_enabled(coordinator) is False

    coordinator.config_entry.options = {CONF_ENABLE_EXPERIMENTAL_MOTION: False}
    assert experimental_motion_enabled(coordinator) is False

    coordinator.config_entry.options = {CONF_ENABLE_EXPERIMENTAL_MOTION: True}
    assert experimental_motion_enabled(coordinator) is True


def test_the_handler_only_calls_helpers_with_signatures_it_satisfies() -> None:
    """🚨 THE TEST THAT WAS MISSING, and its absence shipped a 500.

    The first version of this service called `experimental_motion_status(
    coordinator)` -- but that helper takes two REQUIRED keyword-only arguments,
    `ble_liveness` and `safety`, which a disarm handler has no reason to
    compute. Every unit test here passed, because they all exercised constants
    and option keys AROUND the handler and never the handler itself. It failed
    at HTTP 500 on the first live call after deploy.

    Disarming needs exactly two facts -- is it on, and is a run in progress --
    so the handler reads both directly. This test pins that it does not reach
    for the full status builder again, and that the two helpers it does use are
    callable with a bare coordinator.
    """
    # The helper the handler must NOT use, and why.
    params = inspect.signature(experimental_motion_status).parameters
    required_kwonly = [
        n
        for n, p in params.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
        and p.default is inspect.Parameter.empty
    ]
    assert required_kwonly == ["ble_liveness", "safety"], (
        "experimental_motion_status' signature changed; re-check the handler"
    )

    # The two it does use must be callable with only a coordinator.
    for fn in (experimental_motion_enabled, active_motion_session):
        sig = inspect.signature(fn)
        needs = [
            n
            for n, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty
            and p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        assert needs == ["coordinator"], f"{fn.__name__} now needs {needs}"

    # And the handler's source must not have reacquired the dependency.
    src = pathlib.Path("custom_components/mammotion/services.py").read_text()
    start = src.index("async def handle_disarm_experimental_motion")
    body = src[start : src.index("\n    hass.services.async_register", start)]
    # Comments out -- the handler explains at length WHY it avoids that helper,
    # and matching the explanation would fail the check it is explaining.
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert "experimental_motion_status(" not in code, (
        "handler calls experimental_motion_status again -- it needs "
        "ble_liveness and safety, which this handler does not have"
    )
