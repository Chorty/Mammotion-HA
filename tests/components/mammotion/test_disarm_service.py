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

from types import SimpleNamespace

from custom_components.mammotion import manual_motion
from custom_components.mammotion import services as mammotion_services
from custom_components.mammotion.const import CONF_ENABLE_EXPERIMENTAL_MOTION
from custom_components.mammotion.manual_motion import experimental_motion_enabled


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
