"""Tests for the link-health diagnostics: cloud budget, command timeouts, refresh.

Adapted from upstream mikey0000/Mammotion-HA PRs #800 and #836. The behaviour
tests matter more than the entity plumbing: the command-timeout counter sits in
`async_send_command`, the single funnel every queued command passes through, and
a diagnostic that quietly swallowed the exception would be a control-flow change
wearing a sensor's clothes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pymammotion.transport import CommandTimeoutError

from custom_components.mammotion import coordinator as coordinator_module
from custom_components.mammotion.button import BUTTON_SENSORS
from custom_components.mammotion.coordinator import (
    CLOUD_SEND_LIMIT_STATES,
    MammotionBaseUpdateCoordinator,
)
from custom_components.mammotion.sensor import WORK_SENSOR_TYPES

LOCALES = ("cs", "da", "de", "en", "fr", "hu", "it", "nl", "pl", "ro", "sl", "sv")
NEW_SENSOR_KEYS = ("cloud_sends_24h", "cloud_send_limit", "command_timeouts_24h")
NEW_BUTTON_KEY = "refresh_status"


def _integration_json(name: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3] / "custom_components" / "mammotion"
    return json.loads((root / name).read_text(encoding="utf-8"))


class _ConcreteCoordinator(MammotionBaseUpdateCoordinator):
    """Minimal concrete subclass -- the base declares one abstract method."""

    def get_coordinator_data(self, device: Any) -> Any:
        return device


def _bare_coordinator() -> Any:
    """Build a coordinator carrying only the attributes these properties touch.

    Constructing the real thing needs a HA instance and a config entry; every
    property under test reads `self.manager`, `self.device_name` and
    `self._command_timeouts` and nothing else.
    """
    coordinator = object.__new__(_ConcreteCoordinator)
    coordinator._command_timeouts = coordinator_module.deque()  # noqa: SLF001
    coordinator.device_name = "mower"
    coordinator.manager = SimpleNamespace(mower=lambda _name: None)
    return coordinator


# ---------------------------------------------------------------- translations


def test_new_entities_are_named_in_every_locale() -> None:
    """A missing name renders as a blank entity, so every locale must carry one."""
    for filename in ["strings.json", *[f"translations/{loc}.json" for loc in LOCALES]]:
        doc = _integration_json(filename)
        buttons = doc.get("entity", {}).get("button", {})
        sensors = doc.get("entity", {}).get("sensor", {})
        assert buttons.get(NEW_BUTTON_KEY, {}).get("name"), (
            f"{filename}: button.{NEW_BUTTON_KEY}"
        )
        for key in NEW_SENSOR_KEYS:
            assert sensors.get(key, {}).get("name"), f"{filename}: sensor.{key}"


def test_enum_sensor_states_are_translated_in_every_locale() -> None:
    """An ENUM sensor with an untranslated state renders the raw slug."""
    for filename in ["strings.json", *[f"translations/{loc}.json" for loc in LOCALES]]:
        doc = _integration_json(filename)
        states = (
            doc.get("entity", {})
            .get("sensor", {})
            .get("cloud_send_limit", {})
            .get("state", {})
        )
        assert set(states) == set(CLOUD_SEND_LIMIT_STATES), filename
        for value in states.values():
            assert value, f"{filename}: empty cloud_send_limit state"


def test_non_english_locales_are_not_english_placeholders() -> None:
    """CLAUDE.md forbids copying the English string into the other locales."""
    english = _integration_json("translations/en.json")["entity"]
    for loc in (loc for loc in LOCALES if loc != "en"):
        doc = _integration_json(f"translations/{loc}.json")["entity"]
        for key in ("cloud_sends_24h", "command_timeouts_24h"):
            assert doc["sensor"][key]["name"] != english["sensor"][key]["name"], (
                f"{loc}: sensor.{key} is still the English string"
            )
        assert (
            doc["button"][NEW_BUTTON_KEY]["name"]
            != english["button"][NEW_BUTTON_KEY]["name"]
        ), f"{loc}: button.{NEW_BUTTON_KEY} is still the English string"


def test_new_entities_have_icons() -> None:
    """An icon-less diagnostic entity is hard to find in a long entity list."""
    icons = _integration_json("icons.json")["entity"]
    assert icons["button"][NEW_BUTTON_KEY]["default"]
    for key in NEW_SENSOR_KEYS:
        assert icons["sensor"][key]["default"], key


# ------------------------------------------------------------------- wiring


def test_the_entities_are_registered() -> None:
    """Translations without a registered entity are dead weight, and vice versa."""
    sensor_keys = {d.key for d in WORK_SENSOR_TYPES}
    assert set(NEW_SENSOR_KEYS) <= sensor_keys
    assert NEW_BUTTON_KEY in {d.key for d in BUTTON_SENSORS}


def test_enum_sensor_declares_its_options() -> None:
    """Without `options` HA rejects an ENUM sensor's state at runtime."""
    description = next(d for d in WORK_SENSOR_TYPES if d.key == "cloud_send_limit")
    assert set(description.options or []) == set(CLOUD_SEND_LIMIT_STATES)


# --------------------------------------------------------- timeout accounting


def test_command_timeouts_start_at_zero() -> None:
    """A healthy link reads 0, not unknown."""
    assert _bare_coordinator().command_timeouts_in_window == 0


def test_command_timeouts_accumulate_and_expire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window is rolling: entries older than 24 h fall out of the count."""
    coordinator = _bare_coordinator()
    now = 1_000.0
    monkeypatch.setattr(coordinator_module.time, "monotonic", lambda: now)

    for _ in range(3):
        coordinator.record_command_timeout()
    assert coordinator.command_timeouts_in_window == 3

    now += 23 * 60 * 60  # still inside the window
    assert coordinator.command_timeouts_in_window == 3

    now += 2 * 60 * 60  # 25 h after the first three
    coordinator.record_command_timeout()
    assert coordinator.command_timeouts_in_window == 1


@pytest.mark.asyncio
async def test_a_command_timeout_is_counted_and_STILL_RAISED() -> None:
    """The counter must not become control flow.

    Callers of `async_send_command` already handle `CommandTimeoutError` -- some
    catch it and pass. Swallowing it here to make the accounting tidier would
    silently change what happens when a command times out.
    """
    coordinator = _bare_coordinator()
    coordinator._bluetooth_enabled = True  # noqa: SLF001
    coordinator.update_failures = 0
    coordinator.last_command_failure_reason = None
    coordinator.is_online = lambda: True

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise CommandTimeoutError("some_field", 3)

    coordinator.manager = SimpleNamespace(
        get_device_by_name=lambda _n: SimpleNamespace(),
        send_command_with_args=_raise,
        mower=lambda _n: None,
    )

    with pytest.raises(CommandTimeoutError):
        await coordinator.async_send_command("some_command")
    assert coordinator.command_timeouts_in_window == 1


# ------------------------------------------------------------ cloud transport


def test_cloud_metrics_are_none_without_a_transport() -> None:
    """None is not zero: no cloud transport is a different fact from no sends."""
    coordinator = _bare_coordinator()
    assert coordinator.cloud_sends_in_window is None
    assert coordinator.cloud_send_limit_state is None


@pytest.mark.parametrize(
    ("rate_limited", "expected"), [(False, "ok"), (True, "rate_limited")]
)
def test_cloud_metrics_read_the_transport(rate_limited: bool, expected: str) -> None:
    """Both budget sensors read straight off the live transport."""
    coordinator = _bare_coordinator()
    transport = SimpleNamespace(
        sends_in_window=lambda: 42, is_rate_limited=rate_limited
    )
    coordinator.manager = SimpleNamespace(
        mower=lambda _n: SimpleNamespace(
            has_transport=lambda _t: True,
            get_transport=lambda _t: transport,
            is_transport_connected=lambda _t: True,
        )
    )
    assert coordinator.cloud_sends_in_window == 42
    assert coordinator.cloud_send_limit_state == expected


def test_a_connected_transport_wins_over_a_registered_one() -> None:
    """Two cloud transports can be registered; report the live one."""
    coordinator = _bare_coordinator()
    idle = SimpleNamespace(sends_in_window=lambda: 1, is_rate_limited=False)
    live = SimpleNamespace(sends_in_window=lambda: 99, is_rate_limited=True)
    transports = {
        coordinator_module.TransportType.CLOUD_ALIYUN: idle,
        coordinator_module.TransportType.CLOUD_MAMMOTION: live,
    }
    coordinator.manager = SimpleNamespace(
        mower=lambda _n: SimpleNamespace(
            has_transport=lambda t: t in transports,
            get_transport=lambda t: transports[t],
            is_transport_connected=lambda t: transports[t] is live,
        )
    )
    assert coordinator.cloud_sends_in_window == 99
    assert coordinator.cloud_send_limit_state == "rate_limited"


def test_a_broken_transport_degrades_to_none_rather_than_raising() -> None:
    """A diagnostic must never take the coordinator down with it."""
    coordinator = _bare_coordinator()
    broken = SimpleNamespace()  # neither attribute present
    coordinator.manager = SimpleNamespace(
        mower=lambda _n: SimpleNamespace(
            has_transport=lambda _t: True,
            get_transport=lambda _t: broken,
            is_transport_connected=lambda _t: True,
        )
    )
    assert coordinator.cloud_sends_in_window is None
    assert coordinator.cloud_send_limit_state is None
