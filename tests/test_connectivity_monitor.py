"""Tests for the cloud connectivity watchdog decision logic.

The module under test is loaded directly from its file so these tests run
without Home Assistant installed (``connectivity.py`` is deliberately free of
HA imports).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).parent.parent / "custom_components" / "mammotion" / "connectivity.py"
)
_SPEC = importlib.util.spec_from_file_location("connectivity", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
connectivity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(connectivity)

CloudConnectivityMonitor = connectivity.CloudConnectivityMonitor
WatchdogAction = connectivity.WatchdogAction
RECONNECT_COOLDOWN_SECS = connectivity.RECONNECT_COOLDOWN_SECS


class FakeClock:
    """Controllable monotonic clock."""

    def __init__(self) -> None:
        """Start the clock at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current fake time."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Advance the fake time."""
        self.now += seconds


def _tick(
    monitor: CloudConnectivityMonitor,
    *,
    ble_usable: bool = False,
    cloud_registered: bool = True,
    cloud_connected: bool = True,
    auth_locked: bool = False,
) -> WatchdogAction:
    return monitor.tick(
        ble_usable=ble_usable,
        cloud_registered=cloud_registered,
        cloud_connected=cloud_connected,
        auth_locked=auth_locked,
    )


def test_healthy_ticks_do_nothing() -> None:
    """A connected cloud transport never triggers recovery."""
    monitor = CloudConnectivityMonitor(clock=FakeClock())
    for _ in range(10):
        assert _tick(monitor) is WatchdogAction.NONE
    assert not monitor.detached_warning_pending


def test_reconnect_after_two_disconnected_ticks() -> None:
    """Two consecutive disconnected ticks recommend a reconnect, once."""
    clock = FakeClock()
    monitor = CloudConnectivityMonitor(clock=clock)
    assert _tick(monitor, cloud_connected=False) is WatchdogAction.NONE
    assert _tick(monitor, cloud_connected=False) is WatchdogAction.RECONNECT
    monitor.record_reconnect_attempted()
    # Cooldown suppresses further attempts even after more bad ticks.
    for _ in range(5):
        assert _tick(monitor, cloud_connected=False) is WatchdogAction.NONE


def test_reconnect_allowed_again_after_cooldown() -> None:
    """After the cooldown elapses another reconnect is recommended."""
    clock = FakeClock()
    monitor = CloudConnectivityMonitor(clock=clock)
    _tick(monitor, cloud_connected=False)
    assert _tick(monitor, cloud_connected=False) is WatchdogAction.RECONNECT
    monitor.record_reconnect_attempted()
    # Streak was reset by the attempt; it rebuilds during the cooldown but
    # the cooldown keeps suppressing the action.
    assert _tick(monitor, cloud_connected=False) is WatchdogAction.NONE
    assert _tick(monitor, cloud_connected=False) is WatchdogAction.NONE
    clock.advance(RECONNECT_COOLDOWN_SECS + 1)
    assert _tick(monitor, cloud_connected=False) is WatchdogAction.RECONNECT


def test_recovery_resets_counters() -> None:
    """A healthy tick resets the disconnected streak."""
    monitor = CloudConnectivityMonitor(clock=FakeClock())
    assert _tick(monitor, cloud_connected=False) is WatchdogAction.NONE
    assert _tick(monitor) is WatchdogAction.NONE
    # Streak restarted — first bad tick after recovery doesn't fire.
    assert _tick(monitor, cloud_connected=False) is WatchdogAction.NONE


def test_detached_transport_warns_once_and_never_reconnects() -> None:
    """A missing cloud transport is warning-only (nothing to reconnect)."""
    monitor = CloudConnectivityMonitor(clock=FakeClock())
    assert not monitor.detached_warning_pending
    assert _tick(monitor, cloud_registered=False) is WatchdogAction.NONE
    assert monitor.detached_warning_pending
    monitor.record_detached_warning()
    assert _tick(monitor, cloud_registered=False) is WatchdogAction.NONE
    assert not monitor.detached_warning_pending


def test_detached_warning_rearms_after_recovery() -> None:
    """Once the transport is re-attached a later detach warns again."""
    monitor = CloudConnectivityMonitor(clock=FakeClock())
    _tick(monitor, cloud_registered=False)
    monitor.record_detached_warning()
    # Transport re-attached and healthy.
    assert _tick(monitor) is WatchdogAction.NONE
    # Detached again — warning pending again.
    _tick(monitor, cloud_registered=False)
    assert monitor.detached_warning_pending


def test_ble_usable_suppresses_everything() -> None:
    """A usable BLE link means the watchdog never acts."""
    monitor = CloudConnectivityMonitor(clock=FakeClock())
    for _ in range(5):
        action = _tick(
            monitor, ble_usable=True, cloud_registered=False, cloud_connected=False
        )
        assert action is WatchdogAction.NONE
    assert not monitor.detached_warning_pending


def test_auth_locked_suppresses_everything() -> None:
    """An unrecoverable auth failure is owned by the reauth flow."""
    monitor = CloudConnectivityMonitor(clock=FakeClock())
    for _ in range(5):
        assert (
            _tick(monitor, cloud_connected=False, auth_locked=True)
            is WatchdogAction.NONE
        )
