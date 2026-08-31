"""Cloud connectivity monitoring for Mammotion devices.

Pure decision logic for the coordinator's connectivity watchdog: given the
observed transport state each report tick, decide whether an in-place cloud
reconnect should be attempted.  Kept free of Home Assistant imports so it can
be unit-tested without a ``hass`` fixture.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum

# Consecutive ticks a registered cloud transport must be observed disconnected
# before attempting a reconnect (2 ticks ~= 10 minutes at the 5-minute report
# interval) — a single bad tick is usually pymammotion's own reconnect loop
# already mid-recovery.
DISCONNECTED_TICKS_BEFORE_RECONNECT = 2

# Minimum spacing between reconnect attempts per device, so the watchdog never
# fights pymammotion's own exponential-backoff reconnect.
RECONNECT_COOLDOWN_SECS = 15 * 60.0


class WatchdogAction(Enum):
    """Recovery action recommended by :class:`CloudConnectivityMonitor`."""

    NONE = "none"
    RECONNECT = "reconnect"


class CloudConnectivityMonitor:
    """Track cloud transport health across ticks and recommend recovery.

    ``tick`` is called once per coordinator update with the current transport
    observations and returns the action the caller should take:

    * BLE usable or an unrecoverable auth failure → never act (BLE covers the
      device; auth recovery is owned by the reauth flow).
    * Cloud transport registered but disconnected for
      ``DISCONNECTED_TICKS_BEFORE_RECONNECT`` consecutive ticks →
      ``RECONNECT`` (rate-limited by ``RECONNECT_COOLDOWN_SECS``).
    * Cloud transport missing entirely (detached by pymammotion after a failed
      unbound migration) → nothing can be done in place;
      ``detached_warning_pending`` turns True so the caller can warn once.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Initialize the monitor with an injectable monotonic clock."""
        self._clock = clock
        self._disconnected_ticks = 0
        self._last_reconnect_attempt: float | None = None
        self._detached_seen = False
        self._detached_warned = False

    @property
    def detached_warning_pending(self) -> bool:
        """Return True when a detached cloud transport still needs a warning."""
        return self._detached_seen and not self._detached_warned

    def record_detached_warning(self) -> None:
        """Mark the detached-transport warning as emitted."""
        self._detached_warned = True

    def record_reconnect_attempted(self) -> None:
        """Record that the caller attempted a reconnect just now."""
        self._last_reconnect_attempt = self._clock()
        self._disconnected_ticks = 0

    def tick(
        self,
        *,
        ble_usable: bool,
        cloud_registered: bool,
        cloud_connected: bool,
        auth_locked: bool,
    ) -> WatchdogAction:
        """Evaluate one observation and return the recommended action."""
        if ble_usable or auth_locked:
            self._reset()
            return WatchdogAction.NONE

        if not cloud_registered:
            # The detached (#808) state: nothing registered to reconnect.
            self._disconnected_ticks = 0
            self._detached_seen = True
            return WatchdogAction.NONE

        # A cloud transport is registered again — the detached state cleared.
        self._detached_seen = False
        self._detached_warned = False

        if cloud_connected:
            self._reset()
            return WatchdogAction.NONE

        self._disconnected_ticks += 1
        if self._disconnected_ticks < DISCONNECTED_TICKS_BEFORE_RECONNECT:
            return WatchdogAction.NONE
        if (
            self._last_reconnect_attempt is not None
            and self._clock() - self._last_reconnect_attempt < RECONNECT_COOLDOWN_SECS
        ):
            return WatchdogAction.NONE
        return WatchdogAction.RECONNECT

    def _reset(self) -> None:
        self._disconnected_ticks = 0
        self._detached_seen = False
        self._detached_warned = False
