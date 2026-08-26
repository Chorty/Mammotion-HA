"""Fail-closed authorization and session state for experimental manual motion."""

from __future__ import annotations

import asyncio
import functools
import importlib.metadata
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .backend_capability import backend_capability_report
from .const import CONF_ENABLE_EXPERIMENTAL_MOTION

# The oldest release whose BLE transport has been read end to end here. It is a
# floor, never an authorization: any build at or above it must *additionally*
# prove it carries the audited teardown and reassembly fixes, because the same
# version number can be produced with or without them (a fork, a rebuild, or a
# future upstream release). See ``backend_capability.py`` for the probes.
MINIMUM_AUDITED_PYMAMMOTION_BASE_VERSION = "0.8.12"
#: How many segments of a multi-point path a real (non-dry-run) run may execute.
#:
#: Raised 2 -> 4 in beta31. Two segments reach only ~1-2 m per click (measured:
#: deadline-limited 1300 ms pulses travel 0.3496-0.4192 m, and committed 3-pulse
#: segment sums run 0.522-0.975 m), which is far short of the click-anywhere goal.
#: Each segment re-establishes ground truth against `waypoint_tolerance` at its
#: waypoint, so extra segments compose the same per-segment control law Gate 5
#: validated four times rather than extending any single open-loop leg.
#:
#: NOT a `LUBA_ACCEPTANCE_PROFILE` key -- raising it does not un-accept the
#: profile. It bounds both the schema Range and the runtime re-check in
#: `services.py`, so both follow automatically.
#:
#: Still unmeasured beyond segment 2: the VIO forward-heading offset is refreshed
#: only from linear travel and never re-derived across a turn, so cumulative
#: cross-track error over 3+ segments has never been observed. Watch landing error
#: against segment index on the first 4-segment run.
REAL_CLICK_TO_GO_SEGMENT_LIMIT = 4


class ManualMotionCancelledError(RuntimeError):
    """Raised before a cancelled session can send another nonzero command."""


@functools.lru_cache(maxsize=1)
def installed_pymammotion_version() -> str:
    """Return the installed PyMammotion distribution version."""
    try:
        return importlib.metadata.version("pymammotion")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _version_tuple(value: str) -> tuple[int, ...] | None:
    """Return the numeric release prefix, ignoring pre-release labels."""
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def backend_base_version_audited(version: str | None = None) -> bool:
    """Return whether the installed release is at or above the audited base."""
    installed = _version_tuple(version or installed_pymammotion_version())
    minimum = _version_tuple(MINIMUM_AUDITED_PYMAMMOTION_BASE_VERSION)
    return installed is not None and minimum is not None and installed >= minimum


def motion_backend_verified(
    version: str | None = None,
    *,
    capabilities: dict[str, Any] | None = None,
) -> bool:
    """Return whether the installed backend is proven safe for real motion.

    Two independent conditions, both required: the release is at or above the
    audited base, and the loaded code demonstrably carries both audited BLE
    fixes. The second is measured against the installed code, never inferred
    from the version string.
    """
    if not backend_base_version_audited(version):
        return False
    report = capabilities if capabilities is not None else backend_capability_report()
    return report.get("verified") is True


def experimental_motion_enabled(coordinator: Any) -> bool:
    """Return the explicit opt-in value; absence is always disabled.

    The type check must be ``Mapping``, not ``dict``: Home Assistant exposes
    ``ConfigEntry.options`` as a ``MappingProxyType``, which is a ``Mapping`` but
    is **not** a ``dict`` subclass. Checking for ``dict`` made this return False
    for every real config entry, so the opt-in could never be turned on no matter
    what the options flow stored -- and because the gate fails closed, it denied
    motion silently instead of surfacing an error.
    """
    entry = getattr(coordinator, "config_entry", None)
    options = getattr(entry, "options", None)
    return bool(
        isinstance(options, Mapping)
        and options.get(CONF_ENABLE_EXPERIMENTAL_MOTION, False)
    )


@dataclass(slots=True)
class ManualMotionSession:
    """One exclusive real-motion run owned by a mower coordinator."""

    owner: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    phase: str = "starting"
    started_monotonic: float = field(default_factory=time.monotonic)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    cancelled: bool = False
    cancel_reason: str | None = None
    last_completed_dispatch: dict[str, Any] | None = None
    stop_result: dict[str, Any] | None = None
    error: str | None = None
    owner_done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def as_dict(self) -> dict[str, Any]:
        """Return a service-safe session snapshot."""
        return {
            "session_id": self.session_id,
            "owner": self.owner,
            "phase": self.phase,
            "started_at": self.started_at.isoformat(),
            "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 3),
            "cancelled": self.cancelled,
            "cancel_reason": self.cancel_reason,
            "last_completed_dispatch": self.last_completed_dispatch,
            "stop_result": self.stop_result,
            "error": self.error,
        }


def active_motion_session(coordinator: Any) -> ManualMotionSession | None:
    """Return the active session when it has the expected type."""
    session = getattr(coordinator, "manual_motion_session", None)
    return session if isinstance(session, ManualMotionSession) else None


def assert_session_can_dispatch(coordinator: Any, *, is_stop: bool) -> None:
    """Prevent every later nonzero dispatch after an operator abort."""
    if is_stop:
        return
    session = active_motion_session(coordinator)
    if session is not None and session.cancelled:
        raise ManualMotionCancelledError(
            f"manual motion session {session.session_id} was cancelled"
        )


def record_completed_dispatch(
    coordinator: Any,
    *,
    command: str,
    is_stop: bool,
) -> None:
    """Record the last GATT-confirmed write for card/runtime diagnostics."""
    session = active_motion_session(coordinator)
    if session is None:
        return
    session.last_completed_dispatch = {
        "command": command,
        "is_stop": is_stop,
        "completed_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.monotonic() - session.started_monotonic, 3),
    }


def experimental_motion_status(
    coordinator: Any,
    *,
    ble_liveness: dict[str, Any] | None,
    safety: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the shared fail-closed backend state consumed by services/card."""
    version = installed_pymammotion_version()
    enabled = experimental_motion_enabled(coordinator)
    capabilities = backend_capability_report()
    verified = motion_backend_verified(version, capabilities=capabilities)
    blockers: list[str] = []
    if not enabled:
        blockers.append("experimental_motion_disabled")
    if not verified:
        blockers.append("pymammotion_backend_unverified")
        if not backend_base_version_audited(version):
            blockers.append("pymammotion_below_audited_base_version")
        blockers.extend(capabilities["reasons"])
    if not ble_liveness or ble_liveness.get("live") is not True:
        blockers.append(
            str((ble_liveness or {}).get("reason") or "ble_link_liveness_unavailable")
        )
    if not safety or safety.get("allowed_for_manual_motion") is not True:
        blockers.extend(
            str(blocker)
            for blocker in (safety or {}).get(
                "blockers", ["runtime_safety_unavailable"]
            )
        )
    session = active_motion_session(coordinator)
    last_session = getattr(coordinator, "last_manual_motion_session", None)
    return {
        "enabled": enabled,
        "installed_pymammotion_version": version,
        "minimum_verified_pymammotion_version": (
            MINIMUM_AUDITED_PYMAMMOTION_BASE_VERSION
        ),
        "backend_verified": verified,
        "backend_capabilities": capabilities,
        "ble_only": True,
        "real_click_to_go_segment_limit": REAL_CLICK_TO_GO_SEGMENT_LIMIT,
        "active_session": session.as_dict() if session is not None else None,
        "last_session": (
            last_session.as_dict()
            if isinstance(last_session, ManualMotionSession)
            else None
        ),
        "real_motion_allowed": not blockers and session is None,
        "blockers": list(dict.fromkeys(blockers)),
    }
