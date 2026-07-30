"""Behavioural proof that the installed PyMammotion carries the audited BLE fixes.

A version number cannot prove a fix is present. The same number can be produced
by a fork, a local rebuild, or a future upstream release that does or does not
contain a given commit, so gating real motion on ``>= X.Y.Z`` trusts a label
rather than the code that will actually drive the mower.

These probes exercise the installed code paths directly and observe the fixed
behaviour, so authorization rests on measurement:

``ble_teardown_failure_atomic``
    Every established BLE client must be released on every failure path.
    Probed by driving :meth:`BLETransport.disconnect` with a client that reports
    ``is_connected`` False -- the unfixed code skips cleanup entirely and leaks
    the proxy/adapter connection slot. The two paths that cannot be simulated
    without patching module globals at runtime (post-connect setup failure and a
    failed write) are confirmed to route through the shared teardown helper.

``blufi_reassembly_reset``
    A dropped BluFi fragment must not prefix stale bytes onto the next completed
    report. Probed by feeding a fragment and then skipping a sequence number: the
    unfixed parser keeps the abandoned bytes in its buffer.

Every probe fails closed. Any exception, missing attribute, or unreadable source
reports the capability as absent, which keeps real motion blocked.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Final

_LOGGER = logging.getLogger(__name__)

CAPABILITY_BLE_TEARDOWN: Final = "ble_teardown_failure_atomic"
CAPABILITY_BLUFI_REASSEMBLY: Final = "blufi_reassembly_reset"

#: Both fixes are required before a nonzero movement command may be dispatched.
REQUIRED_MOTION_CAPABILITIES: Final = (
    CAPABILITY_BLE_TEARDOWN,
    CAPABILITY_BLUFI_REASSEMBLY,
)

#: Reported while no probe has run yet. Absence of evidence blocks motion.
REASON_NOT_PROBED: Final = "backend_capability_probe_not_run"

#: ``getTypeValue(1, 19)``: BluFi custom-data subtype 19 with package type 1.
_CUSTOM_DATA_TYPE: Final = 77

_PROBE_TIMEOUT_SECONDS: Final = 5.0

#: Raised by ``inspect.getsource`` for a source-less or non-Python object.
_UNREADABLE_SOURCE_ERRORS: Final = (OSError, TypeError)

_probe_cache: dict[str, Any] | None = None
_probe_lock = asyncio.Lock()


class _UnusedClient:
    """Stand-in for a BleakClient that the reassembly probe never touches."""


class _ProbeClient:
    """Minimal client recording whether teardown actually released it."""

    def __init__(self) -> None:
        """Start disconnected so the unfixed cleanup gate skips this client."""
        self.is_connected = False
        self.disconnect_calls = 0

    async def disconnect(self) -> None:
        """Record the teardown call the fixed transport is required to make."""
        self.disconnect_calls += 1


def _inbound_frame(payload: bytes, sequence: int, *, fragmented: bool) -> bytes:
    """Build one raw inbound BluFi frame carrying *payload*."""
    from pymammotion.bluetooth.data.framectrldata import FrameCtrlData  # noqa: PLC0415

    frame_control = FrameCtrlData.getFrameCTRLValue(False, False, 1, False, fragmented)
    data = len(payload).to_bytes(2, "little") + payload if fragmented else payload
    return bytes([_CUSTOM_DATA_TYPE, frame_control, sequence, len(data)]) + data


def _probe_blufi_reassembly_reset() -> bool:
    """Return True when a sequence gap discards the partial reassembly buffer."""
    from pymammotion.bluetooth.ble_message import BleMessage  # noqa: PLC0415

    unused_client: Any = _UnusedClient()
    receiver = BleMessage(unused_client)
    if receiver.parseNotification(_inbound_frame(b"stale", 0, fragmented=True)) != 1:
        return False
    if receiver.parseNotification(_inbound_frame(b"fresh", 2, fragmented=False)) != 0:
        return False
    # Unfixed parsers return b"stalefresh" here.
    return bytes(receiver.notification.getDataArray()) == b"fresh"


def _teardown_helper_covers_untestable_paths(transport_type: type) -> bool:
    """Return True when connect and write failures also route through teardown.

    Simulating those two paths would mean patching ``establish_connection`` on
    the live module, which could strand a concurrent real connection. Confirm
    the shared helper is called from both instead.
    """
    helper = getattr(transport_type, "_teardown_client", None)
    if helper is None:
        return False
    for method_name in ("connect", "_write_payload"):
        method = getattr(transport_type, method_name, None)
        if method is None:
            return False
        try:
            source = inspect.getsource(method)
        except _UNREADABLE_SOURCE_ERRORS:
            # A source-less install cannot be audited, so it is not verified.
            return False
        if "_teardown_client" not in source:
            return False
    return True


async def _probe_ble_teardown_failure_atomic() -> bool:
    """Return True when teardown releases a client reporting is_connected False."""
    from pymammotion.transport.ble import (  # noqa: PLC0415
        BLETransport,
        BLETransportConfig,
    )

    if not _teardown_helper_covers_untestable_paths(BLETransport):
        return False

    transport: Any = BLETransport(
        BLETransportConfig(
            device_id="mammotion-capability-probe",
            ble_address="00:00:00:00:00:00",
        )
    )
    probe_client = _ProbeClient()
    transport._client = probe_client  # noqa: SLF001
    await transport.disconnect()
    return probe_client.disconnect_calls == 1 and transport._client is None  # noqa: SLF001


async def _run_probe(name: str, probe: Any) -> bool:
    """Run one probe under a timeout, reporting absent on any failure."""
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_SECONDS):
            result = probe()
            if inspect.isawaitable(result):
                result = await result
    except Exception:  # noqa: BLE001 - an unprovable capability must fail closed
        _LOGGER.debug("Backend capability probe %s failed", name, exc_info=True)
        return False
    else:
        return result is True


def _build_report(capabilities: dict[str, bool], *, probed: bool) -> dict[str, Any]:
    """Return the shared capability report consumed by services and the card."""
    missing = [
        name
        for name in REQUIRED_MOTION_CAPABILITIES
        if capabilities.get(name) is not True
    ]
    return {
        "probed": probed,
        "capabilities": dict(capabilities),
        "missing": missing,
        "verified": probed and not missing,
        "reasons": (
            [REASON_NOT_PROBED]
            if not probed
            else [f"backend_missing_{name}" for name in missing]
        ),
    }


def unprobed_backend_capability_report() -> dict[str, Any]:
    """Return the fail-closed report used before any probe has run."""
    return _build_report(
        dict.fromkeys(REQUIRED_MOTION_CAPABILITIES, False), probed=False
    )


def backend_capability_report() -> dict[str, Any]:
    """Return the cached probe result, or the fail-closed report when unprobed."""
    if _probe_cache is None:
        return unprobed_backend_capability_report()
    return _build_report(_probe_cache["capabilities"], probed=True)


async def async_probe_backend_capabilities(*, force: bool = False) -> dict[str, Any]:
    """Probe the installed backend once and cache the result.

    Safe to await from any code path: the probes touch no hardware, allocate no
    connection, and the cached result makes every later call free.
    """
    global _probe_cache  # noqa: PLW0603 - one process-wide backend to describe

    async with _probe_lock:
        if _probe_cache is not None and not force:
            return backend_capability_report()
        capabilities = {
            CAPABILITY_BLE_TEARDOWN: await _run_probe(
                CAPABILITY_BLE_TEARDOWN, _probe_ble_teardown_failure_atomic
            ),
            CAPABILITY_BLUFI_REASSEMBLY: await _run_probe(
                CAPABILITY_BLUFI_REASSEMBLY, _probe_blufi_reassembly_reset
            ),
        }
        _probe_cache = {"capabilities": capabilities}
        report = backend_capability_report()

    if report["verified"]:
        _LOGGER.debug("PyMammotion backend capabilities verified: %s", capabilities)
    else:
        _LOGGER.info(
            "PyMammotion backend is missing audited motion fixes %s; "
            "real experimental motion stays blocked",
            report["missing"],
        )
    return report


def reset_backend_capability_cache() -> None:
    """Forget the cached probe result (used by tests and after a version change)."""
    global _probe_cache  # noqa: PLW0603 - one process-wide backend to describe

    _probe_cache = None
