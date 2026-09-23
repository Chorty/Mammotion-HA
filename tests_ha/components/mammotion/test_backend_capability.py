"""Tests for the PyMammotion backend capability probes.

These probes are the sole evidence that authorizes real motion, so they are
tested in both directions: they must report absent against the installed
release, and present against code that actually carries each audited fix.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pymammotion.bluetooth.ble_message as vendor_ble_message
import pytest

from custom_components.mammotion import backend_capability
from custom_components.mammotion.backend_capability import (
    CAPABILITY_BLE_TEARDOWN,
    CAPABILITY_BLUFI_REASSEMBLY,
    REASON_NOT_PROBED,
    REQUIRED_MOTION_CAPABILITIES,
    _inbound_frame,
    _probe_blufi_reassembly_reset,
    _teardown_helper_covers_untestable_paths,
    async_probe_backend_capabilities,
    backend_capability_report,
    reset_backend_capability_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> Any:
    """Keep the process-wide probe cache from leaking between tests."""
    reset_backend_capability_cache()
    yield
    reset_backend_capability_cache()


def test_report_is_fail_closed_before_any_probe_runs() -> None:
    """An unprobed backend must never read as verified."""
    report = backend_capability_report()

    assert report["probed"] is False
    assert report["verified"] is False
    assert report["missing"] == list(REQUIRED_MOTION_CAPABILITIES)
    assert report["reasons"] == [REASON_NOT_PROBED]


async def test_installed_backend_carries_both_audited_fixes() -> None:
    """The pinned backend must carry both fixes, or real motion stays locked.

    This assertion is the release gate, and it is asserted against whatever
    backend is actually installed -- so it fails if the pin is ever moved to a
    build lacking either fix, including a downgrade to plain upstream 0.8.12.
    It was inverted deliberately when the 0.8.12.post1 fork build was pinned.
    """
    report = await async_probe_backend_capabilities()

    assert report["probed"] is True
    assert report["missing"] == []
    assert report["verified"] is True
    assert all(
        report["capabilities"][name] is True for name in REQUIRED_MOTION_CAPABILITIES
    )


async def test_probe_result_is_cached_until_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated authorization checks must not re-probe on every call."""
    calls = 0

    def counting_probe() -> bool:
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(
        backend_capability, "_probe_blufi_reassembly_reset", counting_probe
    )

    await async_probe_backend_capabilities()
    await async_probe_backend_capabilities()
    assert calls == 1

    await async_probe_backend_capabilities(force=True)
    assert calls == 2


async def test_a_raising_probe_reports_absent_rather_than_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend API drift must block motion, never break the service call."""

    def exploding_probe() -> bool:
        raise RuntimeError("pymammotion internals moved")

    monkeypatch.setattr(
        backend_capability, "_probe_blufi_reassembly_reset", exploding_probe
    )

    report = await async_probe_backend_capabilities()

    assert report["capabilities"][CAPABILITY_BLUFI_REASSEMBLY] is False
    assert report["verified"] is False


async def test_a_hanging_probe_times_out_and_reports_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe that never returns must not wedge the motion gate."""

    async def hanging_probe() -> bool:
        await asyncio.Event().wait()
        return True

    monkeypatch.setattr(backend_capability, "_PROBE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        backend_capability, "_probe_ble_teardown_failure_atomic", hanging_probe
    )

    report = await async_probe_backend_capabilities()

    assert report["capabilities"][CAPABILITY_BLE_TEARDOWN] is False


# ---------------------------------------------------------------------------
# Reassembly probe: exercised against a parser that does have the fix
# ---------------------------------------------------------------------------


class _FixedParser:
    """Minimal stand-in for a BleMessage that resets on a sequence gap."""

    def __init__(self, client: object) -> None:
        """Accept and ignore the client, as BleMessage does for the probe."""
        self.client = client
        self.buffer = b""
        self.expected_sequence = 0

    def parseNotification(self, response: bytes) -> int:  # noqa: N802 - vendor name
        """Accumulate frame data, discarding the buffer on a sequence gap."""
        sequence = response[2]
        fragmented = bool(response[1] & 16)
        if sequence != self.expected_sequence:
            self.buffer = b""  # the behaviour the probe is looking for
        self.expected_sequence = sequence + 1
        data_length = response[3]
        data = response[4 : 4 + data_length]
        self.buffer += data[2:] if fragmented else data
        return 1 if fragmented else 0

    @property
    def notification(self) -> Any:
        """Expose the accumulated bytes the way BlufiNotifyData does."""
        return _Accumulated(self.buffer)


class _Accumulated:
    """Read-only view of accumulated reassembly bytes."""

    def __init__(self, data: bytes) -> None:
        """Store the bytes accumulated so far."""
        self._data = data

    def getDataArray(self) -> bytes:  # noqa: N802 - vendor name
        """Return the accumulated payload."""
        return self._data


class _LeakyParser(_FixedParser):
    """Parser that keeps stale fragments across a sequence gap."""

    def parseNotification(self, response: bytes) -> int:  # noqa: N802 - vendor name
        """Accumulate without ever discarding an abandoned message."""
        self.expected_sequence = response[2] + 1
        data_length = response[3]
        data = response[4 : 4 + data_length]
        self.buffer += data[2:] if bool(response[1] & 16) else data
        return 1 if bool(response[1] & 16) else 0


@pytest.mark.parametrize(
    ("parser_type", "expected"),
    [(_FixedParser, True), (_LeakyParser, False)],
)
def test_reassembly_probe_distinguishes_fixed_from_leaky_parser(
    monkeypatch: pytest.MonkeyPatch,
    parser_type: type,
    expected: bool,
) -> None:
    """The probe must detect the reset, not merely run without error."""
    monkeypatch.setattr(vendor_ble_message, "BleMessage", parser_type)

    assert _probe_blufi_reassembly_reset() is expected


def test_inbound_frame_layout_matches_the_blufi_header() -> None:
    """The probe's frames must be shaped like real inbound BluFi frames."""
    frame = _inbound_frame(b"abc", 7, fragmented=False)

    assert frame[2] == 7
    assert frame[3] == 3
    assert frame[4:] == b"abc"

    fragmented = _inbound_frame(b"abc", 1, fragmented=True)

    assert bool(fragmented[1] & 16) is True
    assert fragmented[3] == 5  # two-byte total-length prefix plus payload
    assert fragmented[6:] == b"abc"


# ---------------------------------------------------------------------------
# Teardown probe: the source-level half covering paths we refuse to simulate
# ---------------------------------------------------------------------------


class _TransportWithoutHelper:
    """Transport lacking the shared teardown helper entirely."""

    async def connect(self) -> None:
        """Connect without any teardown on failure."""

    async def _write_payload(self, payload: bytes) -> None:
        """Write without any teardown on failure."""


class _TransportWithPartialCoverage:
    """Transport whose write path forgets to release the client."""

    async def _teardown_client(self, client: object) -> None:
        """Release a client."""

    async def connect(self) -> None:
        """Connect, releasing the client on failure."""
        await self._teardown_client(None)

    async def _write_payload(self, payload: bytes) -> None:
        """Write, leaking the client on failure."""


class _TransportWithFullCoverage(_TransportWithPartialCoverage):
    """Transport that routes both failure paths through teardown."""

    async def _write_payload(self, payload: bytes) -> None:
        """Write, releasing the client on failure."""
        await self._teardown_client(None)


@pytest.mark.parametrize(
    ("transport_type", "expected"),
    [
        (_TransportWithoutHelper, False),
        (_TransportWithPartialCoverage, False),
        (_TransportWithFullCoverage, True),
    ],
)
def test_teardown_coverage_check_requires_every_failure_path(
    transport_type: type,
    expected: bool,
) -> None:
    """A partial cherry-pick must not be reported as the audited fix."""
    assert _teardown_helper_covers_untestable_paths(transport_type) is expected
