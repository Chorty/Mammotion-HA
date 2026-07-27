#!/usr/bin/env python3
r"""Summarise BLE session lifetimes and disconnect reasons from the HA log.

Built for the heartbeat-cadence experiment: pymammotion holds the GATT link
open with a ``todev_ble_sync(2)`` heartbeat every
``ble_loop._KEEP_ALIVE_BLE_INTERVAL`` seconds (5.0 as shipped, ~1.5 in the
Mammotion app). Sessions on this mower die after ~73 s despite that, so the
question is whether a faster heartbeat lengthens them.

Reads log lines from stdin -- pipe `docker logs` output in -- and pairs
``connected=True`` with the following ``connected=False`` to get one session
per pair, with the negotiated MTU and the disconnect reason code.

Reason codes are ESP-IDF ``esp_gatt_conn_reason_t``:
    0x08 (8)  timeout                 -- supervision timeout (passive starvation)
    0x13 (19) terminate peer user     -- the MOWER hung up deliberately
    0x16 (22) terminate local host    -- we hung up
They discriminate between competing explanations, so the mix matters as much
as the durations.

Usage:
    scripts/ha_ssh.exp 'docker logs --since 90m homeassistant 2>&1' \\
        | .venv/bin/python scripts/ble_session_report.py
"""  # noqa: INP001

from __future__ import annotations

import datetime as dt
import re
import statistics
import sys

MOWER_MAC = "A8:B5:8E:2C:52:40"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_STATE = re.compile(r"connected=(True|False) mtu=(\d+) error=(\d+)")
_SEQ_GAP = "parseNotification read sequence wrong"
_PARSE_FAIL = "Failed to parse incoming bytes as LubaMsg"
_DROP = "dropping frame"

REASONS = {
    0: "clean/none",
    8: "0x08 supervision timeout (starvation)",
    19: "0x13 peer user terminated (mower hung up)",
    22: "0x16 local host terminated (we hung up)",
    62: "0x3E connection failed to establish",
}


def emit(message: str) -> None:
    """Print one report line."""
    print(message)  # noqa: T201


def main() -> None:  # noqa: C901
    """Parse stdin and print the session report."""
    opened: dt.datetime | None = None
    opened_mtu = 0
    sessions: list[tuple[float, int, int]] = []  # (seconds, mtu, reason)
    mtus: list[int] = []
    seq_gaps = parse_fails = drops = 0
    connects = disconnects = unpaired_connects = 0
    reasons: list[int] = []
    first = last = None

    for raw in sys.stdin:
        line = _ANSI.sub("", raw)
        stamp_match = _STAMP.match(line)
        if stamp_match:
            stamp = dt.datetime.strptime(stamp_match.group(1), "%Y-%m-%d %H:%M:%S")
            first = first or stamp
            last = stamp
        if _SEQ_GAP in line:
            seq_gaps += 1
        if _PARSE_FAIL in line:
            parse_fails += 1
        if _DROP in line:
            drops += 1

        if MOWER_MAC not in line or not stamp_match:
            continue
        state = _STATE.search(line)
        if not state:
            continue
        connected, mtu, error = state.groups()
        if connected == "True":
            connects += 1
            if opened is not None:
                # Two connects with no disconnect between them. bleak_esphome
                # can log the state more than once per logical session, so the
                # pair is measured from the LATEST connect -- which biases
                # durations DOWN. Counted so the bias is visible rather than
                # silently folded into the median.
                unpaired_connects += 1
            opened = stamp
            # mtu=0 on a connect means "already cached", not a negotiated
            # value -- bleak_esphome only reports a number on a fresh
            # negotiation. Counting the zeros as an MTU would invent a
            # low-MTU population that does not exist.
            if int(mtu) > 0:
                opened_mtu = int(mtu)
                mtus.append(int(mtu))
        else:
            disconnects += 1
            reasons.append(int(error))
            if opened is not None:
                sessions.append(
                    ((stamp - opened).total_seconds(), opened_mtu, int(error))
                )
                opened = None

    window = (last - first).total_seconds() / 60 if first and last else 0.0
    emit("=" * 66)
    emit(f"window: {window:.0f} min")
    emit(f"connect events: {connects}   disconnect events: {disconnects}"
         f"   paired sessions: {len(sessions)}")
    if unpaired_connects:
        emit(f"  NOTE: {unpaired_connects} connect(s) had no disconnect before the"
             " next connect;")
        emit("        those pairs are timed from the later connect, biasing"
             " durations DOWN.")
    if opened is not None:
        emit("  (one session still open at end of log, excluded)")

    if reasons:
        emit("")
        emit(f"ALL disconnect reasons ({len(reasons)} events, incl. unpaired):")
        for reason in sorted(set(reasons), key=lambda r: -reasons.count(r)):
            label = REASONS.get(reason, f"0x{reason:02X} unknown")
            share = 100 * reasons.count(reason) / len(reasons)
            emit(f"  {reasons.count(reason):>3}x  ({share:4.0f}%)  {label}")

    if sessions:
        durations = sorted(s[0] for s in sessions)
        emit("")
        emit("session lifetime (s):")
        emit(f"  min {durations[0]:.0f}   median {statistics.median(durations):.0f}"
             f"   max {durations[-1]:.0f}")
        emit(f"  all: {', '.join(f'{d:.0f}' for d in durations)}")

        emit("")
        emit("disconnect reasons:")
        by_reason: dict[int, list[float]] = {}
        for seconds, _mtu, reason in sessions:
            by_reason.setdefault(reason, []).append(seconds)
        for reason, secs in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            label = REASONS.get(reason, f"0x{reason:02X} unknown")
            emit(f"  {len(secs):>3}x  {label:<44} median {statistics.median(secs):.0f}s")

    if mtus:
        emit("")
        emit(f"negotiated MTU ({len(mtus)} fresh negotiations; connects logging"
             " mtu=0 reused a cached value):")
        for value in sorted(set(mtus)):
            # A low MTU forces more BluFi fragments per report, and each extra
            # fragment is another chance to lose one and poison the reassembly
            # buffer -- so MTU spread matters for the corruption rate.
            emit(f"  {mtus.count(value):>3}x  mtu={value}")

    emit("")
    emit("link quality over the same window:")
    per_min = f"  ({seq_gaps / window:.1f}/min)" if window else ""
    emit(f"  sequence gaps (packet loss) : {seq_gaps}{per_min}")
    emit(f"  unparseable LubaMsg frames  : {parse_fails}")
    emit(f"  dropped malformed frames    : {drops}")


if __name__ == "__main__":
    main()
