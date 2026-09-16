#!/usr/bin/env python3
"""Trace live BLE RSSI against position while the mower is CONNECTED.

Companion to ``ble_proxy_coverage_monitor.py``, which only records data while
the mower is disconnected and advertising -- useless during an actual driven
leg, since the mower stops advertising the instant it connects and stays
connected for the whole drive. This script covers exactly that gap: while
connected, poll position and the *live* connection RSSI on a tight interval,
tagged with whichever proxy currently holds the connection.

🚨 **This gives ONE proxy's coverage per run, not all four at once.** HA
routes to whichever proxy currently has the connection; there is no way to
see a second proxy's signal to a device that is already connected elsewhere
(the point-to-point nature of BLE, not a limitation of this script). To
compare proxies along the SAME driven path, run this once per proxy, with
the other three temporarily disabled (the ESPHome reflash method already
used and verified 2026-09-15 -- comment out `esp32_ble_tracker`/
`bluetooth_proxy`, reflash, confirm the proxy is gone from
`bluetooth/subscribe_scanner_details` before starting the drive). This
script does not do that disabling itself -- it only records whatever HA
picks for the connection at the time, whether that's the one proxy left
enabled or (if all are enabled) whichever wins by RSSI.

Position and RSSI both come from ``export_runtime_state`` in one call --
``position.x``/``position.y`` (the executor's ``mower_map_xy`` frame) and
``transport.ble_rssi`` (the mower's own self-reported connection RSSI).
🔑 **`ble_rssi` is documented elsewhere in this project as self-reported and
sometimes stale relative to a real drop** -- fine here, since the question
this script answers is "what does the mower itself see from wherever it is
right now", which is exactly what ``ble_rssi`` reports, not "is the link
about to fail" (a different, already-solved question).

Which proxy currently holds the connection comes from a live
``bluetooth/subscribe_connection_allocations`` subscription (the same
mechanism ``ws_bt.py`` snapshots once) -- tracked continuously here instead
of re-queried, so a mid-drive proxy handoff (a real event, seen 2026-09-15)
is captured at the moment it happens, not missed between polls.

Output is append-only (``ble_proxy_connected_trace_log.jsonl``), one row per
position poll, so multiple drives -- one per proxy -- accumulate into a
single growing dataset alongside the passive collector's.

Read-only -- sends no movement commands. Run this ALONGSIDE a real
click-to-path session (in another terminal/background task), not instead of
one; it does not drive the mower itself.

Usage:  .venv/bin/python scripts/ble_proxy_connected_trace.py [seconds] [--interval N]
"""  # noqa: INP001

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

MOWER_MAC = "A8:B5:8E:2C:52:40"
ENTITY = "lawn_mower.back_yard_clip_skywalker"
OUT = Path(__file__).with_name("ble_proxy_connected_trace_log.jsonl")
REPO_ENV = Path(__file__).resolve().parent.parent / ".env"

#: Tight enough to trace a driven path meaningfully; the mower's own report
#: cadence tops out at ~1 Hz anyway (project-wide measured ceiling), so
#: polling much faster than this buys nothing.
DEFAULT_INTERVAL_S = 3.0
DEFAULT_DURATION_S = 1800.0


def load_env() -> dict[str, str]:
    """Return ``HA_URL``/``HA_TOKEN`` from the repo ``.env``."""
    values: dict[str, str] = {}
    for line in REPO_ENV.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def emit(message: str) -> None:
    """Print one line of progress output."""
    print(message, flush=True)  # noqa: T201


class AuthRejected(Exception):
    """Home Assistant rejected the access token. Terminal, never retried."""


class CurrentProxy:
    """The mower's currently-allocated proxy.

    Updated live from the connection-allocations subscription.
    """

    def __init__(self) -> None:
        """Start with no allocation known yet."""
        self.source_mac: str | None = None
        self.name = "?"

    def set(self, source_mac: str | None, scanners: dict[str, str]) -> None:
        """Record the mower's current allocation; ``None`` means disconnected."""
        self.source_mac = source_mac
        self.name = (
            scanners.get(source_mac, source_mac or "?")
            if source_mac
            else "disconnected"
        )


def _handle_allocation_event(
    msg: dict[str, object], current: CurrentProxy, scanners: dict[str, str]
) -> None:
    """Apply one subscription event: a scanner-registry add, or an allocation update."""
    if msg.get("type") != "event":
        return
    event = msg.get("event")
    if msg.get("id") == 1:
        if isinstance(event, dict):
            for entry in event.get("add", []) or []:
                scanners[entry.get("source", "")] = entry.get("name", "?")
        return
    if msg.get("id") == 2 and isinstance(event, list):
        found = next(
            (a.get("source") for a in event if MOWER_MAC in (a.get("allocated") or [])),
            None,
        )
        current.set(found, scanners)


async def _track_allocations_once(
    session: aiohttp.ClientSession,
    ws_url: str,
    token: str,
    deadline: float,
    current: CurrentProxy,
    scanners: dict[str, str],
) -> None:
    """Run one websocket session, applying every event until it ends or ``deadline``."""
    async with session.ws_connect(ws_url, max_msg_size=0) as ws:
        await ws.receive_json()
        await ws.send_json({"type": "auth", "access_token": token})
        auth = await ws.receive_json()
        if auth["type"] != "auth_ok":
            raise AuthRejected(str(auth))
        await ws.send_json({"id": 1, "type": "bluetooth/subscribe_scanner_details"})
        await ws.send_json(
            {"id": 2, "type": "bluetooth/subscribe_connection_allocations"}
        )
        while time.time() < deadline:
            msg = await asyncio.wait_for(
                ws.receive_json(), timeout=max(5.0, deadline - time.time())
            )
            _handle_allocation_event(msg, current, scanners)


async def _track_allocations(
    session: aiohttp.ClientSession,
    ws_url: str,
    token: str,
    deadline: float,
    current: CurrentProxy,
    scanners: dict[str, str],
) -> None:
    """Keep ``current`` updated with whichever scanner has the mower allocated.

    Reconnects on a drop, same backoff discipline as the sibling script --
    a gap here just means the last-known proxy is used a little longer than
    it should be, never a crash.
    """
    backoff = 2.0
    while time.time() < deadline:
        try:
            await _track_allocations_once(
                session, ws_url, token, deadline, current, scanners
            )
            backoff = 2.0
        except AuthRejected:
            raise
        except aiohttp.ClientError, aiohttp.WSMessageTypeError, OSError, TimeoutError:
            # WSMessageTypeError (a non-TEXT frame, e.g. a CLOSE during a
            # proxy hiccup) is a TypeError subclass, NOT an
            # aiohttp.ClientError -- missing it crashed a sibling collector
            # after 10 minutes on 2026-09-16.
            if time.time() >= deadline:
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


async def _poll_loop(
    session: aiohttp.ClientSession,
    base_url: str,
    token: str,
    interval_s: float,
    deadline: float,
    current: CurrentProxy,
    handle: object,
    rows: list[dict[str, object]],
) -> None:
    """Poll position+RSSI on a timer, tag each row with the live current proxy."""
    url = f"{base_url}/api/services/mammotion/export_runtime_state?return_response"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    while time.time() < deadline:
        try:
            async with session.post(
                url,
                json={"entity_id": ENTITY},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status == 200:
                    body = (await resp.json()).get("service_response") or {}
                    position = body.get("position") or {}
                    transport = body.get("transport") or {}
                    x, y = position.get("x"), position.get("y")
                    row = {
                        "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "x": x,
                        "y": y,
                        "proxy": current.name,
                        "ble_rssi": transport.get("ble_rssi"),
                        "connection_label": transport.get("connection_label"),
                    }
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")  # type: ignore[attr-defined]
                    handle.flush()  # type: ignore[attr-defined]
                    emit(
                        f"{row['t_utc']}  pos=({x}, {y})  proxy={row['proxy']:<16} "
                        f"rssi={row['ble_rssi']}"
                    )
                else:
                    emit(f"# poll HTTP {resp.status}")
        except (aiohttp.ClientError, OSError, TimeoutError, ValueError) as exc:
            emit(f"# poll failed ({type(exc).__name__}); retrying next interval")
        await asyncio.sleep(interval_s)


async def main(duration_s: float, interval_s: float) -> None:
    """Run the allocation tracker and position/RSSI poller together for ``duration_s``."""
    env = load_env()
    base_url = env["HA_URL"].rstrip("/")
    ws_url = f"{base_url.replace('https://', 'wss://').replace('http://', 'ws://')}/api/websocket"
    token = env["HA_TOKEN"]

    scanners: dict[str, str] = {}
    current = CurrentProxy()
    rows: list[dict[str, object]] = []
    started = time.time()
    deadline = started + duration_s
    pre_existing = OUT.exists()
    emit(
        f"# tracing while-connected RSSI for {duration_s / 60:.0f} min, "
        f"every {interval_s:.0f}s, "
        f"{'appending to' if pre_existing else 'creating'} {OUT}"
    )
    emit(
        "# run a real click-to-path session in parallel -- this script does not drive the mower"
    )

    async with aiohttp.ClientSession() as session:
        alloc_task = asyncio.create_task(
            _track_allocations(session, ws_url, token, deadline, current, scanners)
        )
        # Give the allocations subscription a moment to receive its first
        # snapshot, so the very first polled row isn't tagged "?" while
        # the websocket handshake is still in flight.
        for _ in range(20):
            if current.name != "?":
                break
            await asyncio.sleep(0.25)
        try:
            with OUT.open("a") as handle:
                await _poll_loop(
                    session,
                    base_url,
                    token,
                    interval_s,
                    deadline,
                    current,
                    handle,
                    rows,
                )
        finally:
            alloc_task.cancel()
            per_proxy: dict[str, int] = {}
            for row in rows:
                per_proxy[str(row["proxy"])] = per_proxy.get(str(row["proxy"]), 0) + 1
            emit("=" * 68)
            emit(
                f"window {(time.time() - started) / 60:.1f} min   samples: {len(rows)}"
            )
            emit(f"per-proxy sample counts: {per_proxy}")
            emit(f"appended to: {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "duration_s",
        nargs="?",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"how long to run, seconds (default {DEFAULT_DURATION_S:.0f})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        dest="interval_s",
        help=f"seconds between polls (default {DEFAULT_INTERVAL_S:.0f})",
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(args.duration_s, args.interval_s))
    except KeyboardInterrupt:
        sys.exit(0)
