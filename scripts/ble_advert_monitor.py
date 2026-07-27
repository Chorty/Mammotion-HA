#!/usr/bin/env python3
"""Record every BLE advertisement Home Assistant hears from the mower.

This is the liveness signal the project lacked. ``sensor.<mower>_ble_rssi`` is
self-reported by the mower, so it holds a plausible value long after the radio
goes quiet (live 2026-07-25: it read -64 while nothing had heard an
advertisement for ten minutes). An advertisement, by contrast, is proof the
radio was on air at that instant, observed by HA's own scanners.

Subscribes to ``bluetooth/subscribe_advertisements`` over the HA websocket API,
logs address/RSSI/scanner per advertisement, and summarises silent gaps plus
per-scanner coverage.

The summary always reports a CONTROL count of advertisements from *all* devices:
a zero for the mower proves nothing unless the stream is shown to be emitting.

Read-only -- sends no commands to the mower.

Usage:  .venv/bin/python scripts/ble_advert_monitor.py [seconds]
"""  # noqa: INP001

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

MOWER_MAC = "A8:B5:8E:2C:52:40"
OUT = Path(__file__).with_name("advert_log.jsonl")
REPO_ENV = Path(__file__).resolve().parent.parent / ".env"


def load_env() -> tuple[str, str]:
    """Return ``(websocket_url, token)`` from the repo ``.env``."""
    values: dict[str, str] = {}
    for line in REPO_ENV.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    url = values["HA_URL"].rstrip("/")
    ws = url.replace("https://", "wss://").replace("http://", "ws://")
    return f"{ws}/api/websocket", values["HA_TOKEN"]


def emit(message: str) -> None:
    """Print one line of progress output."""
    print(message, flush=True)  # noqa: T201


class AuthRejected(Exception):
    """Home Assistant rejected the access token. Terminal, never retried."""


async def _stream_once(
    session: aiohttp.ClientSession,
    ws_url: str,
    token: str,
    deadline: float,
    scanners: dict[str, str],
    hits: list[dict[str, object]],
    handle: object,
    counters: dict[str, int],
) -> None:
    """Consume one websocket session until ``deadline`` or the socket dies.

    The all-device control count accumulates into ``counters`` rather than a
    return value, so a mid-session disconnect cannot discard it. That count is
    the only thing distinguishing "the mower was silent" from "the stream was
    not emitting", and losing it would invalidate the whole measurement.

    Raises ``AuthRejected`` on a bad token (terminal) and lets connection
    errors propagate so the caller can reconnect.
    """
    async with session.ws_connect(ws_url, max_msg_size=0) as ws:
        await ws.receive_json()
        await ws.send_json({"type": "auth", "access_token": token})
        auth = await ws.receive_json()
        if auth["type"] != "auth_ok":
            # Terminal: retrying a rejected token just spins a tight
            # connect/auth storm against HA for the rest of the run.
            raise AuthRejected(str(auth))

        # Scanner registry: maps the `source` MAC on each advert to a name.
        await ws.send_json({"id": 1, "type": "bluetooth/subscribe_scanner_details"})
        await ws.send_json({"id": 2, "type": "bluetooth/subscribe_advertisements"})

        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(
                    ws.receive_json(), timeout=max(5.0, deadline - time.time())
                )
            except TimeoutError:
                continue
            if msg.get("type") != "event":
                continue
            event = msg.get("event") or {}

            if msg.get("id") == 1:
                for entry in event.get("add", []) or []:
                    scanners[entry.get("source", "")] = entry.get("name", "?")
                continue

            for entry in event.get("add", []) or []:
                counters["total_adverts"] += 1
                if (entry.get("address") or "").upper() != MOWER_MAC:
                    continue
                now = time.time()
                row = {
                    "t": round(now, 2),
                    "iso": time.strftime("%H:%M:%S", time.localtime(now)),
                    "rssi": entry.get("rssi"),
                    "scanner": scanners.get(entry.get("source", ""), "?"),
                    "connectable": entry.get("connectable"),
                }
                hits.append(row)
                handle.write(json.dumps(row) + "\n")  # type: ignore[attr-defined]
                handle.flush()  # type: ignore[attr-defined]
                emit(
                    f"{row['iso']}  rssi={row['rssi']:>4}  "
                    f"connectable={row['connectable']}  via {row['scanner']}"
                )


async def main(duration_s: float) -> None:
    """Stream advertisements for ``duration_s`` seconds, then summarise.

    Reconnects if the websocket drops. A long run previously died partway
    through on an aiohttp heartbeat timeout and lost its summary entirely, which
    for a measurement whose whole point is "how long was it silent" is the one
    failure mode that matters.
    """
    ws_url, token = load_env()
    scanners: dict[str, str] = {}
    hits: list[dict[str, object]] = []
    # Shared across reconnects: a mid-session drop must not discard the control
    # count, or the summary can wrongly claim the stream never emitted.
    counters = {"total_adverts": 0}
    started = time.time()
    deadline = started + duration_s
    backoff = 2.0
    emit(f"# monitoring {MOWER_MAC} for {duration_s / 60:.0f} min")

    try:
        async with aiohttp.ClientSession() as session:
            with OUT.open("w") as handle:
                while time.time() < deadline:
                    try:
                        await _stream_once(
                            session,
                            ws_url,
                            token,
                            deadline,
                            scanners,
                            hits,
                            handle,
                            counters,
                        )
                        backoff = 2.0
                    except AuthRejected as exc:
                        emit(f"AUTH FAILED (terminal, not retrying): {exc}")
                        raise
                    except (aiohttp.ClientError, OSError, TimeoutError) as exc:
                        if time.time() >= deadline:
                            break
                        emit(
                            f"# websocket dropped ({type(exc).__name__}) - "
                            f"reconnecting in {backoff:.0f}s"
                        )
                        await asyncio.sleep(backoff)
                        # Bounded exponential backoff so a persistently broken
                        # endpoint cannot become a tight reconnect storm.
                        backoff = min(backoff * 2, 30.0)
    finally:
        # The summary is the deliverable; never lose it to a transport error.
        summarise(hits, started, time.time(), scanners, counters["total_adverts"])


def summarise(
    hits: list[dict[str, object]],
    started: float,
    ended: float,
    scanners: dict[str, str],
    total_adverts: int,
) -> None:
    """Print gap analysis, per-scanner coverage, and the control count."""
    emit("=" * 68)
    emit(f"window {(ended - started) / 60:.1f} min   mower advertisements: {len(hits)}")
    emit(f"CONTROL - advertisements from all devices: {total_adverts}")
    emit(f"scanners registered: {len(scanners)} -> {sorted(set(scanners.values()))}")

    if not hits:
        emit("NO ADVERTISEMENTS FROM THE MOWER IN THE WHOLE WINDOW.")
        if total_adverts == 0:
            emit("...but the CONTROL is also zero, so the stream is not emitting.")
            emit("This result is INVALID -- fix the subscription before concluding.")
        return

    times = [float(hit["t"]) for hit in hits]
    gaps = [(b - a, a) for a, b in zip(times, times[1:], strict=False) if b - a > 5]
    emit(
        f"silent before first advert: {times[0] - started:.0f}s   "
        f"after last: {ended - times[-1]:.0f}s"
    )

    per_scanner: dict[str, list[int]] = {}
    for hit in hits:
        per_scanner.setdefault(str(hit["scanner"]), []).append(int(hit["rssi"]))
    emit("per-scanner:")
    for name, rssis in sorted(per_scanner.items(), key=lambda kv: -len(kv[1])):
        median = sorted(rssis)[len(rssis) // 2]
        emit(
            f"  {name:<30} n={len(rssis):<4} "
            f"rssi min/med/max={min(rssis)}/{median}/{max(rssis)}"
        )

    emit(f"gaps > 5s: {len(gaps)}")
    for gap, at in sorted(gaps, reverse=True)[:15]:
        stamp = time.strftime("%H:%M:%S", time.localtime(at))
        emit(f"  {gap:7.1f}s silent from {stamp}")


if __name__ == "__main__":
    asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 1800.0))
