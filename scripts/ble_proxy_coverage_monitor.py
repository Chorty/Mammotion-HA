#!/usr/bin/env python3
"""Record per-proxy BLE RSSI against mower position, for a per-proxy coverage map.

Extends ``ble_advert_monitor.py``'s ``bluetooth/subscribe_advertisements``
stream (real-time, scanner-tagged, immune to Docker log rotation -- unlike
mining ``habluetooth``'s ``Found N connection path(s)`` log lines, which
turned out to have almost nothing left after 2026-09-15's beta105 restart)
with a periodic position poll, so every RSSI reading gets tagged with where
the mower was when it was heard.

🚨 **The mower only advertises while DISCONNECTED, roughly once per ~10 min**
(documented since 2026-07-25, reconfirmed 2026-09-14/15's proxy-adjacency
test). This is a hard ceiling on data density: while a real motion/BLE test
session holds a live connection, this collector gets nothing from those
minutes. It fills in during idle/reconnect windows -- parked, docked, between
sessions -- which is exactly when a proxy has to win a fresh connection
anyway, so the data it does collect answers the question that matters.

Position comes from ``export_runtime_state``'s ``position.x``/``position.y``
-- already in the executor's ``mower_map_xy`` frame, the same frame every
target/corridor/keep-out check in this repo uses. No lat/lon transform
needed (unlike the older ``ble_coverage_map.py``, which derived one because
its source, ``device_tracker``, only has lat/lon).

**Output is append-only and never overwritten** (``ble_proxy_coverage_log.jsonl``),
so this is meant to be run repeatedly -- or left running in the background --
across many sessions and days, accumulating real per-proxy coverage over time
rather than in one sitting.

Read-only -- sends no commands to the mower, only a read (`export_runtime_state`)
and two Bluetooth subscriptions.

Usage:  .venv/bin/python scripts/ble_proxy_coverage_monitor.py [seconds] [--position-interval N]
        (seconds defaults to a long 8h run; Ctrl+C stops cleanly and still
        flushes the summary)
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
OUT = Path(__file__).with_name("ble_proxy_coverage_log.jsonl")
REPO_ENV = Path(__file__).resolve().parent.parent / ".env"

#: Position polls this often; the mower's own report cadence tops out at
#: ~1 Hz anyway (project-wide measured ceiling), and this collector cares
#: about "which few square metres", not sub-second tracking.
DEFAULT_POSITION_INTERVAL_S = 15.0
DEFAULT_DURATION_S = 8 * 3600.0


def load_env() -> dict[str, str]:
    """Return ``HA_URL``/``HA_TOKEN`` (and anything else present) from the repo ``.env``."""
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


class LatestPosition:
    """The most recently polled mower position, shared between coroutines.

    ``age_s`` lets a consumer judge staleness rather than trusting silently:
    a hit paired with a position last confirmed 10 minutes ago (the poller
    fell behind, or a request errored) is worth flagging, not hiding.
    """

    def __init__(self) -> None:
        """Start with no position known yet."""
        self.x: float | None = None
        self.y: float | None = None
        self.at: float = 0.0

    def snapshot(self) -> tuple[float | None, float | None, float]:
        """Return ``(x, y, age_seconds)``; age is ``inf`` if never updated."""
        return self.x, self.y, time.time() - self.at if self.at else float("inf")

    def update(self, x: float, y: float) -> None:
        """Record a freshly polled position."""
        self.x, self.y, self.at = x, y, time.time()


async def _poll_position(
    session: aiohttp.ClientSession,
    base_url: str,
    token: str,
    interval_s: float,
    latest: LatestPosition,
    stop_at: float,
) -> None:
    """Poll ``export_runtime_state`` on a timer until ``stop_at``.

    Never raises out of the loop on a single bad poll -- a transient HTTP
    error here must not take down the advertisement stream, which is the
    half of this tool that cannot be recovered by trying again a second
    later (an advertisement missed is gone; a position poll missed just
    means the next hit pairs with a slightly older position).
    """
    url = f"{base_url}/api/services/mammotion/export_runtime_state?return_response"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    while time.time() < stop_at:
        try:
            async with session.post(
                url,
                json={"entity_id": ENTITY},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status == 200:
                    body = await resp.json()
                    position = (body.get("service_response") or {}).get(
                        "position"
                    ) or {}
                    x, y = position.get("x"), position.get("y")
                    if isinstance(x, int | float) and isinstance(y, int | float):
                        latest.update(float(x), float(y))
                else:
                    emit(f"# position poll HTTP {resp.status}")
        except (aiohttp.ClientError, OSError, TimeoutError, ValueError) as exc:
            emit(
                f"# position poll failed ({type(exc).__name__}); keeping last known position"
            )
        await asyncio.sleep(interval_s)


async def _stream_once(
    session: aiohttp.ClientSession,
    ws_url: str,
    token: str,
    deadline: float,
    scanners: dict[str, str],
    hits: list[dict[str, object]],
    handle: object,
    counters: dict[str, int],
    latest: LatestPosition,
) -> None:
    """Consume one websocket session until ``deadline`` or the socket dies.

    Mirrors ``ble_advert_monitor.py``'s structure: the all-device control
    count accumulates into ``counters`` (not a return value) so a mid-session
    reconnect cannot discard it, and every mower hit is written and flushed
    immediately so a crash loses at most the in-flight line.
    """
    async with session.ws_connect(ws_url, max_msg_size=0) as ws:
        await ws.receive_json()
        await ws.send_json({"type": "auth", "access_token": token})
        auth = await ws.receive_json()
        if auth["type"] != "auth_ok":
            raise AuthRejected(str(auth))

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
                x, y, position_age_s = latest.snapshot()
                row = {
                    "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                    "rssi": entry.get("rssi"),
                    "proxy": scanners.get(entry.get("source", ""), "?"),
                    "connectable": entry.get("connectable"),
                    "x": x,
                    "y": y,
                    "position_age_s": round(position_age_s, 1)
                    if position_age_s != float("inf")
                    else None,
                }
                hits.append(row)
                handle.write(json.dumps(row) + "\n")  # type: ignore[attr-defined]
                handle.flush()  # type: ignore[attr-defined]
                emit(
                    f"{row['t_utc']}  rssi={row['rssi']:>4}  via {row['proxy']:<16} "
                    f"pos=({x}, {y}) age={row['position_age_s']}s"
                )


async def main(duration_s: float, position_interval_s: float) -> None:
    """Stream mower advertisements paired with position for ``duration_s`` seconds.

    Appends to ``OUT`` -- never truncates -- so repeated runs across sessions
    and days accumulate one growing dataset. Reconnects the websocket on a
    drop; a position-poll failure never takes down the advertisement stream.
    """
    env = load_env()
    base_url = env["HA_URL"].rstrip("/")
    ws_url = f"{base_url.replace('https://', 'wss://').replace('http://', 'ws://')}/api/websocket"
    token = env["HA_TOKEN"]

    scanners: dict[str, str] = {}
    hits: list[dict[str, object]] = []
    counters = {"total_adverts": 0}
    latest = LatestPosition()
    started = time.time()
    deadline = started + duration_s
    backoff = 2.0
    pre_existing = OUT.exists()
    emit(
        f"# monitoring {MOWER_MAC} for {duration_s / 3600:.1f}h, "
        f"position every {position_interval_s:.0f}s, "
        f"{'appending to' if pre_existing else 'creating'} {OUT}"
    )

    async with aiohttp.ClientSession() as session:
        position_task = asyncio.create_task(
            _poll_position(
                session, base_url, token, position_interval_s, latest, deadline
            )
        )
        try:
            with OUT.open("a") as handle:
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
                            latest,
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
                        backoff = min(backoff * 2, 30.0)
        finally:
            position_task.cancel()
            summarise(hits, started, time.time(), scanners, counters["total_adverts"])


def summarise(
    hits: list[dict[str, object]],
    started: float,
    ended: float,
    scanners: dict[str, str],
    total_adverts: int,
) -> None:
    """Print per-proxy hit counts, RSSI spread, and the control count for this run."""
    emit("=" * 68)
    emit(f"window {(ended - started) / 60:.1f} min   mower advertisements: {len(hits)}")
    emit(f"CONTROL - advertisements from all devices: {total_adverts}")
    emit(f"scanners registered: {len(scanners)} -> {sorted(set(scanners.values()))}")
    emit(f"appended to: {OUT}")

    if not hits:
        emit("NO ADVERTISEMENTS FROM THE MOWER THIS RUN.")
        if total_adverts == 0:
            emit("...but the CONTROL is also zero, so the stream is not emitting.")
            emit("This result is INVALID -- fix the subscription before concluding.")
        else:
            emit(
                "The stream is alive; the mower just did not advertise this run "
                "(e.g. it stayed connected the whole time)."
            )
        return

    no_position = sum(1 for h in hits if h["x"] is None)
    if no_position:
        emit(
            f"⚠️ {no_position}/{len(hits)} hits have NO position (poll never succeeded yet)"
        )

    per_proxy: dict[str, list[int]] = {}
    for hit in hits:
        rssi = hit["rssi"]
        if isinstance(rssi, int | float):
            per_proxy.setdefault(str(hit["proxy"]), []).append(int(rssi))
    emit("this run, per-proxy:")
    for name, rssis in sorted(per_proxy.items(), key=lambda kv: -len(kv[1])):
        median = sorted(rssis)[len(rssis) // 2]
        emit(
            f"  {name:<20} n={len(rssis):<4} "
            f"rssi min/med/max={min(rssis)}/{median}/{max(rssis)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "duration_s",
        nargs="?",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"how long to run, seconds (default {DEFAULT_DURATION_S / 3600:.0f}h)",
    )
    parser.add_argument(
        "--position-interval",
        type=float,
        default=DEFAULT_POSITION_INTERVAL_S,
        dest="position_interval_s",
        help=f"seconds between position polls (default {DEFAULT_POSITION_INTERVAL_S:.0f})",
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(args.duration_s, args.position_interval_s))
    except KeyboardInterrupt:
        sys.exit(0)
