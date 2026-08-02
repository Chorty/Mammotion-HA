#!/usr/bin/env python3
"""Point the Lovelace card resource at a new cache-busting version.

Deploying the card file is not enough. Home Assistant serves integration static
files with a 31-day cache header, and browsers key on the resource URL's query
string -- so if the registered `?v=` does not change, every browser keeps the
PREVIOUS card while every server-side check reports the new one. That happened
on 2026-07-31: the registered key was `?v=12`, an arbitrary number that does not
track the version, so a correct beta12 deploy would have kept serving beta11's
Real Go defaults.

Uses the supported `lovelace/resources` websocket API rather than editing
`/config/.storage/lovelace_resources`, which HA holds in memory and would
overwrite.

Usage:
    scripts/ha_set_card_resource.py                      # show current
    scripts/ha_set_card_resource.py 0.6.4-beta13         # dry run
    scripts/ha_set_card_resource.py 0.6.4-beta13 --apply

Requires HA_URL and HA_TOKEN:  set -a && source .env && set +a
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import aiohttp

CARD_FILENAME = "mammotion-custom-path-card.js"


async def _rpc(ws: aiohttp.ClientWebSocketResponse, msg_id: int, payload: dict) -> dict:
    """Send one websocket command and wait for the reply with a matching id."""
    await ws.send_json({"id": msg_id, **payload})
    while True:
        message = await ws.receive_json()
        if message.get("id") == msg_id:
            return message


async def _card_resources(ws: aiohttp.ClientWebSocketResponse) -> list[dict] | None:
    """Authenticate, then return the resources referencing the card, or None."""
    assert (await ws.receive_json())["type"] == "auth_required"
    await ws.send_json({"type": "auth", "access_token": os.environ["HA_TOKEN"]})
    if (await ws.receive_json())["type"] != "auth_ok":
        print("authentication failed")
        return None

    listing = await _rpc(ws, 1, {"type": "lovelace/resources"})
    if not listing.get("success"):
        print("could not list resources:", json.dumps(listing)[:300])
        return None

    matches = [r for r in listing["result"] if CARD_FILENAME in r.get("url", "")]
    if not matches:
        print(f"No registered resource references {CARD_FILENAME}.")
        print("Add it in Settings > Dashboards > Resources first.")
        return None
    return matches


async def _retarget(
    ws: aiohttp.ClientWebSocketResponse,
    resource: dict,
    version: str,
    *,
    apply: bool,
) -> bool:
    """Repoint one resource's cache key, preserving its path. True if fine."""
    # Keep the registered path as-is. The live dashboard may reference the
    # /hacsfiles/ copy rather than the integration-served /mammotion/ one, and
    # silently repointing it is a surprise nobody asked for.
    new_url = f"{resource['url'].split('?', 1)[0]}?v={version}"
    if new_url == resource["url"]:
        print(f"already at {new_url}")
        return True
    if not apply:
        print(f"DRY RUN would set: {new_url}")
        return True
    updated = await _rpc(
        ws,
        2,
        {
            "type": "lovelace/resources/update",
            "resource_id": resource["id"],
            "url": new_url,
            "res_type": resource["type"],
        },
    )
    if not updated.get("success"):
        print("update failed:", json.dumps(updated)[:300])
        return False
    print(f"updated: {new_url}")
    return True


async def run(version: str | None, apply: bool) -> int:
    """Show, and optionally update, the registered card resource URL."""
    url = os.environ["HA_URL"].rstrip("/") + "/api/websocket"
    async with aiohttp.ClientSession() as session, session.ws_connect(
        url, max_msg_size=0
    ) as ws:
        matches = await _card_resources(ws)
        if matches is None:
            return 1
        for resource in matches:
            print(f"current: {resource['url']}   (id {resource['id']})")
        if version is None:
            return 0

        for resource in matches:
            if not await _retarget(ws, resource, version, apply=apply):
                return 1

        if apply:
            after = await _rpc(ws, 3, {"type": "lovelace/resources"})
            for resource in after["result"]:
                if CARD_FILENAME in resource.get("url", ""):
                    print(f"verified: {resource['url']}")
    return 0


def main() -> int:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", help="e.g. 0.6.4-beta13")
    parser.add_argument("--apply", action="store_true", help="write the change")
    args = parser.parse_args()
    return asyncio.run(run(args.version, args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
