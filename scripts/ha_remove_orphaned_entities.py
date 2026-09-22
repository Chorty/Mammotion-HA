#!/usr/bin/env python3
"""Remove orphaned entity-registry rows left by code that no longer exists.

Found during the beta114 deploy verification (2026-09-22): 9 Mammotion
entities read `unavailable`. 1 is unavailable by design
(`button.clip_skywalker_refresh_active_job_settings` requires an active route
job). The other 8 are registry rows with no backing platform code anywhere in
the deployed tree -- confirmed by grepping the deployed `sensor.py`/`button.py`
for each entity's `key`, and tracing each key's origin in git history:

- 6 keys (`soc_temperature`, `soc_uptime`, `mcu_uptime`, `usb_disconnect_count`,
  `process_restart_count`, `soc_coredump_count`) trace to the unmerged branch
  `feat/device-health-diagnostics` (tip `544515d4`), deployed to this host for
  its own testing session and never merged to `main`.
- 1 key (`active_job_revision`) traces to `feat/active-job-revision`: added by
  `9d025c13`, then removed in the SAME session by `4ee7608c` once `WorkData`
  turned out not to carry a job ID or revision (see `docs/TODO.md`). Deployed,
  then immediately walked back in code -- the registry row was never cleaned
  up on the host.
- 1 (`sensor.back_yard_luba_vsplv397_task_area_path`) has NO git history
  anywhere in this repo, reachable or dangling (checked via
  `git log --all -S` and `git fsck --unreachable`) -- it was deployed from an
  uncommitted local edit that was never version-controlled at all.

None of the 8 back real, currently-shipped functionality. Removing the
registry row loses no data (it is HA's device/entity metadata, not the
mower's own state) and does not require any code change -- if one of these
features is ever properly merged and deployed, HA re-registers it fresh on
first report.

Uses the `config/entity_registry/remove` websocket command (there is no REST
equivalent), matching scripts/ha_set_card_resource.py's connection pattern.

Usage:
    scripts/ha_remove_orphaned_entities.py                 # dry run (default)
    scripts/ha_remove_orphaned_entities.py --apply          # remove them
    scripts/ha_remove_orphaned_entities.py --apply entity_id [entity_id ...]

Requires HA_URL and HA_TOKEN:  set -a && source .env && set +a
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import aiohttp

# The 8 confirmed orphans (see module docstring). Deliberately an explicit
# list, not "every unavailable entity" -- a future genuinely-broken entity
# must never be silently swept up by re-running this script unmodified.
DEFAULT_ORPHANS: tuple[str, ...] = (
    "sensor.clip_skywalker_soc_temperature",
    "sensor.clip_skywalker_soc_uptime",
    "sensor.clip_skywalker_mcu_uptime",
    "sensor.clip_skywalker_usb_disconnect_count",
    "sensor.clip_skywalker_process_restart_count",
    "sensor.clip_skywalker_soc_core_dump_count",
    "sensor.clip_skywalker_active_job_revision",
    "sensor.back_yard_luba_vsplv397_task_area_path",
)


async def _rpc(ws: aiohttp.ClientWebSocketResponse, msg_id: int, payload: dict) -> dict:
    """Send one websocket command and wait for the reply with a matching id."""
    await ws.send_json({"id": msg_id, **payload})
    while True:
        message = await ws.receive_json()
        if message.get("id") == msg_id:
            return message


async def _authenticate(ws: aiohttp.ClientWebSocketResponse) -> bool:
    assert (await ws.receive_json())["type"] == "auth_required"
    await ws.send_json({"type": "auth", "access_token": os.environ["HA_TOKEN"]})
    if (await ws.receive_json())["type"] != "auth_ok":
        print("authentication failed")
        return False
    return True


async def run(entity_ids: tuple[str, ...], *, apply: bool) -> int:
    """List, and optionally remove, the given entity-registry rows."""
    url = os.environ["HA_URL"].rstrip("/") + "/api/websocket"
    async with (
        aiohttp.ClientSession() as session,
        session.ws_connect(url, max_msg_size=0) as ws,
    ):
        if not await _authenticate(ws):
            return 1

        listing = await _rpc(ws, 1, {"type": "config/entity_registry/list"})
        if not listing.get("success"):
            print("could not list entity registry:", json.dumps(listing)[:300])
            return 1
        by_id = {e["entity_id"]: e for e in listing["result"]}

        missing = [e for e in entity_ids if e not in by_id]
        if missing:
            print("not in registry (already gone, or a typo):", missing)

        targets = [e for e in entity_ids if e in by_id]
        for entity_id in targets:
            entry = by_id[entity_id]
            print(
                f"{entity_id}  platform={entry.get('platform')}  unique_id={entry.get('unique_id')}"
            )

        if not apply:
            print(
                f"\nDRY RUN — would remove {len(targets)} entities. Re-run with --apply."
            )
            return 0

        msg_id = 2
        failures: list[str] = []
        for entity_id in targets:
            result = await _rpc(
                ws,
                msg_id,
                {"type": "config/entity_registry/remove", "entity_id": entity_id},
            )
            msg_id += 1
            if result.get("success"):
                print(f"removed: {entity_id}")
            else:
                print(f"FAILED to remove {entity_id}: {json.dumps(result)[:300]}")
                failures.append(entity_id)

        verify = await _rpc(ws, msg_id, {"type": "config/entity_registry/list"})
        still_present = {e["entity_id"] for e in verify.get("result", [])} & set(
            targets
        )
        if still_present:
            print("still present after removal attempt:", sorted(still_present))
            return 1
        print(
            f"verified: {len(targets) - len(failures)} entities removed from the registry"
        )
        return 1 if failures else 0


def main() -> int:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "entity_ids",
        nargs="*",
        help="Entity IDs to remove; defaults to the 8 confirmed orphans if omitted.",
    )
    parser.add_argument(
        "--apply", action="store_true", help="actually remove (default: dry run)"
    )
    args = parser.parse_args()
    entity_ids = tuple(args.entity_ids) if args.entity_ids else DEFAULT_ORPHANS
    return asyncio.run(run(entity_ids, apply=args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
