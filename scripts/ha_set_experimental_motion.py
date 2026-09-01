#!/usr/bin/env python3
"""Turn the experimental manual-motion gate on or off, and prove which it is.

This exists because arming motion is the one action worth doing deliberately
rather than inline. It is a single narrow entry point, so it can be allowlisted
without granting arbitrary execution, and it always reports the resulting
runtime state instead of trusting the flow's return value -- on 2026-07-31 the
options flow answered ``create_entry`` with an empty ``data`` payload while
having applied the change correctly, so the reply is not evidence.

Usage:
    scripts/ha_set_experimental_motion.py on|off [--yes]
    scripts/ha_set_experimental_motion.py status

Requires HA_URL and HA_TOKEN:  set -a && source .env && set +a

Turning the gate ON only removes the software block. It commands no motion:
a run still needs both operator confirmations, all eleven safety gates, a live
BLE link, and a live VIO feed. Near dusk, check the VIO feed with a dry run --
the cached HA sensor entities lag the real feed by minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

# ⚠️ Do NOT hardcode the config entry id. Deleting and re-adding the integration
# mints a new one, and the stale constant fails the options flow with a bare
# HTTP 500 whose only detail (`UnknownEntry`) is in the HA container log, not in
# the reply. That cost a live session on 2026-09-01, mid-run-preparation, and it
# reads exactly like a BLE fault because arming is what surfaces it.
DOMAIN = "mammotion"
ENTITY_ID = "lawn_mower.back_yard_clip_skywalker"

# The options-flow field is `prefer_ble_over_wifi`, NOT the `prefer_ble` used
# elsewhere in this integration. Submitting the wrong name fails the whole flow
# with a bare HTTP 400 and no field-level detail.
FLOW_FIELDS = (
    ("prefer_ble_over_wifi", True),
    ("movement_use_wifi", False),
    ("mow_path_fetch_enabled", False),
)


def _api(path: str, payload: dict | None = None) -> Any:
    """Call the HA REST API, surfacing the error body rather than a bare code."""
    request = urllib.request.Request(
        os.environ["HA_URL"].rstrip("/") + path,
        headers={
            "Authorization": f"Bearer {os.environ['HA_TOKEN']}",
            "Content-Type": "application/json",
        },
        data=None if payload is None else json.dumps(payload).encode(),
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read() or "{}")
    except urllib.error.HTTPError as err:
        raise SystemExit(
            f"HTTP {err.code} on {path}: {err.read().decode()[:400]}"
        ) from err


def _entry_id() -> str:
    """Resolve the live mammotion config entry id, never a hardcoded constant."""
    entries = _api("/api/config/config_entries/entry")
    matches = [e for e in entries if e.get("domain") == DOMAIN]
    if not matches:
        raise SystemExit(f"No {DOMAIN} config entry found on this Home Assistant.")
    if len(matches) > 1:
        found = ", ".join(f"{e['entry_id']} ({e.get('title')})" for e in matches)
        raise SystemExit(f"Multiple {DOMAIN} entries; disambiguate manually: {found}")
    return str(matches[0]["entry_id"])


def report() -> bool:
    """Print the live motion gate state and return whether it is enabled."""
    response = _api(
        "/api/services/mammotion/export_runtime_state?return_response",
        {"entity_id": ENTITY_ID},
    )
    state = response.get("service_response", {})
    motion = state.get("experimental_motion", {}) or {}
    session = (motion.get("active_session") or {}).get("session_id")
    print(f"  enabled             : {motion.get('enabled')}")
    print(f"  real_motion_allowed : {motion.get('real_motion_allowed')}")
    print(f"  blockers            : {motion.get('blockers')}")
    print(f"  active_session      : {session}")
    print(f"  work_mode           : {state.get('work_mode_label')}")
    return bool(motion.get("enabled"))


def main() -> int:
    """Apply the requested gate state and verify it took effect."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("on", "off", "status"))
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt when arming",
    )
    args = parser.parse_args()

    print("Current state:")
    enabled = report()

    if args.action == "status":
        return 0

    target = args.action == "on"
    if enabled == target:
        print(f"\nAlready {'enabled' if target else 'disabled'}; nothing to do.")
        return 0

    if target and not args.yes and sys.stdin.isatty():
        print("\nArming lets a card Real Go reach the mower. Blades off, e-stop")
        print("released, operator within reach, daylight for VIO.")
        if input("Type ARM to continue: ").strip() != "ARM":
            print("Aborted; gate unchanged.")
            return 1

    flow = _api("/api/config/config_entries/options/flow", {"handler": _entry_id()})

    # Carry every other option through unchanged: the flow replaces the whole
    # options dict, so an omitted field is silently reset to its default.
    #
    # Read those values from the flow's OWN schema defaults, which is what the
    # UI editor does. Do NOT read them from /api/config/config_entries/entry --
    # that endpoint does not expose `options` at all (it returns {} whatever is
    # configured), so preserving from it would quietly reset every other option
    # to a hardcoded default on each toggle.
    current = {
        field["name"]: field.get("default")
        for field in flow.get("data_schema", [])
        if "name" in field
    }
    submission = {field: current.get(field, default) for field, default in FLOW_FIELDS}
    submission["enable_experimental_motion"] = target
    print(f"\nPreserving: {json.dumps({k: submission[k] for k, _ in FLOW_FIELDS})}")
    _api(f"/api/config/config_entries/options/flow/{flow['flow_id']}", submission)

    print("\nState after change:")
    if report() is not target:
        print("\nFAILED: the gate did not reach the requested state.")
        return 1
    print(f"\nOK: experimental motion is now {'ON' if target else 'OFF'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
