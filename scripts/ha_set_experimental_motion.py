#!/usr/bin/env python3
"""Turn the companion's motion gate on or off, and prove which it is.

This exists because arming motion is the one action worth doing deliberately
rather than inline. It is a single narrow entry point, so it can be allowlisted
without granting arbitrary execution, and it always reports the resulting
runtime state instead of trusting the flow's return value -- on 2026-07-31 the
options flow answered ``create_entry`` with an empty ``data`` payload while
having applied the change correctly, so the reply is not evidence.

Since the 2026-10-06 split the gate lives on the ``mammotion_motion`` companion,
not the upstream ``mammotion`` entry, and it is TWO options:
``enable_experimental_motion`` and ``enable_supervised_qualification``. The
stock backend (pymammotion 0.10.7) is unaudited, so real motion needs both.
They move together: ``on`` sets both true, ``off`` sets both false, and any
readback where they disagree -- or disagree with the request -- is a failure.
A failed ``on`` immediately submits ``off`` before exiting, so a half-armed gate
is never left behind.

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
DOMAIN = "mammotion_motion"
ENTITY_ID = "lawn_mower.back_yard_clip_skywalker"

#: The companion options flow's complete schema. Anything else -- a missing
#: field, an extra one, a renamed one -- means this script no longer knows what
#: it is submitting, so it aborts the flow instead of guessing.
GATE_FIELDS = ("enable_experimental_motion", "enable_supervised_qualification")


class GateError(SystemExit):
    """A gate change that could not be made or proved; exits non-zero."""


def _api(path: str, payload: dict | None = None, *, method: str | None = None) -> Any:
    """Call the HA REST API, surfacing the error body rather than a bare code."""
    request = urllib.request.Request(
        os.environ["HA_URL"].rstrip("/") + path,
        headers={
            "Authorization": f"Bearer {os.environ['HA_TOKEN']}",
            "Content-Type": "application/json",
        },
        data=None if payload is None else json.dumps(payload).encode(),
        method=method or ("GET" if payload is None else "POST"),
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read() or "{}")
    except urllib.error.HTTPError as err:
        raise GateError(
            f"HTTP {err.code} on {path}: {err.read().decode()[:400]}"
        ) from err


def _entry_id() -> str:
    """Resolve the one LOADED companion entry, never a hardcoded constant."""
    entries = _api("/api/config/config_entries/entry")
    matches = [e for e in entries if e.get("domain") == DOMAIN]
    if not matches:
        raise GateError(f"No {DOMAIN} config entry found on this Home Assistant.")
    if len(matches) > 1:
        found = ", ".join(f"{e['entry_id']} ({e.get('title')})" for e in matches)
        raise GateError(f"Multiple {DOMAIN} entries; disambiguate manually: {found}")
    if matches[0].get("state") != "loaded":
        raise GateError(
            f"{DOMAIN} entry {matches[0]['entry_id']} is {matches[0].get('state')!r},"
            " not loaded; its options cannot be proved."
        )
    return str(matches[0]["entry_id"])


def read_gate() -> tuple[bool, bool, dict[str, Any]]:
    """Return (experimental, supervised_qualification, motion report) from HA."""
    response = _api(
        f"/api/services/{DOMAIN}/export_runtime_state?return_response",
        {"entity_id": ENTITY_ID},
    )
    motion = response.get("service_response", {}).get("experimental_motion")
    if not isinstance(motion, dict) or not isinstance(motion.get("enabled"), bool):
        raise GateError("export_runtime_state returned no experimental_motion report.")
    qualification = motion.get("supervised_qualification")
    if not isinstance(qualification, bool):
        raise GateError("export_runtime_state returned no supervised_qualification.")
    return motion["enabled"], qualification, motion


def report() -> tuple[bool, bool]:
    """Print the live gate state and return both flags."""
    enabled, qualification, motion = read_gate()
    session = (motion.get("active_session") or {}).get("session_id")
    print(f"  enabled                  : {enabled}")
    print(f"  supervised_qualification : {qualification}")
    print(f"  real_motion_allowed      : {motion.get('real_motion_allowed')}")
    print(f"  blockers                 : {motion.get('blockers')}")
    print(f"  active_session           : {session}")
    return enabled, qualification


def _submit(target: bool) -> None:
    """Run the options flow once, submitting exactly the two gate fields."""
    flow = _api("/api/config/config_entries/options/flow", {"handler": _entry_id()})
    names = {f["name"] for f in flow.get("data_schema", []) if "name" in f}
    if flow.get("type") != "form" or names != set(GATE_FIELDS):
        if flow.get("flow_id"):
            _api(
                f"/api/config/config_entries/options/flow/{flow['flow_id']}",
                method="DELETE",
            )
        raise GateError(
            f"Unexpected options flow (type={flow.get('type')!r}, fields={sorted(names)});"
            " aborted without submitting."
        )
    result = _api(
        f"/api/config/config_entries/options/flow/{flow['flow_id']}",
        dict.fromkeys(GATE_FIELDS, target),
    )
    if result.get("type") != "create_entry":
        raise GateError(f"Options flow did not save: {json.dumps(result)[:300]}")


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
    current = report()

    if args.action == "status":
        return 0

    target = args.action == "on"
    if current == (target, target):
        print(f"\nAlready {'enabled' if target else 'disabled'}; nothing to do.")
        return 0

    if target and not args.yes and sys.stdin.isatty():
        print("\nArming lets a card Real Go reach the mower. Blades off, e-stop")
        print("released, operator within reach, daylight for VIO.")
        if input("Type ARM to continue: ").strip() != "ARM":
            print("Aborted; gate unchanged.")
            return 1

    try:
        _submit(target)
        print("\nState after change:")
        after = report()
        if after != (target, target):
            raise GateError(f"Gate readback {after} != requested {(target, target)}.")
    except SystemExit:
        if target:
            # Fail closed: never leave a half-armed or unproved gate behind.
            print("\nARM FAILED; submitting OFF before exiting.", file=sys.stderr)
            try:
                _submit(False)
                print("Disarm readback:", report(), file=sys.stderr)
            except SystemExit as err:
                print(f"DISARM ALSO FAILED: {err}", file=sys.stderr)
        raise
    print(f"\nOK: motion gate is now {'ON' if target else 'OFF'} (both flags).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
