#!/usr/bin/env python3
"""Read-only pre-flight for the supervised LUBA acceptance gates 2-4.

Answers one question -- can a real run start right now, and if not, what is
missing -- so a daylight window is not spent discovering blockers one service
call at a time.

Everything here is read-only: it calls ``export_runtime_state`` and reads HA
states. No command reaches the mower.

Usage:
    set -a && source .env && set +a
    .venv/bin/python scripts/mammotion_preflight_gates.py
"""  # noqa: INP001

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mammotion_ha_helpers import load_dotenv, post_service  # noqa: E402

from custom_components.mammotion.services import (  # noqa: E402
    _blade_rpm_stale_verdict,
)

ENTITY = "lawn_mower.back_yard_clip_skywalker"
PREFIX = "sensor.back_yard_clip_skywalker_"
BPREFIX = "binary_sensor.back_yard_clip_skywalker_"

#: Samples the latched-RPM discriminator needs, and the gap between them. The
#: reporting coordinator ticks every 300s, so the position only advances across
#: a tick -- shorter gaps cannot prove the feed is live.
RPM_SAMPLES = 3
RPM_SAMPLE_GAP_SECONDS = 330

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"

_BLE_BLOCKER_PREFIXES = ("ble_", "command_queue_")
_BLE_BLOCKER_NAMES = {
    "device_handle_unavailable",
    "get_transport_unavailable",
    "exclusive_saga_active",
    "no_ble_send_observed",
}


def _states(url: str, tok: str) -> dict[str, Any]:
    """Return all HA states keyed by entity_id."""
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/states", headers={"Authorization": f"Bearer {tok}"}
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        return {s["entity_id"]: s for s in json.load(resp)}


def _row(name: str, verdict: str, detail: str) -> tuple[str, str, str]:
    """Return one printable check row."""
    return (name, verdict, detail)


def _ble_motion_ready(
    live_state: Any,
    active_transport: Any,
    blockers: list[str],
) -> bool:
    """Return whether both the entity and fresh runtime agree BLE is usable."""
    ble_blockers = [
        blocker
        for blocker in blockers
        if blocker.startswith(_BLE_BLOCKER_PREFIXES) or blocker in _BLE_BLOCKER_NAMES
    ]
    return live_state == "on" and active_transport == "ble" and not ble_blockers


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    """Run every pre-flight check and print a verdict table."""
    load_dotenv(Path(".env"))
    url, tok = os.environ.get("HA_URL", ""), os.environ.get("HA_TOKEN", "")
    if not url or not tok:
        print("HA_URL/HA_TOKEN not set (run: set -a && source .env && set +a)")
        return 2
    deep = "--quick" not in sys.argv

    state = _states(url, tok)
    runtime = post_service(
        url, tok, "mammotion", "export_runtime_state", {"entity_id": ENTITY}, 60
    )
    em = runtime.get("experimental_motion") or {}
    pos = runtime.get("position") or {}
    blade = runtime.get("blade") or {}
    transport = runtime.get("transport") or {}
    blockers = [str(blocker) for blocker in em.get("blockers") or []]
    rows: list[tuple[str, str, str]] = []

    rows.append(
        _row(
            "backend verified",
            PASS if em.get("backend_verified") else FAIL,
            f"{em.get('installed_pymammotion_version')} | "
            f"missing={((em.get('backend_capabilities') or {}).get('missing'))}",
        )
    )

    live = (state.get(BPREFIX + "ble_link_live") or {}).get("state")
    active_transport = runtime.get("active_transport")
    ble_ready = _ble_motion_ready(live, active_transport, blockers)
    rows.append(
        _row(
            "BLE link live",
            PASS if ble_ready else FAIL,
            f"entity={live} transport={active_transport} "
            f"rssi={transport.get('ble_rssi')}",
        )
    )

    zone, ptype = pos.get("zone_hash"), pos.get("pos_type_label")
    zone_ok = zone not in (None, 0, "0") and ptype in {
        "AREA_INSIDE",
        "TURN_AREA_INSIDE",
        "CHANNEL_AREA_OVERLAP",
    }
    rows.append(
        _row(
            "in a mapped zone",
            PASS if zone_ok else FAIL,
            f"zone_hash={zone} pos_type={ptype} area={pos.get('area_name')}",
        )
    )

    rows.append(
        _row(
            "position valid",
            PASS if pos.get("valid_for_motion") else FAIL,
            f"rtk={pos.get('rtk_status_label')} x={pos.get('x')} y={pos.get('y')}",
        )
    )

    brightness = (state.get(PREFIX + "camera_brightness") or {}).get("state")
    try:
        feats = int((state.get(PREFIX + "vio_tracked_features") or {}).get("state", 0))
    except TypeError, ValueError:
        feats = 0
    vio_ok = str(brightness).lower() != "dark" and feats >= 5
    rows.append(
        _row(
            "VIO usable (turns only)",
            PASS if vio_ok else WARN,
            f"brightness={brightness} tracked_features={feats} (>=5 needed; ~80 is healthy)",
        )
    )

    mode = runtime.get("work_mode_label")
    rows.append(
        _row(
            "mower ready/paused",
            PASS if mode in {"MODE_READY", "MODE_PAUSE"} else FAIL,
            f"work_mode={mode} charge={runtime.get('charge_state_label')}",
        )
    )

    rows.append(
        _row(
            "experimental motion",
            PASS if em.get("enabled") else FAIL,
            "enabled"
            if em.get("enabled")
            else "OFF -- enable for the session, then disable",
        )
    )

    # The latched-RPM discriminator: only meaningful if RPM is nonzero at all.
    rpm = blade.get("current_cutter_rpm")
    if rpm in (None, 0, "0"):
        rows.append(_row("blade RPM", PASS, "zero -- no latch to discount"))
    elif not deep:
        rows.append(
            _row(
                "blade RPM latch",
                WARN,
                f"rpm={rpm} looks_latched={blade.get('blade_rpm_looks_latched')} "
                "(run without --quick to test the discriminator)",
            )
        )
    else:
        samples = []
        for i in range(RPM_SAMPLES):
            if i:
                time.sleep(RPM_SAMPLE_GAP_SECONDS)
            d = post_service(
                url, tok, "mammotion", "export_runtime_state", {"entity_id": ENTITY}, 60
            )
            samples.append(
                {"blade": d.get("blade") or {}, "position": d.get("position") or {}}
            )
            print(
                f"  ... rpm sample {i + 1}/{RPM_SAMPLES}: rpm="
                f"{samples[-1]['blade'].get('current_cutter_rpm')} "
                f"pos=({samples[-1]['position'].get('x')},{samples[-1]['position'].get('y')})",
                flush=True,
            )
        verdict = _blade_rpm_stale_verdict(samples)
        rows.append(
            _row(
                "blade RPM latch",
                PASS if verdict["stale_register"] else FAIL,
                f"rpm={rpm} stale_register={verdict['stale_register']} "
                f"reasons={verdict['reasons']}",
            )
        )

    rows.append(
        _row(
            "standing gate",
            PASS if not blockers else FAIL,
            f"blockers={blockers or 'none'}",
        )
    )

    width = max(len(r[0]) for r in rows)
    print("\n=== gates 2-4 pre-flight ===")
    for name, verdict, detail in rows:
        print(f"  [{verdict:<4}] {name:<{width}}  {detail}")
    fails = [r[0] for r in rows if r[1] == FAIL]
    print("\nVERDICT:", "READY" if not fails else f"BLOCKED on: {', '.join(fails)}")
    print(
        "Reminder: debug loggers (bleak_esphome, habluetooth) do NOT survive an HA "
        "restart -- re-enable via logger.set_level before measuring."
    )
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
