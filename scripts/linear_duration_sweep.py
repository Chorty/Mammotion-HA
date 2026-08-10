#!/usr/bin/env python3
"""Linear duration sweep, measured by RTK (works in darkness, unlike VIO).

Alternates backward/forward so net displacement stays near zero. Aborts the
whole sweep on any anomaly: blockers, a failed command, a silent BLE transport,
or the mower leaving its area / getting within 1 m of the boundary.
"""

from __future__ import annotations

import ast
import json
import math
import os
import pathlib
import subprocess
import time
import urllib.request

SP = pathlib.Path(__file__).parent
REPO = pathlib.Path("/Users/mattjoslin/Documents/Git Projects/Mammotion-HA")
HA = os.environ["HA_URL"]
TOKEN = os.environ["HA_TOKEN"]
ENTITY = "lawn_mower.back_yard_clip_skywalker"

DURATIONS = [1600, 1000, 700, 500, 400, 300]
MIN_CLEARANCE_M = 1.0

# Real containment implementation, lifted from the integration.
_src = (REPO / "custom_components/mammotion/services.py").read_text()
_tree = ast.parse(_src)
_mod = ast.Module(
    body=[
        n
        for n in _tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_point_in_polygon", "_point_on_segment"}
    ],
    type_ignores=[],
)
_ns: dict = {}
exec(compile(_mod, "s", "exec"), _ns)  # noqa: S102
point_in_polygon = _ns["_point_in_polygon"]

_map = json.loads((SP / "map.json").read_text())["service_response"]
_names = {a["hash"]: a["name"] for a in _map["area_name"]}
POLYS = {
    _names.get(z, "(unnamed)"): [
        {"x": float(q["x"]), "y": float(q["y"])}
        for f in (a.get("data") or [])
        for q in (f.get("data_couple") or [])
    ]
    for z, a in _map["area"].items()
}
BR = POLYS["Backyard Right"]


def edge_clearance(pt: dict[str, float], poly: list[dict[str, float]]) -> float:
    """Return the shortest distance from ``pt`` to any edge of ``poly``."""
    best = 1e9
    for i in range(len(poly)):
        ax, ay = poly[i]["x"], poly[i]["y"]
        bx, by = poly[(i + 1) % len(poly)]["x"], poly[(i + 1) % len(poly)]["y"]
        dx, dy = bx - ax, by - ay
        t = (
            0.0
            if (dx == 0 and dy == 0)
            else max(
                0.0,
                min(
                    1.0,
                    ((pt["x"] - ax) * dx + (pt["y"] - ay) * dy) / (dx * dx + dy * dy),
                ),
            )
        )
        best = min(best, math.hypot(pt["x"] - (ax + t * dx), pt["y"] - (ay + t * dy)))
    return best


def ble_alive() -> int:
    """Return how many BLE sends the HA log recorded in the last 20 seconds."""
    out = subprocess.run(
        [
            str(REPO / "scripts/ha_ssh.exp"),
            'docker logs --since 20s homeassistant 2>&1 | grep -cE "BLETransport send"',
        ],
        capture_output=True,
        text=True,
        timeout=120,
        # grep exits 1 when it counts zero matches; a silent transport is a
        # result to report, not a crash.
        check=False,
    ).stdout
    for tok in reversed(out.split()):
        if tok.strip().isdigit():
            return int(tok.strip())
    return 0


def pulse(action: str, duration_ms: int) -> dict:
    """Send one bounded manual-motion pulse and return the service response."""
    body = json.dumps(
        {
            "entity_id": ENTITY,
            "action": action,
            "speed": 0.6,
            "duration_ms": duration_ms,
            "motion_refresh_interval_ms": 200,
            "stop_mode": "immediate",
            "post_command_sample_delays": [0, 3, 6],
            "dry_run": False,
            "confirm_blades_off": True,
            "confirm_clear_area": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{HA}/api/services/mammotion/manual_velocity_pulse_test?return_response",
        data=body,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=200) as resp:  # noqa: S310
        return json.loads(resp.read())["service_response"]


results = []
print(f"{'dur':>6} {'dir':<9} {'moved_m':>8} {'m/s':>7}  clearance  note")
print("-" * 62)

for i, dur in enumerate(DURATIONS):
    action = "forward" if i % 2 == 0 else "backward"

    n = ble_alive()
    if n < 2:
        print(f"\nABORT before {dur}ms: BLE transport silent ({n} sends/20s)")
        break

    try:
        r = pulse(action, dur)
    except Exception as exc:  # noqa: BLE001
        print(f"\nABORT at {dur}ms: {type(exc).__name__}: {exc}")
        break

    if r.get("blockers"):
        print(f"\nABORT at {dur}ms: blockers {r['blockers']}")
        break
    if (r.get("command_result") or {}).get("ok") is not True:
        print(f"\nABORT at {dur}ms: command not ok -> {r.get('command_result')}")
        break

    samples = r.get("samples") or []
    if len(samples) < 2:
        print(f"\nABORT at {dur}ms: no samples")
        break
    p0 = (samples[0].get("telemetry") or {}).get("position") or {}
    pN = (samples[-1].get("telemetry") or {}).get("position") or {}
    if p0.get("x") is None or pN.get("x") is None:
        print(f"\nABORT at {dur}ms: position unavailable")
        break

    moved = math.hypot(pN["x"] - p0["x"], pN["y"] - p0["y"])
    now = {"x": pN["x"], "y": pN["y"]}
    inside = [nm for nm, P in POLYS.items() if point_in_polygon(now, P)]
    clear = edge_clearance(now, BR)

    note = ""
    if "Backyard Right" not in inside:
        note = "LEFT AREA"
    elif clear < MIN_CLEARANCE_M:
        note = "TOO CLOSE TO EDGE"

    results.append(
        {
            "duration_ms": dur,
            "action": action,
            "moved_m": round(moved, 4),
            "rate": round(moved / (dur / 1000), 4),
            "x": now["x"],
            "y": now["y"],
            "clearance_m": round(clear, 3),
        }
    )
    print(
        f"{dur:>6} {action:<9} {moved:>8.4f} {moved / (dur / 1000):>7.3f}  {clear:>6.2f} m  {note}"
    )

    if note:
        print(f"\nABORT: {note} at ({now['x']:.4f}, {now['y']:.4f})")
        break

    time.sleep(2)

(SP / "sweep_results.json").write_text(json.dumps(results, indent=2))
print(f"\n{len(results)} pulses completed -> sweep_results.json")
