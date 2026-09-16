#!/usr/bin/env python3
"""Render the per-proxy BLE coverage map as a self-contained HTML viewer.

Three data sources, deliberately kept as SEPARATE layers rather than merged:

1. ``docs/evidence-phase1-siting-20260912.json`` -- the 780-cell siting grid.
   This is an **aggregate** RSSI per cell with no proxy attribution, built from
   banked ``device_tracker`` history. It is the only dense coverage data that
   exists today, so it draws as the baseline layer.
2. ``scripts/ble_proxy_coverage_log.jsonl`` -- the passive advertisement
   collector. Every row names the scanner that heard the advertisement, so this
   is the only source that can answer "which proxy covers where". It is sparse
   by nature: the mower advertises roughly once per ten minutes while
   disconnected, and not at all while connected.
3. ``scripts/ble_proxy_connected_trace_log.jsonl`` -- the active tracer, one
   proxy per run (whichever HA allocated), sampled while connected.

🚨 Merging 1 into 2 would be a lie: the baseline cannot be attributed to a
proxy, and presenting it inside a per-proxy layer would invent attribution the
data does not have. The viewer keeps them switchable and labels each.

The output is a local ``file://`` page with the data embedded. It has to be
embedded -- ``file://`` CORS refuses to fetch a sibling ``.jsonl``.

Usage:  .venv/bin/python scripts/build_ble_coverage_map.py [-o OUT.html]
"""  # noqa: INP001

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
SITING = REPO / "docs" / "evidence-phase1-siting-20260912.json"
ADVERT_LOG = REPO / "scripts" / "ble_proxy_coverage_log.jsonl"
TRACE_LOG = REPO / "scripts" / "ble_proxy_connected_trace_log.jsonl"
DEFAULT_OUT = REPO / "docs" / "ble-coverage-map.html"
# 🔑 The click-to-go card reads this at runtime. The integration already serves
# WWW_DIR at "/mammotion" (see async_setup in __init__.py), so the card fetches
# it from "/mammotion/ble-coverage.json" -- an absolute path that holds whether
# the card itself was loaded from /mammotion/ or /hacsfiles/. Unlike the HTML
# viewer this asset IS committed: it is deployed to the host with the card, and
# its own input (the siting evidence JSON) is committed too.
CARD_ASSET = REPO / "custom_components" / "mammotion" / "www" / "ble-coverage.json"

# One metre, matching the "target >= 10 samples per 1 m cell" rule in CLAUDE.md.
BIN_M = 1.0

# 🚨 The ramp domain is computed PER LAYER, in the page, not fixed here.
# Two earlier attempts failed on readability, both verified by rendering:
#   * a fixed -95..-55 window -- the siting grid spans only -80..-56, so every
#     cell drew as nearly the same blue and the weak corners, the entire point
#     of the map, were invisible;
#   * one shared domain computed across all layers -- a SINGLE -93 advertisement
#     sample dragged it to -94..-55 and undid the fix for the 724-cell layer.
# Each layer therefore scales to its own data and the legend prints its own
# endpoints, so a colour is never read across layers without the numbers.

# The dataviz reference palette, validated with scripts/validate_palette.js.
# Slots 1-3 are the only ones that clear the all-pairs gates in both modes,
# which is the pairlist a scatter/choropleth form is held to; a fourth proxy
# folds into the neutral "Other" rather than taking slot 4.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70"]
OTHER_LIGHT = "#898781"
OTHER_DARK = "#898781"


def short_proxy(name: str) -> str:
    """Strip the MAC suffix the collectors append to a scanner name."""
    return name.split(" (", 1)[0].strip() or name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return every parseable row, skipping truncated tail lines.

    A collector appends and flushes per row while it runs, so the last line of
    a live file can be a partial write. Dropping it is correct; aborting the
    whole build over it is not.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return rows


#: Proxy tags that attribute a row to no proxy at all.
UNATTRIBUTED_PROXIES = frozenset({"", "?", "disconnected"})


def normalize_trace_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Turn raw connected-trace rows into coverage samples, counting each drop.

    🚨 The tracer writes ``ble_rssi``; ``bin_samples`` reads ``rssi``. Before this
    existed every trace row was silently discarded -- and miscounted as
    ``dropped_no_position`` -- so a driven collection run could never have
    reached the map. Rules, predeclared in
    ``docs/predeclared-ble-connected-trace-collection-20260916.md`` §4:

    * no proxy attribution (``?``/``disconnected``) -> dropped: a per-proxy
      layer cannot hold an unattributed sample;
    * ``ble_rssi`` missing or ``0`` (``0`` means the mower dozed) -> dropped;
    * a row whose ``x``, ``y`` AND ``ble_rssi`` all equal the row before it ->
      dropped as a re-read. The tracer polls faster than the ~1 Hz report
      bundle, so an unchanged triple is most likely the same report twice, and
      counting it would inflate ``n`` toward the >= 10-per-cell bar. A genuinely
      fresh report that happens to repeat all three is lost too; that errs
      toward undercounting, which is the safe direction for a sufficiency bar.
    """
    kept: list[dict[str, Any]] = []
    dropped = {"unattributed": 0, "no_rssi": 0, "repeat": 0, "no_position": 0}
    previous: tuple[Any, Any, Any] | None = None
    for row in rows:
        triple = (row.get("x"), row.get("y"), row.get("ble_rssi"))
        repeat = triple == previous
        previous = triple
        if str(row.get("proxy") or "") in UNATTRIBUTED_PROXIES:
            dropped["unattributed"] += 1
        elif row.get("ble_rssi") in (None, 0):
            dropped["no_rssi"] += 1
        elif row.get("x") is None or row.get("y") is None:
            dropped["no_position"] += 1
        elif repeat:
            dropped["repeat"] += 1
        else:
            kept.append({**row, "rssi": row["ble_rssi"]})
    return kept, dropped


def bin_samples(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Group positioned samples into per-proxy 1 m cells.

    Rows without a position are dropped and counted by the caller -- an
    advertisement heard while the position poll was stale tells you the proxy
    was in range but not where the mower was, which is not a coverage sample.
    """
    buckets: dict[tuple[str, float, float], list[float]] = defaultdict(list)
    for row in rows:
        x, y, rssi = row.get("x"), row.get("y"), row.get("rssi")
        if x is None or y is None or rssi is None:
            continue
        key = (
            short_proxy(str(row.get("proxy") or row.get("scanner") or "?")),
            math.floor(float(x) / BIN_M) * BIN_M + BIN_M / 2,
            math.floor(float(y) / BIN_M) * BIN_M + BIN_M / 2,
        )
        buckets[key].append(float(rssi))

    cells = []
    for (proxy, cx, cy), values in sorted(buckets.items()):
        ordered = sorted(values)
        cells.append(
            {
                "proxy": proxy,
                "x": round(cx, 2),
                "y": round(cy, 2),
                "rssi": round(statistics.median(ordered), 1),
                "min": ordered[0],
                "max": ordered[-1],
                "n": len(ordered),
                "source": source,
            }
        )
    return cells


def grid_cell_size(grid: list[dict[str, Any]]) -> tuple[float, float]:
    """Infer the siting grid's cell pitch from its own coordinates.

    Measured rather than assumed: the pitch is not recorded in the evidence
    file, and hardcoding 1.0 m would silently draw gaps or overlaps if it is
    anything else.
    """

    def pitch(values: list[float]) -> float:
        uniq = sorted({round(v, 4) for v in values})
        diffs = [b - a for a, b in zip(uniq, uniq[1:], strict=False) if b - a > 1e-6]
        # Rounded: the raw subtraction yields values like 0.4999999999999996,
        # and this figure ships inside a committed, deployed asset.
        return round(min(diffs), 4) if diffs else 1.0

    return pitch([c["x"] for c in grid]), pitch([c["y"] for c in grid])


def build_payload() -> dict[str, Any]:
    """Assemble every layer plus the provenance the page has to display."""
    siting = json.loads(SITING.read_text())
    grid = [c for c in siting.get("grid", []) if c.get("rssi") is not None]
    cw, ch = grid_cell_size(grid)

    advert_rows = read_jsonl(ADVERT_LOG)
    raw_trace_rows = read_jsonl(TRACE_LOG)
    trace_rows, trace_dropped = normalize_trace_rows(raw_trace_rows)
    advert_cells = bin_samples(advert_rows, "advert")
    trace_cells = bin_samples(trace_rows, "trace")
    per_proxy = advert_cells + trace_cells

    proxies: dict[str, dict[str, Any]] = {}
    for cell in per_proxy:
        entry = proxies.setdefault(
            cell["proxy"], {"name": cell["proxy"], "cells": 0, "samples": 0}
        )
        entry["cells"] += 1
        entry["samples"] += cell["n"]
    ranked = sorted(proxies.values(), key=lambda p: (-p["samples"], p["name"]))
    for index, entry in enumerate(ranked):
        entry["slot"] = index if index < len(SERIES_LIGHT) else -1

    positioned = sum(
        1
        for r in advert_rows + trace_rows
        if r.get("x") is not None and r.get("rssi") is not None
    )

    return {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frame": "mower_map_xy (ENU metres)",
        "bin_m": BIN_M,
        "dock": siting.get("dock_map_xy"),
        "parked": siting.get("parked_position_primary"),
        "bands": siting.get("rssi_bands", {}),
        "area": siting.get("area_polygon", []),
        "keep_out": list(siting.get("keep_out_polygons", {}).values()),
        "grid": grid,
        "grid_cell": [cw, ch],
        "grid_source": SITING.name,
        "per_proxy": per_proxy,
        "proxies": ranked,
        "series_light": SERIES_LIGHT,
        "series_dark": SERIES_DARK,
        "other_light": OTHER_LIGHT,
        "other_dark": OTHER_DARK,
        "counts": {
            "advert_rows": len(advert_rows),
            "trace_rows": len(raw_trace_rows),
            "trace_samples_kept": len(trace_rows),
            "trace_dropped": trace_dropped,
            "positioned": positioned,
            "dropped_no_position": len(advert_rows) + len(trace_rows) - positioned,
            "grid_cells": len(grid),
        },
    }


def card_asset_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim the viewer payload down to what the click-to-go card needs.

    🚨 Every piece of MAP GEOMETRY is dropped here on purpose -- the area
    polygon, the keep-out polygons, the dock and park landmarks. The card
    already draws all of that from the live ``export_map`` service, and this
    file's copy came from a months-old evidence snapshot. Shipping both would
    put two disagreeing outlines on one map, and the stale one would look just
    as authoritative. The card gets coverage VALUES only and keeps its own
    geometry as the single source of truth.
    """
    return {
        "generated_at_utc": payload["generated_at_utc"],
        "frame": payload["frame"],
        "bin_m": payload["bin_m"],
        "bands": payload["bands"],
        "grid_cell": payload["grid_cell"],
        "grid_source": payload["grid_source"],
        "baseline": [
            {"x": c["x"], "y": c["y"], "rssi": c["rssi"], "n": c.get("nsamp", 0)}
            for c in payload["grid"]
        ],
        "per_proxy": [
            {
                "proxy": c["proxy"],
                "x": c["x"],
                "y": c["y"],
                "rssi": c["rssi"],
                "n": c["n"],
            }
            for c in payload["per_proxy"]
        ],
        "proxies": payload["proxies"],
        "counts": payload["counts"],
    }


def render(payload: dict[str, Any]) -> str:
    """Return the complete standalone page."""
    return TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backyard BLE Coverage</title>
<style>
  .viz-root {
    color-scheme: light;
    --surface-1: #fcfcfb;
    --plane: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --axis: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --area-fill: rgba(11,11,11,0.035);
    --keepout: rgba(11,11,11,0.13);
    --warning: #fab219;
    --critical: #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root {
      color-scheme: dark;
      --surface-1: #1a1a19;
      --plane: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --axis: #383835;
      --border: rgba(255,255,255,0.10);
      --area-fill: rgba(255,255,255,0.05);
      --keepout: rgba(255,255,255,0.16);
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --plane: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255,255,255,0.10);
    --area-fill: rgba(255,255,255,0.05);
    --keepout: rgba(255,255,255,0.16);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--plane); }
  .viz-root {
    background: var(--plane);
    color: var(--text-primary);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 24px 16px 48px;
    max-width: 1100px;
    margin: 0 auto;
  }
  h1 { font-size: 20px; margin: 0 0 4px; font-weight: 650; }
  .sub { color: var(--text-secondary); margin: 0 0 20px; font-size: 13px; }
  .note {
    border: 1px solid var(--border);
    border-left: 3px solid var(--warning);
    background: var(--surface-1);
    padding: 10px 14px; border-radius: 6px;
    margin: 0 0 18px; font-size: 13px; color: var(--text-secondary);
  }
  .note strong { color: var(--text-primary); }
  .controls {
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    margin: 0 0 14px;
  }
  .controls .spacer { flex: 1 1 12px; }
  button {
    font: inherit; font-size: 13px; cursor: pointer;
    background: var(--surface-1); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 12px; min-height: 32px;
  }
  button[aria-pressed="true"] {
    border-color: var(--text-secondary);
    box-shadow: inset 0 0 0 1px var(--text-secondary);
    font-weight: 600;
  }
  button:focus-visible { outline: 2px solid var(--text-primary); outline-offset: 2px; }
  .swatch {
    display: inline-block; width: 10px; height: 10px; border-radius: 2px;
    margin-right: 6px; vertical-align: -1px;
  }
  .card {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px; overflow: hidden;
  }
  svg { display: block; width: 100%; height: auto; touch-action: none; }
  .legend {
    display: flex; flex-wrap: wrap; gap: 18px; align-items: center;
    margin-top: 12px; font-size: 12px; color: var(--text-secondary);
  }
  .ramp { position: relative; width: 210px; height: 11px; border-radius: 2px;
          border: 1px solid var(--border); }
  .ramp-wrap { display: flex; flex-direction: column; gap: 3px; padding-top: 13px; }
  .ramp-ticks { display: flex; justify-content: space-between; width: 210px;
                font-variant-numeric: tabular-nums; }
  /* The two band thresholds are where the operational meaning lives, so they
     are marked on the ramp itself rather than left to the caption. */
  .mark { position: absolute; top: -2px; bottom: -2px; width: 1px;
          background: var(--text-primary); opacity: .75; }
  /* Above the ramp, not below: below put them in the endpoint row, where the
     -76 label overlapped the "-81 dBm weak" text at the left edge. */
  .mark b { position: absolute; bottom: 13px; left: 0; transform: translateX(-50%);
            font-weight: 500; font-size: 10px; color: var(--text-secondary);
            font-variant-numeric: tabular-nums; }
  #tip {
    position: fixed; pointer-events: none; z-index: 10; opacity: 0;
    transition: opacity .1s; background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 8px 10px; font-size: 12px; line-height: 1.45;
    box-shadow: 0 4px 14px rgba(0,0,0,.14); max-width: 260px;
  }
  #tip b { font-weight: 650; }
  #tip .k { color: var(--text-secondary); }
  table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 8px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid);
           font-variant-numeric: tabular-nums; }
  th { color: var(--text-secondary); font-weight: 600; }
  .empty { color: var(--text-secondary); padding: 10px 0; font-size: 13px; }
  .hint { color: var(--muted); font-size: 12px; margin-top: 10px; }
</style>
</head>
<body>
<div class="viz-root">
  <h1>Backyard BLE Coverage</h1>
  <p class="sub" id="sub"></p>
  <div class="note" id="provenance"></div>

  <div class="controls" id="layers"></div>

  <div class="card">
    <svg id="map" role="img" aria-label="BLE coverage map of the mowing area"></svg>
    <div class="legend" id="legend"></div>
  </div>

  <div class="controls" style="margin-top:14px">
    <button id="zoomin" type="button">Zoom in</button>
    <button id="zoomout" type="button">Zoom out</button>
    <button id="reset" type="button">Reset view</button>
    <span class="spacer"></span>
    <button id="tabletoggle" type="button" aria-pressed="false">Table view</button>
    <button id="theme" type="button">Toggle theme</button>
  </div>
  <p class="hint">Scroll or pinch to zoom, drag to pan. Hover any cell for its
     values; the table view lists the same numbers without relying on colour.</p>

  <div id="tablewrap" hidden></div>
</div>
<div id="tip" role="status" aria-live="polite"></div>

<script>
const D = __PAYLOAD__;

/* ---------- sequential ramp: one hue, light -> dark (dataviz palette) ------ */
const RAMP = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7",
              "#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"];
const hex2rgb = h => [1,3,5].map(i => parseInt(h.slice(i,i+2),16));
const RGB = RAMP.map(hex2rgb);
/* Per-layer domain, padded 1 dB. A layer whose cells all share one value (the
   n=1 case) would otherwise divide by zero, so it gets an arbitrary +/-3 dB
   window -- the colour is meaningless there either way, which is why the
   legend always prints the endpoints and the table view carries the number. */
function domainFor(cells) {
  if (!cells.length) return [-90, -55];
  const vals = cells.map(c => c.rssi);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1e-6) return [Math.floor(lo) - 3, Math.ceil(hi) + 3];
  return [Math.floor(lo) - 1, Math.ceil(hi) + 1];
}
/* Stronger signal reads darker. RSSI is negative, so the domain's upper bound
   is the strong end and takes the darkest step. */
function rampColor(rssi, dom) {
  const [lo, hi] = dom;
  let t = (rssi - lo) / (hi - lo);
  t = Math.max(0, Math.min(1, t));
  const p = t * (RGB.length - 1), i = Math.floor(p), f = p - i;
  const a = RGB[i], b = RGB[Math.min(i + 1, RGB.length - 1)];
  return `rgb(${a.map((v,k) => Math.round(v + (b[k]-v)*f)).join(",")})`;
}
const isDark = () => document.documentElement.dataset.theme
  ? document.documentElement.dataset.theme === "dark"
  : matchMedia("(prefers-color-scheme: dark)").matches;
const seriesColor = slot => slot < 0
  ? (isDark() ? D.other_dark : D.other_light)
  : (isDark() ? D.series_dark : D.series_light)[slot];

/* ---------- world -> screen ------------------------------------------------ */
const W = 900, H = 640, PAD = 34;
const pts = D.area.length ? D.area : D.grid;
const bx = [Math.min(...pts.map(p=>p.x)), Math.max(...pts.map(p=>p.x))];
const by = [Math.min(...pts.map(p=>p.y)), Math.max(...pts.map(p=>p.y))];
for (const c of D.grid) {
  bx[0]=Math.min(bx[0],c.x); bx[1]=Math.max(bx[1],c.x);
  by[0]=Math.min(by[0],c.y); by[1]=Math.max(by[1],c.y);
}
const M = 1.2;
bx[0]-=M; bx[1]+=M; by[0]-=M; by[1]+=M;
const scale = Math.min((W-2*PAD)/(bx[1]-bx[0]), (H-2*PAD)/(by[1]-by[0]));
const ox = PAD + ((W-2*PAD) - (bx[1]-bx[0])*scale)/2;
const oy = PAD + ((H-2*PAD) - (by[1]-by[0])*scale)/2;
const sx = x => ox + (x - bx[0]) * scale;
const sy = y => oy + (by[1] - y) * scale;   /* north up */

const svgNS = "http://www.w3.org/2000/svg";
const el = (n, a) => { const e = document.createElementNS(svgNS, n);
  for (const k in a) e.setAttribute(k, a[k]); return e; };

/* ---------- layers -------------------------------------------------------- */
const LAYERS = [{ id: "baseline", label: "Aggregate baseline", kind: "grid" }]
  .concat(D.proxies.map(p => ({ id: "proxy:" + p.name, label: p.name,
                                kind: "proxy", proxy: p })));
if (D.proxies.length > 1) LAYERS.push({ id: "best", label: "Best server", kind: "best" });
let active = LAYERS[0].id;

function cellsFor(layer) {
  if (layer.kind === "grid") return D.grid.map(c => ({...c, _kind: "grid"}));
  if (layer.kind === "proxy")
    return D.per_proxy.filter(c => c.proxy === layer.proxy.name)
                      .map(c => ({...c, _kind: "proxy"}));
  /* best server: one winner per cell, strongest median RSSI */
  const best = new Map();
  for (const c of D.per_proxy) {
    const k = c.x + "|" + c.y, cur = best.get(k);
    if (!cur || c.rssi > cur.rssi) best.set(k, c);
  }
  return [...best.values()].map(c => ({...c, _kind: "best"}));
}
const slotOf = name => (D.proxies.find(p => p.name === name) || {slot:-1}).slot;

/* ---------- render -------------------------------------------------------- */
const svg = document.getElementById("map");
let viewport;

function draw() {
  const layer = LAYERS.find(l => l.id === active);
  const cells = cellsFor(layer);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = "";
  viewport = el("g", {});
  svg.appendChild(viewport);

  /* yard outline */
  if (D.area.length) {
    viewport.appendChild(el("polygon", {
      points: D.area.map(p => `${sx(p.x)},${sy(p.y)}`).join(" "),
      fill: "var(--area-fill)", stroke: "var(--axis)", "stroke-width": 1 }));
  }

  /* data marks, under the keep-outs so an obstacle is never hidden by a cell */
  const [gw, gh] = D.grid_cell;
  const dom = domainFor(cells);
  for (const c of cells) {
    const w = (c._kind === "grid" ? gw : D.bin_m) * scale;
    const h = (c._kind === "grid" ? gh : D.bin_m) * scale;
    const fill = layer.kind === "best"
      ? seriesColor(slotOf(c.proxy)) : rampColor(c.rssi, dom);
    const r = el("rect", {
      x: sx(c.x) - w/2, y: sy(c.y) - h/2,
      width: Math.max(1, w - 2), height: Math.max(1, h - 2),  /* 2px surface gap */
      rx: 2, fill, "fill-opacity": 0.92 });
    r.addEventListener("pointerenter", e => showTip(e, c, layer));
    r.addEventListener("pointermove", moveTip);
    r.addEventListener("pointerleave", hideTip);
    viewport.appendChild(r);
  }

  /* keep-outs */
  for (const poly of D.keep_out) {
    if (!poly || !poly.length) continue;
    viewport.appendChild(el("polygon", {
      points: poly.map(p => `${sx(p.x)},${sy(p.y)}`).join(" "),
      fill: "var(--keepout)", stroke: "var(--axis)", "stroke-width": 1 }));
  }

  /* landmarks: 2px surface ring so they read over any cell colour */
  landmark(D.dock, "Dock");
  landmark(D.parked, "Park");

  applyTransform();
  renderLegend(layer, cells, dom);
  renderTable(layer, cells);
}

function landmark(pos, label) {
  if (!pos) return;
  const g = el("g", {});
  g.appendChild(el("circle", { cx: sx(pos[0]), cy: sy(pos[1]), r: 7,
    fill: "none", stroke: "var(--surface-1)", "stroke-width": 4 }));
  g.appendChild(el("circle", { cx: sx(pos[0]), cy: sy(pos[1]), r: 7,
    fill: "none", stroke: "var(--text-primary)", "stroke-width": 1.5 }));
  const t = el("text", { x: sx(pos[0]) + 11, y: sy(pos[1]) + 4,
    fill: "var(--text-primary)", "font-size": 12, "paint-order": "stroke",
    stroke: "var(--surface-1)", "stroke-width": 3 });
  t.textContent = label;
  g.appendChild(t);
  viewport.appendChild(g);
}

/* ---------- zoom / pan ---------------------------------------------------- */
let k = 1, tx = 0, ty = 0;
const applyTransform = () =>
  viewport && viewport.setAttribute("transform", `translate(${tx},${ty}) scale(${k})`);
function zoomAt(px, py, factor) {
  const nk = Math.max(1, Math.min(14, k * factor));
  tx = px - (px - tx) * (nk / k); ty = py - (py - ty) * (nk / k); k = nk;
  if (k === 1) { tx = 0; ty = 0; }
  applyTransform();
}
function svgPoint(evt) {
  const r = svg.getBoundingClientRect();
  return [(evt.clientX - r.left) * (W / r.width), (evt.clientY - r.top) * (H / r.height)];
}
svg.addEventListener("wheel", e => {
  e.preventDefault();
  const [px, py] = svgPoint(e);
  zoomAt(px, py, e.deltaY < 0 ? 1.15 : 1/1.15);
}, { passive: false });
let dragging = null;
svg.addEventListener("pointerdown", e => {
  dragging = { x: e.clientX, y: e.clientY }; svg.setPointerCapture(e.pointerId);
});
svg.addEventListener("pointermove", e => {
  if (!dragging) return;
  const r = svg.getBoundingClientRect();
  tx += (e.clientX - dragging.x) * (W / r.width);
  ty += (e.clientY - dragging.y) * (H / r.height);
  dragging = { x: e.clientX, y: e.clientY };
  applyTransform();
});
const endDrag = () => { dragging = null; };
svg.addEventListener("pointerup", endDrag);
svg.addEventListener("pointercancel", endDrag);
document.getElementById("zoomin").onclick  = () => zoomAt(W/2, H/2, 1.4);
document.getElementById("zoomout").onclick = () => zoomAt(W/2, H/2, 1/1.4);
document.getElementById("reset").onclick   = () => { k = 1; tx = 0; ty = 0; applyTransform(); };

/* ---------- tooltip ------------------------------------------------------- */
const tip = document.getElementById("tip");
function showTip(e, c, layer) {
  const rows = [];
  if (layer.kind === "grid") {
    rows.push(["RSSI (aggregate)", c.rssi + " dBm"],
              ["samples", c.nsamp],
              ["clearance", c.clearance + " m"],
              ["distance to dock", c.d_dock + " m"]);
  } else {
    rows.push(["proxy", c.proxy],
              ["RSSI median", c.rssi + " dBm"],
              ["RSSI min / max", c.min + " / " + c.max + " dBm"],
              ["samples", c.n],
              ["source", c.source === "advert" ? "advertisement" : "connected trace"]);
  }
  tip.innerHTML = `<b>x ${c.x.toFixed(2)}, y ${c.y.toFixed(2)}</b><br>` +
    rows.map(([k2, v]) => `<span class="k">${k2}:</span> ${v}`).join("<br>");
  tip.style.opacity = 1;
  moveTip(e);
}
function moveTip(e) {
  const pad = 14;
  let x = e.clientX + pad, y = e.clientY + pad;
  const r = tip.getBoundingClientRect();
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.style.opacity = 0; }

/* ---------- legend -------------------------------------------------------- */
function renderLegend(layer, cells, dom) {
  const box = document.getElementById("legend");
  if (layer.kind === "best") {
    box.innerHTML = D.proxies.map(p =>
      `<span><span class="swatch" style="background:${seriesColor(p.slot)}"></span>${p.name}</span>`
    ).join("") + `<span>Strongest median RSSI wins each cell.</span>`;
    return;
  }
  const grad = RAMP.map((h, i) =>
    `${h} ${(i/(RAMP.length-1)*100).toFixed(1)}%`).join(", ");
  const n = cells.length;
  const [lo, hi] = dom;
  const strong = D.bands.strong_min_dbm ?? -68;
  const dead = D.bands.rejected_below_dbm ?? -76;
  /* Band thresholds sit ON the ramp. The centre caption that used to live
     between the endpoints was removed -- these labels collided with it. */
  const marks = [dead, strong]
    .filter(v => v > lo && v < hi)
    .map(v => `<span class="mark" style="left:${((v-lo)/(hi-lo)*100).toFixed(1)}%"
                     title="${v} dBm"><b>${v}</b></span>`).join("");
  box.innerHTML = `
    <div class="ramp-wrap">
      <div class="ramp" style="background:linear-gradient(90deg, ${grad})">${marks}</div>
      <div class="ramp-ticks ramp-marks"><span>${lo} dBm weak</span>
        <span>strong ${hi}</span></div>
    </div>
    <span>${n} cell${n === 1 ? "" : "s"} shown</span>
    <span>dies below ${dead} dBm &middot; usable above ${strong}</span>`;
}

/* ---------- table view (the colour-free twin) ----------------------------- */
function renderTable(layer, cells) {
  const wrap = document.getElementById("tablewrap");
  if (!cells.length) { wrap.innerHTML = `<p class="empty">No cells in this layer.</p>`; return; }
  const head = layer.kind === "grid"
    ? ["x", "y", "RSSI dBm", "samples", "clearance m", "dock m"]
    : ["x", "y", "proxy", "RSSI dBm", "min", "max", "samples", "source"];
  const body = cells.slice().sort((a,b) => b.rssi - a.rssi).map(c => layer.kind === "grid"
    ? [c.x, c.y, c.rssi, c.nsamp, c.clearance, c.d_dock]
    : [c.x, c.y, c.proxy, c.rssi, c.min, c.max, c.n, c.source]);
  wrap.innerHTML = `<div class="card"><table><thead><tr>${
    head.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${
    body.map(r => `<tr>${r.map(v => `<td>${v}</td>`).join("")}</tr>`).join("")
  }</tbody></table></div>`;
}

/* ---------- chrome -------------------------------------------------------- */
function renderControls() {
  const box = document.getElementById("layers");
  box.innerHTML = "";
  for (const l of LAYERS) {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-pressed", String(l.id === active));
    const count = l.kind === "grid" ? D.grid.length
      : l.kind === "proxy" ? l.proxy.samples
      : D.per_proxy.length;
    b.innerHTML = (l.kind === "proxy"
      ? `<span class="swatch" style="background:${seriesColor(l.proxy.slot)}"></span>` : "")
      + `${l.label} <span style="color:var(--muted)">(${count})</span>`;
    b.onclick = () => { active = l.id; renderControls(); draw(); };
    box.appendChild(b);
  }
}

document.getElementById("tabletoggle").onclick = e => {
  const wrap = document.getElementById("tablewrap");
  wrap.hidden = !wrap.hidden;
  e.currentTarget.setAttribute("aria-pressed", String(!wrap.hidden));
};
document.getElementById("theme").onclick = () => {
  const r = document.documentElement;
  r.dataset.theme = isDark() ? "light" : "dark";
  renderControls(); draw();
};

document.getElementById("sub").textContent =
  `${D.frame} · ${D.bin_m} m bins · built ${D.generated_at_utc}`;

const c = D.counts;
const parts = [`<strong>Baseline layer</strong> is ${c.grid_cells} cells of
  <em>aggregate</em> RSSI from ${D.grid_source} — it has <strong>no proxy
  attribution</strong> and is shown for context only.`];
if (!D.proxies.length) {
  parts.push(`<strong>No per-proxy samples have been collected yet.</strong>
    The passive collector has ${c.advert_rows} row(s); the mower advertises about
    once per ten minutes while disconnected and not at all while connected, so
    per-proxy coverage needs a driven collection run.`);
} else {
  parts.push(`Per-proxy layers hold <strong>${c.positioned} positioned
    sample(s)</strong> across ${D.proxies.length} prox${D.proxies.length===1?"y":"ies"}
    (${c.advert_rows} advertisement rows, ${c.trace_rows} trace rows).
    ${c.positioned < 100 ? "<strong>That is far too few to characterise coverage</strong> — within-cell RSSI sd is 5.5 dB against a between-cell spread of 7.3 dB, so the target is ≥10 samples per cell." : ""}`);
}
if (c.dropped_no_position) {
  parts.push(`${c.dropped_no_position} row(s) dropped for having no position.`);
}
document.getElementById("provenance").innerHTML = parts.join(" ");

renderControls();
draw();
</script>
</body>
</html>
"""


def main() -> None:
    """Build the viewer and report what went into it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = build_payload()
    args.out.write_text(render(payload))
    CARD_ASSET.write_text(
        json.dumps(card_asset_payload(payload), separators=(",", ":"), sort_keys=True)
        + "\n"
    )

    counts = payload["counts"]
    print(f"wrote {args.out}")  # noqa: T201
    print(f"wrote {CARD_ASSET} ({CARD_ASSET.stat().st_size / 1024:.0f} KB)")  # noqa: T201
    print(  # noqa: T201
        f"  baseline grid   : {counts['grid_cells']} cells (aggregate, no proxy attribution)"
    )
    print(  # noqa: T201
        f"  per-proxy samples: {counts['positioned']} positioned "
        f"({counts['advert_rows']} advert + {counts['trace_rows']} trace rows, "
        f"{counts['dropped_no_position']} dropped)"
    )
    for entry in payload["proxies"]:
        print(  # noqa: T201
            f"    {entry['name']:<28} {entry['samples']:>5} samples "
            f"in {entry['cells']} cell(s)"
        )
    if counts["positioned"] < 100:
        print(  # noqa: T201
            "  NOTE: per-proxy layers are far below the >=10-samples-per-cell bar. "
            "The viewer says so on the page."
        )


if __name__ == "__main__":
    main()
