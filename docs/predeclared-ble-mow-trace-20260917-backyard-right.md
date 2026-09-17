# Predeclared: BLE reliability on a Backyard Right mow (2026-09-17)

Written at 03:15Z, after the tracer started (2026-09-17T03:14:50Z) and
before any mowing row was examined. Operator hypothesis: Backyard Right is closer
to the proxies, so the link should perform better than the earlier mows.

## Measures (fixed now; all over rows while `lawn_mower` reads `mowing`)
1. **Link not live** = rows where `binary_sensor.*_ble_link_live` is not `on` — the
   **primary** measure. Chosen because
   `docs/findings-ble-mow-trace-20260917.md` showed the tracer's `disconnected` tag
   misses dead links HA has not released.
2. Disconnected share (proxy `disconnected`/`?`) and frozen-while-connected share
   (identical position, RSSI and proxy for ≥ 15 consecutive rows), reported
   alongside, as defined there.

## Comparison and reading (fixed now)
Descriptive references, both on the same two-proxy RF set, computed after the
fact with measure 1: 2026-09-16 mow **30.8 %**, 2026-09-17 early mow **45.3 %**.
- **≤ 15 %** link not live: performs **better** than both.
- **≥ 30 %**: **not better**.
- Between: **no clear difference**.
A different area is a different place, not a controlled test of distance; this
reads as "this area on this night", nothing more.
