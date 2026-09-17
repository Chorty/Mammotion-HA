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

## Notes added during the mow — recorded, rules unchanged
- **~03:23Z:** the operator opened the Mammotion app to lower speed to 0.7 ft/s.
  `ble_link_live` went off at **03:23:14Z** and trace updates slowed to ~10 s steps
  (cloud-like) while HA still listed the mower on `p1s-printer`. Most likely the
  app took the BLE connection. The app was then closed.
- **03:25:37Z:** on operator request, HA sent `lawn_mower.pause` (paused 03:25:42Z).
  Link back on by **~03:25:50Z**; after 20 s continuously on, HA sent a plain
  `lawn_mower.start_mowing` at 03:26:06Z (resume only, `resume_execute_task`, no
  settings) → `mowing` 03:26:12Z, back on `p1s-printer` at 1 Hz.
- Reported alongside the predeclared figure: link-not-live share **with** and
  **without** the app-caused window 03:23:14–03:25:42Z. The predeclared reading
  uses the full figure.
