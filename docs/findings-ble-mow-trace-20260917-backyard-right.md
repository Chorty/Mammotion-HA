# Findings: BLE reliability on a Backyard Right mow (2026-09-17)

Predeclaration: `docs/predeclared-ble-mow-trace-20260917-backyard-right.md`
(`96a82cca`, before any mowing row; notes on the app drop and HA pause/resume added
during the mow, rules unchanged). Evidence:
`docs/evidence-ble-mow-trace-20260917-backyard-right/` (`trace_log_mow.jsonl`,
`ha_history.json`, `score.json`).

## Reading: **better** — link not live **4.3 %** (≤ 15 %)

| | Backyard Right 03:16–04:11Z | early mow 02:31–03:08Z | 2026-09-16 mow |
| --- | --- | --- | --- |
| RF set | `hot-tub-backyard`, `p1s-printer` | same | same |
| mowing time | 54.8 min | 35.3 min | ~54 min (40 traced) |
| **link not live** | **4.3 %** | 45.3 % | 30.8 % |
| … excluding the app window | **0.0 %** | — | — |
| disconnected | 0.0 % | 11.2 % | 28.9 % |
| frozen while connected (≥ 15 s) | 2.8 % — **one 90 s run inside the app window** | 30.5 % | 0.0 % |
| samples (§4 rules) | **3 009**, all `p1s-printer` | 1 021 | 1 675 |
| 1 m cells / n ≥ 10 | 194 / **142** | 135 / 29 | 132 / 74 |
| cell medians, n ≥ 10 | −84 to −52, median −72 | −84 to −68 | −82 to −56 |
| extent | x 0..15, y −11..5 | south end | x 0..15, y −23..4 |

- **All link loss was caused by the app.** The operator opened the Mammotion app at
  ~03:23Z; `ble_link_live` went off at 03:23:14Z and the trace fell to ~10 s cloud
  steps. After the app was closed and HA paused (03:25:42Z), the link was back in
  ~10 s; HA resumed with `resume_execute_task` only at 03:26:06Z. From 03:26:12Z
  to the end of mowing (**45.4 min**) there was **no** link loss, disconnect or
  stall.
- As predeclared, a different area is not a controlled distance test. What it
  does show: **within about y > −11 the two-proxy set holds a solid link**; the
  failures on the early mow were confined to south of y −11.

## Operational lessons
1. **Opening the Mammotion app takes the mower's BLE connection away from HA.**
   HA keeps listing the old proxy allocation while the link is dead. Keep the app
   closed during any HA-driven or BLE-traced session.
2. **Closing the app returns the link in seconds** (~10 s here), not the 2–4 min a
   range drop costs, because the mower advertises straight away.
3. **Plain `lawn_mower.start_mowing` on a paused job is a resume only** — it sends
   no HA operation settings, so the stale-settings bug in `docs/TODO.md` does not
   apply to it.
