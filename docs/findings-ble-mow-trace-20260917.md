# Findings: BLE disconnect share on a two-proxy mow (2026-09-17)

Predeclaration: `docs/predeclared-ble-mow-trace-20260917.md` (`b0000739`, before the
mow started; notes on speed/path and manual pauses added during it, rules
unchanged). Evidence: `docs/evidence-ble-mow-trace-20260917/` — `trace_log_mow.jsonl`,
`ha_history.json`, `score.json`.

## Reading: **lower** (11.2 % ≤ 17 %) — one mow against one mow

| | 2026-09-17 | 2026-09-16 baseline |
| --- | --- | --- |
| RF set | `hot-tub-backyard`, `p1s-printer` | same |
| mowing time | 35.3 min in 4 stretches (3 operator pauses) | ~54 min, tracer gap 23:55–00:09 |
| **disconnected share** | **11.2 %** (231 / 2 068 mowing rows) | 27.3 % (all rows, incl. return) |
| connected but frozen ≥ 15 s | **30.5 %** (630 rows) | not measured |
| samples (§4 rules) | 1 021, all `p1s-printer` | 1 675 |
| 1 m cells / n ≥ 10 | 135 / **29** | 132 / 74 |
| cell medians, n ≥ 10 | −84 to −68, median −74 | −82 to −56, median −72 |
| settings | wider paths, faster speed (operator) | app defaults |

Per the predeclaration this **supports no claim** about any proxy change: both mows
ran on the same two proxies, with different routes, speeds and baselines.

## What matters more than the verdict

🚨 **"Disconnected" badly understates lost data.** A further **30.5 %** of mowing
rows sat on a frozen position and RSSI while HA still listed the mower on
`p1s-printer` — e.g. 165 s at (13.0, −15.0) from 02:36:04Z. `ble_link_live` went
**off at 02:33:50Z** and stayed off until 02:44:52Z, spanning those frozen runs, so
they were a dead link HA had not yet released, not a quiet live one. Together,
**~42 % of mowing time produced no usable data.** Any future disconnect measure
should count frozen-while-attributed time, or use `ble_link_live`.

**Where it fails:** everything south of about y −11. Disconnect rows cluster at
(9, −14) 129, (12, −16) 32, (11, −16) 23, (14, −19) 22, (6, −11) 20. ⚠️ Those
positions are **where the tracer last saw the mower** while BLE was down (the
position then comes over the cloud, coarsely) — read them as "the link was lost
around here", not as exact points.

⚠️ The link also **held** at times as far south as y −19 (−68 to −80 dBm at
02:45Z), so the boundary is not distance alone.

**Implication:** the south end cannot be mapped or reliably controlled over BLE
from `p1s-printer` / `hot-tub-backyard` where they stand. A proxy near
(10, −15) is the lever; more passes or slower mowing add nothing where no data
arrives.
