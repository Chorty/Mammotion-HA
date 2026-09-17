# Predeclared: BLE disconnect share on a two-proxy mow (2026-09-17)

Written and committed at ~02:31Z, after the tracer started (02:29:50Z) and
**before the mow began** (mower `paused` on the dock at 02:30Z). No mow row
existed when these rules were fixed.

## Question
With `garage-m5stack` and `atom-fireplace` disabled by the operator (~22:14Z on
2026-09-16), leaving `hot-tub-backyard` and `p1s-printer` connectable, is the
mower's BLE link down less of the time while mowing than on the 2026-09-16
evening mow?

## Measure (fixed now)
- **Mowing rows:** tracer rows whose `t_utc` falls while `lawn_mower` history
  reads `mowing` (from HA history, not guessed from motion).
- **Disconnected share** = rows with proxy `disconnected` or `?` ÷ mowing rows.
- Per-proxy counts of attributed rows, and the §4 data rules of
  `docs/predeclared-ble-connected-trace-collection-20260916.md` for the map.

## Baseline — and why it is weak
2026-09-16 mow: **27.3 %** disconnected. That figure is over **all** rows from
23:53:38Z to docking, including the return trip, with a tracer gap 23:55–00:09Z,
and it is **the same two-proxy RF set** as tonight. So this is **not** a
with/without-`garage-m5stack` test — both mows ran without it. It measures
run-to-run variation on one configuration.

## Reading (fixed now)
- **≤ 17 %** (10 points under): lower this time.
- **≥ 37 %** (10 points over): higher this time.
- Between: **no difference shown**.
- Any reading is **one mow against one mow**; it supports no claim about a proxy
  change, and the mowed area or route may differ (the job's zone is recorded).

## Also recorded
Where the link was down (positions of disconnected rows), the mow's zone hash,
battery at start and end, and the scanner list at start and end.
