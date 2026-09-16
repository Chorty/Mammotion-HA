# Findings: first driven per-proxy BLE coverage collection (2026-09-16)

Predeclaration: `docs/predeclared-ble-connected-trace-collection-20260916.md`
(committed `babf9797`, §9 tracer fix `d20b38e2`, both before S1). Evidence:
`docs/evidence-ble-connected-trace-20260916/` (raw `trace_log.jsonl`, per-leg
runner logs/meta/timing, `dispatch_windows.log`, the driver `drive.sh`).

## Verdict: INSUFFICIENT (truncated at S9) — instrument VALID

By §5 as written, **any truncated run is INSUFFICIENT**, and this one stopped at
S9. That holds even though the per-cell bar is met; the rule is applied as
declared, not relaxed after seeing the numbers.

| check | result |
| --- | --- |
| §6.1 RSSI live while driving | **0.867 samples/s** over 211 s of motion windows (bar ≥ 0.3) — PASS |
| §6.2 attribution | **0 of 213** rows in motion windows unattributed (bar ≤ 10 %) — PASS |
| dominant proxy P | `atom-fireplace`, **198 / 198** trace samples (100 %) — no handoff |
| route cells covered for P (≥ 10) | **6 / 6**: n = 24, 21, 19, 15, 18, 24 |
| legs | S1–S8 `target_reached` (0.077–0.118 m); S9 `turn_budget_infeasible` |

393 raw rows → 198 samples; 195 dropped as re-reads, none for any other reason.

## Per-cell medians (atom-fireplace, dBm; route cells marked)

| cell centre | n | median | min | max |
| --- | --- | --- | --- | --- |
| (4.5, -7.5) | 8 | -92 | -96 | -88 |
| (4.5, -6.5) route | 24 | -92 | -98 | -82 |
| (4.5, -5.5) route | 21 | -86 | -104 | -80 |
| (4.5, -4.5) route | 19 | -86 | -94 | -84 |
| (4.5, -3.5) route | 15 | -88 | -98 | -84 |
| (4.5, -2.5) route | 18 | -82 | -96 | -78 |
| (4.5, -1.5) route | 24 | -87 | -96 | -76 |
| (5.5, -7.5) | 7 | -88 | -104 | -88 |
| (5.5, -5.5) | 19 | -90 | -104 | -84 |
| (5.5, -4.5) | 28 | -90 | -108 | -84 |
| (5.5, -3.5) | 15 | -90 | -96 | -84 |

## Sequence

- **S1** 22:01:59Z, 0.113 m.
- **S2** halted twice pre-dispatch on `vio_tracked_features min 68 below 70`
  (sun ≥ 10°) — the first halt stopped the driver, which lacked the predeclared
  retry; the retry was added and the sequence resumed at S2; the second halt
  auto-retried after 60 s; landed 0.106 m. Nothing was sent on either halt.
- **S3–S8** landed 0.077–0.118 m (60 s wait after S4 held).
- **S9** 22:07:41Z halted `turn_budget_infeasible` — the **third** S9 halt of
  this kind on this route (2026-09-13, 2026-09-14 session 2). Not a predeclared
  auto-retry case, so the sequence stopped.
- Gate disarmed and verified after every leg; after S9 also RAW
  `core.config_entries` `enable_experimental_motion: False`.

## Observations (recorded, not claims)

- ⚠️ **Every driven sample read −76 to −108 dBm while every leg completed.**
  The siting baseline estimated −66 to −76 for these cells, and the project rule
  of thumb is "dies below ~−76". Both baseline and trace come from the same
  `ble_rssi` field, but on different days and possibly different proxies. The
  −76 rule may not transfer across proxies, or `ble_rssi`'s scale may; this run
  cannot tell which.
- 🔑 **S9 is the repeat offender on this route.** Its turn is the ~180°
  reversal at the north end; three of four runs have halted there. This bears on
  the unexplained monotonic turn-rate decline (16.01 → 15.52 → 13.25 °/s).
- Before S1 the tracer tagged every row `disconnected` (§9) — the allocation
  subscription sends one-scanner deltas after its first snapshot.
