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

---

## Run 2 (2026-09-16 23:47Z) — INSUFFICIENT (truncated at S2 by VIO) — instrument VALID

Rules: §10 (two proxies) and §11 (operator low-sun override, VIO dips not
retried). Evidence: `evidence-ble-connected-trace-20260916/run2/`
(`trace_log_run2.jsonl` = rows at/after `tracer_start_utc.txt`, 23:47:02Z).

**Before S1:** the mower had been undocked at 22:57Z without driving, so facing
was `unknown` (sources 97.5° apart). The operator confirmed by eye that it faced
south; one 1800 ms forward `manual_velocity_pulse_test` at 23:45:27Z moved it
**0.656 m due south** and facing became `motion_confirmed`, 175.9° compass, all
three sources within 0.9°. The gate had been found **armed at rest** at 23:36Z
(and at ~22:32Z) and was disarmed each time; attribution not established.

| check | result |
| --- | --- |
| RF set | `hot-tub-backyard`, `p1s-printer` (+ `hci0` scan-only); mower on **`p1s-printer`** |
| S1 | `target_reached` 0.121 m (sun −1.5°, override logged by the runner) |
| S2 | HALT pre-dispatch: `vio_tracked_features min 60 below 70` — not retried (§11); sequence stopped |
| §6.1 live RSSI | 27 samples in the 30 s motion window = **0.90 /s** — PASS |
| §6.2 attribution | 0 of 31 rows unattributed — PASS |
| samples | 62 rows → 45 kept (17 re-reads), all `p1s-printer` |
| route cells covered | **0 / 6** — both cells hit are off-route (`x 5..6`) |

| cell centre | n | median | min | max |
| --- | --- | --- | --- | --- |
| (5.5, -4.5) | 21 | -66 | -76 | -62 |
| (5.5, -3.5) | 24 | -66 | -70 | -62 |

🔑 **Observation, not a claim:** those same two cells read **−90** median (n 28,
15) on `atom-fireplace` in run 1, three and a half hours earlier — a **~24 dB**
difference between proxies at the same spot, well outside the 5.5 dB within-cell
sd. It is consistent with proxy placement mattering a great deal, and equally
with `ble_rssi` not being comparable across proxies; two cells and one run each
cannot separate those.

---

## Run 3 (2026-09-16 23:51Z) — INSUFFICIENT; stopped by operator. A mow job took the mower mid-run.

Rules §10–§12 (two proxies, low-sun override, recovered-dip rule). Evidence:
`evidence-ble-connected-trace-20260916/run3/` (`trace_log_run3.jsonl` = rows at or
after `tracer_start_utc.txt`, 23:51:08Z).

- Pre-run 23:50:58Z: gate off; facing `motion_confirmed` SSW 195.4° (sources within
  1.3°); mower on `p1s-printer`.
- **S1** ended in the same second it started (mower already at the S1 target from
  run 2). **S2** `target_reached` 0.089 m.
- **S3** refused before dispatch four times (23:51:49, 23:52:49, 23:53:50,
  23:54:50Z): tracked features at dispatch 44, 59, 0, 0 — "stays below" per §12,
  so the sequence stopped. **Nothing was sent for S3.**
- **At 23:53:38Z `lawn_mower` went `mowing`** — ✏️ **the operator started it from
  the Mammotion app** (confirmed by the operator; the HA logbook showed no user,
  as expected for an app start). The mower drove from (4.87, −5.77) to (14.42, −15.55) by 23:54:59Z,
  `MODE_WORKING`, blades reported on, zone hash `3481535603736850863`. The
  runner's refusals over that window were correct; no run-3 command moved it.
- The operator called a stop at ~23:55Z. Driver, runner and tracer killed; gate
  verified `enabled: False`.
- Trace rows from 23:53:38Z on are the **mow**, not the route; they sit in the
  shared log but are outside every predeclared motion window. The card asset was
  **not** regenerated from them.

---

## Mow trace (2026-09-16 23:53:38Z – 2026-09-17 00:49:45Z) — observational, operator-requested

The operator's app-started mow was traced read-only at their request ("track the
ble signal while it is mowing … use the data to fill the map"). **No predeclared
bar applies and no sufficiency verdict is given**; the data rules of §4 were
applied unmodified. Evidence: `evidence-ble-connected-trace-20260916/mow/`
(`trace_log_mow.jsonl`). The tracer was not running 23:55–00:09:26Z (between
run 3's stop and the mow trace start), so that stretch of the mow is missing.

| | |
| --- | --- |
| rows | 2 501 → **1 675 samples** (684 unattributed, 22 RSSI 0, 120 re-reads) |
| proxy | **`p1s-printer` only** — every attributed row |
| link down | **27.3 %** of rows `disconnected` (e.g. from ~00:10:50Z near (10.2, −18.9)) |
| cells | **132** (1 m), **74 with n ≥ 10**; extent x 0..15, y −23..4 |
| cell medians (n ≥ 10) | median **−72**, range **−82 to −56** dBm |

⚠️ **Unattributed rows are where the link was DOWN, and the map does not draw
them** — a blank cell can mean "never visited" or "visited while disconnected".
Their positions are in the raw log if a disconnect layer is wanted later.
The card asset was regenerated from the whole log (run 1, run 2, run 3's first
minutes, and this mow).

✏️ **Correction (operator, 2026-09-17):** `garage-m5stack` was **disabled by the
operator** at ~22:14Z along with `atom-fireplace` — §10's "cause not established"
is resolved. So **runs 2, 3 and the mow all ran on two connectable proxies**
(`hot-tub-backyard`, `p1s-printer`), and `garage-m5stack` cannot have caused the
mow's 27 % disconnected time. It remains implicated in earlier drops: Window D
(2026-09-15) put nearly every connection and both `error=8` timeouts on it, and
HA's logs ranked it the preferred path 4 of 22 times in 72 h at −94 to −97 dBm.
