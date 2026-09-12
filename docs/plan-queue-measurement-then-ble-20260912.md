# PLAN — finish the queue-timeout measurement, then attack BLE

Written 2026-09-12 (UTC) / 2026-09-11 late evening EDT. Operator's sequencing
call: close the issue that was already open — the Issue 1 step 2 queue-start
timing measurement — **before** touching the BLE signal work that surfaced
after it.

Companions: `docs/plan-post-20260910-session-issues.md` (issue 1),
`docs/predeclared-queue-timeout-measurement-20260911.md` (the criteria, already
committed), `docs/findings-clicktopath-reliability-4m-repeat-20260910.md` §1.5.

---

## Why this order is right, not just preferred

🔑 **Legs 7 and 8 failed under the proxy configuration that exists today.** The
measurement's entire job is to characterise the queue-start distribution *in the
conditions that produced those two refusals*. Add or move a Bluetooth proxy
first and the resulting distribution describes a different radio environment —
it can no longer be compared against the failures it was built to explain, and
the 2026-09-10 evidence becomes unrepeatable.

So the RF environment is **frozen** until Phase 1's data is banked. That freeze
is the load-bearing constraint in this plan, and it is what orders everything
else.

---

## 🚨 Correction to my own §8 amendment, before any data exists

`docs/predeclared-queue-timeout-measurement-20260911.md` §8 said a BLE survey
drive could supply the measurement's legs, because a survey is made of the same
dispatches. **That is wrong in a way that would have corrupted the measurement,
and it is withdrawn here.**

A survey drive deliberately goes where coverage is *weak* — that is its whole
purpose. But §3's "justify raising the constant" criterion fires on a high p95
and a worst-wait fraction ≥ 0.75. Driving into known-bad coverage would inflate
exactly those numbers, and the measurement would then "justify" loosening a
safety constant **on a population chosen for being unrepresentative**. That is
the predeclaration failure mode wearing a different hat: not choosing the
threshold after seeing the data, but choosing the *sample* to suit the
threshold.

✅ **Resolution, fixed now:**

- The §2 population is **only** legs run in known-good coverage, with
  `plan_aligned_leg.py`'s existing `--min-rssi-dbm -76` rejection enforced.
- Survey legs into weak cells are **tagged `survey: true` in the evidence file
  and excluded from the §2 population.** They are still recorded — they are
  useful BLE data and honest per-item records — they simply do not feed the
  queue-timing criteria.
- If a session cannot reach `n ≥ 120` from good-coverage legs alone, the
  verdict is **inconclusive** per §5. It is not topped up with survey legs.

§2–§5 thresholds are otherwise unchanged and are not reopened.

---

## Phase 0 — unblock the mower (now, before anything)

The mower is off-dock at 47 % and falling ~4.3 %/h. Nothing in this plan runs
until it is charging.

1. Operator sends `return_to_dock` (or walks it back). Real motion at night, so
   it is the operator's call and their go.
2. Confirm `binary_sensor.*_charging: on` and battery rising.

⚠️ **Do not "fix" this by switching Bluetooth off.** That is the control path for
a mower stranded in the dark, and re-attaching waits on an advertisement that
may be ~10 minutes away.

---

## Phase 1 — take the measurement (one daylight session)

**Preconditions, all verified live before planning legs:** daylight with real
margin (not the last 45 minutes before sunset); battery ≥ 80 % and charging
completed; RTK `fix`, not `single`; VIO features non-zero; beta104 confirmed on
the host with `motion_dispatch_timing_report` registered.
🚨 **The mower must be PARKED IN A MOWING AREA FIRST — the executor cannot
undock itself** (predeclaration §12.2). On the dock the gate reads
`CHARGE_ON` / `zone_hash: 0` and refuses by design, so `real_motion_ready` and
`position_valid_for_motion` **cannot** go true while docked and must not be
waited on. Operator, app or undock places it; **leg 1 starts from where it is
parked.** Then check POSITIVELY that `pos_type_label` is an accepted area label
and `zone_hash` is non-zero — the absence of a blocker is not the check.
⚠️ **A hand-placed or app-driven mower has stale heading telemetry until it
drives** — derive facing two ways before the first armed dispatch, per the
standing repositioning trap.

**Frozen for the duration:** no proxy added, moved, powered down or
re-provisioned; no change to `motion_refresh_interval_ms`; no change to
`_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS`. ✅ **Record the proxy inventory into
the evidence file at session start** — scanner sources, names and connection
slots — so any later change is attributable and the session stays reproducible.

**Legs — REVISED 2026-09-12, see predeclaration §10–§11.** Plan **~8 legs**,
0.8–1.5 m, each aimed at the live `map_facing_degrees`; ≥ 4 distinct,
≥ 2 running a turn or calibration-drive phase, and **≥ 2 sited 6–8 m from the
dock** (tagged `distance_band: "far"`) so the population is not entirely the
strongest-link regime. 🚨 **The bar is `n ≥ 40` `pulse_open` samples surviving
exclusion** — not 120, and not 40 collected. ~86 % of raw samples are refresh
resends that record identically to pulse-opens; the rate claim needs n ≥ 120 and
is deferred. 8 legs banks ~56–80 pulse-opens, absorbing ~25 % exclusion.

**Protocol, no exceptions:** explicit operator go/no-go immediately before each
dispatch; fresh corridor scan against the map; physical tape measurement on any
corridor under a couple of metres; `docs/accepted-profile.json` sent verbatim
and verified key-by-key; gate verified disarmed from the live API **and** raw
`core.config_entries` at session end; mower returned to the dock at session end.

**Collection:** after **every** leg, call `motion_dispatch_timing_report` and
save the full raw `samples` array — not the percentiles — into
`docs/evidence-queue-timeout-measurement-<date>.json`, tagged by leg, alongside
that leg's `command_result` and any `queue_diagnostics` snapshot. Per-leg
snapshots are what make a `maxlen=500` drop visible instead of silent.

---

## Phase 2 — interpret, and stop there

Report against the four predeclared outcomes (§3 justify / §4 falsifier /
§5 inconclusive), computed **by hand over `queue_budget_seconds == 2.0` samples
only** — never from the reported `worst_wait_fraction_of_budget`, which divides
by `min(budgets)` and so mixes the 5.0 s emergency-stop budget into the 2.0 s
ordinary one.

🛑 **Phase 2 ends with a verdict, not a code change.** Step 3 — moving
`_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS` or `motion_refresh_interval_ms` — is
its own operator decision afterwards, and `motion_refresh_interval_ms` owes its
own predeclaration and Gate 5 as an accepted profile value.

✏️ **My prior, revised 2026-09-12 before any data.** I first recorded §4 as
likely — a tight body with episodic outliers, constant fine. **Predeclaration
§10.5 argues the other way:** `_motion_refresh_window`'s own comment banks
refresh *write* latency at p95 **1029.2 ms** and max **2014.0 ms** across 98
writes, with 59 % exceeding the 200 ms interval. The queue is serialized, so a
dispatch behind a ~1 s write inherits that wait, which materially raises the
chance §3's "p95 ≥ 1000 ms" fires.
🔑 **If it does, the honest reading may be a third answer neither §3 nor §4
anticipates — "write latency on this link is the binding constraint" — not
"raise the constant."** Recorded now so that reading cannot look invented after
the fact.

---

## Phase 3 — BLE, in two tracks with different start times

### 3a. The drain — offline code, starts NOW, in parallel

Safe to run during Phase 1 because it changes no RF condition and sends nothing.

1. **Write a `to_get_dev_low_power_cmd` builder** in Chorty/PyMammotion. The
   protobuf has `dev_low_power_get` / `dev_low_power_set_info`
   (`set_uncharging_lowpower_sta` + `uncharging_low_power` for off-dock,
   and the charging pair) but **pymammotion ships no builder** — the only
   matches anywhere are the `.proto` and generated `_pb2` files, so the command
   can currently be neither sent nor read.
2. **Read the current values.** Read-only. If off-dock low power is simply
   disabled on this unit, that is a clean explanation and a one-value fix, and
   it may make 3a step 3 unnecessary. Do this before building any timer.
3. **Only if the read does not explain it:** an idle BLE release — drop the BLE
   transport after N minutes of no commands, re-attach via the advertisement
   callback already registered unconditionally at `__init__.py`.

🚨 **The idle-release timer must NOT be deployed before Phase 1 completes.** A
release/reconnect cycle mid-session changes queue behaviour outright and would
invalidate the measurement. The read probe in step 2 is harmless and may run
any time; step 3 is strictly post-Phase 1.

⚠️ The low-power command is **unverified** — nobody here has exercised it, and
the behaviour is inferred from field names plus the vendor app's use of it.

### 3b. Coverage and proxy placement — starts only after Phase 1 is banked

1. Re-run `scripts/ble_coverage_map.py` to fold in Phase 1's fresh samples.
2. Place the new proxies against the weak regions measured in the 96 h rebuild
   (`docs/evidence-ble-coverage-96h-20260912.json`): the **north end**
   (x −1..1, y 17..26; median −84 to −89 dBm, worst −98) is by far the worst,
   then the **south end** (x 4..13, y −7..−11; median −77 to −80) and the
   **east edge** (x 13..14, y 1..2). 14 cells sit below the −76 wall with
   30 more marginal at −76…−70.
3. **Do not build proxy-switching logic.** HA already selects the best
   connectable scanner via `bluetooth.async_ble_device_from_address(..., True)`,
   and it is already routing the mower through `hot-tub-backyard`. BLE has no
   roaming, so a switch means disconnect + reconnect, and reconnect routing
   depends on advertisements this mower emits ~once per 10 minutes and not at
   all while connected. Placement is the lever; selection is already automatic.
4. **Then run a dedicated survey session** for coverage only, tagged
   `survey: true`, targeting **≥ 10 samples per 1 m cell** — within-cell RSSI
   sd is 5.5 dB against a between-cell spread of only 7.3 dB, so a single pass
   cannot characterise a cell. Its dispatches do not feed the §2 population.
5. Re-running the queue measurement after placement is **optional and a separate
   decision.** If Phase 2 returned §4, better coverage does not change the
   verdict; if it returned §3, a re-measure post-placement is the natural check.

---

## Order of operations, one table

| # | work | needs mower? | needs daylight? | blocked by |
| --- | --- | --- | --- | --- |
| 0 | dock and charge | yes | no | nothing |
| 1 | queue measurement | yes | yes | phase 0; RF freeze |
| 2 | interpret vs predeclared criteria | no | no | phase 1 |
| 3a.1–2 | low-power read builder + read | no / read-only | no | nothing — runs in parallel |
| 3a.3 | idle BLE release (if needed) | no | no | **phase 1 complete** |
| 3b | proxy placement + survey | yes | yes | **phase 1 banked** |

**The single rule that holds it together:** nothing that changes the radio
environment or the BLE connection lifecycle lands between now and the end of
Phase 1.
