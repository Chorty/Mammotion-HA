# Route 1 +180/7000 repeat — 2a FAILS under E-VIO. The banked flip does not reproduce. (2026-09-01)

**Verdict: FAIL on criterion 2a.** Predeclared in
`docs/phase2-route1-plus180-repeat-predeclared-20260901.md` (commit `e7aa044c`)
**before any capture existed**. Raw samples:
`docs/raw-samples/raw-route1-run2repeat-plus180-step7000-20260901.json`
(sha256 `96acf6fd37e40454fd1a6cdf3e9ab6f76505df69de63806dc241e64d40d22dae`).

## 1. The headline

🏁 **E-VIO ran on hardware for the first time and returned `scoreable: true`.**
Every one of the 147 in-window samples carried `vio_state: 2`, so the
`vio_not_live_throughout` refusal never fired and the rule scored the run on its
own instrument, as designed.

🚨 **And the result contradicts the banked run it was repeating.**

| | banked R2 (2026-08-30, rescored) | this repeat (2026-09-01) |
| --- | --- | --- |
| 2a half-phase rates (°/s) | — | **-8.3789 / -11.7838** |
| 2a `half_diff_deg_per_s` | **0.130** | **3.4049** |
| 2a verdict (bound 1.5 °/s) | **PASS** | **FAIL — 2.27x the bound** |
| 2b last-two diff (°/s) | pass | **1.2547 — PASS** |
| `tau_actuator_s` | 0.7995 | **null** |

**The repeat's 2a margin is 26x the banked run's.** That is not a near miss.

🗑️ **`tau = 0.80 s` MUST NOT BE QUOTED.** It was pinned as a regression fixture,
explicitly "pinned, not blessed — n = 1". At n = 2 the config does not
reproduce a steady step, so the programme still has **no VIO-derived time
constant**. The E-VIO entry in `CLAUDE.md` should be read with this attached.

✅ **The rule's own construction is vindicated.** `tau_actuator_s` and
`omega_step_deg_per_s` both came back **null** because 2a failed — exactly the
behaviour E-VIO was built for, and precisely the failure that produced the
discarded `tau = 7.28 s` on 2026-08-30 when omega was sampled off a ramp. The
rule declined to manufacture a number it had not earned.

## 2. What this settles, and what it does not

**Settles:** the 2026-08-31 offline rescoring's +180 PASS was **n = 1 and did not
generalise**. Under E-VIO, +120 fails 2a (banked) and +180 now fails 2a on
repeat. No route-1 configuration currently has a reproduced 2a pass.

**Does NOT settle:** *why*. Two readings are open and this run cannot separate
them — (a) 7000 ms is genuinely too short for +180 to reach steady rotation, or
(b) the step rotation is not steady in a repeatable way at all. The banked run's
0.130 °/s agreement is either an outlier or evidence the plant sometimes settles.
⚠️ **n = 2, one each way. Do not fit anything to two points.**

## 3. Run record

- Phases landed on schedule: baseline @ 186.61 ms, step @ 3000.24 ms, settle @
  10001.085 ms. Window 15225.8 ms of a 15000 ms plan.
- `reason: "window_complete"` — **not** a guard trip. `aborted_early: false`.
- Informative intervals: baseline 3, step 7, settle 4.
- Cumulative path travel **3.9830 m of the 4.5 m budget (89%)**; net displacement
  3.3911 m. ⚠️ Closer to the guard than any prior step run — worth noting before
  anyone proposes a longer step at this angular speed.
- 75 refresh commands, `refresh_error: None`.
- `rotation_after_zero_deg: -7.8665`.

## 4. Safety

15/15 gates, `blockers: []`. `step_path_contained` reported
`boundary_clearance_m: 5.0` against `required_radius_m: 5.0` with
`live_position_inside: true`. Stop confirmed (`ok: true`, `movement_ok: true`,
103.7 ms). Gate armed only for the dispatch and verified disarmed afterwards from
the **live API and RAW** `core.config_entries`. Battery 41% at dispatch, daylight,
operator on site with per-dispatch go/no-go taken immediately before each of the
two dispatches.

Corridor: 10.0 m square centred on the live start (5.9265, -5.2530), clearance
**5.9199 m** (1.18x). As registered in the predeclaration, that square's far
top-right corner lies outside the mowing-area polygon; travel never approached it
(the mower finished at (5.159, -8.8048), moving away from that corner).

## 5. Two session faults worth recording, neither in the motion path

🐛 **`scripts/ha_set_experimental_motion.py` hardcoded a dead config entry id.**
Arming failed with a bare HTTP 500 whose only detail
(`config_entries.UnknownEntry: 01KVM3JVYBWRKM25ZR8T7FKKJ3`) reached the HA
container log, never the reply. The id predated the 2026-08-31 delete-and-re-add;
the live entry is `01M1CVFWHYWW527S9BM5M2BDP3`. **It surfaces only when arming, so
it reads as a gate or BLE fault.** Fixed in `fa40dd7b` — resolved at runtime now.

🔌 **BLE dropped mid-preparation and the cause was infrastructure.** The
`master_bedroom_proxy` ESPHome device went `unavailable` at 15:34 EDT; the link
survived on an open connection until ~18:06 then could not re-establish
(`never seen by any scanner`, `BleakOutOfConnectionSlotsError`). ⚠️ **Every cheap
indicator lied**: `ble_rssi` read **-60** throughout (self-reported, stale) and
Bermuda counted **3 active proxies** — the other three, none near the yard.
Restored by the operator.

⚠️ **A 97.07° VIO-vs-mirror orientation disagreement appeared while BLE was down**
(`trustworthy: false`, `heading_sources_disagree`) and **resolved itself to
0.678° after the repositioning drive** — `toward` re-derives from real travel and
had simply latched. It did not affect scoring: E-VIO reads *rates* between
consecutive VIO headings, so a shifted absolute frame cancels.

## 6. What this authorizes

**Nothing further.** Per §5 of the predeclaration: not a +120 repeat, not a
`step_ms` change, not a cap raise, and **not resumption of Phase 2 continuous
steering** — standing decision 5 is untouched. Deciding what a 1-for-2 2a record
means is a separate, deliberately-written decision.
