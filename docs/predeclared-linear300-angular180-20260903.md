# Predeclared — the (linear 300, angular 180) operating point (2026-09-03)

**Written before any capture exists. Authorizes ONE short window, nothing else.**

## 1. Why

Every constant behind criterion 2a — the onset deficit Δ ≈ 10.43 °/s, the
residual scatter sd 1.445 °/s, the ~12 °/s steady rate — was measured at
**linear 400**. Phase A exercised `(300, 120)` and found no deadband, but
**`(linear 300, angular 180)` has never been run**: `grep '"linear_speed": 300'`
across `docs/` returns only Phase A.

This run closes that gap. It is a **characterisation, not a criterion test.**

## 2. Configuration

`baseline 3000 / step 1000 / settle 4000` (8000 ms), **linear 300**,
**step_angular_speed 180**, `max_travel_m` **2.5**, corridor a 10 m square on the
live start. Identical to Phase A except the angular command.

Verified on the deployed beta99 bytes before dispatch: **15/15 gates,
`blockers: []`**, clearance **4.5841 m** against 3.00 m required
(`bound_that_binds: travel_budget`; the clock bound is 2.30 m), projected travel
**1.784 m** of the 2.5 m budget, `likely_guard_trip: false`.

## 3. The measurement

**Primary:** VIO rotation rate during the step, and sustained forward speed by
cumulative path ÷ elapsed — the same metric the travel guard accumulates.

**Reported, not scored:** whether the rate at `(300, 180)` is consistent with the
~12 °/s measured at `(400, 180)`.

⚠️ **This run does NOT score 2a.** A 1000 ms step yields ~1 informative step
interval against the rule's ≥3, so `vio_analysis` is expected to be unscoreable
or meaningless. **Any 2a verdict it emits must not be quoted.** Same as Phase A.

## 4. What each outcome means

| outcome | reading |
| --- | --- |
| rate ≈ 12 °/s | angular authority does not depend on linear speed; the 2a constants transfer |
| rate materially lower | angular authority IS linear-speed dependent — the 400-derived constants do not transfer, and several docs need caveating |
| barely rotates | a deadband at `(300, 180)`; that operating point is unusable |

**Authorizes nothing further.** Not a 2a run, not a longer window, not Phase 2 —
standing decision 5 untouched.

## 5. Conditions at predeclaration

19:14 EDT, sunset ~19:55. BLE restored by the operator and verified live
(`is_connected: true`, `queue_depth: 0`, 8 position updates in 8 s), `ble_rssi`
**−70** — at the documented wall, so a mid-run BLE loss is a live possibility and
would abort fail-closed. Battery 77%, VIO 80/80, `light`/`signal_good`, RTK Fix.

⚠️ **`current_orientation.trustworthy` is false** with a 97.08° VIO-vs-mirror
disagreement — the same latched-`toward` signature seen on 2026-09-01, which
resolved after real travel. It does not gate this run: the step probe drives
open loop and needs no VIO calibration, and E-VIO reads *rates*, so a shifted
absolute frame cancels. Recorded because it is visible and must not be
rediscovered as a surprise.
