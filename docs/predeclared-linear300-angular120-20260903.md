# Predeclared — rotation at (linear 300, angular 120), 5 s step (2026-09-03)

**Written before any capture exists.** Companion to
`docs/predeclared-linear300-angular180-20260903.md`; authorizes ONE window.

## 1. Why

Phase A ran `(300, 120)` with a **1000 ms** step, which cannot outlast the
documented ~1–2 s onset lag — so **rotation at that operating point is
unmeasured**, exactly as it was at `(300, 180)` until the 5 s run tonight.

This completes the pair. With both angular commands measured at linear 300, the
claim that the linear-400 constants transfer rests on **two** points instead of
one.

## 2. Configuration

`baseline 3000 / step 5000 / settle 4000` (12000 ms), **linear 300**,
**step_angular_speed 120**, `max_travel_m` 3.0. Identical to tonight's
`(300, 180)` run except the angular command — deliberately, so the two are
directly comparable.

Verified on deployed beta99 before dispatch: **15/15 gates, `blockers: []`**,
`bound_that_binds: travel_budget` (required 3.50 m, clock bound 3.20 m),
projected travel 2.676 m of the 3.0 m budget. ⚠️ **Yard clearance at the live
start is 3.9241 m** — 1.12x the requirement, tighter than the 5.81 m of the
previous run, and verified by scan against the MAP, which the gate does not check.

## 3. Prediction, recorded so it can be falsified

At `(400, 120)` the steady rate was ~**−8.2 °/s**. At `(400 → 300, angular 180)`
the rate rose ~13% (−11.8 → −13.43). **If that scaling is a property of linear
speed, expect ~−9.3 °/s here.** A materially different figure means the effect is
angular-dependent, not a simple linear-speed effect.

## 4. What each outcome means

| outcome | reading |
| --- | --- |
| ~−9.3 °/s | the +13% linear-speed effect is consistent across angular commands |
| ~−8.2 °/s | rotation is independent of linear speed; the (300,180) result needs re-examining |
| materially outside both | the interaction is not a simple scaling — report, do not fit |

⚠️ **2a's verdict is reported, not relied on.** With ~5 step intervals it is
scoreable, but this run is a CHARACTERISATION; any FAIL is expected to reflect
the documented onset bias, not the plant.

## 5. Conditions

19:32 EDT, sunset ~19:55 — **VIO is near the dusk cliff**. `camera_brightness`
`light`, 80/80 features, `signal_good` at predeclaration. If VIO drops during the
window the run is refused with `vio_not_live_throughout` and is UNSCOREABLE by
design; that is an acceptable outcome, not a failure. Battery 76%, BLE live.

**Authorizes nothing further.** Standing decision 5 untouched.
