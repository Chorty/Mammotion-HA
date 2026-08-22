# Proposed Phase 1 criterion revision — prediction, not identity

**2026-08-22, offline, no mower run. THIS IS A PROPOSAL FOR REVIEW.** Nothing
here is implemented in `analyze_phase1_capture.py`, whose verdict remains
`no_go`. It exists so the criterion can be fixed **before** another capture, not
after one.

Operator decisions this implements, taken 2026-08-22: continuous motion
continues because **fluidity is the point**; the criterion becomes **one-step
prediction error**; and **one further mower run** is budgeted, which must be
decisive.

## Why the criterion changes

The shipped `bearing_toward_compass_mirror` check is ill-posed on a rotating
body -- it compares an interval-average chord bearing to an instantaneous
`toward`, and the answer swings by the full per-interval rotation depending on
an arbitrary pairing. Measured: `alpha = -0.253 +- 0.174`, which excludes the
shipped END convention at 7.19 sigma. Details in
`docs/phase1-mirror-criterion-is-ill-posed-20260822.md`.

More fundamentally, **a controller does not consume that identity.** It consumes
a prediction: from the last fix, the last heading, and what it just commanded,
where will the mower be when the next fix arrives? Scoring that directly is
immune to the pairing question entirely, because a constant lag is absorbed into
a fitted rate instead of deciding a pass or fail.

## The proposed criterion

Model: speed `v = k_lin * linear_speed`, yaw rate `w = k_ang * angular_speed`,
integrated as a circular arc of radius `v / w` from the mirror-derived facing at
the interval start. `scripts/replay_arc_predictability.py` implements it.

Proposed thresholds, **to be argued before they are frozen**:

| | proposed | basis |
| --- | ---: | --- |
| median one-step error | <= 0.10 m | today's steady-state 0.025-0.052 m, with room |
| max one-step error | <= 0.15 m | the waypoint tolerance itself |
| minimum steps scored | >= 2 after the spin-up interval | ~3 arrivals per run is the ceiling |

## What today's data already says

Scored with `k_lin` fitted on the straight capture and held out for the arc:

| capture | all intervals | excluding spin-up |
| --- | --- | --- |
| straight, angular 0 | median 0.0769, max 0.1579 m | **median 0.0521, max 0.0769 m** |
| arc, angular 180 | median 0.0248, max 0.1138 m | **median 0.0248, max 0.0248 m** |

🔑 **The dominant error is the acceleration transient, not curvature.** The first
interval of each run is the mower spinning up, so a constant-velocity model
necessarily overshoots it -- 0.1579 m and 0.1138 m, both far worse than every
steady-state interval that follows. Curvature is nearly free by comparison: the
arc predicts *better* than the straight run once spun up.

This matters beyond the criterion. A 4 s window spends its **first quarter** in a
regime the model cannot fit, so short windows flatter the mirror check and
punish the predictor. A genuinely continuous controller would spend
proportionally far less time there.

⚠️ **The arc figures are optimistic**: `k_ang` could only be fitted on the single
arc it is then scored against. That is exactly what the next run must fix.

## The one decisive run

**A second arc at a different `angular_speed`, scored with `k_lin` and `k_ang`
frozen from today's two captures beforehand.**

That makes it a genuine out-of-sample test of the thing in doubt. Today's arc
can only tell us a rate fits itself; a different commanded curvature asks
whether the *model* generalises, which is the only question a controller cares
about. If held-out error stays inside the thresholds, the predictor is sound and
Phase 2 has a foundation. If it does not, the model is wrong and no amount of
further tuning on one arc would have revealed it.

Design notes, none of them authorized yet:

- **Freeze the constants in a committed file before the run.** If they are
  re-fitted afterwards the run proves nothing, which is the same failure as
  moving a threshold.
- A different angular also rotates at a different rate, which incidentally
  sharpens `alpha` -- but that is a bonus, not the purpose.
- Same safety envelope as today: fresh scan, verified corridor margins, frozen
  start with a 0.30 m drift abort, gate armed inside the `try` that disarms,
  explicit authorization for that single window.
- ⚠️ A tighter arc is a more aggressive manoeuvre needing its own corridor. The
  documented `angular 500` figure is a **stationary** turn rate; its arc
  behaviour is unmeasured, so it should not be assumed to scale.

## Open, and not resolved by this proposal

- The **negative `alpha`** is unexplained. It does not affect a prediction-based
  criterion, which is part of the argument for switching, but it remains an
  unexplained property of the telemetry.
- Whether **0.10 / 0.15 m** are the right thresholds. They are proposed from
  four steady-state intervals across two runs. That is thin, and the review
  should treat the numbers as provisional.
