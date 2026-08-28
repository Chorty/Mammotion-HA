# Should the steering refusal be opened? — scored 2026-08-26

Offline analysis only. **No motion was commanded and no code was changed to
produce this.** The refusal at `services.py` →
`steering_not_motion_validated` is still in place.

**Recommendation: YES, open it — but not yet, and not at the current bounds.**
Three conditions below, in order. The reasoning is that the 2026-08-24 failure is
now a *fully explained discrete bug* rather than a tuning problem, and every
safety guard demonstrably worked while it failed.

## 1. The 2026-08-24 run, re-scored against the registered criteria

Criteria are from `docs/continuous-motion-feasibility-plan-20260821.md` →
"Phase 2 pass criteria", written before any Phase 2 code existed.

| Criterion | Result |
| --- | --- |
| No intermediate stop before final/abort stop | PASS |
| Heading error and cross-track trend to zero | **FAIL** |
| No oscillation between saturated ±180 commands | PASS |
| Cross-track never exceeds 0.20 m | **FAIL** |
| 0.30 m hard abort never fires | **FAIL** |
| Motion duty cycle at least 80% | PASS |
| Final stop confirmed and gate disarmed | PASS |

**Verdict stands: FAIL.** This document does not relitigate that.

## 2. The failure is a single explained bug, and the trace proves it

The per-decision record is unambiguous:

```
t_s    cross_m    head_err   desired    angular  reason
0.46    0.0000     46.64     -34.00     180      tracking_route
0.96    0.0000     46.64     -34.00     180      tracking_route
1.28   -0.0163     48.25     -32.39     180      tracking_route
2.19   -0.0163     48.25     -32.39     180      tracking_route
2.29   -0.1867     77.40      -9.65     180      tracking_route
3.00   -0.1867     77.40      -9.65     180      tracking_route
3.33   (abort)                            0      cross_track_limit_reached
```

🔑 **Heading error GREW monotonically — 46.64 → 48.25 → 77.40 — while the command
sat saturated at +180 the entire time.** That is the signature of positive
feedback, not of a weak or mistuned gain. A correctly-signed loop with any
positive gain reduces the error; this one accelerated it.

Both defects fixed on 2026-08-24 bear directly on this exact trace:

* **Defect 2 (inverted sign)** explains the divergence itself. With
  `error = desired − current`, commanding **+180** drove map course further from
  the target, because positive angular *decreases* map course
  (`map = 90.13 − toward`, and positive angular raises `toward`). The corrected
  command would have been **−180**. The sign relationship is confirmed against
  **six banked captures on both command signs, zero contradictions**.
* **Defect 1 (stale `toward` at open)** explains the 46.64° starting error. The
  run opened steering immediately against a heading that had not been confirmed
  by this window's own motion. The current state machine holds
  `angular_speed: 0` until a fresh ≥0.15 m position chord exists, so **this run
  could not start the same way today.**

⚠️ **Neither fix has moved a wheel.** They are verified by offline replay and by
the banked sign evidence, not by hardware. That is the whole reason for
condition A below.

## 3. Every safety guard worked while the control law failed

This is the strongest argument for proceeding, and it is measured rather than
asserted:

| Guard | Behaviour during the failure |
| --- | --- |
| Corridor containment | `inside_area` **true on every decision** |
| Position validity | `position_valid` true on every decision |
| BLE liveness | `ble_live` true throughout |
| Refresh cadence | max gap **0.520 s** against a 0.600 s bound |
| Cross-track hard abort | fired correctly at 0.517 m travelled, 3.33 s |
| Stop | confirmed, `ok: true` |
| Gate | disarmed and verified afterwards |

**The control law failed and the safety envelope contained it exactly as
designed**, at 0.517 m of travel. That is what a guard layer is for, and it is
the evidence that a second attempt is survivable.

## 4. New evidence since the gate doc was written

🔑 **The turn quantum varies 2.6× on identical parameters** (measured 2026-08-26:
21.8 then 57.0 °/pulse — `docs/reach-closed-at-6m-20260826.md`).

🗑️ **CORRECTED 2026-08-26, same day: an earlier draft of this section let that
number stand as if it bounded steering response. It does not, and quoting it that
way is a category error.** Both measurements are **stationary in-place pivots**
(linear 0, angular 500). The continuous controller steers **while moving**, which
is the arc regime, and this project's own record says the two behave differently:
"angular needs 500" is explicitly a **stationary-only** finding and angular 180
actuated fine in an arc, while the 2026-08-12 arc measurement was clean and
linear — `linear 400 + angular 180` rotated course **+22.20°** over 0.5823 m
against **+0.00°** for the zero-angular control, the two distances 1.8 mm apart.
**Do not transfer the 2.6× figure to arc steering.**

🔑 **What DOES carry over is the cause, not the number.** The registered
explanation is BLE-cadence gating: a pulse rotates only while refresh writes
arrive, and a blocked write lets the watchdog stop the motor while the executor
still divides by the whole commanded window. That mechanism is transport-level
and regime-independent, so it applies while moving too — the 2026-08-09 data
shows the same shape (cadence-intact pulses 23–43 °/s, a stalled one 9.23 °/s).
The controller already guards it: `refresh_max_gap_since_last_decision_s` bounded
at 0.60 s, which held at **0.520 s** on 2026-08-24.

**Why none of this is disqualifying:** the continuous controller re-anchors on
**measured heading every ~1 Hz step** rather than integrating a yaw-rate model.
It does not need to know how much rotation a command produces — it commands,
measures, corrects. That is exactly why the 2026-08-22 out-of-sample work
concluded a continuous controller needs accurate heading *feedback* rather than
an accurate yaw *model*, taking yaw calibration off the Phase 2 critical path.

**What it changes, concretely:**

1. **Never tune the gain against a predicted rotation rate.** Plant gain is
   uncertain, so any `K` derived from a °/s figure rests on an unstable number.
2. **It strengthens condition B below.** With uncertain plant gain, a
   proportional loop tuned for a sluggish response overshoots when the response
   is brisk — and "no oscillation between saturated ±180 commands" is a
   registered pass criterion.
3. **It adds a required output for the first steering run.** Nobody has measured
   the actual rotation response per commanded angular **while moving under
   refresh**; the arc data is two points at a single speed. Record it as a
   result of the run rather than assuming it going in.

⚠️ **The structural risk is the feedback interval, and it is not a bug.** At the
configured 0.2482 m/s nominal speed, one ~1 Hz correction interval is **0.248 m
of blind travel against a 0.30 m hard abort budget**. A single badly-aimed
interval can consume most of the margin. This is the real limit on continuous
steering and no defect fix changes it.

🔑 **The gain saturates at only 15° of error**
(`angular_speed_per_heading_degree: 12.0`, `max_abs_angular_speed: 180`). The
failed run began at 46.64°, so it was saturated from the first decision and had
no proportional behaviour to demonstrate at all.

## 5. Recommendation — three conditions, in order

✅ **CONDITION A IS NOW SATISFIED — 2026-08-27, beta79.** The real
`heading_acquisition_window` ran and returned **`heading_acquired`**. Defect 1's
fix is hardware-validated: **both decisions commanded `angular_speed: 0`**
(`acquiring_heading` then `heading_acquired`), so the opening never steered
against an unconfirmed heading — the exact thing the 2026-08-24 run did wrong.
Heading came from a **0.4667 m position chord** (3.1x the 0.15 m floor) at
**0.538°** uncertainty, giving course **278.55°** against a pre-run two-way
estimate of 276.96° — **1.59° apart**. Travel was 0.4667 m inside the 1.06 m
blind disk, post-stop observation returned clean (`wait_reason: null`), stop
confirmed, `blockers: []` throughout, gate disarmed and verified from live API and
RAW storage. Evidence:
`docs/evidence-phase2-acquisition-beta79-20260827.json`.
⚠️ **Defect 2 remains UNTESTED.** This service is `acquisition_only` and never
reaches the steering path. Condition A is met, not bypassed.

**A. Run the acquisition test first, and require it to pass.**
`heading_acquisition_window` with `dry_run: false` exercises defect 1's fix, the
fresh-origin requirement, the zero-angular opening, the stop, and the 3.5 s
stopped observation — **without enabling steering**. It commanded ~0.56 m inside the then-1.06 m disk; **the disk is 1.34 m since the
2026-08-27 budget change**. If the opening state machine misbehaves, it surfaces here at a
fraction of the exposure. The 2026-08-24 corridor blocker is gone: the mower's
current open-lawn position has **5.07 m** clearance against the disk (1.06 m at
the time; **1.34 m** since the 2026-08-27 budget change — still ample).

**B. Open the refusal with tightened bounds for the first steering run, not the
current ones.** Specifically:

* **Cap `max_abs_angular_speed` far below 180** — 60 is a reasonable first value.
  The failed run saturated at 180 on every decision; capping it means a residual
  sign or gain error cannot accelerate divergence faster than the 1 Hz loop can
  observe it. This is the single highest-value de-risking change.
* **Shorten the window** below the 4 s used on 2026-08-24, bounding total
  exposure per attempt.
* **Keep the 0.30 m hard abort and the corridor override unchanged.** They
  worked; do not touch them.
* Require an opening alignment well inside the 15° saturation threshold, so the
  run actually demonstrates proportional control rather than saturation.

**C. Predeclare the scoring before the run.** Score against the same seven
criteria in §1, written down before dispatch, exactly as the 2026-08-24 run did.
That run's honesty about its own failure is why this analysis was possible.

## What would change this recommendation

* The acquisition test failing, or the opening state machine not holding
  `angular_speed: 0` as designed → stop, do not open steering.
* Battery, VIO, or BLE not healthy at run time → defer; nothing here is urgent.
* Any wish to test steering *without* condition A → decline. Defect 1's fix
  governs how the window opens, and opening is where the 2026-08-24 run went
  wrong first.

⚠️ **This document authorizes nothing.** Opening the refusal is a code change
plus a release plus a supervised physical run, each needing its own operator
authorization.
