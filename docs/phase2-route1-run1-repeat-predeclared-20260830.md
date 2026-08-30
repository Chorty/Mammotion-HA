# Route 1 run 1, repeat — predeclared before dispatch

**Written 2026-08-30, after run 1's FAIL and before any second capture exists.**
Nothing here may be edited once the repeat's data is in hand.

## Intent

Run 1 (`docs/evidence-route1-run1-fail-20260830.md`) completed its full window
cleanly for the first time but failed criteria 2a and 2b: the last two step
rates were 2.49°/s apart and the last two settle rates were 2.07°/s apart,
both against the 1.5°/s bound. n = 1. This repeat exists to find out whether
that near-miss reproduces or was a one-off, **before** deciding whether the
criteria, the phase lengths, or nothing at all needs to change.

## What does NOT move

* **Every parameter is identical to run 1**: baseline 3000 / step 5000 /
  settle 5000 ms, `step_angular_speed=120`, `max_travel_m=4.0`,
  `linear_speed=400`.
* **Criteria 2a/2b are unchanged**: ≥3 informative intervals and the last two
  rates within 1.5°/s, in both the step and settle phases.
* The corridor is re-scanned and re-verified at the mower's live position
  immediately before dispatch, per standing practice — never reused from run
  1's placement.

## What this run does and does not settle

* **If it fails the same way** (a comparable near-miss, comparable
  magnitude): that is evidence the near-miss is a property of this
  measurement at this chord length, not a fluke of run 1 — still n = 2, not
  a distribution, but a real hint.
* **If it passes**: run 1's failure was noise on one run, not a structural
  issue. A single pass here does not retroactively make run 1 "actually
  fine" — it would mean criteria 2a/2b are achievable at this configuration,
  and the predeclaration's own "what a pass authorizes" (run 2 at +180) would
  apply to *this* run's pass, not to run 1.
* **Neither outcome licenses touching `step_ms`, `settle_ms`, or the 1.5°/s
  bound in this same sitting.** Any criterion or parameter change is a
  separate, deliberately-written decision made before the next dispatch, not
  a same-run reaction to this repeat's data.

## Safety

Unchanged from the original predeclaration
(`docs/phase2-route1-predeclared-20260830.md`): 15/15 gates and `blockers: []`
required before dispatch, explicit operator confirmation of on-site
supervision and a clear area immediately before dispatch, gate disarmed and
verified from both the live API and RAW `core.config_entries` afterward.
