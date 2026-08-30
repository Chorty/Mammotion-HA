# Route 1 run 1, repeat — FAIL again, but a sharper picture

**2026-08-30, beta87.** Second run at run 1's exact configuration
(`docs/phase2-route1-run1-repeat-predeclared-20260830.md`), n=2 on
baseline 3000 / step 5000 / settle 5000 ms, `step_angular_speed=120`,
`max_travel_m=4.0`. Raw evidence:
`docs/evidence-route1-run1-repeat-fail-20260830.json`. Compare against the
first run: `docs/evidence-route1-run1-fail-20260830.md`.

## Verdict: FAIL

Criterion 2 (2a — step reaches steady rotation) fails again, worse than run 1.
Criterion 3 (2b — settle goes flat) **passes** this time.

| # | criterion | run 1 | run 1 repeat |
| --- | --- | --- | --- |
| 1 | report stream ready | ✅ PASS | ✅ PASS |
| 2 | 2a — last two step rates ≤1.5°/s | ❌ 2.49°/s apart | ❌ **7.28°/s apart — worse** |
| 3 | 2b — last two settle rates ≤1.5°/s | ❌ 2.07°/s apart | ✅ **0.26°/s apart — clean plateau** |
| 4 | containment + stop confirmed | ✅ PASS | ✅ PASS |
| 5 | travel guard does not trip | ✅ PASS | ✅ PASS |
| 6 | gate disarmed after, verified | ✅ PASS | ✅ PASS |

## What the pair now suggests

**The settle phase looks adequate.** Run 2's last three settle intervals held
course nearly flat — 57.635° → 57.810° → 57.709° — a genuine plateau, not
noise decaying toward one. That is a real pass, not a near-miss.

**The step phase looks insufficient, in both runs.** Both times, the step
phase's *final* interval showed the rotation rate *increasing* in magnitude
relative to the interval before it:

* run 1: -5.686 → -8.179 °/s
* run 2: -3.828 → -11.108 °/s

That is the signature of a rotation still accelerating through onset lag when
the step phase ends, not chord noise decaying around a settled value — which
was my working hypothesis after run 1 alone. It matches the predeclaration's
own arithmetic (§3 of `docs/phase2-route1-predeclared-20260830.md`): onset lag
~2 s means a step needs ~5–6 s to reach steady rotation, and this step
commands for exactly 5 s.

⚠️ **n = 2. This reframes the working hypothesis; it does not confirm one.**
Do not fit a rate law to either run, and do not treat "step is too short" as
established — it is now the leading explanation, not a conclusion.

🛑 **Per the repeat's own predeclaration, this does not license touching
`step_ms`, `settle_ms`, or the criteria in this sitting.** Any parameter or
criterion change is a separate, deliberately-written decision made before the
next dispatch.

## The repositioning drive (before this run)

Run 1 ended at `(7.0501, -7.7336)`, where the area-boundary clearance was only
**3.467 m against the 4.50 m required disk (0.77x)** — too tight for a fresh
9.0 m corridor there. The mower had to be moved back toward the verified spot
near `(5.98, -5.24)` before the repeat could be dispatched.

Used `raw_pymammotion_execute_vector_segment` (the accepted closed-loop reach
profile, `docs/accepted-profile.json`, verified key-by-key identical before
real dispatch). Two calls were needed:

1. The required turn was 163.5°, staged into ≤60° segments per the executor's
   existing staged-turn logic. Two 60° stages completed
   (`target_heading_reached`), then the next stage's own feasibility check
   refused on `turn_budget_infeasible` — a narrow margin on the projected
   translation for that stage (0.156 m estimated against a 0.148 m budget).
   **Zero linear commands were sent, no error occurred, and the mower sat
   stationary and `valid_for_motion` throughout** — this stopped safely, it
   did not fail unsafely.
2. A second call, reusing the VIO calibration offset from the first call
   (`vio_heading_offset_degrees`, skipping a redundant forward calibration
   drive), completed the remaining turn and drove to `target_reached` at
   **0.149 m** from `(5.98, -5.24)` — inside the 0.15 m tolerance.

Final position `(5.9796, -5.3895)`, re-scanned boundary clearance **5.822 m
(1.29x)** against the 4.50 m disk before the step-response dry run.

## Safety

15 of 15 gates before dispatch, `blockers: []`. Every sample stayed inside the
corridor. Stop confirmed (`ok: true`, `ack.movement_ok: true`). Explicit
operator confirmation taken immediately before both the repositioning drive
and the step-response dispatch. Gate disarmed and verified from both the live
API and RAW `core.config_entries` after each dispatch.

⚠️ Battery ranged 43–48% across this session, draining and not charging —
below the predeclaration's docked-and-charged precondition. The operator
explicitly authorized proceeding anyway.

## The reason-field bug, again

This host still runs beta87, which has the pre-fix `reason` logic
(commit `af5f547f` is committed but not deployed). The service again reported
`"reason": "travel_guard_tripped"` for a run that in fact completed cleanly —
confirmed the same way as run 1: `motion_refresh.aborted_early` is `False`, 0
of 128 samples carry `travel_guard_tripped: true`, and the window ran its full
13001 ms of a 13000 ms schedule. Deploying the fix would make this field
trustworthy for the next run without changing the analysis here, which is
drawn from the raw samples regardless.

## What this does not establish

* `tau_actuator_s = 2.206 s` from this run is not a settled value either —
  2a still fails.
* n = 2 total on this configuration, both FAIL on 2a. Not a distribution.
* Whether a longer step phase would pass 2a is the leading hypothesis, not a
  proven fact, and `step_ms` stays at its predeclared cap until a separate,
  deliberate decision changes it.

## What a pass would have authorized

Unchanged from run 1: writing the feed-forward design document. Neither run
passed, so nothing further is authorized beyond recording both results.
