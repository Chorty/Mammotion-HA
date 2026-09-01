# Predeclaration — long step at reduced linear speed (2026-09-01)

**Written before any run at this configuration exists.** Authorizes the CODE
CHANGE and, separately, ONE run when an operator authorizes it. Nothing else.

## 1. The problem this addresses

`docs/findings-plus180-split-is-onset-sampling-phase-20260901.md` showed the two
+180/7000 runs agree on the plant to **0.195 °/s** once the onset interval is set
aside, and that the whole 2a verdict split came from **interval 1 alone**
(-5.68 vs -1.35, a 4.33 °/s difference). 2a is therefore partly measuring where
the ~1 Hz VIO sample boundary fell relative to the step command.

Model, verified against the repeat (10.43/3.5 = 2.98 predicted vs 3.4049
observed): with a first half of `k` intervals containing the single onset
interval, `half_diff ≈ |steady − onset| / k`. At the worst observed
contamination (10.43 °/s) the 1.5 °/s bound needs **k ≥ 7**, i.e. ~14 informative
step intervals ≈ a **14 s step** at the measured ~1.02 s VIO cadence.

## 2. Why the obvious change was rejected

Raising `step_ms` alone **cannot work at `linear_speed` 400.** Measured
0.2616 m/s over the 15 s window on 2026-09-01:

| | value |
| --- | --- |
| largest square contained in "Backyard Right" | **10.0 m** → radius 5.0 m |
| therefore max travel | **4.5 m** |
| therefore max window at linear 400 | **~17.2 s** → step ≤ ~10.2 s |
| worst-case `half_diff` at that maximum | **2.09 °/s — still FAILS the bound** |

🔑 **The yard runs out before 2a becomes robust.** A step-length raise on its own
would have spent a release on a config that still fails roughly half the time.

## 3. What changed instead

**`linear_speed` 300 becomes admissible** (default stays 400). It was eliminated
on 2026-08-30 because a 0.116 m/s mower cannot produce the **0.15 m chord** the
RTK course statistic required — but **E-VIO reads VIO heading between consecutive
distinct readings and imposes no travel floor at all**
(`_step_response_vio_intervals`), so that objection does not transfer. At
0.116 m/s a 23 s window travels **~2.5 m**, needing only a ~5.9 m corridor.

Changes, stated plainly:

| | from | to |
| --- | --- | --- |
| `linear_speed` | pinned `[400]` | `[300, 400]`, default 400 |
| `step_ms` ceiling | 7000 | **15000** |
| `_STEP_RESPONSE_MAX_TOTAL_MS` | 16000 | **23000** |
| `max_travel_m` | 4.5 | **4.5 — UNCHANGED** |

🔑 **This raises the CLOCK, not the distance.** The exposure bound that actually
carries the safety is untouched, and at the slower speed the mower travels *less*
than any step run so far.

🆕 **New pre-dispatch refusal `step_window_travel_exceeds_budget`.** A long window
at linear 400 would overrun the travel guard mid-run — safe, but it aborts the
window and censors the measurement, wasting a supervised run. The check uses a
**lower-bound** speed table, so it fires only when the window cannot fit even at
the slowest speed measured; a merely marginal config still dispatches and the
guard keeps carrying the safety. ⚠️ A first version of this used a rounded-UP
speed and refused the schema's own defaults — pinned against by
`test_the_travel_refusal_does_not_tighten_the_existing_defaults`.

## 4. The run this authorizes, when an operator authorizes it

`baseline 3000 / step 15000 / settle 5000` (23000 ms), **`step_angular_speed`
+180**, **`linear_speed` 300**, `max_travel_m` 4.5, corridor a 10.0 m square
(far more than the ~5.9 m required — no reason to use a tighter one).

**Scoring is unchanged and is not to be edited afterwards**: rule E-VIO exactly as
shipped. Dark VIO refuses with `vio_not_live_throughout`; a dark run is
UNSCOREABLE by design, so this needs daylight throughout.

| outcome | reading |
| --- | --- |
| 2a PASS | The long step does what the model predicts. τ becomes computable — still n=1 at this config. |
| 2a FAIL with a small `half_diff` | Model right, margin thin; report the number, do not re-tune. |
| 2a FAIL with a large `half_diff` | The step is not steady even at 15 s — a finding about the plant, not the statistic. |
| guard trip / `vio_not_live_throughout` | FAIL. Not a smaller number. |

⚠️ **Registered before the run:** `linear_speed` 300 is a **different operating
point**, so results are NOT directly comparable to the four banked runs at 400,
and the rotation rate at +180 may itself depend on linear speed — that is
unmeasured. **Do not merge this run into the n-count of the 400 runs.**

## 5. What this authorizes

The code change, a release, and a motion-disabled deploy. **One** run at the
configuration in §4, with explicit per-run operator authorization, daylight, a
docked-and-charged battery, a fresh corridor scan at the live position, and the
gate disarmed and verified from the live API and RAW afterwards. **Standing
decision 5 is untouched — Phase 2 continuous steering remains parked.**
