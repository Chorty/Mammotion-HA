# The VIO heading-continuity guard is implemented, and the predeclared table held exactly (2026-09-03)

Predeclared in `docs/predeclared-vio-heading-continuity-guard-20260903.md`
(commit `36922533`) **before any code was written**. This records the outcome
against that document's §5 prediction and §6 acceptance criteria.

**Offline scoring change. No mower run. Phase 2 stays parked.**

---

## 1. Result

✅ **All six acceptance criteria pass, and the §5 prediction held with no
adjustment.** Exactly one banked run changed, in the declared direction:

| banked run | before | after |
| --- | --- | --- |
| the five route-1 runs | 2a 2.156 / 3.664 / 2.319 / 0.130 / 3.4049 | **identical** |
| `raw-phaseA-linear300-speed-20260903` | scoreable, 2a n=1 | **identical** |
| `raw-linear300-angular120-step5000-20260903` | 2a FAIL 2.8364, 2b PASS | **identical** |
| `raw-linear300-angular180-step5000-20260903` | 2a FAIL 4.3966, 2b PASS | **identical** |
| 🚨 `raw-linear300-angular180-20260903` | `scoreable: true`, 2b **PASS** | **`scoreable: false`** |

The refusal names the offending interval: **−149.7945 °/s**, phase `step`,
`89.769 → −76.705`, against the 30.0 °/s bound.

🔑 **Every published 2a `half_diff`, every 2a/2b verdict, and every ω/τ on the
eight clean runs is byte-identical.** That was acceptance criterion 1, and it was
written as a *falsifier*: a guard that moved an existing verdict would be doing
something other than detecting frame jumps, and would not have shipped.

---

## 2. The two things measurement changed before a line was written

### 2a. 🗑️ The shape the plan proposed refuses every clean run

`docs/NEXT-PLAN-20260903.md` §1a proposed bounding the delta by **what the
commanded angular rate can produce**. Measured against the corpus first:

| phase | commanded angular | worst clean interval |
| --- | --- | --- |
| baseline | 0 | 0.86 °/s |
| step | 120 / 180 | 15.35 °/s |
| settle | **0** | **9.97 °/s** |

**Rotation persists past the command going to zero — that decay is the entire
subject of the step-response programme** (17° after zero, measured 2026-08-29).
A command-scaled bound calls the plant's own decay impossible and refuses every
run on its settle intervals.

🔑 **The transferable form: the bound must be the PLANT's envelope, not the
instantaneous command.** A guard against an instrument fault has to be tolerant
of everything the machine can legitimately do, including the parts the
experiment exists to observe.

### 2b. The bound is uncritical, which is worth saying out loud

| | °/s |
| --- | --- |
| fastest steady rotation, any admissible command | 13.431 |
| fastest single clean interval, 8 banked runs, all phases | 15.35 |
| **the bound** | **30.0** |
| the observed discontinuity | 149.79 |

**~10× separation between clean and broken.** Anything in roughly 20–70 °/s
scores the entire banked corpus identically, so no verdict here rests on a tuned
threshold. 30.0 sits just above the physical envelope with a doubling of margin
rather than at the gap's midpoint, so it survives an operating point that
rotates somewhat faster than any measured today.

---

## 3. Why this was adoptable when Rule D was not

🔑 **The asymmetry runs the opposite way, and that is the whole argument.**

Rule D was **denied** on 2026-09-03 because it was *more permissive by
construction* — excluding the onset makes any run likelier to pass, and it
flipped two banked FAILs to PASSes after one round of in-sample selection.

This guard is **more restrictive by construction**: its only possible effect is
scoreable → unscoreable. **It cannot turn a FAIL into a PASS, so it cannot
manufacture a result**, and adopting it involves no selection among rules that
were compared on verdicts already seen.

⚠️ **That is not a general licence.** "Restrictive" made the *adoption* honest;
it did not make the *bound* right. The bound is right because it was derived
from measurement and validated against a falsifier declared in advance.

---

## 4. What the defect actually cost, stated narrowly

⚠️ **Do not repeat "a bogus rotation rate was scored as a measurement" — it
overstates what happened.** On that run the 1000 ms step yielded **1 informative
interval against the rule's ≥3**, so 2a returned `half_diff: None` and ω/τ came
back `None`. The guards that *did* hold, held.

What the run returned was `scoreable: true`, a **2b PASS** over a discontinuous
track, and an `intervals` table publishing −149.794 °/s as a rate.

🚨 **The jump was harmless only because the step was too short to score.** At the
5000–7000 ms steps the programme actually uses, the same jump lands inside the
half-phase statistic and becomes a certified rotation rate. **The guard exists
for the next run, not the last one.**

---

## 5. ⚠️ What it does not do

- **Catches gross re-referencing, not small frame shifts.** A jump under 30 °/s
  (≈30° at the ~1 Hz reading cadence) passes and is scored. No known instrument
  catches those, and a tighter bound would start refusing real rotation.
- **Does not detect drift.** A slow frame rotation stays invisible.
- **Does not repair a run.** A discontinuity still costs a supervised dispatch.
  That is the correct price; the alternative is a certified non-measurement.
- **Says nothing about `toward`**, which jumped by a different amount at the same
  instant. That remains unexplained and no guard here touches it.
- 🚨 **It is valid only for this probe's command envelope** (linear 300/400,
  |angular| 120/180, always driving forward). Stationary pivots reach ~38 °/s at
  angular 500 and would trip it.
  `test_the_continuity_bound_is_tied_to_the_admissible_commands` reads the
  schema and fails if either set widens without a re-derivation.

---

## 6. A live RED test on `main`, found while doing this

`test_every_banked_run_is_pinned_or_explicitly_excused` **was failing on `main`
at `b9d62007`** — the three 2026-09-03 raw-sample files were committed without
being added to the regression roster.

✅ **The guard test worked exactly as designed.** It was written on 2026-09-03
after the 2026-09-01 repeat sat unpinned for two days, specifically because the
roster was a hand-typed literal; it caught the next three files the same week.

Both 5000 ms runs are now pinned with their verdicts (2a FAIL at 2.8364 and
4.3966 — a 5000 ms step is still ramping at linear 300 too, exactly as the
400-series 5000 ms runs were). The discontinuity run is excused from the 2a
roster **with its reason**: it has no verdict to pin *because* the guard refuses
it, and it is pinned instead as the refusal fixture.

🔑 **Method note: this failure was found by running the suite, not by reading the
diff.** It was invisible to review — nothing in the 2026-09-03 commits looked
wrong; a file simply existed that a globbing test could see.

---

## 7. What this authorizes

**Nothing on the mower.** It does not revive the 2a line, license a longer step,
or change any bound the probe enforces at dispatch. It changes only whether
E-VIO is willing to call a heading track scoreable. Standing decision 5 is
untouched.
