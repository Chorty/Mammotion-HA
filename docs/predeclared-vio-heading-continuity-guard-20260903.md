# PREDECLARED — a VIO heading-continuity guard for rule E-VIO (2026-09-03)

**Written and committed BEFORE implementation**, because this changes *when the
criterion refuses* and the project's rule is that a scoring change is registered
before anyone can see which verdicts it moves. §5 states, in advance, every
banked verdict it is expected to change.

Motivating defect: `docs/evidence-linear300-angular180-heading-discontinuity-20260903.md`.
Plan item: `docs/NEXT-PLAN-20260903.md` §1a.

**This authorizes no mower run.** It is an offline scoring change.

---

## 1. The defect, stated precisely

On 2026-09-03 a run returned `scoreable: true` while the mower drove **straight**
(operator-observed; RTK course moved ~6° across the whole window). At a single
report, VIO heading jumped **−166.47°** and `toward` jumped **+69.12°** — different
magnitudes at the same instant, so no physical rotation explains it. Most
plausibly VIO re-referencing after a mower restart.

🔑 **`vio_state` stayed `2` for all 79 samples, so `vio_not_live_throughout`
never fired: the only liveness guard checks STATE, not CONTINUITY.**

⚠️ **Scope the harm honestly — it is narrower than "a bogus number was scored",
and it is still real.** On *that* run the step phase was 1000 ms, yielding
**1 informative step interval against the rule's ≥3**, so 2a returned
`half_diff: None` and ω/τ came back `None`. What the run *did* return was
`scoreable: true`, a **2b PASS** computed over a discontinuous heading track,
and an `intervals` table publishing **−149.794 °/s** as a rate.

🚨 **The reason this must be fixed rather than noted: the jump was harmless only
because the step was too short to score.** At any step length the programme
actually uses (5000–7000 ms, ≥3 intervals) the same jump lands *inside* the
half-phase statistic and is converted into a certified rotation rate. The guard
exists for the next run, not the last one.

---

## 2. What the statistic is

For every interval already produced by `_step_response_vio_intervals` — that is,
between **consecutive distinct** VIO headings, wrap-normalized — take

```
|rate_deg_per_s| = |normalize_degrees(h1 - h0)| / dt_s
```

and refuse to score the run when **any** interval, in **any** phase, exceeds a
fixed bound.

**Refuse the whole run, do not drop the interval.** Two reasons:
1. A frame discontinuity means the headings before and after it are not in one
   frame, so *every* later interval is referenced to a shifted origin. The run's
   heading track is not one signal; no statistic over it is meaningful.
2. Dropping the offending interval and re-scoring the rest is precisely the
   "just exclude the onset interval" move this project rejected on 2026-09-01,
   for the same reason: choosing which samples to discard *after* seeing which
   verdicts it flips is the failure the mirror-criterion review of 2026-08-23
   registered against.

---

## 3. 🗑️ The shape the plan proposed does NOT work, and this is why

`docs/NEXT-PLAN-20260903.md` §1a proposed refusing when the delta "exceeds what
**the commanded angular rate** can physically produce over that dt". **Measured
against the banked corpus before implementation, that bound refuses every clean
run.**

The commanded angular is **zero** during baseline and settle. But rotation
persists well past the command going to zero — that is the *entire subject* of
the step-response programme (17° of rotation after zero, measured 2026-08-29).
Worst per-phase clean interval across the eight banked runs:

| phase | commanded angular | worst clean interval |
| --- | --- | --- |
| baseline | 0 | 0.86 °/s |
| step | 120 or 180 | **15.35 °/s** |
| settle | **0** | **9.97 °/s** |

🔑 **A command-scaled bound would refuse a 9.97 °/s settle interval as
"impossible" when it is the plant doing exactly what the experiment is designed
to observe.** The bound must therefore be the **plant's envelope** — the fastest
the machine rotates under *any* admissible command, carried through the decay —
not the instantaneous command.

---

## 4. The bound, and where it comes from

```
_STEP_RESPONSE_VIO_MAX_PLAUSIBLE_RATE_DEG_PER_S = 30.0
```

Derivation, from measurement only:

| input | value |
| --- | --- |
| fastest steady rotation measured, any admissible command | **13.431 °/s** at (linear 300, angular 180) |
| fastest single clean interval, all 8 banked runs, all phases | **15.35 °/s** |
| the observed discontinuity | **149.79 °/s** |

The probe's schema admits `linear_speed ∈ {300, 400}` and
`|step_angular_speed| ∈ {120, 180}` only, so 13.431 °/s is the top of the
*measured* envelope and 15.35 °/s the top of the *observed per-interval* one
(transient onset plus reading noise).

**30.0 °/s is ~2.0× the worst clean interval and ~2.2× the fastest steady
rotation, while sitting 5.0× below the observed discontinuity.** The ~10×
separation between "clean" and "broken" is what makes this bound uncritical: any
value in roughly 20–70 °/s produces identical verdicts on the entire banked
corpus. **It is deliberately NOT tuned to the gap's midpoint** — it is placed
just above the physical envelope with a doubling of margin, so it stays
defensible if a future operating point rotates somewhat faster than any measured
today.

⚠️ **This bound is NOT a claim about the plant.** It is a plausibility ceiling
for an instrument-fault detector. Do not quote 30 °/s as a rotation capability,
and do not fit anything to it.

⚠️ **Stationary pivots rotate far faster** (up to ~57°/pulse; the affine night
fit reaches ~38 °/s at angular 500). Those commands are **not admissible to this
probe**, which always drives forward at linear 300/400. If the schema is ever
widened to angular 500 or to in-place pivots, **this bound must be re-derived
first** — a test pins that coupling.

---

## 5. What this changes on the banked corpus — declared in advance

**Predicted: exactly one run changes, and no verdict flips from FAIL to PASS.**

| banked run | before | after (predicted) |
| --- | --- | --- |
| `raw-route1-run1-plus120-step5000-20260830` | 2a FAIL 2.156, 2b PASS | **unchanged** |
| `raw-route1-run1repeat-plus120-step5000-20260830` | 2a FAIL 3.664, 2b PASS | **unchanged** |
| `raw-route1-stepext-plus120-step7000-20260830` | 2a FAIL 2.319, 2b PASS | **unchanged** |
| `raw-route1-run2-plus180-step7000-20260830` | 2a PASS 0.130, 2b PASS | **unchanged** |
| `raw-route1-run2repeat-plus180-step7000-20260901` | 2a FAIL 3.4049, 2b PASS | **unchanged** |
| `raw-phaseA-linear300-speed-20260903` | scoreable, 2a n=1 | **unchanged** |
| `raw-linear300-angular120-step5000-20260903` | 2a FAIL 2.8364, 2b PASS | **unchanged** |
| `raw-linear300-angular180-step5000-20260903` | 2a FAIL 4.3966, 2b PASS | **unchanged** |
| 🚨 `raw-linear300-angular180-20260903` | **`scoreable: true`**, 2b **PASS** | **`scoreable: false`, `vio_heading_discontinuity`** |

🔑 **The adoption argument is an asymmetry, and it is the opposite of Rule D's.**
Rule D was denied on 2026-09-03 because it was *more permissive by construction*
— excluding the onset makes any run likelier to pass, and it flipped two banked
FAILs to PASSes. **This guard is more RESTRICTIVE by construction:** it can only
move a run from scoreable to unscoreable. It cannot turn a FAIL into a PASS, so
it cannot manufacture a result, and adopting it costs no in-sample selection.

---

## 6. Acceptance criteria — all must hold, checked after implementation

1. All **eight** clean banked runs keep `scoreable: true` and every published 2a
   `half_diff`, 2a/2b verdict, ω and τ **byte-identical** to §5's "before".
2. The defective run returns `scoreable: false` with
   `unscoreable_reason: "vio_heading_discontinuity"`.
3. The refusal names what tripped it — the offending interval's endpoints,
   `dt`, rate and phase — so it is never silent and is diagnosable from the
   response alone.
4. A synthetic run with a clean 13.4 °/s rotation throughout still scores.
5. Every file in `docs/raw-samples/` is pinned or excused
   (`test_every_banked_run_is_pinned_or_explicitly_excused` — **currently RED on
   `main`**, see §8).
6. A test pins the bound to the admissible command envelope, so widening the
   schema without re-deriving it fails.

**If criterion 1 fails, the guard is wrong and does not ship.** A guard that
moves an existing verdict is doing something other than detecting frame jumps.

---

## 7. ⚠️ What this guard does NOT do — stated so nobody over-trusts it

- **It catches gross re-referencing, not small frame shifts.** A jump under
  30 °/s (≈30° at the ~1 Hz reading cadence) passes and is scored. There is no
  known instrument that would catch those, and inventing a tighter bound would
  start refusing real rotation.
- **It does not detect drift** — a slow frame rotation stays invisible.
- **It does not repair a run.** The output is "unscoreable", so a discontinuity
  still costs a supervised dispatch. That is the correct price; the alternative
  is a certified number that is not a measurement.
- **It says nothing about `toward`.** The `toward` channel jumped too, by a
  different amount; that is a separate unexplained observation and no guard here
  touches it.

---

## 8. Related, found while measuring this — a live RED test on `main`

`test_every_banked_run_is_pinned_or_explicitly_excused` **fails on `main` at
`b9d62007`**: the three 2026-09-03 raw-sample files were committed without being
added to the regression roster. **The guard test worked exactly as designed** —
it was written on 2026-09-03 after a run sat unpinned for two days, and it caught
the next three the same week. Pinning them is part of this change (§6 criterion
5), and the defective run gets pinned as the *unscoreable* fixture.

---

## 9. What this authorizes

**Nothing on the mower.** Phase 2 stays parked (standing decision 5). This does
not revive the 2a line, does not license a longer step, and does not change any
bound the probe enforces at dispatch. It changes only whether E-VIO is willing
to call a heading track scoreable.
