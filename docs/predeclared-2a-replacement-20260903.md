# Predeclared — candidate replacements for criterion 2a (2026-09-03)

**Registered BEFORE any candidate's verdicts are computed.** The git timestamp of
this commit is what makes the findings document trustworthy; the findings commit
will name this commit's hash. Structure follows
`docs/predeclared-rtk-vio-course-rate-scoring-20260831.md`, which is the
precedent this project already accepted.

**Offline. Nothing here authorizes motion, touches the gate, or changes scoring
code.**

---

## 0. Shared definitions — so no rule can win by quietly redefining a term

- **Interval extraction:** `_step_response_vio_intervals(samples, baseline_ms=,
  step_ms=)` as shipped. Consecutive DISTINCT VIO headings; latched repeats
  produce no interval.
- **Phase assignment:** by interval MIDPOINT against the nominal boundaries from
  each raw file's own `phases`. Not re-derived, not re-tuned.
- **Angle wrap:** `normalize_degrees`, to [−180, 180).
- **Rate:** `normalize_degrees(h1 − h0) / dt_s`.
- 🔑 **The 1.5 °/s bound and the ≥3-informative-interval requirement are NOT
  moved by any rule below.** A rule that needs a different bound to look good is
  a different criterion, not a replacement.
- **Roster:** every file in `docs/raw-samples/` except
  `raw-phaseA-linear300-speed-20260903.json`, which is excused because a 1000 ms
  step yields ~1 informative step interval against the ≥3 requirement — 2a is
  unscoreable there BY DESIGN.
- **Completion** comes from `motion_refresh.aborted_early`, never the `reason`
  field, which reads `"travel_guard_tripped"` wrongly on the two 5000 ms files.
- **Harness:** `scripts/rescore_2a_candidates.py`, which imports the shipped
  functions rather than reimplementing them.

### 0.1 🗑️ A correction to this project's own framing, made before scoring

This programme has repeatedly said 2a is **noise-dominated**. Measured, that is
**wrong**, and the correction matters because it changes what a good replacement
must do.

Pooled from the 12 ex-onset step intervals of the two +180/7000 runs (one
configuration, so the plant is held constant): rate sd **1.445 °/s**, implying
a per-reading heading noise **σ_h ≈ 1.00°**.

| statistic | 2σ noise | vs the 1.5 °/s bound |
| --- | --- | --- |
| shipped endpoint-difference halves, 7 s step | **1.145** | **admissible** |
| shipped endpoint-difference halves, 15 s step | **0.534** | admissible |
| a mean-of-interval-rates variant, k = 7 | 1.544 | **inadmissible** |

🔑 **The shipped statistic is admissible. 2a fails on BIAS, not variance.** The
onset interval depresses the first half's endpoint rate by
`2·Δ·dt / T` — about **2.9 °/s at a 7 s step** and **1.36 °/s at 15 s**, against
the 1.5 bound, where Δ ≈ 10.43 °/s is the measured onset deficit. Both bias and
noise scale as **1/T**, so a longer step cannot change their ratio.

**Consequence for this study: the target is the onset bias.** A rule that merely
reduces variance is not addressing the failure.

---

## 1. Rule A — the status quo, and a predeclared STOP condition

Rule A is the shipped `_step_response_half_phase_agreement(intervals, "step")`:
a time-weighted **endpoint difference** per half, boundary reading shared.

**Reproduction gate.** Rule A must reproduce every shipped verdict a raw file
carries, to ±0.01 °/s, and must match the five statistics pinned in
`tests/components/mammotion/test_step_response_vio_scoring.py`
(2.156 / 3.664 / 2.319 / 0.130 / 3.4049). **If it does not, stop and report — do
not proceed to preference judgements built on a status quo I cannot reproduce.**

---

## 2. Rule B — drop the first step interval, then Rule A

Exclude `step_intervals[0]` and apply Rule A unchanged to the remainder.

🚨 **DISCLOSURE, in the spirit of §2 of the 2026-08-31 study: this rule's
outcomes are ALREADY PUBLISHED and I cannot claim blindness.** It gives
**1.625 FAIL** on the banked +180 run and **0.107 PASS** on the repeat — it flips
the banked run's published PASS. It is registered anyway because excluding it
would be its own kind of selection, but **it starts at a disadvantage under §7
criterion 4** and must not be preferred on a thin margin.

**Rejected if:** it passes either 5000 ms anchor; or its 2σ noise exceeds 1.5 °/s
at the reduced interval count.

---

## 3. Rule C — exclude a declared ONSET WINDOW, then Rule A

Exclude every step interval whose midpoint falls within
**`onset_allowance_ms = 2000`** of the step's start, then apply Rule A.

**2000 ms is not tuned here.** It is this project's own documented onset lag:
*"rotation does not start for ~1–2 s, and the PEAK rate occurs at or after the
command ends"* (2026-08-29 dead-time measurement, `CLAUDE.md`). Declaring a
**physical time window** rather than "the first interval" makes the rule
independent of where sample boundaries happen to fall — which is precisely the
defect that split the two +180 runs.

🔑 **This is the rule I expect to win, stated now so the expectation is on the
record and can be falsified.**

**Rejected if:** it passes either 5000 ms anchor (with a 2000 ms exclusion the
5000 ms runs retain ~3 intervals, so it must still catch their ramp); or the
exclusion leaves fewer than 3 informative intervals on any 7000 ms run, since a
rule that cannot score the existing configuration is useless.

---

## 4. Rule D — settle-anchored plateau agreement

Compare the endpoint rate over the **final 3000 ms of the step** against the
endpoint rate over the **3000 ms preceding it**, both by the §0 rate definition,
and pass when they agree within 1.5 °/s. Ignores the onset entirely by
construction rather than by exclusion, and asks the question 2a actually
means — *had rotation stopped changing by the END of the step?*

**Rejected if:** either window contains fewer than 2 distinct readings on any
7000 ms run; or 2σ noise exceeds 1.5 °/s (short windows are noisier — this is
the rule most likely to die on admissibility, and that is a real possibility, not
a formality).

---

## 5. Rule E — residual-trend (slope) test

Least-squares slope of interval rate against interval midpoint across the step
phase, excluding the §3 onset window. Pass when
**|slope| ≤ 0.30 °/s²**, a bound fixed here as `1.5 °/s` of drift across a
nominal 5 s of post-onset step — i.e. the same tolerance the criterion already
uses, expressed as a trend.

Tests "no residual trend" rather than "halves agree", so a slow steady drift that
symmetric halves can mask is caught.

**Rejected if:** it passes either 5000 ms anchor; or the slope's own 2σ, from the
measured 1.445 °/s scatter, exceeds 0.30 °/s² at 7 informative intervals.

---

## 6. Diagnostics — computed and reported, NOT verdicts

Per rule, per run: 2σ noise of the decisive statistic; informative interval count
after any exclusion; and the RTK `course_series` cross-check where a chord clears
0.15 m. **None of these decides anything.**

---

## 7. Preference criteria — declared before results, in priority order

1. **Admissibility.** A rule whose decisive statistic has 2σ noise above the
   1.5 °/s bound is **inadmissible whatever verdicts it produces**.
2. **Anchor-clean** (§8). Not negotiable.
3. **Addresses the diagnosed failure**: it must remove or bound the onset BIAS
   identified in §0.1, not merely reduce variance.
4. **Minimal disruption** as a tiebreak: fewer published verdicts flipped.

---

## 8. Anchors — what counts as ground truth

- **R1 and R1r (step 5000 ms) genuinely did NOT reach steady rotation.** Both
  runs' final step interval sits at or near the phase's largest magnitude, and
  the ~2 s onset lag against a 5 s step independently predicts it. **Any rule
  that passes 2a on R1 or R1r is REJECTED.**
- ⚠️ **Carried forward verbatim from 2026-08-31: no equivalent anchor exists for
  SX or R2.** Which of those "really" converged is exactly what is in dispute.
  **No rule will be preferred BECAUSE it passes SX, or fails R2, or vice versa.**
- 🆕 **New anchor, declared before recomputation: the two +180/7000 runs must
  receive the SAME verdict.** Their plant agrees to **0.195 °/s** on intervals
  2–7 and their published second-half means to 0.424 °/s
  (`docs/findings-plus180-split-is-onset-sampling-phase-20260901.md`); a rule
  that splits them is reporting sampling phase, not the mower.
  ⚠️ **Declared honestly as derived from data already seen** — it is a
  post-hoc-informed anchor, and it is registered here rather than applied
  silently. A rule failing only this anchor is **demoted, not rejected**.

---

## 9. Deliverables

`docs/findings-2a-replacement-20260903.md`, naming this commit's hash, reporting
every rule against every run, applying §7 in order, and stating the **costs** of
the winner plus a *"what the banked data cannot settle"* section.

⚠️ **n = 5 scoreable runs, 4 configurations, 1 mower.** This cannot establish a
false-positive rate, and no rule selected here is validated until it faces a run
it has never seen.
