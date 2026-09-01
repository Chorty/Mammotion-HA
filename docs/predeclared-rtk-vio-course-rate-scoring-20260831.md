# Predeclaration — candidate scoring rules for criterion 2a (RTK vs VIO)

**2026-08-31, committed BEFORE any verdict is computed.** This file registers
every scoring rule the re-scoring script will evaluate, exactly how each
computes a rate, how each decides pass/fail, and what result would make me
prefer or reject each one. The git timestamp of this commit is what makes the
results in the findings document trustworthy: nothing below was chosen after
seeing which rule produces which answer.

Scope: offline re-scoring of the four banked route-1 step-response runs in
`docs/raw-samples/` (549 samples). No motion, no Home Assistant, no code
change in `custom_components/`. This work answers a methodology question —
which signal criterion 2a should be scored against — and produces a written
recommendation only.

## 0. Shared definitions

All rules share these, so no rule can win by quietly redefining a term:

* **Phase boundaries**: the *nominal* boundaries the shipped
  `_step_response_course_series` used — an interval belongs to `baseline` if
  its midpoint time < `baseline_ms`, to `step` if < `baseline_ms + step_ms`,
  else `settle`. Nominal values are read from each raw file's own `phases`.
  (The recorded `phase_transitions` differ by ~13 ms; the shipped code used
  nominal, and reproduction requires matching it.)
* **Angle wrap**: `normalize_degrees(a) = (a + 180) % 360 − 180`, byte-for-byte
  the codebase's own function, applied to every heading/course difference.
* **Rate**: `normalize_degrees(Δangle) / Δt_seconds`, attributed to the
  interval between the two readings.
* **Informativeness floor (RTK only)**: chord ≥ 0.15 m, the registered floor
  at σ = 0.0031 m. Chords below it have no course and produce no rate.
* **The 1.5 °/s bound and the ≥3-informative-intervals requirement are NOT
  moved by any rule below** except where a rule explicitly declares a
  noise-derived verdict of "indistinguishable" (rule D). No rule lowers the
  interval count or widens the numeric bound.
* **Runs**: R1 = +120/step 5000, R1r = +120/step 5000 repeat,
  SX = +120/step 7000, R2 = +180/step 7000.

## 1. Rule A — RTK chord bearing, status quo (reproduction target)

Exactly the shipped rule, reimplemented:

* Distinct-position extraction: drop samples whose x/y equal the previous
  distinct sample's; chord course = `atan2(Δy, Δx)` in degrees between
  consecutive distinct positions; interval midpoint = mean of the two
  `elapsed_ms` values.
* Step rates: consecutive pairs of *informative step-phase* rows —
  `normalize(Δcourse) / Δmidpoint_s`.
* Settle rates: consecutive pairs over the sequence *(last informative step
  row, then every informative settle row)* — i.e. the first settle rate is
  the carryover pair, matching how the published evidence files scored 2b.
* 2a: ≥3 informative step intervals AND |last step rate − second-to-last| ≤
  1.5 °/s. 2b: ≥3 informative settle intervals AND same bound on the last two
  settle rates.

**Reproduction gate (predeclared stop condition):** this reimplementation
must reproduce the published step/settle rate sequences of all four evidence
files to ±0.01 °/s and the same four 2a and 2b verdicts. If it does not, that
is a finding — stop and report, do not proceed to preference judgments built
on a rule I cannot reproduce.

## 2. Rule B — VIO heading rate between distinct readings

* Distinct-reading extraction: walk the samples in order; a new VIO reading
  exists at the first sample whose `vio.heading` differs from the previous
  sample's. Its timestamp is that sample's `elapsed_ms`.
* Rate between consecutive distinct readings: `normalize(Δheading) / Δt`.
  No chord floor applies (VIO is a direct heading, not a geometric proxy).
  Every interval is informative.
* Phase of an interval: midpoint of its two timestamps against the same
  nominal boundaries as Rule A.
* 2a/2b: same shape as Rule A — ≥3 step intervals, last two step rates within
  1.5 °/s; settle scored with the carryover pair for symmetry.

**Reproduction check (soft):** the 2026-08-30 cross-check doc reports VIO
step-rate sequences and "last two apart" figures (1.96 °/s for SX,
0.38 °/s for R2). I will attempt to reproduce them. ⚠️ Declared now, before
computing: the doc's own SX numbers look internally inconsistent — its listed
last two rates (−6.542, −6.990) are 0.45 °/s apart, not the 1.96 °/s it
claims. If my computation disagrees with the doc, the finding is reported
either way; the hand cross-check is not a calibrated instrument and failing
to reproduce it does NOT invalidate the VIO channel, only the doc's
arithmetic.

## 3. Rule C — agreement of both

2a passes iff Rule A passes AND Rule B passes on the same run. (Same for 2b,
reported for completeness.) This is the conservative middle named in the
cross-check doc: it can only make 2a harder.

## 4. Rule D — noise-floor admissibility (per channel)

Not a different rate computation — a gate on whether a channel's 1.5 °/s
bound is even meaningful at n=1:

* **RTK analytic noise**: with position noise σ = 0.0031 m per axis, a chord
  of length c carries bearing noise σ_b ≈ √2·σ/c radians. The last-two-rate
  difference involves three consecutive bearings (the middle one shared with
  sign structure preserved); propagate linearly using each run's actual
  chord lengths and interval durations to get σ of (r_n − r_{n−1}) for the
  step phase of each run.
* **VIO empirical noise**: pooled over all four runs, the scatter (std) of
  baseline-phase VIO rates about each run's own baseline mean (baseline
  commands angular 0, so rate variation there is mostly channel noise plus
  real steering wobble — an over-estimate of pure channel noise, which is
  the conservative direction), and separately the scatter over the final
  2 s of settle. Use the larger as σ_rate; σ of a last-two-rate difference
  = √2·σ_rate.
* **Verdict semantics**: if 2·σ(last-two-diff) > 1.5 °/s for a channel on a
  run, that channel's 2a verdict on that run is downgraded to
  **INDISTINGUISHABLE** — the bound sits inside the channel's own noise and
  a single run cannot separate "converged" from "one favorable draw".

## 5. Rule E — half-phase mean-rate agreement (both channels)

A lower-noise estimator using the integral property the shipped analysis
itself relies on (endpoint differences are accurate at any sample rate):

* Split the phase's informative readings at the midpoint *by count* (odd
  counts put the extra reading in the first half; the boundary reading ends
  the first half and also starts the second, so both halves are endpoint
  differences over contiguous spans).
* Each half's rate = `normalize(last course − first course) / (last t −
  first t)` within that half.
* 2a: ≥3 informative step intervals (same count requirement as Rule A) AND
  |rate_half2 − rate_half1| ≤ 1.5 °/s. Variants E-RTK and E-VIO.

Declared rationale: last-two-interval agreement stakes the verdict on the two
noisiest single readings of the run; half-phase means average the noise down
by ~√k while still detecting a ramp (a still-accelerating run has
half2 ≠ half1). The risk, declared now: it is *less* sensitive to a late
plateau after an early transient, so it may fail a genuinely-converged run
whose first half contains the ramp. That failure mode is visible in the data
(the rate sequences) and will be reported per run.

## 6. Diagnostics (computed and reported, not verdicts)

1. Per-interval RTK-vs-VIO rate agreement: interpolate VIO rate onto each
   RTK interval's midpoint span; report RMS disagreement per run and its
   correlation with commanded angular speed (does the chord proxy degrade at
   +180 vs +120?), and with chord length.
2. Settle (2b) verdicts under Rule B — does switching channel flip any
   published settle verdict?
3. Chord-length and Δt distributions per run per phase.
4. The count of VIO "distinct-reading" intervals shorter than 0.5 s, and the
   rates they produce — a check on whether VIO latching artifacts inject
   spurious rates.

## 7. Preference criteria — how the recommendation will be chosen

Declared before results, in priority order:

1. **Admissibility first.** A rule whose decisive statistic has 2σ noise
   above the 1.5 °/s bound (rule D) is inadmissible as a single-run
   criterion, whatever verdicts it produces. If BOTH single-channel rules
   are inadmissible on the runs in question, the honest headline is
   "n=1 cannot settle this", and the recommendation must say what a future
   run needs (and may still recommend an interim rule for conservatism).
2. **Cross-channel corroboration of trend, not verdict.** Between admissible
   rules, prefer the one whose rate *sequence shape* (ramp vs plateau) is
   corroborated by the other channel — two instruments agreeing the rate was
   still climbing beats one instrument's last-two-diff.
3. **Lower measured noise wins.** Between estimators on the same channel
   (A vs E-RTK, B vs E-VIO), prefer the one with the smaller measured
   scatter against the diagnostic-1 cross-channel reference, provided it
   still detects the known ramps (both 5000 ms runs must not pass 2a under
   any preferred rule — they are the closest thing to ground truth this data
   has: the step ended while rotation was demonstrably still accelerating,
   and a rule that passes them is broken by construction).
4. **Minimal disruption tiebreak.** Between rules that survive 1–3 equally,
   prefer the one that flips fewer published verdicts (2a and 2b together).

Predeclared ground-truth anchors, for use in criterion 3 above:

* R1 and R1r (step 5000) genuinely did NOT reach steady rotation — both
  channels' published/cross-checked sequences show the final step interval
  at or near the phase's largest magnitude, and the onset-lag arithmetic
  (~2 s lag against a 5 s step) independently predicts it. Any rule that
  passes 2a on R1 or R1r is rejected.
* No equivalent anchor exists for SX or R2 — which run "really" converged is
  exactly what is in dispute, and no rule will be preferred *because* it
  passes SX or fails R2 or vice versa.

## 8. What would make me reject each rule (predeclared)

* **Rule A**: rejected as sole criterion if its step-phase last-two-diff 2σ
  noise (rule D) exceeds 1.5 °/s on any 7000 ms run, or if diagnostic 1
  shows its disagreement with VIO growing with rotation rate (the chord-
  compression mechanism), because then its verdict at +180 measures the
  proxy, not the plant.
* **Rule B**: rejected as sole criterion if its own noise fails the same 2σ
  test, or if diagnostic 4 shows latching artifacts materially contaminating
  step-phase rates, or if baseline-phase VIO rates show scatter comparable
  to the step-phase signal.
* **Rule C**: rejected if both A and B are individually inadmissible (an AND
  of two coin flips is stricter but still a coin flip), or if it fails a
  ground-truth anchor.
* **Rule E**: rejected if it passes 2a on R1 or R1r (anchor violation — too
  insensitive to a ramp), or if its own noise still fails the 2σ test.

## 9. Deliverables after this commit

1. `scripts/rescore_course_rate_rules.py` — stdlib-only, offline, no HA
   import, reads `docs/raw-samples/*.json`, reproduces Rule A against the
   published numbers, then scores all rules and diagnostics.
2. A findings document with the full per-run per-rule table.
3. A recommendation section written last, following §7, stating plainly what
   the data cannot settle.

Nothing in this work authorizes motion, touches the gate, or changes any
scoring code in `custom_components/`.
