# Findings — which signal should criterion 2a be scored against?

**2026-08-31, offline analysis only.** Predeclaration:
`docs/predeclared-rtk-vio-course-rate-scoring-20260831.md`, committed at
`5b320002` **before** any verdict below was computed. Script:
`scripts/rescore_course_rate_rules.py` (stdlib only, no HA import, no
network). Machine-readable results:
`docs/rescore-course-rate-results-20260831.json`. Data: the 549 rescued
samples in `docs/raw-samples/`. No motion was commanded, no hardware touched,
no scoring code in `custom_components/` changed.

## 0. Sanity gates — both passed, so the numbers below are anchored

1. **The rebuilt RTK course series matches the deployed build's own
   `course_series` on all four runs** — every row's phase, informativeness,
   chord and course, to 1e-3.
2. **Rule A reproduces every published number**: all four runs' step and
   settle rate sequences to ±0.01 °/s, and all eight published 2a/2b
   verdicts. The published verdicts are correct readings of the registered
   instrument; nothing below is a re-litigation of arithmetic.

**One finding about the cross-check doc itself**
(`docs/vio-crosscheck-reframes-route1-step-verdicts-20260830.md`): its two
decisive numbers reproduce exactly — SX 1.96 °/s, R2 0.38 °/s — but its
listed VIO rate *sequences* are shifted one interval early: each begins with
the run's final **baseline** interval (0.356 for SX, −0.166 for R2) and omits
the final step interval, and the "last two readings" it names for each run
are therefore not the pair its own diffs were computed from. The actual final
step pairs are (−6.99, −8.952) for SX and (−11.545, −11.165) for R2. The
doc's conclusion survives; its illustrative sequences do not.

## 1. The full per-run, per-rule table

Bound 1.5 °/s, ≥3 informative intervals, exactly as predeclared. Verdict
format: last-two diff (or half-rate diff for rule E) → verdict.

| Rule (2a) | R1 +120/5000 | R1r +120/5000 | SX +120/7000 | R2 +180/7000 |
| --- | --- | --- | --- | --- |
| **A** RTK last-two (status quo) | 2.493 → FAIL | 7.280 → FAIL | 0.112 → **PASS** | 5.640 → FAIL |
| **B** VIO last-two | 0.717 → **PASS** | 1.601 → FAIL | 1.962 → FAIL | 0.380 → **PASS** |
| **C** both A and B | FAIL | FAIL | FAIL | FAIL |
| **E-RTK** half-phase means | 2.737 → FAIL | 2.139 → FAIL | 2.846 → FAIL | 3.567 → FAIL |
| **E-VIO** half-phase means | 2.156 → FAIL | 3.664 → FAIL | 2.319 → FAIL | 0.130 → **PASS** |

Published 2a for reference: FAIL, FAIL, PASS, FAIL.

Full rate sequences (step phase):

| run | RTK rates (°/s) | VIO rates (°/s) |
| --- | --- | --- |
| R1 | −0.339, −8.002, −5.686, −8.179 | −2.191, −5.678, −6.699, −6.666, −7.383 |
| R1r | −0.573, −9.865, −3.828, −11.108 | −1.594, −6.226, −6.726, −9.450, −7.849 |
| SX | −1.718, −3.428, −9.026, −6.127, −8.241, −8.353 | −1.247, −6.014, −6.597, −7.001, −6.542, −6.990, −8.952 |
| R2 | −3.366, −11.885, −10.765, −15.094, −8.246, −13.886 | −5.675, −13.563, −11.443, −14.293, −11.402, −11.545, −11.165 |

Settle (2b), for the disruption question:

| Rule (2b) | R1 | R1r | SX | R2 |
| --- | --- | --- | --- | --- |
| **A** RTK (published) | 2.065 → FAIL | 0.263 → PASS | 1.439 → PASS | 0.770 → PASS |
| **B** VIO | 0.352 → **PASS** | 0.351 → PASS | 0.763 → PASS | 0.713 → PASS |

Switching 2b to VIO flips exactly one published verdict — R1's settle FAIL
becomes a PASS — and §2 shows that published FAIL was inside the RTK
channel's own noise anyway.

## 2. The headline: the status-quo statistic is noise-bound, measured two ways

**Analytic.** At the registered σ = 0.0031 m per axis, a ~0.225 m chord
carries √2·σ/c ≈ 1.12° of bearing noise. Propagated through the last-two-rate
difference (three consecutive bearings, the middle one counted in both
rates), each run's actual chords and interval durations give:

| run | σ(last-two-diff), RTK step | 2σ | bound |
| --- | --- | --- | --- |
| R1 | 2.735 °/s | 5.47 | 1.5 |
| R1r | 2.572 °/s | 5.14 | 1.5 |
| SX | 2.792 °/s | 5.58 | 1.5 |
| R2 | 2.751 °/s | 5.50 | 1.5 |

**The 1.5 °/s bound sits at ~0.55σ of the statistic it bounds.** A genuinely
steady rotation would FAIL rule A about 58% of the time; a still-ramping one
can PASS by draw. Concretely: SX's published PASS margin (0.112 °/s) is 0.04σ
— indistinguishable from luck — and R2's published FAIL margin (5.64 °/s) is
~2.05σ — mostly explicable by noise alone, before any plant behavior.

**Empirical corroboration, cross-channel.** Per-interval RTK-vs-VIO rate
disagreement (step phase RMS): R1 1.447, R1r 2.735, SX 1.659, R2 2.482 °/s —
the same ~1.6–2.7 °/s scale the analytic model predicts for RTK noise
(per-rate σ ≈ 1.6 °/s plus VIO noise and real fluctuation). The two
derivations agree, and neither was tuned to the other.

**VIO, by contrast, resolves the bound.** Pooled over all four runs'
baseline phases (angular 0, so rate variation there is channel noise plus
real steering wobble — an over-estimate): σ_rate = 0.207 °/s (n=12);
settle-tail scatter agrees at 0.203 °/s (n=4). σ(last-two-diff) = 0.293 °/s,
so the 1.5 bound sits at ~5σ on VIO, ~11σ for the half-phase variant. VIO
also latched cleanly: zero distinct-reading intervals shorter than 0.5 s in
549 samples, so no quantization artifacts contaminate its rates.

**Answer to the predeclared diagnostic on chord compression:** the RTK-vs-VIO
disagreement does **not** grow with commanded angular speed — R2 (+180) at
2.482 sits inside the +120 range (1.447–2.735; the worst run is R1r at
+120). At these rates the chord proxy's failure is *noise on short chords*,
not curvature compression. All step chords are 0.218–0.242 m regardless of
config, so chord length offers no correlation to test — the ~1 Hz feed and
~0.23 m/s speed fix the chord, and that chord fixes the noise.

## 3. The anchors did their job: rule B is rejected too, for the opposite flaw

The predeclaration registered R1 and R1r as ground-truth NON-converged
(onset-lag arithmetic plus both channels' own final-interval-largest
sequences), and rejected in advance any rule that passes them.

**Rule B passes R1 (0.717 °/s), violating the anchor.** The mechanism
matters: R1's VIO sequence (−2.191 → −5.678 → −6.699 → −6.666 → −7.383) is
still climbing at phase end — but a smooth exponential-ish ramp has
*adjacent* rates nearly equal long before it converges. On a low-noise
channel, last-two-diff measures the ramp's local smoothness, not its
convergence. So the two single-channel last-two rules fail for mirror-image
reasons:

* **A (RTK)**: statistic dominated by noise → verdicts are draws.
* **B (VIO)**: statistic dominated by local smoothness → passes mid-ramp.

**Rule C (agreement)** is anchor-clean but inherits A's coin flip on the
pass side: a genuinely converged run clears C only when A's ~±2.7 °/s noise
happens to land inside 1.5, i.e. ~42% of the time. An AND with a die roll is
not a stricter criterion, it is a slower one.

**Rule E-RTK** is anchor-clean but still noise-bound (analytic σ of the
half-rate difference 0.89–1.35 °/s → 2σ 1.78–2.70 > 1.5 on every run).

**Rule E-VIO is the only rule that is both admissible and anchor-clean.** It
fails R1 (2.156) and R1r (3.664) — correctly detecting both ramps the
last-two rules mishandled on one channel each — with noise roughly 0.13 °/s
on its decisive statistic, so its calls are ~10–25σ results, not draws.

## 4. What E-VIO says about the two disputed runs — and the physical picture

* **SX (+120, 7000 ms): FAIL, 2.319 °/s.** VIO's second half (−7.447 °/s
  mean) is materially faster than its first (−5.128), driven by a real final
  step interval at −8.952 °/s — the largest magnitude of the whole phase,
  the same "final interval largest → still accelerating" signature that
  condemned both 5000 ms runs. Under E-VIO, **+120 had not reached steady
  rotation even at 7000 ms.** RTK read the same tail as flat (−8.241,
  −8.353) but with ±2.7 °/s of noise it cannot contradict VIO's reading.
* **R2 (+180, 7000 ms): PASS, 0.130 °/s.** VIO's halves agree to 0.13 °/s
  (−11.229 vs −11.359) and its tail is a genuine plateau (−11.402, −11.545,
  −11.165). The published FAIL's "oscillation" (−8.246 → −13.886 swings) is
  quantitatively what σ ≈ 2.75 °/s chord noise looks like superimposed on a
  steady −11.3 °/s rotation.

This account is self-consistent across all four runs on one instrument:
monotone ramps at 5000 ms, a ramp still finishing at +120/7000, a plateau at
+180/7000. Plausibly the stronger command drives through the onset lag
faster; **that is a story fitted to n=1 per config and must not be written
down as a mechanism.**

## 5. Recommendation (written last, per the predeclared cascade)

**Score criterion 2a against VIO heading using half-phase mean-rate
agreement (rule E-VIO), keeping the 1.5 °/s bound and the ≥3-informative-
intervals requirement unchanged.** It is the only candidate that survives
the predeclared preference cascade: admissible (bound ≈ 11σ instead of
0.55σ), anchor-clean (fails both known-ramping runs), and its decisive
statistic uses endpoint differences — the same integral property the shipped
analysis already relies on for τ.

Not implemented anywhere, per the boundaries of this research. If the
operator adopts it, the change would land in
`_step_response_analysis`/`_step_response_course_series` (or a sibling), and
two coherence questions land with it, flagged not decided:

1. **ω and τ should come from the same channel that certifies steadiness.**
   Scoring 2a on VIO while computing `omega_step_deg_per_s` from RTK chords
   reintroduces the exact ramp-sampled-ω failure 2a exists to prevent.
2. **2b** was not predeclared for rule E and is left as-is; for the record,
   switching 2b's channel to VIO (last-two) flips only R1's settle FAIL→PASS,
   and that FAIL was inside RTK noise anyway (2.065 vs 2σ = 5.5).

**Costs, stated plainly:**

* **Flips the headline PASS.** SX's published 2a PASS does not survive:
  `tau_actuator_s = 2.038 s` reverts from "genuine measurement" to "not
  settled" — the step was still accelerating on the admissible instrument.
* **Flips R2's 2a FAIL to a PASS** on steadiness. That does *not*
  retroactively bless R2's published τ (2.162 s), which was computed from
  RTK-chord ω; see coherence question 1.
* Harder to pass than the published rule only in the sense that it stops
  awarding passes by noise draw; on a genuinely converged run it is far
  *easier* to pass reliably (σ 0.13 vs 2.7 °/s).
* VIO dependence: E-VIO inherits VIO's documented weaknesses — light
  dependence and `signal_none` at night. Both 2026-08-30 run days were
  daylight with `state: 2` throughout; a night step-response run could not
  be scored under this rule at all. RTK chords remain the only night-capable
  course source; a rule change should say what happens when VIO is dark
  (refuse to score, do not fall back silently).

**What the banked data cannot settle, at n=1 per configuration:**

* Whether +120 genuinely fails to converge within 7000 ms (SX's late
  −8.952 °/s kick is a single reading of a single run — a disturbance in
  that final interval would produce the same signature).
* Whether the +180 plateau repeats.
* Whether VIO heading carries any *systematic* error during sustained
  rotation (its accuracy here is inferred from internal consistency and its
  baseline noise floor, not from an independent reference — the only
  independent reference in the data is the RTK chord, which is too noisy to
  adjudicate).

A future run that would settle the first two needs only a repeat of each
7000 ms configuration scored under a predeclared E-VIO rule — same corridor
discipline, same per-run authorization, nothing new to build. The third
needs a slower-rotation window where chord noise shrinks relative to signal
(longer chords or slower turn), which conflicts with the travel budget in
the known way and should not be attempted casually.

**Standing decision 5 is unaffected**: this is a scoring-methodology
recommendation for a parked line of work. It authorizes no run, no code
change, and no cap movement.
