# Corrections to the 2026-08-22 continuous-motion analysis

**Written 2026-08-23 after an independent adversarial review of
`docs/phase1-criterion-revision-proposal-20260822.md`. The review REJECTED the
proposal as written.** Every finding below was then re-derived from the banked
evidence before being recorded here; all of them held.

These are corrections to **my own claims from 2026-08-22**. The underlying
measurements stand. What follows are the places I stated more than the data
supports, or stated something false.

## 1. 🗑️ WITHDRAWN: "`w = k_ang * angular_speed` is REFUTED"

**This was an artifact of my own inconsistent fitting.** `k_lin` was fitted
**excluding** each run's spin-up interval; `k_ang` was fitted **including** it,
though `docs/frozen-prediction-constants-20260822.json` states the exclusion
rule. Applying the same rule to both:

| `k_ang` fit rule | predicted w@120 | error vs observed 6.936 deg/s |
| --- | ---: | ---: |
| frozen, spin-up **included** | 4.787 deg/s | **+45%** |
| consistent, spin-up **excluded** | 6.256 deg/s | **+11%** |

The headline "45% higher than predicted" was measuring my own bookkeeping.

Like-for-like steady-state rates are **9.386 deg/s @180 (n=2)** and
**7.813 deg/s @120 (n=6)**, a difference of 1.574 ± 0.681. A proportional model
predicts 3.129. The data sits **2.31 sigma from "no dependence" and 2.28 sigma
from "proportional"** — equidistant. **It cannot distinguish them.**

**Correct statement:** the yaw model over-predicts by 11-25% out of sample; two
arcs inside a 120-180 band cannot establish how yaw rate scales with commanded
angular. The instruction to Phase 2 to "drop `w = k * angular`" is withdrawn.

## 2. 🗑️ WITHDRAWN: "position, `toward` and VIO change on exactly the same
instants, zero exceptions"

True for **VIO**, false for **`toward`**. Counted directly:

| capture | position updates | `toward` updates | VIO updates |
| --- | ---: | ---: | ---: |
| straight 4 s | 4 | **1** | 4 |
| arc180 | 3 | 3 | 3 |
| 8 s sustain | 8 | **1** | 8 |
| guard run | 7 | **1** | 7 |
| arc120 | 7 | 7 | 7 |

**Correct statement:** no heading update ever arrives *without* a position
update, and **VIO heading** is 1:1 with position at ~1 Hz. **`toward` latches on
near-straight motion** — once in 8 seconds.

⚠️ **This undercuts a conclusion I drew from it.** I wrote that a continuous
controller "needs accurate heading FEEDBACK, not an accurate yaw MODEL… heading
feedback it has, bundled with position at ~1 Hz." On the straight-line regime a
lookahead controller spends most of its time in, **`toward` feedback is not at
1 Hz**. VIO is — so the conclusion may survive by leaning on VIO instead, but
that is a different claim and it is untested.

## 3. 🗑️ WITHDRAWN: "the criterion has no minimum chord"

It has one: `MIN_MOVING_STEP_M = 0.01` (`scripts/analyze_phase1_capture.py`).

**Correct statement:** the minimum chord is **0.01 m, three orders below the
position noise floor**, so it excludes only exactly-zero steps and cannot
protect against noise-dominated ones. That was the real defect. I named a
missing feature that is not missing.

## 4. ⚠️ OVERSTATED: "`err@end = err@start + rotation` holds to 0.001 deg on
every row"

It is an **algebraic identity**, exact by construction —
`error_at_end - error_at_start` *is* `end.toward - start.toward`. It cannot come
out otherwise, and presenting it as corroboration dressed a tautology as a
measurement. The substantive point — that the criterion's verdict is set by an
unargued pairing convention — stands without it.

## 5. ⚠️ SUPERSEDED: "alpha = -0.253 +- 0.174, START consistent at 1.45 sigma"

Two failures. The estimate used **2 steps from one arc** when **8 steps across
two arcs** were available by the time it was last repeated; and its noise model
assumed a position sigma of 0.007 m.

**The noise model was measurable and wrong.** The scatter of `err@start` across
16 straight steady steps is **sd 0.979 deg** at a median 0.2584 m chord, which
implies a position sigma of **0.0031 m**, not 0.007. (The same 16 steps give a
mirror offset of **+0.088 +- 0.245 deg** against the assumed 90.0 — consistent
with zero, so using 90.0 was sound. I had not checked that either.)

Re-derived on all 8 informative arc steps with the empirical sigma:

**alpha = -0.165 +- 0.043** (weighted) / **-0.165 +- 0.054** (unweighted)

| convention | distance |
| --- | ---: |
| START (0) | **3.1-3.9 sigma** — now EXCLUDED |
| MIDPOINT (0.5) | 12-16 sigma |
| END (1), shipped | 22-27 sigma |

🔑 **This is worse for me than the review said.** My earlier claim that START is
"consistent at 1.45 sigma" does not survive more data: **no simple pairing is
correct.** END remains by far the worst, so the original diagnosis holds — but
"pair at the start" is not the clean fix I implied.

⚠️ I also misdiagnosed the tight agreement between the first two alphas
(0.008 apart against +-0.23 bars) as "luck". It was **an inflated noise model**.

## 6. ⚠️ CORRECTED: the 0.10 m threshold, and a breach I never published

The budget does **not** close. I justified 0.10 m as two-thirds of the 0.15 m
waypoint tolerance "leaving a third for the steering law and sensing floor". One
third is 0.05 m, and this project's standing decision 3 fixes the **sensing
floor at 0.065 m** — larger than the whole remainder. `0.10 + 0.065 > 0.15`.

And I never scored the 8 s and guard captures against the frozen constants.
Doing so:

| capture | steady steps | median | max |
| --- | ---: | ---: | ---: |
| straight 4 s | 3 | 0.0188 | 0.0463 |
| arc180 | 2 | 0.0138 | 0.0166 |
| **8 s sustain** | 7 | 0.0443 | **0.1418** ❌ |
| guard run | 6 | 0.0285 | 0.0668 |
| arc120 | 6 | 0.0170 | 0.0628 |

**The 8 s run breaches the proposed 0.10 m maximum by 42%** — and it is
*in-sample* for `k_lin`. The failing step (3913 -> 5111 ms, 0.1555 m in 1.198 s)
follows an **810.1 ms refresh write**, the documented BLE-stall failure class in
which the device watchdog stops the motor mid-window. **The prediction criterion
is fully exposed to BLE stalls and my proposed threshold sits below the error
one produces.** The proposal never mentioned it.

## 7. ⚠️ CORRECTED: smaller errors

- "**16 of 17** arc criteria passed" — the arc has **18** criteria; **17**
  passed.
- "Trend **+0.0073 m/s per step**, drifting up" — the slope is
  **+0.0073 +- 0.0130**, i.e. **0.56 sigma**. That is noise. The defensible
  claim is only that speed **did not decay**.
- "Basis: today's steady-state **0.025-0.052 m**" for a threshold on the
  **max** — those figures are medians; the corresponding max was 0.0769 m.
- "**four** steady-state intervals across two runs" — it was **five**.
- `travel_guard.observed_travel_m` records travel **at the trip**, not travel of
  the run (1.6093 vs 1.8074 on the guard run). The name invites misreading.

## What is NOT withdrawn

- The pre-registration was genuine and verified from git: constants committed
  `c6f16f07` at 20:12:43, the run's first sample at 20:18:27, and `k_lin`
  reproduces exactly from the 16 declared steps. The review called it the
  strongest part of the work. It is also what made correction 1 findable.
- The out-of-sample numbers recompute exactly (median 0.0175, max 0.0628
  excluding spin-up). ⚠️ But the **all-intervals** max is **0.1536 m**, which
  fails both candidate thresholds — the spin-up exclusion decides that verdict
  and must be stated whenever the result is quoted.
- Sustain, containment, guard firing, and every physical measurement stand.
- The Phase 1 verdict remains **`no_go`**.
