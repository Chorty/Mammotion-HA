# The prediction model holds out of sample — and its yaw term does not

**2026-08-22 evening, one supervised authorized run on beta71.** An arc at
`angular 120`, scored against constants **committed before it ran**
(`docs/frozen-prediction-constants-20260822.json`, commit `c6f16f07`), so the
model could not be tuned to the run that tests it.

Evidence: `docs/evidence-arc120-outofsample-20260823T001500Z.json` and
`docs/evidence-arc120-corridor-20260823T001500Z.json`.

## 1. Position prediction PASSES out of sample

Frozen: `k_lin = 6.204299e-04` (v@400 = 0.2482 m/s, from 16 steady steps across
three straight runs) and `k_ang = 3.989367e-02` (w@180 = 7.181 deg/s, from the
single angular-180 arc).

| | steps | median error | max error |
| --- | ---: | ---: | ---: |
| all intervals | 7 | 0.0175 m | 0.1536 m |
| **excluding spin-up** | **6** | **0.0175 m** | **0.0628 m** |

Against the proposed 0.10 m max, the steady-state result **passes with 37%
margin**. The only failing interval is the first, which mixes dispatch latency,
drivetrain ramp and ~1 s feed staleness and is excluded by the criterion's own
scope.

🔑 **The multi-run constant beat the single-run fit, which is what "out of
sample" is supposed to reward.** Re-fitting `k_lin` from one straight capture
gives 0.2148 m/s and a median error of 0.0508 m; the frozen 0.2482 m/s, averaged
over three runs, gives **0.0175 m**. Nearly 3x better on data neither was fitted
to.

## 2. ⚠️ WITHDRAWN 2026-08-23 — this section claimed a refutation it cannot support

🚨 **Read `docs/corrections-to-the-20260822-analysis-20260823.md` §1.** The 45% figure below is an artifact of fitting `k_lin`
**excluding** spin-up and `k_ang` **including** it. Under the same rule the error
is **11%**, and like-for-like the data sits 2.31 sigma from "no dependence" and
2.28 sigma from "proportional" — it cannot distinguish them. The section is kept
as written because the *measurements* are right; the **conclusion is withdrawn**,
including the instruction to Phase 2 to drop the model.

## 2. 🗑️ (WITHDRAWN) The yaw term `w = k_ang * angular_speed` is REFUTED

The frozen constant predicted **4.787 deg/s** at `angular 120`. Observed:
**47.1 deg over 6.79 s in-window = 6.94 deg/s**, and 50.8 deg settled across the
whole run. That is **45% higher than predicted**.

Compare the two arcs directly:

| commanded angular | observed yaw rate |
| ---: | ---: |
| 180 | 7.181 deg/s |
| 120 | **6.94 deg/s** |

A 33% cut in commanded angular produced a **3%** change in yaw rate. **Rotation
is very nearly independent of `angular_speed` across 120-180**, not proportional
to it. Any model of the form `w = k * angular` is wrong in this range, and the
pre-registration is the reason that is a finding rather than a fitted parameter.

⚠️ Two points only, both in a narrow band. This does not say the rate is
constant everywhere — `angular 500` is a documented *stationary* figure and
remains unmeasured in an arc. It says linear scaling is refuted where we
measured.

## 3. Why the position test passed anyway — and why that matters

The predictor re-anchors on the **measured** heading at each interval start, so
a wrong yaw model only has ~1 s to do damage. At a 2 deg/s rate error that is
~2 deg of heading over an interval, worth about
`0.25 m x sin(2 deg)` = **0.009 m** — an order of magnitude under the observed
error. The yaw model is wrong and the position prediction barely notices.

🔑 **For a continuous controller this is the useful half of the result: it needs
accurate heading FEEDBACK, not an accurate yaw MODEL.** Heading feedback it has,
bundled with position at ~1 Hz. That removes yaw calibration from the critical
path for Phase 2 and puts the weight back on the feedback cadence, which is
already characterised.

## 4. The distance guard fired for a second time

```
tripped at 6785.7 ms, sampled travel 1.6093 m
window ended 6988.5 ms, aborted_early: true, 34 refreshes
stop confirmed; final position inside the frozen corridor
```

| run | guard | published bound | actual travel | overshoot |
| --- | ---: | ---: | ---: | ---: |
| straight, guard run | 1.5 m | 1.85 m | 1.776 m | 0.276 m |
| **this arc** | 1.5 m | 1.85 m | **1.8074 m** | **0.307 m** |

Two for two, both inside the published `corridor_must_cover_m`. ⚠️ Still do not
tighten `_PROBE_TRAVEL_GUARD_OVERSHOOT_M` from 0.35: two samples averaging 0.29
with the larger one at 0.307 leaves little headroom, the constant is a safety
margin, and being wrong is asymmetric.

## What this changes

- The criterion revision in
  `docs/phase1-criterion-revision-proposal-20260822.md` now has an out-of-sample
  result behind it rather than only in-sample arithmetic. It is still a
  **proposal**; `analyze_phase1_capture.py` is untouched and its verdict remains
  **`no_go`**.
- Any Phase 2 controller design should drop `w = k * angular` and either
  measure the yaw curve properly or, better, lean on measured heading per
  interval as this predictor does.
