# Criterion 2a cannot be fixed by a longer step — the yard runs out first (2026-09-01)

**Offline. No mower, no dispatch.** Derived from the two banked +180/7000 runs
using the shipped scoring functions. ⚠️ **This retires "make the step longer" as
an approach.** It supersedes the reasoning in
`docs/phase2-long-step-slow-speed-predeclared-20260901.md` §1-§3, which is left
in place with its §6 corrections as a record of how the conclusion was reached.

## 1. The model was wrong, and the corrected one is decisive

The predeclaration modelled `half_diff ≈ |steady − onset| / k`, i.e. the onset
interval alone. **There are two terms**, and the second was omitted:

```
half_diff  =  onset/k  +  noise,     noise ~ sd * sqrt(2/k)
```

with the residual scatter measured from the ex-onset step intervals pooled across
both +180 runs: **n = 12, mean −12.136 °/s, sd 1.445 °/s**.

| step | informative n | k | onset/k | noise sd | **P(2a pass)** |
| --- | --- | --- | --- | --- | --- |
| 7 s (both banked runs) | 7.1 | 4 | 2.61 | 1.022 | **13.9%** |
| 15 s (the beta96 config) | 15.1 | 8 | 1.30 | 0.723 | **60.7%** |
| 20 s | 20.2 | 10 | 1.04 | 0.646 | **76.0%** |
| 25 s | 25.2 | 13 | 0.80 | 0.567 | **89.1%** |

Derived two independent ways — an adversarial reviewer's bootstrap of the
observed intervals gave **58.5%** at 15 s where this closed form gives **60.7%**.

🔑 **At 15 s the run is a coin flip.** A FAIL would be recorded as "the plant is
not steady" when it is the statistic's own noise floor, and a PASS is equally
uninformative — the same trap as the 2026-09-01 +180 repeat, re-weighted.

## 2. Why a longer step still does not rescue it

⚠️ **A reviewer claimed `half_diff` "does not tend to 0 as k grows". That is
false** — both terms shrink, so it converges. Their evidence was the banked
+180 run's ex-onset `half_diff` of 1.625 °/s, which reproduces exactly but is
just residual noise at k = 3 (≈1.4σ), not a floor.

**The true statement is that it converges too slowly to reach.** A genuine ~89%
needs a **25 s step**, hence a 33 s window:

| | at linear 300 (~0.16 m/s) |
| --- | --- |
| travel over a 33 s window | **~5.3 m** |
| `max_travel_m` ceiling | **4.5 m** |
| corridor radius required (`travel + 0.50`) | **~5.8 m** → an 11.6 m square |
| largest square contained in "Backyard Right" | **10.0 m** |

🚨 **The yard cannot contain a step long enough to make 2a reliable.** Same shape
as the 2026-08-30 finding that the exposure bound and the measurement requirement
genuinely conflict — and here the geometry, not a policy, is what binds.

## 3. What follows

🗑️ **Do not propose a longer step, another repeat at any length, or a
`step_ms`/`_STEP_RESPONSE_MAX_TOTAL_MS` raise as the way to settle 2a.** All
three are now measured to be dead ends at this site.

🔑 **2a needs a different statistic, not a longer window.** The half-phase
mean-rate comparison spends its resolution on an onset interval it should not be
averaging in. Any replacement must be **predeclared and validated against all
four banked runs before its verdicts are computed** — ⚠️ note that the obvious
candidate, dropping the onset interval, flips both +180 verdicts the other way
(banked 1.625 FAIL, repeat 0.107 PASS), which is exactly why it must not be
chosen after seeing outcomes. This is free, offline work needing no mower.

⚠️ **Everything here is derived at linear 400.** The onset magnitude (10.43 °/s)
and the residual sd (1.445 °/s) come from +180/7000 runs at linear 400. The
`(linear 300, angular ±180)` operating point has **never been exercised** —
`grep '"linear_speed": 300'` finds nothing in `docs/` — so neither constant is
known to transfer.

---

## 4. 🗑️ CORRECTION — §2's "the yard runs out first" is WRONG. The yard is fine.

**The error: §2 sized the yard with the largest axis-aligned SQUARE, but the
containment gate requires a DISK.** `step_path_contained` tests
`max_travel_m + 0.50 m` of clearance in **every direction** — an inscribed
circle, not a square. Searching for squares understated the usable space by the
square/circle factor (√2), and the conclusion inverted.

Exact maximum inscribed radius, measured from the live polygons (edge-sampled,
keep-outs included):

| region | inscribed radius | (largest square, for comparison) |
| --- | --- | --- |
| "Backyard Right" alone | **5.913 m** | 8.36 m |
| "Backyard Right" + "Backyard Hill" | **7.007 m** | 9.91 m |

✅ **Cross-check: 5.913 m matches the 5.9039 m clearance independently measured at
the 2026-09-01 run's own start point.** The geometry is right.

**So "Backyard Right" alone already holds a corridor for a ~26.8 s step, which
reaches ~91.7%.** The yard was never the binding constraint.

🔑 **What actually binds is three schema caps:** `max_travel_m` ≤ 4.5,
`_STEP_RESPONSE_MAX_TOTAL_MS` = 23000, and `step_ms` ≤ 15000. Even at today's
4.5 m travel budget the window could run ~28 s; `step_ms` holds the step at 15 s
and therefore holds 2a at 60.7%.

⚠️ **§1's convergence table and the ~60.7% figure at a 15 s step are UNAFFECTED
and stand.** Only §2's geometric conclusion, and §3's "measured dead end"
framing of a longer step, are withdrawn. A longer step is **not** a dead end —
it is capped by schema values that are ours to move deliberately.

**Superseded by** `docs/phase2-long-step-cap-raise-predeclared-20260902.md`.

---

## 5. 🗑️ CORRECTION 2026-09-03 — §1's table was computed on the WRONG STATISTIC

§1 modelled `half_diff = onset/k + noise`, with `noise ~ sd·sqrt(2/k)`. That
assumes each half rate is a **simple mean of interval rates**. It is not. The
shipped `_step_response_half_phase_agreement` computes a **time-weighted ENDPOINT
difference per half**, sharing the boundary reading:

```python
(t0, a0), (t1, a1) = half[0], half[-1]
half_rates.append(normalize_degrees(a1 - a0) / ((t1 - t0) / 1000))
```

🔑 **The difference is structural, not cosmetic.** An endpoint difference depends
only on the two end readings, so its noise scales as **1/T** (the half's
duration) — it does **not** average down as `1/sqrt(k)` with more intervals. The
old model therefore had both the wrong scaling and the wrong magnitude.

**Re-derived by bootstrapping the SHIPPED function** over 12 real ex-onset step
intervals and the real `dt` distribution (0.711–1.116 s), drawing both observed
onset severities (−5.675 and −1.352 °/s), N = 4000 per point:

| step | intervals | P(2a pass) — corrected | (§1 claimed) |
| --- | --- | --- | --- |
| 7 s | 7.1 | **34.4%** | 13.9% |
| 15 s | 15.3 | **68.7%** | 60.7% |
| 20 s | 20.4 | **81.4%** | 76.0% |
| 25 s | 25.5 | **88.5%** | 89.1% |
| **30 s** | 30.6 | **53.1%** ⚠️ | (not computed) |

🚨 **THE CURVE IS NOT MONOTONIC, AND §1 MISSED IT ENTIRELY.** It peaks near 25 s
and **collapses at 30 s**, because each half rate is an endpoint difference pushed
through `normalize_degrees` (wrapping to [−180, 180)): at the measured 12.136 °/s
a 15 s half accumulates **182°** and silently aliases, flipping the half rate's
sign. Aliasing onset is a **29.7 s step**. This independently reproduces the
aliasing finding from the 2026-09-02 adversarial review of beta97.

✅ **§2's and §3's conclusions are UNCHANGED.** 2a remains noise-dominated at
every reachable step length, the yard/travel budget still caps the step well
below the 25 s peak, and a replacement statistic is still the answer. The numbers
move; the argument does not.

⚠️ **Also corrected:** the ex-onset re-split figures. **1.625 / 0.107** are the
shipped statistic; the **1.727 / 0.503** quoted in `CLAUDE.md` were a simple mean
of interval rates. Both are internally correct — only the shipped pair describes
the shipped rule.
