# The acquisition budget — a safety trade, decided 2026-08-27

`max_heading_acquisition_s` is **2.0 s** and is the only thing now standing
between us and a first steering-sign test. This document states the trade and
recommends a value. **It changes no code and authorizes nothing.**

## Why 2.0 s is marginal by construction

Acquiring heading needs a position chord of at least
`min_travel_for_heading_trust_m` = **0.15 m**, measured between two position
samples. From a standstill that takes about **two samples**: one to anchor, one
showing enough displacement.

The feed's median interval is **1016 ms**, measured this evening over 117
intervals at the same epoch (`docs/evidence-position-cadence-post-reconnect-20260827.json`,
p95 1102 ms, p99 1127, max 1232). So a **2.0 s budget buys about two samples, and
losing either one fails acquisition.**

That is exactly what happened. Attempt 2 acquired at **1.95 s**; attempt 3 needed
a new sample inside a **1.01 s** window against a 1016 ms median and did not get
one. **57% of intervals exceed 1010 ms.** It is close to a coin flip, and neither
outcome was a fault.

## The trade

The budget sizes the blind-acquisition disk directly:

```
required_radius = max_safety_speed_mps x max_heading_acquisition_s + stop_overshoot_m
                = 0.28 x budget + 0.50
```

| Budget | Expected samples | Blind disk | Assessment |
| --- | --- | --- | --- |
| **2.0 s** (current) | 2.0 | **1.06 m** | marginal — one lost interval fails it |
| 2.5 s | 2.5 | 1.20 m | ok |
| **3.0 s** | **3.0** | **1.34 m** | **recommended** |
| 3.5 s | 3.4 | 1.48 m | good, but eats corridor |
| 4.0 s | 3.9 | 1.62 m | exceeds the clearance used so far |

## Recommendation: 3.0 s

**Why 3.0 and not more:** it buys a third sample, so acquisition survives losing
one interval — the actual observed failure — while keeping the disk at **1.34 m**,
inside the ~1.50 m clearance the corridors used in every run today. At 3.5 s the
disk (1.48 m) effectively consumes that clearance and would start refusing runs on
geometry.

**Why 3.0 and not less:** 2.5 s buys only half an extra sample. The failure mode is
discrete — you get the sample or you don't — so a half-sample of margin does not
change the odds much.

⚠️ **This is a real safety cost, stated plainly.** The mower may travel blind for
3.0 s instead of 2.0 s before any heading exists, and the disk that must be clear
grows from 1.06 m to 1.34 m — a **26% larger** required clear radius. Attempt 3
actually travelled 0.5097 m blind under the 2.0 s budget; under 3.0 s the same run
would travel roughly 0.75 m. Both sit inside their respective disks, which is the
point of sizing the disk from the budget.

🔑 **The disk formula must move with the budget.** It already does —
`blind_acquisition_feasibility` computes it from `max_heading_acquisition_s`, so
raising the budget automatically raises the clearance a run must prove before it
may open. Nothing here weakens containment; it enlarges both the exposure and the
proof required for it.

## Rejected alternatives

🗑️ **Lower `min_travel_for_heading_trust_m` below 0.15 m.** Refused. It is the
registered informativeness floor: at the measured sigma = 0.0031 m position noise
a shorter chord carries bearing noise exceeding the thresholds it feeds. This was
already settled when `MIN_MOVING_STEP_M` went 0.01 → 0.15 on 2026-08-23, and
lowering it to make a test pass is precisely the move that repair was written to
prevent.

🗑️ **Drive faster during acquisition to reach 0.15 m sooner.** Refused. The blind
disk scales with speed as well (`max_safety_speed_mps x budget`), so it buys
nothing and worsens the consequence of a bad heading.

🗑️ **Accept a shorter chord only for the first decision.** Refused for the same
reason as the first: a heading whose noise bound exceeds the steering threshold
cannot inform a steering command, and steering on it is the failure the whole
Phase 2 remediation exists to prevent.

## What this needs before it is real

A code change to `ContinuousControllerConfig.max_heading_acquisition_s`, its
tests, a release, and a deploy. **The 1.06 m figure is quoted in several places**
(CLAUDE.md, the Phase 2 remediation notes) and every one must move to 1.34 m in
the same change, or the docs will assert a disk the code no longer uses.
