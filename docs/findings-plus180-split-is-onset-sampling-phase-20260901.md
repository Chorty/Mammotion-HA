# The +180 2a split is the onset interval's sampling phase, not the plant (2026-09-01)

**Offline. No mower, no dispatch, no gate change.** Derived from the two banked
+180/7000 runs, both scored through the *same* shipped E-VIO code
(`_step_response_vio_analysis`, `baseline_ms=3000`, `step_ms=7000`).

## 1. The finding

Step-phase VIO rates, °/s:

```
banked  2026-08-30 (2a PASS, half_diff 0.130):  -5.68 -13.56 -11.44 -14.29 -11.40 -11.55 -11.16
repeat  2026-09-01 (2a FAIL, half_diff 3.405):  -1.35 -14.35  -9.52 -13.00 -12.38 -10.95 -12.03
```

| | banked | repeat | difference |
| --- | --- | --- | --- |
| onset (interval 1) | **-5.68** | **-1.35** | **4.33 °/s** |
| intervals 2-7 mean | -12.233 | -12.038 | **0.195 °/s** |
| intervals 2-7 median | -11.495 | -12.205 | 0.710 |
| intervals 2-7 sd | 1.337 | 1.668 | — |
| published 2a second-half mean | -11.3594 | -11.7838 | **0.424 °/s** |

🔑 **Once the onset interval is set aside the two runs agree to 0.195 °/s on the
mean and 0.424 °/s on the statistic 2a itself uses.** The plant did the same thing
both times: it rotates at roughly **-11.5 to -12 °/s** at +180.

🔑 **The entire 2a disagreement is interval 1.** At ~1 Hz VIO updates and a 7000 ms
step, the first interval straddles the command onset, so its value depends on
**where the sample boundary happened to fall relative to the step** — i.e. how much
of the documented ~1-2 s onset lag got averaged into it. That is sampling phase,
not physics.

## 2. What follows for further runs

🗑️ **A third +180/7000 run does NOT break the tie.** It re-samples the same
sampling phase. n=3 would give a third draw from a coin flip, and would be read —
wrongly — as evidence about the mower.

⚠️ **Do NOT "just exclude the onset interval".** Re-splitting the remaining six
3/3 flips both verdicts the OTHER way:

| | banked | repeat |
| --- | --- | --- |
| ex-onset halves (°/s) | -13.097 / -11.370 | -12.290 / -11.787 |
| ex-onset `half_diff` | **1.727 — FAIL** | **0.503 — PASS** |

**Choosing that rule now, knowing which runs it flips, is exactly the failure the
2026-08-23 mirror-criterion review rejected.** Any revision must be predeclared,
justified before the verdicts are recomputed, and validated against all four
banked runs — not selected after the fact.

## 3. The structural point

2a asks "did the step reach steady rotation". At 7000 ms with ~1 Hz feedback it
gets **7 informative intervals**, one of which is contaminated by onset lag, and
the half-phase split gives that one interval ~29% of the first half's weight. So
the statistic partly measures onset placement rather than steadiness.

🔑 **The lever that would actually address this is a LONGER step**, which shrinks
the onset interval's share — not another repeat and not a re-split. ⚠️ **That is
blocked and deliberately so:** `step_ms` is schema-capped at **7000** and
`_STEP_RESPONSE_MAX_TOTAL_MS` at **16000**, so it needs a code change, a release
and a deploy — and raising it is a **safety-bound change** (more open-loop travel
on an uncorrected curve), which the 2026-08-30 predeclaration required to be
stated plainly rather than slipped through. It is not a tonight action.

## 4. Status

n = 2 at +180/7000. Under E-VIO the *verdicts* are split 1-1; the *measurements*
are not split at all. **No route-1 configuration has a reproduced 2a pass**, and
`tau = 0.80 s` stays unquotable — the banked run's tau was computed off a
first-half mean that this analysis shows was set by onset placement.

This document authorizes nothing and changes no code, threshold or bound. It
exists so that the next decision is made on the measurement rather than on the
verdict.
