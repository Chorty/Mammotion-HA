# The mirror criterion, repaired — and it still says `no_go`

**2026-08-23.** Implements the independent review's recommendation: **repair**
`bearing_toward_compass_mirror` rather than replace it with prediction error.
Two changes to `scripts/analyze_phase1_capture.py`, **both argued from physics
and neither touching the 10 deg threshold**.

## What changed

**1. `toward` is paired with the START of the interval, not its end.**

`bearing` is a chord between two fixes about a second apart — an interval
*average*. `toward` is a single instant. On a body rotating ~10 deg per interval
those differ by the whole rotation, so the pairing decides the verdict: the same
arc scores **2.52 deg** at the start and **12.63 deg** at the end. The end
pairing is what produced the 2026-08-22 `no_go`.

The start is correct because **it is the only heading a controller has.**
Predicting the chord it is about to travel, it knows the heading it holds now;
the heading it will hold when the next fix arrives is exactly what it does not
know. Certifying a reference the controller cannot obtain certifies the wrong
thing.

⚠️ **The start is not exactly right either.** Solving all 8 informative arc steps
for the pairing that would zero the error gives **alpha = -0.149 +- 0.043**
*(corrected 2026-08-23 from -0.165, which used mirror constant 90.00 instead of
90.13; dalpha/dK = 0.1214 per degree)*,
which excludes the start (alpha = 0) at ~3 sigma as well. END is merely far
worse, at ~22 sigma. The residual is unexplained, is not modelled, and the
unchanged 10 deg threshold absorbs it.

**2. `MIN_MOVING_STEP_M` raised 0.01 m -> 0.15 m.**

Position noise alone buys `atan(sigma*sqrt(2)/chord)` of bearing uncertainty. At
the **sigma = 0.0031 m measured across 16 steady in-window steps**, that is
**+-12.2 deg** on the straight capture's 0.0456 m step and **+-7.4 deg** on the
arc's 0.0760 m step — at or above the threshold itself. **A step whose noise
bound exceeds the threshold cannot test anything.** The old 0.01 m floor was
three orders below the noise floor and excluded only exactly-zero steps. At
0.15 m the bound is 1.7 deg.

## 🚨 The repair does NOT flip the verdict

```
verdict: no_go   failed: ['shallow_arc.bearing_toward_compass_mirror']
  straight     PASS  3 steps, max error 1.236 deg
  shallow_arc  FAIL  2 steps, max error 2.521 deg
```

**The failure mode has changed, and that is the whole finding.** The arc's mirror
error is now **2.521 deg against a 10 deg threshold** — comfortable. It fails
because a correctly sized minimum chord leaves it **2 informative steps against
the 3 required**.

🗑️ **CORRECTED 2026-08-23 — I first wrote that the 4 s arc "cannot" produce 3
informative steps. That is false.** Four of the five banked captures produced
exactly 3 informative chords inside their first 4 seconds; only the arc180 did
not, because its fourth fresh arrival never came before the 4000 ms cutoff
(arrivals at 882.7 / 1896.8 / 2919.2 ms). The design is **fragile, not
impossible** — three arrivals can never be enough once the short spin-up chord is
excluded, so it depends on a fourth landing inside a hard boundary. See
`docs/phase1b-arc-protocol-20260823.md`.

🔑 **It is still a defect in the CAPTURE DESIGN, not in the mower and not in the
criterion** — just a fragility rather than an impossibility. This is the ~1 Hz ceiling
(`docs/the-1hz-bundle-is-the-ceiling-20260822.md`) arriving where it was always
going to.

⚠️ **It must not be fixed by lowering the required step count from 3 to 2.** That
is the move this whole exercise exists to avoid, and
`test_the_banked_arc_now_fails_on_STEP_COUNT_not_on_error` pins the current
behaviour so nobody does it quietly.

## What would legitimately flip it

The `angular 120`, 8 s out-of-sample arc
(`docs/evidence-arc120-outofsample-20260823T001500Z.json`) has **7 arrivals and
6 informative steps at >= 0.15 m**, with a START-paired max error of
**2.385 deg**. It would pass comfortably.

It cannot be scored today: `EXPECTED_CONTROLS` hardcodes `angular_speed: 180`
and `EXPECTED_DURATION_MS = 4000`, so that capture is not admissible as a Phase 1
arc.

🗑️ **DECIDED 2026-08-23: it is NOT admitted.** An independent Codex adjudication
rejected it, and refuted the argument I had made for it (see the correction
above). Both pre-registered dimensions would have had to move at once
(angular 180 -> 120 and 4000 -> 8000 ms) with the outcome already known.
`docs/evidence-arc120-outofsample-20260823T001500Z.json` stays **exploratory and
corroborating evidence only**. What was registered instead:
`docs/phase1b-arc-protocol-20260823.md` — the SAME `angular 180` at 8000 ms, with
every criterion unchanged, written before any capture exists.

## What was NOT done, deliberately

**Prediction error was not added as a second criterion.** The review recommends
adding it ANDed with the repaired mirror check, and that is right in principle —
but its threshold is unresolved. The proposed 0.10 m is **breached at 0.1418 m**
by the banked 8 s run, the budget does not close against the 0.065 m sensing
floor, and the failing step follows an 810 ms refresh write (a documented
BLE-stall class). Picking a threshold now, knowing which value would pass, is
the exact failure this repair exists to avoid. It stays open.

## Verification

822 pytest (up from 819), ruff, ruff format, mypy 29 files, 91 frontend, ten
pre-commit hooks, `check_accepted_profile` ACCEPTED. Three new tests pin the
pairing, the minimum chord, and the fact that the banked arc now fails on step
count rather than on error.
