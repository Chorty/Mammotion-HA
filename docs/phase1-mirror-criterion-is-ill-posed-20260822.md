# The Phase 1 mirror criterion is ill-posed, and the fix is a plan decision

**2026-08-22, offline, no mower run.** Re-poses the failed
`bearing_toward_compass_mirror` check against the two banked captures.
`scripts/reanalyze_mirror_pairing.py`, evidence
`docs/evidence-phase1-mirror-repairing-20260822.json`.

🚨 **THE `no_go` STILL STANDS.** `analyze_phase1_capture.py` is the authority and
its verdict is unchanged. Nothing here edits a threshold, and none of it
authorizes Phase 2 or a mower run. This is the evidence a deliberate plan
revision needs, not the revision.

## The result

Only the pairing the shipped criterion happens to use fails.

| capture | filter | `toward` at START | at MIDPOINT | at END (shipped) |
| --- | --- | ---: | ---: | ---: |
| straight | all 4 steps | 2.008 PASS | 2.008 PASS | 2.008 PASS |
| straight | chord >= 0.15 m | 1.236 PASS | 1.236 PASS | 1.236 PASS |
| shallow arc | all 3 steps | 7.930 PASS | 8.854 PASS | **12.631 FAIL** |
| shallow arc | chord >= 0.15 m | **2.521** PASS | 7.576 PASS | **12.631 FAIL** |

Worst absolute mirror error in degrees, threshold 10.

🔑 **On its two informative steps the arc scores 2.521 deg — the same regime as
the straight-line control's 1.236 deg.** Once `toward` is paired with the start
of the interval it measures, an arc looks exactly as consistent as a straight
run. The straight capture rotated ~1 deg total, so its three pairings are
identical to three decimal places; that is the control confirming the mechanism
is rotation, not the arc being different in kind.

## Two independent defects, not one

**1. Pairing.** The check compares a chord bearing, an interval average, against
one instantaneous `toward`. The three pairings differ by exactly the rotation
over the interval -- `err@end = err@start + rotation`, which holds to 0.001 deg
on every row. At ~10 deg of rotation per interval the criterion's answer is
therefore set by an arbitrary convention, not by the mower.

**2. No minimum chord.** Position noise alone buys a bearing uncertainty of
`atan(sigma*sqrt(2) / chord)`. At the ~0.7 cm RMS the feed shows during
continuous motion, that is **+-12.2 deg** on the straight capture's 0.0456 m
step and **+-7.4 deg** on the arc's 0.0760 m step -- at or above the 10 deg
threshold itself. **A step whose noise bound exceeds the threshold cannot test
anything**, yet both are scored. The arc's shortest step is exactly the row that
still reads 7.930 deg under START pairing, and its noise bound is 7.4.

## What is NOT settled

⚠️ **Which pairing is physically correct is NOT determined by this data.** START
scoring better than MIDPOINT is suggestive -- it would mean `toward` at an
arrival already describes the course over the interval that follows, rather than
the one that precedes it -- but that rests on **two** informative steps from
**one** arc. It is a hypothesis with an obvious test, not a finding.

⚠️ **Do not fix this by picking whichever pairing passes.** That is the same
move as moving the threshold, one level of indirection away. The pairing has to
be argued from what a continuous controller would actually consume, then written
down, and only then measured against.

## Recommended plan revision, for review before any re-run

1. **State the pairing explicitly and justify it physically.** The criterion
   must name which instant's `toward` corresponds to a chord and why.
2. **Add a minimum chord length**, or better, skip any step whose noise bound
   exceeds the threshold -- the bound is computable per step from the chord.
3. **Re-run one arc capture after 1 and 2 are committed**, never before. With
   the current 3 steps per 4 s window, of which 1 is noise-dominated, a single
   arc yields 2 usable rows; that is too thin either way.

Item 3 is the only one needing the mower, and it needs the first two landed
first so the criterion cannot be tuned to its own result.
