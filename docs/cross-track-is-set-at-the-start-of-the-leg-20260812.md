# The landing is decided in the first pulse, and the trigger is range-blind

**2026-08-12, off-mower.** Written after Run C failed on segment 1
(`target_requires_reverse_recovery`, 0.1715 m) despite the beta42 guard fix, and
after that fix turned out to explain almost none of the error.

Evidence: every `docs/evidence-beta32-4segment-*.json` on record — 12 completed
approaches, 46 straight-pulse pairs, 92 per-pulse bearing samples.

## 1. beta42's fix is correct and nearly irrelevant

beta42 made the re-aim guard project to the end of the next pulse rather than to
the closest approach. Applied to Run C's suppressed decision, using the pulse's
**actual** measured travel:

```
guard projected                   0.1300      (tolerance 0.150)
same model, ACTUAL pulse travel   0.1264
what it actually landed           0.1723
```

**The pulse-overshoot term accounts for ~4 mm of a 42 mm error.** The arithmetic
is right — the chord `2·d·sin(a/2)` is exactly where you land if you drive
distance `d` pointed `a` off — but it addresses a minor term. It stays, because it
is free and correct. It is not the fix this problem needed.

## 2. What actually moves: the bearing to the target, not the mower

Heading drift while driving straight is **small**. Across 46 consecutive
straight-pulse pairs with no turn between them:

```
mean -1.38 deg/pulse, sd 8.97, mean |drift| 2.84 deg
```

and the spread is dominated by a handful of very short end-of-segment steps where
a travel bearing is barely defined (the −56.09 and −22.72 outliers are 0.13 m and
0.10 m steps). The mower drives acceptably straight.

**The bearing TO THE TARGET, however, rotates enormously — and it is a pure 1/d
effect:**

| remaining distance | n | median rotation per pulse | max |
| --- | --- | --- | --- |
| > 1.0 m | 29 | **1.47°** | 7.29° |
| 0.5 – 1.0 m | 14 | 10.04° | 28.85° |
| 0.3 – 0.5 m | 25 | 23.17° | 132.34° |
| < 0.3 m | 24 | **33.73°** | 136.57° |

This is geometry, not a defect. A fixed lateral offset subtends a larger angle the
closer you get. **Consequence: any decision made from an aim error inside ~0.5 m
is stale before the pulse it governs has finished.** The guard is being asked a
question in units that stop meaning anything exactly where it is being asked.

## 3. 🔑 The cross-track is CONSTANT through the leg. The angle only grows because
the range shrinks.

This is the finding. Both failing segments, traced pulse by pulse — `aim` is the
angle between where the mower travelled and where the target was, and
`cross-track` is `distance · sin(aim)`, the miss it implies:

**Run C, segment 1 (landed 0.1715, reverse-recovery):**

```
pulse   dist     aim    cross-track    18 deg trigger?
  1    1.867   -6.03°     0.1963 m     no
  2    1.525   -7.30°     0.1937 m     no
  3    1.048  -11.70°     0.2125 m     no
  4    0.653  -19.85°     0.2218 m     FIRES
  5    0.295  -67.96°     0.2736 m     FIRES
  6    0.274  -37.61°     0.1673 m     FIRES
```

**2026-08-11, segment 2 (landed 0.1797, reverse-recovery):**

```
pulse   dist     aim    cross-track    18 deg trigger?
  1    1.602   -5.17°     0.1443 m     no
  2    1.167   -7.02°     0.1427 m     no
  3    0.679  -12.50°     0.1469 m     no
  4    0.325  -28.44°     0.1546 m     FIRES
```

**The cross-track barely changes: 0.196 → 0.222, and 0.144 → 0.155.** The mower
is driving a straight line that misses the target by a fixed perpendicular
distance, and it is doing so from the very first pulse. The *angle* triples only
because the range collapses.

Run C's segment 1 was already missing by **0.1963 m against a 0.150 m tolerance
after pulse 1**, with 1.87 m still to run and an aim error of 6°. It was never
going to arrive, and nothing in the control law objected until pulse 4.

## 4. So the defect is the TRIGGER, and it is backwards

The mid-drive re-aim fires on an **angle** — effectively `aim > 18°`
(`max(vio_realign_threshold_degrees, heading_tolerance_degrees)`). An angle is
range-blind, and the miss it permits scales with distance:

| remaining | cross-track allowed by an 18° trigger |
| --- | --- |
| 3.0 m | 0.93 m |
| 1.5 m | 0.46 m |
| 0.5 m | 0.15 m |
| 0.3 m | 0.09 m |

**The trigger is most permissive exactly where an error is cheapest to fix and
most expensive to keep, and most aggressive where the error is already harmless.**
That is the wrong way round, and it explains why long legs fail more often: the
same angular tolerance buys proportionally more cross-track.

It also explains why *correcting late does not rescue it.* By 0.5 m the bearing is
swinging 10–34° per pulse, the correction turn itself translates the mower, and
the minimum pulse is a large fraction of what remains.

**And the guard already computes the right quantity.** `distance · sin(aim)`
against `waypoint_tolerance` is the guard's own suppression criterion. The trigger
and the guard are asking the same question in different units, which is why a
band exists where the trigger never fires while the miss is already unacceptable.

## 5. Corroboration: the segment is decided by 0.5 m out

Perpendicular miss measured at the last pulse where the mower was still ≥ 0.5 m
from its target, against the final landing, n = 12:

```
perp <= 0.113 m  ->  6 of 6 reached, worst landing 0.1030
perp >= 0.147 m  ->  4 of 6 MISSED
                     Pearson r = 0.595 over the whole set
```

r² = 0.354 is moderate, so this is a tendency and not a law — but the split at the
extremes is stark, and it agrees with the 2026-08-10 model
`landing = 0.62 × leg·sin(initial_aim) + 0.065`, which also said the landing is
set at the *start* of the leg.

## 6. What this suggests, and what it does not

**Suggests:** trigger the mid-drive re-aim on the **perpendicular miss** rather
than on an angle — correct when driving on would miss the disc, which is exactly
the negation of the guard's own criterion. No new constant; the trigger and the
guard would finally speak the same language. On Run C's segment 1 that fires at
**pulse 1**, at 1.87 m out, where a correction is cheap, effective, and has 1.8 m
of leg left to absorb the turn's translation.

⚠️ **It does not rescue the 2026-08-11 case cleanly.** That segment's cross-track
sat at 0.1443 → 0.1469 m, just *under* the 0.150 tolerance, until pulse 4. A pure
"miss > tolerance" trigger fires there no earlier than the current one does. That
case needs either a margin — which this project has twice declined to fit — or an
acceptance that a leg starting 0.145 m off-line is marginal by construction.

**It does not establish that early correction works.** No run has yet corrected at
1.8 m out on this control law. The mechanism says it should be cheap (turn
translation ~0.01–0.05 m against 1.8 m of remaining leg to absorb it), but that is
an argument, not a measurement.

**It does not touch reach.** Long single legs reach reliably — 2/3/4 m at 0.0690 /
0.0928 / 0.1023 m. Reach and final-approach accuracy are separable, and only the
first is solved.

## 7. Status

- `max_linear_pulse_ceiling: 14` is adopted and deployed; **Gate 5 re-pass is
  still owed and should not be attempted while a known defect can end a segment in
  reverse-recovery.**
- beta42's guard fix stays: correct, cheap, minor.
- The trigger change above is **not implemented**. It is a motion-control-law
  change resting on n = 2 failures, and the honest next step is a single-variable
  hardware test of *early* correction before any of it is frozen into a profile.
