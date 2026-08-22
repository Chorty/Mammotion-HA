# What continuous motion is worth: 4.88x, and it is all dead time

**2026-08-22, offline, no mower run.** Reproduce with
`scripts/compare_pulsed_vs_continuous_speed.py`; evidence
`docs/evidence-continuous-speed-gain-20260822.json`.

The operator's goal for this work is **fluidity, not accuracy** -- accuracy is
closed at ~0.089 m mean. So the question worth answering before spending the one
budgeted mower run is how much faster continuous motion would actually be.

## The number

| | | |
| --- | ---: | --- |
| in-pulse speed, 212 banked pulses at linear 400 | **0.2584 m/s** | what the drivetrain does while moving |
| continuous window, today's straight capture | **0.2757 m/s** | 1.1029 m in 4.0 s |
| pulsed **end to end**, the 9.0 m Route B chain | **0.0565 m/s** | 9.0 m in 159.3 s wall clock |

**Effective duty cycle 21.9%. End-to-end speed-up 4.88x. A 9 m route goes from
159 s to about 33 s.**

The 9.0 m Route B chain is the right comparison: its junctions are collinear at
0.000000 deg, so it contains no turns. It is pure straight-line travel plus
settle overhead, measured wall clock from its own banked timestamps.

## 🗑️ What this refutes -- including a claim I made earlier today

The obvious guess, and one I asserted before checking, was that **short pulses
never reach full speed**: today's capture appears to show the mower still
accelerating at the end of a 4 s window (0.243 -> 0.266 -> 0.298 m/s), which
would mean pulsed motion is slow *both* because it stops *and* because it never
gets going.

**That is wrong.** The banked corpus shows a **500 ms** pulse already achieving a
median 0.2422 m/s and a 1500 ms pulse 0.2586 m/s -- indistinguishable from the
0.2757 m/s a continuous window sustains. There is no ramp penalty to recover.

The apparent in-window ramp is the **~1 Hz feed's reporting lag unwinding**, not
the drivetrain. The first reported fix describes where the mower was roughly a
second earlier, so early in-window steps under-count displacement and later
steps over-count as the lag pays back. Total travel across the window
(1.1029 m / 4 s = 0.2757 m/s) matches the pulse corpus, which is the check that
settles it.

🔑 **So the entire gain is removing dead time.** Nothing about going faster,
everything about not stopping. 78.1% of a pulsed run is spent stationary.

## What this means for the decision

✅ **The fluidity case is strong and quantified.** 4.88x is not a marginal
improvement, and it does not depend on the mower doing anything it has not
already demonstrated -- it is the same speed, without the pauses.

⚠️ **4.88x is a CEILING, not a forecast.** A real continuous controller must
still steer, and at ~1 Hz feedback it observes every ~0.28 m. Any correction
that requires slowing, or any conservative speed reduction near a target, eats
into this. Treat 4.88x as the prize being competed for, not the expected result.

⚠️ **The longest continuous run ever performed is 4 seconds.** This extrapolates
a 4 s window to a 159 s route. Nothing has demonstrated that speed is sustained,
that BLE holds, or that cross-track stays bounded over a minute of unbroken
motion.

🔑 **A cheaper lever exists and is untested: command a higher speed.** Linear 400
is ~47% of the app's ±850 full scale, and the vendor drives at ~0.55 m/s, twice
what we command. Raising commanded speed multiplies with the duty-cycle win and
needs no new controller -- but it also increases the blind distance per ~1 Hz
correction, which is exactly the budget a continuous controller is short of.
That trade is unmeasured and is not proposed here.

## Recommendation

The fluidity case justifies continuing. But the biggest unknown is no longer the
telemetry criterion -- it is whether **motion sustains beyond 4 seconds at all**,
which is capped by the probe schema (`duration_ms` max 4000) rather than by
anything measured.

Worth considering before the criterion re-run: the one budgeted mower run may be
better spent on a **longer continuous window** than on a second arc. A second arc
sharpens a prediction constant; a longer window tests the assumption the entire
4.88x rests on. Raising the schema cap is a deliberate change with its own
safety review, and the 1.5 m distance limit in the controller's fail-closed list
would bind first at present speeds.

That is a decision for the operator, not a conclusion.
