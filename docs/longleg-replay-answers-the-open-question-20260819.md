# The long-leg run answers it: the reach work is inert — 2026-08-19

🔑 **On the exact geometry the reach work was built for — three legs of 1.96 /
1.75 / 2.31 m with real mid-drive corrections — the old and new re-aim triggers
make IDENTICAL decisions at all 17 decision points.** The change is measured
inert, not extrapolated inert.

Evidence: `docs/evidence-longleg-3segment-20260817.json`, a card-driven run of
2026-08-17T19:52-19:54Z recovered from the operator's Desktop. ⚠️ It had already
been overwritten in the card's single full-run slot and survived only as a
manual download — see "what nearly happened" below.

## The run

Old code (beta56 era: `max_linear_pulse_ceiling: 14`), `turn_mode: vio`,
`waypoint_tolerance: 0.15`.

| seg | leg | landing | stop | turn | linear | realign | suppressed | post-turn aim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.9646 m | **0.0177** | `target_reached` | 2 | 6 | 1 | 0 | −5.851° |
| 2 | 1.7491 m | **0.0992** | `target_reached` | 2 | 6 | 0 | 0 | +0.605° |
| 3 | 2.3121 m | **0.1330** | `target_reached` | 4 | 8 | 1 | 1 | +10.119° |

3/3 inside tolerance, mean 0.0833 m. This is the corpus's only long-leg run with
mid-drive corrections.

## Replay: 3/3 self-validated, and the verdict is identical

    self-validation                    3/3
    ... also reproducing suppressions  3/3
    geometry cross-check               2 points, max 0.000409 deg / 3e-05 m

    TOTAL   old = 1   new = 1     (budget is 3 per segment)

Only one decision point fires under either rule — segment 1 pulse 4, aim
−20.741° at 0.431 m. Every other point agrees.

## 🔑 Two things this settles

**1. `vio_max_realignments: 3` is ample here.** Maximum corrections needed on any
2 m leg was **one**. The scope cut that kept the budget at 3 was right, and the
budget was never the constraint on this geometry.

**2. The trigger change bought nothing on the geometry it targeted.** It was
designed for far-field misses; on real 2 m legs it changed no decision. Combined
with Gate 5 (8 decision points, identical) and run #4 (15 points, identical),
that is **40 measured decision points across three runs with zero divergence**.

## ⚠️ And one thing it exposes — the floor is now the blind spot

Segment 3, pulse 2:

    range 1.850 m   aim +8.106°   projected landing 0.2609 m

**0.26 m projected against a 0.15 m tolerance, and neither trigger fires** —
because 8.106° is under the 15° correctable floor. This is exactly the far-field
blind spot the change set out to close, still open, with the *floor* now doing
the blocking instead of the old 18° gate.

⚠️ **But do not act on that yet, because the segment landed at 0.1330 m anyway.**
The aim error fell back on its own (8.11 → 5.41 → 4.80°) and the projected miss
never materialised. The projection is a single-pulse extrapolation and its own
docstring says to read it as *"which side of tolerance is this on"*, not as a
landing prediction. Here it was pessimistic by 0.13 m.

That is a genuine argument against lowering the floor: the one case where the
floor blocked a large projected miss is also a case where correcting would have
spent a turn — and its translation — to fix something that fixed itself.

## What nearly happened to this data

The card keeps **ten run summaries but only one full run**, in a slot every new
run overwrites. This run was overwritten within a day and survived only because
the operator had manually downloaded it. Had they not, the single most
informative dataset in the corpus would have been reduced to five numbers per
segment.

That retention gap is logged in `docs/roadmap-20260818.md` Phase 4 and remains
open. Until it is fixed the operating rule is: **download the run JSON
immediately after every run.**

## Where this leaves the reach work

- The **length gates** (`segment_too_long`, `linear_budget_insufficient_for_segment`)
  are real safety additions and stand.
- The **ceiling 14 → 22** remains unexercised: this run used 6-8 pulses of 14.
- The **trigger change** is now measured inert across 40 decision points on
  1.75-2.31 m legs. It is not wrong; it simply does not engage.

⚠️ The banked characterization run is **no longer the top priority** — this data
does what that run was going to do, on the same geometry class, for free. What
remains genuinely untested is the ceiling, which needs a leg over ~5 m.
