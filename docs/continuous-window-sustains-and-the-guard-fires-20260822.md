# The 8 s window sustains, and the distance guard fires — 2026-08-22

Two supervised runs on beta71, each separately authorized, in daylight with the
operator present. Both moved the mower, both stopped, both stayed inside their
prevalidated corridors, and the gate was verified disarmed after each.

| | evidence |
| --- | --- |
| 8 s sustain run | `docs/evidence-8s-continuous-window-20260822T233000Z.json` |
| guard-firing run | `docs/evidence-travel-guard-fired-20260822T234500Z.json` |
| corridors | `docs/evidence-8s-corridor-20260822T233000Z.json`, `docs/evidence-travel-guard-corridor-20260822T234500Z.json` |

Both answer questions that were open this morning, and both were the reason the
one budgeted run got spent here rather than on a second arc.

## 1. Speed SUSTAINS past 4 s — the 4.88x case no longer extrapolates

8000.7 ms delivered, **1.8952 m**, reason `completed`, **8 position arrivals ->
7 steady-state steps** against the 3 a 4 s window yields.

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| m/s | 0.199 | 0.312 | 0.242 | 0.130 | 0.297 | 0.294 | 0.261 |

Steady-state mean **0.2479 m/s**; trend **+0.0073 m/s per step**, i.e. drifting
*up*. First half 0.2027, second half 0.2454. **No decay over twice the
previously tested duration.**

🔑 **This matters because the 4.88x fluidity estimate extrapolated a 4 s window
to a 159 s route** (`docs/what-continuous-motion-is-worth-20260822.md`). It no
longer rests on a 4 s sample. It still extrapolates 8 s to 159 s.

**BLE also sustained.** 39 refresh writes sent, **39 completed, in order**;
median write 123.8 ms, only 7 of 39 slower than the 200 ms interval; delivered
cadence median 178.1 ms, first half 174.6 -> second half 194.3 ms. A mild
widening and one 810 ms outlier of the kind already on record — nothing like the
collapse a long run was feared to hit.

**Open-loop tracking was better than expected.** After spin-up the chord bearing
held **241.3-243.5 deg** against a frozen 243.5 deg heading: roughly **5 cm of
cross-track over 1.9 m**, with no steering at all.

## 2. The distance guard FIRES, and its published overshoot is honest

The first run's guard was set at 2.0 m and **did not trip** — the mower reached a
sampled 1.716 m before the 8 s clock ended it. So the abort path was still
unexercised on hardware, and the second run set the guard at 1.5 m to force it.

```
guard tripped at 6771.6 ms, sampled travel 1.5731 m
window ended    7026.4 ms   aborted_early: true   (not the 8000 ms clock)
refreshes       33 of a possible 40
stop            confirmed
```

🔑 **The overshoot prediction held.**

| | |
| --- | ---: |
| guard bound | 1.500 m |
| published `corridor_must_cover_m` | **1.850 m** |
| **actual travel** | **1.776 m** |

Real overshoot **0.276 m** against the predicted **0.35 m** — conservative by
7 cm, right direction and right magnitude. A corridor sized on
`corridor_must_cover_m` contained it with room, which is exactly what that field
exists for.

⚠️ **Do NOT tighten `_PROBE_TRAVEL_GUARD_OVERSHOOT_M` to 0.28 on this.** It is
**one** sample, the constant is a safety margin so being wrong is asymmetric, and
a tighter value buys nothing operationally except permission to freeze a slightly
smaller corridor. The measured 0.276 m confirms 0.35 is the right *kind* of
number; it is not a mandate to shave it.

## 3. Speed varies run to run more than it varies within a run

| run | window | travel | overall m/s |
| --- | ---: | ---: | ---: |
| straight, Phase 1 | 4.0 s | 1.1029 m | **0.2757** |
| 8 s sustain | 8.0 s | 1.8952 m | **0.2369** |
| guard run | 7.03 s | 1.7760 m | **0.2528** |

Spread **0.237-0.276 m/s**, about 15%, while the *within-run* trend is flat or
slightly positive. ⚠️ **0.2757 m/s is the top of the range, not the planning
number.** Anything sized on it — including the "9 m in ~33 s" figure — should use
~0.25 m/s instead, which makes the honest end-to-end speed-up nearer **4.5x**
than 4.88x. The conclusion is unchanged; the arithmetic should not overstate it.

## What is still open

- **The prediction criterion is still unvalidated out of sample.** `k_ang` has
  only ever been fitted on the single arc it was scored against. That needs a
  second arc at a different `angular_speed`, with both constants frozen in a
  committed file beforehand — see
  `docs/phase1-criterion-revision-proposal-20260822.md`, still a proposal.
- ⚠️ **An arc corridor does not currently fit.** Freezing from both of tonight's
  end positions FAILED the arc leg on area margin (0.84 m and 0.01 m against the
  required 1.2 m) while the straight leg passed. The mower has driven into a
  part of "Backyard Right" with no room to curve. **A future arc run needs the
  mower repositioned first**, and the corridor freezer correctly refuses to mark
  such a route prevalidated.
- The Phase 1 analyzer verdict remains **`no_go`** and nothing here changes it.
