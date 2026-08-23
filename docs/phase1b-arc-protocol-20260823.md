# Phase 1b — a pre-registered arc protocol, written BEFORE the run

**2026-08-23. Registered before any capture exists. Nothing has been driven
against this.** It replaces the Phase 1 arc, which failed and whose failure is
now understood.

Written after an independent Codex adjudication that **rejected** the tempting
shortcut. Read `docs/codex-adjudication-20260823.md`.

## 🗑️ First, a claim of mine that is REFUTED

I wrote that the plan's 4 s arc **"cannot test its own criterion"** — that a 4 s
window at ~1 Hz physically cannot yield the 3 informative chords the repaired
mirror criterion needs.

**That is false.** Counted across all five banked captures, informative
(>= 0.15 m) chords **inside the first 4 seconds**:

| capture | fresh arrivals by 4 s | informative chords by 4 s |
| --- | ---: | ---: |
| Phase 1 straight | 4 | **3** |
| **Phase 1 arc180** | **3** | **2** |
| 8 s straight | 4 | **3** |
| guard straight | 4 | **3** |
| arc120 | 4 | **3** |

**Four of five made it.** The arc180 missed because its fourth fresh arrival
never came before the 4000 ms cutoff — arrivals at 882.7, 1896.8, 2919.2 ms and
then the window closed. The design is **fragile, not impossible**: it needs a
fourth arrival inside a hard boundary, and the routinely short spin-up chord
means three arrivals are never enough.

⚠️ **That refutation removes the main argument for admitting the arc120 capture
as a substitute**, which is why this protocol exists instead.

## The decision that was NOT taken

The 8 s `angular 120` arc already banked
(`docs/evidence-arc120-outofsample-20260823T001500Z.json`) has 6 informative
chords and a START-paired max of 2.385 deg. It would pass.

**It is not being admitted.** Both pre-registered dimensions would have to move
at once (angular 180 -> 120, 4000 -> 8000 ms), the outcome is already known, and
the physical-incapability rationale that might have justified it is refuted
above. Admitting it would be outcome-informed reasoning.

It stays **exploratory and corroborating evidence only.**

## The registered protocol

**One arc capture. Nothing about the criterion changes.**

| | |
| --- | --- |
| command | `linear 400`, `angular 180` — **unchanged from the plan** |
| duration | **8000 ms** — the only change, and it is a *duration*, not a control |
| refresh | 200 ms |
| in-window sampling | 100 ms |
| `max_travel_m` | 1.5 m |
| minimum chord | 0.15 m — unchanged |
| informative chords required | >= 3 — unchanged |
| mirror threshold | **10 deg — unchanged** |
| pairing | interval START — as repaired |

🔑 **Only the window length moves, and it moves for a reason that is independent
of any score:** three fresh arrivals cannot yield three informative chords once
the spin-up chord is excluded, so the window must be long enough to make a
fourth arrival reliable rather than lucky. At ~1 Hz, 8 s yields ~7.

The straight capture **keeps its 4000 ms duration** and is not re-run. Any
analyzer change must express duration **per control**, never as a menu of
accepted durations — a menu is how an after-the-fact choice hides.

## Consequence: the 2026-08-22 arc is now INADMISSIBLE, not merely failing

With duration expressed per control, re-running the analyzer over the banked
Phase 1 pair now reports:

```
verdict: no_go
failed : shallow_arc.control_profile
         shallow_arc.maximum_position_arrival_gap
         shallow_arc.bearing_toward_compass_mirror
durations required: {'straight': 4000, 'shallow_arc': 8000}
```

`control_profile` fails because the capture is 4000 ms where 8000 is now
required, and the arrival-gap check fails as a direct consequence — the window
is measured to 8000 ms, so the trailing boundary gap from its last arrival at
2919 ms is 5081 ms against a 2000 ms limit.

🔑 **That is the honest outcome and it is not a regression.** The old arc is not
a Phase 1b capture; it is a Phase 1 capture, and Phase 1's verdict stands at
`no_go` on its own terms. The straight capture is unaffected and still passes at
4000 ms.

⚠️ **A Phase 1b `go` therefore requires a NEW arc.** No banked capture can supply
it, which is the point.

## Safety envelope

Unchanged from the runs of 2026-08-22, plus the beta72 guard fixes:

- fresh corridor scan and freeze, >= 1.2 m area and >= 1.5 m keep-out margin,
  sized to `corridor_must_cover_m` **as reported by the dry run**, not to
  `max_travel_m`;
- facing re-derived two ways (compass mirror and VIO) and agreeing;
- 0.30 m start-drift abort, endpoint never re-derived;
- gate armed **inside** the `try` whose `finally` disarms and verifies;
- explicit operator authorization for that single window;
- daylight, blades off, operator present, e-stop accessible.

## Pre-registered BLE-stall rule — declared before any capture

The prediction criterion (still **not implemented**) is exposed to a documented
failure class: a stalled refresh write lets the device watchdog stop the motor
mid-window. The banked 8 s run's worst prediction error, **0.1418 m**, follows an
**810.1 ms** refresh write.

Excluding that row *after* seeing it fail would be the same trap as moving a
threshold. So the rule is declared now, derived from the protocol rather than
from any observed value:

> With refresh interval `R` (200 ms), a **cadence stall** is any of: a refresh
> error; a sent refresh with no recorded completion; or a gap between successive
> successful refresh completions exceeding **`3R` = 600 ms** (two missed resend
> opportunities). Every prediction interval overlapping a stall, **plus the
> immediately following interval** (the ~1 Hz feed can report displacement one
> arrival late), is **ineligible**.

Two conjunctive criteria, never one:

1. **Kinematic accuracy** — over *eligible* steady-state steps only, and
   requiring >= 3 of them, or the result is **insufficient data, not a pass**.
2. **Transport integrity** — reported separately. An excluded row can never
   become a pass; it can only fail the transport criterion.

⚠️ Applying this retroactively to the banked 8 s capture is diagnostically
useful and **cannot** convert it into confirmatory evidence.

⚠️ `3R` is protocol-derived. A device-watchdog timeout constant would be a better
basis; it was searched for in this repo and not found.

## Threshold for the prediction criterion, if it is ever added

Derived **before** scoring, by Codex, from the tolerance budget alone:

**max eligible steady-state one-step error <= 0.085 m**
= 0.150 m waypoint tolerance − 0.065 m sensing floor.

Scored against the five banked captures afterwards: **four pass, one fails** —
the 8 s straight at 0.1418 m. The threshold was **not** adjusted in response.

⚠️ This supersedes my own 0.10 m, whose stated justification ("two thirds of
tolerance, leaving a third for the steering law and sensing floor") did not
close: one third is 0.05 m against a 0.065 m floor.

⚠️ **The prediction criterion remains unimplemented and unadopted.** Adding it is
a separate decision.
