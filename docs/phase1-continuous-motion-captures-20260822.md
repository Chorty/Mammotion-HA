# Phase 1 continuous-motion captures — `no_go`, and the criterion is why

**2026-08-22. Two separately authorized 4 s windows ran on beta70. Both moved
the mower, both stopped on command, and the gate was verified disarmed after
each. The offline analyzer returns `no_go` on exactly one criterion.**

Evidence, banked including the failure:

| file | sha256 (first 16) |
| --- | --- |
| `docs/evidence-phase1-straight-20260822T202600Z.json` | `beb4b9b8568e5231` |
| `docs/evidence-phase1-shallow-arc-20260822T203400Z.json` | `c20c1b103cb44ed1` |
| `docs/evidence-phase1-corridors-20260822T203400Z.json` | `440c39ea422484e2` |
| `docs/evidence-phase1-analysis-20260822T203400Z.json` | the verdict, with all three input digests |

## The verdict

`shallow_arc.bearing_toward_compass_mirror` — max error **12.631°** against the
written 10° threshold. Every other criterion passed on both captures: **17 of 17
straight, 17 of 18 arc** *(corrected 2026-08-23; the arc has 18 criteria)*.

**The `no_go` stands. It was not re-run, and the threshold was not moved.**

## What the straight capture measured

This is the result Phase 1 existed to get, and it is good.

| | |
| --- | --- |
| fresh position arrivals | **4** (need >= 3), at 948 / 1971 / 2983 / 3903 ms |
| max arrival gap incl. boundaries | **1023 ms** (limit 2000) |
| compass mirror error | **max 2.008°** over 4 moving steps |
| travel | 1.1029 m on movement heading 277.277° |
| refresh writes | 19 sent, 19 completed, ordered |
| containment | every in-window sample inside the frozen corridor |

Position fixes keep arriving at ~1 Hz **during** motion, with no gap approaching
the 2 s limit. The feed does not degrade while the mower drives.

## Why the arc failed, measured rather than assumed

🔑 **The failure is dominated by how the criterion pairs its two quantities, not
by `toward` being wrong.**

The check compares a **chord bearing** — computed between two position fixes
about one second apart, so an *interval average* — against a **single `toward`
sample**. On the arc the mower rotates about 10° between consecutive fixes, so
the answer swings by that whole rotation depending on which end of the interval
you take `toward` from:

| step | chord bearing | error using `toward` at START | at END | rotation over step |
| --- | ---: | ---: | ---: | ---: |
| 0 -> 7 | 285.407° | **7.930°** | 9.778° | 1.849° |
| 7 -> 17 | 278.150° | **2.521°** | 12.631° | 10.110° |
| 17 -> 27 | 267.835° | **2.316°** | 11.317° | 9.000° |

The analyzer pairs with the END sample, which is what produces 11–12°. Pair with
the START and the same two steps read **2.5°** and **2.3°** — comfortably inside
the threshold. Nothing about the mower changed between those two columns; only
the choice of which instant to compare an interval average against.

**VIO independently corroborates that `toward` tracked the rotation correctly.**
Over the same steps VIO heading rotated **−9.92°** and **−9.13°** while `toward`
rotated **+10.11°** and **+9.00°** — agreeing to within ~0.2–1.1° once the known
mirror sign flip is applied. Two independent heading sources agree on how far
the body turned.

Step 0 -> 7 has a separate and equally benign explanation: its chord is only
**0.076 m** long, against this hardware's documented 2–4 cm absolute position
noise floor. A 7.6 cm chord cannot carry a bearing to better than roughly ±15°,
so that row is noise, not signal.

The straight capture is the control that confirms the mechanism: it rotated
~1° total, the start/end pairing therefore barely matters, and its worst mirror
error is **2.008°**.

## What this does and does not license

✅ **Established.** Position fixes arrive at ~1 Hz throughout a 4 s window with
no gap over 1081 ms; `toward` changes progressively during an arc (3 changes,
20.96° total) rather than arriving as one post-hoc step; the 100 ms cache
sampler works with zero extra in-window BLE report requests; both captures
stayed inside their prevalidated corridors; the mandatory stop confirmed both
times.

🗑️ **Refuted, and this corrects a same-session claim of mine.** I first read the
failure as "`toward` is not usable as a heading reference during rotation."
The per-step arithmetic above does not support that, and the VIO cross-check
contradicts it.

⚠️ **NOT established, and do not infer it.** This does not turn the `no_go`
into a `go`. A criterion that is ill-posed for a rotating body has to be fixed
**deliberately, in the plan, before** any re-run — never by editing a threshold
after seeing the data that failed it. That move is precisely what this repo's
evidence discipline exists to prevent, and a revised criterion owes its own
review.

## The open question for whoever revises the criterion

A chord average and an instantaneous heading are different quantities on a
rotating body. Any replacement needs to say which one a continuous controller
would actually consume. Two candidates, neither measured:

1. **Compare like with like** — integrate `toward` across the interval and
   compare that average against the chord.
2. **Shorten the interval** — but position fixes arrive at ~1 Hz, so there is no
   shorter interval available. This is the same 1 Hz that makes stop-measure-go
   expensive, and it bounds how tightly any `toward`-based check can be posed.

Option 2 is bounded by hardware and option 1 is arithmetic on data we now have,
so option 1 is the cheap one — and it can be tested against these two banked
captures with no mower run at all.
