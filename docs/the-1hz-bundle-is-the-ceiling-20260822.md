# Position and heading arrive as ONE ~1 Hz bundle — that is the ceiling

**2026-08-22, offline, no mower run.** Measured across both banked Phase 1
captures. `scripts/measure_telemetry_bundling.py`, evidence
`docs/evidence-phase1-bundling-20260822.json`.

## The measurement

| field | straight | shallow arc |
| --- | ---: | ---: |
| report stamp | 8 updates, **2.00 Hz** | 7 updates, **1.75 Hz** |
| position x/y | 4 updates, **1.00 Hz** | 3 updates, **0.75 Hz** |
| `toward` | 1 update | 3 updates, 0.75 Hz |
| VIO heading | 4 updates, **1.00 Hz** | 3 updates, **0.75 Hz** |
| VIO state | 0 | 0 |
| **heading updated without a new position** | **0** | **0** |

🔑 **Position, `toward`, and VIO heading change on exactly the same instants, in
both captures, without a single exception.** They are one bundle. VIO is not an
independent faster channel; it is delivered in the same frame as the position it
describes.

⚠️ **Report stamps run at ~2 Hz and are NOT feedback.** Only every other frame
carries new `sys.toapp_report_data`. Counting stamps doubles the apparent rate
and is the trap this measurement exists to avoid — `last_report_at` moves for
every frame, which is the same instrument gap the beta23 probe hit.

## What this settles

**There is no faster signal to close a loop on.** A continuous controller can
observe at ~1 Hz and no faster, whatever it is written to consume. At the
measured 1.1029 m per 4 s (~0.28 m/s) that is a correction opportunity every
~0.28 m; the vendor's ~0.55 m/s on this same feed corrects every ~0.55 m.
Against a 0.15 m waypoint tolerance, the loop closes at a spatial resolution
**2–4x coarser than the tolerance it is trying to hold**.

That does not refute continuous motion — the vendor demonstrably drives this
way. It does mean a continuous controller is a **feed-forward-dominated** design
with occasional correction, not a tight tracking loop, and it should be designed
as such.

## What it means for Phase 1's test design

🗑️ **"Take more measurements per run" is dead.** The 4 s window is a schema hard
cap (`duration_ms` max 4000) and the bundle is ~1 Hz, so **every run yields ~4
observations and ~3 steps, one of which is usually too short to mean anything**.
No criterion can be written around that. It is the feed, not the exam.

🗑️ **Driving slower to get more samples per metre makes it WORSE.** The sample
count is set by time, not distance, so a slower run buys the same ~3 steps with
shorter chords — and chord length is what sets bearing uncertainty
(`atan(sigma*sqrt(2)/chord)`). Shorter chords are noisier, not richer.

🔑 **For the open pairing question, the lever is rotation per interval, not more
intervals.** The pairing offset's uncertainty is
`bearing_noise_degrees / rotation_degrees`, so it tightens with a faster turn,
not a longer run. Today's `angular 180` gave ~10 deg per interval and
`alpha = -0.253 +- 0.174` from two steps. A tighter arc that rotates ~30 deg per
interval would cut that uncertainty roughly threefold **from the same two
steps** — one run, decisive, instead of several runs at 180 that would each add
two thin rows.
⚠️ Not proposed as authorized work: a tighter arc is a more aggressive
manoeuvre, needs its own corridor and its own review, and `angular 500` is
documented as a *stationary* turn figure whose arc behaviour is unmeasured.

## Recommendation

Before spending more mower time on the compass-mirror criterion, decide whether
it is worth measuring at all. The alternative already has precedent here:
**one-step position prediction error** — predict the next fix from the last one
plus the commanded velocity, which is literally what the controller consumes.
`scripts/replay_position_predictability.py` already measured **0.029 m median /
0.097 m p90** on straight runs, ~5x better than tolerance. It is insensitive to
the pairing convention entirely, because a constant lag is absorbed as a fitted
parameter rather than deciding the verdict.

That question — *mirror identity, or prediction error?* — is the plan decision
worth making. Both are limited to ~3 steps per run either way.
