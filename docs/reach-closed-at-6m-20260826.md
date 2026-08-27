# Reach is closed at 6.0 m — 2026-08-26

Supersedes `docs/reach-5m-20260826.md` as the reach headline; that document's
method and traps still stand and should be read first.

One supervised, explicitly authorized single vector segment on beta78 +
PyMammotion 0.8.12.post3, daylight, starting aligned, no junction turn.
**Verdict: `target_reached` at 0.11440 m against a 0.15 m tolerance.** The gate
was armed for the run and verified disarmed immediately after, from the live API
and RAW `core.config_entries`.

**Reach is now CLOSED.** `_MAX_SEGMENT_LENGTH_M` is a hard 6.10 m pre-dispatch
refusal (`segment_too_long`), so 6.0 m is the largest single segment that can
exist on this build. There is nowhere further to go on this axis. The next
questions are chains and post-turn accuracy, not reach.

Evidence: `docs/evidence-reach-6m-beta78-20260826.json`, dry run
`docs/evidence-reach-6m-beta78-dryrun-20260826.json`.

## The three-point reach curve

| | 4 m (2026-08-11) | 5 m (2026-08-26) | **6 m (2026-08-26)** |
| --- | --- | --- | --- |
| Landing | 0.1023 m | 0.1015 m | **0.11440 m** |
| Linear pulses | 11 | 17 | **19 of a 22 ceiling** |
| Mid-drive realignments | — | 1 | **2** |
| Cross-track at finish | — | 0.09373 m | 0.09387 m |

**Landing does not degrade with distance.** 0.102 / 0.102 / 0.114 m across a 50%
increase in length. All three stopped on TOLERANCE with the ceiling never
binding, so none of them found the controller's limit.

## What is actually tightening: corrections, not landing

🔑 **The mid-leg aim divergence grows with distance and is being MANAGED at 6 m,
not avoided.** Travel-direction error against the path, per forward pulse:

```
-1 -1 -0 -1 -2 -2 -3 -4 -3 -4 -7 -6 -8 -10 -13 -11 -7 -13 -27
```

It ends at **−27.48°**, the largest terminal heading error on record, and still
growing at the finish. The run landed anyway because two corrections held
cross-track to 0.094 m. The re-aim fired at pulse 15 (aim error 17.016°) and
again at pulse 16 (16.579°), both closing on `target_heading_reached` — the dip
to −7 at pulse 17 is those corrections taking effect.

⚠️ **The binding constraint at 6 m is the CORRECTION budget, not the pulse
ceiling.** This run used 2 of the default `vio_max_realignments: 3`. The pulse
ceiling had 3 spare. A leg only slightly longer would plausibly need a third
correction and exhaust a budget this project has twice tried and failed to raise
safely (2026-08-17: two review rounds found the guarding divergence detector
wrong for two different reasons). **Do not respond to a future exhaustion by
raising that budget.** The 6.10 m cap sits in about the right place.

Per-pulse travel: median **0.3374 m**, max 0.3866, min 0.0528 (a final-approach
pulse, shortened by design once remaining is inside
`final_approach_metres_per_pulse` = 1.06 — NOT a BLE stall).

### Both corrections were spent inside the last metre

| | fired after | at travelled | remaining | travel AFTER it |
| --- | --- | --- | --- | --- |
| re-aim 1 | pulse 15 | 5.0046 m | 0.9954 m | **0.8380 m** |
| re-aim 2 | pulse 16 | 5.3307 m | 0.6693 m | **0.5119 m** |

🔑 **They fired late because the trigger is a projected miss, not an angle.**
Pulse 11 carried −6.7° with 2.29 m still to run and did NOT fire; pulse 15
carried −12.8° with 1.00 m left and did, at a recorded aim error of 17.016°
(projected miss ~0.29 m against a 0.15 m tolerance). The error had grown steadily
since pulse 5; the controller waited until the geometry crossed the threshold.
That is the beta57 trigger change working as designed — the older angle-based
trigger is exactly the one that never fired in the far field.

🔑 **What actually made a −27.5° terminal error survivable was the final-approach
shortening, not the corrections.** The last two pulses travelled 0.1129 m and
0.0528 m, so even 27.5° of heading error buys only ~0.024 m of cross-track on the
final pulse. **This sharpens the case for the cap:** a longer leg would need a
third correction *before* final-approach shortening begins, in the regime where
pulses are still ~0.33 m and a 27° error costs ~0.15 m each.

Total forward-pulse travel was 5.8426 m against a 6.0 m plan; the balance is the
opening turn, the calibration pulse, and the corrections' own translation, none
of which appear in the forward-pulse ledger.

### The two re-aims are visually confirmed

`docs/evidence-reach-6m-reaim-frames-20260826.png` is an 8-frame strip
(GIF frames 460–544, 5 fps) cropped to the mower. The body is near-square to the
frame at the start of the window and progressively rotates through it — the
operator independently observed the mower "going straight then turning" late in
the run, before reviewing any telemetry.

This corroborates the *count and lateness* of the corrections: exactly two turn
events during the drive (plus the opening turn, `turn_commands_sent: 3`), both in
the final quarter. It also rules out the alternative reading that −27.48° was a
telemetry artifact — the machine is physically angled, on camera, at rest, at the
end of the run.

⚠️ **The frame-to-pulse alignment is INFERRED, not synchronised.** Frames were
mapped to pulses by assuming uniform cadence (19 pulses over ~121.8 s ≈ 6.4 s
each), which places pulses 15–16 near frames 480–515. Neither the GIF nor the
evidence file carries an absolute timestamp, so no true synchronisation exists.
**The pulse-indexed numbers above come from telemetry alone; the video supports
the count and the timing-within-the-run, not the per-frame mapping.**

⚠️ **The video does NOT demonstrate crabbing.** Crabbing requires body axis and
travel direction observed simultaneously; the camera is handheld, so apparent
in-frame motion is mower motion plus camera motion, and by the final frames the
mower is already stopped on `target_reached`. Source clip (not committed, 57 MB):
`G6 Turret 8-26-2026 - optimized.gif`, 609 frames at 5 fps.

🔑 **The clip length is an independent cadence check worth keeping.** 121.8 s for
a 6.0 m run is **0.049 m/s** effective, alongside the 0.0565 m/s the 9 m Route B
chain managed end to end — corroborating the ~22% stop-measure-go duty cycle from
a source with no shared failure mode with the position feed. A gross under-report
like the 2026-07-18 incident (telemetry 15 cm against ~82 cm observed) would have
been obvious here, and there is no sign of one. ⚠️ Clip boundaries are not known
to align exactly with dispatch and stop, so treat 121.8 s as approximate.

## The guarded turn works, and its quantum is not predictable

Getting from the 5 m end pose to the 6 m start needed a ~174° about-face. Rather
than autonomous return-to-dock (vendor-planned motion, outside every guard this
project has built), it was done with `raw_pymammotion_turn_to_heading` in two
bounded steps. Evidence: `docs/evidence-guarded-turn1-beta78-20260826.json`,
`docs/evidence-guarded-turn2-beta78-20260826.json`.

| | turn 1 | turn 2 |
| --- | --- | --- |
| Stop reason | `max_commands_reached` | **`target_heading_reached`** |
| Commands | 3 | 2 |
| Rotated | +65.35° | +113.97° |
| **Degrees per pulse** | **21.8** | **57.0** |
| Final error | 21.65° (outside tol 18) | 5.27° |
| Translation | 0.1130 m | 0.0734 m |

Both at **identical parameters**: angular 500, `pulse_duration_ms` 1500,
`motion_refresh_interval_ms` 200, tolerance 18.

⚠️ **SCOPE: these are STATIONARY in-place pivots (linear 0), and the number does
NOT transfer to steering while moving.** The arc regime behaves differently —
"angular needs 500" is explicitly a stationary-only finding, angular 180 actuated
fine in an arc, and the 2026-08-12 arc measurement was clean and linear
(+22.20° of course over 0.5823 m against +0.00° for the zero-angular control).
Quoting 2.6x as a bound on arc or continuous-steering response is a category
error. See `docs/phase2-steering-refusal-recommendation-20260826.md` §4.

🚨 **The pulse quantum varied 2.6x between two back-to-back turns with identical
parameters, and neither matched the model.** The documented night-branch fit
`rotation ≈ 32.2 °/s·t − 2.4` predicts ~45.9° at t=1.5 s; the two measurements
were 21.8 and 57.0. **Do not tune any turn constant against 21.8, 57.0, or
45.9.** This is the rotation-rate variance already on record, and the registered
explanation fits: a pulse rotates only while refresh writes are arriving, so a
blocked write lets the watchdog stop the motor while the executor still divides
by the whole commanded window.

🔑 **What made turn 2 succeed was budget, not prediction.** Turn 1's measured
21.8°/pulse was used to size turn 2 at `max_commands: 5` for a 108.7° target; it
converged in 2 pulses because the quantum happened to be 57.0 that time. The
right lesson is to give a turn enough command budget to absorb a 2.6x spread and
let it close on measured heading — not to predict the pulse count.

⚠️ **Two parameter traps for this service.** `angular_speed_fast` defaults to
**180**, which sits in the stationary deadband (measured 2026-07-25) and will
barely rotate a stopped mower; use 500. And `motion_refresh_interval_ms`
defaults to **0**, which is the one-shot-then-sleep h-watchdog bug; use 200. Set
`angular_speed_slow` to 500 as well, or the slow tier stalls in the deadband on
final approach.

🔑 **`target_heading_degrees` on this service is in the `toward` (compass) frame,
NOT map frame.** `_raw_turn_to_heading_status` compares it directly against
`position.toward` with no conversion — so there is no additive/mirror bug here,
but the caller must convert: `toward = 90.13 − map_heading`.

## Method — unchanged from the 5 m run, and it matters

Both traps from `docs/reach-5m-20260826.md` applied again and were avoided: the
CARD cannot run a long-leg test (it auto-splits above 3.85 m), and the service
schema defaults are NOT the accepted profile. The dispatched payload was
verified key-by-key against `docs/accepted-profile.json` — all 19 keys identical
— so this result is directly comparable to Gate 5 and to the 4 and 5 m runs. No
`LUBA_ACCEPTANCE_PROFILE` key moved and no Gate 5 is owed.

Heading was derived two independent ways before dispatch and agreed to **0.12°**
(mirror 91.93° vs VIO 91.81°), against 0.91° on the 5 m run.

## What this does not establish

⚠️ **n = 1 at 6 m.** Feasibility, not reliability — 3.0 m sat at 5 reached / 1
failed before it accumulated runs. Do not quote 0.114 m as an expected landing.

⚠️ **Nothing about a 6 m leg AFTER a junction turn.** Every point on the reach
curve began aligned by construction. Reach and post-turn landing accuracy are
different properties and only reach is measured here.

⚠️ **Video corroborates gross behaviour only.** An operator camera frame shows
the mower at the far end of the yard and the keep-out object untouched a few
metres off the driven line, consistent with `keep_out_leg_violations: []`. At
that distance a 10 cm landing error is sub-pixel; the telemetry remains the
measurement.
