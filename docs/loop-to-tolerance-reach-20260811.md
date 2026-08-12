# Loop-to-tolerance lifts per-segment reach from ~1 m to 4 m

**2026-08-11, four armed daylight runs on `0.6.4-beta41`.** All authorized
per-run, all disarmed with the gate state verified afterwards. Blades off
throughout. Evidence:

- `docs/evidence-beta32-4segment-20260811T235133Z.json` — 2 × 2.0 m legs, one
  60° junction
- `docs/evidence-beta32-4segment-20260811T235945Z.json` — 1 × 3.0 m leg
- `docs/evidence-beta32-4segment-20260812T001116Z.json` — 1 × 4.0 m leg, **cut
  short by a BLE command failure** at 0.5493 m remaining
- `docs/evidence-beta32-4segment-20260812T002804Z.json` — 1 × 4.0 m leg, the
  same run repeated once BLE recovered

⚠️ **Neither run is on the accepted profile.** Both pass
`max_linear_pulse_ceiling` (10 and 12), which is a frozen
`LUBA_ACCEPTANCE_PROFILE` key the card sends as `null`. That is deliberate: the
point was to *measure* whether the reach is there before anyone pays a Gate 5
for it. **These landings do not compare to Gate 5.** Every other profile key was
sent at its accepted value.

## 1. The result

| run | leg | pulses | landing | stop |
| --- | --- | --- | --- | --- |
| `…235133Z` seg1 | 2.000 m | 5 of 10 | **0.0690 m** | `target_reached` |
| `…235133Z` seg2 | 1.942 m | 5 of 10 | 0.1797 m | `target_requires_reverse_recovery` |
| `…235945Z` seg1 | 3.000 m | 8 of 12 | **0.0928 m** | `target_reached` |
| `…001116Z` seg1 | 4.000 m | 9 of 16 | 0.5493 m | `vio_realign_incomplete` (BLE) |
| `…002804Z` seg1 | 4.000 m | 11 of 16 | **0.1023 m** | `target_reached` |

All three successful segments converged monotonically and **terminated on
tolerance, not on the ceiling** — 5 of 10, 8 of 12, 11 of 16. The ceiling was
never the binding constraint, which is what "the loop works" has to mean.

```
2.0 m leg:  2.0000 -> 1.5579 -> 1.1348 -> 0.7489 -> 0.3603 -> 0.0690
3.0 m leg:  3.0000 -> 2.4309 -> 2.0223 -> 1.7919 -> 1.4151 -> 0.9787
                   -> 0.7904 -> 0.4643 -> 0.0928
4.0 m leg:  4.0000 -> 3.6130 -> 3.2511 -> 2.9543 -> 2.5226 -> 2.0507 -> 1.6368
                   -> 1.3230 -> 0.9162 -> 0.5895 -> 0.2605 -> 0.1039
```

**The counterfactual is not a guess** — it is the third row of each segment's own
trace, since the accepted profile differs only in stopping after pulse 3:

| segment | leg | remaining after 3 pulses | reached with the loop |
| --- | --- | --- | --- |
| `…235133Z` seg1 | 2.000 m | **0.7489 m** | 0.0690 m in 5 |
| `…235133Z` seg2 | 1.942 m | **0.6777 m** | 0.1797 m in 5 |
| `…235945Z` seg1 | 3.000 m | **1.7919 m** | 0.0928 m in 8 |
| `…002804Z` seg1 | 4.000 m | **2.9543 m** | 0.1023 m in 11 |

On the accepted profile all four stop on `max_linear_commands_reached` between
0.68 m and 2.95 m short. This confirms the standing measurement that per-segment
reach is ~1 m, and shows the loop is what removes it.

**Per-click reach therefore goes from ~4 m to ~16 m** at the current
`REAL_CLICK_TO_GO_SEGMENT_LIMIT` of 4. That is the difference between a
repositioning nudge and clicking a point across the yard.

⚠️ **4 m is a demonstrated floor, not a measured limit.** The ceiling has never
bound; every run ended because the mower arrived. Where reach actually breaks is
still unknown, and the 4 m run needed 11 pulses over ~108 s, so the practical
limit may turn out to be patience or link stability rather than the control law.

## 2. 🔑 The loop is robust to BLE stalls, and that is the bigger finding

The 3 m leg ran through three degraded windows and still landed at 9.3 cm.

| pulse | delivered window | nonzero writes | travelled |
| --- | --- | --- | --- |
| 1 | 1630 ms | 7 | 0.4686 m |
| 2 | 1433 ms | 5 | 0.4120 m |
| **3** | **4158 ms** | **2** | **0.2325 m** |
| 4 | 1375 ms | 5 | 0.3817 m |
| 5 | 1331 ms | 5 | 0.4509 m |
| **6** | **2847 ms** | **2** | **0.2016 m** |
| 7 | 1664 ms | 4 | 0.3380 m |
| 8 | 2422 ms | 4 | 0.3763 m |

This is the h-watchdog signature from `docs/HANDOVER-beta31-20260809.md` §2.7,
measured on **linear** travel rather than rotation. Pooling both runs' 18 linear
pulses, travel is flat in write count above 3 writes and collapses at 2:

```
>= 4 writes   n=15   mean 0.4086 m   range 0.338 - 0.495
<= 2 writes   n= 2   mean 0.2170 m   range 0.202 - 0.233
```

⚠️ **n = 2 for the stalled case.** Treat "a stall costs about half a pulse" as
the shape of the effect, not a calibrated number. The relationship is plainly
not proportional — 4, 5, 6 and 7 writes all deliver ~0.34–0.49 m with no trend —
so this is a floor effect, not a rate.

**Why it matters:** against `max_linear_commands: 3` those two stalls cost
~0.4 m of a ~1.05 m budget, which is fatal for anything but a short leg. Under
the loop they cost two extra pulses and about 6 s. **Loop-to-tolerance turns a
BLE stall from a mission failure into a slower run.** BLE write latency stays
worth fixing — it still degrades the rate estimate and the ceiling's guarantee —
but it stops being a blocker for reach.

## 3. Why segment 2 failed, and it was not reach

Segment 2 converged 1.9417 → 0.1797 m and then stopped with
`target_requires_reverse_recovery`. It did not run out of pulses and it did not
drive past the waypoint along the leg. It ended **0.1797 m from the target while
facing 119° away from it**, because it had drifted right of the line — at which
point the beta22 containment correctly refuses to dispatch a U-turn.

The cause is one record. At pulse 4, 0.3246 m out with a −26.914° aim error, the
beta38 re-aim guard suppressed a correction:

```
projected perpendicular miss  0.1469 m
waypoint_tolerance            0.150  m
margin                        0.0031 m       <- 3.1 mm
```

The next pulse then travelled 0.3771 m and it landed 0.1797 m out — the guard
under-predicted by **32.8 mm**.

### The mechanism, and it is not a fudge factor

The guard asks whether driving straight on still lands inside the disc, and
answers with `distance · sin(aim)` — the miss at the point of **closest
approach**. But the mower does not stop at the closest approach. It drives a
whole pulse. Here the closest approach sat 0.2894 m along-track and the pulse
carried it 0.3771 m, so it finished **0.0877 m past** that point:

```
closest-approach miss                 0.1469 m
along-track overshoot past it         0.0877 m
predicted landing (in quadrature)     0.1711 m       actual 0.1797 m
```

That accounts for three quarters of the error and, decisively, **0.1711 m
exceeds the 0.150 m tolerance, so the correction would have fired.**

Replayed across every suppressed re-aim this project has recorded:

```
                             mean |error|
guard, as shipped               0.0212 m
+ next-pulse overshoot term     0.0147 m       n = 13
guard under-predicts on 11 of 13 suppressions
```

### What this does to the 2026-08-10 decision

The projection-margin idea was dropped on 2026-08-10 with the reasoning that the
optimism "came from the uncorrected post-turn error, which beta40 fixes at
source". **That reasoning is not refuted, but it is incomplete.** beta40 fired
correctly on this very segment (18.139° → −5.363°, `correction_attempted: true`,
`passed: true`) and the guard still under-predicted by 3.3 cm. On the 0.7 m legs
the drop decision was written about, the beta40/41 suppressions do land within
about ±1 cm, exactly as it predicted. The gap opens at long legs, where a full
pulse is a larger fraction of the remaining distance.

So the open item is **not** the fitted margin that was rightly rejected. It is a
one-term correction to what the guard projects: **the miss at the end of the
next pulse, not at the closest approach.** It touches no
`LUBA_ACCEPTANCE_PROFILE` key. It is **not implemented** — it changes the motion
control law and deserves its own review rather than being written at dusk with
the mower on the lawn.

## 3b. ⚠️ `last_travel_heading()` is wrong whenever a leg is not straight

The facing guard fired before the first 4 m attempt at **23.8°** with nobody
having touched the mower — and the mirror was still the correct answer.

`last_travel_heading()` returns the straight-line bearing from a segment's start
to its target, 340.00°. But that leg had a mid-drive realignment at pulse 6 that
left the mower facing ~316.04°, and it drove its last two pulses on that
heading. The mirror said **316.23°** — agreeing with the realignment record to
**0.19°**.

So the estimate degrades with exactly what long legs produce: mid-drive
corrections. The fix is to take the bearing of the **last pulse**, from the final
two sample positions, rather than of the whole leg. Script-only, not implemented.

Note this is a *false alarm*, not a false negative — the guard flagged a real
disagreement and the operator-facing advice ("trust the mirror") was correct.
It costs a `--heading` round trip, which is the cheap direction to be wrong in.

## 4. The facing cross-check earned itself on its first live use

The harness now derives the mower's facing twice — the mirror of the live
`toward`, and the bearing of the last leg this project drove — and refuses to
build a path when they disagree past 15°. The operator repositioned from the app
before the first run and it fired immediately:

```
mirror of live `toward`   :  276.58 deg   (toward=173.5469)
last leg we actually drove:  182.31 deg
REFUSING TO BUILD A PATH: the two facing estimates disagree by 94.3 deg
```

That is the exact condition that built a backwards path twice on 2026-08-10 and
cost two daylight runs. It cost one re-run with `--heading 276.58` here, and the
resulting opening turn was **2°**. On the second run, with no app move in
between, the two agreed to **6.03°** and no override was needed.

## 4b. The one failure that was neither reach nor the guard

The first 4 m attempt (`…001116Z`) stopped at 0.5493 m remaining with
`vio_realign_incomplete`, after 9 of 16 pulses. The cause is one line:

```
RuntimeError: BLE motion command did not start before the queue deadline;
              the queued item was disarmed
```

The loop was still converging and had 7 pulses in hand. It asked for a −26.005°
re-aim and the link dropped the command. **It was not light** — the calibration
drive that run recorded `vio_state: 2`, `tracked_features: 80`,
`brightness_label: "Light"`. Repeating the identical run ~17 minutes later, once
BLE recovered, reached target at 0.1023 m.

⚠️ A tempting story — that `ble_rssi` was tracking the mower's distance from the
house, −48 docked through −72 at 15.7 m out, with the failure at the furthest
point — **does not survive the next data point.** After the operator drove the
mower back to the middle of the yard the link read −68 at 8 m, worse than the
−64 it had shown at that same distance an hour earlier, and the BLE client then
dropped entirely while the mower sat still. Elapsed session time explains the
record at least as well as distance does. Treat the distance hypothesis as
unsupported; the standing BLE-session item
(`docs/pymammotion-ble-slot-leak-bug.md`) fits better.

## 4c. Two harness defects the failures exposed

Both found by the harness misbehaving live, not by reading it. Fixed in
`c196b8b1`; script and tests only.

1. 🚨 **An aborted arm left the gate OPEN.** `armed` was set *after* the
   post-enable readback, so when the enable succeeded but `real_motion_allowed`
   came back false — exactly what `ble_client_not_connected` produced — the
   script returned early announcing it had aborted "without sending anything",
   and the `finally` declined to disarm something it had already done. Caught
   and disarmed by hand within the minute. `armed` now means "may have touched
   the gate, owes a disarm" and is set *before* the enable.
2. **The preflight never asked the gate what blocks it.** All eight
   entity-derived checks passed — `ble_link_live` still read `on` — while the
   backend's `blockers` already carried `ble_client_not_connected` and
   `ble_rssi` read 0. That list was printed as informational text. It is now a
   hard check.

## 5. What is NOT established

- **Reach beyond 4 m.** The ceiling was never reached, so 4 m is a demonstrated
  floor, not a measured limit.
- **Reach on the accepted profile.** Adopting `max_linear_pulse_ceiling` changes
  a frozen key, un-accepts the profile and owes a fresh Gate 5. That re-pass is
  now the next genuine milestone.
- **Multi-segment reach.** The 2 m run's second segment failed on cross-track,
  so a long-leg path of more than one segment has not yet completed end to end.
  Fixing the guard projection is the prerequisite, not more reach.
- **The stall magnitude**, n = 2 as noted above.
