# Bounded motion test — slow-tier landing accuracy, 2026-08-07 night

**Status: awaiting operator authorization. No motion is authorized by this
document being written.**

One question, one falsifiable prediction, minimum travel.

## 1. The question

The tolerance analysis predicts that a waypoint approach whose final phase runs
at the slow tier lands within **~0.15 m** of target, built from two measured
quantities:

| error source | at slow tier (~0.10 m/s) |
| --- | --- |
| position-feed staleness, 1031 ms median | ~10 cm |
| stop latency, median 229 ms (n=60) | 2.3 cm |
| stop latency, p90 461 ms | 4.6 cm |

**Nothing has ever tested this.** The slow tier is inferred from speed
measurements; no segment has been run to a target and had its true landing error
measured against 0.15 m. If the prediction is wrong, the whole error budget is
wrong and the tolerance recommendation must be withdrawn.

This is also the same geometry class that **failed at 0.08 m**: Gate 4 passed
only by overshooting and U-turning back — 2.28 m of travel for a 1.04 m path.
The hypothesis is that 0.15 m stops cleanly where 0.08 m could not.

## 2. Why this is night-safe

Pure linear. **No turn is commanded at any point.**

RTK holds in the dark and is currently `Fix`. VIO is dead (`signal_none`), which
is why turning is excluded rather than merely discouraged — and `toward` is
course-over-ground, frozen while stationary, so current orientation is not
trustworthy from telemetry either.

**Heading is therefore established by moving, never by reading it.** One short
forward pulse gives a displacement vector = true heading. Each subsequent
segment's own displacement supplies the heading for the next, so only one
dedicated heading pulse is needed. A 1.0 m segment yields a heading good to
~1.1° given sub-cm stationary noise under Fix.

## 3. Procedure

**Operator preconditions:** mower undocked, blades off, in an open area with
**≥5 m of clear straight-line space** ahead of its current facing, awake with BLE
connected.

**Verified by me before arming** (abort if any fails): `position_valid_for_motion`
on, RTK `Fix` and not degraded, no active session, BLE link live, blades 0,
`backend_verified` true.

1. **Baseline.** Preflight readback; record position, RTK, satellites, base
   station state.
2. **Heading pulse.** One `manual_velocity_pulse_test`, `action: forward`,
   `speed: 0.35` (stick fraction), `duration_ms: 3000` → ~0.6 m. Record position
   before and after; displacement = true heading.
3. **Segment N (×3).** Compute a target **1.0 m** further along the last measured
   displacement. Call `raw_pymammotion_execute_vector_segment` **dry_run first**,
   confirm the plan, then real. Settle, record true final position.
4. **Measure.** `error = |settled_position − target|`, plus pulses used, total
   travel vs path length (the Gate 4 overshoot ratio was 2.19×), and whether
   containment fired.

Each segment re-derives heading from the previous segment's displacement.

## 4. Parameters

Accepted Gate 4 profile, with **one deliberate change** — `waypoint_tolerance`
0.08 → **0.15**, which is the whole point of the test:

```
waypoint_tolerance:        0.15      # the value under test
slow_linear_threshold:     0.15      # default; slow tier for the final approach
linear_speed_fast:         400       # default
linear_speed_slow:         200       # default
max_linear_commands:       3
max_turn_commands:         1         # minimum the schema allows; see below
heading_tolerance_degrees: 18
linear_pulse_duration_ms:  1300
motion_refresh_interval_ms: 200
confirm_blades_off:        true
```

`max_turn_commands` cannot be set to 0. Turning is prevented instead by
**construction**: the target is placed along the measured heading, so aim error
is far inside the 18° tolerance and no turn is ever dispatched. If a turn is
dispatched anyway, that is an abort condition (§5) — it means the heading
estimate is wrong.

## 5. Abort conditions

Stop immediately, disarm, and report on any of:

- any motion blocker appears, or RTK leaves `Fix`
- **any turn command is dispatched** (heading estimate wrong)
- landing error > 0.5 m on any segment
- total travel exceeds 1.6× path length on any segment
- BLE link drops or a stop fails to confirm
- operator says stop

`target_requires_reverse_recovery` is **not** a failure — it is beta22
containment working, and is a valid, recordable outcome that ends the run.

## 6. Bounds

- **Blades off throughout**, confirmed in every call.
- **Max 3 test segments**, ~3.6 m total travel, all in one straight line.
- Gate armed immediately before the run and **disarmed immediately after**,
  verified both times.
- Every call recorded; results written to
  `docs/evidence-slow-tier-validation-20260807.json`.
- No Gate 4 retry and no Gate 5 attempt. This is not a gate run.

## 7. What each outcome means

| result | conclusion |
| --- | --- |
| all 3 land ≤0.15 m, no recovery | prediction confirmed; 0.15 m is shippable and the error budget is sound |
| lands 0.15–0.30 m | budget is optimistic; the honest tolerance is nearer 0.30 m |
| overshoot + `target_requires_reverse_recovery` | 0.15 m is still too tight for this control law; containment verified working |
| wide scatter across the 3 | a single tolerance number is the wrong model; pass criterion should be N-consecutive-runs |

A null result is a real result. The point is to find out, not to pass.
