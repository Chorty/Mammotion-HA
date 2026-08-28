# Phase 2 steering attempt 3 — design, written before the run

Supersedes the run-1 configuration in
`docs/phase2-steering-run1-predeclared-20260827.md`. **The seven pass criteria in
that document are unchanged and still govern.** Only the run *setup* changes, and
the reasons are below so this is not a threshold moved to fit data.

## Attempts 1 and 2 both refused before steering. Neither tested the sign.

| | attempt 1 | attempt 2 |
| --- | --- | --- |
| Build | beta80 | beta81 |
| Refused with | `position_sequence_gap` | `opening_alignment_infeasible` |
| Motion commanded | none | 0.2407 m, all at `angular_speed: 0` |
| Cause | two reliability defects (fixed) | window budget exhausted by the blind phase |

Evidence: `docs/evidence-phase2-steering-run1-refused-20260827.json`,
`docs/evidence-phase2-steering-run2-refused-20260827.json`.

🗑️ **CORRECTION.** I initially reported attempt 2 as having refused "because the
mower was too well aligned". **That was wrong.** The recorded
`alignment_feasibility` names `limiting_factor: "window_s"`:

```
window_budget_s      2.000
blind_time_s         1.950   <- acquiring the 0.15 m chord
remaining_window_s   0.0497
turn_time_s          0.0889  <- needed, at the 8 deg/s model
                     0.0889 > 0.0497  ->  infeasible
```

The 0.711° heading error was incidental. The run refused on **time**, and the
cause was my own run-1 configuration: I set `duration_ms: 2000` to bound exposure
without checking that the blind acquisition phase must complete **inside the same
window**. Acquisition alone consumed 1.95 s of the 2.0 s budget.

## Two changes, each with its reason

**1. `duration_ms` 2000 → 6000.** The window must cover acquisition *plus* enough
1 Hz decisions to show convergence. Attempt 2 measured the blind phase at
**1.95 s / 0.2407 m** (0.123 m/s effective — slower than the 0.2482 nominal
because of the standstill ramp and ~1 Hz sampling granularity). 6 s leaves ~4 s of
steering.

⚠️ **Exposure does not grow proportionally, because distance binds first.** At
0.2482 m/s, `max_distance_m: 1.00` is reached at ~4.0 s, before the 6 s window.
**`max_distance_m` stays 1.00 and remains the real bound.** The 0.30 m cross-track
hard abort is unchanged.

**2. A deliberate 8° route misalignment.** Attempts 1 and 2 aimed the route along
the mower's own heading, so even had the window sufficed, the correction would
have been ~0.7° — far too small to exercise the sign. The route target is now
offset **8°** from the measured heading.

Why exactly 8°:

* With `angular_speed_per_heading_degree: 12.0` and the capped
  `max_abs_angular_speed: 120`, the command **saturates at 10°**. At 8° the
  commanded angular is **96** — inside the cap, so the run demonstrates
  *proportional* control rather than a saturated rail. That is precisely what the
  2026-08-24 run never showed, having opened at 46.64° and saturated immediately.
* It clears the feasibility model: `turn_time = 8/8 = 1.00 s` against ~4 s
  remaining, `turn_distance = 0.248 m` against ~0.76 m remaining.
* Predicted cross-track stays small: **0.033 m** accrues during the blind phase,
  against a 0.20 m admission budget and the 0.30 m hard abort.

## Configuration

| Parameter | Value | Change |
| --- | --- | --- |
| `confirm_steering_validation_run` | true | — |
| `max_abs_angular_speed` | 120 | unchanged (lowest MEASURED arc value) |
| `linear_speed` | 400 | unchanged |
| `duration_ms` | **6000** | was 2000 |
| `max_distance_m` | 1.00 | unchanged — binds before the window |
| `max_cross_track_m` | 0.30 | **unchanged** |
| Route offset | **8° from measured heading** | was 0° |

## Pass criteria — unchanged from run 1

All seven from `docs/continuous-motion-feasibility-plan-20260821.md`. Repeated
here only so the file is self-contained; not restated as new.

1. No intermediate stop before the final/abort stop.
2. Signed heading error and absolute cross-track both trend toward zero.
3. No oscillation between saturated ±angular commands.
4. Cross-track never exceeds 0.20 m.
5. The 0.30 m hard abort never fires.
6. Motion duty cycle at least 80%.
7. Final stop confirmed and the motion gate disarmed afterwards.

🔑 **The decisive question is still: does heading error DECREASE after the first
steering command?** With an 8° opening error and a 96 angular command, the answer
is visible within one or two decisions. **If heading error grows across two
consecutive decisions, stop and re-derive the sign** — do not retune the gain and
do not call it noise.

## Required outputs

Per-decision `heading_error_degrees`, `cross_track_m`, `angular_speed`, `reason`;
the measured rotation response per commanded angular **while moving** (still
unmeasured anywhere); `refresh_max_gap_since_last_decision_s` against 0.60 s;
travel at stop; and gate state afterwards from the live API **and** RAW storage.

## Preconditions

⚠️ **The acquisition budget moved to 3.0 s after this document was written**
(`docs/phase2-acquisition-budget-decision-20260827.md`), so the blind disk is now
**1.34 m**, not 1.06 m, and the corridor must clear that. The 8° offset and 6 s
window below are unaffected.

Daylight is not required — the steering path uses RTK position chords and has no
VIO gate. Battery should be comfortable: attempts so far have run it from 57% to
roughly 45%. BLE has drifted between −42 and −72 today; the record is explicit
that RSSI does not predict cadence, and the real guard is the measured 0.60 s
refresh bound.
