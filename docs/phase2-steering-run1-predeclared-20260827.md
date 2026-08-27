# Phase 2 steering run 1 — criteria predeclared BEFORE dispatch

**Written 2026-08-27, before any steering run existed.** Nothing in this file may
be edited after a capture exists. If a criterion turns out to be ill-posed, that
gets fixed deliberately in a follow-up document with the reasoning stated — never
by editing a threshold after seeing the data that failed it. That rule is why the
2026-08-22 mirror-criterion `no_go` was trustworthy.

## What is being tested

The **corrected steering sign** (defect 2 of 2026-08-24), which has never moved a
wheel. Defect 1 was hardware-validated on 2026-08-27
(`docs/evidence-phase2-acquisition-beta79-20260827.json`).

The one prior attempt (2026-08-24) diverged monotonically — heading error
46.64° → 48.25° → 77.40° while saturated at +180 — and hard-aborted on the 0.30 m
cross-track bound at 0.517 m travelled.

## Configuration for run 1

| Parameter | Value | Why |
| --- | --- | --- |
| `confirm_steering_validation_run` | **true** | Steering is opt-in per call; arming the gate is not sufficient |
| `max_abs_angular_speed` | **120** | Lowest **measured** arc value. ⚠️ NOT 60: turn rate is measured only across 120–180 and must not be scaled outside that band, so a smaller value is unmeasured, not safer |
| `linear_speed` | 400 | Frozen, measured |
| `duration_ms` | **2000** | Half the 4000 ms used on 2026-08-24; bounds exposure |
| `max_distance_m` | **1.00** | Below the 1.50 default |
| `max_cross_track_m` | **0.30** | UNCHANGED — it fired correctly and contained the failure |
| `motion_refresh_interval_ms` | 200 | Unchanged |

## Pass criteria — all seven must hold

From `docs/continuous-motion-feasibility-plan-20260821.md`, written before any
Phase 2 code existed. Unchanged here.

1. No intermediate stop before the final/abort stop.
2. Signed heading error and absolute cross-track both trend toward zero.
3. No oscillation between saturated ±angular commands.
4. Cross-track never exceeds 0.20 m.
5. The 0.30 m hard abort never fires.
6. Motion duty cycle at least 80%.
7. Final stop confirmed and the motion gate disarmed afterwards.

**Any failure is a FAIL.** A run that aborts safely is a successful *guard* and a
failed *criterion set*; both statements go in the record, as they did on
2026-08-24.

## The single decisive question

🔑 **Does heading error DECREASE after the first steering command?**

On 2026-08-24 it grew on every decision. If the sign fix is right, the very first
corrective command must move it the other way. That is visible in decision 2 and
does not require the full window to complete. **If heading error grows across two
consecutive decisions, stop the programme and re-derive the sign** — do not retune
the gain, and do not attribute it to noise.

## Required outputs, recorded whether it passes or fails

* Per-decision `heading_error_degrees`, `cross_track_m`, `angular_speed`, `reason`.
* 🔑 **Measured rotation response per commanded angular WHILE MOVING.** Nobody has
  measured this; the arc data is two points at one speed. It is a required
  result, not an assumption.
* `refresh_max_gap_since_last_decision_s` against its 0.60 s bound.
* Travel at stop, and cross-track at every decision.
* Gate state after the run, from the live API **and** RAW `core.config_entries`.

## Refusals that would abort before dispatch

The corridor must contain the full acquisition disk; the mower must be
`AREA_INSIDE` with RTK Fix; BLE must be live. These are existing gates and none
are relaxed for this run.

## What a PASS does and does not authorize

A pass authorizes **one more** steering run for repeatability. It does **not**
authorize Phase 3 waypoint A/B, longer windows, higher angular authority, or
removing the opt-in. Those need their own predeclared criteria.
