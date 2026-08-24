# The Phase 2 executor is implemented — offline only, no physical run

**2026-08-23. New service `continuous_motion_window`.** Implements the design
decisions and gap fixes recorded this week: straight-line segments only,
extends the bounded-window pattern, corrects on measured heading every ~1 Hz
arrival, stops safely on a detected BLE stall. It is the first dispatch-capable
Phase 2 code -- everything before it (`continuous_controller.py`) is a pure
calculator that never sends anything.

**This is deliberately NOT the same step as a physical run.** `dry_run`
defaults `True` and every test in this file exercises only that path. Nothing
here is deployed, dry-run-verified on the host, or authorized to move the
mower. Moves no `LUBA_ACCEPTANCE_PROFILE` key.

## What it is

One new service, `mammotion.continuous_motion_window`
(`custom_components/mammotion/services.py`), built from three pieces:

- **`_continuous_decision_loop`** — polls the coordinator cache (no BLE I/O,
  the same discipline `_capture_in_window_telemetry` already uses), detects a
  fresh position arrival, builds a `ContinuousObservation`, and calls
  `continuous_control_decision`. It writes the result into a shared
  `command_state` dict for the refresh loop to read, and sets an
  `asyncio.Event` when the decision is `stop`.
- **`_continuous_refresh_window`** — a new function, not a reuse of
  `_motion_refresh_window`, whose contract the plan explicitly says not to
  retrofit a variable command into. Resends whatever `command_state` currently
  holds at the app's refresh cadence, and feeds every completion timestamp back
  to the decision loop for gap tracking.
- **`_continuous_motion_gates`** — extends `_manual_velocity_pulse_gates` (all
  11 existing pulse gates, unchanged) with three new ones: the corridor
  polygon has at least 3 vertices, the frozen route start is itself inside
  that polygon, and live position has not drifted from the frozen start by
  more than 0.30 m.

Python's single-threaded event loop is what makes the `command_state` handoff
safe without a lock -- the same reasoning `_capture_in_window_telemetry`
already relies on for `travel_abort`.

## The two gap fixes, wired all the way through, not just documented

**Corridor breach override (gap 4).** `continuous_control_decision` trusts
`route.contained` unconditionally and does no live geometry check. The
decision loop independently tests every fresh position against the real
frozen polygon (`_point_in_polygon`, already used throughout this project's
keep-out checking) and forces a stop with reason `corridor_breach` if the pure
controller's own decision did not already say so.
`test_a_corridor_breach_forces_a_stop_the_pure_controller_did_not_request`
constructs a position that is within `max_cross_track_m` (so the pure
controller alone would say "drive") but outside the polygon, and confirms the
override fires.

**BLE-stall detector (gap 2).** The decision loop computes
`refresh_max_gap_since_last_decision_s` from the REAL list of refresh
completion timestamps the refresh loop is filling in concurrently -- not an
injected test value. `test_a_stalled_refresh_gap_stops_the_window` feeds a
completions list with a genuine 810 ms gap (the exact size of the real stall
that produced the corpus's largest prediction error) and confirms
`refresh_cadence_stalled` fires.

## What was NOT built, deliberately

- **No turns.** `ContinuousRoute` is a single start/target pair; nothing in
  this service can express a path with a junction.
- **No corridor scanning.** `corridor_polygon` is a required argument the
  caller supplies pre-scanned and margin-verified offline
  (`scripts/freeze_phase1_corridors.py`, `scripts/scan_contained_bearings.py`).
  This service never derives one.
- **No deployment.** Not built into a beta, not installed on the host, not
  dry-run-verified against real coordinator state. The next release still
  needs its own version bump, hash verification, and browser check, per this
  project's standing release discipline.
- **No physical run authorization.** Even once deployed and dry-run-verified,
  a real window needs the same per-run authorization every motion command in
  this project has required: fresh corridor scan, daylight, operator present,
  accessible e-stop, explicit consent for that one window.

## Verification

848 pytest (up from 827), ruff, ruff format, mypy 30 files, 91 frontend, ten
pre-commit hooks, `check_accepted_profile` ACCEPTED, 1295 doc symbols resolve.
20 new tests in `test_continuous_motion_window.py`: schema bounds and strict
point validation, `services.yaml`/translation field parity (a class of bug
this project has hit before), the heading-mirror convention, every dry-run
gate including three new ones, the corridor-breach override, the stall
detector fed a real gap, command-state mutation from a genuine heading error,
and the refresh loop's abort/no-abort behaviour in isolation from BLE.

One project-wide guard caught a real omission before it could ship: a test
that statically discovers every registered service and confirms each
`call.data[...]` read resolves against a minimal valid payload flagged that
`route_start`, `route_target`, and `corridor_polygon` had no sample value yet.
Fixed in `tests/components/mammotion/test_map_task_visibility.py`, not
worked around.

## What is actually next

Per `docs/phase2-gate-readiness-20260823.md`, the pass criteria for a real
run already exist (`docs/continuous-motion-feasibility-plan-20260821.md`).
Nothing new needs designing. What remains, in order:

1. Cut a release and deploy motion-disabled, same as every prior change this
   week -- version quartet, hash verification, host dry-run proof.
2. Dry-run this specific service against LIVE coordinator state on the host,
   the same tail every deploy this week has required, before trusting it
   against real telemetry shapes rather than the fakes in this test file.
3. Only then propose a physical run, scanned fresh, on the day, with its own
   authorization.

None of that is done by this commit.
