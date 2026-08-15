# Beta54 card-driven Night Go — supervised characterization

**Date:** 2026-08-14
**Build exercised:** `0.6.4-beta54`
**Disposition:** safe stop, useful characterization, not an accuracy acceptance

The operator explicitly authorized one supervised Night Go from `(4.954,
-2.643)` to `(5.692, -2.602)`. The requested leg was 0.739138 m. The mower was
in an open mapped area with its blades off. The complete card result is
preserved unchanged as
`docs/evidence-night-go-card-beta54-20260814T180345Z.json` (copied from the card
download
`mammotion-real-go-lawn-mower-back-yard-clip-skywalker-2026-08-14T18-03-45-242Z.json`).

## Measured facts

- The result used `turn_mode: "night"`, RTK `Fix`, angular speed 500, 200 ms
  motion refresh, an 8° heading tolerance, three maximum linear commands, and
  a 0.08 m waypoint tolerance.
- The opening target heading was 86.9575°. Three turn commands
  (`-500`, `-500`, `+500`) ended at `toward` 90.3537°, 3.3962° from the target.
- Three forward commands were sent. Distances remaining after them were
  0.225639, 0.082661, and 0.117085 m.
- Pulse 2 therefore settled 0.002661 m outside the configured tolerance. Pulse
  3 moved 0.036801 m and increased the target distance by 0.034423 m.
- The executor stopped with `no_target_progress`. The final position was
  `(5.8041, -2.6358)`, with RTK `Fix`, `MODE_READY`, blades off, and all recorded
  stop operations successful.
- The beta54 card omitted `sample_delays`, so the backend defaulted to
  `[0, 5, 10, 20, 30, 45, 60]` after each command. That made the run take about
  6.5 minutes even though all three turn-heading updates were visible within
  three seconds.

## Inference and off-mower follow-up

The second pulse had already crossed the target. The night-only continuation
decision nevertheless reused the shared progress diagnostic's pre-pulse target
bearing (9.2578°), so it did not recognize that the settled target was behind
the mower. This is the identified cause of the unnecessary third pulse; it is
not evidence that the mower or RTK failed.

The PR #14 follow-up changes only the night branch to derive its
residual target bearing from the settled post-pulse RTK position. That lets the
existing night reverse-recovery refusal stop after an overshoot. The shared
progress diagnostic and the VIO and legacy paths are unchanged. The card and
harness also set `sample_delays: [0, 3]`, based on the observed heading-update
window, to avoid the minute-long diagnostic waits.

The follow-up tree now also contains a VIO-only Real Go throughput fix. It was
verified locally with **668 pytest tests** and **46 frontend tests**. Ruff check,
Ruff format check, mypy, and all pre-commit hooks also passed. These changes
have not yet been committed, released, deployed, or tested on the mower.

## What this run does not establish

This single landing does not establish a supported night tolerance or an
accuracy distribution. It also does not validate the follow-up controller
change on hardware. Any further movement requires a new, explicit, supervised
authorization.
