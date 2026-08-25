# Phase 2 heading-safety remediation — 2026-08-24

## Status and scope

Implemented offline only. This work does not authorize deployment, Home
Assistant or BLE calls, arming the experimental-motion gate, or physical
motion. The host remains on its previously documented build and the motion
gate remains disarmed.

The corrected steering sign from `4483dd70` is retained. Independent
re-derivation from the raw position changes and both command signs still says
that positive commanded angular decreases map-frame course, so the final
command must carry the negative of the signed course error. This remediation
does not change desired-course, along-track, or cross-track sign conventions.

## The four resolved safety-model problems

1. **A stationary `toward` value was treated as an opening heading.** It is now
   diagnostic only. The controller has an explicit
   `acquiring_heading -> steering -> stopping` lifecycle, holds angular speed
   at zero during acquisition, and accepts only a fresh position chord of at
   least 0.15 m as `HeadingEvidence`.
2. **Blind acquisition had no complete containment envelope.** Before any real
   dispatch, the live start must have at least 1.06 m of polygon-boundary
   clearance: 0.28 m/s times the two-second acquisition timeout, plus the
   existing 0.50 m stop/guard overshoot. There is no dry-run or operator
   confirmation bypass. The 2026-08-24 corridor has only about 0.30 m of live
   clearance and is therefore refused.
3. **Opening alignment was estimated from stale, unsigned geometry.** The
   post-acquisition admission estimate now consumes the live position,
   position-chord course, current lookahead, actual elapsed time, cumulative
   path length, and signed opening/current/turn cross-track values. It checks
   intermediate arc extrema and admits only if every value stays within
   `min(0.20 m, configured hard bound)`. The 0.30 m runtime hard abort remains
   independent. Passing is a conservative admission estimate, not proof the
   controller will null the error.
4. **Independent clocks and origin displacement understated consumed safety
   budget.** A fresh valid origin is required after report streaming starts
   and before dispatch. The one safety clock starts immediately before the
   first movement command, so dispatch latency consumes the four-second
   window. Distance is the sum of every consecutive fresh position segment,
   not origin-to-current displacement. The rolling chord must refresh within
   two seconds; an acquisition timeout, stale heading, refresh stall, corridor
   breach, or exhausted time/distance budget requests a zero-speed stop.

## Public contract and diagnostics

The service payload shape is unchanged. Experimental v1 accepts only the
measured configuration, `linear_speed=400` and
`max_abs_angular_speed=180`. Dry runs report the required acquisition radius,
live boundary clearance, pending heading source/chord state, complete remaining
budgets, and blockers.

`HeadingEvidence` reports its position-chord source, map course, chord length,
measurement time, age, and position-noise-derived angular uncertainty. The
alignment estimate retains the registered conservative 8 deg/s turn-rate
model only for the measured +/-180 command envelope. The project has already
refuted proportional scaling outside the measured 120–180 range; this code
does not extrapolate the admission model beyond v1's fixed command.

## Offline evidence result

- The 2026-08-24 evidence replay produces no correction from the stationary
  opening value, remains in acquisition until a qualifying position chord,
  and the original corridor independently fails the 1.06 m containment gate.
- The Phase 1b and arc120 captures replay with heading derived from rolling raw
  position chords; `toward` is not used by the replay controller.
- Pure and service tests cover chord threshold/polarity, missing and stale
  evidence, signed cross-track admission, inward/outward excursions, invalid
  inputs, fresh-origin refusal, cumulative distance, acquisition transition,
  alignment refusal, stale heading, refresh stalls, and corridor aborts.

Offline completion is deliberately not a motion milestone. Any future deploy
or physical test requires a separate operator decision, a motion-disabled
deployment, a fresh corridor with at least the required live-start clearance,
and the repository's existing per-run authorization process.
