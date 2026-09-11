# PLAN — resolving what the 2026-09-10 session found (written 2026-09-10/11)

Companion to
`docs/findings-clicktopath-reliability-4m-repeat-20260910.md`. That document
records what happened; this one says what to do about it. Three issues
surfaced tonight, in order of how directly each blocks resuming the series.

---

## Issue 1 — the BLE command-queue timeout (§1.5 of the findings doc)

**What it is.** `_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS = 2.0` refuses a
motion pulse that can't *begin* processing in the serialized BLE command queue
within 2.0 s — independent of RSSI, which measures the radio link, not queue
occupancy. Legs 7 and 8 both failed here, each after several pulses had
already succeeded, at a position the coverage map and live RSSI both called
good (−60 to −67 dBm). This is what actually stopped the series, not the
control law and not BLE range.

**Why nothing was changed tonight.** The 2.0 s figure is a deliberately
calibrated safety constant — its own comment explains it protects against a
late-dispatched pulse drifting out of sync with its own timing model. Loosening
it without evidence risks trading a safe, visible refusal for an unsafe, silent
one. Per this project's own routing rule, anything touching the motion control
law needs its own scrutiny, not a same-session patch made while tired at 1 AM
after an aborted run.

**Plan, in order:**

1. **Instrument before touching the constant.** Currently a queue-start
   timeout tells you *that* something was ahead of the pulse, not *what*.
   Capture `_ble_link_liveness(coordinator)` (queue_depth,
   last_send_age_seconds, is_connected, cooldown state) as
   `command_result["queue_diagnostics"]` at the moment of failure, embedded
   directly in the result the caller already gets back. This is purely
   additive — no control-flow change, no new risk. **Implemented this
   session, but scoped to the two failure sites inside
   `_raw_pymammotion_execute_vector_segment` only** (the one service actually
   exercised tonight, in `custom_components/mammotion/services.py` around
   lines 17364 and 17406) — the identical pattern exists at ~17 other sites
   across other executors (`manual_velocity_pulse_test`,
   `execute_multi_segment`, etc.) that were not touched. Extending it there is
   the same small edit, deferred rather than done in bulk at 1 AM on the
   file that runs the motion control law. 1062 tests, ruff, and mypy all
   clean after this change.

   🚨 **Amended 2026-09-11: that scoping left the measurement blind where it
   most needed to see.** The two instrumented sites are the executor's LINEAR
   phase, but `_vio_segment_calibration_drive`, `_raw_pymammotion_turn_to_heading`
   and `_vio_turn_to_heading` are *phases of the same service*, run on every
   leg — and a leg aborting in any of them produced the identical reason with no
   queue snapshot at all. Since leg 4 died 0.29 m into a 4.0 m leg, the
   calibration drive is a live candidate for where it died, and the record
   cannot say which. Extended to all four in-scope functions on operator
   approval; the ~17 sites in genuinely other executors stay deferred. Seven
   capture sites now, mapped in
   `docs/findings-clicktopath-reliability-4m-repeat-20260910.md` §1.5.1.
2. **Reproduce and measure, off the back of that instrumentation.** Next real
   session, watch for whether the occupant is consistently the
   `motion_refresh_interval_ms: 200` traffic, a reconnect retry, or something
   else. Two data points (legs 7, 8) is not enough to diagnose from; it is
   enough to know where to look.
3. **Only then consider a number.** Candidates, not yet chosen: raise
   `_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS` itself (directly loosens the
   guard — needs the most scrutiny), or reduce `motion_refresh_interval_ms`'s
   contribution to queue occupancy (a profile value — needs its own
   predeclaration and Gate 5 per standing practice, since it is an accepted
   LUBA-acceptance parameter, not a free tuning knob). Neither happens without
   the measurement from step 2 first, and neither happens without an explicit
   operator decision — this plan authorizes step 1 only.

## Issue 2 — comms-loss recovery gap

Already fully written up as its own decision request:
`docs/design-comms-loss-recovery-20260910.md`. Short version: nothing today
reacts to a `stop_failed_aborting` / `command_failed` refusal — no
notification, no auto-verify, no auto-dock. That conflicts with standing
decision 2 for any unattended session. Four options laid out there (do
nothing / notify only / notify + auto-verify / auto-return-to-dock), with
notify-only recommended as the safe first step and auto-return-to-dock
explicitly held for its own separate decision.

✅ **RESOLVED 2026-09-11: the operator chose B, and it is built** — with a scope
correction, since B as written would have fired for leg 4 only and stayed silent
for legs 7 and 8. Record: `docs/findings-comms-abort-notify-20260911.md`.
C and D remain unbuilt and undecided; D still needs its own separate decision.

## Issue 3 — the Mammotion integration's own setup failure (§1.6 of the findings doc)

**What it is.** A bootstrap timeout cancelled the integration's own setup
mid-session, independent of the mower — a documented trap, not a new defect.
Recovered cleanly with a config-entry reload in ~20 s; nothing moved during
the outage.

**Plan: none needed beyond what already exists.** The recovery path (reload,
not restart) already worked correctly and is already documented in project
history. This is logged here only so the next session that sees
`Could not find entity lawn_mower...` knows to check config-entry state before
assuming a mower-side fault.

---

## What "resolve" means for each, stated plainly

| issue | resolved tonight? | what's left |
| --- | --- | --- |
| 1. Queue timeout | Instrumentation added, **extended 2026-09-11 to all four in-scope functions** | Measurement, then a decision on a number |
| 2. Comms-loss recovery | Design proposal written; **option B chosen and built 2026-09-11** | C and D undecided |
| 3. Integration setup failure | Fully resolved (reload worked) | Nothing — recorded for awareness only |

No motion-control-law value changes tonight. No profile changes tonight. The
series stays aborted at 3 of 5 scored until issue 1 has been measured, not
just theorized about.
