# PREDECLARATION — RTK square-return test, 2026-09-16

**Written ~03:00Z, before any pulse is dispatched.** Operator's design: drive
~2 m, stop, turn left ~90°, four times, and see how close the mower returns to
its start. Area declared clear by the operator, with enough room that even a
100 % overrun hits nothing.

## 0. What this does and does not measure

🔑 **This is NOT a re-opening of standing decision 3 (accuracy CLOSED).** That
decision covers *click-to-path landing accuracy* — the whole executor loop with
VIO correction. This test runs **no click-to-path leg, no VIO, and no
correction**: raw timed velocity pulses only. It asks a different question.

**It measures:**
- **RTK internal consistency** — do RTK's own reported corner positions form a
  plausible square, and do reported leg lengths match the tape?
- **RTK vs. physical ground truth** — the operator marks the start physically;
  at the end, tape-measure the real offset from start and compare it to the
  offset RTK claims. **This is the headline number.**
- Per-leg driven bearing, derived from RTK displacement (see §2).

**It cannot measure:**
- Whether a commanded 90° turn produces 90°, except after the fact (§2).
- Anything about click-to-path reliability. Closure error here mixes RTK error,
  drive error and turn error together; only the tape-vs-RTK comparison isolates
  RTK.

## 1. Why this is possible tonight with VIO dead

`vio_tracked_features` is **0** and `visual_positioning_status` is
`signal_none` — the yard is fully dark. The vector executor and both VIO turn
paths are therefore unavailable (their `vio_active` gates refuse, correctly).

`manual_velocity_pulse_test` is a **raw timed velocity pulse** with no VIO gate
and no heading requirement (verified in `_manual_velocity_pulse_gates`). Its
gates are: BLE transport + liveness, operator confirmations, blades off,
mower ready, off-dock/not-charging, live position, `AREA_INSIDE`, nonzero pose.
It **is** wrapped by `_wrap_exclusive_manual_motion`, so the motion gate must be
armed and `rtk_not_precise` still blocks — RTK must read `Fix`.

## 2. Heading without a compass: bearings from RTK displacement

With VIO dead there is no live heading (`map_facing.confidence: unknown`). The
bearing actually driven on leg *k* is recovered afterwards from RTK alone:

    bearing_k = atan2(y_end − y_start, x_end − x_start)

and the turn actually achieved between legs is `bearing_(k+1) − bearing_k`.
Turn 1's measured result **calibrates** the pulse duration for turns 2–4; the
first turn is an open-loop guess from banked rotation constants (angular 120 →
−9.175 °/s, angular 180 → −13.431 °/s, both at linear 300, **n = 1 each,
mechanism unexplained — do not fit a law to them**). Expect the first turn to
be off; that is data, not failure.

## 3. Movement parameters, fixed now

- `speed: 0.6` → linear **450**, angular **202** (app-scale, 15 % deadband).
- `duration_ms: 4000` (schema ceiling) per pulse.
- 🚨 `motion_refresh_interval_ms: 200` on **every** pulse. Without it a
  single-shot pulse travels ~4 in regardless of duration (2026-07-22 B1
  finding, ~11× difference). A run with refresh 0 is void.
- `stop_mode: immediate` — every pulse is bounded by its own stop.
- Forward legs: repeat pulses until RTK reports ≥ 2.0 m from the leg's start.
  Closing the loop on RTK here is deliberate and **not circular**: RTK error
  then shows up as *physical* error, which the tape catches independently.

## 4. Scoring, fixed before data exists

Per leg: RTK-reported length, RTK-derived bearing, pulses used.
Per turn: achieved angle (§2) vs. 90° commanded.

**Headline:** at the end, with the mower stopped, tape-measure the real
distance from the physical start mark. Call it `physical_closure_m`. Read
RTK's own claimed offset from the start coordinate, `rtk_closure_m`.

- **`|physical_closure_m − rtk_closure_m|` is the RTK accuracy figure.** Small
  (≲ 0.10 m) means RTK tracked the real motion faithfully regardless of how
  badly the square itself closed. Large means RTK's position is drifting from
  reality — the thing the operator suspects is *better* than believed.
- The square's own closure error (`physical_closure_m` alone) is **not** an RTK
  number; it is dominated by open-loop turn error and is reported separately.

## 5. Abort conditions

Any of these stops the run, gate disarmed, no retry without an operator call:
RTK leaves `Fix`; BLE link fails a pulse's gates (it read **−86 dBm** at the
dock tonight and dropped repeatedly all evening); blade reports anything but
off; position leaves the known area; any pulse's stop fails to confirm.

## 6. Closure

Once the first real pulse is dispatched, §0–§5 are closed to amendment.
