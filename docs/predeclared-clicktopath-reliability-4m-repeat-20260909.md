# PREDECLARED — click-to-path reliability, 4.0 m aligned start, REPEAT on beta103 (2026-09-09)

**Written before any run of this series exists.** Supersedes
`docs/predeclared-clicktopath-reliability-4m-20260903.md` as the live baseline
series. That document's own outcome
(`docs/findings-clicktopath-reliability-4m-20260904.md`) §8.3 requires a **new**
predeclaration for any repeat, and §4 requires that the repeat *fix how the
aligned start is established* before it is worth dispatching. This does both.

🛑 **THIS AUTHORIZES NO RUN.** It fixes the geometry, configuration, scoring
precondition, criterion, falsifier and abort rule **in advance**. Every dispatch
still needs its own explicit operator go/no-go immediately before it.

Serving **standing decision 2** — *"the goal is consistency, not precision:
click-to-go reliable enough to trust without watching."*

---

## 1. Why repeat rather than proceed to the L-path

The 2026-09-04 series is **void, not merely failed.** Its §4 established that the
"aligned start" precondition was **never met on any of the four runs**: the check
compared `target_reported_heading_degrees` against `toward`, which agree *by
construction* whenever the target was placed along `toward`. The executor's own
VIO calibration drive showed true facing was **26 / 27 / 122 / 135°** off, so
runs 3 and 4 opened with real ~120–135° turns — making them **post-turn legs**,
the exact property that series had put out of scope.

A verdict computed over a population that does not match its own inclusion
criterion is not a measurement of anything. **The 4.0 m aligned figure is
therefore unknown on the current control law**, and
`docs/predeclared-clicktopath-reliability-lpath-20260904.md` forbids itself from
dispatching without a same-build baseline to compare against.

**beta102 shipped the fix that makes the repeat meaningful**
(`custom_components/mammotion/services.py`, `_start_alignment_evidence`):
`start_geometry` is filled in **only** from the VIO calibration drive's
independently measured `map_motion_heading_degrees`, and is
`None`-with-a-reason everywhere else. It never derives from `toward`.
beta102 also corrected the heading model to the mirror `90.13 − toward`, which
the banked 43-pulse dataset shows is right to a mean **1.000°** where the
additive offset was off by a mean **87°**.

⚠️ **`docs/accepted-profile.json` is UNTOUCHED by this series and no Gate 5 is
owed.** This measures the accepted profile; it does not tune it.

---

## 2. The prior, stated honestly

| source | n | result |
| --- | --- | --- |
| `docs/loop-to-tolerance-reach-20260811.md` §1 | 2 | 1 reached (0.1023 m), 1 `vio_realign_incomplete` at 0.5493 m (BLE-caused) |
| 2026-09-04 series | 4 | **VOID** — inclusion criterion never met; not a prior for an *aligned* series |

🔑 **The usable prior at 4.0 m aligned is n = 2, one of which failed.** The
already-recorded failure mode is a BLE-caused mid-drive realignment abort, not a
control-law miss. A single failure of that shape is **not a new finding**; what
this series measures is its **rate**.

⚠️ **BLE health is a covariate, not a gate.** Record `ble_link_live`,
`queue_settle` and `ble_rssi` per run. But `ble_rssi` is self-reported and does
**not** predict cadence (within-run median r = +0.042 over 24 runs), so a
marginal RSSI is neither a reason to postpone a run nor to trust one.

---

## 3. Configuration — FROZEN, and the two traps that would void the series

**Service:** `raw_pymammotion_execute_vector_segment`, one segment, **4.0 m**.

🚨 **TRAP 1 — the CARD CANNOT RUN THIS.** It auto-splits any leg over
`SPLIT_LEG_TARGET_METRES` (**3.85 m**, verified in the tree), so a 4.0 m click
becomes two sub-legs and **measures the splitter**. Dispatch the service
directly.

🚨 **TRAP 2 — SCHEMA DEFAULTS ARE NOT THE ACCEPTED PROFILE.**
`max_linear_pulse_ceiling` defaults `None` (accepted **22**) — without it the run
is fixed-budget and stops after ~1 pulse; `waypoint_tolerance` defaults 0.08
(accepted **0.15**); `calibrated_forward_heading_offset_degrees` defaults
**116.5** where the profile is **102.4**.

✅ **Send `docs/accepted-profile.json` verbatim and verify identity key-by-key
against the echoed response.** A run whose echoed profile differs on any key is
**discarded, not scored** — it is not a sample of this population.

**Daylight throughout.** Turns close on VIO.

---

## 4. 🚨 The scoring precondition — the whole point of the repeat

A dispatched leg is **SCORED** only if **all** of the following hold, read from
the run's own response and recorded per item:

1. `start_geometry.basis == "vio_calibration_drive.map_motion_heading_degrees"`
   — never `None`, never any other basis;
2. `start_geometry.aligned_start_confirmed == true`, i.e.
   `initial_heading_error_degrees ≤ 10.0` (the shipped
   `_ALIGNED_START_TOLERANCE_DEGREES`);
3. `runtime_state.map_facing.confidence == "motion_confirmed"` **and**
   `safe_to_aim_dispatch == true` at the moment of dispatch — motion-confirmed
   has a 300 s TTL (`_FACING_MOTION_CONFIRMED_TTL_SECONDS`), so a stale one does
   not qualify;
4. the echoed profile is identical to `docs/accepted-profile.json` on every key.

🔑 **A leg failing any of these is UNSCORED and re-set-up — it is neither a pass
nor a fail.** It never silently joins the population. This is the clause whose
absence voided the 2026-09-04 series.

⚠️ **An unscored leg is still a real dispatch** and gets the full safety
protocol in §7. Unscored means "not evidence", not "not dangerous".

⚠️ **Target n = 5 SCORED legs.** Unscored legs do not count toward n and are
reported separately with their reasons.

---

## 5. The pass criterion — FIXED NOW

**PASS** requires **all** of:

- **(a)** **5 of 5** scored legs return `stop_reason: target_reached`;
- **(b)** every landing ≤ **0.15 m** (implied by (a); stated separately so a
  future tolerance change cannot silently move the bar);
- **(c)** no scored leg exhausts all 3 mid-drive realignments — the correction
  budget retains margin at this length;
- **(d)** zero safety-gate trips, zero containment breaches, stop confirmed on
  every leg.

**Anything else is a FAIL of the series.**
🚨 **A leg that stops safely on a named refusal is a FAIL, not a smaller
number.**

⚠️ **The criterion is deliberately 5/5, not "≥4 of 5".** With a prior of 1/2, a
criterion tolerating one failure cannot distinguish the status quo from an
improvement. A 4/5 is a **FAIL with an informative failure**, and the response
is a predeclared follow-up — **not** a retroactive softening of this line.

⚠️ **State the statistics honestly.** 5/5 gives a 95% lower bound on the true
success rate of only ~**55%** (rule of three). **n = 5 cannot demonstrate
"reliable enough to trust unwatched"**, and no write-up of this series may say
that it does. What it can do is detect a high failure rate (≳40%) with good
probability, and yield a landing **distribution** rather than a point.

---

## 6. The falsifier — FIXED NOW

**The claim under test:** *on beta103, a genuinely aligned 4.0 m single segment
lands inside tolerance repeatably.*

**It is WRONG if** ≥2 of 5 scored legs fail to reach target, **or** any scored
landing exceeds 0.15 m. Either outcome means the L-path series does **not**
open, and the failure mode is written up instead.

**Abort the series immediately** and report partial n if any of:
- any containment breach, or any leg leaving the frozen corridor;
- **two consecutive** legs failing to reach target;
- any leg requiring reverse recovery, or ending outside the corridor;
- battery below **35%** off-dock at the start of a leg;
- `ble_link_live` off, or `queue_settle` not live at depth 0;
- **any `1309` or `1425` in `last_error_code` or `logged_faults` during the
  series** (both are now visible per beta102 — `1309` is the orientation fault
  that was invisible on 2026-09-04, `1425` is the 3D-vision fault seen
  2026-09-08);
- the operator withholding go/no-go for any reason.

🔑 **An aborted series is reported as an aborted series.** Partial n is a result,
not a draft.

---

## 7. Safety preconditions — every leg, no exceptions

- 🚨 **Fresh corridor scan against the MAP every leg.** `step_path_contained`
  measures clearance against the **operator-supplied polygon**, not the mowing
  area — a position with 2.8447 m of real clearance once passed 15/15 gates
  against a 3.20 m requirement because the corridor was centred on the mower by
  construction. **The gate is not a substitute for the scan.**
- 🚨 **The map does not check the ground.** On 2026-09-04 a map scan showed
  3.5 m where the operator's tape measured **2.79 m** to a real fence. **On any
  corridor tighter than a couple of metres, ask for a physical measurement.**
- 🚨 **Derive facing two ways before every armed dispatch** — the last driven
  leg's bearing and `(90.13 − toward)` with `toward` fresh — require agreement,
  and **state the destination in compass terms for the operator**. A short
  "test" move is an armed dispatch.
- ⚠️ **Verify the physical e-stop is CLEAR.** A forgotten e-stop is invisible in
  telemetry; it silently no-op'd five motion commands over ~40 min on
  2026-07-19 while every health indicator read green.
- ✅ **VIO precondition, new for this series:** confirm `vio_tracked_features`
  at its saturated value (**80**) and **no `1425` in the current fault log**
  before the first dispatch of the session. All three recorded `1425` events are
  after dark and therefore confounded; this check in daylight is what turns them
  into a non-issue or a blocker.
- Daylight, operator present, blades off confirmed, `confirm_blades_off` and
  `confirm_clear_area` both true.
- A dry run of the identical payload immediately before each armed dispatch.
- Explicit per-leg operator go/no-go **immediately before dispatch** — not once
  for the series.
- Gate disarmed and verified from the live API **and** RAW `core.config_entries`
  after every leg. ⚠️ HA writes `.storage` lazily; a RAW read taken immediately
  after a disarm can lie for ~15 s.
- ⚠️ **Do not leave the mower off-dock and BLE-connected between legs.** It does
  not doze while the connection is held and drains ~**5.74 %/h** (measured
  2026-09-08, 51% → 0% over 8h53m). Dock it between legs if there is any wait.

---

## 8. What a PASS authorizes — and what it does not

A PASS authorizes **one thing**: opening
`docs/predeclared-clicktopath-reliability-lpath-20260904.md`, which now has its
same-build baseline.

🛑 It does **not** authorize:
- resuming Phase 2 continuous steering — **standing decision 5, CLOSED**;
- reopening accuracy — **standing decision 3, CLOSED** at the 0.065 m floor;
- reopening reach — **CLOSED at 6.0 m**;
- reopening night — **standing decision 4, CLOSED**;
- resuming OTA work — **standing decision 6, PARKED 2026-09-08**;
- raising `vio_max_realignments`, `max_linear_pulse_ceiling`,
  `vio_turn_max_commands`, `max_turn_translation_distance`, or any other bound;
- changing any key in `docs/accepted-profile.json` — that needs its own
  predeclaration and its own Gate 5;
- removing any per-leg operator confirmation, or dispatching unwatched;
- quoting a success **rate** from n = 5, or pooling with pre-beta57 landings —
  **standing decision 7**.

**A FAIL authorizes nothing beyond writing up the failure mode.**
