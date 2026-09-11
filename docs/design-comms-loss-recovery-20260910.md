# DESIGN PROPOSAL — what happens after `stop_failed_aborting`, and who decides (2026-09-10)

✅ **DECIDED 2026-09-11 — options B and C, both built.** The operator chose **notify only,
triggering on BOTH `command_failed` and `stop_failed_aborting`** (see the scope
correction in §4B), then **C as the fast follow** (see the correction in §4C).
**D remains unbuilt and undecided** — predeclaration at
`docs/predeclared-comms-abort-auto-dock-20260911.md`. Implementation record:
`docs/findings-comms-abort-notify-20260911.md`.

**This was a decision request, not a predeclaration and not an implementation.**
Nothing here was authorized when written. It presents the current behavior as verified in the
tree, states the gap against a standing decision, and lays out concrete options
for the operator to choose between. No code changes until one is picked.

**Why now:** leg 4 of the 2026-09-10 series hit this exactly
(`docs/findings-clicktopath-reliability-4m-repeat-20260910.md` §1.3). The
mower was fine — confirmed stationary by three position samples — but that
confirmation took a human checking, minutes after the fact, by choice. Nothing
in the system would have done it otherwise.

---

## 1. Current behavior, verified in the tree — do not change this part

Five call sites in `custom_components/mammotion/services.py`
(`raw_pymammotion_execute_vector_segment`'s calibration-turn phase,
turn-to-heading phase and linear phase; `manual_velocity_pulse_test`; the
final-approach loop) all do the identical thing:

✏️ **Corrected 2026-09-11 against the tree.** Both the count and the attribution
above are wrong, and the correct map is in
`docs/findings-clicktopath-reliability-4m-repeat-20260910.md` §1.5.1. There are
**four** `stop_failed_aborting` sites, not five, in `_vio_turn_to_heading`,
`_raw_pymammotion_execute_segment`, `_vio_segment_calibration_drive` and
`_raw_pymammotion_execute_vector_segment` — not in `manual_velocity_pulse_test`
or a final-approach loop, and one of them (`_raw_pymammotion_execute_segment`)
belongs to a service this series never ran. The shared-comment claim does not
survive a grep either. The behaviour described below is accurate as behaviour:

1. Send a motion pulse.
2. Attempt to stop it.
3. **If the stop cannot be confirmed delivered, abort immediately.** Do not
   send another motion command. Return `stop_reason: "stop_failed_aborting"`.

**This is correct and should not change.** It is the reason leg 4's mower was
confirmed stationary rather than still moving — sending a *second* command
into a connection that just failed to deliver a *stop* would be strictly worse.
The 2026-07-12 date means this predates today's session by two months; it is a
tested, deliberate design, not something written under pressure this
afternoon.

## 2. The gap: verified, not inferred

Grepped the whole integration for `stop_failed_aborting` outside
`services.py`: **zero hits.** Nothing in `coordinator.py`, `lawn_mower.py`, or
the card consumes this value. Concretely, when it fires:

- No HA notification (persistent or mobile) is created.
- No automatic reconnect-and-verify attempt happens.
- No automatic `return_to_dock` is triggered.
- No entity changes state to reflect "a leg aborted uncertainly."

It is a string inside one service-call response. If nobody is watching the
response when it comes back, nobody finds out.

## 3. Why this matters: it conflicts with standing decision 2

*"The goal is consistency, not precision — click-to-go reliable enough to
trust without watching."* An unwatched session that hits this leaves the
mower wherever it stopped, indefinitely, with the only record being a
service-call return value nobody read. **"Reliable enough to trust without
watching" cannot be true while this gap exists**, independent of how well the
control law performs otherwise — today's leg 3 proves the control law is fine.

## 4. Options — pick one, or a sequence

Ordered by how much they change today's motion behavior, least to most.

### A. Do nothing
Status quo. Zero implementation risk, zero new code. Leaves standing decision
2 unreachable for any session that isn't actively supervised end to end,
which so far has been every real session — so this is not a change, it is a
decision to accept the current limit explicitly rather than by default.

### B. Notify only ✅ CHOSEN AND BUILT 2026-09-11
When any of the five sites returns `stop_failed_aborting`, fire an HA
`persistent_notification` (and/or mobile push, if configured) naming the leg,
the last known position, and the time. **No new motion command, ever.** Purely
additive — a read of state that already exists, surfaced instead of silently
returned.

🚨 **Scope correction, 2026-09-11 — this wording would have missed the point.**
Triggering on `stop_failed_aborting` alone covers **leg 4 only**. Legs 7 and 8 —
**the two that tripped the series' own abort rule** — both returned
`command_failed`, so notify-as-written would have stayed silent for exactly the
pair that stopped the series. As built, the trigger set is
`{command_failed, stop_failed_aborting}`, and both `stop_reason` and `reason`
keys are read (the calibration drive uses the latter for the identical
condition).

⚠️ **Mobile push is deliberately NOT integration config.** The build fires a
`mammotion_motion_comms_abort` event on the HA bus alongside the persistent
notification, so an operator automation routes it onward — no notify-service
name is stored in this integration, and no manifest dependency was added.

**Risk:** essentially none. Does not touch the motion path.
**What it does not solve:** the operator still has to physically go check;
it only guarantees they *know* to.

### C. Notify, then auto-verify stationary ✅ CHOSEN AND BUILT 2026-09-11
B, plus: once BLE contact is confirmed restored (`queue_settle.is_connected`),
automatically pull `export_runtime_state` a few times over a short window and
confirm position is unchanged — exactly the manual check done live this
session — then update the notification with the result ("confirmed stopped"
or "position still changing, needs attention").

🚨 **Correction, 2026-09-11 — "confirm position is unchanged" is a TRAP as
written, and implementing it literally would have made C actively dangerous.**
This project has already recorded the failure mode twice
(`telemetry_stream_stale`, `_streak_shows_dead_telemetry`): **bit-identical
position samples mean the feed is dead, not that the mower is still.** After a
comms abort a frozen feed is the *likely* case — so the naive check would report
a confident "confirmed stopped" at precisely the moment it had gone blind, and
going blind demands the opposite operator response (fix the link and go look).

As built, liveness is proven independently of position: `handle.position_epoch`
advances on every position report, and an unchanged position is only allowed to
mean anything if the epoch moved during the window. Four verdicts, deliberately
asymmetric — two of them are "cannot confirm", not "fine":

| verdict | meaning |
| --- | --- |
| `confirmed_stationary` | epoch advanced, spread ≤ 0.05 m — a real confirmation |
| `still_moving` | epoch advanced, spread > 0.05 m — 🚨 alarm |
| `cannot_confirm_feed_stale` | epoch never moved — blind, **not** a stop |
| `cannot_confirm_link_down` | BLE never returned within 20 s |

⚠️ **It reads `_custom_path_telemetry_snapshot`, not `export_runtime_state`, and
requests no reports.** A report request shares the very BLE command queue whose
failure caused the abort. The consequence is honest but real: if nothing else is
driving the report stream, C returns `cannot_confirm_feed_stale` rather than a
verdict. That is the correct answer to "I cannot see the mower", and it is a
candidate for a later decision (whether to spend one report request to get a
real answer) — not something to paper over.

The 0.05 m tolerance is set by the position feed's **absolute** 2–4 cm noise
floor; anything tighter flags noise as motion.

**Risk:** low. Every call involved is already a read-only diagnostic
(`export_runtime_state`); this only automates a sequence already done by hand
today. No motion command is ever sent.
**What it does not solve:** the mower is still wherever it stopped; if that's
an inconvenient or exposed spot, someone still has to retrieve it.

### D. Auto-trigger `return_to_dock` once stationary is confirmed
C, plus: after stationary is confirmed and enough time has passed with no
operator response (configurable), automatically issue `return_to_dock` — the
vendor's own navigation, with its own onboard obstacle handling, not our raw
motion primitives.

**Risk: real, and different in kind from A–C.** This is the one option that
sends a **new, unattended, un-confirmed-by-a-human motion command** after a
failure whose defining feature was "we could not confirm delivery of the last
command." It could reasonably run into the same BLE conditions that caused the
abort in the first place. It is also the only option that touches
`custom_components/mammotion/coordinator.py`'s job-control surface rather than
staying inside the diagnostic/read-only layer.
**What it solves:** the actual "stranded and nobody's coming" scenario, which
is the one standing decision 2 cares about most.

---

## 5. Recommendation, stated as a recommendation

**B now.** It's a small, safe, purely additive change that closes the biggest
part of the gap (operator ignorance) at essentially zero risk, and it does not
require touching anything in the motion control law.

**C as a fast follow**, once B is proven — it's still read-only, and it turns
"a human should check" into "the system already checked and here's the
answer," which is most of what standing decision 2 actually asks for.

**D needs its own separate, explicit decision later**, not bundled into this
one. It's the only option here that meets the bar this project reserves for
motion-control-law changes — real consequence, needs its own predeclaration
and its own scrutiny, and should not inherit approval from B or C.

**This document authorizes none of these.** Say which option (or sequence) to
build, and that becomes the next predeclared implementation task.
