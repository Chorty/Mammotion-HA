# The 4.0 m aligned repeat is INCOMPLETE at 1 of 5 scored, and the practical limiter is BLE range, not the control law (2026-09-10)

Predeclared in `docs/predeclared-clicktopath-reliability-4m-repeat-20260909.md`
(committed `1acc8aa9`) and amended mid-session by
`docs/predeclared-clicktopath-reliability-4m-repeat-20260909-amendment1-20260910.md`
(committed `5aeba62f`, **before leg 4 dispatched**).

Build: **beta103** (`4f908b04`), backend `chorty-0.8.12.post4`, restored to the
host earlier the same day after HACS overwrote it (see
`docs/deploy-runbook-p0.md`). Service
`raw_pymammotion_execute_vector_segment` dispatched directly; the card was not
used. Accepted profile sent verbatim and verified key-by-key on **every** leg —
0 mismatches, 4 of 4.

---

## 0. 🚨 The series did NOT complete. Do not quote it as a rate.

**1 of 5 scored legs.** Four real dispatches produced one scored PASS, two
unscored legs and one FAIL. **n = 1 supports no claim about reliability**, and
the predeclaration's own §5 statistics note applies with more force here, not
less. This document records what happened; it does not conclude the series.

| leg | stop_reason | landing | aligned? | scored |
| --- | --- | --- | --- | --- |
| 1 | `target_reached` | **0.1277 m** | ✅ 6.213° | ✅ **PASS** |
| 2 | `target_reached` | 0.0523 m | ❌ 28.361° (`post_turn_leg`) | unscored |
| 3 | `target_reached` | 0.1417 m | ✅ 3.979° | unscored (old condition 3) |
| 4 | **`stop_failed_aborting`** | 0.29 m of 4.0 m | ❌ 156.797° | **FAIL** |

⚠️ **Leg 3 is the frustrating one.** It satisfied every condition that measures
the leg's own quality — aligned to 3.979°, landed 0.1417 m, 13/13 gates, profile
identical — and was unscored only on the original condition 3
(`map_facing.confidence == motion_confirmed` at dispatch), which had gone stale
on its 300 s TTL during the mandatory between-leg bookkeeping. **Amendment 1
dropped that condition for leg 4 onward and explicitly did NOT rescore legs 2–3.**
Rescoring them after seeing the verdicts would be the exact failure the
predeclaration discipline exists to prevent.

---

## 1. What was learned that is worth more than the verdict

### 1.1 🔑 Aim each leg at the live `map_facing_degrees`, not at a fixed bearing

**This is the session's most reusable finding.** Leg 2 was aimed "due south"
because leg 1 had been — copying an absolute compass bearing. But the executor
stops on *position* tolerance, not on final orientation, so the mower's facing
after leg 1 had drifted to ~239°, not the 270° it had driven. The 31.6° gap
between the fixed bearing and the real facing is what produced leg 2's 28.361°
`post_turn_leg`.

Leg 3 was aimed at the live `map_facing_degrees` reading instead and came back
`aligned_start_confirmed: true` at **3.979°**. Same code, same profile, one
different target-selection rule.

🔑 **`map_facing_degrees` is populated under mere corroboration** (two heading
sources agreeing within 15°) and does **not** require `motion_confirmed`. The
value was available for leg 2 and simply was not used. `safe_to_aim_dispatch`
is the stricter operator-facing gate; the *degrees value* is the planning input.

### 1.2 🚨 The series walked itself into a corner, and that was predictable

Every leg's target was chosen reactively and validated only as "is this one
target inside the polygon". In a ~15 × 16 m area with 4.0 m hops that is an
unmanaged random walk. By leg 4 the mower sat at (6.621, −9.1816) with **no
heading anywhere in the ±10° alignment window still inside the area** — verified
by sweeping all 21 offsets. The only escape was a 156.797° turn, which cannot
score by construction.

✅ **Fixed off-mower the same session:** `scripts/plan_aligned_leg.py` (committed
`2e5f3f36`) judges a candidate on the runway remaining *from its own target*,
searches the whole tolerance window, and names a required reset leg before it is
forced. ⚠️ Among candidates clearing the runway bar it prefers the heading
**closest to the measured facing**, not the one with the most runway — runway is
a threshold, alignment margin is not, and its first run picked the −10.0° window
edge until that rule was corrected.

### 1.3 🚨 BLE range is the practical limiter on how far a series can go

Legs 1–3 marched steadily away from the house. Proxy RSSI degraded with
distance, exactly as the standing "-70 works / -76 dies" note predicts:

| moment | best proxy RSSI |
| --- | --- |
| at the dock | −48 dBm |
| after leg 2 | −77 dBm (`p1s-printer`), −99 dBm (`garage-m5stack`) |
| after leg 3 | −80 to −84 dBm |

Leg 4 then failed on **`stop_failed_aborting`**, and the container log names the
mechanism directly: three `gatt_write` failures in
`pymammotion/bluetooth/ble_message.py` at 19:58:25, :35 and :39. The stop
command could not be written, so the executor aborted rather than continue with
an unconfirmed stop.

🏆 **That is the safety design working exactly as intended**, and it is the same
signature as the 2026-07-25 stop-delivery failure already on the record. The
mower was confirmed stationary immediately afterward by three position samples
(last two bit-identical, 2 mm total drift), and the gate was disarmed and
verified from the live API **and** RAW `core.config_entries`.

⚠️ **Nothing about leg 4 implicates the turn logic.** The 156.797° corner-escape
turn completed cleanly in 3 staged steps, every stage `target_heading_reached`,
and the executor correctly refused a *direct* turn first (`turn_budget`:
~7 commands estimated against a max of 4) before staging it.

### 1.4 ✏️ `ble_link_live` lagged the real transport state, in both directions

Before leg 4 the entity read `off` (374 s stale) while the gate's own blocker
list had already cleared `ble_client_not_connected` — the run then dispatched
fine. Earlier the same session the opposite happened. 🔑 **The gate's blocker
list and `queue_settle` are the authoritative sources; the entity is a lagging
mirror.** This is the standing "diagnose from `queue_settle` and the container
log, never from a proxy's entity state" rule, observed twice in one afternoon.

### 1.5 The VIO/`1425` question got a clean daylight answer

`vio_tracked_features` held at **80** (saturated) through the session with
`vio_feed_live: true`, and **no `1425` fired at any point**. Combined with the
previous day's mow (61 features under real motion, no `1425`), the after-dark
`1425` events look night-specific. ⚠️ Still not proof — nobody has tried to
reproduce one in daylight — but the daylight precondition in the predeclaration's
§7 passed cleanly on every check.

---

## 2. Protocol tension found, and how it was resolved

The original §7 required the gate disarmed and RAW-verified **after every leg**.
The original §4 condition 3 required `motion_confirmed` (300 s TTL) **at
dispatch**. Between legs, the disarm → RAW-verify (which includes a mandatory
~15 s `.storage` settle wait) → report → operator-reply cycle reliably exceeded
300 s. **The two requirements were structurally incompatible for any leg after
the first.**

Amendment 1 resolved it forward-only:
- condition 3 dropped from scoring — `start_geometry` is the live, non-circular
  measurement that actually establishes alignment, and it is taken by each run's
  own calibration drive regardless of any prior `map_facing` state;
- gate disarm + RAW verify moved from per-leg to **once at session end**,
  narrowing the "found armed at rest" guard to what it actually protects against
  without weakening it — every per-leg requirement (fresh fault check, fresh
  dry-run, physical corridor confirmation, explicit go/no-go) is unchanged.

🔑 **Legs 2 and 3 stay unscored.** The amendment is dated and committed before
leg 4 dispatched and does not reach backward.

---

## 3. Safety record

| | |
| --- | --- |
| safety gates | **13/13 passed on all four real dispatches** |
| keep-out violations | zero, every leg (2 zones checked each time) |
| containment breaches | zero |
| profile mismatches | zero, 4 of 4 legs |
| named refusals | one — leg 4's `stop_failed_aborting`, recorded as a FAIL |
| operator | present and confirming each leg's physical corridor individually |
| gate after session | **disarmed, verified from live API AND RAW** |
| mower after session | stationary, confirmed by 3 position samples; 91% battery |

---

## 4. What this authorizes

**Nothing beyond this write-up and the planner already committed.** The series
is incomplete at n = 1 scored.

🛑 In particular this does not reopen accuracy (standing decision 3), reach
(closed at 6.0 m), night (standing decision 4), Phase 2 (standing decision 5) or
OTA (standing decision 6, parked). No bound, tolerance or profile key changed —
`docs/accepted-profile.json` is untouched and no Gate 5 is owed.

**To resume**, the open questions are operational rather than about the control
law:
1. **BLE coverage** bounds how far from the house a series can run. Either a
   proxy moves closer, or legs are planned to stay within range — the planner
   makes the second tractable but does not model RSSI.
2. **Leg budget** must assume more dispatches than scored legs. Four dispatches
   produced one scored leg; reset legs are now expected, not surprises.
