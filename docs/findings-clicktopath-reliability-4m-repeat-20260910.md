# The 4.0 m aligned repeat is ABORTED at 3 of 5 scored, per its own predeclared rule — the practical limiter is a BLE command-queue timeout, not the control law (2026-09-10)

Predeclared in `docs/predeclared-clicktopath-reliability-4m-repeat-20260909.md`
(committed `1acc8aa9`) and amended mid-session by
`docs/predeclared-clicktopath-reliability-4m-repeat-20260909-amendment1-20260910.md`
(committed `5aeba62f`, **before leg 4 dispatched**).

Build: **beta103** (`4f908b04`), backend `chorty-0.8.12.post4`, restored to the
host earlier the same day after HACS overwrote it (see
`docs/deploy-runbook-p0.md`). Service
`raw_pymammotion_execute_vector_segment` dispatched directly; the card was not
used. Accepted profile sent verbatim and verified key-by-key on **every** leg —
0 mismatches, 8 of 8.

---

## 0. 🚨 The series ABORTED per its own rule. Do not quote it as a rate.

**3 of 5 scored legs.** Eight real dispatches: three scored PASS, two unscored
by design (post-turn / deliberate reset), three FAIL. The series stopped
itself on its own predeclared abort condition — **two consecutive legs failing
to reach target** (legs 7 and 8) — rather than being pushed through. **n = 3
supports no claim about reliability**, and every scored leg landing well
inside tolerance is a data point toward the aiming fix in §1.1, not a
conclusion about the series.

| leg | stop_reason | landing | aligned? | scored |
| --- | --- | --- | --- | --- |
| 1 | `target_reached` | **0.1277 m** | ✅ 6.213° | ✅ **PASS** |
| 2 | `target_reached` | 0.0523 m | ❌ 28.361° (`post_turn_leg`) | unscored |
| 3 | `target_reached` | 0.1417 m | ✅ 3.979° | unscored (old condition 3) |
| 4 | `stop_failed_aborting` | 0.29 m of 4.0 m | ❌ 156.797° | **FAIL** |
| 5 | `target_reached` | **0.0516 m** | ✅ 3.013° | ✅ **PASS** |
| 6 | `target_reached` | **0.0740 m** | ✅ 0.683° | ✅ **PASS** |
| 7 | `command_failed` | 0.54 m of 4.0 m (deliberate reset) | ❌ 162.836° | unscored, FAIL |
| 8 | `command_failed` | 1.39 m of 4.0 m | ✅ 0.821° (would have scored) | **FAIL — triggered abort** |

⚠️ **Leg 3 is the frustrating one.** It satisfied every condition that measures
the leg's own quality — aligned to 3.979°, landed 0.1417 m, 13/13 gates, profile
identical — and was unscored only on the original condition 3
(`map_facing.confidence == motion_confirmed` at dispatch), which had gone stale
on its 300 s TTL during the mandatory between-leg bookkeeping. **Amendment 1
dropped that condition for leg 4 onward and explicitly did NOT rescore legs 2–3.**
Rescoring them after seeing the verdicts would be the exact failure the
predeclaration discipline exists to prevent.

🔑 **Leg 8 is the most frustrating loss of the series.** It was correctly aimed (0.821°,
the best of the series bar leg 6), it would have made the scored total 4 of 5,
and it failed for a reason that had nothing to do with alignment, geometry, or
the control law — see §1.5.

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

### 1.5 🚨 The real failure mode is a 2.0 s BLE command-queue timeout — not RSSI, not range

Legs 4, 7 and 8 all failed differently on the surface (`stop_failed_aborting`
once, `command_failed` twice) but all three trace to the same family of
defensive code in `services.py`, and to one specific guard:

✏️ **Corrected 2026-09-11 against the tree.** This paragraph originally read
"`stop_failed_aborting` twice, `command_failed` twice" (four events for three
legs — §0's table says one and two) and described "**five call sites**, all
sharing an identical `# Never keep driving/turning when stops are not
deliverable` comment dated **2026-07-12**". None of that survives a grep: that
exact comment string matches **nothing**, there are **four** `stop_failed_aborting`
assignment sites (a fifth hit is a docstring), they sit in **four different
functions**, there are three comment variants, and only two carry the 2026-07-12
date. The mechanism below is unaffected — but where a fix has to land is not,
which is why it mattered. See §1.5.1.

```python
_BLE_QUEUE_DEPTH_LIMIT = 0
# "Real motion is never allowed behind existing queue work... even one
#  predecessor makes the local pulse timer diverge from the mower's
#  actual execution window."
_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS = 2.0
# "Maximum time a motion item may wait to start in the command queue.
#  If this expires the item is disarmed."
```

Every motion pulse gets **2.0 seconds to begin processing** in the single
serialized BLE command queue before the write itself even starts. That queue
also carries the profile's own `motion_refresh_interval_ms: 200` traffic —
leg 4's own phase data logged 14 refresh commands in a single turn phase. If
anything already occupies that queue slot — a refresh command still in
flight, a brief reconnect retry — for more than 2.0 s when the next pulse
tries to enqueue, the guard fires and the item is disarmed. **This is
deliberate, not a bug**: a late-dispatched pulse would already have drifted
out of sync with its own timing assumptions, so refusing is the safer choice.

#### 1.5.1 ✏️ Where these sites actually are (verified 2026-09-11)

| function | `command_failed` | `stop_failed_aborting` | reached by the vector executor? |
| --- | --- | --- | --- |
| `_vio_turn_to_heading` | ✅ | ✅ | ✅ its turn phase |
| `_raw_pymammotion_turn_to_heading` | ✅ | — | ✅ its turn phase |
| `_vio_segment_calibration_drive` | ✅ | ✅ | ✅ its calibration drive |
| `_raw_pymammotion_execute_vector_segment` | ✅ | ✅ | ✅ its linear phase |
| `_raw_pymammotion_execute_segment` | ✅ | ✅ | ❌ a different service |

🚨 **This is why the 2026-09-10 instrumentation was not enough.** It went into
the linear phase only, but the first three rows are *phases of the same
service*, executed on every leg — a leg dying in the calibration drive or a turn
produced the identical reason with **no queue snapshot at all**. Extended to all
four in-scope functions on 2026-09-11 (operator-approved, beyond the plan's
original step-1 scope); the ~17 sites in genuinely other executors stay deferred.

⚠️ **The calibration drive reports `reason`, not `stop_reason`**, for the
identical condition. Anything consuming these must read both keys.

🔑 **RSSI cannot see this at all.** It is a periodic sample of physical signal
strength; the queue-start timeout is a software-scheduling measurement on the
same connection. Leg 8 failed with `ble_rssi` reading a genuinely good
**−60 dBm** moments before and after, and the BLE coverage map (§1.3) had
independently estimated **−66.9 dBm** at that exact position from 120 real
samples — both correct, both irrelevant to what actually failed. Both leg 7
and leg 8 failed after several pulses had already **succeeded**, consistent
with transient queue contention rather than a standing range problem.

**This reframes §1.3.** BLE range bounds where a series *can* run at all; this
timeout is a second, independent failure mode that can strike inside a zone
with excellent average coverage. The coverage-map constraint in
`plan_aligned_leg.py` is real and correctly kept legs out of dead zones — it
just cannot protect against this.

### 1.6 The Mammotion integration itself failed setup mid-session, independent of the mower

Between legs 6 and 7, `report_stream_probe` returned an empty
`service_response` and `lawn_mower.back_yard_clip_skywalker` briefly did not
exist. The container log named it exactly: *"Setup of config entry
'Luba-VSPLV397' for mammotion integration cancelled"*, traced into
`fetch_rtk_properties` → the Aliyun cloud gateway, with the config entry
landing in `state: setup_error`, `reason: null`. This matches a documented
trap (a bootstrap timeout cancelling Mammotion setup, which never
auto-retries) rather than being a new defect. A config-entry reload (not a
full HA restart) recovered it cleanly in ~20 s — `state: loaded`, gate options
(`enable_experimental_motion`) persisted through the reload, mower position
confirmed unchanged (nothing moved during the outage).

### 1.7 The VIO/`1425` question got a clean daylight answer

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
| safety gates | **13/13 passed on all eight real dispatches** |
| keep-out violations | zero, every leg (2 zones checked each time) |
| containment breaches | zero |
| profile mismatches | zero, **8 of 8 legs** |
| named refusals | three — leg 4 (`stop_failed_aborting`), legs 7 and 8 (`command_failed`), all recorded as FAIL |
| mower confirmed stationary after every abort | ✅ 3 position samples each time (leg 4: 2 mm drift; leg 8: bit-identical) |
| operator | present and confirming each leg's physical corridor individually |
| integration outage mid-session | ✅ recovered by config-entry reload; position confirmed unchanged across it |
| gate after session | **disarmed, verified from live API AND RAW** |
| mower after session | back on the dock (not charging), 66% battery |

**Every abort left the mower exactly where the last confirmed command put it.**
No command executed without being accounted for, no landing exceeded
tolerance, no keep-out was approached. The three failures were all refusals to
act on uncertain state, not incidents of unintended motion.

---

## 4. What this authorizes

**Nothing beyond this write-up and the tooling already committed.** The series
is **aborted, not concluded**, at n = 3 scored — one leg short of even the
weak statistical floor its own §5 describes (a 95% lower bound from 5/5 needs
all five; 3/3 so far says less than that).

🛑 In particular this does not reopen accuracy (standing decision 3), reach
(closed at 6.0 m), night (standing decision 4), Phase 2 (standing decision 5) or
OTA (standing decision 6, parked). No bound, tolerance or profile key changed —
`docs/accepted-profile.json` is untouched and no Gate 5 is owed. In particular
**`_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS` and `motion_refresh_interval_ms`
were NOT touched** despite being implicated in §1.5 — see
`docs/plan-post-20260910-session-issues.md` for why that needs measurement
before it needs a number changed.

**To resume**, in order:
1. **The queue-timeout failure mode (§1.5)** needs its own investigation before
   more legs are worth dispatching — two consecutive command-queue timeouts in
   ten minutes, independent of good RSSI, is the reason the series stopped.
2. **BLE coverage** bounds where a series can run at all; the planner now
   models it, but it is a distance/dead-zone constraint, not a queue-timing one.
3. **Leg budget** must assume more dispatches than scored legs. Eight
   dispatches produced three scored legs; reset legs and comms aborts are both
   now expected, not surprises.

Full plan: `docs/plan-post-20260910-session-issues.md`.
