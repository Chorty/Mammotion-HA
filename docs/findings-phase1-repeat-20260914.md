# FINDINGS — Phase 1 repeat, 2026-09-14

Three sessions, same day, same operator. Sessions 1–2 shared the frozen RF set
(`hci0` scan-only, `hot-tub-backyard`, `p1s-printer`, `garage-m5stack`,
`atom-fireplace`); session 3 ran after the operator physically removed
`hot-tub-backyard`. Scored against
`docs/predeclared-queue-timeout-measurement-repeat-20260913.md`, using
`scripts/score_queue_measurement.py` **unmodified** (`3cd3b22d`) per its §2.8.
Pre-dispatch notes, written before each session's first sample:
`docs/predispatch-note-phase1-repeat-20260914.md` (session 1),
`docs/predispatch-note-phase1-repeat-session2-20260914.md` (session 2),
`docs/predispatch-note-phase1-repeat-session3-20260914.md` (session 3).

---

## 0. Verdict

**Session 1 (20:59–21:41Z): INCONCLUSIVE, truncated by BLE transport loss.**
**Session 2 (22:00–22:34Z): scored, both axes measured. Axis 1 PASS. Axis 2 verdict
`4_inconclusive` — the predeclared middle band, not a protocol failure.**
**Session 3 (23:21–23:42Z, dusk, 3-proxy RF, operator override): all 12 targets
dispatched, 0/71 unclassifiable, but INCONCLUSIVE on axis 1 — the timing history
hit its 500-sample cap mid-session and evicted session-3's own earlier samples.**

- 🛑 `_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS` **stays 2.0.** No session
  produced grounds to move it.
- Session 2's `queue_wait_ms` p95 (pulse-open class, budget-2.0 samples) is
  **289.9 ms** — above the 250 ms "fine" bar, well under the 1000 ms "raise" bar.
  Per predeclaration §5 that range is defined as inconclusive; it is not a
  near-miss of either side, and no threshold was adjusted to produce this reading.
- **Recorded, not over-read:** 363 session-2 samples, all `completed`, zero
  `queue_start_timeout`. Session 1 recorded 219 more before it was truncated;
  session 3 recorded 500 (capped). All `completed`, zero timeouts across every
  session. None are pooled together, and none is rescored under another
  session's outcome.
- **The proxy-adjacency question raised in-session is still unresolved** — see
  §6.3.

---

## 1. Session 1 — truncated by transport loss at the anchor

Dispatched S1–S6c (S6b, S7, S7b sent nothing): landings 0.061–0.173 m, 219
samples, all `completed`. Truncated before S7 could complete because the link
would not hold: HA's own path ranking from the anchor read −76 to −80 dBm on
both reconnects captured in the container log, at the documented failure wall.
The committed scorer's own read: 38 pulse-open samples (< 40), S6 unreconcilable
(a pulse's own second sample never arrived — the link died mid-write), and the
truncation flag set ⇒ `INCONCLUSIVE (axis 1)`. Its 2/40 unclassifiable count
shows **the repeat's classifier held even under a mid-session BLE failure** — the
2026-09-13 defect (23 gaps splitting a pulse from its own refreshes) did not
recur. Evidence: `docs/evidence-phase1-repeat-20260914-session1/`.

Diagnosis, from the HA container log (`habluetooth.wrappers` path-ranking
lines): from the parked position ~3.6 m north of the anchor and from the anchor
itself, every one of the four frozen proxies ranked between −76 and −95 dBm.
There is no strong path from this spot with this proxy set — unlike 2026-09-13,
where the anchor itself measured `strong` (−66 to −68) and only the southern
legs (7, 8) were weak. The difference is plausibly the parked start point, not a
change in proxy behaviour; not isolated by this session.

**Operator observation, addressed and not the cause:** the operator raised
whether S9's turn halt (session 2, see §3) was an RTK dropout. History across
the surrounding 20 minutes shows `rtk_position` held `fix` continuously (no
state change from 22:05:05Z on), and the RTK base's own WiFi/position readings
updated normally at 22:12–22:13Z. The halt's own record cites
`turn_budget_infeasible`, unrelated to position — see §3.

---

## 2. Protocol change mid-day, on operator instruction

Between sessions 1 and 2 the operator asked to (a) not stop for a per-leg go,
since the approval wait was itself the idle time in which the link kept
dropping, and (b) keep the same four proxies rather than move or disable one,
even though two of them (`hot-tub-backyard` and `p1s-printer`) sit physically
adjacent and offer no real path diversity from the backyard. Recorded, not
adjudicated at the time: whether that adjacency was the dominant limiter —
addressed later in session 3, §6.

Session 2 ran under a **standing go for S1–S12**, predeclared before its first
dispatch in `docs/predispatch-note-phase1-repeat-session2-20260914.md` §2. Every
per-leg safety check in `scripts/phase1_leg_runner.py` (§16.3) still ran
unchanged: position/zone/RTK/blade/facing/corridor/excursion/band/dry-run/
stop-reason. What changed was *approval cadence*, not any safety threshold. The
automated sequencer retried a target automatically only on `ble_client_not_connected`
with nothing sent, or a VIO tracked-feature dip with the sun still ≥ 10° (after a
60 s wait); any other halt was designed to stop the sequence and did (S9, §3).
The gate was armed immediately before each dispatch and verified disarmed after
every leg, including the final one — confirmed in the live API and RAW
`core.config_entries` at 22:34:29Z, post-dock.

---

## 3. Session 2 — all 12 targets dispatched

| leg | landing (m) | note |
| --- | --- | --- |
| S1 | 0.142 | turn-around (north→south) |
| S2 | 0.079 | |
| S3 | 0.136 | |
| S4 | 0.086 | turn-around (south→north) |
| S5 | 0.111 | |
| S6 | 0.120 | |
| S7 | 0.116 | |
| S8 | 0.085 | |
| S9 | halt `turn_budget_infeasible` | retried as S9b |
| S9b | 0.145 | |
| S10 | 0.147 | |
| S11 | 0.017 | |
| S12 | 0.114 | |

All 12 targets reached (S9 via its retry). No dry-run mismatch, no keep-out
violation, no unreconcilable leg.

### 3.1 S9's halt, in the executor's own numbers

`turn_feasibility.reason: "translation_cap"`. The turn needed 42° more after two
staged pulses had already covered ~118° of a 171°+ turn-around; the executor's
own conservative model estimated the next pulse would need ~3 commands and
displace the mower **0.1486 m**, at or past its per-turn translation cap. It
refused rather than risk overshoot, exactly as designed. This is the same
failure mode as 2026-09-13's S9 (which also needed a retry, S9b, to land), so it
recurs specifically on turn-arounds near or above 170° and is not new. Standing
guidance (do not raise `vio_max_realignments`) is unrelated — this is a turn
*translation* cap, not a realignment-count cap.

### 3.2 Timing history reset mid-day

The coordinator's `motion_dispatch_timings` deque (`maxlen=500`) held 219
session-1 samples at session 1's close. Those were snapshotted to
`docs/evidence-phase1-repeat-20260914-session1/timing_session1_before_reload2.json`
before a second config-entry reload emptied it, so session 2's own capacity
clause (§ "no_history_capacity_drop") measured session 2 alone rather than
session-1 eviction. No proxy, profile, or motion constant changed; host stayed
beta104 throughout both sessions.

---

## 4. Axis 2, in full (session 2)

```
n_pulse_open_budget_2 = 67
q_p95_queue_wait_ms_pulse_open = 289.943
w_p95_write_ms_classified_completed = 365.431
write_inheritance_ratio = 0.7934      (only informative once Q >= 1000ms; not reached)
worst_wait_fraction_recomputed = 0.2121
legs_at_or_above_raise_fraction (>= 0.75) = 0
queue_start_timeouts_in_session = 0
rate_statement: "no rate claim below 120; the honest bound is 3/67"
verdict: 4_inconclusive
```

Per-leg worst-wait fraction of the 2.0 s budget ranged 0.058 (S5) to 0.212 (S6);
none reached the 0.75 "raise" fraction on any leg. 🔑 **The write-inheritance
ratio test (§12.1) is not evaluated** — it only fires once Q ≥ 1000 ms, and
today's Q is 290 ms. Its 0.79 value here is informational only, not a verdict
input. Session 1's truncation and session 3's capacity drop both mean axis 1
failed first, so neither has an axis 2 computed — the scorer never evaluates
axis 2 unless axis 1 passes.

---

## 5. Session 3 — dusk retest under the 3-proxy RF set

**Operator instruction, before the first dispatch:** physically removed
`hot-tub-backyard` (confirmed `unavailable` in HA at 22:39:27Z, dropped from the
scanner list by 23:11Z) because it sits adjacent to `p1s-printer` with no real
path diversity. This makes session 3's RF set **not comparable** to sessions
1–2's frozen five; it is a new configuration, tested on its own terms. Recorded
before dispatch in `docs/predispatch-note-phase1-repeat-session3-20260914.md`.

**Also on operator instruction:** the sun was below the runner's 10° floor for
the whole session (4.75° at first dispatch, below the horizon by S9). A narrow
`--allow-low-sun` flag was added to `scripts/phase1_leg_runner.py`
(`cca228ec`), on the operator's explicit go citing the 2026-09-12 VIO-collapse
precedent. It skips **only** the sun-elevation clause; `vio_tracked_features`
(≥ 70 over 60s) and `visual_positioning_status` (`signal_good` throughout) stay
fully enforced.

### 5.1 The VIO guard worked exactly as designed, twice

- **First S1 attempt halted before dispatch**, nothing sent: the tracked-feature
  window caught a real dip to 65 at 23:19:20Z, one second below the 70 floor,
  even though point-in-time status read `signal_good` throughout — the same
  pattern as 2026-09-12 (14 vs 80), just far milder. Retried as S1b 41 s later
  once the window cleared; landed 0.145 m.
- **S9 halted mid-leg** with `stop_reason: vio_realign_incomplete`, landing
  0.172 m short (just past the 0.15 m tolerance) — this is the *executor's own*
  realignment failing, a different mechanism from the runner's pre-dispatch
  window check, and not on the sequencer's auto-retry list. The sequence
  correctly stopped itself; the mower was stationary and safe. Retried on
  operator go as S9b (a short 0.17 m leg from the failed landing); reached
  target at 0.133 m.
- Both events are consistent with degraded VIO reliability at dusk, exactly the
  named risk. Neither produced an unsafe outcome: the guard caught one before
  dispatch, and the executor's own realign check caught the other mid-leg.

### 5.2 All 12 targets dispatched, but INCONCLUSIVE on axis 1

| leg | landing (m) | note |
| --- | --- | --- |
| S1 (S1b) | 0.145 | pre-dispatch VIO halt, retried |
| S2 | 0.098 | |
| S3 | 0.112 | |
| S4 | 0.127 | turn-around |
| S5 | 0.114 | |
| S6 | 0.117 | |
| S7 | 0.061 | |
| S8 | 0.117 | |
| S9 (S9b) | 0.172 halt → 0.133 | mid-leg VIO realign halt, retried |
| S10 | 0.117 | |
| S11 | 0.108 | |
| S12 | 0.021 | |

**11 of 12 targets on the first attempt; both retries succeeded.** 71
pulse-open samples, 0/71 unclassifiable — the classifier held cleanly even at
dusk. But the coordinator's `motion_dispatch_timings` deque hit its **500-sample
cap** partway through (first seen at S5's snapshot) and began evicting
session-3's own earliest samples, which the committed scorer's own capacity
check is built to catch (`no_history_capacity_drop: False`) ⇒
**INCONCLUSIVE (axis 1)**, per protocol, not from any data-quality defect.

🔑 **This was a foreseen and accepted risk, not a surprise.** The session-3
pre-dispatch note explicitly declined to clear history mid-day (to avoid mixing
a history-reset variable into the RF-change test) and flagged that eviction
could occur; it did, on the very next session. A future same-day multi-session
plan should clear history between sessions whenever a >~450-sample session
follows another same-day session, unless the pre-dispatch note has a specific
reason not to (as this one did).

### 5.3 Evidence

`docs/evidence-phase1-repeat-20260914-session3/`.

---

## 6. What is next (decisions, not actions)

1. **The queue-start bound question is still open.** No session today supports
   moving the 2.0 s constant; none rules a future measurement finding
   differently. Session 2 remains the only one with a computed axis 2, landing
   in the predeclared inconclusive band.
2. **A same-day multi-session plan needs history hygiene built in.** Clear
   `motion_dispatch_timings` between sessions once the prior session banked
   more than ~450 samples, unless a specific reason argues against it (as
   session 3's did, for good reason — but the cost of that choice landed anyway).
3. **The proxy-adjacency question is raised but not settled — see §6.3 below.**
4. A further repeat, if wanted, should predeclare the RF change (if any) first,
   and should not assume the anchor itself gives a strong link — session 1 found
   ranked paths of −76 to −80 dBm there today, weaker than 2026-09-13's
   measurement at the same point.

### 6.3 The proxy-adjacency question — still not resolved, and this is why

Informally, session 3 ran all 12 dispatches without a single BLE
`ble_client_not_connected` retry — no dropped-link wait at any point, unlike
session 1's repeated drops under the old 5-scanner set. **That alone does not
establish the removal helped**: session 2, run the same afternoon under the
*unchanged* 5-scanner set, also completed all 12 targets without a single BLE
retry. Session 1's drops came right after the mower was freshly parked near the
anchor; sessions 2 and 3 both started from a position already inside the
pattern. The confound (proxy removal vs. settled position vs. simple variance
across three tries) is not separated by today's data. **A repeat that holds RF
fixed and only varies start position, or vice versa, is the way to actually
separate these** — not attempted today.
