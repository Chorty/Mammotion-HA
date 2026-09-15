# FINDINGS — Phase 1 repeat, session 4, 2026-09-15 (post-firmware-fix retest)

Follow-on to `docs/findings-ble-proxy-adjacency-test-20260915.md`. Tests real
click-to-path motion — not passive dock observation — at the two targets that
failed from BLE drops on 2026-09-14 (session 1), and then a full fresh
S1–S12 pass, both under the new ESPHome firmware (Bluetooth Scanning switch +
LUX/power-sensor `update_interval` 5s→60s on both `hot-tub-backyard` and
`p1s-printer`) and the full restored five-scanner set.

⚠️ **Run at dusk, on operator instruction, sun elevation disregarded as a
rule for these runs.** The operator explicitly said to disregard the
sun-elevation floor going forward for this session, after the reason for it
(the 2026-09-12 VIO-collapse incident) was restated. Per that instruction,
`--allow-low-sun` was used throughout. **`vio_tracked_features` and
`visual_positioning_status` were kept fully enforced regardless** — those,
not the sun angle, are what actually caught the 2026-09-12 incident, and
the operator did not ask to relax them. No VIO halt fired during either the
spot checks or the full run.

---

## 0. Verdict

**Both previously-failed legs passed clean, and a full fresh 12-leg run
completed on the first try with zero retries, zero drops, and the best
axis-2 reading of the whole investigation.**

- **Spot check:** S6 and S7 (the two BLE-drop failures from 2026-09-14
  session 1) both `target_reached`, 0.093 m and 0.099 m, zero BLE issues.
- **Full session (session 4):** all 12 targets, first attempt, zero halts.
  Landings 0.059–0.148 m, all under the 0.15 m tolerance. Axis 1 fully
  PASSES; axis 2 verdict is `4_inconclusive` but `q_p95` = **254.2 ms**, only
  4 ms over the 250 ms "clearly fine" bar — the closest any session has come.
- 🛑 `_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS` **still stays 2.0.** This
  session gives no formal grounds to move it (the threshold is fixed and
  wasn't cleared), but it is the strongest evidence yet that the constant has
  real margin under normal conditions.
- **This is one clean session, not proof the firmware fix worked.** The same
  caveat as Window E in the proxy-adjacency test applies: a single good run
  doesn't distinguish "the fix helped" from "today was just a good day,"
  especially given RTK, BLE, and turn-budget hiccups earlier in the same
  session (see §2) that had nothing to do with the firmware change.

---

## 1. Spot check: the two BLE-drop failures, retested

| leg | target | 2026-09-14 result | 2026-09-15 result |
| --- | --- | --- | --- |
| S6 | (4.94, −3.82) | mid-pulse GATT send failure (`command_failed`) | `target_reached`, 0.093 m, 0 drops |
| S7 | (4.94, −2.82) | never sent (`ble_client_not_connected`) | `target_reached`, 0.099 m, 0 drops |

Both ran under real motion (200 ms refresh cadence — far more BLE traffic
than the passive dock tests), on the new firmware, with the full five-scanner
set restored. Evidence: `docs/evidence-phase1-repeat-20260915-session4/log_S6_spotcheck.txt`,
`log_S7_spotcheck.txt`.

---

## 2. Two real halts before the full run started — neither related to BLE or the firmware

Both occurred and were resolved before session 4's first dispatch; recorded
here because they happened in the same continuous testing window and are
worth separating from the firmware question.

1. **RTK dropped to `Float` immediately after an integration reload**
   (used to clear the timing-history deque before the full run). Went
   `unavailable` at 23:10:03Z — the same second as the reload — then `float`
   at 23:10:36Z, and self-recovered to `Fix` at 23:16:01Z, about 5.5 minutes
   later. The runner correctly halted the first S1 attempt on this
   (`RTK not Fix: Float`) rather than proceeding. 🔑 **A config-entry reload
   appears to cost several minutes of degraded RTK while the coordinator
   re-establishes state** — worth remembering before reloading immediately
   before a real run in the future; better to reload with a buffer of idle
   time first.
2. **`turn_budget_infeasible` on the first S1 attempt after the RTK recovery**,
   landing 2.203 m short on a 1.911 m leg that needed a large turn (the mower
   had just been repositioned by the operator, so its facing didn't match the
   direction to target). Retried as S1b; the retry attempt was itself
   interrupted by an explicit operator stop (unrelated — see §3), and the
   sequence was relaunched fresh once the mower was repositioned near the
   anchor. This is the same turn-translation-cap mechanism documented in
   `docs/findings-phase1-repeat-20260914.md` §3.1 and §5.1 — not a new defect.

---

## 3. An in-flight stop, exercised and clean

The operator stopped the running sequence mid-session (`TaskStop`) to
reposition the mower. The interrupted run had only reached "record scanners"
— it had not yet armed the gate for the retry — so no motion was in flight
and no gate state needed correcting. **Verified anyway, live and raw:** gate
was already off in both. This is the first time this session's automation
stack was interrupted externally rather than exiting on its own, and it
behaved safely by construction (the gate is armed only immediately before
`phase1_leg_runner.py` dispatches, never held open across the wait loop).

Separately, the gate was found armed once between the interruption and the
final run — the operator had armed it themselves via the app/card to
reposition the mower (the same ordinary explanation recorded multiple times
in this project's history). Disarmed on confirmation, verified off live and
raw (accounting for the documented ~15 s `.storage` write lag) before the
final run started.

---

## 4. Session 4: full run, in detail

| leg | landing (m) | note |
| --- | --- | --- |
| S1 | 0.059 | |
| S2 | 0.141 | |
| S3 | 0.148 | |
| S4 | 0.111 | turn-around |
| S5 | 0.134 | |
| S6 | 0.114 | |
| S7 | 0.127 | |
| S8 | 0.075 | |
| S9 | 0.120 | |
| S10 | 0.131 | |
| S11 | 0.118 | |
| S12 | 0.087 | |

All 12 on the first attempt. No retries, no halts, no unreconcilable leg, no
dry-run mismatch. Dispatched 23:25:49Z–23:40:30Z, entirely after sunset
(confirmed by the operator override) — VIO never dipped enough to halt.

### 4.1 Axis 2, in full

```
n_pulse_open_budget_2 = 66
q_p95_queue_wait_ms_pulse_open = 254.189
w_p95_write_ms_classified_completed = 333.243
write_inheritance_ratio = 0.7628
worst_wait_fraction_recomputed = 0.1606
legs_at_or_above_raise_fraction (>= 0.75) = 0
queue_start_timeouts_in_session = 0
rate_statement: "no rate claim below 120; the honest bound is 3/66"
verdict: 4_inconclusive
```

Per-leg worst-wait fraction ranged 0.046 (S2) to 0.161 (S5) — none close to
the 0.75 raise fraction. `q_p95` at 254.2 ms is the lowest of any session
scored so far (2026-09-14 session 2: 289.9 ms), narrowly missing the 250 ms
"fine" threshold. 🔑 **The threshold is not moved retroactively** — this
stays `4_inconclusive` by the letter of the predeclared rule, same as every
other session, even though it is the closest result yet to a clean "fine"
verdict.

Evidence: `docs/evidence-phase1-repeat-20260915-session4/`.

---

## 5. What is next (decisions, not actions)

1. **Three sessions have now scored axis 2 in the inconclusive band**
   (2026-09-14 session 2: 289.9 ms; today: 254.2 ms), trending toward "fine"
   but never crossing it. A fourth clean session, or a larger pooled `n`
   under a pre-registered pooling rule, would be needed to either cross the
   250 ms line for real or settle that it won't.
2. **The firmware fix (WiFi-chatter reduction) still has no controlled
   evidence behind it** — today's clean run and Window E from the
   proxy-adjacency test are both single clean sessions on the new firmware,
   and neither rules out "today was just good" the way Window C already
   showed the old firmware could also go clean for 45 minutes.
3. **Reloading the integration immediately before a real-motion session
   costs real RTK recovery time (~5.5 min this session)** — build in a buffer
   before dispatching after any reload, not just an immediate blade/state
   check.
4. ✅ **The `not_charging` flag from earlier the same day resolved on its
   own** — post-dock check reads `charging: on`, 52%. No action needed.
