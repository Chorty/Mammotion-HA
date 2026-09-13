# FINDINGS — Phase 1 queue-start timing measurement, 2026-09-13

Session 2026-09-13, 22:07–22:43Z. Scored against
`docs/predeclared-queue-timeout-measurement-20260911.md` **exactly as committed** —
no section was amended after data existed.

Evidence: `docs/evidence-queue-timeout-measurement-20260913.json` (scoring, per-leg
and per-burst records, raw timing samples, `known_biases`, proxy inventories,
refusals, link drops) and `docs/evidence-phase1-legs-20260913.json` (every
executor result as returned).

---

## 0. Verdict: INCONCLUSIVE — the classifier failed, not the mower

- **Axis 1 fails** on *unclassifiable bursts ≤ 20 %* under every pairing reading
  (§3). No axis-2 verdict is drawn, and per §5 the numbers are not interpreted.
- 🛑 `_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS` **stays 2.0.**
- **Recorded, not interpreted:** 339 timing samples today, all `completed`, zero
  `queue_start_timeout`. `outcomes` is not a complete failure census (§1), and no
  rate claim is made below 120 pulse-opens.

---

## 1. Conditions

- **RF set frozen.** The operator disabled two newly added ESPHome proxies before
  the session. The scanner list was identical at 21:58Z and 22:44Z: `hci0` plus
  `hot-tub-backyard`, `p1s-printer`, `garage-m5stack`, `atom-fireplace`.
- **Camera cleaned** (fault 1068). The mower was moved near the anchor by a started
  and cancelled mow; blades do not run while travelling (operator).
- **Start:** `(4.79, −1.94)`, 1.88 m north of the §16.2 anchor, inside the verified
  3.0 m disc, 3.02 m corridor clearance to S1. No setup leg was needed or run.
- **Per leg:** `scripts/phase1_leg_runner.py`, explicit operator go, gate armed
  immediately before and **disarmed after every leg**. Operator visual facing
  confirmation before S1 (south) and S9b (east).
- **131 pre-session samples** (both 2026-09-12 setup legs) excluded by timestamp
  (§16.4). The history held 470/500 at session end; the oldest 2026-09-12 sample
  was still present, so nothing dropped.
- **Session end:** docked and charging 22:49:25Z. Gate verified off in the live
  API and RAW `core.config_entries`.

---

## 2. Per leg

| leg | dispatched (Z) | result | landing (m) | band | est / live dBm | pulses (calib/turn/linear) | samples: stop / motion | rule-bursts | unclassifiable pulses (B) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 22:07:18 | `vio_realign_incomplete` | 0.173 | moderate | -72 / -64 | 9 (1/1/7) | 9 / 44 | 13 | 3/9 |
| S2 | 22:10:07 | `target_reached` | 0.114 | moderate | -74 / -74 | 4 (1/0/3) | 4 / 18 | 7 | 3/4 |
| S3b | 22:17:21 | `target_reached` | 0.105 | moderate | -76 / -72 | 3 (1/0/2) | 3 / 12 | 4 | 1/3 |
| S4 | 22:18:57 | `target_reached` | 0.127 | moderate | -74 / -84 | 8 (1/4/3) | 8 / 38 | 8 | 0/8 |
| S5b | 22:21:16 | `target_reached` | 0.061 | moderate | -72 / -78 | 5 (1/0/4) | 5 / 20 | 5 | 0/5 |
| S6 | 22:22:01 | `target_reached` | 0.093 | strong | -68 / -80 | 3 (1/0/2) | 3 / 12 | 3 | 0/3 |
| S7b | 22:28:42 | `target_reached` | 0.096 | strong | -66 / -74 | 3 (1/0/2) | 3 / 11 | 4 | 1/3 |
| S8 | 22:30:15 | `target_reached` | 0.087 | strong | -66 / -72 | 4 (1/0/3) | 4 / 15 | 4 | 0/4 |
| S9 | 22:34:12 | `turn_budget_infeasible` | 1.286 | strong | -66 / -66 | 3 (1/2/0) | 3 / 14 | 4 | 1/3 |
| S9b | 22:37:02 | `target_reached` | 0.091 | strong | -66 / -76 | 8 (1/3/4) | 8 / 31 | 15 | 6/8 |
| S10 | 22:38:28 | `target_reached` | 0.086 | strong | -68 / -72 | 7 (1/3/3) | 7 / 24 | 8 | 1/7 |
| S11b | 22:41:12 | `target_reached` | 0.145 | moderate | -72 / -72 | 4 (1/1/2) | 4 / 13 | 8 | 3/4 |
| S12 | 22:42:46 | `target_reached` | 0.129 | moderate | -74 / -76 | 6 (1/1/4) | 6 / 20 | 7 | 1/6 |

"est" is the frozen siting-map estimate at the target; "live" is the mower's
self-reported `ble_rssi` in force at dispatch.

**Nothing sent** on four attempts, each retried later on operator decision under
§16.3 (the `b` labels):

- S3 (22:12:41Z) and S7 (22:23:54Z): `ble_client_not_connected`, `would_send: false`.
- S5 (22:19:54Z) and S11 (22:39:23Z): runner daylight/VIO halt, tracked-feature
  minimum 60 and 69 in the 60 s look-back.

⚠️ The retries were not in §16.2's list; they re-drove the same fixed targets. They
are recorded so a later reader can judge them. This document does not claim the
verdict would change without them.

---

## 3. Axis 1 scoring

**Stops removed first.** Every pulse records exactly one `is_stop: true` sample.
Linear and calibration stops are budget 5.0 (`emergency_stop: true`); **turn-pulse
stops are budget 2.0 with `emergency_stop: false`**, one per turn pulse. With all
`is_stop` samples removed, **every leg's motion samples equal
`len(command_results) + Σ refresh_commands_sent`** (13/13 legs), so §11.1 clause 4
(unreconcilable legs) passes.

The §10.3 rule, applied as written (gap > 500 ms starts a burst), produces **90
rule-bursts for 67 pulses**. How a rule-burst pairs with "that pulse" is not
spelled out, so both readings were scored:

| check | A: literal (burst k ↔ pulse k) | B: generous, never guesses a boundary |
| --- | --- | --- |
| pulse-opens surviving exclusion ≥ 40 | **39 FAIL** | 47 PASS |
| ≥ 4 legs, ≥ 2 with turn/calibration | 13, 13 PASS | 13, 13 PASS |
| pulse-open share 8–35 % | 27.5 % PASS | 26.0 % PASS |
| **unclassifiable bursts ≤ 20 %** | **51/90 = 56.7 % FAIL** | **43/90 = 47.8 % FAIL** (per pulse: 20/67 = 29.9 %) |
| no unreconcilable leg | PASS | PASS |
| not truncated by unrelated cause | PASS (all 12 targets, S12 completed) | PASS |
| no history capacity drop | PASS | PASS |

Reading B delimits each pulse by exact size sums of consecutive rule-bursts. A
pulse split across more than one rule-burst is `UNCLASSIFIABLE` (§11.2). 🔑 **The
unclassifiable clause fails under every reading, including the most generous one
counted per pulse**, so the verdict does not depend on the interpretation.

---

## 4. 🚨 Why the classifier failed — §10.3's premise is refuted

§10.3 justified 500 ms on the claim that *"within-burst gaps span 0–200 ms, never
more."* Today's data:

| | value |
| --- | --- |
| within-burst gaps ≤ 500 ms | n = 182, median **259 ms**, max 500 ms |
| gaps > 500 ms inside a single pulse | **n = 23**, 503–1058 ms, median 614 ms |
| of those, cutting a pulse-open from its own refreshes | **15** |
| `write_ms` of the sample starting the new piece | median 487, 308–1057 ms |

The setup leg on 2026-09-12 hit exactly one such split; today hit 23. Slow writes
stretching the sample spacing past 500 ms is **consistent with** the figures but is
not proven.

🛑 **Today's data is NOT rescored under a better rule.** Choosing the classification
method after seeing which verdict it yields is what the predeclaration forbids.
What a repeat needs is **a new predeclaration committed before new data**, for
example:

- delimit pulses by the executor's own per-pulse counts, which reconciled on all
  13 legs;
- filter on `is_stop`, not budget 5.0.

---

## 5. BLE link dropped three times — the proxies stayed up

| time (Z) | event | proxy |
| --- | --- | --- |
| 22:12:13 | `BLETransport: device Luba-VSPLV397 disconnected` + teardown warning, mower idle 1 min 44 s after S2 | `hot-tub-backyard` |
| 22:15:42 | reconnect; HA ranked **p1s-printer −77**, hot-tub-backyard −81, garage-m5stack −89 | → `p1s-printer` |
| 22:22:47 | same log pair, mower idle 30 s after S6 | `p1s-printer` |
| 22:27:21 | reconnect; **p1s-printer −75**, hot-tub-backyard −86, atom-fireplace −93 | `p1s-printer` |
| ~22:43–22:44 | not on any proxy at 22:44:48 and 22:46:34; **no disconnect line logged** | `p1s-printer` |
| 22:48:26 | connected again | `hot-tub-backyard` |

**Proxy health, checked:** no entity on `hot-tub-backyard` (11) or `p1s-printer`
(10) went `unavailable`/`unknown` between 21:55 and 22:52Z. `hot-tub-backyard`'s
uptime rose continuously (≈46 h) and its WiFi held −60 to −63 dBm. The HA log has
no disconnect or error lines for either proxy. 🔑 **The proxies did not restart or
drop; the mower's link did.**

**Leading explanation (n = 3, not proven):** the best path HA saw from this site
was **−75 to −77 dBm**, at the documented ~−76 wall. With the four frozen proxies
there is no margin. Both logged drops happened while the mower was idle between
legs; none happened mid-leg.

⚠️ `binary_sensor.*_ble_link_live` stayed `on` through the drops. Judge the link
from HA's connection allocations (`bluetooth/subscribe_connection_allocations`),
not from that sensor.

✏️ **Correction to a claim made in-session:** after the first drop I attributed it
to the weak southern edge of the pattern. The second drop happened at the anchor,
in the `strong` band, so location was not the explanation.

---

## 6. Motion stops

- **S1 `vio_realign_incomplete`, 0.173 m from target at 97 % progress, in good
  light** (sun 20°, 80 tracked features). Aim error was −22° after pulse 5
  (corrected) and −17° after pulse 6 (suppressed: projected landing 0.144 m, 6 mm
  inside tolerance). At pulse 7 it was −53°, `turn_budget_infeasible`. 🔑 This is
  the second final-approach stop, and unlike setup leg 2 it was **not dusk**, so
  darkness does not fully explain the pattern. The calibration drive measured
  facing 277.2° against the published 277.65°.
- **S9 `turn_budget_infeasible`** after ~95° of a 172° turn-around (calibration
  drive 0.24 m north first; ended facing east). S4's identical turn-around
  succeeded; S9b completed the turn and landed 0.091 m.
- Standing guidance holds: **do not raise `vio_max_realignments`.**

---

## 7. VIO dips on turns

Every tracked-feature reading below 70 fell inside a turn or heading correction:
S4's turn-around (min **60**, 22:19:06–16Z) and S10 (min **69**). Straight legs
bottomed at 71–74. `visual_positioning_status` stayed `signal_good` throughout.
Because the runner's 60 s look-back included the previous leg, the dips halted
**the next** leg's check twice (S5, S11).

**Operator hypothesis:** the camera swings between sunlit and shaded ground on a
turn and takes a moment to adjust exposure; the sun was 13–22° with long shadows.
It fits the timing. It is not separated from "rotation itself unsettles tracking",
which predicts the same pattern. **Untested; the 70-feature threshold is
unchanged.** Practical rule: wait ≥ 60 s after a turn leg before the next go.

---

## 8. RTK

`rtk_position` was `float` from 02:06:25Z to **17:05:37Z** — nearly six hours after
sunrise (11:21Z), which rules darkness out for this episode. A second brief
`float` (21:47–21:50Z) coincided with leaving the dock. It held `Fix` for every
dispatch.

---

## 9. Process

- 🚨 **The auto-disarm did not fire after S1's halt.** The shell is zsh, where
  `${PIPESTATUS[0]}` is empty, so the guard's test errored. The gate stayed armed
  about 20 s after the halt with nothing dispatched, and was disarmed by hand at
  22:08:24Z. **Fixed from S2 on:** exit code read directly, gate disarmed
  unconditionally after every leg.
- **Nothing was committed to the predeclaration after data existed.**

---

## 10. What is next (decisions, not actions)

1. **A repeat needs a new predeclaration first:** the classifier (§4), `is_stop`
   filtering, and the same §2–§5 thresholds. §16.2's targets and the runner are
   reusable.
2. **RF freeze:** repeat under the same four proxies (keeps comparability with
   legs 7 and 8), or re-enable the two new proxies first and predeclare a
   measurement of that configuration. **Operator call.**
3. The beta105 deploy hold (`b26f8909`, `683f2e52`) follows whichever freeze
   decision is taken.
