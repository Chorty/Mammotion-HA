# PREDECLARED — click-to-path reliability series at 4.0 m (2026-09-03)

**Written before any run of this series exists.** Per
`docs/NEXT-PLAN-20260903.md` §3, and serving **standing decision 4** — *"the goal
is consistency, not precision: click-to-go reliable enough to trust without
watching."*

🛑 **THIS AUTHORIZES NO RUN.** It fixes the configuration, the statistic, the
pass criterion and the abort rule **in advance**, so that the series measures
reliability rather than being scored after the fact. Every dispatch still needs
its own explicit operator go/no-go immediately before it.

---

## 1. Why this and not more step-response

The step-response / 2a line feeds **τ → dead time → Phase 2 continuous
steering**, which is **parked** (standing decision 5) because it buys ~4× speed
and **not capability**. Stop-measure-go already does click-to-path.

**Reach is CLOSED at 6.0 m and landings are flat with distance** —
0.1023 / 0.1015 / 0.1144 m at 4 / 5 / 6 m. 🔑 **Feasibility is proven.
RELIABILITY is not, and reliability is what "trust it without watching" means.**

---

## 2. 🗑️ The prior is worse than the plan stated — corrected here before dispatch

`docs/NEXT-PLAN-20260903.md` §3 records **4.0 m as n = 1 (0.1023 m)**. That is
wrong, and the omission matters because the missing run is a **failure**.

From `docs/loop-to-tolerance-reach-20260811.md` §1, both on the same night:

| run | leg | pulses | landing | stop |
| --- | --- | --- | --- | --- |
| `…001116Z` seg1 | 4.000 m | 9 of 16 | **0.5493 m** | `vio_realign_incomplete` (BLE) |
| `…002804Z` seg1 | 4.000 m | 11 of 16 | 0.1023 m | `target_reached` |

🔑 **The real prior at 4.0 m is 1 reached / 1 failed, n = 2** — and the observed
failure mode is a **BLE-caused mid-drive realignment abort**, not a control-law
miss. It stopped safely at 0.5493 m out.

⚠️ **Two consequences, both registered now rather than discovered later.**
1. A single failure in this series is **not** automatically a new finding — the
   failure mode is already on the record at this length. What the series
   measures is its **rate**.
2. **BLE health is a covariate, not a nuisance.** It is recorded per run (§5)
   and it is the first thing to look at on a failure. ⚠️ But `ble_rssi` is
   self-reported and **does not predict cadence** (within-run median r = +0.042
   over 24 runs) — so a marginal RSSI is neither a reason to postpone a run nor
   to trust one. Record it; do not gate on it.

---

## 3. Configuration — FROZEN, and the two traps that would void the series

**Service:** `raw_pymammotion_execute_vector_segment`, one segment, **4.0 m**.

🚨 **TRAP 1 — the CARD CANNOT RUN THIS.** It auto-splits any leg over
`SPLIT_LEG_TARGET_METRES` (3.85 m), so a 4.0 m click becomes two sub-legs and
**measures the splitter**. Dispatch the service directly, where
`split_leg_target_length_m` defaults to off.

🚨 **TRAP 2 — the SCHEMA DEFAULTS ARE NOT THE ACCEPTED PROFILE.**
`max_linear_pulse_ceiling` defaults to `None` (accepted **22**) — without it the
run is fixed-budget and stops after ~1 pulse; `waypoint_tolerance` defaults
0.08 (accepted **0.15**); and `calibrated_forward_heading_offset_degrees`
defaults to **116.5** where the profile is **102.4**.

✅ **Send `docs/accepted-profile.json` verbatim and verify identity
key-by-key against the response before the run counts.** A run whose echoed
profile differs on any key is **discarded, not scored** — it is not a sample of
this population.

**Start geometry — held constant on purpose.** The segment begins **aligned**,
single segment, no junction turn, exactly as the banked 4 / 5 / 6 m reach runs
did, so results are directly comparable to the 0.1023 m prior.
⚠️ **Post-turn landing accuracy is a DIFFERENT property and is out of scope
here.** Do not mix a post-turn leg into this series; it is a separate series
with its own predeclaration.

**Daylight throughout.** Turns close on VIO.

---

## 4. Target n, and what n = 5 does and does not buy

**Target n ≥ 5 at 4.0 m before any other length is added.**

⚠️ **State the statistics honestly up front, so nobody over-reads the result.**
5 of 5 successes gives a 95% confidence lower bound on the true success rate of
only about **55%** (rule of three: the upper bound on the failure rate is ~3/n =
60%). **n = 5 cannot demonstrate "reliable enough to trust unwatched."** What it
can do is:
- detect a failure rate that is *high* (≳40%) with good probability, and
- produce a landing **distribution** rather than a point, which is what sizing
  any future tolerance argument requires.

🔑 **So the honest framing is: n = 5 is a SCREEN, not a certification.** If it
passes, the next question is more n, not a stronger claim. Say so in the
evidence file.

---

## 5. What is recorded, per run — decided before any data exists

Primary:
1. `stop_reason` — `target_reached` or the named refusal.
2. **Landing distance** (m) from the frozen target.
3. **Linear pulses used**, against the 22 ceiling.
4. **Mid-drive realignments used**, against `vio_max_realignments: 3`.
5. Terminal heading error and cross-track at finish.

Covariates, recorded but **not** gated on:
6. `ble_rssi` at dispatch; refresh writes sent vs completed; max write gap.
7. Battery %, RTK fix state, `tracked_features`.
8. Wall-clock duration.

Safety, every run: gate blockers at dispatch, containment, stop confirmed, and
the gate **disarmed and verified afterwards from the live API AND RAW
`core.config_entries`**.

🔑 **Record the per-pulse remaining-distance trace**, as
`docs/loop-to-tolerance-reach-20260811.md` did. A run that converges
monotonically and one that oscillates into tolerance are different outcomes with
the same landing number, and only the trace separates them.

---

## 6. The pass criterion — FIXED NOW

**PASS** requires **all** of:
- **(a) 5 of 5** dispatched runs return `stop_reason: target_reached`;
- **(b)** every landing ≤ the accepted `waypoint_tolerance` of **0.15 m** (this
  is implied by (a) and is stated separately so a future tolerance change cannot
  silently move the bar);
- **(c)** no run uses all 3 mid-drive realignments — i.e. the correction budget
  retains margin at this length, the constraint that was binding at 6.0 m;
- **(d)** zero safety-gate trips, zero containment breaches, stop confirmed on
  every run.

**Anything else is a FAIL of the series.** In particular:
🚨 **A run that stops safely on a named refusal is a FAIL, not a smaller
number.** `vio_realign_incomplete` at 0.5493 m is the already-observed failure
at this length; recording it as "stopped safely" and moving on is exactly how a
reliability series becomes a feasibility demo.

⚠️ **The criterion is deliberately strict at 4/5 rather than "≥4 of 5".** With
the prior at 1/2, a criterion that tolerates one failure cannot distinguish the
status quo from an improvement. If 4/5 occurs, that is a **FAIL with an
informative failure**, and the response is a predeclared follow-up — **not** a
retroactive softening of this line.

---

## 7. The abort rule — FIXED NOW

Stop the series immediately and write it up, without dispatching the remainder,
if any of:
- **any** containment breach, or any run leaving the frozen corridor;
- **two consecutive** runs failing to reach target;
- any run requiring reverse recovery, or ending outside the corridor;
- battery below **35%** off-dock at the start of a run (the 2026-08-24 incident
  began at 29% off-dock);
- BLE `ble_link_live` off, or a `queue_settle` that is not live at depth 0;
- the operator withholding go/no-go for any reason.

🔑 **An aborted series is reported as an aborted series.** Partial n is a
result, not a draft.

---

## 8. Safety preconditions — every run, no exceptions

- 🚨 **Scan the corridor against the MAP every time.** `step_path_contained` and
  its siblings measure clearance against the **operator-supplied polygon**, not
  the mowing area. On 2026-09-03 a position with 2.8447 m of real yard clearance
  passed 15/15 gates against a 3.20 m requirement because the corridor was
  centred on the mower by construction. **The gate is not a substitute for the
  scan.**
- A 4.0 m segment needs clearance for the leg **plus** overshoot in every
  direction; size from a fresh scan, not from a remembered corridor.
- Daylight, operator present, emergency stop accessible.
- Explicit per-run go/no-go **immediately before dispatch** — not once for the
  series.
- Gate disarmed and verified from live API **and** RAW after every run.
- Dock and charge between runs as needed; do not leave the mower off-dock at low
  battery.

---

## 9. What a PASS authorizes — and what it does not

A PASS authorizes **one thing**: extending the same series to larger n at the
same 4.0 m, or opening a **separately predeclared** series at another length or
with a junction turn.

🛑 It does **not** authorize resuming Phase 2 (standing decision 5), removing any
per-run confirmation, raising any exposure bound, or describing click-to-path as
"trustworthy unwatched" — see §4 for why n = 5 cannot support that sentence.

A FAIL authorizes nothing beyond writing up the failure mode.
