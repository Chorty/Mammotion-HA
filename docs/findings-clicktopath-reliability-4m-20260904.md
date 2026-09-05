# The 4.0 m reliability series is a determined FAIL at n = 4, was separately aborted, and produced one real operational incident (2026-09-04/05)

Predeclared in `docs/predeclared-clicktopath-reliability-4m-20260903.md`, committed
at `dba5ee5f` **before any run of this series existed**. Structured per-run data:
`docs/evidence-clicktopath-reliability-4m-20260904.json`.

Build: beta101, backend `chorty-0.8.12.post4`. Service
`raw_pymammotion_execute_vector_segment` dispatched directly — the card was not
used, per the predeclaration's trap 1 (it auto-splits above
`SPLIT_LEG_TARGET_METRES` and would have measured the splitter).

---

## 0. 🚨 Three separate things happened. Do not conflate them.

This document keeps them apart on purpose, because merging any two of them
produces a sentence that is not true.

| | what it is |
| --- | --- |
| **A. A determined FAIL** | The predeclared criterion was mathematically unreachable once run 2 failed. This is independent of whether a 5th run ever happened. |
| **B. A separate ABORT** | The series was also stopped before its targeted n = 5, for reasons that are **not** "the control law kept failing." |
| **C. An operational incident** | Real hardware moved in an unintended direction, from a heading-trust failure by the orchestrating session. Arguably the most important result of the night. |

🔑 **A is a verdict. B is a sample-size fact. C is a process failure that had
nothing to do with the scored runs.** They are reported separately below.

---

## 1. A — the FAIL is determined, not a rate

Predeclaration §6 fixes **PASS = 5 of 5** `target_reached`, and says so in terms
that were written precisely to stop the argument being had afterwards:

> ⚠️ **The criterion is deliberately strict at 4/5 rather than "≥4 of 5".** …
> If 4/5 occurs, that is a **FAIL with an informative failure**, and the response
> is a predeclared follow-up — **not** a retroactive softening of this line.

**One confirmed failure (run 2) was already in hand at n = 4.** The best outcome
any 5th dispatch could have produced was therefore *4 of 5* — which the
predeclaration names as a FAIL. 🔑 **A 5th pass could not have rescued this
series.** The verdict does not depend on the abort, and the abort does not
soften the verdict.

⚠️ **Do not quote "3 of 4" or "75%" as a reliability rate.** It is a partial n in
an aborted series whose start-geometry precondition was violated (§4). These are
per-item records. Predeclaration §4 already says that even a clean n = 5 is a
**screen, not a certification** — 5 of 5 bounds the true success rate only at
~55% with 95% confidence.

---

## 2. The four runs, per item

All four: `turn_mode: vio`, `linear_execution_mode: loop_to_tolerance`,
`motion_refresh_interval_ms: 200`, RTK **Fix** at dispatch and finish,
`pos_type` AREA_INSIDE in *Backyard Right*, `vio_state` 2 with
`tracked_features` 80.

| run | UTC | stop_reason | landing (m) | linear pulses / 22 | realign / 3 | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 22:34:02 | `target_reached` | **0.1140** | 11 | 1 | ✅ PASS |
| 2 | 22:39:06 | 🚨 `vio_realign_incomplete` | **0.1656** | 11 | 2 | ❌ **FAIL** |
| 3 | 22:58:39 | `target_reached` | **0.1310** | 11 | 1 | ✅ PASS |
| 4 | 23:03:08 | `target_reached` | **0.1354** | 10 | 2 | ✅ PASS |

Landing distribution (a **distribution**, per §4, not a rate): min 0.1140,
median 0.1332, mean 0.1365, max 0.1656 m, against the accepted
`waypoint_tolerance` of 0.15. The banked aligned 4.0 m prior is 0.1023 m — ⚠️
**but see §4 before comparing these to it.**

**The correction budget is shared.** `realignments_used` is incremented both by
the pre-linear post-turn alignment correction and by mid-drive re-aims, so the
column above counts both. Runs 1 and 4 spent one slot on a *pre-linear*
correction before the first forward pulse; run 2 spent both on mid-drive re-aims;
run 3 spent one mid-drive. Counting only mid-drive corrections would understate
usage against the budget.

### Per-pulse traces converged monotonically in all four runs

Predeclaration §5 asks for the trace specifically because "a run that converges
monotonically and one that oscillates into tolerance are different outcomes with
the same landing number." ✅ **Every run converged monotonically in
remaining-distance** — there was no oscillation into tolerance anywhere, run 2
included. Full traces are in the evidence file.

Cross-track tells a different story from remaining-distance, and it is where the
mechanism lives:

| run | peak cross-track (m) | correction fired after pulse | terminal cross-track (m) |
| --- | --- | --- | --- |
| 1 | -0.0398 | — (pre-linear only) | +0.0209 |
| 2 | **+0.3286** | 6 | **-0.1081** (overshot through the line) |
| 3 | +0.2174 | 8 | +0.1258 (under-corrected) |
| 4 | -0.3834 | 5 | +0.0221 (on target) |

🔑 **One late coarse correction from 0.2–0.4 m of accumulated cross-track
produced an overshoot, an under-correction and a good landing in three tries.**
⚠️ n = 3. **Do not fit a law to it** — it is a shape, recorded so the next series
knows where to look.

---

## 3. Criteria check against predeclaration §6, clause by clause

| clause | required | observed | |
| --- | --- | --- | --- |
| **(a)** | 5 of 5 `target_reached` | 3 of 4 dispatched; run 2 refused | ❌ **FAIL** |
| **(b)** | every landing ≤ 0.15 m | 0.1656 m on run 2 | ❌ **FAIL** |
| **(c)** | no run uses all 3 realignments | max 2 of 3 | ✅ pass |
| **(d)** | zero gate trips, zero containment breaches, stop confirmed | 13/13 gates passed on all four runs, zero blockers, zero keep-out violations against 2 zones checked, zero safety overrides applied, 0 failed commands, 0 failed stops | ✅ pass |

🚨 **(a) and (b) fail independently of one another.** Run 2 would have failed the
series on its landing alone even if it had returned `target_reached`, because
0.1656 m is 0.0156 m over the accepted tolerance. Clause (b) exists in the
predeclaration precisely so a future tolerance change cannot silently move that
bar; it did its job.

✅ **The frozen configuration held.** All **19** keys of
`docs/accepted-profile.json` were echoed identically by every **real** run's
response — not merely by the pre-dispatch dry run. No run had to be discarded
under §3's identity rule.

---

## 4. 🚨 The start-geometry precondition was NOT met on any run — and this alone voids the series

This was not visible in any landing number and was found only by checking the
per-item records against the executor's own instrumentation.

Predeclaration §3 freezes the geometry:

> **Start geometry — held constant on purpose.** The segment begins **aligned**,
> single segment, no junction turn … ⚠️ **Post-turn landing accuracy is a
> DIFFERENT property and is out of scope here.**

Every target tonight was placed 4.0 m along
`toward + calibrated_forward_heading_offset_degrees` (102.4 on the accepted
profile), and the resulting `target_reported_heading_degrees` matched the live
`toward` to ~0.0002° every time. **That check is circular** — the target was
constructed from `toward`, so it agrees with `toward` by definition. It measures
nothing.

The executor's own VIO calibration drive measured the direction the mower
*actually* travels when told to go forward:

| run | `toward` at dispatch | facing from `toward + 102.4` | facing from mirror `90.13 - toward` | **measured by the calibration drive** | additive-offset error | mirror error |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | -175.157° | 287.243° | 265.287° | **260.863°** | **+26.38°** | +4.42° |
| 2 | 160.553° | 262.953° | 289.577° | **289.839°** | **-26.89°** | -0.26° |
| 3 | -67.001° | 35.399° | 157.131° | **157.220°** | **-121.82°** | -0.09° |
| 4 | 63.575° | 165.975° | 26.555° | **31.419°** | **+134.56°** | -4.86° |

The rotation was real, not a modelling artefact — the executor opened runs 3 and
4 with **staged** turns whose recorded initial errors were **122.88°** and
**135.553°**, and `toward` physically moved **-20.9 / +22.8 / +120.7 / -138.1°**
across the four opening turns.

🔑 **So runs 3 and 4 are post-turn legs following a ~120–135° opening turn.** That
is the property §3 explicitly puts out of scope, and it is exactly the
thin-evidence area standing decision 7 warns about. ⚠️ **These four landings are
NOT comparable to the banked aligned 4/5/6 m reach runs and must not be pooled
with them.**

### The mirror is right and the additive offset is wrong, measured on 43 real pulses

For every linear pulse, the movement heading the executor measured itself
(`movement_vector_heading_degrees`, from the settled displacement) was compared
against both models evaluated on the `toward` reading live at the **start** of
that pulse:

| | n | mean abs err | median | max |
| --- | --- | --- | --- | --- |
| **compass mirror `90.13 - toward`**, `toward` fresh | 30 | **1.000°** | 0.945° | 3.003° |
| additive offset `toward + 102.4`, `toward` fresh | 30 | 87.478° | 118.436° | 166.825° |
| mirror, `toward` stale (post-rotation) | 13 | 27.547° | 11.604° | 135.099° |

⚠️ **Every one of the 13 "stale" pulses is either the first pulse of a run or the
pulse immediately after a realignment turn.** `toward` lags a rotation by one
full pulse cycle — ~5 s at the observed 2.9–6.8 s send gaps. That is the
documented "only valid when `toward` is fresh" caveat, now **quantified** rather
than asserted.

🔑 **Scope limit, stated so nobody over-reads this.** The additive offset did not
*steer* these runs. In `turn_mode: vio` the executor closes on
`target_vision_heading` derived from its own calibration drive;
`target_reported_heading_degrees` is echoed but only steers the `legacy` and
`night` turn paths. The executor rescued every run. **The damage was done
upstream, where the orchestrating session used the offset to choose where to put
the target** — and it is the same mistake that produced the incident in §6.3.

🛑 **This authorizes no change.** Predeclaration §10: a FAIL authorizes nothing
beyond writing up the failure mode. Any change to
`calibrated_forward_heading_offset_degrees` or to the accepted profile needs its
own predeclaration and its own Gate 5.

---

## 5. Run 2 — the failure, and why it is not the historical prior

Predeclaration §2 records the one prior failure at this length (`…001116Z`,
`vio_realign_incomplete` at 0.5493 m) as **BLE-caused**, and §2 makes BLE "the
first thing to look at on a failure." It has been looked at, and it is excluded.

✅ **BLE was healthy end to end on run 2.** `queue_settle` live, connected,
usable, `queue_depth: 0`, no cooldown; all 14 commands `ok`; all 12 stops `ok`
(median ack 152 ms); **56 of 56** refresh writes completed; zero
`refresh_cadence_broken` pulses; zero `position_feed_stale` pulses. `ble_rssi`
read -70 dBm at dispatch and finish — at the documented boundary, and per
CLAUDE.md it is self-reported and does not predict cadence, so it is recorded and
not read as a cause. **Every deliverable command was delivered.**

⚠️ **The correction budget was not exhausted either.** 2 of 3 used, and the stop
reason is `vio_realign_incomplete`, not `vio_realign_budget_exhausted` — a slot
was still free. A *single* required correction of **57.987°** was refused as
`turn_budget_infeasible`.

**What actually happened, in order:**

1. Cross-track grew monotonically to **+0.3286 m** by pulse 6 with no correction.
2. The first mid-drive re-aim fired after pulse 6 at -19.199° and returned
   `target_heading_reached`. It worked.
3. That single correction was coarse enough to carry the mower **through** the
   line: cross-track ran +0.3286 → +0.0379 → -0.0642 → **-0.1081 m**.
4. 🔑 **The pivotal moment is the pulse-10 suppression.** At 0.2747 m remaining
   and 30.705° of aim error, the guard recorded
   `already_lands_inside_tolerance` with a projected landing of **0.1455 m**
   against the 0.15 m tolerance — a margin of **4.5 mm**. The projection model's
   own docstring warns that its margins have been as thin as 1.1 mm and that a
   projection is "which side of the tolerance is this on", **not** a landing
   prediction.
5. The next pulse ran 0.1484 m (0.54× the remaining distance, inside the
   documented 0.30–1.16× spread) and landed at **0.1656 m** — 0.0201 m worse than
   projected, and 0.0156 m outside tolerance.
6. With 0.1656 m remaining, the bearing to target had swung to 303.719° against a
   facing of 245.906°. 🔑 **That 57.8° is near-field bearing swing, not the mower
   turning 58° off course** — at 0.17 m range a 0.11 m cross-track miss *is* a
   58° bearing error. The correction was geometrically infeasible within the turn
   budget, and the executor refused rather than flail. It stopped safely.

**Classification: a genuine control-law/geometry miss, not a transport failure.**
Per §6, a run that stops safely on a named refusal is a **FAIL**, not a smaller
number.

⚠️ For contrast, run 3's suppression at pulse 10 had a **28 mm** margin
(projected 0.1219, tolerance 0.15) and landed at 0.1310 — inside. In both cases
the projection **under-predicted** the landing, by 0.0201 m and 0.0091 m. ⚠️
**n = 2. Do not fit a bias correction to it**; record it as the direction to
check next.

---

## 6. B and C — the abort, and the five-step incident sequence

After run 4, **five consecutive attempts to establish a safe start
position/heading for run 5 failed, each for a different reason, escalating in
severity.** None of them was "the control law failed again." Reported as an
aborted series per §7: 🔑 **partial n is a result, not a draft.**

### 6.1 A map/corridor near-miss, caught before dispatch

The first candidate start put the polygon boundary — near what proved to be the
dock's own exclusion notch — within **0.29 m** of the 4.0 m corridor plus its
1.0 m overshoot margin. Caught by the pre-dispatch map scan; nothing dispatched.
A "turn right 30°" alternative was scanned and found **worse**: it exits the
mapped area partway through. Also caught before dispatch; nothing sent.

### 6.2 🚨 A confirmed map inaccuracy, caught by physical measurement, not software

At a later candidate the map-based corridor scan showed **3.5 m** of clearance in
the computed direction. The operator directly measured the actual distance to a
real fence in that same direction: **110 inches = 2.79 m** — **0.71 m less than
the map**, and less than the 4.0 m segment itself needs. Nothing was dispatched.

🔑 **State this plainly: a physical operator measurement caught a real hazard
that both software checks would have missed.** The software containment gate
would have shown ample margin, and so would the map-polygon corridor scan that
CLAUDE.md prescribes *as the gate's backstop*. The existing trap says the
containment gate does not check the map. **This adds that the map does not check
the ground.** The polygon is not ground truth, and a fresh scan against it is
necessary but not sufficient.

### 6.3 🚨 THE INCIDENT — a real dispatch in an unintended direction

The operator repositioned and turned the mower **by hand**. With no intervening
real motion, the orchestrating session computed a 0.5 m "test" move from the
then-current `toward` of **91.8054°** using
`target_map_heading = toward + calibrated_forward_heading_offset_degrees`
(102.4), giving a commanded map bearing of **194.205°**. This is the same formula
the production dispatch code echoes in `heading_calibration.formula`, and it had
been cross-checked earlier in the session against the app's own directional arrow
and the operator's dock-relative reasoning to within ~4.6°.

The move dispatched, returned `stop_reason: target_reached`, and looked
successful in telemetry. **The operator then reported directly and unambiguously
that the mower had driven in the direction it was facing BEFORE their manual
reposition, not the direction they had just turned it to.**

Immediately after that move, both device heading sources jumped by roughly a
half-turn:

| | before | after | jump |
| --- | --- | --- | --- |
| raw `toward` | 91.8054° | -101.8713° | ~166.3° |
| raw `vio_heading` (`report_data.vision_info.heading`, surfaced by the `vio_heading` sensor description) | -1.754° | -168.570° | ~166.8° |

and `current_orientation` reported `trustworthy: true` at **0.571°** of
agreement — which reproduces exactly from the post-move raw values (mirror
`90.13 - (-101.8713)` = 192.001, VIO `(-168.570) % 360` = 191.430).

**Most likely mechanism, consistent with everything above:** the device's own
heading estimate — *both* the compass-like `toward` field and VIO — does not
immediately reflect a manual physical reposition. It takes real subsequent motion
to re-anchor what is plausibly a genuine ~180°-scale ambiguity after the machine
is picked up and turned by hand.

⚠️ **One number in this record does not reconcile, and it must be re-derived
before anyone builds on it.** The session recorded the *pre*-move state as
`trustworthy: false` / `heading_sources_disagree` with a ~178° gap. **The quoted
pre-move raw values do not produce that.** Mirror `90.13 - 91.8054` = 358.325 and
VIO `(-1.754) % 360` = 358.246 differ by **0.079°** — far inside the 15°
agreement tolerance, which would have published `trustworthy: true`. Either the
178° figure or the two raw values were read at a different instant. 🚨 **The
second branch is the more alarming one**: it would mean both nominally
independent heading sources were **stale together**, corroborating each other to
0.079° while both described the pre-reposition facing — i.e. that
`current_orientation.trustworthy` is corroboration between two sources, **not
evidence of freshness**, and was actively reassuring at the moment it was most
wrong. Re-derive this from the session transcript.

**Classification: a telemetry-trust and process failure by the orchestrating
session.** ⚠️ **Not a control-law problem.** The mower did exactly what it was
told with the (wrong) input it was given, and the executor's own gates found
nothing to object to because nothing about the command was internally invalid.

**The documented risk that was not applied.** The standing guidance is not merely
"re-derive facing" — it is specific: derive the facing **two ways** before any
armed run (the last driven leg's travel bearing, and `(90.13 - toward)` with
`toward` fresh), require them to agree, **and on disagreement trust the mirror**;
then state the destination in compass terms and have the operator confirm the
ground. That was skipped **because a 0.5 m "test" move made right after a manual
reposition was treated as lower-stakes than an armed predeclared run.** It was
not. §4's measurement says the formula that was used instead is wrong by a mean
of 87° whenever `toward` is fresh, and by ~164° at this particular `toward`.

### 6.4 A second physical-measurement catch, immediately after

The operator measured the actual fence distance in the new (post-move, still not
trusted) facing: **1.83 m** — again well short of the 4.0 m the segment needs. A
subsequent "180° turn" candidate was map-scanned and found to exit the mapped
area at ~2.0 m of the needed 4.0 m, leaving **0.503 m** of clearance. Neither was
dispatched.

### 6.5 BLE instability, then loss of daylight — the system's own gates closed the door

A turn-only diagnostic (`raw_pymammotion_turn_to_heading` — a bounded in-place
rotation, dry-run-verified clean beforehand) was **refused twice** by the live
gate with `ble_client_not_connected`, corroborated by `ble_link_live` reading
`off` at **-74 dBm** — squarely in the documented failure band for this hardware
(works above ~-70, dies below ~-76). On a third attempt BLE reported live
(-66 dBm, `ble_link_live` on, `real_motion_allowed: true`) and the command was
accepted, but the HTTP call to Home Assistant **timed out after 60 s**. No active
motion session was found immediately afterwards — the device-side command
sequence had completed, not hung — but the very next telemetry read showed
`current_orientation.reason: vio_feed_degraded`, `vio_feed_live: false`, and
**`vio_tracked_features: 0`, down from 80** earlier in the session.

🔑 **That is this project's documented nightfall signature, not a sensor glitch**
— tracked features are a cliff, not a gauge; 80 means saturated/enough light and
the count falls fast when the light goes. Standing decision 4 (night is CLOSED)
and the predeclaration's own "daylight throughout" requirement both bind here.

**How the last dispatch came to be attempted, recorded as it happened.** The
orchestrating session had recommended stopping and stated the risk plainly —
deteriorating conditions on top of the §6.3 incident. The operator made the call
to try one bounded, already-dry-run-verified, low-risk command anyway: a
turn-in-place, not a drive. 🔑 **No gate was overridden.** The production
system's own safety gates — first BLE, then implicitly the VIO/darkness condition
discovered right after — are what actually stopped further motion. This is the
operator owning a safety trade-off with the risk stated once, and the machine
holding the line underneath them.

**The session ended here.** Not by a decision to stop, but with conditions
closing the door: real darkness on top of an already-unstable link.

✅ **Gate disarmed and verified `False` from the live API AND RAW
`core.config_entries` at the end of the session**, per §8 and CLAUDE.md's
motion-gate section.

---

## 7. What this authorizes

**Nothing beyond this write-up.** Predeclaration §10 is explicit: a FAIL
authorizes nothing but recording the failure mode.

🛑 In particular, **nothing here reopens anything**:

- **Phase 2 continuous steering stays CLOSED** (standing decision 5).
- **Accuracy stays CLOSED** (standing decision 3). The 0.065 m sensing floor is
  not implicated in any of this.
- **Night stays CLOSED** (standing decision 4). §6.5 is a reason the session
  ended, not a case for running in the dark.
- **No bound, tolerance, budget or profile key changes.** Not
  `vio_max_realignments`, not `waypoint_tolerance`, not
  `calibrated_forward_heading_offset_degrees` — the §4 measurement is an
  observation, and changing the accepted profile needs its own predeclaration and
  its own Gate 5.

---

## 8. Recommended follow-ups

### 8.1 🔴 Add a dated trap entry to CLAUDE.md — "Traps that keep biting"

This is the highest-value output of the night. **Suggested wording, verbatim:**

> ⚠️ **A REPOSITIONED MOWER'S HEADING TELEMETRY IS STALE UNTIL IT DRIVES —
> INCLUDING "SMALL TEST" MOVES.** On 2026-09-04 a 0.5 m move was dispatched
> immediately after the operator turned the mower **by hand**, aimed from the
> then-current `toward` via `toward + calibrated_forward_heading_offset_degrees`.
> It reported `target_reached` and the operator saw it drive in the mower's
> **pre-reposition** direction. Both `toward` and `vio_heading` then jumped ~166°
> on that first real motion — the device's own estimate had not re-anchored.
> 🔑 **`current_orientation` publishes `trustworthy` on corroboration between two
> sources, NOT on freshness: both can be stale together.** And the additive
> offset is not the model — on 30 fresh-`toward` pulses that night the mirror
> `90.13 - toward` predicted the driven direction to a mean 1.000°, while
> `toward + 102.4` was off by a mean 87°.
> ✅ **Before ANY armed dispatch, derive facing two ways — the last driven leg's
> bearing and `(90.13 - toward)` with `toward` fresh — require agreement, and
> state the destination in compass terms for the operator.** A short "test" move
> is an armed dispatch. This one was treated as lower-stakes and was not.

Consider also extending the existing containment-gate trap with one line from
§6.2: 🚨 **the gate does not check the map, and the map does not check the
ground** — a map-polygon corridor scan showed 3.5 m where the operator measured
2.79 m of real fence clearance.

### 8.2 Instrumentation gaps found while writing this up

- ⚠️ **The mid-drive realignment record omits `turn_feasibility`.** The
  pre-linear post-turn record carries it — deliberately, so "a refusal must carry
  its own arithmetic." Run 2's decisive `turn_budget_infeasible` therefore has to
  be re-derived by hand. The same argument that added it to one site applies to
  the other.
- ⚠️ **Battery % is not in the executor response.** Predeclaration §5 item 7 asks
  for it; it was not captured separately either. Recorded as a gap, not a value.
  The §7 abort rule gates on battery below 35% off-dock, so this is a covariate a
  future series needs a way to record.

### 8.3 If the series is repeated

The predeclaration's own answer to a 4/5 outcome is "a predeclared follow-up."
Anything that follows needs a **new** predeclaration, and — on §4's evidence — it
must fix how the aligned start is established before it is worth dispatching,
because tonight's runs 3 and 4 measured a property the series had explicitly
declared out of scope.
