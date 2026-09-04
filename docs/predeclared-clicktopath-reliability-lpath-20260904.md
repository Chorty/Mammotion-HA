# PREDECLARED — click-to-path reliability series, two-segment L-path (2026-09-04)

**Written before any run of this series exists.** Companion to
`docs/predeclared-clicktopath-reliability-4m-20260903.md`, which covers the
*aligned-start, single-segment* half. This one covers the half that real
click-to-path actually does: **a leg that does not start fresh.**

Serving **standing decision 2** — *"the goal is consistency, not precision:
click-to-go reliable enough to trust without watching."*

🛑 **THIS AUTHORIZES NO RUN.** It fixes the geometry, the configuration, the
statistic, the pass criterion, the falsifier and the abort rule **in advance**.
Every dispatch still needs its own explicit operator go/no-go immediately before
it.

🛑 **AND IT MUST NOT BE DISPATCHED BEFORE THE 4.0 m SERIES COMPLETES WITH A
RESULT.** `docs/predeclared-clicktopath-reliability-4m-20260903.md` is this
series' **same-build baseline**. Running this one first leaves the segment-2
numbers with nothing on the same build to sit against, and the comparison in §9
becomes unanswerable. Baseline first, then this.

---

## 1. Why this series, and not more of the 4.0 m one

The 4.0 m series measures a segment that **begins aligned**, single segment, no
junction — the same shape as every banked 4 / 5 / 6 m reach run. That is the
easier half, and it is the half the reach work already demonstrated
(0.1023 / 0.1015 / 0.1144 m at 4 / 5 / 6 m).

**Real click-to-path is a path.** Segment 2 onward inherits wherever segment 1
stopped, plus a turn. CLAUDE.md says this in its own words under *Reach and
landing*: ⚠️ *"Reach ≠ post-turn landing accuracy. Those runs began aligned,
single segment. A leg following a junction turn is a different property, and the
thin evidence there is where the real risk sits."*

Three standing decisions bound what this series is allowed to be:

- **Standing decision 5 — Phase 2 continuous steering is CLOSED**, not parked.
  This series is **stop-measure-go only**. Nothing in it measures τ, dead time or
  loop bandwidth; nothing in it may be written up as evidence for or against
  continuous steering. If a result here looks like it argues for Phase 2, that is
  a misreading of the series, not a reopening.
- **Standing decision 7 — reliability statistics use the beta57+ epoch ONLY.**
  See §2: the pooled figures that motivated this work are excluded from its
  justification.
- **Standing decision 2** is what a PASS would serve, and only in the narrow
  sense §10 allows.

🔑 **There is also a specific, named thing this geometry tests that nothing else
has.** `docs/gate5-beta57-PASSED-20260818.md` closes with: *"is
`vio_max_realignments: 3` enough is still unanswered — no mid-drive correction
has ever fired on the new code."* beta57's mid-drive trigger
(`_mid_drive_realign_decision`, `custom_components/mammotion/services.py:17385`)
decides on `projected_landing_m` — the miss **at the end of the next pulse**,
not at the closest approach — which is precisely the term whose absence caused
the one on-record failure at this geometry (§2). This series is the first
hardware exposure of that fix on the geometry that exposed the bug.

---

## 2. The real prior — per-item records only

### 🗑️ What is deliberately EXCLUDED from the justification

CLAUDE.md:99-103 and `docs/NEXT-PROMPT-20260904.md:37-38,82,85-86` assert a
pooled shape over 81 banked segments: *89% under 1.0 m, 43% at 3.0–3.9 m, **44%
for segment 2+ at ≥ 1.5 m against 73% for segment 1***. That 44%/73% contrast is
the number that motivated this series being written.

🚨 **It has no backing table anywhere in this repository.** A sweep of `docs/`
(the archive included) for "81 banked", "89%", "43%", "44%", "73%" and
"segment 2+" returns those two files and nothing else — no source table, no
per-segment dump, no aggregation script in `scripts/`. The figures are asserted,
not traceable.

⚠️ **Two things follow, and both are registered now rather than argued later.**
1. **This document does not quote 44%/73% as a rate**, and neither may the
   evidence file that comes out of it. Standing decision 7 already forbids
   pooling across the beta37/38/40/42 control-law changes; an untraceable pooled
   figure is worse than a pooled figure. Call it what it is — **an unverifiable
   pooled assertion** — or do not call it anything.
2. **The qualitative claim survives without it.** The per-item records below are
   real, are per-item, and say the same thing: a leg that does not start fresh is
   a thinly-evidenced property, and the one time it was measured near 2 m it
   failed.

### ✅ What the justification actually rests on

**(a) The only beta57+ real-junction data — and it is at 0.8 m, not 2 m.**
`docs/gate5-beta57-PASSED-20260818.md`, four segments, three real −55.0°
junctions, legs 0.7593–0.9032 m:

| seg | leg | landing | stop | linear | turn | realign |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8005 m | 0.1038 | `target_reached` | 3 | 0 | 0 |
| 2 | 0.9032 m | **0.0863** | `target_reached` | 3 | 3 | **1** |
| 3 | 0.8198 m | 0.1261 | `target_reached` | 3 | 2 | 0 |
| 4 | 0.7593 m | 0.1129 | `target_reached` | 3 | 1 | 0 |

4/4 reached. 🔑 But that document flags its own limits, and they bind here: the
one realignment in segment 2 was the **post-turn alignment gate**
(`before_linear: true`, 15.121° corrected), not a mid-drive correction; every
segment used 3 linear pulses against a ceiling of 22, so nothing bound; and *"n =
1 either way, so do not read a trend into it."* **This is not evidence about a
2.0 m post-turn leg. It is evidence that 0.8 m post-turn legs work.**

**(b) The one on-record run at this geometry — and it FAILED.**
`docs/loop-to-tolerance-reach-20260811.md`, beta41, *"2 × 2.0 m legs, one 60°
junction"*:

| run | leg | pulses | landing | stop |
| --- | --- | --- | --- | --- |
| `…235133Z` seg1 | 2.000 m | 5 of 10 | **0.0690 m** | `target_reached` |
| `…235133Z` seg2 | 1.942 m | 5 of 10 | 0.1797 m | **`target_requires_reverse_recovery`** |

Segment 2 converged 1.9417 → 0.1797 m and stopped **0.1797 m out while facing
119° away from the target**, having drifted right of the line; beta22 containment
correctly refused the U-turn. Root cause is one record: at pulse 4, 0.3246 m out
with −26.914° aim, the **beta38 re-aim guard suppressed a correction** on a
projected miss of 0.1469 m against a 0.150 m tolerance — a margin of **3.1 mm** —
and then under-predicted by **32.8 mm**, because the guard measured the miss at
closest approach while the mower drove a whole pulse past it.

⚠️ **Three honest caveats on that prior, stated before it can be leaned on.**
- It is **not on the accepted profile.** That document says so itself:
  `max_linear_pulse_ceiling` was 10, where the profile is 22. Its landings *"do
  not compare to Gate 5."*
- The specific defect is from a **superseded control-law era**. beta57's
  `_mid_drive_realign_decision` now decides on the next-pulse landing, which is
  exactly the missing term. So the failure is not expected to repeat — that is a
  **prediction this series can falsify**, which is why the geometry was chosen.
- n = 1. One failure is not a rate.

**(c) The run that was proposed at almost exactly this shape and never
executed.** `docs/NEXT-SESSION.md:806-814` (beta59 era, beta57+) called for *"a
leg of ≥ 1.9 m after a junction turn"*, noted the operator's original
1.21/2.00/1.92 m path with −99° and −58° junctions was *"close to the right
shape, though the −99° junction … should come down to 45-70°"*, and flagged the
watch item: *"whether any mid-drive correction fires at all — none ever has on
the new code. If one does, whether the budget holds."* It was parked at the
operator's call and never ran.

🔑 **So the honest prior at ~2 m after a genuine junction is: 1 attempt, 1
failure, on a superseded control law and a non-accepted profile.** That is a
worse starting position than the 4.0 m series had (1 reached / 1 failed), and
this document is written to that fact rather than around it.

---

## 3. Configuration — FROZEN, and the four traps that would void the series

**Service:** `raw_pymammotion_execute_multi_segment`
(`custom_components/mammotion/services.py:123`, schema
`RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA` at `services.py:1647`).

🚨 **NOT two calls to `raw_pymammotion_execute_vector_segment`.** Two separate
calls would let the mower sit and re-acquire between legs, which destroys the one
property being measured: segment 2 starting from segment 1's actual finishing
pose, inside one dispatch, with the VIO offset carried across
(`carried_vio_offset`, `services.py:18063`).

### 🚨 TRAP A — `max_real_segments` defaults to **1**, and segment 2 SILENTLY DOES NOT RUN

Verified in code, not inferred. `services.py:1678` defaults
`max_real_segments` to **1**; `services.py:18070-18080`:

```
if not dry_run and segment_index > max_real_segments:
    result["segments"].append({... "real_segment": False, "passed": None,
                               "skipped_reason": "max_real_segments_reached"})
    result["stop_reason"] = "max_real_segments_reached"
    return result
```

So a 3-point path dispatched at the default drives **segment 1 for real and skips
segment 2 entirely**, returning a run record that contains two segments and looks
complete. ✅ **This series requires `max_real_segments: 2`.** The upper bound is
`REAL_CLICK_TO_GO_SEGMENT_LIMIT = 4`
(`custom_components/mammotion/manual_motion.py:42`), so 2 is well inside it.

⚠️ A run that returns `stop_reason: max_real_segments_reached` is a
**configuration error — discarded, not scored.** It is not a sample of this
population and it is not a failure of the control law.

### 🚨 TRAP B — the schema defaults are not the accepted profile, and **10 of the 19 profile keys diverge**

Computed against the live multi-segment schema, not remembered:

| profile key | multi-segment default | **accepted** |
| --- | --- | --- |
| `max_linear_pulse_ceiling` | *absent* (None) | **22** |
| `waypoint_tolerance` | 0.08 | **0.15** |
| `min_progress_distance` | 0.06 | **0.0025** |
| `linear_pulse_duration_ms` | 3500.0 | **1300** |
| `calibrated_forward_heading_offset_degrees` | 116.5 | **102.4** |
| `vio_turn_max_commands` | 8 | **4** |
| `max_linear_commands` | 2 | **3** |
| `max_turn_translation_distance` | 0.25 | **0.3** |
| `ble_auto_recover` | True | **False** |
| `sample_delays` | [0,5,10,20,30,45,60] | **[0, 3]** |

The other nine profile keys already match the multi-segment default and must
still be sent explicitly.

✅ **Send `docs/accepted-profile.json` verbatim — all 19 keys — and verify
identity key-by-key against the echoed payload before the run counts.** A run
whose echoed profile differs on any key is **discarded, not scored.**

⚠️ Note `vio_turn_max_commands` in particular: the multi-segment default of **8**
is *more* permissive than the accepted **4**, so forgetting it makes the junction
turn easier than the accepted profile allows and produces a number that does not
belong to this population. Divergence in the generous direction is still
divergence.

🔑 **The two services' schemas diverge on exactly three keys** —
`max_linear_commands`, `max_turn_commands`, `max_real_segments` — pinned by
`tests/components/mammotion/test_collinear_leg_split.py:556-585`. The accepted
value for `max_turn_commands` (4) happens to equal the multi default; the other
two are covered above. **Every non-profile parameter (`vio_max_realignments`,
`vio_realign_threshold_degrees`, `linear_speed_*`, `angular_speed_*`,
`vio_angular_speed`, `slow_*`, `linear_distance_ceiling_factor`,
`toward_mirror_degrees`, `vio_calibration_pulse_count`,
`min_heading_change_degrees`) takes its schema default, and those defaults are
identical on both services.** Nothing needs a documented override beyond
`max_real_segments: 2`.

### ⚠️ TRAP C — leg splitting: OFF here, but only because the key is omitted

`_split_long_legs` (`services.py:2407`) operates **per leg**, not on total path
length: each leg longer than `target_length_m` becomes
`ceil(length / target)` collinear sub-legs. It is **off by default** —
`split_leg_target_length_m` has no schema default (`services.py:1690`,
`vol.Optional` with no `default=`), and `None` returns the points unchanged.

🔑 **So this series must simply OMIT `split_leg_target_length_m`.** With 2.0 m
legs it would not fire even at the card's 3.85 m target, but omitting it is the
guarantee, not the arithmetic.

🚨 **The card, however, always sends it** (`SPLIT_LEG_TARGET_METRES`, mirrored at
`services.py:15638`). A card-driven version of this path would still be 2 real
legs today — but the card is not the dispatch route here, and if leg 1 were ever
lengthened past 3.85 m the card would insert a collinear sub-leg and the run
would measure the splitter. **Dispatch the service directly.** This is the
multi-segment sibling of TRAP 1 in the 4.0 m predeclaration.

### 🚨 TRAP D — the junction angle is bounded by the accepted profile, and −99° is REFUSED

A `turn_mode: vio` multi-segment path runs a zero-motion geometric preflight on
every junction (`services.py:18010-18055`) and refuses the whole path with
`path_turn_infeasible` if any junction is infeasible. Evaluated at the accepted
profile (`vio_turn_max_commands: 4`, `turn_pulse_duration_ms: 1500`,
`motion_refresh_interval_ms: 200`, `max_turn_translation_distance: 0.3`,
`heading_tolerance_degrees: 18`, `turn_degrees_per_second: 37`):

| junction | commands needed / max | est. translation / cap | verdict |
| --- | --- | --- | --- |
| −55° | 3 / 4 | 0.143 / 0.30 m | feasible |
| **−60°** | **3 / 4** | **0.156 / 0.30 m** | **feasible, 1 command spare** |
| −75° | 4 / 4 | 0.195 / 0.30 m | feasible, **zero spare** |
| −90° | 4 / 4 | 0.234 / 0.30 m | feasible, **zero spare** |
| −99° | 5 / 4 | 0.257 / 0.30 m | 🚫 **`path_turn_infeasible`** |
| −120° | 6 / 4 | 0.312 / 0.30 m | 🚫 refused |

🚨 **The never-run 2026-08 proposal's −99° junction is refused before dispatch on
today's accepted profile.** Anyone reaching for that geometry from
`docs/NEXT-SESSION.md` would burn a daylight window on a zero-command refusal.
This is worth its own line in the record.

### The chosen geometry, and why

**Three points. Leg 1 = 2.000 m, junction −60.0°, leg 2 = 2.000 m.**

1. **−60° is precedent-bounded in both directions.** −55° is flight-proven at
   beta57 (§2a, three of them). The parked proposal explicitly asked for the −99°
   to *"come down to 45-70°"*. −60° sits inside that window and is the exact
   junction of the one prior run at this leg length (§2b). Nothing here is
   invented.
2. **−60° keeps a command of margin** (3 of 4) and 48% translation margin
   (0.156 m against the 0.30 m cap). The project rule is *confirm each bound
   exceeds what the run needs to demonstrate its criterion* — −75° and −90° sit
   at 4/4 with **zero** spare, so one slow turn pulse converts a landing
   measurement into a `turn_budget_infeasible` refusal and the run measures the
   turn guard instead of the leg.
3. **−60° is unambiguously a direction change.** It is 3.3× the
   `heading_tolerance_degrees` of 18, so the junction turn certainly executes; it
   cannot be mistaken for a Route B collinear continuation.
4. **Leg 1 = leg 2 = 2.000 m makes each run a matched pair.** The only difference
   between the two segments is that segment 2 starts from segment 1's actual
   finishing pose and after a turn. Segment 1 is its own within-run control for
   day, light, battery, BLE and yard. Leg 1 also has its own prior at exactly
   this length (0.0690 m, `target_reached`).
5. **2.000 m for leg 2 is the shortest length that can exercise what is being
   tested.** Corpus replay put every old-vs-new re-aim divergence *"at 0.85-1.13 m
   range, 15.5-17.1° aim, on legs of 1.9-4.0 m"* — below ~1.9 m the trigger window
   is unreachable and the run would be silent by construction.
6. **This re-runs the one failing geometry on the accepted profile for the first
   time.** §2b's run was on `max_linear_pulse_ceiling: 10`, off-profile, on a
   superseded guard. Same shape, accepted profile, current control law.

**`turn_mode: vio`. Daylight throughout.** Standing decision 4 closed night, and
`turn_mode: night` with more than one segment is refused outright
(`night_multi_segment_unsupported`, `services.py:18057-18061`) — so this geometry
has no night form at all.

### The frozen payload shape

```
service:  raw_pymammotion_execute_multi_segment
  entity_id:            <the mower>
  points:               [P0, P1, P2]      # P0=start, |P0P1|=2.000, |P1P2|=2.000, junction -60.0 deg
  area_hash:            <from the fresh map read, per run>
  dry_run:              false
  confirm_blades_off:   true
  confirm_clear_area:   true
  max_real_segments:    2                 # TRAP A
  allow_degraded_rtk:   false
  safety_overrides:     []                # none, ever, in this series
  <all 19 keys of docs/accepted-profile.json, verbatim>
  # split_leg_target_length_m: OMITTED    # TRAP C
```

⚠️ **A dry run of the identical payload precedes every armed dispatch**, to read
back `junction_turn_feasibility` and the echoed profile before anything moves.
The dry run is a check, not a sample.

---

## 4. Target n, and what n = 5 does and does not buy

**Target n = 5 runs.** Each run is one two-segment dispatch.

🚨 **The sampling unit is the RUN, not the segment.** 5 runs is **n = 5**, not
n = 10. Segment 2 is conditioned on segment 1's outcome — that dependence is the
entire point of the geometry — so pooling the two into ten independent segments
would be exactly the error standing decision 7 forbids, in miniature. Do not do
it in the evidence file.

The statistics are the same as the 4.0 m series and are stated up front so nobody
over-reads the result. 5 of 5 successes gives a 95% confidence lower bound on the
true success rate of only about **55%** (rule of three: upper bound on the failure
rate ≈ 3/n = 60%). **n = 5 cannot demonstrate "reliable enough to trust
unwatched."** What it can do is:

- detect a failure rate that is *high* (≳ 40%) with good probability;
- produce a landing **distribution** for a post-turn leg where currently one
  number exists, and that number is a failure;
- answer, for the first time, **whether a mid-drive correction fires at all** on
  the current control law.

🔑 **n = 5 is a SCREEN, not a certification.** Say so in the evidence file. If it
passes, the next question is more n, not a stronger claim.

⚠️ **n is not raised to compensate for this being the riskier geometry.** Five
armed daylight two-segment runs is already twice the motion of the 4.0 m series
per run; the correct response to a binding constraint here is a **shorter leg 2
or a smaller n**, never a longer session (see §6, clause c).

---

## 5. What is recorded, per run — decided before any data exists

**Everything below is recorded PER SEGMENT, separately, never summed.** CLAUDE.md:
*verify with per-item records, not aggregates* — net figures have already hidden a
27° turn reversal and a live BLE link in this project.

Primary, for **each** of segment 1 and segment 2:
1. `stop_reason` — `target_reached` or the named refusal.
2. **Landing distance** (m) from that segment's frozen target.
3. **Linear pulses used**, against the 22 ceiling.
4. **Realignments used**, and 🔑 **split by kind**:
   - `post_turn_alignment.correction_attempted` — did the post-turn gate fire,
     and what was its pre/post error;
   - mid-drive entries in `realignments[]` — how many, at what range and aim;
   - entries in `realignments_suppressed[]`, with `projected_landing_m`,
     `perpendicular_miss_m` and `metres_per_pulse` — 🔑 **the suppressions are as
     informative as the corrections**; the §2b failure was a suppression with a
     3.1 mm margin.
5. Terminal heading error and cross-track at finish.
6. **Entry pose for segment 2**: position, VIO heading and `carried_vio_offset` at
   the moment segment 1 stopped — this is the inherited error, the variable the
   whole series exists to expose.
7. Junction turn: commands used against `vio_turn_max_commands: 4`, achieved
   rotation, and translation during the turn against the 0.3 m cap.

Pre-dispatch, from the dry run:
8. `junction_turn_feasibility` — turn degrees, `estimated_commands_needed`,
   `estimated_translation_m`, `reason`.
9. Key-by-key profile identity result (§3 TRAP B) and the echoed
   `max_real_segments` (§3 TRAP A).

Covariates, recorded but **not** gated on:
10. `ble_rssi` at dispatch; refresh writes sent vs completed; max write gap.
    ⚠️ `ble_rssi` is self-reported and **does not predict cadence** (within-run
    median r = +0.042 over 24 runs). Record it; do not gate on it.
11. Battery %, RTK fix state, `tracked_features` (⚠️ 80 is saturation, and it
    falls off a cliff — 79 is "still enough", not margin).
12. Wall-clock duration, and the gap between segment 1 stopping and segment 2
    starting.

Safety, every run: gate blockers at dispatch, containment, stop confirmed, and
the gate **disarmed and verified afterwards from the live API AND RAW
`core.config_entries`** (⚠️ HA writes `.storage` lazily — a RAW read taken
immediately after a disarm can lie for ~15 s).

🔑 **Record the per-pulse remaining-distance trace for both segments**, as
`docs/loop-to-tolerance-reach-20260811.md` did. A run that converges
monotonically and one that oscillates into tolerance are different outcomes with
the same landing number, and only the trace separates them. The §2b failure is
legible **only** in its trace.

---

## 6. The pass criterion — FIXED NOW

**PASS** requires **all** of:

- **(a) 5 of 5 runs return `target_reached` on BOTH segments** — 10 of 10
  segments. A run in which segment 1 reaches and segment 2 does not is a **FAILED
  RUN**, not a half-success.
- **(b)** every landing, both segments, ≤ the accepted `waypoint_tolerance` of
  **0.15 m**. Implied by (a); stated separately so a future tolerance change
  cannot silently move the bar.
- **(c) 🔑 SEGMENT 2 RETAINS REALIGNMENT MARGIN: on every run, segment 2's total
  realignments used — post-turn gate plus mid-drive, they share one counter — is
  ≤ 2 of 3**, and no run stops on `post_turn_realign_budget_exhausted`
  (`services.py:16775-16778`) or `vio_realign_budget_exhausted`
  (`services.py:17421-17424`).
- **(d)** zero safety-gate trips, zero containment breaches, stop confirmed on
  every run.
- **(e)** every scored run is **profile-identical key-by-key** and echoed
  `max_real_segments: 2`. A run that is not is **discarded, not scored**.

### Why clause (c) is a pass criterion and not just a covariate

This is a judgement call and it is recorded as one. **Verified in code:**
`realignments_used` is initialised at `services.py:16470` inside the
single-segment executor, which the multi-segment handler calls once per segment
(`services.py:18155`) — so **the budget of 3 RESETS per segment**. But *within*
segment 2, the post-turn alignment gate and the mid-drive corrections **spend the
same counter**. `docs/NEXT-SESSION.md:812-814` predicted exactly this: *"A leg
following a junction turn has an effective mid-drive budget of 2, not 3."*

It has never been measured, because it needs a segment that both fires the
post-turn gate **and** runs long enough to need a mid-drive correction — and no
beta57+ run has had both. Gate 5's segment 2 fired the post-turn gate (1 of 3) on
a 0.90 m leg that could never reach the mid-drive window.

Clause (c) is a pass criterion rather than a covariate because **the remedy is
constrained in advance and is not available after the fact.** CLAUDE.md is
explicit: ⚠️ *"Do not raise `vio_max_realignments` — tried twice, reverted twice."*
So if the budget binds, the permitted responses are a **shorter leg 2** or a
**smaller n**, both of which change the population and therefore require a new
predeclaration. Leaving budget pressure as a covariate would invite exactly the
retroactive move this discipline exists to prevent: seeing 3-of-3 consumed on a
run that still landed, and calling it a pass with a note.

⚠️ **≤ 2 of 3, not ≤ 3 of 3.** Mirrors clause (c) of the 4.0 m predeclaration:
the bar is *margin retained*, not *budget survived*. A segment that spends its
last correction and lands is one gust of drift from a refusal.

### The criterion is deliberately strict at 5/5

With the prior at 0/1 at this geometry (§2b), a criterion that tolerates one
failure cannot distinguish the status quo from an improvement. **If 4/5 occurs,
that is a FAIL with an informative failure**, and the response is a predeclared
follow-up — **not** a retroactive softening of this line.

🚨 **A run that stops safely on a named refusal is a FAIL, not a smaller number.**
`target_requires_reverse_recovery` at 0.1797 m is the already-observed failure at
this exact geometry; recording it as "stopped safely, 18 cm out" and moving on is
how a reliability series becomes a feasibility demo.

---

## 7. The falsifiers — what result would mean this is wrong

Written now, so no result can be reinterpreted into a success.

1. **The beta57 mid-drive fix is what this geometry was chosen to expose. If NO
   mid-drive correction fires on any of the 10 segments — no `realignments[]`
   entry with `before_linear` absent — then the series is SILENT on it**, whatever
   the landings say. That is an **informative null**: it means 2.0 m still does not
   reach the trigger window, and the follow-up is a **longer leg 2 under a new
   predeclaration**, not a claim that the fix works. A PASS with zero mid-drive
   corrections must say this sentence in the evidence file.
2. **The premise — "a leg that does not start fresh is a different property" — is
   REFUTED at 2.0 m if** all 5 runs pass, segment-2 landings fall inside the range
   of the same runs' segment-1 landings, and no run shows realignment pressure on
   segment 2 above segment 1. Then the risk this series was written about does not
   exist at this length, and the honest follow-up is **length, not geometry**.
3. **The shared-budget prediction is CONFIRMED-but-not-yet-binding if** the series
   passes while segment 2 consistently consumes ≥ 2 realignments where segment 1
   consumes ≤ 1. That result **caps future leg-2 length**; it does not authorize
   longer legs, and it does not authorize raising the budget.
4. **A PASS driven by a suppression that nearly missed is not a clean PASS.** If
   any run records a `realignments_suppressed` entry whose
   `projected_landing_m` is within 10 mm of `waypoint_tolerance` — the §2b
   signature, which had 3.1 mm — the evidence file must name it, even if every
   landing was inside tolerance. **Near-misses are per-item records too.**

### The comparison against the 4.0 m baseline — rule fixed now

Segment-2 landings will be compared to the 4.0 m series' landings **descriptively
only**: do the segment-2 values fall inside the range of the baseline's? 🚫 **No
significance test will be computed and no p-value will be quoted.** At n = 5
against n = 5, with different leg lengths and a dependence structure, any test is
theatre. Fixing this now removes the temptation to pick a test after seeing which
way it points.

---

## 8. The abort rule — FIXED NOW

Stop the series immediately and write it up, without dispatching the remainder,
if any of:

- **any** containment breach, or any run leaving the frozen corridor;
- **two consecutive** failed runs (a run being failed if *either* segment fails);
- **any** run ending in `target_requires_reverse_recovery` — that is the §2b
  failure signature returning on the current control law, and it is a finding
  that deserves analysis before more motion, not another data point;
- **any** budget-exhaustion refusal (`post_turn_realign_budget_exhausted`,
  `vio_realign_budget_exhausted`, `turn_budget_infeasible`);
- battery below **35%** off-dock at the start of a run (the 2026-08-24 incident
  began at 29% off-dock);
- BLE `ble_link_live` off, or a `queue_settle` that is not live at depth 0;
- the operator withholding go/no-go for any reason.

**Not aborts — configuration errors. Fix and re-dispatch; the run is discarded,
not scored:**
- `max_real_segments_reached` (TRAP A);
- `path_turn_infeasible` from the pre-dispatch preflight (TRAP D) — zero commands
  are sent, so nothing moved;
- any key-by-key profile mismatch (TRAP B).

🔑 **An aborted series is reported as an aborted series.** Partial n is a result,
not a draft.

---

## 9. Safety preconditions — every run, no exceptions

- 🚨 **Scan the corridor against the MAP every time.** `step_path_contained` and
  its siblings measure clearance against the **operator-supplied polygon**, not
  the mowing area. On 2026-09-03 a position with 2.8447 m of real yard clearance
  passed 15/15 gates against a 3.20 m requirement because the corridor was centred
  on the mower by construction. **The gate is not a substitute for the scan.**
- 🚨 **An L-path corridor is not a leg corridor.** Size it from a fresh scan for
  **all** of:
  - leg 1 (2.0 m) plus along-track overshoot beyond P1;
  - leg 2 (2.0 m) plus along-track overshoot beyond P2;
  - **the junction turn itself** — the mower sweeps an arc about a rotation
    centre roughly 13.8 cm from the tracked point, and the preflight budgets up to
    0.156 m of translation at −60°, in an unpredicted direction;
  - **cross-track excursion on both legs**, including the drift-right that ended
    the §2b run 119° off target;
  - and the **re-aim manoeuvre space** at the inside of the corner, where a
    mid-drive correction on segment 2 will rotate in place.
  ⚠️ The failure in §2b ended with the mower facing 119° away from its target. Size
  the corridor for a mower that ends up pointing the wrong way, not for the
  intended path.
- Daylight, operator present, emergency stop accessible.
  ⚠️ **Verify the physical e-stop is CLEAR before every run.** A forgotten e-stop
  is invisible in telemetry — it silently no-op'd five real motion commands over
  ~40 min on 2026-07-19 while every health indicator read green.
- Blades off, confirmed; `confirm_blades_off` and `confirm_clear_area` both true.
- Explicit per-run go/no-go **immediately before dispatch** — not once for the
  series, and not once for the pair of segments.
- A dry run of the identical payload immediately before each armed dispatch (§3).
- Gate disarmed and verified from live API **and** RAW after every run.
- Dock and charge between runs as needed; do not leave the mower off-dock at low
  battery.

---

## 10. What a PASS authorizes — and what it does not

A PASS authorizes **one thing**: extending this same series to larger n at this
same geometry, or opening a **separately predeclared** series at a different leg-2
length or junction angle.

🛑 It does **not** authorize:
- resuming Phase 2 continuous steering — **standing decision 5, CLOSED**, and
  nothing in a stop-measure-go series bears on it;
- reopening accuracy — **standing decision 3, CLOSED** at the 0.065 m sensing
  floor;
- reopening night — **standing decision 4, CLOSED**, and this geometry has no
  night form (`night_multi_segment_unsupported`);
- raising `vio_max_realignments`, `max_linear_pulse_ceiling`,
  `vio_turn_max_commands`, `max_turn_translation_distance`, or any other bound;
- removing any per-run operator confirmation, or dispatching a path unwatched;
- quoting a segment-2 success rate as a **rate** — see §4, and standing decision 7;
- describing click-to-path as "trustworthy unwatched." **n = 5 cannot support that
  sentence**, and this document is the record that says so before the data exists.

A PASS is **evidence toward standing decision 2 and nothing more**: that
click-to-path's harder half behaves, at one geometry, on the accepted profile, at
n = 5.

**A FAIL authorizes nothing beyond writing up the failure mode** — which, given
§2b, is likely to be the most valuable output this series can produce.
