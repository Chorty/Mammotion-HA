# Reach to 20 ft, and the re-aim trigger that was limiting it — 2026-08-17

⚠️ **NO MOTION HAS RUN ON ANY OF THIS.** Everything below is code, arithmetic
and replay against already-recorded runs. The changes are built and green on the
full CI gate suite; they have never touched the mower. Nothing here is a
measurement of new behaviour.

🚨 **THE ACCEPTED PROFILE IS NO LONGER ACCEPTED.**
`LUBA_ACCEPTANCE_PROFILE.max_linear_pulse_ceiling` moved 14 → 22, which owes the
§4 re-pinning in `docs/gate4-repass-20260805.md` and **another Gate 5** before
any run on this profile may be described as accepted. That cost was stated to
the operator and accepted before the change was made.

## 0. What was asked for, and what was authorized

The operator asked to move **50 ft (15.24 m) on a single waypoint**, and to add
a pre-dispatch refusal for over-long legs. Presented with two routes, they chose
the single-long-segment route (Route A) but **lowered the target to 20 ft
(6.096 m)** for now, and asked for the re-aim trigger fix as the default rather
than behind a flag.

So the authorized cap is **6.10 m**, and 50 ft is deliberately still refused.

## 1. The finding: the limit was never distance

The mid-drive re-aim fired on this test, unchanged since well before beta56:

```python
needs_correction = (
    abs(aim_error) > vio_realign_threshold_degrees   # 15
    and abs(aim_error) > heading_tolerance_degrees   # 18
)
```

A pure **angle**. But the quantity that decides a run is the **miss**,
`remaining_range × sin(aim_error)` — and the two diverge with range:

| aim error | range | miss | old trigger | new trigger |
| --- | --- | --- | --- | --- |
| 17° | 14.0 m | 4.09 m | **no** (17 < 18) | yes |
| 17° | 0.8 m | 0.23 m | **no** (17 < 18) | yes |
| 12° | 0.5 m | 0.10 m | no | no — lands inside |

The projected-landing machinery (`_projected_landing_after_next_pulse`,
`_realign_cannot_improve_the_landing`, beta42) has always reasoned in metres —
but it could only ever **suppress** a correction, never fire one. So the
controller was blind in exactly the long-leg regime, and a controller triggering
on angle alone is tuned for exactly one range. That is why ~0.8 m legs behaved
and a 1.65 m leg after a turn did not.

**This reframes the ~0.8 m operating rule.** It was a real, correctly-derived
rule for the control law as it stood — `L ≤ 0.79 m` falls out of inverting
`0.62 × L·sin(residual) + 0.065` at a 10° residual and a 0.15 m tolerance. It
was not, however, a property of the mower.

### Why a 6 m leg is credible at all

With a correction available every pulse, the landing is
`pulse_length × sin(final_aim_error)` — **independent of leg length**:

- correcting to 18°: `0.41 × sin(18°)` = **0.127 m** against 0.15 (15% margin)
- correcting to 10°: `0.41 × sin(10°)` = **0.071 m** (comfortable)

Leg length only hurts when the mower **stops** correcting. That is what
`vio_max_realignments: 3` and the angle trigger were jointly causing.

## 2. What changed

| # | Change | Where |
| --- | --- | --- |
| 1 | `segment_too_long` — pre-dispatch cap at `_MAX_SEGMENT_LENGTH_M` = 6.10 m, all non-night modes | `services.py` |
| 2 | `linear_budget_insufficient_for_segment` — refuse when the pulse ceiling cannot reach the leg. **Loop-to-tolerance only** | `services.py` |
| 3 | `_mid_drive_realign_decision` — trigger is the projected miss, gated by the smallest correctable angle | `services.py` |
| 4 | Mid-drive corrections close to 10°, not 18° | `services.py` |
| 5 | `max_linear_pulse_ceiling` 14 → 22 | card, README, frontend tests |
| 6 | Card mirrors both gates and stops claiming an acceptance the profile no longer has | card |

### 🚨 Scope was cut, and the cut is the most useful thing here

An earlier version of this branch also raised `vio_max_realignments` **3 → 10**
and added a **divergence detector** to make that safe. Both are gone. The budget
is back at the accepted 3.

Two rounds of review found the detector wrong **twice, for two different
reasons**, and neither would have been caught by a test:

1. **v1 compared before-vs-after within one correction.** But a correction turn
   translates the mower up to 0.30 m, and translation rotates the bearing by
   `atan(translation / range)` — 7.6° at 0.75 m of range, against a 1.0° margin
   justified by 2–4 cm of position noise. It measured the correction's own
   translation and called it divergence.
2. **v2 compared successive pre-correction errors.** But aim error inflates
   geometrically as range closes — `atan(c/d)` grows as `d` shrinks — so a
   perfectly healthy 6 m leg yields ~10.7 → 11.1 → 11.6 → 12.6 → 16.8 and trips
   every margin. It measured normal terminal geometry and called it divergence.

Both would have aborted good runs. Five of the six second-round findings existed
*only* because the budget went to 10 — the detector, the missing deadband, the
skipped stale-feed and no-progress aborts, unbounded correction translation, and
a zero-command re-aim loop introduced by the fix for a first-round finding.

**The trigger fix was implicated in none of them.** So it ships and the budget
does not. If a 6 m leg exhausts 3 corrections it stops safely on
`vio_realign_budget_exhausted` — which is a *measurement*, and a far better
basis for raising the budget than either geometry argument was.

### The deadband, and why the far-field win is smaller than it looks

`vio_realign_threshold_degrees` stays at its accepted **15**, not 10. The
correction turn closes to 10°, and a trigger floor equal to that tolerance means
a correction ending at 9.9° re-fires next pulse, while an error just past the
floor makes the turn primitive return `target_heading_reached` having sent
nothing. The 5° gap is the deadband.

⚠️ **So the far-field improvement is 18° → 15°, not 18° → 10°.** At 6 m that is
a 1.55 m miss now corrected where it took 1.85 m before. Real, but modest. The
turn primitive cannot hold better than 10°, and a deadband above that is
mandatory — closing the rest of the gap needs a shorter actuation floor, not a
smaller threshold.

### On item 2, and a bug the existing tests caught

The first version applied one conservative 0.30 m/pulse figure to both linear
modes. That is wrong: fixed-budget fires full `linear_pulse_duration_ms` pulses
**measured at 1.0785 / 1.0449 m** (2026-08-01), while loop-to-tolerance shortens
pulses on final approach and averaged ~0.36 m across the reach runs. The
conservative loop figure therefore **refused runs that Gate 4 and Gate 5 both
passed** — caught immediately by nine existing tests. The gate is now
loop-to-tolerance only and the accepted fixed-budget path is untouched.

Worth recording as a method note: the failure was caught by tests that existed
to protect the accepted profile, not by review.

## 2a. Review rounds 1 and 2 — fourteen findings

Two high-effort review passes before any deploy. **Round 2 found that two of
round 1's own fixes were wrong**, which is what triggered the scope cut above.
Round 1's table is kept in full because the superseded entries are the record of
how the detector failed.

### Round 1 — eight findings, all real

Reviewed at high effort before any deploy. Every finding held up; several were
defects that would have shown up as bad hardware runs rather than as failures.

| # | Finding | Fix |
| --- | --- | --- |
| 1 | **Divergence detector tripped on the correction turn's own translation** — 7.6° at 0.75 m range against a 1.0° margin. Would abort healthy runs. | ~~Compare successive pre-correction errors~~ — that fix was ALSO wrong (round 2); detector removed |
| 2 | `vio_max_realignments` default 10 **was the schema maximum**, leaving no headroom on a leg needing ~17 pulses | ~~Schema max 10 → 25~~ — budget reverted to 3, schema back to 10 |
| 3 | Card banner still printed "LUBA acceptance profile … Gate 5 re-pass 2026-08-12" **for a profile no Gate 5 has run** | Default label now states the un-acceptance |
| 4 | Divergence detector had **no wiring test** — deleting the whole block left 689 tests green | ~~Executor-driven test added~~ — detector removed |
| 5 | `current_point = current_after_realign` was dead (last read before the linear loop) | Block removed with fix 1 |
| 6 | Trigger floor (10°) **equals** the correction tolerance (10°), so an error just past it burns a slot on a zero-command turn | ~~Charge a slot only when dispatched~~ — that fix removed the only bound on ineffective re-aims (round 2). Fixed properly by keeping the threshold at 15, restoring a 5° deadband |
| 7 | `linear_budget_insufficient_for_segment` help text **could never render** — `BLOCKER_HELP` only serves `_preflight().blockers` | Card mirrors the budget arithmetic |
| 8 | Backend measures **live position → target**; card measured **waypoint → waypoint**. A 6.05 m plan can measure 6.25 m after the previous landing, refusing mid-path | ~~Card refuses at cap − 0.2 m~~ — that margin refused the 6.096 m leg the cap exists to allow (round 2). Card keeps the exact cap; a drifted later segment is refused by the backend with diagnostics |

Findings 1 and 6 are the ones worth remembering: both are cases where the code
was *internally* consistent and still wrong about the machine — 1 because a turn
is not a pivot, 6 because a correction is an angle with a minimum size.

## 3. What is measured, and what is not

**Measured, and unchanged by this work:**

- 4.0 m on a single straight segment, 11 pulses, 0.1023 m, stopping on tolerance
  (`docs/loop-to-tolerance-reach-20260811.md`). n = 1.
- The 1.65 m post-turn divergence, 0.2514 m out
  (`docs/evidence-real-go-card-beta55-20260815T204747Z.json`).
- Healthy pulse ~0.41 m, BLE-stalled ~0.22 m, 2 of 11 stalled on the 4 m leg.

**Not measured:**

- **Any leg longer than 4.0 m.** 6.10 m is an authorization number.
- Whether the projected-miss trigger converges on hardware.
- **Whether 3 corrections is enough AT 6 m.** Partially answered by replay
  below — but the corpus has no leg over 4 m, so the regime the change was built
  for still has no data. The first run is still the useful instrument.
- Whether 10° mid-drive corrections enter the `sweep_exceeds_any_pulse` regime
  more often than 18° ones did. The post-turn gate has run at 10° without it,
  which is the reason for confidence, not a proof.

## 3a. Replayed against 62 recorded segments — 2026-08-17

`scripts/replay_reaim_trigger.py` replays BOTH triggers against the per-pulse
geometry of every recorded VIO-path segment in `docs/evidence-*.json`. It
imports the shipped `_mid_drive_realign_decision` rather than reimplementing it,
models the four executor gates that stand in front of the re-aim block
(`target_reached`, `command_index < effective_linear_ceiling`,
`_requires_reverse_recovery`, and the shared budget), and self-validates by
replaying the OLD rule and requiring it to reproduce what each run recorded.

**Result: `vio_max_realignments: 3` holds on this corpus. 0 of 62 segments
exceed it; the maximum is 3**, on the 1.65 m beta55 segment, at the same pulses
the old trigger fired. Corpus-wide the new trigger adds **4 fires (18 → 22)**,
all one pulse earlier on the longest legs:

| segment | leg | old | new | the added decision |
| --- | --- | --- | --- | --- |
| `…20260812T001116Z#0` | 4.00 m | [9] | [8,9] | 0.846 m out, −16.72°, projects 0.246 m |
| `…20260812T002804Z#0` | 4.00 m | [9] | [8,9] | 0.915 m out, −16.23°, projects 0.258 m |
| `…20260811T235945Z#0` | 3.00 m | [6] | [5,6] | 0.979 m out, −17.06°, projects 0.291 m |
| `…20260812T185603Z#1` | 1.91 m | [3] | [2,3] | 1.128 m out, +15.49°, projects 0.301 m |

Every one is an aim error **under the old 18° gate** projecting a 0.25–0.30 m
landing against a 0.15 m tolerance — precisely the blind spot the change targets.

**Why believe it.** Self-validation is 55/62, but the ratio is not the load-
bearing evidence — two independent cross-checks are. Reconstructed
`facing`/`bearing`/`aim_error`/`distance` reproduce the executor's own recorded
values at **all 35 recorded decision points to 0.000497° and 4.5e-05 m** (the
`round(x, 3)` residual), and `metres_per_pulse` matches the 6 recorded values at
**0.0**. The 7 residual segments are law-version skew across a fourteen-beta
corpus, six of them identified structurally from their own records (beta36
`min_distance`, beta38 pre-quadrature `perpendicular_miss`, and two 2026-08-02
runs recording a re-aim at `effective_linear_ceiling` that the shipped gate
makes impossible); the seventh predates the suppression guard entirely.

⚠️ **THREE LIMITS, AND THE FIRST IS DECISIVE.**

1. **No leg in the corpus exceeds 4 m.** The authorized cap is 6.10 m. The
   regime this change exists for has no data, and no amount of replay creates
   any. Only hardware closes this.
2. **This is a counterfactual on a FIXED trajectory.** "Would have fired at K of
   N measured decision points" is sound; "the landing would have been X" is not,
   and the harness never claims it.
3. **The bias runs toward over-estimation, consistently.** These trajectories
   were produced by the old, under-correcting trigger, so they are systematically
   worse-aimed than trajectories the new trigger would produce; and an extra
   correction at pulse 8 changes pulse 9's geometry, likely removing the second
   fire. So "0 of 62 exceed 3" is the conservative direction.

🔑 **WHAT THE REASSURING NUMBER DOES NOT COVER — WATCH THIS ON THE RUN.**
A leg that follows a junction turn has an effective mid-drive budget of **2, not
3**: the post-turn alignment gate at `services.py:12184` spends the *same*
counter, and on beta55 segment 1 it took 1 of 3 before the linear loop started.
Nothing else binds first on a 6.10 m leg — the pulse ceiling is 22 against ~17
needed, and `linear_distance_ceiling` is 2 x the leg. If anything binds, it is
this, and the corpus figure says nothing about it.

## 4. What is owed before this is an accepted path

1. **§4 re-pinning** per `docs/gate4-repass-20260805.md`.
2. **Another Gate 5.** The profile changed.
3. A supervised run. The obvious first one is **not** 20 ft — it is a repeat of
   the geometry that already has a recorded failure (the 1.65 m post-turn leg
   from 2026-08-15), because that one has a **counterfactual on record**. A 20 ft
   leg that works proves less than a 1.65 m leg that stops diverging.
4. Only then a 6 m straight leg, then 6 m after a junction turn.

⚠️ Do not run these in the reverse order. The long leg is the interesting one
and the least diagnostic.

## 5. Deliberately not done

- **Route B (auto-splitting a long click into collinear sub-legs).** It reaches
  50 ft *inside* the measured 4 m envelope with no profile change, and it
  remains the cheaper route to the original ask. The operator chose Route A
  knowingly; this is recorded so the option is not lost.
- Raising `REAL_CLICK_TO_GO_SEGMENT_LIMIT` (still 4).
- Anything touching night, which is PARKED per standing decision 2.
