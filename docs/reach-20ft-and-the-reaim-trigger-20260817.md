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
| 5 | Divergence detector — stop when a correction leaves the aim worse | `services.py` |
| 6 | `vio_max_realignments` default 3 → 10; `vio_realign_threshold_degrees` 15 → 10 | `services.py` |
| 7 | `max_linear_pulse_ceiling` 14 → 22 | card, README, frontend tests |
| 8 | Card mirrors the cap (`MAX_REAL_SEGMENT_METRES`) and explains both new blockers | card |

### On item 6, and CLAUDE.md

CLAUDE.md says plainly that **raising `vio_max_realignments` is the WRONG fix**,
and on the evidence available in 2026-08-15 it was: the 1.65 m segment's aim
errors grew **16.96 → 21.22 → 24.975°** while every correction reported
`target_heading_reached`, so more budget would only have bought more corrections
chasing a target moving away faster. beta17 recorded the same shape.

That objection is answered rather than ignored:

- **Item 4 removes the cause.** Those corrections were not failing — they were
  *succeeding* against an 18° tolerance too loose to converge, leaving
  9.7 / 11.5 / 13.6° of residual against a bearing rotating −3.2 / −9.7 /
  −15.4° per pulse.
- **Item 5 detects the symptom.** If a correction leaves the aim error worse by
  more than `_REALIGN_DIVERGENCE_MARGIN_DEGREES` (1.0°), the segment stops on
  `vio_realign_diverging`. The measured signature worsened by 4.26° and 3.75°
  per step, so it is caught on the first one.

**The budget raise is safe only because of the detector. They ship together or
not at all.** A 6 m leg drives ~17 pulses; 3 corrections across that is the same
"stops correcting" failure in a new place.

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
- Whether the divergence detector's 1.0° margin is right. It separates the one
  recorded divergence from position noise; that is a sample of one.
- Whether 10° mid-drive corrections enter the `sweep_exceeds_any_pulse` regime
  more often than 18° ones did. The post-turn gate has run at 10° without it,
  which is the reason for confidence, not a proof.

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
