# Night segment: the implementation plan

**2026-08-13, off-mower.** Produced by a 20-agent design workflow
(`wf_3ae33cc2-1cf`): 4 mapping agents read the code, 4 designed independently (one
mirror fix, three night-gate architectures), 11 adversarially verified, 1
synthesised. Raw agent output:
`~/.claude/projects/.../subagents/workflows/wf_3ae33cc2-1cf/journal.jsonl`.

🚨 **THE VERIFIERS REFUTED EVERY GATE DESIGN, AND THEY WERE RIGHT.** Scores 4–6 of
10. Read §1c before anything else: **three independent verifiers found a second
missing parameter that kills the night segment in its opening turn**, and I had
already written this document claiming there was only one. Two other claims of mine
are corrected in §7.

## 1. 🚨 The two findings that change the plan

Both were found **independently by two agents each**, which is why they lead.

### 1a. The two additive sites CANCEL — fixing the obvious one alone breaks a probe

`_raw_vector_readiness_target_points` (`services.py:9448-9465`) synthesises a map
target as `toward + offset`. The executor (`services.py:11094-11100`) converts it
back as `map − offset`. **They cancel exactly.** That is why
`_raw_vector_readiness_test` aims correctly today *despite* the conversion being
wrong by construction.

**Fixing only `services.py:11095` — the line I had identified as "the" defect —
would break that readiness probe by roughly 205°.** Both sites move together, or
the probe is explicitly pinned to the additive pair.

This is the single most valuable thing the workflow produced. I would have shipped
the isolated fix.

### 1b. `turn_mode: "legacy"` silently gets single-shot turns TODAY

The VIO branch passes `motion_refresh_interval_ms` to the turn primitive
(`services.py:11488`). **The legacy branch at `services.py:11499-11517` does not.**
The primitive defaults it to `0` (`:10210`); both service schemas default it to
`200` (`:1020`, `:1145`).

So an operator calling with `turn_mode: "legacy"` and refresh 200 gets it silently
dropped, the h-watchdog stops the motor, and the turn runs single-shot — the exact
pre-beta47 failure (4 commands, 29° of 90°). **Without this one kwarg a night
segment cannot turn.** My five successful night turns called the primitive
*directly* with refresh, bypassing this branch entirely, which is why it never
showed.

⚠️ Note this is a **live latent defect on a shipped path**, not only a night
concern. The card's Nudge rides `legacy`.

### 1c. 🚨 And a SECOND missing parameter — angular speed defaults to 180, which does not pivot

Found by **three independent verifiers**. Verified by hand at HEAD:

```
services.py:974    angular_speed_fast  default 180
services.py:977    angular_speed_slow  default 180
services.py:11502-11503   the legacy branch passes both straight through
card                      sends angular_speed_fast ZERO times -> the 180 applies
```

**All five converging night turns used `angular ±500`** — confirmed in the raw
evidence, not the write-up (`docs/evidence-night-turn-*-20260813.json`,
`selection.angular_speed: 500`). At **180** the 2026-07-25 A/B measured roughly
**3° total** on a stationary pivot: below the static-friction deadband. A moving
arc actuates fine at 180 because it only needs a track differential; a stationary
pivot has to break static friction on both tracks.

**Consequence:** at ~3°/pulse a 40–90° opening turn exhausts `max_turn_commands`,
the primitive returns `max_commands_reached` (`:10515`), and the segment reports
`turn_phase_incomplete`. **The night segment dies before the linear phase begins.**

**It also makes the feasibility model lie in the unsafe direction.** A
`_night_turn_budget_feasibility` built on the 48.15° quantum — measured at
angular 500 — would admit turns the executor cannot finish at 180. That is exactly
the beta32 defect (planner and executor disagreeing) reintroduced.

🔑 **The general lesson: the standalone turn service and the segment executor's
legacy branch are NOT the same code path.** Five successful night turns prove the
*primitive* works. They prove nothing about the *branch*, which supplies different
defaults for at least two parameters that decide whether the mower moves at all.
Any night work must re-verify at the segment call site.

## 2. The good news: the linear phase is nearly free

Three agents independently confirmed the night linear phase needs almost nothing:

- Every forward pulse hard-codes `angular_speed: 0` (`services.py:11881-11884`).
- `_raw_vector_linear_command_selection` (`:10523-10559`) reads only x/y — no
  heading field anywhere in it.
- Arrival is a 2D disc test against `waypoint_tolerance`
  (`_manual_velocity_completion_status`, `:3680-3711`).

All RTK-backed, all works in the dark, **bit-identical to daylight**.

🔑 **And the map-frame aim error is ALREADY COMPUTED every pulse, from RTK alone,
in both modes — then discarded.** `_manual_velocity_path_progress_diagnostic`
(`:3585`) returns `movement_vector_heading_degrees` = `atan2(dy,dx)` of the
completed pulse and `expected_target_heading_degrees` = bearing to target, both
map-frame (`:3644-3673`). Their difference *is* the night aim error: no VIO, no
`toward`, no mirror, no constant. It is already in `progress_diagnostics` on every
run on disk.

🔑 **A mis-aimed night leg does not run away.** `path_progress_distance` is the
projection onto the unit vector to the target, recomputed from the *current*
position each pulse (`:3639-3647`). Past the waypoint it goes negative and
`max_no_progress_pulses: 3` stops the segment. Worst case ~3 pulses past closest
approach, then `no_target_progress`.

## 3. The decision

**Architecture: an explicit third `turn_mode: "night"`.** Two of the three gate
designs converged on this independently; I am taking the *minimal-surface* one as
the spine and grafting from the others.

**Rejected: the negotiated `turn_mode: "auto"` design.** It self-reported
`changes_daylight_behaviour: true`, which violates the hard constraint — the
daylight path passed Gate 5 twice and is not available to disturb. ⚠️ **But it
produced findings 1a, and both items in §2.** Rejecting the architecture is not
rejecting its evidence.

**Why a third value rather than repurposing `legacy`:** `legacy` is not free to
change. `_raw_vector_readiness_test` forces it (`:9704-9731`) with the additive
inverse, and the card's Nudge forces it (`card:1169`). A third string leaves both
byte-identical for one tuple element and two `vol.In` lists.

**Why `vio_active` is never touched:** `"night"` simply is not `"vio"`, so all
seven `if turn_mode == "vio":` blocks stay inert. The unconditional gate inside
`_vio_turn_to_heading` (`:8061`) backstops any accidental reach. Both schema
defaults stay `"vio"` — an omitted field can never select night.

## 4. Ordered tasks

### Off-mower

1. **Mirror helpers.** `_TOWARD_MIRROR_DEGREES = 90.13` plus
   `_map_heading_to_toward_degrees()` / `_toward_to_map_heading_degrees()`, both
   `(K − h) % 360` (an involution). `grep -rn '90\.13' custom_components/` returns
   zero today — this is genuinely new code.
2. **Fix BOTH conversion sites together** (§1a) — `:11094-11100` and
   `:9448-9465` — or pin the readiness probe to the additive pair explicitly.
   **Do not fix one alone.**
3. **Pass `motion_refresh_interval_ms` AND a usable angular speed on the night
   branch** (§1b, §1c). Night must dispatch at **angular 500**, the only value any
   converging night turn has used; 180 does not break static friction on a
   stationary pivot. Scope both to night only; leave `legacy` byte-identical so
   Nudge is untouched, and record that `legacy`'s defects are *contained, not
   fixed*. ⚠️ **Neither the feasibility model nor any correction floor may be
   derived until both are in place** — the 48.15° quantum is an angular-500,
   refreshed figure and does not describe the branch as it stands.
4. **Schema + tuple:** `_VIO_TURN_MODES` → `("vio","legacy","night")` (rename to
   `_SEGMENT_TURN_MODES`; sole call site is the `turn_mode_valid` gate at `:11157`).
   Both schemas `vol.In([...,"night"])`, defaults unchanged.
5. **Night gates**, inserted *after* the `vio_active` append so that block is not
   edited: `night_requires_precise_rtk` (`allow_degraded_rtk` does **not** override
   at night — RTK is the only witness), and a `night_segment_too_long` cap.
6. **Night aim measurement** from §2's already-computed RTK bearings. Record
   `night_cross_track_uncorrectable` below the correction floor rather than firing
   a 48° actuator at a 15° error.
7. **`heading_calibration` echo** gains `"model": "mirror"|"additive_offset"` and
   `calibrated_forward_heading_offset_applied`, so no later session misreads a
   night evidence file. Keep every existing key.

### Tests

- The VIO daylight path is untouched — assert the dispatched VIO payload and
  `heading_calibration` block are unchanged.
- No frozen `LUBA_ACCEPTANCE_PROFILE` key changed value.
- The readiness probe still aims correctly after §1a.
- `turn_mode` omitted ⇒ `"vio"`, in both schemas.
- ⚠️ One design flagged that **9–10 existing backend tests pass
  `calibrated_forward_heading_offset_degrees=0.0`** to make a fixture "already
  aligned"; under a mirror default they would silently change meaning. Check this
  before changing any default.

### On-mower — and none of it needs daylight

8. **Re-measure the turn quantum under the segment's own dispatch path.** n = 5.
9. **Settle the reverse-travel question** (§5) — one backward pulse with `toward`
   logged before and after. The 2026-08-05 sweeps drove six backward pulses and
   simply did not log the field.
10. **First night segment:** one short leg, heavily instrumented.

## 5. What is NOT established

- **No closed-loop night SEGMENT has ever run.** Only turns.
- **The mirror has never been inside a control loop, at any hour.** Every
  validation on record is observational — the mirror *predicting* a facing that
  something else then measured. Under night it *steers*, so a systematic error in
  90.13 feeds back rather than merely mispredicting.
- **Whether `toward` flips 180° under REVERSE travel is formally open.** The design
  is forward-only and refuses reverse, which contains the risk without settling it.
- **Whether `toward` is course-over-ground or a fused BODY heading.** The two are
  indistinguishable on every forward measurement and differ by exactly 180° under
  reverse. The 99.55°-from-3.8 cm pivot is strong evidence for body heading; nobody
  has confirmed it.
- **`toward` latency during rotation** — still the thing that decides how tight a
  loop can close.
- **No landing accuracy is evidenced at night.** `waypoint_tolerance: 0.15` is a
  VIO-path number and must not be read as a night specification.
- ⚠️ **Two agent claims I have NOT verified and which contradict current notes:**
  that re-deriving the mirror constant over 17 runs gives **90.205 ± 0.145** rather
  than 90.13, and that the harness's 7-drive / 2.738° pin
  (`scripts/beta32_validation_run.py:137-149`) "claims completeness it does not
  have — over all 24 recorded runs the worst residual is 7.986° and six exceed
  3.0°". If true, the harness's 3.0° limit is not a safe design margin. **Verify
  before relying on either number.**

## 6b. 🗑️ Corrections to THIS document, from the verifiers

Recorded rather than silently edited, because the errors are instructive.

1. **"~205 degrees" was impossible and I repeated it.** §1a originally said a
   half-fix breaks the readiness probe "by ~205°". A verifier pointed out that a
   *normalized* heading error cannot exceed 180°. The number came from a design
   agent and I propagated it without checking. **The finding stands — the two sites
   do cancel and a half-fix does break the probe — but the magnitude was wrong.**
   Recompute it before quoting a figure.
2. **The runaway-safety argument cites the wrong mechanism.** §2's claim that
   `max_no_progress_pulses: 3` bounds a mis-aimed leg was challenged as "the wrong
   mechanism, and it misses the right one". **Unresolved — do not rely on the
   stated bound** until the actual stopping mechanism is read out of the code.
3. **A 30° correction floor suppresses corrections this project already judged
   legitimate.** The code's own validation table records
   `d 0.540  aim 23.30  perp 0.214 m -> correct (was allowed, legitimate)`
   (`services.py:10690`), and `_realign_cannot_improve_the_landing` documents that
   it deliberately **fails OPEN** because "suppressing a re-aim is the dangerous
   direction: a mower that stops correcting its aim keeps driving"
   (`:10695-10696`). A blanket night floor installs exactly that danger
   permanently. **The floor must be justified against that comment or dropped.**
4. **The correction floor was derived from the wrong quantum anyway.** At the
   segment call site's real parameters (angular 180, no refresh) the quantum is
   ~7°/pulse, not 48°. Fix §1b and §1c first, *then* derive any floor.
5. **`night_requires_precise_rtk` cannot be derived as specified.**
   `_runtime_motion_safety_summary` suppresses the blocker internally at
   `services.py:2031`, and the vector executor's call at `:11134` does not pass
   what the gate would need. And the claim that this "refuses runs the operator can
   currently force" is **false** — `allow_degraded_rtk` is not reachable on the
   vector-segment path at all.

## 7. 🗑️ One agent claim corrected here

One design lists, under "not established", that the 48.15° quantum "was measured
WITHOUT the refresh window that night mode adds". **That is wrong.** All five night
turns ran `motion_refresh_interval_ms: 200` with 6–7 refresh writes recorded per
pulse (`docs/evidence-night-turn-*-20260813.json`). The quantum **is** the
refreshed quantum. The feasibility model built on that caveat does not need the
hedge — though it does still rest on n = 5.

This is the kind of error the un-run verify phase existed to catch.
