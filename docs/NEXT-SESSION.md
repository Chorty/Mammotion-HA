# Claude handoff: finish Mammotion-HA P0 beta

Updated 2026-08-06 after the beta22 containment deploy. This is the current handoff;
`docs/archive/NEXT-SESSION-2026-07-28.md` and the chronological sections in
`docs/p0-beta-release.md` are evidence, not current instructions.

## 🚨 START HERE, 2026-08-07 — beta23 deployed; the "just ask for a faster feed" fix is REFUTED

The host runs `0.6.4-beta23`, motion-disabled. It adds one read-only diagnostic,
`report_stream_probe`, and changes no motion behaviour. No motion has run.

**The probe answered its question, and the answer is no.** The wire report
`period` field defaults to 1000 ms and nothing in this integration had ever
lowered it, so the obvious hypothesis was a cheap ~5× win. It is refuted:

| requested period | mean median interval |
| --- | --- |
| 1000 ms | 706.9 ms |
| 200 ms | 590.3 ms |

Interleaved 1000/200 three times each. Observed ratio **1.20** against **5.0**
expected if honoured, and the spread *within* a setting (445–640 ms) is larger
than the difference *between* settings (117 ms) — the signature of no effect.
Across all 203 pooled intervals at every setting, p90 is 1002 ms and the max is
1107 ms: nothing ever waited materially longer than ~1 s and nothing responded to
a shorter request. Evidence:
`docs/evidence-report-rate-probe-20260807.json`.

⚠️ **One limitation bounds that conclusion.** `last_report_at` is stamped on
*every* received LubaMsg, not only on periodic position reports, so the probe
measures total inbound traffic rather than the position-report channel. The
refutation holds (a real 5× change in the report channel would still have moved
the total), but the **exact position-report cadence is still unmeasured**. To
close it, discriminate arrivals per `RptInfoType` or diff the parsed position
payload instead of the message timestamp.

**What this does to the plan.** The "expose `period`, request 200 ms" work is
dead — drop it. The *other* half of the feed fix is untouched and still worth
doing on its own merits: the Gate 4/5 executor never holds a continuous
subscription during motion while four other motion paths do
(`_manual_velocity_pulse_test`, `_manual_velocity_cumulative_pulse_test`,
`_experimental_execute_segment_burst`, `_manual_velocity_segment_test`). That is
a defect regardless of the rate the subscription runs at. The tolerance question
(§4 of the direction review) is now the more important branch.

**Also resolved:** the pre-linear post-turn guard site stays unguarded — replay
of all 10 committed `post_turn_alignment` records found zero aim errors ≥90°,
worst 18.78°. See `docs/evidence-post-turn-alignment-replay-20260807.json`.

**Documentation correction:** the runbook's "128 Mammotion entities" is a
readiness-poll value, not a stable count. `scripts/ha_restart.sh` exits as soon
as the count reaches 100, so the number it prints depends on when the poll lands
(it read 130 this time). The settled count is **131**, with 18
unavailable/unknown for an idle mower. Do not treat a difference here as a
regression.

## Historical start block — 2026-08-06 evening — beta22 is deployed motion-disabled; the U-turn path is closed

The host runs `0.6.4-beta22`. It is the beta21 tree plus the reverse-recovery
containment guard, and it is an **unaccepted, motion-disabled staging
candidate**. No motion ran during the deploy and none is authorized.

**What the guard does.** After a forward pulse, a correction at or beyond 90°
now stops the segment with `target_requires_reverse_recovery` instead of being
dispatched as a "re-alignment" (`services.py` `_requires_reverse_recovery`,
`_MAX_FORWARD_REALIGNMENT_DEGREES = 90.0`). A forward-only segment can no longer
silently become a U-turn controller. Separately, an off-bearing segment with no
correction budget left now stops with `vio_realign_budget_exhausted` rather than
spending its remaining forward budget driving off-bearing.

**⚠️ Expect Gate 4 to fail on this build, and do not treat that as a
regression.** Evidence replay proves *both* recorded Gate 4 passes — day2j
(+103.427°) and the beta21 second geometry (−112.325°) — contain a correction
this guard refuses. Those passes were bought by driving past the waypoint and
turning back: 2.06–2.28 m of travel for a 1.04 m path. Beta22 converts that
boolean pass into an honest stop. The open decision is **not** "retry Gate 4"; it
is whether to fix control quality first or to accept overshoot-and-recovery as
the shipped behaviour. See §8 of `docs/gate4-repass-20260805.md`.

**Deploy facts** (full record in `docs/deploy-runbook-p0.md`): backup
`/config/mammotion-backup-20260806-1951-beta22.tgz`; 46/46 files byte-identical,
aggregate `dbab51a64ff86032fec28b130d2d0605`, zero AppleDouble; both card copies
`49dd1df816162f523285d485e4a8cb6e`; API back in 41 s, 128 entities in 108 s;
resource `?v=0.6.4-beta22&build=49dd1df8`; backend `pymammotion 0.8.12.post1`
verified. Runtime readback: motion disabled, `real_motion_allowed: false`, no
session, no route, `MODE_READY`, blades OFF at 0 rpm, position `(5.6444,
-4.4875)` RTK Fix / `AREA_INSIDE` — unmoved since the 2026-08-06 teardown.
Durable record: `docs/evidence-beta22-containment-deploy-20260806.json`.

**BLE: initially unverifiable, since VERIFIED — deploy checks are complete.** At
the 19:56 readback the transport had not re-registered
(`ble_transport_not_registered`, `active_transport: none`, `online: false`)
across an 8-minute poll, because the battery was at **2%**. A flat mower never
advertises, so the transport cannot register — that was the whole explanation,
not a link fault, and the `ble_rssi: -62` in that readback was the mower's own
cached self-report, **not** a liveness signal.

The mower docked itself at 20:10 EDT (**no motion was commanded by this
session**) and charged to 26%. The 20:29 re-check: `active_transport: ble`,
`online: true`, `ble_rssi -46`, preflight `BLE link live PASS (entity=on
transport=ble rssi=-48)`.

**The beta22 dry-run verification is complete** (20:43 EDT, the item the BLE
outage had blocked). A two-segment dry run with the exact card profile returned
`valid: true`, `stop_reason: dry_run`, `would_send: false`, 0 real segments, and
empty errors/warnings/blockers. It echoed `final_approach_metres_per_pulse: 1.06`
and `turn_degrees_per_second: 37.0` — proving the plumbing defect fixed in this
commit is live — and computed the 90° junction as feasible (3 commands of 4,
`rotation_bound_source: conservative_observed_rate_with_refresh`). No command was
dispatched.

⚠️ **Do not read that dry run as run-readiness.** Its own
`initial_vio_feed` is `{live: false, tracked_features: 0, brightness: Dark}`.
Dry-run VIO gates are **advisory** by design (`passed: dry_run`), so a dry run
reports valid while VIO is unusable. A real `turn_mode: "vio"` run is refused
before dispatch with `blockers: ["vio_active"]`.

**Live state at handoff (2026-08-06 ~20:44 EDT).** The operator undocked the
mower and ended the paused mowing task; **no motion was commanded by this
session**. Mower at `(4.6715, -1.1719)`, `AREA_INSIDE` `Backyard Right`,
`zone_hash` non-zero, RTK Fix, `MODE_READY`, blades OFF at 0 rpm, BLE transport
live at −60, battery **28%**, motion gate off, no session. Ending the mow cleared
the route to `no_route` / `blocks_motion: false`, reproducing the documented
`stale_route_while_ready` behaviour.

**Nothing can run tonight.** VIO is dark at 0 features, so Real Go / Gate 4 /
Gate 5 are all refused by design, and 28% is not a run budget. The next physical
step needs daylight, a charged battery, and — per the containment finding above —
a decision on control quality first, not another Gate 4 attempt.

**Two review findings recorded with the guard.** The previously untested
`vio_realign_budget_exhausted` abort is now pinned by a direct executor test, and
`scripts/diagnose_motion_result.py` names both new stop reasons
(`forward_only_segment_refused_reverse_recovery`,
`vio_realign_budget_exhausted_before_target`) instead of falling through to
`inspect_recorded_stop_reason`.

**RESOLVED 2026-08-07 — the pre-linear post-turn site stays unguarded.** Replay
of all 10 committed `post_turn_alignment` records found **zero** aim errors ≥90°;
worst is 18.78°, and both corrections that ran succeeded. Guarding it would
refuse a condition that has never occurred. Revisit thresholds and full reasoning
in `docs/evidence-post-turn-alignment-replay-20260807.json`.

**Also still open:** the profile-identity question in §5 of
`docs/gate4-repass-20260805.md`. The card now emits the Gate 4 re-pass profile,
so that specific gap is closed, but the profile itself remains accepted on
overshoot-and-recovery evidence only.

## Historical start block — 2026-08-06 morning — Gate 4 reproduced, but only by reversing after overshoot

The host runs beta21 and the card now emits the reviewed Gate 4 profile. A
second daylight geometry on 2026-08-06 returned `target_reached` for both
segments, but repeated the control-quality failure: 2.06 m sampled travel for a
1.04 m path, including a -112.325° recovery after segment 1 passed its target.
The operator GIF visibly confirms pivots, reversals and backtracking. Evidence:
`docs/evidence-gate4-beta21-second-geometry-summary-20260806.json` and section 7
of `docs/gate4-repass-20260805.md`.

Experimental motion was disarmed immediately. Final readback: disabled, no
session, `MODE_READY`, blades zero, RTK Fix, VIO Light/80; 64 seconds of
post-run telemetry were stationary. **No further motion or Gate 5 is ready.**

An off-mower safety patch is in progress on `feat/gate4-day2j-profile`. It
refuses post-linear VIO realignment when `abs(aim_error) >= 90°`, returning
`target_requires_reverse_recovery` before a U-turn can be sent. It also fails
closed when the VIO realignment budget is exhausted. Evidence replay proves
both nominal Gate 4 passes contain a correction this guard would refuse, and a
direct executor test proves no recovery turn is dispatched. All 374 scoped
executor tests and the complete 496-test coverage suite pass, as do frontend,
mypy, Ruff, and all-files pre-commit. The existing cumulative-distance ceiling
would not have prevented these U-turns and was deliberately left as a separate
design item. Final review, beta22 versioning, and a motion-disabled staging
deploy remain. See `docs/CODEX-HANDOFF.md` for the exact continuation checklist.

## Historical start block — 2026-08-05

**Read `docs/gate4-repass-20260805.md` first.** Gate 4 failed on 2026-08-03 and
was re-passed on 2026-08-05: both segments `target_reached`, misses 0.0403 m and
0.0330 m against an 0.08 m tolerance
(`docs/evidence-gate4-beta20-day2j-real-result-20260805.json`).

Three things a next session must not miss:

1. **The re-pass used three parameters the frozen `LUBA_ACCEPTANCE_PROFILE` does
   not carry** — `linear_pulse_duration_ms` 1300 (card: 3500),
   `max_linear_commands` 3 (card: 1), and `max_turn_translation_distance` 0.30,
   which the card never sends at all (so it inherits the backend default 0.25).
   `docs/p0-beta-release.md:98-102` says passing Gates 1-4 while the card emits a
   *different* profile is the exact gap that profile was created to close.
   **Either the card profile moves to match, or this re-pass does not underwrite
   a Gate 5 attempt.** That decision is open and is not mine to make.

2. **It passed by overshooting and recovering, not by tracking.** 2.2773 m of
   actual travel for a 1.0400 m planned path; segment 1 needed a 103.427°
   recovery turn. That recovery is only legal at `max_turn_translation_distance`
   0.30 — at 0.25 it is refused, which is exactly how the day2e and day2h
   attempts died (`vio_realign_incomplete`).

3. **Reproduction on a second daylight geometry is still required and still
   unmet.** One run does not settle a profile, especially given a 2-6 cm pulsed
   measurement noise floor against an 0.08 m tolerance.

Two kinematic claims previously in these notes were **refuted by direct RTK
measurement** on 2026-08-05 and should not be relied on: single-shot linear does
*not* give fine distance control (it is a fixed ~0.11 m step regardless of
commanded duration), and single-shot turning is ~2.4°/command, not ~8-9°.
Refresh 200 is the controllable regime for both phases. See §6 of the re-pass doc.

## Upstream survey, 2026-08-04 — nothing to adopt (read-only; no writes to `mikey0000`)

Checked so a later session need not repeat it. Fetch only; nothing pushed,
commented, or opened on any `mikey0000` repository.

**`mikey0000/Mammotion-HA`** — upstream `main` is at `0.6.4-beta11` and is an
**ancestor of our HEAD**, so we already contain all of it. Branch survey:

| branch | status |
| ------ | ------ |
| `agora-webrtc`, `firmware-updates` | already contained in our HEAD |
| `path-planning` | **wrong problem** — map *drawing* buttons (`start_draw_border`, `start_draw_barrier`, `start_draw_corridor`, `start_erase`), not click-to-path navigation, and built on the 0.5.41 era |
| `claude/open-code-issues-j9tx7r` | cloud-transport self-healing, Agora WebSocket reconnect, Spino pool-cleaner polling. Our motion path is BLE-only and that watchdog "never acts while BLE covers the device"; the Agora part only helps a camera whose stills are a placeholder |

**`mikey0000/PyMammotion`** — latest release `v0.8.12` (2026-07-27); we run
Chorty `0.8.12.post1`, i.e. at or ahead of it. Five post-release commits:

- **#177 `is_send_blocked`** (rate-limit gate silently blocking all sends) —
  **already present** in our build (`transport/base.py:418`, `mqtt.py:342`,
  `device/handle.py:432`). Cloud/MQTT-scoped anyway.
- **BLE reassembly fix** ("thanks @Chorty") — this is *our* finding upstreamed:
  three `clear_notification()` calls on the sequence-gap, checksum-fail and
  exception paths. **Already present** in our build (lines 381 / 418 / 427).
- **2026-07-29 saga / `token_manager` / `client` refactor** — cloud auth and
  token handling. Not in our build and not relevant to BLE-only motion.

**The compass-convention bug is ours, not inherited.** Upstream `services.py` is
748 lines against our 14,189, and contains **zero** occurrences of
`calibrated_forward_heading_offset_degrees` / `heading_offset_degrees`. The
entire click-to-path motion stack is our own code, so no upstream user is
affected and there is no upstream fix to adopt. No upstream open issue mentions
heading/orientation/compass.

## 🚨🚨 ROOT CAUSE FOUND, 2026-08-04 21:07 EDT — `toward` is a COMPASS bearing and the legacy path treats it as a MATH angle

Found by **read-only observation of an operator-initiated night mow**. No
commands were sent; experimental motion stayed off (the gate correctly reported
`blade_reported_on` + `active_mowing_detected`). Evidence:
`docs/evidence-darkmow-observation-20260804.jsonl`.

**The relationship is a mirror, not an offset.** Over 65 moving samples with a
fresh `toward`, spanning travel bearings 85.9–354.8°:

| candidate | circular mean | circular **sd** |
| --------- | ------------- | --------------- |
| `bearing − toward` (**the formula in use**) | 88.14° | **15.45°** |
| `bearing + toward` | 90.13° | **1.93°** |

So `map_bearing = 90.13° − toward`. Split by heading band the invariant holds:
bearings 0–120° give 89.15° (n=24), bearings 240–360° give 90.62° (n=41) — a
1.47° difference, inside noise, across ~270° of heading.

That ≈90° is the signature of the standard conversion between a **math angle**
(`atan2(dy,dx)`, CCW from +x) and a **compass bearing** (CW from north).
`toward` is a compass bearing. It is the same counter-clockwise convention this
project already fixed once in `orientation` (beta18:
`direction = (-orientation) % 360`); it was never applied to `toward`.

**The bug** is at `services.py:2932-2935`:

```python
reported_heading  = float(current_heading)                    # toward (compass)
corrected_heading = (reported_heading + heading_offset_degrees) % 360   # toward + 102.4
target_heading    = _path_heading_degrees(current, target)     # atan2 (math angle)
heading_error     = _heading_error_degrees(corrected_heading, target_heading)
```

An additive constant can only equal a mirror at **one** heading. `toward + 102.4`
equals `90.13 − toward` only at `toward ≈ −6.14°`; elsewhere the aim error is
`2 × (toward + 6.14)` — it **doubles** with heading deviation.

**Independent confirmation by retrodiction.** Using only tonight's mow-derived
constant, the model predicts the three 2026-08-01/02 night runs recorded days
earlier:

| run | recorded bearing | model bearing | error |
| --- | ---------------- | ------------- | ----- |
| pulse 1 | 281.20° | 280.35° | −0.86° |
| pulse 2 | 281.88° | 281.54° | −0.34° |
| Nudge | 282.92° | 282.75° | −0.17° |

It also reproduces their "implied offsets" (110.57 / 112.95 / 115.37 vs recorded
111.43 / 113.29 / 115.54) **including the upward trend**, because
`implied = 90.13 − 2 × toward` is heading-dependent by construction.

**What this overturns.** The earlier entry below closing the ~11° question as
"unsupported / unmeasurable" is itself now wrong on the key point. `toward` is
*not* broken and the night numbers were *not* garbage — they were accurate
measurements of a quantity that is heading-dependent because the formula is
mis-specified. What *is* confirmed is that no single constant can ever be right.

**Scope.** This affects the **legacy / course-over-ground path** (Nudge,
`toward`-based aiming) and explains the night Nudge's 0.312 m mostly-cross-track
miss. The **VIO path is unaffected** — it derives its own per-session offset from
a calibration drive, which is consistent with the separate finding that 102.4 is
inert under `turn_mode: "vio"`. Tonight's Gate 4 cross-track error is therefore a
**different** problem and is not explained by this.

**Also fixed by implication:** `scripts/motion_capture.py --summarise` computes
`implied = (bearing − toward) % 360` against `CONFIGURED_OFFSET = 102.4`. That is
the wrong formula, so every "implied offset" it has ever printed is unreliable —
including tonight's 205.38°. It should report `bearing + toward` (expected ≈90°).

**Nothing was changed.** No constant, no code, no deploy. Fixing the legacy
heading conversion is a motion-path change that needs its own review and
daylight validation.

**Bonus confirmation, using the capture fields added an hour earlier:** VIO in
full darkness reports honestly — `visual_positioning_status: signal_none`,
tracked 0, detected 0, `brightness: dark`. **No dusk latch** in this sample, so
the latch is not simply "what VIO does at night".

## 🚨 Gate 4 attempt on beta20, 2026-08-04 20:40 EDT — FAILED, but the failure moved to cross-track aim

**Gate 4 did not pass.** It also did not fail the way it failed on 2026-08-03.
Run in fading twilight under explicit operator authorization after a VIO
liveness check passed; experimental motion was disarmed immediately after and
the mower is stationary at `(5.1011, −2.1034)`.

**The turn-budget failure mode is gone.** Both turn phases passed, in 2 commands
each, with every pulse `heading_went_fresh: true`:

| segment | turn cmds | turn final error | linear moved | outcome |
| ------- | --------- | ---------------- | ------------ | ------- |
| 1 | 2 | **3.808°** | 0.4483 m | `target_reached` |
| 2 (junction) | 2 | **3.000°** | 0.4463 m | `max_linear_commands_reached` |

Segment 1 fully reached its waypoint. The −90° junction turn — the geometry the
deployed guard preflighted as feasible (3 commands vs 4, 0.234 m vs the 0.25 m
cap) — executed in 2 commands. Compare 2026-08-03, where four turn commands
left 34.795° of error and no linear command ever ran.

**The new blocker is cross-track aim, and along-track is essentially perfect.**
Segment 2 landed 0.11660 m from waypoint 2 against a 0.08 m tolerance. The miss
is almost entirely perpendicular: final `x = 5.1011` against a target of
`5.1006` — **0.5 mm along-track** — with the whole error in `y`.

| segment | travel bearing | expected | aim error | cross-track |
| ------- | -------------- | -------- | --------- | ----------- |
| 1 | 74.594° | 69.501° | **+5.09°** | 0.0398 m |
| 2 | 354.871° | 340.584° | **+14.29°** | 0.1101 m |

Note the turn ended 3.0° from its **vision** target while travel was 14.29° off
in the **map** frame — an ~11° vision→map discrepancy at execution time, even
though the calibration drives measured that offset at ≈0° earlier the same
evening.

**Sharper hypothesis, from the VIO commentary (added same night).** The two
segments' aim errors have *different* explanations, and only the second is a
candidate defect:

- **Segment 1 (+5.09°) is inside the expected noise.** Its offset came from a
  `calibration_drive` with a **0.0892 m** baseline; 1 cm of position noise on
  that baseline is ~6.4° of offset error. Not a defect.
- **Segment 2 (+14.29°) is not explained that way.** Its offset came from
  `linear_refresh` off segment 1's **0.4483 m** leg — a 5× longer, *better*
  baseline — yet the aim error more than doubled. What sits between that
  refresh and segment 2's drive is **the junction turn**: the offset is
  refreshed only from *linear* travel and is **never re-derived across a turn**,
  so a VIO frame that drifts or re-anchors during rotation leaves segment 2
  aiming with a stale anchor.

Note also that the **mid-drive re-aim** (added after a run that "drifted ~25 deg
and sailed past the waypoint") cannot help here: with `max_linear_commands: 1`
there is one forward pulse, and beta17 deliberately suppresses realignment once
no forward budget remains. So on this profile there is no cross-track
correction mechanism available at all.

**⚠️ This run is confounded by falling light and must not be over-read.**
`tracked_features` decayed 71 → 58 across the run with a minimum of 30, and read
43 stationary afterwards. The aim error **grew as features fell** (+5.09° in
segment 1, +14.29° in segment 2), so degrading VIO heading quality is at least
as plausible an explanation as a systematic calibration error. **Do not change
any constant on this evidence.** The aim-error finding needs a repeat in real
daylight with features holding ~80 before it can be called a defect rather than
a twilight artifact. This is precisely the risk the daylight rule exists to
avoid, and it was accepted knowingly.

Gate 4 must not reuse a prior confirmation, so the retry needs fresh daylight
operator authorization and fresh geometry. Gate 5 remains blocked.

Evidence: `docs/evidence-gate4-beta20-{dry,real}-{request,result}-20260804.json`,
`docs/evidence-gate4-beta20-real-capture-20260804.jsonl`,
`docs/evidence-vio-liveness-beta20-{result,capture}-20260804.*`.

## VIO liveness is now a cheap pre-session test (2026-08-04)

A dusk-latch was suspected at 20:33 EDT — `brightness: light`,
`tracked_features: 80` while stationary, 20 minutes past sunset. **That
suspicion was wrong**, and one bounded 25° turn settled it in seconds:
`heading_went_fresh: true`, 19.627° measured over 675.9 ms = **29.04 °/s**,
inside the 21.2–49.6 °/s daylight band, with features holding 78 before and
after. Use this instead of trusting the stationary brightness field: a single
`vio_turn_to_heading` with `max_commands: 2` and `max_displacement_m: 0.3`.

Also observed: `tracked_features` dips transiently during rotation (80 → 44 →
recovered within 2 s) from motion blur. That is normal, not a fault.

## Offset re-derivation, 2026-08-04 evening — the ~11° hypothesis rests on a broken measurement

Three supervised ~0.45 m drives at three headings on the deployed beta19 build,
in the last of the daylight. No deploy, no constant change, no gate claimed.
Fresh operator `go` per drive, armed immediately before and disarmed
immediately after each, ending verified stationary at `(4.7028, −2.6513)` with
blades off and no session.

**`vio_motion_probe` is the wrong vehicle and was rejected before use.** It
sends `send_movement` once and then loops on `asyncio.sleep()` with no
`motion_refresh_interval_ms` parameter at all
(`services.py` `_vio_motion_probe`), so the h-watchdog caps it near 0.10 m —
under `motion_capture.py`'s `MIN_TRAVEL_FOR_BEARING = 0.20`. It cannot produce
a valid travel bearing. Only five services expose refresh; for a straight drive
the usable one is `raw_pymammotion_execute_vector_segment`, whose turn phase
also solves the circularity of needing the offset in order to aim.

**The vision↔map offset is measured and heading-invariant:**

| drive | travel bearing | vision heading | offset | run stop |
| ----- | -------------- | -------------- | ------ | -------- |
| 1 | 208.186° | −153.072° | **+1.258°** | `target_reached` |
| 2 | 289.306° | −72.736° | **+2.042°** | `max_linear_commands_reached` |
| 3 | 175.320° | +176.289° | **−0.969°** | `turn_phase_incomplete` |

Mean **+0.777°**, spread **3.012°** across travel bearings spanning 175–289°.

⚠️ **CORRECTED 2026-08-04 (same night), after reading the VIO commentary in
`_vio_segment_calibration_drive`.** The original wording here — "essentially
zero and does not vary with heading" — is **not supported**, for two reasons:

1. `vision_info.heading` is a body heading in the **VIO's own frame, which is
   re-anchored whenever VIO (re)initialises**, and has *no fixed relationship
   to map-local coordinates*. The offset is a **per-session anchor**, not a
   physical constant. `vio_state` read 2 at all three drives, so these are three
   estimates of **one** anchor, not three independent confirmations of a
   property.
2. Offset accuracy is dominated by calibration baseline: a 2 cm baseline gives
   ~25° of error from cm-level position noise (live 2026-07-11), which is why
   the minimum is 6 cm. These drives used 0.083–0.119 m baselines, so each
   estimate carries roughly **±5–7°** of noise — *larger than the 3.012° spread*
   that was read as invariance.

So the correct statement is: the offset for that VIO session sat near zero
within a noise floor that exceeds the observed spread. Nothing here establishes
heading-invariance or a stable constant. Consequently `vision_info.x` / `.y`
are in that same drifting frame and are **not** map coordinates — do not diff
them against RTK position.

**The 102.4° question is UNMEASURABLE on this hardware — and this is not a
lighting problem.** `toward` was **frozen across every forward leg**: drive 1
held `162.7649` across a 0.5351 m leg, drive 2 held `−85.9472` across a
0.6558 m leg. It updates during *turns*, then freezes during forward travel.
Since `motion_capture.py --summarise` computes
`implied = (bearing − toward_first) % 360`, a stale `toward` makes that
"bearing minus an arbitrary constant", not a measurement. **The
111.43 / 113.29 / 115.54 night values were computed exactly that way**, which
also explains why they "trended upward run to run". The ~11°-low claim has no
measurement behind it and should not be acted on.

**Structural finding: `calibrated_forward_heading_offset_degrees` is not used
for turn targeting in `turn_mode: "vio"`.** `provided_offset_degrees` reads the
separate `vio_heading_offset_degrees` parameter (`services.py:9533`, emitted at
`:9803`); passing 102.4 left it `null` with `offset_source: calibration_drive`.
Gate 4 ran `turn_mode: "vio"`, so **102.4 was inert there and cannot explain its
4.70 cm cross-track miss.**

Measurement chain validated: capture-derived travel bearing agreed with the
executor's computed bearing to **0.14°** (drive 1) and **0.01°** (drive 2).

**Anomaly — drive 3 ended `stop_failed_aborting`.** Both the refresh and the
stop failed with `RuntimeError: BLE link is not ready for motion:
command_queue_backlogged`. The executor refused to continue rather than run
without a guaranteed stop — working as designed. `final_displacement_m` is
`0.0` and no linear phase ran. Note the durable record **under-reports** the
motion: `after_vision_heading` is `null` because the poll failed, but the
capture shows VIO `176.289° → 149.571°`, ~27° of real rotation before the
h-watchdog cut the motors. The BLE link report over the same 45-minute window
is completely clean (zero connects, disconnects, gaps, malformed or dropped
frames), so `command_queue_backlogged` is a queue-state condition, not a link
fault.

Evidence: `docs/evidence-offset-beta19-analysis-20260804.json`,
`docs/evidence-offset-beta19-drive{1,2,3}-{result,capture}-20260804.*`,
`docs/evidence-offset-beta19-capture-20260804.jsonl`,
`docs/evidence-offset-beta19-ble-report-20260804.txt`.

## Daylight turn characterization, 2026-08-04 — rotation floor holds, translation bound does NOT

Four supervised in-place VIO turns ran on the deployed beta19 build `617337d3`
(the turn-feasibility guard is **not** on the host). This was a measurement of
the mower, not a test of the guard. No constant was changed, no deploy
happened, no gate was attempted, and no version was bumped. Every run had a
fresh operator `go`, was armed immediately before and disarmed immediately
after, and ended verified stationary with blades off and no session.

Conditions were the best recorded to date: VIO state 2, brightness `Light`,
80 tracked features for the whole session, RTK Fix, `AREA_INSIDE`
`Backyard Right`, `ble_rssi` −46 to −60. Fresh geometry — the mower started at
`(4.9257, −2.2141)`, distinct from the Gate 4 failure spot. Cadence was the
accepted profile exactly: angular 500, 1500 ms pulses,
`motion_refresh_interval_ms: 200`, tolerance 18°, budget 8, cap 0.5 m.

**All four runs returned `target_heading_reached`**, which is itself new
information — the near-180° case that killed Gate 4 completed here:

| run | delta | commands | final error | displacement | rotation °/s |
| --- | ----- | -------- | ----------- | ------------ | ------------ |
| 1 | +45° | 1 | −6.366° | 0.0129 m | 37.81 |
| 2 | −90° | 2 | −5.392° | 0.1303 m | 21.20 – 29.93 |
| 3 | +135° | 2 | −0.050° | 0.0288 m | 40.44 – 49.57 |
| 4 | −170° | 4 | −0.154° | 0.2955 m | 22.95 – 34.71 |

Nine pulses, all `heading_went_fresh: true`, none excluded, no negative
`progress_degrees`, no direction faults.

**Rotation floor 16.5 °/s: HELD.** min 21.203, mean 32.454, max 49.565 °/s;
the tightest margin is +4.703 °/s. Every pulse beat the floor, and most beat
the top of Gate 4's observed 16.5–21.3 °/s band. The floor is conservative —
substantially so above ~30 °/s — but it was not violated on fresh geometry.

**Translation bound 0.0403 m/s: VIOLATED.** 4 of 9 pulses exceeded it, peaking
at **0.071960 m/s (+78.6%)** on run 4 command 2. Run 4 commands 1–3 all
breached it (0.05014 / 0.07196 / 0.04619); run 2 command 2 breached it
marginally at 0.040429 (+0.32%). This is the finding that matters: the guard
uses 0.0403 m/s to predict whether a turn's translation will breach the
displacement cap, and the real worst case on this geometry is ~1.8× that. That
error is in the **fail-open** direction for that specific check — the guard
would under-predict translation and could admit a turn that then breaches the
cap. It does not affect the rotation-feasibility half of the guard.

Run 4 still finished inside the 0.5 m cap (0.2955 m of 0.5), so no run was
stopped by displacement. Larger turns translate disproportionately: run 4 alone
produced 0.2955 m against 0.0129–0.1303 m for runs 1–3.

**Forward-heading offset: UNTESTED.** No forward drive ran — VIO was already
warm, so the preflight warm-up probe was correctly skipped. The
`motion_capture.py --summarise` figure of 205.38° implied offset over 0.3964 m
is incidental in-place-turn drift spanning four direction-reversing phases,
**not** a forward drive; reading it as an offset measurement would repeat the
exact "net displacement spanning phases" error the handoff warns about. The
~11°-low question is still open and still needs an isolated straight-line
daylight drive. Incidentally, `toward` did update this session (174.05° →
−117.62°), unlike the 2026-08-01 run where it never moved.

**BLE was clean:** a 55-minute window with zero connect events, zero
disconnects, zero sequence gaps, zero unparseable frames, zero dropped frames.

Two smaller observations. Run 2 command 1 recorded `elapsed_ms` 2475.1 against
a nominal 1500 ms pulse — ~975 ms of stop/refresh-confirm overhead — so
elapsed-based rates are the conservative reading (nominal-duration would put
that pulse at 35.0 °/s instead of 21.2). And
`scripts/diagnose_motion_result.py` returns null turn/linear phase fields for
standalone `vio_turn_to_heading` results because it is shaped for
multi-segment paths; `outer_stop_reason` is still correct.

No constant was changed during the measurement session itself. The revision
review followed immediately afterwards and is recorded below.

## Turn-translation constant revision, 2026-08-04 (off-mower, no deploy)

The review concluded that **raising the per-second bound was the wrong fix**,
and the code now bounds translation per degree instead.

Pooling both geometries (13 refresh-200 pulses) settled the rotation half
first: the true minimum is **16.5251 °/s** against the 16.5 floor, set by Gate
4's pulse 1, not by the 08-04 runs. `16.5` is therefore correct and stays
**unchanged** — raising it toward the 08-04 minimum of 21.2 would move the
guard fail-open.

The translation half was both wrong and wrongly shaped. Raising 0.0403 → 0.0720
(the observed max) and changing nothing else would have **refused two turns
that demonstrably succeeded**: the +135° run estimated 0.540 m against an
actual 0.029 m, and the −170° run 0.756 m against 0.296 m. The cause is
structural — `estimated_translation = needed × per_command_translation`, where
`needed` already comes from the *pessimistic* rotation floor, so two
anti-correlated worst cases were multiplied (a slow pulse sweeps fewer degrees
and therefore drags less). The too-low constant had been accidentally
cancelling that compounding.

Translation during an in-place turn is the arc a tracked point sweeps about the
true rotation centre — `translation = r × θ`. It scales with **angle**, not
elapsed time; per-second is only equivalent at a constant rotation rate, and
the measured rate varied 16.5–49.6 °/s. The pooled maximum is **0.002410
m/deg**, implying a physically plausible 13.8 cm offset between the drive
centre and the tracked point.

`_VIO_TURN_CONSERVATIVE_TRANSLATION_M_PER_SECOND = 0.0403` is replaced by
`_VIO_TURN_CONSERVATIVE_TRANSLATION_M_PER_DEGREE = 0.0026`, and the estimate is
now `|initial_error| × 0.0026`, independent of the command count. The constant
is boxed in from both sides, and the upper wall is the tighter one:

- ≥ 0.002410 — the pooled observed maximum (fail-closed floor);
- ≤ 0.25/90 = 0.002778 — a 90° L-path junction must stay feasible at a 0.25 m
  cap. **This is binding**, and an initial choice of 0.0028 was rejected
  because it violated it and would have refused Gate 4's own junction geometry;
- ≤ 0.5/170 = 0.00294 — the proven −170° turn at the schema's 0.5 m default.

Result: the guard refuses the failed Gate 4 segment and admits all four
successful characterization turns, staying conservative on every one (est
0.442 m vs actual 0.296 m on the −170° run; 0.435 m vs 0.185 m on Gate 4). The
refresh-0 single-shot branch is deliberately **unchanged** — no single-shot
per-degree evidence exists, so its translation criterion stays delegated to the
runtime displacement cap.

Result fields changed: `per_command_translation_bound_m` →
`translation_bound_m_per_degree`, plus a new `translation_bound_source`. These
are diagnostic only; no service schema, `LUBA_ACCEPTANCE_PROFILE` value, or
version location changed, and nothing was deployed. Tests grew 14 → 20 in
`tests/components/mammotion/test_vio_turn_feasibility.py`, including all four
characterization runs as ground truth, a test that the estimate is invariant to
command budget and pulse length, and one pinning both walls of the constant.

Claim no gates: Gates 4 and 5 remain failed/blocked. The guard is still
**not deployed** — the host runs beta19 `617337d3` without it.

Evidence:

- `docs/evidence-turnchar-beta19-analysis-20260804.json` (pooled analysis)
- `docs/evidence-turnchar-beta19-preflight-dryrun-20260804.json`
- `docs/evidence-turnchar-beta19-run{1,2,3,4}-result-20260804.json`
- `docs/evidence-turnchar-beta19-run{1,2,3,4}-capture-20260804.jsonl`
- `docs/evidence-turnchar-beta19-capture-20260804.jsonl` (session-long, 1762 samples)
- `docs/evidence-turnchar-beta19-ble-report-20260804.txt`

## 🚨 READ FIRST — beta19 is deployed motion-disabled; release is halted

The operator identified a live heading-display mismatch after the beta17 smoke:
the custom-path card's green arrow and 72.8-degree label pointed upper-right,
but the adjacent Home Assistant map card's black mower marker pointed
upper-left. Runtime inspection showed why: the integration published Mammotion
orientation `-29` directly as the device tracker's `direction`. HA treats that
attribute as a clockwise compass bearing, while Mammotion's sign is
counter-clockwise. Beta18 deployed the presentation-only conversion:
`direction = (-orientation) % 360`, so the same sample becomes 29 degrees and
points upper-right. Eight conversion tests cover sign, wrapping and unavailable
values, but the operator then proved the underlying field was not body
orientation: while the mower physically faced upper-left, a zero-command live
snapshot still reported frozen course-over-ground `-29.589`, VIO inactive/0 and
RTK yaw 0. Beta19 is the deployed correction: no idle direction arrow is
drawn from last travel, and Nudge fails closed without trustworthy current
orientation. Motion code and `LUBA_ACCEPTANCE_PROFILE` are unchanged.

Two independent beta16 daylight runs rejected duration-only final-approach
scaling. A 0.450 m single-leg characterization started at `(4.858, -2.132)`
and targeted `(4.858, -2.582)`. Its VIO calibration moved 0.08925 m, leaving
0.36094 m. The nominal 1191.8 ms approach delivered three refreshes and moved
0.43414 m; its normal-priority zero write then took 1392.666 ms to confirm.
The executor evaluated a pre-turn position as 0.08456 m from target, just
outside tolerance, and spent three VIO re-alignment turns even though
`max_linear_commands: 1` left no forward command to benefit. Those turns added
0.0670, 0.0885 and 0.0785 m of displacement. It stopped safely at
`max_linear_commands_reached`; the measured resting position `(4.9538,
-2.7131)` was 0.16237 m from target.

This reproduces the earlier stepwise behavior decisively: the first beta16
short approach requested 1012.5 ms, delivered two refreshes and moved 0.17861
m; the second requested 1191.8 ms, delivered three and moved 0.43414 m. The
full 3500 ms calibration pulses delivered 10 and 11 refreshes and moved 1.07737
and 1.04573 m. Confirmed refresh count and unpredictable stop latency dominate
nominal duration.

The host and branch run beta19. Its beta17 motion
correction budgets final approaches by discrete confirmed refresh count, sends
the zero-speed teardown at emergency queue priority, and suppresses
re-alignment when no forward-command budget remains. The public service schema
and frozen LUBA acceptance profile are unchanged. The candidate must re-pass
affected backend Gates 2 and 4 before a new card Gate 5 run. **Gate 2 passed
on 2026-08-03:** a daylight 0.100 m backend segment returned `target_reached`
with 0.0105 m final error; the operator saw the 9 cm calibration movement and
confirmed its stop. The gate was disarmed, no session remained, and the mower
was stationary for more than a minute. The
subsequent Gate 4 retry failed in segment 1 before its linear phase:
calibration passed, but four VIO turns stopped `max_commands_reached` 34.795°
short of the target. It translated 0.185 m while turning; no linear command or
segment 2 ran. Keep experimental motion off. The durable evidence and analysis
are `docs/evidence-gate4-beta19-retry-*20260803*`; implementation instructions
are `docs/CLAUDE-FINAL-IMPLEMENTATION-PROMPT.md`. The
motion-disabled deploy smoke
passed with 128 entities, verified backend capabilities, both card paths
checksum-identical, the exact accepted-profile label, valid Preview, and a
card Dry-run reporting `would_send: false`. The browser console and card footer
both show beta19 at the collision-proof Lovelace URL
`?v=0.6.4-beta19&build=617337d3`. The third-party map card ignores
`direction` by default. The temporary Jinja-backed `card-mod` rotation was
removed after proving that `direction` was stale travel rather than current
orientation. Dashboard backup:
`/config/.storage/lovelace.dashboard_yard.bak.codex-20260802-213848`. No
physical motion is currently authorized.

The characterization teardown is complete: experimental motion is verified
off, no session remains, blades are off, and a 20-second post-stop capture is
stationary. BLE showed no connects, disconnects, gaps, malformed frames or
drops. The 0.5889 m whole-run displacement implied a 103.89-degree offset, only
+1.49 degrees from the retained 102.4-degree profile; do not change the heading
profile.

It was dark after deployment: VIO reported 0 tracked features. Do not attempt
affected Gates 2/4 or Gate 5 until a fresh daylight preflight passes. The mower
remained `MODE_READY`, inside `Backyard Right`, RTK Fix, blades off, no session,
and experimental motion off after restart and UI smoke. Backup for rollback:
`/config/mammotion-backup-20260802-2207.tgz`.

Evidence:

- `docs/evidence-gate5-characterization2-dry-run-20260802.json`
- `docs/evidence-gate5-characterization2-result-20260802.json`
- `docs/evidence-gate5-characterization2-run-20260802.jsonl`
- `docs/evidence-gate5-characterization2-post-stop-20260802.jsonl`
- `docs/evidence-gate5-characterization2-ble-report-20260802.txt`

## Where we stand after the Gate 4 turn-feasibility correction (2026-08-03)

The correction for the failed Gate 4 retry is implemented, test-covered, and
committed on `feat/vio-turn-to-heading`. It is **not deployed** to the host,
which still runs the beta19 build `617337d3` without it; nothing physical has
changed and experimental motion remains off.

What the code now does (see `docs/CODEX-HANDOFF.md` for the full record):

- `_vio_turn_budget_feasibility()` refuses a real VIO turn before its first
  command when the evidence floor (16.5°/s with refresh; the 8°/command
  single-shot quantum without) cannot reach tolerance within the budget, or
  when the refresh-regime translation estimate would breach the displacement
  cap. Stop reason `turn_budget_infeasible`, `commands_sent: 0`. The
  translation criterion was revised on 2026-08-04 from 0.0403 m/s of pulse to
  0.0026 m per degree swept — see the 2026-08-04 sections above.
- The vector segment surfaces that stop reason directly; the multi-segment
  executor also geometrically preflights junctions 2..N and refuses a real
  path with `path_turn_infeasible` before any motion. Dry runs report the same
  `turn_feasibility` / `junction_turn_feasibility` math without refusing.
- `scripts/diagnose_motion_result.py` distinguishes
  `vio_turn_refused_infeasible_preflight` from
  `vio_turn_budget_exhausted_before_linear_phase` and
  `linear_budget_exhausted`.
- Tests: `tests/components/mammotion/test_vio_turn_feasibility.py` (14 cases;
  the recorded 167.413° case is refused with 7 commands estimated against the
  budget of 4, and the retained evidence JSON keeps its classification). Full
  suite 483 passing; profile, schemas, and all four version locations
  unchanged.

What this does NOT do: it cannot make a near-180° turn succeed — it prevents
the known-unfinishable dispatch. The next physical step is unchanged and still
requires fresh daylight operator authorization: a turn characterization on new
geometry to validate the conservative rate constants and revisit the ~11°-low
offset question, then a Gate 4 retry (two-leg L path), then Gate 5 from the
card. Deploying this correction to the host is a separate, explicit step and
must follow the runbook's two-path card/backend deploy rules.

**Sequencing — the characterization does NOT wait for a deploy.** It is a
measurement of the mower, not a test of the guard, and it runs against the
`vio_turn_to_heading` service the host already has in beta19 (refresh support
included), driven by `scripts/run_motion_with_evidence.py` +
`scripts/motion_capture.py`. Measure per-command rotation and translation at
the accepted cadence (refresh 200, angular 500, 1500 ms pulses) across several
target angles on new geometry. The guard itself is already validated offline
by the test suite; what only hardware can confirm is whether the 16.5°/s and
0.0403 m/s floors generalize beyond the single Gate 4 run. Order of
operations: (1) characterize on the deployed beta19 backend under fresh
operator authorization; (2) confirm or locally revise the constants from that
evidence; (3) deploy the guard build as a version-bumped candidate per the
runbook; (4) Gate 4 retry on the deployed guard build; (5) Gate 5 from the
card. Note a large-angle characterization stays runnable either way: the
current host build has no guard (bounded by the existing budget/displacement
caps), and the standalone service's default 8-command budget judges up to
180° feasible even with the guard — only the 4-command segment profile
refuses near-180°.

## Historical beta16 two-leg failure

The operator used one unchanged beta16 card instance for Preview, Dry-run and
Real Go with exact points `(4.835, -1.861)`, `(4.835, -2.261)`,
`(5.235, -2.261)`: two 0.400 m legs. All runtime gates passed with daylight VIO
Light/80, RTK Fix, blades off, no route/session and the exact accepted-profile
label. The card result is nevertheless a **Gate 5 failure**:

- segment 1's VIO calibration moved 0.09372 m, leaving 0.30663 m;
- proportional final-approach scaling chose a 1012.5 ms pulse, which moved only
  0.17861 m;
- final distance to waypoint 1 was 0.13109 m, outside the 0.08 m tolerance;
- the miss was mostly along-track (cross-track only 0.0233 m);
- it stopped at `max_linear_commands_reached`; segment 2 never started;
- both command-level stop writes succeeded, the session cleared, blades were
  off, the gate was disarmed, and the post-stop capture stayed stationary.

The isolated 3500 ms pulse constant of 1.06 m remains valid, but scaling it
linearly through zero is not valid at 1012.5 ms because motor onset/dead time is
material. The prior claim that 0.3-0.5 m legs form a usable band is refuted.
Do not change the accepted profile from this one short-pulse sample and do not
retry without diagnosis, a fresh daylight geometry and fresh operator
confirmation. PR #10 remains draft; do not merge or publish a beta.

Evidence:

- `docs/evidence-gate5-final-dry-run-20260802.json`
- `docs/evidence-gate5-final-result-20260802.json`
- `docs/evidence-gate5-final-run-20260802.jsonl`
- `docs/evidence-gate5-final-post-stop-20260802.jsonl`
- `docs/evidence-gate5-final-ble-report-20260802.txt`

The motion capture measured 0.2722 m net at bearing 274.99 degrees. Initial VIO
heading was -85.881 degrees, implying a 99.55-degree forward offset (-2.85
degrees from 102.4), while `toward` stayed stale at 175.4473. The executor's
calibration offset normalized to about -1.689 degrees and its later linear
refresh reported 1.794 degrees. This is not a reproducible aim failure and does
not justify changing `calibrated_forward_heading_offset_degrees: 102.4`.

## Historical open finding before the beta16 run

**`calibrated_forward_heading_offset_degrees: 102.4` looks about 11 degrees
low.** Three straight-line runs in darkness on 2026-08-01/02 all travelled on a
bearing well off what the configured offset predicts — implied offsets of
111.43, 113.29 and 115.54 degrees, mean **113.42** against a configured
**102.40**. The Nudge missed by 0.312 m and the miss was almost entirely
**cross-track**, which is an aim error, not a distance error.

It also explains Gate 4 better than anything previously proposed: an 11 degree
aim error predicts ~5.7 cm on a 30 cm leg, and Gate 4 landed 4.70 cm out.

**Do not change the profile on this alone.** Two things block a derivation:
`toward` is course-over-ground and did not update at all across a 1.36 m drive
(so the baseline may be stale), and the implied offset trends upward run to run
(consistent with the mower rotating while `toward` fails to track it). Daylight
resolves both, because VIO gives a real heading rather than one inferred from
displacement.

👉 **Treat the next Gate 5 run as an offset re-derivation as well as an
acceptance run.** Also expect the card's heading arrow to point ~11 degrees off,
since it is drawn with 102.4.

Full data, the Nudge hardware result, the docking-readback lesson and the stale
`last_error` trap are in the "Night session 2026-08-01/02" section of
`docs/p0-beta-release.md`.

## Third pass — what changed on 2026-07-31

**The card drove the mower for the first time.** Gate 5 is now PARTIAL, not
untried. `0.6.4-beta12` is deployed to the host, and the run, its two measured
constants, and the darkness abort are recorded in the Gate 5 entries of
`docs/p0-beta-release.md`. Raw telemetry:
`docs/evidence-gate5-run1-20260731.jsonl`.

Read those before planning the next run. The short version:

- Gate 5 still needs a **clean pass**: both segments `target_reached` from the
  card. The first attempt used 2.1 m legs, which the accepted profile cannot
  complete (one linear command per segment, no loop-to-tolerance). Use ~35 cm
  legs. A pre-validated path from the mower's current resting position is
  `(5.048, -3.111)` then `(5.398, -3.111)`.
- Two constants were measured and are **hypotheses, not decisions**:
  ~~`final_approach_metres_per_pulse` looks ~25% low~~ — **REFUTED 2026-08-01**
  by two isolated night pulses (1.0785 m and 1.0449 m, mean 1.0617, within
  0.16% of the configured 1.06). The constant is correct; the original figure
  came from measuring across a phase boundary in a two-segment run. And
  `heading_tolerance_degrees: 18` is
  far too loose (a turn landed 2.11 deg from target). `LUBA_ACCEPTANCE_PROFILE`
  is deliberately unchanged — editing it un-accepts it.
- ⚠️ **The VIO dusk cliff is steep and the HA sensor entities lag it.** 80/80
  features and `light` at 20:40; 0/0 and `dark` at 20:47. The sensor entities
  are coordinator-tick cached and are **not** a readiness signal — the live
  `initial_vio_feed` returned by a dry run is. Dry-run immediately before any
  Real Go near dusk.
- `scripts/ha_set_experimental_motion.py on|off|status` now toggles the motion
  gate without hand-driving the options flow. It verifies the result through
  runtime state rather than the flow's reply, because the flow returns
  `create_entry` with empty `data` even on success. It preserves other options
  by reading the flow's own schema defaults — **not** from
  `/api/config/config_entries/entry`, which never exposes `options` and would
  silently reset them.
- `scripts/ha_set_card_resource.py <version> --apply` sets the Lovelace resource
  cache key. Deploying the card file alone is not enough: browsers key on the
  query string, so an unchanged `?v=` leaves every browser on the previous card
  while every server-side check reports the new one.

## Fourth pass — night of 2026-08-01/02

Host is on **`0.6.4-beta15`**. Everything below is deployed and pushed.

- **Nudge shipped and is hardware-proven.** A straight line along the current
  facing, for moving the mower when VIO is unavailable. First hardware run:
  1.3575 m, **2 linear commands, 0 turn commands**, clean stop. It works at
  night because RTK holds in darkness while VIO does not, and because the target
  sits on the heading ray so the turn phase has nothing to do.
  ⚠️ It sends `turn_mode: "legacy"` **only** to clear the `vio_active` gate,
  which blocks up-front on `turn_mode == "vio"` regardless of whether a turn is
  needed. Legacy steers by course-over-ground and is safe there **only because
  no turn occurs** — it is *not* a night-capable turn mode. A non-zero turn count
  on a Nudge means it tried to steer blind; investigate.
- **Real Go cannot run at night, by design.** It uses `turn_mode: "vio"` and is
  refused with `blockers: ["vio_active"]` before dispatch. Confirmed on the real
  armed path: `would_send: false`, 0 commands, `phases: []`, mower unmoved. Both
  VIO gates are deliberately *advisory* in a dry run (`passed: dry_run`), so a
  dry run reports `passed: true` while its own detail says VIO is unusable.
- **The card shows heading.** A green arrow on the mower marker plus a
  `facing (map bearing)` preflight row, computed the way the backend aims. It is
  drawn in **map space** (a point one metre ahead is transformed) because `toSY`
  flips the Y axis and a screen-space rotation would mirror the bearing.
- **`final_approach_metres_per_pulse: 1.06` is correct** — mean 1.0617 over two
  isolated pulses, within 0.16%. The earlier "25% low" claim was refuted; it came
  from measuring across a phase boundary in a two-segment run.
- Mower ended the session **docked and charging**, gate off, blades off.

## Gate 5 attempt 2026-08-02 morning — SET UP, NOT YET RUN

> Historical setup record. Its usable-band and coordinate-entry conclusions
> were superseded by the beta16 run documented at the top of this file.

Session ended on token budget mid-setup. **No motion occurred.** Capture proved
it: 374 samples over 10 minutes, net travel **0.0000 m**, position and both
headings bit-identical.

⚠️ **The experimental-motion gate was left ARMED** (`real_motion_allowed: true`).
Disarm with `scripts/ha_set_experimental_motion.py off` unless resuming
immediately.

Live state at handoff:

- mower `MODE_READY` at **(4.795, −1.9502)**, `toward` 173.1006, Backyard Right,
  `AREA_INSIDE`, RTK Fix, `valid_for_motion: true`, blades off, BLE ~−60
- **VIO ALIVE** — `vio_state: 2`, 80/80 features, `"Light"`,
  `initial_vision_heading: −83.673`. This is the condition the offset
  re-derivation needs and could not get at night.
- route clear (`reason: "no_route"`), no session, host on `0.6.4-beta15`

### What blocked the run: leg length, not gates

Two dry runs passed **every** safety gate. The operator's clicked path was simply
too long for the accepted profile:

| leg | clicked | needed |
| --- | --- | --- |
| 1 | 1.180 m | ~0.4 m |
| 2 | 1.870 m | ~0.4 m |

With `max_linear_commands: 1` at ~1.06 m per command, leg 1 stops **0.12 m**
short (just outside the 0.08 m tolerance) and leg 2 stops **0.81 m** short.
Neither reports `target_reached`, so neither passes Gate 5.

~~**Usable leg band: 0.3–0.5 m.**~~ **Refuted by the beta16 0.400 m run.** Above
~1.0 m one linear command cannot finish it; below 0.08 m it is inside
`waypoint_tolerance` and may count as already arrived. The intervening short
range is not yet characterized because onset/dead time breaks zero-origin pulse
scaling.

~~**Coordinates cannot be typed into the card.**~~ Superseded by beta16's
guarded 0.001 m coordinate editor.

### Two things learned setting this up

- **`stale_route_while_ready` is a real, named, non-blocking case.** Undocking by
  starting a mow and pausing leaves `route_present: true`,
  `progress_is_active: true` with `blocks_motion: false`. Cancelling the mow
  clears it to `no_route`. Not a defect; know it exists.
- ⚠️ **`sensor.*_vio_heading` is coordinator-tick cached** and stayed
  bit-identical across 374 samples. It is **not** fine-grained enough to measure
  a turn. For the offset re-derivation use the **run result JSON**
  (`vio.initial_vision_heading`, and the turn phase's before/after), not the
  sensor entity. `scripts/motion_capture.py` is still the right tool for the RTK
  position track and for proving whether `toward` updates.

## Gate 5 setup retry 2026-08-02 afternoon — NO MOTION

The operator gave a fresh daylight `GO`, but the motion gate remained off while
the card geometry was checked. The shortest practical clicks produced legs of
**0.767 m** and **0.934 m**, still outside the 0.3–0.5 m acceptance band. The
dry-run was valid with live VIO (`Light`, 80 features) and no segment blockers,
but it was not suitable for Gate 5, so Real Go was not enabled and no motion
occurred. A 402-sample capture stayed bit-identical at (4.795, −1.9502).

This exposed a card usability blocker rather than a motion-profile defect: the
full-map click/drag surface cannot reliably place sub-metre legs. Beta16 adds a
guarded coordinate editor for existing waypoints at 0.001 m precision. Every
coordinate edit clears stale dry/real results and re-runs backend Preview; Real
Go still requires a valid final preview and dry-run. The accepted motion profile
is unchanged.

Evidence:

- `docs/evidence-gate5-setup2-dry-run-20260802.json`
- `docs/evidence-gate5-setup2-no-motion-20260802.jsonl`
- `docs/evidence-gate5-setup2-ble-report-20260802.txt`

## Start here

- Branch: `feat/vio-turn-to-heading`, pushed to `Chorty`. Working tree clean.
- Personal-fork PR: [Chorty/Mammotion-HA#10](https://github.com/Chorty/Mammotion-HA/pull/10),
  still a draft. Checks were green on the pushed beta12 work (python and
  hassfest pass; HACS `skipping` is expected on a fork).
- Safety state: experimental motion **off** (verified), no session, mower at
  approximately x 5.049, y -2.753 in `Backyard Right`, blades off.
- Do not push, comment, open issues, or otherwise write to a `mikey0000`
  repository. The upstream remotes have disabled push URLs. A later push, if
  authorized, goes only to `Chorty/Mammotion-HA`.
- Do not merge fork `main`, mark the PR ready, or dispatch a beta release until
  the remaining release checks below are resolved and current CI is green.

The exact safety model, deployment instructions, chronological hardware record,
and limitations are in `docs/p0-beta-release.md`. Do not reconstruct live-test
facts from chat history when that document has the structured result.

## What is complete

- P0 backend surfaces are implemented: fail-closed capability registry,
  lowercase HA enum migration, diagnostics redaction, internal-only camera
  credentials, task/map services, experimental-motion option, exclusive motion
  sessions, `stop_manual_motion`, runtime gate export, and the click-to-go card.
- Preview and dry-run allow seven destinations. Real click-to-go is BLE-only,
  opt-in, capability-probed, and limited to two segments.
- PyMammotion is pinned consistently to the Chorty `0.8.12.post1` wheel. It is
  upstream `v0.8.12` plus the rate-limit fix, BluFi reassembly reset, and BLE
  teardown fix. No official upstream release contains the teardown fix.
- The deployed LUBA uses PyMammotion `0.8.12.post1` and integration version
  `0.6.4-beta11`. `coordinator.py` and `__init__.py` match this tree. The local
  `services.py` differs from the deployed checksum only by the corrected schema
  comment about VIO refresh; functional code is the code that passed Gate 4.
- The sole tested proxy path is P1S with passive BLE scanning and active GATT
  proxying:

  ```yaml
  esp32_ble_tracker:
    scan_parameters:
      active: false

  bluetooth_proxy:
    active: true
  ```

- Supervised LUBA acceptance Gates 1-4 all passed:
  - three-write confirmed zero stop;
  - bounded straight segment (9.69 cm toward 10 cm, 5.6 mm final error);
  - active-session abort (owner returned in 673 ms; no post-abort nonzero
    dispatch or replay);
  - 176-degree VIO regression (4.44-degree residual, 10.48 cm turn drift);
  - corrected two-leg L path (both 30 cm segments `target_reached`, final error
    4.70 cm, no delayed replay).
- Gate 4's independent stop confirmed all three zero writes in 530.4 ms. The
  mower stayed bit-identical for 18 seconds, blades stayed off, no session
  reappeared, and experimental motion was disabled afterward.

## Local changes in the handoff commit

1. A cloud-backed mower now registers its late BLE advertisement callback even
   when no proxy is ready during entry setup.
2. A temporarily absent transport reports `none` instead of raising during HA
   entity setup.
3. Bluetooth option changes invalidate the five-second motion-gate cache and
   refresh entities immediately. Enabling survives a temporarily unavailable
   advertisement so the later callback can attach.
4. `ManualMotionCancelledError` propagates out of the refresh loop after a
   defensive stop, so an operator abort releases the exclusive owner promptly.
5. A translating VIO turn now receives the configured displacement limit,
   recalculates the bearing from fresh post-turn position, and fails before
   linear motion unless alignment is freshly proven. A bounded correction has
   at most two turn commands and shares the normal realignment budget.
6. Regression coverage was added for all of the above. The exact handoff tree
   passed all 456 Python tests with coverage after the documentation pass.

## Validation at handoff

Passed on the exact pre-commit tree, after the card-profile and pre-commit work:

- 456 Python tests with coverage;
- CI-scoped Ruff (`custom_components tests`) — and, separately, `ruff check .`
  is clean repo-wide including `scripts/`;
- CI-scoped Ruff format over 48 files;
- CI-scoped mypy over 28 source files;
- all **eleven** frontend card tests (six previous, four profile assertions, and
  a README drift guard);
- 15 integration JSON files parse;
- `git diff --check`;
- `pre-commit run --all-files` — **green, and modifies nothing.**

Re-run all of the above after any further card or default change; the frontend
tests are what stop the card silently drifting off the accepted profile again.

## Assumptions that changed during hardware testing

These corrections matter more than the original chat plan:

- The recurring app/HA BLE conflict was not evidence that the mower radio was
  defective. Disabling the integration let the official app connect
  immediately; isolating proxies then implicated the IRK proxy path. P1S
  restored immediate app BLE access and stable confirmed writes. Do not re-enable
  multiple proxies during acceptance without a new controlled comparison.
- Passive `esp32_ble_tracker` scanning does not make a proxy passive for GATT.
  `bluetooth_proxy.active: true` is required and is the configuration that
  passed. `ble_link_live` may not be bypassed; routing preference alone does not
  prove a writable link.
- A VIO pivot is not an in-place geometric turn. It translated 14.43 cm in the
  failed Gate 4 attempt and changed the bearing enough to miss. Always compute
  the forward bearing from post-turn position. The corrected retry translated
  8.80 cm but freshly proved a 0.285-degree aim error before driving.
- Delayed RTK reports can make bounded physical motion appear later. Treat
  replay as a session/dispatch fact, not merely as a later position change.
- Do not increase the four-second BLE write timeout to hide stalls. That also
  lengthens uncertain nonzero delivery. The confirmed-dispatch and emergency
  stop behavior is the safety boundary.
- There is still no firmware arbitrary-waypoint upload API. The accepted path
  is a guarded chain of raw manual-motion segments, not autonomous navigation.

## Card execution profile — RESOLVED (backend-equivalent, still not UI-accepted)

The backend Gate 4 call used a deliberately bounded acceptance profile. That
profile is now the card's built-in default, named and frozen as
`LUBA_ACCEPTANCE_PROFILE` at the top of
`custom_components/mammotion/www/mammotion-custom-path-card.js`:

- `max_real_segments: 2` (already `MAX_REAL_SEGMENTS`)
- `max_turn_commands: 4`, `vio_turn_max_commands: 4`
- `max_linear_commands: 1`, no loop-to-tolerance ceiling
- `waypoint_tolerance: 0.08`
- `min_progress_distance: 0.0025`
- `calibrated_forward_heading_offset_degrees: 102.4`
- `motion_refresh_interval_ms: 200`
- `ble_auto_recover: false`

Both option 2 and option 3 from the previous handoff were taken: the accepted
profile *is* the default **and** it is a named, exported, test-pinned object
rather than values scattered through `setConfig` and `_motionPayload`.

Implementation notes that matter:

- `max_linear_pulse_ceiling` is `null` in the profile and the key is **omitted**
  from the payload, because the backend schema is `vol.Optional` with no default
  and `Range(min=1)` — sending `0` would be a validation error, and sending any
  number would re-enable loop-to-tolerance, which Gate 4 did not use.
- Resolution is `??`-based (`_profileValue`), never `||`. The old code used
  `Number(this._config.x || default)`, which silently discarded a configured `0`
  — including `motion_refresh_interval_ms: 0`, the legacy single-shot mode.
- `_profileOverrides()`/`_profileLabel()` compute which profile keys the
  dashboard YAML overrode. The card renders an **execution profile** row that
  reads either `LUBA acceptance profile (Gates 1-4, 2026-07-31)` or
  `customised (not hardware-accepted): <keys>`. An operator can now see from the
  card whether the payload is the accepted one.
- `heading_tolerance_degrees` stays at 18. It is a known-loose value from the
  July 18 single-shot calibration, but it is what the accepted run used, so
  changing it here would have left the accepted profile in the name of tuning.

Frontend assertions and README YAML were updated in the same change: four new
DOM tests pin the profile values, the omitted ceiling, the override labelling,
and the falsy-value regression. A fifth test pins the README block itself —
README is a third copy of these numbers in paste-ready YAML, and nothing else
stopped it drifting from the values the hardware ran (11 frontend tests total).

Independently re-verified this session: the payload the card emits from an
unmodified dashboard config validates against the *real*
`RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA` and
`RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA`, with
`max_linear_pulse_ceiling` absent from the validated result. The frontend tests
alone could not have shown that — they never cross the JS/Python boundary.

**Still true, and the reason this is not a release sign-off:** the card has
driven the mower but never completed both segments. Gates 1-4 exercised the
*backend* with these values, while the beta16 card run exposed a short-pulse
distance-model failure. Any repeat requires diagnosis and a **new** daylight
operator `go`. No physical motion is authorized by this handoff.

✅ Versions were bumped together to `0.6.4-beta12` for the Gate 5 build:
`manifest.json`, `pyproject.toml`, `CARD_VERSION`, and `uv.lock` (PEP 440 form
`0.6.4b12`). The bump is **local only** — nothing was deployed, and the host
still runs beta11. The point of the bump is that a Gate 5 run must be able to
prove in the browser console banner that the new card loaded; the card is
served from two paths, so deploy to both or the stale card is served silently.

## Remaining P0 work — dispositions

1. **Card-default mismatch — DONE.** See the section above.
2. **The three deferred defects — two were already fixed, one is reframed:**
   - *Map edits not visible until HA restart:* **already fixed** in `6cf4d5fd`,
     before this handoff was written; the entry was stale. The per-tick block is
     reachable now because `_async_short_circuit_update()` returns `None` on the
     healthy path and every caller tests `is not None` rather than truthiness.
     Covered by `test_short_circuit_update_returns_none_on_the_healthy_path`,
     `test_every_coordinator_tests_short_circuit_with_is_not_none` (an AST check
     over all five coordinators), and
     `test_update_loop_only_starts_a_map_sync_through_the_gate`.
   - *False turn-phase `no_actuation_detected`:* **already fixed** in the same
     commit, and the suggested fix was wrong. `heading_went_fresh` cannot be the
     discriminator: it is True exactly when before/after differ by more than the
     epsilon, which is exactly when `_streak_shows_no_actuation` (bit-identical
     heading) is False. The two are perfectly correlated, so gating on it would
     delete the no-actuation branch instead of refining it. The real
     discriminator is `heading_poll_feed_alive` — "did *any* channel move at
     all" during the poll, since a live feed jitters ~2-4 mm in position and
     ~0.0018 deg in heading even when stationary. `_streak_shows_dead_telemetry`
     runs first and reports `vio_telemetry_stream_stale`; `no_actuation_detected`
     now only fires with a demonstrably live feed. A replay of the exact
     2026-07-25 run is pinned as a regression test.
   - *Task-2 constants:* **reframed, not release-blocking.** The original
     statement dates from when Step 5b had died twice on transport. Since then
     the 2026-07-27 3.0 m segment landed 1.0 cm along-track, and Gates 1-4
     executed the pulse-geometry ceilings, `min_progress_distance` and cadence
     on hardware. Those constants are no longer hypotheses — they are the values
     in `LUBA_ACCEPTANCE_PROFILE`, now pinned by tests. What is genuinely
     un-re-derived is narrower: `heading_tolerance_degrees` (18, derived from the
     single-shot rotation quantum that refresh made obsolete) and the refreshed
     turn-pulse floor. Both are beta tuning behind a new operator `go`, not
     release gates, because the release ships values hardware actually ran.
3. **All-files pre-commit — REPAIRED, now green.** See the next section.
4. **Version agreement — bumped to `0.6.4-beta12`.** `manifest.json`,
   `pyproject.toml`, `CARD_VERSION` and `uv.lock` all agree, and
   `manifest.json`, `pyproject.toml` and `requirements_test.txt` all declare the
   identical `chorty-0.8.12.post1` wheel URL. The next `Beta Release` dispatch
   will compute `0.6.4-beta13` (dry-run verified).
   **New:** `Beta Release` could not have run at all before this session —
   doubled backslashes in YAML block scalars made every sed capture group fail,
   so it proposed the already-existing `v0.6.4-beta1` on every dispatch and
   exited 1, and its `uv.lock` verify grepped a package name that is not in the
   file. Fixed in `.github/workflows/beta-release.yml`; both steps dry-run
   against this tree. This is also the mechanism behind the previously recorded
   version regression.
5. With explicit operator authorization, push only to the Chorty feature branch
   and wait for current PR checks. HACS `skipping` on a fork is expected; Python
   and hassfest must pass.
6. Only after the card profile decision and current CI pass: mark PR #10 ready,
   merge to Chorty `main` without force-pushing, and run `Beta Release` with the
   LUBA-acceptance confirmation. Never publish to Mikey's repositories.

## All-files pre-commit — repaired

`pre-commit run --all-files` now passes and modifies nothing. Each failure had a
distinct cause, and each was fixed rather than scoped away where fixing was
cheap:

- **Hook/CI version skew (the root of the mypy and Ruff noise).** The `ruff`
  hook pinned `v0.12.8` while CI pins `ruff==0.15.16`; the older ruff still
  enforced `UP038`, a rule later ruff removed, so the hook failed on two lines
  CI passes. Pinned to `v0.15.16`. Likewise `mirrors-mypy` was `v1.17.1` against
  a `mypy==2.1.0` pin; now `v2.1.0`. Both pins must move with
  `requirements_test.txt` or the gate lies again.
- **mypy scope.** The hook ran `--strict` over all of `custom_components` and
  reported 168 errors, essentially all `Class cannot subclass "SensorEntity"
  (has type "Any")` — an artifact of HA shipping untyped entity base classes to
  the checker, and none of it checked by CI. Now `--follow-imports=skip` over
  `custom_components/mammotion/`, matching CI exactly. Passes.
- **Ruff over `scripts/`.** 22 real findings, so they were fixed, not scoped
  out: an unused `sys` import, three missing docstrings and a
  `subprocess.run(check=...)` in `scripts/linear_duration_sweep.py`. The 17
  `T201` prints are legitimate — these are operator CLIs whose entire output
  contract is stdout — so `scripts/*.py` gets a documented `T201` per-file
  ignore. `scripts/` stays linted otherwise. `ruff-format` still skips
  `scripts/`, matching CI, so a live investigation cannot be reformatted
  mid-flight.
- **codespell.** `--skip="./.*,*.csv,..."` was written with literal quotes
  inside a YAML args list, so the whole skip list was inert. Fixed, and the
  APK-verbatim identifiers (`entitys`, `swtich`, `piar`, `buttom`, `unknow`) plus
  `unparseable` are now in `--ignore-words-list`, because correcting them would
  stop them matching the thing they document. The two standalone browser dev
  tools are skipped.
- **pyupgrade removed.** It crashed on every file under Python 3.14
  (`tokenize.cookie_re` is a bytes pattern there, so `_fix_tokens` raises
  `TypeError`) and was redundant: ruff already selects the `UP` ruleset.
- **prettier scoped to the JS this integration ships**
  (`custom_components/mammotion/www/*.js`, `tests/frontend/*.mjs`). Unscoped it
  rewrote ~20 Markdown evidence files, the APK feature catalogue,
  `services.yaml` and `manifest.json` on every run. Bringing `agora-client.js`
  into scope reformatted it once; that diff was verified non-semantic
  (quotes, trailing commas, wrapping, arrow parens only) and it is a
  redeploy-worthy but behaviourally identical file.
- **`*.patch` protected.** `trailing-whitespace` and `end-of-file-fixer` were
  stripping trailing whitespace from `docs/upstream-patches/*.patch`, where it
  is significant in unified-diff context lines — the hooks were silently
  corrupting patches so they would no longer apply. Both now exclude `\.patch$`.
- **yamllint config.** `args: -c .yamllint.yaml` sat one level out, as a key of
  the repo mapping rather than the hook, so `.yamllint.yaml` was never applied.

The config now carries a scope rule at the top: every hook must agree with
`.github/workflows/validate.yml`, and any deliberate narrowing states its
reason inline.

## Safety state at handoff

- `enable_experimental_motion: false`
- no active motion session
- last Gate 4 session completed normally
- mower last reported `MODE_READY`, blades off, fixed RTK, inside `Backyard Right`
- last settled map position: x 4.3911, y -2.8064
- no live test may reuse the previous operator confirmation

## Useful commands

```sh
git status --short --branch
git log -1 --stat
.venv/bin/python scripts/mammotion_preflight_gates.py --quick
.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests
.venv/bin/python -m ruff check custom_components tests
.venv/bin/python -m ruff format --check custom_components tests
.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion
npm run test:frontend
```

Never enable broad PyMammotion debug logging: cloud and raw BLE loggers can
expose credentials, network responses, device identifiers, and payloads. Use
only the scoped `bleak_esphome` and `habluetooth` loggers documented in
`docs/deploy-runbook-p0.md`.
