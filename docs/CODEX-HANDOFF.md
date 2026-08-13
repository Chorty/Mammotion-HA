# Handoff — beta52 night v1; items 15–17 measured and contained

Night v1 off-mower items 1–14 and the item-17 runtime diagnostics are deployed
as `0.6.4-beta52`. Both card paths match md5
`9512f504f4b861488e98f4d29ced6e4f`; Lovelace resource is
`?v=0.6.4-beta52&build=9512f504`; `services.py` matches local md5
`7c94607698ce6d9f55a4fd4a1a30f85f`; pymammotion is `0.8.12.post1`.
Final local verification produced 662 pytest and 39 frontend tests; ruff,
format, mypy, and all pre-commit hooks passed. The acceptance-profile values
were not changed. The motion gate read back `enabled: false`,
`real_motion_allowed: false`, with no active session, and no movement command
was sent during deployment.

Plan §7 item 15 is complete. The one night-branch turn pulse measured −54.2208°
at angular −500 / 1,500 ms / six 200 ms refresh writes, with 0.07459 m turn
translation. The following forward pulse measured 0.43648 m but an 81.416° aim
error; the night guard refused further motion. Evidence and measured/inferred
separation are in `docs/night-segment-turn-quantum-20260813.md`.

Item 16 is also complete. One angular-only +500 / 1,500 ms pulse produced 73
concurrent runtime samples. `toward` remained 43.1856 through the refreshed
window and then arrived as one +36.3064° step; no intermediate headings were
observed at roughly 0.1-second cadence. See
`docs/night-toward-latency-20260813.md`.

A later read-only autonomous-mow capture recorded 291 samples over 179.895 s
and three complete vendor pivots. `toward` streamed progressive headings during
continuous vendor motion; 40 usable moving steps gave `bearing + toward =
90.57°`, circular SD 2.02°. This narrows the stepwise item-16 result to the
bounded pulse/report path. See `docs/autonomous-mow-observation-20260813.md`.

Item 17 is complete. One backward-only pulse moved 0.418536 m on bearing
96.433921° while `toward` remained exactly 173.1023°. The mirror-derived body
heading was 277.0277°, so reverse was predicted at 97.0277°, only 0.593779°
from measured course. This settles `toward` as body heading under reverse.
RapidState `fuse_status` remained 0 `NO_POSE` in all 81 records and was
non-informative on this manual path. Raw evidence is commit `ff8e1f09`; analysis
is `docs/night-reverse-heading-20260813.md`.

The gate is off, no session is active, the mower is `MODE_READY`, RTK Fix, BLE
live, and blades zero. Item 18 remains unauthorised; item 17 does not resolve
item 15's separate forward-course mismatch. Every further physical run needs
fresh explicit supervised-motion authorization.

---

# Handoff prompt — beta30; GATE 5 PASSED, all five gates complete

## 🏁 2026-08-08 — READ FIRST. Gate 5 passed twice.

Host runs `0.6.4-beta30`. Gate off, no session, blades off, battery ~22%.

Two card-driven two-segment runs, both completing both segments with the
accepted profile, zero errors, zero reverse-recovery, no overshoot. Landings
0.0485 / 0.0836 / 0.0558 / **0.1449** m, all inside the adopted 0.15 m
tolerance; the worst would have failed at the old 0.08 m. Evidence:
`docs/evidence-gate5-PASSED-20260808.json`.

`waypoint_tolerance` is now **0.15** in `LUBA_ACCEPTANCE_PROFILE` with the full
§4 re-pinning done (card profile, payload, frontend pin, README, label).

⚠️ **Two fragilities remain, and the pass does not remove them:**

1. **Turn budget has no margin near 90°** — attempt 5 used 4 of `max 4` turn
   commands while turn rate varied **2.6×** across identical 1500 ms pulses.
2. **The BLE `TimeoutError` is intermittent, not fixed** — it failed attempt 3 at
   80.6°, yet attempt 5 completed *larger* turns. Attempt 5 ran with degraded BLE
   (writes median 540 ms, stops to 1819 ms vs a 77–230 ms norm) without tripping
   it, placing the timeout as the **tail of an observable latency distribution**.

Also settled 2026-08-08: the ~1031 ms position feed is a **link property**, not a
motion-pattern artifact (confirmed against the mower's own autonomous dock
return at 1.15 s median); the stop-lead work item is **dropped** (60 samples,
18× spread — a constant cannot correct a variable); per-segment reach is
**~0.9–1.5 m**.

---

# (earlier) Handoff prompt — beta29; base station answers, slow-tier test queued for daylight

## ☀️ 2026-08-07 late — READ FIRST. Host runs beta29; one test is queued.

Gate **off**, no session, blades off, mower stationary and `area_inside`.
Worktree clean and pushed to `Chorty`. All four version sites agree at
`0.6.4-beta29` / `0.6.4b29`.

**Queued and ready:** the slow-tier landing validation,
`docs/plan-slow-tier-validation-20260807.md` (operator-authorized). It was
attempted 2026-08-07 night and **refused before any motion command**.
⚠️ The `vio_active` gate keys off `turn_mode == "vio"` **unconditionally**, not
off whether a turn is needed; `_VIO_TURN_MODES` is `("vio", "legacy")` only and
`passed = dry_run or calibration_will_warm` requires a bright scene. **Closed-loop
segments cannot run after dark, by design.** 11 of 12 gates passed. Re-run it in
daylight, together with the turn-quantum work (VIO is alive only then).

**Settled tonight — do not re-litigate:**

1. **RTK freshness: closed permanently.** Legitimate quiet reaches **3745 s
   (62.4 min)** against a ~3 h fault. Set too low twice already. No active
   liveness probe exists, so no threshold is sitable.
2. **The base station ANSWERS** `request_basestation_info_t` — the earlier
   negative was a wrong read path. Baseline under Fix: `rtk_status` 1, 28 sats,
   WiFi −72 dBm, coordinates 34.0245718145 / −84.7698523612.
3. **`score_info` is `null`** — `base_moved` / `base_moving` are never populated
   by this hardware. **That avenue is closed.**
4. **Chain is `internet → base (WiFi) → LoRa E22 → mower`.** Base reports
   `rtk_over_internet`, mower `rtk_over_datalink`. The survey hypothesis is
   **demoted**; an upstream outage matches the observed signature exactly.
5. **Drop the stop-lead item.** 60 samples: median 229 ms, p90 461, max 1393,
   stdev 186 — an 18× spread. A constant cannot correct a variable.
6. **Mirror heading relation confirmed:** `map_bearing = 90.13 − toward` agreed
   within **0.89°** with a measured 0.45 m displacement; the code's 116.5
   constant mis-aimed by **9.80°**.

**Tolerance:** two independent routes both give **~0.15 m**; **0.08 m is
arithmetically unreachable at the fast tier** (21–33 cm feed staleness, and
beta23 proved no faster feed is available). The decision is the operator's, and
changing it un-accepts the profile.

Evidence: `docs/evidence-slow-tier-validation-20260807.json`,
`docs/evidence-basestation-query-20260807.json`,
`docs/evidence-rtk-watch-20260807.json`,
`docs/vendor-tool-analysis-20260807.md` §6.

---

# (earlier) Handoff prompt — beta23; the faster-feed fix is refuted, tolerance is next

## 🚨 2026-08-07 night — RTK freshness is UNVERIFIABLE (beta26); read before any RTK work

Two conclusions from earlier the same day were **refuted by measurement**:

1. **The forced report burst cannot detect a latch.** Repeated with RTK healthy
   and `Fix`: 49 messages, **zero** RTK channel updates, age still climbing —
   indistinguishable from latched. No positive liveness probe exists.
2. **No staleness threshold works.** A healthy Fix-locked stationary mower went
   **3573 s** without an RTK payload change; it updates ~hourly at rest while the
   observed fault lasted ~3 h. Both 300 s and 1800 s false-blocked, and 1800 s
   shipped live in beta25.

**beta26 inverts it:** age is reported for auditing and **never blocks**; the
**quality gate** (`rtk_not_precise`, non-Fix, override `allow_degraded_rtk`) is
the real guard and catches the fault actually seen — a latched *Float*.
**Do not re-add an age blocker.** Reasoning: `docs/rtk-hardening-plan-20260807.md`.

**Verified:** 534 recorded `Fix` states, zero prior `Float`/`Single` — 2026-08-07
was the first episode, so **past gate results are not tainted**.

## ✅ 2026-08-07 18:39 EDT — RTK RESTORED TO FIX by a base-station power cycle

RTK read `Float` from 15:40 until the operator power-cycled the dock and RTK
base station. At **18:39:08** the group refreshed: `rtk_position` → **fix**,
`position_level` 2 → **1** (so 1 = Fix, 2 = Float), `satellites_robot` 26 → 23,
`l1_signal_quality` 35 → 29. The fault was the **base station's own
survey/solution** — rover reception was healthy (24 co-viewed satellites)
throughout and a rover-side `sync_rtk_and_dock` could not clear it.
**Precision work is unblocked.** Record:
`docs/evidence-rtk-float-investigation-20260807.json`.

⚠️ **Two measurement traps, both walked into this session:**
1. **The RTK sensors LATCH.** They were frozen 15:40→18:39, so polling re-read
   one stale value that looked like a stable signal. Caught by forcing 50
   reports and seeing **no** RTK entity refresh. `rtk_position` comes from
   `basestation_info.rtk_status` (`sensor.py:570`) and holds its last value
   rather than going unavailable. Claims of Float *persistence*, and the resync
   and relocation as RTK tests, were invalidated and corrected in the evidence.
2. **`rtk_correction_age` / `rtk_signal_quality` are dead fields** — unpopulated
   since the 00:20 restart. Their zeros mean "no data", not "no corrections".

**Method rule:** bit-identical values across polls are evidence of a dead feed
until proven otherwise; check `last_updated` against an entity known to be
moving. A freshness guard on RTK is worth more than any threshold choice.

**OPEN DECISION — the motion gate does not check RTK at all.**
`_is_valid_motion_position` validates coordinates, zero-pose, `pos_type` and
`zone_hash` only, so `valid_for_motion` read `True` throughout this Float session
and a real precision run would have been permitted on 13.9 cm positioning. Every
prior gate ran at Fix by luck. Choose before the next run: hard blocker
(`rtk_not_fixed`), warn-and-record, or blocker with an explicit
`allow_degraded_rtk` override. Hard-blocking is fail-closed but halts all motion
testing while this fault persists — which is why it was not chosen unilaterally.

**Schema note learned the hard way:** `max_linear_commands` is capped at **3** in
the vector-segment schema. Covering more ground needs longer pulses, not more
commands.

## 🚨 2026-08-07 — read this before the beta22 sections below

The host runs `0.6.4-beta23`, motion-disabled, adding only a read-only
`report_stream_probe` diagnostic. Deploy record is in `docs/deploy-runbook-p0.md`.

**The report-rate hypothesis is REFUTED.** The wire `period` field defaults to
1000 ms and nothing had ever lowered it, so a cheap ~5× feed improvement looked
available. Interleaving 1000 ms and 200 ms three times each gave mean medians of
706.9 ms and 590.3 ms — ratio **1.20** against **5.0** if honoured — and the
within-setting spread (445–640 ms) exceeds the between-setting difference
(117 ms). Across 203 pooled intervals at every setting, p90 is 1002 ms and max is
1107 ms. Evidence: `docs/evidence-report-rate-probe-20260807.json`.

⚠️ **Bounded by one limitation:** `last_report_at` stamps *every* LubaMsg, not
just periodic position reports, so this measures total inbound traffic. The
refutation stands, but the **position-report cadence itself remains unmeasured**.
Close it by stamping arrivals per `RptInfoType` or diffing the parsed position
payload.

**Consequences.** Drop the "expose `period`, request 200 ms" work. Keep the other
half: the Gate 4/5 executor still never holds a continuous subscription during
motion while four other motion paths do — a defect independent of rate. The
tolerance decision (§4 of `docs/direction-review-20260806.md`) is now the leading
item.

**Also resolved 2026-08-07:** the pre-linear post-turn guard site stays
unguarded — zero of 10 committed `post_turn_alignment` records reached 90°, worst
18.78°. See `docs/evidence-post-turn-alignment-replay-20260807.json`.

⚠️ **`docs/direction-review-20260806.md` carries a correction.** Its headline
"1.11 s position update interval" was the *sampler's* interval, not the feed's;
that claim and everything derived from it are withdrawn. The surviving argument
is in its §7 (code findings). Read the correction box before citing that review.

---

# Handoff prompt — beta22 contains the Gate 4 U-turn; decide control quality next

Paste the block below to resume. It is written to be self-contained; everything
it references is committed on `feat/gate4-day2j-profile` in
`Chorty/Mammotion-HA`.

---

You are resuming P0 completion on the Mammotion-HA repo, branch
`feat/gate4-day2j-profile` (remote `origin` = `Chorty/Mammotion-HA`). The branch
is one review commit ahead of `feat/vio-turn-to-heading`; confirm the worktree
is clean and inspect that commit before making another change.

**Read first, in this order:** `CLAUDE.md`, `docs/NEXT-SESSION.md` (its
2026-08-06 evening block), `docs/gate4-repass-20260805.md`, then the matching
section of `docs/p0-beta-release.md`. Confirm
`git status --short --branch` is clean and inspect `git log -1 --stat` before
editing. Do not reconstruct live-test facts from chat history; those documents
hold the structured record.

## Immediate safety state

The 2026-08-06 test teardown verified experimental motion **OFF**, no active
session, `MODE_READY`, blades zero, RTK Fix, and VIO Light/80, and the beta22
deploy readback that evening confirmed the same. Still check first:

```sh
set -a && source .env && set +a
scripts/ha_set_experimental_motion.py status
```

If it reports `enabled: True` and you are not about to run, turn it off with
`scripts/ha_set_experimental_motion.py off`.

The host runs the motion-disabled `0.6.4-beta22` staging candidate. The final
2026-08-06 position was `(5.6444, -4.4875)` inside `Backyard Right`; 59 samples
over 64 seconds measured 0.0000 m post-run movement, and the post-deploy
readback found the mower still there. Experimental motion must remain off unless
a new, exact test receives fresh operator authorization.

## Where this stands — 2026-08-06 evening

Gate 4 reproduced its **boolean** pass on a second 0.52 m x 0.52 m daylight L:
both segments returned `target_reached`, with misses 0.07195 m and 0.03374 m.
It did not reproduce clean tracking. Segment 1 used 0.9743 m of linear travel
on its 0.52 m leg and recovery errors of -22.179°, -45.510°, and -112.325°.
Dense telemetry sampled about 2.06 m cumulative travel for the 1.04 m path.
The operator's 119.5-second GIF independently shows repeated pivots, reversals,
and partial backtracking rather than two controlled perpendicular legs. Read
`docs/evidence-gate4-beta21-second-geometry-summary-20260806.json` and section 7
of `docs/gate4-repass-20260805.md`.

**The containment guard is now written, tested, versioned as beta22, and
deployed motion-disabled.** The executor used to call a >=90° correction a VIO
"re-alignment", silently converting a forward-only segment into a U-turn after
passing its target. `_requires_reverse_recovery()` now sits at the post-linear
realignment decision: at `abs(aim_error) >= 90°` the segment stops with
`target_requires_reverse_recovery` before dispatching the U-turn. It also stops
with `vio_realign_budget_exhausted` instead of continuing forward after the
realignment budget is gone. Tests replay both Gate 4 passes and directly prove
that only the initial turn runs in the >=90° case.

Verification completed on the shipped tree:

- the complete coverage suite passes: **499 tests**;
- all 19 frontend tests, scoped mypy, Ruff lint, Ruff format, and all-files
  pre-commit pass, and pre-commit modified nothing.

Two review findings were closed with the guard: the previously untested
`vio_realign_budget_exhausted` abort is pinned by a direct executor test, and
`scripts/diagnose_motion_result.py` now classifies both new stop reasons
(`forward_only_segment_refused_reverse_recovery`,
`vio_realign_budget_exhausted_before_target`) instead of falling through to
`inspect_recorded_stop_reason`.

The existing `linear_distance_ceiling_factor` was reviewed but deliberately not
expanded here: it is enforced only in loop-to-tolerance mode, checked after a
pulse, and defaults to 2.0, so it would not have prevented either recorded
U-turn. That is a separate cumulative-travel design change, not a substitute
for the geometric guard.

## The next decision is control quality, not another gate attempt

Do **not** proceed to Gate 5, and do not queue a Gate 4 retry expecting a pass.
The guard refuses a correction present in **both** recorded Gate 4 passes, so a
Gate 4 run on beta22 will most likely stop `target_requires_reverse_recovery`
where beta20/beta21 reported `target_reached`. That is the intended trade — an
honest failure replacing a boolean pass bought by a U-turn — and it means the
open question is how to stop overshooting in the first place:

1. Lead the stop by `speed × latency` rather than managing overshoot with pulse
   sizing, which only works when the leg length is known in advance
   (§8.3 of `docs/gate4-repass-20260805.md`). Measured interrupted-stop overshoot
   was 0.15–0.26 m across day2d/e/f/h.
2. Or accept overshoot-and-recovery as shipped behaviour, which means relaxing
   the guard deliberately and documenting why — not leaving it to the executor
   to decide silently.

**RESOLVED 2026-08-07 — leave the pre-linear post-turn site unguarded.** An
offline replay of every committed multi-segment result (11 files, 10
`post_turn_alignment` records) found **zero** post-turn aim errors at or beyond
90°. The worst is **18.78°** — about a fifth of the boundary — the largest turn
displacement is 0.207 m, and both corrections that ran succeeded. Guarding it
would refuse a condition that has never occurred, and at this site a ≥90°
reading more likely indicates a bad VIO offset than a genuine overshoot, because
the mower has not yet driven forward on that segment. Revisit if a run exceeds
~45°, if `max_turn_translation_distance` goes above 0.30, or if legs shorter than
~0.30 m are adopted. Evidence:
`docs/evidence-post-turn-alignment-replay-20260807.json`.

## beta22 deploy — 2026-08-06 19:51-19:56 EDT, motion-disabled, no motion commanded

Backup `/config/mammotion-backup-20260806-1951-beta22.tgz`. All **46**
integration files byte-identical to the local tree (aggregate
`dbab51a64ff86032fec28b130d2d0605`), zero AppleDouble entries; both card copies
`49dd1df816162f523285d485e4a8cb6e`. HA API returned in **41 s**, all **128**
Mammotion entities in **108 s**, no `setup_error`. Lovelace resource read back as
`?v=0.6.4-beta22&build=49dd1df8`. Container backend verified
`pymammotion 0.8.12.post1`. Runtime readback: `real_motion_allowed: false`,
`enabled: false`, no active session, no route, `MODE_READY`, blade `OFF` at 0 rpm
with `blade_safe_for_motion: true`, position `(5.6444, -4.4875)` RTK `Fix` /
`AREA_INSIDE`. `LUBA_ACCEPTANCE_PROFILE` is byte-identical and the
execution-profile label still reads exactly
`LUBA acceptance profile (Gate 4 re-pass, 2026-08-05)`; only `services.py` and
`CARD_VERSION` changed relative to beta21. Full record in
`docs/deploy-runbook-p0.md` and
`docs/evidence-beta22-containment-deploy-20260806.json`.

**BLE: initially unverifiable, since VERIFIED.** At the 19:56 readback the
transport had not re-registered (`ble_transport_not_registered`,
`active_transport: none`, `online: false`) across an 8-minute poll, because the
mower battery was at **2%** — a flat mower never advertises, so the transport
cannot register. That was the whole explanation, not a link fault; the
`ble_rssi: -62` in that readback was the mower's own cached self-report, which is
**not** a liveness signal.

The mower docked itself at 20:10 EDT (**no motion was commanded by this
session**) and charged to 26%. The 20:29 re-check is clean: `active_transport:
ble`, `online: true`, `ble_rssi -46`, and the read-only preflight reports
`BLE link live PASS (entity=on transport=ble rssi=-48)` — matching the documented
excellent dock proxy coverage. **The deploy verification is complete.**

**The beta22 dry-run verification is complete** (20:43 EDT). A two-segment dry
run with the exact card profile returned `valid: true`, `stop_reason: dry_run`,
`would_send: false`, 0 real segments, empty errors/warnings/blockers; it echoed
`final_approach_metres_per_pulse: 1.06` and `turn_degrees_per_second: 37.0`
(proving the plumbing fix is live) and judged the 90° junction feasible at 3
commands of 4. No command was dispatched.

⚠️ That dry run is **not** run-readiness: its own `initial_vio_feed` is
`{live: false, tracked_features: 0, brightness: Dark}`, and dry-run VIO gates are
advisory by design. A real `turn_mode: "vio"` run is refused before dispatch with
`blockers: ["vio_active"]`.

**Live state at handoff (~20:44 EDT).** The operator undocked the mower and ended
the paused mowing task; **no motion was commanded by this session**. Mower at
`(4.6715, -1.1719)`, `AREA_INSIDE` `Backyard Right`, `zone_hash` non-zero, RTK
Fix, `MODE_READY`, blades OFF at 0 rpm, BLE live at −60, battery **28%**, gate
off, no session, route cleared to `no_route`. **Nothing can run tonight**: VIO is
dark at 0 features and 28% is not a run budget.

Two operational notes from that session. `scripts/linear_duration_sweep.py`
needs `custom_components.mammotion` at DEBUG or its `ble_alive()` guard aborts
on a false negative — it greps HA logs for `BLETransport send`, which are not
emitted at INFO; the log level was returned to `info` at the end of the session.
The same script also needs `scripts/map.json`, which is deliberately not in the
repo — regenerate it from the `get_map_data` service. Separately, a full disk
destroyed one run's durable result mid-session (`day2b`): the evidence runner
keeps capture and request but cannot survive ENOSPC, so check free space before
a live run.

The beta19 deploy smoke passed: 128 Mammotion entities, backend verified against
`pymammotion 0.8.12.post1`, BLE live, both card paths checksum-identical, exact
accepted-profile label, valid Preview, and a card Dry-run with `valid: true`,
`would_send: false`, and `stop_reason: dry_run`. Real Go remained disabled and
the card was reset to zero waypoints. The live resource URL is
`?v=0.6.4-beta19&build=617337d3`. Rollback backup:
`/config/mammotion-backup-20260802-2207.tgz`.

The beta18 direction conclusion was incomplete. A zero-command snapshot taken
while the operator visually confirmed the mower faced upper-left showed frozen
course-over-ground `toward: -29.589`, `location.orientation: -29`, VIO inactive
with heading 0, and RTK yaw 0. Neither available feed reports the stationary
body orientation. Beta19 keeps the projected last-travel bearing as explicit
diagnostic text but draws no orientation arrow from it and disables Nudge
unless a trustworthy current orientation exists. The earlier third-party-map
`card-mod` rotation was removed and read back successfully. Dashboard backup:
`/config/.storage/lovelace.dashboard_yard.bak.codex-20260802-213848`.
The beta19 browser readback found one mower-position dot, zero green heading
lines, zero arrowheads, the explicit `not mower orientation` label, and disabled
Nudge. Experimental motion remained off and no session existed.

## The blocking Gate 5 results and correction candidate

**Gate 5 = UI-to-mower acceptance.** Gates 1–4 passed on 2026-07-31 but were all
*service calls*, so they prove nothing about the card. Gate 5 requires the
operator to drive **from the card**: check the execution-profile row, then
Preview → Dry-run → Real Go. A service call is NOT Gate 5. You cannot click the
card; the operator must.

Before any beta19 final dry-run, confirm the browser console banner reports
`v0.6.4-beta19` and the execution-profile row reads exactly
`LUBA acceptance profile (Gates 1-4, 2026-07-31)`. Save the card's emitted
payload and dry-run result. Do not edit the waypoints or profile between that
dry-run and Real Go; use the same card instance for all three steps. Save the
Real Go result afterward.

Pass criteria remain: both segments report `target_reached`, final error < 8 cm,
and Abort remains effective. The beta16 run on 2026-08-02 did **not** pass:

- the unchanged card path contained two exact 0.400 m legs;
- segment 1 calibrated VIO with a 0.09372 m move, leaving 0.30663 m;
- proportional final-approach scaling selected 1012.5 ms from the isolated
  full-pulse constant of 1.06 m / 3500 ms, but travelled only 0.17861 m;
- it stopped 0.13109 m from waypoint 1 with only 0.0233 m cross-track error;
- `max_linear_commands_reached` stopped the segment and segment 2 never began;
- both stop writes succeeded, the session cleared, the gate was disarmed, and
  the 20-second post-stop capture remained stationary.

The independent 0.450 m characterization reproduced the failure with a
different short duration: its 1191.8 ms approach delivered three refreshes and
moved 0.43414 m, while the earlier 1012.5 ms approach delivered two and moved
0.17861 m. The normal-priority stop on the second run took 1392.666 ms to
confirm. Three post-linear VIO realignments then added drift even though the
single forward-command budget was already exhausted. This establishes that
confirmed refresh count and stop latency dominate nominal duration.

The beta17 candidate replaces proportional-duration shortening with a discrete
refresh-command budget, sends normal pulse teardown zero writes at emergency
queue priority, and prevents realignment after the final linear command. It
does not change the public service schema or `LUBA_ACCEPTANCE_PROFILE`. Treat
it as unproven beyond the acceptance listed below. The full local suite and
GitHub validation workflow pass and the motion-disabled deploy smoke is
complete.

**Gate 2 passed on 2026-08-03 (daylight, backend service).** The operator
visually confirmed an approximately 9 cm move and stop. The single 0.100 m
segment returned `target_reached`; its final error was 0.0105 m. VIO
calibration itself travelled 0.090417 m, placing the mower inside tolerance,
so no normal linear pulse was necessary. Emergency teardown, a cleared session,
experimental-motion disarm, and over one minute of stationary post-stop
telemetry were confirmed; the BLE report contains no observed link/frame
faults. Evidence is `docs/evidence-gate2-beta19-*20260803*`.

**🚦 GATE 4 RE-PASSED 2026-08-05 — read `docs/gate4-repass-20260805.md` before
acting on anything below.** The 2026-08-03 failure described in the next
paragraph is superseded as a *status*, though its evidence remains valid. The
re-pass: both segments `target_reached`, misses `0.0403 m` and `0.0330 m`
against an `0.08 m` tolerance, two-leg backend L path, fresh operator
authorization, no reuse of a prior confirmation. Evidence is
`docs/evidence-gate4-beta20-day2j-*20260805*`.

**It does not yet clear the way to Gate 5, for two reasons.**

1. It ran on three parameters the frozen `LUBA_ACCEPTANCE_PROFILE` does not
   carry: `linear_pulse_duration_ms` 1300 (card 3500), `max_linear_commands` 3
   (card 1), and `max_turn_translation_distance` 0.30 — which the card **never
   sends**, so a card run inherits the backend default 0.25 and would still fail
   the way the day2e/day2h attempts did (`vio_realign_incomplete`).
   `docs/p0-beta-release.md:98-102` says passing Gates 1-4 while the card emits a
   *different* profile is the exact gap that profile exists to close. **Either
   the card profile moves to match, or this re-pass does not underwrite a Gate 5
   Real Go.** That decision is open and was deliberately left to the operator.
2. It passed by overshooting and recovering, not by tracking: `2.2773 m` of
   actual travel for a `1.0400 m` planned path, including a `103.427°` recovery
   turn that is legal only at the 0.30 cap. Reproduction on a second daylight
   geometry remains **required and unmet**.

Two kinematic claims in the older sections below were **refuted by direct RTK
measurement on 2026-08-05** and must not be relied on: single-shot linear does
not give fine distance control (fixed ~`0.11 m` step across a 5× duration
range), and single-shot turning is ~`2.4°`/command, not ~8-9°. Refresh 200 is
the controllable regime for both phases. Sweep data:
`docs/evidence-linear-sweep-refresh200-20260805.json` and
`docs/evidence-linear-sweep-singleshot-20260805.json`.

**Gate 4 retry failed on 2026-08-03 (superseded 2026-08-05, see above).** The durable
multi-segment result records `segment_failed` at segment 1:
`turn_phase_incomplete` / `max_commands_reached`. VIO calibration passed with
offset `3.779947°`, then four turn commands progressed from `6.480°` to
`139.098°` toward a `173.892°` target, leaving `34.795°` error and `0.185 m`
turn translation. `linear_commands_sent` is zero and segment 2 never started;
this was **not** `max_linear_commands_reached`. Experimental motion was
disarmed, the session cleared, and telemetry remained stationary afterward.
See `docs/evidence-gate4-beta19-retry-real-*20260803*` and
`docs/evidence-gate4-beta19-retry-diagnosis-20260803.json`. The reusable
offline analyzer is `scripts/diagnose_motion_result.py`; the durable evidence
runner is `scripts/run_motion_with_evidence.py`. Do not retry a path or change
the profile before a separately authorized daylight turn characterization.

**The turn-planning correction is implemented and locally verified
(2026-08-03, committed on-branch, NOT deployed).** The recorded failure was a
feasibility failure: the 167.413° post-calibration error could never reach the
18° tolerance in four commands at the observed 16.5–21.3°/s rotation rate, yet
the executor dispatched the turn anyway. `_vio_turn_budget_feasibility()` in
`custom_components/mammotion/services.py` now judges every real VIO turn
before its first command, using evidence-anchored conservative bounds:
16.5°/s (the minimum observed Gate 4 rate) times the configured pulse length
when `motion_refresh_interval_ms > 0`, the proven 8°/command single-shot
quantum floor at refresh 0, and — refresh regime only — a worst-case
translation estimate checked against the displacement cap (revised 2026-08-04
from 0.0403 m/s of pulse to 0.0026 m per degree swept). An infeasible turn is
refused fail-closed with stop reason `turn_budget_infeasible` and
`commands_sent: 0`; the vector segment executor surfaces that reason directly
(instead of collapsing it into `turn_phase_incomplete`), and the multi-segment
executor geometrically preflights junctions 2..N and refuses a real path with
`path_turn_infeasible` before any motion. Dry runs report the identical math
(`turn_feasibility`, `junction_turn_feasibility`) without refusing.
`scripts/diagnose_motion_result.py` classifies the refusal as
`vio_turn_refused_infeasible_preflight`, distinct from
`vio_turn_budget_exhausted_before_linear_phase` (which still classifies the
retained evidence) and `linear_budget_exhausted`. Replayed against the
recorded case, the guard refuses before dispatch with an estimate of 7
commands needed against the budget of 4. Tests:
`tests/components/mammotion/test_vio_turn_feasibility.py` (14 cases, including
the retained evidence JSON). No service schema, profile value,
`LUBA_ACCEPTANCE_PROFILE`, or version location changed. The guard prevents the
known-unfinishable dispatch but does NOT make the turn succeed: Gates 4 and 5
remain failed/blocked, and the conservative rate constants plus the ~11°-low
offset question still require a separately authorized daylight turn
characterization on fresh geometry before any retry.

**The daylight turn characterization ran on 2026-08-04 (measurement only, no
deploy, no constant change, no gate claimed).** Four supervised in-place turns
(+45°, −90°, +135°, −170°) at the accepted cadence on the deployed beta19
build, on fresh geometry in `Backyard Right` with VIO state 2 / `Light` / 80
features throughout. **All four returned `target_heading_reached`** — including
the −170° near-worst case, in 4 commands with −0.154° final error. Verdicts on
the guard's evidence floors, over 9 pulses all `heading_went_fresh: true`:

- **rotation ≥ 16.5 °/s: HELD** — min 21.203, mean 32.454, max 49.565 °/s.
- **translation ≤ 0.0403 m/s: VIOLATED** — 4 of 9 pulses over, peaking at
  0.071960 m/s (**+78.6%**). Run 4 commands 1–3 all breached it. The error is
  **fail-open** for the guard's displacement-cap prediction only; the
  rotation-feasibility half is unaffected. Run 4 still finished within the
  0.5 m cap at 0.2955 m.
- **forward-heading offset: UNTESTED** — no forward drive ran (VIO was already
  warm, so the warm-up probe was correctly skipped). The `--summarise` 205.38°
  figure is in-place-turn drift across four reversing phases and must not be
  read as an offset.

BLE was clean over 55 minutes: zero connects, disconnects, sequence gaps,
unparseable or dropped frames. Full record and evidence filenames are in the
2026-08-04 section of `docs/NEXT-SESSION.md`; pooled numbers are
`docs/evidence-turnchar-beta19-analysis-20260804.json`.

**The constant revision followed (2026-08-04, off-mower, committed, NOT
deployed).** The rotation floor stays 16.5 °/s — pooled across both geometries
the true minimum is 16.5251 °/s, set by Gate 4, so the floor is correct and
raising it would move the guard fail-open. The translation criterion was
**re-shaped, not just re-valued**: raising 0.0403 → 0.0720 alone would have
refused the +135° and −170° turns that succeeded (estimates 0.540 m and
0.756 m against actuals 0.029 m and 0.296 m), because the old model multiplied
a per-command translation by a command count derived from the pessimistic
rotation floor — two anti-correlated worst cases compounded. Translation during
an in-place turn is `r × θ`, so it now scales with angle:
`_VIO_TURN_CONSERVATIVE_TRANSLATION_M_PER_DEGREE = 0.0026`, estimate
`|initial_error| × 0.0026`. The constant sits between the pooled observed max
(0.002410) and the binding over-refusal limit 0.25/90 = 0.002778, which keeps a
90° L-path junction feasible; 0.0028 was tried and rejected for violating it.
The guard now refuses Gate 4 and admits all four characterization turns. The
refresh-0 branch is unchanged. Diagnostic fields
`per_command_translation_bound_m` → `translation_bound_m_per_degree` plus a new
`translation_bound_source`; no schema, profile, or version location changed.
Tests 14 → 20.

**The offset re-derivation ran on 2026-08-04 evening and CLOSES the ~11°
question as unsupported.** Three ~0.45 m drives at three headings on beta19.
The vision↔map offset — the one `turn_mode: "vio"` actually uses — measured
+1.258 / +2.042 / −0.969° across travel bearings spanning 175–289° (mean
+0.777°, spread 3.012°). ⚠️ **Do not read that as a stable constant.**
`vision_info.heading` lives in the VIO's own frame, **re-anchored on every VIO
(re)initialisation**, with no fixed relation to map coordinates; `vio_state` was
2 throughout, so these are three noisy estimates of **one session anchor**. With
0.083–0.119 m baselines each carries ~±5–7° of noise, *exceeding* the 3.012°
spread, so heading-invariance is not demonstrated. `vision_info.x/y` are in that
same frame and are **not** map coordinates. The 102.4° figure is a **different quantity** (`toward`-based)
and is **unmeasurable on this hardware**: `toward` stayed frozen across every
forward leg (0.5351 m and 0.6558 m). `--summarise` computes
`(bearing − toward_first)`, so with `toward` stale it reports bearing minus an
arbitrary constant — which is how 111.43 / 113.29 / 115.54 arose and why they
drifted upward. Do not act on the ~11°-low claim. Separately,
`calibrated_forward_heading_offset_degrees` is **not used for turn targeting**
in VIO mode (`provided_offset_degrees` reads `vio_heading_offset_degrees`,
`services.py:9533`), so 102.4 was inert during Gate 4 and cannot explain its
4.70 cm cross-track miss. One anomaly: drive 3 aborted `stop_failed_aborting`
on `command_queue_backlogged` (safe, by design) while the BLE link report was
clean — a queue-state condition, not a link fault. Evidence:
`docs/evidence-offset-beta19-*20260804*`.

**beta20 IS DEPLOYED (2026-08-04 evening) and the guard is live on the host.**
All four version locations read `0.6.4-beta20` / `0.6.4b20`; all 46 files
byte-identical; both card paths `2b1d37bb99069020d2c3eea54b512e9b`; Lovelace
resource `?v=0.6.4-beta20&build=2b1d37bb`; backend `pymammotion 0.8.12.post1`;
backup `/config/mammotion-backup-20260804-2016.tgz`. `LUBA_ACCEPTANCE_PROFILE`
is byte-identical and no entity or schema changed. Dark-safe dry runs confirmed
the deployed guard reports `translation_bound_m_per_degree: 0.0026`, refuses a
179.571° turn against a 4-command budget, and judges the −90° L-path junction
feasible at 0.234 m against a **0.25 m** cap. That 0.25 m junction cap — not
the schema's 0.5 m default — is the binding limit on the constant. Experimental
motion is off, no session, no motion ran, **no gate is claimed**. Sequencing
step 3 is complete; the next step is the Gate 4 retry in daylight.

**Gate 4 was attempted on beta20 (2026-08-04 20:40 EDT) and FAILED — but the
failure moved.** Both turn phases passed in 2 commands each with final errors
**3.808°** and **3.000°**, every pulse `heading_went_fresh: true`; segment 1
returned `target_reached`; the −90° junction turn the guard preflighted as
feasible executed cleanly. The 2026-08-03 turn-budget failure mode is gone.
Segment 2 stopped `max_linear_commands_reached`, landing **0.11660 m** from
waypoint 2 against a 0.08 m tolerance. The miss is **cross-track, not
along-track**: final `x = 5.1011` vs target `5.1006` (0.5 mm), with the entire
error in `y`. Travel bearing was **+5.09°** off expected in segment 1 and
**+14.29°** in segment 2, i.e. an ~11° vision→map discrepancy at execution time
despite calibration drives measuring that offset at ≈0° the same evening.
**Sharper reading (from the VIO commentary, same night):** segment 1's +5.09° is
inside expected noise — its offset came from a `calibration_drive` with an
**0.0892 m** baseline, where 1 cm of position noise is ~6.4° of offset error.
Segment 2's +14.29° is not: its offset came from `linear_refresh` off a
**0.4483 m** leg, a 5× better baseline, yet the error doubled. The junction turn
sits between that refresh and segment 2's drive, and **the offset is refreshed
only from linear travel, never across a turn** — so a VIO frame that drifts or
re-anchors during rotation leaves segment 2 aiming on a stale anchor. The
mid-drive re-aim cannot compensate, because `max_linear_commands: 1` leaves no
forward budget and beta17 suppresses realignment once it is exhausted.

⚠️ **Confounded by falling light** — `tracked_features` decayed 71 → 58 (min 30)
and the aim error grew as features fell, so twilight VIO degradation is at
least as plausible as a systematic error. **Change no constant on this
evidence**; repeat in real daylight with features holding ~80 first. Gate 4
must not reuse a prior confirmation. Evidence:
`docs/evidence-gate4-beta20-*20260804*`.

**VIO liveness is now a cheap pre-session test.** A suspected dusk-latch at
20:33 (stationary `light`/80, 20 min past sunset) was **disproved** by one
bounded 25° turn: `heading_went_fresh: true`, 29.04 °/s, features 78 before and
after. Use a `vio_turn_to_heading` with `max_commands: 2`,
`max_displacement_m: 0.3` rather than trusting the stationary brightness field.
Features dip transiently during rotation (80 → 44 → recovered) from motion
blur; that is normal.

## The open measurement to fold into that run

`calibrated_forward_heading_offset_degrees: 102.4` looks **~11° low**. Three
isolated straight-line runs in darkness implied 111.43 / 113.29 / 115.54 (mean
**113.42**). The night Nudge missed by 0.312 m and the miss was almost entirely
**cross-track** — an aim error, not a distance error. An 11° aim error predicts
~5.7 cm on a 30 cm leg, and Gate 4 landed 4.70 cm out.

**Do not change the profile on that evidence.** Two blockers: `toward` is
course-over-ground and did not update *at all* across a 1.36 m drive, and the
implied offset trended upward run to run. Daylight + live VIO resolves both.

The failed run measured net travel bearing 274.99 degrees against initial VIO
heading -85.881 degrees, implying 99.55 degrees and a -2.85-degree discrepancy
from 102.4; `toward` again did not update. The in-run calibration calculated a
normalized offset of about -1.689 degrees, later refreshed from linear travel
to 1.794 degrees. These data do not justify changing the accepted 102.4-degree
profile. Treat any future Gate 5 run as an offset re-derivation as well. Take the VIO
figures from the **run result JSON** (`vio.initial_vision_heading`, turn-phase
before/after) — **not** from `sensor.*_vio_heading`, which is coordinator-tick
cached and stayed bit-identical across 374 samples.

## Rules that must not be broken

- **Never** push, comment, or open/close anything on a `mikey0000` repository.
  All four upstream remotes have disabled push URLs. Pushes go only to `Chorty`.
- Do not mark PR #10 ready, merge, or dispatch `Beta Release` until Gate 5
  passes and CI is green.
- Gate 5 is currently failed and the release is halted. Any new physical run
  requires a diagnosis, a fresh daylight geometry and fresh operator `go`.
- No physical motion without a **fresh** operator confirmation each time.
  Daylight is required for anything using `turn_mode: "vio"`.
- Start `scripts/motion_capture.py` and `scripts/ble_session_report.py` before
  arming so the complete Gate 5 window is evidence, not recollection.
- **Always disarm after the run**, whether it passes, fails, or is aborted. In a
  finally-style teardown run `scripts/ha_set_experimental_motion.py off`, then
  verify `enabled: False`, no active session, blades off, and stationary
  telemetry. Restore any temporary logger levels as well.
- `LUBA_ACCEPTANCE_PROFILE` is the profile hardware accepted. Editing it
  un-accepts it. Reproduce a measurement on a second geometry first.
- Bump `manifest.json`, `pyproject.toml`, `CARD_VERSION` **and** `uv.lock`
  together, and bump the Lovelace key with
  `scripts/ha_set_card_resource.py <version> --apply` — otherwise browsers keep
  the previous card while every server-side check reports the new one.

## Tools built for this work

| script | purpose |
| --- | --- |
| `scripts/ha_set_experimental_motion.py on\|off\|status` | arm/disarm the motion gate, verified via runtime state |
| `scripts/ha_set_card_resource.py <ver> --apply` | bump the Lovelace resource cache key |
| `scripts/motion_capture.py --seconds N --out f.jsonl` | per-sample RTK/heading capture; `--summarise` reports travel bearing and implied offset |
| `scripts/ble_session_report.py` | capture BLE connection lifetime across the complete supervised window |

## Validation matrix (run all after any change)

```sh
.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests  # 499 pass
.venv/bin/python -m ruff check custom_components tests
.venv/bin/python -m ruff format --check custom_components tests
.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion
npm run test:frontend                                # 19 pass
.venv/bin/python -m pre_commit run --all-files       # green, modifies nothing
```

## Hard-won traps, so they are not rediscovered

- `command_ok` never proves BLE delivery; verify by observed effect.
- Dry-run VIO gates are **advisory** (`passed: dry_run`); the real path enforces
  via `if blockers and not dry_run: return result`.
- `manual_velocity_pulse_test` sends `mammotion.move_forward(speed=0.55)`, a
  different command on a different scale from the vector executor — useless for
  calibration.
- Position telemetry arrives in **bursts** during motion; flat runs are RTK
  batching, not stalls.
- `MODE_READY` a metre from the dock is not a failed dock; it may re-approach.
- `last_error` can be stale and its timestamp is **UTC**.
- Derive constants from an **isolated command**, never from net displacement
  spanning phases — that error produced a confident, wrong 25% claim that a
  clean measurement later refuted.
