# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Start here

**Read `docs/NEXT-SESSION.md` first**, then the "Current build" section directly below.
Those two carry the live state. Everything from "Gate history" down is settled
provenance — accurate as a record, but **do not act on any build state it
describes**, and note that measurement has since refuted several of its claims
(the "unexplained" turn-rate variance and the turn-budget framing, both below).

## Current build: `0.6.4-beta41` — deployed and VALIDATED ON HARDWARE, gate disarmed

Host and branch agree at beta41, deployed 2026-08-10 (46/46 byte-identical, both
card paths `174f317d`, resource `?v=0.6.4-beta41&build=174f317d`).
`docs/NEXT-SESSION.md` carries the live mower state and the queued run; this
section carries what changed and why.

🏁 **GATE 5 RE-PASSED ON THE REACH-ENABLED PROFILE, 2026-08-12.** Card-driven,
four segments, all `target_reached`, landings 0.0674 / 0.1032 / 0.0807 / 0.0607
against 0.15 — **mean 0.0780, the best four-segment result on record.** Zero
reverse-recovery, zero budget exhaustion. The real payload carried
`max_linear_pulse_ceiling: 14`, so **the reach profile is hardware-accepted and
profile identity is proven in fact.** Read
`docs/gate5-repass-PASSED-20260812.md`; evidence
`docs/evidence-gate5-repass-2-20260812.json`.

⚠️ **Do not overstate it.** beta43 (post-turn correction budget 2 → 4) was **not
exercised** — the only correction was −10.477°, inside the old 21.50° envelope,
and the first attempt's 29.647° refusal did not recur because segment 3's
geometry differed. The **ceiling never bound** either: 0.8 m legs use 2–3 pulses
of 14, so reach is evidenced by `docs/loop-to-tolerance-reach-20260811.md`, not
by this gate. That was deliberate — long legs are 5/7 on control-law grounds,
short legs 28/28.

🏁 **REACH IS SOLVED, 2026-08-11: 4 m on a single segment, measured.** Read
`docs/loop-to-tolerance-reach-20260811.md`. With `max_linear_pulse_ceiling` set,
a 2.0 m leg landed **0.0690 m** in 5 pulses, a 3.0 m leg **0.0928 m** in 8, and a
4.0 m leg **0.1023 m** in 11 — all stopping on **tolerance, not on the ceiling**,
which never bound on any run. The counterfactual is each segment's own third row:
on the accepted profile they sit 0.7489 / 0.6777 / 1.7919 / **2.9543 m** short on
`max_linear_commands_reached`. Per-click reach goes ~4 m → **~16 m** at 4
segments. **4 m is a demonstrated floor, not a limit** — where it breaks is
unknown.

🚨 **A harness bug left the motion gate OPEN once on 2026-08-11 (fixed,
`c196b8b1`).** `scripts/beta32_validation_run.py` set its `armed` flag *after*
the post-enable readback, so an enable that succeeded while `real_motion_allowed`
came back false — BLE dropping between preflight and arm — returned early
claiming it had aborted "without sending anything" and never disarmed. **Any
script that can open the gate must treat "I called enable" as what obliges the
disarm, never "enable succeeded".** Same commit makes the backend's own
`blockers` list a hard preflight check: all eight entity-derived checks passed
while the gate already knew the BLE client was gone.

⚠️ **`max_linear_pulse_ceiling` is a frozen `LUBA_ACCEPTANCE_PROFILE` key that
the card sends as `null`, so NEITHER RUN IS ON THE ACCEPTED PROFILE and the
landings do not compare to Gate 5.** Every other key was sent at its accepted
value. Adopting it un-accepts the profile and owes a fresh Gate 5 — which is now
the next genuine milestone. Measuring first was the entire point.

🔑 **The loop is robust to BLE stalls, and that reframes the standing BLE item.**
The 3 m leg drove through two 2-write pulses (4158 ms and 2847 ms windows) that
travelled 0.2325 / 0.2016 m against 0.34–0.49 m for the 15 cadence-intact pulses,
and still landed at 9.3 cm — it just took two more pulses. Against
`max_linear_commands: 3` those stalls are fatal. BLE latency still degrades the
rate estimate, but it **is no longer a blocker for reach**. (n = 2 stalled
pulses: the shape of the effect, not a calibrated number.)

⚠️ **One open defect, well characterised and NOT implemented.** The 2 m run's
second segment failed on **cross-track, not reach**: the beta38 re-aim guard
suppressed a correction at a projected 0.1469 m miss against 0.150 m tolerance —
3.1 mm of margin — and landed 0.1797 m out. The guard projects the miss at the
**closest approach**, but the mower drives a whole pulse and finished 0.0877 m
past it; in quadrature that predicts 0.1711 m, which exceeds tolerance and would
have fired the correction. Over all 13 recorded suppressions the extra term cuts
mean error 0.0212 → 0.0147 m and the guard under-predicts on **11 of 13**. This
is **not** the fitted margin dropped on 2026-08-10 — that drop's rationale holds
for the 0.7 m legs it was written about and is merely incomplete at long legs.
Touches no profile key. Give it its own review before writing it.

- **beta41** — a segment's **opening turn decomposes instead of refusing**
  (`_vio_turn_to_heading_staged`). It tries the direct turn first and, ONLY on a
  `turn_budget_infeasible` refusal, splits the rotation into stages of ≤60°. Each
  turn call gets its own command budget and displacement allowance, which is why
  chained 60° junctions accumulate 180° where a single 180° turn is refused.
  Wired into the **opening turn only** — mid-drive re-aim and post-turn correction
  keep calling the primitive directly, since their rotations are small by
  construction. Translation is budgeted across the WHOLE staged turn, not per
  stage. If a 60° stage is also refused, the ORIGINAL `turn_budget_infeasible` is
  reported (`staging_cannot_help`), because slicing finer cannot fix a budget that
  dispatches nothing.
  🏁 **Validated 2026-08-10:** a **165.048°** opening turn completed in three
  stages, total staged displacement 0.1326 m of a 0.30 budget, and the beta40
  post-turn gate then corrected the +13.557° residual staging left. The two
  changes compose. Evidence:
  `docs/evidence-beta32-4segment-20260811T001250Z.json`.

✅ **The gate is DISARMED and was verified disarmed after every run.** The
2026-08-10 ARMED-at-rest posture ended with that session; normal posture is
disarmed, opened only for the ~100 s of a supervised run.

- **beta40** — the post-turn alignment gate gets its **own** tolerance,
  `_POST_TURN_ALIGNMENT_TOLERANCE_DEGREES = 10.0`, instead of borrowing
  `vio_realign_threshold_degrees` (the *mid-drive* trigger) through a `min()`. At
  the old `min(18, 15) = 15` the gate never fired. **10 is a floor, not a
  preference:** a correction fires only when the error exceeds the tolerance, so
  the worst sweep is `error + tolerance`, and the affine bound `40 °/s·t + 12°` at
  the 200 ms actuation floor still sweeps 20° — the guarantee needs
  `error + tolerance ≥ 20`, i.e. **tolerance ≥ 10**. Below that, corrections enter
  `sweep_exceeds_any_pulse` and the gate manufactures overshoot. Tightening
  further needs a shorter actuation floor or a tighter sweep bound, **not** a
  smaller number.
  🏁 **Validated 2026-08-10:** four segments reached target, landings
  0.0585 / 0.0867 / 0.1393 / 0.0979 m (**mean 0.0956**, best 4-segment result on
  record). The gate fired once, correcting −16.551° → −7.331°. The correction turn
  displaced 0.0108 m = **0.97° of induced error to buy 10.038°** — the
  "fix reproduces the problem" risk is real but ~10:1 in our favour.
  Evidence: `docs/evidence-beta32-4segment-20260810T205937Z.json`.

**Shipped 2026-08-09/10, in order, each on measurement:**

- **beta35** — refresh writes fire on a fixed cadence from the window start
  rather than one interval after the previous write completed. Delivered-window
  overruns fell from +117% to +29% and **no run has aborted on BLE since.**
- **beta37** — the turn model rebuilt on 35 measured pulses.
  `_MIN_SCALED_TURN_PULSE_MS` 400 → **200** (200 ms actuates; there is no
  threshold near 400). The overshoot bound is no longer a rate: rotation measures
  `33.18 °/s·t + 4.63°`, which no single `C` can bound, so it is now the affine
  envelope `40 °/s·t + 12°`. `_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND`
  16.5 → **14.4**, which the first two changes finally made affordable.
- **beta38** — the mid-drive re-aim guard, corrected. beta36's version compared
  distance alone and suppressed two REAL corrections (40° and 78° aim errors,
  confirmed by RTK independently of VIO), ruining a segment. It now skips only
  when driving on would still land inside the disc:
  `distance·sin(aim) ≤ waypoint_tolerance`.
- **beta39** — the mid-drive re-aim guard follows `effective_linear_ceiling`, not
  `max_linear_commands`. Inert until loop-to-tolerance is enabled; it was the
  last prerequisite blocking it.

✅ **beta38's re-aim guard is VALIDATED on hardware, 2026-08-10.** Two armed runs,
four suppression events, **zero false suppressions** — every suppressed re-aim had
a `perpendicular_miss_m` genuinely under `waypoint_tolerance`, and a −47.812° aim
error was correctly *not* suppressed and corrected to −1.98°. The 60° run reached
target on **all four segments**. Evidence:
`docs/evidence-beta32-4segment-20260810T{185433,193833}Z.json`.

🔑 **THE ACCURACY WALL IS SOLVED, AND IT IS THE TURN'S OWN TRANSLATION.** Read
`docs/turn-translation-explains-the-landing-wall-20260810.md`. A VIO turn does not
pivot in place — it displaced the mower 0.028–0.131 m on the 2026-08-10 runs, and
sideways displacement at the start of a 0.6–0.7 m leg rotates the bearing to the
target by `atan(translation/leg)`. The turn primitive closes on **VIO body
heading**, so it cannot see this: the heading did not change, the target's bearing
moved. Across all five completed segments, map-frame aim error minus VIO-frame
error equals `atan(translation/leg)` to within **0.02–1.25°**.

**Consequence: `heading_tolerance_degrees` is the WRONG LEVER and lowering it
18 → 11 would have changed none of those five segments.** It governs the VIO-frame
error (mean 5.5°, already fine); the landing is set by the map-frame error (mean
8.0°). The lever that works is **not a profile key** — the post-turn gate is
`min(heading_tolerance_degrees, vio_realign_threshold_degrees)` = `min(18, 15)` =
15, and every map-frame error fell inside it. Lowering the backend default
`vio_realign_threshold_degrees` 15 → ~5 catches all five and moves no frozen key.

⚠️ **Do not act on the paragraph below as a plan — it is kept because its
measurements stand, but its diagnosis was superseded the same day.** The standing
fit is
`landing = 0.62 × leg·sin(initial_aim) + 0.065 m` (R² = 0.69, n = 12), which says a
0.9 m leg needs initial aim within **8.8°** while `heading_tolerance_degrees`
permits **18**. That geometric inconsistency between `heading_tolerance_degrees`
and `waypoint_tolerance` is real and both are `LUBA_ACCEPTANCE_PROFILE` keys.
**But it does not explain the observed landings.** Segments 2/3/4 of the 60° run
finished their turns at **−3.7 / −5.1 / +4.6°** — far inside any tolerance under
discussion — and still landed 0.1431 / 0.1447 / 0.1229 m, where the model predicts
0.089 / 0.102 / 0.097. It under-predicts by 26–54 mm. The aim error **develops
mid-leg** (−3.7° → +18.2° in one pulse; −5.1° → −34.7°; +4.6° → +26.0°), part
genuine heading drift and part bearing-to-target rotating as cross-track
accumulates; this run cannot separate them.

**So tightening `heading_tolerance_degrees` would have changed nothing here**, and
the shorter-legs alternative looks worse, not better: the beta33 reference run
drove **0.9 m** legs at the same 60° junctions for a mean landing of **0.098 m**,
while this run's **0.6–0.7 m** legs averaged **0.1312 m**. That comparison is
uncontrolled (six intervening betas, different day) — a warning flag, not a
refutation. **Do not spend a Gate 5 on either profile key until the mid-leg
divergence is characterised.**

**Settled, so do not re-derive:**

- Rotation is **not predictable from duration** better than ~40% at p90 — ten
  pulses at matched ~200 ms windows spread 5.44–15.20°, 2.79×, with duration,
  cadence and direction held constant. The estimate can only improve a landing;
  the bound carries the safety.
- The "directional turn asymmetry" is **refuted** (three runs: 8/8, 1/1, 1/6;
  pooled over 33 samples the directions differ by 0.5%).
- A **90° junction dispatches and completes** — measured, 3 of 4 commands.
- A single **180° turn is refused pre-dispatch**; the largest that dispatches is
  ~114°. Chain junctions instead (`--reposition`).
- Per-**click** reach is 4 segments; per-**segment** reach is ~1 m. A 2.0 m leg is
  not dispatchable.
- `turning_mode` (`zero_turn` / `multipoint`, `nav_sys_param_cmd` ID 6) is a
  MOWING-turnaround planner setting. Click-to-path turns bypass it entirely by
  sending raw `DrvMotionCtrl` velocities. Untested but expected irrelevant.

## Gate history — all five gates complete

**Gate 2** passed 2026-08-03.

**Gate 4** failed on 2026-08-03 before its first linear command, was **re-passed
on 2026-08-05**, and **reproduced on a second daylight geometry on 2026-08-06**.
Read `docs/gate4-repass-20260805.md` before acting on this; the evidence is
`docs/evidence-gate4-beta20-day2j-*20260805*` and
`docs/evidence-gate4-beta21-second-geometry-summary-20260806.json`.

⚠️ **Neither Gate 4 pass tracked its path, and beta22 deliberately refuses the
behaviour that produced them.** Both runs passed by driving past the waypoint and
turning back — 2.28 m and 2.06 m of travel for a 1.04 m path, with 103.427° and
−112.325° recovery turns. beta22 treats a correction of 90° or more as a change
of motion contract and stops with `target_requires_reverse_recovery` rather than
dispatching a U-turn, so **a Gate 4 run on the current build is expected to fail
where beta20/beta21 passed.** That is containment, not regression.

🔑 **2026-08-10 — why those runs overshot is now known, and it was not bad luck.**
Landing error is set by the aim error at the *start* of the leg:
`landing = 0.62 × leg·sin(initial_aim) + 0.065 m` (R² = 0.69, n = 12). A 0.9 m leg
needs initial aim within **8.8°** to land inside 0.15 m, and
`heading_tolerance_degrees` permits **18**. The control law was always going to
miss; the U-turn recovery is what rescued the Gate 4 number, and Gate 5 passed
because its segments happened to start better aligned (worst landing 0.1449 m,
1 mm inside tolerance). Resolving that conflict is the open decision — see
"Current build" above.

*(Historic build state, 2026-08-08: the host ran the motion-disabled
`0.6.4-beta30` candidate with experimental motion verified off, and the branch
was at the undeployed `0.6.4-beta31`. Both are now beta39.)*

The card now emits the Gate 4 re-pass profile, so the profile-identity gap
(`docs/p0-beta-release.md:98-102`) is closed for those three fields
(`linear_pulse_duration_ms` 1300, `max_linear_commands` 3,
`max_turn_translation_distance` 0.30 sent explicitly). The profile is still
accepted on overshoot-and-recovery evidence only.

Do not change the accepted profile casually; changing it obligates the card
copy, a `CARD_VERSION` bump deployed to both serving paths, and the pinning
tests listed in §4 of the re-pass doc. See
`docs/CLAUDE-FINAL-IMPLEMENTATION-PROMPT.md` for the older implementation
handoff, noting that its turn-planning premise was overtaken by the 2026-08-05
measurements. No motion is authorized by this handoff. *(Superseded: the gate is currently
ARMED — see "Current build" above.)*

The card's Real Go defaults are frozen as `LUBA_ACCEPTANCE_PROFILE` in
`www/mammotion-custom-path-card.js` and pinned by frontend tests.

🏁 **GATE 5 PASSED 2026-08-08 — all five gates are complete.** Two card-driven
two-segment runs finished both segments with the accepted profile, zero errors,
zero reverse-recovery and no overshoot. Landings 0.0485 / 0.0836 / 0.0558 /
**0.1449** m against the adopted `waypoint_tolerance: 0.15`; the worst would have
failed at the old 0.08. Evidence: `docs/evidence-gate5-PASSED-20260808.json`.
Profile identity is now proven in fact — the card demonstrably sent the accepted
profile to the mower.

⚠️ Two fragilities the pass does **not** remove — **both rewritten 2026-08-08**
after the raw per-command record was recovered
(`docs/evidence-gate5-attempt5-segment1-raw-20260808.json`; analysis in
`docs/turn-rate-variance-and-reach-analysis-20260808.md`). Read that evidence
file before re-deriving any of this.

**The turn budget is NOT the fragility — that claim is refuted.** The
`turn_commands_sent: 4` was three turn-phase pulses plus one mid-drive
realignment on a *separate* budget; the turn phase stopped at
`target_heading_reached` on command **3 of 4**. The counter is reporting-only and
the true per-segment ceiling is **14**. The real fragility is **overshoot against
tolerance**: pulse 3 overshot the target heading by **13.258°** against
`heading_tolerance_degrees: 18` — **4.74° of margin**. The 2.6× rate spread is
partly an accounting artifact (`services.py:8091` divides by *nominal* pulse
duration, never measured `elapsed_ms`); on elapsed time two of the three pulses
agree to ~3% and only pulse 3 is anomalous. Pulse 3's rotation is nonetheless
real, and unexplained.

**The BLE `TimeoutError` is intermittent, not fixed** — it failed one attempt at
a 80.6° turn, yet a later run completed *larger* turns while showing degraded BLE
(writes median 540 ms) without tripping. Treat it as the tail of a latency
distribution, not a mystery. ⚠️ The stop confirmations 1175/1819/402/628 ms are
the **calibration and linear stops**, not turn stops — turn pulses record no stop
duration at all (`services.py:3321-3333`).

⚠️ `waypoint_tolerance` changed 0.08 → **0.15** in beta30 on hardware evidence
(`docs/evidence-slow-tier-validation-20260808.json`). The position feed is
~1031 ms stale and the mower covers 30–47 cm in that time, so 0.08 could never be
confirmed before the mower had passed the point.

⚠️ **The host and the branch have diverged.** The host still runs the deployed
`0.6.4-beta30`; the branch is at `0.6.4-beta31`, which is **built but never
deployed and never run on hardware**. Everything below describing runtime
behaviour is beta30 unless it says otherwise. See "beta31 (undeployed)" at the end
of this section.

The deployed `0.6.4-beta30` candidate is still unaccepted. On top of
beta22 it adds the read-only `report_stream_probe` diagnostic (beta23, now with
per-channel attribution) and an **RTK quality gate**: non-Fix refuses with
`rtk_not_precise` unless the caller passes `allow_degraded_rtk`, because Float
produced a 13.9 cm stationary jump on 2026-08-07 against an 0.08 m tolerance.
⚠️ RTK payload **age is reported but never blocks** — two thresholds (300 s,
1800 s) both false-blocked, a stationary mower is legitimately quiet for **up to
62.4 min measured**, and a forced burst cannot distinguish quiet from dead
either. This is **closed, not deferred**: do not turn age back into a blocker
without an active liveness probe, which does not exist. See
`docs/rtk-hardening-plan-20260807.md`.

beta27–29 add the read-only `basestation_info_probe`. It established that the
base **does** answer `request_basestation_info_t` — but returns
`score_info: null`, so **`base_moved` / `base_moving` are never populated on this
hardware** and that diagnostic avenue is closed. It also established the
correction chain: **internet source → base station (WiFi) → LoRa E22 → mower**
(base reports `rtk_over_internet`, mower `rtk_over_datalink`), which demotes the
"base survey never converged" hypothesis. ⚠️ Replies bearing the base's own
`iot_id` reduce onto `RTKBaseStationDevice`, **not** the mower's
`report_data.basestation_info` — reading only the mower will call a live base
silent. `MammotionRTKCoordinator` already queries this every tick.

⚠️ **Closed-loop segments cannot run after dark.** The `vio_active` gate keys off
`turn_mode == "vio"` unconditionally, not off whether a turn is needed, and
`_VIO_TURN_MODES` is `("vio", "legacy")` only. *(Refined 2026-08-11 — read
`docs/night-motion-options-20260811.md`. The gate is created ONLY for
`turn_mode == "vio"` (`services.py:10965`), so `legacy` skips it; but `legacy`
closes on `position.toward`, which is course-over-ground and therefore blind to
in-place rotation **at any hour**, not just at night. The real constraint is not
"no heading at night", it is **"no heading while stationary"** — which is why an
ARC, never once sent by this project despite the wire accepting both axes, is the
open lead. 🗑️ **IR is CLOSED**: the mower really does dock on rear-facing IR, but
zero `infrared`/`ir_*`/`photoelectric`/`beacon` fields exist in the integration or
pymammotion, so it is firmware-internal and unreachable. Ultrasonic entities are
`SensorCheckState` self-check enums, not distances; `location.RTK.yaw` is `None`
on this hardware.)* Plan real-motion tests for
daylight. A zero-command
live snapshot proved Mammotion exposes only frozen course-over-ground while
stationary (`toward: -29.589`, VIO inactive/0, RTK yaw 0), so since beta19 the card stops
drawing that last-travel projection as current mower orientation and blocks
Nudge unless a trustworthy current orientation is explicitly available. `manifest.json`,
`pyproject.toml`, `CARD_VERSION` and `uv.lock` (PEP 440 — currently `0.6.4b31`) must always agree, and the
`Beta Release` workflow verifies all four. The card is served from **two**
paths, so deploy to both and bump the Lovelace resource key or the browser can
silently load the stale card. The live Lovelace URL includes the unique build
suffix `?v=<version>&build=<card md5 prefix>` (currently serving beta30). The misleading third-party-map
`card-mod` rotation was removed with verified config readback; its pre-change
backup remains `/config/.storage/lovelace.dashboard_yard.bak.codex-20260802-213848`.

## (history) beta31 — reach 4 segments + turn overshoot ceiling

Built 2026-08-08 on the branch. **No motion has run on it and it is not on the
host.** All CI gates pass locally (533 pytest, 20 frontend, ruff, mypy,
pre-commit). It touches **no `LUBA_ACCEPTANCE_PROFILE` key**, so the profile stays
accepted and no §4 re-pinning is owed.

1. **`REAL_CLICK_TO_GO_SEGMENT_LIMIT` 2 → 4** (`manual_motion.py:24`, mirrored by
   the card's `MAX_REAL_SEGMENTS`). ⚠️ **Segment 3+ has never been executed.** The
   VIO forward-heading offset is refreshed only from linear travel and never
   re-derived across a turn, so cumulative cross-track error past segment 2 is
   unmeasured — and attempt 5's segment 2 already produced the worst landing of
   the four (0.1449 m against 0.15 m).
2. **A turn overshoot ceiling**, `_VIO_TURN_CONSERVATIVE_MAX_DEGREES_PER_SECOND =
   60.0`. Caps each turn pulse so that even at 60 °/s it cannot sweep past
   `|error| + tolerance`. ⚠️ It **routinely becomes the active bound** on final
   approach rather than acting as a rare backstop, and it **couples turn dynamics
   to `heading_tolerance_degrees`**, which is a profile key. Below ~12° of
   tolerance the 400 ms actuation floor wins and the guarantee does not hold.
   Validated by replay arithmetic only — **zero hardware**.
3. **The rotation-rate estimator now divides by measured `elapsed_ms`**, not the
   commanded `pulse_ms` (`services.py`, the `heading_went_fresh` block). On its own
   this makes overshoot slightly *worse*, which is why item 2 ships with it.
4. Two reporting fixes: `motion_refresh_commands_sent` now folds in turn and
   realignment refreshes (it under-reported 6 against 15), and the mid-drive
   realignment no longer dispatches a no-op turn for aim errors already inside the
   turn tolerance — which makes `vio_realign_threshold_degrees` inert in the gap
   between it and `heading_tolerance_degrees`.

Handover, open attacks and the validation-run design:
`docs/HANDOVER-beta31-20260809.md`.

## (history) beta32 — beta31 reviewed, one fix, NOT cleared as-is

beta31 was adversarially reviewed on 2026-08-09 before any deployment and **did
not clear**. beta32 = beta31 + one refusal-side fix. Read
`docs/HANDOVER-beta31-20260809.md` §2.6 before touching turn code.

**Fixed:** `_vio_turn_budget_feasibility` assumed every turn command ran a full
`turn_pulse_duration_ms`, while beta31's ceiling shortens them — so the preflight
admitted turns the executor cannot finish (the two models disagree over
**100–117°** at a 4-command budget). It now replays the executor's own policy via
the same `_turn_final_approach_pulse_ms` the turn loop calls. A 90° junction reads
4 commands, not 3: feasible at **exactly** the budget, no margin.

**Open, and blocking a 90° L-path:**
1. ⚠️ **The ceiling costs ~18° of turn capability.** Replayed through the shipped
   code: a 90° junction completes on beta30 and **exhausts the 4-command budget on
   beta31** at 14.49/14.90 °/s — the rates Gate 5 attempt 5 actually measured. The
   handover's excuse for this was arithmetic that counted pulses to zero error
   instead of to tolerance; it is corrected in §2.2. Fix is to widen the overshoot
   allowance from `K = tolerance` to `K = 2 × tolerance` (~4.5° cost instead of
   ~18°). Not implemented.
2. ⚠️ **The ceiling's guarantee is in commanded ms; the mower rotates for the
   delivered window.** At the +260/+543 ms overruns already on record it holds only
   to 48.0/39.4 °/s — below the 49.56 °/s the hardware has produced.
3. ⚠️ **`_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND = 16.5` is not a floor** —
   14.490 °/s measured. Deliberately not lowered: at 14.4 a 90° junction needs 5
   commands against a budget of 4, so a truthful floor and L-path junctions are
   mutually exclusive until item 1 is fixed.
4. The ceiling biases turn landings toward the tolerance edge, feeding a *tighter*
   post-turn gate (15 vs 18) — expect more post-turn corrections and more
   cross-track error, working against the reach change.

**The validation run keeps every junction in the 45–70° band** — maximum exposure
to the ceiling (it binds below 72°) while clear of the contested 86–100° band.

🏁 **REACH GOAL MET 2026-08-09 — four segments executed on beta33.** Landings
0.0819 / 0.0662 / 0.1452 / 0.0990 m against `waypoint_tolerance: 0.15`, zero
reverse-recovery, zero realignments. **Error does NOT compound with segment
index** (seg4−seg1 slope +0.017 m), so the §2.4 worry is unsupported. Evidence:
`docs/evidence-beta32-4segment-20260809T183129Z.json`. The overshoot ceiling
works: three junction turns closed in **one command each**, landing −5.1 / −2.4 /
−0.3°, against Gate 5's 13.258° overshoot.

⚠️ **THE ROTATION-RATE VARIANCE IS LARGELY A BLE ARTEFACT — read
`docs/HANDOVER-beta31-20260809.md` §2.7 before touching any turn constant.** A
pulse rotates only while refresh writes arrive; when a write blocks, the mower's
watchdog stops the motor and the executor still divides by the whole window. A
1303.7 ms pulse that sent **one of six** refreshes, on a write that took
**1303.972 ms**, measured "9.23 °/s". Cadence-intact pulses that day measured
**23–43 °/s**. This substantially explains the Gate 5 overshoot the ceiling was
built for (a low estimate *lengthens* later pulses), and means
`_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND = 16.5` is probably **not** falsified —
do not lower it on the 14.49/14.90 readings, which are stall-degraded.

beta33 excludes such pulses from the rate estimate (`refresh_cadence_broken`,
`refresh_cadence_broken_pulses`). **Two earlier recommendations are WITHDRAWN:**
"K = 2 × tolerance unlocks 90° L-paths" (at a sustained slow rate no K helps — the
4-command budget caps rotation at ~55°), and the delivered-window shave (it
strictly worsens the binding constraint and would tune against a +0.03%–112%
spread). **The real open lead is BLE write latency, not turn tuning.**

Per-**click** reach is 4 segments; per-**segment** reach is ~1 m
(`max_linear_commands: 3` × ~0.35–0.42 m/pulse). A 2.0 m leg is not dispatchable
and stops on `max_linear_commands_reached`.

**DEPLOYED 2026-08-09 01:16–01:22 EDT, motion-disabled.** The host now runs
`0.6.4-beta32` (it skipped beta31 entirely); all 46 files byte-identical, both
card paths at `16d883fa`, resource `?v=0.6.4-beta32&build=16d883fa`,
`real_motion_allowed: false` read back. A zero-motion dry run confirms the new
preflight executes on the host (`command_count_model:
"executor_pulse_policy_replay"`, ladder `[1300.0, 942.5, 683.3]` for a 60°
junction — pulse 1 already ceiling-bound at 1300 ms instead of 1500). Evidence:
`docs/evidence-beta32-deploy-dryrun-20260809.json`; deploy record in
`docs/deploy-runbook-p0.md`. **No motion has run on beta31 or beta32.** The
4-segment validation run is pending daylight, a charged battery (mower is docked
at `CHARGE_ON`) and per-run authorization.

`pre-commit run --all-files` is green as of 2026-07-31 and is now a usable
gate. Its hook pins must move with `requirements_test.txt`: the Ruff and mypy
hook revs are pinned to the same versions CI installs, and skew between them is
what previously made the hook report failures CI does not have.

Repositories owned by `mikey0000` are read-only for this work. Do not push,
comment, open/close issues or PRs, or publish anything there. A later authorized
push goes only to the `Chorty` fork.

## Build Commands

There is no global `uv` on the dev machine — `uv sync`/`uv run` fail with
`command not found`. Use the project venv directly. (`.venv/bin/uv` exists only
because it was pip-installed into the venv to regenerate `uv.lock`; it cannot
bootstrap the venv it lives in.)

Run the same commands CI runs, so a green local run means a green CI run
(`.github/workflows/validate.yml` is the source of truth):

- Tests: `.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests`
- Lint: `.venv/bin/python -m ruff check custom_components tests`
- Format check: `.venv/bin/python -m ruff format --check custom_components tests`
- Type checking: `.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion`
- Frontend card tests: `npm run test:frontend`
- Pre-commit: `.venv/bin/python -m pre_commit run --all-files`
- Install/refresh deps: `.venv/bin/python -m pip install -r requirements_test.txt`
  (CI's install step; keep the `pymammotion` pin here identical to
  `manifest.json` or CI tests a different backend than the one shipped)

⚠️ Use mypy's `--follow-imports=skip custom_components/mammotion` form above, not
a whole-tree `mypy custom_components/`. The broader form reports 4 pre-existing
errors in `select.py` and `device_tracker.py` that CI never checks — they are not
regressions from your change.

Note: `ruff format` rewrites `except (A, B):` (no `as` binding) into
`except A, B:`. That is **correct and intentional** — PEP 758 allows
unparenthesized exception tuples from Python 3.14, which this project targets
(`requires-python = ">=3.14.2"`). It looks like the Python 2 form and is not;
verified parsing and catching on 3.14.6. Do not "fix" it back.

## Code Style Guidelines

- Follow Home Assistant integration patterns
- Use async/await patterns (prefix functions with `async_`)
- Line ending format: LF (not enforced by any hook or editor config)
- Prefer specific exception types over broad ones

When making changes, follow existing patterns in similar files and follow Home Assistant best practices.

## Home Assistant Integration Rules

- All imports within the integration must be relative (e.g. `from . import Foo`, `from .services import bar`). Never use `from custom_components.mammotion import ...` — HA loads integrations in a way that makes absolute imports from `custom_components` fail at runtime.

## Model and Subagent Routing

**Route by cost of being wrong, not by price per token.** Sonnet 5 is only ~1.67×
cheaper than Opus 5 ($3/$15 against $5/$25 per MTok), so a cheaper model that
needs two passes where Opus needs one has already cost more — before counting the
time spent reviewing the bad first pass. The token gap is the small term. A
plausible-but-wrong claim that reaches an evidence file or shapes a hardware run
is the large one, and this project has been bitten by exactly that repeatedly.

**Sonnet when a machine catches a wrong answer, or the work is high-volume and
mechanical:**

- Deploys — md5 comparison and the `real_motion_allowed: false` readback catch errors
- Version bumps across the four sites — the `Beta Release` workflow verifies all four
- Running the CI gate suite and reporting pass/fail
- Translations sweeps across every language file — JSON parse plus key-presence check
- Broad symbol/reference sweeps where only the conclusion is needed

**Opus when the output is a claim or carries a consequence:**

- Anything touching the motion control law, or any `LUBA_ACCEPTANCE_PROFILE` decision
- Interpreting a run's telemetry; deciding whether a fix actually worked
- Adversarial review, and adjudicating findings
- Analysis written into a `docs/evidence-*` file — it becomes load-bearing for later sessions
- Supervising real motion

**Testing: separate the run from the diagnosis.** Have the cheap session run the
suite and report **raw output only** — which tests failed and the actual
traceback, no interpretation — then stop. Most runs pass, so the common case is
cheap. On a failure, `/model opus` continues in the *same* session with the output
already in context; that invalidates the prompt cache once, but it is not a
restart. A plausible-looking diagnosis from a cheaper model is the exact failure
mode to avoid here.

**For search, prefer inline grep over a subagent.** A subagent returns a summary,
and this project's rule is *verify with per-item records, not aggregates*. On
2026-08-09 two verifier agents wrongly reported that
`REAL_CLICK_TO_GO_SEGMENT_LIMIT` does not exist; one inline grep disproved it.
Use the `finder` agent only when the sweep is genuinely broad, spans several
naming conventions, and the conclusion is all that is needed. When the individual
hits matter — which is most of the time in this repo — grep inline.

**In workflows** (`Workflow` tool), set `opts.model` per stage: cheap models for
find/scan stages, Opus for verify and adjudicate. That is the shape the 2026-08-08
turn-variance investigation used — six Sonnet finders, six Opus verifiers, one
Opus critic. Named agents follow the same split: `finder` for scans, `verifier`
for confirming or refuting a candidate, Opus for fix authoring.

## Translations

- When adding or renaming any entity (sensor, switch, button, number, select, etc.) or an ENUM entity state, you MUST update the translations in **every** language file, not just English.
- The files to keep in sync: `custom_components/mammotion/strings.json` (the source) **and** every file under `custom_components/mammotion/translations/`. Treat that directory listing as the source of truth for which languages exist.
- Translate the entity `name` and every ENUM `state` value into each language's own language — do not copy the English text into the other locales as a placeholder.
- Also add an icon entry in `custom_components/mammotion/icons.json` for the new entity where appropriate.
- After editing, confirm every JSON file still parses and that the new key (with all its `state` values) is present in each file before considering the change complete.
