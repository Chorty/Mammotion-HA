# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current P0 Handoff

Read `docs/CODEX-HANDOFF.md`, then `docs/NEXT-SESSION.md`, before continuing P0
or click-to-go work. The host runs the still-unaccepted, motion-disabled
`0.6.4-beta30` candidate and experimental motion is verified off; the branch is
at the undeployed `0.6.4-beta31`. Gate 2 passed
on 2026-08-03, and Gate 4 — which
failed on 2026-08-03 before its first linear command — was **re-passed on
2026-08-05** and **reproduced on a second daylight geometry on 2026-08-06**.
Read `docs/gate4-repass-20260805.md` before acting
on this; the evidence is `docs/evidence-gate4-beta20-day2j-*20260805*` and
`docs/evidence-gate4-beta21-second-geometry-summary-20260806.json`.

**Neither pass tracked its path, and beta22 deliberately refuses the behaviour
that produced them.** Both runs passed by driving past the waypoint and turning
back — 2.28 m and 2.06 m of travel for a 1.04 m path, with 103.427° and
−112.325° recovery turns. Beta22 treats a correction of 90° or more as a change
of motion contract and stops with `target_requires_reverse_recovery` rather than
dispatching a U-turn, so **a Gate 4 run on the current build is expected to fail
where beta20/beta21 passed.** That is containment, not regression. The open
question is control quality — lead the stop by `speed × latency`, or accept
overshoot-and-recovery explicitly — **not** a Gate 4 retry or a Gate 5 attempt.

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
measurements. No motion is authorized by this handoff.

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
`_VIO_TURN_MODES` is `("vio", "legacy")` only. Plan real-motion tests for
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

## beta31 (undeployed, unvalidated) — reach 4 segments + turn overshoot ceiling

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

## beta32 (undeployed) — beta31 reviewed, one fix, NOT cleared as-is

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

## Subagent Model Routing

Keep expensive reasoning on the session model; route delegated work to cheaper models automatically:

- **Search/scan fan-out** (code-review finder angles, diff scans, symbol/caller hunts, convention audits): use the `finder` agent, or pass `model: sonnet` when spawning a general-purpose agent for this kind of work.
- **Verification and adjudication** (confirming/refuting review candidates, checking that a library API actually exists, call-site impact analysis): use the `verifier` agent, or pass `model: opus`.
- **Fix authoring in a subagent**: `model: opus`.
- Only leave a subagent on the session (inherited) model when the task genuinely needs top-tier long-horizon reasoning — orchestration itself already runs there.

## Translations

- When adding or renaming any entity (sensor, switch, button, number, select, etc.) or an ENUM entity state, you MUST update the translations in **every** language file, not just English.
- The files to keep in sync: `custom_components/mammotion/strings.json` (the source) **and** every file under `custom_components/mammotion/translations/`. Treat that directory listing as the source of truth for which languages exist.
- Translate the entity `name` and every ENUM `state` value into each language's own language — do not copy the English text into the other locales as a placeholder.
- Also add an icon entry in `custom_components/mammotion/icons.json` for the new entity where appropriate.
- After editing, confirm every JSON file still parses and that the new key (with all its `state` values) is present in each file before considering the change complete.
