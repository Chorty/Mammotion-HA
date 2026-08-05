# Handoff prompt — reaccept the deployed beta19 motion correction

Paste the block below to resume. It is written to be self-contained; everything
it references is committed on `feat/vio-turn-to-heading` in
`Chorty/Mammotion-HA`.

---

You are resuming P0 completion on the Mammotion-HA repo, branch
`feat/vio-turn-to-heading` (remote `origin` = `Chorty/Mammotion-HA`).

**Read first, in this order:** `CLAUDE.md`, then `docs/NEXT-SESSION.md` (its
"🚨 READ FIRST" block), then the
matching section of `docs/p0-beta-release.md`. Confirm
`git status --short --branch` is clean and inspect `git log -1 --stat` before
editing. Do not reconstruct live-test facts from chat history; those documents
hold the structured record.

## Immediate safety state

⚠️ The experimental-motion gate may have been left **ARMED**. Check first:

```sh
set -a && source .env && set +a
scripts/ha_set_experimental_motion.py status
```

If it reports `enabled: True` and you are not about to run, turn it off with
`scripts/ha_set_experimental_motion.py off`.

After the failed Gate 4 retry on 2026-08-03, the mower was stationary at
approximately `(5.4960, -2.8510)`, inside `Backyard Right`, RTK Fix, blades
off, and had no active session. The host and branch run the still-unaccepted
`0.6.4-beta19` candidate. Experimental motion was explicitly disarmed after
the test and must remain off unless a newly authorized live test is underway.

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

Gate 4 still requires a fresh daylight operator authorization. Only after it
passes may a fresh unchanged-card Gate 5 Real Go be run. Gate 4 must be a
two-leg backend L path and must not reuse a prior confirmation.

**Gate 4 retry failed on 2026-08-03; release remains halted.** The durable
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
+1.258 / +2.042 / −0.969° across travel bearings spanning 175–289°: mean
+0.777°, spread 3.012°, heading-invariant, and re-derived per run by the
executor anyway. The 102.4° figure is a **different quantity** (`toward`-based)
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
.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests  # 483 pass
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
