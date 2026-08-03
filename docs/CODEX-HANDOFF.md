# Handoff prompt — deploy beta19, then reaccept the motion correction

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

Last known after the beta18 motion-disabled deployment: mower `MODE_READY` at
approximately (4.9524, −2.7114), inside `Backyard Right`, RTK Fix, blades off,
no session. The host runs the still-unaccepted `0.6.4-beta18` candidate; the
branch is the undeployed `0.6.4-beta19` stale-orientation safety correction.
Beta19 does not change Real Go motion or the accepted profile. The motion gate
is verified off. It is dark with VIO at
0 tracked features, so no motion is authorized by this handoff.

The beta18 deploy smoke passed: 128 Mammotion entities, backend verified against
`pymammotion 0.8.12.post1`, BLE live, both card paths checksum-identical, exact
accepted-profile label, valid Preview, and a card Dry-run with `valid: true`,
`would_send: false`, and `stop_reason: dry_run`. Real Go remained disabled and
the card was reset to zero waypoints. The live resource URL is
`?v=0.6.4-beta18&build=6da6c3d3`. Rollback backup:
`/config/mammotion-backup-20260802-2129.tgz`.

The beta18 direction conclusion was incomplete. A zero-command snapshot taken
while the operator visually confirmed the mower faced upper-left showed frozen
course-over-ground `toward: -29.589`, `location.orientation: -29`, VIO inactive
with heading 0, and RTK yaw 0. Neither available feed reports the stationary
body orientation. Beta19 keeps the projected last-travel bearing as explicit
diagnostic text but draws no orientation arrow from it and disables Nudge
unless a trustworthy current orientation exists. The earlier third-party-map
`card-mod` rotation was removed and read back successfully. Dashboard backup:
`/config/.storage/lovelace.dashboard_yard.bak.codex-20260802-213848`.

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
it as unproven on hardware. The full local suite and GitHub validation workflow
pass and the motion-disabled deploy smoke is complete. Affected backend Gates 2
and 4 still require fresh daylight operator authorization. Only after those
pass may a fresh unchanged-card Gate 5 be run.

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
.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests  # 469 pass
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
