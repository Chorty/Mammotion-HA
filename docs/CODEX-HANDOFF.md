# Handoff prompt — diagnose Gate 5 short-pulse failure

Paste the block below to resume. It is written to be self-contained; everything
it references is committed on `feat/vio-turn-to-heading` in
`Chorty/Mammotion-HA`.

---

You are resuming P0 completion on the Mammotion-HA repo, branch
`feat/vio-turn-to-heading` (remote `origin` = `Chorty/Mammotion-HA`).

**Read first, in this order:** `CLAUDE.md`, then `docs/NEXT-SESSION.md` (its
"🚨 READ FIRST" block and the "Gate 5 beta16 final attempt" section), then the
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

Last known after the failed run: mower `MODE_READY` at (4.8583, −2.1320),
`toward` 175.4473, inside `Backyard Right`, RTK Fix, blades off, no session.
The host and working tree run `0.6.4-beta16`, including the precise-coordinate
editor. The motion gate is verified off.

## The blocking Gate 5 result

**Gate 5 = UI-to-mower acceptance.** Gates 1–4 passed on 2026-07-31 but were all
*service calls*, so they prove nothing about the card. Gate 5 requires the
operator to drive **from the card**: check the execution-profile row, then
Preview → Dry-run → Real Go. A service call is NOT Gate 5. You cannot click the
card; the operator must.

Before the final dry-run, confirm the browser console banner reports
`v0.6.4-beta16` and the execution-profile row reads exactly
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

This is primarily a distance-model failure, not an aim failure. A proportional
model through zero is invalid for this short pulse because motor onset/dead time
is material. The former **0.3–0.5 m usable-band claim is refuted**. Do not retry
the same geometry or tune from this single short-pulse sample.

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
.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests  # 456 pass
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
