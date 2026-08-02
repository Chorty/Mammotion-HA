# Handoff prompt — resume Gate 5 (UI-to-mower acceptance)

Paste the block below to resume. It is written to be self-contained; everything
it references is committed on `feat/vio-turn-to-heading` in
`Chorty/Mammotion-HA`.

---

You are resuming P0 completion on the Mammotion-HA repo, branch
`feat/vio-turn-to-heading` (remote `origin` = `Chorty/Mammotion-HA`).

**Read first, in this order:** `CLAUDE.md`, then `docs/NEXT-SESSION.md` (its
"🚨 READ FIRST" block and the "Gate 5 attempt 2026-08-02 morning" section), then
the "Night session 2026-08-01/02" section of `docs/p0-beta-release.md`. Confirm
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

Last known: mower `MODE_READY` at (4.795, −1.9502), `toward` 173.1006, inside
`Backyard Right`, RTK Fix, blades off, no session, VIO alive (80/80 features).
Host runs `0.6.4-beta15`; the working tree is also beta15.

## The one open release gate

**Gate 5 = UI-to-mower acceptance.** Gates 1–4 passed on 2026-07-31 but were all
*service calls*, so they prove nothing about the card. Gate 5 requires the
operator to drive **from the card**: check the execution-profile row, then
Preview → Dry-run → Real Go. A service call is NOT Gate 5. You cannot click the
card; the operator must.

Pass criteria: both segments report `target_reached`, final error < 8 cm, Abort
remains effective.

**Why the last attempt did not run:** legs were 1.180 m and 1.870 m. With
`max_linear_commands: 1` at ~1.06 m per command, both stop short and neither
reports `target_reached`. Usable band is **0.3–0.5 m per leg**. Coordinates
cannot be typed into the card — click roughly, dry-run, read the per-segment
`distance`, drag and repeat until both legs are in band.

## The open measurement to fold into that run

`calibrated_forward_heading_offset_degrees: 102.4` looks **~11° low**. Three
isolated straight-line runs in darkness implied 111.43 / 113.29 / 115.54 (mean
**113.42**). The night Nudge missed by 0.312 m and the miss was almost entirely
**cross-track** — an aim error, not a distance error. An 11° aim error predicts
~5.7 cm on a 30 cm leg, and Gate 4 landed 4.70 cm out.

**Do not change the profile on that evidence.** Two blockers: `toward` is
course-over-ground and did not update *at all* across a 1.36 m drive, and the
implied offset trended upward run to run. Daylight + live VIO resolves both.

Treat the next Gate 5 run as an offset re-derivation as well. Take the VIO
figures from the **run result JSON** (`vio.initial_vision_heading`, turn-phase
before/after) — **not** from `sensor.*_vio_heading`, which is coordinator-tick
cached and stayed bit-identical across 374 samples.

## Rules that must not be broken

- **Never** push, comment, or open/close anything on a `mikey0000` repository.
  All four upstream remotes have disabled push URLs. Pushes go only to `Chorty`.
- Do not mark PR #10 ready, merge, or dispatch `Beta Release` until Gate 5
  passes and CI is green.
- No physical motion without a **fresh** operator confirmation each time.
  Daylight is required for anything using `turn_mode: "vio"`.
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

## Validation matrix (run all after any change)

```sh
.venv/bin/python -m pytest -q tests                 # 456 pass
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check custom_components tests
.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion
npm run test:frontend                                # 17 pass
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
