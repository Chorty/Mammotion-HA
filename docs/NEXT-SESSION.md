# Next session

## Current status

- Branch: `feat/vio-turn-to-heading`
- Upstream Mammotion-HA `main` through `v0.6.4-beta11` is merged.
- Draft integration PR:
  [Chorty/Mammotion-HA#10](https://github.com/Chorty/Mammotion-HA/pull/10),
  green with all nine Copilot review threads verified fixed and resolved.
- CI passes hassfest, Ruff, formatting, mypy, 419 Python tests, six frontend
  tests, and JSON validation. `hacs` reports `skipping` from its own workflow
  condition on forks, not a failure.
- Experimental manual motion defaults off.
- Preview/dry-run supports seven destinations; real click-to-go is capped at
  two segments.
- The public camera-token service is removed; use native camera entities.
- Maturity: end of **Alpha**. Stage definitions and the Alpha-to-Beta list live
  in `docs/p0-beta-release.md`.

## The backend gate is open — for the first time

PyMammotion is pinned to **0.8.12.post1**, a Chorty fork build:
<https://github.com/Chorty/PyMammotion/releases/tag/chorty-0.8.12.post1>

It is released `v0.8.12` plus exactly three commits — upstream's rate-limit fix
(`97e5854`), upstream's own BluFi reassembly reset (`68e0095`), and our teardown
failure-atomicity fix (`ffc7857`). It deliberately **excludes** upstream `main`'s
later ~4,100-line saga/token/transport refactor (`fa03b5c`), which is unaudited
for motion safety.

Both capability probes now report present, so `real_motion_allowed` becomes true
once the operator opts in. `enable_experimental_motion` still defaults false.

Authorization is by measurement, never by version number:
`backend_capability.py` probes the loaded code, and every failure mode —
exception, missing attribute, unreadable source, timeout, never probed — reads as
"capability absent". Verified in both directions off-mower, including that each
probe independently detects a partial cherry-pick of only one fix.

Requirement-form facts worth keeping:

- Pinned space-free as `pymammotion@<url>`. **hassfest rejects any requirement
  string containing a space**, so the usual `name @ url` spelling fails
  validation. Confirmed against `script/hassfest/requirements.py` and then
  empirically in CI.
- `requirements_test.txt` must carry the identical pin, or CI tests a different
  backend than the one shipped.
- HA re-installs a URL requirement on **every** start, so an offline restart can
  fail integration setup. Rollback is inherently safe — revert the pin and the
  probes re-lock motion by themselves.

## Upstream state

**Mikey merged our reassembly fix himself** on 2026-07-28 (`68e0095`, *"fix bug
in ble comms thanks @Chorty"*), hand-applying all three `clear_notification()`
calls. It is in `main` but not in any release. **PR #181 is therefore superseded**
and can be closed whenever the operator chooses; #180 (teardown) is still the
only fix with no upstream home.

- Teardown: [mikey0000/PyMammotion#180](https://github.com/mikey0000/PyMammotion/pull/180)
- Reassembly: [mikey0000/PyMammotion#181](https://github.com/mikey0000/PyMammotion/pull/181)
  — superseded by `68e0095`

`mikey0000` repositories are **read-only**: sync, view PRs, read issues, publish
nothing. Push URLs on the `mikey-ha`, `upstream`, `upstream-ha`, and
`upstream-sync` remotes are set to `DISABLED-read-only-upstream` in both
checkouts, so an accidental push fails while fetch keeps working. The prepared
reply to the #181 review question is held in
`docs/pymammotion-181-reply-draft.md` for the operator to send.

Two traps in that repo:

- PRs #180/#181 are open **from `Chorty/PyMammotion` branches**, so pushing
  `agent/ble-failure-atomic-teardown` or `agent/blufi-reassembly-recovery`
  updates the upstream PR. Further work belongs on a fresh branch name.
- The fork inherited upstream's `release.yml`, which publishes to **PyPI** on any
  `v*` tag, and Actions are enabled on the fork. The fork release is therefore
  tagged `chorty-0.8.12.post1` — never tag a fork build `v*`.

Detailed reports:

- `docs/pymammotion-ble-slot-leak-bug.md`
- `docs/pymammotion-ble-reassembly-bug.md`
- `docs/archive/NEXT-SESSION-2026-07-28.md`
- `docs/codex-working-plan.md`

## Next actions

1. Move a BLE proxy nearer the dock before any live run. This is the real
   constraint: median session 59 s and 42% `0x08` supervision timeouts over an
   8-hour docked baseline. Then re-measure with `scripts/ble_session_report.py`
   to test whether the now-pinned slot-leak fix lengthens sessions.
2. Deploy once, with experimental motion **off**, and confirm setup, entities,
   maps/tasks, diagnostics, native camera behaviour, and card preview/dry-run.
   `export_runtime_state.experimental_motion.backend_verified` should read true
   with an empty `blockers` list once opt-in is on.
3. Run the supervised daylight LUBA acceptance sequence — the four gates are
   written out in `docs/p0-beta-release.md`, which is now the single source.
4. Then align `manifest.json`, `pyproject.toml`, and `CARD_VERSION`, mark PR #10
   ready, merge to fork `main`, and dispatch the `Beta Release` workflow with
   `confirmed_luba_acceptance`.

Do not merge to fork `main` or create a beta release until CI and the supervised
LUBA acceptance both pass.
