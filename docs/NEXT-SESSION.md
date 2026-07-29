# Next session

## Current status

- Branch: `feat/vio-turn-to-heading`
- Upstream Mammotion-HA `main` through `v0.6.4-beta11` is merged.
- PyMammotion is pinned to 0.8.12.
- Draft integration PR:
  [Chorty/Mammotion-HA#10](https://github.com/Chorty/Mammotion-HA/pull/10).
- CI passes hassfest, Ruff, formatting, mypy, 404 Python tests, six frontend
  tests, and JSON validation. Re-audited at `7ed86551`: hassfest, python, and
  both Socket checks pass; `hacs` reports `skipping` from its workflow
  condition. All nine Copilot review threads are verified fixed and resolved.
- Experimental manual motion defaults off.
- Preview/dry-run supports seven destinations; real click-to-go is capped at
  two segments.
- The public camera-token service is removed; use native camera entities.

## Release blocker

Real motion must remain locked. PyMammotion 0.8.12 still has two documented BLE
teardown leaks. `export_runtime_state.experimental_motion` reports
`pymammotion_backend_unverified` plus the specific missing capability
(`backend_missing_ble_teardown_failure_atomic`,
`backend_missing_blufi_reassembly_reset`), and nonzero dispatch stays
unauthorized until a backend carrying both fixes is installed.

The upstream duplicate search found no matching issue/PR. The fixes are now
published separately:

- Teardown/connection-slot cleanup:
  [mikey0000/PyMammotion#180](https://github.com/mikey0000/PyMammotion/pull/180)
- Receive reassembly recovery:
  [mikey0000/PyMammotion#181](https://github.com/mikey0000/PyMammotion/pull/181)

Both upstream workflows currently require maintainer approval before GitHub
Actions will run. No release containing either fix exists yet: PyPI and the
upstream release list both stop at 0.8.12 (2026-07-27).

The gate no longer trusts a version number. `backend_capability.py` probes the
installed backend for both audited fixes and real motion requires them to be
observed, so an unfixed 0.8.12 is blocked by measurement rather than by an
unreachable `0.8.13` constant. Verified in both directions off-mower: both
probes report absent against the installed release and present against a copy
with the patches applied, and each probe independently detects a partial
cherry-pick of only one fix. This is what makes a self-built backend safe to
pin -- a fork cannot self-certify by choosing a version string.

Upstream #181 has maintainer feedback from `mikey0000` on
`pymammotion/bluetooth/ble_message.py:355` — "Why was this removed". The anchor
drifted (GitHub marks the thread outdated and pins it to diff position 1); the
PR's only real removal is the manual `hash_map` copy loop in `get_json_string`,
replaced by `dict(hash_map)`. The `parseNotification` fix is purely additive.
Per operator instruction the `mikey0000` repositories are read-only: sync, view
PRs, and read issues there, but publish nothing. All changes go to
`Chorty/Mammotion-HA` and `Chorty/PyMammotion`. The prepared #181 reply is held
in `docs/pymammotion-181-reply-draft.md` for the operator to send.

Push URLs for the `mikey-ha`, `upstream`, `upstream-ha`, and `upstream-sync`
remotes are set to `DISABLED-read-only-upstream` so an accidental push fails
while fetch keeps working.

Note that #180 (`agent/ble-failure-atomic-teardown`) and #181
(`agent/blufi-reassembly-recovery`) are open from `Chorty/PyMammotion` branches,
so pushing either branch updates the upstream pull request. Further PyMammotion
work belongs on a fresh branch name.

Detailed reports:

- `docs/pymammotion-ble-slot-leak-bug.md`
- `docs/pymammotion-ble-reassembly-bug.md`
- `docs/archive/NEXT-SESSION-2026-07-28.md`
- `docs/codex-working-plan.md`

## Next safe actions

1. Publish a fork build: fast-forward `Chorty/PyMammotion` `main` to upstream,
   merge both agent branches (each is released `v0.8.12` plus upstream's own
   merged #177 plus one fix), tag it, and attach a built wheel to the release.
   Point `manifest.json`/`pyproject.toml` at that wheel URL rather than
   `git+https`, which needs a git binary and a build backend inside the HA
   container. Confirm hassfest accepts a URL requirement -- a push to a Chorty
   branch answers that in one CI run. Note HA re-installs a URL requirement on
   every start (`is_installed` returns False whenever the requirement has a
   URL), so an offline restart is the risk to watch.
2. Or wait for an official PyMammotion release containing the teardown fix.
3. Update the exact pin. The capability probes then confirm the fixes are
   actually present; the audited base version stays 0.8.12.
4. Deploy with experimental motion disabled and verify setup, maps/tasks,
   diagnostics, native camera behavior, and card preview/dry-run.
5. Only then perform the supervised daylight LUBA acceptance sequence from
   `docs/p0-beta-release.md`.

Do not merge to fork `main` or create a beta release until CI, the fixed
PyMammotion release, and supervised LUBA acceptance all pass.
