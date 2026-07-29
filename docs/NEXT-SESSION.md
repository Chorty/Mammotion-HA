# Next session

## Current status

- Branch: `feat/vio-turn-to-heading`
- Upstream Mammotion-HA `main` through `v0.6.4-beta11` is merged.
- PyMammotion is pinned to 0.8.12.
- Draft integration PR:
  [Chorty/Mammotion-HA#10](https://github.com/Chorty/Mammotion-HA/pull/10).
- CI passes hassfest, Ruff, formatting, mypy, 404 Python tests, six frontend
  tests, and JSON validation.
- Experimental manual motion defaults off.
- Preview/dry-run supports seven destinations; real click-to-go is capped at
  two segments.
- The public camera-token service is removed; use native camera entities.

## Release blocker

Real motion must remain locked. PyMammotion 0.8.12 still has two documented BLE
teardown leaks. `export_runtime_state.experimental_motion` reports
`pymammotion_backend_unverified`, and the integration requires a future audited
release before nonzero dispatch is authorized.

The upstream duplicate search found no matching issue/PR. The fixes are now
published separately:

- Teardown/connection-slot cleanup:
  [mikey0000/PyMammotion#180](https://github.com/mikey0000/PyMammotion/pull/180)
- Receive reassembly recovery:
  [mikey0000/PyMammotion#181](https://github.com/mikey0000/PyMammotion/pull/181)

Both upstream workflows currently require maintainer approval before GitHub
Actions will run.

Detailed reports:

- `docs/pymammotion-ble-slot-leak-bug.md`
- `docs/pymammotion-ble-reassembly-bug.md`
- `docs/archive/NEXT-SESSION-2026-07-28.md`
- `docs/codex-working-plan.md`

## Next safe actions

1. Obtain upstream review, CI, and merge for PyMammotion #180 and #181.
2. Wait for an official PyMammotion release containing the teardown fix.
3. Update the exact pin and verified backend floor together.
4. Deploy with experimental motion disabled and verify setup, maps/tasks,
   diagnostics, native camera behavior, and card preview/dry-run.
5. Only then perform the supervised daylight LUBA acceptance sequence from
   `docs/p0-beta-release.md`.

Do not merge to fork `main` or create a beta release until CI, the fixed
PyMammotion release, and supervised LUBA acceptance all pass.
