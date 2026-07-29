# Next session

## Current status

- Branch: `feat/vio-turn-to-heading`
- Upstream Mammotion-HA `main` through `v0.6.4-beta11` is merged.
- PyMammotion is pinned to 0.8.12.
- Python tests, Ruff, formatting, mypy, JSON parsing, and frontend tests pass.
- Experimental manual motion defaults off.
- Preview/dry-run supports seven destinations; real click-to-go is capped at
  two segments.
- The public camera-token service is removed; use native camera entities.

## Release blocker

Real motion must remain locked. PyMammotion 0.8.12 still has two documented BLE
teardown leaks. `export_runtime_state.experimental_motion` reports
`pymammotion_backend_unverified`, and the integration requires a future audited
release before nonzero dispatch is authorized.

The upstream duplicate search found no matching open issue/PR. Publishing the
two PyMammotion fixes is blocked locally until GitHub CLI authentication for
`Chorty` is repaired with:

```bash
gh auth login -h github.com
```

Detailed reports:

- `docs/pymammotion-ble-slot-leak-bug.md`
- `docs/pymammotion-ble-reassembly-bug.md`
- `docs/archive/NEXT-SESSION-2026-07-28.md`
- `docs/codex-working-plan.md`

## Next safe actions

1. Reauthenticate `gh`.
2. Implement and publish separate PyMammotion teardown and reassembly PRs.
3. Wait for an official PyMammotion release containing the teardown fix.
4. Update the exact pin and verified backend floor together.
5. Deploy with experimental motion disabled and verify setup, maps/tasks,
   diagnostics, native camera behavior, and card preview/dry-run.
6. Only then perform the supervised daylight LUBA acceptance sequence from
   `docs/p0-beta-release.md`.

Do not merge to fork `main` or create a beta release until CI, the fixed
PyMammotion release, and supervised LUBA acceptance all pass.
