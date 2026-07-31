# Claude handoff: finish Mammotion-HA P0 beta

Updated 2026-07-31 after the corrected supervised Gate 4 run. This is the
current handoff; `docs/archive/NEXT-SESSION-2026-07-28.md` and the chronological
sections in `docs/p0-beta-release.md` are evidence, not current instructions.

## Start here

- Branch: `feat/vio-turn-to-heading`.
- Personal-fork PR: [Chorty/Mammotion-HA#10](https://github.com/Chorty/Mammotion-HA/pull/10),
  still open as a draft. Its last pushed checks are green, but they do **not**
  include the local handoff commit until the operator deliberately pushes it.
- The working tree should be clean after the handoff commit. Confirm with
  `git status --short` and inspect `git log -1 --stat` before editing.
- Do not push, comment, open issues, or otherwise write to a `mikey0000`
  repository. The upstream remotes have disabled push URLs. A later push, if
  authorized, goes only to `Chorty/Mammotion-HA`.
- Do not merge fork `main`, mark the PR ready, or dispatch a beta release until
  the remaining release checks below are resolved and current CI is green.

The exact safety model, deployment instructions, chronological hardware record,
and limitations are in `docs/p0-beta-release.md`. Do not reconstruct live-test
facts from chat history when that document has the structured result.

## What is complete

- P0 backend surfaces are implemented: fail-closed capability registry,
  lowercase HA enum migration, diagnostics redaction, internal-only camera
  credentials, task/map services, experimental-motion option, exclusive motion
  sessions, `stop_manual_motion`, runtime gate export, and the click-to-go card.
- Preview and dry-run allow seven destinations. Real click-to-go is BLE-only,
  opt-in, capability-probed, and limited to two segments.
- PyMammotion is pinned consistently to the Chorty `0.8.12.post1` wheel. It is
  upstream `v0.8.12` plus the rate-limit fix, BluFi reassembly reset, and BLE
  teardown fix. No official upstream release contains the teardown fix.
- The deployed LUBA uses PyMammotion `0.8.12.post1` and integration version
  `0.6.4-beta11`. `coordinator.py` and `__init__.py` match this tree. The local
  `services.py` differs from the deployed checksum only by the corrected schema
  comment about VIO refresh; functional code is the code that passed Gate 4.
- The sole tested proxy path is P1S with passive BLE scanning and active GATT
  proxying:

  ```yaml
  esp32_ble_tracker:
    scan_parameters:
      active: false

  bluetooth_proxy:
    active: true
  ```

- Supervised LUBA acceptance Gates 1-4 all passed:
  - three-write confirmed zero stop;
  - bounded straight segment (9.69 cm toward 10 cm, 5.6 mm final error);
  - active-session abort (owner returned in 673 ms; no post-abort nonzero
    dispatch or replay);
  - 176-degree VIO regression (4.44-degree residual, 10.48 cm turn drift);
  - corrected two-leg L path (both 30 cm segments `target_reached`, final error
    4.70 cm, no delayed replay).
- Gate 4's independent stop confirmed all three zero writes in 530.4 ms. The
  mower stayed bit-identical for 18 seconds, blades stayed off, no session
  reappeared, and experimental motion was disabled afterward.

## Local changes in the handoff commit

1. A cloud-backed mower now registers its late BLE advertisement callback even
   when no proxy is ready during entry setup.
2. A temporarily absent transport reports `none` instead of raising during HA
   entity setup.
3. Bluetooth option changes invalidate the five-second motion-gate cache and
   refresh entities immediately. Enabling survives a temporarily unavailable
   advertisement so the later callback can attach.
4. `ManualMotionCancelledError` propagates out of the refresh loop after a
   defensive stop, so an operator abort releases the exclusive owner promptly.
5. A translating VIO turn now receives the configured displacement limit,
   recalculates the bearing from fresh post-turn position, and fails before
   linear motion unless alignment is freshly proven. A bounded correction has
   at most two turn commands and shares the normal realignment budget.
6. Regression coverage was added for all of the above. The exact handoff tree
   passed all 456 Python tests with coverage after the documentation pass.

## Validation at handoff

Passed on the exact pre-commit tree:

- 456 Python tests with coverage;
- CI-scoped Ruff and Ruff format checks;
- CI-scoped mypy over 28 source files;
- all six frontend card DOM tests;
- JSON validation inside pre-commit;
- `git diff --check`.

The repository-wide `pre-commit run --all-files` is **not green** and must not be
reported as passing. It currently combines several baseline/tooling problems:
legacy Ruff scans scripts outside the CI scope, codespell flags extracted APK
identifiers and existing fixtures, pyupgrade crashes under Python 3.14, and the
hook's broad mypy invocation reports 168 errors that the documented CI-scoped
command does not. Prettier also rewrites many unrelated evidence files. The
hook's automatic unrelated edits were reverted. Fix the hook configuration or
its baseline deliberately before making all-files pre-commit a release gate.

## Assumptions that changed during hardware testing

These corrections matter more than the original chat plan:

- The recurring app/HA BLE conflict was not evidence that the mower radio was
  defective. Disabling the integration let the official app connect
  immediately; isolating proxies then implicated the IRK proxy path. P1S
  restored immediate app BLE access and stable confirmed writes. Do not re-enable
  multiple proxies during acceptance without a new controlled comparison.
- Passive `esp32_ble_tracker` scanning does not make a proxy passive for GATT.
  `bluetooth_proxy.active: true` is required and is the configuration that
  passed. `ble_link_live` may not be bypassed; routing preference alone does not
  prove a writable link.
- A VIO pivot is not an in-place geometric turn. It translated 14.43 cm in the
  failed Gate 4 attempt and changed the bearing enough to miss. Always compute
  the forward bearing from post-turn position. The corrected retry translated
  8.80 cm but freshly proved a 0.285-degree aim error before driving.
- Delayed RTK reports can make bounded physical motion appear later. Treat
  replay as a session/dispatch fact, not merely as a later position change.
- Do not increase the four-second BLE write timeout to hide stalls. That also
  lengthens uncertain nonzero delivery. The confirmed-dispatch and emergency
  stop behavior is the safety boundary.
- There is still no firmware arbitrary-waypoint upload API. The accepted path
  is a guarded chain of raw manual-motion segments, not autonomous navigation.

## Important remaining release blocker

The backend Gate 4 call used a deliberately bounded acceptance profile:

- `max_real_segments: 2`
- `max_turn_commands: 4`, `vio_turn_max_commands: 4`
- `max_linear_commands: 1`, no loop-to-tolerance ceiling
- `waypoint_tolerance: 0.08`
- `min_progress_distance: 0.0025`
- `calibrated_forward_heading_offset_degrees: 102.4`
- `motion_refresh_interval_ms: 200`
- `ble_auto_recover: false`

The card's built-in defaults still reflect an older July 18 profile (including
three linear commands, a 30-pulse ceiling, 15 cm waypoint tolerance, 6 cm
progress threshold, 116.5-degree offset, and BLE auto-recovery). Therefore:

1. Do **not** describe the card's default Real Go profile as hardware-accepted.
2. Decide whether to make the conservative Gate 4 profile the card default or
   expose a clearly named LUBA acceptance profile.
3. Update frontend payload assertions and README YAML together.
4. Repeat preview/dry-run and, only with a new daylight operator `go`, one Real
   Go from the actual card if release criteria require UI-to-mower acceptance.

This is the main direction change from the earlier assumption that passing the
backend gates was immediately followed by versioning and release.

## Remaining P0 work, in order

1. Resolve the card-default mismatch above. No physical motion is authorized by
   this handoff.
2. Fix or explicitly defer with a release-blocking rationale:
   - map edits are not visible until HA restarts because the per-tick map refresh
     block is unreachable in steady state;
   - `no_actuation_detected` can fire falsely in the turn phase even when
     `heading_went_fresh` proves rotation;
   - Task-2 constants were not re-derived after the transport failures.
3. Repair or explicitly scope the all-files pre-commit baseline described above.
   The CI-scoped test, lint, format, type, frontend, JSON, and diff checks pass
   on this handoff; rerun them after any card/default change.
4. Recheck version agreement. At handoff, `manifest.json`, `pyproject.toml`, and
   `CARD_VERSION` are all `0.6.4-beta11`, and the three dependency declarations
   use the identical wheel URL. Let the repaired beta workflow choose the next
   monotonic beta number.
5. With explicit operator authorization, push only to the Chorty feature branch
   and wait for current PR checks. HACS `skipping` on a fork is expected; Python
   and hassfest must pass.
6. Only after the selected card profile and current CI pass: mark PR #10 ready,
   merge to Chorty `main` without force-pushing, and run `Beta Release` with the
   LUBA-acceptance confirmation. Never publish to Mikey's repositories.

## Safety state at handoff

- `enable_experimental_motion: false`
- no active motion session
- last Gate 4 session completed normally
- mower last reported `MODE_READY`, blades off, fixed RTK, inside `Backyard Right`
- last settled map position: x 4.3911, y -2.8064
- no live test may reuse the previous operator confirmation

## Useful commands

```sh
git status --short --branch
git log -1 --stat
.venv/bin/python scripts/mammotion_preflight_gates.py --quick
.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests
.venv/bin/python -m ruff check custom_components tests
.venv/bin/python -m ruff format --check custom_components tests
.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion
npm run test:frontend
```

Never enable broad PyMammotion debug logging: cloud and raw BLE loggers can
expose credentials, network responses, device identifiers, and payloads. Use
only the scoped `bleak_esphome` and `habluetooth` loggers documented in
`docs/deploy-runbook-p0.md`.
