# Claude handoff — independently review and release PR #14

You are taking over `Chorty/Mammotion-HA` after the beta54 Night Go and Real Go
throughput follow-up. Work from the repo root on the current tip of
`agent/night-real-go-followup`. Draft PR #14 targets `main` at released/tagged
`0.6.4-beta54` (`0bd35160`). The branch is pushed and clean. Raw hardware
evidence is commit `dd53e266`; implementation/tests are commit `801c1798`;
later commits only reconcile the handoff documents. Confirm the current PR head
instead of assuming a SHA from this prompt.

## Read first, in order

1. `CLAUDE.md`, starting at **Current build**.
2. The 2026-08-14 handoff at the top of `docs/NEXT-SESSION.md`. Treat everything
   below that live section as measurement history, not current build state.
3. `docs/CODEX-HANDOFF.md`.
4. `docs/real-go-throughput-hardware-20260814.md` and both raw JSON records it
   links.
5. `docs/night-go-card-beta54-20260814.md` and its raw JSON record.
6. Section 4 of `docs/night-segment-implementation-plan-v1-20260813.md` before
   proposing any motion-path change.

## Your job

Act as the independent reviewer for draft PR #14. Do not rubber-stamp it merely
because local tests and CI are green.

1. Review the complete diff from beta54 to the PR tip, including every
   per-command hardware record.
2. Confirm the invariants below and resolve any review finding before merging.
3. Run the exact six-command suite below if you change anything. If you make no
   change, independently inspect the existing local/CI records and state what
   you personally reran.
4. When satisfied, mark PR #14 ready and merge it into the Chorty fork only.
   Never push, comment, or merge on a `mikey0000` repository.
5. Prepare the next beta using the repository's established beta workflow
   (expected next identifier: `0.6.4-beta55`; verify rather than assume).
6. Install that beta on Home Assistant **motion-disabled**, following
   `docs/deploy-runbook-p0.md`: back up first, deploy the complete integration,
   synchronize both card-serving paths, update the Lovelace resource with the
   card checksum suffix, restart, and verify exact hashes, version quartet,
   pymammotion version, entity recovery, and gate state.
7. Finish with `experimental motion: off`, `real_motion_allowed: false`, and no
   active session. Report the measured deployment results and actual test counts.

## What PR #14 changes

- Night Go recomputes the residual target bearing from settled post-pulse RTK
  position. This lets the existing reverse-recovery refusal stop a crossed
  waypoint instead of sending another forward pulse.
- The separate Night Go profile and harness send `sample_delays: [0, 3]`. Beta54
  inherited a diagnostic schedule through 60 seconds and made six bounded
  commands take about 6.5 minutes.
- Real Go removes additive feedback waits: VIO calibration uses one four-second
  window, settled linear telemetry is reused, and the card removes one duplicate
  successful runtime reload.
- After each VIO settled-position check, the executor runs the existing bounded
  BLE queue-settle helper. It records `post_feedback_queue_settle` and refuses
  with `ble_link_not_ready_after_feedback` if the queue remains non-live.

The first optimized hardware run is an important safe-failure record: after one
successful forward pulse, the standing gate refused the second before dispatch
with `command_queue_backlogged`. That was not a BLE disconnect. The bounded
queue-settle correction was then deployed, and a separately authorized 0.70 m
Real Go run reached `target_reached` in 19.2 seconds with 0.093100 m landing
error. Its three queue-settle records reached depth zero in about 100–101 ms;
all movement commands and mandatory stops succeeded. The gate was independently
verified off afterward.

## Review invariants

- `LUBA_ACCEPTANCE_PROFILE` must remain byte-identical to beta54. The prior
  review computed the same SHA-256 on both sides:
  `e58c270c188ff5e25339d30b341d6f8151164115b4b77e50754852a66a5923c1`.
- The legacy turn branch must remain byte-identical, including angular 180 and
  the omitted refresh kwarg. Prior review hash:
  `c9165d36fc9837a0c227d788d1770712322ecad16eaaf324067a8365b06b255f`.
- All nine original `turn_mode == "vio"` safety/control blocks and both
  `vio_active` construction sites remain intact. PR #14 adds VIO-scoped feedback
  handling; it does not bypass those gates.
- Real Go's dispatched card payload remains sourced from the frozen profile.
- The VIO response is intentionally **not** byte-identical to beta54: it adds
  settled-feedback and queue-settle provenance. Do not remove those fields to
  satisfy the older night-v1 response-invariance wording. The heading conversion
  and accepted-profile echo remain unchanged.
- Night and legacy feedback timing are unchanged by the Real Go optimization.
- No entity, enum, translation, icon, or import change is required by this PR.

## Exact verification commands

There is no global `uv`; use the project venv.

```sh
.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests
.venv/bin/python -m ruff check custom_components tests
.venv/bin/python -m ruff format --check custom_components tests
.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion
npm run test:frontend
.venv/bin/python -m pre_commit run --all-files
```

The implementation session personally produced 668 pytest passes at 50%
coverage and 46 frontend passes. Ruff check, Ruff format (58 files), mypy (28
source files), and all nine pre-commit hooks passed. PR #14's Python, hassfest,
and Socket checks passed; HACS was skipped by workflow policy. These are prior
facts, not permission to quote them as newly run results.

## Safety boundary

This handoff authorizes repository review, Chorty PR/release work, and a
motion-disabled Home Assistant installation. It authorizes **no mower motion**.

- Do not enable experimental motion.
- Do not arm the motion gate.
- Do not send a movement command or run an armed harness mode.
- Do not infer authorization from earlier supervised runs; each was consumed.
- No additional motion test is required for this beta release.
- If a later hardware movement is proposed, stop and obtain fresh, exact,
  per-run supervised authorization.

## Advice from the implementation session

- Treat per-command records as authoritative. Aggregate stop reasons hid both
  the safe pre-dispatch backlog refusal and earlier crossed-target behavior.
- Do not replace the bounded queue check with another blind sleep. The queue
  reached depth zero in ~100 ms on the successful run; a fixed delay would be
  slower when healthy and less safe when unhealthy.
- Do not interpret the one Real Go pass as a reliability population. It proves
  the specific correction on one path.
- Do not claim night landing accuracy. Night item 18 and the beta54 card run are
  characterization, and the item-15 forward-course/mirror disagreement remains
  unresolved.
- The card is served from two paths. A correct backend deploy with a stale card
  cache is still a failed deployment.
- If review finds a safety or invariance defect, keep the PR draft and stop
  before merge/release. Green CI is necessary, not sufficient.

Report back with the review findings first, then PR/merge/release identifiers,
actual verification counts, deployed hashes, restart/entity timings, and final
motion-gate readback.
