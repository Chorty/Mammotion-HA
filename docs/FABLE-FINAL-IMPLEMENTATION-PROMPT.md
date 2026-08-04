# Fable handoff — safely finish the blocked beta19 VIO-turn code

Work in `Chorty/Mammotion-HA` on branch `feat/vio-turn-to-heading`.

## Hard safety boundary

Do **not** arm experimental motion, deploy to Home Assistant, call a motion
service, click Real Go, merge, or publish a release. The mower is disarmed and
the release is halted. This task is source-code analysis, implementation, and
local verification only.

Read, in order:

1. `CLAUDE.md`
2. `docs/CODEX-HANDOFF.md`
3. `docs/CLAUDE-FINAL-IMPLEMENTATION-PROMPT.md`
4. `docs/evidence-gate4-beta19-retry-diagnosis-20260803.json`
5. `docs/evidence-gate4-beta19-retry-real-result-20260803.json`

## The confirmed failure

Gate 4 failed in segment 1 before the linear phase:

- VIO calibration passed with a `3.779947°` offset.
- Target VIO heading: `173.892032°`.
- Four turn commands progressed VIO heading from `6.480°` to `139.098°`.
- The remaining error was `34.795°`, above the `18°` tolerance.
- Turn translation was `0.185095 m`, below the `0.25 m` cap.
- The recorded stop reason is `max_commands_reached` in the turn phase.
- `linear_commands_sent` is zero; segment 2 never started.

This is a VIO turn-budget feasibility failure, **not** a linear-budget failure.

## Stage 1 — analyze before editing

Inspect the VIO turn and multi-segment execution flow in
`custom_components/mammotion/services.py`. Report, in your working notes:

1. where target VIO heading is calculated;
2. where the turn-command budget is enforced;
3. what diagnostics are already returned after a failed turn; and
4. the smallest fail-closed change that prevents a known-unfinishable
   near-180° turn from starting a multi-segment path.

Do not solve this by blindly increasing `vio_turn_max_commands`, changing the
102.4° accepted offset, or weakening displacement/tolerance safeguards.

## Stage 2 — implement a fail-closed correction

Implement a conservative, testable feasibility guard or planning outcome.

It must:

- preserve existing service schemas and public API compatibility;
- preserve fail-closed behavior for non-LUBA hardware;
- use configured turn budget, heading tolerance, expected bounded turn progress,
  cadence, and translation cap when deciding feasibility;
- return a precise stop reason and useful diagnostics when it refuses a path;
- avoid dispatching motion for a turn it judges infeasible; and
- leave `LUBA_ACCEPTANCE_PROFILE` plus beta19 version locations unchanged.

Prefer a narrow helper with unit tests over a broad executor rewrite.

## Stage 3 — verify locally

Add focused Python tests for:

1. the recorded near-180° case being rejected before physical execution;
2. a feasible turn continuing to normal planning; and
3. diagnostics clearly distinguishing turn-budget exhaustion from
   `max_linear_commands_reached`.

Run:

```sh
.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests
.venv/bin/python -m ruff check custom_components tests
.venv/bin/python -m ruff format --check custom_components tests
.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion
npm run test:frontend
.venv/bin/python -m pre_commit run --all-files
```

## Stage 4 — record the handoff

Update `docs/CODEX-HANDOFF.md`, `docs/NEXT-SESSION.md`, and
`docs/deploy-runbook-p0.md` with the implemented guard, exact tests, and the
fact that a new daylight operator-authorized turn characterization remains
required. Do not claim Gate 4, Gate 5, PR readiness, or release success.

Commit only the implementation, tests, and documentation to
`Chorty/feat/vio-turn-to-heading`. Never act on a `mikey0000` repository.
