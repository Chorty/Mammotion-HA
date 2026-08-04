# Claude handoff — finish the blocked beta19 VIO-turn implementation

You are continuing `Chorty/Mammotion-HA` on branch
`feat/vio-turn-to-heading`. Read `CLAUDE.md`, then
`docs/CODEX-HANDOFF.md`, before editing.

## Non-negotiable safety state

Experimental motion is **off**, no session is active, and the mower was last
stationary at approximately `(5.4960, -2.8510)`. Do not arm, deploy, command,
or click Real Go. Any later live diagnostic requires explicit fresh operator
confirmation, daylight, blades physically off, clear path, released e-stop,
and direct observation.

## Exact blocking fact

The 2026-08-03 Gate 4 retry used the guarded multi-segment VIO executor. The
durable result is
`docs/evidence-gate4-beta19-retry-real-result-20260803.json`; its offline
summary is `docs/evidence-gate4-beta19-retry-diagnosis-20260803.json`.

Segment 1 stopped `turn_phase_incomplete` because its VIO turn returned
`max_commands_reached`:

- calibration passed: map/VIO offset `3.779947°`, 0.101625 m calibration move;
- target map heading `177.671979°`; target VIO heading `173.892032°`;
- four `-500` turn commands made positive progress (24.805°, 36.423°, 34.741°,
  36.648°), ending at `139.097502°`;
- final error was `34.795°`, above the 18° tolerance;
- total turn translation was `0.185095 m`, below its 0.25 m displacement cap;
- `linear_commands_sent: 0`, so this was not
  `max_linear_commands_reached`; segment 2 did not start.

The retained profile is an accepted profile; do not tune its 102.4° heading
offset or simply raise an execution budget on this single geometry.

## Implementation objective

Implement a conservative, test-covered turn-planning correction that prevents
an obviously unfinishable near-180° VIO turn from entering a multi-segment
path. Prefer a fail-closed preflight/plan outcome or a bounded, explicitly
tested strategy over silently expanding physical-motion authority.

The implementation must:

1. preserve the existing service schema and fail-closed behavior for non-LUBA
   hardware;
2. expose enough result diagnostics to distinguish turn budget exhaustion from
   linear budget exhaustion without relying on ephemeral command output;
3. use the observed turn progress/translation evidence when assessing whether a
   target is feasible under the configured `vio_turn_max_commands`, tolerance,
   pulse cadence, and displacement cap;
4. refuse execution with a specific stop reason if the requested turn cannot be
   bounded safely, rather than automatically increasing the turn budget;
5. add focused Python tests for the observed near-180° scenario, an already
   feasible turn, and result diagnostics; and
6. leave `LUBA_ACCEPTANCE_PROFILE` and all four beta19 version locations
   unchanged unless a separate, reviewed release decision says otherwise.

`scripts/run_motion_with_evidence.py` now records the service response and
telemetry in the same foreground process. `scripts/diagnose_motion_result.py`
is offline and classifies failed turn versus failed linear phase. Keep them
usable and add tests if their behavior changes.

## Verification and handoff

Run the CI-equivalent suite from `CLAUDE.md`, plus focused tests for the new
turn-planning behavior. Update `docs/CODEX-HANDOFF.md`, `docs/NEXT-SESSION.md`,
and `docs/deploy-runbook-p0.md` with the implemented behavior and its test
evidence. Commit and push only to `Chorty/feat/vio-turn-to-heading`; never act
on a `mikey0000` repository.

Do not mark PR #10 ready, merge, publish beta20, or attempt Gate 5. A future
supervised daylight turn characterization must independently confirm the
implementation before Gate 4 is retried.
