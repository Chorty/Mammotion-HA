# Claude takeover prompt — independently review beta77/post3

Take over the Mammotion position-cadence safety remediation from the local
workspace at `/Users/mattjoslin/Documents/Git Projects/Mammotion-HA`. This is
an independent two-repository review, not a request to rubber-stamp the prior
implementation. Re-derive the safety contract from the raw beta76 evidence and
the code. If the implementation is incomplete or wrong, fix it offline and
explain why.

## Scope and stopping point

The deliberate handoff boundary is before commit, push, release, deployment,
Home Assistant calls, report-stream reconfiguration, or mower contact.

- Last deployed pair: Home Assistant `0.6.4-beta76` and PyMammotion
  `0.8.12.post2`, deployed motion-disabled on 2026-08-25.
- Offline HA candidate: `0.6.4-beta77`, local branch
  `feat/position-subscription-lease`, based at `dbfd652c`.
- Offline library candidate: `0.8.12.post3`, sibling repository
  `/Users/mattjoslin/Documents/Git Projects/PyMammotion`, local branch
  `release/0.8.12.post3`, based at `413ee71`.
- Both candidates are intentionally uncommitted and unpushed so you can review
  their complete working-tree diffs.
- The HA dependency already names the future Chorty post3 release URL. That URL
  will not resolve until post3 is published; do not deploy HA first.

This prompt authorizes offline inspection, edits, and verification only. It
does not authorize pushes, PRs, releases, deployment, live service calls,
report-subscription changes, gate changes, or mower motion. Ask the operator
before crossing any of those boundaries. Earlier live authorizations are
consumed and do not transfer to this session.

## Preserve these user-owned changes

The following were present before this implementation and are outside scope:

- modified `docs/agora_outbound_audio_probe.md`
- untracked `.vscode/`

Do not stage, revert, format, or otherwise absorb them. Both upstream
`mikey0000` remotes are read-only by policy; never push or open writes there.
The writable fork remotes are `origin` under `Chorty`.

The workspace requires every shell command to begin with `rtk`; read
`AGENTS.md` and `/Users/mattjoslin/.codex/RTK.md` first.

## Read first

1. `CLAUDE.md`, but treat the new beta77 handoff block at the top as current
   and the long older sections as history.
2. `docs/position-cadence-safety-followup-plan-20260825.md`.
3. The three raw stationary artifacts below. Verify their SHA-256 values before
   using them:
   - `/private/tmp/position-cadence-beta76-20260825.json`
     `35153299d2020bc0fd544c09407813b3234136a2117aa0507099c4f2af6f8385`
   - `/private/tmp/position-cadence-beta76-1000-retry-20260825.json`
     `39a2ce6846d443bc969d11038f15de1c1b5339d9933ba6d9b4f40f8a2c37e648`
   - `/private/tmp/position-cadence-beta76-composite-20260825.json`
     `f25360d56691927122161cb1564f0adc29213decbcbb00c535fcc7eddd34f942`
4. The complete uncommitted diffs in both repositories, not just this summary.
5. `docs/deploy-runbook-p0.md` only if the operator later authorizes release or
   deployment.

The composite JSON is explicitly derived by substituting the isolated retry
for anomalous original cell 12. Never present it as the untouched matrix.

## Evidence you must independently re-derive

The beta76 stationary matrix indicates that HA/controller delivery is normally
tens of milliseconds, while position payload cadence is approximately 1 Hz
regardless of requested 100, 250, 500, or 1000 ms periods. Across accepted
cells, 28 of 1432 intervals exceeded two seconds and the largest was 2.909
seconds. In original cell 12, 118 generic reports arrived while only three
position payloads did; an isolated retry recovered immediately. Determine for
yourself whether those raw files support each claim.

The intended safety conclusions are:

- generic `last_report_at` changes cannot prove position freshness;
- continuous steering must remain blocked on this feed;
- no second BLE connection should be opened;
- report ownership/readiness needs deterministic stationary acceptance before
  any future segmented-motion design is considered;
- extending the two-second blind-motion window to wait out multi-second
  telemetry gaps is not acceptable.

Disagree explicitly if your independent review of the raw evidence or code
does not support those conclusions.

## Candidate implementation to review

PyMammotion changes:

- immutable `ReportSubscriptionLease` and `ReportSubscriptionGeneration`;
- one async lock serializing temporary report owners;
- background quiescence before the lease is yielded;
- identity-based current-lease validation and monotonic generations;
- exactly one background rearm on normal release or cancellation;
- version bump to `0.8.12.post3`;
- serialization, stale-lease, and cancellation tests.

Home Assistant changes:

- `report_stream_probe` requires a valid ordered position after the START
  command returns, within the same lease/generation/epoch and without queue
  replacement or sequence gaps;
- generic traffic without position evidence reports `position_channel_stalled`;
- uncertain START acknowledgement paths still attempt STOP in `finally`;
- `report_stream_sequence_probe` holds one lease across all cells;
- `scripts/position_subscription_transition_test.py` plans or executes 30
  stationary transitions, with execution opt-in only;
- the cadence matrix uses one serialized sequence instead of releasing
  ownership between cells;
- latency summaries cover receipt, decode, broker, reducer, state apply,
  publication, and controller consumption;
- coordinator diagnostics distinguish presentation-stream replacements from
  invocation-owned safety-stream drops;
- real heading acquisition uses the lease/generation boundary, retains the
  original two-second zero-angular motion envelope, stops, then observes the
  full 3.5-second stationary window and uses only its newest consecutive valid
  fix for diagnostic heading evidence;
- continuous steering remains refused;
- synchronous entity availability no longer performs distribution metadata I/O;
- version bump to `0.6.4-beta77` and dependency pin to future post3.

## Review questions that are intentionally unresolved

Do not call the candidate complete until you adjudicate these points:

1. The written plan requests command enqueue, send, acknowledgement,
   first-generic, and first-position timestamps. The current result records a
   generation request boundary, command-return acknowledgement, position stage
   times, controller consumption, and a generic-advanced boolean. It does not
   expose a precise lower-level enqueue/send or first-generic timestamp. Decide
   whether more PyMammotion instrumentation is required before release.
2. The library lease quiesces and rearms; HA callers perform the owned STOP.
   Audit every return, exception, cancellation, and lost-ack path for the exact
   required order: STOP attempted and acknowledged when possible, stream
   closed, lease released, one background rearm. Decide whether teardown
   ownership belongs entirely in PyMammotion instead.
3. Confirm no sample queued before START acknowledgement, no pre-generation
   replacement, and no epoch/generation crossing can satisfy readiness.
4. Confirm observation and motion clocks do not start before position readiness
   and command latency is charged where motion is possible.
5. Confirm the stopped 3.5-second observer cannot resume steering in the same
   invocation and does not enlarge commanded travel or the 1.06 m blind disk.
6. Treat 3.5 seconds as a conservative stationary-test default based on the
   beta76 2.909-second maximum, not as a post3-proven distribution or motion
   validation.
7. Confirm the new service schema, YAML descriptions, strings, and English
   translations remain aligned and the public motion service payload shape is
   unchanged.

## Prior verification — evidence, not a substitute for your rerun

The implementation session recorded:

- HA full suite: 930 passed.
- Focused HA safety/lifecycle suite: 533 passed.
- Focused PyMammotion handle/BLE/client suite: 170 passed.
- Wider PyMammotion unit attempt: 993 passed and one pre-existing missing
  `examples/dev_output/mow_progress_1.geojson` artifact; collection of another
  existing test requires absent `pymap3d`. The live-test directory also has
  unrelated `examples.scenarios` collection dependencies.
- Ruff and mypy passed for the modified HA surface.
- Modified PyMammotion source passed isolated mypy with imports skipped.
- `git diff --check` passed in both repositories.
- A local post3 wheel built and imported; final recorded wheel SHA-256:
  `f5c1d41a016b967df3871017e383bfdd845c0b2458b8dabf7d0c1bce188efc82`.

Use the HA project's `.venv`; the sibling PyMammotion repository has no local
venv. At minimum rerun:

```sh
rtk .venv/bin/pytest -q -p no:cacheprovider
rtk .venv/bin/ruff check custom_components/mammotion scripts/position_cadence_matrix.py scripts/position_subscription_transition_test.py tests
rtk .venv/bin/ruff format --check custom_components/mammotion scripts/position_cadence_matrix.py scripts/position_subscription_transition_test.py tests
rtk .venv/bin/mypy custom_components/mammotion scripts/position_cadence_matrix.py scripts/position_subscription_transition_test.py
rtk env PYTHONPATH='../PyMammotion' .venv/bin/pytest -q '../PyMammotion/tests/unit/device/test_handle.py' '../PyMammotion/tests/unit/device/test_ble_loop.py' '../PyMammotion/tests/unit/test_client.py' -p no:cacheprovider
rtk env MYPYPATH='../PyMammotion' .venv/bin/mypy --config-file '../PyMammotion/pyproject.toml' --follow-imports=skip '../PyMammotion/pymammotion/device/handle.py' '../PyMammotion/pymammotion/device/position.py'
rtk git diff --check
rtk git -C '../PyMammotion' diff --check
```

Do not run rewrite flags across the legacy PyMammotion files: its broad Ruff
baseline contains unrelated findings and a different formatter invocation can
create a large non-semantic diff. Review only the intended changes.

## If the offline review passes

First report findings to the operator. Do not silently cross into external
writes. With fresh authorization, the safe order is:

1. Commit the PyMammotion post3 candidate separately, push only to the Chorty
   fork, build the wheel, publish the expected Chorty post3 release/tag, and
   verify the release asset URL and hash.
2. Re-resolve and test the HA candidate against that exact published wheel.
3. Commit/push/release beta77 only after the dependency exists.
4. With separate deployment authorization, back up and deploy beta77
   motion-disabled using `docs/deploy-runbook-p0.md`; verify the backend/card
   version quartet, hashes, entities, PyMammotion version, no active motion
   session, and experimental-motion setting from both live state and storage.
5. With explicit authorization for stationary report reconfiguration, run at
   least 30 transitions using the new harness. This commands no motion but does
   alter the mower's report subscription temporarily.
6. Only if all 30 generations become position-ready with zero safety drops and
   complete stage timing, rerun the untouched randomized matrix under one
   lease.

Do not enable continuous steering. Do not arm the motion gate. Do not send any
movement command. Any future physical acquisition or segmented-motion test
requires a new clear-area check and exact per-run operator authorization.

## Report format

Report in this order:

1. Independent findings, highest severity first, with file/line evidence.
2. Agreement or disagreement with the raw-evidence diagnosis.
3. Any offline fixes made and why.
4. Exact verification commands and counts actually rerun.
5. Remaining blockers and a clear `ready` or `not ready` verdict for post3 and
   beta77 separately.
6. Only if separately authorized: commit IDs, release URLs/hashes, deployed
   hashes/version readback, stationary-transition evidence, and final safety
   state.
