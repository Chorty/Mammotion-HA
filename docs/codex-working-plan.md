# Codex working plan / handoff memory

Last updated: 2026-07-06

This file is the repo-local memory index for Codex work on this branch. Use it as the source of truth when a new chat needs context. It intentionally avoids secrets, HA tokens, passwords, and live mower credentials.

## Current operating rule

- This chat should be the control point for repo edits.
- If another chat made changes, audit the working tree before editing:

  ```bash
  git status --short
  git diff --stat
  git diff
  ```

- Do not deploy to Home Assistant or restart HA unless explicitly approved.
- Existing user changes in the working tree should be preserved unless explicitly reverted.

## Current high-level state

- Current local working version is `0.6.4-beta65`.
- `0.6.4-beta37` was copied to HA and HA was restarted so new service registrations were visible. `0.6.4-beta38` contains the follow-up read-only runtime diagnostics batch. `0.6.4-beta39` adds route-staleness classification so residual route GeoJSON does not block paused/ready manual-motion diagnostics by itself. `0.6.4-beta40` adds heading-offset candidate evaluation for cumulative pulse and experimental segment burst diagnostics so the controller is not locked to a single fixed offset. `0.6.4-beta51` adds raw pymammotion motion calibration. `0.6.4-beta52` adds the Part 1 raw linear one-segment executor. `0.6.4-beta53` adds the Part 2 raw angular calibration loop. `0.6.4-beta54` adds the guarded raw turn-to-heading probe. `0.6.4-beta55` adds consolidated raw motion readiness.
- A config-entry reload does not necessarily unload/reimport already-imported custom-component Python modules. New service/entity/platform discovery may require a full HA restart before it appears.
- After beta37 deploy/restart, HA confirmed:
  - HA version: `2026.7.0b2`
  - `mammotion` config entry loaded: yes
  - mower entity `lawn_mower.back_yard_clip_skywalker` available: yes
  - `mammotion.manual_velocity_cumulative_pulse_test` registered: yes
  - `mammotion.experimental_execute_segment_burst` registered: yes
- Current beta65 work may still be uncommitted locally until the post-movement feedback refresh integration batch is completed, deployed, and committed.
- `main` is pushed to `origin/main` on `Chorty/Mammotion-HA`.
- Claude merged upstream Mammotion-HA beta7 content into this branch, then committed/pushed the beta10/beta11 working tree.
- Claude reports `0.6.4-beta11` was deployed to HA and the Mammotion config entry was reloaded through the HA REST API.
- HA may still display stale integration metadata in some places until Home Assistant restarts because custom integration metadata is cached in the running HA process.
- Current manifest/pyproject version:

  ```json
  "version": "0.6.4-beta65"
  ```

- Motion execution state:
  - `mammotion.raw_vector_readiness_test` remains the canonical gate for vector motion.
  - Proven vector readiness defaults are `max_turn_commands: 4` and `max_linear_commands: 2`.
  - `scripts/mammotion_motion_suite.py` is the standard orchestration script for vector readiness and optional multi-segment checks.
  - `mammotion.raw_pymammotion_execute_multi_segment` was added in beta60 as an experimental guarded wrapper over the proven one-segment vector primitive.
  - Multi-segment execution accepts 2-4 points, defaults to dry-run, defaults to `max_real_segments: 1`, rechecks safety before each segment, and stops on the first failed segment.
  - `mammotion.forward_two_pulse_latency_test` was added in beta61 to send two forward raw `send_movement(200, 0)` pulses with a 5-second gap and measure telemetry latency after the second pulse.
  - In beta62 the same diagnostic supports `pulse_count` 2-5 and telemetry timeout up to 300 seconds so larger cumulative forward movement can be tested without adding path-execution behavior.
  - Real beta62 five-pulse testing physically moved the mower five times, but the current normalized HA/pymammotion position source stayed unchanged after roughly three minutes. This means closed-loop custom-path execution is blocked until a more reliable feedback source is found.
  - `mammotion.position_feedback_diagnostic` was added in beta63 to capture normalized telemetry plus raw likely pymammotion/HA position fields, optionally send a small bounded forward pulse burst, then run safe refresh/status paths (`request_report_snapshot`, `request_reports`, `start_report_stream`, forced `ensure_fresh_state`, BLE sync type 3, and HA coordinator refresh) and report which sources changed.
  - In beta64 the same diagnostic also tries direct pymammotion IoT reporting requests: one-shot `request_iot_sync` and a bounded `request_iot_sync_continuous` window that is explicitly stopped. These are reporting requests, not movement/blade commands.
  - Real beta64 one-pulse diagnostic artifact `/tmp/mammotion_position_feedback_diagnostic/20260703-202643` proved that one `send_movement(200, 0)` pulse can produce usable map-local feedback when followed by `request_reports(count=5)`. The first actual position update appeared after `request_reports_count_5`, about 4 seconds after the command, changing `report_data.locations[0]` from `(x=4.6787, y=0.4108)` to `(x=4.6729, y=0.3532)`, about `0.0579 m` movement. This makes `request_reports(count=5)` the current best native feedback refresh after raw movement.
  - In beta65 the raw linear segment, raw turn-to-heading, and vector-segment linear phase call the proven `request_reports(count=5)` feedback refresh after each successful raw `send_movement` command before judging progress from telemetry.
  - `scripts/mammotion_position_feedback_diagnostic.py` calls the beta63 diagnostic service, saves full JSON artifacts under `/tmp/mammotion_position_feedback_diagnostic`, and defaults to dry-run/no movement.
  - Full drawn-path / arbitrary long autonomous execution is still not enabled.
  - 2026-07-05 live forward validation closure (this session):
    - Deployment/reload from this macOS host was restored and validated against the live HA config entry.
    - Baseline profile is now frozen as validated: speed `200`, visible `3` pulses with `8s` gap, plus visible cumulative `5` pulses with `8s` gap.
    - Frozen baseline batch at speed `200`:
      - `visible_repeat_1`: `position_source_changed`, `commands_sent=3`, no blockers.
      - `visible_repeat_2`: `position_source_changed`, `commands_sent=3`, no blockers.
      - `visible_cumulative_5`: `position_source_changed`, `commands_sent=5`, no blockers.
      - `latency_profile_5`: `telemetry_position_change_detected`, `commands_sent=5`, first-change timings `20.007s` (cmd1), `15.005s` (cmd2), `0.001s` (final), final delta distance `0.0305 m`.
    - Final comparison batch at speed `320` (same sequence):
      - Visible repeats and cumulative run all passed with `position_source_changed` and no blockers.
      - `latency_profile_5`: `telemetry_position_change_detected`, first-change timings `89.09s` (cmd1), `84.088s` (cmd2), `69.085s` (final), final delta distance `0.3195 m`.
    - Interpretation: speed `320` yields larger displacement but materially slower telemetry confirmation; speed `200` remains the default recommended operating profile for fast feedback/repeatability in guarded forward diagnostics.

- The latest APK-derived control batch was implemented, committed, pushed, and reportedly deployed/reloaded by Claude.
- The naming cleanup request was accidental and was effectively reverted/no-op. Current intended names remain the pre-cleanup keys:
  - `device_wifi_enabled`
  - `device_4g_enabled`
  - button label `Run camera wiper`
- Current working tree should be clean except local handoff/documentation files and the untracked APK/XAPK unless more work has happened. Always verify with `git status --short`.

## Claude handoff audit summary

Claude worked in a separate chat without this Codex plan. Its handoff was audited against the repo on 2026-06-28.

Verified repo facts:

- `HEAD`, `origin/main`, and `origin/HEAD` pointed to:
  - `1c305902 Merge remote main (automation commits) into local merge`
- Recent commits included:
  - `ffe65af2 Add pymammotion compat shim and map/task visibility tests`
  - `bbb4c624 chore: bump version to 0.6.4-beta11`
  - `ac6c9aa5 Merge 0.6.4-beta10: upstream beta7 sync + stream auth, map tracking, radio switches`
  - `9025e697 Merge upstream 0.6.4-beta7: area names and dynamics line BLE fixes`
- `custom_components/mammotion/manifest.json` and `pyproject.toml` both showed `0.6.4-beta11`.
- The expected Codex feature keys existed in HEAD:
  - `get_tasks`
  - `get_areas`
  - `last_map_sync`
  - `last_task_sync`
  - `last_map_task_error`
  - `STREAM_AUTH_ERROR_CODE`
  - `prompt_volume`
  - `voice_language`
  - `camera_wiper`
  - `device_wifi_enabled`
  - `device_4g_enabled`
  - `pymammotion_compat`

Classification after audit:

- Keep:
  - Claude's committed beta11 merge.
  - Upstream beta7 merge.
  - map/task visibility.
  - cloud login parser fix.
  - camera 401 retry.
  - APK-derived controls.
- Revert:
  - Nothing obvious from Claude's committed changes.
- Needs review:
  - Non-English translation files were missing the new keys from `strings.json` / `en.json`.
- New useful Claude discovery:
  - HA config entry ID was identified in Claude's session.
  - HA config entry reload should use the service endpoint, not `/api/config/config_entries/{id}/reload`.
  - SSH key auth uses a passphrase-protected key; direct `sshpass` can fail by answering the key passphrase prompt instead of the remote password prompt.
  - SMB deployment was reliable.
  - An old `join_webrtc` / "Start camera on mower" entity can remain orphaned in HA from earlier versions.
  - A GitHub automation workflow may auto-push version bumps and cause repeated manifest/pyproject conflicts. Keep the higher beta number when resolving.
- Unsafe/unknown:
  - `.env` contains deploy credentials and must not be copied into docs or committed.
  - The large APK/XAPK should not be committed.

## Implemented plan: Better Map/Task Visibility

Read-only map/task visibility was implemented.

Services:

- `mammotion.get_tasks`
- `mammotion.get_areas`

Diagnostic sensors:

- `sensor.<mower>_task_count`
- `sensor.<mower>_enabled_task_count`
- `sensor.<mower>_area_count`
- `sensor.<mower>_map_area_name_count`
- `sensor.<mower>_last_map_sync`
- `sensor.<mower>_last_task_sync`
- `sensor.<mower>_last_map_task_error`

Coordinator metadata tracking:

- `last_map_sync`
- `last_task_sync`
- `last_map_task_error`

Sync metadata is updated on:

- `async_sync_maps()`
- `async_sync_schedule()`
- `async_refresh_mower_tasks()`

Known test coverage added:

- `tests/components/mammotion/test_map_task_visibility.py`

## Implemented fix: cloud login parser compatibility

Problem observed:

- Mammotion/pymammotion cloud login could fail parsing share-notice data when `initiatorAlias` was missing.

Implemented:

- Compatibility patch module:
  - `custom_components/mammotion/pymammotion_compat.py`
- Wired into:
  - `custom_components/mammotion/__init__.py`
  - `custom_components/mammotion/config_flow.py`

Known test coverage added:

- `tests/components/mammotion/test_pymammotion_compat.py`

## Implemented fix: camera stream token 401 retry

Problem observed:

- Camera stream could fail with stale/expired cloud stream token.

Implemented:

- Camera stream token request now detects 401, refreshes cloud credentials/token, stores refreshed credentials, and retries.

Main file:

- `custom_components/mammotion/coordinator.py`

Existing camera hotfix tests were run and passed.

## Implemented locally: APK-derived controls

These were re-added cleanly after earlier TTS/audio experiments were rewound.

Entities added locally:

- `number.<mower>_prompt_volume`
- `select.<mower>_voice_language`
- `button.<mower>_camera_wiper`
- `switch.<mower>_device_wifi_enabled`
- `switch.<mower>_device_4g_enabled`

User-facing labels currently intended:

- `Prompt volume`
- `Voice language`
- `Run camera wiper`
- `Device Wi-Fi`
- `Device 4G`

Important technical caveat:

- pymammotion does not currently expose a separate field named `au_volume`.
- The relevant app/protobuf field is `au_switch` on readback.
- The pymammotion reducer maps `au_switch` into `mower_state.audio.volume`.
- The command builder writes the same multimedia audio oneof via `set_car_volume` / `MulSetAudio.at_switch`.
- Therefore `prompt_volume` is a cleaner HA-facing entity for the app’s prompt audio setting, while existing `voice_volume` remains for backward compatibility. They may currently drive the same underlying SDK field.

Coordinator wrappers added locally:

- `async_set_prompt_volume`
- `async_set_voice_language`
- `async_run_camera_wiper`
- `async_set_device_wifi_enabled`
- `async_set_device_4g_enabled`

## APK / reverse engineering context

Verified APK/XAPK path used during investigation:

```text
Mammotion_2.3.8.19_APKPure.xapk
```

Do not commit this large APK/XAPK unless intentionally wanted. It is currently untracked in prior status outputs.

Useful pymammotion findings:

- Multimedia command builder exists in pymammotion:
  - `set_car_volume`
  - `set_car_voice_language`
  - `set_car_volume_sex`
  - `set_car_wiper`
  - `get_car_audio_cfg`
- Network command builder exists:
  - `set_device_wifi_enable_status`
  - `set_device_4g_enable_status`
  - `get_device_network_info`

## Agora/TTS conclusion

Arbitrary TTS through the mower speaker via Agora still looks unlikely with the known paths.

## Ready To Hand Over

Release status (what passed):

- RC local validation gate for motion-critical files passed (`services.py`, `coordinator.py`, `__init__.py`, `pymammotion_compat.py`).
- Full local test suite passed: `180 passed`.
- Live deployment/reload path from this macOS host was validated.
- Live guarded forward diagnostics passed at speed `200` baseline profile and at speed `320` comparison profile.

What remains (not release-blocking for guarded forward diagnostics):

- Broad mypy backlog outside the motion-critical release gate still exists (`mypy custom_components` reported remaining errors in non-critical files during this session).
- Full autonomous drawn-path/arbitrary long execution remains intentionally disabled pending stronger closed-loop telemetry confidence.
- Keep verifying that all entity/enum translation keys remain synchronized across `strings.json` and all locale files when future entities/states are added.

Exact validated profile (frozen baseline):

- Default operating profile: speed `200`.
- Validation sequence:
  - Visible repeat #1: `position_feedback_diagnostic(linear_speed=200, pulse_count=3, pulse_gap_seconds=8.0)`.
  - Visible repeat #2: `position_feedback_diagnostic(linear_speed=200, pulse_count=3, pulse_gap_seconds=8.0)`.
  - Visible cumulative: `position_feedback_diagnostic(linear_speed=200, pulse_count=5, pulse_gap_seconds=8.0)`.
  - Latency profile: `forward_two_pulse_latency_test(linear_speed=200, pulse_count=5, pulse_gap_seconds=5.0)`.
- Acceptance outcome from frozen run:
  - Visible diagnostics: `reason=position_source_changed`, expected `commands_sent` (`3`, `3`, `5`), no blockers.
  - Latency profile: `reason=telemetry_position_change_detected`, `commands_sent=5`, first-change timings `20.007s` (cmd1), `15.005s` (cmd2), `0.001s` (final), final delta `0.0305 m`.

Comparison profile (final reference):

- Speed `320` with the same sequence passed with no blockers.
- Tradeoff observed: larger displacement (`final delta 0.3195 m`) but much slower first-change confirmation (`89.09s`, `84.088s`, `69.085s`).
- Operational recommendation: keep speed `200` as default for repeatable guarded-forward diagnostics with faster telemetry confirmation.

Operator runbook (concise):

1. Preconditions:
  - Ensure mower is undocked, blades off, clear area confirmed, and no active mowing route.
  - Confirm `allowed_for_manual_motion=true` via `mammotion.export_runtime_state`.
2. Baseline execution order (default speed `200`):
  - Run visible repeat #1 (`3` pulses, `8s` gap).
  - Run visible repeat #2 (`3` pulses, `8s` gap).
  - Run visible cumulative (`5` pulses, `8s` gap).
  - Run latency profile (`5` pulses, `5s` gap).
3. Pass criteria:
  - No safety blockers.
  - Visible runs return `reason=position_source_changed`.
  - Latency run returns `reason=telemetry_position_change_detected`.
4. Stop/abort criteria:
  - Any `safety_gates_failed`, `mower_state_unsafe`, or `telemetry_quality_degraded` result.
  - Any new unexpected blocker in runtime gate.
5. Artifacts and records:
  - Save diagnostic JSON artifacts under `/tmp/mammotion_position_feedback_diagnostic`.
  - Log command payloads, reasons, timing fields, and final delta in release notes/handoff.
6. Deployment notes:
  - Prefer config-entry reload for code iteration; use full HA restart when new service/entity registrations do not appear due to module metadata caching.

Handoff decision:

- Guarded forward-motion validation is complete for this release scope.
- Ship with speed `200` as the validated default diagnostic profile.

RC decision note (2026-07-06):

- Release decision for guarded forward diagnostics scope: `GO`.
- Scope basis: full regression tests passed and motion-critical typing gate passed.
- Residual risk accepted as non-blocking for this scope: broad mypy backlog outside motion-critical files.
- RC readiness tag: `RC-READY (guarded-forward scope)`.

One-page RC checklist:

1. Quality gates:
  - [x] Full pytest passed (`180 passed`).
  - [x] Motion-critical mypy passed (`services.py`, `coordinator.py`, `__init__.py`, `pymammotion_compat.py`).
  - [x] Broad mypy clean (`0 errors in 26 files`).
2. Live safety and behavior:
  - [x] Runtime motion gate verified before each real batch (`allowed_for_manual_motion=true`).
  - [x] Baseline speed `200` profile validated (3x8 repeat twice + 5-pulse cumulative + latency profile).
  - [x] Final speed `320` comparison batch completed and documented.
3. Camera/cloud reliability stage:
  - [x] Refresh controls present (`refresh_camera_stream`, `refresh_cloud_session`).
  - [x] Diagnostic sensors present (`active_transport`, `last_cloud_login_success`, `last_token_refresh`, `last_command_failure_reason`, `last_camera_stream_failure_code`).
  - [x] Focused reliability tests passed (`test_camera_hotfix.py`, `test_diagnostics.py`).
4. Release notes and handoff:
  - [x] Baseline/tradeoff interpretation documented in this plan.
  - [x] Operator runbook documented in this plan.
  - [x] Explicit GO/NO-GO decision recorded.

Release packaging notes / changelog draft (2026-07-06):

- Release tag target: `0.6.4-beta65-rc` (working draft label for packaging notes).
- Included in this packaging snapshot:
  - Live guarded-forward validation closure with frozen baseline profile (speed `200`) and documented speed `320` comparison.
  - Camera/cloud reliability diagnostics stage marked complete (`refresh_camera_stream`, `refresh_cloud_session`, and diagnostic telemetry entities).
  - Broad-mypy backlog burn-down on top two files completed:
    - `custom_components/mammotion/button.py`: typing updates for optional config-entry access and button press callback signatures.
    - `custom_components/mammotion/camera.py`: typing alignment for stream subscription response wrappers and optional HA state/entity resolution.
- Validation for this packaging snapshot:
  - `pytest -q`: `180 passed`.
  - `mypy --follow-imports=skip custom_components/mammotion/button.py custom_components/mammotion/camera.py`: clean.
  - `mypy --follow-imports=skip custom_components/mammotion/agora_websocket.py custom_components/mammotion/sensor.py`: clean.
  - `mypy --follow-imports=skip custom_components/mammotion/select.py custom_components/mammotion/number.py`: clean.
  - `mypy --follow-imports=skip custom_components/mammotion/config_flow.py custom_components/mammotion/update.py`: clean.
  - `mypy --follow-imports=skip custom_components/mammotion/switch.py custom_components/mammotion/lawn_mower.py`: clean.
  - `mypy --follow-imports=skip custom_components/mammotion/entity.py custom_components/mammotion/device_tracker.py`: clean.
  - `mypy --follow-imports=skip custom_components/mammotion/agora_sdp.py`: clean.
  - Broad mypy snapshot after burn-down: `0 errors in 26 files` (reduced from `127 errors in 13 files`).
- Known non-blocking backlog for next batch:
  - Broad mypy backlog for `custom_components/mammotion` is fully cleared in this pass.
- Packaging decision:
  - Proceed with RC packaging for guarded-forward scope.
  - Broad typing backlog for `custom_components/mammotion` is cleared for this release.
  - Do not expand scope with autonomous long-path execution in this release.

Final release notes / changelog block (ready to publish):

- Release: `0.6.4-beta65-rc`
- Status: RC-ready for guarded forward-motion scope.
- Validation summary:
  - Full tests: `pytest -q` => `180 passed`.
  - Broad typing gate: `mypy custom_components` => `0 errors in 26 files`.
  - Live validation: frozen speed-200 baseline passed (`3x8` visible repeat twice, `5x8` cumulative, `5x5` latency profile); speed-320 comparison passed and documented.
- Reliability and diagnostics included:
  - Camera/cloud refresh controls: `refresh_camera_stream`, `refresh_cloud_session`.
  - Runtime diagnostics: `active_transport`, `last_cloud_login_success`, `last_token_refresh`, `last_command_failure_reason`, `last_camera_stream_failure_code`.
- Typing hardening completed in this release pass:
  - `button.py`, `camera.py`
  - `agora_websocket.py`, `sensor.py`
  - `select.py`, `number.py`
  - `config_flow.py`, `update.py`
  - `switch.py`, `lawn_mower.py`
  - `entity.py`, `device_tracker.py`
  - `agora_sdp.py`
- Scope guardrails:
  - Guarded forward diagnostics are validated and release-complete.
  - Autonomous arbitrary long-path execution remains intentionally out of scope.

Clean commit plan (grouped for review):

1. `chore(typing-core): services/coordinator/init compat hardening`
   - Files:
     - `custom_components/mammotion/services.py`
     - `custom_components/mammotion/coordinator.py`
     - `custom_components/mammotion/__init__.py`
     - `custom_components/mammotion/pymammotion_compat.py`
2. `chore(typing-entities-1): button camera agora-websocket sensor`
   - Files:
     - `custom_components/mammotion/button.py`
     - `custom_components/mammotion/camera.py`
     - `custom_components/mammotion/agora_websocket.py`
     - `custom_components/mammotion/sensor.py`
3. `chore(typing-entities-2): select number switch update`
   - Files:
     - `custom_components/mammotion/select.py`
     - `custom_components/mammotion/number.py`
     - `custom_components/mammotion/switch.py`
     - `custom_components/mammotion/update.py`
4. `chore(typing-platform): config-flow lawn-mower entity device-tracker agora-sdp`
   - Files:
     - `custom_components/mammotion/config_flow.py`
     - `custom_components/mammotion/lawn_mower.py`
     - `custom_components/mammotion/entity.py`
     - `custom_components/mammotion/device_tracker.py`
     - `custom_components/mammotion/agora_sdp.py`
5. `docs(release): rc handoff, validation matrix, packaging notes`
   - Files:
     - `docs/codex-working-plan.md`

Observed Agora SDK diagnostics:

- Browser SDK can join the channel and see mower user/video.
- Local audio track appeared live/enabled but Agora local audio stats stayed at zero send bytes/packets/bitrate.
- Mower published video but did not publish/subscribe audio in the observed way needed for speaker playback.

Practical conclusion:

- Focus on built-in spoken prompt settings rather than arbitrary TTS unless a separate app talkback mode/command table is found.

## Completed stage: camera/cloud reliability diagnostics

This stage is now implemented in the codebase and validated locally.

Implemented controls and diagnostics:

- `button.<mower>_refresh_camera_stream`
- `button.<mower>_refresh_cloud_session`
- diagnostic sensor: `active_transport`
- diagnostic sensor: `last_cloud_login_success`
- diagnostic sensor: `last_token_refresh`
- diagnostic sensor: `last_command_failure_reason`
- diagnostic sensor: `last_camera_stream_failure_code`
- BLE-only fallback surfaced via transport state semantics (`active_transport=ble` and related runtime diagnostics)

Validation evidence:

- Focused reliability tests passed:
  - `tests/components/mammotion/test_camera_hotfix.py`
  - `tests/components/mammotion/test_diagnostics.py`
  - Result: `9 passed`
- Feature coverage also present in `tests/components/mammotion/test_map_task_visibility.py` for sensor export/entity registration and refresh button behavior.

Outcome:

- Camera/cloud failures are exposed with actionable diagnostics and explicit refresh controls rather than only a generic temporary-unavailable symptom.

## Implemented locally after beta12: custom path visibility foundation

Read-only custom path planning services were added locally. These do not send movement commands and do not start mower tasks.

Services:

- `mammotion.export_map`
  - Exports normalized areas, map-local area polygons, and raw map data useful for route planning/debugging.
  - Coordinate system is explicitly reported as `mower_map_xy`.
- `mammotion.export_tasks`
  - Exports normalized tasks plus task count, enabled task count, last task sync, and last map/task error.
- `mammotion.validate_custom_path`
  - Validates a proposed map-local `x`/`y` path.
  - Requires blade mode `off`.
  - Checks minimum point count, maximum point count, non-zero distance, speed warnings, optional `area_hash`, and point containment against known area geometry when available.
  - Returns `valid`, `errors`, `warnings`, `distance`, normalized points, and validation metadata.
- `mammotion.preview_custom_path`
  - Reuses the same validation logic.
  - Returns display-ready GeoJSON with start/end markers and a LineString preview.
  - Still read-only; it never sends mower movement, blade, task, or path-upload commands.

Frontend:

- `custom_components/mammotion/www/mammotion-custom-path-card.js`
  - Served by the integration at `/mammotion/mammotion-custom-path-card.js`.
  - Loads `export_map`, draws map-local area polygons, lets the user click/drag custom path points, and calls `preview_custom_path`.
  - No execution button exists in this first version.

Research:

- `docs/custom-path-execution-research.md`
  - Documents why custom path execution is not approved yet.
  - Lists the command-path and blades-off questions that must be answered before any execution service is implemented.

Current scope:

- No custom path execution.
- No mower movement commands.
- No blade commands.
- No task creation or path upload.

Next safe step after this:

- Use `export_map` to inspect real mower map polygons and test `validate_custom_path` / `preview_custom_path` with known safe points.
- Only after validation/preview is reliable should execution command paths be investigated.

## Implemented locally/deployed by beta37: experimental manual movement probes

Custom-path execution remains experimental and intentionally limited, but the
manual movement command path has been characterized.

Implemented services:

- `mammotion.manual_velocity_cumulative_pulse_test`
  - Sends a bounded burst of stopped manual velocity pulses only when `dry_run`
    is false and explicit blade/clear-area confirmations are provided.
  - Waits once for cumulative telemetry after the pulse burst.
  - Used to characterize delayed telemetry without weakening stricter closed-loop
    probes.
- `mammotion.experimental_execute_segment_burst`
  - One-segment only.
  - Uses capped pulse bursts and delayed cumulative telemetry between bursts.
  - Does not enable full custom-path execution.
- `mammotion.move_forward` / `move_left` / `move_right` / `move_backward`
  registration bug fixed in beta40 working tree:
  - `services.py` had registered movement handlers through a synchronous lambda
    returning the `handle_movement()` coroutine.
  - HA logged `coroutine 'async_setup_services.<locals>.handle_movement' was never awaited`.
  - Replaced the lambda with an async wrapper that awaits `handle_movement()`.

Observed real movement:

- Three forward pulses at speed `0.4`, `750 ms` each produced visible forward
  movement and telemetry-confirmed movement after a delayed update.
- Example telemetry-confirmed delta:
  - distance: approximately `0.128 m`
  - telemetry latency: `120 s`
  - command ack: true
  - stop ack: true
- Manual velocity movement works, but telemetry is too delayed for fast
  responsive closed-loop control.
- Later beta40 tests after restart showed that stopped 750 ms pulses and direct
  movement service calls can acknowledge without telemetry-confirmed movement:
  - one `experimental_execute_segment_burst` run sent 3 pulses, selected heading
    offset `90`, and returned `no_cumulative_progress`;
  - a forced-forward cumulative probe sent 5 stopped forward pulses and measured
    only about `0.004 m`;
  - direct `mammotion.move_forward` calls before the await fix were affected by
    the unawaited coroutine bug;
  - direct `mammotion.move_forward` after the await fix executed without the
    warning but still produced no telemetry movement during the 60 s observation
    window.
- Current working theory:
  - `async_send_command()` returning true means the command was accepted by the
    manager/transport without exception; it does not prove mower-side motion.
  - The immediate zero-speed stop primitive may cancel movement before firmware
    starts the nudge.
  - The mower may reject manual motion while HA still reports `paused` and stale
    active route/progress, even when `work_mode` is `MODE_READY`.
  - Next movement implementation should add explicit command response diagnostics
    for movement services and a guarded firmware-nudge-style mode that can
    intentionally omit or delay the zero-stop command while sampling telemetry.

Blade safety conclusion:

- `blade.reported_state == 1` / label `ON` must be treated as unsafe even when
  `current_cutter_rpm == 0`.
- `current_cutter_rpm` is diagnostic only, not the authoritative blade safety
  signal.

## Mowing telemetry and active route discoveries

Read-only mowing captures were saved locally under:

```text
/tmp/mammotion_mowing_capture
```

Important discoveries:

- The useful map-local live position source is `report_data.locations[0]`.
- During firmware-managed mowing, this position source updates well enough to
  reconstruct route traversal.
- During active mowing, telemetry cadence looked bursty: repeated samples then
  jumps, roughly useful around 10-second intervals in observed runs.
- Active mowing route/progress is exposed through existing GeoJSON services:
  - `mammotion.get_mow_path_geojson`
  - `mammotion.get_mow_progress_geojson`
- `get_mow_path_geojson` captured multiple route features such as `mow_path`
  and `border_pass`.
- `get_mow_progress_geojson` captured active progress:
  - `type_name: mow_progress`
  - `is_active: true`
  - `now_index`
  - `total_points`
  - `path_hash`
- Stored task exports returned zero tasks during active mowing, so active route
  diagnostics should prefer runtime GeoJSON over `data.map.plan`.
- Camera support was discussed as a future diagnostic/safety upgrade but is out
  of scope for the current implementation batch.

Implemented locally in the runtime diagnostic batch:

- Read-only `mammotion.export_runtime_state`.
- Read-only `mammotion.export_active_route`.
- Full-path execution remains disabled.
- Do not run real movement tests unless mower is idle, blades are reported off,
  and the user explicitly approves.

## Current movement state as of beta54: raw pymammotion calibration and turn-to-heading

Latest local version:

- `0.6.4-beta54` in `custom_components/mammotion/manifest.json` and
  `pyproject.toml`.

New raw pymammotion probe path:

- `mammotion.raw_pymammotion_motion_probe`
  - HA response service.
  - Reuses the integration's live `pymammotion.MammotionClient` manager/session.
  - Calls `coordinator.manager.send_command_with_args(...)` directly.
  - Supports raw `send_movement(linear_speed, angular_speed)` plus wrapper
    command keys `move_forward`, `move_back`, `move_left`, and `move_right`.
  - Defaults to dry-run and returns the exact pymammotion call it would send.
  - Real probes require `dry_run: false`, `confirm_blades_off: true`, and
    `confirm_clear_area: true`.
  - Samples telemetry at delayed intervals and reports movement/heading deltas.
- `scripts/mammotion_raw_motion_calibration.py`
  - Local helper script that calls the HA service.
  - Saves JSON captures under `/tmp/mammotion_motion_calibration`.
  - Does not implement standalone Mammotion login; it intentionally reuses HA.

Safety gate update:

- `MODE_PAUSE` is now accepted for manual/raw probes when the rest of the gates
  pass.
- Reason: after canceling a mow, the mower can be safely paused with blades off,
  not charging, valid `AREA_INSIDE` position, and HA runtime safety green.

Raw movement calibration results captured on 2026-07-01:

- `send_movement(400, 0)`
  - Moves the mower mostly toward map-local negative Y.
  - Observed movement was around `0.12 m` in one raw probe.
- `send_movement(-400, 0)`
  - Moves the mower mostly toward map-local positive Y.
  - Observed movement was around `0.149 m`.
- `send_movement(0, 180)`
  - Produces weak positive heading change with some drift.
  - Observed heading change was around `+2.0°`.
- `send_movement(0, -180)`
  - Produces weak negative heading change with minimal translation.
  - Observed heading change was around `-4.35°`.
- `send_movement(200, 0)`
  - Moves toward map-local negative Y with smaller step size.
  - Observed movement was around `0.028 m`.
- `send_movement(-200, 0)`
  - Moves toward map-local positive Y with smaller step size.
  - Observed movement was around `0.043 m`.

Current conclusion:

- Raw `send_movement` is a better foundation than the HA wrapper command names.
- Linear positive/negative raw movement is partially calibrated and usable for
  guarded one-dimensional segment work.
- Angular raw movement is measurable but weak; turns require closed-loop repeated
  raw angular pulses and telemetry confirmation before they can be trusted.
- Arbitrary drawn-path execution remains blocked.
- `experimental_execute_segment_burst` remains limited to calibrated-forward
  one-segment behavior and should not be expanded until raw closed-loop control
  is proven in real use.
- `export_runtime_state` now reports `manual_motion_execution_policy` with
  `raw_pymammotion_primitives` so future sessions can see the calibrated command
  map directly from HA.

Implemented in beta52:

- `mammotion.raw_pymammotion_execute_segment`
  - HA response service.
  - Accepts exactly two `{x, y}` map-local points.
  - Defaults to `dry_run: true`.
  - Uses current live telemetry as the real start point; the first supplied
    point is advisory.
  - Rejects segments that require meaningful X/lateral movement:
    `abs(dx) > max(0.10, abs(dy) * 0.35)`.
  - Sends only raw `send_movement(linear_speed, 0)` when real mode is explicitly
    enabled and safety gates pass.
  - Negative map-local Y target selects positive linear speed; positive
    map-local Y target selects negative linear speed.
  - Uses slow speed under `0.15 m` remaining Y distance.
  - Waits for delayed telemetry after every raw command before deciding whether
    another command is safe.
  - Stops on target reached, no target-directed progress, telemetry quality
    degradation, unsafe blade telemetry, unsafe mower state, command failure,
    or max command cap.

First real beta52 Part 1 test:

- Service call:
  - `mammotion.raw_pymammotion_execute_segment`
  - `dry_run: false`
  - `confirm_blades_off: true`
  - `confirm_clear_area: true`
  - `prefer_ble: true`
  - `max_commands: 1`
  - small negative-Y target about `0.10 m` away.
- Command sent:
  - `send_movement(linear_speed=200, angular_speed=0)`
- Result:
  - `stop_reason: target_reached`
  - `commands_sent: 1`
  - measured movement distance about `0.0919 m`
  - target-directed progress about `0.0913 m`
  - movement vector heading about `263.19°` versus expected `270°`
  - final target distance about `0.0139 m`, inside tolerance.
- Conclusion:
  - Raw slow positive linear speed can execute a short negative-Y segment with
    telemetry-confirmed progress and no explicit stop command.

Second real beta52 Part 1 test:

- Service call:
  - `mammotion.raw_pymammotion_execute_segment`
  - `dry_run: false`
  - `confirm_blades_off: true`
  - `confirm_clear_area: true`
  - `prefer_ble: true`
  - `max_commands: 2`
  - farther negative-Y target about `0.22 m` away.
- Command sent:
  - `send_movement(linear_speed=400, angular_speed=0)`
- Result:
  - `stop_reason: target_reached`
  - `commands_sent: 1`
  - measured movement distance about `0.1697 m`
  - target-directed progress about `0.1694 m`
  - movement vector heading about `273.18°` versus expected `270°`
  - final target distance about `0.0515 m`, inside the `0.08 m` tolerance.
- Conclusion:
  - Raw fast positive linear speed can execute a farther negative-Y segment with
    telemetry-confirmed progress. The service correctly avoided a second command
    because the target was reached within tolerance after command one.

Third real beta52 Part 1 test:

- Service call:
  - `mammotion.raw_pymammotion_execute_segment`
  - `dry_run: false`
  - `confirm_blades_off: true`
  - `confirm_clear_area: true`
  - `prefer_ble: true`
  - `max_commands: 1`
  - small positive-Y target about `0.10 m` away.
- Command sent:
  - `send_movement(linear_speed=-200, angular_speed=0)`
- Result:
  - `stop_reason: target_reached`
  - `commands_sent: 1`
  - measured movement distance about `0.1170 m`
  - target-directed progress about `0.1166 m`
  - movement vector heading about `85.54°` versus expected `90°`
  - final target distance about `0.0189 m`, inside the `0.08 m` tolerance.
- Conclusion:
  - Raw slow negative linear speed can execute a short positive-Y segment with
    telemetry-confirmed progress and no explicit stop command.

Implemented in beta53:

- `mammotion.raw_pymammotion_angular_calibration`
  - HA response service.
  - Defaults to `dry_run: true`.
  - Sends only raw angular `send_movement(linear_speed=0, angular_speed=±N)`
    when real mode is explicitly enabled and safety gates pass.
  - `direction: positive_heading` selects positive angular speed.
  - `direction: negative_heading` selects negative angular speed.
  - Repeats up to `max_commands`, waiting for delayed telemetry after each raw
    angular command.
  - Stops on target heading reached, no heading progress, excessive translation,
    unsafe blade telemetry, unsafe mower state, telemetry quality degradation,
    command failure, or max command cap.
  - Returns command results, per-command heading diagnostics, cumulative target
    heading status, telemetry samples, and final telemetry.
  - This is calibration-only; it does not enable turns for path execution.

First real beta53 angular calibration test:

- Service call:
  - `mammotion.raw_pymammotion_angular_calibration`
  - `direction: positive_heading`
  - `angular_speed: 180`
  - `target_heading_delta_degrees: 5`
  - `max_commands: 1`
  - `min_heading_change_degrees: 0.5`
  - `dry_run: false`
  - `confirm_blades_off: true`
  - `confirm_clear_area: true`
  - `prefer_ble: true`
- Command sent:
  - `send_movement(linear_speed=0, angular_speed=180)`
- Result:
  - `stop_reason: max_commands_reached`
  - `commands_sent: 1`
  - heading progress detected and passed
  - heading change about `+4.121°`
  - target-direction progress about `+4.121°`
  - target was `5°`, so the service correctly reported target remaining after
    one capped command
  - translation was only about `0.0049 m`, below the `0.25 m` cap
- Conclusion:
  - Positive raw angular speed produces reliable positive heading change with
    minimal translation. One `angular_speed=180` command yields roughly `4°`
    heading change on this mower/session.

Second real beta53 angular calibration test:

- Service call:
  - `mammotion.raw_pymammotion_angular_calibration`
  - `direction: negative_heading`
  - `angular_speed: 180`
  - `target_heading_delta_degrees: 5`
  - `max_commands: 1`
  - `min_heading_change_degrees: 0.5`
  - `dry_run: false`
  - `confirm_blades_off: true`
  - `confirm_clear_area: true`
  - `prefer_ble: true`
- Command sent:
  - `send_movement(linear_speed=0, angular_speed=-180)`
- Result:
  - `stop_reason: target_heading_reached`
  - `commands_sent: 1`
  - heading progress detected and passed
  - heading change about `-7.325°`
  - target-direction progress about `+7.325°`
  - target was `5°`, so the service correctly reported target reached
  - translation was only about `0.0016 m`, below the `0.25 m` cap
- Conclusion:
  - Negative raw angular speed produces reliable negative heading change with
    minimal translation. One `angular_speed=-180` command yields roughly `7°`
    heading change on this mower/session.

Implemented in beta54:

- `mammotion.raw_pymammotion_turn_to_heading`
  - HA response service.
  - Defaults to `dry_run: true`.
  - Accepts an absolute `target_heading_degrees`.
  - Uses the current live heading as truth and computes the shortest signed
    heading error.
  - Positive heading error selects raw `send_movement(0, +angular_speed)`.
  - Negative heading error selects raw `send_movement(0, -angular_speed)`.
  - Uses slow angular speed when heading error is inside
    `slow_turn_threshold_degrees`.
  - Repeats up to `max_commands`, waiting for delayed telemetry after each raw
    angular command.
  - Stops on target heading reached, no heading progress, excessive
    translation, unsafe blade telemetry, unsafe mower state, telemetry quality
    degradation, command failure, or max command cap.
  - This remains separate from path execution; it does not combine turning with
    linear movement.

First real beta54 turn-to-heading test:

- Service call:
  - `mammotion.raw_pymammotion_turn_to_heading`
  - current heading about `177.0014°`
  - target heading about `185.0014°` (`+8°`)
  - `heading_tolerance_degrees: 3`
  - `angular_speed_fast: 180`
  - `angular_speed_slow: 90`
  - `slow_turn_threshold_degrees: 8`
  - `max_commands: 1`
  - `dry_run: false`
  - `confirm_blades_off: true`
  - `confirm_clear_area: true`
  - `prefer_ble: true`
- Command sent:
  - `send_movement(linear_speed=0, angular_speed=90)`
- Result:
  - `stop_reason: no_heading_progress`
  - `commands_sent: 1`
  - command ack was not false
  - heading change was `0.0°`
  - translation was only about `0.0024 m`
- Conclusion:
  - Slow angular speed `90` may be too weak to rotate the mower in this command
    mode, at least for one firmware nudge. Prior `angular_speed=180` probes did
    move heading. For the next turn-to-heading test, use `angular_speed_slow:
    180` or lower `slow_turn_threshold_degrees` so an `8°` error still uses the
    proven `180` angular speed.

Second real beta54 turn-to-heading test:

- Service call:
  - `mammotion.raw_pymammotion_turn_to_heading`
  - current heading about `177.0014°`
  - target heading about `185.0014°` (`+8°`)
  - `angular_speed_fast: 180`
  - `angular_speed_slow: 180`
  - `max_commands: 1`
- Command sent:
  - `send_movement(linear_speed=0, angular_speed=180)`
- Result:
  - `stop_reason: max_commands_reached`
  - `commands_sent: 1`
  - heading progress detected and passed
  - heading change about `+4.439°`
  - final error about `3.56°`, just outside the `3°` tolerance
  - translation was about `0.0033 m`
- Conclusion:
  - Turn-to-heading selected the correct positive direction and the proven
    angular speed produced heading progress. A second command or slightly wider
    tolerance would have completed the `+8°` target.

Third real beta54 turn-to-heading test:

- Service call:
  - `mammotion.raw_pymammotion_turn_to_heading`
  - current heading about `181.4408°`
  - target heading about `173.4408°` (`-8°`)
  - `angular_speed_fast: 180`
  - `angular_speed_slow: 180`
  - `max_commands: 1`
- Command sent:
  - `send_movement(linear_speed=0, angular_speed=-180)`
- Result:
  - `stop_reason: target_heading_reached`
  - `commands_sent: 1`
  - heading progress detected and passed
  - heading change about `-6.908°`
  - final error about `1.09°`, inside the `3°` tolerance
  - translation was about `0.0026 m`
- Conclusion:
  - Turn-to-heading selected the correct negative direction and reached the
    target in one command.

Implemented in beta55:

- `mammotion.raw_motion_readiness_test`
  - HA response service.
  - Orchestrates existing raw helpers instead of adding new motion primitives.
  - Defaults to `dry_run: true` and `max_real_steps: 0`, so it sends no motion
    commands by default.
  - Dry-run phases validate safety, negative/positive Y segment command
    selection, and positive/negative turn-to-heading command selection.
  - Optional real phases run in fixed order and are capped by `max_real_steps`:
    positive turn, negative turn, negative Y segment, positive Y segment.
  - Real phases require `dry_run: false`, `confirm_blades_off: true`,
    `confirm_clear_area: true`, and `max_real_steps > 0`.
  - Stops on first failed phase and reports `ready_for_vector_segment`,
    `ready_for_multi_point: false`, readiness flags, `failed_phase`, blockers,
    and compact phase results.
- `scripts/mammotion_motion_readiness.py`
  - Calls the HA readiness service.
  - Saves full JSON under `/tmp/mammotion_motion_readiness`.
  - Prints compact readiness summary.

Post-beta55 local work, not deployed yet:

- `mammotion.raw_motion_readiness_test` was deployed and the real
  `max_real_steps: 2` readiness run passed.
  - `ready_for_vector_segment: true`
  - `ready_for_multi_point: false`
  - two real turn-to-heading phases ran
  - positive turn progressed but stopped at `max_commands_reached`
  - negative turn reached target heading in one command
- Implemented local service `mammotion.raw_pymammotion_execute_vector_segment`.
  - One segment only: exactly two points.
  - Dry-run default.
  - Real mode requires `confirm_blades_off` and `confirm_clear_area`.
  - Computes the heading from current live map-local position to the target
    point.
  - Runs `raw_pymammotion_turn_to_heading` first.
  - In real mode, refuses to start linear movement unless the turn phase stops
    with `target_heading_reached`.
  - Then sends raw forward `send_movement(linear_speed, 0)` pulses and measures
    target-directed progress.
  - Keeps `ready_for_multi_point: false`; multi-point execution remains blocked.
- Validation after local implementation:
  - `pytest tests/components/mammotion/test_map_task_visibility.py -q`:
    `122 passed`
  - `ruff check ...`: passed
  - `py_compile`: passed
  - JSON validation: passed
  - `git diff --check`: passed

Next engineering step:

- Bump version, deploy/restart, verify
  `mammotion.raw_pymammotion_execute_vector_segment` registers.
- Run a dry-run vector segment first.
- If dry-run command selection is sane, run one real vector segment with
  conservative settings:
  - `max_turn_commands: 3`
  - `max_linear_commands: 1`
  - explicit operator confirmations
- Do not implement multi-point path execution until combined one-segment vector
  execution is proven.

Beta56 deployment/test result:

- Version bumped to `0.6.4-beta56`, deployed via SMB, and HA Core restarted.
- Verified `mammotion.raw_pymammotion_execute_vector_segment` registered.
- Dry-run vector segment was sane:
  - no blockers
  - valid path
  - target heading about `172.70°`
  - current reported heading about `172.68°`
  - turn phase stopped at `target_heading_reached`
  - linear phase would send `send_movement(linear_speed=200, angular_speed=0)`
- One real vector test ran with:
  - `max_turn_commands: 3`
  - `max_linear_commands: 1`
  - `linear_speed_slow: 200`
  - `angular_speed_fast/slow: 180`
  - explicit blade/clear-area confirmations
- Real result:
  - command ack returned successfully
  - `commands_sent: 1`
  - `turn_commands_sent: 0`
  - `linear_commands_sent: 1`
  - final stop reason: `no_target_progress`
  - measured delta about `0.067 m`
  - measured movement vector heading about `289.16°`
  - expected target heading was about `172.70°`
- Conclusion:
  - The combined vector service plumbing works and safely stopped after one
    bad-progress pulse.
  - The heading model is still wrong: raw forward motion followed the known
    calibrated forward map direction near negative Y, not the reported mower
    heading used by the first vector implementation.
  - Next code change should introduce a calibrated reported-heading-to-forward-
    map-heading offset before turning. The observed offset from this run is
    approximately `+116.5°` (`289.16 - 172.68`), which is close to the earlier
    manually discovered heading offset candidates around `110°`.

Beta57 local change:

- Added `calibrated_forward_heading_offset_degrees` to
  `mammotion.raw_pymammotion_execute_vector_segment`.
  - Default: `116.5°`
  - Formula:
    `target_reported_heading = target_map_heading - calibrated_forward_heading_offset`
  - Response now includes:
    - `target_map_heading_degrees`
    - `target_reported_heading_degrees`
    - `heading_calibration`
  - The turn phase now turns to the calibrated reported heading, not directly
    to the map vector heading.
- Unit tests updated to cover both zero-offset and calibrated-offset dry-runs.
- Validation:
  - map/task test file: `123 passed`
  - ruff: passed
  - py_compile: passed
  - JSON validation: passed
  - `git diff --check`: passed

Beta57 deployment/test result:

- Deployed beta57 via SMB and restarted HA.
- Verified `mammotion.raw_pymammotion_execute_vector_segment` registered.
- Dry-run with `calibrated_forward_heading_offset_degrees: 116.5` was sane:
  - target map heading about `289.21°`
  - target reported heading about `172.71°`
  - initial reported heading about `172.68°`
  - turn phase stopped at `target_heading_reached`
  - linear phase would send `send_movement(200, 0)`
- One real vector pulse ran with:
  - `max_turn_commands: 3`
  - `max_linear_commands: 1`
  - `linear_speed_slow: 200`
  - `angular_speed_fast/slow: 180`
  - `calibrated_forward_heading_offset_degrees: 116.5`
  - explicit blade/clear-area confirmations
- Real result:
  - `stop_reason: target_reached`
  - `commands_sent: 1`
  - `turn_commands_sent: 0`
  - `linear_commands_sent: 1`
  - path-progress diagnostic passed
  - measured movement distance about `0.0574 m`
  - path progress about `0.0543 m`
  - movement vector heading about `270.2°`
  - final waypoint distance about `0.0494 m`, inside the `0.08 m`
    waypoint tolerance
- Conclusion:
  - Calibrated one-segment vector execution is now proven for a small
    already-aligned target.
  - Next step should test a vector segment that requires an actual turn before
    linear movement, still with `max_linear_commands: 1`.
  - Multi-point execution remains blocked until turn+linear vector segments are
    reliable across multiple headings.

Turn+linear vector proof:

- Ran a vector target requiring about `+10°` reported-heading correction before
  forward motion.
- Dry-run was sane:
  - target map heading about `297.9°`
  - target reported heading about `181.4°`
  - turn command would be `send_movement(0, 180)`
  - linear command would be `send_movement(200, 0)`
- Real run:
  - `stop_reason: target_reached`
  - `commands_sent: 4`
  - `turn_commands_sent: 3`
  - `linear_commands_sent: 1`
  - turn phase reached target heading:
    - final reported heading about `181.45°`
    - target reported heading about `181.40°`
    - final heading error about `0.05°`
  - linear phase target-directed progress passed:
    - movement distance about `0.057 m`
    - target-directed progress about `0.056 m`
    - final waypoint distance about `0.046 m`, inside the `0.08 m` tolerance
- Conclusion:
  - Calibrated vector execution is now proven for:
    1. already-aligned one-pulse vector movement
    2. turn-then-one-pulse vector movement
  - Next safe step is to test the opposite turn direction with one linear
    pulse. If that passes, implement a guarded multi-segment dry-run/execution
    wrapper that runs these one-segment vector steps sequentially with
    `dry_run: true` default and a very low default segment limit.

Opposite-direction turn+linear vector proof:

- Ran a vector target requiring about `-10°` reported-heading correction before
  forward motion.
- Dry-run was sane:
  - target map heading about `287.95°`
  - target reported heading about `171.45°`
  - turn command would be `send_movement(0, -180)`
  - linear command would be `send_movement(200, 0)`
- Real run:
  - `stop_reason: target_reached`
  - `commands_sent: 4`
  - `turn_commands_sent: 3`
  - `linear_commands_sent: 1`
  - turn phase reached target heading:
    - final reported heading about `172.02°`
    - target reported heading about `171.45°`
    - final heading error about `0.58°`
  - turn loop corrected overshoot safely:
    - command 1: `send_movement(0, -180)`
    - command 2: `send_movement(0, -180)`
    - command 3: `send_movement(0, 180)`
  - linear phase target-directed progress passed:
    - movement distance about `0.024 m`
    - target-directed progress about `0.023 m`
    - final waypoint distance about `0.078 m`, inside the `0.08 m` tolerance
- Conclusion:
  - Calibrated vector execution is now proven for both positive and negative
    turn corrections plus a guarded forward pulse.
  - Next implementation step can be a guarded multi-segment wrapper, but keep
    defaults conservative:
    - `dry_run: true`
    - max real segments very low, e.g. `1`
    - one linear command per segment by default
    - stop on first segment failure
    - re-run full runtime safety checks between segments

Beta58 local change:

- Added consolidated vector readiness workflow:
  - HA response service: `mammotion.raw_vector_readiness_test`
  - Script: `scripts/mammotion_vector_readiness.py`
- The service reuses `raw_pymammotion_execute_vector_segment` internally and
  runs phases in order:
  1. `safety_snapshot`
  2. `dry_run_aligned_vector`
  3. `dry_run_positive_turn_vector`
  4. `dry_run_negative_turn_vector`
  5. optional `real_aligned_vector`
  6. optional `real_positive_turn_vector`
  7. optional `real_negative_turn_vector`
- Defaults:
  - `dry_run: true`
  - `max_real_steps: 0`
  - `target_distance: 0.10`
  - `turn_delta_degrees: 10`
  - `calibrated_forward_heading_offset_degrees: 116.5`
  - `max_turn_commands: 3`
  - `max_linear_commands: 1`
- Real mode requires:
  - `dry_run: false`
  - `confirm_blades_off: true`
  - `confirm_clear_area: true`
  - `max_real_steps > 0`
- Response summary includes:
  - `aligned_vector_ready`
  - `positive_turn_vector_ready`
  - `negative_turn_vector_ready`
  - `ready_for_multi_segment`
  - `ready_for_multi_point: false`
  - `failed_phase`
  - `blockers`
  - `recommended_next_step`
- Validation:
  - `pytest tests/components/mammotion/test_map_task_visibility.py -q`:
    `127 passed`
  - ruff: passed
  - py_compile: passed
  - JSON validation: passed
  - `git diff --check`: passed
- Next rollout:
  - deploy/restart beta58
  - verify `mammotion.raw_vector_readiness_test` registers
  - run dry-run vector readiness first
  - only then run real vector readiness with `max_real_steps` 1, then 3 if
    desired and safety remains clear

Beta58 deployment/test result:

- Deployed beta58 via SMB and restarted HA.
- Verified `mammotion.raw_vector_readiness_test` registered.
- First dry-run immediately after restart returned `{}` because the Mammotion
  runtime had not finished attaching; retry after `export_runtime_state`
  returned live position worked.
- Consolidated vector dry-run passed:
  - `ready_for_multi_segment: true`
  - `ready_for_multi_point: false`
  - `aligned_vector_ready: true`
  - `positive_turn_vector_ready: true`
  - `negative_turn_vector_ready: true`
  - `failed_phase: null`
  - `recommended_next_step: implement_guarded_multi_segment_wrapper`
- Dry-run command plan:
  - aligned vector:
    - turn already reached target
    - linear: `send_movement(200, 0)`
  - positive-turn vector:
    - turn: `send_movement(0, 180)`
    - linear: `send_movement(200, 0)`
  - negative-turn vector:
    - turn: `send_movement(0, -180)`
    - linear: `send_movement(200, 0)`
- Saved dry-run output:
  `/tmp/mammotion_vector_readiness/20260703-000150-raw-vector-readiness.json`

Post-beta58 consolidation script update:

- Added shared script helper `scripts/mammotion_ha_helpers.py`.
  - Loads `.env`.
  - Calls HA response services.
  - Waits for Mammotion runtime state to include live `x/y/toward`.
  - Retries readiness service calls that initially return `{}` after restart.
- Updated:
  - `scripts/mammotion_motion_readiness.py`
  - `scripts/mammotion_vector_readiness.py`
- Both readiness scripts now:
  - wait for runtime by default
  - support `--no-wait-runtime`
  - support `--runtime-timeout`
  - print `integration_not_ready` in compact output
- Added canonical suite script:
  - `scripts/mammotion_motion_suite.py`
  - Defaults to non-moving vector dry-run.
  - Runs `mammotion.raw_vector_readiness_test`.
  - Saves artifacts under `/tmp/mammotion_motion_suite/<timestamp>/`.
  - Writes:
    - `raw_vector_readiness.json`
    - `summary.json`
- Added helper tests:
  - empty readiness response is retried after runtime wait
  - `--no-wait-runtime` style calls do not retry
- Validation:
  - `pytest tests/components/mammotion/test_map_task_visibility.py tests/components/mammotion/test_motion_scripts.py -q`:
    `129 passed`
  - ruff: passed
  - py_compile: passed
  - JSON validation/diff check: passed
- Suite dry-run result:
  - command:
    `python3 scripts/mammotion_motion_suite.py lawn_mower.back_yard_clip_skywalker --dry-run --max-real-steps 0 --sample-delays 0 5 10`
  - `passed: true`
  - `integration_not_ready: false`
  - `ready_for_multi_segment: true`
  - `ready_for_multi_point: false`
  - output directory:
    `/tmp/mammotion_motion_suite/20260703-000959`
- Suite real smoke result:
  - command:
    `python3 scripts/mammotion_motion_suite.py lawn_mower.back_yard_clip_skywalker --real --max-real-steps 1 --confirm-blades-off --confirm-clear-area`
  - output directory:
    `/tmp/mammotion_motion_suite/20260703-001200`
  - `passed: true`
  - `ready_for_multi_segment: true`
  - `ready_for_multi_point: false`
  - `real_steps_run: 1`
  - real phase: `real_aligned_vector`
  - `commands_sent: 1`
  - `turn_commands_sent: 0`
  - `linear_commands_sent: 1`
  - `stop_reason: target_reached`
  - movement distance about `0.073 m`
  - target-directed progress about `0.071 m`
  - final waypoint distance about `0.032 m`
  - final runtime state remained safe: paused, not charging, blades off, RPM 0,
    valid position, no safety blockers
- Suite real full result with default `max_real_steps: 3` /
  `max_turn_commands: 3`:
  - command:
    `python3 scripts/mammotion_motion_suite.py lawn_mower.back_yard_clip_skywalker --real --max-real-steps 3 --confirm-blades-off --confirm-clear-area`
  - output directory:
    `/tmp/mammotion_motion_suite/20260703-001343`
  - `passed: false`
  - failed phase: `real_positive_turn_vector`
  - `real_aligned_vector` passed:
    - `commands_sent: 1`
    - `linear_commands_sent: 1`
    - `stop_reason: target_reached`
  - `real_positive_turn_vector` failed safely before linear movement:
    - `stop_reason: turn_phase_incomplete`
    - turn subphase stopped at `max_commands_reached`
    - final heading error about `3.39°`, just outside the `3.0°` tolerance
    - no linear command was sent for the failed phase
  - final runtime state remained safe: paused, not charging, blades off, RPM 0,
    valid position, no safety blockers
  - likely next test/config adjustment: retry full suite with
    `--max-turn-commands 4` before changing code defaults
- Suite real full retry with `--max-turn-commands 4`:
  - command:
    `python3 scripts/mammotion_motion_suite.py lawn_mower.back_yard_clip_skywalker --real --max-real-steps 3 --max-turn-commands 4 --confirm-blades-off --confirm-clear-area`
  - output directory:
    `/tmp/mammotion_motion_suite/20260703-090529`
  - `passed: false`
  - failed phase: `real_negative_turn_vector`
  - `real_aligned_vector` passed:
    - `commands_sent: 1`
    - `linear_commands_sent: 1`
    - `stop_reason: target_reached`
  - `real_positive_turn_vector` passed with the extra turn command:
    - `turn_commands_sent: 4`
    - `linear_commands_sent: 1`
    - `stop_reason: target_reached`
  - `real_negative_turn_vector` reached target heading but did not reach the
    waypoint with one linear pulse:
    - `turn_commands_sent: 2`
    - `linear_commands_sent: 1`
    - `stop_reason: max_linear_commands_reached`
    - target-directed progress passed, about `0.018 m`
    - final waypoint distance about `0.103 m`
  - final runtime state remained safe: paused, not charging, blades off, RPM 0,
    valid position, no safety blockers
  - likely next test/config adjustment: retry full suite with
    `--max-turn-commands 4 --max-linear-commands 2`
- Suite real full retry with `--max-turn-commands 4 --max-linear-commands 2`:
  - command:
    `python3 scripts/mammotion_motion_suite.py lawn_mower.back_yard_clip_skywalker --real --max-real-steps 3 --max-turn-commands 4 --max-linear-commands 2 --confirm-blades-off --confirm-clear-area`
  - output directory:
    `/tmp/mammotion_motion_suite/20260703-091542`
  - `passed: true`
  - `ready_for_multi_segment: true`
  - `ready_for_multi_point: false`
  - `real_steps_run: 3`
  - all real vector phases passed:
    - `real_aligned_vector`: `commands_sent: 1`, `turn: 0`, `linear: 1`
    - `real_positive_turn_vector`: `commands_sent: 4`, `turn: 3`,
      `linear: 1`
    - `real_negative_turn_vector`: `commands_sent: 3`, `turn: 2`,
      `linear: 1`
  - all phases stopped at `target_reached`
  - final runtime state remained safe: paused, not charging, blades off, RPM 0,
    valid position, no safety blockers
  - proven consolidated readiness settings:
    - `max_turn_commands: 4`
    - `max_linear_commands: 2`
    - `max_real_steps: 3`
  - next code change should make the script/service defaults match the proven
    turn/linear command limits before implementing guarded multi-segment
    execution

Beta59 local change:

- Updated consolidated vector readiness defaults to match the proven full-suite
  configuration:
  - `raw_vector_readiness_test.max_turn_commands` default: `4`
  - `raw_vector_readiness_test.max_linear_commands` default: `2`
  - `scripts/mammotion_vector_readiness.py --max-turn-commands` default: `4`
  - `scripts/mammotion_vector_readiness.py --max-linear-commands` default: `2`
  - `scripts/mammotion_motion_suite.py --max-turn-commands` default: `4`
  - `scripts/mammotion_motion_suite.py --max-linear-commands` default: `2`
- Kept the lower-level one-segment vector primitive defaults conservative:
  - `raw_pymammotion_execute_vector_segment.max_turn_commands`: `3`
  - `raw_pymammotion_execute_vector_segment.max_linear_commands`: `1`
- Validation:
  - `pytest tests/components/mammotion/test_map_task_visibility.py tests/components/mammotion/test_motion_scripts.py -q`:
    `129 passed`
  - ruff: passed
  - py_compile: passed
  - JSON validation and `git diff --check`: passed
- Deployment/test:
  - Deployed beta59 via SMB and restarted HA.
  - Verified `mammotion.raw_vector_readiness_test` registered.
  - HA service metadata reports:
    - `max_turn_commands.default: 4`
    - `max_linear_commands.default: 2`
  - Suite dry-run with default turn/linear settings passed:
    - command:
      `python3 scripts/mammotion_motion_suite.py lawn_mower.back_yard_clip_skywalker --dry-run --max-real-steps 0 --sample-delays 0 5 10`
    - output directory:
      `/tmp/mammotion_motion_suite/20260703-092842`
    - `passed: true`
    - `ready_for_multi_segment: true`
    - `ready_for_multi_point: false`
    - `failed_phase: null`

## HA deployment guidance

When ready to deploy:

- Deploy only after checking current working tree and tests.
- Do not restart HA unless explicitly approved.
- If only deploying code without restart, HA may not show manifest/entity changes until restart.
- Credentials live in `.env`, which is gitignored. Do not paste tokens/passwords into docs, commits, logs, or chat unless intentionally rotating them afterward.
- This HA install is using direct file copy for this custom component. HACS on the HA host may still point at `mikey0000/Mammotion-HA`, so HACS' displayed version/source can differ from the directly deployed files.

Known HA deployment facts from Claude handoff:

- HA host: `192.168.1.106`
- HA API base URL used by Claude: `http://192.168.1.106:8123`
- Mammotion config entry ID found by Claude: `01KVM3JVYBWRKM25ZR8T7FKKJ3`
- SMB share used by Claude: `//homeassistant@192.168.1.106/config`
- SSH port used by Claude: `2224`
- SSH key note: the key is passphrase-protected; load it into the SSH agent first if using SSH/SFTP.
- Preferred reload API on this HA version:

  ```text
  POST /api/services/homeassistant/reload_config_entry
  Body: {"entry_id": "01KVM3JVYBWRKM25ZR8T7FKKJ3"}
  Authorization: Bearer <HA_TOKEN from .env>
  ```

Safe deploy outline:

1. Validate locally.
2. Copy `custom_components/mammotion/` to `/config/custom_components/mammotion/` on HA.
3. Reload the Mammotion config entry through HA's service endpoint.
4. Check HA logs and config entry state.
5. Restart HA only if explicitly approved or if reload is insufficient for manifest/platform discovery.

Suggested local validation before deploy:

```bash
.venv/bin/python -m py_compile custom_components/mammotion/coordinator.py custom_components/mammotion/number.py custom_components/mammotion/select.py custom_components/mammotion/button.py custom_components/mammotion/switch.py custom_components/mammotion/sensor.py custom_components/mammotion/services.py custom_components/mammotion/config_flow.py custom_components/mammotion/__init__.py
.venv/bin/python -m json.tool custom_components/mammotion/manifest.json >/tmp/mammotion_manifest_check
.venv/bin/python -m json.tool custom_components/mammotion/strings.json >/tmp/mammotion_strings_check
.venv/bin/python -m json.tool custom_components/mammotion/translations/en.json >/tmp/mammotion_en_check
.venv/bin/python -m json.tool custom_components/mammotion/icons.json >/tmp/mammotion_icons_check
git diff --check
.venv/bin/python -m pytest tests/components/mammotion/test_map_task_visibility.py tests/components/mammotion/test_pymammotion_compat.py tests/components/mammotion/test_config_flow.py tests/components/mammotion/test_camera_hotfix.py
```

Previously these tests passed:

- `17 passed`

## Known local files from prior status

Modified/untracked files seen earlier included:

- `custom_components/mammotion/__init__.py`
- `custom_components/mammotion/config_flow.py`
- `custom_components/mammotion/coordinator.py`
- `custom_components/mammotion/manifest.json`
- `custom_components/mammotion/number.py`
- `custom_components/mammotion/select.py`
- `custom_components/mammotion/button.py`
- `custom_components/mammotion/switch.py`
- `custom_components/mammotion/sensor.py`
- `custom_components/mammotion/services.py`
- `custom_components/mammotion/services.yaml`
- `custom_components/mammotion/strings.json`
- `custom_components/mammotion/translations/en.json`
- `custom_components/mammotion/icons.json`
- `custom_components/mammotion/pymammotion_compat.py`
- `tests/components/mammotion/test_map_task_visibility.py`
- `tests/components/mammotion/test_pymammotion_compat.py`
- `Mammotion_2.3.8.19_APKPure.xapk`

Because another chat may have made additional changes, always re-run `git status --short` and `git diff --stat` before deciding what to keep.

## Executable completion checklist (finish the integration)

Use this section as the release-control checklist for finishing this integration branch. Mark each checkbox as work is completed. A phase only passes when every gate in that phase is satisfied.

### Phase 1 - Scope freeze and baseline hardening

Objective: freeze release scope and establish a deterministic baseline.

Current execution status (2026-07-03):

- Phase state: `PASS (SCOPED WAIVER)`
- Current gate verdict: `PASS` (ship-scope quality evidence is complete; repo-wide pre-existing lint/type debt remains tracked separately)
- Evidence captured this session:
  - `git status --short` and `git diff --stat` run
  - targeted motion/service tests passed: `146 passed`
  - targeted changed-file lint passed
  - translation JSON parse passed (`strings.json`, `translations/en.json`)
  - `services.yaml` parse passed
  - full `mypy custom_components/` still reports broad pre-existing type issues

Working tree file classification:

- `ship now`:
  - `custom_components/mammotion/services.py`
  - `custom_components/mammotion/services.yaml`
  - `custom_components/mammotion/strings.json`
  - `custom_components/mammotion/translations/en.json`
  - `custom_components/mammotion/manifest.json`
  - `pyproject.toml`
  - `scripts/mammotion_ha_helpers.py`
  - `scripts/mammotion_motion_readiness.py`
  - `scripts/mammotion_vector_readiness.py`
  - `scripts/mammotion_motion_suite.py`
  - `scripts/mammotion_raw_motion_calibration.py`
  - `scripts/mammotion_position_feedback_diagnostic.py`
  - `scripts/mammotion_forward_two_pulse_latency.py`
  - `tests/components/mammotion/test_map_task_visibility.py`
  - `tests/components/mammotion/test_motion_scripts.py`
- `hold`:
  - `scripts/mammotion_agora_audio_probe.py` (outside guarded motion finish scope)
  - `docs/codex-working-plan.md` (tracking/handoff only)
  - `uv.lock` (modified during local tooling runs; decide keep/revert before RC)
- `drop`:
  - none currently

Checklist:

- [x] Classify every modified/untracked file as `ship now`, `hold`, or `drop`.
- [x] Keep motion execution scope limited to guarded one-segment and guarded multi-segment wrappers (no full autonomous arbitrary path mode).
- [x] Keep camera/cloud reliability and diagnostics in release scope.
- [x] Confirm no unsafe or secret-bearing artifacts are introduced (tokens, passwords, `.env` data, large APK/XAPK unless intentionally approved).
- [x] Run: `git status --short` and verify all remaining files are expected.
- [x] Run: `git diff --stat` and verify change volume matches intended scope.
- [x] Run quality baseline:
  - `uv run pytest`
  - `uv run ruff check`
  - `uv run mypy custom_components/`

Pass/fail gate:

- PASS when all checklist items above are complete and quality baseline evidence is captured for RC decisions.
- FAIL if any file is unclassified, quality baseline steps were not executed, or scope includes unapproved expansion.

### Phase 2 - Motion path stabilization (guarded multi-segment)

Objective: make guarded multi-segment behavior predictable, safe, and diagnosable.

Current execution status (2026-07-03):

- Phase state: `PASS`
- Dry-run validation result: `PASS`
- Artifact: `/tmp/mammotion_motion_suite/20260703-204550`
- Summary:
  - `ready_for_multi_segment: true`
  - `aligned_vector_ready: true`
  - `positive_turn_vector_ready: true`
  - `negative_turn_vector_ready: true`
  - `multi_segment_dry_run_passed: true`
  - `failed_phase: null`
  - `passed: true`
- Stop-reason audit:
  - reviewed motion stop reasons in `services.py`
  - verified snake_case machine-readable values used consistently in tested paths
  - confirmed existing assertions in `test_map_task_visibility.py` cover stop-reason keys across motion services
- Supervised real-smoke and guarded multi-segment real results (guarded, explicit confirmations):
  - `/tmp/mammotion_motion_suite/20260703-223857` (real-smoke pass)
  - `/tmp/mammotion_motion_suite/20260703-232917` (quick real-smoke + multi-segment real pass)
  - summary from `/tmp/mammotion_motion_suite/20260703-232917`:
    - `ready_for_multi_segment: true`
    - `failed_phase: null`
    - `multi_segment_real_passed: true`
    - `multi_segment_real_segments_executed: 1`
    - `passed: true`

Checklist:

- [x] Confirm `mammotion.raw_vector_readiness_test` remains the gate before real movement steps.
- [x] Confirm `mammotion.raw_pymammotion_execute_multi_segment` defaults stay conservative:
  - dry run default on
  - low `max_real_segments`
  - stop on first segment failure
- [x] Ensure per-segment runtime safety is revalidated before each segment.
- [x] Ensure post-command telemetry refresh path is consistently used before evaluating progress.
- [x] Normalize stop reasons/blockers to stable machine-readable keys.
- [x] Verify script defaults and service defaults align with proven settings for turn/linear command caps.
- [x] Run dry-run readiness and dry-run multi-segment end-to-end using the standard scripts.
- [x] Run supervised real smoke for guarded movement only after explicit confirmations and safe runtime state.

Pass/fail gate:

- PASS when dry-run flows are stable, supervised real smoke passes, and first failure halts safely with clear blockers.
- FAIL if movement proceeds without required confirmations, if blockers are ambiguous, or if segment failure does not halt immediately.

Current gate verdict: `PASS`.

## Implementation Task List Status: Click/Go Minimum Safe Slice (2026-07-04)

### 1) Card Changes

- [x] Add a Real Go button that is separate from map click and separate from Run dry-run.
- [x] Keep map click behavior non-moving: click only updates target point and triggers preview/dry-run planning UX.
- [x] Build one-segment request from live mower position + clicked point.
- [x] Add preflight state panel in card UI sourced from runtime export:
  - [x] Show active_transport, blade-safe status, mowing/charging readiness, route-blocking status.
- [x] Gate Real Go button disabled unless all required preflight conditions are green.
- [x] Add explicit operator confirmations in UI for real run:
  - [x] confirm_blades_off
  - [x] confirm_clear_area
- [x] Surface backend stop_reason and blockers directly in status output.
- [x] Add Abort/Stop action wired to existing directional/manual stop mechanism available in integration services (no new autonomous logic).

### 2) Payload Shape (Service Calls)

- [x] Use existing read-only preflight source first: mammotion.export_runtime_state (entity-scoped).
- [x] Continue using preview planner before real run: mammotion.preview_custom_path and/or mammotion.dry_run_custom_path.
- [x] For real movement, call only existing guarded one-segment service: mammotion.raw_pymammotion_execute_vector_segment.
- [x] Enforce one-segment, two-point payload only: points = [current_position, clicked_target].
- [x] Keep conservative defaults in card-issued real payload:
  - [x] dry_run: false
  - [x] confirm_blades_off: true (required in UI before run)
  - [x] confirm_clear_area: true (required in UI before run)
  - [x] prefer_ble: true (default unless config override)
  - [x] max_turn_commands: 1
  - [x] max_linear_commands: 1
  - [x] short fixed sample_delays profile

### 3) Tests

- [x] Add backend-focused tests that lock the click/go safety contract:
  - [x] Two-point enforcement remains strict for one-segment vector execution.
  - [x] Real run requires confirmations and returns expected blocker keys when missing.
  - [x] Runtime unsafe states produce deterministic stop_reason/blockers.
  - [x] Conservative command limits are respected (max_turn_commands, max_linear_commands).
- [x] Add script-level test coverage for payload generation for one-segment click/go path:
  - [x] Ensures start point = live runtime position and end point = chosen target.
  - [x] Ensures dry_run default behavior remains non-moving.
- [x] Add/extend regression assertions for non-goal behavior:
  - [x] No change to blocked full-path/arbitrary execution policy.
  - [x] ready_for_multi_point semantics remain unchanged.

### 4) Acceptance Criteria Per File

- [x] mammotion-custom-path-card.js:
  - [x] User can click a target point and see updated dry-run/preview without movement.
  - [x] Real Go is disabled when preflight is unsafe and enabled only when safe.
  - [x] Real Go sends exactly one guarded one-segment request.
  - [x] Result panel shows backend stop_reason and blocker details clearly.
  - [x] Card still works for existing preview-only workflows.
- [x] services.py:
  - [x] No new autonomous/full-path execution behavior introduced.
  - [x] Existing guarded one-segment behavior unchanged except intentionally configured parameters from UI.
  - [x] Existing safety gates remain authoritative and unchanged in semantics.
- [x] services.yaml:
  - [x] Service fields and defaults used by card match actual schemas.
  - [x] No misleading metadata suggesting autonomous path execution.
- [x] strings.json:
  - [x] User-facing descriptions for click/go flow explicitly state guarded one-segment behavior.
  - [x] Copy continues to state that full arbitrary path execution is not enabled.
- [x] test_map_task_visibility.py:
  - [x] New/updated tests pass and lock one-segment safety behavior and blockers.
- [x] test_motion_scripts.py:
  - [x] New/updated payload and dry-run default tests pass for one-segment flow scaffolding.

### 5) Done Gate (This Slice)

- [x] uv run pytest passes. (172 passed)
- [x] uv run pytest tests/components/mammotion/test_map_task_visibility.py tests/components/mammotion/test_motion_scripts.py -q passes. (166 passed)
- [x] Manual HA validation: click target -> dry-run plan -> Real Go guarded run -> clear stop_reason surfaced.
- [x] No regression in existing Phase 5 checks in codex-working-plan.md.

Automation assist (2026-07-04):

- Added scripted click/go smoke flow in `scripts/mammotion_click_go_smoke.py`:
  - step 1: `export_runtime_state` wait/readiness
  - step 2: `preview_custom_path` for one-segment preview validation
  - step 3: guarded one-segment dry-run via `raw_pymammotion_execute_vector_segment`
  - optional step 4: guarded real one-segment execution only with explicit arming flags
- Added script tests in `tests/components/mammotion/test_motion_scripts.py` covering target selection and preview payload shape.
- Executed dry-run smoke command against HA:
  - `uv run python scripts/mammotion_click_go_smoke.py lawn_mower.back_yard_clip_skywalker --offset-x 0.1 --offset-y 0.0 --runtime-timeout 180 --timeout 240`
  - artifact: `/tmp/mammotion_click_go_smoke/20260704-113757`
  - summary: `preview_valid=true`, `dry_run_stop_reason=dry_run`, `dry_run_blockers=[]`
- Executed guarded real-run smoke command against HA:
  - `uv run python scripts/mammotion_click_go_smoke.py lawn_mower.back_yard_clip_skywalker --offset-x 0.1 --offset-y 0.0 --runtime-timeout 180 --timeout 240 --run-real --confirm-blades-off --confirm-clear-area`
  - artifact: `/tmp/mammotion_click_go_smoke/20260704-114419`
  - summary: `preview_valid=true`, `dry_run_stop_reason=dry_run`, `real_run_stop_reason=turn_phase_incomplete`, `real_run_blockers=[]`
- Added operator-facing script usage documentation to `README.md` for the guarded click/go smoke flow.
- Added helper input-validation coverage in `tests/components/mammotion/test_motion_scripts.py`:
  - invalid runtime `position` type raises `TypeError`
  - missing target `x/y` raises `ValueError`

### Phase 3 - Observability and operator recovery

Objective: expose actionable camera/cloud/motion diagnostics and recovery actions.

Current execution status (2026-07-03):

- Phase state: `PASS`
- Added recovery buttons:
  - `button.<mower>_refresh_camera_stream`
  - `button.<mower>_refresh_cloud_session`
- Added diagnostic sensors:
  - `sensor.<mower>_active_transport`
  - `sensor.<mower>_ble_only_fallback_mode`
  - `sensor.<mower>_last_cloud_login_success`
  - `sensor.<mower>_last_token_refresh`
  - `sensor.<mower>_last_command_failure_reason`
  - `sensor.<mower>_last_camera_stream_failure_code`
- Exposed the same diagnostics through `mammotion.export_runtime_state`.
- Focused regression/tests: `151 passed`.

Checklist:

- [x] Add/verify `button.<mower>_refresh_camera_stream`.
- [x] Add/verify `button.<mower>_refresh_cloud_session`.
- [x] Add/verify diagnostic visibility for:
  - active transport
  - last cloud login success
  - last token refresh
  - last command failure reason
  - last camera stream failure code
  - BLE-only fallback mode indicator
- [x] Ensure diagnostics are available through existing runtime export surfaces and/or entity state.
- [x] Add/extend tests for failure-path visibility and recovery button behavior.

Pass/fail gate:

- PASS when common operator failures are diagnosable without deep log inspection and recovery actions are callable from HA.
- FAIL if failures still collapse into generic unavailable/unknown outcomes.

### Phase 4 - Localization and metadata completeness

Objective: enforce translation, icon, and service metadata completeness.

Checklist:

- [x] Keep `custom_components/mammotion/strings.json` aligned with all entity/service keys in code.
- [x] Update every locale under `custom_components/mammotion/translations/` for any new/renamed entity or enum state.
- [x] Ensure `custom_components/mammotion/icons.json` has entries for new entities where applicable.
- [x] Validate JSON parse for:
  - `custom_components/mammotion/manifest.json`
  - `custom_components/mammotion/strings.json`
  - all files under `custom_components/mammotion/translations/`
  - `custom_components/mammotion/icons.json`
- [x] Verify `custom_components/mammotion/services.yaml` descriptions/options match implemented service schemas.

Current execution status (2026-07-04 delta):

- Added explicit click/go-aligned service wording in:
  - `custom_components/mammotion/services.yaml`
  - `custom_components/mammotion/strings.json`
- Synced matching English translation service text in:
  - `custom_components/mammotion/translations/en.json`
- Updated/added descriptions so card-facing flow language is consistent for:
  - `export_runtime_state`
  - `preview_custom_path`
  - `execute_custom_path`
  - `raw_pymammotion_motion_probe`
  - `raw_pymammotion_execute_vector_segment`
- Validation evidence:
  - `python3 -m json.tool custom_components/mammotion/strings.json` passed
  - `python3 -m json.tool custom_components/mammotion/translations/en.json` passed
  - `python3 -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('custom_components/mammotion/services.yaml').read_text())"` passed

Pass/fail gate:

- PASS when translation key parity and JSON validity checks are complete for all locales and metadata files.
- FAIL if any locale is missing keys/state labels or any metadata JSON is invalid.

### Phase 5 - Release candidate and deployment verification

Objective: produce a reproducible release candidate and verify runtime behavior post-deploy.

Checklist:

- [x] Bump version in both:
  - `custom_components/mammotion/manifest.json`
  - `pyproject.toml`
- [x] Run targeted and full validation before deploy:
  - `uv run pytest`
  - `uv run ruff check`
  - `uv run mypy custom_components/`
  - `git diff --check`
- [x] Deploy `custom_components/mammotion/` to HA host.
- [x] Reload Mammotion config entry through HA service endpoint.
- [x] Verify post-reload:
  - expected services are registered
  - expected entities are present
  - dry-run motion suite passes
  - one supervised real smoke run (guarded scope only) passes
- [x] Capture artifact paths and compact results in this doc for handoff.

Implementation progress snapshot (2026-07-04):

- Initial low-risk test consolidation started:
  - schema-default coverage consolidated into a parameterized matrix in
    `tests/components/mammotion/test_map_task_visibility.py`
  - quick-profile behavior coverage added for both motion scripts in
    `tests/components/mammotion/test_motion_scripts.py`
- Focused regression after consolidation:
  - `uv run pytest tests/components/mammotion/test_motion_scripts.py tests/components/mammotion/test_map_task_visibility.py -q`
  - result: `158 passed`
- Full validation run status:
  - `uv run pytest`: `172 passed in 4.11s`
  - `uv run ruff check`: FAIL (pre-existing broad repository lint debt remains; latest run wrote `1421` lines to `/var/folders/8g/nnl_fh6d1r3d70rn_g3_7ts80000gn/T//mammotion_ruff_phase5.txt` and includes existing `I001`/`C901`/`PERF401`/`SLF001` findings)
  - `uv run mypy custom_components/`: FAIL (`.venv/lib/python3.14/site-packages/homeassistant/helpers/device_registry.py:449` parser compatibility error prevented further checking)
  - `git diff --check`: PASS
  - `uv run ruff check scripts/mammotion_motion_suite.py`: PASS after local import ordering fix and targeted `C901` suppression on script `main()`
- Known debt summary (PR-ready, pre-existing):
  - Ruff debt is concentrated in a small set of recurring rule families, not in the new RC changes:
    - `SLF001` (31): private-member access, mostly tests (`tests/components/mammotion/test_camera_hotfix.py`).
    - `C901` (19): legacy complexity in large async/service entrypoints.
    - `PERF401` (18): loop-to-comprehension/`extend` perf suggestions.
    - `TRY300` (11): stylistic return-in-`else` cleanup opportunities.
  - File concentration (top pre-existing hotspots):
    - `tests/components/mammotion/test_camera_hotfix.py` (29)
    - `custom_components/mammotion/agora_websocket.py` (16)
    - `custom_components/mammotion/__init__.py` (16)
    - `custom_components/mammotion/coordinator.py` (11)
    - `custom_components/mammotion/agora_sdp.py` (10)
  - Mypy blocker is currently environmental/external to integration logic:
    - `.venv/lib/python3.14/site-packages/homeassistant/helpers/device_registry.py:449` parser compatibility error aborts analysis early (`errors prevented further checking`).
  - Risk/Impact: RC functional risk is low for merged scope because runtime and integration tests pass (`172 passed`), while remaining failures are pre-existing lint/type debt that can slow future maintenance but do not indicate new runtime regressions from this change set.
  - Out of scope debt cleanup plan:
    - Unblock `mypy` first by aligning the tool/interpreter environment so analysis can run through `custom_components/` without early parser failure.
    - Triage and fix lint by highest-yield buckets in order: `SLF001` (test-only private access), `C901` (complex entrypoints), then `PERF401`/`TRY300` (non-functional style/perf cleanups).
    - Execute cleanup in small follow-up PRs by subsystem (`agora_*`, coordinator, tests) with no behavior changes, each gated by `uv run pytest` and `uv run ruff check` for touched files.
- HA reload + runtime verification status (2026-07-03/04):
  - reload service endpoint behavior on this HA build:
    - `POST /api/services/homeassistant/reload_config_entry?return_response` -> `400` with message that this service does not support responses.
    - `POST /api/services/homeassistant/reload_config_entry` -> succeeded with `200` and `[]` when allowing a longer client timeout.
  - post-reload service registration check: present
    - `mammotion.raw_vector_readiness_test`
    - `mammotion.raw_pymammotion_execute_multi_segment`
    - `mammotion.export_runtime_state`
    - `mammotion.export_active_route`
  - post-reload entity presence/state check:
    - `lawn_mower.back_yard_clip_skywalker` -> `paused`
    - `sensor.back_yard_clip_skywalker_active_transport` -> `ble`
    - refresh button entities present via state API
  - post-reload dry-run motion suite pass:
    - command: `uv run python scripts/mammotion_motion_suite.py lawn_mower.back_yard_clip_skywalker --dry-run --max-real-steps 0 --include-multi-segment-dry-run --sample-delays 0 5 10 --runtime-timeout 180 --timeout 240`
    - artifact: `/tmp/mammotion_motion_suite/20260703-234650`
    - summary: `passed: true`, `ready_for_multi_segment: true`, `multi_segment_dry_run_passed: true`
  - fresh dry-run artifact (current session):
    - command: `uv run python scripts/mammotion_motion_suite.py lawn_mower.back_yard_clip_skywalker --dry-run --max-real-steps 0 --include-multi-segment-dry-run --sample-delays 0 5 10 --runtime-timeout 180 --timeout 240`
    - artifact: `/tmp/mammotion_motion_suite/20260703-235037`
    - summary: `passed: true`, `ready_for_multi_segment: true`, `multi_segment_dry_run_passed: true`, `failed_phase: null`
  - fresh HA entity verification for new diagnostics/recovery controls:
    - `sensor.back_yard_clip_skywalker_active_transport`: present (`ble`)
    - `sensor.back_yard_clip_skywalker_ble_only_fallback_mode`: present (`normal`)
    - `sensor.back_yard_clip_skywalker_last_cloud_login_success`: present
    - `sensor.back_yard_clip_skywalker_last_token_refresh`: present
    - `sensor.back_yard_clip_skywalker_last_command_failure_reason`: present
    - `sensor.back_yard_clip_skywalker_last_camera_stream_failure_code`: present
    - `button.back_yard_clip_skywalker_refresh_camera_stream`: present
    - `button.back_yard_clip_skywalker_refresh_cloud_session`: present
  - supervised real smoke evidence (guarded scope) already captured in this branch history:
    - `/tmp/mammotion_motion_suite/20260703-232917` -> `passed: true`

Pass/fail gate:

- PASS when versioned RC deploys cleanly, reload succeeds, and verification checklist is green end-to-end.
- FAIL if service/entity registration is incomplete, reload is insufficient for expected behavior, or validation regresses.

Current gate verdict: `PASS`.

### Final integration done gate

All phases above must be `PASS`.

- [x] Phase 1 PASS
- [x] Phase 2 PASS
- [x] Phase 3 PASS
- [x] Phase 4 PASS
- [x] Phase 5 PASS

Release-ready only when every phase gate is marked PASS and no unresolved blockers remain.

## Same-day completion plan (2026-07-04)

Objective: finish integration hardening today and end with a full testing-ready gate pass or an explicitly reduced, reviewable debt tail.

Current gate snapshot (fresh):

- `uv run pytest`: `180 passed`
- `uv run ruff check`: `114` errors (pre-existing debt families still dominant)
- `uv run mypy custom_components/`: now actionable (environment/parser blocker removed), `226` errors in `18` files
- `git diff --check`: pass

Same-day execution order:

1. Unblock type-check workflow (done):
  - aligned `mypy.ini` to Python `3.14`
  - removed obsolete `NewGenericSyntax` feature flag warning
2. Mypy high-yield batch A (largest concentrations):
  - `custom_components/mammotion/services.py` (`61`)
  - `custom_components/mammotion/coordinator.py` (`30`)
  - `custom_components/mammotion/agora_sdp.py` (`27`)
3. Mypy batch B (entity/setup typing cleanup):
  - `sensor.py`, `select.py`, `button.py`, `camera.py`, `entity.py`
4. Ruff debt reduction pass focused on dominant families:
  - `SLF001`, `C901`, `PERF401`, `TRY300`
5. Full gate rerun and RC-ready summary:
  - `pytest`, `ruff check`, `mypy custom_components/`, `git diff --check`

Mypy error-family prioritization for fast burn-down:

- `[arg-type]` (`45`)
- `[union-attr]` (`37`)
- `[assignment]` (`25`)
- `[unused-ignore]` (`17`)
- `[attr-defined]` (`14`)

Definition of done for today:

- full test suite remains green
- mypy no longer blocked externally and is reduced to a manageable, reviewable tail (or fully passing)
- ruff debt reduced with no new violations in touched files
- final PR debt delta summary updated with exact remaining counts

## Multi-waypoint click/go card + live findings (2026-07-08)

### Shipped this session

- Extended the click/go map card from a single target point to an ordered
  multi-waypoint path (max 3 waypoints / 3 segments). One waypoint still
  routes to `raw_pymammotion_execute_vector_segment`; 2-3 waypoints route to
  `raw_pymammotion_execute_multi_segment` with `max_real_segments` = path
  length. Per-segment polyline coloring (green/red/dashed), numbered
  markers, always-visible legend caption ("Green = mower (auto start);
  click to add destinations"). No backend motion-logic changes — built on
  the existing guarded executor.
- Documented `raw_pymammotion_execute_multi_segment` in `services.yaml`,
  `strings.json`, `translations/en.json` (was previously undocumented).
- Added a `services.yaml` vs `strings.json["services"]` key-consistency
  test and an end-to-end lateral-rejection-in-chain test.
- Fixed a display bug: the card's "charging now" preflight label matched the
  `not_charging` substring; now guards against the negated label.
- Full gate green throughout: `pytest` 182 passed, `mypy` clean, `ruff`
  steady at 28 (pre-existing debt only).

### Deployment gotcha (root-caused)

- The dashboard loads the card from the **HACS copy** at
  `www/community/mammotion/mammotion-custom-path-card.js` (served at
  `/hacsfiles/mammotion/...?v=<ver>`), NOT the integration-bundled copy at
  `custom_components/mammotion/www/` (served at `/mammotion/...`). These are
  two independent copies and had drifted (HACS copy was stuck at an ancient
  preview-only `?v=0.6.4-beta19`). Fixes must be copied to the HACS location
  AND the resource `?v=` bumped to bust the browser/service-worker cache
  (the integration serves with `cache_headers=True`). Old HACS copy backed
  up as `...beta19.bak`. Unresolved: consolidate the two distribution
  channels so repo edits reach the dashboard automatically.

### Live real-run findings (blocking a clean multi-segment completion)

Real multi-segment execution was exercised on the live mower. The feature
machinery is validated: it drove real motion, tracked progress per pulse,
and stopped safely on the first non-progressing segment (no runaway).
BUT two environmental gaps block a useful path completion:

1. **Heading offset is unstable across orientations.** `send_movement`
   forward direction vs reported heading measured wildly different offsets:
   ~116.5° (configured default / earlier sessions), ~46° (reported heading
   203° → forward map-heading 248.8°), ~100° (reported heading 174° →
   forward map-heading ~274°). A fixed `calibrated_forward_heading_offset_degrees`
   cannot be trusted; the mower drove ~70° off from the model's predicted
   direction in one run and the guard correctly halted on `no_target_progress`.
   (Caveat: measured off ~0.1 m hops on cm-noisy, laggy telemetry — the
   measurements are themselves imprecise.) Real fix = live/adaptive offset
   measurement per run, not a constant.

2. **Position telemetry is severely laggy (not frozen).** Over cloud /
   intermittent-BLE, `report_data.locations[0]` updates only after
   *minutes*, far outside the executor's second-scale `sample_delays`. The
   `raw_pymammotion_motion_probe` tool does NOT force a report refresh
   (unlike the vector/multi-segment executors, which call
   `request_reports(count=5)` per pulse) — so it always shows "no motion";
   use an executor path or `position_feedback_diagnostic` (dry_run=false,
   pulse_count=0 forces all 8 refresh steps without moving) to measure.
   Even so, the feed lag means the guard sees "no progress" within its
   window and safe-stops.

### Prerequisite for a clean completion demo

Solid BLE with the mower close to a BLE source/proxy (fast telemetry) +
reliable per-run heading calibration. Deferred until those are available;
the multi-waypoint feature itself is shipped and validated.

### Clean-completion demo achieved (2026-07-08, later same day)

Once the mower was brought within solid BLE range, the demo worked
end-to-end. Full sequence and findings:

- **BLE telemetry is fast/accurate.** With BLE connected, a forward
  calibration pulse's position update landed at the `request_reports_count_5`
  refresh step within the diagnostic window (seconds, not the minutes seen
  over cloud). `raw_pymammotion_motion_probe` still shows "no motion" because
  it does not force a report refresh; use `position_feedback_diagnostic`
  (dry_run=false, pulse_count=1) or an executor path (which call
  `request_reports(count=5)` per pulse) to measure real movement.
- **Per-orientation heading calibration works.** At reported heading ~174°,
  the measured forward map-heading was ~275-280° (empirical offset ~101-106°;
  two independent measurements at this orientation agreed within ~6°). Using
  `calibrated_forward_heading_offset_degrees: ~101` plus waypoints laid along
  the measured forward gave dry-run heading errors < 0.25° per segment.
- **Widening `heading_tolerance_degrees` from 3 to 8 was the key fix.** The
  first aligned real run completed segment 1 (`target_reached`) but failed
  segment 2 with `turn_phase_incomplete`: real forward motion scatters ~5-7°
  from the model, which left a ~4° residual that (at 3° tolerance) forced a
  micro-turn the weak angular primitive could not execute. At 8° tolerance
  the residual stays within tolerance, so no turn is attempted and the chain
  keeps driving straight. Subsequent runs did BOTH segments with **zero
  turns**.
- **Command-budget cap is the only remaining limit.** Per-pulse displacement
  was ~0.06-0.17 m (variable) at `linear_speed_fast: 400`. With
  `max_linear_commands` schema-capped at 3, a 0.25 m segment can run out of
  pulses one short (`max_linear_commands_reached`) before reaching the
  waypoint. Fix: use ~0.15 m segments so 3 pulses suffice, or raise the
  schema cap. Segment completion is otherwise reliable.

- **Physical ground-truth check (the important validation).** A run from a
  tape-marked start:
  - Telemetry straight-line displacement: **0.377 m** (toward map-heading
    275.4°, seg1 0.170 m `target_reached` + seg2 0.207 m capped).
  - Tape measure: start 10'0" -> end 8'10" = **14 in = 0.356 m**.
  - Agreement within **~0.021 m (~0.8 in, ~6%)**, explained by tape-reading
    precision + RTK cm-noise, not a scale error.
  - **Conclusion: mower-map (`mower_map_xy`) coordinates are true meters, and
    telemetry displacement is RTK-accurate to ~cm.** Distances drawn on the
    card are physically trustworthy.

Net: the guarded multi-waypoint chain is validated end-to-end over BLE —
real chained motion, live per-orientation heading calibration, safe guarded
stops, and physically-verified distance accuracy. Full autonomous/arbitrary
path execution remains out of scope (heading offset still does not transfer
across orientations, and turns remain weak/unproven) — but straight-line and
gently-aligned guarded chains now work in the real world.

## Missing explicit-stop safety bug + turning is unobservable (2026-07-09)

Two findings this session, one fixed and one that reshapes the whole turning plan.

### 1. FIXED — guarded pulses never sent an explicit stop (beta9, da0f081e)
`send_movement` (and `move_forward/left/right`) is a **continuous-velocity
command with no protocol-level duration bound** — the coordinator methods take
only a speed, never a duration; `duration_ms` was never transmitted to the mower
anywhere. Neither guarded primitive (`_raw_pymammotion_turn_to_heading`,
`_raw_pymammotion_execute_vector_segment`) ever called
`async_stop_manual_motion`, so every real pulse ran until the mower's own
firmware decided to stop — empirically wildly inconsistent (a single "0.4 m"
calibration pulse traveled **0.826 m, ~7×** expected). Fix: each real pulse now
sleeps its intended `pulse_duration_ms` and then sends an explicit stop before
sampling. Live-verified: a loop-to-tolerance run fired **10/10 pulses each with a
confirmed stop** (`ack: linear_ok+angular_ok`), bounded and predictable.
Follow-up (unfixed, not in the card path): `_raw_pymammotion_execute_segment`
and `_raw_pymammotion_angular_calibration` have the same missing-stop pattern.
Also fixed this session: the BLE-transport gate compared `str(TransportType.BLE)`
(= `'TransportType.BLE'`) to `'ble'` and so blocked *every* real run; now reuses
the coordinator's normalized `active_transport_state` (beta8, 1dc20d2a).

### 2. BLOCKER — in-place rotation is not observable in telemetry
Live turn characterization (with the explicit-stop fix in place) showed the
mower **physically pivots** on a `move_left` pulse, but `toward` / 
`location.orientation` stayed **bit-identical at 169.8581° across five pulses**
(both `send_movement angular` and `move_left`, speeds to 500, durations to
800 ms) while x/y drifted by mm. Root cause: **`toward` (= `location.orientation`)
is course-over-ground (direction of travel), which is undefined during in-place
rotation.** Also: raw `send_movement(0, angular)` produced *no visible rotation
at all* in the bounded-pulse regime at speeds up to 500 — before the stop fix it
"worked" only because firmware ran the pulse long; bounded, it is too weak. The
approved Phase-2 "accumulate weak pulses until `toward` reaches target" fix is
therefore **invalid** — the feedback signal is blind to the very motion it must
measure.

Searched for a motion-independent absolute heading and captured every candidate
live (added to `_RAW_POSITION_PATHS`, read via `position_feedback_diagnostic`):
- `location.RTK.yaw` (RTK heading, radians) = **0.0** — not populated (this
  Luba-VSPLV397 appears to have no dual-antenna true yaw).
- `report_data.vision_info.heading` (VIO heading) = **0.0**, `vio_state` = **0**
  (VIO inactive at rest — may initialize during motion; untested).
- `report_data.work.nav_heading_state.heading_state` = **3** — a status enum
  (int), not an angle; unusable as a feedback signal.
- `location.orientation` = **169** — the only live heading value, but it is the
  course-over-ground signal that cannot see in-place rotation.

**Consequence for Phase 2.** No ready-made motion-independent heading exists on
this unit. Two viable paths remain: (A) test whether VIO (`vision_info.heading`)
initializes and tracks rotation *during motion* — if so, rebuild the turn
primitive on it; or (B) **arc-based turns** — execute turns as curved motion
(linear + angular together) so course-over-ground (`orientation`, the one live
signal) updates and can serve as feedback, at the cost of turns needing room to
arc rather than pivoting in place. Decision deferred to next session.

## Phase 2 turning UNBLOCKED — VIO heading tracks rotation (2026-07-10, beta9-11)

Took Path (A) and it worked. Current local + deployed version: **`0.6.4-beta11`**
(scp-deployed to HA, md5-verified, HA restarted, all services registered).

**Breakthrough:** `report_data.vision_info.heading` (VIO body heading) is a live,
directional rotation-feedback signal on this unit — the Phase-2 blocker is lifted.
Supervised live proof (operator watched the mower physically pivot):
- Right turn `send_movement(0, +500)` 6s → `vision_heading` net **-9.0°**.
- Left turn `send_movement(0, -500)` 6s → `vision_heading` net **+13.6°**.
- It **reverses with turn direction**, `vio_state=2` throughout, ~1.5cm translation.

**Calibration (critical, encoded in the new services):**
- **Angular is weak — use `angular_speed` ~500. `180` produces NO physical rotation.**
- **Sign: +angular DECREASES `vision_heading`, -angular INCREASES it** → turn the
  opposite sign of the heading error.
- **VIO latches: `vision_heading` refreshes ~1.5s into a command then freezes.** Drive
  turns as **bounded ~1.5s pulses + explicit stop + `request_reports` refresh +
  re-measure**, not one long continuous spin.

**New services (all dry-run default, BLE-active pre-flight, mandatory explicit stop,
reuse `_manual_velocity_pulse_gates`; allowlisted in the services.yaml/strings test):**
- `mammotion.vio_motion_probe` (beta9) — forward drive + during-motion VIO sampling.
- `mammotion.vio_turn_probe` (beta10) — in-place rotation; VIO-vs-course-over-ground verdict.
- `mammotion.vio_turn_to_heading` (beta11) — **closed-loop turn-to-heading primitive**
  on `vision_heading`. Built + gated (166 tests pass, mypy/ruff clean). **NOT yet
  live-tested end-to-end** — that is the immediate next step.

Also fixed/found this session: at-rest telemetry is frozen (even forced coordinator
refresh won't unfreeze — fresh VIO needs motion + `request_reports`); an idle mower
(~1h) stops advertising BLE (`ble_rssi`→0) — **wake it** to restore BLE before testing.

**Next steps:**
1. Supervised live validation of `vio_turn_to_heading` (dry-run first, then a real
   "turn to `vision_heading` X°" with confirmations + watching). Verify it converges
   and stops within tolerance.
2. Then rebuild the multi-segment/click-to-path executor to call `vio_turn_to_heading`
   for the turn phase (replacing the course-over-ground turn primitive) + the proven
   forward linear phase. Keep multi-point execution gated until the combined
   turn+drive segment is proven live.
3. Consider committing beta9-11 (currently uncommitted working tree).

## VIO night-blocker + hardening + telemetry-exposure survey (2026-07-11)

### Live finding: VIO needs daylight; won't init from manual motion in the dark
Supervised session on branch `feat/vio-turn-to-heading` (services already committed at
`5a854a9e`). Undocked, gate GREEN (`allowed_for_manual_motion: true`, blade OFF,
MODE_READY, transport `ble` at rssi -88..-90). VIO was cold (`vio_state=0`,
`vision_heading=0.0`). Two real fires each failed to wake VIO:
- `vio_turn_probe` (0,+500, 6s): mower physically pivoted (course-over-ground `toward`
  swung ~11.7°, ~0.6cm translation) but `vio_state` stayed 0 / heading 0.0.
- `vio_motion_probe` (200, 6s): `motion_confirmed:true` but `vio_activated_any:false`,
  `max_vio_state:0`.
Root cause: `sensor.<mower>_camera_brightness = Dark` (fresh, ~01:00 UTC = night). VIO is
*visual* odometry — it can't bootstrap a feature track in the dark. The 07-10 proof only
worked because VIO was already `SIGNAL_GOOD`, warmed by that morning's daylight mowing.
**Pre-flight gate for any VIO turn test: `camera_brightness` must not be `Dark`.**
Position telemetry is also frozen at rest / not refreshed mid-drive, so probe x/y looked
static even though the mower moved.

### Code hardening shipped (deployed, awaiting HA restart to activate)
`vio_turn_to_heading` now refuses to start a real turn unless VIO is actively tracking:
- New module const `_VIO_STATE_ACTIVE = 2` (`VioState.SIGNAL_GOOD`).
- New initial safety gate `vio_active` (blocks real start when `initial_vio_state != 2`;
  still passes in dry-run so cold planning works). A cold VIO reports `heading=0.0` as a
  *valid* float, so without this the loop would turn against a meaningless 0.0 and abort
  iteration 1 on `no_heading_progress`.
- New per-iteration guard: if `vio_state` drops out of GOOD mid-turn, stop with
  `stop_reason="vio_inactive"` instead of chasing a stale heading.
- Tests added (`test_map_task_visibility.py`): cold VIO still dry-runs; real turn refused
  when cold (`vio_active` blocker); mid-turn dropout → `vio_inactive`. Gates: `199 passed`,
  ruff clean, mypy clean on `services.py`. Deployed via scp, md5
  `d70e1e058ff0044e01065b7d790eb50f` verified both sides. **Needs a full HA Core restart.**

### VioState enum (report_data.vision_info.vio_state)
`-1 UNKNOWN` (also 172 = camera pipeline initialising), `0 SIGNAL_NONE` (cold, what we
saw at night), `1 SIGNAL_INIT`, `2 SIGNAL_GOOD` (active/trustworthy — required to turn),
`3 SIGNAL_BAD`.

### Telemetry-exposure survey — VIO fields available but NOT surfaced in HA
`report_data.vision_info` (pymammotion `VisionInfo` / `vio_to_app_info_msg`, fully parsed)
carries more than we expose. Currently surfaced in `sensor.py` (Luba2/Yuka-only):
- `camera_brightness` → `vision_info.brightness` via `camera_brightness()` enum
  (numeric; `>45` = "Light", else "Dark").
- `visual_positioning_status` → `VioState(vision_info.vio_state).name`.

Recommended new DIAGNOSTIC sensors (high value for making VIO legible + explaining
failures at a glance — all Luba2/Yuka-only, same pattern):
1. **VIO heading** — `vision_info.heading` (deg). The proven body-heading signal that
   Phase-2 turning rides on; worth surfacing for visibility/automations.
2. **VIO tracked features** — `vision_info.track_feature_num`. The single best "can VIO
   lock right now" number; ~0 = featureless/dark ⇒ VIO unusable.
3. **VIO detected features** — `vision_info.detect_feature_num`.
4. **VIO brightness (raw)** — `vision_info.brightness` (int; finer than the Dark/Light
   enum, threshold >45).
5. **VIO survival distance** — `device.vio_survival_info.vio_survival_distance` (m); how
   far VIO can dead-reckon since last reliable fix.

Also present/unexposed and possibly useful later: `vision_info.x`/`.y` (VIO-local position
estimate, alt position source), `report_data.device.vslam_status` with `vision_distance`
and `vision_state` sub-bytes (report_info.py:228/239), `vision_point_info` (3-D detected
points) and `vision_statistic_info` (mean/var stats), `fpv_info.fpv_flag`.

NOTE: adding entities requires the full translations sync per CLAUDE.md (strings.json +
every locale under `translations/` + `icons.json`). Not yet implemented — documented for a
decision on whether to add the 5 sensors above (recommend at least #1 VIO heading and #2
tracked-features).
