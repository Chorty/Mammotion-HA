# Codex working plan / handoff memory

Last updated: 2026-07-11

This file is the repo-local memory index for Codex work on this branch. Use it as the source of truth when a new chat needs context. It intentionally avoids secrets, HA tokens, passwords, and live mower credentials.

## GOALS / CURRENT STATE (read this first)

**What this integration is for:** a Home Assistant integration for Mammotion mowers (test
unit: Luba-VSPLV397) that goes beyond status/telemetry toward **"click-to-path"** — let a
user draw a path on the map and have the mower drive it safely. Delivered in gated phases:
1. Read-only map/task visibility + custom-path planning/preview — **DONE**.
2. Guarded manual motion: forward/linear — **DONE, live-proven**; in-place turning — **turn
   primitive built + gated, awaiting daylight live-validation**.
3. Multi-segment executor chaining turn+drive under safety gates — **NOT STARTED**.
4. Full arbitrary drawn-path execution — **intentionally still disabled**.

**Where things stand (2026-07-11):**
- Branch `feat/vio-turn-to-heading` @ `cb349b3e` (pushed to origin). **Actual version is
  `0.6.4-beta11`** in `manifest.json` + `pyproject.toml`. IGNORE the "beta65" references
  later in this file — that is stale Codex text; a GitHub bot regresses the version, and
  deploy is by file copy + md5, not by the version string (see `reference-ha-host` memory).
- **Turning is UNBLOCKED:** `report_data.vision_info.heading` (VIO body heading) tracks
  in-place rotation. The closed-loop `mammotion.vio_turn_to_heading` primitive is built,
  gated to require an active VIO track (`vio_state==2`), and deployed — **but not yet
  live-validated end-to-end.**
- **VIO needs DAYLIGHT:** it will not initialize in the dark / from manual motion when
  `camera_brightness=Dark`; warm it with a FORWARD drive (`vio_motion_probe`), not a pivot.
- **12 diagnostic sensors added + live:** 5 VIO (heading, tracked/detected features,
  brightness, survival distance) + 7 safety (bumper, 4 ultrasonics, fuse, lock).

**Remaining to finish (full detail in the 2026-07-11 sections at the END of this file):**
1. Deferred read-only re-probes (FPV while camera streaming; RTK accuracy + base-station
   info when docked with a fix) → expose any that populate as sensors.
2. Daylight supervised live-validation of `vio_turn_to_heading` (warm VIO → real turn →
   confirm it converges and stops within tolerance).
3. Wire `vio_turn_to_heading` into the multi-segment/click-to-path executor as the turn
   phase (replacing the course-over-ground primitive); prove one combined turn+drive
   segment live; keep multi-point execution gated until proven.

**Live-test essentials:** BLE is required for real motion (gate refuses cloud); every
real-motion command needs a fresh explicit user "go" while they watch; deploy via scp + md5;
a changed service or new entity needs a full HA Core restart the user triggers. Gate every
code change with `.venv/bin/pytest` + `mypy` + `ruff` (`uv` is not on PATH). Full procedure
in the `mower-live-testing-workflow` and `reference-ha-host` memories.

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
every locale under `translations/` + `icons.json`).

### UPDATE 2026-07-11 — 5 VIO sensors implemented + wider field probe + 7 safety sensors
The 5 VIO sensors above were implemented (see commit `e035f3fa`). Then a read-only
shortlist probe was added to `position_feedback_diagnostic` (`_RAW_POSITION_PATHS`, kept
for future re-probes) and run live on this Luba (undocked, paused, night, camera idle, not
on RTK base). Probe results:
- **Populated + meaningful:** `dev.self_check_status`=10, `dev.fuse_status`=1,
  `dev.sensor_status` group (bumper + `ult_left/left_front/right_front/right` all `OK`=0 at
  rest), `dev.lock_state.lock_state`=0, `dev.fpv_info.fpv_flag`=0 (0 because camera idle),
  `connect.wifi_is_available`=1.
- **State-dependent zero (re-probe before building):** `rtk.lat_std/lon_std/top4_total_mean`
  =0 (undocked/night), `basestation_info.*`=0 (not connected to base), `rtk.dis_status`=a
  packed int needing decode.
- **Absent on this hardware (skip):** `dev.mnet_info.*` / `connect.mnet_inet` (no 4G),
  `dev.collector_status`=0 (Luba has no collector).

Implemented from the populated set (this commit): obstacle/safety group as ENUM sensors
(`bumper_status`, `ultrasonic_left/left_front/right_front/right_status`; OK/Warning/Error
from `SensorCheckState`) + numeric `fuse_status` + numeric `lock_state`. Full translations
(names + enum states) across all 12 locales + icons. All 7 registered live (all `OK` / fuse
1 / lock 0). Entity_ids are name-slugged (e.g. `..._left_ultrasonic_status`); unique_ids
use the entity keys.

Deferred re-probes (probe paths remain in place, no redeploy needed): FPV status with the
camera **streaming** (expect fpv_flag→1); RTK accuracy (`lat_std/lon_std`) + base-station
info when **docked with a solid fix**; `self_check_status` bit-layout decode.

## VIO turn wired into the click-to-path executor (2026-07-11, later session — CODE DONE, NOT DEPLOYED)

The turn phase of `raw_pymammotion_execute_vector_segment` / `_multi_segment` (the two
services the custom-path card `www/mammotion-custom-path-card.js` drives) was rebuilt on
VIO. **Uncommitted working tree; gates green (205 passed, mypy/ruff clean); NOT yet
deployed to HA.**

Design (all in `services.py`):
- New `turn_mode` param, default **"vio"** (old model available via `"legacy"`). Legacy
  course-over-ground turning is disproven live: `angular_speed=180` does not rotate, and
  the fixed `calibrated_forward_heading_offset_degrees=116.5` was coincidental.
- **`_vio_segment_calibration_drive()`** — VIO frame is re-anchored per initialisation, so
  the map→VIO offset is derived live: short forward pulses (speed-200 profile, explicit
  stop + `request_reports` refresh), then `offset = atan2(dy,dx) map-motion heading −
  fresh vision_heading`. Doubles as the VIO warm-up/refresh. Needs ≥0.02 m displacement +
  `vio_state==2` to pass; else `vio_calibration_failed`.
- Vector segment vio flow: `vio_active` gate (refuses real motion when `vio_state != 2`;
  dry-run still plans) → calibration drive (skipped when `vio_heading_offset_degrees`
  provided) → **re-anchor position/target heading on post-calibration telemetry** (the
  drive moved the mower; may even complete tiny segments) → `target_vision_heading =
  target_map_heading − offset` → the proven `_vio_turn_to_heading` primitive (angular 500,
  bounded pulses) → unchanged proven linear phase. Fresh post-turn snapshot baselines the
  linear phase (the VIO primitive reports headings, not telemetry).
- Multi-segment carries segment 1's derived offset to later segments (no recalibration);
  new params threaded; result exposes `turn_mode` + a `vio` block (offset, source, target,
  calibration detail).
- Schemas/handlers: new optional `turn_mode`, `vio_heading_offset_degrees`,
  `vio_turn_max_commands` (8), `vio_angular_speed` (500), `vio_calibration_pulse_count`
  (2). The card's existing payloads work unchanged (defaults apply; its legacy knobs are
  ignored in vio mode). `_raw_vector_readiness_test` pins `turn_mode="legacy"` (it
  validates the legacy pipeline).
- Tests: legacy tests pinned with `turn_mode="legacy"`; 6 new vio tests (dry-run plan,
  cold-VIO refusal, calibrate→turn→drive flow incl. offset math, failed-calibration halt,
  calibration-drive offset unit test, multi-segment offset carry).

**Daylight status this session:** `camera_brightness=Light`, `vio_state=2`,
`vision_heading=90.23°` latched — VIO warm. Blockers for live work: `ble_rssi=0` (mower
stopped advertising; needs user wake) and `active_transport=cloud`. FPV re-probe
inconclusive (fpv_flag stayed 0; camera entity idle — needs a real WebRTC viewer, curl
can't consume it; RTK/basestation re-probe still awaits docked+fix).

**Next:** deploy `services.py` (scp+md5) → user "restart HA" → dry-run vector segment in
vio mode (card payload shape) → supervised live combined turn+drive segment (daylight,
BLE, per-fire "go"): expect calibration drive ~2 pulses → VIO turn to target → forward to
waypoint, `stop_reason=target_reached`. Then commit + offer PR.

## LIVE: VIO turn phase PROVEN in executor; v2 steering fixes (2026-07-11 continued)

Supervised daylight session. Committed `9b0c5fb2` (wiring) + `94777ff4` (v2 fixes), both
deployed (final md5 `3ece4fb51b07476261d66ce0f2c57274`), **HA restart pending** for v2.

**Live results (3 real runs, per-fire user go):**
- Dry-run in vio mode: sane (`turn_mode: vio`, planned calibration + vio turn @500).
- Run 1 (169° required turn): calibration 1 pulse/2.0cm → offset 191.1°; **8 VIO turn
  pulses, error 159.7°→80.4°, perfectly monotonic ~11°/pulse**; hit 8-pulse cap →
  correctly refused linear (`turn_phase_incomplete`).
- Run 2 ("turn 90° right, drive 1 m"): facing probe (`vio_motion_probe` 4s) → facing
  341.3°, offset 161.84; **turn phase `target_heading_reached` (8 pulses, → −6.5°),
  operator-confirmed clean 90° right turn**; linear 12×1500ms pulses moved only ~0.34 m
  net (~3 cm/pulse — firmware ramp eats short pulses).
- Run 3 (continuation, 2000ms pulses): turn again perfect (7 pulses → −1.0°); linear
  moved **~1.0 m real** but **~25° off-bearing**, guard stopped it (`no_target_progress`).

**Root cause of off-bearing drive:** offset from a 2 cm calibration baseline carries
~25° noise; forward pulses can't steer. **v2 fixes (committed `94777ff4`):** calibration
baseline min 2→6 cm; continuous offset refresh from each passing linear pulse
(`offset_source: linear_refresh`); mid-drive re-aim via bounded VIO turn when facing
drifts >15° off bearing (≤3 realignments, `realignments` reported in result). New params:
`vio_realign_threshold_degrees`, `vio_max_realignments`.

**Operational discoveries:** mower restart (iOS app) → fresh BLE advertising → prompt
promotion (vs. 10+ min of failed toggles at rssi −90); VIO **self-initializes in
daylight** after mower restart (no warm-up drive needed). Linear reality: ~3 cm/pulse at
1500 ms, ~8 cm/pulse at 2000 ms (speed 200/400 mix).

**Next:** user "restart HA" → one clean run (fresh 6cm calibration → turn → steered
drive) expecting `target_reached` → then multi-segment click-to-path proof + PR.

## ✅ MILESTONE: combined turn+drive segment PROVEN LIVE — target_reached (2026-07-11 evening)

**The v2 executor completed a full closed-loop segment on VIO**: from (3.693, −0.997) to
target (4.41, 0.06): turn phase 4 pulses (error −41.7°→+2.7°, `target_heading_reached`),
linear phase 12×2s pulses all on-bearing (~10cm each, ~1.2m), continuous offset refresh
active (`offset_source: linear_refresh`, 108.54°→106.56°), zero re-aims needed, landed
**~7cm from target** (tolerance 15cm), `stop_reason: target_reached`, `complete: true`.
Artifact: scratchpad `live_vio_segment_9.json`. Commits `9b0c5fb2`+`94777ff4`+`76de94e4`+
`03d8d09e` all pushed.

**Working parameter set (proven):** `heading_tolerance_degrees: 8`,
`vio_turn_max_commands: 16` (MUST pass — default 8 only covers ~100° of turn),
`linear_pulse_duration_ms: 2000`, `max_linear_pulse_ceiling: 20`,
`max_no_progress_pulses: 4`, `waypoint_tolerance: 0.15`, `sample_delays: [0,3,6]`;
`vio_heading_offset_degrees` reusable across runs in-session (pose-independent
frame-to-frame offset; each run's result reports the refreshed value).

**Hard-won operational playbook (all live-verified today):**
1. **iOS app holds the mower's single BLE slot** — force-close it before testing;
   promotion then lands in ~20-120s (vs never while app open).
2. **Every mower restart kills the device-side report stream** → position telemetry
   freezes/lags minutes. Fix: `position_feedback_diagnostic` real-mode `pulse_count: 0`
   (zero motion, runs the full re-init arsenal: report snapshot/stream, iot sync, BLE
   sync) → position feed goes live again (visible ±3cm jitter = healthy).
3. **The re-init's iot-sync window flaps BLE to cloud** → one toggle after re-init
   re-promotes (~20s when mower freshly booted).
4. Speed-200 pulses barely move this unit (~1-2cm real per 2s; firmware ramp); speed-400
   ≈ 8-10cm per 2s pulse. Calibration now runs at fast speed (`03d8d09e`).
5. Idle mower stops advertising within ~30min → wake via mower restart or button.
6. User has full-area UniFi camera coverage of the mower's reachable area (supervision).

**Remaining to finish:** (1) multi-segment click-to-path proof (2-3 points through
`raw_pymammotion_execute_multi_segment`, offset carries across segments — code ready,
needs one supervised run); (2) then consider ungating multi-point + PR to main; (3)
deferred re-probes (FPV while streaming; RTK accuracy when docked).

## 🏆 PHASE 2 COMPLETE: multi-segment click-to-path PROVEN LIVE (2026-07-11 evening)

Supervised L-path run (`raw_pymammotion_execute_multi_segment`, 3 points, 2 real
segments, artifact `live_multi_segment_1.json`): **both segments `target_reached`,
`ready_for_multi_segment: true`**, landed ~11cm from the final point.
- Seg 1 (0.75m −y): turn 12 pulses −126.5°→−1.0°; drive 8 pulses; **mid-drive re-aim
  triggered live and worked** (21.9° drift after pulse 1 → corrected → clean drive).
- Seg 2 (0.70m −x): **carried offset, zero recalibration**; turn 5 pulses −50.7°→−1.1°;
  drive 8 pulses all on-bearing.
- Offset self-refined 106.56→108.28→109.63 (`linear_refresh`) across the whole path.

Every executor feature validated in one run: VIO turn phase, fast-speed live
calibration, offset carry across segments, continuous offset refresh, mid-drive re-aim,
per-segment safety re-checks, waypoint completion. The card→multi-segment→turn+drive
pipeline is functional end-to-end.

**Next session:** decide whether to lift point-count/segment caps (currently 4 points /
max_real_segments) + surface `turn_mode`/`vio_turn_max_commands: 16` as card defaults;
open the PR to main; deferred FPV/RTK re-probes; consider exposing exact per-command UTC
timestamps in command_results (user request for camera-footage correlation).

## Wrap-up 2026-07-11 late evening: PR open, card tested to the VIO gate, dusk ended play

All four follow-ups landed (commits pushed through `2c1dd028`, **PR #10 open**:
https://github.com/Chorty/Mammotion-HA/pull/10):
- Multi-segment caps lifted to 2-8 points / 7 real segments (5-point dry-run verified live
  post-restart); card `MAX_WAYPOINTS` raised to 7.
- Card sends proven VIO defaults (`turn_mode: vio`, `vio_turn_max_commands: 16`, 2s linear
  pulses, 0.15 tolerance, `[0,3,6]` delays, ≤7 segments).
- `sent_at_utc` on every calibration/turn/linear command result.
- Probes: **`fpv_flag` confirmed 1 while streaming → `fpv_status` sensor shipped**
  (inactive/streaming/error, all locales); RTK `lat/lon_std` + `basestation_info.*`
  stayed 0 even docked with a fix → **dead on this firmware, intentionally not exposed**.

**First card-driven run** (user clicking the map): front-end works end-to-end — service
fired, full JSON returned. It correctly stopped at `vio_active` (VIO cold: mower
stationary since undock + dusk; `camera_brightness=Dark` ended the day). TWO gotchas for
the next card session:
1. **Browser cache fingerprint**: if a card response shows `sample_delays: [0,5,10]` /
   `waypoint_tolerance: 0.08`, the browser is running the OLD card — cache-bust via
   Settings→Dashboards→Resources, append `?v=N` to the card URL. New card sends
   `[0,3,6]` / `0.15`.
2. Old-card params make long segments fail (300ms default pulses ≈ 2-3cm → ceiling).

**Daylight card-session recipe:** cache-bust card → undock (release_from_dock button,
avoids app) → app closed → BLE toggle if needed → warm VIO (`vio_motion_probe` 4s forward,
needs go) → card dry-run (expect `[0,3,6]` fingerprint) → Real Go with 2 waypoints ~1-2m
first, then scale to multi-waypoint paths.

## Wrap-up 2026-07-13/14: code review of the stale-heading + BLE-recovery work, 5 hardening fixes

The 07-12 session's work (stale-VIO poll fix + BLE auto-recovery) is committed as
`71d2c8d8`. A high-effort /code-review over it (8 finder angles + verification) produced
10 findings; the 5 blocking ones are FIXED in the working tree (+72/−35 in services.py,
**uncommitted, not deployed**; 187 tests pass, ruff clean):
1. Both executors now run BLE recovery BEFORE the `initial_telemetry` snapshot — safety
   gates/target math previously judged state up to ~93s stale after the recovery wait.
2. The recovery cooldown guard was DEAD: `availability.ble_in_cooldown` doesn't exist in
   pymammotion (test fixtures fabricate it!). New `_ble_connect_cooldown_active()` reads
   `get_transport(TransportType.BLE)._connect_cooldown_until`; the mid-budget toggle now
   defers while cooldown is active.
3. No-progress-streak pulses in `_vio_turn_to_heading` use `slow_pulse_duration_ms`
   (blind rotation was unbounded — `max_displacement_m` only caps translation).
4. Fresh-heading poll sleep floored at `max(refresh_wait_seconds, 0.5)` —
   `refresh_wait_seconds: 0` used to hammer request_reports on the BLE queue.
5. `ble_auto_recover` wiring: dead schema key removed from execute_segment; multi-segment
   forwards the flag per segment (explicit false was being overridden); readiness probe
   pinned to `ble_auto_recover=False` (keeps the diagnostic fast-fail).

Deferred findings (known, unfixed): calibration drive + `linear_refresh` still consume
single possibly-stale VIO samples (silently wrong offset); standalone `vio_turn_to_heading`
gets no auto-recovery; `ble_auto_recover` missing from strings.json; recovery runs before
the cheap confirmation gates; sleeping mower burns the full 90s budget.

Also set up **subagent model routing** (`.claude/agents/finder.md`=sonnet,
`verifier.md`=opus, rule in CLAUDE.md) so review fan-out runs on cheaper models.

**Re-review of the fix diff is IN PROGRESS** (session ended near limit): 5/8 angles done.
Top candidate (2 independent finders): pymammotion's connect cooldown default (120s)
outlasts recovery's 90s budget → a cooldown-blocked recovery never toggles and the final
`reason` misdirects to check_phone_app/needs_wake instead of naming the cooldown; also
`_ble_connect_cooldown_active` has zero direct test coverage (fixture handle lacks
`get_transport`). **Next session:** finish the 3 remaining angles (removed-behavior,
efficiency, altitude), opus-verify, apply the cooldown-reason fix + a real cooldown test,
then commit the fixes and deploy via scp for a supervised daylight run.

## Wrap-up 2026-07-14/15 evening: fixes deployed + first live runs on review-hardened code

Deployed the 9 review fixes to HA (scp, md5 2303a2691d…, confirmed live via the new
`effective_poll_interval_seconds` field in a dry run). Map-blank-after-restart fixed by
the documented recipe (mower awake → config-entry reload). Two supervised card runs:

- **Run 1 (3.58m):** calibration 1 pulse; VIO turn PERFECT (one 700ms pulse, error
  9.0°→2.6°, fresh-heading poll working); 18 linear pulses dead on bearing (~1.02m);
  ended when BLE hit its 120s cooldown — stop hung 32.7s → `stop_failed_aborting`
  (hardening worked). Throughput facts: ~8cm per 2s speed-400 pulse; 14s/pulse cadence
  (10s of it card `sample_delays [0,5,10]`) → card config now `[0,3]`.
- **Run 2 (176° U-turn):** calibration ✓; turn pulses 1-3 textbook (~12.5°/pulse);
  then SUNSET killed VIO mid-turn (features 80→0, `vio_state` stayed 2!) — heading
  latched bit-identical, new streak logic capped blind pulse to 700ms and aborted
  `no_heading_progress` at streak 2. Blind rotation bounded (~29° toward-swing).
  Live-confirmed review finding: a 0.0018° noise wiggle passed the float-inequality
  freshness check.

**Next session (Opus prompt prepared):** commit the fixes; freshness epsilon (~0.1°,
wrap-aware); VIO liveness gate on brightness/tracked-features (vio_state alone lies at
dusk); `linear_pulse_duration_ms` schema cap 2000→4000 (+services.yaml selectors);
slow-streak cap only when sample stale; review backlog (cooldown test coverage, dead
`ble_in_cooldown` diagnostic+fixtures, ha_state refetch-after-recovery). Joystick card
idea DEFERRED (interim: grid card of emergency_nudge buttons). Mower parked ~(6.09,
−2.80) facing ~−94°; card config carries `sample_delays [0,3]`.

## Wrap-up 2026-07-15: dusk-latch hardening (epsilon + VIO liveness gate) + review backlog

All six queued items landed (198 tests, was 187; ruff clean on touched files;
NOT deployed — deploy checklist below):

1. **Committed** the 9 review fixes as `1c196843` (were live on HA but uncommitted).
2. **Freshness epsilon:** the vio_turn poll now requires
   `abs(_heading_error_degrees(before, after)) > _VIO_HEADING_FRESH_EPSILON_DEGREES`
   (0.1°) instead of float inequality — run 2's 0.0018° noise wiggle no longer counts
   as fresh. Both stale-poll tests now jitter sub-epsilon; new regression test
   `sub_epsilon_wiggle_is_not_fresh`.
3. **VIO liveness gate:** new `_vio_feed_liveness()` reads
   `vision_info.track_feature_num` (< `_VIO_MIN_TRACKED_FEATURES`=5 → degraded; missing
   field → live, so non-reporting devices aren't blocked) + brightness label. Wired:
   vio_turn entry gate `vio_feed_live` + per-pulse `stop_reason: vio_feed_degraded`;
   vector-executor gate (only when vio_state==2 — cold-start warm-up path untouched);
   calibration drive refuses the offset (`vio_feed_degraded`) when the post-drive feed
   is blind. Feed dicts surfaced in results (`initial_vio_feed`, `final_vio_feed`,
   `vio_feed`).
4. **Pulse cap:** `linear_pulse_duration_ms` max 2000→4000 in both schemas +
   both services.yaml selectors (user wants 4s-pulse throughput tests).
5. **Streak refinement:** slow-pulse cap during a no-progress streak now applies only
   when the last sample was stale (`last_heading_went_fresh` False); fresh-but-stalled
   streaks keep the full pulse. Docstring + tests updated.
6. **Review backlog:** cooldown helper direct tests (deadline read, API-drift
   fallback, pinned-pymammotion attr contract); dead `availability.ble_in_cooldown`
   diagnostic replaced with live `ble_connect_cooldown_active` (fixtures de-fabricated,
   `get_transport` added); `refetch_runtime_context` callback threaded handler→both
   executors (and multi→per-segment) so post-recovery gates judge fresh
   ha_state/active_route — test proves "started mowing during the 90s recovery" blocks.

**Deploy checklist (user deploys via scp):** `custom_components/mammotion/services.py`
+ `custom_components/mammotion/services.yaml` → HA, restart, verify via a dry run that
the result carries `initial_vio_feed`. Next live objectives: daylight multi-segment run
with 4000ms pulses; confirm `vio_feed_degraded` fires at dusk instead of
`no_heading_progress`.

## Wrap-up 2026-07-15 (later): deploy + first daylight live run + position-settle fix

Deployed the dusk-latch hardening (services.py md5 15c06373…, services.yaml bc23b1f8…)
via scp, restarted HA Core, verified live: dry runs carry `initial_vio_feed` and the
4000ms pulse is accepted. BLE promoted instantly on a single switch-reassert-ON
(rssi −76→−68). Mower entity is `lawn_mower.back_yard_clip_skywalker`.

**First daylight real motion (2 × `vio_motion_probe` 4s @ speed 400):** VIO stayed
healthy (state 2, 80 features, Light) so the new liveness gate correctly stayed green
with zero false aborts. BUT telemetry claimed ~11cm + ~9cm while the mower physically
moved **< 6 inches total** (user-observed). Root cause: the map-local x/y feed lags ~4s
and updates in JUMPS, so pulse 2's "displacement" was pulse 1's delayed registration —
`motion_confirmed`/`displacement_m` over-attributed across back-to-back pulses. Finding:
**4s pulses are not a throughput win (~2–3" each); the 4000 cap won't speed up path
runs.** The dusk-degradation code paths remain unverified live (can't trigger in
daylight).

**Fix authored (committed, NOT deployed — needs daylight tape-measure validation):** the
linear phase now runs a bounded position-SETTLE poll after each pulse's stop
(`_settle_linear_position_feed`), mirroring the turn phase's fresh-heading poll. Settling
requires the feed to both move off the pre-pulse position AND stop jumping, so per-pulse
displacement is attributed to the right pulse; a blocked pulse times out `settled=False`
→ existing no-progress logic handles it. Module constants
`_LINEAR_POSITION_SETTLE_EPSILON_M` (0.01) / `_LINEAR_POSITION_SETTLE_TIMEOUT_SECONDS`
(6.0); poll bounded by count so no-op-sleep tests don't spin. Applied to the vector
executor (the click-to-go path) only; `_raw_pymammotion_execute_segment` + the other
legacy linear loop share the bug and are NOT yet patched. 200 tests (was 187). NEXT:
deploy in daylight, tape-measure a single pulse to confirm `position_settled` improves
attribution, then decide whether to extend the settle poll to the other executors and
whether to shorten pulses back to ~2s given the throughput finding.

## Wrap-up 2026-07-16: 4 motion-reliability issues + self-review

Implemented the approved 4-issue plan off-mower (branch feat/vio-turn-to-heading,
commits 0684bf54 + the review-fix commit below). NOT deployed -- needs daylight tape
validation.

- #1 (extend settle poll to legacy linear executor): the review found the premise was
  wrong -- `_raw_pymammotion_execute_segment` issues NO software stop (relies on firmware
  auto-stop), so a settle wait there only prolongs blind motion. REVERTED the settle poll
  from that executor; kept the dual-source instrumentation. The vector executor (the card
  path, which does stop) keeps its settle poll from a35afdd3.
- #2 (vio_motion_probe honesty): judge motion_confirmed + final_displacement_m across
  post_stop samples (where the lagged real move lands); add displacement_source. Done.
- #3 (phantom instrumentation): `_position_source_comparison` (read-only) logs both
  position sources + RTK quality per pulse in both executors and both VIO probes. Detector
  deferred until a daylight run yields phantom-vs-real data.
- #4 self-review (high, 5 findings): FIXED #1-revert (finding 2), settle-poll docstring now
  states the phantom limitation (finding 1), extracted the duplicated
  `_make_refetch_runtime_context` factory (finding 5).

**Deferred review findings (noted, not fixed -- decide during daylight validation):**
- The position-settle poll can settle on a PHANTOM feed-jump (epsilon 1cm << the ~9cm
  phantom seen live), so over-attribution persists on no-op pulses -- only the #3 detector
  can fix this; the settle poll only handles LAG.
- `_VIO_MIN_TRACKED_FEATURES=5` is re-checked EVERY pulse in `_vio_turn_to_heading`, so a
  single transient feature dip (occlusion) aborts an otherwise-healthy turn with
  vio_feed_degraded -- same single-sample-abort class we fixed for heading; consider a
  consecutive-degraded streak before aborting.
- The settle poll's wait is ADDITIVE to the card's sample_delays window (~+4s/pulse), a
  throughput cost; once validated, trim sample_delays (the settle poll already waits out
  the feed) or have the executor use the settle telemetry as `after` instead of re-sampling.

Tests 200 -> 203; ruff clean. Deploy checklist unchanged (services.py only; daylight scp +
restart + tape validation, collect position_source_comparison data).

## Wrap-up 2026-07-16: resolve xhigh-review findings (2 hardening + cleanups)

Worked through the resolvable xhigh-review findings off-mower. 2 hardening + cleanups
(commit below). NOT deployed.

Hardening (correctness):
- H1 (transient feed-dip tolerance): the per-pulse vio_feed_live check in
  _vio_turn_to_heading now re-confirms a degraded read with a bounded read-only poll
  (_reconfirm_vio_feed_degraded, _VIO_FEED_RECONFIRM_POLLS=2) before aborting, so a
  single transient feature dip no longer kills a good turn while sustained dusk
  blindness still aborts. No motion during the wait (mower already stopped).
- H2 (wrong-direction slow-cap): the no-progress slow-pulse cap now also fires when the
  last pulse made NEGATIVE progress (moved away from target, e.g. sign miscalibration),
  not only when the sample was stale -- bounds wrong-way full-power rotation. Safe
  (only shortens pulses). Split the streak test into wrong-direction (slow-capped) vs
  creeping-toward-target (full pulse kept).

Cleanups:
- C1 _vio_feed_live_gate() helper shared by both executors (unifies the drifted detail
  strings). C2 _vio_scene_brightness() shared by _vio_scene_is_bright + _vio_feed_liveness.
  C7 pulse-1 reuses initial_feed (entry gate already proved it live). C8 added the
  vio_feed_live gate to the multi-segment ENTRY (was only per-segment). C3 documented the
  settle-poll `telemetry` return field as the reserved hook for the deferred throughput fix.

Deferred (with rationale, NOT done): the settle-poll phantom limitation (needs the #3
detector + daylight data); max_displacement_m-can't-fire-mid-drive and the sub-epsilon
slow-turn timing (change proven motion, want live confirmation); the additive-settle-wait
throughput fix (changes the progress-`after` source; validate first). Reuse findings C4
(agreement_m hypot) / C5 (settle refresh helper) / C6 (source-comparison on raw_sources)
skipped: the reuse would add coupling/clunkiness for marginal benefit (verified
_telemetry_position_delta wants dict-shaped inputs, not the 2-tuples).

Tests 204 -> 207; ruff clean. Deploy checklist unchanged (services.py only).

## Wrap-up 2026-07-18: off-mower session — schema parity sweep, hassfest fix, PR prep, segment stop, telemetry research

Off-mower (mower unavailable, night). Commits `0ebdf7c1`..`8271f02f`, all pushed to
PR #10. Tests 207 -> 260; mypy + ruff clean throughout. NOTHING deployed.

1. **Schema/handler key-parity audit (the KeyError->500 class).** AST audit of all 50
   service registrations found ONE more real gap beyond the multi-segment
   `ble_auto_recover` bug: `manual_velocity_segment_test`'s handler reads
   `stop_mode`/`stop_delay_ms` but the schema declared neither -> every call omitting
   them 500'd. Fixed (defaults immediate/0 matching the pulse-test sibling + executor
   signature; yaml selectors added). svg_add/update x_move/y_move flagged but safely
   membership-guarded. NEW generic regression sweep in the test file: parses
   services.py's AST, extracts each handler's unguarded `call.data[...]` reads
   (following one level of call-passing helpers, e.g. handle_movement), and asserts
   every key resolves after applying the real schema to a minimal required-keys
   payload. 47 parametrized cases + a discovery meta-test. Kills the bug class.

2. **Hygiene.** `mypy custom_components/` (repo config): clean. pre-commit: applied the
   one scoped codespell fix (unparseable->unparsable in services.py); did NOT apply the
   ruff-format/prettier churn on pre-existing drift (17 files reverted per-file);
   pre-commit's mypy --strict hook fails on ~170 pre-existing environmental errors
   (isolated env without HA) across entity platforms — out of scope, repo mypy is the
   real gate.

3. **PR #10 prepped.** Branch 31 commits ahead of main, 0 behind — no rebase needed;
   version beta11 > main's beta8 kept per the standing rule. CI triage: `python` check
   = requirements_test.txt pip conflict (pre-existing), `hacs` = repo settings
   (license/issues/topics, not code), `hassfest` = uppercase translation state keys +
   http/web_rtc deps (pre-existing on main) PLUS uppercase `OK/WARNING/ERROR` states
   the branch itself added — FIXED: the 5 safety sensors now emit
   `SensorCheckState(...).name.lower()` and the state keys are lowercased across
   strings.json + all 12 locales (values kept translated; no per-state icons). NOTE:
   sensor.py + translations now differ from what's deployed on HA — include them in
   the next deploy (previously services.py-only). PR description rewritten
   (hardening-since-open section + validation state). MERGEABLE; left unmerged for
   human review.

4. **C4/C5/C6 reuse findings re-judged — all three skips stand.** C4: the comparison
   is two same-instant xy tuples; `_telemetry_position_delta` wants start/end telemetry
   dicts + availability gating — reuse means fabricating fake telemetry shapes.
   C5: `_position_feedback_refresh_attempt` is a diagnostic multiplexer with its own
   pacing sleep (double-sleep risk) and attempt-report shape; the settle loop's bare
   `async_get_reports(count=5)` is the file-wide idiom. C6: `_position_feedback_raw_sources`
   is a heavyweight raw dump (3 locations + transport/handle probing) — per-pulse reuse
   costs more and still needs all the scaling/filtering; the real shared layer is
   `_safe_attr_path`/`_scale_report_position`/`_latest_location`.

5. **`_raw_pymammotion_execute_segment` got its software stop (+ settle poll).** Now
   mirrors the vector executor per pulse: bounded `linear_pulse_duration_ms` (new
   param/schema/yaml, default 300ms, range 50-4000) -> `_manual_velocity_stop_attempt`
   -> abort `stop_failed_aborting` on undeliverable stop -> `_settle_linear_position_feed`
   (settled/moved/wait per command). 2 new tests + no-progress test asserts the settle
   fields. **DO NOT DEPLOY until a supervised daylight run validates it** — suggest
   passing `linear_pulse_duration_ms: 3000-4000` on that run (taped model: sub-2s
   pulses risk physical no-ops; default 300 mirrors the sibling schema, not the
   proven pulse length).

6. **Dormant obstacle/report telemetry during autonomous mowing — ROOT CAUSE FOUND
   (research only, no code).** The report coordinator's `_async_update_data` NEVER
   solicits reports — it re-reads pymammotion's in-memory state (REPORT_INTERVAL=5min
   tick is a cache read; that's why `homeassistant.update_entity` does nothing).
   Freshness while mowing comes entirely from pymammotion's own cadence loops:
   - **BLE connected:** `ble_polling_loop` maintains a continuous count=0 report
     stream in ACTIVE mode (renew 8s, stale watchdog 15s, ~1 report/s). The
     "subscribe to DEV_STA push" idea ALREADY EXISTS on the BLE path.
   - **Cloud only:** `mqtt_activity_loop` polls count=1 every 10-15 MINUTES in
     ACTIVE mode (quota protection; env-overridable `MAMMOTION_POLL_ACTIVE_SECS`).
   - `NO_REQUEST_MODES` does NOT include MODE_WORKING — not deliberately blocked.
   Tonight's dormancy signature (only advertisement-derived ble_rssi live) means the
   BLE connection was NOT held during the mow (app slot theft and/or range) and the
   user runs BLE-only (cloud switch off) -> no fallback stream at all. HA already
   auto-retries BLE on every advertisement (debounced 60s, `_add_ble_device` ->
   `ble.connect()`), so mid-mow re-latch SHOULD happen with the app closed.
   **Recommended next steps:** (a) live diagnosis next daylight mow, app force-closed:
   watch `active_transport`/`ble_stream_active` — if BLE re-latches, sensors go live
   at ~1s cadence with zero code changes; (b) if cloud-side coverage is wanted, the
   cheap HA change is in `MammotionReportUpdateCoordinator._async_update_data`: when
   device_mode ACTIVE and not `handle.ble_stream_active`, call
   `async_start_report_stream(duration_ms=330_000)` per 5-min tick — RPT_START goes
   via best transport (verified `_send_report_stream_start` uses `send_raw`), repeat
   calls send RPT_KEEP, pushes damped by no_change_period=4000; bounded cloud cost,
   config-gate it. Deferred: needs the user's call on cloud quota + a live mow to
   validate; useless for the current BLE-only setup until cloud is re-enabled.

**Deploy checklist changed:** next scp must include `services.py`, `services.yaml`
(segment-test + execute-segment selectors), `sensor.py`, `strings.json`, and ALL
`translations/*.json` (lowercase ENUM states) — then restart + re-run a multi-segment
dry-run and check the 5 safety sensors show translated states.

## Wrap-up 2026-07-18 (on-mower, daylight): expanded deploy verified, software stop proven, turn-tolerance bug found+fixed, card corrected

Live supervised session on `feat/vio-turn-to-heading` @ `330117ce`. Working tree was
clean, so everything deployed was the committed branch state. Blade OFF for the entire
session (see the scope note in item 8).

### 1. Expanded deploy + post-restart verification (all green)

scp'd the full 16-file set (`services.py` `e72343fa`, `services.yaml` `ad427ffc`,
`sensor.py` `b900684a`, `strings.json` `62ee44bd`, all 12 `translations/*.json`),
md5-matched both sides, restarted HA Core (~105s). All four checks passed:

- safety sensors now emit lowercase `ok` (were `OK` pre-restart) — the hassfest fix live
- multi-segment dry-run HTTP 200 with `initial_vio_feed {live, 80 features, Light}`
- `manual_velocity_segment_test` omitting `stop_mode`/`stop_delay_ms` returns **200**
  (the KeyError->500 fix confirmed on hardware)
- `execute_segment` accepts `linear_pulse_duration_ms: 3500` and echoes it back

BLE promoted to `ble` on its own through the restart — no toggle needed.

### 2. `_raw_pymammotion_execute_segment` software stop: VALIDATED (tape)

Two real single pulses (`max_commands: 1`, 3500ms, speed 400). Both stops delivered and
**dual-axis ACKed (`linear_ok`/`angular_ok`) in 0.83ms / 1.06ms** — versus the 07-14 run
where a stop hung **32.7s** then threw `BLEUnavailableError`. No `stop_failed_aborting`.
`position_settled: true` + `position_moved: true` on both.

**Settle poll fixes cross-pulse bleed (the 07-15 bug):** pulse 1 ended (4.3864, 2.0409),
pulse 2 started (4.3874, 2.0400) — pulse 1's travel had fully registered before pulse 2
began. **Consecutive-pulse no-op did NOT reproduce** (2/2 executed back-to-back).

### 3. Position feed has a ~2-6cm ABSOLUTE noise floor (not a % error)

Both pulses taped **exactly 4in (10.16cm)**; telemetry said 11.97cm then 14.52cm. First
read as "+18%/+43% inflation" — **corrected by the operator's 10-foot hand-push**: feed
said 3.107m vs 3.048m actual (**+5.9cm**). Absolute error is ~2-6cm at BOTH 10cm and 3m
scales => constant noise floor, not scaling.

Consequences: (a) per-pulse displacement is hopeless (noise ≈ signal at 10cm/pulse);
(b) segment-scale distance-to-target is reliable (~2% at 3m); (c) **`min_progress_distance`
defaults of 0.01/0.003m sit an order of magnitude INSIDE the floor** — the no-progress
detector is largely reading noise.

### 4. TURN-TOLERANCE BUG: deadband below the physical rotation quantum

First L-path attempt died `turn_phase_incomplete` on segment 1 with only a **−4.8°**
error: pulse 1 (+500, 700ms) swung **8.2°** through zero to +3.42°; pulse 2 (−500,
700ms) swung **15.5°** to −12.11° — *worse than it started*; abort at 2 no-progress.

**Root cause: rotation has a fixed minimum quantum of ~8-15° per pulse (700ms gave 8.2°
and 15.5°; 1500ms gives ~12.5° — angular yield is NOT proportional to duration, same as
the linear step), but `heading_tolerance_degrees` defaults to 3.0°.** The controller
cannot land inside its own deadband, so any error smaller than the step oscillates and
diverges. When initial error < step size, turning is strictly counterproductive.
H2 wrong-direction slow-cap + the 2-pulse abort worked exactly as designed and bounded
the damage — the safety layer is sound, the SETPOINT was wrong.

### 5. Multi-segment L-path PROVEN with `heading_tolerance_degrees: 18` (one param)

Retry succeeded: **both segments `target_reached`, 2/2 real, `path_complete`.**
seg1 turn −18.4°->−11.2° (1 pulse) + 4 linear, landed 13.6cm; **seg2 turn +83.4°->+9.6°
in 5 pulses (~14.8°/pulse) + 7 linear, landed 6.5cm from target** — the ~90° VIO turn
phase proven again (broken since the 07-15 sunset abort). 2m22s at ble_rssi −94.
**Cross-validation of the tape: 11 linear pulses moved 1.072m = 9.75cm = 3.84in/pulse.**

### 6. Card fixed (`v2026.07.18b1`) — its own defaults were the failure modes

The card was sending `linear_pulse_duration_ms: 2000` (**a taped PHYSICAL NO-OP**),
`heading_tolerance_degrees: 8` (inside the 8-15° step band), no `min_progress_distance`
(-> schema 0.01, inside the noise floor), no `turn_pulse_duration_ms` (-> 300ms), and
`sample_delays [0,5,10]`. All corrected to the proven config (3500 / 18 / 0.06 / 1500 /
[0,3], `max_linear_commands` 3); stale "2s pulses move ~8-10cm" comment replaced with
the taped findings. Dry-run confirmed every value lands.

**⚠️ THE CARD LIVES IN TWO PLACES.** The dashboard resource is
`/hacsfiles/mammotion/mammotion-custom-path-card.js?v=N` ->
`/config/www/community/mammotion/`, **NOT** the integration's
`/config/custom_components/mammotion/www/` (served at `/mammotion/`). Deploying to only
the integration path md5-verifies GREEN while the browser loads the old card — a silent
failure. Deploy to BOTH. Also: bump `CARD_VERSION` + `?v=N` every deploy, and HA's
frontend **service worker** can still serve stale JS afterwards (reset frontend cache).

**Blank card map fixed** by the documented config-entry reload *while the mower was
awake* — containment then passed (`valid: true`, no `no_area_geometry` warning) and BLE
survived the reload.

### 7. ❗ OPEN: card run stalled in the turn, and the feed may under-report ~5x

Real card run needed −67.7°. Pulses yielded **9.5°, 9.4°, 5.7°, 0.001° (poll timed out
8.07s, `heading_went_fresh` FALSE), 0.49°** -> `no_heading_progress`, final error −42.6°,
`linear_commands_sent: 0`. **VIO was HEALTHY throughout (live, 80 features, Light)** — not
a dusk path. VIO heading (−25.05°) and course-over-ground `toward` (+26.0°) INDEPENDENTLY
agree it rotated ~25-30° then stopped. Yield was ~9.5°/pulse vs **14.8° in the successful
run 90 min earlier at identical params** — location/moment-specific, not parameter-driven.

**UNRESOLVED, TOP PRIORITY:** the run logged **15-16cm** total displacement; the operator
observed **~1ft forward + ~2.5ft left ≈ 82cm** — a **~5x UNDER-report**, on pulses
commanded at `linear_speed: 0`. (`displacement_m` did climb 1.7->7.1cm across those
pulses, so some translation registered, just far too little.) If real, the mower converts
commanded rotation into forward/lateral lurch AND the feed misses most of it — which
would undermine the linear phase's progress logic and cast doubt on prior "landed Xcm
from target" claims. **Caveat: the operator figure is a recollection, not a tape, and
could NOT be verified — the mower auto-docked at 01:06Z before a check was possible.**

**NEXT DAYLIGHT SESSION, FIRST TEST (~2 min): mark the ground, fire ONE rotation pulse,
tape BOTH the rotation and any translation.** Do not redesign progress logic before this.

### 8. Scope correction (operator)

An "autonomous-mow telemetry test" was queued from the 07-18 plan; the operator stopped
it — the goal is **point-and-click map movement with the blade OFF**, not mowing/blade
automation. That telemetry item is a separate side-quest, already root-caused off-mower.
An item appearing on a queue is not sufficient reason to run it.

### Defaults to fix off-mower (each with live evidence)

1. `heading_tolerance_degrees` **3.0 -> ~18** — strongest; proven necessary AND sufficient
   in back-to-back runs an hour apart.
2. `execute_segment.linear_pulse_duration_ms` **300 -> ~3500** — 300ms is below the no-op
   threshold, so any caller taking the default gets stop-validated pulses that never move.
3. `min_progress_distance` **0.01/0.003 -> ~0.06** — inside the measured noise floor.
   **Handle with care**: changes abort behaviour in proven motion loops; wants tests.
4. Minor: the multi-segment result does NOT echo `linear_pulse_duration_ms`/
   `turn_pulse_duration_ms`/`vio_turn_max_commands`/`vio_angular_speed`/
   `max_linear_pulse_ceiling` (the handler DOES forward them — verified). Echo-only gap,
   but it blocks post-run forensics on the numbers that matter most.

Also noted: `position_source_comparison`'s `mowing_state_xy` is flat **[0.0, 0.0]** with
`rtk_status` dropping 4(`Fix`)->0 during motion — the second position source is dead on
this firmware, so the deferred phantom detector (#3) has only ONE live input.

Session end state: mower docked 01:06Z, `Dark`, 0 tracked features, transport back to
`cloud_aliyun`, blade OFF.

## PLAN 2026-07-20: app-parity motion (control.py / rocker_util.py finding)

### The discovery

Two pymammotion files nobody on this project had opened describe how the phone app
actually drives the mower:

- **`pymammotion/mammotion/control.py`** (`JoystickControl`) — a working reference
  driver. Its critical line is `PeriodicThread(0.2, self.run_movement)`: it **re-sends
  `send_movement` every 200 ms, continuously, while the control is held**, and it does
  NOT dedupe on unchanged values — it fires every tick regardless.
- **`pymammotion/utility/rocker_util.py`** (`RockerControlUtil`) — the app's on-screen
  thumbstick math, carrying `"""generated source for class ..."""` docstrings (the
  fingerprint of automated Java->Python decompiler output, i.e. the upstream authors'
  decompile of the app, not ours).

**Our motion model is the opposite.** Every motion site in `services.py` sends ONE
`send_movement`, then `_motion_open_sleep(duration)`, then a stop — the helper's name
encodes the assumption that the command stays "open" and the mower keeps moving.

**HYPOTHESIS (H-WATCHDOG):** `send_movement` grants motion for a short window and the
mower self-halts unless refreshed. The app never notices because it refreshes at 5 Hz.
This would explain every taped anomaly at once: the fixed ~4in step independent of
duration (2s->0", 4s->4", 6s->4"), the marginal/no-op 2s pulses, the ~8-15 degree
rotation quantum that makes the 3.0 degree turn deadband oscillate, and why 3.5m needs
~35 pulses at ~14s cadence (which outlives the BLE link).

### ✅ CONFIRMED BY APK DECOMPILE (2026-07-20, see Phase C below)

The app-side half of H-WATCHDOG is now **proven from the app's own source**, not inferred.
`com.agilexrobotics.command.CarRemoteControlManage2` (decompiled from
`Mammotion_2.3.8.19_APKPure.xapk`, `classes2.dex`):

```java
public static float frequency = 0.2f;          // 200 ms — also DeviceDeployModule.frequency
private long delay = 200;
...
long j = (long) (frequency * 1000.0f);         // 200
timer.schedule(this.countDownTask, 0L, j);     // fire now, then EVERY 200 ms
    -> run() { ... send(3); ... }              // -> maCommandHelper.sendControl(linear, angular)
    -> if (linearSpeed == 0 && angularSpeed == 0) cancelTimer();   // stops only when stick released
```

And `MACommandApiHelper.sendControl(int,int)` builds:

```java
MctlDriver.newBuilder().setTodevDevmotionCtrl(
    DrvMotionCtrl.newBuilder().setSetLinearSpeed(i).setSetAngularSpeed(i2))   // subtype 51, needAck=false
```

That is **byte-identical to pymammotion's `send_movement`**. So the app uses the exact
command we already use — the ONLY difference is that it re-sends it on a fixed 200 ms
timer for the entire time the stick is held, and stops by sending zeros then cancelling.
`CarRemoteControlManage` (v1) is the same design via `DeviceDeployConstants.frequency1`.

Corroborating detail: the app ships a **debug HUD** (`text_frequency`, `text_linear`,
`text_angular`) that displays the send interval in ms live — they exposed the send rate
as a tunable diagnostic, which is only worth doing if the rate is load-bearing.

Still strictly unproven: that the *firmware* enforces a timeout (the app would look the
same if it re-sent purely to track stick movement). But the timer is fixed-rate with no
dedupe on unchanged values, and our tape data shows a fixed distance quantum that the
duration parameter cannot influence. B1 settles it in one taped run.

**Also confirmed: `needAck=false`.** The app fires movement commands without waiting for
an ack. Our per-pulse `command_ok`/ack accounting is therefore not evidence of delivery —
consistent with the 2026-07-19 e-stop finding, where every health indicator read green
while five real motion commands were silently no-op'd.

### Speed-scale finding (independent of H-WATCHDOG)

`transform_both_speeds(theta_linear, theta_angular, pct_linear, pct_angular)` computes
`linear_speed = sin(rad)*pct*10` and `angular_speed = int(cos(rad)*pct*4.5)`, where
`get_percent()` applies a **15% deadband** (`pct<=15 -> 0`, else `pct-15`), so a
full-deflection input of 100 yields 85.

| | app full scale | ours today | note |
|---|---|---|---|
| linear | +/-850 (`control.py` comments say 1000) | **400** | we run at ~47% throttle |
| angular | +/-382 (comments say 450) | **500** | **above** the app's ceiling; may be clamped |

`mammotion_command.py` already exposes `move_forward/back/left/right(0.0-1.0)` which run
this transform properly — and the integration's directional-movement services use them.
Only our VIO/click-to-path code bypasses it with raw `send_movement(400, 0)`. The
"angular is weak, needs ~500" calibration may be an artifact of fighting the watchdog
and/or of exceeding the valid range, not a property of the machine.

### Phase A — off-mower, tonight (code + tests, NO deploy)

- **A1. Repeat-pulse primitive.** New helper that re-sends `send_movement` every
  `refresh_interval_ms` for the pulse duration, then the existing explicit stop.
  **Default 200 ms — matches the app exactly** (`CarRemoteControlManage2.frequency = 0.2f`,
  confirmed by decompile, not guessed). Mirror the app's stop semantics too: send
  `(0, 0)` and then cancel, rather than only cancelling. Add as an **opt-in parameter**
  (`motion_refresh_interval_ms`, 0 = current single-shot behaviour) so a single live run
  can A/B it against today's proven path without a rollback. Record per-pulse
  `commands_sent` in the result for forensics.
- **A2. Speed-scale plumbing.** Accept speed as the app's 0.0-1.0 scale and run it
  through pymammotion's `transform_both_speeds`/`get_percent`, so we stop guessing raw
  units. Add a read-only diagnostic that reports the resolved raw values (and flags
  anything above the +/-850 / +/-382 ceilings) WITHOUT moving the mower.
- **A3. `passed:False` bug** (standing TOP BUG): a segment that reaches its target is
  marked `passed:False` because short final-approach pulses fail `min_progress_distance`,
  so later segments never run. Fix + regression test.
- **A4.** Tests for A1-A3; ruff + mypy + full pytest. Deploy queued for daylight only.

### Phase B — daylight, supervised, tape measure (needs a fresh "go" per pulse)

Run in this order; each answers one question and gates the next.

- **B1. THE decisive A/B.** Forward 4s single-shot (today's behaviour), taped. Then
  forward 4s with 200 ms refresh, taped. **If repeat-mode travels several times further,
  H-WATCHDOG is confirmed** and the whole pulse/throughput model changes.
- **B2. Speed sweep** (only if B1 confirms): repeat-mode at 0.4 vs 1.0 app-scale, taped.
- **B3. Angular:** repeat-mode at our 500 vs the in-range 382, taped rotation.
- **B4.** Re-derive throughput, then revisit `turn tolerance 18`, `min_progress_distance`,
  pulse cadence and `sample_delays` against the new numbers.

### Phase C — protocol discovery (off-mower, parallel to A/B)

**Correction to the record: we never decompiled the APK.** `docs/custom-path-execution-research.md`
documents a lightweight **string scan** of the local `2.3.8.19` XAPK and explicitly says
"This scan is not as strong as a JADX decompile". The decompiled artefacts we have
(`rocker_util.py`, `bluetooth/data/notifydata.py`) are the pymammotion authors' work,
shipped in the dependency.

- **C1. Real JADX decompile — ✅ DONE 2026-07-20.** `brew install jadx` (1.5.6), unzipped
  the XAPK (base APK `com.agilexrobotics.apk`, 9 dex files), decompiled `classes2.dex`
  (3233 classes) to scratch. Findings are in the CONFIRMED section above: 200 ms
  fixed-rate re-send, identical `DrvMotionCtrl` wire command, `needAck=false`, debug HUD
  for the send interval. **No firmware-side timeout constant is visible in app code** (it
  would live in mower firmware, not the APK) — so B1 remains the decisive test, but the
  app-side cadence question is closed.
  Key classes for future reference: `com.agilexrobotics.command.CarRemoteControlManage2`,
  `com.agilexrobotics.command.app.MACommandApiHelper#sendControl`,
  `com.agilexrobotics.base_module.utils.RockerControlUtil`,
  `com.agilexrobotics.map.view.RockerTouchView{,2,3}`.
- **C2. Manual-mode entry — ✅ RESOLVED, not needed for driving.** `DrvMowCtrlByHand` is
  sent by `MACommandApiHelper#OperateOnDevice(main_ctrl, cut_knife_ctrl, cut_knife_height,
  max_run_speed, position)`, and its only callers are in `map/ManualLawnMowingManager`
  — i.e. it belongs to the manual **mowing** flow (blade + speed cap), not to driving.
  `CarRemoteControlManage2` is self-contained: it calls `sendControl` and nothing else, so
  **no mode-entry command precedes joystick movement**. Our raw `send_movement` is the
  correct and complete command; the only thing we were missing is the repeat.
  (One related detail worth keeping: when `deviceState == MODE_MANUAL_MOWING`, the app
  auto-raises the blade whenever both speeds hit zero — a blade-safety pattern, not a
  movement prerequisite. Irrelevant to us: we run blades OFF.)
- **C3. Ground truth (NOT needed).** An Android HCI snoop log would only re-confirm what
  C1 established from source. Skip unless B1 contradicts the decompile.

### Phase D — dependent on B1

- **D1.** If continuous drive works: the **hold-to-drive joystick card** (user-approved
  2026-07-15, deferred) becomes a thin wrapper over A1 — build it.
- **D2.** Re-test the BLE wall: a 3.5m path that takes ~30s instead of ~10min may simply
  dodge the coverage problem that has been killing runs.

### Explicitly NOT doing

- **No code review this cycle.** Two xhigh passes already ran; 305 tests, mypy + ruff
  clean. Nothing blocking is a code-quality problem.
- PR #10 stays open for human review; no merge.

## Wrap-up 2026-07-20 (late): Phase A implemented off-mower (`b91de636`)

A1-A4 all landed in one commit. **NOT deployed** — the cadence change is opt-in and
gated on the B1 tape A/B. Tests 282 -> 293 in the motion file (323 across the suite);
ruff + mypy clean.

**A1 — app-parity motion cadence.** New `_motion_refresh_window()` holds a pulse open
the way the app does: it calls a `resend` callable every interval for the pulse
duration, then the caller runs its existing mandatory stop. Opt-in via a new
`motion_refresh_interval_ms` parameter (0 = the proven single-shot path) on
`manual_velocity_pulse_test`, `raw_pymammotion_execute_vector_segment` and
`raw_pymammotion_execute_multi_segment` (schemas + handlers + services.yaml).

Design decisions worth remembering:
- **Refresh sends are counted separately from pulses** (`motion_refresh_commands_sent`,
  never `linear_commands_sent`). Folding them together would have silently broken the
  pulse ceilings that bound a run.
- **The loop is bounded by a computed command count as well as by wall clock.** A
  wall-clock-only loop spins forever when sleeps do not advance the clock (virtual-clock
  tests), and the count doubles as a hard ceiling if a bad interval slips through.
- **A refresh is never sent with no window left**, so the final command is always
  followed by real motion time rather than an immediate stop.
- **A failed refresh does not raise** — it records `refresh_error` and stops refreshing,
  leaving the caller's stop intact. A half-refreshed window is a shorter drive, never a
  runaway one.
- Interval is clamped to 50-1000 ms.

**A2 — app speed scale.** `_app_scale_speeds()` / `_app_speed_scale_report()` mirror the
app's rocker transform (15% deadband, x10 linear, x4.5 angular). Implemented locally so
the numbers stay deterministic across dependency bumps, with
`test_app_speed_scale_matches_pymammotion` pinning them to
`pymammotion.utility.movement`. The report is read-only and rides along in results; it
flags that our angular 500 is above the app's 382 ceiling and our linear 400 is ~47%
throttle. Note `manual_velocity_pulse_test`'s `speed` is ALREADY app-scale 0.0-1.0 (it
routes through the coordinator's directional helpers), unlike the executors' raw values.

**A3 — arrived segments no longer fail.** `_raw_multi_segment_phase_passed` required
EVERY per-pulse progress diagnostic to clear `min_progress_distance`. Final-approach
pulses necessarily move less than that, so a segment could reach its target and still be
marked `passed: False`, stopping the run so later segments never executed. Arrival now
decides (`target_reached` + valid + unblocked); the genuinely-stuck case is still caught
by the executor's consecutive-no-progress abort, which never reports `target_reached`.

**Deliberately left alone:** `_raw_pymammotion_execute_segment` (the legacy executor)
still runs single-shot. It now has the same shape as the vector executor so wiring it is
mechanical, but it is not on the card path and the diff should not widen before B1.

**Next session is B1** — see the Phase B section above and `docs/NEXT-SESSION.md` for the
exact service call. Everything downstream (throughput re-derivation, turn tolerance,
joystick card) waits on that one measurement.

## Wrap-up 2026-07-21/22: read-only observation of a live autonomous mow + full-APK decompile

Off-mower, read-only session (operator away from the mower). Nothing deployed during the
observation, no motion command sent, the mow was not disturbed. Three standing beliefs
were overturned by data.

### 1. ❌ REFUTED: "obstacle/report telemetry goes dormant during autonomous mowing"

The channel is **live**; the field is simply always zero.

- `sensor.fuse_status` reads `report_data.dev.fuse_status` and has **no writer anywhere in
  pymammotion's `device/state_reducer.py`**, so it can only be set by the wholesale
  `device.update_report_data()` on `toapp_report_data` (the binary RptDevStatus).
  It **toggled 1↔2 at 23:21:37/41/47 and again at 23:49:33/35 — mid-mow.**
- `sensor_status` — which packs bumper (bits 0-2) and all four ultrasonics (bits 12-23) —
  lives in that **same `DeviceData` struct** (`report_info.py:147`). It was refreshed on
  those same ingests and decoded to all-zeros.

**Why it is always zero (full-APK decompile):** `sensor_status` is a hardware
**self-check** readout, not a live obstacle channel. `SelfCheckFragment.checkResult()` is an
explicit **pre-work checklist** (battery >15%, RTK lock, bumper presence, vision) driven by a
500 ms poll and run *before* a job starts — not a live-mowing overlay. `DeviceUltEvent` (fired
when an ultrasonic bit changes) is consumed only by an internal engineering screen. Our bit
decode is structurally correct and matches the app's own decode.

**Consequence: obstacle-based HA automations cannot work — and a "keep the report stream
alive" coordinator fix would change nothing.** The previously-designed cloud keepalive is
therefore **withdrawn**, not deferred. HA history over 11 days / 27 mow sessions: the five
safety sensors changed value **zero** times (only `OK`↔`unavailable` restart gaps and the
07-18 case rename).

**⚠️ Method trap — this cost two wrong conclusions before the right one.** In this HA,
`last_reported` never advances without `last_updated` (0 same-value writes observed across 42
polls). **A frozen `last_reported` proves only that the value never changed — it is not
evidence of a dead channel.** Also, `battery_val` and `sys_status` are MQTT-fed
(`state_reducer.py:701,703,766,767`), so their movement proves nothing about the binary
report. `fuse_status` and `vio_survival_info` are the clean binary-report witnesses.

### 2. 🏆 Continuous-motion baseline: the position feed is ~10x better than documented

First characterization under *known-continuous* movement (154.6 m of BLE-tracked mowing path;
118 samples at 6 s plus a 100-sample 0.77 s burst). Previously we had only ever measured it
during our own pulsed motion.

| metric | pulsed motion (prior) | continuous motion (measured 07-21) |
|---|---|---|
| position error | ~2-6 cm absolute floor | **0.70 cm cross-track RMS** (median 0.42, max 2.00) |
| update behaviour | "jumps", frequent freezes | **0 of 86 consecutive samples bit-identical** |
| update interval | ~4 s lag | ~0.77 s median between distinct positions (max gap 5.33 s) |

Cross-track RMS is measured as the perpendicular residual about each run's own best-fit
principal axis, over 8 straight runs of 5-10 m. It bounds *random* position noise; it does not
bound along-track error, which is where feed latency shows up. The feed tracked the mower
faithfully from 0.2 to 0.65 m/s, cross-checked against `work.man_run_speed`.

**So the documented "2-6 cm noise floor + jumpy updates" is an artifact of pulsed measurement
(start/stop transients, ~4 s lag, single-sample reads), not a property of the feed.**
If B1 confirms app-parity refresh yields continuous motion, `min_progress_distance: 0.06` —
chosen to clear a floor that largely is not there — can likely be tightened substantially.

**Throughput gap, now quantified:** autonomous cruise is 0.2-0.65 m/s; our pulsed model yields
~10 cm per ~4 s pulse ≈ **0.025 m/s, i.e. ~25x slower than the machine's own cruise.**
Independent support for H-WATCHDOG and for keeping B1 as the top priority.

### 3. 🔌 The position feed is BLE-only; on cloud it is stone dead

Live, mid-mow: **20 of 21 consecutive `cloud_aliyun` polls were bit-identical frozen; 74 of 74
`ble` polls moved.** Transport flapped `ble`↔`cloud_aliyun` repeatedly within one mow as the
mower drove out of proxy range (rssi -64 → -98). Same proxy-coverage wall as before, now
measured from the data side rather than the log side.

### 4. 📡 The app's report-subscription path (full 9-dex decompile)

Full decompile at `/Users/mattjoslin/mammotion-apk-decompile/src` (415 MB, 30 867 `.java`,
**outside the repo**; `*.xapk` is gitignored). The previous session had decompiled only
`classes2.dex` — one ninth of the app.

The app's BLE "start streaming" message is `MctlSys.todev_report_cfg` (field 38), built by
`MACarDataManager.requestMapLocationBTData()` → `MACommandApiHelper.requestMapLocationBTorIOTData()`
(`MACommandApiHelper.java:1373-1378`):

```
act=RPT_START, timeout=10000ms, period=2000ms, no_change_period=4000ms, count=0 (continuous),
sub=[RIT_CONNECT, RIT_RTK, RIT_DEV_LOCAL, RIT_WORK, RIT_DEV_STA, RIT_VISION_POINT,
     RIT_VIO, RIT_VISION_STATISTIC, RIT_BASESTATION_INFO, RIT_CUTTER_INFO]
```

`DeviceUtils.setCountKeep(0)` is forced immediately before every BT send
(`MACarDataManager.java:8432`). **The app never sends RPT_STOP over BLE** —
`stopMapIotMessage()` early-returns when the link is Bluetooth
(`MACarDataManager.java:9019-9036`); it is an IOT-only path.

Two divergences in our stack (**not acted on** — see below): pymammotion's `get_report_cfg()`
(`commands/messages/system.py:354-416`) defaults **`count=1` (one-shot)** where the app forces
**`count=0`**, and substitutes `RIT_FW_INFO` for `RIT_CUTTER_INFO`; period 1000 vs 2000.
`mower_api.py:44,107-109` calls it argless on a ~5 s throttle, i.e. we re-poll one-shot reports
rather than opening one continuous stream. **Deliberately not changed:** this was only ever
interesting as a dormancy fix, and dormancy is disproven — on BLE the feed already streams
fine, and on cloud a subscribe will not fix a dead link. Low value, real regression risk on a
proven path.

Also: the boolean on `sendOrderMsg_Sys` is a **BLE-vs-IOT routing hint, not an ack flag**.

### 5. ✅ Task-2 re-verification against the full tree: prior conclusion HOLDS

"`DrvMowCtrlByHand` / `OperateOnDevice` is only called from `ManualLawnMowingManager`, so no
manual-mode entry precedes joystick driving" was derived from `classes2.dex` alone. Re-tested
exhaustively across all nine dex files: **called from exactly one class,
`ManualLawnMowingManager`, and nowhere else.** Confirmed, unchanged.

### 6. 🔎 The e-stop may not be fully invisible after all

HA history shows `sensor.last_error` = **`mcu: STOP button triggered`** at 2026-07-19T18:35:12
(`last_error_time` 18:31:27) — the forgotten-e-stop incident. The 07-19 raw-field snapshot
checked `lock_state` / `self_check_status` / `sys_status` / `sensor_status` / `bumper_state`
but **never looked at the error channel.**

**Caveat: it surfaced ~40 min after the press, at the end of the incident**, so on this single
datapoint it is a confirmatory signal, not a real-time detector. **Queued for the next on-mower
session (cheap):** press the e-stop deliberately and watch `sensor.last_error`. That
distinction decides whether it is worth wiring into `no_actuation_detected`'s hint.

### Honest limits of this session

The APK analysis workflow **hit the account's monthly spend limit mid-run** — 9 of 60 agents
completed; the adversarial verify pass and the final synthesis never ran. The Task-1
conclusions above are from live data gathered directly and are independently reproducible from
HA history. The report-subscription numbers in §4 are high-confidence quotes with file:line
evidence but only two claims received adversarial verification — **re-check before acting on
them.** §5 rests on an exhaustive `rg` sweep of the full tree, which is self-verifying.

**B1 remains the next mower action, unchanged.** Nothing in this session moves it.

## Wrap-up 2026-07-22 (later, off-mower): executors default to app-parity refresh (Task 2, code half)

**What changed (code, NOT deployed):** the two click-to-path executors —
`raw_pymammotion_execute_vector_segment` and `raw_pymammotion_execute_multi_segment` —
now default `motion_refresh_interval_ms` to **200** (was 0) in their **voluptuous schemas**
(`services.py` ~928 / ~1041) and `services.yaml` (the two executor field blocks). B1
(2026-07-22) proved refresh-200 drives ~11x further than a single shot, so the services the
card actually drives now get continuous linear motion by default.

**Deliberately left single-shot (default 0):** `manual_velocity_pulse_test` (the bare-pulse
A/B harness) and `vio_turn_probe` (the turn probe). They exist to compare 0 vs 200 *explicitly*;
defaulting them on would corrupt the experiment. Refresh is also proven **speed-gated** — it did
nothing for the under-powered angular-180 turn — so a defaulted-on turn probe would be
actively misleading.

**Convention followed:** as with the prior "proven live config" changes
(`linear_pulse_duration_ms` 300→3500, `heading_tolerance_degrees` 3.0→18,
`min_progress_distance` 0.01/0.005→0.06), only the **schema + yaml** defaults moved. The
executor **function signatures keep their conservative primitive defaults** (`= 0` here, like
`linear_pulse_duration_ms: float = 300.0`). The readiness probe (`services.py` ~6963) passes its
diagnostic values explicitly, so it is unaffected. Refresh is consumed **only** in the linear
pulse loop (`_motion_refresh_window` at ~8569); the calibration drive and VIO turn phase never
call it — verified by grep (the only three `_motion_refresh_window` call sites are the linear
loop, the manual-pulse harness, and the turn probe).

**Tests:** the parametrized schema-defaults guard now pins both executors at 200; a new
`test_motion_refresh_default_split_executors_on_harnesses_off` locks the whole invariant
(executors 200, harnesses 0) so a future edit can't silently flip either side. **326 tests pass,
mypy clean, ruff clean, services.yaml parses (58 services).**

### Re-derivation plan (Task 2, second half — needs the supervised segment run)

The default flip changes the *physics* of a linear pulse, so three tuned constants that were
derived against the old single-shot ~4in step must be re-derived against continuous drive.
**None of these can be re-derived from the desk** — they need one supervised segment run of the
executor with refresh on, which is currently blocked by the map-sync bug (Task 3). What follows
is the model to test and the measurements to take.

**The number that changed.** One linear pulse used to be a fixed ~4in step regardless of
duration; with refresh it is genuine continuous velocity. B1 measured **~28 cm/s** (112 cm taped
over a 4 s window; the ~1.3 m glide corroborates ~32 cm/s). So a `linear_pulse_duration_ms=3500`
pulse now covers **~1.0 m**, not ~4 in — a ~10x jump in per-pulse distance. A 3.5 m path is now
**~3–4 pulses / ~12 s of drive**, vs ~35 pulses / 10+ min that always outlived the BLE link.

1. **Pulse geometry — `max_linear_commands` / `max_linear_pulse_ceiling` /
   `linear_distance_ceiling_factor`.** At ~1 m/pulse a 3.5 m segment needs ~4 loop iterations,
   not ~35. The ceiling logic (derived assuming ~10 cm/pulse) will now let the loop overshoot
   badly if left as-is — a single default-ceiling run could drive several metres past target
   before the pulse count trips. **Re-derive the ceiling from `segment_length / ~1.0 m` plus a
   small margin, and re-check `linear_distance_ceiling_factor` (the distance-based cap) against
   the new per-pulse distance.** Also decide: the schema caps `linear_pulse_duration_ms` at
   4000 ms (~1.1 m); keeping ~3.5 s pulses gives a re-aim/progress check every ~1 m (good for
   tracking), which is probably right — do **not** raise the cap chasing fewer stops until the
   veer behaviour (below) is re-measured.

2. **`min_progress_distance`** (currently 0.06, chosen to clear the *pulsed* 2–6 cm noise
   floor). Two competing facts from the mow-observation session: during **steady continuous**
   motion the feed is **sub-cm** (0.70 cm RMS, zero frozen samples), which argues for tightening
   0.06 → ~0.02; **but** during **fast/pulsed** motion the feed **lags ~4 s and under-reports**
   (B1 pulse B read 3.98 cm for a taped 112 cm). The executor judges progress *after*
   stop+settle, so the governing question is **whether the settle poll fully absorbs the ~4 s lag
   before the progress read.** Re-derivation: on the segment run, log per-pulse
   `settle_polls` / `observed_jitter` and the settled displacement vs a tape, and only tighten
   `min_progress_distance` once the settled read is proven to reflect the true ~1 m pulse. If the
   settle does **not** fully catch up at ~28 cm/s, the fix is a longer/settle-until-quiescent
   poll, **not** a smaller `min_progress_distance`.

3. **Cadence / BLE exposure.** Wall-clock per pulse = drive (~3.5 s) + stop (~ms) + settle
   (1–2 s, maybe more to absorb lag) + `sample_delays`. `sample_delays` is **forensic, not
   control** — during a real run it should be trimmed to near-zero so BLE exposure ≈ drive +
   settle. Target budget for a 3.5 m path: ~4 × (3.5 + ~2) ≈ **~22 s**, comfortably inside the
   BLE window that killed the old ~10 min runs. The hypothesis to confirm on the run: **faster
   drive dodges the −70 BLE coverage wall on its own.**

**Validation run (gated on Task 3 map-sync fix, then a fresh operator "go"):** one supervised
`raw_pymammotion_execute_vector_segment` of ~2–3 m with refresh at its new default. Capture
per-pulse: commanded vs taped distance, `settle_polls`/`observed_jitter`, settled displacement,
`distance_to_target` trajectory, aim-error realignments, and the BLE rssi trace. Then set the
three constants from that data. **Do not ship new values for (1)–(3) before this run** — they
are currently *hypotheses* from bare-pulse (B1) and autonomous-mow (read-only) data, neither of
which exercised the executor's own settle/sample/progress loop under refresh.

**Also queued by the same finding (not Task 2, tracked here so it isn't lost):** the veer —
the 07-19 run needed 15–19° realignments and tracked off the ideal line; re-measure
straight-line tracking under continuous drive *after* the throughput constants are set, and the
D1 hold-to-drive joystick card is now unblocked (thin wrapper over `_motion_refresh_window`).

## Wrap-up 2026-07-22 (later, off-mower): map-sync bug diagnosed + two recovery fixes (Task 3)

**Symptom.** After a reload/restart the zone polygon geometry never re-projects for an idle
mower: `get_geojson` returns points + a line but no Polygon, `map_sync_status: out_of_sync`,
and `raw_pymammotion_execute_vector_segment` dry-run fails `path_validation` /
`area_hash_not_found`. Two config-entry reloads + a mower restart did not fix it.

**Root cause — `coordinator.data.map.area` (the polygon vertex frames) is empty; both symptoms
follow from that.** Traced the dependency chain:

- Containment does **not** read the geojson. `_validate_custom_path` → `_area_polygons`
  (`services.py:1407`) reads `map.area[hash].data[].data_couple[]` frames directly.
  `area_hash_not_found` fires only when the hash isn't a key in `map.area` (`services.py:2025`).
- The geojson (`generated_geojson`) is *derived* from those same frames via
  `generate_geojson(RTK, dock)`. Empty `map.area` → points (dock/RTK) + a line, no polygons.
- So "never re-projects" = "the map-sync saga never populated `map.area`, and nothing recovers
  it for an idle mower."

**Why it doesn't self-heal — contributors, ranked:**

- **A (transport/convergence, likely dominant).** `_async_update_data` fires `start_map_sync`
  whenever `not is_map_synced` (`coordinator.py:2332`), but: restore silently falls back to an
  **empty `MowingDevice`** on `InvalidFieldValue` (`coordinator.py:1552`); and if the saga can't
  complete over the current transport, `saga.result is None` → on-complete skips restoring
  `root_hash_lists` (`client.py:2115`) → `update_hash_lists` filters `map.area` down to nothing
  (`hash_list.py:475`) after `invalidate_maps` cleared the area hashes. Repeated reports →
  repeated invalidation → never converges. This is transport-sensitive: a mower restart drops
  BLE to cloud (the −76 wall), exactly when the saga is least likely to finish — which is why
  "reload while awake" failed that night.
- **B (geojson regen gaps, genuine code bugs).** Even with frames present the geojson has no
  idle-time regen path: the triggers are the saga on-complete (**skipped when
  `RTK.latitude == 0.0`**, `client.py:2120`) and two `state_reducer` paths that only fire on the
  mowing report hot path (`state_reducer.py:338,487`). And pymammotion ships
  `regenerate_stale_geojson()` whose docstring says *"call after `restore_device()` in the HA
  coordinator"* (`client.py:730`) — **our integration never called it** (grep: zero hits).

**Fixes shipped (code, NOT deployed, NOT committed; gated 330 tests + mypy + ruff clean):**

1. **`coordinator.async_restore_data` now calls `self.manager.regenerate_stale_geojson(self.device_name)`**
   right after `handle.restore_device(mower_state)` — closes the documented-contract gap so a
   restored `map.area` re-projects at startup instead of waiting for a mow. pymammotion guards it
   (no-op when `map.area` is empty or the yaw/hashes are unchanged), so it's safe.
2. **New `mammotion.force_map_resync` service** (`SERVICE_FORCE_MAP_RESYNC`, entity-scoped,
   `SupportsResponse.ONLY`, allowlisted in the test's `known_undocumented`). Coordinator method
   `async_force_map_resync()` runs a **non-destructive** recovery: refresh RTK/dock (so the
   on-complete rebuild isn't skipped) → fetch the area-name list (some cloud sessions never push
   `toapp_all_hash_name`) → run the saga (its on-complete restores `root_hash_lists`, the
   convergence fix) → `regenerate_stale_geojson`. Returns a step-by-step result
   (`map_sync_status_before`/`after`, `steps`, `error`, `last_map_task_error`) for the card to
   surface. Non-destructive on purpose: the existing cache is left intact until the saga replaces
   it, so a failed resync never leaves the map worse off. Deliberately **not** a new button entity
   (would need all 12 locale translations); a dashboard button can bind to the service.

**Still open — one live read decides A vs B (can't from the desk).** If an active mow re-projects
the map but idle never does → **B** (the two fixes resolve it). If even a mow leaves `map.area`
empty → **A** (a BLE-coverage/transport problem, not a code bug; the fixes only aid recovery).
Read without motion via `get_map_data` / `_export_mower_map` (does `map.area` have keys?
computed vs reported `bol_hash`? RTK/dock latitude?), `sensor.<mower>_last_map_task_error`, and
`map_sync_status`. Then, on a good-BLE moment, call `mammotion.force_map_resync` and re-check.
This unblocks the Task-2 validation segment run.

## Wrap-up 2026-07-22/23 (off-mower): full APK 2.3.8.19 multi-agent feature sweep

**Goal:** preserve the entire app's discoverable feature surface for later use, not only the
motion findings that originally motivated the decompile.

**Scope and output.** Swept the complete nine-dex JADX tree (30,867 Java files, ~415 MB,
177 first-party `com.agilexrobotics.*` manifest components). The durable catalog lives in
`docs/apk-feature-catalog/`:

- `00-overview` — snapshot, method, package census and reading cautions;
- `01`–`09` — onboarding/connectivity, mapping/deployment, work planning/execution,
  manual control/safety, device settings/maintenance, camera/video/vision,
  account/sharing/cloud, SPINO pool cleaner, and hidden diagnostics/testing;
- `10-protocol-report-index` — reachable commands, reports, topics, APIs, routing and ack
  semantics;
- `11-ha-opportunity-index` — future HA backlog classified by reversibility, transport,
  safety/security risk and confidence;
- `12-coverage-and-open-questions` — honest coverage statement, audit history and runtime
  verification queue;
- `13-model-capability-matrix` — product codes, LUBA/YUKA/mini/SPINO/RTK/dock gates,
  firmware/server/runtime capability checks and identity collisions;
- `14-architecture-glossary` — startup, device abstraction, BLE/BLUFI/cloud/MQTT,
  command/report/cache/map flows, native/H5/RN boundaries, persistence and terminology.

**Quality passes.** After the subsystem reports landed, separate agents ran:

1. an omission audit against manifest activities/services, all first-party package owners,
   resource strings, API interfaces and command-manager methods;
2. an adversarial exact-claim pass over motion cadence/scaling, routing boolean and ack
   semantics, report-subscription configurations, map ordering, SPINO plan wire fields and
   Agora stream encryption;
3. a mechanical citation/table audit (1,580 citations in its first pass; 76 optimistic or
   invalid ranges found), followed by disjoint repair passes; and
4. a final whole-catalog validation including the later synthesis reports.

**Material corrections caught by the audits:** SPINO `PlanJobSet` uses `fixed32` for
`start_time`/`day`/`weeks`/`enable` (with inverted enable polarity), Agora explicitly uses
`AES_256_GCM2`, `requestIOTMessage` differs by helper generation (nine vs ten report types),
Wi-Fi closure only adopts an existing BLE link in that call path, the blade RSSI ten-sample
rule governs automatic shutdown rather than reliably gating initial start, and the misleading
`AudoBackwashPop` class is a generic 4G-disable warning—not a pool backwash feature.

**Omissions added after independent discovery:** SIM/iNavi Shopify purchase/renewal links,
server-driven tips/show/read tracking, RTK positioning-optimization guidance, Mammotion's own
behavioral telemetry endpoint, downloaded/versioned localized error-code catalogs, and
post-onboarding Wi-Fi/4G radio + APN controls.

**Honest limit:** this is a full static decompile sweep/catalog, not proof of every
server-controlled, firmware-gated, RN-hotfix/H5-delivered feature and not authorization to
implement hazardous commands. Packet capture, representative hardware, test accounts, and
newer-APK diffs remain explicitly queued in report 12.

## Deploy 2026-07-23: Task 2 refresh defaults + Task 3 map recovery live-loaded

Deployed the committed `1b22da7a` versions of `coordinator.py`, `services.py`, and
`services.yaml` to `/config/custom_components/mammotion/`; SHA-256 matched local/host for all
three. HA Core restart was explicitly authorized and returned HTTP 200. API was healthy after
41 s; Mammotion reached 121 entities after 129 s.

Post-restart read-only verification:

- 59 Mammotion services registered;
- `force_map_resync`, `vio_turn_probe`, vector executor, and multi executor present;
- deployed service defaults: vector refresh 200, multi refresh 200, turn-probe refresh 0;
- startup logs: normal Mammotion domain/platform setup, no import/setup exception;
- only pre-existing warnings: custom integration not HA-tested and
  `MammotionTracker.battery_level` deprecation for HA 2027.7.

No mower-motion service was called. `force_map_resync` was not fired; its first live use still
belongs to the good-BLE, read-before/read-after map-recovery workflow in `NEXT-SESSION.md`.

## Wrap-up 2026-07-24 (off-mower, night): zone_hash read the map checksum; map is healthy again

**Conditions.** Night, mower docked (`CHARGE_ON`, battery 100), `camera_brightness: Dark`,
0 tracked features, `SIGNAL_NONE`, `active_transport: cloud_aliyun`, `ble_rssi: 0`. Both
headline experiments (the `vio_turn_probe` refresh A/B, and any segment run) were
impossible. No motion service was called; nothing was deployed.

### The map recovered — the Task-3 blocker is clear

Read-only via `get_map_data` / `get_areas` / `get_geojson` / `validate_custom_path`:
`map.area` holds all 4 areas with full polygon frames (Backyard Right 72 pts, Front Right
62, Front Main 60, Backyard Hill 61), `area_name` has 4 entries, the GeoJSON carries 7
Polygons, and `validate_custom_path` returns `valid: true` for real paths — including with
an explicit `area_hash` of a real area. So the Task-2 validation segment run is **no longer
gated on the map**.

`area_hash_not_found` still fires correctly for a hash that is not an area. Worth knowing:
the docked mower's reported hash (`8311072749804434520`) is exactly such a hash, and
`get_area_entity_name()` returns the literal `"path"` for anything not in `map.area` — so a
`"path"` area label means *"not a mapped area"*, not a path feature.

A-vs-B (transport-convergence vs geojson-regen) was **not** decided: the symptom resolved
before it could be attributed, and what fixed it — the deployed `regenerate_stale_geojson`
call, the 07-23 restart, or an intervening mow — is unknown. `force_map_resync` still has
never been fired.

**A real leftover: `map_sync_status` reads `out_of_sync` on a complete, usable map.** Not
cosmetic — `coordinator.py:2396` fires `start_map_sync` on every coordinator tick while that
holds, so an exclusive saga repeatedly takes the device command queue for nothing. Added
`coordinator.map_sync_diagnostics()` (read-only, sends nothing) which breaks
`is_map_synced()` back into its three conditions — bol-hash match, incomplete-area hashes,
area-name coverage — and surfaced it in `get_map_data` under `map_sync` and in the
`force_map_resync` result as `map_sync_diagnostics_before`/`_after`. One read next session
identifies the failing condition.

*Honest limit:* the reported `bol_hash` is `8311072749804434520` and no permutation of the
four area-frame hashes MurMurs to it, but `computed_bol_hash` is built from
`root_hash_lists` (not `map.area` keys), which is not readable remotely — so a hash mismatch
is likely but unproven. That is precisely what the new diagnostic settles.

### 🐛 `zone_hash` was reading `bol_hash` — five guards inert at once

`rpt_dev_location` carries two distinct fields: **`zone_hash` (proto field 5)**, the mowing
zone the mower is currently inside, and **`bol_hash` (field 6)**, a MurMur checksum of the
device's whole area set. `services.py` read `bol_hash` at both sites where it meant
`zone_hash` (introduced in `491e0bf9`, "Use valid area telemetry for pulse safety").

A map checksum is non-zero whenever any map exists and is constant across a run, so the
substituted value could never be 0 and never changed. Consequences, all one bug:

1. `_is_stale_zero_area_out_pose` — the (0,0)/AREA_OUT stale-dock-pose rejection needs
   `pos_type == 0` **and** `zone_hash == 0`; it never saw a zero, so it was dead code;
2. the `location_metadata` overlay that corrects a stale pose never triggered;
3. `zone_hash_unavailable` in `_manual_velocity_quality_degradation` never fired;
4. `zone_hash_changed` — leaving one zone mid-run was undetectable;
5. the zone half of `_is_valid_motion_position` / `_position_has_known_area` was inert,
   leaving `pos_type_label` to carry the gate alone.

**Evidence, three independent sources.** The APK proto declares both fields on the same
message (`sources/com/agilexrobotics/proto/MctrlSys.java:60136-60143`,
`ZONE_HASH_FIELD_NUMBER = 5` / `BOL_HASH_FIELD_NUMBER = 6`, both compared in its
`equals()`). The app reads `locationsList.get(0).getZoneHash()` for the live zone alongside
`RealPosX/Y/Toward/PosType` and logs it as `judgeAccessibility zoneHash=`
(`MACarDataManager.java:4821`), while it logs `bolHash` against its own `getDBCmHash()` as a
**map** comparison — "车端bolHash … 本地hashUnsigned" (`HashDataManager.java:303`).
pymammotion's `RptDevLocation` has the same numbering, and writes field 5 to
`location.work_zone` (`device.py:201`). Live on the docked mower, one message reported
`zone_hash = 0` and `bol_hash = 8311072749804434520` simultaneously.

**Fix.** Both sites read field 5. The checksum is still reported, now under its own name
`position.map_bol_hash`, and `_position_feedback_raw_sources` lists `zone_hash` and
`bol_hash` side by side so past-run forensics stay interpretable. 4 regression tests, each
verified to fail against the old read and pass with the fix. 336 tests, mypy + ruff clean.

**⚠️ This makes the motion gate strictly stricter.** If the firmware reports `zone_hash: 0`
while `pos_type` is `AREA_INSIDE`, motion that used to run will now be refused — fail-safe,
but blocking. The docked reading (`zone_hash: 0`, `pos_type: 5`/`CHARGE_ON`) is correct and
proves nothing either way. Pre-flight check, read-only, before the next real command:
`position_feedback_diagnostic` with `pulse_count: 0, dry_run: true`, then confirm
`raw_sources."report_data.locations"[0].zone_hash` is non-zero while `pos_type` is 1.

### Method note

`get_map_data` returns `area`/`svg`/`area_name` at the **top level** of the service
response, not nested under a `raw` key like `export_map`/`_export_mower_map` does. Parsing
it with the `export_map` shape yields empty results and reads exactly like an empty map —
which is how this session initially, and wrongly, "confirmed" the map was still broken.

## Wrap-up 2026-07-24 (later): map-sync root cause identified; the exclusive saga made motion-aware

**Deploy.** `f2074722` (`services.py` + `coordinator.py`) scp'd and md5-matched both sides
(`aa20e913…` / `2307dddf…`). The operator started an HA update mid-session, and that restart
loaded it — no separate restart was needed. Verified live: `get_map_data` returns the new
`map_sync` block; 122 Mammotion entities; API healthy.

### Root cause of the permanent `out_of_sync`: the checksum, and only the checksum

The new `map_sync_diagnostics()` answered it on the first read:

| condition | value |
|---|---|
| `bol_hash_matches` | **False** — reported `8311072749804434520` vs computed `3951449155367542529` |
| `incomplete_area_hashes` | `[]` |
| `area_names_covered` | `True` |
| `area_frame_counts` | 4 areas, 1 frame each |

The map is complete and correctly named; only the hash disagrees. Sharpening it:
`computed_bol_hash` is **not** any of the 24 permutations of the four `map.area` hashes, and
the reported value is not any ordered subset of them either. Since `computed_bol_hash` is
built from `root_hash_lists` (not `map.area` keys), the local root manifest holds a
**different set** than the areas we actually have — extra entries, duplicates, or a stale
manifest.

Root fix deliberately **deferred**: it likely belongs in pymammotion's `is_map_synced()` /
`area_root_hashlist`, and the local consequence is now harmless. Next step when picked up is
to dump `root_hash_lists` on the host and compare it against `map.area`.

*This also supersedes the desk-side guess in the previous section — the earlier note said a
mismatch was "likely but unproven" because `computed_bol_hash` could not be read remotely.
It is now measured, and the specific value rules out the simple "same four areas, different
order" explanation.*

### 🚨 The bigger find: an exclusive saga every 5 minutes, in a queue motion shares

Because `is_map_synced()` is permanently false here, `_async_update_data` enqueued a
`MapFetchSaga` every `REPORT_INTERVAL` (5 min), indefinitely. That is not free:

- `MapFetchSaga` runs at `Priority.EXCLUSIVE` and holds the mower's command queue;
- motion goes out at `Priority.NORMAL` with `skip_if_saga_active=False`
  (`coordinator.async_send_command` → `client.send_command_with_args` →
  `queue.enqueue(..., priority=Priority.NORMAL)`), and `_process` does
  `await self._exclusive_active.wait()` — motion **blocks** behind the saga;
- `_COMMAND_TTL = 120.0` **silently drops** anything undispatched for 2 minutes
  (`EMERGENCY` exempt);
- nothing in the motion path consulted `is_saga_active` — its only caller was the
  `map_sync_status` sensor label.

So a saga landing mid-run stalls pulses and collapses the 200 ms refresh cadence, which makes
the mower self-halt. **Candidate explanation for the still-open 07-18 rotation decay** (9.5°,
9.4°, 5.7°, 0.001°, 0.49°; "location/moment-specific, not parameter-driven", 90 min after a
clean run at identical params). **Explicitly a candidate, not a diagnosis** — recorded as a
hypothesis to test, not as the cause.

**Checked and rejected:** the theory that a saga also freezes the report stream (which would
have explained the 07-19 mid-run feed freeze). `queue.on_saga_start` is wired to a **no-op**
in this pymammotion version (`device/handle.py:293`); poll items use
`skip_if_saga_active=True` instead. Not a supported explanation.

**Fixes** — both behind one testable predicate, `coordinator._should_start_map_sync()`:

1. **Back-off.** A repeat attempt against the same `bol_hash` waits out `MAP_INTERVAL`
   (60 min) rather than retrying every 5. A *changed* `bol_hash` syncs immediately, so a real
   device-side map edit is never delayed. Uses `last_map_sync` — which already existed
   (`coordinator.py:184`) and was **never read for gating**, only surfaced as a sensor — plus
   a new `last_map_sync_bol_hash`. Both stamped through `_record_map_sync_attempt()`, which
   also fixes a smaller gap: the per-tick path called `manager.start_map_sync()` directly and
   never stamped `last_map_sync`, so the sensor under-reported syncs.
2. **Motion-aware.** No saga starts while a guarded motion run holds the mower. The claim
   moved from the `_ACTIVE_MANUAL_MOTION_RUNS` module dict (`services.py`) onto
   `coordinator.manual_motion_owner`, because `services` imports `coordinator` and the flag
   must be readable from both sides. Atomic check-and-set (no `await` between read and write)
   and release-on-every-exit-path, including cancellation, are preserved.

Per the decision taken: **no new motion gate.** Refusing a command the operator just issued
is worse than the wait, and 1 + 2 make a mid-run saga rare.

**Tests.** 341 pass (was 336), mypy + ruff clean on touched files. `_async_update_data` needs
a full HA instance, so the logic lives in `_should_start_map_sync` (4 unit tests: first
attempt, back-off vs elapsed `MAP_INTERVAL`, changed-hash-syncs-now, yields-to-motion) and the
wiring is pinned by an AST test asserting the `start_map_sync` call inside `_async_update_data`
is guarded — mirroring how `_async_opportunistic_ble_reconnect` was extracted for the same
reason. Both the AST test and the motion-aware clause were verified to fail when reverted.

## Correction 2026-07-24 (same session): the "saga every 5 minutes" claim was wrong

The section immediately above claimed the exclusive `MapFetchSaga` fired every `REPORT_INTERVAL`
(5 min) indefinitely, and floated it as a candidate for the open 07-18 rotation decay. **Both
claims are withdrawn.** They came from reading the call site without checking how often it is
actually reached, and the post-deploy verification refuted them within minutes.

**The evidence.** `sensor.<mower>_map_sync_status` reports `syncing` while a saga holds the
queue, so its history is a direct log of saga activity. Over 2026-07-22 → 07-25 it shows only
**5 `syncing` episodes**, each **~8–12 s**, all clustered around restarts/reloads — and **zero**
during the ~25 h of continuous `out_of_sync` since 07-24 01:40. The sensor also reached `synced`
several times, so `is_map_synced()` has not been permanently false historically either.

**The mechanism.** `MammotionReportUpdateCoordinator._async_update_data` opens with

```python
if data := await super()._async_update_data():
    return data
```

and the base `_async_update_data` ends with `return self.data`. `MowingDevice` defines neither
`__bool__` nor `__len__`, so it is **always truthy** — the early return fires on every healthy
tick, and the RTK/dock + bol-hash/map-sync block after it is **unreachable in steady state**.
It runs roughly once per HA start.

**What this changes.**

- The 5-minute churn does not occur; `_should_start_map_sync`'s back-off guards a currently
  theoretical problem.
- Motion-contention exposure is far smaller than stated: ~5 sagas in 3 days at ~10 s each,
  rather than one every 5 minutes. The mechanism itself (exclusive slot, `Priority.NORMAL`
  waiting, 120 s TTL drop) is code-verified and unchanged.
- **The 07-18 rotation-decay hypothesis is withdrawn.** At that rate, a saga landing inside a
  specific ~2-minute run window is improbable, and no evidence places one there. That failure
  remains open and unexplained.
- Both new guards are therefore **no-ops in practice today**. They are kept as defence in depth,
  and become load-bearing the moment the call site is made reachable.

**The genuine bug this uncovered, still open — and it points the other way.** Because that block
is unreachable, a device-side map edit (a changed `bol_hash`) is **never picked up while HA is
running**; the map only re-syncs on restart. `_map_callback`'s comment — "Map freshness is
enforced in `_async_update_data()` via bol_hash checks" — does not hold. Fixing it would make
the block execute on every tick, which is precisely the condition under which the back-off is
required, so the two belong in one change: make the block reachable *and* keep it rate-limited
and motion-aware.

**Method lesson.** The deploy-time verification is what caught this, and only because it looked
for the effect (`syncing` episodes in sensor history) rather than re-reading the code. The first
attempt at verification — grepping `docker logs` for `enqueuing MapFetchSaga` — returned 0 and
was *worthless*, because pymammotion DEBUG logging is off on this host (0 debug lines in 20 min);
a zero there means nothing. Prefer a signal the system records independently of log level.

## Wrap-up 2026-07-24/25: the guard was on the wrong call site; map-sync converges on demand

Follow-up to the correction above, prompted by challenging the claim that the saga guard was
"no longer needed". It is needed — it was simply installed where nothing happens.

### Three saga entry points; the guard covered only the dead one

| Entry point | Guarded before | Actually fires? |
|---|---|---|
| `coordinator.py` per-tick bol_hash check | yes (`_should_start_map_sync`) | **no** — unreachable in steady state |
| `async_sync_maps()` ← `button.<mower>_sync_maps` (`button.py:109`) | **no** | **yes**, whenever pressed |
| `async_sync_maps()` ← `async_force_map_resync` | **no** | **yes**, on demand |

The live history made the point: `button.back_yard_clip_skywalker_sync_maps` was last pressed
`2026-07-22T23:49:09`, and `map_sync_status` went `out_of_sync → syncing` at `23:49:10`. The
five saga episodes in three days were **operator-triggered, not background churn**.

This inverts the risk picture rather than shrinking it. A background timer was never the
threat; a dashboard button pressable at any moment — including mid-run, while diagnosing the
kind of stalled run we have been chasing — is far more plausible, and it was entirely
unguarded. A deliberate press on 07-25 held the queue **12–17 s**, which is a real stall
window for a refreshed segment run.

**Fix:** `_raise_if_manual_motion_in_progress()` on `async_sync_maps()`, so all operator paths
inherit it. The button raises `HomeAssistantError` naming the owning service; `force_map_resync`
refuses up front with `error: manual_motion_in_progress` + `busy_owner` and sends nothing (all
of its steps enqueue device commands, so a partway bail is not good enough). `_should_start_map_sync`
stays as documented defence in depth. 343 tests; both new guards verified to fail when removed.

**Not live-tested:** the refusal itself needs a real motion run to hold the claim, which needs
daylight and an operator. The unit tests cover it; the live test confirmed the *trigger* half
(press → saga → 12–17 s exclusive hold), which was the load-bearing empirical claim.

### The map-sync mismatch resolved itself in 17 seconds — `is_map_synced()` is fine

The same button press converged the map completely:

| | before | after |
|---|---|---|
| `computed_bol_hash` | `3951449155367542529` | `8311072749804434520` |
| `bol_hash_matches` | False | **True** |
| `map_sync_status` | `out_of_sync` | **`synced`** |

The saga's on-complete handler restores `root_hash_lists` from the saga result — the documented
convergence fix — so a stale local `root_hash_lists` was the entire cause. **~25 h of
`out_of_sync` cleared by one press.**

**Withdraw** the framing that `is_map_synced()` is "permanently false on this mower" and the
plan's deferred item to fix it upstream in pymammotion. Nothing is wrong with `is_map_synced()`,
`area_root_hashlist`, or the saga. The condition persisted only because **nothing ever ran a
sync automatically** — the unreachable-block bug. That single defect explains the whole chain:
stale `root_hash_lists` → `bol_hash` mismatch → `out_of_sync` forever → the map never
re-projecting after a reload (the original Task-3 symptom, 2026-07-22).

**So the one remaining fix in this area is to make the per-tick block reachable** — at which
point `_should_start_map_sync`'s back-off and motion-awareness become the guards that keep it
from being a contention problem. Make-it-reachable and keep-it-rate-limited are one change, as
already noted.

**Method note.** Each of the three claims corrected in this area (map empty → not empty;
churn every 5 min → operator-triggered; `is_map_synced` broken → fine) came from reading code
and inferring frequency or state, and each was overturned by looking at what the system
actually recorded — service responses, sensor history, a deliberate button press. Code reading
establishes mechanism; it does not establish whether, or how often, a path executes.

## Wrap-up 2026-07-25 (on-mower, daylight): turn refresh proven ~7x; BLE collapsed; two real gaps found

Supervised session. Step 4 (zone_hash pre-flight) passed, Step 5a (turn A/B) succeeded and is
the headline result, Step 5b (segment run) failed twice on transport and remains unachieved.

### Step 4 — the zone_hash fix does not over-block

Off-dock in `Backyard Right`: `pos_type_label AREA_INSIDE`, `zone_hash 1343645155037768237`,
`valid_for_motion true`, `blockers []`. The stricter gate from `f2074722` is validated on
hardware. `position.map_bol_hash` reports separately and `area_name` resolves correctly rather
than the old misleading `"path"`.

### 🏆 Step 5a — refresh is worth ~7x on a properly-powered turn

`vio_turn_probe`, angular 500, 4.0 s, compass flat on the deck as ground truth:

| pulse | refresh | compass | VIO |
|---|---|---|---|
| A | 0 | 170° → 179° (**+9°**) | −8.75° |
| B | 200 (21 re-sends) | 179° → 241° (**+62°**) | −62.92° |

This closes the question left open 2026-07-22 (the turn half of B1 was inconclusive because
`manual_velocity_pulse_test` caps angular at ~202). **Refresh is speed-gated**: nothing at
angular 180, decisive at 500. Consistent with the linear result (~11x).

Notable secondary result: **VIO tracked both turns to within ~1° of compass**, course-over-ground
agreeing too. The standing assumption that VIO blinds on a fast turn did not reproduce in good
daylight at 78–80 tracked features. One session; do not over-generalise, but it suggests the
compass is a check rather than a necessity when the feed is healthy.

Actionable: `heading_tolerance_degrees: 18` exists only because single-shot turning was quantised
into ~8–15° steps. With continuous rotation it should come down substantially. Next code item is
wiring `motion_refresh_interval_ms` into `vio_turn_to_heading` (still single-shot) and re-deriving
the tolerance.

### 🐛 GAP 1 — the turn phase has no stale-feed detector

Step 5b attempt 1 aborted `no_actuation_detected` after two turn pulses that reported
bit-identical `vision_heading` (90.29915121519771) *and* bit-identical `displacement_m`
(0.006754257916307457). **The operator watched the mower move ~4 inches.** Server logs from the
same window show corrupted frames being dropped outright:

```
dropping frame: malformed report data failed deserialization (249 bytes):
  Field "pos_type" of type int has invalid value [76,117,98,97,45,86,83,80,76,86,51,57,55]
```

— ASCII `"Luba-VSPLV397"`, the device name, landing in an int field.

`no_actuation_detected` was built (2026-07-19) for the e-stop case, where the discriminator is
bit-identical heading *and* flat position. It cannot separate that from "the feed froze while the
mower actuated normally", which is what happened here. The linear phase received
`telemetry_stream_stale` for exactly this class of problem on 2026-07-19; **the turn phase never
got the equivalent.**

The discriminating signal was already present in the result and ignored: `heading_went_fresh:
false` with `heading_poll_seconds: 8.01` (the full timeout) on *both* pulses. A live feed goes
fresh; a dead one times out. Fix: when the freshness poll times out, report a stale-feed reason
instead of blaming actuation. This is the fourth instance of the same underlying lesson —
**always require positive evidence the sensor is live before concluding the mower is not moving.**

### ⚠️ GAP 2 / safety event — a stop command that could not be delivered

Attempt 2 sent the calibration pulse (linear 400) and then:

```
stop_result: { attempted: true, ok: false,
  error: "BLEUnavailableError: BLE connect ... in cooldown (120s remaining)",
  duration_ms: 8992.7 }
stop_reason: vio_calibration_failed   (calibration reason: stop_failed_aborting)
```

BLE entered a fresh connect cooldown *during* the stop attempt. The run aborted rather than
continuing to pulse — the hardening behaved exactly as designed — but there was no positive
confirmation the mower halted on command. Position afterwards was unchanged within ~2 mm,
consistent with the documented single-shot self-halt. Keep this as the reference example for why
`stop_failed_aborting` exists.

### The session's dominant problem was the radio, not the code

- 4–5 transport flips to `cloud_aliyun` with repeatedly re-armed 120 s cooldowns.
- `BleakOutOfConnectionSlotsError` while **all proxies were healthy**: `6 scanner(s) registered,
  6 scanning, 6 connectable` but `last advertisement 613s ago`. The mower's radio had gone silent
  — the ~10–13 min idle doze — not a proxy-capacity problem. It bit us repeatedly **because
  diagnostics between commands take longer than the doze window.**
- **`ble_rssi` is not a liveness signal.** It is self-reported by the mower
  (`report_data.connect.ble_rssi`), so it holds a stale value once the mower stops reporting: it
  read a healthy −64 while nothing had heard an advertisement in 10 minutes. Bit-identical rssi
  across polls is the same stale-feed tell as everywhere else in this project.
- **Cloud-routed restart does not work on this setup.** `button.<mower>_restart_mower` →
  `remote_restart` returned HTTP 200, then
  `WARNING [pymammotion.aliyun.cloud_gateway] Error in sending cloud command: 20056 -
  gateway.hsf.invoke.timeout`. Nothing happened. The app-triggered restart works (10–20 s to
  reconnect).
- The click-to-path card caught a full blackout my polling missed:
  `No transport available ... [cloud_aliyun=connected, ble=disconnected] (mqtt_reported_offline=True)`.
- `switch.<mower>_bluetooth` `turn_on` returned **HTTP 500** once and silently failed to apply
  another time. **Verify switch state after toggling; never trust the HTTP response.**

### Smaller findings

- **`toward` is unreliable after a restart**: read 97.06° against a compass 241° (~144° off),
  stable within 0.002° across 6 polls, surviving two restarts (97.0647 → 97.0629 → 94.5699) while
  position x/y re-converged correctly. **It does not affect the executor** — verified by code
  read: calibration derives map heading itself via `atan2(dy, dx)` from a live position delta
  (`services.py:7932`) and mid-drive re-aim uses `vision_heading` + that fresh offset
  (`services.py:8762`). Neither consults `toward`.
- **`max_linear_pulse_ceiling` is not echoed** by the vector executor (echoes `None` while being
  correctly honoured at `services.py:8477`). Multi-segment had this fixed 2026-07-19; vector did
  not. Cosmetic, but it blocks post-run forensics on the parameter that matters most.
- **`max_linear_commands` defaults to 1** — a segment call stops after a single linear pulse
  unless `max_linear_pulse_ceiling` is passed. With refresh covering ~1 m/pulse, always pass a
  ceiling for a multi-metre segment.
- The map emptied again after real motion and recovered via `force_map_resync`, consistent with
  `invalidate_maps()` plus the unreachable-auto-resync bug. The new `map_sync` diagnostic made
  this legible immediately.

### Step 5b is still open

Both attempts died in transport before reaching the linear phase, so **the Task-2 constants
(pulse-geometry ceilings, `min_progress_distance`, cadence) remain un-re-derived hypotheses.**
Retry needs a healthy BLE window: move the mower near a proxy, and fire promptly after a wake
rather than spending the doze window on diagnostics.

## Wrap-up 2026-07-25 (later, off-mower): the BLE instability root-caused by measurement

The operator made BLE the primary investigation. It was treated as one problem; it is
**four**, and three of them were misattributed. Everything below comes from what the
system recorded — HA's own advertisement stream, its scanner path-scoring log, and
DEBUG logs enabled for the occasion — not from re-reading code.

### The instrument that was missing: HA's raw advertisement stream

`sensor.<mower>_ble_rssi` is self-reported by the mower, so it holds a plausible value
after the radio goes quiet — the project already knew that but had no replacement.
HA's `bluetooth/subscribe_advertisements` websocket command is the replacement: an
advertisement is proof the radio was on air at that instant, observed by HA's own
scanners. Script: `ble_advert_monitor.py` (scratchpad), pairs with `state_sampler.py`.

**Positive control first** (a zero proves nothing unless the instrument is shown to
emit): in 45 s the stream delivered **444 advertisements from 107 distinct devices**,
and the mower appeared in **none** of them.

### 🚨 Finding 1 — the mower barely advertises. This is the root cause.

Measured over a **30-minute window while the mower was actively mowing**
(`MODE_WORKING`, `active_transport: cloud_aliyun`):

| | |
|---|---|
| advertisements heard from the mower | **2** (a single burst at 18:56:55 and :56) |
| scanners that heard that burst | atom-fireplace (−97), p1s-printer-a5774c (−76) |
| advertisements in the other ~29 minutes | **0** |

A normal connectable BLE peripheral advertises every 20 ms–1.28 s while disconnected.
This one emits roughly **one burst per ~10 minutes**. That single fact produces every
`BleakOutOfConnectionSlotsError ... unknown (never seen by any scanner)` and every
`last advertisement 613s ago` in the logs.

It is **not** proxy capacity: 6 scanners registered, 6 scanning, 6 connectable, and
four of them report `slots=3/3 free` at the moment of failure. It is **not** the
documented −70/−76 RSSI wall: whenever the mower *is* heard, it is heard at −62 to −69.

It is also **not only** the idle doze as documented — this window was a mower actively
working, not an idle one dozing.

### Finding 2 — two failure modes were being conflated

| mode | signature | what it means | count in a 2.5 h log |
|---|---|---|---|
| **A** | `unknown (never seen by any scanner)` / `only in non-connectable history` — **no proxy connect is attempted at all** | the mower is not advertising | 6 of 8 |
| **B** | a proxy **does** attempt: `Connecting v3 with cache` → ~20 s of nothing → `ESP_GATTC_OPEN_EVT in DISCONNECTING state (status=133)` | the mower was advertising at −64 and still refused/timed out the connection | 2 |

Both end identically — `BLETransport[...]: cooling down for 120s` — which is why they
read as one problem. They need opposite fixes. Mode B at −64 dBm flatly refutes a
coverage explanation for those events.

### Finding 3 — HA keeps routing connects to the proxy that just failed

From the scanner path-scoring lines, the failure penalty is negligible next to RSSI:

```
17:39:03  esphomes3-irk       RSSI=-67  failures=0  score=-67.00
          p1s-printer-a5774c  RSSI=-64  failures=2  score=-67.06   <- 0.06 margin
```

`p1s-printer-a5774c` sits closest to the mower, so it usually wins on RSSI, and both
mode-B failures happened on it. It is also the only proxy that never has a free
allocation (`slots=2/3 free` in **every** sample — one connection permanently in use).
Moving the mower's usual parking spot nearer `esphomes3-irk`, or reducing what the P1S
proxy carries, is a placement change worth trying before any code change.

### 🐛 Finding 4 — the corrupted frames are a pymammotion reassembly bug (root-caused)

The 2026-07-25 garbage decodes cleanly as protobuf:

```
0x1a 0x0b "a1LLmy1zc0j"     field 3, len 11  -> the Aliyun product key
0x22 0x0d "Luba-VSPLV397"   field 4, len 13  -> the device name
```

That is a **device-identity message spliced into a report frame** — meaningful bytes
from another message, not bit-flips on the wire. The mechanism:

- `BleMessage.parseNotification` accumulates fragments into `self.notification` via
  `addData` (an appending `BytesIO`).
- `BLETransport._notification_handler` calls `clear_notification()` **only** when
  `parseNotification` returns 0 (a complete frame).
- A lost fragment (`return 1`, still waiting), a checksum failure (`return -4`) or an
  exception (`return -100`) all make the handler return early — **the partial buffer is
  never reset**.
- The next message's fragments append to that stale partial. When its non-fragmented
  terminator arrives, `parseBlufiNotifyData` hands back *old partial + new message*
  concatenated, and `handle.py` drops the whole thing.

`parseNotification` even **detects** the loss and does nothing about it — on a sequence
discontinuity (`ble_message.py:391`) it resyncs the counter and keeps the poisoned
buffer.

**Live confirmation (DEBUG enabled this session):** `parseNotification read sequence
wrong` fired **11 times in ~3 minutes** of connected BLE (19:04–19:07), with gaps of
1–3+ packets (`15 14`, `126 123`, `201 199`, …). So the precondition is common on this
link; poisoning only surfaces when a gap lands mid-fragment, which matches the observed
rarity of `dropping frame` (4 in 2.5 h).

**Impact:** one lost packet mid-fragment costs at least two reports. This is the
mechanism behind the 07-25 turn reporting bit-identical `vision_heading` *and*
bit-identical `displacement_m` while the mower physically turned.

**Fix (upstream, pymammotion):** reset the accumulation buffer whenever the sequence
check detects a gap, and on the `-4`/`-100` paths.

### 🐛 Finding 5 — our own code contributes, and the dead region is bigger than documented

The 2026-07-24 correction established that
`MammotionReportUpdateCoordinator._async_update_data` early-returns on every healthy
tick because `MowingDevice` is always truthy, making the map-sync block unreachable.
**`_async_opportunistic_ble_reconnect()` is in that same dead region** (coordinator.py,
three statements past the early return).

Proven by recorded evidence rather than code reading — with DEBUG enabled:

```
DEBUG [custom_components.mammotion] Finished fetching mammotion data in 0.000 seconds (success: True)
```
appears on every tick, while `Updated Mammotion device` (a `LOGGER.debug` three lines
past the early return) appears **0 times** across every tick observed. "in 0.000
seconds" is itself the tell.

That function was written for exactly this symptom — its docstring cites "live
2026-07-19: repeatedly stuck on `cloud_aliyun` at healthy RSSI with the cooldown long
expired". **It has never run.**

The consequence chains straight into Finding 1: when one of the mower's rare
advertisements finally lands, HA *does* immediately push a fresh `BLEDevice`
(`poll_debouncer` → `_add_ble_device` → `set_ble_device`, `immediate=True`) — but
**nothing then calls `connect()`**. BLE returns only when something else happens to
send a BLE command. Observed this session: advertisement burst at **18:56:56**,
`active_transport` back to `ble` at **19:04:47** — roughly 8 minutes of usable link
thrown away.

**Checked and cleared:** the advertisement callback registration itself is fine.
`_async_start()` registers it properly; the copy inside the dead region is a redundant
backup, not the live path. (Suspected it, checked it, it was wrong — recorded so the
next session does not re-suspect it.)

This makes off-mower item 4 (make the per-tick block reachable) the highest-value BLE
fix available, and it is now load-bearing for two subsystems, not one.

### Method notes

- The 30-minute advertisement window is one sample of one mower on one evening. It is
  strong evidence for "advertises rarely", not proof of the duty cycle. Re-run it while
  the mower is **docked and idle** before treating ~10 minutes as the number.
- `logger.set_level` changes are runtime-only and revert on an HA restart; nothing was
  persisted.

### Code shipped this session (off-mower, NOT deployed) — 346 tests, mypy + ruff clean

**1. Turn-phase stale-feed detector (`vio_telemetry_stream_stale`).**

The handoff prescribed "when the freshness poll times out, emit a stale-feed reason
instead of `no_actuation_detected`". **That does not work, and the reason matters:**
`heading_went_fresh` is True only when before/after differ by more than the epsilon,
which is exactly when `_streak_shows_no_actuation` (bit-identical heading) is False.
The two are **perfectly correlated**, so gating on it would have deleted the
no-actuation branch rather than refined it.

The signal that actually discriminates is *"did any channel move at all"*. A live feed
is never perfectly still: position jitters ~2–4 mm between reads on a stationary mower,
and a VIO heading latched by dusk still emits sub-epsilon noise (~0.0018°, run 2). The
poll loop now records `heading_poll_count` and `heading_poll_feed_alive` per pulse, and
`_streak_shows_dead_telemetry` fires only when heading **and** position were
bit-identical across every poll of the streak (≥ `_STALE_FEED_MIN_POLLS`).

Ordering: dead-telemetry → no-actuation → no-heading-progress.

Verified to fail with the fix reverted (reproduces the exact 07-25 misdiagnosis). The
two dusk-latch tests **pass untouched**, which is the real check that the discriminator
is right rather than fitted to the new test.

**Consequence worth knowing:** a replay of the 2026-07-19 e-stop run now reports
`vio_telemetry_stream_stale`, not `no_actuation_detected` — because that run's feed was
frozen too (heading bit-identical for 45 min). That is the honest answer: telemetry
never saw the e-stop, the operator did. `no_actuation_detected` now means what it says
— the link was demonstrably alive and the mower still did not move.

**2. `motion_refresh_interval_ms` wired into `vio_turn_to_heading`.**

Mirrors the linear phase via the existing `_motion_refresh_window`; refreshes are
counted in `motion_refresh_commands_sent` so they never inflate `commands_sent` (which
drives `max_commands`). Schema + services.yaml + handler wired.

**Deliberately left defaulting to 0.** Refresh gave ~7x at angular 500 on 2026-07-25,
but `heading_tolerance_degrees: 18` exists only because single-shot turning was
quantised into ~8–15° steps. Turning refresh on by default before re-deriving the
tolerance would drive continuous rotation into a deadband sized for discrete steps.
**Re-deriving that tolerance needs a mower session** — it cannot be done at the desk.

**3. `max_linear_pulse_ceiling` (and siblings) now echoed by the vector executor.**

Also `turn_pulse_duration_ms`, `linear_pulse_duration_ms`, `vio_turn_max_commands`,
`vio_angular_speed`, `vio_heading_offset_degrees` — matching the multi-segment executor
(fixed there 2026-07-19, missed here). Regression test asserts all six on a dry run.

**Not done: off-mower item 4** (make the per-tick coordinator block reachable). Finding
5 above changes its scope — it now governs the BLE reconnect as well as map sync, and
making it reachable turns on two behaviours at once on every 5-minute tick. That is an
operator-visible behaviour change and wants a deliberate decision, not a drive-by.

### Diagnostic tooling added (`scripts/`)

- **`ble_advert_monitor.py`** — subscribes to HA's raw advertisement stream and
  records every advertisement heard from the mower (RSSI, scanner, connectable),
  then summarises silent gaps and per-scanner coverage. Prints a **CONTROL** count
  of advertisements from all devices, so a zero for the mower can never be read as
  a result until the instrument is shown to be emitting. Read-only.
  `.venv/bin/python scripts/ble_advert_monitor.py 1800`
- **`state_sampler.py`** — samples the mower's self-reported transport/rssi/mode
  on a cadence, recording each entity's `last_updated` alongside its value so a
  frozen timestamp beside a healthy-looking number is visible at a glance.

Sample taken while writing this up: **1 mower advertisement vs 311 from all
devices in 60 seconds.**

To re-enable the DEBUG logging this investigation used (runtime only, reverts on
restart; it was set back to `info`/`warning` afterwards):

```yaml
service: logger.set_level
data:
  pymammotion.bluetooth.ble_message: debug   # parseNotification sequence gaps
  pymammotion.transport.ble: debug
  custom_components.mammotion: debug         # coordinator dead-region check
  pymammotion.device.handle: debug
```

## Wrap-up 2026-07-25 (later still): item 4 — the dead region made reachable

Off-mower item 4 done. Scope turned out **larger than documented**, in two ways.

### The map-sync block is not where the notes said it was

Previous notes (and `_should_start_map_sync`'s own docstring) placed the per-tick map
sync in `MammotionReportUpdateCoordinator._async_update_data`. It is actually in
**`MammotionMapUpdateCoordinator._async_update_data`**, which runs on `MAP_INTERVAL`
(60 min), not `REPORT_INTERVAL` (5 min). Docstring corrected.

### Five coordinators had the bug, not one

`if data := await super()._async_update_data(): return data` appears in **five**
subclasses — report, maintenance, version, map, and error — and the base's terminal
`return self.data` was always truthy in every one of them. So five dead regions, each
running only once per HA start:

| coordinator | interval | what was unreachable | guarded on its own? |
|---|---|---|---|
| report | 5 min | `async_save_data`, backup BLE callback registration, **`_async_opportunistic_ble_reconnect`** | yes — `is_usable` / not connected / timeout |
| maintenance | 30 min | returns fresher `report_data.maintenance` | n/a, pure read |
| version | 30 min | up to 4 device-info commands, `check_firmware_version()`, the OTA-firmware HTTP fetch, **and its own `update_interval` self-throttle** | yes — each command skipped once `already_set` |
| map | 60 min | RTK/dock fetch + **map-sync saga** | yes — `is_map_synced` + `_should_start_map_sync` back-off |
| error | 30 min | HTTP error-code fetch | yes — skipped once populated |

Every region was already individually guarded, which is why turning them on is a
contained change: the guards were written, they simply never got a chance to run.

### The fix: an explicit contract instead of truthiness

The base method is renamed **`_async_short_circuit_update() -> DataT | None`**. It
returns the data to publish when the update must stop (device gone, disabled, offline,
mid-map-edit, failing repeatedly) and **`None`** when the caller should carry on. All
five call sites now read:

```python
if (data := await self._async_short_circuit_update()) is not None:
    return data
```

`is not None` rather than truthiness is the point — truthiness is what hid this for
months, and it would equally misread a legitimately falsy payload as "carry on". The
rename also stops the base overriding HA's `DataUpdateCoordinator._async_update_data`
with a widened return type; every subclass overrides it anyway, so the base never
needed one.

**6 new tests**, including one verified to fail against the old terminal return, and an
AST test pinning `is not None` at all five call sites so the regression cannot be
reintroduced. 352 tests, mypy + ruff clean.

### What this changes at runtime — read before deploying

- **BLE reconnect now attempts every 5 minutes** instead of once per HA start. This is
  the fix for the finding above (a rare advertisement lands, HA caches a fresh
  `BLEDevice`, and nothing ever calls `connect()`). It is bounded: it no-ops unless
  `prefer_ble` and `is_usable` and not already connected, and it is wrapped in a
  timeout so a slow connect cannot stall the tick.
- **The map-sync check now runs every 60 minutes.** `_should_start_map_sync` becomes
  load-bearing exactly as predicted: a repeat attempt against an unchanged `bol_hash`
  waits out `MAP_INTERVAL`, and no saga starts while a guarded motion run holds the
  mower. This is what finally fixes "a device-side map edit is never picked up until an
  HA restart".
- Version/error/maintenance regions are no-ops once their data is populated.

**Hardware confirmation to take after the next deploy:**
1. `sensor.<mower>_active_transport` should stop sitting on `cloud_aliyun` for long
   stretches while the mower is advertising — compare against a
   `scripts/ble_advert_monitor.py` run over the same window.
2. Edit an area on the mower and confirm `map_sync_status` converges **without** an HA
   restart or a `sync_maps` press.
3. Confirm `sensor.<mower>_last_map_sync` advances at most once per hour, not per tick.

### Second advertisement sample: docked and idle — the duty cycle is not a mowing artefact

The first sample was taken while the mower was working, so the obvious objection was
that mowing (distance, motor noise, power management) explained the silence. Repeated
with the mower **docked and `MODE_READY`**:

| | |
|---|---|
| window | ~20 min (the websocket dropped before the full 25) |
| mower advertisements | **4, in exactly 2 bursts** — 20:11:30 and 20:16:52 |
| silent before the first burst | ~13 min |
| RSSI when heard | −84 (garage-m5stack), then −68 (esphomes3-irk) and **−50** (p1s-printer) |

**Same behaviour docked as mowing: roughly one burst every 5–10 minutes.** So the sparse
duty cycle is a property of the mower, not of mowing — which is the confirmation the
first sample could not provide on its own.

Note the −50 reading: when this mower does advertise, at least one proxy hears it
*loudly*. Nothing about the link quality is marginal. The problem is purely **how rarely
the radio is on air**.

Also worth recording: HA's scanners observed −84 in the first burst while
`sensor.<mower>_ble_rssi` was self-reporting **−52** at the same time. So that sensor is
not a usable proxy for link quality either, not merely for liveness.

**Script hardened as a result:** the first long run died partway through on an aiohttp
heartbeat timeout (`No PONG received after 15.0s`) and lost its summary — the one
failure mode that matters for a measurement whose whole point is "how long was it
silent". `ble_advert_monitor.py` now reconnects on a dropped socket and always emits the
summary from a `finally` block.

## 🚨 2026-07-25 (config): `prefer_ble_over_wifi` is stored **false** — and it gates the item-4 fix

Read from `/config/.storage/core.config_entries`, not inferred from the options form
(whose default is `True`, so an unchecked box means the key was explicitly set `False`):

```json
{"movement_use_wifi": false, "mow_path_fetch_enabled": false, "prefer_ble_over_wifi": false}
```

### What `prefer_ble` actually controls — three sites, and they differ

| site | uses | effect today (`false`) |
|---|---|---|
| `active_transport()` (handle.py:1883) | `self._prefer_ble` | **none.** Its own docstring: prefer_ble "no longer affects which transport is returned (a connected BLE always wins; otherwise a usable MQTT wins)" — it only biases a log de-dup key. This is why `active_transport: ble` still appeared for long stretches. |
| `_do_send()` (handle.py:772) | `self._prefer_ble`, **no override** | **reconnect-on-send disabled.** A command sent while BLE is disconnected-but-usable never schedules a BLE reconnect. |
| `send_raw()` (handle.py:1668) | `self.prefer_ble if prefer_ble is None else prefer_ble` | **caller can override.** Our motion services pass `prefer_ble=True` explicitly, so they *do* trigger a reconnect. |

That last row explains the observed pattern precisely: BLE recovers when something
explicitly asks for it (a motion service, a probe) but not from routine background
traffic — which is why an advertisement at 18:56:56 was only followed by
`active_transport: ble` at 19:04:47.

### The coupling that matters

There are exactly two *automatic* BLE-reconnect paths, and **both are disabled right
now**:

1. pymammotion's reconnect-on-send — off because `prefer_ble=False`;
2. our `_async_opportunistic_ble_reconnect` — dead code (the region bug fixed today).

They are not independent. Our function **also** short-circuits on the same flag:

```python
if not (handle.prefer_ble and ble.is_usable and not ble.is_connected and self._bluetooth_enabled):
    return
```

**So the item-4 fix on its own would have changed nothing on this system.** Enabling
`prefer_ble_over_wifi` is required for it to do anything. Recorded because the previous
section could otherwise be read as "the fix closes the gap" — it does not, unaided.

### Recommendation on the other two options

- **`movement_use_wifi` — leave OFF.** `services.py` never reads it (verified: zero
  references); only `button.py` does, where it makes `_nudge_available` return `True`
  unconditionally and routes the nudge buttons over cloud. Cloud-routed motion is what
  the safety model refuses — the position feed is BLE-only and stone dead on cloud, so
  it is driving blind. It would also mask the "BLE selected but not usable" signal that
  exposed the 2026-07-19 gate bug.
- **`mow_path_fetch_enabled` — no BLE effect.** It gates `MowPathSaga` over MQTT only.
  Neutral; adds cloud command-queue traffic.

### Expected side effect of enabling prefer_ble

More connect attempts and more cooldown churn in the log: `is_usable` can be `True` on a
cached-but-stale `BLEDevice` (pymammotion deliberately keeps it when tripping a
cooldown), so attempts will fire against a mower that is not currently advertising and
fail into the 120 s cooldown. That is the intended bounded-retry design, and it is the
mechanism by which BLE gets re-established — but expect the log to look busier, not
quieter.

Saving options fires `_async_update_listener` → `async_reload(entry)`, i.e. an
integration reload (not an HA restart). Do it while the mower is **awake**, per the known
blank-card-map gotcha.

### First 25 minutes after `prefer_ble_over_wifi: true` (saved 20:46:13 local)

Sampled every 20 s (`scripts/state_sampler.py`, 75 samples, 20:46:59 → 21:11:40), mower
`MODE_WORKING` throughout — i.e. mowing, so its distance from the proxies changed across
the window. That is a confound; read this as encouraging, not as proof.

| period | transport | `ble_rssi` |
|---|---|---|
| 20:46:59 → 21:02:00 | `cloud_aliyun` | `0` (no link) |
| 21:02:00 → 21:11:20 | **`ble`**, one 20 s cloud blip | live every sample, **−60 degrading to −90** |
| 21:11:20 → end | `cloud_aliyun` | back to `0` |

What the server log shows in the same window:

```
21:01:54  P1S Printer: Connecting v3 with cache -> Connection open      (2s)
21:04:37  P1S Printer: Connecting v3 with cache -> Connection open      (16s)
21:06:01  BLETransport: device Luba-VSPLV397 disconnected
21:06:01  Bluetooth Proxy: Connecting v3 without cache
21:06:03  Bluetooth Proxy: Connection open + Service discovery complete (2s, DIFFERENT proxy)
21:11:02  BLETransport: out of connection slots / device unreachable — cooling down 120s
```

Three successful connects, a **2-second hand-off to a different proxy** on disconnect,
and a ~9-minute continuous BLE hold. During that hold `ble_rssi` updated on essentially
every sample and degraded monotonically −60 → −90 as the mower drove away — a *live*
feed, in direct contrast to the frozen values recorded earlier in the evening.

**What this does and does not establish.** The window contains **no `status=133`
failures and no "never seen by any scanner"** — the eventual drop at 21:11 is
`device unreachable` after the RSSI decayed to −90, i.e. the mower simply mowed out of
range. That is the *ordinary* failure mode, not the pathological ones that dominated
earlier. But the mower's position changed throughout, there is no control window under
matched conditions, and immediate reconnect-on-disconnect was already observed before
the change (motion services pass `prefer_ble=True` explicitly, so they always had it).

**So: consistent with the flag helping, not attributable to it yet.** The clean test is
a docked-and-idle window compared against the docked-and-idle sample taken earlier
tonight (4 adverts / 2 bursts / ~20 min, transport stuck on cloud at −52 self-reported).
Run that before crediting the change.

### Docked-and-idle control, `prefer_ble=true` (21:43 → 22:08) — and a ceiling on what the fix can buy

The matched control for the earlier docked sample. Mower `MODE_READY` on the dock,
`ble_rssi` self-reporting −50, `active_transport: cloud_aliyun`.

| | earlier docked (`prefer_ble=false`) | this run (`prefer_ble=true`) |
|---|---|---|
| window | ~20 min | 25 min |
| mower advertisements | 4 (2 bursts) | **1** |
| RSSI when heard | −84 / −68 / −50 | **−50** |
| control (all devices) | — | **9,393** |

**The flag did not change the advertising rate**, which is exactly right: advertising is
device-side and nothing in HA can influence it. That part of the model holds.

**But this is the cleanest statement of the root cause so far: one advertisement in
25 minutes, at −50 dBm, sitting on its own dock.** Signal quality is superb; the radio is
simply almost never on air. Ratio to the rest of the house: 1 : 9,393.

#### ⚠️ What this implies for the deploy — read before expecting a transformation

A central can only open a BLE connection **in response to a connectable advertisement**.
So the advertisement rate is a hard ceiling on when BLE can be (re)established, and
neither `prefer_ble=true` nor the item-4 reconnect fix can raise it. Concretely:

- `_async_opportunistic_ble_reconnect` will now run every 5 minutes, but it no-ops
  unless `is_usable` — and most of those ticks will land in a silent window.
- pymammotion deliberately *keeps* the cached `BLEDevice` when it trips a cooldown, so
  `is_usable` can be True against a mower that is not currently advertising. Those
  attempts will fail and re-arm the 120 s cooldown. **Expect a busier log with a
  significant failure fraction — that is the design working, not a regression.**

So the honest expectation for the deploy is **"BLE is re-established promptly whenever
the mower gives us a chance", not "BLE is continuously available"**. The remaining gap is
device-side and not addressable from this integration.

#### Open question, deliberately not answered

Advertising rate does **not** track docked-vs-mowing cleanly:

| window | state | adverts |
|---|---|---|
| 18:47 → 19:17 | mowing | 0 in 30 min |
| ~20:00 → 20:20 | docked idle | 4 in 20 min |
| 21:01 → 21:11 | mowing | 3 successful *connects*, so adverts were present |
| 21:43 → 22:08 | docked idle | 1 in 25 min |

(Note a connected peripheral normally stops advertising, so the 21:02–21:11 BLE hold
would itself show no adverts — the connects at 21:01:54, 21:04:37 and 21:06:01 are the
evidence that adverts were available then.)

No clean dependence on activity state, and the earlier "~10–13 min idle doze" framing
does not survive as a general rule either — it is sparser and more irregular than that.
Do not commit to a mechanism yet; it needs more windows under recorded conditions.

## ⚠️ REFINEMENT (same night, docked): connecting is not the bottleneck — **sessions die in ~73 s**

The "the mower barely advertises" headline above is correct as a measurement but it is
**not the whole diagnosis**, and this refines it. Correlating ESPHome connect events
against `sensor.<mower>_active_transport` while the mower sat **docked and idle**
(`MODE_READY` from 21:16:53):

| BLE session | ended | held |
|---|---|---|
| 21:16:53 | 21:18:06 | **73 s** |
| 21:46:53 | 21:48:07 | **74 s** |
| 22:01:56 | 22:06:56 | 300 s |
| 22:16:53 | (ongoing at time of writing) | — |

And the ESPHome side shows `Connection open` for the mower at **21:01:54, 21:16:52,
21:46:52, 22:01:54, 22:16:52** — i.e. roughly a **15-minute grid**, with almost no
matching `BLETransport: device ... disconnected` lines.

**So HA does manage to connect regularly — about every 15 minutes — and the link then
dies after ~73 seconds.** That is a materially different problem from "we can never get
a connection", and a more actionable one: pymammotion runs a `todev_ble_sync(2)`
heartbeat every **5 s** (`ble_loop.py::_KEEP_ALIVE_BLE_INTERVAL`) precisely to hold the
GATT link open, with a 30-failure (~150 s) tolerance before giving up. A link that dies
in 73 s with that heartbeat running is a real defect, not a radio limitation.

**What this does and does not change:**

- The advertisement measurements stand — 1 advert / 25 min docked, control 9,393. That
  is still extraordinary and still device-side.
- But because a *held* connection needs no further advertisements, sparse advertising
  only gates the **initial** connect after a drop. If sessions survived, one advert per
  25 minutes would be plenty.
- **Therefore the dominant docked-state problem is session lifetime, not advertisement
  rate.** Both are real; the ordering was wrong.

Note the 73 s / 74 s figures are suspiciously repeatable and match none of the obvious
constants (`_KEEP_ALIVE_BLE_INTERVAL` 5 s, `_BLE_HEARTBEAT_FAIL_LIMIT` 30 → ~150 s,
`_BLE_STREAM_STALE_THRESHOLD` 15 s, the docked poll intervals 60 s / 300 s). The 300 s
session does match `_BLE_POLL_INTERVAL[DOCKED_FULL/IDLE]`. Do **not** guess a mechanism
from these numbers — DEBUG logging for `pymammotion.transport.ble`,
`pymammotion.device.handle` and `pymammotion.device.ble_loop` was enabled at ~22:20 to
capture the next cycle directly.

**Next session: this is the thread to pull.** It is fully answerable with the mower on
the dock, needs no motion, and if sessions can be made to persist it removes most of the
BLE pain regardless of the advertising rate.

## Deploy 2026-07-25 (late): 3 files live — and a restart-time hazard found

`coordinator.py`, `services.py`, `services.yaml` scp'd to
`/config/custom_components/mammotion/` and **md5-matched on both sides**
(`a0c352ab` / `5373e70c` / `022443be`). The other 41 integration files and the card were
already byte-identical. Manifest reads `0.6.4-beta11` on both sides — unchanged by this
work, and therefore useless as a deploy indicator, exactly as the repo gotcha warns.

HA Core restarted: API back in 61 s, 122 mammotion entities at 190 s. 53 mammotion
services registered.

### 🐛 The restart left the integration DEAD, and it did not self-recover

```
22:29:42 ERROR [homeassistant.config_entries] Error setting up entry ... for mammotion
  aioesphomeapi.core.TimeoutAPIError: Timeout waiting for BluetoothGATTNotifyResponse,
  BluetoothGATTErrorResponse, BluetoothDeviceConnectionResponse after 10.0s
  ...
  File "/config/custom_components/mammotion/__init__.py", line 502, in async_setup_entry
    await _await_device_connection(...)
  File "/config/custom_components/mammotion/__init__.py", line 332, in _await_device_connection
    await handle.connect_transport(TransportType.BLE)
22:29:52 WARNING [homeassistant.config_entries] Unloading matt.joslin@me.com (mammotion)
```

Result: config entry state **`setup_error`**, 121 of 122 entities `unavailable`, and
**no automatic retry** — HA only auto-retries `setup_retry` (i.e. `ConfigEntryNotReady`).
A raw `TimeoutAPIError` propagating out of `async_setup_entry` yields `setup_error`,
which is terminal until someone reloads by hand. A manual entry reload fixed it
immediately (state `loaded`, 122 entities, 17 unavailable — the normal BLE-gated ones).

**This is not caused by the deployed change** — `__init__.py` was byte-identical before
and after, and the failure is a GATT `start_notify` timeout through the ESPHome proxy.
It is the BLE flakiness biting at the worst possible moment.

**But it is a real robustness bug, and a nasty one given this hardware:** `_await_device_connection`
attempts a BLE connect during setup, and on a marginal link that timeout takes the whole
integration down permanently. Restarting HA while BLE is unhealthy — which, on the
measurements above, is most of the time — can leave the mower integration dead with no
retry and no notification. The plausible fix is to wrap the setup-time BLE connect so a
failure raises `ConfigEntryNotReady` (retry with backoff) or degrades to cloud, rather
than escaping as a raw transport error. **Not attempted this session; queued.**

### Verification

- 53 services registered; `vio_turn_to_heading`, `raw_pymammotion_execute_vector_segment`
  and `force_map_resync` all present.
- No syntax/import error from the two changed Python files.
- **Dead-region marker:** `Updated Mammotion device` — which appeared **zero** times
  across every tick before the fix — now appears. Confirming it *repeats* across
  successive `REPORT_INTERVAL` ticks (rather than the old once-per-start) is the actual
  proof; a watcher is running for the third occurrence.

### ✅ The dead-region fix is PROVEN LIVE ON HARDWARE

`Updated Mammotion device` — the `LOGGER.debug` three statements past the early return,
which appeared **zero** times across every tick before the fix:

```
2026-07-25 22:32:28  Updated Mammotion device Luba-VSPLV397
2026-07-25 22:38:04  Updated Mammotion device Luba-VSPLV397
2026-07-25 22:43:19  Updated Mammotion device Luba-VSPLV397
```

Three occurrences ~5 min apart, matching `REPORT_INTERVAL`. The first alone proved
nothing (that is the once-per-HA-start case which happened before the fix too); the
**repetition** is the proof. So as of this deploy:

- `_async_opportunistic_ble_reconnect()` runs every ~5 minutes instead of never;
- `async_save_data()` runs per tick;
- the map coordinator's `bol_hash` / map-sync block runs on its 60-minute tick, with
  `_should_start_map_sync`'s back-off now load-bearing.

Also verified on hardware by dry run (no motion):

- `vio_turn_to_heading` accepts and echoes `motion_refresh_interval_ms: 200` and reports
  `motion_refresh_commands_sent` — HTTP 200 where the old schema rejected the key.
- The vector executor echoes `max_linear_pulse_ceiling: 12` (previously `None`) plus
  `turn_pulse_duration_ms`, `linear_pulse_duration_ms`, `vio_turn_max_commands`,
  `vio_angular_speed`.

**Remaining post-deploy checks that need wall-clock time:** `sensor.<mower>_last_map_sync`
should advance at most hourly, and a device-side map edit should converge without a
restart or a `sync_maps` press.

DEBUG logging was set back to `info` afterwards. To re-arm the BLE session-death capture
(the next session's top thread) — note these reset on every HA restart:

```yaml
service: logger.set_level
data:
  pymammotion.transport.ble: debug
  pymammotion.device.handle: debug
  pymammotion.device.ble_loop: debug
```

## ❓→❌ "Can we drive over Wi-Fi instead of BLE?" — No. The app doesn't either.

Raised 2026-07-26: the mower moves from the app while it is on Wi-Fi, so perhaps BLE is
only needed for blades-on manual mowing. This would be a way around the BLE wall, so it
was checked against the decompile rather than reasoned about. **It does not hold.**

### The app has exactly two links, and the driving command is hard-wired to BLE

`MALinkManagerAPI` owns precisely two managers — `espBleManager` (BLE) and `maIotManager`
(IoT/cloud) — with `LinkType`, `trySwitchToIOT` and `_trySwitchToBT`. There is **no
Wi-Fi/LAN link manager** in `device/source/links/managers/`.

Two delivery paths hang off it, and they are not symmetric:

```java
// MALinkManager.java:410 — binary commands. Unconditionally BLE.
public void postCustomeDateByte(byte[] bArr, String str) {
    EspBleManagerApi espBleManagerApi = this.espBleManager;
    if (espBleManagerApi == null || bArr == null) return;
    espBleManagerApi.postCustomeDateByte(bArr, str);      // <- no IoT branch at all
}

// MALinkManager.java:419 — JSON. This is the cloud path.
public void postJsonString(String str, String str2, boolean z2, String str3) {
    IotManagerApi iotManagerApi = this.maIotManager;      // <- IoT
    ...
}
```

`DrvMotionCtrl` passes transport flag `false`, which routes to `postCustomeDateByte`
(`MACommandHelper.java:217-226,229-256`) — i.e. **straight to `EspBleManager`**. So the
app's joystick drives over **BLE**, and the earlier catalog note that this is "not
intrinsically BLE-only" is too cautious: with only two managers and the binary path
hard-wired to BLE, there is nowhere else for it to go.

### Why it *looks* like Wi-Fi

`MALinkManager` does expose `getAllWifi()`, `getDeviceWifiList()`, `setWiFiOpen/Close()`
— but these are **BluFi Wi-Fi provisioning carried over BLE** (ESP BluFi: you use
Bluetooth to hand the device its Wi-Fi credentials). The mower's Wi-Fi is what gets it to
the **cloud**; it is not a control link for the app. The app's WIFI/BLE indicator showing
Wi-Fi does not mean the joystick is using it.

### The operator's other belief is correct, and stronger than stated

Manual mowing (`DrvMowCtrlByHand`, refreshed every 800 ms) is **Bluetooth-only** via a
caller-level gate, and the blade auto-stops when filtered BLE RSSI ≤ **−80 dBm** — a
safety interlock that only makes sense on a BLE link. So BLE is required for blades-on
mowing *and* for plain driving.

### What our own `use_wifi` / `movement_use_wifi` actually is

Not the same thing. pymammotion has three transports — `ble.py`, `mqtt.py`,
`aliyun_mqtt.py` — and **no local/LAN transport**. `async_stop_manual_motion(use_wifi=True)`
just sets `prefer_ble=False`, which routes over **Aliyun cloud MQTT**: a path the app
deliberately never uses for motion. Even setting aside safety, it is unusable here —
cloud report cadence is 10–60 min (sensors measured 9–15 min stale tonight), the position
feed is dead on cloud (20 of 21 polls bit-identical, 2026-07-21), and the proven 200 ms
refresh cadence cannot survive a cloud round trip.

**Conclusion: there is no Wi-Fi shortcut around the BLE work.** The BLE session-lifetime
thread is not optional — it is the only path to reliable click-to-path.

## 💡 2026-07-26: the comms module is an ESP32 — Wi-Fi/BLE coexistence is the leading hypothesis

Operator supplied the hardware fact: the mower's Wi-Fi and BLE are the same ESP32.
Confirmed in both codebases — the app ships `com.agilexrobotics.espressif.BlufiClientImpl`
/ `BlufiNotifyData` / `FrameCtrlData`, and pymammotion implements the same BluFi framing
(`ble_message.py`, `mBlufiMTU`, `BlufiNotifyData`). BluFi is Espressif's own BLE
protocol, so this is an ESP32 at the mower end.

**This matters because an ESP32 has ONE 2.4 GHz radio shared between Wi-Fi and BLE.**
When both are active the coexistence scheduler time-slices them: BLE advertising events
get delayed or skipped while Wi-Fi has the radio, and an active BLE connection can miss
enough connection events to hit its supervision timeout and drop.

### It explains both open mysteries at once, and the paradox between them

| observation | coexistence explanation |
|---|---|
| 1 advertisement per ~25 min while **docked and idle** | advertising events are being starved by Wi-Fi, not skipped for power saving |
| yet heard at **−50 dBm** when it does advertise | signal was never the problem — radio *time* is |
| sessions connect fine, then die at ~**73 s** | missed connection events under Wi-Fi contention → supervision timeout |
| `ble_rssi` self-reported −50 while scanners heard −84 | consistent with sporadic, poorly-timed transmissions rather than a weak link |

Nothing else proposed so far explains "excellent signal, almost never on air".

### It probably also explains the proxy-side failures

Our ESPHome proxies are **also ESP32s sharing one radio**. The two `status=133`
(`ESP_GATT_ERROR`) connect failures both landed on **`p1s-printer-a5774c`** — a proxy
co-located with a 3D printer, i.e. very likely carrying real Wi-Fi traffic. Same
mechanism, other end of the link. That is a better explanation than "that proxy is
broken", and it fits the observation that it also permanently shows `slots=2/3 free`.

### The test — and its risk

`switch.<mower>_device_wifi` is real: `async_set_device_wifi_enabled` sends
`set_device_wifi_enable_status` to the mower (over BLE, `prefer_ble=True`). So the
hypothesis is directly testable: **turn the mower's Wi-Fi off, then re-run
`scripts/ble_advert_monitor.py` and watch session lifetime.** If advertising rate and
session length jump, coexistence is confirmed and the whole BLE problem becomes a
configuration question rather than a mystery.

**⚠️ Do not run this unattended.** Wi-Fi is this mower's *only* cloud path — the 4G
switch is off and `mobile_rssi` reads 0. Turning Wi-Fi off means:

- cloud transport disappears entirely;
- if BLE does *not* improve, the mower is unreachable from HA;
- the command to turn Wi-Fi back on must itself go over BLE.

So it should be done with the operator physically present and the mower in reach.

**Zero-risk alternative worth trying first:** the mower's own `wifi_rssi` reads −69/−74,
which is mediocre. A weak Wi-Fi link means more retries and therefore *more* radio time
spent on Wi-Fi, starving BLE further. Improving the mower's Wi-Fi coverage at the dock
(closer AP / better placement) would reduce contention without disabling anything, and
predicts a measurable BLE improvement on its own.

**Status: strong hypothesis, not confirmed.** It has the best explanatory fit of anything
proposed, but it has not been tested. Test before recording it as the cause.

## 🔑 2026-07-26: the mower is HANGING UP ON US — `error=19` (peer user terminated)

Chasing the MTU refined itself into something much better. The first
`bleak_esphome` connection-state debug line for the mower reads:

```
p1s-printer-a5774c [C4:DD:57:A5:77:4E]: Luba-VSPLV397 - A8:B5:8E:2C:52:40:
  Connection state changed to connected=False mtu=0 error=19
```

`error=19` is `0x13` = **`ESP_GATT_CONN_TERMINATE_PEER_USER`** — "connection terminated
by peer user". The **mower deliberately closed the link.** This is not:

| code | meaning | would imply |
|---|---|---|
| `0x08` (8) | connection/supervision timeout | passive radio starvation, missed connection events |
| `0x16` (22) | terminated by local host | *we* dropped it |
| **`0x13` (19)** | **terminated by peer user** | **the mower made a decision** |

That distinction matters a lot. A pure Wi-Fi/BLE coexistence starvation (the hypothesis
from earlier tonight) would surface as **0x08**, not 0x13. So the sessions are not simply
being starved off the air — the mower's firmware is choosing to hang up.

### The chain this completes

pymammotion already sends a `todev_ble_sync(2)` heartbeat specifically to hold the link
open — but note its own interval comment (`ble_loop.py`):

> The device drops out of its "synced" state ... after roughly its ~10 s keep-alive
> window ... **The APK sends sync every ~1.5 s; we use 5 s as a balance** — well under
> the ~10 s timeout while ~3x less BLE/ESPHome-proxy traffic than the APK cadence.

5 s is comfortably inside a 10 s window **only if every heartbeat lands**. And we
measured that they do not: **11 `parseNotification read sequence wrong` events in ~3
minutes** of connected BLE. Lose two consecutive 5 s heartbeats and the gap exceeds the
device's ~10 s window; at the app's 1.5 s cadence you would need ~7 consecutive losses to
do the same damage.

So the pieces fit into one chain, each link independently measured:

```
packet loss on the link  ->  missed 5 s heartbeats  ->  device keep-alive window (~10 s)
exceeded  ->  mower terminates the connection (error=19)  ->  ~73 s / ~5 min sessions
->  long cloud-only gaps waiting for the next rare advertisement
```

### What to try — cheap, reversible, upstream-shaped

**Raise the BLE heartbeat rate toward the app's cadence.** `_KEEP_ALIVE_BLE_INTERVAL`
is `5.0` in `pymammotion/device/ble_loop.py`; the app uses ~1.5 s. The existing comment
shows 5 s was chosen to reduce proxy traffic, explicitly trading margin for airtime —
that trade looks wrong on a link that is losing packets. This is the single highest-value
experiment available and it needs no mower motion.

Caveat worth stating: more heartbeats means more BLE traffic, which on a shared-radio
ESP32 could aggravate the coexistence pressure. So it is genuinely a test, not an obvious
win — measure session lifetime before and after.

### Confidence

**One disconnect sample.** The code semantics are certain and the supporting
measurements (heartbeat interval, sequence-gap rate, session lifetimes) are each solid,
but the `error=19` observation itself needs repeating before it is recorded as *the*
cause. A collector is running for further samples; if some disconnects come back `0x08`
instead, the starvation story is back in play alongside this one.

**This supersedes the MTU line of inquiry**, which was refuted (largest send ever: 54
bytes) though it did surface a real latent defect — see
`docs/pymammotion-ble-reassembly-bug.md`.

## ⚠️ CORRECTION (same night): `error=19` is NOT the whole story, and the MTU is UNSTABLE

The single `error=19` sample above was recorded with an explicit caveat — "if some
disconnects come back `0x08`, the starvation story is back in play". More samples
arrived, and that is exactly what happened.

### Every mower connection transition captured

| connects | disconnects |
|---|---|
| `mtu=517` ×3 | `error=19` (peer terminated) ×2 |
| `mtu=250` ×2 | `error=0` ×2 |
| `mtu=23` ×1 | **`error=8` (supervision timeout) ×1** |

**Two corrections follow.**

### 1. Both failure mechanisms are present, not one

`error=8` is `ESP_GATT_CONN_TIMEOUT` — the supervision timeout, i.e. the *passive
starvation* signature the coexistence hypothesis predicted. So the picture is mixed:
the mower sometimes deliberately hangs up (`0x13`, 2×) and sometimes the link simply
times out (`0x08`, 1×), with two more disconnects reporting `error=0`.

**Withdraw the framing that "the mower is hanging up on us" is *the* cause.** It happens,
and it is real, but it accounts for at most 2 of 5 observed disconnects. The
heartbeat-cadence experiment (`_KEEP_ALIVE_BLE_INTERVAL` 5 s → ~1.5 s) is still worth
running — it plausibly addresses both mechanisms — but it should not be sold as a fix for
a single diagnosed cause.

### 2. The MTU is not stable, which partly REVIVES the MTU angle

Negotiated MTU varies across connections: **517, 250, and 23**. The 23 case is the BLE
default, i.e. no successful negotiation at all.

This does **not** resurrect the send-side theory — sends are still ≤54 bytes and
`chunk_size=517` is still never reached. But it matters a great deal on the **receive**
side, because a BluFi frame carries at most `min(MTU-3, 255)` bytes:

| negotiated MTU | fragments for a ~249-byte report | exposure to the reassembly bug |
|---|---|---|
| 517 | 1 (unfragmented) | **none** — an unfragmented frame clears the buffer immediately |
| 250 | ~2 | moderate |
| **23** | **~13+** | **severe** — every fragment is another chance to lose one and poison the buffer |

So when the link negotiates a low MTU, fragmentation explodes and the buffer-poisoning
bug becomes dramatically more likely. That is a concrete mechanism connecting MTU
instability to the corrupted frames — and it is a *better* explanation of why the
corruption is intermittent than "reports happen to exceed 255 bytes".

Worth noting alongside: `bleak_esphome` caches the negotiated MTU
(`if not self._mtu: self._mtu = mtu`, plus `set_gatt_mtu_cache`). A bad negotiation
cached at 23 could persist across reconnects.

### Method note

This is the fifth time on this project that a confident single-sample conclusion has been
walked back by more data. The caveat was stated when the claim was made, which is why the
correction is cheap — but the lesson stands: **one sample of a categorical code is a
hypothesis, not a finding.**

## 📊 2026-07-27 BASELINE: 8 h of BLE sessions at the shipped 5 s heartbeat

Measured before changing anything, with `scripts/ble_session_report.py`. Mower docked
and idle throughout, `_KEEP_ALIVE_BLE_INTERVAL = 5.0` (confirmed on the host).

### Disconnect reasons — the most reliable statistic here

31 disconnect events, independent of session pairing:

| reason | count | share |
|---|---|---|
| `0x08` supervision timeout (starvation) | **13** | **42%** |
| clean / none | 9 | 29% |
| `0x13` peer user terminated (mower hung up) | 7 | 23% |
| `0x100` unknown | 2 | 6% |

### Session lifetime, and the reason predicts it

17 paired sessions: min 17 s, **median 59 s**, max **5453 s (91 min)**. Clearly bimodal —
11 short (17–93 s) and 6 long (212–5453 s). Crucially:

| disconnect reason | median session length |
|---|---|
| `0x08` supervision timeout | **41 s** |
| clean / none | 59 s |
| `0x13` peer terminated | **699 s (~12 min)** |

**Short sessions are supervision timeouts. When the mower deliberately hangs up, it does
so after ~12 minutes.**

*Caveat, stated because it biases the headline:* 56 connect events produced only 31
disconnects, and 38 connects had no disconnect before the next connect. Those pairs are
timed from the later connect, so the durations above are biased **downward**. The reason
tally is unaffected.

### 🔄 This reframes the heartbeat experiment — probably downward

The planned change (`_KEEP_ALIVE_BLE_INTERVAL` 5 s → ~1.5 s) targets the *device's*
keep-alive window: pymammotion's comment says the mower "drops out of its synced state
after roughly its ~10 s keep-alive window". That is an **application-level** timeout, and
the disconnect it would produce is the mower hanging up — `0x13`.

But `0x13` sessions last a **median of 12 minutes**. They are not what kills a 2–3 minute
path run. The sessions that die in 41 s die of `0x08`, a **link-layer** supervision
timeout: the radio link itself stopped exchanging connection events. A faster application
heartbeat does not obviously help that, and on a shared-radio ESP32 more airtime could
plausibly make it worse.

**So the heartbeat experiment is now a lower-value test than it looked**, and its
hypothesis should be stated honestly before running: *"does more frequent traffic reduce
supervision timeouts?"* — not *"this fixes the short sessions"*.

**What the data points at instead is the radio environment**, which is also what the
ESP32 coexistence hypothesis predicts. The zero-code actions are the ones to try first:
a proxy nearer the dock, reducing what the (permanently one-slot-busy) `p1s-printer`
proxy carries, and improving the mower's own Wi-Fi signal so its shared radio spends less
time on retries.

### Other results from the same window

- **MTU: 22 fresh negotiations, all 517.** The 23/250 negotiations seen on 07-26 did
  **not** recur, so MTU instability is intermittent rather than the current state.
  (Connects logging `mtu=0` reused a cached value — they are not low-MTU negotiations;
  an earlier version of the report script miscounted them as such.)
- **Link quality: 720 sequence gaps (1.5/min), 193 unparseable frames, 10 dropped**
  across 8 h — the same order as every previous sample, so packet loss is a stable
  property of this link, not an episode.
- A 91-minute session proves the link *can* hold for a long time; nothing here is a hard
  ceiling.
