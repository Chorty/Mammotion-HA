# Codex working plan / handoff memory

Last updated: 2026-06-28

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

- Current local working version is `0.6.4-beta12`.
- `0.6.4-beta12` was copied to HA via SMB and the Mammotion config entry reload returned success (`[]`) on 2026-06-28.
- A config-entry reload does not necessarily unload/reimport already-imported custom-component Python modules. New entity classes/platform setup may require a full HA restart before they appear.
- After beta12 deploy/reload, HA confirmed:
  - HA state: `RUNNING`
  - HA version: `2026.7.0b2`
  - `mammotion` loaded: yes
  - `mammotion.camera` loaded: yes
  - `mammotion.get_tasks` service registered: yes
  - `mammotion.get_areas` service registered: yes
- After beta12 deploy/reload, map/task diagnostic sensors were visible, but APK-derived controls such as `prompt_volume`, `voice_language`, `camera_wiper`, `device_wifi_enabled`, and `device_4g_enabled` were not visible yet. Likely next step: restart HA with explicit approval.
- Current committed version on `origin/main` may still be `0.6.4-beta11` until beta12 cleanup/deploy changes are committed and pushed.
- `main` is pushed to `origin/main` on `Chorty/Mammotion-HA`.
- Claude merged upstream Mammotion-HA beta7 content into this branch, then committed/pushed the beta10/beta11 working tree.
- Claude reports `0.6.4-beta11` was deployed to HA and the Mammotion config entry was reloaded through the HA REST API.
- HA may still display stale integration metadata in some places until Home Assistant restarts because custom integration metadata is cached in the running HA process.
- Current manifest/pyproject version:

  ```json
  "version": "0.6.4-beta12"
  ```

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

Observed Agora SDK diagnostics:

- Browser SDK can join the channel and see mower user/video.
- Local audio track appeared live/enabled but Agora local audio stats stayed at zero send bytes/packets/bitrate.
- Mower published video but did not publish/subscribe audio in the observed way needed for speaker playback.

Practical conclusion:

- Focus on built-in spoken prompt settings rather than arbitrary TTS unless a separate app talkback mode/command table is found.

## Recommended next implementation batch

Camera/cloud reliability and diagnostics should be next, before more write controls:

- `button.<mower>_refresh_camera_stream`
- `button.<mower>_refresh_cloud_session`
- diagnostic sensor: active transport
- diagnostic sensor: last cloud login success
- diagnostic sensor: last token refresh
- diagnostic sensor: last command failure reason
- diagnostic sensor: last camera stream failure code
- diagnostic sensor/binary sensor: BLE-only fallback mode

Goal:

- Make camera and cloud failures visible/actionable instead of generic “temporarily unavailable”.

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
