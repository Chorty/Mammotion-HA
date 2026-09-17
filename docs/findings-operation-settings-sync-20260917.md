# Findings: HA's mowing-settings entities vs the running job (2026-09-17)

**Trigger:** `docs/TODO.md` top entry, the operator's report from ~02:50Z. A mow
was started from the Mammotion app. HA's *Working speed* was changed 2 → 1.6 ft/s
mid-mow, and the mower's blade height then went to ~1″.

**Method:** code read of this repo at `main` `97577425` and the installed
backend `pymammotion 0.8.12.post4` (`.venv/lib/python3.14/site-packages/pymammotion`);
the vendor APK decompile; upstream `mikey0000/*`, read-only; **read-only** HA API
GETs (`/api/states`, `/api/history/period`, `/api/diagnostics`). **No command was
sent to the mower, no `mammotion.*` service was called, and HA was not restarted
or reloaded.** The live BLE trace and its files were not touched.

**Fix branch:** `fix/operation-settings-running-job-sync` (not deployed). §6 covers the fix, §7 the test runs,
and §8 the hardware test plan, which still needs a separate operator go.

---

## 0. Verdicts on the previous session's claims

| Claim | Verdict | Evidence |
| --- | --- | --- |
| All three entities read/write a local `OperationSettings()` built in `__init__` from pymammotion defaults | **Confirmed** | `coordinator.py:184`; defaults `speed 0.3` / `channel_width 25` / `blade_height 0` at `pymammotion/data/model/device_config.py:19,22,24` |
| Nothing fills it from the device | **Confirmed**, with one refinement | See §1.4: `async_modify_plan_route` overlays *some* fields from `data.work` (`coordinator.py:2003-2011`), but **not** blade height, speed or path spacing |
| `blade_height` and `working_speed` call `async_modify_plan_if_mowing()`, which sends the whole local plan | **Confirmed**, plus a third caller | `number.py:186,202`; `coordinator.py:2143-2150`. **Also** the `bypass_mode` select (`select.py:341`) |
| This is how the speed change set the blade to ~1″ | **Confirmed by timing on the device's own report** | §2: blade 60 → 35 → **25 mm** within 20 s of HA's speed write |
| `working_speed` range 0.2–0.6 m/s is below 2 ft/s | **Refuted for this mower** | The hardcoded range is overridden by the model's `DeviceLimits` (`number.py:231-235,370-372`). Live entity: min 0.6 / max 4.0 ft/s = **0.2–1.2 m/s** |

---

## 1. Write path, end to end (pre-fix code)

### 1.1 Entity → coordinator

| Entity | `set_fn` (sync, local) | `set_async_fn` (sends) | `get_fn` (display) |
| --- | --- | --- | --- |
| `blade_height` (mm) | `operation_settings.blade_height = int(value)` `number.py:182-184` | `async_modify_plan_if_mowing()` `number.py:185-187` | `operation_settings.blade_height` `number.py:188` |
| `working_speed` (m/s) | `operation_settings.speed = value` `number.py:204-206` | `async_modify_plan_if_mowing()` `number.py:201-203` | **none** |
| `path_spacing` (cm) | `operation_settings.channel_width = value` `number.py:215-217` | **none** (never sent mid-job) | **none** |
| `bypass_mode` select | `operation_settings.ultra_wave` `select.py:336-340` | `async_modify_plan_if_mowing()` `select.py:341` | n/a |

`MammotionWorkingNumberEntity.async_set_native_value` (`number.py:404-413`) runs
`set_fn` and then `set_async_fn`.

### 1.2 Coordinator gate and send

- `async_modify_plan_if_mowing` (`coordinator.py:2143-2150`) sends when
  `int(report_data.work.bp_hash) in data.work.zone_hashs` and progress ≠ 100.
  🚨 **It never checks `sys_status`.** Cached `data.work` from an earlier job on
  the same zone is enough to pass, so a change while docked also re-issues the
  route. The pre-fix test run shows this: `test_idle_changes_only_update_the_plan`
  got *"Awaited 2 times"* at `MODE_READY` (§7).
- `async_modify_plan_route` (`coordinator.py:1998-2017`) copies `areas, toward,
  toward_mode, toward_included_angle, mowing_laps(edge_mode), job_mode, job_id,
  job_version` from `data.work`, then sends everything else from the local plan.
- `generate_route_information` (`coordinator.py:1932-1967`) maps the plan into
  `GenerateRouteInformation`. `blade_height=operation_settings.blade_height`
  (`:1955`), `speed` (`:1948`), `channel_width` (`:1957`). `job_id`/`job_version`
  are **not** copied into it.

### 1.3 pymammotion → wire

`MammotionCommand.modify_route_information`
(`pymammotion/mammotion/commands/messages/navigation.py:556-575`) builds
`MctlNav.bidire_reqconver_path = NavReqCoverPath(pver=1, sub_cmd=3, …)`.

| Wire field (`mctrl_nav.proto:177-196`) | Sent from | Unit | Sent when untouched? |
| --- | --- | --- | --- |
| `zoneHashs` (13) | `data.work.zone_hashs` | hash | yes, from the cache |
| `jobMode` (4) | `data.work.job_mode` | enum | yes, from the cache |
| `edgeMode` (6) | `data.work.edge_mode` | laps | yes, from the cache |
| `knifeHeight` (7) | **local plan** `blade_height` | mm, int32 | **yes: local value, default 0** |
| `speed` (12) | **local plan** `speed` | m/s, float32 | **yes: local value, default 0.3** |
| `UltraWave` (9) | **local plan** `ultra_wave` | enum | **yes: local value, default 2** |
| `channelWidth` (8) | **local plan** `channel_width` | cm, int32 | **yes: local value, default 25** |
| `channelMode` (10) | **local plan** `channel_mode` | enum | **yes: local value, default 0** |
| `toward` (11) | `data.work.toward` | deg | yes, from the cache |
| `reserved` (15) | **local** `create_path_order(...)` (`device_config.py:36-58`) | 8 bytes: border laps, obstacle laps, start progress, grass freq | **yes: local values** |
| `jobId`, `jobVer`, `toward_mode`, `toward_included_angle` | **not set** on sub_cmd=3 | — | — |

✅ **The app sends the same set.** `MACommandHelper.modifyGenerateRouteInformation`
(APK `command/MACommandHelper.java:1188-1197`) sets exactly `pver, subCmd=3,
zoneHashs, jobMode, edgeMode, knifeHeight, speed, ultraWave, channelWidth,
channelMode, toward, reserved`, and zeroes `toward` when `towardMode == 0`. So
sub_cmd=3 **always carries every setting**, and what matters is where each value
comes from.

### 1.4 Why the entities showed the wrong values

- **`working_speed` and `path_spacing` have no `get_fn`**, so they only ever show
  the entity's own `_attr_native_value`: `native_min_value` at construction
  (`number.py:311`), pushed into the plan by `set_fn` (`number.py:322-326`), then
  whatever `RestoreNumber` restores (`number.py:344-355`).
- **`blade_height` shows the plan with a display-only clamp.** Construction
  reads `get_fn` → plan `blade_height` (default **0**) and skips `set_fn` (the
  `elif` at `number.py:322`). `MammotionWorkingNumberEntity` then displays
  `max(value, min)` (`number.py:383-386`) **without writing it back**. The entity
  shows the minimum while the plan holds 0, until a restore or user write calls
  `set_fn`. Upstream #856 reports exactly this: the entity at 0, and 1202 after a
  speed change.
- **HA's number unit conversion (read from `homeassistant/components/number/__init__.py`):**
  the display unit is a per-entity registry option (`:520-540`), and
  `_convert_to_state_value` rounds to the **native value's decimal places**
  (`:468-500`). An integer mm value in inches rounds to **0 decimals**: native
  13–38 mm all read **"1.0 in"**. The step is **not** converted (`:342-356`), so
  `blade_height` in inches steps 1 in at a time. HA could not set 2.2″ at all.

---

## 2. What happened at 02:34Z, from HA's own recorder

HA history GET, entity prefix `back_yard_clip_skywalker_`. Inches converted to mm
(× 25.4), ft/s to m/s (× 0.3048). **Raw rows, no smoothing.**

| UTC | Entity | Value |
| --- | --- | --- |
| 02:31:09 | `sensor.activity_mode` | `mode_working` (app start) |
| 02:31:14 | `sensor.blade_height` (= `report_data.work.knife_height`, `sensor.py:201`) | 55 mm |
| 02:31:15 | `sensor.blade_height` | **60 mm** |
| 02:31:13 – 02:34:25 | `sensor.mowing_speed` (= `man_run_speed/100`, `sensor.py:480`) | straight-line samples **0.58–0.60 m/s** |
| 02:34:26 | `number.working_speed` | 0.7 ft/s (= 0.213 m/s, see §5.2) |
| 02:34:34.9 | `number.working_speed` | **2.0 ft/s = 0.610 m/s** (HA write) |
| 02:34:44 | `sensor.blade_height` | **35 mm** |
| 02:34:46.8 | `number.working_speed` | **1.6 ft/s = 0.488 m/s** (HA write) |
| 02:34:54 | `sensor.blade_height` | **25 mm** ≈ 0.98″ |
| 02:35:04 | `sensor.blade_height` | 45 mm |
| 02:35:14 | `sensor.blade_height` | **70 mm**, held until 03:09:54 |
| 02:35:34, 02:36:04 | `sensor.mowing_speed` | straight-line peaks **0.49 m/s** |
| whole window | `number.blade_height` | **1.0 in** (never changed) |
| whole window | `number.path_spacing` | **8.0 in** (native 20 cm → 7.87 in, never changed) |

**Reading:**
- The blade left the job's 60 mm within 10 s of HA's first write and reached
  **25 mm**. That is a value in the band HA's entity displayed as "1.0 in", and
  the slider floor on the hardcoded description (`number.py:179`). ✅ Consistent
  with a sub_cmd=3 carrying HA's local blade height.
- ✅ **The speed change also took effect:** straight-line peaks went from
  0.58–0.60 to 0.49 m/s.
- ⚠️ The report cadence drops from ~1 s to ~10 s at 02:33:50, so the blade's
  intermediate steps are coarse.
- ⚠️ **The rise to 70 mm at 02:35:04–14 is not explained by the code.** Both HA
  writes send the same local blade height. See §9.

**Live now (GET `/api/states`, 17:18Z, mower `mode_ready`):**
`number.working_speed` 1.6 ft/s (min 0.6, max 4.0, step 0.1);
`number.blade_height` 1.0 in (min 0.0, max 3.0, **step 1**);
`number.path_spacing` 8.0 in (min 7, max 14); `sensor.blade_height` 1.9685 in = 50 mm.

---

## 3. Read path: where the running job's settings live

| Source | Fields | Live or cached | HA entity today |
| --- | --- | --- | --- |
| `toapp_report_data` → `report_data.work` (`mctrl_sys.proto` field 20 `knife_height`, 18 `man_run_speed`) | **knife_height** (mm), **man_run_speed** (cm/s, instantaneous ground speed), `path_hash`, `bp_hash`, `area` (progress << 16) | **Live**, pushed in the report stream (~1 s on BLE, ~10 s otherwise in §2) | `sensor.*_blade_height`, `sensor.*_mowing_speed` |
| `bidire_reqconver_path` (NavReqCoverPath) → `data.work` (`CurrentTaskSettings`, `pymammotion/data/model/work.py:9-29`), rebound at `pymammotion/device/state_reducer.py:408-411` | **knife_height, speed (planned), channel_width**, ultra_wave, channel_mode, edge_mode, job_mode, zone_hashs, toward, toward_mode, toward_included_angle, reserved (obstacle laps etc.), path_hash, job_id/ver | **Cached**: only updates when the mower sends a `bidire_reqconver_path`. **Never tied to a job**, so it can hold an earlier job's values | none (pre-fix) |
| Query `query_generate_route_information` (`navigation.py:585-594`, NavReqCoverPath `sub_cmd=2`) | Whatever the reply carries (above) | **On demand** | none |

- **Planned speed and path spacing cannot be read live.** Only `bidire_reqconver_path`
  carries them. `man_run_speed` is ground speed, which reads 0.13 m/s on turns in
  §2, so it is not the setting.
- **HA sends the sub_cmd=2 query only on its own start/resume paths**
  (`lawn_mower.py:306,312`). For an app-started job, nothing in HA requests it.
  pymammotion's auto path-fetch saga does **not** query either: with
  `skip_planning=True` and nothing cached it fails
  (`pymammotion/messaging/mow_path_saga.py:200`), despite its docstring (`:36-38`).
- **The app does query.** `HomeMapFragment` (APK `map/fragment/HomeMapFragment.java:1221-1225`)
  calls `queryGenerateRouteInformation()` when the device state enters 13
  (working), and again on cancel of the in-job editor (`:3030`). Its in-job
  editor confirms via `modifyGenerateRouteInformation` on LubaPro and later
  models, and via `setKnifeHight` only on others (`:3063-3068`). The RN settings
  path does the same when `isWorking` (APK `rn/module/WorkSettingModule.java:531-541`).
- ⚠️ **The reply's content is not verified.** Nothing in this session saw an
  actual sub_cmd=2 reply, so it is unconfirmed that it carries a non-zero
  `knifeHeight`/`speed`/`channelWidth`. The fix refuses to send if any of them is
  zero (§6).
- **What `data.work` held at 02:34 cannot be read without a service call.** The
  diagnostics endpoint returns coordinator health only (`diagnostics.py:16-80`).
  The pre-fix gate (§1.2) **did pass**, since the blade moved, so `data.work.zone_hashs`
  contained the breakpoint zone. Whether it held *this* job or an earlier one on
  the same zone is unknown.

---

## 4. Upstream (read-only)

- **`mikey0000/Mammotion-HA` issue #856** (open, 2026-08-20), *"Changing speed
  triggers blade jam 1202"*: same mechanism, diagnosed the same way. The
  maintainer replied *"your correct ive not accounted for users changing the blade
  height on tasks, will address that"*.
- **Upstream commit `67bf27405`** (2026-09-10, "mid-task setting preservation",
  bumps pymammotion to 0.9.0b4) adds `_seed_operation_settings_from_running_job`,
  `async_change_blade_height_if_working` and `async_change_speed_if_working`.
  These seed speed, channel width, blade, ultra_wave, channel mode, angles and job
  ids from **`data.work`** before a sub_cmd=3 re-issue, and send `set_blade_height`
  only on non-LubaPro. It includes `tests/test_blade_height_running_job.py`.
  **Gaps relative to this report:**
  1. It seeds from the **cached** `data.work` and does not re-query, so an
     app-started job can be re-issued with an earlier job's settings.
  2. It sends no query, and its gate (`bp_hash in work.zone_hashs`) is false when
     `work` is empty. On a fresh HA with an app-started job, the change is
     silently dropped.
  3. It does not change what the entities *display*.
  4. It does not seed `reserved` (obstacle laps, border laps).
- **Upstream PR #877** (open, third-party): clamps working-number initial and
  restored values into model limits **and writes them back into the plan**.
  This addresses the "shown 25, sent 0" half of §1.4.
- **`mikey0000/PyMammotion`:** `modify_route_information` and `CurrentTaskSettings`
  on the default branch are **unchanged** from our wheel. No backend fix exists,
  so the fix belongs in HA.

---

## 5. Units and ranges

### 5.1 Per-field

| Setting | Native unit (entity) | Wire | This mower's limits | Notes |
| --- | --- | --- | --- | --- |
| Blade height | mm | `knifeHeight` int32 mm | live entity 0.0–3.0 in: HA floors/ceils the mm limits to 0 decimals, so min is < 25.4 mm and max is in (50.8, 76.2] mm (70 mm was reached, so ≥ 70). Exact values not read (the model's product key was not looked up) | Reports move in 5 mm steps (25, 35, 45, 50, 55, 60, 65, 70 all observed). **The in-inches step is 1 in** (§1.4) |
| Working speed | m/s | `speed` float32 m/s | 0.2–1.2 m/s (entity 0.6–4.0 ft/s) | 2 ft/s = 0.61 m/s is **in range**. Measured straight-line 0.58–0.60 m/s before the change and 0.49 m/s after |
| Path spacing | cm | `channelWidth` int32 cm | live entity 7–14 in, consistent with 20–35 cm | Never sent mid-job, before or after the fix |

### 5.2 The 02:34:26 "0.2 → 0.7" row

`number.working_speed` went 0.2 → 0.7 eight seconds before the first typed
value. 0.2 m/s = 0.656 ft/s, which rounds to 0.7. **Inference, not verified:**
this is the entity's display unit being switched to ft/s, not a write. HA writes
no service call for a unit change, and the registry was not read.

---

## 6. The fix (branch `fix/operation-settings-running-job-sync`)

**Behaviour:**
1. **Mid-job change = read the job, change one field, send.**
   `async_apply_working_setting(field)` (coordinator) is the only path for
   `blade_height`, `working_speed` and `bypass_mode`:
   - **Not working/paused** (`sys_status` ∉ {13, 19}, or progress 100): nothing
     is sent. The value stays in HA's plan for the next job HA starts. This also
     closes the docked re-issue in §1.2.
   - **Luba 2 and later** (`DeviceType.is_luba_pro`): send `query_generate_route_information`,
     then wait for `bidire_reqconver_path` and use **that reply**, never the
     cached `data.work`. Build route settings on a **copy** of the plan, seeded
     with the job's zones, job mode, edge mode, blade, speed, channel width,
     channel mode, ultra_wave, toward (zeroed when `toward_mode` is 0, as the
     app does) and `reserved`-decoded border laps, obstacle laps, start progress
     and grass frequency. Set the one edited field, then send sub_cmd=3.
   - **Luba 1:** blade height → `set_blade_height` only. Speed and detection →
     nothing, which is what the app's in-job editor does.
2. **Fail closed.** If the query errors or times out, or the reply has no zones
   or a zero speed, channel width or blade height (blade not required on Yuka),
   **nothing is sent** and the entity raises `running_job_unreadable`, translated
   in all 13 files. The plan still holds the new value.
3. **Display.** All three entities read `working_setting_value(field)`: the
   running job's value if HA has read it **and** the mower is still in that job
   (`sys_status` working/paused **and** `report_data.work.path_hash` unchanged
   since the read); otherwise HA's plan. The new attribute **`value_source`** is
   `running_job` | `running_job_after_ha_change` | `next_job_plan`.
4. **The plan stays inside model limits.** Out-of-range initial or restored plan
   values are clamped **into the plan**, not just the display, so a fresh
   install no longer plans a job with `knife_height` 0.

**Deliberately not changed:**
- **`path_spacing` is still not applied mid-job.** It never was, and the app's
  in-job editor does not expose it (as far as the decompile shows).
- **The `lawn_mower.start_mow` service with `modify: true`** (`lawn_mower.py:252-288`)
  still calls `async_modify_plan_route` with a copy of the plan plus the service
  arguments. **The same stale-fill hazard remains there** for any field the
  service call does not pass. Not in this report's path.
- **No automatic query when an app job starts.** Until HA reads the job (on the
  first HA change), the entities show the plan, labelled `next_job_plan`.
  Querying automatically on entry to working would mirror the app, but it sends
  traffic on every job start, so it is left for an operator call.
- **The inch-step problem** (§1.4) is untouched.

---

## 7. Tests

`tests/components/mammotion/test_running_job_settings.py`, 17 cases. They drive
the real entity descriptions and real coordinator methods, stubbing only
`send_command_and_wait` and `async_send_command`.

**Before the fix** (source at `main` `97577425`, final test file): **16 failed,
1 passed**. Raw failure lines, in test order:

```
E   AssertionError: assert 25 == 60                       # speed change sent blade 25, job had 60
E   assert 0.2 == 0.61 ± 6.1e-07                          # blade change sent speed 0.2, job had 0.61
E   AssertionError: Expected mock to have been awaited once. Awaited 0 times.   # never read the job
E   AttributeError: ... no attribute 'async_apply_working_setting'              # detection-mode test (new API)
E   Failed: DID NOT RAISE <class 'homeassistant.exceptions.HomeAssistantError'> # x5: sent despite unreadable/incomplete job
E   AssertionError: Expected mock to not have been awaited. Awaited 2 times.    # re-issued the route while MODE_READY
E   AssertionError: Expected mock to have been awaited once. Awaited 0 times.   # Luba 1: route re-issue, no knife command
E   TypeError: 'NoneType' object is not callable                                # x3: speed/spacing have no get_fn
E   AttributeError: ... no attribute 'working_setting_source'
E   assert 0 == 30                                        # plan blade 0 while entity showed the minimum
16 failed, 1 passed in 0.25s
```

The one pre-fix pass is `test_a_mid_job_change_does_not_overwrite_the_other_local_plan_values`,
a guard on the fix's copy semantics. The old code did not mutate those fields either.
⚠️ The detection-mode case fails pre-fix on a missing method, not on behaviour.
The speed and blade cases carry the behavioural proof for the same code path.

**After the fix:** `17 passed in 0.34s`.

---

## 8. Hardware test plan — ⛔ NOT RUN, needs a separate explicit operator go

Preconditions: the fix deployed through the normal beta workflow (not done here);
the other session's trace finished; daylight; the Mammotion app **closed after
starting the job**, since it takes the BLE link (CLAUDE.md).

1. In the app, start a mow at a blade height and speed that differ from HA's
   plan, e.g. **blade 60 mm** and **speed 0.6 m/s** (2.0 ft/s). Close the app.
2. Wait for `sensor.*_blade_height` to settle at the app value. Record it and
   `number.*_blade_height`, `number.*_working_speed`, `number.*_path_spacing`
   with their `value_source` attributes (expect `next_job_plan`).
3. In HA, set **Working speed** to a clearly different value, e.g. 1.6 ft/s.
4. **Pass:** `sensor.*_blade_height` stays at the step-1 value for ≥ 2 min.
   Straight-line `sensor.*_mowing_speed` peaks move to ~0.49 m/s. The three
   numbers now show the job's values with `value_source: running_job_after_ha_change`.
   **Fail:** any blade height change, or an entity error other than
   `running_job_unreadable`.
5. If HA raises `running_job_unreadable` instead: record it. The mower must show
   **no** setting change; that is the fail-closed path, not a pass of step 4.
6. Stop the mow in the app or HA as the operator prefers, and confirm
   `value_source` returns to `next_job_plan` once the mower leaves the job.

---

## 9. Not established

1. **Why the blade rose 45 → 70 mm at 02:35:04–14 and stayed at 70.** Both HA
   writes carry the same local blade height. Whether the operator changed it in
   the app, or the firmware reacted, is not in HA's record.
2. **The exact local plan values at 02:34.** `number.blade_height` "1.0 in"
   covers native 13–38 mm, and a plan value of 0 is also possible (§1.4). The
   device reached 25 mm, which fits 25.
3. **Whether HA's `data.work` at 02:34 was this job or an earlier one** (§3). It
   cannot be read without a service call.
4. **Whether a sub_cmd=2 reply carries non-zero knife height, speed and channel
   width.** Unobserved. The fix fails closed if not.
5. **The operator's "2.2″" vs the reported 60 mm (2.36″)** for the app job. The
   report moves in 5 mm steps, and 55 mm = 2.17″ was seen only for ~1 s at start.
6. **Why straight-line speed plateaued at 0.39–0.40 m/s from 02:45Z** (after app
   pauses at 02:40–02:45), rather than 0.49. The same plateau appears in the
   00:00Z mow.
7. **Whether the 02:34:26 row was a unit switch** (§5.2).
8. **Whether the app's in-job editor exposes path spacing.** Only the confirm
   path was read.
