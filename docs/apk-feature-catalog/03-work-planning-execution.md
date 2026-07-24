# Work planning, task execution, schedules, and work settings

## Scope and confidence

This catalog is based on the decompiled Android sources under
`/Users/mattjoslin/mammotion-apk-decompile/src/sources`. The supplied path omitted
the decompiler's intermediate `sources/` directory. Primary scope was
`com/agilexrobotics/work/setting`, the work controls in `home`, relevant command
helpers and generated protobuf classes, and `base_module` work-report and plan
beans. `services/` was searched; it contains service/help/touch-mode material but
no additional mowing-plan implementation beyond links and feature presentation.

Evidence references are source-relative and include decompiler line numbers.
“Confirmed” means the UI/model and a backing field or command are both visible.
“Probable” means naming and data flow are strong but the generated/decompiled
code does not expose a complete value mapping. “Latent” means the protocol or
configuration supports it even if this app path does not visibly offer it.

Decompiler caveats:

* Several large methods are only partly reconstructed, and Kotlin metadata often
  preserves better names than JADX method bodies.
* Java field defaults are app-side defaults, not necessarily firmware defaults.
* Enum integer meanings are reported only where an enum, resource mapping, or
  branch proves them. Unmapped integers must not be treated as stable API values.
* Units are normalized inconsistently: speed is commonly stored in tenths of
  m/s, path spacing in centimetres, knife height in millimetres, and protocol
  speed may be a float.

## Architecture and data flow

| Stage | Representation | Confirmed contents | Evidence |
|---|---|---|---|
| Capability/config download | `WorkingSettingResBean.DataDTO.DetailVosDTO` | setting `code`, show/force flags, type, min/max/default/range; cached per device as `device_working_setting_<deviceName>` | `work/setting/api/WorkingSettingManage.java:129-181` |
| Editable task settings | `OperationModeDB` | mode bitmask, spacing, speed, obstacle mode, route pattern/angle, mowing and obstacle laps, border/path order, progress offset, mowing/sweeping/edge flags, collection frequency, knife height, angle strategy, ride-on-boundary distance | `base_module/db/OperationModeDB.java:10-32` |
| Area selection | `WorkSettingBean` / `TaskAssignmentManager.mapHashList` | full area objects plus ordered area hash list | `work/setting/api/WorkSettingBean.java:8-38`; `work/setting/api/TaskAssignmentManager.java:23-35,245-247` |
| Scheduled plan | `PlanBean1` | identifiers, time/date/weekday/interval trigger, zone hashes, route and mower settings, enable/result/progress metadata | `base_module/entity/PlanBean1.java:21-76` |
| Device command | `MACommandApiHelper`, `CommandManager`, generated protobuf | plan add/query/delete/edit, immediate route generation/modification, start/pause/resume/end/return, DND, save/discard task | `home/utils/HomeCommandHelper.java:152-274`; `command/CommandManager.java:658-776,887-962`; `proto/SpinoCtrlOuterClass.java:2806` |
| Runtime state | device state machine / `DeviceWorkState` | ready, working, corridor-working, pause, corridor-pause, breakpoint/fuse/zone context, progress | `home/utils/HomeCommandHelper.java:203-271`; `home/fragment/DeviceStateFragment.java:3913-4005` |
| History/report | REST beans plus task-report file transfer | paginated summary, detail metrics/path/events, calendar days, binary/file report exchange | `base_module/bean/workreport/*.java`; `proto/MctrlSys.java` descriptor (`task_report_*`) |

The practical Home Assistant boundary is therefore not the UI fragments. It is
the combination of (1) device capability/config, (2) `OperationModeDB`-equivalent
task parameters, (3) ordered zone hashes, (4) command/ack transport, and (5)
state-machine telemetry.

## Feature catalog: planning and area selection

| Feature | Behavior and values | Status | HA relevance | Evidence |
|---|---|---|---|---|
| Select one or more work areas | Selection is carried as `List<Long>` hashes; UI also retains `AreaBean` objects. Empty/invalid selection is blocked in the task-assignment flow. | Confirmed | Expose areas by stable hash, with display name as metadata. Never use display name as identity. | `work/setting/api/WorkSettingBean.java:8-38`; `work/setting/select_area/SelectAreaViewModel.java`; `work/setting/simpleworksetting/fragment/TaskAssignmentFragment.java` |
| Area naming | Reads mapped regions from local DB, overlays device `defAreaNameMap`, synthesizes “Area N” where absent, then returns `realAreaNameMap`. | Confirmed | Entity registry should refresh names while preserving hash IDs. | `work/setting/viewmodel/JobScheduleViewModel.java:74-123` |
| Ordered multi-zone task | `zoneHashs` is a list in `PlanBean1`; the UI has `PathOderFragment` and routing-map previews. | Confirmed | Preserve list order in service calls; do not serialize as a set. | `base_module/entity/PlanBean1.java:73`; `work/setting/simpleworksetting/fragment/PathOderFragment.java` |
| Route generation | `getGenerateRouteInformation` and `modifyGenerateRouteInformation` accept `GenerateRouteInformation`. | Confirmed | A “preview/estimate route” call and a distinct “modify active/generated route” call are plausible services. | `command/CommandManager.java:658-668,764-774` |
| Start from percentage | `start_progress`, default/reset `0`, displayed as `%`; existing settings are reset to zero for a new assignment. | Confirmed, model-configured | Useful as an advanced number, but likely unsafe without route-generation semantics. | `base_module/db/OperationModeDB.java:23`; `work/setting/api/TaskAssignmentManager.java:165-188,191-193` |
| Unreachable areas | Schedule/plan bean has `unArriveAreaIndex`. | Probable telemetry/result | Surface as diagnostic attributes or an event after plan validation/execution. | `base_module/entity/PlanBean1.java:69` |
| Outside/link flags | `isOutSide` and `isLink` are plan booleans, default false/true. | Probable plan metadata | Retain on round trip; do not invent UI semantics. | `base_module/entity/PlanBean1.java:35-37,91-94` |

## Feature catalog: work content, patterns, and numeric settings

### Work-content bitmask

`WorkConfig` proves a composable bitmask:

| Value | Meaning |
|---:|---|
| 0 | none |
| 2 | edge |
| 4 | dump / sweeping or grass collection |
| 6 | dump + edge |
| 8 | mow |
| 10 | mow + edge |
| 12 | mow + dump |
| 14 | mow + dump + edge |

Evidence: `work/setting/api/WorkConfig.java:12-26`. `OperationModeDB` also carries
`isMow`, `isDump`, and `isEdge` booleans
(`base_module/db/OperationModeDB.java:24-26`). “Dump” is the decompiled internal
term; YUKA UI classes (`CollectGrassConfigFragment`, `MowFrequencyFragment`) show
that it represents the sweeper/grass-collection attachment rather than waste
dumping alone.

### Settings

| Feature | Values/defaults and interactions | Status / gates | HA relevance | Evidence |
|---|---|---|---|---|
| Mowing / sweeping / edge combinations | Bitmask above; plan also has `isMowWork` and `isSweepingWork`. | Mower/attachment dependent | Use a select or explicit booleans that compile to the bitmask; reject unsupported combinations. | `WorkConfig.java:12-26`; `PlanBean1.java:36,39` |
| Cutting height | `knifeHeight`, app default 60 mm; min/max come from `DeviceMultimodelHelper`; out-of-range saved value is reset to max minus 10. | Hidden on some YUKA/231-like models and exposed on others according to downloaded setting code 3 | Number entity with device-derived limits and step; never assume 60 or global limits. | `OperationModeDB.java:28`; `TaskAssignmentManager.java:108-113,177-188`; `WorkingSettingManage.java:233-252` |
| Work speed | `tesk_speed`, app default 3 (tenths conversion is used); min/max model-derived; UI converts to m/s or fps. | Capability code 4; pool has separate float speed | Number entity using capability bounds; preserve raw and normalized units. | `OperationModeDB.java:15`; `WorkingSettingManage.java:118-124,253-259`; `TaskAssignmentManager.java:231-238` |
| Line/path spacing | `path_space`, app default 25 cm; device-derived min/max. Defaults include 9, 10, 12, 20, 25, and 50 by family. Mini/231-like guard clamps values over 12 back to 10. | Model-specific | Number entity with dynamic range; migration must validate old values after model/firmware change. | `OperationModeDB.java:14`; `WorkingSettingManage.java:260-285`; `TaskAssignmentManager.java:115-124` |
| Route/pattern mode | Values proven by UI: `3` no grid, `0` single grid, `1` double grid, `2` adaptive/single2. General range order `[3,0,1,2]`; X5 `[3,0,1]`. | X5 omits value 2 | Select with raw value retained. | `WorkingSettingManage.java:82-84`; `TaskAssignmentManager.java:199-222` |
| Angle strategy | `towardMode`: 0 efficiency, 1 custom, other displayed random. | Capability-configured | Select; only expose numeric angle when custom. | `OperationModeDB.java:29`; `TaskAssignmentManager.java:183-189` |
| Route/custom angle | `route_angle` and `demond_angle`; defaults 0 and 90. Both survive in plan representations (`routeAngle`, `demond_angle`). | UI includes `CuttingRouteFragment`; downloaded setting codes 6/9 | Number entity, likely degrees; exact accepted range not proven here. | `OperationModeDB.java:18,22`; `WorkingSettingManage.java:60-64,286-305`; `PlanBean1.java:27,55` |
| Perimeter mowing laps | `mowing_laps`, app default 3; displayed directly. | Capability flag `showTaskMowingLaps` | Number entity; derive min/max from setting detail, not this default. | `OperationModeDB.java:19`; `TaskAssignmentManager.java:127-130`; `WorkingSettingManage.java:73` |
| Obstacle/no-go laps | `mowing_laps_obs`, app default 0; UI label path is “Nogo”; `getNogoStatus` disables it for route mode 3. | Pattern-dependent | Advanced number; disable for no-grid/random route. | `OperationModeDB.java:20`; `TaskAssignmentManager.java:195-198` |
| Border/path order | `border_mode`: UI maps 0 to path-order B, nonzero to A; stored as `pathOrder` in plan. | Disabled for route mode 3; exposed through `showTaskPathOrder` | Select with raw 0/1. Human labels are localization-dependent. | `OperationModeDB.java:21`; `TaskAssignmentManager.java:174-177,225-227`; `PlanBean1.java:49` |
| Ride on boundary / edge distance | `rideBoundaryDistance`, app default 0.5; zero means off and nonzero means on in summary UI. | Only shown when new capability data says `showRideEdge` | Switch plus numeric distance if protocol supports both; avoid reducing nonzero values to a boolean. | `OperationModeDB.java:32`; `TaskAssignmentManager.java:179-182,223-224`; `WorkingSettingManage.java:74,78` |
| Obstacle detection mode | `detect_mode`, app default 10. UI mapping: 1 off; 2 or 10 standard; 11 ProGuard; unknown falls back to off. Ranges are normally `[1,10,11]`, X5 `[10,11]`. | X5 cannot select off; capability/config and vision model dependent | Select with explicit raw IDs. Good candidate for a mower setting entity. | `OperationModeDB.java:16`; `WorkingSettingManage.java:80-82,158-166`; `TaskAssignmentManager.java:240-265` |
| Grass collection frequency | `collectGrassFrequency`, default 10; UI translates as area and supports a collection-config fragment. | YUKA sweeper only | Number entity in configured area unit; semantics are area between collection/dump actions. | `OperationModeDB.java:27`; `TaskAssignmentManager.java:137-140`; `mow_settings/CollectGrassConfigFragment.java` |
| Automatic/work presets | Internal ranges `[1,10,11]`; fragments `AutomaticFragment` and `AutomaticTypeFragment`; `app_display_mode` retained. | Downloaded configuration/model dependent | Prefer raw preset select plus expanded settings readback. | `WorkingSettingManage.java:80-82`; `OperationModeDB.java:31` |
| Maximum speed | `max_speed` exists independently of task speed. | Latent in this UI path | Diagnostic/read-only until a command mapping is proven. | `OperationModeDB.java:30` |

## Capability codes and model gates

The server/device working-setting response is the authoritative UI schema. For
ordinary mowers the app manufactures fallback entries for codes 1–20; PC210
creates codes 201–207. Confirmed fallback mappings are:

| Code | Meaning inferred from assignment | Type / important gate |
|---:|---|---|
| 1 | mowing/collection frequency or YUKA-specific first option | hidden for non-YUKA and selected 231-like models |
| 3 | cutting height | model-derived min/max; different YUKA visibility |
| 4 | task speed | numeric, model min/max |
| 5 | path spacing | numeric, model min/max/default |
| 6 | route angle | numeric |
| 7 | route pattern mode | enum `[3,0,1,2]`, X5 `[3,0,1]` |
| 8 | angle/toward mode | enum |
| 9 | demanded/custom angle | numeric |
| 10 | mowing laps | enum/range from configuration |
| 11 | obstacle/no-go laps | enum/range from configuration |
| 201–207 | PC210/pool settings | mostly scalar; 205 enum-like, 206 numeric float |

Evidence: `work/setting/api/WorkingSettingManage.java:182-305`. Codes 2 and
12–20 are constructed but their complete switch body is not reliably recovered
in the inspected decompilation; feature booleans in the same class prove likely
targets (`showTaskNoGo`, `showTaskPathOrder`, `showRideEdge`, `showFloor`,
`showWall`, `showLine`, `showCustom`, `showSwimmingSpeed`, `showSwimmingSize`)
but the exact code-to-feature mapping remains uncertain.

Additional gates:

* `isX5DeviceTyp()` removes adaptive route mode and obstacle-off mode.
* `is231_500()` uses the X5 automatic-mode range.
* YUKA variants change path-spacing defaults and work-content visibility.
* LUBA MN/LD/MD/LA/MB and LUBA 2 Pro/VA/HM receive different spacing defaults.
* PC210 switches to the pool-oriented 201-series schema.
* Settings marked `force` or `show` by downloaded config can override fallback
  presentation. HA should consume capability data rather than duplicate this
  growing model matrix.

## Schedules and calendars

| Feature | Confirmed behavior | HA relevance | Evidence |
|---|---|---|---|
| Immediate task | `jobFrequency == -1` is displayed as “now”; new assignments reset progress. | Normal start service with task parameters. | `TaskAssignmentManager.java:126-135,177-193` |
| Weekly schedule | `weeks: List<Integer>`, plus legacy `week`; time in `startTime`. | Calendar-like entity or CRUD services; preserve weekday encoding until enum mapping is verified. | `PlanBean1.java:70-71`; `ScheduleWeekFragment.java` |
| Fixed date | UI schedule-frequency 2 stores `startDate`, sets plan `jobType=2`. | One-shot schedule. | `ScheduleCalendarFragment.java:95-104,156-164` |
| Interval schedule | UI schedule-frequency 3 stores `startDate` + `intervalDays`, sets plan `jobType=1`. | Repeating every N days. | `ScheduleCalendarFragment.java:95-112,164-167` |
| Start time | Separate time-picker fragment writes plan start time. | Time field; account for device timezone. | `ScheduleTimepickerFragment.java`; `PlanBean1.java:62` |
| End date/time | Both fields exist in the plan, though not all UI paths visibly populate them. | Preserve during read/edit; latent until confirmed per model. | `PlanBean1.java:28-29` |
| Enable/disable | `isEnable` defaults true and is parcelled with the schedule. | Switch per schedule. | `PlanBean1.java:33,89` |
| Name and IDs | task/job names and `planId`, `taskId`, `jobId`; equality is by `planId`. | Stable schedule identity must be plan ID, not name. | `PlanBean1.java:31-32,40-42,50-52,190-200` |
| CRUD | Pool protobuf explicitly defines `ADD=1`, `QUERY=2`, `DELETE=3`, `EDIT=4`, `DELETE_ALL=5`; mower `subCmd` likewise carries operation but exact mower numeric mapping should be verified before reuse. | Expose list/create/update/delete services with ack handling. | `proto/SpinoCtrlOuterClass.java:2806`; `PlanBean1.java:60` |
| Calendar summary | Report summary contains `workCalendars: List<Long>` and calendar UI decorates worked dates. | Historical calendar sensor, not future schedule source. | `ReportByDeviceNameResBean.java:7-15`; `work/setting/activity/CalendarActivity.java` |
| RTK readiness guard | Schedule view model checks bound RTK/Lora and OTA/status conditions before prompting. | Start/schedule validation should surface positioning prerequisites. | `JobScheduleViewModel.java:126-192` |
| Non-work/DND hours | A single unable interval (`unableStartTime`, `unableEndTime`) is editable/deletable; start is intercepted when current time falls inside DND. | Two time entities plus enable/delete service; start should return a blocked reason. | `HomeCommandHelper.java:60-151,213-230`; `CommandManager.java:776-786` |
| Sunrise/DND metadata | Plan exposes `isDndTime` and `isSunriseTime`. | Preserve/read diagnostically; no complete scheduling UI semantics proven. | `PlanBean1.java:34,38,92-94` |

No independent “rain schedule” is present in the scoped work planner. Rain is a
device sensor/test and a runtime constraint elsewhere in the system, not a
calendar recurrence field here. Therefore HA should model rain as condition/
telemetry and an automation guard, not fabricate a plan property. The generated
system protobuf includes a QC rain test, while this work plan does not contain a
rain field (`proto/MctrlSys.java`, `QC_APP_TEST_RAIN`).

## Task execution and state flow

```text
select ordered zones + settings
        |
        v
generate/submit route or scheduled PlanBean1
        |
        v
READY --startJob--> WORKING / CORRIDOR_WORKING
                         |
                      pause
                         v
                 PAUSE / CORRIDOR_PAUSE
                         |
       valid breakpoint + safety checks
                         |
            cancelPauseExecuteTask (resume)
                         v
                       WORKING
                         |
          +--------------+----------------+
          |              |                |
       closeJob     returnCharge     stop-and-save /
        (end)                         stop-no-save
```

| Action | App command path | Guards / nuance | HA service recommendation |
|---|---|---|---|
| Start | `MACommandApiHelper.startJob(null)` | From ready; blocks during DND, self-check errors, and may require location/RTK guidance. | `start` returning explicit command result/error. |
| Pause | `pauseExecuteTask(null)` | Working and corridor-working states map the main button to pause. | `pause`. |
| Resume | `cancelPauseExecuteTask(null)` | Resume is named “cancel pause”; depends on breakpoint type, zone/channel hashes, fuse state, and model safety gates. | `resume`; never implement as `start`. |
| End current job | `closeJob(null)` | Semantically distinct from task-save/discard. | `stop`/`end`. |
| Return to charger | `returnCharge(null)` | Runtime action; separate cancellation via misspelled `claseBacktoRecharge`. | `dock` and, only if supported, `cancel_return`. |
| Stop and save task/map | `stopAndSaveTask()` and `CommandManager.stopAndSaveTask` | Corridor/planning dialog offers save-and-end. | Diagnostic/advanced service with confirmation. |
| Stop without save | `CommandManager.stopAndNotSaveTask` | Destructive to in-progress task/map result. | Advanced service requiring confirmation. |
| Modify generated plan | `modifyGenerateRouteInformation(...)` | Separate from creating/querying route info. | `modify_plan`; require complete current config to avoid resetting omitted fields. |
| Blade/cutter control | system protobuf `todev_knife_ctrl`; runtime reports `current_cutter_mode`; blade usage reset/warn-time commands also exist | Work UI normally lets firmware coordinate blades. Direct cutter control is maintenance/manual functionality, not ordinary task start. | Prefer read-only blade state/usage; gate direct control as unsafe/diagnostic. |

Runtime state evidence: `home/utils/HomeCommandHelper.java:152-274`;
`home/fragment/DeviceStateFragment.java:3664-3703,3827-3869,3913-4005`;
`command/CommandManager.java:951-971`; and `proto/MctrlSys.java` descriptor fields
`todev_knife_ctrl`, `current_cutter_mode`, `todev_reset_blade_used_time`, and
`blade_used_warn_time`.

Important resume logic:

* `breakPointType` 0 or -1 is treated as a device bug/no generated breakpoint,
  and the app still sends cancel-pause.
* If current zone hash and channel hash differ, READY sends start while PAUSE
  sends cancel-pause.
* Matching zone/channel hashes add fuse/model checks. Radar and 231-like models
  resume directly; some older models reject resume when fuse status indicates
  automatic lowering is unsafe.

These branches make a single optimistic HA “toggle” inappropriate. Commands
should be state-aware, while the device remains the source of truth.

## Commands and protocol fields

### Plan/task payload

The mower-side app model (`PlanBean1`) exposes the following protocol-relevant
fields:

| Group | Fields |
|---|---|
| Version/operation | `pver`, `version`, `subCmd`, `result` |
| Identity | `id`, `userId`, `deviceId`, `planId`, `taskId`, `jobId`, `taskName`, `jobName` |
| Timing | `startTime`, `endTime`, `startDate`, `endDate`, `week`, `weeks`, `jobType`, `intervalDays`, `countDown`, `workTime`, `requiredTime`, `isEnable` |
| Work geometry | `zoneHashs`, `area`, `routeModel`, `routeAngle`, `routeSpacing`, `towardMode`, `demond_angle`, `pathOrder`, `rideBoundaryDistance` |
| Mower behavior | `model`, `edgeMode`, `knifeHeight`, `speed`, `ultrasonicBarrier`, `mowingLaps`, `isMowWork`, `isSweepingWork` |
| Sequencing/result | `totalPlanNum`, `PlanIndex`, `unArriveAreaIndex`, `reserved`, `isLink`, `isOutSide`, `isDndTime`, `isSunriseTime` |

Evidence: `base_module/entity/PlanBean1.java:21-76`.

The generated pool protocol class independently confirms a similar plan contract:

| Field no. | Field | Type/enum |
|---:|---|---|
| 1 | `cmd` | `PLAN_CMD` |
| 2 | `work_mode` | `APP_WORK` |
| 3 | `sub_mode` | repeated `APP_WORK` |
| 4–5 | `userId`, `deviceId` | string |
| 6 | `startTime` | fixed32 |
| 7–9 | `totalPlanNum`, `PlanIndex`, `result` | int / `PLAN_ACK` |
| 10–13 | `speed`, `operating_power`, `jobName`, `jobId` | float/string/fixed64 |
| 14–15 | `startDate`, `endDate` | string |
| 16 | `triggerType` | `PLAN_TYPE` |
| 17–18 | `day`, repeated `weeks` | fixed32 |
| 19–20 | `remained_seconds`, `enable` | int64/fixed32 |

`APP_WORK` is explicitly `IDLE=0`, `AUTO=1`, `FLOOR=2`, `WALL=3`, `ECO=4`,
`LINE=5`, `CUSTOM=6`. Evidence:
`sources/com/agilexrobotics/proto/SpinoCtrlOuterClass.java:2806`.

### Immediate and operational commands

| Command/API | Parameters visible in app | Purpose |
|---|---|---|
| `getGenerateRouteInformation` | device name, owner, link manager, `GenerateRouteInformation`, callback | generate/query work route |
| `modifyGenerateRouteInformation` | same shape | modify route/plan |
| `startJob`, `pauseExecuteTask`, `cancelPauseExecuteTask`, `closeJob` | callback/null | execution lifecycle |
| `returnCharge`, `claseBacktoRecharge` | callback/null | dock/cancel return |
| `stopAndSaveTask`, `stopAndNotSaveTask` | device name/owner/callback in manager variant | terminate mapping/task with persistence choice |
| `modifyJobDoNotDisturb` | `subCmd`, delete flag, start/end unable times | create/edit/delete non-work interval |
| `get/setRechargeAndContinueWorking` | `id`, read/write or context values | recharge-and-resume policy |
| `setPC210Plan` | `PlanJobSPBean` | pool schedule submission |

Evidence: `command/CommandManager.java:658-786,887-909,951-971`;
`home/utils/HomeCommandHelper.java:152-274`.

## Job history, reports, and diagnostics

| Surface | Fields / behavior | HA relevance | Evidence |
|---|---|---|---|
| Report summary | device, last-work time, total area/count, carbon reduction, saved time, calendar timestamps | Long-term statistics sensors. | `ReportByDeviceNameResBean.java:7-15` |
| Paged history | page/current/size/total; each record has work ID/name/type/result, start/end, area, progress, duration, product/device | History API or event import. | `DeviceWorkReportPageReq.java:5-7`; `DeviceWorkReportPageResBean.java:7-28` |
| Detail lookup | `deviceName`, `workId`, `workReportId` | Fetch detail on demand. | `DeviceWorkReportDetailReq.java:5-7` |
| Report detail | carbon, channel width, charging/other time, energy, job content, knife height, requested params, start/end, area, map path, progress, result, speed, duration/type | Rich diagnostics; path can support map rendering. | `ReportDetailResBean.java:7-27` |
| Work event timeline | repeated `WorkProcessDTO(eventCode,timestamp)` | Emit event timeline; code dictionary is not recovered here. | `ReportDetailResBean.java:68-70` |
| Pool request params | nested `customList[4]`, material, power level, speed level | Pool-report attributes. | `ReportDetailResBean.java:29-33` |
| Device report transfer | interaction signal; file request includes business ID, history/new type, file index/count/progress and packages; packages include work ID/name/data/size/MD5/frame metadata/error | Useful for diagnostics and report ingestion, but requires binary framing and integrity validation. | `proto/MctrlSys.java`, messages `task_report_interaction_t`, `FilePackage`, `FileTransferRequest/Response/Result` |
| Hidden debug reporting | report config/data, `debug_common_report`, `debug_errorcode_report`, debug enable/read/write/capability | Diagnostic only; potentially destabilizing writes. | `proto/MctrlSys.java` descriptor |

`JobHistoryActivity` and `JobHistoryDetailFragment` consume these report models;
the calendar and report-card bindings show both list/detail and home summary
surfaces. The backend report is distinct from live state, so HA should not use
report arrival to infer that the mower has stopped.

## Pool-cleaning analogues surfaced in this scope

Pool support is not merely dead protobuf: the home package contains dedicated
fragments, controls, local plan persistence, schedule retrieval, and RN work-mode
navigation.

| Feature | Confirmation | Evidence |
|---|---|---|
| Modes | Auto, floor, wall, eco, line, custom; custom can carry repeated sub-modes. | `proto/SpinoCtrlOuterClass.java:2806`; `home/fragment/DeviceStateSwimmingPoolSPFragment.java:1767-1791` |
| Work settings | swimming speed default 0.15 and size default 1; config flags for floor/wall/line/custom/speed/size. | `WorkingSettingManage.java:64-77` |
| Plan CRUD and scheduling | `PlanJobSPBean`, local LitePal cache, list retrieval, add/update by job ID, `setPC210Plan`; protobuf CRUD enum. | `DeviceStateSwimmingPoolSPFragment.java:1651-1682,1826-1888`; `CommandManager.java:887-897` |
| Start/action controls | Select-mode, start-clean, working-action, and return controls are bound in home. | `DeviceStateSwimmingPoolSPFragment.java:2014-2040` |
| Runtime states | idle, prepare, wait-water, working, pause-go-charge, end-go-charge, charging, leave-dock, recalling. | `proto/MctrlSys.java`, enum `SpinoSysStatus` |
| Surface/material configuration | wall material (glass/ceramics/sandstone), pool bottom shape (right-angle/curve × simple/complex), floor speed. | `proto/MctrlSys.java`, `app_downlink_cmd_t` and associated enums |
| Map/line/area clean | get map, get line, and `AreaClean {type, repeated points}` commands. | `proto/MctrlSys.java`, `app_downlink_cmd_type_e` |
| Docking | disabled, immediate, or timed docking with hour/minute point. | `proto/MctrlSys.java`, `DockingTime` |

Pool commands and mower commands must remain separate domains in HA even where
their lifecycle vocabulary overlaps.

## Hidden, latent, and safety-sensitive features

| Feature | Classification | Notes |
|---|---|---|
| Direct knife control | Safety-sensitive | Protocol exists, but normal work flow lets firmware manage cutters. Require explicit opt-in and state checks if ever exposed. |
| Reset blade-used time / blade warning threshold | Maintenance | Suitable as a button/number only with model capability and confirmation. |
| Recharge-and-continue | Latent/confirmed command | Getter/setter exists with opaque integer `id/context/rw`; semantics need wire capture or caller trace before HA exposure. |
| Start at progress percentage | Advanced | Confirmed setting, but route ownership and resume semantics make arbitrary values risky. |
| Debug config read/write and simulation | Diagnostic | Do not expose by default. Writes can alter firmware behavior. |
| Report file transfer | Diagnostic | Validate file size, MD5, sequence, and total frames before accepting data. |
| Self-check gate | Safety | Vision, battery, charge-state, and pass-through checks occur before work setup; command rejection should be surfaced, not bypassed. |
| Animal-protect/car touch mode | Adjacent service feature | `services/touchmode` configures interaction/animal protection, but no evidence makes it part of a task-plan payload. Model separately. |

## Recommended HA entity/service model

| HA surface | Suggested implementation |
|---|---|
| Lawn/zone registry | Read-only entities keyed by zone hash with mutable names. |
| Task configuration | Selects/numbers for content, pattern, angle strategy/value, speed, spacing, height, laps, obstacle mode, path order, edge distance, collection frequency. Availability and ranges must follow capability data. |
| Lifecycle | Services `start`, `pause`, `resume`, `end`, `dock`; use observed state for availability and wait for ack/state transition. |
| Plan service | One structured service taking ordered zone hashes and a complete settings object; optionally preview/generate then execute. |
| Schedule service | List/create/update/delete/enable with stable plan IDs and explicit trigger type (weekly/fixed date/interval). |
| DND | Enable plus start/end time; distinguish “delete interval” from disabling a schedule. |
| Sensors | Device work state, progress, current zone/channel, current cutter state, next schedule, active task IDs, report summary. |
| Events | Command failure, blocked start reason, task completed/result, unreachable zones, report available. |
| Diagnostics | Report detail/timeline and optionally map path; direct blade/debug/report-transfer controls disabled by default. |

Implementation rules:

1. Discover capabilities per product/firmware and preserve raw enum values.
2. Keep ordered zone hashes and plan IDs stable.
3. Treat resume as cancel-pause, not start.
4. Do not optimistically change mower state before device acknowledgement.
5. Preserve unknown plan fields during edit to avoid destructive partial updates.
6. Separate cloud report/history data from live device state.
7. Apply safety confirmation to cutter, stop-without-save, debug, and maintenance
   reset commands.

## Uncertainties requiring wire capture or a second artifact

| Question | What is known | What remains unknown |
|---|---|---|
| Mower schedule `subCmd` integers | Field exists and UI supports create/edit/delete/list. | Exact mower enum numeric mapping is not proven by the pool enum and should not be assumed identical. |
| Route-angle ranges | Two angle fields and custom/efficiency/random strategy are clear. | Accepted degree range, normalization, and distinction between `route_angle` and `demond_angle` on every model. |
| Obstacle-lap semantics | `mowing_laps_obs` and “Nogo” UI are clear. | Whether laps are around no-go islands, detected obstacles, or both on each firmware. |
| `border_mode` labels | 0 maps to localized path-order B, nonzero to A. | Human semantics (perimeter first/last) need strings or runtime capture. |
| Work-config codes 2, 12–20 | Capability flags and fragments reveal features. | Exact code mapping is obscured by incomplete decompilation. |
| Rain behavior | Rain sensor/test exists; planner has no rain field. | Firmware pause/return policy and any global rain setting live outside this scoped task payload. |
| Recharge-and-continue integers | Command getter/setter exists. | Enum meanings and persistence scope. |
| Pool schedule trigger enum values | Fields and CRUD are explicit. | Complete `PLAN_TYPE` numeric mapping was truncated in the generated descriptor excerpt. |
| Units on report timestamps | Names imply timestamps but Java types vary (`Integer`, `Long`). | Epoch unit and timezone need API samples. |

## Files reviewed

The following files were read directly or searched for relevant call sites and
fields. Generated bindings/resources were used only to confirm that a surface
exists, not as primary behavior evidence.

### Work planning and UI

* `work/setting/api/{WorkConfig,WorkSettingBean,WorkingSettingManage,TaskAssignmentManager,CarModelManager}.java`
* `work/setting/activity/{CalendarActivity,JobHistoryActivity,JobScheduleActivity,NewWorkSettingActivity,WorkSettingActivity,WorkSettingMainActivity}.java`
* `work/setting/fragment/{NewSelectAreaFragment,ScheduleCalendarFragment,ScheduleTimepickerFragment,ScheduleWeekFragment,JobHistoryDetailFragment}.java`
* `work/setting/select_area/{SelectAreaBean,SelectAreaFragment,SelectAreaViewModel,SelectAreaAdapter,SelectAreaAdapterNew}.java`
* `work/setting/select_mode/**`
* `work/setting/cutting_height/CuttingHeightFragment.java`
* `work/setting/mow_settings/{MowSettingsFragment,CollectGrassConfigFragment}.java`
* `work/setting/simpleworksetting/fragment/{AutomaticFragment,AutomaticTypeFragment,CuttingHeightSettingFragment,CuttingPathFragment,CuttingRouteFragment,DisplayMapPathFragment,DistanceProcessFragment,FirstPageFragment,FrequencyFragment,MowFrequencyFragment,NogoZoneFragment,PathOderFragment,PerimeterMovingFragment,RideEdgeFragment,SpeedBarFragment,StartProcessFragment,TaskAssignmentFragment}.java`
* `work/setting/viewmodel/{WorkSettingViewModel,JobScheduleViewModel,JobPlanSettingModel}.java`
* `work/setting/{JobPlanDBHelper,JobPlanSettingContract,JobPlanSettingPresenter,WorkSettingHelperImpl}.java`

### Home/runtime controls

* `home/utils/HomeCommandHelper.java`
* `home/fragment/{DeviceStateFragment,DeviceStateSwimmingPoolFragment,DeviceStateSwimmingPoolSPFragment,HomeFragment,HomeFragmentNew}.java`
* `home/viewmodel/{HomeStateViewModule,HomeViewModel,SwimmingToolsViewModule}.java`
* `home/view/SettingCuttingHeightDialog.java`
* relevant `home/databinding/{HomeItemTaskBinding,HomeItemTaskStartBinding,ItemMowingReportCardBinding,ItemNextTaskCardBinding,LayoutDeviceStateTaskBinding,LayoutDeviceStateWorkingBinding,DeviceStateSwimmingPoolFragmentBinding,DeviceStateSwimmingPoolSpFragmentBinding}.java`

### Models, commands, protocols, and reports

* `base_module/db/OperationModeDB.java`
* `base_module/entity/PlanBean1.java`
* `base_module/bean/req/WorkingSettingBean.java`
* all files in `base_module/bean/workreport/`
* `command/{CommandManager,MACommandHelper,MACarDataManagerAPI}.java`
* `command/app/{MACommandApiHelper,MACarDataManager}.java`
* `command/app/contract/{BleJobPlanListener,ReportDeviceListener,SwimmingPoolMapListener}.java`
* `command/entitys/PlanBean1.java`
* `command/menus/{MsgCmdType,PbMsgType}.java`
* generated protocol classes, especially `proto/{SpinoCtrlOuterClass,MctrlSys,MctrlPept}.java`
* `services/` recursively, especially `services/touchmode/**`, to check for
  adjacent work controls.
