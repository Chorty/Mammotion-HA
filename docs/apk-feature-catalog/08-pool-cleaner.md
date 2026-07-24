# Pool cleaner / SPINO feature catalog

## Scope and reading notes

This catalog covers the complete decompiled Android source tree at
`/Users/mattjoslin/mammotion-apk-decompile/src`, with functional evidence taken
from Mammotion's first-party `com.agilexrobotics` packages. The review began with
case-insensitive matches for `SPINO`, `Swimming`, `PoolRobot`, pool/line/map/work
concepts, charging/docking piles, LoRa, waterline/floor/wall, environment/speed,
OTA, manual control, home state, reports, and protobufs. It found 290 matching
first-party Java files (522 files including generated resources and third-party
false positives). Generated binding/Hilt/R classes and translated strings were
used to confirm UI exposure but are not treated as protocol authority.

Paths below are relative to
`src/sources/com/agilexrobotics/` unless prefixed with `resources/`. Line numbers
refer to this decompile and can move in a different APK/JADX run. Decompiled
names such as `BUTTOM`, `Spion`, `Piar`, `clase`, and `Swtich` are preserved when
they identify actual symbols.

## Executive summary

| Capability | Legacy SPINO / S1 / E1 | SP / PC210 | Home Assistant relevance |
|---|---|---|---|
| Identity | `SWIMMINGPOOL` (9, `Spino`), S1 (19, PC200), E1 (20, PC100) | `SWIMMINGPOOL_SP` (26, PC210); pile is `SD_PX` (27) | Do not infer protocol solely from display name. Use resolved device type/product key. |
| Transport | Local-add pool models are explicitly recognized; BLE/local path is primary and IoT sync is suppressed | Supports normal pool classification and SP-specific/cloud-backed state; command path can still use the local framing variant | Integration should preserve BLE and cloud capabilities separately. |
| Cleaning modes | All, floor, wall, ECO, line | All, floor, wall, water surface, line, custom; repeated sub-modes | Expose a select plus optional sub-mode list for SP. |
| Start parameters | Mode is sent separately; generic start/pause/stop | One command carries operation, sub-modes, target device, speed, suction power | SP service schema should accept speed and operating power atomically with start. |
| Maps | Pool map/line retrieval exists through system downlink commands and callbacks | Dedicated swimming map UI plus area-clean command; cloud state can supplement | Map retrieval/report decoding is viable but map payload format remains partly opaque. |
| Plans | No evidence of `SpinoCtrl` schedules for E1/S1 legacy path | Full add/query/edit/delete/delete-all plans with recurrence and enable state | Strong candidate for calendar entities/services. |
| Environment | Wall material, bottom geometry, floor speed | Same command family is present and SP-specific settings screens route to it | Useful config entities; preserve enum numeric values. |
| Docking | Timed waterline/edge docking command | Exposed in SP settings; paired pile has separate identity and LoRa metadata | Separate dock device/entity and timed docking controls. |
| OTA | Pool-specific three-stage transfer exists | Dedicated PC210/SP correlation, firmware selection and log upload | OTA is complex/high-risk; initially expose version/update availability only. |

## Product and model gates

### Canonical device identities

| Enum | Value | Product-name match | Code name | Observed behavior |
|---|---:|---|---|---|
| `SWIMMINGPOOL` | 9 | `Spino` | `Spino` | Generic/early SPINO; local-add pool protocol. |
| `SWIMMINGPOOL_S1` | 19 | `Spino-S1` | `PC200` | S1 legacy enum. A name-only resolver may intentionally remap this to SP/PC210. |
| `SWIMMINGPOOL_E1` | 20 | `Spino-E1` | `PC100` | E1 legacy enum; local-add pool protocol. |
| `SWIMMINGPOOL_SP` | 26 | `Spino-SP,Spino-S1` | `PC210` | New SP/PC210 protocol, schedules, custom/sub-modes and pile support. |
| `SD_PX` | 27 | `SDPX` | `SDPX` | PC210 charging/docking pile; recognized by name or PC210-pile product key. |

Evidence: `device/source/device/enums/DeviceType.java:23`,
`:33-40`, `:143-165`, `:367-430`, `:713-718`, `:769-786`.

### Important gate behavior and inconsistencies

| Gate/call site | Exact behavior | Consequence |
|---|---|---|
| `isSwimmingPool()` | Includes generic, E1, S1, SP and `SD_PX` | A broad “pool” test also includes the charging pile. Check `isSwimmingPoolChargingPile()` before assuming robot capabilities. |
| `isSupportLocalAddSwimmingPool()` | Generic + E1 + S1 only; excludes SP and pile | Used by command framing/connection logic; it is a protocol-family gate, not merely a UI feature flag. |
| `isSwimmingPool_SP()` | SP only | Best explicit PC210 robot gate. |
| Name resolution | A string containing legacy `SWIMMINGPOOL_S1.product_name` returns `SWIMMINGPOOL_SP` in one resolver | “Spino-S1” can mean PC200 or PC210 depending on product key/resolver. |
| SP enum product names | `Spino-SP,Spino-S1` | Confirms the marketed S1 name spans protocol generations. |
| Display label | SP returns `"Spino S1"`; fallback has typo `"Spion"` | Never use user-facing text for model logic. |
| Command framing | `sendSwtichSwimmingSPWorkModule` uses `sendOrderMsg_Sys` for SP **or local-add pool**, otherwise `sendOrderMsg_Sys2` | Device resolution and link mode affect wire envelope even for the same protobuf body. |
| IoT sync | `sendInitIotDataSync` returns immediately for local-add pool devices | Legacy pools do not participate in normal app IoT-link switching. |

Evidence: `DeviceType.java:453-459`, `:713-718`, `:769-786`;
`command/app/MACommandApiHelper.java:180-206`, `:1612-1620`;
`command/app/MACarDataManager.java:8555-8572`.

### Capability differences

| Feature | Generic / E1 / PC100 | S1 / PC200 legacy | S1/SP / PC210 |
|---|---|---|---|
| `MctrlSys.work_mode_t` | Yes | Yes | Yes |
| Modes 1–5 | Yes | Yes | Yes |
| Mode 6 custom | No positive UI evidence | No positive UI evidence | Yes |
| Repeated sub-modes | No positive evidence | No positive evidence | Yes |
| Start speed and suction/power | Not in legacy mode setter | Not in legacy mode setter | Yes |
| `SpinoCtrlOuterClass` plans | No | No | Yes |
| Dedicated SP home fragment | No | No | Yes |
| Separate pile product/LoRa binding | No positive evidence | Ambiguous | Yes |
| Cloud working-setting overlay | No positive evidence | No positive evidence | Yes |

“No positive evidence” means the feature was not reached from the matching
first-party call sites; it does not prove firmware cannot implement it.

## Cleaning modes and controls

### Mode values

| Value | Legacy `SwimmingWorkModule` | SP `SwimmingSPWorkModule` / `APP_WORK` | UI meaning |
|---:|---|---|---|
| -1 | `UKNOWN_WORK` | `UKNOWN_WORK` | Unknown/not initialized |
| 0 | — | `APP_WORK.IDLE` | Idle (plan enum only) |
| 1 | `AUTO_WORK` (`ALL`) | `ALL` / `AUTO` | Clean all |
| 2 | `BUTTOM_WORK` (`FLOOR`) | `FLOOR` | Floor/bottom |
| 3 | `WALL_WORK` | `WALL` | Wall |
| 4 | `WATER_WORK` (`ECO`) | `WATER_SURFACE` / `ECO` | Naming conflict: legacy calls this ECO; SP UI calls it water surface |
| 5 | `LINE_WORK` | `LINE` | Waterline/line |
| 6 | — | `CUSTOM` | Custom combination |

Evidence: `device/source/device/enums/SwimmingWorkModule.java:12-18`;
`device/source/device/enums/SwimmingSPWorkModule.java:12-19`;
`proto/SpinoCtrlOuterClass.java:58-66`;
`home/fragment/DeviceStateSwimmingPoolSPFragment.java:864-892`.

The value-4 mismatch is integration-significant: report it as a raw mode plus a
model-aware label. Do not globally map `4` to either “ECO” or “water surface.”

### Control commands

| App method / action | Payload | Envelope/log type | Notes and evidence |
|---|---|---:|---|
| Legacy mode switch | `MctlSys.set_work_mode.work_mode` | Sys, 24 | `sendSwtichSwimmingWorkModule`; `MACommandApiHelper.java:1623-1627`. |
| SP mode/start setup | `work_mode_t`: operation/mode, repeated sub-modes, speed, operating power (suction), and device selector | Sys, 24 | `sendSwtichSwimmingSPWorkModule`; local/SP framing gate at `MACommandApiHelper.java:1612-1620`. |
| Generic start | Existing generic `startJob` machinery | varies | Pool home/map screens call common start/pause/end flows; `MACommandApiHelper.java:1910`. |
| Generic close/stop | Common sys close job | varies | `MACommandApiHelper.java:532`. |
| Manual speed | `MctrlDriver.DrvSrSpeed{speed,rw=1}` | Driver, 46 | Clamped to model min/max; `MACommandApiHelper.java:1847-1856`. This is not the SP plan speed field. |
| Read speed | bidirectional driver-speed read | Driver | `MACommandApiHelper.java:1059`. |
| Area clean | `app_downlink_cmd_t`, command 6 | Sys | Proto-supported; no stable public high-level wrapper found. |

The UI has a dedicated `SwimmingMapActivity`, bottom-speed activity,
environment activity, and swimming settings layout. Routes are registered in
the app graph (`MyApplication_HiltComponents.java:143`) and the map route is
declared in `router/RouterHub.java`.

### Manual controls

The reviewed pool map/home activities reuse common job controls rather than
introducing a separate joystick protobuf. A dedicated swimming test screen
exists and can exercise wheels, pump and factory controls, but it is a
manufacturing/debug surface, not proof of a supported consumer API:

- `testing/activity/SwimmingPoolTestToolsActivity.java`
- `resources/res/layout/activity_swimming_test_tools.xml`
- `testing/activity/ChargingPilePairActivity.java`
- `resources/res/layout/activity_charging_pile_pair.xml`

HA implication: expose start/pause/resume/stop and mode selection first. Treat
wheel/pump direct actuation as hidden/unsafe diagnostic functionality.

## Pool environment, speed and waterline docking

### `MctrlSys.app_downlink_cmd_t`

The pool environment protocol is a tagged command with `cmd`, `ack`, and a
one-of parameter. Command IDs and parameter fields are:

| Command enum | Value | Parameter / value type | Intended operation |
|---|---:|---|---|
| `app_wall_material_cmd` | 0 | `wall_material` enum | Set/query wall material |
| `app_bottom_type_cmd` | 1 | `bottom_type` enum | Set/query pool-bottom geometry |
| `app_floor_speed_cmd` | 2 | `floor_speed` float | Set/query floor speed |
| `app_get_map_cmd` | 3 | map transfer/report | Request pool map |
| `app_get_line_cmd` | 4 | line transfer/report | Request waterline/route line |
| `app_docking_time_cmd` | 5 | `DockingTime` | Set/query waterline docking |
| `app_area_clean_cmd` | 6 | area clean parameter | Clean selected area |

Evidence: `proto/MctrlSys.java:32520-32528`, `:33974-34025`; serialized
descriptor at `:75677`.

Acknowledgement intent is explicit: the docking query uses `INQUIRY`, while a
setting uses `WAIT_ACK` (`MACommandApiHelper.java:1875-1881`). The exact numeric
values of the ack enum should be read from the generated enum, rather than
assuming request/response semantics from transport acknowledgements.

### Environment enums

| Type | Numeric values | Evidence |
|---|---|---|
| Wall material | 0 glass; 1 ceramics; 2 sandstone | `proto/MctrlSys.java:75015-75027` |
| Bottom geometry | 0 right-angle/simple; 1 right-angle/complex; 2 curved/simple; 3 curved/complex | `proto/MctrlSys.java:51660-51676` |
| Docking type | 0 disabled; 1 immediate; 2 timed | `proto/MctrlSys.java:43610-43622` |
| Docking time | nested `time.hours`, `time.minutes`, plus `type` | `proto/MctrlSys.java:1947-2868`; builder use at `MACommandApiHelper.java:1875-1876` |

`SPEnvironmentSettingActivity` and `SPButtomSpeedSettingActivity` are concrete,
user-facing settings screens. `WaterlineDockingDetailActivity` exposes timed
docking and is backed by the same query/set method. The local state cache keeps
`isStairsModule`, `isForceModule`, `isWaterlineModule`, docking hour/minute,
wheel status and pump status
(`device/source/device/entity/SwimmingPoolDeviceStatue.java:8-19`, `:21-121`).

HA entities suggested:

- select: wall material
- select: pool-bottom geometry
- number: floor speed (retain firmware min/max when learned)
- switch/select + time: docking disabled/immediate/timed and HH:MM
- diagnostic sensors: stairs/force/waterline module, pump, wheel

## Maps and work areas

| Workflow | App behavior | Evidence / uncertainty |
|---|---|---|
| Open pool map | Routes to `SwimmingMapActivity`, backed by `SwimmingPoolViewModule` and `SwimmingMapViewModel` | `map/swimming/SwimmingMapActivity.java`; `map/viewmodel/SwimmingPoolViewModule.java`. |
| Get SP map | Public command helper `getSpMap()` sends the pool map request | `command/app/MACommandApiHelper.java:1055-1057`. |
| Get map/line | Proto command IDs 3 and 4 | `proto/MctrlSys.java:33977-33980`. |
| Receive map | `SwimmingPoolMapListener` is held by `MACarDataManager` and receives decoded map events | `command/app/MACarDataManager.java:319-320`, `:8812-8813`; `command/app/contract/SwimmingPoolMapListener.java`. |
| Draw/display | `LineDrawingView` and swimming-map activity render path/shape data | `map/swimming/LineDrawingView.java`; `SwimmingMapActivity.java`. |
| Area clean | Proto command 6 and one-of parameter | `proto/MctrlSys.java:33980-33981`; call-site semantics are not fully recovered. |

The app contains both protocol retrieval and rendering, but this review did not
recover a self-contained specification for the binary map/line geometry.
`MctrlSys.MapTrans` and report framing indicate chunked transfer. For HA, retain
raw frames and hashes during implementation, validate ordering/scale against a
captured app session, and only then normalize to SVG/GeoJSON.

## SP/PC210 work plans (`SpinoCtrlOuterClass`)

`SpinoCtrlOuterClass` contains only one control one-of:
`SpinoCtrl.plan_job_set` at field 1. It is not a general pool-control proto;
cleaning and environment controls remain in `MctrlSys`.

Evidence: `proto/SpinoCtrlOuterClass.java:2294-2297`, `:2794-2829`;
receive dispatch in `command/app/MACarDataManager.java:825-829`.

### Plan enums

| Enum | Values |
|---|---|
| `PLAN_CMD` | 0 NULL, 1 ADD, 2 QUERY, 3 DELETE, 4 EDIT, 5 DELETE_ALL |
| `PLAN_ACK` | 0 success, 1 fail |
| `PLAN_TYPE` | 0 WEEK, 1 DAY, 2 DATE, 3 RUN |
| `APP_WORK` | 0 idle, 1 auto/all, 2 floor, 3 wall, 4 eco/water-surface, 5 line, 6 custom |

Evidence: `proto/SpinoCtrlOuterClass.java:58-66`, `:157-160`,
`:235-242`, `:333-338`.

### `PlanJobSet` wire fields

| Field | No. | Wire/API type | App meaning / HA mapping |
|---|---:|---|---|
| `cmd` | 1 | enum/int | Add/query/delete/edit/delete-all |
| `work_mode` | 2 | enum/int | Primary cleaning mode |
| `sub_mode` | 3 | repeated enum/int | Custom/combined cleaning stages |
| `user_id` | 4 | string | Owner/account identifier |
| `device_id` | 5 | string | Target robot identifier |
| `start_time` | 6 | `fixed32` / Java `int` | Plan start time; the app casts `PlanJobSPBean.startTime` from `long` to `int` before serialization |
| `total_plan_num` | 7 | int32 | Query pagination/total count |
| `plan_index` | 8 | int32 | On-device plan index |
| `result` | 9 | enum/int | Ack/result |
| `speed` | 10 | float | Cleaning speed |
| `operating_power` | 11 | float | Suction/operating power |
| `job_name` | 12 | string | User-visible name |
| `job_id` | 13 | int64 | Stable job identifier |
| `start_date` | 14 | string | Validity start |
| `end_date` | 15 | string | Validity end |
| `trigger_type` | 16 | int/plan type | Week/day/date/run recurrence |
| `day` | 17 | `fixed32` / Java `int` | Day selector; preserve unsigned 32-bit wire encoding |
| `weeks` | 18 | repeated `fixed32` / Java `int` | Weekday selectors; preserve unsigned 32-bit wire encoding |
| `remained_seconds` | 19 | signed `int64` | UI uses the value for next-execution ordering; the exact seconds-unit semantic remains medium confidence |
| `enable` | 20 | `fixed32` / Java `int` | Inverted application state: enabled `true → 0`, false `→ 1` |

Evidence: constants at `proto/SpinoCtrlOuterClass.java:421-443`;
app persistence mirror at `base_module/entity/PlanJobSPBean.java:36-84`.

### Plan command flow

| Operation | Framing / behavior | Evidence |
|---|---|---|
| Query one/index | `PLAN_CMD.QUERY` + `plan_index`; send SP ctrl log type 16 | `command/MACommandHelper.java:1292-1293`; API variant `MACommandApiHelper.java:1239-1240`. |
| Add/edit | App fills all applicable fields from `PlanJobSPBean`; send log type 68 | `MACommandApiHelper.java:1537-1559`; helper variant `MACommandHelper.java:1551-1572`. |
| Delete | `PLAN_CMD.DELETE` + `job_id`; log type 17 | `MACommandApiHelper.java:597-598`. |
| Delete all | Same method with `DELETE_ALL` (job ID may remain default) | Enum/call contract; validate firmware behavior before exposing. |
| Receive | Parse `LubaMsg.ctrl.plan_job_set`, map to `PlanJobSPBean`, persist/query list | `command/MACarDataManagerAPI.java:1285-1289`; `base_module/entity/PlanJobSPBean.java`. |
| Home next-plan | Read plan records, sort by `remainedSeconds`, render next time | `home/fragment/DeviceStateSwimmingPoolSPFragment.java:746-795`. |

The app converts `enabled` to a backend `enable` integer with inverted-looking
semantics (`true -> 0`, `false -> 1`) at
`DeviceStateSwimmingPoolSPFragment.java:762-778`. HA should keep the protobuf
boolean and cloud integer representations distinct.

## Home status and reports

### Local pool status model

| Cached field | Meaning |
|---|---|
| `sys_status` | Robot/system state |
| `charge_status` | Charging state |
| `bat_val` | Battery percent/value |
| `wheel_status` | Drive/wheel state |
| `pump_status` | Pump state |
| `work_mode` | Current pool mode |
| `isSingHint` | Sound/single-hint capability/state |
| `isStairsModule` | Stair capability/state |
| `isForceModule` | Force module capability/state |
| `isWaterlineModule` | Waterline module capability/state |
| `waterlineDockHour/minute` | Configured dock time |

Evidence: `device/source/device/entity/SwimmingPoolDeviceStatue.java:8-19`,
`:21-121`.

`DeviceStateSwimmingPoolFragment` is the legacy home surface;
`DeviceStateSwimmingPoolSPFragment` is the PC210 surface. The latter merges
device/link state with cloud `WorkingSettingResBean.detailVos`, permits a
background-data fallback, displays mode, next plan, connection state and
controls, and only shows BLE-success UI when the selected link is Bluetooth
(`DeviceStateSwimmingPoolSPFragment.java:819-856`, `:897-914`).

### Core protobuf reports relevant to pool devices

| Message/report | Fields or cases | HA relevance |
|---|---|---|
| `SysWorkState` | `deviceState` 1, `chargeState` 2, `cmHash` 3, `pathHash` 4 | Main state and map/path invalidation. `MctrlSys.java:31296-31303`. |
| `SysBatUp` | `batVal` | Battery sensor. Serialized descriptor at `MctrlSys.java:75677`. |
| `work_mode_t` / set-work-mode | current operation/mode, sub-mode list, speed/power/device selectors | Current mode and start parameters. `MctrlSys.java:8324`, `:17047`. |
| `app_downlink_cmd_t` ack/report | command, ack/result and one-of environment/map data | Setting readback and map transfer. `MctrlSys.java:32520-32528`. |
| `SpinoCtrl.plan_job_set` | all plan fields above | Schedule readback. `SpinoCtrlOuterClass.java:421-443`. |
| `rpt_lora` | scan code, channel, locid, netid, connection status | Dock/pile link diagnostics. Serialized descriptor at `MctrlSys.java:75677`. |
| `device_fw_info` / module info | result, version, repeated module identify/version | Pool firmware/version sensors. Same descriptor. |
| `SpinoSysStatus` | idle, pause/end go-charge, working, leave dock and other system statuses | State-machine mapping. `MctrlSys.java:21818-21875`. |

`MACarDataManager` is the main decoder/fan-out point. It handles Luba top-level
`SYS`, `NAV`, `NET`, `DRIVER`, `CTRL`, `OTA`, base and media cases; the CTRL
branch currently has only `PLAN_JOB_SET`
(`command/app/MACarDataManager.java:760-829`). `MACarDataManagerAPI` duplicates
the plan parsing path for explicit manager instances
(`command/MACarDataManagerAPI.java:1285-1326`).

HA should retain unknown enum values and raw report IDs. Generated protobuf
`UNRECOGNIZED` values demonstrate forward-compatibility is expected.

## Charging/docking pile and LoRa pairing

### Data model

`SD_PX` is the PC210 pile product. Local pair persistence uses
`SpinoPiarInfoDB`, keyed by robot/device name and storing at least the dock
device and LoRa number. Pair list responses contain records and are replaced as
a unit in local storage.

Evidence:

- `device/source/device/enums/DeviceType.java:40-41`, `:159-165`
- `base_module/helper/CommonDBHelper.java:333`, `:525`, `:627`, `:685-705`
- `signal/newstatus/SpinoPileSettingActivity.java:264-269`

### Pairing workflow

1. Discover/resolve the robot and candidate pile.
2. Query suggested pile or submit a pair request through the task backend.
3. Backend APIs use authenticated map bodies (`swimmingSuggestPairPile`,
   `swimmingPairPile`, `loraPairPool`).
4. Device-side pairing/config uses LoRa identifiers and local command
   `LoraCfgReq/LoraCfgRsp` / pairing deletion.
5. Fetch pair records and persist `SpinoPiarInfoDB`.
6. Pile settings display paired device and LoRa number.
7. Unbind sends `deletePairing()`, calls cloud pair deletion, removes pair DB
   and relevant local device rows.

Evidence:

- `bind/device/api/BindDeviceApiUtils.java:21-25`
- `bind/device/api/BindDeviceApiService.java:39-43`
- `device/source/device/api/DeviceSourceApiService.java:22-32`
- `signal/newstatus/SpinoPileSettingActivity.java:98-103`, `:162`, `:183`,
  `:243`, `:264-269`, `:299-316`
- `device/info/api/DeviceInfoApiService.java:91`
- `device/info/api/DeviceInfoApiUtils.java:38`
- LoRa protobuf descriptor: `proto/MctrlSys.java:75677`

The exact REST paths are partly supplied by Retrofit/base-URL configuration and
are not visible as annotations on every decompiled method. API labels include
`loraPairPool`, `swimmingPairPileDelete`, and the task base URL. Do not hardcode
method labels as URL paths without a network capture.

## Pool OTA

### API and transfer stages

| Stage | App method / data | Evidence |
|---|---|---|
| Check | `checkPoolRobotVersion(PoolRobotVersionCheckReq)` returns `PoolRobotVersionCheckResp` | `device/info/api/DeviceInfoApiService.java:46`; request/response beans. |
| Correlate modules | `SwimmingPoolOtaCorrelation` selects/checks compatible robot/pile/module versions | `device/info/SwimmingPoolOtaCorrelation.java` and coroutine companions. |
| Begin | version/module/size/checksum/name metadata | `MACommandApiHelper.sendSwimmingPoolDeviceOtaFirst`: `:1582-1589`. |
| Transfer | indexed integer-byte package chunks with module/count metadata | `sendSwimmingPoolDeviceOtaPackage`: `:1590-1600`. |
| Finish/verify | terminal metadata/checksums | `sendSwimmingPoolDeviceOtaSecond`: `:1601-1611`. |
| Observe | `SwimmingPoolDeviceOTAListener` receives progress/result | `command/app/contract/SwimmingPoolDeviceOTAListener.java`; manager setter `MACarDataManager.java:8808-8809`. |
| Upgrade/log helper | SP-specific firmware send and device-log upload | `device/info/SwimmingPoolSPUpgradeAndUpLogHelperImpl.java`; `SwimmingPoolUpgradeAndUpLogHelper.java`. |
| Cloud mark/start | generic `device/upgrade` API and `startSwimmingPoolDeviceOta` | `DeviceInfoApiService.java:48-50`; `command/MACommandHelper.java:1964`. |

The generated OTA/error enums include frame-size, total-size, sequence, MD5,
file-count, file-get and unknown failures (`proto/MctrlSys.java:43693+`).
OTA requires ordered transfer, retries, checksums, module compatibility and
foreground link stability. HA should initially expose installed versions and
update availability; initiating firmware transfer should remain opt-in until
captured and tested against every model family.

## REST/cloud objects

| Object | Relevant fields/purpose | Source |
|---|---|---|
| `GetSwimmingPairPileReq/Resp` | Retrieve robot/pile associations | `base_module/bean/req/GetSwimmingPairPileReq.java`; corresponding response classes |
| `SwimmingBindPileResp` | Pair operation result and pile object | base-module and device-source variants |
| `SwimmingSuggestBindPileResp` | Suggested pile candidate | `base_module/bean/resp/SwimmingSuggestBindPileResp.java` |
| `SwimmingDeletePileResp` | Unbind/delete result | `base_module/bean/resp/SwimmingDeletePileResp.java` |
| `SpinoLoraRespon` | Pair records, robot/pile/LoRa identifiers | base-module and device-source variants |
| `PoolRobotVersionCheckReq/Resp` | Pool OTA availability and package metadata | `base_module/bean/req` and `bean/resp` |
| `SwimmingVersionBean` | Module/version tuple | `base_module/bean/req/SwimmingVersionBean.java` |
| `WorkingSettingResBean` | Cloud working-state detail values used by SP home | consumed at `DeviceStateSwimmingPoolSPFragment.java:897-914` |

Several packages contain duplicate DTOs with the same conceptual names. Their
serialized field names/types must be compared before sharing one HA decoder.

## Hidden, debug and incomplete features

| Feature | Evidence | Assessment |
|---|---|---|
| Area-selective cleaning | `app_area_clean_cmd` (6) | Protocol-visible but no complete consumer workflow recovered; likely unfinished/hidden. |
| Custom multi-stage cleaning | mode 6 plus repeated sub-modes | Clearly active for PC210 schedules/start; candidate for advanced HA service. |
| Water-surface mode | SP mode 4 while legacy labels 4 ECO | Active but model-sensitive. |
| Stair/force/waterline modules | Cached integer flags | Hardware capability/status appears supported; UI exposure may be conditional. |
| Direct wheel/pump tests | Swimming test tools and state fields | Factory-only; do not expose by default. |
| Charging pile factory pairing | `ChargingPilePairActivity` | Testing route separate from consumer LoRa pair flow. |
| Pool encryption bootstrap | `getPoolEncryption` API | Local protocol likely needs server-supplied key material; `DeviceSourceApiService.java` and `MACarDataManager.java:1555`. |
| Cloud background state | SP home can force/use backend detail values | Useful when robot is unreachable, but freshness/source must be surfaced. |
| Map and line hashes | `SysWorkState.cmHash/pathHash` | Can drive conditional refresh even before map payload is fully decoded. |

## HA implementation priorities

1. Resolve product type robustly from product key + device name; keep robot and
   `SD_PX` pile as separate devices.
2. Decode battery, system/charging state, work mode, wheel/pump/module flags and
   map/path hashes. Include source (BLE/IoT/cloud) and timestamp.
3. Implement safe controls: start, pause/resume, stop, mode; add SP sub-modes,
   speed and suction power only for PC210.
4. Add environment read/write with acknowledged enum values and timed docking.
5. Add PC210 plan query before mutation; preserve job ID, plan index,
   recurrence, enable polarity, speed and power exactly.
6. Model pile association and LoRa diagnostics without silently invoking cloud
   bind/unbind operations.
7. Capture map/line and OTA traffic before implementing either write-heavy
   workflow.

## Uncertainties and cautions

- The APK is a decompile, not original source. Some switch bodies, generic
  signatures and Kotlin metadata are clearer than reconstructed Java bodies.
- PC210 is marketed/resolved as both `Spino-SP` and `Spino-S1`. Product name
  alone cannot distinguish it from PC200 S1.
- `isSwimmingPool()` includes `SD_PX`; broad pool gates can accidentally offer
  robot controls to a pile.
- Mode 4 has incompatible labels (`ECO`, `WATER_WORK`, `WATER_SURFACE`).
- The map transfer, selected-area payload, all system-state numeric meanings,
  and some report subscription IDs require live packet validation.
- REST annotations are incomplete in decompiled interfaces; logical API names
  are not guaranteed literal URL paths.
- Plan cloud conversion appears to invert enable (`protobuf true` to backend
  integer `0`). Verify with create/disable/readback before writing schedules.
- Generic mower charging, RTK and LoRa code was reviewed when reached from a
  pool gate but is not cataloged as a pool feature unless a SPINO call site or
  pool DTO/proto ties it to the cleaner.

## Reviewed files

The exhaustive keyword inventory consisted of 290 first-party matching files.
The following functional groups were inspected; generated Hilt, factory,
databinding and `R.java` companions in the same packages were checked for
routes/layout exposure but omitted from this list for readability.

| Group | Reviewed source files |
|---|---|
| Identity/model | `device/source/device/enums/DeviceType.java`; `extensions/DeviceTypeExtensionsKt.java`; `utils/DeviceUtils.java`; `utils/DeviceVersionUtils.java`; `bean/SwimmingPoolDevice.java`, `SwimmingPoolDeviceE1.java`, `SwimmingPoolDeviceS1.java`, `SwimmingPoolDeviceSP.java`; `entity/SwimmingPoolDeviceStatue.java`; `interfaces/ICarDevice.java`; `manager/DeviceManager.java` |
| Protocol | `proto/SpinoCtrlOuterClass.java`; pool/LoRa/work/map/OTA portions of `proto/MctrlSys.java`; top-level dispatch in `proto/LubaMsgOuterClass.java`; `command/menus/PbMsgType.java` |
| Commands/reports | `command/MACommandHelper.java`; `command/app/MACommandApiHelper.java`; `command/MACarDataManagerAPI.java`; `command/app/MACarDataManager.java`; `command/CommandManager.java`; `command/listener/SwimmingSettingListener.java`; `command/app/contract/SwimmingPoolMapListener.java`, `SwimmingPoolDeviceOTAListener.java` |
| Home/state | `home/fragment/DeviceStateSwimmingPoolFragment.java`; `DeviceStateSwimmingPoolSPFragment.java` and schedule coroutine companion; `home/view/SwimmingPoolHomePop.java`; `home/viewmodel/HomeStateViewModule.java`; `HomeViewModel.java`; `SwimmingToolsViewModule.java`; `device/source/device/bean/CarStateMachineBean.java`, `CarWorkingStateMachineBean.java` |
| Maps/settings | `map/swimming/SwimmingMapActivity.java`; `LineDrawingView.java`; `map/viewmodel/SwimmingMapViewModel.java`, `SwimmingPoolViewModule.java`; `map/activity/SPEnvironmentSettingActivity.java`, `SPButtomSpeedSettingActivity.java`; `device/setting/activity/WaterlineDockingDetailActivity.java`, `CarSettingDrawerActivity.java`; `device/setting/fragment/appsetting/CarSettingDrawerFragment.java`, `DrawerSettingsViewModel.java` |
| Plans | `base_module/entity/PlanJobSPBean.java`; `home/fragment/DeviceStateSwimmingPoolSPFragment.java`; plan send/receive sites in both command helpers/managers |
| Pile/LoRa | `signal/newstatus/SpinoPileSettingActivity.java`, `SignalConnectionHomepageActivity.java`, `SignalHelper.java`; `base_module/helper/CommonDBHelper.java`; `device/source/device/db/SpinoPiarInfoDB.java`; all three first-party `SpinoLoraRespon.java` variants; `base_module/event/SpinoSupportBindEvent.java`, `LoraEvent.java`; pair request/response DTOs; bind/device-source/device-info API services and utils |
| OTA | `device/info/SwimmingPoolOtaCorrelation.java` and coroutine companions; `SwimmingPoolSPUpgradeAndUpLogHelperImpl.java`; `SwimmingPoolUpgradeAndUpLogHelper.java`; `SwimmingPoolSendFirmwareListener.java`; `api/SwimmingPoolSPUpgradeAndUpLogHelper.java`; `DeviceInfoApiService.java`, `DeviceInfoApiUtils.java`; pool version request/response beans |
| Binding/onboarding | pool branches in `bind/device/scan/DeviceBindingHelper.java`, `ScanAndBindDeviceHelper.java`, scan view models, `bind/device/sw/BluetoothPairingActivity.java`, `bind/device/utils/BindDeviceUtil.java`, `bind/device/select_type/SelectDeviceTypeViewModel.java` |
| Test/hidden | `testing/activity/SwimmingPoolTestToolsActivity.java`; `ChargingPilePairActivity.java`; `DeviceTestToolsActivity.java`; `FactoryTestActivity.java`; `testing/api/ProductTestingApiService.java`; corresponding layouts |
| Resources/routes | `router/RouterHub.java`; pool routes in generated ARouter groups; `resources/AndroidManifest.xml`; base English/default pool strings; `activity_swimming_map.xml`, `device_state_swimming_pool_fragment.xml`, `layout_device_state_swimming.xml`, `layout_swimming_pool_setting.xml`, `activity_car_setting_swimming.xml`, `activity_spino_pile_setting.xml`, `activity_waterline_docking_detail.xml`, pool selector drawables and test/pair layouts |
