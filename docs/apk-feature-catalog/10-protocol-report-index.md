# Mammotion Android APK cross-subsystem protocol/report/API index

## Scope, confidence, and notation

This is a semantic index of reachable first-party operations in the complete decompile at
`/Users/mattjoslin/mammotion-apk-decompile/src`, not a catalog of generated protobuf accessors.
Evidence paths below are relative to that directory. The APK is a Java decompile of Kotlin and
protobuf-lite code; names are reliable where preserved, while numeric arguments and collapsed
switches can be wrong. **[ambiguous]** marks a decompiler-dependent interpretation and
**[unverified]** marks an apparently reachable protocol surface that was not followed to a
successful runtime exchange. “Counterpart” means the closest obvious API in pinned
`pymammotion==0.8.8` or this HA integration; it does not assert wire equivalence.

The principal command envelope is `LubaMsg`:

`msgtype` + `sender` + `receiver` + `msgattr` + oneof
`sys|nav|driver|ota|mul|pept|base|ctrl|net`. Command helpers build one subsystem message, put it
in this envelope, and route the serialized bytes through `MALinkManager`. BLE uses BLUFI custom
data; cloud uses the MA-IoT `device_protobuf_sync_service`. Received bytes converge on
`MACarDataManagerAPI._parseReceivedPBData`, whose subsystem/oneof switches update state or fulfill
a keyed one-shot request. Evidence:
`src/sources/com/agilexrobotics/command/MACommandHelper.java:184-385`;
`src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:920-1324`;
`src/sources/com/agilexrobotics/device/source/links/managers/MALinkManager.java:410-440`;
`src/sources/com/agilexrobotics/device/source/links/managers/MAIotManager.java:633-743`.

## Routing and acknowledgement model

| Layer | Route and semantics | Evidence | pymammotion / HA analogue |
|---|---|---|---|
| Semantic sender | `MACommandHelper` builds subsystem oneofs; `sendOrderMsg_*` supplies command/log type, `needAck`, and optional explicit `MALinkManager`. | `src/sources/com/agilexrobotics/command/MACommandHelper.java:184-385` | `MammotionCommand` plus transport `send_command`; HA `async_send_command` / `async_send_and_wait`. |
| Route selection | `MALinkManager` exposes `NONE`, `BLUETOOTH`, `IOT`; successful BLE is preferred, and disconnect/failure can fall back to IoT. | `src/sources/com/agilexrobotics/device/source/links/MALinkManagerAPI.java:74-129`; `src/sources/com/agilexrobotics/device/source/links/managers/MALinkManager.java:89-181,258-303,546-607` | `BLETransport` and MQTT/cloud transports coordinated by HA. |
| BLE data plane | Protobuf bytes go to `EspBleManager` as BLUFI custom-data subtype 19/GATT. The app targets MTU 200 and has timeout/reconnect handling. | `src/sources/com/agilexrobotics/device/source/links/managers/EspBleManager.java:63-96,1090-1107`; `src/sources/com/agilexrobotics/espressif/BlufiClientImpl.java:1406-1432` | `pymammotion.transport.ble`. |
| Cloud command plane | Serialized protobuf is base64 content in MA-IoT service invoke, identifier `device_protobuf_sync_service`; HTTP route is `/v1/mqtt/rpc/thing/service/invoke`. | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1326-1418`; `src/sources/com/agilexrobotics/maiot_module/utils/Constants.java:18-60`; `src/sources/com/agilexrobotics/maiot_module/api/MaIoTApiService.java:92-98` | pymammotion MA-IoT HTTP command transport. |
| No-ack stream/control | `needAck=false` is used for continuous/manual motion and some fire-and-forget actions. Delivery is transport-level only; state/report feedback is the practical confirmation. | `src/sources/com/agilexrobotics/command/CarRemoteControlManage2.java:156-240`; `src/sources/com/agilexrobotics/command/MACommandHelper.java:1461-1509` | `MammotionCommand.move_*`; HA manual-motion probes use telemetry to verify displacement. |
| One-shot ack | `CommandManager.comObserverFun/fetchCallbackFun` keys `PBResponseInfo` by `PbMsgType.getResuestName(device)` and lifecycle owner; parser publishes matching values to `DeviceStatueUploadMsgManager.singleRequest`. | `src/sources/com/agilexrobotics/command/CommandManager.java:528-634,820-851`; `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:946-1324` | HA `async_send_and_wait`; pymammotion state callback matching. |
| Multipart ack | `mutiResponse` accumulates `BaseMutiResponse` by request name until `isFinishReceive()`, then emits one completion and clears temporary state. | `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:1279-1324` | Map/hash/path assembly in pymammotion reducers. |
| Push/report | Reports update `CarWorkingStateMachineBean`/specialized listeners and are not command acks. MQTT status/property/event callbacks are likewise asynchronous. | `src/sources/com/agilexrobotics/device/source/device/manager/DeviceStatueUploadMsgManager.java:13-87`; `src/sources/com/agilexrobotics/maiot_module/MQTTService.java:314-404` | `DeviceSnapshot`, state reducer, MQTT status/property/event updates. |

## Reachable protobuf device commands

The “wire” column names the semantically relevant oneof/message/fields. Numeric log types are
diagnostic routing tags unless explicitly described as protocol fields.

### Motion, docking, task execution, and work

| App method / operation | Protobuf wire | Route; ack/callback | Gates / user feature | Evidence | Counterpart |
|---|---|---|---|---|---|
| `OperateOnDevice`; remote joystick `send/stop` | NAV/driver operation message **[ambiguous oneof due collapsed sender body]**; direction/speeds/knife arguments | BLE preferred; repeated no-ack control, report telemetry confirms | Mowers; manual drive, blades | `src/sources/com/agilexrobotics/command/MACommandHelper.java:409-412`; `src/sources/com/agilexrobotics/command/CarRemoteControlManage2.java:156-240` | `move_forward/back/left/right`; HA `async_move_*`, `async_stop_manual_motion`. |
| `sendControl(type, action)` | `MctrlNav.NavTaskCtrl` / `todevTaskctrl`, fields `type`, `action`, `result` | BLE/IoT; ack parsed from `todevTaskctrl.result==0` to `NAV_TODEV_TASKCTRL` | Mower work state | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1504-1509`; `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:1238-1256` | mower start/pause/resume/cancel commands. |
| `startJob`, `pauseExecuteTask`, `cancelPauseExecuteTask` | NAV task-control variants | BLE/IoT; expected task-control ack plus work reports | Mowers; start/pause/resume | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1213-1216,1952-1957` | HA `start_task`, mower pause/resume. |
| `stopAndSaveTask`, `stopAndNotSaveTask`, `saveTask`, `closeJob` | NAV `NavTaskCtrl` family **[numeric actions unverified]** | BLE/IoT; first two have `CommandManager` boolean callback | Mowers; end work / preserve history | `src/sources/com/agilexrobotics/command/MACommandHelper.java:515-518,1489-1492,1976-1999`; `src/sources/com/agilexrobotics/command/CommandManager.java:951-973` | HA cancel task; no distinct public save/no-save match obvious. |
| `returnCharge`, `cancelBacktoRecharge`, `autoUnderPile` | NAV task/dock control **[oneof ambiguous]** | BLE/IoT; state reports confirm docking | Mowers; return, cancel return, leave/enter dock | `src/sources/com/agilexrobotics/command/MACommandHelper.java:465-486,1477-1484` | mower dock/return; HA `async_leave_dock`. |
| `breakPointContinue`, `breakPointAnywhereContinue` | NAV continuation control | BLE/IoT; work report/state feedback | Mowers; resume interrupted job | `src/sources/com/agilexrobotics/command/MACommandHelper.java:473-480` | mower resume; route restoration in reducer. |
| `setKnifeHight` | subsystem setting **[oneof ambiguous]**, height field | BLE/IoT; `BleOperateOnListener.onHeightUpdate` report | Height-adjustable mower | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1843-1847`; `src/sources/com/agilexrobotics/command/app/contract/BleOperateOnListener.java:5-13` | HA `async_blade_height`; number entity. |
| `setSpeed` / `getSpeed` | work speed command/response **[oneof ambiguous]** | BLE/IoT; speed callback/report | Mowers | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1125-1132,1881-1892` | HA `async_set_speed`; task route speed. |
| `manualGrassCollection`, `manualPourGrass`; dumping enter/exit/add/recover/revoke | NAV dumping/collection controls **[unverified]** | BLE/IoT; asynchronous state | YUKA sweeper/collector-capable models | `src/sources/com/agilexrobotics/command/MACommandHelper.java:606-615,1175-1187,1209-1212,1304-1307,1485-1488` | No obvious pinned pymammotion public counterpart. |
| `outDropDumpingAdd`, `addDumpPoint`, `revokeDumpPoint` | map/work dump-point operation | BLE/IoT; map/work response | YUKA collection | `src/sources/com/agilexrobotics/command/MACommandHelper.java:417-420,1209-1212,1485-1488` | No obvious counterpart. |

### Mapping, map objects, routes, and deployment

| App method / operation | Protobuf wire | Route; ack/callback | Gates / user feature | Evidence | Counterpart |
|---|---|---|---|---|---|
| `startDrawBorder`, `startDrawBorder431`, `endDrawBorder`, `cancelCurrentRecord` | NAV manual-map/boundary control | BLE normally; report location/edge points; final map/hash response | Mowers, generation-specific 431 branch | `src/sources/com/agilexrobotics/command/MACommandHelper.java:487-490,585-591,1934-1943` | SVG/map upload and map reducer; no identical high-level live-draw API. |
| `startDrawCorridor`, add/end/give-up/recover line or point | NAV corridor controls; common-data ack for completion | BLE/IoT; `CommandManager.giveUpDrawCorridor` gets `CommData` | Mowers; channels/corridors | `src/sources/com/agilexrobotics/command/MACommandHelper.java:413-416,592-595,1145-1148,1296-1303,1944-1947`; `src/sources/com/agilexrobotics/command/CommandManager.java:753-763` | pymammotion SVG channel model; HA SVG services. |
| `startDrawBarrier` | NAV map drawing command | BLE; report stream supplies points | Mowers; no-go boundary | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1930-1933` | SVG no-go zones / map upload. |
| `startErase`, `endErase`, `cancelErase` | NAV erase command | BLE; final common-data/map response | Mowers; map editing | `src/sources/com/agilexrobotics/command/MACommandHelper.java:491-494,596-599,1948-1951` | SVG map editing, no direct live eraser. |
| `deleteAll`, `deleteMapElements(type,hash)` | NAV common-data command; response `toappGetCommondataAck` fields `subCmd,result,action,type,dataHash,paternalHashA/B,totalFrame,currentFrame,data,name/time` | BLE/IoT; one-shot or multipart `CommData` callback | Mowers; delete map/all elements | `src/sources/com/agilexrobotics/command/MACommandHelper.java:539-580`; `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:1125-1190` | pymammotion `build_svg_delete`; HA map delete service. |
| `addManualElementMessage`, `deleteManualElementMessage` | NAV `ManualElementMessage`; `type,shape,subCmd,dataHash,ifHide,reserved,center,width/height,rotateRadius`; ack `toappManualElement` | BLE/IoT; typed callback | New map objects/pattern geometry | `src/sources/com/agilexrobotics/command/MACommandHelper.java:421-430,562-569`; `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:1257-1274` | Partial SVG/map-object support; no exact public counterpart obvious. |
| `setPatternHideOrShow` | NAV manual-element/pattern hash + visibility | BLE/IoT; push/map refresh | Pattern-capable models | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1867-1870` | HA map task visibility handling; no command wrapper obvious. |
| `areaRename`, `getAreaNameList` | NAV area hash/name and `AppGetAllAreaHashName`; response list `hashnames(hash,name)` + `deviceId` | BLE/IoT; callback and state cache | Mowers with zones | `src/sources/com/agilexrobotics/command/MACommandHelper.java:458-464,768-893`; `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:1275-1317` | HA `async_get_area_list`, `async_set_area_name`. |
| `getHashResponse`, `synchronizeHashData` | NAV hash synchronization | BLE/IoT; `HashDataListener` and multipart state | Mowers; map synchronization | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1034-1037,2029-2032`; `src/sources/com/agilexrobotics/command/app/contract/HashDataListener.java:1-10` | pymammotion hash/map reducer; HA `async_sync_maps`. |
| `getLineInfo`, `getLineInfoList` | NAV `todevZigzagAck` (`pver,currentHash,subCmd`) or `appRequestCoverPaths` (`hashList,transactionId,subCmd`) | BLE/IoT; multipart cover-path response | Mowers; render generated paths | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1068-1078` | map route/path synchronization. |
| `GenerateRouteInformation`, query/modify/end | NAV route generation / `NavReqCoverPath`; fields include `toward,towardMode,ultraWave,channelMode,width,jobId/ver/mode,edgeMode,knifeHeight,zoneHashs,reserved(path order),speed,rideBoundaryDistance,appDisplayMode` | BLE/IoT; generated-route callback/string or report; modify/query variants | Mowers; route planning | `src/sources/com/agilexrobotics/command/MACommandHelper.java:386-408,600-605,1188-1202,1247-1252`; `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:1191-1237` | `GenerateRouteInformation`; HA `async_plan_route`, `async_get_plan_route`, `async_modify_plan_route`. |
| `setNavStarPoint`, `synNavStarPointData` | NAV start-point message | BLE/IoT; map/location feedback | Some mower firmware | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1853-1858,2000-2003` | No obvious public counterpart. |
| `startUpdateMap`, `responseEdgewiseMapping`, `setEdgewiseMapping`, `setEditBoundary` | NAV boundary update/edge-point ack | BLE/IoT; `BleSysInfoUpdateListener.editBoundaryRealStatus` | Mowers; edit/optimize boundary | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1473-1476,1823-1830,1972-1975`; `src/sources/com/agilexrobotics/command/app/contract/BleSysInfoUpdateListener.java:5` | SVG update; HA map services. |
| charge-point/base deployment: `confirmBaseStation`, `resetBaseStation`, delete charge point/LD charge point, `startPositioning431All`, `rtk dock location` | NAV task control / base-position map records **[mixed oneofs]** | Usually BLE; task-control ack and localization reports | Mower/RTK generations | `src/sources/com/agilexrobotics/command/MACommandHelper.java:534-538,547-561,1461-1464,1958-1963` | HA `async_relocate_charging_station`, `async_rtk_dock_location`. |

### Schedules, work plans, sleep, and battery

| App method / operation | Protobuf wire | Route; ack/callback | Gates / user feature | Evidence | Counterpart |
|---|---|---|---|---|---|
| `singleSchedule` | NAV single schedule string **[payload schema ambiguous]** | BLE/IoT; `responseSingleSchedule(code,text)` | Mowers; schedule | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1901-1905`; `src/sources/com/agilexrobotics/command/app/contract/BleJobPlanListener.java:9-15` | pymammotion mower plan CRUD; HA task services. |
| `readPlan_SP`, `sendSchedule_SP`, `deletePlan_SP` | CTRL `PlanJobSet` / `PLAN_CMD`, pool plan fields | BLE/IoT; plan listener / `SYS_TO_APP_SP_210_WORKING_PLAN` | SPINO/PC210 pool robots | `src/sources/com/agilexrobotics/command/MACommandHelper.java:581-584,1292-1295,1550-1574`; `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:1297-1315` | HA pool plan create/edit/delete/refresh. |
| `setPC210Plan` | CTRL plan command via `PlanJobSPBean` | BLE/IoT; callback string | PC210/SPINO | `src/sources/com/agilexrobotics/command/CommandManager.java:887-898` | `PoolPlan`, HA `async_create_spino_task` etc. |
| job DND read/set/delete; `setPlanUnableTime` | NAV `NavUnableTimeSet`: `subCmd,trigger,deviceId,unableStartTime,unableEndTime,result,reserved`; trigger 99 interpreted as sunrise interval | BLE/IoT; typed `JobDNDBean` callback and state update | Mowers | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1153-1174,1871-1876`; `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:1198-1237` | HA `async_set_non_work_hours`; schedule model. |
| charging/uncharging sleep get/set | SYS sleep-status messages **[exact oneof ambiguous]** | BLE/IoT; callback/status | Supported mower/pool firmware | `src/sources/com/agilexrobotics/command/MACommandHelper.java:927-930,1141-1144,1741-1744,1893-1896`; `src/sources/com/agilexrobotics/command/CommandManager.java:696-714,910-928` | No direct public counterpart obvious. |
| battery query/set | SYS battery config: smart charge, target progress, peak/valley switch and start/end | BLE/IoT; keyed callback string | Battery-config-capable models | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1241-1246,1691-1698`; `src/sources/com/agilexrobotics/command/CommandManager.java:809-819,854-864` | Device state exposes battery; no obvious pinned write API. |
| recharge threshold / continue-working get/set | NAV `nav_sys_param_msg`, `id` 14/15, `context`, `rw`; response routed to separate request keys | BLE/IoT; keyed string callback | Supported mowers | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1088-1091,1877-1880`; `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:1217-1237` | Generic `read_write_device`; HA has matching route/read-write helpers. |

### Settings, networking, maintenance, and OTA

| App method / operation | Protobuf wire | Route; ack/callback | Gates / user feature | Evidence | Counterpart |
|---|---|---|---|---|---|
| generic read/write settings | Multiple SYS/NAV messages and IDs; includes traversal, turn mode, rain, lights, wildlife, boundary distance | BLE/IoT; settings/report ack | Model capability matrix | `src/sources/com/agilexrobotics/command/MACommandHelper.java:431-453,1286-1291,1704-1740`; `src/sources/com/agilexrobotics/device/source/device/entity/SettingVos.java:11-108` | `MammotionCommand.read_write_device`, `traverse_mode`, `turning_mode`, `boundary_ride_distance`; HA setting methods. |
| animal/wildlife protection all/job read/write | SYS/NAV setting IDs **[exact message ambiguous]** | BLE/IoT; report/settings response | Camera/vision mower variants | `src/sources/com/agilexrobotics/command/MACommandHelper.java:431-453,753-767,1153-1157` | HA `async_set_wildlife_safety` / read. |
| lamps, side/night/manual light | SYS lamp controls; night-light response includes id/result/manual switch | BLE/IoT; setting ack/report | Models with lights | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1286-1291,1704-1718`; `src/sources/com/agilexrobotics/device/source/device/bean/DeviceNightLightState.java:7-51` | HA side/manual/night light methods. |
| audio config, voice language/volume/sex | MUL/audio configuration messages | BLE/IoT; audio report/config response | Speaker-equipped models | `src/sources/com/agilexrobotics/command/MACommandHelper.java:916-922,1719-1736` | HA audio fetch, voice/prompt volume, on/off, gender, language. |
| camera wiper | SYS/MUL wiper command **[oneof unverified]** | BLE/IoT; fire-and-forget/report | Camera models | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1737-1740` | HA `async_run_camera_wiper`. |
| cutter mode and blade warning/reset | SYS cutter/reset messages; `todevResetBladeUsedTime=1` | BLE/IoT; response/report | Mowers | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1465-1468,1514-1517,1575-1578,1699-1703` | HA cutter mode read/set; blade warning time; reset blade time. |
| device Wi-Fi/4G enable, APN, network info/list | NET messages (`DrvWifiList`, `todevBleSync`, IoT controls) and JSON legacy command 69 | BLE for provisioning, BLE/IoT for settings; result listeners | Network hardware dependent | `src/sources/com/agilexrobotics/command/MACommandHelper.java:745-752,998-1001,1092-1104,1665-1668,1767-1822,2039-2048` | HA device Wi-Fi/4G toggles; status telemetry. |
| iNavi/NetRTK link selection and pairing code | SYS set MQTT RTK, NAV/NET link mode, RTK pairing config; response `setRtkModeError` | BLE/IoT; boolean / `RtkParingCode` callback | RTK/iNavi eligible models/account/region | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1267-1285,1669-1690,1831-1866`; `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:946-967` | RTK status models; no complete pairing counterpart obvious. |
| device sync, time sync, MTU, ZMQ, debug/factory/test | SYS/NET diagnostic messages and raw tool orders | Mostly BLE; varied/no ack | Factory/internal; **[unverified production reachability]** | `src/sources/com/agilexrobotics/command/MACommandHelper.java:616-740,1493-1545,1631-1664,1745-1766,1848-1852,1897-1900,2004-2038` | No supported HA counterpart; deliberately diagnostic. |
| base/device reset/restart | NAV reset-base task; SYS reset/restart messages | BLE/IoT; `RestartResultListener` | Maintenance | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1312-1315,1461-1472`; `src/sources/com/agilexrobotics/command/app/contract/RestartResultListener.java:5-7` | HA `async_restart_mower`; base relocation differs from reboot. |
| log query/upload/cancel | SYS log info/upload/socket request; cloud progress event | BLE/IoT + HTTP upload; `UploadLogListener` / MQTT `otaProgress`-like event | Support/feedback | `src/sources/com/agilexrobotics/command/MACommandHelper.java:503-506,994-997,1811-1818`; `src/sources/com/agilexrobotics/feedback/api/FeedbackApiService.java:35-59` | Diagnostics only; no direct public counterpart. |
| mower OTA query/package/finish | OTA oneof (`baseInfo`, `getInfoRsp`, package/status) | BLE/IoT + HTTP version/upgrade; OTA listeners | Firmware/model gated | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1002-1033,1203-1208`; `src/sources/com/agilexrobotics/command/app/contract/BleOTAListener.java:5-13`; `src/sources/com/agilexrobotics/command/app/contract/BleOTAStatusListener.java:9-13` | pymammotion firmware check/update; HA update entity. |
| pool OTA first/package/second/start | Pool OTA messages with package seq and success | BLE; staged callbacks `check`, `sendPkg`, `start` | SPINO/pool | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1595-1624,1964-1971`; `src/sources/com/agilexrobotics/command/app/contract/SwimmingPoolDeviceOTAListener.java:8-12` | HA pool firmware update. |

### Pool cleaner work, map, and settings

| App method / operation | Protobuf wire | Route; ack/callback | Gates / user feature | Evidence | Counterpart |
|---|---|---|---|---|---|
| `sendSwtichSwimmingWorkModule` | CTRL `SwimmingSPWorkModule` | BLE/IoT; pool status report | SPINO/pool | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1625-1630` | HA `async_set_work_mode`. |
| `spEnvironmentUpdate` get/set | SYS `app_downlink_cmd_t`; command, ack, wall material | BLE/IoT; `SwParamRsp`; success ack numbers 1 or 4 | Pool; wall/bottom material **[field mapping partly ambiguous]** | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1906-1915`; `src/sources/com/agilexrobotics/command/CommandManager.java:715-725,929-939`; `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:968-993` | HA `async_set_wall_material`, `async_set_bottom_type`. |
| `spSpeedUpdate` get/set | SYS `app_downlink_cmd_t`; command 2 uses `floorSpeed` | BLE/IoT; `SwParamRsp` | Pool | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1916-1925`; `src/sources/com/agilexrobotics/command/CommandManager.java:742-752,940-950` | HA `async_set_floor_speed`. |
| `getSpMap`, `getSpLine` | pool map/line messages | BLE/IoT; `SwimmingPoolMapListener.mapData/lineData(SwMapInfo)` | Pool | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1109-1124`; `src/sources/com/agilexrobotics/command/app/contract/SwimmingPoolMapListener.java:10-12` | HA `async_fetch_pool_map`, `async_fetch_pool_line`. |
| channel line / dumping / grass operations | pool/mower subtype overlap **[unverified for each model]** | BLE/IoT; state/report | Pool or YUKA depending operation | `src/sources/com/agilexrobotics/command/MACommandHelper.java:606-615,1175-1187,1926-1929` | Partial pool work-mode support; no dumping API. |

## Continuous report configuration

`MctrlSys.report_info_cfg` is the central continuous-report subscription. Fields are:
`act` (`RPT_START` and corresponding stop action), repeated `subValue` report type numbers,
`timeout`, `period`, `noChangePeriod`, and `count`. `getMctrlSysBuilder` is the authoritative
builder. A generic IoT subscription defaults to timeout 5000 ms, period 1000 ms,
no-change period 2000 ms, count 0. The map/location and maintenance helpers instead use timeout
10000 ms, caller-supplied periods, and `DeviceUtils.getCountKeep()`. Evidence:
`src/sources/com/agilexrobotics/command/MACommandHelper.java:1079-1087,1389-1460`.

| Configuration entry point | Included `rpt_info_type` values | Transport / lifecycle | Consumer and counterpart |
|---|---|---|---|
| `requestIot_Sys(act, subvalues, …)` | Caller-selected | SYS `todevReportCfg`; cloud service invoke; callback object receives service result. Start/stop is controlled by `act`. | Generic report subscription; pymammotion report-config command. |
| `requestMapLocationData` | `RIT_CONNECT`, `RIT_RTK`, `RIT_DEV_LOCAL`, `RIT_WORK`, `RIT_DEV_STA`, `RIT_VISION_POINT`, `RIT_VIO`, `RIT_VISION_STATISTIC`, `RIT_BASESTATION_INFO` | IoT, log type 10036 | Primary live map/status stream; HA `async_start_report_stream`. |
| `requestMapLocationBTorIOTData` | Same nine map/location types | Explicit BLE bytes when `isBT`; otherwise cloud IoT service (`setDeviceIotService`, tag 1469) | Unified local/cloud live map feed. |
| `requestMAINITAINData` | `RIT_CONNECT`, `RIT_DEV_STA`, `RIT_VISION_POINT`, `RIT_VIO`, `RIT_MAINTAIN` | IoT, log type 10036 | Maintenance/status screen; pymammotion `Maintain` reducer. |
| Snapshot/finite configuration | Same builder with finite `count`, caller period/no-change period | Transport independent | HA `async_request_report_snapshot`; pymammotion report cfg. |

`requestIOTMessage` is implementation-generation dependent:
`MACommandHelper` subscribes to the nine map/location types above, while
`MACommandApiHelper` subscribes to ten by additionally including
`RIT_CUTTER_INFO`. Both use timeout 10000 ms, period 3000 ms, no-change period
4000 ms, and count 0
(`src/sources/com/agilexrobotics/command/MACommandHelper.java:1326-1332`;
`src/sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1273-1279`).

Incoming `SYS.toappReportData` is `report_info_data`. At minimum the decompiled parser directly
touches `dev`, `connect`, and `rtk`; broader app consumers and generated schema expose the report
families below. The parser’s large switch is partially collapsed, so individual update line
mappings are **[decompiler-ambiguous]** even where the message names are certain.

## Incoming protobuf reports, responses, and listener events

| Incoming event / message | Semantically used fields | Delivery / callback semantics | User feature / gates | Evidence | Counterpart |
|---|---|---|---|---|---|
| `SYS.toappReportData` / `report_info_data` | composite `dev`, `connect`, `rtk`, localization, work, vision, VIO, maintain, base station | Push; state-machine update, not command ack | All connected products; subreports capability-gated | `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:994-1012`; `src/sources/com/agilexrobotics/proto/MctrlSys.java:52130-53500` | pymammotion `ReportData`/`DeviceData`, state reducer, `DeviceSnapshot`. |
| connection report | online/link/network type, Wi-Fi/4G/BLE quality fields | Continuous push | Network hardware dependent | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1423-1460`; `src/sources/com/agilexrobotics/signal/newstatus/SignalHelper.java:621-808` | availability and connectivity entities. |
| device status report | battery, charging, mower/pool system state and errors | Continuous push; drives working state | Product-specific enums | `src/sources/com/agilexrobotics/device/source/device/bean/CarWorkingStateMachineBean.java:18-337`; `src/sources/com/agilexrobotics/device/source/device/manager/DeviceStatueUploadMsgManager.java:13-87` | mower/pool state model and HA entities. |
| localization/RTK/base report | device pose, position mode/quality; RTK link/status; base info | Continuous map push | Mowers/RTK capable | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1432-1460`; `src/sources/com/agilexrobotics/signal/newstatus/SignalHelper.java:761-925` | map position, RTK entities, dock/base location. |
| work report | work mode/status, progress, current area/path | Continuous push | Mower/pool | same report configuration above; `src/sources/com/agilexrobotics/base_module/bean/workreport/ReportDetailResBean.java:1-240` | mower activity, task progress, state reducer. |
| vision point/statistics and VIO | vision obstacle/point/statistical localization data | Continuous push | Vision-camera products | report config at `src/sources/com/agilexrobotics/command/MACommandHelper.java:1432-1460` | pymammotion VIO/vision state; HA diagnostic and motion safeguards. |
| maintenance report | module/maintenance counters and statuses | Continuous push | Supported firmware | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1423-1431` | pymammotion `Maintain`; maintenance coordinator. |
| operation updates | knife state, height, configured/real-time speed, BLE RSSI | `BleOperateOnListener` push callbacks | Mowers | `src/sources/com/agilexrobotics/command/app/contract/BleOperateOnListener.java:5-13` | mower settings and telemetry. |
| live edit-boundary status | status plus `x,y`, mode/type and float value **[parameter names absent]** | `BleSysInfoUpdateListener` push | Map editing | `src/sources/com/agilexrobotics/command/app/contract/BleSysInfoUpdateListener.java:5` | live pose/map editing; no direct listener API. |
| work history and work-report command | `WorkReportCmdData(subCmd=1,getInfoNum)`; history records | request/stream; `BleNavInfo*` updates DB; `ReportDeviceListener` confirms/sends report and work IDs | Mowers; history UI | `src/sources/com/agilexrobotics/command/MACommandHelper.java:1419-1422`; `src/sources/com/agilexrobotics/command/app/contract/BleNavInfoOnListener.java:7`; `src/sources/com/agilexrobotics/command/app/contract/ReportDeviceListener.java:11-15` | HA `async_get_reports`; report history mostly cloud-backed. |
| map/hash/route multipart data | hashes, SVG/map chunks, cover paths, common-data frames | accumulated until complete; then one callback | Mowers/pool depending message | `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:1125-1190,1279-1324`; `src/sources/com/agilexrobotics/command/app/contract/HashDataListener.java:1-10` | pymammotion map/hash/path reducer. |
| network/file upload | `DeviceNetInfo`; power status; file name/type/frame/content **[listener args ambiguous]** | `BleFileToAppListener` callbacks | Diagnostics/network | `src/sources/com/agilexrobotics/command/app/contract/BleFileToAppListener.java:7-11` | diagnostics; no HA entity for raw files. |
| OTA status | base info, info response, module version list, package progress/success | staged listeners, some command-response and some push | Firmware/model gated | `src/sources/com/agilexrobotics/command/app/contract/BleOTAStatusListener.java:9-13`; `src/sources/com/agilexrobotics/command/app/contract/BleModuleVersionListener.java:8` | update coordinator/entity. |
| pool status/map/line/settings | `report_info_t`, `SwMapInfo`, `app_downlink_cmd_t`, work module | continuous status plus keyed settings/map callbacks | SPINO/pool | `src/sources/com/agilexrobotics/command/app/contract/SwimmingPoolMapListener.java:10-12`; `src/sources/com/agilexrobotics/command/MACarDataManagerAPI.java:968-993` | `PoolCleanerDevice`, pool coordinators. |
| send timeout / weak link / reconnect | current `LinkType`, weak-state callback, BLE connect events | transport callback; may trigger route fallback | All | `src/sources/com/agilexrobotics/command/app/contract/BleDataTimeoutListener.java:7`; `src/sources/com/agilexrobotics/command/app/contract/LinkDataStateListener.java:5`; `src/sources/com/agilexrobotics/command/app/contract/BleConnectListener.java:6-10` | transport availability/fallback logic. |

## MQTT topics and payload models

The decompile does not expose a single authoritative full topic template. `TopicUtils` splits by
`/`, takes device name and method by fixed positions, and treats property topics specially;
therefore exact full topic strings are **[unverified]** and should be captured at runtime rather
than inferred. Known suffix/method fragments and models are:

| Topic/event | Payload / dispatch | Semantics | Evidence | Counterpart |
|---|---|---|---|---|
| `/thing/status` | `TopicDeviceStatus {action,productKey,deviceName,iotId,gmtCreate}`; `action=="online"` becomes `iotOnLineEvent` | Device online/offline | `src/sources/com/agilexrobotics/maiot_module/utils/Constants.java:89-101`; `src/sources/com/agilexrobotics/maiot_module/MQTTService.java:331-348`; `src/sources/com/agilexrobotics/maiot_module/mqtt/topic/TopicDeviceStatus.java:11-79` | `ThingStatusMessage`, `StatusType`. |
| property post | `TopicProperty {id,version,sys,method,params,time}` | IoT property update | `src/sources/com/agilexrobotics/maiot_module/MQTTService.java:314-333,374-404`; `src/sources/com/agilexrobotics/maiot_module/mqtt/topic/TopicProperty.java:12-91` | `ThingPropertiesMessage`. |
| log-progress event | same property/event envelope; special params | support-log progress | `src/sources/com/agilexrobotics/maiot_module/utils/Constants.java:89-101`; `src/sources/com/agilexrobotics/maiot_module/MQTTService.java:350-368` | notification/diagnostics only. |
| protobuf-message event | property envelope params contain device protobuf content, decoded and sent to car data parser | Main cloud downlink for reports/responses | `src/sources/com/agilexrobotics/maiot_module/utils/Constants.java:89-101`; `src/sources/com/agilexrobotics/maiot_module/MQTTService.java:374-404`; `src/sources/com/agilexrobotics/device/source/links/managers/MAIotManager.java:473-632` | `ThingEventMessage`; pymammotion protobuf event decoder/state reducer. |
| OTA progress | property/event whose serialized body contains `otaProgress`; params augmented with topic-derived `deviceName` | Firmware progress | `src/sources/com/agilexrobotics/maiot_module/MQTTService.java:350-368` | HA update progress where available. |

Subscriptions use requested QoS, are cached, and are replayed after reconnect. MQTT JWT credentials
come from `/v1/mqtt/auth/jwt`; broker `mqtts` is converted to SSL, keepalive is 60 seconds,
connection timeout 14 seconds, and automatic reconnect is enabled.
Evidence: `src/sources/com/agilexrobotics/maiot_module/MQTTService.java:186-220,482-509`;
`src/sources/com/agilexrobotics/maiot_module/mqtt/MQTTClient.java:236-319,462-530`.

## First-party HTTP/API control-plane index

This table includes device-, map-, work-, media-, and pool-relevant interfaces. Pure account,
forum, analytics, and UI-content routes are omitted unless they affect device reachability.
Relative routes without a leading slash are preserved exactly; their effective host/prefix is
selected by Retrofit `Domain-Name` and is **[deployment-dependent]**.

| App API method / route | Purpose and response semantics | Gates | Evidence | Counterpart |
|---|---|---|---|---|
| MA-IoT auth: `POST /v1/auth/authorization`, region/login/token/refresh, `POST /v1/mqtt/auth/jwt` | Cloud session and broker credentials | Account/region | `src/sources/com/agilexrobotics/maiot_module/utils/Constants.java:18-60`; `src/sources/com/agilexrobotics/maiot_module/api/MaIoTApiService.java:50-94` | pymammotion cloud login/MQTT auth. |
| MA-IoT device: bind, unbind, list, nickname | Account-device inventory/control plane | Owner/share permissions | `src/sources/com/agilexrobotics/maiot_module/api/MaIoTApiService.java:46-58,100-110` | pymammotion account devices. |
| `POST /v1/mqtt/rpc/thing/service/invoke` | Cloud command RPC; async HTTP result, actual device ack may arrive later as MQTT protobuf event | Online MA-IoT device | `src/sources/com/agilexrobotics/maiot_module/api/MaIoTApiService.java:96-98` | cloud command transport. |
| MA-IoT get/set properties | Device shadow/property reads/writes | MA-IoT device capability | `src/sources/com/agilexrobotics/maiot_module/api/MaIoTApiService.java:70-72,104-106` | cloud properties transport. |
| `GET /device-server/v1/device/list`; nickname, setting, wakeup, function | Canonical app device list and server-side capability/settings/wake | Account and model | `src/sources/com/agilexrobotics/home/api/HomeApiService.java:61-67,83-85,108-130` | pymammotion device list; HA device discovery/capability gates. |
| `POST /device-server/v1/device/setting[/info]` | Read/write server-side setting records | Model/firmware | `src/sources/com/agilexrobotics/device/setting/api/DeviceSettingApiService.java:19-26`; `src/sources/com/agilexrobotics/device/info/api/DeviceInfoApiService.java:94-96` | Some settings are device protobuf writes instead; no one-to-one guarantee. |
| work reports: `POST device-server/v1/device/work-report/page`, `/summary/search`, `/detail` | Paginated history, summary, detail; Rx Observable HTTP completion | Authenticated account | `src/sources/com/agilexrobotics/base_module/api/CommonApiService.java:31-48` | report history/data models. |
| firmware: `version/check`, `device/upgrade`, pool `/device-server/v1/pool-robot/version/check` | Version eligibility and cloud-side upgrade start | Product/version | `src/sources/com/agilexrobotics/device/info/api/DeviceInfoApiService.java:39-55` | pymammotion/HA update entities. |
| pool appointed upgrade: `POST /device-server/v1/pool-robot/appoint-version-upgrade` | Select target pool firmware | Pool/internal testing eligibility | `src/sources/com/agilexrobotics/testing/api/ProductTestingApiService.java:25-29` | No normal HA counterpart. |
| map backup CRUD/check/list/progress/recovery | `/device-server/v1/map/backup...` routes | Mower map backup entitlement/ownership | `src/sources/com/agilexrobotics/map/api/MapApiService.java:53-128` | No complete pymammotion backup API obvious. |
| point cloud map + progress | `POST /device-server/v1/device/point/cloud/map`; `GET .../progress/{bizId}` | Vision/point-cloud models | `src/sources/com/agilexrobotics/map/api/MapApiService.java:118-138`; `src/sources/com/agilexrobotics/signal/api/SignalApiService.java:82-89` | HA map/camera features do not expose full cloud generation. |
| lawn patterns | `POST /device-server/v1/lawn-pattern/list` | Pattern-capable mower/account | `src/sources/com/agilexrobotics/map/api/MapApiService.java:128-131` | Map pattern display; no direct command API. |
| map `task`, `subscription`, token | Legacy/auxiliary map task and subscription service **[effective base URL ambiguous]** | Backend generation | `src/sources/com/agilexrobotics/map/api/MapApiService.java:49-51,142-152` | No obvious direct counterpart. |
| device location list/page/setting/sync | Find-device cloud location and sync | Cloud online/location permission | `src/sources/com/agilexrobotics/find/device/api/FindDeviceApiService.java:19-33` | device tracker; state reports are primary live source. |
| video resource `GET video-resource/{deviceId}` | Fetch device media/video resource metadata | Camera-capable models | `src/sources/com/agilexrobotics/home/api/HomeApiService.java:78-81` | HA camera stream APIs; separate Agora/WebRTC setup. |
| feedback `videoInfo`, fault/report, log path/progress/server, 4G upload | Support media and diagnostics upload | User support flow/network | `src/sources/com/agilexrobotics/feedback/api/FeedbackApiService.java:27-59` | diagnostics only. |
| SIM activation/detail/limit and 4G support | `iot/sim/activation`, `/device-server/v1/iot/sim/detail`, `/device/sim/limit/activate` | Cellular SKUs | `src/sources/com/agilexrobotics/signal/api/SignalApiService.java:43-55` | 4G status/toggle entities; activation not exposed. |
| iNavi enable/status/devices/handoff, NetRTK enable, IoT pairing | `/device-server/v1/iot/i-navi/enable`, i-naive legacy spelling, `/invai/devices`, `/device/handoff/iNvavi/box`, `iot/net-rtk/enable`, `iot/device/pairing` | Entitlement, region, model, RTK hardware | `src/sources/com/agilexrobotics/signal/api/SignalApiService.java:58-114` | RTK status and partial settings; pairing gap. |
| pool/reference pairing and encryption | `device/lora-pair/pool[/suggest]`, user-device pool page, `/v2/pool-robot/encryption` | Pool/RTK reference station | `src/sources/com/agilexrobotics/bind/device/api/BindDeviceApiService.java:37-43`; `src/sources/com/agilexrobotics/device/source/device/api/DeviceSourceApiService.java:20-32` | No obvious HA counterpart. |
| share ownership/page/specify/QR/cancel/confirm | `/user-server/v1/share/device/...`; owned-info | Owner/share permissions | `src/sources/com/agilexrobotics/device/share/api/DeviceShareApiService.java:27-71` | pymammotion shared-device inventory/compat handling. |

## Model and capability gates

Protocol reachability is not equivalent to product support. The app creates distinct device
classes for LUBA generations/mini/AWD, YUKA, RTK/base, and SPINO/pool products, and UI callers use
`DeviceType` extension predicates before exposing operations. The strongest gates in this index
are:

- Pool commands/plans/maps/settings: `PoolCleanerDevice`/SPINO/PC210 only.
- Dumping, grass collection, and dump points: YUKA collector/sweeper configurations.
- Vision/VIO/point cloud/media/wiper: camera-equipped generations.
- Height, cutter speed, lamps, 4G/SIM, RTK/iNavi, and wildlife modes: hardware and firmware
  capability predicates, not universal mower settings.
- Cloud RPC/MQTT: authenticated account, IoT identity, region endpoint, and online device.
- BLE: paired/authorized GATT/BLUFI device in range; some ownership checks remain server-side.
- Shared devices may be forced to IoT and may not have owner-only map/settings/upgrade rights.

Evidence:
`src/sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:16-71`;
`src/sources/com/agilexrobotics/device/source/device/extensions/DeviceTypeExtensionsKt.java:1-420`;
`src/sources/com/agilexrobotics/device/source/device/manager/DeviceManager.java:680-940`;
`src/sources/com/agilexrobotics/device/source/links/managers/MALinkManager.java:220-303`.

## Known gaps and decompiler cautions

1. `MACarDataManagerAPI._parseReceivedPBData` explicitly says it decompiled incorrectly and has
   collapsed switch bodies. Message/oneof names and visible field reads are evidence; missing
   branches are not evidence that a report is ignored.
2. Several `MACommandHelper` private sender bodies are collapsed into a large method region.
   Public method names and visible builders prove semantic reachability, but rows marked
   **[ambiguous]** should not be used to assign numeric enum/action values without bytecode or
   runtime captures.
3. `needAck=true` means the app expects response handling, but MA-IoT HTTP success only confirms
   service invocation. Device acceptance is a protobuf response/report and may arrive later.
4. MQTT suffixes and parsing are proven; complete topic templates are not. Do not hard-code a
   guessed prefix.
5. HTTP routes that lack `Domain-Name` or a leading slash may belong to a legacy base URL. They
   are preserved as decompiled and marked deployment-dependent.
6. Factory/test/debug/ZMQ and legacy JSON commands are indexed only as families. Their presence
   does not imply supported consumer use.
