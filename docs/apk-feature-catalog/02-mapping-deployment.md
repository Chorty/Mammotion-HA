# Mammotion Android APK feature catalog: mapping and deployment

## Scope and confidence

This catalog is a static-analysis reconstruction of the decompiled Android application under `/Users/mattjoslin/mammotion-apk-decompile/src`. It covers `com/agilexrobotics/map`, `com/agilexrobotics/device/deploy`, Mapbox rendering, map-related command/protobuf/model/database code, the manifest, and relevant resources. Line numbers refer to the decompiled tree as inspected on 2026-07-22.

The application contains overlapping implementations for several device generations. Names such as LUBA, YUKA, LUBA 2, LUBA mini/VA/HM/ME/LA, CM900, MN231, PC210 and swimming-pool/SP devices appear in gates. Decompiled control flow is sometimes damaged or obfuscated; a command name is high-confidence evidence that a capability exists, but not always enough to recover every numeric field meaning.

Confidence labels:

- **High**: UI path and command/API implementation are both visible.
- **Medium**: behavior is clear but one layer is inferred from names, resources, or callbacks.
- **Low/uncertain**: dormant/test code, obfuscated branches, or an incomplete decompilation prevents confirmation.

## Architecture and map lifecycle

The app is not the authoritative map store. The mower supplies map-element and route hashes plus element payloads; the app caches geometry in LitePal tables, draws that cache through Mapbox, and requests only missing or changed hashes. Cloud services separately support named map snapshots and generated point-cloud maps.

```text
Robot navigation firmware
  |  NAV_TOAPP_GETHASH_ACK / SVG / common-data / position / dynamic line
  v
MACarDataManager -> HashDataManager
  | compare aggregate MurMur hashes and per-element hashes
  | request missing regions and route lines; retry failed chunks
  v
MapElementDB + LineListDB + hash/order/time metadata
  |
  v
MapDataImpl / AreaDBHelper -> MapBoxManager -> Mapbox layers/sources
  |
  +--> selection/accessibility validation -> work-plan creation
  +--> edit/create commands -> robot -> new hashes -> reconciliation loop
```

Creation/edit lifecycle:

```text
preflight/self-check
  -> choose manual/automatic creation or element type
  -> remote-control mower / automatic scan
  -> start boundary, obstacle/no-go, channel or safety-zone command
  -> live position/edge points update preview
  -> finish, cancel, undo, or discard
  -> firmware validates/generates element
  -> hash/SVG refresh
  -> local DB update and redraw
  -> optional rename, route settings, backup, or work-plan creation
```

Deployment/relocation lifecycle:

```text
installation guide
  -> device-family-specific RTK/dock instructions
  -> charging and sensor/self-check gates
  -> remote-control positioning
  -> reset/relocate charging pile or RTK calibration
  -> wait for charging/base-position acknowledgement
  -> map refresh; existing-map compatibility may require correction/restore
```

Primary evidence: `command/app/HashDataManager.java:45-105,160-247`; `map/MapDataImpl.java:24-154`; `command/menus/PbMsgType.java:118-180`; `map/activity/MapManualActivityNew.java:950-1007,2549-2901,3400-3414`; `device/deploy/device/ui/creatmapprepare/CreateMapCheckSelfActivity.java:1017-1163`.

## Exhaustive feature table

“HA relevance” describes practical Home Assistant integration value. **Direct** means a command/state is a plausible integration primitive; **indirect** means useful for diagnostics, entities, or map presentation; **app/cloud-only** means difficult to reproduce without private APIs or rendering logic.

| Area / feature | User-visible behavior | Gates / conditions | Command, protocol, API or data | Key classes / methods | Evidence | HA relevance / confidence |
|---|---|---|---|---|---|---|
| Map home | Displays mower, dock, areas, channels, routes, progress and status; opens work, settings, edit and manual-control surfaces | Activity variant chosen by family/orientation: standard, video, landscape, MN231 | Position/nav reports; cached map DB; Mapbox sources/layers | `MapActivity`, `MapActivityLandNew`, `MapVideoActivity`, `MapActivityLand231`, `HomeMapFragment`, `MapBoxManager` | `map/activity/MapActivity.java:123-205,573-597`; `map/activity/MapActivityLandNew.java:751-774`; `map/mn231/MapActivityLand231.java:569-626` | Indirect: map/status dashboard. High |
| Initial map sync | Requests robot map state after connection and populates local cache | BLE and IoT paths; compatible fallback for older devices | `sendInitBTDataSync`, `sendInitIotDataSync`, `requestMapLocation*`; `NAV_TOAPP_GETHASH(_ACK)` | `MACarDataManager`, `HashDataManager` | `command/app/MACarDataManager.java:8424-8508,8535-8576`; `command/menus/PbMsgType.java:147-150` | Direct: map-refresh service. High |
| Hash reconciliation | Compares robot region/line hash lists with DB; fetches changed/missing payloads | Per-device DB; update serialized to avoid concurrent map/line loads | MurMur aggregate hash, per-element hashes, common-data requests | `HashDataManager.getDBPathHash`, `getHashAre`, `updateMapElementDB`, `getHashLineNew` | `command/app/HashDataManager.java:213-247,255-300`; `map/MapDataImpl.java:73-151` | Direct/indirect: reliable incremental map fetch. High |
| Hash retry/recovery | Retries bad region and line loads up to 10 times; emits progress/timeouts; can mark update failed | Retry timers 12333/12334/100001 and route-generation timers 86018-86021 | Re-request common data / hash line | `HashDataManager.Handler`, `UpdateLineProgress` | `command/app/HashDataManager.java:99-158,176-183` | Direct: transport resilience model. High |
| Dynamic route sync | Periodically requests changing route while mower is working; distinguishes absent/pending/available status | Only working state and currently selected device | Dynamic line command/status; path hash | `HashDataManager.getDynamicsLineCommand`, handler 100003 | `command/app/HashDataManager.java:127-136,194-205` | Direct: live path entity/overlay. Medium |
| Map multipart ordering/correlation | Persists region and line order, timestamps, visibility lists, selected path hash and transaction id; payload positions establish ordering and multipart correlation. The evidence does not establish one universal “map frame” abstraction. | Device-specific cache; MN231 has a separate deletion path | `mapHashOrder`, `mapHashLineOrder`, `mapHashTime`, hash+position DB methods | `HashDataManager`; `MapDataImpl.deleteMapElementDB231`, `updateHashDB` | `command/app/HashDataManager.java:61-97`; `map/MapDataImpl.java:25-61,145-153`; `command/MACommandHelper.java:1073-1076` | Indirect: preserve topology/order, not just polygons. High |
| Dynamic JSON frames | Parses numbered object members (`num` or `connect_roads_num`, e.g. prefix0…N) | Defensive empty/error handling | Gson dynamic-list parser | `MapDataParser.parseDynamicList` | `map/utils/MapDataParser.java:25-51` | Indirect: useful when decoding map payload JSON. High |
| Mapbox geometry | Draws areas, channels, routes, obstacles, mower and dock with device-specific symbols/styles | YUKA vs LUBA VA/HM style branches; multiple portrait/landscape setups | GeoJSON sources, line/fill/symbol layers | `MapBoxManager`, `MapDrawUtil`, `MapBoxManagerUtils`, `MapRectificationStyleSetup` | `map/mapbox/MapBoxManager.java:3467-3483`; `map/mapbox/MapBoxTag.java:1-188`; `map/mapbox/MapColorTag.java:1-59` | App-only rendering reference. High |
| Camera/map rectification | Lets user pan/rotate/scale or rectify map presentation and persist/display a correction value | Dedicated activity/view; also carried in backup requests | Map transform/correction value; no mower geometry mutation confirmed | `MapRectificationActivity`, `MapRectificationView`, `MapRectificationStyleSetup` | `map/activity/MapRectificationActivity.java:31-233`; `map/view/MapRectificationView.java:47-315`; `map/viewmodel/BackupsViewModule.java:1398-1410` | Indirect/app-only. Medium |
| Manual mapping entry | User chooses “map manually,” then drives mower around a work area | Connection, positioning, state and self-check gates; family-specific activity | Remote motion + draw-boundary start/end | `SelectCreateModeFragmentNew`, `MapManualActivityNew`, `CreateMapCheckSelfActivity` | `map/activity/MapManualActivityNew.java:3400-3414,2873-2901`; `device/deploy/device/ui/creatmapprepare/CreateMapCheckSelfActivity.java:1040-1163` | Direct command candidate, high safety burden. High |
| Automatic map creation | Offers automatic creation/scanning and progress/tips; can fall back to manual handling | Visual/radar-capable families and feature flags; positioning/sensor readiness | Automatic-generation/manual-generate messages; map progress | `AutoDrawMapFragment`, `AutomaticCreationTipsFragment`, `MapCreateViewModel` | `map/fragment/AutoDrawMapFragment.java:52-95`; `map/fragment/AutomaticCreationTipsFragment.java:30-76`; `command/menus/PbMsgType.java:179-181` | Direct but device-gated and safety-sensitive. Medium |
| Boundary/work-area creation | Starts perimeter recording while user drives; previews points and closes/saves boundary | Adequate positioning; minimum geometry/area checks; cannot conflict with work state | `NAV_TODEV_DRAW_BORDER`, `NAV_TODEV_DRAW_BORDER_END`, `NAV_TOAPP_OPT_BORDER_INFO`, ACK | `PlanMapLandFragment`, `PlanMapLand231Fragment`, `BoundaryPointsUtils` | `command/menus/PbMsgType.java:121-136`; `map/fragment/PlanMapLandFragment.java:2670-2860`; `map/utils/BoundaryPointsUtils.java:15-161` | Direct: create-area command, safety confirmation required. High |
| Boundary ride/edge mode | Drives along edge and renders collected edge points; supports directional guidance | Mapping state and location quality | Edge command and nav edge-point acknowledgements | `PlanMapLandFragment`, `DirectionalGuideFragment`, `NavEdgePointsAckBean`; packaged `FragmentRideEdge` layout | `map/fragment/PlanMapLandFragment.java:2670-2860`; `command/entitys/NavEdgePointsAckBean.java:12-88`; `command/menus/PbMsgType.java:132` | Direct/diagnostic. High |
| Zone naming/rename | Prompts for area name; updates displayed and firmware name | Name validation and hash identity | `NAV_MAP_NAME_MSG`; all-hash-name query/report | `InputAreaNameDialog`, map edit/name callbacks, `CommandManager.getAreaNameList` | `command/menus/PbMsgType.java:175,178`; `command/CommandManager.java:635-646`; `map/activity/MapManualActivityNew.java:4333-4341` | Direct: rename service. High |
| Connection/job channels | Creates corridors between areas and charge paths; validates whether selected areas are connected | Start/end area accessibility; duplicate and corridor-error checks; MN231 distinct flow | `NAV_TODEV_CHL_LINE`, `_END`, line data; generic element Add; `giveUpDrawCorridor` | `PlanMapLandFragment`, `PlanMapLand231Fragment`, `MapDataImpl.isHaveJobChannel` | `command/menus/PbMsgType.java:120,133-134`; `command/CommandManager.java:753-763`; `map/MapDataImpl.java:87-101`; `map/fragment/mn231/PlanMapLand231Fragment.java:2523-2648,3101-3204` | Direct: corridor CRUD, high topology complexity. High |
| Charge channel | Creates/checks route from a work area to dock; reports missing charge path | Selected-area and dock reachability checks | Channel element plus charge-pile relation | `MapDataImpl.isHasAreaHaveChargeChannel`, `isHaveChargeChannel`; `MissingChargePath*Fragment` | `map/MapDataImpl.java:80-94`; `map/fragment/MissingChargePathFragment.java:26-108` | Direct/diagnostic. High |
| Restricted/no-go zone | User chooses restricted zone then drives perimeter; save/cancel/undo | Must be inside/near a valid area; feature selection differs by family | `NAV_TODEV_DRAW_OBS`, `_END`, `NAV_TOAPP_OPT_OBS_INFO`; generic element/manual element | `SelectCreateStrictedZoneModeFragmentNew`, `PlanMapLandFragment`, `MapManualActivityNew.onRestrictedZone` | `command/menus/PbMsgType.java:123-135`; `map/activity/MapManualActivityNew.java:974-1007`; `map/fragment/SelectCreateStrictedZoneModeFragmentNew.java:35-109` | Direct: no-go CRUD; require confirmation. High |
| Obstacle/no-go edit/move | Selects an existing obstacle, moves or redraws it, then applies or cancels | Element type/shape; positioning and collision validation | Generic element update/delete; manual-element add/delete | `BaseMapFragment.updateNoGoZone/updateObstacle`, `MapEditingFragment`, `CommandManager.deleteManualElementMessage`; packaged `FragmentMovingNoGo` layout | `map/fragment/BaseMapFragment.java:11421-11640`; `command/CommandManager.java:471-516`; `map/fragment/MapEditingFragment.java:119-636` | Direct: element mutation. High |
| Safety/security zone | Creates a special safe/security polygon separately from ordinary restricted zone | Exposed only in newer selector and supporting firmware | `NAV_TOAPP_MANUAL_ELEMENT`; `ManualElement` type/shape/hash | `MapManualActivityNew.onSecurityZone`, `CommandManager.addManualElementMessage` | `map/activity/MapManualActivityNew.java:991-1007`; `command/CommandManager.java:471-482`; `command/entitys/ManualElement.java:11-180` | Direct but semantics need packet capture. Medium |
| Obstacle detection during mapping | Radar/vision obstacle callbacks can interrupt or warn while drawing | Radar-capable families; obstacle timeout dialogs | `BleRadarObstacleListener`; obstacle message delay | `PlanMapLandFragment`, `PlanMapLand231Fragment`, `RadarView` | `map/fragment/mn231/PlanMapLand231Fragment.java:3587-3595`; `map/view/RadarView.java:24-355` | Diagnostic/safety entity. Medium |
| Map edit menu | Lists editable elements and delete footer; selects area/line/no-go items on map | Edit availability depends on state/model and loaded hashes | Element type/hash/position | `MapEditingFragment`, `MapEditAdapter`, `MapEditingViewModel` | `map/fragment/MapEditingFragment.java:119-636`; `map/adapter/MapEditAdapter.java:31-163`; `map/viewmodel/MapEditingViewModel.java:19-126` | Direct CRUD reference. High |
| Delete one element | Deletes region/channel/manual element by hash/type/shape/position | Confirmation; special MN231 DB behavior | `deleteMapElements`, `deleteManualElementMessage`, element Delete subcommand | `CommandManager`; `MapDataImpl` | `command/CommandManager.java:505-527`; `map/MapDataImpl.java:25-61` | Direct: delete service; destructive confirmation essential. High |
| Delete all map | Clears all map elements and local DB/hash state | Strong confirmation and idle-state checks | `CommandManager.deleteAll`; local `deleteLineListDBAndHashDB` | `CommandManager`, `MapDataImpl`, `MapEditingFragment` | `command/CommandManager.java:494-504`; `map/MapDataImpl.java:25-30` | Direct/destructive. High |
| Cancel/undo creation | Cancels active draw, gives up corridor, or reverts most recent point/element | Active creation/edit transaction | `NAV_TODEV_CANCEL_DRAW_CMD`; `giveUpDrawCorridor`; revoke handlers | `PlanMapLandFragment`, `PlanMapLand231Fragment` | `command/menus/PbMsgType.java:142`; `command/CommandManager.java:753-763`; `map/fragment/mn231/PlanMapLand231Fragment.java:5918-5928` | Direct, useful emergency/recovery operation. High |
| Area accessibility graph | Checks if selected areas are mutually reachable, which selected index is inaccessible, and which areas are reachable from current position/hash | Requires channel graph and current position type/hash | Local graph queries over area and line DB | `MapDataImpl.judgeAccessibility2/3/Index`, `getDeviceAccessibleAreas` | `map/MapDataImpl.java:63-79,103-126` | Indirect: validate HA work requests before send. High |
| Duplicate area selection/hash | Rejects duplicate or invalid selected-area hash combinations | Work-plan creation | `isSelectAreaHashDump` | `MapDataImpl` / `AreaDBHelper` | `map/MapDataImpl.java:96-102` | Direct validation logic. Medium |
| Work-plan map selection | Selects one or more zones, orders them, previews route, chooses pattern/settings and starts work | Reachability, charge channel, mower state, family capabilities | plan/task protobufs and route generation | `CreateWorkNewFragment`, `NewPreviewRouteFragment`, `SelectArea*Fragment`; packaged `FragmentPathOder` layout | `map/fragment/CreateWorkNewFragment.java:84-413`; `map/fragment/NewPreviewRouteFragment.java:50-312`; `map/fragment/SelectCreateModeFragmentNew.java:34-131` | Direct: task config/start. High |
| Route generation/preview | Requests generated route for selected areas and previews coverage before execution | Newer generic protocol; requires valid selected areas/settings | `getGenerateRouteInformation`, `modifyGenerateRouteInformation`; `NAV_APP_REQUEST_COVER_PATHS`, `NAV_COVER_PATH_UPLOAD` | `CommandManager`, `NewPreviewRouteFragment` | `command/CommandManager.java:658-670,764-775`; `command/menus/PbMsgType.java:165-166` | Direct route-preview API. High |
| Route settings | Adjusts path direction/angle, spacing, border laps, traversal/order and related mowing route parameters | Options vary by mower/work mode | `GenerateRouteInformation`; route modify command; zigzag messages | `RoutSettingActivity`, `RoutMapFragment`, `CuttingRouteFragment`, `WorkSettingsView` | `map/activity/RoutSettingActivity.java:22-155`; `command/menus/PbMsgType.java:151-153`; `command/CommandManager.java:658-775` | Direct: map route option entities. High |
| Pattern/print settings | Enables per-area lawn pattern and retrieves pattern list; preserves patterns associated with recharge areas | Supported models/firmware and cloud availability | `/lawn-pattern/list`, SVG visibility/pattern callback | `PatternSettingsPop`, `LawnPrintingManage`, `MapDataImpl.getPatternInRecharge` | `map/api/MapApiService.java:128-131`; `map/MapDataImpl.java:73-79`; `map/view/PatternSettingsPop.java:80-151` | Indirect/direct setting. Medium |
| Environment setting | Swimming/SP environment type selector; reads and writes device parameter | SP/PC210 class devices | `getSpEnvironmentUpdate`, `spEnvironmentUpdate` | `SPEnvironmentSettingActivity`, `CommandManager` | `command/CommandManager.java:715-725,929-939`; `map/activity/SPEnvironmentSettingActivity.java:38-290` | Direct select entity. High |
| Speed setting | Reads/writes swimming/SP bottom speed as float | SP/PC210 class devices | `getSpSpeedUpdate`, `spSpeedUpdate` | `SPButtomSpeedSettingActivity`, `CommandManager` | `command/CommandManager.java:742-752,940-950`; `map/activity/SPButtomSpeedSettingActivity.java:37-271` | Direct number entity. High |
| Mowing/work settings | Area-specific cutting height, speed, path spacing, laps, obstacle strategy and pattern controls | Device capability/model and selected work mode | plan/job models, driver speed/knife commands, route config | `WorkSettingsView`, `CuttingHeightActivity`, `SeekBarSelectActivity`; packaged `FragmentMowSettings` layout | `map/view/WorkSettingsView.java:87-1130`; `map/activity/CuttingHeightActivity.java:26-99`; `command/menus/PbMsgType.java:182-188` | Direct controls; many already relevant to HA. High |
| Map-position reporting | Live mower coordinate, heading, positioning level/type and start percentage update map | BLE or IoT; subscription while activity active | `NAV_TOAPP_POS_UP`, `NAV_TOAPP_LAT_UP`, `NAV_ZONE_START_PRECENT` | `MACarDataManager`, `DevicePositionModel`, `MapBoxManager` | `command/menus/PbMsgType.java:117-119,167`; `command/app/MACarDataManager.java:7815-7830,8424-8508`; `map/viewmodel/DevicePositionModel.java:9-21` | Direct: position/quality sensors. High |
| Positioning readiness | Shows not-positioned/outside-area/missing-path states and blocks mapping/work | Position level/type, RTC/RTK readiness, model-specific self-check | nav/system state and app flags | `MapWorkNotPositionedFragment`, `DeviceOutsideAreaFragment`, `CreateMapCheckSelfActivity` | `map/fragment/MapWorkNotPositionedFragment.java:25-147`; `map/fragment/DeviceOutsideAreaFragment.java:26-100`; `device/deploy/device/ui/creatmapprepare/CreateMapCheckSelfActivity.java:1097-1163` | Direct diagnostics. High |
| RTK status display | Displays RTK/base-station connection and positioning status on map | RTK-capable devices; omitted/changed for pure visual/radar models | driver RTK config/mask query, base-station messages | `RTKStatusFragment`, `MapTopStateFragment` | `map/fragment/RTKStatusFragment.java:42-401`; `command/menus/PbMsgType.java:189-199` | Direct RTK status sensors. High |
| RTK pairing code | Reads or sets RTK pairing/configuration code | RTK/base-station devices | `readAndSetRtKParingCode(op,cgf)`; `RtkParingCode` | `CommandManager` | `command/CommandManager.java:832-843`; `command/entitys/RtkParingCode.java:11-135` | Direct configuration, sensitive. High |
| Network RTK mode/channel | Selects mower network-RTK link channel/mode; base station can switch MQTT RTK transport | iNavi/network RTK-capable devices | `setNetRtkLinkMode`; `SYS_APP_TO_DEV_SET_MQTT_RTK_MSG`; base MQTT RTK | `CommandManager`, `NetRtkMqttBean` | `command/CommandManager.java:865-886`; `command/menus/PbMsgType.java:57-58,198-199` | Direct select/config entity. High |
| iNavi calibration | Can cancel an active iNavi calibration | iNavi-capable devices | `cancelInaviCalibration` | `CommandManager` | `command/CommandManager.java:483-493` | Direct diagnostic/control. Medium |
| RTK deployment guide | Walks user through mounting/placement/connection of RTK station | Device type and guide content | Guide resources, no unique map mutation command | `RTKDeployGuidelinesActivity`, `DeployGuideLinesHelper` | `device/deploy/device/ui/deployguidelines/RTKDeployGuidelinesActivity.java:31-221`; `device/deploy/device/ui/deployguidelines/DeployGuideLinesHelper.java:17-194` | App-only guidance. High |
| Dock deployment | Step-by-step dock installation and charging verification, with model-specific images/text | Device family; charging state required before completion | charge-state observation; navigation pile info | `StepDeploymentGuideActivity`, `DetailStepDeploymentGuideActivity` | `device/deploy/device/ui/deploymentguide/StepDeploymentGuideActivity.java:143-350`; `device/deploy/device/ui/deploymentguide/DetailStepDeploymentGuideActivity.java:329-544` | Diagnostic/setup workflow. High |
| Dock relocation/reset | User drives/positions mower, confirms new dock, waits for charge check and reset acknowledgement | Idle, connected, positioned and often physically charging | `NAV_TODEV_RESET_CHG_PILE`, `NAV_TOAPP_CHGPILETO`; remote motion | `ResetChargePileActivity`, `PlanMapLandRestPileFragment` | `command/menus/PbMsgType.java:129,141`; `device/deploy/device/ui/deployresetpile/ResetChargePileActivity.java:236-249`; `map/fragment/mn231/PlanMapLandRestPileFragment.java:41-414` | Direct but high-risk setup command. High |
| One-touch leave dock | Commands mower to autonomously leave charging station | Supported device and valid charging/dock state | `NAV_TODEV_ONE_TOUCH_LEAVE_PILE` | command helper / deployment and map controls | `command/menus/PbMsgType.java:143`; `map/fragment/LowerThePileFragment.java:31-140` | Direct button entity. High |
| Manual dock charge | Provides manual direction/charge UI when automatic route/dock state is unavailable | Connected and manual-control capable | remote motion and charge state | `ManualChargeFragment`, `UpperLowerPileFragment`, `LowerThePileFragment` | `map/fragment/ManualChargeFragment.java:34-191`; `map/fragment/UpperLowerPileFragment.java:31-207` | Direct but safety-sensitive. Medium |
| Remote positioning | Dual/one-handed rocker control moves mower during mapping/deployment | Local link/connectivity and device state; speed initialized before motion | driver motion control, `CarRemoteControlManage2` | `RockerTouchViewAll`, `CarRemoteControlManage`, `OneHandedRemoteControlDialogCommon` | `device/deploy/device/manage/CarRemoteControlManage.java:31-252`; `device/deploy/ui/dialog/OneHandedRemoteControlDialogCommon.java:67-490`; `command/menus/PbMsgType.java:182` | Direct but unsuitable for unattended HA automation. High |
| Mapping preflight | Checks charge, sensors, positioning, connectivity and environment before map creation; gives family-specific instructions | Pure visual, pure radar, CM900, LUBA VA/HM/ME/LA, YUKA ML, 3000/5000/10000 variants | state/RTC/radar readiness; no single command | `CreateMapBeforeCheckActivity`, `CreateMapCheckSelfActivity` | `device/deploy/device/ui/creatmapprepare/CreateMapCheckSelfActivity.java:1017-1163,1242-1503`; `device/deploy/device/ui/creatmapprepare/CreateMapBeforeCheckActivity.java:100-376` | Excellent source for HA precondition diagnostics. High |
| Point-cloud request | Requests server generation/download of a point-cloud map for a device | Cloud/network, map areas present, server support | `POST /device/point/cloud/map` | `PointCloudActivity`, `PointCloudViewModule.getMapPiontCloud` | `map/api/MapApiService.java:133-136`; `map/viewmodel/PointCloudViewModule.java:264-302`; `map/activity/PointCloudActivity.java:147-420` | App/cloud-only unless API authenticated. High |
| Point-cloud progress/file | Polls generation progress by business ID, downloads/loads result and associates map area list | Active bizId and successful cloud job | `GET /device/point/cloud/map/progress/{bizId}`; `PointCloudBean`, `PointCloudProgressBean` | `PointCloudViewModule.getMapProgress`, `PointCloudActivity.loadPointCloudFile` | `map/api/MapApiService.java:118-121`; `map/viewmodel/PointCloudViewModule.java:281-302`; `map/activity/PointCloudActivity.java:263-420` | Diagnostic/visualization, cloud-only. High |
| Map backup list | Lists snapshots, backup-capable devices and maps recoverable onto a device | Logged-in/cloud/network; ownership/model compatibility | `GET /map/backup/list`, POST backup/recovery lists | `BackupsMapActivity`, `MapBackupDetailActivity`, `BackupsViewModule` | `map/api/MapApiService.java:94-102,138-140`; `map/viewmodel/BackupsViewModule.java:1431-1526` | App/cloud-only; valuable service if private API is implemented. High |
| Create/update backup | Names a snapshot and starts backup; can update existing snapshot; includes correction value | Device has map; check endpoint passes; cloud network | `POST/PUT /map/backup`; `BackupMapRequest`, `BackupMapUpdateRequest` | `NamingBackupActivity`, `BackupProgressActivity`, `BackupsViewModule` | `map/api/MapApiService.java:69-82`; `map/viewmodel/BackupsViewModule.java:1398-1417`; `map/entity/BackupMapRequest.java:11-51` | App/cloud-only. High |
| Backup progress/cancel | Polls backup/recovery progress and supports cancellation | bizId and operation type | `POST /map/backup/progress`, cancel backup/recovery | `BackupProgressActivity`, `BackupsViewModule.getBackupsProgress` | `map/api/MapApiService.java:79-106`; `map/viewmodel/BackupsViewModule.java:1413-1419,1499-1504` | App/cloud-only. High |
| Restore/recovery | Restores snapshot to selected compatible device; verifies/obtains ownership | Ownership and device compatibility; may overwrite current map | `/map/backup/recovery`, `/recovery/ownership`, `/recovery/list` | `MapBackupDetailActivity`, `SelectDeviceActivity`, `BackupsViewModule` | `map/api/MapApiService.java:59-67,123-126,138-140`; `map/viewmodel/BackupsViewModule.java:1386-1391,1514-1526` | Direct but destructive/cloud-private. High |
| Backup rename/delete | Renames or deletes cloud snapshot | Snapshot ownership | `/map/backup/rename`, `DELETE /map/backup/{bizId}` | `MapBackupDetailActivity`, `BackupsViewModule` | `map/api/MapApiService.java:53-67`; `map/viewmodel/BackupsViewModule.java:1381-1397` | App/cloud-only. High |
| Swimming map sync | Loads local pool boundary/line then asks device for current swimming map and route | Swimming/SP device only | `getSpMap`, `getSpLine`; `SwimmingPoolMapListener`; `SwMapInfo` | `SwimmingMapActivity`, `SwimmingMapViewModel` | `command/CommandManager.java:726-741`; `map/viewmodel/SwimmingMapViewModel.java:38-121`; `command/app/contract/SwimmingPoolMapListener.java:3-14` | Direct for pool-device integration. High |
| Swimming map persistence | Saves/query pool map and line in `SwpMapDB`; transforms ordered point sets to screen coordinates | Swimming device ID and map/line type | `SwMapInfo`, local sorted point maps/lines | `SwimmingMapViewModel.getLocalData/getSwimming*Data` | `map/viewmodel/SwimmingMapViewModel.java:27-121`; `command/entitys/SwMapInfo.java:10-77` | Indirect decoding/rendering. High |
| Swimming line drawing | Renders pool outline/path in custom view and supports map screen interaction | Swimming device | Canvas line drawing, ordered points | `SwimmingMapActivity`, `LineDrawingView` | `map/swimming/LineDrawingView.java:17-115`; `map/swimming/SwimmingMapActivity.java:55-541` | App-only visualization. High |
| Stop and save / discard task | Stops current swimming/SP task either preserving or discarding task/map result | Active applicable task | `stopAndSaveTask`, `stopAndNotSaveTask` | `CommandManager` | `command/CommandManager.java:951-973` | Direct buttons; confirm semantics per firmware. High |
| Cost map / fog | Firmware can upload a cost-map/fog layer used as environment awareness overlay | Supporting navigation firmware | `Nav_TOAPP_COSTMAP` | map command/data pipeline | `command/menus/PbMsgType.java:171` | Indirect diagnostic map layer. Medium |
| SVG map transport | Uploads/synchronizes SVG representation, including MN231 map updates and patterns | Newer/MN231 map stack | `NAV_TODEV_SVG_MSG`, `NAV_TOAPP_SVG_MSG`, `sendSvgDate` | `PlanMapLand231Fragment`, `MACommandApiHelper`, `SvgUtils` | `command/menus/PbMsgType.java:176-177`; `map/fragment/mn231/PlanMapLand231Fragment.java:2534-2648`; `map/utils/SvgUtils.java:17-232` | Indirect/direct map snapshot transport. High |

## Protocol and model notes

### Legacy navigation command family

The most important concrete protobuf message identifiers are in `PbMsgType`:

- Position and live map: `NAV_TOAPP_LAT_UP`, `NAV_TOAPP_POS_UP`, `NAV_ZONE_START_PRECENT`, `Nav_TOAPP_COSTMAP`.
- Geometry creation: `NAV_TODEV_DRAW_BORDER(_END)`, `NAV_TODEV_DRAW_OBS(_END)`, `NAV_TODEV_CHL_LINE(_END)`, and `NAV_TODEV_CANCEL_DRAW_CMD`.
- Geometry feedback: `NAV_TOAPP_OPT_BORDER_INFO`, `NAV_TOAPP_OPT_OBS_INFO`, `NAV_TOAPP_OPT_LINE_UP` and their ACKs.
- Dock: `NAV_TOAPP_CHGPILETO`, `NAV_TODEV_RESET_CHG_PILE`, `NAV_TODEV_ONE_TOUCH_LEAVE_PILE`, `NAV_TODEV_RECHGCMD`.
- Map synchronization: `NAV_TOAPP_GETHASH(_ACK)`, `NAV_TODEV_GET_COMMONDATA`, `NAV_TOAPP_GET_COMMONDATA_ACK`.
- Work/route: `NAV_TODEV_MOW_TASK`, `NAV_TODEV_PLANJOB_SET`, `NAV_PLAN_TASK_EXECUTE`, cover-path request/upload and task-control messages.
- Newer map representation: `NAV_MAP_NAME_MSG`, `NAV_TODEV_SVG_MSG`, `NAV_TOAPP_SVG_MSG`, `NAV_TOAPP_ALL_HASH_NAME`, `NAV_TOAPP_MANUAL_ELEMENT`, `NAV_TOAPP_MANUAL_GENERATE`, `NAV_TOAPP_MODIFY_GENERATE`.

Evidence: `command/menus/PbMsgType.java:117-181`. Numeric command/subcommand pairs are present in that enum and should be preferred over reverse-engineering UI constants.

### Generic element model

Newer flows use element messages with a subcommand (`Add`, update/delete variants), element type, shape, hash and geometry. The manual-element manager exposes:

```text
addManualElementMessage(deviceName, ManualElement, owner, callback)
deleteManualElementMessage(deviceName, hash, type, shape, isAll, link, owner, callback)
deleteMapElements(deviceName, hash, type, owner, callback)
```

This is the clearest HA-facing CRUD abstraction, but the numeric `type` and `shape` mapping should be validated against real packets before exposing writes. Evidence: `command/CommandManager.java:471-527`; `command/entitys/ManualElement.java:11-180`; `map/fragment/mn231/PlanMapLand231Fragment.java:3101,3203,3406,5299`.

### Local database semantics

`MapElementDB` stores geometry keyed by device and hash; `LineListDB` stores route/channel geometry. `MapDataImpl` delegates to `AreaDBHelper` for:

- delete by device, hash, type, state or position;
- save element geometry separately from element metadata;
- save/update total and individual hashes;
- save line hashes/order;
- test charge/job channels and graph accessibility;
- retrieve patterns within recharge-linked areas.

The explicit hash+position APIs and separate MN231 deletion method indicate that element order/slot is semantically important and that a polygon-only HA representation would lose information. Evidence: `map/MapDataImpl.java:24-154`; `map/db/AreaDBHelper.java:63-889`; `base_module/db/MapElementDB.java:12-327`; `base_module/db/LineListDB.java:11-136`.

## Model and firmware gates

The decompile exposes capability predicates more often than literal firmware version strings. Important gates include:

- `isPureVisual()` and `isPureRadar()` alter preflight, positioning and sensor checks.
- `isSupportRadarSelfCheck()` adds radar preflight.
- `isLubaVA()`, `isLubaHM()`, `isLubaME()`, `isLubaLA()`, `isCM900()`, `isYukaML()`, `isYuKa()` and `isLuba2()` select different instructions and checks.
- LUBA VA with `DeviceMultimodelHelper.isHaveRTKMode()` and CM900 expose RTK-related preparation.
- `is3000Type()` and `isLubaVA442_5000_10000()` change RTC/readiness requirements.
- Map rendering specifically branches for YUKA versus LUBA VA/HM.
- MN231 has separate activity, fragments, SVG synchronization, editing, corridor and reset-pile implementations.
- SP/PC210/swimming devices use a separate map, line, environment, speed and plan protocol.

Evidence: `device/deploy/device/ui/creatmapprepare/CreateMapCheckSelfActivity.java:1017-1163,1226-1503`; `map/mapbox/MapBoxManager.java:3467-3483`; `map/mn231/MapActivityLand231.java:446-468`; `map/fragment/mn231/PlanMapLand231Fragment.java:1298-1339`; `command/CommandManager.java:715-764,887-950`.

Exact firmware thresholds remain uncertain because most checks are hidden behind extension/helper methods or remote capability state. HA should capability-detect from device-reported product/firmware data rather than hard-code the marketing family alone.

## Hidden, debug and test features

| Feature | Evidence and interpretation | Confidence |
|---|---|---|
| Simulation commands | `SYS_SIMULATION_CMD` and `NAV_SIMULATION_CMD` exist in the production protocol enum (`command/menus/PbMsgType.java:52,155`). Likely factory/development state injection, not normal UI. | High existence; low usable semantics |
| Debug map retry toasts/logs | `HashDataManager` emits Chinese test toasts for region/route retry exhaustion and tracks detailed counters (`command/app/HashDataManager.java:99-158`). These expose otherwise hidden sync-failure states. | High |
| Factory map release | `MACarDataManager.releaseFactoryMap()` exists (`command/app/MACarDataManager.java:8293-8296`). No normal user workflow was confirmed. | Medium |
| Test guide activity | `ActivityTestGuideBinding` and test/guide resources are packaged; manifest exposure was not confirmed as exported. | Medium |
| Cost-map fog | `Nav_TOAPP_COSTMAP` (“map fog”) is a protocol message but no complete production renderer path was recovered. | Medium |
| SVG/pattern hiding | `PatternSettingsPop` tracks `isHideSvg`, suggesting a hidden/display toggle for generated lawn art (`map/view/PatternSettingsPop.java:80-151`). | Medium |
| Report-map/error tins | `ReportMapFragment`, `ErrorMapTinsFragment` and `MissionTinMapFragment` layouts indicate diagnostic/report overlays around map tiling or route generation. Some methods are decompilation stubs. | Low-medium |
| Automatic touch/type-touch | Packaged `FragmentautomaticTouch`, `FragmentAutomaticTypeTouch` and `FragmentAutomaticCreationTips` layouts imply internal automatic map interaction modes beyond the primary flow. | Medium |
| Navigation debug protobufs | `SYS_DEBUG_*`, `SYS_MOW_TO_APP_QCTOOLS_INFO` and report bus messages can expose factory/QA telemetry (`command/menus/PbMsgType.java:48-72`). | High existence |

The manifest registers map/deployment activities as internal application components; no map creation/edit activity was found with an external browsable intent filter. Deep-link navigation is instead abstracted through `MapNavigator` and `DeviceDeployNavigator`. Evidence: `resources/AndroidManifest.xml:379-612`; `map/api/MapNavigator.java:10-73`; `device/deploy/api/DeviceDeployNavigator.java:10-69`.

## Home Assistant implementation relevance

Highest-value read-only work:

1. Decode and expose region/line hash state, position quality/type, RTK/base status, dock/charge state, active area/hash and dynamic route.
2. Mirror `HashDataManager` reconciliation: aggregate MurMur comparison, per-hash fetch, route/region separation, ordering, retries, and transactional refresh.
3. Store typed elements (area, channel, obstacle/no-go, safety/manual element, dock) with hash, type, shape, order/frame and raw payload, not only GeoJSON.
4. Expose map sync health: loading, missing hashes, retry count, route generation progress and update-failed state.
5. Validate selected work zones with the local accessibility/channel graph before starting a job.

Reasonable opt-in writes:

- area rename;
- route/environment/speed/work settings;
- request map refresh;
- map backup/restore only if cloud authentication is already supported;
- one-touch leave dock and calibration cancellation as explicitly confirmed buttons.

High-risk writes that should remain disabled by default:

- manual remote drive;
- boundary/channel/no-go/safety-zone creation or mutation;
- delete-one/delete-all map;
- dock relocation/reset;
- RTK pairing/link reconfiguration;
- restore-over-current-map.

Every map write should require current-map hash preconditions and then wait for a new robot hash/SVG before declaring success. Optimistically mutating only HA’s local geometry would diverge from firmware authority.

## Uncertainties and decompilation caveats

- Several very large methods decompile with synthetic names, damaged branches, or `UnsupportedOperationException`; UI resources and callbacks sometimes provide stronger evidence than method bodies.
- “Obstacle,” “restricted zone,” “no-go,” “security zone,” and “manual element” overlap across protocol generations. Numeric type/shape values were not assigned friendly names unless corroborated.
- “Frame” is represented by positions/order, transaction IDs, hash timestamps and numbered JSON members; no single public model named `MapFrame` was found.
- Map correction/rectification appears to affect presentation and backup compatibility, but whether firmware stores the same correction value was not conclusively established.
- Automatic map creation is clearly packaged, but its exact device-side command construction is partly hidden behind generic manual-generate messages.
- Cloud endpoints are visible, but authentication, quotas, server-side snapshot format and cross-model restore rules cannot be established statically.
- Point-cloud output format/file URL handling is partially obscured by coroutine decompilation.
- Cost-map fog and some diagnostic fragments may be dormant or firmware-limited.
- Capability helpers hide some exact product and firmware thresholds. Runtime capability reports should be treated as authoritative.

## Files reviewed

The analysis searched all 312 non-generated Java files under `com/agilexrobotics/map`, all 90 Java files under `com/agilexrobotics/device/deploy` (including generated bindings for UI discovery), and 743 keyword-matched related files under command, proto, device, base-module, work, home, services, testing and resources. The following groups received direct behavioral review:

- `com/agilexrobotics/map/activity/`: map home/manual/video, rectification, route/environment/speed, point cloud and backup activities.
- `com/agilexrobotics/map/fragment/`: creation, editing, boundary/edge, no-go, channel, selection, route preview, positioning, RTK/dock and MN231 fragments.
- `com/agilexrobotics/map/mapbox/`: `MapBoxManager`, tags, draw utilities, color/style and custom map view.
- `com/agilexrobotics/map/viewmodel/`: map/create/edit/position, backup, point-cloud and swimming view models.
- `com/agilexrobotics/map/api/`, `entity/`, `db/`, `utils/`, `swimming/`, `mapping_guideline/` and relevant views/adapters.
- `com/agilexrobotics/device/deploy/device/ui/`: creation preflight, deployment/RTK guides, detailed steps and reset-pile flow.
- `com/agilexrobotics/device/deploy/device/manage/`, `api/`, point-cloud models and deployment resources.
- `com/agilexrobotics/command/CommandManager.java`, `MapCommandManager.java`, `app/MACommandApiHelper.java`, `app/MACarDataManager.java`, `app/HashDataManager.java`, contracts, entities and `menus/PbMsgType.java`.
- `com/agilexrobotics/proto/`: `MctrlNav`, `MctrlSys`, `MctrlDriver`, `Basestation`, `Common`, `LubaMsgOuterClass`, `LubaMul`, `Pdt` and `SpinoCtrlOuterClass`, searched for map/RTK/dock/route messages and field names.
- Map database/model support in `com/agilexrobotics/base_module/db`, entities and helpers; device type/capability helpers in `com/agilexrobotics/device/source`.
- `resources/AndroidManifest.xml`, base and localized strings, map/deployment layouts, drawables and XML resources.

Generated Hilt injectors, `R.java` constants and most generated data-binding bodies were inventoried but not treated as behavioral evidence except to establish packaged UI surfaces.
