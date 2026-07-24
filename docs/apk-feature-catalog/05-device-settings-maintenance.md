# Mammotion Android APK feature catalog: device settings, maintenance, firmware, errors, and hardware

## Scope and interpretation

This catalog is based on the decompiled Android application under `/Users/mattjoslin/mammotion-apk-decompile/src`, principally:

- `sources/com/agilexrobotics/device/info/`
- `sources/com/agilexrobotics/device/setting/`
- `sources/com/agilexrobotics/find/device/`
- `sources/com/agilexrobotics/signal/`
- relevant `base_module` entities/enums, `command`, `proto`, manifest, layouts, and English resources.

Paths below are relative to `src/`. “Command” is the exact decompiled Java method or HTTP endpoint. Argument meaning is stated only where the caller, entity, protobuf, or UI makes it defensible. Numeric meanings that are inferred are marked as such. Decompiled identifiers preserve upstream misspellings such as `Rian`, `ionfoType`, `geteway`, `INaive`, and `Swtich`.

HA relevance:

- **High** — natural entity/service and state useful to automations.
- **Medium** — useful diagnostic or occasional service, but potentially cloud-, BLE-, safety-, or workflow-dependent.
- **Low** — app navigation, destructive/support action, or poor fit for routine automation.

## Top-level settings and device administration

| Feature/control | Exact command, field, or route | Behavior / state | Gates and evidence | HA relevance |
|---|---|---|---|---|
| Rename device | Device-details/rename activity; cloud/device manager update | Changes user-visible robot name | `sources/com/agilexrobotics/device/info/activity/DeviceDetailsActivity.java:2777-2820`; `sources/com/agilexrobotics/device/info/activity/DeviceRenameActivity.java:43-147` | Medium: configuration flow, not a routine entity |
| Unbind/remove device | `MapCommandManager.removeDevice(deviceTitle)` plus unbind workflow | Removes local command-manager registration and cloud binding | `sources/com/agilexrobotics/device/info/activity/DeviceDetailsActivity.java:1037-1041,3005-3014` | Low; destructive and should require explicit confirmation |
| Robot restart | `MACommandApiHelper.remoteRestart(int,int)`; device-details `sendRestart()` | Remote reboot, with timeout/result listener | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1259-1262`; `sources/com/agilexrobotics/device/info/activity/DeviceDetailsActivity.java:1757-1763,2875-2902`; warning text `resources/res/values/strings.xml:1159-1160` | Medium service; safety warning on slopes/current tasks |
| Factory reset | `MACommandApiHelper.resetSystem()` | Resets robot system; UI presents factory-reset confirmation | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1421-1424`; `sources/com/agilexrobotics/device/info/activity/DeviceDetailsActivity.java:1858-1869`; `resources/res/values/strings.xml:638` | Low/destructive; not a normal HA service |
| RTK/base reset | `resetBaseStation()` and RTK reset flows | Resets reference/base station or its mapped position | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1413-1416`; `resources/res/values/strings.xml:477-478,557,751` | Low; can invalidate positioning/map |
| Charging-station reset/relocate | map setting and deployment flow | Re-locates a moved dock; robot may undock and scan | `sources/com/agilexrobotics/device/setting/activity/MapSettingActivity.java:786-810`; `resources/res/values/strings.xml:933-934,1084-1085,1517` | Medium service, but gated by position/state |
| Warranty/activation | `POST /device-server/v1/device/warranty`; `POST active-time` | Reads warranty and activation time | `sources/com/agilexrobotics/device/info/api/DeviceInfoApiService.java:52-59` | Low/diagnostic |
| Device identity/details | serial/model/product fields and `getDeviceProductModel()` | Model, serial, IDs, activation, SIM/RTK association | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:953-956`; `sources/com/agilexrobotics/command/CommandManager.java:647-656`; details binding IDs `sources/com/agilexrobotics/device/info/databinding/ActivityDeviceDetailsBinding.java:320-527` | High as diagnostic attributes |
| Upload logs | log-upload action with a 5 s timeout | Uploads device logs for support | `sources/com/agilexrobotics/device/info/activity/DeviceDetailsActivity.java:1215-1230`; upload gate `sources/com/agilexrobotics/device/setting/fragment/appsetting/CarSettingDrawerFragment.java:3192-3201` | Low; support action, potentially privacy-sensitive |

## Firmware, versions, and update workflow

| Feature | Exact command / fields | Details | Evidence | HA relevance |
|---|---|---|---|---|
| Check all device versions | `POST version/check`, body `AllDeviceVersionCheckRequstBean` | Checks robot and associated modules for upgrade eligibility/current/latest versions | `sources/com/agilexrobotics/device/info/api/DeviceInfoApiService.java:39-46`; cache of `currentVersion` at `sources/com/agilexrobotics/device/info/viewmodel/DrawerDeviceViewModel.java:87-109` | High: update entity |
| Start cloud upgrade | `POST device/upgrade`, body `UpgradeBean` | Requests server-managed upgrade | `sources/com/agilexrobotics/device/info/api/DeviceInfoApiService.java:48-50` | High update service, with state checks |
| Pool robot version check | `POST /device-server/v1/pool-robot/version/check`, `PoolRobotVersionCheckReq` | Separate swimming-pool product update path | `sources/com/agilexrobotics/device/info/api/DeviceInfoApiService.java:43-46` | Medium; model-specific |
| Pool OTA phase 1 | `sendSwimmingPoolDeviceOtaFirst(int,int,int firmwareType,String iotId,String)` | Starts a component OTA; exact first two integer meanings are not recoverable from names | `sources/com/agilexrobotics/device/info/SwimmingPoolOtaCorrelation$startUpdate$1$1.java:112-123`; helper signature `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1582-1585` | Medium; expose only behind robust model gating |
| Pool OTA package transfer | `sendSwimmingPoolDeviceOtaPackage(List<Integer>,int,int,int)` | Local packetized firmware transfer | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1590-1593`; `sources/com/agilexrobotics/device/info/SwimmingPoolOtaCorrelation.java:343-388` | Low for HA; transport-level operation |
| Pool OTA phase 2 | `sendSwimmingPoolDeviceOtaSecond(int firmwareType,int fileSize,String path,String latestVersion)` | Sends file metadata/path and begins second phase | caller `sources/com/agilexrobotics/device/info/SwimmingPoolOtaCorrelation.java:365-379`; helper `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1601-1604` | Low/medium |
| OTA info/version query | `getDeviceOTAInfo(int)`; `getDeviceVersionInfo(int)`; `getDeviceVersionMain(String)`; `getDeviceVersionMain2()` | Queries upgrade state and module versions | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:947-972`; calls `sources/com/agilexrobotics/device/info/activity/DeviceVersionActivity.java:391-466` | High sensors |
| Update progress | state fields/events include progress, result, version, download speed | UI presents current/new/partial versions, retry, progress, transfer speed and errors | `sources/com/agilexrobotics/device/info/view/FirmwareUpdateKTView.java:939-1043`; `sources/com/agilexrobotics/device/info/view/FirmwareUpdateView.java:339-348,795-841` | High: update progress/installed/latest/error attributes |
| Component versions | main controller, communication, LoRa, RTK, left/right motors, hub/wheel-hub, water pump/spray, transmission-message module | Version screen dynamically shows hardware-specific components | `sources/com/agilexrobotics/device/info/databinding/ActivityDeviceVersionBinding.java:173-272` | High diagnostics |
| Version history/release notes | `VersionHistoryActivity` and version info | Displays historical release/version information | `sources/com/agilexrobotics/device/info/activity/VersionHistoryActivity.java:50-182` | Medium |
| Automatic firmware update | device setting “Keep updated automatically”; idle, docked, Wi-Fi requirement | Server-backed device setting; app says update occurs in idle period while charging and on Wi-Fi | `resources/res/values/strings.xml:500,1210`; `sources/com/agilexrobotics/device/info/activity/DeviceDetailsActivity.java:2014-2056` | High switch if backend field can be identified safely |
| Update preconditions | battery, charging, Wi-Fi, work state, paired versions | UI blocks/redirects based on state; firmware pair mismatch is tracked by `VersionPairStatus` | `sources/com/agilexrobotics/device/info/OtaCorrelationViewModel.java:758-860,964-1028`; `sources/com/agilexrobotics/device/setting/fragment/appsetting/CarSettingDrawerFragment.java:455-456,3124-3137`; strings `resources/res/values/strings.xml:592,1088,1210` | Important availability and error metadata |
| OTA timeouts | standard ~50 s, swimming local ~180 s, other stage-specific 3/5/10/15/18 s | These are app workflow timeouts, not guaranteed protocol deadlines | `sources/com/agilexrobotics/device/info/OtaCorrelationViewModel.java:790-802,1022-1028,1082-1093` | Implementation detail |
| Update firmware gates | `isSupportAutoUpgrade`, `isSupportINaviFirmwareVersion`, `isSupportNewSingleSwitchFirmwareVersion`, `isSupportLocalUpgradeSwimmingPool`, version matching | Controls visibility/path selection by model and firmware | examples `sources/com/agilexrobotics/device/info/activity/DeviceDetailsActivity.java:2025-2056`; `sources/com/agilexrobotics/device/info/OtaCorrelationViewModel.java:790`; `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:713-721` | Must be mirrored before exposing entities |

## Network, Wi‑Fi, 4G, RTK, and positioning

| Setting/status | Exact command / route / field | Details and values | Evidence | HA relevance |
|---|---|---|---|---|
| Current Wi‑Fi/network info | `getDeviceNetWorkInfo()`; `DeviceNetInfo` fields `wifi_ssid`, `wifi_mac`, `wifi_rssi`, integer `ip`, `mask`, misspelled `geteway` | Read-only connection diagnostics | helper `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:943-946`; fields `sources/com/agilexrobotics/base_module/entity/DeviceNetInfo.java:23-28,73-95` | High diagnostic sensors; MAC may be sensitive |
| Wi‑Fi scan/connect | device network activity and BLE Wi‑Fi config listener | Scans SSIDs, accepts password, configures robot | `sources/com/agilexrobotics/device/setting/activity/DeviceNetworkActivity.java:343-540,745-810`; `sources/com/agilexrobotics/device/setting/view/WifiConfigInpuPswPop.java:40-89` | Medium configuration flow; avoid storing password |
| Saved Wi‑Fi records | `getRecordWifiList(boolean)` / `getRecordWifiList2()` | Lists remembered networks | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1038-1050`; `sources/com/agilexrobotics/device/setting/activity/DeviceNetworkActivity.java:588-610` | Medium diagnostic/config |
| Disconnect/forget current Wi‑Fi | `close_clear_connect_current_wifi(String,int,boolean)` / variant `...wifi2(String,int)` | Disconnects and/or clears current record; boolean semantics uncertain | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:541-552`; UI strings `resources/res/values/strings.xml:605,614` | Medium service |
| Wi‑Fi radio enable | `setDeviceWifiEnableStatus(boolean)` | Enables/disables device Wi‑Fi | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1797-1800` | Medium; disabling can remove connectivity |
| Network diagnostics | visible “Check Network” / “Network Diagnostics” and status pages | Displays Wi‑Fi, 4G, RTK and connection details | `resources/res/values/strings.xml:566,706,791`; `sources/com/agilexrobotics/signal/activity/CarStatusDetailsActivity.java:4000-4235` | High diagnostic entity/button |
| 4G/SIM status and activation | `POST iot/sim/activation`, `/iot/sim/detail`, `/device/sim/limit/activate` | SIM details, activation and traffic-limit activation | `sources/com/agilexrobotics/signal/api/SignalApiService.java:43-69` | Medium; status high, activation low |
| RTK user/device list | `GET rtk/devices` and `/device-server/v1/rtk/devices` | Lists user's RTK stations | `sources/com/agilexrobotics/signal/api/SignalApiService.java:91-93`; `sources/com/agilexrobotics/device/info/api/DeviceInfoApiService.java:70-73` | Medium |
| RTK cloud pairing | `POST iot/device/pairing`, `NetRtkPairingReq` or `NetRtkPairingReq2` | Pairs robot/reference station via backend | `sources/com/agilexrobotics/signal/api/SignalApiService.java:95-101` | Low/medium setup service |
| RTK pairing code | `readAndSetRtKParingCode(int op,String cgf,String deviceName,...)`; helper `readAndSetRtKParingCode(int,String,String)` | Reads or writes Datalink/LoRa pairing code. `op` and `cgf` names are protocol-level; callers use operation-specific values | `sources/com/agilexrobotics/command/CommandManager.java:832-842`; helper `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1225-1228`; calls `sources/com/agilexrobotics/signal/activity/RTKSettingActivity.java:2537-2584` | Medium configuration |
| Positioning/RTK link mode | `setNetRtkLinkMode(deviceName, channel, ...)`; legacy `setNetRtkLinkMode(int)` | Selects channel; one caller explicitly sets `channel=0`. UI distinguishes Antenna Over Datalink and Over Internet | `sources/com/agilexrobotics/command/CommandManager.java:876-886`; `sources/com/agilexrobotics/signal/viewmodel/CarStatusDetailViewModule.java:1144`; strings `resources/res/values/strings.xml:757-759` | High select, but enum mapping needs runtime confirmation |
| NetRTK backend enable | `POST iot/net-rtk/enable`, body `NetRtkModeReq` | Enables/disables network RTK mode | `sources/com/agilexrobotics/signal/api/SignalApiService.java:107-109` | High switch if service entitlement exists |
| iNavi service status/enable | `POST /iot/i-naive/status`, `/iot/i-naive/enable`, newer `/iot/i-navi/enable` | Service entitlement/status and enable flow; both old misspelled and newer routes coexist | `sources/com/agilexrobotics/signal/api/SignalApiService.java:57-74`; device-info duplicate status route `sources/com/agilexrobotics/device/info/api/DeviceInfoApiService.java:61-64` | High diagnostics; switch may involve purchase/terms |
| iNavi box selection | `GET /invai/devices`; `POST /device/handoff/iNvavi/box` | Lists and hands robot off to an iNavi box | `sources/com/agilexrobotics/signal/api/SignalApiService.java:76-79,111-114` | Medium select |
| Signal indicators | Wi‑Fi RSSI, 4G state/signal, RTK/LoRa versions and status, no-signal/weak-signal dialogs | Status-bar fragments poll and update visible indicators | `sources/com/agilexrobotics/signal/fragment/StatusBarExtendWiFiFragment.java:275-324`; `sources/com/agilexrobotics/signal/fragment/StatusBarExtend4gFragment.java:539-626`; `resources/res/values/strings.xml:1408` | High diagnostic sensors |
| Firmware gate for RTK service | minimum firmware `"1.14.0"` in multiple positioning screens | App refuses/redirects when robot firmware is older | `sources/com/agilexrobotics/signal/newstatus/SignalConnectionHomepageActivity.java:1088-1103`; `sources/com/agilexrobotics/signal/activity/PositioningModeActivity.java:2179-2190` | Mirror as entity availability |

## Audio, voice, lighting, camera, and wiper

| Control | Exact command / state | Values or semantics | Gates/evidence | HA relevance |
|---|---|---|---|---|
| Read audio config | `getCarAudioCfg()` | Retrieves current voice/audio configuration | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:865-868`; invocation `sources/com/agilexrobotics/device/setting/fragment/appsetting/DrawerSettingsViewModel.java:368-379` | High |
| Voice on/off | `setCarVolume(int)` through `DrawerSettingsViewModel.setCarVolume` | UI treats argument as volume/audio switch state; exact numeric range should be confirmed from telemetry | `sources/com/agilexrobotics/device/setting/activity/VoiceNormalActivity.java:203-209`; view model `sources/com/agilexrobotics/device/setting/fragment/appsetting/DrawerSettingsViewModel.java:579-588` | High switch |
| Voice gender | `setCarVolumeSex(int)` | UI sends `0` for one selection and `1` for the other; buttons are male/female. Decompiled click ordering must be checked at runtime before naming enum values | `sources/com/agilexrobotics/device/setting/activity/VoiceNormalActivity.java:191-195,332-350`; strings `resources/res/values/strings.xml:639,689` | Medium select |
| Voice language | `setCarVoiceLanguage(int)` | Integer language ID | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1712-1715`; `sources/com/agilexrobotics/device/setting/activity/VoiceNormalActivity.java:354-372` | Medium select; enumerate dynamically |
| Volume level | `setVolumeValue(int)` | Separate volume-value control exists in command layer | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1859-1862` | High number if supported |
| Pool sound prompts | `sendSingTinsOpenOrClose(device, boolean)`; state `SwimmingPoolDeviceStatue.isSingHint` | Toggles pool-cleaner sound hints | `sources/com/agilexrobotics/device/setting/fragment/appsetting/CarSettingDrawerFragment.java:2888-2900`; `sources/com/agilexrobotics/device/setting/fragment/appsetting/DrawerSettingsViewModel.java:450-461` | High switch, pool models only |
| Read night-light state | `getCarNightLight(int)` | Queries night-light setting | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:872-875` | High |
| Automatic/night light | `setCarLampCtrlNightLight(boolean,int,int)` | Boolean enable plus two integer parameters, likely mode/time-related; exact meanings uncertain | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1701-1704`; caller `sources/com/agilexrobotics/device/setting/fragment/appsetting/DrawerSettingsViewModel.java:560-565` | High switch; extra fields need protocol validation |
| Manual/fill light | `setCarLampCtrlHandMovement(boolean,int)` | Manual light toggle plus integer lamp/channel argument | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1697-1700`; feature text `resources/res/values/strings.xml:1328-1329,1343` | High light entity |
| Side light | `readAndSetSidelight(boolean,int)` | Read/write side-light control | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1229-1232`; caller `sources/com/agilexrobotics/device/setting/fragment/appsetting/DrawerSettingsViewModel.java:627-632` | High switch |
| Camera/video capability | model capability `isSupportVideo()` / `isSupportVision()`; camera-oriented settings are shown only on capable devices | Video/vision is a model feature; this scope exposes navigation/status more than a direct stream command | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:761-765`; details/video activity `sources/com/agilexrobotics/device/info/malfunction/VideoPlayActivity.java:42-154` | Medium; do not infer a camera entity from UI alone |
| Camera wiper | `setCarWiper(2)` | UI button triggers one-shot wiping and toast. Value `2` means the app's wipe action in this call site; other enum values are not established | `sources/com/agilexrobotics/device/setting/activity/CarSettingDrawerActivity.java:2676-2682`; duplicate fragment flow `sources/com/agilexrobotics/device/setting/fragment/appsetting/CarSettingDrawerFragment.java:2937-2951` | High button on supported vision models |

## Charging, battery, and power management

| Feature | Exact command / fields | Semantics | Evidence | HA relevance |
|---|---|---|---|---|
| Battery level | `MACarDataManager.getDeviceBattery()` / per-device map | Battery percentage/state cache | `sources/com/agilexrobotics/command/app/MACarDataManager.java:7755-7766` | High sensor |
| Charge state | `getDevicechargeState()` / `getDeviceChargeState(String)` | Current docking/charging state | `sources/com/agilexrobotics/command/app/MACarDataManager.java:7779-7789,7880-7883` | High sensor |
| Battery health/capacity | UI text describes maximum capacity relative to new battery | Health diagnostic is present; backing protobuf/state field should be located in integration transport before implementation | `resources/res/values/strings.xml:1366`; battery query command below | High diagnostic sensor |
| Query battery settings | `CommandManager.queryBatteryInfo(deviceName, owner, linkManager, callback)` | Reads charging policy as response string decoded by observer | `sources/com/agilexrobotics/command/CommandManager.java:809-820` | High |
| Set smart charging | `setBatteryInfo(deviceName,owner,linkManager,boolean isSmart,int progress,boolean peakValleyChargeSwitch,int valleyChargeStartTime,int valleyChargeEndTime,callback)` | Exact fields: smart mode, charge-limit progress, off-peak switch, start/end times | `sources/com/agilexrobotics/command/CommandManager.java:854-870` | High switch/number/time entities |
| Charge limit | `progress` in `setBatteryInfo`; UI describes stop at selected limit | App supports configurable maximum; includes “Charge to 100%” override | command above; strings `resources/res/values/strings.xml:645,1059-1061,1077` | High number + button |
| Off-peak charging | `peakValleyChargeSwitch`, `valleyChargeStartTime`, `valleyChargeEndTime` | Pauses at safe 20%, charges in scheduled valley period; period should be at least 3 h | command above; `resources/res/values/strings.xml:1067-1068` | High switch + time selectors |
| Smart task-aware charging | `isSmart`; app text says pause at 80% and resume to 100% before next task | Charge policy coordinated with schedule | `resources/res/values/strings.xml:1060` | High switch |
| Recharge threshold / continue work | `getRechargeAndContinueWorking(deviceName,...,int id,int rw,...)`; setter adds `int context` | Read/write recharge-and-resume policy; numeric IDs/context/rw are protocol selectors | `sources/com/agilexrobotics/command/CommandManager.java:685-695,899-909`; text `resources/res/values/strings.xml:1499-1500` | High number/select after enum recovery |
| Return to dock/cancel recharge | application work commands, visible Recharge/Cancel controls | Ends/pauses task and sends robot to station | strings `resources/res/values/strings.xml:536,554,734,760`; state guidance `:994` | High buttons; likely covered by main integration command catalog |
| Smart sleep | `getSleepStatus(deviceName,...,int type)` / `setSleepStatus(...,int type,boolean open)` | Separate docked and idle sleep types; exact `type` mapping requires caller/runtime confirmation | `sources/com/agilexrobotics/command/CommandManager.java:696-706,910-920`; strings `resources/res/values/strings.xml:1173,1186-1187,1275` | High switches if 4G availability is enforced |
| Battery cycles | `MaintainBean.totalBatteryCycles`; model gate `isSupportBatteryLoopCount()` | Lifetime charge-cycle counter | `sources/com/agilexrobotics/base_module/entity/MaintainBean.java:11,23-25`; `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:674-676` | High diagnostic sensor |

## Cutter, blades, cutting height, and work mechanics

| Feature | Exact command / field | Details | Evidence | HA relevance |
|---|---|---|---|---|
| Cutter mode read/write | `sendGetCutterMode()` / `sendSetCutterMode(int)` | Reads/sets cutting-disc mode; UI selects numeric mode | `sources/com/agilexrobotics/device/setting/fragment/appsetting/DrawerSettingsViewModel.java:400-407,473-484`; helper `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1491-1494,1562-1565` | High select if enum mapping recovered |
| Blade/cutter speed | `getSpeed()` / `setSpeed(float)`; `BladeSpeedActivity` | Adjustable cutter speed; higher speed helps dense/tall grass but costs power | helper `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1059-1062,1847-1850`; `sources/com/agilexrobotics/device/setting/activity/BladeSpeedActivity.java:121-146`; text `resources/res/values/strings.xml:1007-1008` | High number/select |
| Blade speed model gate | `DeviceType.isSupportBladeSpeed()` | Not exposed on all models | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:678-680`; UI gate `sources/com/agilexrobotics/device/setting/fragment/appsetting/CarSettingDrawerFragment.java:2388-2405` | Entity availability |
| Blade use time | `MaintainBean.bladeUsedTime` | Lifetime/current-blade seconds; UI converts to hours | `sources/com/agilexrobotics/base_module/entity/MaintainBean.java:9,15-17`; `sources/com/agilexrobotics/device/info/activity/DeviceMaintainActivity.java:84-118` | High sensor |
| Blade warning threshold | `MaintainBean.bladeUsedWarnTime`; `setBladeWarningTime(int)` | Configurable replacement reminder, UI range 1–150 hours | entity `sources/com/agilexrobotics/base_module/entity/MaintainBean.java:10,19-21`; command `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1692-1695`; strings `resources/res/values/strings.xml:1009-1011` | High number |
| Reset blade timer | `resetBladeTime()` | Marks blades replaced/resets counter | `sources/com/agilexrobotics/device/info/activity/BladeReplacementActivity.java:54-60`; `sources/com/agilexrobotics/device/info/activity/DeviceMaintainActivity.java:137-141` | High button |
| Replacement due | `MaintainBean.shouldReplaceBlade()` | Due when nonzero warning threshold minus used time is <= 0 hours | `sources/com/agilexrobotics/base_module/entity/MaintainBean.java:55-60` | High binary sensor |
| Cutting height | robot state/setting, UI reports success/failure and selected height | Exact low-level command is outside clearly named methods in this scope; do not guess from obfuscated calls | strings `resources/res/values/strings.xml:1267-1268`; height UI references `sources/com/agilexrobotics/device/setting/fragment/appsetting/CarSettingDrawerFragment.java:1410-1519` | High number where existing integration transport identifies field |
| Total mowing time | `MaintainBean.totalMowingTime` | Lifetime counter | `sources/com/agilexrobotics/base_module/entity/MaintainBean.java:13,31-33` | High diagnostic sensor |
| Total mileage | `MaintainBean.totalMileage` | Lifetime travel counter | `sources/com/agilexrobotics/base_module/entity/MaintainBean.java:12,27-29` | High diagnostic sensor |
| No-area work | `noAreaWork()` | Starts work without mapped area on supported models | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1141-1144`; gate `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:725-727` | Medium button |
| Grass collection/no-mow mode | `sendLawnNoMow(device,boolean)`; state `collectGrassEnable` | Toggles lawn/no-mow or collection behavior (name and state disagree; treat label as uncertain) | `sources/com/agilexrobotics/device/setting/fragment/appsetting/CarSettingDrawerFragment.java:2524-2533`; view model `sources/com/agilexrobotics/device/setting/fragment/appsetting/DrawerSettingsViewModel.java:428-438` | Medium until protocol semantics confirmed |

## Weather, safety, security, locks, and animal protection

| Feature | Exact command / field | Semantics | Evidence | HA relevance |
|---|---|---|---|---|
| No mowing in rain | `sendNoCuttingInRianOrder(device,boolean)`; state `noCuttingInRianEnable` | Toggle weather stop. Caller often passes current state to helper, so helper/protocol may invert it; confirm before writing | helper `sources/com/agilexrobotics/device/setting/DeviceSettingHelperImpl.java:84-90`; UI `sources/com/agilexrobotics/device/setting/fragment/appsetting/CarSettingDrawerFragment.java:2805-2814`; error remediation `sources/com/agilexrobotics/device/info/malfunction/ErrorDetailActivity.java:140-149` | High switch with careful inversion test |
| Rain pause/error | robot pauses work and error detail offers disabling the rain setting | Current error can invoke setting change | strings `resources/res/values/strings.xml:602,1228,1446` | High binary sensor/event |
| Frost/weather protection | Resource/UI references exist for weather behavior, but no unambiguous named command was recovered in the scoped command calls | Treat as capability/uncertainty, not an implemented write | relevant settings resources located by `resources/res/values/strings.xml` weather entries; no exact scoped call | Potentially high, currently unsupported evidence |
| Anti-theft | `POST setting`, body `EquipmentSwitchBean`; shipped UI copy describes an alert beyond 50 m from the working area, but 50 m is not verified as a local command/server threshold | Cloud-side switch and track service; actual policy may be server-controlled | `sources/com/agilexrobotics/find/device/api/FindDeviceApiService.java:27-29`; UI-policy text `resources/res/values/strings.xml:976` | High switch/event; likely cloud-dependent |
| Locate/find | `POST location/sync` with `DeviceBean`; map seek UI | Requests/synchronizes current device location | `sources/com/agilexrobotics/find/device/api/FindDeviceApiService.java:31-33`; `sources/com/agilexrobotics/find/device/activity/MapSeekActivity.java:95-310` | High device-tracker/update button |
| Location history | `POST location/page`, `TrackBean(deviceName,localDate,pageNumber,pageSize)` | Paginated daily location history | API `sources/com/agilexrobotics/find/device/api/FindDeviceApiService.java:23-25`; fields `sources/com/agilexrobotics/base_module/entity/TrackBean.java:7-16,19-48` | Medium; privacy-sensitive |
| Find-service limitations | GPS/other positioning depends on network/device state and is not guaranteed | App disclaimer explicitly frames this as recovery assistance | `resources/res/values/strings.xml:943` | Document limitations; do not promise real-time tracking |
| Device lock/PIN | `DeviceLockType` exists and device details contain lock/unlock paths; PIN UX is not mapped to a clear command in this scope | Lock state/type is protocol-backed elsewhere, but no safe write signature was established here | `sources/com/agilexrobotics/device/source/device/bean/DeviceLockType.java:1-23`; device detail lock references `sources/com/agilexrobotics/device/info/activity/DeviceDetailsActivity.java:2440-2555` | High lock/binary sensor once exact transport is recovered; no speculative service |
| Animal protection | `getAnimalProtectMode(int,int)`, `getAllAnimalProtect(int,int)`, `allAnimalProtect(int,int,int)` | Reads mode/status and writes protection configuration; three integers are protocol selectors/values | helper `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:775-786,418-421`; UI `sources/com/agilexrobotics/device/setting/activity/AnimalNormalActivity.java:124-130` | High switch/select |
| “Clever” animal mode | `allpowerfullRW(...)` / `allpowerfullRWAdapterX3(...)` used by animal and generalized settings | Newer multi-parameter read/write pathway; name is decompiler/upstream jargon | helper `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:422-440`; screen launch `sources/com/agilexrobotics/device/setting/activity/CarSettingDrawerActivity.java:3173-3178` | Medium until parameter IDs are decoded |
| Do-not-disturb/work-stop | `jobDoNotDisturb(JobDNDBean)`, `jobDoNotDisturbRead(int)`, `jobDoNotDisturbDel()`; fields `deviceId`, `unableStartTime`, `unableEndTime` | Prevents work in a time window; UI uses minute-of-day strings including 1320 | `sources/com/agilexrobotics/device/setting/activity/WorkStopActivity.java:50-59,241-269`; helper `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1089-1100` | High switch + start/end time |

## Sensors, diagnostics, status indicators, and error history

| Diagnostic/status | Exact field/path | Details | Evidence | HA relevance |
|---|---|---|---|---|
| Device operational state | `ICarDevice.getStateMachine()` and `CarStateMachineBean` values | Central source for battery, charge, work, rain, cutter, connectivity, and feature status | state refresh `sources/com/agilexrobotics/device/setting/fragment/appsetting/CarSettingDrawerFragment.java:1807-2035`; signal detail `sources/com/agilexrobotics/signal/activity/CarStatusDetailsActivity.java:988-1301` | High |
| Radar | model capabilities `isSupportRadar`, `isSupportRadarRTKSwitch`, `isSupportRadarSelfCheck` | Radar status/toggle/self-check are model-dependent hardware features | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:737-745`; signal screens use gates throughout `sources/com/agilexrobotics/signal/activity/CarStatusDetailsActivity.java:1480-1690` | High binary sensors/switch if command mapping exists |
| Vision/fill light | `isSupportVision()`, `isSupportVideo()`, `isSupportFillLight()` | Model gates for camera/vision and illumination | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:701-703,761-765` | Entity availability |
| Positioning point cloud | `POST /device/point/cloud/map` and progress endpoint | Retrieves point-cloud map/progress for capable devices | `sources/com/agilexrobotics/signal/api/SignalApiService.java:81-89`; gate `DeviceType.isSupportPointCloud()` at `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:729-731` | Low/medium diagnostic |
| Error list/tips | `ErrorTinsFragment` observes mission/error tips | Presents active/history items and opens detail/video remedies | `sources/com/agilexrobotics/device/info/malfunction/ErrorTinsFragment.java:90-290` | High event/diagnostic sensor |
| Error detail actions | rain disable, DND reset, RTK/reset/navigation, video guidance | Remediation depends on error type and device state | `sources/com/agilexrobotics/device/info/malfunction/ErrorDetailActivity.java:140-149,304-315,424-530` | Medium; expose code/message first, remediation selectively |
| Error media | `VideoPlayActivity` | Some faults include instructional video | `sources/com/agilexrobotics/device/info/malfunction/VideoPlayActivity.java:42-154` | Low for HA |
| Error/network timeouts | UI handlers commonly use 5–60 s timeouts | Indicates asynchronous command acknowledgment; not itself a robot state | examples `sources/com/agilexrobotics/device/info/malfunction/ErrorDetailActivity.java:591-596`; `sources/com/agilexrobotics/device/setting/activity/DeviceNetworkActivity.java:570-608` | Integration implementation guidance |
| RTK diagnostics | Datalink/Internet mode, RTK status, LoRa versions, pairing code, station network | Distinguishes robot and station connectivity and firmware | `sources/com/agilexrobotics/signal/activity/RTKSettingActivity.java:1060-1432`; troubleshooting strings `resources/res/values/strings.xml:1447-1452` | High sensors |
| Network status detail | Wi‑Fi/4G/RTK pages and weak-signal dialogs | Connectivity indicators with polling and reconnect behavior | `sources/com/agilexrobotics/signal/activity/CarStatusDetailsActivity.java:4029-4235`; `sources/com/agilexrobotics/signal/view/SignalWeakDialog.java:35-145` | High |

The decompile does not expose a single clean “error history API” in `device/info`; the app composes error/mission-tip data from device state and shared data/view models. Therefore a HA integration should consume the existing telemetry error code/list rather than assume a REST endpoint from the UI fragment.

## Pool-cleaner and other hardware-specific settings

| Feature | Exact command/field | Details | Evidence | HA relevance |
|---|---|---|---|---|
| Pool work module | `sendSwtichSwimmingWorkModule(SwimmingWorkModule)` | Switches pool-cleaner work mode | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1623-1626`; enum `sources/com/agilexrobotics/device/source/device/enums/SwimmingWorkModule.java:1-69` | High select |
| Pool SP work module | `sendSwtichSwimmingSPWorkModule(int,List<Integer>,ICarDevice,float,int)` | More complex pool mode with zones/list, speed/float and option integer; exact field meanings need protobuf correlation | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1612-1616` | Medium |
| Timed waterline docking | `spTimedWaterlineUpdate(...)`; UI `WaterlineDockingDetailActivity.sendModeToDevice(mode,...)` | Configures waterline/docking behavior and timing | `sources/com/agilexrobotics/device/setting/activity/WaterlineDockingDetailActivity.java:350-381`; helper call sites `sources/com/agilexrobotics/device/setting/activity/CarSettingDrawerActivity.java:1968-1985` | High select/time for supported pool models |
| Pool sound hint | `SwimmingPoolDeviceStatue.isSingHint` and `sendSingTinsOpenOrClose` | Audible prompt switch | evidence in audio table | High |
| Pool history | `requestSwimmingJobHistory()` | Retrieves completed pool-cleaning history | call `sources/com/agilexrobotics/device/setting/fragment/appsetting/CarSettingDrawerFragment.java:2157-2164` | Medium |
| Pool OTA/local-add gates | `isSupportLocalAddSwimmingPool`, `isSupportLocalUpgradeSwimmingPool`, `isSupportSwimmingLowPowerVersion` | Selects local provisioning/update and low-battery behavior | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:713-721`; OTA use `sources/com/agilexrobotics/device/info/OtaCorrelationViewModel.java:790` | Required availability gates |
| Water pump/spray firmware | component-version UI fields | Indicates pump/spray controller hardware on some products | `sources/com/agilexrobotics/device/info/databinding/ActivityDeviceVersionBinding.java:257-269` | High diagnostics |

## Model and firmware gates

The app does not treat all Mammotion products as one capability set. HA entities should be capability-discovered and unavailable/omitted when unsupported.

| Gate | Meaning / affected features | Evidence |
|---|---|---|
| `isSupportBladeSpeed()` | Cutter-speed UI | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:678-680` |
| `isSupportBatteryLoopCount()` | Battery cycle maintenance statistic | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:674-676` |
| `isSupportChargeStationDeploy()` | Dock installation/location workflow | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:686-688` |
| `isSupportFillLight()` | Manual/fill light | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:701-703` |
| `isSupportNRTK()` / `isSupportRtkService()` | Network RTK/service screens | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:721-723,753-755` |
| `isSupportRadar()` / `isSupportRadarRTKSwitch()` / `isSupportRadarSelfCheck()` | Radar state, RTK interaction and self-check | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:737-745` |
| `isSupportVideo()` / `isSupportVision()` | Camera/video/wiper/vision features | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:761-765` |
| `isSupportPointCloud()` | Point-cloud map | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:729-731` |
| `isSupportNoAreaWorkDeviceModel()` | Work without mapped area | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:725-727` |
| `isSupportLocalAddSwimmingPool()` / `isSupportLocalUpgradeSwimmingPool()` | Pool provisioning/OTA transport | `sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:713-721` |
| `VersionPairStatus.VERSION_OK` | Robot/accessory firmware compatibility; settings can be blocked on mismatch | `sources/com/agilexrobotics/device/setting/fragment/appsetting/CarSettingDrawerFragment.java:455-456,3124-3137` |
| Firmware `>= 1.14.0` | RTK service/new positioning behavior | `sources/com/agilexrobotics/signal/newstatus/SignalConnectionHomepageActivity.java:1094-1103`; `sources/com/agilexrobotics/signal/activity/PositioningModeActivity.java:2179-2190` |
| `RESET_CHARGE_PILE_VERSION` | New charging-station reset workflow | `sources/com/agilexrobotics/device/setting/activity/MapSettingActivity.java:797-810` |
| `isOldAnimalVersion()` / model exclusions | Old versus clever animal-protection UI | `sources/com/agilexrobotics/device/setting/fragment/appsetting/CarSettingDrawerFragment.java:2270-2285` |
| `isSupportUploadLogVersion()` | Device log upload | `sources/com/agilexrobotics/device/setting/fragment/appsetting/CarSettingDrawerFragment.java:3192-3201` |
| `isSupportAutoUpgrade()` | Automatic-update switch | `sources/com/agilexrobotics/device/info/activity/DeviceDetailsActivity.java:2025-2056` |

## Recommended Home Assistant surface

| HA platform | Candidate entities/services |
|---|---|
| `sensor` | Battery %, charge state, Wi‑Fi RSSI/SSID, 4G/RTK signal and mode, firmware installed/latest/progress, component versions, blade hours/threshold, total mowing hours, mileage, battery cycles/health, active error code/message |
| `binary_sensor` | Charging/docked, update available/in progress, blade replacement due, rain detected/paused, anti-theft enabled/alarm, Wi‑Fi/4G/RTK connected, radar/vision status, lock state |
| `switch` | Auto firmware update, rain no-mow, anti-theft, voice prompts, pool sound hints, manual/fill/side/night light, NetRTK, smart charging, off-peak charging, smart sleep, animal protection |
| `number` | Charge limit, recharge threshold, blade reminder hours, cutter speed, volume, cutting height |
| `select` | Positioning/RTK link mode, voice language/gender, cutter mode, animal mode, pool work/waterline mode |
| `time` | Off-peak start/end, DND/work-stop start/end |
| `button` | Refresh location, wipe camera, reset blade timer, charge to 100%, restart, upload diagnostics, return to dock/cancel recharge |
| update entity | Robot/accessory update availability, installed/latest versions, progress, release notes, install action |
| device tracker | Current robot location; history should be opt-in and not continuously imported by default |

Safety/destructive operations—factory reset, unbind, reset RTK/base/dock, raw OTA packet transfer, pairing-code writes—should not be ordinary entities. If implemented at all, they should be explicit services with confirmation, state checks, and model/firmware gates.

## Hidden, beta, support, and test features

| Feature | Finding | Evidence / caution |
|---|---|---|
| Beta features page | `BetaFeaturesActivity` and beta-feature item binding exist; entries are dynamically/model selected | `sources/com/agilexrobotics/device/setting/activity/BetaFeaturesActivity.java:1-164`; `sources/com/agilexrobotics/device/setting/databinding/ItemBetaFeatureBinding.java:1-58` |
| Maintenance test override | If preference `MaintenanceTools == 1`, `MaintainBean.shouldReplaceBlade()` forces `bladeUsedTime=3600` seconds before evaluation | `sources/com/agilexrobotics/base_module/entity/MaintainBean.java:55-60`; clearly test behavior, never reproduce in HA |
| Test-tool entities | `base_module/entity/testtool/` includes device test tab data | `sources/com/agilexrobotics/base_module/entity/testtool/event/DeviceTestToolTabBean.java:1-45`; no evidence this is normal consumer functionality |
| OTA flash clear | `PopupOtaFlashClearBinding` exists | `sources/com/agilexrobotics/device/info/databinding/PopupOtaFlashClearBinding.java:1-99`; potentially destructive/service-only |
| Drop-mow | `DropMowActivity` and notification settings exist | `sources/com/agilexrobotics/device/setting/activity/DropMowActivity.java:1-406`; `sources/com/agilexrobotics/device/setting/activity/DropMowFunctionNotificationActivity.java:1-62`; behavior is product/beta-specific |
| Edgewise mapping | command `setEdgewiseMapping(int)` exists without a prominent normal settings flow | `sources/com/agilexrobotics/command/app/MACommandApiHelper.java:1801-1804`; latent/model-specific |
| Misleading `AudoBackwashPop` class name | Not a product feature: the generic confirmation dialog is used with `popup_only4g_cut` before disabling the only active 4G path. No backwash strings or command callers were found. | `sources/com/agilexrobotics/signal/view/AudoBackwashPop.java:32`; `sources/com/agilexrobotics/signal/newstatus/StatusBarExtend4gActivity.java:348,524` |
| Raw battery setter | `CommandManager.setBatteryInfo(...)` is a legitimate charging-policy setter, while `MACarDataManager.setDeviceBattery(String,int)` appears to update cached state, not robot charge | `sources/com/agilexrobotics/command/app/MACarDataManager.java:8666-8675`; avoid mistaking cache mutation for a command |

## Uncertainties and reverse-engineering cautions

1. `allpowerfullRW`, `allAnimalProtect`, and several integer-rich methods are generated wrappers around protocol IDs. Their exact parameter enums cannot be safely named from method signatures alone.
2. Boolean setters are sometimes called with the current value and may invert inside the command helper. Rain no-mow is the clearest example. Validate wire payloads or acknowledgments before implementing writes.
3. Camera/vision/video capability and a wiper action are proven, but this scope does not prove an externally usable live camera-stream endpoint.
4. Frost protection appears in product resources/behavioral concepts but lacks a clear named scoped command; it should remain unimplemented until a protobuf field or command ID is traced.
5. Lock/PIN concepts exist, but the exact safe read/write command is not established in these packages. Do not infer it from UI labels.
6. Some screens are duplicated as an activity and a drawer fragment, and old/new or mower/pool paths coexist. Duplicate UI does not imply duplicate protocol features.
7. HTTP paths without `Domain-Name: base_url` may use a module-specific Retrofit base URL. Paths in this document are therefore route suffixes, not standalone public URLs.
8. Decompiled line numbers are stable only for this decompile. Re-running JADX or changing options will move them.

## Files reviewed

The review inventoried all files in the four requested package trees (81 `device/info`, 94 `device/setting`, 24 `find/device`, and 103 `signal` files at review time), then read the behavior-bearing activities, fragments, view models, API interfaces, entities, command helpers, capability enum, bindings/layout IDs, and matching resources. Generated Hilt injectors, `BuildConfig`, module `R.java` constant dumps, and most pure data-binding boilerplate were inventoried but not individually behavior-audited except where binding IDs established visible hardware/version fields.

Principal behavior files:

- `sources/com/agilexrobotics/device/info/activity/{DeviceDetailsActivity,DeviceMaintainActivity,BladeReplacementActivity,DeviceVersionActivity,VersionHistoryActivity}.java`
- `sources/com/agilexrobotics/device/info/{OtaCorrelationViewModel,SwimmingPoolOtaCorrelation,DeviceUpdateHelperImpl}.java`
- `sources/com/agilexrobotics/device/info/view/{FirmwareUpdateView,FirmwareUpdateKTView,FirmwareUpdatePop,VersionPairPop}.java`
- `sources/com/agilexrobotics/device/info/malfunction/{ErrorTinsFragment,ErrorDetailActivity,VideoPlayActivity}.java`
- `sources/com/agilexrobotics/device/info/api/DeviceInfoApiService.java`
- `sources/com/agilexrobotics/device/setting/activity/{CarSettingDrawerActivity,DeviceNetworkActivity,VoiceNormalActivity,CarDrivingActivity,BladeSpeedActivity,AnimalNormalActivity,AnimalCleverActivity,WorkStopActivity,MapSettingActivity,NetRtkActivity,WaterlineDockingDetailActivity,BetaFeaturesActivity,DropMowActivity}.java`
- `sources/com/agilexrobotics/device/setting/fragment/appsetting/{CarSettingDrawerFragment,DrawerSettingsViewModel}.java`
- `sources/com/agilexrobotics/find/device/activity/{MapSeekActivity,TrackActivity}.java`
- `sources/com/agilexrobotics/find/device/viewmodel/{MapSeekViewModel,TrackViewModel}.java`
- `sources/com/agilexrobotics/find/device/api/FindDeviceApiService.java`
- `sources/com/agilexrobotics/signal/activity/{CarStatusDetailsActivity,RTKSettingActivity,PositioningModeActivity,INaviBoxActivity,INaviDetailActivity}.java`
- `sources/com/agilexrobotics/signal/newstatus/{SignalConnectionHomepageActivity,SpinoPileSettingActivity,StatusBarExtend4gActivity,StatusBarExtendWiFiActivity}.java`
- `sources/com/agilexrobotics/signal/fragment/{NetRtkFragment,StatusBarExtend4gFragment,StatusBarExtendWiFiFragment}.java`
- `sources/com/agilexrobotics/signal/viewmodel/{CarStatusDetailViewModule,INaviViewModule,ActiveSimViewModel}.java`
- `sources/com/agilexrobotics/signal/api/SignalApiService.java`
- `sources/com/agilexrobotics/command/{CommandManager,app/MACommandApiHelper,app/MACarDataManager}.java`
- `sources/com/agilexrobotics/base_module/entity/{MaintainBean,DeviceNetInfo,DeviceVersionInfoBean,TrackBean,AllDeviceVersionCheckRequstBean,CheckDevicesResponseBean,ErrorBlockBean,NetRtkModeReq,NetRtkPairingReq,NetRtkPairingReq2,RTKOTAStatusReq}.java`
- `sources/com/agilexrobotics/device/source/device/enums/{DeviceType,SwimmingWorkModule,SwimmingSPWorkModule,VersionPairStatus,DeviceWorkState}.java`
- `sources/com/agilexrobotics/proto/MctrlOta.java` and OTA-related generated protobuf classes referenced by the update flow
- `resources/AndroidManifest.xml`, `resources/res/values/strings.xml`, and relevant `activity_*`, `fragment_*`, `dialog_*`, `pop_*`, and settings layouts/bindings.
