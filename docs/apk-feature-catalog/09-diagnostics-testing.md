# 09 — Diagnostics, engineering testing, logging, and support tools

## Scope and confidence

This catalog covers the decompiled first-party `com.agilexrobotics.testing`, `feedback`, `log`, `trace`, and `base_module/entity/testtool` code; manifest registrations; diagnostic assets/resources; beta-feature UI; and the command/API call sites reached by those tools. It describes the examined APK, not necessarily current server policy or firmware behavior.

Confidence labels:

- **Confirmed** — explicit UI, route, field, endpoint, or command call is present.
- **Inferred** — intent is clear from names/UI and adjacent state, but the receiving firmware/server implementation is outside this APK.
- **Uncertain** — decompilation, duplicate option IDs, server-delivered configuration, or missing receiver implementation prevents a firm conclusion.

## Executive findings

The production build contains a complete engineering toolbox, factory acceptance workflow, BLE/binding stress tools, pool-service tools, GNSS signal masking, charging-pile pairing, arbitrary debug configuration injection, error/simulator controls, ICCID update, and feedback/log upload. Module `BuildConfig.DEBUG` is `false`; these are deliberately packaged release features, not merely debug-build leftovers (`sources/com/agilexrobotics/testing/BuildConfig.java`, `sources/com/agilexrobotics/feedback/BuildConfig.java`).

The principal mower test menu is gated twice:

1. The app requests developer-group information and persists `APP_TEST_TOOLS_FLAG` only when the response contains group/permission `Developer:testTool`; it separately records `code-debug` (`sources/com/agilexrobotics/home/viewmodel/HomeViewModel.java:1773`).
2. An authorized account must perform a rapid repeated-click gesture; this toggles `QUICK_CLICK_FLAG`, sets `testtool=true`, exposes the menu, and may open the overlay tool when draw-over-app permission exists (`sources/com/agilexrobotics/device/setting/fragment/appsetting/DrawerSettingsViewModel.java:325`).

Pool robots are an exception: their test entry is made visible directly. The manifest-registered test activities have no intent filters and the floating service is explicitly non-exported. Activities without explicit `exported` values have no launcher/deep-link filter in this manifest, so normal exposure is in-app navigation rather than a public Android entry point (`resources/AndroidManifest.xml:645`).

## Reachability

| Surface | Reachability in this APK | Gate / entry evidence | Notes |
|---|---|---|---|
| Feedback/support | Normal UI | `FeedBackActivity`; ARouter feedback route; feedback fragments | User-facing feedback, media, app logs, and mower logs. |
| Device test tools | Hidden engineering UI | Server `Developer:testTool` permission + rapid multi-click; pool devices visible without that account gate | Opens `DeviceTestToolsActivity`; can also run in floating overlay. |
| Floating test overlay | Hidden + Android permission | Test mode plus `Settings.canDrawOverlays`; preference `floating_window_button_enabled` | Service is `exported=false`; overlay can reconnect BLE and open the full tool panel. |
| Factory test | Nested hidden UI | Tool option `1118`; also hidden shortcuts in map/settings | Requires device name and tester workflow; posts records to test/MES services. |
| GNSS/nav-star | Nested hidden UI | Tool option `1116` | Directly writes GNSS mask/config strings to the mower. |
| Charging-pile pairing | Nested hidden UI | Tool option `1150` | Reads or writes five-character GFSK pairing ID. |
| Device bind stress test | Nested hidden UI | Tool option `1121`; user-info shortcut only when whitelist + quick-click flags are true | Repeated BLE/bind/provisioning operations. |
| Environment switch | Nested hidden UI | Tool option `1127` | Changes app/server environment; exact destinations are outside this activity’s call site. |
| Debug-config editor | Nested hidden UI / server config | Tool option `13` and `ConfigSettingActivity` | Server/device returns keys; UI calls `setDebugConfig(key,value)` without a local allowlist. |
| Pool test tools | Normal for pool devices; hidden shortcut elsewhere | Pool type makes tool visible | Sends `allpowerfullRW` pool service operations. |
| Beta features | Normal device-settings UI when the containing settings item is supplied/visible | `BetaFeaturesActivity`, explicitly non-exported | Current list is model/capability dependent. |
| WorkManager diagnostics receiver | System/developer broadcast | `androidx.work.diagnostics.REQUEST_DIAGNOSTICS` | AndroidX library diagnostics, not a Mammotion mower-control menu. |

## Device test tool catalog

The authoritative menu is the bundled `assets/device_test_tools.json`, loaded by `TestUtils.getTestToolData()` (`sources/com/agilexrobotics/base_module/entity/testtool/utils/TestUtils.java:55`, `resources/assets/device_test_tools.json`). `type` controls presentation (switch, chooser, action, slider, etc.); it is not a protocol opcode. Unless intercepted locally below, the view model passes `option` and integer parameters to `sendToolsOrder`; options 22/23 take the alternate/system path (`sources/com/agilexrobotics/testing/viewmodel/DeviceTestToolsViewModel.java:348`, `sources/com/agilexrobotics/testing/repository/TestToolsRepository.java:134`).

The numeric option tables below are transcribed from the bundled engineering
configuration and dispatch code. Treat them as a discovery index, not a safe or
normative protocol specification: safety-sensitive constants require direct
source anchoring and live validation before use.

### Simulator, actuator, sensor, and endurance controls

| Option | Control (translated) | Values / exact call where visible | HA relevance | Risk |
|---:|---|---|---|---|
| `0` | Indoor simulator | `indoorSimulation(value)`; simulator state cached | Useful only for protocol research | High: defeats normal localization assumptions. |
| `-3` | After-sales RTK run | `fastAutoTest(-3)`; requires charging | Possible service mode sensor/state source | High: autonomous movement. |
| `-1` / `-2` | Indoor no-satellite / indoor visual no-satellite run | `fastAutoTest(option)`; requires charging | None for normal HA | High: autonomous movement under degraded localization. |
| `1` | Simulation speed | `500`, `1000`, `1500` | Simulator entity only | Medium. |
| `2` | Blade during simulation | switch | Potential blade diagnostic | Critical: cutting motor may run. |
| `3` | MTBF mode | switch/order | Long-run testing | High: prolonged unattended operation. |
| `4` | Vision test mode | switch/order | Vision diagnostics | Medium/high. |
| `20` | Side blade direction | `1` off, `2` forward, `3` reverse | Side-blade control if supported | Critical. |
| `21` | Side blade speed | `1` low, `2` high | Diagnostic speed entity | Critical. |
| `22` | Cutting-disc switch | `1` on, `2` off; system command path | Direct mower command candidate | Critical. |
| `23` | Cutting-disc RPM | slider; system command path; adapter maps one range from 2200–3600 | Valuable read-only RPM sensor; never expose write casually | Critical. |
| `5` | RTK switch/order | Boolean | Read-only RTK diagnostics useful | High if written. |
| `6` | Ignore collision bar | Boolean | Could expose diagnostic state only | Critical: disables collision protection. |
| `7` | Rain detection switch | Boolean | Sensor diagnostic | High: bypasses weather safety. |
| `110`–`113` | Left, left-front, right-front, right ultrasonic | per-tool parameters/order | Excellent read-only sensor candidates | Critical if they disable/forge obstacle sensors. |
| `8` | Simulate RTK off | Boolean | Fault injection only | High. |
| `9` | RTK simulation | `BAD_LOCALIZATION=0`, `RTK_SINGLE=1`, `RTK_PSRDIFF=2`, `RTK_FIX=4`, `RTK_FLOAT=5` | Useful vocabulary for HA state decoding | High: falsifies localization. |
| `10` / `11` / `12` | Battery simulation, battery offset, charging simulation | switch / slider / switch | Useful only for test integration | High: false battery/charge state can trigger movement/charging logic. |
| `19` | Send front-wheel jam | Boolean | Error-event testing | High: intentional fault injection. |
| `100`–`109` | Docking aging, enhanced docking aging, MTBF, indoor run, straight emergency-stop aging, collision aging, spin aging, Yuka shuttle/rotation aging, track-follow aging, “aging test 10” | Generic tool commands | Could reveal undocumented state transitions | High/critical: repeated autonomous motion and actuator cycling. |
| `19999` | Steering calibration | `0` release steering motor; `1` set current position as steering zero; `2` automatic calibration | Potential service button only with strong safeguards | Critical: changes calibration / releases actuator. |

### App-side diagnostics, fault injection, and feature switches

| Option | Control | Implementation / persistence | HA relevance and risk |
|---:|---|---|---|
| `1110` | Floating window | `float_switch` preference | App-only convenience; overlay increases accidental access risk. |
| `1135` | Network monitor | `KEY_PING_MONITOR_SWITCH`; immediately calls `BackendPingMonitorManager.updateMonitorEnabled` | Strong HA relevance as a connectivity diagnostic concept; app-only. |
| `1123` | IoT reduced-data display | `AppCache.isIotMessage` | Useful for decoding IoT traffic; privacy/noise risk. |
| `1122` | Factory preset switch | `FACTORY_PRESET_SWITCH_KEY`; posts `FactorySwitchEvent` | High risk: changes factory UI/workflow state. |
| `14` (first occurrence) | Selectable vision logs in feedback | `AppCache.showOldLogUI` (naming mismatch) | Log-selection behavior; duplicate ID makes exact mapping uncertain. |
| `15` (first occurrence) | NRTK | `AppCache.isNRTK` | HA capability discovery; app-side gate. |
| `13` | Parameter tuning | choices “data reporting” `500`, “config injection” `1000`, “internal errors” `1500`; opens config fetched from server/device | High: arbitrary debug key/value injection via `setDebugConfig(key,value)`. |
| `1115` | Manual mowing BLE display | preference `BluetoothDisplay` | Useful read-only BLE diagnostic. |
| `1117` | Radar panel | preference `RadarInstrumentPanel`, plus RxBus event | Good read-only radar diagnostic candidate. |
| `1126` | Force device version update | action type | Firmware lifecycle is relevant, but forcing updates is high risk. Exact call was not recovered at this menu branch. |
| `1119` | VIO path comparison line | local preference; disabling clears VIO/RTK lines | App visualization only. |
| `1129` | Mammotion logo/log pattern | preference `LogPatternPrinting` | Likely rendering/log instrumentation; low operational risk. |
| `1139` | Real-time blade RPM display | preference `RealTimeSpeedOfTheCutterhead` | Excellent read-only HA sensor candidate. |
| `14` / `15` (second occurrences) | Vision-obstacle data / ultrasonic data | generic tool command due duplicate IDs unless list/order handling disambiguates elsewhere | Very useful telemetry, but duplicate IDs create dispatch ambiguity. |
| `16` | Map fog switch | generic command | App/map diagnostic; low/medium. |
| `17` | App log switch | `LogUtil.updateIsDebug(value==1)` | Useful for troubleshooting; sensitive-data/volume risk. |
| `18` | Simulate map load failure | reverses `MapTransportUtils.success` | Fault injection; app-only. |
| `24` | H5 environment switch | `HttpConstants.isReleaseTool` state | Risk of talking to non-production content/services. |
| `200` | Simulate IoT offline | `AppCache.BIND_4G_IOT_SWITCH` | Useful HA failure testing; app binding workflow only. |
| `201` | Simulate internal SIM | `BIND_4G_CARD_SWITCH` | ICCID/SIM workflow testing. |
| `202` | SIM requires activation | `BIND_4G_ACTIVATION_SWITCH` | SIM activation workflow testing. |
| `203` | 4G activation master switch | `BIND_4G_TEST_SWITCH` | SIM workflow testing. |
| `204` | Binding error simulation | `28612` already bound by admin; `28573` outside binding window; `6618` device offline | Useful error-code mapping for HA diagnostics. |
| `205` | Simulate fill/night light | `setCarLampCtrlNightLight(value==1, 2, 1122)` | Potential light entity, but engineering call carries magic source `1122`. |
| `206` | Show abnormal status bar | preference `isCvStatusBarVision` | UI-only. |
| `207` | No work outside lawn / regional protection | checks capability then writes regional-protection value | Relevant setting candidate; high safety/geofence risk. |
| `208` | Blade used one hour | preference `MaintenanceTools` | Maintenance-state simulation; useful concept, app-only. |
| `209` | RTK-base score simulator | levels `0/1/2` set scores `100/81/65` | Helps interpret signal score thresholds. |
| `210` | MN231 RTK tool | `isOpenMN231RTKDisplay` | Read-only display candidate. |
| `211` | Abnormal-phenomena display | `IsExceptionNotifyShow` | Useful error visibility. |
| `212` | Low-power tool | `19` enters low-power through timed command; `20` POST wakeup; `21` POST location sync | Useful wake/sleep status; high availability risk. |
| `213` | PC210 exhibition mode | `isPC210MeetingDisplay` | App presentation mode. |
| `1100` | 231 RTK test switch | preference `VisionTest231RtkTool` | Diagnostic display. |
| `1120` | BLE connection stress tool | Enables `BLETestToolHelper`, repeated disconnect/connect, counts success and average connection time | Excellent robustness methodology; high disruption risk. |
| `1121` | Device-binding stress tool | Opens bind test; accepts device name, SSID, password, mode | Credentials/privacy and repeated provisioning risk. |
| `1124` | Control-command parameter display | `CarRemoteControlManage2.isViewVisible` | Highly relevant for protocol discovery; may expose sensitive raw values. |
| `1125` | Control-command frequency | slider, generic command path | Useful rate testing; high flood/actuator risk. |
| `1127` | Environment switch | navigation to `DeviceChangeTestActivity` | High account/data-isolation risk. |
| `1130` | INaviBox display | `AppCache.isINaviBoxDisPlay` | Read-only navigation diagnostic candidate. |
| `1131` | RTCM-ready display | preference `rtc_ready` | Strong HA diagnostic sensor candidate. |
| `1132` | Radar detection | preference `radar_detect` | Strong HA diagnostic sensor candidate. |
| `1133` | Connection-type display | preference `connect_type` | Strong HA diagnostic sensor candidate. |
| `1134` | Optional log upload | preference `upload_logs_optional` | Enables granular log selection only while test mode is also true. |

## Command and protocol call sites

| Function | Exact visible call / fields | Interpretation and uncertainty |
|---|---|---|
| Generic engineering command | `sendToolsOrder(option, List<Integer>)`; options 22/23 call `testToolOrderToSys(2, option, list)` | Receiver/proto serialization is below `MACommandApiHelper`; numeric semantics come from the bundled menu. |
| Debug mode | `setDebugEnable(value)`; startup can call `setAllDebugConfig()` | Writes mower debug state/config. |
| Debug configuration | `setDebugConfig(key, value)` | No client-side key allowlist is visible; keys may be server/device supplied. |
| Power/service R/W | `allpowerfullRW(type, value, rw)` | Pool view reads types `21`, `22`, `23` with `(type,0,0)` and writes `(type,isOpen,1)`; repository also uses type `11`. Meanings beyond the UI labels remain firmware-dependent. |
| Pool work mode | `sendSwtichSwimmingWorkModule(WATER_WORK)` then later `AUTO_WORK` | Factory test temporarily changes pool robot work mode. |
| Factory item | `factoryTestOrder(id, (int)time_1, expect)` | Runs a selected acceptance test with duration/expected value. |
| Charging-pile pairing | read `readAndSetGfskCfg(0,"")`; write `readAndSetGfskCfg(1,pairId)` | Pair ID is uppercased and limited to five characters. |
| GNSS sync | `synNavStarPointData(255)` | Requests all nav-star configuration. |
| GNSS commands | `setNavStarPoint("mask\\0")`, `"saveconfig\\0"`, and `"{mask\|unmask} bds prn {N}\\0"` | Exact NUL-terminated shell-like strings are visible. Other constellation/signal writes are assembled from numeric group/item IDs. |
| Night/fill light | `setCarLampCtrlNightLight(enabled, 2, 1122)` | Real device light command. |
| Regional protection | repository capability check then write | Exact helper method is present in repository; firmware semantics should be confirmed before HA exposure. |
| Device wake/location | POST `/device-server/v1/device/wakeup` and `/device-server/v1/device/location/sync`, body `DeviceNameReq` | Cloud-side operations, not direct mower protocol. |
| ICCID bind update | POST `/tests-server/v1/device/iccid`, body `TestUpIccidReq` | Factory/support operation; can alter SIM-device association. |
| Pool appointed firmware | POST `/device-server/v1/pool-robot/appoint-version-upgrade`, body `PoolRobotVersionUpgradeReq` | Dangerous lifecycle action. |

## Factory acceptance, ICCID, pairing, and error tools

`FactoryTestActivity` loads server test configuration and/or the bundled `factory_test_tool.json`, collects tester name/device/date, executes tests, stores a local JSON record, and uploads results through `record` and legacy MES `mesateapi.asmx/KMAPPCode`. The bundled checks are:

| IDs | Checks |
|---|---|
| `1–4` | Left/right front bump, emergency-stop button, emergency-stop unlock |
| `6–8` | Rain sensor, left tilt, right tilt |
| `9–12` | Ultrasonic clear for 3 seconds, then left/center/right obstruction for 3 seconds |
| `14–20` | BLE signal, mower satellite count, base satellite count, mower/base signal quality, base connection, positioning status |
| `22–24` | Audible voice, docking, indoor run (manual pass/fail class) |

Evidence: `resources/assets/factory_test_tool.json`, `sources/com/agilexrobotics/testing/view/FactoryTestPop2.java:190`, `sources/com/agilexrobotics/testing/api/ProductTestingApiService.java:25`.

The factory screen has dedicated ICCID update row/binding and calls the tests-server ICCID endpoint. Exact `TestUpIccidReq` fields should be taken from that entity before any implementation; this review confirms the mutable association operation, not server authorization behavior.

`ErrorInsideActivity` and `InsideErrorAdapter` expose internal error collections received from test/tool data. `FactoryTestPop2` also reads `DeviceMessageDB.errorCode` while evaluating tests. This is diagnostically useful but does not establish a stable public error schema.

## Navigation-star tool

The bundled configuration exposes GPS (`L1/L2/L5`), BDS (`B1/B2/B3/BD3B1C/BD3B2A/BD3B2B`, plus `MASK BD2`), GLONASS (`R1/R2/R3`), Galileo (`E1/E5A/E5B/E6C`), and QZSS (`Q1/Q2/Q5/Q1C`) (`resources/assets/device_nav_star_point2.json`).

The activity can mask/unmask whole signal groups and individual BDS PRNs, retries a BDS PRN command up to three times, requests state with `synNavStarPointData(255)`, and persists with `saveconfig\0`. This is one of the most dangerous menus: saved satellite exclusions can silently degrade positioning after the session ends.

## Feedback, telemetry, and log upload

### User-visible workflow

Feedback supports device selection, feedback categories, description/contact email, photos, video, app logs, mower logs, retry/cancel, Wi-Fi quality checks, and a reduced 4G path. Strings explicitly warn not to close the app or power off the mower during transfer and state that 4G sends simplified logs while Wi-Fi enables full diagnostics (`resources/res/values/strings.xml:593`, `resources/res/values/strings.xml:1674`, `resources/res/values/strings.xml:2431`).

Optional per-path mower-log selection is enabled only when both `upload_logs_optional` and `testtool` preferences are true (`sources/com/agilexrobotics/feedback/fragment/Feedback1Fragment.java:482`). The server returns a hierarchy of log names/paths from `device-server/v2/feedback/log-path`; the client propagates group selection to child paths.

### Network operations and fields

| Endpoint | Payload / purpose |
|---|---|
| `POST fault/report` | Legacy fault report |
| `POST feedback` | Legacy feedback |
| `POST device-server/v2/feedback` | Optional/new feedback |
| `POST device-server/v2/feedback/log-path` | Body `deviceName`; returns selectable device log paths |
| `GET log-server` | Retrieves log server URL/config using query map |
| `POST issue-instruction` | Requests mower log upload; `deviceId`, `logBizId`, `logType` |
| `POST logProgress` | Polls device log progress |
| `POST support-4G-upload` | Checks/activates simplified cellular upload support |
| `POST videoInfo` | Persists uploaded video metadata |
| File upload URL (`HttpConstants.UPLOAD_FILES`) | Multipart field `files`; images/video/app-log archives |

Feedback report fields visible in the APK include `bizId`, `contactEmail`, `description`, `deviceId`, `deviceName`, `deviceVersion`, `feedbackType`, `images`, `logBizId`, `logFiles`, `logPath`, `logType`, `productKey`, `reportType`, fixed/default `type=7`, and `video` (`sources/com/agilexrobotics/feedback/api/entity/FaultReportBean.java:13`, `sources/com/agilexrobotics/feedback/api/entity/OptionalFeedbackReq.java:11`, `sources/com/agilexrobotics/feedback/api/FeedbackApiService.java:20`).

Privacy/risk: uploads can combine account/contact data, device identity/version, free text, images/video, app logs, mower logs, and server-selected internal file paths. Logs may contain BLE identifiers, network state, device names, command parameters, location/RTK data, errors, or credentials depending on upstream logging. The client uses bearer authorization, but retention, redaction, and server-side path validation cannot be established from the APK.

### Local logging and telemetry

- `AppLogUtils` writes XLog files only after `init(path,true)` grants its internal permission flag. Files use a custom filename/flattener, never-backup strategy, and delete-after-age cleaning of `259200000 ms` (three days) (`sources/com/agilexrobotics/log/AppLogUtils.java:49`).
- Automated BLE tests tag records with device, aim, module, subfunction, and source class. They record connection attempts, disconnect codes, success counts, and average connection time.
- `LogUtil` supports a runtime debug toggle (`option 17`), so release code can increase diagnostic output.
- Firebase telemetry logs arbitrary event names with one `data` string, combining account and caller-provided text, truncated to 98 characters; the default event is `login_failure_reason_report` (`sources/com/agilexrobotics/trace/firebase/FirebaseBuryPointUtils.java:17`).
- Mammotion also has a first-party event collection pipeline:
  `/user-server/v1/user/collection`, reached by `TraceHelper`, with queued
  payload fields for app/device/product/phone identity, event ID/value, time,
  and area (`sources/com/agilexrobotics/base_module/trace/ma/TraceApiService.java:14`;
  `sources/com/agilexrobotics/base_module/trace/ma/TraceHelper.java:73`;
  `sources/com/agilexrobotics/base_module/trace/ma/DataCollectBeans.java:5`).
  This is privacy-sensitive telemetry and should not be reproduced by HA.

### Downloaded error-code catalog

The app's error vocabulary is not limited to bundled strings. Startup checks a
remote error-code version and downloads localized, paginated code/remediation
data through `code/version` and `code/page-lan`
(`sources/com/agilexrobotics/home/api/HomeApiService.java:87,122`;
`sources/com/agilexrobotics/activity/MainLayoutActivity.java:342`;
`sources/com/agilexrobotics/home/api/HomeApiUtils.java:49`). Any future error
decoder should therefore preserve unknown codes and avoid assuming the APK's
static catalog is exhaustive.
- The APK includes third-party AndroidX WorkManager diagnostics and transport/analytics components. Their presence is not evidence that every library diagnostic is reachable through Mammotion UI.

## Beta features

`BetaFeaturesActivity` is a normal, non-exported settings screen whose list is assembled at runtime and routes by adapter position (`sources/com/agilexrobotics/device/setting/activity/BetaFeaturesActivity.java`). Adjacent manifest/settings evidence identifies at least driving, animal-protection, drop-mow, blade-speed, and related model-dependent feature screens. Because the decompiled list construction is partially collapsed and capability/server data can alter entries, position-to-feature mapping is **uncertain** and should not be treated as a stable API.

## HA integration relevance

Recommended read-only candidates:

- Connection type, ping/backend reachability, BLE reconnect statistics.
- RTCM-ready, RTK fix mode, mower/base satellite counts and signal quality.
- Radar/ultrasonic/vision-obstacle observations.
- Real-time cutting-disc RPM and maintenance hours.
- Internal error/fault events and log-upload progress.
- Charging-pile pair ID as a diagnostic attribute only if firmware offers a safe read.

Do **not** expose these as ordinary HA controls:

- Blade/side-blade direction or RPM, steering release/zero/calibration.
- Collision/rain/ultrasonic bypasses, RTK masking, regional-protection changes.
- Indoor/no-star/MTBF/aging autonomous runs.
- Battery/charge/localization fault simulation.
- Factory pairing, ICCID reassociation, forced firmware update, environment switch, arbitrary debug config.

If any service action is implemented, require an explicit service-mode enable, device-local authorization, mower stationary/docked checks, short expiry, confirmation describing physical motion, audit logging, and no automation eligibility by default.

## Important uncertainties

1. The APK calls into `MACommandApiHelper`; firmware-side validation and final protobuf/topic encoding were not reconstructed for every helper in this scoped review.
2. Duplicate menu option IDs `14` and `15` label different controls. Dispatch is integer-based, so behavior may depend on menu revision, ordering, model filtering, or a decompilation artifact.
3. Some test configuration is server-delivered (`config/page`), so the bundled JSON is a baseline, not proof of every production control.
4. Developer permission is server-derived, but cached preferences and hidden shortcuts exist. Server revocation, cache lifetime, and anti-tamper behavior are not proven.
5. Manifest inclusion proves packaging, not successful execution on every model/firmware.
6. “Read” and “write” meanings for `allpowerfullRW` are inferred from the third argument (`0` during initialization, `1` during toggles) and should be verified against captured traffic.

## Files reviewed

Manifest/resources/assets:

- `resources/AndroidManifest.xml`
- `resources/assets/device_test_tools.json`
- `resources/assets/device_nav_star_point.json`
- `resources/assets/device_nav_star_point2.json`
- `resources/assets/factory_test_tool.json`
- `resources/res/values/strings.xml`
- Diagnostic/factory/test/feedback layouts and generated bindings under `resources/res/layout`

First-party source reviewed:

- All files under `com/agilexrobotics/testing/`, including activities, bind-device tests, API/service definitions, floating overlay, repository, view models, adapters, presenter, helper, and factory popup.
- All files under `com/agilexrobotics/feedback/`, including activity/fragments, service, API entities/endpoints, model/view model/presenter, upload tasks, optional-log selection, and dialogs.
- All files under `com/agilexrobotics/log/` and `com/agilexrobotics/trace/`.
- All files under `com/agilexrobotics/base_module/entity/testtool/`.
- Reachability and adjacent feature call sites in `HomeViewModel`, `DrawerSettingsViewModel`, `CarSettingDrawerActivity`, `CarSettingDrawerFragment`, `HomeMapFragment`, `UserInfoActivity`, `BetaFeaturesActivity`, `AppCache`, `AppConstants`, `HttpConstants`, `RouterHub`, and generated ARouter route tables.
- Command-facing types/call sites in `MACommandApiHelper` consumers, factory/test request entities, device message/config events, and pool work-mode calls.

Third-party packages were inspected only where manifest or first-party code made them relevant; generic library test/debug classes were not mistaken for Mammotion support features.
