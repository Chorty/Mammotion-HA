# Mammotion Android app architecture, terminology, and data-flow reference

## Scope, notation, and confidence

This is a static-analysis reference for the decompiled Android application at
`/Users/mattjoslin/mammotion-apk-decompile/src`, informed by feature reports
`01`–`09` in this directory. It describes the examined APK, not a public
Mammotion protocol contract and not necessarily current firmware or backend
behavior.

Unless a path starts with `docs/`, source paths are relative to
`/Users/mattjoslin/mammotion-apk-decompile/src/sources/com/agilexrobotics/`.
Manifest/resource paths are relative to
`/Users/mattjoslin/mammotion-apk-decompile/src/resources/`.

- **Observed** means the behavior is directly represented in decompiled code.
- **Inference** means the named architectural role or end-to-end relationship is
  reconstructed from multiple call sites. An inference is not a recovered
  design document.
- Generated protobuf Java is useful evidence for fields and enums, but generated
  descriptors do not prove that every message is active on every model.
- Decompiled names, spelling, and branch structure may be imperfect. Preserve
  raw enum values, message IDs, hashes, and frames during protocol work.

## Architecture at a glance

```text
Android process / MyApplication
  |
  +-- reflected feature initializers (RN, login, map, bind, home, ...)
  +-- account, HTTP, SSE, push, MQTT and lifecycle services
  +-- DeviceManager: account-visible ICarDevice objects
        |
        +-- MADeviceManager: one MACarDataManager + MALinkManager per device
              |
UI / RN bridge / ViewModel / helper
              |
        CommandManager or MACommandApiHelper
              |
        generated protobuf payload inside LubaMsg
              |
              +-- BLE: MALinkManager -> EspBleManager -> BLUFI custom data -> GATT
              |
              +-- cloud: MALinkManager -> MAIotManager -> service invoke
                    `device_protobuf_sync_service`
                    MQTT returns asynchronous status/property/protobuf events
              |
        MACarDataManager/API parses LubaMsg reports
              |
        state-machine beans + request-result LiveData + listeners/events
              |
        local caches/DB and UI redraw
```

**Inference:** this is a layered but not strictly clean architecture. The same
responsibilities recur in older/newer, API/non-API, mower/pool, and
Ali-IoT/MA-IoT variants. Treat pairs such as `MACommandHelper` /
`MACommandApiHelper` and `MACarDataManagerAPI` / `app/MACarDataManager` as
parallel generations or façades, not automatically interchangeable
implementations. Evidence:
`command/MACommandHelper.java:180-235,316-334`;
`command/app/MACommandApiHelper.java:180-206,304-337`;
`command/MACarDataManagerAPI.java:960-1317`;
`command/app/MACarDataManager.java:8424-8576`.

## 1. App startup and modules

The manifest names `com.agilexrobotics.MyApplication` as the process
application. Hilt wraps it, while `BaseApplication` extends the Alibaba IoT
`AApplication`. Startup is therefore both Android/Hilt startup and a
Mammotion-owned feature-initializer sequence. Evidence:
`AndroidManifest.xml:154-159`;
`Hilt_MyApplication.java:13-44`;
`base_module/base/BaseApplication.java:16-33,61-88`.

`BaseApplication.onCreate()` calls `appInit()`, then reflectively constructs each
registered `BaseApplicationInit` and invokes its `onCreate()`. Configuration,
low-memory, and termination callbacks are forwarded to those instances.
Evidence: `base_module/base/BaseApplication.java:25-39,61-88`.

`MyApplication.appInit()` registers these modules in order:

1. `RNContainerApplication`
2. `LoginApplication`
3. `ProductTestingApplication`
4. `MapApplication`
5. `BindDeviceApplication`
6. `DeviceShareApplication`
7. `ServiceTabApplication`
8. `DeviceInfoApplication`
9. `HomeApplication`
10. `DeviceDeployApplication`
11. `SignalApplication`
12. `DeviceSettingApplication`

Evidence: `MyApplication.java:323-337`. Each class is a lightweight module hook;
examples are `rn/RNContainerApplication.java:14-27`,
`map/MapApplication.java:13-28`, and `home/HomeApplication.java:11-25`.

Before and after the module hooks, `MyApplication` initializes logging,
preferences, crash protection, language state, ARouter, LitePal, network
clients, thread pools, MMKV, activity tracking, device-receiver creation, app
foreground checks, MA-IoT, and a high-priority notification channel. Evidence:
`MyApplication.java:260-310,580-634`.

The `DeviceMessageCallBack` installed at startup is important: when a device
needs a receiver, it reuses an existing `MACarDataManager` by device name or
creates one and registers it with `MADeviceManager`. This is the process-level
bridge from device inventory to per-device protocol state. Evidence:
`MyApplication.java:598-616`.

**Inference:** the classes named `*Application` are feature-module initializers,
not separate Android processes or applications. Their registration order may
matter, but no general dependency graph is declared beyond that order.

## 2. Device abstraction and model gates

### 2.1 Identity and ownership

`ICarDevice` is the common robot/base abstraction. It carries product key,
nickname, link type, bind time, local/selected state, and exposes both a
`CarStateMachineBean` and a `CarWorkingStateMachineBean`. It also centralizes
questions such as owned, connected, local, MA-IoT, Ali-IoT, RTK, pool, X5,
vision, and radar. Evidence:
`device/source/device/interfaces/ICarDevice.java:18-33,57-159`.

Do not collapse these identifiers:

| Identifier | Role |
|---|---|
| `deviceName` | Firmware/cloud-visible device name and common lookup key. |
| `deviceId` | Account/backend device identity used by ownership and manager APIs. |
| `iotId` | Cloud IoT routing identity. |
| `productKey` | Cloud product family and an input to model resolution. |
| `DeviceType` | App enum derived from names/product keys; a coarse capability gate. |
| `intMod` / inside code | Finer hardware/SKU key used by multi-model configuration. |
| BLE address | Local GATT reconnect key, persisted separately per device. |

Evidence:
`device/source/device/interfaces/IDevice.java`;
`device/source/device/interfaces/ICarDevice.java:42-45,60-67,82-124`;
`device/source/device/db/DeviceBleAddressDB.java:15-31`;
`device/source/device/helpers/DeviceMultimodelHelper.java:44-74`.

`DeviceManager` owns the account-facing device list, ownership and identity
maps, selection, and listeners. `MADeviceManager` owns the per-device data/link
manager instances. They are related but not synonyms. Evidence:
`device/source/device/manager/DeviceManager.java:151-170,186-201`;
`device/source/links/managers/MADeviceManager.java`;
`MyApplication.java:598-616`.

### 2.2 Model resolution and capability gates

`DeviceType` maps numeric values, product-name prefixes, code names, and product
keys to mower, RTK, pool, and dock families. It then exposes many hard-coded
feature predicates (`isSupportIOT`, `isSupportVideo`, `isSupportVision`,
`isSupportRadar`, `isSupportPointCloud`, pool predicates, and others).
Evidence:
`device/source/device/enums/DeviceType.java:12-53,62-90,442-486,700-790`.

`DeviceMultimodelHelper` adds a second gate layer. It loads
`device_configuration_form.json` by internal model and
`device_configuration_form_2.json` by product key, and supplies model-specific
limits such as control speed, cutting height, and route spacing. Evidence:
`device/source/device/helpers/DeviceMultimodelHelper.java:27-38,76-147,149-180`;
`assets/device_configuration_form.json`;
`assets/device_configuration_form_2.json`.

Firmware/version helpers and server-returned function/config data add further
gates. Reports 02–08 show UI branches based on family, firmware, region,
ownership/share mode, link type, and downloaded settings, not only
`DeviceType`. Representative evidence:
`device/source/device/utils/DeviceVersionUtils.java`;
`device/source/device/extensions/DeviceTypeExtensionsKt.java`;
`device/source/device/utils/DeviceUtils.java:404-439,952-1006,1082-1134`;
`work/setting/api/WorkingSettingManage.java:64-77,233-252`.

**Inference:** capability evaluation is additive:

```text
coarse family enum
  + product key / internal SKU configuration
  + firmware comparison
  + cloud function/config response
  + account ownership/share and region
  + current transport and live state
  = feature actually enabled
```

Consequently, model-name checks alone are insufficient for an integration.

## 3. Connectivity: BLE, BLUFI, cloud, and MQTT

### 3.1 Link abstraction

`MALinkManagerAPI` exposes `NONE`, `BLUETOOTH`, and `IOT` link states plus
connection listeners. `MALinkManager` prefers a successful BLE connection,
falls back to IoT after BLE failure/off/disconnect when network and device
presence permit, and may continue probing BLE while using IoT. Evidence:
`device/source/links/MALinkManagerAPI.java:74-129`;
`device/source/links/managers/MALinkManager.java:89-130,220-235,258-303,448-607`.

Wi-Fi and cellular are generally robot uplinks to the same cloud command route;
they are not separate phone-to-robot command protocols in this APK. Evidence:
`device/source/links/managers/MAIotManager.java:633-743`;
`signal/newstatus/SignalHelper.java:621-735,792-808`.

### 3.2 BLE and BLUFI envelopes

BLUFI serves two jobs:

- onboarding: device-side Wi-Fi scan and station SSID/password provisioning;
- runtime: custom-data subtype 19 carrying higher-level command bytes.

Evidence:
`device/source/links/managers/EspBleManager.java:152-237,736-779,1090-1107`;
`espressif/BlufiClientImpl.java:1175-1185,1406-1432`;
`espressif/params/BlufiParameter.java:41-49`.

BLUFI frames contain type/subtype, sequence, payload length, and frame-control
bits for encryption, checksum, direction, acknowledgement, and fragmentation.
The BLE manager targets MTU 200 and handles GATT errors/reconnects. Evidence:
`espressif/FrameCtrlData.java:8-53`;
`espressif/BlufiClientImpl.java:746-845`;
`device/source/links/managers/EspBleManager.java:63-96`.

Runtime egress is:

```text
LubaMsg.toByteArray()
  -> MALinkManager.sendData()
  -> EspBleManager.postCustomeDateByte()
  -> BlufiClientImpl.postCustomData()
  -> BLUFI custom-data frame(s)
  -> GATT characteristic
```

Evidence:
`command/MACommandHelper.java:180-235`;
`device/source/links/managers/MALinkManager.java:410-440`;
`device/source/links/managers/EspBleManager.java:1090-1107`;
`espressif/BlufiClientImpl.java:1406-1432`.

### 3.3 Cloud command envelope

For current MA-IoT, command bytes are represented in a JSON/service invocation
and posted through `MAIotManager`. The service identifier is
`device_protobuf_sync_service`; `/v1/mqtt/rpc/thing/service/invoke` receives
IoT ID, product key, device name, identifier, and arguments. Evidence:
`device/source/links/managers/MAIotManager.java:633-743`;
`maiot_module/bean/request/ServiceInvokeReq.java:18-71`;
`maiot_module/api/MaIoTApiService.java`;
`maiot_module/utils/Constants.java:18-60`.

The legacy `iot_module` retains Alibaba device-panel login, token, event
subscription, property query, and service invocation. `ICarDevice.linkType`
distinguishes Ali IoT (`"0"`), MA-IoT (`"1"`), and local (`"2"`). Evidence:
`iot_module/third/api/device/DispatchNetAPI.java:52-107`;
`iot_module/third/api/device/DevicePanelApi.java:71-159`;
`device/source/device/interfaces/ICarDevice.java:90-120`.

**Inference:** “same command over BLE or cloud” means the inner protobuf
semantics are substantially shared. It does not mean the outer transport,
authentication, retry, acknowledgement, or latency semantics are identical.

### 3.4 MQTT receive envelope

The app obtains JWT MQTT credentials from `/v1/mqtt/auth/jwt`; the response
contains host, client ID, username, and JWT password. Paho uses TLS, a 60-second
keepalive, 14-second connection timeout, and automatic reconnect. Evidence:
`maiot_module/MQTTService.java:186-220`;
`maiot_module/mqtt/MQTTClient.java:236-280`;
`maiot_module/utils/Constants.java:27`.

`MQTTService` caches topics, resubscribes, and dispatches messages. Known outer
models include device-status fields (`action`, product key, device name, IoT ID,
timestamp) and property/event fields (`id`, `version`, `method`, `params`,
`time`). Constants identify status, property-post, log-progress, and protobuf
events. Evidence:
`maiot_module/MQTTService.java:272-384`;
`maiot_module/mqtt/MQTTClient.java:283-402`;
`maiot_module/mqtt/topic/TopicDeviceStatus.java`;
`maiot_module/mqtt/topic/TopicProperty.java`;
`maiot_module/utils/Constants.java:89-109`;
`maiot_module/utils/TopicUtils.java:21-43`.

Do not infer the full topic grammar from constants alone; runtime subscriptions
or server responses may supply topic strings.

## 4. Command manager and protobuf flow

### 4.1 Envelope anatomy

Feature helpers build a domain protobuf (`MctrlSys.MctlSys`,
`MctrlNav.MctlNav`, `MctrlDriver`, `DevNet`, `MctrlOta`, multimedia, or pool
control), then place it in `LubaMsgOuterClass.LubaMsg` with:

- `MsgCmdType`: subsystem/protocol family;
- `MsgDevice`: destination module;
- `MsgAttr`: request/response/report role;
- sequence/user/device metadata;
- the typed nested protobuf body.

Evidence:
`command/MACommandHelper.java:236-334`;
`command/app/MACommandApiHelper.java:278-337`;
`proto/LubaMsgOuterClass.java`.

For example, system commands use `MSG_CMD_TYPE_EMBED_SYS`,
`DEV_MAINCTL`, and `MSG_ATTR_REQ`; navigation commands use
`MSG_CMD_TYPE_NAV`. The built `LubaMsg` bytes then select BLE or IoT according
to command flags, device capability, and current link. Evidence:
`command/MACommandHelper.java:196-235,316-334`;
`command/app/MACommandApiHelper.java:304-337`.

`sendOrderMsg_Sys2` is a distinct framing/route variant used by some pool/local
branches; it must not be mechanically replaced by `sendOrderMsg_Sys`. Evidence:
`command/app/MACommandApiHelper.java:180-206,1612-1620`;
`device/source/device/enums/DeviceType.java:709-718,769-786`.

### 4.2 Public command façade and correlation

`CommandManager` provides lifecycle-aware, device-name-based operations. It
resolves the per-device command/data managers, sends a command, observes a
`PBResponseInfo` keyed by `PbMsgType.getResuestName(deviceName)`, removes
observers, and implements timeout-safe callback helpers. Evidence:
`command/CommandManager.java:64,318-385,462-544,658-909`.

`PbMsgType` is therefore a correlation vocabulary, not the full wire protocol.
Log-type integers passed to helpers are also not necessarily protobuf command
numbers; many are logging/subscription categories. Evidence:
`command/menus/PbMsgType.java`;
`command/CommandManager.java:462-544`;
`command/app/MACommandApiHelper.java:958,1347-1353`.

**Inference:** the practical request lifecycle is:

```text
caller selects device + operation
 -> register typed response observer/correlation key
 -> build nested protobuf and LubaMsg
 -> route over BLE or cloud service invoke
 -> parse an asynchronous report/ack
 -> publish PBResponseInfo
 -> callback fires or timeout removes observer
```

Do not assume FIFO request/response behavior; unsolicited reports and transport
switches coexist with command acknowledgements.

## 5. Report, reducer, and cache flow

There is no first-party class literally named `Reducer` in the inspected
command/state path. “Reducer” below is an architectural description.

### 5.1 Receive and decode

`MACarDataManager` and `MACarDataManagerAPI` are the central inbound decoders.
They parse `LubaMsg` and nested subsystem cases, convert selected messages into
app entities, invoke specialist listeners, update state-machine beans, update
map/report managers, and publish correlated `PBResponseInfo` values. Evidence:
`command/MACarDataManagerAPI.java:960-1317`;
`command/app/MACarDataManager.java:8424-8576`;
`command/app/contract/ReportDeviceListener.java`;
`command/app/contract/SwimmingPoolMapListener.java`.

### 5.2 Reduced live state

`ICarDevice` exposes two long-lived mutable state objects:

- `CarStateMachineBean`: connection and broad device status;
- `CarWorkingStateMachineBean`: work/navigation/report-derived state, including
  a `lastMapResult` response cache.

Evidence:
`device/source/device/interfaces/ICarDevice.java:57-95`;
`device/source/device/bean/CarStateMachineBean.java`;
`device/source/device/bean/CarWorkingStateMachineBean.java`;
`command/MACarDataManagerAPI.java:1314-1317`.

`DeviceStatueUploadMsgManager` supplies a single-request message store used by
`CommandManager` observers. Parsed acknowledgements are converted to
`PBResponseInfo` and inserted under a device-qualified request name. Evidence:
`device/source/device/manager/DeviceStatueUploadMsgManager.java`;
`command/MACarDataManagerAPI.java:960-1069,1099-1255`;
`command/CommandManager.java:318-385,824-825`.

**Inference:** report handling behaves like a reducer:

```text
raw transport event
 -> protobuf case dispatch
 -> normalized app entity or scalar
 -> mutate per-device state/cache
 -> emit listener, event, or LiveData value
 -> UI/ViewModel observes and renders
```

It is not a pure Redux reducer: parsing, mutation, persistence, callbacks, and
side effects are interleaved in large managers.

### 5.3 Cache classes

Keep these cache meanings distinct:

| Cache | Purpose |
|---|---|
| state-machine beans | Latest per-device live state reduced from reports. |
| `lastMapResult` / `singleRequest` | Last typed result and command-correlation delivery. |
| `BatchCacheLiveData` | Batches/cache-publishes command message values. |
| `AppCache` / `AppNewCache` | Process/global preferences and feature/debug switches. |
| LitePal DB rows | Durable account, device, map, plan, report, and diagnostic data. |
| MQTT topic cache | Subscription recovery, not device-state truth. |
| HTTP/WebView caches | Network/content optimization, not robot telemetry. |

Evidence:
`command/message_queue/BatchCacheLiveData.java`;
`base_module/utils/AppCache.java`;
`base_module/utils/AppNewCache.java`;
`maiot_module/mqtt/MQTTClient.java:319-402`;
`hybrid/agentweb/WebViewCache.java`;
`network/framework/interceptor/NetCacheInterceptor.java`.

Cloud work-history/report APIs are separate from live state reports. A late
history record must not be used to infer the current robot state. Evidence:
`base_module/bean/workreport/ReportByDeviceNameResBean.java:7-15`;
`base_module/bean/workreport/DeviceWorkReportPageResBean.java:7-28`;
`base_module/bean/workreport/ReportDetailResBean.java:7-33,68-70`.

## 6. Map hash, frame, common-data, and SVG flow

The robot is the operational source of map hashes and element payloads. The app
maintains a device-scoped LitePal geometry cache and renders it through Mapbox.
Cloud map snapshots and point-cloud generation are adjacent, separate services.
Evidence:
`command/app/HashDataManager.java:61-158,176-300`;
`map/MapDataImpl.java:25-153`;
`base_module/db/MapElementDB.java`;
`base_module/db/LineListDB.java`;
`base_module/db/TotalHashDB.java`.

### 6.1 Reconciliation loop

```text
connect / explicit refresh
 -> request NAV hash list, possibly in multiple frames
 -> acknowledge each hash-list frame
 -> compare aggregate MurMur hash + per-element hashes with local DB
 -> request missing/changed common data by type + hash
 -> receive framed geometry, validate/order/retry
 -> update MapElementDB / LineListDB and hash-order metadata
 -> MapDataImpl / AreaDBHelper read cache
 -> MapBoxManager refreshes sources and layers
```

Evidence:
`command/app/MACarDataManager.java:8424-8508,8535-8576`;
`command/app/HashDataManager.java:61-158,176-300`;
`command/MACommandHelper.java:758,1035,2030`;
`map/MapDataImpl.java:25-61,73-153`;
`map/mapbox/MapBoxManager.java`.

Hash-list messages carry `totalFrame` and `currentFrame`; the app acknowledges
frame receipt. Common-data responses carry action, type, data hash, parent
hashes, total/current frame, points, and optional naming/time data. Evidence:
`command/MACommandHelper.java:1035`;
`command/MACarDataManagerAPI.java:1099-1101`;
`command/entitys/CommData.java`.

`HashDataManager` serializes map/line loading, retries bad region/line loads up
to ten times, tracks progress/timeouts, and periodically requests a dynamic
route while working. Evidence:
`command/app/HashDataManager.java:99-205`.

Hashes are both identity and invalidation tokens:

- aggregate MurMur hashes indicate whether a collection changed;
- element hashes identify areas, channels, obstacles, routes, and SVG objects;
- pool `cmHash` / `pathHash` can signal map/line change;
- transaction IDs and saved order preserve topology and request context.

Evidence:
`command/app/HashDataManager.java:61-97,213-300`;
`device/source/device/entity/SwimmingPoolDeviceStatue.java:8-121`;
`proto/MctrlSys.java` (`SysWorkState`);
`proto/MctrlNav.java`.

### 6.2 SVG is a protocol map element, not the Mapbox renderer

The navigation protocol has a framed SVG message with transform and asset
metadata: `x_move`, `y_move`, `scale`, `rotate`, file name/data, name count,
base width/height, data count, data hash, parent hash, total/current frame, and
result. The app can send an SVG payload and a frame acknowledgement. Evidence:
`command/MACommandHelper.java:1546-1548,1590-1593`;
`command/entitys/SvgDataBean.java`;
`proto/MctrlNav.java` (`svg_message_t`, `svg_message_ack_t`).

**Inference:** SVG is one geometry/content exchange format within the robot map
protocol; Mapbox is the native display engine for normalized cached geometry.
Do not describe the whole map as “an SVG,” and do not discard SVG transforms,
parent hashes, or frame metadata after extracting paths.

### 6.3 Map implementation rules

- Key durable geometry by device plus type/hash, not display name alone.
- Preserve element order, parent hashes, visibility, timestamps, and transaction
  IDs.
- Assemble all frames and validate counts before replacing a good cached value.
- Treat a hash change as invalidation, not as geometry.
- Keep live route/path separate from static areas and channels.
- Retain raw pool frames until scale, coordinate system, ordering, and hash
  semantics are confirmed by capture.

These rules are partly **inference** from the app’s retry/reconciliation design,
not explicit protocol guarantees.

## 7. UI architectures: native, hybrid, and React Native

### 7.1 Native Android

Most device control is native Android: activities/fragments, data binding,
ViewModels/LiveData, helpers/presenters, ARouter navigation, Mapbox map views,
and Agora FPV. Representative roots:
`home/`, `map/`, `work/setting/`, `device/setting/`, `signal/`,
`bind/device/`, and `testing/`.

Native UI often calls `CommandManager`, `MACommandApiHelper`, or a feature
helper directly and observes mutable device state/listeners. This is not a
single MVVM implementation; legacy presenters, fragments, activities, and
newer ViewModels coexist. Evidence:
`home/utils/HomeCommandHelper.java:152-273`;
`home/viewmodel/HomeViewModel.java`;
`work/setting/JobPlanSettingPresenter.java`;
`map/activity/MapActivity.java:123-205,573-597`.

### 7.2 Hybrid web

`hybrid/` uses AgentWeb/WebView surfaces for academy, support, privacy,
agreements, update history, and Zoho support. A native JavaScript bridge exposes
navigation, language, device status/settings, and experimental functions.
Evidence:
`hybrid/agentweb/`;
`hybrid/bridge/`;
`hybrid/agentweb/ZohoWebActivity.java`;
`AndroidManifest.xml` hybrid activity registrations.

**Inference:** hybrid pages are remotely changeable presentation/support
surfaces. A bridge method proves that web content can request a native action;
it does not prove which current remote page invokes it.

### 7.3 React Native

The app is a `ReactApplication` and initializes an RN container module. It
declares portrait, landscape, and transform containers, checks
`POST rn/version/check`, and uses `RNHotfixWorkManager` for downloaded/hotfix
bundles. Evidence:
`MyApplication.java:124,296-297`;
`rn/RNContainerApplication.java:14-27`;
`rn/RNHotfixWorkManager.java`;
`rn/api/`;
`AndroidManifest.xml` RN container registrations.

Native RN modules expose message center/share acceptance, mowing reports, work
settings, battery management, pool plans, guides, localization/user
context, analytics, and logs. Evidence:
`rn/module/CommonModule.java`;
`rn/module/MessageCenterModule.java:262-466`;
`rn/module/MowingReportDataModule.java`;
`rn/module/WorkSettingModule.java`;
`rn/module/PoolWorkPlanModule.java`.

Downloaded JavaScript can contain routes and behavior absent from the decompiled
Java. Conversely, a native module proves an available bridge, not that a
particular JS bundle currently uses every method.

### 7.4 UI boundary rule

Native, web, and RN are presentation clients of overlapping native account,
device, map, and command services. They are not independent protocol stacks.
The strongest protocol evidence remains the native command/data/link managers
and generated protobufs.

## 8. Persistence and databases

### 8.1 LitePal

`MyApplication` initializes LitePal at startup, and most first-party DB entities
extend `LitePalSupport`. Evidence:
`MyApplication.java:260-263`;
`base_module/db/MapElementDB.java:11-14`;
`base_module/db/LineListDB.java:10-13`;
`base_module/db/JobPlanDB.java:11-14`.

Major durable groups include:

| Domain | Representative entities |
|---|---|
| Account/share/messages | `UserDB`, `DeviceMessageDB`, `SharedCreateDB`, `SharedNotifiyBean` |
| Device/link | `device/source/device/db/DeviceDB`, `DevicePropertiesDB`, `DeviceBleAddressDB`, `DeviceSWPDB`, `SpinoPiarInfoDB` |
| Map | `MapElementDB`, `LineListDB`, `ChannelDB`, `TotalHashDB`, `BackElementDB`, `SwpMapDB`, `ObstacleDB` |
| Work/plans/history | `JobPlanDB`, `TaskDB`, `JobDB`, `JobHistoryDB`, `OperationModeDB`, `UnableTimeDB`, `PlanJobSPBean` |
| Diagnostics | `FactoryTestDB`, `FactoryTestServiceDB`, `DebugErrorDB`, `ErrCodeCntIndexDB`, `UtilTestDB`, `BuryPointDB` |
| Media | `UserVideoListDB` |

Evidence: corresponding files under `base_module/db/`,
`device/source/device/db/`, `base_module/entity/PlanJobSPBean.java`, and
`base_module/trace/ma/BuryPointDB.java`.

No first-party Room database was established in the inspected app code.

### 8.2 Preferences and key-value state

`SharedPreferencesMgr` uses the `agilexrobotics` preference namespace; MMKV is
also initialized. `AppCache` and `AppNewCache` expose many persisted/global
feature and diagnostic switches. Evidence:
`MyApplication.java:292-294,583-596`;
`utils/SharedPreferencesMgr.java`;
`base_module/utils/AppCache.java`;
`base_module/utils/AppNewCache.java`.

Do not treat cached login state as proof that a token remains server-valid.
Evidence: `login/UserStateImpl.java:13-58`.

### 8.3 Persistence authority

**Inference:** persistence has mixed authority:

- robot reports/hashes are authoritative for live state and operational map;
- cloud APIs are authoritative for account ownership, sharing, remote config,
  and work-history records;
- LitePal is an app cache/local work store;
- preferences/MMKV are app settings and feature flags.

A durable row should therefore carry source, device/account key, update time,
and if relevant firmware/map hash. It should not silently override a newer
device report.

## 9. Background services, push, SSE, and work

The manifest registers the Mammotion Firebase messaging service,
`DeviceLogService`, MA-IoT `MQTTService`, Paho `MqttService`, WorkManager
initializers/services, Alibaba push components, and diagnostic/floating
services. Evidence:
`AndroidManifest.xml:223-226,645,948-966,1092-1272`.

`MyFirebaseMessagingService` receives push and participates in token
synchronization. Vendor push integrations exist for Alibaba and device-vendor
channels. Notification permission/channel behavior varies by Android version.
Evidence:
`mvp/fieldmower/service/firebase/MyFirebaseMessagingService.java`;
`MyApplication.java:626-631`;
`AndroidManifest.xml:1170-1259`.

Authenticated SSE is a second asynchronous cloud channel. The application hosts
an `SseClient`, initializes/terminates it with application lifecycle, and uses
reconnect/backoff and no-cache semantics. Evidence:
`MyApplication.java:135,628-636`;
`message/sse/ApplicationScopes.java`;
`message/sse/AppSseBootstrapKt.java`;
`message/sse/SseClient.java`.

WorkManager supports deferred RN hotfix/bundle work and other scheduled tasks.
It is not the robot command queue. Evidence:
`rn/RNHotfixWorkManager.java`;
`AndroidManifest.xml:948-963,1258-1272`.

Push, SSE, MQTT, and live device reports overlap but have different roles:

| Channel | Primary observed role |
|---|---|
| BLE | Local commands and immediate reports; provisioning via BLUFI. |
| HTTP service invoke | Authenticated remote command request. |
| MQTT | IoT online/property/protobuf/log events and asynchronous downlink. |
| SSE | Authenticated account/message event stream. |
| FCM/vendor push | OS-delivered notification/wakeup path. |
| REST | Account, device inventory, sharing, config, reports, tokens, OTA metadata. |

**Inference:** consumers should deduplicate cloud events by stable event/message
ID and timestamp where available. Arrival on push/SSE/MQTT does not by itself
make that channel authoritative for current mower state.

## 10. End-to-end data-flow recipes

### 10.1 Local command

1. UI/ViewModel resolves selected `ICarDevice` and validates model/live-state
   gates.
2. `CommandManager` optionally installs a device-qualified response observer.
3. `MACommandApiHelper` builds a subsystem protobuf and wraps it in `LubaMsg`.
4. The current per-device `MALinkManager` selects BLE.
5. Bytes become BLUFI custom-data frames and GATT writes.
6. A returned custom-data payload is decoded by `MACarDataManager`.
7. The manager updates state/listeners and/or publishes `PBResponseInfo`.
8. UI callback observes the response; durable map/plan data may also be updated.

Evidence:
`command/CommandManager.java:318-544`;
`command/app/MACommandApiHelper.java:278-337`;
`device/source/links/managers/MALinkManager.java:410-440`;
`device/source/links/managers/EspBleManager.java:1090-1107`;
`command/MACarDataManagerAPI.java:960-1317`.

### 10.2 Remote command

Steps 1–3 are shared. The helper/link manager then serializes the protobuf for
MA-IoT service invocation. HTTP acceptance is transport-level; asynchronous
ack/state may arrive through MQTT and is reduced by the same per-device data
manager. Evidence:
`command/MACommandHelper.java:196-235`;
`device/source/links/managers/MAIotManager.java:633-743`;
`maiot_module/MQTTService.java:294-384`.

**Inference:** never optimistically finalize robot state from service-invoke
success alone. Wait for a protocol ack or a compatible state report.

### 10.3 Map refresh

Connection triggers initial sync; hash lists are assembled; local hashes are
compared; missing/changed common data is fetched and persisted; Mapbox redraws
from the cache. A working robot also supplies position and dynamic route
updates. Evidence:
`command/app/MACarDataManager.java:8424-8508`;
`command/app/HashDataManager.java:99-300`;
`map/MapDataImpl.java:73-153`.

### 10.4 Cloud message to UI

MQTT/SSE/push receives an outer JSON event, associates device/account identity,
persists or posts an app event when applicable, and native or RN message-center
surfaces render it. Some device errors are persisted in `DeviceMessageDB` and
converted to app events. Evidence:
`maiot_module/MQTTService.java:294-384`;
`message/sse/`;
`base_module/db/DeviceMessageDB.java`;
`rn/module/MessageCenterModule.java:262-466`.

## 11. Glossary and acronyms

| Term | Meaning in this APK |
|---|---|
| Ack / `WAIT_ACK` / `INQUIRY` | Protocol acknowledgement state or requested operation mode. Exact enum semantics are message-specific. |
| Ali IoT | Legacy Alibaba IoT device-panel path under `iot_module`; `ICarDevice.linkType == "0"`. |
| ARouter | In-app native route/navigation framework initialized at startup. |
| BLE | Bluetooth Low Energy, used for discovery, local control, and BLUFI transport. |
| BLUFI | Espressif BLE framing/provisioning protocol; carries Wi-Fi setup and custom runtime bytes. |
| Common data | Navigation payload fetched by element type and hash, often framed; used for map geometry/metadata. |
| `DeviceType` | Coarse app enum for product families/SKUs, not a complete capability contract. |
| DND | Do not disturb/non-work interval. |
| FCM | Firebase Cloud Messaging. |
| FPV | First-person live camera view, implemented with Agora rather than MQTT/BLE video. |
| GATT | BLE Generic Attribute Profile used for characteristic reads/writes. |
| Hash | Map/content identity and invalidation value; not geometry itself. |
| H5 | Hosted HTML/web content shown in hybrid WebViews. |
| IoT | Generic cloud device route; in this APK may mean legacy Ali IoT or current MA-IoT. |
| `iotId` | Cloud routing identity for an IoT device. |
| `ICarDevice` | Common in-memory abstraction for mower, pool robot, RTK, or related device. |
| `LubaMsg` | Top-level protobuf envelope carrying command family, device/module, attr, metadata, and typed body. |
| LitePal | ORM/local SQL persistence used by first-party DB entities. |
| LoRa | Long-range radio link/pairing used by RTK or pool robot/dock relationships. |
| MA-IoT | Current Mammotion IoT stack under `maiot_module`; link type `"1"`. |
| `MACarDataManager` | Per-device receive/parser/state/listener hub. |
| `MACommandApiHelper` | Low-level typed command builder/sender. |
| `MADeviceManager` | Registry/factory for per-device data/link managers. |
| `MALinkManager` | Per-device BLE-versus-IoT transport selector. |
| MQTT | Broker protocol used for asynchronous cloud status/property/protobuf events. |
| MMKV | Key-value persistence initialized alongside preferences and LitePal. |
| MTU | BLE maximum transmission unit; target 200 in the observed manager. |
| NRTK / NetRTK | Network-delivered RTK correction mode. |
| `PBResponseInfo` | App-level correlated command result wrapper, not a wire envelope. |
| `PbMsgType` | App correlation/request vocabulary for selected parsed responses. |
| Protobuf / PB | Protocol Buffers; generated classes under `proto/`. |
| `pver` | Protocol-version field inside many protobuf messages. |
| Reducer | In this document, the report-to-state mutation role performed by data managers; not a literal class name. |
| RN | React Native. |
| RTK | Real-time kinematic positioning/base-station functionality. |
| SSE | Server-Sent Events account/message stream. |
| SP / SPINO | Pool-cleaner family vocabulary; `PC210` is the SP model branch in this APK. |
| SVG | Framed navigation map/art payload with transforms and hashes; distinct from Mapbox rendering. |
| VIO | Visual-inertial odometry/status. |
| WorkManager | Android deferred/background work framework; used for app work such as RN hotfix handling. |

## 12. Source-path map

| Concern | Primary paths |
|---|---|
| Process startup and DI | `MyApplication.java`; `Hilt_MyApplication.java`; `base_module/base/{BaseApplication,BaseApplicationInit,ApplicationHelper}.java` |
| Feature module hooks | `rn/RNContainerApplication.java`; `login/LoginApplication.java`; `map/MapApplication.java`; `bind/device/BindDeviceApplication.java`; `home/HomeApplication.java`; other `*Application.java` files registered at `MyApplication.java:323-337` |
| Device abstraction/inventory | `device/source/device/interfaces/{IDevice,ICarDevice,IDeviceManager}.java`; `device/source/device/manager/DeviceManager.java`; `device/source/links/managers/MADeviceManager.java` |
| Model and capability gates | `device/source/device/enums/DeviceType.java`; `extensions/DeviceTypeExtensionsKt.java`; `helpers/DeviceMultimodelHelper.java`; `utils/{DeviceUtils,DeviceVersionUtils,DeviceProductKey}.java`; `assets/device_configuration_form*.json` |
| Link coordination | `device/source/links/MALinkManagerAPI.java`; `device/source/links/managers/{MALinkManager,EspBleManager,MAIotManager,MAScanManager,SubscriptionManager}.java` |
| BLUFI/GATT | `espressif/{BlufiClientImpl,FrameCtrlData}.java`; `espressif/security/BlufiClient.java`; `espressif/params/BlufiParameter.java`; `espressif/blue_utils/` |
| Current cloud/MQTT | `maiot_module/{MaIoTApp,MQTTService}.java`; `maiot_module/mqtt/MQTTClient.java`; `maiot_module/api/`; `maiot_module/bean/`; `maiot_module/utils/{Constants,TopicUtils}.java` |
| Legacy cloud | `iot_module/third/api/device/{DispatchNetAPI,DevicePanelApi}.java`; related `iot_module/` login/event classes |
| Command façades | `command/CommandManager.java`; `command/MapCommandManager.java`; `command/{MACommandHelper,MACarDataManagerAPI}.java`; `command/app/{MACommandApiHelper,MACarDataManager}.java` |
| Protocol schemas | `proto/{LubaMsgOuterClass,MctrlSys,MctrlNav,MctrlDriver,MctrlOta,DevNetOuterClass,SpinoCtrlOuterClass}.java` |
| Parsed state/correlation | `device/source/device/bean/{CarStateMachineBean,CarWorkingStateMachineBean,PBResponseInfo}.java`; `device/source/device/manager/DeviceStatueUploadMsgManager.java`; `command/message_queue/`; `command/menus/{PbMsgType,MsgCmdType}.java` |
| Map synchronization | `command/app/HashDataManager.java`; `map/{MapDataImpl,MapBoxManager}.java`; `map/utils/MapDataParser.java`; `base_module/db/{MapElementDB,LineListDB,TotalHashDB,ChannelDB,BackElementDB}.java` |
| Map/SVG entities | `command/entitys/{CommData,SvgDataBean}.java`; SVG/common-data messages in `proto/MctrlNav.java` |
| Native home/work UI | `home/`; `work/setting/`; `device/setting/`; `signal/`; `bind/device/`; `device/deploy/` |
| Hybrid web | `hybrid/agentweb/`; `hybrid/bridge/`; `hybrid/activity/` |
| React Native | `rn/`; especially `rn/module/`, `rn/api/`, `RNHotfixWorkManager.java`, and RN container activities |
| Persistence | `base_module/db/`; `device/source/device/db/`; `base_module/helper/CommonDBHelper.java`; `base_module/utils/{SharedPreferencesMgr,AppCache,AppNewCache}.java` |
| Reports/history | `base_module/bean/workreport/`; `work/setting/activity/JobHistoryActivity.java`; `work/setting/fragment/JobHistoryDetailFragment.java`; report transfer messages in `proto/MctrlSys.java` |
| Push/SSE/background | `mvp/fieldmower/service/firebase/MyFirebaseMessagingService.java`; `message/sse/`; `maiot_module/MQTTService.java`; `feedback/DeviceLogService.java`; `rn/RNHotfixWorkManager.java`; `AndroidManifest.xml` |
| Camera/vision | `map/viewmodel/MapDeviceModel.java`; `base_module/bean/resp/{VideoResp,VideoTokenResp}.java`; `base_module/event/VioToAppInfoEvent.java`; Agora-related map activities |
| Diagnostics/testing | `testing/`; `feedback/`; `log/`; `trace/`; `base_module/entity/testtool/`; `assets/device_test_tools.json` |

## 13. High-value cautions for protocol consumers

1. Keep one logical device with multiple transports; do not expose BLE and IoT
   as duplicate robots.
2. Preserve the exact device/product/IoT/BLE identifiers and their roles.
3. Evaluate capabilities dynamically; `DeviceType` is only the first gate.
4. Preserve unknown protobuf fields/enums where the library permits and always
   preserve unknown numeric values in normalized models.
5. Separate service-invoke acceptance, protocol ack, and observed state change.
6. Treat reports as asynchronous and potentially duplicated or reordered across
   BLE/MQTT reconnects.
7. Key state, response caches, and geometry by device.
8. Reassemble and validate framed hash/common-data/SVG/report/OTA transfers
   before committing them.
9. Keep live state, local cache, and cloud history as separate data sources.
10. Do not infer support from a generated protobuf, native bridge, hidden test
    menu, manifest component, or resource string alone.

Items 1, 4, 6, 7, and 8 are **inference-backed implementation guidance** based
on the observed architecture; they are not vendor guarantees.

## 14. Relationship to reports 01–09

This reference consolidates:

- onboarding, BLE/BLUFI, IoT/MQTT, and link selection from
  `docs/apk-feature-catalog/01-onboarding-connectivity.md`;
- map hashes, frames, persistence, deployment, and Mapbox from
  `02-mapping-deployment.md`;
- plans, execution, acknowledgements, and report/history separation from
  `03-work-planning-execution.md`;
- manual-control and safety gating from `04-manual-control-safety.md`;
- settings, model/firmware gates, and maintenance from
  `05-device-settings-maintenance.md`;
- Agora/FPV and vision boundaries from `06-camera-video-vision.md`;
- account, sharing, REST, push, SSE, hybrid, and RN from
  `07-account-sharing-cloud.md`;
- pool-specific framing, plans, maps, LoRa, and OTA from
  `08-pool-cleaner.md`;
- hidden/test tooling and app caches from `09-diagnostics-testing.md`.

Where those reports describe a feature surface and this reference describes a
cross-cutting flow, the cited decompiled source remains the evidence of record.
