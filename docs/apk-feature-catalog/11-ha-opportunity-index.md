# Home Assistant opportunity and risk index

## Purpose and limits

This is a future-backlog index derived from the Mammotion Android app
`2.3.8.19` subsystem reports `00` through `10`, the model/capability matrix
`13`, and selected decompile call-site checks. It is not a roadmap,
compatibility promise, or statement that a protocol-visible feature is safe to
expose. A shipped app screen can still be restricted by firmware, server
capability, region, account role, entitlement, current device state, or
transport.

The baseline for “already represented” is the integration source at the time of
this review. Some entities and services are diagnostic or experimental, and
their presence does not imply equal support on every model. Priorities mean:

- **P0 — preserve/harden:** existing high-value behavior or safety boundary.
- **P1 — investigate next:** strong evidence, useful in HA, and a favorable
  read-only or tightly bounded risk profile.
- **P2 — later/opt-in:** useful, but requires more protocol, capability, UX, or
  state-machine work.
- **P3 — research only:** weakly evidenced, credential-heavy, destructive, or
  unsuitable for routine automation.
- **Do not expose:** retain only as a threat-model, diagnostic vocabulary, or
  explicit non-goal unless a separate safety design changes the conclusion.

Confidence describes the app evidence, not implementation readiness:
**High** means a user-facing caller and command/report/API path agree;
**Medium** means one side, enum mapping, or reachability remains incomplete;
**Low** means only a latent protocol, resource, hidden tool, or damaged
decompile path is available.

## Cross-cutting acceptance rules

1. Gate from a layered identity: raw product key/name, parsed type, code/service
   item, subtype, `pileIsPile`, reported capabilities, firmware, ownership, and
   transport. Marketing names, service item, and enum order alone are unsafe;
   notably, `Spino-S1` spans protocol generations, service item `1313` is
   ambiguous, broad pool includes `SD_PX`, and the app's ordinal `has4G()` test
   includes unrelated later types. See the
   [identity hazards](13-model-capability-matrix.md#identity-hazards-that-matter-to-ha).
2. Retain raw enum values and source timestamps. Cloud fallback, MQTT, BLE, and
   app-local caches do not have equal freshness.
3. Treat a successful send as transport success only. Cloud HTTP success can
   precede the device protobuf response, and push reports are not command acks.
   Where state matters, wait for the keyed one-shot/multipart response or an
   authoritative report and expose timeout/failure. See the
   [routing and acknowledgement model](10-protocol-report-index.md#routing-and-acknowledgement-model).
4. Use stable IDs: area hash, plan ID, work ID, IoT/device ID, and pile ID.
   Names are mutable labels.
5. For compound settings, query first, preserve unknown fields, write the full
   object, then read back. This is essential for work plans, battery policy,
   schedules, and pool plans.
6. Keep credentials and sensitive identifiers out of logs and diagnostics,
   including access/refresh tokens, Agora token/secret/channel/UID, Wi-Fi
   passwords, email, ICCID, MAC, pairing codes, and precise location unless
   explicitly requested.
7. Any movement lease must fail closed on stale telemetry, disconnect, lock,
   self-check failure, charging-state conflict, or writer contention.

## Already represented in Mammotion-HA

These are baseline capabilities to preserve and harden, not necessarily
feature-complete matches to the app.

| App feature / candidate | Evidence | Current or likely HA shape | Transport / data source | Model gates | Safety / security concerns | Confidence | Priority |
|---|---|---|---|---|---|---|---|
| Core mower lifecycle: start, pause, resume, stop/cancel, return to dock | [Work execution](03-work-planning-execution.md#task-execution-and-state-flow) | `lawn_mower` actions plus start/cancel services | Device protobuf over BLE or cloud RPC; async state reports | Mower family, work state, DND, RTK/self-check readiness | Resume is not start; do not conflate task stop, map stop, blade stop, and e-stop | High | P0 — preserve guards and result reporting |
| Battery, charging, work state, progress, area, timing, position, RTK and signal telemetry | [Settings/status](05-device-settings-maintenance.md#charging-battery-and-power-management), [mapping](02-mapping-deployment.md#home-assistant-implementation-relevance) | Sensors, charging binary sensor, device tracker | MQTT/device reports, cloud state, map cache | Per report field and device family | Surface source age; redact precise coordinates from diagnostics when appropriate | High | P0 — improve freshness/source attributes |
| Work parameters: areas, speed, spacing, height, pattern/angle, laps, border order, work-content flags | [Work settings](03-work-planning-execution.md#feature-catalog-work-content-patterns-and-numeric-settings) | Area switches; number/select/switch entities; start-task services | Cached map/task data plus protobuf plan/task writes | Server-downloaded capability codes and model-derived ranges | Preserve ordered hashes and unknown fields; validate accessibility and dynamic bounds | High | P0 — capability-driven availability |
| Task/schedule discovery and CRUD | [Schedules](03-work-planning-execution.md#schedules-and-calendars) | Task sensors/buttons and create/edit/rename/enable/delete/copy/start/refresh services | Device plan protobuf, cache, and integration task store | Mower versus PC210 plan protocol; firmware capability | Query-before-write; preserve IDs, recurrence, enable polarity, timezone, and ordered areas | High | P0 — readback and round-trip tests |
| Map synchronization, areas, route/path export and GeoJSON | [Map lifecycle](02-mapping-deployment.md#architecture-and-map-lifecycle) | Map-sync button/status, area entities, device tracker, map/route/export services | Device hash/common-data/SVG reports and local decoded cache | Legacy versus MN231/SVG map stack; pool map is separate | Robot is authoritative; avoid optimistic geometry mutation | High | P0 — sync integrity and unknown-element retention |
| Connectivity visibility and user-selectable Bluetooth/cloud/Wi-Fi/4G controls | [Connectivity](01-onboarding-connectivity.md#feature-catalog), [network settings](05-device-settings-maintenance.md#network-wifi-4g-rtk-and-positioning) | Transport/MQTT/RSSI sensors and Bluetooth, cloud, device Wi-Fi/4G switches | Integration link manager, BLE, cloud/MQTT, device network reports | Local-add pool differs; shared devices may be cloud-only | A network write can strand the device; expose last-good transport and rollback guidance | High | P0 — harden failover semantics |
| Audio, voice, lamps, rain behavior, wildlife/obstacle behavior, cutter mode, maintenance counters | [Device settings](05-device-settings-maintenance.md#audio-voice-lighting-camera-and-wiper), [vision](06-camera-video-vision.md#4-vision-vio-obstacles-and-ai) | Existing switches/selects/numbers/sensors and blade maintenance services | Device system/settings protobuf and reports | Strong model/capability gates; enums vary | Unknown enum values must remain visible; safety settings should not be silently changed by automations | High | P0 — verify enum/model matrices |
| Camera live view, FPV status, stream refresh, and wiper | [Camera transport](06-camera-video-vision.md#1-live-view-and-stream-lifecycle) | Camera entity, FPV/brightness/VIO sensors, stream services, wiper button | Mammotion token API plus Agora media; wiper via device command | Vision/video models, firmware, network, quota/entitlement, camera generation | Redact all media credentials; movement safety must not depend on video continuity | High | P0 — credential hygiene and lifecycle tests |
| Guarded manual movement and experimental path/SVG tooling | [Manual control](04-manual-control-safety.md#8-home-assistant-implementation-relevance), [map writes](02-mapping-deployment.md#home-assistant-implementation-relevance) | Direction services, diagnostic probes, dry-run/preview/path and SVG services | Primarily direct device commands; map/SVG protobuf | Manual-control and map-generation family | Dead-man lease, final zero, fresh telemetry, one writer; experimental services must remain clearly segregated | High | P0 — treat as hazardous experimental surface |
| SPINO basics: battery/status/mode/error, start/stop controls, buzzer, cleaning options, map/line fetch, floor/environment settings | [Pool priorities](08-pool-cleaner.md#ha-implementation-priorities) | Vacuum-like control plus pool sensors, switches, selects, number and fetch buttons | Legacy pool local BLE or PC210 device/cloud reports | Resolve product key; exclude `SD_PX` pile from robot entities | Mode `4` label differs by generation; retain raw mode and source | High | P0 — strengthen protocol-family gates |
| Firmware version and update workflow | [Firmware](05-device-settings-maintenance.md#firmware-versions-and-update-workflow) | Update entity and component/version attributes | Cloud version/check and upgrade APIs; device OTA reports | Auto-update, pool OTA, module and firmware gates | Enforce charging/battery/Wi-Fi/idle preconditions; local pool transfer is a distinct high-risk path | High | P0 — keep update initiation conservative |

## Low-risk read-only entities

These are the strongest backlog opportunities. “Low-risk” means no device
mutation; privacy, data volume, and stale-state risks still apply.

| App feature / candidate | Evidence | Likely HA shape | Transport / data source | Model gates | Safety / security concerns | Confidence | Priority |
|---|---|---|---|---|---|---|---|
| Map accessibility and preflight result | [Accessibility/preflight](02-mapping-deployment.md#home-assistant-implementation-relevance) | Diagnostic sensor and validation response listing inaccessible area index/reason | Decoded area/channel graph plus current `posType`/hash and fresh self-check state | Map stack and localization model | Default unknown, not accessible, when graph or position state is stale | High | P1 |
| Map/hash synchronization health | [Map sync](02-mapping-deployment.md#home-assistant-implementation-relevance) | Sensors for total/region/route hashes, missing hashes, retries, route generation and last reconciliation | Device hash/common-data/SVG reports and integration cache | Legacy hash versus MN231/SVG | Hashes are opaque IDs, not secrets; avoid unbounded diagnostic payloads | High | P1 |
| Layered device identity and capability eligibility | [Model identity](13-model-capability-matrix.md#canonical-device-identities), [gating recommendations](13-model-capability-matrix.md#ha-capability-gating-recommendations) | Diagnostic attributes and an internal yes/no/conditional/unknown capability registry | Cloud device inventory, raw transport identity, firmware, subtype and runtime state | Every model/accessory; especially YUKA mini variants, RTK3, S1/SP and `SD_PX` | Do not expose product keys as credentials, collapse unknown to false, or infer modem/pile/camera support from enum order or label | High / conditional | P1 |
| Device network details | [Network info](05-device-settings-maintenance.md#network-wifi-4g-rtk-and-positioning) | Diagnostic sensors/attributes for SSID, RSSI, IP, mask, gateway and active uplink | Device network-info query/report | Wi-Fi/4G hardware | MAC, IP, SSID and ICCID are sensitive; redact by default | High | P1 |
| Battery policy and health readback | [Battery settings](05-device-settings-maintenance.md#charging-battery-and-power-management) | Battery health/capacity, charge limit, smart/off-peak status and configured window sensors | `queryBatteryInfo` response | Supporting battery firmware/models | Do not infer health from percent; preserve raw response until schema is stable | High for policy fields; Medium for health | P1 |
| Self-check mask, lock bits, e-stop state and command-block reason | [Safety state](04-manual-control-safety.md#4-stop-lock-emergency-stop-and-self-check) | Diagnostic sensors and binary sensors; standardized “why unavailable” attributes | Device state reports | Bit meanings and predicates vary by generation | Never offer remote e-stop clear from this evidence; stale state must block hazardous controls | High for raw state; Medium for labels | P1 |
| Real-time blade RPM and richer maintenance telemetry | [Diagnostics options](09-diagnostics-testing.md#app-side-diagnostics-fault-injection-and-feature-switches), [maintenance](05-device-settings-maintenance.md#sensors-diagnostics-status-indicators-and-error-history) | RPM, blade hours/warning, distance, battery-cycle and component-health sensors | Normal device reports where available; hidden display flag is only discovery evidence | Cutter hardware and reporting capability | Read only; do not activate engineering mode merely to obtain it | Medium | P1 after normal report trace |
| Bumper and per-direction ultrasonic/radar state | [Engineering catalog](09-diagnostics-testing.md#device-test-tool-catalog), [existing vision diagnostics](06-camera-video-vision.md#4-vision-vio-obstacles-and-ai) | Diagnostic binary sensors per direction plus radar readiness | Normal telemetry if present; test menu supplies vocabulary only | Ultrasonic/radar/vision hardware | Never send test/fault-injection writes; rate-limit state churn | Medium | P1 after normal report trace |
| VIO/vision readiness detail | [VIO status](06-camera-video-vision.md#vio-and-vision-status) | VIO state, brightness, heading/features/survival-distance and RTC-ready sensors | Device state/VIO reports; app event bus | Vision-capable models | Avoid presenting diagnostic estimates as safety guarantees | Medium-High | P1 |
| Work history, result metrics and event timeline | [Job reports](03-work-planning-execution.md#job-history-reports-and-diagnostics) | On-demand service and bounded event/statistics sensors for last work, area, duration, energy, result and event codes | Cloud report REST or validated device report transfer | Account/region; pool report fields differ | Location path and work history are private; do not create high-cardinality permanent entities | High | P1 for summaries; P2 for detail/path |
| Component firmware versions and update prerequisites | [Firmware details](05-device-settings-maintenance.md#firmware-versions-and-update-workflow) | Update attributes and diagnostic sensors per controller, RTK, LoRa, motor, hub and pool module | Cloud version check plus device OTA info reports | Hardware-specific component list | Avoid exposing package URLs or checksums if credentials are embedded | High | P1 |
| Ownership, shared-user status and effective privilege | [Sharing model](07-account-sharing-cloud.md#3-device-sharing-ownership-roles-and-permissions) | Diagnostic attributes such as owner/shared/backend generation; entity availability policy input | Authenticated cloud device/share metadata | MA-IoT versus legacy Ali IoT; account region | Redact owner email/account; shared access never implies owner privileges | High | P1 |
| Cloud event stream and recent device notifications | [Notifications/SSE](07-account-sharing-cloud.md#4-notifications-messages-push-and-sse) | Integration events and a bounded “last notification/error” sensor | Authenticated SSE with reconnect/backoff; message REST | Region/account; SSE URL is dynamic | Sanitize message content; no FCM token reuse; cap retention/cardinality | High for SSE transport | P2 |
| Pool wheel, pump, module, map/path hash and pile-link status | [Pool reports](08-pool-cleaner.md#home-status-and-reports), [pile data](08-pool-cleaner.md#chargingdocking-pile-and-lora-pairing) | Diagnostic sensors; separate pile device with LoRa/channel/link attributes | Pool system/downlink/LoRa reports and local pair metadata | PC210 versus legacy; pile is `SD_PX`, not a robot | State vocabulary is incomplete; unknown enum values remain raw | High for fields; Medium for labels | P1 |
| SIM/4G status, quota and service state | [Connectivity](01-onboarding-connectivity.md#feature-catalog), [network settings](05-device-settings-maintenance.md#network-wifi-4g-rtk-and-positioning) | Activation/traffic/quota/service diagnostic sensors | Cloud SIM detail/limit APIs plus device network reports | Cellular-equipped SKU and region/service | ICCID is sensitive; distinguish status from entitlement and billing | High | P2 |

## Bounded and reversible writes

These are candidates only when a readback exists, inputs are capability-derived,
and rollback or a safe neutral state is understood.

| App feature / candidate | Evidence | Likely HA shape | Transport / data source | Model gates | Safety / security concerns | Confidence | Priority |
|---|---|---|---|---|---|---|---|
| Smart charging, charge limit and off-peak window | [Battery policy](05-device-settings-maintenance.md#charging-battery-and-power-management), [protocol index](10-protocol-report-index.md#schedules-work-plans-sleep-and-battery) | Switch, percentage number, time entities and “charge to 100% once” button | Compound device battery query/set protobuf | Supporting battery firmware/model | Query first and preserve every field; invalid windows or partial writes can reduce availability | High | P1 |
| Non-work/DND interval | [Schedules](03-work-planning-execution.md#schedules-and-calendars), [wire shape](10-protocol-report-index.md#schedules-work-plans-sleep-and-battery) | Enable plus start/end time entities or create/edit/delete service | Device `NavUnableTimeSet` command and typed readback | Models exposing DND | Device timezone/DST, trigger `99` sunrise semantics and wrap-midnight behavior need tests; start must report DND block | High | P1 |
| Audio/voice values and normal lighting | [Audio and lighting](05-device-settings-maintenance.md#audio-voice-lighting-camera-and-wiper) | Switch/number/select/light entities | Device settings protobuf with reported state | Capability/model-specific language, lamps and channels | Dynamically enumerate languages; retain unexplained mode/time fields; avoid rapid toggling | High | P1 where not already represented |
| Obstacle mode, wildlife/animal protection and rain tactics | [Vision settings](06-camera-video-vision.md#obstacle-avoidance-and-ai-modes), [safety settings](05-device-settings-maintenance.md#weather-safety-security-locks-and-animal-protection) | Select/switch with raw-value attribute | Work/settings protobuf and state report | Server option list, vision hardware, X5 and firmware rules | These alter collision/weather behavior; require explicit user action and reject unknown mappings | High for feature; Medium for enum matrix | P1 hardening / P2 expansion |
| Recharge-and-continue policy | [Operational commands](03-work-planning-execution.md#commands-and-protocol-fields) | Switch/select with readback | Device get/set recharge-and-continue command | Capability and task type | Changing policy during work can alter unattended movement; apply only in safe state | Medium-High | P2 |
| Area rename | [Map relevance](02-mapping-deployment.md#home-assistant-implementation-relevance), [area naming](03-work-planning-execution.md#feature-catalog-planning-and-area-selection) | Rename service keyed by area hash | Device map/name message and refreshed map cache | Map generation/protocol | Name is not identity; wait for device readback and handle collisions/length limits | High | P1 |
| Request map, schedule, network, firmware or pool-state refresh | [Map](02-mapping-deployment.md#home-assistant-implementation-relevance), [pool](08-pool-cleaner.md#home-status-and-reports) | Buttons/services | Read/query commands or cloud GET/POST | Per subsystem | Debounce to avoid device/cloud flooding; a refresh is not a repair | High | P1 |
| PC210 environment and waterline docking schedule | [Pool environment](08-pool-cleaner.md#pool-environment-speed-and-waterline-docking) | Wall/bottom selects, floor-speed number, docking mode select and time entity | `app_downlink_cmd_t` query/set with ack/report | PC210/SP only; not `SD_PX` or legacy by assumption | Query before write; preserve firmware limits; timed docking timezone semantics need validation | High | P1 |
| PC210 mode/sub-mode, speed and suction plan settings | [Pool controls/plans](08-pool-cleaner.md#cleaning-modes-and-controls) | Mode select and advanced start/plan service | SP work-mode and `SpinoCtrl` plan protobuf | PC210/SP only | Keep mode `4` model-aware; preserve plan IDs, order, enable polarity and unknown fields | High | P2 |
| Camera position selection | [Camera controls](06-camera-video-vision.md#2-camera-controls-wiper-and-manual-fpv) | Select tied to stream restart | Token request/Agora camera UID or legacy device join command | Multi-camera vision models and stream generation | Verify `1`/`3` and UID mapping per model; restart must not affect movement state | Medium-High | P2 |
| Automatic firmware update preference | [Firmware](05-device-settings-maintenance.md#firmware-versions-and-update-workflow) | Switch with prerequisite/status attributes | Server-backed device setting | `isSupportAutoUpgrade`, region and firmware | Read back server state; communicate idle/docked/Wi-Fi behavior | High | P2 |
| Network RTK link mode | [Positioning](05-device-settings-maintenance.md#network-wifi-4g-rtk-and-positioning) | Select after enum discovery | Device RTK-link command plus cloud service status | RTK/iNavi capability and minimum firmware | Bad selection can remove positioning; require current-mode readback and rollback path | High for command; Medium for enum | P2 |

## High-risk movement, blade, security, and account operations

These should not become ordinary automatable entities. A future implementation
would require a separate safety/security design, explicit opt-in, strict
preconditions, auditability, and in many cases local presence.

| App feature / candidate | Evidence | Likely HA shape, if ever | Transport / data source | Model gates | Safety / security concerns | Confidence | Priority |
|---|---|---|---|---|---|---|---|
| Continuous manual drive / remote positioning | [Manual motion](04-manual-control-safety.md#1-direct-manual-motion) | Operator-only leased service, never a persistent direction switch | Raw driver command, primarily local/current link, repeated about 5 Hz | Manual-control capable device and fresh state | Dead-man lease, one writer, final repeated zero, stale-state deny, no unattended automation | High | P3 — keep experimental |
| Manual mowing and direct blade start | [Manual blade safety](04-manual-control-safety.md#2-manual-lawn-mowing-and-blade-control) | Separate enter/start/stop/exit actions with local confirmation | Caller-enforced Bluetooth device command and state feedback | Blade-capable mower; model speed/height limits | Reproduce consent, RSSI threshold, 8 s initial-motion and 3 s stationary stops; fail closed | High | Do not expose as a normal switch |
| Direct cutter/side-blade direction or RPM writes | [Engineering actuator tools](09-diagnostics-testing.md#device-test-tool-catalog) | No routine HA shape; read-only RPM is separate | Hidden system/test commands | Specific cutter hardware | Critical injury/equipment risk; engineering paths may bypass consumer guards | High existence; Medium semantics | Do not expose |
| Map boundary, channel, no-go/safety element add/update/delete and delete-all | [Map writes](02-mapping-deployment.md#generic-element-model) | Advanced transaction service only after live validation | Navigation/manual-element/SVG protobuf | Legacy versus MN231 element schemas | Can create escape paths or erase safety geometry; require hash precondition, preview, backup and readback | High | P3 |
| Dock relocation/reset, manual docking and one-touch leave | [Deployment](02-mapping-deployment.md#home-assistant-implementation-relevance) | Explicit guarded buttons/services only | Navigation/dock commands plus motion and charge reports | Docked/positioned state and model | Robot moves; reset can invalidate navigation; require local observation and fresh charge/position state | High | P2 for leave; P3 for relocation/manual docking |
| RTK pairing code, base reset, iNavi handoff and calibration writes | [RTK/deployment](02-mapping-deployment.md#model-and-firmware-gates), [network settings](05-device-settings-maintenance.md#network-wifi-4g-rtk-and-positioning) | Setup flow, not routine entities | Device command plus cloud pairing/handoff APIs | RTK/iNavi/base family and firmware | Can strand positioning or invalidate maps; pairing codes are sensitive | High | P3 |
| Wi-Fi provisioning/forget/radio disable | [Provisioning](01-onboarding-connectivity.md#espressif-blufi-over-ble), [network settings](05-device-settings-maintenance.md#network-wifi-4g-rtk-and-positioning) | Config flow with confirmation and rollback guidance | BLE BLUFI for credentials; device network commands | BLUFI/local reachability and product generation | Password secret handling; disabling/forgetting can strand remote access | High | P2 provisioning; P3 remote disable |
| Robot restart, factory reset, unbind/remove | [Administration](05-device-settings-maintenance.md#top-level-settings-and-device-administration) | Restart may remain guarded button; reset/unbind setup-only | Device system command plus cloud binding API | Ownership and model | Work/slope warning for restart; reset/unbind are destructive and may erase access/configuration | High | P2 restart; do not expose reset/unbind routinely |
| Share creation/cancellation, invitation acceptance and ownership actions | [Sharing](07-account-sharing-cloud.md#3-device-sharing-ownership-roles-and-permissions) | Dedicated setup flow only | Authenticated user/device cloud APIs | Owner versus authorized user, backend generation, region | Changes authorization; QR/account data is sensitive; requires exact privilege checks and user confirmation | High | P3 |
| Account password/email/identity changes, logout and deletion | [Account lifecycle](07-account-sharing-cloud.md#2-account-profile-security-preferences-and-lifecycle) | Reauthentication flow only; no HA service for deletion | Authenticated account APIs and email verification | Account/provider/region | Account takeover or irreversible loss; never automate or expose account deletion | High | Do not expose |
| Engineering simulation, safety bypass, fault injection, calibration and aging tests | [Diagnostics catalog](09-diagnostics-testing.md#device-test-tool-catalog) | None; vocabulary may inform decoders/tests | Hidden debug/test commands and app preferences | Developer permission, hidden menus, hardware | Includes collision/rain bypass, fake RTK/battery/charge, prolonged motion and actuator calibration | High existence | Do not expose |
| Pool wheel/pump tests and charging-pile factory pairing | [Pool hidden tools](08-pool-cleaner.md#hidden-debug-and-incomplete-features), [test tools](09-diagnostics-testing.md#device-test-tool-catalog) | None outside a supervised engineering build | `allpowerfullRW`, factory pairing and test commands | Pool/pile models | Direct actuator operation and pairing mutation outside consumer workflow | High existence; Medium mapping | Do not expose |

## Cloud-only or credential-heavy

These may be valuable, but they increase dependence on private services,
regional host discovery, token lifecycle, quotas, and sensitive-data handling.

| App feature / candidate | Evidence | Likely HA shape | Transport / data source | Model gates | Safety / security concerns | Confidence | Priority |
|---|---|---|---|---|---|---|---|
| Live camera token acquisition and Agora receive path | [Live view](06-camera-video-vision.md#1-live-view-and-stream-lifecycle) | Camera entity or external Agora receive/proxy sidecar | Authenticated Mammotion stream-token API plus Agora RTC | Video/vision model, firmware, stream generation, network and quota | Token/secret/channel/UID redaction, renewal, encryption/private-cloud handling, resource cleanup | High | P0 hardening; P2 for broader model support |
| Cloud login, region discovery and token refresh | [Authentication](07-account-sharing-cloud.md#1-login-authentication-agreements-and-regions) | Config flow/reauthentication infrastructure | Regional Mammotion OAuth/user services | Region, agreement, account/provider state | Store secrets in HA config entry; do not extract/reimplement embedded proprietary client signing material | High | P0 foundational |
| Cloud RPC and asynchronous MQTT status | [Transport](01-onboarding-connectivity.md#cloud-api-and-mqtt), [cross-subsystem routing](10-protocol-report-index.md#routing-and-acknowledgement-model) | Integration transport, not user entity; diagnostics for connectivity | Base64 protobuf via `/v1/mqtt/rpc/thing/service/invoke`, service `device_protobuf_sync_service`, MQTT downlink | IoT identity, online state, shared-device rules | Least privilege, timeout/rate-limit handling, no arbitrary MQTT command assumption; HTTP completion is not device acceptance | High | P0 |
| Map backup list/create/rename/delete/restore | [Map backups](02-mapping-deployment.md#exhaustive-feature-table) | On-demand backup service and status, with restore in a separate confirmed flow | Private authenticated map backup REST APIs | Ownership, network, model/map compatibility | Restore/delete are destructive; require explicit snapshot ID, current-hash check and confirmation | High | P2 backup/list; P3 restore/delete |
| Point-cloud generation/progress/download | [Point cloud](02-mapping-deployment.md#exhaustive-feature-table) | On-demand generation service and downloadable diagnostic artifact | Authenticated point-cloud REST job APIs | `isSupportPointCloud`, cloud/network and mapped areas | Potentially large and location-sensitive files; validate URL/auth lifetime and storage policy | High | P2 |
| Work-history/report REST and device log upload | [Reports](03-work-planning-execution.md#job-history-reports-and-diagnostics), [support](07-account-sharing-cloud.md#5-feedback-support-and-log-upload) | On-demand report import; explicit support upload action only | Cloud report, feedback, upload-server and progress APIs | Account/model/network; 4G support differs | Reports/logs may contain location, identifiers, credentials and high-volume data; redact and require consent | High | P1 reports; P3 uploads |
| SSE message stream and message-center APIs | [Notifications](07-account-sharing-cloud.md#4-notifications-messages-push-and-sse) | Events and bounded recent-message sensor | Authenticated dynamic SSE endpoint and cloud message APIs | Region/account/app-generation | Reconnect/backoff and deduplication; sanitize content; avoid app push-token registration | High | P2 |
| SIM activation, traffic-limit activation and iNavi/NetRTK entitlement | [Network services](05-device-settings-maintenance.md#network-wifi-4g-rtk-and-positioning) | Read-only entitlement first; writes as setup flow | Authenticated SIM and positioning service APIs | Cellular/RTK SKU, region, paid/service state | ICCID and account/service state are sensitive; activation may have contractual or billing effects | High | P2 read; P3 write |
| Cloud pairing for RTK and PC210 pile | [RTK pairing](05-device-settings-maintenance.md#network-wifi-4g-rtk-and-positioning), [pool pairing](08-pool-cleaner.md#chargingdocking-pile-and-lora-pairing) | Setup wizard with explicit readback | Authenticated pairing APIs plus device LoRa/config commands | Exact robot/base/pile product generation | Mutates device association; enforce ownership and never log codes/LoRa IDs unnecessarily | High | P3 |
| Device sharing metadata and management | [Sharing APIs](07-account-sharing-cloud.md#current-share-apis) | Read-only privilege metadata; management outside normal entity model | Authenticated user/device cloud APIs | MA-IoT/legacy, region and owner role | Account/QR/email data, authorization changes and cross-region uncertainty | High | P1 metadata; P3 management |
| Voice-assistant linking, academy/help and remote RN/web content | [Voice](07-account-sharing-cloud.md#6-alexa-google-home-and-ma-voice), [hybrid/RN](07-account-sharing-cloud.md#7-hybridweb-features-and-native-bridge) | Documentation links at most | External OAuth/deep links and server-selected web/RN content | Region/account/product/external app | Duplicates HA capabilities; remote content changes independently and may introduce new terms | High existence | P3 / no control backlog |

## Unsupported or needs live validation

These features should remain unimplemented until packet capture, physical-device
tests, a second artifact, or an authoritative schema closes the named gap.

| App feature / candidate | Evidence | Likely HA shape after validation | Transport / data source | Model gates | Safety / security concerns | Confidence | Priority |
|---|---|---|---|---|---|---|---|
| Robot snapshot and recording | [Negative finding](06-camera-video-vision.md#3-recording-snapshots-and-non-live-video) | Camera snapshot/recording service only if a real source is found | No app-owned robot capture or recorder path found; rendered Agora frames are only a potential hook | Camera models | Storage, privacy and media credential lifecycle | High confidence unsupported in reviewed app | No backlog promise |
| Direct RTSP/WebRTC URL | [Camera implications](06-camera-video-vision.md#6-home-assistant-implementation-implications) | Standard camera source only if independently discovered | App uses Agora RTC; no Mammotion RTSP or generic WebRTC endpoint evidenced | Camera models | Avoid falsely presenting an Agora channel as a stable URL | High confidence unsupported | No |
| Pool area-selective cleaning | [Hidden pool features](08-pool-cleaner.md#hidden-debug-and-incomplete-features) | Area-clean service keyed by validated pool geometry | Protocol command `6`, but no complete consumer call path | Pool generation unknown | Robot movement and incomplete payload/ack semantics | Low-Medium | P3 research |
| Pool map/line decoding and selective geometry semantics | [Pool maps](08-pool-cleaner.md#maps-and-work-areas) | Map entity/export and hash-driven refresh | Pool map/line reports and local DB | Legacy versus PC210 | Coordinate frame, ordering and hashes require captures; do not write geometry | Medium | P2 read-only research |
| PC210 plan create/edit/disable polarity | [Pool cautions](08-pool-cleaner.md#uncertainties-and-cautions) | Plan CRUD services | `SpinoCtrl` plan protobuf plus cloud conversion/cache | PC210/SP | Enable polarity appears inverted in conversion; require create/read/disable/readback tests | Medium-High | P2 after live test |
| Generic map element type/shape writes | [Element model](02-mapping-deployment.md#generic-element-model) | Transactional map-edit service | Manual-element/navigation/SVG messages | Protocol generation | Numeric type/shape overlap; wrong mapping can corrupt safety geometry | Medium | P3 |
| Cost map/fog, raw obstacle borders and AI detections | [Mapping hidden features](02-mapping-deployment.md#hidden-debug-and-test-features), [vision](06-camera-video-vision.md#4-vision-vio-obstacles-and-ai) | Diagnostic image/entity only if normal report path and frame are known | Navigation cost-map/obstacle messages | Navigation/vision firmware | Large/high-rate data; do not treat inferred obstacle data as a safety sensor | Medium existence; Low reachability | P3 |
| Lock/PIN write or remote emergency-stop clear | [Settings uncertainties](05-device-settings-maintenance.md#uncertainties-and-reverse-engineering-cautions), [stop taxonomy](04-manual-control-safety.md#4-stop-lock-emergency-stop-and-self-check) | None without an explicit safe command and policy | No sufficiently traced consumer command | Model/security generation | Security bypass and physical hazard; physical e-stop is documented as manually unlocked | Low/negative | Do not expose |
| Exact manual-motion units, watchdog and firmware interlocks | [Manual uncertainties](04-manual-control-safety.md#10-uncertainties-and-decompiler-hazards) | Leased operator service with validated physical units | Driver motion protobuf and reports | Manual-control models | Packet send success does not prove watchdog, acceleration limit, collision or e-stop enforcement | Medium | P3; physical test required |
| Calibration start semantics | [Calibration](04-manual-control-safety.md#6-calibration-and-diagnostics) | Supervised setup action | Result/event path found; initiating command incomplete in reviewed path | Applicable dock/steering/navigation model | Can alter steering or charging-station frame | Low-Medium | P3 |
| Frost protection, maximum speed and latent work metadata | [Settings uncertainties](05-device-settings-maintenance.md#uncertainties-and-reverse-engineering-cautions), [work settings](03-work-planning-execution.md#feature-catalog-work-content-patterns-and-numeric-settings) | Read-only first, then entity if command/readback is traced | Field/resource evidence without complete active command path | Model/server capability | Avoid inventing semantics or ranges from labels/defaults | Low-Medium | P3 |
| Firmware-force/local pool OTA internals | [Pool OTA](08-pool-cleaner.md#pool-ota), [engineering force update](09-diagnostics-testing.md#app-side-diagnostics-fault-injection-and-feature-switches) | Update entity only after model-matrix and recovery testing | Multi-stage packet transfer or hidden force action | Pool module/version matching | Bricking risk, ordered chunks/checksums, long link stability and recovery path | High existence; Medium complete semantics | P3 |
| Debug reports, arbitrary debug config and factory/service commands | [Command call sites](09-diagnostics-testing.md#command-and-protocol-call-sites) | None in production; possibly an offline decoder fixture | Hidden debug/test protobuf and server-supplied key/value config | Developer/server permission and firmware | Arbitrary config injection, safety bypass, sensitive logs and undefined persistence | High existence | Do not expose |

## Suggested backlog order

1. **Preserve the baseline:** add capability/readback tests around existing
   entities and services, especially movement, blade, task round trips, pool
   family detection, camera credential redaction, and update prerequisites.
2. **Fill read-only gaps:** accessibility/preflight diagnostics, sync health,
   battery policy readback, component versions, richer safety/obstacle/VIO
   telemetry, ownership role, pool module/pile state, and bounded report
   summaries.
3. **Add reversible settings selectively:** smart charging, DND, area rename,
   acknowledged PC210 environment settings, and only then model-verified RTK or
   camera selection.
4. **Treat private-cloud expansions as separate projects:** reports, SSE,
   backups, point cloud, SIM/entitlement and pairing each need regional auth,
   privacy, quota, failure, and API-change handling.
5. **Keep hazardous and unsupported surfaces out of normal automation:**
   manual/blade control, map mutation, dock relocation, pairing/reset/account
   actions, engineering tools, and unvalidated protocol features need their own
   explicit safety or research track.

## Decompile spot checks used for this index

The subsystem reports remain the primary evidence. The following claims were
also checked directly in
`/Users/mattjoslin/mammotion-apk-decompile/src/sources/com/agilexrobotics`:

- `command/CommandManager.java:854-862`: battery policy is one compound write
  carrying smart mode, limit, off-peak enable, start and end.
- `map/MapDataImpl.java:98-116` and `map/db/AreaDBHelper.java:1555-1645`:
  accessibility depends on selected hashes plus current position type/hash and
  the local area/channel graph.
- `device/source/device/enums/DeviceType.java:713-786`: local-add pool, broad
  pool, charging-pile and PC210/SP predicates are distinct; broad pool includes
  `SD_PX`.
- `device/source/links/managers/MAIotManager.java:44`: the cloud protobuf RPC
  service identifier is `device_protobuf_sync_service`.
- `utils/HttpConstants.java:206`: the app composes the SSE endpoint ending in
  `channel-server/v1/sse/connect`.
- `map/activity/MapVideoActivity.java:78` and
  `map/activity/MapManualVideoActivity.java:1056`: the consumer wiper action
  sends `setCarWiper(2)`.
