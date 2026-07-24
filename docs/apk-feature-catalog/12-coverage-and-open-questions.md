# Coverage and open-questions ledger

## Status and scope

This ledger closes the static catalog pass for the Mammotion Android app
**2.3.8.19** (`versionCode 247`). It records what was swept, what was
adversarially checked, what the audit corrected, and what still requires
runtime evidence.

The catalog is a **full decompile sweep and feature catalog**, not proof of every
server-controlled, account-gated, firmware-gated, React Native, hybrid-web, or
dynamically delivered feature. “Covered” below means that the relevant
first-party source owner was inventoried, routed to a report, searched for
feature-bearing code, and behavior-bearing call paths were reviewed. It does
not mean that every generated class was read line by line or that every
cataloged feature was exercised on hardware.

| Scope metric | Audited value | Interpretation |
|---|---:|---|
| Decompiled Java files | **30,867** | All dex-derived Java files in the XAPK decompile were available to the sweep, not only one `classes*.dex`. |
| First-party manifest components | **177** unique `com.agilexrobotics.*` entries | Activities, services, receivers, and providers were inventoried as reachability and ownership clues. |
| First-party Java root | `sources/com/agilexrobotics` | All first-party package groups were assigned to a subsystem report or a cross-cutting index. |
| Catalog reports read for this ledger | **14** existing reports | Reports `00`–`11`, `13`, and `14` were reconciled; this file is report `12`. |
| Mechanical citation audit | **1,580 citations checked in the first pass; 1,842 paths and 1,790 line anchors in the final pass** | The first pass found **76 defects**. After repair, the final whole-catalog validator reported **0 defects** across paths, anchors, high-risk claim relevance, tables, links, headings, placeholders, secrets, and whitespace. |

Source snapshot and top-level package counts are recorded in
[00-overview.md](00-overview.md#source-snapshot) and
[00-overview.md](00-overview.md#first-party-package-inventory).

## Report and source-owner coverage matrix

| Report | Primary evidence owner(s) | Coverage disposition |
|---|---|---|
| [00 — Overview](00-overview.md) | Manifest, build metadata, first-party package inventory | Snapshot, package census, method, device-family routing, and catalog boundaries. |
| [01 — Onboarding and connectivity](01-onboarding-connectivity.md) | `bind`, `espressif`, `command/bleconnect`, `iot_module`, `maiot_module`, `device/source`, `signal` | BLE discovery/binding, BLUFI provisioning, Wi-Fi/4G/SIM, APN/radio controls, link selection, MQTT/cloud routing, RTK/iNavi status, and positioning guidance. |
| [02 — Mapping and deployment](02-mapping-deployment.md) | `map`, `mapbox`, `device/deploy`; map-related `command`, `proto`, `base_module`, `home`, `work`, `services`, `testing` | Map lifecycle, geometry, hashes, deployment, dock/RTK, automatic/manual creation, backup/restore, SVG, and destructive map operations. |
| [03 — Work planning and execution](03-work-planning-execution.md) | `work`; work-related `home`, `services`, `command`, `proto`, `base_module` | Area selection, settings, plans, schedules, DND, lifecycle commands, reports, and work-state reduction. |
| [04 — Manual control and safety](04-manual-control-safety.md) | Manual-control code in `command`, `map`, `device/deploy`, `home`, `work`, `services` | Joystick conversion, cadence, stop behavior, manual mowing/blades, lock/self-check gates, calibration, and safety limitations. |
| [05 — Settings and maintenance](05-device-settings-maintenance.md) | `device/info`, `device/setting`, `find/device`, `signal`; supporting command/state owners | Firmware, hardware settings, networking, charging, cutters, weather, anti-theft, maintenance, errors, and latent/beta controls. |
| [06 — Camera, video, and vision](06-camera-video-vision.md) | Camera/video paths in `map`, `home`, `device`, `command`; Agora and vision wiring | Live-stream token/signaling, camera selection, FPV/wiper, VIO/vision/AI, media limits, privacy, and negative capture/RTSP findings. |
| [07 — Account, sharing, and cloud](07-account-sharing-cloud.md) | `login`, `me`, `device/share`, `message`, `feedback`, `hybrid`, `rn`, `services`; account/cloud APIs | Authentication, profile, ownership/sharing, notifications/SSE, support uploads, voice assistants, H5/RN surfaces, commerce, server tips, and telemetry. |
| [08 — Pool cleaner](08-pool-cleaner.md) | Pool/SPINO paths across `home`, `map`, `work`, `signal`, `command`, `proto`, `testing` | Device identities, cleaning modes, plans, maps, status, pile/LoRa pairing, OTA, and incomplete factory features. |
| [09 — Diagnostics and testing](09-diagnostics-testing.md) | `testing`, `feedback`, `log`, trace owners, hidden settings/tools | Developer reachability, actuator/sensor tools, fault injection, factory workflows, logs/uploads, first-party telemetry, and downloaded error vocabulary. |
| [10 — Protocol/report index](10-protocol-report-index.md) | `command`, `proto`, `device/source`, MQTT/HTTP API interfaces | Cross-report command, report, topic, API, routing, acknowledgement, model-gate, and HA-counterpart index. |
| [11 — HA opportunity index](11-ha-opportunity-index.md) | Current integration plus reports `01`–`10` | Read/write opportunity classification, safety/security risk, implementation status, unsupported surfaces, and backlog order. |
| **12 — This ledger** | All reports plus targeted decompile re-checks | Coverage accounting, audit history, corrected omissions/false positives, static limits, live-verification queue, and safety hold list. |
| [13 — Model/capability matrix](13-model-capability-matrix.md) | `DeviceType`, capability helpers, firmware and runtime gates, representative callers | Device identity normalization, model/firmware/server gates, capability hazards, and per-family validation. |
| [14 — Architecture glossary](14-architecture-glossary.md) | App startup, device/link/command/report/cache/map/UI/persistence/background owners | Cross-package architecture, terminology, data flows, persistence authority, native/H5/RN boundaries, and protocol-consumer cautions. |

Generated `R`, Hilt/Dagger, data-binding, `BuildConfig`, and generated protobuf
files were inventoried. They were used as supporting evidence for packaged UI,
dependency wiring, resources, and wire fields, but were not treated as proof of
reachable behavior without a first-party caller or reducer.

## Audit methodology

The catalog and this ledger used the following audit sequence:

1. Census the complete decompile, manifest, first-party package tree, resources,
   model enums, and major generated protocol surfaces.
2. Assign first-party source owners to reports by subsystem; retain cross-owner
   features in both their behavioral report and the protocol/architecture
   indexes.
3. Trace claims from UI or service entry point through activity/fragment,
   view-model/manager, command/API builder, routing layer, and report/cache
   readback where decompilation allowed.
4. Search repository-wide for command symbols, callers, endpoint suffixes,
   report fields, model gates, resources, and apparently orphaned activities.
5. Separate evidence classes: reachable behavior, latent/hidden behavior,
   generated wire capability, server/RN/H5-delivered behavior, and negative
   finding.
6. Require corroboration before assigning exact numeric semantics. Preserve raw
   enum or protocol values when JADX control flow, a collapsed `switch`, or
   integer-rich wrapper prevents a defensible label.
7. Mechanically validate citation syntax, target existence, and cited line
   ranges, then manually re-read repaired claims. The first audit pass checked
   **1,580 citations** and found **76 defects**. After repair, the final pass
   checked 1,842 paths, 1,790 line anchors, 1,508 numeric/high-risk claim lines,
   107 tables, 130 links, and 311 headings with **0 remaining defects**.
8. Reconcile each subsystem’s uncertainty section into this ledger so a
   catalog entry cannot silently become an implementation claim.

## Completed adversarial checks

The completed static adversarial work included:

- repository-wide caller/sink searches for manual motion, blade/manual-mow,
  stop, lock, self-check, calibration, map mutation, schedule, network,
  camera, diagnostics, and factory commands;
- comparison of duplicated command-helper and remote-control stacks, including
  routing booleans versus acknowledgement behavior;
- verification of final-zero/manual-stop timing, Bluetooth-only manual-mow
  caller gates, and the absence of a separately proven software emergency-stop
  primitive;
- checks for hidden gamepad paths, service/touch-mode protocol callers,
  manifest export/deep-link reachability, dormant resources, and generated-only
  “features”;
- cross-checks of model helpers against representative UI call sites so a
  friendly model name was not substituted for runtime capability data;
- distinction between device commands and local cache setters, especially
  battery/state methods;
- searches for commerce/payment paths, server tips, first-party event
  collection, remote error-code data, and downloaded RN/H5 boundaries;
- direct spot checks of runtime Wi-Fi/4G radio toggles, APN editing,
  positioning-optimization guidance, Shopify SIM/iNavi links, tip
  show/read routes, and error-catalog routes;
- negative searches supporting the conclusions that there is no proven native
  Play Billing implementation, no Mammotion RTSP URL, no app-owned robot
  snapshot/recording path, and no product “auto backwash” feature in the
  reviewed APK.

The most safety-focused adversarial record is
[04-manual-control-safety.md](04-manual-control-safety.md#9-adversarial-checks-performed);
cross-subsystem implementation spot checks are also listed in
[11-ha-opportunity-index.md](11-ha-opportunity-index.md#decompile-spot-checks-used-for-this-index).

## Omissions found, corrected, and re-audited

These features were absent or materially incomplete in an earlier catalog
state. They were added to the named reports and then included in the citation
re-audit.

| Corrected omission | Static evidence established | Corrected in |
|---|---|---|
| Shopify SIM/iNavi links | `GET /device-server/v1/shopify/goods/link` supplies a server-generated destination used by 4G/SIM and iNavi purchase/renewal screens. This corrects “no native billing” into the narrower and accurate “no native Play Billing/payment processor; server-backed commerce exists.” | [07, subscriptions and commerce](07-account-sharing-cloud.md#11-subscriptions-and-commerce) |
| Server tips resource/show/read | The service tab fetches `/user-server/v2/tips/resource` and reports `/tips/push/show` and `/tips/push/read`; content selection and campaign state are server driven. | [07, server-driven tips](07-account-sharing-cloud.md#111-server-driven-tips-and-campaign-tracking) |
| Positioning optimization | `PointOptimizationActivity` derives one of four guidance panels from mower and RTK satellite/status inputs. The thresholds are presentation logic, not proven firmware limits. | [01, feature catalog](01-onboarding-connectivity.md#feature-catalog) |
| First-party telemetry | Mammotion’s own queued behavioral-event pipeline posts app/device/product/phone identity, event/value, time, and area to `/user-server/v1/user/collection`, separate from Firebase. | [07, first-party telemetry](07-account-sharing-cloud.md#112-first-party-behavioral-telemetry); [09, local logging and telemetry](09-diagnostics-testing.md#local-logging-and-telemetry) |
| Downloaded error catalog | Startup checks `code/version` and downloads localized paginated remediation data from `code/page-lan`; bundled strings are not an exhaustive error vocabulary. | [09, downloaded error-code catalog](09-diagnostics-testing.md#downloaded-error-code-catalog) |
| Runtime radio/APN controls | Post-onboarding screens can toggle mower Wi-Fi and 4G and edit cellular APN. These are high-risk writes because they can strand the active cloud route. | [01, feature catalog](01-onboarding-connectivity.md#feature-catalog); [05, network and positioning](05-device-settings-maintenance.md#network-wifi-4g-rtk-and-positioning) |

## False positives corrected

| Earlier interpretation | Corrected finding | Consequence |
|---|---|---|
| `AudoBackwashPop` implied an automatic backwash feature. | The class is a generic confirmation dialog used with the “only 4G” warning before disabling cellular. No backwash resource text or product command caller was found. | **Auto backwash is not cataloged as a Mammotion product feature.** See [05, hidden/beta/support/test features](05-device-settings-maintenance.md#hidden-beta-support-and-test-features). |
| “Anti-theft activates at 50 m” was treated as a confirmed command threshold. | Shipped UI copy describes an alert beyond 50 m from the working area, but the reviewed client only establishes a cloud-side setting/track flow. It does not prove that 50 m is a client-settable, locally enforced, or server-fixed threshold. | The catalog records the **50 m wording with low threshold confidence**, not a protocol constant or HA number entity. See [05, weather/safety/security](05-device-settings-maintenance.md#weather-safety-security-locks-and-animal-protection). |

## Remaining static-analysis limits

- JADX can collapse Kotlin coroutine/state-machine bodies, `switch` branches,
  synthetic accessors, lambdas, and generated oneof dispatch. A visible method
  or field is not always enough to recover polarity, enum meaning, or order.
- Generated protobuf Java establishes fields and possible messages, but not
  deployed firmware support, accepted ranges, units, watchdogs, interlocks, or
  whether an apparently valid oneof is used by this app build.
- Resource strings and manifest entries establish packaged surfaces, not
  reachability for a particular account, region, model, firmware, entitlement,
  experiment cohort, or server capability response.
- Legacy and current API generations coexist. Endpoint presence does not prove
  that all regions still serve it or that request/response schemas are stable.
- HTTP hosts are selected at runtime; authentication, quotas, authorization,
  retention, server-side validation, and failure policy cannot be recovered
  completely from interface annotations.
- The APK exposes native RN bridges and H5 containers, but a downloaded React
  Native bundle or remote page can add, remove, or redefine page behavior
  without changing these Java sources.
- Local databases and caches show client representation, not necessarily
  firmware or server authority. Cache mutation is not proof of a device write.
- Negative static searches are bounded to this APK/decompile. They do not prove
  absence from firmware, private server code, dynamically delivered content,
  another platform, or a newer build.
- Static analysis cannot establish real radio behavior, timing under packet
  loss, mower kinematics, coordinate frames, timezone handling, cloud
  consistency, or physical safety enforcement.

## Claims requiring additional evidence

| Evidence needed | Claims/questions it must resolve |
|---|---|
| Packet capture | Exact command envelopes and ack correlation; mower schedule `subCmd` values; opaque read/write IDs and boolean polarity; route-angle units/ranges; map element type/shape values; RTK-link enums; APN/radio results; pool plan enable polarity; report timestamp units. |
| Live mower/RTK/dock/pool hardware | Motion units, cadence tolerance, watchdog, acceleration and collision interlocks; release-to-zero behavior; blade/RSSI stops; self-check/lock bits; dock/calibration start semantics; positioning guidance inputs; map hash/SVG reconciliation; pool maps, plans, pump/wheel commands, OTA recovery, and radio-disable rollback. |
| Live server/account matrix | Regional hosts, auth/token lifecycle, owner/shared permissions, SSE schemas, capability/config responses, report/history schemas, error-catalog pagination/versioning, Shopify eligibility/status, tips campaign semantics, iNavi/SIM entitlements, API quotas, and server authorization. |
| Extracted/downloaded RN bundle and remote H5 | Actual route inventory and page-level behavior behind native bridges; commercial, academy/help, support, and remotely configured features not encoded in Java. |
| Newer APK or second artifact | Whether negative findings remain true; renamed/removed legacy APIs; new model codes, capability gates, commands, report fields, safety flows, and repaired decompiler bodies. |

## Safety-sensitive areas intentionally unimplemented

Static existence is not authorization to expose these through Home Assistant.
They remain unimplemented or outside normal automation pending a dedicated
safety/security design and the live evidence above:

- unattended continuous manual drive, remote positioning, and physical
  controller paths;
- manual mowing, direct blade start, cutter direction/RPM, and engineering
  actuator tests;
- a purported software emergency-stop clear, lock/PIN writes, or any action
  that bypasses a physical stop or self-check;
- boundary/channel/no-go/safety-zone mutation, delete-all map, dock relocation,
  base reset, and restore-over-current-map;
- RTK pairing/handoff/calibration writes that can strand positioning or
  invalidate map assumptions;
- remote Wi-Fi/4G disable, Wi-Fi forget, APN mutation, or provisioning without a
  local recovery path and secret-handling design;
- factory reset, unbind, account deletion, ownership/share mutation, and
  third-party-account unlinking as ordinary services;
- firmware force/local pool OTA, debug config writes, fault injection, safety
  bypasses, factory pairing, and endurance/aging tools;
- obstacle/vision/debug data presented as a certified safety sensor.

The implementation risk classification and existing-integration comparison are
maintained in [11-ha-opportunity-index.md](11-ha-opportunity-index.md).

## Prioritized verification queue

| Priority | Verification work | Exit criterion |
|---:|---|---|
| **P0** | Existing HA safety/readback baseline | Tests prove fresh-state gating, command/result correlation, no optimistic state, movement lease/final stop behavior, blade restrictions, capability-based availability, and secret redaction. |
| **P1** | Read-only live telemetry and capability matrix | Captures across representative LUBA, YUKA, mini, RTK, and pool families map raw states/reports to model/firmware gates without speculative enums. |
| **P1** | Schedule, DND, work-setting, and report round trips | Create/read/update/delete captures establish numeric enums, polarity, timestamps/timezones, unknown-field preservation, and acknowledgement/readback behavior. |
| **P1** | Map and positioning read path | Captures establish coordinate frames, hashes, SVG/common-data ordering, positioning inputs, map refresh rules, and non-destructive read-only pool geometry. |
| **P1** | Regional cloud/account matrix | Test accounts establish hosts, token refresh, owner/shared authorization, SSE payloads, server capabilities, downloaded errors, tips, and entitlement states with redacted fixtures. |
| **P2** | Bounded reversible settings | Query/write/query tests validate charging policy, DND, area rename, rain polarity, camera selection, pool environment, and other acknowledged settings with rollback. |
| **P2** | Radio/APN and RTK/iNavi configuration | Supervised local tests establish exact enums/results and a recovery route before any radio, APN, pairing, handoff, or positioning-mode write is considered. |
| **P2** | Camera/Agora lifecycle | Validate token renewal, UID/camera mapping, encryption/private-cloud variants, cleanup, quotas, model gates, and credential redaction; do not infer RTSP or recording. |
| **P2** | Pool plan/map/OTA behavior | PC210/SP hardware tests establish plan polarity/IDs, map/line semantics, pile pairing, module/version gates, ordered transfer, failure recovery, and safe read-only surfaces. |
| **P3** | RN/H5 extraction and newer-APK differential | Archive and analyze delivered bundles/pages where lawful, then diff a newer APK to identify new/removed features and invalidate stale negative findings. |
| **Hold** | Hazardous motion, blade, map mutation, factory/debug, account-destructive operations | No implementation until each feature has a separately reviewed threat/safety model, local-presence policy, failure-mode tests, audit trail, and explicit opt-in. |

## Final coverage statement

The 2.3.8.19 catalog provides a broad, first-party-package-complete static sweep
of the full 30,867-file decompile, its 177 first-party manifest components, and
the feature-bearing native, protocol, cloud-interface, hybrid, and RN-bridge
surfaces assigned above. It is suitable as a static implementation and
verification map.

It is **not** an assertion that every server-controlled or dynamically delivered
feature has been observed, that every command is safe or supported on every
device, or that static evidence substitutes for packet capture and physical
testing. The first citation audit’s 76 defects were repaired. The final
whole-catalog verification reported zero remaining defects across source paths,
line ranges, high-risk claim relevance, Markdown structure, internal links,
headings, placeholders, secrets, and whitespace.
