# Mammotion app 2.3.8.19 — feature-catalog overview

## Purpose

This catalog is a durable reverse-engineering reference for the Mammotion Android
app. It records app-visible behavior, device/model gates, command and report
semantics, and possible Mammotion-HA integration opportunities. It is a catalog,
not a claim that every app feature is safe or practical to expose in Home
Assistant.

## Source snapshot

- App package: `com.agilexrobotics`
- Version: `2.3.8.19` (`versionCode 247`)
- Decompiled tree: `/Users/mattjoslin/mammotion-apk-decompile/src`
- Decompiled Java files: 30,867
- Decompiled tree size at analysis time: approximately 415 MB
- First-party Java root: `sources/com/agilexrobotics`
- First-party manifest components: 177 unique `com.agilexrobotics.*` entries
- Source evidence:
  - `resources/AndroidManifest.xml:3-4`
  - `sources/com/agilexrobotics/BuildConfig.java:15`

The source tree contains the output from every dex file in the XAPK, not just
`classes2.dex`. The catalog excludes generic third-party implementation details
unless the app's wiring of a library establishes user-visible behavior.

## Method

The sweep is divided by product subsystem. Each section:

1. inventories the relevant first-party packages, activities, services, resources,
   APIs, command builders, and protocol call sites;
2. traces user-visible behavior to command/report or HTTP/MQTT implementation where
   possible;
3. records model, firmware, region, account, and transport gates;
4. cites decompiled source paths and line numbers;
5. distinguishes confirmed behavior from inference and decompiler ambiguity; and
6. notes possible Mammotion-HA relevance without treating it as an implementation
   commitment.

Cross-cutting claims should be checked in more than one place: UI caller, manager or
view-model, and command/protocol implementation. Exact numeric constants are only
treated as high confidence when the decompiled expression and its call path agree.

## First-party package inventory

The largest feature-bearing first-party package groups (Java-file counts at sweep
time) are:

| Package | Files | Broad responsibility |
|---|---:|---|
| `map` | 558 | Mapping, map editing, live map, video, pool maps |
| `base_module` | 472 | Shared models, enums, utilities, lifecycle and reports |
| `device` | 446 | Deployment, info, settings, sharing and device abstraction |
| `home` | 188 | Main device dashboard and cross-feature orchestration |
| `widgets` | 122 | Shared app UI |
| `login` | 116 | Authentication and account linking |
| `work` | 109 | Work setup, schedules, history and mowing parameters |
| `signal` | 103 | Connectivity and signal diagnostics |
| `services` | 102 | Manual/touch modes and service workflows |
| `testing` | 102 | Internal test and factory/engineering tools |
| `me` | 101 | User/account settings and external integrations |
| `command` | 88 | Device command construction, transport and queues |
| `bind` | 87 | Device selection, discovery, binding and provisioning |
| `network` | 73 | HTTP stack |
| `feedback` | 61 | Support, logs and feedback |
| `maiot_module` | 49 | MQTT/cloud-device integration |
| `iot_module` | 42 | IoT abstraction and third-party-cloud calls |
| `rn` | 37 | React Native bridges and hot-updated features |
| `hybrid` | 33 | Web/hybrid features |
| `message` | 31 | Notifications and server-sent events |
| `espressif` | 26 | Wi-Fi provisioning |
| `find` | 24 | Device-location and recovery workflows |
| `mapbox` | 19 | Map rendering helpers |
| `proto` | 12 | App-local generated protocol surfaces |

## Device families seen in app gates

The app contains explicit branches for multiple generations and variants rather
than one homogeneous mower:

- LUBA/LUBA 1 variants
- LUBA 2 and LUBA 2 AWD/PRO variants
- LUBA mini and related model codes
- YUKA and YUKA mini variants
- SPINO S1/E1 swimming-pool cleaners
- RTK reference stations and charging/docking stations as separately managed
  devices

Raw model-code strings are not consistently human-readable after decompilation.
Each subsystem report records the exact gate or helper method rather than assuming
two similarly named variants have identical capabilities.

## Catalog sections

- `01-onboarding-connectivity.md` — device discovery, binding, BLE/Wi-Fi/4G,
  cloud/MQTT and signal diagnostics
- `02-mapping-deployment.md` — deployment, RTK/dock, map creation/editing,
  geometry, hashes, backups and map sync
- `03-work-planning-execution.md` — mowing/cleaning setup, tasks, schedules,
  parameters and work lifecycle
- `04-manual-control-safety.md` — joystick/manual movement, command cadence,
  blades, calibration and safety behavior
- `05-device-settings-maintenance.md` — settings, firmware, maintenance, errors,
  hardware controls and find-device behavior
- `06-camera-video-vision.md` — live video, camera, VIO/vision, streaming and
  AI-assisted features
- `07-account-sharing-cloud.md` — accounts, sharing, notifications, support,
  voice assistants and cloud user features
- `08-pool-cleaner.md` — SPINO-specific map, cleaning and hardware behavior
- `09-diagnostics-testing.md` — internal engineering, factory, log and diagnostic
  features
- `10-protocol-report-index.md` — cross-subsystem command/report/topic/API index
- `11-ha-opportunity-index.md` — Home Assistant opportunity and risk index
- `12-coverage-and-open-questions.md` — coverage audit, gaps and claims needing
  live or adversarial verification

## Reading cautions

- Decompiled Kotlin often becomes large Java state machines; synthetic method names
  and line mappings can be misleading.
- Resource strings reveal shipped UI copy but do not prove a feature is reachable
  for every account, region, firmware or model.
- A protobuf command builder proves the app can construct a command; a reachable UI
  caller or manager call path is needed to establish that the feature is exposed.
- Boolean arguments on transport APIs may be routing hints rather than acknowledgments.
- App behavior can depend on server-provided capability flags and React Native
  bundles not fully represented by one static Java call path.
- Commands that move hardware or operate blades require independent safety design
  before any Home Assistant exposure.
