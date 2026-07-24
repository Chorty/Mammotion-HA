# Camera, video, vision, AI, and live media

## Scope and confidence

This catalog covers the decompiled Android application at
`/Users/mattjoslin/mammotion-apk-decompile/src` (manifest version `2.3.8.19`,
version code `247`) and concentrates on Mammotion-owned wiring. Bundled Agora,
AndroidX Camera, picture-selector, ExoPlayer, Mapbox camera, and other generic
library internals are intentionally excluded unless application code invokes
them.

Evidence is static and decompiler-derived. Names and control flow are generally
high confidence; numeric enum meanings, server behavior, and firmware behavior
that are not decoded by the app remain uncertain.

## Executive catalog

| Capability | Finding | Home Assistant relevance |
|---|---|---|
| Robot live view / FPV | Implemented with Agora RTC, versioned token acquisition, remote camera selection, reconnect handling, 4G quota UI, and optional media encryption. | High. A future HA camera entity would need the cloud token API plus Agora-compatible media handling; MQTT/BLE alone is insufficient. |
| Manual control with FPV | Full-screen and mini-player FPV are embedded beside dual-stick/manual movement, mower controls, lamp, camera switching, wiper, map, and job history. | High for a dashboard/control panel, but safety and command transport must remain separate from the media channel. |
| Camera switching | Positions `1` and `3` are used for front/rear camera selection; YUKA receives multiple camera UIDs and low remote stream type. | Medium. Expose as a select entity only after camera-position semantics are verified per model. |
| Wiper | One-shot `setCarWiper(2)` command, labelled “Vision Module Wiper.” | High and relatively tractable as a button entity. |
| Live snapshot / recording | No app-owned capture or recorder call is wired into the robot FPV surfaces. Agora rendered frames are observed, but the callback is used for stream-health/UI behavior, not persistence. Local phone camera/video selection exists for feedback uploads. | Snapshot/recording should be treated as unsupported by the observed protocol, not inferred from Android media permissions. |
| Video teaching/tutorials | Dedicated activity fetches and plays remote tutorial content; packaged MP4s also support feature guidance. | Low for device control; useful for documentation links only. |
| Vision/VIO status | App displays VIO state and ambient/vision brightness; command protocol includes a minimal `vision_ctrl_msg {type, cmd}`. | Medium. Status sensors may be possible if the underlying state-report fields are available in the integration protocol. |
| AI obstacle detection | Work settings expose model-dependent detection modes and describe off/standard/sensitive behavior; animal protection has a separate state/mode and packaged explainer video. | High. Candidate select/switch entities, but numeric values and model constraints must be mapped carefully. |
| Diagnostics | Hidden/secondary RTC statistics model and `VideoReportLayout` display local/remote audio/video metrics; extensive logs include stream parameters and RTC state. | Useful for diagnostics, but tokens/secrets must be redacted. |

## 1. Live view and stream lifecycle

### Entry points and user experience

`MapVideoActivity` is the standalone live-view route. It creates
`JoinChannelVideo.newInstance(1, deviceId)`, embeds it, provides camera switching
and wiper controls, and only shows the YUKA camera setting for YUKA devices
(`sources/com/agilexrobotics/map/activity/MapVideoActivity.java:27-28,64-80,99-116,150-178`).

`MapManualVideoActivity` is the FPV/manual-control experience. It can construct
both a main stream (`newInstance(0, deviceId)`) and mini stream
(`newInstance(1, deviceId)`), switches among map, mini-map, full video, and mini
video fragments, and retains manual movement/control state
(`sources/com/agilexrobotics/map/activity/MapManualVideoActivity.java:131-133,1297-1314,2028-2098`).
The same screen controls headlamp state and mower/manual controls
(`MapManualVideoActivity.java:443-515,1531-1575,1882-1895`).

The live fragment has explicit loading, refresh, reconnect, offline resume, token
renewal, remote-user join/offline, first-frame, and remote-video-state handling
(`sources/com/agilexrobotics/map/video/JoinChannelVideo.java:197-464,626-673,943-1022,1663-1696`).
Requests are deduplicated for 1.5 seconds and normal requests get a 10-second
loading timeout (`JoinChannelVideo.java:89-90,642-669`).

### Token/signaling paths

There are two credential paths, gated by state-machine field `newversionfpv`:

1. Legacy: authenticated `streamSubscription` receives a `VideoResp` containing
   `appid`, `channelName`, `token`, `uid`, and a camera list. The app also sends
   `deviceAgoraJoinChannelWithPosition(1)` to the robot, apparently instructing
   it to publish/join (`JoinChannelVideo.java:652-666`;
   `sources/com/agilexrobotics/base_module/bean/resp/VideoResp.java:7-55`).
2. New: authenticated `getStreamToken` posts a `StreamTokenReq` with device and
   camera-state data and receives `StreamTokenRspon`; no legacy robot join
   command is issued in this branch
   (`sources/com/agilexrobotics/map/viewmodel/MapDeviceModel.java:41-101`).

Both Retrofit calls use the access token from shared preferences as the
`Authorization` header. They are created against
`HttpConstants.STREAM_SUBSCRIPTION`; the decompiled annotations retain method
names but not a literal relative path, so the final URL is unresolved here
(`sources/com/agilexrobotics/map/api/MapApiUtils.java:24-35`;
`sources/com/agilexrobotics/map/api/MapApiService.java:144,152`).

The credential DTOs are sufficient to identify the protocol contract but not to
reproduce server authorization policy. Server-side ownership checks, token TTL,
and whether tokens are Agora dynamic keys remain unknown.

### Agora configuration and media behavior

The app initializes `RtcEngine` with the response `appId`, live-broadcast channel
profile `1`, global area code, optional private-cloud access points, and an
application-scenario parameter (`appScenario=100`, `serviceType=11`). It
registers an `IVideoFrameObserver`
(`sources/com/agilexrobotics/map/video/JoinChannelVideo.java:677-704`).

On join it:

- creates a `SurfaceView`;
- configures YUKA camera UIDs to remote stream type `1`;
- sets client role `1`;
- enables video while disabling local camera and local microphone;
- auto-subscribes to remote audio/video;
- nevertheless sets `publishMicrophoneTrack` and `publishCameraTrack` true in
  `ChannelMediaOptions`;
- joins with cloud token, channel ID, and UID
  (`JoinChannelVideo.java:720-773`).

The contradictory “publish true” options versus disabled local tracks mean static
evidence supports a flow intended predominantly for remote viewing, but cannot
prove strict receive-only runtime behavior. No app-owned local camera capture is
attached to the reviewed flow.

Rendered remote frames are observed (`onRenderVideoFrame`), while capture,
pre-encode, and media-player callbacks simply satisfy the observer interface
(`JoinChannelVideo.java:473-579`). This is a potential future snapshot hook, but
the reviewed code does not encode, save, or expose those frames.

### Encryption and sensitive logging

New-generation streaming explicitly configures Agora `AES_256_GCM2`. The response
`data.key` becomes `encryptionKey`; Base64-decoded `data.salt` is copied into
`encryptionKdfSalt`. Encryption is enabled before channel join when the key is
non-empty (`JoinChannelVideo.java:754-756,881-882,1260-1279`;
`StreamTokenRspon.java:18-23,66-83`). Server-side key derivation and rotation
remain unknown.

The app logs complete response objects and explicitly logs channel ID, UID, and
token (`JoinChannelVideo.java:723,808-837`;
`MapDeviceModel.java:91-100`). Any HA implementation should never reproduce this
behavior: token, encryption secret, channel, app ID, device ID, and UID should be
redacted from logs and diagnostics.

### 4G limits and model gates

FPV over mobile-network transport (`NET_USED_TYPE_MNET`) is quota-aware. The app
tracks `availableTime` and `availableTime_service`, presents “3 minutes used”
and monthly/free-time exhaustion prompts, and can require Wi-Fi
(`JoinChannelVideo.java:777-805`;
`resources/res/values/strings.xml:2424-2425`).

`FPV4GVideoStateMannager` refreshes after a three-second startup delay, retries,
shows timeout/refresh UI, and uses device/model/firmware checks for down-converted
4G FPV (`sources/com/agilexrobotics/map/video/FPV4GVideoStateMannager.java:127-164,179-233,241-333`).
Its support list names YUKA Mini/VP and LUBA MN/LD/VP/VA families, with additional
type and firmware comparisons in `DeviceUtils`
(`sources/com/agilexrobotics/device/source/device/utils/DeviceUtils.java:404-439,952-1006,1082-1134`).

HA should not assume a simple online/offline gate. Live-view availability depends
on product family, firmware, current network, server entitlement/free time, and
the new/legacy FPV generation.

## 2. Camera controls, wiper, and manual FPV

The camera selector toggles numeric position `1` ↔ `3`. In manual FPV, switching
away from position `1` hides the rain-brush/wiper button
(`MapManualVideoActivity.java:944-954`). The standalone viewer performs the same
toggle (`MapVideoActivity.java:64-72`). The labels/artwork imply front and rear
cameras (`resources/res/drawable-xxhdpi/icon_fond_camera.webp`,
`icon_back_camera.webp`), but the numeric mapping should be confirmed on each
model.

`JoinChannelVideo.setVideoPosition` updates the requested camera position and
restarts/reconnects the stream (`JoinChannelVideo.java:1636-1661`). Legacy
signaling uses the robot command helper; new signaling rebuilds the token request
with camera state. YUKA responses may include multiple `(uid, camera)` entries
(`VideoResp.java:7-55`; `JoinChannelVideo.java:743-749`).

The wiper is a direct device command: both live-view activities invoke
`MACommandApiHelper.setCarWiper(2)` and show “Cleaning the vision module”
(`MapVideoActivity.java:75-80`; `MapManualVideoActivity.java:1053-1058`;
`resources/res/values/strings.xml:2892,3670`). Protocol menus identify
`MUL_SET_WIPER`/ACK and camera-video set/ACK families
(`sources/com/agilexrobotics/command/menus/PbMsgType.java:15-20`).

Manual FPV combines video with remote movement, blade/height/speed controls,
headlamp, status, map, and reconnect-to-BLE assistance. This is an operator
control surface, not autonomous “video steering.” HA should expose movement only
with explicit press/hold/release semantics and stop-on-disconnect safeguards.

## 3. Recording, snapshots, and non-live video

### Robot stream

No robot-FPV snapshot button, `MediaRecorder`, Agora recording API, or
`MediaStore` persistence call was found in `JoinChannelVideo`,
`MapVideoActivity`, or `MapManualVideoActivity`. The live frame observer gives
the app decoded frames but currently uses them for first-frame/health behavior.
Therefore:

- live robot snapshots are **not evidenced as a shipped feature**;
- robot-side or cloud recording is **not evidenced**;
- Android camera/media permissions do not prove either feature.

### Phone camera and feedback attachments

The application can invoke/select local photos and videos for feedback. The
manifest declares camera, microphone, image/video/audio media-read, legacy
external-storage, and flashlight permissions, and queries system image/video
capture intents (`resources/AndroidManifest.xml:41-63,64-72`).
Picture-selector strings include “Record Video,” album/camera permission errors,
duration limits, and saving a video to the phone
(`resources/res/values/strings.xml:2347-2417`). Feedback code includes
`GridVideoAdapter1`, `UploadVideoTask`, and `VideoCompressUtil`; these are local
support-upload workflows, not robot-camera capture.

`READ_PRIVILEGED_PHONE_STATE`, `READ_SETTINGS`, `READ_LOGS`,
`WRITE_MEDIA_STORAGE`, and some legacy storage permissions are privileged or
obsolete for an ordinary Play-installed app. Their declaration does not mean
they are granted (`AndroidManifest.xml:30-36,54-60`).

### Video teaching and playback

`VideoTeachingActivity` is a dedicated tutorial screen with an insert-bumper
title and a button that closes the screen
(`sources/com/agilexrobotics/work/setting/activity/VideoTeachingActivity.java:31-46`).
The map API exposes `getUserGuideVideo` (`sources/com/agilexrobotics/map/api/MapApiService.java:47-52`).
Packaged media includes `res/raw/rtk_video_tips.mp4` and
`res/raw/animal_protect_video.mp4`.

No app-owned React Native native module dedicated to camera/video was found in
`com/agilexrobotics/rn/module`; that directory contains common, guide, report,
battery, work-plan, and work-setting bridges. Searches also found no
`ReactVideoView`/`VideoViewManager` implementation. React Native screens may
still render web/bundled JS content or use a dependency omitted/obfuscated by
the decompiler, so “no RN video bridge” is a scoped static conclusion, not proof
that no RN screen can display video.

## 4. Vision, VIO, obstacles, and AI

### VIO and vision status

`StatusBarVioActivity` is a routed status detail screen. It observes the current
device state, reads `CarStatusBean.vioState` and brightness, and delegates display
mapping to `SignalHelper.refreshVioSignal` and `refreshVioBrightnessSignal`
(`sources/com/agilexrobotics/signal/newstatus/StatusBarVioActivity.java:27-29,52-99,107-133`).
Resources label this “Visual Positioning Status,” include “3D vision is not
initialized,” and provide a vision progress/status layout
(`resources/res/values/strings.xml:452,1689-1690,3607-3610`;
`resources/res/layout/activity_statusbar_vio.xml`).

`VioToAppInfoEvent` is a small event DTO carrying VIO-to-app data, but direct
consumers were not found outside generated/event plumbing
(`sources/com/agilexrobotics/base_module/event/VioToAppInfoEvent.java:1-57`).
This may be vestigial, consumed reflectively/EventBus-style, or superseded by
state-machine reports.

The navigation protobuf includes `vision_ctrl_msg` with integer fields
`type=1` and `cmd=2`, carried as `MctlNav.vision_ctrl` at oneof/submessage slot
51 (`sources/com/agilexrobotics/proto/MctrlNav.java:48690-49084,49489-49508`).
The command menu labels `NAV_VISION_CTRL (cmd 1, subtype 51)` as a radar static
test command (`sources/com/agilexrobotics/command/menus/PbMsgType.java:168`).
This looks diagnostic/manufacturing-oriented rather than a normal consumer
vision toggle.

### Obstacle avoidance and AI modes

The app presents “AI Obstacle Detection” with:

- Off: collision/bumper-based detection;
- Standard: automatic obstacle and hazard avoidance;
- Sensitive: obstacle, hazard, and non-grass-surface avoidance
  (`resources/res/values/strings.xml:970,1433,1593,1617,2712,2825`).

Work-setting state stores the selection as `detect_mode` and serializes it in the
historic protocol field `ultrasonicBarrier`. Observed values include `0`, `1`,
`2`, `10`, and `11`; UI mapping and server-provided defaults/forced values vary
by device generation (`sources/com/agilexrobotics/work/setting/view/SettingOptionsView.java:464-591`;
`sources/com/agilexrobotics/work/setting/api/WorkingSettingManage.java:745-747,873-875,940-941`;
`sources/com/agilexrobotics/base_module/entity/PlanBean1.java:410-411,657-658`).
Do not label these numeric values in HA without tracing the active model's option
array and firmware capability response.

Animal protection is separate from work-plan obstacle sensitivity. The state
machine carries `animalProtectStatus` and `animalProtectMode`, settings expose an
animal-protection result code, and the app bundles an explanatory video
(`sources/com/agilexrobotics/device/source/device/bean/CarStateMachineBean.java`
metadata near the animal-protection accessors;
`sources/com/agilexrobotics/device/setting/DeviceSettingHelperImpl.java:43-49`;
`resources/res/raw/animal_protect_video.mp4`).

Vision also supports positioning and mapping behavior: resources describe
vision-module installation/checks, visual positioning, automatic surroundings
checks, and camera lighting/visibility requirements
(`resources/res/values/strings.xml:1098,1265,1688-1690,2941`).
The navigation protocol includes obstacle border/data transfers, manual no-go
zone construction, a costmap payload (`width`, `height`, center, yaw, resolution,
cost array), and cover paths (`MctrlNav.java:20365-23102,48690-49659`).
These are map/navigation data structures; static code does not prove that raw
camera imagery or AI detections are uploaded to the phone.

## 5. Hidden and secondary diagnostics

`VideoReportLayout` accepts Agora local/remote audio/video stats
(`sources/com/agilexrobotics/map/view/VideoReportLayout.java:15,92-113`).
`StatisticsInfo` stores last-mile probe, RTC, and local/remote media statistics
(`sources/com/agilexrobotics/base_module/bean/model/StatisticsInfo.java:3-13,82-106`).
No normal production navigation path to this layout was found, suggesting a
hidden, debug, or currently dormant report surface.

The app logs RTC connection state, first frame, user join/offline, retry reasons,
4G quota transitions, token responses, channel parameters, and encryption
status throughout `JoinChannelVideo` and `FPV4GVideoStateMannager`. These logs
are useful for reproducing state machines but are unsafe as-is for HA issue
diagnostics because credential-bearing DTO `toString()` methods include tokens
(`VideoResp.java:55`).

The vision control command's “radar static test” label, VIO detail screen, vision
log resource title, RTC report widget, and extensive manufacturing/test package
are the principal hidden/secondary diagnostic indicators. No callable app UI
for exporting raw vision frames or AI model output was established.

## 6. Home Assistant implementation implications

Recommended entity candidates, in increasing difficulty:

1. **Button:** vision-module wiper, mapping to `setCarWiper(2)`.
2. **Sensors:** VIO state, vision brightness, FPV availability/network type,
   entitlement time, and animal-protection status, if their state reports are
   present in the existing integration transport.
3. **Select/switch:** obstacle-detection mode, animal-protection mode, and camera
   position, only after per-model capability and enum mapping.
4. **Camera entity:** requires authenticated stream-token acquisition, Agora
   channel join, token renewal, optional encryption/private-cloud handling,
   camera UID selection, and lifecycle/quota behavior. A direct RTSP/WebRTC URL
   is not exposed by the reviewed app.

Agora is the actual live-media transport. Although Agora internally uses
real-time media technologies, no Mammotion-owned `org.webrtc` call site was
found. Describing this as a generic WebRTC/RTSP endpoint would overstate the
evidence.

For an HA camera implementation, keep cloud access-token handling server-side,
redact media credentials, avoid persistent storage unless explicitly enabled,
and stop all movement commands independently of video reconnect state. A proxy
or sidecar capable of Agora receive/decode may be more practical than embedding
Agora directly in Home Assistant Core.

## 7. Uncertainties and validation targets

- Literal REST paths and base host behind `HttpConstants.STREAM_SUBSCRIPTION`
  were not recovered in this pass; Retrofit service names and request contracts
  are evidenced.
- Token TTL, server ownership checks, and server-side encryption-key derivation
  and rotation need runtime network capture or unobfuscated configuration. The
  client cipher is confirmed as Agora `AES_256_GCM2`.
- Camera position `1`/`3` and camera UID ordering need physical verification
  across YUKA/LUBA variants.
- `publishCameraTrack=true`/`publishMicrophoneTrack=true` conflicts with disabled
  local tracks; runtime Agora behavior should be confirmed.
- No robot snapshot/recording feature was found. A server-side feature, newer
  app version, native-only implementation, or dynamically delivered module
  could still provide one.
- `VioToAppInfoEvent` has no obvious direct consumer; EventBus/reflection or
  obsolete code is possible.
- AI mode numeric values are reused through a legacy `ultrasonicBarrier` field
  and are model/config dependent.
- The decompile includes native libraries/resources not represented as Java
  source. This catalog does not claim semantic coverage of proprietary native
  binaries.

## Files and areas reviewed

Primary app-owned files:

- `sources/com/agilexrobotics/map/video/JoinChannelVideo.java`
- `sources/com/agilexrobotics/map/video/FPV4GVideoStateMannager.java`
- `sources/com/agilexrobotics/map/video/DownFPVTipsDialogCommon.java`
- `sources/com/agilexrobotics/map/activity/MapVideoActivity.java`
- `sources/com/agilexrobotics/map/activity/MapManualVideoActivity.java`
- `sources/com/agilexrobotics/map/viewmodel/MapDeviceModel.java`
- `sources/com/agilexrobotics/map/api/MapApiService.java`
- `sources/com/agilexrobotics/map/api/MapApiUtils.java`
- `sources/com/agilexrobotics/map/view/VideoReportLayout.java`
- `sources/com/agilexrobotics/base_module/bean/resp/VideoResp.java`
- `sources/com/agilexrobotics/base_module/bean/resp/VideoTokenResp.java`
- `sources/com/agilexrobotics/base_module/entity/StreamTokenReq.java`
- `sources/com/agilexrobotics/base_module/entity/StreamTokenRspon.java`
- `sources/com/agilexrobotics/base_module/bean/model/GlobalSettings.java`
- `sources/com/agilexrobotics/base_module/bean/model/StatisticsInfo.java`
- `sources/com/agilexrobotics/device/source/device/utils/DeviceUtils.java`
- `sources/com/agilexrobotics/device/source/device/bean/CarStateMachineBean.java`
- `sources/com/agilexrobotics/command/menus/PbMsgType.java`
- `sources/com/agilexrobotics/command/MACarDataManagerAPI.java`
- `sources/com/agilexrobotics/proto/MctrlNav.java`
- `sources/com/agilexrobotics/proto/LubaMul.java`
- `sources/com/agilexrobotics/signal/newstatus/StatusBarVioActivity.java`
- `sources/com/agilexrobotics/base_module/event/VioToAppInfoEvent.java`
- `sources/com/agilexrobotics/work/setting/view/SettingOptionsView.java`
- `sources/com/agilexrobotics/work/setting/api/WorkingSettingManage.java`
- `sources/com/agilexrobotics/work/setting/api/TaskAssignmentManager.java`
- `sources/com/agilexrobotics/work/setting/activity/VideoTeachingActivity.java`
- `sources/com/agilexrobotics/device/setting/DeviceSettingHelperImpl.java`
- feedback video adapter/task/compression helpers and all
  `com/agilexrobotics/rn/module` files.

Resources and configuration:

- `resources/AndroidManifest.xml`
- `resources/res/values/strings.xml`
- camera/video/VIO/FPV layouts and drawables under `resources/res`
- `resources/res/raw/rtk_video_tips.mp4`
- `resources/res/raw/animal_protect_video.mp4`
- relevant route constants and resource IDs.

Third-party packages were inventory-searched for Agora, WebRTC, React Native
video, CameraX, picture selection, and playback presence; only Mammotion-owned
imports/call sites and application-visible resources were used as behavioral
evidence.
