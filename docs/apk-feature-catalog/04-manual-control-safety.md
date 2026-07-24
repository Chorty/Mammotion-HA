# APK feature catalog: manual control, motion, calibration, and safety

## Scope and evidence standard

This catalog covers the decompiled Android application's `command/`, `services/touchmode`, manual-control map activities/fragments, relevant home/work controls, and generated protobuf use. Evidence paths are relative to:

`/Users/mattjoslin/mammotion-apk-decompile/src/sources`

Unless stated otherwise, a behavior is **confirmed** when both the UI/call site and command builder are visible. “Inferred” means the protobuf field or UI text is clear but firmware-side semantics are unavailable. Decompiled names such as `f12054i` and compiler-generated branches are treated cautiously.

## Executive summary for Home Assistant

- Direct driving is `MctrlDriver.MctlDriver.todev_devmotion_ctrl` / `DrvMotionCtrl`, with signed integer `set_linear_speed` and `set_angular_speed`. The app normally emits it every **200 ms (5 Hz)** while either component is nonzero. It sends via the currently selected local link and explicitly disables IoT fallback for this command.
- On-screen stick input is angle-plus-radius. The app snaps headings into axial/diagonal sectors, multiplies the forward component by **10** and angular component by **4.5**, and applies a separate **15-unit radial subtraction** to the two-stick path. A physical gamepad scales axes by **115** before the same conversion.
- Releasing one stick zeros only that axis. The periodic sender then emits the combined state; when both axes reach zero it sends a final zero command and cancels its timer. The UI event itself does not synchronously force a zero packet.
- Manual mowing is a separate stateful command, `MctrlDriver.MctlDriver.mow_ctrl_by_hand` / `DrvMowCtrlByHand`, carrying `main_ctrl`, `cut_knife_ctrl`, `cut_knife_height`, and `max_run_speed`. It is refreshed every **800 ms**, but only over Bluetooth.
- Blade start requires an app disclaimer followed by a slide-to-start gesture. The app auto-stops the blade after **8 s** if the initial motion condition is not satisfied, after **3 s** continuously stationary while manual mowing, or when filtered Bluetooth RSSI is at/below **-80 dBm** after the ten-sample filter has filled. The start gate reads the cached average (initially zero), so the ten-sample requirement applies to automatic weak-signal shutdown, not reliably to initial blade start.
- The app blocks stick motion on lock, location error, relevant self-check lock bits, low-battery stop, hidden/disabled controls, and (in the full manual activity) while the settings popup is open. These are client-side gates, not a substitute for firmware interlocks.
- No app-issued electronic emergency-stop/unlock command was found in the scoped dex-derived call sites. Emergency stop is surfaced as a reported fault requiring manual recovery. “Software stop” for driving is a zero `DrvMotionCtrl`; task stop/pause and manual-mow exit are different commands.
- `services/touchmode` contains instructional text/video for obstacle, recharge-route, turnaround, blade, and pool modes. It does not send motion commands. “Follow Robot” elsewhere is a map-camera orientation mode, not person-following motion.

## 1. Direct manual motion

### 1.1 Command and envelope

`CarRemoteControlManage2.send()` forwards its current integer pair to `MACommandApiHelper.sendControl()` (`com/agilexrobotics/command/CarRemoteControlManage2.java:158-173`). The command is:

```text
LubaMsg
  msgtype = MSG_CMD_TYPE_EMBED_DRIVER
  sender  = DEV_MOBILEAPP
  rcver   = DEV_MAINCTL
  msgattr = MSG_ATTR_REQ
  version = 1
  subtype = numeric account/user id (fallback 102)
  seqs    = monotonically incremented app sequence
  timestamp = monotonic clock
  driver.todev_devmotion_ctrl:
    set_linear_speed  = signed int
    set_angular_speed = signed int
```

Evidence: envelope construction and sequence/account behavior at `com/agilexrobotics/command/MACommandHelper.java:121-145,217-268`; payload construction at `com/agilexrobotics/command/app/MACommandApiHelper.java:1469-1471` and the link-explicit twin at `com/agilexrobotics/command/MACommandHelper.java:1504-1508`.

The boolean transport flag passed to `sendOrderMsg_Driver` is `false`. In `sendMsg`, `false` goes directly to `postCustomeDateByte` and bypasses IoT service invocation (`MACommandHelper.java:217-226,229-256`). This establishes local/current-link routing at the helper, but it is not intrinsically “BLE-only”: concrete raw delivery still depends on the selected link manager. The older deploy path supplies an explicit `MALinkManager`; the newer app helper uses its bound transport.

### 1.2 Joystick scaling and direction behavior

The canonical converter is `RockerControlUtil.transfrom3(angle, radius)`:

- angle is in degrees;
- output 0 is `int(sin(adjustedAngle) * radius)`;
- output 1 is `int(cos(adjustedAngle) * radius)`;
- the caller sets `linearSpeed = output[0] * 10`;
- the caller sets `angularSpeed = int(output[1] * 4.5)`.

Evidence: `com/agilexrobotics/base_module/utils/RockerControlUtil.java:551-691`, and `com/agilexrobotics/command/CarRemoteControlManage2.java:225-236`.

The converter's named thresholds are **30°**, **7°**, and **15°** (`RockerControlUtil.java:10-12`). The decompiled branch structure shows:

- ±7° snap windows around 90° and 270°;
- ±15° snap windows around 0°/360° and 180°;
- broad 30° sector clamps around diagonal transitions;
- special reflection logic around 270°, presumably to preserve right-stick turn direction.

The code is decompiler-damaged in several duplicated branches. The trigonometric endpoint and the principal snap constants are reliable; a complete piecewise mathematical reconstruction of every boundary is not.

The two-stick path calls `getPercent(radius)`: radius ≤15 becomes 0, otherwise 15 is subtracted (`CarRemoteControlManage2.java:51-56,225-226`). This is a **radial dead zone plus offset**, not merely a clamp. The single-stick `transfromSpeed()` path does not subtract 15 (`CarRemoteControlManage2.java:240-252`), but no scoped production call site to that method was found.

Physical controllers use:

- left vertical axis 1: negative → 90°, positive → 270°;
- right horizontal axis 11: negative → 180°, positive → 360°;
- magnitude `abs(axis) * 115`.

Evidence: `com/agilexrobotics/map/activity/MapManualActivityNew.java:3269-3282` and `MapManualSimpleActivity.java:258-277`. At full axis, the two-stick dead-zone subtraction leaves radius 100, producing nominal maxima near **±1000 linear** and **±450 angular**, subject to angle snapping and integer truncation. No explicit physical-controller dead-zone is applied before multiplication; platform axis normalization may provide one.

### 1.3 Cadence, start, release, and final stop

`CarRemoteControlManage2.frequency` defaults to `0.2f`; the timer period is `frequency * 1000`, i.e. **200 ms** (`CarRemoteControlManage2.java:20,37,58-91`). The first nonzero stick update creates a `Timer`, scheduled with initial delay 0 and a 200 ms period. Subsequent stick callbacks update shared speed fields; they do not create additional timers.

Each tick:

1. emits the current pair;
2. if state is manual mowing, reports stationary/nonstationary to the blade manager;
3. if both values are zero, cancels the timer.

Evidence: `CarRemoteControlManage2.java:58-88`.

Release behavior:

- left release calls `stopLeft()` (linear = 0);
- right release calls `stopRight()` (angular = 0);
- both are reflected on the next timer tick;
- when both become zero, that tick sends `(0,0)` and then cancels.

Evidence: setters at `CarRemoteControlManage2.java:203-214`; activity releases at `MapManualActivityNew.java:3819-3841` and `MapManualSimpleActivity.java:619-645`.

`transfromDoubleRockerSpeed()` with no arguments zeros both values only once per prior moving period and attempts to start/continue the timer (`CarRemoteControlManage2.java:216-223`). If no timer exists and both are already zero, `testSendControl` returns without sending. Consequently, this is a software stop by zero velocity only when a moving timer/state exists; it is not an independently retried emergency-stop primitive.

### 1.4 UI motion gates

The full manual activity replaces requested motion with zero and resets the control UI when any of these are true:

- `MODE_LOCK` (17);
- `MODE_LOCATION_ERROR` (37);
- `lowBatteryStopControlCar`;
- `isSendContorl == false`;
- `BlockErrorBeanChangeUtils.isLockStatus(selfCheck)`;
- manual-mow/settings popup is showing;
- the remote-control view is not visible.

Evidence: `MapManualActivityNew.java:3795-3815`. The simple activity applies lock, location-error, low-battery, and self-check-lock gates (`MapManualSimpleActivity.java:584-616`).

The state values are defined at `com/agilexrobotics/device/source/device/enums/DeviceWorkState.java:4-38`, notably ready 11, working 13, returning 14, charging 15, lock 17, pause 19, manual mowing 20, and location error 37.

Model/map-mode restrictions also hide the rocker during automatic corridor exploration: `DeviceWorkState.isShowRockerView()` returns false only for `MODE_CORRIDOR_WORKING` (45) (`DeviceWorkState.java:192-194`). Planning fragments additionally zero controls when their UI/state switches away from manual input; all identified call sites are listed under cross-checks below.

## 2. Manual lawn mowing and blade control

### 2.1 Proto and values

`OperateOnDevice()` builds:

```text
MctrlDriver.MctlDriver.mow_ctrl_by_hand = DrvMowCtrlByHand {
  main_ctrl       = 0 exit / 1 enter-manual mode
  cut_knife_ctrl  = 0 off / 1 on
  cut_knife_height = app-selected integer height
  max_run_speed   = app-selected float
}
```

It uses command/service number **51**, transport-routing flag `true`, and a `MSG_ATTR_REQ` driver envelope (`MACommandHelper.java:409-411`; app twin `MACommandApiHelper.java:406-408`). Field meanings 0/1 are confirmed by caller state transitions and logs, not protobuf enum types.

The manager starts with `main_ctrl=0`, blade off, height 0, and maximum run speed **0.3** (`com/agilexrobotics/map/ManualLawnMowingManager.java:66-72`). `initState()` enters manual mode (`main_ctrl=1`), publishes the speed to the UI, and starts command refreshing (`ManualLawnMowingManager.java:595-602`). `foceManualModeAndStart(boolean)` sets manual mode and optionally blade-on, then sends immediately (`:553-557`).

### 2.2 Bluetooth-only routing and keepalive

Despite `OperateOnDevice` being capable of IoT routing in the generic helper, `ManualLawnMowingManager.send()` refuses to call it unless the current link type is Bluetooth. On non-Bluetooth it disposes the refresh subscription and returns (`ManualLawnMowingManager.java:393-407`).

After every state-changing send (all `position` values except 7), the manager replaces an Rx interval with a new **800 ms** interval. Every interval tick resends the complete state with diagnostic position 7 (`ManualLawnMowingManager.java:421-423,484-495`). Refresh stops when both `main_ctrl` and `cut_knife_ctrl` are zero (`:417-420`) or on explicit cleanup.

The `position` argument is log-only; it is not serialized into `DrvMowCtrlByHand`. Observed positions:

- 1: automatic/explicit blade off;
- 2: speed update;
- 3: height update;
- 4: blade start;
- 5: normal manual-mode exit;
- 7: 800 ms refresh;
- 8: cleanup exit;
- 9: force manual mode/start;
- 10: height setter;
- 113: extra terminal zero command during `cancel3`.

Evidence: `ManualLawnMowingManager.java:118-140,234-278,392-424,497-557,700-703`.

### 2.3 Blade start, stop, and app safety gates

Starting the blade is deliberately multi-step:

1. reject if locked, charging, or the cached filtered Bluetooth RSSI is ≤ -80 dBm (the cache starts at zero, so this does not prove ten samples are required before initial blade start);
2. show four-part manual-mowing disclaimer and require agreement;
3. show a slide-to-start popup;
4. only its completion callback sets `cut_knife_ctrl=1`.

Evidence: disclaimer construction/callbacks at `ManualLawnMowingManager.java:304-365`; slide callback at `:262-278`; gates at `:761-790`.

Confirmed stop conditions:

- User stop: handler event 204 sets blade off and sends (`:118-140,792-797`).
- Initial no-motion timeout: blade start schedules event 201 after **8000 ms** (`:274-275`). Motion cancels this initial timeout through `knifeDiscOff(false)`.
- Stationary timeout: while device state is `MODE_MANUAL_MOWING`, the 200 ms motion timer reports `(linear,angular)==(0,0)`; `knifeDiscOff(true)` schedules blade-off after **3000 ms**, and movement cancels it (`CarRemoteControlManage2.java:65-79`; `ManualLawnMowingManager.java:604-615`).
- Weak Bluetooth: the manager keeps up to 10 RSSI samples, sorts them, drops min/max, averages the remainder, and if average ≤ **-80 dBm** while blade is on sends event 202/off (`ManualLawnMowingManager.java:660-693`). It requires at least 10 samples.
- Exit/cancel, app-level cleanup, or losing Bluetooth: zero state and/or refresh disposal (`:497-551,618-632,800-810`; application cleanup call at `com/agilexrobotics/MyApplication.java:395`).
- Device lock dismisses both blade/settings popups (`ManualLawnMowingManager.java:705-725`).

The blade feedback path consumes reported knife state and overwrites local `cut_knife_ctrl`, updating the animation/text (`ManualLawnMowingManager.java:650-658`). This is state reconciliation, not an explicit per-command acknowledgement.

Speed and height changes resend the full manual-mow state. Generic speed setting separately clamps to `DeviceMultimodelHelper`'s model-specific min/max before sending `DrvSrSpeed{rw=1,speed}` (`MACommandHelper.java:1881-1891`). Manual-mow `max_run_speed` is not visibly clamped in `OperateOnDevice`; its popup/model configuration is therefore part of the effective validation boundary.

## 3. Ack semantics and routing

All driver commands are wrapped as `MSG_ATTR_REQ`; sequence IDs increment per envelope (`MACommandHelper.java:125-136,259-268`). The boolean passed to `sendOrderMsg_*` is **not “wait for acknowledgement.”** It controls whether the helper may use IoT service routing:

- `false`: send raw bytes on the selected link;
- `true`: use IoT service when link is IoT; with explicit manager, use BLE when a BLE client exists, otherwise fall back to IoT.

Evidence: `MACommandHelper.java:217-256`.

Thus:

- `DrvMotionCtrl`: `false`, raw/current link, no IoT service fallback;
- `DrvMowCtrlByHand`: `true` at helper level, but caller enforces Bluetooth before invoking it;
- speed/height/cutter-mode settings: `true`, routable and expected to produce reported state/response.

No command-specific retry/ack tracker for `DrvMotionCtrl` was found. Reliability is achieved by 5 Hz repetition and a final zero. Manual mowing similarly refreshes complete state at 1.25 Hz. UI feedback comes from asynchronous state reports (`onKnifeState`, device state, self-check), not matching a motion response to `seqs`.

## 4. Stop, lock, emergency stop, and self-check

### 4.1 Stop taxonomy

These operations must not be conflated:

- **Drive software stop:** `DrvMotionCtrl(0,0)`.
- **Manual-mow/blade stop:** `DrvMowCtrlByHand(main_ctrl/cut_knife_ctrl)` with blade and/or mode zero.
- **Task pause/stop/return:** navigation task-control commands in home/work/map flows.
- **Mapping stop-and-save:** `NavTaskCtrl{type=1,action=4,result=0}` (`MACommandHelper.java:1988-1997`).
- **Mapping abandon:** `NavTaskCtrl{type=1,action=18,result=0}` (`:1976-1985`).
- **Physical emergency stop:** reported safety state; no scoped app command to clear it.

Resource text explicitly says emergency stop requires checking the robot and manual unlock (`resources/res/values/strings.xml:1453,2905`; source root rather than `sources/`). The absence of a clear command call site is safety-significant: HA should expose e-stop as a sensor/condition, not advertise an app-derived remote reset.

### 4.2 Lock and self-check

Device state 17 is `MODE_LOCK` (`DeviceWorkState.java:16`). Incoming lock state is a bitfield decoded into five booleans:

- bit 0;
- bit 1, with the decompiled special case `lockState == 1`;
- bits 2, 3, and 4.

Evidence: `com/agilexrobotics/command/app/MACarDataManager.java:5235-5239`. The semantic names of those five booleans require inspection of `DeviceLockStatue`/`DeviceLockType`; the bit positions are confirmed.

Self-check is reported as an integer/bitmask and used through `BlockErrorBeanChangeUtils` predicates. Manual control specifically uses `isLockStatus`; home/work gates also use `isControlError`, `isBeginContinueWorkingError`, `isReturnChargeError`, and `isCreateMapError` before their respective actions. Representative evidence:

- manual control: `MapManualActivityNew.java:3795-3815`;
- home control gate: `com/agilexrobotics/home/fragment/HomeFragmentNew.java:3244`;
- home work/continue gates: `HomeFragmentNew.java:2601,3159,3338`;
- map-creation gate: `HomeFragmentNew.java:2951`.

`SelfCheckFragment` presents checks for battery, charging, bumper, RTK, pass-through, and model-gated vision; it animates/checks asynchronously and does not itself send motion (`com/agilexrobotics/work/setting/self_check/SelfCheckFragment.java:76-170,249-321`). Home also shows `SelfCheckDialog` before quick/start-now work flows (`HomeFragmentNew.java:7189-7239,7807-7923`).

The app gates are defense in depth only. HA should retain firmware state validation and should default-deny manual movement when lock/self-check data is stale or unavailable.

## 5. Heading, turning, follow, and remote modes

### 5.1 Heading and turning

There is no separate “turn-to-heading” command in the direct joystick path. Turning is signed angular velocity in `DrvMotionCtrl`; a stationary turn is linear 0 with angular nonzero. Direction snap sectors in `transfrom3` make axial forward/back and in-place turn inputs stable.

The map UI also has “Follow Robot” orientation: robot remains visually upward while the map rotates. Resource evidence is `resources/res/values/strings.xml:1551,3747`. This changes presentation, not mower navigation.

`services/touchmode` defines `MODE_TYPE_TURN` and builds turnaround instructional entries/videos (`com/agilexrobotics/services/touchmode/AnimalProtectActivity.java:26-31`; `CarModeActivity.java:76-94`; `CarModeCustomView.java:132-149`). No command helper is referenced by those builders/activities. Treat these as help content, not control modes.

### 5.2 Follow and remote

No person-following remote-drive proto or production call site was found in the scoped dex-derived sources. The word “follow” has three unrelated meanings:

- map follows robot orientation (display only);
- recharge follows the perimeter (navigation strategy/help text);
- safety instructions tell the human to follow the mower during mapping.

The recharge help item is built in `CarModeActivity.java:57-75` / `CarModeCustomView.java:114-131`. Again, `services/touchmode` only displays content and can float that content over other activities; it does not issue commands.

“Remote control” in class/view names means local app joystick. `remoteRestart()` is a maintenance reboot request, not a drive mode: `remote_reset_req_t{magic=1916956532,bizid=currentTimeMillis,reset_mode=0,force_reset,account}` (`MACommandHelper.java:1312-1314`).

## 6. Calibration and diagnostics

### 6.1 Charging-station/rotation calibration

`CalibrationActivity` is an event-driven UI for leaving the dock and rotating:

- starts a **30 s** timeout (`WorkRequest.DEFAULT_BACKOFF_DELAY_MILLIS`);
- result/type event filter is `RotationBean.action==0 && type==5`;
- result 3 starts a repeating 18 s rotation animation;
- result 0/1 stops, reports success/failure, and optionally returns to manual control;
- when returning, `operate=2` if `breakPointType>0`, otherwise 0.

Evidence: `com/agilexrobotics/map/activity/CalibrationActivity.java:47-64,66-120,140-177`.

The activity itself does not originate the firmware calibration command in the visible code; it consumes `RotationBean` and forwards regional data to `getRegionalData`. Therefore the initiating proto/value is unresolved in this activity and should not be synthesized for HA.

MN231 charging-station calibration/reset also lives in `PlanMapLandRestPileFragment`, with explicit calibration state and message 305 (`com/agilexrobotics/map/fragment/mn231/PlanMapLandRestPileFragment.java:116,191-192`). Its joystick call sites use the same `CarRemoteControlManage2` conversion.

INavi cancellation is explicit: `MctrlSys.MctlSys.app_to_dev_set_mqtt_rtk_msg.stop_nrtk_flag=1`, command/service **11123** (`MACommandHelper.java:495-501`; app twin `MACommandApiHelper.java:492-494`).

### 6.2 Diagnostics and maintenance commands relevant to safety

Confirmed helper commands include:

- get/set cutter work mode: `current_cutter_mode/AppGetCutterWorkMode` and `cutter_mode_ctrl_by_hand/AppSetCutterWorkMode{cutter_mode}`, service 46 (`MACommandHelper.java:1514-1516,1575-1577`);
- cutting height: `todev_knife_hight_set/DrvKnifeHeight{knife_height}`, service 45 (`:1843-1846`);
- read speed: `bidire_speed_read_set/DrvSrSpeed{rw=0}`, service 50 (`:1125-1131`);
- set speed: `DrvSrSpeed{rw=1,speed}`, model-clamped, service 46 (`:1881-1891`);
- reset blade usage time: `MctrlSys.todev_reset_blade_used_time=1` (`:1465-1467`);
- radar static test and factory/QC test helpers exist, but are diagnostics rather than normal control (`:1257-1265,1510-1512`);
- remote reboot as described above.

These commands should be separated in HA from routine controls, with explicit confirmation for blade mode, calibration, reset counters, tests, and reboot.

## 7. Model and feature gates

Confirmed model gates affecting this scope:

- Luba Pro NAV envelopes route to `DEV_NAVIGATION` instead of the caller-supplied NAV receiver; driver control remains `DEV_MAINCTL` (`MACommandHelper.java:121-158`).
- Calibration animation/art differs for Luba VA/HM (`CalibrationActivity.java:71-77,122-134`).
- Manual-mow settings initialize with mode 1 for YuKa and 2 otherwise (`ManualLawnMowingManager.java:230-234`).
- Cutter speed, turn, recharge, obstacle, blade, and vision help entries are selected through device capability predicates in `CarModeActivity`/`CarModeCustomView`; they do not imply command support by themselves.
- Generic speed is clamped by `DeviceMultimodelHelper.getControlSpeedMin/Max()` (`MACommandHelper.java:1881-1889`).
- Vision self-check UI is capability-gated (`SelfCheckFragment.java:143-161`).

Exact per-model numerical speed limits are not in the reviewed command method and remain unresolved.

## 8. Home Assistant implementation relevance

Recommended entity/service split:

- sensors: work state, lock state/bitfield, self-check mask, link type, RSSI, blade state, blade height, configured speed, charge state;
- guarded services: `manual_drive(linear, angular, duration/lease)`, `manual_mow_enter`, `blade_start`, `blade_stop`, `manual_mow_exit`, `set_cut_height`, `set_speed`;
- separate maintenance services: calibration cancel, cutter mode, blade-time reset, diagnostics, reboot;
- never label zero velocity as emergency stop.

Required safety properties for an HA implementation:

1. Use a short movement lease/dead-man timer; refresh at approximately 5 Hz and send repeated zero on lease expiry/disconnect.
2. Require fresh lock/self-check/charge/link telemetry before enabling motion.
3. Restrict blade/manual-mow to Bluetooth unless firmware behavior is independently proven for another link.
4. Reproduce blade start confirmation and enforce the 8 s initial-motion, 3 s stationary, and RSSI safety stops locally.
5. Treat command enqueue/send success as transport success only. Await reported state where an acknowledgement matters.
6. Serialize motion writers. The app has one mutable speed pair/timer; multiple HA automations must not race.
7. Clamp values by model and do not infer physical units from the app integers. `1000` and `450` are protocol-scale maxima observed in UI conversion, not proven m/s or rad/s.

## 9. Adversarial checks performed

- Searched all `src/sources/**/*.java` call sites for `sendControl`, `OperateOnDevice`, `transfromDoubleRockerSpeed`, and `ManualLawnMowingManager`; no additional payload builders were found.
- Checked both duplicated helper stacks (`MACommandHelper` and `MACommandApiHelper`) and both remote managers (`CarRemoteControlManage` and `CarRemoteControlManage2`). Their payload/scaling logic matches; cadence source differs only by `DeviceDeployConstants.frequency1` versus local `frequency=0.2`.
- Distinguished `position` logging values from serialized proto fields.
- Verified that the helper routing boolean controls IoT eligibility, not ack waiting.
- Searched command/map/home/work/services/device trees and resources for software stop, emergency stop, lock/unlock, follow/remote, heading/turn, calibration, blade, and self-check terms.
- Checked physical gamepad call sites separately from touch-stick callbacks.
- Checked stop-on-release for a final zero and found a one-tick asynchronous stop, not an immediate direct send.
- Checked whether manual mowing could route through generic IoT support and found a caller-level Bluetooth-only gate.
- Checked `services/touchmode` for command/proto use and found none.
- Checked lock handling at UI, manual-mow manager, incoming state decode, and home/work gates.

## 10. Uncertainties and decompiler hazards

- `RockerControlUtil.transfrom3` has JADX duplicated/control-flow warnings. Constants, trigonometric output, and caller multipliers are reliable; some exact angular interval boundaries are not.
- Protocol field units are not declared in generated Java. Linear/angular integers and speed floats must not be assigned SI units without firmware/proto schema documentation or packet experiments.
- `sendOrderMsg_*` service numbers (51, 46, etc.) are IoT routing identifiers/operation IDs in the helper; they are not protobuf field numbers.
- Firmware acknowledgements, watchdog duration, acceleration limiting, collision interlocks, and e-stop enforcement are outside the APK and cannot be concluded from app code.
- The app has no visible dedicated e-stop-clear command in scope. Absence in decompiled call sites does not prove the firmware protocol lacks one.
- The initiating charging-station rotation/calibration command was not identified from `CalibrationActivity`; only its event/result semantics are confirmed.
- Self-check bit meanings and the five lock-bit boolean names require a dedicated error/lock catalog; this document records only control-relevant predicates and bit positions.
- Physical-controller dispatch does not visibly apply the same UI safety predicates before calling the converter. The surrounding activity/state/UI may gate delivery, but this is weaker evidence than the touch callbacks and merits runtime testing.
- Timer fields are read/written across UI and `TimerTask` threads without `volatile`/locking. Java visibility and release-to-zero timing are potential race concerns.

## 11. Dex-derived call-site cross-check

Direct conversion/control call sites found by repository-wide `rg`:

- `com/agilexrobotics/map/activity/MapManualActivityNew.java:3280-3281,3797-3813,4230-4240`
- `com/agilexrobotics/map/activity/MapManualSimpleActivity.java:272-276,588-615`
- `com/agilexrobotics/map/activity/MapManualVideoActivity.java:1540-1553`
- `com/agilexrobotics/map/fragment/PlanMapLandFragment.java:4997-5039,5724`
- `com/agilexrobotics/map/fragment/mn231/PlanMapLand231Fragment.java:4960-4999`
- `com/agilexrobotics/map/fragment/mn231/PlanMapLandRestPileFragment.java:2127-2157`
- payload sinks only: `CarRemoteControlManage2.java:172`, `device/deploy/device/manage/CarRemoteControlManage.java:169`, `MACommandHelper.java:1504-1508`, `MACommandApiHelper.java:1469-1471`.

Manual-mow payload call sites:

- `ManualLawnMowingManager.java:414,542`
- payload builders only: `MACommandHelper.java:409-411`, `MACommandApiHelper.java:406-408`.

No other `sendControl` or `OperateOnDevice` production call site was found under the complete decompiled `src/sources`.

## 12. Files reviewed

Primary, read in detail:

- `com/agilexrobotics/command/CarRemoteControlManage2.java`
- `com/agilexrobotics/device/deploy/device/manage/CarRemoteControlManage.java`
- `com/agilexrobotics/base_module/utils/RockerControlUtil.java`
- `com/agilexrobotics/device/deploy/utils/RockerControlUtil.java`
- `com/agilexrobotics/command/MACommandHelper.java`
- `com/agilexrobotics/command/app/MACommandApiHelper.java`
- `com/agilexrobotics/command/CommandManager.java`
- `com/agilexrobotics/command/app/MACarDataManager.java`
- `com/agilexrobotics/map/ManualLawnMowingManager.java`
- `com/agilexrobotics/map/activity/MapManualActivityNew.java`
- `com/agilexrobotics/map/activity/MapManualSimpleActivity.java`
- `com/agilexrobotics/map/activity/MapManualVideoActivity.java`
- `com/agilexrobotics/map/activity/CalibrationActivity.java`
- `com/agilexrobotics/map/fragment/PlanMapLandFragment.java`
- `com/agilexrobotics/map/fragment/mn231/PlanMapLand231Fragment.java`
- `com/agilexrobotics/map/fragment/mn231/PlanMapLandRestPileFragment.java`
- `com/agilexrobotics/device/source/device/enums/DeviceWorkState.java`
- `com/agilexrobotics/work/setting/self_check/SelfCheckFragment.java`
- `com/agilexrobotics/home/fragment/HomeFragmentNew.java`
- all Java files in `com/agilexrobotics/services/touchmode/`.

Generated proto classes searched/cross-referenced:

- `com/agilexrobotics/proto/MctrlDriver.java`
- `com/agilexrobotics/proto/MctrlSys.java`
- `com/agilexrobotics/proto/MctrlNav.java`
- `com/agilexrobotics/proto/LubaMsgOuterClass.java`.

Supporting files searched:

- manual/map remote-control views and fragments;
- `BlockErrorBeanChangeUtils` call sites;
- device type/capability helpers and lock/state beans;
- English/default resource strings for safety, emergency stop, follow, turn, blade, and calibration;
- all dex-derived Java sources via repository-wide `rg` for the command and proto symbols cataloged above.
