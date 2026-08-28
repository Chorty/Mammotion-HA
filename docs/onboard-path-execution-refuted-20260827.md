# Can the mower's own path execution serve click-to-path? — NO. APK ground truth, 2026-08-27

Settles §5 item 1 of `docs/firmware-constraints-from-the-apk-20260827.md`, the
highest-ranked open research question.

Offline reading of the decompiled vendor app at
`/Users/mattjoslin/mammotion-apk-decompile/src`. **No device contact, no motion,
no deploy.** Host state was not queried and nothing was changed.

## The answer

🗑️ **NO. A short two-point route CANNOT be handed to the mower's onboard
controller.** The protocol has no message — on any channel — that is app→device,
carries caller-chosen coordinates, and causes the mower to drive.

**Consequence, stated plainly because it is the point of asking:
stop-measure-go and Phase 2 are the only two options that exist.** There is no
third path where the firmware does the driving for us. The hoped-for escape from
remote closed-loop control over a 1 Hz link is not available.

⚠️ **This does not resurrect the refuted "vendor proves 1 Hz is enough"
argument.** Both things are true at once: the onboard planner is real and much
faster than 1 Hz, *and* we cannot address it. §2 of the firmware-constraints doc
stands unchanged.

## 1. The exhaustive negative

Swept every generated protobuf class the app speaks — all twelve modules under
`com/agilexrobotics/proto/` (nav, driver, sys, pept, ota, basestation, spino,
common, luba_msg, luba_mul, pdt, dev_net) — for any setter matching
`setTarget*`, `setWaypoint*`, `setGoto*`, `setDestination*`, `setMoveTo*`,
`setNavigate*`, `setGoal*`, `setTargetPose`:

```
$ grep -roihE "(setTarget[A-Za-z]*|setWaypoint[A-Za-z]*|setGoto[A-Za-z]*|\
setDestination[A-Za-z]*|setMoveTo[A-Za-z]*|setNavigate[A-Za-z]*|setGoalX|\
setGoalY|setTargetPose)" com/agilexrobotics/proto/ | sort -u
(no output)
```

**Zero hits.** No point-to-point navigation field exists anywhere in the
protocol, in any module, for any device model.

## 2. The driver channel is joystick-only — verified in full, not sampled

`mctrl_driver.proto` read end to end. `MctlDriver` is a 14-member oneof and
exactly **one** member is a motion command:

```protobuf
message DrvMotionCtrl {
  int32 setLinearSpeed  = 1;
  int32 setAngularSpeed = 2;
}
```

The other thirteen are blades, knife height, cutter mode, collect/unload motors,
speed read/set, and RTK config. **There is no position, waypoint, pose or target
command on the driver channel.** This confirms the firmware-constraints doc's
claim by exhaustion rather than by example.

The app's own send site is a bare joystick write
(`command/app/MACommandApiHelper.java`, `sendControl(int, int)`):

```java
sendOrderMsg_Driver(MctrlDriver.MctlDriver.newBuilder().setTodevDevmotionCtrl(
  MctrlDriver.DrvMotionCtrl.newBuilder().setSetLinearSpeed(i3)
    .setSetAngularSpeed(i4).build()).build(), 51, false, ...)
```

That is the same primitive this project already drives on.

## 3. Census: every coordinate-bearing message, with its real direction

The decisive test is not "does a message with x/y exist" — many do — but "does
the **app** ever send one, populated with coordinates **it** chose".

Swept the entire decompiled app for point-list pushes:

```
$ grep -rn "addAllDataCouple\|addDataCouple" --include="*.java" . | grep -v "proto/"
com/agilexrobotics/command/app/MACommandApiHelper.java:1480:  ... setToappManualElement(...)
```

**Exactly one hit in the whole application.** Everything else that carries
coordinates flows device→app, or is never sent at all.

Send-site count per candidate, app-side only (generated proto excluded):

| Message | app-side send sites | verdict |
| --- | --- | --- |
| `NavCHlLineData` (todev_chl_line_data) | **0** | schema only — never sent |
| `zone_start_precent_t` | **0** | schema only — never sent |
| `NavTaskBreakPoint` (toapp_bp) | **0** | device→app; app cannot set one |
| `cover_path_upload_t` | **0** | device→app (mower uploads its plan) |
| `NavUploadZigZagResult` | **0** | device→app |
| `costmap_t`, `chargePileType` | **0** | device→app |
| `NavEdgePoints` | **0** | device→app (only its *Ack* is sent, no points) |
| `ManualElementMessage` | 2 | **app→device, carries points** — see §4 |

🚨 **Three of the most promising leads die here, and they die the same way.**
`NavCHlLineData` carries `repeated CommDataCouple dc` and is named `todev_` —
it looks exactly like "hand the mower a list of points". It appears **only**
inside the generated `MctrlNav.java` (162 occurrences, all generated code) and
in **zero** application files. Same for `zone_start_precent_t` (x, y, index) and
`NavTaskBreakPoint` (x, y, toward). ⚠️ **A `todev_` prefix is not evidence the
app sends a message.** Anyone re-checking this should count send sites, not
field names.

## 4. The one coordinate push that IS real — and why it does not help

`ManualElementMessage` genuinely goes app→device carrying a polygon
(`MACommandApiHelper.sendDate(ElementMessageBean)`), converting screen
coordinates to protobuf points:

```java
for (int i3 = 0; i3 < listPointS.size(); i3++) {
    ScreenCoordinate screenCoordinate = listPointS.get(i3);
    arrayList.add(Common.CommDataCouple.newBuilder()
        .setX((float) screenCoordinate.getX())
        .setY((float) screenCoordinate.getY()).build());
}
```

So arbitrary geometry **can** be pushed to the device. But its type enum
(`base_module/entity/ElementMessageBean.java`) is exhaustively:

```java
public enum ElementMessageType { VirtualWall, RestrictedZone, SecurityZone; }
public enum ElementMessageShape { Circle, Rect, LineSegment; }
```

**Keep-out geometry only.** There is no mowable-area type and no route type. The
callers confirm it — `showRestrictedZoneRect`, `showSecurityZoneRect`,
`showVirtualWallView` in the map-editing fragment, plus a manually placed
越障点 ("obstacle-crossing point") dropped 0.3 m in front of the mower. Creating
one of these tells the mower where **not** to go. It commands no motion.

🔑 **This kills the "author a tiny zone and mow it" workaround too.** A mowable
area cannot be created by pushing points — see §6.

## 5. NavTaskCtrl carries no coordinates at all

`NavTaskCtrl` is `{type, action, result, reserved}` — the app's primary
app→device nav command, and it is a bare verb with no payload. Its action codes,
read off the Chinese log strings at every send site:

| action | log string | English |
| --- | --- | --- |
| 2 | 暂停指令 | pause |
| 3 | 取消暂停指令 | cancel pause |
| 4 | 结束作业 | end job |
| 5 | 回充指令 | return to charge |
| 7 | 从断点继续作业 | resume work from breakpoint |
| 9 | 从车当前位置继续作业 | continue work from vehicle's current position |
| 10 | 回充测试指令 | recharge test |
| 16 | x5机型无图作业指令 | X5 map-less job |
| 17 | 走廊录制,结束走廊录制 | corridor recording / end corridor recording |
| 18 | 放弃地图保存 | abandon map save |
| 19 | 更新地图 | update map |
| type 3 / 1 | 重置充电桩，基站位置 | reset dock / base position |

The two that sound most like "just drive" are both destination-free:
**action 9** resumes an *existing* job from wherever the mower is, and
**action 16** mows with no map at all. Neither takes a target. Nothing in this
command space accepts a coordinate, because the message has no field for one.

## 6. 🔑 The clincher: the vendor's own answer to "get from A to B" is the joystick

The strongest evidence is not in the protocol, it is in the app's user-facing
instruction text. Verbatim from `resources/res/values/strings.xml`:

* `cw_channecl_info_start` — *"[Channel mapping] Control the robot to a mapped
  lawn or near the charging station to set the channel start point"*
* `cw_channecl_info_end` — *"[Channel mapping] Control the robot to another lawn
  or near the charging station as the channel end point"*
* `cw_channel_planning` — *"Please control robot to another task area"*
* `cw_border_planning` — *"Please control robot to map the lawn along its
  perimeter"*
* `cw_border_ready` — *"Please manually control robot to the perimeter of the
  target lawn first, then tap [icon] to map a task area"*
* `cw_redraw_border` — *"Control your robot to draw a new boundary"*

**Both areas and channels are created by driving the mower manually.** When the
vendor's own app needs the machine to travel to a chosen point, it does not send
a command — **it asks the human to joystick it there.** A vendor that had a
"drive to this coordinate" primitive would not write these strings.

This also explains the `NavGetCommData` action verbs, which are recording verbs
throughout: 录制 / 取消当前录制 / 结束画边界 (record / cancel current recording /
end drawing border). That channel pulls frames and controls a recording session;
it never pushes geometry.

## 7. The onboard planner is real — and unreachable

⚠️ **Do not read this as "the firmware cannot navigate".** It plainly can.
Return-to-dock (`NavTaskCtrl` type 1 action 5) and one-touch leave-dock
(`todev_one_touch_leave_pile(1)`) both send **no path whatsoever** and the mower
crosses the yard on its own. Coverage of a zone is planned onboard and uploaded
to the app for display, never the reverse.

So an onboard point-to-point planner exists and runs at the firmware's internal
rate. **Its destination is simply not parameterizable.** Both commands are bare
integers. The planner has exactly two addressable destinations — the dock, and
"wherever the coverage plan for these zone hashes says" — and neither is a point
we choose.

That is the whole finding in one line: **the capability exists, the API does
not.**

## 8. The path-upload residual — RESEARCHED AND RETIRED, same day

An earlier draft of this doc left one avenue open: `cover_path_upload_t` sits in
the shared `MctlNav` oneof with no schema-level direction enforcement, so an
app-originated path was *untested* rather than disproved. It recommended against
testing it on hardware because the failure mode is corrupting the mower's
navigation store.

🗑️ **That residual is now CLOSED, and it was closed offline.** The question was
attacked from the wrong end. Whether the firmware would *accept* an inbound path
is indeed unknowable without the device — but it does not matter, because a
**second, independent gate blocks execution even if the first one opened.**

### 8.1 There is no command that names a path to execute

`NavReqCoverPath` has a `pathHash` field (14). **The app never sets it.** Swept
every `setPathHash` call in the decompiled app:

```
$ grep -rn "setPathHash" --include="*.java" . | grep -v "proto/MctrlNav.java"
com/agilexrobotics/proto/MctrlSys.java:31455:   setPathHash(sysWorkState.getPathHash());
com/agilexrobotics/proto/MctrlSys.java:65498:   setPathHash(rpt_workVar.getPathHash());
com/agilexrobotics/device/.../CarWorkingStateMachineBean.java:292
com/agilexrobotics/base_module/event/SlowPackageEvent.java:90
```

**Every hit is on the system/telemetry side or an app-internal bean — none is on
a nav command.** And both protobuf hits are device→app reports:

```protobuf
message SysWorkState { int32 deviceState = 1; int32 chargeState = 2;
                       int64 cmHash = 3; int64 pathHash = 4; }
message rpt_work    { int32 plan = 1; int64 path_hash = 2; int32 progress = 3;
                      ... int32 path_pos_x = 10; int32 path_pos_y = 11; }
```

🔑 **`path_hash` is STATUS, not a selector.** It is the mower telling the app
*which path it is currently executing* — alongside its progress and its position
along that path. It never travels the other way.

`NavSysHashOverview.pathHashOverview` — the checksum of the device-side path
store — likewise has **zero** app-side references. The mower owns the path store
outright.

### 8.2 Jobs are selected by plan identity, never by geometry

The only "execute now" command is `nav_plan_task_execute`, and it takes a
**string id**:

```java
public void singleSchedule(String str) {
    MctrlNav.nav_plan_task_execute b = MctrlNav.nav_plan_task_execute
        .newBuilder().setSubCmd(1).setId(str).build();
    ...  "发送指令--立即执行作业计划指令 cmd=1"     // execute work plan immediately
}
```

A plan is `NavPlanJobSet`, whose spatial content is `repeated fixed64 zoneHashs`.
So the full chain is:

```
app names a plan id  ->  plan names zone hashes  ->  DEVICE plans the path
                     ->  device stores it under a path_hash it chose
                     ->  device executes it, reporting path_hash + progress
```

**There is no point in that chain where app-supplied geometry can enter.** The
app's only influence on the path is *which zones* and *route parameters* (angle,
spacing, speed, edge mode) — and route generation itself is a device-side
operation the app merely triggers (`NavReqCoverPath` subCmd 0 = generate,
3 = modify params, 9 = end generation, 2 = query config).

### 8.3 Why this retires the risk, not just the question

Even granting the most optimistic reading of the first gate — the firmware
accepts an inbound path frame and stores it — the upload would be:

1. written into a store the device **recomputes** whenever a route is generated;
2. keyed to a `zone_hash` that must already exist as a real mapped zone; and
3. reachable only by starting a job that covers **that entire zone**.

A two-point route is not representable as a zone, and nothing can aim a job at a
specific path. **So the hazardous experiment could not pay off even if it
worked.** That is a strictly better outcome than leaving it open: the residual is
retired on evidence rather than on caution.

⚠️ **What remains genuinely unknown, stated so nobody re-opens it as new:**
whether the firmware would accept or reject an inbound `cover_path_upload`. That
is still unanswerable offline. It is now **moot**, not pending — and it is not a
reason to run the experiment.

## 9. What this closes and what it does not

🗑️ **CLOSED — do not spend further sessions on it:** "hand a route to the
onboard controller" as a route to click-to-path. §5 item 1 of the
firmware-constraints doc is answered: **no.**

**Unchanged and still open:** §5 item 2 (is the internal position rate
observable?) and item 3 (does `no_change_period = 4000` matter?). Neither is
affected by this result.

**Unchanged:** the Phase 2 steering sign has still never moved a wheel. This
finding removes an alternative to that test; it does not remove the test. The
attempt-4 design and its predeclared criteria stand exactly as they were.
