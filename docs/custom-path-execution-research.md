# Custom path execution research

This document records the current safety position for custom mower paths.

Status: research only. No custom path execution service has been implemented.
The current implementation is limited to read-only map export, task export,
custom path validation, and custom path preview.

Current conclusion: as of this pass, Mammotion/pymammotion does not expose a
proven safe "follow these arbitrary waypoints with blades guaranteed off"
command path. Do not add an execute button or execution service yet.

## Current foundation

The safe path format is `mower_map_xy`:

```json
{
  "points": [
    {"x": 1.0, "y": 1.0},
    {"x": 5.0, "y": 5.0}
  ],
  "area_hash": 123456,
  "speed": 0.2,
  "blade_mode": "off"
}
```

This coordinate system matches the Mammotion map-local coordinates already used
by map geometry and SVG placement. It is intentionally not GPS latitude/longitude.

## Known command paths

The integration already has command paths that can move or control the mower:

- manual movement commands (`move_forward`, `move_left`, `move_right`,
  `move_backward`);
- task/schedule commands (`start_task`, task create/edit/enable/delete);
- SVG/map-object commands (`svg_add`, `svg_update`, `svg_delete`).

None of these are currently proven to be a safe arbitrary waypoint-following
API with guaranteed blades-off behavior.

Manual movement is not enough by itself. In pymammotion this is
`DrvMotionCtrl(setLinearSpeed, setAngularSpeed)`, sent by
`MessageDriver.send_movement()`. It can command velocity, but a custom path
would require a new controller loop, position feedback, stop conditions,
obstacle handling, transport latency handling, and explicit blade-state proof.
That is not firmware-level waypoint following.

Task/schedule execution is not safe as a custom path execution path until we
prove the firmware supports a navigation-only task mode or another route mode
that cannot spin blades.

SVG/map-object commands are useful for map-local coordinate handling, but they
do not execute mower motion.

## Pymammotion/protobuf findings

The route-planning path in pymammotion is area based, not arbitrary waypoint
based:

- `MessageNavigation.generate_route_information()` sends
  `MctlNav.bidire_reqconver_path = NavReqCoverPath(...)`.
- `NavReqCoverPath` contains route settings such as `jobMode`, `edgeMode`,
  `knifeHeight`, `speed`, `channelWidth`, `channelMode`, `toward`, and repeated
  `zoneHashs`.
- `NavReqCoverPath` does not contain a repeated waypoint/point list supplied by
  the app.
- `MowPathSaga` asks the mower to generate or report a cover path and then
  fetches `cover_path_upload` frames. This is useful for reading/generated path
  visibility, but it is not an app-to-mower custom route upload API.

The task execution path is also not arbitrary waypoint based:

- `MessageNavigation.start_job()` sends `NavTaskCtrl(type=1, action=1)`.
- `MessageNavigation.single_schedule(plan_id)` sends
  `NavPlanTaskExecute(sub_cmd=1, id=plan_id)`.
- `lawn_mower.async_start_mowing(..., plan_only=True)` is safe because it plans
  but intentionally skips `start_job`.
- Once `start_job` is sent, execution is the normal device task/mowing path; no
  inspected field proves blades are guaranteed off for an arbitrary custom path.

Blade control exists, but does not prove safe route execution:

- Non-Luba1 blade control uses
  `DrvMowCtrlByHand(main_ctrl, cut_knife_ctrl, cut_knife_height,
  max_run_speed)`.
- Luba1 blade control uses `set_blade_control(on_off=0/1)`.
- Turning blades off before a job is not equivalent to proving the firmware will
  keep blades off after a later task-start command.

There is a Yuka-specific mode byte derived from `OperationSettings.is_mow`,
`is_dump`, and `is_edge`. When `is_mow=False`, `create_path_order()` encodes a
different mode value. This is promising for future research, but it still feeds
the same area-based route generation/task execution path. It is not proof of
safe arbitrary waypoint following.

`NavTaskBreakPoint` and `zone_start_precent_t` include x/y fields, but the
protobuf marks them as report/ack style messages (`toapp_bp`,
`zone_start_precent`) and pymammotion does not implement a command builder that
uses them as arbitrary target waypoints.

## APK string-scan findings

The local Mammotion `2.3.8.19` XAPK was checked with a lightweight string scan.
That found UI strings for manual mowing, zigzag paths, adaptive zigzag paths,
and "customized path" wording, but did not reveal an obvious app-to-mower
custom waypoint upload command. This scan is not as strong as a JADX decompile;
it only supports the current pymammotion/protobuf conclusion.

## Questions that must be answered before execution

Before adding `mammotion.execute_custom_path`, research must answer:

- Can the mower follow arbitrary waypoints, or only stored plans/areas?
- Is there a firmware-supported navigation-only mode?
- Can blades be explicitly commanded off and independently verified off before
  movement?
- Which mower states permit safe movement: idle, ready, paused, docked,
  charging, error, lost-position, or active mowing?
- Which transport is safest for movement commands: BLE, Wi-Fi/local, MQTT,
  Mammotion cloud, or Aliyun?
- What telemetry proves that the mower is still localized and following the
  intended path?
- What stop command is available, and does it work across all selected
  transports?

## Current safety assessment

Arbitrary custom path execution is not approved yet.

The current safe answer is: no execute button/service.

The safest likely execution path, if firmware support exists, would be:

1. validate the path against known area geometry;
2. require `blade_mode: off`;
3. require a separate user confirmation field such as
   `confirm_blades_off: true`;
4. reject active mowing/task states;
5. reject unknown/lost position;
6. use a low default speed, currently `0.2`;
7. start in `dry_run: true`;
8. continuously monitor position, command failures, and transport state;
9. stop immediately on invalid state or command failure.

Cloud-only movement should be treated as high risk because latency and command
delivery guarantees are weaker than local/BLE paths. Local Wi-Fi or BLE may be
safer, but this must be proven against pymammotion/APK/protobuf behavior before
implementation.

## Approval gate

Do not implement mower movement, blade control, path upload, or route execution
until a follow-up research pass identifies a concrete command path and the user
explicitly approves implementation.

The future execution service, if approved, should default to `dry_run: true` and
reject real movement unless all safety fields are explicitly set.
