# Custom path execution research

This document records the current safety position for custom mower paths.

Status: guarded execution envelope only. The integration now includes
`mammotion.execute_custom_path`, but that service is intentionally non-moving:
it performs readiness checks and returns a blocked movement plan. It does not
send mower movement, task, blade, or stop commands.

Current conclusion: as of this pass, Mammotion/pymammotion does not expose a
proven safe "follow these arbitrary waypoints with blades guaranteed off"
command path. Real movement remains blocked in code.

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

## Implemented safe stage

`mammotion.execute_custom_path` now exists as a blocked execution envelope. It:

- validates the path;
- calculates segments, headings, distances, and estimated durations;
- captures live telemetry;
- runs a simulated manual velocity controller decision;
- checks whether blades are reported off;
- checks whether live map-local position is available;
- reports whether manual velocity was explicitly requested;
- always returns `real_execution_allowed: false`;
- sends no mower movement, task, blade, or stop commands.

The service accepts future-facing safety fields such as `dry_run`,
`confirm_blades_off`, and `allow_manual_velocity`, but those fields only affect
the returned readiness report. They do not unlock real movement.

The simulated controller consumes the current telemetry snapshot and path points,
then returns one next action:

- `forward`;
- `turn_left`;
- `turn_right`;
- `stop`.

It also returns the service call it would have used, such as
`mammotion.move_forward`, under `command_not_sent`. This is intentionally only a
decision report; no movement command is called.

`mammotion.manual_velocity_pulse_test` now exists as the first guarded
real-motion probe. It defaults to `dry_run: true`. When explicitly run with
`dry_run: false`, it requires:

- `confirm_blades_off: true`;
- `confirm_clear_area: true`;
- telemetry reporting blades off and cutter RPM zero/unknown;
- live map-local position;
- the internal stop primitive `async_stop_manual_motion()`.

If all gates pass, it sends one tiny low-level movement pulse, then always
attempts the stop primitive and returns before/after telemetry plus measured
movement delta. This is for proving telemetry and stop behavior only; it is not
full path execution.

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

Before enabling real movement from `mammotion.execute_custom_path`, research
must answer:

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
- Does `manual_velocity_pulse_test` prove that position and heading update
  quickly enough while moving?
- Does the zero-speed stop primitive reliably stop both linear and angular
  manual motion on the real mower?

## Current safety assessment

Arbitrary custom path execution is not approved for real movement yet.

The current safe answer is: expose a blocked readiness service, not an execute
button that moves the mower.

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

Do not implement real mower movement, blade control, path upload, or route
execution until a follow-up research pass identifies a concrete command path and
the user explicitly approves implementation.

The future real execution path should default to `dry_run: true` and reject real
movement unless all safety fields are explicitly set.

One-segment and full-path execution remain blocked until the pulse probe has
been tested on the real mower in a clear area.
