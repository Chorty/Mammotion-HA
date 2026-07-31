# Custom path execution research

This document records the architectural research that led to the guarded
click-to-go design. The protocol conclusion remains current: no firmware-level
arbitrary-waypoint upload with a blades-off guarantee was found.

Status updated 2026-07-31: `mammotion.execute_custom_path` remains a non-moving
readiness envelope, but the integration now also has a separate experimental
executor built from bounded raw manual-motion segments. Preview and dry-run
allow seven destinations; real execution is BLE-only, explicitly opted in,
capability-probed, session-exclusive, operator-confirmed, and capped at two
segments. Supervised LUBA tests passed a short straight segment, active abort,
a 176-degree turn, and a two-leg L path. This is **not** firmware autonomous
navigation and does not change the protocol finding below.

## Current foundation

The safe path format is `mower_map_xy`:

```json
{
  "points": [
    { "x": 1.0, "y": 1.0 },
    { "x": 5.0, "y": 5.0 }
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
- mower work mode `MODE_READY`;
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

## Questions and answers that shaped guarded execution

`mammotion.execute_custom_path` was deliberately never turned into the moving
service. The experimental raw executor answered only the narrower questions
needed for a fail-closed LUBA trial:

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

Firmware-level arbitrary custom path execution is still not available or
approved. The narrow experimental alternative is a guarded manual-motion
controller, and it remains disabled by default.

The original safe answer still applies to `execute_custom_path`: it exposes a
blocked readiness service, not an execute button. Movement lives behind the
separate raw vector/multi-segment services and central authorization boundary.

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

Cloud/Wi-Fi manual motion is now rejected as `manual_motion_requires_ble`.
Supervised testing proved the bounded BLE path on one LUBA only. Unknown models,
other mower families, RTK stations, SPINO, charging, active mowing, stale
telemetry, an unverified backend, and an unavailable BLE link all fail closed.

## Approval gate

Do not broaden the accepted scope into firmware path upload, blade control,
cloud/Wi-Fi motion, more than two real segments, or non-LUBA hardware without a
new research and hardware-acceptance cycle.

Dry-run and preview remain the defaults. Every physical run still requires a
fresh operator confirmation, clear mapped area, blades off, safe mower state,
verified backend, live BLE, and a new explicit `go`. The prior operator approval
cannot be reused. See `docs/p0-beta-release.md` for the tested parameters and
`docs/NEXT-SESSION.md` for the unresolved card-default mismatch.
