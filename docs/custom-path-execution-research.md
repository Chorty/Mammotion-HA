# Custom path execution research

This document records the current safety position for custom mower paths.

Status: research only. No custom path execution service has been implemented.
The current implementation is limited to read-only map export, task export,
custom path validation, and custom path preview.

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

Manual movement is not enough by itself because it would require a controller
loop, position feedback, stop conditions, obstacle handling, transport latency
handling, and explicit blade-state proof.

Task/schedule execution is not safe as a custom path execution path until we
prove the firmware supports a navigation-only task mode or another route mode
that cannot spin blades.

SVG/map-object commands are useful for map-local coordinate handling, but they
do not execute mower motion.

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
