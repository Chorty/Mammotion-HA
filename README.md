# Mammotion - Home Assistant Integration [![Discord](https://img.shields.io/discord/1247286396297678879)](https://discord.gg/vpZdWhJX8x)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=mikey0000&repository=mammotion-HA&category=Integration)

💬 [Join us on Discord](https://discord.gg/vpZdWhJX8x)

This integration allows you to control and monitor Mammotion products, e.g robot lawn mowers using Home Assistant.

⚠️ **Please note:** This integration is still a work in progress. You may encounter unfinished features or bugs. If you come across any issues, please open an issue on the GitHub repository. 🐛

## Roadmap 🗺️

- [x] Bluetooth (BLE) support
- [x] Wi-Fi support (Including SIM 3G/4G)
- [x] Camera stream
- [ ] Scheduling
- [ ] Mapping and zone management
- [x] Maps
- [x] Firmware updates
- [x] Automations
- [ ] More...

## Features ✨

- Start, stop, pause, and dock the mower
- Monitor the mower's status (e.g., mowing, charging, idle)
- View the mower's battery level
- Start a mow based on configuration
- Start an existing scheduled task/s
- More features being added all the time!

- Supports Spino pool cleaners

## Prerequisites 📋

> [!WARNING]
> **Home Assistant Minimum Version 2026.1.0**

- A second account with your mower/s shared to it for using Wi-Fi (If you use your primary accouunt it will log you out of your mobile app)
- (Optional)[Bluetooth proxy for Home Assistant](https://esphome.io/components/bluetooth_proxy.html)

## Troubleshooting

- Sometimes using the account number works instead of email address when adding via discovery (not sure why)

- Connection timeout to host https://api.link.aliyun.com/living/account/region/get - unblock china

## Installation 🛠️

This integration can be installed using [HACS](https://hacs.xyz/)

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg?style=for-the-badge)](https://github.com/hacs/integration)

This integration is not available in the default HACS store. You will need to add it as a custom repository.

1. Go to HACS > Integrations and click on the 3 dots in the top right corner.
2. Select "Custom repositories".
3. In the "Repository" field, paste this URL: `https://github.com/mikey0000/Mammotion-HA`
4. For "Category", select "Integration".
5. Click "Add".
6. You can now search for "Mammotion" within HACS and install it.
7. After installation, restart Home Assistant.
8. Go to **Settings > Devices & Services** and click **+ Add Integration** to configure Mammotion.

## Usage 🎮

### Getting Started

See the wiki for how to [get started](https://github.com/mikey0000/Mammotion-HA/wiki/Getting-Started)

Once the integration is set up, you can control and monitor your Mammotion mower using Home Assistant. 🎉

### Guarded Click/Go Smoke Script

For guarded one-segment click/go validation from the command line, use:

```bash
.venv/bin/python scripts/mammotion_click_go_smoke.py lawn_mower.your_mower_entity
```

What this does by default:

- Waits for live runtime position via `mammotion.export_runtime_state`
- Builds one segment from current position to a nearby target offset
- Runs read-only preview via `mammotion.preview_custom_path`
- Runs guarded non-moving dry-run via `mammotion.raw_pymammotion_execute_vector_segment`
- Writes artifacts under `/tmp/mammotion_click_go_smoke/<timestamp>/`

To run a guarded real one-segment smoke step (movement-producing), explicit confirmations are required:

```bash
.venv/bin/python scripts/mammotion_click_go_smoke.py lawn_mower.your_mower_entity \
	--run-real --confirm-blades-off --confirm-clear-area
```

Useful options:

- `--target-x` and `--target-y` to use an explicit map-local target point
- `--offset-x` and `--offset-y` to use runtime-relative target generation (default)
- `--max-turn-commands` and `--max-linear-commands` to keep command caps conservative
- `--sample-delays` to control telemetry sample timing

### Experimental Click-to-Go Card

Click-to-go is an operator-supervised experiment, not an autonomous navigation
feature. Preview and dry-run support up to seven destinations. Real execution
is limited to four segments and is disabled unless every backend safety gate
passes.

The bounded backend re-passed the two-leg Gate 4 path on 2026-08-05. Gate 5 then
passed twice from the card: two two-segment runs on 2026-08-08 and a
four-segment reach-enabled run on 2026-08-12. The latter landed 0.0674 / 0.1032 /
0.0807 / 0.0607 m from its waypoints using the built-in profile below. This is
supervised click-to-go acceptance on the tested LUBA, not autonomous navigation
or acceptance for another mower family.

Overriding any motion field in the card YAML leaves the accepted profile. The
card then labels its **execution profile** row `customised (not
hardware-accepted)` and names the overridden fields. Treat that state as
untested.

The mower marker draws a heading arrow only when the backend supplies an
explicitly trustworthy, map-aligned `current_orientation`. Otherwise the card
reports current orientation as unavailable and shows any last-travel projection
as diagnostics only. It never presents that projection as mower orientation.

> `toward` is a body-heading observation on the acceptance LUBA: one commanded
> reverse pulse travelled 0.4185 m almost exactly opposite the heading derived
> from `toward`, while `toward` itself did not flip. Its map conversion is a
> mirror, approximately `(90.13 - toward) mod 360` on this installation. The
> backend does not yet expose that as trusted card orientation, so Nudge remains
> fail-closed rather than treating a mower-specific measurement as universal.

Add the integration-served JavaScript as a dashboard resource:

```text
/mammotion/mammotion-custom-path-card.js?v=<installed-version>
```

Use resource type `JavaScript module`. The version query is required because
Home Assistant serves integration static files with cache headers; update it to
the installed release version after every upgrade.

Minimal card YAML. Every motion field defaults to the accepted profile below,
so omitting them is the supported configuration:

```yaml
type: custom:mammotion-custom-path-card
entity: lawn_mower.your_mower
card_height: 520
```

Click or drag waypoints for coarse placement. When a short or exact segment is
needed, edit that waypoint's X/Y fields below the map; coordinates use mower-map
metres with 0.001 m input precision. Every edit clears prior run results and
automatically re-runs Preview, so run Dry-run again after the final edit.

The built-in defaults, written out. These are the values the supervised LUBA
acceptance run executed; listing them explicitly changes nothing, and changing
any of them puts the card outside the accepted profile:

```yaml
type: custom:mammotion-custom-path-card
entity: lawn_mower.your_mower
card_height: 520
speed: 0.2
prefer_ble: true
turn_mode: vio
max_turn_commands: 4
vio_turn_max_commands: 4
max_linear_commands: 3
max_linear_pulse_ceiling: 14
max_no_progress_pulses: 3
heading_tolerance_degrees: 18
waypoint_tolerance: 0.15
min_progress_distance: 0.0025
calibrated_forward_heading_offset_degrees: 102.4
turn_pulse_duration_ms: 1500
linear_pulse_duration_ms: 1300
max_turn_translation_distance: 0.3
motion_refresh_interval_ms: 200
final_approach_metres_per_pulse: 1.06
turn_degrees_per_second: 37
ble_auto_recover: false
sample_delays:
  - 0
  - 3
```

Notes on the profile:

- `max_linear_pulse_ceiling: 14` enables **loop-to-tolerance**, and it is what
  makes per-click reach usable. Without it a segment stops after three linear
  commands at roughly 1 m; four measured legs stopped 0.68 / 0.68 / 1.79 /
  2.95 m short. With it, 2 m / 3 m / 4 m legs reached target at 0.0690 /
  0.0928 / 0.1023 m, every one stopping on tolerance rather than on the
  ceiling. Per-segment reach is ~4 m, per-click ~16 m at four segments.
  Adopted and card-accepted by the 2026-08-12 Gate 5 re-pass.
  `max_linear_commands` stays at 3 so disabling the ceiling falls back to the
  fixed-budget behaviour.
- A runaway is bounded by distance, not by the pulse count:
  `linear_distance_ceiling_factor` (2.0) stops a segment at twice its leg
  length regardless of how many pulses it has left.
- `calibrated_forward_heading_offset_degrees` is a per-mower measurement taken
  on the acceptance LUBA. Re-derive it for a different mower instead of
  assuming 102.4 transfers.
- `ble_auto_recover: false` keeps a failed BLE gate a fast failure rather than
  a ~90 s in-run recovery attempt.
- The Real Go throughput fix preserves this exact payload and every mandatory
  stop. It removes additive feedback waits by using the VIO position-settle
  result as the post-pulse sample, reduces cold VIO calibration from a
  2+4-second wait to one four-second window, and removes a duplicate final card
  reload. After it, each pulse also runs a bounded BLE queue-settle check and
  refuses the next command with `ble_link_not_ready_after_feedback` rather than
  dispatching into a backlog. One supervised 0.70 m run reached its target in
  19.2 s with 0.093100 m landing error, all three queue-settle records live at
  depth zero. That is a single path, not a reliability population. See
  `docs/real-go-throughput-hardware-20260814.md`.
- `heading_tolerance_degrees: 18` is a known-loose value carried over from the
  July 18 calibration. It is unchanged from the accepted run, but reducing it
  is open beta work.

#### Nudge — requires trustworthy current orientation

**Nudge** is a bounded straight-line helper, but it is fail-closed unless the
backend supplies a trustworthy, map-aligned current orientation.

The earlier implementation treated `toward + calibrated offset` as current
facing. Live testing disproved that additive conversion. Later reverse evidence
established `toward` as body heading on this LUBA and found a mirror conversion
for this map, but the integration deliberately does not promote a
mower-specific measurement into a universal trusted-orientation contract.
Nudge therefore remains fail-closed.

- Capped at **2 m**, so a mistake is bounded by geometry rather than vigilance.
- Requires the **clear area** confirmation. The blades-off checkbox is not
  required, because a nudge is a blades-off manoeuvre and the mower's own state
  and cutter RPM are still gated separately by `mower_reports_blades_off`.
- Every other gate is unchanged: BLE liveness, stop primitive, containment,
  ready state.
- Unavailable when only last-travel projection or unqualified `toward` telemetry
  exists. Neither is exposed as portable trusted orientation.
- Leaves the accepted profile, so the card labels the run accordingly.

> ⚠️ Do not override the orientation blocker with last-travel projection. The
> measured body-heading conversion is installation-specific and is not yet a
> general card contract.

#### Experimental Night Go

The card exposes **Night dry-run** and **Night Go** as a separate experimental
control. It does not change the accepted Real Go profile: Real Go remains
`turn_mode: vio`, while Night Go constructs its own `turn_mode: night` payload.

Night Go is deliberately restricted to the backend's measured v1 envelope:

- exactly one forward-only mapped segment, at most **1.0 m**;
- **RTK Fix** with no degraded-RTK override;
- angular speed 500 and 200 ms command refresh;
- an 8° opening-heading tolerance and a fixed three-command linear budget;
- no loop-to-tolerance ceiling, reverse recovery, mid-drive correction, or
  multi-segment chaining;
- the same blades-off and clear-area confirmations and backend motion gates as
  Real Go, plus a separate acknowledgement that night accuracy is unproven.

This is a characterization control, not accepted navigation. The first 0.7 m
night run stopped 0.1143 m from its target after three pulses, so night landing
accuracy and a supported tolerance remain unproven. Use **Night dry-run** first,
watch the mower continuously, and retain the complete result JSON from every
physical run.

One supervised 0.70 m harness segment reached its opening 8° turn tolerance and
stopped safely on `no_target_progress` 0.1143 m from target after three forward
pulses. A later beta54 card-driven 0.739 m run also stopped safely after three
turn and three forward commands, 0.1171 m from target. In that run the second
forward pulse was already 0.0827 m away, but a stale pre-pulse target bearing
allowed a third pulse after the mower crossed the target. See
`docs/night-go-card-beta54-20260814.md`.

The follow-up recomputes that night-only residual bearing from the settled
post-pulse RTK position, so the existing reverse-recovery refusal stops a
crossed waypoint, and sends `sample_delays: [0, 3]` from Night Go to avoid
beta54's inherited minute-long diagnostic waits. Replayed against the recorded
beta54 geometry it refuses the third pulse and lands 0.0827 m out instead of
0.1171 m — still outside the configured 0.08 m tolerance, so it stops on
`target_requires_reverse_recovery` rather than reaching target. It is verified
off-mower and installed motion-disabled, but **no night run has exercised it on
hardware**. These runs record feasibility; they do not establish a night
accuracy tolerance or acceptance population. Use the accepted VIO card profile
for ordinary click-to-go operation.

Real motion additionally requires the integration option **Enable experimental
BLE-only manual motion**, a positively verified PyMammotion backend, a fresh
and idle LUBA, live BLE/queue evidence, blades off, a clear mapped area, and
both per-run confirmations. This branch pins the capability-probed Chorty
PyMammotion `0.8.12.post1` wheel because no official upstream release contains
the remaining BLE teardown fix. Upstream `0.8.12` by itself stays locked out.
`movement_use_wifi` is retained only for option migration and cannot bypass
the BLE-only gate.

Abort calls `mammotion.stop_manual_motion`. It marks the backend session
cancelled before issuing a bounded confirmed zero-velocity sequence, so the
aborted owner cannot later replay a nonzero command.

To roll back, first disable **Enable experimental BLE-only manual motion**,
then restore the prior integration release in HACS and restart Home Assistant.

## Map Position Offset

Satellite map tiles (Google Maps, OpenStreetMap, etc.) are sometimes misaligned relative to RTK GPS coordinates by several metres. Each mower exposes two number entities to correct this:

- **Map offset latitude** — shifts the mower pin north (positive) or south (negative), in metres
- **Map offset longitude** — shifts the mower pin east (positive) or west (negative), in metres

**How to calibrate:**

1. Add a [Map card](https://www.home-assistant.io/dashboards/map/) and both offset entities to a Lovelace dashboard.
2. Start the mower so it is moving at a known location you can identify on satellite imagery.
3. Adjust **Map offset latitude** and **Map offset longitude** until the pin aligns with the mower's real position on the satellite layer.
4. Values are saved automatically and survive restarts.

Typical offsets are within ±20 m. Positive latitude = north, positive longitude = east.

## Dashboard Plugins

Companion HACS dashboard plugins that extend the Mammotion integration with visual tools.

### Mammotion Assets

Images and scripts for displaying Mammotion mowers on a map in Home Assistant — mower card backgrounds, side-profile images, map icons, RTK/dock assets, and the `geojson.js` script that renders mowing areas with labels.

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=mikey0000&repository=ha-mammotion-assets&category=plugin)

### Mammotion GeoJSON Map Plugin

A Lovelace resource that renders GeoJSON mowing areas on the map with area names and zone labels.

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=mikey0000&repository=ha-mammotion-geojson-map-plugin&category=plugin)

### Mammotion SVG Pick and Place

An interactive Lovelace card for placing, editing, and deleting SVG pattern tiles on your mower's map directly from the dashboard. Load an SVG, drag it into position, scale and rotate it, then send it to the device in one click via the `mammotion.svg_add` service.

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=mikey0000&repository=ha-mammotion-svg-pick-n-place&category=plugin)

## Troubleshooting 🔧

If you encounter any issues with the Mammotion integration, please check the Home Assistant logs for error messages. You can also try the following troubleshooting steps:

- Verify that you have Bluetooth proxy setup with Home Assistant.
- Ensure that your mower is connected to your home network and accessible from Home Assistant.
- Restart Home Assistant and check if the issue persists.
- Make sure your not blocking China (Connection timeout to host https://api.link.aliyun.com/living/account/region/get)

## Contributing to Translations

We use Crowdin to manage our translations. If you'd like to contribute:

1. Visit our [Crowdin project page](https://crowdin.com/project/mammotion-ha)
2. Select the language you'd like to translate to
3. Start translating!

Your contributions will be automatically submitted as pull requests to this repository.

## PyMammotion Library 📚

This integration uses the [PyMammotion library](https://github.com/mikey0000/PyMammotion) to communicate with Mammotion mowers. PyMammotion provides a Python API for controlling and monitoring Mammotion robot mowers via MQTT, Cloud, and Bluetooth.

If the problem continues, please file an issue on the GitHub repository for further assistance. 🙏

## Support me

<a href='https://ko-fi.com/DenimJackRabbit' target='_blank'><img height='46' style='border:0px;height:46px;' src='https://az743702.vo.msecnd.net/cdn/kofi3.png?v=0' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

### Referral Links

[Buy a Mammotion Lawn Mower (Amazon)](https://amzn.to/4cOLULU)
[Buy a Mammotion Lawn Mower (Mammotion)](https://mammotion.com/?ref=denimjackrabbit)

## Credits 👥

[![Contributors](https://contrib.rocks/image?repo=mikey0000/Mammotion-HA)](https://github.com/mikey0000/Mammotion-HA/graphs/contributors)
