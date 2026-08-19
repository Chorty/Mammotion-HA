# The five VIO telemetry fields — identified 2026-08-18

All five are already shipped as HA diagnostic sensors on `LUBA_2_YUKA_ONLY_TYPES`
(`sensor.py:206-266`), with translations complete in every language file. None of
them was documented, and one is **captured nowhere in the evidence corpus** —
see the gap at the end.

| sensor key | protocol source | type |
| --- | --- | --- |
| `vio_heading` | `report_data.vision_info.heading` | double, degrees |
| `vio_tracked_features` | `vision_info.track_feature_num` | int32 |
| `vio_detected_features` | `vision_info.detect_feature_num` | int32 |
| `vio_brightness_raw` | `vision_info.brightness` | int32 |
| `vio_survival_distance` | `dev.vio_survival_info.vio_survival_distance` | float, metres |

The first four are fields 3, 7, 6 and 5 of `VioToAppInfoMsg`; the fifth is a
separate one-field message, `VioSurvivalInfoT`.

## What each one is

**`vio_heading`** — the VIO body heading, and **the value the whole turn
controller closes on**. Not course-over-ground: unlike `position.toward` it
tracks in-place rotation, which is why every VIO turn uses it and why turns were
daylight-only before the night branch existed. The map-frame bearing is
`vio_heading + calibrated_forward_heading_offset_degrees`, and that offset is
re-derived from linear travel each segment (`vio.offset_source:
"linear_refresh"`).

**`vio_tracked_features`** — features successfully *tracked across frames*. This
is the liveness signal the executor gates on: below
`_VIO_MIN_TRACKED_FEATURES` the feed is declared degraded and a real run is
refused with `vio_feed_live`, **regardless of what `vio_state` says**.

🔑 **It saturates at 80 and falls off a cliff.** Operator-stated and confirmed
across the corpus: 80 appears 387 times, the rest cluster 68-79, and the night
runs read 0. So 80 means "enough light", not "lots of margin" — it has no useful
dynamic range above sufficiency. Watched live on 2026-08-18 it read 80 at 20:20
and **0 by 21:00**.

**`vio_detected_features`** — features *detected in the current frame*, before
tracking. See the gap below; this is the more diagnostic of the pair.

**`vio_brightness_raw`** — the raw scene-brightness int. The `camera_brightness`
sensor maps it to an enum, and the mapping is odd enough to write down
(`pymammotion/utility/constant/device_constant.py:406`):

    value 0        -> "Dark"
    value 1        -> "Light"
    value > 45     -> "Light"
    anything else  -> "Dark"

So `0` and `1` are treated as *flags*, and any other value as a *level* with a
threshold at 45. A raw `1` therefore means "Light" while a raw `40` means
"Dark" — do not read the raw number as a monotonic brightness.

**`vio_survival_distance`** — distance in metres from `VioSurvivalInfoT`, a
message carrying this single field. ⚠️ **Its meaning is not established.** The
name suggests how far VIO can still navigate on its current map//feature memory,
but nothing in pymammotion, the proto, or this project's measurements confirms
that, and it has read `0.0` on every observation so far. **Treat it as
unidentified in meaning, not merely undocumented.** HA displays it in the user's
unit system, so it shows as `ft` on an imperial install despite the sensor
declaring metres.

## Live snapshot, 2026-08-18 ~21:00 EDT (full dark)

    camera_brightness            dark
    visual_positioning_status    signal_none      (VioState.SIGNAL_NONE = 0)
    vio_heading                  0.0
    vio_tracked_features         0
    vio_detected_features        0
    vio_survival_distance        0.0

`VioState` (`device_constant.py:204`) is sourced from the APK's
`SignalHelper.VioSignalType`: `SIGNAL_NONE 0`, `SIGNAL_INIT 1`, `SIGNAL_GOOD 2`,
`SIGNAL_BAD 3`, anything else `SIGNAL_UNKNOWN -1` (172 has been observed while
the camera pipeline initialises).

## 🚨 The gap: detected features are never recorded

`_vio_feed` (`services.py:7580`) reads **only** `track_feature_num`. Every
`initial_vio_feed` block in every `docs/evidence-*.json` therefore carries
`tracked_features` and no detected count — confirmed by sweeping the corpus:
**zero samples** with `detect_feature_num`.

That matters, because **detected vs tracked separates two different failures**:

- `detected` low → the scene has no texture to see. Darkness, blank grass, fog.
- `detected` high but `tracked` low → the camera sees plenty and *cannot match it
  between frames*. Motion blur, exposure hunting, or a rate problem — a
  fundamentally different fault with a different remedy.

Today only the second number is kept, so every VIO degradation on record is
indistinguishable between those two causes. Adding `detect_feature_num` to
`_vio_feed` is a one-line read that would make the distinction available on
every future run, and would cost nothing at runtime.

⚠️ Not done here. It changes what the executor records on every run, and this
was identified at night with the mower parked — it belongs with the next change
that touches that function, not as a drive-by.
