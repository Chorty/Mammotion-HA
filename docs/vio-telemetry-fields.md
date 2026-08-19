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

**`vio_survival_distance`** — 🔑 **IDENTIFIED 2026-08-18 from the APK
decompile.** It is **how far the mower can still navigate on vision alone**, and
the app shows it only while the mower is actually doing so.

In `CarStatusBarUtil.java` the status bar branches on `fuseStatus`, and this
value appears in exactly one branch:

```java
} else if (fuseStatus == 2 || fuseStatus == 3) {
    ...ivStatusBarPos.setImageResource(R.drawable.img_status_bar_vision);
    ...pbStatusBarVision.setProgress(dis);
    ...tvStatusBarVisionDis.setText(tranMetricUnit(vioSurvivalDistance));
```

`fuseStatus` `-1` and `1` draw the RTK/position icons; **`2` and `3` draw the
*vision* icon**, and only then does the app reveal a vision panel carrying a
progress bar plus this distance as text. The bar is coloured by a separate 0-100
value `dis` — green at 50+, **orange 20-49, red below 20** — so the app treats a
falling vision reserve as a warning state.

That settles the name: "survival" is how much navigating the mower has left on
vision before it loses its reference, surfaced when vision is carrying the
position fix rather than RTK.

⚠️ **It reads `0.0` here and that is expected, not a fault.** This mower runs on
RTK Fix, so `fuse_status` is never in the 2/3 vision-fused regime — the same
field this project already measured as `0` (`NO_POSE`) in all 81 records of the
2026-08-13 night capture. The counter has no reason to be populated.

⚠️ Two things still NOT established: the units the device reports (the proto says
float, our sensor declares metres, and HA re-displays in the user's unit system —
so an imperial install shows `ft`), and whether the value counts down during
vision navigation or is a static capability figure. Neither can be settled
without observing the mower in a vision-fused state, which has not happened.

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

## Does the long-leg reach test need VIO? Yes — unavoidably

Asked 2026-08-18. The answer is **yes**, for two independent reasons in the
code, so the banked characterization run cannot be done after dark.

**1. The re-aim block is inside `if turn_mode == "vio"`.** The mid-drive
correction — the thing under test — sits in that branch of the linear loop.
`legacy` and `night` never reach it at all.

**2. Within that branch it additionally requires a live VIO track**
(`reading["vio_state"] == _VIO_STATE_ACTIVE`), and the executor separately
refuses a real run through `vio_feed_live` when tracked features fall below
`_VIO_MIN_TRACKED_FEATURES` (5) — *regardless of what `vio_state` claims*, which
is the dusk-latch guard.

Neither non-VIO mode is a way round it:

| mode | why it cannot test this |
| --- | --- |
| `night` | hard-capped at `_NIGHT_MAX_SEGMENT_LENGTH_M` = **1.0 m**, refused pre-dispatch by `night_segment_too_long`; also refuses loop-to-tolerance (`night_linear_loop_unsupported`); and it runs a **different** aim controller (`night_aim`), not the trigger under test |
| `legacy` | never enters the re-aim branch — no mid-drive correction of any kind |

The run needs a leg of **≥ 1.9 m**, so night's 1.0 m cap rules it out even
before the controller difference does.

**Consequence for scheduling:** the characterization run is daylight-only, and
`vio_tracked_features` saturating at 80 means the feed will look perfectly
healthy right up until it does not. Go with hours of light in hand, not minutes.
