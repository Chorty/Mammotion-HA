# Firmware camera/vision/navigation stack review — 2026-09-22

## Scope and method

Static, read-only inspection of the recovered `1.30.29.24` middleware
filesystem (`ota_work/recovered/1.30.29.24/filesystem`, gitignored). No
extracted binary or script was executed. No mower, cloud, or network request
was made. Covers `pkgs/vision/mower_perception`, `pkgs/vision/mower_vslam_vio`,
`pkgs/nav` (`agilex_navigation`), and `pkgs/ins_fusion` — the four processes
`startup/agl_monitor_process.sh` watches for the camera/vision/navigation
stack (see `docs/session-state-firmware-capability-work-20260919.md`'s
2026-09-22 update for how that watchdog script was found).

Version strings recovered from each package's own `version` file
cross-checked cleanly against the live `deviceOtherInfo` health payload
pulled the same day (`docs/session-state-firmware-capability-work-20260919.md`):
`vslam_vio 1.6.1 (33b015a)` matched exactly; `perception 1.12.14_master`
matched on version with a differing trailing commit
(`2ec2cb598` live vs `eb859483` / `-s29hotfix` in the recovered image — a
minor hotfix difference, not investigated further); `ins_fusion 1.5.14
(4a41ffe)` matched exactly. This is corroborating evidence that the recovered
image is genuinely representative of what is running, not proof of byte-for-
byte identity.

## `mower_perception` — stereo obstacle detection and a face-mosaic privacy pass

### Seven CNN models, all loaded unconditionally at startup

`config/config.yaml` configures seven ONNX/compiled model paths under both a
`pc:` (host-simulation) and `soc:` (on-device) section, all loaded the same
way with no per-model enable flag in this file:

| Model | File | Apparent purpose |
| --- | --- | --- |
| `segment` | `model/segment.bin` (+ `segment_dark.bin`, `segment_strict.bin` variants) | Semantic segmentation (13-class confusion matrix present, `config/segment_confusion_matrix.yaml`) |
| `detect` | `model/detect.bin` | Object detection |
| `face_detect` | `model/face_detect.bin` | Face detection — see below |
| `dirty` | `model/dirty.bin` | Camera-lens dirty detection (see below) |
| `water_detect` | `model/water_detect.bin` | Water/pond surface detection |
| `mono_depth` | `model/mono_depth.bin` | Monocular depth (disabled at runtime — see below) |
| `camera_occ` | (soc-only path) | Camera occlusion detection |

### The face-detect model feeds a face-mosaic privacy pipeline, not a safety stop

`mower_perception_node` contains a class `FaceDetectProcess` and a distinct
thread/class `DataEncryptionProcess`. Their log strings, in order, read as a
batch pipeline:

```
DataEncryptionProcess start run!
DataEncryptionProcess start ProcessFiles!
DataEncryptionProcess Skip folder not found: / Skip non-existent folder -
DataEncryptionProcess mosaic_times_txt: / read mosaic_time txt failed!
DataEncryptionProcess Image not found with index: / Failed to read image:
DataEncryptionProcess check nv12-size width:
DataEncryptionProcess Mosaic rects size:
DataEncryptionProcess no human faces in the image!
DataEncryptionProcess Failed to write image: / Processed and saved:
DataEncryptionProcess read_image_duration / cnn_detect_duration_ms /
  mosaic_duration_ms / write_image_duration_ms / whole_duration_ms cost_time:
```

Read together: this walks a folder of saved NV12 images tracked by a
`mosaic_time.txt` progress file, runs face detection (`CNNFaceDetect::detect`)
on each, computes mosaic rectangles over any detected face(s), writes the
mosaiced image back out, and records progress so it resumes rather than
reprocessing. **This is a face-redaction pass applied to stored images before
disposition (upload, retention, or app display), not a "stop near a person"
safety feature.** No `process_obstacles`/detection config in
`mover_post_process_config.yaml` or `config.yaml` references "face" or
"person" for obstacle purposes — the obstacle pipeline's `occ_person_thres`
(below) comes from the separate semantic-segmentation model's "person" class,
not from `face_detect`.

I could not identify the exact source folder `DataEncryptionProcess` walks
from the recovered strings alone (no absolute path string was found near
these log lines) — **open question**, not resolved by this pass. The presence
and working shape of the pipeline is confirmed; its exact trigger and target
image set is not.

### Obstacle detection: a real stereo pipeline, water/pond and pole detection on, road detection off

`config/mover_post_process_config.yaml` (Chinese comments, translated below)
configures a full stereo-based obstacle-detection pipeline distinct from
face/segmentation:

- `process_obstacles: 1` (on), `process_pond: 1` (on — water/pond obstacle
  detection), `process_tube: 1` (on — pole/tube detection), `process_roads: 0`
  (off), `process_mono_depth: 0` (off — the mono-depth model is loaded but
  unused for obstacle processing on this build).
- Per-model camera geometry (extrinsics `R_c_b`/`t_c_b`, mount height,
  depth/obstacle thresholds) is configured separately for `LUBA2`, `LUBA2_PRO`,
  `LUBA_MINI`, `YUKA`, and `YUKA_MINI` — confirming per-SKU camera calibration,
  not one shared geometry.
- Each model block carries `occ_vegetation_thres`, `occ_car_thres`,
  `occ_person_thres` (all 0.2–0.3 range) for an occupancy-grid classifier, but
  `is_send_occ: 0` on every model — **the occupancy grid is computed but not
  published** on this build, the same "capability present, not wired to an
  output" shape as the `vision_proxy` finding from the same day.
- `water_filter_obstacles: 0` at the top level (off) — a documented
  "water-mist/glare obstacle filtering" feature exists (`water_obstacle_distance`,
  `water_intersect` thresholds are configured) but is disabled by default.
- Max 10 obstacles reported per frame (`max_obstacles_num: 10`), with
  detection-confidence (`detection_score: 0.6`) and classification-rate
  (`classification_rate: 0.6`) thresholds fusing the CNN detector with depth
  segmentation.

### Dirty-lens detection matches the already-documented fault 1068

`dirty_model_path` loads a CNN dirty-lens classifier
(`libcnn_dirty.so`/`model/dirty.bin`). This corroborates the existing,
already-documented fault `1068` ("vision camera is dirty") from
`docs/findings-setup-leg2-dusk-realign-halt-20260912.md` — the firmware has a
dedicated model for this, not a heuristic on image statistics.

## `mower_vslam_vio` — stereo+IMU+GNSS+wheel VIO, and where `max_cnt: 80` comes from

`config/vio_parameters.yaml` is a standard VIO-style parameter file (tightly-
coupled optimization-based fusion, `ceres_thread_num: 1`, `max_solver_time:
0.04` s). Confirmed sensor inputs: 1 IMU + 2 cameras (stereo), plus
**GNSS (`use_gnss: 1`) and wheel odometry (`use_wheel: 1`) fused in**, not a
camera+IMU-only VIO. Also on: failure-triggered reboot
(`enable_failture_reboot: 1`), dark-scene detection
(`enable_dark_detect: 1`), and zero-velocity update
(`enable_zupt: 1`) — a standard technique to suppress drift while stationary.

🔑 **`max_cnt: 80` is the configured maximum tracked-feature count for the
feature tracker** (`feature_type: 0` — a custom "luba fast" detector variant,
not the standard FAST/GFTT trackers the comment lists as alternatives 1/2).
This is the direct source of the "~80 is healthy" ceiling already observed
live via `mammotion_preflight_gates.py` (`tracked_features=80 (>=5 needed;
~80 is healthy)`) and documented in memory
(`[[vio-tracked-features-is-a-cliff]]`) — it was previously only known
empirically; this firmware file is the reason why 80 is the number, not
some other value.

## `agilex_navigation` and `ins_fusion` — build-branch names that corroborate existing findings

Both binaries are stripped ARM aarch64 ELF executables with build-info strings
readable via `strings`. Two branch names are directly relevant to work already
recorded in this repo:

- `agilex_navigation` — `ver:1.0.0`, **branch
  `release_30_x3_feat_turn_back_close_knift`** ("turn back, close blade"),
  built 2026-08-20, commit `fa92c3855`. The mower's own navigation firmware
  has a named feature branch specifically about turning and closing
  ("close-knife" = stopping the blade) — a candidate explanation for
  turn-related blade behavior, not yet cross-referenced against any specific
  finding in this repo. Flagged, not chased further this session.
- `ins_fusion` — `ver:1.5.14`, **branch `fix_exit_charge_jump`**
  ("fix exit-charge jump"), built 2026-03-13, commit `4a41ffe`. This directly
  corroborates the project's own independently-discovered
  2026-09-04 finding (`docs/findings-clicktopath-reliability-4m-20260904.md`,
  and the CLAUDE.md "traps that keep biting" entry on stale heading telemetry
  after repositioning) that a repositioned/just-undocked mower's heading
  telemetry jumps sharply (~166° was measured) on its first real motion. The
  firmware has a named bugfix branch for exactly this phenomenon — evidence
  the vendor is aware of and has worked on the same class of bug, not proof
  the fix fully covers the case this project measured.

No lidar-mapping binaries (`mower_lidar_mapping`, `fast_lio_slam_node`) exist
anywhere in this firmware image — consistent with `agl_monitor_process.sh`
only starting them when `firmware_id == "LMNX3RMidWare"`, and this image's own
identity being `LubavX3Midware` (camera/VIO-based, non-lidar SKU family).

## Negative results and limits

- No source folder path was recovered for `DataEncryptionProcess`'s face-mosaic
  input — its trigger condition and target image set are unconfirmed.
- `mower_perception_node`'s occupancy-grid output (`is_send_occ`) is off on
  every model in the recovered config; this pass does not establish whether
  any deployed configuration turns it on.
- This is a static string/config read, not a decompile of the control-flow
  binaries (`agilex_navigation`, `mower_perception_node`, `ins_fusion` core
  logic) — branch names and log strings are read as evidence of what exists
  and roughly what it does, not proof of exact behavior.
- No firmware binary or script was executed, and no mower/cloud/network
  request was made.

## Reproduction

```sh
cd ota_work/recovered/1.30.29.24/filesystem
cat pkgs/vision/mower_perception/config/config.yaml
cat pkgs/vision/mower_perception/config/mover_post_process_config.yaml
strings -a pkgs/vision/mower_perception/mower_perception_node | grep -iE "DataEncryption|FaceDetectProcess"
cat pkgs/vision/mower_vslam_vio/config/vio_parameters.yaml
cat pkgs/nav/version pkgs/ins_fusion/version
```
