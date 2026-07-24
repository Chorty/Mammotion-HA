# Next-session handoff — click-to-path (guarded mower motion)

Single "start here" for resuming this work on any machine. Deep detail lives in
`docs/codex-working-plan.md` (see the dated sections, newest last). Host
connection details and credentials are **not** in git — they're in `.env`
(gitignored) at the repo root; recreate `.env` on a new machine from your own
records.

## Where we are

- **Phase 1 (straight-line click-to-path): done, deployed, live-validated.** The
  card drives guarded segments to tolerance via a bounded, explicitly-stopped
  pulse loop (`raw_pymammotion_execute_vector_segment`).
- **Phase 2 (turning): done and proven live.** In-place rotation IS observable
  via `report_data.vision_info.heading` (VIO), which resolved the old
  "turning is unobservable" blocker on 2026-07-10. `toward` is
  course-over-ground and stays frozen during a pivot — do not use it for turns.
- **Multi-segment click-to-path: proven live** (L-path, both segments
  `target_reached`). PR #10 is open against main and mergeable.
- **Full APK 2.3.8.19 feature sweep: complete (off-mower, 2026-07-22).**
  The complete 9-dex/30,867-Java-file decompile was split across parallel
  subsystem passes, then independently checked for omissions, exact protocol
  claims, model gates, and citation integrity. The durable catalog starts at
  `docs/apk-feature-catalog/00-overview.md`; use
  `10-protocol-report-index.md` for commands/reports,
  `11-ha-opportunity-index.md` for future integration backlog,
  `12-coverage-and-open-questions.md` for limits/live-verification work, and
  `13-model-capability-matrix.md` for model gates. This is a complete static
  sweep, not proof of server/RN/H5-delivered behavior or hardware safety.
- **Deploy state (2026-07-23): current through `1b22da7a`.** Deployed
  `coordinator.py`, `services.py`, and `services.yaml`, checksum-matched all
  three, and restarted HA Core. API returned after 41 s; Mammotion returned
  with 121 entities after 129 s. Verified all 59 Mammotion services registered,
  including `force_map_resync`; deployed refresh defaults are vector=200,
  multi=200, and `vio_turn_probe`=0. Startup logs show no Mammotion import/setup
  failure (only the existing custom-integration and deprecated tracker-property
  warnings). `lawn_mower.py` was unchanged and remains at the previously
  deployed `0a5bc4ab` level.
- **VIO needs daylight.** It will not initialize in a dark scene; the gates
  refuse rather than drive blind. Check `camera_brightness` is not `Dark` and
  `track_feature_num` is healthy before any VIO run.

## The big result — app-parity refresh cadence PROVEN (B1, 2026-07-22)

A JADX decompile of the app (2.3.8.19) showed `CarRemoteControlManage2` re-sends
the **identical** `DrvMotionCtrl` command every **200 ms** while the stick is
held (`needAck=false`), where our executors sent it **once** and slept. The
tape A/B settled it live:

- **Forward 4 s, single-shot → ~4 in. Forward 4 s, `motion_refresh_interval_ms:
  200` → 44 in (~11×).** The mower grants a short motion window (~one 4-in step)
  and self-halts unless the command is refreshed. This one bug was the root cause
  of the fixed ~4-in step, the ~8–15° rotation quantum, and the ~35-pulse/3.5 m
  path that always outlived the BLE link. A follow-up ~1.3 m continuous glide
  (refresh 200) confirmed it holds over real distance, dead straight.

- **Refresh is SPEED-GATED — this is the key nuance.** It only helps when the
  per-command speed is *above* the mower's actuation threshold. Linear 400 clears
  it (→ continuous drive). A turn at angular **180** did **not** benefit (~3°,
  single-shot ≈ refresh) because 180 is below this mower's rotation threshold —
  which *vindicates* the old "angular needs 500" calibration; that part was real,
  not a watchdog artefact. Re-sending a too-weak command is still too weak.

- Confirmed off the mower: refresh in the executors is scoped to the **linear
  phase only** (calibration drive + VIO turn are untouched), so wiring it in is
  safe. `needAck=false` also means per-pulse `command_ok` never proved delivery.

## Immediate next steps (all doable off-mower)

1. **Test refresh on a *properly-powered* turn.** `manual_velocity_pulse_test`
   caps angular at ~202 (speed ≤ 0.6), so it *cannot* command the ≥382/500 a real
   turn needs — that is why B1's turn half was inconclusive. **Done this session:**
   `vio_turn_probe` now takes `motion_refresh_interval_ms` (its `angular_speed`
   already reaches 500). Next mower session: A/B a refresh-500 turn:
   ```yaml
   service: mammotion.vio_turn_probe
   data:
     entity_id: lawn_mower.back_yard_clip_skywalker
     angular_speed: 500
     drive_seconds: 4.0
     motion_refresh_interval_ms: 0     # then re-run with 200
     dry_run: false
     confirm_blades_off: true
     confirm_clear_area: true
   ```
   Measure rotation by phone compass flat on the deck (VIO heading latches/blinds
   during a fast or dark turn — do not trust it). If refresh-500 turns
   continuously, `heading_tolerance_degrees` can drop far below 18.

2. **✅ DONE (code, 2026-07-22, NOT deployed): the executors' linear refresh now
   defaults to 200** (`raw_pymammotion_execute_vector_segment` / `_multi_segment`
   schema + yaml; the `manual_velocity_pulse_test` / `vio_turn_probe` harnesses
   stay single-shot at 0; 326 tests, mypy+ruff clean). **Still open — re-derive
   the three constants that assumed the old ~4in step: pulse-geometry ceilings,
   `min_progress_distance`, and cadence, against continuous drive** (a 3.5 m path
   is now ~1 m/pulse → ~3–4 pulses / ~12 s, which should dodge the BLE wall). The
   re-derivation is a *plan only* until one supervised segment run exercises the
   executor's settle/sample/progress loop under refresh — full model + the
   measurements to take are in `docs/codex-working-plan.md` (2026-07-22 "later"
   wrap-up). **That run is gated on the Task-3 map-sync fix** (containment can't
   validate a segment while the zone polygon is missing).

3. **✅ DIAGNOSED + two fixes shipped (code, 2026-07-22, NOT deployed).** The
   map-sync bug: after a reload/restart the zone *polygon* geometry never
   re-projects (geojson points+line only, `map_sync_status: out_of_sync`,
   containment `area_hash_not_found`). **Root cause:** `coordinator.data.map.area`
   (the polygon frames) is empty — containment reads those frames *directly*
   (`_area_polygons`), and the geojson is derived from them, so both symptoms are
   one state. The map-sync saga isn't populating/re-projecting the frames for an
   idle mower, and nothing recovers it (the only geojson-regen triggers fire on
   the mowing report hot path; the saga's on-complete rebuild is skipped when
   `RTK.latitude == 0.0`; and — confirmed gap — our integration never called
   pymammotion's `regenerate_stale_geojson()` after `restore_device()`, contrary
   to its docstring). **Fixes:** (a) call `regenerate_stale_geojson()` after
   restore in `coordinator.async_restore_data`; (b) new `mammotion.force_map_resync`
   service (non-destructive recovery: refresh RTK/dock → fetch area names → run
   the saga → re-project; returns step-by-step result). Full analysis in
   `docs/codex-working-plan.md` (2026-07-22 map-sync section).
   **STILL OPEN — one live read confirms A-vs-B** (I can't from the desk): if an
   active mow re-projects the map but idle never does → contributor **B**
   (frames present, geojson stale) and the two fixes resolve it. If even a mow
   leaves `map.area` empty → contributor **A** (saga can't populate over the
   transport — a BLE-coverage problem, not code) and the fixes only aid recovery.
   Read via `get_map_data`/`_export_mower_map` (does `map.area` have keys?),
   `sensor.<mower>_last_map_task_error`, `map_sync_status`, and RTK/dock latitude.
   Then, on a good-BLE moment, call `mammotion.force_map_resync` and re-check.

Note `manual_velocity_pulse_test`'s `speed` is on the app's 0.0–1.0 scale
(default **0.55** → raw linear 400, matching the executors); its `duration_ms`
now caps at 4000 and `speed` at 0.6. The `app_speed_scale` block in the result
shows the resolved raw values.

## Known walls (not code bugs)

- **BLE proxy coverage is the hard limit.** Works above ~-70 rssi, dies below
  ~-76 (ESPHome GATT status=133 → 120 s cooldown). Long runs outlive the link.
  A faster drive (refresh cadence) may dodge this on its own.
  - **The reliable fix when stuck on a far proxy (frozen rssi, no hand-off):**
    hold a BLE proxy right next to the mower and **restart the mower** — it drops
    the stuck link and HA's proxies race to reconnect, the close one wins
    (proven live 2026-07-22, −86 → −58). A plain reload/toggle does *not* force
    re-selection.
  - **This mower dozes to `ble_rssi: 0` within ~10 min idle** and drops BLE, so
    the window closes between commands — fire follow-up pulses fast after a wake,
    and `rssi: 0` means wake it physically (a toggle can't reach a non-advertising
    mower).
- **A physical e-stop is invisible in telemetry.** On 2026-07-19 a forgotten
  e-stop silently no-op'd five real motion commands over ~40 minutes while every
  health indicator read green; `lock_state` is *not* e-stop. If commands report
  OK and nothing moves, check the physical button before debugging code.

## Live-testing workflow (essential)

- **Deploy code by copying to the HA host, then RESTART HA Core.** A config-entry
  reload does NOT reload changed Python — you must restart. Restart needs an
  explicit operator "restart HA".
- **Real motion requires BLE** (a gate refuses cloud). BLE is flaky here; after a
  restart it comes back on cloud — toggle the mower's `bluetooth` switch off→on
  and confirm it holds. Rapid toggling trips a ~120s BLE reconnect cooldown.
- **Force-close the iOS app before BLE testing** — it holds the mower's single
  BLE connection slot. `ble_rssi` = 0 means the mower is asleep and not
  advertising; wake it.
- **Every real-motion command needs a fresh operator "go"**; re-check state
  (paused, blade OFF, BLE, valid-for-motion) right before each fire.
- **Keep the mower in open space, clear of the dock** — dock obstruction masks
  motion and looks like frozen telemetry.
- **Trust the tape, not the telemetry, for distance *when pulsing*.** During our
  bounded-pulse motion the map-local feed lags ~4 s, updates in jumps, and shows a
  ~2-6 cm absolute error. **But that is a property of pulsed measurement, not of the
  feed** — measured during a real autonomous mow on 2026-07-21, under known-continuous
  motion the same feed is **sub-centimetre** (0.70 cm cross-track RMS over 8 straight
  5-10 m runs) and **never froze once** in 86 consecutive samples. Treat the 2-6 cm
  figure as the pulsed-motion budget and the sub-cm figure as the feed's real
  precision; if B1 shows refresh-mode produces continuous motion, re-derive
  `min_progress_distance` against the sub-cm number rather than the 6 cm one.
- **The position feed is BLE-only.** On `cloud_aliyun` it is stone dead — 20 of 21
  consecutive polls bit-identical during a live mow, vs 74 of 74 moving on `ble`.
- Gate every change: `uv run pytest`, `uv run mypy custom_components/`,
  `uv run ruff check`.

## Repo gotcha

A GitHub Actions workflow auto-pushes version-bump commits and **regresses** the
version (was beta11 in June, back to beta7 by July). Standing rule: keep the
higher beta when resolving; don't trust the manifest version to reflect what's
deployed (deploy is by file copy + md5, independent of the version string).
Consider disabling/fixing that workflow.
