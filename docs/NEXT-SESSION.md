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
  **Deploy 2026-07-24: `f2074722` (`services.py` + `coordinator.py`) is LIVE** —
  scp'd and md5-matched both sides, then loaded by the operator's own HA update
  restart. Verified on hardware: `get_map_data` returns the new `map_sync`
  block, 122 Mammotion entities. **Host is now behind again by the same two
  files**, carrying the map-saga fix below; they deploy together and need a
  restart.
- **VIO needs daylight.** It will not initialize in a dark scene; the gates
  refuse rather than drive blind. Check `camera_brightness` is not `Dark` and
  `track_feature_num` is healthy before any VIO run.
- **The map is HEALTHY again (read-only check, 2026-07-24 night).** `map.area`
  holds all 4 areas with full polygon frames (60–72 vertices each), `area_name`
  has 4 entries, and the GeoJSON carries 7 Polygons. `validate_custom_path`
  passes for real paths, and passes with an explicit `area_hash` of a real area.
  **So the Task-3 blocker on the Task-2 validation segment run is CLEAR** — the
  segment run no longer waits on a map fix. What recovered it (the deployed
  `regenerate_stale_geojson` fix, the 07-23 restart, or an intervening mow) was
  not determined. `force_map_resync` has still never been fired.

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

## 🐛 zone_hash was reading the map checksum (fixed 2026-07-24, NOT deployed)

`rpt_dev_location` carries **two** distinct fields: `zone_hash` (proto field 5 —
the mowing zone the mower is currently inside) and `bol_hash` (field 6 — a
MurMur checksum of the device's entire area set). `services.py` read `bol_hash`
wherever it meant `zone_hash`.

Because a map checksum is non-zero whenever the device has *any* map, the
substituted value was never 0 and never changed during a run. That silently
disabled **five** guards at once:

1. `_is_stale_zero_area_out_pose` — the (0,0)/AREA_OUT stale-dock-pose
   rejection could never fire (needs `pos_type == 0` **and** `zone_hash == 0`);
2. the `location_metadata` overlay that corrects a stale pose;
3. `zone_hash_unavailable` in `_manual_velocity_quality_degradation`;
4. `zone_hash_changed` — leaving one zone mid-run was undetectable, since the
   checksum is constant across a run;
5. the zone half of `_is_valid_motion_position` / `_position_has_known_area`,
   leaving `pos_type_label` doing all the work.

Evidence: the APK proto declares `ZONE_HASH_FIELD_NUMBER = 5` and
`BOL_HASH_FIELD_NUMBER = 6` on the same message
(`sources/com/agilexrobotics/proto/MctrlSys.java:60136-60143`); the app reads
`locationsList.get(0).getZoneHash()` for the live zone
(`MACarDataManager.java:4821`) and logs `bolHash` against its own locally
computed hash as a **map** comparison (`HashDataManager.java:303`);
pymammotion's `RptDevLocation` has both fields with the same numbering. Live
confirmation on the docked mower: the same message reported `zone_hash = 0`
(via `location.work_zone`) and `bol_hash = 8311072749804434520`.

Fixed in both read sites; the checksum is still reported, now as
`position.map_bol_hash`, and raw diagnostics list `zone_hash` and `bol_hash`
side by side. 4 regression tests, each verified to fail against the old read.

**⚠️ PRE-FLIGHT BEFORE THE NEXT REAL MOTION RUN — this fix makes the gate
strictly stricter.** If the firmware reports `zone_hash: 0` while `pos_type` is
`AREA_INSIDE`, motion that used to run will now be refused (fail-safe, but
blocking). Undock into a mapped area and check read-only *before* any real
command — no motion involved:

```yaml
service: mammotion.position_feedback_diagnostic
data: {entity_id: lawn_mower.back_yard_clip_skywalker, pulse_count: 0, dry_run: true}
```
Look at `raw_sources."report_data.locations"[0]`: `zone_hash` must be non-zero
while `pos_type` is 1 (`AREA_INSIDE`). It read 0 while docked (`pos_type: 5`,
`CHARGE_ON`), which is correct — the docked case proves nothing either way.

## The exclusive map saga vs. motion (fixed 2026-07-24, deployed)

> **⚠️ CORRECTION — read this first.** This section originally claimed the saga
> fired **every 5 minutes, forever**. That was **wrong**, and it was refuted the
> same day by three days of `sensor.<mower>_map_sync_status` history: only **5
> `syncing` episodes between 07-22 and 07-25**, each ~8–12 s, all clustered
> around restarts/reloads — and **none at all** in the ~25 h of continuous
> `out_of_sync` since 07-24 01:40.
>
> **Why:** `MammotionReportUpdateCoordinator._async_update_data` opens with
> `if data := await super()._async_update_data(): return data`, and the base
> method ends with `return self.data`. `MowingDevice` defines neither
> `__bool__` nor `__len__`, so it is **always truthy** — the early return fires
> on every healthy tick and **the bol-hash/map-sync block below it is
> unreachable in steady state**. It runs about once per HA start.
>
> **Consequences:** (a) the "every 5 minutes" churn does not happen, so the
> back-off below fixes a problem that is currently theoretical; (b) the
> motion-contention exposure is far smaller than claimed — ~5 sagas in 3 days,
> ~10 s each; (c) the "candidate for the 07-18 rotation decay" hypothesis is
> **withdrawn** — at that rate a saga landing inside a specific ~2-minute run
> window is improbable, and there is no evidence one did.
>
> **The real bug here is the opposite one**, and it is still open: because that
> block is unreachable, a device-side map edit (`bol_hash` change) is **never
> picked up while HA is running** — only on restart. `_map_callback`'s comment
> ("Map freshness is enforced in `_async_update_data()` via bol_hash checks")
> therefore does not hold. Fixing that would make the block live on every tick,
> which is exactly when the back-off below becomes necessary — so the two belong
> in the same change.

The contention **mechanism** below is code-verified and still worth guarding;
only its frequency was overstated:

- `MapFetchSaga` runs at `Priority.EXCLUSIVE` and **holds the mower's command
  queue** until it completes;
- motion commands go out at `Priority.NORMAL` with `skip_if_saga_active=False`,
  and `_process` does `await self._exclusive_active.wait()` — they **block**
  behind the saga;
- `_COMMAND_TTL = 120.0` **silently drops** anything undispatched for 2 minutes
  (only `EMERGENCY` is exempt);
- nothing in the motion path consulted `is_saga_active` — the only caller was
  the `map_sync_status` sensor label.

A saga landing mid-run therefore stalls pulses and collapses the 200 ms refresh
cadence, which makes the mower self-halt (the H-watchdog). Sagas do still happen
around restarts and reloads — exactly when someone is likely to be testing — so
the guard is worth having; it is just a low-probability event, not the routine
hazard first described.

**The guard was initially installed on the wrong call site.** `_should_start_map_sync`
covers only the per-tick path, which never runs. The paths that *actually* start
sagas are operator-triggered and were unguarded:

- **`button.<mower>_sync_maps`** — pressable at any moment, including mid-run.
  History shows a press at `23:49:09` followed by `syncing` at `23:49:10`, and a
  deliberate press on 07-25 held the queue **12–17 s**. This is the real hazard:
  not a background timer, but a dashboard button someone can hit while debugging
  the very run it would stall.
- **`force_map_resync`** — same, on demand, and *every* step it runs enqueues
  device commands.

Both now go through `_raise_if_manual_motion_in_progress()` on `async_sync_maps()`:
the button raises `HomeAssistantError` naming the owning service, and
`force_map_resync` refuses up front with `error: manual_motion_in_progress` +
`busy_owner`, sending nothing. The `_should_start_map_sync` back-off stays where
it is as defence in depth — a currently-inert guard that becomes load-bearing the
moment the tick path is made reachable:

1. **Back-off** — a repeat attempt against the same `bol_hash` waits out
   `MAP_INTERVAL` (60 min) instead of retrying every 5 min. A *changed*
   `bol_hash` still syncs immediately, so a real device-side map edit is never
   delayed. Uses the previously-unread `last_map_sync` plus a new
   `last_map_sync_bol_hash`.
2. **Motion-aware** — no saga starts while a guarded motion run holds the mower.
   The manual-motion claim moved from the `_ACTIVE_MANUAL_MOTION_RUNS` module
   dict in `services.py` onto `coordinator.manual_motion_owner`, because
   `services` imports `coordinator` and the flag has to be readable from both.
   Atomic check-and-set semantics preserved.

Deliberately **no new motion gate**: refusing a command the operator just issued
is worse than the wait, and these two make a mid-run saga rare.

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

   **✅ DONE 2026-07-25 — refresh gives ~7x (9° → 62°), compass-verified.** See
   the 2026-07-25 on-mower section below. Next code item from it: wire
   `motion_refresh_interval_ms` into `vio_turn_to_heading` and re-derive
   `heading_tolerance_degrees`.

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
   wrap-up). **NO LONGER GATED on the map** — containment now validates (see
   Task 3 below); the run just needs daylight, an operator, and good BLE.

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
   **UPDATE 2026-07-24 — the empty-map symptom is GONE and containment passes.**
   Read-only check on the idle/docked mower found `map.area` populated with all
   4 areas (full polygon frames), `area_name` complete, GeoJSON with 7 Polygons,
   and `validate_custom_path` returning `valid: true` — including with an
   explicit `area_hash` of a real area. `area_hash_not_found` still fires
   correctly for a hash that is not an area, which is what the mower's *dock*
   position hashes to. A-vs-B was therefore **not** decided: the symptom
   resolved before it could be attributed. `force_map_resync` remains unfired.

   **What is still wrong: `map_sync_status` reads `out_of_sync` on a map that is
   complete and usable.** That is not cosmetic — `coordinator.py:2396` fires
   `start_map_sync` on *every* coordinator tick while that holds, so an
   exclusive saga keeps taking the device command queue for no reason.
   `is_map_synced()` folds three conditions into one boolean (bol-hash match /
   no incomplete areas / area names covered), so the failing one was invisible.
   **✅ ANSWERED 2026-07-24 (live, read-only, after the fix went in):** the
   **bol-hash match is the sole failing condition.**

   | condition | value |
   |---|---|
   | `bol_hash_matches` | **False** — reported `8311072749804434520` vs computed `3951449155367542529` |
   | `incomplete_area_hashes` | `[]` — every declared area has its frames |
   | `area_names_covered` | `True` |
   | `area_frame_counts` | all 4 areas, 1 frame each |

   So the map is complete and correctly named; only the checksum disagrees.
   Sharpening it further: `computed_bol_hash` is **not** any permutation of the
   4 `map.area` hashes (all 24 checked), so `root_hash_lists` — which
   `computed_bol_hash` is actually built from — holds a **different set** than
   the areas we hold (extra entries, duplicates, or a stale manifest). The
   reported value isn't any ordered subset of the 4 either.

   **✅ RESOLVED the same night — and there is nothing wrong with
   `is_map_synced()`.** Pressing `button.<mower>_sync_maps` once converged it in
   **17 s**: `computed_bol_hash` `3951449155367542529` → `8311072749804434520`,
   `bol_hash_matches` → **True**, `map_sync_status` → **`synced`**. The saga's
   on-complete handler restores `root_hash_lists` from the saga result, which is
   exactly the documented convergence fix.

   So the stale local `root_hash_lists` was the whole story, and **~25 h of
   `out_of_sync` was resolved by one button press.** No pymammotion change is
   needed. Withdraw the earlier framing that `is_map_synced()` is "permanently
   false on this mower" — it was false only because **nothing was ever running a
   sync automatically** (see the unreachable-block bug below). That is the real
   defect, and it is still open.

Note `manual_velocity_pulse_test`'s `speed` is on the app's 0.0–1.0 scale
(default **0.55** → raw linear 400, matching the executors); its `duration_ms`
now caps at 4000 and `speed` at 0.6. The `app_speed_scale` block in the result
shows the resolved raw values.

## Long-term goal: how much do we miss when the integration is on cloud, not BLE?

**Queued 2026-07-25**, prompted by a concrete incident. The user manually drove
the mower (wheel-rolling, several distinct actions) between **12:03–12:19 PM
local** to clean it off. HA's history shows almost none of it:

- `active_transport` was `cloud_aliyun` for **12:05:07–12:35:27 PM local**, a
  30-minute span covering nearly all of the described activity.
- In that entire 30-minute cloud window, **exactly one** report update reached
  HA: `activity_mode → MODE_MANUAL_MOWING` / position → `AREA_BORDER_ON` →
  `AREA_INSIDE`, lasting **20 seconds** (12:19:48–12:20:08 PM), then silence
  again. Confirmed with `significant_changes_only=0` (the raw, unfiltered
  history) — this is not deduplication hiding repeated identical polls; the
  entity genuinely was not written to at any other point in that window.
- The mower was `DOCKED_FULL` (on dock, battery 100%) going into the window.
  pymammotion's own one-shot MQTT poll table
  (`pymammotion/device/mqtt_loop.py::_MQTT_POLL_INTERVAL` /
  `_MQTT_NEW_POLL_INTERVAL`) gives that mode a **60-minute** cadence — `IDLE` is
  10–15 min, `ACTIVE` (mowing) is 10–20 min. Whichever bucket applied, it is
  coarse enough that several short, discrete real-world actions can start and
  finish entirely between polls and leave **zero** trace, not just a
  low-resolution one. The one blip we did catch is more likely a device-pushed
  mode-change report than a scheduled poll landing — worth confirming, not
  assumed.

This sharpens (not just repeats) the 2026-07-21/22 finding that "the position
feed is BLE-only, stone dead on cloud" — that finding was about resolution
during a *continuous* mow; this incident shows *entire actions can vanish*
during a docked/idle cloud period.

**Investigation to run, off-mower, next time this comes up:**

1. Confirm which `_DeviceMode` bucket governed the 60-min gap (`device_mode()`
   in `pymammotion/device/handle.py`) and whether `MODE_MANUAL_MOWING` maps to
   `ACTIVE`, `IDLE`, or something uncounted — check `MOWING_ACTIVE_MODES`
   against the `sys_status` seen during this incident.
2. Determine whether the one blip we caught was a scheduled one-shot poll or a
   device-initiated push on mode change — if the device pushes on its own
   state transitions, that's a materially better story than "60-minute blind
   spots," and worth confirming rather than assuming the worse case.
3. Decide whether an operator should be told, in the card or via an automation,
   when the integration has been on cloud for more than N minutes — right now
   there is no visible warning that "what you did just now may not be
   reflected here at all," which is exactly what happened.
4. Do **not** conclude cloud is silent — the MODE_MANUAL_MOWING blip proves
   some report traffic gets through on cloud. The open question is *how much
   is missed*, not *whether everything is missed*.

## 🏆 2026-07-25 ON-MOWER: turn refresh PROVEN (~7x), then BLE collapsed

Daylight supervised session. **Step 4 (zone_hash pre-flight) PASSED and Step 5a
(turn A/B) SUCCEEDED.** Step 5b (segment run) was attempted twice and both
attempts aborted — correctly — on transport failures, not logic failures.

### ✅ Step 4 — the `zone_hash` fix validated on hardware

Off-dock in `Backyard Right`, live read: `pos_type_label: AREA_INSIDE`,
`zone_hash: 1343645155037768237` (non-zero), `valid_for_motion: true`,
`safety.allowed_for_manual_motion: true`, `blockers: []`. So the stricter gate
introduced by `f2074722` does **not** over-block. Pre-flight is closed out.

Also confirmed live: `position.map_bol_hash` reports separately from
`zone_hash`, and `area_name` correctly resolves to `Backyard Right` rather than
the old misleading `"path"`.

### 🏆 Step 5a — refresh gives ~7x more rotation at angular 500 (compass-verified)

`vio_turn_probe`, angular 500, 4.0 s, blade off, BLE −56..−58:

| pulse | `motion_refresh_interval_ms` | compass (ground truth) | VIO `vision_heading` |
|---|---|---|---|
| A | 0 (single-shot) | 170° → 179° = **+9°** | −8.75° (sign inverted, magnitude agrees) |
| B | 200 (21 re-sends) | 179° → 241° = **+62°** | −62.92° |

**~6.9x more rotation from refresh alone.** This closes the question left open
on 2026-07-22, where the turn half of the B1 A/B was inconclusive because
`manual_velocity_pulse_test` caps angular at ~202 and could not reach the
threshold. Refresh is **speed-gated**: useless at angular 180 (below this
mower's actuation threshold), decisive at 500. Mirrors the linear result
(single-shot ~4 in vs refresh-200 44 in, ~11x).

**Unexpected bonus: VIO tracked BOTH turns accurately** (within ~1° of compass,
and course-over-ground agreed too). The standing worry that VIO blinds/latches
on a fast turn did **not** reproduce here in good daylight with 78–80 tracked
features. Do not over-generalise from one session, but the compass may be less
essential than assumed when the feed is healthy.

**Implication:** `heading_tolerance_degrees: 18` was chosen because single-shot
turning was quantised into ~8–15° steps that could not land inside a tighter
deadband. With refresh producing continuous rotation, that tolerance can likely
drop a lot. **Next off-mower code item: wire `motion_refresh_interval_ms` into
`vio_turn_to_heading` (still single-shot) and re-derive the tolerance.**

### ❌ Step 5b attempt 1 — `no_actuation_detected`, but the mower DID move

Two turn pulses (angular −500, 1500 ms then 700 ms), then abort:

```
before_vision_heading: 90.29915121519771   after: 90.29915121519771  (bit-identical)
displacement_m:        0.006754257916307457 (bit-identical across BOTH pulses)
heading_poll_seconds:  8.01   heading_went_fresh: FALSE   (both pulses)
```

**Operator observed ~4 inches of real movement.** So the mower actuated and the
telemetry reported nothing. Server logs from the same window corroborate:

```
Failed to parse incoming bytes as LubaMsg (386 bytes)
dropping frame: malformed report data failed deserialization (249 bytes):
  Field "pos_type" of type int has invalid value [76,117,98,97,45,86,83,80,76,86,51,57,55]
```

Those bytes are ASCII **`"Luba-VSPLV397"`** — the device's own name landing in an
int field, i.e. a corrupted/misaligned BLE frame being dropped wholesale.

**🐛 REAL GAP FOUND: the turn phase has no stale-feed detector.** The linear
phase got `telemetry_stream_stale` on 2026-07-19 (bit-identical position across
≥3 polls = dead stream, not a stopped mower). The turn phase never got the
equivalent, so `no_actuation_detected` cannot distinguish *"the link is dead"*
(the 07-19 e-stop case it was built for) from *"the feed froze while the mower
turned fine"* (tonight). **The discriminating signal was already in the result
and unused: `heading_went_fresh: false` with an 8.01 s poll timeout on BOTH
pulses.** Fix: when the freshness poll times out, do not emit
`no_actuation_detected` — emit a stale-feed reason instead. Same "always require
positive evidence the sensor is live" lesson as the three earlier false-positive
bugs.

### ⚠️ Step 5b attempt 2 — calibration pulse sent, THEN THE STOP FAILED

```
phase: vio_calibration_drive, linear_speed 400, ok: true
stop_result: { attempted: true, ok: false,
  error: "BLEUnavailableError: BLE connect for 'Luba-VSPLV397' is in cooldown (120s remaining)",
  duration_ms: 8992.7 }
stop_reason: vio_calibration_failed  (calibration reason: stop_failed_aborting)
```

BLE dropped into a fresh connect cooldown **during** the stop attempt; it tried
~9 s, failed, and the run aborted rather than continuing. **The safety hardening
worked as designed** — but there was no positive confirmation the mower stopped
on command. Position afterwards was unchanged within ~2 mm, and the **operator
visually confirmed the mower stopped and safe** — consistent with the documented
single-shot self-halt holding even when the explicit stop cannot be delivered.
Treat this as the reference example of why `stop_failed_aborting` exists: the
undelivered stop was caught and escalated rather than silently ignored, and the
bounded-pulse design meant an undeliverable stop was not a runaway.

### 🚨 The session's dominant problem: BLE was collapsing all evening

Not a code bug — a radio/link problem, and worse than the usual coverage wall:

- **4–5 transport flips** to `cloud_aliyun`, repeatedly re-armed 120 s connect
  cooldowns.
- **`BleakOutOfConnectionSlotsError` with all proxies healthy**: `6 scanner(s)
  registered, 6 scanning, 6 connectable` but **`last advertisement 613s ago`** —
  the mower's radio had gone silent entirely, not a proxy-capacity issue. This is
  the ~10–13 min idle-doze, and it bit us repeatedly *because diagnostics between
  commands take longer than the doze window*.
- **`ble_rssi` is self-reported by the mower** (`report_data.connect.ble_rssi`),
  so it holds a stale value when the mower goes quiet — it read a healthy −64
  while nothing had heard an advertisement for 10 minutes. **Do not trust
  `ble_rssi` as a liveness signal.** Bit-identical rssi across many polls = no
  new reports, i.e. the same stale-feed tell as everywhere else.
- **The cloud-routed restart does not work here.** `button.<mower>_restart_mower`
  → `remote_restart` returned HTTP 200 but the dispatch failed:
  `WARNING [pymammotion.aliyun.cloud_gateway] Error in sending cloud command:
  20056 - gateway.hsf.invoke.timeout`. Nothing happened. Worth retrying only when
  the cloud channel is healthy.
- **Full blackout observed** by the click-to-path card:
  `No transport available ... [cloud_aliyun=connected, ble=disconnected]
  (mqtt_reported_offline=True)`.
- `switch.<mower>_bluetooth` `turn_on` returned **HTTP 500** once (it tries to
  force a BLE connect to a non-advertising device) and silently failed to apply
  another time — **always verify the switch state after toggling, never trust the
  HTTP response.**

**Ops recipe that did work, repeatedly:** app-triggered mower restart →
reconnects in 10–20 s. The bluetooth switch toggle works *only* when no connect
cooldown is armed (by design — recovery defers during cooldown). Waiting out the
120 s cooldown without probing is better than hammering it, since failed attempts
re-arm it.

### Smaller findings

- **`toward` is unreliable after a restart.** It read `97.06°` while the compass
  read `241°` (~144° off), stayed within 0.002° across 6 polls / 30 s, and
  survived two restarts (`97.0647` → `97.0629` → `94.5699`). Position x/y
  re-converged correctly; only heading was wrong. **This does not affect the
  executor** — verified by code read: the calibration drive derives map heading
  itself via `atan2(dy, dx)` from a live position delta (`services.py:7932`), and
  mid-drive re-aim uses `vision_heading` + that fresh offset (`services.py:8762`).
  Neither reads `toward`. Still worth understanding.
- **`max_linear_pulse_ceiling` is not echoed** in the vector executor's result
  (it echoes `None` even when passed and honoured — confirmed it *is* wired in at
  `services.py:8477`). The multi-segment executor got this echo gap fixed on
  2026-07-19; the vector one still has it. Cosmetic, blocks forensics.
- **The map emptied again after real motion** (`map.area` count 0,
  `out_of_sync`) and recovered on `force_map_resync`. Consistent with
  `invalidate_maps()` firing on a report whose `bol_hash` mismatches, plus the
  known unreachable-auto-resync bug. The new `map_sync` diagnostic block made
  this legible in seconds.
- The default `max_linear_commands: 1` means a segment call **stops after one
  linear pulse** unless `max_linear_pulse_ceiling` is passed — with refresh now
  covering ~1 m per pulse, always pass a ceiling for a multi-metre segment.

### Where Step 5b stands

**Not achieved.** Both attempts died on transport, never reaching the linear
phase, so **the Task-2 constants (pulse-geometry ceilings, `min_progress_distance`,
cadence) are still un-re-derived and remain hypotheses.** Retry needs a healthy
BLE session; consider moving the mower closer to a proxy first, and fire the run
promptly after a wake rather than spending the doze window on diagnostics.

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
