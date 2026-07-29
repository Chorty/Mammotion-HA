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
  **✅ Deploy 2026-07-25 (late): `services.py`, `services.yaml` and
  `coordinator.py` are LIVE** — md5-matched both sides (`5373e70c` / `022443be` /
  `a0c352ab`), HA Core restarted (API 61 s, 122 entities 190 s, 53 services).
  **The host is current with the working tree.** The dead-region fix is PROVEN on
  hardware: `Updated Mammotion device` now logs every ~5 min (22:32:28 / 22:38:04
  / 22:43:19) where it appeared **zero** times before, so the opportunistic BLE
  reconnect actually runs. Both service changes confirmed by dry run.
  **⚠️ The restart left the integration in `setup_error` with no auto-retry** (a
  BLE GATT `start_notify` timeout escaping `async_setup_entry` as a raw
  `TimeoutAPIError`); a manual entry reload fixed it. See the hazard section in
  the plan doc — restarting HA while BLE is unhealthy can leave the integration
  dead silently.
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
> **The real bug here is the opposite one:** because that block is unreachable, a
> device-side map edit (`bol_hash` change) is **never picked up while HA is
> running** — only on restart. `_map_callback`'s comment ("Map freshness is
> enforced in `_async_update_data()` via bol_hash checks") therefore does not
> hold. Fixing that makes the block live on every tick, which is exactly when the
> back-off below becomes necessary — so the two belong in the same change.
>
> **✅ FIXED 2026-07-25 (later, NOT deployed)** — see the item-4 section below.
> The block turned out to be in `MammotionMapUpdateCoordinator` (60 min), not the
> report coordinator, and the same bug existed in **five** coordinators. The
> back-off is now load-bearing as predicted.

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

~~Deliberately **no new motion gate**: refusing a command the operator just issued
is worse than the wait, and these two make a mid-run saga rare.~~

**⚠️ REVERSED 2026-07-26 (adversarial review).** Both premises stopped holding:

1. **Sagas are no longer only operator-triggered.** Making the per-tick block
   reachable means they also fire automatically (~hourly), so the operator has
   no idea one started — the "they'd know why" half of the argument is gone.
2. **The "wait" was never benign.** Motion is `Priority.NORMAL` with
   `skip_if_saga_active=False`, so it *blocks* on the exclusive slot while the
   executor sleeps out its pulse on **local timing**. The pulse can elapse
   before the mower moves at all, and the movement and its stop can be
   separated — or either dropped at the 120 s `_COMMAND_TTL`. That breaks the
   bounded-pulse guarantee the safety model rests on; it is not a late start.

The original objection was really to an *unexplained* refusal.
`_exclusive_saga_active()` now gates the motion wrapper and returns the
existing busy shape with `busy_owner: map_sync_saga`, so the operator is told
it is a map sync and can retry in seconds. The probe is
positively-True-only — any unreadable piece allows motion, so pymammotion API
drift can never block everything — and it is a synchronous read, so the
check-and-claim stays one uninterrupted block on the event loop.

## 🏆 2026-07-27 ON-MOWER: final approach PROVEN; turn granularity is the new wall

Three runs after deploying the sub-pulse final approach (`b62a988c`). Deploy
verified live: `final_approach_metres_per_pulse` registered with default 1.06.

### Run 1 — 3.0 m straight segment: `target_reached` ✅

The fix works. Target (5.3562, −1.2272) from (8.3562, −1.2272), ~1° of turn
needed, so the run isolates linear control almost perfectly.

| Pulse | Duration | Remaining | m/pulse used | Source | Moved |
|---|---|---|---|---|---|
| 1 | 3500 ms | 3.015 m | 1.06 | default | **0.528 m** |
| 2 | 3500 ms | 2.488 m | 0.528 | observed | 1.108 m |
| 3 | 3500 ms | 1.383 m | 0.818 | observed | 0.997 m |
| 4 | **1593 ms** ← scaled | 0.400 m | 0.878 | observed | 0.370 m |

Landed at (5.3660, −1.3502): **along-track error 1.0 cm**, cross-track 12.3 cm,
total 12.3 cm inside the 0.15 m tolerance. Drove 3.003 m for a 3.015 m target.
No overshoot, no re-aim, no stale feed, no `no_actuation`, 58 refreshes
delivered, BLE held throughout.

Pulse 1 moved only 0.528 m while pulses 2–3 moved 1.108 and 0.997. **The
first reading of this — "the first pulse loses about half a pulse to
spin-up" — was WRONG and is withdrawn; see run 4 below, where the first pulse
moved 1.0159 m.**

**Self-calibration absorbed the low first pulse and erred safe.** It dragged the
running average to 0.878, so the final pulse asked for slightly *less* than
needed and landed 3 cm short of its own prediction rather than past it. That
property is real and holds regardless of what caused the low reading.

**Longitudinal control is now essentially solved; the residual error is
lateral.** 12.3 cm of cross-track over 3.0 m is ~2.3° of heading error.

### Run 2 — return segment: `turn_phase_incomplete` (turn budget, not the fix)

The 176° turn burned all 8 commands at 13.0°/command (179.69° → 75.60°) and the
executor correctly sent **zero** linear commands rather than drive misaligned.

🐛 **Wiring gap found (NOT yet fixed):** `_vio_turn_to_heading` accepts
`motion_refresh_interval_ms` and the standalone service forwards it
(`services.py` ~12057), but the **vector executor's two internal call sites**
(~8684 and ~9090) do not pass it. The 2026-07-25 "refresh wired into
`vio_turn_to_heading`" item was only half done: the service, not the executor.

⚠️ **CORRECTED later the same day:** a first pass also claimed the executor
forced `heading_tolerance_degrees: 3.0` into the turn. **Wrong.** The schema
default is **18.0** on every executor and all four handlers forward it; the 3.0
in the Python signature only ever applies to direct calls (i.e. tests). This
run's turn used 18. Its failure was purely the command budget — 176° at
13°/command needs ~14 commands and the budget was 8.

### Run 3 — the turn A/B: refresh is ~3.8x, and it overshoots 🚨

Standalone `vio_turn_to_heading`, angular 500, refresh 200, 1500 ms pulses.
Clean A/B: `pulse_duration_ms` defaults to 1500 **in the function**, and the
executor does not override it, so both arms used 1500 ms at angular 500. (The
executor's `turn_pulse_duration_ms: 300` is not forwarded to the VIO turn at
all — it belongs to the legacy path.)

| Cmd | Heading | Change | Error before | Notes |
|---|---|---|---|---|
| 1 | 75.60° → 27.36° | −48.24° | 71.98° | good progress |
| 2 | 27.36° → **−23.56°** | −50.92° | 23.74° | **overshot 27°**, `progress_degrees −3.43` |
| 3 | −23.56° → 2.39° | +25.94° | 27.18° | angular **−500**, reversed, 700 ms slow pulse |

⚠️ **METHOD LESSON — a net figure hid a reversal.** The first read of this run
reported "24.4°/command", computed as net 73.2° over 3 commands. The **operator
watching the mower** reported it turned one way then came back — which the
per-command dump confirmed. The parse that produced 24.4° had printed only
`heading_went_fresh` per command (the other keys were filtered out), so
monotonicity was *assumed*, not observed. Net-over-count arithmetic averages an
overshoot away. Same family as the "a zero from a log grep proves nothing if
DEBUG is off" lesson: check the per-item record, not the aggregate.

**Real result: ~49.6°/pulse at refresh 200 (48.24 and 50.92) vs 13.0°/command
single-shot = ~3.8x.** And cmd3's 700 ms slow pulse gave 25.94° → ~37°/s vs
~33°/s for the 1500 ms pulses, so **rotation is now proportional to duration
under refresh**, the same unlock as linear. The old ~8–15° rotation quantum is
gone when refresh is on.

🚨 **The turn phase has exactly the granularity bug the linear phase just had.**
At ~50°/pulse a 1500 ms pulse cannot service a 23.7° error — only stop short or
blow past. The existing guard does not help: `slow_threshold_degrees` is 15°, so
a 23.7° error is *above* it and fires a full 1500 ms pulse; and even the 700 ms
"slow" pulse is ~26° at this rate.

Both arms also ran the same `heading_tolerance_degrees` (18 — see the correction
above), so `motion_refresh_interval_ms` really was the only variable that
differed between them.

**Fix = the same shape as `_final_approach_pulse_ms`:** scale the turn pulse to
the remaining angle, hard-guarded on `refresh > 0`, self-calibrating from
observed degrees-per-pulse. At ~33°/s the 23.7° error needed roughly a 720 ms
pulse. `heading_tolerance_degrees: 18` then becomes the wrong constant — it was
derived from the 13° single-shot quantum and can drop a long way once the pulse
is scaled, which is also what would close run 1's 12.3 cm cross-track error.

🐛 Still unfixed: `final_displacement_m` came back `None` again (the 2026-07-19
turn-path honesty bug), even though per-command `displacement_m` was populated
(0.037 / 0.068 / 0.046 m — the turn drifted ~15 cm total).

### Run 4 — return segment, 3.09 m: `target_reached`, 2.5 mm along-track 🏆

Best result of the project. Same params as run 1 plus `vio_max_realignments: 0`, so no mid-run re-aim goes
through the known-broken un-refreshed turn path. (`heading_tolerance_degrees: 18`
was also passed explicitly, but that was a **no-op** — 18 is already the schema
default; see the correction in the run-2 section.)

| Pulse | Duration | Remaining | m/pulse | Source | Moved |
|---|---|---|---|---|---|
| 1 | 3500 ms | 2.995 m | 1.06 | default | 1.0159 m |
| 2 | 3500 ms | 1.980 m | 1.0159 | observed | 1.0588 m |
| 3 | **3114 ms** ← scaled | 0.923 m | 1.0374 | observed | 0.9264 m |

(5.2628, −1.3058) → landed (8.3587, −1.1418) against target (8.3562, −1.2272):
**along-track error 2.5 mm**, cross-track 8.5 cm, total 8.5 cm. **Turn commands:
0** — the mower was already inside the 18° tolerance. 49 refreshes, no aborts. The scaled pulse asked for 0.9231 m at 1.0374 m/pulse → 3114.5 ms and
delivered 0.9264 m: **prediction error 3.3 mm**.

⚠️ **CORRECTION — the run-1 "spin-up" claim is withdrawn.** Run 1's 0.528 m
first pulse did **not** reproduce here (1.0159 m), and all three pulses were a
consistent ~1.0 m. The difference is what preceded the linear phase: run 1 ran
**6 turn commands**, run 4 ran **0**. So a short first pulse tracks *a preceding
turn*, not spin-up — most plausibly the position feed still catching up after
the turn, truncating the first measurement (run 1's pulse 2 then read 1.108,
above the true rate). **Hypothesis with n=1 each way, not a finding.** To settle
it: run two segments back to back, one preceded by a turn and one not, and
compare first-pulse distance.

**And 1.06 was a good constant after all** — run 4's pulses averaged 1.00 and
self-calibration converged to 1.0374, within 2% of the baked-in default.

## 🚨 2026-07-28: EVERY MAP CONTAINMENT CHECK WAS INERT (fixed `0a591eb4`, DEPLOYED)

The single most important find of the project so far, and it was found by
accident. The operator moved the mower off a fence; asking "what does telemetry
look like near a boundary?" led to measuring the real polygon, which exposed
this.

**`_point_in_polygon` returned True for the entire plane.** So
`validate_custom_path` could never emit
`path_points_outside_known_area_geometry`, and **every "path valid: true" this
project has ever relied on proved nothing about containment** — including the
ones used to justify the 2026-07-27 runs.

Root cause: the mower sends area polygons as **CLOSED RINGS** (first vertex ==
last). `_point_in_polygon` seeds its loop with `previous = polygon[-1]`,
`current = polygon[0]` — the same point, a zero-length segment. In
`_point_on_segment` that made `cross` and `dot` identically zero and reduced the
final check to `0 <= 0 + tolerance`, returning True for any point. Containment
short-circuited to True before casting a single ray.

Live evidence: a target **1.97 m outside every mapped area** was reported inside
**all four** areas at once — including Front Main and Front Right at 19 m and
28 m — and `validate_custom_path` returned `valid: true`, `errors: []`,
`warnings: []`. It was about to be used as a real motion target.

**Why the tests missed it:** all four existing containment tests build **OPEN**
polygons (first vertex != last), so the degenerate closing segment never existed
in a test. The new tests use a closed ring as the device does, and assert both
that an outside point is rejected AND an interior point still accepted (so the
fix cannot degrade to "reject everything"). Both were verified to fail against
the previous behaviour.

**Verified live after deploy:** the same 2 m-outside target now returns
`valid: false` with `path_points_outside_known_area_geometry`. First time that
check has ever done anything.

⚠️ Same shape as the 2026-07-24 `zone_hash`/`bol_hash` defect: a guard that
reads as working while being a no-op. **Worth auditing every other guard on this
basis** — "is there an input shape the real device sends that makes this
vacuously true?"

### `pos_type` is NOT a boundary-proximity signal

Measured 2026-07-28: `pos_type` read **`AREA_INSIDE`** while the mower sat
**79.5 cm** from its mapped area edge. It evidently only flips to
`AREA_BORDER_ON` when actually on/over the line, so it cannot serve as an
early-warning "approaching the boundary" signal. Any such guard must be computed
from the polygon geometry directly.

### Deploy state 2026-07-28 (all 41 files checksum-matched)

Full-directory deploy + restart. API up in 36 s, 121 entities at 129 s, 59
services, `turn_degrees_per_second` registered on both motion services. Also
deployed: the `device_tracker` translation nesting fix, and `manifest.json` now
declaring `camera`/`http`/`web_rtc`.

🪤 **macOS `tar` trap:** `tar czf` smuggled AppleDouble `._*` files into the
tarball and they extracted onto the host (82 files instead of 41). Harmless but
filthy. Use `COPYFILE_DISABLE=1 tar czf …` next time, and verify the host file
COUNT, not just the checksums of files you expected.

🪤 A burst of `Could not find entity lawn_mower.…` errors right after restart is
a **startup race**, not a fault — they stop the instant the platform finishes
loading (90 of them, ending at the exact second the entity appeared).

### 🔴 BLE dropped twice tonight; the motion gate refused both times (correct)

Two `go`-authorised runs were aborted **before sending any command** because the
pre-flight keepalive check read 0. Both aborts were correct, and verified not to
be the DEBUG-logging trap (45 DEBUG lines from `pymammotion.transport.ble` in the
same 5 minutes; last keepalive at 20:44:42, ~18 s before the check).

**Open question, NOT a diagnosis:** no GATT disconnect was logged for the mower
MAC at all — the keepalives simply stopped. Meanwhile 8 ESPHome devices showed
`reconnect_logic` churn in 10 minutes (`esphomes3-irk` 5, `garage-outlet-kmc` 4,
2 unexpected disconnects). That is suggestive of network/proxy instability
rather than the mower's advertising rate, but the causal link is unproven.
**Next time this happens, run `scripts/ble_advert_monitor.py`** — it distinguishes
"mower radio off air" from "link held/killed by something else", which is the
discriminator we lack.

### 🔴 Third attempt: `vio_calibration_failed` — NO ACTUATION (unresolved)

After an operator mower-restart brought BLE back (4 keepalives/20 s, VIO 61
features, blockers `[]`), the run fired and died in the calibration drive:

```
stop_reason  : vio_calibration_failed
reason       : insufficient_calibration_distance
distance_m   : 0.0009055        <- 0.9 MM over 2 pulses
pulses_sent  : 2                 (both command_result ok: True)
vio_state    : 2                 vision_heading 90.077
```

0.9 mm is noise, not movement. The calibration drive runs single-shot (refresh
count 0), which by the H-watchdog finding should still give ~10 cm/pulse. **The
mower accepted the commands and did not move.**

Every indicator green at the time: `MODE_READY`, `lock_state 0`, `fuse_status 1`,
`bumper ok`, `last_error_code 5002` stale from the afternoon mow, battery 67%.

**This is the 2026-07-19 invisible-e-stop signature** — commands accepted,
nothing moves, all health fields green, `lock_state` is NOT e-stop. Operator was
asked to check the physical e-stop. **UNRESOLVED at session end.**

Other candidates not ruled out: a post-restart state that accepts but ignores
motion; a physical obstruction. Note the mower had been restarted moments
earlier, so "needs longer to become motion-ready after restart" is a live
hypothesis worth testing before assuming e-stop.

**The turn-scaling regression test (`d39e3cdd`) is therefore STILL UNVERIFIED on
hardware.** Three `go`-authorised attempts tonight: two aborted at the BLE gate
before sending anything, one reached the mower and found no actuation. The code
is deployed and unit-tested; nothing about it has been proven live.

## 🚨 2026-07-28 BLE ROOT-CAUSE LEAD: the keepalive stops SILENTLY (top priority)

The best lead yet on the BLE instability, and it reframes it. **The link is not
dropping — the keepalive task stops.**

`pymammotion` holds the GATT link with a `todev_ble_sync(2)` heartbeat every
`_KEEP_ALIVE_BLE_INTERVAL` (5.0 s), so a healthy link shows **11-12
`BLETransport send` per minute**. Measured tonight:

```
20:42  2     20:47 16     20:52 27
20:43 11     20:48 11     20:53  3   <- LAST SEND 20:53:12.727
20:44  9     20:49 12     20:54  0
20:45  0 <-  20:50 34     20:55  0
20:46  7     20:51 11
```

When it stops:
- **No GATT disconnect is logged for the mower MAC.** Last GATT event was a
  reconnect at 20:46; nothing at 20:53.
- **No error, no exception, no task-cancellation** anywhere in the 20:53:05-40
  window.
- **Reports were still arriving 5 s before** ("Manually updated mammotion data",
  "activity mode 11" at 20:53:07).
- `sensor.<mower>_active_transport` **keeps reporting `ble`** and goes stale —
  it does NOT reflect the stall. Another field that must not be trusted.
- RSSI is NOT the cause: proxy-measured **−64 dBm** on `p1s-printer-a5774c` at
  the time, well above the ~−70 working threshold and far from the ~−76 wall.

**This explains all three failed motion attempts tonight**, including the one
that reached the mower: the two `send_movement` commands were accepted
(`ok: True`) and never transmitted — the transport was silent 20:47:25-20:47:52,
straight through the run — and the mower moved **0.9 mm**, which is the correct
response to being told nothing.

⚠️ **`command_result.ok` proves NOTHING about delivery.** It only means the send
call did not raise (`needAck=false`). Tonight it read `True` for commands that
never went on the wire. Any future "the mower did not move" diagnosis MUST check
`BLETransport send` timestamps spanning the command window before blaming the
mower, an e-stop, or actuation.

**What is NOT yet known:** why the keepalive stops. No evidence gathered yet for
the loop dying vs. being starved vs. an await that never returns. Next session,
off-mower: read `pymammotion`'s keepalive task (`ble_loop` /
`_KEEP_ALIVE_BLE_INTERVAL`), find whether its exception path is swallowed, and
add a transport-side watchdog that detects "no send in >15 s while nominally
connected" and forces a reconnect. That is also the honest fix for the motion
gate, which currently only samples the last 20-25 s.

Recovery observed: an operator mower-restart brought it back once, and an
automatic proxy reconnect at 20:46 brought it back once.

### ✅ The mower and BLE delivery are both FINE (proven by the reverse test)

With the transport alive, two `manual_velocity_pulse_test` backward pulses ran
clean and were measured by **RTK position, which works in full darkness** (VIO
was dead: this is the tool for night-time linear tests):

| pulse | duration | moved | implied rate |
|---|---|---|---|
| 1 | 4000 ms | 1.250 m | 0.313 m/s |
| 2 | 2400 ms | **0.938 m** | 0.391 m/s |

⚠️ **Distance is NOT proportional to duration** — 60% of the duration gave 75%
of the distance. Two points fit ≈ a fixed 0.47 m component plus 0.195 m/s.
**Both `_final_approach_pulse_ms` and `_turn_final_approach_pulse_ms` assume
pure proportionality**, so if a fixed component is real, a scaled short final
pulse systematically OVERSHOOTS and both need an intercept term.

**n=2, NOT a finding.** Both could be settle-lag artifacts: in each run the feed
showed only 20-79% of the final distance at `after_stop` and did not settle
until +3 s. The planned duration sweep (1600/1000/700/500/400/300 ms, alternating
direction, RTK-measured) aborted at pulse 1 on the BLE stall. **Re-run it** — it
answers both the proportionality question and the linear actuation floor, and it
does not need daylight.

### PR #10 CI: green except one deliberate deferral (2026-07-27)

`python` **passes** — ruff clean, format clean, mypy clean, 370 tests, 42%
coverage. It had never been green: `requirements_test.txt` pinned homeassistant
2025.3.1 and pymammotion 0.5.3 (against 0.8.8 shipped) and could not even
resolve, so CI failed before running a single check.

`hacs` is **skipped on forks**. It validates repository *publishability* —
topics, issue tracker, license — which no code change can satisfy on a fork:
a fork of an unlicensed upstream cannot add a license, that not being ours to
grant. Upstream carries neither a license nor topics either, so the job is red
there too and has never passed for this project. Left running on the publishing
repo. The fork's issue tracker was enabled (a real gap).

`hassfest` is down to **one class of error, deliberately deferred**: uppercase
ENUM translation keys. HA requires `[a-z0-9-_]+`, and the codebase uses
`MODE_READY`, `AREA_INSIDE`, `ENGLISH` and so on. Fixing it means changing the
keys **and the sensor state values that produce them**, across `strings.json`,
all 12 locale files, and the sensor code — a breaking change for anyone with
automations or dashboards keyed on the current states. Not something to bury in
a VIO feature PR. **Its own scoped task**, and per CLAUDE.md every locale file
must move together.

Two related fixes that did land: `manifest.json` now declares `camera`, `http`
and `web_rtc` (all imported, none declared — `camera` in particular is what
pulls `PyTurboJPEG`), and keys are sorted per hassfest.

⚠️ The workflow now derives HA **component** requirements by walking the
installed homeassistant manifests, because `pip install homeassistant` ships core
only. Do not re-add them by hand to `requirements_test.txt` — hand-pinning is
exactly what rotted this workflow, and a stale pin there conflicts with what
HA's own manifests ask for.

### Ready-to-run: next daylight session, in order

Pre-flight before EVERY real-motion call (the operator gives a fresh "go" each
time): blade off, area clear, and **BLE verified by keepalive traffic**, not by
`ble_rssi` or `active_transport`:

```bash
set -a && source .env && set +a
scripts/ha_ssh.exp 'docker logs --since 25s homeassistant 2>&1 | grep -cE "BLETransport send"'   # want >= 2
```
(after any HA restart, re-enable the logger first — it resets:
`logger.set_level` with `pymammotion.transport.ble: debug`)

**0. Deploy + restart**, then confirm `turn_degrees_per_second` is registered on
both `vio_turn_to_heading` and `raw_pymammotion_execute_vector_segment`.

**1. The 176 deg return segment that failed** — the direct regression test for
the turn scaling and the executor forwarding, together:
```yaml
service: mammotion.raw_pymammotion_execute_vector_segment
data:
  entity_id: lawn_mower.back_yard_clip_skywalker
  points: [{x: <current_x>, y: <current_y>}, {x: <target_x>, y: <target_y>}]   # validate_custom_path FIRST
  dry_run: false
  confirm_blades_off: true
  confirm_clear_area: true
  motion_refresh_interval_ms: 200
  linear_pulse_duration_ms: 3500
  waypoint_tolerance: 0.20
  heading_tolerance_degrees: 18
  max_linear_pulse_ceiling: 6
  sample_delays: [0, 3]
```
Expect: the turn now completes instead of `turn_phase_incomplete`, and the run
reaches its linear phase. Check each turn command's `final_approach` block.

**2. Find the real turn-pulse floor** (replaces the 400 ms guess). Step
`pulse_duration_ms` and watch where rotation stops tracking duration:
```yaml
service: mammotion.vio_turn_to_heading
data:
  entity_id: lawn_mower.back_yard_clip_skywalker
  target_vision_heading: <current + 30>
  angular_speed: 500
  motion_refresh_interval_ms: 200
  pulse_duration_ms: 700      # then 500, then 400, then 300
  heading_tolerance_degrees: 5
  max_commands: 2
  dry_run: false
  confirm_blades_off: true
  confirm_clear_area: true
```
Read `measured_change_degrees` per command; rotation should stay ~33-37 deg/s
until it doesn't.

**3. Re-derive `heading_tolerance_degrees`** — only after 2. This is what closes
the residual cross-track error (8.5-12.3 cm over 3 m = 1.6-2.3 deg).

**4. Settle the first-pulse question** — two segments back to back, one preceded
by a turn and one not, comparing first-pulse distance. Run 1 gave 0.528 m after
6 turn commands; run 4 gave 1.0159 m after 0.

### Night addendum — actuation survives a blind VIO; measurement does not

After full dark (`vio_brightness 0`, `vio_tracked_features 0`,
`visual_positioning_status SIGNAL_NONE`) a single blind `send_movement` at
angular 500 via `raw_pymammotion_motion_probe` **did rotate the mower** (operator
observed "a little"). So **VIO darkness blocks the closed loop, not the drive** —
the motors do not care about light. Everything that fails at night fails because
nothing can *measure* the result.

Worth knowing for the service inventory: on the deployed build, **no service can
do a bounded, explicitly-stopped angular pulse at >=382 without a VIO gate.**
`vio_turn_to_heading`/`vio_turn_probe` gate on VIO; `manual_velocity_pulse_test`
caps angular at ~202 (below this mower's rotation threshold);
`raw_pymammotion_angular_calibration` closes the loop on `toward`, which freezes
during a pivot; and `raw_pymammotion_motion_probe` **never sends a stop** — it
relies on the mower's own H-watchdog self-halt. There is also no standalone
motion-stop service to pair with it. If blind repositioning ever needs to be a
supported operation, that gap is the thing to close.

⚠️ Telemetry could not confirm the rotation and was never expected to: `toward`
freezes during a pivot and the position feed polls every 5 min when idle, so
identical samples 3 s apart mean nothing. Operator observation was the only
feedback channel.

### Ops note — I caused a BLE drop by trusting `ble_rssi`

`active_transport` read `cloud_aliyun` and `ble_rssi` read 0, so BLE looked
dead and the BLE switch was toggled. The log showed `BLETransport send: 27 bytes`
every ~5 s right through that moment: **BLE was fine, and the toggle is what
disconnected it** (`error=0`, a clean local disconnect). The advertisement
monitor then confirmed the radio had never slept (−62 dBm via
`p1s-printer-a5774c`). `ble_rssi` is self-reported and stale — the rule already
in the memory file — and it got trusted anyway. **Verify BLE with keepalive
traffic in the log, not with `ble_rssi` or `active_transport`.**

Also: after an HA restart, `pymammotion.transport.ble` is NOT at debug unless
explicitly set, so a `grep -c "BLETransport send"` returning 0 means nothing.

## Immediate next steps (all doable off-mower)

> **STATUS 2026-07-25 (end of the later off-mower session).** The whole off-mower
> queue is **clear**: the turn-phase stale-feed detector, refresh wired into
> `vio_turn_to_heading`, the vector-executor echo fix, and the five-coordinator
> dead-region fix are all done (352 tests, mypy + ruff clean, **not deployed**).
> The BLE investigation is root-caused (see the 2026-07-25 "later" section).
>
> **Everything still open needs the mower, daylight, and an operator:**
> 1. **Deploy + restart** (`services.py`, `services.yaml`, `coordinator.py`), then
>    run the three confirmation checks in the plan doc's item-4 section.
> 2. **Step 5b — the supervised segment run**, still unachieved; the Task-2
>    constants (pulse-geometry ceilings, `min_progress_distance`, cadence) remain
>    un-re-derived hypotheses.
> 3. **Re-derive `heading_tolerance_degrees`** with
>    `motion_refresh_interval_ms: 200` passed explicitly to
>    `vio_turn_to_heading` (it defaults to 0 precisely because 18 was derived from
>    the single-shot quantum).
>
> **SUPERSEDED 2026-07-27 — see the on-mower section above.** Item 1 (deploy) and
> item 2 (the supervised segment run) are **done**: a 3.0 m segment reached
> `target_reached` with 1.0 cm along-track error. Item 3 is now better specified:
> refresh gives ~3.8x on turns (~49.6°/pulse) and rotation became proportional to
> duration, so the turn needs a **scaled final pulse** before the tolerance is
> re-derived — tuning 18 alone cannot fix a 50° granularity. The top off-mower
> queue is now:
> - ~~**(a) Scale the turn pulse to the remaining angle**~~ **DONE `d39e3cdd`.**
> - ~~**(b) Forward `motion_refresh_interval_ms` into both
>   `_vio_turn_to_heading` call sites**~~ **DONE `d39e3cdd`.**
>   (`heading_tolerance_degrees` turned out to be forwarded already; a test now
>   pins it.)
> - ~~**(c) Fix `final_displacement_m: None`**~~ **DONE `d39e3cdd`.**
>
> **All three shipped 2026-07-27 in `d39e3cdd` (370 tests, mypy + ruff clean,
> NOT DEPLOYED).** `_turn_final_approach_pulse_ms()` calibrates on a **rate**
> (deg/s) rather than degrees-per-pulse, so samples at different pulse lengths
> stay comparable and a scaled pulse is still a valid sample; it only accumulates
> from pulses whose heading went fresh, because a latched sample reads ~0 deg for
> a pulse that really turned and would collapse the rate. Scaling is layered on
> top of the existing slow-pulse safety cap, never instead of it. The end-to-end
> test was verified to fail against the old behaviour with `no_heading_progress`
> — the live overshoot-reverse signature.
>
> ⚠️ **`_MIN_SCALED_TURN_PULSE_MS = 400.0` is NOT proven.** The shortest turn
> pulse ever measured is 700 ms, and the single-shot path had a hard actuation
> floor (a 2000 ms single-shot pulse was a measured physical no-op). Rotation is
> proportional to duration under refresh, so a shorter pulse *should* just turn
> less — but nobody has found the refreshed path's floor. **To prove it:** run
> `vio_turn_to_heading` at refresh 200 / angular 500 with `pulse_duration_ms`
> stepped 700 → 500 → 400 → 300 and find where rotation stops tracking duration.
>
> **Next mower session, in order:**
> 1. Deploy `services.py` + `services.yaml`, restart, confirm
>    `turn_degrees_per_second` is registered on both services.
> 2. **Re-run the 176° return segment that failed** — it should now complete the
>    turn and reach its linear phase. This is the direct regression test for (a)
>    and (b) together.
> 3. Step the pulse floor down (above) to replace the 400 ms guess with a
>    measurement.
> 4. **Then** re-derive `heading_tolerance_degrees`; with a scaled pulse it
>    should drop well below 18, which is what closes the residual cross-track
>    error (8.5–12.3 cm over 3 m ≈ 1.6–2.3°).
> 5. Settle the first-pulse question: two segments back to back, one preceded by
>    a turn and one not, comparing first-pulse distance.
> 4. **File the pymammotion reassembly patch** —
>    `docs/pymammotion-ble-reassembly-bug.md` has a ready-to-file diff. It cannot
>    land here: pymammotion is a pinned PyPI release (`==0.8.8`), not a fork.

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
   the 2026-07-25 on-mower section below.

   **✅ CODE DONE 2026-07-25 (later, NOT deployed): `motion_refresh_interval_ms`
   is now wired into `vio_turn_to_heading`** (schema + services.yaml + handler,
   via the existing `_motion_refresh_window`; refreshes counted separately in
   `motion_refresh_commands_sent` so they never inflate `commands_sent`).
   **Left defaulting to 0 on purpose.** `heading_tolerance_degrees: 18` exists
   only because single-shot turning was quantised into ~8–15° steps; enabling
   refresh by default before re-deriving it would drive continuous rotation into
   a deadband sized for discrete steps. **Re-deriving the tolerance needs a mower
   session** — pass `motion_refresh_interval_ms: 200` explicitly and measure.

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
   sync automatically** (the unreachable-block bug). That was the real defect,
   and it is **✅ fixed as of 2026-07-25 (later, NOT deployed)** — see the item-4
   section. Confirm on hardware after deploying: edit an area on the mower and
   check `map_sync_status` converges without a restart or a `sync_maps` press.

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
turned fine"* (tonight).

**✅ FIXED 2026-07-25 (later, NOT deployed) — but not the way this note
proposed.** The prescription was to gate on `heading_went_fresh: false`. That
**does not discriminate**: `heading_went_fresh` is True only when before/after
differ by more than the epsilon, which is exactly when
`_streak_shows_no_actuation` (bit-identical heading) is False. The two are
perfectly correlated, so gating on it would have deleted the no-actuation branch
rather than refined it.

The signal that works is *"did any channel move at all"* — a live feed is never
perfectly still (position jitters ~2–4 mm; a dusk-latched heading still emits
~0.0018° noise). New `heading_poll_count` / `heading_poll_feed_alive` per pulse
feed `_streak_shows_dead_telemetry`, which fires only when heading **and**
position were bit-identical across every poll. New reason:
`vio_telemetry_stream_stale`. Verified to fail with the fix reverted; the two
dusk-latch tests pass untouched.

**Note the semantic change:** a replay of the 07-19 e-stop run now reports
`vio_telemetry_stream_stale`, because that run's feed was frozen too (heading
bit-identical for 45 min). That is the honest answer — telemetry never saw the
e-stop, the operator did. `no_actuation_detected` now means the link was
demonstrably alive and the mower still did not move.

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
  (it echoes `None` even when passed and honoured). The multi-segment executor
  got this echo gap fixed on 2026-07-19; the vector one still had it.
  **✅ FIXED 2026-07-25 (later, NOT deployed)** — the vector executor now echoes
  `max_linear_pulse_ceiling`, `turn_pulse_duration_ms`,
  `linear_pulse_duration_ms`, `vio_turn_max_commands`, `vio_angular_speed` and
  `vio_heading_offset_degrees`, with a regression test over all six.
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

## 🚨 2026-07-25 (later): BLE root-caused — the mower barely advertises

Full detail in `docs/codex-working-plan.md` (2026-07-25 "later" section). The
headline, measured with HA's own advertisement stream rather than inferred:

**In a 30-minute window while the mower was actively mowing, HA's six scanners
heard exactly TWO advertisements from it** — one burst, at −76 and −97. Positive
control in the same session: 444 advertisements from 107 other devices in 45 s.
A normal connectable peripheral advertises every 20 ms–1.28 s; this one emits
roughly one burst per ~10 minutes.

That single fact produces every `never seen by any scanner` / `last advertisement
613s ago` failure. **Retire the "−70/−76 coverage wall" as the primary
explanation** — when the mower *is* heard it is heard at −62 to −69, and four of
six proxies have all connection slots free at the moment of failure.

Four separate problems were being treated as one:

1. **The mower is not advertising** (6 of 8 cooldowns): no proxy connect is even
   attempted. Root cause; not fixable from our side.
2. **A connect is attempted at −64 and hangs ~20 s, then `status=133`
   (`ESP_GATTC_OPEN_EVT in DISCONNECTING state`)** — both instances on
   `p1s-printer-a5774c`. Not coverage.
3. **HA keeps preferring the proxy that just failed.** The failure penalty is
   negligible next to RSSI (score −67.06 with 2 failures vs −67.00 with 0). The
   P1S proxy is closest to the mower and is the only one never showing a free
   allocation (`slots=2/3 free` in every sample).
4. **Corrupted frames are a pymammotion reassembly bug**, root-caused below.

**Ops action worth trying before any code change:** park the mower nearer
`esphomes3-irk`, or reduce what the P1S printer proxy carries.

### 🐛 The corrupted frames — root cause found (upstream, pymammotion)

The garbage decodes as protobuf: `"a1LLmy1zc0j"` (Aliyun product key) and
`"Luba-VSPLV397"` (device name) — a device-identity message **spliced into** a
report frame, not bit-flips.

`BleMessage.parseNotification` accumulates fragments into a buffer that
`_notification_handler` clears **only on a complete frame**. A lost fragment,
checksum failure, or exception returns early and leaves the partial buffer in
place; the next message appends to it and the concatenation is delivered as one
"complete" frame. The sequence check even *detects* the gap and only resyncs the
counter — it never discards the poisoned buffer.

**Live: `parseNotification read sequence wrong` fired 11 times in ~3 minutes of
connected BLE.** One lost packet mid-fragment costs at least two reports — this
is the mechanism behind the 07-25 turn reporting bit-identical `vision_heading`
*and* `displacement_m` while the mower physically turned.

*Fix (upstream): reset the accumulation buffer on a detected sequence gap and on
the checksum/exception paths.*

### 🐛 Our own code contributes: the dead region also contains the BLE reconnect

`_async_opportunistic_ble_reconnect()` sits in the **same unreachable block** as
the map-sync check (past `if data := await super()._async_update_data(): return
data`). Proven with DEBUG on: `Finished fetching mammotion data in 0.000 seconds
(success: True)` on every tick while `Updated Mammotion device` — three lines
past the early return — appears **zero** times.

That function was written for exactly this symptom ("stuck on `cloud_aliyun` at
healthy RSSI with the cooldown long expired"). **It has never run.** So when one
of the mower's rare advertisements lands, HA *does* immediately push a fresh
`BLEDevice` (that path works), but nothing then calls `connect()`. Observed:
advertisement burst 18:56:56 → `active_transport: ble` at 19:04:47, ~8 minutes
of usable link discarded.

**This makes off-mower item 4 (make the per-tick block reachable) the
highest-value BLE fix available**, and it now governs two subsystems, not one.

**✅ DONE 2026-07-25 (later still, NOT deployed).** The bug was in **five**
coordinators, not one, and the map-sync block turned out to live in
`MammotionMapUpdateCoordinator` (60 min), not the report coordinator — the older
notes had that wrong. The base method is now
`_async_short_circuit_update() -> DataT | None`, returning `None` for "carry on",
and all five call sites test `is not None` rather than truthiness. Every dead
region was already individually guarded, so turning them on is contained. 6 new
tests (one verified to fail against the old return, plus an AST test pinning
`is not None` at all five sites); 352 tests, mypy + ruff clean.

**Runtime effect to expect after deploy:** the BLE reconnect attempts every
5 min instead of once per HA start, and the map-sync check runs every 60 min
(which is what makes `_should_start_map_sync`'s back-off load-bearing, and what
finally fixes "a device-side map edit is never picked up until a restart").
Confirmation steps are listed at the end of the matching plan-doc section.

*Checked and cleared: the advertisement callback registration is fine —
`_async_start()` registers it; the copy in the dead region is a redundant backup.*

### 🚨 …and the config option `prefer_ble_over_wifi` was `false`, which gated the fix

Read from `/config/.storage/core.config_entries` (the options form defaults
`prefer_ble` to `True`, so an unchecked box meant it was explicitly `False`).
What the flag actually does differs per call site:

| site | uses | effect when `false` |
|---|---|---|
| `active_transport()` (handle.py:1883) | `self._prefer_ble` | **none** — its docstring says it "no longer affects which transport is returned"; it only biases a log de-dup key. This is why `active_transport: ble` still appeared. |
| `_do_send()` (handle.py:772) | `self._prefer_ble`, no override | **reconnect-on-send disabled** |
| `send_raw()` (handle.py:1668) | caller's value if given, else `self.prefer_ble` | our motion services pass `prefer_ble=True` explicitly, so **they** still reconnect |

That last row explains the whole observed pattern: BLE recovered when something
explicitly asked for it (a motion service, a probe) but not from routine
background traffic.

**The coupling that matters:** the two *automatic* reconnect paths — pymammotion's
reconnect-on-send and our `_async_opportunistic_ble_reconnect` — were **both**
disabled, and not independently: our function also short-circuits on
`handle.prefer_ble`. **So the item-4 fix alone would have changed nothing on this
system.**

**✅ The operator set `prefer_ble_over_wifi: true` on 2026-07-25 at 20:46:13
local.** Expect the log to look *busier*: `is_usable` can be True on a
cached-but-stale `BLEDevice`, so attempts fire at a non-advertising mower and fail
into the 120 s cooldown. That is the intended bounded retry.

**Leave `movement_use_wifi` OFF** — `services.py` never reads it (verified: zero
references); only `button.py` does, where it makes `_nudge_available` return True
unconditionally and routes the nudge buttons over cloud. Cloud-routed motion has
no position feedback (the feed is BLE-only), so it is blind driving, and it masks
the "BLE selected but not usable" signal that exposed the 07-19 gate bug.
`mow_path_fetch_enabled` has no BLE effect (it gates `MowPathSaga` over MQTT).

*Next measurement: re-run `ble_advert_monitor.py` with the mower **docked and
idle** before treating ~10 minutes as the real duty cycle. One evening, one
mower, one window.*

## Known walls (not code bugs)

- **BLE: the mower's advertising duty cycle is the hard limit** (measured
  2026-07-25, above). The older framing below held that proxy coverage was the
  wall; it is at most a secondary factor.
  - Historical note: works above ~-70 rssi, dies below ~-76 (ESPHome GATT
    status=133 → 120 s cooldown). The status=133 failures observed on 2026-07-25
    happened at **−64**, so status=133 is not a signal-strength tell.
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
