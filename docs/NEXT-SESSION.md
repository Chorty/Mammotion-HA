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
- **VIO needs daylight.** It will not initialize in a dark scene; the gates
  refuse rather than drive blind. Check `camera_brightness` is not `Dark` and
  `track_feature_num` is healthy before any VIO run.

## The current frontier — app-parity motion cadence (2026-07-20)

A JADX decompile of the Mammotion app (2.3.8.19) showed that
`CarRemoteControlManage2` re-sends the **identical** `DrvMotionCtrl` movement
command every **200 ms** for as long as the on-screen stick is held, then stops
by sending zero speeds before cancelling its timer. `sendControl` builds exactly
what pymammotion's `send_movement` builds, with `needAck=false`.

Our executors have always sent that command **once** and slept out the pulse.
That is the leading explanation for the tape-measured behaviour: a fixed ~4in
step regardless of pulse duration (2s → 0", 4s → 4", 6s → 4"), the ~8-15°
rotation quantum, and a 3.5 m path needing ~35 pulses — longer than the BLE link
survives.

Also learned, and true regardless of how the cadence test lands:
- **`needAck=false`** — our per-pulse `command_ok` has never been evidence of
  delivery. Consistent with the 07-19 e-stop incident (below).
- **Speed scale**: the app's full scale is ±850 linear / ±382 angular (15% stick
  deadband, ×10 linear, ×4.5 angular). We drive linear **400** (~47% throttle)
  and angular **500**, which is *above* anything the app can produce and may be
  clamped — so "angular is weak, use 500" may be an artefact.
- **No manual-mode entry is needed to drive.** `DrvMowCtrlByHand` belongs to the
  manual *mowing* flow (blade + speed cap); the joystick path calls `sendControl`
  and nothing else.

**Still unproven:** that the *firmware* enforces a refresh timeout. No such
constant is in the APK (it would live in mower firmware), and the app would look
identical if it repeated purely to track a moving stick.

## Immediate next step — the tape A/B (needs daylight + operator)

Code is in and opt-in (`motion_refresh_interval_ms`, 0 = proven single-shot).
Run the same pulse twice and measure both with a tape:

```yaml
service: mammotion.manual_velocity_pulse_test
data:
  entity_id: lawn_mower.back_yard_clip_skywalker
  action: forward
  duration_ms: 4000
  motion_refresh_interval_ms: 0      # then re-run with 200
  dry_run: false
  confirm_blades_off: true
  confirm_clear_area: true
```

- **If refresh-mode travels several times further**, the cadence was throttling
  every run: adopt it, then re-derive throughput, turn tolerance and
  `min_progress_distance`, and build the hold-to-drive joystick card on top.
- **If both go ~4 inches**, the quantum is something else — pivot to the speed
  scale (try full throttle, and angular at the in-range 382).

Note `manual_velocity_pulse_test`'s `speed` is already on the app's 0.0–1.0
scale (it goes through the coordinator's directional helpers), *not* the raw 400
our executors send. The `app_speed_scale` block in the result shows what it
resolved to.

## Known walls (not code bugs)

- **BLE proxy coverage is the hard limit.** Works above ~-70 rssi, dies below
  ~-76 (ESPHome GATT status=133 → 120 s cooldown). Long runs outlive the link.
  A faster drive may dodge this on its own.
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
- **Trust the tape, not the telemetry, for distance.** The map-local feed lags
  ~4 s, updates in jumps, and has a ~2-6 cm absolute noise floor.
- Gate every change: `uv run pytest`, `uv run mypy custom_components/`,
  `uv run ruff check`.

## Repo gotcha

A GitHub Actions workflow auto-pushes version-bump commits and **regresses** the
version (was beta11 in June, back to beta7 by July). Standing rule: keep the
higher beta when resolving; don't trust the manifest version to reflect what's
deployed (deploy is by file copy + md5, independent of the version string).
Consider disabling/fixing that workflow.
