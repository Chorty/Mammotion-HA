# Next-session handoff — click-to-path (guarded mower motion)

Single "start here" for resuming this work on any machine. Deep detail lives in
`docs/codex-working-plan.md` (see the dated sections, newest last). Host
connection details and credentials are **not** in git — they're in `.env`
(gitignored) at the repo root; recreate `.env` on a new machine from your own
records.

## Where we are

- **Phase 1 (straight-line click-to-path): done, deployed, live-validated.** The
  card can drive near-straight guarded segments to tolerance via a bounded,
  explicitly-stopped pulse loop (`raw_pymammotion_execute_vector_segment`).
- **Two safety bugs found and fixed this session:**
  1. The guarded primitives never sent an explicit stop after `send_movement`
     (a continuous-velocity command with no duration bound), so a single "pulse"
     could travel ~7× its intended distance. Now each pulse sleeps its duration
     then calls `async_stop_manual_motion`. Live-verified (10/10 clean stops).
  2. The BLE-transport gate compared `str(TransportType.BLE)` (=`'TransportType.BLE'`)
     to `'ble'` and blocked every real run; now uses the coordinator's normalized
     `active_transport_state`.
- **Phase 2 (turning): blocked by a design finding — see below.**

## The turning blocker (read before touching turns)

In-place rotation is **not observable** in this mower's telemetry:
- `toward` / `location.orientation` is **course-over-ground** (direction of
  travel), so it stays frozen during an in-place pivot. Confirmed live: the mower
  physically pivoted while `toward` stayed bit-identical across 5 pulses.
- Raw `send_movement(0, angular)` produces no visible rotation in the bounded
  regime (it only "worked" before the stop fix because firmware ran it long).
- Candidate absolute-heading fields are dead at rest: `location.RTK.yaw`=0,
  `report_data.vision_info.heading`=0 (`vio_state`=0, VIO inactive),
  `nav_heading_state.heading_state` is a status enum (not an angle).

So the original "accumulate weak turn pulses until `toward` reaches target" plan
is invalid — the feedback signal can't see the motion. Diagnostic paths for all
heading candidates were added to `_RAW_POSITION_PATHS` and are deployed; read them
via the `position_feedback_diagnostic` service (no motion).

## Immediate next step

**Run the VIO-during-motion test.** VIO odometry may only initialize once the
mower is moving; if `report_data.vision_info.heading` comes alive during a drive
and tracks rotation, we can rebuild turns on it. Starter prompt:

> Continue the Mammotion click-to-path work — Phase 2 turning. Run the
> VIO-during-motion test: fire one supervised forward pulse while capturing
> `report_data.vision_info.heading`, `vio_state`, and `location.RTK.yaw`
> before/during/after, to see if VIO initializes during motion and could serve as
> a rotation-feedback signal. Context is in `docs/NEXT-SESSION.md` and
> `docs/codex-working-plan.md`. I'll be watching the camera.

Then pick the turning approach:
- **(A) VIO-heading feedback** — if VIO tracks rotation, rebuild the turn
  primitive on `vision_info.heading` instead of `toward`.
- **(B) Arc-based turns** — if no motion-independent heading exists, execute turns
  as curved motion (linear + angular together) so course-over-ground updates and
  can serve as feedback, at the cost of turns needing room to arc.

## Live-testing workflow (essential)

- **Deploy code by copying to the HA host, then RESTART HA Core.** A config-entry
  reload does NOT reload changed Python — you must restart. Restart needs an
  explicit operator "restart HA".
- **Real motion requires BLE** (a gate refuses cloud). BLE is flaky here; after a
  restart it comes back on cloud — toggle the mower's `bluetooth` switch off→on
  and confirm it holds. Rapid toggling trips a ~120s BLE reconnect cooldown.
- **Every real-motion command needs a fresh operator "go"**; re-check state
  (paused, blade OFF, BLE, valid-for-motion) right before each fire.
- **Keep the mower in open space, clear of the dock** — dock obstruction masks
  motion and looks like frozen telemetry.
- Gate every change: `uv run pytest`, `uv run mypy custom_components/`,
  `uv run ruff check`.

## Repo gotcha

A GitHub Actions workflow auto-pushes version-bump commits and **regresses** the
version (was beta11 in June, back to beta7 by July). Standing rule: keep the
higher beta when resolving; don't trust the manifest version to reflect what's
deployed (deploy is by file copy + md5, independent of the version string).
Consider disabling/fixing that workflow.
