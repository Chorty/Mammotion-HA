# Real Go throughput hardware check — 2026-08-14

This record separates the measured first run from the correction inferred from
it. The raw per-command service response is
`docs/evidence-beta32-4segment-20260814T202705Z.json`.

## Measured first run

- Authorized path: `(5.0068, -3.0411)` to `(4.5118, -3.5361)`, requested
  distance 0.700036 m, `turn_mode: "vio"`.
- The mower began `MODE_READY`, RTK Fix, inside Backyard Right, with blades
  off. The opening heading was already within tolerance, so no turn command was
  sent.
- One VIO calibration drive and one linear pulse succeeded. The linear pulse
  sent six 200 ms refresh writes over 1294.161 ms, stopped successfully, and
  obtained settled position feedback after 2.030 s / two polls.
- The settled position `(4.7140, -3.3575)` was 0.269783 m from the target.
- The second linear attempt was refused before dispatch in 0.064 ms with
  `RuntimeError: BLE link is not ready for motion:
  command_queue_backlogged`. No second linear movement write was sent.
- The executor returned `command_failed`; the harness returned
  `segment_failed`. It disarmed in `finally`. Independent readback showed
  `real_motion_allowed: false`, no active session, `MODE_READY`, RTK Fix, and
  blades off.

This is a safe failure, not a landing result. The measured 0.269783 m residual
is where the safety gate stopped the run after one successful forward pulse.

## Inference and correction

Reusing settled telemetry successfully removed the blind post-pulse three-second
sample delay, but the report requests used to settle position could still be in
the BLE command queue when the next movement was attempted. The immediate
`command_queue_backlogged` refusal is evidence for that queue timing; it is not
evidence that the BLE link disconnected.

The local correction runs the existing bounded BLE queue-settle check after VIO
position feedback. It returns immediately when the queue is empty, polls only
the existing transient backlog/paused states, never overrides another liveness
failure, and refuses the next command as
`ble_link_not_ready_after_feedback` if the queue does not drain. Each result
records `post_feedback_queue_settle`. Night and legacy branches are unchanged.

Off-mower verification after the correction personally produced 668 pytest
passes and 46 frontend passes. Ruff check, Ruff format check, mypy, and all nine
pre-commit hooks passed.

## Corrected motion-disabled deployment

The corrected tree was deployed to the Home Assistant host and restarted with
experimental motion disabled.

- Backup: `/config/mammotion-backup-20260814-pre-queue-settle.tgz`
- Deployment archive SHA-256:
  `ec46d8fb0fce6aefcf7b2032c88a17b0693cf14cdd5bce086fb9396da5674b5d`
- Deployed `services.py` MD5: `d6ab89ff4e4286b33d4fa5755bba5b0d`
- Card MD5 at both serving paths: `4846aa9b6f9e0eefe67cb95f8326c3ba`
- Manifest MD5: `ef8d5273c8e9730bc964579efb63116a`
- HA API returned after 30 s; 132 Mammotion entities were present after 116 s.
- Container `pymammotion`: `0.8.12.post1`.
- Post-restart gate: disabled, `real_motion_allowed: false`, no active session,
  `MODE_READY`.

Read-only preflight then measured RTK Fix at `(4.7113, -3.3620)`, inside
Backyard Right, BLE RSSI −64 dBm, VIO Light with 80 tracked features, mower
ready, and blades off. A dry run proposed the next 0.70 m path from
`(4.711, -3.362)` to `(4.216, -3.857)`, with an approximately 2° opening turn.
That preview sent no movement and grants no authorization for another run.

## Corrected supervised run

The operator separately authorized the previewed path. The complete per-command
response is `docs/evidence-beta32-4segment-20260814T204300Z.json`.

- Executed path: `(4.7113, -3.3620)` toward `(4.2163, -3.8570)`, 0.70 m.
- Result: `target_reached` in 19.2 s; one segment passed with 0.093100 m landing
  error against the accepted 0.15 m tolerance.
- Commands: one VIO calibration drive, zero turns, and three forward pulses.
  All movement commands and all mandatory stops succeeded.
- The three forward pulses sent 6 / 2 / 1 refresh writes and moved 0.358613 /
  0.104511 / 0.073792 m respectively.
- Each settled-position record replaced the requested `[0, 3]` sample waits.
  Each new `post_feedback_queue_settle` record was live with queue depth zero,
  completing in 101.225 / 100.702 / 100.242 ms. No backlog refusal occurred.
- Final telemetry: `(4.2954, -3.8079)`, `MODE_READY`, RTK Fix, inside Backyard
  Right, BLE −64 dBm, and blades off.
- The harness disarmed in `finally`. Independent readback confirmed experimental
  motion disabled, `real_motion_allowed: false`, no active session, and
  `MODE_READY`.

Measured conclusion: the bounded queue-settle correction prevents the specific
immediate second-pulse backlog seen in the first run while preserving the
standing safety gate. This single pass validates the correction on this path;
it is not a statistical latency or reliability population.
