# FINDINGS — second setup leg halted at dusk; session ended on darkness

2026-09-12. **Phase 1 still has NOT run — zero scored legs.** Evidence:
`docs/evidence-setup-leg2-realign-halt-20260912.json`. Plan and halt rules:
`docs/predeclared-queue-timeout-measurement-20260911.md` §16.

## 1. Two halts, both working as designed

1. **Corridor-clearance halt, nothing sent.** From `(4.45, 0.54)` the runner
   measured **0.66 m** start clearance (dock exclusion notch, behind the mower)
   against §16.3's 1.0 m floor. Not overridden; operator drove it ~1.5 m south
   to 2.11 m and it passed.
2. **`vio_realign_incomplete`.** From `(4.68, −1.03)` toward the anchor
   `(4.94, −3.82)`: aligned start (5.5°), 5 linear pulses, 78 % progress, then
   aim error **19.4°** against the 18° tolerance ~0.6 m out; the realign turn
   hit `max_commands_reached` and **overshot** (VIO swung ~59° where ~19° was
   needed). Landed `(4.84, −3.26)`, **0.57 m** from the anchor. Gate disarmed.

## 2. 🚨 That leg ran AFTER SUNSET — a protocol lapse, mine

Dispatched **23:55:07–23:56:04Z; sunset was 23:49:30Z.** I did not re-check sun
elevation or VIO before arming. The protocol requires daylight; night is a
closed standing decision.

**Corroborated by the leg's own minutes** (n = 1, corroboration not proof):

| time (Z) | signal |
| --- | --- |
| 23:50 | `vio_tracked_features` 71 |
| 23:53 | 51 |
| 23:55:12 | `visual_positioning_status` → **signal_bad** |
| 23:55:53 | → **signal_none** (11 s before the halt) |
| 23:56 | features 34; 13 by 23:59 |

🔑 **`vio_brightness` (1) and `camera_brightness` (`light`) read "good" the whole
time** — the documented cliff where the brightness fields lie. The overshoot and
the post-halt heading disagreement (VIO 329.6° vs compass mirror 176.5°, 153°)
happened as VIO collapsed.
✅ **Fix for next session:** the leg runner must HALT on a VIO/daylight check
before dispatch (e.g. `visual_positioning_status` not `signal_good`, tracked
features below a floor, or sun elevation under a margin) — a check the §16.3
list lacked.

## 3. Other observations

- **Fault 5004** — no description in the cloud row or the bundled 449-code
  table. Logged 23:41:12Z (coincides with a VIO dip 80→63 during the operator's
  app move) and 20:46:28Z. **Not caused by docking** (32 min earlier). Unknown.
- **Fault 1068 "vision camera is dirty"** at 21:21:06Z — a live candidate
  contributor to VIO degradation. ⚠️ Clean the camera before the next session.
- **Queue timing: 131 cumulative samples, all `completed`, zero timeouts** —
  both setup legs, excluded from the population.

## 4. Session end state

Operator called it on darkness. `lawn_mower.dock` sent 00:13:17Z, **docked and
charging at 00:14:18Z** (~61 s), RTK `fix`, battery 99 %. Gate **disarmed,
verified from the live API AND raw `core.config_entries`**
(`enable_experimental_motion: false`).

**Next session:** clean the camera; daylight with margin; add the VIO/sun halt;
setup leg into the pocket; then S1–S12 per §16.2. Timing samples are in memory
only — an HA restart clears them; all are banked.
**No code changed. No deploy. No control-law or profile value moved.**

## 5. Overnight RTK watch — nothing to start, read it tomorrow

2026-09-11 night the correction path failed after dark (`rtk_position: single`,
`position_level: 0`, fault 1300) while the mower tracked 24 satellites, and it
was never explained. The recorder is confirmed logging all three entities, so
this costs nothing: **tomorrow, read HA history from 2026-09-13T00:00Z** for
`sensor.back_yard_clip_skywalker_rtk_position`,
`sensor.back_yard_clip_skywalker_position_level` and the RTK base's satellites
sensor (which read 28 at 2026-09-13T00:4xZ, up from last night's 0).
**If `single` recurs with the mower docked, it is a night pattern, not a
one-off.** ⚠️ Docked is not the same arrangement as last night's off-dock
stranding, so a clean night does not rule the pattern out.
