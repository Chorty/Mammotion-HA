# 🏁 A closed-loop turn converged in the dark, with no VIO

**2026-08-12 night, `0.6.4-beta47`.** Two armed turns, gate armed for each and
disarmed with the state verified. Blades off, RTK Fix, `tracked_features: 0`,
`camera_brightness: dark`. Evidence:
`docs/evidence-night-closed-loop-turn-20260812.json` (without refresh) and
`docs/evidence-night-closed-loop-turn-refreshed-20260812.json` (with).

## 1. The result

`mammotion.raw_pymammotion_turn_to_heading` — the **legacy** turn primitive,
which closes on `position.toward` and carries no `vio_active` gate:

```
stop_reason  target_heading_reached      commands_sent 2
toward       -156.85  ->  285.37         target 293.15, error 7.78 deg (tol 18)

cmd1  rotated 44.24 deg   7 refresh writes   1584 ms
cmd2  rotated 37.98 deg   3 refresh writes   1653 ms
```

**Two commands, 82.2 deg of rotation, converged and stopped itself, in full
darkness with no VIO.**

## 2. The single-variable comparison

The same service, same target offset (+90 deg), same speeds, same command budget.
The only difference is `motion_refresh_interval_ms`:

| | commands | per-command rotation | outcome |
| --- | --- | --- | --- |
| refresh **off** | 4 | +9.79 / +6.77 / +5.37 / +7.02 | 29 deg of 90, `max_commands_reached` |
| refresh **200** | 2 | **+44.24 / +37.98** | 82 deg of 90, **`target_heading_reached`** |

**~6x more rotation per command.** The un-refreshed pulse is the single-shot
h-watchdog quantum: the motor stops when writes stop arriving, which is the same
mechanism that produced the fixed ~4 inch linear step before refresh was
discovered on 2026-07-22.

🔑 **The failure was predicted before the first run, not explained after it.**
The note written ahead of it said the loop "may exhaust `max_commands` before
converging — a plausible failure that wouldn't mean `toward` is unusable. If that
happens, adding refresh here is the obvious next change." That is exactly what
happened, and it is why the first run counts as evidence FOR `toward` rather than
against it.

## 3. Why the first run already proved the important part

Even without converging, the un-refreshed run showed the loop working:

```
toward 174.20 -> 203.15
per command  +9.79  +6.77  +5.37  +7.02      sum = 28.95
174.2007 + 28.95 = 203.1508  == the reported final heading, exactly
```

Every command rotated in the commanded direction, and the per-command changes sum
precisely to the total. **`toward` behaved as a coherent heading signal through
four closed-loop iterations with VIO at zero.** The loop read it, computed error,
and drove monotonically toward the target. It simply could not get there on 7 deg
a command.

## 4. What this changes

Closed-loop motion has been daylight-only because every turn closes on VIO
heading, and VIO dies in the dark. The justification was that the alternative —
`toward` — is course-over-ground and therefore blind to in-place rotation. That
premise was refuted earlier the same night
(`docs/toward-tracks-in-place-rotation-20260812.md`), and this run shows the
consequence: **a turn primitive that closes on `toward` converges in the dark.**

The primitive already existed. Nothing had to be written except the refresh.

## 5. What is NOT established

- **n = 1 on the refreshed path.** And the BLE variance is right there in it: 7
  writes on the first command, 3 on the second.
- **A turn is not a segment.** A night *segment* also needs the linear phase, and
  the `vio_active` gate still refuses `turn_mode: "vio"` unconditionally
  regardless of whether a turn is needed. **Nothing has driven a full closed-loop
  segment in the dark.**
- **The 7.78 deg residual sits inside an 18 deg tolerance**, which is loose.
  Whether `toward` supports a tighter one is unmeasured, and the landing accuracy
  work says tolerance is what governs cross-track.
- **`toward`'s latency DURING rotation is still unmeasured.** Every reading here
  is settled and post-pulse. A tighter loop would need to know it.
- **The heading frame was supplied by hand.** The target was computed as
  `current toward + 90` and passed directly, deliberately bypassing the vector
  executor's `calibrated_forward_heading_offset_degrees: 102.4` conversion, which
  is wrong by construction (`toward` is a compass bearing needing a mirror, not
  an additive offset). **A night segment path would hit that conversion**, and it
  must be fixed before one is attempted.

## 6. Next

1. **Repeat the refreshed turn** two or three times for n, ideally at different
   target angles and both directions.
2. **Sample `toward` DURING the pulse** rather than after, which is what a
   tighter loop would depend on.
3. **Fix the map-heading conversion** (mirror, not offset) before any night
   segment.
4. Only then a night segment, which needs its own gate story — the `vio_active`
   refusal exists for good reasons and must not simply be deleted. The honest
   shape is a night mode that selects the legacy turn and refuses anything
   requiring VIO, not a bypass.
