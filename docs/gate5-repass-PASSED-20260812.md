# 🏁 Gate 5 re-passed on the reach-enabled profile — 2026-08-12

Card-driven, four segments, all `target_reached`. Evidence:
`docs/evidence-gate5-repass-2-20260812.json`.

```
planned 4   executed 4   real 4   stop_reason: target_reached   errors: []

seg 1   target_reached   0.0674 m   turn 0  linear 2
seg 2   target_reached   0.1032 m   turn 1  linear 3   1 suppression
seg 3   target_reached   0.0807 m   turn 1  linear 3   1 suppression
seg 4   target_reached   0.0607 m   turn 2  linear 2   post-turn correction fired

max 0.1032   mean 0.0780   tolerance 0.15
zero reverse-recovery · zero budget exhaustion · zero failed safety gates
```

**Mean 0.0780 m is the best four-segment result on record** — against beta40's
0.0956 and the original Gate 5's 0.0832.

## What this certifies

`max_linear_pulse_ceiling: 14` was adopted into `LUBA_ACCEPTANCE_PROFILE` on
2026-08-12 and **is now hardware-accepted**. The real payload carried it, so the
profile-identity invariant (`docs/p0-beta-release.md:98-102`) is closed for the
reach key: the card demonstrably sent what was accepted, and the mower executed
it.

Per-segment reach is ~4 m and per-click reach ~16 m at four segments
(`docs/loop-to-tolerance-reach-20260811.md`), and that capability is now behind
an accepted profile rather than a harness override.

## ⚠️ What it does NOT certify

**beta43 was not exercised.** The build that carried this gate raised the
post-turn correction's budget from 2 commands to 4, fixing the refusal that
failed the first attempt (29.647° at a 10° tolerance needing 3 commands). In
this run the only correction was **−10.477° → 0.558°**, inside the old
2-command envelope of 21.50°. The 29.6° case remains proven by replay only.

Nor did the first attempt's failure recur naturally: its segment 3 had a 54°
opening turn and a −29.647° post-turn error, where this run's segment 3 saw
−4.711°. **The geometry differed; the fix was not what carried the gate.**

**The ceiling never bound.** At 0.8 m legs the linear phase used 2–3 pulses of
14. Loop-to-tolerance ran and every segment stopped on tolerance, but the
accepted key's headline capability is evidenced by
`docs/loop-to-tolerance-reach-20260811.md`, not by this gate. That was a
deliberate choice: the long-leg regime is 5/7 on control-law grounds while the
short-leg regime is 28/28, and a gate is not the place to test the riskier one.

## ⚠️ A hole in the record, now fixed

Two accepted-profile keys came back `null` in the response:

- `motion_refresh_interval_ms` — **provable anyway**: the per-segment echo read
  200 and refresh observably ran at 200 ms across all 14 pulses.
- `max_no_progress_pulses` — **unprovable from the record.** Dismissed only
  because it acts solely when a pulse makes no progress, and every pulse
  progressed, so it could not have altered the outcome.

The card sends both (`_motionPayload`); the executors simply did not echo them.
Having to argue around a hole in a gate that exists to prove exactly this is
unacceptable, so **beta44 echoes every profile key from both the vector and
multi-segment results**, pinned by a test that reads the card's frozen profile
and asserts each key is echoed.

## Run conditions

```
mower start   (5.7839, -4.8819) Backyard Right, MODE_PAUSE
facing        272.57° (mirror of toward 177.5642)
path          4 × 0.8 m, junctions 60 / -60 / 60, opening turn ~0°
clearance     >= 3.99 m at every waypoint; 1.6 m overrun still 3.28 m inside
VIO           live, 80 tracked features, Light
RTK           Fix, AREA_INSIDE, valid_for_motion
blades        OFF, 0 rpm, blade_rpm_looks_latched false
build         0.6.4-beta43, gate armed for the run and disarmed after
```

⚠️ The operator had started and cancelled two mows shortly before. The blade
latch discriminator explicitly reported `blade_rpm_looks_latched: false` with
~24 min of HA uptime (it needs ~15), and the cancelled mows left
`route_present: true` with `blocks_motion: false`, reason
`stale_route_while_ready`.

## Status of the five gates

All five are complete, and Gate 5 has now been passed twice: once on the
fixed-budget profile (2026-08-08) and once on the reach-enabled profile
(2026-08-12). The remaining work is capability, not gates.
