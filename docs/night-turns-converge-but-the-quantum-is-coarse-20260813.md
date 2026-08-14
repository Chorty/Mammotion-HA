# Night turns converge, five for five — but the pulse quantum is ~48° and nothing scales it

**2026-08-13 night, `0.6.4-beta47`.** Four more armed turns on top of the two from
2026-08-12, all in full darkness with `tracked_features: 0`, RTK **Fix**, blades
off. Gate armed per run and verified disarmed after each. Evidence:
`docs/evidence-night-turn-{a-plus90,b-minus90,c-tol8,d-tol8-slowtier}-20260813.json`.

## 1. The headline

**Five of five turns reached `target_heading_reached` in the dark with no VIO**,
both directions, at two tolerances. The legacy primitive
(`raw_pymammotion_turn_to_heading`, closing on `position.toward`) is a working
night turn. That is no longer n = 1.

| run | target | tol | cmds | rotation per command | final error |
| --- | --- | --- | --- | --- | --- |
| prior (08-12) | +90° | 18 | 2 | +44.24 / +37.98 | **7.78** |
| a | +90° | 18 | 2 | +58.16 / +48.13 | **16.28** |
| b | −90° | 18 | 2 | −45.54 / −41.75 | **2.71** |
| c | +60° | 8 | 3 | +49.11 / +54.11 / **−50.13** | **6.91** |
| d | +60° | 8 | 1 | +52.36 | **7.64** |

Zero divergence, zero budget exhaustion, zero runaway. Run c is the important
one: it overshot to 103.22° against a 60° target and **the loop pulled it back**.
So the controller is *stable* — it detects an overshoot and reverses.

## 2. 🚨 But four of the five converged on luck, not on control

Look at the margins rather than the errors:

```
run    final error   tolerance   margin     err/tol
b        2.7114         18       15.29       0.15
prior    7.7775         18       10.22       0.43
c        6.9057          8        1.09       0.86
d        7.6446          8        0.36       0.96
a       16.2811         18        1.72       0.90
```

Three runs landed in the last 15% of their tolerance window. **The terminal error
is distributed roughly uniformly across the tolerance band**, which is exactly
what you expect when the final pulse is far larger than the remaining error: the
mower rotates a fixed quantum and stops wherever that lands, provided it lands
inside.

The tolerance is therefore acting as a **catch net, not a target**. Tightening it
from 18 → 8 barely moved the absolute error (mean 8.92° → 7.28°, n = 3 and 2)
because in both regimes the last pulse is the same size and simply has a smaller
window to land in — it just takes more attempts to get lucky.

Run c makes this concrete. After command 1 it had **10.89°** of error remaining
against a tolerance of 8. The smallest thing the primitive can do is rotate ~48°,
so it rotated **54.11°** — a 5× over-correction — then spent a third command
undoing it.

**Do not read "5/5 converged at tolerance 8" as "night turns are accurate."**

## 3. The quantum, measured

Ten refreshed pulses across the five runs, all `angular ±500`, all `tier: fast`,
windows 1509–1725 ms:

```
rotation   mean 48.15°   sd 5.70   min 37.98   max 58.16   spread 1.53x
rate       mean 30.34 °/s  sd 4.30   min 22.97   max 38.44
```

🔑 **This is dramatically tighter than the in-place VIO turn has ever been.** The
2026-08-08 characterisation found ten pulses at matched windows spreading
5.44–15.20°, a **2.79×** spread. These spread **1.53×**. Rotation on `toward` at
angular 500 with refresh is a far more predictable actuator than the record
suggests, and that is a genuinely encouraging result for a night controller.

The h-watchdog signature is visible but small here: the one pulse that delivered
only 3 of 7 writes produced the slowest rate (22.97 °/s) and the lowest rotation
(37.98°). Pulses at 7/7 and 6/7 writes are indistinguishable (48.36 vs 47.83
mean), so the degradation is not gradual — it needs a real stall.

## 4. 🔑 The fix is already visible in our own data: a second, fine quantum exists

The first night turn (2026-08-12) ran with **refresh off** and was written up as a
failure — 4 commands, 29° of 90, `max_commands_reached`. Re-read as a
*measurement* it is the most useful run of the series:

```
un-refreshed   +9.79  +6.77  +5.37  +7.02
               mean 7.24°   sd 1.60   min 5.37   max 9.79
```

**A single-shot pulse rotates 7.24° ± 1.60°.** That is precisely the fine
increment the final approach needs, and it already exists — it is what the
h-watchdog leaves you when writes stop.

Both points sit on one line. The single shot delivers roughly one command
duration (~300 ms) of motor time before the watchdog cuts it:

```
rotation ≈ 32.2 °/s · t − 2.4°      through (0.30 s, 7.24°) and (1.57 s, 48.15°)
```

So rotation is **continuously controllable by window length** from ~7° to ~58°,
and a 10° correction needs a **385 ms** window — comfortably above the 200 ms
actuation floor established in beta37.

### What to build

Port the VIO path's `_turn_final_approach_pulse_ms` — which scales pulse duration
to the remaining angle — **into the legacy turn**. The legacy turn currently fires
a fixed ~1500 ms window at one of two *speeds*, and has no window scaling at all.
With the rate constant above, the port is arithmetic we already have.

Expected result: terminal error ~3° instead of ~8°, without spending extra
commands, since the final pulse shrinks rather than repeating.

## 5. ⚠️ The slow tier is not the answer, and run d did not test it

Run d was designed to exercise the slow tier (`angular_speed_slow: 180`,
`slow_turn_threshold_degrees: 25`) after run c's reversal. **It never engaged** —
command 1 converged, and the record confirms `speed_tier: fast` on the only pulse
fired. The single command rotated 52.36° of a 60° target and landed 7.64° short of
an 8° tolerance. That is the luck described in §2, not the slow tier working.

Runs a, b and c are worse: they were configured `angular_speed_slow: 500`, i.e.
**identical to fast**, so the tier was disabled entirely. That was my error in
setting up the runs.

**The slow tier is probably unusable anyway.** It reduces *speed*, and the
2026-07-25 A/B found `angular 180` barely actuates a stationary pivot (~3° total)
— below the static-friction deadband. Arcs at angular 180 rotate fine (22.2° in
2 s) because a translating machine only needs a track differential, but that does
not carry to a pivot. So the slow tier's configured value sits in a dead zone, and
its usable range is unmeasured.

**Scale the window, not the speed.** The window has a measured linear response
across 5× of range; the speed axis has a deadband we have already been bitten by.

## 6. What is NOT established

- **A turn is still not a segment.** Nothing has driven a full closed-loop segment
  in the dark. The linear phase and the `vio_active` gate are untouched.
- **The map→`toward` conversion is still wrong by construction** (mirror, not the
  additive 102.4). Every target in this series was computed as
  `current toward + Δ` and passed by hand, deliberately bypassing it. **A night
  segment would hit it.**
- **The window-scaling model is a two-point fit.** 7.24° at ~300 ms and 48.15° at
  ~1570 ms. Nothing has been measured in between, and the 300 ms figure is
  inferred from command duration rather than a commanded window.
- **`toward`'s latency during rotation remains unmeasured.** Every reading is
  settled and post-pulse.
- **The slow tier's usable angular range is unknown** — 180 is likely below the
  pivot deadband, 500 is the fast value, and nothing between has been tried in
  place.

## 7. Next

1. **Port final-approach window scaling to the legacy turn** (§4). Off-mower, and
   the highest-value change available.
2. **Validate the window→rotation line** at 400 / 700 / 1000 ms before trusting
   the fit outside its two anchors.
3. **Fix the map-heading conversion** (mirror, not offset) before any night
   segment is attempted.
4. Only then a night segment, with its own gate story — a night mode that selects
   the legacy turn and refuses anything requiring VIO, not a bypass of
   `vio_active`.
