# Item 18 — first planned night segment

Item 18 ran once under explicit supervision across 2026-08-13/14. The complete,
unaltered response is
`evidence-night-segment-item18-20260814T000022Z.json`, committed separately as
`f0803af2` before this interpretation.

This is a hardware characterization, not a night landing-accuracy acceptance
pass. The executor stopped on `no_target_progress`; it did not report
`target_reached`.

## Planned and dispatched envelope

- one forward-only mapped leg, 0.699963 m
- start `(4.9862, -3.1717)`; target `(5.6835, -3.1107)`
- target map bearing `4.999523°`, approximately perpendicular to the measured
  starting body heading `277.0277°`
- `turn_mode: night`; heading tolerance 8°
- maximum four turn commands; maximum three linear commands
- fixed budget: `max_linear_pulse_ceiling: null`
- angular speed 500; turn duration 1,500 ms; refresh interval 200 ms
- turn-translation ceiling 0.30 m

The zero-motion preview returned `valid: true`, `errors: []`, and
`would_send: false`. All live preflight checks passed before the one armed run.

## Per-command measurements

### Opening turn

The target in `toward` space was 85.130477°. Initial `toward` was 173.1023°,
an error of -87.971823°. One `angular_speed: -500` pulse delivered six refresh
writes in 1,689.885 ms. It ended at `toward: 91.5568°`, leaving -6.426323°,
inside the configured 8° tolerance. Turn translation was 0.04776 m. The stop
acknowledgement succeeded.

### Linear commands

| command | speed | refresh writes | measured move (m) | course (°) | distance after (m) | aim record | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 400 | 7 | 0.433718 | 0.515212 | 0.299360 | 7.850° | drive-on projection inside tolerance |
| 2 | 400 | 3 | 0.295230 | 359.184871 | 0.106360 | 20.594° | drive-on projection inside tolerance |
| 3 | 200 | 1 | 0.028709 | 358.602819 | 0.114277 | below aim baseline | `no_target_progress` |

The third pulse's signed path progress was -0.004337 m: it moved slightly away
from the waypoint, so the existing progress gate stopped the run. All three
linear stops acknowledged successfully. No reverse or correction turn was sent.

## Mirror observations

The three per-pulse `movement bearing + toward` observations were 92.072012°,
90.741671°, and 89.156919°. They span 2.915093° and bracket the configured
90.13° mirror. This sharply differs from item 15's 14.3069° observation and
shows that item 15 was not a stable property of every night forward pulse.

This single run does not establish why item 15 disagreed, nor does it establish
a population distribution for the mirror. It does establish that the item-18
turn conversion had the correct sign and that three forward pulses subsequently
produced mirror observations near 90.13°.

## Landing and limits

The final position was `(5.7257, -3.2169)`, 0.114277 m from the target. The
night service's default `waypoint_tolerance` in this run was 0.08 m, so the
completion record correctly remained `target_remaining`. The 0.114277 m value
would fit the daylight card's 0.15 m accepted tolerance, but that tolerance is
not a night specification and must not be used to relabel this result as a pass.

Measured outcome: a valid perpendicular night segment completed its opening
turn, used all three allowed linear commands, and stopped safely at 0.114277 m
on `no_target_progress`. Inference: a shorter final action or an explicit night
tolerance decision may improve the terminal outcome, but neither is justified
by this one run.

Independent final readback found the gate disabled, no active session,
`MODE_READY`, BLE live at -58 dBm, RTK Fix, and blades zero. No repeat was run.
