# Item 17 — reverse heading and RapidState fusion

Item 17 ran once under explicit supervision on 2026-08-13. The complete,
unaltered service response and 81 concurrent runtime records are in
`evidence-night-reverse-fusion-20260813T234313Z.json`. Raw evidence was committed
separately as `ff8e1f09` before this interpretation was written.

## Command actually dispatched

- `linear_speed: -400`; `angular_speed: 0`
- duration 1,300 ms; refresh interval 200 ms
- five refresh commands in 1,334.641 ms
- BLE-preferred transport
- the stop completed successfully in 304.864 ms
- service result `completed`, with no recorded error

The post-run independent preflight found `MODE_READY`, BLE live at -64 dBm,
RTK Fix, blades zero, and the motion gate disabled. `real_motion_allowed` was
false and the sole standing blocker was `experimental_motion_disabled`.

## Measured records

These are every capture row at which position, heading, fusion, or session phase
changed. They are per-item observations, not aggregate summaries.

| elapsed (s) | x | y | `toward` | Rapid fuse | vision raw | device VSLAM fuse | session phase |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0.433018 | 5.0331 | -3.5876 | 173.1023 | 0 `NO_POSE` | 0 | 1 | none |
| 1.030853 | 5.0331 | -3.5876 | 173.1023 | 0 `NO_POSE` | 0 | 1 | running |
| 3.469296 | 4.9857 | -3.1712 | 173.1023 | 0 `NO_POSE` | 0 | 1 | running |
| 4.177251 | 4.9861 | -3.1764 | 173.1023 | 0 `NO_POSE` | 0 | 1 | running |
| 5.197835 | 4.9865 | -3.1731 | 173.1023 | 0 `NO_POSE` | 0 | 1 | running |
| 6.210636 | 4.9862 | -3.1717 | 173.1023 | 0 `NO_POSE` | 0 | 1 | running |

The remaining capture rows repeat the last values. The service's own six
post-command samples independently report `toward: 173.1023` at 0, 0.25, 0.5,
1, 2, and 3 seconds.

From first to settled position, the measured displacement was:

- `dx = -0.0469`, `dy = +0.4159`
- distance `0.418536 m`
- map-frame movement bearing `96.433921°`
- unchanged `toward = 173.1023°`

## What the records establish

Using the already measured map convention, the body-facing map heading is
`(90.13 - 173.1023) mod 360 = 277.0277°`. A backward course should therefore be
`97.0277°`. The measured course was `96.433921°`, a circular difference of only
`0.593779°`.

That is the reverse discriminator: `toward` stayed fixed to the mower body while
the course of travel was approximately 180 degrees opposite. It did not flip to
follow course-over-ground. Equivalently, the reverse-adjusted relation was
`movement bearing + toward + 180 = 89.536221°`, within 0.593779 degrees of the
90.13-degree mirror.

Therefore the body-vs-course question is settled for this mower and firmware:
`toward` is a body-heading observation under reverse travel. This is one
backward run, so it establishes the semantic discriminator; it is not a new
precision estimate for the mirror constant.

## What the records do not establish

`mowing_state.fuse_status`, decoded from RapidState tard-state word 16 bits
8-15, stayed `0 (NO_POSE)` in all 81 runtime records. `vision_state_raw` stayed
0. The distinct `report_data.dev.fuse_status` field stayed 1. At the same time,
the integration's independent location source reported RTK Fix throughout.

Measured fact: those values were constant during this manual reverse session.
Inference: the RapidState fusion byte was not carrying useful live fusion state
on this path. Do not reinterpret `NO_POSE` as proof that RTK was absent, and do
not substitute the separate device VSLAM field merely because its value was 1.
A pivot-specific RapidState transition was not observed in this run.

Item 17's blocking body-vs-course question is complete. This does not itself
authorize item 18 or resolve item 15's separate 75.823-degree forward-course
disagreement.
