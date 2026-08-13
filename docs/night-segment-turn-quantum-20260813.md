# Night-segment turn quantum — 2026-08-13

Complete response: `evidence-night-segment-turn-quantum-20260813T214605Z.json`.
This was one explicitly authorized, supervised, blades-off run through
`turn_mode: "night"`. The gate was disarmed afterward and independently read
back `enabled: false`, `real_motion_allowed: false`, with no active session.

## Measured facts

- The opening error was −62.3072°. One 1,500 ms pulse dispatched
  `linear_speed: 0`, `angular_speed: -500` over BLE.
- The pulse delivered six refresh writes at the configured 200 ms interval.
  The refresh window elapsed 1,591.1 ms; the six write durations were 183.087,
  64.222, 104.131, 174.594, 178.173, and 390.377 ms.
- Raw `toward` changed 97.4064 → 43.1856: −54.2208° for the one pulse. The
  residual was −8.0864°, inside the configured 18° tolerance, so the turn phase
  returned `target_heading_reached`.
- RTK translation during the turn was 0.07459 m, below the 0.30 m refusal
  threshold. The BLE stop acknowledgement succeeded in 202.598 ms.
- The executor then sent one forward pulse at `linear_speed: 400`, with seven
  refresh writes. It travelled 0.43648 m on map bearing 331.1213°.
- The bearing from the post-pulse position to the target was 52.5378°. The
  recorded aim error was 81.416°, and the observed `movement bearing + toward`
  mirror constant was 14.3069°, not the configured 90.13°.
- The night-only guard stopped the run with
  `night_reaim_required_but_unavailable`. No second linear pulse was sent.

## Conclusions and limits

The segment-call-site turn quantum is now measured: 54.2208° in this one
1,500 ms, angular-500 refreshed pulse (`n = 1`). It is consistent with the
coarse standalone-turn population, but one sample does not establish a rate
distribution or a correction floor.

The forward-pulse result is a direct refutation of treating the 90.13° mirror
as established for night segment control. The evidence establishes the
disagreement; it does not by itself identify whether the cause is `toward`
latency, a body/course-frame distinction, or another telemetry/control-frame
effect. The re-aim guard contained the failure as designed. Do not proceed to
the plan's first-night-segment acceptance task or add a correction model until
the heading disagreement is explained. The next diagnostic remains §7 item 16
(`toward` sampled during and for about three seconds after one rotation), and it
requires a new explicit supervised-motion authorization.
