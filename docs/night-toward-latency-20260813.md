# `toward` latency during one night turn — 2026-08-13

Complete response and concurrent capture:
`evidence-night-toward-latency-20260813T215934Z.json`.

This was one explicitly authorized, supervised, blades-off, angular-only pulse.
The harness contains no linear or reverse command. Final independent readback:
gate disabled, `real_motion_allowed: false`, no active session, `MODE_READY`,
BLE live, RTK Fix, and blade RPM zero.

## Measured facts

- Initial `toward` was 43.1856 and the standalone turn target was 103.1856.
  Exactly one command dispatched `linear_speed: 0`, `angular_speed: 500`.
- The configured pulse was 1,500 ms with 200 ms refresh. Five refresh writes
  were delivered; the measured refresh window was 1,550.945 ms. The explicit
  BLE stop succeeded in 97.579 ms.
- The pulse changed `toward` 43.1856 → 79.492: +36.3064°. RTK translation was
  0.11555 m, below the 0.30 m refusal limit. The expected one-command exit was
  `max_commands_reached` because 23.6936° remained.
- The concurrent capture contains 73 runtime samples over 7.55 seconds. Median
  request time was 85.714 ms (min 78.456, max 170.258 ms).
- Only two `toward` values occur anywhere in the capture: 43.1856 and 79.492.
  Twenty samples with the active session still reported 43.1856. The last stale
  sample completed at elapsed 2.929716 s; the first changed sample completed at
  3.114884 s. No intermediate heading was observed.

## Conclusions and limits

Measured: on this pulse, `toward` did not stream progressive headings at the
capture's roughly 0.1-second cadence. It arrived as one post-pulse step. This
answers plan §7 item 16 for this mower and call path (`n = 1`).

Inference: a night controller cannot assume useful mid-pulse `toward` feedback
at this cadence. It can close between bounded pulses after feedback settles,
but this sample does not establish a universal latency bound. The capture did
not instrument the mower protocol timestamp, so the exact delay from physical
stop to report arrival must not be claimed from the local request timestamps.

This result does not explain away item 15's 81.416° forward-course mismatch:
the segment executor already waits for post-command feedback before its forward
phase. The body-heading versus course-over-ground question therefore remains
open. Do not proceed to item 18. Plan item 17 (one backward pulse plus
`RapidState.fuse_status`) is the next hardware discriminator and requires a new
explicit supervised-motion authorization.
