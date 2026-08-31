# Route 1, step extension — the first full PASS

⚠️ **QUALIFIED THE SAME DAY — read
`docs/vio-crosscheck-reframes-route1-step-verdicts-20260830.md` alongside
this.** VIO's own independent heading track disagrees with this run's own
RTK-chord measurement: it shows the last two step-phase rates 1.96°/s apart
(the same 1.5°/s bound this document's 2a criterion uses would FAIL that
reading), against the 0.11°/s this document reports. The PASS below is
correct as a reading of the probe's registered instrument — it is not an
unambiguous physical fact independent of which channel is asked.

**2026-08-30, beta88.** One supervised, operator-authorized run of
`raw_pymammotion_step_response_probe` against
`docs/phase2-route1-step-extension-predeclared-20260830.md`: baseline 3000 /
**step 7000** / settle 5000 ms, `step_angular_speed=120`, `max_travel_m=4.5`,
10.0 m square corridor at the mower's live position. Raw evidence:
`docs/evidence-route1-step-extension-pass-20260830.json`. Compare against the
two 5000 ms-step attempts:
`docs/evidence-route1-run1-fail-20260830.md` and
`docs/evidence-route1-run1-repeat-fail-20260830.md`.

## Verdict: PASS — all six criteria

| # | criterion | run 1 (5000 ms) | repeat (5000 ms) | **step extension (7000 ms)** |
| --- | --- | --- | --- | --- |
| 1 | report stream ready | ✅ | ✅ | ✅ |
| 2 | 2a — last two step rates ≤1.5°/s | ❌ 2.49°/s | ❌ 7.28°/s | ✅ **0.11°/s — passes** |
| 3 | 2b — last two settle rates ≤1.5°/s | ❌ 2.07°/s | ✅ 0.26°/s | ✅ 1.44°/s — passes, narrowly |
| 4 | containment + stop confirmed | ✅ | ✅ | ✅ |
| 5 | travel guard does not trip | ✅ | ✅ | ✅ **and the `reason` field agrees** |
| 6 | gate disarmed after, verified | ✅ | ✅ | ✅ |

**`tau_actuator_s = 2.038 s` may be quoted as a genuine measurement** — not a
lower bound, not censored, not sampled off a ramp. This is the first time
that has been true anywhere in the route-1 effort.

## The onset-lag hypothesis is now supported, not just suggested

Both 5000 ms-step attempts failed 2a with the same signature: the step
phase's *final* interval was always the *largest*-magnitude rate of the whole
phase — still accelerating when the phase ended. This run's step-rate
sequence shows the mechanism directly:

```
-1.718 -> -3.428 -> -9.026 -> -6.127 -> -8.241 -> -8.353   (deg/s)
```

Climb, overshoot slightly, then flatten — the last two values are nearly
identical (0.11°/s apart). The extra 2000 ms of step time is what the two
shorter runs were missing to reach exactly this shape. That is a mechanism
match, not just a correlated pass.

## The settle pass is real but thin

4 informative intervals (one fewer than either 5000 ms run's settle phase,
which each got 5), and the last two settle rates are 1.44°/s apart — inside
the 1.5°/s bound, but by only 0.06°/s. The course held roughly flat
(-159.031° → -159.619° → -158.699°) with visible residual wobble, not as
clean a plateau as run 1's repeat (0.26°/s margin). **A repeat at this exact
configuration could plausibly fail 2b on the same chord noise every run today
has shown**, even if 2a is now reliably passing.

## The reason-field fix, confirmed on hardware for the first time

The service reported `"reason": "window_complete"` — correctly, for the
first time all day. Confirmed independently from the raw evidence exactly as
the two prior runs required manual correction for: `motion_refresh
.aborted_early` is `False`, 0 of 146 samples carry `travel_guard_tripped:
true`, and the window ran its full 15001.7 ms of a 15000 ms schedule. Before
this run, the fix (commit `af5f547f`, deployed in beta88) had only unit
tests — this is its first real-hardware exercise, and it worked.

## The repositioning drive (before this run)

The mower had been driven, by the operator, to `(6.718, -3.3715)`, where the
worst-case clearance (area boundary 4.44 m, nearer keep-out 4.379 m) was only
**0.89x** the 5.0 m disk this run needs — insufficient for a fresh 10.0 m
corridor there. A single call of the accepted closed-loop reach profile
(`raw_pymammotion_execute_vector_segment`, `docs/accepted-profile.json`,
verified key-by-key) drove it back to `target_reached` at 0.145 m from
`(5.98, -5.24)` — no staged turn needed this time, 1 turn command, 8 linear
commands. Final position `(6.1232, -5.2363)`, re-scanned worst-case clearance
**5.849 m (1.17x)** against the 5.0 m disk before the step-response dry run.

## Safety

15 of 15 gates before dispatch, `blockers: []`. Every sample stayed inside
the corridor. Stop confirmed (`ok: true`, `ack.movement_ok: true`).
Cumulative travel 2.902 m of the 4.5 m budget. Explicit operator confirmation
taken immediately before both the repositioning drive and the step-response
dispatch. Gate disarmed and verified from both the live API and RAW
`core.config_entries` after each dispatch.

## What this does not establish

* **n = 1 at this configuration.** No repeat has run at +120 with the 7000 ms
  step. The 2b margin (0.06°/s) is thin enough that a repeat could fail it.
* This says nothing about +180 yet — that run has not been dispatched.
* This does not license Phase 2 steering, another `step_ms` increase, or the
  feed-forward design document on its own.

## What this pass authorizes

Per `docs/phase2-route1-step-extension-predeclared-20260830.md` §7: repeating
at **+180** using this same step length (baseline 3000 / step 7000 /
settle 5000, `max_travel_m=4.5`). Nothing more — not another cap change, not
Phase 2 steering, not the feed-forward design document without that +180 run
first.
