# Click-to-path: the goal, and how far it has moved

Written 2026-08-09. This is the orientation document — what we are trying to
build, how the target has been re-cut three times, and what is actually left.
`docs/NEXT-SESSION.md` carries the working queue; `docs/p0-beta-release.md`
carries the formal exit criteria and the chronological evidence.

## The goal, which has not changed

**Move the mower around a map by pointing and clicking, with the blade off.**

Set 2026-07-18, on-mower, by the operator stopping a test I had queued:

> "I did not think the point of this was to have the blade move, just to move the
> mower around with a point and click map."

Not autonomous mowing. Not blade automation. The repository is a Home Assistant
integration for Mammotion mowers (a fork of `mikey0000/Mammotion`) and that
remains its base; click-to-path is a workstream inside it.

## What "working" has meant, three times over

| | Target | Status |
| --- | --- | --- |
| **1. Prove it is safe** | Pass five supervised LUBA acceptance gates | ✅ complete 2026-08-08 |
| **2. Prove it repeats** | Per-click reach beyond one segment | ✅ 4 segments, 3.6 m, 2026-08-09 |
| **3. Make it useful** | Click *anywhere* on the map | ⬜ ~16% there |

Each target was honest when set and each turned out to be a proxy. Passing the
gates proved the control law was sound but said nothing about reach. Reaching
four segments proved error does not compound but exposed that the per-*segment*
limit, not the segment count, is what bounds a click.

**The gap, stated plainly.** Backyard Right is roughly 15 × 16 m, a ~22 m
diagonal. Maximum reach per click is **3.6 m** (4 segments × ~0.9 m). Clicking
"anywhere" is therefore about **one sixth** solved. The control law works and is
trustworthy inside its envelope; the envelope is small.

## Formal status against the P0 exit criteria

From `docs/p0-beta-release.md`:

| Stage | Criteria | Status |
| --- | --- | --- |
| **Alpha** | Every safety gate fails closed; no unbounded motion; abort always wins | ✅ **Met.** 2026-08-09 produced four separate aborts — a preflight refusal, a BLE queue refusal, a turn-budget refusal and a mid-run command failure — every one clean, zero unsafe motion |
| **Beta** | Turn granularity solved | ✅ **Met 2026-08-09.** Overshoot ceiling validated; a 90° junction closes in 3 of 4 commands landing −2.66°; turn landings across the day −0.3° to −5.5° |
| | **BLE link holds a full path run** | ⚠️ **Much improved.** beta35's fixed-cadence refresh cut delivered-window overruns from +117% to +29%; **no run has aborted on BLE since**. Three consecutive clean runs is evidence, not proof |
| | No known way to strand a live client | ⚠️ Unverified since the slot-leak work; needs a deliberate check |
| **Release** | Non-LUBA hardware characterized or refused; no open safety defect | ⬜ Not started |

**The project is one criterion away from Beta, and that criterion is BLE.**

## Why BLE is the whole story

Measured across all 98 refresh writes of five real runs on 2026-08-09:

```
p50   225.6 ms      59% of writes exceed the 200 ms app interval
p75   345.7 ms      17% exceed 500 ms
p90   572.0 ms
p95  1029.2 ms
p99  2014.0 ms
```

**The median write already exceeds the refresh interval.** Motion continues only
while refresh writes keep arriving — that is why the app re-sends every 200 ms —
so a link whose median write costs 226 ms cannot sustain the cadence the motion
model assumes.

Three distinct failure signatures in one day, all the same root cause:

- a single write blocking **1303.972 ms**, starving the watchdog and
  manufacturing a phantom "9.23 °/s" rotation rate that is not rotation at all;
- linear pulses delivered **+112%** and **+117%** long against a 1300 ms command;
- an outright `TimeoutError` on a command start, ending a segment mid-path.

It degrades the rotation-rate estimate, the overshoot ceiling's guarantee,
along-track accuracy and segment gating simultaneously. Four problems, one cause.

## The working queue

1. **BLE write latency.** Both the top of the queue and, literally, the Beta exit
   criterion. Off-mower work. Start at `docs/pymammotion-ble-slot-leak-bug.md`
   and the cadence analysis above.
2. **The ~0.145 m landing-error ceiling — EXPLAINED 2026-08-10, and it is not a
   control bug.** Across 12 completed approaches, landing error is predicted by
   the aim error at the start of the leg:
   `landing = 0.62 × leg·sin(initial_aim) + 0.065 m` (R² = 0.69). To land inside
   0.15 m a 0.9 m leg needs initial aim within **8.8°**, but
   `heading_tolerance_degrees` is **18** — so a turn may legally finish at an aim
   error that guarantees a miss. `heading_tolerance_degrees` and
   `waypoint_tolerance` are geometrically inconsistent at these leg lengths.
   Both are `LUBA_ACCEPTANCE_PROFILE` keys, so resolving it owes a fresh Gate 5;
   shortening the legs is the alternative that touches no key.
3. **Per-segment reach past ~1 m.** Now the binding constraint on the actual
   goal. Means enabling loop-to-tolerance (`max_linear_pulse_ceiling`).
   ⚠️ **Prerequisite:** the mid-drive re-aim guard tests
   `command_index < max_linear_commands` rather than `effective_linear_ceiling`.
   Harmless while the ceiling is null; fix it *first* or cross-track correction
   silently stops after pulse 3 while the mower keeps driving.
4. **Release-stage work** — non-LUBA hardware. Correctly deferred.

Items 1 and 3 are the same goal from opposite ends: #1 makes what exists
reliable, #3 makes it reach. #1 gates #3, because longer segments mean more BLE
writes per segment.

## Ground rules that outlive this document

- Repositories owned by `mikey0000` are **read-only**. Authorized pushes go only
  to the `Chorty` fork.
- No motion without explicit per-run operator authorization. Arm immediately
  before, disarm immediately after, verify both.
- Changing any `LUBA_ACCEPTANCE_PROFILE` key un-accepts the profile and obligates
  a fresh Gate 5.
- Verify with per-item records, not aggregates. Both major analysis errors in
  this project came from reading a cumulative field as per-pulse and from
  trusting a prose summary over the raw array.
