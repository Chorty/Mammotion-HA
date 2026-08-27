# Attempt 3 drove 0.51 m while believing it had moved 0.021 m — 2026-08-27

Third Phase 2 steering attempt, on beta81. **It refused before issuing any
steering command**, so the sign is still untested and the predeclared criteria
are still unexercised. But the refusal exposed something more important than the
sign test.

Evidence: `docs/evidence-phase2-steering-run3-blind-20260827.json`.

## The finding

| | |
| --- | --- |
| Stop reason | `heading_acquisition_timeout` at 2.03 s |
| Position sequence at t=1.02 s | 17 |
| Position sequence at t=2.03 s | **17 — unchanged** |
| Distance the executor believed it had travelled | **0.0211 m** |
| **Distance the mower actually travelled** | **0.5097 m** |
| Under-observation factor | **24x** |

Start `(4.9889, -3.0019)`; position after the run `(5.0633, -3.5061)`. The
executor commanded `linear 400` for 2.159 s of refresh window, which at the
measured ~0.25 m/s predicts ~0.54 m — matching the 0.51 m actually travelled. **The
mower did exactly what it was told. The executor could not see it.**

## The safety envelope held, and that distinction matters

✅ Blind travel 0.5097 m against the **1.06 m** acquisition disk — inside.
✅ No heading evidence → no steering command → stop. Fail-closed worked exactly
as designed, and the stop was confirmed.

⚠️ **But it stopped because it was blind, not because it noticed it was blind.**
Nothing in the run flagged that its own travel estimate was 24x low. Had the feed
delivered just enough to satisfy the 0.15 m chord and then gone quiet, the
controller would have begun steering on a stale position. That is precisely the
hazard the position-cadence programme exists to prevent, and it is the same shape
as the beta76 cell-12 anomaly: subscription status healthy, generic traffic fine,
position payloads absent.

## What the subscription actually did

```
baseline_position_sequence   14
fresh_origin sample          16     (15 and 16 arrived)
decision t=1.02 s            17
decision t=2.03 s            17     <- nothing new for a full second
~60 s later, idle            19
queue_settle                 live: true, depth 0, saga_active: false
report_stream                started: true, continuous_started: true, error: null
```

So sequences 15, 16, 17 arrived within roughly the first second — briefly faster
than 1 Hz — and then the channel went quiet for the rest of the window. **A burst
followed by silence**, with every status field reporting healthy.

🔑 **`baseline_position_epoch` was 2, where earlier runs today ran at epoch 1.**
The epoch advances on transport teardown/replacement, so BLE was re-established
between attempt 2 and attempt 3. That is a correlation worth chasing, not a
demonstrated cause: the burst-then-stall followed a reconnect. ⚠️ **n = 1. Do not
write this down as the mechanism.**

Attempt 2, minutes earlier, saw updates fine and measured 0.2407 m of travel
correctly. **The behaviour is intermittent.**

## The separate design problem this also exposed

🔑 **Acquisition is marginal by construction, independent of tonight's stall.** At
~1 Hz, establishing a 0.15 m chord from a standstill needs about two position
samples — roughly 2 s — and `max_heading_acquisition_s` is exactly **2.0 s**.
Attempt 2 made it at 1.95 s; attempt 3 did not. It is a coin flip on a healthy
feed.

⚠️ **Raising that budget is not free.** It sizes the blind disk directly:
`0.28 m/s x max_heading_acquisition_s + 0.50 m`. At 2.0 s that is the familiar
1.06 m; at 3.0 s it becomes 1.34 m, which demands more clearance before a run may
open. Any change here is a safety trade, not a tuning tweak.

🗑️ **Do NOT lower `min_travel_for_heading_trust_m` (0.15 m) to make acquisition
easier.** It is the registered informativeness floor: at the measured
sigma = 0.0031 m position noise, a shorter chord carries bearing noise exceeding
the thresholds it would feed.

## Status

🛑 **The corrected steering sign has still never moved a wheel.** Three attempts,
three distinct refusals:

| Attempt | Build | Refusal | Cause |
| --- | --- | --- | --- |
| 1 | beta80 | `position_sequence_gap` | two reliability defects (fixed) |
| 2 | beta81 | `opening_alignment_infeasible` | window budget eaten by the blind phase |
| 3 | beta81 | `heading_acquisition_timeout` | position channel delivered one sample in two seconds |

Every one failed closed, and no unintended motion occurred in any of them.

## What to do next, in order

1. **Investigate the burst-then-stall.** The lease and generation machinery built
   for exactly this question is deployed; use `report_stream_sequence_probe`
   around a transport reconnect and see whether epoch advance correlates.
2. **Decide the acquisition budget deliberately**, as a stated safety trade
   between `max_heading_acquisition_s` and the blind-disk radius it sets.
3. **Only then attempt the sign again.** The attempt-3 configuration (6 s window,
   8° route offset) is sound and untested; the run never reached it.

⚠️ **Do not treat "three refusals" as evidence the steering law is wrong.** It has
not run. Nothing about the sign has been measured either way.
