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
hazard the position-cadence programme exists to prevent. ⚠️ **This is NOT the
beta76 cell-12 anomaly** — see the refutation below; the feed was healthy and one
ordinary interval was missed.

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

Sequences 15, 16 and 17 arrived within roughly the first second, and no further
sample landed before the 2.03 s decision. ⚠️ **Read that as ONE missed interval,
not a burst-then-silence anomaly** — the refutation immediately below shows the
feed's median interval is 1016 ms, so a 1.01 s window with no new sample is
close to a coin flip.

🗑️ **REFUTED THE SAME EVENING — THERE WAS NO STALL.** This section originally
read the missing sample as a burst-then-silence anomaly and flagged the epoch-2
reconnect as a correlation to chase. Measurement says otherwise, and the
"burst-then-stall" framing was an over-read of a single ordinary interval.

A stationary `report_stream_sequence_probe` run **at the same epoch 2**, minutes
later, found the feed perfectly healthy: 4 of 4 cells ready, ~30 payloads per 30 s
cell, **zero drops, zero sequence gaps**, and over 117 intervals a median of
**1016 ms**, p95 1102, p99 1127, max 1232. Evidence:
`docs/evidence-position-cadence-post-reconnect-20260827.json`.

🔑 **The arithmetic settles it.** Attempt 3 needed a NEW sample between t=1.02 s
and t=2.03 s — a **1.01 s window** — against a feed whose **median interval is
1016 ms**, with **57% of intervals exceeding 1010 ms**. Seeing no new sample in
that window is close to a coin flip. **One ordinary interval, not a fault.**

⚠️ **The epoch-2 correlation is withdrawn.** Attempt 2 ran at epoch 2 as well and
saw its samples fine. The reconnect explains nothing here, and recording it as a
lead would have sent the next session chasing a ghost — exactly the failure this
document was written to prevent.

**What remains true and important:** the executor really did travel 0.5097 m while
believing 0.0211 m, and nothing flagged the discrepancy. That is a real property of
a ~1 Hz feed under motion, not a malfunction — which makes it a DESIGN problem
rather than a bug to hunt.

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

1. ✅ **DONE — there is no stall to investigate.** See the refutation above. The
   feed is healthy at ~1 Hz and attempt 3 simply lost a coin flip on one interval.
2. **Decide the acquisition budget deliberately**, as a stated safety trade
   between `max_heading_acquisition_s` and the blind-disk radius it sets. **This
   is now the ONLY thing standing between us and a sign test**, and item 1's
   result makes it sharper: at a 1016 ms median interval, a 2.0 s budget buys
   about two samples, and losing either one fails acquisition.
3. **Only then attempt the sign again.** The attempt-3 configuration (6 s window,
   8° route offset) is sound and untested; the run never reached it.

⚠️ **Do not treat "three refusals" as evidence the steering law is wrong.** It has
not run. Nothing about the sign has been measured either way.
