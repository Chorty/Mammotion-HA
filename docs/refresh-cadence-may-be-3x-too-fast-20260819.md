# We may be refreshing 3× faster than the mower needs — 2026-08-19

🔑 **Median pulse speed is flat out to ~700 ms between refresh writes, and we
refresh every 200 ms.** If the motor watchdog really tolerates ~700 ms, we send
roughly **three times more BLE writes than the mower requires** — and BLE write
latency, not cadence, is this project's documented failure mode.

Derived from data already on disk. No mower run.
`scripts/analyze_refresh_cadence.py`.

## Where 200 ms came from

The vendor app re-sends the same movement command every 200 ms while the stick
is held (APK decompile, 2026-07-20). That finding produced
`motion_refresh_interval_ms: 200` and the B1 tape A/B — the same 4 s pulse
taped at **4 in** single-shot versus **44 in** with refresh, ~11×.

Independently corroborated 2026-08-19 in upstream pymammotion's own
`examples/pyjoystick_example.py`:

```python
self.worker = PeriodicThread(0.2, self.run_movement, name="luba-process_movements")
```

So 200 ms is well-founded as *"what the app does"*. It has never been tested as
*"what the mower needs"*.

## Method

Every recorded linear pulse carries `motion_refresh.refresh_write_durations_ms`
and a measured travel distance. A long write is a long stretch with no refresh
reaching the mower, so `max(write_durations)` proxies the longest gap. Bucket by
that and compare median speed (`travel / elapsed`).

⚠️ **Cruising pulses only.** The final-approach planner deliberately shortens
pulses near the target, so including them depresses the short-gap buckets and
manufactures a trend that is really just "short pulses are slow". Filtering on
`final_approach.applied is False` drops 326 pulses to 120 — and the effect
survives, which is the point of doing it.

## Result — 120 cruising pulses

| longest write / gap | n | median m/s | vs 150–250 ms |
| --- | --- | --- | --- |
| 0–150 ms | 8 | 0.247 | — |
| 150–250 ms | 40 | 0.255 | 100% |
| 250–400 ms | 30 | 0.265 | **104%** |
| 400–700 ms | 27 | 0.254 | **99%** |
| 700–1200 ms | 13 | 0.207 | 81% |
| 1200+ ms | 2 | 0.100 | 39% |

Flat to ~700 ms, then a cliff. Note the 0–150 ms bucket is **not faster** than
250–400 — refreshing harder buys nothing.

## Why it is worth acting on

BLE write latency is the measured failure mode, not cadence:

- a 1303.7 ms pulse that landed **one of six** refreshes, on a write that took
  1303.972 ms, measured **9.23 °/s** against 23–43 °/s for cadence-intact pulses
- `refresh_cadence_broken` exists precisely because such pulses corrupt the
  rotation-rate estimate
- BLE-stalled linear pulses travel ~0.22 m against ~0.41 m healthy

Halving or quartering the write count attacks queue pressure directly. ⚠️ This
is **motion-control** work, not the BLE-reliability work CLAUDE.md says not to
fund on a 0.2–31.9% confidence interval — the intervention is a parameter we
already own, and the evidence is 120 pulses rather than a 7% abort rate.

## ⚠️ What this is NOT

- **Correlational, on a proxy.** `max(write_duration)` is not the gap; it is the
  longest single write. A slow write also consumes the window it is in.
- **Thin in the tail** — n=13 and n=2 in the two degraded buckets.
- It **bounds** the watchdog above 400 ms and *suggests* ~700 ms. It does not
  measure it. Do not change `motion_refresh_interval_ms` on this alone.

## The experiment that would settle it

One fixed-duration forward pulse at refresh **200 / 400 / 600 / 800 ms**,
measuring travel each time. Four pulses of ~0.4 m, one daylight session,
everything else held constant — the same single-variable shape as the B1 tape
A/B that produced the 4 in → 44 in result.

Expected: travel flat at 200/400/600 and dropping at 800 if the watchdog sits
near 700 ms. If travel drops at 400, the corpus signal is a confound and
200 ms stands — which is equally worth knowing.

Cheap, bounded, and it either halves our BLE traffic or closes the question.
