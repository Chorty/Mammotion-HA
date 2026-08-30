# Route 1 — raising the travel bound to make the dead-time measurement possible

**Written 2026-08-30, before any route-1 run exists.** Nothing here may be edited
once a capture exists. Operator chose option B on 2026-08-29; routes 2 and 3 were
eliminated on 2026-08-30 (route 2 **by measurement**, route 3 as answering a
different question).

🛑 **This document authorizes nothing.** It requires a code change, a release, a
deploy, a fresh corridor scan and explicit per-run operator authorization. Phase 2
continuous steering stays PARKED under standing decision 5 — **this is
measurement, and no controller is involved.**

## 1. This raises a safety bound. Saying so plainly.

**`max_travel_m` goes 3.00 → 4.00 m**, and the schema ceiling that enforces it
goes 3.0 → 4.5. The mower will be permitted to drive **1.00 m further, open loop,
on a curve nothing is correcting**, than any step-probe run so far.

⚠️ **That is the cost, and it is not hidden by the fact that the corridor also
grows.** A bigger corridor keeps *containment* proportionate; it does not reduce
the distance the mower travels uncorrected.

🔑 **Why it is nonetheless the right trade:** the alternative is not a safer
measurement, it is **no measurement**. Two attempts have now failed inside the
3.00 m cap for two different reasons, and §3 shows the arithmetic why a third
would too. **A bound that is safe but makes the test impossible is a wasted run,
not a conservative one** — the standing check in `CLAUDE.md` says exactly this,
and it is being applied here in the direction that costs something.

## 2. Two other caps also block it, and both must move

| constant | now | proposed | why |
| --- | --- | --- | --- |
| `max_travel_m` schema max | 3.0 | **4.5** | the bound above |
| `_STEP_RESPONSE_MAX_TOTAL_MS` | 12000 | **14000** | 3000 + 5000 + 5000 = 13000 ms is refused today |
| `step_ms` schema max | 5000 | **unchanged** | 5000 is retained deliberately; see §4 |

## 3. Why 3.00 m cannot work — the arithmetic, from measurement

* **Onset lag ~2 s.** On 2026-08-29 the commanded rotation had not peaked when a
  2500 ms step ended; the peak arrived ~2 s *into* the settle phase.
* **A step must therefore run well past 2 s** to produce steady rotation to
  measure. 5000 ms leaves ~3 s of steady after onset — about 3 samples at ~1 Hz.
* **Settle must outlast the decay.** 6000 ms flattened the course on
  2026-08-30 attempt 1; 4500 ms did not on attempt 2.
* **Baseline must reach speed.** Attempt 1 proved 1500 ms does not: every chord
  before settle fell below the 0.15 m floor at 0.139 m/s.

```
baseline 3000 + step 5000 + settle 5000            = 13.0 s
path, modelled from measurement
  (2 s ramp at ~0.13 m/s, then ~0.26 m/s steady)   ~ 3.12 m
plus 0.50 m stop overshoot                          ~ 3.62 m of clearance
```

**Against a 3.00 m cap the window is ~11.5 s — 1.5 s short before it starts.**

🔑 **The guard is set to 4.00 m, not 3.12 m, ON PURPOSE.** A guard at the expected
travel *is* the truncation that censored both attempts. **0.88 m of headroom is
what makes it a backstop rather than a stopwatch.**

## 4. What is deliberately NOT changed

🗑️ **`linear_speed` stays 400.** Route 2 is dead by measurement: linear 300 gives
**0.116 m/s**, and at ~1 Hz that is a 0.116 m chord — **below the 0.15 m
informativeness floor**, so heading cannot be read at all. ⚠️ And the estimate
that suggested otherwise was wrong: predicted 0.195, measured 0.116, because a
25% command cut produced a 39% speed cut. **Do not re-derive a speed from a fit.**

🗑️ **`step_ms` stays capped at 5000.** Raising it too would let a single change
buy both a longer step *and* more travel, and §5's criterion is designed to tell
us whether 5000 is enough. **If the step turns out too short, that is a result to
record, not a cap to raise in the same breath.**

🗑️ **The 0.15 m chord floor stays.** It is the registered informativeness floor at
sigma = 0.0031 m.

🗑️ **`stop_overshoot_m` stays 0.50 m**, and the corridor budgets the full amount.

## 5. Pass criteria — all six

Criteria 1, 3 and 4 are unchanged from
`docs/phase2-feedforward-measurement-predeclared-20260829.md`. **Criterion 2 is
SPLIT**, because 2026-08-30 attempt 2 showed a run can flatten in settle while
`omega` was still measured off the ramp — and that produced a τ of 7.28 s that
had to be thrown away.

1. Report stream **ready**, no `readiness_reason`, `position_sequence` advancing.
2. **2a — the STEP reaches steady rotation:** ≥3 informative intervals in the
   step phase, and **the last two step rates agree within 1.5 °/s**.
   🔑 **This is the new one, and it is what makes `omega` trustworthy.** Without
   it `omega` can be sampled mid-ramp, which is exactly how attempt 2 produced an
   unusable τ.
3. **2b — the SETTLE goes flat:** ≥3 informative intervals in settle, last two
   agreeing within 1.5 °/s.
4. Containment holds: every sample inside the corridor, stop confirmed.
5. The travel guard **does not trip**. 🔑 **A trip means the window was truncated
   and τ is censored again — that is a FAIL**, not a smaller number.
6. Gate disarmed afterwards, verified from the live API **and** RAW
   `core.config_entries`.

**Any failure is a FAIL.** A censored or ramp-sampled τ is a failed run.

## 6. Corridor

**A 9.0 m axis-aligned square**, half-width 4.50 m ≥ the required
`4.00 + 0.50 = 4.50 m`.

Verified against the live map: squares up to **10.0 m** are fully contained in
"Backyard Right" — a 10.0 m square fits centred near **(5.40, −5.80)**, and 9.2 m
near **(4.80, −6.00)**. There is room.

⚠️ **Re-scan and re-verify containment at the LIVE position before every
dispatch.** A corridor is never reused across a run that moves the mower, and
each of these runs moves it ~3 m.

## 7. Run 2 is unchanged

After a passing run 1, repeat at **+180** to ask whether ω scales with command.
🗑️ **Two points are a direction, not a law.** Do not fit a curve.
🗑️ **The opposite signs are not repeated** — |ω| measured 4.694 vs 4.147 °/s
across ±120 on 2026-08-29, within 12%, so asymmetry is answered.

## 8. Preconditions

* 🔋 **Docked and charged first.** These are the longest runs attempted; the
  mower finished 2026-08-30 at 61%.
* The **`maxsize=1` fix ships in the same release** — it is committed and the host
  does not have it. Without it, any probe that opens its own position stream can
  still trip a false `position_sequence_gap`.
* Off dock, `AREA_INSIDE`, RTK **Fix**, BLE live, blades off. Daylight is **not**
  required for the probe itself; ⚠️ it **is** required for the repositioning drive,
  whose turns close on VIO.
* `route_start` read from the **live position after placement** —
  `_CONTINUOUS_MAX_START_DRIFT_M` is 0.30 m and that gate passes unconditionally
  in a dry run.
* A dry run showing **15/15 gates** and `blockers: []` on the exact configuration.

## 9. What a pass authorizes

Writing the feed-forward design document, with its own predeclared criteria.
**Nothing else** — not a steering run, not a controller change, not unparking
Phase 2.
