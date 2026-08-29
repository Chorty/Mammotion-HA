# Option B, step 1 — the two measurements a feed-forward design needs

**Written 2026-08-29, before either run exists.** Nothing here may be edited once
a capture exists. Operator chose option B — *design for the dead time instead of
against it* — on 2026-08-29, after
`docs/evidence-dead-time-measured-20260829.json`.

🛑 **This document authorizes nothing.** Both runs need explicit per-run operator
authorization, a freshly scanned corridor, and the usual disarm-and-verify tail.
Phase 2 continuous steering stays PARKED under standing decision 5; **this is
measurement, not a steering run, and no controller is involved.**

## Why more measurement before any design

The 2026-08-29 runs measured τ for the first time, and both numbers are unusable
as design inputs for two stated reasons:

1. 🚨 **τ is CENSORED.** Both runs tripped the travel guard with the mower still
   rotating; the 4 s settle never captured the full decay. τ ≥ 2.6–3.6 s is a
   floor, not a value. **A predictive controller needs the value.**
2. 🚨 **ω was measured at ONE commanded angular (|120|).** A feed-forward design
   predicts where the mower *will be* from what it was *told to do*, so it needs
   ω as a **function** of command. One point is not a function.

⚠️ **And a limit that no run fixes:** the probe cannot separate actuator lag from
observer lag — both present as "course keeps changing after the command stops".
τ must therefore be treated as an **effective loop dead time**, the quantity a
controller actually faces, and never as a mechanical property of the mower. The
~1 Hz bundle is the device's own choice (beta77 cadence matrix: requested
100/250/500/1000 ms, p95 **1119–1372 ms at every one**), so no faster feed exists
to separate them on this hardware.

## Run 1 — τ, uncensored

The settle phase must outlast the decay it is measuring, which today it did not.

| parameter | today | run 1 | why |
| --- | --- | --- | --- |
| `baseline_ms` | 2000 | **1500** | one informative chord is enough; buys settle budget |
| `step_ms` | 3000 | **2500** | ω was already resolved in 3 intervals |
| `settle_ms` | 4000 | **6000** | **the measurement.** Must outlast τ ≥ 3.6 s |
| total | 9000 | **10000** | ~2.60 m of path at the measured ~0.26 m/s |
| `max_travel_m` | 2.50 | **3.00** | 2.60 m of path needs headroom; 3.00 is the schema ceiling |
| `step_angular_speed` | +120 | **+120** | unchanged, so run 1 is comparable to today |

**Corridor: a 7.2 m square**, half-width 3.60 m ≥ the required
`3.00 + 0.50 = 3.50 m`. Corner reach 5.091 m against 5.965 m of yard clearance at
the max-inscribed point **(5.98, −5.24)** — fits with 0.874 m to spare.
⚠️ **Re-scan and re-verify containment at the live position before dispatch.** A
corridor cannot be reused across a run that moves the mower.

🔑 **Run 1 succeeds if the course goes FLAT before the window ends.** If it is
still changing at the last sample, τ is censored again and the run has failed on
its own terms — **record that and lengthen settle rather than quoting the number.**

## Run 2 — does ω scale with command?

Identical to run 1 except `step_angular_speed` **+180**, the top of the measured
120–180 band.

**The question:** is ω proportional to commanded angular, or does it saturate?

| outcome | reading | consequence for the design |
| --- | --- | --- |
| ω(180)/ω(120) ≈ 1.5 | proportional | feed-forward can use a linear command→rate map |
| ω(180) ≈ ω(120) | saturated | **the 2026-08-22 arc result repeats** (a 33% cut in angular moved the rate 3%), and the map is flat — a predictive design cannot modulate rate at all in this band |
| anything else | neither | record it; do not fit a curve to two points |

🗑️ **Two points do not make a law.** Whatever comes out, it is a *direction*, not
a calibration. `docs/frozen-prediction-constants-20260822.json` exists because
this project has fitted a yaw-rate law to sparse data before and had it refuted.

## What is NOT being run, and why

🗑️ **The opposite signs are not repeated.** 2026-08-29 measured |ω| at 4.694 vs
4.147 °/s across +120 and −120 — within 12%, so the drivetrain is not
meaningfully asymmetric. **That question is answered; re-running it buys nothing.**
🗑️ **No steering run, no controller, no closed loop.** Option B's controller does
not exist yet and must not be written before these two numbers do.

## Pass criteria — all four

1. The report stream starts and reports **ready** with no `readiness_reason`, and
   `position_sequence` advances through the window. *(Without this the run
   measures the probe, as four runs did on 2026-08-28/29.)*
2. At least **3 informative intervals in the settle phase**, and the last two
   agree to within 1.5 °/s — i.e. **the course has gone flat**.
3. `max |cross_track|` is not applicable; instead **containment holds**: every
   sample inside the corridor and the stop confirmed.
4. Gate disarmed afterwards and verified from the live API **and** RAW
   `core.config_entries`.

**Any failure is a FAIL**, recorded as such. A censored τ is a failed run, not a
smaller number.

## Preconditions

* Mower **docked and charged first** — it ran to 66% on 2026-08-29 and both runs
  here are longer than any so far.
* Off dock, `AREA_INSIDE`, RTK **Fix**, BLE live, blades off. Daylight not
  required: there is no VIO gate in this path.
* `route_start` read from the **live position after placement** —
  `_CONTINUOUS_MAX_START_DRIFT_M` is 0.30 m and that gate passes unconditionally
  in a dry run.
* A dry run showing **15/15 gates** and `blockers: []` on the exact configuration.

## What a pass authorizes

A pass authorizes **writing the feed-forward design document** — with its own
predeclared criteria — and nothing else. It does not authorize a steering run, a
controller change, or unparking Phase 2.
