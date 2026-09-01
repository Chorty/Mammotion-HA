# Predeclaration — route-1 +180/7000 repeat, scored by E-VIO (2026-09-01)

**Status: predeclared before any capture exists. This document authorizes ONE
run and nothing else.**

## 1. Why this run

`docs/evidence-route1-run2-plus180-fail-20260830.json` recorded +180/7000 as an
overall FAIL under the then-registered RTK chord statistic. beta95 replaced that
statistic with rule **E-VIO** (`docs/findings-rtk-vio-course-rate-scoring-20260831.md`),
under which the same banked run **flips to 2a PASS** with `tau = 0.80 s`.

That flip is **n = 1 and is pinned as a regression fixture, not blessed**
(`tests/components/mammotion/test_step_response_vio_scoring.py`). It is also the
only VIO-derived time constant the programme has. This run takes it to n = 2.

⚠️ **E-VIO has never populated on hardware.** Dry runs return before sampling, so
`vio_analysis` first computes on this run. A null or refused `vio_analysis` is
itself a reportable outcome.

## 2. Configuration — identical to the 2026-08-30 R2 run, nothing changed

| field | value |
| --- | --- |
| `route_start` | (6.0038, -5.3077) — the R2 start, re-verified |
| `corridor_polygon` | 10.0 m square centred on `route_start` |
| `baseline_ms` / `step_ms` / `settle_ms` | 3000 / 7000 / 5000 (15000 ms) |
| `step_angular_speed` | **+180** |
| `linear_speed` | 400 (pinned) |
| `max_travel_m` | 4.5 |
| `motion_refresh_interval_ms` / `sample_interval_ms` | 200 / 100 |

Clearance at `route_start` re-measured today: **5.9039 m** against the probe's
`max_travel_m + 0.50 = 5.00 m` requirement (**1.18x**). The 2026-08-30 run
recorded 5.904 m at the same point, so the map is unchanged.

⚠️ **Registered before dispatch:** the corridor's far top-right corner
(~(10.50, -0.81)) lies outside the "Backyard Right" polygon. This is inherited
verbatim from the authorized R2 geometry. It is unreachable under a 4.5 m travel
cap from the centre, and 10.0 m is the *minimum* square that admits
`max_travel_m = 4.5` under the probe's own containment rule — shrinking it would
refuse the run rather than make it safer. Recorded as a known property of the
corridor, not discovered afterwards.

## 3. Scoring — E-VIO, already shipped; NOT to be edited after the fact

Scoring is whatever beta95's deployed code computes. It is not re-derived here
and **must not be adjusted once the numbers are visible**:

- **2a** — half-phase mean-rate agreement across the step, on VIO heading.
- **2b** — last-two settle rates with carryover, on VIO heading.
- `omega`/`tau` come from the same channel; **`tau` exists only when 2a passes.**
- Any sample with `vio_state != 2` ⇒ `vio_not_live_throughout`, refuse to score.
  There is no RTK fallback. A dark run is UNSCOREABLE by design.
- RTK `course_series`/`analysis` are emitted as diagnostics only and **carry no
  verdict**.

A travel-guard trip is a **FAIL**, not a smaller number.

## 4. What each outcome means

| outcome | reading |
| --- | --- |
| 2a PASS, tau near 0.80 s | The R2 flip reproduces. n = 2; tau becomes quotable *with* its n. |
| 2a PASS, tau far from 0.80 s | Plateau is real, its value is not repeatable. tau stays unquotable. |
| 2a FAIL | The banked flip does not generalise. E-VIO's +180 pass was n = 1 luck. |
| `vio_not_live_throughout` | Refusal working as designed. Says nothing about the plant. |

## 5. What this authorizes

**Nothing beyond this single window.** Not a +120 repeat, not another `step_ms`
change, not a cap raise, and explicitly **not resumption of Phase 2 continuous
steering** — standing decision 5 is untouched. This repairs the n on a parked
instrument's one measurement; it does not unpark the work.

## 6. Preconditions, all verified before dispatch

- Daylight with VIO live throughout (dark ⇒ unscoreable).
- Operator on site, supervising, area clear, blades off, e-stop reachable.
- A repositioning drive of 5.3969 m at map bearing 285.507° is required first
  (accepted profile verbatim, verified key-by-key).
- Gate armed only for the run; disarmed and verified afterwards from the **live
  API and RAW** `core.config_entries`.

Battery at authorization: **45%**, off dock. Operator elected to proceed;
recorded here because it is below the standing dock-and-charge guidance.
