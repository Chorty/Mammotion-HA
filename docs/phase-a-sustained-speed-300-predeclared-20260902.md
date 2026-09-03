# Phase A — predeclared: measure sustained speed at linear 300 (2026-09-02)

**Predeclared before any capture exists. Authorizes ONE short run when an
operator authorizes it, in daylight. Nothing else.**

## 1. Why

Every figure this project has for `linear_speed` 300 is **one 4 s ramp-inclusive
sample (0.116 m/s)** scaled by a 1.37x ratio observed at linear 400. There is no
sustained measurement. That single unmeasured number now gates real decisions:

- it sizes any long-step window, and a **+19% error censors the run outright**
  (`docs/phase2-long-step-cap-raise-predeclared-20260902.md` §3);
- `_STEP_RESPONSE_TYPICAL_SPEED_BY_LINEAR[300] = 0.16` and
  `_STEP_RESPONSE_MIN_SPEED_BY_LINEAR[300] = 0.10` are both extrapolations;
- **`(linear 300, angular ±180)` has never been exercised at all**, and every
  onset/scatter constant behind criterion 2a was measured at linear 400.

🔑 **Phase A needs NO code change.** Every value below is inside the deployed
beta97 schema.

## 2. Configuration

| field | value |
| --- | --- |
| `baseline_ms` / `step_ms` / `settle_ms` | **3000 / 1000 / 4000** (8000 ms) |
| `linear_speed` | **300** |
| `step_angular_speed` | **120** (the gentlest measured value) |
| `max_travel_m` | **2.5** (the schema default, well under the 4.5 ceiling) |
| `motion_refresh_interval_ms` / `sample_interval_ms` | 200 / 100 |
| corridor | 10.0 m square centred on the live start, as used 2026-09-01 |

**Exposure is far below anything run recently:** an 8 s window against the 15.2 s
of 2026-09-01, and a 2.5 m budget against 4.5 m. At the estimated 0.16 m/s it
travels ~1.3 m; even at 0.25 m/s it travels ~2.0 m, inside the budget.
⚠️ `step_ms` is deliberately **1000 ms** — this run measures SPEED, not 2a, and a
short step keeps the path close to straight.

## 3. The measurement

**Primary:** cumulative path travel over the full window ÷ elapsed, computed the
same way `_apply_travel_guard` accumulates it (sum of per-sample `|chord|`) — so
the number is directly comparable to the guard that will consume it.

**Also report, because they are free and both are open questions:**
- travel over the step phase alone, to see whether angular 120 at linear 300
  changes forward speed;
- the VIO rate series, the first evidence of whether the mower rotates at all at
  `(300, 120)` — if the angular command sits in a deadband at this linear speed,
  the entire 2a argument fails to transfer and that is the finding.

⚠️ **This run does NOT score 2a.** With a 1000 ms step there will be ~1
informative step interval against the rule's ≥3, so `vio_analysis` is expected to
be unscoreable. **That is by design and is not a failure.**

## 4. What each outcome authorizes

| measured sustained speed | reading |
| --- | --- |
| ≈ 0.16 m/s | the extrapolation holds; Phase B can be sized from it with margin |
| < 0.16 m/s | better — a longer window fits inside `max_travel_m` 4.5 |
| **> ~0.17 m/s** | **Phase B may not fit at `max_travel_m` 4.5 at all.** That is a legitimate outcome and **NOT** a reason to raise the exposure bound. |
| mower barely rotates at (300, 120) | angular authority is linear-speed dependent; the 2a constants do not transfer and the long-step plan needs rethinking |

**Phase A authorizes nothing beyond itself.** Phase B is a separate release, a
separate predeclaration, and a separate per-run authorization.

## 5. Preconditions

Daylight (the repositioning drive closes turns on VIO), operator on site with the
e-stop reachable, docked-and-charged battery, a fresh corridor scan at the live
position, per-run go/no-go immediately before dispatch, and the gate disarmed and
verified from the live API **and** RAW `core.config_entries` afterwards.

🛑 Standing decision 5 is untouched — Phase 2 continuous steering stays parked.
