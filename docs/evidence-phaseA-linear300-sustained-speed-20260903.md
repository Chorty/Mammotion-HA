# Phase A — sustained speed at linear 300 MEASURED, and it kills Phase B (2026-09-03)

**Result: the measurement succeeded and the conclusion is negative.** Predeclared
in `docs/phase-a-sustained-speed-300-predeclared-20260902.md` before any capture
existed. Raw: `docs/raw-samples/raw-phaseA-linear300-speed-20260903.json`
(sha256 `4a4578c1…`, 79 samples). Reposition:
`docs/evidence-phaseA-reposition-20260903.json`.

## 1. The measurement

| | value |
| --- | --- |
| whole-window (what the guard accumulated over 8.6 s) | **0.1574 m/s** |
| **sustained, post-ramp** | **0.223 m/s** |
| the extrapolation Phase B was sized with | 0.16 m/s |

Per-interval, showing the ramp explicitly:

```
0.0287  0.2084  0.1921  0.2848  0.2342  0.2062  0.2278   m/s
  ^ ramp from standstill        ^--------- sustained ---------^
```

🚨 **The whole-window figure matched the extrapolation to 1.6% — and that is
exactly why it was misleading.** An 8 s ramp-inclusive average understates a 28 s
window for the same reason the 4 s average understated the 8 s one. **The ramp
must be separated before any window is sized. This is the third time this trap
has appeared in eight days.**

## 2. What it kills

Sizing with the measured 0.223 m/s (ramp ~2 s / ~0.10 m, then sustained):

| window | travel | vs `max_travel_m` 4.5 |
| --- | --- | --- |
| **28 s (the proposed Phase B)** | **5.90 m** | **DOES NOT FIT** |
| max that fits | 21.7 s | 4.50 m |

A 21.7 s window leaves a **14.7 s step → k=7 → P(2a pass) 50.5%** — *worse* than
the 60.7% today's 15 s cap already gives.

🔑 **So the cap raise buys nothing at linear 300, and Phase B is withdrawn.** Per
§4 of the Phase A predeclaration this outcome was registered in advance as
legitimate and explicitly **not** a reason to raise `max_travel_m`.

## 3. Two corrections to constants, one of them safety-relevant

🚨 **`_PROBE_SPEED_PER_LINEAR_UNIT_MS` was 6% LOW — the unsafe direction**, since
it sizes corridor clearance in the clock-bound containment gate. Raised
**7.0e-04 → 7.5e-04**. Post-ramp speeds, measured:

| linear | sustained | implied constant |
| --- | --- | --- |
| 300 | 0.223 m/s (Phase A) | 7.43e-04 |
| 400 | 0.295 m/s (2026-09-01 run) | 7.38e-04 |

✅ 0.295 at linear 400 **independently matches the 0.280–0.293 m/s measured during
the 2026-08-12 arcs**, from a different service on a different day.

🗑️ **AND THE "NOT LINEAR" CLAIM IS REFUTED.** This project has said since
2026-08-30 that "a 25% command cut produced a 39% speed cut". Sustained:
**0.223 / 0.295 = 0.756 against a command ratio of 0.750** — essentially linear.
The non-linearity was an artifact of comparing 4 s **ramp-inclusive** averages,
where the slower run spends a larger fraction of its window ramping. **Do not
quote it as a property of the drivetrain.**

`_STEP_RESPONSE_TYPICAL_SPEED_BY_LINEAR` is now sustained speeds
`{300: 0.223, 400: 0.295}` rather than ramp-inclusive window averages.

## 4. Two open questions closed for free

✅ **The mower DOES rotate at `(linear 300, angular 120)`** — VIO step rate
−1.15 °/s over the single step interval. There is no actuation deadband at the
slower linear speed, which was an explicit unknown.

✅ **`vio_analysis` returned `scoreable: true`** even at a 1 s step. ⚠️ Its 2a
verdict is meaningless here (one informative step interval against the rule's ≥3)
and **must not be quoted** — the run was never intended to score 2a.

## 5. Safety

15/15 gates, `blockers: []`, `reason: window_complete`, `aborted_early: false`.
Travel **1.356 m of the 2.5 m budget (54%)**, against a 1.28 m projection. Stop
confirmed. Gate armed only for each dispatch and verified disarmed afterwards from
the live API **and** RAW `core.config_entries`. Preceded by a 4.5555 m
repositioning drive that reached `target_reached` at **0.06544 m**, the best
landing on record. Corridor clearance 5.8612 m against 3.00 m required (1.95x).
Battery 100%, VIO 80/80 `light`/`signal_good`, RTK Fix, BLE −56 dBm.

## 6. What this authorizes

**Nothing further.** Phase B is withdrawn on its own terms, not deferred. The open
question — a replacement statistic for 2a — is unchanged and is offline work.
🛑 Standing decision 5 untouched.
