# Findings: RTK square-return repeat, four turns (2026-09-17)

**INCONCLUSIVE on the headline, for a second time — the ground truth was lost.**
The run itself was clean: four legs, four turns, no aborts, heading restored to
within ~3.5°. But the operator's start marker could not be found afterwards, so
RTK's claimed **0.288 m** closure has nothing to be compared against.

Predeclaration: `docs/predeclared-rtk-square-return-20260916.md` (unchanged —
its §0 always specified four turns). Prior run:
`docs/findings-rtk-square-return-20260916.md`. Evidence:
`docs/evidence-rtk-square-return-repeat-20260917/` (`rtk_square_real.json`,
`rtk_square_dry.json`).

---

## 1. What was fixed before the run

`scripts/rtk_square_return.py` drove a turn after legs 1–3 only (`if index < 3`),
which is what cost the 2026-09-16 run its headline number: the mower closed on
its start *position* ~100° off its start *heading*, so every body point
displaced differently and the tape-vs-RTK comparison inherited the unknown
antenna offset.

Fixed in `fa520e29`, **committed before the first pulse**, with tests that fail
on the old behaviour (`tests/scripts/test_rtk_square_return.py`: four turns
driven, gate disarmed even when the new fourth turn aborts).

## 2. The run

22:54:38Z → 23:11:52Z. Gate armed once, disarmed in the `finally` block,
**verified disarmed afterwards in the live API and RAW `core.config_entries`**.
Start `map_xy (5.2552, −4.1545)`, inside Backyard Right with **5.19 m to the
nearest area edge** and **6.04 m to the nearest keep-out** (live `export_map`
scan before dispatch). Operator gave a standing go for the square.

| Leg | Length | Bearing driven | Turn | Commanded | Achieved | Rate |
| --- | --- | --- | --- | --- | --- | --- |
| leg1 | 2.566 m | 278.592° | turn1 | 6000 ms | +86.681° | 14.447 °/s |
| leg2 | 2.843 m | 5.273° | turn2 | 6229 ms | +96.139° | 15.434 °/s |
| leg3 | 2.786 m | 101.412° | turn3 | 5831 ms | +83.722° | 14.358 °/s |
| leg4 | 2.727 m | 185.134° | turn4 | 6268 ms | — (no leg 5) | — |

All legs took 2 pulses. RTK held `Fix` throughout. **RTK-claimed closure
0.2877 m**, end `(5.2747, −3.8675)`.

**Heading was restored.** Turns 1–3 sum to 266.5°; turn 4 was commanded at
turn 3's measured rate for ~90°, giving ~356.5° total — a residual of about
3.5°. At that residual a body point 0.5 m from centre displaces only ~3 cm
differently from the antenna, so a tape measurement *would* have been directly
comparable. The method worked; only the measurement failed.

## 3. Why it is inconclusive

The start marker was not findable in the grass afterwards. Without it there is
no independent distance, so **0.288 m remains RTK's own claim about itself.**

🗑️ **The camera frames are NOT a substitute, and no number is quoted from them.**
The operator supplied before/after UniFi frames and confirmed fixed framing.
That much checks out — registering the static scene gives a **0 px vertical,
1 px horizontal** shift, so the camera did not move and pixel differences are
real motion. Template matching puts the mower **~14 px** from its start
(dy +12, dx −7), NCC peak 0.61.

**But every candidate ruler failed:**
- The mower's own bright shell measures **91 px wide in the first frame and
  130 px in the second** — the shots are ~25 min apart and the later sun washes
  out its edges.
- Mowing-stripe period disagrees with itself across three patches beside the
  mower (**50 / 40 / 83 px**, weak peaks), so the stripes are not one spacing.
- Any fixed object far from the mower (hot tub, fence) needs a homography to
  transfer scale across the frame, which needs ≥4 known ground points.

⚠️ **One unresolved hint, recorded and not acted on:** if the mower is roughly
0.5–0.6 m wide and spans ~70–90 px, 14 px is ~8–11 cm — materially *less* than
RTK's 0.288 m, which would mean RTK **overstates** its closure error. That rests
on a width that could not be measured cleanly, and on a moderate NCC peak. It is
a hypothesis for the next run, not a result.

## 4. What this run does establish

- ✅ **Four turns drive correctly on hardware.** The script defect is fixed and
  exercised; heading came back to ~3.5°.
- **Distance repeatability:** 2.566 / 2.843 / 2.786 / 2.727 m on identical
  two-pulse commands — spread 0.28 m, ~10 %. Leg 1 was the short one here;
  leg 4 was the short one (2.446 m) on 2026-09-16. ⚠️ **Which leg is short is
  not consistent between runs**, so this is variance, not a position effect.
- 🔑 **The monotonic turn-rate decline did NOT reproduce.** Tonight:
  14.447 → 15.434 → 14.358 °/s. 2026-09-16: 16.01 → 15.52 → 13.25 °/s. Same
  command (angular 202, pure rotation), opposite pattern in the middle turn.
  **Battery sag is weakened as an explanation**; n = 2 runs settles nothing, but
  a systematic decline would have repeated.
- **Pure-rotation calibration now has six points** at angular 202, spanning
  13.25–16.01 °/s, mean ≈ 14.9. ⚠️ Do not fit a law to them.
- RTK held `Fix` for a 17-minute run, on a night it had been `float` for
  13 hours beforehand (06:40Z → ~22:16Z).

## 5. What the next attempt must change

🚨 **Stop using a ground marker.** It gets lost in grass and the mower drives
over it. Instead, before the run, tape-measure from **two permanent landmarks**
(e.g. a named fence post and the hot tub's near corner) to one identified body
point on the mower, and repeat both measurements afterwards. Two distances from
fixed references pin the position with nothing to lose, and they are more
precise than chalk in grass.

Everything else is ready: the runner is correct, the method is VIO-free and
needs no daylight, and it takes ~15 minutes.
