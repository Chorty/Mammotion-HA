# Rotation at (linear 300, angular 120) — the predeclared prediction HELD (2026-09-03)

Predeclared in `docs/predeclared-linear300-angular120-20260903.md` (`57d18fff`),
**including a numeric prediction, before any capture existed.** Raw:
`docs/raw-samples/raw-linear300-angular120-step5000-20260903.json` (119 samples).

## 1. The prediction, and the result

§3 of the predeclaration committed to a falsifiable number in advance:

> At `(400, 120)` the steady rate was ~−8.2 °/s. At angular 180 the rate rose
> ~13% when linear dropped 400 → 300. **If that scaling is a property of linear
> speed, expect ~−9.3 °/s here.**

| | value |
| --- | --- |
| **predicted** | **~−9.3 °/s** |
| **measured** | **−9.175 °/s** (sd 0.484, n = 4) |
| error | **1.3%** |

✅ **The alternative hypothesis is excluded.** "Rotation is independent of linear
speed" predicted ~−8.2 °/s; the measurement sits **2.0σ** away from that and
**0.26σ** from the +13% prediction.

✅ **Independently corroborated by RTK course**: step rates **−7.04, −9.76,
−9.00** once rotation is established — the same figure from a channel sharing no
mechanism with VIO.

## 2. What the pair now establishes

Both angular commands measured at linear 300, with a 5 s step:

| | at linear 400 | at linear 300 | change |
| --- | --- | --- | --- |
| angular 120 | ~−8.2 °/s | **−9.175** | **+12%** |
| angular 180 | ~−11.8 °/s | **−13.431** | **+13%** |

🔑 **A consistent ~+12–13% rise in yaw rate when linear speed drops 400 → 300, at
both angular commands.** Predicted at one and confirmed at the other, so this is
no longer a single observation — it is a small, reproducible effect.

⚠️ **Do not fit a law to two points.** Both are n = 1 per configuration, and the
mechanism is unexplained. Plausibly less forward momentum resisting yaw, but this
run cannot show that.

**Onset deficit** is **7.70 °/s** here against 11.39 at `(300, 180)` and 10.43 at
`(400, 180)` — it scales with the commanded rate, as expected, and remains the
dominant term in 2a's bias.

## 3. 🚨 A sixth demonstration of the onset bias — the cleanest yet

2a **FAILED** at `half_diff` **2.8364**. Look at why:

```
step rates:  -1.47   -8.57  -9.42  -9.68  -9.03
              ^onset  ^------ steady, sd 0.484 -----^
```

🔑 **The four steady intervals have a standard deviation of 0.484 °/s — the
tightest plateau in the entire banked corpus.** The plant unambiguously reached
steady rotation. The statistic still says it did not, because the onset interval
drags the first half to −6.49 against the second's −9.32.

**If any single run demonstrates that 2a measures onset placement rather than
steadiness, it is this one.** ✅ 2b passed at **0.0515 °/s**, the tightest settle
agreement on record. `omega`/`tau` correctly null.

## 4. Safety

**15/15 gates**, `blockers: []`, `window_complete`, `aborted_early: false`,
**0 of 119 samples tripped the travel guard**, travel **2.5025 m of 3.0 (83%)**,
stop confirmed, gate disarmed and verified. No heading discontinuity. VIO live
throughout despite the run finishing close to the dusk cliff.

Yard clearance at the live start was **3.9241 m** against 3.50 m required —
verified by scan against the map, which the containment gate does not check.

## 5. What this authorizes

**Nothing further.** Standing decision 5 untouched.
