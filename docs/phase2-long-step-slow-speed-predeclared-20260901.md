# Predeclaration — long step at reduced linear speed (2026-09-01)

**Written before any run at this configuration exists.** Authorizes the CODE
CHANGE and, separately, ONE run when an operator authorizes it. Nothing else.

## 1. The problem this addresses

`docs/findings-plus180-split-is-onset-sampling-phase-20260901.md` showed the two
+180/7000 runs agree on the plant to **0.195 °/s** once the onset interval is set
aside, and that the whole 2a verdict split came from **interval 1 alone**
(-5.68 vs -1.35, a 4.33 °/s difference). 2a is therefore partly measuring where
the ~1 Hz VIO sample boundary fell relative to the step command.

Model, verified against the repeat (10.43/3.5 = 2.98 predicted vs 3.4049
observed): with a first half of `k` intervals containing the single onset
interval, `half_diff ≈ |steady − onset| / k`. At the worst observed
contamination (10.43 °/s) the 1.5 °/s bound needs **k ≥ 7**, i.e. ~14 informative
step intervals ≈ a **14 s step** at the measured ~1.02 s VIO cadence.

## 2. Why the obvious change was rejected

Raising `step_ms` alone **cannot work at `linear_speed` 400.** Measured
0.2616 m/s over the 15 s window on 2026-09-01:

| | value |
| --- | --- |
| largest square contained in "Backyard Right" | **10.0 m** → radius 5.0 m |
| therefore max travel | **4.5 m** |
| therefore max window at linear 400 | **~17.2 s** → step ≤ ~10.2 s |
| worst-case `half_diff` at that maximum | **2.09 °/s — still FAILS the bound** |

🔑 **The yard runs out before 2a becomes robust.** A step-length raise on its own
would have spent a release on a config that still fails roughly half the time.

## 3. What changed instead

**`linear_speed` 300 becomes admissible** (default stays 400). It was eliminated
on 2026-08-30 because a 0.116 m/s mower cannot produce the **0.15 m chord** the
RTK course statistic required — but **E-VIO reads VIO heading between consecutive
distinct readings and imposes no travel floor at all**
(`_step_response_vio_intervals`), so that objection does not transfer. At
0.116 m/s a 23 s window travels **~2.5 m**, needing only a ~5.9 m corridor.

Changes, stated plainly:

| | from | to |
| --- | --- | --- |
| `linear_speed` | pinned `[400]` | `[300, 400]`, default 400 |
| `step_ms` ceiling | 7000 | **15000** |
| `_STEP_RESPONSE_MAX_TOTAL_MS` | 16000 | **23000** |
| `max_travel_m` | 4.5 | **4.5 — UNCHANGED** |

🔑 **This raises the CLOCK, not the distance.** The exposure bound that actually
carries the safety is untouched, and at the slower speed the mower travels *less*
than any step run so far.

🆕 **New pre-dispatch refusal `step_window_travel_exceeds_budget`.** A long window
at linear 400 would overrun the travel guard mid-run — safe, but it aborts the
window and censors the measurement, wasting a supervised run. The check uses a
**lower-bound** speed table, so it fires only when the window cannot fit even at
the slowest speed measured; a merely marginal config still dispatches and the
guard keeps carrying the safety. ⚠️ A first version of this used a rounded-UP
speed and refused the schema's own defaults — pinned against by
`test_the_travel_refusal_does_not_tighten_the_existing_defaults`.

## 4. The run this authorizes, when an operator authorizes it

`baseline 3000 / step 15000 / settle 5000` (23000 ms), **`step_angular_speed`
+180**, **`linear_speed` 300**, `max_travel_m` 4.5, corridor a 10.0 m square
(far more than the ~5.9 m required — no reason to use a tighter one).

**Scoring is unchanged and is not to be edited afterwards**: rule E-VIO exactly as
shipped. Dark VIO refuses with `vio_not_live_throughout`; a dark run is
UNSCOREABLE by design, so this needs daylight throughout.

| outcome | reading |
| --- | --- |
| 2a PASS | The long step does what the model predicts. τ becomes computable — still n=1 at this config. |
| 2a FAIL with a small `half_diff` | Model right, margin thin; report the number, do not re-tune. |
| 2a FAIL with a large `half_diff` | The step is not steady even at 15 s — a finding about the plant, not the statistic. |
| guard trip / `vio_not_live_throughout` | FAIL. Not a smaller number. |

⚠️ **Registered before the run:** `linear_speed` 300 is a **different operating
point**, so results are NOT directly comparable to the four banked runs at 400,
and the rotation rate at +180 may itself depend on linear speed — that is
unmeasured. **Do not merge this run into the n-count of the 400 runs.**

## 5. What this authorizes

The code change, a release, and a motion-disabled deploy. **One** run at the
configuration in §4, with explicit per-run operator authorization, daylight, a
docked-and-charged battery, a fresh corridor scan at the live position, and the
gate disarmed and verified from the live API and RAW afterwards. **Standing
decision 5 is untouched — Phase 2 continuous steering remains parked.**

---

## 6. CORRECTIONS to this document, made BEFORE any run at this configuration

Found by re-deriving the numbers and by an adversarial review of the change
(2026-09-01, same evening). **Both corrections are to my own reasoning above.
The predeclaration is corrected in place rather than quietly restated later.**

### 6.1 §2's travel figure was wrong: ~2.5 m should read ~3.7 m

§2 sized the run with **0.116 m/s**, which is a **4 s average INCLUDING ramp and
stop** (2026-08-30), not a sustained speed. At linear 400 the same ramp-inclusive
figure (0.191 m/s) understates the sustained 15 s measurement (0.2616 m/s) by
**1.37x**. Applying that factor to 300 gives **~0.159 m/s sustained**:

| | claimed in §2 | corrected |
| --- | --- | --- |
| sustained speed at linear 300 | 0.116 m/s | **~0.159 m/s** |
| travel over a 23 s window | ~2.5 m | **~3.65-3.7 m** |
| share of the 4.5 m budget | ~56% | **~81-82%** |

Derived twice independently (by hand and by the review) to 3.65 vs 3.7 m.

⚠️ **Additional risk the review surfaced and this document had not considered:**
the guard sums **per-sample |chord|**, so ~23 samples each carrying 2-4 cm of
position noise inflate `cumulative_travel_m` above true displacement. A longer
window accumulates more of that phantom travel, pushing the effective figure
higher still.

🚨 **There is NO sustained-speed measurement at linear 300 anywhere in the
record.** Every number above is an extrapolation from a single 4 s ramp-inclusive
average. **A travel-guard trip on this run is a live possibility, and a guard trip
is a FAIL, not a smaller number.**

### 6.2 §1's "k >= 7, hence a 14 s step" clears the bound only at typical cadence

Pooled VIO update cadence across all five banked runs is **991.1 ms** (n = 68,
median 1014.8, max 1316.5).

| cadence | intervals in a 15 s step | k | worst-case `half_diff` | verdict |
| --- | --- | --- | --- | --- |
| pooled mean 991 ms | 15.13 | 7.57 | **1.378** | passes by **8.1%** |
| slowest observed 1316 ms | 11.39 | 5.70 | **1.831** | **FAILS** |

🔑 **So 15000 ms is necessary but NOT sufficient.** §1 and §3 read as though the
longer step fixes 2a; it does not. It clears the bound at typical cadence and
fails if the feed runs slow — and cadence is the device's choice, not ours
(`docs/the-1hz-bundle-is-the-ceiling-20260822.md`).

**Consequence for §4's outcome table: a 2a FAIL on this run must NOT be read as
"the plant is not steady".** It is equally consistent with the cadence having run
slow. Report the realised interval count and cadence alongside any verdict.

### 6.3 What does not change

`max_travel_m` stays 4.5, the containment gate still requires 5.0 m of clearance
in every direction, and the travel guard still fails closed. These corrections
narrow the expected MARGIN and weaken the expected VERDICT; they do not weaken a
safety bound.
