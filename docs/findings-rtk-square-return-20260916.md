# FINDINGS — RTK square-return test, 2026-09-16 (03:03–03:18Z)

Predeclared in `docs/predeclared-rtk-square-return-20260916.md` (`688f5714`),
runner committed before the first pulse (`c9467099`). Evidence:
`docs/evidence-rtk-square-return-20260916/`.

Ran fully dark (`vio_tracked_features` 0, `camera_brightness` dark) on raw
timed velocity pulses — no VIO, no click-to-path, no heading feedback.

---

## 0. Verdict

**The tape and RTK are mutually consistent — nothing here shows RTK is wrong.
But this run does NOT bound RTK's accuracy**, because of an implementation
error (§3) that left ~100° of residual rotation and made the comparison
depend on the unknown antenna position.

🔑 **What is solidly established is repeatability, not absolute accuracy.**

---

## 1. What ran

| leg | length | bearing | | turn | commanded | achieved |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.709 m | 279.1° | | 1 | 6000 ms | **+96.0°** |
| 2 | 2.681 m | 15.2° | | 2 | 5622 ms | **+87.2°** |
| 3 | 2.780 m | 102.4° | | 3 | 5799 ms | **+76.8°** |
| 4 | 2.446 m | 179.3° | | | | |

Four legs, **three** turns, zero aborts, RTK `Fix` throughout, gate armed
03:03:22Z and disarmed 03:18:28Z. Start (5.1563, −4.6299) → end
(5.2305, −3.8053); **RTK-claimed closure 0.828 m**.

Operator tape, front-left and rear-right tire contact patches (start mark →
end mark): **46.5″ = 1.181 m** and **44″ = 1.118 m**.

---

## 2. Reconciling tape against RTK

Total rotation was 260.14°, leaving **99.86° residual**. Under a rigid-body
transform every point swings about a common pivot, and displacement scales as
`2·sin(θ/2)·r` = **1.5306 · r** at this angle. Working each measurement back to
its radius from that pivot:

| point | measured displacement | implied radius from pivot |
| --- | --- | --- |
| front-left tire | 1.181 m | 0.772 m |
| rear-right tire | 1.118 m | 0.730 m |
| RTK antenna | 0.828 m | 0.541 m |

🔑 **The two wheels land within 41 mm of each other despite being diagonal
corners**, which places the pivot near the perpendicular bisector of that
diagonal. The body centre then sits 0.64–0.69 m from the pivot (for a
0.60–0.80 m diagonal), implying the **antenna is ~10–15 cm off the wheelbase
centre** — an ordinary mounting position.

**So RTK reporting less displacement than either tire is exactly what the
geometry predicts, not a discrepancy.**

⚠️ **This is consistency, not a bound.** The unknown antenna position and the
large residual rotation leave enough freedom that an RTK error of 10–20 cm
would fit these numbers equally well. This run cannot separate "RTK is
excellent" from "RTK is off by 15 cm."

---

## 3. 🚨 Implementation error: three turns, not four

The operator's design — and §0 of the predeclaration — called for "drive 2 m,
turn left 90°, **four times**." The runner drove four legs with only **three**
turns between them. That traces the square and returns near the start
*position*, but leaves the mower ~100° off its original *heading*.

**Consequence:** with heading restored (4 turns), every body point displaces
identically, and a single tape measurement equals RTK's number directly — no
geometry, no antenna unknown. With heading not restored, the comparison needs
the reconstruction in §2 and inherits its ambiguity. **The error is what cost
this run its headline number.**

Guidance for the repeat: `scripts/rtk_square_return.py` must drive a turn
after *every* leg, including the last. The operator had already measured and
removed the markers before the shortfall was noticed, so the 4th turn could
not be added retroactively.

---

## 4. What this run does establish

- **Distance repeatability.** Identical two-pulse commands produced
  2.709 / 2.681 / 2.780 m on legs 1–3 — within ~1% — and per-pulse 1.334 vs
  1.324 m on consecutive legs. Leg 4 was the outlier at 2.446 m (its first
  pulse ran 1.163 m against ~1.33–1.39 m elsewhere); unexplained.
- **RTK stationary jitter ~2 mm** read-to-read (seen during the dry run).
- **RTK held `Fix` for the whole 15-minute run**, tracking a coherent closed
  path, on a night when it had spent 47 minutes in `Float` shortly before
  (02:01:59–02:48:19Z — the third such night-time episode after 2026-09-12 and
  2026-09-13).
- 🔑 **New rotation calibration at angular 202, pure rotation (linear 0):**
  **16.01 / 15.52 / 13.25 °/s** across turns 1–3. The banked points were
  9.175 °/s (angular 120) and 13.431 °/s (angular 180), both measured *with*
  linear 300 applied — these are the first pure-rotation figures.
- ⚠️ **The rotation rate declined monotonically across the three turns**
  (16.01 → 15.52 → 13.25 °/s), each turn calibrated from the one before and
  still overshooting the prediction downward. Cause unknown — battery sag,
  terrain, or wheel slip are all candidates, none tested. **This is a real
  finding and it matters for any open-loop turn budget.**

---

## 5. What is next (decisions, not actions)

1. **A repeat with four turns would give the clean number** this run missed:
   heading restored ⇒ one tape measurement directly comparable to RTK, no
   geometry. ~15 minutes, and it needs no daylight (this whole method is
   VIO-free).
2. **The declining turn rate deserves its own look** — it is independent of
   RTK and bears directly on the `turn_budget_infeasible` halts seen in the
   Phase 1 sessions.
3. 🛑 **Standing decision 3 (accuracy CLOSED) is untouched.** This measured
   raw RTK position against ground truth with no VIO and no correction; it is
   not a click-to-path landing-accuracy result and must not be quoted as one.
