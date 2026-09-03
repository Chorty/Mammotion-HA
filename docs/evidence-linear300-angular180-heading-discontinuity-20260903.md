# (linear 300, angular 180): the speed transfers, the rotation is UNMEASURED — and E-VIO scored a heading-frame jump (2026-09-03)

Predeclared in `docs/predeclared-linear300-angular180-20260903.md` (commit
`740e5995`) before any capture existed. Raw:
`docs/raw-samples/raw-linear300-angular180-20260903.json` (79 samples).

## 1. 🚨 The headline is a DEFECT IN THE SCORING RULE, not a plant measurement

`vio_analysis` returned **`scoreable: true`** and a step rate of
**−149.794 °/s**, against ~−11.5 to −12 °/s at `(400, 180)`. **That number is
not rotation.** The mower did not rotate.

At **t = 4018 ms**, in a single report:

| channel | before | after | jump |
| --- | --- | --- | --- |
| VIO heading | 89.769 | −76.705 | **−166.47°** |
| `toward` | 97.273 | 166.394 | **+69.12°** |
| position | advancing smoothly at ~0.23 m/s | | — |

🔑 **The independent RTK channel refutes rotation outright.** Course from
position chords across the whole window: **−74.5 → −76.5 → −73.2 → −79.8 →
−77.4 → −79.9 → −80.5** — about **6° total**, i.e. essentially straight. Chords
were 0.19–0.26 m and all informative.

🔑 **And the two heading fields jumped by DIFFERENT amounts at the same instant**
(−166.47 vs +69.12), which no single physical rotation can produce. This is a
**heading-frame discontinuity** — most plausibly VIO re-referencing after the
operator restarted the mower minutes earlier.

### Why this matters more than the run's stated purpose

⚠️ **`vio_state` remained `2` for all 79 samples, so E-VIO's only liveness guard
never fired.** `vio_not_live_throughout` checks *state*, not *continuity*. A
frame jump inside a real step phase would be silently converted into a rate and
scored as a measurement.

**This is a fail-closed gap in the shipped criterion**, found by a run that was
looking for something else. It is exactly the failure class the programme has
been fighting: an instrument reporting a number it has not earned.

⚠️ **NOT YET FIXED.** A continuity guard — refuse to score when a between-reading
heading delta exceeds what the commanded angular rate can produce — is the
obvious shape, but it must be **predeclared before implementation**, since it
changes when the criterion refuses.

## 2. What the run DID measure: sustained speed at (300, 180)

Position-derived, so unaffected by the heading discontinuity:

| | value |
| --- | --- |
| cumulative path | **1.6206 m** over 8.035 s |
| whole-window speed | **0.2017 m/s** |
| **post-ramp sustained** | **0.2279 m/s** |
| Phase A at `(300, 120)` | 0.223 m/s |

🔑 **Forward speed at linear 300 is unchanged by the angular command** — 0.2279
at angular 180 against 0.223 at angular 120, a 2% difference. That is consistent
with the 2026-08-12 arc sweep, where speed held ~0.28 m/s across angular
180/300/500 at linear 400. **The `_STEP_RESPONSE_TYPICAL_SPEED_BY_LINEAR[300] =
0.223` constant is corroborated at a second angular command.**

## 3. What remains UNMEASURED

🗑️ **The rotation rate at `(linear 300, angular 180)` is still unknown.** The
operating point remains unexercised for its angular behaviour, and every 2a
constant (Δ ≈ 10.43 °/s, sd 1.445 °/s, ~12 °/s steady) is still measured only at
linear 400. §4 of the predeclaration cannot be answered from this run.

⚠️ **Do not quote −149.794 °/s, and do not quote this run's 2a verdict.** The
predeclaration said in advance that a 1000 ms step yields ~1 informative interval
against the rule's ≥3, so no 2a verdict here is meaningful — that caveat now has
a second, stronger reason.

## 4. Safety

**15/15 gates**, `blockers: []`, `reason: window_complete`,
`aborted_early: false`, **0 of 79 samples tripped the travel guard**. Travel
**1.621 m of the 2.5 m budget (65%)** against a 1.784 m projection. Stop
confirmed. Gate armed only for the dispatch and disarmed afterwards. First run on
**beta99**, whose containment reported `bound_that_binds: travel_budget`
(required 3.00 m, clock bound 2.30 m) against 4.5841 m of clearance.

⚠️ Registered in the predeclaration and unchanged by the outcome: `ble_rssi` was
**−70**, at the documented wall, and `current_orientation.trustworthy` was
already **false** with a 97.08° disagreement before dispatch. **That standing
disagreement is now better explained** — the heading frame was already displaced
when the run began, and it jumped again mid-window.

## 5. What this authorizes

**Nothing further.** Not a 2a run, not a longer window, not Phase 2 — standing
decision 5 untouched. The continuity-guard question is a separate, predeclared
piece of work.

---

## 6. 🗑️ CORRECTION — the run could NEVER have measured rotation. My design error.

**The operator observed the mower go straight**, which the data confirms and
which corrects §3's framing.

RTK course, the trustworthy channel here, per informative interval:

```
baseline  -3.28   step  +2.93   settle  -7.21  +2.34  -2.51  -0.53   deg/s
```

Signs alternate; total course change is **−5.99° across 6.91 s**. That is chord
noise on a straight path, not rotation. ✅ **The operator's visual observation is
independent corroboration that the −166° VIO jump was not physical.**

🚨 **Why it went straight is a flaw in MY configuration, not a property of the
mower.** The step phase was **1000 ms**. This project measured on 2026-08-29 that
**rotation does not start for ~1–2 s** (onset lag). **A 1 s step ends before
rotation begins**, so this run could not have produced rotation at *any* angular
command.

I copied `step_ms: 1000` from Phase A, where it was correct because Phase A
measured **speed**. Carrying it into a run whose stated purpose was **rotation**
made the primary measurement impossible before dispatch.

⚠️ **§4 of the predeclaration is therefore unsound as written.** Its "barely
rotates ⇒ deadband at (300, 180)" reading **must not be applied to this run** —
absence of rotation here is fully explained by the step length. **There is no
evidence of a deadband, and none against one.**

**What survives unchanged:**
- ✅ the sustained-speed result (§2), which is position-derived;
- ✅ the heading-discontinuity defect (§1), which is independent of step length
  and is now visually corroborated.

**What a real rotation test at (300, 180) needs:** a step of at least ~5 s, so
the phase outlasts the onset lag — the same arithmetic that drove the 5000 →
7000 ms extension in 2026-08-30. That is a separate predeclared run, and it is
**not** authorized here.

🔑 **The transferable lesson: a phase length is part of the hypothesis.** Copying
a duration from a run with a different purpose silently invalidated the
measurement, and the predeclaration did not catch it because it reasoned about
outcomes without checking the configuration could produce them. This is the same
class as the 2026-08-28 standing check — *confirm each bound exceeds what the
run needs to demonstrate its criterion* — which exists precisely for this.
