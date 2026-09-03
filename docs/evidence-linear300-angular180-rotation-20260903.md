# (linear 300, angular 180) rotation MEASURED — the linear-400 constants transfer (2026-09-03)

Predeclared in `docs/predeclared-linear300-angular180-20260903.md` (`740e5995`).
Raw: `docs/raw-samples/raw-linear300-angular180-step5000-20260903.json` (117
samples). Reposition: `docs/evidence-linear300-angular180-reposition-20260903.json`.

**This supersedes the 1000 ms attempt earlier the same evening**
(`docs/evidence-linear300-angular180-heading-discontinuity-20260903.md`), whose
step was too short to outlast the onset lag — a design error corrected in its §6.
The fix was the step length: **5000 ms**.

## 1. The measurement, and both channels agree

VIO step-phase rates: **−2.04** (onset), then **−11.84, −13.89, −15.35, −12.65**.

| | value |
| --- | --- |
| steady rotation, ex onset | **−13.431 °/s** (sd 1.530, n = 4) |
| at `(400, 180)` | ~−11.5 to −12 °/s |
| difference | **~+13%, i.e. slightly FASTER at the lower linear speed** |

✅ **Independently corroborated by the RTK course channel**, which shares no
mechanism with VIO: step rates **−13.41, −17.37, −9.55** and settle **−11.79,
−14.57**, mean ≈ **−13.4 °/s** — the same figure.

🔑 **There is NO deadband at `(300, 180)`.** The operating point works, and the
prior run's apparent absence of rotation was entirely its 1 s step.

## 2. What this settles for criterion 2a's constants

Every 2a constant was measured at linear 400 and carried a standing caveat that
it might not transfer. It does, approximately:

| constant | at linear 400 | at linear 300 |
| --- | --- | --- |
| steady rotation at angular 180 | ~11.8 °/s | **13.43 °/s** |
| onset deficit Δ | 10.43 °/s | **11.39 °/s** |
| residual scatter sd | 1.445 °/s | **1.530 °/s** |

🔑 **All three agree within ~15%.** The onset-bias diagnosis, the admissibility
arithmetic, and the aliasing threshold were all derived at linear 400 and are
**not invalidated by a change of linear speed**. ⚠️ n = 1 at 300 — treat these as
corroboration that the constants transfer, **not** as replacement values.

## 3. 🚨 And it is a fifth independent demonstration of the onset bias

2a **FAILED** at `half_diff` **4.3966** (halves −9.534 / −13.931), with
`omega` and `tau` correctly **null**.

The cause is visible in one line: the steady rates are **−11.84, −13.89, −15.35,
−12.65** — tightly clustered — while the onset interval reads **−2.04**. The
first half carries that onset and is dragged to −9.53. **The plant reached steady
rotation; the statistic says it did not.** This is exactly the bias documented in
`docs/findings-2a-replacement-20260903.md`, now seen at a new operating point.

⚠️ **Do not read this FAIL as a plant finding.** Per the predeclaration this run
is a characterisation; its 2a verdict is reported for what it demonstrates about
the *instrument*, not about the mower.

✅ 2b passed (1.1183 °/s).

## 4. Safety

**15/15 gates**, `blockers: []`, `reason: window_complete`,
`aborted_early: false`, **0 of 117 samples tripped the travel guard**. Travel
**2.5477 m of the 3.0 m budget (85%)**. Stop confirmed. Containment reported
`bound_that_binds: travel_budget` (required 3.50 m, clock bound 3.20 m).

🔑 **A pre-dispatch scan refused the first candidate position and it mattered.**
At (6.776, −8.356) the yard clearance was **2.8447 m** against a **3.20 m clock
bound** — the run would have been permitted by the gate, because
`step_path_contained` measures against the **operator-supplied corridor**, not
the mowing area. ⚠️ **The gate does not check the map.** The corridor must be
verified against the area separately, every time.

Repositioned 3.374 m to (5.93, −5.09) first: `target_reached`, landing
**0.12107 m**, clearance **5.8089 m**. A hard in-script check refused to dispatch
the window unless the landing gave ≥3.5 m.

✅ **No heading discontinuity this run** — the −166° VIO jump seen earlier did not
recur, consistent with it having been a re-reference after the mower restart.
**The continuity-guard gap in E-VIO remains real and unfixed** regardless.

## 5. What this authorizes

**Nothing further.** Not a 2a run, not Phase 2 — standing decision 5 untouched.
