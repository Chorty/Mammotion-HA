# Phase 2 steering attempt 5 — criteria and configuration predeclared BEFORE dispatch

**Written 2026-08-28, after attempt 4 and before attempt 5 exists.** Nothing in
this file may be edited once a capture exists. It supersedes the *configuration*
in `docs/phase2-steering-run1-predeclared-20260827.md` and **repairs one of its
seven criteria**, with the reasoning stated below. It does not touch the other
six.

⚠️ **Read `docs/evidence-phase2-steering-attempt4-20260828.json` first.** This
document exists because of one specific finding in it, and the repair is only
honest if you can see what it is responding to.

## 1. Why a repair is owed at all

Attempt 4 scored **6 of 7**. The single failure was criterion 2:

> *2. Signed heading error and absolute cross-track both trend toward zero.*

`|cross_track|` grew monotonically **0.0089 → 0.0371 → 0.0706 → 0.0715 m** and
never turned over, so criterion 2 failed and **the FAIL stands as recorded**.

🔑 **The two predeclared parameters were never simultaneously satisfiable, and
that is provable without reference to the outcome.** Cross-track is the *integral*
of heading error: its rate is `v·sin(heading_error)`, so `|cross_track|` **cannot**
decrease until heading error crosses zero. In attempt 4 heading error crossed zero
at **~1.00 m** of travel — and `max_distance_m` was **1.00 m**. A run bounded at
the null can never exhibit the integral unwinding. **Criterion 2 and
`max_distance_m: 1.00` contradicted each other the moment both were written down.**

The attempt-4 record independently corroborates that this was a budget artefact
rather than a control failure, though **that corroboration is not the argument** —
the structural point above stands on its own:

| decision | \|cross-track\| | increment | heading error |
| --- | --- | --- | --- |
| d1 | 0.0371 m | +0.0281 m | +9.213° |
| d2 | 0.0706 m | +0.0335 m | +12.225° |
| d3 | 0.0715 m | **+0.0010 m** | +5.429° |
| d4 (stop) | — | — | **−1.883°** |

The increment collapsed 34.4 mm → 1.0 mm, and at the stop the error had crossed
zero, so `v·sin(err)` had already reversed from **+0.0237** to **−0.0082 m/s**.

🚨 **This is the same mistake, in the same shape, for the third time.** Attempt 2
set `duration_ms: 2000` to bound exposure without checking that acquisition had to
fit inside it. Attempt 3's design then wrote *"`max_distance_m` stays 1.00 and
remains the real bound"* — correct about exposure, never checked against what the
run had to demonstrate. **The recurring error is bounding exposure without
checking the bound against the criterion the run exists to test.** Section 6 adds
a standing check so it does not happen a fourth time.

## 2. The repair to criterion 2

Criterion 2 bundles a first-order quantity with its own integral. They have
different time constants and are not jointly testable in a window sized to null
the first one. It is **split**, not weakened:

* **2a — signed heading error trends toward zero.**
  *(Attempt 4 satisfies this: 9.213 → 12.225 → 5.429 → −1.883.)*
* **2b — `|cross_track|` reaches a maximum and then DECREASES across at least two
  consecutive decisions.**
  *(Attempt 4 cannot test this. It is new, and it is the point of attempt 5.)*

⚠️ **Both must hold. 2b is a stricter test than the original text, not a looser
one** — "trend toward zero" was never quantified; "decreases across two
consecutive decisions" is.

🔑 **This repair is written by someone who already knows attempt 4's cross-track
was about to turn over. That is exactly the conflict of interest the
predeclaration rule exists to contain, so 2b is deliberately constructed to be
FALSIFIABLE:** cross-track must actually turn over **and** decrease twice in a
row. If the loop hunts about the route instead of settling — which section 4 says
is a live possibility — 2b fails, and so does criterion 3. **A longer window is
not a free pass.**

**Criteria 1, 3, 4, 5, 6 and 7 are UNCHANGED**, verbatim from
`docs/phase2-steering-run1-predeclared-20260827.md`:

1. No intermediate stop before the final/abort stop.
3. No oscillation between saturated ±angular commands.
4. Cross-track never exceeds 0.20 m.
5. The 0.30 m hard abort never fires.
6. Motion duty cycle at least 80%.
7. Final stop confirmed and the motion gate disarmed afterwards.

**Any failure is a FAIL.** A run that aborts safely is a successful *guard* and a
failed *criterion set*; both statements go in the record.

## 3. Configuration, and where the distance budget comes from

The budget is derived from time constants **measured in attempt 4**, not from its
outcome:

```
acquisition consumed (d0 -> d1, measured)          0.321 m
heading error crossed zero at (measured)          ~1.00 m
distance per ~1 Hz decision (measured, mean of 3)  0.264 m

minimum for 2b = 1.00 + 2 x 0.264                = 1.53 m
with one spare decision                          = 1.79 m
```

| Parameter | Value | Change from attempt 4 | Why |
| --- | --- | --- | --- |
| `confirm_steering_validation_run` | **true** | — | Steering is opt-in per call |
| `max_abs_angular_speed` | **120** | — | Lowest MEASURED arc value; do not scale below the measured 120–180 band |
| `linear_speed` | 400 | — | Frozen, measured |
| `duration_ms` | **8000** | was 6000 | Schema maximum; at the measured 0.2235 m/s effective speed this caps travel at ~1.79 m — the same place the distance bound lands |
| `max_distance_m` | **1.75** | was 1.00 | 0.22 m clear of the 1.53 m minimum, ~1 spare decision |
| `max_cross_track_m` | **0.30** | — | UNCHANGED — it is the hard abort and it stays |
| `motion_refresh_interval_ms` | 200 | — | Unchanged |
| Route offset | **8°** | — | Unchanged |

⚠️ **Exposure grows and this is stated as such, not buried.** Commanded travel goes
1.00 → 1.75 m; worst-case reach from the start becomes
`1.75 + 0.50 stop overshoot + 0.30 cross-track = 2.55 m`. The 0.30 m hard abort and
the corridor-breach override are untouched, so the *containment* mechanisms are
unchanged — the *distance over which they must hold* is what grew.

## 4. Prediction registered before dispatch: attempt 5 may fail on criterion 3

Attempt 4 measured, for the first time, the rotation response **while moving**:

| commanded angular | measured rate |
| --- | --- |
| −111 | −0.61 °/s |
| −120 | +6.48 °/s |
| −65 | **+7.39 °/s** |

🚨 **The −65 command produced a HIGHER rate than −120.** That is roughly **1 s of
actuation lag**, not a gain curve — each interval still carries the previous
command. **n = 3 and lag-contaminated: do not fit a turn constant to these three
numbers, and do not use them to retune the gain.**

With ~1 s of lag, ~7 °/s of authority and proportional-only control, attempt 4
**already overshot**: +5.429° → −1.883° in a single decision. Expected hunt
half-period ≈ `1.0 + 5.429/7.0 = 1.78 s`, full period ≈ **3.6 s**. A 1.75 m run
lasts ~7.8 s ≈ **two full cycles**.

🔑 **So the honest prediction is: attempt 5 has a real chance of failing criterion
3 (oscillation) instead of criterion 2.** Registering that now is what makes a
criterion-3 pass meaningful and a criterion-3 failure interpretable.
⚠️ **If it does oscillate, the answer is NOT to add a derivative term mid-programme
and re-run.** Record the failure, then design the damping deliberately with its own
predeclaration.

## 5. Corridor — 7.0 × 7.0 m, and why the attempt-4 one no longer suffices

`max_distance_m: 1.75` requires **2.55 m** of boundary clearance at the start. The
attempt-4 5.0 × 5.0 m corridor gives **2.50 m** — **0.05 m short.** It cannot be
reused.

| corridor | centre clearance | placement tolerance | verdict |
| --- | --- | --- | --- |
| 5.0 × 5.0 m | 2.50 m | **−0.05 m** | **too small** |
| 6.0 × 6.0 m | 3.00 m | +0.45 m | works, tight |
| **7.0 × 7.0 m** | **3.50 m** | **+0.95 m** | **use this** |

**Corridor, centred on (5.98, −5.24):**

```json
[{"x": 2.48, "y": -8.74}, {"x": 9.48, "y": -8.74},
 {"x": 9.48, "y": -1.74}, {"x": 2.48, "y": -1.74}]
```

Verified offline against the live `export_map`: `polygon_is_valid` true, all four
vertices inside "Backyard Right" and outside both keep-outs, no corridor edge
crosses any area or keep-out edge. Real yard clearance at the centre is
**5.965 m** against a corner reach of **4.950 m**, fitting with **1.015 m** to
spare.

🔑 **It stays SQUARE deliberately.** The route heading is derived at dispatch from
the mower's measured heading plus the 8° offset, so a narrow oriented corridor
bets on a heading not yet known — the bet that left attempt 3 with 0.16 m of
margin.

⚠️ **A bigger corridor is a bigger geometric backstop, not weaker containment.**
The binding bounds remain `max_distance_m` and the 0.30 m cross-track abort; the
corridor only has to contain them. A 7 × 7 m box inside a region with 5.97 m of
real clearance is still well within the mowable area.

## 6. Preconditions, and the standing check that stops the recurring mistake

* Mower `AREA_INSIDE` with **RTK Fix**, BLE live, blades off, off the dock.
* Placed within **0.95 m** of **(5.98, −5.24)**. *(Attempt 4 was placed to 0.113 m
  via the card, so this is comfortable.)*
* 🔑 **`route_start` MUST be read from the live position AFTER placement.**
  `_CONTINUOUS_MAX_START_DRIFT_M` is **0.30 m**, and that gate reads
  `"passed": dry_run or (drift <= 0.30)` — **it passes unconditionally in a dry
  run**, so a green dry run proves nothing about placement.
* A dry run must return **all gates passing** with `blockers: []` and
  `required_radius_m 1.34` against a `boundary_clearance_m` of at least 2.55 m.
* Explicit per-run operator authorization. The gate is armed only for the run and
  **disarmed and verified from the live API AND RAW `core.config_entries`
  afterwards.**
  ⚠️ **A RAW read taken immediately after a disarm can still show the old value** —
  HA writes `.storage` lazily; attempt 4 read `[True]` for ~15 s after the live API
  read false. Re-read before concluding a disarm failed.
* ⚠️ **The card arms the motion gate.** If the mower is repositioned with the card,
  the gate will be left `enabled: true` with `blockers: []`. Check it.

🛑 **STANDING CHECK — apply before every future Phase 2 dispatch.** For each
exposure bound (`duration_ms`, `max_distance_m`, `max_heading_acquisition_s`),
write down the distance or time the run needs in order to *demonstrate* each
criterion, and confirm the bound exceeds it. Three attempts have now been spent on
runs whose exposure bound made their own criterion unreachable. **A bound that is
safe but makes the test impossible is a wasted run, not a conservative one.**

## 7. What a pass authorizes

A pass authorizes **one repeat run for repeatability, at this same
configuration**. It does **not** authorize Phase 3 waypoint A/B, longer windows,
higher angular authority, a gain change, a derivative term, or removing the
per-call opt-in. Each of those needs its own predeclared criteria.
