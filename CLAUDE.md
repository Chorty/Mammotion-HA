# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Start here

Read, in order, and stop at the provenance line:

1. **"Current build"** directly below — what is released and installed.
2. **"Standing decisions"** — the operator's scope calls. They override any
   older recommendation in this file.
3. `docs/NEXT-SESSION.md` §0 for live mower state (its first ~197 lines; the
   2,700 below are history under their own banner).

⚠️ **Everything from "Build provenance" down is history**, including entries
that read as open items. It is accurate as a *record* — the measurements stand,
and several entries usefully record why an approach was rejected — but **do not
act on any build state, open item, or "next step" it describes.** Measurement
has since refuted several of its claims, among them the "unexplained" turn-rate
variance and the turn-budget framing.

**Verify before you act on any claim in this file.** On 2026-08-14 a paragraph
here described an already-shipped fix as "NOT implemented" for thirteen betas
and cost a session real work. `scripts/check_doc_symbols.py` (a pre-commit hook)
now fails when these docs name code that does not exist — but it checks *names*,
not whether the prose around them is still true. One grep against the tree beats
this file every time.

## Current build: `0.6.4-beta72` released and installed motion-disabled

🏁🏁 **PHASE 1b PASSED — VERDICT `go`, 2026-08-23.** One supervised, authorized
8000 ms arc (`linear 400, angular 180` — the ORIGINAL plan's command, only the
duration changed) run against the protocol registered the same day before any
capture existed. **All 18 arc criteria passed**: 7 informative moving steps
(need >= 3), max mirror error **4.264°** (limit 10° — more than half the budget
unused), 39/39 refresh writes, distance guard fired correctly at 1.5568 m,
containment held. The paired straight capture is the unchanged 2026-08-22 file
and still passes all 17. Read `docs/phase1b-go-20260823.md`.

🔒 **Scope, verbatim from the analyzer's own output:** `"Phase 1 telemetry
feasibility only; never authorizes Phase 2 or motion."` This means the
telemetry can support a closed-loop attempt — it does NOT mean such a
controller is safe, accurate, or worth building. **What starts now is Phase 2
DESIGN DISCUSSION, not implementation and not another physical run.**
🔑 Two things measured this week still bound that design regardless of this
`go`: the ~1 Hz position/heading bundle
(`docs/the-1hz-bundle-is-the-ceiling-20260822.md`), and the unexplained
alpha = -0.149 ± 0.043 pairing residual (`docs/codex-adjudication-20260823.md`,
~7 mm/step cost, unmodelled). ⚠️ The prediction-error criterion is still NOT
implemented; this `go` rests on the repaired mirror criterion alone.

🆕 **beta72 CLOSES THE GUARD'S FAIL-OPEN PATHS AND MAKES THE ARC SCOREABLE,
2026-08-23.** Both are prerequisites for the Phase 1b arc.
**(1)** A frozen feed, a missing position, or a dead sampler each now **trip**
the travel guard instead of silently returning the run to the wall clock;
`_PROBE_TRAVEL_GUARD_OVERSHOOT_M` is **0.50 m** (was 0.35, which measurement
showed would be exceeded ~42% of the time); and `corridor_must_cover_m` is the
**worst case**, not the nominal one. 🔑 Verified live: the identical dry run
reports **2.24 m** where beta71 said **1.85 m** — 39 cm more corridor for the
same command.
**(2)** Phase 1 duration is now **per control** — straight 4000 ms, arc
**8000 ms** — never a menu, pinned by two tests.
⚠️ **The banked 2026-08-22 arc is therefore INADMISSIBLE**, not merely failing;
a Phase 1b `go` needs a NEW arc. Phase 1's `no_go` stands on its own terms.
🚨 **Not browser-verified.** Record: `docs/deploy-runbook-p0.md` → beta72.

## (history) beta71 released and installed motion-disabled

🔧 **THE MIRROR CRITERION IS REPAIRED AND STILL SAYS `no_go`, 2026-08-23.** Per
the review: **repaired, not replaced.** `toward` now pairs with the **START** of
the interval (the only heading a controller has when predicting the chord it is
about to travel), and `MIN_MOVING_STEP_M` went **0.01 → 0.15 m** because at the
measured σ = 0.0031 m a 0.076 m chord carries ±7.4° of bearing noise — a step
whose noise bound exceeds the threshold cannot test anything. **The 10° threshold
is untouched.**
🚨 **The repair does NOT flip the verdict, and the reason is the finding.** The
arc's error is now **2.521°** — comfortable — but the minimum chord leaves it
**2 informative steps against the 3 required**. ⚠️ **Do NOT fix it by lowering
the step count to 2** —
`test_the_banked_arc_now_fails_on_STEP_COUNT_not_on_error` pins that.
🗑️ **CORRECTED 2026-08-23: I wrote that a 4 s arc "cannot test its own
criterion". FALSE.** Four of five banked captures produced 3 informative chords
inside their first 4 seconds; the arc180 alone missed, its fourth arrival never
landing before the cutoff. **Fragile, not impossible** — and that refutation
removed the main argument for substituting the 8 s arc.
🔑 **DECIDED 2026-08-23 by independent Codex adjudication: the `angular 120` 8 s
arc is NOT admitted**, though it would pass at 2.385°. Both pre-registered
dimensions would have moved at once with the outcome already known. It stays
exploratory evidence. Registered instead:
`docs/phase1b-arc-protocol-20260823.md` — the **same `angular 180` at 8000 ms**,
every criterion unchanged, written before any capture exists. Also registered
there: a `3R = 600 ms` BLE-stall eligibility rule and, if prediction error is
ever adopted, a **0.085 m** threshold (0.150 tolerance − 0.065 sensing floor),
derived before scoring and **failing the banked 8 s run at 0.1418 m** without
being relaxed.
⚠️ **Prediction error was deliberately NOT added yet** — its threshold is
unresolved (0.10 m is breached at 0.1418 m by the banked 8 s run) and choosing
one now, knowing which passes, is the failure this repair avoids. Read
`docs/mirror-criterion-repaired-20260823.md`.

🚨 **AN INDEPENDENT REVIEW REJECTED THE CRITERION PROPOSAL AND FOUND FOUR FALSE
CLAIMS OF MINE, 2026-08-23.** All re-derived from the banked evidence and all
confirmed. Read `docs/corrections-to-the-20260822-analysis-20260823.md` before quoting anything from 2026-08-22.
🗑️ **The yaw "REFUTED" is WITHDRAWN** — an artifact of fitting `k_lin` excluding
spin-up and `k_ang` including it; same rule gives 11%, not 45%, and the data is
2.31σ from "no dependence" against 2.28σ from "proportional", i.e. it cannot
tell them apart. 🗑️ **"`toward` updates with position, zero exceptions" is FALSE**
— true for VIO, but `toward` **latches on straight motion**, updating once in 8 s.
🗑️ **"No minimum chord" is FALSE** — `MIN_MOVING_STEP_M` exists; it is 0.01 m,
which is far too small, and that was the real defect. ⚠️ **`alpha = −0.253 ± 0.174`
is SUPERSEDED** — on all 8 arc steps with the measured noise it is
**−0.165 ± 0.043**, which excludes **START too** at 3.1σ; no simple pairing is
right. ⚠️ **The proposed 0.10 m threshold is breached by my own unscored data** —
the 8 s run maxes at **0.1418 m**, 42% over, after an 810 ms refresh write, and
the budget does not close against the 0.065 m sensing floor.
🔑 **The pre-registration held** (verified from git) and is what made the yaw
error findable. ⚠️ **The verdict stays `no_go`** and the criterion revision is
REJECTED as written; the review's recommendation is to **repair** the mirror
criterion and **add** prediction error rather than substitute it, because
substituting loosens the certified heading bound from 10° to ~23°, which would
saturate the Phase 0 steering law's ±180 clamp.

🏁 **THE PREDICTION MODEL HOLDS OUT OF SAMPLE, AND ITS YAW TERM DOES NOT,
2026-08-22.** An `angular 120` arc scored against constants **committed before it
ran** (`docs/frozen-prediction-constants-20260822.json`, `c6f16f07`). Excluding
spin-up: **6 steps, median 0.0175 m, max 0.0628 m** against a proposed 0.10 m —
passes with 37% margin. 🔑 The multi-run frozen `k_lin` (0.2482 m/s, 16 steps over
three runs) beat a single-run refit (0.2148) **3x on data neither saw**.
🗑️ **`w = k_ang × angular_speed` is REFUTED**: predicted 4.787 °/s at angular 120,
observed **6.94 °/s**. A 33% cut in commanded angular moved the yaw rate 3%
(7.181 → 6.94 °/s) — rotation is very nearly **independent** of `angular_speed`
across 120–180. ⚠️ Two points in a narrow band; `angular 500` in an arc is still
unmeasured. 🔑 **Position prediction passed anyway because it re-anchors on
MEASURED heading each interval**, so a 2 °/s rate error buys only ~0.009 m — a
continuous controller needs accurate heading **feedback**, not an accurate yaw
**model**, which takes yaw calibration off the Phase 2 critical path.
🔑 **The guard fired a second time** (1.6093 m sampled → **1.8074 m** actual
against the published 1.85), so overshoot is 0.276 / 0.307 m on two runs — still
**do not** tighten the 0.35 m constant. Read
`docs/prediction-model-holds-out-of-sample-20260823.md`.

🏁 **THE 8 s WINDOW SUSTAINS AND THE DISTANCE GUARD FIRES, 2026-08-22.** Two
separately authorized supervised runs on beta71. **(1) Speed does not decay:**
8000.7 ms, 1.8952 m, 8 position arrivals → **7 steady steps** (against 3 in a 4 s
window), steady mean **0.2479 m/s**, trend **+0.0073 m/s per step** — drifting
*up*. BLE held too: **39 refresh writes, 39 completions**, median 123.8 ms. Open-loop
bearing held 241.3–243.5° against a frozen 243.5°, ~**5 cm cross-track over
1.9 m**. So the 4.88x fluidity case no longer extrapolates from 4 s.
**(2) The guard fired on hardware:** at a 1.5 m bound it tripped at 6771.6 ms /
1.5731 m sampled, `aborted_early: true`, 33 of 40 refreshes, stop confirmed —
**actual travel 1.776 m against the published `corridor_must_cover_m` of 1.850**,
so the 0.35 m overshoot estimate is conservative by 7 cm and honest.
⚠️ **Do NOT tighten `_PROBE_TRAVEL_GUARD_OVERSHOOT_M` on one sample** — it is a
safety margin and being wrong is asymmetric.
⚠️ **0.2757 m/s is the TOP of the range, not the planning number**: three runs
span 0.237–0.276 m/s (~15%) while the within-run trend is flat, so size on
~0.25 m/s and call the end-to-end gain **~4.5x**, not 4.88x.
⚠️ **An arc corridor fits only SHORT from where the mower now sits** — at a
3.5 m length the arc leg fails area margin (0.84 / 0.01 m against 1.2 m), but at
**2.5 m it passes at 1.2987 m**. No repositioning needed; a shorter corridor is.
⚠️ The freezer sizes arc lateral drift from the `angular 180` radius, so it
UNDER-estimates a tighter arc — go gentler than 180, not tighter. Read
`docs/continuous-window-sustains-and-the-guard-fires-20260822.md`.

🆕 **beta71 BOUNDS THE MOTION WINDOW BY DISTANCE, NOT BY A TIME PROXY,
2026-08-22.** The probe's `duration_ms` was capped at 4000 ms because "the only
thing limiting travel is the window" — a time proxy that capped the longest
continuous run at ~1.1 m, while the 4.88x case extrapolates a 4 s window to a
159 s route. The in-window sampler is now also the guard: it measures
displacement from the window start and aborts once `max_travel_m` is exceeded,
which can only shorten a drive. **Fails closed** — the ceiling moves to 12000 ms
but a window over 4000 ms is REFUSED without BOTH `max_travel_m` and in-window
sampling (`duration_over_4000ms_requires_max_travel_m`,
`duration_over_4000ms_requires_in_window_sampling`). ⚠️ The guard trips **late by
~0.35 m** (~1 Hz cache plus one refresh interval at 0.2757 m/s), so the response
reports `expected_overshoot_m` and `corridor_must_cover_m`; **size a corridor for
the guard doing nothing**, since it is the thing under test. Verified executing
on the host by zero-motion dry runs including both refusals. Moves no
`LUBA_ACCEPTANCE_PROFILE` key. 🚨 **Not browser-verified**; no physical long
window has been run. Record: `docs/deploy-runbook-p0.md` → beta71.

## (history) beta70 released and installed motion-disabled

🏁 **PHASE 1 CAPTURES ARE DONE AND THE VERDICT IS `no_go`, 2026-08-22.** Both
separately authorized 4 s windows ran on beta70, moved the mower, stopped on
command, stayed inside their prevalidated corridors, and left the gate verified
disarmed. The analyzer failed **one** criterion of 17:
`shallow_arc.bearing_toward_compass_mirror` at **12.631°** against a 10°
threshold. **The `no_go` stands; it was not re-run and the threshold was not
moved.**

🔑 **The failure is the criterion's pairing, not `toward`.** The check compares a
chord bearing — an interval average between position fixes ~1 s apart — against a
single `toward` sample. On the arc the mower rotates ~10° per interval, so the
result swings by that whole rotation depending on which end you pair with:
**2.5° / 2.3°** using the interval's START against **12.6° / 11.3°** using its
END. VIO independently agrees `toward` tracked the rotation (VIO −9.92/−9.13°
against `toward` +10.11/+9.00°, the known mirror sign flip). The one remaining
row is a **0.076 m** chord against a 2–4 cm position-noise floor, so it is noise.
⚠️ **This does NOT make it a `go`.** An ill-posed criterion gets fixed
deliberately in the plan, before a re-run — never by editing a threshold after
seeing the data that failed it. Read
`docs/phase1-continuous-motion-captures-20260822.md`.

🔑 **CONFIRMED OFFLINE THE SAME DAY: the shipped END pairing is the ONLY one that
fails.** Re-scored against both banked captures, worst error is START **7.930°**
/ MIDPOINT **8.854°** / END **12.631°**, and `err@end = err@start + rotation`
holds to 0.001° on every row. Filtered to informative steps the arc scores
**2.521°** under START — the same regime as the straight control's 1.236°. A
**second, independent defect**: the criterion has no minimum chord, so it scores
steps whose position-noise bearing bound is **±12.2°** and **±7.4°**, at or above
the 10° threshold — those rows cannot test anything. ⚠️ Which pairing is
physically right is now **MEASURED, not chosen**: writing
`err(alpha) = err_at_start + alpha × rotation` (exact to 0.001° on every row) and
solving gives, over all 8 informative arc steps at the corrected mirror constant
90.13, **alpha = −0.149 ± 0.043** — START 3.50σ, MIDPOINT 15.26σ, END 27.02σ, so
**no simple pairing is right** and END is merely worst.
🐛 The earlier −0.253 and −0.165 came from `scripts/reanalyze_mirror_pairing.py` using
mirror constant **90.00**; `dalpha/dK = 0.1214/°`. Fixed.
🔑 **VIO gives the same sign independently (−0.175 ± 0.043)**, which rules out
every `toward`-only mechanism. Survivors: position lag relative to heading
(~0.65 s), or `toward` being a **body heading** (item 15). ⚠️ No offline test
separates them — no per-field timestamps, and both arcs turn the same way.
⚠️ Practical cost is **~7 mm per 1 Hz step**, so this is interesting and
operationally minor. Do not write a mechanism down as established. Read
`docs/codex-adjudication-20260823.md`. The fix must still not be "pick whichever passes". Read
`docs/phase1-mirror-criterion-is-ill-posed-20260822.md`;
`scripts/reanalyze_mirror_pairing.py` re-derives it with no mower.

✅ **What the straight capture DID establish**, all 17 criteria passed: position
fixes arrive at ~1 Hz **during** motion (4 arrivals, max gap **1023 ms** against
a 2000 ms limit), mirror error **max 2.008°**, 1.1029 m travelled, 19 refresh
writes all completing in order, full containment. The ~1 Hz feed does not
degrade while the mower drives.

🏁 **CONTINUOUS MOTION IS WORTH 4.88x, AND IT IS ALL DEAD TIME, 2026-08-22.**
In-pulse speed over 212 banked linear-400 pulses is **0.2584 m/s**; a continuous
window sustains **0.2757 m/s**; the 9.0 m collinear Route B chain managed
**0.0565 m/s** end to end (9.0 m in 159.3 s wall clock). Effective duty cycle
**21.9%**; a 9 m route would go 159 s → ~33 s.
🗑️ **This refutes the guess that short pulses never reach full speed** — a
**500 ms** pulse already medians 0.2422 m/s, indistinguishable from continuous.
The apparent in-window ramp (0.243 → 0.266 → 0.298 m/s) is the ~1 Hz feed's
reporting lag unwinding, not the drivetrain; total window travel matches the
pulse corpus, which is the check that settles it. **The entire gain is not
stopping.** ⚠️ 4.88x is a **ceiling, not a forecast** — a real controller must
still steer on ~1 Hz feedback — and it extrapolates a **4 s** window to a 159 s
route, which nothing has demonstrated. 🔑 An untested cheaper lever: linear 400
is ~47% of the app's ±850 scale and the vendor drives ~0.55 m/s, so commanding
faster multiplies with the duty-cycle win — at the cost of more blind distance
per correction. Read `docs/what-continuous-motion-is-worth-20260822.md`.

🔑 **POSITION AND HEADING ARE ONE ~1 Hz BUNDLE — THAT IS THE FEEDBACK CEILING,
2026-08-22.** Across both Phase 1 captures, `position x/y`, `toward` and VIO
heading change on **exactly the same instants, zero exceptions**. VIO is not an
independent faster channel. ⚠️ Report stamps run at ~2 Hz and are **not**
feedback — only every other frame carries new `sys.toapp_report_data`, so
counting stamps doubles the apparent rate. Consequences: a continuous controller
can observe at ~1 Hz whatever it consumes, which at the measured 0.28 m/s is a
correction every ~0.28 m against a 0.15 m tolerance — **feed-forward-dominated
by necessity, not a tight tracking loop**. 🗑️ **"More measurements per run" is
dead**: `duration_ms` is schema-capped at 4000 ms, so every run yields ~4
observations and ~3 steps regardless of criterion. 🗑️ **Driving slower is
worse** — sample count is set by time, not distance, so shorter chords just add
noise. 🔑 For the pairing question the lever is **rotation per interval, not more
intervals** (uncertainty = bearing noise / rotation). Read
`docs/the-1hz-bundle-is-the-ceiling-20260822.md`;
`scripts/measure_telemetry_bundling.py` re-derives it with no mower.

🧪 **CONTINUOUS MOTION PHASE 1 INSTRUMENTATION IS DEPLOYED.** The existing
bounded raw probe now records 100 ms coordinator-cache samples inside refreshed
motion windows, including x/y, `toward`, VIO, report timestamps, active command,
and refresh completions. It remains opt-in and fails closed; there is no new
continuous executor. Deployed straight and shallow-arc dry runs both planned 41
samples and returned `would_send: false`, `command_result.attempted: false`.
Browser beta70 and the final disabled gate were verified. No mower command was
sent. See `docs/evidence-beta70-continuous-phase1-deploy-20260821.json`.

✅ **THE PHASE 1 ANALYZER IS BANKED OFFLINE.**
`scripts/analyze_phase1_capture.py` evaluates the required straight and
shallow-arc response files plus fresh frozen-corridor metadata without importing
Home Assistant or exposing network, BLE, gate, service, or dispatch access. It
recomputes the written timing, compass-mirror, turn, stop, refresh, and
containment criteria; records SHA-256 input hashes; and fails closed on missing
evidence. A `go` is telemetry-feasibility evidence only and never authorizes
Phase 2 or mower motion. Commit `d105f4ca`; usage:
`docs/phase1-capture-analyzer.md`.

🌙 **LATEST OPERATOR STATE:** the beta70 physical Phase 1 captures have not
been run. The mower battery died and the operator put it on the charger at
night. No telemetry readback verified charging or changed the last deployed
gate evidence. Do only offline work until daylight, an operator is present, the
emergency stop is accessible, and each individual 4 s run is explicitly
authorized after a fresh contained-route scan.

✅ **SEGMENT-LEVEL KEEP-OUT CONTAINMENT IS DEPLOYED AND VERIFIED.**
`_keep_out_leg_violations` checks legal-endpoint legs against every keep-out
edge, `_validate_custom_path` refuses with `path_legs_cross_keep_out_zone`, and
the card blocks the same path locally. Boundary touches and collinear overlap
count as violations. Live zero-motion validation against the real map refused
the crossing `(9.0, -0.76) → (15.0, -0.76)` solely by the new reason and
accepted the legal control `(9.0, -5.0) → (15.0, -5.0)`. Browser verification
passed: beta69 footer/console, red dashed crossing, named refusal, Real Go
disabled. Full deployment record: `docs/deploy-runbook-p0.md` → beta69.

✅ **THE DISARM AUTOMATION IS INSTALLED, 2026-08-22.** The gate had been found
armed at rest four times and the automation had been deferred four times.
`automation.mammotion_disarm_motion_gate_when_left_armed` is now live on the
host (id `1755900000001`, state `on`, never yet triggered), appended to
`/config/automations.yaml` from `docs/automations/disarm-motion-gate.yaml` and
loaded by `automation.reload` — no HA restart. It fires 15 minutes after
`binary_sensor.back_yard_clip_skywalker_real_motion_ready` reads on, plus a
23:00 sweep, and — added 2026-08-22 — a third `armed_but_blocked` trigger. It
notifies only when it actually closed something.

🚨 **FIFTH ARMED-AT-REST OCCURRENCE, 2026-08-22.** The gate was found
`enabled: true` with the sole blocker `position_not_valid_for_motion` — armed,
and held shut only by the mower being on the dock. **That blocker evaporates the
moment the mower is moved off the dock**, so the gate would have gone live with
an empty blocker list without anyone arming it. The first two triggers could not
see it: `real_motion_ready` is `off` whenever *any* blocker fires, so an armed
gate behind the dock is invisible to them. The gate's `enabled` flag is a
config-entry option with no entity, so the new trigger keys off the readiness
sensor's `blockers` attribute instead — `experimental_motion_disabled` is
present exactly when the gate is closed, so its **absence** means armed.
Disarmed on the operator's instruction and verified
(`enabled: false`, two blockers).
⚠️ **It is one-way and cannot interrupt a run** — there is no arm service, and
`disarm_experimental_motion` refuses while a session is active. Backup of the
pre-change host file: `/config/automations.yaml.bak.claude-20260821-disarm`.
Record: `docs/deploy-runbook-p0.md` → "disarm automation installed".

✅ **ROUTE B 3 x 3.0 m COMPLETED ON BETA69.** A later supervised, explicitly
authorized run completed all three collinear 3.0000 m legs with
`target_reached` landings at **0.14388 / 0.11413 / 0.06070 m** (mean 0.10624
m). The frozen 9.0000 m route ran from `(4.8756, -2.4530)` to
`(11.7193, -8.2980)` with zero blockers. The gate was then verified DISARMED,
with no active session and `MODE_PAUSE`. Current 3.0 m evidence is **5 reached /
1 failed, n=6**: feasibility is proven, reliability is not.
`docs/evidence-route-b-3x3m-beta69-20260821T193417Z.json`.

✅ **CONTINUOUS MOTION PHASE 0 IS OFFLINE ONLY.** A pure lookahead controller
and standalone JSON replay now return bounded steering or fail-closed zero-speed
decisions without importing Home Assistant, registering a service, or exposing
a dispatch path. No continuous executor exists and no mower moved for this
phase. Phase 1 instrumentation is now deployed, but its two separately
authorized physical captures remain pending; see
`docs/continuous-motion-feasibility-plan-20260821.md`. Analyze the resulting
files with `scripts/analyze_phase1_capture.py` before any Phase 2 design.

## (history) beta68 released and installed motion-disabled

🚨 **2026-08-21 — A LEG CAN BE DRAWN STRAIGHT THROUGH A KEEP-OUT ZONE, AND
NEITHER THE CARD NOR THE BACKEND REFUSES IT.** Found in a browser, not by a
test: click two legal points either side of an obstacle zone and the path is
drawn — and would be driven — through it. Containment is **PER-POINT** on both
sides, so both endpoints being outside is enough to pass. beta68's
`_legsCrossingKeepOuts` tests the leg against every zone edge, paints a crossing
leg red/dashed, and says in the banner that neither will refuse it.
⚠️ **It WARNS, it does not block** — the backend still dispatches such a path,
so a card refusal would be stricter than the machine that drives, and the
operator's standing decision is that being wrongly blocked is the worse failure.
**Segment-level containment in the BACKEND is the real fix and is still open.**

✅ **beta67 was fully verified in a browser (2026-08-21)** — version, two dashed
red `obstacle` zones rendering, a click inside one refused, and the leg-length
advisory reading *"Longest leg is 3.11 m, over the 0.58 m the controller can
protect… can miss by up to 0.80 m. This is a warning, not a blocker."*
3.11 × sin 15° = 0.805, so the bound is right end to end on real geometry.

🔑 **THE ONE-SIDED AIM DRIFT IS REAL — three independent runs.** A 2.27 m card
segment reached target at **0.11378 m** while running **8 of 8 pulses negative,
mean −10.87°**, growing −7.7° → −19.8° as range closed — the same signature as
the chain run's failed sub-leg 2 (−10.29°, 9 of 9). It is **not noise**: it is
consistent within a run and inflates as bearing-to-target rotates. That run
succeeded *because the leg was shorter*, which is direct support for the bound
below. `docs/evidence-vector-segment-2p27m-20260820.json`.

🆕 **2026-08-20 — THE CARD NOW DRAWS KEEP-OUTS AND REFUSES A CLICK INSIDE ONE.**
`export_map.keep_out_polygons` has been available since beta63 and the card
referenced it **zero** times, so an obstacle click looked exactly like a legal
one — how a 10.8 m run drove into a trampoline. Zones render as dashed red
polygons **after** areas (both are filled and SVG paints in document order, so
drawing them first hides a keep-out inside its containing area), and
`_onMapClick` refuses by kind. The banner also warns when the longest planned
sub-leg exceeds the **0.58 m** the controller can protect, naming the miss an
uncorrected sub-floor aim error buys. ⚠️ The advisory **never blocks** — 3.0 m
reached target at 0.094 m. ⚠️ The keep-out test is **PER-POINT** exactly like the
backend's, pinned by a test; a leg clipping a corner is caught by neither.
🚨 **No browser has rendered this yet** — bytes verified on the host, behaviour
not. See `docs/deploy-runbook-p0.md` → beta67 for the four-point browser check.

🏁 **2026-08-20 — THE CONTROLLER CANNOT PROTECT A LEG LONGER THAN 0.58 m, AND
NOTHING COMPUTED THAT UNTIL NOW.** A mid-drive correction fires only once aim
error reaches `_MIN_CORRECTABLE_AIM_ERROR_DEGREES` (= post-turn tolerance 10 +
deadband 5 = **15°**), so an error just under the floor is never corrected
whatever it costs, and it buys `distance × sin(floor)`. Setting that equal to
`waypoint_tolerance` gives `limit = tolerance / sin(floor)` = **0.580 m** on the
accepted profile. At 3.0 m the same floor permits an uncorrectable **0.776 m**
miss — 5× tolerance. **That is why ~0.8 m is the measured-good regime.**
beta57 fixed the re-aim *trigger* from an angle to a projected distance; the
*floor* is still an angle, so the controller now correctly SEES the miss coming
and declines to act because the angle looks small.
`_correctable_leg_length_limit_m` and three `split.*` fields expose it, **verified
executing on the host** by a zero-motion dry run.
⚠️ **ADVISORY, not a refusal, and deliberately pessimistic** — a 3.0 m sub-leg
reached target at 0.094 m the same day; a test pins that case so nobody hardens
this into a gate. 🚨 **Do NOT respond to a breach by lowering the floor**: it is
set by the turn primitive's actuation limit, protecting a 3.0 m leg would need a
~2.9° floor, and the affine sweep bound still permits 20° at the 200 ms floor.
Moves no `LUBA_ACCEPTANCE_PROFILE` key; profile still ACCEPTED.

🔑 **ROUTE B AT 3.0 m: 3 of 3 now completed.** The earlier chain's sub-leg 1
`target_reached` at **0.094 m**;
sub-leg 2 failed `vio_realign_incomplete` at 0.2594 m when a **51.025°**
correction came due at 0.26 m to run and was refused `turn_budget_infeasible`.
the new beta69 chain added three successful landings, making the combined record
**5 reached / 1 failed, n=6**. It can complete, but is not yet proven reliable.
`docs/evidence-routeb-3m-chain-20260820.json` and
`docs/evidence-route-b-3x3m-beta69-20260821T193417Z.json`.

🗑️ **A BLE prediction of mine was REFUTED by that run and is recorded as such.**
Cadence did NOT degrade across the chain — sub-leg 1 mean 0.85, sub-leg 2 mean
**0.89**, zero stalled pulses in 19 across 130 s. The time-into-run cadence trend
(`docs/evidence-ble-cadence-predictors-20260821.json`) is a **population tendency
across 24 runs, not a within-run law**, and must not be used to attribute an
individual failure.

🔑 **`ble_rssi` DOES NOT PREDICT BLE CADENCE** — within-run median r = **+0.042**
over 24 runs (14 pos / 10 neg). Pooled it reads −0.245, which is a
between-session confound. A marginal RSSI is **not** a reason to postpone a run,
nor to trust one.

🔑 **THE POSITION FEED RUNS AT ~1 Hz, MOVING OR NOT**, so the settle loop's 1.0 s
poll is already matched to it and **polling faster buys nothing**. The ~2.85 s
settle is therefore near the **floor** of stop-measure-go, not slack. Motion runs
at a **29% duty cycle** (4.55 s cycle, 1.30 s of it moving; 57% of a run is
position-settle). ⚠️ The vendor drives **continuously at ~0.55 m/s on this same
~1 Hz feed**, so 1 Hz does not block a continuous controller — it is what makes
stop-measure-go expensive. And next position IS predictable from last fix +
commanded velocity to **0.029 m median / 0.097 m p90** when refresh cadence holds
(180 of 262 pulses), which is ~5× better than tolerance.
`docs/evidence-position-predictability-20260821.json`.

⚠️ **Open items with the check that verifies each: `docs/open-items-20260821.md`.**

## (history) beta63 — keep-out zones are now checked

🏁 **2026-08-20 — KEEP-OUT ZONES ARE NOW CHECKED.** Containment tested inclusion
in a mowing area and never exclusion from a keep-out, so a supervised 10.8 m run
stayed inside "Backyard Right" the whole way and drove into an obstacle zone
containing a trampoline. The geometry was never missing: `HashList` keeps it in
sibling dicts beside `map.area` (`obstacle`, `no_go_zone`, `virtual_wall`,
`no_go_zone_variant`, `visual_obstacle_zone`), already in map-local x/y.
`_keep_out_polygons` / `_keep_out_violations` now read all five,
`_validate_custom_path` refuses with `path_points_inside_keep_out_zone`, and
`export_map` exposes `keep_out_polygons`. **Verified on the host: the exact
recorded click is refused pre-dispatch**, naming split point 2 and obstacle hash
`1529607395159402290` — and the position where the mower actually stopped tests
inside that polygon. `docs/evidence-beta63-keepout-refusal-20260820.json`.
⚠️ **Still PER-POINT** — a leg clipping a corner with neither endpoint inside is
not caught. Segment-level containment is the real fix.

## (history) beta62 — ⚠️ GATE WAS ARMED AT DEPLOY

🆕 **2026-08-20 — beta62 ADDS DELIBERATE SAFETY-GATE OVERRIDE TOGGLES.** One
toggle per firing blocker (28 registered gates), each rendering the reason the
gate exists. Off by default, **reset after every run**, never persisted, and
echoed into the run record with `original_passed: false` so an overridden run
can never look like a clean one. A typo is refused by schema, not ignored. Four
gates are absent as *incoherent* to override, not vetoed —
`stop_primitive_available` (a `hasattr` check; an override does not create the
stop method), `turn_mode_valid`, and the two `operator_confirmed_*` gates, which
ARE the operator's deliberate act. Read `docs/deploy-runbook-p0.md` → beta62.

🚨 **beta62 WAS DEPLOYED WITH THE MOTION GATE ARMED** (`blockers: []`, mower off
the dock), on the operator's explicit instruction after the deploy was paused
and the state reported. **Fourth armed-at-rest occurrence.** `enabled` is STILL
TRUE. The disarm automation remains uninstalled. No motion was commanded.
*(Superseded 2026-08-22 — the automation is now installed; see "Current build".
Kept because it records the posture at the time of the beta62 deploy.)*

🆕 **2026-08-19 — ROUTE B IS DEPLOYED (`0.6.4-beta61`), MOTION-DISABLED.**
**No motion has run on it.** Gate verified `real_motion_allowed: false`, no
session. Deploy record and exact hashes: `docs/deploy-runbook-p0.md` → beta61.
A host dry run proves the splitter executes: a 50 ft click became 4 sub-legs of
3.810000 m with all three junctions at `estimated_commands_needed: 0` and
`estimated_translation_m: 0.0` (`docs/evidence-beta61-50ft-dryrun-20260819.json`).
🔑 **A 50 ft straight click DOES fit** — longest contained chord is 20.52 m,
measured from the live polygons; the 12.74 × 9.73 m figure was recorded
*positions*, not area extent. One click on a distant
point now auto-splits into collinear sub-legs of at most **3.85 m**, so a 50 ft
click becomes 4 legs of 3.8100 m with every junction at 0.000000°. It moves **no
`LUBA_ACCEPTANCE_PROFILE` key and owes no Gate 5**;
`scripts/check_accepted_profile.py` still reports ACCEPTED. Read
`docs/route-b-collinear-split-20260819.md`. ⚠️ Splitting does **not** improve
accuracy — cross-track error has unity gain across a collinear junction — and
15.40 m has never been driven.

⚠️ **Route A (`beta60`) is measured INERT.** Replayed across 32 decision points
on three hardware runs, the old and new re-aim triggers made identical decisions
every time. It was correct, tested and Gate 5-accepted, and it changed nothing.
Do not fund more work on that trigger.

## (history) Current build: `0.6.4-beta59` released and installed motion-disabled

⚠️ **The heading below this one said `beta55` until 2026-08-17** — stale since
the beta56 release on 2026-08-16, and it made a session ask which build was
real. All four version sites and tag `v0.6.4-beta56` agree; beta56 was a
backend-only deploy adding the read-only `ota_info_probe` and changed nothing in
the motion path, so every motion claim written under "beta55" below still
describes what is running.

🏁 **beta57 SHIPS THE REACH WORK AND GATE 5 PASSED ON IT, 2026-08-18** — four
card-driven segments, 4/4 `target_reached`, mean 0.1073 m, profile identity
proven key-by-key with `max_linear_pulse_ceiling: 22` dispatched.
`docs/gate5-beta57-PASSED-20260818.md`. The profile is **accepted again** and
`scripts/check_accepted_profile.py` now reports so.
⚠️ **Acceptance is not validation:** replay shows old and new triggers made
identical decisions at all eight decision points, and the ceiling never bound
(3 pulses of 22). Both halves of the reach work only bite on legs of ~1.9 m and
up, so **the control-law change is still untested on hardware** and *"is
`vio_max_realignments: 3` enough"* remains unanswered — no mid-drive correction
has ever fired on the new code.

*(Superseded, kept for the record:)* **beta57 UN-ACCEPTED THE PROFILE.** PR #15 merged
(`5d9aa759`), released and deployed motion-disabled 2026-08-18; deploy record and
exact hashes in `docs/deploy-runbook-p0.md` → "beta57".
`LUBA_ACCEPTANCE_PROFILE.max_linear_pulse_ceiling` moved **14 → 22** to reach a
20 ft leg, which owes the §4 re-pinning in `docs/gate4-repass-20260805.md` and
**another Gate 5**. Also changed: a pre-dispatch `segment_too_long` cap at
**6.10 m** (the daylight path had no length gate at all), a
`linear_budget_insufficient_for_segment` gate, the mid-drive re-aim trigger
(angle → projected miss), and the mid-drive correction tolerance (18° → 10°).
⚠️ `vio_max_realignments` **stays at 3** — a raise was attempted, reviewed twice,
and reverted; see the note further down. **Read
`docs/reach-20ft-and-the-reaim-trigger-20260817.md` before acting on any of it.**
Full CI suite green (687 pytest, 46 frontend, all nine hooks). **The host runs
beta57 and NO MOTION HAS RUN ON THE NEW CONTROL LAW.** Both new gates are
verified executing live by a zero-motion dry run.

🔑 **The open question "is `vio_max_realignments: 3` enough" is now ANSWERED ON
REPLAY, not on hardware**: 0 of 62 recorded segments exceed 3, maximum is 3
(`scripts/replay_reaim_trigger.py`, §3a of the reach doc). ⚠️ The corpus has **no
leg over 4 m** against a 6.10 m cap, so the regime the change exists for still
has no data. And a leg following a junction turn has an effective mid-drive
budget of **2, not 3** — the post-turn gate spends the same counter.

⚠️ Two entries below are now qualified by that work — the "~0.8 m leg" operating
rule and the "raising `vio_max_realignments` is the WRONG fix" warning. Both are
marked in place. Neither is deleted: they were correct for the control law as it
stood, and the second is still correct without the divergence detector.

⚠️ *The paragraph below is the beta55 record and its head/tag line is stale —
`main` is now at `a2351048` with tag `v0.6.4-beta56`. Kept because everything it
says about the motion path is still what is running.*

`main`, `origin/main`, and tag `v0.6.4-beta55` agreed at `5ef37511`. Beta55
releases the reviewed and merged PR #14 on top of beta54's guarded **Night
dry-run** and **Night Go** card controls, still without changing any value in
`LUBA_ACCEPTANCE_PROFILE`; Real Go remains the accepted VIO path. The beta55
motion-disabled installation is complete and **no motion was commanded by it**.
The motion gate is verified **DISARMED** (`real_motion_allowed: false`, no
active session, no last session). The mower is **docked** at
`(4.3764, 3.1923)`, `CHARGE_ON`, `zone_hash 0`, so `position_not_valid_for_motion`
is the expected second blocker. Deploy record and exact hashes:
`docs/deploy-runbook-p0.md` → "beta55".

PR #14 was independently reviewed and **merged** (`efa1eda8`), then released as
beta55. It carries a night-only fix: use the settled post-pulse RTK position for
the residual target bearing, allowing the existing reverse-recovery refusal to
stop after an overshoot. The card and harness also send `sample_delays: [0, 3]`;
beta54 omitted it and inherited the backend's `[0, 5, 10, 20, 30, 45, 60]`,
making the physical run take about 6.5 minutes. It also carries a Real Go
throughput fix: VIO calibration now waits one four-second feedback window
instead of 2+4 seconds, settled linear telemetry is reused instead of adding
another three-second sample wait, and the card does one final runtime reload
instead of two. Each VIO pulse then runs the existing bounded BLE queue-settle
check and refuses the next command with `ble_link_not_ready_after_feedback`
rather than dispatching into a backlog. The dispatched Real Go payload, stops,
safety gates, legacy/night timing, frozen acceptance-profile values, and all
`vio_active` construction sites remain unchanged. The VIO response now records
`post_settle_feedback`, `post_feedback_queue_settle`, and one settled-position
sample per successful pulse.

The review verified this independently rather than trusting green CI: the
`LUBA_ACCEPTANCE_PROFILE` literal and the legacy turn branch are each
byte-identical across beta54 and the PR tip (SHA-256 `0a0ab014…d858ea` and
`7665c302…8cef2fdbc`); the nine original `turn_mode == "vio"` blocks all survive
and the four new sites are feedback handling only; the queue-settle refusal at
the `ble_link_not_ready_after_feedback` refusal precedes both the mid-drive re-aim dispatch and the next
linear dispatch, so no dispatch can follow a non-live queue. Replayed against
the recorded beta54 geometry, the night fix refuses pulse 3 (residual bearing
155.636° vs movement heading 0.423°) and lands 0.0827 m out instead of 0.1171 m
— ⚠️ still outside the 0.08 m tolerance, so it stops on
`target_requires_reverse_recovery`, **not** `target_reached`. Verification rerun
after a README correction: **668 pytest (50% coverage), 46 frontend**, ruff
check, ruff format (58 files), mypy (28 files), and all nine pre-commit hooks
green, modifying nothing.

The first supervised Real Go throughput run safely refused its second linear
command because the position-feedback report requests still occupied the BLE
queue. A bounded post-feedback queue-settle correction was then deployed. A
fresh supervised 0.70 m Real Go run reached its target in 19.2 s with 0.093100 m
landing error; all three queue-settle records were live at depth zero and every
movement/stop succeeded. Read `docs/real-go-throughput-hardware-20260814.md`.
⚠️ **That is one path, not a reliability population**, and **no night run has
ever exercised the night fix on hardware.**

The segment executor's legacy branch
(the `else` arm of the segment executor's turn branch)
omits `motion_refresh_interval_ms` (primitive default
`0`) and passes `angular_speed_fast/slow` at the schema default **180**, which
does not break static friction on a stationary pivot (~3°/pulse). Every
converging night turn used **angular 500 with refresh**, by calling the primitive
**directly**. **The standalone service and the segment's legacy branch are not the
same code path.** Found by three independent verifiers, confirmed by hand.
⚠️ `legacy` keeps both defects in the v1 plan (containment, not a fix), so the
card's **Nudge still turns single-shot in a deadband** — do not let that drop.

🏁 **All five gates complete, Gate 5 passed twice** (2026-08-08 fixed-budget,
2026-08-12 reach-enabled), and **the branch reached BETA on 2026-08-12** — all
three exit criteria met, assessed in `docs/p0-beta-release.md` → "Alpha to
Beta". ⚠️ Criterion 2 ("BLE holds a full path run") turned on an interpretation
that is written down there rather than assumed: it asks whether a run can finish
before the link dies, and 9 runs have completed every planned segment. The
residual is ~7% of runs aborting on BLE, which is **a hint, not a measurement**
(95% CI 0.2–31.9%) — do not fund BLE work on it.

🚨 **A harness bug left the motion gate OPEN once on 2026-08-11 (fixed,
`c196b8b1`).** `scripts/beta32_validation_run.py` set its `armed` flag *after*
the post-enable readback, so an enable that succeeded while `real_motion_allowed`
came back false — BLE dropping between preflight and arm — returned early
claiming it had aborted "without sending anything" and never disarmed. **Any
script that can open the gate must treat "I called enable" as what obliges the
disarm, never "enable succeeded".** Same commit makes the backend's own
`blockers` list a hard preflight check: all eight entity-derived checks passed
while the gate already knew the BLE client was gone.

✅ **The gate is DISARMED and was verified disarmed after every run.** The
2026-08-10 ARMED-at-rest posture ended with that session; normal posture is
disarmed, opened only for the ~100 s of a supervised run.

🔑 **THE ACCURACY WALL IS SOLVED, AND IT IS THE TURN'S OWN TRANSLATION.** Read
`docs/turn-translation-explains-the-landing-wall-20260810.md`. A VIO turn does not
pivot in place — it displaced the mower 0.028–0.131 m on the 2026-08-10 runs, and
sideways displacement at the start of a 0.6–0.7 m leg rotates the bearing to the
target by `atan(translation/leg)`. The turn primitive closes on **VIO body
heading**, so it cannot see this: the heading did not change, the target's bearing
moved. Across all five completed segments, map-frame aim error minus VIO-frame
error equals `atan(translation/leg)` to within **0.02–1.25°**.

**Consequence: `heading_tolerance_degrees` is the WRONG LEVER and lowering it
18 → 11 would have changed none of those five segments.** It governs the VIO-frame
error (mean 5.5°, already fine); the landing is set by the map-frame error (mean
8.0°). The lever that works is **not a profile key** — the post-turn gate is
`min(heading_tolerance_degrees, vio_realign_threshold_degrees)` = `min(18, 15)` =
15, and every map-frame error fell inside it. Lowering the backend default
`vio_realign_threshold_degrees` 15 → ~5 catches all five and moves no frozen key.

⚠️ **QUALIFIED 2026-08-17 — the rule below was correct for the control law as it
stood, but the mechanism was misattributed to distance.** The re-aim trigger was
an ANGLE (`aim > 18°`) while the objective is a DISTANCE
(`range × sin(aim)`), so corrections never fired in the far field: 17° with 14 m
to run is a 4.09 m miss that fired nothing. ~0.8 m is where an angle-triggered
controller happens to work. The trigger is now the projected miss and the cap is
a pre-dispatch gate at 6.10 m — **untested on hardware**, so keep planning legs
at ~0.8 m until a supervised run says otherwise. Read
`docs/reach-20ft-and-the-reaim-trigger-20260817.md`. Everything below still
stands as measurement.

🔑 **OPERATING RULE: plan legs at ~0.8 m. Reach is not landing accuracy.**
Measured 2026-08-15, first four-segment card run on beta55
(`docs/evidence-real-go-card-beta55-20260815T204747Z.json`). Segment 1 (1.17 m,
no turn needed) reached target at 0.141665 m — 3.3 mm inside tolerance. Segment
2 (1.65 m, after a 48.6° junction turn) **diverged and stopped safely** on
`vio_realign_budget_exhausted` 0.251406 m out.

The mechanism is leg length × the aim error a turn leaves behind, not leg length
alone. The post-turn gate is allowed to succeed at up to
`_POST_TURN_ALIGNMENT_TOLERANCE_DEGREES` = 10°, and the landing fit predicted
the miss almost exactly at that residual:
`0.62 × 1.65 × sin(9.676°) + 0.065 = 0.237 m` against 0.2514 m measured.
Inverting the fit at a 0.15 m tolerance and a 10° residual gives **L ≤ 0.79 m** —
which is the Gate 5 re-pass configuration: 0.8 m legs, 4/4 `target_reached`,
mean **0.0780 m**, the best result on record. So ~0.8 m is "run the validated
configuration", not a threshold to trust on faith.

⚠️ **Do not read the 4 m reach result as a licence to plan a 4 m leg.** Reach
was measured on *straight single segments starting aligned*; it says a segment
can travel that far stopping on tolerance, not that it lands accurately after a
junction turn. The two are different properties and only reach is measured at
that length.

⚠️ The fit **under-predicts the no-turn case by 2×** (segment 1: predicted
0.070 m, measured 0.1417 m) because aim error develops mid-leg on straight runs.
Treat it as sound for the post-turn residual case only.

✅ **REINFORCED 2026-08-17 — this warning was tested and it won.** A branch tried
raising the default 3 → 10, guarded by a "divergence detector" meant to make it
safe. Two review rounds found the detector wrong **twice, for two different
reasons**: v1 compared before-vs-after within one correction and so measured the
correction turn's own translation (`atan(0.10/0.75)` = 7.6° against a 1.0°
margin); v2 compared successive pre-correction errors and so measured the
geometric inflation of aim error as range closes (`atan(c/d)` grows as `d`
shrinks), which happens on a **perfectly healthy leg**. Both would have aborted
good runs. **The budget stayed at 3.** Five of six second-round findings existed
only because the budget had been raised. If you think you have a way to make a
bigger budget safe, assume it is wrong until hardware says otherwise — and note
that a leg exhausting the budget stops safely, which is a measurement.
`docs/reach-20ft-and-the-reaim-trigger-20260817.md`.

⚠️ **Raising `vio_max_realignments` (default 3, shared by the post-turn gate at
the `_POST_TURN_ALIGNMENT_TOLERANCE_DEGREES` gate and `_mid_drive_realign_decision`) is the WRONG fix.** The
loop was diverging — aim errors grew 16.96 → 21.22 → 24.975° while each
correction "succeeded" against an 18° turn tolerance, leaving 9.7 / 11.5 / 13.6°
of residual against a bearing rotating −3.2 / −9.7 / **−15.4°** per pulse and
accelerating as range closed. More budget buys more corrections chasing a target
moving away faster, each adding turn translation. beta17 already recorded this
exact failure.

🔑 **`toward` IS ACCURATE — ~0.8° median, measured over 169 pulses across 30
evidence files, 2026-08-18.** `movement_heading + toward` = **89.819°** with 98.2%
of pulses inside 3°, corroborating the recorded 90.13° mirror from a far larger
sample. Accuracy improves with travel length exactly as a fixed position-noise
floor predicts (median 0.88° at 0.20-0.35 m, 0.76° above 0.35 m). **Exactly one
outlier in 169** — and it is the already-documented item-15 case, a forward pulse
immediately after a *night* turn. Read `docs/toward-accuracy-measured-20260818.md`.

⚠️ **This corrects a framing, not a fact.** `toward` is not too imprecise to steer
by; at ~1° it is comparable to VIO. What blocks closing a loop on it is *timing
and granularity* — it stays bit-identical through a bounded pulse and arrives as
one post-hoc step, and the refreshed night turn quantum is 48.15° ± 5.70 with
nothing scaling it. ⚠️ Every sample is validated against travel, so this says
nothing about `toward` as a **body heading after a rotation**; that is item 15,
still open, and the sole outlier.

**Settled, so do not re-derive:**

- Rotation is **not predictable from duration** better than ~40% at p90 — ten
  pulses at matched ~200 ms windows spread 5.44–15.20°, 2.79×, with duration,
  cadence and direction held constant. The estimate can only improve a landing;
  the bound carries the safety.
- The "directional turn asymmetry" is **refuted** (three runs: 8/8, 1/1, 1/6;
  pooled over 33 samples the directions differ by 0.5%).
- A **90° junction dispatches and completes** — measured, 3 of 4 commands.
- A single **180° turn is refused pre-dispatch**; the largest that dispatches is
  ~114°. Chain junctions instead (`--reposition`).
- Per-**click** reach is 4 segments (`REAL_CLICK_TO_GO_SEGMENT_LIMIT`).
- Per-**segment** reach depends on the execution mode, and there is **no hard
  distance cap on the VIO path** — it is emergent from the pulse budget:
  - *fixed budget* (`max_linear_pulse_ceiling: null`) — `max_linear_commands` is
    schema-capped at **3**, so ~1.0–1.3 m, then `max_linear_commands_reached`;
  - *loop-to-tolerance* (the accepted profile sends **14**) — **4 m measured** in
    11 pulses at 0.1023 m, stopping on tolerance, the ceiling never binding.
    Cumulative travel is separately capped at `segment_length ×
    linear_distance_ceiling_factor` (default 2.0).
  ⚠️ The old line here read "per-segment reach is ~1 m; a 2.0 m leg is not
  dispatchable." That was the pre-ceiling number and was left standing under
  "do not re-derive" through the entire reach programme. Corrected 2026-08-15.
- **Night is hard-capped at 1.0 m** (`_NIGHT_MAX_SEGMENT_LENGTH_M`), refused
  pre-dispatch by `night_segment_too_long`, and night also refuses
  loop-to-tolerance (`night_linear_loop_unsupported`). That 1.0 is chosen, not
  measured.
- `turning_mode` (`zero_turn` / `multipoint`, `nav_sys_param_cmd` ID 6) is a
  MOWING-turnaround planner setting. Click-to-path turns bypass it entirely by
  sending raw `DrvMotionCtrl` velocities. Untested but expected irrelevant.


## Standing decisions — 2026-08-14

These are the operator's, not derivable from the code. They override any older
recommendation below.

1. **Audience: this yard only.** A bespoke tool for this LUBA. Per-mower
   constants are fine as they are. `p0-beta-release.md`'s "non-LUBA hardware
   characterised" release criterion is **moot** — do not treat it as a blocker
   and do not propose upstream-shaped work (auto-derivation, per-device
   calibration flows) as required.
2. **Night is contained exploration and is PARKED.** Not a work queue. The
   beta55 night fix stays deployed and unexercised on purpose. Do not propose
   night runs, a night landing population, or resolving item 15's mirror
   disagreement as next work.
3. **Accuracy is closed.** Achieved is ~0.089 m mean (n = 16 landings across
   four multi-segment runs, all inside the 0.15 m tolerance). The
   `0.62 × leg·sin(initial_aim) + 0.065` fit's **0.065 m intercept is a sensing
   floor** — 2–4 cm position noise plus ~1031 ms feed staleness — not a tuning
   target. The report-rate hypothesis is refuted and the stop-lead item dropped;
   do not reopen either. The one known failure class was the re-aim guard, fixed
   in beta42.
   🔑 **If the floor is ever reconsidered, the missing instrument now exists
   upstream.** The beta23 probe could not measure position-report cadence because
   `last_report_at` stamps *every* LubaMsg, and it named the fix: stamp arrivals
   per report type. Upstream pymammotion `c21ec18` ("modifications to report
   cfg") implements exactly that as `last_report_data_at`, set only on frames
   carrying `sys.toapp_report_data`. It is **not** in our pinned
   `chorty-0.8.12.post1`; it would arrive with a backend bump, which also needs
   our BLE-leak fix rebased (upstream `47c3c54` touches the same
   `transport/ble.py`). This refutes nothing and reopens nothing — it only means
   the tooling no longer has to be built. Read-only survey, 2026-08-15.
4. **The goal is consistency, not precision** — click-to-go reliable enough to
   trust without watching.

⚠️ **Documentation is the known weak point, not capability.** On 2026-08-14 an
audit of the three session-entry docs found a paragraph calling an
already-shipped fix "NOT implemented" (13 betas stale) and a constant removed 18
betas earlier still framed as a live constraint.
`scripts/check_doc_symbols.py` now fails a commit when these docs cite code that
does not exist, but **it only checks names**. A green run does not mean the prose
is true.

## OTA firmware capture — PAUSED 2026-08-16

Separate thread from the turn/accuracy work above: an attempt to capture a
readable copy of the mower's OTA firmware for research. **Paused at the
operator's request** to wait a day or two before the next real attempt — not
because anything failed unsafely. Full writeup, findings, and a resume
checklist: `docs/ota-firmware-capture-investigation-20260816.md`.

**Do not act on this as open work without reading that doc first.** Short
version: the firmware itself was never captured and the remaining wall is
cryptographic (the mower's own Aliyun device credentials, which no known
software-only method can obtain) — no amount of better timing changes that.
One permanent, real capability *did* come out of it: `ota_info_probe`, a
read-only HA service (deployed `0.6.4-beta56`) that queries the mower's own
OTA status over BLE — first time this request/response path has ever been
exercised, confirmed working live, but it structurally cannot carry a
download URL (see the doc for why). ⚠️ **UniFi Hardware Acceleration is
currently OFF on the gateway** (a deliberate, real router-performance
tradeoff made to get passive capture visibility) and the UniFi block-sta API
is confirmed broken (`api.err.Invalid`) — check both before trusting either.


## Build provenance — accurate as record, NOT as build state

⚠️ **Everything below this line is history.** It is kept because the
measurements stand and because several entries record why an approach was
rejected — but **do not act on any build state, open item, or "next step" it
describes**. Relocated here 2026-08-14 from a "Current build" section that had
grown to 393 lines and was hiding stale claims among live ones.

One beta54 card-driven 0.739138 m Night Go stopped safely on
`no_target_progress` at 0.117085 m after three turn and three forward commands.
The second forward pulse had settled only 0.002661 m outside the configured
0.08 m tolerance, but the controller reused a pre-pulse target bearing and sent
an unnecessary third pulse after crossing the target. Read
`docs/night-go-card-beta54-20260814.md`.

✅ **§7 item 17 is complete.** One backward-only pulse moved 0.418536 m on map
bearing 96.433921° while `toward` remained bit-identical at 173.1023°. The
mirror-derived body heading was 277.0277°, whose reverse is 97.0277°: only
0.593779° from measured travel. `toward` is therefore body heading, not
course-over-ground, under reverse. RapidState `fuse_status` stayed 0 `NO_POSE`
in all 81 records and was non-informative on this manual path. Read
`docs/night-reverse-heading-20260813.md`. It does not resolve item 15's separate
forward-course disagreement.

✅ **§7 item 18 is complete as characterization, not acceptance.** One 0.70 m
perpendicular night segment reached the 8° turn tolerance in one pulse, sent
three forward pulses, and stopped on `no_target_progress` at 0.114277 m from
target. Its per-pulse mirror observations were 92.0720 / 90.7417 / 89.1569°,
so item 15's 14.3069° result is not stable across all night forward pulses.
Read `docs/night-segment-item18-20260814.md`. No night landing tolerance or
accuracy population is established.

✅ **§7 item 15 is complete.** One night-branch pulse at angular −500, 1,500 ms,
and 200 ms refresh changed `toward` by −54.2208° with 0.07459 m translation.
The following forward pulse travelled 0.43648 m on a bearing 81.416° away from
the target direction; the night re-aim guard stopped further motion with
`night_reaim_required_but_unavailable`. Read
`docs/night-segment-turn-quantum-20260813.md`. This measured disagreement
refutes treating the 90.13° mirror as established segment-control truth.
§7 item 16 is complete: across 73 concurrent
runtime samples, `toward` stayed bit-identical throughout the 1.551 s refreshed
pulse and arrived as one post-pulse +36.3064° step; no intermediate heading was
observed. Read `docs/night-toward-latency-20260813.md`. The next hardware
discriminator was item 17, now complete as described above. Item 18 is also
complete as characterization; the separate item-15 mismatch remains unresolved.

🔎 **Same-day read-only autonomous-mow comparison:** 291 samples over 179.895 s
captured three complete vendor pivots. Unlike the bounded manual pulse,
`toward` streamed progressive headings during continuous vendor motion. Forty
usable moving steps gave `travel bearing + toward = 90.57°` with 2.02° circular
SD. Read `docs/autonomous-mow-observation-20260813.md`. This narrows item 16's
stepwise result to the manual pulse/report cadence; it does not settle reverse
or `fuse_status`.

**beta48/49 were card-only** — run-record downloads, a per-segment landing table
(leg / landing / tolerance / verdict / pulses / mean), a readiness banner that
names every blocker code *and* explains it, grouped toolbar, collapsed
diagnostics. No `LUBA_ACCEPTANCE_PROFILE` key touched. ⚠️ beta49 fixed four
defects that only appeared when the card was rendered against **real**
`export_runtime_state` output — duplicate blocker codes from two overlapping
backend lists, two emitted codes with no help text, a restored run presented as
current, and a tofu-risk glyph. **Render against live state, not fixtures.**

> **Historical pre-night-v1 finding (superseded by beta50+):** A night segment
> did not work through the legacy branch because of two missing parameters,
> not because of the turn primitive. The dedicated `turn_mode: "night"` branch
> now supplies both and beta54 exposes it through Night Go. The legacy branch
> remains deliberately unchanged.

🏁 **CLOSED-LOOP TURNS WORK IN THE DARK WITH NO VIO — 5 of 5, 2026-08-12/13.**
Read `docs/night-closed-loop-turn-works-20260812.md` then
`docs/night-turns-converge-but-the-quantum-is-coarse-20260813.md`. The legacy
primitive (`raw_pymammotion_turn_to_heading`, closes on `toward`, no `vio_active`
gate) reached `target_heading_reached` on **five armed night turns**, both
directions, at tolerances 18 and 8, all at `tracked_features: 0`. Single-variable
against the same run without refresh: 4 commands and 29° of 90° becomes 2
commands and 82° — beta47 wraps that pulse in `_motion_refresh_window`.
**Daylight-only was never a property of the machine**; it was the turn
primitive's heading source plus a missing refresh.

🚨 **But four of those five converged on LUCK, not control.** The refreshed pulse
quantum is **48.15° ± 5.70** (n = 10) and *nothing scales it*, so the terminal
error lands wherever the last pulse falls inside the tolerance band — margins of
1.72 / 1.09 / 0.36°. Run c had 10.89° of error remaining and the smallest
available action rotated **54.11°**, a 5× over-correction it then spent a command
undoing. Tightening tolerance 18 → 8 barely moved absolute error (8.92° → 7.28°).
🔑 **The fix is in our own data:** an *un-refreshed* single shot rotates
**7.24° ± 1.60**, and both points sit on `rotation ≈ 32.2 °/s·t − 2.4°`. So port
the VIO path's `_turn_final_approach_pulse_ms` **window scaling** into the legacy
turn — scale the window, ⚠️ **not** the speed (`angular_speed_slow: 180` sits in
the stationary deadband; the slow tier has never actually engaged).
⚠️ A turn is not a segment; `vio_active` still refuses `turn_mode: "vio"`; and
🚨 **the shared legacy map→`toward` conversion is wrong by construction**
(mirror, not the additive 102.4). Night v1 contains this by applying the mirror
strictly inside `turn_mode: "night"`; the two cancelling legacy conversion sites
remain deliberately unchanged.


🚨 **`toward` TRACKS IN-PLACE ROTATION, 2026-08-12 — the night premise is
REFUTED.** Read `docs/toward-tracks-in-place-rotation-20260812.md`. Two pivots in
full darkness with VIO at zero: `angular +500` moved `toward` **+99.55°** with
**3.8 cm** of travel; `angular −500` moved it **−61.43°** with 3.0 cm. 99.55°
cannot come from 3.8 cm when the position noise floor is 2–4 cm. **This kills the
premise that `toward` is blind to in-place rotation**, which is why every turn is
forced onto VIO and why closed-loop motion is daylight-only — so the night path
may be a legacy-style turn rather than an arc controller. Corroborated by a 184 s
sample of the vendor **mowing in the dark**: straight rows and in-place pivots,
which also 🗑️ **refutes my inference that the vendor arcs**. ⚠️ n = 2, no closed
loop has run on `toward`, and its latency DURING rotation is unmeasured — that
last one decides whether a loop can close at all.

🔑 **ARCS WORK, 2026-08-12 — measured and linear.** Read
`docs/arcs-work-20260812.md`. Two armed pulses back to back, one variable apart:
`linear 400 + angular 180` travelled 0.5823 m and rotated its course **+22.20°**;
`linear 400 + angular 0` travelled 0.5840 m and rotated **+0.00°**. 1.8 mm apart
in distance. Implied arc radius **1.512 m**. **`toward` tracked the rotation
exactly**, which is the mechanism a night controller needs — translation keeps
course-over-ground live, and a live `toward` closes a heading loop with no VIO.
⚠️ **"Angular needs 500" is a STATIONARY-only finding** — 180 actuated fine in an
arc; the 2026-07-25 A/B measured a pure in-place turn. ⚠️ Still open-loop, still
daylight-only, and the radius-vs-angular curve is unmeasured.

🏁 **GATE 5 RE-PASSED ON THE REACH-ENABLED PROFILE, 2026-08-12.** Card-driven,
four segments, all `target_reached`, landings 0.0674 / 0.1032 / 0.0807 / 0.0607
against 0.15 — **mean 0.0780, the best four-segment result on record.** Zero
reverse-recovery, zero budget exhaustion. The real payload carried
`max_linear_pulse_ceiling: 14`, so **the reach profile is hardware-accepted and
profile identity is proven in fact.** Read
`docs/gate5-repass-PASSED-20260812.md`; evidence
`docs/evidence-gate5-repass-2-20260812.json`.

⚠️ **Do not overstate it.** beta43 (post-turn correction budget 2 → 4) was **not
exercised** — the only correction was −10.477°, inside the old 21.50° envelope,
and the first attempt's 29.647° refusal did not recur because segment 3's
geometry differed. The **ceiling never bound** either: 0.8 m legs use 2–3 pulses
of 14, so reach is evidenced by `docs/loop-to-tolerance-reach-20260811.md`, not
by this gate. That was deliberate — long legs are 5/7 on control-law grounds,
short legs 28/28.

🏁 **REACH IS SOLVED, 2026-08-11: 4 m on a single segment, measured.** Read
`docs/loop-to-tolerance-reach-20260811.md`. With `max_linear_pulse_ceiling` set,
a 2.0 m leg landed **0.0690 m** in 5 pulses, a 3.0 m leg **0.0928 m** in 8, and a
4.0 m leg **0.1023 m** in 11 — all stopping on **tolerance, not on the ceiling**,
which never bound on any run. The counterfactual is each segment's own third row:
on the accepted profile they sit 0.7489 / 0.6777 / 1.7919 / **2.9543 m** short on
`max_linear_commands_reached`. Per-click reach goes ~4 m → **~16 m** at 4
segments. **4 m is a demonstrated floor, not a limit** — where it breaks is
unknown.

~~⚠️ `max_linear_pulse_ceiling` is a frozen key the card sends as `null`, so
NEITHER RUN IS ON THE ACCEPTED PROFILE.~~ **SUPERSEDED 2026-08-12.** True when
the reach runs were measured; the key was adopted (`null` → 14) that day and
**accepted by the Gate 5 re-pass the same day**. The card now sends 14 and the
landings above are directly comparable to Gate 5. Kept because it records why
the reach runs were deliberately measured off-profile first.

🔑 **The loop is robust to BLE stalls, and that reframes the standing BLE item.**
The 3 m leg drove through two 2-write pulses (4158 ms and 2847 ms windows) that
travelled 0.2325 / 0.2016 m against 0.34–0.49 m for the 15 cadence-intact pulses,
and still landed at 9.3 cm — it just took two more pulses. Against
`max_linear_commands: 3` those stalls are fatal. BLE latency still degrades the
rate estimate, but it **is no longer a blocker for reach**. (n = 2 stalled
pulses: the shape of the effect, not a calibrated number.)

🏁 **FIXED IN beta42 — this was described as open for thirteen betas.** The 2 m
run's second segment failed on **cross-track, not reach**: the beta38 re-aim
guard suppressed a correction at a projected 0.1469 m miss against 0.150 m
tolerance — 3.1 mm of margin — and landed 0.1797 m out. The guard projected the
miss at the **closest approach**, but the mower drives a whole pulse and
finished 0.0877 m past it; in quadrature that predicts 0.1711 m, which exceeds
tolerance and would have fired the correction.

`_projected_landing_after_next_pulse` (commit `7e1d5afd`, beta42) implements
exactly that quadrature term and is pinned by seven tests in
`tests/components/mammotion/test_reaim_guard_next_pulse.py`, including the
0.1797 m case. ⚠️ **The paragraph that used to sit here said "NOT implemented"
and was never updated when beta42 shipped**, which on 2026-08-14 caused a
session to propose redoing finished work. It is the reason
`scripts/check_doc_symbols.py` exists — and the reason that check is not
sufficient, because this claim named no symbol at all.

Note what the fix buys: the 0.0212 → 0.0147 m figure is the guard's *prediction*
error, not landing error. It removes a failure class; it does not shave the mean.

- **beta41** — a segment's **opening turn decomposes instead of refusing**
  (`_vio_turn_to_heading_staged`). It tries the direct turn first and, ONLY on a
  `turn_budget_infeasible` refusal, splits the rotation into stages of ≤60°. Each
  turn call gets its own command budget and displacement allowance, which is why
  chained 60° junctions accumulate 180° where a single 180° turn is refused.
  Wired into the **opening turn only** — mid-drive re-aim and post-turn correction
  keep calling the primitive directly, since their rotations are small by
  construction. Translation is budgeted across the WHOLE staged turn, not per
  stage. If a 60° stage is also refused, the ORIGINAL `turn_budget_infeasible` is
  reported (`staging_cannot_help`), because slicing finer cannot fix a budget that
  dispatches nothing.
  🏁 **Validated 2026-08-10:** a **165.048°** opening turn completed in three
  stages, total staged displacement 0.1326 m of a 0.30 budget, and the beta40
  post-turn gate then corrected the +13.557° residual staging left. The two
  changes compose. Evidence:
  `docs/evidence-beta32-4segment-20260811T001250Z.json`.

- **beta40** — the post-turn alignment gate gets its **own** tolerance,
  `_POST_TURN_ALIGNMENT_TOLERANCE_DEGREES = 10.0`, instead of borrowing
  `vio_realign_threshold_degrees` (the *mid-drive* trigger) through a `min()`. At
  the old `min(18, 15) = 15` the gate never fired. **10 is a floor, not a
  preference:** a correction fires only when the error exceeds the tolerance, so
  the worst sweep is `error + tolerance`, and the affine bound `40 °/s·t + 12°` at
  the 200 ms actuation floor still sweeps 20° — the guarantee needs
  `error + tolerance ≥ 20`, i.e. **tolerance ≥ 10**. Below that, corrections enter
  `sweep_exceeds_any_pulse` and the gate manufactures overshoot. Tightening
  further needs a shorter actuation floor or a tighter sweep bound, **not** a
  smaller number.
  🏁 **Validated 2026-08-10:** four segments reached target, landings
  0.0585 / 0.0867 / 0.1393 / 0.0979 m (**mean 0.0956**, best 4-segment result on
  record). The gate fired once, correcting −16.551° → −7.331°. The correction turn
  displaced 0.0108 m = **0.97° of induced error to buy 10.038°** — the
  "fix reproduces the problem" risk is real but ~10:1 in our favour.
  Evidence: `docs/evidence-beta32-4segment-20260810T205937Z.json`.

**Shipped 2026-08-09/10, in order, each on measurement:**

- **beta35** — refresh writes fire on a fixed cadence from the window start
  rather than one interval after the previous write completed. Delivered-window
  overruns fell from +117% to +29%. ⚠️ **The follow-on claim that "no run has
  aborted on BLE since" is REFUTED** — the 2026-08-12 4 m run died on a BLE
  queue deadline (`vio_realign_incomplete`). Measured over 20 armed
  multi-segment runs: 2 lost to BLE overall, 1 of 15 since beta35 (~7%). Both
  aborted the run rather than doing anything unsafe.
- **beta37** — the turn model rebuilt on 35 measured pulses.
  `_MIN_SCALED_TURN_PULSE_MS` 400 → **200** (200 ms actuates; there is no
  threshold near 400). The overshoot bound is no longer a rate: rotation measures
  `33.18 °/s·t + 4.63°`, which no single `C` can bound, so it is now the affine
  envelope `40 °/s·t + 12°`. `_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND`
  16.5 → **14.4**, which the first two changes finally made affordable.
- **beta38** — the mid-drive re-aim guard, corrected. beta36's version compared
  distance alone and suppressed two REAL corrections (40° and 78° aim errors,
  confirmed by RTK independently of VIO), ruining a segment. It now skips only
  when driving on would still land inside the disc:
  `distance·sin(aim) ≤ waypoint_tolerance`.
- **beta39** — the mid-drive re-aim guard follows `effective_linear_ceiling`, not
  `max_linear_commands`. Inert until loop-to-tolerance is enabled; it was the
  last prerequisite blocking it.

✅ **beta38's re-aim guard is VALIDATED on hardware, 2026-08-10.** Two armed runs,
four suppression events, **zero false suppressions** — every suppressed re-aim had
a `perpendicular_miss_m` genuinely under `waypoint_tolerance`, and a −47.812° aim
error was correctly *not* suppressed and corrected to −1.98°. The 60° run reached
target on **all four segments**. Evidence:
`docs/evidence-beta32-4segment-20260810T{185433,193833}Z.json`.

⚠️ **Do not act on the paragraph below as a plan — it is kept because its
measurements stand, but its diagnosis was superseded the same day.** The standing
fit is
`landing = 0.62 × leg·sin(initial_aim) + 0.065 m` (R² = 0.69, n = 12), which says a
0.9 m leg needs initial aim within **8.8°** while `heading_tolerance_degrees`
permits **18**. That geometric inconsistency between `heading_tolerance_degrees`
and `waypoint_tolerance` is real and both are `LUBA_ACCEPTANCE_PROFILE` keys.
**But it does not explain the observed landings.** Segments 2/3/4 of the 60° run
finished their turns at **−3.7 / −5.1 / +4.6°** — far inside any tolerance under
discussion — and still landed 0.1431 / 0.1447 / 0.1229 m, where the model predicts
0.089 / 0.102 / 0.097. It under-predicts by 26–54 mm. The aim error **develops
mid-leg** (−3.7° → +18.2° in one pulse; −5.1° → −34.7°; +4.6° → +26.0°), part
genuine heading drift and part bearing-to-target rotating as cross-track
accumulates; this run cannot separate them.

**So tightening `heading_tolerance_degrees` would have changed nothing here**, and
the shorter-legs alternative looks worse, not better: the beta33 reference run
drove **0.9 m** legs at the same 60° junctions for a mean landing of **0.098 m**,
while this run's **0.6–0.7 m** legs averaged **0.1312 m**. That comparison is
uncontrolled (six intervening betas, different day) — a warning flag, not a
refutation. **Do not spend a Gate 5 on either profile key until the mid-leg
divergence is characterised.**

## Gate history — all five gates complete

**Gate 2** passed 2026-08-03.

**Gate 4** failed on 2026-08-03 before its first linear command, was **re-passed
on 2026-08-05**, and **reproduced on a second daylight geometry on 2026-08-06**.
Read `docs/gate4-repass-20260805.md` before acting on this; the evidence is
`docs/evidence-gate4-beta20-day2j-*20260805*` and
`docs/evidence-gate4-beta21-second-geometry-summary-20260806.json`.

⚠️ **Neither Gate 4 pass tracked its path, and beta22 deliberately refuses the
behaviour that produced them.** Both runs passed by driving past the waypoint and
turning back — 2.28 m and 2.06 m of travel for a 1.04 m path, with 103.427° and
−112.325° recovery turns. beta22 treats a correction of 90° or more as a change
of motion contract and stops with `target_requires_reverse_recovery` rather than
dispatching a U-turn, so **a Gate 4 run on the current build is expected to fail
where beta20/beta21 passed.** That is containment, not regression.

🔑 **2026-08-10 — why those runs overshot is now known, and it was not bad luck.**
Landing error is set by the aim error at the *start* of the leg:
`landing = 0.62 × leg·sin(initial_aim) + 0.065 m` (R² = 0.69, n = 12). A 0.9 m leg
needs initial aim within **8.8°** to land inside 0.15 m, and
`heading_tolerance_degrees` permits **18**. The control law was always going to
miss; the U-turn recovery is what rescued the Gate 4 number, and Gate 5 passed
because its segments happened to start better aligned (worst landing 0.1449 m,
1 mm inside tolerance). Resolving that conflict is the open decision — see
"Current build" above.

*(Historic build state, 2026-08-08: the host ran the motion-disabled
`0.6.4-beta30` candidate with experimental motion verified off, and the branch
was at the undeployed `0.6.4-beta31`. Both are now beta39.)*

The card now emits the Gate 4 re-pass profile, so the profile-identity gap
(`docs/p0-beta-release.md:98-102`) is closed for those three fields
(`linear_pulse_duration_ms` 1300, `max_linear_commands` 3,
`max_turn_translation_distance` 0.30 sent explicitly). The profile is still
accepted on overshoot-and-recovery evidence only.

Do not change the accepted profile casually; changing it obligates the card
copy, a `CARD_VERSION` bump deployed to both serving paths, and the pinning
tests listed in §4 of the re-pass doc. See
`docs/CLAUDE-FINAL-IMPLEMENTATION-PROMPT.md` for the older implementation
handoff, noting that its turn-planning premise was overtaken by the 2026-08-05
measurements. No motion is authorized by this handoff. *(Superseded: the gate is currently
ARMED — see "Current build" above.)*

The card's Real Go defaults are frozen as `LUBA_ACCEPTANCE_PROFILE` in
`www/mammotion-custom-path-card.js` and pinned by frontend tests.

🏁 **GATE 5 PASSED 2026-08-08 — all five gates are complete.** Two card-driven
two-segment runs finished both segments with the accepted profile, zero errors,
zero reverse-recovery and no overshoot. Landings 0.0485 / 0.0836 / 0.0558 /
**0.1449** m against the adopted `waypoint_tolerance: 0.15`; the worst would have
failed at the old 0.08. Evidence: `docs/evidence-gate5-PASSED-20260808.json`.
Profile identity is now proven in fact — the card demonstrably sent the accepted
profile to the mower.

⚠️ Two fragilities the pass does **not** remove — **both rewritten 2026-08-08**
after the raw per-command record was recovered
(`docs/evidence-gate5-attempt5-segment1-raw-20260808.json`; analysis in
`docs/turn-rate-variance-and-reach-analysis-20260808.md`). Read that evidence
file before re-deriving any of this.

**The turn budget is NOT the fragility — that claim is refuted.** The
`turn_commands_sent: 4` was three turn-phase pulses plus one mid-drive
realignment on a *separate* budget; the turn phase stopped at
`target_heading_reached` on command **3 of 4**. The counter is reporting-only and
the true per-segment ceiling is **14**. The real fragility is **overshoot against
tolerance**: pulse 3 overshot the target heading by **13.258°** against
`heading_tolerance_degrees: 18` — **4.74° of margin**. The 2.6× rate spread is
partly an accounting artifact (`observed_rotation_ms` accumulates *nominal* pulse
duration, never measured `elapsed_ms`); on elapsed time two of the three pulses
agree to ~3% and only pulse 3 is anomalous. Pulse 3's rotation is nonetheless
real, and unexplained.

**The BLE `TimeoutError` is intermittent, not fixed** — it failed one attempt at
a 80.6° turn, yet a later run completed *larger* turns while showing degraded BLE
(writes median 540 ms) without tripping. Treat it as the tail of a latency
distribution, not a mystery. ⚠️ The stop confirmations 1175/1819/402/628 ms are
the **calibration and linear stops**, not turn stops — turn pulses record no stop
duration at all (`_vio_turn_to_heading` records none).

⚠️ `waypoint_tolerance` changed 0.08 → **0.15** in beta30 on hardware evidence
(`docs/evidence-slow-tier-validation-20260808.json`). The position feed is
~1031 ms stale and the mower covers 30–47 cm in that time, so 0.08 could never be
confirmed before the mower had passed the point.

⚠️ **The host and the branch have diverged.** The host still runs the deployed
`0.6.4-beta30`; the branch is at `0.6.4-beta31`, which is **built but never
deployed and never run on hardware**. Everything below describing runtime
behaviour is beta30 unless it says otherwise. See "beta31 (undeployed)" at the end
of this section.

The deployed `0.6.4-beta30` candidate is still unaccepted. On top of
beta22 it adds the read-only `report_stream_probe` diagnostic (beta23, now with
per-channel attribution) and an **RTK quality gate**: non-Fix refuses with
`rtk_not_precise` unless the caller passes `allow_degraded_rtk`, because Float
produced a 13.9 cm stationary jump on 2026-08-07 against an 0.08 m tolerance.
⚠️ RTK payload **age is reported but never blocks** — two thresholds (300 s,
1800 s) both false-blocked, a stationary mower is legitimately quiet for **up to
62.4 min measured**, and a forced burst cannot distinguish quiet from dead
either. This is **closed, not deferred**: do not turn age back into a blocker
without an active liveness probe, which does not exist. See
`docs/rtk-hardening-plan-20260807.md`.

beta27–29 add the read-only `basestation_info_probe`. It established that the
base **does** answer `request_basestation_info_t` — but returns
`score_info: null`, so **`base_moved` / `base_moving` are never populated on this
hardware** and that diagnostic avenue is closed. It also established the
correction chain: **internet source → base station (WiFi) → LoRa E22 → mower**
(base reports `rtk_over_internet`, mower `rtk_over_datalink`), which demotes the
"base survey never converged" hypothesis. ⚠️ Replies bearing the base's own
`iot_id` reduce onto `RTKBaseStationDevice`, **not** the mower's
`report_data.basestation_info` — reading only the mower will call a live base
silent. `MammotionRTKCoordinator` already queries this every tick.

~~⚠️ **Closed-loop segments cannot run after dark.**~~ **SUPERSEDED by night v1
items 15–18.** The historical reasoning below led to the later measurements but
its course-over-ground premise was refuted. The `vio_active` gate keys off
`turn_mode == "vio"` unconditionally, not off whether a turn is needed, and
`_VIO_TURN_MODES` was `("vio", "legacy")` only. ⚠️ **That symbol no longer
exists**: it is now `_SEGMENT_TURN_MODES = ("vio", "legacy", "night")`
(`_SEGMENT_TURN_MODES`), so the "no night-safe mode" half of this argument is
dead — night v1 added exactly that. *(Refined 2026-08-11 — read
`docs/night-motion-options-20260811.md`. The gate is created ONLY for
`turn_mode == "vio"` (the `vio_active` gate), so `legacy` skips it; but `legacy`
closes on `position.toward`, which was then believed to be course-over-ground and therefore blind to
in-place rotation **at any hour**, not just at night. The real constraint is not
"no heading at night", it is **"no heading while stationary"** — which is why an
ARC, never once sent by this project despite the wire accepting both axes, is the
open lead. 🗑️ **IR is CLOSED**: the mower really does dock on rear-facing IR, but
zero `infrared`/`ir_*`/`photoelectric`/`beacon` fields exist in the integration or
pymammotion, so it is firmware-internal and unreachable. Ultrasonic entities are
`SensorCheckState` self-check enums, not distances; `location.RTK.yaw` is `None`
on this hardware.)* Plan real-motion tests for
daylight. A zero-command
live snapshot proved Mammotion exposes only frozen course-over-ground while
stationary (`toward: -29.589`, VIO inactive/0, RTK yaw 0), so since beta19 the card stops
drawing that last-travel projection as current mower orientation and blocks
Nudge unless a trustworthy current orientation is explicitly available. `manifest.json`,
`pyproject.toml`, `CARD_VERSION` and `uv.lock` (PEP 440 — currently `0.6.4b31`) must always agree, and the
`Beta Release` workflow verifies all four. The card is served from **two**
paths, so deploy to both and bump the Lovelace resource key or the browser can
silently load the stale card. The live Lovelace URL includes the unique build
suffix `?v=<version>&build=<card md5 prefix>` (currently serving beta30). The misleading third-party-map
`card-mod` rotation was removed with verified config readback; its pre-change
backup remains `/config/.storage/lovelace.dashboard_yard.bak.codex-20260802-213848`.

## (history) beta31 — reach 4 segments + turn overshoot ceiling

Built 2026-08-08 on the branch. **No motion has run on it and it is not on the
host.** All CI gates pass locally (533 pytest, 20 frontend, ruff, mypy,
pre-commit). It touches **no `LUBA_ACCEPTANCE_PROFILE` key**, so the profile stays
accepted and no §4 re-pinning is owed.

1. **`REAL_CLICK_TO_GO_SEGMENT_LIMIT` 2 → 4** (`REAL_CLICK_TO_GO_SEGMENT_LIMIT`, mirrored by
   the card's `MAX_REAL_SEGMENTS`). ⚠️ **Segment 3+ has never been executed.** The
   VIO forward-heading offset is refreshed only from linear travel and never
   re-derived across a turn, so cumulative cross-track error past segment 2 is
   unmeasured — and attempt 5's segment 2 already produced the worst landing of
   the four (0.1449 m against 0.15 m).
2. **A turn overshoot ceiling**, `_VIO_TURN_CONSERVATIVE_MAX_DEGREES_PER_SECOND =
   60.0` — ⚠️ **removed in beta37** ("the turn model, rebuilt on measurement"),
   so this item and the beta32 objections built on it describe a constant that
   no longer exists. Caps each turn pulse so that even at 60 °/s it cannot sweep past
   `|error| + tolerance`. ⚠️ It **routinely becomes the active bound** on final
   approach rather than acting as a rare backstop, and it **couples turn dynamics
   to `heading_tolerance_degrees`**, which is a profile key. Below ~12° of
   tolerance the 400 ms actuation floor wins and the guarantee does not hold.
   Validated by replay arithmetic only — **zero hardware**.
3. **The rotation-rate estimator now divides by measured `elapsed_ms`**, not the
   commanded `pulse_ms` (`services.py`, the `heading_went_fresh` block). On its own
   this makes overshoot slightly *worse*, which is why item 2 ships with it.
4. Two reporting fixes: `motion_refresh_commands_sent` now folds in turn and
   realignment refreshes (it under-reported 6 against 15), and the mid-drive
   realignment no longer dispatches a no-op turn for aim errors already inside the
   turn tolerance — which makes `vio_realign_threshold_degrees` inert in the gap
   between it and `heading_tolerance_degrees`.

Handover, open attacks and the validation-run design:
`docs/HANDOVER-beta31-20260809.md`.

## (history) beta32 — beta31 reviewed, one fix, NOT cleared as-is

beta31 was adversarially reviewed on 2026-08-09 before any deployment and **did
not clear**. beta32 = beta31 + one refusal-side fix. Read
`docs/HANDOVER-beta31-20260809.md` §2.6 before touching turn code.

**Fixed:** `_vio_turn_budget_feasibility` assumed every turn command ran a full
`turn_pulse_duration_ms`, while beta31's ceiling shortens them — so the preflight
admitted turns the executor cannot finish (the two models disagree over
**100–117°** at a 4-command budget). It now replays the executor's own policy via
the same `_turn_final_approach_pulse_ms` the turn loop calls. A 90° junction reads
4 commands, not 3: feasible at **exactly** the budget, no margin.

**Open, and blocking a 90° L-path:**
1. ⚠️ **The ceiling costs ~18° of turn capability.** Replayed through the shipped
   code: a 90° junction completes on beta30 and **exhausts the 4-command budget on
   beta31** at 14.49/14.90 °/s — the rates Gate 5 attempt 5 actually measured. The
   handover's excuse for this was arithmetic that counted pulses to zero error
   instead of to tolerance; it is corrected in §2.2. Fix is to widen the overshoot
   allowance from `K = tolerance` to `K = 2 × tolerance` (~4.5° cost instead of
   ~18°). Not implemented.
2. ⚠️ **The ceiling's guarantee is in commanded ms; the mower rotates for the
   delivered window.** At the +260/+543 ms overruns already on record it holds only
   to 48.0/39.4 °/s — below the 49.56 °/s the hardware has produced.
3. ⚠️ **`_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND = 16.5` is not a floor** —
   14.490 °/s measured. Deliberately not lowered: at 14.4 a 90° junction needs 5
   commands against a budget of 4, so a truthful floor and L-path junctions are
   mutually exclusive until item 1 is fixed.
4. The ceiling biases turn landings toward the tolerance edge, feeding a *tighter*
   post-turn gate (15 vs 18) — expect more post-turn corrections and more
   cross-track error, working against the reach change.

**The validation run keeps every junction in the 45–70° band** — maximum exposure
to the ceiling (it binds below 72°) while clear of the contested 86–100° band.

🏁 **REACH GOAL MET 2026-08-09 — four segments executed on beta33.** Landings
0.0819 / 0.0662 / 0.1452 / 0.0990 m against `waypoint_tolerance: 0.15`, zero
reverse-recovery, zero realignments. **Error does NOT compound with segment
index** (seg4−seg1 slope +0.017 m), so the §2.4 worry is unsupported. Evidence:
`docs/evidence-beta32-4segment-20260809T183129Z.json`. The overshoot ceiling
works: three junction turns closed in **one command each**, landing −5.1 / −2.4 /
−0.3°, against Gate 5's 13.258° overshoot.

⚠️ **THE ROTATION-RATE VARIANCE IS LARGELY A BLE ARTEFACT — read
`docs/HANDOVER-beta31-20260809.md` §2.7 before touching any turn constant.** A
pulse rotates only while refresh writes arrive; when a write blocks, the mower's
watchdog stops the motor and the executor still divides by the whole window. A
1303.7 ms pulse that sent **one of six** refreshes, on a write that took
**1303.972 ms**, measured "9.23 °/s". Cadence-intact pulses that day measured
**23–43 °/s**. This substantially explains the Gate 5 overshoot the ceiling was
built for (a low estimate *lengthens* later pulses), and means
`_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND = 16.5` is probably **not** falsified —
do not lower it on the 14.49/14.90 readings, which are stall-degraded.

beta33 excludes such pulses from the rate estimate (`refresh_cadence_broken`,
`refresh_cadence_broken_pulses`). **Two earlier recommendations are WITHDRAWN:**
"K = 2 × tolerance unlocks 90° L-paths" (at a sustained slow rate no K helps — the
4-command budget caps rotation at ~55°), and the delivered-window shave (it
strictly worsens the binding constraint and would tune against a +0.03%–112%
spread). **The real open lead is BLE write latency, not turn tuning.**

Per-**click** reach is 4 segments; per-**segment** reach is ~1 m
(`max_linear_commands: 3` × ~0.35–0.42 m/pulse). A 2.0 m leg is not dispatchable
and stops on `max_linear_commands_reached`.

**DEPLOYED 2026-08-09 01:16–01:22 EDT, motion-disabled.** The host now runs
`0.6.4-beta32` (it skipped beta31 entirely); all 46 files byte-identical, both
card paths at `16d883fa`, resource `?v=0.6.4-beta32&build=16d883fa`,
`real_motion_allowed: false` read back. A zero-motion dry run confirms the new
preflight executes on the host (`command_count_model:
"executor_pulse_policy_replay"`, ladder `[1300.0, 942.5, 683.3]` for a 60°
junction — pulse 1 already ceiling-bound at 1300 ms instead of 1500). Evidence:
`docs/evidence-beta32-deploy-dryrun-20260809.json`; deploy record in
`docs/deploy-runbook-p0.md`. **No motion has run on beta31 or beta32.** The
4-segment validation run is pending daylight, a charged battery (mower is docked
at `CHARGE_ON`) and per-run authorization.

`pre-commit run --all-files` is green as of 2026-07-31 and is now a usable
gate. Its hook pins must move with `requirements_test.txt`: the Ruff and mypy
hook revs are pinned to the same versions CI installs, and skew between them is
what previously made the hook report failures CI does not have.

Repositories owned by `mikey0000` are read-only for this work. Do not push,
comment, open/close issues or PRs, or publish anything there. A later authorized
push goes only to the `Chorty` fork.

## Build Commands

There is no global `uv` on the dev machine — `uv sync`/`uv run` fail with
`command not found`. Use the project venv directly. (`.venv/bin/uv` exists only
because it was pip-installed into the venv to regenerate `uv.lock`; it cannot
bootstrap the venv it lives in.)

Run the same commands CI runs, so a green local run means a green CI run
(`.github/workflows/validate.yml` is the source of truth):

- Tests: `.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests`
- Lint: `.venv/bin/python -m ruff check custom_components tests`
- Format check: `.venv/bin/python -m ruff format --check custom_components tests`
- Type checking: `.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion`
- Frontend card tests: `npm run test:frontend`
- Pre-commit: `.venv/bin/python -m pre_commit run --all-files`
- Install/refresh deps: `.venv/bin/python -m pip install -r requirements_test.txt`
  (CI's install step; keep the `pymammotion` pin here identical to
  `manifest.json` or CI tests a different backend than the one shipped)

⚠️ Use mypy's `--follow-imports=skip custom_components/mammotion` form above, not
a whole-tree `mypy custom_components/`. The broader form reports 4 pre-existing
errors in `select.py` and `device_tracker.py` that CI never checks — they are not
regressions from your change.

Note: `ruff format` rewrites `except (A, B):` (no `as` binding) into
`except A, B:`. That is **correct and intentional** — PEP 758 allows
unparenthesized exception tuples from Python 3.14, which this project targets
(`requires-python = ">=3.14.2"`). It looks like the Python 2 form and is not;
verified parsing and catching on 3.14.6. Do not "fix" it back.

## Code Style Guidelines

- Follow Home Assistant integration patterns
- Use async/await patterns (prefix functions with `async_`)
- Line ending format: LF (not enforced by any hook or editor config)
- Prefer specific exception types over broad ones

When making changes, follow existing patterns in similar files and follow Home Assistant best practices.

## Home Assistant Integration Rules

- All imports within the integration must be relative (e.g. `from . import Foo`, `from .services import bar`). Never use `from custom_components.mammotion import ...` — HA loads integrations in a way that makes absolute imports from `custom_components` fail at runtime.

## Model and Subagent Routing

**Route by cost of being wrong, not by price per token.** Sonnet 5 is only ~1.67×
cheaper than Opus 5 ($3/$15 against $5/$25 per MTok), so a cheaper model that
needs two passes where Opus needs one has already cost more — before counting the
time spent reviewing the bad first pass. The token gap is the small term. A
plausible-but-wrong claim that reaches an evidence file or shapes a hardware run
is the large one, and this project has been bitten by exactly that repeatedly.

**Sonnet when a machine catches a wrong answer, or the work is high-volume and
mechanical:**

- Deploys — md5 comparison and the `real_motion_allowed: false` readback catch errors
- Version bumps across the four sites — the `Beta Release` workflow verifies all four
- Running the CI gate suite and reporting pass/fail
- Translations sweeps across every language file — JSON parse plus key-presence check
- Broad symbol/reference sweeps where only the conclusion is needed

**Opus when the output is a claim or carries a consequence:**

- Anything touching the motion control law, or any `LUBA_ACCEPTANCE_PROFILE` decision
- Interpreting a run's telemetry; deciding whether a fix actually worked
- Adversarial review, and adjudicating findings
- Analysis written into a `docs/evidence-*` file — it becomes load-bearing for later sessions
- Supervising real motion

**Testing: separate the run from the diagnosis.** Have the cheap session run the
suite and report **raw output only** — which tests failed and the actual
traceback, no interpretation — then stop. Most runs pass, so the common case is
cheap. On a failure, `/model opus` continues in the *same* session with the output
already in context; that invalidates the prompt cache once, but it is not a
restart. A plausible-looking diagnosis from a cheaper model is the exact failure
mode to avoid here.

**For search, prefer inline grep over a subagent.** A subagent returns a summary,
and this project's rule is *verify with per-item records, not aggregates*. On
2026-08-09 two verifier agents wrongly reported that
`REAL_CLICK_TO_GO_SEGMENT_LIMIT` does not exist; one inline grep disproved it.
Use the `finder` agent only when the sweep is genuinely broad, spans several
naming conventions, and the conclusion is all that is needed. When the individual
hits matter — which is most of the time in this repo — grep inline.

**In workflows** (`Workflow` tool), set `opts.model` per stage: cheap models for
find/scan stages, Opus for verify and adjudicate. That is the shape the 2026-08-08
turn-variance investigation used — six Sonnet finders, six Opus verifiers, one
Opus critic. Named agents follow the same split: `finder` for scans, `verifier`
for confirming or refuting a candidate, Opus for fix authoring.

## Translations

- When adding or renaming any entity (sensor, switch, button, number, select, etc.) or an ENUM entity state, you MUST update the translations in **every** language file, not just English.
- The files to keep in sync: `custom_components/mammotion/strings.json` (the source) **and** every file under `custom_components/mammotion/translations/`. Treat that directory listing as the source of truth for which languages exist.
- Translate the entity `name` and every ENUM `state` value into each language's own language — do not copy the English text into the other locales as a placeholder.
- Also add an icon entry in `custom_components/mammotion/icons.json` for the new entity where appropriate.
- After editing, confirm every JSON file still parses and that the new key (with all its `state` values) is present in each file before considering the change complete.
