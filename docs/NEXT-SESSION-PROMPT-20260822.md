Continue the Mammotion-HA work. Read `CLAUDE.md` "Start here" and
`docs/NEXT-SESSION.md` §0 first, then this. **At handoff it is night and the
operator reports the mower is charging after its battery died. Do not perform
hardware work until daylight and a fresh safety check.**

## UNCOMMITTED WORKTREE — preserve and review before staging

The preserved work below was created after clean base **`2aca5e0a`**. Run
`git log -1 --oneline` for the handoff note's own commit. The worktree is
intentionally **not clean** at handoff. These pre-existing/user-owned changes
appeared after that base:

- modified `docs/NEXT-SESSION.md` with an open refresh-cadence finding;
- untracked `docs/refresh-cadence-may-be-3x-too-fast-20260819.md`;
- untracked `scripts/analyze_refresh_cadence.py`; and
- untracked `PyMammotion examples/` containing 13 example Python files.

Do not discard, overwrite, or commit these wholesale. Inspect provenance and
intent first. The refresh analyzer passes Ruff and reproduces 120 cruising
pulses with median speed flat through the 400–700 ms write-duration bucket,
then lower in the thin 700–1200 ms (`n=13`) and 1200+ ms (`n=2`) tails. That is
correlational proxy evidence, not permission to change the frozen 200 ms
profile. The copied examples are untracked and six files contain
authentication-related fields; perform a secrets/provenance review before any
staging. No network or mower call was used to verify this handoff note.

## ✅ SUPERSEDED 2026-08-22 — the Phase 1 captures are DONE, verdict `no_go`

Both windows ran under separate authorizations, moved the mower, stopped, and
left the gate verified disarmed. One criterion of 17 failed,
`shallow_arc.bearing_toward_compass_mirror` at 12.631° against 10°. The
diagnosis is that the criterion compares an interval-average chord bearing to an
instantaneous `toward` sample on a body rotating ~10° per interval; VIO
corroborates that `toward` tracked the rotation correctly. **The `no_go` stands
and Phase 2 is not unlocked.** Full writeup and the next question:
`docs/phase1-continuous-motion-captures-20260822.md`. The numbered procedure
below is kept as the record of how the captures were run.

## (superseded) CURRENT NEXT ACTION — Phase 1 captures at first safe daylight opportunity

Offline preparation is complete at pushed commit **`d105f4ca`**. Do not design
or implement Phase 2 yet. When the mower is charged and the operator is present:

1. Re-read the live host state. Treat the last beta70 disarm result as evidence,
   not current truth; verify the experimental gate is off before preparation.
2. Freshly scan and freeze contained routes from the mower's then-current
   position. Each route needs at least 1.2 m area margin and 1.5 m keep-out
   margin. Do not reuse the Route B scan or guess corridor metadata.
3. Confirm daylight, clear route, blades off, operator present, and accessible
   emergency stop. Show the exact straight 4 s route and obtain authorization
   for that individual run only.
4. Run the predeclared straight profile (`linear 400`, `angular 0`, 4,000 ms,
   200 ms refresh, 100 ms cache sampling) inside a `try`/`finally` that disarms
   and verifies the gate afterward. Bank the complete response.
5. Repeat the route display and explicit-authorization step separately for the
   shallow arc (`linear 400`, `angular 180`); do not treat straight-run consent
   as arc-run consent. Bank the complete response and verified final disarm.
6. Run `scripts/analyze_phase1_capture.py` offline with both responses and the
   fresh frozen-corridor JSON. A `no_go` stops the experiment. A `go` permits
   Phase 2 design discussion only; it does not authorize another physical run.

The deferred timed-disarm automation is **no longer on hold — it was installed
2026-08-22** (see open item 3 below). Recovering the older
2.72 m card-history result is useful archival evidence but is not a prerequisite
for Phase 1 and should not displace the fresh capture work.

## UPDATE — beta69 supersedes the state and Task 1 below

Beta69 is released, installed motion-disabled, and browser-verified. It closes
the keep-out segment gap: the backend and card refuse a crossing/touching leg
with `path_legs_cross_keep_out_zone`. Real-map crossing and legal-control
previews passed; browser beta69 footer/console, named refusal, red/dashed leg,
and disabled Real Go passed. Gate DISARMED, no active session, no motion
commanded. See `docs/deploy-runbook-p0.md` → beta69.

The beta67 browser check in Task 1 is complete and superseded. Do not expect
Real Go to remain available for a crossing leg on beta69; refusal is the fix.

## UPDATE — the 3 x 3.0 m Route B chain completed

The supervised beta69 run completed **3 of 3** collinear 3.0000 m sub-legs with
`target_reached` landings at **0.14388 / 0.11413 / 0.06070 m** (mean 0.10624 m),
no blockers, and no failed segment. The 9.0000 m frozen route ran from
`(4.8756, -2.4530)` to `(11.7193, -8.2980)` at heading 319.5°. Its pre-run scan
held 1.2 m area and 1.5 m keep-out margins; measured minimum clearances were
1.21 m and 4.41 m respectively. Full evidence:
`docs/evidence-route-b-3x3m-beta69-20260821T193417Z.json`.

The gate was armed only around the authorized run and verified DISARMED after:
`enabled: false`, `real_motion_allowed: false`, no active session,
`MODE_PAUSE`. Combined 3.0 m evidence is now **5 reached / 1 failed, n = 6**;
this proves a 3-leg chain can complete, not that 3.0 m is reliable.

## UPDATE — continuous motion Phase 1 instrumentation is deployed

A pure lookahead controller and JSON replay now exercise bounded steering and
fail-closed fault decisions with **no Home Assistant import or dispatch path**.
The existing bounded raw probe now also accepts opt-in
`in_window_sample_interval_ms`; at 100 ms it concurrently records cached x/y,
`toward`, VIO, `last_report_at`, active command, and refresh completions without
extra in-window BLE report requests. It fails closed if refresh is off or stream
startup fails, and dry run starts no stream and sends no command.

Beta70 is deployed motion-disabled and browser-verified. The exact straight and
shallow-arc plans both passed deployed dry run with 41 planned samples,
`would_send: false`, and no command or stream attempt. No physical capture was
run and no mower command was sent. No continuous executor/service exists. Each
later 4 s physical window needs separate explicit authorization.
Exact go/no-go criteria:
`docs/continuous-motion-feasibility-plan-20260821.md`.

The offline Phase 1 analyzer is also implemented and tested. After the two
separately authorized captures exist, use `scripts/analyze_phase1_capture.py`
with the straight response, shallow-arc response, and freshly validated frozen
corridor metadata. It recomputes the criteria from raw samples, hashes all
inputs, and cannot dispatch or authorize motion. Usage and input schema:
`docs/phase1-capture-analyzer.md`.

## STATE (verified 2026-08-21 ~end of session — RE-VERIFY BEFORE ACTING)

- Host runs **`0.6.4-beta70`**, motion-disabled after the Phase 1 instrumentation
  deploy; deployment and browser verification are recorded in
  `docs/deploy-runbook-p0.md`.
- The preserved cadence work began after documentation commit **`2aca5e0a`**;
  analyzer commit **`d105f4ca`** is its parent. The worktree has the uncommitted
  cadence-analysis and example files listed at the top of this file; do not
  infer cleanliness from the later handoff-note commit.
- Motion gate was last **DISARMED and verified** after beta70
  (`enabled: false`); re-read it before treating that as current state.
- Last banked telemetry saw the mower at `(11.7615, -8.2563)`, `AREA_INSIDE`,
  RTK Fix, `MODE_PAUSE`. Afterward its battery died; the operator reports it is
  now charging. Charging state and any position change are not telemetry-verified.
- Current baseline: **806 pytest, 91 frontend**, ruff, ruff format, mypy, ten
  pre-commit hooks, `check_doc_symbols.py` 1172 claims, and
  `check_accepted_profile.py` **ACCEPTED**.

Everything below this state block is retained as evidence or a superseded task
record. The current next action above takes precedence.

## TASK 1 — the browser check nobody has done (do this first, it is free)

beta67 ships card keep-out rendering and the leg-length advisory. **The bytes
are verified on the host; no browser has rendered them.** beta49 is the
precedent: four card defects existed *only* against real output.

Ask the operator to load the dashboard and confirm four things:

1. Card footer **and** console banner both read `0.6.4-beta67`. If either says
   beta66, it is a stale cache and the deploy failed regardless of the hashes.
2. Two dashed red zones labelled `⛔ obstacle` appear on the map.
3. A click **inside** one is refused: "that point is inside a keep-out zone
   (obstacle)".
4. A click ~3 m out shows the ⚠️ advisory naming **0.58 m** — as a *warning*,
   with the run still available. **If it blocks the run, that is a bug**; the
   advisory must never gate.

⚠️ The mower is off the dock, so `position_not_valid_for_motion` may or may not
appear as a blocker depending on where it sits. On the dock it is expected.

## TASK 2 — COMPLETE: 3.0 m Route B chain

**Yes: a 3-sub-leg collinear chain completed at 3.0 m.** The beta69 run reached
all three targets at 0.14388 / 0.11413 / 0.06070 m. Earlier runs got 1-of-3 and
2-of-3, so the result demonstrates feasibility but not reliability.

Current record at 3.0 m: **5 reached / 1 failed, n = 6.**
A **2.27 m** single vector segment also reached target at **0.11378 m**
(`docs/evidence-vector-segment-2p27m-20260820.json`, recovered from the card's
run history 2026-08-21 before it was overwritten).

🔑 **The one-sided aim drift is now seen in three independent runs** — that
2.27 m segment ran **8 of 8 pulses negative, mean −10.87°**, growing −7.7° →
−19.8° as range closed, the same signature as the chain run's sub-leg 2
(−10.29°, 9 of 9). It is **not noise**. It reached target anyway *because the
leg was shorter*, which is direct support for the correctable-leg-length bound:
outcome is set by how much cross-track the leg length lets an uncorrectable aim
error accumulate.

- 3.0 m single sub-leg → `target_reached` at **0.1484 m** (1.6 mm of margin)
- 3.0 m chain → sub-leg 1 **0.094 m**, sub-leg 2 failed at 0.2594 m
- A complete 3.0 m chain is now demonstrated, but one of six measured 3.0 m
  landings still failed and the conservative controller bound remains open.

⚠️ **One run is still unrecovered.** The card's landing table shows an earlier
2-segment run at **2.72 m** (landings 0.1319 / 0.0716, mean 0.1017, 2 of 2
`target_reached`). It is not the file above and the card keeps only ~3–5 full
results, so it may already be gone. If it is still in the history, download and
bank it — it is two more long-leg landings.

Procedure that worked, in order:

1. **Scan from the mower's ACTUAL current position** —
   `scripts/scan_contained_bearings.py --distance 9.0 --margin-area 1.2
   --margin-keepout 1.5`. It checks the area polygon **and**
   `export_map.keep_out_polygons`, samples the whole leg every 5 cm, and holds
   a clearance margin.
2. **FREEZE the scanned endpoint.** 🚨 On 2026-08-20 a dispatch script re-derived
   the endpoint from live position "to be safe" and silently drove a path the
   scan never covered. The dispatcher must compare live position to the frozen
   start and **ABORT if drift > 0.30 m**, never re-derive.
3. **Send the full `LUBA_ACCEPTANCE_PROFILE`.** A direct service call that omits
   it gets backend schema defaults — `max_linear_pulse_ceiling: None`,
   `waypoint_tolerance: 0.08`, `linear_pulse_duration_ms: 3500` — and a 3.0 m
   sub-leg then dies at ~1 m on `max_linear_commands_reached`. Extract the
   literal from the card.
4. **Send `max_real_segments: 4`.** Omitting it defaults to **1**, which is why
   the first run drove only one sub-leg.
5. **Use `split_leg_target_length_m: 3.2`, not 3.0.** `ceil()` is a step
   function: a 9.000007 m frozen leg divided by 3.0 gives `ceil(3.000002) = 4`
   sub-legs of 2.25 m. At 3.2 it gives exactly 3 × 3.0000 m.
6. Dry run → explicit per-run authorization → arm **inside** the same `try`
   whose `finally` disarms → run → **disarm and verify** → bank evidence
   immediately.

⚠️ **Do not promote this single complete chain into a reliability claim.** The
earlier failure was the control law, not the splitter and not BLE.

## THE OPEN PROBLEM, STATED PRECISELY

**Cross-track accumulates over a long leg, and the correction floor cannot act
until the error is already too expensive.**

`_MIN_CORRECTABLE_AIM_ERROR_DEGREES` = post-turn tolerance 10 + deadband 5 =
**15°**. A correction fires only at or above that, so an error just under it is
**never corrected**, and it buys `distance × sin(floor)`. Hence
`_correctable_leg_length_limit_m` = `tolerance / sin(floor)` = **0.580 m** on
the accepted profile. At 3.0 m the same floor permits an uncorrectable
**0.776 m** miss. That is why ~0.8 m is robust while 3.0 m can still fail.

Sub-leg 2 died exactly this way: a one-sided aim error grew −7.3° → −18.1° over
nine pulses, corrected once, then needed **51.025°** at 0.2594 m to run and was
refused `turn_budget_infeasible`.

🚨 **Do NOT lower the floor.** It is set by the turn primitive's actuation
limit: protecting a 3.0 m leg would need a ~2.9° floor while the affine sweep
bound still permits 20° at the 200 ms floor, so a sub-floor correction
manufactures the error it was meant to remove. A test pins that arithmetic.
🚨 **Do NOT raise `vio_max_realignments`.** Tried, reviewed twice, reverted
twice, for two different reasons.

The lever is **reducing cross-track accumulation**, or **correcting earlier
while the geometry is still forgiving** — not more late corrections.

## WHAT 2026-08-20 MEASURED (read-only, no motion — do not re-derive)

- **The position feed is ~1 Hz**, moving or stationary. The settle loop already
  polls at 1.0 s, so polling faster buys nothing. The 2.85 s settle is near the
  **floor** of stop-measure-go, not slack.
- Motion runs at a **29% duty cycle** — 4.55 s cycle, 1.30 s of it moving; 57%
  of a run is position-settle.
- **The vendor drives continuously at ~0.55 m/s on this same ~1 Hz feed**, so
  1 Hz does not block a continuous controller. It is what makes stop-measure-go
  expensive.
- **Next position IS predictable** from last fix + commanded velocity to
  **0.029 m median / 0.097 m p90** when refresh cadence holds (180 of 262
  pulses) — ~5× better than tolerance.
- 🗑️ **`ble_rssi` does NOT predict cadence** (within-run median r = +0.042 over
  24 runs). A marginal RSSI is not a reason to postpone a run *or* to trust one.
- 🗑️ **The time-into-run cadence trend is a population tendency, NOT a
  within-run law.** It failed to predict the chain run, which had zero stalled
  pulses in 19 across 130 s. Do not use it to attribute an individual failure.

## OTHER OPEN ITEMS — `docs/open-items-20260821.md` names the check for each

1. ✅ **Keep-out segment containment shipped in beta69.** Backend and card
   refuse crossing/touching legal-endpoint legs; real-map and browser checks
   passed. `test_a_leg_that_clips_a_corner_is_caught` pins the closed gap.
2. **`safety_overrides` is not wired into the movement primitives** —
   `MOVEMENT_SCHEMA` and `MANUAL_VELOCITY_PULSE_TEST_SCHEMA` cannot express an
   override. That gap is *why* the nudge buttons had to be ungated outright.
3. ✅ **CLOSED 2026-08-22 — the disarm automation IS installed.** The gate had
   been found armed at rest four times, twice on 2026-08-20 alone.
   `automation.mammotion_disarm_motion_gate_when_left_armed` is live on the host
   (id `1755900000001`, state `on`, `last_triggered: None`), appended to
   `/config/automations.yaml` and loaded by `automation.reload`. It is one-way —
   no arm service exists — and `disarm_experimental_motion` refuses while a
   session is active, so it cannot interrupt a supervised run. Host backup:
   `/config/automations.yaml.bak.claude-20260821-disarm`.
   *Checked:* live `/api/states` shows the automation entity on; entity ids in
   the YAML match the live install.
4. Ceiling `14 → 22` still untested; needs a leg over ~5 m.

## DISCIPLINE THIS REPO ENFORCES (all of these bit someone on 2026-08-20)

- **Verify against the tree.** An audit that day found a wrong gate count and
  **16 rotting `file.py:NNNN` citations** — 8 of the first 10 sampled pointed at
  unrelated code, two of them annotated "line numbers verified". All in-repo
  code citations are now symbol references, and `check_doc_symbols.py` refuses
  new ones. **Cite symbols, never line numbers.**
- **Check gate EXIT CODES.** Never `cmd | tail` — the pipeline's status is
  `tail`'s. This was reported as a green `EXIT: 0` twice that day.
- **`pre-commit run --all-files` does NOT check untracked files.** A new test
  file shipped 4 ruff errors past a fully green run; `ruff check
  custom_components tests` caught them. `git add` before trusting pre-commit.
- **PUSH before dispatching the Beta Release workflow.** It was fired once while
  `main` was 10 commits ahead of `origin` and would have cut the release from
  the wrong commit. Cancelled in time.
- **NO MOTION without explicit per-run authorization.** Arm only for the run,
  then disarm and verify. Arm *inside* the `try` whose `finally` disarms —
  calling enable is what obliges the disarm, never "enable succeeded".
- **VIO is a cliff, not a gauge.** 80 = saturated. It went 80 → 62 → **0** in
  about 30 minutes at dusk. `camera_brightness` latched on "Light" 41 minutes
  stale with the sun below the horizon — **do not trust it**; check the live
  feed via a dry run. A green 11/11 gate list does **not** mean a VIO run can
  proceed; the executor refuses later.
- **Bank evidence into `docs/` immediately**, including failures. Selectively
  discarding a failed run biases the record.

## THE NUDGE BUTTONS ARE DELIBERATELY UNGATED

`_nudge_available` returns `True` unconditionally and `_unguarded_nudge` calls
the coordinator primitive directly. Operator's explicit decision, taken after
the trade-offs were stated, because the mower stranded itself in a no-go zone
where every guarded path refused. **Do not "fix" this back** —
`tests/components/mammotion/test_nudge_buttons_ungated.py` pins it.

## USEFUL TOOLING ADDED 2026-08-20

- `scripts/scan_contained_bearings.py` — longest contained click, area **and**
  keep-outs, whole-leg sampling with margins. Bearing convention is the
  integration's own: `atan2(dy, dx)` CCW from +x.
- `scripts/replay_position_predictability.py` — one-step prediction error.
- `scripts/replay_ble_cadence.py` — what predicts cadence collapse (within-run,
  not pooled; pooling reverses the RSSI result).
