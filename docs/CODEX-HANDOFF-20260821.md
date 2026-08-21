# Codex handoff — 2026-08-21

## UPDATE — segment-level containment implemented, not released

The primary task below is now implemented in the working tree and remains
**unreleased/uninstalled**. The host still runs beta68 with the old per-point
backend behavior.

- `_keep_out_leg_violations` checks each legal-endpoint leg against every
  keep-out edge and reports the leg indices, endpoints, zone kind, and hash.
- `_validate_custom_path` refuses it by name with
  `path_legs_cross_keep_out_zone`.
- Backend and card both include boundary touches and collinear edge overlap;
  the card now blocks Real Go locally instead of showing an advisory-only
  warning.
- The former gap test is now `test_a_leg_that_clips_a_corner_is_caught`.
- Split interaction is pinned: an already-split crossing path remains refused;
  inserted points inside a zone retain the existing point-level reason.
- Verification: 755 pytest, 91 frontend, ruff, ruff format, mypy, all ten
  pre-commit hooks, `check_doc_symbols.py`, and
  `check_accepted_profile.py` ACCEPTED.

Release behavior change: paths with legal waypoints whose connecting leg
crosses or touches a keep-out will now be refused. On the next deployed build,
repeat the browser check below but expect **Real Go to be disabled**, with the
named crossing reason. No mower motion is needed to verify this.

Read `CLAUDE.md` "Start here" and `docs/NEXT-SESSION.md` §0 before acting.
This file is the session-specific brief; those two are the standing state.

⚠️ **`/codex:review` and `/codex:adversarial-review` are DIFF-scoped.** On a
clean tree they report "no changes to review" and that looks like a pass. To
review this session's work, pass an explicit base:

    /codex:adversarial-review --base b28252d4 <focus text>

`b28252d4` is the commit before this session started; everything since is the
work described below.

## ⚠️ UNDEPLOYED COMMIT ON MAIN — fold it into your next build

`main` is **one card fix ahead of the host**. beta68 is deployed; commit
`1b27cf15` is not.

**What it fixes:** the keep-out crossing colour was applied *before* the
segment-verdict block in the SVG draw loop, so `if (segments)` overwrote it.
Once a run existed, a leg crossing an obstacle zone was repainted **green** by
`seg.passed === true` — the hazard vanished from the map exactly when there was
a completed drive through the zone to look at. The crossing colour is now
applied last, pinned by a test asserting source order (a value assertion would
not catch a re-ordering, which is the real failure mode).

**Nothing is broken on the host meanwhile**: the 🚨 banner warning is deployed
and confirmed working live on beta68. Only the redundant map colouring is
affected, and only after a run completes.

**Action:** no separate release needed — bump and deploy it with whatever you
ship next, and re-run the beta68 browser check below afterwards.

## STATE (re-verify before acting)

- Host runs **`0.6.4-beta68`**, motion-disabled. 46/46 byte-identical, card md5
  `f143465a5bb120ed759ab328c15dad9f` at both paths, resource
  `?v=0.6.4-beta68&build=f143465a`, config entry `loaded`.
- `main` clean and pushed, **one commit ahead of the host** (see above).
- Motion gate **DISARMED**. ⚠️ It was found **ARMED at rest** before this
  deploy — the third such occurrence in two days.
- Mower off the dock, `MODE_PAUSE`, around `(4.88, −2.45)`. **It moves between
  sessions; re-scan, never trust a recorded position.**
- Gate baseline: **752 pytest, 90 frontend**, ruff, ruff format, mypy, ten
  pre-commit hooks, `check_doc_symbols.py`, `check_accepted_profile.py`
  **ACCEPTED**.

## THE ONE THING MOST WORTH DOING: segment-level containment

**A leg can be driven straight through a keep-out zone and nothing refuses it.**

Containment is **per-point** on both sides. Click two legal points either side
of an obstacle zone and the path goes through it: `_validate_custom_path` tests
each point, sees both outside, and passes. Confirmed in a browser on
2026-08-21, and it is easy to do by accident.

- Backend: `_keep_out_violations` is per-point, pinned deliberately by
  `test_a_leg_that_clips_a_corner_is_not_caught`.
- Card (beta68): `_legsCrossingKeepOuts` now detects the crossing, paints the
  leg red/dashed and warns — but **the card cannot refuse what the backend will
  still dispatch.**

**The fix is segment-level containment in the backend.** If you implement it:

1. Test the **segment** against every keep-out edge, not just its endpoints.
   The card's `_segmentsIntersect` / `_legsCrossingKeepOuts` are a working
   reference — mirror the semantics so the two cannot disagree.
2. It must refuse with a **named reason**, matching how every other gate in
   `services.py` refuses.
3. ⚠️ **`test_a_leg_that_clips_a_corner_is_not_caught` should then FAIL.** That
   is the point of it — rewrite it to assert the leg IS caught, and say in the
   docstring that the gap is closed.
4. ⚠️ Consider the **split** interaction: `_split_long_legs` inserts collinear
   points *before* validation, so a long leg through a zone may already produce
   an inserted point inside it. Check whether segment-level containment changes
   behaviour for already-split paths.
5. It moves no `LUBA_ACCEPTANCE_PROFILE` key, so it owes no Gate 5 — but it
   **can refuse runs that previously dispatched**, which is a behaviour change
   the operator should be told about explicitly.

## SECOND: the control-law problem, stated precisely

**Cross-track accumulates over a long leg, and the correction floor cannot act
until the error is already too expensive.**

`_MIN_CORRECTABLE_AIM_ERROR_DEGREES` = post-turn tolerance 10 + deadband 5 =
**15°**. A correction fires only at or above that, so an error just under it is
**never corrected**, and it costs `distance × sin(floor)`. Hence
`_correctable_leg_length_limit_m` = `tolerance / sin(floor)` = **0.580 m**. At
3.0 m the same floor permits an uncorrectable **0.776 m** miss.

**The drift is real and measured in three independent runs** — one-sided,
consistent within a run, growing monotonically as range closes:

| run | leg | per-pulse aim error | outcome |
|---|---|---|---|
| card segment | 2.27 m | 8/8 negative, mean −10.87°, −7.7 → −19.8° | reached, 0.1138 m |
| chain sub-leg 1 | 3.00 m | +7.87°, 0/10 negative | reached, 0.094 m |
| chain sub-leg 2 | 3.00 m | −10.29°, 9/9 negative | **failed**, 0.2594 m |

Sub-leg 2 died needing **51.025°** at 0.2594 m to run, refused
`turn_budget_infeasible`.

🚨 **Two fixes are already REFUTED — do not propose either:**

- **Lowering the floor.** It is set by the turn primitive's actuation limit:
  protecting a 3.0 m leg needs a ~2.9° floor while the affine sweep bound still
  permits 20° at the 200 ms floor, so a sub-floor correction manufactures the
  error it was meant to remove. A test pins that arithmetic.
- **Raising `vio_max_realignments`.** Tried, reviewed twice, reverted twice,
  for two different reasons.

The lever is **reducing cross-track accumulation**, or **correcting earlier
while the geometry is still forgiving**.

## MEASURED 2026-08-20/21 — do not re-derive

- **Position feed is ~1 Hz**, moving or stationary. The settle loop already
  polls at 1.0 s; polling faster buys nothing. The 2.85 s settle is near the
  **floor** of stop-measure-go, not slack.
- Motion runs at a **29% duty cycle**; 57% of a run is position-settle.
- **The vendor drives continuously at ~0.55 m/s on this same ~1 Hz feed.** 1 Hz
  does not block a continuous controller — it is what makes stop-measure-go
  expensive.
- **Next position IS predictable** from last fix + commanded velocity to
  **0.029 m median / 0.097 m p90** when refresh cadence holds.
- 🗑️ **`ble_rssi` does NOT predict cadence** (within-run median r = +0.042 over
  24 runs; pooled −0.245 is a between-session confound).
- 🗑️ **The time-into-run cadence trend is a population tendency, NOT a
  within-run law.** It failed to predict the chain run, which had zero stalled
  pulses in 19 across 130 s.

## HARDWARE, IF DAYLIGHT AND AUTHORIZATION ALLOW

**Does a 3-sub-leg collinear chain complete at 3.0 m?** Best so far is 2 of 3.
Current record at 3.0 m: **2 reached / 1 failed, n = 3**.

Five payload traps that each cost a run:

1. **Freeze the scanned endpoint.** A dispatcher that re-derives it from live
   position silently drives a path the scan never covered. Abort if drift
   > 0.30 m.
2. **Send the full `LUBA_ACCEPTANCE_PROFILE`.** Omitting it yields schema
   defaults (`max_linear_pulse_ceiling: None`, `waypoint_tolerance: 0.08`,
   `linear_pulse_duration_ms: 3500`) and a 3.0 m leg dies at ~1 m.
3. **Send `max_real_segments: 4`** — it defaults to **1**.
4. **Use `split_leg_target_length_m: 3.2`, not 3.0.** `ceil()` is a step
   function; a 9.000007 m leg over 3.0 gives 4 sub-legs of 2.25 m.
5. **Arm inside the same `try` whose `finally` disarms.** Calling enable is what
   obliges the disarm, never "enable succeeded".

Scan with `scripts/scan_contained_bearings.py --distance 9.0 --margin-area 1.2
--margin-keepout 1.5` — it checks area **and** keep-outs, samples the whole leg
every 5 cm, and holds a margin.

## DISCIPLINE (every one of these bit someone in the last two days)

- **Cite symbols, never line numbers.** An audit found 16 rotting
  `file.py:NNNN` citations; 8 of the first 10 sampled pointed at unrelated code,
  two annotated "line numbers verified". `check_doc_symbols.py` now refuses new
  in-repo code citations.
- **Check gate EXIT CODES.** Never `cmd | tail` — the status is `tail`'s.
- **`pre-commit run --all-files` does NOT check untracked files.** A new test
  file shipped 4 ruff errors past a green run. `git add` first.
- **PUSH before dispatching the Beta Release workflow.** It was fired once with
  `main` 10 commits ahead of `origin`.
- **NO MOTION without explicit per-run authorization.** Arm only for the run,
  disarm and verify after.
- **VIO is a cliff, not a gauge.** 80 → 62 → 0 in ~30 minutes at dusk.
  `camera_brightness` latched "Light" 41 minutes stale with the sun down. A
  green 11/11 gate list does **not** mean a VIO run can proceed.
- **Bank evidence into `docs/` immediately, including failures.** Discarding a
  failed run biases the record.
- **A screenshot found what 85 tests missed.** Render against real output.

## DO NOT "FIX" THESE

- **The nudge buttons are deliberately ungated.** `_nudge_available` returns
  `True` unconditionally — operator's explicit decision after the mower
  stranded itself in a no-go zone. `test_nudge_buttons_ungated.py` pins it.
- **The leg-length advisory must never block.** A 3.0 m leg reached target at
  0.094 m; the bound is deliberately pessimistic. A test pins that case.
- **The keep-out leg warning must never block** while the backend still
  dispatches such paths.

## OWED BROWSER CHECK (only the operator can do this)

beta68 is verified on the host but not in a browser. Draw a path **through** a
keep-out zone and confirm the leg renders **red and dashed** with the 🚨 banner
line, and that **Real Go stays available**.
