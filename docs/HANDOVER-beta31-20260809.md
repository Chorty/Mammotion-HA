# Handover — `0.6.4-beta31`, built and unvalidated

Written 2026-08-09 for a reviewer who has read nothing else. It is written to be
**attacked**: the weakest parts are named first, with the evidence needed to kill
them. Nothing here is settled by authority — every claim carries a `file:line` or
an evidence file, and where something is unproven it says so.

## 1. Status in one paragraph

All five gates are complete (Gate 5 passed twice, 2026-08-08). The operator's
actual goal — clicking anywhere on the map — is **not** met: per-click reach is
~1–2 m. beta31 raises the real segment limit 2 → 4 and adds a hard ceiling that
bounds turn overshoot. It is **built, fully green on CI, never deployed, and no
motion has ever run on it.** The host is still on beta30. beta31 touches **no
`LUBA_ACCEPTANCE_PROFILE` key**, so the hardware-accepted profile remains accepted
and no §4 re-pinning is owed.

```
538 pytest · 20 frontend · ruff check · ruff format · mypy · pre-commit  — all pass
manifest.json · pyproject.toml · CARD_VERSION · uv.lock  — all 0.6.4-beta31 / 0.6.4b31
```

> **Superseded 2026-08-09 by `0.6.4-beta32`.** beta31 was adversarially reviewed
> before deployment and **not** cleared as-is; see §2.6. beta32 is beta31 plus one
> fix — the turn feasibility preflight now models the overshoot ceiling instead of
> assuming full-length pulses — and nothing else. It is refusal-side only: it can
> refuse a turn earlier, it cannot make the mower do anything beta31 would not.
> Still no `LUBA_ACCEPTANCE_PROFILE` key touched. `541 pytest · 20 frontend ·
> ruff · mypy · pre-commit` green; all four version files at
> `0.6.4-beta32` / `0.6.4b32`. The §2.2 overshoot-allowance fix is deliberately
> **not** in beta32, so the validation run measures reach against turn dynamics
> that have not just been rewritten.

## 2. Attack these first

### 2.1 The overshoot ceiling has never touched hardware

`_VIO_TURN_CONSERVATIVE_MAX_DEGREES_PER_SECOND = 60.0` (`services.py`) exists
because of **one unexplained pulse**. On 2026-08-08 Gate 5 attempt 5, turn pulse 3
swept 57.630° with 44.372° remaining — 13.258° past the target against an 18°
tolerance, finishing on 4.74° of margin. Its rate was 19.21°/write against
7.62 / 7.39 / 9.60 for the other three commands in the same segment, under the
same BLE conditions, seconds apart.

**Nobody knows why.** No terrain, slope, motor-current or battery-voltage field
exists anywhere in this codebase, so the cause is not observable from telemetry.

The obvious attack: *if the cause can produce more than 60 °/s, the ceiling is
inert and beta31's headline fix does nothing.* That attack is correct and
unanswerable from committed data. The ceiling is a bet that 21% above the highest
rate ever recorded (49.5649 °/s,
`docs/evidence-turnchar-beta19-analysis-20260804.json:117`) is enough.

### 2.2 The ceiling is not a rare backstop — it is usually the active bound

A ceiling `C` binds when `(|error| + tolerance) / C < pulse_duration`, i.e. below
`|error| = C × pulse_seconds − tolerance`. At C=60, a 1500 ms pulse and an 18°
tolerance that is **exactly 72°** — so on any normal final approach the ceiling,
not the self-calibrating estimate, decides the pulse length. Pinned by
`test_turn_final_approach_ceiling_binding_threshold`.

This was a deliberate trade (a bound that does not depend on the estimator being
right) but it has a cost: pulses are systematically shorter, so a slow turn needs
more of them against a 4-command budget. Worked example, Gate 5 geometry at the
rates actually observed: still 3 pulses. At a uniformly slow 14.5 °/s: 5 pulses,
which exceeds the budget.

> ⚠️ **CORRECTED 2026-08-09.** This section previously excused that with "though
> that geometry also fails without the ceiling (97° at 21.75°/pulse is 4.5
> pulses)". **That was wrong.** It counted pulses to *zero* error; the loop
> returns at `abs(error) <= heading_tolerance_degrees`, so it needs
> `(97 − 18) / 21.75 = 3.6` → **4 pulses, which beta30 completes.** The ceiling
> does not make an existing failure mode "slightly more likely" — it *creates*
> one in this band. Replayed through the shipped `_turn_final_approach_pulse_ms`,
> max completable junction turn within the 4-command budget at zero write
> overrun:
>
> | true rate | beta30 | beta31 | lost |
> | --------- | ------ | ------ | ---- |
> | 14.49 °/s | 104.9° | 86.2°  | 18.7° |
> | 14.90 °/s | 107.3° | 89.1°  | 18.2° |
> | 21.20 °/s | 145.1° | 131.6° | 13.5° |
>
> 14.49 and 14.90 °/s are pulses 1 and 2 of Gate 5 attempt 5 on elapsed time —
> not a hypothetical slow tier. **A 90° junction completes on beta30 and exhausts
> the budget on beta31 at those rates.** BLE write overrun partly masks it: at the
> ~260 ms median overrun of that run, beta31 clears 110°. So the outcome now
> depends on BLE latency, which is not a property anyone controls.
>
> The lever is the overshoot allowance `K`, hard-coded as `K = tolerance` — the
> strictest possible choice. `K = 2 × tolerance` costs only ~4.5° of capability
> instead of ~18°, lands the Gate 5 incident pulse at +0.52° instead of +10.34°,
> and still bounds the worst case at 49.56 °/s to 4° outside tolerance — one
> recovery pulse, far from the 90° reverse-recovery wall. **Not yet implemented**;
> it is the open item.

**If you want to attack the value:** lowering C weakens the guard and widens
margin-to-overshoot; raising it strengthens the guard and costs pulses. The two
walls are `C ≥ ~50` (any lower and the observed maximum is inside the bound) and
`C ≤ ~?` (unmeasured — nobody has established where extra pulses start failing
turns). The upper wall is the gap.

### 2.3 A new coupling to a frozen profile key

The ceiling reads `heading_tolerance_degrees`, which **is** a
`LUBA_ACCEPTANCE_PROFILE` key (18). Turn dynamics now depend on the tolerance,
which was not true before beta31. Below ~12° of tolerance the ceiling falls under
`_MIN_SCALED_TURN_PULSE_MS` (400 ms) and **the floor wins, so the anti-overshoot
guarantee does not hold there**. That resolution is deliberate — an overshoot is
recoverable by the next pulse, a pulse too short to actuate makes no progress and
walks the turn into `no_heading_progress` with its budget spent — and it is
surfaced as `ceiling_below_actuation_floor` in the result rather than resolved
silently. Pinned by `test_turn_final_approach_ceiling_vs_actuation_floor`.

### 2.4 Four segments is unvalidated

**No run has ever executed a third segment.** The VIO forward-heading offset is
refreshed only from linear travel and is never re-derived across a turn, so
cumulative cross-track error past segment 2 is unmeasured. The one datum pointing
at this is unfavourable: attempt 5's segment 2 produced the worst landing of the
four (0.1449 m against a 0.15 m tolerance). Extrapolating that trend to segment 4
is exactly what the validation run must check.

### 2.5 `vio_realign_threshold_degrees` is now partly inert

The mid-drive realignment used to dispatch a turn for aim errors in
(threshold 15, tolerance 18], which returned `target_heading_reached` at its entry
check without sending a command — burning a realignment slot for nothing. beta31
skips it. Mower behaviour is unchanged, but the **effective trigger is now the
tolerance**, and the threshold parameter is dead in the gap between them. To make
it live again, lower the tolerance below it.

### 2.6 Found in adversarial review, 2026-08-09 — three more

Attacking the branch before deploying it turned up four things §2.1–2.5 missed.
One is fixed in beta32; the rest are open.

1. **FIXED in beta32 — the preflight did not model the ceiling.**
   `_vio_turn_budget_feasibility` assumed every command ran a full
   `turn_pulse_duration_ms`, so it admitted turns beta31's shortened pulses
   cannot finish: at a 4-command budget the two models disagree over
   **100–117°**, and the guard's documented fail-closed property — the whole
   reason it exists after Gate 4 — was broken from the planning side. It now
   replays the executor's own policy by calling the same
   `_turn_final_approach_pulse_ms` the turn loop calls, so the two cannot drift
   again. A 90° junction reads 4 commands, not 3: feasible at **exactly** the
   budget, with no margin left.

2. **OPEN — the ceiling's guarantee is written in commanded milliseconds, but the
   mower rotates for the delivered window.** This is the same
   commanded-vs-delivered error item 3 fixes in the estimator, left unfixed in
   the ceiling ten lines away. `ceiling_ms` becomes `duration_seconds` for
   `_motion_refresh_window`, whose `elapsed_ms` overran nominal by
   +30 / +260 / +543 ms on the three Gate 5 pulses (+975 ms in the beta19
   characterization). For the pulse-3 geometry the guarantee then holds only to
   58.3 / 48.0 / 39.4 °/s — and **48.0 and 39.4 are below the 49.56 °/s the
   hardware has actually produced.** Two of the three pulses on the run that
   motivated the ceiling were already past that point. The stated 21% margin is
   not there.

3. **OPEN — `_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND = 16.5` is no longer a
   floor.** Gate 5 attempt 5 measured 14.905 and 14.490 °/s against delivered
   windows, ~12% below it; they were invisible on the old denominator. It has
   deliberately **not** been lowered: at 14.4 °/s the ceiling-aware model needs 5
   commands for a 90° junction against a budget of 4, so **a truthful rate floor
   and L-path junctions cannot both hold at the current `vio_turn_max_commands`.**
   That is a capability decision, not a constant edit. Pinned by
   `test_conservative_rate_floor_is_known_optimistic`. Fixing §2.2's `K` resolves
   it: at `K = 2 × tolerance` a 90° junction completes at 14.49 °/s.

4. **OPEN — the ceiling changes where turns land, into a tighter gate.** It
   converges from above and stops on first entry into tolerance, so landing error
   clusters just inside 18° instead of near 0 (replay: mean |landing| ~13° vs
   ~6°). The post-turn alignment gate is
   `min(heading_tolerance, vio_realign_threshold)` = **15**, tighter than the
   turn's own tolerance, so expect materially more post-turn corrections — each
   spending one of three realignment slots, with `post_turn_realign_incomplete`
   aborting the segment. A larger residual heading error entering each linear leg
   also means more cross-track error, on exactly the axis §2.4 says to measure.
   **The two beta31 changes work against each other here.**

Also, minor: §2.5 says "mower behaviour is unchanged". True of motion, not of
gating — those skipped dispatches also re-verified blade-safe, `work_mode`, VIO
feed liveness and BLE transport before returning. And no test covers multi-pulse
convergence against `vio_turn_max_commands`; every ceiling test is single-pulse
arithmetic, which is why item 1 above survived 538 green tests.

## 3. What changed, with citations

| # | change | file | profile key? |
| - | ------ | ---- | ------------ |
| 1 | `REAL_CLICK_TO_GO_SEGMENT_LIMIT` 2 → 4 | `manual_motion.py:24` | no |
| 1 | `MAX_REAL_SEGMENTS` 2 → 4 | card js `:2` | no |
| 2 | `_VIO_TURN_CONSERVATIVE_MAX_DEGREES_PER_SECOND = 60.0` + cap in `_turn_final_approach_pulse_ms` | `services.py` | no |
| 3 | rate estimator divides by measured `elapsed_ms`, not commanded `pulse_ms` | `services.py`, the `heading_went_fresh` block | no |
| 4 | `motion_refresh_commands_sent` folds in turn / post-turn / realignment refreshes | `services.py`, 3 sites | no |
| 5 | realignment skips no-op dispatches | `services.py` | no |

The schema `vol.Range` and the runtime re-check both reference
`REAL_CLICK_TO_GO_SEGMENT_LIMIT` (`services.py:1054`, `:11490-11497`), so they
follow the constant automatically.

**Items 2 and 3 must ship together.** The denominator fix alone makes overshoot
*worse*: measured `elapsed_ms` exceeds nominal (2043 / 1530 / 1760 against 1500),
so a correct denominator yields a lower rate, a larger `needed_ms`, less
shortening and longer pulses.

## 4. Evidence you should read before disputing anything

- `docs/evidence-gate5-attempt5-segment1-raw-20260808.json` — the recovered
  per-command record. Read `v1_errors_corrected` at the bottom: the first version
  of that analysis contained a tautological "control" and a back-solved free
  parameter, both caught by adversarial audit and withdrawn. Treat the derived
  sections as audited, the verbatim sections as ground truth.
- `docs/turn-rate-variance-and-reach-analysis-20260808.md` — ⚠️ **append-ordered.**
  Its §B1 reach numbers are **wrong** and corrected further down. The header
  routes you to the three correction blocks; read those first.
- `docs/turn-rate-variance-completeness-critique-20260808.md` — the completeness
  critic, which found the stop-latency mis-attribution and the `_vio_turn_probe`
  precedent.

Investigation provenance: 6 finder angles, each attacked by an independent
adversarial verifier, plus a completeness critic. 13 agents, final tally
**110 CONFIRMED / 19 REFUTED / 1 UNVERIFIABLE**.

## 5. Still open, documented, not blocking

- **Pulse 2's refresh window is unexplained.** Predicted 1254.135 ms against an
  actual 1530.326 — 276 ms unaccounted for — and a third refresh write the code
  requires (`t = 1454 < 1500` deadline) never fired, with no `refresh_error`
  recorded. Awkward for §2.1's argument, since the same pulse is anomalous in two
  unrelated ways.
- **Pulse 3's cause.** See §2.1.
- **Turn stops use `Priority.NORMAL`; linear stops use `Priority.EMERGENCY`**
  (`services.py:5719-5720`, `:3307`). The code carries a committed live
  observation that a normal-priority stop took 1392.7 ms while the mower continued
  past its target (`:3301-3306`). No comment justifies the asymmetry. Untouched by
  beta31.
- **The mid-drive re-aim guard tests `command_index < max_linear_commands`**, not
  `effective_linear_ceiling` (`services.py:11212` region). Harmless today because
  `max_linear_pulse_ceiling` is `null`, but it must be fixed **before** anyone
  enables loop-to-tolerance, or cross-track correction silently stops after pulse
  3 while the mower keeps driving.

## 6. Deploy and validate

**Deploy** (`docs/deploy-runbook-p0.md`): the card is served from **two** paths on
the host — deploy to both and bump the Lovelace resource key, or the browser
silently loads the stale card. Deploy motion-disabled and read back
`real_motion_allowed: false` before anything else.

**Validation run — one daylight window, freshly authorized.** Preconditions:
daylight (the `vio_active` gate keys off `turn_mode == "vio"` unconditionally, so a
closed-loop segment cannot run after dark), RTK Fix, `tracked_features` ≳ 70,
blades off, and a charged battery — the mower was left at ~22% and needs docking
first. Arm immediately before, disarm immediately after, verify both.

### The run is scripted — `scripts/beta32_validation_run.py`

Prepared and rehearsed 2026-08-09 against the live host. **Safe by default:**
without `--arm` it never opens the motion gate and never sends a movement
command, so preview it as often as you like.

```sh
set -a && source .env && set +a
.venv/bin/python scripts/beta32_validation_run.py          # preflight + dry run
.venv/bin/python scripts/beta32_validation_run.py --arm    # the real run
```

It does, in order: preflight (8 hard gates, printed pass/fail) → build a
4-segment path from the **live** position that stays inside the mapped polygon →
dry-run validate and print junction feasibility → **arm and verify
`real_motion_allowed: true`** → execute → **write the complete response JSON to
`docs/evidence-beta32-4segment-<stamp>.json` before parsing a single field** →
print the §6 record → **disarm in a `finally` and verify it closed.** A crash, a
timeout or a Ctrl-C cannot leave the gate open. If preflight fails it refuses to
arm.

Rehearsed off-mower: the path builder finds a valid in-polygon path from **all
203** grid start points inside Backyard Right; the full acceptance profile is
schema-accepted and echoes back key-for-key identical; all three 60° junctions
preflight feasible at 3 of 4 commands.

**Morning gate order** — the three that were failing overnight are the only ones
outstanding: daylight (`tracked_features ≥ 70`; it was 0 and brightness `dark`),
`position_valid_for_motion` (mower was docked at `CHARGE_ON`), and
`work_mode` in `{MODE_READY, MODE_PAUSE}`.

⚠️ **Watch the work mode before arming.** Between 01:15 and 01:30 EDT the mower
cycled `mode_returning` ↔ `mode_ready` ten times while sitting on the dock, with
position shifting a few cm. `mode_returning` is not an allowed motion mode and
the executor re-checks it *between* commands, so a mower still doing this will
abort a segment with `aborted_unsafe_mode` partway through. Confirm it has
settled before arming. (Battery is fine and was never the problem — 58% and
rising; the overnight swings were two real mow cycles, not telemetry noise.)

**Junction angles: keep every turn in the 45–70° band.** This is a deliberate
constraint from §2.2/§2.6, not caution for its own sake. Below 72° the ceiling is
the active bound from the first pulse of every turn, so the path gives *maximum*
exposure to the new code — `bounded_by_max_rate_ceiling` on essentially every
turn pulse — while staying clear of the 86–100° band where the budget and the
rate floor are in doubt. A 90° L-path is the geometry you eventually want and is
**not** the geometry to debut on; it sits exactly on the contested edge.

Run a **4-segment path with a turn at each junction**, and record:

1. per-segment landing error against `waypoint_tolerance` 0.15
2. **whether error grows with segment index** — the specific risk from §2.4
3. `turn_commands_sent` **broken down by phase**, not as one number (conflating
   them is what produced the refuted "budget exhausted" claim)
4. every `final_approach` block — confirm `bounded_by_max_rate_ceiling` fires on
   final approaches and `cruising_full_pulse_fits` on large turns
5. `heading_error_after` per turn pulse — the direct overshoot measure, and the
   only way to tell whether the ceiling actually worked
6. zero `target_requires_reverse_recovery`, zero `vio_realign_budget_exhausted`

**Save the complete response JSON before writing any summary prose.** The single
most expensive gap in this whole investigation was that attempt 5's
`command_results` existed only in a browser pane and had to be recovered by hand
days later. Commit it as `docs/evidence-beta31-4segment-<date>.json`.

## 7. Ground rules that outlive this handover

- Repositories owned by `mikey0000` are **read-only**. No pushes, comments, issues
  or PRs. Authorized pushes go only to the `Chorty` fork.
- Changing any `LUBA_ACCEPTANCE_PROFILE` key
  (`custom_components/mammotion/www/mammotion-custom-path-card.js:31-61`)
  un-accepts the profile and obligates the §4 checklist in
  `docs/gate4-repass-20260805.md` plus a fresh Gate 5.
- Verify with **per-item records, not aggregates**. Both major errors corrected
  during this investigation came from reading a cumulative field as per-pulse and
  from trusting a prose summary over the raw array.
- A zero from a log grep proves nothing if the logging is off. Verify via an
  effect the system records independently.
