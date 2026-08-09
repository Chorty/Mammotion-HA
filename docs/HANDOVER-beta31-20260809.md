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
which exceeds the budget — **though that geometry also fails without the ceiling**
(97° at 21.75°/pulse is 4.5 pulses). The ceiling makes an existing failure mode
slightly more likely; it does not create one.

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
