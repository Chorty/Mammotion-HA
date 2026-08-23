# Phase 2 gap reconciliation — a fifth gap found by testing, not guessing

**2026-08-23. Offline only. No dispatch, no mower run, no
`LUBA_ACCEPTANCE_PROFILE` change.** Closes the four reconciliation gaps
recorded in `docs/phase2-continuous-motion-design-20260823.md`, and records a
fifth the four did not anticipate — found by replaying the Phase 0 controller
against real banked telemetry rather than assuming the fix was a constant swap.

## Gaps 1 and 3 — closed as expected

- **`nominal_speed_mps`** 0.28 → **0.2482**, the frozen `k_lin`-derived speed
  from 16 steady-state steps across three straight captures
  (`docs/frozen-prediction-constants-20260822.json`).
- **`angular_speed_per_heading_degree`** is now documented in-line as a
  steering gain, explicitly distinct from the `w = k_ang * angular_speed`
  relationship refuted this week. No value changed.

## Gap 4 — confirmed, not assumed

Read the full 357-line module (only 200 had been read for the design pass).
**Confirmed: no live keep-out or area-edge geometry check exists anywhere in
`continuous_controller.py`.** `route.contained` is a caller-supplied bool,
never re-derived. This is now documented on `ContinuousRoute` itself, pointing
at the same corridor-freezing discipline (`scripts/freeze_phase1_corridors.py`,
`scripts/scan_contained_bearings.py`) already used for Phase 1 captures.

## Gap 2 — NOT a constant swap. `max_refresh_age_s` alone cannot detect a stall.

The design doc flagged `max_refresh_age_s` (1.20 s) as loose against the
registered `3R = 600 ms` BLE-stall rule and proposed tightening it. **Building
`scripts/replay_continuous_controller_against_capture.py` to test that fix
against `docs/evidence-8s-continuous-window-20260822T233000Z.json` showed
tightening the number is not sufficient — the check has the wrong SHAPE.**

`refresh_age_s` is "time since the most recent completion, sampled right now."
That capture has two real stalls (664 ms and 810 ms gaps between consecutive
refresh completions — confirmed directly from `refresh_write_completions_elapsed_ms`).
The replay caught the first. It **missed the second entirely**, even at the
tightened 0.60 s bound:

```
t=3.913s  age=0.000s  gap=0.810s  <-- the 810 ms stall is invisible to `age`
```

A fast 106 ms recovery write completed at 3912.8 ms, essentially simultaneous
with the next ~1 Hz decision at 3913.2 ms. `refresh_age_s` read ~0 at the exact
instant checked. **That 810 ms stall is the one that produced the largest
prediction error in the whole corpus, 0.1418 m**
(`docs/prediction-model-holds-out-of-sample-20260823.md`). A point-sampled
staleness check is structurally blind to a stall that resolves between two
decision instants — no threshold on that field fixes it, because the field is
answering the wrong question.

### The fix: track the worst gap, not the most recent timestamp

Added `refresh_max_gap_since_last_decision_s` to `ContinuousObservation` — a
running max over consecutive refresh-completion gaps the caller must track
between decisions, not a point sample. Added `max_refresh_gap_s = 0.60` to
`ContinuousControllerConfig`, matching the registered `3R` rule exactly, and a
new fail-closed reason `refresh_cadence_stalled`. `max_refresh_age_s` is kept
as a distinct, looser check ("is the refresh loop alive at all right now") —
the two answer different questions and both are needed, which resolves the
ambiguity the design doc flagged but could not settle without this test.

### Replayed against both banked continuous captures, before and after

| capture | before (no gap check) | after (gap check, 0.60 s) |
| --- | --- | --- |
| 8 s straight (has 2 real stalls) | never stops | stops at **both**: t=2.270s (`refresh_age_exceeded`, 0.661 s) and t=3.913s (`refresh_cadence_stalled`, 0.810 s -- the one `age` alone misses) |
| guard run (no real stall) | never stops | never stops -- correct negative control, no false positive |

## What this changes for the design decisions already made

Nothing about the four architecture decisions changes. This is entirely inside
"extend the bounded-window pattern" and "stop safely on a BLE stall" -- it is
the mechanism that makes the stop-safely decision actually work against real
telemetry, not a new decision.

⚠️ **One consequence for whoever builds the executor**: it must track a running
max of refresh-completion gaps between decisions and pass it in, not just the
timestamp of the most recent completion. That is more state than the module
needed before today, and it is why this belongs in the design record rather
than being silently absorbed into a constant change.

## Verification

827 pytest (up from 824), ruff, ruff format, mypy 29 files, 91 frontend, ten
pre-commit hooks, `check_accepted_profile` ACCEPTED. Three tests added or
updated in `test_continuous_controller.py`: the fault matrix gains
`refresh_cadence_stalled`, one test reproduces the exact age-blind-spot bug
directly, one confirms a sub-threshold gap does not stop. Three existing tests
had hardcoded expectations derived from the old `nominal_speed_mps = 0.28` and
were updated to the reconciled value, not loosened.
