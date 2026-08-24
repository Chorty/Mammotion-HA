# Handoff to Codex: independent adjudication of the Phase 2 sign fix

**2026-08-24. Prepared, not yet run.** Codex CLI confirmed ready on this
machine (`codex-cli 0.144.5`, ChatGPT auth active, ready: true). This doc is
self-contained — a fresh session (or Codex itself) should not need anything
else in this repo's history to act on it, beyond the files it names.

## Why Codex, not another Claude pass

Precedent: `docs/codex-adjudication-20260823.md`. This project has used Codex
once before as an independent adjudicator — a different model, no stake in
the answer, pointed at raw evidence, explicitly told a prior summary had
already contained false claims, and deliberately NOT told which answer was
preferred. That adjudication refuted one of my central arguments and was
recorded as such. Same shape of ask here: two Claude passes (Sonnet, then
Opus) already diagnosed and fixed a safety-critical steering-sign bug in the
same session that found it. That's exactly the condition under which this
project's own history says a second, differently-motivated check is worth
having before the next physical run.

## What happened today (2026-08-24), in order

1. First-ever physical run of `mammotion.continuous_motion_window` (the
   Phase 2 executor, built 2026-08-23, never run before today). Diverged and
   safely hard-aborted on the 0.30 m cross-track bound. Raw run + an offline
   replay through the pure controller (0 mismatches) + scoring against the
   pass criteria: `docs/evidence-phase2-first-physical-run-20260824.json`.
   Criteria: `docs/phase2-gate-readiness-20260823.md`. Verdict: FAIL.
2. Diagnosed and fixed two defects, committed as `4483dd70` ("fix: Phase 2
   continuous controller steering sign inversion + stale-heading gate") and
   documented in `CLAUDE.md`'s "Current build" section (search
   `FIRST PHYSICAL RUN OF THE PHASE 2 EXECUTOR, 2026-08-24`) and
   `docs/NEXT-SESSION.md` (same anchor text).
3. **Neither fix has been physically re-tested. Nothing has been deployed —
   the host still runs beta73 with the OLD, unfixed controller.**

## The specific claim to adjudicate

`custom_components/mammotion/continuous_controller.py` — the steering law
converts a heading error into a commanded `angular_speed`. The diagnosis:
the ORIGINAL code assumed positive commanded angular increases map-frame
`course_heading_degrees`; it actually decreases it, because
`course_heading_degrees = 90.13 − toward` (`services.py:7323`,
`_continuous_course_heading`) is a REFLECTION, not an offset, and positive
commanded angular is separately established to INCREASE `toward`
(`docs/toward-tracks-in-place-rotation-20260812.md`, 2026-08-12, in-place
pivots with VIO off: angular +500 → toward +99.55°, angular −500 → toward
−61.43°). So the original `angular = +K × (desired_course − actual_course)`
law was positive feedback, not correction — and that is what today's
divergence looks like (heading error grew 46.6° → 48.3° → 77.4° under a
continuously saturated "correction").

The fix (already applied, in `git show 4483dd70 -- custom_components/mammotion/continuous_controller.py`):
sign flipped at the final command, not the gain or the error term, so
`heading_error_degrees`/`cross_track_m`/`along_track_m` reporting is
unchanged.

Verification already done, twice, by two different Claude passes: checked
against six banked real hardware captures across both command signs
(today's run, the certified Phase 1b arc, the arc120 out-of-sample capture,
two 2026-08-12 arc-sweep points, and the 2026-08-12 stationary night pivot)
— reported as zero contradictions in either pass.

## What to ask Codex to do

Point it at:
- `custom_components/mammotion/continuous_controller.py` (current, post-fix)
- `custom_components/mammotion/services.py:7323` (`_continuous_course_heading`)
- `docs/evidence-phase2-first-physical-run-20260824.json` (today's raw run)
- `docs/toward-tracks-in-place-rotation-20260812.md`
- `docs/arcs-work-20260812.md`
- `docs/phase1b-go-20260823.md` and its raw evidence (for the certified arc)
- `docs/prediction-model-holds-out-of-sample-20260823.md` (arc120)
- `git show 4483dd70` for the actual diff
- `tests/components/mammotion/test_continuous_controller.py` and
  `test_continuous_motion_window.py` for what was added

Ask it, **without telling it the diagnosis is believed correct**:

1. Independently re-derive the sign relationship between commanded
   `angular_speed` and `course_heading_degrees` from the raw evidence files
   directly — not from the prose in this handoff or in CLAUDE.md. Does it
   agree the original code had the sign backwards, and does it agree the fix
   in `4483dd70` corrects it without disturbing anything else (cross-track,
   along-track, desired-course reporting)?
2. Check the six-capture verification claim itself — recompute at least the
   Phase 1b certified arc and one other independently rather than trusting
   the "zero contradictions" summary.
3. Read the new `opening_alignment_feasible` preflight gate
   (`continuous_controller.py`, search `alignment_feasibility`, and its
   wiring in `services.py` around `opening_alignment_feasible`) and check
   whether its stated limits (today's run: `total_time_s` 6.434s,
   `total_cross_track_m` 0.666 m, refused) hold up under its own stated
   model, and whether the physics it uses (turn time + the stale-heading
   fix's 0.15 m blind-travel floor) are combined correctly.
4. Anything else that looks wrong, unverified, or overclaimed — this
   project's own standing rule is that a wrong "it's fixed" claim here has
   real physical consequences, so a genuine, well-argued dissent is a more
   useful outcome than confirmation.

## Hard constraints for whoever runs this

- Offline only. No HA/BLE calls, no `HA_URL`/`HA_TOKEN`, nothing that
  reaches the physical mower. Codex is not authorized to run, or propose
  triggering, a physical test — adjudication only.
- Do not `git commit`/`push` on Codex's own initiative; report findings back
  to the operator.
- The motion gate is disarmed as of 2026-08-24 (confirmed via raw on-disk
  `core.config_entries` storage, not just the live API) and should stay that
  way unless the operator is present and explicitly re-arms it for a
  supervised run.

## What happens if Codex confirms

Per `CLAUDE.md`'s own note on this: cut a release, deploy motion-disabled,
dry-run-verify `continuous_motion_window` against live coordinator state
with a fresh corridor scan, then — only with the operator physically present
and giving explicit per-run authorization — attempt the next physical run.
That sequencing is unchanged by this handoff; Codex's job is the
adjudication step before it, not the run itself.

## What happens if Codex disagrees

Record the dissent the same way `docs/codex-adjudication-20260823.md` did:
verbatim, re-derived locally where possible, and do not physically re-test
until it's resolved.
