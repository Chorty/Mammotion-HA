# Resume prompt — Mammotion-HA, post-beta100

Read `CLAUDE.md` in full first — it is now **321 lines** (it was 3,258 on
2026-09-03) and all of it is live. The history is verbatim in
`docs/history/CLAUDE-archive-through-beta100.md`; **do not act on anything in
there**, and do not re-import it into `CLAUDE.md`.

Then read `docs/NEXT-PLAN-20260903.md` for background only — **§1a, §1c and §2a
are DONE**, and §0's framing has since been superseded by standing decisions 5–7,
which formally closed Phase 2 and OTA.

## Live state (verify before acting — true at 2026-09-04 end of session)

- Host runs **0.6.4-beta100**, backend `chorty-0.8.12.post4`, deployed and
  **browser-verified**. Gate disarmed; mower docked at 100%, BLE live, RTK Fix.
- ⚠️ **THE TREE IS AHEAD OF THE HOST ON THE CARD.** Tree card md5 `191ec3d5`,
  host `77971e3f` — the 0.58 m advisory rewording (commit `958a99ff`) has not
  shipped. It is cosmetic operator-facing text, no behaviour change.
- `main == origin/main` at `958a99ff`. Working tree carries only the operator's
  uncommitted `docs/agora_outbound_audio_probe.md` and an untracked `.vscode/`.
  🚨 **Stage by explicit path — `git add -A` has swept that file in twice.**
- Suite: **1,019 tests in ~24 s**, 54% coverage (5,783 statements missed).
  It was 138 s until three real waits were stripped on 2026-09-04.

## Framing — read this before choosing work

**Standing decision 2 is the objective: click-to-go reliable enough to trust
without watching.** Reach is CLOSED at 6.0 m with flat landings; **feasibility is
proven and RELIABILITY is not.** Mower time goes to landing populations.

🛑 **Phase 2 and OTA are CLOSED, not parked** (standing decisions 5 and 6). That
closes criterion 2a, τ, the dead-time question and the step-response probe as a
research instrument. **Do not propose 2a repeats, longer steps, `step_ms` /
`_STEP_RESPONSE_MAX_TOTAL_MS` raises, Rule D, or night runs.**

📏 **Standing decision 7: reliability statistics use the beta57+ epoch only.**
Pooling all 81 banked segments gives authoritative-looking rates (89% under
1.0 m, 43% at 3.0–3.9 m, 44% for segment 2+ at ≥1.5 m) that span four material
control-law changes. **Shape, never a rate.**

## Do, in this order

### 1. Split `test_map_task_visibility.py` — offline, no mower

13,207 lines and 476 tests (**47% of the suite**) under a name that describes
~3% of its contents. It is the default dumping ground; the 90 s test hid there
for months, and `test_long_segment_reach.py` already reaches across to import its
fixtures, which is itself the symptom.

Split by what is under test, so each file's name predicts its contents:

| new file | takes |
| --- | --- |
| `test_service_schema_contracts.py` | the 58-instance AST handler sweep, schema-default parametrisation, entity-service-schema checks |
| `test_vector_segment_execution.py` | segment executor, pulses, re-aim, post-turn correction |
| `test_turn_primitives.py` | final-approach bounds, actuation floor, turn budget |
| `test_ble_transport_health.py` | link liveness, queue settle, recovery, cooldown |
| `test_night_motion.py` | night mirror, night refusals, night segment caps |
| `test_map_task_visibility.py` | **what the name means** — map task/zone visibility |

Move shared fixtures (`_pulse_coordinator` and friends) to
`tests/components/mammotion/conftest.py`.

🚨 **A test-file split can silently delete coverage. Four guard rails, all
required:**
1. **Move only — zero edits to test bodies** in the split commit. Anything that
   needs changing goes in a separate follow-up commit.
2. **Test count identical: 1,019 before and after.** A pure move cannot change it.
3. **Coverage byte-identical: 54%, 5,783 statements missed.**
4. **Collect-only diff is empty.** Capture before and after and diff:
   ```sh
   .venv/bin/python -m pytest tests --collect-only -q | grep :: \
     | sed 's/.*:://' | sort > /tmp/nodes_before.txt
   ```
   Compare with the same command after. **A non-empty diff means a test was lost
   or renamed — stop and fix it before committing.**

### 2. Predeclare the L-path reliability series — offline, no mower

`docs/predeclared-clicktopath-reliability-4m-20260903.md` covers **segment 1,
aligned start** — the easier half. The banked shape says the risk is a long leg
that does **not** start fresh (44% for segment 2+ at ≥1.5 m against 73% for
segment 1), and that is what real click-to-path does.

⚠️ **State the confound honestly in the predeclaration**: "segment 2+" mixes
junction turns with collinear continuations (Route B's are collinear); what they
share is inheriting the previous leg's cross-track error, not necessarily a turn.

Write it in the same shape as the 4.0 m one: two segments with a **genuine
direction change**, second leg ~2.0 m, criteria and abort rule fixed before any
dispatch, `docs/accepted-profile.json` verbatim and verified key-by-key.
🔑 **Do not dispatch it before the 4.0 m series** — the 4.0 m run is the
same-build baseline this one gets compared against.

### 3. Release + deploy — carries the card fix from step 1's release

Use the `release-and-deploy` skill. Motion-disabled, gate verified disarmed from
the live API **and** RAW before and after.
⚠️ **Verify with a DISCRIMINATING check, not a version string.** For this release
the card text is the change, so the check is the rendered advisory wording —
which means it needs the operator's browser. Bytes on the host are necessary and
not sufficient.

### 4. The 4.0 m reliability series — daylight, operator present

`docs/predeclared-clicktopath-reliability-4m-20260903.md`. It authorizes nothing
on its own; every dispatch needs its own go/no-go immediately before.

Non-negotiable per run: fresh corridor scan **against the map** (the containment
gate measures the polygon you supply, not the mowing area), daylight throughout,
`docs/accepted-profile.json` verbatim and verified key-by-key, gate disarmed and
verified afterwards from live API **and** RAW.

🚨 **The card auto-splits above 3.85 m** — dispatch
`raw_pymammotion_execute_vector_segment` directly or you measure the splitter.
🚨 **A run that stops safely on a named refusal is a FAIL**, not a smaller number.
⚠️ **n = 5 is a screen, not a certification** — 5/5 bounds the true success rate
at only ~55% with 95% confidence. Do not let a pass become "trustworthy
unwatched".

### 5. Then the L-path series, if 4 completes

## Boundaries

- No motion without explicit per-run operator go/no-go immediately before dispatch.
- Never push to `mikey0000/*`; pass `-R Chorty/Mammotion-HA` to every `gh` command.
- Do not relax the safety discipline in `CLAUDE.md` → "How this project works".
  It is separate from the narrative that was archived, and it caught a wrong
  bound shape on 2026-09-03 before any code was written.
- Verify before acting on any claim in `CLAUDE.md` — grep the tree first.

## Model routing

Opus for anything producing a claim: the L-path predeclaration, evidence files,
telemetry interpretation, supervising real motion. Sonnet for the mechanical half
of the test split, the deploy, and CI runs — but **the four guard rails are the
gate**, and an Opus pass should confirm the collect-only diff is empty before the
split is committed. Separate the test RUN from the diagnosis: have the cheap pass
report raw output only, then switch to Opus in the same session if it fails.
