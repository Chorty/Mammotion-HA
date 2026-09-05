# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Start here

1. **"Current build"** below — what is released and installed, and the live state.
2. **"Standing decisions"** — the operator's scope calls. They override anything
   older, including anything in the archive.
3. **"Operating facts"** — the measured constants and the traps that keep biting.
4. **"How this project works"** — the discipline that makes results trustworthy.

🗄️ **Everything historical now lives in
`docs/history/CLAUDE-archive-through-beta100.md`.** This file was 3,258 lines, of
which ~250 were live instruction; the rest was a rolling log that loaded into
context every session. **Nothing was deleted** — the archive is verbatim.

🚨 **Verify before you act on any claim in this file.** On 2026-08-14 a paragraph
here described an already-shipped fix as "NOT implemented" for thirteen betas and
cost a session real work; a reach bullet went stale twice under a "do not
re-derive" heading. `scripts/check_doc_symbols.py` (a pre-commit hook) fails when
these docs name code that does not exist — but it checks *names*, not whether the
prose around them is still true. **One grep against the tree beats this file.**

---

## Current build: beta103 (deployed 2026-09-05; backend `chorty-0.8.12.post4`)

✅ **Bytes verified end to end** — 50/50 files byte-identical, card md5
`1b3a404d` at both serving paths, Lovelace `?v=0.6.4-beta103&build=1b3a404d`,
backend read from inside the container, config entry present and not
`disabled_by`, 133 entities loaded, API back in 31 s.
✅ **beta102 and beta101 were both browser-confirmed** (card footer
`v0.6.4-beta102`, and the 0.58 m advisory reads as a calibration note).
⚠️ beta103's card text is unchanged from beta102, so nothing new needs a browser
check. Record: `docs/deploy-runbook-p0.md`.

**What beta103 shipped** (`4f908b04`): 🐛 **the travel guard tripped at ZERO
travel on every real run.** It took its sequence baseline from
`handle.latest_position_sample` *after* opening the position stream, so any
payload arriving in that gap failed the first contiguity check. Found on
hardware: two dispatches died at 344 ms and 273 ms having sent 1 of 11 refresh
writes, travelling ~0.09 m against a 0.40 m bound. **`max_travel_m` was unusable
in practice.** The first sample now seeds the baseline, as `last_position`
already did. Also splits `position_samples_dropped` out of
`position_sequence_gap` — the two shared one string, which cost two dispatches
spent guessing which had fired.
⚠️ **The old harness modelled `latest_position_sample = None`** — baseline 0
against a stream starting at 1, the one arrangement where the bug cannot appear.
That is why 1053 passing tests said nothing about it.

**What beta102 shipped** (`09e4335e`, the three orientation defects from
`docs/findings-clicktopath-reliability-4m-20260904.md`, all offline):
- 🔴 **Device fault codes now reach the operator.** The live push is kept
  instead of parsed-and-discarded, every message leads with the numeric code,
  a bundled 449-code offline table (from the vendor app's own asset) backs up a
  blank cloud row, and all ten log slots are exposed as sensor attributes
  instead of only slot 0. 🏆 **Confirmed live**: `sensor.*_last_error` now reads
  `1309 (navigation): Heading calibration failed because the robot starts
  working at the perimeter of a task area — Please control the robot into a task
  area and retry`, and `logged_faults` shows **all five 1309 events at the exact
  timestamps the vendor app showed**. 🔑 **They were on the host the whole time**
  — the old accessor read slot 0 only and omitted the number from the message.
- 🔑 **`runtime_state.map_facing` is the ONE place to ask for facing.** It uses
  the reflection and nothing else, and separates corroboration from freshness:
  `motion_confirmed` (the bearing of the leg the mower actually drove agrees
  with its published facing — the only value that sets `safe_to_aim_dispatch`),
  `corroborated_not_motion_confirmed`, or `unknown`. `current_orientation`
  keeps its old meaning and now says so in the payload.
- 🚨 **`start_geometry` replaces the circular alignment check.** Filled in only
  from the VIO calibration drive's independently measured
  `map_motion_heading_degrees`; `None`-with-a-reason everywhere else. Never
  from `toward`.
- `_TOWARD_MIRROR_DEGREES` is one constant again (was three copies of 90.13).

⚠️ `docs/accepted-profile.json` is UNTOUCHED. No Gate 5 is owed.

**What beta101 shipped** (carried from `958a99ff`, cut same-day as beta100 but
not deployed until 2026-09-04):
- ✏️ **Re-derived the 0.58 m advisory's interpretation, arithmetic unchanged.**
  Both the backend docstring and the card previously asserted it "is why the
  measured-good regime is ~0.8 m and why 3.0 m legs miss" — refuted, since
  reach is closed at 6.0 m and landing does not degrade with distance
  (0.1023 / 0.1015 / 0.1144 m at 4 / 5 / 6 m). The ~0.8 m rule was an artifact
  of the pre-beta57 angle-triggered re-aim. The bound is a **guarantee** bound,
  not a prediction, and since it fires on essentially every useful leg the
  card wording now calibrates rather than alarms.

**What beta100 shipped** (deployed 2026-09-03, browser-confirmed then):
- 🛡️ **E-VIO refuses heading-frame discontinuities** (`vio_heading_discontinuity`).
  `vio_state` checks liveness, **not continuity**; any interval above
  `_STEP_RESPONSE_VIO_MAX_PLAUSIBLE_RATE_DEG_PER_S = 30.0` refuses the whole run.
  Predeclared first: `docs/predeclared-vio-heading-continuity-guard-20260903.md`,
  outcome in `docs/findings-vio-heading-continuity-guard-20260903.md`.
  ⚠️ **Deployed but unexercised on hardware** — it only populates on a real
  step-response run, and none is planned (that line serves closed work).
- 🐛 `raw_pymammotion_motion_probe`'s clock-bound corridor omitted the stop
  overshoot — the sibling of a beta98 fix that was never propagated.
- 🧹 Stale speed constants retired from the operator-facing service text.

⚠️ **Live state was true at 2026-09-05 ~16:30 UTC. Requery HA and the mower
before acting on it.** Mower **docked and charging**, 80%, RTK Fix, VIO 80
features, BLE **-56 dBm**. Gate **disarmed and verified from live API AND RAW**
after the run. ✏️ **The gate was armed twice today and BOTH were the operator**
arming it deliberately to use the click-to-go card. **Zero defect sightings.**
I recorded them as occurrences seven and eight before they corrected me.
🔑 **Ask before attributing an armed gate to the defect — a card session is the
ordinary explanation, and the two look identical in the API.** The standing
count below stays at six.

🏆 **`lawn_mower.dock` worked FIRST TRY from 8.7 m out, in ~75 s, with no
`1309`** — the same command that failed twice last night. Corroborates the
2026-09-04 §6.6 diagnosis that those failures were orientation. ⚠️ n = 1 and the
conditions differ in several ways; corroboration, not proof.

🏆 **The facing question is ANSWERED and the model works on the ground.**
`map_facing` predicted the driven direction to a **mean 1.382°** over four
scored legs (max 1.961°) against a predeclared 10° bar — **PASS at 4 of 4**.
Record: `docs/findings-facing-prediction-20260905.md` +
`docs/evidence-facing-prediction-20260905.json`.
🚨 **But read §0 of that findings doc before quoting it**: every leg ran within
~4° of the heading where the mirror and the additive offset CROSS, so the series
says nothing about which model is right. That case still rests entirely on the
banked 43-pulse data.

🚨 **The 4.0 m reliability series ran 2026-09-04 and is a determined FAIL at
n = 4** (3 `target_reached` at 0.1140 / 0.1310 / 0.1354 m, 1 refusal at 0.1656 m).
Full record: `docs/findings-clicktopath-reliability-4m-20260904.md` +
`docs/evidence-clicktopath-reliability-4m-20260904.json`.
🔑 **Read that before any further motion work — it found three things bigger than
the verdict**: the heading model used to place every target was wrong by a mean
87° (the mirror `90.13 - toward` is right to 1.000°), the "aligned start" check
was circular so runs 3–4 were secretly post-turn legs, and the device emits
`Robot orientation unavailable (1309)` that this integration cannot see.

---

## Standing decisions

The operator's, not derivable from the code. **They override anything older.**

1. **Audience: this yard only.** A bespoke tool for this LUBA. Per-mower constants
   are fine. Do not propose upstream-shaped work (auto-derivation, per-device
   calibration) as required.
2. **The goal is consistency, not precision** — click-to-go reliable enough to
   trust without watching. This is the objective everything else serves.
3. **Accuracy is CLOSED.** ~0.089 m mean over 16 landings, all inside the 0.15 m
   tolerance. The `0.62 × leg·sin(aim) + 0.065` fit's **0.065 m intercept is a
   sensing floor** (2–4 cm position noise plus ~1031 ms staleness), not a tuning
   target. Do not reopen.
4. **Night is CLOSED** (was "parked", closed 2026-09-04). Three independent gates
   refuse a real segment in the dark — `vio_active`, `night_segment_too_long`
   (1.0 m cap), `night_linear_loop_unsupported` — and night mode runs a fixed
   3-pulse budget with no mid-drive correction, so it cannot exercise what
   click-to-path depends on. Do not propose night runs.
5. 🛑 **Phase 2 continuous steering is CLOSED** (was "parked" 2026-08-28; closed
   2026-09-04). Closed on **value, not failure**: it buys ~4x speed and **not
   capability**, and stop-measure-go already delivers click-to-path at 6.0 m.
   🔑 **The blocker is measured and does not expire:** loop dead time is ≥ the
   ~1 Hz decision period, and that rate is **the device's choice** — the beta77
   cadence matrix measured requested periods of 100 / 250 / 500 / 1000 ms all
   arriving at p95 **1119–1372 ms**. No config, transport, or integration change
   moves it.
   ⚠️ **This closes the whole downstream line with it: criterion 2a, τ, the
   dead-time question, and the step-response probe as a research instrument.**
   Do not propose 2a repeats, longer steps, `step_ms` /
   `_STEP_RESPONSE_MAX_TOTAL_MS` raises, or Rule D (denied 2026-09-03 with a
   recorded reversal condition). The probe stays in the tree and stays safe; it
   is simply not where effort goes.
   **Reopening is an operator call, not a code question.**
6. **OTA firmware capture is CLOSED with a negative result** (was "paused"
   2026-08-16; closed 2026-09-04). The firmware was never captured and the
   remaining wall is **cryptographic** — the mower's own Aliyun device
   credentials, which no software-only method obtains. Better timing does not
   change that. ✅ One permanent capability came out of it: `ota_info_probe`, a
   read-only BLE service that works. Full record:
   `docs/ota-firmware-capture-investigation-20260816.md`.
   ⚠️ Unrelated leftovers to check before trusting either: UniFi Hardware
   Acceleration was deliberately left **OFF**, and the UniFi block-sta API is
   confirmed broken.
7. 📏 **Reliability statistics use the beta57+ epoch ONLY** (declared 2026-09-04).
   beta57 is the current control law and the Gate 5-accepted profile. Landings
   from beta32–beta56 span the beta37 turn-model rebuild, the beta38 re-aim
   guard, the beta40 post-turn gate and the beta42 quadrature fix.
   🚨 **They are valid as individual runs and MISLEADING as a rate** — pooling all
   81 banked segments yields authoritative-looking figures (89% under 1.0 m, 43%
   at 3.0–3.9 m, 44% for segment 2+ at ≥1.5 m) that are uncontrolled across four
   material control-law changes. **Quote them as shape, never as a measured
   rate**, and never mix epochs into one population.

---

## Operating facts

### Reach and landing — the click-to-path core

🏁 **Reach is CLOSED at 6.0 m.** `_MAX_SEGMENT_LENGTH_M` is a hard **6.10 m**
pre-dispatch refusal, so 6.0 m is the largest segment that can exist.
**Landing does not degrade with distance**: 0.1023 m @ 4 m, 0.1015 m @ 5 m,
0.1144 m @ 6 m. At 6 m the binding constraint is the **correction** budget
(2 of 3 `vio_max_realignments`), not the pulse ceiling (3 spare).
⚠️ **Do not raise `vio_max_realignments`** — tried twice, reverted twice, and a
leg that exhausts it stops safely.

⚠️ **Reach ≠ post-turn landing accuracy.** Those runs began **aligned**, single
segment. A leg following a junction turn is a different property, and the thin
evidence there is where the real risk sits (see standing decision 7).

🚨 **TWO TRAPS THAT SILENTLY VOID A LONG-LEG TEST.**
1. **The CARD cannot run one.** It auto-splits above
   `SPLIT_LEG_TARGET_METRES` (3.85 m), so a 4 m click measures the *splitter*.
   Use `raw_pymammotion_execute_vector_segment` directly.
2. **Schema defaults are NOT the accepted profile.** `max_linear_pulse_ceiling`
   defaults `None` (accepted **22**) — without it the run is fixed-budget and
   stops after ~1 pulse; `waypoint_tolerance` defaults 0.08 (accepted **0.15**);
   `calibrated_forward_heading_offset_degrees` defaults **116.5** where the
   profile is **102.4**.
   ✅ **Send `docs/accepted-profile.json` verbatim and verify key-by-key.**

### Measured constants

**Sustained (post-ramp) speed:** linear **300 → 0.223 m/s**, **400 → 0.295 m/s**.
🔑 The command/speed relation is **essentially LINEAR** (0.756 against a command
ratio of 0.750). 🗑️ The long-quoted *"a 25% command cut gave a 39% speed cut"* is
**REFUTED** — an artifact of comparing 4 s ramp-inclusive averages.

**Rotation**, four points, **n = 1 each, mechanism unexplained:**

| | linear 400 | linear 300 |
| --- | --- | --- |
| angular 120 | ~-8.2 °/s | **-9.175** |
| angular 180 | ~-11.8 °/s | **-13.431** |

Dropping linear 400 → 300 **raises** yaw rate ~12-13% (predicted in a committed
predeclaration and confirmed). ⚠️ **Do NOT fit a law to four points.**

**The ~1 Hz bundle is the ceiling.** Position, `toward` and VIO heading change on
exactly the same instants. Requested report periods of 100/250/500/1000 ms all
measure p95 1119–1372 ms — **the rate is the device's choice.**

### Traps that keep biting

🚨 **A REPOSITIONED MOWER'S HEADING TELEMETRY IS STALE UNTIL IT DRIVES —
INCLUDING "SMALL TEST" MOVES.** On 2026-09-04 a 0.5 m move was dispatched
immediately after the operator turned the mower **by hand**, aimed from the
then-current `toward` via `toward + calibrated_forward_heading_offset_degrees`.
It reported `target_reached` and the operator saw it drive in the mower's
**pre-reposition** direction. Both `toward` and `vio_heading` then jumped ~166°
on that first real motion — the device's own estimate had not re-anchored.
🔑 **`current_orientation` publishes `trustworthy` on corroboration between two
sources, NOT on freshness: both can be stale together.** And the additive offset
is not the model — on 30 fresh-`toward` pulses that night the mirror
`90.13 - toward` predicted the driven direction to a mean **1.000°**, while
`toward + 102.4` was off by a mean **87°**.
✅ **Before ANY armed dispatch, derive facing two ways — the last driven leg's
bearing and `(90.13 - toward)` with `toward` fresh — require agreement, and state
the destination in compass terms for the operator.** A short "test" move is an
armed dispatch. Full record: `docs/findings-clicktopath-reliability-4m-20260904.md`.

🚨 **"TARGET HEADING MATCHES `toward`" PROVES NOTHING — IT IS CIRCULAR.** If the
target was placed along `toward`, the echoed `target_reported_heading_degrees`
agrees with `toward` by construction. On 2026-09-04 that check was read as
"aligned start confirmed" on all four runs; the executor's own VIO calibration
drive showed the true facing was **26 / 27 / 122 / 135°** off, and runs 3 and 4
opened with real ~120–135° turns — making them post-turn legs, a property their
own predeclaration put out of scope. 🔑 **Alignment is only confirmed by a source
that does not derive from the number being checked** — the calibration drive's
measured `map_motion_heading_degrees`, or the operator's eyes.

🚨 **THE GATE DOES NOT CHECK THE MAP, AND THE MAP DOES NOT CHECK THE GROUND.**
On 2026-09-04 a map-polygon corridor scan showed **3.5 m** of clearance where the
operator's tape measured **2.79 m** to a real fence — 0.71 m of error, in the
unsafe direction, invisible to both the containment gate and the polygon scan
that CLAUDE.md prescribes as the gate's backstop. ✅ **On any corridor tighter
than a couple of metres, ask for a physical measurement.** The operator's tape
caught two real hazards that night that no software check would have.

⚠️ **SEPARATE THE RAMP BEFORE SIZING ANY WINDOW.** This cost three measurements in
eight days. 🚨 One 8 s figure agreed with the extrapolation to **1.6%** and was
**still wrong** for sizing. **Agreement with an estimate is not validation.**

⚠️ **A phase length is part of the hypothesis.** Copying `step_ms` from a run with
a different purpose silently made a measurement impossible before dispatch.

🚨 **THE CONTAINMENT GATE DOES NOT CHECK THE MAP.** `step_path_contained` measures
clearance against the **operator-supplied corridor polygon**, not the mowing area.
A position with 2.8447 m of real yard clearance passed 15/15 gates against a
3.20 m requirement because the corridor was centred on the mower by construction.
🔑 **Scan the map every time; the gate is not a substitute.**

⚠️ **A staged-but-unfinished deploy will eventually deploy itself.** beta96 was
staged with the restart deliberately interrupted; HA restarted overnight and
loaded it. **Finish it or back it out, never leave it.**

### BLE

Works above ~-70 dBm, dies below ~-76. `ble_rssi` is **self-reported and stale** —
it read -60 through a total outage — and **does not predict cadence**
(within-run median r = +0.042 over 24 runs). `ble_rssi 0` **does** mean the mower
has dozed; a mower restart clears it.
🔑 **Diagnose from `report_stream_probe`'s `queue_settle` and the HA container
log**, never from a proxy's entity state.
🗑️ **The mower is NOT paired to `master_bedroom_proxy`** — that inference came
from "mammotion" appearing in its entity *names* and was wrong.

### The motion gate

🚨 **Found armed at rest six times.** `automation.mammotion_disarm_motion_gate_when_left_armed`
now sweeps it. **Always verify a disarm from the live API AND RAW
`core.config_entries`** — and note HA writes `.storage` lazily, so a RAW read
taken immediately after a disarm can lie for ~15 s.

---

## How this project works

**This discipline is what makes the results trustworthy rather than anecdotal.
Do not relax it** — it is separate from the narrative that was archived, and it
has repeatedly caught real errors before they reached hardware.

- **Predeclare before you dispatch.** Any scoring rule, criterion or threshold is
  written and committed *before* the data exists. Choosing a rule after seeing
  which verdicts it flips is the failure this exists to prevent.
- **Write the falsifier.** State in advance what result would mean the change is
  wrong. The beta100 continuity guard shipped because it moved exactly one banked
  verdict, as predicted; a guard that moved any other would not have shipped.
- **Confirm each bound exceeds what the run needs to demonstrate its criterion.**
  A bound that is safe but makes the test impossible is a wasted run.
- **Verify with per-item records, not aggregates.** Net figures have hidden a 27°
  turn reversal and a live BLE link.
- **Every real run:** explicit operator go/no-go immediately before dispatch,
  fresh corridor scan against the map, daylight, gate verified disarmed after.
- **A run that stops safely on a named refusal is a FAIL**, not a smaller number.

### Credentials

`.env` is gitignored and **has never been committed** — verified 2026-09-04. Keep
it that way; it holds the HA host SSH password, the HA API token, and the
Mammotion cloud account login.

✅ **`HA_SSH_PASS` was ROTATED** by the operator (confirmed 2026-09-04), closing
the exposure from the 2026-08-31 `scripts/ha_ssh.exp` defect, where an
`exp_continue` left the password pattern armed and re-sent the real password into
the SSH stream on any later output containing a "password:"-like substring. The
script now allows exactly one send per invocation.

🚨 **`MAMMOTION_PASSWORD` was exposed into a session transcript on 2026-09-04 by
my own command** — `grep -o 'MAMMOTION[A-Z_]*=[^ ]*' .env`, run to find the HA
API variable names. **Rotate it in the Mammotion app if that has not been done**;
it is a live credential that controls the mower over the cloud path.
🔑 **When you need to know which variables exist, print the NAMES only:**
`grep -oE '^[A-Z_]+=' .env`. Never a pattern that captures the value. This
applies to any file that holds secrets, and the transcript is as much an
exposure surface as a commit.

⚠️ **Never `git add -A`. Stage by explicit path.** The rule stands on its own:
this repo's working tree routinely holds scratch files and operator edits, and
`-A` has swept them into commits twice.

✏️ **Its old justification was a phantom, corrected 2026-09-05.** This bullet
used to say the operator "keeps an uncommitted edit in
`docs/agora_outbound_audio_probe.md`" — so every session since August tiptoed
around that file as if the edit were deliberate. It was two stray keystrokes
(`/st` typed into the middle of a word, and a deleted `/`). The operator has
since fixed both and **the working tree is clean**. 🔑 **If a file looks
permanently dirty, read the diff before deciding it is sacred.**

⚠️ **Repositories owned by `mikey0000` are read-only.** Push only to `Chorty`, and
pass `-R Chorty/Mammotion-HA` to every `gh` command.

---

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
