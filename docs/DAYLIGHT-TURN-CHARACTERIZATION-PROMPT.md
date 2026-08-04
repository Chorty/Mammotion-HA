# Daylight turn-characterization session prompt (Opus/Sonnet)

Paste the block below into a fresh session. It is written to be self-contained
and procedural: the design thinking is already committed, this session executes
a measurement protocol. Everything it references is committed on
`feat/vio-turn-to-heading` in `Chorty/Mammotion-HA`.

---

You are running the daylight VIO turn characterization for the Mammotion-HA P0
work, repo `Chorty/Mammotion-HA`, branch `feat/vio-turn-to-heading`.

**Read first, in this order:** `CLAUDE.md`, then `docs/CODEX-HANDOFF.md`
(especially the implemented turn-feasibility guard section), then the
"Sequencing" block in `docs/NEXT-SESSION.md`. Confirm `git status` is clean
before starting. Do not reconstruct facts from chat history; the docs are the
record.

## What this session is and is not

The host (HA at 192.168.1.106) runs the beta19 build `617337d3`, which does
NOT contain the new turn-feasibility guard. That is intentional: this session
is a **hardware measurement**, not a test of the guard. You are measuring, on
fresh daylight geometry, whether the guard's evidence floors generalize:

- per-command rotation ≥ **16.5°/s × pulse** at refresh 200 / angular 500 /
  1500 ms (Gate 4 observed 16.5–21.3°/s);
- per-command translation ≤ **0.0403 m/s × pulse**;
- opportunistically, the implied forward-heading offset vs the accepted
  102.4° (the open ~11°-low question — measure, do not act on it).

**Hard boundaries.** Do NOT: deploy anything to the host; bump any version;
edit `LUBA_ACCEPTANCE_PROFILE` or any profile constant (including the new
feasibility constants — revision is a separate review session); attempt Gate 4
or Gate 5; touch PR #10; push, comment, or act on any `mikey0000` repository.
Commit only evidence files and doc updates to `Chorty/feat/vio-turn-to-heading`.

**Safety rules, non-negotiable.** Physical motion requires the operator
physically present, daylight, blades off, clear area, released e-stop, and a
fresh explicit operator "go" for EVERY armed run. Arm only immediately before
a run; after every run (pass, fail, or abort) disarm in a finally-style step
and verify `enabled: False`, no active session, blades off, stationary
telemetry. `command_ok` never proves delivery — verify by observed effect.

## Environment

```sh
cd <repo> && set -a && source .env && set +a   # provides HA_URL / HA_TOKEN
scripts/ha_set_experimental_motion.py status    # must report enabled: False
```

Entity: `lawn_mower.back_yard_clip_skywalker`. If BLE looks dead (`ble_rssi`
0 / no adverts), the mower is dozing (~10–13 min idle) — wake it per
`mower-live-testing-workflow` (app wake or proxy-restart trick) before
preflight. The mower self-reports `ble_rssi`; it is NOT a liveness signal.

## Preflight (no motion, run before asking for any "go")

1. `scripts/ha_set_experimental_motion.py status` → `enabled: False`.
2. Start the session-long evidence collectors in separate terminals:
   `scripts/motion_capture.py --seconds 3600 --out docs/evidence-turnchar-beta19-capture-20260804.jsonl`
   and `scripts/ble_session_report.py` (redirect to
   `docs/evidence-turnchar-beta19-ble-report-20260804.txt`).
3. Dry-run probe (dry_run is the schema default; sends nothing):
   call `mammotion.vio_turn_to_heading` with `target_vision_heading: 0` and
   read back `initial_vio_state`, `initial_vio_feed`, `initial_vision_heading`.
   VIO must be state 2 with a live feed (brightness not Dark). ⚠️ Take every
   VIO figure from service responses, never from `sensor.*_vio_heading`
   (coordinator-tick cached; stayed bit-identical across 374 samples live).
4. If VIO is cold (stationary overnight ⇒ likely inactive): warm it with ONE
   short supervised forward drive using `mammotion.vio_motion_probe`
   (real mode, both confirms, operator go, arm/disarm around it). It drives,
   samples VIO, and always stops. Re-check step 3 afterward.
5. Confirm position `valid_for_motion: true`, `AREA_INSIDE`, nonzero
   `zone_hash`, RTK Fix, blades OFF, work mode READY/PAUSE.

## The measurement runs

Four turns at the accepted cadence, alternating direction so the mower stays
local. For each: read the current vision heading from a fresh dry-run (step 3
form), compute `target = normalize(current + delta)` into (-180, 180], then
run via the evidence runner so request + result + telemetry are retained in
one process:

| run | delta | notes |
| --- | ----- | ----- |
| 1 | +45° | ~2 commands expected |
| 2 | −90° | ~3–4 commands |
| 3 | +135° | ~5–6 commands |
| 4 | −170° | near-worst-case; ~7 commands, watch displacement |

Request payload per run (write to a JSON file, then use
`scripts/run_motion_with_evidence.py --service vio_turn_to_heading --request
<file> --result docs/evidence-turnchar-beta19-run<N>-result-20260804.json
--capture docs/evidence-turnchar-beta19-run<N>-capture-20260804.jsonl`):

```json
{
  "entity_id": "lawn_mower.back_yard_clip_skywalker",
  "target_vision_heading": <computed>,
  "heading_tolerance_degrees": 18,
  "angular_speed": 500,
  "pulse_duration_ms": 1500,
  "max_commands": 8,
  "max_displacement_m": 0.5,
  "motion_refresh_interval_ms": 200,
  "prefer_ble": true,
  "dry_run": false,
  "confirm_blades_off": true,
  "confirm_clear_area": true
}
```

⚠️ `motion_refresh_interval_ms` defaults to **0** on this service — omitting
it silently measures the single-shot regime and invalidates the run.

Per-run sequence: fresh operator **go** → `ha_set_experimental_motion.py on`
→ run → `ha_set_experimental_motion.py off` → verify disarmed, stationary,
blades off, no session. Between runs, re-verify VIO is still live (a turn can
end facing shade). Stop the session entirely on: any `stop_failed_aborting`,
`no_actuation_detected`, `vio_telemetry_stream_stale`, unexpected physical
behavior, or operator discomfort — disarm, keep the evidence, write up.

## Analysis (offline, after the runs)

From each result JSON's `command_results`:

- rotation rate per pulse = `measured_change_degrees` /
  (`motion_refresh.elapsed_ms` / 1000) — but only for pulses with
  `heading_went_fresh: true`;
- translation per pulse = delta of `displacement_m` between consecutive
  commands, / (`elapsed_ms` / 1000);
- note any pulse with negative `progress_degrees` (direction fault) and any
  scaled final-approach pulse (`final_approach.applied: true` — its rate is
  still valid, that is by design).

Report, per run and pooled: min/mean rotation °/s vs the 16.5 floor; max
translation m/s vs 0.0403; and `scripts/motion_capture.py --summarise` implied
offset for the forward (probe) drives vs 102.4°. State plainly whether each
floor held, was violated, or was untested. Do NOT change any constant, even if
violated — that decision belongs to a separate review session with this
evidence in hand.

## Wrap-up

1. Verify: experimental motion off, no session, blades off, mower stationary,
   collectors stopped and files saved under `docs/evidence-turnchar-beta19-*`.
2. Append a dated session record to `docs/NEXT-SESSION.md` (measurements,
   floor verdicts, anomalies, exact evidence filenames) and note it in
   `docs/CODEX-HANDOFF.md`. Claim no gates.
3. Run the CI-equivalent validation matrix from `CLAUDE.md` (docs-only changes
   should be trivially green; `pre-commit run --all-files` must stay green —
   evidence JSON files need trailing newlines).
4. Commit evidence + docs to `Chorty/feat/vio-turn-to-heading` only.

If anything surprising happened mid-run, do not diagnose deeply in this
session — retain the evidence and hand off; the constant-revision and
deploy/Gate-4-retry decisions are explicitly out of scope here.
