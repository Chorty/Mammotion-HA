# Codex handoff prompt

Paste everything between the rules below into Codex as the opening message.
Updated 2026-08-14 after released/deployed `0.6.4-beta54`, its first supervised
card-driven Night Go, and the verified/deployed Night and Real Go follow-ups in
draft PR #14.

⚠️ Codex reads `AGENTS.md` by convention; this repo's instructions live in
`CLAUDE.md`. The prompt below points at it explicitly, so no rename is needed —
but if you later add an `AGENTS.md`, keep it a pointer rather than a fork, or the
two will drift.

---

You are picking up an in-progress hardware project. Read before you write.

## The project

`Mammotion-HA` is a Home Assistant custom integration that drives a Mammotion
LUBA robot mower to points clicked on a map, **with the blades OFF**. The goal is
point-and-click movement, not mowing. It is at BETA; five acceptance gates are
complete and Gate 5 has passed twice.

Working directory is the repo root. Branch: `agent/night-real-go-followup` at
`801c1798`, clean and pushed; draft PR #14 targets `main`. Raw evidence is in
preceding commit `dd53e266`. `main`, `origin/main`, and `v0.6.4-beta54` agree at `0bd35160`.
`origin` is the **Chorty** fork —
`mikey0000/*` repositories are strictly read-only, never push or comment there.

## Read these first, in order

1. `CLAUDE.md` — start at "Current build", which is the live state.
2. `docs/NEXT-SESSION.md` — the "2026-08-13 HANDOFF" section at the top.
   **Everything below that section is stale as build state** but still valid as
   measurement evidence.
3. `docs/night-segment-implementation-plan-v1-20260813.md` — implementation
   record and hardware sequence. Off-mower items 1–14 and items 15–18 are complete.

## Your task

The off-mower night implementation, beta54 card control, and §7 items 15–18 are
complete. Item 15 measured a −54.2208° quantum from one angular −500 / 1,500 ms
refreshed pulse, then the first forward pulse exposed an 81.416° aim mismatch;
the night guard stopped safely. Read
`docs/night-segment-turn-quantum-20260813.md`. Item 16 then found `toward`
arrived as one post-pulse step, with no intermediate values across 73 samples;
read `docs/night-toward-latency-20260813.md`. The item-17 backward pulse kept
`toward` at 173.1023° while moving 0.418536 m almost exactly opposite the
inferred body heading, settling `toward` as body heading under reverse.
RapidState `fuse_status` stayed 0 `NO_POSE` in all 81 records and was
non-informative; read `docs/night-reverse-heading-20260813.md`. Item 18 then ran
one 0.699963 m perpendicular segment, reached its 8° opening-turn tolerance,
and stopped after three linear pulses on `no_target_progress` at 0.114277 m. It
is characterization, not an acceptance pass; read
`docs/night-segment-item18-20260814.md`. Item 15's separate mismatch remains
unexplained.

Beta54 subsequently ran one card-driven 0.739138 m Night Go. It reached heading
tolerance after three turns, then stopped safely on `no_target_progress` at
0.117085 m after three forward pulses. Pulse 2 was already only 0.082661 m from
the target. A pre-pulse bearing was reused after the mower crossed the target,
causing the unnecessary third pulse. Read
`docs/night-go-card-beta54-20260814.md`.

The PR branch has committed, unreleased night and Real Go fixes. Night
calculates the residual bearing from settled post-pulse RTK and sends
`sample_delays: [0, 3]` from the card/harness. Real Go now uses one four-second
VIO-calibration feedback window, reuses settled linear telemetry instead of
adding a three-second sample wait, and performs one final card reload. Its
payload, mandatory stops, safety gates, legacy/night timing, and frozen profile
values remain unchanged. One supervised Real Go run safely stopped before its
second linear dispatch on `command_queue_backlogged`. The subsequent correction
uses the existing bounded queue-settle check after VIO position feedback. A
fresh supervised 0.70 m run reached its target in 19.2 s with 0.093100 m landing
error; all three queue checks reached depth zero and all movement/stops
succeeded. It is deployed motion-disabled after all six checks passed. Read
`docs/real-go-throughput-hardware-20260814.md`. Until fresh authorization,
hardware work is read-only.

A same-day read-only autonomous-mow comparison then captured progressive
`toward` changes through three vendor pivots. Read
`docs/autonomous-mow-observation-20260813.md`: the one-step item-16 result is
specific to the bounded manual pulse/report cadence, not all rotation.

## Hard constraints — violating any of these is a failed change

1. **`LUBA_ACCEPTANCE_PROFILE` in `custom_components/mammotion/www/mammotion-custom-path-card.js` is FROZEN.**
   Changing any key's *value* un-accepts a hardware-accepted profile and obliges
   a full re-pin plus a fresh acceptance gate. Night v1 did not change the card;
   beta54 later added a separate Night Go control without changing any frozen
   profile value.
2. **Do not change behaviour on the VIO daylight path.** It passed Gate 5 twice.
   A `turn_mode: "vio"` run's dispatched payload and response fields must be
   byte-identical after your change.
3. **Do not delete or bypass the `vio_active` safety gate.** It has two
   construction sites (`services.py:8061` unconditional inside the VIO primitive,
   `:11167` already scoped to `turn_mode == "vio"`). Night works by *not being*
   `"vio"`, so all nine `turn_mode == "vio"` blocks stay inert. Neither site is
   edited.
4. **All imports inside the integration must be relative** (`from .services import x`).
   Absolute `custom_components.mammotion...` imports fail at runtime in HA.
5. **If you add or rename any entity or ENUM state**, update
   `custom_components/mammotion/strings.json` *and every file* under
   `custom_components/mammotion/translations/`, translated into each language —
   not English placeholders — plus an `icons.json` entry.

## Traps that have already caught people here

1. **The standalone turn service and the segment executor's legacy branch are NOT
   the same code path.** Five night turns converged by calling
   `raw_pymammotion_turn_to_heading` **directly**. The segment's legacy branch
   (`services.py:11498-11517`) omits `motion_refresh_interval_ms` (primitive
   default `0`) and passes `angular_speed_fast/slow` at the schema default
   **180**, which does not break static friction on a stationary pivot
   (~3°/pulse vs ~48° at 500). Night must dispatch at **angular 500 with refresh
   forwarded**, or the segment dies in its opening turn.
2. **Two heading-conversion sites CANCEL.** `_raw_vector_readiness_target_points`
   (`services.py:9448-9465`) builds `toward + offset`; the executor
   (`:11094-11100`) converts back `map − offset`. Fixing one alone breaks the
   readiness probe by a **heading-dependent** amount (~4.3° at `toward` 176,
   ~132° at `toward` 60) — so a half-fix passes review at whatever heading you
   happen to test. The plan avoids this by scoping the mirror to night only.
   **Do not "also fix" the shared sites.**
3. `ruff format` rewrites `except (A, B):` to `except A, B:`. That is **correct** —
   PEP 758, and this project targets Python 3.14. Do not revert it.
4. Use `mypy --follow-imports=skip custom_components/mammotion` exactly. A
   whole-tree run reports 4 pre-existing errors CI never checks.

## Verification — run these, they are exactly what CI runs

There is **no global `uv`**; use the project venv directly.

```sh
.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests
.venv/bin/python -m ruff check custom_components tests
.venv/bin/python -m ruff format --check custom_components tests
.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion
npm run test:frontend
.venv/bin/python -m pre_commit run --all-files
```

Current PR #14 code results personally produced: **668 pytest, 46 frontend**,
all six commands green; GitHub Python/hassfest/Socket checks passed. **Run them again after any change and report the counts you
actually produced** — do not quote a number you did not
generate. If something fails, paste the real traceback rather than summarising.

## Hardware — do not touch it

The mower is real and outdoors. The last post-run sample was `(4.2954,
-3.8079)`, `MODE_READY`, RTK Fix, blades off; do not treat that dated
result as its current position. A Home Assistant host at `192.168.1.106` runs
the corrected working tree motion-disabled.

- **Do not enable experimental motion, do not arm the motion gate, and do not
  send any movement command.** Real motion requires explicit per-run
  authorization from the human operator, who supervises it in person.
- Do not deploy to the host or restart Home Assistant unless asked.
- The motion gate is currently **disarmed** (`real_motion_allowed: false`) and
  must stay that way. If any script you touch can open that gate, it must treat
  *"I called enable"* as what obliges the disarm — never *"enable succeeded"*.
  That exact bug once left the gate open.

Your work is entirely off-mower: code, tests, and the CI gates above.

## How this project expects you to work

- **Verify with per-item records, not aggregates.** Summary fields have hidden
  the truth here repeatedly — a net figure once concealed a 27° turn reversal.
- **Never claim a file, symbol, or field exists without reading it.** Quote
  `file:line`.
- **Distinguish measured from inferred**, explicitly, in code comments and in
  what you report back.
- Comments should record *why* a constant has its value and what evidence backs
  it — match the existing density, which is high on the motion path deliberately.
- If you find a real problem with the plan, say so in a sentence and keep going
  under a stated assumption; don't silently narrow the task.

## What is genuinely unsettled — do not paper over these

- Item 15, item 18, and the beta54 card-driven run have exercised the night
  path. The latter two stopped at 0.114277 m and 0.117085 m. None is a
  landing-accuracy pass.
- The mirror relation has now been used by one control loop and disagreed with
  the measured forward course. The cause is not yet established.
- `toward` did not flip under item 17's reverse pulse; body-vs-course is settled.
  What remains open is item 15's separate forward-course/mirror disagreement.
- One bounded manual turn showed a single post-pulse `toward` step, while
  continuous vendor turns streamed intermediate headings. Do not generalise
  either cadence beyond its measured command path.
- **No landing accuracy is evidenced at night.** `waypoint_tolerance: 0.15` is a
  VIO-path number and must not be presented as a night specification.

Start by reading the three documents, then summarise back what you understand the
task to be and any disagreement with the plan, **before** editing code.
