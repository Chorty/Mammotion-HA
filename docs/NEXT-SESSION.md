# Claude handoff: finish Mammotion-HA P0 beta

⚠️ **Everything below the "2026-08-13 HANDOFF" section is older and its build
state is stale.** Those sections remain accurate as *evidence* — the
measurements stand — but do not act on any build/host/gate state they describe.

---

## 🚧 2026-08-17 — REACH WORK ON PR #15, UNMERGED; PROFILE UN-ACCEPTED

`agent/reach-20ft-reaim-trigger` (PR #15) no longer matches the deployed
beta56, and the difference changes a frozen profile key. **Read
`docs/reach-20ft-and-the-reaim-trigger-20260817.md` first.**

| | |
| --- | --- |
| Host | Still **beta56**, unchanged. Nothing was deployed. |
| Branch | `agent/reach-20ft-reaim-trigger`, **PR #15**, pushed. Not merged, not released, not deployed |
| Motion gate | **DISARMED**, untouched. **No motion has run on any of this work.** |
| Profile | 🚨 `max_linear_pulse_ceiling` **14 → 22**, so `LUBA_ACCEPTANCE_PROFILE` is **un-accepted**; owes §4 re-pinning and another Gate 5 |
| CI | Green: 687 pytest, 46 frontend, ruff check/format, mypy, all nine pre-commit hooks |
| Review | **Two high-effort rounds, 14 findings, all real.** Round 2 found two of round 1's own fixes wrong, which forced a scope cut: `vio_max_realignments` stays at the accepted **3** and the divergence detector was removed entirely |

**What it does:** adds the daylight path's first pre-dispatch length cap
(`segment_too_long`, 6.10 m = 20 ft) and changes the mid-drive re-aim trigger
from an angle test to a projected-miss test. The finding behind it: the trigger
was `abs(aim) > 18°` while the objective is `range × sin(aim)`, so a 17° error
with 14 m to run — a **4.09 m miss** — fired no correction at all. That, not
distance, is what limited leg length.

**The next action is a supervised run, and it is NOT the 20 ft leg.** Repeat the
1.65 m post-turn geometry from 2026-08-15 first: it is the only case with a
recorded counterfactual. See §4 of that doc.

---

# 🚦 2026-08-14 HANDOFF — beta55 released; PR #14 reviewed and merged

⚠️ **Superseded by a beta56 deploy on 2026-08-16** — see the note right below
this table. The table itself otherwise still reflects the beta55 motion/gate
state, which beta56 did not touch.

## 0. Live state, verified 2026-08-14 after the beta55 motion-disabled install

| | |
| --- | --- |
| Branch | `main` at `a2351048` (was `5ef37511` at beta55), clean; PR #14 merged as `efa1eda8`. `agent/night-real-go-followup` is merged and can be deleted |
| Released version | `0.6.4-beta56`, tag `v0.6.4-beta56`; manifest, pyproject, card, and lock version sites agree (`0.6.4-beta56` / `0.6.4b56`) |
| Host | Running beta56. All 46 files byte-identical, card `58acc9569907b1a16f4a89e785b0feeb` at **both** paths, Lovelace `?v=0.6.4-beta56&build=58acc956`. Backup `/config/mammotion-backup-20260815-2321-pre-beta56.tgz` |
| Motion gate | ✅ **DISARMED.** `real_motion_allowed: false`, `enabled: false`, no active session, no last session. **No motion was commanded by the beta56 release or install** — it only added a new read-only service, `ota_info_probe` (see below). |
| Mower state at install | **Docked**, `MODE_READY`, RTK `Fix` earlier verified; re-verify current position/charge state before any motion work, since beta56 was a backend-only deploy done independent of the motion/turn work below. |

**beta56 (2026-08-16) added exactly one thing to the shipped integration:**
`ota_info_probe`, a read-only HA service that queries the mower's own OTA
status over BLE. It is unrelated to the turn/accuracy work in the rest of
this handoff. Full context, and the paused firmware-capture investigation
that produced it: `docs/ota-firmware-capture-investigation-20260816.md` and
the pointer in `CLAUDE.md` right before "Build provenance". ⚠️ That
investigation also left the UniFi gateway's **Hardware Acceleration
disabled** and found its **block-sta API broken** — check both before
relying on either for anything, motion-related or not.

⚠️ **The night fix in beta55 has never run on hardware.** It is verified
off-mower and by replay against the recorded beta54 geometry only. The Real Go
throughput fix has exactly one supervised pass (0.70 m, 0.093100 m landing) —
one path, not a reliability population. Any physical run needs fresh, explicit,
per-run supervised authorization.

## 1. What to do next

✅ **Night v1 is implemented and beta54 adds a guarded card control.** The backend adds an
explicit `turn_mode: "night"` for one forward-only segment, night-only mirror
conversion, angular 500 with refresh, fixed-budget/RTK/length/heading/re-aim/
reverse/multi-segment refusals, and the `--night-segment` harness mode. The
frozen card profile and VIO/legacy paths remain pinned by tests.

✅ **§7 item 15 is measured.** One 1,500 ms angular −500 pulse delivered six
200 ms refresh writes, changed `toward` by −54.2208°, and translated 0.07459 m.
The subsequent forward pulse moved 0.43648 m but its movement vector missed the
target direction by 81.416°. The night guard correctly stopped with
`night_reaim_required_but_unavailable`. Read
`docs/night-segment-turn-quantum-20260813.md` and the complete JSON it links.

✅ **§7 item 16 is also measured.** One angular-only +500 pulse produced 73
concurrent runtime samples. `toward` stayed 43.1856 throughout the 1.551 s
refreshed window, then appeared in one step as 79.492 (+36.3064°); no
intermediate value was observed at the roughly 0.1-second capture cadence. Read
`docs/night-toward-latency-20260813.md`.

✅ **§7 item 17 is measured.** One backward-only pulse moved 0.418536 m on
bearing 96.433921° while `toward` stayed exactly 173.1023°. The body-heading
prediction for reverse was 97.0277°, only 0.593779° away. This settles
`toward` as body heading rather than course-over-ground under reverse.
RapidState `fuse_status` remained 0 `NO_POSE` in all 81 capture records, so it
was not a useful live fusion discriminator on this manual path. Read
`docs/night-reverse-heading-20260813.md` and its linked raw JSON.

✅ **§7 item 18 is measured, but is not an acceptance pass.** One 0.699963 m
perpendicular night segment reached its opening 8° tolerance in one turn pulse,
then used all three linear commands and stopped on `no_target_progress` at
0.114277 m from target. Per-pulse mirror observations were 92.0720 / 90.7417 /
89.1569°. Read `docs/night-segment-item18-20260814.md` and the raw evidence it
links. The run does not establish a night tolerance or landing distribution.

✅ **A beta54 card-driven Night Go is also measured.** The 0.739138 m requested
leg reached its opening heading after three turns. It used three forward pulses
and stopped safely on `no_target_progress` at 0.117085 m. After pulse 2 it was
only 0.082661 m away, 0.002661 m outside the configured tolerance; pulse 3 moved
away from a target already behind it. See
`docs/night-go-card-beta54-20260814.md` and its linked complete JSON.

✅ **The corrected Real Go check passed.** The separately authorized 0.70 m run
reached its target in 19.2 s with 0.093100 m landing error. All three
post-feedback queue-settle checks reached depth zero in about 100–101 ms and
every movement/stop succeeded. The gate was independently verified off after
the run. Next software action: review and merge draft PR #14, then prepare the
next beta. The night change recomputes the residual target bearing from the
settled post-pulse RTK position, so the existing reverse-recovery refusal can
stop the crossed-target case. The card/harness set `sample_delays: [0, 3]` to
replace beta54's inherited 0/5/10/20/30/45/60-second waits. Real Go now avoids
additive calibration/linear feedback waits and one duplicate card reload while
preserving its payload, stops, gates, and frozen profile values. The first
hardware run safely refused its second linear command on
`command_queue_backlogged`; the correction now explicitly settles that queue
after feedback and refuses without dispatch if it remains non-live. The final
local tree produced **668 pytest and 46 frontend tests**, with all six
verification commands green. It is deployed motion-disabled and hardware-tested,
but not released. Read
`docs/real-go-throughput-hardware-20260814.md`.

⚠️ Item 15's 14.3069° mirror observation remains unexplained, although item 18's
three observations near 90.13° show it is not stable across all night pulses.
The latency result does not explain that mismatch because the executor waits
for post-command feedback before driving. Any further physical run requires
separate, fresh, supervised authorization; this handoff grants none.

🔎 **Autonomous comparison, read-only:** while the mower continued its own
Backyard Hill route, a 179.895 s capture recorded 291 samples and three complete
pivots. `toward` streamed progressively during those continuous vendor turns;
40 usable steps gave `bearing + toward = 90.57°`, circular SD 2.02°. This means
item 16's single post-pulse update characterizes our bounded pulse/report path,
not all mower rotation. See `docs/autonomous-mow-observation-20260813.md`.

The original plan was produced by a 20-agent adversarial design workflow and **every gate design
in it was refuted at least once before the plan was written**. Read §4 before
changing anything — it records 21 defects already fixed and 8 risks knowingly
accepted, so a "fix" you invent may already have been considered and rejected.

Companion: `docs/night-segment-design-plan-20260813.md` is the **findings log**
(what was discovered, and corrections to earlier drafts). ⚠️ Its §2
runaway-safety claim is **wrong**; the banner at its top says so.

## 2. 🚨 Traps that will bite a session starting cold

1. **The standalone turn service and the segment executor's legacy branch are NOT
   the same code path.** Five night turns converged by calling
   `raw_pymammotion_turn_to_heading` **directly**. The segment's legacy branch
   (`services.py:11498-11517`) supplies *different defaults* for two parameters
   that decide whether the mower moves at all: it omits
   `motion_refresh_interval_ms` (primitive default `0`) and passes
   `angular_speed_fast/slow` at the schema default **180**, which does not break
   static friction on a stationary pivot (~3°/pulse). **Night must dispatch at
   angular 500 with refresh forwarded.**
2. **Two heading-conversion sites CANCEL.** `_raw_vector_readiness_target_points`
   (`:9448-9465`) builds `toward + offset`; the executor (`:11094-11100`)
   converts back `map − offset`. Fixing one alone breaks the readiness probe by a
   *heading-dependent* amount (~4.3° at `toward` 176, ~132° at `toward` 60) — so
   a half-fix passes review at whatever heading you happen to test. The plan
   avoids this by scoping the mirror to `turn_mode: "night"` only.
3. **`LUBA_ACCEPTANCE_PROFILE` (card JS) is frozen.** Changing any key's *value*
   un-accepts a twice-Gate-5-passed profile and owes a re-pin plus a fresh Gate 5.
   Night v1 did not change it. Beta54 later added a separate Night Go card
   control while leaving every frozen profile value intact.
4. **The card is served from TWO paths.** Deploy to both and bump the Lovelace
   resource key, or the browser silently loads the stale card.
   ⚠️ `scripts/ha_set_card_resource.py` does **not** append the `build=` suffix —
   pass it inside the version argument:
   `ha_set_card_resource.py "0.6.4-betaNN&build=<md5 prefix>" --apply`.
5. **Night turns converge but are NOT accurate.** 5/5 reached target, but the
   pulse quantum is **48.15° ± 5.70** and nothing scales it; four of five
   converged on luck. Do not read "5/5" as control.
6. **Verify with per-item records, not aggregates.** This repo has been burned
   repeatedly by summary fields hiding the truth.

## 3. Running the gates (these are exactly what CI runs)

```sh
.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests
.venv/bin/python -m ruff check custom_components tests
.venv/bin/python -m ruff format --check custom_components tests
.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion
npm run test:frontend
.venv/bin/python -m pre_commit run --all-files
```

There is **no global `uv`** — use `.venv/bin/python` directly. PR #14's code
commit personally produced **668 pytest, 46 frontend**, all six commands green;
its Python, hassfest, and Socket GitHub checks also passed. Run them again after later changes; do not carry these
counts forward as if newly measured.

## 4. Hardware rules — non-negotiable

- **Real motion needs explicit per-run authorization from the operator.** Never
  arm on your own initiative.
- **Arm immediately before, disarm immediately after, and verify both.** A
  script that can open the gate must treat *"I called enable"* as what obliges
  the disarm — never *"enable succeeded"* (that bug left the gate open once,
  fixed in `c196b8b1`).
- **Save the complete response JSON** as `docs/evidence-*.json` and commit it
  **before** writing any prose about the run. The card now has a
  "Download last run JSON" button for exactly this.
- Blades OFF. This project never runs blades-on — the goal is point-and-click
  movement, not mowing.
- Night-mode turns work without VIO. Item 15, item 18, and one beta54
  card-driven Night Go have run; none establishes a night landing-accuracy
  specification.

## 5. What is genuinely unsettled

- Three night-path characterizations have run. Item 15 stopped after one pulse
  on `night_reaim_required_but_unavailable`; item 18 stopped at 0.114277 m; the
  beta54 card run stopped at 0.117085 m and exposed the stale pre-pulse-bearing
  decision. None is a landing-accuracy pass.
- The mirror has now been consumed by one control loop and disagreed with the
  measured forward course by 75.823° (`movement bearing + toward` was 14.3069°
  rather than 90.13°). Cause remains unknown.
- `toward` does **not** flip under reverse: item 17 held it at 173.1023° while
  the mower travelled 0.418536 m almost exactly opposite the inferred body
  heading. The remaining orientation question is why item 15's forward course
  disagreed with the same mirror relation.
- On one refreshed turn, `toward` was stepwise rather than live: no intermediate
  value appeared at ~0.1-second cadence. The exact physical-stop-to-report delay
  was not measured at the protocol timestamp.
- **No landing accuracy is evidenced at night.** `waypoint_tolerance: 0.15` is a
  VIO-path number.

---

## 🚦 (older) START HERE 2026-08-11 night — REACH IS SOLVED. 4 m on one segment, measured.

Host and branch both `0.6.4-beta41`. **Gate DISARMED**, verified after every run.
Mower is **DOCKED and charging** at (4.3188, 3.2862), `CHARGE_ON`, MODE_READY,
blades off, battery 59% when it docked. VIO went 80 → 77 → 59 → **0** between
20:27 and 20:37 EDT; `camera_brightness` reads `dark`.

📋 **TOMORROW'S PLAN IS THE NEXT SECTION.** Read it before anything else.

*(Docked with `lawn_mower.dock` at 20:39 — the mower's own RTK navigation covered
~12 m in ~65 s in full darkness. Worth remembering: the VENDOR's nav works at
night; only our closed-loop executor needs VIO. That is a night-repositioning
tool we were not using.)*

## 🏁 START HERE — A CLOSED-LOOP TURN CONVERGED IN THE DARK, NO VIO

Read `docs/night-closed-loop-turn-works-20260812.md`. `raw_pymammotion_turn_to_heading`
— the **legacy** primitive, which closes on `toward` and has no `vio_active`
gate — reached its target in full darkness with `tracked_features: 0`:

```
stop_reason  target_heading_reached   commands_sent 2
toward       -156.85 -> 285.37        target 293.15, error 7.78 deg (tol 18)
cmd1 rotated 44.24 deg (7 writes)     cmd2 rotated 37.98 deg (3 writes)
```

Single-variable, same service and target offset, only `motion_refresh_interval_ms`
differing:

| | commands | per-command | outcome |
| --- | --- | --- | --- |
| refresh off | 4 | +9.79/+6.77/+5.37/+7.02 | 29° of 90°, `max_commands_reached` |
| refresh 200 | 2 | **+44.24/+37.98** | 82° of 90°, **`target_heading_reached`** |

**Daylight-only was never a property of the machine. It was the turn primitive's
choice of heading source, plus a missing refresh.**

### ⚠️ What is NOT done — read before planning a night segment

- **n = 1** on the refreshed path.
- **A turn is not a segment.** The `vio_active` gate still refuses
  `turn_mode: "vio"` unconditionally. **No closed-loop segment has run in the
  dark.**
- 🚨 **The heading conversion is wrong by construction.** The target above was
  computed as `current toward + 90` and passed by hand, deliberately bypassing
  `calibrated_forward_heading_offset_degrees: 102.4`. A night SEGMENT would hit
  that conversion — `toward` is a compass bearing needing a MIRROR
  (`map = 90.13 − toward`), not an additive offset. **Fix that first.**
- `toward`'s latency DURING rotation is unmeasured; all readings are settled.
- The 7.78° residual sits inside a loose 18° tolerance.

### Next on this thread

1. Repeat the refreshed turn 2–3× for n, both directions.
2. Sample `toward` during the pulse, not after.
3. Fix the map-heading conversion (mirror, not offset).
4. Only then a night segment — and it needs a night MODE that selects the legacy
   turn and refuses what needs VIO, **not** a bypass of `vio_active`.

## 🚨 (the finding that made it possible) `toward` TRACKS IN-PLACE ROTATION

Read `docs/toward-tracks-in-place-rotation-20260812.md`. Two armed pivots in
**full darkness with VIO dead** (`tracked_features: 0`):

| | command | translation | `toward` change |
| --- | --- | --- | --- |
| pivot 1 | `angular +500` | **0.0382 m** | **+99.55°** |
| pivot 2 | `angular −500` | **0.0295 m** | **−61.43°** |

Opposite command, opposite sign, ~3 cm of travel each. **99.55° cannot be a
course-over-ground computed from 3.8 cm** — the position noise floor is 2–4 cm.

🚨 **This refutes the premise that forces every turn onto VIO.**
`docs/night-motion-options-20260811.md` §1 says `toward` is "blind to in-place
rotation **at any hour of the day**", which is why `turn_mode: legacy` is
considered dead and why closed-loop motion is daylight-only. **The night path may
be a legacy-style turn, not an arc controller** — and that primitive already
exists.

Corroborated by the vendor: a 184 s passive sample of the mower **mowing in the
dark** shows 21 course changes with <0.15 m of travel against 6 while
translating — straight rows and in-place pivots, ~157° in ~0.4 m at each row end.
🗑️ That also **refutes my own inference that the vendor arcs**.

### ⚠️ Before anyone builds on this

- **n = 2**, and the two rates differ 50% (41.3 vs 27.3 °/s — explained by 8 vs 3
  refresh writes, which is an explanation, not a measurement).
- **No closed loop has run on `toward`.** Reporting correctly ≠ steerable.
- **`toward`'s latency DURING rotation is unmeasured** — these are settled values
  read after the pulse. That is what decides whether a loop can close.
- **Why 2026-08-02 disagreed is inferred**: the stale-feed defect (fixed in
  beta45, and it fooled this session for five minutes) and angular 180 both fit.
  Nobody has re-run that test.

### Next on this thread

1. **Re-run the 2026-08-02 in-place turn test** with the readback fix, to settle
   the contradiction rather than infer it.
2. **Sample `toward` DURING a pivot** at ~200 ms, not after.
3. Only then a night turn mode — and it needs its own gate story; the
   `vio_active` refusal exists for good reasons.

## 🔑 (still valid) 2026-08-12 — ARCS WORK, and steering is linear

Read `docs/arcs-work-20260812.md`. Two armed pulses, back to back, same speed and
window, one variable apart:

| | command | displacement | course change |
| --- | --- | --- | --- |
| **arc** | `linear 400, angular 180` | **0.5823 m** | **+22.20°** |
| **linear** | `linear 400, angular 0` | 0.5840 m | +0.00° |

1.8 mm apart in distance, 22.20° apart in course. Implied arc radius **1.512 m**.

🔑 **`toward` tracked the rotation exactly** — 22.20° of course change for 22.20°
of turn. That was the open question, and it is the mechanism a night controller
needs: translation keeps course-over-ground live, and a live `toward` closes a
heading loop with **no VIO**.

🔑 **"Angular needs 500" is a STATIONARY-only finding.** Angular 180 actuated
fine in an arc. The 2026-07-25 A/B measured a pure in-place turn; a stationary
skid-steer must break static friction on both tracks, a moving one only needs a
differential. Do not carry 500 into arc work.

⚠️ **The probe was blind to its own motion and nearly produced a false negative.**
The arc reported four bit-identical samples; the true position reached the cache
~5 minutes later. Fixed in beta45 — the probe now forces
`request_reports_count_5` and waits for the settle, like every real executor
already did. **Bit-identical telemetry is a stale feed, not a stationary mower**
— written down twice before and still nearly missed.

### Next on this thread

1. **Arcs at two or three more angular speeds** for radius vs angular. Cheap,
   daylight, one battery. An `arc_to_point` controller needs that curve.
2. Then closed-loop `arc_to_point` — but only if `toward` tracks when sampled
   DURING an arc, not just settled after it. Everything so far is open-loop.
3. Nothing has been tested at night yet, deliberately: both pulses ran in
   daylight so VIO was available as independent ground truth.

## 🏁 (still current) 2026-08-12 evening — GATE 5 RE-PASSED ON THE REACH PROFILE

Read `docs/gate5-repass-PASSED-20260812.md`. Card-driven, four segments, all
`target_reached`, landings **0.0674 / 0.1032 / 0.0807 / 0.0607** against 0.15 —
**mean 0.0780, the best four-segment result on record.** Zero reverse-recovery,
zero budget exhaustion, zero failed gates. Evidence:
`docs/evidence-gate5-repass-2-20260812.json`.

**`max_linear_pulse_ceiling: 14` is now HARDWARE-ACCEPTED.** The real payload
carried it, so profile identity is proven in fact and per-click reach of ~16 m
sits behind an accepted profile rather than a harness override.

### ⚠️ Three things this gate does NOT prove — do not overstate it

1. **beta43 was not exercised.** It raised the post-turn correction budget 2 → 4
   to fix the refusal that failed the first attempt (29.647° at tol 10 needing 3
   commands). This run's only correction was −10.477°, inside the OLD 21.50°
   envelope. The 29.6° case is still replay-only.
2. **The first attempt's failure did not recur naturally.** Its segment 3 had a
   54° opening turn and −29.647° post-turn error; this run's saw −4.711°. The
   geometry differed.
3. **The ceiling never bound** — 0.8 m legs used 2–3 pulses of 14. Reach is
   evidenced by `docs/loop-to-tolerance-reach-20260811.md`, not by this gate.
   Deliberate: long legs are 5/7 on control-law grounds, short legs 28/28.

### The build

`0.6.4-beta44` supersedes the beta43 that carried the gate. It adds nothing to
motion — it updates the card's execution-profile label (the re-pass is no longer
"PENDING") and **echoes every accepted-profile key from both executors**, because
two keys came back `null` and the gate had to be argued around the hole:
`motion_refresh_interval_ms` was provable from the per-segment echo and the
delivered writes; `max_no_progress_pulses` was not, and was dismissed only as
un-exercised. Pinned by a test that reads the card's frozen profile.

### Next

The five gates are complete and Gate 5 has passed twice — once fixed-budget
(2026-08-08), once reach-enabled (2026-08-12). **Remaining work is capability,
not gates.** The open items, in the order the evidence supports:

1. **The final-approach wall**, unchanged and now well characterised —
   `docs/cross-track-is-set-at-the-start-of-the-leg-20260812.md`, including its
   §8 refutation. Two genuine control failures in 39 approaches, both ~2 cm
   outside tolerance. **Do not tune the control law on n = 2.**
2. **A "Download JSON" button on the card.** The record was the hard part twice
   in one day — a wrong file once, a 50k paste truncation twice. `_persistLastRun`
   also swallows quota failures silently. Card-only.
3. **`last_travel_heading()` should read the last pulse, not the whole leg.**
4. **The arc probe** (`docs/night-motion-options-20260811.md`) — the only route
   to night capability, and still only an existence check away.

## (superseded) START HERE 2026-08-12 afternoon — the defect is the TRIGGER, not the guard

Two runs on beta42. **Run A passed**, Run C failed, and the analysis that came out
of it is the most useful thing today produced. Read
`docs/cross-track-is-set-at-the-start-of-the-leg-20260812.md`.

| run | geometry | result |
| --- | --- | --- |
| `…185603Z` | 2 × 2.0 m, 60° junction | **both segments reached**, 0.1030 / 0.0912, mean 0.0971 |
| `…191850Z` | 3 × 2.5 m | seg1 `target_requires_reverse_recovery` at 0.1715 |

### 🔑 The cross-track is set in the FIRST pulse and barely changes

Both failing segments, traced pulse by pulse, show a near-constant perpendicular
miss — Run C's segment 1 was already missing by **0.1963 m against a 0.150
tolerance after pulse 1**, with 1.87 m still to run. The *angle* triples over the
leg only because the range collapses. Nothing objected until pulse 4.

**The mid-drive re-aim triggers on an ANGLE (18°), which is range-blind.** At 3 m
that permits 0.93 m of cross-track; at 0.3 m it permits 0.09 m. The trigger is
most permissive exactly where an error is cheapest to fix and most costly to
keep. That is backwards, and it is why long legs fail more.

**The guard already computes the right quantity** — `distance · sin(aim)` against
`waypoint_tolerance`. Trigger and guard ask the same question in different units,
which is why a band exists where the trigger never fires while the miss is
already unacceptable.

### ⚠️ beta42's guard fix is correct and nearly irrelevant

Using the failing pulse's *actual* measured travel, the pulse-overshoot term
explains **~4 mm of a 42 mm error**. It stays — it is free and right — but it was
not the mechanism. Do not credit it with Run A's pass either: both guard decisions
in that run were ones where the old and new rules agree.

### The measured picture, so it is not re-derived

- **Heading drift while driving straight is small**: n=46 pairs, mean |drift|
  2.84°/pulse.
- **Bearing-to-target rotation is a 1/d effect**: median 1.47°/pulse beyond 1 m,
  **33.73°/pulse inside 0.3 m** (max 136°). Any decision made from an aim error
  inside ~0.5 m is stale before its own pulse finishes.
- **The segment is decided by 0.5 m out**: perpendicular miss ≤ 0.113 m at that
  range → 6 of 6 reached; ≥ 0.147 m → 4 of 6 missed.

### ⛔ Then that remedy was REFUTED the same afternoon — §8 of the same doc

**Do not implement the trigger change. Two independent checks killed it.**

1. **The turn primitive cannot deliver it.** 65 corrections on record: smallest
   rotation ever **4.93°** (once), median **35.23°**. Fixing a 6° aim error at
   1.87 m needs ±4.6°. At the median it would turn a 0.196 m cross-track into
   ~0.91 m. Corrections only help where required precision ≥ deliverable, i.e.
   **d ≲ 0.86 m**, and inside that window the existing 18° trigger already fires
   at about the right cross-track.
2. **Initial cross-track does not predict the landing.** Over 39 approaches,
   r² = 0.068; segments that REACHED had up to **0.4218 m** of initial
   cross-track, segments that MISSED as little as 0.0928 m. Run A segment 2
   started 0.4218 m off-line and landed **0.0912 m** because a correction fired
   at 0.797 m and worked.

### 🔑 The real picture: 33 of 39 approaches reached

```
20260809T195940 seg2  0.5830  command_failed          BLE
20260812T001116 seg1  0.5493  vio_realign_incomplete  BLE
20260809T210241 seg3  0.1641  max_linear_commands     old fixed-budget profile
20260810T002506 seg3  0.2548  max_linear_commands     beta36 guard, since fixed
20260811T235133 seg2  0.1797  reverse_recovery        genuine
20260812T191850 seg1  0.1715  reverse_recovery        genuine
```

**Two genuine control failures in 39, both ~2 cm outside a 15 cm tolerance.**
beta38/39 6/6, beta40 8/8. Short legs 28/28 on control grounds, long legs 5/7.

### Next: STOP TUNING. Run Gate 5.

Two plausible mechanisms were proposed and refuted in one afternoon — beta42's
guard fix (worth 4 mm of a 42 mm error) and the early-correction trigger
(actively harmful). There is no defect here big enough to justify another
motion-law change on n = 2.

**The outstanding obligation is Gate 5 on the adopted profile, card-driven.** It
must come from the CARD, not the harness — a harness run cannot close the
profile-identity gap (`docs/p0-beta-release.md:98-102`). The regime it exercises
(short-to-moderate legs) is the one running 28/28.

Reach is unaffected and still solved: 2/3/4 m single legs at 0.0690 / 0.0928 /
0.1023 m. Reach and final-approach accuracy are separable and only the first is
done.

## 📋 (done) THE PLAN FOR 2026-08-12 — PHASE 0 IS DONE AND DEPLOYED

✅ **`0.6.4-beta42` is on the host, motion-disabled, verified** (46/46
byte-identical, both card paths `09a1d05e`, resource
`?v=0.6.4-beta42&build=09a1d05e`, `real_motion_allowed: false`). It carries the
re-aim guard fix **and** the adopted reach profile. Deploy record in
`docs/deploy-runbook-p0.md`.

~~🚨 THE PROFILE IS NOW UN-ACCEPTED.~~ **RESOLVED THE SAME DAY — the Gate 5
re-pass passed, see the top of this file.** `max_linear_pulse_ceiling` moved
`null` → **14** on the operator's decision, which un-accepted the profile and
owed a fresh card-driven Gate 5. That gate ran on 2026-08-12 and passed 4/4, so
the key is accepted and the card's label no longer says "PENDING". Kept because
it records that the debt existed and was paid, not skipped.

Order matters: **A before C.** Each is single-variable, and A's failure would
invalidate C.

### Phase 0 — DONE 2026-08-12, kept for the record

✅ **0.1 The re-aim guard projects to the end of the next pulse.** It answered
`distance · sin(aim)` — the miss at *closest approach* — but the mower drives a
whole pulse and sails past that point. With the planner aiming the next pulse at
the remaining distance, the projection collapses to the chord
**`2·d·sin(a/2)`**: you drove the right distance in the wrong direction. Derived,
no new constant. `_projected_landing_after_next_pulse` in `services.py`.

⚠️ **This plan's own advice about the estimator was WRONG, and the replay caught
it — do not re-derive it.** It said to prefer "the segment's own measured
pulses". A full measured pulse badly over-estimates the *next* one, because the
final-approach logic shortens the pulse near the target. Replayed, that version
fixed 1 real miss and **broke 5 good suppressions**. The correct input is the
executor's own planned step, `min(remaining, metres_per_pulse)`, now produced by
a `_effective_metres_per_pulse` helper the planner and the guard **share** so
they cannot disagree.

Net effect on the record: fixes the one suppression that missed, and newly
corrects two at 52.0° and 54.2° of aim error that had scraped inside — an
accepted cost with a test named for it. ⚠️ The fix clears tolerance by **1.1 mm**;
it pins which side of the boundary a decision is on, not where the mower lands.

✅ **0.2 `realignments_suppressed` now carries `facing`/`bearing`**, plus the
projection and the metres-per-pulse it used.

❌ **NOT done: `last_travel_heading()` still reads the whole leg, not the last
pulse.** Still worth doing; still costs a `--heading` round trip after any
mid-drive correction.

✅ **0.3 Shipped and deployed.** See the deploy record above.

### Phase 1 — first light, two runs, A then C

Undock first: the dock sits ~1 m outside the polygon. **Drive to ~(6.5, −1.0)** —
room in every direction, and it is where the beta39/40 runs started.
`one_touch_leave_pile()` would do the undock without a walk, but it is not wired
up yet.

⚠️ **`--pulse-ceiling` is no longer needed and passing it now LEAVES the accepted
profile** — 14 is the profile value, so an override is the deviation. The harness
prints which case it is in.

**Run A — the guard fix, single-variable. Repeat last night's exact geometry.**

```sh
.venv/bin/python scripts/beta32_validation_run.py --leg 2.0 --segments 2 \
    --heading <TRUE FACING>
```

This is the highest-diagnostic run of the day and it must go first. That exact
path failed at a suppression with **3.1 mm** of margin (projected 0.1469 against
0.150, landed 0.1797, `target_requires_reverse_recovery`). **Pass = segment 2
corrects instead of suppressing, and reaches.**

⚠️ Not perfectly single-variable any more: the ceiling also moved 10 → 14. It
did not bind on that run (5 pulses used), so the geometry should behave
identically — but say so rather than assuming it.

If segment 2 still suppresses, read `projected_landing_m` in
`realignments_suppressed` (new in beta42) against `waypoint_tolerance` before
touching anything: it says exactly what the guard decided on.

*(The 180° staged-turn run is **dropped** — it existed mostly to walk the mower
back toward the middle of the yard, which `lawn_mower.dock` and
`one_touch_leave_pile` do better. Staged turn ahead of a long leg is still
untested and worth a run eventually, just not at the cost of daylight.)*

**Run C — the goal, in miniature: click across the yard.**

```sh
.venv/bin/python scripts/beta32_validation_run.py --leg 3.0 --segments 3 \
    --heading <TRUE FACING>
```

~9 m of path over three segments with two junctions. **This is the first run that
would look, to the operator, like the thing the project is for.** It needs A to
have passed, because segment-to-segment inheritance of end-of-leg mis-pointing is
what killed the 90° run on 2026-08-10 and the 2 m run tonight.

**Run D — the arc probe, if light and battery remain.** Read
`docs/night-motion-options-20260811.md` first. We have **never sent a command
with both axes non-zero**, though the wire has always accepted it, and an arc
keeps the mower translating — which keeps `toward` live and closes a heading loop
**with no VIO at all**. That is Phase 1 of the vendor's own two-tier design, the
thing we watched dock the mower in the dark.

```
raw_pymammotion_motion_probe   linear_speed=400  angular_speed=180  single shot
```

Zero code, ~10 cm of travel: an existence check that both axes actuate together,
not a measurement. **Run it in DAYLIGHT even though the goal is night capability**
— VIO is the only independent ground truth for heading, and a night-only
controller with nothing to check it against is how you ship something plausible
and quietly wrong. If it actuates, the next step is one schema key
(`motion_refresh_interval_ms` on that probe) and then the real question: **does
`toward` track an arc closely enough to close a loop on?** If it lags or noises
out beyond ~5°, the whole direction is dead — and that is a cheap thing to learn.

*(Also unused and night-capable: `one_touch_leave_pile()` would undock without the
manual step above. Not wired up.)*

### Phase 2 — GATE 5 RE-PASS. This is the milestone now.

The adoption decision is **made** (operator, 2026-08-12) and the §4 re-pinning is
**done** — card profile, payload, frontend pin, README, execution-profile label,
`CARD_VERSION`, and the harness mirror. What remains is the part that cannot be
done off-mower: **Gate 5, card-driven, on the reach-enabled profile.**

Until it passes, the profile is adopted but unaccepted, and the card says so.
Run it after A and C pass — a Gate 5 attempt on a build whose guard fix has not
been validated would be spending the gate to learn something a cheaper run
teaches.

**Drive it from the CARD, not the harness.** The whole point of
`LUBA_ACCEPTANCE_PROFILE` is proving the card emits what was accepted; a harness
run does not close that gap (`docs/p0-beta-release.md:98-102`).

### Hygiene, learned tonight

- **Measure `vio_tracked_features`, never `camera_brightness`.** The latter sat
  latched on `light` for ten hours today and only flipped to `dark` at 20:36.
  Features went 80 → 0 in ten minutes; check them before every run, not once.
- **Watch BLE across the session, not just at preflight.** It degraded −48 → −72,
  dropped the client entirely once, and killed a run outright. If runs start
  failing on BLE, restart the link rather than tuning anything.
- **Expect a `--heading` round trip** after any app-driven move; that is the
  facing guard working, not a fault.
- **Budget battery by run length**, not by run count: tonight's short runs cost
  ~1% each, not the ~7% the four-segment runs cost.

### What the four runs settled

**Loop-to-tolerance lifts per-segment reach from ~1 m to 4 m.** Full analysis:
`docs/loop-to-tolerance-reach-20260811.md`.

| run | leg | pulses | landing | stop |
| --- | --- | --- | --- | --- |
| `…235133Z` seg1 | 2.000 m | 5 of 10 | **0.0690** | `target_reached` |
| `…235133Z` seg2 | 1.942 m | 5 of 10 | 0.1797 | `target_requires_reverse_recovery` |
| `…235945Z` seg1 | 3.000 m | 8 of 12 | **0.0928** | `target_reached` |
| `…001116Z` seg1 | 4.000 m | 9 of 16 | 0.5493 | `vio_realign_incomplete` (BLE) |
| `…002804Z` seg1 | 4.000 m | 11 of 16 | **0.1023** | `target_reached` |

All three good segments **stopped on tolerance, not on the ceiling**. The
counterfactual is each segment's own third row: on the accepted profile they sit
0.7489 / 0.6777 / 1.7919 / **2.9543** m short on `max_linear_commands_reached`.
Per-click reach goes ~4 m → **~16 m** at 4 segments.

⚠️ **4 m is a demonstrated floor, not a limit** — the ceiling has never bound.
Where reach actually breaks is unknown; the 4 m run took 11 pulses over ~108 s,
so patience or link stability may bind before the control law does.

🔑 **The loop is robust to BLE stalls, and that is the bigger finding.** The 3 m
leg ran through two 2-write pulses (4158 ms and 2847 ms windows) that travelled
0.2325 / 0.2016 m against 0.34–0.49 m for the 15 cadence-intact pulses, and still
landed at 9.3 cm — it just took two more pulses. Against `max_linear_commands: 3`
those stalls are fatal. **BLE latency stops being a blocker for reach**, though
it still degrades the rate estimate. ⚠️ n = 2 stalled pulses; that is the shape
of the effect, not a calibrated number.

### ⚠️ Do NOT read these as profile results

Both runs passed `--pulse-ceiling`, i.e. `max_linear_pulse_ceiling`, a frozen
`LUBA_ACCEPTANCE_PROFILE` key the card sends as `null`. Every other key was sent
at its accepted value. **The landings do not compare to Gate 5.** Adopting the
key un-accepts the profile and owes a fresh Gate 5 — which is now the next
genuine milestone, and the reason to measure first was exactly this.

### The one open defect, and it is well characterised

Segment 2 of the 2 m run failed on **cross-track, not reach**. The beta38 re-aim
guard suppressed a correction at a projected 0.1469 m miss against the 0.150 m
tolerance — **3.1 mm of margin** — and it landed 0.1797 m out.

The guard projects the miss at the **closest approach**, but the mower drives a
whole pulse and finished 0.0877 m past that point. In quadrature that predicts
0.1711 m, which **exceeds tolerance, so the correction would have fired**.
Replayed over all 13 recorded suppressions the extra term cuts mean error
0.0212 → 0.0147 m, and the guard under-predicts on **11 of 13**.

⚠️ This is **not** the fitted margin dropped on 2026-08-10, and that drop was not
wrong — its rationale (the bias came from post-turn error, which beta40 fixes at
source) holds on the 0.7 m legs it was written about, where beta40/41
suppressions land within ~±1 cm. It is incomplete at long legs, where one pulse
is a large fraction of what remains. beta40 fired correctly on this very segment
(18.139° → −5.363°) and the guard still missed by 3.3 cm.

**Not implemented.** It changes the motion control law; give it its own review.

### 🚨 A harness bug left the gate OPEN once tonight — fixed, but read this

`c196b8b1`. `armed` was set **after** the post-enable readback, so when the
enable succeeded but `real_motion_allowed` came back false — which is what
`ble_client_not_connected` produced when BLE dropped between preflight and arm —
the script returned early saying it had aborted "without sending anything" and
the `finally` declined to disarm. It left `enabled: true` behind, one BLE
reconnect from an unattended open gate. Caught and disarmed by hand within the
minute; verified against a pre-fix copy that the new test catches it.

Second defect, same commit: the preflight passed all eight entity-derived checks
— `ble_link_live` still read `on` — while the backend's `blockers` already said
`ble_client_not_connected`. The gate's own blocker list is now a hard check.

**Rule this reinforces:** any script that can open the gate must treat "I called
enable" as the thing that obliges a disarm, never "enable succeeded".

### ⚠️ `last_travel_heading()` is wrong whenever a leg is not straight

It fired the facing guard at 23.8° with nobody having moved the mower, and the
**mirror was right**. It returns the straight-line start→target bearing of the
last segment (340.00°), but that leg had a mid-drive realignment at pulse 6 that
left the mower facing ~316.04° for its final two pulses; the mirror said 316.23°,
agreeing to **0.19°**. Long legs cause mid-drive corrections, so this gets worse
exactly where reach now lives. **Fix: take the bearing of the last pulse from the
final two samples, not of the whole leg.** Script-only, not implemented. It is a
false *alarm*, not a false negative — it costs a `--heading` round trip.

### The facing cross-check shipped and earned itself immediately

`scripts/beta32_validation_run.py` now derives the facing twice — the mirror of
the live `toward` and the last leg we drove — and **refuses to build a path when
they disagree past 15°**, handing back the mirror value to pass as `--heading`.
The operator repositioned from the app before run 1 and it fired at **94.3°**,
the exact condition that cost two runs on 2026-08-10. Run 2, with no app move in
between, agreed to 6.03° and needed no override. Pinned by 18 tests, including
the mirror against all 7 recorded calibration drives (worst residual 2.738°).

### Off-mower queue, in priority order

1. **The re-aim guard's projection** (above). Highest value: it is the only thing
   between here and a multi-segment long-leg path.
2. **Decide whether to take `max_linear_pulse_ceiling` into the profile.** That
   is a Gate 5 re-pass, and it is an operator call, not a commit.
3. **`last_travel_heading()` should read the last pulse, not the whole leg**
   (above). Cheap, and it removes a `--heading` round trip from every long run.
4. **`realignments_suppressed` lacks `facing`/`bearing`**, unlike `realignments`.
   Small instrumentation gap; it cost a reconstruction step in tonight's analysis.
5. **BLE session degradation** — `docs/pymammotion-ble-slot-leak-bug.md`. Demoted
   for reach by tonight's robustness result, but it still killed one run outright
   and the link deteriorated across the session (−48 → −72, then a full client
   drop while stationary, then recovery). ⚠️ The tempting "rssi tracks distance
   from the house" story **does not hold** — after driving the mower back to the
   middle of the yard it read −68 at 8 m, worse than −64 at that same distance an
   hour earlier. Elapsed session time fits the record better than distance.
5. ⚠️ **NOTHING IS PUSHED.** The branch has no upstream. `origin` is the Chorty
   fork; the `mikey0000` remote has its push URL disabled.

## (superseded) START HERE 2026-08-11 — beta41 deployed. One daylight run answers the reach question.

Host and branch both `0.6.4-beta41`. **Gate DISARMED.** Mower is **DOCKED** at
(4.3188, 3.2862) and charging; it was at 42% when docked. It is **dark**
(`tracked_features: 0`, `brightness: dark`), so nothing closed-loop can run until
daylight — see "night" below.

### The one thing to do next, and it is cheap

**Measure whether loop-to-tolerance lifts per-segment reach past ~1 m.** One run,
two 2 m legs, ~4 m of travel. Everything for it is built, deployed and committed.

```sh
set -a && source .env && set +a
# undock first: the dock is ~1 m outside the polygon
.venv/bin/python scripts/beta32_validation_run.py --leg 2.0 --segments 2 \
    --pulse-ceiling 10 --heading <TRUE FACING>          # preview
#   ... then --arm
```

**Aim the first leg along the mower's actual facing** so the opening turn is ~2°
and the LINEAR phase is the only variable. Prediction under test: on the accepted
profile a 2 m leg dies around 1 m on `max_linear_commands_reached`; with
`--pulse-ceiling 10` it should loop to the waypoint in ~5–6 pulses.

⚠️ `--pulse-ceiling` sends `max_linear_pulse_ceiling`, a frozen
`LUBA_ACCEPTANCE_PROFILE` key the card sends as `null`. The harness prints a loud
warning: that run is **NOT** the accepted profile and does not compare to Gate 5.
This is deliberate — measure first, so nobody pays a Gate 5 for an unknown.

### 🚨 How to get the mower's TRUE facing — this cost two runs on 2026-08-10

`last_travel_heading()` reads the last **driven** leg out of the evidence files.
That is sound until **the operator repositions the mower from the app, which
invalidates it silently and the script cannot detect it.** It happened three times
in one session; twice it built a backwards path and the run was refused with
`turn_budget_infeasible` at a ~177° opening turn.

**Use the mirror relation on a FRESH `toward` instead**, and cross-check:

```
map_facing = (90.13 - toward) % 360
```

Validated against both of 2026-08-10's calibration drives to **0.179°** and
**2.677°**, and independently corroborated by the operator eyeballing "South".
Cross-check it against the last driven leg's travel bearing; on 2026-08-10 night
those agreed at 325.63 / 328.34 (legs) vs 327.44 (mirror). **If the two disagree
by more than a few degrees, the mower was moved — trust the mirror, and only if
`toward` is fresh (i.e. it was just driven).**

⚠️ `toward` is a COMPASS bearing, so compass = `90.13 - map_facing`. Useful for
talking to the operator: map 130° is compass 320° = northwest.

Fixing this in the harness is the top off-mower item (below).

### Night: closed-loop motion is refused, and this is settled

`vio_active` is appended as a safety gate **only when VIO is cold**
(`services.py`, the `turn_mode == "vio" and vio_state != ACTIVE` branch), and it
passes only when the scene is bright enough that the calibration drive can warm
VIO. In the dark it refuses. A whole night was already lost to this on
2026-08-07: 11 of 12 gates passed and this was the 12th. **Do not try to bypass it
with `--skip-preflight-gate`** — that only skips the harness's own check; the
backend gate still refuses. Night motion is linear-only via Nudge, which is capped
at 2 m and does no closed-loop approach, so it cannot answer the reach question.

In daylight VIO is usually already warm and the gate is never even created — a
healthy run shows **11** safety gates, not 12.

### What shipped 2026-08-10, all deployed and hardware-validated

- **beta40** — the post-turn alignment gate gets its own tolerance,
  `_POST_TURN_ALIGNMENT_TOLERANCE_DEGREES = 10.0`, instead of borrowing
  `vio_realign_threshold_degrees` through a `min()`. At the old `min(18,15)=15` it
  never fired. **10 is a floor, not a preference** — a correction fires only when
  the error exceeds the tolerance, so the worst sweep is `error + tolerance`, and
  the affine bound `40 °/s·t + 12°` at the 200 ms actuation floor still sweeps 20°.
  The guarantee needs `error + tolerance ≥ 20`, i.e. **tolerance ≥ 10**. Tightening
  further needs a shorter actuation floor or a tighter sweep bound, **not** a
  smaller number.
- **beta41** — a segment's **opening turn decomposes instead of refusing**
  (`_vio_turn_to_heading_staged`). Tries the direct turn first and, only on
  `turn_budget_infeasible`, splits into stages of ≤60°. Validated on hardware: a
  **165.048° opening turn** completed in three stages (60/60/41.25), total staged
  displacement **0.1326 m of a 0.30 budget shared across stages**. Gap 2 closed —
  a user can click a point behind the mower and it turns and goes.

### The three beta40/41 runs, for anyone re-deriving

| run | landings (tol 0.15) | mean |
| --- | --- | --- |
| `…20260810T205937Z` | 0.0585 / 0.0867 / 0.1393 / 0.0979 | 0.0956 |
| `…20260810T232848Z` | 0.1396 / 0.0489 / 0.0117 / 0.0960 | **0.0741** |
| `…20260811T001250Z` (staged) | 0.1323 / 0.0591 / 0.1260 / 0.1467 | 0.1160 |

**Final map-frame aim predicts the landing: Pearson r = 0.923, r² = 0.852 (n=6).**
The segment that arrived at 2.07° landed **1.2 cm**. Corrected and uncorrected
segments land the SAME on average (0.0809 vs 0.0793) — the gate is not magic, it
moves segments down the aim axis and where they land on that axis is what matters.

**The landing wall is the turn's own translation** — see
`docs/turn-translation-explains-the-landing-wall-20260810.md`. Identity holds
across **14 segments / 4 runs, mean residual 0.46°**. Consequence:
`heading_tolerance_degrees` is the WRONG lever (it governs the VIO-frame error,
already ~5°); the landing is set by the map-frame error.

### Off-mower queue, in priority order

1. **Fix the heading trap** (above). Cross-check `last_travel_heading()` against
   the mirror on a fresh `toward` and refuse to build a path when they disagree.
   Would have caught both 2026-08-10 failures before any motion. Script-only.
2. **`realignments_suppressed` lacks `facing`/`bearing`**, unlike `realignments`.
   Small instrumentation gap; costs an analysis reconstruction step.
3. **BLE write latency** — standing item, `docs/pymammotion-ble-slot-leak-bug.md`.
   ⚠️ It HAS aborted a run since beta35 — the 2026-08-12 4 m run, on a queue
   deadline. 1 of 15 runs since beta35 (~7%); 2 of 20 overall. Also degrades
   the rate estimate.
4. ⚠️ **NOTHING IS PUSHED.** The branch has no upstream configured. `origin` is
   the Chorty fork; the `mikey0000` remote has its push URL disabled. Today's work
   exists only on the dev machine.

### If the reach run works

Adopting `max_linear_pulse_ceiling` changes a frozen profile key, which
un-accepts the profile and **owes a fresh Gate 5**. That re-pass — Gate 5 on a
reach-enabled profile — is the next genuine milestone. All five acceptance gates
are otherwise complete (Gate 5 passed 2026-08-08); the remaining work is
capability, not gates.

## (superseded) START HERE 2026-08-10 night — beta40 DEPLOYED AND VALIDATED. Best run on record.

Host and branch both `0.6.4-beta40`. **Gate DISARMED**, verified after every run.
Mower undocked in Backyard Right at **(7.4419, −6.4162)**, MODE_READY, blades OFF
at 0 rpm, RTK Fix, battery **39%** and falling ~7%/run.

⚠️ **Light is fine — `brightness: light`, `tracked_features: 80` measured after
the run.** An earlier revision of this block said "it is getting dark, dock it";
that was inferred from the clock and never measured, and it was wrong. Measure the
VIO feed rather than guessing from the time of day.

⚠️ **BLE has drifted to −72 dBm** (it was −62 at the start of the session). The
documented wall is ~−76, so the link is into its marginal band — a run refused on
BLE from here is the link, not a regression.

### The result

Four segments `target_reached` in 102.8 s. Landings **0.0585 / 0.0867 / 0.1393 /
0.0979 m** against 0.15 — **mean 0.0956**, the best 4-segment result on record
(beta33's was 0.0981; the beta39 run of the same 60° geometry was 0.1312).
Evidence: `docs/evidence-beta32-4segment-20260810T205937Z.json`.

**The new post-turn gate fired once and behaved exactly as designed:**

| seg | map-frame aim | tol | corrected | landing |
| --- | --- | --- | --- | --- |
| 2 | +8.632° | 10.0 | no | 0.0867 |
| 3 | **−9.733°** | 10.0 | no — passed by 0.27° | **0.1393** ← worst |
| 4 | **−16.551°** | 10.0 | **yes → −7.331°** | 0.0979 |

Segment 3 is the accidental control: it squeaked under the threshold, went
uncorrected, and produced the worst landing. Segment 4's error came from the turn
*overshooting* (+34.784 → −17.430°), not translation — the gate catches map-frame
error whatever its source. n=1 each side, so suggestive, not proven.

**⚠️ The unmeasured risk is now measured and it is small.** The correction turn
displaced 0.0108 m: on a 0.636 m leg that is **0.97° of induced bearing error to
buy 10.038° of correction**, roughly 10:1. Not a non-issue in principle, but not a
blocker.

**The translation identity held on three more segments** — residuals +0.26 / +0.08
/ +0.03°. That is 8 segments across 3 runs now. See
`docs/turn-translation-explains-the-landing-wall-20260810.md`.

### 🗑️ The projection-margin idea is DROPPED — do not revive it

The suppression projections were accurate this run (0.0895 → landed 0.0867;
0.1390 → landed **0.1393**, 0.3 mm). Earlier they under-predicted by up to 4.9 cm.
**That bias came from the uncorrected post-turn error, which beta40 fixes at
source.** The margin would have padded a symptom, and its draft broke a
hardware-derived pinned test ("allowed, then oscillated") to do it.

### ⚠️ The stale-`toward` trap cost a run — read this before building any path

The first beta40 attempt was refused pre-dispatch: `turn_budget_infeasible`,
`initial_error_degrees: 177.056`. **`scripts/beta32_validation_run.py` builds its
path from `last_travel_heading()`, which reads `toward` — frozen
course-over-ground at rest.** It reported ~88° while the calibration drive then
measured the mower's true facing as **266.712°**, nearly opposite, so the path was
laid out backwards and segment 1 needed a 177° turn.

Refusal was correct (a single 180° turn is refused pre-dispatch — settled). The
workaround is `--heading <true facing>`; the calibration drive's
`map_motion_heading_degrees` is the most reliable source for it. Evidence:
`docs/evidence-beta32-4segment-20260810T205514Z.json`. **A real fix would derive
the preview heading from something other than a field known to be frozen when
stationary.**

### Next

1. **Watch the battery** — 39% against a 30% preflight floor is roughly one more
   four-segment run, not several. Dock before it gets close.
2. The 10° threshold is the **turn primitive's floor, not an optimum.** Segments
   at 8.6° and 9.7° still go uncorrected. Buying anything below 10 requires a
   shorter actuation floor or a tighter sweep bound — that is the next accuracy
   lever, and it is a turn-primitive question, not a profile question.
3. **The profile conflict is no longer the blocker it looked like.** Option A
   (`heading_tolerance_degrees` 18 → 11) targets the VIO-frame error, which was
   never the problem. No profile key has been changed and none is owed.
4. Fix the stale-`toward` path preview (above).
5. BLE write latency remains the standing background item.

## (superseded) START HERE 2026-08-10 evening — beta38 VALIDATED; the accuracy premise moved

Two armed runs on beta39, both authorized per-run, both disarmed and verified.
Mower is **undocked** in Backyard Right at (6.7995, −0.0455), MODE_READY, blades
OFF at 0 rpm, RTK Fix, battery ~50%. Gate **disarmed**.

| run | geometry | result | landings (tol 0.15) |
| --- | --- | --- | --- |
| `…20260810T185433Z` | 4 × 0.7 m, **90°** junctions | seg3 `turn_budget_infeasible` | 0.0679 / 0.1424 / — / — |
| `…20260810T193833Z` | 4 × ~0.65 m, **60°** junctions | **all four `target_reached`** | 0.1142 / 0.1431 / 0.1447 / 0.1229 |

### 1. The beta38 re-aim guard is validated — do not re-litigate it

Four suppression events across the two runs, **zero false suppressions**. Every
one had a `perpendicular_miss_m` genuinely under `waypoint_tolerance`, verified by
recomputing `distance · sin(aim)` independently:

```
0.3035 m @ 18.152 deg -> 0.0946    0.2390 m @ 34.655 deg -> 0.1359
0.2606 m @ 25.980 deg -> 0.1141    0.2017 m @ 34.733 deg -> 0.1149
```

And the other half works: a **−47.812°** aim error was correctly NOT suppressed,
dispatched, and corrected to −1.98°. That is precisely the case beta36 got wrong.

### 2. ⚠️ But the guard's projection is systematically optimistic

All four suppressed re-aims landed **worse** than the guard predicted:

```
predicted -> actual     error
0.1149    -> 0.1424    +0.0275
0.0945    -> 0.1431    +0.0486
0.1359    -> 0.1447    +0.0088
0.1142    -> 0.1229    +0.0087
```

The guard compares a biased-low projection against the **full** tolerance, so it
spends the entire error budget on every decision — which is why the 60° run's four
landings cluster at 0.114–0.145 rather than spreading. A margin change is drafted
for review; it touches **no `LUBA_ACCEPTANCE_PROFILE` key**. Four samples is thin
for fitting a constant, so the margin should be explicit and conservative, not
fitted.

### 3. 🔑 SOLVED — the turn's own translation is what puts the mower off-bearing

**Read `docs/turn-translation-explains-the-landing-wall-20260810.md`.** Derived
off-mower from the two committed evidence files; no motion was commanded for it.

A VIO turn does not pivot in place — it displaced the mower **0.028–0.131 m** on
these runs. Sideways displacement at the *start* of a 0.6–0.7 m leg rotates the
bearing to the target by `atan(translation / leg)`, and the turn primitive closes
on **VIO body heading** so it cannot see this: the mower's heading did not change,
the *target's bearing* moved. Across all five completed segments the map-frame aim
error minus the VIO-frame error equals `atan(translation/leg)` to within **0.02–
1.25°**. That is a mechanism, not a correlation.

It also repairs the landing model: fed the map-frame aim instead of the VIO-frame
aim, `0.62 × leg·sin(aim) + 0.065` goes from mean |residual| 0.0310 m to 0.0178 m
and loses its one-directional bias. **The model was never wrong — it was being fed
the wrong aim error.**

**So Option A is dead.** `heading_tolerance_degrees` governs the VIO-frame error,
mean 5.5°, already inside any tolerance under discussion; the landing is set by
the map-frame error, mean 8.0°, which that key does not control. Lowering it
18 → 11 would have changed **none** of these five segments.

**The real lever is not a profile key.** The post-turn gate is
`alignment_tolerance = min(heading_tolerance_degrees, vio_realign_threshold_
degrees)` = `min(18, 15)` = **15**, and all five map-frame errors (3.079–11.452°)
fell inside it, so no correction was ever attempted. Lowering
`vio_realign_threshold_degrees` 15 → ~5 makes it 5 and catches all five, touching
no `LUBA_ACCEPTANCE_PROFILE` key. The mid-drive trigger does **not** move (it
resolves to `aim > 18` either way). ⚠️ Unmeasured risk: a correction turn also
translates — small, but the same shape as the problem it fixes.

### (superseded by §3) The initial-aim model looked INCOMPLETE

`landing = 0.62 × leg·sin(initial_aim) + 0.065` does **not** explain these
landings. Segments 2/3/4 finished their turns at **−3.7 / −5.1 / +4.6°** and still
landed 0.1431 / 0.1447 / 0.1229 m against predictions of 0.089 / 0.102 / 0.097 —
under-predicting by 26–54 mm. The aim error **develops mid-leg**: −3.7° → +18.2°
in a single pulse, −5.1° → −34.7°, +4.6° → +26.0°. Part of that is genuine heading
drift and part is the bearing-to-target rotating as cross-track accumulates, and
**this run cannot separate the two** — that separation is the next real question.

**Consequences for the profile decision, which is therefore NOT ready:**

- **Option A (lower `heading_tolerance_degrees` 18 → 11) would have changed
  nothing here.** All three turns already finished inside 5.1°.
- **Option C (shorter legs) points the wrong way.** beta33 drove **0.9 m** legs at
  the same 60° junctions for a mean of **0.098 m**; this run's **0.6–0.7 m** legs
  averaged **0.1312 m**. Uncontrolled across six betas — a flag, not a refutation.
- A newly measured tension against Option A: the preflight computes
  `required_rotation = initial_error − heading_tolerance`, so **tightening the
  tolerance makes `turn_budget_infeasible` refusals MORE likely** against the
  fixed 4-command budget. The 90° run died exactly there (104.475° opening,
  86.475° required, 5 commands needed of 4). Option A likely has to move
  `max_turn_commands` with it — a second profile key and a wider Gate 5.

### 4. Why the 90° run died, and what it exposed

Segment 3 was refused pre-dispatch at a **104.475°** opening turn against a 90°
plan. The extra ~14.5° was ~9.8° of end-of-leg mis-pointing plus ~3.5° of bearing
shift from segment 2's 0.1424 m landing. **The guard optimises the current
segment's landing and is silent about the heading the segment ends in**, which the
next segment inherits as its opening turn. Landing inside the disc while pointed
35° off the line is a legal answer to the question the guard asks. At 60° the same
inherited error lands ~74°, inside budget — which is why the second run completed.

### Next

1. **Characterise the mid-leg divergence** — separate genuine heading drift from
   bearing rotation due to cross-track. Until that is done the profile decision
   rests on a model known to be incomplete.
2. **Review the drafted guard margin** (§2 above).
3. The profile conflict (§3) stays the operator's call and is **not** ready to
   decide.

## (superseded) START HERE 2026-08-10 afternoon — ready to run, one operator step first

Host and branch both `0.6.4-beta39`. ⚠️ **The motion gate is ARMED**
(`enabled: true`) at the operator's request — this is NOT the usual
disarmed-at-rest posture. `real_motion_allowed` is currently `false` only because
the mower is docked.

```
mower     CHARGE_ON (4.337, 3.360)   battery 58%   charging off
light     bright, tracked_features 80        RTK Fix        BLE −46
blockers  ['position_not_valid_for_motion']   <- docking, nothing else
```

**The one operator step: undock and drive it into Backyard Right.** The dock sits
~1 m outside the polygon; roughly (6.5, 0.0) gives comfortable margin. Everything
else is green.

### The run that is queued

```sh
set -a && source .env && set +a
.venv/bin/python scripts/beta32_validation_run.py --junction 90            # preview
.venv/bin/python scripts/beta32_validation_run.py --junction 90 --arm      # go
```

Four segments, 90° junctions, 0.7 m legs. It is a **single-variable test of the
beta38 re-aim guard** — the previous attempt (0.0943 / 0.0753 / **0.2548** m) had
its third segment ruined by the beta36 guard suppressing two real corrections,
and beta38 replaced the criterion. Nothing else changed since.

**What to look for:** `realignments_suppressed` should now be empty or carry only
entries whose `perpendicular_miss_m` is genuinely under `waypoint_tolerance`. If a
40°+ aim error is suppressed again, the criterion is still wrong.

### Then, and it needs a decision not a commit

The accuracy conflict below. **`heading_tolerance_degrees: 18` and
`waypoint_tolerance: 0.15` are geometrically inconsistent** — a 0.9 m leg needs
initial aim within 8.8°, and the turn is allowed to finish at 18°. Both are
`LUBA_ACCEPTANCE_PROFILE` keys, so it owes a fresh Gate 5. Shorter legs are the
no-key alternative.

## (context) 2026-08-10 — the accuracy wall is EXPLAINED, and it is a profile conflict

Host and branch both run `0.6.4-beta38`, deployed motion-disabled,
`real_motion_allowed: false` verified. Mower is out in Backyard Right around
(7.02, −2.12), battery ~53%, blades off, gate off. **It is dark — dock it.**

### The finding that matters most, and it needs an operator decision

Across **12 completed approaches** spanning every run of 2026-08-09/10, landing
error is predicted by the aim error at the *start* of the leg:

```
landing = 0.62 × leg·sin(initial_aim) + 0.065 m        R² = 0.69, n = 12
```

The 0.065 m intercept is the floor from stop latency and the ~1031 ms position
feed. The slope is the controllable part, and it says:

```
0.7 m leg → initial aim must be within 11.3° to land inside 0.15 m
0.9 m leg → within  8.8°
```

**`heading_tolerance_degrees` is 18.** A turn is therefore *allowed* to finish at
an aim error that geometrically guarantees a miss — 0.9·sin(18°) = 0.28 m of
cross-track against a 0.15 m `waypoint_tolerance`. **The two keys are mutually
inconsistent**, and that is the mechanism behind the ~0.145 m wall seen at Gate 5
and twice since; those segments simply started closer to aligned.

Both are `LUBA_ACCEPTANCE_PROFILE` keys, so changing either un-accepts the profile
and owes a fresh Gate 5. **Not acted on — this is the operator's call.** The
cheap alternative that touches no key is shorter legs, since the requirement
scales with leg length.

### What shipped on 2026-08-09/10

- **beta35** — refresh writes fire on a fixed cadence instead of
  interval-after-completion. Delivered-window overruns fell from +117% to +29%,
  and BLE aborts became rare. ⚠️ **"Has not aborted a run since" is REFUTED** —
  1 of 15 runs since beta35 died on a BLE queue deadline (2026-08-12).
- **beta36** — a mid-drive re-aim guard that was **wrong**; see beta38.
- **beta37** — the turn model rebuilt on 35 measured pulses:
  `_MIN_SCALED_TURN_PULSE_MS` 400 → 200 (200 ms actuates; there is no threshold
  near 400), the overshoot bound changed from `C×t` to the measured affine
  `40 °/s·t + 12°`, and `_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND` 16.5 → 14.4,
  which the first two changes finally made affordable.
- **beta38** — the re-aim guard, corrected to ask the right question: skip only
  when driving straight on would still land inside the disc
  (`distance·sin(aim) ≤ waypoint_tolerance`), not when merely close.

### Settled, so it is not re-derived

- **Rotation is not predictable from duration** to better than ~40% at p90. Ten
  pulses at matched ~200 ms windows spread 5.44–15.20°, 2.79×, with duration,
  cadence and direction held constant. The *estimate* can only improve a landing;
  the bound has to carry the safety.
- **The "directional asymmetry" is refuted.** Three runs gave 8/8, 1/1, then 1/6.
  Pooled over 33 samples the two directions differ by 0.5%.
- **A 90° junction dispatches and completes** — measured, 3 of 4 commands.
- **A single 180° turn is refused pre-dispatch**; chain junctions instead.

### Next

1. **Re-run the 4-segment / 90° / 0.7 m attempt on beta38** — a single-variable
   test of the corrected guard. Needs daylight, RTK Fix, and authorization.
2. **Decide the profile conflict above.**
3. BLE write latency remains the standing item, though it has not aborted a run
   since beta35.

## (superseded) START HERE 2026-08-09 evening — REACH GOAL MET. Four segments ran.

Supersedes the Gate 5 block below, which is now evidence rather than
instructions. **Host runs `0.6.4-beta33`; branch is at `0.6.4-beta34`, built and
CI-green but never deployed and never run.** Gate off, `real_motion_allowed:
false` verified. Mower idle in Backyard Right around (6.889, −1.122), battery
~45%, blades off.

Three real runs today, all armed and disarmed with both states verified:

1. `docs/evidence-beta32-4segment-20260809T170941Z.json` — segment 1 reached
   (0.0916 m), segment 2 refused pre-dispatch on a BLE queue backlog.
2. `docs/evidence-beta32-4segment-20260809T183129Z.json` — **all four segments
   reached target.** Landings 0.0819 / 0.0662 / 0.1452 / 0.0990 m.
3. `docs/evidence-beta33-reposition-20260809T184618Z.json` — a 180° U-turn built
   from three 60° junctions; the final 2.0 m leg stopped on
   `max_linear_commands_reached`, which was a planning error, not a defect.

**Settled today, so do not re-derive:**

- **Landing error does NOT compound with segment index** (seg4−seg1 slope
  +0.017 m). Handover §2.4's central worry is unsupported.
- **The overshoot ceiling works.** Three junction turns closed in one command
  each at −5.1 / −2.4 / −0.3°, against Gate 5's 13.258° overshoot.
- **Per-CLICK reach is 4 segments; per-SEGMENT reach is ~1 m** (`max_linear_
  commands: 3` × ~0.35–0.42 m/pulse). A 2.0 m leg is not dispatchable.
- **A single 180° turn is refused pre-dispatch** — 8 commands needed against a
  budget of 4, and 0.468 m of estimated drift against a 0.30 m cap. Largest
  dispatchable single turn is ~114°. Use chained junctions instead.
- ⚠️ **The rotation-rate variance is largely a BLE artefact.** See
  `docs/HANDOVER-beta31-20260809.md` §2.7 before touching any turn constant. A
  pulse that sent one of six refreshes, on a write that blocked 1303.972 ms,
  reported "9.23 °/s". Cadence-intact pulses measured 23–43 °/s. This
  substantially explains the Gate 5 overshoot, and means
  `_VIO_TURN_CONSERVATIVE_DEGREES_PER_SECOND = 16.5` is probably **not**
  falsified. Two recommendations are **withdrawn** there: "K = 2 × tolerance
  unlocks 90° L-paths" and the delivered-window shave.

### The queue, in priority order

1. **BLE write latency — the highest-leverage open item, and now doubly
   attested.** The 2026-08-08 block below already called the intermittent
   `TimeoutError` "the tail of an observable latency distribution … the basis for
   the next work item"; today's stall finding arrived at the same place
   independently. One 1304 ms GATT write manufactured a phantom rotation rate; a
   linear pulse ran +112% long (2757 ms against 1300). It degrades the rate
   estimate, the ceiling's guarantee, along-track accuracy and segment gating at
   once. Prior art: `docs/pymammotion-ble-slot-leak-bug.md`.
2. **The ~0.145 m landing-error ceiling.** Gate 5's worst was 0.1449 m; today's
   was 0.1452 m — 0.3 mm apart across independent runs, against a 0.15 m
   tolerance. That is a systematic mode, not noise. Linear-phase question, and
   plausibly the same BLE root cause.
3. **Test a 90° junction directly.** At the real rates (≥23 °/s) the shipped
   ceiling completes 169°, so the 45–70° band may be unnecessarily conservative.
   This is a measurement, not a code change — cheaper than any of the K work.
4. **Deploy beta34 and confirm `refresh_cadence_broken` fires** on a run with a
   stalled write. Built and green, never run.
5. **Extending per-segment reach past ~1 m** means enabling loop-to-tolerance
   (`max_linear_pulse_ceiling`). ⚠️ **Prerequisite:** the mid-drive re-aim guard
   tests `command_index < max_linear_commands` rather than
   `effective_linear_ceiling` (handover §5). Harmless while the ceiling is null;
   it must be fixed **first**, or cross-track correction silently stops after
   pulse 3 while the mower keeps driving.

## (superseded) START HERE 2026-08-09 morning — GATE 5 IS PASSED. All five gates complete.

⚠️ **Host and branch have diverged as of 2026-08-08.** The **host** runs the
deployed `0.6.4-beta30`. The **branch** is at `0.6.4-beta31`, which is built,
fully green on CI, and **never deployed or run on hardware**. Read
`docs/HANDOVER-beta31-20260809.md` before touching anything — it carries the
changes, the open attacks against them, and the validation-run design.

Gate off, no session, blades off. Mower
undocked in Backyard Right, battery ~22% (dock it).

**Gate 5 passed twice on 2026-08-08**, card-driven, with the accepted profile —
`docs/evidence-gate5-PASSED-20260808.json`. Four segment landings: 0.0485,
0.0836, 0.0558, **0.1449** m, all inside the adopted 0.15 m tolerance. The worst
would have failed at the old 0.08 m.

**`waypoint_tolerance` is now 0.15 in `LUBA_ACCEPTANCE_PROFILE`** (beta30), with
the full §4 re-pinning done: card profile, payload, frontend pin, README copy,
and the execution-profile label, which now reads
`LUBA acceptance profile (Gate 4 re-pass 2026-08-05; tolerance 2026-08-08)`.

### ⚠️ Two fragilities the pass does NOT remove

**Both were rewritten on 2026-08-08 after the raw per-command record was
recovered** from the card's response pane and committed as
`docs/evidence-gate5-attempt5-segment1-raw-20260808.json`. The earlier wording,
written from the prose summary, asserted a turn-budget exhaustion that **did not
happen**. Read that file before re-deriving anything here.

1. **The turn phase lands on tolerance with little margin — the budget is not the
   problem.** ~~Attempt 5 used `turn_commands_sent: 4` of `max 4`.~~ **REFUTED.**
   That 4 is three turn-phase pulses **plus one mid-drive realignment on a
   separate budget**; the turn phase itself stopped at `target_heading_reached`
   on command **3 of 4**, with a command in hand. `turn_commands_sent` is
   reporting-only — nothing in `services.py` ever reads it in a comparison — and
   the true per-segment ceiling is **14**, not 4.
   The real fragility is **overshoot against tolerance**: pulse 3 rotated 57.630°
   when 44.372° remained, overshooting the target heading by **13.258°** against
   `heading_tolerance_degrees: 18`. The clean finish depended on the outlier
   overshooting by less than the tolerance — **4.74° of margin**.
   On the 2.6× rate variance: it is partly an accounting artifact. Against
   *nominal* duration the pulses read 20.31 / 14.78 / 38.42 °/s; against
   *measured* elapsed they read 14.91 / 14.49 / 32.74, so pulses 1 and 2 agree to
   ~3% and **only pulse 3 is anomalous**. The artifact is proven:
   `services.py:8091` accumulates `observed_rotation_ms += pulse_ms`, the nominal
   duration, never the measured `elapsed_ms`. Pulse 3's rotation is nonetheless
   **real, not a measurement error** — see the five-step argument in the evidence
   file. Why it rotated 2.6× per delivered write is unexplained and is not
   observable from telemetry.
2. **The BLE `TimeoutError` is intermittent, not fixed.** It failed attempt 3 at
   a 80.6° turn; attempt 5 completed *larger* turns (93.5°, 82.9°). Attempt 5 ran
   with visibly degraded BLE — refresh writes median 540 ms (max 861) — without
   tripping it. **The timeout is the tail of an observable latency distribution**,
   which is a real diagnosis and the basis for the next work item.
   ⚠️ **Attribution corrected:** the stop confirmations 1175/1819/402/628 ms are
   the **calibration drive stop plus the three linear stops**, not turn stops.
   Turn pulses record no stop duration at all (`services.py:3321-3333`), so no
   rate table may add stop latency to turn-pulse elapsed time.

### The measured picture, for anyone re-deriving it

- Position feed updates ~**1031 ms**, confirmed twice — once from our pulsed
  motion, once from the mower's own autonomous dock return (median 1.15 s). It is
  a property of the device/link, **not** an artifact of how we drive.
- The mower travels **~29 cm median, 59 cm max** between position updates. Its
  own vendor navigation is positionally blind at that scale, which is why 0.08 m
  was never achievable.
- Stop latency: n=60, median 229 ms, p90 461, max 1393, stdev 186 — an 18×
  spread. **A fixed stop-lead constant cannot correct a variable**; that work
  item is dropped.
- Per-segment reach is **~0.9–1.5 m** (`max_linear_commands: 3`,
  `max_linear_pulse_ceiling: null`), and the card limits to 2 segments.

### (superseded) 2026-08-08 earlier — the daylight window has a queue; host runs beta29

**Host and branch run `0.6.4-beta29`** (all four version sites agree,
`0.6.4b29`). Gate **off**, no session, blades off, mower stationary and
`area_inside`. Worktree clean, pushed to `Chorty`.

### ✅ 2026-08-08 — SLOW-TIER VALIDATION PASSED. The tolerance question is answered.

Ran on beta29 in daylight. Evidence:
`docs/evidence-slow-tier-validation-20260808.json`.

| segment | landing error | travel ratio | linear cmds | turns |
| --- | --- | --- | --- | --- |
| 1 | **0.0882 m** | 0.917 | 3 | 0 |
| 2 | **0.1241 m** | 0.880 | 2 | 0 |
| 3 | **0.0317 m** | 0.978 | 2 | 0 |

All three `target_reached`, **zero turn commands**, no errors, no reverse-recovery
containment. **Max 0.1241 m, mean 0.081 m — every run inside 0.15 m.** The error
budget predicted ~0.15 m and the hardware agreed.

**The mechanism is now visible.** Every segment **undershot** (ratios 0.88–0.98).
Gate 4 at 0.08 m overshot at **2.19×** and needed a U-turn recovery — same control
law, same profile, only the tolerance differed. At 0.08 the mower cannot confirm
arrival before it is already past the point; at 0.15 it stops on the first
approach.

⚠️ **0.08 m is marginal, not impossible** — 2 of 3 runs would have met it (0.0317,
and 0.0882 just outside). Intermittent success is *worse* than reliable failure:
it produces sporadic recovery maneuvers, which is exactly what the beta20/beta21
Gate 4 passes did.

**Operator decision now unblocked:** adopting `waypoint_tolerance: 0.15` has
hardware evidence behind it. Changing it un-accepts the profile and obligates the
§4 pinning work in `docs/gate4-repass-20260805.md` (card copy, `CARD_VERSION`
bump to both serving paths, frontend pinning tests).

**VIO warm-up confirmed working:** the forward heading pulse took `vio_state`
0 → **2** with 80 tracked features, which is what let the `vio_active` gate pass
in daylight where it refused at night.

### (superseded) The test that was queued

**Slow-tier landing validation.** Plan is written and was operator-authorized:
`docs/plan-slow-tier-validation-20260807.md`. It was attempted on 2026-08-07
night and **refused before any motion command** — see
`docs/evidence-slow-tier-validation-20260807.json`.

⚠️ **Why it failed, so it is not repeated:** the `vio_active` safety gate keys
off `turn_mode == "vio"` **unconditionally**, not off whether a turn is actually
needed. `_VIO_TURN_MODES` was `("vio", "legacy")` only — there was no night-safe
mode — and the gate is `passed = dry_run or calibration_will_warm`, where
warming requires a bright scene. ~~**A closed-loop segment cannot run after dark,
by design.**~~ 11 of 12 gates passed; this was the 12th.

⚠️ **Superseded.** That symbol is now `_SEGMENT_TURN_MODES = ("vio", "legacy",
"night")` (`services.py:10939`), and night v1 is the night-safe mode this
paragraph says does not exist.

In daylight it should simply run. The prediction under test: a waypoint approach
whose final phase is slow-tier lands within **~0.15 m**. Everything else in the
plan (bounds, abort conditions, parameters, what each outcome means) still
stands unchanged.

**Run the turn-quantum work in the same window** — VIO is alive only in daylight,
and it is the other genuinely open empirical question.

### 🔑 What tonight settled, so it is not re-litigated

1. **RTK freshness is closed, permanently.** Two watch logs (90 + 152 samples at
   30 s) put maximum *legitimate* quiet at **3745 s (62.4 min)**. Against a ~3 h
   fault that is far too little separation. It has now been set too low twice
   (300 s, 1800 s) and measured too close a third time. **Do not revisit without
   an active probe** that can tell a quiet channel from a dead one — §2.1 of
   `docs/rtk-hardening-plan-20260807.md` shows none exists.
2. **The base station ANSWERS `request_basestation_info_t`.** An earlier "no
   response" was a wrong-read-path artifact. Baseline under Fix: `rtk_status` 1,
   28 sats, WiFi −72 dBm, own coordinates 34.0245718145 / −84.7698523612.
3. **…but `score_info` is `null`.** `base_moved` / `base_moving` — the survey
   discriminator — are **never populated by this hardware**. That avenue is
   **closed**; do not plan work around it.
4. **The correction chain is `internet source → base (WiFi) → LoRa E22 → mower`.**
   The base reports `position_mode: rtk_over_internet`; the mower reports
   `rtk_over_datalink`. The base does **not** primarily run on its own survey, so
   the survey hypothesis is **demoted**. An upstream outage degrades what the
   base relays while changing nothing the base reports about itself — exactly the
   2026-08-07 signature, and why a rover-side resync did nothing but a power
   cycle worked.
5. **The stop-lead (`speed × latency`) work item should be dropped.** 60 committed
   samples: median 229 ms, p90 461 ms, max 1393 ms, stdev 186 ms. An 18× spread —
   **a constant cannot correct a variable**. More driving yields more samples of
   a distribution already characterised.
6. **The mirror heading relation is confirmed on fresh data.**
   `map_bearing = 90.13 − toward` predicted 277.97° against a measured 278.86°
   travel heading — **0.89° agreement** — while the code's additive constant
   (116.5) mis-aimed by **9.80°**. Measured from a clean, attributable 0.45 m
   displacement.

### Tolerance: the analysis is done, the decision is the operator's

Two independent routes both land on **~0.15 m**:

| error source | fast tier (0.28 m/s) | slow tier (0.10 m/s) |
| --- | --- | --- |
| feed staleness (1031 ms median) | 21–33 cm | ~10 cm |
| stop latency, median 229 ms | 6.4 cm | 2.3 cm |
| stop latency, p90 461 ms | 12.9 cm | 4.6 cm |
| **realistic total** | **~0.30 m** | **~0.15 m** |

**0.08 m is arithmetically unreachable at the fast tier** — you cannot stop
within 8 cm when position knowledge is 21–33 cm stale, and beta23 proved the
device will not give a faster feed. Changing `waypoint_tolerance` un-accepts the
profile and obligates the §4 pinning work in `docs/gate4-repass-20260805.md`.

### Monitoring now running

`scripts/rtk_watch.py` (shipped, tested) logs **both ends of the chain** in one
record — mower fix state plus base `position_mode` / `wifi_rssi` / satellites /
coordinates. Entity prefixes auto-discovered; failed fetches written as explicit
gap records. It deliberately does **not** log `base_moved` (never populated; a
column of zeros would read as "has not moved").

## 🚨 START HERE, 2026-08-07 night — RTK freshness is UNVERIFIABLE; quality is the guard (beta26)

Two of the same day's conclusions were **refuted by measurement that evening**.
Read this before touching anything RTK. Full reasoning:
`docs/rtk-hardening-plan-20260807.md`.

**1. A forced report burst cannot detect a latch.** Repeating the afternoon's
test with RTK verifiably healthy and `Fix` gave 49 messages, **zero** RTK /
position / VIO channel updates, and the age still climbing 570.9 s → 601.1 s —
identical to the latched case. The afternoon's inference from that test was
invalid; there is **no positive liveness probe**.

**2. No staleness threshold works, and 1800 s was live in beta25.** A healthy
Fix-locked stationary mower reached **3573 s (59.5 min)** of unchanged RTK
payload. The payload changes ~hourly at rest while the one observed fault lasted
~3 h — a 3x separation from one sample each. 300 s and 1800 s both false-blocked.

**So the design inverted (beta26):** RTK payload age is **reported for auditing
and never blocks**. The real guard is the **quality gate** — non-Fix refuses with
`rtk_not_precise` unless `allow_degraded_rtk` is passed. That directly catches
the fault actually observed (a latched *Float*); freshness only ever addressed a
latched *Fix*, never seen on this hardware. **Do not turn age back into a
blocker** — the constant's comment records why.

**Good news, verified:** mining every evidence file found **534** recorded `Fix`
states and **zero** prior `Float`/`Single`. 2026-08-07 was the first episode, so
**past gate results are not tainted**. (265 `None` records are the dead
`mowing_state` candidate, not real states.)

**Still unknown:** why the base station failed. A power cycle is a remedy, not a
diagnosis, and nothing in the mower's telemetry reaches the base's internal
state. Ranked hypotheses and a recurrence-log plan are in §P3 of the plan doc.

## ✅ MEASURED, 2026-08-07 19:30 EDT — the position feed updates ~1x/second during motion

**The number the direction review needed now exists, measured soundly.** Sampled
at 11.2 Hz (89 ms) — ~12x oversampling, so the 2026-08-06 confound cannot recur —
across an authorized 1.5 m drive under RTK Fix.

| position update interval | value |
| --- | --- |
| min | **774 ms** |
| p10 | 820 ms |
| **median** | **1031 ms** |
| p90 / max | 23.5 s / 62.9 s (stationary gaps between pulses) |

**Distance travelled between updates while moving: median 47.1 cm, max 57.4 cm.**
Measured, not inferred.

### What this settles

A stop decision is made against a position up to ~47 cm stale, so **an 0.08 m
waypoint tolerance is not achievable at the fast tier by any control law** — the
information arrives less often than the error accumulates. The direction
review's conclusion is restored, this time on a sound measurement rather than the
withdrawn one.

- 8 cm needs ≈ **0.078 m/s**. The existing slow tier measures ~0.10 m/s → ~10 cm
  per update: still marginally too fast.
- ~0.15 m tolerance is consistent with the slow tier as it stands.

So the tolerance decision is now a real choice between **slowing the final
approach below the current slow tier** and **relaxing the tolerance to ~0.15 m**.
Evidence: `docs/evidence-position-cadence-during-motion-20260807.json`.

### Two bonuses from the same run

**RTK held Fix across all 1627 samples** — the base-station power cycle survives
driving. That was an open risk and is now closed.

**The run was clean:** `target_reached`, 1 turn + 3 linear pulses
(0.475/0.581/0.394 m = 1.451 m for a 1.5 m leg), no reverse-recovery trigger.
First well-tracked pass of the session, under Fix. Do not over-read one run, but
it contrasts sharply with the Float-era overshoot behaviour.

## Superseded — beta24 deployed; the next measurement needs a MOVING mower

Host runs `0.6.4-beta24`, motion-disabled. Two additions, both off-mower:

**1. An RTK freshness guard** — the motion gate now refuses with
`rtk_telemetry_stale` when the RTK payload has not changed for 300 s, and
reports `rtk_report_age_seconds` either way. This exists because on 2026-08-07 a
three-hour-old fix presented as current and `valid_for_motion` stayed `True`.
Detection reuses the project's own principle: a live feed is never perfectly
still. An unmeasurable age is `None` and does **not** block, so diagnostic
callers are unaffected. **It deliberately does NOT require RTK to read Fix** —
that threshold is still the operator's open decision (below).

**2. Per-channel report attribution** — `report_stream_probe` now splits
arrivals into position / RTK / VIO instead of counting all inbound LubaMsgs,
closing the limitation recorded on 2026-08-06.

### ⚠️ What the per-channel probe found, and why it matters

Stationary mower, RTK Fix, 40 s: **75 messages arrived at ~1.9 Hz and not one
carried a changed position, RTK or VIO payload.** The probe is not at fault —
the position path is populated and a unit test proves the split works.

So **the position-report cadence cannot be measured from a stationary mower.**
The device appears not to push these channels when nothing is moving. The
measurement the direction review actually needs — how often position updates
*during motion*, which bounds how precisely a stop can be timed — **requires a
moving mower, daylight, and fresh operator authorization.** That is the next
physical step, and it is a straight line, not a gate attempt.
Evidence: `docs/evidence-per-channel-report-probe-20260807.json`.

Silver lining: this probe would have caught the RTK latch in 40 seconds instead
of three hours, because it reports a silent channel while total traffic looks
healthy — exactly the shape that hid it.

### Still open — the operator's call

Whether the motion gate should require RTK **Fix**: hard blocker, warn-and-record,
or blocker with an explicit `allow_degraded_rtk` override for characterisation
runs. Freshness is now guarded regardless; this is about quality.

## ✅ RESOLVED, 2026-08-07 18:39 EDT — RTK is back to Fix; a base power cycle did it

RTK read `Float` from 15:40. The operator power-cycled the **dock and RTK base
station** at ~18:15, and at **18:39:08** the whole RTK group refreshed together:
`rtk_position` float → **fix**, `position_level` 2 → **1**, `satellites_robot`
26 → 23, `l1_signal_quality` 35 → 29. (So `position_level` **1 = Fix, 2 =
Float**.)

The cause was the **base station's own solution/survey**, not the rover: its
reception was healthy throughout (24 co-viewed satellites) and it had not been
moved. A rover-side `sync_rtk_and_dock` could not clear it; a base power cycle
did. **Precision work is unblocked.** Full record:
`docs/evidence-rtk-float-investigation-20260807.json`.

### ⚠️ Two measurement traps this session walked into — do not repeat them

1. **RTK sensors LATCH.** From 15:40 to 18:39 the entire RTK group was frozen,
   so repeated polling re-read one stale value while looking like a stable
   signal. A forced burst of 50 reports refreshed **no** RTK entity, which is how
   it was caught. `rtk_position` derives from `basestation_info.rtk_status`
   (`sensor.py:570`) and holds its last value instead of going unavailable when
   the device stops sending that structure — same failure class as latched blade
   RPM, cached `vio_heading`, frozen `toward`. Mid-session conclusions about
   Float *persistence*, and both the resync and relocation as RTK tests, were
   invalidated by this and have been corrected in the evidence file.
2. **`rtk_correction_age` and `rtk_signal_quality` are dead fields** —
   unpopulated since the 00:20 EDT restart. Their zeros were briefly read as
   "no corrections arriving"; they mean "no data".

**Method rule:** bit-identical values across repeated polls are evidence of a
dead feed until proven otherwise. Check `last_updated` against an entity known to
be moving before calling a value stable.

**Still worth fixing:** a freshness guard, so a latched RTK value can never be
read as current. That matters more than any threshold choice, because a threshold
on stale data is worthless.

### ⚠️ The gate gap this exposed — decide before the next run

`_is_valid_motion_position` (`services.py`) validates coordinates, zero-pose,
`pos_type` and `zone_hash`. **It does not check RTK quality at all.**
`valid_for_motion` read `True` throughout this Float session, so the backend
would have permitted a real precision run on 13.9 cm positioning. Every prior
gate ran at Fix by luck, not design.

The fix is **not yet chosen** and is the operator's call, because hard-blocking
non-Fix would halt all motion testing while this fault persists:
hard blocker (`rtk_not_fixed`) / warn-and-record / blocker with an explicit
`allow_degraded_rtk` override for characterisation runs.

### The 1.6 m relocation (authorized, clean)

`(4.8768, −1.8125)` → `(5.0026, −3.4058)`, net **1.598 m** on a 1.5 m plan, one
turn command, three linear pulses (0.568/0.466/0.492 m), stopped
`max_linear_commands_reached` **3.6 mm** outside a deliberately loosened 0.15 m
tolerance. Teardown verified: disarmed, no session, blades 0, `MODE_READY`.
Note `max_linear_commands` is capped at **3** by the vector-segment schema, so
reaching further needs longer pulses, not more commands.

`scripts/camera_motion_check.py` got its first true-positive validation on this
run — see the 2026-08-07 tooling note below.

## START HERE (earlier), 2026-08-07 — beta23 deployed; the "just ask for a faster feed" fix is REFUTED

The host runs `0.6.4-beta23`, motion-disabled. It adds one read-only diagnostic,
`report_stream_probe`, and changes no motion behaviour. No motion has run.

**The probe answered its question, and the answer is no.** The wire report
`period` field defaults to 1000 ms and nothing in this integration had ever
lowered it, so the obvious hypothesis was a cheap ~5× win. It is refuted:

| requested period | mean median interval |
| --- | --- |
| 1000 ms | 706.9 ms |
| 200 ms | 590.3 ms |

Interleaved 1000/200 three times each. Observed ratio **1.20** against **5.0**
expected if honoured, and the spread *within* a setting (445–640 ms) is larger
than the difference *between* settings (117 ms) — the signature of no effect.
Across all 203 pooled intervals at every setting, p90 is 1002 ms and the max is
1107 ms: nothing ever waited materially longer than ~1 s and nothing responded to
a shorter request. Evidence:
`docs/evidence-report-rate-probe-20260807.json`.

⚠️ **One limitation bounds that conclusion.** `last_report_at` is stamped on
*every* received LubaMsg, not only on periodic position reports, so the probe
measures total inbound traffic rather than the position-report channel. The
refutation holds (a real 5× change in the report channel would still have moved
the total), but the **exact position-report cadence is still unmeasured**. To
close it, discriminate arrivals per `RptInfoType` or diff the parsed position
payload instead of the message timestamp.

**What this does to the plan.** The "expose `period`, request 200 ms" work is
dead — drop it. The *other* half of the feed fix is untouched and still worth
doing on its own merits: the Gate 4/5 executor never holds a continuous
subscription during motion while four other motion paths do
(`_manual_velocity_pulse_test`, `_manual_velocity_cumulative_pulse_test`,
`_experimental_execute_segment_burst`, `_manual_velocity_segment_test`). That is
a defect regardless of the rate the subscription runs at. The tolerance question
(§4 of the direction review) is now the more important branch.

**Also resolved:** the pre-linear post-turn guard site stays unguarded — replay
of all 10 committed `post_turn_alignment` records found zero aim errors ≥90°,
worst 18.78°. See `docs/evidence-post-turn-alignment-replay-20260807.json`.

**Documentation correction:** the runbook's "128 Mammotion entities" is a
readiness-poll value, not a stable count. `scripts/ha_restart.sh` exits as soon
as the count reaches 100, so the number it prints depends on when the poll lands
(it read 130 this time). The settled count is **131**, with 18
unavailable/unknown for an idle mower. Do not treat a difference here as a
regression.

## Historical start block — 2026-08-06 evening — beta22 is deployed motion-disabled; the U-turn path is closed

The host runs `0.6.4-beta22`. It is the beta21 tree plus the reverse-recovery
containment guard, and it is an **unaccepted, motion-disabled staging
candidate**. No motion ran during the deploy and none is authorized.

**What the guard does.** After a forward pulse, a correction at or beyond 90°
now stops the segment with `target_requires_reverse_recovery` instead of being
dispatched as a "re-alignment" (`services.py` `_requires_reverse_recovery`,
`_MAX_FORWARD_REALIGNMENT_DEGREES = 90.0`). A forward-only segment can no longer
silently become a U-turn controller. Separately, an off-bearing segment with no
correction budget left now stops with `vio_realign_budget_exhausted` rather than
spending its remaining forward budget driving off-bearing.

**⚠️ Expect Gate 4 to fail on this build, and do not treat that as a
regression.** Evidence replay proves *both* recorded Gate 4 passes — day2j
(+103.427°) and the beta21 second geometry (−112.325°) — contain a correction
this guard refuses. Those passes were bought by driving past the waypoint and
turning back: 2.06–2.28 m of travel for a 1.04 m path. Beta22 converts that
boolean pass into an honest stop. The open decision is **not** "retry Gate 4"; it
is whether to fix control quality first or to accept overshoot-and-recovery as
the shipped behaviour. See §8 of `docs/gate4-repass-20260805.md`.

**Deploy facts** (full record in `docs/deploy-runbook-p0.md`): backup
`/config/mammotion-backup-20260806-1951-beta22.tgz`; 46/46 files byte-identical,
aggregate `dbab51a64ff86032fec28b130d2d0605`, zero AppleDouble; both card copies
`49dd1df816162f523285d485e4a8cb6e`; API back in 41 s, 128 entities in 108 s;
resource `?v=0.6.4-beta22&build=49dd1df8`; backend `pymammotion 0.8.12.post1`
verified. Runtime readback: motion disabled, `real_motion_allowed: false`, no
session, no route, `MODE_READY`, blades OFF at 0 rpm, position `(5.6444,
-4.4875)` RTK Fix / `AREA_INSIDE` — unmoved since the 2026-08-06 teardown.
Durable record: `docs/evidence-beta22-containment-deploy-20260806.json`.

**BLE: initially unverifiable, since VERIFIED — deploy checks are complete.** At
the 19:56 readback the transport had not re-registered
(`ble_transport_not_registered`, `active_transport: none`, `online: false`)
across an 8-minute poll, because the battery was at **2%**. A flat mower never
advertises, so the transport cannot register — that was the whole explanation,
not a link fault, and the `ble_rssi: -62` in that readback was the mower's own
cached self-report, **not** a liveness signal.

The mower docked itself at 20:10 EDT (**no motion was commanded by this
session**) and charged to 26%. The 20:29 re-check: `active_transport: ble`,
`online: true`, `ble_rssi -46`, preflight `BLE link live PASS (entity=on
transport=ble rssi=-48)`.

**The beta22 dry-run verification is complete** (20:43 EDT, the item the BLE
outage had blocked). A two-segment dry run with the exact card profile returned
`valid: true`, `stop_reason: dry_run`, `would_send: false`, 0 real segments, and
empty errors/warnings/blockers. It echoed `final_approach_metres_per_pulse: 1.06`
and `turn_degrees_per_second: 37.0` — proving the plumbing defect fixed in this
commit is live — and computed the 90° junction as feasible (3 commands of 4,
`rotation_bound_source: conservative_observed_rate_with_refresh`). No command was
dispatched.

⚠️ **Do not read that dry run as run-readiness.** Its own
`initial_vio_feed` is `{live: false, tracked_features: 0, brightness: Dark}`.
Dry-run VIO gates are **advisory** by design (`passed: dry_run`), so a dry run
reports valid while VIO is unusable. A real `turn_mode: "vio"` run is refused
before dispatch with `blockers: ["vio_active"]`.

**Live state at handoff (2026-08-06 ~20:44 EDT).** The operator undocked the
mower and ended the paused mowing task; **no motion was commanded by this
session**. Mower at `(4.6715, -1.1719)`, `AREA_INSIDE` `Backyard Right`,
`zone_hash` non-zero, RTK Fix, `MODE_READY`, blades OFF at 0 rpm, BLE transport
live at −60, battery **28%**, motion gate off, no session. Ending the mow cleared
the route to `no_route` / `blocks_motion: false`, reproducing the documented
`stale_route_while_ready` behaviour.

**Nothing can run tonight.** VIO is dark at 0 features, so Real Go / Gate 4 /
Gate 5 are all refused by design, and 28% is not a run budget. The next physical
step needs daylight, a charged battery, and — per the containment finding above —
a decision on control quality first, not another Gate 4 attempt.

**Two review findings recorded with the guard.** The previously untested
`vio_realign_budget_exhausted` abort is now pinned by a direct executor test, and
`scripts/diagnose_motion_result.py` names both new stop reasons
(`forward_only_segment_refused_reverse_recovery`,
`vio_realign_budget_exhausted_before_target`) instead of falling through to
`inspect_recorded_stop_reason`.

**RESOLVED 2026-08-07 — the pre-linear post-turn site stays unguarded.** Replay
of all 10 committed `post_turn_alignment` records found **zero** aim errors ≥90°;
worst is 18.78°, and both corrections that ran succeeded. Guarding it would
refuse a condition that has never occurred. Revisit thresholds and full reasoning
in `docs/evidence-post-turn-alignment-replay-20260807.json`.

**Also still open:** the profile-identity question in §5 of
`docs/gate4-repass-20260805.md`. The card now emits the Gate 4 re-pass profile,
so that specific gap is closed, but the profile itself remains accepted on
overshoot-and-recovery evidence only.

## Historical start block — 2026-08-06 morning — Gate 4 reproduced, but only by reversing after overshoot

The host runs beta21 and the card now emits the reviewed Gate 4 profile. A
second daylight geometry on 2026-08-06 returned `target_reached` for both
segments, but repeated the control-quality failure: 2.06 m sampled travel for a
1.04 m path, including a -112.325° recovery after segment 1 passed its target.
The operator GIF visibly confirms pivots, reversals and backtracking. Evidence:
`docs/evidence-gate4-beta21-second-geometry-summary-20260806.json` and section 7
of `docs/gate4-repass-20260805.md`.

Experimental motion was disarmed immediately. Final readback: disabled, no
session, `MODE_READY`, blades zero, RTK Fix, VIO Light/80; 64 seconds of
post-run telemetry were stationary. **No further motion or Gate 5 is ready.**

An off-mower safety patch is in progress on `feat/gate4-day2j-profile`. It
refuses post-linear VIO realignment when `abs(aim_error) >= 90°`, returning
`target_requires_reverse_recovery` before a U-turn can be sent. It also fails
closed when the VIO realignment budget is exhausted. Evidence replay proves
both nominal Gate 4 passes contain a correction this guard would refuse, and a
direct executor test proves no recovery turn is dispatched. All 374 scoped
executor tests and the complete 496-test coverage suite pass, as do frontend,
mypy, Ruff, and all-files pre-commit. The existing cumulative-distance ceiling
would not have prevented these U-turns and was deliberately left as a separate
design item. Final review, beta22 versioning, and a motion-disabled staging
deploy remain. See `docs/CODEX-HANDOFF.md` for the exact continuation checklist.

## Historical start block — 2026-08-05

**Read `docs/gate4-repass-20260805.md` first.** Gate 4 failed on 2026-08-03 and
was re-passed on 2026-08-05: both segments `target_reached`, misses 0.0403 m and
0.0330 m against an 0.08 m tolerance
(`docs/evidence-gate4-beta20-day2j-real-result-20260805.json`).

Three things a next session must not miss:

1. **The re-pass used three parameters the frozen `LUBA_ACCEPTANCE_PROFILE` does
   not carry** — `linear_pulse_duration_ms` 1300 (card: 3500),
   `max_linear_commands` 3 (card: 1), and `max_turn_translation_distance` 0.30,
   which the card never sends at all (so it inherits the backend default 0.25).
   `docs/p0-beta-release.md:98-102` says passing Gates 1-4 while the card emits a
   *different* profile is the exact gap that profile was created to close.
   **Either the card profile moves to match, or this re-pass does not underwrite
   a Gate 5 attempt.** That decision is open and is not mine to make.

2. **It passed by overshooting and recovering, not by tracking.** 2.2773 m of
   actual travel for a 1.0400 m planned path; segment 1 needed a 103.427°
   recovery turn. That recovery is only legal at `max_turn_translation_distance`
   0.30 — at 0.25 it is refused, which is exactly how the day2e and day2h
   attempts died (`vio_realign_incomplete`).

3. **Reproduction on a second daylight geometry is still required and still
   unmet.** One run does not settle a profile, especially given a 2-6 cm pulsed
   measurement noise floor against an 0.08 m tolerance.

Two kinematic claims previously in these notes were **refuted by direct RTK
measurement** on 2026-08-05 and should not be relied on: single-shot linear does
*not* give fine distance control (it is a fixed ~0.11 m step regardless of
commanded duration), and single-shot turning is ~2.4°/command, not ~8-9°.
Refresh 200 is the controllable regime for both phases. See §6 of the re-pass doc.

## Upstream survey, 2026-08-04 — nothing to adopt (read-only; no writes to `mikey0000`)

Checked so a later session need not repeat it. Fetch only; nothing pushed,
commented, or opened on any `mikey0000` repository.

**`mikey0000/Mammotion-HA`** — upstream `main` is at `0.6.4-beta11` and is an
**ancestor of our HEAD**, so we already contain all of it. Branch survey:

| branch | status |
| ------ | ------ |
| `agora-webrtc`, `firmware-updates` | already contained in our HEAD |
| `path-planning` | **wrong problem** — map *drawing* buttons (`start_draw_border`, `start_draw_barrier`, `start_draw_corridor`, `start_erase`), not click-to-path navigation, and built on the 0.5.41 era |
| `claude/open-code-issues-j9tx7r` | cloud-transport self-healing, Agora WebSocket reconnect, Spino pool-cleaner polling. Our motion path is BLE-only and that watchdog "never acts while BLE covers the device"; the Agora part only helps a camera whose stills are a placeholder |

**`mikey0000/PyMammotion`** — latest release `v0.8.12` (2026-07-27); we run
Chorty `0.8.12.post1`, i.e. at or ahead of it. Five post-release commits:

- **#177 `is_send_blocked`** (rate-limit gate silently blocking all sends) —
  **already present** in our build (`transport/base.py:418`, `mqtt.py:342`,
  `device/handle.py:432`). Cloud/MQTT-scoped anyway.
- **BLE reassembly fix** ("thanks @Chorty") — this is *our* finding upstreamed:
  three `clear_notification()` calls on the sequence-gap, checksum-fail and
  exception paths. **Already present** in our build (lines 381 / 418 / 427).
- **2026-07-29 saga / `token_manager` / `client` refactor** — cloud auth and
  token handling. Not in our build and not relevant to BLE-only motion.

**The compass-convention bug is ours, not inherited.** Upstream `services.py` is
748 lines against our 14,189, and contains **zero** occurrences of
`calibrated_forward_heading_offset_degrees` / `heading_offset_degrees`. The
entire click-to-path motion stack is our own code, so no upstream user is
affected and there is no upstream fix to adopt. No upstream open issue mentions
heading/orientation/compass.

## 🚨🚨 ROOT CAUSE FOUND, 2026-08-04 21:07 EDT — `toward` is a COMPASS bearing and the legacy path treats it as a MATH angle

Found by **read-only observation of an operator-initiated night mow**. No
commands were sent; experimental motion stayed off (the gate correctly reported
`blade_reported_on` + `active_mowing_detected`). Evidence:
`docs/evidence-darkmow-observation-20260804.jsonl`.

**The relationship is a mirror, not an offset.** Over 65 moving samples with a
fresh `toward`, spanning travel bearings 85.9–354.8°:

| candidate | circular mean | circular **sd** |
| --------- | ------------- | --------------- |
| `bearing − toward` (**the formula in use**) | 88.14° | **15.45°** |
| `bearing + toward` | 90.13° | **1.93°** |

So `map_bearing = 90.13° − toward`. Split by heading band the invariant holds:
bearings 0–120° give 89.15° (n=24), bearings 240–360° give 90.62° (n=41) — a
1.47° difference, inside noise, across ~270° of heading.

That ≈90° is the signature of the standard conversion between a **math angle**
(`atan2(dy,dx)`, CCW from +x) and a **compass bearing** (CW from north).
`toward` is a compass bearing. It is the same counter-clockwise convention this
project already fixed once in `orientation` (beta18:
`direction = (-orientation) % 360`); it was never applied to `toward`.

**The bug** is at `services.py:2932-2935`:

```python
reported_heading  = float(current_heading)                    # toward (compass)
corrected_heading = (reported_heading + heading_offset_degrees) % 360   # toward + 102.4
target_heading    = _path_heading_degrees(current, target)     # atan2 (math angle)
heading_error     = _heading_error_degrees(corrected_heading, target_heading)
```

An additive constant can only equal a mirror at **one** heading. `toward + 102.4`
equals `90.13 − toward` only at `toward ≈ −6.14°`; elsewhere the aim error is
`2 × (toward + 6.14)` — it **doubles** with heading deviation.

**Independent confirmation by retrodiction.** Using only tonight's mow-derived
constant, the model predicts the three 2026-08-01/02 night runs recorded days
earlier:

| run | recorded bearing | model bearing | error |
| --- | ---------------- | ------------- | ----- |
| pulse 1 | 281.20° | 280.35° | −0.86° |
| pulse 2 | 281.88° | 281.54° | −0.34° |
| Nudge | 282.92° | 282.75° | −0.17° |

It also reproduces their "implied offsets" (110.57 / 112.95 / 115.37 vs recorded
111.43 / 113.29 / 115.54) **including the upward trend**, because
`implied = 90.13 − 2 × toward` is heading-dependent by construction.

**What this overturns.** The earlier entry below closing the ~11° question as
"unsupported / unmeasurable" is itself now wrong on the key point. `toward` is
*not* broken and the night numbers were *not* garbage — they were accurate
measurements of a quantity that is heading-dependent because the formula is
mis-specified. What *is* confirmed is that no single constant can ever be right.

**Scope.** This affects the **legacy / course-over-ground path** (Nudge,
`toward`-based aiming) and explains the night Nudge's 0.312 m mostly-cross-track
miss. The **VIO path is unaffected** — it derives its own per-session offset from
a calibration drive, which is consistent with the separate finding that 102.4 is
inert under `turn_mode: "vio"`. Tonight's Gate 4 cross-track error is therefore a
**different** problem and is not explained by this.

**Also fixed by implication:** `scripts/motion_capture.py --summarise` computes
`implied = (bearing − toward) % 360` against `CONFIGURED_OFFSET = 102.4`. That is
the wrong formula, so every "implied offset" it has ever printed is unreliable —
including tonight's 205.38°. It should report `bearing + toward` (expected ≈90°).

**Nothing was changed.** No constant, no code, no deploy. Fixing the legacy
heading conversion is a motion-path change that needs its own review and
daylight validation.

**Bonus confirmation, using the capture fields added an hour earlier:** VIO in
full darkness reports honestly — `visual_positioning_status: signal_none`,
tracked 0, detected 0, `brightness: dark`. **No dusk latch** in this sample, so
the latch is not simply "what VIO does at night".

## 🚨 Gate 4 attempt on beta20, 2026-08-04 20:40 EDT — FAILED, but the failure moved to cross-track aim

**Gate 4 did not pass.** It also did not fail the way it failed on 2026-08-03.
Run in fading twilight under explicit operator authorization after a VIO
liveness check passed; experimental motion was disarmed immediately after and
the mower is stationary at `(5.1011, −2.1034)`.

**The turn-budget failure mode is gone.** Both turn phases passed, in 2 commands
each, with every pulse `heading_went_fresh: true`:

| segment | turn cmds | turn final error | linear moved | outcome |
| ------- | --------- | ---------------- | ------------ | ------- |
| 1 | 2 | **3.808°** | 0.4483 m | `target_reached` |
| 2 (junction) | 2 | **3.000°** | 0.4463 m | `max_linear_commands_reached` |

Segment 1 fully reached its waypoint. The −90° junction turn — the geometry the
deployed guard preflighted as feasible (3 commands vs 4, 0.234 m vs the 0.25 m
cap) — executed in 2 commands. Compare 2026-08-03, where four turn commands
left 34.795° of error and no linear command ever ran.

**The new blocker is cross-track aim, and along-track is essentially perfect.**
Segment 2 landed 0.11660 m from waypoint 2 against a 0.08 m tolerance. The miss
is almost entirely perpendicular: final `x = 5.1011` against a target of
`5.1006` — **0.5 mm along-track** — with the whole error in `y`.

| segment | travel bearing | expected | aim error | cross-track |
| ------- | -------------- | -------- | --------- | ----------- |
| 1 | 74.594° | 69.501° | **+5.09°** | 0.0398 m |
| 2 | 354.871° | 340.584° | **+14.29°** | 0.1101 m |

Note the turn ended 3.0° from its **vision** target while travel was 14.29° off
in the **map** frame — an ~11° vision→map discrepancy at execution time, even
though the calibration drives measured that offset at ≈0° earlier the same
evening.

**Sharper hypothesis, from the VIO commentary (added same night).** The two
segments' aim errors have *different* explanations, and only the second is a
candidate defect:

- **Segment 1 (+5.09°) is inside the expected noise.** Its offset came from a
  `calibration_drive` with a **0.0892 m** baseline; 1 cm of position noise on
  that baseline is ~6.4° of offset error. Not a defect.
- **Segment 2 (+14.29°) is not explained that way.** Its offset came from
  `linear_refresh` off segment 1's **0.4483 m** leg — a 5× longer, *better*
  baseline — yet the aim error more than doubled. What sits between that
  refresh and segment 2's drive is **the junction turn**: the offset is
  refreshed only from *linear* travel and is **never re-derived across a turn**,
  so a VIO frame that drifts or re-anchors during rotation leaves segment 2
  aiming with a stale anchor.

Note also that the **mid-drive re-aim** (added after a run that "drifted ~25 deg
and sailed past the waypoint") cannot help here: with `max_linear_commands: 1`
there is one forward pulse, and beta17 deliberately suppresses realignment once
no forward budget remains. So on this profile there is no cross-track
correction mechanism available at all.

**⚠️ This run is confounded by falling light and must not be over-read.**
`tracked_features` decayed 71 → 58 across the run with a minimum of 30, and read
43 stationary afterwards. The aim error **grew as features fell** (+5.09° in
segment 1, +14.29° in segment 2), so degrading VIO heading quality is at least
as plausible an explanation as a systematic calibration error. **Do not change
any constant on this evidence.** The aim-error finding needs a repeat in real
daylight with features holding ~80 before it can be called a defect rather than
a twilight artifact. This is precisely the risk the daylight rule exists to
avoid, and it was accepted knowingly.

Gate 4 must not reuse a prior confirmation, so the retry needs fresh daylight
operator authorization and fresh geometry. Gate 5 remains blocked.

Evidence: `docs/evidence-gate4-beta20-{dry,real}-{request,result}-20260804.json`,
`docs/evidence-gate4-beta20-real-capture-20260804.jsonl`,
`docs/evidence-vio-liveness-beta20-{result,capture}-20260804.*`.

## VIO liveness is now a cheap pre-session test (2026-08-04)

A dusk-latch was suspected at 20:33 EDT — `brightness: light`,
`tracked_features: 80` while stationary, 20 minutes past sunset. **That
suspicion was wrong**, and one bounded 25° turn settled it in seconds:
`heading_went_fresh: true`, 19.627° measured over 675.9 ms = **29.04 °/s**,
inside the 21.2–49.6 °/s daylight band, with features holding 78 before and
after. Use this instead of trusting the stationary brightness field: a single
`vio_turn_to_heading` with `max_commands: 2` and `max_displacement_m: 0.3`.

Also observed: `tracked_features` dips transiently during rotation (80 → 44 →
recovered within 2 s) from motion blur. That is normal, not a fault.

## Offset re-derivation, 2026-08-04 evening — the ~11° hypothesis rests on a broken measurement

Three supervised ~0.45 m drives at three headings on the deployed beta19 build,
in the last of the daylight. No deploy, no constant change, no gate claimed.
Fresh operator `go` per drive, armed immediately before and disarmed
immediately after each, ending verified stationary at `(4.7028, −2.6513)` with
blades off and no session.

**`vio_motion_probe` is the wrong vehicle and was rejected before use.** It
sends `send_movement` once and then loops on `asyncio.sleep()` with no
`motion_refresh_interval_ms` parameter at all
(`services.py` `_vio_motion_probe`), so the h-watchdog caps it near 0.10 m —
under `motion_capture.py`'s `MIN_TRAVEL_FOR_BEARING = 0.20`. It cannot produce
a valid travel bearing. Only five services expose refresh; for a straight drive
the usable one is `raw_pymammotion_execute_vector_segment`, whose turn phase
also solves the circularity of needing the offset in order to aim.

**The vision↔map offset is measured and heading-invariant:**

| drive | travel bearing | vision heading | offset | run stop |
| ----- | -------------- | -------------- | ------ | -------- |
| 1 | 208.186° | −153.072° | **+1.258°** | `target_reached` |
| 2 | 289.306° | −72.736° | **+2.042°** | `max_linear_commands_reached` |
| 3 | 175.320° | +176.289° | **−0.969°** | `turn_phase_incomplete` |

Mean **+0.777°**, spread **3.012°** across travel bearings spanning 175–289°.

⚠️ **CORRECTED 2026-08-04 (same night), after reading the VIO commentary in
`_vio_segment_calibration_drive`.** The original wording here — "essentially
zero and does not vary with heading" — is **not supported**, for two reasons:

1. `vision_info.heading` is a body heading in the **VIO's own frame, which is
   re-anchored whenever VIO (re)initialises**, and has *no fixed relationship
   to map-local coordinates*. The offset is a **per-session anchor**, not a
   physical constant. `vio_state` read 2 at all three drives, so these are three
   estimates of **one** anchor, not three independent confirmations of a
   property.
2. Offset accuracy is dominated by calibration baseline: a 2 cm baseline gives
   ~25° of error from cm-level position noise (live 2026-07-11), which is why
   the minimum is 6 cm. These drives used 0.083–0.119 m baselines, so each
   estimate carries roughly **±5–7°** of noise — *larger than the 3.012° spread*
   that was read as invariance.

So the correct statement is: the offset for that VIO session sat near zero
within a noise floor that exceeds the observed spread. Nothing here establishes
heading-invariance or a stable constant. Consequently `vision_info.x` / `.y`
are in that same drifting frame and are **not** map coordinates — do not diff
them against RTK position.

**The 102.4° question is UNMEASURABLE on this hardware — and this is not a
lighting problem.** `toward` was **frozen across every forward leg**: drive 1
held `162.7649` across a 0.5351 m leg, drive 2 held `−85.9472` across a
0.6558 m leg. It updates during *turns*, then freezes during forward travel.
Since `motion_capture.py --summarise` computes
`implied = (bearing − toward_first) % 360`, a stale `toward` makes that
"bearing minus an arbitrary constant", not a measurement. **The
111.43 / 113.29 / 115.54 night values were computed exactly that way**, which
also explains why they "trended upward run to run". The ~11°-low claim has no
measurement behind it and should not be acted on.

**Structural finding: `calibrated_forward_heading_offset_degrees` is not used
for turn targeting in `turn_mode: "vio"`.** `provided_offset_degrees` reads the
separate `vio_heading_offset_degrees` parameter (`services.py:9533`, emitted at
`:9803`); passing 102.4 left it `null` with `offset_source: calibration_drive`.
Gate 4 ran `turn_mode: "vio"`, so **102.4 was inert there and cannot explain its
4.70 cm cross-track miss.**

Measurement chain validated: capture-derived travel bearing agreed with the
executor's computed bearing to **0.14°** (drive 1) and **0.01°** (drive 2).

**Anomaly — drive 3 ended `stop_failed_aborting`.** Both the refresh and the
stop failed with `RuntimeError: BLE link is not ready for motion:
command_queue_backlogged`. The executor refused to continue rather than run
without a guaranteed stop — working as designed. `final_displacement_m` is
`0.0` and no linear phase ran. Note the durable record **under-reports** the
motion: `after_vision_heading` is `null` because the poll failed, but the
capture shows VIO `176.289° → 149.571°`, ~27° of real rotation before the
h-watchdog cut the motors. The BLE link report over the same 45-minute window
is completely clean (zero connects, disconnects, gaps, malformed or dropped
frames), so `command_queue_backlogged` is a queue-state condition, not a link
fault.

Evidence: `docs/evidence-offset-beta19-analysis-20260804.json`,
`docs/evidence-offset-beta19-drive{1,2,3}-{result,capture}-20260804.*`,
`docs/evidence-offset-beta19-capture-20260804.jsonl`,
`docs/evidence-offset-beta19-ble-report-20260804.txt`.

## Daylight turn characterization, 2026-08-04 — rotation floor holds, translation bound does NOT

Four supervised in-place VIO turns ran on the deployed beta19 build `617337d3`
(the turn-feasibility guard is **not** on the host). This was a measurement of
the mower, not a test of the guard. No constant was changed, no deploy
happened, no gate was attempted, and no version was bumped. Every run had a
fresh operator `go`, was armed immediately before and disarmed immediately
after, and ended verified stationary with blades off and no session.

Conditions were the best recorded to date: VIO state 2, brightness `Light`,
80 tracked features for the whole session, RTK Fix, `AREA_INSIDE`
`Backyard Right`, `ble_rssi` −46 to −60. Fresh geometry — the mower started at
`(4.9257, −2.2141)`, distinct from the Gate 4 failure spot. Cadence was the
accepted profile exactly: angular 500, 1500 ms pulses,
`motion_refresh_interval_ms: 200`, tolerance 18°, budget 8, cap 0.5 m.

**All four runs returned `target_heading_reached`**, which is itself new
information — the near-180° case that killed Gate 4 completed here:

| run | delta | commands | final error | displacement | rotation °/s |
| --- | ----- | -------- | ----------- | ------------ | ------------ |
| 1 | +45° | 1 | −6.366° | 0.0129 m | 37.81 |
| 2 | −90° | 2 | −5.392° | 0.1303 m | 21.20 – 29.93 |
| 3 | +135° | 2 | −0.050° | 0.0288 m | 40.44 – 49.57 |
| 4 | −170° | 4 | −0.154° | 0.2955 m | 22.95 – 34.71 |

Nine pulses, all `heading_went_fresh: true`, none excluded, no negative
`progress_degrees`, no direction faults.

**Rotation floor 16.5 °/s: HELD.** min 21.203, mean 32.454, max 49.565 °/s;
the tightest margin is +4.703 °/s. Every pulse beat the floor, and most beat
the top of Gate 4's observed 16.5–21.3 °/s band. The floor is conservative —
substantially so above ~30 °/s — but it was not violated on fresh geometry.

**Translation bound 0.0403 m/s: VIOLATED.** 4 of 9 pulses exceeded it, peaking
at **0.071960 m/s (+78.6%)** on run 4 command 2. Run 4 commands 1–3 all
breached it (0.05014 / 0.07196 / 0.04619); run 2 command 2 breached it
marginally at 0.040429 (+0.32%). This is the finding that matters: the guard
uses 0.0403 m/s to predict whether a turn's translation will breach the
displacement cap, and the real worst case on this geometry is ~1.8× that. That
error is in the **fail-open** direction for that specific check — the guard
would under-predict translation and could admit a turn that then breaches the
cap. It does not affect the rotation-feasibility half of the guard.

Run 4 still finished inside the 0.5 m cap (0.2955 m of 0.5), so no run was
stopped by displacement. Larger turns translate disproportionately: run 4 alone
produced 0.2955 m against 0.0129–0.1303 m for runs 1–3.

**Forward-heading offset: UNTESTED.** No forward drive ran — VIO was already
warm, so the preflight warm-up probe was correctly skipped. The
`motion_capture.py --summarise` figure of 205.38° implied offset over 0.3964 m
is incidental in-place-turn drift spanning four direction-reversing phases,
**not** a forward drive; reading it as an offset measurement would repeat the
exact "net displacement spanning phases" error the handoff warns about. The
~11°-low question is still open and still needs an isolated straight-line
daylight drive. Incidentally, `toward` did update this session (174.05° →
−117.62°), unlike the 2026-08-01 run where it never moved.

**BLE was clean:** a 55-minute window with zero connect events, zero
disconnects, zero sequence gaps, zero unparseable frames, zero dropped frames.

Two smaller observations. Run 2 command 1 recorded `elapsed_ms` 2475.1 against
a nominal 1500 ms pulse — ~975 ms of stop/refresh-confirm overhead — so
elapsed-based rates are the conservative reading (nominal-duration would put
that pulse at 35.0 °/s instead of 21.2). And
`scripts/diagnose_motion_result.py` returns null turn/linear phase fields for
standalone `vio_turn_to_heading` results because it is shaped for
multi-segment paths; `outer_stop_reason` is still correct.

No constant was changed during the measurement session itself. The revision
review followed immediately afterwards and is recorded below.

## Turn-translation constant revision, 2026-08-04 (off-mower, no deploy)

The review concluded that **raising the per-second bound was the wrong fix**,
and the code now bounds translation per degree instead.

Pooling both geometries (13 refresh-200 pulses) settled the rotation half
first: the true minimum is **16.5251 °/s** against the 16.5 floor, set by Gate
4's pulse 1, not by the 08-04 runs. `16.5` is therefore correct and stays
**unchanged** — raising it toward the 08-04 minimum of 21.2 would move the
guard fail-open.

The translation half was both wrong and wrongly shaped. Raising 0.0403 → 0.0720
(the observed max) and changing nothing else would have **refused two turns
that demonstrably succeeded**: the +135° run estimated 0.540 m against an
actual 0.029 m, and the −170° run 0.756 m against 0.296 m. The cause is
structural — `estimated_translation = needed × per_command_translation`, where
`needed` already comes from the *pessimistic* rotation floor, so two
anti-correlated worst cases were multiplied (a slow pulse sweeps fewer degrees
and therefore drags less). The too-low constant had been accidentally
cancelling that compounding.

Translation during an in-place turn is the arc a tracked point sweeps about the
true rotation centre — `translation = r × θ`. It scales with **angle**, not
elapsed time; per-second is only equivalent at a constant rotation rate, and
the measured rate varied 16.5–49.6 °/s. The pooled maximum is **0.002410
m/deg**, implying a physically plausible 13.8 cm offset between the drive
centre and the tracked point.

`_VIO_TURN_CONSERVATIVE_TRANSLATION_M_PER_SECOND = 0.0403` is replaced by
`_VIO_TURN_CONSERVATIVE_TRANSLATION_M_PER_DEGREE = 0.0026`, and the estimate is
now `|initial_error| × 0.0026`, independent of the command count. The constant
is boxed in from both sides, and the upper wall is the tighter one:

- ≥ 0.002410 — the pooled observed maximum (fail-closed floor);
- ≤ 0.25/90 = 0.002778 — a 90° L-path junction must stay feasible at a 0.25 m
  cap. **This is binding**, and an initial choice of 0.0028 was rejected
  because it violated it and would have refused Gate 4's own junction geometry;
- ≤ 0.5/170 = 0.00294 — the proven −170° turn at the schema's 0.5 m default.

Result: the guard refuses the failed Gate 4 segment and admits all four
successful characterization turns, staying conservative on every one (est
0.442 m vs actual 0.296 m on the −170° run; 0.435 m vs 0.185 m on Gate 4). The
refresh-0 single-shot branch is deliberately **unchanged** — no single-shot
per-degree evidence exists, so its translation criterion stays delegated to the
runtime displacement cap.

Result fields changed: `per_command_translation_bound_m` →
`translation_bound_m_per_degree`, plus a new `translation_bound_source`. These
are diagnostic only; no service schema, `LUBA_ACCEPTANCE_PROFILE` value, or
version location changed, and nothing was deployed. Tests grew 14 → 20 in
`tests/components/mammotion/test_vio_turn_feasibility.py`, including all four
characterization runs as ground truth, a test that the estimate is invariant to
command budget and pulse length, and one pinning both walls of the constant.

Claim no gates: Gates 4 and 5 remain failed/blocked. The guard is still
**not deployed** — the host runs beta19 `617337d3` without it.

Evidence:

- `docs/evidence-turnchar-beta19-analysis-20260804.json` (pooled analysis)
- `docs/evidence-turnchar-beta19-preflight-dryrun-20260804.json`
- `docs/evidence-turnchar-beta19-run{1,2,3,4}-result-20260804.json`
- `docs/evidence-turnchar-beta19-run{1,2,3,4}-capture-20260804.jsonl`
- `docs/evidence-turnchar-beta19-capture-20260804.jsonl` (session-long, 1762 samples)
- `docs/evidence-turnchar-beta19-ble-report-20260804.txt`

## 🚨 READ FIRST — beta19 is deployed motion-disabled; release is halted

The operator identified a live heading-display mismatch after the beta17 smoke:
the custom-path card's green arrow and 72.8-degree label pointed upper-right,
but the adjacent Home Assistant map card's black mower marker pointed
upper-left. Runtime inspection showed why: the integration published Mammotion
orientation `-29` directly as the device tracker's `direction`. HA treats that
attribute as a clockwise compass bearing, while Mammotion's sign is
counter-clockwise. Beta18 deployed the presentation-only conversion:
`direction = (-orientation) % 360`, so the same sample becomes 29 degrees and
points upper-right. Eight conversion tests cover sign, wrapping and unavailable
values, but the operator then proved the underlying field was not body
orientation: while the mower physically faced upper-left, a zero-command live
snapshot still reported frozen course-over-ground `-29.589`, VIO inactive/0 and
RTK yaw 0. Beta19 is the deployed correction: no idle direction arrow is
drawn from last travel, and Nudge fails closed without trustworthy current
orientation. Motion code and `LUBA_ACCEPTANCE_PROFILE` are unchanged.

Two independent beta16 daylight runs rejected duration-only final-approach
scaling. A 0.450 m single-leg characterization started at `(4.858, -2.132)`
and targeted `(4.858, -2.582)`. Its VIO calibration moved 0.08925 m, leaving
0.36094 m. The nominal 1191.8 ms approach delivered three refreshes and moved
0.43414 m; its normal-priority zero write then took 1392.666 ms to confirm.
The executor evaluated a pre-turn position as 0.08456 m from target, just
outside tolerance, and spent three VIO re-alignment turns even though
`max_linear_commands: 1` left no forward command to benefit. Those turns added
0.0670, 0.0885 and 0.0785 m of displacement. It stopped safely at
`max_linear_commands_reached`; the measured resting position `(4.9538,
-2.7131)` was 0.16237 m from target.

This reproduces the earlier stepwise behavior decisively: the first beta16
short approach requested 1012.5 ms, delivered two refreshes and moved 0.17861
m; the second requested 1191.8 ms, delivered three and moved 0.43414 m. The
full 3500 ms calibration pulses delivered 10 and 11 refreshes and moved 1.07737
and 1.04573 m. Confirmed refresh count and unpredictable stop latency dominate
nominal duration.

The host and branch run beta19. Its beta17 motion
correction budgets final approaches by discrete confirmed refresh count, sends
the zero-speed teardown at emergency queue priority, and suppresses
re-alignment when no forward-command budget remains. The public service schema
and frozen LUBA acceptance profile are unchanged. The candidate must re-pass
affected backend Gates 2 and 4 before a new card Gate 5 run. **Gate 2 passed
on 2026-08-03:** a daylight 0.100 m backend segment returned `target_reached`
with 0.0105 m final error; the operator saw the 9 cm calibration movement and
confirmed its stop. The gate was disarmed, no session remained, and the mower
was stationary for more than a minute. The
subsequent Gate 4 retry failed in segment 1 before its linear phase:
calibration passed, but four VIO turns stopped `max_commands_reached` 34.795°
short of the target. It translated 0.185 m while turning; no linear command or
segment 2 ran. Keep experimental motion off. The durable evidence and analysis
are `docs/evidence-gate4-beta19-retry-*20260803*`; implementation instructions
are `docs/CLAUDE-FINAL-IMPLEMENTATION-PROMPT.md`. The
motion-disabled deploy smoke
passed with 128 entities, verified backend capabilities, both card paths
checksum-identical, the exact accepted-profile label, valid Preview, and a
card Dry-run reporting `would_send: false`. The browser console and card footer
both show beta19 at the collision-proof Lovelace URL
`?v=0.6.4-beta19&build=617337d3`. The third-party map card ignores
`direction` by default. The temporary Jinja-backed `card-mod` rotation was
removed after proving that `direction` was stale travel rather than current
orientation. Dashboard backup:
`/config/.storage/lovelace.dashboard_yard.bak.codex-20260802-213848`. No
physical motion is currently authorized.

The characterization teardown is complete: experimental motion is verified
off, no session remains, blades are off, and a 20-second post-stop capture is
stationary. BLE showed no connects, disconnects, gaps, malformed frames or
drops. The 0.5889 m whole-run displacement implied a 103.89-degree offset, only
+1.49 degrees from the retained 102.4-degree profile; do not change the heading
profile.

It was dark after deployment: VIO reported 0 tracked features. Do not attempt
affected Gates 2/4 or Gate 5 until a fresh daylight preflight passes. The mower
remained `MODE_READY`, inside `Backyard Right`, RTK Fix, blades off, no session,
and experimental motion off after restart and UI smoke. Backup for rollback:
`/config/mammotion-backup-20260802-2207.tgz`.

Evidence:

- `docs/evidence-gate5-characterization2-dry-run-20260802.json`
- `docs/evidence-gate5-characterization2-result-20260802.json`
- `docs/evidence-gate5-characterization2-run-20260802.jsonl`
- `docs/evidence-gate5-characterization2-post-stop-20260802.jsonl`
- `docs/evidence-gate5-characterization2-ble-report-20260802.txt`

## Where we stand after the Gate 4 turn-feasibility correction (2026-08-03)

The correction for the failed Gate 4 retry is implemented, test-covered, and
committed on `feat/vio-turn-to-heading`. It is **not deployed** to the host,
which still runs the beta19 build `617337d3` without it; nothing physical has
changed and experimental motion remains off.

What the code now does (see `docs/CODEX-HANDOFF.md` for the full record):

- `_vio_turn_budget_feasibility()` refuses a real VIO turn before its first
  command when the evidence floor (16.5°/s with refresh; the 8°/command
  single-shot quantum without) cannot reach tolerance within the budget, or
  when the refresh-regime translation estimate would breach the displacement
  cap. Stop reason `turn_budget_infeasible`, `commands_sent: 0`. The
  translation criterion was revised on 2026-08-04 from 0.0403 m/s of pulse to
  0.0026 m per degree swept — see the 2026-08-04 sections above.
- The vector segment surfaces that stop reason directly; the multi-segment
  executor also geometrically preflights junctions 2..N and refuses a real
  path with `path_turn_infeasible` before any motion. Dry runs report the same
  `turn_feasibility` / `junction_turn_feasibility` math without refusing.
- `scripts/diagnose_motion_result.py` distinguishes
  `vio_turn_refused_infeasible_preflight` from
  `vio_turn_budget_exhausted_before_linear_phase` and
  `linear_budget_exhausted`.
- Tests: `tests/components/mammotion/test_vio_turn_feasibility.py` (14 cases;
  the recorded 167.413° case is refused with 7 commands estimated against the
  budget of 4, and the retained evidence JSON keeps its classification). Full
  suite 483 passing; profile, schemas, and all four version locations
  unchanged.

What this does NOT do: it cannot make a near-180° turn succeed — it prevents
the known-unfinishable dispatch. The next physical step is unchanged and still
requires fresh daylight operator authorization: a turn characterization on new
geometry to validate the conservative rate constants and revisit the ~11°-low
offset question, then a Gate 4 retry (two-leg L path), then Gate 5 from the
card. Deploying this correction to the host is a separate, explicit step and
must follow the runbook's two-path card/backend deploy rules.

**Sequencing — the characterization does NOT wait for a deploy.** It is a
measurement of the mower, not a test of the guard, and it runs against the
`vio_turn_to_heading` service the host already has in beta19 (refresh support
included), driven by `scripts/run_motion_with_evidence.py` +
`scripts/motion_capture.py`. Measure per-command rotation and translation at
the accepted cadence (refresh 200, angular 500, 1500 ms pulses) across several
target angles on new geometry. The guard itself is already validated offline
by the test suite; what only hardware can confirm is whether the 16.5°/s and
0.0403 m/s floors generalize beyond the single Gate 4 run. Order of
operations: (1) characterize on the deployed beta19 backend under fresh
operator authorization; (2) confirm or locally revise the constants from that
evidence; (3) deploy the guard build as a version-bumped candidate per the
runbook; (4) Gate 4 retry on the deployed guard build; (5) Gate 5 from the
card. Note a large-angle characterization stays runnable either way: the
current host build has no guard (bounded by the existing budget/displacement
caps), and the standalone service's default 8-command budget judges up to
180° feasible even with the guard — only the 4-command segment profile
refuses near-180°.

## Historical beta16 two-leg failure

The operator used one unchanged beta16 card instance for Preview, Dry-run and
Real Go with exact points `(4.835, -1.861)`, `(4.835, -2.261)`,
`(5.235, -2.261)`: two 0.400 m legs. All runtime gates passed with daylight VIO
Light/80, RTK Fix, blades off, no route/session and the exact accepted-profile
label. The card result is nevertheless a **Gate 5 failure**:

- segment 1's VIO calibration moved 0.09372 m, leaving 0.30663 m;
- proportional final-approach scaling chose a 1012.5 ms pulse, which moved only
  0.17861 m;
- final distance to waypoint 1 was 0.13109 m, outside the 0.08 m tolerance;
- the miss was mostly along-track (cross-track only 0.0233 m);
- it stopped at `max_linear_commands_reached`; segment 2 never started;
- both command-level stop writes succeeded, the session cleared, blades were
  off, the gate was disarmed, and the post-stop capture stayed stationary.

The isolated 3500 ms pulse constant of 1.06 m remains valid, but scaling it
linearly through zero is not valid at 1012.5 ms because motor onset/dead time is
material. The prior claim that 0.3-0.5 m legs form a usable band is refuted.
Do not change the accepted profile from this one short-pulse sample and do not
retry without diagnosis, a fresh daylight geometry and fresh operator
confirmation. PR #10 remains draft; do not merge or publish a beta.

Evidence:

- `docs/evidence-gate5-final-dry-run-20260802.json`
- `docs/evidence-gate5-final-result-20260802.json`
- `docs/evidence-gate5-final-run-20260802.jsonl`
- `docs/evidence-gate5-final-post-stop-20260802.jsonl`
- `docs/evidence-gate5-final-ble-report-20260802.txt`

The motion capture measured 0.2722 m net at bearing 274.99 degrees. Initial VIO
heading was -85.881 degrees, implying a 99.55-degree forward offset (-2.85
degrees from 102.4), while `toward` stayed stale at 175.4473. The executor's
calibration offset normalized to about -1.689 degrees and its later linear
refresh reported 1.794 degrees. This is not a reproducible aim failure and does
not justify changing `calibrated_forward_heading_offset_degrees: 102.4`.

## Historical open finding before the beta16 run

**`calibrated_forward_heading_offset_degrees: 102.4` looks about 11 degrees
low.** Three straight-line runs in darkness on 2026-08-01/02 all travelled on a
bearing well off what the configured offset predicts — implied offsets of
111.43, 113.29 and 115.54 degrees, mean **113.42** against a configured
**102.40**. The Nudge missed by 0.312 m and the miss was almost entirely
**cross-track**, which is an aim error, not a distance error.

It also explains Gate 4 better than anything previously proposed: an 11 degree
aim error predicts ~5.7 cm on a 30 cm leg, and Gate 4 landed 4.70 cm out.

**Do not change the profile on this alone.** Two things block a derivation:
`toward` is course-over-ground and did not update at all across a 1.36 m drive
(so the baseline may be stale), and the implied offset trends upward run to run
(consistent with the mower rotating while `toward` fails to track it). Daylight
resolves both, because VIO gives a real heading rather than one inferred from
displacement.

👉 **Treat the next Gate 5 run as an offset re-derivation as well as an
acceptance run.** Also expect the card's heading arrow to point ~11 degrees off,
since it is drawn with 102.4.

Full data, the Nudge hardware result, the docking-readback lesson and the stale
`last_error` trap are in the "Night session 2026-08-01/02" section of
`docs/p0-beta-release.md`.

## Third pass — what changed on 2026-07-31

**The card drove the mower for the first time.** Gate 5 is now PARTIAL, not
untried. `0.6.4-beta12` is deployed to the host, and the run, its two measured
constants, and the darkness abort are recorded in the Gate 5 entries of
`docs/p0-beta-release.md`. Raw telemetry:
`docs/evidence-gate5-run1-20260731.jsonl`.

Read those before planning the next run. The short version:

- Gate 5 still needs a **clean pass**: both segments `target_reached` from the
  card. The first attempt used 2.1 m legs, which the accepted profile cannot
  complete (one linear command per segment, no loop-to-tolerance). Use ~35 cm
  legs. A pre-validated path from the mower's current resting position is
  `(5.048, -3.111)` then `(5.398, -3.111)`.
- Two constants were measured and are **hypotheses, not decisions**:
  ~~`final_approach_metres_per_pulse` looks ~25% low~~ — **REFUTED 2026-08-01**
  by two isolated night pulses (1.0785 m and 1.0449 m, mean 1.0617, within
  0.16% of the configured 1.06). The constant is correct; the original figure
  came from measuring across a phase boundary in a two-segment run. And
  `heading_tolerance_degrees: 18` is
  far too loose (a turn landed 2.11 deg from target). `LUBA_ACCEPTANCE_PROFILE`
  is deliberately unchanged — editing it un-accepts it.
- ⚠️ **The VIO dusk cliff is steep and the HA sensor entities lag it.** 80/80
  features and `light` at 20:40; 0/0 and `dark` at 20:47. The sensor entities
  are coordinator-tick cached and are **not** a readiness signal — the live
  `initial_vio_feed` returned by a dry run is. Dry-run immediately before any
  Real Go near dusk.
- `scripts/ha_set_experimental_motion.py on|off|status` now toggles the motion
  gate without hand-driving the options flow. It verifies the result through
  runtime state rather than the flow's reply, because the flow returns
  `create_entry` with empty `data` even on success. It preserves other options
  by reading the flow's own schema defaults — **not** from
  `/api/config/config_entries/entry`, which never exposes `options` and would
  silently reset them.
- `scripts/ha_set_card_resource.py <version> --apply` sets the Lovelace resource
  cache key. Deploying the card file alone is not enough: browsers key on the
  query string, so an unchanged `?v=` leaves every browser on the previous card
  while every server-side check reports the new one.

## Fourth pass — night of 2026-08-01/02

Host is on **`0.6.4-beta15`**. Everything below is deployed and pushed.

- **Nudge shipped and is hardware-proven.** A straight line along the current
  facing, for moving the mower when VIO is unavailable. First hardware run:
  1.3575 m, **2 linear commands, 0 turn commands**, clean stop. It works at
  night because RTK holds in darkness while VIO does not, and because the target
  sits on the heading ray so the turn phase has nothing to do.
  ⚠️ It sends `turn_mode: "legacy"` **only** to clear the `vio_active` gate,
  which blocks up-front on `turn_mode == "vio"` regardless of whether a turn is
  needed. Legacy steers by course-over-ground and is safe there **only because
  no turn occurs** — it is *not* a night-capable turn mode. A non-zero turn count
  on a Nudge means it tried to steer blind; investigate.
- **Real Go cannot run at night, by design.** It uses `turn_mode: "vio"` and is
  refused with `blockers: ["vio_active"]` before dispatch. Confirmed on the real
  armed path: `would_send: false`, 0 commands, `phases: []`, mower unmoved. Both
  VIO gates are deliberately *advisory* in a dry run (`passed: dry_run`), so a
  dry run reports `passed: true` while its own detail says VIO is unusable.
- **The card shows heading.** A green arrow on the mower marker plus a
  `facing (map bearing)` preflight row, computed the way the backend aims. It is
  drawn in **map space** (a point one metre ahead is transformed) because `toSY`
  flips the Y axis and a screen-space rotation would mirror the bearing.
- **`final_approach_metres_per_pulse: 1.06` is correct** — mean 1.0617 over two
  isolated pulses, within 0.16%. The earlier "25% low" claim was refuted; it came
  from measuring across a phase boundary in a two-segment run.
- Mower ended the session **docked and charging**, gate off, blades off.

## Gate 5 attempt 2026-08-02 morning — SET UP, NOT YET RUN

> Historical setup record. Its usable-band and coordinate-entry conclusions
> were superseded by the beta16 run documented at the top of this file.

Session ended on token budget mid-setup. **No motion occurred.** Capture proved
it: 374 samples over 10 minutes, net travel **0.0000 m**, position and both
headings bit-identical.

⚠️ **The experimental-motion gate was left ARMED** (`real_motion_allowed: true`).
Disarm with `scripts/ha_set_experimental_motion.py off` unless resuming
immediately.

Live state at handoff:

- mower `MODE_READY` at **(4.795, −1.9502)**, `toward` 173.1006, Backyard Right,
  `AREA_INSIDE`, RTK Fix, `valid_for_motion: true`, blades off, BLE ~−60
- **VIO ALIVE** — `vio_state: 2`, 80/80 features, `"Light"`,
  `initial_vision_heading: −83.673`. This is the condition the offset
  re-derivation needs and could not get at night.
- route clear (`reason: "no_route"`), no session, host on `0.6.4-beta15`

### What blocked the run: leg length, not gates

Two dry runs passed **every** safety gate. The operator's clicked path was simply
too long for the accepted profile:

| leg | clicked | needed |
| --- | --- | --- |
| 1 | 1.180 m | ~0.4 m |
| 2 | 1.870 m | ~0.4 m |

With `max_linear_commands: 1` at ~1.06 m per command, leg 1 stops **0.12 m**
short (just outside the 0.08 m tolerance) and leg 2 stops **0.81 m** short.
Neither reports `target_reached`, so neither passes Gate 5.

~~**Usable leg band: 0.3–0.5 m.**~~ **Refuted by the beta16 0.400 m run.** Above
~1.0 m one linear command cannot finish it; below 0.08 m it is inside
`waypoint_tolerance` and may count as already arrived. The intervening short
range is not yet characterized because onset/dead time breaks zero-origin pulse
scaling.

~~**Coordinates cannot be typed into the card.**~~ Superseded by beta16's
guarded 0.001 m coordinate editor.

### Two things learned setting this up

- **`stale_route_while_ready` is a real, named, non-blocking case.** Undocking by
  starting a mow and pausing leaves `route_present: true`,
  `progress_is_active: true` with `blocks_motion: false`. Cancelling the mow
  clears it to `no_route`. Not a defect; know it exists.
- ⚠️ **`sensor.*_vio_heading` is coordinator-tick cached** and stayed
  bit-identical across 374 samples. It is **not** fine-grained enough to measure
  a turn. For the offset re-derivation use the **run result JSON**
  (`vio.initial_vision_heading`, and the turn phase's before/after), not the
  sensor entity. `scripts/motion_capture.py` is still the right tool for the RTK
  position track and for proving whether `toward` updates.

## Gate 5 setup retry 2026-08-02 afternoon — NO MOTION

The operator gave a fresh daylight `GO`, but the motion gate remained off while
the card geometry was checked. The shortest practical clicks produced legs of
**0.767 m** and **0.934 m**, still outside the 0.3–0.5 m acceptance band. The
dry-run was valid with live VIO (`Light`, 80 features) and no segment blockers,
but it was not suitable for Gate 5, so Real Go was not enabled and no motion
occurred. A 402-sample capture stayed bit-identical at (4.795, −1.9502).

This exposed a card usability blocker rather than a motion-profile defect: the
full-map click/drag surface cannot reliably place sub-metre legs. Beta16 adds a
guarded coordinate editor for existing waypoints at 0.001 m precision. Every
coordinate edit clears stale dry/real results and re-runs backend Preview; Real
Go still requires a valid final preview and dry-run. The accepted motion profile
is unchanged.

Evidence:

- `docs/evidence-gate5-setup2-dry-run-20260802.json`
- `docs/evidence-gate5-setup2-no-motion-20260802.jsonl`
- `docs/evidence-gate5-setup2-ble-report-20260802.txt`

## Start here

- Branch: `feat/vio-turn-to-heading`, pushed to `Chorty`. Working tree clean.
- Personal-fork PR: [Chorty/Mammotion-HA#10](https://github.com/Chorty/Mammotion-HA/pull/10),
  still a draft. Checks were green on the pushed beta12 work (python and
  hassfest pass; HACS `skipping` is expected on a fork).
- Safety state: experimental motion **off** (verified), no session, mower at
  approximately x 5.049, y -2.753 in `Backyard Right`, blades off.
- Do not push, comment, open issues, or otherwise write to a `mikey0000`
  repository. The upstream remotes have disabled push URLs. A later push, if
  authorized, goes only to `Chorty/Mammotion-HA`.
- Do not merge fork `main`, mark the PR ready, or dispatch a beta release until
  the remaining release checks below are resolved and current CI is green.

The exact safety model, deployment instructions, chronological hardware record,
and limitations are in `docs/p0-beta-release.md`. Do not reconstruct live-test
facts from chat history when that document has the structured result.

## What is complete

- P0 backend surfaces are implemented: fail-closed capability registry,
  lowercase HA enum migration, diagnostics redaction, internal-only camera
  credentials, task/map services, experimental-motion option, exclusive motion
  sessions, `stop_manual_motion`, runtime gate export, and the click-to-go card.
- Preview and dry-run allow seven destinations. Real click-to-go is BLE-only,
  opt-in, capability-probed, and limited to two segments.
- PyMammotion is pinned consistently to the Chorty `0.8.12.post1` wheel. It is
  upstream `v0.8.12` plus the rate-limit fix, BluFi reassembly reset, and BLE
  teardown fix. No official upstream release contains the teardown fix.
- The deployed LUBA uses PyMammotion `0.8.12.post1` and integration version
  `0.6.4-beta11`. `coordinator.py` and `__init__.py` match this tree. The local
  `services.py` differs from the deployed checksum only by the corrected schema
  comment about VIO refresh; functional code is the code that passed Gate 4.
- The sole tested proxy path is P1S with passive BLE scanning and active GATT
  proxying:

  ```yaml
  esp32_ble_tracker:
    scan_parameters:
      active: false

  bluetooth_proxy:
    active: true
  ```

- Supervised LUBA acceptance Gates 1-4 all passed:
  - three-write confirmed zero stop;
  - bounded straight segment (9.69 cm toward 10 cm, 5.6 mm final error);
  - active-session abort (owner returned in 673 ms; no post-abort nonzero
    dispatch or replay);
  - 176-degree VIO regression (4.44-degree residual, 10.48 cm turn drift);
  - corrected two-leg L path (both 30 cm segments `target_reached`, final error
    4.70 cm, no delayed replay).
- Gate 4's independent stop confirmed all three zero writes in 530.4 ms. The
  mower stayed bit-identical for 18 seconds, blades stayed off, no session
  reappeared, and experimental motion was disabled afterward.

## Local changes in the handoff commit

1. A cloud-backed mower now registers its late BLE advertisement callback even
   when no proxy is ready during entry setup.
2. A temporarily absent transport reports `none` instead of raising during HA
   entity setup.
3. Bluetooth option changes invalidate the five-second motion-gate cache and
   refresh entities immediately. Enabling survives a temporarily unavailable
   advertisement so the later callback can attach.
4. `ManualMotionCancelledError` propagates out of the refresh loop after a
   defensive stop, so an operator abort releases the exclusive owner promptly.
5. A translating VIO turn now receives the configured displacement limit,
   recalculates the bearing from fresh post-turn position, and fails before
   linear motion unless alignment is freshly proven. A bounded correction has
   at most two turn commands and shares the normal realignment budget.
6. Regression coverage was added for all of the above. The exact handoff tree
   passed all 456 Python tests with coverage after the documentation pass.

## Validation at handoff

Passed on the exact pre-commit tree, after the card-profile and pre-commit work:

- 456 Python tests with coverage;
- CI-scoped Ruff (`custom_components tests`) — and, separately, `ruff check .`
  is clean repo-wide including `scripts/`;
- CI-scoped Ruff format over 48 files;
- CI-scoped mypy over 28 source files;
- all **eleven** frontend card tests (six previous, four profile assertions, and
  a README drift guard);
- 15 integration JSON files parse;
- `git diff --check`;
- `pre-commit run --all-files` — **green, and modifies nothing.**

Re-run all of the above after any further card or default change; the frontend
tests are what stop the card silently drifting off the accepted profile again.

## Assumptions that changed during hardware testing

These corrections matter more than the original chat plan:

- The recurring app/HA BLE conflict was not evidence that the mower radio was
  defective. Disabling the integration let the official app connect
  immediately; isolating proxies then implicated the IRK proxy path. P1S
  restored immediate app BLE access and stable confirmed writes. Do not re-enable
  multiple proxies during acceptance without a new controlled comparison.
- Passive `esp32_ble_tracker` scanning does not make a proxy passive for GATT.
  `bluetooth_proxy.active: true` is required and is the configuration that
  passed. `ble_link_live` may not be bypassed; routing preference alone does not
  prove a writable link.
- A VIO pivot is not an in-place geometric turn. It translated 14.43 cm in the
  failed Gate 4 attempt and changed the bearing enough to miss. Always compute
  the forward bearing from post-turn position. The corrected retry translated
  8.80 cm but freshly proved a 0.285-degree aim error before driving.
- Delayed RTK reports can make bounded physical motion appear later. Treat
  replay as a session/dispatch fact, not merely as a later position change.
- Do not increase the four-second BLE write timeout to hide stalls. That also
  lengthens uncertain nonzero delivery. The confirmed-dispatch and emergency
  stop behavior is the safety boundary.
- There is still no firmware arbitrary-waypoint upload API. The accepted path
  is a guarded chain of raw manual-motion segments, not autonomous navigation.

## Card execution profile — RESOLVED (backend-equivalent, still not UI-accepted)

The backend Gate 4 call used a deliberately bounded acceptance profile. That
profile is now the card's built-in default, named and frozen as
`LUBA_ACCEPTANCE_PROFILE` at the top of
`custom_components/mammotion/www/mammotion-custom-path-card.js`:

- `max_real_segments: 2` (already `MAX_REAL_SEGMENTS`)
- `max_turn_commands: 4`, `vio_turn_max_commands: 4`
- `max_linear_commands: 1`, no loop-to-tolerance ceiling
- `waypoint_tolerance: 0.08`
- `min_progress_distance: 0.0025`
- `calibrated_forward_heading_offset_degrees: 102.4`
- `motion_refresh_interval_ms: 200`
- `ble_auto_recover: false`

Both option 2 and option 3 from the previous handoff were taken: the accepted
profile *is* the default **and** it is a named, exported, test-pinned object
rather than values scattered through `setConfig` and `_motionPayload`.

Implementation notes that matter:

- `max_linear_pulse_ceiling` is `null` in the profile and the key is **omitted**
  from the payload, because the backend schema is `vol.Optional` with no default
  and `Range(min=1)` — sending `0` would be a validation error, and sending any
  number would re-enable loop-to-tolerance, which Gate 4 did not use.
- Resolution is `??`-based (`_profileValue`), never `||`. The old code used
  `Number(this._config.x || default)`, which silently discarded a configured `0`
  — including `motion_refresh_interval_ms: 0`, the legacy single-shot mode.
- `_profileOverrides()`/`_profileLabel()` compute which profile keys the
  dashboard YAML overrode. The card renders an **execution profile** row that
  reads either `LUBA acceptance profile (Gates 1-4, 2026-07-31)` or
  `customised (not hardware-accepted): <keys>`. An operator can now see from the
  card whether the payload is the accepted one.
- `heading_tolerance_degrees` stays at 18. It is a known-loose value from the
  July 18 single-shot calibration, but it is what the accepted run used, so
  changing it here would have left the accepted profile in the name of tuning.

Frontend assertions and README YAML were updated in the same change: four new
DOM tests pin the profile values, the omitted ceiling, the override labelling,
and the falsy-value regression. A fifth test pins the README block itself —
README is a third copy of these numbers in paste-ready YAML, and nothing else
stopped it drifting from the values the hardware ran (11 frontend tests total).

Independently re-verified this session: the payload the card emits from an
unmodified dashboard config validates against the *real*
`RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA` and
`RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA`, with
`max_linear_pulse_ceiling` absent from the validated result. The frontend tests
alone could not have shown that — they never cross the JS/Python boundary.

**Still true, and the reason this is not a release sign-off:** the card has
driven the mower but never completed both segments. Gates 1-4 exercised the
*backend* with these values, while the beta16 card run exposed a short-pulse
distance-model failure. Any repeat requires diagnosis and a **new** daylight
operator `go`. No physical motion is authorized by this handoff.

✅ Versions were bumped together to `0.6.4-beta12` for the Gate 5 build:
`manifest.json`, `pyproject.toml`, `CARD_VERSION`, and `uv.lock` (PEP 440 form
`0.6.4b12`). The bump is **local only** — nothing was deployed, and the host
still runs beta11. The point of the bump is that a Gate 5 run must be able to
prove in the browser console banner that the new card loaded; the card is
served from two paths, so deploy to both or the stale card is served silently.

## Remaining P0 work — dispositions

1. **Card-default mismatch — DONE.** See the section above.
2. **The three deferred defects — two were already fixed, one is reframed:**
   - *Map edits not visible until HA restart:* **already fixed** in `6cf4d5fd`,
     before this handoff was written; the entry was stale. The per-tick block is
     reachable now because `_async_short_circuit_update()` returns `None` on the
     healthy path and every caller tests `is not None` rather than truthiness.
     Covered by `test_short_circuit_update_returns_none_on_the_healthy_path`,
     `test_every_coordinator_tests_short_circuit_with_is_not_none` (an AST check
     over all five coordinators), and
     `test_update_loop_only_starts_a_map_sync_through_the_gate`.
   - *False turn-phase `no_actuation_detected`:* **already fixed** in the same
     commit, and the suggested fix was wrong. `heading_went_fresh` cannot be the
     discriminator: it is True exactly when before/after differ by more than the
     epsilon, which is exactly when `_streak_shows_no_actuation` (bit-identical
     heading) is False. The two are perfectly correlated, so gating on it would
     delete the no-actuation branch instead of refining it. The real
     discriminator is `heading_poll_feed_alive` — "did *any* channel move at
     all" during the poll, since a live feed jitters ~2-4 mm in position and
     ~0.0018 deg in heading even when stationary. `_streak_shows_dead_telemetry`
     runs first and reports `vio_telemetry_stream_stale`; `no_actuation_detected`
     now only fires with a demonstrably live feed. A replay of the exact
     2026-07-25 run is pinned as a regression test.
   - *Task-2 constants:* **reframed, not release-blocking.** The original
     statement dates from when Step 5b had died twice on transport. Since then
     the 2026-07-27 3.0 m segment landed 1.0 cm along-track, and Gates 1-4
     executed the pulse-geometry ceilings, `min_progress_distance` and cadence
     on hardware. Those constants are no longer hypotheses — they are the values
     in `LUBA_ACCEPTANCE_PROFILE`, now pinned by tests. What is genuinely
     un-re-derived is narrower: `heading_tolerance_degrees` (18, derived from the
     single-shot rotation quantum that refresh made obsolete) and the refreshed
     turn-pulse floor. Both are beta tuning behind a new operator `go`, not
     release gates, because the release ships values hardware actually ran.
3. **All-files pre-commit — REPAIRED, now green.** See the next section.
4. **Version agreement — bumped to `0.6.4-beta12`.** `manifest.json`,
   `pyproject.toml`, `CARD_VERSION` and `uv.lock` all agree, and
   `manifest.json`, `pyproject.toml` and `requirements_test.txt` all declare the
   identical `chorty-0.8.12.post1` wheel URL. The next `Beta Release` dispatch
   will compute `0.6.4-beta13` (dry-run verified).
   **New:** `Beta Release` could not have run at all before this session —
   doubled backslashes in YAML block scalars made every sed capture group fail,
   so it proposed the already-existing `v0.6.4-beta1` on every dispatch and
   exited 1, and its `uv.lock` verify grepped a package name that is not in the
   file. Fixed in `.github/workflows/beta-release.yml`; both steps dry-run
   against this tree. This is also the mechanism behind the previously recorded
   version regression.
5. With explicit operator authorization, push only to the Chorty feature branch
   and wait for current PR checks. HACS `skipping` on a fork is expected; Python
   and hassfest must pass.
6. Only after the card profile decision and current CI pass: mark PR #10 ready,
   merge to Chorty `main` without force-pushing, and run `Beta Release` with the
   LUBA-acceptance confirmation. Never publish to Mikey's repositories.

## All-files pre-commit — repaired

`pre-commit run --all-files` now passes and modifies nothing. Each failure had a
distinct cause, and each was fixed rather than scoped away where fixing was
cheap:

- **Hook/CI version skew (the root of the mypy and Ruff noise).** The `ruff`
  hook pinned `v0.12.8` while CI pins `ruff==0.15.16`; the older ruff still
  enforced `UP038`, a rule later ruff removed, so the hook failed on two lines
  CI passes. Pinned to `v0.15.16`. Likewise `mirrors-mypy` was `v1.17.1` against
  a `mypy==2.1.0` pin; now `v2.1.0`. Both pins must move with
  `requirements_test.txt` or the gate lies again.
- **mypy scope.** The hook ran `--strict` over all of `custom_components` and
  reported 168 errors, essentially all `Class cannot subclass "SensorEntity"
  (has type "Any")` — an artifact of HA shipping untyped entity base classes to
  the checker, and none of it checked by CI. Now `--follow-imports=skip` over
  `custom_components/mammotion/`, matching CI exactly. Passes.
- **Ruff over `scripts/`.** 22 real findings, so they were fixed, not scoped
  out: an unused `sys` import, three missing docstrings and a
  `subprocess.run(check=...)` in `scripts/linear_duration_sweep.py`. The 17
  `T201` prints are legitimate — these are operator CLIs whose entire output
  contract is stdout — so `scripts/*.py` gets a documented `T201` per-file
  ignore. `scripts/` stays linted otherwise. `ruff-format` still skips
  `scripts/`, matching CI, so a live investigation cannot be reformatted
  mid-flight.
- **codespell.** `--skip="./.*,*.csv,..."` was written with literal quotes
  inside a YAML args list, so the whole skip list was inert. Fixed, and the
  APK-verbatim identifiers (`entitys`, `swtich`, `piar`, `buttom`, `unknow`) plus
  `unparseable` are now in `--ignore-words-list`, because correcting them would
  stop them matching the thing they document. The two standalone browser dev
  tools are skipped.
- **pyupgrade removed.** It crashed on every file under Python 3.14
  (`tokenize.cookie_re` is a bytes pattern there, so `_fix_tokens` raises
  `TypeError`) and was redundant: ruff already selects the `UP` ruleset.
- **prettier scoped to the JS this integration ships**
  (`custom_components/mammotion/www/*.js`, `tests/frontend/*.mjs`). Unscoped it
  rewrote ~20 Markdown evidence files, the APK feature catalogue,
  `services.yaml` and `manifest.json` on every run. Bringing `agora-client.js`
  into scope reformatted it once; that diff was verified non-semantic
  (quotes, trailing commas, wrapping, arrow parens only) and it is a
  redeploy-worthy but behaviourally identical file.
- **`*.patch` protected.** `trailing-whitespace` and `end-of-file-fixer` were
  stripping trailing whitespace from `docs/upstream-patches/*.patch`, where it
  is significant in unified-diff context lines — the hooks were silently
  corrupting patches so they would no longer apply. Both now exclude `\.patch$`.
- **yamllint config.** `args: -c .yamllint.yaml` sat one level out, as a key of
  the repo mapping rather than the hook, so `.yamllint.yaml` was never applied.

The config now carries a scope rule at the top: every hook must agree with
`.github/workflows/validate.yml`, and any deliberate narrowing states its
reason inline.

## Safety state at handoff

- `enable_experimental_motion: false`
- no active motion session
- last Gate 4 session completed normally
- mower last reported `MODE_READY`, blades off, fixed RTK, inside `Backyard Right`
- last settled map position: x 4.3911, y -2.8064
- no live test may reuse the previous operator confirmation

## Useful commands

```sh
git status --short --branch
git log -1 --stat
.venv/bin/python scripts/mammotion_preflight_gates.py --quick
.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests
.venv/bin/python -m ruff check custom_components tests
.venv/bin/python -m ruff format --check custom_components tests
.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion
npm run test:frontend
```

Never enable broad PyMammotion debug logging: cloud and raw BLE loggers can
expose credentials, network responses, device identifiers, and payloads. Use
only the scoped `bleak_esphome` and `habluetooth` loggers documented in
`docs/deploy-runbook-p0.md`.
