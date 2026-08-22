# Outstanding items — 2026-08-21

Compiled by walking the tree, not by copying the previous handoff. Every claim
below was checked against code, a test, or a live host query on 2026-08-21; the
check is named next to each item so the next session can re-run it rather than
trust this file. `CLAUDE.md`'s own warning applies here too — **one grep beats
this document.**

Ordered by consequence, not by effort.

---

## 1. 🚨 The motion gate keeps arming itself at rest — now observed twice in ONE session

**Status: unresolved, and the frequency is increasing.**

`CLAUDE.md` records four armed-at-rest occurrences. This session adds two more:

- At session start the gate read `enabled: true`, `real_motion_allowed: true`,
  `blockers: []`, mower off the dock — despite the handoff stating it was
  "DISARMED and verified". It was disarmed and independently verified
  (`enabled: false`, `blockers: ['experimental_motion_disabled']`).
- **Under 20 minutes later it read armed again.** That second arming was the
  operator deliberately driving with the click-to-go card, so it is explained —
  but it is exactly why the *first* reading cannot be dismissed as noise: there
  is no record distinguishing "armed on purpose" from "left armed".

⚠️ **The disarm here may itself have interrupted an operator run.** An agent
that disarms on sight and an operator who arms to drive will fight each other
silently. That is an argument FOR the automation, not against it: a timer-based
disarm is visible and predictable, an agent's judgement call is neither.

`docs/automations/disarm-motion-gate.yaml` **exists in the repo** and is
**not installed** on the host. Operator's call — it has been deferred four
times, and the cost of it firing mid-run is one re-arm.

*Checked:* two live `export_runtime_state` calls; `ls docs/automations/`.

---

## 2. 🆕 Motion is 71% standing still — the "slow and not fluid" complaint, measured

**Status: offline controller/replay and Phase 1 capture analyzer implemented;
no runtime executor or hardware continuous-control test exists. The mower is
charging at night, so both separately authorized Phase 1 captures remain
pending until daylight.**

`continuous_controller.py` now implements a pure, non-dispatching lookahead
decision with bounded prediction and fail-closed results for stale telemetry,
refresh/BLE faults, invalid RTK/area/blade/work-mode state, cancellation,
containment, cross-track, time, and distance limits. The standalone replay says
`dispatch_capable: false`, sends zero commands, and has focused fault tests.
The standalone Phase 1 analyzer evaluates paired capture files, independently
recomputes the written timing/course/containment criteria, hashes its inputs,
and also has no dispatch path. Its `go` is not motion authorization. See
`docs/phase1-capture-analyzer.md`.
The staged go/no-go plan is
`docs/continuous-motion-feasibility-plan-20260821.md`.

Next evidence: freshly scan contained straight and shallow-arc routes, run the
two 4 s profiles under separate explicit authorizations, bank both complete
responses, and evaluate them with the offline analyzer. Do not begin Phase 2
from dry-run evidence or from the existence of the analyzer itself.

Reconstructed from `sent_at_utc` in
`docs/evidence-routeb-retry-overshoot-20260820.json`:

| | |
|---|---|
| span, first to last dispatch | **65.3 s** |
| linear pulses | 13 |
| mean pulse-to-pulse cycle | **4.55 s** |
| of which commanded motion | **1.30 s** (`linear_pulse_duration_ms`) |
| **duty cycle** | **29%** |
| total spent in position-settle | **37.1 s (57% of the run)** |

Net progress was 3.83 m in 65.3 s = **0.059 m/s**, against roughly **0.30 m/s**
while a pulse is actually running. **The mower is ~5× slower end-to-end than it
is while moving.** That gap is the complaint.

🔑 **The settle wait is quantized to its own poll interval.** The 13 recorded
waits were 3.01 / 4.01 / 2.0 / 3.01 / 3.01 / 3.01 / 3.0 / 3.01 / 3.0 / 3.0 /
3.01 / 1.0 / 3.01 — **every one within 20 ms of a whole second**, because
`_settle_linear_position_feed` polls at `poll_interval_seconds: 1.0`. Mean
2.85 s is therefore "about three polls", not a measured physical settling time.

⚠️ **Note what is NOT the cause.** The beta55 `sample_delays: [0, 3]` fix works:
all 13 commands recorded `post_settle_feedback.additional_wait_seconds: 0.0`
with `requested_sample_delays_skipped: [0.0, 3.0]`. Do not re-fix that.

Levers, cheapest first — **none of these are validated, and the first two are
measurements waiting to be taken, not recommendations:**

1. ~~**`poll_interval_seconds` 1.0 → 0.5** in `_settle_linear_position_feed`.~~
   🗑️ **REFUTED THE SAME DAY — do not spend time here.** The position feed
   updates at **~1 Hz while moving**, so the settle loop's 1.0 s poll is already
   matched to it. Two banked dense captures, each oversampling ~10×, both give a
   median inter-arrival of **1.02 s**, and **none of the 9 pooled intervals is
   under 0.5 s**. Halving the poll would double the BLE load and read the same
   value half the time. `docs/evidence-position-report-cadence-20260821.json`.

   🔑 **And this reframes the settle as a floor, not slack.** Settling needs the
   feed to move off the pre-pulse value, then two consecutive snapshots to agree
   — at 1 Hz that is at least two arrivals (~2 s), plus the time for the pulse's
   motion to land in a feed that is itself ~1 s stale. The measured 2.85 s mean
   is **near the physical floor of stop-measure-go at this feed rate.** Corroborated
   independently by `docs/evidence-report-rate-probe-20260807.json`, which found
   the requested wire period is *not honoured* (observed ratio 1.2 against 5.0
   expected) and saw a ~1 s ceiling in total traffic.

   ⚠️ Note what this kills: **the cheap fix does not exist.** Dead time per pulse
   cannot be tuned down. It can only be paid fewer times.

2. **`linear_speed_fast` 400 → higher — now the only lever left inside the
   pulsed design.** Since the ~3 s settle is per pulse and nearly fixed, the
   only way to cut total dead time is to need fewer pulses. Also **not** a frozen profile key
   (verified against the `LUBA_ACCEPTANCE_PROFILE` literal — the frozen set is
   19 keys and linear speed is not among them). The app's own ceiling is 850, so
   400 is 47% throttle. Fewer, longer pulses per metre means fewer settle waits,
   so this compounds. 🚨 **But `waypoint_tolerance: 0.15` — which IS frozen —
   was chosen because the mower covers 30–47 cm during the ~1031 ms feed
   staleness at speed 400.** At 600 that becomes ~45–70 cm and the tolerance
   almost certainly has to move with it, which owes a Gate 5. Speed and landing
   accuracy are coupled through the stale feed; treat them as one change.
3. **Continuous motion instead of stop-measure-go.** This is the actual answer
   to "not fluid" — the jerkiness *is* the pulse cycle — and the evidence that
   it is feasible already exists:
   - the position feed during **continuous** vendor motion measured **sub-cm**
     (0.70 cm cross-track RMS, zero frozen samples, 2026-07-21/22). The 2–4 cm
     noise floor that motivates settling is **a pulsed-measurement artifact**;
   - `toward` tracks course continuously while translating — 40 moving steps
     gave `travel bearing + toward = 90.57°`, 2.02° circular SD;
   - arcs are linear and controllable (`docs/arcs-work-20260812.md`).

   ⚠️ This is a **new control law**, not a tuning change: it re-opens every gate,
   and the whole safety argument (bounded pulse, stop between decisions) is built
   on the current shape. Large. Do not start it casually.

*Checked:* timeline reconstructed from the evidence file; profile key list
extracted from the card literal; `poll_interval_seconds` read at
`services.py` `_settle_linear_position_feed`.

---

## 3. Keep-out containment is now per-segment in the working tree

**Status: fixed, released, installed, and verified zero-motion in beta69.**

`_keep_out_leg_violations` now checks every legal-endpoint leg against every
keep-out edge, including boundary touches and collinear overlap.
`_validate_custom_path` refuses with `path_legs_cross_keep_out_zone`, while the
card mirrors the same geometry and blocks Real Go locally. The former gap test
is now `test_a_leg_that_clips_a_corner_is_caught`; split-path behavior and a
clear negative case are pinned too. A real-map crossing preview was refused
solely by the new reason, its legal control passed, and the browser showed the
named blocker with Real Go disabled.

beta63 made waypoint exclusion work; beta69 closes the segment gap. The scanner
still samples the whole leg every 5 cm because its clearance margin is stricter
than boundary-only containment, not because the backend is blind between
points.

*Checked:* `test_a_leg_that_clips_a_corner_is_caught`, live real-map crossing
and legal-control previews, and deployed browser behavior.

---

## 4. Route B retry at 3.0 m sub-legs — completed

**Status: completed 3 of 3 on beta69; reliability question remains open.**

The authorized 9.0000 m route split into three 3.0000 m collinear legs and all
three stopped `target_reached` at **0.14388 / 0.11413 / 0.06070 m** (mean
0.10624 m). It ran from `(4.8756, -2.4530)` to `(11.7193, -8.2980)`, with
measured scan clearances of 1.21 m to the area edge and 4.41 m to keep-outs.
The gate was verified disarmed afterward. Evidence:
`docs/evidence-route-b-3x3m-beta69-20260821T193417Z.json`.

Combined 3.0 m landing evidence is **5 reached / 1 failed, n = 6**. This closes
the feasibility question (a 3-of-3 chain can complete), but one complete run
does not make 3.0 m a reliable control regime.

`split_leg_target_length_m` accepts **0.5–6.10** (`services.py:1426`) and is
**not** a `LUBA_ACCEPTANCE_PROFILE` key, so 3.0 costs no Gate 5.

The prior risk remains informative. The 3.85 m failure was an 18.083° opening
aim error → a mid-drive correction after pulse 2 → two suppressions on final
approach → 0.16734 m landing on `target_requires_reverse_recovery`. **Leg length
was not obviously the proximate cause**, and a 3.0 m leg opening at 18° can
reproduce every step. The new success does not erase that failure mode.

🔑 **The reachable distance depends entirely on where the mower is standing, and
it moves.** Two scans 20 minutes apart in this session:

| mower position | best contained reach |
|---|---|
| (5.702, −5.1366) | **10.15 m** — 11.5 m did not fit at *any* bearing, even at zero margin (max 11.45 m) |
| (7.3963, −0.3307) | **12.30 m** at heading 303.0° |

**So "an ~11.5 m click" is not a property of the yard — re-scan immediately
before every run.** Route B is now **1 for 3** end-to-end; neither earlier
failure was the splitter.

*Checked:* schema range read from source; scans run live against `export_map`;
full beta69 response banked and final gate state verified.

---

## 5. `max_linear_pulse_ceiling: 14 → 22` is still untested

Needs a leg over ~5 m to bind. The profile is currently **ACCEPTED**
(`scripts/check_accepted_profile.py`, exit 0, matching
`docs/evidence-gate5-beta57-20260818.json`), so this is an untested *headroom*
value rather than an unaccepted one.

---

## 6. `safety_overrides` is not wired into the movement primitives

`MOVEMENT_SCHEMA` and `MANUAL_VELOCITY_PULSE_TEST_SCHEMA` **cannot express an
override** — verified by extracting both schema literals; neither mentions
`safety_overrides`, while the segment executor's schemas at `services.py:1258`
and `:1395` do.

That gap is *why* the nudge buttons had to be ungated outright rather than
override-gated. Closing it would let the deliberate-override mechanism cover the
stranded-mower case, and let the nudge buttons go back behind a gate.

⚠️ **Until then, do not "fix" the ungated nudge buttons** —
`tests/components/mammotion/test_nudge_buttons_ungated.py` pins the operator's
2026-08-20 decision and explains it.

---

## 7. The card does not draw keep-out zones

`export_map.keep_out_polygons` has been available since beta63; the card
references it **zero** times. Refusing at click time beats refusing at dispatch,
and it would have made the trampoline click un-clickable rather than merely
refused.

---

## 8. Two standing measurement gaps

- **No leg over 4 m exists in the corpus**, against a 6.10 m pre-dispatch cap.
  The regime the reach work exists for still has no data.
- **`vio_max_realignments: 3`** is answered on replay (0 of 62 segments exceed
  it) but not on hardware in the far field. A leg following a junction turn has
  an effective mid-drive budget of **2, not 3** — the post-turn gate spends the
  same counter. ✅ Raising it was tried, reviewed twice, and reverted twice;
  **do not retry without hardware evidence.**

---

## Explicitly NOT open

- **Accuracy** — closed by standing decision; the 0.065 m intercept is a sensing
  floor, not a tuning target.
- **Night** — contained exploration, parked on purpose.
- **The beta42 quadrature term** — verified applied this session; the claim that
  it "needs checking" was wrong and is corrected in `845ddf3e`'s evidence file
  and in `docs/NEXT-SESSION.md`.
- **Non-LUBA hardware** — moot; this yard only.
