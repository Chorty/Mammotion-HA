# Turn-rate variance & per-click reach — read-only investigation, 2026-08-08

Produced on build `0.6.4-beta30`, branch `feat/gate4-day2j-profile`, after Gate 5
passed. **No repo file was modified, no command was sent to the mower, and Home
Assistant was not contacted.** Every claim is cited to `file:line`.

Method: six independent investigation angles, each attacked by a separate
adversarial verifier, then a completeness critic. 13 agents, 0 errors, final
tally **110 CONFIRMED / 19 REFUTED / 1 UNVERIFIABLE**. Companion file:
`turn-rate-variance-completeness-critique-20260808.md`.

## ⚠️ READ THE CORRECTIONS — this document is append-ordered

Later sections **supersede** earlier ones, in the project's usual house style.
Before acting on anything above, read these three, in order:

| section | what it overturns |
| ------- | ----------------- |
| `FINAL STATUS (workflow run 2)` | §A2's spread framing — the *overall* spread only goes 2.60x → 2.26x |
| `§B1 CORRECTED` | **the reach numbers in §B1 are wrong.** Per-segment reach is ~0.5–1.0 m, not 1.2–1.7 m |
| `COMPLETENESS CRITIC` | the four stop latencies are **not** turn-pulse stops; the 110.26° sum is not trustworthy |

Two standing cautions carried from the investigation:

- `path_m` in `evidence-gate5-PASSED-20260808.json` is the **planned** leg
  length, not travel. Only `independent_trace` carries real travel.
- Two verifiers wrongly reported `REAL_CLICK_TO_GO_SEGMENT_LIMIT` does not
  exist. It does: `manual_motion.py:24`, imported `services.py:42`, Range
  `services.py:1054`, enforced `services.py:11490-11497`.

---

# Turn-rate variance & per-click reach — analysis

Status: **interim**, written 2026-08-08 while workflow `wf_cec0c5f1-051` verifiers
were still running. Everything below was read directly from the tree at
`/Users/mattjoslin/Documents/Git Projects/Mammotion-HA` on build `0.6.4-beta30`.
Raw agent output is the sibling file `FINDINGS-turn-variance-and-reach.md`.

Read-only session: nothing in the repo was modified, nothing was sent to the
mower or to Home Assistant.

---

## PROBLEM A — is the rotation varying, or the measurement?

### A0. The premise "displacement scaled WITH rotation" does not survive contact with the code

`command_result["displacement_m"]` is **cumulative from the start of the whole
turn**, not a per-pulse delta:

```python
displacement = _telemetry_position_delta(
    initial_telemetry, after_telemetry
).get("distance")
```

`services.py:8077-8079`. `initial_telemetry` is captured once at
`services.py:7654`, before the pulse loop, and is never reassigned. The comment
at `services.py:8095-8098` says so outright: *"Cumulative translation during the
turn."*

So `0.0505 / 0.0726 / 0.1122` is a **monotonically non-decreasing running total**.
A running total rising alongside a running total cannot be evidence that
per-pulse displacement tracks per-pulse rotation. Implied per-pulse values are:

| pulse | rotation (deg) | cumulative disp (m) | per-pulse disp (m) | m/deg as-if-per-pulse | m/deg actual |
| ----- | -------------- | ------------------- | ------------------ | --------------------- | ------------ |
| 1 | 30.46 | 0.0505 | 0.0505 | 0.00166 | 0.00166 |
| 2 | 22.17 | 0.0726 | 0.0221 | 0.00328 | 0.00100 |
| 3 | 57.63 | 0.1122 | 0.0396 | 0.00195 | 0.00069 |

Read correctly, m/deg **falls** monotonically across the three pulses — the
opposite of "scaled with". Note also `_telemetry_position_delta` is a
straight-line distance between two points, so during an in-place turn about an
offset centre it is not even guaranteed to increase.

**Therefore the displacement channel does not discriminate between the
hypotheses, in either direction.** It should be dropped from the argument.
(This is the project's own `verify-with-per-item-records-not-aggregates` rule.)

### A1. The code reading is pinned against the observed run

The rate estimator accumulates **nominal** pulse duration, not measured elapsed:

- `services.py:8089-8091` — `observed_rotation_degrees += abs(measured_change)`;
  `observed_rotation_ms += pulse_ms` (the *commanded* duration).
- `_turn_final_approach_pulse_ms`, `services.py:7446-7452` — `rate =
  observed_rotation_degrees / (observed_rotation_ms / 1000)`, a cumulative
  average.

Check against the operator's report that the executor revised
`degrees_per_second` from the 37 default to 20.3, then 17.5:

- 30.46 / 1.5 s = **20.31**
- (30.46 + 22.17) / 3.0 s = 52.63 / 3.0 = **17.54**

Both reproduce exactly. That pins this reading of the code to the actual run.
`37.0` is `_DEFAULT_TURN_DEGREES_PER_SECOND` (`services.py:7389`).

### A2. Recomputing the rate against three different actuation units

| pulse | rotation | nominal 1500 ms | measured elapsed_ms | non-zero writes (refreshes+1) |
| ----- | -------- | --------------- | ------------------- | ----------------------------- |
| 1 | 30.46° | 20.31 °/s | 2043 → **14.91 °/s** | 4 → **7.62 °/write** |
| 2 | 22.17° | 14.78 °/s | 1530 → **14.49 °/s** | 3 → **7.39 °/write** |
| 3 | 57.63° | 38.42 °/s | 1760 → **32.74 °/s** | 3 → **19.21 °/write** |

- Against **nominal time**: pulses 1 and 2 differ by **37%**.
- Against **measured elapsed**: pulses 1 and 2 differ by **2.9%**.
- Against **non-zero write count**: pulses 1 and 2 differ by **3.1%**.

So two of the three "identical" pulses were not identical at all — they had
different actual actuation, and once you divide by real actuation instead of
commanded actuation they agree to ~3%. **Pulse 3 is the sole genuine anomaly**,
at ~2.2× the others under either corrected unit.

Elapsed and write-count cannot be separated with two agreeing points; both fit
equally well.

### A3. Why elapsed_ms differs from the nominal 1500 ms — the mechanism

`_motion_refresh_window`, `services.py:4895-4946`:

- `max_refreshes = int(duration_seconds / interval_seconds)` = int(1.5/0.2) = **7**
  (`:4905`) — so the count was never limited by the budget.
- Each loop iteration is `sleep(min(0.2, remaining))` (`:4913`) **then** `await
  resend()` (`:4918`), and the deadline is only re-checked *before* the resend
  (`:4914`), never during it. A write that starts at 1.4 s and takes 861 ms
  therefore runs the window to 2.26 s.
- Hence `elapsed_ms` can exceed `duration_seconds`, and the refresh count is set
  by write latency, not by the interval.

The model reproduces the observations:

- 2 refreshes, elapsed 1760 ms ⇒ 400 ms of sleeps + W1 + W2 = 1760 ⇒ ΣW = 1360 ms
  (consistent with one write near the observed 861 ms max).
- 2 refreshes, elapsed 1530 ms ⇒ ΣW ≈ 930 ms, the third sleep truncated at the
  deadline.
- 3 refreshes, elapsed 2043 ms ⇒ 600 ms of sleeps + ΣW = 1443 ms.

At a median write of 540 ms the effective re-send period is ~740 ms, versus the
app's 200 ms that the whole refresh design copies (`services.py:4840-4856`).

### A4. The linear phase already solved this problem; the turn phase never got the fix

`_final_approach_pulse_ms` docstring, `services.py:9972-9977`:

> *"Duration-only refreshed scaling was also disproved live on 2026-08-02:
> 1012.5 ms delivered two refreshes and moved 0.1786 m, while 1191.8 ms delivered
> three and moved 0.4341 m. Confirmed BLE write and stop latency make nominal
> time a poor actuation unit. Bound the discrete non-zero writes instead."*

The linear phase acts on that: it bounds a pulse by `refresh_command_limit`
(`services.py:10034-10037`) and normalises every measurement by write count
(`_normalised_linear_pulse_distance`, `services.py:10049-10055`, against
`_DEFAULT_REFRESH_COMMANDS_PER_LINEAR_PULSE = 10`, `:9941`).

The turn phase does not. `_turn_final_approach_pulse_ms` is still a pure
degrees-per-**second** model (`services.py:7446-7463`), and it feeds on nominal
ms. That is the same defect, on the other axis, already refuted by this
project's own hardware evidence.

### A5. Verdict on the three hypotheses

- **(a) refresh write timing changes actual motor-on time — SUPPORTED, and it is
  the largest single term.** It fully accounts for pulse 1 vs pulse 2. Caveat: no
  device-side motion watchdog value exists anywhere in this repo or in vendored
  pymammotion (searched `watchdog`, `DrvMotionCtrl`, `send_movement`), so whether
  the mower coasts between slow writes or duty-cycles is **not determined from
  code** — only that the *commanded* actuation varied.
- **(b) the 2 s VIO poll aliases the measurement — PLAUSIBLE, unresolved.** The
  measurement window is variable, not fixed: `before_heading` at
  `services.py:7847-7849`, `after_heading` from a poll loop that sleeps
  `max(refresh_wait_seconds, 0.5)` (`:8028`) and breaks on the **first** sample
  exceeding `_VIO_HEADING_FRESH_EPSILON_DEGREES = 0.1` (`:7158`, break at
  `:8049-8054`). `refresh_wait_seconds` is **not passed** at either vio call site
  (`:10647-10667`, `:11238-11253`), so it takes the default **2.0**
  (`:7617`) — against a feed the docstring says lags ~4 s (`:7641-7652`).
  A truncated read on pulse N is then credited to pulse N+1, because
  `before_heading(N+1)` is read immediately after `after_heading(N)` with no
  intervening motion. Sum check: 30.46+22.17+57.63 = **110.26°** for a turn the
  evidence file puts at **93.5°** — a 16.8° excess, which is inside the 18°
  `heading_tolerance_degrees`, so it is not by itself proof of anything.
- **(c) terrain / motor — untestable from the available data.** Nothing records
  slope, motor current or battery voltage per pulse. The 2026-08-04 daylight
  characterization already logged 21.20–49.57 °/s across 9 pulses (a 2.3× spread)
  on flat ground, so a 2.6× spread is **not anomalous against the existing
  record** — it is the same spread, now visible within one geometry.

**Bottom line: mostly MEASUREMENT plus mostly VARIABLE-ACTUATION, in that the
"identical commands" were not identical. Neither `°/s` nor the 2.6× figure is a
property of the mower — both are artifacts of dividing by commanded rather than
delivered actuation.** Pulse 3 remains genuinely unexplained.

### A6. Blocking limitation

**The raw per-command JSON for Gate 5 attempts 4 and 5 is not committed.**
`docs/evidence-gate5-PASSED-20260808.json` is a hand-written summary with no
`command_results` array. So the pairing of the four stop latencies
(1175/1819/402/628 ms) to specific pulses cannot be verified, and neither can
`heading_poll_seconds` or `heading_went_fresh` per pulse. Every conclusion above
is bounded by that.

The good news: **all the fields needed are already recorded per command** — no
code change is required to answer this, only retention of the full result JSON:

- `motion_refresh.elapsed_ms` and `.refresh_write_durations_ms` — `services.py:4942-4946`
- `heading_poll_seconds`, `heading_went_fresh`, `heading_poll_count` — `services.py:8057-8062`
- `final_approach` (the scaling decision and the rate it used) — `services.py:7942`

---

## Plumbing gaps found along the way (all latent today, none currently biting)

1. **`turn_pulse_duration_ms` is inert in `turn_mode: "vio"`.** It is not passed
   at the vio call site (`services.py:10647-10667`); the executor uses
   `_vio_turn_to_heading`'s hardcoded default `pulse_duration_ms: int = 1500`
   (`:7614`). It *is* used by the legacy path (`:10679`) and by the pre-dispatch
   junction feasibility preview (`:11631`). Harmless only because the card also
   sends 1500 — change the profile value and preview and execution silently diverge.
2. **`max_turn_commands` is inert in `turn_mode: "vio"`.** Only consumed at
   `:10676` (legacy). The binding budget is `vio_turn_max_commands` (`:10652`).
   The card sends both as 4, which hides this.
3. **`max_no_progress_pulses` is not forwarded to the turn.** Card sends 3; the
   turn uses its own default 2 (`:7621`).
4. **The slow-pulse branch is unreachable on this profile.**
   `slow_pulse_duration_ms=700` fires when `abs(error) <= slow_threshold_degrees`
   = 15 (`:7898-7908`), but the loop already returns `target_heading_reached` at
   `abs(error) <= heading_tolerance_degrees` = 18 (`:7887`). 18 > 15, so the
   branch is dead. All turn pulses are full-length unless final-approach scaling
   shortens them — consistent with all three observed pulses being 1500 ms.

## "turn_commands_sent: 4 of max 4 — FULLY CONSUMED" is probably a misreading

`result["turn_commands_sent"]` is **assigned** from the turn phase
(`services.py:10697`) and then **incremented** by mid-drive realignment commands
(`:11255`). Realignments carry their own fresh budgets — `min(2,
vio_turn_max_commands)` for the post-turn correction (`:10816`) and `min(6,
vio_turn_max_commands)` per mid-drive re-aim (`:11243`) — with
`vio_max_realignments` defaulting to **3** (`:10230`; not in the card profile, so
the default applies). Nothing enforces a cap on the summed counter.

So a segment can legitimately spend up to 4 + 2 + 3×4 = **18** turn commands, and
a reported 4 does not establish that any budget was exhausted. Whether attempt
5's four were all turn-phase pulses is **not determined** — the committed
evidence does not break the count down by phase, and `command_results` was not
retained. If they were all turn-phase, the fragility claim stands; if one was a
realignment, it does not.

---

## PROBLEM B — extending per-click reach

### B1. What actually caps reach today, ranked by which binds first

| cap | value | location | binds |
| --- | ----- | -------- | ----- |
| `max_linear_commands` schema `Range(min=1, max=3)` | **3** | `services.py:940-941` (vector), `:1063-1064` (multi) | **first, within a segment** |
| `REAL_CLICK_TO_GO_SEGMENT_LIMIT` | **2** | `manual_motion.py:24`, enforced `services.py:11490-11497` | **first, across segments** |
| card `MAX_REAL_SEGMENTS` | **2** | card js `:2`, sent at `:965-967` | mirrors the above |
| `max_real_segments` schema Range | `max=REAL_CLICK_TO_GO_SEGMENT_LIMIT` | `services.py:1054` | derived |
| `max_linear_pulse_ceiling` | **null** (loop-to-tolerance OFF) | card js `:39`, backend `services.py:10882` | inactive |
| `linear_distance_ceiling_factor` | 2.0, `Range(1.0, 10.0)` | `services.py:948`, `:1077` | only when ceiling set |
| card `MAX_WAYPOINTS` | 7 | card js `:1` | dry-run only |

Measured per-pulse travel from committed evidence: **0.394–0.581 m**
(`docs/NEXT-SESSION.md:238` — 0.475/0.581/0.394; `:336` — 0.568/0.466/0.492).
Three pulses × ~0.4–0.6 m = **1.2–1.7 m**, matching the observed 0.9–1.5 m.
Two segments therefore reach **~2–3 m total**. That is the whole envelope.

### B2. What `max_linear_pulse_ceiling` actually changes when set

Setting it flips `loop_to_tolerance = True` (`services.py:10882`) and changes
**four** things, not one:

1. The while-loop bound becomes the ceiling instead of `max_linear_commands`
   (`:10883-10887`, `:10916`).
2. A **cumulative distance ceiling activates** — `segment_length ×
   linear_distance_ceiling_factor`, aborting with
   `linear_distance_ceiling_reached` (`:10893-10897`, `:11306-11313`). This is
   the actual runaway guard, and it exists **only** in loop-to-tolerance mode.
3. No-progress handling changes: it aborts after `max_no_progress_pulses`
   consecutive failures rather than after the fixed budget (`:11296-11304`).
4. The terminal stop reason becomes `max_linear_pulse_ceiling_reached`
   (`:11315-11319`).

⚠️ **One asymmetry worth flagging:** the mid-drive re-aim guard is still
`command_index < max_linear_commands` (`services.py:11212`), **not**
`effective_linear_ceiling`. With a ceiling of e.g. 10 and `max_linear_commands`
3, cross-track correction silently switches off after pulse 3 while the mower
keeps driving for another 7 pulses. That is exactly the failure mode the re-aim
was added to prevent (a run that "drifted ~25 deg and sailed past the waypoint",
`:11195-11198`). **Any move to loop-to-tolerance should treat this as a
prerequisite fix, not a follow-up.**

Travel stays bounded by: waypoint completion (`:11138-11145`), the cumulative
distance ceiling, the pulse ceiling, no-progress abort, per-pulse safety gates
(`:10920-10947`), the reverse-recovery guard (`:11217-11229`), and the realign
budget (`:11231-11233`).

### B3. The smallest change that meaningfully extends reach

**Recommendation: raise `REAL_CLICK_TO_GO_SEGMENT_LIMIT` from 2.** It touches
**no `LUBA_ACCEPTANCE_PROFILE` key at all** — it is not one of the 20 frozen keys
(card js `:31-61`) — so it does not un-accept the profile. Reach scales linearly
with it (2 → 4 segments ≈ 2–3 m → 4–6 m) using **exactly the per-segment control
law that Gate 5 just validated four times**, with the waypoint tolerance
re-establishing ground truth at every junction. Each segment stays a bounded,
already-proven unit; only the count changes.

Cost: `manual_motion.py:24`, plus the card's `MAX_REAL_SEGMENTS` (card js `:2`)
and the status copy at `:89` and `:1598` — which does mean a `CARD_VERSION` bump
to both serving paths, but **not** the §4 profile re-pinning.

The alternatives, for contrast:

- **Set `max_linear_pulse_ceiling`** — this *is* a profile key (card js `:39`,
  currently `null`), so it un-accepts the profile and obligates the full §4 work
  *and* a fresh Gate 5. It also activates three behaviours at once (B2) and rides
  on the `:11212` re-aim bug. Highest capability, highest risk. Not first.
- **Raise `max_linear_commands` above 3** — profile key *and* a schema `Range`
  change in two places. Also the weakest lever: more pulses on a stale
  ~1031 ms feed accumulates cross-track error between corrections.
- **Chain multiple card runs manually** — zero code, zero profile impact,
  available today. Worth stating to the operator as the no-change baseline.

### B4. New acceptance evidence a reach change needs

For the segment-limit route, the per-segment control law is unchanged, so what is
genuinely new is **junction behaviour repeated more times** and **cumulative
error growth**. Minimum: one daylight card-driven run at the new limit over a
multi-segment path, recording per-segment landing error, `turn_commands_sent`
broken down by phase, `travel_ratio`, and zero `target_requires_reverse_recovery`
— i.e. the existing Gate 5 evidence shape, extended to N segments. Existing Gate 5
already covers the 2-segment case twice.

For the pulse-ceiling route: everything above, **plus** a fresh Gate 5 against the
changed profile, plus the §4 re-pinning listed in `docs/gate4-repass-20260805.md`.

---

## Explicit answer to the profile constraint

| recommendation | touches `LUBA_ACCEPTANCE_PROFILE`? |
| -------------- | ---------------------------------- |
| Stop dividing rotation by nominal ms; use `elapsed_ms` or non-zero write count | **No** — internal to `services.py` |
| Retain full `command_results` JSON from runs | **No** — an evidence-handling change only |
| Fix the `:11212` re-aim guard to use `effective_linear_ceiling` | **No** |
| Raise `REAL_CLICK_TO_GO_SEGMENT_LIMIT` | **No** (but does need a `CARD_VERSION` bump) |
| Set `max_linear_pulse_ceiling` to a number | **YES** — un-accepts, needs §4 + fresh Gate 5 |
| Raise `max_linear_commands` above 3 | **YES** — un-accepts, plus a schema Range change |
| Change `turn_pulse_duration_ms` / `max_turn_commands` | **YES** — and note they are currently *inert* on the vio path |

## Repo facts worth recording

- The card exists at **one** path in the repo,
  `custom_components/mammotion/www/mammotion-custom-path-card.js`
  (md5 `8ec0fb0189f01ea339237f3ae1ef988d`). `www/` does **not** exist at the repo
  root — the "two serving paths" in CLAUDE.md are on the HA host, not here.
- `CARD_VERSION = "0.6.4-beta30"` (card js `:9`).
- Frontend profile pin: `tests/frontend/mammotion-custom-path-card.test.mjs:146`
  asserts `max_linear_commands === 3`.

---

## Verification status at stop (session limit, 2026-08-08)

Workflow `wf_cec0c5f1-051` was stopped early. Completed: **6/6 finders**,
**3/6 adversarial verifiers**. The completeness critic did **not** run.

Across the 3 verifier passes that finished: **33 CONFIRMED, 4 REFUTED.** The four
refutations were all against *finder* claims, not against this file — three were
off-by-a-few line-number citations, one was an over-claim about a code comment.

One refutation is itself wrong and should be ignored: a verifier reported it
could not find `REAL_CLICK_TO_GO_SEGMENT_LIMIT`. It exists — defined at
`custom_components/mammotion/manual_motion.py:24`, imported at
`custom_components/mammotion/services.py:42`, used in the schema Range at
`services.py:1054`, enforced at `services.py:11490-11497`, and surfaced as
`real_click_to_go_segment_limit` at `manual_motion.py:198`. Read directly, not
inferred.

**Still unverified (no verifier pass completed on these):** the
A-refresh-timing arithmetic and the A-measurement-window analysis. Both are
reproduced independently in sections A1–A4 above from first-hand reads, but
neither has had an adversary attack it.

### Correction — exactly which verifiers completed

All six verifiers were spawned; three returned verdicts before the stop. Named:

| topic | verifier | outcome |
| ----- | -------- | ------- |
| A-displacement-discriminator | `a7eaaad9cd61637bd` | **completed** |
| B-ceiling-semantics | `a869db41e1802c513` | **completed** |
| B-acceptance-obligations | `a9c53b8093c91acef` | **completed** |
| A-refresh-timing | `a40d5150e5453570b` | killed mid-run, no verdict |
| A-measurement-window | `afc93945b93b58cfc` | killed mid-run, no verdict |
| **B-reach-arithmetic** | `a4e72c62ee5794fa2` | killed mid-run, no verdict |

So the unverified set is **three**, not two: sections A2/A3 (refresh timing),
A5(b) (measurement window), **and B1/B3 (the reach arithmetic and the
smallest-change recommendation)**. B2 (ceiling semantics) and the profile-
obligation table in B4 *were* verified.

The three killed agents emitted no `StructuredOutput`, so nothing is salvageable
from them; their partial transcripts are `agent-a40d5150e5453570b.jsonl`,
`agent-a4e72c62ee5794fa2.jsonl` and `agent-afc93945b93b58cfc.jsonl` in the
workflow dir. To finish, re-run:

    Workflow({scriptPath: ".../workflows/scripts/mammotion-turn-variance-and-reach-wf_cec0c5f1-051.js",
              resumeFromRunId: "wf_cec0c5f1-051"})

The six finders and three verifiers are cached, so only the three missing
verifiers plus the completeness critic would actually run.

---

## FINAL STATUS (workflow run 2, 2026-08-08) — 11/13 agents done

Resume completed. **6/6 finders, 5/6 verifiers.** Two agents died on an
Anthropic **monthly spend limit**, not on an error:

- `verify:B-reach-arithmetic` — FAILED (spend limit)
- `completeness-critic` — FAILED (spend limit)

Tally across the 5 completed verifiers: **81 CONFIRMED, 15 REFUTED,
1 UNVERIFIABLE.**

### Corrections that land on THIS file

1. **§A2 spread framing.** The verifier is right that the *overall* spread
   barely moves: nominal max/min = 38.42/14.78 = **2.60x**; elapsed-denominated
   = 32.74/14.49 = **2.26x**. The collapse to ~3% is only between **pulses 1 and
   2** (which is what the table's last column says, and is the substantive
   point) — it is NOT a claim that the whole 2.6x disappears. Pulse 3 dominates
   the spread under every unit.
2. **§A0 table CONFIRMED numerically.** Incremental m/deg 0.001658 / 0.000997 /
   0.000687, monotonically decreasing. Independently recomputed.
3. **`_VIO_TURN_CONSERVATIVE_TRANSLATION_M_PER_DEGREE = 0.0026`** sits above
   every incremental m/deg observed in attempt 5 — i.e. the translation guard is
   behaving fail-closed, as designed. No action.
4. Verifiers again reported `REAL_CLICK_TO_GO_SEGMENT_LIMIT` "could not be
   verified". **Ignore this twice over** — it is at `manual_motion.py:24`,
   imported `services.py:42`, Range at `:1054`, enforced `:11490-11497`.

### What is STILL unverified

- **§B1/§B3 — the reach arithmetic and the `REAL_CLICK_TO_GO_SEGMENT_LIMIT`
  recommendation.** Its verifier never ran. This is the main open item.
- **No completeness critic ran**, so nothing has asked "what did all six angles
  collectively miss?"

Both need budget before they can run. Everything else is verified.

---

## §B1 CORRECTED — the reach numbers in this file were wrong

`verify:B-reach-arithmetic` completed on the second resume (16 CONFIRMED,
1 REFUTED) and it corrected the reach arithmetic. **Use these numbers, not the
ones in §B1 above.**

**What §B1 got wrong.** It cited 0.475/0.581/0.394 and 0.568/0.466/0.492 m from
`docs/NEXT-SESSION.md:238` and `:336` and concluded ~0.4-0.6 m per pulse.
Those runs were **not** on the accepted profile. The 1.06 m
`final_approach_metres_per_pulse` fallback was measured at
**3500 ms** pulses (`services.py:9932-9933`), while the accepted profile runs
**1300 ms** (card js `:55`).

**Corrected, re-derived from the raw JSONs:**

| quantity | value | source |
| -------- | ----- | ------ |
| deadline-limited pulse @ 1300 ms / refresh 200 | **0.3496 - 0.4192 m** | `evidence-gate4-beta20-day2i/day2j-real-result-20260805.json` (4 four-write pulses) |
| bounded final-approach pulse | 0.045 - 0.22 m | same |
| all committed per-pulse values (n=22) | 0.0453 - 0.7835 m | the largest are 3500 ms-era only |
| committed 3-pulse **segment sums** | **0.522 - 0.975 m** | `docs/gate4-repass-20260805.md:37` |
| only committed 3-command 1.0 m leg | 0.9174 m net | `evidence-slow-tier-validation-20260808.json` seg 1 |

So per-segment reach on the accepted profile is **~0.5 - 1.0 m**, not the
1.2-1.7 m §B1 claimed, and two segments give **~1.0 - 2.0 m total**, not 2-3 m.

**This strengthens the recommendation rather than weakening it.** Reach is
tighter than stated, so the segment-count lever matters more, and its cost is
unchanged: `REAL_CLICK_TO_GO_SEGMENT_LIMIT` is confirmed to be a Python constant
wired into **both** the multi-segment schema `vol.Range` upper bound **and** a
second runtime enforcement check, so raising it raises both at once
(verdict 10, CONFIRMED). It remains a non-profile key.

Two traps the verifier flagged in the source data, worth keeping:

- `path_m` in `evidence-gate5-PASSED-20260808.json` is the **planned** leg
  length, not travel — it recomputes exactly from the file's own `points`. The
  same file's `independent_trace` gives real travel: 1.8458 m against 1.4809 m
  planned.
- Only `segment_1` of each Gate 5 attempt carries a `commands` block. Segment 2
  has no command count at all, so attributing 3 pulses to it is unverifiable.

### A candidate this analysis omitted, and why to reject it

**Raising `linear_pulse_duration_ms` 1300 -> 3500** is a real reach lever the
option list above missed (profile key, schema already allows 50-4000 at
`services.py:981-983` and `:1110-1112`). **Reject it.** It re-introduces exactly
the defect the 2026-08-05 day2j fix removed — `docs/gate4-repass-20260805.md:47-52`
names `linear_pulse_duration_ms: 3500 -> 1300` as "the dominant cause" of the
fix, because a 3500 ms pulse commands more travel than the leg needs, the
executor must interrupt it mid-flight, and the interrupted stop lands late
(0.15-0.26 m overshoot across day2d/e/f/h). The raw records agree: day2d/e/f each
fired one 3500 ms pulse travelling 0.7835 / 0.6614 / 0.7467 m.

---

## COMPLETENESS CRITIC (full text: `CRITIQUE-completeness.md`) — 13/13 agents, 0 errors

Three corrections that land on this file:

**1. The four stop latencies do not belong to the turn pulses at all.**
`_vio_turn_to_heading` calls `_stop_manual_motion_confirmed`
(`services.py:7982-7984`), which returns a literal `{"movement_ok": True}` with
**no duration field** (`:3321-3333`). Only `_manual_velocity_stop_attempt` times
itself (`:3286`) — and that is used by the calibration drive (`:10141`) and the
linear phase (`:11047`). Attempt 5 seg 1 was `calibration: 1, linear: 3` =
**exactly the four values 1175/1819/402/628**. So any "elapsed + stop latency"
rate table is not merely unverifiable, it is **contradicted**. Discard it.

**2. New, unflagged: turn stops are `Priority.NORMAL`; linear stops are
`Priority.EMERGENCY`** (`services.py:5719-5720`, `:3307`). The code's own comment
records the consequence: *"Live 2026-08-02 a normal-priority stop took 1392.7 ms
to confirm while the mower continued past its target"* (`:3301-3306`). So the
project already has committed live evidence that **the mower keeps moving during
a normal-priority stop**, and the turn phase is the only remaining path using it.
That is an unmeasured contributor to per-pulse rotation, and no comment anywhere
justifies the asymmetry.

**3. §A2 independently reproduced, and sharpened.** deg/write 7.615 / 7.390 /
19.210 (pulses 1–2 agree to 3%); **ms-of-window per write 510.8 / 510.0 / 586.7 —
near-constant**, matching the 540 ms median write latency. The window was
entirely write-latency-bound. Reframing: *not* "why is the rate noisy" but **"why
did pulse 3 produce 2.6x the rotation per delivered write."**

**Also: `_vio_turn_probe` already documents the mechanism.** `services.py:7059-7064`
records live that on a short pulse the ONLY during-command sample is bit-identical
to baseline and every real change lands in `post_stop` — a taped 13.18° pivot came
back as `vision_heading_static_during_command`. The turn executor's poll breaks on
the **first** sample past a 0.1° epsilon with **no settle requirement**, unlike
`_settle_linear_position_feed` which needs two consecutive agreeing samples
(`:8318-8321`). Measurement contamination is documented behaviour of the
neighbouring probe, not a hypothesis.

### ⚠️ Do not trust the 110.26° sum

If the three rotations were same-signed against a 93.5° initial error, the loop
would have returned `target_heading_reached` after pulse 3 (`:8107-8109`, tolerance
18). It did not. **At least one premise is false** — signs differ, or the VIO-frame
target error ≠ 93.5° (it is derived at `:10643-10646`, not from raw path bearing),
or pulse 4 was a reversal. `heading_error_after` is recorded per command (`:8073`)
and settles it in one line.

### Next step is FREE and read-only

**Recover the attempt-5 response JSON before authorizing any run** — HA's
service-call trace or the card's response pane. `heading_poll_count` +
`heading_poll_seconds` for pulses 1–3 already partially discriminate (a poll
needing 2+ iterations was read at the instant registration began = truncated),
and `heading_error_after` on pulse 4 resolves the contradiction above. This may
make a run unnecessary.

If a run is needed: 3–5 **isolated single pulses**, each bracketed by a
`vio_turn_probe` `dry_run: true` settled reading, using `max_commands: 1`.
Single-pulse isolation removes cross-pulse attribution — the confound no
multi-pulse run can resolve. Full parameters in `CRITIQUE-completeness.md` §3.
**No code change required**; every needed field is already recorded.

### Problem B — confirmed, with a real caveat

Raising the real-segment limit touches no profile key and needs **zero test
churn**: the frontend test *imports* `MAX_REAL_SEGMENTS` rather than hard-coding
it (`tests/frontend/mammotion-custom-path-card.test.mjs:40`, `:128`), the README
pin test iterates only `PROFILE_KEYS` (`:233`), and
`REAL_CLICK_TO_GO_SEGMENT_LIMIT` appears in no test file. Sites:
`manual_motion.py:24`, card js `:2`, `README.md:107`. Only obligation is the
unconditional `CARD_VERSION` bump + both serving paths.

**Caveat:** segment 3+ is measured nowhere. The VIO forward-heading offset is
refreshed only from linear travel and never across a turn, and attempt 5's
segment 2 already gave the worst landing of the four (0.1449 m against 0.15).
That is the wrong trend to extrapolate.
