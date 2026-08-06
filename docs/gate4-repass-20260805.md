# Gate 4 re-pass, 2026-08-05 — result, profile delta, and what adopting it requires

Written for a handoff. Gate 4 had already passed as backend acceptance on
2026-07-31 (`docs/p0-beta-release.md` line 97); the 2026-08-03 beta19 retry
failed and the gate needed re-passing. This records the re-pass, the parameters
it required, and the decisions that are **not** made yet.

No motion is authorized by this document. Experimental motion was disabled at
the end of the session and the mower is stationary and in-zone.

## 1. Result

`docs/evidence-gate4-beta20-day2j-real-result-20260805.json` records:

- top-level `stop_reason: "target_reached"`, `failed_segment_index: null`
- `segments_planned: 2`, `real_segments_executed: 2`, `errors`/`warnings`/`blockers` all empty
- **segment 1** `passed: true`, `target_reached`, final miss **0.04031 m**
- **segment 2** `passed: true`, `target_reached`, final miss **0.03304 m**

against `waypoint_tolerance: 0.08`. The request file is byte-identical to the
request block embedded in the result, and carries `dry_run: false`, so the
recorded payload is the payload that ran.

Both misses sit at roughly half the tolerance. That matters, because the
position feed's pulsed-measurement noise floor is 2–6 cm: a result *at* the
0.08 m boundary could not be distinguished from a pass, whereas these can.

### The path was not clean

This is the part that must not be smoothed over. The run passed by
**overshooting and recovering**, not by tracking:

| | segment 1 | segment 2 |
| --- | --- | --- |
| commands sent | 11 (1 calibration + 7 turn + 3 linear) | 6 (3 turn + 3 linear) |
| linear pulse distances (m) | 0.4192, 0.2180, 0.3376 | 0.3496, 0.0934, 0.0787 |
| sum of linear travel | **0.975 m on a 0.520 m leg** | 0.522 m on a 0.520 m leg |
| mid-run realignments | +33.513°, then **+103.427°** | −16.032°, −19.818° |
| turn phase final error | 2.738° | −13.534° |

Telemetry capture shows **2.2773 m of actual travel for a 1.0400 m planned
path**. Segment 1 drove past waypoint 1, discovered a 103.427° aim error,
turned back, and landed on the last of its three permitted linear pulses.

## 2. Why it passed — three parameters, each with a measured reason

**`linear_pulse_duration_ms: 3500 → 1300`.** The dominant cause. At the
executor's effective rate a 3500 ms pulse commands far more travel than a 0.52 m
leg needs, so the executor must interrupt it mid-flight, and the interrupted
stop lands late — measured overshoot 0.15–0.26 m across day2d/e/f/h. A pulse
sized to the leg **ends naturally** and stops clean. This is a stop-latency
defect, not a distance-model defect.

**`max_linear_commands: 1 → 3`.** With 1, day2d stopped on
`max_linear_commands_reached` while still 0.102 m out — it was forbidden from
correcting. Capping at 2 (day2i) produced the opposite failure: segment 2
undershot and ran out. 3 permits the small final-approach hops
(0.0787–0.0934 m) that actually close the gap.

**`max_turn_translation_distance: 0.25 → 0.30`.** Originally raised to unblock
the initial ~104° junction turn, which the pre-check refused at 0.25
(`turn_budget_infeasible`, estimated 0.271 m against a 0.25 m cap). Its more
important role emerged later: segment 1's **103.427° recovery turn** estimates
at `103.427 × 0.0026 = 0.269 m`, which is refused at 0.25 and permitted at 0.30.
**The cap increase is what makes overshoot recoverable.** day2e and day2h both
died on `vio_realign_incomplete` for exactly this reason.

Measured actual turn drift never exceeded 0.178 m tonight, against a 0.271 m
estimate for the same turn — the estimator is roughly 50% pessimistic.

## 3. Profile delta

`LUBA_ACCEPTANCE_PROFILE`, `custom_components/mammotion/www/mammotion-custom-path-card.js:26`.
Only one copy exists in the repo; the second serving path is on the HA host.

| field | frozen profile | day2j | status |
| --- | --- | --- | --- |
| `linear_pulse_duration_ms` | 3500 | **1300** | changed |
| `max_linear_commands` | 1 | **3** | changed |
| `max_turn_translation_distance` | *(absent — backend default 0.25)* | **0.30** | **must be added** |
| `prefer_ble` | true | true | same |
| `turn_mode` | vio | vio | same |
| `max_turn_commands` | 4 | 4 | same |
| `vio_turn_max_commands` | 4 | 4 | same |
| `max_no_progress_pulses` | 3 | 3 | same |
| `heading_tolerance_degrees` | 18 | 18 | same |
| `waypoint_tolerance` | 0.08 | 0.08 | same |
| `min_progress_distance` | 0.0025 | 0.0025 | same |
| `calibrated_forward_heading_offset_degrees` | 102.4 | 102.4 | same |
| `turn_pulse_duration_ms` | 1500 | 1500 | same |
| `motion_refresh_interval_ms` | 200 | 200 | same |
| `final_approach_metres_per_pulse` | 1.06 | 1.06 | same |
| `turn_degrees_per_second` | 37 | 37 | same |
| `ble_auto_recover` | false | false | same |
| `sample_delays` | [0, 3] | [0, 3] | same |

The third row is the subtle one: the card **never sends**
`max_turn_translation_distance`, so it inherits the backend default of 0.25 —
the value that refused the recovery turn. A card run today would still fail the
way day2e and day2h did.

## 4. What must change if this profile is adopted

Not done. Listed so the decision is costed.

- `custom_components/mammotion/www/mammotion-custom-path-card.js:26` — the three fields above.
- `CARD_VERSION` bump, and deploy to **both** serving paths (`/mammotion/` and `/hacsfiles/`), plus a Lovelace resource-key bump. Confirm the banner in the browser console, not the upload.
- The card's execution-profile row text, currently required by
  `docs/p0-beta-release.md:111` to read `LUBA acceptance profile (Gates 1-4, 2026-07-31)`.
  That date is now wrong if the profile moves.
- Pinning tests that assert the old values:
  - `tests/frontend/mammotion-custom-path-card.test.mjs:146` (`max_linear_commands == 1`)
  - `tests/components/mammotion/test_motion_scripts.py:275` (`max_linear_commands == 1`)
  - `tests/components/mammotion/test_map_task_visibility.py:1469, 1533, 4174, 4217, 4254, 4292, 4600, 4739, 4747`
- Backend schema defaults in `custom_components/mammotion/services.py` (multiple service schemas: `max_linear_commands` at lines 894 and 1018, `max_turn_translation_distance` at 927 and 1051) — decide whether defaults move or whether the card carries the values explicitly.

## 5. Open questions and caveats

**This is one run.** The handoff already required reproduction on a second
daylight geometry; that requirement is unaffected and unmet.

**The profile-identity invariant is at risk.** `docs/p0-beta-release.md:98-102`
states that passing Gates 1-4 while the card emitted a *different* profile is
precisely the gap `LUBA_ACCEPTANCE_PROFILE` was created to close. Re-passing
Gate 4 on parameters the card does not emit **re-opens that gap**. Either the
card profile moves to match this run, or this run does not underwrite a Gate 5
attempt. This is a decision, not a formality.

**Pulse-to-pulse variance is large.** Six identical 1300 ms commands produced
0.0787, 0.0934, 0.2180, 0.3376, 0.3496 and 0.4192 m. Some of that spread is the
final-approach shortening (below), but not all of it. Wheel slip, dirt, grass
drag and battery state are uncontrolled.

**`final_approach_metres_per_pulse` behaviour is not understood.** Early pulses
appear to run the full commanded duration regardless of remaining distance
(day2h pulse 2 travelled 0.334 m with ~0.17 m remaining), yet late pulses are
small and well-scaled (0.0787, 0.0934). Both observations are in the evidence
and they are not reconciled. Do not build on either until this is explained.

**Tolerance versus noise floor.** `waypoint_tolerance: 0.08` sits at or below
the 2–6 cm pulsed-measurement noise floor. day2h missed by 8 mm — a margin far
smaller than the instrument error, and not meaningfully distinguishable from a
pass. A tolerance derived from the measured noise floor, or a pass criterion
requiring N consecutive runs, would be more defensible than the current single
threshold. Not changed here: moving a gate's own criterion to make a run pass
would be goalpost-shifting, and it is the project's call.

**Two models asserted during this session were refuted by direct measurement**
(see §6). Both had been fitted to a handful of executor-level datapoints rather
than measured. Treat any executor-derived kinematic claim in this repo as
provisional until a sweep confirms it.

**One run's evidence was destroyed by a full disk** (day2b, 2026-08-05 18:48).
The request and capture survive; the durable result does not. The evidence
runner cannot protect against ENOSPC.

## 6. Characterization data

`scripts/linear_duration_sweep.py`, RTK-measured, alternating forward/backward
so net displacement stays near zero. Calls `manual_velocity_pulse_test` — a
**different service path** from the segment executor, whose effective rate
differed (~0.20–0.32 m/s). Do not assume these transfer directly.

**`motion_refresh_interval_ms: 200`** — `docs/evidence-linear-sweep-refresh200-20260805.json`

| duration (ms) | moved (m) | rate (m/s) |
| --- | --- | --- |
| 1600 | 0.4690 | 0.293 |
| 1000 | 0.3346 | 0.335 |
| 700 | 0.2276 | 0.325 |
| 500 | 0.1516 | 0.303 |
| 400 | 0.1061 | 0.265 |
| 300 | 0.1325 | 0.442 |

Distance is proportional to commanded duration at roughly **0.30 m/s**. The
300 ms point is the outlier and sits within the noise floor of the measurement.
There is **no fixed floor** — refuting the "0.386 m + 0.160 × seconds" affine
model asserted earlier in the session from two executor datapoints.

**`motion_refresh_interval_ms: 0`** — `docs/evidence-linear-sweep-singleshot-20260805.json`

| duration (ms) | moved (m) |
| --- | --- |
| 1600 | 0.0984 |
| 1000 | 0.1199 |
| 700 | 0.1006 |
| 500 | 0.1119 |
| 400 | 0.1029 |
| 300 | 0.1141 |

A **fixed ~0.11 m step independent of commanded duration** — a 5× range in
commanded time produces the same travel. Single-shot has no fine distance
control. This refutes the plan to use single-shot for the linear phase, and
contradicts the "1.0 cm along-track precision" figure of 2026-07-27 as a
description of *this* behaviour.

Corollary: **refresh 200 is the controllable regime for both phases.** The
earlier conclusion that turn and linear need opposite refresh settings, and
therefore a code change to decouple them, is withdrawn.

Single-shot turning was also measured, in day2g: **2.4°/command** over 17
commands, with 0.024 m of drift per command, aborting on the runtime
displacement cap. The ~8–9°/command figure in the project notes does not
describe it.

## 7. Suggested next steps

1. Decide the profile question in §5 before any Gate 5 attempt.
2. Reproduce Gate 4 on a second daylight geometry — still required, still unmet.
3. Explain the `final_approach_metres_per_pulse` inconsistency in §5.
4. Consider whether the stop-latency overshoot deserves a code fix (lead the
   stop by `speed × latency`) rather than being managed by pulse sizing, which
   only works when the leg length is known in advance.
5. `scripts/linear_duration_sweep.py` requires `custom_components.mammotion` at
   DEBUG, or its `ble_alive()` guard aborts on a false negative — it greps HA
   logs for `BLETransport send`, which are not emitted at INFO. It also requires
   `scripts/map.json`, which is not in the repo; regenerate it from the
   `get_map_data` service.
