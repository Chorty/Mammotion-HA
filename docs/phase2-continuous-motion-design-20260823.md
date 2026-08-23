# Phase 2 continuous motion — design decisions, 2026-08-23

**This is a design document. It authorizes no implementation, no mower run, and
no `LUBA_ACCEPTANCE_PROFILE` change.** Phase 1b's `go`
(`docs/phase1b-go-20260823.md`) is scoped to telemetry feasibility only, per the
analyzer's own output. This records the four architecture decisions the
operator made to unblock design, and the concrete gaps between them and the
Phase 0 skeleton that already exists.

## The four decisions

| Question | Decision | Why |
| --- | --- | --- |
| First target | **Straight-line segments only, no turns** | Turning-while-continuous is untested — only isolated fixed-angular arcs exist. Stacking continuous straight-line and continuous turning at once makes a v1 failure undiagnosable. |
| Architecture | **Extend the bounded-window pattern** | Reuses everything proven this week: the refresh loop, the beta72 distance guard, corridor containment. A persistent velocity loop would need all of that re-derived. |
| Correction cadence | **Every ~1 Hz position arrival** | Matches the measured feedback rate exactly; the prediction model is already validated at this cadence (median 1.75 cm error, out of sample). |
| BLE-stall response | **Stop safely, same as the travel guard** | Matches the pattern already tested on hardware today (`docs/evidence-travel-guard-fired-20260822T234500Z.json`, `docs/evidence-arc120-outofsample-20260823T001500Z.json`). Never drive on a guess during the one condition proven to cause the worst errors. |

## What already exists — build on this, do not rebuild it

`custom_components/mammotion/continuous_controller.py` is Phase 0: a pure,
dispatch-free calculator that already matches three of the four decisions.
`ContinuousRoute` is straight-only (`start`, `target`, no waypoints). Its
`_predict_position` / `ContinuousDecision` shape is exactly the
observe-predict-correct-repeat loop the correction-cadence decision calls for.
Its `ContinuousObservation` already carries `refresh_healthy` and
`ble_live` as fail-closed inputs, so a stop-safe stall response is already the
module's default posture, not a new concept.

## Four concrete gaps to close, none of them decisions — just reconciliation

1. **`nominal_speed_mps = 0.28` does not match the measured constant.**
   `docs/frozen-prediction-constants-20260822.json` gives `k_lin`, which at
   `linear_speed 400` implies **0.2482 m/s**, from 16 steady-state steps across
   three runs. The Phase 0 stub's 0.28 predates that measurement. Replace it
   with the frozen value, or explicitly justify keeping a different number.

2. **`max_refresh_age_s = 1.20 s` does not match the registered stall rule.**
   `docs/phase1b-arc-protocol-20260823.md` defines a cadence stall as a gap
   between successful refresh completions exceeding **`3R` = 600 ms**. The
   Phase 0 stub's threshold is exactly double that. These may legitimately be
   two different things — telemetry-staleness tolerance for *predicting a
   stale fix* versus a *stall detector* that stops the run — but that
   distinction is not written down anywhere and should be, before either
   number is treated as settled.

3. **`angular_speed_per_heading_degree = 12.0` is a steering GAIN, not a yaw-rate
   model — confirm this reading before touching it.** It converts a heading
   error into a commanded angular speed for the *next* command; it is not the
   `w = k_ang * angular_speed` relationship that was refuted as non-proportional
   this week (`docs/prediction-model-holds-out-of-sample-20260823.md`). The two
   are easy to conflate because both involve "angular speed" and "degrees."
   They answer different questions — "how hard do I turn to fix this error" versus
   "how fast will the mower actually rotate at a given command" — and only the
   second was measured and found non-proportional. Worth stating explicitly in
   the module's own docstring so a future reader does not merge them.

4. **No mid-run corridor or keep-out check was found in the portion of the
   module read for this design pass (lines 1-200 of 357).** The existing
   pulsed executor has segment-level keep-out containment
   (`_keep_out_leg_violations`, `_validate_custom_path`). Before implementation
   starts, confirm whether Phase 0 already calls into an equivalent check or
   whether continuous motion needs its own — do not assume either answer
   without reading the remaining 157 lines first.

## Non-goals for v1 — explicit, so scope does not creep

- **No turns.** A junction turn stays on the existing pulsed executor even in a
  path that also contains continuous straight legs, until turning-while-continuous
  has its own measurement.
- **No commanded speed above `linear 400`.** The vendor drives faster
  (~0.55 m/s) and that lever is real, but it increases blind distance per
  ~1 Hz correction and was explicitly flagged as unmeasured, not proposed, in
  `docs/what-continuous-motion-is-worth-20260822.md`.
- **No replacement of the pulsed executor.** It stays the default path;
  continuous motion is additive, gated behind its own explicit selection.

## What still bounds this design regardless of the decisions above

- The **~1 Hz bundle** is a hardware ceiling, not a tuning parameter
  (`docs/the-1hz-bundle-is-the-ceiling-20260822.md`).
- The **alpha = -0.149 +- 0.043** pairing residual is unexplained and
  unmodelled, worth ~7 mm per step (`docs/codex-adjudication-20260823.md`). It
  does not block this design; it is not resolved by it either.
- The **prediction-error criterion is still not implemented or adopted** as a
  Phase 1 gate. Phase 2 can use the prediction model internally without that.

## Before any physical run: a Gate 5-style validation, not yet defined

Every prior control-law change in this project passed a dedicated Gate 5
validation before being trusted (`docs/gate5-repass-PASSED-20260812.md` and
earlier). Continuous motion is a new control law and needs the same treatment
— its own pass/fail criteria, written down before the first supervised run,
not derived from whatever the first run happens to produce. That gate is not
designed yet and is explicitly out of scope for this document.

## Recommended next step

Implement the four gaps above **offline first**, in the same no-dispatch
discipline as Phase 0: extend `continuous_controller.py`, then validate it by
**replaying it against the captures already banked** —
`docs/evidence-8s-continuous-window-20260822T233000Z.json` and
`docs/evidence-travel-guard-fired-20260822T234500Z.json` — before proposing any
new physical run. Both already contain real telemetry at the cadence and
duration Phase 2 would consume; replay is free.
