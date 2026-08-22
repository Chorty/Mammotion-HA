# Continuous-motion feasibility plan — zero-motion phase

Status: **offline controller, Phase 1 probe instrumentation, and offline capture
analyzer implemented; no continuous controller executor or new mower dispatch
path exists. No Phase 1 physical capture has been run.**

The pulsed controller's 9 m Route B success took 162.7 s, or 0.055 m/s. The
vendor moves continuously at roughly 0.55 m/s on the same ~1 Hz position feed.
The question is therefore not whether continuous commands move the mower—they
do—but whether feedback arrives soon enough to steer and stop a continuously
refreshed command safely.

## What is already proven

- `send_movement(linear, angular)` held at the app's 200 ms refresh cadence
  drives continuous straight and arc motion.
- At linear 400, angular 180 / 300 / 500 produced measured turn rates of
  11.09 / 19.31 / 32.21 deg/s with r² = 0.9997. Angular 180 is the initial
  steering ceiling because it is the shallowest measured arc: radius 1.512 m.
- `toward` follows course during continuous translation, and the position feed
  updates at approximately 1 Hz.
- Projecting the next position from the last fix and commanded velocity measured
  0.029 m median / 0.097 m p90 error when refresh cadence held.

What is **not** proven is the critical link: the timestamps and latency of
position/`toward` changes *inside* a continuous command window, followed by an
actual steering change while motion remains open.

## Phase 0 — implemented without Home Assistant or motion

`custom_components/mammotion/continuous_controller.py` is a pure calculation
module. It accepts a prevalidated straight route and one observation, then
returns either a bounded `(linear, angular)` request or a literal zero-speed
stop decision. It imports no Home Assistant, coordinator, BLE, or service code.

The initial steering law uses a 0.8 m lookahead point on the route centerline.
It predicts a recent stale fix along the reported map-course heading, aims at
the lookahead point, applies a signed heading correction, and clamps angular
speed to ±180. These values are **provisional replay settings**, not accepted
profile keys and not authorized hardware settings.

It fails closed on every input an eventual executor must own:

- operator cancellation or missing stop primitive;
- route not prevalidated as contained;
- BLE loss, refresh failure, or excessive refresh age;
- stale/non-finite/invalid telemetry;
- invalid position, area, RTK, blades, or work mode;
- 4 s wall-clock limit, 1.5 m distance limit, or 0.30 m cross-track limit;
- target reached or passed.

`scripts/replay_continuous_controller.py` exercises the module from JSON or a
built-in shallow-error sequence. Its output states
`dispatch_capable: false` and `commands_sent: 0`. It loads only the pure module
file, so it runs without importing the Home Assistant integration package.

## Phase 1 — bounded in-window telemetry measurement (instrumentation ready)

Do this before writing a closed-loop executor. The current motion probe records
the settled result after its window; that proved arc geometry but not feedback
latency during the arc. `motion_capture.py` samples through Home Assistant REST
and is useful corroboration, but its multiple sequential HTTP reads are not a
precise pulse-timescale clock.

The existing bounded raw motion probe now has an opt-in
`in_window_sample_interval_ms` instrumentation field. Zero is the default and
preserves the old probe behavior. A positive value requires
`motion_refresh_interval_ms > 0`; otherwise a real probe fails closed before
stream startup or motion.

The implementation:

1. Starts the existing bounded report-stream helpers before command dispatch;
   a startup failure sends no movement command.
2. Samples the coordinator cache every 100 ms on a concurrent task without
   sending extra BLE report requests.
3. Records monotonic elapsed time, UTC capture time, x/y, `toward`, VIO heading
   and state, `DeviceHandle.last_report_at`, active command, and every
   refresh-write completion.
4. Preserve the existing 4,000 ms hard window, 200 ms refresh, mandatory stop,
   cancellation stop, exclusive motion owner, and final forced readback.
5. Dry run shows the complete plan and sends nothing. At 4,000 ms / 100 ms it
   declares a maximum of 41 cache samples, the stream-start plan, and zero extra
   BLE report requests during the window.

The response also summarizes fresh report stamps, fresh x/y arrivals, gaps
including the start/end boundaries, and `toward` changes observed before stop.
These are measurements for the criteria below, not a pass verdict. Focused
tests cover schema bounds, disabled-by-default behavior, dry-run inertness,
refresh-required refusal, stream-start failure before motion, and summary math.

After both responses are banked, run the non-dispatching analyzer documented in
`docs/phase1-capture-analyzer.md`. It recomputes timing, compass-mirror,
pre-stop-turn, and containment criteria from the raw sample arrays, fails
closed on missing evidence, records input SHA-256 digests, and returns a scoped
`go` or `no_go`. Its `go` never authorizes Phase 2 or another physical run.

Two separately authorized physical windows are then sufficient:

| control | linear | angular | window | purpose |
|---|---:|---:|---:|---|
| straight | 400 | 0 | 4.0 s | position-arrival latency and baseline course |
| shallow arc | 400 | 180 | 4.0 s | prove `toward` changes before stop |

Both routes require a fresh full-segment containment scan with at least 1.2 m
area and 1.5 m keep-out margin, frozen start/endpoint, daylight, blades off,
clear area, operator present, accessible emergency stop, and explicit
authorization for each window. The experimental gate is armed inside the same
`try` whose `finally` disarms and verifies it.

### Phase 1 go/no-go criteria

Proceed only if both captures show:

- confirmed stop and no refresh/queue error;
- at least three fresh position arrivals inside each 4 s window;
- no gap between fresh position arrivals over 2.0 s;
- shallow-arc `toward` changes before the stop, rather than appearing only in
  the forced settled readback;
- moving-step `bearing + toward` remains within 10° of the measured 90° compass
  mirror;
- observed travel remains inside the prevalidated corridor.

Failure is evidence that the ~1 Hz feed cannot safely close this loop as
currently exposed. Do not compensate by widening the stale threshold.

## Phase 2 — first closed-loop steering window

Only after Phase 1 passes, design a new experimental executor around the pure
controller. Do not retrofit variable commands into `_motion_refresh_window`,
whose contract intentionally resends an identical command. The executor needs
one serialized writer that owns command refresh, feedback decisions, and stop.

The first run remains capped at 4 s / 1.5 m, linear 400, angular ±180, on a
straight route with a deliberate 5–10° opening error. It changes steering only
on a fresh observation and holds the last bounded command between observations.
Every stop decision must deliver zero speeds immediately and leave the device
watchdog as an independent backstop.

It must abort on the Phase 0 reasons plus a broken refresh gap derived from
Phase 1. The full response records every observation, decision, refresh write,
stop attempt, and final state.

### Phase 2 pass criteria

- no intermediate stop before the final/abort stop;
- signed heading error and absolute cross-track both trend toward zero;
- no oscillation between saturated ±180 commands;
- cross-track never exceeds 0.20 m and the 0.30 m hard abort never fires;
- motion duty cycle is at least 80%;
- final stop is confirmed and the motion gate is disarmed.

## Phase 3 — waypoint A/B

After the steering window passes, compare the continuous executor with the
existing pulsed controller on matched, separately authorized contained 3 m
routes. Predeclare success as:

- 3 of 3 continuous landings within the accepted 0.15 m tolerance;
- no containment, stale-feed, BLE, refresh, or stop failure;
- maximum cross-track at or below 0.20 m;
- motion duty cycle at least 80%;
- average speed at least 0.20 m/s, versus the measured pulsed ~0.055 m/s;
- every completion and abort ends with a confirmed stop and disarmed gate.

Only that result answers “continuous waypoint control works.” Phase 1 answers
whether the telemetry can support it; Phase 2 answers whether steering can be
closed while moving.

## Explicit non-goals

- No continuous executor or Home Assistant service exists yet.
- No frozen `LUBA_ACCEPTANCE_PROFILE` value changes.
- No attempt is made to lower the 15° pulsed realignment floor or raise its
  realignment count; continuous lookahead is a separate control law.
- No hardware run is authorized by this document or by the offline prototype.

References: `docs/arcs-work-20260812.md`,
`docs/evidence-position-predictability-20260821.json`,
`docs/evidence-position-report-cadence-20260821.json`, and
`docs/evidence-route-b-3x3m-beta69-20260821T193417Z.json`.
