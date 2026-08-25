# Position-event polling remediation plan — 2026-08-25

## Offline implementation status

Offline implementation and the supporting PyMammotion fork release are
complete; no Home Assistant deployment or call, BLE call, or motion has been
performed. PyMammotion `0.8.12.post2` was published from commit `413ee71` as a
489063-byte wheel with SHA-256
`40ff8b3e21b67b77c7d490ffc8a285c8fc8911f8aa30d8ee404a84b01e7b09f4`.
Its focused handle/BLE/client-loop suite passes 168 tests, its available broader unit suite
passes 954 tests, and focused mypy is clean. Two unrelated PyMammotion modules
remain uncollectable in this environment: one needs optional `pymap3d`, and one
expects absent `examples/dev_output` reference files.

Home Assistant now has position-specific origin/controller/travel-guard
consumers, pipeline diagnostics, an acquisition-only action, a hard block on
real continuous steering, exclusive report ownership, and a randomized
three-repeat cadence-matrix runner. The full Home Assistant suite passes 921
tests; Ruff and mypy are clean. The integration now pins the verified post2
wheel. Final integration packaging review is the remaining release blocker.

## Decision

Replace message-generic freshness with position-specific evidence delivered
through the existing PyMammotion connection. Migrate origin admission, the
in-window decision loop, and every position/travel staleness guard together;
leaving any one of them on `DeviceHandle.last_report_at` would move the same
ambiguity into a different safety path.

The live tests prove that aggregate inbound traffic plateaued near 2 Hz in the
tested configuration. They do **not** prove the mower firmware's position
cadence or that requested report periods were ignored: the current probe counts
all `LubaMsg` traffic, and background PyMammotion report-stream commands can
replace the tested subscription. Position cadence remains unattributed until a
position-specific, isolated experiment measures it.

Do not open a second BLE connection. Do not enable steering in this
remediation. Retain the 1.06 m blind-acquisition disk as a geometric bound on
the configured two seconds of motion plus stopping overshoot; it does not
depend on claiming a particular telemetry cadence.

## Evidence and corrected diagnosis

The 2026-08-25 preflight reported `MODE_READY`, RTK `Fix`, `AREA_INSIDE`, blades
off, cutter RPM 0, and a live BLE link. Eight-second stationary probes requested
1000, 500, 250, and 100 ms report periods and observed 1.581, 1.936, 2.069, and
1.843 Hz aggregate traffic respectively. The 100 ms cell had a 1.8403 s maximum
gap. These results demonstrate a handle-visible aggregate ceiling during the
test, but the flat/non-monotonic rates and mixed message stream cannot attribute
that ceiling to position payloads or firmware.

The fresh-origin failure is proven independently of that cadence question:

- PyMammotion stamps `last_report_at` for every parsed `LubaMsg`, before broker
  handling and reduction.
- PyMammotion emits state changes only when the reduced model differs, so an
  identical fresh position payload is invisible to the Home Assistant
  coordinator.
- `_wait_for_fresh_continuous_origin` requires both a new generic report stamp
  and changed x/y. A stationary mower with stable coordinates therefore cannot
  produce admissible evidence.
- The authorized guarded invocation timed out after 2.0002 s and attempted no
  movement, while an external 198.4 ms-median HA poll saw unchanged cached
  coordinates.

Two additional safety defects must be fixed in the same change:

- `_continuous_decision_loop` treats any changed `last_report_at` as a fresh
  position and uses the cached x/y to update cumulative distance.
- It currently supplies `telemetry_age_s=0.0`, disabling the controller's
  telemetry-staleness gate and prediction horizon.

Raw temporary artifacts remain local and must not be copied into logs, release
notes, or external reviews:

- `/private/tmp/daytime-polling-bearing-scan.json`
- `/private/tmp/daytime-polling-motion-window.json`
- `/private/tmp/daytime-polling-motion-capture.jsonl`

## Implementation

### PyMammotion position evidence

Add immutable `PositionSample` and `PositionSampleStream` public types.
`PositionSample` contains:

- monotonically increasing position sequence and connection epoch;
- x, y, `toward`, position type, zone hash, RTK status, source, and transport;
- transport receipt and publication monotonic timestamps;
- optional decode, broker, reducer, and state-apply timestamps for diagnostics;
- `valid_for_motion` plus a rejection reason.

Stamp receipt before protobuf parsing and before debug serialization. Guard the
existing `to_dict()` debug work with `isEnabledFor(DEBUG)` so logging cost cannot
be misreported as transport latency.

Recognize `toapp_report_data.locations[0]` and rapid-state
`system_tard_state_tunnel` position payloads before reduction. After successful
reduction and state-machine application, publish a sample from the resulting
snapshot even when x/y is identical to the preceding sample. Invalid or
non-finite payloads may be published for diagnostics but are never admissible
for motion.

`DeviceHandle.open_position_sample_stream(maxsize=1)` returns a closeable RAII
subscription with an `asyncio.Queue[PositionSample]`. Delivery is latest-wins
and records dropped samples. Increment the epoch and invalidate queued evidence
on transport disconnect and transport replacement. Preserve `last_report_at`
unchanged for aggregate transport health.

Release this contract as Chorty PyMammotion `0.8.12.post2`, based on
`release/0.8.12.post1`.

### Home Assistant consumers and diagnostics

Pin `0.8.12.post2` and fail closed with `position_stream_unavailable` when the
API is absent; never fall back to the old x/y-plus-generic-stamp admission.

Convert all position freshness consumers in one integration change:

- fresh-origin admission accepts a valid sample received after the report
  request baseline, even if x/y is unchanged;
- `_continuous_decision_loop` consumes position sequence/epoch rather than
  polling `coordinator.data` for generic report stamps;
- cumulative distance advances only across consecutive position samples;
- `_apply_travel_guard` and position feed-stall detection use position evidence,
  not aggregate traffic;
- receipt-to-consumption age supplies `telemetry_age_s`; decode/reducer time is
  therefore charged rather than hidden;
- an epoch change, sequence regression, sequence gap, queue drop, or evidence
  older than 2.0 s requests stop immediately.

The coordinator keeps a separate RAII position subscription for presentation
and diagnostics while normal entity updates continue through
`async_set_updated_data`. Expose only derived ages/durations—not sensitive raw
transport data—in `export_runtime_state.position_pipeline`: latest
sequence/epoch/source/transport, receipt age, pipeline latency, coordinator
latency, payload cadence, and dropped-sample count.

Extend `report_stream_probe` additively. Preserve current aggregate-report and
changed-coordinate fields, but label them accurately and add a
`position_payloads` channel based on `PositionSample` arrivals. Wrap every
temporary stream/subscription in `try/finally` so early refusal, cancellation,
and exceptions close subscriptions and stop requested report streams.

### Isolated cadence experiment

Do not publish a firmware-period conclusion from the existing 8-second matrix.
Add a diagnostic mode that temporarily suppresses the handle's background BLE
stream renewal while it owns the test subscription, restores it in `finally`,
and refuses to run during motion or another stream owner.

For each requested 1000, 500, 250, and 100 ms period:

- hold `period` and `no_change_period` equal;
- collect at least 100 position payload arrivals per cell;
- run at least three repeats in randomized order;
- report inter-arrival distribution, gaps, invalid samples, sequence gaps, and
  stage latencies;
- classify a period as honored only when position-payload p95 is no greater
  than 1.5 times the requested period and no competing reconfiguration occurred.

The controller continues requesting the existing 1000 ms stream until this
experiment proves a different stable position-payload cadence.

### Acquisition-only safety action

Add `heading_acquisition_window`; do not silently repurpose the steering
service. Reuse the existing blade, ready/off-dock, BLE, RTK, map, operator, and
1.06 m acquisition-clearance gates. Experimental v1 remains fixed at
`linear_speed=400`, `angular_speed=0`, 200 ms refresh, a 2.0 s acquisition
limit, and 1.0 m hard distance.

Subscribe before requesting reports, establish a fresh position origin, then
start the shared safety clock immediately before the first movement dispatch.
Charge command/refresh latency to the acquisition budget. Drive straight only
until the first qualifying 0.15 m chord or any timeout, stale evidence,
sequence/epoch fault, refresh/BLE failure, corridor breach, or distance bound.
Every exit requests stop through a `finally`-protected path.

After acknowledged stop, wait at most 2.0 s for a newer valid position sample
and use the latest sample to capture delayed final displacement. Return heading
evidence only when the final chord remains at least 0.15 m. Persist no heading
token and never resume steering. Real `continuous_motion_window` execution is
blocked with `steering_not_motion_validated`; its dry-run geometry remains
available.

## Verification and acceptance

### PyMammotion tests

- Generic messages advance `last_report_at` but publish no position sample.
- Identical position payloads publish distinct increasing sequences.
- Both supported location sources publish after reducer/state application.
- Receipt precedes decode, broker, reducer, state application, and publication.
- Samples are immutable; queue overflow is latest-wins and counted.
- Disconnect/replacement changes epoch and invalidates prior evidence.
- Closing a subscription or stopping a handle leaves no task or queue leak.

### Integration tests

- Unchanged fresh x/y satisfies origin freshness; unrelated traffic cannot.
- Origin, decision, distance, and stale-feed paths contain no position use of
  `last_report_at`.
- Receipt-to-consumption age drives `telemetry_age_s` and can trip
  `telemetry_stale`.
- Sequence gaps, epoch changes, queue drops, and stale samples stop.
- Cumulative distance cannot silently undercount skipped position samples.
- Early return, exception, cancellation, and normal completion all unsubscribe,
  stop report streams, issue the required motion stop when applicable, and
  leave the experimental gate disarmed.
- Chords below 0.15 m remain diagnostic only; 0.15 m qualifies.
- Delayed post-stop telemetry cannot extend blind motion.
- Existing heading-sign, corridor, blade, BLE, refresh, and hard-abort tests
  retain their safety effect.

Run focused and full pytest suites in both repositories, Ruff, mypy/type checks,
and pre-commit without rewrite flags.

Release/deploy only after code review and clean offline verification. Deploy
motion-disabled, run the isolated stationary cadence experiment, and require no
position sequence gaps, no controller queue drops, correct identical-payload
counting, and receipt-to-consumption age below the 2.0 s safety limit. A tighter
software-latency target may be adopted only from the measured distribution; the
former arbitrary 250 ms gate is removed.

A physical acquisition-only run requires a fresh scan and new per-run
authorization. Steering remains deferred.
