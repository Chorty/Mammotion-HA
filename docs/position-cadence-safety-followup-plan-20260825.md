# Position-cadence safety follow-up plan — 2026-08-25

## Status and decision

Home Assistant `0.6.4-beta76` and PyMammotion `0.8.12.post2` were deployed
motion-disabled. The isolated stationary report-period matrix completed without
movement. The mower remained at the same reported position, cutter RPM stayed
zero, the experimental-motion gate stayed disabled, and no motion session was
created.

The event path is not the dominant latency. Across the three accepted repeats
per requested period, position publication pipeline p95 was 15.2–23.1 ms and
p99 was 24.4–39.0 ms. The ordinary maximum was 49.0 ms, with one 1526.5 ms
pipeline outlier that requires stage attribution. Safety-consumer probe queues
reported zero drops and zero sequence gaps.

The mower/report-subscription behavior is the limiting factor:

- Position interval medians were approximately 1.015 seconds for every
  requested period.
- Position p95 was 1.117–1.149 seconds when pooled by requested period.
- Requested 100, 250, and 500 ms periods were not honored. The 1000 ms period
  met the predefined p95 criterion after one isolated retry.
- There were 28 position intervals longer than 2.0 seconds across 1432 accepted
  intervals. Per-period maxima ranged from 2.328 to 2.909 seconds.
- In the original final 1000 ms cell, 118 generic reports continued while only
  three position payloads arrived. A new isolated retry immediately recovered
  to 120 valid position samples. Generic traffic therefore cannot establish
  position freshness, and subscription ownership/readiness is not yet reliable
  enough for motion.

Keep real continuous steering blocked. Keep the controller request at 1000 ms;
requesting faster periods adds configuration traffic without improving the
measured position cadence. Do not open a second BLE connection.

Raw artifacts remain local:

- `/private/tmp/position-cadence-beta76-20260825.json` is the original matrix.
- `/private/tmp/position-cadence-beta76-1000-retry-20260825.json` is the isolated
  retry.
- `/private/tmp/position-cadence-beta76-composite-20260825.json` is explicitly
  marked as a derived classification that substitutes the retry for anomalous
  cell 12. It must not be represented as the untouched original matrix.

## Remediation plan

### 1. Make report ownership serialized and observable

- Replace the boolean exclusive flag with one async lease/lock carrying an
  owner ID and subscription generation.
- Keep one exclusive lease across a multi-cell experiment. Do not release and
  rearm the background BLE/MQTT loops between adjacent cells.
- On lease acquisition, cancel or quiesce background renewal and wait for an
  explicit acknowledgement before sending the test configuration.
- On release, stop the owned subscription, wait for command completion, then
  rearm background polling exactly once. Cancellation and exception paths must
  perform the same teardown.
- Record requested configuration, transport, command enqueue/send/ack times,
  generation, first generic report, and first position report. Do not call a
  subscription ready merely because its command returned.

### 2. Gate readiness on position evidence

- After each `RPT_START`, require a new valid position sample in the current
  generation before starting an observation or motion budget.
- Detect the observed failure mode explicitly: generic reports advancing while
  position sequence does not. Report `position_channel_stalled` and stop or
  refuse; never fall back to `last_report_at`.
- Treat a generation change, lease loss, background reconfiguration, stream
  replacement, queue drop, or sequence gap as a fail-closed boundary.
- Add a short stationary verification mode that repeats start/stop ownership
  transitions many times and checks that every generation reaches position
  readiness. This targets the anomalous twelfth-cell failure without requiring
  another 24-minute full matrix.

### 3. Attribute the latency tail

- Report separate receipt-to-decode, decode-to-broker, broker-to-reducer,
  reducer-to-state-apply, state-apply-to-publication, and
  publication-to-controller-consumption distributions.
- Include p50/p95/p99/max and counts instead of relying only on raw arrays.
- Distinguish presentation-stream latest-wins replacements from safety-stream
  drops. The coordinator's maxsize-one diagnostic stream replacement counter
  must not be presented as a controller evidence gap.
- Investigate the 1526.5 ms publication outlier. Until attributed, charge the
  complete receipt-to-consumption age to every safety budget.

### 4. Change acquisition to stopped observation

- Do not extend the two-second blind-motion window merely to wait through a
  2.3–2.9 second telemetry gap; that would enlarge the required clearance disk.
- Keep angular speed zero and cap commanded movement using the existing shared
  time, distance, and 1.06 m blind-clearance envelope.
- Stop at the bounded acquisition deadline regardless of telemetry, then wait
  while stationary for a delayed final position sample. Deriving a heading
  after acknowledged stop is diagnostic evidence only and must never resume
  steering in the same invocation.
- Set the stationary post-stop observation timeout from a new no-motion gap
  distribution. It may exceed two seconds because the mower is already stopped;
  it must not extend commanded travel.
- A future motion controller should use stop-and-wait segments synchronized to
  verified position generations. Do not re-enable continuous feedback steering
  on an approximately 1 Hz feed with demonstrated multi-second gaps.

### 5. Verification and release gates

- Pure/unit tests: lease serialization; cancellation; background rearm exactly
  once; config generation mismatch; generic-without-position stall; first
  position readiness; stage timing order; presentation replacement versus
  safety drop.
- HA tests: no observation clock before position readiness; stalled position
  channel refusal; repeated isolated cells under one lease; every exit stops the
  subscription; motion gate and active session remain unchanged.
- Stationary live acceptance: at least 30 start/reconfigure/stop transitions,
  zero generations without a first valid position, zero safety-queue drops,
  and complete stage timing for every position payload.
- Re-run the randomized three-repeat matrix only after the transition test
  passes. Accept 1000 ms as the sole supported request period only if all three
  untouched cells contain at least 100 valid position payloads with no evidence
  gaps.
- Deploy motion-disabled. Physical acquisition still requires a new clear-area
  scan and per-run authorization. Continuous steering remains blocked until a
  separate safety review approves a stopped-segment design from live evidence.

## Separate deployment warning

The beta76 restart also reproduced Home Assistant event-loop warnings from
`installed_pymammotion_version()` calling `importlib.metadata.version()` inside
entity availability evaluation. Cache the installed version during async setup
or resolve it in the executor, then serve the cached value from synchronous
properties. This warning did not explain the one-hertz device cadence, but it
is a real integration defect and belongs in the next offline patch.
