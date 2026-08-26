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
  🔑 **Re-derived independently 2026-08-25.** The 28/1432 figures come from the
  COMPOSITE. On the untouched matrix it is **24 of 1315**, same 2.909 s maximum.
  Both are correct; say which one is being quoted.
- 🔑 **The number that actually sizes the 3.5 s bound is the publication
  interval, not the receipt interval.** Reconstructed as
  `interval_i + pipeline_(i+1) - pipeline_i` over both raw files: max
  **2910.1 ms**, p99 2344.2 ms, n=1434, and **zero** intervals above 3.5 s. So
  3.5 s holds, at a margin of **1.20x** over roughly 24 minutes of one
  stationary session. That is adequate and thin; treat it as a conservative
  stationary default, never as a proven distribution.
- ⚠️ **The HA pipeline itself showed one 1526.5 ms receipt-to-publication stall**
  (original cell 4, sample 18) against a 10.7 ms median / 21.1 ms p95 / 32.0 ms
  p99 over 1327 samples. It happened to overlap a receipt gap rather than extend
  one, so it does not widen the worst publication interval above. Treat that as
  luck, not structure: "delivery is tens of milliseconds" is the median claim,
  not a bound.
- 🔑 **A second, independent confirmation that change detection cannot prove
  position freshness:** the probe's own `channels.position` fingerprint detector
  reported **0 updates in all 12 cells**, including the eleven where 120 payloads
  demonstrably arrived. A stationary mower's x/y/`toward` never change, so every
  change-based freshness test -- including `last_report_at` -- is blind exactly
  where it matters. Only the payload-sequence counter saw the feed.
- 🔑 **Cell 12 is not a marginal gap.** Its three payloads arrived in a burst
  (206 ms, then 303 ms) and were followed by roughly 119 seconds of position
  silence while generic traffic held its normal ~1 Hz shape. Any future
  "position channel stalled" verdict should be held against that signature, not
  against a single long interval.
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

## Implementation status — beta77 / post3 (reviewed and released)

The code portion of this plan is implemented as Home Assistant `0.6.4-beta77`
and PyMammotion `0.8.12.post3`. post3 is published on the Chorty fork
(`chorty-0.8.12.post3`); the wheel at the exact URL named in `manifest.json`
was downloaded back and hashes
`cd3b0c3558d05c3ea6c7b6f2faad68c9c9eac523e70406bb930c5b01045a887a`.
It has **not** been used to command the mower and has **not** reconfigured a
live report subscription.

- PyMammotion now serializes temporary report ownership with an async lease,
  records an owner and generation for every START, enqueues a quiescing
  `RPT_STOP` before yielding ownership, and rearms the background owner once on
  release or cancellation.
  ⚠️ **Corrected 2026-08-25 during independent review:** the lease does *not*
  prove the background stream went quiet. `DeviceCommandQueue.enqueue` returns
  on queueing, a `BACKGROUND` item is dropped outright while a saga is active,
  and the owner flag cannot preempt a background loop iteration already past its
  guard. The lease field is therefore named `background_stop_enqueued_at_monotonic`
  and is evidence of intent only. The single positive proof that a configuration
  is live remains a position payload inside its own generation.
- The isolated HA probe requires the first ordered, valid position payload in
  the current generation, received at or after the START command has been
  flushed off the device command queue (not merely returned to the caller).
  Generic reports without a position now produce `position_channel_stalled`;
  lease loss, generation change, epoch change, queue replacement, and sequence
  gaps fail closed. Non-isolated calls keep the beta76 behaviour and take no
  readiness evidence.
- `report_stream_sequence_probe` and
  `scripts/position_subscription_transition_test.py` hold one lease across all
  cells. The cadence-matrix runner now submits its randomized schedule as one
  serialized sequence instead of repeatedly releasing background ownership.
- Position diagnostics separate presentation-stream replacements from unknown
  per-consumer safety drops and report receipt, decode, broker, reducer, state,
  publication, and controller-consumption latency distributions.
- A fresh-origin timeout reports `fresh_origin_timeout` with a separate
  `generic_report_advanced` field, and does NOT reuse `position_channel_stalled`.
  That wait is bounded by `max_heading_acquisition_s` (2.0 s), and 1.95 percent
  of healthy stationary intervals already exceed 2.0 s, so promoting a routine
  tail gap to a channel-fault verdict would have made the two indistinguishable.
  The readiness probe keeps `position_channel_stalled` at its 3.5 s budget, where
  nothing in 1434 intervals exceeded the bound.
- The acquisition window records `saga_active_before_request`. Both report-start
  calls reach the command queue at `Priority.BACKGROUND` with
  `skip_if_saga_active=True`, so an active saga drops them silently while the
  result still reports `started`; without that capture a dispatch failure is
  scored as telemetry evidence.
- Heading acquisition still commands at most the original two-second blind
  motion window. It then stops and permits up to 3.5 seconds of stationary
  observation; delayed heading evidence is diagnostic only and cannot reopen
  steering in that invocation.
- Installed PyMammotion version lookup is cached and warmed in Home Assistant's
  executor during setup, removing distribution-metadata I/O from synchronous
  entity availability evaluation.

Offline verification completed with 930 HA tests and 170 focused PyMammotion
tests passing. A wider PyMammotion unit run passed 993 tests and hit one
pre-existing missing reference artifact; collection of another existing test
also requires the absent `pymap3d` development dependency. Ruff and mypy pass
for the modified HA surface, the modified PyMammotion source is type-clean in
isolated checking, and a local `0.8.12.post3` wheel built and imported with the
new public lease types.

The next gate is still stationary live evidence: run at least 30 ownership
transitions, and only if every generation becomes position-ready rerun the
untouched randomized matrix. Continuous steering remains blocked, and nothing
here authorizes physical motion.

⚠️ **The 30-transition criterion only became meaningful with the flush-boundary
fix.** Before it, a cell could be certified position-ready by a payload still
arriving from the configuration it was replacing, so a clean 30/30 would have
been partly unfalsifiable. When the run happens, also record
`background_stop_enqueued` per lease and `saga_active_before_request` per cell,
so a clean result is *attributable* rather than merely clean.

⚠️ **The matrix rerun is now one ~28-minute HTTP request with no partial
results.** One lease across every cell is the point of the redesign, but a
client-side drop at cell 11 loses all eleven completed cells and Home Assistant
keeps holding the lease until the service returns. The runner prints this before
starting; plan the connection accordingly.

## Claude handoff boundary — review completed 2026-08-25

The independent review is **done**, and its three adjudication questions are
answered below with what the code actually does rather than what was intended.
The takeover prompt is `docs/CLAUDE-BETA77-TAKEOVER-PROMPT-20260825.md`.

Answers, in the order the questions are posed:

1. **No further PyMammotion instrumentation is required for the stationary
   phase**, now that the boundary is the queue flush. That is the tightest
   observable that exists on this path and is provably a lower bound on "the
   START reached the transport". A true device-side send/ack timestamp is worth
   adding before any *motion* use of this evidence, not before the stationary
   acceptance run.
2. **Teardown ordering holds and ownership stays split.** Every path — normal
   return, early readiness failure, exception, cancellation — routes through
   `finally`: STOP attempted, stream closed, lease released, exactly one rearm.
   `subscription_attempted` (not `subscription_started`) drives teardown, so a
   lost ack still tears down. Moving teardown into the library would require a
   **blocking** drain in lease acquisition; against the standing decision that
   being wrongly blocked is the worse failure, that is an operator call and was
   not taken unilaterally.
3. **3.5 s is an acceptable stationary default and is presented as nothing
   more.** See the publication-interval derivation above: 1.20x margin, one
   session, not a post3 distribution, and not motion validation.

Claude must adjudicate these contract questions from the code before calling
the patch release-ready:

1. The plan asks for command enqueue, send, acknowledgement, first-generic,
   and first-position timestamps. The current HA surface records the generation
   request boundary, a command queue-flush boundary (NOT a device
   acknowledgement -- no such acknowledgement is available on this path),
   position pipeline stages, controller consumption, and whether generic traffic
   advanced, but it does not expose an exact lower-level enqueue/send or
   first-generic timestamp.
   Decide whether the existing evidence is sufficient or whether PyMammotion
   needs another immutable command/report event.
2. PyMammotion's lease quiesces background owners and rearms exactly once;
   HA's lease users own the explicit STOP in their `finally` paths. Verify every
   cancellation and uncertain-ack path preserves the required
   `stop -> acknowledgement -> lease release -> one rearm` ordering, or move
   more of that lifecycle into the library.
3. The 3.5-second stationary observation bound is conservative relative to the
   beta76 maximum 2.909-second position gap, but it is not a newly collected
   post3 distribution. Confirm that this is an acceptable stationary
   acceptance-test default rather than presenting it as motion validation.

No earlier authorization carries into the handoff. Publishing, deployment,
stationary report reconfiguration, and especially physical movement each need
the operator's current direction. Continuous steering must stay unavailable
through this review and the stationary acceptance phase.

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
