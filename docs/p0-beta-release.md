# P0 beta release status

## Maturity stage

This branch has completed **Alpha implementation and supervised backend
acceptance**: every LUBA Gate 1-4 test passed and the safety gates fail closed,
but known release blockers remain. In particular, the card's built-in Real Go
defaults still differ from the deliberately bounded profile used for Gate 4;
backend acceptance must not be presented as acceptance of that older card
profile. The three stages are exit criteria, not version labels -- the version
scheme stays `0.6.x-betaN` because
`beta-release.yml` numbers from it and prior builds already shipped as `-betaN`.

| Stage       | Meaning                              | Exit criteria                                                                                 |
| ----------- | ------------------------------------ | --------------------------------------------------------------------------------------------- |
| **Alpha**   | Features complete, known bugs remain | Every safety gate fails closed; no unbounded motion; abort always wins                        |
| **Beta**    | Fewer bugs; safety items resolved    | Turn granularity solved; BLE link holds a full path run; no known way to strand a live client |
| **Release** | All safety work done bar cosmetics   | Non-LUBA hardware characterized or explicitly refused; no open safety defect                  |

See "Alpha to Beta" below for what closes the current gap.

## Safety model

- Experimental manual motion defaults off and is BLE-only.
- Real motion is authorized by **measuring the installed backend**, never by a
  version number. `custom_components/mammotion/backend_capability.py` probes the
  loaded code for `ble_teardown_failure_atomic` and `blufi_reassembly_reset` and
  requires both, plus a release at or above the audited base 0.8.12. A fork, a
  rebuild, or a future upstream release carrying any given number therefore
  cannot self-certify, and every probe fails closed: an exception, a missing
  attribute, unreadable source, a timeout, or simply never having probed all
  read as "capability absent".
- The pinned backend is **pymammotion 0.8.12.post1**, a Chorty fork build
  ([release](https://github.com/Chorty/PyMammotion/releases/tag/chorty-0.8.12.post1)),
  because no upstream release carries the teardown fix. It is released `v0.8.12`
  plus three commits: upstream's rate-limit fix, upstream's own BluFi reassembly
  reset, and the teardown failure-atomicity fix. It deliberately excludes
  upstream `main`'s later saga/token/transport refactor, which is unaudited for
  motion safety.
- Every real service run requires positive LUBA capability evidence, fresh BLE
  queue/liveness evidence, safe runtime state, both operator confirmations, and
  an exclusive backend session.
- `mammotion.stop_manual_motion` marks the active session cancelled before it
  queues three emergency-priority confirmed zero-velocity writes. Cancelled
  sessions cannot issue another nonzero confirmed write.
- Preview and dry-run accept seven destinations. Real click-to-go is limited to
  two segments.
- YUKA, RTK, SPINO, accessories, and unknown products are fixture-characterized
  and fail closed for hazardous actions until hardware acceptance exists.

## Deployment

1. Install the beta and restart Home Assistant with experimental motion off.
2. Confirm integration setup, maps/tasks, diagnostics, native camera entities,
   and card preview/dry-run.
3. Register
   `/mammotion/mammotion-custom-path-card.js?v=<installed-version>` as a
   JavaScript module dashboard resource.
4. Do not enable real motion while
   `export_runtime_state.experimental_motion.backend_verified` is false. When it
   is false, `blockers` names the specific missing capability.
5. Then run the supervised acceptance sequence below. The development LUBA
   completed it on 2026-07-31; every new hardware family and any materially
   changed motion profile requires its own acceptance.

Note: the backend is pinned as a wheel URL, and HA re-installs a URL requirement
on **every** start (`is_installed()` returns False whenever a requirement has a
URL). A restart without internet access can therefore fail integration setup.

## Supervised LUBA acceptance sequence

Run in order, stopping at the first failure. Preconditions, all required:

- Daylight. VIO will not initialize in a dark scene -- confirm
  `camera_brightness` is not `Dark` and `track_feature_num` is healthy.
- Blade off and physically verified. Operator present, within reach of the
  mower, with the physical e-stop **released** (an engaged e-stop is invisible in
  telemetry and silently no-ops every motion command).
- `export_runtime_state` shows `backend_verified: true` with no blockers once
  opt-in is on, and `ble_link_live` passing.
- Experimental motion enabled for this session only, and disabled afterwards.
- Capture `scripts/ble_session_report.py` across the whole window so session
  lifetime is measured rather than assumed.

| #   | Gate                   | Pass criteria                                                                                                                                                                                                                                 |
| --- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Confirmed zero stop    | `mammotion.stop_manual_motion` reports `stop_confirmed` and `all_stop_writes_confirmed` true; the session is marked cancelled _before_ the three emergency writes; a subsequent nonzero dispatch is refused with `ManualMotionCancelledError` |
| 2   | Short straight segment | One segment reaches `target_reached` within tolerance, with an explicit stop after the final pulse                                                                                                                                            |
| 3   | Abort mid-run          | Operator stop during a multi-pulse run; no movement command arrives after the stop, and no delayed replay occurs when the queue drains                                                                                                        |
| 4   | Two-segment L-path     | Both segments report `target_reached`; the second only starts after the first is marked passed                                                                                                                                                |

Abort rule: if any gate fails, stop the session, disable experimental motion,
and record the failure before retrying. Do not iterate on a failing gate with
the mower live.

### Results

Entries are chronological. Statements such as “Gates 3-4 remain untested” in
an earlier failed attempt describe the decision at that timestamp; the corrected
Gate 4 result at the end of this section is the current acceptance status.

**Gate 1 — PASSED 2026-07-29 22:32 EDT** on pymammotion 0.8.12.post1, in the dark,
mower docked at `MODE_READY`, BLE live at −50 dBm. `stop_confirmed: true`,
`all_stop_writes_confirmed: true`, all three emergency writes `ok`, total 2911 ms.
Delivery confirmed over BLE in the logs (30-byte `BLETransport send` frames, versus
the 28-byte keepalives).

Scope note: `session_was_active: false`, so this validated the **delivery** half of
the gate — three confirmed zero-velocity writes reaching the mower — not the
session-cancellation half, which needs an active run to abort. Gate 1 requires no
experimental-motion opt-in: `stop_manual_motion` is registered without the
authorization wrapper by design, so a stop always works.

**P1S revalidation — PASSED 2026-07-31 11:38 EDT.** With passive scanning and
active GATT proxying, the docked mower remained position- and heading-identical
while all three confirmed zero writes completed in 224.5, 104.3, and 84.5 ms
(413.4 ms total). The result again reported `stop_confirmed: true`,
`all_stop_writes_confirmed: true`, `owner_exited: true`, and no active session.
No proxy timeout, connection cooldown, or write failure followed.

**Gate 2 initial attempt — NOT PASSED, stopped 2026-07-30 20:10 EDT.** A preparatory single
750 ms pulse at service speed 0.4 proved the new idle/manual-motion report
stream: telemetry measured 5.18 cm immediately after the pulse and 4.77 cm two
seconds later, matching the operator's observation of forward motion with no
meaningful rollback. The write completed in 408 ms and its explicit zero stop
completed in 168 ms. That pulse also measured the live map-to-reported heading
offset at approximately +102.4 degrees.

The actual 10 cm straight-segment attempt did not actuate. Its first nonzero
confirmed write timed out at 4008 ms and the normal stop write then timed out at
4004 ms. The executor aborted with `command_failed`; position remained
bit-identical. The dedicated emergency stop immediately confirmed all three
zero-motion writes in 688, 172, and 342 ms, after which repeated position samples
showed no delayed replay. Runtime then degraded to `ble_send_stalled`. Per the
abort rule, no live retry was attempted and Gates 3-4 remain untested.

Proxy logs confirmed the failure below PyMammotion's motion controller. The
preferred ESPHome proxy hung on a GATT write for 30 seconds, disconnected,
reconnected about five seconds later, then returned GATT status 14 on the next
write. PyMammotion correctly made BLE unusable and armed its 120-second
connection cooldown. A second ESPHome proxy also lost its Home Assistant API
connection during the same interval. A momentary `connected` snapshot therefore
does not establish a safe motion window; stabilize the proxy/network path before
retrying Gate 2.

**Gate 2 P1S retry — NOT PASSED, stopped 2026-07-31 11:51 EDT.** The retry used
P1S as the only active proxy after the stationary/app A/B below. A fresh
immediate preflight passed every gate at -58 dBm: BLE active, queue open and
empty, RTK fixed, mapped `AREA_INSIDE` position, `MODE_READY`, blades at zero,
and no active route. The bounded executor sent exactly one nonzero
`send_movement(linear_speed=200, angular_speed=0)` write with refresh disabled,
then its explicit stop. The nonzero write confirmed in 171.7 ms and the stop in
124.1 ms, so the earlier transport timeout did not recur. The operator observed
roughly 0.5 in (1.27 cm) of physical forward movement, while position telemetry
advanced only 0.8 mm, below the 2.5 mm progress threshold, and remained fixed
through the 4, 8, 15, and 25 second samples. The command therefore actuated, but
the map-position feed did not resolve the small displacement. The phase
correctly failed closed as `no_target_progress`; there was no target completion
and no delayed replay.

The independent post-run stop then confirmed all three zero writes in 157.5,
80.9, and 115.8 ms (354.2 ms total). No session remained active, the mower still
reported blades off and `MODE_READY`, and experimental motion was disabled.
The scoped 15-minute BLE report showed MTU 517, no sequence gaps, no malformed
frames, and one clean local disconnect; the current P1S session remained open.
This retry clears the proxy/write-completion failure and shows that a single
300 ms command at linear speed 200 produces only a short firmware/dead-man step
that is not reliably visible in map telemetry. Do not repeat the same live
parameters: the executor cannot close the loop on displacement it cannot
measure. Re-derive the smallest bounded app-parity refresh window or a higher
single-shot speed offline, then obtain a new operator confirmation before
another physical run. Gates 3-4 remain untested.

**Gate 2 P1S two-write retry — PASSED 2026-07-31 12:02 EDT.** A fresh preflight
again passed every gate with P1S as the sole active proxy at -60 dBm. The test
used an aligned 10 cm target, linear speed 400, and the app's 200 ms refresh
cadence. Final-approach scaling reduced the nominal 3500 ms pulse to 330.2 ms,
which hard-bounded the run to exactly one initial nonzero write plus one refresh;
no turn command was scheduled. The initial write confirmed in 128.4 ms, the
single refresh completed inside the bounded window, and the mandatory stop
confirmed in 214.3 ms.

RTK measured 9.69 cm of travel on a 10 cm target, ending 5.6 mm from the target.
The executor reported `target_reached`, one linear pulse, one refresh, and no
blockers. Position settled after one second and remained stable through the 4,
8, 15, and 25 second samples, proving there was no delayed replay. The separate
post-run emergency stop confirmed three more zero writes in 114.5, 112.0, and
108.8 ms (335.3 ms total). The mower remained `MODE_READY`, blades off, RTK
fixed, and inside `Backyard Right`; no session remained active and experimental
motion was disabled. The scoped BLE report showed no disconnects, sequence
gaps, malformed frames, or dropped frames during the window. Gate 3 (abort
mid-run) is next; Gate 4 remains untested.

**Gate 3 P1S active-session abort — PASSED 2026-07-31 12:50 EDT.** Offline
preparation first found that `ManualMotionCancelledError` was being folded into
the refresh loop's ordinary resend-failure path. Nonzero dispatch was still
blocked, but the owner could remain alive in feedback/sample waits instead of
returning `operator_stop`. The cancellation now delivers a defensive stop and
propagates to the exclusive session wrapper; its focused tests and the complete
454-test Mammotion suite passed before deployment.

The live test used an aligned 30 cm fallback-bounded segment at linear speed
400 with the app's 200 ms cadence. An independent monitor waited for the first
nonzero GATT-confirmed dispatch, then called `stop_manual_motion` immediately.
The original executor returned `operator_stop` after 673 ms with the same
session ID, `cancelled: true`, and `cancel_reason: operator_stop`. The stop
service observed the active session, marked it aborted, confirmed all three
zero writes in 172.2, 263.6, and 106.7 ms, and reported `owner_exited: true` in
542.5 ms. The last completed dispatch was a stop.

Eleven runtime samples across the following 20 seconds remained bit-identical
at x 4.9575, y -3.0168, heading 173.4769. No session reappeared, no delayed
nonzero command replayed, BLE remained active, and blade RPM remained zero.
The scoped BLE report showed no disconnects, sequence gaps, malformed frames,
or dropped frames. Experimental motion was disabled afterward. Gate 4 remains
untested; the 176-degree turn regression must be revalidated before the
two-segment L-path.

The next preflight eventually reported x 4.9592, y -3.0386, a 2.19 cm net
displacement from the confirmed pre-abort command. This is delayed position
telemetry, not evidence of a delayed command: the session was already aborted,
its last completed dispatch remained a stop for the entire observation, and no
session or queue activity reappeared. It does mean a short unchanged-position
window cannot by itself prove that an abort produced zero physical travel.

**176-degree VIO turn regression — PASSED 2026-07-31.** From a fresh VIO heading
of -82.942 degrees, the standalone closed-loop turn targeted 93.058 degrees.
Three same-direction pulses advanced 58.417, 57.419, and 55.724 degrees, with a
confirmed zero stop after every pulse. The final VIO heading was 88.617 degrees:
171.559 degrees of measured rotation and only 4.441 degrees of residual error,
well inside the 18-degree gate. There was no overshoot/reversal, stale-heading
sample, no-progress streak, command failure, or safety blocker.

The turn translated 10.48 cm, below its 25 cm displacement guard but material
for click-to-go route accuracy. The independent post-run stop confirmed three
more zero writes in 208.8, 104.8, and 102.8 ms. Position and both heading feeds
then remained stable for 15 seconds; no session remained active, blades stayed
off, BLE stayed selected, and experimental motion was disabled. The scoped BLE
report again showed no disconnects, sequence gaps, or malformed/dropped frames.
Gate 4 can proceed, but its result must prove the second segment recalculates
from the post-turn position rather than assuming a perfectly in-place pivot.

**Gate 4 first L-path attempt — NOT PASSED, stopped 2026-07-31.** Segment 1
passed: one 10.46 cm calibration pulse derived a 358.206-degree map/VIO offset,
then one scaled refreshed pulse reached 5.6 cm from its 30 cm waypoint. Segment
2's initial VIO turn also passed, using two monotonic pulses to finish 14.56
degrees from its pre-turn target. Every calibration, turn, and linear pulse had
a confirmed zero stop.

The turn translated 14.43 cm. That changed the bearing from the mower's fresh
position, but the executor did not recalculate until after sending its linear
pulse. It therefore drove roughly 23 degrees off the new bearing and stopped
18.05 cm from the target at the one-pulse ceiling. Later position telemetry
settled another 9.5 cm away at x 4.7262, y -2.5376; no session or dispatch
reappeared, so this was delayed reporting of the bounded run rather than a
delayed replay. The independent emergency stop confirmed all three writes in
136.7, 104.3, and 101.6 ms, and experimental motion was disabled.

The executor now recomputes bearing after every translating VIO turn, performs
a bounded pre-linear correction when the fresh aim error exceeds the tighter of
the heading and realignment tolerances, and verifies the corrected heading from
fresh position before any forward write. The VIO turn now also receives the
caller's turn-translation limit; it had previously retained its 0.5 m default.
If correction is unavailable, exhausts its budget, or remains misaligned, the
segment fails before linear motion. Two focused regressions plus the complete
456-test Mammotion suite passed, the fix was deployed with a matching checksum,
and a fresh 60 cm L-path dry run from the settled position passed containment
and both segment plans. Gate 4 still requires a new supervised real retry.

**Gate 4 corrected L-path retry — PASSED 2026-07-31.** A fresh preflight
passed every gate at the exact dry-run start position, x 4.7262, y -2.5376,
with BLE at -58 dBm, fixed RTK, healthy daylight VIO, blades at zero, and no
active route. The bounded 60 cm path used two 30 cm legs and the two-real-
segment ceiling. Segment 1 was marked passed before segment 2 began and ended
0.96 cm from its waypoint. Segment 2 used two same-direction VIO turn pulses,
each with a confirmed stop. Although the turn translated 8.80 cm, the new
post-turn check recalculated the bearing from the fresh position: facing
171.894 degrees versus a 171.610-degree bearing, an aim error of only 0.285
degrees. No corrective pulse was needed, and the linear write was allowed only
after that alignment passed.

Both segments reported `target_reached`; the final position was x 4.3911,
y -2.8064, 4.70 cm from the requested endpoint and inside the 8 cm tolerance.
The run ended with `failed_segment_index: null`, two executed real segments,
and no active session. The independent stop then confirmed all three zero
writes in 219.0, 200.8, and 110.5 ms (530.4 ms total). Position and heading
remained bit-identical for the following 18 seconds, blades remained off, and
no session or delayed dispatch appeared. The scoped 15-minute BLE report had
no disconnect, sequence-gap, malformed-frame, or dropped-frame event. Finally,
experimental motion was disabled and runtime now blocks real motion with
`experimental_motion_disabled`. Gates 1-4 of the supervised LUBA acceptance
sequence are complete.

Do not enable broad `pymammotion: debug` logging during that diagnosis. Its cloud
gateway logs authenticated request data and network responses. Use only the
scoped `bleak_esphome` and `habluetooth` loggers from
`docs/deploy-runbook-p0.md`, and return them to normal levels after capture.
Do not enable `pymammotion.transport.ble: debug` either: a live stationary
capture proved that it logs raw BLE payloads and device identifiers.

### Stationary BLE isolation capture (2026-07-30)

A motion-disabled 30-minute capture separated scanner coverage from GATT
lifecycle behavior:

- Home Assistant received 13,434 control advertisements from other devices and
  three connectable mower advertisements at strong RSSI (-56 to -63 dBm).
- Four fresh mower connections all negotiated MTU 517. The three sessions that
  closed inside the report lasted 43, 287, and 607 seconds; a fourth was still
  open at the report boundary and closed locally about 195 seconds later.
- Every observed close was `error=0` and the owning proxy logged an explicit
  local `Disconnecting` command. There were no peer-terminated (`0x13`) or
  supervision-timeout (`0x08`) closes, sequence gaps, unparseable messages, or
  dropped malformed frames.
- The behavior crossed two proxies, so it is not isolated to the preferred
  printer proxy. One close followed another proxy's Home Assistant API loss by
  0.5 seconds; the mower link then reopened on its original proxy 4.8 seconds
  later. Other clean local closes had no matching proxy API event.
- During one live session, backend keepalives were current and the runtime
  motion gate reported no BLE blocker while the public `ble_link_live` entity
  remained off. This is a safe false-negative in the preflight, but it confirms
  the entity is not refreshed reliably enough to be an independent liveness
  source.

This narrows the current failure to client-side lifecycle churn: an HA Bluetooth
scanner/source change, or PyMammotion teardown after a notification/write
operation fails. It does not support weak signal, low MTU, corrupt-frame
reassembly, or mower-initiated disconnect as the primary cause of this capture.
The existing logs do not record a sanitized teardown initiator/error type, so a
stationary instrumented build is required to distinguish the remaining two
causes.

The follow-up direct-app comparison further reduced the mower-hardware
likelihood. Disabling the Mammotion integration produced an explicit clean
disconnect from the HA-owned proxy session, after which the official app
connected over BLE immediately without restarting the mower. With the integration
left disabled, the app then held BLE for 15 minutes. HA's passive scanner received
6,034 control advertisements from other devices and only the mower's initial
connectable advertisement; the mower never returned to advertising during the
900-second window. The operator repositioned the mower once, but otherwise
treated the run as the requested clean comparison.

This proves the recent "app requires a mower restart" symptom was caused by HA
holding the mower's single BLE connection, not by a radio that required rebooting.
It also shows the mower can sustain a substantially longer direct-app session
than the 43-, 287-, and 607-second HA sessions in the preceding capture. The next
isolation boundary is therefore native PyMammotion BLE versus PyMammotion through
HA/ESPHome; do not resume physical motion based on the app result alone.

### Single-proxy passive-scan A/B (2026-07-31)

The next stationary comparison isolated the ESPHome path without powering down
unrelated devices:

- With IRK Capture S3 as the only remote proxy, the mower advertised connectably
  at -63 to -65 dBm, but the proxy repeatedly lost its Home Assistant API
  session. Handshakes took 22-60 seconds, the proxy disappeared from the active
  scanner registry, and unrelated adjustable-bed/fitness-device GATT attempts
  failed through the same path. A seven-second Mammotion Bluetooth reset could
  not reconnect; PyMammotion correctly armed its 120-second cooldown.
- P1S was then flashed with passive advertisement scanning but active GATT
  proxying:

  ```yaml
  esp32_ble_tracker:
    scan_parameters:
      active: false

  bluetooth_proxy:
    active: true
  ```

  As the only remote proxy it registered as connectable, heard the mower at
  -42 dBm, and was selected with zero failures and two of three slots free.
  Mammotion became active over BLE at -44 dBm and `ble_link_live` passed after
  the initial command queue drained.

- Turning the Mammotion Bluetooth switch off completed in 0.16 seconds. P1S
  logged the mower disconnect 90 ms later with `error=0`, freed one connection
  slot, and did not replay a delayed reconnect. The official app then connected
  immediately and the operator drove the mower off the dock over BLE without
  restarting it. The mower was redocked and the app closed before HA reclaimed
  the link.
- After deploying the defects found by this test and restarting HA, the entry
  loaded, automatically reattached through P1S at -46 dBm, and returned
  `ble_link_live: on`. A final off/on cycle changed liveness immediately to
  `ble_client_not_connected`, then passed again after the normal queue-settle
  interval.

This rules out passive scanning, the mower radio, and ESPHome proxies in general
as the cause of the failed isolation window. The evidence points specifically
to the IRK proxy's firmware/configuration/API state. It also found two
Mammotion-HA defects: a cloud-backed mower did not register a late BLE
advertisement callback when no proxy was ready during setup, and cached motion
gate entities were not invalidated on a Bluetooth toggle. Both now have
regression tests. This stationary/app result does **not** reopen Gate 2 by
itself; integration-driven physical motion still requires a new supervised
operator confirmation and a stable P1S window.

⚠️ **Confirmed-write latency is closer to the guard timeout than expected.** The
three writes took 739, **1982**, and 191 ms on a _good_ −50 dBm link.
`_BLE_MOTION_WRITE_TIMEOUT_SECONDS` is 4.0 s, so the worst observed write used half
the budget. Gate 2 subsequently hit that deadline on both the movement and normal
stop writes. Increasing the deadline is not automatically safe: it also extends
the period in which a nonzero write has uncertain completion. Resolve the BLE
stall first, then derive the timeout from measured successful and failed
distributions. These are completion-confirmed times, not the sub-millisecond
`command_ok` acks that were already shown to prove nothing.

**Turn accuracy is deliberately not a gate.** Turns are bounded and always
explicitly stopped, so an inaccurate turn is a quality defect rather than a
safety one. It is tracked as the headline Alpha-to-Beta item below.

## Breaking migrations

| Previous HA enum state/option                      | New state/option                       | Compatibility                                                      |
| -------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------ |
| `MODE_READY` and other uppercase mower enum labels | `mode_ready` and lowercase equivalents | Original label is in `raw_protocol_value`.                         |
| `AUTO`, `FLOOR`, `WALL`, etc.                      | `auto`, `floor`, `wall`, etc.          | Select entity methods normalize legacy case during migration.      |
| `MAN`, `WOMAN`, language labels                    | `man`, `woman`, lowercase language     | Wire commands are converted back to vendor enum names.             |
| Uppercase RTK, task-area, and SPINO sensor states  | Lowercase equivalent                   | Update automations, templates, and dashboard conditions.           |
| `mammotion.get_tokens`                             | Removed                                | Use the native camera/WebRTC entity; credentials stay server-side. |

## Verified limitations

- Only LUBA is eligible for supervised live acceptance in this release.
- Refreshed VIO turn heading accuracy passed the 176-degree regression, finishing
  4.44 degrees from target without reversal. The nominally in-place turn drifted
  10.48 cm, so turn translation remains the quality limitation. Straight
  segments land within about 1 cm along-track.
- The BLE link is the practical constraint, not the code. Measured over 8 hours
  while docked: median session 59 s, and 42% of disconnects are `0x08`
  supervision timeouts with a 41 s median. A long path run may outlive its link.
- No P1/P2 feature additions are included.
- RTK and SPINO firmware installation remains blocked pending hardware-derived
  prerequisite acceptance.

## Alpha to Beta

- **Card execution profile.** The backend Gate 4 call used one linear command
  per segment, an 8 cm waypoint tolerance, 2.5 mm progress threshold, 102.4
  degree forward offset, four VIO turn commands, and BLE auto-recovery off. The
  card still emits its older July 18 defaults, including a 30-pulse linear
  ceiling, 15 cm tolerance, 6 cm progress threshold, 116.5 degree offset, and
  BLE auto-recovery on. Align or explicitly profile these values, update
  frontend assertions and README together, and do not call the default card
  Real Go hardware-accepted until then.
- **Turn translation and final tolerance.** Gate 4 now compensates from fresh
  post-turn position and passed 4.70 cm from its final target, but its turn
  still translated 8.80 cm and the standalone 176-degree turn drifted 10.48
  cm. Reduce the 18-degree heading tolerance and measure the refreshed
  turn-pulse floor without weakening the displacement guard.
- **BLE session lifetime.** The complete bounded Gate 4 run fit inside one link
  with no disconnect or malformed-frame event, but the longer docked baseline
  still has a 59-second median. Re-measure a stationary soak with
  `scripts/ble_session_report.py` and compare against the 2026-07-27 baseline;
  do not attribute improvement solely to teardown because the dependency jump
  also included other upstream changes.
- **Task-2 constants** remain un-re-derived after the transport failures.
- **Map edits are not picked up until an HA restart** -- the per-tick map block is
  unreachable in steady state.
- **`no_actuation_detected` fires falsely in the turn phase**; the unused
  discriminator is `heading_went_fresh`.
- **All-files pre-commit baseline.** The CI-scoped Ruff, format, mypy, frontend,
  JSON, diff, and 456-test checks pass, but `pre-commit run --all-files` is not
  a clean release gate: its legacy Ruff/codespell scope includes historical
  evidence and scripts, pyupgrade crashes on Python 3.14, and its mypy command
  differs from CI. Repair or deliberately scope those hooks instead of
  committing automatic formatting across unrelated evidence files.

## Rollback

Disable experimental motion first, restore the previous HACS version, restart
Home Assistant, and remove or update the click-to-go resource cache key.

Reverting the backend pin is inherently safe: with plain upstream 0.8.12
installed, the capability probes report the teardown fix absent and re-lock real
motion automatically, with no other change required.
