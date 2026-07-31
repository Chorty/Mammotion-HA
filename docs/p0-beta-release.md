# P0 beta release status

## Maturity stage

This release completes **Alpha**: the features are done and every safety gate
fails closed, but known bugs remain. The three stages are exit criteria, not
version labels -- the version scheme stays `0.6.x-betaN` because
`beta-release.yml` numbers from it and prior builds already shipped as `-betaN`.

| Stage | Meaning | Exit criteria |
| --- | --- | --- |
| **Alpha** | Features complete, known bugs remain | Every safety gate fails closed; no unbounded motion; abort always wins |
| **Beta** | Fewer bugs; safety items resolved | Turn granularity solved; BLE link holds a full path run; no known way to strand a live client |
| **Release** | All safety work done bar cosmetics | Non-LUBA hardware characterized or explicitly refused; no open safety defect |

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
5. Then run the supervised acceptance sequence below.

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

| # | Gate | Pass criteria |
| --- | --- | --- |
| 1 | Confirmed zero stop | `mammotion.stop_manual_motion` reports `stop_confirmed` and `all_stop_writes_confirmed` true; the session is marked cancelled *before* the three emergency writes; a subsequent nonzero dispatch is refused with `ManualMotionCancelledError` |
| 2 | Short straight segment | One segment reaches `target_reached` within tolerance, with an explicit stop after the final pulse |
| 3 | Abort mid-run | Operator stop during a multi-pulse run; no movement command arrives after the stop, and no delayed replay occurs when the queue drains |
| 4 | Two-segment L-path | Both segments report `target_reached`; the second only starts after the first is marked passed |

Abort rule: if any gate fails, stop the session, disable experimental motion,
and record the failure before retrying. Do not iterate on a failing gate with
the mower live.

### Results

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

**Gate 2 — NOT PASSED, stopped 2026-07-30 20:10 EDT.** A preparatory single
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

⚠️ **Confirmed-write latency is closer to the guard timeout than expected.** The
three writes took 739, **1982**, and 191 ms on a *good* −50 dBm link.
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

| Previous HA enum state/option | New state/option | Compatibility |
| --- | --- | --- |
| `MODE_READY` and other uppercase mower enum labels | `mode_ready` and lowercase equivalents | Original label is in `raw_protocol_value`. |
| `AUTO`, `FLOOR`, `WALL`, etc. | `auto`, `floor`, `wall`, etc. | Select entity methods normalize legacy case during migration. |
| `MAN`, `WOMAN`, language labels | `man`, `woman`, lowercase language | Wire commands are converted back to vendor enum names. |
| Uppercase RTK, task-area, and SPINO sensor states | Lowercase equivalent | Update automations, templates, and dashboard conditions. |
| `mammotion.get_tokens` | Removed | Use the native camera/WebRTC entity; credentials stay server-side. |

## Verified limitations

- Only LUBA is eligible for supervised live acceptance in this release.
- Turn accuracy is unresolved: the rotation quantum is roughly 50 degrees per
  pulse, so a requested heading can overshoot. Straight segments land within
  about 1 cm along-track.
- The BLE link is the practical constraint, not the code. Measured over 8 hours
  while docked: median session 59 s, and 42% of disconnects are `0x08`
  supervision timeouts with a 41 s median. A long path run may outlive its link.
- No P1/P2 feature additions are included.
- RTK and SPINO firmware installation remains blocked pending hardware-derived
  prerequisite acceptance.

## Alpha to Beta

- **Turn granularity.** Wire `motion_refresh_interval_ms` into
  `vio_turn_to_heading` -- app-parity refresh gave roughly 7x on a properly
  powered turn at angular 500 -- then re-derive heading tolerance and
  `min_progress_distance` from taped measurements.
- **BLE session lifetime.** A full path run should fit inside one link. Re-measure
  with `scripts/ble_session_report.py` now that the slot-leak fix is pinned, and
  compare against the 2026-07-27 baseline.
- **Task-2 constants** remain un-re-derived after the transport failures.
- **Map edits are not picked up until an HA restart** -- the per-tick map block is
  unreachable in steady state.
- **`no_actuation_detected` fires falsely in the turn phase**; the unused
  discriminator is `heading_went_fresh`.

## Rollback

Disable experimental motion first, restore the previous HACS version, restart
Home Assistant, and remove or update the click-to-go resource cache key.

Reverting the backend pin is inherently safe: with plain upstream 0.8.12
installed, the capability probes report the teardown fix absent and re-lock real
motion automatically, with no other change required.
