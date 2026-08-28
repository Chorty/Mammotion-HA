# Is the firmware working against us? — APK ground truth, 2026-08-27

Offline analysis of the decompiled vendor app at
`/Users/mattjoslin/mammotion-apk-decompile/src`. **No motion, no device contact.**

Short answer: the firmware is the binding constraint on Phase 2, but it is not
sabotaging us — and one hypothesis I raised is refuted outright by the app's own
source, which also overturns an inference this project has been leaning on.

## 1. The channel-dilution hypothesis is REFUTED

I proposed that subscribing to seven report channels when the controller needs two
might dilute the position rate, and that trimming the set could buy cadence.

`MACommandHelper.java:1317` — the vendor's own location subscription:

```java
getMctrlSysBuilder(RPT_START,
  {RIT_CONNECT, RIT_RTK, RIT_DEV_LOCAL, RIT_WORK, RIT_DEV_STA,
   RIT_VISION_POINT, RIT_VIO, RIT_VISION_STATISTIC,
   RIT_MAINTAIN, RIT_BASESTATION_INFO},
  10000,   // timeout
  1000,    // period
  4000,    // no_change_period
  count)
```

🗑️ **The app requests TEN channels; we request seven.** If channel count diluted
position cadence the vendor would be worse off than us, not better. **Do not run a
channel-set experiment** — it would have cost a release and a stationary run to
learn what the source says for free.

🔑 **The app asks for `period = 1000`.** Its own defaults are `iotMsgHz = 2000`
and `iotSameMsgHz = 4000` (`IotResp.java:36-37`) — *slower* than what we request.
Nothing in the app asks for sub-second position. Combined with the beta76 matrix
finding that the device ignores 100/250/500 ms requests, **~1 Hz is the intended
design rate over this link, not a clamp we are accidentally triggering.**

## 2. 🚨 The "vendor proves 1 Hz is enough" inference is UNSOUND

CLAUDE.md currently reasons: *"the vendor drives continuously at ~0.55 m/s on this
same ~1 Hz feed, so 1 Hz does not block a continuous controller."* That inference
does not hold, and it is load-bearing for Phase 2.

**The nav protocol has no point-to-point drive primitive.** Enumerating every
message class in `MctrlNav.java` turns up `NavReqCoverPath`, `NavTaskCtrl`,
`NavEdgePoints`, `NavTaskBreakPoint` — coverage planning and task control. There is
no "drive to this coordinate" command. The app's only raw motion primitive is
`DrvMotionCtrl` with `setLinearSpeed` / `setAngularSpeed`
(`MACommandHelper.java:1507`) — the joystick path, which is what we use.

So when the vendor mows at 0.55 m/s it is **not** closing a position loop over BLE.
It requests a coverage path, and **the mower executes that path with its own
onboard controller**, using sensors at whatever internal rate the firmware runs.
The 1 Hz stream is telemetry for display, not the control signal.

🔑 **Therefore the mower's INTERNAL position/control rate is almost certainly much
higher than 1 Hz.** No firmware drives a 0.55 m/s machine on 1 Hz position. What
we see over BLE is a downsampled report, and the vendor's fluency is evidence
about *onboard* control, not about what a remote 1 Hz loop can achieve.

⚠️ **This does not make Phase 2 impossible.** It does mean Phase 2 is attempting
something the vendor architecture never attempts: remote closed-loop steering at a
telemetry rate far below the machine's own control rate. The existing plan already
concedes the consequence — "feed-forward-dominated by necessity, not a tight
tracking loop" — but the vendor comparison should stop being cited as evidence
that a 1 Hz loop is sufficient.

## 3. Differences worth knowing, none yet shown to matter

* **`no_change_period`:** the app sends **4000**, we send **1000**. Irrelevant
  while moving (values are changing), but it means our stationary probes may see
  *more* traffic than the vendor's. Do not quote our idle cadence as "what the
  device does" without noting this.
* **Channels we never request:** `RIT_RTK`, `RIT_VISION_POINT`,
  `RIT_VISION_STATISTIC`. No evidence we need them — RTK status reaches us through
  the channels we do request — but it is a real difference in what is asked for.
* **`timeout = 10000`** matches ours; the 10 s device-side subscription window and
  its renewal requirement are shared.

## 4. What the firmware genuinely does constrain — all previously established

None of this is new, but it belongs in one place as the answer to "is the firmware
working against us":

* **~1 Hz position reporting**, ignoring shorter requested periods. Root cause of
  acquisition needing ~2 samples, ~0.248 m of blind travel per correction against
  a 0.30 m abort, and attempt 3's coin-flip miss.
* **The motion watchdog:** it stops the motor unless movement commands are
  re-sent (~200 ms in the app). Produced the fixed 4-inch step and the rotation
  quantum until it was found.
* **An actuation deadband:** angular 180 will not pivot a stationary mower; 500
  will.
* **`no_change_period` suppression:** a stationary mower is legitimately quiet.

⚠️ **None of these caused the four Phase 2 refusals.** Attempt 1 was my
`maxsize=1` bug, attempt 2 my `duration_ms: 2000`, attempt 3 one ordinary
interval. Blaming firmware for those would have prevented the fixes.

## 5. Research items this opens, ranked

1. 🔑 **Can the firmware's own path execution serve click-to-path?** If the mower
   can be asked to drive a short planned route with its onboard controller, that
   delivers fluidity without remote closed-loop control at all — the thing Phase 2
   is straining to build. `NavReqCoverPath` takes zone hashes and is
   coverage-shaped, so it may not accept an arbitrary two-point path; that needs
   checking before it is either adopted or dismissed. **Highest value if it works.**
2. **Is the internal position rate observable?** RTK raw fields, nav status, or a
   faster channel would confirm §2's reframing and bound what a remote loop could
   ever see.
3. **Does `no_change_period = 4000` change anything for us?** Cheap to test in a
   stationary probe; likely irrelevant while moving.
4. 🗑️ **Channel-set trimming — CLOSED.** Refuted above; do not spend a run on it.

## 6. Correction owed elsewhere

`CLAUDE.md` states the vendor comparison as support for 1 Hz being workable. That
sentence should be qualified with §2 — the vendor is not doing what we are doing.
