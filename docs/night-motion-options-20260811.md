# Night motion without VIO: what exists, what is closed, and the one real lead

> ## 🚨 §1'S PREMISE IS REFUTED — read `docs/toward-tracks-in-place-rotation-20260812.md`
>
> This document says `toward` is course-over-ground and therefore "blind to
> in-place rotation **at any hour of the day**", and builds the whole arc case on
> it. **Measured 2026-08-12 in full darkness: a pure in-place pivot moved
> `toward` by +99.55° with 3.8 cm of travel, and a reverse pivot by −61.43° with
> 3.0 cm.** `toward` tracks rotation. The night path may therefore be a
> legacy-style turn rather than an arc controller.
>
> §2's closures (IR, ultrasonic, RTK yaw, no vendor go-to-point) still stand, and
> arcs are real and measured (`docs/arcs-work-20260812.md`) — but §3's claim that
> an arc is "the only route" does not.

**2026-08-11, off-mower.** Written after the mower docked itself in full darkness
in ~65 s over ~12 m, which prompted the question: what motion is available to us
that does not depend on VIO?

Everything below is read from the code or measured live. Where something is a
hypothesis it says so.

## 1. The VIO gate is narrower than the docs have implied

`vio_active` is appended **only** when `turn_mode == "vio"`
([`services.py:10965`](../custom_components/mammotion/services.py#L10965)).
`turn_mode="legacy"` never creates it. Earlier notes describing the gate as
blocking "the vector executor, even with zero turns planned" are correct in
practice — `vio` is the default and the gate does not care whether a turn is
needed — but the gate is keyed to the *mode*, not to the executor.

**So why is `legacy` still not a night path?** Because of something sharper than
darkness. The legacy turn closes on `position.toward`
([`services.py:10036-10038`](../custom_components/mammotion/services.py#L10036)),
which is **course-over-ground**. A mower rotating in place does not translate, so
`toward` does not change, so the loop is blind to the rotation **at any hour of
the day**. Legacy was never a daylight-only turn primitive that happens to fail
at night; it cannot pivot, full stop.

🔑 **The real constraint is not "no heading at night". It is "no heading while
stationary."** That distinction is what opens §3.

## 2. What else was checked, and closed

### 🗑️ IR — CLOSED. Do not re-investigate.

The operator watched the dock return and reported the terminal phase: the mower
approached on RTK, then **turned around and used IR on its rear against IR on the
dock** for the last stretch. That is real and it is how the machine behaves.

**None of it is exposed.** Searched the integration and the whole of
pymammotion's generated protobuf for `infrared`, `ir_signal`, `ir_status`,
`ir_data`, `photoelectric`, `beacon`, `dock_sensor`, `pile_signal` — **zero
matches**. There is no IR telemetry to read and no command to invoke IR-guided
approach at anything other than the dock. The terminal docking controller is
firmware-internal.

This is recorded as closed for the same reason `base_moved` was: it looks
promising enough to be rediscovered and re-investigated in six months.

### Ultrasonic sensors are self-check states, not distances

The four `ultrasonic_*_status` entities
([`sensor.py:289-325`](../custom_components/mammotion/sensor.py#L289)) resolve
through `SensorCheckState` — pass/fail enums off `report_data.dev.ult_*`. They
carry no range. Consistent with the 2026-07-21/22 finding that these fields are a
pre-job self-check and never carry live obstacle data. **We have no local
proximity sensor of any kind.**

### RTK yaw does not exist on this hardware

`location.RTK.yaw` is read at
[`services.py:6610`](../custom_components/mammotion/services.py#L6610) and
returns `None` on this device. No dual-antenna heading. Dead end.

### There is no vendor "go to point" command

pymammotion's navigation surface is job-, plan- and dock-shaped. Nothing accepts
an arbitrary target. The vendor's night-capable navigation is reachable only
through `todev_rechgcmd` (return to dock), job start, and
`break_point_anywhere_continue`.

### Two vendor primitives we are not using

- **`one_touch_leave_pile()`** — a night-capable undock. Every session so far has
  undocked by hand.
- **`chargePileType {toward, x, y}`** via `toapp_chgpileto` — the dock's pose in
  the map frame. ⚠️ **The integration never reads it, and whether this hardware
  populates it is UNVERIFIED.** Probe before designing anything around it; this
  project has already lost time to `score_info`, which exists in the proto and is
  permanently `null` here.

## 3. 🔑 The lead: we have never sent an arc

Every motion command this project has ever issued is **pure-axis**. All 55
`send_movement` call sites are either `linear_speed=N, angular_speed=0` or
`linear_speed=0, angular_speed=N`. The executor's whole model is *pivot, then
drive straight*.

The wire command has always accepted both at once:

```python
# pymammotion/mammotion/commands/messages/driver.py:117
def send_movement(self, linear_speed: int, angular_speed: int) -> bytes:
    ...DrvMotionCtrl(set_linear_speed=linear_speed, set_angular_speed=angular_speed)
```

**An arc keeps the mower translating, and translation is exactly what makes
`toward` live.** That closes a heading loop with no VIO at all: RTK gives
position, course-over-ground gives heading, and steering corrects both at once.

The heading source is good. Across all seven calibration drives on record, the
mirror of `toward` predicted the drive's own measured facing to within **2.738°**
worst case, and on 2026-08-11 it matched a post-realignment facing to **0.19°**.

### Why this is more than a guess

§2's IR finding tells us the vendor's architecture, and it is two-tier:

| phase | sensor | range | works in dark |
| --- | --- | --- | --- |
| 1 — approach | RTK + course-over-ground | metres | **yes** |
| 2 — terminal | IR, mower rear ↔ dock | last ~1 m | yes |

**Phase 1 is the arc controller described above.** We are not speculating about
whether RTK-plus-heading can navigate this yard in the dark; we watched it do 12 m
in 65 s. What we lack is Phase 2, and Phase 2 is closed to us.

### Which matters much less than it sounds

Docking needs centimetre precision *and* orientation, because two connectors have
to mate. Our goal is to stop within **0.15 m** of a clicked point with the blades
off. **That is a Phase-1-only problem.** The IR tier exists for a requirement we
do not have.

It also explains the landing wall honestly: the vendor does not beat the ~1031 ms
stale position feed with better RTK, it beats it by **switching sensors** for the
endgame. We cannot switch. So ~0.15 m is close to the floor for any approach we
can build — which is the tolerance we already adopted on independent evidence.

## 4. What is NOT established

- **No arc has ever been sent by this project.** Everything in §3 is mechanism
  plus one observation of the vendor doing it, not our own measurement.
- **`toward` latency during an arc is unmeasured.** The position feed is ~1031 ms;
  whether course-over-ground lags it, leads it, or is noisier while turning is
  unknown, and a heading loop's stability depends on exactly that.
- ⚠️ **The mirror relation is validated on FORWARD travel only.** Docked at 20:45
  the mower read `toward` 179.99 → mirror 270.1°, having driven +y (~90°) to get
  there — consistent with **backing into the dock**, which would flip the relation
  by 180°. A controller that arcs in reverse, or that reads `toward` after a
  reverse manoeuvre, needs this settled first.
- **An arc cannot pivot.** Tight geometry, and any final approach needing a large
  heading change over a short distance, still has no answer at night.
- **No existing service can send a refreshed arc.**
  `raw_pymammotion_motion_probe` takes both axes at ±1000
  ([`services.py:555-580`](../custom_components/mammotion/services.py#L555)) but
  has **no** `motion_refresh_interval_ms`, so the h-watchdog caps a single shot at
  roughly 10 cm. `manual_velocity_pulse_test` has the refresh but takes a
  one-axis `action` enum (`forward`/`backward`/`turn_left`/`turn_right`).

## 5. Cheapest path to an answer

**Do this in DAYLIGHT even though the goal is night capability.** VIO is the only
independent ground truth we have for heading; building a night-only controller
with nothing to check it against is how you ship something plausible and quietly
wrong. Validate against VIO in the light, then run it in the dark.

1. **Arc existence check — zero code.** `raw_pymammotion_motion_probe` with
   `linear_speed=400, angular_speed=180`, single shot. Proves both axes actuate
   together and gives a first curvature. ~10 cm of travel, so it is an existence
   proof, not a measurement.
2. **Add `motion_refresh_interval_ms` to that probe.** One schema key and one
   pass-through. Then arcs are real: measure turning radius against angular speed,
   and — the actual question — whether `toward` tracks the arc closely enough to
   close a loop on.
3. **`arc_to_point`**, if and only if step 2's heading tracking holds up.

### What would kill it

If `toward` during an arc lags or noises out beyond ~5°, the loop cannot close and
the whole direction is dead. **Step 2 is the decision point**, and it is cheap.
Measure that before writing any controller.
