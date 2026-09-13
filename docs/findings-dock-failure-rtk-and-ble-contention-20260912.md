# FINDINGS — the 2026-09-12 dock failure, the RTK correction path, and BLE contention

Session 2026-09-11 evening EDT / 2026-09-12 early UTC. **No measurement session
ran** — it was dark, so Issue 1 step 2 was not attempted. What this records is
what was found while trying to get the mower back on its dock, plus three
results that bear directly on the work that *is* planned.

Companions: `docs/plan-queue-measurement-then-ble-20260912.md`,
`docs/predeclared-queue-timeout-measurement-20260911.md`.

---

## 0. What actually happened, in order

| time (UTC) | event |
| --- | --- |
| 21:57:23 | mower leaves dock on a brief `mowing` transition, lands at "Backyard Right" |
| 02:11:57 | telemetry settles: `paused`, not charging, RTK `single` |
| 03:05:13 | **fault 1300 logged** — poor positioning |
| 03:18:07 | `lawn_mower.dock` dispatched, HTTP 200 |
| **03:18:08** | **`ble_link_live` → `off`**, one second later |
| 03:21:29 | `ble_link_live` → `on` again, unaided |
| 03:24:08 | mower has **not moved**; 45 %, still off-dock |

Battery over the episode: 69 % at 20:34 → **45 %** at 03:24, ~4.3 %/h.

🔑 **The dock command was accepted and produced no motion.** `command ok` proving
nothing about delivery is already project doctrine; this is another instance.

---

## 1. 🚨 The RTK correction path failed while the mower's own receiver was fine

This is the finding with the longest reach, and it has not been caught in the
act before.

| source | reading |
| --- | --- |
| `sensor.*_rtk_position` | `single` — no fix |
| `sensor.*_position_level` | **0** |
| `sensor.*_last_error_code` | **1300** — "Poor positioning status" |
| `sensor.*_satellites_robot` | **24** |
| `sensor.*_l1_satellites_co_viewing` | 26 |
| `sensor.rtkbna235279309_satellites` | **0** |
| `device_tracker.rtk_backyard` | `not_home` |

🔑 **The mower is tracking two dozen satellites and still cannot reach a fix.**
That separates the two halves cleanly: the mower's GNSS receiver is working, and
what is missing is the *correction* stream. The base station reporting zero
satellites and `not_home` is the obvious suspect.

✅ **Three independent fields agree** — `rtk_position: single`,
`position_level: 0`, and fault `1300` are not one value echoed three ways. That
matters given this project's history with circular checks.

⚠️ **Not proven to be the base station's fault.** The base could be a second
symptom rather than the cause. And 🚨 **`sensor.rtkbna235279309_longitude` reads
`-520.769852361167`**, which is not a possible longitude — at least one field on
that device is mis-parsed, so treat every reading from it with suspicion until
that is chased down.

**Why a dock cannot work in this state.** Docking is navigation, navigation needs
position, and the gate agrees independently: `blockers` now includes
`rtk_not_precise` alongside `experimental_motion_disabled`.

🔑 **This is a live candidate for the night-time `1309` docking failures** in
`docs/findings-clicktopath-reliability-4m-20260904.md` §6.6, which were diagnosed
as orientation. Orientation may still be right for those — but "no RTK fix after
dark" is a mechanism nobody had measured, and it should be checked before the
orientation explanation is treated as settled.

---

## 2. 🚨 A different integration is saturating the BLE proxies

Measured from the HA container log:

- **121 `Found 5 connection path(s)` scans in 60 minutes** — roughly one every
  30 s, each enumerating all five scanners.
- Driven by `custom_components.omron` retrying an unrelated Omron blood-pressure
  monitor (`F4:07:7B:F5:E8:65`), each attempt failing
  `BleakNotFoundError ... Failed to connect after 4 attempt(s): TimeoutError`.
- The retries drive `failures=` counters to **8–9 on every proxy**, including
  `hot-tub-backyard` — the one holding the mower's connection.

⚠️ **This is not the mower's own command queue**, so it cannot by itself trip
`_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS`, which guards pymammotion's
per-device serialized queue. Do not overstate it.

🔑 **But it is sustained contention on the same proxy radio that carries the
mower's GATT writes**, and a write delayed behind that contention is exactly the
shape of thing the queue-start bound would observe. It is the first concrete
candidate "occupant" for the §4 branch of
`docs/predeclared-queue-timeout-measurement-20260911.md` — the branch predicted
as the likely outcome.

✅ **Actions for Phase 1:** record the proxy inventory *and* the Omron retry rate
into the evidence file at session start, and consider disabling that integration
for the duration of the measurement so the queue distribution is not measured
against avoidable background contention.

🚨 **The mammotion integration logged NOTHING.** Zero `mammotion`/`luba` lines in
a 20-minute window spanning both the dock command and the BLE disconnect. A
transport drop the integration itself reports as `ble_link_live: off` left no log
record at all. That is a real observability gap for anything diagnosed from logs
later.

---

## 3. ✅ BLE self-recovered in 3.5 minutes, not ~10

The link dropped at 03:18:08 and returned at 03:21:29 with **no intervention** —
the advertisement callback registered unconditionally in `__init__.py` did its
job.

🔑 **This corrects an estimate made earlier the same session.** Reasoning from
the mower's ~1-advertisement-per-10-minutes rate, re-acquisition was predicted to
cost up to ~10 minutes; measured once, it cost 3.5. ⚠️ **n = 1** — this is one
observation, not a distribution, and the mower was stationary with good RSSI.

**Consequence:** the re-acquisition cost of the idle-BLE-release drain fix
(`docs/plan-queue-measurement-then-ble-20260912.md` phase 3a.3) is lower than the
plan assumed. It does not change the plan's gating — that fix still lands only
after Phase 1 — but it makes the approach more attractive than written.

---

## 4. The drain, confirmed a second time

69 % → 45 % in under seven hours off-dock, **~4.3 %/h**, with `ble_link_live: on`
for nearly all of it. The 2026-09-08 episode measured 5.74 %/h and ran 51 % → 0 %.
Two independent occurrences, same signature, consistent with
`[[ble-connection-blocks-doze-drains-battery]]`.

⚠️ **Still not an experiment.** Neither occurrence tested the counterfactual —
nobody has released the link and watched the drain flatten. That test is worth
designing, but not while the mower is stranded and the link is the only control
path.

---

## 5. ✏️ A correction about the dock command's timing

An early read this session implied fault 1300 appeared alongside the dock
command. **It did not** — 1300 was logged at 03:05:13, thirteen minutes before
the 03:18:07 dispatch, and shares a root with the `rtk_position: single` that was
flagged in the pre-flight. The command did not cause the fault; the fault is why
the command did nothing.

The BLE drop at 03:18:08 *is* one second after dispatch, and that correlation
stands unexplained. With n = 1 it could be coincidence, and no log line supports
either reading (see §2).

---

## 6. State at close (2026-09-12T03:24Z)

Mower **off-dock** at "Backyard Right", `paused`, **not charging**, **45 %**.
`ble_link_live: on` at −58 dBm. `real_motion_ready: off`. RTK `single`,
`position_level: 0`, fault `1300` standing. Dark — `camera_brightness: dark`,
`vio_brightness: 0`, `vio_tracked_features: 0`. Gate **disarmed**, verified from
the live API: `enabled: false`, `real_motion_allowed: false`,
`blockers: ['experimental_motion_disabled', 'rtk_not_precise']`.

🛑 **The dock was attempted once, on explicit operator go, and not retried** — the
condition that defeated it had not changed. Recovery is an operator action:
carry it in, or power-cycle the RTK base and watch
`sensor.rtkbna235279309_satellites` come off zero.

**Nothing was changed in code this session.** No control-law value, no profile
value, no `_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS`, no
`motion_refresh_interval_ms`.

---

## 7. ✏️ CORRECTIONS — §1 partly REFUTED by daylight data (2026-09-12T15:26Z)

Everything recovered without intervention. Mower **docked and charging** since
14:49:53Z, **51 %** and rising, `rtk_position: fix` since 14:55:04Z,
`position_level: 1`, VIO back to 74 features, full daylight (sun 47°).
`device_tracker.rtk_backyard` is `home` again.

🚨 **The base station's satellite count was NOT the mechanism.**
`sensor.rtkbna235279309_satellites` still reads **0**, unchanged since
02:11:58Z — *and the mower reached `fix` anyway*. §1 named the base's zero
satellites as "the obvious suspect"; that is now refuted by direct observation
11.5 h later. Either the field is meaningless on this hardware or it is
unrelated to the correction stream — consistent with the standing note that
corrections arrive **over the internet** and are relayed via LoRa, so the base
does not need its own satellite lock to pass them on.
✅ **§1's hedge was the load-bearing part**: it said "not proven to be the base
station's fault. The base could be a second symptom rather than the cause."
That hedge is what survived. **The correlation was real and the causal reading
was wrong.**

✏️ **The impossible longitude was TRANSIENT, not a standing parse bug.** §1 said
the base's longitude sensor reads `-520.77` and to "distrust its readings until
chased down". It now reads **`-84.7698871238333`** — correct for this location.
🔑 **The two values share their decimals** (`-520.7698523536123` against
`-84.7698871238333`): the integer part was corrupted while the fraction was
intact, so it is the *same* underlying value mangled during the fault window,
not a field that is always wrong. **The right reading is that this device emits
corrupt values while unhealthy** — which makes a garbage longitude a useful
*symptom* of the fault rather than a reason to distrust the device permanently.

🔑 **What actually stands from §1**, and it is still the important part: **the
mower tracked 24 satellites and could not reach a fix**, three independent
fields agreed (`rtk_position: single`, `position_level: 0`, fault `1300`), and
**a mower with no fix cannot dock** — the command was accepted and produced zero
motion. The correction path failed; *why* it failed is still unexplained, and
the base's satellite count is no longer a candidate answer.
⚠️ **It cleared on its own, overnight-to-daylight.** Whether daylight, time, or
the trip back to the dock did it is unknown — do not assume a night-only
pattern from n = 1.

⚠️ **Fault `2709` (low battery) fired at 07:29:02Z** while the mower was still
stranded — the drain reached the device's own low-battery threshold before it got
back on the dock. That is the same code as the 2026-09-08 episode and has the
same benign-once-explained cause.

⚠️ **Still not motion-ready even docked with RTK `fix`:**
`real_motion_ready: off` and the gate now reports
`blockers: ['experimental_motion_disabled', 'position_not_valid_for_motion']` —
note `rtk_not_precise` cleared but a *different* position blocker replaced it.
**Resolve that before planning Phase 1 legs**, rather than assuming a `fix`
reading is sufficient.

---

## 8. 🗑️ §2 RETRACTED — omron is off-path (corrected 2026-09-13)

§2 called `custom_components.omron` "sustained contention on the same proxy
radio that carries the mower's GATT writes" and told Phase 1 to record its rate
and consider disabling it. **Both are withdrawn.**

The **121/h** figure counted habluetooth `Found 5 connection path(s)` lines —
**path enumeration, not connect attempts** — and the `failures=` counters are
accumulated historical score. Every actual omron connect attempt goes
`via source=D8:3A:DD:C3:CE:CD`, the host's built-in `hci0` adapter: 102 of 102
in 6 h, **zero** through any ESP32 proxy. On 2026-09-13 01:58Z the omron cuff is
allocated on `hci0`. The evidence was in §2's own log capture — every omron
connect line named that source — and I read the enumeration lines instead.

🔑 **Phase 1 must not disable omron**; it does not share the mower's radio. The
mower's proxy also moves between reconnects (`hot-tub-backyard` / `p1s-printer`),
so "the proxy holding the mower" was never a fixed thing either.
Full measurement: `docs/findings-ble-write-latency-mechanism-20260912.md` §4.2.

---

## 9. 🚨 Overnight RTK watch, 2026-09-13 — degradation RECURRED after dark, docked

Read from HA history at 2026-09-13T06:44Z (still dark; sunrise 11:20Z).

| time (Z) | signal |
| --- | --- |
| 2026-09-12 14:55 | `rtk_position` → `fix` (previous day's recovery) |
| 2026-09-13 00:14 | mower docked and charging |
| **02:06:25** | **`rtk_position` → `float`** |
| 02:06:32–02:07:50 | **`device_tracker.rtk_backyard` flaps** home ↔ unavailable (6 transitions) |
| 00:01 → 02:16 | mower `satellites_robot` 24 → 17 |
| 06:44 | still `float` — **4.6 h, not recovered** |

🔑 **Second night running that the correction path degraded after dark.** It is
not the same failure: last night went to `single` with `position_level: 0` and
fault `1300` while stranded off-dock; tonight went to `float` with
`position_level` still 1 and no new fault, on the dock.
🔑 **New co-occurring signal: the RTK base dropped off the network for ~80 s at
the exact minute RTK degraded.** `device_tracker.rtk_backyard` is network
presence, and corrections reach the mower over the internet via the base's
relay (standing memory `rtk-corrections-come-from-the-internet`). That makes
**base-station connectivity the leading candidate**, not darkness itself.
⚠️ **n = 2 and not proven.** A night-only pattern and a base-connectivity pattern
are not yet separable, and the base's own `satellites` (28) and `longitude`
sensors have not changed since 2026-09-12 19:54Z, so they are likely stale and
should not be read as live.
✅ **Next:** re-read after sunrise. If `float` persists into daylight, darkness is
ruled out and the base's network link becomes the thing to instrument (its
Wi-Fi RSSI read −73 dBm on 2026-09-12). Phase 1 requires RTK `Fix`, so check
this before the next session.
