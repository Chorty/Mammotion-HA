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
