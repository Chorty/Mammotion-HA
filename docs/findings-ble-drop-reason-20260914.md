# FINDINGS — why the mower's BLE link goes silent (reason codes captured)

2026-09-14, 01:01–02:15Z. Evidence: `docs/evidence-ble-drop-reason-20260914.json`
(every connection-state event, reconnect proxy rankings, raw log lines).

Question from the 2026-09-13 Phase 1 session
(`docs/findings-phase1-queue-measurement-20260913.md` §5): were the link drops the
mower, ESPHome, or the ESPHome proxies?

---

## 0. Answer

**Connection timeouts on a marginal radio path — not the mower hanging up, not an
ESPHome or proxy fault, not the integration.** Every captured drop carried
**error 8, `ESP_GATT_CONN_TIMEOUT`** ("Connection failed due to timeout"): the
proxy stopped hearing the mower for the connection's supervision interval. n = 4,
one location, one night.

---

## 1. Method

- HA logger `bleak_esphome` set to `debug` with `logger.set_level` at 01:01:33Z and
  back to `info` at 02:15:40Z. **No proxy was reflashed or restarted**, so the
  frozen RF set was untouched.
- The operator mowed, cancelled, sent the mower toward the dock and cancelled
  docking, leaving it **idle at `map_xy (5.74, −9.22)`**, 12.6 m from the dock in the
  southern weak zone, RTK `Fix`, `lawn_mower: paused` from 01:16:26Z.
- It stayed put: no position change on the device tracker (verified through
  01:28:56Z) and `device_position_type` unchanged through 02:05Z. The operator sent
  it home at ~02:14Z; docked and charging 02:15:01Z.
- The error codes come from `bleak_esphome`'s own table, read from the HA container:
  8 = connection timeout, 19 = terminated by peer (mower), 22 = terminated by local
  host (proxy/HA), 62 = failed to establish.

---

## 2. What happened

| drop (Z) | proxy | connected before | code | reconnect (Z) | down |
| --- | --- | --- | --- | --- | --- |
| 01:19:33.673 | `p1s-printer` | since mowing (≥ 3 min idle) | **8** | 01:22:16.794 | 163 s |
| 01:28:07.084 | `p1s-printer` | 350 s | **8** | 01:32:21.131 | 254 s |
| 01:43:54.853 | `hot-tub-backyard` | 694 s | **8** | 01:47:25.459 | 211 s |
| 02:05:24.089 | `hot-tub-backyard` | 1079 s | **8** | 02:07:28.219 | 124 s |

No drop during the drive back to the dock. No code 19 or 22 at any point.

**Best path HA ranked at each reconnect**, mower stationary throughout:

| reconnect (Z) | ranking (dBm) |
| --- | --- |
| 01:22:16 | p1s-printer −76, hot-tub-backyard −78, atom-fireplace −94, garage-m5stack −97 |
| 01:32:20 | hot-tub-backyard −80, p1s-printer −81, atom-fireplace −96, garage-m5stack −96 |
| 01:47:22 | hot-tub-backyard −78, p1s-printer −80, atom-fireplace −95, garage-m5stack −95 |
| 02:07:26 | hot-tub-backyard −78, garage-m5stack −91, p1s-printer −92, atom-fireplace −94 |

The best path was −76 to −80 dBm, at or below the documented ~−76 wall, and
p1s-printer's path swung from −76 to −92 with nothing moving.

---

## 3. Why each suspect is or is not it

| suspect | verdict | evidence |
| --- | --- | --- |
| The integration | **not it** | The warning comes from bleak's unexpected-disconnect callback, and the heartbeat give-up path (30 failures) logs its own warning first — none appeared. |
| An ESPHome proxy fault | **not it** | Drops on two different proxies. On `hot-tub-backyard` a Schlage lock (`SCH130/230`, `5C:02:72:9E:1B:32`) **stayed allocated through both mower drops** (slot log `allocated=[101165582654258]` = its MAC) and **timed out on its own at 01:54:15.842Z while the mower stayed connected**. A proxy reset would drop both at once. |
| The mower hanging up | **not seen** | A deliberate close is code 19; 0 of 4 drops. |
| The radio path | **the failure** | Code 8 on a path ranked −76 to −80 dBm and fluctuating. A mower radio fading without a clean close would also read as 8, so that cannot be fully excluded, but signal margin is by far the likelier explanation. |

---

## 4. The costly part is the reconnect, not the drop

The lock's own code-8 timeout recovered in **0.8 s** (01:54:15.842 → 01:54:16.647Z).
The mower took **124–254 s every time.** This fits the mower advertising only
about once per ~10 minutes, and not at all while connected: HA cannot reconnect
until it hears one, so a brief radio hiccup becomes minutes without control.

Two separate fixes follow:

1. **Fewer drops:** more signal margin in the southern zone (a proxy placed nearer
   it — the two disabled proxies are candidates, after Phase 1).
2. **Faster recovery:** set by the mower's advertising; no code change here can
   shorten it, which is another reason to keep critical work out of weak coverage.

---

## 5. Corrections and notes

- ✏️ **The integration's drop warning is not a reliable drop detector.**
  `BLETransport: device Luba-VSPLV397 disconnected` appeared for **1 of 4** drops
  (01:43:54Z). The `bleak_esphome` debug line is the reliable record.
- The per-20 s connection-allocation snapshots showed a false "no link" at 01:19:09Z,
  24 s before the logged disconnect; the debug log is authoritative.
- Connection length before a drop grew across the night (350 → 694 → 1079 s). With
  n = 3 complete cycles that is an observation, not a trend.
- The mower raised fault **1425** (3D vision module) at 00:55:45Z during the
  preceding night mow, re-stamped 01:04:27Z; unrelated to BLE, recorded for context.
