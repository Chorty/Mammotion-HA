# FINDINGS — BLE proxy-adjacency test, 2026-09-14/15 (overnight into evening)

Follow-on to `docs/findings-phase1-repeat-20260914.md` §6.3, which left open
whether `hot-tub-backyard` sitting physically adjacent to `p1s-printer` (no
apparent path diversity) explains the weak dock-side BLE link. Predeclared in
`docs/predeclared-ble-proxy-adjacency-test-20260915.md` before Window A's first
sample. **No motion, no VIO, no gate arming at any point** — the mower sat on
the dock, RTK `fix`, blade off, for the entire investigation.

---

## 0. Verdict

**The original hypothesis did not hold up, and removing a proxy made things
worse, not better.** Five passive observation windows plus two firmware
changes:

| window | config | duration | drops |
| --- | --- | --- | --- |
| A | 3 proxies (`hot-tub-backyard` powered off) | 20 min | 0 |
| B | 5 proxies (all four connectable) | 20 min | 2 |
| C | 5 proxies (same as B) | 45 min | 0 |
| D | 3 proxies (`p1s-printer` BLE disabled in YAML) | 45 min | 7 (severe) |
| E | 4 proxies, both devices reflashed with less WiFi chatter | 20 min | 0 |

**No axis of this investigation supports removing a proxy to fix the link.**
Window D — the cleanest isolation of "does removing a specific proxy help" —
produced the worst result of the whole investigation: 7 drop/reconnect cycles
in 45 minutes, ~80% of the window disconnected, and two reproductions of the
`error=8` "CODE 8 CONNECTION TIMEOUT" signature from
`docs/findings-ble-drop-reason-20260914.md`.

---

## 1. Windows A–C: the original adjacency question, inconclusive by itself

Session 1 (`docs/predeclared-ble-proxy-adjacency-test-20260915.md`) compared A
(3 proxies) against B (5 proxies): A held clean for 20 minutes, B saw 2 drops
in the same span. Read alone this favors the "remove the proxy" hypothesis —
but Window C, run immediately after B under the **identical 5-proxy
configuration**, went the full 45 minutes with zero drops. Two runs of the same
configuration giving 2-drops-in-20-min and 0-drops-in-45-min means the outcome
is dominated by run-to-run variance, not a stable property of having 5 proxies
versus 4. **A and B/C together do not separate "proxy count" from "which
20-minute stretch you happened to sample."**

Evidence: `docs/evidence-ble-proxy-adjacency-test-20260915/window_{A,B,C}.json`.

---

## 2. Window D: removing `p1s-printer` reproduced the real failure mode

**Design:** the mirror of Window A — keep `hot-tub-backyard`, disable
`p1s-printer`'s BLE (its `esp32_ble_tracker`/`bluetooth_proxy` blocks commented
out in its YAML and reflashed), to test whether `p1s-printer` itself, not
proxy adjacency, was the variable that mattered.

**Result:** 7 connect/disconnect cycles in 45 minutes. Each connection lasted
38 seconds to under 2 minutes before dropping; gaps between drop and reconnect
ran 3–5 minutes. Roughly 80% of the window was spent disconnected.
`garage-m5stack` served nearly every connection, `atom-fireplace` served one,
and **`hot-tub-backyard` — despite being present with historically the
strongest signal (−48 to −58 dBm all night) — served zero connections and
never once appears in the log alongside the mower's MAC.**

### 2.1 Why `hot-tub-backyard` never got a turn

The mower advertises only while disconnected, roughly once per ~10 minutes (a
mechanism this project has documented before). With `p1s-printer` removed,
whichever of the two remaining proxies happens to be listening at that rare
moment is the only one HA can offer as a reconnect candidate — it is not
picking the objectively strongest path each time, it is picking from whoever
overheard the last advertisement. Every reconnect attempt in Window D's log
only ever listed `atom-fireplace` and `garage-m5stack` as candidates
(−84 to −91 dBm), never `hot-tub-backyard`, confirming it simply never caught
one during this window despite sitting idle with 3/3 free slots the whole
time.

### 2.2 Two real `error=8` reproductions

Two disconnects during Window D carry the exact `Connection state changed to
connected=False mtu=0 error=8` signature documented as "CODE 8 CONNECTION
TIMEOUT" in `docs/findings-ble-drop-reason-20260914.md` — both on
`garage-m5stack`, both on connections that had only just been established
(9 s and ~2 min after connecting respectively). This is the first
reproduction of that exact signature outside the original finding, and it
happened specifically once the mower was forced onto the weaker fallback
paths.

Evidence: `docs/evidence-ble-proxy-adjacency-test-20260915/window_D.json`,
`docs/evidence-ble-proxy-adjacency-test-20260915/p1s-printer.yaml.final` (as
flashed at the time, BLE disabled — see §4 for the restored version).

---

## 3. A switch was built, tested, and confirmed NOT to do what was needed

On the operator's request, `esp32_ble_tracker`/`bluetooth_proxy` blocks were
given an `id:` and a template switch (`Bluetooth Scanning`) was added to both
`hot-tub-backyard.yaml` and `p1s-printer.yaml`, calling
`esp32_ble_tracker.start_scan`/`stop_scan`. Both devices were flashed and the
switch verified present (`switch.hot_tub_backyard_hot_tub_backyard_bluetooth_scanning`,
`switch.scooter_p1s_printer_new_bluetooth_scanning`).

**Tested live on `p1s-printer` while it held the mower's active connection:**
turning the switch off did not remove `p1s-printer` from HA's scanner list,
did not change its connection-slot allocation, and did not disturb the
mower's existing connection at all, even 30+ seconds later. `stop_scan` only
pauses advertisement scanning; it has no effect on `bluetooth_proxy`'s ability
to serve or hold GATT connections, which is a compile-time (`active: true`)
setting with no discovered runtime toggle in stock ESPHome. **The switch is a
legitimate scanning control but cannot substitute for physically
removing/reflashing a proxy in a future test.**

---

## 4. A real fix attempt: reducing WiFi chatter on both proxies

### 4.1 The hypothesis

Both `hot-tub-backyard` and `p1s-printer` are original ESP32 boards
(`esp-wrover-kit`/`esp32`), which share a single 2.4 GHz radio between WiFi and
Bluetooth via a time-multiplexed coexistence scheduler — a documented ESP32
characteristic, not speculation about whether the mechanism exists. Both
devices' YAML pushed LUX and HLW8012 power-monitor (current/voltage/watts)
readings to Home Assistant over WiFi every 5 seconds, continuously. Concrete
supporting evidence from the same night: `p1s-printer`'s own OTA upload (pure
WiFi traffic) failed twice with `Connection reset by peer` before succeeding
on a third attempt — direct evidence its WiFi channel was under real stress.
The working hypothesis: this continuous WiFi cadence competes with the shared
radio's ability to listen for the mower's brief, rare BLE advertisement.

### 4.2 The change

`update_interval: 5s` → `60s` on the LUX and HLW8012 sensors, on both devices
(matching the cadence their own `wifi_signal` sensor already used). Not a
disable — basic monitoring is retained at a coarser interval. Diffs banked in
`docs/evidence-ble-proxy-adjacency-test-20260915/*.yaml.final`.

At the same session, `p1s-printer`'s BLE was restored (the operator had
disabled it for Window D by commenting out its `esp32_ble_tracker`/
`bluetooth_proxy` blocks and the switch's corresponding actions) — all
uncommented back to the working state before this flash.

### 4.3 Window E: clean, but not proof the fix worked

20 minutes, all four proxies present, both devices on the new firmware: zero
drops, connection held the whole time on `atom-fireplace`. 🔑 **This is not
strong evidence the WiFi-chatter reduction helped** — Window C already showed
the system can go 45 clean minutes on the *old* firmware under the same
5-proxy configuration, so one clean 20-minute window doesn't distinguish
"the fix worked" from "this was just a good stretch," the same problem that
made A vs. B/C inconclusive. What Window E does establish: the flash
introduced no regression.

Evidence: `docs/evidence-ble-proxy-adjacency-test-20260915/window_E.json`.

---

## 5. What is next (decisions, not actions)

1. **Both proxies are now running new firmware** (switch + slower sensor
   cadence) as of this session. Any future BLE observation is against this
   new baseline, not the one sessions 1–3 of the Phase 1 repeat ran under.
2. **The radio-contention hypothesis (§4.1) is plausible and partially
   evidenced (the OTA failure) but not confirmed as the fix.** A real test
   needs a longer run, or a controlled comparison (one device reverted to 5s
   as a control while the other stays at 60s), not a single 20-minute window.
3. **`hot-tub-backyard`'s advertisement-catching problem (§2.1) is a bigger,
   more general finding than the original adjacency question.** Any proxy can
   sit idle and useless for reconnects purely because of when it happens to be
   listening relative to the mower's ~10-minute advertisement cadence. This
   argues for keeping *more* proxies in the pool (more listeners = more
   chances to catch the rare advertisement), which is the opposite of the
   hypothesis this whole investigation started from.
4. **The `error=8` reproduction in Window D is worth folding into a future
   queue-timing/BLE investigation** — it happened specifically when the mower
   was forced onto weak fallback paths, consistent with (not proof of) a
   signal-strength-driven timeout, separate from the proxy-count question.
5. ⚠️ **Battery observed `not_charging` at 54% while docked, continuously
   from ~00:24Z through the end of this session (~22:54Z on 2026-09-15)** —
   flagged here, not investigated; check before the next hardware session.
