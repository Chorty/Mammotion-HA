# PREDECLARATION — BLE proxy-adjacency test (passive, no motion), 2026-09-15

**Written ~00:26Z, before either window's data exists.** Answers the question
`docs/findings-phase1-repeat-20260914.md` §6.3 left open: does removing
`hot-tub-backyard` (adjacent to `p1s-printer`, no apparent path diversity)
actually reduce BLE drops, independent of the mower's position? Session 3's
data couldn't separate that from "the mower had already settled."

## Design

**No motion, no VIO, no gate arming, no dispatch of any kind.** The mower stays
on the dock, exactly where it already is, for both windows — position and
charge state held constant. Only the proxy configuration changes between
windows, in sequence (they cannot run simultaneously, so time-of-night is a
known, stated confound — see Limitations).

- **Window A (now):** current configuration — `hci0` (scan-only),
  `p1s-printer`, `garage-m5stack`, `atom-fireplace`. `hot-tub-backyard`
  physically off.
- **Window B (after A):** `hot-tub-backyard` physically powered back on by the
  operator, otherwise identical.
- **Duration:** 20 minutes each, back-to-back.
- **Instrument:** `bleak_esphome` and `habluetooth` HA loggers set to `debug`
  via `logger.set_level` (no restart, no proxy change of its own) — the
  documented reliable record for real disconnects
  (`docs/findings-ble-drop-reason-20260914.md`). Cross-checked every 30 s
  against the live Bluetooth connection-allocation list for the mower's MAC
  (`A8:B5:8E:2C:52:40`), the same mechanism `scripts`'s `ws_bt.py` helper uses.

## Metric and comparison, fixed now

For each window: **count of genuine disconnect events** — a
`bleak_esphome` debug line reading `Connection state changed to
connected=False` for the mower's MAC, or a 30 s poll finding the MAC absent
from every proxy's allocation list where the previous poll found it present.
Also recorded per window: which proxy the connection lands on at window start
and end, and every RSSI reading logged for it.

**Read as informative, not a formal pass/fail bar** (n=1 window per
configuration is too small for a rate claim): fewer disconnects in B than A is
weak evidence hot-tub-backyard helped; equal counts is evidence proxy count
doesn't matter here; more disconnects in B is evidence it actively hurts
(interference) rather than merely being redundant.

## Limitations, stated before the fact

- **Time-of-night is NOT controlled.** Window B necessarily runs ~20+ minutes
  later than Window A. If BLE quality drifts with time (band congestion,
  temperature, anything) rather than being stable across the whole session,
  that drift is indistinguishable from a proxy effect here. A future, more
  rigorous version would interleave short windows (A/B/A/B) rather than one
  block each.
- **n=1 per configuration.** This is a single overnight observation, not a
  statistical sample. It can suggest a direction; it cannot establish a rate.
- **Position is the dock, not the anchor.** The 2026-09-14 findings' drops were
  captured with the mower off-dock near the anchor. The dock may have
  different RF characteristics (proximity to the house, different multipath).
  This test answers "does removing the proxy help at the dock", which is a
  real question but not identical to "does it help at the anchor".

## Closure

Once Window A's first sample exists, this document is closed to amendment.
