# PRE-DISPATCH NOTE — Phase 1 repeat, session 2026-09-14

**Written 2026-09-14 ~19:45Z, before the repeat's first dispatch. No repeat sample
exists.** Governs nothing new: it records three departures from the session
protocol in `docs/predeclared-queue-timeout-measurement-repeat-20260913.md` §4 and
why, so a later reader can judge them. §2 (classifier), §3 (RF) and every
threshold are untouched, and the scorer is used as committed (`3cd3b22d`).

## 1. Battery below §4.1's 80 % — operator waiver

Pre-flight read **48 %**, `charge_state: not_charging` on the dock since ~16:45Z
(after a 14:46–15:33Z mow and a 19 → 48 % charge). The operator waived the 80 %
floor explicitly ("48% on the battery is fine"). Consequence, unchanged from §4.7:
if battery ends the session before S12 is dispatched, the session is
**INCONCLUSIVE**.

## 2. Start ~3.6 m from the anchor, not ~1 m (§4.3)

The operator moved the mower off the dock before the session; it parked at
`map_xy (4.63, −0.22)`, `AREA_INSIDE`, zone `1343645155037768237`, RTK `Fix`,
facing south (`map_facing` 275.8°, `motion_confirmed`; last driven leg 277.2° and
compass mirror 275.9° agree to 1.3°). **S1 is dispatched from there** — a ~4.6 m
leg straight ahead, inside the 6.1 m segment cap, target unchanged. No setup leg is
run. S1 therefore carries more pulses than a ~1 m S1 would; every target and band
is unchanged, and the runner's corridor, excursion and band checks apply as
written.

## 3. Timing history cleared by an integration reload before the first dispatch

The coordinator's `motion_dispatch_timings` deque held **470 / 500** samples, all
from 2026-09-12/13. The committed scorer sets `history_dropped` when **any**
baseline sample disappears between snapshots, so the first leg evicting the
oldest pre-session samples would fail axis 1's capacity clause without a single
session sample being lost. No service clears the deque; a config-entry reload does
(it is in-memory). That is **not a deploy** — host code stays beta104, and no
proxy, profile or constant changes.

- The 470 samples were snapshotted first:
  `docs/evidence-timing-history-pre-reload-20260914.json`. 404 of them were
  already banked (339 in the 2026-09-13 evidence, 65 in the 2026-09-12 setup-leg
  evidence); the rest are the second 2026-09-12 setup leg.
- ⚠️ With an empty baseline the capacity clause still fails if the session itself
  exceeds 500 samples. 2026-09-13 produced 339 over the same 12 targets.

## 4. Session start record

- Scanners at ~19:41Z: `hci0` (scan-only), `hot-tub-backyard`, `p1s-printer`,
  `garage-m5stack`, `atom-fireplace` — identical to 2026-09-13. Two newly added
  proxies remain disabled.
- Host beta104, 68 mammotion services, `motion_dispatch_timing_report` present,
  gate disarmed. Sun ≥ 10° until ~23:20Z.
