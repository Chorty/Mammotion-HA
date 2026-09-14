# PRE-DISPATCH NOTE — Phase 1 repeat, SESSION 3, 2026-09-14 (dusk retest)

**Written ~23:20Z, before session 3's first dispatch.** Extends sessions 1–2's
notes (`docs/predispatch-note-phase1-repeat-20260914.md`,
`docs/predispatch-note-phase1-repeat-session2-20260914.md`); rules unchanged
otherwise: `docs/predeclared-queue-timeout-measurement-repeat-20260913.md` §2,
scored with `scripts/score_queue_measurement.py` as committed (`3cd3b22d`).
Sessions 1–2 findings: `docs/findings-phase1-repeat-20260914.md`.

## 1. RF configuration change — NEW, not comparable to sessions 1–2

The operator physically powered off `hot-tub-backyard` at ~22:39Z (confirmed
`unavailable` in HA at 22:39:27Z and dropped from the Bluetooth scanner list by
23:11Z), because it sits physically adjacent to `p1s-printer` and offered no
real path diversity — raised as an open question in
`docs/findings-phase1-repeat-20260914.md` §5.2. **This is now the test of that
hypothesis.**

- **Active connectable scanners:** `p1s-printer`, `garage-m5stack`,
  `atom-fireplace`. `hci0` remains scan-only. Down from five to four total
  (three connectable, was four).
- 🛑 **Not comparable to the frozen-set sessions.** Session 1/2's RF freeze
  existed for comparability with 2026-09-10's legs 7/8 failure; this session
  breaks that comparability on purpose, per the operator's own instruction. Its
  result stands on its own for the proxy-adjacency question, and separately
  contributes pulse-open samples toward the queue-timing question under the same
  classifier and thresholds (RF composition doesn't bear on those).
- No code, profile, or motion constant changed for the RF question itself. Host
  remains beta104. Per operator instruction, the two other pending code changes
  (BLE keep-alive/dynamics-poller port, services.yaml turn-field fix) are **NOT**
  deployed alongside this test, to keep it a single-variable retest of the proxy
  change alone.

## 2. Daylight guard — explicit operator override, sun-elevation clause ONLY

Sun elevation at session start is **below the runner's 10° floor** (≈5.6° at
23:16Z; 0° at 23:43Z). The operator was told the specific precedent this guard
exists for — `docs/findings-clicktopath-reliability-4m-20260904.md` and CLAUDE.md's
"recheck daylight before every armed leg": a 2026-09-12 setup leg armed ~6 min
after sunset, VIO collapsed (`visual_positioning_status` → `signal_none` 11s
before the halt while `vio_brightness`/`camera_brightness` still read good), and
the leg overshot on final approach. The operator explicitly chose to override
tonight, understanding that risk.

**What changed in code** (`scripts/phase1_leg_runner.py`,
`daylight_vio_verdict` + `main`): a narrow `--allow-low-sun` flag that skips
**only** the sun-elevation clause. It does **not** relax:

- `vio_tracked_features` minimum ≥ 70 over the trailing 60s window — this,
  not the point-in-time status read, is what actually caught the 2026-09-12
  collapse (14 vs 80 in that leg's window).
- `visual_positioning_status` must read `signal_good` throughout the same
  window.

Every other §16.3 check (position/zone/RTK/blade/facing/corridor/excursion/
band/dry-run/stop-reason) is unchanged. A halt on either VIO clause still stops
the leg — the override cannot be worked around by the sequencer's automatic
retry logic (that logic only retries `ble_client_not_connected` with nothing
sent, or a VIO dip while the sun is ≥10°; a VIO dip tonight is a hard stop).

🔑 **Given the elevated conditions (dusk + guard override), tonight reverts to a
per-leg operator go, not sessions 1–2's standing go for the whole run.** Each leg
is announced and confirmed before it dispatches.

## 3. Pre-flight, recorded

- RTK `Fix`; camera cleaned earlier today (fault 1068, confirmed clear all
  session); gate off; battery 53% and was charging on the dock.
- Blade register: 0 rpm, no latch, at last check (mower still on dock;
  re-verified once it is moved to the start point).
- Host beta104, `motion_dispatch_timing_report` present, 0 samples (last cleared
  before session 2; session 2's 363 samples already banked and will not be
  re-cleared for this session — they stay in history unless eviction becomes a
  risk, tracked the same way as before).

## 4. Start

Mower is currently docked. The operator will move it to the §16.2 anchor
vicinity (as in sessions 1–2) before the first dispatch. **Session 3's first
dispatch is its `session_start`** for scoring purposes; its samples are pooled
with no prior session.
