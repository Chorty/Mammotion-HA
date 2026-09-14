# PRE-DISPATCH NOTE — Phase 1 repeat, SESSION 2, 2026-09-14

**Written ~21:55Z, before session 2's first dispatch.** Rules unchanged:
`docs/predeclared-queue-timeout-measurement-repeat-20260913.md` §2–§3 and every
threshold, scored with `scripts/score_queue_measurement.py` as committed
(`3cd3b22d`). Session 1's note: `docs/predispatch-note-phase1-repeat-20260914.md`.

## 1. Session 1 is closed: INCONCLUSIVE

20:59–21:41Z. Dispatched S1–S6c (S6b, S7, S7b sent nothing), 219 samples, all
`completed`, zero `queue_start_timeout`. Truncated before S12 by **transport loss**
(§4.7): at the anchor the best path HA ranked was −76 to −80 dBm, and the link
dropped within minutes whenever the mower sat idle waiting for a per-leg go.
Committed scorer output: 38 `pulse_open` (FAIL ≥ 40), S6 unreconcilable (link lost
mid-leg), truncated ⇒ **INCONCLUSIVE (axis 1)**. Unclassifiable 2/40 — the §2
classifier held. Evidence: `docs/evidence-phase1-repeat-20260914-session1/`.
🛑 **Session 1's samples are never pooled with session 2's**, and session 1 is not
rescored.

## 2. What session 2 changes — protocol only (§4), on operator instruction

- 🔑 **Standing operator go for the whole session** (replaces §4.4's per-leg go).
  The operator asked for the legs to run without stopping for approval, because
  the approval wait was the idle time in which the link dropped. The operator stays
  with the mower. Each leg still runs `scripts/phase1_leg_runner.py` with every
  §16.3 check, arms immediately before and **disarms unconditionally after**, and
  the gate state is verified off after every leg.
- **Automatic same-target retries** (labelled b, c, …; ≤ 6 attempts per target),
  only when nothing was sent: `ble_client_not_connected`, or a runner VIO
  tracked-feature halt while the sun is ≥ 10° (after a 60 s wait). **Any other halt
  stops the session** and the operator decides.
- Dispatch waits for the mower's BLE connection allocation to be present and still
  present 15 s later (session 1's S7b dispatched 1 s after an allocation appeared
  and failed).
- A latched blade RPM register is cleared, if present, with the read-only
  `report_stream_probe` (sends no motion; it cleared the latch in session 1).

## 3. Carried from session 1's note

- Battery below 80 % by operator waiver (37 % at 21:49Z).
- Timing history cleared by a config-entry reload at 21:50:11Z, after banking
  session 1's 219 samples. Host remains beta104; no proxy, profile or constant
  changed.

## 4. Start

Mower at the anchor `(4.872, −3.854)`, `AREA_INSIDE`, RTK `Fix`, facing north
(92°). S1 (south) therefore opens with a ~180° turn-around. Sun 23° at 21:49Z,
≥ 10° until ~23:20Z. Scanner list recorded by the sequencer at start and end;
session 1 ended with the same five scanners it started with.

**Session 2's first dispatch is its `session_start`.**
