# PREDECLARATION — Phase 1 repeat: stop-delimited classifier, same four proxies

**Written 2026-09-13 (~23:55Z), after the inconclusive session banked in
`0d1ea032`, and before any repeat sample exists.** It changes only what the
2026-09-13 session showed to be broken — the burst classifier — and fixes the RF
configuration and session protocol for the repeat.

Operator decisions taken 2026-09-13: **classifier option A (split at stops)** and
**RF option A (same four proxies, freeze held).**

Parent: `docs/predeclared-queue-timeout-measurement-20260911.md` (called "the
parent" below). Session record: `docs/findings-phase1-queue-measurement-20260913.md`.

---

## 0. Provenance, stated plainly

🔑 **The new classifier was designed with the 2026-09-13 data in view.** That is
legitimate only because **the verdict is computed on the repeat session's samples
alone.** The 2026-09-13 session stays **INCONCLUSIVE**; its samples are never
rescored under this rule and never pooled with the repeat.

---

## 1. Inherited from the parent, UNCHANGED

- §1 instrument properties (budget mixing, `outcomes` is not a failure census,
  `maxlen=500` → snapshot every leg).
- §3 and §4 thresholds **as restated over the `pulse_open` class** (§10.2):
  p95 ≥ 1000 ms / ≤ 250 ms; worst-wait fraction ≥ 0.75 on ≥ 2 legs / ≤ 0.40,
  recomputed by hand over budget-2.0 `pulse_open` samples.
- §12.1 write-inheritance ratio test and its fixed evaluation order.
- §13.1 (`n ≥ 40` `pulse_open` surviving exclusion, from ≥ 4 legs, ≥ 2 with a turn
  or calibration phase) and §13.2's two axes and verdict table.
- §13.3 (a `queue_start_timeout` is signal, not failure).
- §10.4(B): no rate claim below 120 `pulse_open`; the honest bound is `3/n`.
- §11.3 `known_biases` block — required in the evidence file.
- §14.1 bands, §16.2 anchor and targets S1–S12, §16.3 automatic halts.

---

## 2. REPLACED: the classifier (parent §10.3, §11.2, and the meaning of "burst")

### 2.1 Why the parent's rule is withdrawn

The parent's 500 ms gap rule rested on *"within-burst gaps span 0–200 ms, never
more."* On 2026-09-13, **23 gaps inside single pulses were 503–1058 ms**, 15 of
them cutting a pulse-open away from its own refreshes. Any timing threshold is a
guess the next slow write can break, so **no timing gap is used.**

### 2.2 Stops are delimiters, never samples

Every sample with **`is_stop: true` is removed from both classes, whatever its
`queue_budget_seconds`.** Linear and calibration stops record budget 5.0; turn
stops record **budget 2.0 with `emergency_stop: false`** — filtering on budget
alone is wrong.

### 2.3 The rule

For each dispatched leg, take the samples recorded during that leg (the
difference between its timing snapshot and the previous one), sorted by
`recorded_at_utc`:

1. A **segment** is the run of `is_stop: false` samples ending immediately before
   the next `is_stop: true` sample.
2. Segment *k* pairs with **`command_results[k]`**, in the executor's own order.
3. The **first** sample of a segment is `pulse_open`; every later sample in it is
   `refresh`.

### 2.4 Cross-check — independent of the split

The split uses only the stop samples. The check uses only the executor's refresh
count:

✅ **Segment *k* must contain exactly `1 + command_results[k].motion_refresh.refresh_commands_sent`
samples** (the calibration drive has no `motion_refresh`; its count is 0).
A segment that differs is **UNCLASSIFIABLE** and excluded from both classes. The
boundary is never re-drawn by hand.

### 2.5 Misalignment and reconciliation

- If a leg's **segment count ≠ `len(command_results)`**, or **motion samples follow
  its last stop**, every segment of that leg is UNCLASSIFIABLE.
- Parent §11.1 clause 4 ("unreconcilable leg" ⇒ inconclusive) is defined here as:
  **a leg whose motion-sample total ≠ `len(command_results) + Σ refresh_commands_sent`.**

### 2.6 The 20 % clause, with its denominator fixed

**Unclassifiable share = UNCLASSIFIABLE segments ÷ Σ `len(command_results)` over
all dispatched legs.** Above 20 % ⇒ INCONCLUSIVE. The count of excluded segments
and excluded samples is reported alongside `n`, always.

### 2.7 Validation before use, and the rule's own falsifier

Applied to the 2026-09-13 data (design evidence, not a verdict): **67/67 pulses
classified exactly on all 13 dispatched legs**, every stop after its pulse's
refreshes, no motion after a final stop. The 2026-09-12 setup legs record only leg
totals, so they cannot test it.

🚨 **If the repeat shows > 20 % unclassifiable, the session is INCONCLUSIVE and
this rule is recorded as failed** — not adjusted after the fact.

### 2.8 The scorer is committed before the first dispatch

The rule above is implemented as code in `scripts/`, with a test that reproduces
67/67 on `docs/evidence-queue-timeout-measurement-20260913.json` and
`docs/evidence-phase1-legs-20260913.json`, **committed before the repeat's first
dispatch.** Scoring the repeat uses that committed code unmodified.

---

## 3. RF configuration — FROZEN (option A)

- **Active scanners:** `hci0` (scan-only), `hot-tub-backyard`, `p1s-printer`,
  `garage-m5stack`, `atom-fireplace` — the set legs 7 and 8 failed under.
- **The two proxies added 2026-09-13 stay disabled.** No proxy added, moved,
  powered down or re-provisioned.
- **No deploy.** The host stays on beta104 (verified at session start);
  `b26f8909` / `683f2e52` remain held. No change to `motion_refresh_interval_ms`
  or `_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS`.
- 🚨 **The scanner list is recorded at session start and end. Any difference ⇒
  INCONCLUSIVE** (the RF set changed mid-session).

---

## 4. Session protocol — fixed now, from 2026-09-13's lessons

1. **Pre-flight:** RTK `fix`; camera clean; battery ≥ 80 %; gate disarmed;
   beta104 and `motion_dispatch_timing_report` present; scanner list recorded;
   compute the time the sun reaches 10° and plan to finish well before it.
2. **Population:** only samples recorded at or after the session's first dispatch.
   Everything earlier in the history is excluded by timestamp.
3. **Start:** the operator places the mower within ~1 m of the anchor
   `(4.94, −3.82)` (app or a cancelled mow; blades do not run while travelling).
   No setup leg is planned. If one proves necessary, a separate pre-dispatch note
   is committed first and that leg is excluded.
4. **Per leg:** `scripts/phase1_leg_runner.py`; explicit operator go immediately
   before; **arm → run → disarm unconditionally**, with the runner's exit code
   read directly (never through a pipe or `PIPESTATUS`); the runner's per-leg
   timing snapshot kept.
5. **After any leg containing a turn, wait ≥ 60 s before the next dispatch** so the
   runner's VIO look-back does not see that turn.
6. **Halts** (§16.3) stop the leg. The operator may retry the **same target**;
   retries are labelled. Every leg that dispatched commands contributes its
   samples, halted or not. Refusals that sent nothing contribute nothing.
7. **Truncation (parent §13.2, unchanged):** if transport loss, a VIO/daylight
   refusal, containment or battery ends the session before S12 is dispatched ⇒
   **INCONCLUSIVE.**
8. **End:** dock on operator go; gate verified off in the live API **and** RAW
   `core.config_entries`; final timing snapshot; scanner list recorded.

---

## 5. Sizing

The same 12 targets. On 2026-09-13 the dispatched legs produced **67 pulses**, so
`n ≥ 40` `pulse_open` surviving exclusion has headroom if classification holds.
No extra legs are added.

---

## 6. Evidence file

`docs/evidence-queue-timeout-measurement-<date>.json` must carry: `known_biases`;
per leg the segment sizes, expected sizes and classification; excluded segment
and sample counts beside `n`; start and end scanner lists; band, estimate and
live `ble_rssi` per leg; every refusal, halt and link drop; and the full raw
samples.

---

## 7. Closure

Once the repeat's first sample exists, **§2–§6 of this document are closed to
amendment.**
