# FINDINGS — the setup leg, and what it corrected before Phase 1 ran

2026-09-12. **Phase 1 did NOT run.** One unscored setup leg dispatched, to move
the mower into the §14.2 pocket. The session then stopped on its own budget
before the scored legs, deliberately, because a truncated run is **predeclared
inconclusive** and would have bought no verdict while stranding the mower.

Evidence: `docs/evidence-setup-leg-timing-20260912.json`.
Predeclaration: `docs/predeclared-queue-timeout-measurement-20260911.md` (§15
recorded this leg's exclusion **before** it dispatched, while `sample_count`
was 0).

---

## 1. The leg itself — clean

`target_reached`, `path_complete`, landing **0.0506 m** from target.
19/19 accepted-profile keys echoed with **zero mismatches**; 13/13 safety gates
passed; no errors, warnings, keep-out violations or blockers. Start
`(4.4654, −0.5030)` → `(4.9416, −3.8236)`, 3.399 m. Facing recovered to
`motion_confirmed` / `safe_to_aim_dispatch: true` afterwards.

⚠️ It dispatched with `safe_to_aim_dispatch: false` (motion confirmation had
aged out at 808 s to `corroborated_not_motion_confirmed`). Resolved the way the
standing trap prescribes — **the operator confirmed visually** that the mower
pointed away from the dock, the non-circular source. The executor's own VIO
calibration drive was the second.

---

## 2. 🚨 The contamination runs OPPOSITE to what §10.1/§10.2 assumed

| class | n | p50 | p95 |
| --- | --- | --- | --- |
| pulse-open | 11 | 179.939 ms | **432.424 ms** |
| refresh | 44 | **0.456 ms** | 1.382 ms |
| **pooled** | 55 | — | **188.89 ms** |

🔑 **Refreshes wait essentially zero.** They fire 200 ms apart on a fixed cadence
and each write takes ~186 ms, so the queue drains before the next is enqueued.
Pulse-opens wait 180–432 ms because they queue behind the **stop** from the
previous pulse.

✏️ **Both `mammotion-ha-0f` and I reasoned the refresh contamination would
INFLATE the numbers toward §3 ("raise the constant"). It does the reverse** —
pooling **deflates** p95 by ~2.3× (188.89 against 432.424) and masks the
pulse-open tail, biasing toward §4 exoneration.

**What survives from §10.1 unchanged:** the `outcomes` census still conflates a
harmless swallowed refresh timeout with one that killed a leg, and the classes
are still asymmetric in consequence. The defect was real and worth fixing. Only
the predicted *direction* on the distribution was wrong.

---

## 3. ✅ The classifier was validated — and a gap in §10.3 found

🚨 **Stops must be filtered out BEFORE burst-splitting, and §10.3 does not say
so.** Emergency-stop dispatches carry `queue_budget_seconds` 5.0 and are neither
pulse-open nor refresh. Omitting that step gave 12 bursts and a **+8** refresh
disagreement. It follows from §1's budget rule but was not spelled out.

After filtering (55 motion samples of 65): **11 bursts** inferred, against the
executor's own independent **`commands_sent` 10** and
**`motion_refresh_commands_sent` 45**. The total (55) agrees exactly; the split
is off by one, because one pulse-open-to-first-refresh gap exceeded the 500 ms
threshold and split a burst in two.

🛑 **The predeclared rule was NOT changed.** A strictly better method exists —
use `commands_sent` to select the N−1 largest gaps, exact by construction — and
it is deliberately **not adopted**, because `sample_count` is no longer 0 and
§15 closed §2–§14. The predeclared rule handles this correctly on its own terms:
that burst is `UNCLASSIFIABLE` and excluded per §11.2, ~10% against a ≤20 %
tolerance. 🔑 **The better method belongs in a future predeclaration, written
before its data exists.**

---

## 4. 🚨 §10.4's per-leg pulse-open estimate is too optimistic — Phase 1 needs
## MORE legs

Measured: **10 pulse-opens for 3.4 m** (9 linear + 1 calibration, 0 turns).
Pulse-open count scales with distance, so a **1.0 m leg yields only ~4**.

§10.4 estimated 7–10 pulse-opens per 0.8–1.5 m leg, derived from
distance-per-pulse arithmetic rather than measurement. Against this leg that is
roughly **2× too high**.

**Consequence for the plan:** 8 legs of 1.0 m banks **~32–48** pulse-opens
against the `n ≥ 40`-after-exclusions bar — at it or under it.
✅ **Phase 1 should plan ~10–12 legs at 1.0 m, or 1.5 m legs turning back every
2** (same 3.0 m excursion cap, ~5–6 pulse-opens each). ⚠️ **This is a sizing
correction, not a criteria change**: `n ≥ 40` and every threshold in §2–§14 stand
exactly as predeclared.

---

## 5. Preview against the criteria — and why it is only a preview

On this **excluded** leg: pulse-open `p95` **432.424 ms** fails §4 (needs
≤ 250 ms) **and** fails §3 (needs ≥ 1000 ms), landing in the **§5 inconclusive
band**. Worst-wait fraction **0.216** passes §4's ≤ 0.40 clause.
`Q/W` ratio (§12.1) = **1.396**.

⚠️ **n = 11 pulse-opens from one leg of the wrong length, outside the planned
excursion disc, in the strong band only.** It supports no verdict and must never
be quoted as one. 🔑 **But it is a fair warning: if the scored legs look like
this, the honest outcome is inconclusive**, and that was flagged to the operator
before the scored legs were planned rather than after.

---

## 6. `services.yaml` is missing fields the real schema accepts

`turn_mode` and `vio_turn_max_commands` are absent from
`custom_components/mammotion/services.yaml` but ARE accepted by
`RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA`.

✅ **It does not affect API calls.** Validation comes from the `schema=` argument
at registration; `services.yaml` is HA frontend metadata and the integration
never reads it for validation. The run was correct because the full profile was
sent from `docs/accepted-profile.json` and the echo verified 19/19.

🚨 **But filtering a payload against `services.yaml` silently ran
`vio_turn_max_commands` at its default 8 instead of the accepted 4** — my own
error on the first attempt, caught by the echo check. 🔑 **This is the standing
"schema defaults are NOT the accepted profile" trap one level up: the UI
metadata is not the schema either. Verify the echo, never the request.**

⏸️ **Fix deferred** — it only changes the HA UI service picker and would need a
deploy, against the RF freeze held until Phase 1 is banked.

---

## 7. Session end state

Gate **DISARMED** and verified from the live API: `enabled: false`,
`real_motion_allowed: false`, `blockers: ['experimental_motion_disabled']`.
Mower left parked at `(4.9416, −3.8236)` in Backyard Right at ~100 %; **the
operator is docking it.** ⚠️ Off-dock drain is ~4.3 %/h, so it should not sit
overnight.

🔑 **The 65 timing samples live on the HA coordinator, not in any session** — a
future session reads them with `motion_dispatch_timing_report`. ✏️ **But they are
IN MEMORY ONLY** (`deque(maxlen=500)` on the coordinator object): **an HA restart
or integration reload clears them.** They are banked in full in
`docs/evidence-setup-leg-timing-20260912.json`, so nothing is lost if that
happens. **Either way they are excluded from the §2 population** — a Phase 1
session must exclude them if still present, and must not assume they are.
⚠️ Per-leg snapshots (§11.3) matter more than the deque for that reason.

**No code changed. No deploy. No control-law or profile value moved.
`_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS` is still 2.0.**
