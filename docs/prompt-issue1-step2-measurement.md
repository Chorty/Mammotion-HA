# Session prompt — Issue 1 step 2: measure the queue-start wait

Paste the block below to start the measurement session. Everything above the
line is context for whoever is deciding whether to run it.

**Why this exists:** beta104 shipped the instrument but took no measurement.
`_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS` is still 2.0 and stays there until a
distribution exists. Plan: `docs/plan-post-20260910-session-issues.md` issue 1.

🔑 **This does NOT require resuming the 4.0 m series, and should not.** Queue
contention accumulates **per pulse**, not per metre — legs 7 and 8 both failed
after several pulses had already succeeded. Short legs near the dock generate the
same `motion_refresh_interval_ms: 200` traffic per pulse at a fraction of the
risk, and the 4.0 m series is aborted on its own predeclared rule and stays that
way. Getting n from short legs is strictly better than getting it from long ones.

---

```
Continue the Mammotion click-to-go work: take the Issue 1 step 2 measurement.

FIRST, before anything else:
- Requery live HA + mower state fresh. Do not trust any prior chat history or
  CLAUDE.md's live-state block; it is a snapshot, not a source of truth.
- Confirm the host runs beta104 and that `motion_dispatch_timing_report` is
  registered. Confirm the motion gate is DISARMED before you plan anything.

READ, in order:
1. docs/plan-post-20260910-session-issues.md — issue 1, especially why step 3
   (changing a number) is forbidden until step 2 produces data.
2. docs/findings-clicktopath-reliability-4m-repeat-20260910.md §1.5 and §1.5.1 —
   the failure mode and the verified site map.
3. docs/findings-comms-abort-notify-20260911.md — what beta104 actually ships.
4. CLAUDE.md "How this project works" — predeclare, write the falsifier,
   per-item records, operator go/no-go before every real dispatch.

WHAT IS ALREADY TRUE (verify, don't assume):
- Every confirmed motion dispatch records its enqueue->started wait, outcome,
  budget and GATT write duration into a bounded deque (maxlen 500) on the
  coordinator. This is NOT scoped to one executor — it sits in
  `_send_ble_motion_command_confirmed`, which all motion passes through.
- Read it with the read-only service `motion_dispatch_timing_report`. It sends
  nothing. It reported `sample_count: 0` at deploy time.
- `command_result["queue_diagnostics"]` captures an `_ble_link_liveness`
  snapshot at the instant of refusal, at 7 of 15 abort sites — all four
  functions the vector executor actually runs. The other 8 belong to executors
  this measurement will not use.

THE TASK:

1. WRITE A PREDECLARATION AND COMMIT IT BEFORE ANY LEG DISPATCHES.
   It must state, before the data exists:
   - the minimum sample count that would make the distribution worth acting on,
     and your reasoning for that number (n=2 is what we have now and is why this
     session exists);
   - what result would JUSTIFY raising the 2.0 s constant;
   - what result would mean the constant is NOT the problem and the real cause
     is elsewhere (this is the falsifier — write it, do not skip it);
   - what result would mean the measurement itself was inconclusive.
   Choosing these after seeing the numbers is the exact failure the
   predeclaration discipline exists to prevent.

2. PLAN THE LEGS FOR SAMPLE COUNT, NOT DISTANCE.
   Contention builds per pulse. Prefer many short legs close to the dock over
   few long ones. Do NOT resume the 4.0 m aligned-start series — it is ABORTED
   on its own rule and nothing here reopens it. Use
   `scripts/plan_aligned_leg.py` if it helps, and respect the BLE coverage
   constraint it already models.

3. RUN THEM UNDER THE STANDING PROTOCOL, NO EXCEPTIONS.
   Daylight. Operator present with explicit go/no-go immediately before each
   dispatch. Fresh corridor scan against the map; ask for a physical tape
   measurement on any corridor tighter than a couple of metres. Send
   docs/accepted-profile.json verbatim and verify it key-by-key. Gate verified
   disarmed from the live API AND RAW core.config_entries at session end.

4. COLLECT AND REPORT PER-ITEM, NOT AGGREGATE.
   After each leg, call `motion_dispatch_timing_report` and save the raw
   `samples` array — not just the percentiles. Net figures in this project have
   already hidden a 27 degree turn reversal and a live BLE link. Write the
   evidence to docs/evidence-*.json alongside the findings doc.

5. ONLY THEN interpret. The number to look at is
   `worst_wait_fraction_of_budget`, plus the p50/p95 spread and the
   `outcomes` counts. Report what the data says against the predeclared
   criteria, including if it says the constant is fine.

DO NOT, under any circumstances:
- change `_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS` or
  `motion_refresh_interval_ms` in this session. Step 3 needs its own operator
  decision AFTER the measurement, and `motion_refresh_interval_ms` is an
  accepted LUBA-acceptance profile value that owes its own predeclaration and
  Gate 5.
- resume the 4.0 m series.
- build Option D (auto return_to_dock). It is unauthorized;
  docs/predeclared-comms-abort-auto-dock-20260911.md explains why.

If the mower cannot run today (weather, battery, dark, BLE), say so and stop —
do not substitute a simulated or reasoned-about result for a measured one.
```
