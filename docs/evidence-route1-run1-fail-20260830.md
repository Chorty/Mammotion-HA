# Route 1 run 1 — FAIL, but the first window to complete cleanly

**2026-08-30, beta87.** One supervised, explicitly authorized run of
`raw_pymammotion_step_response_probe` against
`docs/phase2-route1-predeclared-20260830.md`: baseline 3000 / step 5000 /
settle 5000 ms, `step_angular_speed=120`, `max_travel_m=4.0`, 9.0 m square
corridor centred on the mower's live position. Raw evidence:
`docs/evidence-route1-run1-fail-20260830.json`.

## Verdict: FAIL

Criteria 2 (2a — step reaches steady rotation) and 3 (2b — settle goes flat)
both fail. Per the predeclaration, "any failure is a FAIL" — `tau_actuator_s =
2.049 s` came out of the analysis field and must **not** be quoted as a settled
result.

| # | criterion | result |
| --- | --- | --- |
| 1 | report stream ready | ✅ PASS |
| 2 | 2a — last two step rates within 1.5°/s | ❌ **FAIL — 2.49°/s apart** |
| 3 | 2b — last two settle rates within 1.5°/s | ❌ **FAIL — 2.07°/s apart** |
| 4 | containment + stop confirmed | ✅ PASS |
| 5 | travel guard does not trip | ✅ PASS (see bug below) |
| 6 | gate disarmed after, verified live API + RAW | ✅ PASS |

## What actually happened

Unlike the two 2026-08-30 attempts recorded in
`docs/evidence-option-b-blocked-by-travel-budget-20260830.json`, this window
ran to completion for the first time: the full 13000 ms elapsed (last sample
at 13000.739 ms), phase transitions landed exactly on schedule
(0 / 3000.7 / 8000.7 ms), cumulative travel finished at **2.7111 m of the
4.00 m budget**, and 0 of 127 samples carry `travel_guard_tripped: true`.
5 informative intervals were captured in both the step and settle phases
(≥3 required).

The failure is in the **shape** of the rotation, not in reach or safety: the
step phase's rate was still increasing in magnitude at the end
(-5.686 → -8.179 °/s, a 2.49°/s jump) and the settle phase's rate crossed from
positive back through near-zero in its last two intervals (+1.929 → -0.136,
a 2.07°/s swing) — the same non-monotonic "chord noise on ~0.26 m chords"
signature the 2026-08-29 and 2026-08-30 evidence already documented, now
showing up inside a window that otherwise ran clean.

## 🐛 A code bug was found and fixed the same day

The service's own `reason` field reported `"travel_guard_tripped"` for this
run — which is what first suggested the run had failed for the same reason as
the two earlier truncated attempts. It hadn't: the phase timing and per-sample
evidence above show the window completed on schedule with the guard nowhere
near tripping.

**Root cause:** `_step_response_probe_impl`'s `finally` block sets the
`travel_abort` event **unconditionally** as part of mandatory-stop teardown —
it does this specifically to unblock the phase and sampler tasks, regardless
of whether `_continuous_refresh_window` returned because the guard genuinely
fired mid-window or because the window simply finished. The `reason` field
then read that same event afterward, so it read `"travel_guard_tripped"` on
**every** real run, trip or no trip.

**Fix:** `_step_response_completion_reason()` now reads
`motion_refresh["aborted_early"]` instead — set only when
`_continuous_refresh_window`'s own loop observed the abort event **while still
running**, which is the one signal that actually distinguishes a real trip
from normal completion. Two unit tests
(`test_reason_is_window_complete_when_the_refresh_loop_never_saw_the_abort`,
`test_reason_is_travel_guard_tripped_only_when_the_loop_observed_it`) pin both
branches directly against the extracted helper.

⚠️ **This bug means the `reason` field cannot be trusted retroactively on its
own.** Two prior evidence files
(`docs/evidence-dead-time-measured-20260829.json`,
`docs/evidence-option-b-blocked-by-travel-budget-20260830.json`) also recorded
`"reason": "travel_guard_tripped"` on every real run they describe. This does
**not** automatically invalidate their conclusions — those were drawn from
independent evidence (informative-interval counts, course_series values,
explicit "still rotating at the last sample" observations), which the bug
does not touch — but the `reason` field itself in those files should not be
read as confirmation of an early trip without checking the underlying samples,
exactly as this run required. Not re-audited here; flagged for awareness.

## Safety

15 of 15 gates passed before dispatch, `blockers: []`. Corridor: 9.0 m square,
re-scanned and re-verified at the mower's live position immediately before
dispatch (boundary clearance 5.934 m against the 4.50 m required disk). Every
sample stayed inside the corridor. Stop confirmed
(`stop_result.ok: true`, `ack.movement_ok: true`). Explicit operator
confirmation (on-site, supervising, area clear, blades off) was taken
immediately before dispatch. Gate disarmed afterward and verified from both
the live API and RAW `core.config_entries` (`enable_experimental_motion:
false`, after the documented ~15 s lazy-write delay).

⚠️ Battery was 48% at dispatch, below the predeclaration's "docked and charged
first" precondition. The operator explicitly authorized proceeding anyway.

## What this does not establish

* `tau_actuator_s = 2.049 s` is not a settled value — neither phase reached a
  stable rate this run captured.
* n = 1, one sign (+120). A FAIL does not authorize run 2 (+180); it was not
  dispatched.
* Whether a longer step or settle phase would pass 2a/2b is unknown, and
  `step_ms` is deliberately capped at 5000 by the predeclaration's own
  reasoning (§4) — raising it is not a free lever.

## What a pass would have authorized

Writing the feed-forward design document (predeclaration §9). This is a FAIL,
so it authorizes nothing beyond recording the result.
