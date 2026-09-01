# Raw per-sample records — route 1 step-response runs, 2026-08-30

**Rescued from ephemeral session storage on 2026-08-31.** These are the
complete `service_response` payloads of the four real
`raw_pymammotion_step_response_probe` dispatches on 2026-08-30, preserved
because **they were never committed anywhere else and existed only in `/tmp`.**

⚠️ **This corrects a claim made the same day.** Both
`docs/vio-crosscheck-reframes-route1-step-verdicts-20260830.md` and
`CLAUDE.md` originally said every route-1 sample was "banked" and therefore
re-scorable offline for free. That was **wrong**: the four
`docs/evidence-route1-*.json` files carry only the derived `course_series`
(13–14 rows each, RTK-chord-derived course only, **no VIO**). Without the
files in this directory, any VIO-vs-RTK re-scoring would have required new
physical runs. It does not now.

## What is here

| file | config | samples | has `position` | has `vio` |
| --- | --- | --- | --- | --- |
| `raw-route1-run1-plus120-step5000-20260830.json` | +120, step 5000 ms | 127 | 127 | 127 |
| `raw-route1-run1repeat-plus120-step5000-20260830.json` | +120, step 5000 ms | 128 | 128 | 128 |
| `raw-route1-stepext-plus120-step7000-20260830.json` | +120, step 7000 ms | 146 | 146 | 146 |
| `raw-route1-run2-plus180-step7000-20260830.json` | +180, step 7000 ms | 148 | 148 | 148 |

Every sample carries `elapsed_ms`, `position` (RTK `x`/`y`/`toward`,
`position_sequence`, `position_epoch`), `vio` (`heading`, `state`), `ble`
(`is_connected`, `queue_depth`, …), `active_command`, and
`cumulative_travel_m`. Each file also retains that run's own
`phase_transitions`, `course_series`, `analysis`, `safety_gates`,
`stop_result`, `report_stream` and `motion_refresh`.

## Provenance and integrity

These are the **verbatim service responses**, not re-derived or re-formatted
records — the only transformation applied was unwrapping the outer
`{"changed_states": [], "service_response": {…}}` envelope and pretty-printing.
Each file's own `analysis` and `course_series` therefore still match what the
deployed build computed at run time, so any re-scoring can be checked against
the original numbers rather than trusted.

⚠️ All four ran on builds **before** the `reason`-field fix (commit
`af5f547f`) was deployed, except the two 7000 ms runs. The `reason` field in
the two 5000 ms files reads `"travel_guard_tripped"` and is **known wrong** —
see `docs/evidence-route1-run1-fail-20260830.md`. Score from
`motion_refresh.aborted_early` and the per-sample `travel_guard_tripped`
flags instead, which are correct in all four.

## Why they matter

They are the entire evidence base for the parked RTK-vs-VIO course-rate
question (`docs/vio-crosscheck-reframes-route1-step-verdicts-20260830.md`),
and they are **irreplaceable without commanding new motion** — 549 samples
across four supervised, operator-authorized runs that consumed roughly half a
battery charge to produce.
