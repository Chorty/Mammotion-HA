# Route 1, step extension — predeclared before any code change or dispatch

**Written 2026-08-30, after run 1 and its repeat, before any capture at a
longer step phase exists.** Nothing here may be edited once that capture
exists. This tests the onset-lag hypothesis raised by
`docs/evidence-route1-run1-repeat-fail-20260830.md` — it does not assume the
answer.

🛑 **This document authorizes nothing.** It requires two code changes, a
release, a deploy, a fresh corridor scan and explicit per-run operator
authorization, same discipline as every prior step in this project.

## 1. The hypothesis this tests

Both route-1 run-1 captures FAILED criterion 2a (step reaches steady
rotation), and in **both** of them the step phase's *final* interval showed
the rotation rate *increasing* in magnitude versus the interval before it:

* run 1: -5.686 → -8.179 °/s
* run 1 repeat: -3.828 → -11.108 °/s

That is the signature of a rotation still accelerating through onset lag when
the 5000 ms step ends — not chord noise decaying around a settled value.
Criterion 2b (settle goes flat), by contrast, passed cleanly on the repeat
(0.26°/s) using the same 5000 ms length, once the mower was already turning.
**If the step phase simply needs more time than 5000 ms to clear onset lag
and reach steady rotation, extending it — and nothing else — should let 2a
pass where it has now failed twice.** This run tests exactly that and only
that.

⚠️ **n = 2 on the current failure. This is a hypothesis worth testing, not an
established fact.** If a longer step *also* fails 2a, that is real evidence
against the onset-lag explanation, not a reason to keep extending `step_ms`
indefinitely.

## 2. What moves, and why each one is a safety-relevant choice

| constant | now | proposed | why |
| --- | --- | --- | --- |
| `step_ms` schema max | 5000 | **7000** | needed to command the longer step |
| `_STEP_RESPONSE_MAX_TOTAL_MS` | 14000 | **16000** | 3000 + 7000 + 5000 = 15000 ms is refused today |
| `max_travel_m` (operational value for this dispatch) | 4.0 | **4.5** | at the existing 4.5 m schema ceiling — no code change, a dispatch-parameter choice |
| `baseline_ms`, `settle_ms` | 3000, 5000 | **unchanged** | not implicated by either run's evidence — settle already passed once at 5000 ms; touching it now would confound this test |
| `linear_speed` | 400 | **unchanged** | pinned; route 2 (slower) is closed by measurement |

🔑 **This raises a safety bound, stated plainly, same as the original route-1
predeclaration:** `step_ms` going 5000 → 7000 ms means the mower drives
**2000 ms further open loop, on the same uncorrected curve**, than either run
today. `max_travel_m` at 4.5 (already-authorized ceiling, not a new one)
means the guard permits 0.5 m more travel than either run today used.

## 3. The arithmetic for 7000 ms, not a round number picked freely

Using the same ramp/steady model as the original predeclaration (2 s ramp at
~0.13 m/s during baseline, ~0.26 m/s steady afterward):

```
baseline 3000 + step 7000 + settle 5000           = 15.0 s
path, modelled from measurement
  (2 s ramp at ~0.13 m/s, then ~0.26 m/s steady)  ~ 3.90 m
plus 0.50 m stop overshoot                         ~ 4.40 m of clearance
```

**Against the 4.5 m ceiling this leaves only ~0.10 m of headroom on the
model** — thin, but the guard's own accounting is conservative
(cumulative consecutive-fix distance, not the model), and `max_travel_m=4.5`
gives the guard a real number to enforce independent of the model being
right. 🔑 **Why not go straight to a bigger step (e.g. 8000 ms) to be more
decisive in one run:** that pushes modelled distance to ~4.4 m before the
stop overshoot is even added, which would need `max_travel_m` **above** the
existing 4.5 m ceiling — a second ceiling raise in the same day, on top of
one already spent this session. **7000 ms is the largest step increase that
fits inside the ceiling already authorized**, which is why it is proposed
first rather than a larger jump.

⚠️ **Why the total cap moves to 16000, not to the sum of every field's own
max (5000+7000+6000=18000):** the total-window cap is deliberately tighter
than the sum of per-field maxima, by design (see §2 of the original route-1
predeclaration) — it forces one specific, tested combination rather than
letting every phase be maxed at once. 16000 gives the intended 15000 ms
combination a 1000 ms margin, matching the margin the 12000→14000 raise gave
its own intended combination.

## 4. What is deliberately NOT changed

🗑️ **`settle_ms` stays 5000.** It already passed 2b cleanly once (0.26°/s) at
this length; changing it now would confound whether a pass is attributable to
the settle change or the step change.

🗑️ **`baseline_ms` stays 3000.** Not implicated by either run's evidence.

🗑️ **`linear_speed` stays 400** and **`max_travel_m`'s schema ceiling stays
4.5** (using it, not raising it).

🗑️ **The 0.15 m chord floor and `stop_overshoot_m` (0.50 m) are untouched.**

🗑️ **Criteria 2a and 2b are UNCHANGED.** Still ≥3 informative intervals and
the last two rates within 1.5°/s, in each phase. This run either meets that
bar at a longer step length or it does not — the bar itself does not move.

## 5. Pass criteria — unchanged from the original six

Identical to `docs/phase2-route1-predeclared-20260830.md` §5. This document
changes phase lengths and the travel budget only, not what counts as a pass.

1. Report stream ready, no `readiness_reason`, `position_sequence` advancing.
2. 2a — the STEP reaches steady rotation: ≥3 informative intervals, last two
   step rates agree within 1.5°/s.
3. 2b — the SETTLE goes flat: ≥3 informative intervals, last two settle rates
   agree within 1.5°/s.
4. Containment holds: every sample inside the corridor, stop confirmed.
5. The travel guard does not trip. 🔑 **Score this from the raw per-sample
   evidence (`travel_guard_tripped` flags, `motion_refresh.aborted_early`,
   phase-transition timing), not from the service's own `reason` field** —
   that field is known wrong on this host until the fix in commit `af5f547f`
   is deployed.
6. Gate disarmed afterward, verified from the live API **and** RAW
   `core.config_entries`.

**Any failure is a FAIL**, same as before.

## 6. Corridor

**A 10.0 m axis-aligned square**, half-width 5.0 m ≥ the required
`4.5 + 0.5 = 5.0 m` — an exact match, not a margin. The original route-1
predeclaration verified squares up to 10.0 m contained in "Backyard Right";
this uses the full extent of what was verified, with nothing to spare.

⚠️ **Re-scan and re-verify containment at the LIVE position before dispatch,
same as every prior run.** Given the thin margin here, if the live-position
scan shows anything less than the full 10.0 m fitting cleanly, **do not
shrink the square to fit — refuse the run and reposition instead**, the same
rule that governed run 1's own corridor sizing.

## 7. What a pass here would authorize

If this run passes ALL SIX criteria at the extended step length: the onset-lag
hypothesis is supported (not proven — n=1 at this new length), and per the
original route-1 predeclaration's own §7/§9, **that would authorize repeating
at +180 using this same, now-longer, step length** — nothing more. It would
not authorize touching `step_ms` again, would not authorize Phase 2 steering,
and would not itself write the feed-forward design document without a second
run at +180 first, exactly as originally planned.

If it FAILS 2a again: the onset-lag explanation loses support, and the next
move is a separate, deliberately-written decision, not a further `step_ms`
increase in the same sitting.

## 8. Preconditions

* 🔋 **Docked and charged first.** Battery has been in the low-to-mid 40s%
  and draining across both runs today, below this project's own precondition
  for its longest runs, waived twice already by explicit operator
  authorization. This run is longer still — charge before dispatching it.
* Both code changes (§2) shipped in a release and deployed motion-disabled,
  verified in the deployed bytes exactly as beta87's cap changes were.
* The `reason`-field fix (commit `af5f547f`) should ship in the same release
  — it is already committed and unrelated in mechanism, and deploying it
  makes this run's own `reason` field trustworthy instead of needing the same
  manual per-sample correction both runs today required.
* Off dock, `AREA_INSIDE`, RTK Fix, BLE live, blades off.
* A dry run showing 15/15 gates and `blockers: []` on the exact configuration,
  at the mower's actual live position.
* Explicit per-run operator authorization, confirmed immediately before
  dispatch (on-site, supervising, area clear, blades off) — same as both
  runs today.
