# Route 1, run 2 (+180) — FAIL, the pass does not generalize

⚠️ **QUALIFIED THE SAME DAY — read
`docs/vio-crosscheck-reframes-route1-step-verdicts-20260830.md` alongside
this.** VIO's own independent heading track disagrees with this run's own
RTK-chord measurement: it shows the last two step-phase rates only 0.38°/s
apart — a clean PASS by the same 1.5°/s bound this document's 2a criterion
uses, against the 5.64°/s FAIL this document reports. The FAIL below is
correct as a reading of the probe's registered instrument — it is not an
unambiguous physical fact independent of which channel is asked, and "the
+120 pass does not generalize" (this document's title) is itself now in
question.

**2026-08-30, beta88.** The repeat authorized by
`docs/phase2-route1-step-extension-predeclared-20260830.md` §7: same phases
that passed at +120 (baseline 3000 / step 7000 / settle 5000, `max_travel_m=
4.5`, 10.0 m corridor), `step_angular_speed=180` instead. Raw evidence:
`docs/evidence-route1-run2-plus180-fail-20260830.json`. Compare against the
pass this repeats: `docs/evidence-route1-step-extension-pass-20260830.md`.

## Verdict: FAIL

Criterion 2a fails badly — worse than either 5000 ms-step attempt earlier
today.

| # | criterion | +120 (passed) | **+180** |
| --- | --- | --- | --- |
| 1 | report stream ready | ✅ | ✅ |
| 2 | 2a — last two step rates ≤1.5°/s | ✅ 0.11°/s | ❌ **5.64°/s apart** |
| 3 | 2b — last two settle rates ≤1.5°/s | ✅ 1.44°/s | ✅ 0.77°/s |
| 4 | containment + stop confirmed | ✅ | ✅ |
| 5 | travel guard does not trip | ✅ | ✅ |
| 6 | gate disarmed after, verified | ✅ | ✅ |

## The step phase does not converge at +180

The +120 run's step-rate sequence climbed then flattened
(`-9.026 → -6.127 → -8.241 → -8.353`, last two nearly identical). At +180 the
same 7000 ms step produces:

```
-3.366 -> -11.885 -> -10.765 -> -15.094 -> -8.246 -> -13.886   (deg/s)
```

No convergent trend — the rate swings between roughly -8 and -15°/s across
the whole phase with no sign of settling. **The 7000 ms step length that
fixed 2a at +120 does not generalize to +180.** Whatever combination of onset
lag and mechanical response the extension addressed, it evidently does not
scale simply with a 50% larger commanded angular speed.

🔑 **`omega_step_deg_per_s` did scale roughly as expected in magnitude**
(-6.12 at +120 → -10.65 at +180), consistent with prior findings that
rotation rate increases with commanded angular speed in this band. But that
is a directional observation from a noisy sequence, not a fitted law — the
underlying step-rate data at +180 is too unstable to support any
quantitative scaling claim.

## Settle is robust again — third clean pass in a row

0.77°/s on the last two settle intervals, comparable to the +120 run's
1.44°/s and run 1 repeat's 0.26°/s. **This is now three of three step-response
runs where the settle phase converged cleanly at 5000 ms**, regardless of
whether the step phase itself converged — settle length looks solid.
Criterion 2a, not 2b, is where this project's open question now sits.

## The reason-field fix, confirmed a second time

`"reason": "window_complete"` again, matching the raw evidence
(`aborted_early: False`, 0/148 tripped, full 15001.6 ms of a 15000 ms window)
without any manual correction. Second real-hardware confirmation of commit
`af5f547f`.

## The repositioning drive (before this run)

The mower (moved again between runs) sat at `(3.8728, -7.4005)`, where the
worst-case clearance was only 3.841 m against the 5.0 m required disk
(0.77x). A single call of the accepted closed-loop reach profile drove it to
`target_reached` at **0.073 m** from `(5.98, -5.24)` — the tightest landing
of any repositioning drive today, 4 turn commands and 12 linear commands.
Final position `(6.0038, -5.3077)`, re-scanned worst-case clearance
**5.904 m (1.18x)** before dispatch.

## Safety

15 of 15 gates before dispatch, `blockers: []`. Every sample stayed inside
the corridor. Stop confirmed. Cumulative travel 2.954 m of the 4.5 m budget.
Explicit operator confirmation taken immediately before both the
repositioning drive and the step-response dispatch. Gate disarmed and
verified from both the live API and RAW `core.config_entries` after each
dispatch.

## What this does not establish

* n = 1 at +180 with this step length. The oscillation could in principle be
  one noisy run, but it looks qualitatively different from the small chord
  noise seen elsewhere today — a large, sustained swing, not perturbations
  around a trend.
* `tau_actuator_s = 2.162 s` from this run is not a settled value.
* Whether a longer step would fix 2a at +180, whether the effect is specific
  to +180 or would also appear at other larger commands, and whether
  anything about +180 specifically (versus +120) explains the difference are
  all open.

## What this run authorizes

Nothing further on its own terms — neither a pass nor a fail at +180
authorizes another `step_ms` change, Phase 2 steering, or the feed-forward
design document. The predeclaration's authorization chain for this specific
pair of runs (+120 pass → +180 repeat) is now exhausted; any next step is a
separate, deliberately-written decision.
