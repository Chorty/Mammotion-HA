# 🏁 Gate 5 PASSED on the reach profile — 2026-08-18

Card-driven, four segments, **4/4 `target_reached`**, every landing inside the
0.15 m tolerance. **The acceptance debt created by beta57 is discharged.**

Evidence: `docs/evidence-gate5-beta57-20260818.json` (the card's own run-record
export, renamed to the `evidence-` convention).

## Result

| seg | leg | landing | stop | linear | turn | realign |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8005 m | **0.1038** | `target_reached` | 3 | 0 | 0 |
| 2 | 0.9032 m | **0.0863** | `target_reached` | 3 | 3 | 1 |
| 3 | 0.8198 m | **0.1261** | `target_reached` | 3 | 2 | 0 |
| 4 | 0.7593 m | **0.1129** | `target_reached` | 3 | 1 | 0 |

**Mean 0.1073 m.** Zero reverse-recovery, zero budget exhaustion, zero errors.
Geometry: four ~0.8 m legs with three −55.0° junctions, 3.20 m total.

⚠️ Worse than the 2026-08-12 re-pass (0.0674 / 0.1032 / 0.0807 / 0.0607, mean
**0.0780**). Both pass; the difference is uncontrolled — different geometry,
different day, and this run went at 20:28 with VIO at **79** tracked features,
one below saturation. n = 1 either way, so do not read a trend into it.

## Profile identity — proven in fact

Every key of the dispatched payload is byte-identical to
`LUBA_ACCEPTANCE_PROFILE`, checked mechanically rather than by eye:

    drifting keys: NONE
    max_linear_pulse_ceiling dispatched: 22

So the card demonstrably sent the candidate profile to the mower, which is the
question Gate 5 exists to answer. `docs/accepted-profile.json` has been
re-snapshotted to this profile, so `scripts/check_accepted_profile.py` and every
future release body now report **accepted**.

## 🚨 What this pass does NOT establish

**Gate 5 accepted the profile. It did not validate the control-law change.**

Replayed through `scripts/replay_reaim_trigger.py` (self-validating, 4/4 on this
run), the old and new re-aim triggers made **identical decisions at all eight
decision points**:

| seg | pulse | dist | aim | projected | old | new |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0.166 | −28.039° | 0.0803 | no | no |
| 2 | 2 | 0.235 | +10.015° | 0.0411 | no | no |
| 3 | 2 | 0.199 | +31.308° | 0.1076 | no | no |
| 4 | 2 | 0.231 | +28.351° | 0.1131 | no | no |

Three suppressions fired (`already_lands_inside_tolerance`) — aim errors of
28–31° at 0.17–0.23 m range, correctly declined because driving on still lands
inside the disc. That is the **beta42** quadrature guard working, and the old
rule would have suppressed all three identically.

The one `realignment` in segment 2 was the **post-turn alignment gate**
(`before_linear: true`, `alignment_tolerance_degrees: 10`, 15.121° → corrected),
which is **beta40** behaviour, not this change.

The ceiling change is equally unexercised: every segment used **3** linear
pulses against a ceiling of **22**, so 22 never bound and 14 would have served.

🔑 **Both halves of beta57 only matter on long legs**, and Gate 5's validated
configuration is 0.8 m legs. Corpus replay put every old-vs-new divergence on
legs of 1.9–4.0 m. So:

- the profile is accepted, correctly and on real evidence;
- the reach work remains **unvalidated on hardware**, exactly as it was before
  this run;
- *"is `vio_max_realignments: 3` enough"* is **still unanswered** — no mid-drive
  correction has ever fired on the new code.

## Conditions

VIO 79 features / "Light" at dispatch, RTK Fix, `Backyard Right`, blades off,
20:28 EDT — past sunset, at the edge of usable light. 🔑 The operator notes
`tracked_features` **saturates at 80 in adequate light and falls off a cliff**
when it goes, so 79 is "still enough", not "plenty of margin". A longer run
might not have held.

## What is still owed

Nothing for acceptance. For the reach work: a supervised run on a leg long
enough to make the trigger fire — the corpus says **1.9 m or more after a
junction turn**. Until then beta57's changes are shipped, accepted as a profile,
and untested as a control law.
