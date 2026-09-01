# VIO cross-check reframes both step-extension verdicts — correction

**2026-08-30, same day as both runs.** Prompted by the operator asking
whether run 2's FAIL was a VIO problem. It was not — VIO read `state: 2`
(live) continuously through both runs, matching
`visual_positioning_status: signal_good` from telemetry. But checking VIO's
own independent heading track against the probe's own course measurement
(RTK-position chord bearing, `_step_response_course_series`) surfaced
something more consequential: **the two instruments disagree about which run
converged, and disagree in the opposite direction from what was already
published.**

🗑️ **This does not overturn `docs/evidence-route1-step-extension-pass-20260830.md`
or `docs/evidence-route1-run2-plus180-fail-20260830.json` — both stand as
correct readings of what the probe's registered instrument (RTK-position
chords) recorded, and criterion 2a is defined against that instrument. This
document adds a cross-check that weakens confidence in treating either
verdict as settled physical fact.**

## The cross-check

Same method as criterion 2a — last two step-phase rates, but computed from
VIO's own heading field instead of RTK-position-chord bearing. VIO updates in
discrete steps roughly once per second (consistent with the ~1 Hz report
cadence; it holds a value across several 100 ms samples then jumps), so rates
are computed between consecutive distinct VIO readings.

| | RTK-chord (published) | **VIO (this cross-check)** |
| --- | --- | --- |
| +120 (published PASS) | 0.11°/s apart — PASS | **1.96°/s apart — FAILS the same 1.5°/s bound** |
| +180 (published FAIL) | 5.64°/s apart — FAIL | **0.38°/s apart — PASSES cleanly** |

The two instruments do not merely disagree by degree — they disagree about
**which run** shows the cleaner convergence. RTK-chord says +120 converged
and +180 didn't; VIO says the reverse.

### Raw VIO step-phase rate sequences

**+120:** `0.356 → -1.247 → -6.014 → -6.597 → -7.001 → -6.542 → -6.990` — a
clean climb to a plateau around -6.5 to -7.0°/s, then two final readings
(-6.542, -6.990) 1.96°/s apart, just over the bound.

**+180:** `-0.166 → -5.675 → -13.563 → -11.443 → -14.293 → -11.402 → -11.545`
— a noisier climb with a larger transient overshoot, but the **last two**
readings (-11.402, -11.545) land only 0.38°/s apart.

## What this does and does not mean

**Does not mean:** that the RTK-chord course measurement is wrong, that VIO
is the "true" answer, or that either published verdict should be reversed.
VIO is itself quantized (holds for ~0.9–1.1 s between updates) and shows real
transient variability before settling in both runs — it is not simply a
cleaner, noise-free channel being ignored.

**Does mean:** criterion 2a, computed from either instrument alone, is
operating close enough to that instrument's own noise floor that a single
run's verdict is not robust to which instrument answers the question. Two
independent ~1 Hz position/heading channels, measuring the same physical
event, produced opposite pass/fail calls on both runs. That is evidence the
measurement method — not necessarily the plant — is the limiting factor
right now, for both the pass and the fail.

## What this does NOT authorize

🛑 Per this project's own standing rule against reacting to same-day data:
this finding does **not** authorize touching criterion 2a's threshold,
switching `_step_response_course_series` to use VIO instead of RTK position,
adding VIO as a second required channel, or re-running either configuration
to "settle" which instrument is right. Any of those is a separate,
deliberately-written decision. What it does do is put a caveat on both of
today's headline results that a future session must not skip past:

* `docs/evidence-route1-step-extension-pass-20260830.md`'s "first full PASS"
  is a pass against ONE instrument's noisy 1 Hz reading, not an
  unambiguous physical fact. A repeat could plausibly show 2a passing on one
  instrument and failing on the other again.
* `docs/evidence-route1-run2-plus180-fail-20260830.md`'s FAIL is likewise not
  unambiguous — VIO's own reading of the same run shows a comparably clean
  convergence to the +120 run's RTK-chord reading.

## Open question for a future, deliberate decision

Which channel (or a fusion of both) should criterion 2a actually be scored
against, and whether n=1 per configuration was ever enough to distinguish
"converged" from "one favorable noise draw" on either channel. Not answered
here on purpose.

---

## 🚩 FLAGGED FOR LATER — parked deliberately, 2026-08-31

**Operator decision: this is flagged, not being worked.** Nobody should pick
it up reflexively as "the obvious next task". It blocks nothing that
currently works — click-to-path via stop-measure-go is unaffected, since it
never consults criterion 2a or the step-response probe at all. What it gates
is the route-1 dead-time line of work, which is itself downstream of Phase 2
continuous steering, parked on value grounds by standing decision 5.

### The framing is sharper than it was when this doc was written

Re-checked against the pinned backend on 2026-08-31 (`chorty-0.8.12.post4`):

* `vision_info` exists only in `pymammotion/data/model/report_info.py` — VIO
  heading arrives **inside the same `sys.toapp_report_data` payload as
  `locations[0]`**, corroborating `docs/the-1hz-bundle-is-the-ceiling-20260822.md`
  (position, `toward` and VIO heading change on exactly the same instants).
* `last_report_data_at` is **still absent** from post4, so there is still no
  per-report-type arrival stamp.

🔑 **Consequence: the two "instruments" are not independently-timed channels.
They are two fields of one ~1 Hz bundle.** So the disagreement cannot be a
freshness, sampling-lag or transport artifact, and no connectivity/transport
fix can address it (checked explicitly against upstream `f4428d47`, which is
already ported here and is unrelated).

That reduces the open question to something better posed than "which channel
is right":

> RTK-chord bearing is a **geometric proxy derived across two payloads**;
> VIO heading is a **direct sensor reading within one payload**. Which is the
> better estimator of rotation rate — particularly at higher curvature, where
> the chord proxy degrades because more heading change is compressed into the
> same ~1 s chord?

⚠️ **That framing argues against the status quo but does not settle it.** VIO
has its own documented weaknesses — light-dependence, a live calibration
offset per run, and discrete ~1 Hz latching — so "switch to VIO" is not
obviously correct either. The dual-channel-agreement option (require both to
show convergence) remains the conservative middle, at the cost of making 2a
materially harder to pass.

### What resuming this would need, in order

1. A written decision on what 2a is scored against — RTK chord, VIO heading,
   or agreement between both — made **before** any new capture, per the rule
   in "What this does NOT authorize" above.
2. Only then, code changes to `_step_response_course_series` /
   `_step_response_analysis`, with the existing runs re-scored offline from
   their banked raw samples first (both are fully recorded — no new motion is
   needed to re-score them).
3. Only then, and only if re-scoring leaves it genuinely ambiguous, new
   physical runs.

🔑 **Step 2 is free and needs no mower** — but only because the raw data was
rescued on 2026-08-31. See `docs/raw-samples/README.md`.

🗑️ **CORRECTION, 2026-08-31.** This section originally said every route-1
sample was already banked and re-scorable for free. **That was wrong.** The
four `docs/evidence-route1-*.json` files contain only the derived
`course_series` (13–14 rows each, RTK-chord course only, **no VIO field at
all**). The per-sample records — the only place VIO heading was ever
written — existed solely in ephemeral `/tmp` session files and had never been
committed. Had they been cleared, this question would have required **new
physical runs** to answer.

✅ They are now preserved at `docs/raw-samples/raw-route1-*-20260830.json`:
**549 samples across all four runs, every one carrying both `position` and
`vio`.** With those in the tree the "free offline re-scoring" claim is true
as written — it simply was not true when first written.
