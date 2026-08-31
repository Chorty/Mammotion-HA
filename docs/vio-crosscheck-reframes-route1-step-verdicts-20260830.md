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
