# Research prompt — which signal should criterion 2a be scored against?

**Hand this to a fresh session. Intended model: Fable.** Everything below is
offline analysis of banked data. Copy the block under "The prompt" verbatim.

---

## The prompt

You are researching a measurement-methodology question in the Mammotion-HA
project. **This is offline analysis of already-recorded data. No mower, no
Home Assistant, no motion, at any point.**

### Absolute boundaries — these are not negotiable

1. 🛑 **Command no motion and touch no hardware.** Do not call any
   `mammotion.*` service, do not touch the experimental motion gate, do not
   SSH to the Home Assistant host, do not run anything in `scripts/ha_*`. If
   you believe you need live data, stop and say so instead.
2. 🛑 **Work only on the branch `research/rtk-vio-course-rate-scoring`.**
   Never commit to `main`, never merge into it, never rebase it. Commit your
   work on the research branch only.
3. 🛑 **Never push to any `mikey0000/*` repository.** They are read-only for
   this work. The only writable remote is the `Chorty` fork, and only on the
   research branch.
4. 🛑 **Do not change any scoring code in `custom_components/` as part of
   your investigation.** The deliverable is a written recommendation, not an
   implementation. If your conclusion is that code should change, say what
   should change and why; leave it unimplemented.

### Read first, in this order

1. `docs/vio-crosscheck-reframes-route1-step-verdicts-20260830.md` — the
   finding you are investigating, including its "FLAGGED FOR LATER" section
   which has the sharpened framing. **This is the most important file.**
2. `docs/raw-samples/README.md` — your data, its provenance, and one caveat
   about a known-wrong `reason` field.
3. `docs/phase2-route1-predeclared-20260830.md` §5 and
   `docs/phase2-route1-step-extension-predeclared-20260830.md` §5 — the
   definition of criteria 2a and 2b as they currently stand.
4. The four `docs/evidence-route1-*.json` files, for each run's published
   verdict and the reasoning behind it.

### The question

Criterion 2a asks whether the step phase reached a steady rotation rate: **≥3
informative intervals, and the last two rates within 1.5 °/s.** It is
currently scored against course computed from RTK-position chord bearing.

On 2026-08-30 a hand cross-check found VIO's own heading track disagrees
about which runs converged — in the opposite direction from the published
verdicts:

| run | RTK-chord (published) | VIO (cross-check) |
| --- | --- | --- |
| +120, step 7000 ms | 0.11 °/s — PASS | 1.96 °/s — fails |
| +180, step 7000 ms | 5.64 °/s — FAIL | 0.38 °/s — passes |

**Your question: which signal — RTK chord bearing, VIO heading, or agreement
between both — should criterion 2a be scored against, and what does the
banked data actually support?**

A key structural fact, already verified against `chorty-0.8.12.post4`:
`vision_info` and `locations[0]` arrive **in the same `sys.toapp_report_data`
payload**, so these are two fields of one ~1 Hz bundle, not independently
timed channels. The disagreement cannot be freshness or sampling lag. RTK
chord bearing is a geometric proxy derived **across two payloads**; VIO
heading is a direct sensor reading **within one**.

### Method — the order matters, and it is the point

This project has repeatedly been burned by choosing a criterion after seeing
which choice produces the desired answer. Structure your work so that cannot
happen:

1. **Predeclare first, in a committed file.** Before computing any verdict,
   write down every candidate scoring rule you will evaluate, exactly how
   each computes a rate, how it decides pass/fail, and what result would make
   you prefer or reject each one. Commit that file. Its git timestamp is what
   makes the rest trustworthy.
2. **Only then compute.** Score all four runs under every candidate rule.
3. **Sanity-check against the instrument itself.** Each raw file retains the
   `analysis` and `course_series` the deployed build produced at run time.
   Your reimplementation of the *existing* RTK rule must reproduce the
   published numbers. If it does not, that is a finding — stop and report it
   rather than proceeding on a rule you cannot reproduce.
4. **Write the recommendation last**, and state plainly what the data cannot
   settle.

Candidate rules worth covering, at minimum — add others if you see better
ones, but declare them in step 1:

* the current RTK chord-bearing rule, unchanged;
* VIO heading rate, computed between consecutive **distinct** VIO readings
  (VIO latches, holding one value across several 100 ms samples, so naive
  per-sample differencing yields spurious zeros);
* agreement of both (2a passes only if both channels independently pass);
* any noise-floor-aware variant you can justify — e.g. comparing each rule's
  residual against that channel's own scatter, rather than against a fixed
  1.5 °/s bound applied to both.

### Questions worth answering along the way

* Does the RTK chord proxy degrade measurably as rotation rate rises? The
  +180 run turns roughly 50% faster than +120 with the same ~1 s chords, so
  more heading change is compressed into each chord. Is the disagreement
  correlated with commanded angular speed, or is that just the story that
  fits n=1 per config?
* Is 1.5 °/s even the right bound for *either* channel, given each one's
  measured scatter across 549 samples? A bound near a channel's own noise
  floor cannot distinguish "converged" from "one favorable draw".
* What does the settle phase say? It passed on three of four runs under the
  RTK rule. Does it agree under VIO too? A rule that flips settle verdicts
  is more disruptive than one that only touches 2a.
* Can anything here be answered at all with n=1 per configuration, or is the
  honest conclusion "insufficient data, and here is exactly what a future
  run would need to measure"?

### Deliverables, all on the research branch

1. The predeclaration file from step 1 (committed **before** any results).
2. A re-scoring script under `scripts/`, runnable offline with no network,
   no Home Assistant import, and no mower access — following the pattern of
   the existing offline analyzers such as
   `scripts/analyze_phase1_capture.py` and
   `scripts/reanalyze_mirror_pairing.py`.
3. A findings document with the full per-run, per-rule results table.
4. A recommendation section: which rule, why, what it would cost (does it
   flip any published verdict? does it make 2a harder to pass, and by how
   much?), and what remains unresolved.

### What a good answer looks like

Not "VIO is better". Something closer to: *"Under rule X, runs A and B flip
and C does not, because <mechanism>; the mechanism predicts <observable>,
which the data does/does not show; I recommend X with these caveats, and the
following remains unresolvable at n=1."*

⚠️ **A conclusion of "the banked data cannot settle this" is a perfectly
acceptable, even likely, outcome.** If that is where the evidence lands, say
so clearly and specify what would settle it. Do not manufacture a
recommendation the data does not support.

---

## Notes for whoever hands this off

* The branch `research/rtk-vio-course-rate-scoring` already exists and
  already contains the rescued raw samples plus the corrections to
  `CLAUDE.md` and the cross-check doc.
* Nothing in this research authorizes any physical run. If it recommends
  one, that run still needs its own predeclaration, corridor scan, dry run
  and explicit per-run operator authorization, exactly as before.
* Standing decision 5 (Phase 2 continuous steering is parked) is unaffected
  by this work either way — this only concerns how a measurement is scored.
