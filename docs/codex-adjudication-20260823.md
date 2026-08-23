# Independent Codex adjudication, 2026-08-23

Two questions were put to Codex (a different model, no stake in the answer)
after a Claude review had already rejected the criterion revision proposal.
Codex was pointed at raw evidence and explicitly told the summaries had already
been found to contain false claims. It was **not** told which answers were
preferred.

Everything below was re-derived locally before being recorded. Where my
re-derivation disagrees, that is stated.

## Question 1 — admit the arc120 capture as the Phase 1 arc?

**Codex verdict: NO.** Recorded and accepted. See
`docs/phase1b-arc-protocol-20260823.md` for what was registered instead.

🗑️ **It refuted my central argument.** I claimed a 4 s window cannot produce the
3 informative chords the criterion needs. Four of five banked captures produced
exactly that inside their first 4 seconds; the arc180 alone did not, because its
fourth fresh arrival never came before the cutoff. **Fragile, not impossible.**

## Question 2 — a defensible prediction-error threshold

Codex derived **0.085 m** (0.150 tolerance − 0.065 sensing floor) **before**
looking at any score, then scored: four of five banked captures pass, the 8 s
straight fails at 0.1418 m. It did not adjust the threshold afterwards, and
explicitly refused to retroactively exclude the failing row.

It also supplied the `3R = 600 ms` stall rule now registered in the Phase 1b
protocol.

## The pairing offset — what changed, and one thing I could not reproduce

🐛 **A real bug of mine.** `scripts/reanalyze_mirror_pairing.py` used mirror
constant **90.00** while `replay_arc_predictability.py` and the project's own
mirror finding use **90.13**. Last night's Claude review spotted the mismatch and
called it immaterial at 0.0006 m of position. For position it is; for the
pairing offset it is not — `dalpha/dK = 0.1214 per degree`, so 0.13 deg moves
alpha by 0.016. **Both the review and I were wrong about it mattering.** Fixed.

Re-derived locally at the corrected constant, 8 informative arc steps:

| | alpha | START | MIDPOINT | END |
| --- | ---: | ---: | ---: | ---: |
| K = 90.00 (old, wrong) | −0.1648 ± 0.0425 | 3.87σ | 15.63σ | 27.39σ |
| **K = 90.13 (correct)** | **−0.1490 ± 0.0425** | **3.50σ** | 15.26σ | 27.02σ |

🔑 **VIO shows the same negative offset independently: −0.1746 ± 0.0427.** That
kills every `toward`-only explanation at once — filtering, extrapolation,
quantisation, and a wrong mirror constant. Whatever causes this is shared by both
heading sources.

⚠️ *(My first spot-check of the VIO figure returned +0.1746. That was my own sign
error: `toward` needs `alpha = −e0/rot` because the mirror flips sign, while a
map-frame heading like VIO needs `+e0/rot`. Codex's sign is the right one.)*

⚠️ **One Codex number I could NOT reproduce.** It reports the lever-arm model
implying a **0.159–0.177 m** offset between the reported position point and the
heading reference, and calls that physically sensible. Working it as
`offset_angle = atan(omega * a / v)` I get **a = 0.036 m (arc180) and 0.039 m
(arc120)** — self-consistent across both arcs, but ~4.5x smaller. I could not
reconstruct Codex's model. **Treat the lever-arm magnitude as unresolved**; the
*mechanism* remains plausible either way, and 3.6 cm is also physically sensible.

## Surviving mechanisms for the negative offset

Codex ruled out, with reasons: `toward`-only low-pass filtering (wrong sign),
quantisation (~0.0025 of alpha at most), and a wrong mirror constant alone
(would need ~5.35 deg of error, incompatible with straight-run net bearings of
88.2–91.3 deg, and VIO does not use the constant).

Left standing:

- **position lag relative to heading**, implying a differential of ~0.65–0.68 s;
- **`toward` is a body heading, not course over ground** — this repo's open
  "item 15" — via a reference-point offset.

⚠️ **No offline test can separate them.** The banked JSON has no per-field
acquisition timestamps, both arcs turn the same direction, and angular rate is
confounded with capture and location.

## Does it matter?

**~7 mm of lateral prediction error per 1 Hz step** (6.5 mm at the corrected
alpha). Against a 0.15 m tolerance this is a scientifically interesting anomaly
and an operationally minor one. It compounds across steps, so it is not nothing —
but it does not justify a mower run on its own.

## The test that would settle it

One bounded daylight run with **synchronised overhead video**: fiducials on the
mower's longitudinal axis, fixed camera, LED/timestamp sync, full raw-report
logging rather than the 100 ms cache, and a sequence of straight plus
**alternating ±angular** pulses — both turn directions, which the banked data
lacks. Fitting position-vs-video, `toward`-vs-video and VIO-vs-video delays
separates staleness from body-heading geometry, and would close item 15.

⚠️ **Not proposed as authorized work.** It needs equipment this project does not
have set up, and the payoff is ~7 mm.
