# Phase 2 gate — adopted from the pre-registered plan, not reinvented

**2026-08-23. Documentation only.** The design decisions and gap
reconciliation this week (`docs/phase2-continuous-motion-design-20260823.md`,
`docs/phase2-gap-reconciliation-20260823.md`) flagged that "a Gate 5-style
validation is required before any physical run and is not yet designed."

**That was wrong — it already exists.**
`docs/continuous-motion-feasibility-plan-20260821.md`, written 2026-08-21,
before any of this week's work, has a "Phase 2 pass criteria" section. This
document does not replace it. It confirms the criteria still hold given
what's been measured since, checks whether the prerequisite the plan itself
names ("only after Phase 1 passes") is now met, and records one open
discrepancy the criteria's authors could not have known about.

## The prerequisite is met

> "Only after Phase 1 passes, design a new experimental executor around the
> pure controller."

Phase 1b passed, `go`, 2026-08-23 (`docs/phase1b-go-20260823.md`), against a
repaired criterion and a fresh out-of-sample capture. **The plan's own gate to
start executor design is satisfied.**

## The pre-registered pass criteria, unchanged, checked against this week

From `docs/continuous-motion-feasibility-plan-20260821.md`, verbatim:

- no intermediate stop before the final/abort stop;
- signed heading error and absolute cross-track both trend toward zero;
- no oscillation between saturated ±180 commands;
- cross-track never exceeds 0.20 m and the 0.30 m hard abort never fires;
- motion duty cycle is at least 80%;
- final stop is confirmed and the motion gate is disarmed.

None of these needed changing. They are outcome criteria — what the run must
show — not implementation details, so this week's controller reconciliation
(speed constant, the stall-gap fix, the containment-discipline note) doesn't
touch them.

🔑 **One line already anticipated this week's fifth-gap finding:** "It must
abort on the Phase 0 reasons plus a broken refresh gap derived from Phase 1."
That is exactly `refresh_cadence_stalled`
(`docs/phase2-gap-reconciliation-20260823.md`) — the plan named the
requirement two days before the mechanism that satisfies it was built. The gap
fix is not new scope; it is the plan's own stated abort condition, now
implemented.

## What is still missing before a run could be proposed

**The executor does not exist.** Everything built through today —
`continuous_controller.py` and its reconciliation — is the pure decision
function the plan calls for ("design a new experimental executor **around**
the pure controller"), not the executor itself: no dispatch, no refresh loop,
no live gap tracking, no corridor wiring. That is unchanged from the plan's
own stated scope and is the next implementation step, not a gap in this gate.

## An open discrepancy, flagged rather than resolved

`docs/arcs-work-20260812.md` fit a single point each at angular 180 / 300 /
500 (one pulse per value, no internal replication) to
`w = 0.0659 x angular - 0.638`, quoting r² = 0.9997. That fit predicts:

| angular | Aug-12 fit predicts | measured this week (multiple steady steps) |
| --- | ---: | ---: |
| 120 | 7.270 deg/s | **7.813 deg/s** (+7.5%) |
| 180 | 11.224 deg/s | **9.386 deg/s** (-16%) |

**These do not move in the direction of noise around one line.** This week's
180 reading sits below the old fit; this week's 120 reading sits above it —
consistent with a genuinely flatter slope in the 120-180 band than the
180-500 fit implies, not with two samples scattered around the same
relationship. r² = 0.9997 on three single-pulse points is also weak evidence
of true linearity on its own -- three points nearly always fit a line well,
and no per-point uncertainty was ever computed for that measurement.

⚠️ **Not resolved here, and not urgent.** The chosen v1 architecture corrects
on **measured** heading every ~1 Hz step rather than trusting a commanded-rate
model at all (`docs/phase2-continuous-motion-design-20260823.md`), so neither
fit is on the executor's critical path. It matters only if
`angular_speed_per_heading_degree` or the "no oscillation between saturated
±180 commands" criterion is later tuned assuming one relationship over the
other. Flagged so nobody quietly picks one without noticing the conflict.

## Net effect

The Phase 2 gate is **defined and has been since 2026-08-21**. Nothing this
week required rewriting it. What changed is that its prerequisite (Phase 1
`go`) is now satisfied, its stall-abort line has a concrete implementation, and
one measurement conflict from before this week is now on the record for
whoever tunes the executor's steering constants.

**Next implementation step, per the plan's own words:** design the executor
around the reconciled `continuous_controller.py` -- "one serialized writer
that owns command refresh, feedback decisions, and stop" -- still offline and
dry-run-first, per every prior deploy in this project.
