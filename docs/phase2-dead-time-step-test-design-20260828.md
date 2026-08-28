# Dead-time step test — design for Q1 and Q2, written while the programme is PARKED

**Written 2026-08-28.** Phase 2 continuous steering is **parked** by operator
decision (standing decision 5 in `CLAUDE.md`). This document exists so that *if*
it resumes, the one experiment worth running is already designed and does not have
to be re-derived. **Nothing here is authorized and nothing has been run.**

## What it answers

| | question | why it is decisive |
| --- | --- | --- |
| **Q1** | Is the dead time in the **actuator** or the **observer**? | Actuator lag can be damped. Observer lag cannot — no controller change fixes a heading estimate that structurally reports the past. **This determines whether any fix exists.** |
| **Q2** | How large is it, in seconds? | n = 1 today. Sizes any damping term, and separates "1.2 s, tractable" from "3 s, hopeless at 1 Hz". |

Attempt 5 established that **some** of it is actuator: commanded angular was zero
across interval 4→5 and the mower still rotated **+7.861 °/s**
(`docs/evidence-phase2-steering-attempt5-20260828.json`). It cannot separate the
two, and one interval is not a measurement.

## Half of it is arithmetic, not an experiment — do that first, for free

**Observer lag does not need the mower.** The heading comes from a position chord
over the preceding interval, so it estimates the course at the interval
**midpoint**, not at the decision instant:

```
chord interval            ~1.0 s   (the ~1 Hz bundle)
-> estimate is centred at ~0.5 s before the decision
+ position report staleness ~1.031 s   (documented, docs/p0-beta-release.md lineage)
```

⚠️ **Compute this from banked captures before commanding any motion.** If observer
lag alone already exceeds the ~1 s decision period, **Q1 is answered without a run
and the answer is "observer"** — meaning damping cannot work and the only remaining
option is a faster heading source. Do not spend a supervised run to learn something
five banked evidence files can already tell you.

🔑 **`docs/the-1hz-bundle-is-the-ceiling-20260822.md` is the file to check**: it
found position, `toward` and VIO heading changing on **exactly the same instants,
zero exceptions**. If that still holds, there is no faster heading source on this
hardware and the programme closes on a measurement rather than a failure.

## The measurement trick that makes a ~1 s lag visible at ~1 Hz

You cannot resolve the *shape* of a 1 s transient with a 1 Hz feed. **You do not
need to.**

🔑 **Measure the INTEGRAL, not the rate.** Total angle accumulated after the
command goes to zero is measured accurately at any sample rate, because it is a
difference of two absolute headings rather than a derivative:

```
tau_actuator  ~=  (total rotation AFTER commanding angular 0) / (steady rate BEFORE it)
```

Both terms are chord-derived headings that the existing telemetry already
produces. This is the same reasoning that made the 2026-08-12 arc measurement
trustworthy: total travel and total course change, not per-sample rates.

## Protocol

Open loop. **No steering law, no route, no aim point, no closed loop** — that is
what makes this lower-risk than either steering attempt.

| phase | command | duration | purpose |
| --- | --- | --- | --- |
| A | `linear 400, angular 0` | ~3 s | establish a clean course baseline from a straight chord |
| B | `linear 400, angular -120` | ~3 s | measure the steady rotation rate `omega` |
| C | `linear 400, angular 0` | ~4 s | measure total additional rotation `dtheta` — **this is the whole experiment** |
| D | stop | — | confirmed stop, as every run does |

Then `tau_actuator ~= dtheta / omega`.

⚠️ **Phases B and C must run without stopping in between.** A stop resets exactly
the carryover being measured, which is why this cannot be assembled from existing
probe windows (see below).

**Required recording:** per-sample position, `toward`, and command state at the
existing 100 ms in-window cadence, plus refresh write completions. The 100 ms
sampler reads a cache that only refreshes at ~1 Hz, which is fine — what it buys
is **arrival timestamps to 100 ms**, bounding when rotation started and stopped.

## ✅ The service now exists — `raw_pymammotion_step_response_probe`

**Built 2026-08-28 at the operator's request, after this document was written.**
The paragraph this replaces said the test could not be run with existing
services; that was true, and it is why the service was added.

Why nothing existing could do it: `_motion_refresh_window`'s contract is
explicitly to resend an **identical** command, so it cannot express a step.
`_continuous_refresh_window` **can** resend a changing one, but its only other
caller is `continuous_motion_window` — the closed-loop steering service that
standing decision 5 parks. Assembling the step from two bounded probe windows
fails too: **the stop between them resets exactly the carryover being measured.**

The new service reuses `_continuous_refresh_window` as its one serialized writer
and `_capture_in_window_telemetry` as its sampler and distance guard. It adds
**no controller** — no route, no aim point, no steering law, no corridor-breach
override, no heading state machine.

| | |
| --- | --- |
| phases | `baseline_ms` 3000 → `step_ms` 3000 → `settle_ms` 4000, total capped at 12000 |
| `step_angular_speed` | **±120 or ±180 only** — the measured band, both signs |
| `max_travel_m` | 2.50 default, 3.0 ceiling; trips `travel_abort` and brings the mandatory stop forward |
| containment | `step_path_contained` requires `max_travel_m + 0.50 m` of clearance in **every** direction |
| opt-in | `confirm_step_response_run` **per call** — arming the motion gate is deliberately not sufficient |
| default | `dry_run: true`, sends nothing |

🔑 **Both helper tasks trip the guard if they die.** A dead sampler means the
distance guard is gone; a phase scheduler that dies mid-step leaves a turn
command standing for the rest of the window. Both set `travel_abort`, which stops
the refresh loop and brings the stop forward — the same fail-closed shape as
beta72's `_abort_if_sampler_died`.

⚠️ **Both signs are offered deliberately.** A one-sided step cannot distinguish
rotational carryover from a direction-dependent drivetrain asymmetry. Run both
before believing either.

**Outputs:** `course_series` (per-interval chord courses, each labelled with its
phase by **midpoint** and flagged `informative` against the 0.15 m floor) and
`analysis` (`omega_step_deg_per_s`, `rotation_after_zero_deg`,
`tau_actuator_s`). ⚠️ **The series is the deliverable and the analysis is a
convenience** — this project's standing rule is to verify with per-item records,
not aggregates.

🛑 **Built, tested offline, NOT deployed and NEVER RUN.** 23 offline tests, no
coordinator I/O, no BLE, no mower command. Running it needs a release, a deploy,
a fresh corridor scan and explicit per-run authorization, exactly like every
other physical step in this project.

## Containment

* Off dock, `AREA_INSIDE`, RTK **Fix**, BLE live, blades off, daylight not required.
* ~10 s of driving at ~0.25 m/s on a curving path: budget **~2.5 m** of travel and
  require **>= 3.0 m** of boundary clearance around the start. The 6.0 x 6.0 m
  corridor at **(5.98, -5.24)** used for attempt 5 satisfies this.
  ⚠️ **The path CURVES**, so straight-line corridor reasoning does not transfer —
  re-verify containment against the actual arc before dispatch.
* A distance guard, as on every bounded probe. Stop confirmed. Gate armed only for
  the run, then disarmed and verified from the live API **and** RAW
  `core.config_entries` — re-reading RAW after ~15 s, since HA writes `.storage`
  lazily and a read taken immediately after a disarm can still show the old value.
* ⚠️ **`stop_overshoot_m` is 0.50 m and attempt 5 measured 0.4544 m of post-stop
  creep — 9% margin.** Budget the full 0.50 m and do not raise commanded speed.

## What each outcome would mean, written before any data exists

| result | reading |
| --- | --- |
| `tau_actuator` well **under** the ~1 s decision period | the dead time is dominated by the **observer**; damping cannot fix it; a faster heading source is the only route, and the 1 Hz bundle says there is none |
| `tau_actuator` **comparable to or above** ~1 s | genuine actuator carryover; damping or command-rate limiting is worth designing, with its own predeclared criteria |
| `omega` not reproducible between phases | the rotation quantum is not stable enough to model at all — consistent with the 2.6x stationary spread — and continuous steering should stay closed |

⚠️ **None of these outcomes authorizes a steering run.** Each would feed a new
predeclaration, exactly as `docs/phase2-steering-attempt5-predeclared-20260828.md`
did.
