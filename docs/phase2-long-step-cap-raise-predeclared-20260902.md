# Predeclared proposal — raise the step-probe CLOCK caps, in two phases (2026-09-02)

**Offline. No run is authorized by this document.** It proposes a code change and
a two-phase measurement, each phase separately authorized. Written before any run
at either configuration exists.

## 1. What this fixes

Criterion 2a is currently a coin flip. `half_diff = onset/k + noise` with
`noise ~ sd·sqrt(2/k)`, measured `sd = 1.445 °/s` and worst onset contamination
`10.43 °/s`:

| step | k | P(2a pass) |
| --- | --- | --- |
| 7 s (both banked +180 runs) | 4 | 13.9% |
| **15 s (today's ceiling)** | **8** | **60.7%** |
| 21 s | 11 | **81.5%** |
| 24 s | 12 | 85.8% |
| 27 s | 14 | 91.7% |

🗑️ **The earlier claim that the yard cannot hold a long enough step was WRONG** —
it compared against the largest axis-aligned square where the gate requires a
disk. Corrected in
`docs/findings-2a-cannot-be-fixed-by-a-longer-step-20260901.md` §4.

## 2. 🔑 The exposure bound does NOT need to move

Maximum inscribed radius in "Backyard Right" is **5.913 m** (cross-checks against
the 5.9039 m measured at the 2026-09-01 run's own start).

| `max_travel_m` | required radius | margin | ratio | step reachable | P(2a) |
| --- | --- | --- | --- | --- | --- |
| **4.5 (UNCHANGED)** | 5.00 m | 0.913 m | **1.18x** | ~21 s | **81.5%** |
| 5.0 | 5.50 m | 0.413 m | 1.08x | ~24 s | 85.8% |
| 5.41 | 5.91 m | 0.003 m | 1.00x | ~27 s | 91.7% |

🔑 **Proposal: raise ONLY the clock caps and leave `max_travel_m` at 4.5.**
That takes 2a from 60.7% to **81.5%** with **no increase in the distance the
mower may travel open loop** and a comfortable 1.18x corridor margin. The
remaining 10 points would cost a real exposure increase and nearly all the
margin; they are not worth it.

**Changes proposed:**

| | from | to |
| --- | --- | --- |
| `step_ms` ceiling | 15000 | **22000** |
| `_STEP_RESPONSE_MAX_TOTAL_MS` | 23000 | **30000** |
| `max_travel_m` ceiling | 4.5 | **4.5 — UNCHANGED** |
| `linear_speed` | [300, 400] | unchanged |

## 3. 🚨 Why this needs TWO phases, and why phase B cannot be sized yet

**The sustained speed at `linear_speed` 300 has never been measured.** Every
figure for it is one 4 s ramp-inclusive sample (0.116 m/s) scaled by a 1.37x
ratio observed at 400. The longer the window, the more that error matters:

Planning a 31 s window against `max_travel_m` 5.0:

| true sustained speed | travel | outcome |
| --- | --- | --- |
| 0.12 m/s | 3.75 m | ok |
| 0.16 m/s (the estimate) | 5.00 m | ok, exactly at budget |
| **0.18 m/s** | 5.62 m | **guard trips — run censored** |
| 0.22 m/s | 6.88 m | guard trips |

🔑 **A +19% speed error alone censors the run**, and the estimate has no error
bar at all. ⚠️ **Sizing the window from an unmeasured speed is exactly the mistake
that produced the withdrawn ~2.5 m travel figure on 2026-09-01.** It must not be
repeated at four times the window length.

### Phase A — measure sustained speed at linear 300 (cheap, low exposure)

`baseline 3000 / step 1000 / settle 4000` at **linear 300, `step_angular_speed`
120, `max_travel_m` 2.5**, in the existing verified corridor. Needs **no code
change** — every value is inside today's schema. Report cumulative path travel
over the full window divided by elapsed, the same metric the guard uses.

**This also retires a second unknown**: `(linear 300, angular ±180)` is an
operating point never exercised, and the onset/scatter constants in §1 were all
measured at linear 400. Phase A is the first evidence either transfers.

### Phase B — the long step, sized from Phase A's measurement

Only after Phase A returns a number. Window sized so that travel fits
`max_travel_m` **at the measured speed plus a 25% margin**, not at the point
estimate. If Phase A shows 300 sustains faster than ~0.17 m/s, **phase B may not
fit at all at `max_travel_m` 4.5** — and that is a legitimate outcome, not a
reason to raise the bound.

## 4. Scoring — unchanged, and not to be edited afterwards

Rule E-VIO exactly as shipped. `tau` exists only when 2a passes. Dark VIO refuses
with `vio_not_live_throughout`. A travel-guard trip is a **FAIL**, not a smaller
number. ⚠️ **A 2a PASS at 81.5% is still one draw** — report the realised
interval count and cadence with any verdict, since k is what the whole argument
rests on.

## 5. 🛑 What is NOT proposed

- **No `max_travel_m` increase.** The distance exposure bound does not move.
- **No union of "Backyard Right" and "Backyard Hill".** The union offers a 7.007 m
  radius against 5.913 m, but crossing between areas changes `zone_hash`, which
  raises **`zone_hash_changed`** (`services.py:4783`) — a guard that exists to
  catch the mower having left the area it was authorized in. It would abort the
  run mid-window, and relaxing it is a containment decision, not a config tweak.
  **Backyard Right alone is sufficient**, so the union buys ~4.6 percentage
  points in exchange for weakening a containment guard. Rejected.
- **No resumption of Phase 2 continuous steering.** Standing decision 5 is
  untouched; this repairs a parked instrument.

## 6. Preconditions for either phase

Explicit per-run operator authorization, daylight throughout (dark VIO is
unscoreable by design), docked-and-charged battery, a fresh corridor scan at the
live position, and the gate disarmed and verified from the live API **and** RAW
afterwards. ⚠️ The host must first be returned to a known-good build — see
`docs/SESSION-STATE-20260901-2000.md` §6.

---

## 7. AMENDED 2026-09-02 after adversarial review of beta97 — the numbers move, the recommendation does not

Two findings from the beta97 review change this proposal's own arithmetic. ⚠️ Both
were **unverified** (their verifiers died on a spend limit) and were confirmed by
hand instead.

### 7.1 The containment gate was understated, and it now binds the window

`step_path_contained` computed `max_travel_m + 0.50` only, **assuming the distance
guard works** — but a documented latched-position mode makes the guard a no-op and
lets the window run to the wall clock. Fixed in `a659535d`: the requirement is now
`max(travel bound, clock bound)` where `clock bound = 0.0007 × linear_speed ×
window_s`. 🔑 **The clock bound is what limits the window now, not `max_travel_m`.**

In "Backyard Right" (inscribed radius **5.913 m**):

| linear | clock speed | max window | max step | k | P(2a) | aliases at |
| --- | --- | --- | --- | --- | --- | --- |
| **300** | 0.210 m/s | **28.2 s** | **21.2 s** | 11 | **81.5%** | ≥17.0 °/s |
| 400 | 0.280 m/s | 21.1 s | 14.1 s | 7 | 50.5% | ≥25.5 °/s |

🔑 **§2's 24 s and 27 s rows (85.8% / 91.7%) are WITHDRAWN — they do not fit.**
🔑 **Linear 300 is not a preference any more, it is required**: at 400 the corridor
caps the step at 14.1 s and 2a stays a coin flip.

### 7.2 The E-VIO half rate aliases past 180° per half — a second, independent cap

Each half rate is an endpoint difference through `normalize_degrees`, which wraps
to [-180, 180), so a half accumulating ≥180° silently aliases and flips sign.

⚠️ **The reviewer's claim that the shipped 15 s cap "sits 2-5% under" this is
WRONG** — at 15 s the half is 7.5 s and aliasing needs ≥24 °/s against a measured
~12 °/s, a 2x margin. **But it is a real cap on longer steps**: at the 21.2 s step
above the threshold is 17.0 °/s, a 1.4x margin, and the withdrawn 27 s row sat at
13.3 °/s — **inside measurement noise of the observed rate.** That row was unsafe
for two independent reasons.

### 7.3 Revised proposal

| | from | to |
| --- | --- | --- |
| `step_ms` ceiling | 15000 | **22000** (unchanged from §2 — 21.2 s fits under it) |
| `_STEP_RESPONSE_MAX_TOTAL_MS` | 23000 | **29000** (was 30000; 28.2 s is the real limit) |
| `max_travel_m` ceiling | 4.5 | **4.5 — still UNCHANGED** |

**The run is now fully specified: `baseline 3000 / step 21000 / settle 4000`
(28000 ms) at `linear_speed` 300, `max_travel_m` 4.5, in a corridor with ≥5.88 m
of clearance.** Expected P(2a pass) **81.5%**, aliasing margin 1.4x.

🔑 **Three independent constraints — corridor geometry, VIO aliasing, and the
noise floor — all land on ~21 s and ~81%.** That agreement is the strongest reason
to believe the number.

⚠️ **Phase A is unchanged and still comes first.** Every figure above uses the
sizing constant `0.0007 × linear_speed` for containment, which is deliberately
conservative, but the *travel guard* still compares against real measured travel —
and the sustained speed at 300 remains unmeasured. Nothing here removes that
dependency.

---

## 8. 🗑️ WITHDRAWN 2026-09-03 — Phase A measured the speed and this proposal does not survive it

**Phase A ran and the number came back 39% above the extrapolation this proposal
was sized with.** Read
`docs/evidence-phaseA-linear300-sustained-speed-20260903.md`.

| | assumed here | measured |
| --- | --- | --- |
| sustained speed at linear 300 | 0.16 m/s | **0.223 m/s** |
| travel over the proposed 28 s window | ~4.5 m | **5.90 m** |
| against `max_travel_m` 4.5 | fits | **DOES NOT FIT** |

The largest window that fits at `max_travel_m` 4.5 is **21.7 s**, leaving a 14.7 s
step at **50.5%** — worse than the 60.7% today's unchanged 15 s cap already
delivers. 🔑 **So the cap raise buys nothing and is withdrawn**, exactly as §4 of
the Phase A predeclaration registered in advance. **`max_travel_m` stays 4.5.**

⚠️ **The specific trap, because it has now caught this project three times in
eight days:** the whole-window speed Phase A measured (0.1574 m/s) agreed with the
extrapolation to **1.6%** — and was still wrong for sizing, because both include
ramp-up. **Separate the ramp before sizing any window.** A ramp-inclusive average
is only valid for a window of the same length.

**What survives:** §7.1's clock-bound containment fix is real, shipped, and
independent of this — it was a genuine gap regardless. The convergence table in
§1 also stands as a description of the statistic; what fails is the belief that a
long enough step is reachable here.

**What remains open, unchanged:** criterion 2a needs a different statistic, not a
longer window. Offline work, no mower.
