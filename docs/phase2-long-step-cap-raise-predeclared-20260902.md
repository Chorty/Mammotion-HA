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
