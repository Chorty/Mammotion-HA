# PREDECLARATION — the Issue 1 step 2 queue-start timing measurement

**Written 2026-09-11 (local EDT), before any leg was dispatched and while
`motion_dispatch_timing_report` read `sample_count: 0` on the live host.**
Verified at 2026-09-12T02:51Z: 68 mammotion services registered,
`motion_dispatch_timing_report` present, gate `enabled: false`,
`real_motion_allowed: false`, `blockers: ['experimental_motion_disabled']`.

Companion to `docs/plan-post-20260910-session-issues.md` issue 1 and to
`docs/findings-clicktopath-reliability-4m-repeat-20260910.md` §1.5 / §1.5.1.

**What this predeclaration governs.** Step 2 only: *measure* the distribution of
enqueue→started waits that `_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS` bounds.
🛑 **It does not authorize step 3.** No criterion below permits changing
`_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS` or `motion_refresh_interval_ms`. The
most a "raise is defensible" verdict buys is the right to *propose* a number to
the operator, who decides separately.

---

## 0. Why the numbers below are fixed now

The bound was **unfalsifiable from telemetry** until beta104: a refusal recorded
only that the wait exceeded 2.0 s, and a successful pulse recorded nothing at
all. That is a censored observation, and it cannot separate "typical waits are
50 ms and 2.0 s is a rare pathology" from "typical waits are 1.8 s and the bound
is marginal". Those two worlds demand **opposite** actions, and choosing the
threshold after seeing the histogram would let either one be argued. Hence this
file, committed before the first dispatch.

---

## 1. Three properties of the instrument, stated before the data exists

These are read off `custom_components/mammotion/services.py` as it stands, and
they change how the output must be handled. They are **not** defects to fix in
the measuring session.

1. 🚨 **`worst_wait_fraction_of_budget` mixes two different budgets and must not
   be quoted as reported.** `_summarise_motion_dispatch_timings` computes
   `max(waits) / (min(budgets) * 1000)`. But `queue_budget_seconds` is **2.0**
   for an ordinary pulse and **5.0** (`_BLE_MOTION_WRITE_TIMEOUT_SECONDS + 1.0`)
   for an `emergency_stop`. A long wait belonging to a 5.0 s emergency stop is
   therefore divided by 2.0 and reported as though it had nearly exhausted the
   ordinary bound. ✅ **Every figure in §3–§5 is computed by hand over the raw
   `samples` array, filtered to `queue_budget_seconds == 2.0`.** The reported
   field is recorded in evidence for completeness and is not a criterion input.
2. ⚠️ **`outcomes` is not a complete failure census.** A GATT write that fails
   *after* the queue slot starts propagates out of
   `_send_ble_motion_command_confirmed` without recording a sample — only
   `completed`, `queue_start_timeout` and `cancelled_before_start` are ever
   written. So a low `queue_start_timeout` count does **not** by itself mean the
   session went well; it must be cross-checked against each leg's own
   `command_result` and the `queue_diagnostics` snapshots.
3. ⚠️ **The history is `maxlen=500` and silently drops the oldest.** At
   `motion_refresh_interval_ms: 200` a handful of legs can approach that.
   ✅ **Snapshot `motion_dispatch_timing_report` after *every* leg and keep each
   snapshot separately**, so a drop is visible as a discontinuity rather than
   silently truncating the population.

---

## 2. Minimum sample count — and why this number

**The measurement is only interpreted at `n ≥ 120` samples with
`queue_budget_seconds == 2.0`, drawn from `≥ 4` distinct legs, of which `≥ 2`
included a turn or calibration-drive phase.**

Reasoning, in the order it constrains:

- **`n ≥ 12` is a hard floor imposed by the code itself.** `pct()` picks
  `index = round(fraction * (n - 1))`. For `n ≤ 11`, `round(0.95 * (n-1))`
  lands on `n-1`, so **`p95` is literally the same order statistic as `max`**
  and carries no independent information. Below 12, no percentile claim is made
  at all.
- **`n ≥ 120` is what makes a *rate* statement possible.** By the rule of three,
  observing zero `queue_start_timeout` events in 120 samples bounds the true
  per-dispatch timeout rate at roughly **2.5%** with 95% confidence. That is the
  weakest interesting claim — anything looser cannot distinguish "this does not
  happen" from "we did not run long enough to see it". It also puts ~6 samples
  above the p95 index, so the tail rests on more than one observation.
- **`≥ 4 distinct legs`, because samples within one leg are not independent.**
  Queue contention is a state that persists across the pulses of a single leg; a
  single congested leg could supply 120 correlated samples and look like a
  population. Legs are the independent unit here, not pulses.
- **`≥ 2 legs with a turn or calibration phase`, because that is where the
  record was blind.** §1.5.1 establishes that `_vio_segment_calibration_drive`,
  `_raw_pymammotion_turn_to_heading` and `_vio_turn_to_heading` are phases of the
  same service run on every leg, and leg 4 died 0.29 m into a 4.0 m leg — a live
  candidate for the calibration drive. A linear-only sample set would repeat the
  2026-09-10 blind spot.

🔑 **n = 2 is what exists today (legs 7 and 8) and is why this session exists.**

---

## 3. What would JUSTIFY raising the 2.0 s constant

**All three must hold**, computed over `queue_budget_seconds == 2.0` samples:

1. `p95(queue_wait_ms) ≥ 1000 ms` — one pulse in twenty comes within 2× of
   refusal, so the bound is doing work in *normal* operation rather than
   catching an exceptional event; **and**
2. a recomputed worst-wait fraction `≥ 0.75` of the 2.0 s budget on **≥ 2
   distinct legs** — so one contention spike cannot carry the verdict; **and**
3. `≥ 1` `queue_start_timeout` outcome actually occurred in the session — the
   failure is reproducible, not merely theorised.

**Verdict text if met:** *the bound is routinely marginal; a raise is defensible
and a specific number may be proposed.* 🛑 Still not changed here — step 3 is a
separate operator decision.

---

## 4. THE FALSIFIER — what would mean the constant is NOT the problem

**All three holding means the 2.0 s value is exonerated and step 3 should not be
attempted at all:**

1. `p95(queue_wait_ms) ≤ 250 ms`; **and**
2. recomputed worst-wait fraction `≤ 0.40` (i.e. the worst observed ordinary
   wait stayed under 800 ms, leaving ≥ 2.5× headroom); **and**
3. `n ≥ 120` from `≥ 4` legs, per §2.

**Verdict text if met:** *the 2026-09-10 refusals were a transient pathology, not
a marginal bound. Raising the constant would treat the symptom.* The cause then
lies elsewhere — a reconnect or cooldown episode, queue work from another
subsystem, or a link-layer stall — and the next action is identifying what
**occupied** the queue during those specific episodes, using the
`queue_diagnostics` snapshots, not moving a number.

🚨 **The strongest form of this falsifier:** if a `queue_start_timeout` **does**
occur while `p95` stays `≤ 250 ms`, that is the most decisive result available
and it overrides §3 even if §3's clauses 2 and 3 are met. A tight body with a
multi-second outlier is a **bimodal** distribution — an episode, not a tail.
Raising 2.0 → 4.0 against an episodic occupant merely lengthens the wait before
the identical refusal, while degrading the guard for every healthy pulse. In
that case the recorded verdict is *"the constant is not the problem; find the
occupant"*, and §3 is not invoked.

---

## 5. What would make the measurement INCONCLUSIVE

Any one of these, and **no verdict is recorded, the constant stays at 2.0**, and
the raw per-item data is banked for the next attempt:

- `n < 120` qualifying samples, or fewer than 4 contributing legs, or no leg
  that ran a turn/calibration phase (§2 unmet);
- `> 50%` of samples carry `emergency_stop: true` or `queue_budget_seconds
  != 2.0` — the population is stops, not the motion pulses the bound governs;
- the session aborted for a reason **unrelated** to queue timing (transport
  loss, a VIO refusal, a containment refusal, battery) before §2 was reached —
  the sample set is then truncated by a mechanism that may correlate with
  queue state;
- a `history_capacity` drop is detected between per-leg snapshots and the
  dropped samples cannot be reconstructed from the earlier snapshots.

⚠️ **"Inconclusive" is a real outcome and is to be reported as one.** It is not
a licence to interpret the numbers anyway with a caveat attached.

---

## 6. Leg plan — sized for pulses, not metres

Exact coordinates are appended at run time, **after** a fresh corridor scan
against the map, because they depend on where the mower actually is. The
**sizing rule** is fixed now:

- 🔑 **Contention accumulates per pulse, not per metre.** Legs 7 and 8 both
  failed after several pulses had already succeeded. A short leg produces the
  same `motion_refresh_interval_ms: 200` queue traffic per pulse as a long one,
  at a fraction of the exposure.
- **≥ 4 short legs** (target **0.8–1.5 m** each), near the dock, rather than few
  long ones. Each leg's calibration drive + turn + linear phases are expected to
  contribute tens of dispatches, so 4–6 legs should clear `n ≥ 120`.
- 🛑 **This is not the 4.0 m aligned-start series.** That series aborted on its
  own predeclared rule at 3 of 5 scored and stays aborted. Nothing here reopens
  it, and no leg in this session is scored for landing accuracy — landings are
  recorded but are not the measurement.
- Each leg aimed at the **live `map_facing_degrees`**, never a fixed compass
  bearing, per the 2026-09-10 finding.
- Candidate positions screened with `scripts/plan_aligned_leg.py`, whose BLE
  coverage constraint (`scripts/ble_coverage_map.py`, reject below −76 dBm
  average) is respected as-is.

**Protocol, no exceptions:** daylight; operator present with explicit go/no-go
immediately before each dispatch; fresh corridor scan, with a physical tape
measurement requested on any corridor tighter than a couple of metres;
`docs/accepted-profile.json` sent verbatim and verified key-by-key; gate
verified disarmed from the live API **and** raw `core.config_entries` at session
end.

**Per-item recording:** after each leg, the full `samples` array from
`motion_dispatch_timing_report` is saved — not the percentiles — into
`docs/evidence-queue-timeout-measurement-<date>.json`, tagged by leg, alongside
that leg's `command_result` and any `queue_diagnostics` snapshot. Net figures in
this project have already hidden a 27° turn reversal and a live BLE link.

---

## 7. Why the 2026-09-11 session did not run

Recorded here so the next session does not re-derive it. Live at
2026-09-12T02:51Z: sun **35.5° below the horizon**, `camera_brightness: dark`,
`vio_brightness: 0`, `vio_tracked_features: 0` — night is a closed standing
decision and `vio_active` refuses the vector executor regardless. The mower was
**off-dock** at "Backyard Right", `paused`, not charging, **48%** and falling
~4.3%/h, `real_motion_ready: off`, RTK degraded to `single`. No dispatch was
made and no result is simulated or inferred in place of one.

---

## 8. ✏️ Amendment, same session — the BLE survey can supply these legs

Raised by the operator before any leg ran: map BLE strength around the yard, and
drive around to gather more points. Both compose with this measurement, and the
combination is *better* than running them separately — a survey drive is made of
exactly the motion dispatches §2 needs, so one daylight session yields both.

**The §2–§5 criteria are unchanged and are not reopened by this.** What changes
is only where the legs go: a survey pattern rather than four short legs clustered
near the dock. It qualifies for this measurement **if and only if** it still
satisfies §2 in full — `n ≥ 120` samples at `queue_budget_seconds == 2.0`, from
`≥ 4` distinct legs, `≥ 2` of which ran a turn or calibration-drive phase.

🚨 **Survey value never overrides a safety constraint.** A cell being unsampled
is a reason to *want* data there, which is precisely the pressure that the
corridor scan, the map check, the tape measurement on tight corridors and the
existing `--min-rssi-dbm -76` rejection exist to resist. An unsampled cell is
unsampled *and* unverified; it does not become safer by being interesting.

**Survey design target, derived from the 96 h rebuild (2026-09-12T02:55Z,
27 440 matched samples, fit RMS 0.0000 m):** the limiting quantity is samples
**per cell**, not samples in total.

- 88.7 % of all samples sit in the single 1 m cell at the dock. The yard outside
  it holds ~3 100.
- Of 800 cells in the sampled bounding box (16 m × 50 m), **186 hold any sample
  (23.2 %), 60 hold ≥ 10, and 16 hold ≥ 30**.
- Within one cell the RSSI standard deviation is a median **5.5 dB** (median
  spread 26 dB); between cells the means vary by only **7.3 dB**. The ratio is
  **1.31** — position currently explains barely more variance than standing
  still does, which is why a single pass through a cell cannot characterise it.
- With 5.5 dB of at-point noise, a cell mean has standard error `5.5/√n`.
  Resolving the ~7.3 dB spatial signal to ~2 dB needs **n ≥ 10 per cell**.

✅ **Target: ≥ 10 samples in each surveyed 1 m cell**, which means traversing
slowly or pausing, not passing through once. ⚠️ **Driving fixes coverage; it does
not fix the noise ratio** — only repeat sampling per cell does.

⚠️ **`scripts/ble_coverage_map.json` was deliberately NOT overwritten** with the
96 h rebuild. `_estimate_rssi` in `scripts/plan_aligned_leg.py` is an unweighted
mean over a radius, so folding in 24 331 dock samples would swamp every estimate
near the dock and could silently move planner verdicts. Replacing the planner's
input is its own decision, made deliberately and not as a side effect of a
survey.

---

## 9. 🚨 §8 IS WITHDRAWN IN PART — corrected 2026-09-12, still before any data

§8 said a BLE survey drive could supply this measurement's legs. **The part
claiming survey legs may feed the §2 population is withdrawn.** It would have
corrupted the measurement in a way worth naming precisely, because the mistake
is subtle and the discipline exists to catch exactly this.

A survey drive deliberately enters cells where coverage is *weak* — that is its
purpose. §3 justifies raising the constant on a high p95 and a worst-wait
fraction ≥ 0.75. Feeding survey legs into that population would inflate both
numbers by construction, and the measurement would then "justify" loosening a
safety constant **on a sample selected for being unrepresentative**. That is not
choosing the threshold after seeing the data; it is choosing the *sample* to fit
the threshold. Same failure, different hat.

✅ **What stands from §8:** the survey is worth doing, the two tasks share a
daylight window, and the per-cell targets (≥ 10 samples per 1 m cell, from the
5.5 dB within-cell vs 7.3 dB between-cell finding) are unchanged.

✅ **What replaces the withdrawn part:**

- The §2 population is **only** legs in known-good coverage, with
  `plan_aligned_leg.py`'s `--min-rssi-dbm -76` rejection enforced.
- Survey legs are tagged **`survey: true`** in the evidence file and **excluded
  from the §2 population**. They are still recorded per-item — they are honest
  data and useful for coverage — they simply do not feed these criteria.
- A session short of `n ≥ 120` from good-coverage legs alone is **inconclusive**
  per §5. It is **not** topped up with survey legs.
- Consequently the survey runs as its **own session, after** the measurement is
  banked, not interleaved with it.

§2–§5 thresholds are unchanged and are not reopened by this correction.
Sequencing and the RF freeze that follows from it:
`docs/plan-queue-measurement-then-ble-20260912.md`.
