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

---

## 10. 🚨 AMENDMENT — the sample population was contaminated. Restated per class,
## still at `sample_count: 0`.

Raised by session `mammotion-ha-0f` (which built the instrument) and
independently confirmed by `mammotion-ha-2b`. **I verified every link myself by
grep-and-read before accepting any of it**, per this repo's rule that a peer
summary is a candidate, not a finding. All of it holds.

### 10.1 The defect, verified

Every **refresh resend** records a timing sample **indistinguishable** from a
pulse-opening dispatch:

- the three resend sites (`services.py` 13699, 15893, 17855) all call
  `_send_manager_command_with_args(coordinator, "send_movement",
  prefer_ble=prefer_ble, ...)` — two via `functools.partial`, one via
  `_resend_turn`;
- `"send_movement"` is in `RAW_PYMAMMOTION_MOTION_COMMANDS` (line 157);
- `_send_manager_command_with_args` routes to
  `_send_ble_motion_command_confirmed` on
  `prefer_ble and command in RAW_PYMAMMOTION_MOTION_COMMANDS`, and
  `docs/accepted-profile.json` sets **`prefer_ble: true`**;
- the recorded fields are therefore identical: `command="send_movement"`,
  `is_stop=False` (kwargs are never all-zero for a drive),
  `emergency_stop=False`, `queue_budget_seconds=2.0`.

**Ratio, from the code not an estimate.**
`max_refreshes = max(int(duration_seconds / interval_seconds), 0)` with the
accepted profile's `motion_refresh_interval_ms: 200`:
`int(1300/200) = 6` refreshes per linear pulse, `int(1500/200) = 7` per turn
pulse. So **6:1 and 7:1 — roughly 86 % of samples are refreshes.**

🚨 **The two classes are asymmetric in consequence, which is what makes this
misleading rather than merely noisy.**
A pulse-open failure aborts the leg (`command_failed`). A refresh failure is
**swallowed** by `_motion_refresh_window`'s `except Exception` — it sets
`refresh_error` and breaks, because "a half-refreshed window is a shorter drive,
never a runaway one" (verified verbatim, ~line 6820). **A harmless swallowed
refresh timeout lands in `outcomes` identically to one that killed a leg**, and
that census is what §3 would use to justify loosening a safety constant.

🚨 **§5's tripwire cannot catch it.** "> 50 % carry `emergency_stop: true` or
`queue_budget_seconds != 2.0`" passes cleanly on a 86 %-refresh population,
because refreshes are non-stop, non-emergency, budget 2.0.

### 10.2 ✏️ Correcting the peer on one point: this IS a criteria change

`mammotion-ha-0f` framed this as a sizing issue and wrote that "§3/§5 do not
depend on [§6], so this is not a criteria change." **That is wrong, and in the
direction that matters.** §3 and §5 state their thresholds on
`p95(queue_wait_ms)` computed over *"`queue_budget_seconds == 2.0` samples"* —
a population the defect shows is 86 % refreshes. So the **criteria themselves**
were contaminated, not only the leg count. Refreshes fire on a fixed 200 ms
cadence *inside an already-open window*, which is a systematically different
queue state from a pulse-open fired after a stop-settle-measure gap; pooling
their distributions would have made §3's p95 a statement about refresh latency.

✅ **All of §2–§5 are hereby restated over the `pulse_open` class only.** The
numeric thresholds (p95 ≥ 1000 ms / ≤ 250 ms, fractions 0.75 / 0.40) are
**unchanged** — only the population they are computed over is corrected.

### 10.3 Classification, predeclared — no deploy required

The classes are separable from what beta104 already records, so **Phase 1 does
not wait on a beta105.** `_utc_timestamp()` is
`datetime.now(UTC).isoformat()` — microsecond resolution.

**Rule, fixed now:** sort a leg's samples by `recorded_at_utc`; a gap
**> 500 ms** from the previous sample starts a new burst; the **first sample of
each burst is `pulse_open`**, every later sample in it is `refresh`.

**Why 500 ms separates them with margin on both sides:** within a burst the
cadence is fixed at 200 ms *measured from window start*, and a write slower than
the interval yields a **zero** sleep — so within-burst gaps span 0–200 ms, never
more. Between bursts sit the caller's mandatory stop, a settle, and a position
measure gated by the device's **~1 Hz** telemetry bundle, so ≥ 1 s.

✅ **Mandatory cross-check, not optional:** for each burst, the inferred refresh
count **must equal** that pulse's own
`command_result["motion_refresh"]["refresh_commands_sent"]`, which the executor
already reports independently. **Any leg where they disagree is excluded from
the population and the disagreement is reported per-leg.** This is what makes
the classification verified rather than assumed — the instrument's own
bookkeeping adjudicates it.

### 10.4 Sizing, restated per class — operator decision taken 2026-09-12

🔑 **The two claims are separated, because they need different n.**

**(A) The §3/§4 verdict — p95 and headroom: `n ≥ 40` `pulse_open` samples, from
`≥ 4` distinct legs, `≥ 2` with a turn or calibration phase.**
This is what §3 and §4 actually turn on. ⚠️ At n = 40 the p95 index is
`round(0.95 × 39) = 37`, leaving only 2 samples above it — **a coarse estimate
with wide uncertainty, and stated as such.** It still discriminates because
§3 and §4 are deliberately **4× apart** (1000 ms vs 250 ms): a noisy p95 lands
clearly in one camp, or in neither — and "neither" is already §5's inconclusive,
which is a legitimate outcome.

**(B) The rate claim: `n ≥ 120` `pulse_open` samples.** Rule of three gives a
95 % upper bound of ~2.5 % on an unobserved timeout rate. 🛑 **Explicitly
DEFERRED unless the session reaches it** — no rate is quoted below 120.
At n = 40 the honest bound is **3/40 ≈ 7.5 %**, and that is the number to state
if a rate is mentioned at all.

**Leg count.** ~0.29–0.38 m per linear pulse (1.3 s at the measured sustained
0.223–0.295 m/s), so a 0.8–1.5 m leg is ~2–5 linear pulses plus ~2 calibration
and ~3 turn pulses ⇒ **roughly 7–10 pulse-opens per leg**. So **(A) ≈ 6 legs**
and (B) ≈ 12–17. ⚠️ **These leg counts are CONSERVATIVE**: the per-pulse
distance uses *sustained* speed while the pulse includes its ramp, so real
distance per pulse is lower and pulses per leg higher — CLAUDE.md's standing
"separate the ramp before sizing any window" trap, cutting in the favourable
direction here.

🗑️ **§6's "4–6 legs should clear n ≥ 120" is withdrawn.** It was only ever
reachable through refresh inflation. Pooled counting would have hit 120 in
**~2 legs**, declaring success on two legs' worth of independent information
while §2's own "≥ 4 distinct legs" rationale did all the real work.

### 10.5 🚨 A banked prior that argues AGAINST my own §4 prediction

`_motion_refresh_window`'s own comment records, from all 98 refresh writes of
the five real runs of 2026-08-09: **write latency p50 225.6 / p90 572.0 /
p95 1029.2 / max 2014.0 ms, with 59 % of writes exceeding the 200 ms interval.**

That is *write* duration, not queue-start wait — a different quantity, and it
does not transfer directly. But the queue is **serialized**, so a dispatch
enqueued behind an in-flight write of ~1 s inherits that wait. 🔑 **This
materially raises the prior that §3's "p95 ≥ 1000 ms" could fire** — and for a
reason that has nothing to do with the 2.0 s bound being miscalibrated.

✏️ **§2 of the plan recorded my prediction that §4 (the constant is fine) was
the likely outcome. This prior argues the other way, and I am recording that
before the data rather than after.** If §3 does fire, the honest reading may be
neither "raise the constant" nor "the constant is fine" but **"write latency on
this link is the binding constraint"** — a third answer that neither §3 nor §4
anticipates, and which §5's inconclusive branch should absorb rather than force.

### 10.6 Two smaller items, recorded for the evidence file

⚠️ **`write_ms` is stamped late.** `_dispatch` calls `started.set()` and then
runs `_ble_link_liveness` — a plain `def` with no awaits — before its first
yield, so `started_monotonic` is taken *after* that snapshot. Magnitude is
sub-millisecond against a 2000 ms budget and immaterial, **but the direction
biases `queue_wait_ms` upward**, which is the direction favouring §3. Note it in
the evidence file.

🚨 **"Contention accumulates per pulse, not per metre" is an ASSUMPTION, not a
finding.** It originates with `mammotion-ha-0f`, appears in
`docs/prompt-issue1-step2-measurement.md`, and is what §6 used to justify short
legs near the dock. ⚠️ **Short legs near the dock are the strongest-link
regime**, while legs 7 and 8 failed at −60 to −67 dBm roughly 8 m out. So this
sizing choice may systematically sample the regime *least* likely to reproduce
the failure. It is kept — a measurement of normal operation is what §3/§4 ask
for — but **a null result must not be read as "the failure does not happen",
only as "it did not happen in the strongest-link regime."**

---

## 11. AMENDMENT, continued — four operator refinements. Still `sample_count: 0`.

### 11.1 🚨 §5's third clause is REPLACED — the old tripwire was untestable

**Withdrawn:** *"`> 50 %` of samples carry `emergency_stop: true` or
`queue_budget_seconds != 2.0`."*

🔑 **It could not have caught the very defect that motivated this amendment.**
Refreshes are non-stop, non-emergency, budget 2.0 — the clause passes cleanly on
a 86 %-refresh population. A falsifier that cannot fire on the actual
contamination is decoration. §10.1 observed this; this section actually fixes it.

**Replacement — class-based, and it fires on exactly that failure.** Any one of
these makes the measurement **inconclusive**:

1. **`pulse_open` share outside 8 %–35 % of classified samples.** The code
   predicts ~1 pulse-open per 7 samples for linear (6 refreshes) and 1 per 8 for
   turn (7) — i.e. **12.5 %–14.3 %** expected. A share near 100 % means the
   classification collapsed and every sample was labelled one class (the
   original defect); a share near 0 % means burst boundaries were missed
   wholesale. The band is deliberately wide so only a *structural* failure trips
   it.
2. **Unclassifiable bursts > 20 % of all bursts** (§11.2).
3. **`pulse_open` count after exclusions < 40** (§11.4).
4. **Any leg whose per-burst cross-check disagreements cannot be reconciled**
   from that leg's own `command_result`.

### 11.2 ✏️ Exclusion is per BURST, not per leg — correcting §10.3

§10.3 said a *leg* whose counts disagree is excluded. **Too coarse, and it
discards good data.** Restated:

✅ **If a burst's member count does not equal that pulse's
`motion_refresh.refresh_commands_sent`, the burst is marked `UNCLASSIFIABLE` and
excluded from BOTH classes.** The boundary is never guessed. Other bursts in the
same leg are unaffected and still count.

✅ **The excluded burst count and excluded sample count are reported alongside
`n`, always** — not as a footnote. An `n` quoted without its exclusion count is
not a reportable figure in this measurement.

### 11.3 `write_ms` bias is a REQUIRED evidence field, not a remark

§10.6 said to "note it in the evidence file". Made concrete:

✅ **`docs/evidence-queue-timeout-measurement-<date>.json` MUST carry a
top-level `known_biases` object**, present even when empty of anything else,
containing at least:

```
"known_biases": {
  "queue_wait_ms_biased_upward": true,
  "mechanism": "_dispatch calls started.set() then runs _ble_link_liveness (plain def, no awaits) before its first yield, so started_monotonic is taken after that snapshot",
  "magnitude": "sub-millisecond against a 2000 ms budget; immaterial in size",
  "direction_favours": "section 3 (p95 >= 1000 ms justifies raising the constant)"
}
```

🔑 **The direction is the point, not the magnitude.** A sub-ms bias that happens
to push toward loosening a safety constant belongs on the record, so a later
reader never has to wonder whether it was known at the time.

### 11.4 🔑 `n ≥ 40` is AFTER exclusions — and the leg plan carries headroom

The two decisions interact, and the interaction bites at analysis time:
**exclusions come off the top**, so a session can satisfy the burst-count rule
and still land under the bar.

✅ **The bar is `n ≥ 40` `pulse_open` samples surviving exclusion** — not 40
collected.

✅ **Plan ~8 legs to bank 40, not the bare 4–6.** At ~7–10 pulse-opens per short
leg, 6 legs yields ~42–60 *before* exclusions, which clears 40 only if almost
nothing is excluded. 8 legs yields ~56–80, absorbing a ~25 % exclusion rate and
still clearing. 🚨 **A shortfall discovered at analysis time is unrecoverable
with the mower already docked** — that is the failure this headroom exists to
prevent, and it is cheaper to drive two extra short legs than to return an
inconclusive verdict on arithmetic.

⚠️ §2's independence requirements still bind and are not relaxed by the higher
leg count: `≥ 4` distinct legs, `≥ 2` with a turn or calibration phase.

### 11.5 The "per pulse, not per metre" assumption — how the legs address it

§10.6 demoted it to an assumption. Making that operational rather than just
disclaimed:

✅ **Of the ~8 planned legs, at least 2 are sited 6–8 m from the dock**, in
coverage the map calls good, at a distance comparable to where legs 7 and 8
failed (−60 to −67 dBm, ~8 m out). They are tagged `distance_band: "far"` in the
evidence file; the rest are `"near"`.

🔑 **This does not make the session a test of the assumption** — 2 legs decides
nothing, and no criterion in §2–§5 is conditioned on the split. It exists so the
population is not *entirely* the strongest-link regime, and so a later session
has a per-class, per-band starting point instead of nothing.
🛑 **Both bands remain subject to `plan_aligned_leg.py`'s `--min-rssi-dbm -76`
rejection and every standing protocol requirement.** "Far" means farther from
the dock in verified-good coverage — **not** into the weak cells §9 excluded,
and not a survey leg.

⚠️ **A null result still cannot be read as "the failure does not happen"** — only
as "it did not happen in these bands at this n."

---

## 12. AMENDMENT — the third answer gets a PREDECLARED discriminator, and Phase 1
## cannot start from the dock. Still `sample_count: 0`.

### 12.1 🚨 §10.5's "third answer" was an escape hatch. Fixed.

`mammotion-ha-0f` caught this and is right. §10.5 introduced a third possible
outcome — *"write latency on this link is the binding constraint"* — **with no
criterion attached.** An unfalsifiable extra branch, available after the data
lands, is precisely what this document exists to forbid. I added it while
correcting a contamination defect, which is how these things get in.

✅ **It now has a test, from fields already banked — no new instrumentation.**
`write_ms` is recorded on every completed dispatch, so the comparison costs
nothing:

```
W = p95(write_ms)        over ALL classified COMPLETED samples (refreshes included)
Q = p95(queue_wait_ms)   over the pulse-open class only
write_inheritance_ratio  = Q / W
```

🔑 **Refreshes belong in `W` on purpose.** They are the writes that actually
occupy the serialized queue, so they are the right estimate of "how long the
thing ahead of a pulse-open takes". Only completed samples carry `write_ms`
(a timeout records `None`), and that restriction is stated here so it cannot be
quietly relaxed later.

**Evaluation order is fixed now, because order decides verdicts:**

1. **§4 first, on absolutes.** If `Q ≤ 250 ms` and the worst-wait fraction
   `≤ 0.40`, §4 stands — the bound is exonerated and nothing needs explaining,
   whatever the ratio says.
2. **Then the ratio discriminates §3 from the third answer**, but only when §3's
   own `Q ≥ 1000 ms` clause is met:
   - **`ratio ≤ 1.5`** ⇒ **THIRD ANSWER: write latency is the binding
     constraint.** The wait is consistent with a *single* in-flight write ahead
     of the pulse, not with multiple items queued. 🛑 **§3 is NOT invoked and the
     constant is NOT a candidate to move** — raising it would only let the pulse
     start later and further out of sync with its own timing model, which is the
     exact harm the bound exists to prevent.
   - **`ratio > 1.5`** ⇒ genuine multi-item contention beyond what one write
     explains, so **§3's clauses apply as written**.
3. Anything else — `Q` between 250 and 1000 ms, or §3's other clauses unmet —
   remains **§5 inconclusive**. The third answer does not absorb the middle.

**Why 1.5:** one in-flight write plus queue and scheduling overhead should land
at or just above 1.0; more than 1.5× means more than one write's worth of work
sat ahead. The margin is deliberately generous so ordinary noise cannot flip the
verdict.

⚠️ **`W` and `Q` are different quantities and the ratio is a heuristic**, not a
proof of mechanism. It is stated in advance precisely so it cannot be tuned
afterwards, and a ratio near the 1.5 boundary should be reported as ambiguous
rather than rounded into a verdict.

### 12.2 🚨 Phase 1 CANNOT start from the dock — the executor cannot undock itself

I flagged `position_not_valid_for_motion` to the operator as unresolved and
gating Phase 1. ✏️ **That was wrong and is retracted.** It is the ordinary docked
state: `_position_has_known_area` needs `pos_type_label` in
{AREA_INSIDE, TURN_AREA_INSIDE, CHANNEL_AREA_OVERLAP} **and** `zone_hash` not in
(None, 0, "0"), and on the dock the live readings are `CHARGE_ON` with
`zone_hash: 0`. Position itself is fine (x 4.3188, y 3.2862) and RTK reads
`Fix`, so `rtk_not_precise` cleared correctly — nothing replaced it and nothing
regressed. The deploy skill's "Known-benign readings" already says exactly this.

✅ **The operational consequence, which is real and belongs in the run plan:**
the guarded executor **cannot be what leaves the dock**, because the gate
refuses while docked. So:

1. The mower is placed inside a mowing area **first** — by the operator, the
   vendor app, or an undock — **not** by the vector executor.
2. **Leg 1 starts from wherever the operator parks it**, not from the dock.
   §11.5's "6–8 m from the dock" band is measured from that parked position's
   relationship to the dock, not from a leg that begins on the charger.
3. 🔑 **Confirm the gate POSITIVELY before the first armed dispatch** — that
   `pos_type_label` reads an accepted area label and `zone_hash` is non-zero.
   **The absence of a blocker is not the check**; a pre-flight that waits for
   `position_valid_for_motion` to go true *while docked* will refuse forever.
4. After parking, re-run the standing pre-flight in full: fresh corridor scan
   against the map, tape measurement on any corridor under a couple of metres,
   `docs/accepted-profile.json` verbatim and verified key-by-key, and the
   mower's facing derived two ways per the standing repositioning trap — 🚨 **a
   hand-placed or app-driven mower has stale heading telemetry until it drives**,
   which is exactly the arrangement that produced the 2026-09-04 wrong-direction
   dispatch.

---

## 13. 🚨 §4's clause 3 CONTRADICTS §11.4 — fixed. And what "passing" means.
## Still `sample_count: 0`.

### 13.1 The contradiction, and why it would have mattered

§4 clause 3 still reads **"`n ≥ 120` from `≥ 4` legs, per §2"**. But §10.4 and
§11.4 restated the §3/§4 verdict bar as **`n ≥ 40` pulse-open surviving
exclusion**, deferring n ≥ 120 to the rate claim alone.

🚨 **As written, §4 was unreachable under the sizing actually planned.** A
session banking 45 pulse-opens at p95 80 ms and a worst-wait fraction of 0.15
would satisfy clauses 1 and 2 and then fail clause 3 — collapsing to §5
inconclusive and discarding a clean, decisive exoneration on arithmetic alone.
That would have defeated the operator's "split the claims" decision entirely,
and it would have surfaced at analysis time with the mower already docked.

✅ **§4 clause 3 is replaced by:** *`n ≥ 40` pulse-open samples surviving
exclusion, from `≥ 4` distinct legs with `≥ 2` running a turn or calibration
phase, per §11.4.*

🔑 **§3 needed no equivalent fix** — its three clauses never referenced `n`; the
sample floor reaches it through §11.1's inconclusive clause 3, which applies to
every verdict equally.

### 13.2 There is no "pass". Two independent axes.

Recorded because "did it pass?" is the natural question and the honest answer is
that this document does not define one.

**AXIS 1 — is the session VALID?** This is the only genuine pass/fail here. All
must hold:

- `n ≥ 40` pulse-open samples **surviving exclusion**;
- from `≥ 4` distinct legs, `≥ 2` with a turn or calibration phase;
- pulse-open share within **8 %–35 %** of classified samples (§11.1);
- unclassifiable bursts **≤ 20 %** of all bursts;
- no leg with an unreconcilable per-burst cross-check disagreement;
- the session was **not truncated by a cause unrelated to queue timing** —
  transport loss, a VIO refusal, a containment refusal, battery.

Fail any one ⇒ **INCONCLUSIVE**, the constant stays 2.0, and the per-item data
is banked for the next attempt. ⚠️ **"Inconclusive" is a real outcome to be
reported as one**, not a licence to interpret the numbers anyway with a caveat.

**AXIS 2 — what is the VERDICT?** Only meaningful if axis 1 passed. Four
mutually exclusive outcomes, in this fixed evaluation order:

| order | verdict | condition |
| --- | --- | --- |
| 1 | **§4 — the constant is FINE** | `p95 ≤ 250 ms` **and** worst-wait fraction `≤ 0.40` |
| 1b | **§4 strongest form** | a `queue_start_timeout` occurs **while** `p95 ≤ 250 ms` — bimodal, an episode not a tail; **overrides §3 outright** |
| 2 | **Third answer — WRITE LATENCY binds** | `p95 ≥ 1000 ms` **and** `ratio ≤ 1.5` (§12.1) |
| 3 | **§3 — a raise is DEFENSIBLE** | `p95 ≥ 1000 ms` **and** `ratio > 1.5` **and** fraction `≥ 0.75` on ≥ 2 legs **and** ≥ 1 real timeout |
| 4 | **§5 — inconclusive** | anything else, notably `p95` between 250 and 1000 ms |

🛑 **No outcome on this axis changes the constant in this session.** Even verdict
3 earns only the right to *propose* a number to the operator; verdicts 1, 1b and
2 each say explicitly that moving it is the wrong action.

🔑 **The outcome I expect is verdict 1 or 2, and either is a good result.** "The
constant is fine" is an answer, not a failure — the question was *how much
headroom does the bound have*, and a confident "plenty" closes issue 1 as surely
as a number change would.

### 13.3 ⚠️ The standing "a safe refusal is a FAIL" rule does NOT apply here

CLAUDE.md's discipline says *"a run that stops safely on a named refusal is a
FAIL, not a smaller number."* 🔑 **That rule governs runs demonstrating a
capability** — reach, landing accuracy — where a refusal means the capability
was not shown.

**This session is a measurement, and a `queue_start_timeout` is the signal it
exists to observe.** A leg that aborts on one has produced the most valuable
sample in the set, and §3 clause 3 positively *requires* at least one. It is not
a failure.

⚠️ **What a mid-session abort does cost is samples**, which is handled by axis 1
(the `n ≥ 40`-after-exclusions bar and the unrelated-truncation clause) and by
§11.4's 8-leg headroom — not by treating the refusal itself as a bad outcome.

---

## 14. AMENDMENT — bands re-keyed to MEASURED RSSI, and the run geometry fixed.
## Still `sample_count: 0`.

### 14.1 🚨 §11.5's bands were keyed to the wrong variable

§11.5 required "≥ 2 legs sited 6–8 m from the dock" to avoid sampling only the
strongest-link regime. ✏️ **Dock distance is the wrong key and I should not have
used it.** Link strength depends on distance from the **serving proxy** —
`hot-tub-backyard` — which is not the dock. The error is visible in the data:
the cells 6.65–8.78 m from the dock read **−67 to −75 dBm**, while legs 7 and 8
actually failed at **−60 to −67 dBm**. The "far" band was *better*-signalled than
the failures it was meant to bracket, so it captured a weaker-link regime only
by accident.

✅ **Re-keyed to the coverage-map RSSI estimate at the planned target**, which is
the quantity the failure hypothesis actually concerns and is knowable before
dispatch:

| band | coverage-map median RSSI at target |
| --- | --- |
| `strong` | **≥ −68 dBm** |
| `moderate` | **−68 > estimate ≥ −76 dBm** |
| rejected | below −76 dBm — unchanged, enforced by `plan_aligned_leg.py --min-rssi-dbm -76` |

**Quota: ≥ 2 legs in EACH band**, tagged `rssi_band` in the evidence file
alongside the estimate, its sample count, and the live `ble_rssi` at dispatch.

**Why −68:** it splits the reachable ground near-evenly (56 vs 57 cells in the
recommended excursion disc, which spans **14 dB**, −62 to −76), and it sits just
above the −60…−67 range where legs 7 and 8 failed — so `moderate` is genuinely
weaker-link than the observed failures while `strong` brackets them.

🔑 **The quota is now satisfiable from a single parked position**, so the mower
is placed once and not moved mid-session. Under the old dock-distance keying it
was not: parking at 7.2 m put every leg in the "far" band with no "near" legs
available at all.

⚠️ **Unchanged: this is still not a test of the per-pulse-vs-per-metre
assumption.** No criterion in §2–§5 or §12–§13 is conditioned on the band split;
it exists so the population is not monolithic and so a later session has a
per-band starting point. **A null result means "not in these bands at this n."**
A real test of that assumption needs deliberate band contrast and its own
predeclaration.

### 14.2 Run geometry — bounded by construction, not predicted

🔑 **The endpoint cannot be predicted and does not need to be.** Each leg aims at
the **live** `map_facing_degrees`, which drifts because the executor stops on
position tolerance, not orientation — so the path is a walk. It is therefore
**bounded by design** instead, which is what the 4 m series failed to do when it
"walked itself into a corner of Backyard Right".

**Computed over the live area polygon, both keep-outs, and the 96 h coverage
map** (`docs/evidence-ble-coverage-96h-20260912.json`):

| excursion radius | cells outside area | cells < −76 dBm | cells with no BLE data |
| --- | --- | --- | --- |
| **3.0 m** | **0** | **0** | **0** |
| 4.0 m | 0 | 1 | 1 |
| 4.5 m | 0 | 2 | 5 |
| 6.0 m | 24 | 6 | 13 |

✅ **Maximum excursion is 3.0 m from the parked position** — the largest radius
that is fully verified in every direction, with no unsampled cell and nothing
below the wall.

✅ **Leg rule: ~1.0 m legs, turn back every 3 legs.** Excursion is then
`3 × 1.0 = 3.0 m` exactly, and 8 legs finish within ~2 m of the start. The two
turn-backs also satisfy §2's "≥ 2 legs with a turn or calibration phase"
naturally rather than through contrived extra turns.
⚠️ **1.5 m legs would require turning back every 2 legs** to hold the same bound.
Either is acceptable; what is fixed is the **3.0 m excursion cap**, not the leg
length.

**Parked position, primary:** `map_xy (5.4, −3.8)` — **≈1.1 m EAST and 7.1 m
SOUTH of the dock**, range 7.21 m. Coverage-map estimate **−70 dBm** (n = 69),
**4.9 m clearance** to the nearest boundary or keep-out.
**Alternate:** `map_xy (4.9, −3.3)` — 6.65 m out, estimate **−67 dBm**, slightly
less margin.

🔑 **Use the DOCK-RELATIVE offset, not an absolute lat/lon.** The offset is a
difference inside one frame, so the frame reference cancels and only the ~0.11 %
scale term survives (≈8 mm over 7 m). An absolute lat/lon computed without the
full affine fit is a rough cross-check only and must not be used for placement.

**Resulting footprint:** everything stays within `x 2.4–8.4, y −6.8 to −0.8`.

🛑 **None of this substitutes for the standing per-leg protocol.** Every leg
still needs its own dry run, a fresh corridor scan against the map, a physical
tape measurement on any corridor under a couple of metres, and explicit operator
go/no-go immediately before dispatch. 🚨 **And the mower is hand- or app-placed,
so its heading telemetry is stale until it drives** — derive facing two ways
before the first armed dispatch (§12.2).

---

## 15. PRE-DISPATCH NOTE — the setup leg is EXCLUDED from the §2 population.
## Written before it was dispatched, while `sample_count` was still 0.

The mower ended up at `map_xy (4.4688, −0.5013)` after the operator cancelled a
mow — 3.5 m north of §14.2's parked position and unable to satisfy the
`moderate` band quota there (−62 dBm; 96 strong cells against 1 moderate inside
its 3.0 m disc). One **unscored setup leg** moves it into the pocket: 3.4 m at
heading 277.683° (0.0° off the measured facing), target `(4.9234, −3.8708)`.

🚨 **That leg dispatches through `_send_ble_motion_command_confirmed`, so it WILL
record timing samples and take `sample_count` off 0.** Its status is therefore
fixed here, before it runs:

✅ **Tagged `leg_role: "setup"` in the evidence file. Its samples are recorded
in full and EXCLUDED from the §2 population.** Reasons, stated in advance:

- it is **3.4 m**, not the ~1.0 m of §14.2's leg rule, so it carries a
  different pulse count and a different share of final-approach behaviour;
- it **starts outside the planned excursion disc**, in the strong band only, so
  including it would skew the band composition the §14.1 quota exists to
  balance;
- it was **not planned under §14.2's geometry** at all — it exists to satisfy
  §14.2's precondition, not to sample it.

🔑 **This is not a licence to discard inconvenient legs later.** The exclusion
covers exactly one leg, identified by position and heading above, recorded
before its data existed. Every leg dispatched *after* the mower reaches the
pocket is in the population unless it trips an exclusion rule already written in
§11.1, §11.2 or §13.2.

⚠️ **`sample_count` will be non-zero from this point on**, so no later amendment
to §2–§14 is legitimate. Those sections are closed.


---

## 16. PRE-DISPATCH NOTE — Phase 1 run plan, fixed before any scored leg.
## Operator directed the full run on 2026-09-12 after the budget risk was stated twice.

**Operator decision, recorded as such:** push to finish 10–12 scored legs this
session. The risk was stated twice — a session-limit truncation is predeclared
**inconclusive** by §13.2 axis 1 — and the operator chose to proceed. That rule
is **unchanged**: if the session is cut off before `n ≥ 40` pulse-opens survive
exclusion, the verdict is inconclusive and is reported as such.

### 16.1 A second setup leg is EXCLUDED, like §15's

The mower is at `(4.45, 0.54)`, 4.38 m north of the pocket. One unscored setup
leg drives it to the anchor `(4.94, −3.82)`. Tagged `leg_role: "setup"` and
**excluded from the §2 population**, for §15's reasons. This covers exactly this
one leg.

### 16.2 Scored targets are ABSOLUTE and fixed now

Each scored leg drives from the live position to a fixed absolute target on the
N–S axis through the anchor. **Driving TO a fixed target corrects landing drift
by construction**, so pattern-centre drift cannot accumulate across legs — this
resolves the drift risk raised for the longer run. Every target is within
**3.0 m** of the anchor (§14.2's cap). Bands are the coverage-map estimate at the
target, assigned here so they cannot be chosen after the fact:

| leg | target | offset from anchor | est. RSSI | band | clearance |
| --- | --- | --- | --- | --- | --- |
| S1 | (4.94, -4.82) | -1 m | -72.0 | `moderate` | 4.9 m |
| S2 | (4.94, -5.82) | -2 m | -74.0 | `moderate` | 4.95 m |
| S3 | (4.94, -6.82) | -3 m | -76.0 | `moderate` | 4.39 m |
| S4 | (4.94, -5.82) | -2 m | -74.0 | `moderate` | 4.95 m |
| S5 | (4.94, -4.82) | -1 m | -72.0 | `moderate` | 4.9 m |
| S6 | (4.94, -3.82) | +0 m | -68.0 | `strong` | 4.85 m |
| S7 | (4.94, -2.82) | +1 m | -66.0 | `strong` | 3.91 m |
| S8 | (4.94, -1.82) | +2 m | -66.0 | `strong` | 2.91 m |
| S9 | (4.94, -2.82) | +1 m | -66.0 | `strong` | 3.91 m |
| S10 | (4.94, -3.82) | +0 m | -68.0 | `strong` | 4.85 m |
| S11 | (4.94, -4.82) | -1 m | -72.0 | `moderate` | 4.9 m |
| S12 | (4.94, -5.82) | -2 m | -74.0 | `moderate` | 4.95 m |

**Band quota (§14.1, ≥ 2 each):** `strong` 5, `moderate` 7. Turn-backs
fall at S4 (south→north) and S9 (north→south), satisfying "≥ 2 legs with a turn
phase".

### 16.3 Automatic HALT conditions — any one stops the series

Enforced per leg by the runner before and after dispatch: position not
`AREA_INSIDE` / wrong zone / not valid; RTK not `Fix`; blade not reported off,
latched, or any blade blocker; facing confidence `unknown`; corridor leaving the
area or under **1.0 m** clearance; target excursion over 3.0 m; target band
missing or below −76 dBm; dry run not `valid`, or with errors, keep-out
violations, blockers, any failing safety gate, or **any accepted-profile echo
mismatch**; real `stop_reason` other than `target_reached`.
🛑 **A HALT is not overridden in-session.** The gate is disarmed and the
operator decides.

### 16.4 Collection

After every leg the runner saves the leg result and a full
`motion_dispatch_timing_report` snapshot (§11.3). Classification applies §10.3
**after filtering out budget-5.0 stop samples**, per §1's budget rule — the gap
recorded in `docs/findings-setup-leg-and-classifier-validation-20260912.md` §3.
Setup-leg samples (today's first 65 plus 16.1's) are excluded by timestamp.

⚠️ **Sizing:** 12 legs of ~1.0 m banks roughly 12 × 4 ≈ 48 pulse-opens before
exclusions — only modest headroom over 40, per the measured ~4 per 1.0 m leg.
