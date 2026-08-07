# Direction review — 2026-08-06

Written off-mower at the end of the beta22 session, in response to "review the
progress we have or have not made against the overall goals, and whether the
current path needs to change."

This is not a code review. It asks three questions: is the acceptance criterion
right, is the control architecture right, and is the iteration converging.

**Headline: the project has been optimising against a mis-stated constraint.**
The position feed is not noisy — it is *slow*. Every measurement below is
derived from evidence already committed to this repo.

## 1. The measurement that reframes everything

> ## ⚠️ CORRECTION — 2026-08-07
>
> **The "1.11 s position update interval" reported in the original version of
> this section was the *sampler's* polling interval, not the feed's update
> rate.** `scripts/motion_capture.py` defaults to `--interval 1.5` and both
> captures actually ran at **1.113 s median**, identical to the figure reported
> as a measurement. The feed's true rate is therefore **unresolved below
> ~1.1 s** — it may well be faster.
>
> **Withdrawn:** "position updates arrive every 1.11 s"; "~29–31 cm travelled
> between updates"; and the boxed claim that 8 cm "is not reliably achievable by
> any control law." None of those is established.
>
> **Retained** (unaffected by sampling rate): stationary jitter; per-tier speeds;
> the 1000 ms wire `period` default read from code (§7); the missing continuous
> subscription read from code (§7); and the 28%-vs-63% comparison below.
>
> This is precisely the confounded-measurement error the project's own notes
> warn against — "derive constants from an isolated command, never from net
> displacement spanning phases." The corrected direction is unchanged but now
> rests on §7's code findings rather than on this timing claim, and the true feed
> rate is the subject of a dedicated zero-motion probe.

The repo states a "2–6 cm pulsed-measurement noise floor"
(`docs/gate4-repass-20260805.md:25,182`) and uses it to argue that
`waypoint_tolerance: 0.08` sits at or below instrument error. **That framing is
still wrong**, and it has shaped the tolerance debate ever since — but for the
reason in the next paragraph, not the withdrawn one.

Measured from `docs/evidence-darkmow-observation-20260804.jsonl` (1595 samples,
1799 s of an operator-run mow) and
`docs/evidence-gate4-beta20-day2j-real-capture-20260805.jsonl` (216 samples,
240 s of our own executor run). **Sampling method: both captured by
`motion_capture.py` at 1.113 s median; any quantity with units of time is
bounded below by that and is reported as such.**

| quantity | value | status |
| --- | --- | --- |
| position jitter while **stationary** (mow) | mean **0.044 cm**, max **0.553 cm**, n=690 | **valid** — magnitude, not rate |
| samples whose position **changed**, executor run | **28%** (61 distinct / 216) | **valid** — same sampler both rows |
| samples whose position **changed**, continuous mow | **63%** (997 distinct / 1595) | **valid** |
| feed update interval | **unresolved** (< ~1.1 s) | withdrawn |

**The feed is sub-millimetre at rest**, so the uncertainty in a stop decision is
not sensor noise. And the executor's feed is measurably **staler than a
subscribed one**: with the same sampler, position changed on 28% of samples
during our Gate 4 run versus 63% during a continuous mow. §7 explains why —
the executor never subscribes.

### Consequence

Per-tier speeds, measured across every linear command in the two nominal Gate 4
passes (day2i, day2j) from the executor's own `elapsed_ms` and measured deltas
— **not** from the capture, so unaffected by the sampling confound:

| tier | commanded | measured |
| --- | --- | --- |
| fast | 400 | 0.207–0.322 m/s (~0.28) |
| slow | 200 | 0.084, 0.123 m/s (~0.10) |

The open question is now quantitative and testable: **at what interval does the
device actually report?** If the wire `period` of 1000 ms (§7) is what it
honours, then at 0.28 m/s roughly 28 cm passes between updates and an 8 cm
tolerance is out of reach at that speed. If the device honours a shorter period,
the picture changes entirely. That is a measurement, not an argument, and it has
not yet been made.

The recorded overshoot of 0.15–0.26 m remains consistent with a staleness
mechanism rather than a tuning defect, but the specific "half to one update
interval" arithmetic is withdrawn along with the interval it rested on.

## 2. Answering the three questions

### Is the acceptance criterion right? — No, and its provenance is missing

`waypoint_tolerance: 0.08` has **no derivation anywhere in the repo**. Grepping
every doc finds it only ever used, never justified. It is an inherited number.

Two independent problems:

1. It **may** be below what the telemetry can support at the speeds in use. Per
   the §1 correction this is now an open question rather than a finding, and it
   is exactly what the report-rate probe is designed to settle.
2. It has never been checked against the actual goal. The stated scope is
   point-and-click movement with the blade off. For "move the mower over there,"
   8 cm is a very demanding requirement; 15–20 cm would very likely satisfy the
   use case — and would have passed weeks ago. **This problem stands on its own
   and is unaffected by the correction.**

The re-pass doc raised the tolerance question and deferred it as "the project's
call." **That call has never been made, and until it is, every Gate 4 result —
pass or fail — is uninterpretable.** This is the highest-leverage open item,
ahead of any code change.

### Is the control architecture right? — It is open-loop, and that is the root

Every distinct failure this project has recorded reduces to one property:

| symptom | mechanism |
| --- | --- |
| overshoot | cannot stop on a position that is already stale |
| cross-track drift | cannot steer during a pulse |
| U-turn "re-alignment" | correcting only *after* passing the target |
| `max_linear_commands_reached` short | pulse sized from a stale distance |

These are not four bugs. They are one architecture — **open-loop pulse, then
correct afterwards** — surfacing in four places. Ten betas have tuned the pulse
parameters; none has changed the loop.

**The loop skeleton already exists.** `_motion_refresh_window`
(`services.py:4225`) already runs a 200 ms cadence with correct cancellation and
guaranteed-stop semantics — but its docstring specifies re-sending "the
*identical* movement command," and `resend` is a `functools.partial` bound to
fixed kwargs before the window opens. It is a keep-alive, not a controller. The
h-watchdog finding of 2026-07-22 was adopted as continuous **actuation** and
never as continuous **control**.

⚠️ One caveat that survives the §1 correction in weakened form: **a controller
cannot beat its own sensor rate.** Closed-loop would fix cross-track steering and
remove the U-turn class outright, which is worth having regardless. But terminal
accuracy is bounded by speed ÷ update rate, so whether closed-loop can reach 8 cm
depends on the feed rate the probe measures. Do not assume either answer.

### Is the iteration converging? — Not yet; each fix has surfaced the next

beta12 → beta22 in six days (2026-07-31 → 2026-08-06), nine deploys, four on
2026-08-02 alone. The failure has moved every time rather than closing:

| build | change | outcome |
| --- | --- | --- |
| beta16 | proportional duration scaling | refuted by measurement |
| beta17 | discrete refresh-count budget | turn-budget failure appears |
| beta19/20 | turn-feasibility guard | turn fixed; cross-track aim failure appears |
| beta21 | day2j profile (1300 ms, 3 cmds, 0.30 cap) | passes — by U-turning |
| beta22 | refuse ≥90° recovery | expected to fail again |

That is the signature of treating symptoms at a level below the cause. It is not
wasted work — the measurements are what made this review possible — but the
pattern should be named rather than continued.

## 3. What is genuinely strong, and should not change

The measurement discipline. This project repeatedly refutes its own confident
claims with better data: the "25% low" metres-per-pulse constant, the ~11°
heading offset, single-shot turning at 2.4°/command rather than 8–9°, the
map-sync "every 5 minutes" self-correction. That is rare, and it is the only
reason the finding in §1 was recoverable from committed evidence.

The safety envelope is also sound: fail-closed gates, exclusive sessions,
guaranteed stop-on-cancel, and the evidence runner. beta22's guard is a correct
containment even though it does not address the root cause — it converts a
misleading pass into an honest failure.

## 4. Recommended change of path

**Stop tuning pulse parameters. Settle the criterion first, then match the
approach speed to the telemetry rate.**

**Revised 2026-08-07 after the §1 correction and the §7 code findings.** The
first step is now a measurement, not a decision, because the decision depends on
a number nobody has measured soundly.

0. **Measure the feed rate** with a zero-motion probe that samples far faster
   than the feed, at wire `period` / `no_change_period` of 1000, 500, 200 and
   100 ms. This resolves what §1 could not and gates everything below. Note this
   requires exposing `period`, which no current call path does (§7).
1. **Fix the feedback path** (§7): give the vector executor a continuous
   subscription for the duration of a segment, as four other motion paths
   already have. This is a defect fix and is worth doing on its own merits
   regardless of what the probe finds.
2. **Decide the tolerance against the use case, not the instrument.** If the goal
   is point-and-click repositioning, 0.15–0.20 m is likely right, but state the
   rationale. This is a product decision and belongs to the operator. Sequence it
   after step 0, since a faster feed may make 0.08 m reachable.
3. **Make the final approach speed satisfy `speed × measured_interval <
   tolerance`.** The constant comes from step 0, not from the withdrawn 1.11 s.
4. **Then** reconsider closed-loop control, for the cross-track and U-turn
   classes rather than for terminal accuracy.
5. Treat the `speed × latency` stop-lead as a refinement *within* the update
   interval, not as the fix — and fit it only after step 1, or it will bake in
   the very staleness being removed.

Sequence matters: 2 without 0 is guesswork, and 4 without 3 will not improve the
number the gate measures.

## 5. Limits of this review

Stated so it is not over-read:

- ⚠️ **The largest limit is the §1 correction itself**: every timing claim in the
  first version of this review was bounded below by a 1.113 s sampler and could
  not resolve the feed rate. Any future timing claim in this repo must state its
  sampling method alongside the number.
- The darkmow session was the mower's own mowing, not our commanded motion, so
  its 0.263 m/s speed is the mower's, not ours. The transferable comparison is
  the **fraction of samples that changed** (28% executor vs 63% mow), which is
  sampler-independent because both used the same sampler.
- Per-tier speeds come from 8 fast and 2 slow commands. The slow tier has **two**
  samples (0.084, 0.123 m/s); treat ~0.10 m/s as provisional.
- Whether pulsed motion itself degrades the feed, beyond the missing
  subscription in §7, is **not established** and deserves its own measurement.
- I did not evaluate whether Mammotion's own app achieves better positioning, or
  what telemetry it uses. §7 identifies a specific mechanism by which this
  integration under-subscribes relative to the app's own `count=0` pattern, but
  no app-side comparison was made.
- No claim here is a hardware test. Nothing was run on the mower for this review.

## 6. What this does not change

beta22 stays as deployed. The guard is correct and honest, the release gates and
safety model are unaffected, and Gate 5 remains blocked. No motion is authorized
by this document.

## 7. Code findings — added 2026-08-07

Read directly from source, so unaffected by the §1 sampling confound. These now
carry the argument that the withdrawn timing claim used to.

**7.1 — The Gate 4 / Gate 5 executor never subscribes to reports during motion.**
`_raw_pymammotion_execute_vector_segment` (1144 lines) contains no call to
`async_start_report_stream` or `async_start_continuous_reports`. Four other
motion paths do call them: `_manual_velocity_pulse_test`
(`services.py:4777,4784`), `_manual_velocity_cumulative_pulse_test`
(`:11436,11443`), `_experimental_execute_segment_burst` (`:11751,11758`), and
`_manual_velocity_segment_test` (`:12093,12100`). The one path that runs the
release gates is the one that does not.

**7.2 — Its only feedback is a post-hoc poll plus a fixed sleep.**
`_refresh_position_after_raw_motion` (`services.py:7584`) issues
`async_get_reports(count=5)` and then `await asyncio.sleep(2.0)`. It runs
*after* each pulse's stop, so no subscription is held *during* the pulse.

**7.3 — The wire report period is never lowered from the library default.**
`request_iot_sys` (`pymammotion/mammotion/commands/messages/system.py:218-226`)
takes `period: int = 1000` and `no_change_period: int = 1000` and writes them
into `ReportInfoCfg` — these are device-side protocol fields, not client-side
polling. `handle.request_reports` (`device/handle.py:1376`) omits `period`
entirely, and `coordinator.async_start_continuous_reports`
(`coordinator.py:1504`) routes through it, so **that path cannot set the period
at all**. Only `client.request_iot_sync_continuous` exposes it, and the sole
in-repo caller hardcodes `period=1000` (`services.py:5694`).

Together these mean position-derived decisions — waypoint tolerance, progress,
displacement caps — run against a feed that is polled after the fact at a
device-side cadence of 1000 ms that nothing has ever tried to lower. **Whether
the device would honour a shorter period is unknown and is the subject of the
probe in §4 step 0.** The coordinator's own docstring (`coordinator.py:1515`)
already notes that `count=0` with `RPT_START` is "the same continuous
subscription the Mammotion app uses, reported at the library's default 1000 ms
period" — so the app-parity claim covers the subscription mode but not the rate.
