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

The repo states a "2–6 cm pulsed-measurement noise floor"
(`docs/gate4-repass-20260805.md:25,182`) and uses it to argue that
`waypoint_tolerance: 0.08` sits at or below instrument error. **That framing is
wrong**, and it has shaped the tolerance debate ever since.

Measured from `docs/evidence-darkmow-observation-20260804.jsonl` (1595 samples,
1799 s of an operator-run mow):

| quantity | value | n |
| --- | --- | --- |
| position jitter while **stationary** | mean **0.044 cm**, max **0.553 cm** | 690 |
| position **update interval** while moving | median **1.11 s**, p90 1.15 s | 654 |
| travel between updates at 0.263 m/s | median **29.3 cm** | 654 |

Cross-checked against our own executor run,
`docs/evidence-gate4-beta20-day2j-real-capture-20260805.jsonl` (61 distinct
position updates):

| quantity | value |
| --- | --- |
| update interval | median **1.11 s**, p90 **4.44 s**, max 6.69 s |
| distance per update while moving | median 6.3 cm, p90 **22.8 cm**, max **33.6 cm** |

**The feed is sub-millimetre at rest and identical in cadence across two
independent sessions — a mower-driven mow and our own pulsed executor.** So the
uncertainty in a stop decision is not sensor noise. It is *staleness*: the
mower's reported position is up to one update interval behind reality, and at
operating speed one interval is 20–30 cm.

### Consequence

Per-tier speeds, measured across every linear command in the two nominal Gate 4
passes (day2i, day2j):

| tier | commanded | measured | travel per 1.11 s update |
| --- | --- | --- | --- |
| fast | 400 | 0.207–0.322 m/s (~0.28) | **~31 cm** |
| slow | 200 | 0.084, 0.123 m/s (~0.10) | **~11 cm** |

Against `waypoint_tolerance: 0.08`:

> **At 0.28 m/s with a 1.11 s position update, landing inside 8 cm is not
> reliably achievable by any control law.** The information required to stop
> within tolerance arrives less often than the error accumulates. Even the
> existing slow tier (~0.10 m/s) yields ~11 cm per update — still outside
> tolerance.

The recorded overshoot of 0.15–0.26 m is not a defect to be tuned away. It is
approximately half to one update interval of travel — exactly what this model
predicts.

## 2. Answering the three questions

### Is the acceptance criterion right? — No, and its provenance is missing

`waypoint_tolerance: 0.08` has **no derivation anywhere in the repo**. Grepping
every doc finds it only ever used, never justified. It is an inherited number.

Two independent problems:

1. It is below what the telemetry can support at the speeds in use (§1).
2. It has never been checked against the actual goal. The stated scope is
   point-and-click movement with the blade off. For "move the mower over there,"
   8 cm is a very demanding requirement; 15–20 cm would very likely satisfy the
   use case — and would have passed weeks ago.

The re-pass doc raised the tolerance question and deferred it as "the project's
call." **That call has never been made, and until it is, every Gate 4 result —
pass or fail — is uninterpretable.** This is the highest-leverage open item,
ahead of any code change.

### Is the control architecture right? — It is open-loop, and that is the root

Every distinct failure this project has recorded reduces to one property:

| symptom | mechanism |
| --- | --- |
| overshoot | cannot stop on a position that is one update stale |
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

⚠️ But note what §1 implies: **closed-loop control does not rescue an 8 cm
tolerance either.** A controller cannot beat its own sensor rate. Closed-loop
would fix cross-track steering and remove the U-turn class outright, which is
worth having — but the terminal accuracy limit is set by speed ÷ update rate,
not by the control law.

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

1. **Decide the tolerance against the use case, not the instrument.** If the
   goal is point-and-click repositioning, propose 0.15–0.20 m and state the
   rationale. This is a product decision and belongs to the operator.
2. **Make the final approach speed satisfy `speed × 1.11 s < tolerance`.** For
   8 cm that needs ≤0.072 m/s — below the current slow tier. For 15 cm, ~0.13
   m/s, which the existing slow tier already delivers. This is the single change
   most likely to make a gate pass mean something.
3. **Then** reconsider closed-loop control, for the cross-track and U-turn
   classes rather than for terminal accuracy.
4. Treat the `speed × latency` stop-lead as a refinement *within* the update
   interval, not as the fix. It corrects the systematic component; the residual
   scatter of ±half an update interval remains.

Sequence matters: 2 without 1 is guesswork, and 3 without 2 will not improve the
number the gate measures.

## 5. Limits of this review

Stated so it is not over-read:

- The darkmow session was the mower's own mowing, not our commanded motion. The
  transferable claim is the **update cadence**, which matched our own run
  exactly (1.11 s median in both); its 0.263 m/s speed is the mower's, not ours.
- Per-tier speeds come from 8 fast and 2 slow commands. The slow tier has **two**
  samples (0.084, 0.123 m/s); treat ~0.10 m/s as provisional.
- p90 update gaps in our pulsed run (4.44 s) are far worse than in the
  continuous mow (1.15 s). Whether pulsed motion itself degrades the feed is
  **not established** and is worth a dedicated measurement.
- I did not evaluate whether Mammotion's own app achieves better positioning, or
  what telemetry it uses. If it does, there may be a faster feed available that
  this integration is not subscribing to. **That is the highest-value unknown.**
- No claim here is a hardware test. Nothing was run on the mower for this review.

## 6. What this does not change

beta22 stays as deployed. The guard is correct and honest, the release gates and
safety model are unaffected, and Gate 5 remains blocked. No motion is authorized
by this document.
