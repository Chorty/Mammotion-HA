# CORRECTION: the step probe stalled its own position feed

**Written 2026-08-29, the same day as the runs it corrects.** Read this before
any of the three evidence files listed below. Their **measurements stand**; their
**interpretations do not**.

## The bug, in one paragraph

`DeviceHandle.exclusive_report_subscription` does not merely serialize ownership.
Its **first act** is to stop the report stream:

```python
if self._ble_stream_active:
    await self._enqueue_ble_stream_command(RptAct.RPT_STOP, count=1)
    self._ble_stream_active = False
```

and it sets `_report_subscription_owner` so the background loop cannot start a
new configuration for the life of the lease. Its own docstring says it
*"[s]erialize[s] temporary report configuration **and stop[s] background
renewals**."*

`continuous_motion_window` restarts the stream inside that lease —
`begin_report_subscription_generation`, `async_start_report_stream`,
`async_start_continuous_reports`, then a queue settle, then a readiness wait for
a position payload inside its own generation.

🐛 **`raw_pymammotion_step_response_probe` took the lease and never restarted the
stream.** It even called `async_stop_continuous_reports` in its `finally` —
stopping something it had never started. So every step-probe run drove with **no
report configuration active at all.**

## It explains every observation, and more simply than what I proposed

| observation | my published reading | actual cause |
| --- | --- | --- |
| `position_sequence` frozen | device/backend stopped emitting position | no report configuration was running |
| `last_report_at` advancing | frames arriving without position | command acks and other traffic, correctly |
| all four outbound BLE facts healthy | "the fault is inbound, outside this integration" | outbound was never involved |
| reproduces on every run, both signs, three builds | "reproducible, not intermittent" | **deterministic — it is a code path, not a fault** |
| blind travel ~0.43 m every time | unexplained consistency | the guard trips at a fixed time because the feed is *always* dead |
| **steering attempt 5's feed worked** | "two different regimes" | **two different code paths.** That service starts the stream |

That last row is the discriminator I repeatedly treated as a puzzle. It was the
answer.

## What is retracted

🗑️ **`docs/evidence-position-feed-stalls-during-motion-20260829.json`** — its
headline, *"Q1 ANSWERED: the position stream STOPS DELIVERING during a motion
window"*, is **withdrawn**. The stream was not delivering because this probe had
stopped it. The `position_sequence` reading itself is sound and the discriminator
logic is correct; only the attribution is wrong.

🗑️ **`docs/evidence-feed-stall-is-not-our-dispatch-path-20260829.json`** — its
headline, *"the fault is INBOUND … outside this integration"*, is **withdrawn and
inverted**. The fault was in this integration, in the probe. The four BLE fields
did their job: they correctly showed outbound health, and outbound was fine. What
they could not see was that no report configuration existed.
⚠️ Its **two-sign** finding survives in a narrower form: `+120` and `−120` behaved
identically, which is consistent with a deterministic code path and says nothing
about the drivetrain.

🗑️ **`docs/evidence-step-response-probe-aborted-20260828.json`** — the first run,
same cause.

🚨 **n DROPS FROM 5 TO 1.** Four of the five "occurrences" were this bug.
**Only steering attempt 3 (2026-08-27, 0.51 m blind) remains**, and it ran on
`continuous_motion_window`, which *does* start the stream. **That observation
stands, is unexplained, and is n = 1** — which is where it was before this probe
existed. `docs/phase2-steering-attempt3-blind-travel-20260827.md`.

## The fix

The probe now does what the continuous window does, and **fails closed** if it
does not work:

* `begin_report_subscription_generation(report_lease)` for an evidence boundary;
* `async_start_report_stream` and `async_start_continuous_reports`;
* `_settle_ble_command_queue`, because both start calls return on **enqueue** —
  only the post-settle instant proves the START reached the transport;
* `_wait_for_position_subscription_ready` against that generation, with
  `_STEP_RESPONSE_READINESS_TIMEOUT_S = 3.5`. **No position payload inside its own
  generation ⇒ `position_subscription_not_ready` and nothing is commanded.**

🔑 **The readiness wait is the part that matters.** Without it the probe would
drive, record nothing, and the null would read as a device fault — exactly what
happened four times.

Two regression tests pin it: one asserts both start calls are made under a fresh
generation, one asserts an empty stream refuses to drive rather than driving
blind.

## The lesson worth keeping

⚠️ **Reusing a lease wrapper is not the same as reusing what the lease is for.**
The probe copied `continuous_motion_window`'s `exclusive_report_subscription`
block because that is the shape every safety-critical service here uses — and
silently dropped the twenty lines that make the lease useful. The lease *takes
away* the stream; the caller must put it back.

🔑 **And the tell was in the evidence the whole time.** Four runs produced
**zero** informative intervals with **bit-identical** positions. A real
intermittent fault does not produce a perfect null four times running. **When a
"reproducible fault" reproduces too perfectly, suspect the instrument.**
