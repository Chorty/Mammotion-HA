# RTK: settle it properly — plan, 2026-08-07

Written after the 2026-08-07 Float episode and the guard work that followed. The
investigation behind it **changed the design substantially**, including
invalidating two conclusions I reached earlier the same day, so read §2 before
acting on anything already committed.

No motion is authorized by this document.

## 1. What actually happened, and what is verified

**The episode.** RTK read `Float` from 15:40 to 18:39 EDT. Rover reception was
healthy throughout (24 co-viewed satellites both bands, 26 tracked,
`rtk_over_datalink`) and the base station had not been moved. A rover-side
`sync_rtk_and_dock` did nothing. Power-cycling the **dock and base station**
restored `Fix` at 18:39:08, with the whole RTK group refreshing together
(`position_level` 2 → 1, so **1 = Fix, 2 = Float**).

**The cost of Float, measured:** a **13.9 cm** position jump with no command
sent, against **0.044 cm mean / 0.55 cm max** stationary jitter under Fix. A
single Float excursion exceeds the entire 0.08 m waypoint tolerance.

**This was the first recorded Float episode.** Mining every committed evidence
file: **534** recorded `Fix` states, **zero** `Float` or `Single` outside
today's own investigation file. (The 265 `None` records are the dead
`mowing_state` position candidate, which always reports `rtk_status 0`, not real
episodes.) **Past gate results are therefore not tainted** — every prior gate ran
on Fix. That was a live worry and it is now closed.

## 2. ⚠️ Two of my own conclusions from today were wrong

Both were load-bearing for the guard already committed, and both are refuted by
measurements taken this evening.

**2.1 — "A forced report burst caught the latch." It cannot catch anything.**
This afternoon I forced 50 reports, saw no RTK entity refresh, and concluded the
channel was latched. Repeating that test at 20:35 with RTK **verifiably healthy
and `Fix`**: 49 messages arrived, **zero** RTK/position/VIO channel updates, and
the payload age kept climbing 570.9 s → 601.1 s. A healthy feed and a latched
feed look **identical** under a forced burst. The test has no discriminating
power and the afternoon's inference from it was invalid.

**2.2 — The 1800 s staleness threshold will false-block. It is a live defect in
the deployed beta25.** Watching a healthy, stationary, Fix-locked mower, the RTK
payload age reached **3573 s (59.5 minutes)** before resetting. The RTK payload
changes roughly **once an hour** at rest. The deployed threshold is 1800 s, so
motion will be refused with `rtk_telemetry_stale` after ~30 idle minutes with
nothing wrong. This is the *second* time this constant has been set too low —
300 s failed the same way this afternoon.

**The structural conclusion:** legitimate quiet reaches ~1 h and the observed
fault lasted ~3 h. A 3x separation, from one sample of each, is far too poor to
site a timeout on. Combined with 2.1 — no passive threshold works, and no active
probe exists — **RTK freshness is not verifiable with any signal currently
available.** The guard as designed rests on a discrimination that does not exist.

## 3. The corrected design

**Freshness becomes advisory. Quality becomes the guard.**

The key realisation: the fault we actually observed was a latched **Float**
value. The quality gate (`rtk_not_precise`, non-Fix refuses) blocks that
directly, with no freshness reasoning required. Freshness was only ever needed
for a *different*, **hypothetical and never-observed** failure — a latched
**Fix** that silently stops being true. Guarding a hypothetical with a signal
that cannot discriminate, at the cost of false-blocking every run after 30 idle
minutes, is a bad trade.

| concern | mechanism | status |
| --- | --- | --- |
| RTK degraded (Float/Single) | `rtk_not_precise` blocker + `allow_degraded_rtk` override | committed, **not deployed** |
| RTK feed latched | `rtk_report_age_seconds` **reported, never blocking** | needs change — currently blocks |
| Run auditability | `rtk_status_label`, `rtk_degraded`, `rtk_degraded_override`, age in every result | committed |
| Base-station fault | operator runbook + recurrence watch | §5 |

## 4. Actions

### P0 — remove the live false-block (do first)

`_RTK_REPORT_STALE_SECONDS` currently gates motion in deployed beta25 and will
refuse legitimate runs. Change `rtk_telemetry_stale` from a blocker to a reported
diagnostic in `_runtime_motion_safety_summary` (`services.py`). Keep
`rtk_report_age_seconds` in the output — it is genuinely useful for auditing a
run after the fact, it just must not gate one. Record in the constant's comment
why a threshold cannot work, so it is not reintroduced a third time.

Ship together with the already-committed quality gate as **beta26**, since that
gate is the thing that actually protects a precision run and is not yet on the
host.

### P1 — verify the quality gate on hardware

The `rtk_not_precise` blocker has never run against a real degraded state. Once
deployed, confirm on the host that a normal Fix run is unaffected and that
`allow_degraded_rtk` is echoed in results. The refusal path itself can only be
exercised opportunistically — the next time RTK is genuinely Float, check that it
refuses and that the override permits.

### P2 — ✅ ANSWERED 2026-08-07 22:09. Freshness stays advisory, permanently.

Two watch logs, 90 + 152 samples at 30 s, spanning 44.2 min (undocked, in yard)
and 75.8 min (yard then docked). Summary committed as
`docs/evidence-rtk-watch-20260807.json`; raw JSONL was session-scratchpad only.

**The quiet distribution does not support a timeout.** Across both logs the RTK
payload changed **once each**, and the maximum observed legitimate quiet was
**3654.9 s (60.9 min)** docked and **3573.5 s (59.6 min)** in the yard, with
median age ~1380 s. Every one of those samples was healthy `Fix` with the mower
stationary. So legitimate quiet reliably reaches ~1 h against an observed fault
of ~3 h — the same 3x separation as before, now from a much larger sample rather
than one anecdote. That is nowhere near enough margin to site a timeout on.

**This closes the question rather than deferring it.** Freshness is reported
(`rtk_report_age_seconds`) and never blocks. It has now been set too low twice
(300 s, 1800 s) and measured too close to legitimate behaviour a third time. Do
not revisit it without a fundamentally new signal — an *active* probe that can
distinguish a quiet channel from a dead one, which §2.1 established does not
currently exist.

**No early-warning signal was found.** Satellite count and L1 signal quality show
no drift preceding any transition; they track sky view, not solution state.

**Two incidental findings, both new:**

1. **The dock roughly halves satellite visibility.** In the yard the mower sees a
   median of **32** satellites / **25** co-viewed; docked, **17** / **16**. RTK
   nonetheless holds `Fix` throughout at the lower count, so the dock's reduced
   sky view is not by itself disqualifying — worth knowing before reading any
   dock-side measurement as a fault.
2. **A single `Float` sample was captured at 21:19:05 — and it is not a
   recurrence.** ⚠️ It is precisely the sample in which `pos_type` flips
   `AREA_INSIDE` → `CHARGE_ON`, position jumps ~8 m, and satellites drop 30 → 17:
   the operator was driving the mower into the dock at that moment. A momentary
   Float across a large sky-view change is ordinary RTK behaviour and looks
   nothing like the 2026-08-07 fault (3 hours, stationary, reception unchanged).
   Do **not** count it as a second episode. It does confirm the logger will catch
   a degraded state when one occurs.

### P3 — root cause, substantially narrowed 2026-08-07 21:12

**⚠️ This section corrects an overstatement made earlier in this document.** It
claimed "nothing in the mower's telemetry reaches the base station's internal
state". That is **wrong**: `sensor.*_last_error` carries
`"mcu: The RTK reference station has been disconnected, Please wait for
automatic reconnection"`, which is a direct base-station signal.

**What the error history establishes.** Twelve hours of history contains exactly
two errors: a battery-low from 2026-08-06, and the reference-station disconnect
with device timestamp **2026-08-07T22:19:07Z = 18:19:07 EDT**. That is roughly
four minutes *after* the operator power-cycled the base — so the error is the
**power cycle itself**, not the fault.

**The decisive negative:** no reference-station error fired during the
15:40–18:39 Float window. The mower never reported losing the base. Together
with 24 co-viewed satellites throughout, the base was **connected and
transmitting for the entire episode**.

That reshapes the hypothesis space:

| hypothesis | status |
| --- | --- |
| Base disconnect / dead datalink | **ruled out** — no error fired, 24 co-viewed satellites |
| Base firmware hang stopping transmission | **unlikely** — would be expected to raise the disconnect error |
| **Base transmitting usable but WRONG corrections** (bad or never-completed survey, bad reference position) | **leading** — the only mechanism consistent with corrections flowing while the rover cannot resolve |
| Ambiguity resolution failure from base-side multipath | possible, and would look identical from the rover |

So the fault is almost certainly in the *content* of the corrections, not their
delivery. A rover-side resync cannot fix that, which is exactly what was
observed; only restarting the base — forcing a fresh survey — can.

**Monitoring that follows from this.** `last_error` is a genuine base-station
observable and should be watched. ⚠️ Two traps: its timestamp is **UTC**, and it
**latches** — it was still showing the 18:19 error at 21:12 with RTK healthy and
`Fix`. HA also picked it up ~46 minutes after the device recorded it, so it is
not a real-time signal. Use `last_error_time`, not the state's `last_changed`.

Remaining ranked hypotheses, with what would distinguish each:

1. **Base survey-in never completed or completed badly.** Most consistent with
   the evidence: rover fine, corrections arriving, only a base restart helps.
   Distinguisher — whether the base exposes survey state or a fixed reference
   position anywhere in its UI or telemetry.
2. **Base firmware hang** emitting stale or malformed corrections. Distinguisher
   — recurrence pattern and whether it correlates with uptime.
3. **Ambiguity-resolution failure from base-side multipath.** Distinguisher —
   whether the base has a clear sky view, and whether Float recurs at the same
   time of day (constellation geometry).

⚠️ **SUPERSEDED 2026-08-07 22:0x — the base station IS reachable.** This
paragraph originally read "none of these is reachable from the mower's
telemetry… the integration cannot diagnose the base station". That is wrong.
See `docs/vendor-tool-analysis-20260807.md`.

pymammotion ships `proto/basestation.proto` defining
`request_basestation_info_t` / `response_basestation_info_t`, exposing
`rtk_status`, `sats_num`, and a `base_score` block containing **`base_moved`**
and **`base_moving`** — precisely the survey-hypothesis discriminator. This
integration never requests any of it; `basestation_info` is only read for
display. The base's GNSS receiver is also identified as a **Unicore UM980**,
whose survey-in / fixed-position behaviour matches the observed signature
exactly.

So the next step is concrete rather than a waiting game: query the base
read-only, log `base_score` alongside RTK state, and catch the next Float
episode with both. A recurrence log is still worth keeping — date, duration,
what cleared it — but it is no longer the only avenue.

### P3b — 🔑 the correction chain is NOT what this plan assumed (2026-08-07 22:5x)

First live base-station query, on beta28. Evidence:
`docs/evidence-basestation-query-20260807.json`. No motion commanded.

**The single most important finding, and it reframes P3.** This installation has
a **separate RTK device**, `rtkbna235279309`, with its own entities. Reading them
directly:

| | |
| --- | --- |
| base `position_mode` | **`rtk_over_internet`** |
| base `connection_type` | `con_wifi`, `wifi_rssi` **−72 dBm** |
| base satellites | 26 |
| mower `position_mode` | `rtk_over_datalink` |

So the correction chain is **internet source → base station (WiFi) → LoRa E22 →
mower**. The base is **not** running on its own surveyed position as the leading
hypothesis assumed; it takes corrections from the internet and relays them.

That makes a whole class of upstream failure — NTRIP/MQTT correction outage, WiFi
drop, ISP hiccup — capable of degrading what the base relays **without changing
anything the base reports about itself**, while the LoRa datalink stays up and
satellites stay visible. That is precisely the 2026-08-07 signature, and it
explains both why a rover-side `sync_rtk_and_dock` did nothing and why only a
power cycle (forcing a WiFi/correction-source reconnect) helped.

**What the base's own history shows.** Its self-reported position moved exactly
once: `34.0245718145, −84.7698523611666` → `34.0245515906667, −84.7698970755001`
at **22:19:14 UTC = 18:19:14 EDT**, reverting at 19:05 EDT. That is **4.7 m**,
and it is the **power cycle**, not the Float onset — the base re-deriving its
position on reboot and converging back over ~47 min. A base that converges
correctly.

⚠️ **No change is recorded in base `position_mode`, satellites, latitude or
longitude across the 15:40 EDT Float onset.** But RTK-device telemetry is
recorded only ~9 times in 12.5 h — roughly hourly — so this negative is **weakly
resolved** and cannot exclude a sub-hour transient.

**Net effect on the ranked hypotheses:** "base survey never converged" is
**demoted** — the base doesn't primarily survey, and it demonstrably converges
when it does. "Corrections wrong at the source, upstream of the base" is now
**leading**, and it is monitorable: `position_mode`, `wifi_rssi` and
`connect_status_since_poweron` are all readable per-tick rather than needing a
query the hardware may not answer.

**The probe also found a defect in itself.** `basestation_info_probe` sent the
command and sampled 101 times at 150 ms over 15 s without ever seeing a
query-only field, with no clobber detected — so the report-channel race is ruled
out and it is a real negative *for that read path*. But the path may be wrong:
the probe reads `mower.report_data.basestation_info`, while `base.to_app` frames
carrying the RTK device's own `iot_id` are reduced by `RTKStateReducer` onto
`RTKBaseStationDevice` instead. Before concluding the hardware ignores
`request_basestation_info_t`, the probe must also read the RTK device's state.

### P4 — the operator runbook

Short, in `docs/deploy-runbook-p0.md`: if RTK reads Float, do not spend a session
on it. Confirm reception is healthy (co-viewed satellites non-zero), try
`sync_rtk_and_dock` once, then **power-cycle the dock and base station** and
allow ~20 minutes. That is the sequence that worked; the rover-side steps did
not.

## 5. What is deliberately not being done

- **No attempt to gate on freshness with a tuned constant.** Two attempts, two
  false-blocks, and P2 now measures legitimate quiet at ~1 h. Settled.
- ⚠️ ~~**No base-station integration work.** Nothing in the mower's telemetry
  reaches the base's internal state.~~ **Struck 2026-08-07.** This was wrong and
  is the same error corrected in P3: `last_error` carries reference-station
  events, and `basestation.proto` exposes a full query. The read-only
  `basestation_info_probe` was built on it (`6093506c`). What remains out of
  scope is *writing* base configuration — including the
  `app_to_base_mqtt_rtk_t` NTRIP-caster fields — which is an operator decision,
  not a diagnostic step.
- **No change to `waypoint_tolerance`.** That decision belongs with the separate
  cadence finding (position updates ~1031 ms during motion, ~47 cm travelled
  between updates), not with RTK.

## 6. Verification

- Full matrix per `CLAUDE.md`: pytest with coverage, ruff check, ruff format
  check, scoped mypy, `npm run test:frontend`, all-files pre-commit.
- Tests must cover: freshness reported but never blocking; the quality gate
  refusing Float/Single/None; the override permitting while still recording
  `rtk_degraded: true`; unknown status staying advisory.
- Deploy verification per the runbook: 46/46 file md5 match, zero AppleDouble,
  both card paths identical, backend verified, gate off, no session.
- Post-deploy: confirm on the host that `rtk_telemetry_stale` no longer appears
  in `blockers` while `rtk_report_age_seconds` is still reported, and that a
  normal Fix state produces an empty blocker list.

## 7. Open questions

- ~~Is legitimate RTK quiet bounded?~~ **Answered (P2):** it reaches ~1 h, so no
  usable threshold exists. Closed.
- ~~Is there any early-warning signal before a Float transition?~~ **Answered
  (P2):** none found; satellite count and signal quality track sky view, not
  solution state. Closed.
- Why did the base fail? (P3.) Still open, but no longer purely a waiting game —
  the base-station firmware analysis in
  `docs/vendor-tool-analysis-20260807.md` §6 found `setRTKBaseLocation`,
  `rtk_position_reset` and a "basestation pos status" guard, so the base
  **stores** a reference position rather than re-surveying each boot. That is a
  concrete mechanism for corrections-against-a-wrong-reference, and
  `base_moved` / `base_moving` may be that same guard surfaced to the app.
  Needs the live query plus a recurrence.
- Does the mower ever report a latched **Fix** that has stopped being true? Still
  never observed.
