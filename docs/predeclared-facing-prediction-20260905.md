# Predeclared: does the corrected facing derivation predict the driven direction on the ground? (2026-09-05)

**Committed before any run of this series exists.** Written per CLAUDE.md →
"How this project works": the criterion, the falsifier and the abort rules are
fixed here, in advance, so that choosing a rule after seeing which verdicts it
flips is not possible.

Build under test: **beta102** (`09e4335e`, deployed 2026-09-05, browser-confirmed
`card v0.6.4-beta102`). Backend `chorty-0.8.12.post4`.

🛑 **This is a NEW measurement, not a resumption of the failed 4.0 m reliability
series** (`docs/findings-clicktopath-reliability-4m-20260904.md`). Nothing here
reopens a standing decision. `docs/accepted-profile.json` is not touched and no
Gate 5 is owed or claimed.

---

## 1. The question

**Does `runtime_state.map_facing` — the reflection-based, motion-confirmed
facing shipped in beta102 — predict the direction the mower actually drives?**

That is the single thing 2026-09-04 could not answer. The additive offset
`toward + calibrated_forward_heading_offset_degrees` was measured wrong by a
mean **87.478°** on 43 real pulses; the compass mirror `90.13 - toward` was
right to a mean **1.000°**. Both figures come from banked telemetry. This series
asks whether the shipped derivation reproduces that on the ground, today, with
an operator watching.

## 2. 🔑 Why this shape is safe: the prediction does not steer

**Every leg is a bounded FORWARD drive.** No heading is supplied to the mower and
no target is placed. The mower goes wherever it is already facing; only *my
prediction of that direction* is on trial.

That inverts the 2026-09-04 incident, in which a facing computed from the wrong
model was used to *place a target*, and the mower dutifully drove to it. Here a
wrong prediction produces a wrong number in a record and moves nothing. This is
deliberate and is the reason the series can run at all with facing currently
unknown.

⚠️ **A "small test move" is still an armed dispatch.** Every leg gets the full
per-run protocol in §6.

## 3. Instrument and frozen parameters

Service: **`raw_pymammotion_motion_probe`**, chosen because it carries a real
**distance** guard (`max_travel_m`) rather than only a clock bound.

| parameter | value | why this value |
| --- | --- | --- |
| `command` | `send_movement` | |
| `linear_speed` | **400** | the calibrated command; sustained **0.295 m/s** measured post-ramp |
| `angular_speed` | **0** | no commanded rotation anywhere in this series |
| `motion_refresh_interval_ms` | **200** | app-parity cadence. Without it a single-shot pulse travels ~0.10 m, which is at the position feed's noise floor and cannot yield a bearing |
| `duration_ms` | **2400** | 0.708 m nominal at 0.295 m/s — deliberately LONGER than the distance bound needs, so distance is what stops the leg and the clock is only a backstop |
| `max_travel_m` | **0.40** | the real bound |
| `in_window_sample_interval_ms` | **100** | dense enough to reconstruct the displacement vector |
| `prefer_ble` | **true** | |
| `dry_run` | **false** on scored legs, **true** on every pre-dispatch check | |
| `confirm_blades_off` / `confirm_clear_area` | **true** | operator-confirmed each leg |

⚠️ **These are frozen. A parameter change voids the series** and requires a new
predeclaration. Echoed values are verified against this table on every leg;
a mismatch discards that leg.

### 3.1 The bound, stated honestly

`_PROBE_TRAVEL_GUARD_OVERSHOOT_M` is **0.50 m** — the guard lets the mower run
that far past `max_travel_m` before the stop lands, and the constant's own
comment explains why it was raised from 0.35 after two firings overshot 0.276 and
0.307 m.

🔑 **So the honest worst case per leg is `0.40 + 0.50` = 0.90 m, not 0.40 m.**
The clock backstop independently caps travel at ~0.71 m plus stop overshoot.
**Both bounds are stated to the operator before each dispatch as "up to about
0.9 m".**

✅ **Confirming the bound exceeds what the run needs to demonstrate its
criterion**, per the runbook rule: a bearing over 0.40 m of travel against a
2–4 cm position noise floor carries an angular uncertainty of roughly
`atan(0.04 / 0.40)` ≈ **5.7°** worst case, and ~2.9° at the 2 cm floor. The
criterion below is 10.0°, which exceeds that. A shorter leg would make the
measurement impossible; a longer one is not needed.

## 4. Corridor requirement — physical, not software

🚨 **The gate does not check the map, and the map does not check the ground.**
On 2026-09-04 a map-polygon corridor scan showed 3.5 m where the operator's tape
measured 2.79 m to a real fence — 0.71 m of error in the unsafe direction,
invisible to both the containment gate and the polygon scan CLAUDE.md prescribes
as the gate's backstop.

**Therefore, before every leg:**

1. The operator **physically measures** clearance in the direction the mower is
   facing.
2. **Required: ≥ 2.0 m.** That is more than twice the 0.90 m worst case.
3. A fresh map-polygon corridor scan is also run — as a second check, never as a
   substitute for the tape.
4. If either check fails, or they disagree, the leg is not dispatched.

## 5. Legs, and which of them are scored

### ✏️ AMENDED 2026-09-05, BEFORE ANY LEG WAS DISPATCHED — n = 4, all 4 scored

**The precondition for excluding leg 1 stopped being true before the series
started, so the criterion TIGHTENS from 3 of 3 to 4 of 4.**

Leg 1 was reserved as an unscored re-anchor drive because, as originally
written, the mower was docked with its two heading sources disagreeing by
**178.391°** and `map_facing.confidence` reading `unknown` — and the shipped
model refuses an unconfirmed facing, so scoring it would have scored a case the
code already declines to stand behind.

The operator then started, paused and cancelled a mow session to get the mower
off the dock. **That drive re-anchored the estimate on its own.** Before any leg
of this series was dispatched, `map_facing` read:

| source | map bearing |
| --- | --- |
| `vio_heading` | 279.336° |
| compass mirror | 279.985° |
| last driven leg | 280.952° |

`confidence: motion_confirmed`, `safe_to_aim_dispatch: true`, disagreement
collapsed from 178.391° to **0.649°**.

🔑 **Every leg will therefore be dispatched `motion_confirmed`, which §5 already
names as the condition for a leg to be scored.** Leg 1 now meets that condition
by the rule as written; excluding it would mean discarding a valid leg on a
technicality.

⚠️ **This amendment is recorded because amending a predeclaration is exactly the
move the discipline exists to police.** Two facts make it legitimate, and both
are checkable: it was made **before any data existed** — no verdict can have been
flipped by it — and it moves the bar **up**, from 3 of 3 to 4 of 4, not down.
🛑 **No comparable amendment may be made once a leg has run.**

The re-anchoring itself becomes a §7 secondary observation rather than leg 1's
job, and it is already answered: the collapse from 178.391° to 0.649° is the
behaviour the model predicts, observed unprompted.

### Legs 1, 2, 3 and 4 — all scored.

Each must be dispatched with `map_facing.confidence == "motion_confirmed"` and
`safe_to_aim_dispatch == true`, read from `export_runtime_state` immediately
before dispatch. **A leg dispatched without that is not scored and the series is
reported at the reduced n**, per "partial n is a result, not a draft".

## 6. Per-leg protocol — every item, every leg

1. Read `export_runtime_state`; record `map_facing` and `current_orientation`
   verbatim.
2. **Derive facing two ways** — the last driven leg's bearing (from
   `map_facing.motion_evidence`) and `(90.13 - toward)` with `toward` fresh — and
   require agreement. On disagreement, trust the mirror and say so.
3. **State the destination to the operator in compass and landmark terms** — "it
   will drive roughly WSW, toward the fence, up to about 0.9 m" — and have them
   confirm the ground matches.
4. Operator tape-measures clearance (§4). Fresh map corridor scan.
5. Dry run. Confirm zero blockers and the parameter echo against §3.
6. **Explicit operator go/no-go, immediately before dispatch.**
7. Dispatch. Record the full probe response.
8. Operator states, in their own words, which direction the mower actually went.
9. Re-read `export_runtime_state`.

## 7. Criterion — fixed here, before any data exists

Per leg, define:

- `predicted` = `map_facing.map_facing_degrees` read immediately before dispatch.
- `measured` = `atan2(dy, dx)` over the leg's start→end position, from the
  probe's own in-window samples.
- `error` = the signed-normalised absolute difference, in degrees.

> ### **PASS = 4 of 4 scored legs with `error` ≤ 10.0°.**
>
> *(Was 3 of 3; tightened by the §5 amendment before any leg was dispatched.)*

⚠️ **The criterion is deliberately strict at 4/4, not "≥ 3 of 4".** With the
banked mirror error at a mean 1.000° and a max of 3.003°, and the measurement
noise floor at ~5.7°, a leg above 10° is not bad luck — it is the model failing.
**3 of 4 is a FAIL with an informative failure**, and the response is a
predeclared follow-up, not a retroactive softening of this line.

### The falsifier, stated plainly

🔑 **If any scored leg exceeds 10.0°, the shipped facing derivation does not
predict the ground and must not be used to aim a dispatch.** In that case
`safe_to_aim_dispatch` is overclaiming and beta102's §2 change is wrong. That
is the result this series exists to be able to return.

### Secondary observations — RECORDED, NOT SCORED

Putting a criterion on these would be fitting a rule to what is already
suspected. They are recorded per-item and reported as observations:

- ✅ **Already answered before the series began**: whether the estimate
  re-anchors on real motion. The operator's undocking mow session collapsed
  `current_orientation.disagreement_degrees` from **178.391° to 0.649°** and
  flipped `map_facing.confidence` to `motion_confirmed`, unprompted. Recorded as
  an observation; it was never a scored claim.
- Whether `toward` and `vio_heading` jump on any leg the way they did on
  2026-09-04 (~166°), and by how much.
- Actual travel per leg against the 0.40 m bound, and the realised guard
  overshoot — two prior firings are the whole sample, so this is a third and
  fourth data point on `_PROBE_TRAVEL_GUARD_OVERSHOOT_M`, recorded, **not**
  fitted to.
- Any device fault code surfacing through the new `last_error` path,
  `1309` especially.

## 8. Abort rules — any one of these ends the series

Fixed here, before dispatch:

- The operator withholds go/no-go, or reports the mower went somewhere they did
  not expect. **Immediate stop, no further legs.**
- On a scored leg, the operator's stated facing and `map_facing` disagree by more
  than **20°**.
- Tape-measured clearance below **2.0 m**, or the tape and the map disagree.
- `ble_rssi` below **-70 dBm** at dispatch, or two consecutive BLE refusals.
- `vio_tracked_features` below **40**, or `vio_feed_live` false.
- Battery below **35%** off dock.
- Loss of daylight. Standing decision 4 (night is CLOSED) binds.
- Any named refusal from the executor. 🔑 **A run that stops safely on a named
  refusal is a FAIL, not a smaller number.**

## 9. What a PASS authorizes

**Only this**: that `map_facing.safe_to_aim_dispatch` may be relied on to aim a
dispatch, in this yard, at this scale. It authorizes **no** change to
`docs/accepted-profile.json`, no change to any bound, tolerance or budget, and no
resumption of the 4.0 m series — which, per its own findings, must first fix how
an aligned start is established.

## 10. What a FAIL authorizes

**Nothing beyond writing up the failure mode**, and reverting reliance on
`safe_to_aim_dispatch`.
