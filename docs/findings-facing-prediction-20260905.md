# The corrected facing derivation predicts the driven direction to a mean 1.382° — PASS at 4 of 4, with one scope limit that matters more than the verdict (2026-09-05)

Predeclared in `docs/predeclared-facing-prediction-20260905-ed2.md`, committed at
`d292e645` **before any scored leg existed**. First edition:
`docs/predeclared-facing-prediction-20260905.md` (`50fc34d2`), amended at
`c34ed965`, both before any leg was dispatched. Structured per-leg data:
`docs/evidence-facing-prediction-20260905.json`.

Build: **beta103** (`4f908b04`), backend `chorty-0.8.12.post4`. Service
`raw_pymammotion_motion_probe` dispatched directly; the card was not used.

---

## 0. 🚨 Read this before quoting the verdict

**This series does NOT discriminate the compass mirror from the additive offset,
and must never be cited as if it did.**

The two models are equal where `90.13 - toward == toward + 102.4`, i.e. at
`toward = 173.865°`. Every leg ran at `toward` between **167.53° and 170.15°** —
within **3.7–6.3°** of that crossing. The two models therefore differed by only
**7.44°** here, *less than the 10° criterion itself*.

🔑 **So the mower happened to be pointing almost exactly where the wrong model
looks right.** That is not bad luck; it is the precise circumstance
`_TOWARD_MIRROR_DEGREES`' own comment says allowed the additive offset to survive
undetected for months.

Under the additive offset the same four scored legs would have scored **8.849 /
9.126 / 12.192 / 14.297°** — a FAIL, but on only two of four legs and by a margin
that says nothing about model *form*.

✅ **What actually discriminates the two models is the banked 43-pulse dataset**
(`docs/evidence-clicktopath-reliability-4m-20260904.json`): mirror mean
**1.000°**, additive mean **87.478°**, across many headings. This series confirms
the shipped derivation works on the ground *today*; it adds essentially nothing
to the case against the additive offset.

⚠️ **A future series that wants to discriminate must start the mower at a
`toward` far from 173.865°** — near 353.865° the two models are 180° apart.

---

## 1. The verdict

Predeclaration §7 fixes **PASS = 4 of 4 scored legs with absolute error ≤ 10.0°**.

| leg | scored | predicted | measured | abs error | travel |
| --- | --- | --- | --- | --- | --- |
| A (re-anchor) | no | 279.125° | 281.744° | 2.619° | 0.6264 m |
| **1** | yes | 280.344° | 281.394° | **1.050°** | 0.6844 m |
| **2** | yes | 280.772° | 281.671° | **0.899°** | 0.6258 m |
| **3** | yes | 281.954° | 283.571° | **1.617°** | 0.6631 m |
| **4** | yes | 282.266° | 284.227° | **1.961°** | 0.6714 m |

> ## ✅ **PASS — 4 of 4, mean 1.382°, median 1.333°, max 1.961°, against a 10.0° bar.**

`predicted` is `map_facing.map_facing_degrees` read immediately before each
dispatch; `measured` is the executor's own
`motion_interpretation.movement_heading_degrees` over the settled displacement.
Every leg was dispatched with `confidence: motion_confirmed` and
`safe_to_aim_dispatch: true`, as §5 requires for a leg to be scored.

The max is **five times inside** the criterion and comfortably inside the
±3.3–3.7° bearing noise floor that 0.63–0.68 m of travel implies. This is
consistent with the banked mirror result (mean 1.000°) rather than an improvement
on it.

---

## 2. Safety and protocol

| | |
| --- | --- |
| safety gates | **11/11 passed** on all five dry runs and all five real dispatches |
| blockers | zero, every leg |
| named refusals | **zero** |
| operator | present and watching throughout, with a stop available at any point |
| corridor | map clearance **9.28 m** at start; net displacement **3.270 m** |
| gate after | **disarmed and verified from the live API AND RAW** `core.config_entries` |

⚠️ **One declared protocol deviation.** Predeclaration §6.6 asks for an explicit
operator go/no-go immediately before *each* leg. The four scored legs ran on a
**single standing go** covering the sequence, because the 300 s
motion-confirmation TTL cannot survive five separate confirmation round-trips —
the first leg after any pause is never `motion_confirmed`, which is exactly why
leg A exists. This was stated to the operator before the run and they retained a
stop at any point. 🔑 **Recorded as a deviation, not glossed as compliance.**

⚠️ **This is a real design tension worth naming**: a freshness rule tight enough
to catch a manual reposition is also tight enough to forbid a deliberative
per-leg confirmation loop. It is not obviously wrong — the conservative failure
is "ask a human" — but it makes the by-the-book protocol unrunnable as written,
and the next predeclaration should say what it wants instead.

---

## 3. The guard fix is confirmed on hardware

The two void beta102 dispatches (§7.3 of the predeclaration) died at 344 ms and
273 ms having sent **1 of 11** refresh writes, travelling 0.0949 m and 0.0578 m
against a 0.40 m bound. After `4f908b04`:

| | before | after |
| --- | --- | --- |
| refresh writes | 1 of 11, both runs | **10 or 11 of 11**, all five |
| travel | 0.058–0.095 m | **0.626–0.684 m** |
| false `position_sequence_gap` trips | 2 of 2 | **0 of 5** |
| window survived | 273–344 ms | 2125–2367 ms |

✅ **And the guard still does its real job**: two legs stopped on
`max_travel_reached` at **0.4296 m** and **0.4006 m** against the 0.40 m bound.
The fix removed a false trip without removing the true one.

**Overshoot past the trip point: 0.1968 m and 0.2252 m.** Against the prior
sample of 0.276 and 0.307 m that makes four points, all inside the 0.50 m
`_PROBE_TRAVEL_GUARD_OVERSHOOT_M` allowance. ⚠️ **n = 4. Do not fit anything to
it** — record it as the constant continuing to hold.

---

## 4. Secondary observations — recorded, not scored

Predeclaration §7 puts these outside the criterion on purpose. Putting a bar on
them after the fact would be fitting a rule to what was already suspected.

### 4.1 ⚠️ Every error has the same sign

All five signed errors are **positive**: +2.619, +1.050, +0.899, +1.617, +1.961.
The mower drives consistently **clockwise of where its own estimate points**, by
a mean of +1.382° across the scored legs.

🔑 **Five of five with the same sign is a systematic effect, not noise** — noise
would scatter. But the magnitude sits inside the bearing noise floor of each
individual leg, so the *size* is not established even though the *direction* is.
🛑 **n = 5. Do not fit a correction to it.** It is recorded as the direction to
check first if this is ever revisited, and a future series wanting to establish
it needs longer legs, not more of these.

### 4.2 Re-anchoring, confirmed twice

The shipped model says a stale heading estimate cannot recover until the mower
drives. Observed twice today, both unprompted:

1. On the dock the two heading sources disagreed by **178.391°**
   (VIO 89.967, mirror 271.575), `confidence: unknown`. The operator's undocking
   mow session collapsed it to **0.649°** and flipped `confidence` to
   `motion_confirmed`.
2. The beta103 restart cleared the in-memory motion tracker, dropping
   `confidence` to `corroborated_not_motion_confirmed` with
   `reason: no_driven_leg_on_record`. Leg A restored `motion_confirmed`.

⚠️ **(2) is a real operational cost, not just an observation:** the facing
tracker does not survive a Home Assistant restart, so the first dispatch after
every restart is unscorable by construction. Whether that should persist is a
design question this series does not answer.

### 4.3 The facing drifted ~4° over the series, and all three sources tracked it

| after leg | VIO | mirror | last driven leg |
| --- | --- | --- | --- |
| A | 280.344 | 279.985 | 279.413 |
| 1 | 280.772 | 279.985 | 280.559 |
| 2 | 281.954 | 281.151 | 283.069 |
| 3 | 282.266 | 282.600 | 283.808 |
| 4 | 283.442 | 283.801 | 285.087 |

The mower veered gently right across 3.27 m of driving. The three independent
sources stayed within ~1.8° of each other throughout, which is the corroboration
the design depends on behaving as intended.

### 4.4 🔴 An undocumented fault code, 5004, surfaced twice

At **14:45:07** and **15:11:40 UTC**, reaching the operator as:

> `5004: no description available for this code`

`5004` is absent from the bundled 449-code table, absent from the vendor APK's
368-row `servicecode.csv`, and the live cloud table — which **is** loaded — has no
row for it.

🔑 **This is the beta102 error path degrading exactly as designed.** Before
today it would have rendered as `"mcu: , "` or `"Error message not found"`; the
operator now gets a number they can search for and report. ⚠️ **What 5004 means
is unknown and is not guessed at here.** It is the first genuinely unmapped code
seen since the change and is worth watching for a correlation with motion.

### 4.5 Link and battery

BLE finished at **-76 dBm** — the documented death threshold — having started at
-54. Battery 100% → 81% across the session. No leg suffered a transport failure,
but a longer series would have run into the link.

---

## 5. What this authorizes

Per predeclaration §9: **only** that `map_facing.safe_to_aim_dispatch` may be
relied on to aim a dispatch, in this yard, at this scale.

🛑 It authorizes **no** change to `docs/accepted-profile.json`, no change to any
bound, tolerance or budget, and **no resumption of the 4.0 m reliability
series** — which per its own findings must first fix how an aligned start is
established. Nothing here reopens a standing decision.

⚠️ And per §0 above: it authorizes **nothing whatsoever** to be said about the
additive offset. That case rests entirely on the banked 43-pulse data.
