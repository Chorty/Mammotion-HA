# The turn's own translation is what puts the mower off-bearing

**2026-08-10, derived entirely off-mower from committed evidence. No motion was
commanded to produce this.** Sources:
`docs/evidence-beta32-4segment-20260810T185433Z.json` and
`docs/evidence-beta32-4segment-20260810T193833Z.json`.

## The question this answers

Both 2026-08-10 runs left segments finishing their turns well aligned — the turn
phase reported final heading errors of −3.7 / −5.1 / +4.6° — and landing anyway at
0.1229–0.1447 m against a 0.15 m tolerance. The aim error then grew to 18–35°
during the leg. The open question was whether that was **genuine heading drift**
(the mower physically turning while driving straight) or **bearing rotation**
(the bearing to the target swinging as cross-track error accumulated).

**It is neither.** The error is present before the first linear pulse, and it is
caused by the turn itself.

## The mechanism

A VIO turn does not pivot in place. It displaces the mower — `post_turn_alignment.
turn_displacement_m` measured 0.028–0.131 m here, against a
`max_turn_translation_distance` cap of 0.30. Being displaced sideways at the
*start* of a 0.6–0.7 m leg rotates the bearing to the target by roughly
`atan(translation / leg)`.

The turn primitive closes on **VIO body heading**. It cannot see this, because
nothing about the mower's heading changed — the *target's* bearing moved.

## The evidence

`post_turn_alignment.before.aim_error_degrees` is the executor's own map-frame aim
error immediately after the turn. Compared against the VIO-frame error the turn
phase reported:

| run | seg | leg | translation | VIO-frame err | map-frame err | \|difference\| | `atan(transl/leg)` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 193833 | 2 | 0.599 m | 0.1310 m | −3.732° | **+7.360°** | 11.09° | 12.34° |
| 193833 | 3 | 0.674 m | 0.0768 m | −5.073° | **−11.452°** | 6.38° | 6.50° |
| 193833 | 4 | 0.649 m | 0.0681 m | +4.585° | **+10.607°** | 6.02° | 5.99° |
| 185433 | 1 | 0.700 m | 0.0283 m | +1.716° | **+3.079°** | 1.36° | 2.32° |
| 185433 | 2 | 0.746 m | 0.0698 m | −12.447° | **−7.304°** | 5.14° | 5.34° |

**The last two columns are the same quantity.** Residuals: +1.25, +0.12, −0.03,
+0.95, +0.20 degrees. Three of five agree to better than 0.2°.

The leg length used as the denominator is `true_start → target`; the geometrically
exact denominator is the post-turn distance to target, which is not separately
recorded. That approximation is worth at most a few tenths of a degree at these
translations and does not affect the conclusion.

## It repairs the landing model

`landing = 0.62 × leg·sin(aim) + 0.065` fed with each aim error:

```
aim source     residuals (predicted - actual, metres)          mean |residual|
map-frame     -0.0305  +0.0033  +0.0162  +0.0205  -0.0185          0.0178
VIO-frame     -0.0539  -0.0427  -0.0257  +0.0101  +0.0224          0.0310
```

The model is ~1.7x better on the map-frame aim, and on the three 60°-run segments
the VIO-frame version under-predicts by 26–54 mm while the map-frame version is
mixed-sign. **The model was not wrong; it was being fed the wrong aim error.**

⚠️ n = 5. The landing-model comparison is suggestive, not decisive. The
turn-translation identity in the table above is the strong result.

## What it means for the profile conflict

**`heading_tolerance_degrees` is the wrong lever, and lowering it 18 → 11 would
have changed none of these five segments.** That key governs the VIO-frame error,
which averaged 5.5° and was already comfortably inside any tolerance under
discussion. The landing is set by the map-frame error, which averaged 8.0° and
which that key does not control.

The lever that does control it is **not a `LUBA_ACCEPTANCE_PROFILE` key.** The
post-turn gate already exists (`services.py`, the `post_turn_alignment` block):

```python
alignment_tolerance = min(
    float(heading_tolerance_degrees),      # 18  <- PROFILE KEY
    float(vio_realign_threshold_degrees),  # 15  <- backend default, NOT a profile key
)
```

It evaluates to **15**, and all five map-frame errors (3.079–11.452°) fell inside
it, so `correction_attempted` was `false` every time. Lowering
`vio_realign_threshold_degrees` 15 → ~5 makes it `min(18, 5) = 5` and catches all
five.

**Checked side effect: the mid-drive re-aim trigger does not move.** It is
`abs(aim) > vio_realign_threshold_degrees and abs(aim) > heading_tolerance_
degrees`, so at threshold 5 and tolerance 18 it still resolves to `aim > 18` —
identical to today. The change is surgical to the post-turn gate.

⚠️ **The unmeasured risk: a correction turn also translates.** Correcting 3–11°
costs a short pulse, so the induced translation should be far smaller than the
original — but that is reasoning, not measurement, and it is the same shape as the
problem it fixes. It needs a hardware check, not confidence.

## Two loose ends closed, so they are not re-chased

1. **`target_map_heading_degrees` is not buggy.** It appears to disagree with
   `atan2(true_start → target)` by up to 15.8°. It does not: the post-turn block
   deliberately overwrites it with the post-turn bearing
   (`result["target_map_heading_degrees"] = fresh_bearing`). The recorded value is
   correct and the turn aimed from the true start with the provided offset.
2. **There is no VIO→map frame-registration error.** The offset refresh derives
   from real measured displacement (`atan2(measured_delta dy, dx) − vision_
   heading`), and the offset stayed within ~1.3° across every segment of both
   runs. Travel bearing tracks VIO heading faithfully.

## Method note

The reconstruction that first suggested this — computing travel bearing from
`position_source_comparison.locations_xy` between pulses — is **contaminated** and
should not be reused. Measuring from `true_start` includes the turn's own
translation, which is the very effect under test, and a 0.07 m lateral offset on a
0.33 m step is ~12° of apparent bearing error. The executor's own
`post_turn_alignment` record is the right instrument and was already there.
