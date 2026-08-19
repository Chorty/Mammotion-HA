# How accurate is `toward`? — measured across 169 pulses, 2026-08-18

**`toward` is accurate to about a degree as course-over-ground.** Its weakness
is not precision; it is *when* it updates.

Measured from data already on disk — no new mower run. Every recorded forward
pulse carries two independent headings: `position.toward` as published by the
mower, and the movement vector `atan2(dy, dx)` derived from RTK position deltas.
Comparing them tests `toward` directly.

## Method

169 forward pulses across **30 evidence files**, filtered to travel ≥ 0.20 m
(below that the 2-4 cm position-feed noise floor dominates the derived heading —
the same threshold `_NIGHT_MIN_AIM_BASELINE_M` uses, and for the same reason).

The hypothesis under test is the mirror this project already uses:
`map_bearing = C − toward`, i.e. `movement_heading + toward` should be constant.

## Result

    movement_heading + toward   circular mean 89.819°   circular SD 5.538°

    |deviation|   mean 1.415°   median 0.775°   max 75.512°
    within 1°   59.8%
    within 3°   98.2%
    within 5°   99.4%

⚠️ **The SD is misleading and the median is the honest figure.** 5.538° is
inflated almost entirely by a single sample; 98.2% of pulses land within 3°.

The measured constant **89.819°** sits 0.31° from the **90.13°** this project
already records as `toward_mirror_degrees`. Independent corroboration of that
constant from a much larger sample than it was derived on.

## Accuracy improves with travel distance, exactly as noise-limited geometry says

| travel | n | median \|dev\| | 90th pct | max |
| --- | --- | --- | --- | --- |
| 0.05–0.10 m | 6 | 2.26° | 3.97° | 5.52° |
| 0.10–0.20 m | 34 | 1.52° | 2.51° | 9.08° |
| 0.20–0.35 m | 68 | **0.88°** | 2.11° | 3.31° |
| ≥ 0.35 m | 101 | **0.76°** | 1.84° | 75.51° |

A fixed position error subtends a smaller angle over a longer baseline, so the
derived heading sharpens as the pulse lengthens. Note this cuts both ways: part
of the scatter at short travel is the *measurement*, not `toward`.

## 🔑 Exactly one outlier in 169, and it is the known one

    dev 75.51°  travel 0.436 m  move_hdg 331.12  toward 43.19  pulse 1
    evidence-night-segment-turn-quantum-20260813T214605Z.json

That is the **§7 item 15** case already on record: *"travelled 0.43648 m on map
bearing 331.1213°"*, aim error 81.416°, which stopped the run with
`night_reaim_required_but_unavailable`. This corpus-wide sweep rediscovered it
independently and found **nothing else** — which both corroborates the anomaly
and bounds it as isolated rather than typical.

Its distinguishing feature: it is a forward pulse taken **immediately after a
night turn**. Every other sample in the set follows either a straight run or a
VIO-path turn.

## What this does and does not license

✅ **Established:** `toward` reports course-over-ground to ~0.8° median, ~2°
at p90 on pulses of 0.2 m or more, and the mirror constant is ~89.8-90.1°.
As a *travel-direction* sensor it is good.

❌ **NOT established:** that `toward` is an accurate **body heading after a
rotation**. Every sample here is validated against travel, so the test only
speaks to pulses where the mower was moving forward. The one case that isolates
post-turn behaviour is the single outlier, and item 15 remains open.

**So the earlier framing needs correcting.** `toward` is not too *imprecise* to
steer by — at ~1° it is comparable to VIO. What makes it hard to close a loop on
is *timing and granularity*: it stays bit-identical through a bounded pulse and
arrives as one post-hoc step (item 16, 73 samples), and the refreshed night turn
quantum is 48.15° ± 5.70 with nothing scaling it. Accuracy was never the
limitation.

⚠️ Reproduce with the snippet in this file's git history, or re-derive: pair
`progress_diagnostics[].movement_vector_heading_degrees` with the
`samples[linear_N].telemetry.position.toward` of the same `command_index`,
filter travel ≥ 0.20 m, and take circular statistics of the sum.
