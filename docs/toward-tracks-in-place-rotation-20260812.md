# 🚨 `toward` is NOT blind to in-place rotation — the night premise is refuted

**2026-08-12 night, in full darkness with VIO dead** (`tracked_features: 0`,
`camera_brightness: dark`). Two armed pivots plus a passive mowing trace.
Evidence: `docs/evidence-night-pivot-20260812.json`,
`docs/evidence-night-pivot-reverse-20260812.json`, and the sampled mowing track
summarised in §2.

## 1. The measurement

Pure in-place rotation — `linear_speed: 0` — held open at the app cadence and
explicitly stopped:

| | command | translation | `toward` change | rate |
| --- | --- | --- | --- | --- |
| pivot 1 | `angular +500`, 2410 ms | **0.0382 m** | **+99.55°** | 41.3 °/s |
| pivot 2 | `angular −500`, 2251 ms | **0.0295 m** | **−61.43°** | 27.3 °/s |

**Opposite command, opposite sign, ~3 cm of travel each.**

The contrast with the same evening's translating motion is what makes it
unambiguous:

```
arc  angular 180   0.5823 m  ->  +22.20 deg
arc  angular 300   0.6198 m  ->  +43.42 deg
arc  angular 500   0.5184 m  ->  +66.85 deg
linear only        0.5840 m  ->   +0.00 deg
PIVOT angular 500  0.0382 m  ->  +99.55 deg     15x less travel, MORE rotation
```

**99.55° cannot be a course-over-ground computed from 3.8 cm.** This machine's
position noise floor is 2–4 cm, so a bearing derived from that displacement is
arbitrary; it could not twice produce a value matching a real pivot, with the
correct sign, from the commanded direction.

⚠️ The two rates differ by 50%. The record explains it — pivot 2 delivered only
**3 refresh writes** against 8 — which is the h-watchdog signature measured all
week on linear travel. That is an explanation, not a measurement.

## 2. Corroboration: the vendor's own night mowing

Sampled passively for 184 s while the mower mowed in the dark, zero commands
sent. 60 samples, `MODE_WORKING`, RTK Fix, VIO at zero throughout.

```
course change >8 deg while translating >0.15 m :  6
course change >8 deg with <0.15 m of travel    : 21
```

Straight rows at a very steady **0.46 m/s** with `toward` constant to 0.00°,
then at each row end a burst of near-stationary steps:

```
0.0083 m  +36.30 deg
0.0111 m  +32.09 deg
0.0901 m  +35.77 deg
0.0363 m  +39.31 deg
```

~157° of rotation in ~0.4 m of travel, four times over. **The vendor pivots at
night; it does not arc** — and `toward` follows those pivots.

🗑️ That also **refutes an inference of my own from the same day**: the dock
return led me to write that the vendor's approach phase is an arc. It is not.
Arcs remain real and measured (`docs/arcs-work-20260812.md`), but they are not
what the vendor does.

## 3. What this overturns

`docs/night-motion-options-20260811.md` §1 says, as the sharpened explanation for
why `turn_mode: legacy` cannot work:

> *"the legacy turn closes on `position.toward`, which is course-over-ground. A
> mower rotating in place does not translate, so `toward` does not change, so the
> loop is blind to the rotation **at any hour of the day**."*

**That is refuted.** `toward` reports a heading that tracks in-place rotation.
The premise is load-bearing: it is why every turn is forced onto VIO, and
therefore why closed-loop motion is daylight-only.

**Consequence: the night path may be far simpler than an arc controller.** A
legacy-style turn closing on `toward` is plausible in the dark, and that
primitive already exists rather than needing to be written.

## 4. Why the earlier observation disagreed — inferred, not proven

The 2026-08-02 finding that `toward` "stays frozen after an in-place pivot" has
two candidate explanations, both of which were live that day:

1. **The stale-feed defect.** The mower does not push position while stationary,
   so a reader that does not force `request_reports_count_5` sees a cached value.
   That exact defect existed in `raw_pymammotion_motion_probe` until today and
   **fooled this session for five minutes** on the first arc — four bit-identical
   samples that were nearly written up as "the arc did not actuate".
2. **Angular 180.** The 2026-07-25 turn A/B used it, and it barely actuates a
   stationary pivot (~3° total). Tonight's pivots used 500.

⚠️ **Neither is proven.** Nobody has re-run the 2026-08-02 test with a forced
readback. Until someone does, "the old observation was an artefact" is the
likeliest reading and not a fact.

## 5. What is NOT established

- **Two pivots is two pivots.** n = 2, rates 50% apart.
- **No closed loop has run on `toward`.** Reporting correctly and being
  *steerable* are different claims — the same distinction drawn about arcs
  earlier the same day.
- **`toward`'s latency during rotation is unmeasured.** These are settled values
  read after the pulse, not tracking during it. A control loop needs the latter.
- **Nothing was driven closed-loop at night.** The `vio_active` gate still
  refuses a night segment, correctly, because nothing has yet shown a turn can
  *close* on `toward`.

## 6. Next

1. **Re-run the 2026-08-02 in-place turn test** with the readback fix, to settle
   §4 rather than infer it.
2. **Measure `toward` DURING a pivot** — sample at ~200 ms through the window
   rather than reading the settled value. That is what decides whether a loop can
   close on it.
3. Only then consider a night turn mode. ⚠️ It would need its own gate story: the
   `vio_active` refusal exists for good reasons and must not simply be deleted.
