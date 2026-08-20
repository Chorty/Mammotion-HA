# Containment cannot see no-go zones — found by driving into one

**2026-08-20, `0.6.4-beta62`, supervised. The mower pushed a trampoline inside a
no-go zone. The operator judged no damage and let it continue.**

## What happened

The first hardware Route B run: one 10.8 m click at bearing 38°, split into 3
collinear sub-legs of 3.599987 m. Sub-leg 1 reached target at **0.092 m**.
Sub-leg 2 drove into a no-go zone containing a trampoline and stopped on
`telemetry_quality_degraded` 1.3724 m from its target.

Evidence: `docs/evidence-routeb-first-hardware-run-20260820.json`.

## ✅ RESOLVED IN CODE, SAME DAY — the geometry was never missing

`HashList` stores map geometry in **per-type dicts on the same object**:

```
area:                  dict[int, FrameList]   # type 0   <- the only one we read
obstacle:              dict[int, FrameList]   # type 1   <- "Keep-out / no-go obstacle boundary points"
no_go_zone:            dict[int, FrameList]   # type 23
no_go_zone_variant:    dict[int, FrameList]   # type 22
virtual_wall:          dict[int, FrameList]   # type 21
visual_obstacle_zone:  dict[int, FrameList]   # type 26
```

`_area_polygons` read `map.area` and nothing else. The keep-outs sit beside it,
**already in map-local x/y** — the same frame the path planner uses. No
coordinate conversion, no upstream change, no new transport call.

`_keep_out_polygons()` + `_keep_out_violations()` now read all five keep-out
fields, `_validate_custom_path` tests exclusion alongside inclusion, and
`export_map` exposes `keep_out_polygons` so the card can draw them.

⚠️ **A detour worth recording as a dead end.** The first attempt went via
`get_geojson`, which does carry the obstacles — with the exact hash the mower
reported, named "Obstacle 1", "Obstacle in Backyard Right", ~4.0 x 4.1 m. But
geojson is **WGS84 lat/lon** while planning is map-local metres, and deriving
the transform from the four areas present in both frames FAILED: per-axis scale
gave a 3.46 m mean residual and a full affine gave **6.10 m**. A richer model
fitting worse means the point CORRESPONDENCE is wrong — same polygon, same point
count, different start index or winding. Any conclusion drawn from those
transforms is void, including two "does not cross the obstacle" results that the
mower had already disproved by driving into it. The sibling-dict route avoids
the question entirely.

## The defect

**`_validate_custom_path` does not fail to check no-go zones. It has no no-go
geometry to check.**

* `export_map` returns `area_polygons`, `areas`, `coordinate_system`, `raw`.
* `raw` contains only `area`, `area_name`, `svg`.
* The zone hash the mower reported at contact, `1529607395159402290`, appears
  **nowhere in the map payload** — it is not one of the four area hashes.

The mower knows. It reported that hash live and flipped `pos_type_label`
`AREA_INSIDE` → `OBS_ON` the moment it entered. Nothing surfaces that geometry
to the integration, so every containment check in this project — the backend's
per-point check, the card's `_preflight`, and the bearing scan used to pick this
run's heading — validates against **mowing-area polygons only**.

⚠️ **This was survivable while legs were ~0.8 m and is not now.** Route B makes
10–15 m legs routine, and a long leg is far more likely to cross a no-go zone.
The reach work and this gap compound.

## ⚠️ What this run does NOT establish

The operator let the mower push the trampoline rather than stopping it, so
**sub-leg 2's numbers are contaminated and must not be read as accuracy.** The
per-pulse trace locates the contact exactly:

| leg 2 pulse | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| travelled (m) | 0.4183 | 0.3785 | 0.3869 | 0.3507 | 0.3737 | **0.2175** | **0.1207** |

Pulses 1-5 are indistinguishable from sub-leg 1's healthy 0.34-0.43 m. Contact
begins at **pulse 6**: travel halves, then halves again. The mower tracked
cleanly for ~1.91 m and pushed for the last ~0.34 m.

So **`distance_to_target: 1.3724` is where it stopped while pushing, NOT a
landing**, and **Route B end-to-end across three sub-legs is UNTESTED.** What
this run validates is one clean sub-leg plus one free junction.

🔑 **The travel collapse is a usable obstacle signature, and nothing acted on
it.** `min_progress_distance` is 0.0025 m -- three orders of magnitude below a
0.1207 m pulse -- so the no-progress abort never came close to firing, and
`max_no_progress_pulses: 3` would have needed three more near-zero pulses. What
stopped the run was the `zone_hash` change, in two pulses. As configured, the
progress detector would have let it push indefinitely.

## What worked

* **The collinear junction cost ZERO turn commands on hardware.** Sub-leg 2
  reported `turn_commands_sent: 0`. That is Route B's whole premise, previously
  argued from the early return in `_vio_turn_to_heading` and now measured.
* **The split geometry is exact.** 3 × 3.599987 m, headings identical to nine
  decimal places (37.999869047).
* **Landing accuracy holds at 3.6 m — on ONE leg.** Sub-leg 1 landed 0.092 m,
  better than the 4.0 m single-segment record of 0.1023 m. n = 1.
* **The safety net fired.** `quality_degradation` reported
  `pos_type_not_valid_manual_motion_area` and `zone_hash_changed`, and the chain
  stopped rather than continuing to sub-leg 3. It fires on ENTRY, not before.

## What to fix, in order

1. ✅ **DONE — exclusion check shipped.** `_keep_out_polygons` /
   `_keep_out_violations`, wired into `_validate_custom_path` and `export_map`.
   Pinned by `tests/components/mammotion/test_keep_out_zones.py`.
   ⚠️ **NOT YET VERIFIED ON THE HOST** — that needs a deploy, and the decisive
   test is whether `export_map.keep_out_polygons` returns hash
   1529607395159402290 and whether the recorded 10.8 m click is refused.
2. ⚠️ **The check is PER-POINT, like the inclusion check beside it.** A leg
   clipping a keep-out corner with neither endpoint inside is NOT caught. Route
   B's split narrows the gap (a point every ~3.85 m) but a 4 m keep-out can
   still fit between two split points. Segment-level containment is the real
   fix; `test_a_leg_that_clips_a_corner_is_not_caught` pins the limitation so it
   cannot be mistaken for coverage.
3. **Card work still owed.** The card should draw keep-outs from
   `keep_out_polygons` and refuse the click before the operator ever sends it.
4. **Consider a travel-collapse obstacle detector.** The signature here was
   unmistakable and arrived two pulses before the zone gate. It is a separate
   question from `min_progress_distance`, which is set for final-approach
   pulses and cannot serve both purposes.
5. **Until either lands, keep Route B legs short and eyes on the mower.** The
   0.8 m confirmation run is unaffected; the risk is proportional to leg length.

## Method note

The bearing for this run was chosen by scanning 360 headings against the area
polygon and taking the longest fully-contained chord. That scan was correct and
incomplete in the same way the product is: it proved the line stayed inside the
mowing area and said nothing about what was inside that area. **A containment
result is only as good as the geometry it was checked against**, and nothing in
the response distinguishes "checked against everything" from "checked against
the only thing we had".
