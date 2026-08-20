# Containment cannot see no-go zones — found by driving into one

**2026-08-20, `0.6.4-beta62`, supervised. The mower pushed a trampoline inside a
no-go zone. The operator judged no damage and let it continue.**

## What happened

The first hardware Route B run: one 10.8 m click at bearing 38°, split into 3
collinear sub-legs of 3.599987 m. Sub-leg 1 reached target at **0.092 m**.
Sub-leg 2 drove into a no-go zone containing a trampoline and stopped on
`telemetry_quality_degraded` 1.3724 m from its target.

Evidence: `docs/evidence-routeb-first-hardware-run-20260820.json`.

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

1. **Find the no-go geometry upstream.** The mower reports a zone hash we cannot
   resolve, so the data exists on the device. Check whether pymammotion parses
   an obstacle/exclusion frame we simply do not export — this is a read-only
   investigation and costs nothing.
2. **If it is reachable, add it to `export_map` and to
   `_validate_custom_path`,** as a per-point exclusion test alongside the
   existing inclusion test. The split runs before the preview precisely so
   inserted points get checked; they would get this check for free.
3. **If it is NOT reachable, say so in the card.** A containment check that
   cannot see no-go zones must not present itself as "path validated". The
   readiness banner should state the limit rather than imply coverage it does
   not have.
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
