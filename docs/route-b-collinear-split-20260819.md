# Route B — one click, four collinear sub-legs, 50 ft of reach

**Built 2026-08-19 on `0.6.4-beta61`. NOT DEPLOYED. NO MOTION HAS RUN ON IT.**

## Why this exists

The ask from 2026-08-17 was: **move the mower 50 ft (15.24 m) from a single
waypoint click.**

Route A shipped instead — the per-segment cap raised to 6.10 m and the mid-drive
re-aim trigger rewritten from an angle test to a projected-miss test. It was
reviewed three times, passed a Gate 5, and deployed as `0.6.4-beta60`.

**It does nothing.** Replayed across **32 decision points on three hardware
runs** (legs 0.70–2.31 m, one with real mid-drive corrections), the old and new
triggers made **identical decisions every time**
(`docs/longleg-replay-answers-the-open-question-20260819.md`). Correct, tested,
accepted — and inert on every geometry ever measured.

Route B reaches the distance using only pieces that have been measured.

## The mechanism

A leg longer than the card's 3.85 m target becomes `n = ceil(d / target)`
sub-legs of `d / n` metres, by linear interpolation between the operator's two
clicks. Every inserted point therefore lies **exactly** on the original line.

A collinear junction is a 0° turn, and **a 0° turn costs zero turn commands and
zero translation** — `_vio_turn_to_heading` returns `target_heading_reached`
and returns before dispatching anything when
`|error| <= heading_tolerance_degrees` (`services.py`, the early return above
the `turn_feasibility` block). Verified in code, and exercised directly by
`test_a_collinear_junction_costs_zero_turn_commands_and_zero_translation`, which
calls the primitive and asserts the mower's command channel was never touched.

Each sub-leg then gets its own fresh pulse budget. That is what makes 15 m
reachable inside a control law validated at ~4 m: not a longer leg, four short
ones.

A true 50 ft click splits into **4 sub-legs of 3.8100 m**, every junction
**0.000000°** (`docs/evidence-collinear-split-geometry-20260819.json`).

## Why 3.85 and not 3.81

`n = ceil(d / target)` is a step function and `15.24 / 3.81 = 4.000` exactly. A
centimetre of drift between the card's check and the backend's snapshot flips it
to `n = 5`, and 5 sub-legs exceed the 4-segment budget, so the run is refused.
3.85 buys 0.16 m of headroom before the count rounds up, and 15.24 m still
splits into 4.

## What this does NOT prove

- **That 15.40 m is drivable.** It has never been driven. The longest straight
  leg ever executed is **4.0 m, n = 1** (landing 0.1023 m against 0.15 m,
  stopping on tolerance). 3.81 m is 95% of that single datapoint. 3.85 is not
  proven better than 4.0, only shorter. If the first hardware run disappoints,
  **3.0 m (~39 ft)** is the conservative fallback — still ~4× today's usable
  reach.
- **That splitting improves accuracy. It does not.** Cross-track error has
  **unity gain** across a collinear junction, not contraction: each sub-leg
  re-aims from the mower's live position to the next point on the original line,
  so `miss_{k+1} ≈ miss_k` plus noise. A 0.10 m junction miss opens the next
  3.81 m leg at **1.50°** — below the 15° smallest-correctable-aim floor and
  below the 18° turn tolerance — so **nothing corrects it**. The fresh budget
  prevents *ceiling exhaustion*; it does not reduce lateral error.
- **Anything about a 50 ft click fitting the yard.** 1,165 recorded positions
  span **12.74 × 9.73 m**, so a 15.24 m straight line only fits on the diagonal
  and may fail containment. **20–40 ft is the realistic everyday value.**

**The failure this names:** an intermediate landing near the 0.15 m tolerance
puts the next sub-leg on the tolerance edge and may end on
`target_requires_reverse_recovery`. That is bounded and self-announcing, and it
is the thing to watch **per sub-leg index** on the first hardware run.

## What changed

### Backend — `custom_components/mammotion/services.py`

- `_split_long_legs(points, *, target_length_m)` beside `_path_heading_degrees`.
  Returns `applied / target_length_m / requested_points / points /
  requested_leg_count / sub_leg_count / legs`. **No rounding** of the
  interpolated coordinates — rounding to 3 dp would inject ~1.4 mm of
  non-collinearity, which is the difference between a junction that dispatches
  nothing and one that spends a turn command and its translation.
- `_SPLIT_LEG_TARGET_LENGTH_M = 3.85`, beside `_MAX_SEGMENT_LENGTH_M`,
  documenting that this is the card's number and that 15.40 m has never been
  driven.
- `split_leg_target_length_m` on the multi-segment schema:
  `vol.Any(None, Range(min=0.5, max=6.10))`. ⚠️ **The bound is a LITERAL** —
  `_MAX_SEGMENT_LENGTH_M` is defined ~10,000 lines below the schema, so
  referencing it there is a `NameError` at import.
  `test_the_schema_literal_still_mirrors_the_segment_length_cap` is the standing
  check. **No default**, so a caller that omits it is unchanged by its
  existence — `gate4-repass` §4 forbids changing schema *defaults*.
- Wired into `_raw_pymammotion_execute_multi_segment` **before the preview**, so
  `_validate_custom_path`'s per-point containment check judges the inserted
  points too. A concave area can contain both clicks while the line between them
  leaves it; `test_an_inserted_point_outside_the_area_fails_containment` proves
  the ordering by showing the unsplit path validating and the split one failing.
- New gate `split_exceeds_real_segment_budget`, **firing on dry runs too** — a
  dry run that passes while Real Go refuses is the trap. Its early return sits
  above `invalid_point_count` so a handful of long clicks names the real reason.
- The response echoes `split`, `requested_points` and `split_leg_target_length_m`
  beside `points`. Echo it or it is unprovable — the beta44 discipline.
- `services.yaml`: the new field, plus that entry's stale description ("1-3
  moves (2-4 points)" → up to 4 moves, 2-8 points) and its stale
  `max_real_segments` selector max (3 → 4).

### Card — `custom_components/mammotion/www/mammotion-custom-path-card.js`

- 🔑 **`_longestSegmentMetres()` now measures the SPLIT points.** As written it
  measured destination-to-destination, so a 15.24 m click would trip
  `segment_too_long` in `_preflight()` and the split would never get a chance to
  run. **The single most important card edit.**
- `SPLIT_LEG_TARGET_METRES = 3.85` and `_plannedSplit()` / `_plannedLegCount()`,
  mirroring the backend. **Explicitly not a `LUBA_ACCEPTANCE_PROFILE` key** —
  adding it would un-accept the hardware-accepted profile and owe another Gate
  5, the exact cost Route B exists to avoid.
- `_preflight()` gets a **distinct** `split_exceeds_real_segment_budget`, guarded
  on `applied` so five short clicks still report only `real_segment_limit_4`.
  Both directions are pinned by tests.
- `_motionPayload()` routes on **driven legs**, not clicked points, and sends
  the key on the multi-segment branch only. `points` stays the operator's
  clicks.
- `_nightPreflight()` filters the new code. Night is otherwise untouched and
  never sends the parameter.
- Operator-visible: counter reads "N legs (M destinations, auto-split)",
  readiness headline names legs *and* destinations, the map draws hollow
  non-draggable dots at inserted points, and a legend line explains them.

### Two pre-existing bugs fixed in the same pass

1. **Dry-run `max_real_segments` violated its own schema.** The card sent
   `Math.min(points.length - 1, MAX_WAYPOINTS)` — up to **7** — into
   `vol.Range(min=0, max=4)`, so a 5+-waypoint dry run was rejected by schema
   validation before reaching the handler, and a frontend test **pinned the
   broken value**. Clamped; behaviour-neutral because `max_real_segments` is
   only read behind `if not dry_run`.
2. **Map segment colouring mis-mapped after a split.** It indexed
   `runResult.segments[i]` against `[start, ...waypoints]`, so segment 1's
   verdict would have coloured the whole leg and segments 2–4 vanished. Now
   drawn and coloured against the split path.

## One assumption verified rather than assumed

Before Route B a two-point path **always** went to the vector-segment service; a
long one now goes to multi-segment. The two service schemas' defaults are **not**
identical — they differ on `max_linear_commands` (2 vs 1), `max_turn_commands`
(4 vs 3) and `max_real_segments` (multi only). Routing is safe only because the
card sends all three explicitly. `test_routing_a_long_two_point_leg_to_multi_segment_changes_no_dispatch`
pins the divergence set, so a future key that diverges without the card sending
it fails a test rather than silently changing what a long click dispatches.

## Verification (all by exit code, never `| tail`)

| gate | result |
| --- | --- |
| pytest | **716 passed** (baseline 700; +16 in `test_collinear_leg_split.py`) |
| frontend | **57 passed** (baseline 50) |
| ruff check / ruff format | exit 0 |
| mypy (`--follow-imports=skip`) | exit 0, 28 files |
| pre-commit (ten hooks) | exit 0 |
| `scripts/check_accepted_profile.py` | **ACCEPTED**, 2026-08-18 |

Off-mower geometry evidence, banked:
`docs/evidence-collinear-split-geometry-20260819.json`.

## Still owed before this is real

1. **Host dry run, gate DISARMED**: `split.applied`, `sub_leg_count === 4`,
   `requested_points.length === 2`, `points.length === 5`, three
   `junction_turn_feasibility` entries all at ~0°, `would_send === false`, no
   session created. Bank it.
2. **Refusal dry run** (3 × 5 m) with the banner text recorded verbatim.
3. **Only then** a supervised daylight run, with the run JSON downloaded
   immediately — the card still keeps only one full result.
