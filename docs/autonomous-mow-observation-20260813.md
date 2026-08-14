# Read-only autonomous-mow observation — 2026-08-13

Raw JSONL: `evidence-autonomous-mow-observation-20260813T220700Z.jsonl`.

The vendor mower was already autonomously mowing Backyard Hill. This session
only read Home Assistant state; it did not arm experimental motion, call a
movement service, or alter the route. The manual-motion gate remained disabled
and the active-mowing/active-route blockers remained present.

## Measured facts

- 291 samples span 179.895 seconds at 0.613 s median cadence (0.595 s minimum,
  0.749 s maximum).
- The window contains stable rows in both directions, three complete pivots,
  and the start of a fourth. `toward` took 109 distinct values.
- During vendor pivots, `toward` changed progressively rather than arriving as
  one final step. Examples from the first captured pivot are −26.2402,
  −40.2679, −52.7041, and −68.9202 on successive samples. The next two pivots
  show the same pattern in both turn directions.
- The repository's per-step summarizer found 40 usable moving steps with
  `travel bearing + toward = 90.57°`, circular standard deviation 2.02°. It
  skipped 46 steps whose `toward` value had not changed; this is a per-item
  result, not the misleading first-to-last net bearing.
- Across all 291 sequentially fetched samples, the absolute residual between
  VIO heading and `(90° − toward)` had median 1° and 95th percentile 3°.
  The 24° maximum occurred during a fast pivot and is not a simultaneous-sensor
  measurement: the capture fetches the runtime and VIO entities sequentially.

## Conclusions and limits

Measured: `toward` can stream progressive rotation during continuous
vendor-controlled motion. Therefore item 16's one post-pulse step is specific
to the bounded manual-pulse/report cadence observed there, not proof that the
underlying field is intrinsically blind until all rotation ends.

Inference: tighter night control would need either continuous-command telemetry
whose update behavior matches this autonomous case, or conservative pulse-stop-
settle control. The observation does not prove which transport/report mechanism
causes the difference.

This capture contains no identified reverse manoeuvre and does not expose
`RapidState.fuse_status`, so it does not settle plan item 17 or authorize item
18. No claim about reverse body heading versus course-over-ground follows from
this dataset.
