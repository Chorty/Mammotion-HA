# Offline Phase 1 capture analyzer

`scripts/analyze_phase1_capture.py` evaluates the straight and shallow-arc
captures required by the continuous-motion feasibility plan. It reads local
JSON files only. It has no Home Assistant client, network access, coordinator,
BLE, service-call, motion-owner, gate, or dispatch path.

A `go` verdict means only that both captures satisfy the written Phase 1
telemetry criteria. It does not authorize Phase 2, arm the experimental gate,
or authorize any mower motion.

## Inputs

The straight and arc files may contain either the bare service response or the
REST wrapper with a top-level `service_response` object. The analyzer
recomputes its findings from `in_window_telemetry.samples`; it does not trust
the service response's summary fields as a verdict.

The third file supplies containment evidence that is intentionally outside the
capture response:

```json
{
  "straight": {
    "prevalidated": true,
    "area_margin_m": 1.2,
    "keepout_margin_m": 1.5,
    "frozen_start": {"x": 0.0, "y": 0.0},
    "frozen_endpoint": {"x": 1.0, "y": 0.0},
    "polygon": [
      {"x": -0.2, "y": -0.2},
      {"x": 1.2, "y": -0.2},
      {"x": 1.2, "y": 0.5},
      {"x": -0.2, "y": 0.5}
    ]
  },
  "shallow_arc": {
    "prevalidated": true,
    "area_margin_m": 1.2,
    "keepout_margin_m": 1.5,
    "frozen_start": {"x": 0.0, "y": 0.0},
    "frozen_endpoint": {"x": 1.0, "y": 0.0},
    "polygon": [
      {"x": -0.2, "y": -0.2},
      {"x": 1.2, "y": -0.2},
      {"x": 1.2, "y": 0.5},
      {"x": -0.2, "y": 0.5}
    ]
  }
}
```

Replace the example coordinates with each freshly scanned and frozen route.
The analyzer verifies the declared margins, start drift, polygon shape, and
observed containment. It cannot prove that the polygon actually came from a
fresh scan, so `prevalidated: true` is operator-supplied evidence and must not
be guessed or copied from an older route.

## Run it

```bash
.venv/bin/python scripts/analyze_phase1_capture.py \
  --straight path/to/straight-response.json \
  --arc path/to/shallow-arc-response.json \
  --corridors path/to/phase1-corridors.json \
  --output path/to/phase1-analysis.json
```

The command prints the same JSON it writes. Exit status `0` means `go`; status
`1` means `no_go`. The saved result includes each input path and SHA-256 digest
so the decision can be tied to the exact source bytes.

## Fail-closed checks

Both captures must have the exact 4,000 ms profile, 200 ms identical-command
refresh, 100 ms cache-only sampling, zero extra in-window BLE report requests,
confirmed command, live report queue, ordered refresh-write completions,
explicit stop, and `completed` reason. Samples must be finite and well formed.

The analyzer independently requires:

- at least three fresh x/y arrivals no later than 4,000 ms;
- no boundary-inclusive arrival gap above 2,000 ms;
- at least three moving steps whose `bearing + toward` mirror error is at most
  10 degrees;
- a prevalidated frozen route, start drift at most 0.30 m, area margin at least
  1.2 m, and keep-out margin at least 1.5 m;
- every observed position inside or on the supplied corridor polygon; and
- a pre-stop `toward` change for the shallow arc.

Samples after 4,000 ms are excluded from arrival, compass, and `toward` credit.
They remain subject to the containment check, making unexpected late travel
visible rather than silently discarding it. Missing or malformed evidence is a
named failure, never an implicit pass.
