#!/usr/bin/env python3
"""Run and classify the isolated, position-specific report-period matrix."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PERIODS_MS = (1000, 500, 250, 100)
REPEATS = 3
MIN_POSITION_PAYLOADS = 100


def _service_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the response body from Home Assistant's service wrapper."""
    response = payload.get("service_response", payload)
    if not isinstance(response, dict):
        raise TypeError("Home Assistant returned no service_response object")
    return response


def classify_matrix(runs: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: C901
    """Classify periods only from three complete isolated position-payload cells."""
    by_period: dict[int, list[dict[str, Any]]] = {period: [] for period in PERIODS_MS}
    for run in runs:
        period = int(run.get("period_ms", 0))
        if period in by_period:
            by_period[period].append(run)

    periods: dict[str, Any] = {}
    for period, cells in by_period.items():
        blockers: list[str] = []
        if len(cells) != REPEATS:
            blockers.append("three_repeats_required")
        for index, cell in enumerate(cells, start=1):
            payloads = cell.get("position_payloads", {})
            if cell.get("isolated") is not True:
                blockers.append(f"repeat_{index}_not_isolated")
            if cell.get("period_ms") != cell.get("no_change_period_ms"):
                blockers.append(f"repeat_{index}_periods_differ")
            if payloads.get("observed", 0) < MIN_POSITION_PAYLOADS:
                blockers.append(f"repeat_{index}_insufficient_payloads")
            if payloads.get("dropped_samples", 0) or payloads.get("sequence_gaps", 0):
                blockers.append(f"repeat_{index}_evidence_gap")
            if payloads.get("p95_interval_ms") is None:
                blockers.append(f"repeat_{index}_missing_p95")
        meets = not blockers and all(
            cell["position_payloads"]["p95_interval_ms"] <= period * 1.5
            for cell in cells
        )
        periods[str(period)] = {
            "honoured": meets if not blockers else None,
            "blockers": blockers,
            "p95_intervals_ms": [
                cell.get("position_payloads", {}).get("p95_interval_ms")
                for cell in cells
            ],
        }
    return {
        "periods": periods,
        "complete": all(not p["blockers"] for p in periods.values()),
    }


def call_sequence_probe(
    *,
    ha_url: str,
    token: str,
    entity_id: str,
    periods_ms: list[int],
    observation_s: float,
    readiness_timeout_s: float = 3.5,
) -> dict[str, Any]:
    """Run one serialized, position-ready sequence through Home Assistant."""
    body = json.dumps(
        {
            "entity_id": entity_id,
            "periods_ms": periods_ms,
            "observation_seconds": observation_s,
            "readiness_timeout_seconds": readiness_timeout_s,
        }
    ).encode()
    request = urllib.request.Request(
        f"{ha_url.rstrip('/')}/api/services/mammotion/"
        "report_stream_sequence_probe?return_response",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    timeout = len(periods_ms) * (observation_s + readiness_timeout_s + 5.0) + 60.0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return _service_result(json.load(response))


def main() -> int:
    """Plan or execute the isolated cadence matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity_id")
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call Home Assistant; without this flag only print the randomized plan.",
    )
    args = parser.parse_args()
    schedule = [period for period in PERIODS_MS for _ in range(REPEATS)]
    random.Random(args.seed).shuffle(schedule)
    if not args.execute:
        print(json.dumps({"seed": args.seed, "schedule_ms": schedule}, indent=2))
        return 0

    ha_url = os.environ.get("HA_URL")
    token = os.environ.get("HA_TOKEN")
    if not ha_url or not token:
        parser.error("--execute requires HA_URL and HA_TOKEN environment variables")
    # One lease across every cell is the point of the redesign, but it also means
    # ONE long HTTP call with no partial results: a client-side drop at cell 11
    # loses all eleven completed cells, and Home Assistant keeps executing (and
    # keeps holding the lease) until the service returns. Say so before starting
    # rather than after losing half an hour.
    estimated_s = len(schedule) * (args.duration_seconds + 3.5 + 5.0) + 60.0
    try:
        print(
            f"serialized sequence: {len(schedule)} cells under one lease, "
            f"~{estimated_s / 60.0:.0f} min in a single request. "
            "There are no partial results: if this connection drops, every "
            "completed cell is lost and the lease stays held until the service "
            "returns on the host.",
            file=sys.stderr,
        )
        sequence = call_sequence_probe(
            ha_url=ha_url,
            token=token,
            entity_id=args.entity_id,
            periods_ms=schedule,
            observation_s=args.duration_seconds,
        )
    except urllib.error.URLError as err:
        print(f"probe failed: {err}", file=sys.stderr)
        return 2
    runs = list(sequence.get("cells") or [])
    artifact = {
        "seed": args.seed,
        "schedule_ms": schedule,
        "sequence": {key: value for key, value in sequence.items() if key != "cells"},
        "runs": runs,
        "classification": classify_matrix(runs),
    }
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    return 0 if artifact["classification"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
