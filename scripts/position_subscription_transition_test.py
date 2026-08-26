#!/usr/bin/env python3
"""Plan or run stationary report-subscription transition reliability checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from position_cadence_matrix import call_sequence_probe  # noqa: E402


def main() -> int:
    """Run no-motion position-readiness transitions under one report lease."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity_id")
    parser.add_argument("--transitions", type=int, default=30)
    parser.add_argument("--period-ms", type=int, default=1000)
    parser.add_argument("--readiness-timeout-seconds", type=float, default=3.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call Home Assistant; otherwise print the stationary plan only.",
    )
    args = parser.parse_args()
    if not 1 <= args.transitions <= 64:
        parser.error("--transitions must be between 1 and 64")
    if not 100 <= args.period_ms <= 10000:
        parser.error("--period-ms must be between 100 and 10000")
    periods = [args.period_ms] * args.transitions
    plan = {
        "motion_commanded": False,
        "periods_ms": periods,
        "observation_seconds": 0.0,
        "readiness_timeout_seconds": args.readiness_timeout_seconds,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    ha_url = os.environ.get("HA_URL")
    token = os.environ.get("HA_TOKEN")
    if not ha_url or not token:
        parser.error("--execute requires HA_URL and HA_TOKEN environment variables")
    try:
        result = call_sequence_probe(
            ha_url=ha_url,
            token=token,
            entity_id=args.entity_id,
            periods_ms=periods,
            observation_s=0.0,
            readiness_timeout_s=args.readiness_timeout_seconds,
        )
    except urllib.error.URLError as err:
        print(f"transition test failed: {err}", file=sys.stderr)
        return 2
    artifact = {"plan": plan, "result": result}
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete": result.get("complete"),
                "reason": result.get("reason"),
                "cells": len(result.get("cells") or []),
                "failed_cells": result.get("failed_cells"),
            },
            indent=2,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
