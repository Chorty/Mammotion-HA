#!/usr/bin/env python3
"""Explain a saved guarded-motion service result without contacting HA.

Use after ``run_motion_with_evidence.py``.  This is intentionally offline: it
cannot arm, command, or inspect a live mower.  It distinguishes a failed VIO
turn from a failed linear phase so operator decisions are based on the recorded
service result rather than the map trace alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _round(value: Any, digits: int = 3) -> float | None:
    return round(float(value), digits) if isinstance(value, int | float) else None


def diagnose(document: dict[str, Any]) -> dict[str, Any]:
    """Classify a saved guarded-motion result by which phase actually failed."""
    result = document.get("result", document)
    segments = result.get("segments") or []
    failed_index = result.get("failed_segment_index")
    segment_entry = next(
        (entry for entry in segments if entry.get("index") == failed_index),
        segments[0] if segments else {},
    )
    segment = segment_entry.get("result") or {}
    phases = segment.get("phases") or []
    turn = next(
        (
            phase.get("result") or {}
            for phase in phases
            if phase.get("name") == "turn_to_target_heading"
        ),
        {},
    )
    linear = next(
        (
            phase.get("result") or {}
            for phase in phases
            if phase.get("name") == "linear_forward_to_target"
        ),
        {},
    )
    vio = segment.get("vio") or {}

    turn_stop = turn.get("stop_reason")
    linear_stop = linear.get("stop_reason")
    linear_count = segment.get("linear_commands_sent", 0)
    turn_count = segment.get("turn_commands_sent", 0)
    if (
        turn_stop == "turn_budget_infeasible"
        or segment.get("stop_reason") == "turn_budget_infeasible"
        or result.get("stop_reason") == "path_turn_infeasible"
    ):
        classification = "vio_turn_refused_infeasible_preflight"
        conclusion = (
            "The feasibility preflight refused the turn before any turn command "
            "was dispatched: the configured turn budget provably cannot reach the "
            "target heading tolerance under the evidence-bounded per-command "
            "progress. Zero turn commands ran and no turn translation occurred."
        )
    elif turn_stop == "max_commands_reached" and not linear_count:
        classification = "vio_turn_budget_exhausted_before_linear_phase"
        conclusion = (
            "The VIO turn phase exhausted its command budget before the target "
            "heading tolerance was reached. No normal forward linear command ran."
        )
    elif segment.get("stop_reason") == "target_requires_reverse_recovery":
        classification = "forward_only_segment_refused_reverse_recovery"
        conclusion = (
            "The mower ended a forward pulse with the waypoint at or behind 90 "
            "degrees, so no forward command could close the remaining distance. "
            "The segment stopped rather than dispatching a U-turn and calling it "
            "a re-alignment. This is the recorded overshoot-and-recovery path "
            "being refused, not a new fault: expect it whenever a pulse "
            "overshoots a short leg."
        )
    elif segment.get("stop_reason") == "vio_realign_budget_exhausted":
        classification = "vio_realign_budget_exhausted_before_target"
        conclusion = (
            "The segment was off the bearing to its waypoint by more than the "
            "re-alignment threshold with no correction budget left, so it "
            "stopped instead of spending the remaining forward budget driving "
            "off-bearing."
        )
    elif (
        linear_stop == "max_linear_commands_reached"
        or segment.get("stop_reason") == "max_linear_commands_reached"
    ):
        classification = "linear_budget_exhausted"
        conclusion = (
            "The turn completed, but the permitted linear-command budget was exhausted."
        )
    else:
        classification = "inspect_recorded_stop_reason"
        conclusion = "Use the recorded phase stop reasons below; this run does not match a known budget-exhaustion pattern."

    commands = turn.get("command_results") or []
    return {
        "outer_stop_reason": result.get("stop_reason"),
        "failed_segment_index": failed_index,
        "segment_stop_reason": segment.get("stop_reason"),
        "classification": classification,
        "conclusion": conclusion,
        "reverse_recovery_guard": segment.get("reverse_recovery_guard"),
        "junction_turn_feasibility": result.get("junction_turn_feasibility"),
        "turn": {
            "stop_reason": turn_stop,
            "commands_sent": turn_count,
            "turn_feasibility": turn.get("turn_feasibility")
            or segment.get("turn_feasibility"),
            "final_heading_error_degrees": _round(
                turn.get("final_heading_error_degrees")
            ),
            "translation_m": _round(turn.get("final_displacement_m")),
            "target_vision_heading": _round(vio.get("target_vision_heading")),
            "final_vision_heading": _round(turn.get("final_vision_heading")),
            "progress_degrees": [
                _round(item.get("progress_degrees")) for item in commands
            ],
        },
        "linear": {
            "stop_reason": linear_stop,
            "commands_sent": linear_count,
            "started": bool(linear_count),
        },
        "safe_next_step": (
            "Keep experimental motion disabled. Diagnose the VIO turn budget/translation with a separately authorized daylight turn characterization; do not retry the full path automatically."
        ),
    }


def main() -> int:
    """Read a result JSON, print its diagnosis, and optionally save it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = diagnose(json.loads(args.result.read_text()))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
