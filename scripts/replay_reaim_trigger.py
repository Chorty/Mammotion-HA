#!/usr/bin/env python3
"""Replay the mid-drive re-aim trigger against recorded runs.

WHY THIS EXISTS. The 2026-08-17 reach work changed the trigger from an angle
test to a projected-miss test and left one open question: **is the accepted
`vio_max_realignments: 3` still enough?** The new trigger fires earlier and more
often, so three corrections could be spent in the far field with none left for
the approach. That is an empirical question, and every past run already recorded
the per-pulse geometry needed to answer it without touching the mower.

WHAT IT DOES. For each linear pulse in a recorded segment it reconstructs the
decision the executor faced -- position, range to target, direction of travel --
and asks both the OLD and the NEW trigger what they would have done.

🔑 TWO RULES MAKE THIS TRUSTWORTHY, AND WITHOUT THEM IT IS WORTHLESS:

1. **It imports the SHIPPED decision function.** `_mid_drive_realign_decision`
   and `_realign_cannot_improve_the_landing` come from `services.py`. A
   reimplementation would only prove the reimplementation agrees with itself.

2. **It self-validates on the old trigger.** Replaying the OLD rule must
   reproduce the `realignments` the run actually recorded. If it does not, the
   reconstruction is wrong and the new-trigger numbers mean nothing. `--validate`
   reports this per segment and it is the first thing to read.

⚠️ WHAT IT CANNOT TELL YOU. This is a counterfactual on a FIXED trajectory. The
moment the new trigger fires a correction the real run did not, the real
trajectory would have diverged, so pulse N+1's geometry is no longer what was
recorded. Therefore:

  * "would have fired at K of N decision points" is SOUND -- it is evaluated on
    geometry that was actually measured;
  * "the landing would have been X" is NOT, and this script never claims it.

The sound quantity is the CORRECTION RATE, which is exactly what the open
question needs: a rate above 3 per segment is evidence the budget binds.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from custom_components.mammotion.services import (  # noqa: E402
    _MIN_CORRECTABLE_AIM_ERROR_DEGREES,
    _heading_error_degrees,
    _mid_drive_realign_decision,
    _path_heading_degrees,
    _realign_cannot_improve_the_landing,
)


def _old_trigger_fires(
    *,
    aim_error_degrees: float,
    distance_to_target_m: float,
    waypoint_tolerance: float,
    metres_per_pulse: float,
    realign_threshold_degrees: float,
    heading_tolerance_degrees: float,
) -> bool:
    """Return whether the pre-2026-08-17 rule would have fired.

        needs_correction = abs(aim) > vio_realign_threshold_degrees
                           and abs(aim) > heading_tolerance_degrees
        ... and the correction ran only when NOT already_lands_inside.

    Kept as its own function so the replay can be validated against what the
    runs actually did before any new-trigger number is believed.
    """
    aim = abs(float(aim_error_degrees))
    if not (aim > realign_threshold_degrees and aim > heading_tolerance_degrees):
        return False
    return not _realign_cannot_improve_the_landing(
        distance_to_target_m=distance_to_target_m,
        aim_error_degrees=aim_error_degrees,
        waypoint_tolerance=waypoint_tolerance,
        metres_per_pulse=metres_per_pulse,
    )


def _segment_results(doc: Any) -> list[dict]:
    """Every executor result in an evidence file, across schema generations.

    Multi-segment runs nest under `segments[].result`; single-segment runs are
    the result itself. Both shapes appear in docs/evidence-*.json.
    """
    out: list[dict] = []
    if not isinstance(doc, dict):
        return out
    segs = doc.get("segments")
    if isinstance(segs, list) and segs:
        out.extend(
            s["result"]
            for s in segs
            if isinstance(s, dict) and isinstance(s.get("result"), dict)
        )
    if "progress_diagnostics" in doc or "linear_commands_sent" in doc:
        out.append(doc)
    return out


def _decisions(
    diags: list[dict],
    settled: dict[int, dict],
    *,
    target: tuple[float, float],
    tol: float,
    heading_tol: float,
    threshold: float,
    observed_mpp: float,
) -> list[dict]:
    """Evaluate both triggers at every recorded decision point."""
    tx, ty = target
    out: list[dict] = []
    for d in diags:
        if d.get("action") != "forward":
            continue
        idx = d.get("command_index")
        moved = d.get("movement_vector_heading_degrees")
        pos = settled.get(idx) if isinstance(idx, int) else None
        if moved is None or pos is None:
            continue
        x, y = float(pos["x"]), float(pos["y"])
        dist = math.hypot(tx - x, ty - y)
        if dist <= 0:
            continue
        bearing = _path_heading_degrees({"x": x, "y": y}, {"x": tx, "y": ty})
        # Direction of travel is the MEASURED facing proxy, and it is EXACT:
        # the calibration defines the offset so vision_heading + offset is the
        # heading the mower actually travelled. Verified against the executor's
        # own recorded facing_degrees on beta55 segment 1 (326.051 vs
        # 326.05120..., 316.38 vs 316.38048...).
        aim = _heading_error_degrees(float(moved), float(bearing))
        old = _old_trigger_fires(
            aim_error_degrees=aim,
            distance_to_target_m=dist,
            waypoint_tolerance=tol,
            metres_per_pulse=observed_mpp,
            realign_threshold_degrees=threshold,
            heading_tolerance_degrees=heading_tol,
        )
        new = _mid_drive_realign_decision(
            distance_to_target_m=dist,
            aim_error_degrees=aim,
            waypoint_tolerance=tol,
            metres_per_pulse=observed_mpp,
            realign_threshold_degrees=threshold,
        )
        out.append(
            {
                "command_index": idx,
                "distance_to_target_m": round(dist, 4),
                "aim_error_degrees": round(aim, 3),
                "projected_landing_m": round(new["projected_landing_m"], 4),
                "old_fires": old,
                "new_fires": bool(new["needs_correction"]),
                "correctable_floor": new["correctable_floor_degrees"],
            }
        )
    return out


def _settled_positions(result: dict) -> dict[int, dict]:
    """Map pulse index -> absolute settled position from `samples`."""
    settled: dict[int, dict] = {}
    for s in result.get("samples") or []:
        label = str(s.get("label") or "")
        if not (label.startswith("linear_") and label.endswith("_position_settled")):
            continue
        try:
            n = int(label.split("_")[1])
        except IndexError, ValueError:
            continue
        pos = (s.get("telemetry") or {}).get("position") or {}
        if pos.get("x") is not None and pos.get("y") is not None:
            settled[n] = pos
    return settled


def _print_table(rows: list[dict]) -> None:
    """Print the per-segment replay comparison."""
    print(f"\nfloor in use: {_MIN_CORRECTABLE_AIM_ERROR_DEGREES} deg\n")
    hdr = (
        f"{'file':52} {'seg':>3} {'len':>6} {'pulses':>6} "
        f"{'rec':>4} {'old':>4} {'new':>4} {'>3?':>4}  stop_reason"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (-r["new_trigger_would_fire"], r["file"])):
        print(
            f"{r['file'][:52]:52} {r['segment_index']:>3} "
            f"{r['segment_length_m']:>6.2f} {r['decision_points']:>6} "
            f"{r['recorded_decisions']:>4} {r['old_trigger_would_fire']:>4} "
            f"{r['new_trigger_would_fire']:>4} "
            f"{'YES' if r['new_exceeds_budget_3'] else '':>4}  {r['stop_reason']}"
        )
    over = [r for r in rows if r["new_exceeds_budget_3"]]
    print(f"\nsegments where the NEW trigger would exceed a budget of 3: {len(over)}")


def replay_segment(result: dict) -> dict | None:
    """Replay one segment. Returns None when it is not replayable."""
    if result.get("turn_mode") not in (None, "vio"):
        return None
    target = result.get("target")
    start = result.get("true_start") or result.get("initial_telemetry", {}).get(
        "position"
    )
    diags = result.get("progress_diagnostics")
    if not (isinstance(target, dict) and isinstance(start, dict) and diags):
        return None

    tol = float(result.get("waypoint_tolerance") or 0.15)
    heading_tol = float(result.get("heading_tolerance_degrees") or 18.0)
    # The runs pre-date the parameter being emitted; 15 was the default
    # throughout, and the old rule's effective threshold was the max of the two.
    threshold = float(result.get("vio_realign_threshold_degrees") or 15.0)
    mpp_default = float(result.get("final_approach_metres_per_pulse") or 1.06)

    # Observed per-pulse travel is the honest metres_per_pulse: it is what the
    # executor's own `_effective_metres_per_pulse` converges to once a few
    # pulses have been seen.
    travels = [
        float((d.get("measured_delta") or {}).get("distance") or 0.0)
        for d in diags
        if (d.get("action") == "forward")
    ]
    observed_mpp = (sum(travels) / len(travels)) if travels else mpp_default

    tx, ty = float(target["x"]), float(target["y"])

    # 🔑 ABSOLUTE SETTLED POSITIONS, NOT ACCUMULATED DELTAS.
    #
    # The first version of this harness summed progress_diagnostics'
    # measured_delta dx/dy from true_start. That is WRONG and it failed the
    # self-validation: correction turns also move the mower (~0.044 m each on
    # the beta55 run, up to max_turn_translation_distance = 0.30), and those
    # displacements appear in NO forward-pulse delta. The reconstruction drifted
    # 6.3 deg of bearing by pulse 2, which dropped the aim error from 21.22 to
    # 14.92 -- under the 15 deg floor, so the replay saw no correction where the
    # executor fired three.
    #
    # `samples[]` carries `linear_N_position_settled` with the absolute settled
    # position after pulse N, which includes every displacement whatever caused
    # it. Verified against the executor's own recorded bearings on
    # docs/evidence-real-go-card-beta55-20260815T204747Z.json segment 1:
    #   after pulse 2: (7.5473, -3.279)  -> 304.83 vs recorded 304.831
    #   after pulse 3: (7.861,  -3.5026) -> 291.40 vs recorded 291.405
    settled: dict[int, dict] = {}
    for s in result.get("samples") or []:
        label = str(s.get("label") or "")
        if not (label.startswith("linear_") and label.endswith("_position_settled")):
            continue
        try:
            n = int(label.split("_")[1])
        except IndexError, ValueError:
            continue
        pos = (s.get("telemetry") or {}).get("position") or {}
        if pos.get("x") is not None and pos.get("y") is not None:
            settled[n] = pos
    if not settled:
        # Older evidence schemas predate the settled-position samples. Skipping
        # is the honest move: reconstructing by accumulation is exactly the bug
        # documented above, and a smaller trustworthy corpus beats a larger
        # invented one.
        return {"skipped": "no_settled_position_samples"}

    decisions = _decisions(
        diags,
        settled,
        target=(tx, ty),
        tol=tol,
        heading_tol=heading_tol,
        threshold=threshold,
        observed_mpp=observed_mpp,
    )
    # 🚨 COMPARE LIKE WITH LIKE, OR THE SELF-VALIDATION IS WORTHLESS.
    #
    # An earlier version compared against len(realignments) and reported a clean
    # 4/4 -- which was partly LUCK. That list mixes two different controllers,
    # and it also under-counts:
    #
    #  * the POST-TURN alignment gate writes an entry with `before_linear: true`
    #    and its own `alignment_tolerance_degrees` (10). It is not the mid-drive
    #    re-aim and this harness does not model it.
    #  * a decision that WANTED a correction but had no budget left writes NO
    #    entry at all -- the executor stops on `vio_realign_budget_exhausted`.
    #
    # On beta55 segment 1 those two errors cancelled exactly: 1 post-turn entry
    # offset 1 budget-blocked decision, giving 3 == 3 for the wrong reason.
    realignments = result.get("realignments") or []
    recorded = len(realignments)
    mid_drive_recorded = sum(1 for a in realignments if "after_linear_pulse" in a)
    budget_blocked = (
        1 if result.get("stop_reason") == "vio_realign_budget_exhausted" else 0
    )
    recorded_decisions = mid_drive_recorded + budget_blocked
    old_n = sum(1 for d in decisions if d["old_fires"])
    new_n = sum(1 for d in decisions if d["new_fires"])
    return {
        "stop_reason": result.get("stop_reason"),
        "segment_length_m": round(
            math.hypot(tx - float(start["x"]), ty - float(start["y"])), 4
        ),
        "linear_commands_sent": result.get("linear_commands_sent"),
        "effective_linear_ceiling": result.get("effective_linear_ceiling"),
        "linear_execution_mode": result.get("linear_execution_mode"),
        "waypoint_tolerance": tol,
        "heading_tolerance_degrees": heading_tol,
        "observed_metres_per_pulse": round(observed_mpp, 4),
        "decision_points": len(decisions),
        "recorded_realignments_total": recorded,
        "recorded_mid_drive": mid_drive_recorded,
        "budget_blocked_decision": budget_blocked,
        "recorded_decisions": recorded_decisions,
        "old_trigger_would_fire": old_n,
        "new_trigger_would_fire": new_n,
        # The self-validation. Replayed OLD must match what the run recorded.
        "replay_matches_recorded": old_n == recorded_decisions,
        "new_exceeds_budget_3": new_n > 3,
        "decisions": decisions,
    }


def _collect(files: list[str]) -> tuple[list[dict], list[tuple]]:
    """Replay every segment in every file; return (rows, skipped)."""
    rows: list[dict] = []
    skipped: list[tuple] = []
    for f in files:
        p = pathlib.Path(f)
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"{p.name}: unreadable ({exc})", file=sys.stderr)
            continue
        for i, res in enumerate(_segment_results(doc)):
            rep = replay_segment(res)
            if rep is None:
                continue
            if "skipped" in rep:
                skipped.append((p.name, i, rep["skipped"]))
                continue
            rep["file"] = p.name
            rep["segment_index"] = i
            rows.append(rep)
    return rows, skipped


def main() -> int:
    """Run the replay over the given evidence files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--validate", action="store_true", help="only the self-check")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows, skipped = _collect(args.files)

    if args.json:
        print(json.dumps(rows, indent=1))
        return 0

    matched = [r for r in rows if r["replay_matches_recorded"]]

    print(f"replayable segments: {len(rows)}")
    if skipped:
        print(f"skipped segments: {len(skipped)}")
        for reason, n in collections.Counter(s[2] for s in skipped).items():
            print(f"  {reason}: {n}")
    print(
        f"replay self-validation (old trigger reproduces recorded realignments): "
        f"{len(matched)}/{len(rows)}"
    )
    if args.validate:
        for r in rows:
            if not r["replay_matches_recorded"]:
                print(
                    f"  MISMATCH {r['file']}#{r['segment_index']}: "
                    f"replayed {r['old_trigger_would_fire']} vs recorded "
                    f"{r['recorded_decisions']} "
                    f"(mid-drive {r['recorded_mid_drive']} + budget-blocked "
                    f"{r['budget_blocked_decision']})"
                )
        return 0

    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
