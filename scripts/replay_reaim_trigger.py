#!/usr/bin/env python3
"""Replay the mid-drive re-aim trigger against recorded runs.

WHY THIS EXISTS. The 2026-08-17 reach work changed the trigger from an angle
test to a projected-miss test and left one open question: **is the accepted
`vio_max_realignments: 3` still enough?** The new trigger fires earlier and more
often, so three corrections could be spent in the far field with none left for
the approach. That is an empirical question, and every past run already recorded
the per-pulse geometry needed to answer it without touching the mower.

WHAT IT DOES. For each linear pulse in a recorded segment it reconstructs the
decision the executor faced -- position, range to target, direction of travel,
`metres_per_pulse` -- replays the four gates that stand in front of the
decision (`target_reached`, the linear ceiling, reverse-recovery, and the shared
realignment budget), and asks both the OLD and the NEW trigger what they would
have done.

⚠️ SKIPPING IS THE POINT. A segment whose decision state cannot be rebuilt from
what was recorded is skipped with a named reason and excluded from every number
printed. A smaller trustworthy corpus beats a larger invented one.

🔑 TWO RULES MAKE THIS TRUSTWORTHY, AND WITHOUT THEM IT IS WORTHLESS:

1. **It imports the SHIPPED decision functions.** `_mid_drive_realign_decision`,
   `_realign_cannot_improve_the_landing`, `_requires_reverse_recovery`,
   `_effective_metres_per_pulse` and `_normalised_linear_pulse_distance` all
   come from `services.py`. A reimplementation would only prove the
   reimplementation agrees with itself.

2. **It self-validates on the old trigger.** Replaying the OLD rule must
   reproduce the `realignments` the run actually recorded -- the same pulse
   indices, in order, plus the budget-blocked decision that stops a run without
   writing a record. If it does not, the reconstruction is wrong and the
   new-trigger numbers mean nothing. `--validate` reports this per segment and
   it is the first thing to read.

WHERE IT STANDS (2026-08-17, 129 evidence files):

    replayable segments                                       62
    old-trigger self-validation                            55/62
      ... on the 26 segments the run recorded a decision in 19/26
      ... on builds with no structural law-version conflict 35/36
    geometry cross-check vs the executor's own records       35 points,
                                                     max 0.000497 deg
    metres_per_pulse cross-check vs recorded values           6 points, 0.0 m

⚠️ **55/62 IS THE HONEST CEILING, AND THE RESIDUAL IS NOT RECONSTRUCTION ERROR.**
The geometry cross-check settles that separately and much more strongly: at
every one of the 35 decision points the executor wrote down, the reconstructed
`facing`, `bearing`, `aim_error` and `distance_to_target` reproduce the recorded
values to 0.0005 deg and 0.05 mm -- the residual of `round(x, 3)`. What the
seven mismatching segments disagree about is WHICH CONTROL LAW RAN, and six of
the seven say so structurally in their own records (see `_build_markers`):
three beta36-era suppression records, three beta38-era ones written before
beta42 added the quadrature term, and two 2026-08-02 runs that recorded a
re-aim at `effective_linear_ceiling` -- which the shipped gate at
`services.py:12720` makes impossible, and whose own comment says it was added
BECAUSE of that run. The seventh (`evidence-beta33-reposition-20260809T184618Z`
segment 0) carries no marker because it never suppressed anything: it fired at
0.2009 m and 20.934 deg of aim, which every guard version that has ever shipped
would suppress, so the build that fired had no suppression guard at all -- beta33
predates beta36, where the guard was introduced.

Replaying the shipped law against a fourteen-beta corpus is expected to
disagree with the older builds in it. Hiding those segments would raise the
ratio and lower its value, so they are kept, counted against the harness, and
labelled.

⚠️ WHAT IT CANNOT TELL YOU. This is a counterfactual on a FIXED trajectory. The
moment the new trigger fires a correction the real run did not, the real
trajectory would have diverged, so pulse N+1's geometry is no longer what was
recorded. Therefore:

  * "would have fired at K of N decision points" is SOUND -- it is evaluated on
    geometry that was actually measured;
  * "the landing would have been X" is NOT, and this script never claims it.

The sound quantity is the CORRECTION RATE, which is exactly what the open
question needs: a rate above 3 per segment is evidence the budget binds.

⚠️ ONE ASSUMPTION IS NOT CHECKABLE FROM THE RECORD. The executor's re-aim block
also requires `vio_state == _VIO_STATE_ACTIVE` at that instant
(`services.py:12699-12701`), and no evidence file stamps VIO state per pulse.
The replay assumes VIO stayed active for every pulse of a segment whose final
`vio.offset_source` is `linear_refresh` -- i.e. a linear pulse did refresh the
offset -- and skips the segment otherwise. A VIO dropout mid-segment would show
up as a self-validation mismatch, not as a silent error.
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
    _DEFAULT_METRES_PER_LINEAR_PULSE,
    _MIN_CORRECTABLE_AIM_ERROR_DEGREES,
    _effective_metres_per_pulse,
    _heading_error_degrees,
    _mid_drive_realign_decision,
    _normalised_linear_pulse_distance,
    _path_heading_degrees,
    _realign_cannot_improve_the_landing,
    _requires_reverse_recovery,
)

#: The executor only refreshes the map->VIO offset from a pulse that moved at
#: least this far (`services.py:12675`). Below it the facing at the decision
#: point is `vision_heading + a STALE offset`, which the record does not carry,
#: so such a segment is skipped rather than guessed. Zero of the 183 forward
#: pulses in docs/evidence-*.json fall below it, so this is a guard, not a
#: filter.
_OFFSET_REFRESH_MIN_DISTANCE_M = 0.05
#: Schema default (`services.py:1067`, `:1198`, `:11365`). Absent from every
#: recorded result because the card never sends it -- it is not a key of
#: `LUBA_ACCEPTANCE_PROFILE`, so voluptuous supplied the default on every run.
_DEFAULT_VIO_MAX_REALIGNMENTS = 3
#: Schema default (`services.py:1064`, `:1195`, `:11364`), same reasoning.
_DEFAULT_REALIGN_THRESHOLD_DEGREES = 15.0


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


def _old_trigger_suppressed(
    *,
    aim_error_degrees: float,
    distance_to_target_m: float,
    waypoint_tolerance: float,
    metres_per_pulse: float,
    realign_threshold_degrees: float,
    heading_tolerance_degrees: float,
) -> bool:
    """Return whether the OLD rule would have written a suppression record.

    The executor records a suppressed re-aim when the angle test passed but the
    projection lands inside tolerance (`services.py:12803-12829`). Those records
    are the only trace of a low-angle decision point in the run files, so they
    are a second, independent check on the same reconstruction.
    """
    aim = abs(float(aim_error_degrees))
    if not (aim > realign_threshold_degrees and aim > heading_tolerance_degrees):
        return False
    return _realign_cannot_improve_the_landing(
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


def _settled_positions(result: dict) -> dict[int, dict[str, float]]:
    """Map pulse index -> the ABSOLUTE position the executor decided on.

    🔑 ABSOLUTE POSITIONS, NOT ACCUMULATED DELTAS.

    The first version of this harness summed `progress_diagnostics`'
    `measured_delta` dx/dy from `true_start`. That is WRONG and it failed the
    self-validation. Correction turns and the VIO calibration drive also move
    the mower, and those displacements appear in NO forward-pulse delta -- every
    one of the 190 `progress_diagnostics` entries in the corpus has
    `action: "forward"`, because the vector executor only ever calls the shared
    diagnostic with that action (`services.py:12592`). Measured over the 62
    replayable segments the accumulated position was a mean 0.089 m and a
    maximum 0.250 m away from the run's own `final_telemetry`, which moved the
    reconstructed bearing by up to 81 deg.

    `samples[]` carries the absolute position after every pulse, and the
    executor's own decision point is `after = command_samples[-1]["telemetry"]`
    (`services.py:12584-12588`) -- so the rule is "the LAST `linear_N_*` sample",
    whatever its label.

    ⚠️ Select on the index in the label, NOT on `label.endswith(
    "_position_settled")`. That label only exists on the beta55 settle-reuse
    branch (`services.py:12546`) and covers 4 of the 62 segments; the older
    schema emits `linear_N_sample_K_Ds` (`services.py:12566`). Filtering on it
    self-validates 4/4 by discarding 58 segments, including every beta32
    4-segment run and both Gate 5 re-passes.
    """
    positions: dict[int, dict[str, float]] = {}
    for sample in result.get("samples") or []:
        label = str(sample.get("label") or "")
        if not label.startswith("linear_"):
            continue
        try:
            index = int(label.split("_")[1])
        except IndexError, ValueError:
            continue
        pos = (sample.get("telemetry") or {}).get("position") or {}
        if pos.get("x") is None or pos.get("y") is None:
            continue
        # Last sample for the pulse wins, matching `command_samples[-1]`.
        positions[index] = {"x": float(pos["x"]), "y": float(pos["y"])}
    return positions


def _pulse_records(result: dict) -> tuple[list[dict], str | None]:
    """Pair every forward pulse with the state the executor decided on.

    Returns `(records, skip_reason)`. A skip reason means the segment cannot be
    faithfully reconstructed and must not be counted -- a smaller trustworthy
    corpus beats a larger invented one.
    """
    diags = [
        d
        for d in (result.get("progress_diagnostics") or [])
        if d.get("action") == "forward"
    ]
    if not diags:
        return [], "no_forward_pulses"

    positions = _settled_positions(result)
    # Linear pulses appear in `command_results` in dispatch order tagged with
    # this phase (`services.py:12409`); mid-drive and post-turn correction turns
    # are extended into the same list but carry no `phase` key. A failed pulse
    # (`command_failed`, `services.py:12453`) appends its record and returns
    # before any diagnostic, so there can be one MORE command result than
    # diagnostic -- never fewer.
    linear_commands = [
        c
        for c in (result.get("command_results") or [])
        if c.get("phase") == "linear_forward_to_target"
    ]
    if len(linear_commands) < len(diags):
        return [], "linear_command_results_missing"

    records: list[dict] = []
    for offset, diag in enumerate(diags):
        index = diag.get("command_index")
        if not isinstance(index, int):
            return [], "pulse_index_missing"
        position = positions.get(index)
        if position is None:
            return [], "no_absolute_position_sample"
        moved = diag.get("movement_vector_heading_degrees")
        if moved is None:
            return [], "no_movement_vector_heading"
        measured = (diag.get("measured_delta") or {}).get("distance")
        if measured is None:
            return [], "no_measured_delta"
        command = linear_commands[offset]
        speed = (command.get("selection") or {}).get("linear_speed")
        if speed is None:
            return [], "no_linear_speed"
        records.append(
            {
                "command_index": index,
                "position": position,
                # 🔑 THE FACING IS NOT A PROXY, IT IS AN IDENTITY.
                #
                # `services.py:12679-12691` sets
                # `offset = motion_heading - vision_heading` from the pulse that
                # has just ended, and `:12722` then computes
                # `facing = vision_heading + offset` from the SAME
                # `reading` object -- so `facing` collapses algebraically to
                # that pulse's motion heading, which is exactly
                # `movement_vector_heading_degrees`. Measured against the
                # executor's own recorded `facing_degrees` on every recorded
                # decision in the corpus: max error 0.0005 deg, the residual of
                # `round(x, 3)`.
                "facing_degrees": float(moved),
                "measured_distance_m": float(measured),
                "linear_speed": int(speed),
                "observation": command.get("final_approach_observation"),
            }
        )
    return records, None


def _build_markers(results: list[dict]) -> dict[str, Any]:
    """Structural evidence of WHICH control law a run's build was carrying.

    🚨 THE REPLAY RUNS THE SHIPPED LAW; THE CORPUS SPANS FOURTEEN BETAS. Where a
    recorded run executed an OLDER guard, disagreement is expected and is not a
    reconstruction failure -- so it must be identified from the record itself,
    never inferred from the disagreement (that would be circular, and would let
    the harness excuse any mismatch it produced).

    Two markers are purely structural, i.e. readable without replaying anything:

    * **the shape of a suppression record.** beta36 wrote `min_distance_m` with
      `reason: "bearing_ill_conditioned_near_waypoint"`; beta38 replaced it with
      `perpendicular_miss_m`; beta42 added `projected_landing_m` alongside it
      (`services.py:12820-12825`). Only the last matches the shipped
      `_realign_cannot_improve_the_landing`. The marker is a LOWER BOUND -- a run
      that never suppressed anything records nothing and stays `unknown`.
    * **a recorded mid-drive re-aim at or past `effective_linear_ceiling`.** The
      shipped executor cannot produce one (`services.py:12720`), and the comment
      at `services.py:12703-12712` says that gate was added BECAUSE of the
      2026-08-02 run this marker catches.

    Markers are computed per FILE, because one evidence file is one build.
    """
    guard = "unknown"
    ceiling_gate_absent = False
    reverse_gate_absent = False
    for result in results:
        for entry in result.get("realignments_suppressed") or []:
            if "projected_landing_m" in entry:
                guard = "beta42_projected_landing"
            elif "perpendicular_miss_m" in entry and guard == "unknown":
                guard = "beta38_perpendicular_miss"
            elif "min_distance_m" in entry and guard == "unknown":
                guard = "beta36_min_distance"
        ceiling = result.get("effective_linear_ceiling")
        for entry in result.get("realignments") or []:
            index = entry.get("after_linear_pulse")
            if index is None:
                continue
            if ceiling is not None and int(index) >= int(ceiling):
                ceiling_gate_absent = True
            aim = entry.get("aim_error_degrees")
            if aim is not None and _requires_reverse_recovery(float(aim)):
                reverse_gate_absent = True
    return {
        "suppression_guard_record_schema": guard,
        "ceiling_gate_absent": ceiling_gate_absent,
        "reverse_recovery_gate_absent": reverse_gate_absent,
    }


def _recorded_decisions(result: dict) -> dict[str, Any]:
    """Read back what the executor itself recorded about its mid-drive decisions."""
    realignments = result.get("realignments") or []
    fired = [
        int(a["after_linear_pulse"]) for a in realignments if "after_linear_pulse" in a
    ]
    # The POST-TURN alignment gate (`services.py:12183-12267`) writes an entry
    # with `before_linear: true`. It is a different controller and this harness
    # does not model it -- but it SPENDS THE SAME BUDGET
    # (`realignments_used += 1` at `services.py:12184`), so it has to be
    # subtracted from the budget the linear loop starts with.
    pre_spent = sum(1 for a in realignments if a.get("before_linear"))
    suppressed = [
        int(a["after_linear_pulse"])
        for a in (result.get("realignments_suppressed") or [])
        if "after_linear_pulse" in a
    ]
    guard = result.get("reverse_recovery_guard") or {}
    return {
        "fired": fired,
        "pre_spent_budget": pre_spent,
        "suppressed": suppressed,
        "reverse_guard_pulse": guard.get("after_linear_pulse"),
        # A decision that WANTED a correction but had no budget left writes NO
        # entry at all -- the executor stops on `vio_realign_budget_exhausted`
        # (`services.py:12831-12833`) before appending. Its pulse is therefore
        # the last one the run recorded.
        "budget_blocked": (result.get("stop_reason") == "vio_realign_budget_exhausted"),
    }


def _geometry_residuals(decisions: list[dict], result: dict) -> dict[str, float | int]:
    """Compare reconstructed geometry against what the executor wrote down.

    A far stronger check than any count: the executor recorded `facing`,
    `bearing` and `aim_error` at every decision it actually evaluated, so each
    one is an independent test of the reconstruction rather than of the trigger.
    """
    by_index = {d["command_index"]: d for d in decisions}
    recorded: list[dict] = []
    recorded.extend(
        a for a in (result.get("realignments") or []) if "after_linear_pulse" in a
    )
    recorded.extend(
        a
        for a in (result.get("realignments_suppressed") or [])
        if "after_linear_pulse" in a
    )
    guard = result.get("reverse_recovery_guard")
    if isinstance(guard, dict) and "after_linear_pulse" in guard:
        recorded.append(guard)

    worst_deg = 0.0
    worst_m = 0.0
    worst_mpp = 0.0
    checked = 0
    mpp_checked = 0
    for entry in recorded:
        replayed = by_index.get(int(entry["after_linear_pulse"]))
        if replayed is None:
            continue
        checked += 1
        for key in ("facing_degrees", "bearing_degrees", "aim_error_degrees"):
            if entry.get(key) is None:
                continue
            worst_deg = max(worst_deg, abs(float(entry[key]) - float(replayed[key])))
        if entry.get("distance_to_target_m") is not None:
            worst_m = max(
                worst_m,
                abs(
                    float(entry["distance_to_target_m"])
                    - float(replayed["distance_to_target_m"])
                ),
            )
        # `metres_per_pulse` is the one guard input the harness has to REBUILD
        # rather than read: `_effective_metres_per_pulse` over
        # `_normalised_linear_pulse_distance` of every pulse so far AT THE SAME
        # SPEED, floored at 1.06. Suppression records since beta42 carry the
        # executor's own value, so the rebuild is directly checkable.
        if entry.get("metres_per_pulse") is not None:
            mpp_checked += 1
            worst_mpp = max(
                worst_mpp,
                abs(
                    float(entry["metres_per_pulse"])
                    - float(replayed["metres_per_pulse"])
                ),
            )
    return {
        "geometry_points_checked": checked,
        "geometry_max_error_degrees": round(worst_deg, 6),
        "geometry_max_error_m": round(worst_m, 6),
        "metres_per_pulse_points_checked": mpp_checked,
        "metres_per_pulse_max_error": round(worst_mpp, 6),
    }


def _law_version_conflicts(markers: dict[str, Any]) -> list[str]:
    """Name every structural way this build's control law differs from the shipped one."""
    conflicts: list[str] = []
    schema = markers["suppression_guard_record_schema"]
    if schema in ("beta36_min_distance", "beta38_perpendicular_miss"):
        conflicts.append(schema)
    if markers["ceiling_gate_absent"]:
        conflicts.append("reaim_recorded_at_or_past_linear_ceiling")
    if markers["reverse_recovery_gate_absent"]:
        conflicts.append("reaim_recorded_past_reverse_recovery_threshold")
    return conflicts


def _evaluate_decisions(  # noqa: C901
    records: list[dict],
    *,
    target: tuple[float, float],
    tol: float,
    heading_tol: float,
    threshold: float,
    default_mpp: float,
    ceiling: int,
) -> tuple[list[dict], collections.Counter[str], str | None]:
    """Walk the linear loop pulse by pulse, replaying every gate in order.

    🚨 THREE GATES STAND IN FRONT OF THE TRIGGER, AND OMITTING THEM MANUFACTURES
    DECISIONS THAT NEVER EXISTED. Replaying the trigger against every recorded
    pulse produced 21 phantom fires across this corpus -- more than the 18 real
    ones -- and both of the "replayed 1 vs recorded 0" mismatches that started
    this work were terminal pulses the executor had already returned on.

    Returns `(decisions, gate_counts, skip_reason)`.
    """
    tx, ty = target
    observed_by_speed: dict[int, list[float]] = {}
    decisions: list[dict] = []
    gate_counts: collections.Counter[str] = collections.Counter()
    for record in records:
        index = record["command_index"]
        x, y = record["position"]["x"], record["position"]["y"]
        distance = math.hypot(tx - x, ty - y)

        # The executor feeds the pulse it just measured into the scale factor
        # BEFORE the re-aim block reads it (`services.py:12608-12626` precedes
        # `:12663`), so this pulse's own observation counts.
        observation = record["observation"]
        if isinstance(observation, dict):
            observed_by_speed.setdefault(record["linear_speed"], []).append(
                _normalised_linear_pulse_distance(
                    float(observation["measured_distance"]),
                    int(observation["nonzero_writes"]) - 1,
                )
            )

        # GATE A -- `target_reached` returns at `services.py:12628-12647`, BEFORE
        # the re-aim block. With two path points `completion_status["complete"]`
        # is exactly `distance <= waypoint_tolerance`
        # (`_manual_velocity_next_waypoint`, `services.py:1660`), and all 62
        # replayable segments have exactly two points. 51 of the 62 segments end
        # this way, so this gate alone removes 51 phantom decision points -- and
        # their aim errors are absurd (101-176 deg) precisely because the
        # bearing to a target 7-14 cm away swings wildly inside the disc.
        if distance <= tol:
            gate_counts["target_reached"] += 1
            break

        # GATE B -- the re-aim block requires `command_index <
        # effective_linear_ceiling` (`services.py:12720`). The final permitted
        # pulse never gets a decision; the loop then exits with
        # `max_linear_commands_reached`.
        if index >= ceiling:
            gate_counts["linear_ceiling"] += 1
            break

        if record["measured_distance_m"] < _OFFSET_REFRESH_MIN_DISTANCE_M:
            return decisions, gate_counts, "pulse_below_offset_refresh_threshold"

        facing = record["facing_degrees"]
        bearing = _path_heading_degrees({"x": x, "y": y}, {"x": tx, "y": ty})
        aim = _heading_error_degrees(facing, bearing)
        mpp = _effective_metres_per_pulse(
            observed_by_speed.get(record["linear_speed"], []), default_mpp
        )

        # GATE C -- at 90 deg or more the executor stops the segment
        # (`services.py:12725-12737`) instead of correcting. Omitting it is
        # perverse: `_realign_cannot_improve_the_landing` explicitly refuses to
        # suppress at >= 90 (`services.py:10954`), so an aim the executor treats
        # as a segment-ending STOP would be scored as a correction.
        if _requires_reverse_recovery(aim):
            decisions.append(
                {
                    "command_index": index,
                    "distance_to_target_m": distance,
                    "facing_degrees": facing,
                    "bearing_degrees": bearing,
                    "aim_error_degrees": aim,
                    "metres_per_pulse": round(mpp, 4),
                    "outcome": "reverse_recovery_stop",
                    "old_fires": False,
                    "old_suppressed": False,
                    "new_fires": False,
                    "projected_landing_m": None,
                }
            )
            gate_counts["reverse_recovery"] += 1
            break

        old = _old_trigger_fires(
            aim_error_degrees=aim,
            distance_to_target_m=distance,
            waypoint_tolerance=tol,
            metres_per_pulse=mpp,
            realign_threshold_degrees=threshold,
            heading_tolerance_degrees=heading_tol,
        )
        old_suppressed = _old_trigger_suppressed(
            aim_error_degrees=aim,
            distance_to_target_m=distance,
            waypoint_tolerance=tol,
            metres_per_pulse=mpp,
            realign_threshold_degrees=threshold,
            heading_tolerance_degrees=heading_tol,
        )
        new = _mid_drive_realign_decision(
            distance_to_target_m=distance,
            aim_error_degrees=aim,
            waypoint_tolerance=tol,
            metres_per_pulse=mpp,
            realign_threshold_degrees=threshold,
        )
        decisions.append(
            {
                "command_index": index,
                "distance_to_target_m": distance,
                "facing_degrees": facing,
                "bearing_degrees": bearing,
                "aim_error_degrees": aim,
                "metres_per_pulse": round(mpp, 4),
                "outcome": "evaluated",
                "old_fires": bool(old),
                "old_suppressed": bool(old_suppressed),
                "new_fires": bool(new["needs_correction"]),
                "projected_landing_m": round(new["projected_landing_m"], 4),
            }
        )

    return decisions, gate_counts, None


def replay_segment(result: dict, markers: dict[str, Any] | None = None) -> dict:
    """Replay one segment's linear loop. Always returns a dict.

    A dict carrying only `skipped` means the segment could not be faithfully
    reconstructed and must be excluded from every number this script prints.
    """
    markers = markers or {
        "suppression_guard_record_schema": "unknown",
        "ceiling_gate_absent": False,
        "reverse_recovery_gate_absent": False,
    }
    turn_mode = result.get("turn_mode")
    if turn_mode not in (None, "vio"):
        return {"skipped": f"turn_mode_{turn_mode}"}
    target = result.get("target")
    start = result.get("true_start") or (result.get("initial_telemetry") or {}).get(
        "position"
    )
    if not (isinstance(target, dict) and isinstance(start, dict)):
        return {"skipped": "missing_target_or_start"}
    records, reason = _pulse_records(result)
    if reason is not None:
        return {"skipped": reason}
    vio = result.get("vio") or {}
    if vio.get("offset_degrees") is None:
        # No map->VIO offset means the executor's re-aim block never ran at all
        # (`offset_now is not None`, `services.py:12697`). Nothing to replay.
        return {"skipped": "vio_offset_unavailable"}
    if vio.get("offset_source") != "linear_refresh" and len(records) > 1:
        # With two or more pulses the second one must have refreshed the offset
        # from linear travel unless VIO dropped out. See the module docstring.
        return {"skipped": "vio_offset_never_refreshed_by_a_pulse"}

    tol = float(result.get("waypoint_tolerance") or 0.15)
    heading_tol = float(result.get("heading_tolerance_degrees") or 18.0)
    threshold = float(
        result.get("vio_realign_threshold_degrees")
        or _DEFAULT_REALIGN_THRESHOLD_DEGREES
    )
    max_realignments = int(
        result.get("vio_max_realignments") or _DEFAULT_VIO_MAX_REALIGNMENTS
    )
    default_mpp = float(
        result.get("final_approach_metres_per_pulse")
        or _DEFAULT_METRES_PER_LINEAR_PULSE
    )
    ceiling = result.get("effective_linear_ceiling")
    if ceiling is None:
        return {"skipped": "no_effective_linear_ceiling"}
    ceiling = int(ceiling)

    tx, ty = float(target["x"]), float(target["y"])
    last_index = records[-1]["command_index"]

    decisions, gate_counts, loop_skip = _evaluate_decisions(
        records,
        target=(tx, ty),
        tol=tol,
        heading_tol=heading_tol,
        threshold=threshold,
        default_mpp=default_mpp,
        ceiling=ceiling,
    )
    if loop_skip is not None:
        return {"skipped": loop_skip}

    recorded = _recorded_decisions(result)

    # GATE D -- the budget. `services.py:12831` returns
    # `vio_realign_budget_exhausted` WITHOUT appending a record, and the
    # post-turn gate has already spent part of the same budget.
    budget = max(0, max_realignments - recorded["pre_spent_budget"])
    old_fire_indices = [d["command_index"] for d in decisions if d["old_fires"]]
    replayed_fired = old_fire_indices[:budget]
    replayed_blocked = (
        old_fire_indices[budget] if len(old_fire_indices) > budget else None
    )
    replayed_suppressed = [d["command_index"] for d in decisions if d["old_suppressed"]]

    recorded_blocked = last_index if recorded["budget_blocked"] else None
    fires_match = replayed_fired == recorded["fired"]
    block_match = replayed_blocked == recorded_blocked
    suppression_match = replayed_suppressed == recorded["suppressed"]

    new_fire_indices = [d["command_index"] for d in decisions if d["new_fires"]]

    row: dict[str, Any] = {
        "stop_reason": result.get("stop_reason"),
        "segment_length_m": round(
            math.hypot(tx - float(start["x"]), ty - float(start["y"])), 4
        ),
        "linear_commands_sent": result.get("linear_commands_sent"),
        "effective_linear_ceiling": ceiling,
        "linear_execution_mode": result.get("linear_execution_mode"),
        "waypoint_tolerance": tol,
        "heading_tolerance_degrees": heading_tol,
        "realign_threshold_degrees": threshold,
        "pulses_recorded": len(records),
        "decision_points": len(decisions),
        "gates_applied": dict(gate_counts),
        "budget_available": budget,
        "recorded_fired_pulses": recorded["fired"],
        "recorded_budget_blocked_pulse": recorded_blocked,
        "recorded_suppressed_pulses": recorded["suppressed"],
        "replayed_fired_pulses": replayed_fired,
        "replayed_budget_blocked_pulse": replayed_blocked,
        "replayed_suppressed_pulses": replayed_suppressed,
        "recorded_decisions": len(recorded["fired"])
        + (1 if recorded_blocked is not None else 0),
        "old_trigger_would_fire": len(old_fire_indices),
        "new_trigger_would_fire": len(new_fire_indices),
        "new_fired_pulses": new_fire_indices,
        # The self-validation. Replaying the OLD trigger must reproduce the
        # realignments the run recorded -- the same pulses, in order, plus the
        # budget-blocked decision that leaves no record.
        "replay_matches_recorded": fires_match and block_match,
        "replay_matches_suppressions": suppression_match,
        "replay_matches_strict": fires_match and block_match and suppression_match,
        # Non-trivial means the run itself recorded at least one decision to
        # check against. A segment with nothing recorded matches by default and
        # must not be allowed to pad the ratio.
        "has_recorded_decision": bool(
            recorded["fired"] or recorded["suppressed"] or recorded_blocked is not None
        ),
        "law_version_conflicts": _law_version_conflicts(markers),
        "new_exceeds_budget_3": len(new_fire_indices) > _DEFAULT_VIO_MAX_REALIGNMENTS,
        "decisions": [
            {
                "command_index": d["command_index"],
                "distance_to_target_m": round(d["distance_to_target_m"], 4),
                "aim_error_degrees": round(d["aim_error_degrees"], 3),
                "metres_per_pulse": d["metres_per_pulse"],
                "projected_landing_m": d["projected_landing_m"],
                "outcome": d["outcome"],
                "old_fires": d["old_fires"],
                "old_suppressed": d["old_suppressed"],
                "new_fires": d["new_fires"],
            }
            for d in decisions
        ],
    }
    row.update(_geometry_residuals(decisions, result))
    return row


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
        results = _segment_results(doc)
        markers = _build_markers(results)
        for i, res in enumerate(results):
            rep = replay_segment(res, markers)
            if "skipped" in rep:
                skipped.append((p.name, i, rep["skipped"]))
                continue
            rep["file"] = p.name
            rep["segment_index"] = i
            rows.append(rep)
    return rows, skipped


def _print_table(rows: list[dict]) -> None:
    """Print the per-segment replay comparison."""
    print(f"\nfloor in use: {_MIN_CORRECTABLE_AIM_ERROR_DEGREES} deg\n")
    hdr = (
        f"{'file':50} {'seg':>3} {'len':>6} {'pul':>4} {'dec':>4} "
        f"{'rec':>4} {'old':>4} {'new':>4} {'ok':>3} {'>3?':>4}  stop_reason"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(
        rows,
        key=lambda r: (-r["new_trigger_would_fire"], r["file"], r["segment_index"]),
    ):
        print(
            f"{r['file'][:50]:50} {r['segment_index']:>3} "
            f"{r['segment_length_m']:>6.2f} {r['pulses_recorded']:>4} "
            f"{r['decision_points']:>4} "
            f"{r['recorded_decisions']:>4} {r['old_trigger_would_fire']:>4} "
            f"{r['new_trigger_would_fire']:>4} "
            f"{'y' if r['replay_matches_recorded'] else 'NO':>3} "
            f"{'YES' if r['new_exceeds_budget_3'] else '':>4}  {r['stop_reason']}"
        )
    dist = collections.Counter(r["new_trigger_would_fire"] for r in rows)
    print(
        "\nNEW-trigger fires per segment: "
        + ", ".join(f"{k}: {dist[k]}" for k in sorted(dist))
    )
    over = [r for r in rows if r["new_exceeds_budget_3"]]
    print(f"segments where the NEW trigger would exceed a budget of 3: {len(over)}")
    for r in over:
        print(f"  {r['file']}#{r['segment_index']}: {r['new_fired_pulses']}")
    old_total = sum(r["old_trigger_would_fire"] for r in rows)
    new_total = sum(r["new_trigger_would_fire"] for r in rows)
    points = sum(r["decision_points"] for r in rows)
    print(
        f"\ndecision points: {points}  old fires: {old_total}  new fires: {new_total}"
    )


def _print_summary(rows: list[dict], skipped: list[tuple]) -> None:
    """Print corpus counts and the self-validation ratio."""
    matched = [r for r in rows if r["replay_matches_recorded"]]
    strict = [r for r in rows if r["replay_matches_strict"]]
    checked = sum(r["geometry_points_checked"] for r in rows)
    worst_deg = max((r["geometry_max_error_degrees"] for r in rows), default=0.0)
    worst_m = max((r["geometry_max_error_m"] for r in rows), default=0.0)
    print(f"replayable segments: {len(rows)}")
    if skipped:
        print(f"skipped segments: {len(skipped)}")
        for reason, n in sorted(collections.Counter(s[2] for s in skipped).items()):
            print(f"  {reason}: {n}")
    print(
        f"replay self-validation (old trigger reproduces recorded realignments): "
        f"{len(matched)}/{len(rows)}"
    )
    print(f"  ... also reproducing recorded suppressions: {len(strict)}/{len(rows)}")
    live = [r for r in rows if r["has_recorded_decision"]]
    live_strict = [r for r in live if r["replay_matches_strict"]]
    print(
        f"  ... restricted to segments the run recorded a decision in: "
        f"{len(live_strict)}/{len(live)}"
    )
    clean = [r for r in rows if not r["law_version_conflicts"]]
    clean_ok = [r for r in clean if r["replay_matches_strict"]]
    print(
        f"  ... restricted to builds with NO structural law-version conflict: "
        f"{len(clean_ok)}/{len(clean)}"
    )
    conflicted = collections.Counter(
        c for r in rows for c in r["law_version_conflicts"]
    )
    if conflicted:
        print("  structural law-version conflicts in the corpus (segments):")
        for reason, n in sorted(conflicted.items()):
            print(f"    {reason}: {n}")
    mpp_checked = sum(r["metres_per_pulse_points_checked"] for r in rows)
    mpp_worst = max((r["metres_per_pulse_max_error"] for r in rows), default=0.0)
    print(
        f"geometry cross-check against the executor's own records: {checked} points, "
        f"max error {worst_deg} deg / {worst_m} m"
    )
    print(
        f"metres_per_pulse cross-check: {mpp_checked} points, max error {mpp_worst} m"
    )


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

    _print_summary(rows, skipped)

    if args.validate:
        for r in rows:
            if r["replay_matches_recorded"] and r["replay_matches_suppressions"]:
                continue
            kind = "MISMATCH" if not r["replay_matches_recorded"] else "supp-only"
            print(
                f"  {kind} {r['file']}#{r['segment_index']} "
                f"(stop={r['stop_reason']}, decisions={r['decision_points']}):"
            )
            print(
                f"      fired   replayed {r['replayed_fired_pulses']} "
                f"vs recorded {r['recorded_fired_pulses']}"
            )
            print(
                f"      blocked replayed {r['replayed_budget_blocked_pulse']} "
                f"vs recorded {r['recorded_budget_blocked_pulse']}"
            )
            print(
                f"      supp    replayed {r['replayed_suppressed_pulses']} "
                f"vs recorded {r['recorded_suppressed_pulses']}"
            )
            print(
                f"      law-version conflicts in this build: "
                f"{r['law_version_conflicts'] or 'NONE -- unexplained'}"
            )
        return 0

    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
