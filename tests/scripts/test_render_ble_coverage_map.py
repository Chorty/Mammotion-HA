"""Pins for the BLE coverage map renderer."""  # noqa: INP001

from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.render_ble_coverage_map import (
    DIES_BELOW_DBM,
    PLANNER_MIN_RSSI_DBM,
    PLANNER_RADIUS_M,
    RSSI_BINS,
    WORKS_ABOVE_DBM,
    build_cells,
    point_in_polygon,
    render_html,
    summarize,
)

_SQUARE = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
_AREAS = [{"name": "Test", "points": _SQUARE}]


def _planner_default(flag: str) -> float:
    """Read an argparse default straight out of `plan_aligned_leg`'s source.

    The planner builds its parser inside `main()`, so there is no object to
    introspect without running it. Scanning the source is mechanical and
    survives reformatting, which is what this pin needs: it exists to fail
    when the planner's gate moves and this renderer's copy of it does not.
    """
    source = Path("scripts/plan_aligned_leg.py").read_text()
    match = re.search(
        rf'"{re.escape(flag)}",.*?default=(-?\d+(?:\.\d+)?)', source, re.DOTALL
    )
    assert match is not None, f"{flag} not found in plan_aligned_leg.py"
    return float(match.group(1))


def test_renderer_draws_the_planners_actual_gate() -> None:
    """The map is only meaningful if it depicts the decision the planner makes."""
    assert _planner_default("--min-rssi-dbm") == PLANNER_MIN_RSSI_DBM
    assert _planner_default("--coverage-radius-m") == PLANNER_RADIUS_M


def test_bin_edges_are_the_measured_regime() -> None:
    """Bin boundaries are observations of this link, not round numbers."""
    floors = [floor for floor, _, _, _ in RSSI_BINS]
    assert WORKS_ABOVE_DBM in floors
    assert DIES_BELOW_DBM in floors
    assert floors == sorted(floors, reverse=True), "bins must run strongest first"
    # The bottom bin is a catch-all, so no reading can fall out of the ramp.
    assert floors[-1] < -200


def test_cells_with_no_samples_nearby_are_omitted_not_guessed() -> None:
    """Absent evidence must render as bare ground, never as an interpolated value.

    Optimistically filling unvisited ground is the exact assumption that walks
    a leg into a dead zone, so the renderer's contract is that such cells are
    simply not in the output.
    """
    samples = [{"x": 1.0, "y": 1.0, "rssi": -60.0}]

    cells, in_area = build_cells(samples, _AREAS, cell_m=1.0, radius=PLANNER_RADIUS_M)

    assert in_area == 100, "every 1 m cell centre inside the 10x10 square"
    assert 0 < len(cells) < in_area
    # Nothing further than the pooling radius from the single sample survives.
    assert all(
        (cell[0] + 0.5 - 1.0) ** 2 + (cell[1] + 0.5 - 1.0) ** 2
        <= PLANNER_RADIUS_M**2 + 1e-9
        for cell in cells
    )


def test_cells_outside_the_mowing_area_are_never_drawn() -> None:
    """A sample banked off the map must not paint ground the mower cannot use."""
    samples = [{"x": 50.0, "y": 50.0, "rssi": -55.0}]

    cells, in_area = build_cells(samples, _AREAS, cell_m=1.0, radius=PLANNER_RADIUS_M)

    assert in_area == 100
    assert cells == []


def test_cell_carries_mean_worst_and_count_separately() -> None:
    """A mean hides a dropout, so the worst reading is kept alongside it."""
    samples = [
        {"x": 5.0, "y": 5.0, "rssi": -55.0},
        {"x": 5.2, "y": 5.0, "rssi": -85.0},
    ]

    cells, _ = build_cells(samples, _AREAS, cell_m=1.0, radius=PLANNER_RADIUS_M)
    middle = next(c for c in cells if c[0] == 4.0 and c[1] == 4.0)

    assert middle[2] == -70.0  # mean -- the planner would accept this
    assert middle[3] == -85.0  # worst -- a dropout the mean conceals
    assert middle[4] == 2


def test_point_in_polygon_handles_edges_and_outside() -> None:
    """The containment test decides which ground gets drawn at all."""
    assert point_in_polygon(5.0, 5.0, _SQUARE)
    assert not point_in_polygon(-1.0, 5.0, _SQUARE)
    assert not point_in_polygon(11.0, 5.0, _SQUARE)
    assert not point_in_polygon(5.0, 11.0, _SQUARE)


def test_summary_counts_refusals_against_the_planner_threshold() -> None:
    """The headline refusal count must use the planner's own wall."""
    cells = [
        [0.0, 0.0, -60.0, -62.0, 4],
        [0.0, 1.0, -72.0, -80.0, 3],
        [1.0, 0.0, -80.0, -84.0, 2],
    ]
    samples = [{"x": 0.0, "y": 0.0, "rssi": -60.0}]

    summary = summarize(cells, 10, samples)

    assert summary["cells_refused"] == 1
    assert summary["cells_marginal"] == 1
    assert summary["cells_with_evidence"] == 3
    assert summary["evidence_fraction"] == 0.3
    assert summary["worst_cell"][2] == -80.0


def test_payload_cannot_break_out_of_the_host_script_tag() -> None:
    """Area names come off the device and are injected into a <script> block."""
    payload = {
        "areas": [{"name": "</script><script>alert(1)</script>", "points": []}],
        "cells": [],
        "samples": [],
    }

    html = render_html(payload)
    body = html.split('id="payload"')[1].split("</script>")[0]

    assert "alert(1)" in body, "the name is still carried, just neutralized"
    assert "<\\/script>" in body
    assert json.loads(body.split(">", 1)[1])["areas"][0]["name"].endswith("</script>")
