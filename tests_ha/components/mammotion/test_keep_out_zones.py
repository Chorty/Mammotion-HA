"""Containment must test EXCLUSION from keep-outs, not only inclusion in areas.

Anchor: 2026-08-20, supervised, `0.6.4-beta62`. A 10.8 m Route B click stayed
inside "Backyard Right" for its whole length and drove into an obstacle zone
containing a trampoline. `_validate_custom_path` passed the path as valid
because it asked one question -- "is every point inside a mowing area?" -- and
never asked "is any point inside a keep-out?".

The geometry was never missing. `HashList` stores it in sibling dicts on the
same object as `map.area` (`obstacle`, `no_go_zone`, `virtual_wall`, …), in the
SAME map-local x/y frame, and `get_geojson` has always exposed it -- the mower
reported obstacle hash 1529607395159402290 at contact and the geojson names it
"Obstacle 1", "Obstacle in Backyard Right", roughly 4.0 x 4.1 m. `_area_polygons`
simply read one dict of several.

Containment now tests both waypoints and the complete segment between each pair.
`test_a_leg_that_clips_a_corner_is_caught` pins the closed gap so a future
per-point regression cannot silently restore it.
"""

from __future__ import annotations

from types import SimpleNamespace

from custom_components.mammotion.services import (
    _KEEP_OUT_MAP_FIELDS,
    _keep_out_leg_violations,
    _keep_out_polygons,
    _keep_out_violations,
    _segments_intersect,
    _split_long_legs,
    _validate_custom_path,
)

from .conftest import _coordinator


def _frames(points: list[tuple[float, float]]) -> SimpleNamespace:
    return SimpleNamespace(
        data=[
            SimpleNamespace(
                current_frame=0,
                data_couple=[SimpleNamespace(x=x, y=y) for x, y in points],
            )
        ]
    )


# A 4 x 4 m keep-out, the size the real "Obstacle 1" reports.
_TRAMPOLINE = [(8.0, -3.0), (12.0, -3.0), (12.0, 1.0), (8.0, 1.0)]


def _with_keep_out(field_name: str = "obstacle"):
    coordinator = _coordinator()
    coordinator.data.map.area = {
        123: _frames([(-20.0, -20.0), (20.0, -20.0), (20.0, 20.0), (-20.0, 20.0)])
    }
    setattr(
        coordinator.data.map, field_name, {1529607395159402290: _frames(_TRAMPOLINE)}
    )
    return coordinator


def test_the_trampoline_run_would_now_be_refused() -> None:
    """The regression this exists for, in one assertion.

    Both endpoints inside the mowing area; a midpoint inside the keep-out.
    """
    coordinator = _with_keep_out()

    result = _validate_custom_path(
        coordinator,
        [{"x": 5.0, "y": -5.0}, {"x": 10.0, "y": -1.0}, {"x": 15.0, "y": 3.0}],
        area_hash=123,
    )

    assert result["valid"] is False
    assert "path_points_inside_keep_out_zone" in result["errors"]
    # And it says WHICH point and WHICH zone, not just that something is wrong.
    violation = result["keep_out_violations"][0]
    assert violation["point_index"] == 1
    assert violation["keep_out_type"] == "obstacle"
    assert violation["keep_out_hash"] == "1529607395159402290"
    assert result["keep_out_zones_checked"] == 1


def test_a_clean_path_through_the_same_area_still_passes() -> None:
    """The paired assertion: exclusion must not refuse a legal path."""
    coordinator = _with_keep_out()

    result = _validate_custom_path(
        coordinator,
        [{"x": -10.0, "y": -10.0}, {"x": -5.0, "y": -5.0}],
        area_hash=123,
    )

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["keep_out_violations"] == []
    assert result["keep_out_zones_checked"] == 1


def test_every_keep_out_field_is_honoured_not_just_obstacle() -> None:
    """Only `obstacle` is confirmed populated live; the rest must still work.

    A keep-out we cannot see is the exact failure this fixes.
    """
    for field_name in _KEEP_OUT_MAP_FIELDS:
        coordinator = _with_keep_out(field_name)
        result = _validate_custom_path(
            coordinator, [{"x": 10.0, "y": -1.0}, {"x": 11.0, "y": 0.0}], area_hash=123
        )
        assert result["valid"] is False, field_name
        assert result["keep_out_violations"][0]["keep_out_type"] == field_name


def test_absent_keep_out_geometry_warns_rather_than_passing_silently() -> None:
    """A clean pass and an unchecked pass are different answers.

    Silence is how the trampoline run passed validation.
    """
    coordinator = _coordinator()
    coordinator.data.map.area = {
        123: _frames([(-20.0, -20.0), (20.0, -20.0), (20.0, 20.0), (-20.0, 20.0)])
    }

    result = _validate_custom_path(
        coordinator, [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 2.0}], area_hash=123
    )

    assert result["valid"] is True
    assert "no_keep_out_geometry_available_for_exclusion_check" in result["warnings"]
    assert result["keep_out_zones_checked"] == 0


def test_a_keep_out_is_honoured_even_with_no_area_geometry() -> None:
    """Exclusion does not depend on inclusion being checkable."""
    coordinator = _coordinator()
    coordinator.data.map.area = {}
    coordinator.data.map.obstacle = {77: _frames(_TRAMPOLINE)}

    result = _validate_custom_path(
        coordinator, [{"x": 10.0, "y": -1.0}, {"x": 11.0, "y": 0.0}]
    )

    assert result["valid"] is False
    assert "path_points_inside_keep_out_zone" in result["errors"]


def test_a_leg_that_clips_a_corner_is_caught() -> None:
    """Pin the closed gap: legal endpoints cannot hide an illegal segment."""
    coordinator = _with_keep_out()

    result = _validate_custom_path(
        coordinator,
        [{"x": 6.0, "y": -1.0}, {"x": 14.0, "y": -1.0}],
        area_hash=123,
    )

    assert result["valid"] is False
    assert result["keep_out_violations"] == []
    assert "path_legs_cross_keep_out_zone" in result["errors"]
    assert result["keep_out_leg_violations"] == [
        {
            "leg_index": 0,
            "start_point_index": 0,
            "end_point_index": 1,
            "start": {"x": 6.0, "y": -1.0},
            "end": {"x": 14.0, "y": -1.0},
            "keep_out_type": "obstacle",
            "keep_out_hash": "1529607395159402290",
        }
    ]


def test_a_leg_touching_a_keep_out_boundary_is_caught() -> None:
    """Containment includes the boundary, for both points and segments."""
    coordinator = _with_keep_out()

    result = _validate_custom_path(
        coordinator,
        [{"x": 6.0, "y": -3.0}, {"x": 14.0, "y": -3.0}],
        area_hash=123,
    )

    assert result["valid"] is False
    assert "path_legs_cross_keep_out_zone" in result["errors"]


def test_splitting_a_crossing_leg_remains_refused() -> None:
    """Inserted collinear points do not weaken segment containment.

    This split lands points inside the zone, so the established point-level
    reason fires too. The path was already refused before segment containment;
    the new check preserves that behaviour.
    """
    coordinator = _with_keep_out()
    split = _split_long_legs(
        [{"x": 6.0, "y": -1.0}, {"x": 14.0, "y": -1.0}],
        target_length_m=3.2,
    )

    result = _validate_custom_path(coordinator, split["points"], area_hash=123)

    assert result["valid"] is False
    assert "path_points_inside_keep_out_zone" in result["errors"]


def test_segment_and_leg_helpers_ignore_a_clear_path() -> None:
    """The paired negative case prevents the edge test from blocking broadly."""
    polygon = [{"x": x, "y": y} for x, y in _TRAMPOLINE]
    points = [{"x": 6.0, "y": -5.0}, {"x": 14.0, "y": -5.0}]

    assert not _segments_intersect(points[0], points[1], polygon[0], polygon[1])
    assert not _keep_out_leg_violations(points, {"obstacle:1": polygon})


def test_polygons_come_back_in_map_local_xy_with_no_conversion() -> None:
    """Same frame as the path planner -- the whole reason this is cheap."""
    coordinator = _with_keep_out()

    polygons = _keep_out_polygons(coordinator)

    assert list(polygons) == ["obstacle:1529607395159402290"]
    assert polygons["obstacle:1529607395159402290"][0] == {"x": 8.0, "y": -3.0}
    assert _keep_out_violations([{"x": 10.0, "y": -1.0}], polygons)
    assert not _keep_out_violations([{"x": 0.0, "y": 0.0}], polygons)


def test_a_degenerate_keep_out_cannot_refuse_everything() -> None:
    """A 2-point 'polygon' bounds no area and must not block a path."""
    coordinator = _coordinator()
    coordinator.data.map.obstacle = {5: _frames([(0.0, 0.0), (1.0, 1.0)])}

    assert (
        _keep_out_violations([{"x": 0.5, "y": 0.5}], _keep_out_polygons(coordinator))
        == []
    )
