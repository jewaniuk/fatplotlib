"""scene geometry."""

import math

import numpy as np
import pytest

from fatplotlib.paths import PathBuilder
from fatplotlib.scene import BBox, Group, Markers, Path, Polylines
from fatplotlib.style import Stroke
from fatplotlib.transform import Affine


def test_cubic_bbox_uses_extrema_not_control_points():
    # control points reach y = 1, the curve only y = 0.75 at t = 0.5
    path = PathBuilder().move_to(0, 0).cubic_to(0, 1, 1, 1, 1, 0).path()
    assert path.bbox() == BBox(0, 0, 1, 0.75)


def test_cubic_bbox_includes_stroke():
    path = PathBuilder().move_to(0, 0).cubic_to(0, 1, 1, 1, 1, 0).path(stroke=Stroke("#000000", width=0.2))
    assert path.bbox() == BBox(-0.1, -0.1, 1.1, 0.85)


def test_group_bbox_applies_transform():
    line = PathBuilder().move_to(0, 0).rel_line_to(2, 0).path()
    group = Group([line], transform=Affine.rotate(math.pi / 2))
    box = group.bbox()
    assert box is not None
    assert math.isclose(box.x0, 0, abs_tol=1e-12) and math.isclose(box.y1, 2)


def test_group_bbox_is_cut_to_clip():
    line = PathBuilder().move_to(-5, 1).line_to(15, 1).path()
    assert Group([line], clip=BBox(0, 0, 10, 10)).bbox() == BBox(0, 1, 10, 1)
    assert Group([line], clip=BBox(20, 20, 30, 30)).bbox() is None


def test_length_of_lines_and_close():
    path = PathBuilder().move_to(0, 0).line_to(3, 0).line_to(3, 4).close().path()
    assert math.isclose(path.length(), 12)
    # a move adds nothing
    assert math.isclose(PathBuilder().move_to(0, 0).line_to(1, 0).move_to(5, 5).line_to(5, 7).path().length(), 3)


def test_length_of_a_full_circle_from_arc_to():
    path = PathBuilder().move_to(2, 0).arc_to(0, 360, 2).path()
    # 90 degree cubic arcs bulge up to 3e-4 of the radius past the circle, 1.4e-4 longer in all
    assert math.isclose(path.length(), 4 * math.pi, rel_tol=2e-4)


def test_length_of_a_cubic_matches_dense_sampling():
    p = np.array([(0, 0), (1, 3), (4, -2), (5, 1)], dtype=float)
    t = np.linspace(0, 1, 200_001)[:, None]
    points = (1 - t) ** 3 * p[0] + 3 * t * (1 - t) ** 2 * p[1] + 3 * t**2 * (1 - t) * p[2] + t**3 * p[3]
    reference = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    path = PathBuilder().move_to(0, 0).cubic_to(1, 3, 4, -2, 5, 1).path()
    assert math.isclose(path.length(), reference, rel_tol=1e-9)


def test_point_at_endpoints_and_range():
    path = PathBuilder().move_to(1, 2).line_to(4, 2).line_to(4, 6).path()
    start, end = path.point_at(0), path.point_at(1)
    assert (start.x, start.y, start.angle) == (1, 2, 0)
    assert math.isclose(end.x, 4) and math.isclose(end.y, 6) and math.isclose(end.angle, 90)
    mid = path.point_at(5 / 7)  # 5 of 7 along: 2 up the second segment
    assert math.isclose(mid.x, 4) and math.isclose(mid.y, 4)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            path.point_at(bad)
    with pytest.raises(ValueError):
        Path((("M", 0.0, 0.0),)).point_at(0.5)


def test_point_at_midpoint_of_a_symmetric_cubic():
    path = PathBuilder().move_to(0, 0).cubic_to(0, 3, 4, 3, 4, 0).path()
    mid = path.point_at(0.5)
    assert math.isclose(mid.x, 2, abs_tol=1e-9) and math.isclose(mid.y, 2.25)
    assert math.isclose(mid.angle, 0, abs_tol=1e-9)
    # a control point on the start: the tangent comes from just inside
    assert math.isclose(PathBuilder().move_to(0, 0).cubic_to(0, 0, 1, 1, 2, 0).path().point_at(0).angle, 45, abs_tol=1e-3)


def test_point_at_follows_a_circle():
    path = PathBuilder().move_to(2, 0).arc_to(0, 360, 2).path()
    for fraction in (0.1, 0.25, 0.4, 0.75, 0.9):
        p = path.point_at(fraction)
        theta = 360 * fraction
        assert math.isclose(math.hypot(p.x, p.y), 2, rel_tol=1e-3)
        assert math.isclose(math.degrees(math.atan2(p.y, p.x)) % 360, theta, abs_tol=0.05)
        # tangent is 90 degrees ahead of the radius, compared on the circle of angles
        assert abs((p.angle - theta - 90 + 180) % 360 - 180) < 0.05


def test_polylines_bbox_skips_non_finite_points():
    xs = np.array([[0, 1, 2], [0, 1, 2]], dtype=float)
    ys = np.array([[0, 5, np.nan], [-1, 2, 3]], dtype=float)
    assert Polylines(xs, ys, stroke=Stroke("#000000", 1)).bbox() == BBox(-0.5, -1.5, 2.5, 5.5)
    assert Polylines(np.full((1, 2), np.nan), np.zeros((1, 2))).bbox() is None


def test_markers_bbox_reaches_each_shape_and_skips_non_finite_points():
    xs, ys = np.array([0.0, 10.0, np.nan]), np.array([0.0, 5.0, 1.0])
    assert Markers(xs, ys, "s", 4).bbox() == BBox(-2, -2, 12, 7)
    # the diamond's corners reach past half the size, the thin one is narrower
    diamond = Markers(xs, ys, "D", 4, stroke=Stroke("#000000", 0.5)).bbox()
    assert diamond is not None
    h = 2 * math.sqrt(2) + 0.25
    assert np.allclose([diamond.x0, diamond.y0, diamond.x1, diamond.y1], [-h, -h, 10 + h, 5 + h])
    thin = Markers(xs, ys, "d", 4).bbox()
    assert thin is not None and math.isclose(thin.x0, -0.6 * 2 * math.sqrt(2))
    assert Markers(np.array([np.nan]), np.array([0.0])).bbox() is None
