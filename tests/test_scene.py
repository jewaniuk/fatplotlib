"""scene geometry."""

import math

from fatplotlib.paths import PathBuilder
from fatplotlib.scene import BBox, Group
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
