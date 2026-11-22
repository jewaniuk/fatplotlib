"""path builder curves: arcs, quadratics, svg arcs, fillets, and drawing in a panel frame."""

import math
import xml.etree.ElementTree as ET

import pytest

from fatplotlib.panel import Frame
from fatplotlib.paths import PathBuilder
from fatplotlib.render.svg import render
from fatplotlib.scene import BBox, Ellipse
from fatplotlib.style import Stroke


def close(a: tuple[float, ...], b: tuple[float, ...], tol: float = 1e-9) -> bool:
    return all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b, strict=True))


def bbox_close(a: BBox | None, b: BBox, tol: float = 1e-9) -> bool:
    assert a is not None
    return close((a.x0, a.y0, a.x1, a.y1), (b.x0, b.y0, b.x1, b.y1), tol)


def sample(builder: PathBuilder, n: int = 50) -> list[tuple[float, float]]:
    """Points along every cubic in the path."""
    points, current = [], (0.0, 0.0)
    for command in builder.commands:
        if command[0] == "C":
            _, x1, y1, x2, y2, x, y = command
            (x0, y0) = current
            for i in range(n + 1):
                t = i / n
                w = ((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3)
                points.append((w[0] * x0 + w[1] * x1 + w[2] * x2 + w[3] * x, w[0] * y0 + w[1] * y1 + w[2] * y2 + w[3] * y))
        if command[0] != "Z":
            current = command[-2:]  # type: ignore[assignment]
    return points


# arc_to: unit circle about the origin, pen at angle theta_in


def arc(theta_in: float, theta_out: float) -> PathBuilder:
    a = math.radians(theta_in)
    return PathBuilder().move_to(math.cos(a), math.sin(a)).arc_to(theta_in, theta_out, 1)


def test_arc_bbox_quarter():
    assert bbox_close(arc(0, 90).path().bbox(), BBox(0, 0, 1, 1))


def test_arc_bbox_three_quarters():
    assert bbox_close(arc(0, 270).path().bbox(), BBox(-1, -1, 1, 1))


def test_arc_bbox_across_360():
    # 300 to 60 crosses 0, so the arc reaches x = 1 and stays right of x = 0.5
    assert bbox_close(arc(300, 60).path().bbox(), BBox(0.5, -math.sqrt(3) / 2, 1, math.sqrt(3) / 2))


def test_arc_full_circle_stays_on_radius():
    builder = arc(0, 360)
    assert sum(c[0] == "C" for c in builder.commands) == 4
    # the 90° cubic is within 3e-4 of the circle
    assert all(abs(math.hypot(x, y) - 1) < 3e-4 for x, y in sample(builder))


def test_arc_starts_at_pen_ccw():
    builder = PathBuilder().move_to(5, 5).arc_to(-90, 0, 2)  # centre (5, 7), ends at (7, 7)
    assert close(builder.current, (7, 7))
    assert close(builder.commands[-1][-2:], (7, 7))
    assert bbox_close(builder.path().bbox(), BBox(5, 5, 7, 7))


@pytest.mark.parametrize(
    ("theta_in", "theta_out", "end", "cubics"),
    [
        # matplotlib Path.arc: theta_out moves into (theta_in, theta_in + 360]
        (0, 90, 90, 1),
        (90, 0, 360, 3),
        (0, -90, 270, 3),
        (0, 720, 360, 4),
        (10, 370, 370, 4),
        (45, 45, 45, 0),
    ],
)
def test_arc_wraps_as_matplotlib(theta_in: float, theta_out: float, end: float, cubics: int):
    builder = arc(theta_in, theta_out)
    e = math.radians(end)
    assert close(builder.current, (math.cos(e), math.sin(e)))
    assert sum(c[0] == "C" for c in builder.commands) == cubics


# quad_to


def test_quad_elevation_is_exact():
    p0, c, p1 = (1.0, 2.0), (4.0, 7.0), (6.0, -1.0)
    builder = PathBuilder().move_to(*p0).quad_to(*c, *p1)
    for t, (x, y) in zip([i / 50 for i in range(51)], sample(builder)):
        qx = (1 - t) ** 2 * p0[0] + 2 * t * (1 - t) * c[0] + t**2 * p1[0]
        qy = (1 - t) ** 2 * p0[1] + 2 * t * (1 - t) * c[1] + t**2 * p1[1]
        assert close((x, y), (qx, qy), 1e-12)


def test_rel_quad_matches_absolute():
    a = PathBuilder().move_to(1, 1).quad_to(3, 4, 5, 1)
    b = PathBuilder().move_to(1, 1).rel_quad_to(2, 3, 4, 0)
    assert a.commands == b.commands


# elliptical_arc_to


def test_svg_arc_semicircle():
    # (0, 0) to (2, 0) radius 1: sweep (ccw) passes below, through (1, -1)
    below = PathBuilder().move_to(0, 0).elliptical_arc_to(1, 1, 0, False, True, 2, 0)
    above = PathBuilder().move_to(0, 0).elliptical_arc_to(1, 1, 0, False, False, 2, 0)
    assert bbox_close(below.path().bbox(), BBox(0, -1, 2, 0))
    assert bbox_close(above.path().bbox(), BBox(0, 0, 2, 1))


@pytest.mark.parametrize(
    ("large", "sweep", "centre", "box"),
    [
        # (1, 0) to (0, 1) on unit circles centred at (0, 0) or (1, 1)
        (False, True, (0, 0), BBox(0, 0, 1, 1)),
        (True, True, (1, 1), BBox(0, 0, 2, 2)),
        (False, False, (1, 1), BBox(0, 0, 1, 1)),
        (True, False, (0, 0), BBox(-1, -1, 1, 1)),
    ],
)
def test_svg_arc_flags(large: bool, sweep: bool, centre: tuple[float, float], box: BBox):
    builder = PathBuilder().move_to(1, 0).elliptical_arc_to(1, 1, 0, large, sweep, 0, 1)
    assert bbox_close(builder.path().bbox(), box)
    assert builder.commands[-1][-2:] == (0, 1)
    assert all(abs(math.hypot(x - centre[0], y - centre[1]) - 1) < 3e-4 for x, y in sample(builder))


def test_svg_arc_scales_up_small_radii():
    # radius 0.1 cannot span (0, 0) to (4, 0): it grows to 2, a semicircle
    builder = PathBuilder().move_to(0, 0).elliptical_arc_to(0.1, 0.1, 0, False, False, 4, 0)
    assert bbox_close(builder.path().bbox(), BBox(0, 0, 4, 2))
    assert all(abs(math.hypot(x - 2, y) - 2) < 1e-3 for x, y in sample(builder))


def test_svg_arc_scales_ellipse_keeping_ratio():
    builder = PathBuilder().move_to(0, 0).elliptical_arc_to(0.2, 0.1, 0, False, True, 4, 0)
    assert bbox_close(builder.path().bbox(), BBox(0, -1, 4, 0))


def test_svg_arc_rotated_ellipse():
    # a 2 x 1 ellipse turned 90°: from its bottom (0, -2) ccw to its top (0, 2) through (1, 0)
    builder = PathBuilder().move_to(0, -2).elliptical_arc_to(2, 1, 90, False, True, 0, 2)
    assert bbox_close(builder.path().bbox(), BBox(0, -2, 1, 2))


def test_svg_arc_degenerate():
    assert PathBuilder().move_to(1, 1).elliptical_arc_to(1, 1, 0, False, True, 1, 1).commands == [("M", 1, 1)]
    assert PathBuilder().move_to(0, 0).elliptical_arc_to(0, 1, 0, False, True, 3, 4).commands[-1] == ("L", 3, 4)


def test_rel_svg_arc_matches_absolute():
    a = PathBuilder().move_to(1, 1).elliptical_arc_to(2, 1, 30, True, False, 3, 2)
    b = PathBuilder().move_to(1, 1).rel_elliptical_arc_to(2, 1, 30, True, False, 2, 1)
    assert a.commands == b.commands


# lines with fillets


def test_lines_without_radius_are_line_to():
    builder = PathBuilder().move_to(0, 0).lines([(1, 0), (1, 1), (0, 1)])
    assert builder.commands == [("M", 0, 0), ("L", 1, 0), ("L", 1, 1), ("L", 0, 1)]


def test_fillet_is_tangent_to_both_segments():
    builder = PathBuilder().move_to(0, 0).lines([(10, 0), (10, 10)], radius=2)
    kinds = [c[0] for c in builder.commands]
    assert kinds == ["M", "L", "C", "L"]
    _, line_x, line_y = builder.commands[1]
    _, c1x, c1y, c2x, c2y, x, y = builder.commands[2]
    assert close((line_x, line_y), (8, 0)) and close((x, y), (10, 2))
    # tangent along +x leaving the first segment, along +y into the second
    assert math.isclose(c1y, 0, abs_tol=1e-12) and c1x > line_x
    assert math.isclose(c2x, 10, abs_tol=1e-12) and c2y < y
    assert all(abs(math.hypot(px - 8, py - 2) - 2) < 6e-4 for px, py in sample(builder))


def test_fillet_turns_right_clockwise():
    builder = PathBuilder().move_to(0, 0).lines([(10, 0), (10, -10)], radius=2)
    assert all(abs(math.hypot(px - 8, py + 2) - 2) < 6e-4 for px, py in sample(builder))


def test_fillet_clamps_to_half_the_shorter_segment():
    builder = PathBuilder().move_to(0, 0).lines([(2, 0), (2, 10)], radius=5)
    assert close(builder.commands[1][1:], (1, 0)) and close(builder.commands[2][-2:], (2, 1))


def test_neighbouring_fillets_do_not_overlap():
    # a 2-long middle segment: each corner may use 1 of it
    builder = PathBuilder().move_to(0, 0).lines([(10, 0), (10, 2), (20, 2)], radius=3)
    lines = [c[1:] for c in builder.commands if c[0] == "L"]
    assert close(lines[0], (9, 0)) and close(lines[1], (10, 1)) and close(lines[2], (20, 2))


def test_fillet_skips_straight_corner():
    builder = PathBuilder().move_to(0, 0).lines([(1, 0), (2, 0)], radius=1)
    assert [c[0] for c in builder.commands] == ["M", "L", "L"]


# drawing in a frame


def test_frame_maps_points_and_keeps_fixed_radii():
    frame = Frame(BBox(10, 20, 210, 120), scale=2.0, em=8.0)
    builder = PathBuilder(frame).move_to(0, 0).line_to(10, 0).arc_to(-90, 0, "4pt")
    assert builder.commands[0] == ("M", 10, 20) and builder.commands[1] == ("L", 30, 20)
    # "4pt" is 2 panel units: the quarter arc ends 4 pt right and up of the pen
    assert close(builder.commands[-1][-2:], (34, 24))
    assert close(builder.current, (12, 2))


def test_frame_scales_number_radii():
    frame = Frame(BBox(0, 0, 200, 100), scale=2.0, em=8.0)
    builder = PathBuilder(frame).move_to(0, 0).arc_to(-90, 0, 3)
    assert close(builder.commands[-1][-2:], (6, 6))


# ellipse


def test_ellipse_bbox_includes_stroke():
    assert Ellipse(5, 3, 4, 2, stroke=Stroke("#000000", 0.5)).bbox() == BBox(0.75, 0.75, 9.25, 5.25)


def test_ellipse_svg():
    svg = render(Ellipse(5, 3, 4, 2, fill="#ff0000"), 10, 6)
    (ellipse,) = ET.fromstring(svg).iter("{http://www.w3.org/2000/svg}ellipse")
    assert (ellipse.get("cx"), ellipse.get("cy"), ellipse.get("rx"), ellipse.get("ry")) == ("5", "3", "4", "2")
