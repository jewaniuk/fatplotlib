"""axes: nice-number ticks, labels, the data transform and decorations."""

import math

import numpy as np
import pytest

from fatplotlib.axes import Axes, Scale
from fatplotlib.scene import BBox, Text
from fatplotlib.style import Stroke


def test_linear_ticks_take_the_smallest_nice_step_that_fits():
    assert Scale("linear", (0, 3995)).ticks(8) == [500.0 * k for k in range(8)]
    assert Scale("linear", (0, 1)).ticks(4) == [0, 0.25, 0.5, 0.75, 1]


def test_linear_ticks_have_no_float_noise():
    assert Scale("linear", (0, 0.7)).ticks(7) == [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert Scale("linear", (0, 0.7)).labels([0, 0.1, 0.3]) == ["$0.0$", "$0.1$", "$0.3$"]


def test_reversed_negative_and_zero_width_ranges():
    assert Scale("linear", (1, -1)).ticks(4) == [-1, -0.5, 0, 0.5, 1]
    assert Scale("linear", (1, -1)).fraction([1, -1]).tolist() == [0, 1]
    assert Scale("linear", (-0.5, 0)).labels([-0.5, 0]) == ["$-0.5$", "$0.0$"]
    assert Scale("linear", (0, 0)).lim == (-0.05, 0.05)
    assert Scale("log", (10, 10)).lim == (1, 100)


def test_log_ticks_stride_from_the_lowest_decade():
    scale = Scale("log", (1e-3, 1e8))
    majors = scale.ticks(5)
    assert majors == [10.0**k for k in range(-3, 8, 2)]
    assert scale.labels(majors)[:2] == ["$10^(-3)$", "$10^(-1)$"]
    assert scale.minor_ticks(majors) == [10.0**k for k in range(-2, 9, 2)]


def test_log_minors_fill_decades_when_majors_are_adjacent():
    scale = Scale("log", (0.7, 1.3e3))
    majors = scale.ticks(5)
    assert majors == [1, 10, 100, 1000]
    minors = scale.minor_ticks(majors)
    assert minors[0] == 0.7 and minors[-1] == 900
    assert 2 in minors and 10 not in minors


def test_linear_minors_split_steps_in_four_or_five():
    assert Scale("linear", (0, 10)).minor_ticks([0, 5, 10]) == [1, 2, 3, 4, 6, 7, 8, 9]
    assert Scale("linear", (0, 4)).minor_ticks([0, 2, 4]) == [0.5, 1, 1.5, 2.5, 3, 3.5]


def test_log_scale_needs_positive_limits():
    with pytest.raises(ValueError, match="positive"):
        Scale("log", (0, 10))


def axes(**kwargs: object) -> Axes:
    """A 100 x 50 pt log-linear axes at (20, 10)."""
    return Axes(BBox(20, 10, 120, 60), xlim=(0, 10), ylim=(1, 1000), yscale="log", **kwargs)  # pyright: ignore[reportArgumentType]


def test_data_and_axes_coords_map_to_pt():
    ax = axes()
    x, y = ax.to_pt([0, 10, 5], [1, 1000, math.sqrt(1000)])
    assert np.allclose(x, [20, 120, 70]) and np.allclose(y, [10, 60, 35])
    x, y = ax.to_pt([0.5], [0.5], "axes")
    assert np.allclose(x, [70]) and np.allclose(y, [35])


def test_line_breaks_where_log_cannot_map():
    path = axes().line([1, 2, 3, 4], [10, 0, 10, 10], Stroke("#000000"))
    assert [c[0] for c in path.commands] == ["M", "M", "L"]


def test_outer_box_reaches_measured_labels():
    ax = axes(xlabel="Length", ylabel="Rate")
    outer = ax.outer()
    assert outer.x0 < ax.inner.x0 - 10 and outer.y0 < ax.inner.y0 - 10
    # the last tick labels overhang the right and top spines
    assert outer.x1 > ax.inner.x1 + 2 and outer.y1 > ax.inner.y1 + 2


def test_twin_x_labels_the_top():
    ax = axes()
    ax.twin_x(ticks=[0, 5, 10], labels=["0", "1", "2"], label="Nodes")
    tops = [t for t in _texts(ax) if t.y > ax.inner.y1]
    assert [t.markup for t in tops] == ["0", "1", "2", "Nodes"]
    assert ax.outer().y1 > ax.inner.y1 + 10


def _texts(ax: Axes) -> list[Text]:
    """Every text node in the axes, depth first."""
    found, stack = [], [ax.group()]
    while stack:
        node = stack.pop(0)
        if isinstance(node, Text):
            found.append(node)
        stack[:0] = getattr(node, "children", [])
    return found
