"""axes: the data transform, drawing and decorations."""

import math

import numpy as np
import pytest

from fatplotlib.axes import Axes, Colorbar, Marker
from fatplotlib.colormap import DEFAULT, Colormap, Norm
from fatplotlib.scene import BBox, Group, Markers, Path, Polylines, Rect, Text, Themed
from fatplotlib.style import LinearGradient, Stroke


def axes(**kwargs: object) -> Axes:
    """A 100 x 50 pt log-linear axes at (20, 10), with 9 pt em."""
    ax = Axes(xlim=(0, 10), ylim=(1, 1000), yscale="log", **kwargs)  # pyright: ignore[reportArgumentType]
    ax.place(BBox(20, 10, 120, 60), 9)
    return ax


def test_data_and_axes_coords_map_to_pt():
    ax = axes()
    x, y = ax.to_pt([0, 10, 5], [1, 1000, math.sqrt(1000)])
    assert np.allclose(x, [20, 120, 70]) and np.allclose(y, [10, 60, 35])
    x, y = ax.to_pt([0.5], [0.5], "axes")
    assert np.allclose(x, [70]) and np.allclose(y, [35])


def test_line_breaks_where_log_cannot_map():
    ax = axes()
    ax.line([1, 2, 3, 4], [10, 0, 10, 10], Stroke("#000000"))
    data = ax.group().children[0]
    assert isinstance(data, Group) and isinstance(path := data.children[0], Path)
    assert [c[0] for c in path.commands] == ["M", "M", "L"]


def test_lines_share_x_and_map_every_curve():
    ax = axes()
    ax.lines([0, 5, 10], [[1, 10, 100], [1000, 0, 1]], Stroke("#000000"), alpha=0.2)
    data = ax.group().children[0]
    assert isinstance(data, Group) and isinstance(node := data.children[0], Polylines)
    assert node.alpha == 0.2 and node.xs.shape == node.ys.shape == (2, 3)
    assert np.allclose(node.xs, [[20, 70, 120]] * 2)
    assert np.allclose(node.ys[0], [10, 10 + 50 / 3, 10 + 100 / 3]) and node.ys[1, 0] == 60
    assert not np.isfinite(node.ys[1, 1])


def test_drawing_waits_for_placement():
    ax = Axes(xlim=(0, 10), ylim=(0, 1))
    ax.line([0, 10], [0, 1], Stroke("#000000"))
    with pytest.raises(RuntimeError, match="layout"):
        ax.group()
    ax.place(BBox(0, 0, 100, 50), 9)
    ax.place(BBox(0, 0, 200, 50), 9)
    data = ax.group().children[0]
    assert isinstance(data, Group) and data.bbox() == BBox(0, 0, 200, 50)


def test_text_sizes_scale_with_em():
    ax = axes(xlabel="Length")
    assert sorted({round(t.size, 6) for t in _texts(ax)}) == [8, 9]
    # bigger text, fewer ticks fit
    ax.place(BBox(20, 10, 120, 60), 18)
    assert sorted({round(t.size, 6) for t in _texts(ax)}) == [16, 18]


def test_outer_box_reaches_measured_labels():
    ax = axes(xlabel="Length", ylabel="Rate")
    outer, inner = ax.outer(), ax.inner
    assert inner is not None
    assert outer.x0 < inner.x0 - 10 and outer.y0 < inner.y0 - 10
    # the last tick labels overhang the right and top spines
    assert outer.x1 > inner.x1 + 2 and outer.y1 > inner.y1 + 2


def test_twin_x_labels_the_top():
    ax = axes()
    ax.twin_x(ticks=[0, 5, 10], labels=["0", "1", "2"], label="Nodes")
    inner = ax.inner
    assert inner is not None
    tops = [t for t in _texts(ax) if t.y > inner.y1]
    assert [t.markup for t in tops] == ["0", "1", "2", "Nodes"]
    assert ax.outer().y1 > inner.y1 + 10


def _texts(ax: Axes) -> list[Text]:
    """Every text node in the axes, depth first."""
    found, stack = [], [ax.group()]
    while stack:
        node = stack.pop(0)
        if isinstance(node, Text):
            found.append(node)
        stack[:0] = getattr(node, "children", [])
    return found


def test_markers_size_in_em_and_line_markers_are_clipped_data():
    ax = axes()
    ax.line([0, 10], [1, 1000], Stroke("#000000"), marker=Marker("s", "0.5em"))
    data = ax.group().children[0]
    assert isinstance(data, Group) and isinstance(dots := data.children[1], Markers)
    assert dots.size == 4.5 and dots.shape == "s"
    assert np.allclose(dots.xs, [20, 120]) and np.allclose(dots.ys, [10, 60])
    ax.place(BBox(20, 10, 120, 60), 18)
    dots = ax.group().children[0].children[1]  # pyright: ignore[reportAttributeAccessIssue]
    assert dots.size == 9


def test_unclipped_markers_count_toward_outer():
    ax = axes()
    before = ax.outer()
    ax.markers([1.1], [0.5], Marker(size=20), "axes", clip=False)
    # centred at 20 + 1.1 * 100 = 130, radius 10
    assert ax.outer().x1 == 140 and ax.outer().x0 == before.x0


def test_colour_mapped_markers_group_by_lut_colour():
    ax = axes()
    cmap = Colormap("#000000", "#ffffff", levels=2)
    ax.markers([1, 2, 3, 4], [10, 10, 10, 10], values=[0, 0.2, 0.9, np.nan], cmap=cmap, norm=Norm("linear", (0, 1)))
    group = ax.group().children[0].children[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(group, Group) and [len(m.xs) for m in group.children] == [2, 1]  # pyright: ignore[reportAttributeAccessIssue]
    assert [m.fill for m in group.children] == ["#000000", "#ffffff"]  # pyright: ignore[reportAttributeAccessIssue]


def test_per_theme_colormap_markers_split_by_theme():
    ax = axes()
    cmap = Colormap.per_theme(light=Colormap("#000000", "#ff0000"), dark=Colormap("#ffffff", "#00ff00"))
    ax.markers([1, 2], [10, 100], values=[0, 1], cmap=cmap)
    node = ax.group().children[0].children[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(node, Themed) and set(node.children) == {"light", "dark"}


def test_heatmap_groups_cells_by_colour_and_skips_or_colours_bad_cells():
    ax = axes()
    z = [[0.0, 1.0], [np.nan, 0.1]]
    ax.heatmap([0, 5, 10], [1, 10, 1000], z, Colormap("#000000", "#ffffff", levels=2))
    ax.heatmap([0, 5, 10], [1, 10, 1000], z, Colormap("#000000", "#ffffff", levels=2, bad="#ff0000"))
    plain, with_bad = ax.group().children[0].children  # pyright: ignore[reportAttributeAccessIssue]
    assert [p.fill for p in plain.children] == ["#000000", "#ffffff"]
    assert [p.fill for p in with_bad.children] == ["#ff0000", "#000000", "#ffffff"]
    dark = plain.children[0]
    assert [c[0] for c in dark.commands].count("M") == 2
    # cell (1, 1): x 5 to 10 is 70 to 120 pt, y 10 to 1000 is a third of the way up to the top
    assert np.allclose([c[1:] for c in dark.commands[5:9]], [(70, 10 + 50 / 3), (120, 10 + 50 / 3), (120, 60), (70, 60)])


def test_fill_between_closes_a_region_per_finite_run():
    ax = axes()
    ax.fill_between([0, 1, 2, 3, 4], 1, [10, 10, np.nan, 10, 10], "#000000")
    path = ax.group().children[0].children[0]  # pyright: ignore[reportAttributeAccessIssue]
    ops = [c[0] for c in path.commands]
    assert ops == ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
    # along the top, then back along the flat bottom
    assert np.allclose([c[1:] for c in path.commands[:4]], [(20, 10 + 50 / 3), (30, 10 + 50 / 3), (30, 10), (20, 10)])


def test_grid_lines_sit_at_the_major_ticks_under_the_data():
    ax = axes()
    ax.line([0, 10], [1, 1000], Stroke("#000000"))
    ax.grid("both")
    data = ax.group().children[0]
    assert isinstance(data, Group)
    grid = data.children[0]
    assert isinstance(grid, Path)
    starts = [c[1:] for c in grid.commands if c[0] == "M"]
    xs = sorted({x for x, y in starts if y == 10})
    ys = sorted({y for x, y in starts if x == 20})
    majors = ax._decorations()[0]  # the major tick path, starting each tick on a spine
    assert isinstance(majors, Path)
    tick_starts = [c[1:] for c in majors.commands if c[0] == "M"]
    assert xs == sorted({x for x, y in tick_starts if y == 10})
    assert ys == sorted({y for x, y in tick_starts if x == 20})


def strip(bar: Colorbar) -> object:
    """The first node drawn in a placed colorbar's clipped data."""
    return bar.group().children[0].children[0]  # pyright: ignore[reportAttributeAccessIssue]


def tick_starts(bar: Colorbar) -> list[tuple[float, float]]:
    """Where each major tick starts, on a spine."""
    majors = bar._decorations()[0]
    assert isinstance(majors, Path)
    return [c[1:] for c in majors.commands if c[0] == "M"]  # pyright: ignore[reportReturnType]


def test_vertical_colorbar_labels_the_right_and_fills_a_gradient():
    bar = Colorbar(DEFAULT, Norm("linear", (0, 1)), label="Rate")
    bar.place(BBox(100, 10, 109, 110), 9)
    rect = strip(bar)
    assert isinstance(rect, Rect) and isinstance(gradient := rect.fill, LinearGradient)
    assert len(gradient.stops) == 64 and (gradient.x2, gradient.y2) == (0, 1)
    assert gradient.stops[0].colour == DEFAULT.entries()[0] and gradient.stops[-1].colour == DEFAULT.entries()[-1]
    assert {x for x, _ in tick_starts(bar)} == {109}
    ticks, label = bar._decorations()[3:]
    assert isinstance(ticks, Group) and all(isinstance(t, Text) and t.anchor == "start" for t in ticks.children)
    # the axis label reads top to bottom, past the tick labels
    assert isinstance(label, Group) and label.transform is not None
    assert (box := label.bbox()) is not None and (tbox := ticks.bbox()) is not None and box.x0 > tbox.x1
    assert bar.label is False and bar.width == "1em" and bar.height is None


def test_horizontal_colorbar_labels_below_and_runs_the_gradient_along_x():
    bar = Colorbar(DEFAULT, Norm("linear", (0, 1)), side="bottom")
    bar.place(BBox(10, 10, 110, 19), 9)
    rect = strip(bar)
    assert isinstance(rect, Rect) and isinstance(gradient := rect.fill, LinearGradient)
    assert (gradient.x2, gradient.y2) == (1, 0)
    assert {y for _, y in tick_starts(bar)} == {10}
    (ticks,) = bar._decorations()[3:]
    assert isinstance(ticks, Group) and all(isinstance(t, Text) and t.valign == "cap" for t in ticks.children)


def test_log_colorbar_ticks_decades_with_minors():
    bar = Colorbar(DEFAULT, Norm("log", (1e-9, 1e7)), ticks=(1e-9, 1e-5, 1e-1, 1e3, 1e7))
    bar.place(BBox(100, 10, 109, 110), 9)
    _, minors, labels = bar._resolved()["right"]
    assert labels[1] == "$10^(-5)$" and minors
    ys = sorted(y for _, y in tick_starts(bar))
    assert np.allclose(ys, [10, 35, 60, 85, 110])


def test_stepped_colorbars_stack_a_rect_per_level_or_bin():
    cmap = Colormap("#ffffff", "#fc69a0", levels=4)
    bar = Colorbar(cmap, Norm("linear", (0, 1)))
    bar.place(BBox(100, 10, 109, 110), 9)
    bands = strip(bar)
    assert isinstance(bands, Group)
    assert [(r.y, r.height, r.fill) for r in bands.children] == [  # pyright: ignore[reportAttributeAccessIssue]
        (10 + 25 * i, 25, cmap.entries()[i]) for i in range(4)
    ]
    cmap = Colormap("#ffffff", "#fc69a0", levels=3)
    bar = Colorbar(cmap, Norm("linear", (0, 1), boundaries=[0, 0.5, 0.75, 1]))
    bar.place(BBox(100, 10, 109, 110), 9)
    bands = strip(bar)
    assert isinstance(bands, Group)
    assert [(r.y, r.height, r.fill) for r in bands.children] == [  # pyright: ignore[reportAttributeAccessIssue]
        (10, 50, cmap.entries()[0]),
        (60, 25, cmap.entries()[1]),
        (85, 25, cmap.entries()[2]),
    ]


def test_per_theme_colorbar_strip_splits_by_theme():
    cmap = Colormap.per_theme(light=Colormap("#ffffff", "#000000"), dark=Colormap("#000000", "#ffffff"))
    bar = Colorbar(cmap, Norm("linear", (0, 1)))
    bar.place(BBox(100, 10, 109, 110), 9)
    themed = strip(bar)
    assert isinstance(themed, Themed) and set(themed.children) == {"light", "dark"}


def test_colorbar_notes_go_through_text_past_an_end():
    bar = Colorbar(DEFAULT, Norm("log", (1e-9, 1e7)))
    bar.place(BBox(100, 10, 109, 110), 9)
    bar.text("$<= 10^(-9)$", (0.5, -0.05), "axes", anchor="middle", valign="top")
    # the note hangs from 5 pt below the bar, and counts toward the outer box
    assert bar.outer().y0 < 5
