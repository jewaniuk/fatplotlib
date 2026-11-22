"""layout: solving rows and columns, panels, targets and panel labels."""

import warnings

import pytest

from fatplotlib.axes import Axes, Colorbar
from fatplotlib.colormap import DEFAULT, Norm
from fatplotlib.layout import Column, Figure, Row, Target
from fatplotlib.panel import Frame, Panel
from fatplotlib.scene import BBox, Group, Rect, Text

PAPER = Target(columnwidth="3.4in", base="9texpt")


def plot(**kwargs: object) -> Axes:
    """A log-linear axes with labels."""
    return Axes(xlim=(0, 4000), ylim=(1, 1000), yscale="log", ylabel="Rate", **kwargs)  # pyright: ignore[reportArgumentType]


def labels(root: Group) -> list[Text]:
    """The panel labels in a built scene."""
    return [n for n in root.children if isinstance(n, Text) and n.class_ == "fpl-label"]


def test_column_shares_inner_edges_and_splits_inner_heights():
    top, bottom = plot(), plot(xlabel="Length")
    top.twin_x(label="Nodes")
    fig = Figure(height="3.4in")
    fig.layout(Column(top, bottom, share="x"))
    fig.scene(PAPER)
    a, b = top.inner, bottom.inner
    assert a is not None and b is not None
    assert a.x0 == pytest.approx(b.x0) and a.x1 == pytest.approx(b.x1)
    assert a.height == pytest.approx(b.height)
    # outer boxes fill the page inside the default 1pt pad, with the default 1em gap between them
    assert top.outer().y1 == pytest.approx(3.4 * 72 - 1, abs=1e-6)
    assert bottom.outer().y0 == pytest.approx(1, abs=1e-6)
    assert top.outer().y0 - bottom.outer().y1 == pytest.approx(PAPER.base)


def test_row_ratios_split_inner_widths():
    left, right = plot(), plot()
    fig = Figure("textwidth")
    fig.layout(Row(left, right, ratios=(2, 1), share="y"))
    fig.scene(PAPER)
    assert left.inner is not None and right.inner is not None
    assert left.inner.width == pytest.approx(2 * right.inner.width)


def test_share_must_run_across_the_stack():
    with pytest.raises(ValueError, match='"x"'):
        Column(plot(), plot(), share="y")


def test_target_re_solves_with_bigger_text():
    ax = plot(xlabel="Length")
    fig = Figure(height="2in")
    fig.layout(ax)
    fig.scene(PAPER)
    paper = ax.inner
    fig.scene(PAPER.with_(base="14pt"))
    assert paper is not None and ax.inner is not None
    assert ax.inner.x0 > paper.x0 and ax.inner.y0 > paper.y0


def test_panel_fits_and_centres_with_fixed_sizes():
    seen: list[Frame] = []

    def draw(f: Frame) -> list[Rect]:
        seen.append(f)
        return [Rect(*f.pt(0, 0), f.length(100), f.length("1em"))]

    panel = Panel(2, draw)
    panel.place(BBox(0, 0, 100, 100), 9)
    assert panel.filled == 0.5
    with pytest.warns(UserWarning, match="fills 50%"):
        panel.warn_gap()
    assert panel.outer() == BBox(0, 25, 100, 75)
    rect = panel.group().children[0]
    assert isinstance(rect, Rect) and (rect.y, rect.width, rect.height) == (25, 100, 9)


def test_labels_letter_in_reading_order_and_share_a_left_edge():
    top, bottom = plot(), plot(xlabel="Length")
    top.twin_x(label="Nodes", labels=["$10^(5)$"], ticks=[2000])
    fig = Figure(height="3.4in")
    fig.layout(Column(top, bottom, share="x"))
    a, b = labels(fig.scene(PAPER)[0])
    assert (a.markup, b.markup) == ("*a*", "*b*")
    assert a.x == b.x == min(top.outer().x0, bottom.outer().x0)
    assert a.y == pytest.approx(top.outer().y1) and a.valign == "cap"


def test_labels_skip_override_and_switch_off():
    first, second, third = plot(label=False), plot(label="z"), plot()
    fig = Figure("textwidth")
    fig.layout(Row(first, second, third))
    assert [t.markup for t in labels(fig.scene(PAPER)[0])] == ["*z*", "*b*"]
    assert labels(fig.scene(PAPER.with_(labels=False))[0]) == []


def test_schematic_label_anchors_on_drawn_content():
    def draw(f: Frame) -> list[Rect]:
        return [Rect(*f.pt(20, 10), f.length(50), f.length(20))]

    panel = Panel(2, draw)
    fig = Figure("4in", height="2in", pad=0)
    fig.layout(panel)
    (a,) = labels(fig.scene(PAPER)[0])
    drawn = panel.group().bbox()
    assert drawn is not None and drawn != panel.outer()
    assert (a.x, a.y) == (drawn.x0, drawn.y1)


def test_gap_warns_once_from_the_solved_slot():
    def draw(f: Frame) -> list[Rect]:
        return [Rect(*f.pt(0, 0), f.length(100), f.length(100))]

    # the seed placement into the 2:1 page leaves half of it empty, the solved square slots do not
    fig = Figure("4in", height="2in", pad=0)
    fig.layout(Row(Panel(1, draw), Panel(1, draw), gap=0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig.scene(PAPER)
    lone = Figure("4in", height="2in", pad=0)
    lone.layout(Panel(1, draw))
    with pytest.warns(UserWarning, match="fills 50%") as caught:
        lone.scene(PAPER)
    assert len(caught) == 1


def test_colorbar_spans_a_column_with_a_fixed_width_and_no_letter():
    a, b, c = plot(), plot(), plot(xlabel="Length")
    bar = Colorbar(DEFAULT, Norm("log", (1, 1000)), label="Rate")
    fig = Figure("textwidth", height="4in")
    fig.layout(Row(Column(a, b, c, share="x"), bar, share="y"))
    root = fig.scene(PAPER)[0]
    assert a.inner is not None and c.inner is not None and bar.inner is not None
    assert bar.inner.y1 == pytest.approx(a.inner.y1) and bar.inner.y0 == pytest.approx(c.inner.y0)
    assert bar.inner.width == pytest.approx(PAPER.base)
    assert [t.markup for t in labels(root)] == ["*a*", "*b*", "*c*"]


def test_fixed_width_is_kept_and_left_out_of_ratios():
    left, middle, right = plot(), plot(), plot(width="1in")
    fig = Figure("textwidth")
    fig.layout(Row(left, middle, right, share="y"))
    fig.scene(PAPER)
    assert left.inner is not None and middle.inner is not None and right.inner is not None
    assert right.inner.width == pytest.approx(72)
    assert left.inner.width == pytest.approx(middle.inner.width) and left.inner.width > 72
