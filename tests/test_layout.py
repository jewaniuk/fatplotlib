"""layout: solving rows and columns, panels, targets and panel labels."""

import pytest

from fatplotlib.axes import Axes
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
    with pytest.warns(UserWarning, match="fills 50%"):
        panel.place(BBox(0, 0, 100, 100), 9)
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
