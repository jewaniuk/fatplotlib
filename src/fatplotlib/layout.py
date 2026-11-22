"""layout: figures, targets, rows and columns of axes and panels, and panel labels.

a script builds axes and panels standalone, then places them once with a nested expression:
`fig.layout(Column(photons, comm, share="x"))`. nothing is solved until `Figure.scene` gets a
`Target`, which sets the figure width and the base font size em lengths scale, so one script
re-solves for a paper or a talk.

the solve uses kiwisolver (cassowary, as matplotlib's constrained layout). every item gets an
outer box; an axes's inner box (the plot rectangle) sits inside it by margins measured from its
typst text. rows and columns stack outer boxes with a gap, `ratios` split the inner sizes along
the stack, and `share` aligns the inner edges across it. the number of ticks depends on the
rectangle and the margins on the ticks, so the solve repeats until the margins settle.

panel labels never take space. each sits in the top-left corner of its panel's outer box, its
left edge on the box's left and its cap height on the box's top. then labels across a row share
the highest top and labels down a column the leftmost left edge.
"""

import warnings
from collections.abc import Iterator
from typing import Literal

import kiwisolver as kiwi

from fatplotlib.axes import Axes
from fatplotlib.panel import Panel
from fatplotlib.scene import BBox, Group, Node, Text
from fatplotlib.units import length

type Leaf = Axes | Panel
type Item = Leaf | Row | Column
type Margins = tuple[float, float, float, float]
"""left, bottom, right and top distance from an inner box out to its outer box, in pt."""

# the most solves while the margins settle, and the change in pt that counts as settled
_PASSES = 4
_SETTLED = 0.01


class Target:
    """Where a figure goes: page widths, base font size and whether panels get labels.

    Attributes:
        textwidth (float): the full text width in pt.
        columnwidth (float): one column's width in pt.
        base (float): the font size in pt that em lengths scale.
        labels (bool): whether panel labels are drawn.
    """

    def __init__(
        self,
        textwidth: float | str = "7in",
        columnwidth: float | str = "3.4in",
        base: float | str = "10pt",
        labels: bool = True,
    ) -> None:
        """Sets up the target.

        Args:
            textwidth (float | str): the full text width, pt or a length such as "7in".
            columnwidth (float | str): one column's width.
            base (float | str): the base font size, e.g. "9texpt".
            labels (bool): whether panel labels are drawn.
        """
        self.textwidth = length(textwidth)
        self.columnwidth = length(columnwidth)
        self.base = length(base)
        self.labels = labels

    def with_(self, **changes: float | str | bool) -> Target:
        """A copy with some fields changed, e.g. `paper.with_(base="14pt", labels=False)`.

        Args:
            **changes (float | str | bool): new values by field name.

        Returns:
            Target: the copy.
        """
        fields = {"textwidth": self.textwidth, "columnwidth": self.columnwidth, "base": self.base, "labels": self.labels}
        return Target(**(fields | changes))  # pyright: ignore[reportArgumentType]


class _Stack:
    """Children laid side by side along one direction.

    Attributes:
        children (tuple[Item, ...]): the items, left to right or top to bottom.
        ratios (tuple[float, ...]): the split of inner sizes along the stack.
        gap (float | str): space between outer boxes, pt or a length such as "1em".
        share (Literal["x", "y"] | None): the direction across the stack whose inner edges align.
    """

    across: Literal["x", "y"]

    def __init__(
        self,
        *children: Item,
        ratios: tuple[float, ...] | None = None,
        gap: float | str = "1em",
        share: Literal["x", "y"] | None = None,
    ) -> None:
        """Sets up the stack.

        Args:
            *children (Item): the items.
            ratios (tuple[float, ...] | None): the split of inner sizes, equal if None.
            gap (float | str): space between outer boxes.
            share (Literal["x", "y"] | None): align inner edges across the stack: "y" in a
                `Row`, "x" in a `Column`.

        Raises:
            ValueError: no children, ratios of the wrong length, or `share` along the stack.
        """
        if not children:
            raise ValueError(f"{type(self).__name__} needs at least one child")
        if ratios is not None and len(ratios) != len(children):
            raise ValueError(f"{len(ratios)} ratios for {len(children)} children")
        if share is not None and share != self.across:
            raise ValueError(f'{type(self).__name__} can only share "{self.across}"')
        self.children = children
        self.ratios = ratios or (1.0,) * len(children)
        self.gap = gap
        self.share = share


class Row(_Stack):
    """Children left to right; `share="y"` aligns their inner tops and bottoms."""

    across = "y"


class Column(_Stack):
    """Children top to bottom; `share="x"` aligns their inner left and right edges."""

    across = "x"


class Figure:
    """A page of panels and axes, solved per target.

    Attributes:
        width (float | str): "textwidth", "columnwidth" or a length.
        height (float | str): a length.
        pad (float | str): space between the page edge and the layout.
        root (Item | None): the layout, None until `layout` sets it.
    """

    def __init__(self, width: float | str = "columnwidth", height: float | str = "3in", pad: float | str = 0) -> None:
        """Sets up the figure.

        Args:
            width (float | str): "textwidth", "columnwidth" (of the target) or a length.
            height (float | str): a length.
            pad (float | str): space between the page edge and the layout.
        """
        self.width = width
        self.height = height
        self.pad = pad
        self.root: Item | None = None

    def layout(self, root: Item) -> None:
        """Places the axes and panels, e.g. `fig.layout(Row(s, Column(a, b, share="x")))`.

        Args:
            root (Item): the layout.
        """
        self.root = root

    def scene(self, target: Target) -> tuple[Group, float, float]:
        """Solves the layout for a target and builds the scene.

        Args:
            target (Target): the page widths, font size and labels.

        Returns:
            tuple[Group, float, float]: the scene, page width and page height in pt.

        Raises:
            RuntimeError: `layout` was never called.
        """
        if self.root is None:
            raise RuntimeError("the figure has no layout: call fig.layout(...)")
        em = target.base
        widths = {"textwidth": target.textwidth, "columnwidth": target.columnwidth}
        width = widths[self.width] if isinstance(self.width, str) and self.width in widths else length(self.width, em)
        height = length(self.height, em)
        pad = length(self.pad, em)
        _solve(self.root, BBox(pad, pad, width - pad, height - pad), em)
        root = Group([leaf.group() for leaf in _leaves(self.root)])
        if target.labels:
            root.add(*_labels(self.root, root, em))
        return root, width, height


def _leaves(item: Item) -> Iterator[Leaf]:
    """The axes and panels in a layout, depth first."""
    if isinstance(item, _Stack):
        for child in item.children:
            yield from _leaves(child)
    else:
        yield item


def _solve(root: Item, area: BBox, em: float) -> None:
    """Places every leaf, re-solving until the measured margins settle."""
    leaves = list(_leaves(root))
    for leaf in leaves:
        leaf.place(area, em)
    margins = {id(leaf): _margins(leaf) for leaf in leaves}
    for _ in range(_PASSES):
        inners = _constrain(root, area, em, margins)
        for leaf in leaves:
            leaf.place(inners[id(leaf)], em)
        measured = {id(leaf): _margins(leaf) for leaf in leaves}
        settled = all(abs(a - b) < _SETTLED for k in margins for a, b in zip(margins[k], measured[k]))
        margins = measured
        if settled:
            return


def _margins(leaf: Leaf) -> Margins:
    """How far the placed leaf's decorations reach past its inner box; panels have none."""
    if isinstance(leaf, Panel):
        return (0.0, 0.0, 0.0, 0.0)
    inner, outer = leaf.inner, leaf.outer()
    assert inner is not None  # placed by the caller
    return (
        max(0.0, inner.x0 - outer.x0),
        max(0.0, inner.y0 - outer.y0),
        max(0.0, outer.x1 - inner.x1),
        max(0.0, outer.y1 - inner.y1),
    )


class _Box:
    """Solver variables for one item's outer box, and its inner box as expressions.

    Attributes:
        x0 (kiwi.Variable): outer left.
        y0 (kiwi.Variable): outer bottom.
        x1 (kiwi.Variable): outer right.
        y1 (kiwi.Variable): outer top.
        inner (tuple[kiwi.Expression, ...]): inner left, bottom, right and top.
    """

    def __init__(self, margins: Margins) -> None:
        self.x0, self.y0, self.x1, self.y1 = (kiwi.Variable() for _ in range(4))
        left, bottom, right, top = margins
        self.inner = (self.x0 + left, self.y0 + bottom, self.x1 - right, self.y1 - top)


def _constrain(root: Item, area: BBox, em: float, margins: dict[int, Margins]) -> dict[int, BBox]:
    """Solves the layout once for fixed margins.

    Returns:
        dict[int, BBox]: each leaf's inner box, keyed by `id(leaf)`.
    """
    solver = kiwi.Solver()
    boxes: dict[int, _Box] = {}

    def visit(item: Item) -> _Box:
        """Adds the item's constraints and returns its box."""
        box = boxes[id(item)] = _Box(margins.get(id(item), (0.0, 0.0, 0.0, 0.0)))
        x0, y0, x1, y1 = box.inner
        solver.addConstraint(x1 >= x0)
        solver.addConstraint(y1 >= y0)
        if not isinstance(item, _Stack):
            return box
        kids = [visit(child) for child in item.children]
        gap = length(item.gap, em)
        row = isinstance(item, Row)
        # along the stack: outer boxes edge to edge with gaps, filling the stack
        if row:
            starts, ends = [k.x0 for k in kids], [k.x1 for k in kids]
            solver.addConstraint(starts[0] == box.x0)
            solver.addConstraint(ends[-1] == box.x1)
        else:
            starts, ends = [k.y1 for k in kids], [k.y0 for k in kids]
            solver.addConstraint(starts[0] == box.y1)
            solver.addConstraint(ends[-1] == box.y0)
        for end, start in zip(ends, starts[1:]):
            solver.addConstraint((start == end + gap) if row else (start == end - gap))
        # inner sizes along the stack in the given ratios
        sizes = [k.inner[2] - k.inner[0] if row else k.inner[3] - k.inner[1] for k in kids]
        for size, ratio in zip(sizes[1:], item.ratios[1:]):
            solver.addConstraint((size * item.ratios[0] == sizes[0] * ratio) | "strong")
        # across the stack: inside the stack, filling it where alignment allows
        lo, hi = (1, 3) if row else (0, 2)
        outer_lo, outer_hi = (box.y0, box.y1) if row else (box.x0, box.x1)
        for k in kids:
            k_lo, k_hi = (k.y0, k.y1) if row else (k.x0, k.x1)
            solver.addConstraint(k_lo >= outer_lo)
            solver.addConstraint(k_hi <= outer_hi)
            solver.addConstraint((k_lo == outer_lo) | "medium")
            solver.addConstraint((k_hi == outer_hi) | "medium")
        if item.share is not None:
            for k in kids[1:]:
                solver.addConstraint((k.inner[lo] == kids[0].inner[lo]) | "strong")
                solver.addConstraint((k.inner[hi] == kids[0].inner[hi]) | "strong")
        return box

    top = visit(root)
    for var, value in ((top.x0, area.x0), (top.y0, area.y0), (top.x1, area.x1), (top.y1, area.y1)):
        solver.addConstraint(var == value)
    solver.updateVariables()
    return {
        id(leaf): BBox(*(e.value() for e in boxes[id(leaf)].inner))
        for leaf in _leaves(root)
    }


def _labels(root: Item, scene: Group, em: float) -> list[Text]:
    """Panel labels for the placed leaves, lettered in reading order and aligned."""
    labelled = [leaf for leaf in _leaves(root) if leaf.label is not False]
    outers = {id(leaf): leaf.outer() for leaf in labelled}
    # reading order: rows top to bottom (a leaf joins a row if its top is above the row's first
    # leaf's middle), then left to right
    order: list[Leaf] = []
    rows: list[list[Leaf]] = []
    for leaf in sorted(labelled, key=lambda leaf: -outers[id(leaf)].y1):
        first = outers[id(rows[-1][0])] if rows else None
        if first is not None and outers[id(leaf)].y1 > (first.y0 + first.y1) / 2:
            rows[-1].append(leaf)
        else:
            rows.append([leaf])
    for row in rows:
        order += sorted(row, key=lambda leaf: outers[id(leaf)].x0)

    corners = {id(leaf): [outers[id(leaf)].x0, outers[id(leaf)].y1] for leaf in labelled}
    _align(root, corners)
    texts = []
    for i, leaf in enumerate(order):
        letter = leaf.label if isinstance(leaf.label, str) else chr(ord("a") + i)
        x, y = corners[id(leaf)]
        texts.append(Text(f"*{letter}*", x, y, em, "start", "cap", class_="fpl-label"))
    _check_overlaps(texts, scene, labelled)
    return texts


def _align(item: Item, corners: dict[int, list[float]]) -> None:
    """Gives labels across each row the highest top and down each column the leftmost left."""
    if not isinstance(item, _Stack):
        return
    row = isinstance(item, Row)
    edge = [leaf for child in item.children for leaf in _edge(child, row) if id(leaf) in corners]
    if edge:
        i, pick = (1, max) if row else (0, min)
        value = pick(corners[id(leaf)][i] for leaf in edge)
        for leaf in edge:
            corners[id(leaf)][i] = value
    for child in item.children:
        _align(child, corners)


def _edge(item: Item, top: bool) -> list[Leaf]:
    """The leaves along an item's top edge (top=True) or left edge."""
    if not isinstance(item, _Stack):
        return [item]
    # a stack along the edge contributes all its children, one across it only its first
    along = isinstance(item, Row) if top else isinstance(item, Column)
    children = item.children if along else item.children[:1]
    return [leaf for child in children for leaf in _edge(child, top)]


def _check_overlaps(labels: list[Text], scene: Group, leaves: list[Leaf]) -> None:
    """Warns when a label's box overlaps text or a plot rectangle."""
    boxes = list(_text_boxes(scene)) + [leaf.inner for leaf in leaves if isinstance(leaf, Axes) and leaf.inner]
    for label in labels:
        box = label.bbox()
        for other in boxes:
            hit = box.intersection(other)
            if hit is not None and hit.width > 0 and hit.height > 0:
                warnings.warn(f"panel label {label.markup} overlaps other content", stacklevel=3)
                break


def _text_boxes(node: Node) -> Iterator[BBox]:
    """The boxes of every text in a scene; a transformed group holding text counts as one box."""
    if isinstance(node, Text):
        yield node.bbox()
    elif isinstance(node, Group):
        if node.transform is not None:
            if any(True for _ in _text_boxes(Group(node.children))):
                box = node.bbox()
                if box is not None:
                    yield box
            return
        for child in node.children:
            yield from _text_boxes(child)
