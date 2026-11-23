"""layout: figures, targets, rows, columns and grids of axes and panels, and panel labels.

a script builds axes and panels standalone, then places them once with a nested expression:
`fig.layout(Column(photons, comm, share="x"))`. nothing is solved until `Figure.scene` gets a
`Target`, which sets the figure width and the base font size em lengths scale, so one script
re-solves for a paper or a talk.

the solve uses kiwisolver (cassowary, as matplotlib's constrained layout). every item gets an
outer box; an axes's inner box (the plot rectangle) sits inside it by margins measured from its
typst text, and keeps a fixed `width` or `height` if the axes has one. a panel's inner box is its
outer box. `gap` is the distance between the inner boxes of neighbours, so plots sit `gap` apart
whatever text hangs off them; their outer boxes must also stay `_CLEARANCE` apart, so decorations
only push plots further apart where they would collide. an axis label longer than its plot is
left out of the outer box along its axis and may spill into the gap, with a warning if it hits
other text.

rows and columns stack their children, `ratios` split the inner sizes of the children without a
fixed size, and `share` aligns the inner edges across the stack. a stack's own inner box runs from
its first child's inner edge to its last child's along it, so `Row(Column(a, b, share="x"), bar,
share="y")` spans the bar from a's top to b's bottom. across a sharing stack it is the shared
inner edges, so `Column(a, Row(b, bar, share="y"), share="x")` splits a's and b's heights by
the ratios, not a's height against the row's outer box.

a `Grid` is filled by slicing, as matplotlib's gridspec: `g[0, 0] = a`, `g[0:2, 1] = key`. every
item's inner edges lie on the edges of the rows and columns it spans, so a bar in one column
lines up with a plot in another, and `heights` / `widths` split the rows and columns without a
fixed-size item. the number of ticks depends on the rectangle and the margins on the ticks, so
the solve repeats until the margins settle.

panel labels never take space, and colorbars get none. each sits in the top-left corner of its
item's outer box (an axes's decorations, a panel's drawn content), its left edge on the box's left
and its cap height on the box's top. then labels across a row (of a stack or a grid) share the
highest top and labels down a column the leftmost left edge.
"""

import warnings
from collections.abc import Iterator
from itertools import pairwise
from typing import Literal

import kiwisolver as kiwi

from fatplotlib.axes import Axes
from fatplotlib.panel import Panel
from fatplotlib.scene import BBox, Group, Node, Text
from fatplotlib.units import length

type Leaf = Axes | Panel
type Item = Leaf | Row | Column | Grid
type Span = int | slice
type Margins = tuple[float, float, float, float]
"""left, bottom, right and top distance from an inner box out to its outer box, in pt."""

# the most solves while the margins settle, and the change in pt that counts as settled
_PASSES = 4
_SETTLED = 0.01
# the least space between the outer boxes of neighbours
_CLEARANCE = "0.25em"


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
        gap (float | str): space between the children's inner boxes, pt or a length such as "1em".
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
            gap (float | str): space between the children's inner boxes.
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


class Grid:
    """Items in rows and columns, filled by slicing: `g[0, 0] = a`, `g[0:2, 1] = key`.

    Attributes:
        rows (int): the number of rows, top to bottom.
        cols (int): the number of columns, left to right.
        heights (tuple[float, ...]): the split of inner heights over the rows without a
            fixed-height item; the entries of fixed rows are ignored.
        widths (tuple[float, ...]): the split of inner widths over the columns without a
            fixed-width item.
        gap (float | str): space between the inner boxes of neighbouring rows and columns.
        cells (list[tuple[Item, range, range]]): each item with the rows and columns it spans,
            in the order they were set.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        heights: tuple[float, ...] | None = None,
        widths: tuple[float, ...] | None = None,
        gap: float | str = "1em",
    ) -> None:
        """Sets up an empty grid.

        Args:
            rows (int): the number of rows.
            cols (int): the number of columns.
            heights (tuple[float, ...] | None): the split of inner heights, equal if None.
            widths (tuple[float, ...] | None): the split of inner widths, equal if None.
            gap (float | str): space between inner boxes, pt or a length such as "1em".

        Raises:
            ValueError: heights or widths of the wrong length.
        """
        if heights is not None and len(heights) != rows:
            raise ValueError(f"{len(heights)} heights for {rows} rows")
        if widths is not None and len(widths) != cols:
            raise ValueError(f"{len(widths)} widths for {cols} columns")
        self.rows = rows
        self.cols = cols
        self.heights = heights or (1.0,) * rows
        self.widths = widths or (1.0,) * cols
        self.gap = gap
        self.cells: list[tuple[Item, range, range]] = []

    def __setitem__(self, at: tuple[Span, Span], item: Item) -> None:
        """Puts an item in the rows and columns it spans.

        Args:
            at (tuple[Span, Span]): a row and a column, each a non-negative index or a slice.
            item (Item): the item.

        Raises:
            IndexError: a span empty or past the grid.
        """
        spans = []
        for i, n in zip(at, (self.rows, self.cols)):
            start, stop = (i.start or 0, n if i.stop is None else i.stop) if isinstance(i, slice) else (i, i + 1)
            if not 0 <= start < stop <= n or isinstance(i, slice) and i.step not in (None, 1):
                raise IndexError(f"{at} is not a block of cells in a {self.rows} x {self.cols} grid")
            spans.append(range(start, stop))
        self.cells.append((item, *spans))

    @property
    def children(self) -> list[Item]:
        """The items, in the order they were set."""
        return [item for item, _, _ in self.cells]


class Figure:
    """A page of panels and axes, solved per target.

    Attributes:
        width (float | str): "textwidth", "columnwidth" or a length.
        height (float | str): a length.
        pad (float | str): space between the page edge and the layout, enough by default that
            antialiased edges and text are not cut off.
        root (Item | None): the layout, None until `layout` sets it.
    """

    def __init__(self, width: float | str = "columnwidth", height: float | str = "3in", pad: float | str = "1pt") -> None:
        """Sets up the figure.

        Args:
            width (float | str): "textwidth", "columnwidth" (of the target) or a length.
            height (float | str): a length.
            pad (float | str): space between the page edge and the layout, 1pt by default.
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
        groups = {id(leaf): leaf.group() for leaf in _leaves(self.root)}
        root = Group(list(groups.values()))
        _check_spills([leaf for leaf in _leaves(self.root) if isinstance(leaf, Axes)], groups)
        if target.labels:
            root.add(*_labels(self.root, groups, em))
        return root, width, height


def _leaves(item: Item) -> Iterator[Leaf]:
    """The axes and panels in a layout, depth first."""
    if isinstance(item, _Stack | Grid):
        for child in item.children:
            yield from _leaves(child)
    else:
        yield item


def _solve(root: Item, area: BBox, em: float) -> None:
    """Places every leaf, re-solving until the measured margins settle, then warns about panels
    that leave much of their final slot empty."""
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
            break
    for leaf in leaves:
        if isinstance(leaf, Panel):
            leaf.warn_gap()


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
        inner (tuple[kiwi.Expression | kiwi.Variable, ...]): inner left, bottom, right and top, from the margins;
            a stack replaces it with its children's edges once they are visited.
    """

    def __init__(self, margins: Margins) -> None:
        self.x0, self.y0, self.x1, self.y1 = (kiwi.Variable() for _ in range(4))
        left, bottom, right, top = margins
        self.inner: tuple[kiwi.Expression | kiwi.Variable, ...] = (self.x0 + left, self.y0 + bottom, self.x1 - right, self.y1 - top)


def _constrain(root: Item, area: BBox, em: float, margins: dict[int, Margins]) -> dict[int, BBox]:
    """Solves the layout once for fixed margins.

    Returns:
        dict[int, BBox]: each leaf's inner box, keyed by `id(leaf)`.
    """
    solver = kiwi.Solver()
    boxes: dict[int, _Box] = {}
    clearance = length(_CLEARANCE, em)

    def visit(item: Item) -> _Box:
        """Adds the item's constraints and returns its box."""
        box = boxes[id(item)] = _Box(margins.get(id(item), (0.0, 0.0, 0.0, 0.0)))
        x0, y0, x1, y1 = box.inner
        solver.addConstraint(x1 >= x0)
        solver.addConstraint(y1 >= y0)
        if isinstance(item, _Stack):
            constraints = _stack(item, box, [visit(child) for child in item.children], length(item.gap, em), clearance)
        elif isinstance(item, Grid):
            constraints = _grid(item, box, [visit(child) for child in item.children], length(item.gap, em), clearance)
        else:
            constraints = _fixed_sizes(item, box, em)
        for constraint in constraints:
            solver.addConstraint(constraint)
        return box

    top = visit(root)
    for var, value in ((top.x0, area.x0), (top.y0, area.y0), (top.x1, area.x1), (top.y1, area.y1)):
        solver.addConstraint(var == value)
    solver.updateVariables()
    return {
        id(leaf): BBox(*(e.value() for e in boxes[id(leaf)].inner))
        for leaf in _leaves(root)
    }


def _stack(item: _Stack, box: _Box, kids: list[_Box], gap: float, clearance: float) -> list[kiwi.Constraint]:
    """A stack's constraints, and its inner box set from its children's."""
    row = isinstance(item, Row)
    # the stack's inner box: first child's inner start to last child's inner end along it, and
    # across it the shared inner edges, or the outer box if the children are not aligned
    first, last = kids[0].inner, kids[-1].inner
    if row:
        bottom, top = (first[1], first[3]) if item.share else (box.y0, box.y1)
        box.inner = (first[0], bottom, last[2], top)
    else:
        left, right = (first[0], first[2]) if item.share else (box.x0, box.x1)
        box.inner = (left, last[1], right, first[3])
    # along the stack: the children fill it, their inner boxes a gap apart
    constraints = (
        [kids[0].x0 == box.x0, kids[-1].x1 == box.x1] if row else [kids[0].y1 == box.y1, kids[-1].y0 == box.y0]
    )
    for before, after in pairwise(kids):
        constraints += _gap(before, after, row, gap, clearance)
    constraints += _ratios(item, kids)
    # across the stack: inside the stack, filling it where alignment allows
    lo, hi = (1, 3) if row else (0, 2)
    for k in kids:
        constraints += _inside(k, box, edges=(not row, row, not row, row))
    if item.share is not None:
        for k in kids[1:]:
            constraints.append((k.inner[lo] == kids[0].inner[lo]) | "strong")
            constraints.append((k.inner[hi] == kids[0].inner[hi]) | "strong")
    return constraints


def _grid(item: Grid, box: _Box, kids: list[_Box], gap: float, clearance: float) -> list[kiwi.Constraint]:  # noqa: C901 - one block per grid rule
    """A grid's constraints, and its inner box set from its rows' and columns' edges."""
    # inner bottom and top of each row, left and right of each column
    rows = [(kiwi.Variable(), kiwi.Variable()) for _ in range(item.rows)]
    cols = [(kiwi.Variable(), kiwi.Variable()) for _ in range(item.cols)]
    box.inner = (cols[0][0], rows[-1][0], cols[-1][1], rows[0][1])
    constraints: list[kiwi.Constraint] = []
    for (bottom, _), (_, lower_top) in pairwise(rows):
        constraints.append((lower_top == bottom - gap) | "strong")
    for (_, right), (next_left, _) in pairwise(cols):
        constraints.append((next_left == right + gap) | "strong")
    for bottom, top in rows:
        constraints.append(top >= bottom)
    for left, right in cols:
        constraints.append(right >= left)
    for k, (child, rs, cs) in zip(kids, item.cells):
        # inner edges on the spanned rows and columns, the outer box inside the grid
        edges = (cols[cs[0]][0], rows[rs[-1]][0], cols[cs[-1]][1], rows[rs[0]][1])
        constraints += [(inner == edge) | "strong" for inner, edge in zip(k.inner, edges)]
        constraints += _inside(k, box, edges=(cs[0] == 0, rs[-1] == item.rows - 1, cs[-1] == item.cols - 1, rs[0] == 0))
    # neighbours in adjacent rows (columns) that overlap in columns (rows) keep their outer boxes apart
    for k, (_, rs, cs) in zip(kids, item.cells):
        for other, (_, other_rs, other_cs) in zip(kids, item.cells):
            if other_rs[0] == rs[-1] + 1 and set(cs) & set(other_cs):
                constraints.append(other.y1 <= k.y0 - clearance)
            if other_cs[0] == cs[-1] + 1 and set(rs) & set(other_rs):
                constraints.append(other.x0 >= k.x1 + clearance)
    # ratios among the rows and columns no single-cell item fixes the size of
    for row, sizes, ratios in ((False, rows, item.heights), (True, cols, item.widths)):
        fixed = {
            (cs if row else rs)[0]
            for child, rs, cs in item.cells
            if len(cs if row else rs) == 1 and _fixed(child, row) is not None
        }
        free = [(end - start, ratio) for i, ((start, end), ratio) in enumerate(zip(sizes, ratios)) if i not in fixed]
        constraints += [(size * free[0][1] == free[0][0] * ratio) | "strong" for size, ratio in free[1:]]
    return constraints


def _gap(before: _Box, after: _Box, row: bool, gap: float, clearance: float) -> list[kiwi.Constraint]:
    """Neighbours along a row (or down a column): inner boxes a gap apart, strong, and outer
    boxes at least the clearance apart, required."""
    if row:
        return [(after.inner[0] == before.inner[2] + gap) | "strong", after.x0 >= before.x1 + clearance]
    return [(after.inner[3] == before.inner[1] - gap) | "strong", after.y1 <= before.y0 - clearance]


def _inside(k: _Box, box: _Box, edges: tuple[bool, bool, bool, bool]) -> list[kiwi.Constraint]:
    """A child's outer box inside its parent's, and on the parent's left, bottom, right and top
    edges where `edges` says, medium so alignment wins."""
    constraints = [k.x0 >= box.x0, k.y0 >= box.y0, k.x1 <= box.x1, k.y1 <= box.y1]
    pairs = ((k.x0, box.x0), (k.y0, box.y0), (k.x1, box.x1), (k.y1, box.y1))
    constraints += [(mine == theirs) | "medium" for (mine, theirs), on in zip(pairs, edges) if on]
    return constraints


def _fixed(item: Item, row: bool) -> float | str | None:
    """An item's fixed inner width (along a row) or height (along a column), None if free."""
    if not isinstance(item, Axes):
        return None
    return item.width if row else item.height


def _ratios(stack: _Stack, kids: list[_Box]) -> list[kiwi.Constraint]:
    """Strong constraints splitting the inner sizes along a stack in its ratios, among the
    children without a fixed size."""
    row = isinstance(stack, Row)
    sized = [
        (k.inner[2] - k.inner[0] if row else k.inner[3] - k.inner[1], ratio)
        for k, ratio, child in zip(kids, stack.ratios, stack.children)
        if _fixed(child, row) is None
    ]
    return [(size * sized[0][1] == sized[0][0] * ratio) | "strong" for size, ratio in sized[1:]]


def _fixed_sizes(leaf: Leaf, box: _Box, em: float) -> list[kiwi.Constraint]:
    """Required constraints holding a leaf's inner box to its fixed width and height."""
    x0, y0, x1, y1 = box.inner
    sizes = ((_fixed(leaf, True), x1 - x0), (_fixed(leaf, False), y1 - y0))
    return [size == length(fixed, em) for fixed, size in sizes if fixed is not None]


def _labels(root: Item, groups: dict[int, Group], em: float) -> list[Text]:
    """Panel labels for the placed leaves, lettered in reading order and aligned."""
    labelled = [leaf for leaf in _leaves(root) if leaf.label is not False]
    outers = {id(leaf): _outer(leaf, groups[id(leaf)]) for leaf in labelled}
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
    _check_overlaps(texts, Group(list(groups.values())), labelled)
    return texts


def _outer(leaf: Leaf, group: Group) -> BBox:
    """The box a leaf's label anchors to: an axes's outer box, or the bbox of a panel's drawn
    content, so a schematic's label sits on its ink rather than on the fitted box."""
    if isinstance(leaf, Panel):
        drawn = group.bbox()
        if drawn is not None:
            return drawn
    return leaf.outer()


def _align(item: Item, corners: dict[int, list[float]]) -> None:
    """Gives labels across each row the highest top and down each column the leftmost left."""
    if isinstance(item, Grid):
        for i in range(item.rows):
            _share(
                [leaf for child, rs, _ in item.cells if rs[0] == i for leaf in _edge(child, True)], corners, top=True
            )
        for j in range(item.cols):
            _share(
                [leaf for child, _, cs in item.cells if cs[0] == j for leaf in _edge(child, False)], corners, top=False
            )
    elif isinstance(item, _Stack):
        row = isinstance(item, Row)
        _share([leaf for child in item.children for leaf in _edge(child, row)], corners, top=row)
    else:
        return
    for child in item.children:
        _align(child, corners)


def _share(leaves: list[Leaf], corners: dict[int, list[float]], top: bool) -> None:
    """Gives the labelled leaves the highest top (top=True) or the leftmost left."""
    edge = [leaf for leaf in leaves if id(leaf) in corners]
    if edge:
        i, pick = (1, max) if top else (0, min)
        value = pick(corners[id(leaf)][i] for leaf in edge)
        for leaf in edge:
            corners[id(leaf)][i] = value


def _edge(item: Item, top: bool) -> list[Leaf]:
    """The leaves along an item's top edge (top=True) or left edge."""
    if isinstance(item, Grid):
        return [leaf for child, rs, cs in item.cells if (rs if top else cs)[0] == 0 for leaf in _edge(child, top)]
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


def _check_spills(axes: list[Axes], groups: dict[int, Group]) -> None:
    """Warns when an axis label spilling past its plot overlaps another leaf's text."""
    for leaf in axes:
        others = [box for key, group in groups.items() if key != id(leaf) for box in _text_boxes(group)]
        for spill in leaf.spills():
            for other in others:
                hit = spill.intersection(other)
                if hit is not None and hit.width > 0 and hit.height > 0:
                    warnings.warn("an axis label longer than its plot overlaps other text", stacklevel=3)
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
