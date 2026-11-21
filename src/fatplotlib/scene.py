"""scene nodes: a y-up tree of shapes in pt, which renderers walk.

every node takes an optional `id` and `class_` (the svg `class`), so css and animation can
address it, and has `bbox()` in its parent's coordinates, stroke included.
"""

import math
from dataclasses import dataclass, field
from typing import Literal

from fatplotlib import text
from fatplotlib.style import Paint, Stroke
from fatplotlib.transform import Affine

type Command = (
    tuple[Literal["M", "L"], float, float]
    | tuple[Literal["C"], float, float, float, float, float, float]
    | tuple[Literal["Z"]]
)


@dataclass(frozen=True, slots=True)
class BBox:
    """An axis-aligned box.

    Attributes:
        x0 (float): left.
        y0 (float): bottom.
        x1 (float): right.
        y1 (float): top.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """float: x1 - x0."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """float: y1 - y0."""
        return self.y1 - self.y0

    def union(self, other: BBox) -> BBox:
        """The smallest box holding both boxes.

        Args:
            other (BBox): the other box.

        Returns:
            BBox: the union.
        """
        return BBox(
            min(self.x0, other.x0), min(self.y0, other.y0), max(self.x1, other.x1), max(self.y1, other.y1)
        )

    def padded(self, pad: float) -> BBox:
        """The box grown by `pad` on every side.

        Args:
            pad (float): the margin.

        Returns:
            BBox: the grown box.
        """
        return BBox(self.x0 - pad, self.y0 - pad, self.x1 + pad, self.y1 + pad)

    def transformed(self, transform: Affine) -> BBox:
        """The box holding this box's corners after a transform.

        Args:
            transform (Affine): the transform.

        Returns:
            BBox: the transformed box.
        """
        return _points_bbox(
            [transform.apply(x, y) for x in (self.x0, self.x1) for y in (self.y0, self.y1)]
        )


def _points_bbox(points: list[tuple[float, float]]) -> BBox:
    """The box holding some points."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _cubic_extrema(p0: float, p1: float, p2: float, p3: float) -> list[float]:
    """Parameters t in (0, 1) where one coordinate of a cubic bezier has zero derivative."""
    # the derivative / 3 is a t^2 + b t + c
    a = -p0 + 3 * p1 - 3 * p2 + p3
    b = 2 * (p0 - 2 * p1 + p2)
    c = p1 - p0
    if abs(a) < 1e-12:
        roots = [] if abs(b) < 1e-12 else [-c / b]
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return []
        sq = math.sqrt(disc)
        roots = [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]
    return [t for t in roots if 0 < t < 1]


def _cubic_at(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
    """One coordinate of a cubic bezier at parameter t."""
    s = 1 - t
    return s**3 * p0 + 3 * s * s * t * p1 + 3 * s * t * t * p2 + t**3 * p3


@dataclass(slots=True, kw_only=True)
class Node:
    """Fields every node shares.

    Attributes:
        id (str | None): the svg id.
        class_ (str | None): the svg class.
        alpha (float): opacity of the whole node, 0 to 1.
    """

    id: str | None = None
    class_: str | None = None
    alpha: float = 1.0

    def bbox(self) -> BBox | None:
        """The node's extent in its parent's coordinates, stroke included.

        Returns:
            BBox | None: the box, None if the node draws nothing.
        """
        raise NotImplementedError


@dataclass(slots=True, kw_only=True)
class Shape(Node):
    """A node with a fill and an outline.

    Attributes:
        fill (Paint | None): the fill paint, no fill if None.
        stroke (Stroke | None): the outline, none if None.
    """

    fill: Paint | None = None
    stroke: Stroke | None = None

    def _pad(self) -> float:
        """Half the stroke width, the distance the outline reaches past the geometry."""
        return 0.0 if self.stroke is None else self.stroke.width / 2


@dataclass(slots=True)
class Group(Node):
    """A list of nodes under one transform.

    Attributes:
        children (list[Node]): the nodes, drawn in order.
        transform (Affine | None): maps child coordinates to the parent's.
    """

    children: list[Node] = field(default_factory=list)
    transform: Affine | None = None

    def add(self, *nodes: Node) -> None:
        """Appends nodes, drawn above the existing ones.

        Args:
            *nodes (Node): the nodes.
        """
        self.children.extend(nodes)

    def bbox(self) -> BBox | None:
        """The union of the children's boxes in the parent's coordinates.

        Returns:
            BBox | None: the box, None if no child draws anything.
        """
        boxes = [b for b in (child.bbox() for child in self.children) if b is not None]
        if not boxes:
            return None
        box = boxes[0]
        for other in boxes[1:]:
            box = box.union(other)
        return box if self.transform is None else box.transformed(self.transform)


@dataclass(slots=True)
class Path(Shape):
    """An svg path; build one with `paths.PathBuilder`.

    Attributes:
        commands (tuple[Command, ...]): absolute M, L, C and Z commands.
    """

    commands: tuple[Command, ...]

    def bbox(self) -> BBox | None:
        """The box around the curve itself (cubic extrema, not control points), stroke included.

        Returns:
            BBox | None: the box, None if the path is empty.
        """
        points: list[tuple[float, float]] = []
        current = start = (0.0, 0.0)
        for command in self.commands:
            match command:
                case ("M", x, y):
                    current = start = (x, y)
                    points.append(current)
                case ("L", x, y):
                    current = (x, y)
                    points.append(current)
                case ("C", x1, y1, x2, y2, x, y):
                    (x0, y0) = current
                    for t in _cubic_extrema(x0, x1, x2, x) + _cubic_extrema(y0, y1, y2, y):
                        points.append((_cubic_at(t, x0, x1, x2, x), _cubic_at(t, y0, y1, y2, y)))
                    current = (x, y)
                    points.append(current)
                case ("Z",):
                    current = start
        return _points_bbox(points).padded(self._pad()) if points else None


@dataclass(slots=True)
class Rect(Shape):
    """A rectangle, optionally with rounded corners.

    Attributes:
        x (float): left.
        y (float): bottom.
        width (float): width.
        height (float): height.
        radius (float): corner radius.
    """

    x: float
    y: float
    width: float
    height: float
    radius: float = 0.0

    def bbox(self) -> BBox:
        """The rectangle, stroke included.

        Returns:
            BBox: the box.
        """
        return BBox(self.x, self.y, self.x + self.width, self.y + self.height).padded(self._pad())


@dataclass(slots=True)
class Circle(Shape):
    """A circle.

    Attributes:
        cx (float): centre x.
        cy (float): centre y.
        r (float): radius.
    """

    cx: float
    cy: float
    r: float

    def bbox(self) -> BBox:
        """The square around the circle, stroke included.

        Returns:
            BBox: the box.
        """
        return BBox(self.cx - self.r, self.cy - self.r, self.cx + self.r, self.cy + self.r).padded(self._pad())


@dataclass(slots=True)
class Text(Node):
    """Typst markup set at a point.

    The text box runs from the font's ascender to its descender (taller only for tall math), so
    "x", "g" and "d" get the same box and baseline.

    Attributes:
        markup (str): typst markup, e.g. "Number of Photons" or "$10^(-2)$".
        x (float): anchor x.
        y (float): anchor y.
        size (float): font size in pt.
        anchor (Literal["start", "middle", "end"]): which part of the box sits at x.
        valign (Literal["baseline", "top", "middle", "bottom"]): which part of the box sits at y.
        font (str | None): font family, the preamble's if None.
        fill (Paint): the text colour.
    """

    markup: str
    x: float
    y: float
    size: float
    anchor: Literal["start", "middle", "end"] = "start"
    valign: Literal["baseline", "top", "middle", "bottom"] = "baseline"
    font: str | None = None
    fill: Paint = "#000000"

    def shape(self) -> text.TextShape:
        """The typeset markup, cached across nodes.

        Returns:
            text.TextShape: the glyphs and metrics.
        """
        return text.typeset(self.markup, self.size, self.font)

    def origin(self) -> tuple[float, float]:
        """Where the top-left corner of the text box sits.

        Returns:
            tuple[float, float]: the corner in the parent's y-up coordinates.
        """
        shape = self.shape()
        left = self.x - {"start": 0.0, "middle": 0.5, "end": 1.0}[self.anchor] * shape.width
        drop = {"baseline": shape.baseline, "top": 0.0, "middle": shape.height / 2, "bottom": shape.height}
        return left, self.y + drop[self.valign]

    def bbox(self) -> BBox:
        """The text box.

        Returns:
            BBox: the box.
        """
        shape = self.shape()
        left, top = self.origin()
        return BBox(left, top - shape.height, left + shape.width, top)
