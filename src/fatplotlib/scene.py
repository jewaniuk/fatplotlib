"""scene nodes: a y-up tree of shapes in pt, which renderers walk.

every node takes an optional `id` and `class_` (the svg `class`), so css and animation can
address it, and has `bbox()` in its parent's coordinates, stroke included. geometry must not
depend on the theme, so `bbox()` reads weight tokens from the light theme.
"""

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from fatplotlib import text
from fatplotlib.style import Paint, Stroke, Token
from fatplotlib.theme import LIGHT
from fatplotlib.transform import Affine

type Command = (
    tuple[Literal["M", "L"], float, float]
    | tuple[Literal["C"], float, float, float, float, float, float]
    | tuple[Literal["Z"]]
)

type MarkerShape = Literal["o", "s", "^", "v", "d", "D", "+", "x"]
"""a marker: circle, square, triangle up or down, thin diamond, diamond, plus or cross."""

# circle control point offset for a radius of 0.5
_K = 0.5 * 0.5522847498
# diamond half-diagonal: the unit square turned 45 degrees
_H = math.sqrt(2) / 2
# each marker at size 1 centred on the origin, in matplotlib's proportions: the circle's
# diameter, the square's side, the triangles' width and height and the crosses' arms are 1
MARKER_SHAPES: dict[MarkerShape, tuple[Command, ...]] = {
    "o": (
        ("M", 0.5, 0.0),
        ("C", 0.5, _K, _K, 0.5, 0.0, 0.5),
        ("C", -_K, 0.5, -0.5, _K, -0.5, 0.0),
        ("C", -0.5, -_K, -_K, -0.5, 0.0, -0.5),
        ("C", _K, -0.5, 0.5, -_K, 0.5, 0.0),
        ("Z",),
    ),
    "s": (("M", -0.5, -0.5), ("L", 0.5, -0.5), ("L", 0.5, 0.5), ("L", -0.5, 0.5), ("Z",)),
    "^": (("M", 0.0, 0.5), ("L", -0.5, -0.5), ("L", 0.5, -0.5), ("Z",)),
    "v": (("M", 0.0, -0.5), ("L", 0.5, 0.5), ("L", -0.5, 0.5), ("Z",)),
    "d": (("M", 0.6 * _H, 0.0), ("L", 0.0, _H), ("L", -0.6 * _H, 0.0), ("L", 0.0, -_H), ("Z",)),
    "D": (("M", _H, 0.0), ("L", 0.0, _H), ("L", -_H, 0.0), ("L", 0.0, -_H), ("Z",)),
    "+": (("M", -0.5, 0.0), ("L", 0.5, 0.0), ("M", 0.0, -0.5), ("L", 0.0, 0.5)),
    "x": (("M", -0.5, -0.5), ("L", 0.5, 0.5), ("M", -0.5, 0.5), ("L", 0.5, -0.5)),
}

# gauss-legendre nodes and weights on [0, 1] for cubic arc length
_GAUSS_T, _GAUSS_W = np.polynomial.legendre.leggauss(24)
_GAUSS_T, _GAUSS_W = (_GAUSS_T + 1) / 2, _GAUSS_W / 2


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

    def intersection(self, other: BBox) -> BBox | None:
        """The overlap of both boxes.

        Args:
            other (BBox): the other box.

        Returns:
            BBox | None: the overlap, None if the boxes do not overlap.
        """
        box = BBox(max(self.x0, other.x0), max(self.y0, other.y0), min(self.x1, other.x1), min(self.y1, other.y1))
        return box if box.x0 <= box.x1 and box.y0 <= box.y1 else None

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


def _cubic_velocity(segments: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Derivatives of cubics, control points shape (..., 4, 2), at parameters t shape (k,): (..., k, 2)."""
    p0, p1, p2, p3 = (segments[..., i, None, :] for i in range(4))
    t = t[:, None]
    s = 1 - t
    return 3 * (s * s * (p1 - p0) + 2 * s * t * (p2 - p1) + t * t * (p3 - p2))


def _cubic_lengths(segments: np.ndarray, upto: float = 1.0) -> np.ndarray:
    """Arc lengths of cubics, control points shape (..., 4, 2), from t = 0 to `upto`."""
    speed = np.linalg.norm(_cubic_velocity(segments, upto * _GAUSS_T), axis=-1)
    return upto * speed @ _GAUSS_W


def _cubic_param(segment: np.ndarray, distance: float, length: float) -> float:
    """The parameter t at which one cubic, control points shape (4, 2), has drawn `distance` of
    its `length`: newton steps, falling back to bisection when a step leaves the bracket."""
    lo, hi = 0.0, 1.0
    t = distance / length
    for _ in range(50):
        error = float(_cubic_lengths(segment, t)) - distance
        if abs(error) <= 1e-12 * length:
            break
        lo, hi = (lo, t) if error > 0 else (t, hi)
        speed = float(np.linalg.norm(_cubic_velocity(segment, np.array([t]))))
        step = t - error / speed if speed > 0 else lo
        t = step if lo < step < hi else (lo + hi) / 2
    return t


def _line_cubic(a: tuple[float, float], b: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    """A straight segment as a cubic with control points at thirds, so its speed is constant."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = (x1 - x0) / 3, (y1 - y0) / 3
    return (a, (x0 + dx, y0 + dy), (x1 - dx, y1 - dy), b)


@dataclass(slots=True, kw_only=True)
class Node:
    """Fields every node shares.

    Attributes:
        id (str | None): the svg id.
        class_ (str | None): the svg class.
        alpha (float | Token): opacity of the whole node, 0 to 1, or an alpha token.
    """

    id: str | None = None
    class_: str | None = None
    alpha: float | Token = 1.0

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
        return 0.0 if self.stroke is None else LIGHT.number(self.stroke.width) / 2


@dataclass(slots=True)
class Group(Node):
    """A list of nodes under one transform, optionally clipped to a box.

    Attributes:
        children (list[Node]): the nodes, drawn in order.
        transform (Affine | None): maps child coordinates to the parent's.
        clip (BBox | None): the box, in child coordinates, outside which nothing is drawn.
    """

    children: list[Node] = field(default_factory=list)
    transform: Affine | None = None
    clip: BBox | None = None

    def add(self, *nodes: Node) -> None:
        """Appends nodes, drawn above the existing ones.

        Args:
            *nodes (Node): the nodes.
        """
        self.children.extend(nodes)

    def bbox(self) -> BBox | None:
        """The union of the children's boxes, cut to the clip, in the parent's coordinates.

        Returns:
            BBox | None: the box, None if nothing is drawn.
        """
        boxes = [b for b in (child.bbox() for child in self.children) if b is not None]
        if not boxes:
            return None
        box: BBox | None = boxes[0]
        for other in boxes[1:]:
            box = box.union(other)
        if self.clip is not None:
            box = box.intersection(self.clip)
        if box is None or self.transform is None:
            return box
        return box.transformed(self.transform)


@dataclass(frozen=True, slots=True)
class PathPoint:
    """A position along a path.

    Attributes:
        x (float): x.
        y (float): y.
        angle (float): the tangent direction in degrees, counterclockwise from +x.
    """

    x: float
    y: float
    angle: float


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

    def length(self) -> float:
        """The drawn length: lines exact, cubics by 24-node gauss-legendre quadrature, Z as a
        line back to the subpath start. moves add nothing.

        Returns:
            float: the length.
        """
        return float(_cubic_lengths(self._segments()).sum())

    def point_at(self, fraction: float) -> PathPoint:
        """The point a fraction of the way along the drawn length, with its tangent. for an
        absolute distance d use `point_at(d / path.length())`.

        Args:
            fraction (float): 0 at the start to 1 at the end.

        Returns:
            PathPoint: the position and tangent angle.

        Raises:
            ValueError: the fraction is outside 0 to 1, or the path draws nothing.
        """
        if not 0 <= fraction <= 1:
            raise ValueError(f"fraction {fraction} is outside 0 to 1")
        segments = self._segments()
        lengths = _cubic_lengths(segments)
        # zero-length pieces have no tangent, and drawing them moves nothing
        segments, lengths = segments[lengths > 0], lengths[lengths > 0]
        if len(lengths) == 0:
            raise ValueError("the path draws nothing")
        ends = np.cumsum(lengths)
        target = fraction * ends[-1]
        i = min(int(np.searchsorted(ends, target)), len(lengths) - 1)
        segment, length = segments[i], float(lengths[i])
        t = _cubic_param(segment, min(max(target - (ends[i] - length), 0.0), length), length)
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = segment.tolist()
        dx, dy = _cubic_velocity(segment, np.array([t]))[0].tolist()
        if math.hypot(dx, dy) <= 1e-12 * length:
            # a control point on an endpoint: take the direction from just inside
            dx, dy = _cubic_velocity(segment, np.array([min(max(t, 1e-6), 1 - 1e-6)]))[0].tolist()
        return PathPoint(
            _cubic_at(t, x0, x1, x2, x3), _cubic_at(t, y0, y1, y2, y3), math.degrees(math.atan2(dy, dx))
        )

    def _segments(self) -> np.ndarray:
        """The drawn pieces as cubic control points, shape (n, 4, 2); lines and Z become cubics."""
        segments: list[tuple[tuple[float, float], ...]] = []
        current = start = (0.0, 0.0)
        for command in self.commands:
            match command:
                case ("M", x, y):
                    current = start = (x, y)
                case ("L", x, y):
                    segments.append(_line_cubic(current, (x, y)))
                    current = (x, y)
                case ("C", x1, y1, x2, y2, x, y):
                    segments.append((current, (x1, y1), (x2, y2), (x, y)))
                    current = (x, y)
                case ("Z",):
                    segments.append(_line_cubic(current, start))
                    current = start
        return np.array(segments, dtype=float).reshape(-1, 4, 2)


@dataclass(slots=True, eq=False)
class Polylines(Shape):
    """Many polylines in one style, emitted as one svg path with a subpath per curve. a curve
    breaks wherever a point is not finite, like `Axes.line`.

    Attributes:
        xs (np.ndarray): x positions, shape (curves, points).
        ys (np.ndarray): y positions, the same shape.
    """

    xs: np.ndarray
    ys: np.ndarray

    def bbox(self) -> BBox | None:
        """The box around the finite points, stroke included.

        Returns:
            BBox | None: the box, None if no point is finite.
        """
        finite = np.isfinite(self.xs) & np.isfinite(self.ys)
        if not finite.any():
            return None
        xs, ys = self.xs[finite], self.ys[finite]
        return BBox(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())).padded(self._pad())


@dataclass(slots=True, eq=False)
class Markers(Shape):
    """One marker shape at many points in one style, emitted as one svg path. points that are
    not finite are skipped.

    Attributes:
        xs (np.ndarray): centre x positions, shape (points,).
        ys (np.ndarray): centre y positions, the same shape.
        shape (MarkerShape): the marker, one of `MARKER_SHAPES`.
        size (float): the circle's diameter in pt; other shapes keep matplotlib's proportions to it.
    """

    xs: np.ndarray
    ys: np.ndarray
    shape: MarkerShape = "o"
    size: float = 3.0

    def bbox(self) -> BBox | None:
        """The box around every drawn marker, stroke included.

        Returns:
            BBox | None: the box, None if no point is finite.
        """
        finite = np.isfinite(self.xs) & np.isfinite(self.ys)
        if not finite.any():
            return None
        xs, ys = self.xs[finite], self.ys[finite]
        reach = [abs(v) for c in MARKER_SHAPES[self.shape] for v in c[1:]]
        # x and y alternate in every command, so even entries are x and odd are y
        hx, hy = max(reach[0::2]) * self.size, max(reach[1::2]) * self.size
        box = BBox(float(xs.min()) - hx, float(ys.min()) - hy, float(xs.max()) + hx, float(ys.max()) + hy)
        return box.padded(self._pad())


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
class Ellipse(Shape):
    """An axis-aligned ellipse; rotate one with a `Group` transform.

    Attributes:
        cx (float): centre x.
        cy (float): centre y.
        rx (float): x radius.
        ry (float): y radius.
    """

    cx: float
    cy: float
    rx: float
    ry: float

    def bbox(self) -> BBox:
        """The box around the ellipse, stroke included.

        Returns:
            BBox: the box.
        """
        return BBox(self.cx - self.rx, self.cy - self.ry, self.cx + self.rx, self.cy + self.ry).padded(self._pad())


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
        valign (Literal["baseline", "cap", "top", "middle", "bottom"]): which part of the box sits
            at y; "cap" is the top of capitals and digits, so labels hang from their ink.
        font (str | None): font family, the preamble's if None.
        fill (Paint): the text colour, the theme's foreground by default.
    """

    markup: str
    x: float
    y: float
    size: float
    anchor: Literal["start", "middle", "end"] = "start"
    valign: Literal["baseline", "cap", "top", "middle", "bottom"] = "baseline"
    font: str | None = None
    fill: Paint = Token("fg")

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
        drop = {"baseline": shape.baseline, "cap": shape.cap, "top": 0.0, "middle": shape.height / 2, "bottom": shape.height}
        return left, self.y + drop[self.valign]

    def bbox(self) -> BBox:
        """The text box.

        Returns:
            BBox: the box.
        """
        shape = self.shape()
        left, top = self.origin()
        return BBox(left, top - shape.height, left + shape.width, top)
