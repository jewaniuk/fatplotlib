"""axes: scales, ticks, spines and data drawing on a rectangle that layout places.

an `Axes` is built before it has a rectangle: `line`, `lines`, `markers`, `heatmap`, `fill_between`,
`grid`, `arrow`, `text`, `draw` and `inset` record what to draw, and `group()` builds the scene nodes
once `place` has given the rectangle (`inner`) and the font size that em lengths scale. around the
rectangle it draws spines on all four sides and, per `Side`, inward major and minor ticks and,
on labelled sides, typst tick labels and an axis label: a plain axes ticks every side and labels
the bottom and left (and the top for a twin axis). data (everything but `text`, unclipped markers
and insets) is clipped to the rectangle, with the grid under it. `outer()` is the box the
decorations and notes (text, unclipped markers) reach, which layout measures.

a `Colorbar` is an axes whose data is a colormap's strip and whose one labelled side is the norm.

ticks follow `Scale`'s nice-number rule, with `max_gaps` from the axis length and label size, as
in matplotlib. odd ranges (under a decade on a log axis, big values with tiny spans) take explicit
`ticks=`.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from fatplotlib.colormap import DEFAULT, Colormap, Norm
from fatplotlib.panel import Panel
from fatplotlib.scale import EPS, Scale
from fatplotlib.scene import BBox, Command, Group, Markers, MarkerShape, Node, Path, Polylines, Rect, Text, Themed
from fatplotlib.style import LinearGradient, Paint, Stop, Stroke, Token
from fatplotlib.transform import Affine
from fatplotlib.units import length

type Coords = Literal["data", "axes"]
"""what a position means: data values, or fractions of the axes rectangle (0 to 1)."""

type Resolved = tuple[list[float], list[float], list[str]]
"""an axis's major ticks, minor ticks and tick labels, as `Axis.resolved` gives them."""

type SideName = Literal["bottom", "left", "top", "right"]
"""a side of the axes rectangle."""

# stops sampled from a continuous colormap for a colorbar's gradient
_GRADIENT_STOPS = 64

@dataclass(slots=True)
class Axis:
    """The ticks and labels along one side of an axes.

    Attributes:
        scale (Scale): maps data to the side.
        ticks (Sequence[float] | None): major tick values, chosen by the scale if None.
        minor (Sequence[float] | bool | None): minor tick values; True has the scale choose them,
            False draws none, None does the scale's default (chosen on log, none on linear).
        labels (Sequence[str] | bool): markup for each major tick; True formats the values,
            False draws none.
        label (str | None): the axis label markup, none if None.
    """

    scale: Scale
    ticks: Sequence[float] | None = None
    minor: Sequence[float] | bool | None = None
    labels: Sequence[str] | bool = True
    label: str | None = None

    def resolved(self, max_gaps: int) -> Resolved:
        """The major ticks, minor ticks and tick labels to draw.

        Args:
            max_gaps (int): the most gaps between automatic major ticks.

        Returns:
            Resolved: majors, minors and one label per major
                (empty if labels are off).
        """
        majors = list(self.ticks) if self.ticks is not None else self.scale.ticks(max_gaps)
        match self.minor:
            case True:
                minors = self.scale.minor_ticks(majors)
            case None:
                minors = self.scale.minor_ticks(majors) if self.scale.kind == "log" else []
            case False:
                minors = []
            case _:
                minors = list(self.minor)
        match self.labels:
            case True:
                labels = self.scale.labels(majors)
            case False:
                labels = []
            case _:
                labels = list(self.labels)
        return majors, minors, labels


@dataclass(frozen=True, slots=True)
class Side:
    """What one side of the rectangle draws besides its spine.

    Attributes:
        axis (Axis): the axis whose ticks the side draws.
        labelled (bool): whether the side also draws the axis's tick labels and axis label;
            False for a mirror, which only ticks.
    """

    axis: Axis
    labelled: bool = True


@dataclass(frozen=True, slots=True)
class AxesStyle:
    """Sizes and paints of axes decorations. lengths follow matplotlib's defaults in pt; text
    sizes are relative to the target's base size, 8 / 9 em for tick labels as in the 8 and 9
    latex pt of the reference figures.

    Attributes:
        tick_label_size (float | str): tick label font size, pt or a length such as "0.9em".
        label_size (float | str): axis label and `text` font size, pt or a length.
        major (float): major tick length in pt.
        minor (float): minor tick length in pt.
        tick_pad (float): gap from the spine to the tick labels' ink (cap height below, baseline above) in pt.
        label_pad (float): gap from the tick labels to the axis label in pt.
        spine (float | Token): spine weight.
        major_width (float | Token): major tick weight.
        minor_width (float | Token): minor tick weight.
        paint (Paint): colour of spines, ticks and text.
    """

    tick_label_size: float | str = f"{8 / 9}em"
    label_size: float | str = "1em"
    major: float = 3.5
    minor: float = 2.0
    tick_pad: float = 3.5
    label_pad: float = 4.0
    spine: float | Token = Token("normal")
    major_width: float | Token = Token("normal")
    minor_width: float | Token = Token("fine")
    paint: Paint = Token("fg")


@dataclass(frozen=True, slots=True)
class Marker:
    """How markers look.

    Attributes:
        shape (MarkerShape): the marker shape.
        size (float | str): the circle's diameter, pt or a length such as "0.4em"; other shapes
            keep matplotlib's proportions to it.
        fill (Paint | None): the fill paint, none if None.
        stroke (Stroke | None): the outline, none if None (`+` and `x` need one).
    """

    shape: MarkerShape = "o"
    size: float | str = 3.0
    fill: Paint | None = Token("fg")
    stroke: Stroke | None = None


class Axes:
    """A plot: data drawing plus spines, ticks and labels, built once layout places it.

    The right side mirrors the left axis's ticks and the top mirrors the bottom's, unless
    `twin_x` gives the top its own axis.

    Attributes:
        x (Axis): the bottom axis.
        y (Axis): the left axis.
        top (Axis | None): the twin axis on top, None if the top mirrors `x`.
        style (AxesStyle): decoration sizes and paints.
        label (str | bool | None): the panel label: None letters it automatically, False skips
            it, a string replaces the letter.
        width (float | str | None): a fixed rectangle width, pt or a length such as "1em", that
            layout keeps; free if None.
        height (float | str | None): a fixed rectangle height, free if None.
        inner (BBox | None): the plot rectangle in pt, None until placed.
        em (float | None): the font size in pt that em lengths scale, None until placed.
    """

    def __init__(
        self,
        *,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        xscale: Literal["linear", "log"] = "linear",
        yscale: Literal["linear", "log"] = "linear",
        xticks: Sequence[float] | None = None,
        yticks: Sequence[float] | None = None,
        xminor: Sequence[float] | bool | None = None,
        yminor: Sequence[float] | bool | None = None,
        xticklabels: Sequence[str] | bool = True,
        yticklabels: Sequence[str] | bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        style: AxesStyle = AxesStyle(),
        label: str | bool | None = None,
        width: float | str | None = None,
        height: float | str | None = None,
    ) -> None:
        """Sets up the axes; see `Axis` for what each tick and label argument means.

        Args:
            xlim (tuple[float, float]): bottom axis start and end.
            ylim (tuple[float, float]): left axis start and end.
            xscale (Literal["linear", "log"]): bottom axis spacing.
            yscale (Literal["linear", "log"]): left axis spacing.
            xticks (Sequence[float] | None): bottom major ticks.
            yticks (Sequence[float] | None): left major ticks.
            xminor (Sequence[float] | bool | None): bottom minor ticks.
            yminor (Sequence[float] | bool | None): left minor ticks.
            xticklabels (Sequence[str] | bool): bottom tick labels.
            yticklabels (Sequence[str] | bool): left tick labels.
            xlabel (str | None): bottom axis label.
            ylabel (str | None): left axis label.
            style (AxesStyle): decoration sizes and paints.
            label (str | bool | None): the panel label.
            width (float | str | None): a fixed rectangle width.
            height (float | str | None): a fixed rectangle height.
        """
        self.x = Axis(Scale(xscale, xlim), xticks, xminor, xticklabels, xlabel)
        self.y = Axis(Scale(yscale, ylim), yticks, yminor, yticklabels, ylabel)
        self.top: Axis | None = None
        self.style = style
        self.label = label
        self.width = width
        self.height = height
        self.inner: BBox | None = None
        self.em: float | None = None
        # builders run by group(): clipped data, unclipped notes, insets on top
        self._data: list[Callable[[], Node]] = []
        self._notes: list[Callable[[], Node]] = []
        self._insets: list[Callable[[], Node]] = []

    def place(self, inner: BBox, em: float) -> None:
        """Gives the axes its rectangle; layout calls this, possibly more than once.

        Args:
            inner (BBox): the plot rectangle in pt.
            em (float): the font size in pt that em lengths scale.
        """
        self.inner = inner
        self.em = em

    def twin_x(
        self,
        lim: tuple[float, float] | None = None,
        scale: Literal["linear", "log"] | None = None,
        *,
        ticks: Sequence[float] | None = None,
        minor: Sequence[float] | bool | None = None,
        labels: Sequence[str] | bool = True,
        label: str | None = None,
    ) -> Axis:
        """Gives the top side its own axis, labelled above the rectangle, in place of the mirror.

        Args:
            lim (tuple[float, float] | None): start and end, the bottom axis's if None.
            scale (Literal["linear", "log"] | None): spacing, the bottom axis's if None.
            ticks (Sequence[float] | None): major ticks.
            minor (Sequence[float] | bool | None): minor ticks.
            labels (Sequence[str] | bool): tick labels.
            label (str | None): axis label.

        Returns:
            Axis: the top axis.
        """
        bottom = self.x.scale
        self.top = Axis(Scale(scale or bottom.kind, lim or bottom.lim), ticks, minor, labels, label)
        return self.top

    def to_pt(self, x: ArrayLike, y: ArrayLike, coords: Coords = "data") -> tuple[np.ndarray, np.ndarray]:
        """Maps positions to pt on the page.

        Args:
            x (ArrayLike): x positions.
            y (ArrayLike): y positions.
            coords (Coords): data values (bottom and left axes) or fractions of the rectangle.

        Returns:
            tuple[np.ndarray, np.ndarray]: x and y in pt, non-finite where the scale cannot map a
                value (log of values <= 0).

        Raises:
            RuntimeError: the axes has not been placed.
        """
        r = self._rect()
        if coords == "data":
            fx, fy = self.x.scale.fraction(x), self.y.scale.fraction(y)
        else:
            fx, fy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return r.x0 + fx * r.width, r.y0 + fy * r.height

    def line(
        self,
        x: ArrayLike,
        y: ArrayLike,
        stroke: Stroke,
        coords: Coords = "data",
        *,
        marker: Marker | None = None,
        **node: object,
    ) -> None:
        """Draws a polyline through the points, broken wherever a point cannot be mapped.

        Args:
            x (ArrayLike): x positions.
            y (ArrayLike): y positions.
            stroke (Stroke): the line style (solid, or dashed with `dash`).
            coords (Coords): what the positions mean.
            marker (Marker | None): markers drawn over the line at each point, none if None.
            **node (object): other `Path` fields such as id or alpha, for the line only.
        """
        self._data.append(lambda: self._line(x, y, stroke, coords, node))
        if marker is not None:
            self.markers(x, y, marker, coords)

    def markers(
        self,
        x: ArrayLike,
        y: ArrayLike,
        marker: Marker = Marker(),
        coords: Coords = "data",
        *,
        values: ArrayLike | None = None,
        cmap: Colormap = DEFAULT,
        norm: Norm | None = None,
        clip: bool = True,
        **node: object,
    ) -> None:
        """Draws a marker at each point, skipping points that cannot be mapped.

        With `values` the fill comes from the colormap instead of the marker: points are grouped
        into one `Markers` per lut colour, so markers of one colour draw together and overlaps
        between colours follow lut order, not point order.

        Args:
            x (ArrayLike): x positions.
            y (ArrayLike): y positions.
            marker (Marker): the marker style.
            coords (Coords): what the positions mean.
            values (ArrayLike | None): a value per point, coloured through `cmap`.
            cmap (Colormap): the colormap for `values`.
            norm (Norm | None): maps `values` to the colormap, linear over their finite range if None.
            clip (bool): clip to the rectangle like data; False draws them as notes, which count
                toward `outer()` so layout makes room (hand-made legends, points on spines).
            **node (object): other node fields such as id or alpha, on the `Markers`, or on the
                group around them with `values`.
        """
        layer = self._data if clip else self._notes
        layer.append(lambda: self._markers(x, y, marker, coords, values=values, cmap=cmap, norm=norm, node=node))

    def _markers(
        self,
        x: ArrayLike,
        y: ArrayLike,
        marker: Marker,
        coords: Coords,
        *,
        values: ArrayLike | None,
        cmap: Colormap,
        norm: Norm | None,
        node: dict[str, object],
    ) -> Node:
        """The markers for `markers`, in pt: one `Markers`, or a group of them per lut colour,
        split by theme when the colormap is chosen per theme."""
        px, py = (np.ravel(v) for v in self.to_pt(x, y, coords))
        size = self._size(marker.size)
        if values is None:
            return Markers(px, py, marker.shape, size, fill=marker.fill, stroke=marker.stroke, **node)  # pyright: ignore[reportArgumentType]
        v = np.ravel(np.asarray(values, dtype=float))

        def build(chosen: np.ndarray, paint: Paint) -> Markers:
            """The markers of one colour."""
            return Markers(px[chosen], py[chosen], marker.shape, size, fill=paint, stroke=marker.stroke)

        return _by_colour(cmap, (norm or _span(v)).fraction(v), build, node)

    def heatmap(
        self,
        x_edges: ArrayLike,
        y_edges: ArrayLike,
        z: ArrayLike,
        cmap: Colormap = DEFAULT,
        norm: Norm | None = None,
        *,
        edges: Stroke | None = None,
        **node: object,
    ) -> None:
        """Fills a grid of cells by value, clipped like data. cells of one colour are one path of
        rectangles, so they meet without seams; nan cells take `cmap.bad`, or are not drawn.

        Args:
            x_edges (ArrayLike): cell edges along x in data values, shape (nx + 1,).
            y_edges (ArrayLike): cell edges along y, shape (ny + 1,).
            z (ArrayLike): a value per cell, shape (nx, ny), as `np.meshgrid(..., indexing="ij")` gives.
            cmap (Colormap): the colormap.
            norm (Norm | None): maps z to the colormap, linear over its finite range if None.
            edges (Stroke | None): an outline around every cell, none if None.
            **node (object): other node fields such as id or alpha, on the group of cells.
        """
        self._data.append(lambda: self._heatmap(x_edges, y_edges, z, cmap, norm, edges=edges, node=node))

    def _heatmap(
        self,
        x_edges: ArrayLike,
        y_edges: ArrayLike,
        z: ArrayLike,
        cmap: Colormap,
        norm: Norm | None,
        *,
        edges: Stroke | None,
        node: dict[str, object],
    ) -> Node:
        """The cells for `heatmap`, in pt, one `Path` per lut colour."""
        r = self._rect()
        px = (r.x0 + self.x.scale.fraction(x_edges) * r.width).tolist()
        py = (r.y0 + self.y.scale.fraction(y_edges) * r.height).tolist()
        values = np.asarray(z, dtype=float)
        # cells with an edge the scale cannot map (log of values <= 0) are left out
        drawable = np.isfinite(px[:-1]) & np.isfinite(px[1:])
        drawable = drawable[:, None] & (np.isfinite(py[:-1]) & np.isfinite(py[1:]))[None, :]

        def build(chosen: np.ndarray, paint: Paint) -> Path:
            """The cells of one colour, a closed rectangle each."""
            commands: list[Command] = []
            for i, j in zip(*(a.tolist() for a in np.nonzero(chosen & drawable))):
                x0, x1, y0, y1 = px[i], px[i + 1], py[j], py[j + 1]
                commands += [("M", x0, y0), ("L", x1, y0), ("L", x1, y1), ("L", x0, y1), ("Z",)]
            return Path(tuple(commands), fill=paint, stroke=edges)

        return _by_colour(cmap, (norm or _span(values)).fraction(values), build, node)

    def fill_between(
        self,
        x: ArrayLike,
        y0: ArrayLike,
        y1: ArrayLike,
        fill: Paint,
        stroke: Stroke | None = None,
        **node: object,
    ) -> None:
        """Fills between two curves sharing x, clipped like data, as one path with a closed
        region per run of points that can all be mapped.

        Args:
            x (ArrayLike): x positions.
            y0 (ArrayLike): one edge's y positions, or one value for a flat edge.
            y1 (ArrayLike): the other edge's y positions, or one value.
            fill (Paint): the fill paint.
            stroke (Stroke | None): the outline of each region, none if None.
            **node (object): other `Path` fields such as id or alpha.
        """
        self._data.append(lambda: self._fill_between(x, y0, y1, fill, stroke, node))

    def _fill_between(  # noqa: PLR0917 - private, mirrors fill_between
        self, x: ArrayLike, y0: ArrayLike, y1: ArrayLike, fill: Paint, stroke: Stroke | None, node: dict[str, object]
    ) -> Path:
        """The regions for `fill_between`, in pt: along y1, back along y0, closed."""
        xs, lows, highs = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (x, y0, y1)))
        px, p0 = self.to_pt(xs, lows)
        _, p1 = self.to_pt(xs, highs)
        finite = np.isfinite(px) & np.isfinite(p0) & np.isfinite(p1)
        # runs of finite points: starts where finite turns on, ends where it turns off
        edges = np.diff(np.concatenate([[False], finite, [False]]).astype(int))
        commands: list[Command] = []
        for start, stop in zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()):
            top = list(zip(px[start:stop].tolist(), p1[start:stop].tolist()))
            bottom = list(zip(px[start:stop].tolist(), p0[start:stop].tolist()))[::-1]
            commands.append(("M", *top[0]))
            commands += [("L", *point) for point in top[1:] + bottom]
            commands.append(("Z",))
        return Path(tuple(commands), fill=fill, stroke=stroke, **node)  # pyright: ignore[reportArgumentType]

    def grid(self, axis: Literal["x", "y", "both"] = "y", stroke: Stroke = Stroke(Token("grid"), Token("hair"))) -> None:
        """Draws lines across the rectangle at the major ticks, under all data.

        Args:
            axis (Literal["x", "y", "both"]): "y" draws horizontal lines at the left axis's ticks,
                "x" vertical lines at the bottom axis's ticks.
            stroke (Stroke): the line style.
        """

        def build() -> Path:
            """The grid lines: ticks as long as the rectangle."""
            r = self._rect()
            resolved = self._resolved()
            commands: list[Command] = []
            if axis in ("x", "both"):
                commands += self._ticks(self.x.scale, resolved["bottom"][0], "bottom", r.height)
            if axis in ("y", "both"):
                commands += self._ticks(self.y.scale, resolved["left"][0], "left", r.width)
            return Path(tuple(commands), stroke=stroke)

        self._data.insert(0, build)

    def _line(self, x: ArrayLike, y: ArrayLike, stroke: Stroke, coords: Coords, node: dict[str, object]) -> Path:
        """The polyline for `line`, in pt."""
        px, py = self.to_pt(x, y, coords)
        commands: list[Command] = []
        lifted = True
        for xi, yi in zip(px.tolist(), py.tolist()):
            if not (math.isfinite(xi) and math.isfinite(yi)):
                lifted = True
                continue
            commands.append(("M", xi, yi) if lifted else ("L", xi, yi))
            lifted = False
        return Path(tuple(commands), stroke=stroke, **node)  # pyright: ignore[reportArgumentType]

    def lines(self, x: ArrayLike, ys: ArrayLike, stroke: Stroke, coords: Coords = "data", **node: object) -> None:
        """Draws many curves sharing x as one `Polylines`, far cheaper than a `line` per curve.
        each curve breaks wherever a point cannot be mapped.

        Args:
            x (ArrayLike): x positions, shape (points,), shared by every curve.
            ys (ArrayLike): y positions, shape (curves, points).
            stroke (Stroke): the line style, shared by every curve.
            coords (Coords): what the positions mean.
            **node (object): other `Polylines` fields such as id or alpha.
        """
        self._data.append(lambda: self._lines(x, ys, stroke, coords, node))

    def _lines(self, x: ArrayLike, ys: ArrayLike, stroke: Stroke, coords: Coords, node: dict[str, object]) -> Polylines:
        """The polylines for `lines`, in pt."""
        ys = np.atleast_2d(np.asarray(ys, dtype=float))
        px, py = self.to_pt(np.broadcast_to(np.asarray(x, dtype=float), ys.shape), ys, coords)
        return Polylines(px, py, stroke=stroke, **node)  # pyright: ignore[reportArgumentType]

    def arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        stroke: Stroke,
        *,
        coords: Coords = "data",
        head: tuple[float, float] = (4.0, 3.0),
        both: bool = False,
    ) -> None:
        """Draws a straight arrow with a filled triangular head at the end, or at both ends.

        Args:
            start (tuple[float, float]): the tail position.
            end (tuple[float, float]): the head position.
            stroke (Stroke): the shaft style; the heads are filled with its paint.
            coords (Coords): what the positions mean.
            head (tuple[float, float]): head length and width in pt.
            both (bool): whether the tail gets a head too.
        """
        self._data.append(lambda: self._arrow(start, end, stroke, coords=coords, head=head, both=both))

    def _arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        stroke: Stroke,
        *,
        coords: Coords,
        head: tuple[float, float],
        both: bool,
    ) -> Group:
        """The shaft and heads for `arrow`, in pt."""
        (x0, x1), (y0, y1) = self.to_pt([start[0], end[0]], [start[1], end[1]], coords)
        length, width = head
        dx, dy = float(x1 - x0), float(y1 - y0)
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm

        def tip(x: float, y: float, sx: float, sy: float) -> Path:
            """A head with its point at (x, y), facing along (sx, sy)."""
            bx, by = x - length * sx, y - length * sy
            ox, oy = -sy * width / 2, sx * width / 2
            corners: tuple[Command, ...] = (("M", x, y), ("L", bx + ox, by + oy), ("L", bx - ox, by - oy), ("Z",))
            return Path(corners, fill=stroke.paint)

        tail = (float(x0) + length * ux, float(y0) + length * uy) if both else (float(x0), float(y0))
        shaft = Path((("M", *tail), ("L", float(x1) - length * ux, float(y1) - length * uy)), stroke=stroke)
        arrow = Group([shaft, tip(float(x1), float(y1), ux, uy)])
        if both:
            arrow.add(tip(float(x0), float(y0), -ux, -uy))
        return arrow

    def text(
        self,
        markup: str,
        at: tuple[float, float],
        coords: Coords = "data",
        *,
        size: float | str | None = None,
        anchor: Literal["start", "middle", "end"] = "start",
        valign: Literal["baseline", "cap", "top", "middle", "bottom"] = "baseline",
        fill: Paint | None = None,
    ) -> None:
        """Sets typst markup at a position, unclipped.

        Args:
            markup (str): typst markup.
            at (tuple[float, float]): the anchor position.
            coords (Coords): what the position means.
            size (float | str | None): font size, pt or a length such as "0.9em", the style's
                label size if None.
            anchor (Literal["start", "middle", "end"]): which part of the text sits at x.
            valign (Literal["baseline", "cap", "top", "middle", "bottom"]): which part sits at y.
            fill (Paint | None): text colour, the style's paint if None.
        """

        def build() -> Text:
            """The text in pt."""
            (x,), (y,) = self.to_pt([at[0]], [at[1]], coords)
            pt = self._size(self.style.label_size if size is None else size)
            return Text(markup, float(x), float(y), pt, anchor, valign, fill=fill or self.style.paint)

        self._notes.append(build)

    def draw(self, component: Callable[[Axes], Node]) -> None:
        """Adds a user component drawn once the axes is placed, clipped like data. the component
        gets the placed axes, so it can map with `to_pt` and size things in pt.

        Args:
            component (Callable[[Axes], Node]): builds the nodes, in pt.
        """
        self._data.append(lambda: component(self))

    def inset(self, panel: Panel, box: tuple[float, float, float, float], coords: Coords = "axes") -> None:
        """Places a panel inside the axes, fit and centred in a box, above everything else and
        unclipped. insets do not count toward `outer()` and get no panel label.

        Args:
            panel (Panel): the panel.
            box (tuple[float, float, float, float]): x0, y0, x1, y1 of the box.
            coords (Coords): what the box corners mean.
        """

        def build() -> Node:
            """The placed panel."""
            (x0, x1), (y0, y1) = self.to_pt([box[0], box[2]], [box[1], box[3]], coords)
            panel.place(BBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), self._em())
            panel.warn_gap()
            return panel.group()

        self._insets.append(build)

    def group(self) -> Group:
        """Everything drawn so far: data, then spines, ticks and labels, then notes and insets.

        Returns:
            Group: the axes as scene nodes, in page pt.

        Raises:
            RuntimeError: the axes has not been placed.
        """
        data = Group([build() for build in self._data], clip=self._rect())
        notes = Group([build() for build in self._notes])
        insets = Group([build() for build in self._insets])
        return Group([data, *self._decorations(), notes, insets])

    def outer(self) -> BBox:
        """The box around the rectangle and everything drawn outside it (tick and axis labels,
        notes), from measured text. data is clipped and insets sit inside, so neither counts.

        Returns:
            BBox: the box in pt.

        Raises:
            RuntimeError: the axes has not been placed.
        """
        box = Group([*self._decorations(), *(build() for build in self._notes)]).bbox()
        assert box is not None  # the spines always draw
        return box

    def _rect(self) -> BBox:
        """The placed rectangle."""
        if self.inner is None:
            raise RuntimeError("the axes has no rectangle yet: put it in a Figure layout")
        return self.inner

    def _em(self) -> float:
        """The placed font size."""
        if self.em is None:
            raise RuntimeError("the axes has no font size yet: put it in a Figure layout")
        return self.em

    def _size(self, value: float | str) -> float:
        """A length in pt, em resolved against the placed font size."""
        return length(value, self._em())

    def _sides(self) -> dict[SideName, Side]:
        """What each side draws: ticks on all four, labels on the bottom and left, and on top
        for a twin axis. a side left out draws only its spine."""
        return {
            "bottom": Side(self.x),
            "left": Side(self.y),
            "top": Side(self.x, labelled=False) if self.top is None else Side(self.top),
            "right": Side(self.y, labelled=False),
        }

    def _resolved(self) -> dict[SideName, Resolved]:
        """Each drawn side's majors, minors and tick labels, as many ticks as fit along it at
        the tick label size."""
        r, size = self._rect(), self._size(self.style.tick_label_size)
        across, up = int(r.width // (3 * size)), int(r.height // (2 * size))
        return {name: side.axis.resolved(across if name in ("bottom", "top") else up) for name, side in self._sides().items()}

    def _decorations(self) -> list[Node]:
        """Spines, ticks, and each labelled side's tick labels and axis label."""
        s, r = self.style, self._rect()
        sides, resolved = self._sides(), self._resolved()
        major: list[Command] = []
        minor: list[Command] = []
        for name, side in sides.items():
            majors, minors, _ = resolved[name]
            major += self._ticks(side.axis.scale, majors, name, s.major)
            minor += self._ticks(side.axis.scale, minors, name, s.minor)
        nodes: list[Node] = [
            Path(tuple(major), stroke=Stroke(s.paint, s.major_width)),
            Path(tuple(minor), stroke=Stroke(s.paint, s.minor_width)),
        ]
        # spines last among the lines, so their corners cover the tick ends
        nodes.append(Rect(r.x0, r.y0, r.width, r.height, stroke=Stroke(s.paint, s.spine)))
        for name, side in sides.items():
            if side.labelled:
                nodes += self._side_labels(name, side.axis, resolved[name])
        return nodes

    def _side_labels(self, name: SideName, axis: Axis, resolved: Resolved) -> list[Node]:
        """One side's tick labels, a tick pad out from the spine, and its axis label a label pad
        beyond them. side axis labels read bottom to top on the left and top to bottom on the
        right, their bottom edge facing the axis either way."""
        s, r = self.style, self._rect()
        size, label_size = self._size(s.tick_label_size), self._size(s.label_size)
        majors, _, labels = resolved
        horizontal = name in ("bottom", "top")
        start, span = (r.x0, r.width) if horizontal else (r.y0, r.height)
        at = [(start + f * span, m) for f, m in zip(axis.scale.fraction(majors).tolist(), labels) if -EPS <= f <= 1 + EPS]
        match name:
            case "bottom":
                texts = [Text(m, p, r.y0 - s.tick_pad, size, "middle", "cap", fill=s.paint) for p, m in at]
            case "top":
                texts = [Text(m, p, r.y1 + s.tick_pad, size, "middle", "baseline", fill=s.paint) for p, m in at]
            case "left":
                texts = [Text(m, r.x0 - s.tick_pad, p, size, "end", "middle", fill=s.paint) for p, m in at]
            case "right":
                texts = [Text(m, r.x1 + s.tick_pad, p, size, "start", "middle", fill=s.paint) for p, m in at]
        ticks = Group(list(texts))
        if axis.label is None:
            return [ticks]
        box = ticks.bbox() or BBox(r.x0 - s.tick_pad, r.y0 - s.tick_pad, r.x1 + s.tick_pad, r.y1 + s.tick_pad)
        mid_x, mid_y = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        match name:
            case "bottom":
                label: Node = Text(axis.label, mid_x, box.y0 - s.label_pad, label_size, "middle", "top", fill=s.paint)
            case "top":
                label = Text(axis.label, mid_x, box.y1 + s.label_pad, label_size, "middle", "bottom", fill=s.paint)
            case "left" | "right":
                x, turn = (box.x0 - s.label_pad, math.pi / 2) if name == "left" else (box.x1 + s.label_pad, -math.pi / 2)
                turned = Affine.translate(x, mid_y) @ Affine.rotate(turn)
                label = Group([Text(axis.label, 0, 0, label_size, "middle", "bottom", fill=s.paint)], transform=turned)
        return [ticks, label]

    def _ticks(self, scale: Scale, values: Sequence[float], side: SideName, size: float) -> list[Command]:
        """Inward tick segments of a length in pt for values along one side."""
        r = self._rect()
        commands: list[Command] = []
        for f in _visible(scale.fraction(values)):
            if side in ("bottom", "top"):
                x = r.x0 + f * r.width
                y = r.y0 if side == "bottom" else r.y1
                commands.append(("M", x, y))
                commands.append(("L", x, y + (size if side == "bottom" else -size)))
            else:
                y = r.y0 + f * r.height
                x = r.x0 if side == "left" else r.x1
                commands.append(("M", x, y))
                commands.append(("L", x + (size if side == "left" else -size), y))
        return commands


class Colorbar(Axes):
    """A colormap's key: a strip of its colours along a long axis that maps values through the
    norm, with ticks, tick labels and an axis label on one side and nothing on the others.

    It never gets a panel label. the strip is data, a gradient for a continuous map and a rect
    per colour for a stepped one (`levels` on the map, or `boundaries` on the norm). notes past
    an end, such as "$<= 10^(-9)$" for values under the map, go through `text` in axes coords.

    Attributes:
        cmap (Colormap): the colormap.
        norm (Norm): maps values to the map; its scale is the long axis.
        side (SideName): the labelled side, "right" or "left" for a vertical bar, "top" or
            "bottom" for a horizontal one.
    """

    def __init__(
        self,
        cmap: Colormap,
        norm: Norm,
        *,
        side: SideName = "right",
        thickness: float | str = "1em",
        ticks: Sequence[float] | None = None,
        minor: Sequence[float] | bool | None = None,
        labels: Sequence[str] | bool = True,
        label: str | None = None,
        style: AxesStyle = AxesStyle(),
    ) -> None:
        """Sets up the bar; see `Axis` for what each tick and label argument means.

        Args:
            cmap (Colormap): the colormap.
            norm (Norm): maps values to the map.
            side (SideName): the labelled side.
            thickness (float | str): the strip's short side, pt or a length, fixed in layout.
            ticks (Sequence[float] | None): major ticks along the long axis.
            minor (Sequence[float] | bool | None): minor ticks.
            labels (Sequence[str] | bool): tick labels.
            label (str | None): the axis label markup (not a panel label, which a bar never has).
            style (AxesStyle): decoration sizes and paints.
        """
        vertical = side in ("left", "right")
        super().__init__(
            xlim=(0, 1),
            ylim=(0, 1),
            style=style,
            label=False,
            width=thickness if vertical else None,
            height=None if vertical else thickness,
        )
        long = Axis(norm.scale, ticks, minor, labels, label)
        if vertical:
            self.y = long
        else:
            self.x = long
        self.cmap = cmap
        self.norm = norm
        self.side: SideName = side
        self._data.append(self._strip)

    def _sides(self) -> dict[SideName, Side]:
        """Only the labelled side draws ticks and labels."""
        return {self.side: Side(self.y if self.side in ("left", "right") else self.x)}

    def _strip(self) -> Node:
        """The strip filling the rectangle, split by theme for a map chosen per theme."""
        if self.cmap.themes:
            return Themed({name: self._strip_of(m) for name, m in self.cmap.themes.items()})
        return self._strip_of(self.cmap)

    def _strip_of(self, cmap: Colormap) -> Node:
        """One map's strip, in pt, lut entries placed along the long axis by the norm."""
        r = self._rect()
        vertical = self.side in ("left", "right")
        entries, edges = cmap.entries(), self.norm.boundaries
        if cmap.levels is None and edges is None:
            # the lut sampled evenly, its fraction along the map also its fraction along the axis
            fractions = np.linspace(0, 1, _GRADIENT_STOPS)
            stops = tuple(Stop(f, entries[i]) for f, i in zip(fractions.tolist(), cmap.index(fractions).tolist()))
            end = (0.0, 1.0) if vertical else (1.0, 0.0)
            return Rect(r.x0, r.y0, r.width, r.height, fill=LinearGradient(stops, 0.0, 0.0, *end))
        if edges is None:
            # one band per level, evenly along the axis
            bounds = np.linspace(0, 1, cmap.size + 1)
            colours = (bounds[:-1] + bounds[1:]) / 2
        else:
            # one band per bin, spaced by the norm's scale, coloured as the norm colours the bin
            bounds = self.norm.scale.fraction(edges)
            colours = self.norm.fraction((edges[:-1] + edges[1:]) / 2)
        bands: list[Node] = []
        for f0, f1, i in zip(bounds[:-1].tolist(), bounds[1:].tolist(), cmap.index(colours).tolist()):
            lo, hi = min(f0, f1), max(f0, f1)
            if vertical:
                bands.append(Rect(r.x0, r.y0 + lo * r.height, r.width, (hi - lo) * r.height, fill=entries[i]))
            else:
                bands.append(Rect(r.x0 + lo * r.width, r.y0, (hi - lo) * r.width, r.height, fill=entries[i]))
        return Group(bands)


def _span(values: np.ndarray) -> Norm:
    """A linear norm over the finite values."""
    return Norm("linear", (float(np.nanmin(values)), float(np.nanmax(values))))


def _by_colour[N: Node](
    cmap: Colormap, fractions: np.ndarray, build: Callable[[np.ndarray, Paint], N], node: dict[str, object]
) -> Node:
    """One node per lut colour present, built from a mask of the items with that colour (bad
    items left out unless the map colours them), split by theme for a per-theme map."""

    def grouped(cmap: Colormap) -> list[Node]:
        index = cmap.index(fractions)
        return [build(index == i, paint) for i in np.unique(index).tolist() if (paint := cmap.paint(i)) is not None]

    if cmap.themes:
        return Themed({name: Group(grouped(m)) for name, m in cmap.themes.items()}, **node)  # pyright: ignore[reportArgumentType]
    return Group(grouped(cmap), **node)  # pyright: ignore[reportArgumentType]


def _visible(fractions: np.ndarray) -> list[float]:
    """The fractions that land on the axis."""
    return [f for f in fractions.tolist() if -EPS <= f <= 1 + EPS]

