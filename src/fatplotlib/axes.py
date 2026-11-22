"""axes: scales, ticks, spines and data drawing on a rectangle that layout places.

an `Axes` is built before it has a rectangle: `line`, `lines`, `markers`, `arrow`, `text`, `draw` and
`inset` record what to draw, and `group()` builds the scene nodes once `place` has given the rectangle
(`inner`) and the font size that em lengths scale. around the rectangle it draws spines on all four
sides, inward major and minor ticks on every side, typst tick labels and axis labels on the bottom and
left (and on top for a twin axis). data drawn with `line`, `lines`, `markers`, `arrow` and `draw` is
clipped to the rectangle, `text`, unclipped markers and insets are not. `outer()` is the box the
decorations and notes (text, unclipped markers) reach, which layout measures.

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
from fatplotlib.style import Paint, Stroke, Token
from fatplotlib.transform import Affine
from fatplotlib.units import length

type Coords = Literal["data", "axes"]
"""what a position means: data values, or fractions of the axes rectangle (0 to 1)."""

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

    def resolved(self, max_gaps: int) -> tuple[list[float], list[float], list[str]]:
        """The major ticks, minor ticks and tick labels to draw.

        Args:
            max_gaps (int): the most gaps between automatic major ticks.

        Returns:
            tuple[list[float], list[float], list[str]]: majors, minors and one label per major
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
        """
        self.x = Axis(Scale(xscale, xlim), xticks, xminor, xticklabels, xlabel)
        self.y = Axis(Scale(yscale, ylim), yticks, yminor, yticklabels, ylabel)
        self.top: Axis | None = None
        self.style = style
        self.label = label
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
        norm = norm or Norm("linear", (float(np.nanmin(v)), float(np.nanmax(v))))
        fractions = norm.fraction(v)

        def grouped(cmap: Colormap) -> list[Node]:
            """One `Markers` per lut index present, bad points left out unless the map colours them."""
            index = cmap.index(fractions)
            groups: list[Node] = []
            for i in np.unique(index).tolist():
                paint = cmap.paint(i)
                if paint is not None:
                    chosen = index == i
                    groups.append(Markers(px[chosen], py[chosen], marker.shape, size, fill=paint, stroke=marker.stroke))
            return groups

        if cmap.themes:
            return Themed({name: Group(grouped(m)) for name, m in cmap.themes.items()}, **node)  # pyright: ignore[reportArgumentType]
        return Group(grouped(cmap), **node)  # pyright: ignore[reportArgumentType]

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

    def _decorations(self) -> list[Node]:
        """Spines, ticks, tick labels and axis labels."""
        s, r = self.style, self._rect()
        size, label_size = self._size(s.tick_label_size), self._size(s.label_size)
        x_gaps = int(r.width // (3 * size))
        y_gaps = int(r.height // (2 * size))
        bottom, left = self.x.resolved(x_gaps), self.y.resolved(y_gaps)
        top = bottom if self.top is None else self.top.resolved(x_gaps)
        top_scale = self.x.scale if self.top is None else self.top.scale

        major: list[Command] = []
        minor: list[Command] = []
        for scale, (majors, minors, _), side in (
            (self.x.scale, bottom, "bottom"),
            (top_scale, top, "top"),
            (self.y.scale, left, "left"),
            (self.y.scale, left, "right"),
        ):
            major += self._ticks(scale, majors, side, s.major)
            minor += self._ticks(scale, minors, side, s.minor)
        nodes: list[Node] = [
            Path(tuple(major), stroke=Stroke(s.paint, s.major_width)),
            Path(tuple(minor), stroke=Stroke(s.paint, s.minor_width)),
        ]
        # spines last among the lines, so their corners cover the tick ends
        nodes.append(Rect(r.x0, r.y0, r.width, r.height, stroke=Stroke(s.paint, s.spine)))

        labels = Group(
            [
                *(Text(m, x, r.y0 - s.tick_pad, size, "middle", "cap", fill=s.paint) for x, m in self._placed(self.x.scale, bottom, "x")),
                *(Text(m, r.x0 - s.tick_pad, y, size, "end", "middle", fill=s.paint) for y, m in self._placed(self.y.scale, left, "y")),
            ]
        )
        if self.top is not None:
            labels.add(*(Text(m, x, r.y1 + s.tick_pad, size, "middle", "baseline", fill=s.paint) for x, m in self._placed(top_scale, top, "x")))
        nodes.append(labels)

        texts = [t for t in labels.children if isinstance(t, Text)]
        mid_x, mid_y = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if self.x.label is not None:
            below = [t for t in texts if t.valign == "cap"]
            y = min((t.bbox().y0 for t in below), default=r.y0 - s.tick_pad) - s.label_pad
            nodes.append(Text(self.x.label, mid_x, y, label_size, "middle", "top", fill=s.paint))
        if self.top is not None and self.top.label is not None:
            above = [t for t in texts if t.valign == "baseline"]
            y = max((t.bbox().y1 for t in above), default=r.y1 + s.tick_pad) + s.label_pad
            nodes.append(Text(self.top.label, mid_x, y, label_size, "middle", "bottom", fill=s.paint))
        if self.y.label is not None:
            beside = [t for t in texts if t.anchor == "end"]
            x = min((t.bbox().x0 for t in beside), default=r.x0 - s.tick_pad) - s.label_pad
            # read bottom to top, with the text's bottom edge facing the axis
            turned = Affine.translate(x, mid_y) @ Affine.rotate(math.pi / 2)
            nodes.append(Group([Text(self.y.label, 0, 0, label_size, "middle", "bottom", fill=s.paint)], transform=turned))
        return nodes

    def _ticks(self, scale: Scale, values: Sequence[float], side: str, size: float) -> list[Command]:
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

    def _placed(
        self, scale: Scale, resolved: tuple[list[float], list[float], list[str]], direction: str
    ) -> list[tuple[float, str]]:
        """(position in pt, markup) for each visible labelled major tick."""
        majors, _, labels = resolved
        r = self._rect()
        start, span = (r.x0, r.width) if direction == "x" else (r.y0, r.height)
        fractions = scale.fraction(majors).tolist()
        return [(start + f * span, m) for f, m in zip(fractions, labels) if -EPS <= f <= 1 + EPS]


def _visible(fractions: np.ndarray) -> list[float]:
    """The fractions that land on the axis."""
    return [f for f in fractions.tolist() if -EPS <= f <= 1 + EPS]

