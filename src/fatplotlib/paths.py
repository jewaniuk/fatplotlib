"""PathBuilder: the move / line / cubic / arc helpers of planting-seeds' figtools, as one object.

the builder tracks the current point, so the relative commands need no start point and the
absolute ones return nothing to thread through. given a panel `Frame`, it draws in panel units:
points map through `frame.pt` and radii through `frame.length`, so a schematic scales with its
panel while strokes keep their width. arcs, quadratics and fillets are stored as cubics, so a path
only ever holds M, L, C and Z.
"""

import math
from typing import Any

from fatplotlib.panel import Frame
from fatplotlib.scene import Command, Path
from fatplotlib.units import length

# the widest arc one cubic approximates
_MAX_SPAN = math.pi / 2


class PathBuilder:
    """Builds path commands one segment at a time; every method returns the builder for chaining.

    Attributes:
        frame (Frame | None): the panel the builder draws in, None to draw in pt.
        commands (list[Command]): the absolute commands so far, in pt.
        current (tuple[float, float]): the pen position, in the builder's units.
    """

    def __init__(self, frame: Frame | None = None) -> None:
        """Starts an empty path with the pen at the origin.

        Args:
            frame (Frame | None): the panel to draw in, None to draw in pt.
        """
        self.frame = frame
        self.commands: list[Command] = []
        self.current: tuple[float, float] = (0.0, 0.0)
        self._start = self.current

    def _pt(self, x: float, y: float) -> tuple[float, float]:
        """Maps a point in the builder's units to pt."""
        return (x, y) if self.frame is None else self.frame.pt(x, y)

    def _radius(self, value: float | str) -> float:
        """A radius in the builder's units: numbers as given, strings as fixed lengths such as "2pt"."""
        if self.frame is None:
            return length(value)
        return self.frame.length(value) / self.frame.scale

    def move_to(self, x: float, y: float) -> PathBuilder:
        """Starts a new subpath.

        Args:
            x (float): x.
            y (float): y.

        Returns:
            PathBuilder: this builder.
        """
        self.commands.append(("M", *self._pt(x, y)))
        self.current = self._start = (x, y)
        return self

    def line_to(self, x: float, y: float) -> PathBuilder:
        """Draws a straight segment.

        Args:
            x (float): end x.
            y (float): end y.

        Returns:
            PathBuilder: this builder.
        """
        self.commands.append(("L", *self._pt(x, y)))
        self.current = (x, y)
        return self

    def cubic_to(self, c1x: float, c1y: float, c2x: float, c2y: float, x: float, y: float) -> PathBuilder:
        """Draws a cubic bezier.

        Args:
            c1x (float): first control x.
            c1y (float): first control y.
            c2x (float): second control x.
            c2y (float): second control y.
            x (float): end x.
            y (float): end y.

        Returns:
            PathBuilder: this builder.
        """
        self.commands.append(("C", *self._pt(c1x, c1y), *self._pt(c2x, c2y), *self._pt(x, y)))
        self.current = (x, y)
        return self

    def quad_to(self, cx: float, cy: float, x: float, y: float) -> PathBuilder:
        """Draws a quadratic bezier, stored as the exactly equal cubic.

        Args:
            cx (float): control x.
            cy (float): control y.
            x (float): end x.
            y (float): end y.

        Returns:
            PathBuilder: this builder.
        """
        x0, y0 = self.current
        c1 = (x0 + 2 / 3 * (cx - x0), y0 + 2 / 3 * (cy - y0))
        c2 = (x + 2 / 3 * (cx - x), y + 2 / 3 * (cy - y))
        return self.cubic_to(*c1, *c2, x, y)

    def arc_to(self, theta_in: float, theta_out: float, radius: float | str) -> PathBuilder:
        """Draws a circular arc counterclockwise from the pen, as figtools' `arc_to`.

        the angles place the pen and the end on the circle, measured from its centre, so the
        centre sits `radius` away from the pen at `theta_in + 180`. `theta_out` wraps as in
        matplotlib's `Path.arc`: it is moved into (theta_in, theta_in + 360], and equal angles draw
        nothing.

        Args:
            theta_in (float): the pen's angle on the circle, in degrees.
            theta_out (float): the end's angle on the circle, in degrees.
            radius (float | str): the radius, a number in the builder's units or a fixed length
                such as "2pt".

        Returns:
            PathBuilder: this builder.
        """
        if theta_out == theta_in:
            return self
        theta_out -= 360 * math.floor((theta_out - theta_in) / 360)
        if theta_out <= theta_in:
            theta_out += 360
        r = self._radius(radius)
        start = math.radians(theta_in)
        x, y = self.current
        centre = (x - r * math.cos(start), y - r * math.sin(start))
        return self._arc(centre, r, r, 0.0, start, math.radians(theta_out - theta_in))

    def elliptical_arc_to(
        self, rx: float | str, ry: float | str, rotation: float, large: bool, sweep: bool, x: float, y: float
    ) -> PathBuilder:
        """Draws an elliptical arc to a point, with the svg `A` command's meaning.

        of the four arcs of the ellipse through the pen and the end, `large` picks one spanning
        more than 180° and `sweep` one drawn in the positive angle direction, which is
        counterclockwise here since y points up. radii too small to reach the end are scaled up
        until they do, and a zero radius draws a straight line.

        Args:
            rx (float | str): the x radius before rotation, in the builder's units or a fixed length.
            ry (float | str): the y radius before rotation, in the builder's units or a fixed length.
            rotation (float): the ellipse's x axis angle, in degrees counterclockwise.
            large (bool): take the arc spanning more than 180°.
            sweep (bool): take the arc drawn counterclockwise.
            x (float): end x.
            y (float): end y.

        Returns:
            PathBuilder: this builder.
        """
        x0, y0 = self.current
        if (x0, y0) == (x, y):
            return self
        rx, ry = abs(self._radius(rx)), abs(self._radius(ry))
        if rx == 0 or ry == 0:
            return self.line_to(x, y)
        # svg implementation notes, appendix b.2.4: endpoint to centre parameterization
        phi = math.radians(rotation)
        cos, sin = math.cos(phi), math.sin(phi)
        hx, hy = (x0 - x) / 2, (y0 - y) / 2
        px, py = cos * hx + sin * hy, -sin * hx + cos * hy
        reach = (px / rx) ** 2 + (py / ry) ** 2
        if reach > 1:
            rx, ry = rx * math.sqrt(reach), ry * math.sqrt(reach)
        num = max(0.0, (rx * ry) ** 2 - (rx * py) ** 2 - (ry * px) ** 2)
        k = math.sqrt(num / ((rx * py) ** 2 + (ry * px) ** 2))
        if large == sweep:
            k = -k
        qx, qy = k * rx * py / ry, -k * ry * px / rx
        centre = (cos * qx - sin * qy + (x0 + x) / 2, sin * qx + cos * qy + (y0 + y) / 2)
        start = math.atan2((py - qy) / ry, (px - qx) / rx)
        span = math.atan2((-py - qy) / ry, (-px - qx) / rx) - start
        if sweep and span < 0:
            span += 2 * math.pi
        elif not sweep and span > 0:
            span -= 2 * math.pi
        self._arc(centre, rx, ry, phi, start, span)
        # land exactly on the requested end, free of rounding in the centre
        self.commands[-1] = (*self.commands[-1][:5], *self._pt(x, y))  # type: ignore[assignment]
        self.current = (x, y)
        return self

    def lines(self, points: list[tuple[float, float]], radius: float | str = 0) -> PathBuilder:
        """Draws straight segments through each point, optionally with rounded corners.

        with a radius, every corner between two segments (the pen counts as the first point) gets
        a circular fillet tangent to both. at each corner the fillet is shrunk so it uses at most
        half of either neighbouring segment, leaving room for the next corner's fillet.

        Args:
            points (list[tuple[float, float]]): the points, in order.
            radius (float | str): the corner radius, in the builder's units or a fixed length.

        Returns:
            PathBuilder: this builder.
        """
        r = self._radius(radius)
        corners = [self.current, *points]
        for (ax, ay), (bx, by), (cx, cy) in zip(corners, corners[1:], corners[2:]):
            if r > 0:
                self._fillet((ax, ay), (bx, by), (cx, cy), r)
            else:
                self.line_to(bx, by)
        if points:
            self.line_to(*points[-1])
        return self

    def _fillet(self, a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], r: float) -> None:
        """Draws towards corner b and rounds it, leaving the pen where the fillet meets bc."""
        lab, lbc = math.dist(a, b), math.dist(b, c)
        if lab == 0 or lbc == 0:
            self.line_to(*b)
            return
        u = ((b[0] - a[0]) / lab, (b[1] - a[1]) / lab)
        v = ((c[0] - b[0]) / lbc, (c[1] - b[1]) / lbc)
        turn = math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])
        if abs(turn) < 1e-12 or abs(turn) > math.pi - 1e-12:
            self.line_to(*b)
            return
        # the distance from the corner to each tangent point is r tan(turn / 2)
        half = math.tan(abs(turn) / 2)
        reach = min(r * half, lab / 2, lbc / 2)
        r = reach / half
        t1 = (b[0] - reach * u[0], b[1] - reach * u[1])
        side = 1 if turn > 0 else -1  # left turns curve counterclockwise
        centre = (t1[0] - side * r * u[1], t1[1] + side * r * u[0])
        self.line_to(*t1)
        self._arc(centre, r, r, 0.0, math.atan2(t1[1] - centre[1], t1[0] - centre[0]), turn)
        self.current = (b[0] + reach * v[0], b[1] + reach * v[1])

    def _arc(self, centre: tuple[float, float], rx: float, ry: float, phi: float, start: float, span: float) -> PathBuilder:
        """Draws an elliptical arc from its centre form as cubics of at most 90° each.

        the pen must already sit at the arc's start; `phi` rotates the ellipse, and `start` and
        `span` are angles of the unrotated unit circle, in radians.
        """
        n = max(1, math.ceil(abs(span) / _MAX_SPAN - 1e-9))
        step = span / n
        k = 4 / 3 * math.tan(step / 4)
        cos, sin = math.cos(phi), math.sin(phi)

        def point(ux: float, uy: float) -> tuple[float, float]:
            ex, ey = rx * ux, ry * uy
            return centre[0] + cos * ex - sin * ey, centre[1] + sin * ex + cos * ey

        for i in range(n):
            a, b = start + i * step, start + (i + 1) * step
            c1 = point(math.cos(a) - k * math.sin(a), math.sin(a) + k * math.cos(a))
            c2 = point(math.cos(b) + k * math.sin(b), math.sin(b) - k * math.cos(b))
            self.cubic_to(*c1, *c2, *point(math.cos(b), math.sin(b)))
        return self

    def rel_move_to(self, dx: float, dy: float) -> PathBuilder:
        """Starts a new subpath relative to the pen.

        Args:
            dx (float): x offset.
            dy (float): y offset.

        Returns:
            PathBuilder: this builder.
        """
        return self.move_to(self.current[0] + dx, self.current[1] + dy)

    def rel_line_to(self, dx: float, dy: float) -> PathBuilder:
        """Draws a straight segment relative to the pen.

        Args:
            dx (float): x offset of the end.
            dy (float): y offset of the end.

        Returns:
            PathBuilder: this builder.
        """
        return self.line_to(self.current[0] + dx, self.current[1] + dy)

    def rel_cubic_to(
        self, c1dx: float, c1dy: float, c2dx: float, c2dy: float, dx: float, dy: float
    ) -> PathBuilder:
        """Draws a cubic bezier with every point relative to the pen at its start.

        Args:
            c1dx (float): first control x offset.
            c1dy (float): first control y offset.
            c2dx (float): second control x offset.
            c2dy (float): second control y offset.
            dx (float): end x offset.
            dy (float): end y offset.

        Returns:
            PathBuilder: this builder.
        """
        x, y = self.current
        return self.cubic_to(x + c1dx, y + c1dy, x + c2dx, y + c2dy, x + dx, y + dy)

    def rel_quad_to(self, cdx: float, cdy: float, dx: float, dy: float) -> PathBuilder:
        """Draws a quadratic bezier with both points relative to the pen at its start.

        Args:
            cdx (float): control x offset.
            cdy (float): control y offset.
            dx (float): end x offset.
            dy (float): end y offset.

        Returns:
            PathBuilder: this builder.
        """
        x, y = self.current
        return self.quad_to(x + cdx, y + cdy, x + dx, y + dy)

    def rel_elliptical_arc_to(
        self, rx: float | str, ry: float | str, rotation: float, large: bool, sweep: bool, dx: float, dy: float
    ) -> PathBuilder:
        """Draws an elliptical arc to a point relative to the pen, as `elliptical_arc_to`.

        Args:
            rx (float | str): the x radius before rotation.
            ry (float | str): the y radius before rotation.
            rotation (float): the ellipse's x axis angle, in degrees counterclockwise.
            large (bool): take the arc spanning more than 180°.
            sweep (bool): take the arc drawn counterclockwise.
            dx (float): end x offset.
            dy (float): end y offset.

        Returns:
            PathBuilder: this builder.
        """
        x, y = self.current
        return self.elliptical_arc_to(rx, ry, rotation, large, sweep, x + dx, y + dy)

    def close(self) -> PathBuilder:
        """Closes the subpath back to its start.

        Returns:
            PathBuilder: this builder.
        """
        self.commands.append(("Z",))
        self.current = self._start
        return self

    def path(self, **style: Any) -> Path:
        """The finished path node.

        Args:
            **style (Any): `Path` keyword fields such as fill, stroke, id or alpha.

        Returns:
            Path: the node.
        """
        return Path(tuple(self.commands), **style)
