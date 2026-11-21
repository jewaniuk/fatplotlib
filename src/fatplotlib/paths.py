"""PathBuilder: the move / line / cubic helpers of planting-seeds' figtools, as one object.

the builder tracks the current point, so the relative commands need no start point and the
absolute ones return nothing to thread through.
"""

from typing import Any

from fatplotlib.scene import Command, Path


class PathBuilder:
    """Builds path commands one segment at a time; every method returns the builder for chaining.

    Attributes:
        commands (list[Command]): the absolute commands so far.
        current (tuple[float, float]): the pen position.
    """

    def __init__(self) -> None:
        """Starts an empty path with the pen at the origin."""
        self.commands: list[Command] = []
        self.current: tuple[float, float] = (0.0, 0.0)
        self._start = self.current

    def move_to(self, x: float, y: float) -> PathBuilder:
        """Starts a new subpath.

        Args:
            x (float): x.
            y (float): y.

        Returns:
            PathBuilder: this builder.
        """
        self.commands.append(("M", x, y))
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
        self.commands.append(("L", x, y))
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
        self.commands.append(("C", c1x, c1y, c2x, c2y, x, y))
        self.current = (x, y)
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
