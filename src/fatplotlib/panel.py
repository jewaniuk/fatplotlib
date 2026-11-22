"""panels: free-form drawings with a fixed aspect, fit and centred in the space they are given.

a panel's coordinates run 0 to 100 along its longer side, y-up, with the origin at its bottom-left.
its content comes from a function called once the panel is placed, with a `Frame` that maps panel
coordinates to pt and resolves fixed sizes (pt, em). positions scale with the panel while strokes,
text and fixed radii keep their size.
"""

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from fatplotlib.scene import BBox, Group, Node
from fatplotlib.units import length

# the share of its space a fitted panel may leave empty along one side before it warns
_MAX_GAP = 0.1


@dataclass(frozen=True, slots=True)
class Frame:
    """A placed panel, as its drawing function sees it.

    Attributes:
        box (BBox): the panel's box in pt.
        scale (float): pt per panel unit.
        em (float): the font size in pt that em lengths scale.
    """

    box: BBox
    scale: float
    em: float

    def pt(self, x: float, y: float) -> tuple[float, float]:
        """Maps a point in panel units to pt.

        Args:
            x (float): x, 0 at the panel's left.
            y (float): y, 0 at the panel's bottom.

        Returns:
            tuple[float, float]: the point in pt.
        """
        return self.box.x0 + x * self.scale, self.box.y0 + y * self.scale

    def length(self, value: float | str) -> float:
        """A size in pt: numbers are panel units and scale with the panel, strings are fixed
        lengths such as "0.2em" or "1.5pt".

        Args:
            value (float | str): the size.

        Returns:
            float: the size in pt.
        """
        return value * self.scale if not isinstance(value, str) else length(value, self.em)


class Panel:
    """A drawing with a fixed aspect, placed by layout or as an inset.

    Attributes:
        aspect (float): width / height.
        draw (Callable[[Frame], Iterable[Node]]): builds the content in pt from the placed frame.
        label (str | bool | None): the panel label: None letters it automatically, False skips
            it, a string replaces the letter.
        box (BBox | None): the fitted box in pt, None until placed.
        em (float | None): the font size in pt that em lengths scale, None until placed.
    """

    def __init__(self, aspect: float, draw: Callable[[Frame], Iterable[Node]], *, label: str | bool | None = None) -> None:
        """Sets up the panel.

        Args:
            aspect (float): width / height.
            draw (Callable[[Frame], Iterable[Node]]): builds the content.
            label (str | bool | None): the panel label.
        """
        self.aspect = aspect
        self.draw = draw
        self.label = label
        self.box: BBox | None = None
        self.em: float | None = None

    def place(self, slot: BBox, em: float) -> None:
        """Fits the panel in a slot, centred, warning when it leaves much of the slot empty.

        Args:
            slot (BBox): the space in pt.
            em (float): the font size in pt that em lengths scale.
        """
        width = min(slot.width, slot.height * self.aspect)
        height = width / self.aspect
        filled = width * height / (slot.width * slot.height) if slot.width * slot.height > 0 else 1.0
        if filled < 1 - _MAX_GAP:
            warnings.warn(f"a panel with aspect {self.aspect:.3g} fills {filled:.0%} of its space", stacklevel=2)
        x0 = slot.x0 + (slot.width - width) / 2
        y0 = slot.y0 + (slot.height - height) / 2
        self.box = BBox(x0, y0, x0 + width, y0 + height)
        self.em = em

    def frame(self) -> Frame:
        """The placed panel for drawing.

        Returns:
            Frame: the box, scale and font size.

        Raises:
            RuntimeError: the panel has not been placed.
        """
        if self.box is None or self.em is None:
            raise RuntimeError("the panel has no box yet: put it in a Figure layout or an inset")
        return Frame(self.box, max(self.box.width, self.box.height) / 100, self.em)

    def group(self) -> Group:
        """The content, drawn into the placed frame.

        Returns:
            Group: the panel as scene nodes, in page pt.

        Raises:
            RuntimeError: the panel has not been placed.
        """
        return Group(list(self.draw(self.frame())))

    def outer(self) -> BBox:
        """The fitted box; panel labels anchor to the drawn content inside it instead.

        Returns:
            BBox: the box in pt.

        Raises:
            RuntimeError: the panel has not been placed.
        """
        return self.frame().box
