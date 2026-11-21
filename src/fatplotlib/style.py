"""paints and strokes.

a paint is a css colour string ("#rrggbb", or "#rrggbbaa" for a colour with alpha), a theme
`Token`, or a `RadialGradient`.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Token:
    """A named colour that a theme resolves (light and dark), e.g. `Token("fg")`.

    Attributes:
        name (str): the token name.
    """

    name: str


@dataclass(frozen=True, slots=True)
class Stop:
    """A gradient colour stop.

    Attributes:
        offset (float): position along the gradient, 0 to 1.
        colour (str | Token): the stop colour.
        alpha (float): the stop opacity, 0 to 1.
    """

    offset: float
    colour: str | Token
    alpha: float = 1.0


@dataclass(frozen=True, slots=True)
class RadialGradient:
    """A radial gradient in the shape's bounding box units, so one gradient serves every shape.

    Attributes:
        stops (tuple[Stop, ...]): colour stops from centre (0) to edge (1).
        cx (float): centre x as a fraction of the bounding box width.
        cy (float): centre y as a fraction of the bounding box height.
        r (float): radius as a fraction of the bounding box size.
    """

    stops: tuple[Stop, ...]
    cx: float = 0.5
    cy: float = 0.5
    r: float = 0.5


type Paint = str | Token | RadialGradient


@dataclass(frozen=True, slots=True)
class Stroke:
    """An outline.

    Attributes:
        paint (Paint): the stroke paint.
        width (float): line width in pt.
        dash (tuple[float, ...] | None): alternating dash and gap lengths in pt, solid if None.
        cap (Literal["butt", "round", "square"]): line ends.
        join (Literal["miter", "round", "bevel"]): line corners.
    """

    paint: Paint
    width: float = 1.0
    dash: tuple[float, ...] | None = None
    cap: Literal["butt", "round", "square"] = "butt"
    join: Literal["miter", "round", "bevel"] = "miter"


def with_alpha(colour: str, alpha: float) -> str:
    """Adds an alpha channel to a "#rrggbb" colour.

    Args:
        colour (str): a "#rrggbb" colour.
        alpha (float): opacity, 0 to 1.

    Returns:
        str: the "#rrggbbaa" colour.
    """
    return f"{colour[:7]}{round(alpha * 255):02x}"
