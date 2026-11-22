"""paints and strokes.

a paint is a css colour string ("#rrggbb", or "#rrggbbaa" for a colour with alpha), a theme
`Token`, a `Mix` of two colours, or a `RadialGradient` or `LinearGradient`.

colours blend in oklab by default, a perceptual space where a mix halfway between two colours
looks halfway between them. "srgb" blends the stored channel values, as matplotlib does.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

type Space = Literal["oklab", "srgb"]
"""the colour space a blend interpolates in."""

# linear srgb -> lms cone response, and cube-rooted lms -> oklab (bjorn ottosson's matrices)
_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_LAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)


@dataclass(frozen=True, slots=True)
class Token:
    """A named value that a theme resolves: a colour (`Token("fg")`), an alpha or a line weight (`Token("thin")`).

    Attributes:
        name (str): the token name.
    """

    name: str


@dataclass(frozen=True, slots=True)
class Stop:
    """A gradient colour stop.

    Attributes:
        offset (float): position along the gradient, 0 to 1.
        colour (str | Token | Mix): the stop colour.
        alpha (float): the stop opacity, 0 to 1.
    """

    offset: float
    colour: str | Token | Mix
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


@dataclass(frozen=True, slots=True)
class LinearGradient:
    """A linear gradient in the shape's bounding box units, like `RadialGradient`. the box is
    measured in scene coordinates, which are y-up, so y = 0 is the box's bottom edge.

    Attributes:
        stops (tuple[Stop, ...]): colour stops from the start (0) to the end (1).
        x1 (float): start x as a fraction of the bounding box width.
        y1 (float): start y as a fraction of the bounding box height, from the bottom.
        x2 (float): end x.
        y2 (float): end y.
    """

    stops: tuple[Stop, ...]
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 0.0


@dataclass(frozen=True, slots=True)
class Mix:
    """Two colours blended, either side a literal or a token, so a mix toward `Token("bg")` follows the theme.

    Attributes:
        a (str | Token): the base colour.
        b (str | Token): the colour mixed in.
        t (float): how much of b, 0 (all a) to 1 (all b).
        space (Space): the colour space the blend interpolates in.
    """

    a: str | Token
    b: str | Token
    t: float
    space: Space = "oklab"


type Paint = str | Token | Mix | RadialGradient | LinearGradient


@dataclass(frozen=True, slots=True)
class Stroke:
    """An outline.

    Attributes:
        paint (Paint): the stroke paint.
        width (float | Token): line width in pt, or a weight token.
        dash (tuple[float, ...] | None): alternating dash and gap lengths in pt, solid if None.
        cap (Literal["butt", "round", "square"]): line ends.
        join (Literal["miter", "round", "bevel"]): line corners.
    """

    paint: Paint
    width: float | Token = 1.0
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


def to_rgba(colour: str) -> np.ndarray:
    """The channels of a "#rrggbb" or "#rrggbbaa" colour.

    Args:
        colour (str): the colour.

    Returns:
        np.ndarray: red, green, blue and alpha, 0 to 1, shape (4,).
    """
    return np.array([int(colour[i : i + 2] or "ff", 16) / 255 for i in (1, 3, 5, 7)])


def to_hex(rgba: np.ndarray) -> str:
    """The css colour of some channels, inverse to `to_rgba`.

    Args:
        rgba (np.ndarray): red, green, blue and optionally alpha, 0 to 1, shape (3,) or (4,).

    Returns:
        str: "#rrggbb", or "#rrggbbaa" if alpha is below 1.
    """
    rgb = "#" + "".join(f"{round(c * 255):02x}" for c in np.clip(rgba[:3], 0, 1).tolist())
    return rgb if len(rgba) == 3 or rgba[3] >= 1 else with_alpha(rgb, float(rgba[3]))


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Converts srgb colours to oklab.

    Args:
        rgb (np.ndarray): srgb channels 0 to 1, shape (..., 3).

    Returns:
        np.ndarray: oklab L, a, b, shape (..., 3).
    """
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.cbrt(linear @ _LMS.T) @ _LAB.T


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Converts oklab colours to srgb, clipped to the srgb gamut.

    Args:
        lab (np.ndarray): oklab L, a, b, shape (..., 3).

    Returns:
        np.ndarray: srgb channels 0 to 1, shape (..., 3).
    """
    linear = np.clip(((lab @ np.linalg.inv(_LAB).T) ** 3) @ np.linalg.inv(_LMS).T, 0, 1)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def mix_colours(a: str, b: str, t: float, space: Space = "oklab") -> str:
    """Blends two literal colours as css `color-mix` does, with alpha premultiplied.

    Args:
        a (str): the base "#rrggbb" or "#rrggbbaa" colour.
        b (str): the colour mixed in.
        t (float): how much of b, 0 to 1.
        space (Space): the colour space the blend interpolates in.

    Returns:
        str: the "#rrggbb" colour, or "#rrggbbaa" if the result is not opaque.
    """
    rgba = np.array([to_rgba(a), to_rgba(b)])
    rgb, alpha = rgba[:, :3], rgba[:, 3:]
    channels = srgb_to_oklab(rgb) if space == "oklab" else rgb
    mixed_alpha = float((1 - t) * alpha[0, 0] + t * alpha[1, 0])
    if mixed_alpha == 0:
        return "#00000000"
    premultiplied = (1 - t) * channels[0] * alpha[0] + t * channels[1] * alpha[1]
    mixed = premultiplied / mixed_alpha
    rgb = oklab_to_srgb(mixed) if space == "oklab" else mixed
    return to_hex(np.append(rgb, mixed_alpha))
