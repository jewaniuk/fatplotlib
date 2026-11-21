"""lengths, in pt (1/72 in) unless named otherwise.

pt is the unit of typst, pdf and svg output. latex sizes are tex points (1/72.27 in),
so latex "9pt" is `9 * TEX_PT` here.
"""

import re

PT = 1.0
INCH = 72.0
MM = INCH / 25.4
TEX_PT = INCH / 72.27

_UNITS = {"pt": PT, "in": INCH, "mm": MM, "texpt": TEX_PT}
_LENGTH = re.compile(r"\s*([-+]?(?:\d+\.?\d*|\.\d+))\s*([a-z]+)\s*")


def length(value: float | str, em: float | None = None) -> float:
    """Converts a length to pt.

    Args:
        value (float | str): a number in pt, or a string such as "3.4in", "8.5mm", "9texpt" or "1.5em".
        em (float | None): the font size in pt that "em" lengths scale.

    Returns:
        float: the length in pt.

    Raises:
        ValueError: if a string length does not parse, names an unknown unit, or is in em with no `em`.
    """
    if not isinstance(value, str):
        return float(value)
    match = _LENGTH.fullmatch(value)
    if match is None or (match[2] not in _UNITS and match[2] != "em"):
        raise ValueError(f"not a length: {value!r}")
    number, unit = float(match[1]), match[2]
    if unit != "em":
        return number * _UNITS[unit]
    if em is None:
        raise ValueError(f"{value!r} needs a font size (em)")
    return number * em
