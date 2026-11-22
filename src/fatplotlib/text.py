"""the typst bridge: typesets markup into svg glyphs plus the metrics layout needs.

one `typst.Compiler` is reused (a fresh one per call costs ~180 ms) and results are cached by
(markup, size, font). the text box runs from the font's ascender to its descender, so every
string at one size has the same box unless math makes it taller. the baseline comes from a
zero-width marker glyph set before the markup, so it is exact even for tall math. the cap
height comes from one extra compile per (size, font), so labels can hang from their capitals.
"""

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import typst

_PACKAGE = Path(__file__).parent
PREAMBLE = (_PACKAGE / "typst" / "preamble.typ").read_text()
FONTS = _PACKAGE / "fonts" / "newcm"

# fill for the text itself, removed from the output so the glyphs inherit the node's fill
SENTINEL = "#010203"
# fill for the baseline marker glyph, removed from the output once measured
MARKER = "#030201"

_COMPILER = typst.Compiler(font_paths=[str(FONTS)], ignore_system_fonts=True)
_VIEWBOX = re.compile(r'viewBox="0 0 ([\d.]+) ([\d.]+)"')
# the marker box: an optional translate (when tall math moves the line down), then the glyph
_MARKER_GLYPH = re.compile(
    r'<g(?: transform="translate\([-\d.]+ ([-\d.]+)\)")?>'
    rf'<g transform="matrix\(1 0 0 -1 [-\d.]+ ([-\d.]+)\)"><use xlink:href="#([^"]+)"[^>]*fill="{MARKER}"[^>]*/></g></g>'
)
_DEF = re.compile(r'<(\w+) id="([^"]+)".*?</\1>', re.DOTALL)


@dataclass(frozen=True, slots=True)
class TextShape:
    """Typeset markup.

    Attributes:
        body (str): svg elements in y-down pt with the box's top-left at the origin. glyphs in
            the text colour carry no fill, so they inherit it from an enclosing group, and
            strokes in it (fraction bars) use `currentColor`, the group's `color`.
        defs (tuple[tuple[str, str], ...]): (id, element) pairs the body references. typst
            hashes glyph ids from their outlines, so equal ids are equal glyphs.
        width (float): box width in pt.
        height (float): box height in pt.
        baseline (float): distance from the top of the box down to the first baseline in pt.
        cap (float): distance from the top of the box down to the font's cap height in pt,
            where the tops of capitals and digits sit.
    """

    body: str
    defs: tuple[tuple[str, str], ...]
    width: float
    height: float
    baseline: float
    cap: float


def _source(markup: str, size: float, font: str | None, top_edge: str, bottom_edge: str) -> str:
    """A typst document holding only the markup, on a page that fits it, with no background."""
    font_rule = "" if font is None else f'#set text(font: "{font}")\n'
    return (
        f"{PREAMBLE}\n"
        "#set page(width: auto, height: auto, margin: 0pt, fill: none)\n"
        f'#set text(size: {size}pt, top-edge: "{top_edge}", bottom-edge: "{bottom_edge}", fill: rgb("{SENTINEL}"))\n'
        f"{font_rule}"
        f"{markup}"
    )


def _compile(source: str) -> str:
    """The svg of a one-page typst document."""
    output = _COMPILER.compile(input=source.encode(), format="svg")
    # one page in, so one svg out
    assert isinstance(output, bytes)
    return output.decode()


def _size(svg: str) -> tuple[float, float]:
    """Width and height of a typst svg page in pt."""
    # typst always writes a viewBox
    viewbox = _VIEWBOX.search(svg)
    assert viewbox is not None
    width, height = map(float, viewbox.groups())
    return width, height


@functools.cache
def _cap_height(size: float, font: str | None) -> float:
    """Height of the font's capitals above the baseline in pt: a box from cap height to baseline."""
    return _size(_compile(_source("x", size, font, "cap-height", "baseline")))[1]


@functools.cache
def typeset(markup: str, size: float, font: str | None = None) -> TextShape:
    """Typesets typst markup on a page that fits it, with no background.

    Args:
        markup (str): typst markup, e.g. "Number of Photons" or "$10^(-2)$".
        size (float): font size in pt.
        font (str | None): font family, the preamble's if None.

    Returns:
        TextShape: the glyphs and metrics.
    """
    marked = f'#box(width: 0pt, text(fill: rgb("{MARKER}"))[x]);{markup}'
    svg = _compile(_source(marked, size, font, "ascender", "descender"))
    width, height = _size(svg)
    # the marker is always set, so it is always found
    marker = _MARKER_GLYPH.search(svg)
    assert marker is not None
    baseline, marker_id = float(marker[1] or 0) + float(marker[2]), marker[3]
    svg = svg[: marker.start()] + svg[marker.end() :]

    start = svg.index(">", svg.index("<svg")) + 1
    end = svg.find("<defs>")
    body = svg[start : svg.rindex("</svg>") if end < 0 else end]
    body = body.replace(f' fill="{SENTINEL}"', "").replace(f'stroke="{SENTINEL}"', 'stroke="currentColor"')
    defs = tuple(
        (m[2], m[0])
        for m in _DEF.finditer(svg[end:] if end >= 0 else "")
        if m[2] != marker_id or f"#{marker_id}" in body
    )
    return TextShape(body, defs, width, height, baseline, baseline - _cap_height(size, font))
