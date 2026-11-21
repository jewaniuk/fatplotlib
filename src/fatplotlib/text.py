"""the typst bridge: typesets markup into svg glyphs plus the metrics layout needs.

one `typst.Compiler` is reused (a fresh one per call costs ~180 ms) and results are cached by
(markup, size, font). the text box runs from the font's ascender to its descender, so every
string at one size has the same box unless math makes it taller. the baseline comes from a
zero-width marker glyph set before the markup, so it is exact even for tall math.
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
    """

    body: str
    defs: tuple[tuple[str, str], ...]
    width: float
    height: float
    baseline: float


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
    font_rule = "" if font is None else f'#set text(font: "{font}")\n'
    source = (
        f"{PREAMBLE}\n"
        "#set page(width: auto, height: auto, margin: 0pt, fill: none)\n"
        f'#set text(size: {size}pt, top-edge: "ascender", bottom-edge: "descender", fill: rgb("{SENTINEL}"))\n'
        f"{font_rule}"
        f'#box(width: 0pt, text(fill: rgb("{MARKER}"))[x]);{markup}'
    )
    output = _COMPILER.compile(input=source.encode(), format="svg")
    # one page in, so one svg out; typst's own output, so the patterns below always match
    assert isinstance(output, bytes)
    svg = output.decode()

    viewbox, marker = _VIEWBOX.search(svg), _MARKER_GLYPH.search(svg)
    assert viewbox is not None and marker is not None
    width, height = map(float, viewbox.groups())
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
    return TextShape(body, defs, width, height, baseline)
