"""figure -> svg, pdf and png files, one per theme.

every format starts from the themed svg: pdf wraps it in a typst page with `#image`, so it stays
vector, and png rasterises it with resvg. the layout is solved once and only emission runs per
theme. the page is transparent unless a background paint (a token such as "bg", or a literal
colour) is given, which becomes a page-sized rectangle under the figure.
"""

from pathlib import Path

import resvg_py

from fatplotlib.layout import Figure, Target
from fatplotlib.render.svg import render
from fatplotlib.scene import Group, Node, Rect
from fatplotlib.style import Token
from fatplotlib.text import COMPILER
from fatplotlib.theme import DARK, LIGHT, Theme

FORMATS = ("svg", "pdf", "png")

# a page the size of the svg with the svg as its only content
_PDF_PAGE = (
    "#set page(width: {width}pt, height: {height}pt, margin: 0pt, fill: none)\n"
    '#image(bytes(sys.inputs.svg), format: "svg", width: 100%)'
)


def save(
    fig: Figure,
    path: Path,
    target: Target | None = None,
    themes: str | Theme = "both",
    dpi: float = 300,
    background: str | Token | None = None,
) -> list[Path]:
    """Solves a figure for a target and writes it in the format the path's suffix names.

    Args:
        fig (Figure): the figure.
        path (Path): where to write, ending in .svg, .pdf or .png. with themes="both" the theme
            name goes before the suffix: fig.pdf writes fig.light.pdf and fig.dark.pdf.
        target (Target | None): the page widths, font size and labels, the default if None.
        themes (str | Theme): "both", "light", "dark" or a theme; one theme writes path as is.
        dpi (float): png resolution in pixels per inch.
        background (str | Token | None): the page paint, transparent if None.

    Returns:
        list[Path]: the files written.

    Raises:
        ValueError: the suffix is not a known format.
    """
    root, width, height = fig.scene(target or Target())
    if background is not None:
        root = Group([Rect(0, 0, width, height, fill=background), root])
    if themes == "both":
        outputs = [(theme, path.with_suffix(f".{theme.name}{path.suffix}")) for theme in (LIGHT, DARK)]
    else:
        outputs = [({"light": LIGHT, "dark": DARK}[themes] if isinstance(themes, str) else themes, path)]
    for theme, out in outputs:
        out.write_bytes(export(root, width, height, theme, out.suffix.removeprefix("."), dpi))
    return [out for _, out in outputs]


def export(root: Node, width: float, height: float, theme: Theme, format: str, dpi: float = 300) -> bytes:
    """Renders a scene to the bytes of one file.

    Args:
        root (Node): the scene, y-up, with the origin at the bottom-left of the page.
        width (float): page width in pt.
        height (float): page height in pt.
        theme (Theme): resolves tokens to literals.
        format (str): "svg", "pdf" or "png".
        dpi (float): png resolution in pixels per inch.

    Returns:
        bytes: the file contents.

    Raises:
        ValueError: the format is not known.
    """
    svg = render(root, width, height, theme)
    match format:
        case "svg":
            return svg.encode()
        case "pdf":
            source = _PDF_PAGE.format(width=width, height=height)
            pdf = COMPILER.compile(input=source.encode(), format="pdf", sys_inputs={"svg": svg})
            # one page in, so one pdf out
            assert isinstance(pdf, bytes)
            return pdf
        case "png":
            # resvg reads the pt page size at this dpi
            return resvg_py.svg_to_bytes(svg_string=svg, dpi=dpi, skip_system_fonts=True)
        case _:
            raise ValueError(f"cannot export {format!r}, choose from {', '.join(FORMATS)}")
