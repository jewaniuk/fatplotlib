"""export: svg, pdf and png files per theme, with and without a background."""

import struct
from pathlib import Path

import pytest

from fatplotlib.axes import Axes
from fatplotlib.layout import Figure, Target
from fatplotlib.render.export import export, save
from fatplotlib.style import Token
from fatplotlib.theme import DARK, LIGHT

PAPER = Target(columnwidth="3.4in", base="9texpt")


def figure() -> Figure:
    """A one-axes figure, 3.4 in by 1.7 in."""
    fig = Figure(height="1.7in")
    fig.layout(Axes(xlim=(0, 1), ylim=(0, 1), xlabel="Length", ylabel="Rate"))
    return fig


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height in pixels from a png's header."""
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def test_both_themes_write_one_file_each(tmp_path: Path):
    written = save(figure(), tmp_path / "fig.svg", PAPER)
    assert written == [tmp_path / "fig.light.svg", tmp_path / "fig.dark.svg"]
    # literal colours, no css variables
    assert all("var(--" not in path.read_text() for path in written)
    assert written[0].read_text() != written[1].read_text()


def test_one_theme_writes_the_path_as_given(tmp_path: Path):
    assert save(figure(), tmp_path / "fig.svg", PAPER, themes="dark") == [tmp_path / "fig.svg"]


def test_background_resolves_per_theme(tmp_path: Path):
    light, dark = save(figure(), tmp_path / "fig.svg", PAPER, background=Token("bg"))
    assert LIGHT.colour(Token("bg")) in light.read_text()
    assert DARK.colour(Token("bg")) in dark.read_text()


def test_pdf_is_vector_and_page_sized():
    root, width, height = figure().scene(PAPER)
    pdf = export(root, width, height, LIGHT, "pdf")
    assert pdf.startswith(b"%PDF")
    assert b"/Subtype /Image" not in pdf and b"/Subtype/Image" not in pdf
    assert f"{width:g} {height:g}".encode() in pdf


def test_png_size_follows_dpi():
    root, width, height = figure().scene(PAPER)
    assert png_size(export(root, width, height, LIGHT, "png", dpi=300)) == (1020, 510)
    assert png_size(export(root, width, height, LIGHT, "png", dpi=150)) == (510, 255)


def test_unknown_format_raises():
    root, width, height = figure().scene(PAPER)
    with pytest.raises(ValueError):
        export(root, width, height, LIGHT, "eps")
