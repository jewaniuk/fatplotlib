"""theme resolution and themed rendering."""

import pytest

from fatplotlib.render.svg import render
from fatplotlib.scene import Group, Rect, Text
from fatplotlib.style import Stroke, Token
from fatplotlib.theme import DARK, LIGHT


def scene() -> Group:
    """A box drawn entirely in tokens, with default-coloured text."""
    box = Rect(0, 0, 20, 10, fill=Token("grid"), stroke=Stroke(Token("fg"), Token("thin")), alpha=Token("wisp"))
    return Group([box, Text("x", 10, 5, size=8)])


def test_resolve_and_with():
    pink = DARK.with_(accent_pink="#f2a3c4")
    assert pink.colour(Token("accent_pink")) == "#f2a3c4"
    assert pink.colour(Token("fg")) == DARK.colour(Token("fg"))
    assert pink.name == "dark" and DARK.with_("slides").name == "slides"
    assert LIGHT.number(0.5) == 0.5
    with pytest.raises(KeyError, match="accent_pink"):
        DARK.colour(Token("accent_pink"))
    with pytest.raises(TypeError, match="thin"):
        DARK.colour(Token("thin"))


def test_preview_keeps_tokens_as_css_variables():
    svg = render(scene(), 20, 10)
    for declaration in ("fill:var(--grid)", "stroke:var(--fg)", "stroke-width:var(--thin)", "opacity:var(--wisp)"):
        assert declaration in svg
    assert "color:var(--fg)" in svg


def test_text_recolours_per_theme():
    themes = [theme.with_(wisp=0.3) for theme in (LIGHT, DARK)]
    light, dark = (render(scene(), 20, 10, theme) for theme in themes)
    assert "var(--" not in light + dark
    assert 'fill="#000000" color="#000000"' in light
    assert 'fill="#ffffff" color="#ffffff"' in dark
    assert 'stroke-width="0.4"' in dark and 'opacity="0.3"' in dark


def test_geometry_does_not_depend_on_theme():
    box = scene().children[0]
    assert box.bbox() == Rect(0, 0, 20, 10, stroke=Stroke("#000000", 0.4)).bbox()
