"""colormaps, norms and oklab conversion."""

import numpy as np
import pytest

from fatplotlib.colormap import BAD, DEFAULT, OVER, UNDER, Colormap, Norm
from fatplotlib.style import Mix, Token, oklab_to_srgb, srgb_to_oklab, to_rgba
from fatplotlib.theme import DARK, LIGHT


def lightness(colours: np.ndarray) -> np.ndarray:
    """Oklab lightness of rgba rows."""
    return srgb_to_oklab(colours[..., :3])[..., 0]


def test_oklab_round_trip():
    rgb = np.random.default_rng(0).random((100, 3))
    assert np.allclose(oklab_to_srgb(srgb_to_oklab(rgb)), rgb, atol=1e-7)
    assert np.allclose(srgb_to_oklab(np.array([1.0, 1.0, 1.0])), [1, 0, 0], atol=1e-6)


def test_stops_blend_in_either_space():
    oklab = Colormap("#000000", "#ffffff", levels=3)
    srgb = Colormap("#000000", "#ffffff", levels=3, space="srgb")
    assert oklab.entries() == ["#000000", "#636363", "#ffffff"]
    assert srgb.entries()[1] == "#808080"
    # positioned stops: a quarter of the way the map is already at the middle stop
    placed = Colormap((0, "#000000"), (0.25, "#ff0000"), (1, "#ffffff"), levels=5, space="srgb")
    assert placed.entries()[1] == "#ff0000"


def test_levels_index_like_matplotlib():
    cmap = Colormap("#000000", "#ffffff", levels=4)
    assert cmap.size == 4 and len(cmap.lut(LIGHT)) == 4
    assert cmap.index([0, 0.24, 0.25, 0.99, 1]).tolist() == [0, 0, 1, 3, 3]
    assert Colormap("#000000", "#ffffff").size == 256


def test_bad_under_and_over():
    cmap = Colormap("#000000", "#ffffff", levels=4, over="#ff0000")
    index = cmap.index([np.nan, -0.1, 1.1, -np.inf]).tolist()
    assert index == [BAD, UNDER, OVER, UNDER]
    assert cmap.paint(BAD) is None
    assert cmap.paint(UNDER) == "#000000" and cmap.paint(OVER) == "#ff0000"


def test_log_norm_and_boundaries():
    norm = Norm("log", (1, 100))
    assert np.allclose(norm.fraction([1, 10, 100]), [0, 0.5, 1])
    assert np.isnan(norm.fraction([0]))[0]
    stepped = Norm("linear", (0, 6), boundaries=[0, 1, 3, 6])
    f = stepped.fraction([0, 2, 3, 6, -1, 7, np.nan])
    assert np.allclose(f[:4], [1 / 6, 0.5, 5 / 6, 5 / 6])
    assert f[4] < 0 and f[5] > 1 and np.isnan(f[6])
    # a level per bin gives each bin its own colour
    assert Colormap("#000000", "#ffffff", levels=3).index(f[:4]).tolist() == [0, 1, 2, 2]


def test_token_stops_resolve_per_theme():
    cmap = Colormap(Token("bg"), "#fc69a0", levels=3)
    assert cmap.entries()[0] == Token("bg") and cmap.entries()[1] == Mix(Token("bg"), "#fc69a0", 0.5)
    assert np.allclose(cmap.lut(LIGHT)[0], to_rgba("#ffffff"))
    assert np.allclose(cmap.lut(DARK)[0], to_rgba("#1e1e1e"))
    assert not np.allclose(cmap.lut(LIGHT)[1], cmap.lut(DARK)[1])


def test_per_theme_maps():
    both = Colormap.per_theme(light=Colormap("#000000", "#ff0000"), dark=Colormap("#ffffff", "#00ff00"))
    assert np.allclose(both.lut(DARK)[0], [1, 1, 1, 1]) and np.allclose(both.lut(LIGHT)[0], [0, 0, 0, 1])
    with pytest.raises(ValueError, match="levels"):
        Colormap.per_theme(light=Colormap("#000000", "#ff0000", levels=3), dark=Colormap("#ffffff", "#00ff00"))


def test_named_maps():
    viridis = Colormap.named("viridis")
    assert viridis.size == 256 and viridis.stops[0][1] == "#440154"
    assert Colormap.named("tab10").size == 10


def test_default_lightness_rises_and_ends_clear_of_both_backgrounds():
    stops = np.array([to_rgba(c) for _, c in DEFAULT.stops if isinstance(c, str)])
    assert np.all(np.diff(lightness(stops)) > 0)
    # across the 256 colours only 8-bit rounding may dip
    assert np.diff(lightness(DEFAULT.lut(LIGHT))).min() > -1 / 255
    ends = lightness(DEFAULT.lut(LIGHT)[[0, -1]])
    for theme in (LIGHT, DARK):
        bg = lightness(to_rgba(theme.colour(Token("bg"))))
        assert np.all(np.abs(ends - bg) > 0.12)
