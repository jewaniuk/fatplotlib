"""typst text metrics."""

from fatplotlib.text import typeset


def test_baseline_and_height_equal_for_x_g_d():
    shapes = [typeset(s, 8) for s in ("x", "g", "d")]
    assert len({s.baseline for s in shapes}) == 1
    assert len({s.height for s in shapes}) == 1
    assert len({s.width for s in shapes}) == 3


def test_tall_math_keeps_text_baseline():
    # a display fraction grows the box; the baseline still lines up with plain text's
    plain, tall = typeset("x", 8), typeset("$display(1/2)$", 8)
    assert tall.height > plain.height
    assert tall.baseline > plain.baseline


def test_cap_height_sits_between_top_and_baseline_for_any_string():
    shapes = [typeset(s, 8) for s in ("x", "g", "$10^(-3)$")]
    plain = shapes[0]
    assert 0 < plain.cap < plain.baseline
    # the cap line keeps its distance above the baseline even when math grows the box
    assert len({round(s.baseline - s.cap, 9) for s in shapes}) == 1
