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
