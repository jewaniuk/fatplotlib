"""scales: nice-number ticks, minor ticks and labels."""

import pytest

from fatplotlib.scale import Scale


def test_linear_ticks_take_the_smallest_nice_step_that_fits():
    assert Scale("linear", (0, 3995)).ticks(8) == [500.0 * k for k in range(8)]
    assert Scale("linear", (0, 1)).ticks(4) == [0, 0.25, 0.5, 0.75, 1]


def test_linear_ticks_have_no_float_noise():
    assert Scale("linear", (0, 0.7)).ticks(7) == [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert Scale("linear", (0, 0.7)).labels([0, 0.1, 0.3]) == ["$0.0$", "$0.1$", "$0.3$"]


def test_reversed_negative_and_zero_width_ranges():
    assert Scale("linear", (1, -1)).ticks(4) == [-1, -0.5, 0, 0.5, 1]
    assert Scale("linear", (1, -1)).fraction([1, -1]).tolist() == [0, 1]
    assert Scale("linear", (-0.5, 0)).labels([-0.5, 0]) == ["$-0.5$", "$0.0$"]
    assert Scale("linear", (0, 0)).lim == (-0.05, 0.05)
    assert Scale("log", (10, 10)).lim == (1, 100)


def test_log_ticks_stride_from_the_lowest_decade():
    scale = Scale("log", (1e-3, 1e8))
    majors = scale.ticks(5)
    assert majors == [10.0**k for k in range(-3, 8, 2)]
    assert scale.labels(majors)[:2] == ["$10^(-3)$", "$10^(-1)$"]
    assert scale.minor_ticks(majors) == [10.0**k for k in range(-2, 9, 2)]


def test_log_minors_fill_decades_when_majors_are_adjacent():
    scale = Scale("log", (0.7, 1.3e3))
    majors = scale.ticks(5)
    assert majors == [1, 10, 100, 1000]
    minors = scale.minor_ticks(majors)
    assert minors[0] == 0.7 and minors[-1] == 900
    assert 2 in minors and 10 not in minors


def test_linear_minors_split_steps_in_four_or_five():
    assert Scale("linear", (0, 10)).minor_ticks([0, 5, 10]) == [1, 2, 3, 4, 6, 7, 8, 9]
    assert Scale("linear", (0, 4)).minor_ticks([0, 2, 4]) == [0.5, 1, 1.5, 2.5, 3, 3.5]


def test_log_scale_needs_positive_limits():
    with pytest.raises(ValueError, match="positive"):
        Scale("log", (0, 10))
