"""scales: data values -> fractions of an axis, with nice-number ticks and typst tick labels.

ticks follow a nice-number rule: the smallest step of 1, 2, 2.5 or 5 x 10^k (linear), or the
smallest decade stride (log), that leaves at most `max_gaps` gaps between labelled ticks. axes
and colormap norms both map values through a scale.
"""

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

# nice steps for linear ticks, as multiples of a power of ten
_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)
EPS = 1e-9
"""slack for comparing floats that should be whole or equal."""


class Scale:
    """Maps data values to fractions of an axis, 0 at the start of `lim` and 1 at its end.

    Attributes:
        kind (Literal["linear", "log"]): how values are spaced.
        lim (tuple[float, float]): the data values at the start and end of the axis; start >
            end reverses the axis. a zero-width range is widened (linear by 5 %, log by a
            decade each way).
    """

    def __init__(self, kind: Literal["linear", "log"], lim: tuple[float, float]) -> None:
        """Sets up the scale.

        Args:
            kind (Literal["linear", "log"]): how values are spaced.
            lim (tuple[float, float]): the data values at the start and end of the axis.

        Raises:
            ValueError: a log scale has a limit <= 0.
        """
        lo, hi = map(float, lim)
        if kind == "log" and (lo <= 0 or hi <= 0):
            raise ValueError(f"log scale limits must be positive, got {lim}")
        if lo == hi:
            lo, hi = (lo / 10, hi * 10) if kind == "log" else (lo - (0.05 * abs(lo) or 0.05), hi + (0.05 * abs(hi) or 0.05))
        self.kind: Literal["linear", "log"] = kind
        self.lim = (lo, hi)
        self._start, self._end = self._space(np.array(self.lim))

    def _space(self, values: np.ndarray) -> np.ndarray:
        """Values in the space where the scale is linear; log of values <= 0 gives nan or -inf."""
        if self.kind == "linear":
            return values
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(values)

    def fraction(self, values: ArrayLike) -> np.ndarray:
        """Where values sit along the axis.

        Args:
            values (ArrayLike): data values.

        Returns:
            np.ndarray: fractions, 0 at the start of `lim` and 1 at its end; non-finite for
                values the scale cannot map (log of values <= 0).
        """
        return (self._space(np.asarray(values, dtype=float)) - self._start) / (self._end - self._start)

    def ticks(self, max_gaps: int) -> list[float]:
        """Major tick values inside `lim` by the nice-number rule.

        Args:
            max_gaps (int): the most gaps allowed between the first and last tick.

        Returns:
            list[float]: the values, ascending.
        """
        max_gaps = max(max_gaps, 1)
        lo, hi = sorted(self._space(np.array(self.lim)).tolist())
        if self.kind == "log":
            first, last = math.ceil(lo - EPS), math.floor(hi + EPS)
            stride = 1
            while (last - first) // stride > max_gaps:
                stride += 1
            return [10.0**k for k in range(first, last + 1, stride)]
        magnitude = 10 ** math.floor(math.log10((hi - lo) / max_gaps))
        # 20 x magnitude is over twice the even split, so it always fits
        step = next(m * magnitude for m in (*_STEPS, 20.0) if len(_multiples(lo, hi, m * magnitude)) - 1 <= max_gaps)
        # round away float noise (0.1 * 3 = 0.30000000000000004) a digit below the step
        digits = max(0, 1 - math.floor(math.log10(step)))
        return [round(k * step, digits) + 0.0 for k in _multiples(lo, hi, step)]

    def minor_ticks(self, majors: Sequence[float]) -> list[float]:
        """Minor tick values inside `lim` that fit between the majors.

        Linear majors are split in 5 (steps of 1 or 5 x 10^k) or 4 (2 or 2.5 x 10^k). Log majors
        a decade apart get 2 to 9 x each decade; majors further apart get the decades skipped.

        Args:
            majors (Sequence[float]): the major tick values, ascending and evenly spaced.

        Returns:
            list[float]: the values, ascending, with the majors left out.
        """
        lo, hi = sorted(self.lim)
        if self.kind == "log":
            exponents = [math.log10(t) for t in majors]
            if len(majors) > 1 and all(abs(e - round(e)) < EPS for e in exponents) and exponents[1] - exponents[0] > 1.5:
                decades = range(math.ceil(math.log10(lo) - EPS), math.floor(math.log10(hi) + EPS) + 1)
                values = [10.0**k for k in decades if k not in {round(e) for e in exponents}]
            else:
                decades = range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
                # parse "7e-1" rather than multiply, which gives 0.7000000000000001
                values = [float(f"{s}e{k}") for k in decades for s in range(2, 10)]
            return [v for v in values if lo * (1 - EPS) <= v <= hi * (1 + EPS)]
        if len(majors) < 2:
            return []
        step = majors[1] - majors[0]
        mantissa = step / 10 ** math.floor(math.log10(step))
        sub = step / (4 if round(mantissa, 6) in (2.0, 2.5) else 5)
        values = [k * sub for k in _multiples(lo, hi, sub)]
        return [v for v in values if all(abs(v - m) > EPS * step for m in majors)]

    def labels(self, ticks: Sequence[float]) -> list[str]:
        """Typst markup for tick values: `$10^(k)$` for decades on a log scale, else plain numbers
        with as few decimals as every value needs.

        Args:
            ticks (Sequence[float]): the values.

        Returns:
            list[str]: one label per value.
        """
        if self.kind == "log":
            exponents = [math.log10(t) for t in ticks]
            if all(abs(e - round(e)) < EPS for e in exponents):
                return [f"$10^({round(e)})$" for e in exponents]
        decimals = next(
            (d for d in range(10) if all(abs(round(t, d) - t) <= EPS * max(1.0, abs(t)) for t in ticks)), 10
        )
        return [f"${round(t, decimals) + 0.0:.{decimals}f}$" for t in ticks]


def _multiples(lo: float, hi: float, step: float) -> range:
    """The whole numbers k with k * step inside [lo, hi], allowing for float error."""
    return range(math.ceil(lo / step - EPS), math.floor(hi / step + EPS) + 1)
