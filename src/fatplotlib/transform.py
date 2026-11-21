"""affine transforms in the svg matrix order."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Affine:
    """An affine map (x, y) -> (a x + c y + e, b x + d y + f), the svg `matrix(a b c d e f)`.

    `A @ B` applies B first, then A. The default is the identity.

    Attributes:
        a (float): x scale.
        b (float): y shear.
        c (float): x shear.
        d (float): y scale.
        e (float): x translation.
        f (float): y translation.
    """

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    @classmethod
    def translate(cls, dx: float, dy: float) -> Affine:
        """A translation.

        Args:
            dx (float): x offset.
            dy (float): y offset.

        Returns:
            Affine: the translation.
        """
        return cls(e=dx, f=dy)

    @classmethod
    def scale(cls, sx: float, sy: float | None = None) -> Affine:
        """A scale about the origin.

        Args:
            sx (float): x factor.
            sy (float | None): y factor, sx if None.

        Returns:
            Affine: the scale.
        """
        return cls(a=sx, d=sx if sy is None else sy)

    @classmethod
    def rotate(cls, theta: float, about: tuple[float, float] = (0.0, 0.0)) -> Affine:
        """A counterclockwise rotation (in a y-up frame) about a point.

        Args:
            theta (float): angle in radians.
            about (tuple[float, float]): centre of rotation.

        Returns:
            Affine: the rotation.
        """
        cos, sin = math.cos(theta), math.sin(theta)
        x, y = about
        return cls(cos, sin, -sin, cos, x - cos * x + sin * y, y - sin * x - cos * y)

    def __matmul__(self, other: Affine) -> Affine:
        """Composes two transforms, applying `other` first.

        Args:
            other (Affine): the transform applied first.

        Returns:
            Affine: the composition.
        """
        return Affine(
            self.a * other.a + self.c * other.b,
            self.b * other.a + self.d * other.b,
            self.a * other.c + self.c * other.d,
            self.b * other.c + self.d * other.d,
            self.a * other.e + self.c * other.f + self.e,
            self.b * other.e + self.d * other.f + self.f,
        )

    def apply(self, x: float, y: float) -> tuple[float, float]:
        """Maps a point.

        Args:
            x (float): x.
            y (float): y.

        Returns:
            tuple[float, float]: the mapped point.
        """
        return self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f
