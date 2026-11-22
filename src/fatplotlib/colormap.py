"""colormaps and norms: data values -> colours.

a `Norm` maps values to fractions of the map (0 to 1) through a `Scale`, and a `Colormap` turns
fractions into indices into its lut (look-up table): N colours sampled evenly from its stops,
256 unless `levels` says otherwise. indices below 0 are sentinels for values that are bad (nan),
under the map or over it.

each lut entry is a paint, so colours resolve per theme like any other. between two literal
stops an entry is a literal colour; next to a token stop it is a `Mix` of the two stops, which
export resolves for whatever theme it writes and the preview emits as css `color-mix`. a map
chosen per theme (`Colormap.per_theme`) is the one case drawing has to split by theme, with a
`scene.Themed` node.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from fatplotlib.scale import Scale
from fatplotlib.style import Mix, Space, Token, mix_colours, oklab_to_srgb, to_hex, to_rgba
from fatplotlib.theme import Theme

type Colour = str | Token
"""a literal colour or a colour token."""

type ColourStop = Colour | tuple[float, Colour]
"""a stop colour, optionally with its position along the map (0 to 1)."""

BAD = -1
"""the index of values that are not finite, or that the norm cannot map."""
UNDER = -2
"""the index of values below the norm's range."""
OVER = -3
"""the index of values above the norm's range."""


class Colormap:
    """Stops blended into a lut of colours.

    Attributes:
        stops (tuple[tuple[float, Colour], ...]): (position, colour) pairs, positions ascending
            from 0 to 1. literal stops give the same map in every theme, token stops follow it.
        space (Space): the colour space stops blend in.
        levels (int | None): the number of lut colours, 256 if None; a few levels give a stepped map.
        bad (Colour | None): the colour of bad values, not drawn if None.
        under (Colour | None): the colour of values under the range, the first lut colour if None.
        over (Colour | None): the colour of values over the range, the last lut colour if None.
        themes (dict[str, Colormap]): theme name -> the map used in that theme, empty unless built
            with `per_theme`.
    """

    def __init__(
        self,
        *stops: ColourStop,
        space: Space = "oklab",
        levels: int | None = None,
        bad: Colour | None = None,
        under: Colour | None = None,
        over: Colour | None = None,
    ) -> None:
        """Sets up the map.

        Args:
            *stops (ColourStop): two or more colours, spread evenly, or (position, colour) pairs;
                either every stop has a position or none does.
            space (Space): the colour space stops blend in.
            levels (int | None): the number of lut colours, 256 if None.
            bad (Colour | None): the colour of bad values.
            under (Colour | None): the colour of values under the range.
            over (Colour | None): the colour of values over the range.
        """
        if all(isinstance(stop, tuple) for stop in stops):
            positioned = stops
        else:
            positioned = zip(np.linspace(0, 1, len(stops)).tolist(), stops)
        self.stops: tuple[tuple[float, Colour], ...] = tuple((float(p), c) for p, c in positioned)  # pyright: ignore[reportAssignmentType]
        self.space: Space = space
        self.levels = levels
        self.bad = bad
        self.under = under
        self.over = over
        self.themes: dict[str, Colormap] = {}
        self._entries: list[Colour | Mix] | None = None

    @classmethod
    def per_theme(cls, *, light: Colormap, dark: Colormap) -> Colormap:
        """A map that is `light` in the light theme and `dark` in the dark one.

        Args:
            light (Colormap): the light theme's map.
            dark (Colormap): the dark theme's map.

        Returns:
            Colormap: the combined map, which acts as `light` outside a theme.

        Raises:
            ValueError: the maps have different numbers of levels, so an index would mean
                different values in each theme.
        """
        if light.size != dark.size:
            raise ValueError(f"per-theme maps need the same levels, got {light.size} and {dark.size}")
        cmap = cls(*light.stops, space=light.space, levels=light.levels, bad=light.bad, under=light.under, over=light.over)
        cmap.themes = {"light": light, "dark": dark}
        return cmap

    @classmethod
    def named(cls, name: str, levels: int | None = None) -> Colormap:
        """A published map by name, from the `cmap` catalogue ("viridis", "cmocean:thermal",
        "tab10"). its colours are literal, so it is the same in every theme.

        Args:
            name (str): the catalogue name.
            levels (int | None): the number of lut colours, 256 if None; a categorical map
                ("tab10") always has one level per colour.

        Returns:
            Colormap: the map, its stops already sampled in srgb.
        """
        # the only use of the package, so it loads when a named map is asked for
        import cmap

        source = cmap.Colormap(name)
        rows = source.lut(256)
        categorical = source.interpolation == "nearest"

        def colour(c: cmap.Color | None) -> str | None:
            return None if c is None else to_hex(np.array(c.rgba))

        return cls(
            *(to_hex(row) for row in rows),
            space="srgb",
            levels=len(rows) if categorical else levels,
            bad=colour(source.bad_color),
            under=colour(source.under_color),
            over=colour(source.over_color),
        )

    @property
    def size(self) -> int:
        """int: the number of lut colours."""
        return self.levels or 256

    def entries(self) -> list[Colour | Mix]:
        """The lut as paints, sampled evenly from the first stop to the last.

        Returns:
            list[Colour | Mix]: `size` paints: literal colours between literal stops, mixes next
                to token stops.
        """
        if self._entries is None:
            self._entries = [self._at(f) for f in np.linspace(0, 1, self.size).tolist()]
        return self._entries

    def _at(self, fraction: float) -> Colour | Mix:
        """The paint a fraction of the way along the stops."""
        positions = [p for p, _ in self.stops]
        k = min(max(int(np.searchsorted(positions, fraction, side="right")) - 1, 0), len(self.stops) - 2)
        (p0, a), (p1, b) = self.stops[k], self.stops[k + 1]
        t = 1.0 if p1 == p0 else (fraction - p0) / (p1 - p0)
        if t <= 0:
            return a
        if t >= 1:
            return b
        if isinstance(a, str) and isinstance(b, str):
            return mix_colours(a, b, t, self.space)
        return Mix(a, b, t, self.space)

    def index(self, fractions: ArrayLike) -> np.ndarray:
        """The lut index of each fraction, as matplotlib picks it: fraction x size, rounded down.

        Args:
            fractions (ArrayLike): fractions from a `Norm`.

        Returns:
            np.ndarray: integer indices, `UNDER` below 0, `OVER` above 1 and `BAD` for nan.
        """
        f = np.asarray(fractions, dtype=float)
        with np.errstate(invalid="ignore"):
            index = np.clip(np.floor(f * self.size), 0, self.size - 1)
        index = np.where(f < 0, UNDER, np.where(f > 1, OVER, index))
        return np.where(np.isnan(f), BAD, index).astype(int)

    def paint(self, index: int) -> Colour | Mix | None:
        """The paint for one lut index or sentinel.

        Args:
            index (int): from `index`.

        Returns:
            Colour | Mix | None: the paint, None for bad values with no `bad` colour.
        """
        if index == BAD:
            return self.bad
        if index == UNDER:
            return self.entries()[0] if self.under is None else self.under
        if index == OVER:
            return self.entries()[-1] if self.over is None else self.over
        return self.entries()[index]

    def lut(self, theme: Theme) -> np.ndarray:
        """The lut's colours in one theme.

        Args:
            theme (Theme): resolves token stops and picks the per-theme map.

        Returns:
            np.ndarray: red, green, blue and alpha, 0 to 1, shape (size, 4).

        Raises:
            KeyError: the map is per theme and has no map for this theme's name.
        """
        if self.themes:
            return self.themes[theme.name].lut(theme)
        return np.array([to_rgba(theme.colour(entry)) for entry in self.entries()])


class Norm:
    """Maps data values to colormap fractions.

    Attributes:
        scale (Scale): maps values in `lim` to 0 to 1, linear or log.
        boundaries (np.ndarray | None): bin edges, ascending, for a stepped map: each bin gets
            the middle of an equal share of the map, so a map with one level per bin gives each
            bin its own colour. the scale still spaces them on a colorbar.
    """

    def __init__(
        self,
        kind: Literal["linear", "log"],
        lim: tuple[float, float],
        boundaries: Sequence[float] | None = None,
    ) -> None:
        """Sets up the norm.

        Args:
            kind (Literal["linear", "log"]): how values are spaced.
            lim (tuple[float, float]): the values at the start and end of the map.
            boundaries (Sequence[float] | None): bin edges for a stepped map.

        Raises:
            ValueError: a log norm has a limit <= 0.
        """
        self.scale = Scale(kind, lim)
        self.boundaries = None if boundaries is None else np.asarray(boundaries, dtype=float)

    def fraction(self, values: ArrayLike) -> np.ndarray:
        """Where values sit along the map.

        Args:
            values (ArrayLike): data values.

        Returns:
            np.ndarray: fractions, below 0 or above 1 outside the range, nan where the norm
                cannot map a value (nan, or <= 0 on a log norm, as matplotlib masks them).
        """
        v = np.asarray(values, dtype=float)
        if self.boundaries is None:
            return np.where(v <= 0, np.nan, self.scale.fraction(v)) if self.scale.kind == "log" else self.scale.fraction(v)
        edges = self.boundaries
        bins = len(edges) - 1
        bin_ = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, bins - 1)
        f = np.where(v < edges[0], -np.inf, np.where(v > edges[-1], np.inf, (bin_ + 0.5) / bins))
        return np.where(np.isnan(v), np.nan, f)


def _oklch_stops(anchors: Sequence[tuple[float, float, float]], count: int) -> list[str]:
    """Colours evenly along a path through oklch anchors (lightness, chroma, hue in degrees,
    unwrapped so it can pass 360), each coordinate linear between anchors."""
    a = np.array(anchors, dtype=float)
    t = np.linspace(0, len(anchors) - 1, count)
    lightness, chroma, hue = (np.interp(t, np.arange(len(anchors)), a[:, i]) for i in range(3))
    lab = np.stack([lightness, chroma * np.cos(np.radians(hue)), chroma * np.sin(np.radians(hue))], axis=-1)
    return [to_hex(rgb) for rgb in oklab_to_srgb(lab)]


DEFAULT = Colormap(*_oklch_stops([(0.45, 0.13, 295), (0.61, 0.17, 355), (0.75, 0.14, 415), (0.86, 0.11, 530)], 16))
"""the default sequential map, built in oklch: lightness rises steadily from 0.45 to 0.86, clear of
the light background (white, 1) and the dark one (#1e1e1e, 0.24), while the hue runs lavender,
pink, orange, then through yellow-green to teal. every anchor's chroma stays inside srgb."""
