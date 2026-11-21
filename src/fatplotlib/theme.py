"""themes: token name -> colour, alpha or line weight.

a figure names tokens (`Token("fg")`) wherever a colour, opacity or stroke width goes. the
preview leaves them as css custom properties (`var(--fg)`) and defines each theme as a css
class; export swaps in each theme's literal values. token names are python identifiers so
`DARK.with_(accent_teal="#5fd3c6")` can set them.
"""

from dataclasses import dataclass, field
from types import MappingProxyType

from fatplotlib.style import Token

type Value = str | float


@dataclass(frozen=True, slots=True)
class Theme:
    """A named set of token values.

    Attributes:
        name (str): the theme name, also its css class in the preview.
        tokens (MappingProxyType[str, Value]): token name -> "#rrggbb" or "#rrggbbaa" colour,
            or a number (alpha 0 to 1, weight in pt).
    """

    name: str
    tokens: MappingProxyType[str, Value] = field(default_factory=lambda: MappingProxyType({}))

    def with_(self, name: str | None = None, **tokens: Value) -> Theme:
        """A copy with some tokens changed or added.

        Args:
            name (str | None): the new theme's name, this one's if None.
            **tokens (Value): token values to set.

        Returns:
            Theme: the derived theme.
        """
        return Theme(name or self.name, MappingProxyType({**self.tokens, **tokens}))

    def colour(self, value: str | Token) -> str:
        """The colour behind a token, or the colour itself.

        Args:
            value (str | Token): a colour token or a literal colour.

        Returns:
            str: the colour.

        Raises:
            KeyError: the theme has no such token.
            TypeError: the token holds a number.
        """
        return self._resolve(value, str)

    def number(self, value: float | Token) -> float:
        """The alpha or weight behind a token, or the number itself.

        Args:
            value (float | Token): an alpha or weight token, or a literal number.

        Returns:
            float: the number.

        Raises:
            KeyError: the theme has no such token.
            TypeError: the token holds a colour.
        """
        return self._resolve(value, (int, float))

    def _resolve[T](self, value: T | Token, kind: type[T] | tuple[type, ...]) -> T:
        """A token's value checked against the kind the caller needs, or a literal unchanged."""
        if not isinstance(value, Token):
            return value
        try:
            resolved = self.tokens[value.name]
        except KeyError:
            raise KeyError(f"theme {self.name!r} has no token {value.name!r}") from None
        if not isinstance(resolved, kind):
            raise TypeError(f"token {value.name!r} holds {resolved!r}, which is not a {'colour' if kind is str else 'number'}")
        return resolved  # pyright: ignore[reportReturnType]

    def css(self, selector: str) -> str:
        """A css rule defining every token as a custom property.

        Args:
            selector (str): the css selector, e.g. ".dark".

        Returns:
            str: the rule.
        """
        return f"{selector}{{{';'.join(f'--{k}:{v}' for k, v in self.tokens.items())}}}"


LIGHT = Theme(
    "light",
    MappingProxyType(
        {
            "fg": "#000000",
            "bg": "#ffffff",
            "muted": "#6b6b6b",
            "grid": "#d9d9d9",
            "accent_orange": "#e89145",
            "accent_lavender": "#c5aff0",
            "accent_teal": "#6ce4bd",
            "thin": 0.4,
            "fine": 0.6,
            "normal": 0.8,
            "bold": 1.2,
        }
    ),
)
"""black on white."""

DARK = LIGHT.with_("dark", fg="#ffffff", bg="#1e1e1e", muted="#a6a6a6", grid="#3d3d3d")
"""white on near-black, as on the dark slides. accents and weights match light (geometry must not move)."""
