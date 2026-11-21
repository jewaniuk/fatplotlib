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

    def resolve[T](self, value: T | Token) -> T | Value:
        """The literal behind a token, or the value itself if it is not a token.

        Args:
            value (T | Token): a token or a literal.

        Returns:
            T | Value: the literal.

        Raises:
            KeyError: the theme has no such token.
        """
        if not isinstance(value, Token):
            return value
        try:
            return self.tokens[value.name]
        except KeyError:
            raise KeyError(f"theme {self.name!r} has no token {value.name!r}") from None

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
        {"fg": "#000000", "bg": "#ffffff", "muted": "#6b6b6b", "grid": "#d9d9d9", "thin": 0.4, "normal": 0.8, "bold": 1.2}
    ),
)
"""black on white."""

DARK = LIGHT.with_("dark", fg="#ffffff", bg="#1e1e1e", muted="#a6a6a6", grid="#3d3d3d")
"""white on near-black, as on the dark slides. weights match light so geometry does not move."""
