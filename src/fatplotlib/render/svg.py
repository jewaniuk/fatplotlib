"""scene -> svg string.

the scene is y-up in pt; one flip at the root maps it to svg's y-down. text groups flip back
locally. `<defs>` (gradients, clip paths, typst glyphs) are collected once per document and keyed by a
content hash or typst's own glyph hash, so ids are stable across runs and duplicates collapse.

without a theme every token becomes a css custom property (`fill:var(--fg)`), which the preview
defines per theme. with a theme the scene is resolved to literal values first, for export.
"""

import hashlib
from dataclasses import replace
from html import escape

from fatplotlib.scene import BBox, Circle, Ellipse, Group, Node, Path, Rect, Shape, Text
from fatplotlib.style import Paint, RadialGradient, Stroke, Token
from fatplotlib.theme import Theme


def render(root: Node, width: float, height: float, theme: Theme | None = None) -> str:
    """Renders a scene to a standalone svg document.

    Args:
        root (Node): the scene, y-up, with the origin at the bottom-left of the page.
        width (float): page width in pt.
        height (float): page height in pt.
        theme (Theme | None): resolves tokens to literals; None leaves them as css variables.

    Returns:
        str: the svg document.

    Raises:
        KeyError: the scene uses a token the theme does not define.
    """
    if theme is not None:
        root = _resolved(root, theme)
    defs: dict[str, str] = {}
    body = _node(root, defs)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{_num(width)}pt" height="{_num(height)}pt" viewBox="0 0 {_num(width)} {_num(height)}">'
        f"<defs>{''.join(defs.values())}</defs>"
        f'<g transform="matrix(1 0 0 -1 0 {_num(height)})">{body}</g>'
        "</svg>"
    )


def _resolved(node: Node, theme: Theme) -> Node:
    """A copy of the scene with every token replaced by the theme's value."""

    def paint(value: Paint | None) -> Paint | None:
        if isinstance(value, RadialGradient):
            stops = tuple(replace(stop, colour=theme.colour(stop.colour)) for stop in value.stops)
            return replace(value, stops=stops)
        return value if value is None else theme.colour(value)

    changes: dict[str, object] = {"alpha": theme.number(node.alpha)}
    match node:
        case Group():
            changes["children"] = [_resolved(child, theme) for child in node.children]
        case Shape():
            changes["fill"] = paint(node.fill)
            if node.stroke is not None:
                stroke = node.stroke
                changes["stroke"] = replace(stroke, paint=paint(stroke.paint), width=theme.number(stroke.width))
        case Text():
            changes["fill"] = paint(node.fill)
    return replace(node, **changes)


def _num(value: float) -> str:
    """A number with six significant digits and no trailing zeros."""
    return f"{value:.6g}"


def _attrs(pairs: list[tuple[str, str]], style: list[str]) -> str:
    """Attribute text, with css declarations (theme tokens) merged into one style attribute."""
    if style:
        pairs = [*pairs, ("style", ";".join(style))]
    return "".join(f' {name}="{escape(value)}"' for name, value in pairs)


def _paint(prop: str, paint: Paint | None, defs: dict[str, str]) -> tuple[list[tuple[str, str]], list[str]]:
    """Attributes and css declarations for a fill or stroke paint."""
    match paint:
        case None:
            return [(prop, "none")], []
        case Token(name):
            return [], [f"{prop}:var(--{name})"]
        case RadialGradient():
            return [(prop, f"url(#{_gradient(paint, defs)})")], []
        case str() if len(paint) == 9 and paint.startswith("#"):
            return [(prop, paint[:7]), (f"{prop}-opacity", _num(int(paint[7:], 16) / 255))], []
        case _:
            return [(prop, paint)], []


def _gradient(gradient: RadialGradient, defs: dict[str, str]) -> str:
    """Adds a gradient to the defs once and returns its id."""
    stops = []
    for stop in gradient.stops:
        pairs, style = _paint("stop-color", stop.colour, defs)
        if stop.alpha != 1:
            pairs.append(("stop-opacity", _num(stop.alpha)))
        stops.append(f"<stop{_attrs([('offset', _num(stop.offset)), *pairs], style)}/>")
    inner = "".join(stops)
    geometry = f'cx="{_num(gradient.cx)}" cy="{_num(gradient.cy)}" r="{_num(gradient.r)}"'
    gid = "grad-" + hashlib.sha1(f"{geometry}{inner}".encode()).hexdigest()[:10]
    defs.setdefault(gid, f'<radialGradient id="{gid}" {geometry}>{inner}</radialGradient>')
    return gid


def _clip(box: BBox, defs: dict[str, str]) -> str:
    """Adds a rectangular clip path to the defs once and returns its id."""
    rect = f'x="{_num(box.x0)}" y="{_num(box.y0)}" width="{_num(box.width)}" height="{_num(box.height)}"'
    cid = "clip-" + hashlib.sha1(rect.encode()).hexdigest()[:10]
    defs.setdefault(cid, f'<clipPath id="{cid}"><rect {rect}/></clipPath>')
    return cid


def _stroke(stroke: Stroke | None, defs: dict[str, str]) -> tuple[list[tuple[str, str]], list[str]]:
    """Attributes and css declarations for an outline."""
    if stroke is None:
        return [], []
    pairs, style = _paint("stroke", stroke.paint, defs)
    if isinstance(stroke.width, Token):
        style.append(f"stroke-width:var(--{stroke.width.name})")
    else:
        pairs.append(("stroke-width", _num(stroke.width)))
    if stroke.dash is not None:
        pairs.append(("stroke-dasharray", " ".join(_num(d) for d in stroke.dash)))
    if stroke.cap != "butt":
        pairs.append(("stroke-linecap", stroke.cap))
    if stroke.join != "miter":
        pairs.append(("stroke-linejoin", stroke.join))
    return pairs, style


def _common(node: Node) -> tuple[list[tuple[str, str]], list[str]]:
    """Attributes and css declarations every node may carry: id, class and opacity."""
    pairs, style = [], []
    if node.id is not None:
        pairs.append(("id", node.id))
    if node.class_ is not None:
        pairs.append(("class", node.class_))
    if isinstance(node.alpha, Token):
        style.append(f"opacity:var(--{node.alpha.name})")
    elif node.alpha != 1:
        pairs.append(("opacity", _num(node.alpha)))
    return pairs, style


def _shape_attrs(node: Shape, geometry: list[tuple[str, str]], defs: dict[str, str]) -> str:
    """Attribute text for a shape: geometry, then fill, outline and common attributes."""
    fill, fill_style = _paint("fill", node.fill, defs)
    stroke, stroke_style = _stroke(node.stroke, defs)
    common, common_style = _common(node)
    return _attrs(geometry + fill + stroke + common, fill_style + stroke_style + common_style)


def _path_data(node: Path) -> str:
    """The svg `d` string of a path."""
    return " ".join(command[0] + " ".join(_num(v) for v in command[1:]) for command in node.commands)


def _node(node: Node, defs: dict[str, str]) -> str:
    """Svg text for one node and its children, adding what it references to `defs`."""
    match node:
        case Group():
            pairs, style = _common(node)
            if node.transform is not None:
                t = node.transform
                pairs.insert(0, ("transform", f"matrix({' '.join(_num(v) for v in (t.a, t.b, t.c, t.d, t.e, t.f))})"))
            if node.clip is not None:
                pairs.append(("clip-path", f"url(#{_clip(node.clip, defs)})"))
            return f"<g{_attrs(pairs, style)}>{''.join(_node(child, defs) for child in node.children)}</g>"
        case Path():
            return f"<path{_shape_attrs(node, [('d', _path_data(node))], defs)}/>"
        case Rect():
            geometry = [(k, _num(v)) for k, v in (("x", node.x), ("y", node.y), ("width", node.width), ("height", node.height))]
            if node.radius:
                geometry.append(("rx", _num(node.radius)))
            return f"<rect{_shape_attrs(node, geometry, defs)}/>"
        case Circle():
            geometry = [("cx", _num(node.cx)), ("cy", _num(node.cy)), ("r", _num(node.r))]
            return f"<circle{_shape_attrs(node, geometry, defs)}/>"
        case Ellipse():
            geometry = [("cx", _num(node.cx)), ("cy", _num(node.cy)), ("rx", _num(node.rx)), ("ry", _num(node.ry))]
            return f"<ellipse{_shape_attrs(node, geometry, defs)}/>"
        case Text():
            shape = node.shape()
            for gid, element in shape.defs:
                defs.setdefault(gid, element)
            left, top = node.origin()
            fill, fill_style = _paint("fill", node.fill, defs)
            # strokes in the text (fraction bars) follow `color`; alpha is per channel, so drop it
            colour, colour_style = _paint("color", node.fill, defs)
            colour = [(name, value) for name, value in colour if name == "color"]
            common, common_style = _common(node)
            pairs = [("transform", f"matrix(1 0 0 -1 {_num(left)} {_num(top)})"), *fill, *colour, *common]
            return f"<g{_attrs(pairs, fill_style + colour_style + common_style)}>{shape.body}</g>"
        case _:
            raise TypeError(f"cannot render {type(node).__name__}")
