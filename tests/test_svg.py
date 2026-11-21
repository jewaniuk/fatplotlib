"""svg rendering."""

import xml.etree.ElementTree as ET

import resvg_py

from fatplotlib.paths import PathBuilder
from fatplotlib.render.svg import render
from fatplotlib.scene import BBox, Circle, Group, Rect, Text
from fatplotlib.style import RadialGradient, Stop, Stroke, with_alpha
from fatplotlib.theme import LIGHT
from fatplotlib.transform import Affine


def scene() -> Group:
    """One of each node, two photons sharing a gradient and two labels sharing glyphs."""
    glow = RadialGradient((Stop(0, "#ffffff"), Stop(1, "#1f77b4", alpha=0.7)))
    wisp = PathBuilder().move_to(10, 10).line_to(30, 14).rel_cubic_to(5, 5, 10, -5, 15, 0)
    return Group(
        [
            Rect(5, 5, 60, 40, radius=3, fill=with_alpha("#1f77b4", 0.05), stroke=Stroke("#1f77b4", 0.5)),
            wisp.path(stroke=Stroke("#1f77b4", 0.3, dash=(1, 1)), id="wisp"),
            Group([Circle(20, 30, 3, fill=glow), Circle(40, 30, 3, fill=glow)], transform=Affine.translate(1, 0)),
            Text("Photons $10^(-2)$", 35, 50, size=8, anchor="middle"),
            Text("Photons", 35, 2, size=8, anchor="middle", fill="#d62728"),
        ],
        class_="panel",
    )


def test_svg_parses_and_renders_in_resvg():
    svg = render(scene(), 70, 60, LIGHT)
    ET.fromstring(svg)
    png = resvg_py.svg_to_bytes(svg_string=svg, dpi=72)  # dpi 0 (the default) cannot size pt units
    assert bytes(png[:8]) == b"\x89PNG\r\n\x1a\n"


def test_defs_are_deduplicated_and_stable():
    svg = render(scene(), 70, 60)
    root = ET.fromstring(svg)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    defs = root.find("svg:defs", ns)
    assert defs is not None
    ids = [el.get("id") for el in defs]
    assert len(ids) == len(set(ids))
    assert len(root.findall("svg:defs/svg:radialGradient", ns)) == 1
    assert render(scene(), 70, 60) == svg


def test_clip_is_one_shared_def():
    clip = BBox(0, 0, 10, 10)
    box = Rect(0, 0, 20, 20, fill="#000000")
    svg = render(Group([Group([box], clip=clip), Group([box], clip=clip)]), 20, 20)
    root = ET.fromstring(svg)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    (clip_path,) = root.findall("svg:defs/svg:clipPath", ns)
    assert svg.count(f'clip-path="url(#{clip_path.get("id")})"') == 2
