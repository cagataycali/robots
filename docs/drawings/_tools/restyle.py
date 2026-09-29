"""restyle.py: map an Excalidraw SVG export onto the docs site's tokens.

Excalidraw cannot embed Space Grotesk or JetBrains Mono, so scenes are authored with
fontFamily 2 (Helvetica) for words and 3 (Cascadia) for code, and this step swaps the family
names and embeds the site's woff2 subsets (docs/assets/fonts) as ``data:`` URIs, the same trick
the site's earlier hand-drawn SVGs used. The paper hex values from ``excal.py`` are rewritten
for the dark scheme from the table below (docs/stylesheets/extra.css, ``[data-md-color-scheme=
"slate"]``), so one source yields both files.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

FONTS = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"

# paper -> dark, straight from docs/stylesheets/extra.css
DARK = {
    "#000000": "#ffffff",  # ink
    "#3d3c3c": "#b6b6b6",  # text
    "#767373": "#999696",  # muted
    "#007a3d": "#00cc60",  # accent
    "#e6f6e7": "#0f2a1a",  # accent soft (surface green -> a dark green wash)
    "#f4f4f4": "#28292a",  # chip
    "#999696": "#3d3c3c",  # pill border
    "#ffffff": "#000000",  # bg
}

FAMILY = {
    "Helvetica": "Space Grotesk",
    "Cascadia": "JetBrains Mono",
    "Virgil": "Space Grotesk",
    "Excalifont": "Space Grotesk",
}


def _font_face() -> str:
    faces = []
    for fam, file, weight in (
        ("Space Grotesk", "SpaceGrotesk-400.woff2", 400),
        ("Space Grotesk", "SpaceGrotesk-500.woff2", 500),
        ("JetBrains Mono", "JetBrainsMono-500.woff2", 500),
    ):
        data = base64.b64encode((FONTS / file).read_bytes()).decode("ascii")
        faces.append(
            f"@font-face{{font-family:'{fam}';font-weight:{weight};font-style:normal;"
            f"src:url(data:font/woff2;base64,{data}) format('woff2');}}"
        )
    return "<style>" + "".join(faces) + "</style>"


def _swap_colors(svg: str, table: dict[str, str]) -> str:
    # one pass with a placeholder so #000000 -> #ffffff does not feed the reverse rule
    marks = {k: f"@@{i}@@" for i, k in enumerate(table)}
    for k, m in marks.items():
        svg = re.sub(re.escape(k), m, svg, flags=re.IGNORECASE)
    for k, m in marks.items():
        svg = svg.replace(m, table[k])
    return svg


def restyle(svg: str, *, scheme: str, title: str) -> str:
    # drop Excalidraw's own embedded @font-face blocks (Excalifont, Cascadia...) and the
    # fixed pixel size; the viewBox keeps the aspect ratio and CSS sizes the picture
    svg = re.sub(r"<style[^>]*>.*?</style>", "", svg, flags=re.S)
    svg = re.sub(r'<svg([^>]*?)\swidth="[^"]*"\sheight="[^"]*"', r"<svg\1", svg, count=1)
    for old, new in FAMILY.items():
        svg = re.sub(rf'font-family="{old}[^"]*"', f'font-family="{new}"', svg)
        svg = svg.replace(f"font-family: {old}", f"font-family: {new}")
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S)  # exporter metadata comments
    if scheme == "dark":
        svg = _swap_colors(svg, DARK)
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;")
    svg = svg.replace(
        "<svg ",
        f'<svg role="img" aria-label="{safe_title}" class="sr-drawing sr-drawing--{scheme}" ',
        1,
    )
    head, sep, tail = svg.partition(">")
    return head + sep + f"<title>{safe_title}</title>" + _font_face() + tail
