#!/usr/bin/env python3
"""Expand ``{{drawing:<id>}}`` and ``{{sim:<id>}}`` tokens into the site's pictures.

A drawing is an Excalidraw scene under ``docs/drawings/`` exported by
``docs/drawings/_tools/render.py`` to ``docs/assets/drawings/<id>.paper.svg`` and
``<id>.dark.svg``. A sim artifact is a frame or clip a page's own code produced,
committed under ``docs/assets/sim/<id>.png`` (and optionally ``<id>.webm``) by
``docs/hooks/sim_frames.py``. Either way the page writes one token, so a picture
costs one word against the site ceiling, and this hook writes the ``<figure>``:

    {{drawing:d01_what_is}}
    {{sim:first-robot-1|what the code above built}}

The drawing's alt text comes from the scene file (``strandsDrawing.alt``); the sim
frame's caption is the text after ``|`` (default "what the code above built"). Both
schemes ship for a drawing and Material's ``#only-light`` / ``#only-dark`` fragment
convention swaps them with the palette toggle. Every image loads lazily. An unknown
id is a build warning, which ``mkdocs build --strict`` turns into an error, so a
token cannot ship as literal text.
"""

from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.visuals")

_DOCS = Path(__file__).resolve().parents[1]
_DRAWINGS = _DOCS / "drawings"
_DRAWING_SVGS = _DOCS / "assets" / "drawings"
_SIM = _DOCS / "assets" / "sim"
_TOKEN = re.compile(r"\{\{(drawing|sim):([A-Za-z0-9_-]+)(?:\|([^}]*))?\}\}")


def drawing_alt(drawing_id: str) -> str | None:
    """The alt text the scene declares, or None when the scene does not exist."""
    scene = _DRAWINGS / f"{drawing_id}.excalidraw"
    if not scene.is_file():
        return None
    data = json.loads(scene.read_text(encoding="utf-8"))
    return str((data.get("strandsDrawing") or {}).get("alt") or drawing_id)


def drawing_html(drawing_id: str, prefix: str) -> str | None:
    """The figure for one drawing, or None when a scheme's SVG or the scene is missing."""
    alt = drawing_alt(drawing_id)
    paper = _DRAWING_SVGS / f"{drawing_id}.paper.svg"
    dark = _DRAWING_SVGS / f"{drawing_id}.dark.svg"
    if alt is None or not paper.is_file() or not dark.is_file():
        return None
    safe = html.escape(alt, quote=True)
    return (
        f'<figure class="sr-drawing">'
        f'<img src="{prefix}assets/drawings/{drawing_id}.paper.svg#only-light" alt="{safe}" loading="lazy">'
        f'<img src="{prefix}assets/drawings/{drawing_id}.dark.svg#only-dark" alt="{safe}" loading="lazy">'
        f"</figure>"
    )


def sim_html(sim_id: str, caption: str | None, prefix: str) -> str | None:
    """The figure for one sim artifact (video when a webm sits beside the png), or None."""
    png = _SIM / f"{sim_id}.png"
    if not png.is_file():
        return None
    text = html.escape((caption or "what the code above built").strip())
    webm = _SIM / f"{sim_id}.webm"
    if webm.is_file():
        media = (
            f'<video src="{prefix}assets/sim/{sim_id}.webm" poster="{prefix}assets/sim/{sim_id}.png" '
            f'muted loop playsinline controls preload="none"></video>'
        )
    else:
        media = f'<img src="{prefix}assets/sim/{sim_id}.png" alt="{text}" loading="lazy">'
    return f'<figure class="sr-sim">{media}<figcaption>{text}</figcaption></figure>'


def substitute(markdown: str, page_path: str = "<string>") -> str:
    """Replace every visual token in ``markdown``; warn on one that has no files."""
    # Directory URLs: ``project/architecture.md`` is served at ``/project/architecture/``,
    # so a relative asset path climbs one level per path segment; an ``index.md`` is
    # served at its directory and climbs one less.
    depth = page_path.count("/") + (0 if page_path.endswith("index.md") else 1)
    prefix = "../" * depth

    def _one(match: re.Match[str]) -> str:
        kind, ident, caption = match.group(1), match.group(2), match.group(3)
        out = drawing_html(ident, prefix) if kind == "drawing" else sim_html(ident, caption, prefix)
        if out is None:
            where = "docs/assets/drawings + docs/drawings" if kind == "drawing" else "docs/assets/sim"
            log.warning("%s: {{%s:%s}} has no files under %s", page_path, kind, ident, where)
            return match.group(0)
        return out

    return _TOKEN.sub(_one, markdown)


def referenced_ids(markdown: str) -> set[tuple[str, str]]:
    """Every ``(kind, id)`` a page references; the grader pairs it with the files on disk."""
    return {(m.group(1), m.group(2)) for m in _TOKEN.finditer(markdown)}


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    """mkdocs hook entry point: expand every visual token."""
    return substitute(markdown, page.file.src_path)
