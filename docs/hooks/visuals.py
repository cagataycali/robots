#!/usr/bin/env python3
"""Expand ``{{drawing:<id>}}`` and ``{{sim:<id>}}`` tokens into the site's pictures, and label sketches.

A drawing is a scene module under ``docs/drawings/scenes/<id>.py`` rendered by
``docs/drawings/_tools/scene.py`` to ``docs/assets/drawings/<id>.paper.svg`` and
``<id>.dark.svg``. A sim artifact is a frame or clip a page's own code produced,
committed under ``docs/assets/sim/<id>.png`` (and optionally ``<id>.webm``) by
``docs/hooks/sim_frames.py``. Either way the page writes one token, so a picture
costs one word against the site ceiling, and this hook writes the ``<figure>``:

    {{drawing:d01_what_is}}
    {{sim:first-robot-1|what the code above built}}

The drawing's alt text is the ``<title>`` the scene wrote into its SVG; the sim
frame's caption is the text after ``|`` (default "what the code above built"). Both
schemes ship for a drawing and Material's ``#only-light`` / ``#only-dark`` fragment
convention swaps them with the palette toggle. Every image loads lazily. An unknown
id is a build warning, which ``mkdocs build --strict`` turns into an error, so a
token cannot ship as literal text.

The same pass relabels the fences the docs never run. A fence written
``python title="sketch"`` is the machine marker ``docs/hooks/check_sketches.py`` and the
graders key on; a reader is told what it means instead: the label becomes
"python, not run on this page: needs a robot on USB". An author who knows the reason
writes ``title="sketch: needs a GPU"`` and that reason replaces the default. The label
renders as the site's chip on the fence's edge (``extra.css``), not as a filename tab.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.visuals")

_DOCS = Path(__file__).resolve().parents[1]
_SCENES = _DOCS / "drawings" / "scenes"
_DRAWING_SVGS = _DOCS / "assets" / "drawings"
_SVG_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL)
_SIM = _DOCS / "assets" / "sim"
_TOKEN = re.compile(r"\{\{(drawing|sim):([A-Za-z0-9_-]+)(?:\|([^}]*))?\}\}")
_SKETCH = re.compile(r'^(```+\s*python\b[^\n]*?\btitle=)"sketch(?::\s*([^"]+))?"', re.MULTILINE)
SKETCH_DEFAULT_REASON = "needs a robot on USB"


def sketch_label(reason: str | None) -> str:
    """The label a reader sees on a fence the page did not run."""
    return f"python, not run on this page: {(reason or SKETCH_DEFAULT_REASON).strip()}"


def relabel_sketches(markdown: str) -> str:
    """``title="sketch"`` and ``title="sketch: <reason>"`` become the reader's label."""
    return _SKETCH.sub(lambda m: f'{m.group(1)}"{sketch_label(m.group(2))}"', markdown)


def drawing_alt(drawing_id: str) -> str | None:
    """The alt text the scene wrote as the SVG's title, or None when the scene or the SVG is missing."""
    scene = _SCENES / f"{drawing_id}.py"
    paper = _DRAWING_SVGS / f"{drawing_id}.paper.svg"
    if not scene.is_file() or not paper.is_file():
        return None
    match = _SVG_TITLE.search(paper.read_text(encoding="utf-8"))
    return html.unescape(match.group(1)).strip() if match else drawing_id


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
    # Directory URLs: ``concepts/architecture.md`` is served at ``/concepts/architecture/``,
    # so a relative asset path climbs one level per path segment; an ``index.md`` is
    # served at its directory and climbs one less.
    depth = page_path.count("/") + (0 if page_path.endswith("index.md") else 1)
    prefix = "../" * depth

    def _one(match: re.Match[str]) -> str:
        kind, ident, caption = match.group(1), match.group(2), match.group(3)
        out = drawing_html(ident, prefix) if kind == "drawing" else sim_html(ident, caption, prefix)
        if out is None:
            where = "docs/assets/drawings + docs/drawings/scenes" if kind == "drawing" else "docs/assets/sim"
            log.warning("%s: {{%s:%s}} has no files under %s", page_path, kind, ident, where)
            return match.group(0)
        return out

    return _TOKEN.sub(_one, markdown)


def referenced_ids(markdown: str) -> set[tuple[str, str]]:
    """Every ``(kind, id)`` a page references; the grader pairs it with the files on disk."""
    return {(m.group(1), m.group(2)) for m in _TOKEN.finditer(markdown)}


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    """mkdocs hook entry point: expand every visual token, then label the sketches."""
    return relabel_sketches(substitute(markdown, page.file.src_path))
