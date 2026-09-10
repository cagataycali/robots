"""mkdocs hook: robot catalog cards generated from the registry.

``{{robot_cards:<category>[,<category>...]}}`` in a page becomes one card per
registered robot in those categories, in registry order: the sim render from
``docs/assets/sim_render_<name>.png`` when one exists (captioned as a sim
render, never anything else), the ``Robot("<name>")`` one-liner, joint count,
sim / real badges, and the aliases ``Robot()`` accepts. Hand-written catalog
tables are gone, so a robot added to ``robots.json`` appears on its family
page at the next build and a stale row can no longer exist.

Filesystem only (``robots.json`` + ``docs/assets``); no package import.
An unknown category warns, which ``mkdocs build --strict`` fails on.
"""

from __future__ import annotations

import html
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.robot_cards")

_REPO = Path(__file__).resolve().parents[2]
_REGISTRY = _REPO / "strands_robots" / "registry" / "robots.json"
_ASSETS = _REPO / "docs" / "assets"
_TOKEN = re.compile(r"^\{\{\s*robot_cards:([a-z_,\s]+)\}\}\s*$", re.M)


@lru_cache(maxsize=1)
def registry() -> dict[str, dict]:
    """The robots.json registry, read once per build."""
    return json.loads(_REGISTRY.read_text(encoding="utf-8"))["robots"]


def render_path(name: str) -> Path | None:
    """The docs-relative render for a robot, or None when it has none."""
    path = _ASSETS / f"sim_render_{name}.png"
    return path if path.is_file() else None


def card(name: str, spec: dict, assets_prefix: str) -> str:
    """One robot as an HTML card (md_in_html lets it sit in a page)."""
    desc = html.escape(spec.get("description", ""))
    joints = spec.get("joints")
    badges = ['<span class="rc-badge rc-sim">sim</span>'] if spec.get("asset") else []
    if spec.get("hardware"):
        badges.append('<span class="rc-badge rc-real">real</span>')
    figure = ""
    if render_path(name):
        figure = (
            f'<figure><img src="{assets_prefix}sim_render_{name}.png" alt="{name} sim render" loading="lazy">'
            f"<figcaption>sim render</figcaption></figure>"
        )
    aliases = spec.get("aliases") or []
    alias_html = (
        "<details><summary>"
        + f"{len(aliases)} alias{'es' if len(aliases) != 1 else ''}"
        + "</summary>"
        + " ".join(f"<code>{html.escape(a)}</code>" for a in aliases)
        + "</details>"
        if aliases
        else ""
    )
    joints_html = f"<dd>{joints} joints</dd>" if isinstance(joints, int) else "<dd>joints: n/a (no sim asset)</dd>"
    return (
        f'<article class="robot-card" id="robot-{html.escape(name)}">'
        f"{figure}"
        f"<h3><code>{html.escape(name)}</code> {' '.join(badges)}</h3>"
        f"<p>{desc}</p>"
        f'<pre><code>Robot("{html.escape(name)}")</code></pre>'
        f"<dl>{joints_html}</dl>"
        f"{alias_html}"
        "</article>"
    )


def cards(categories: list[str], assets_prefix: str = "../assets/") -> str:
    """HTML for one card per registry robot in the given categories."""
    reg = registry()
    known = {spec["category"] for spec in reg.values()}
    for category in categories:
        if category not in known:
            log.warning("unknown robot_cards category %r (known: %s)", category, ", ".join(sorted(known)))
    items = [card(n, s, assets_prefix) for n, s in reg.items() if s["category"] in categories]
    return '<div class="robot-grid" markdown="0">\n' + "\n".join(items) + "\n</div>\n"


def substitute(markdown: str, assets_prefix: str = "../assets/") -> str:
    """Replace every {{cards:...}} token in a page with rendered cards."""

    def _one(match: re.Match[str]) -> str:
        categories = [c.strip() for c in match.group(1).split(",") if c.strip()]
        return cards(categories, assets_prefix)

    return _TOKEN.sub(_one, markdown)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    """mkdocs hook: substitute card tokens before the page is rendered."""
    # Raw HTML is not rewritten by mkdocs, so the prefix is computed from the
    # built URL: robots/arms.md -> robots/arms/index.html -> ../../assets/.
    depth = page.url.rstrip("/").count("/") + (1 if page.url.rstrip("/") else 0)
    if not config.get("use_directory_urls", True):
        depth = page.file.src_path.count("/")
    return substitute(markdown, "../" * depth + "assets/")
