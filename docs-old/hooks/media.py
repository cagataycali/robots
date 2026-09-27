"""mkdocs hook: content images carry ``loading="lazy"`` in the HTML.

A browser starts fetching an ``<img>`` the moment the parser reaches its tag,
so an attribute a script assigns from a page-ready handler arrives after the
request is already in flight. Writing it at build time is what actually defers
the fetch: measured on a 390x844 viewport over an emulated 4G connection,
``policies/wbc-rollouts`` pulled 664 KB of below-the-fold GIF before any
scroll, and 0 KB once the attribute was in the HTML.

``on_page_content`` sees the rendered markdown alone, so the theme's own
chrome is untouched, and a tag that already declares ``loading`` - the robot
cards emit their own - is returned exactly as it was.
"""

from __future__ import annotations

import re

#: One image tag, however the converter spelled it.
_IMG = re.compile(r"<img\b[^>]*?/?>", re.I)

#: A tag that already states its loading strategy is left alone.
_DECLARED = re.compile(r"\bloading\s*=", re.I)

_LAZY = ' loading="lazy"'


def lazify(html: str) -> str:
    """Add ``loading="lazy"`` to every image tag that does not declare it."""

    def rewrite(match: re.Match[str]) -> str:
        tag = match.group(0)
        if _DECLARED.search(tag):
            return tag
        close = "/>" if tag.endswith("/>") else ">"
        return f"{tag[: -len(close)].rstrip()}{_LAZY}{close}"

    return _IMG.sub(rewrite, html)


def on_page_content(html: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    """Rewrite the rendered content of one page."""
    return lazify(html)
