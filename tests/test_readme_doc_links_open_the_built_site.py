"""The README links a docs page through the built site, never the Markdown source.

Most pages under ``docs/`` are filled in at build time by the MkDocs hooks
listed in ``mkdocs.yml`` (``{{env_vars}}``, ``{{robot_cards}}``,
``{{extras:table}}`` ...). github.com and PyPI render the README without those
hooks, so a relative ``docs/<page>.md`` link opened there shows the raw token
where the table or figure should be (on PyPI the relative link is a 404).
A link to ``site_url`` lands on the expanded page on every surface.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_README = (_ROOT / "README.md").read_text(encoding="utf-8")
_SITE = re.search(r"^site_url:\s*(\S+)", (_ROOT / "mkdocs.yml").read_text(encoding="utf-8"), re.M).group(1)  # type: ignore[union-attr]


def _source_page(url: str) -> Path:
    """Return the ``docs/`` source a site URL is built from (directory URLs)."""
    slug = url.removeprefix(_SITE).split("#", 1)[0].strip("/")
    page = _ROOT / "docs" / f"{slug}.md"
    return page if slug and page.is_file() else _ROOT / "docs" / slug / "index.md"


def test_readme_doc_links_use_the_site_and_name_a_real_page() -> None:
    relative = re.findall(r"\]\((docs/[^)]+\.md[^)]*)\)", _README)
    assert not relative, f"README links Markdown sources that render raw macros off-site: {relative}"

    site_links = re.findall(r"\]\((" + re.escape(_SITE) + r"[^)]*)\)", _README)
    assert len(site_links) >= 10, f"the sweep lost the README's docs links: {site_links}"
    missing = [url for url in site_links if not _source_page(url).is_file()]
    assert not missing, f"README links site pages with no docs/ source: {missing}"
