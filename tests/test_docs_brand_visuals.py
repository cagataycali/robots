"""Repo hygiene: the brand assets the site and README point at are real files.

The from-scratch docs carry one brand asset, the Strands mark
(``docs/assets/img/mark.svg``), wired in as the site logo and favicon in
``mkdocs.yml``. The three animated SVGs of the old site (``hero_loop.svg``,
``architecture_flow.svg``, ``mesh_network.svg``) left the tree with it; the
README still embeds them by repo-relative path, and GitHub renders a broken
image for every path that resolves to nothing.

This guard fails fast if the mark is deleted or stops being valid XML, if
``mkdocs.yml`` stops pointing the logo and favicon at it, or if any
repo-relative image the README embeds is missing from the tree.
"""

from __future__ import annotations

import re
import xml.dom.minidom
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
MARK = DOCS / "assets" / "img" / "mark.svg"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
README = REPO_ROOT / "README.md"

_IMG_SRC = re.compile(r"""<img[^>]*\ssrc=["']([^"']+)["']""")
_MD_IMG = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")


def test_brand_mark_exists_and_is_valid_svg() -> None:
    """The Strands mark ships under docs/assets/img and parses as XML."""
    assert MARK.is_file(), f"missing brand asset: {MARK.relative_to(REPO_ROOT)}"
    text = MARK.read_text(encoding="utf-8")
    # Raises on malformed XML: the browser would otherwise show a broken favicon.
    xml.dom.minidom.parseString(text)
    assert "<svg" in text


def test_mkdocs_points_logo_and_favicon_at_the_mark() -> None:
    """Both theme slots use the same mark, relative to docs/."""
    config = MKDOCS_YML.read_text(encoding="utf-8")
    for slot in ("logo", "favicon"):
        match = re.search(rf"^\s+{slot}:\s*(\S+)\s*$", config, re.M)
        assert match, f"mkdocs.yml has no theme {slot}"
        assert (DOCS / match.group(1)).is_file(), f"theme {slot} {match.group(1)} is not a file under docs/"


def _readme_repo_images() -> list[str]:
    text = README.read_text(encoding="utf-8")
    srcs = _IMG_SRC.findall(text) + _MD_IMG.findall(text)
    return [s for s in srcs if not s.startswith(("http://", "https://", "data:"))]


def test_readme_embeds_at_least_one_repo_image() -> None:
    """The README carries repo-hosted figures; a README with none would make the next test vacuous."""
    assert _readme_repo_images(), "README embeds no repo-relative image"


def test_every_repo_image_the_readme_embeds_exists() -> None:
    """Every repo-relative <img src> or ![](...) in the README resolves to a file GitHub can render."""
    missing = [src for src in _readme_repo_images() if not (REPO_ROOT / src).is_file()]
    assert not missing, f"README embeds images that are not in the tree (GitHub renders a broken image): {missing}"
