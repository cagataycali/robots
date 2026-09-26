"""Repo hygiene: every registered policy provider has a docs page in the nav.

A provider registered in ``strands_robots/registry/policies.json`` is part of
the public API: ``create_policy("<provider>")`` works for it. If the MkDocs
site has no page for that provider, a user who discovers it via
``list_providers()`` lands on a dead end. This guard ties the registry to the
documentation so a new provider cannot ship without a docs page wired into the
``mkdocs.yml`` navigation.

``mock`` is exempt: it is a built-in testing stub documented inline in the
policy overview, not a standalone provider page.

The page does not have to sit under ``docs/policies/``. ``remote`` is
documented with the client/server split it is half of
(``docs/inference/remote.md``), and the overview's provider matrix links there,
so what this rule needs is that *some* page in the nav is named for the
provider - not that a second page is kept beside the first to satisfy a path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICIES_JSON = REPO_ROOT / "strands_robots" / "registry" / "policies.json"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
DOCS_DIR = REPO_ROOT / "docs"

# Providers documented inline rather than on a standalone page.
_INLINE_DOCUMENTED = {"mock"}


def _registered_providers() -> set[str]:
    data = json.loads(POLICIES_JSON.read_text(encoding="utf-8"))
    return set(data["providers"].keys())


def _nav_page_stems() -> set[str]:
    """Return the stem of every doc page referenced in the mkdocs nav."""
    nav = MKDOCS_YML.read_text(encoding="utf-8")
    # Normalise hyphens to underscores so a provider id (``lerobot_local``)
    # matches a hyphenated doc stem (``lerobot-local.md``).
    return {stem.replace("-", "_") for stem in re.findall(r"([a-z0-9_-]+)\.md", nav)}


def test_every_provider_has_a_docs_page_in_nav() -> None:
    """Each non-mock registered provider has a docs page wired into the nav."""
    providers = _registered_providers() - _INLINE_DOCUMENTED
    nav_pages = _nav_page_stems()
    missing = sorted(p for p in providers if p not in nav_pages)
    assert not missing, (
        f"registered policy providers with no docs page in mkdocs.yml nav: "
        f"{missing}. Add a page named for the provider - beside the subject it "
        f"belongs to is fine - and wire it into the nav."
    )


def test_nav_policy_pages_exist_on_disk() -> None:
    """Every policy page referenced in the nav resolves to a real file."""
    nav = MKDOCS_YML.read_text(encoding="utf-8")
    stems = re.findall(r"policies/([a-z0-9_-]+)\.md", nav)
    missing = sorted(stem for stem in stems if not (DOCS_DIR / "policies" / f"{stem}.md").is_file())
    assert not missing, f"mkdocs.yml nav references missing policy docs: {missing}"
