"""Repo hygiene: every registered policy provider has a docs page.

A provider registered in ``strands_robots/registry/policies.json`` is part of
the public API: ``create_policy("<provider>")`` works for it. If the MkDocs
site has no page for that provider, a user who discovers it via
``list_providers()`` lands on a dead end. This guard ties the registry to the
documentation so a new provider cannot ship without a page. Reachability is
graded in ``tests/test_docs_two_lane_architecture.py``: a page under
``docs/reference/`` is published by the generated index.

``mock`` is exempt: it is a built-in testing stub documented inline in the
policy overview, not a standalone provider page.

The page does not have to sit under ``docs/reference/policies/``. ``remote`` is
documented with the client/server split it is half of
(``docs/reference/inference/remote.md``), and the overview's provider matrix links there,
so what this rule needs is that *some* page is named for the
provider - not that a second page is kept beside the first to satisfy a path.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICIES_JSON = REPO_ROOT / "strands_robots" / "registry" / "policies.json"
DOCS_DIR = REPO_ROOT / "docs"

# Providers documented inline rather than on a standalone page.
_INLINE_DOCUMENTED = {"mock"}


def _registered_providers() -> set[str]:
    data = json.loads(POLICIES_JSON.read_text(encoding="utf-8"))
    return set(data["providers"].keys())


def _page_stems() -> set[str]:
    """Return the stem of every page on the site, provider-id spelled."""
    # Normalise hyphens to underscores so a provider id (``lerobot_local``)
    # matches a hyphenated doc stem (``lerobot-local.md``).
    return {page.stem.replace("-", "_") for page in DOCS_DIR.rglob("*.md")}


def test_every_provider_has_a_docs_page() -> None:
    """Each non-mock registered provider is named by a page on the site."""
    providers = _registered_providers() - _INLINE_DOCUMENTED
    pages = _page_stems()
    missing = sorted(p for p in providers if p not in pages)
    assert not missing, (
        f"registered policy providers with no docs page: {missing}. Add a page "
        f"named for the provider - beside the subject it belongs to is fine."
    )
