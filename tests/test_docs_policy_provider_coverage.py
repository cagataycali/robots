"""Repo hygiene: every registered policy provider has a docs page.

A provider registered in ``strands_robots/registry/policies.json`` is part of
the public API: ``create_policy("<provider>")`` works for it. If the MkDocs
site has no page for that provider, a user who discovers it via
``list_providers()`` lands on a dead end. This guard ties the registry to the
documentation so a new provider cannot ship without a page. Reachability is
the strict build's job: every page is in the nav or ``not_in_nav``.

``mock`` is exempt: it is a built-in testing stub documented inline in the
provider matrix (``docs/learn/policies/index.md``), not a standalone page.

The matrix is generated from the registry by ``docs/hooks/providers.py``
(``{{providers:table}}``) and links each provider to its page when one exists,
through the hook's own page map (``wbc_gait`` shares ``wbc.md``). So the claim
is graded on the rendered matrix: every non-mock provider's row links a page
that is on disk. A page shared with the provider it extends is fine; a bare,
unlinked row is the dead end this rule exists to catch.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests._docs_hooks import docs_hook

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICIES_JSON = REPO_ROOT / "strands_robots" / "registry" / "policies.json"
DOCS_DIR = REPO_ROOT / "docs"
POLICIES_DIR = DOCS_DIR / "learn" / "policies"
_MATRIX = POLICIES_DIR / "index.md"

# Providers documented inline rather than on a standalone page.
_INLINE_DOCUMENTED = {"mock"}

#: A matrix row: linked ``[`name`](page.md)`` or bare ```name``` in the first cell.
_ROW = re.compile(r"^\|\s*(?:\[`([a-z0-9_]+)`\]\(([\w./-]+\.md)\)|`([a-z0-9_]+)`)\s*\|")


def _registered_providers() -> set[str]:
    data = json.loads(POLICIES_JSON.read_text(encoding="utf-8"))
    return set(data["providers"].keys())


def _rendered_matrix() -> str:
    """The provider matrix page with ``{{providers:...}}`` expanded by the shipped hook."""
    module = docs_hook("providers")
    source = _MATRIX.read_text(encoding="utf-8")
    rendered = module.substitute(source, "learn/policies/index.md")
    assert rendered != source, "learn/policies/index.md carries no {{providers:table}} token"
    return rendered


def _matrix_rows() -> dict[str, str | None]:
    """Provider id -> the page its matrix row links, or ``None`` for a bare row."""
    rows: dict[str, str | None] = {}
    for line in _rendered_matrix().splitlines():
        match = _ROW.match(line)
        if match is None:
            continue
        linked, page, bare = match.groups()
        rows[linked or bare] = page
    return rows


def test_every_provider_has_a_docs_page() -> None:
    """Each non-mock registered provider's matrix row links a page that exists."""
    rows = _matrix_rows()
    providers = _registered_providers()
    assert providers <= set(rows), f"the rendered matrix has no row for {sorted(providers - set(rows))}"
    missing = sorted(
        name
        for name in providers - _INLINE_DOCUMENTED
        if (page := rows[name]) is None or not (POLICIES_DIR / page).is_file()
    )
    assert not missing, (
        f"registered policy providers whose matrix row links no docs page: {missing}. Add a page "
        f"named for the provider under docs/learn/policies/ (or map it to the page of the provider "
        f"it extends in docs/hooks/providers.py, as wbc_gait is)."
    )


def test_the_inline_documented_providers_are_named_by_the_matrix() -> None:
    """``mock`` has no page by design, so its row is the documentation; it must exist."""
    rows = _matrix_rows()
    for name in sorted(_INLINE_DOCUMENTED):
        assert name in _registered_providers(), f"{name} is no longer registered; drop it from _INLINE_DOCUMENTED"
        assert name in rows, f"the matrix has no row for the inline-documented `{name}`"
