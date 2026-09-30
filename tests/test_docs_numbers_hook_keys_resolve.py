"""The docs facts hook (docs/hooks/facts.py) resolves every token the pages use.

The hook substitutes ``{{n:key}}`` at build time from the registry and the
package tree, so a page can no longer state a robot count the registry does
not hold. This grader keeps the two ends honest: every token spelled in
``docs/**/*.md`` names a key the hook derives, and the derived values match
the tree read independently here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._docs_hooks import docs_hook

_REPO = Path(__file__).resolve().parents[1]
_TOKEN = re.compile(r"\{\{\s*n:([a-z_]+)\s*\}\}")


def _load_hook():
    return docs_hook("facts")


def _tokens_in_docs() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for page in sorted((_REPO / "docs").rglob("*.md")):
        for key in _TOKEN.findall(page.read_text(encoding="utf-8")):
            found.setdefault(key, []).append(str(page.relative_to(_REPO)))
    return found


def test_every_token_the_docs_spell_is_a_key_the_hook_derives() -> None:
    hook = _load_hook()
    known = hook.numbers()
    unknown = {key: pages for key, pages in _tokens_in_docs().items() if key not in known}
    assert not unknown, f"docs spell {{{{n:...}}}} keys the hook cannot resolve: {unknown}"


def test_the_docs_use_the_hook_at_all() -> None:
    assert _tokens_in_docs(), "no page spells a {{n:...}} token - the hook is dead weight"


def test_robot_counts_match_the_registry() -> None:
    hook = _load_hook()
    values = hook.numbers()
    # The union the hooks document: robots.json plus the robot_descriptions URDF tail.
    robots = docs_hook("registry_view").merged()
    curated = docs_hook("registry_view").curated()
    assert values["robots"] == len(robots)
    assert values["robots_curated"] == len(curated)
    assert values["urdf_robots"] == len(robots) - len(curated)
    assert values["categories"] == len({spec["category"] for spec in robots.values()})
    assert sum(values[c] for c in {spec["category"] for spec in robots.values()}) == len(robots)


def test_an_unknown_key_is_left_in_place_and_warned(caplog: pytest.LogCaptureFixture) -> None:
    hook = _load_hook()
    with caplog.at_level("WARNING", logger="mkdocs.hooks.numbers"):
        out = hook.substitute("we ship {{n:robots}} robots and {{n:nonsense}} unicorns", "x.md")
    assert out.startswith(f"we ship {hook.numbers()['robots']} robots")
    assert "{{n:nonsense}}" in out
    assert any("unknown numbers key" in rec.message for rec in caplog.records)
