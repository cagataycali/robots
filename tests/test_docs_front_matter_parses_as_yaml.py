"""Every docs page's front matter is YAML the site can read.

An unquoted ``description:`` that itself contains a colon is not YAML: Material
stops parsing, renders the whole block as the first paragraph (an H2 reading
"description: ...") and falls back to "Index" as the page title. Seven pages
shipped that way. This test reads every front matter block line by line (the
way the other workflow and docs pins do: ``types-PyYAML`` is not a dev
dependency, so ``import yaml`` would fail ``mypy`` under
``ignore_missing_imports = false``) and refuses a scalar value that YAML would
read as a nested mapping or a comment: an unquoted value containing ``": "`` or
``" #"``, or ending in ``":"``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_SCALAR = re.compile(r"^(?P<key>[A-Za-z_][\w-]*):\s*(?P<value>.*?)\s*$")


def _pages() -> list[Path]:
    return sorted(p for p in _DOCS.rglob("*.md") if "hooks" not in p.parts)


def _front_matter(page: Path) -> list[str] | None:
    text = page.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    return None if end == -1 else text[4:end].splitlines()


def _breaks_yaml(value: str) -> bool:
    if not value or value[0] in "\"'[{":
        return False
    return ": " in value or " #" in value or value.endswith(":")


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(_DOCS)))
def test_front_matter_scalars_are_quoted_when_they_carry_a_colon(page: Path) -> None:
    block = _front_matter(page)
    if block is None:
        return
    offenders = []
    for line in block:
        match = _SCALAR.match(line)
        if match and _breaks_yaml(match["value"]):
            offenders.append(f"{match['key']}: {match['value'][:60]}")
    assert not offenders, (
        f"{page.relative_to(_DOCS)}: front matter value(s) YAML reads as a nested mapping or a comment, "
        f"so Material renders the block as an H2 and titles the page Index: {offenders}. Quote the value."
    )
