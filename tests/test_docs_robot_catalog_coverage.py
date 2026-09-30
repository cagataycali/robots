"""Repo hygiene: robot *counts* stated in the docs agree with ``registry/robots.json``.

The registry is the source of truth for what ``Robot("<name>")`` accepts, and
several documents used to restate its size for humans: the README feature list,
the hero and architecture SVGs, the architecture page, the quickstart "see also"
and the ``docs/robots/`` index cards. Nothing tied those restated numbers to the
registry, so they drifted independently: the tree simultaneously claimed "40+",
"50+" and "68" robots for a registry holding 72.

Membership is no longer restated at all: ``docs/hooks/robot_pages.py`` generates
one card per registry entry on the catalog and the family pages, so a robot
cannot be missing from the catalog and a card cannot name a robot the registry
does not hold. Nor are the numbers typed any more: a page writes ``{{n:robots}}``
or ``{{n:arm}}`` and ``docs/hooks/facts.py`` fills it at build time. What is
graded here is the seam between the two:

* :func:`test_the_numbers_hook_counts_what_the_registry_holds` checks the
  hook's arithmetic (total, decade, categories, one count per category)
  against the registry, so a token cannot render a wrong number.
* :func:`test_the_catalog_states_its_size_through_tokens_and_filters_every_family`
  pins the catalog page: its size is a token, and its filter chips are exactly
  the registry's categories.
* :func:`test_approximate_robot_count_claims_match_the_current_decade` allows the
  README's deliberately round "N+ robots" form, pinned to the current multiple
  of ten.
* :func:`test_no_robot_count_is_typed_in_the_docs` is the net: a digit before
  the word "robots" anywhere under ``docs/`` is a count someone typed, and it
  fails here rather than becoming the next stale number.

Counts are derived from ``robots.json`` directly rather than from
:func:`~strands_robots.registry.list_robots`, because ``list_robots()`` also
returns robots registered at runtime through ``register_robot()`` and from the
user registry on disk, neither of which the docs describe. This mirrors the
reasoning in ``tests/test_docs_policy_coverage.py``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from tests._docs_hooks import docs_hook

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOTS_JSON = REPO_ROOT / "strands_robots" / "registry" / "robots.json"
DOCS = REPO_ROOT / "docs"
README = REPO_ROOT / "README.md"
CATALOG = DOCS / "robots" / "index.md"

#: Claims that count something other than registry entries, so they are not
#: this guard's business. The README teleoperation section counts the robots a
#: teleoperator can drive, which is a property of the teleop matrix; "ROS 2
#: robots" names a middleware version, not a count.
EXEMPT_CLAIMS: tuple[re.Pattern[str], ...] = (re.compile(r"drive \d+ robots"), re.compile(r"ROS [12] robots"))

_FILTER_CHIP = re.compile(r'<button[^>]*class="sr-filter-btn"[^>]*data-family="([a-z_]+)"')
_TOKEN = re.compile(r"\{\{\s*n:([a-z_]+)\s*\}\}")

#: A robot count stated in prose or in SVG label text. Requires the plural so
#: "ROS 2 robot" and "so100 robot" are not mistaken for counts, and a
#: non-identifier character before the digits so "so100 robots" is not either.
COUNT_CLAIM_RE = re.compile(r"(?:^|[^0-9A-Za-z_])(\d+)\+? robots\b")


def _registry() -> dict[str, dict]:
    """Return the built-in robot registry, keyed by canonical name."""
    return json.loads(ROBOTS_JSON.read_text(encoding="utf-8"))["robots"]


def _category_counts() -> Counter[str]:
    """Return the number of registered robots per category."""
    return Counter(entry.get("category", "") for entry in _registry().values())


def _facts() -> dict[str, int]:
    """The numbers hook's table, loaded by path: the docs venv is not the test venv."""
    module = docs_hook("facts")
    return module.numbers()


def _count_claims(files: list[Path]) -> list[tuple[Path, int, str, int]]:
    """Return every robot-count claim in ``files`` as ``(path, lineno, line, claimed)``."""
    claims: list[tuple[Path, int, str, int]] = []
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(exempt.search(line) for exempt in EXEMPT_CLAIMS):
                continue
            for match in COUNT_CLAIM_RE.finditer(line):
                claims.append((path, lineno, line.strip(), int(match.group(1))))
    return claims


def test_approximate_robot_count_claims_match_the_current_decade() -> None:
    """The README rounds the registry size down to the current ten.

    It states a deliberately round "N+ robots" so that adding one robot does
    not require a copy edit. Pinning that to the current multiple of ten keeps
    the claim both true and current: it only needs a bump when the registry
    crosses the next ten, which is exactly when "70+" starts understating a
    registry of 80.
    """
    total, categories = len(_registry()), len(_category_counts())
    decade = total // 10 * 10
    text = f"{decade}+ robots across {categories} categories"
    assert text in README.read_text(encoding="utf-8"), (
        f"README.md should state {text!r} (robots.json holds {total} robots, which rounds down to {decade})"
    )


def test_the_numbers_hook_counts_what_the_registry_holds() -> None:
    """Every count a ``{{n:...}}`` token can render agrees with ``robots.json``."""
    counts = _category_counts()
    total = sum(counts.values())
    facts = _facts()
    expected = {
        "robots": total,
        "robots_decade": total // 10 * 10,
        "categories": len(counts),
        **dict(counts),
    }
    wrong = {key: (facts.get(key), value) for key, value in expected.items() if facts.get(key) != value}
    assert not wrong, f"docs/hooks/facts.py renders counts the registry does not support (hook, registry): {wrong}"


def test_the_catalog_states_its_size_through_tokens_and_filters_every_family() -> None:
    """The catalog's size is a token, and its filter chips are the registry's categories."""
    page = CATALOG.read_text(encoding="utf-8")
    tokens = set(_TOKEN.findall(page))
    assert {"robots", "categories"} <= tokens, (
        f"docs/robots/index.md states its size with tokens {sorted(tokens)}; it needs {{{{n:robots}}}} and {{{{n:categories}}}}"
    )
    chips = _FILTER_CHIP.findall(page)
    assert chips and chips[0] == "all", "the catalog filter has no leading 'All' chip"
    assert sorted(chips[1:]) == sorted(_category_counts()), (
        f"docs/robots/index.md filters families {sorted(chips[1:])}; the registry has {sorted(_category_counts())}"
    )
    assert "{{robot_cards}}" in page, "the catalog no longer places the generated cards"


def test_no_robot_count_is_typed_in_the_docs() -> None:
    """A robot count under ``docs/`` is a token, never a digit.

    The registry once had "40+", "50+" and "68" robots claimed for it at the
    same time. The rewrite writes every count as ``{{n:key}}``, so a digit
    before the word "robots" anywhere in the tree is a claim someone typed,
    and it fails here rather than silently becoming the next stale number.
    """
    typed = [
        (str(path.relative_to(REPO_ROOT)), lineno, claimed, line)
        for path, lineno, line, claimed in _count_claims(sorted(DOCS.rglob("*.md")))
    ]
    assert not typed, f"robot counts typed by hand under docs/ (write a {{{{n:key}}}} token instead): {typed}"


def test_no_readme_count_claim_outside_the_registry_numbers() -> None:
    """A robot count in the README is one the registry supports."""
    counts = _category_counts()
    total = sum(counts.values())
    allowed = {total, total // 10 * 10, *counts.values()}
    stale = [(lineno, claimed, line) for _, lineno, line, claimed in _count_claims([README]) if claimed not in allowed]
    assert not stale, (
        f"README robot counts that no registry number supports (allowed: {sorted(allowed)}): {stale}. "
        "Update the claim, or add it to EXEMPT_CLAIMS if it counts something else."
    )
