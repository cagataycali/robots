"""Repo hygiene: robot *counts* stated in the docs agree with ``registry/robots.json``.

The registry is the source of truth for what ``Robot("<name>")`` accepts, and
several documents restate its size for humans: the README feature list, the hero
and architecture SVGs, ``docs/architecture.md``, the quickstart "see also" and
the ``docs/robots/`` index cards. Nothing tied those restated numbers to the
registry, so they drifted independently - the tree simultaneously claimed "40+",
"50+" and "68" robots for a registry holding 72.

Membership is no longer restated at all: ``docs/hooks/robot_cards.py`` generates
one card per registry entry on the family pages, so a robot cannot be missing
from the catalog and a card cannot name a robot the registry does not hold.
``tests/test_docs_robot_cards_hook.py`` grades that generator. What is left here
is the numbers, which are still typed by hand:

* The ``test_..._claim(s)_match(es)_the_registry`` guards pin the exact sites
  that quote a count, and their failure message is the string to write, so a fix
  needs no arithmetic.
* :func:`test_approximate_robot_count_claims_match_the_current_decade` allows the
  deliberately round "N+ robots" form, pinned to the current multiple of ten.
* :func:`test_no_robot_count_claim_outside_the_known_sites` is the net that
  catches a *new* claim added somewhere none of the above look. The guards above
  are precise about the sites they know; this one refuses an unknown number.

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

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOTS_JSON = REPO_ROOT / "strands_robots" / "registry" / "robots.json"
DOCS = REPO_ROOT / "docs"
README = REPO_ROOT / "README.md"

#: Claims that count something other than registry entries, so they are not
#: this guard's business. The README teleoperation section counts the robots a
#: teleoperator can drive, which is a property of the teleop matrix.
EXEMPT_CLAIMS: tuple[re.Pattern[str], ...] = (re.compile(r"drive \d+ robots"),)

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


def _count_claims() -> list[tuple[Path, int, str, int]]:
    """Return every robot-count claim as ``(path, lineno, line, claimed)``."""
    files = [README, *sorted(DOCS.rglob("*.md")), *sorted(DOCS.rglob("*.svg"))]
    claims: list[tuple[Path, int, str, int]] = []
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(exempt.search(line) for exempt in EXEMPT_CLAIMS):
                continue
            for match in COUNT_CLAIM_RE.finditer(line):
                claims.append((path, lineno, line.strip(), int(match.group(1))))
    return claims


def test_total_robot_count_claims_match_the_registry() -> None:
    """Documents quoting an exact registry size quote the current one."""
    total, categories = len(_registry()), len(_category_counts())
    expected = [
        (DOCS / "robots" / "index.md", f"description: {total} robots across {categories} categories."),
        (DOCS / "robots" / "index.md", f"registry of **{total} robots** across {categories} categories"),
        (DOCS / "architecture.md", f"{total} robots, {categories} categories"),
        (DOCS / "getting-started" / "quickstart.md", f"all {total} robots."),
    ]
    for path, text in expected:
        assert text in path.read_text(encoding="utf-8"), (
            f"{path.relative_to(REPO_ROOT)} should state {text!r} "
            f"(robots.json holds {total} robots in {categories} categories)"
        )


def test_approximate_robot_count_claims_match_the_current_decade() -> None:
    """Documents rounding the registry size round it down to the current ten.

    The README and the SVG labels state a deliberately round "N+ robots" so that
    adding one robot does not require a copy edit. Pinning that to the current
    multiple of ten keeps the claim both true and current: it only needs a bump
    when the registry crosses the next ten, which is exactly when "70+" starts
    understating a registry of 80.
    """
    total, categories = len(_registry()), len(_category_counts())
    decade = total // 10 * 10
    expected = [
        (DOCS / "assets" / "hero_loop.svg", f"{decade}+ robots"),
        (DOCS / "assets" / "architecture_flow.svg", f"{decade}+ robots"),
        (README, f"**{decade}+ robots, {categories} categories.**"),
        (README, f"{decade}+ robots across {categories} categories"),
        (README, f"robots.json ({decade}+)"),
    ]
    for path, text in expected:
        assert text in path.read_text(encoding="utf-8"), (
            f"{path.relative_to(REPO_ROOT)} should state {text!r} "
            f"(robots.json holds {total} robots, which rounds down to {decade})"
        )


def test_per_category_count_claims_match_the_registry() -> None:
    """The category cards and the arms page quote their real category sizes."""
    counts = _category_counts()
    index = DOCS / "robots" / "index.md"
    expected = [
        (index, f"**Arms** \u00b7 {counts['arm']}"),
        (index, f"**Bimanual** \u00b7 {counts['bimanual']}"),
        (index, f"**Humanoids** \u00b7 {counts['humanoid']}"),
        (index, f"**Hands** \u00b7 {counts['hand']}"),
        (index, f"**Mobile** \u00b7 {counts['mobile']}"),
        (index, f"**Mobile manip** \u00b7 {counts['mobile_manip']}"),
        (index, f"**Aerial** \u00b7 {counts['aerial']}"),
        (index, f"**Expressive** \u00b7 {counts['expressive']}"),
        (DOCS / "robots" / "arms.md", f"description: {counts['arm']} single-arm manipulators"),
        (DOCS / "robots" / "arms.md", f"**{counts['arm']} robots in this category.**"),
    ]
    for path, text in expected:
        assert text in path.read_text(encoding="utf-8"), (
            f"{path.relative_to(REPO_ROOT)} should state {text!r} (from robots.json)"
        )


def test_alias_count_claim_matches_the_registry() -> None:
    """The architecture table quotes the real number of registered aliases."""
    registry = _registry()
    aliases = sum(len(entry.get("aliases", [])) for entry in registry.values())
    text = f"{len(registry)} robots, {aliases} aliases, {len(_category_counts())} categories"
    assert text in (DOCS / "architecture.md").read_text(encoding="utf-8"), f"docs/architecture.md should state {text!r}"


def test_no_robot_count_claim_outside_the_known_sites() -> None:
    """A robot count stated anywhere in the docs is one the registry supports.

    The tests above check the sites that exist today. This one refuses a number
    that matches neither the registry total, its round-down, nor any category
    size, so a new claim written somewhere unexpected fails here rather than
    silently becoming the next stale number.
    """
    counts = _category_counts()
    total = sum(counts.values())
    allowed = {total, total // 10 * 10, *counts.values()}
    stale = [
        (str(path.relative_to(REPO_ROOT)), lineno, claimed, line)
        for path, lineno, line, claimed in _count_claims()
        if claimed not in allowed
    ]
    assert not stale, (
        f"robot counts that no registry number supports (allowed: {sorted(allowed)}): {stale}. "
        "Update the claim, or add it to EXEMPT_CLAIMS if it counts something else."
    )
