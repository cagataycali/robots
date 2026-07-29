"""Repo hygiene: the robot catalog docs agree with ``registry/robots.json``.

The registry is the source of truth for what ``Robot("<name>")`` accepts, but
five separate documents restate its contents for humans: the README feature
list, the hero and architecture SVGs, ``docs/architecture.md``, the quickstart
"see also", and the ``docs/robots/`` catalog pages. Nothing tied those restated
numbers to the registry, so they drifted independently - the tree simultaneously
claimed "40+", "50+" and "68" robots for a registry holding 72, and two
hardware-only robots (``hope_jr_hand``, ``lekiwi_client``) were absent from
every catalog table, making them undiscoverable to anyone reading the docs.

Four guards, each with a distinct job:

* :func:`test_every_registered_robot_appears_in_a_catalog_table` and its inverse
  pin *membership* - the catalog lists exactly the registered names.
* The ``test_..._claim_matches_registry`` tests pin *the numbers*, and their
  failure message is the exact string to write, so a fix needs no arithmetic.
* :func:`test_no_robot_count_claim_outside_the_known_sites` is the net that
  catches a *new* claim added somewhere none of the above look. The tests above
  are precise about the sites they know; this one refuses an unknown number.
* :func:`test_the_alias_column_lists_every_alias_the_registry_accepts` pins the
  catalog's *Aliases* column cell by cell, so a name ``resolve_name()`` accepts
  cannot be missing from the row that advertises the robot.

Counts are derived from ``robots.json`` directly rather than from
:func:`~strands_robots.registry.list_robots`, because ``list_robots()`` also
returns robots registered at runtime through ``register_robot()`` and from the
user registry on disk, neither of which the docs describe. This mirrors the
reasoning in ``tests/test_docs_policy_coverage.py``.

The alias column is a *faithful projection* of ``robots.json``: every alias the
entry declares, in the order the entry declares it, joined with ", " (or ``-``
when the entry declares none). It used to show only the first three, silently,
which hid 19 accepted names -- including ``franka_panda``, one of the first
spellings a reader guesses. Mirroring the registry's own order rather than
sorting keeps the projection transformation-free, so the expected cell can be
quoted verbatim in the failure message.
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

#: Catalog page -> the registry categories it documents.
CATALOG_PAGES: dict[str, tuple[str, ...]] = {
    "arms.md": ("arm",),
    "bimanual.md": ("bimanual",),
    "hands.md": ("hand",),
    "humanoids.md": ("humanoid", "expressive"),
    "mobile.md": ("mobile", "mobile_manip", "aerial"),
}

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


def _catalog_rows() -> dict[str, str]:
    """Return every name listed in a catalog table, mapped to its page.

    Only the ``## Catalog`` section of each page is read, so a name mentioned in
    a code sample or a "featured render" heading elsewhere on the page does not
    count as being catalogued.
    """
    listed: dict[str, str] = {}
    for page in CATALOG_PAGES:
        text = (DOCS / "robots" / page).read_text(encoding="utf-8")
        section = re.search(r"\n## Catalog\b(.*?)(?:\n## |\Z)", text, re.DOTALL)
        assert section, f"{page} is missing a '## Catalog' section"
        for line in section.group(1).splitlines():
            row = re.match(r"\| `([^`]+)` \|", line)
            if row:
                name = row.group(1)
                assert name not in listed, f"{name} is listed twice ({listed.get(name)}, {page})"
                listed[name] = page
    return listed


def _catalog_alias_cells() -> dict[str, str]:
    """Return each catalogued robot's raw *Aliases* cell, keyed by robot name.

    Reads only the ``## Catalog`` table, and asserts each row has the expected
    four columns so a stray ``|`` inside a cell fails here rather than silently
    shifting which text is read as the alias list.
    """
    cells: dict[str, str] = {}
    for page in CATALOG_PAGES:
        text = (DOCS / "robots" / page).read_text(encoding="utf-8")
        section = re.search(r"\n## Catalog\b(.*?)(?:\n## |\Z)", text, re.DOTALL)
        assert section, f"{page} is missing a '## Catalog' section"
        for line in section.group(1).splitlines():
            row = re.match(r"\| `([^`]+)` \|", line)
            if not row:
                continue
            columns = line.split("|")
            assert len(columns) == 6, (
                f"{page}: expected a 4-column row for {row.group(1)!r}, got {len(columns) - 2}: {line!r}. "
                "A literal '|' inside a cell must be escaped as '\\|'."
            )
            cells[row.group(1)] = columns[4].strip()
    return cells


def _expected_alias_cell(entry: dict) -> str:
    """Return the *Aliases* cell text a registry entry should produce."""
    aliases = entry.get("aliases", [])
    return ", ".join(f"`{alias}`" for alias in aliases) if aliases else "-"


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


def test_every_registered_robot_appears_in_a_catalog_table() -> None:
    """Every name ``Robot()`` accepts is discoverable from the catalog pages."""
    missing = sorted(set(_registry()) - set(_catalog_rows()))
    assert not missing, (
        f"registered but absent from every docs/robots/ catalog table: {missing}. "
        "A robot missing from the catalog cannot be discovered by a reader."
    )


def test_no_catalog_row_for_an_unregistered_robot() -> None:
    """The catalog does not advertise a name the registry cannot resolve."""
    rows = _catalog_rows()
    unknown = sorted(set(rows) - set(_registry()))
    assert not unknown, f"catalogued but not in robots.json: {[(n, rows[n]) for n in unknown]}"


def test_each_robot_is_catalogued_on_the_page_for_its_category() -> None:
    """A robot appears on the page documenting its registry category."""
    registry, rows = _registry(), _catalog_rows()
    page_for = {cat: page for page, cats in CATALOG_PAGES.items() for cat in cats}
    misfiled = [
        (name, rows[name], page_for[registry[name]["category"]])
        for name in sorted(rows)
        if name in registry and rows[name] != page_for.get(registry[name].get("category", ""))
    ]
    assert not misfiled, f"listed on the wrong catalog page (name, listed_on, expected): {misfiled}"


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


def test_the_alias_column_lists_every_alias_the_registry_accepts() -> None:
    """Each catalog row advertises exactly the aliases its registry entry declares.

    ``resolve_name()`` accepts every alias in ``robots.json``, so an alias the
    row omits is a working name no reader can find, and an alias the row invents
    is a name that does not resolve. Both are failures of the same projection,
    so the whole cell is compared and the expected text is quoted verbatim -
    fixing a row is a copy-paste, not a merge.
    """
    registry, cells = _registry(), _catalog_alias_cells()
    wrong = [
        (name, cells[name], _expected_alias_cell(registry[name]))
        for name in sorted(cells)
        if name in registry and cells[name] != _expected_alias_cell(registry[name])
    ]
    assert not wrong, "docs/robots/ alias cells that do not match robots.json:\n" + "\n".join(
        f"  {name}\n    is:     {actual}\n    should: {expected}" for name, actual, expected in wrong
    )
