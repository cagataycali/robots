"""The site is one nav organised by purpose, and no page falls out of it.

A reader arriving from a link and an agent resolving a symbol want opposite
things from the same site. The site used to serve both with two lanes: a short
nav for newcomers and a generated index for the reference tree. The rewrite
folds both into one nav of six tabs (Home, Start, Robots, Learn, Reference,
Project), each tab a directory under ``docs/``, so the URL and the sidebar
say the same thing about a page. The reference lane is now the ``Reference``
tab: a hand-written index that points at the API, tool, CLI, configuration and
refusal-code pages, every one of them generated from the source.

Three ways that arrangement can quietly break, each graded here:

* **A page falls out of the nav.** MkDocs reports a page missing from the nav
  as ``INFO``, which ``--strict`` does not fail on, so an unnavigated page
  ships and is reachable only by search. Every page under ``docs/`` is in the
  nav or in ``not_in_nav``.
* **The nav stops matching the tree.** A page filed under one tab but living
  under another directory gives the reader two names for the same place, and
  a seventh tab clips the strip. The tabs are the IA's six, and every page a
  tab lists lives under that tab's directory.
* **A published URL stops resolving.** The old site had 115 pages, and every
  one of those URLs is in the wild: in a release note, an issue, a bookmark.
  ``mkdocs-redirects`` serves each old URL a redirect to its new home, and a
  redirect is graded both ways: its source must not be a page on disk (that
  would shadow a real page), its destination must be, and no redirect points
  at another redirect (the plugin does not chain).
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DOCS = _REPO / "docs"
_MKDOCS = _REPO / "mkdocs.yml"

#: Tab label -> the docs directory (or page) it lists. The landing page is a
#: tab of its own so the logo and "Home" agree.
TABS: dict[str, str] = {
    "Home": "index.md",
    "Start": "start/",
    "Robots": "robots/",
    "Learn": "learn/",
    "Reference": "reference/",
    "Project": "project/",
}

#: The reference index and the API index are hand-written tables; each names
#: the pages beside it exactly once.
REFERENCE_INDEX = "reference/index.md"
API_INDEX = "reference/api/index.md"

#: The old nav listed 115 pages; a redirect map that shrank below that lost
#: URLs that are in the wild.
MINIMUM_REDIRECTS = 115

_NAV_ITEM = re.compile(r"^(?P<indent> *)- (?P<body>.+?)\s*$")
_REDIRECT_ROW = re.compile(r"^\s+(?P<source>[\w./-]+\.md): (?P<target>[\w./-]+\.md)$", re.M)
_LINK = re.compile(r"\[[^\]]*\]\((?P<target>[^)\s#]+)")


def _nav_lines() -> list[str]:
    lines = _MKDOCS.read_text(encoding="utf-8").splitlines()
    start = lines.index("nav:") + 1
    out: list[str] = []
    for line in lines[start:]:
        if not line.strip():
            continue
        if not line.startswith((" ", "-")):
            break
        out.append(line)
    assert out, "no nav entries parsed out of mkdocs.yml"
    return out


def _nav_targets() -> list[str]:
    """Every page path the nav points at, in nav order."""
    targets: list[str] = []
    for line in _nav_lines():
        match = _NAV_ITEM.match(line)
        assert match, f"mkdocs.yml nav line is not a list item: {line!r}"
        body = match["body"]
        target = body.split(":", 1)[1].strip() if ":" in body else body
        if target.endswith(".md"):
            targets.append(target)
    return targets


def _tabs() -> dict[str, list[str]]:
    """Top-level nav label -> every page path listed beneath it."""
    tabs: dict[str, list[str]] = {}
    current = ""
    for line in _nav_lines():
        match = _NAV_ITEM.match(line)
        assert match, f"mkdocs.yml nav line is not a list item: {line!r}"
        if match["indent"] == "  ":
            label, _, rest = match["body"].partition(":")
            current = label.strip()
            tabs[current] = [rest.strip()] if rest.strip().endswith(".md") else []
        else:
            body = match["body"]
            target = body.split(":", 1)[1].strip() if ":" in body else body
            if target.endswith(".md"):
                tabs[current].append(target)
    return tabs


def _not_in_nav() -> set[str]:
    """Pages mkdocs.yml explicitly keeps out of the nav."""
    match = re.search(r"^not_in_nav:\s*\|\s*$(.*?)^\S", _MKDOCS.read_text(encoding="utf-8"), re.M | re.S)
    if not match:
        return set()
    return {line.strip() for line in match.group(1).splitlines() if line.strip()}


def _redirects() -> dict[str, str]:
    """The ``redirect_maps`` table as ``{old page: new page}``."""
    config = _MKDOCS.read_text(encoding="utf-8")
    block = config.split("redirect_maps:", 1)
    assert len(block) == 2, "mkdocs.yml declares no redirect_maps; moved URLs resolve to nothing"
    body = block[1].split("\nnav:", 1)[0]
    return {m["source"]: m["target"] for m in _REDIRECT_ROW.finditer(body)}


def _pages() -> set[str]:
    """Every page under ``docs/``, as a docs-relative posix path."""
    return {p.relative_to(_DOCS).as_posix() for p in _DOCS.rglob("*.md")}


def _links_of(page: str) -> list[str]:
    """Relative ``.md`` links in a page, resolved to docs paths, in order."""
    base = Path(page).parent.as_posix()
    text = (_DOCS / page).read_text(encoding="utf-8")
    targets = [m["target"] for m in _LINK.finditer(text)]
    return [_normalise(f"{base}/{t}") for t in targets if t.endswith(".md") and not t.startswith(("http", "/"))]


def _normalise(path: str) -> str:
    """Collapse ``.`` and ``..`` segments of a docs-relative posix path."""
    parts: list[str] = []
    for part in path.split("/"):
        if part == "..":
            parts.pop()
        elif part and part != ".":
            parts.append(part)
    return "/".join(parts)


class TestNoPageFallsOutOfTheNav:
    """Every page is in the nav, and nothing is in the nav twice."""

    def test_every_page_is_in_the_nav_or_declared_out_of_it(self) -> None:
        reachable = set(_nav_targets()) | _not_in_nav()
        orphans = sorted(_pages() - reachable)
        assert not orphans, (
            f"pages under docs/ the nav does not reach: {orphans}. A page nothing links to "
            f"is found only by search: list it under its tab, or in not_in_nav when it is "
            f"reached from a page instead."
        )

    def test_every_nav_entry_is_a_page_and_is_listed_once(self) -> None:
        targets = _nav_targets()
        missing = sorted(set(targets) - _pages())
        assert not missing, f"mkdocs.yml nav points at pages that are not on disk: {missing}"
        doubled = sorted({t for t in targets if targets.count(t) > 1})
        assert not doubled, f"mkdocs.yml nav lists a page twice: {doubled}"


class TestTheNavMatchesTheTree:
    """Six tabs, each one a directory, and the URL agrees with the sidebar."""

    def test_the_tabs_are_the_six_of_the_information_architecture(self) -> None:
        assert list(_tabs()) == list(TABS), (
            f"mkdocs.yml nav has top-level sections {list(_tabs())}; the design has "
            f"{list(TABS)}. Material lays the tab strip out in one row, and a seventh "
            f"tab is the first one it clips."
        )

    def test_every_page_lives_under_the_directory_its_tab_names(self) -> None:
        strays = {
            label: [page for page in pages if not page.startswith(TABS[label])] for label, pages in _tabs().items()
        }
        strays = {label: pages for label, pages in strays.items() if pages}
        assert not strays, (
            f"nav tabs list pages outside their own directory: {strays}. The URL and the "
            f"sidebar say different things about where a page is."
        )


class TestTheReferenceIndexIsTheTree:
    """The hand-written reference tables and the directory agree, both ways."""

    def test_the_reference_index_names_every_reference_page_once(self) -> None:
        listed = _links_of(REFERENCE_INDEX)
        expected = sorted(
            p for p in _pages() if p.startswith("reference/") and p.count("/") == 1 and p != REFERENCE_INDEX
        )
        expected.append(API_INDEX)
        own = [t for t in listed if t.startswith("reference/")]
        assert sorted(own) == sorted(expected), (
            f"docs/{REFERENCE_INDEX} and docs/reference/ disagree: "
            f"missing {sorted(set(expected) - set(own))}, extra {sorted(set(own) - set(expected))}."
        )
        assert len(own) == len(set(own)), "the reference index lists a page twice"

    def test_the_api_index_names_every_api_page_once(self) -> None:
        listed = [t for t in _links_of(API_INDEX) if t.startswith("reference/api/")]
        expected = sorted(p for p in _pages() if p.startswith("reference/api/") and p != API_INDEX)
        assert sorted(listed) == expected, (
            f"docs/{API_INDEX} and docs/reference/api/ disagree: "
            f"missing {sorted(set(expected) - set(listed))}, extra {sorted(set(listed) - set(expected))}."
        )
        assert len(listed) == len(set(listed)), "the API index lists a page twice"


class TestEveryMovedUrlStillResolves:
    """A redirect for each retired URL, pointing at a page that exists."""

    def test_the_redirect_map_covers_the_old_site(self) -> None:
        redirects = _redirects()
        assert len(redirects) >= MINIMUM_REDIRECTS, (
            f"redirect_maps holds {len(redirects)} entries; the old site had "
            f"{MINIMUM_REDIRECTS} pages and each URL is in the wild."
        )

    def test_no_redirect_shadows_a_page_or_points_at_a_missing_one(self) -> None:
        pages = _pages()
        redirects = _redirects()
        shadowing = sorted(source for source in redirects if source in pages)
        assert not shadowing, (
            f"redirect_maps sources that are real pages: {shadowing}. A redirect at a live "
            f"URL hides the page it sits on."
        )
        dangling = sorted(target for target in redirects.values() if target not in pages)
        assert not dangling, f"redirect_maps targets that are not pages on disk: {dangling}"

    def test_no_redirect_points_at_another_redirect(self) -> None:
        redirects = _redirects()
        chained = sorted(f"{source} -> {target}" for source, target in redirects.items() if target in redirects)
        assert not chained, f"redirects that land on a redirect (mkdocs-redirects does not chain): {chained}"
