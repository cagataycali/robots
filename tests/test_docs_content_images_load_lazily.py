"""Content images declare ``loading="lazy"`` in the HTML the build emits.

A browser starts fetching an ``<img>`` the moment the parser reaches its tag,
so an attribute assigned from a page-ready handler arrives after the request
is already in flight: a site reads as lazy while every below-the-fold image
still downloads. Measured on a 390x844 viewport over an emulated 4G
connection with no scrolling, the old release-clip pages pulled 664 KB and
751 KB before the attribute moved into the build, and 0 KB after.

The site's content images are the robot thumbnails ``docs/hooks/robot_pages.py``
writes into the catalog cards, the family pages and every generated robot page,
70 of them on the catalog alone. Three halves have to agree for the fetch to
wait for the scroll, and each has a cell here: the hook writes the attribute on
every tag it emits, mkdocs runs the hook, and no hand-written page, template
or script claims the job the parser has already started.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._docs_hooks import docs_hook

_REPO = Path(__file__).resolve().parents[1]
_DOCS = _REPO / "docs"
_OVERRIDES = _REPO / "overrides"
_MKDOCS = _REPO / "mkdocs.yml"

_IMG_TAG = re.compile(r"<img\b[^>]*>", re.I)
_LOADING = re.compile(r'\bloading="(lazy|eager)"')
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?")
_SCRIPT_SETS_LOADING = re.compile(r"""\.loading\s*=|setAttribute\(\s*["']loading["']""")


def _hook():
    """The robot-pages hook, loaded by path: the docs venv is not the test venv."""
    return docs_hook("robot_pages")


def _emitted_html() -> dict[str, str]:
    """Every chunk of markup the hook can put in a page, by the page it serves."""
    hook = _hook()
    chunks = {"robots/index.md (cards)": hook.cards(None, "")}
    chunks.update({f"robots/{name}.md": hook.robot_page(name) for name in hook.registry()})
    chunks.update({f"robots/{family}/index.md": hook.family_page(family) for family in hook._families_in_order()})
    return chunks


def _declared_scripts() -> list[Path]:
    """Local scripts mkdocs.yml declares under extra_javascript."""
    text = _MKDOCS.read_text(encoding="utf-8")
    match = re.search(r"^extra_javascript:\s*$(.*?)^\S", text, re.M | re.S)
    assert match, "mkdocs.yml declares no extra_javascript block"
    paths = re.findall(r"^\s*-\s*(?:path:\s*)?(\S+)\s*$", match.group(1), re.M)
    return [_DOCS / path for path in paths if not path.startswith("http")]


def test_the_hook_defers_every_image_it_emits() -> None:
    """Every ``<img>`` the hook writes states its strategy, and states it once."""
    eager: list[str] = []
    doubled: list[str] = []
    for page, html in _emitted_html().items():
        for tag in _IMG_TAG.findall(html):
            strategies = _LOADING.findall(tag)
            if not strategies:
                eager.append(f"{page}: {tag[:80]}")
            elif len(strategies) > 1:
                doubled.append(f"{page}: {tag[:80]}")
    assert not eager, (
        f"docs/hooks/robot_pages.py emits images without loading=: {eager[:5]} "
        f"({len(eager)} total). The catalog page carries one thumbnail per robot, "
        f"and every one of them downloads before the reader scrolls."
    )
    assert not doubled, f"an <img> declares loading twice: {doubled[:5]}"


def test_the_catalog_thumbnails_are_lazy() -> None:
    """The catalog is the page with the most images; its cards are all deferred."""
    cards = _emitted_html()["robots/index.md (cards)"]
    tags = _IMG_TAG.findall(cards)
    assert tags, "the catalog cards emit no thumbnails; the hook lost its <img>"
    assert all('loading="lazy"' in tag for tag in tags), "a catalog thumbnail is not lazy"


def test_mkdocs_runs_the_hook() -> None:
    """A hook mkdocs does not list is a file nobody executes."""
    listed = re.findall(r"^\s*-\s*(docs/hooks/[\w.]+)\s*$", _MKDOCS.read_text(encoding="utf-8"), re.M)
    assert "docs/hooks/robot_pages.py" in listed, (
        f"mkdocs.yml runs {listed}; without the robot-pages hook the catalog has "
        f"no cards and the {{{{robot_cards}}}} token ships as text."
    )


@pytest.mark.parametrize(
    "root",
    [_DOCS, _OVERRIDES],
    ids=["pages", "templates"],
)
def test_no_hand_written_image_is_eager(root: Path) -> None:
    """A page or template that places an image says how it loads.

    Markdown images take the attribute through ``attr_list``
    (``![alt](src){ loading=lazy }``); raw tags carry it inline.
    """
    offenders: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".md", ".html"} or "hooks" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO)
        offenders += [f"{rel}: {tag[:80]}" for tag in _IMG_TAG.findall(text) if not _LOADING.search(tag)]
        offenders += [
            f"{rel}: {m.group(0)[:80]}" for m in _MD_IMAGE.finditer(text) if "loading=" not in (m.group(1) or "")
        ]
    assert not offenders, f"images with no loading strategy: {offenders[:10]}"


def test_no_script_promises_a_fetch_it_cannot_defer() -> None:
    """Assigning loading after the parse leaves the request already in flight."""
    scripts = _declared_scripts()
    assert scripts, "mkdocs.yml declares no local script; the viewer component is gone"
    for script in scripts:
        assert not _SCRIPT_SETS_LOADING.search(script.read_text(encoding="utf-8")), (
            f"{script.name} sets a loading attribute from script, which runs after "
            f"the parser has started the fetch; docs/hooks/robot_pages.py writes it "
            f"at build time."
        )
