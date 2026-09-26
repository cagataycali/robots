"""Content images declare ``loading="lazy"`` in the HTML the build emits.

A browser starts fetching an ``<img>`` the moment the parser reaches its tag,
so an attribute assigned from a page-ready handler arrives after the request
is already in flight - the site read as lazy while every below-the-fold clip
still downloaded. Measured on a 390x844 viewport over an emulated 4G
connection with no scrolling, ``policies/wbc-rollouts`` pulled 664 KB, and
``hardware/universal-robots`` 751 KB, before the fix and 0 KB after it.

``docs/hooks/media.py`` writes the attribute at build time instead. Three
halves have to agree for that to hold, and each has a cell here: the hook
rewrites the tag shapes the converter actually emits, mkdocs runs the hook,
and ``docs/assets/docs.js`` no longer claims the job it cannot do in time.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_HOOK = _REPO / "docs" / "hooks" / "media.py"
_SCRIPT = _REPO / "docs" / "assets" / "docs.js"
_MKDOCS = _REPO / "mkdocs.yml"


def _hook():
    """The media hook, loaded by path: the docs venv is not the test venv."""
    spec = importlib.util.spec_from_file_location("docs_media_hook", _HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ('<img alt="" src="a.gif">', '<img alt="" src="a.gif" loading="lazy">'),
        ('<img src="a.png"/>', '<img src="a.png" loading="lazy"/>'),
        ('<p><img src="a.png" width="480"></p>', '<p><img src="a.png" width="480" loading="lazy"></p>'),
        (
            '<figure><img src="a.svg" class="brand-svg"></figure>',
            '<figure><img src="a.svg" class="brand-svg" loading="lazy"></figure>',
        ),
        ('<img src="a.png" loading="eager">', '<img src="a.png" loading="eager">'),
        ('<img src="a.png" alt="x" loading="lazy">', '<img src="a.png" alt="x" loading="lazy">'),
        ("<p>no image here</p>", "<p>no image here</p>"),
    ],
    ids=["plain", "self-closing", "with-attributes", "in-a-figure", "declared-eager", "already-lazy", "no-image"],
)
def test_the_hook_defers_every_image_that_does_not_choose_for_itself(html: str, expected: str) -> None:
    """A tag stating its own strategy is kept; anything else becomes lazy."""
    assert _hook().lazify(html) == expected


def test_deferring_twice_declares_it_once() -> None:
    """The hook runs per page, and a second pass must not double the attribute."""
    once = _hook().lazify('<img src="a.gif">')
    assert _hook().lazify(once) == once


def test_mkdocs_runs_the_hook() -> None:
    """A hook mkdocs does not list is a file nobody executes."""
    listed = re.findall(r"^\s*-\s*(docs/hooks/[\w.]+)\s*$", _MKDOCS.read_text(encoding="utf-8"), re.M)
    assert "docs/hooks/media.py" in listed, (
        f"mkdocs.yml runs {listed}; without the media hook every content image "
        f"ships eager and the release clips download before the reader scrolls."
    )


def test_the_script_does_not_promise_a_fetch_it_cannot_defer() -> None:
    """Assigning loading after the parse leaves the request already in flight."""
    assert "loading" not in _SCRIPT.read_text(encoding="utf-8"), (
        f"{_SCRIPT.name} sets a loading attribute from a page-ready handler, "
        f"which runs after the parser has started the fetch; the build-time "
        f"hook docs/hooks/media.py is what defers it."
    )
