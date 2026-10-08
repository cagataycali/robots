"""The Isaac Lab page shows the zoo's own Hub clips, and every number under them was read, not typed.

``docs/hooks/hub_videos.py`` expands ``{{hub_videos:<set>}}`` into a grid whose
``<video>`` streams a model repo's ``playback.mp4`` from the Hub and whose caption is
built from the numbers ``--refresh`` cached in ``docs/hooks/data/hub_videos.json``.
Three things keep that honest: each repo the data file names exists on the Hub (skipped
offline), no figure preloads or eagerly fetches anything, and the page's own prose
carries none of the cached numbers (a typed copy would drift the day the model is
retrained).
"""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tests._docs_hooks import docs_hook

_REPO = Path(__file__).resolve().parents[1]
_DOCS = _REPO / "docs"
_DATA = _DOCS / "hooks" / "data" / "hub_videos.json"
_VIDEO = re.compile(r"<video\b[^>]*>", re.I)
_IMG = re.compile(r"<img\b[^>]*>", re.I)


def _hook():
    return docs_hook("hub_videos")


def _data() -> dict:
    return json.loads(_DATA.read_text(encoding="utf-8"))


def _pages_with_tokens() -> dict[Path, set[str]]:
    hook = _hook()
    found = {}
    for page in sorted(_DOCS.rglob("*.md")):
        if "hooks" in page.parts:
            continue
        sets = hook.referenced_sets(page.read_text(encoding="utf-8"))
        if sets:
            found[page] = sets
    return found


def _online() -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError:
        return False
    return True


def test_the_isaaclab_page_references_the_zoo() -> None:
    """The grader has a page to grade."""
    pages = _pages_with_tokens()
    assert _DOCS / "learn" / "training" / "isaaclab.md" in pages
    for page, sets in pages.items():
        assert sets <= set(_data()), f"{page}: {sets - set(_data())} missing from {_DATA.name}"


def test_every_referenced_set_has_cached_numbers() -> None:
    """``--check`` is clean: the build never needs the network."""
    assert _hook().check(_data()) == []


@pytest.mark.parametrize("set_id", sorted(_data()))
def test_the_grid_streams_from_the_hub_and_preloads_nothing(set_id: str) -> None:
    """One figure per repo; the video points at the repo's own mp4 and poster, preload none, no <img>."""
    hook = _hook()
    html = hook.grid_html(set_id)
    assert html is not None
    items = _data()[set_id]["items"]
    videos = _VIDEO.findall(html)
    assert len(videos) == len(items)
    for item, video in zip(items, videos, strict=True):
        assert hook.resolve(item["repo"], "playback.mp4") in video
        assert hook.resolve(item["repo"], "frame.png") in video
        assert 'preload="none"' in video and "muted" in video and "playsinline" in video
    assert not _IMG.findall(html), "the grid uses <video poster>, not an eager <img>"
    assert _data()[set_id]["collection"] in html


@pytest.mark.parametrize("set_id", sorted(_data()))
def test_the_page_types_none_of_the_cached_numbers(set_id: str) -> None:
    """A success rate or reward in prose would drift from the Hub; the caption is the only place."""
    hook = _hook()
    for page, sets in _pages_with_tokens().items():
        if set_id not in sets:
            continue
        text = page.read_text(encoding="utf-8")
        for item in _data()[set_id]["items"]:
            for key in ("success_rate", "mean_reward"):
                value = item.get(key)
                if value is None:
                    continue
                for literal in {f"{value}", f"{value * 100:.0f} percent", f"{value:.1f}"}:
                    assert literal not in text, f"{page.name} types {key}={literal}; let {hook.__name__} render it"


@pytest.mark.skipif(not _online(), reason="huggingface.co unreachable")
@pytest.mark.parametrize("repo", sorted(item["repo"] for entry in _data().values() for item in entry["items"]))
def test_every_named_repo_serves_its_clip(repo: str) -> None:
    """HEAD on the mp4: a renamed or private repo would leave a dead player on the page."""
    url = _hook().resolve(repo, "playback.mp4")
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310 - https URL built by the hook
            assert resp.status == 200
    except urllib.error.HTTPError as exc:
        pytest.fail(f"{url}: HTTP {exc.code}")
