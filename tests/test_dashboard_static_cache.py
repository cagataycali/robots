"""Cache-Control on the built frontend (Q96).

MEASURED on the running dashboard before this existed: /index.html and /sw.js came back with an ETag
and NO Cache-Control, and so did every hashed asset. With no Cache-Control a browser may INVENT
freshness from Last-Modified (typically 10% of the file's age), so a reload can serve index.html from
its own cache without asking the server - the structural other half of the eleven-hour-old bundle a
phone in Seattle was running, and one no amount of service-worker polling can cure, because the poll
never gets to happen.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard.server import create_app, static_cache_control

IMMUTABLE = "public, max-age=31536000, immutable"

# REAL filenames from a real dist, not invented ones: my first pattern wanted a dot-separated hash
# (index.BB6lyXA6.css) and matched none of these, so every asset silently stayed no-cache and the
# "fix" changed nothing at all.
HASHED = [
    # Q116, THE REGRESSION THIS FILE MISSED FOR A DAY: a real vite hash with NO DIGIT IN IT. Every
    # example above happened to contain one, so the "must contain a digit" rule looked correct and
    # refused the main bundle of one build in four (8 base64 chars, 10 of 64 are digits => ~25%).
    # The defect came and went with the hash, which is why a green test could sit on top of it.
    "index-BGRlFtdn.js",
    "assets/index-BGRlFtdn.js",
    # The DIRECTORY is the evidence: vite content-hashes everything it emits into assets/, so a name
    # this function cannot recognise is still immutable when it lives there. Both call sites pass a
    # path that keeps the directory (StaticFiles gives the real file path; the fallback route the
    # URL sub-path), and this is the shape a Windows path arrives in.
    "assets/whatever-vite-emits-next.js",
    r"C:\\dist\\assets\\index-BGRlFtdn.js",
    "index-BB6lyXA6.css",
    "index-CxO2NtX_.js",
    "workbox-window.prod.es5-BqEJf4Xk.js",
    "workbox-e97c6ee1.js",
    "assets/index-BB6lyXA6.css",
]
# Entry points. Getting ANY of these from a cache pins the whole app at an old build.
ENTRY = ["index.html", "sw.js", "registerSW.js", "manifest.webmanifest", ""]
# Named, unhashed assets. apple-touch-icon.png is the trap: allowing a hyphen inside the hash matched
# it, and a year-long immutable cache on a file whose name never changes cannot be fixed from here.
UNHASHED = ["apple-touch-icon.png", "maskable-192.png", "icon-192.png", "icon.svg",
            # The reason the digit rule could not simply be DROPPED: eight lowercase letters that a
            # person typed. Mixed case and digits are what a hash has and a hand-written name does
            # not, so this must stay revalidated - its bytes can change under a name that never does.
            "favicon-original.png", "sw-registration-helper.js",
            "strands-dashboard.png", "robots.txt",
            # These two are the mutation-hardening cases. apple-touch-icon.png survives a LOOSER
            # charset only because of the digit rule, and maskable-192.png survives a dropped digit
            # rule only because "192" is short - so a name with BOTH a hyphenated tail and a digit is
            # the case that needs stating outright, or neither guard is really pinned.
            "apple-touch-icon-192.png", "camera-preview-2x.png"]


@pytest.mark.parametrize("name", HASHED)
def test_a_content_hashed_asset_is_immutable(name: str) -> None:
    assert static_cache_control(name) == IMMUTABLE


@pytest.mark.parametrize("name", ENTRY)
def test_an_entry_point_must_be_revalidated(name: str) -> None:
    # no-cache, NOT no-store: the ETag still turns an unchanged file into a 304, so honesty here does
    # not cost the bytes again on every reload.
    assert static_cache_control(name) == "no-cache"


@pytest.mark.parametrize("name", UNHASHED)
def test_an_unhashed_name_is_never_cached_forever(name: str) -> None:
    assert static_cache_control(name) != IMMUTABLE


def test_every_real_dist_asset_is_recognised_as_hashed() -> None:
    """The immutable label must actually apply to the files vite really emits.

    A tool that can be NARROWED reports how much it checked: if the dist is absent this skips
    LOUDLY rather than passing on an empty loop.
    """
    assets = Path(__file__).resolve().parents[1] / "strands_robots/dashboard/frontend/dist/assets"
    if not assets.is_dir():
        pytest.skip(f"no built dist at {assets} - build the frontend to check real filenames")
    names = sorted(p.name for p in assets.iterdir() if p.is_file())
    assert names, f"{assets} is empty"
    missed = [n for n in names if static_cache_control(n) != IMMUTABLE]
    assert not missed, (
        f"{len(missed)} of {len(names)} real dist assets were not recognised as content-hashed "
        f"and would be revalidated on every load: {missed}"
    )


def test_the_served_response_actually_carries_the_header() -> None:
    """The rule is worthless if the route does not apply it - the defect was in the RESPONSE."""
    dist = Path(__file__).resolve().parents[1] / "strands_robots/dashboard/frontend/dist"
    if not (dist / "index.html").is_file():
        pytest.skip("no built dist - cannot exercise the static routes")
    os.environ.setdefault("STRANDS_DASHBOARD_NO_MESH", "1")
    with TestClient(create_app()) as client:
        r = client.get("/index.html")
        assert r.status_code == 200
        assert r.headers.get("cache-control") == "no-cache", r.headers

        spa = client.get("/some/client/route")
        assert spa.status_code == 200
        assert spa.headers.get("cache-control") == "no-cache", "an SPA fallback IS the entry point"

        assets = dist / "assets"
        asset = next((p for p in sorted(assets.iterdir()) if p.is_file()), None)
        if asset is not None:
            a = client.get(f"/assets/{asset.name}")
            assert a.status_code == 200
            assert a.headers.get("cache-control") == IMMUTABLE, a.headers
