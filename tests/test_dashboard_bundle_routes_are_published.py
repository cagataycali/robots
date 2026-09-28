"""Every /api route the shipped bundle calls is one this server publishes.

The page warns "older server, N features are dark" by matching the bundle's
``BUNDLE_ROUTES`` against ``/openapi.json`` (``frontend/src/lib/darkFeatures.ts``).
Built from the same commit, that list must be empty; a route the bundle calls
that no template matches is either a missing route or a false banner.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from strands_robots.dashboard.server import create_app  # noqa: E402

_APP_JS = Path(__file__).resolve().parents[1] / "strands_robots" / "dashboard" / "static" / "app.js"


def _bundle_routes() -> list[str]:
    block = re.search(r"BUNDLE_ROUTES\s*=\s*\[(.*?)\]", _APP_JS.read_text(encoding="utf-8"), re.S)
    assert block, "static/app.js carries no BUNDLE_ROUTES"
    return re.findall(r'"(/api/[^"]*)"', block.group(1))


def _served(templates: list[str], route: str) -> bool:
    """darkRoutes(): a base ending in '/' is served by anything beneath it; else serverAge.templateMatches."""
    if route.endswith("/"):
        return any(t.startswith(route) for t in templates)
    rx = [re.compile("^" + "[^/]+".join(map(re.escape, re.split(r"\{[^}]*\}", t))) + "$") for t in templates]
    return any(r.match(route) for r in rx)


def test_no_route_the_bundle_calls_is_dark_on_its_own_server():
    templates = list(create_app().openapi()["paths"])
    routes = _bundle_routes()
    assert len(routes) > 50
    assert [r for r in routes if not _served(templates, r)] == []
