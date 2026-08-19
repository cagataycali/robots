"""An unrouted /api path must be a 404, not the SPA shell.

Q21 in BUGS.md, and the first link in the record-panel crash chain: the SPA
catch-all answered every unknown /api path with 200 text/html, so
`recordApi`'s "use the real backend if it answers" probe never fell back to
its mock, and a plain fetch().json() died on "Unexpected token '<'" instead
of seeing a status it could branch on.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard import server as srv


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """The real machine's settings.json carries a live auth token; a test
    client that reads it gets a 401 before it ever reaches the catch-all."""
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}


class _StubBridge:
    """create_app() with no bridge builds a real mesh session (and this machine's
    env demands mtls); the catch-all does not care what the bridge is."""

    peers: dict = {}

    def snapshot(self):
        return {"peers": {}}

    def start(self):
        pass

    def stop(self):
        pass


def _client() -> TestClient:
    return TestClient(srv.create_app(bridge=_StubBridge()))


def test_unknown_api_path_is_json_404() -> None:
    c = _client()
    r = c.get("/api/nope/does-not-exist")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["error"] == "not found"
    assert "/api/nope/does-not-exist" in body["detail"]


def test_unknown_api_path_never_returns_html() -> None:
    # the exact shape the frontend probes: a route family that does not exist
    c = _client()
    for path in ("/api/record/status", "/api/definitely-not-here", "/api"):
        r = c.get(path)
        assert r.status_code in (404, 200), path
        if r.status_code == 200:
            # a real endpoint answered (fine) - it just must not be the shell
            assert "<!doctype html" not in r.text[:200].lower(), path
        else:
            assert "text/html" not in r.headers["content-type"], path


def test_http_get_on_a_websocket_path_is_404_not_a_page() -> None:
    c = _client()
    r = c.get("/ws/chat")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")


def test_real_spa_route_still_serves_the_shell() -> None:
    if not srv.FRONTEND_DIST.exists():  # unbuilt checkout
        return
    c = _client()
    r = c.get("/settings")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
