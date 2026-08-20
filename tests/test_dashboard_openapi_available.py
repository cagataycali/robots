"""The app must keep publishing /openapi.json — a UI feature now depends on it (Q79).

The frontend tells "this server has no such route" apart from "this server has no such camera" by asking
the server for its own route list. Measured on this Mac 2026-08-20: the running dashboard was missing
EIGHT routes the shipped bundle calls (camera modes, deploy snippet, policy-fit, checkpoint features,
network hint, training output-dir, per-peer cameras, spawn-remembered) because the process predates them,
and each one failed with a bare 404 that reads like the resource is absent rather than the server old.

So this is not decoration: turning the schema off (openapi_url=None, a common "harden the API" move)
would silently take that explanation away and hand those features their misleading 404s back. The test
exists to make that a decision rather than an accident.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard import server as srv


class _StubBridge:
    peers: dict = {}

    def snapshot(self):
        return {"peers": {}}

    def start(self):
        pass

    def stop(self):
        pass


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """This machine's real settings.json carries a live token; an un-isolated client 401s (and Q62:
    an override outlives the test that made it)."""
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}


def _client() -> TestClient:
    return TestClient(srv.create_app(bridge=_StubBridge()))


def test_openapi_lists_the_api_routes() -> None:
    body = _client().get("/openapi.json").json()
    paths = body["paths"]
    # A handful of representative shapes, including a templated one: the frontend's matcher turns
    # "{index}" into one path segment, so the template form must survive here.
    assert "/api/fleet" in paths
    assert "/api/devices/camera/{index}/modes" in paths
    assert sum(1 for p in paths if p.startswith("/api/")) > 40


def test_every_route_the_app_registers_is_published() -> None:
    """A route missing from the schema would be judged "this server is too old" while it works."""
    app = srv.create_app(bridge=_StubBridge())
    registered = {
        r.path
        for r in app.routes
        if getattr(r, "path", "").startswith("/api/") and getattr(r, "include_in_schema", True)
    }
    published = set(TestClient(app).get("/openapi.json").json()["paths"])
    missing = sorted(registered - published)
    assert not missing, f"routes the schema hides from the frontend's age check: {missing}"
