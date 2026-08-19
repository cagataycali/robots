"""The /api and /ws gate: static token, WebAuthn JWT session, local-only open posture.

The middleware is exercised as raw ASGI (no zenoh, no lifespan); the auth
routes through the real app with a stub bridge and no startup events.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from strands_robots.dashboard import auth
from strands_robots.dashboard import server as srv
from strands_robots.dashboard.server import PUBLIC_PATHS, TokenAuthMiddleware


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", raising=False)
    auth._cache_key = None
    auth._cache = {}
    # never let a machine-level static token leak into these tests
    monkeypatch.setattr(srv.settings, "get", lambda *a, **k: None)
    yield


# --- raw ASGI harness ---------------------------------------------------------


class Passed(Exception):
    """Inner app was reached."""


async def _inner_app(scope, receive, send):
    raise Passed()


def _http_scope(path, client=("127.0.0.1", 4444), headers=None, method="GET"):
    return {
        "type": "http",
        "method": method,
        "path": path,
        "client": client,
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "query_string": b"",
    }


def run_scope(scope):
    loop = asyncio.new_event_loop()
    try:
        mw = TokenAuthMiddleware(_inner_app)
        sent = []

        async def receive():
            return {"type": "websocket.connect"}

        async def send(message):
            sent.append(message)

        try:
            loop.run_until_complete(mw(scope, receive, send))
        except Passed:
            return "passed"
        for m in sent:
            if m["type"] == "http.response.start":
                return m["status"]
            if m["type"] == "websocket.close":
                return m["code"]
        return None
    finally:
        loop.close()


# --- open posture: local-only -------------------------------------------------


def test_open_posture_allows_loopback():
    assert run_scope(_http_scope("/api/fleet", client=("127.0.0.1", 1))) == "passed"


def test_open_posture_allows_testclient_and_missing_client():
    assert run_scope(_http_scope("/api/fleet", client=("testclient", 1))) == "passed"
    assert run_scope(_http_scope("/api/fleet", client=None)) == "passed"


def test_open_posture_refuses_lan_client():
    assert run_scope(_http_scope("/api/fleet", client=("192.168.1.50", 1))) == 401


def test_open_posture_lan_client_can_load_shell_and_health():
    assert run_scope(_http_scope("/", client=("192.168.1.50", 1))) == "passed"
    assert run_scope(_http_scope("/api/health", client=("192.168.1.50", 1))) == "passed"


def test_open_posture_refuses_lan_websocket_with_1008():
    scope = _http_scope("/ws/mesh", client=("192.168.1.50", 1))
    scope["type"] = "websocket"
    scope.pop("method")
    assert run_scope(scope) == 1008


# --- passkeys enabled ----------------------------------------------------------


def test_passkeys_on_requires_session_even_from_loopback(monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "true")
    assert run_scope(_http_scope("/api/fleet", client=("127.0.0.1", 1))) == 401


def test_passkeys_on_ceremony_endpoints_stay_public(monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "true")
    for path in sorted(PUBLIC_PATHS):
        assert run_scope(_http_scope(path, client=("203.0.113.7", 1))) == "passed", path


def test_passkeys_on_valid_jwt_passes(monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "true")
    token = auth.issue_token("cred1", name="phone")
    scope = _http_scope("/api/fleet", client=("203.0.113.7", 1),
                        headers={"authorization": f"Bearer {token}"})
    assert run_scope(scope) == "passed"


def test_passkeys_on_jwt_in_ws_query_string(monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "true")
    token = auth.issue_token("cred1")
    scope = _http_scope("/ws/mesh", client=("203.0.113.7", 1))
    scope["type"] = "websocket"
    scope.pop("method")
    scope["query_string"] = f"token={token}".encode()
    assert run_scope(scope) == "passed"


def test_passkeys_on_garbage_jwt_refused(monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "true")
    scope = _http_scope("/api/fleet", client=("127.0.0.1", 1),
                        headers={"authorization": "Bearer nonsense"})
    assert run_scope(scope) == 401


# --- static token still honoured ------------------------------------------------


def test_static_token_passes_and_jwt_also_accepted(monkeypatch):
    monkeypatch.setattr(srv.settings, "get", lambda *a, **k: "sekrit-static")
    ok = _http_scope("/api/fleet", client=("203.0.113.7", 1),
                     headers={"authorization": "Bearer sekrit-static"})
    assert run_scope(ok) == "passed"
    jwt_scope = _http_scope("/api/fleet", client=("203.0.113.7", 1),
                            headers={"authorization": f"Bearer {auth.issue_token('c')}"})
    assert run_scope(jwt_scope) == "passed"
    bad = _http_scope("/api/fleet", client=("203.0.113.7", 1),
                      headers={"authorization": "Bearer wrong"})
    assert run_scope(bad) == 401


# --- the real routes (stub bridge, no lifespan) ---------------------------------


class _StubBridge:
    peer_id = "dash-test"
    peers: dict = {}

    def snapshot(self):
        return {"peers": {}}


@pytest.fixture()
def client():
    from starlette.testclient import TestClient

    app = srv.create_app(bridge=_StubBridge())
    # no `with`: lifespan (mesh startup) must not run
    return TestClient(app)


def test_auth_status_route(client):
    out = client.get("/api/auth/status").json()
    assert out["setup_required"] is True
    assert out["authenticated"] is False
    assert out["enabled"] is False


def test_register_begin_first_time_open_then_gated(client, tmp_path):
    out = client.post("/api/auth/register/begin", json={"label": "phone"})
    assert out.status_code == 200
    assert out.json()["challenge_id"]
    # simulate an enrolled credential -> second enrollment needs a session
    path = tmp_path / "auth.json"
    data = json.loads(path.read_text())
    data["credentials"] = [{"id": "c1", "public_key": "cGs", "sign_count": 0}]
    path.write_text(json.dumps(data))
    import os as _os

    _os.utime(path, (time.time() + 2, time.time() + 2))
    denied = client.post("/api/auth/register/begin", json={"label": "second"})
    assert denied.status_code == 401
    token = auth.issue_token("c1")
    allowed = client.post(
        "/api/auth/register/begin",
        json={"label": "second"},
        headers={"authorization": f"Bearer {token}"},
    )
    assert allowed.status_code == 200


def test_login_begin_without_enrollment_is_400(client):
    assert client.post("/api/auth/login/begin").status_code == 400
