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


def test_open_posture_refuses_proxied_client_even_from_loopback():
    # cloudflared/nginx connect FROM 127.0.0.1 on behalf of remote clients;
    # any forwarding header means the ORIGINAL client is not local.
    for header in ("cf-connecting-ip", "x-forwarded-for", "x-real-ip"):
        scope = _http_scope(
            "/api/fleet", client=("127.0.0.1", 1), headers={header: "203.0.113.9"}
        )
        assert run_scope(scope) == 401, f"{header} must defeat the loopback pass"


def test_open_posture_proxied_websocket_refused_with_1008():
    scope = _http_scope("/ws/chat", client=("127.0.0.1", 1), headers={"cf-connecting-ip": "203.0.113.9"})
    scope["type"] = "websocket"
    scope.pop("method")
    assert run_scope(scope) == 1008


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


# --- Q20: browser cross-origin writes are refused by the guard itself ----------
#
# CORS only hides the RESPONSE; the side effect of a mutating request fires
# regardless, and a header-less POST (e-stop needs no body) is a "simple
# request" browsers send with no preflight. The guard therefore refuses writes
# and websocket handshakes whose Origin disagrees with the Host and is not in
# security.cors_origins. Clients without an Origin header (curl, the spawn
# watcher) are untouched.


def _origin_scope(method="POST", origin="https://evil.example", host="localhost:8090", path="/api/agent/reset"):
    return _http_scope(
        path, client=("127.0.0.1", 1), method=method,
        headers={"origin": origin, "host": host},
    )


def test_cross_origin_post_refused_even_from_loopback():
    assert run_scope(_origin_scope()) == 403


def test_cross_origin_delete_and_put_refused():
    assert run_scope(_origin_scope(method="DELETE")) == 403
    assert run_scope(_origin_scope(method="PUT")) == 403


def test_same_origin_post_passes():
    assert run_scope(_origin_scope(origin="http://localhost:8090")) == "passed"


def test_no_origin_post_passes_loopback():
    assert run_scope(_http_scope("/api/agent/reset", client=("127.0.0.1", 1), method="POST")) == "passed"


def test_cross_origin_get_is_a_read_and_passes_the_origin_check():
    # reads have no side effect; without an ACAO header the page cannot see
    # the body anyway. The 401/loopback rules still apply afterwards.
    assert run_scope(_origin_scope(method="GET")) == "passed"


def test_allowlisted_origin_post_passes(monkeypatch):
    from strands_robots.dashboard import server as srv2

    def fake_get(section, key, default=None):
        if (section, key) == ("security", "cors_origins"):
            return ["https://tools.example"]
        return None

    monkeypatch.setattr(srv2.settings, "get", fake_get)
    assert run_scope(_origin_scope(origin="https://tools.example")) == "passed"
    assert run_scope(_origin_scope(origin="https://evil.example")) == 403


def test_cross_origin_websocket_refused_with_1008():
    scope = _origin_scope(path="/ws/chat")
    scope["type"] = "websocket"
    scope.pop("method")
    assert run_scope(scope) == 1008


def test_same_origin_websocket_passes():
    scope = _origin_scope(path="/ws/chat", origin="http://localhost:8090")
    scope["type"] = "websocket"
    scope.pop("method")
    assert run_scope(scope) == "passed"


def _with_cors(monkeypatch, origins):
    from strands_robots.dashboard import server as srv2

    def fake_get(section, key, default=None):
        if (section, key) == ("security", "cors_origins"):
            return origins
        return None

    monkeypatch.setattr(srv2.settings, "get", fake_get)


def test_wildcard_cors_does_not_license_cross_origin_writes(monkeypatch):
    # The default is [] now, but installs that ran the old default persisted
    # ["*"] into settings.json - and "*" answers "who may READ", never "who
    # may move the arms". A wildcard must not turn any open tab into a
    # motor-control client.
    _with_cors(monkeypatch, ["*"])
    assert run_scope(_origin_scope()) == 403
    assert run_scope(_origin_scope(method="DELETE")) == 403


def test_wildcard_cors_still_allows_reads_and_same_origin_writes(monkeypatch):
    _with_cors(monkeypatch, ["*"])
    assert run_scope(_origin_scope(method="GET")) == "passed"
    assert run_scope(_origin_scope(origin="http://localhost:8090")) == "passed"


def test_wildcard_cors_does_not_license_cross_origin_websockets(monkeypatch):
    _with_cors(monkeypatch, ["*"])
    scope = _origin_scope(path="/ws/chat")
    scope["type"] = "websocket"
    scope.pop("method")
    assert run_scope(scope) == 1008


def test_wildcard_beside_a_named_origin_keeps_only_the_named_one(monkeypatch):
    _with_cors(monkeypatch, ["*", "https://tools.example"])
    assert run_scope(_origin_scope(origin="https://tools.example")) == "passed"
    assert run_scope(_origin_scope(origin="https://evil.example")) == 403


# --- the allowlist itself ------------------------------------------------------
# Q128: the tests above prove every PUBLIC_PATHS entry PASSES the gate. Nothing proved the entries
# still name real routes, and that is the direction which locks everybody out: membership is exact
# string equality (`path in PUBLIC_PATHS`), so renaming a ceremony route leaves its exemption
# pointing at nothing and the new name is gated behind the very session it exists to create. Remote
# users get a 401 loop and a login form that can never succeed — the Aug-19 iOS wedge's shape, from
# one rename. Both directions are cheap to pin from the source, so they are pinned here.

import re as _re
from pathlib import Path as _Path

_ROUTE_RE = _re.compile(r'@(\w+)\.(get|post|put|patch|delete|websocket)\(\s*["\']([^"\']+)["\']')
_PREFIX_RE = _re.compile(r'(\w+)\s*=\s*APIRouter\(\s*prefix\s*=\s*["\']([^"\']*)["\']')


def _declared_routes() -> set[str]:
    """Every route path the dashboard package registers, prefixes resolved."""
    dash = _Path(srv.__file__).parent
    found: set[str] = set()
    for f in dash.rglob("*.py"):
        text = f.read_text(errors="ignore")
        prefixes = {m.group(1): m.group(2) for m in _PREFIX_RE.finditer(text)}
        for m in _ROUTE_RE.finditer(text):
            found.add(prefixes.get(m.group(1), "") + m.group(3))
    assert len(found) >= 60, f"route discovery collapsed ({len(found)}) — this test is broken"
    return found


def test_every_public_path_names_a_real_route():
    routes = _declared_routes()
    dead = sorted(p for p in PUBLIC_PATHS if p not in routes)
    assert not dead, (
        "PUBLIC_PATHS exempts paths that no route serves — if one of these was RENAMED, the new "
        f"name is now behind the login it is supposed to perform: {dead}"
    )


def test_no_public_path_carries_a_parameter():
    """An exemption with a {param} in it can never match: the gate compares strings, not templates."""
    templated = sorted(p for p in PUBLIC_PATHS if "{" in p)
    assert not templated, f"these exemptions are silently dead (exact-match gate): {templated}"


def test_the_whole_login_ceremony_is_reachable_without_a_session():
    """Whatever else changes, these three must stay public or nobody can ever log in again."""
    for path in ("/api/auth/status", "/api/auth/login/begin", "/api/auth/login/finish"):
        assert path in PUBLIC_PATHS, f"{path} must be public — it is part of logging in"


def test_credential_management_is_never_public():
    """The mirror image: /api/auth/* is not a blanket-public prefix.

    Listing and DELETING passkeys live under the same prefix as the ceremony, and an exemption added
    by prefix-thinking rather than path-thinking would let an anonymous caller remove the only key
    to the dashboard. Q124 gave the revoke button its UI; this keeps its gate.
    """
    creds = sorted(p for p in _declared_routes() if p.startswith("/api/auth/credentials"))
    assert creds, "expected the credential routes to exist (Q124)"
    for path in creds:
        assert path not in PUBLIC_PATHS, f"{path} must require a session"


def test_http_requests_also_accept_the_token_in_the_query_string(monkeypatch):
    """Q129: pinned as it IS today, with the trade-off written down rather than quietly changed.

    A browser cannot set headers on a WebSocket handshake, so ?token= exists for /ws — but
    `_presented` reads the query string for HTTP scopes too, which means a credential can ride in a
    URL: access logs, browser history and the Referer of anything that page loads. Nothing in the
    UI needs it (every fetch sends a header, AuthedImg fetches picture bytes rather than pointing an
    <img> at a route, there are no download anchors, and the audits put ?token= on the PAGE url
    where lib/endpoints absorbs it into storage), so narrowing this to websockets is defensible.

    It is NOT done here because it cannot be proven safe from inside the repo: a curl habit, a
    phone shortcut or a script on another machine may authenticate this way, and locking the owner
    out of a publicly tunnelled dashboard is worse than a credential in a log line he controls.
    /api/health's trusted-reader check reads ?token= on purpose (_session_presented), so the two
    rails must be decided together. This test exists so the day someone narrows it, they narrow it
    deliberately and see the reason and the second call site.
    """
    # The env matters and cost this test one run: with NOTHING configured the server is open to
    # loopback ONLY, so a remote client is refused whatever it presents — it never reaches the
    # credential branch at all. That is its own small reassurance about the LAN-dev posture.
    monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "true")
    token = auth.issue_token("cred1")
    scope = _http_scope("/api/fleet", client=("203.0.113.7", 1))
    scope["query_string"] = f"token={token}".encode()
    assert run_scope(scope) == "passed"
