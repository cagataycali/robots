"""POST /api/auth/handoff — a signed-in session copied into a SHORT-lived URL token.

Why it exists: the LAN hint sends the operator to http://<lan-ip>:8090, where WebAuthn
is impossible (no secure context), so the sign-in must ride along in the URL. The token
is minutes-long because URLs land in history, logs and screenshots.
"""

from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import auth
from strands_robots.dashboard import server as srv


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("STRANDS_DASH_AUTH_HANDOFF_TTL", raising=False)
    auth._cache_key = None
    auth._cache = {}
    yield


# --- the verdict: pure rule -------------------------------------------------------


def test_handoff_refuses_without_claims():
    assert auth.handoff_verdict(None, now=1000.0)["ok"] is False
    assert auth.handoff_verdict({}, now=1000.0)["ok"] is False
    assert "expiry" in auth.handoff_verdict({"sub": "c"}, now=1000.0)["reason"]


def test_handoff_refuses_an_expired_session():
    v = auth.handoff_verdict({"exp": 999}, now=1000.0)
    assert v["ok"] is False and "expired" in v["reason"]


def test_handoff_exp_is_short_and_never_outlives_the_session():
    # plenty of session left: now + ttl
    v = auth.handoff_verdict({"exp": 1000.0 + 86400}, now=1000.0, ttl=300)
    assert v == {"ok": True, "exp": 1300}
    # session ends sooner than the ttl: capped at the session's own exp
    v = auth.handoff_verdict({"exp": 1120.0}, now=1000.0, ttl=300)
    assert v == {"ok": True, "exp": 1120}


def test_handoff_ttl_env_override_and_garbage_fallback(monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_HANDOFF_TTL", "60")
    assert auth.handoff_ttl() == 60
    monkeypatch.setenv("STRANDS_DASH_AUTH_HANDOFF_TTL", "soon")
    assert auth.handoff_ttl() == 300


# --- the mint ----------------------------------------------------------------------


def test_issue_handoff_carries_identity_and_marks_origin():
    parent = auth.issue_token("cred1", name="phone")
    claims = auth.verify_token(parent)
    out = auth.issue_handoff(claims)
    assert 0 < out["expires_in"] <= auth.handoff_ttl()
    minted = auth.verify_token(out["token"])  # signed by the same store secret
    assert minted["sub"] == "cred1"
    assert minted["name"] == "phone"
    assert minted["via"] == "handoff"
    # iat0 is the parent's ORIGINAL sign-in: the renewal age cap survives the copy
    assert minted["iat0"] == claims["iat0"]
    assert minted["exp"] == out["exp"]


def test_issue_handoff_refuses_expired_claims():
    with pytest.raises(HTTPException) as e:
        auth.issue_handoff({"sub": "c", "exp": time.time() - 5})
    assert e.value.status_code == 401


def test_handoff_token_is_a_working_session():
    parent = auth.issue_token("cred1")
    out = auth.issue_handoff(auth.verify_token(parent))
    assert auth.session_is_valid(out["token"]) is True


# --- the route ---------------------------------------------------------------------


class _StubBridge:
    peer_id = "dash-test"
    peers: dict = {}

    def snapshot(self):
        return {"peers": {}}


@pytest.fixture()
def client():
    from starlette.testclient import TestClient

    app = srv.create_app(bridge=_StubBridge())
    return TestClient(app)  # no `with`: lifespan (mesh startup) must not run


def test_route_mints_from_a_valid_session(client, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "true")
    parent = auth.issue_token("cred1", name="phone")
    r = client.post("/api/auth/handoff", headers={"Authorization": f"Bearer {parent}"})
    assert r.status_code == 200
    body = r.json()
    assert body["expires_in"] <= auth.handoff_ttl()
    assert auth.verify_token(body["token"])["via"] == "handoff"


def test_route_mints_from_the_static_token(client, monkeypatch):
    real_get = srv.settings.get

    def fake_get(*a, **k):
        if a[:2] == ("security", "auth_token"):
            return "sekrit-static"
        return real_get(*a, **k)

    monkeypatch.setattr(srv.settings, "get", fake_get)
    r = client.post("/api/auth/handoff", headers={"Authorization": "Bearer sekrit-static"})
    assert r.status_code == 200
    body = r.json()
    minted = auth.verify_token(body["token"])
    assert minted["via"] == "handoff"
    assert minted["exp"] - time.time() <= auth.handoff_ttl() + 2


def test_route_says_why_when_auth_is_off(client):
    # open-loopback mode: the TestClient counts as local, nothing is configured
    r = client.post("/api/auth/handoff")
    assert r.status_code == 200
    body = r.json()
    assert body["token"] is None
    assert "not enabled" in body["why"]
