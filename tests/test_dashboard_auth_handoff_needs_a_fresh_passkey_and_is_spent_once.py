"""A handoff to another device needs a fresh passkey tap, answers with a one-time code, and is spent once.

``POST /api/auth/handoff`` used to answer any caller presenting the session
cookie with a bearer JWT in the body. A script running in the page rides that
cookie, so it could mint the bearer, read it, and replay it from anywhere as
``Authorization: Bearer`` for the rest of its lifetime - and because the server
preferred a bearer over the cookie, a copy the page stored overrode the cookie
on every request.

Now minting takes an assertion over a ``/handoff/begin`` challenge on top of
the session, the answer is a code that no route accepts as a credential, the
code is spent by ``/handoff/redeem`` exactly once (the redeeming device gets the
session as its own ``HttpOnly`` cookie), and that session is honoured from the
cookie only. When a request carries both a cookie and a bearer, the cookie is
the one read.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import access, auth, settings  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402
from tests._dashboard_connection import connection  # noqa: E402
from tests._dashboard_passkeys import issue_enrolled  # noqa: E402

GUARDED = "/api/auth/credentials"
OWNER = "AQEBAQEBAQEBAQEBAQEBAQ"  # a credential id as the authenticator spells it (base64url)


@pytest.fixture()
def owner(tmp_path, monkeypatch) -> Iterator[TestClient]:
    """A browser signed in with a passkey: the session cookie and nothing else."""
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.clear_overrides()
    settings.load(refresh=True)
    client = TestClient(create_app(), base_url="http://localhost")
    client.cookies.set(access.COOKIE, issue_enrolled(OWNER, "Owner"))
    yield client
    settings.clear_overrides()
    settings.load(refresh=True)


def _tap(monkeypatch) -> None:
    """The authenticator said yes: the webauthn library's verifier succeeds (no authenticator in CI)."""
    monkeypatch.setattr(auth, "verify_authentication_response", lambda **kw: SimpleNamespace(new_sign_count=1))


def _mint(client: TestClient, monkeypatch) -> str:
    begun = client.post("/api/auth/handoff/begin", json={}).json()
    _tap(monkeypatch)
    answer = client.post("/api/auth/handoff", json={"challenge_id": begun["challenge_id"], "credential": {"id": OWNER}})
    assert answer.status_code == 200, answer.text
    assert set(answer.json()) == {"code", "exp", "expires_in"}, answer.json()
    return answer.json()["code"]


def _bare(code_or_token: str) -> TestClient:
    """Another device: no cookie, the value presented as a bearer."""
    client = TestClient(create_app(), base_url="http://localhost")
    client.headers["Authorization"] = f"Bearer {code_or_token}"
    return client


class TestTheCookieAloneCannotMintAHandoff:
    @pytest.mark.parametrize("body", [None, {}, {"challenge_id": "c1"}, {"credential": {"id": OWNER}}])
    def test_a_cookie_only_request_gets_no_bearer(self, owner, body) -> None:
        response = owner.post("/api/auth/handoff", json=body) if body is not None else owner.post("/api/auth/handoff")
        assert response.status_code == 400, response.text
        assert "token" not in response.json() and "code" not in response.json()

    def test_a_sign_in_challenge_does_not_buy_a_handoff(self, owner, monkeypatch) -> None:
        """Challenges are kind-bound: one fetched for login cannot be spent here, nor the reverse."""
        login = auth.begin_authentication(connection())
        _tap(monkeypatch)
        response = owner.post(
            "/api/auth/handoff", json={"challenge_id": login["challenge_id"], "credential": {"id": OWNER}}
        )
        assert response.status_code == 400, response.text
        handoff = owner.post("/api/auth/handoff/begin", json={}).json()
        refused = owner.post(
            "/api/auth/login/finish", json={"challenge_id": handoff["challenge_id"], "credential": {"id": OWNER}}
        )
        assert refused.status_code == 400, refused.text


class TestTheCodeIsNotACredentialAndIsSpentOnce:
    def test_the_code_opens_no_guarded_route(self, owner, monkeypatch) -> None:
        assert _bare(_mint(owner, monkeypatch)).get(GUARDED).status_code == 401

    def test_redeeming_sets_this_devices_cookie_once(self, owner, monkeypatch) -> None:
        code = _mint(owner, monkeypatch)
        device = TestClient(create_app(), base_url="http://localhost")
        first = device.post("/api/auth/handoff/redeem", json={"code": code})
        assert first.status_code == 200, first.text
        assert "token" not in first.json() and isinstance(first.json()["exp"], int)
        assert device.get(GUARDED).status_code == 200, "the redeeming device is signed in by its cookie"
        assert TestClient(create_app()).post("/api/auth/handoff/redeem", json={"code": code}).status_code == 401

    def test_the_handoff_session_is_refused_as_a_bearer(self, owner, monkeypatch) -> None:
        """Lifted off the device that redeemed it, the session opens nothing."""
        device = TestClient(create_app(), base_url="http://localhost")
        device.post("/api/auth/handoff/redeem", json={"code": _mint(owner, monkeypatch)})
        lifted = device.cookies.get(access.COOKIE)
        assert lifted and auth.verify_token(lifted)["via"] == "handoff"
        assert _bare(lifted).get(GUARDED).status_code == 401


# cookie, bearer -> status of a guarded route
PRESENTED = [
    ("valid", "junk", 200),
    ("junk", "valid", 401),
    ("", "valid", 200),
    ("", "", 401),
]


@pytest.mark.parametrize("cookie,bearer,status", PRESENTED)
def test_the_cookie_wins_over_a_bearer(owner, cookie, bearer, status) -> None:
    """A bearer is read only when there is no cookie: a script-held copy never shadows the browser's session."""
    valid = owner.cookies.get(access.COOKIE)
    client = TestClient(create_app(), base_url="http://localhost")
    if cookie:
        client.cookies.set(access.COOKIE, valid if cookie == "valid" else "junk")
    if bearer:
        client.headers["Authorization"] = f"Bearer {valid if bearer == 'valid' else 'junk'}"
    assert client.get(GUARDED).status_code == status
