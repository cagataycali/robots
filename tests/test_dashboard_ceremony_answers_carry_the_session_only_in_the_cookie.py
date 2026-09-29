"""A finished passkey ceremony hands the browser its session once, in the ``HttpOnly`` cookie.

``/api/auth/register/finish`` and ``/api/auth/login/finish`` set the session as
the ``strands_dash`` cookie with ``HttpOnly`` and ``SameSite=Strict``, which is
exactly what keeps a page script from reading it. They then returned the same
token in the JSON body, and the page saved that copy in ``localStorage`` and
sent it as a bearer on every request, so any script that ever ran in the
origin held a full session for the hardware routes and the cookie protected
nothing (finding f015, Medium).

The ceremonies are browser-only (``navigator.credentials``), and a same-origin
page rides the cookie without holding anything, so the body now carries the
answer minus the token. The expiry is kept in the body, a number the page may
know without holding the secret, so it can still warn before the session lapses.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import auth, settings  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402
from tests._dashboard_passkeys import issue_enrolled  # noqa: E402

CEREMONIES = [("/api/auth/register/finish", "finish_registration"), ("/api/auth/login/finish", "finish_authentication")]


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.clear_overrides()
    settings.load(refresh=True)
    yield tmp_path
    settings.clear_overrides()
    settings.load(refresh=True)


def _cookie_value(response) -> str:
    raw = response.headers.get("set-cookie", "")
    first = raw.split(";", 1)[0]
    assert first.startswith("strands_dash="), raw
    return first.removeprefix("strands_dash=")


@pytest.mark.parametrize("path,verb", CEREMONIES, ids=["register", "login"])
class TestTheBodyDoesNotRepeatTheCookie:
    def test_the_token_is_in_the_cookie_and_not_in_the_body(self, isolated, monkeypatch, path, verb):
        minted = auth.issue_token("owner", "Owner")
        monkeypatch.setattr(
            auth, verb, lambda request, cid, cred: {"ok": True, "token": minted, "credential_id": "cred-1"}
        )
        client = TestClient(create_app())

        response = client.post(path, json={"challenge_id": "c1", "credential": {"id": "cred-touchid"}})

        assert response.status_code == 200
        assert _cookie_value(response) == minted
        body = response.json()
        assert "token" not in body, "the session the HttpOnly cookie protects is repeated where a script can read it"
        assert minted not in response.text
        assert body["ok"] is True and body["credential_id"] == "cred-1"

    def test_the_body_says_when_the_session_lapses(self, isolated, monkeypatch, path, verb):
        # The expiry is read back through ``auth.verify_token``, which honours a
        # session only while its passkey is enrolled, so the minted subject is one.
        minted = issue_enrolled("owner", "Owner")
        monkeypatch.setattr(auth, verb, lambda request, cid, cred: {"ok": True, "token": minted})
        client = TestClient(create_app())

        body = client.post(path, json={"challenge_id": "c1", "credential": {"id": "cred-touchid"}}).json()

        assert body["exp"] == auth.verify_token(minted)["exp"]

    def test_a_token_the_page_could_not_decode_is_still_not_repeated(self, isolated, monkeypatch, path, verb):
        monkeypatch.setattr(auth, verb, lambda request, cid, cred: {"token": "minted-session", "user": "owner"})
        client = TestClient(create_app())

        response = client.post(path, json={"challenge_id": "c1", "credential": {"id": "cred-touchid"}})

        assert _cookie_value(response) == "minted-session"
        assert response.json() == {"user": "owner"}
