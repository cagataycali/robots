"""The passkey ROUTES: what :mod:`strands_robots.dashboard.routes_auth` adds to a verdict.

Who may enrol or sign in is never decided here - those verdicts belong to
:mod:`strands_robots.dashboard.auth`, and are pinned by
``test_dashboard_auth_ceremony_finish``. What this file grades is the HTTP the
routes shape around a verdict, which is the part a browser depends on and the
auth module cannot see:

* a finished ceremony hands the browser a cookie the page's own scripts cannot
  read, that no other origin can ride, and whose ``Secure`` flag follows the
  connection rather than a setting - otherwise ``http://localhost`` could never
  sign in;
* the enrolled keys are managed by a passkey session only, so neither the
  static token nor the fresh-install loopback posture may remove a key or mint
  a handoff to another device;
* each route hands the auth module what the caller actually presented.

The auth module is monkeypatched throughout: no authenticator exists in CI, and
substituting the verdict is what isolates the shaping from the decision.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import access, auth, settings  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """A fresh install: empty passkey store, default settings, no env override."""
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.clear_overrides()
    settings.load(refresh=True)
    yield tmp_path
    settings.clear_overrides()
    settings.load(refresh=True)


@pytest.fixture()
def client(isolated):
    """A loopback caller on a fresh install: admitted, but not as a passkey."""
    return TestClient(create_app())


@pytest.fixture()
def passkey(isolated):
    """A ``Bearer`` header carrying a session the auth module accepts."""
    return {"authorization": f"Bearer {auth.issue_token('owner', 'Owner')}"}


def session_cookie(response) -> tuple[str, set[str]]:
    """The session cookie's value and its attributes, as a browser reads them.

    Split into tokens rather than matched as a substring: ``Path=/api`` contains
    the text ``Path=/`` while scoping the session away from the pages that need
    it, so an attribute is only present if it is present exactly.
    """
    line = next((c for c in response.headers.get_list("set-cookie") if c.startswith(f"{access.COOKIE}=")), "")
    if not line:
        return "", set()
    pair, _, attributes = line.partition(";")
    return pair.split("=", 1)[1], {a.strip() for a in attributes.split(";") if a.strip()}


#: The two ceremonies that mint a session, and the auth verb each one asks.
CEREMONIES = [("/api/auth/register/finish", "finish_registration"), ("/api/auth/login/finish", "finish_authentication")]

#: Managing an enrolled key: the route, its method, and the verb it delegates to.
KEY_MANAGEMENT = [
    ("delete", "/api/auth/credentials/cred-touchid", "delete_credential"),
    ("post", "/api/auth/handoff", "issue_handoff"),
]


class TestAFinishedCeremonyHandsTheBrowserASessionCookie:
    @pytest.mark.parametrize("scheme,secure", [("http", False), ("https", True)], ids=["http", "https"])
    @pytest.mark.parametrize("path,verb", CEREMONIES, ids=["register", "login"])
    def test_the_cookie_is_httponly_strict_and_secure_only_over_tls(
        self, isolated, monkeypatch, path, verb, scheme, secure
    ):
        """``Secure`` follows the connection: a cookie pinned to TLS could never
        be set on the ``http://localhost`` the operator's own browser loads."""
        monkeypatch.setattr(auth, verb, lambda request, cid, cred: {"token": "minted-session", "user": "owner"})
        client = TestClient(create_app(), base_url=f"{scheme}://testserver")

        response = client.post(path, json={"challenge_id": "c1", "credential": {"id": "cred-touchid"}})

        assert response.status_code == 200
        # The auth module's answer reaches the caller unedited.
        assert response.json() == {"token": "minted-session", "user": "owner"}
        value, attributes = session_cookie(response)
        assert value == "minted-session"
        assert "HttpOnly" in attributes, "a page script must not be able to read the session"
        assert "SameSite=strict" in attributes, "no other origin may ride the session"
        assert "Path=/" in attributes, "the session is carried on every request, not only the one that set it"
        assert ("Secure" in attributes) is secure

    @pytest.mark.parametrize("path,verb", CEREMONIES, ids=["register", "login"])
    def test_the_ceremony_is_verified_by_what_the_caller_sent(self, client, monkeypatch, path, verb):
        seen = {}

        def record(request, challenge_id, credential):
            seen.update(challenge_id=challenge_id, credential=credential)
            return {"token": "minted-session"}

        monkeypatch.setattr(auth, verb, record)
        client.post(path, json={"challenge_id": "c-42", "credential": {"id": "cred-yubikey"}})

        assert seen == {"challenge_id": "c-42", "credential": {"id": "cred-yubikey"}}


class TestOnlyAPasskeySessionManagesTheEnrolledKeys:
    """A loopback caller on a fresh install is admitted - ``via='loopback'`` - and
    so is a static token. Neither is the owner's authenticator, so neither may
    remove a key or mint a session for another device.
    """

    @pytest.mark.parametrize("method,path,verb", KEY_MANAGEMENT, ids=["delete-credential", "handoff"])
    def test_a_session_that_is_not_a_passkey_is_refused(self, client, monkeypatch, method, path, verb):
        monkeypatch.setattr(auth, verb, lambda *a, **k: pytest.fail(f"{verb} was reached without a passkey"))

        response = getattr(client, method)(path)

        assert response.status_code == 403
        assert "passkey" in response.json()["error"]

    @pytest.mark.parametrize("method,path,verb", KEY_MANAGEMENT, ids=["delete-credential", "handoff"])
    def test_a_passkey_session_reaches_the_auth_module(self, client, monkeypatch, passkey, method, path, verb):
        monkeypatch.setattr(auth, verb, lambda *a, **k: {"done": verb})

        response = getattr(client, method)(path, headers=passkey)

        assert response.status_code == 200
        assert response.json() == {"done": verb}

    def test_the_enrolled_keys_are_the_auth_modules_list(self, client, monkeypatch):
        monkeypatch.setattr(auth, "list_credentials", lambda: [{"id": "cred-touchid", "name": "macbook"}])

        assert client.get("/api/auth/credentials").json() == {
            "credentials": [{"id": "cred-touchid", "name": "macbook"}]
        }


class TestRenewal:
    """The route decides nothing about WHEN a session renews - that cap is the
    auth module's. It decides only what it renews and where the result goes.
    """

    def test_a_session_that_is_not_a_passkey_is_never_renewed(self, client, monkeypatch):
        monkeypatch.setattr(auth, "renew_if_due", lambda *a, **k: pytest.fail("renewal was attempted"))

        response = client.post("/api/auth/renew")

        assert response.status_code == 200
        assert response.json() == {"renewed": False}
        assert session_cookie(response) == ("", set()), "nothing was renewed, so nothing is re-set"

    @pytest.mark.parametrize("fresh,renewed", [("longer-lived-session", True), (None, False)], ids=["due", "fresh"])
    def test_a_renewal_replaces_the_cookie_and_is_asked_about_the_presented_token(
        self, client, monkeypatch, passkey, fresh, renewed
    ):
        seen = {}

        def record(token):
            seen["token"] = token
            return fresh

        monkeypatch.setattr(auth, "renew_if_due", record)

        response = client.post("/api/auth/renew", headers=passkey)

        assert response.json() == {"renewed": renewed}
        assert seen["token"] == passkey["authorization"].removeprefix("Bearer "), (
            "the token to renew is the one presented, not one read back from the store"
        )
        assert (session_cookie(response)[0] == fresh) is renewed


class TestABodyIsReadOnlyAsJson:
    """A cross-site ``fetch`` may send ``text/plain`` without a preflight, so a
    write route that parses it turns any page on the web into its caller. The
    415 and the non-object 400 are pinned by ``test_dashboard_server_core``;
    what is left is a body that claims JSON and is not, and no body at all.
    """

    def test_a_body_that_claims_json_and_is_not_is_400(self, client):
        response = client.post(
            "/api/auth/register/begin", content=b"{not json", headers={"content-type": "application/json"}
        )

        assert response.status_code == 400
        assert response.json()["error"] == "body is not JSON"

    @pytest.mark.parametrize(
        "body,label,bootstrap",
        [
            (None, "passkey", ""),
            ({}, "passkey", ""),
            ({"label": "", "bootstrap": "one-time-proof"}, "passkey", "one-time-proof"),
            ({"label": "x" * 200}, "x" * 64, ""),
        ],
        ids=["no-body", "empty-object", "blank-label", "overlong-label"],
    )
    def test_an_enrolment_names_the_key_within_a_bounded_label(self, client, monkeypatch, body, label, bootstrap):
        """No body is a body: an enrolment may carry nothing and still be shaped
        into the auth module's call, with a label the store cannot be grown by."""
        seen = {}

        def record(request, label, bootstrap):
            seen.update(label=label, bootstrap=bootstrap)
            return {"options": {}}

        monkeypatch.setattr(auth, "begin_registration", record)
        if body is None:
            client.post("/api/auth/register/begin")  # no body at all, not even "{}"
        else:
            client.post("/api/auth/register/begin", json=body)

        assert seen == {"label": label, "bootstrap": bootstrap}

    def test_a_sign_in_ceremony_is_the_auth_modules_to_start(self, client, monkeypatch):
        monkeypatch.setattr(auth, "begin_authentication", lambda request: {"challenge_id": "c-7", "allow": []})

        assert client.post("/api/auth/login/begin").json() == {"challenge_id": "c-7", "allow": []}
