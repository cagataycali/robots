# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fresh-install open posture admits a caller on the bootstrap proof, never on topology alone.

Finding f002 (CWE-290 / CWE-348). The first-enrollment fix (F-007) made the
ownership-granting ceremony require the bootstrap token: the configured
``STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN`` or the ``0600`` file the server writes
beside the credential store, because "at this machine" is not a property a
request can assert. ``access.open_posture`` kept admitting the WHOLE API on the
signal that fix declared insufficient: a loopback socket peer, no forwarding
header, a loopback ``Host`` and no ``Origin``. A same-host L4 forwarder
(``socat``, ``ssh -L``, ``docker -p``, a DNAT rule) and any second local
account send exactly that, so before the owner enrolled, a stranger could spawn
a real arm, start a task with ``"confirmed": true`` and grant standing agent
motion, while enrolling the owner passkey was the one thing they could not do.

Now the open posture needs the same proof the first enrollment does, presented
as the bearer token. Reading the ``0600`` file is still the local act a
forwarded or second-account caller cannot perform. The topology checks stay,
because they close a different door (the operator's own browser running someone
else's page), and the posture still closes on its own once a passkey exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import access, auth, settings
from tests._dashboard_connection import STRANGER, connection


@pytest.fixture()
def fresh_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No passkey, no static token, no env override: the posture the finding is about."""
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", raising=False)
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENROLL_TOKEN_FILE", raising=False)
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.clear_overrides()
    settings.load(refresh=True)
    yield tmp_path
    settings.clear_overrides()
    settings.load(refresh=True)


def _local_token(tmp_path: Path) -> str:
    """The token the server minted beside the store, read the way the owner reads it."""
    return (tmp_path / "enroll_token").read_text(encoding="utf-8").strip()


def _bearer(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


class TestTheL4ForwarderCaseIsRefused:
    """Loopback peer, no forwarding header, loopback Host, no Origin, no token: the request the old rule admitted."""

    @pytest.mark.parametrize("peer", ["127.0.0.1", "127.0.0.53", "::1", "localhost"])
    def test_a_bare_loopback_peer_is_not_admitted(self, fresh_install: Path, peer: str) -> None:
        request = connection(peer=peer, path="/api/whoami")
        assert access.came_through_a_proxy(request) is False, "this must be the header-free case"
        assert access.open_posture(request) is False
        with pytest.raises(HTTPException) as raised:
            access.caller(request)
        assert raised.value.status_code == 401

    def test_a_guessed_token_is_refused(self, fresh_install: Path) -> None:
        auth._first_enrollment_proof()  # the file exists, so this is a mismatch and not an absence
        request = connection(path="/api/whoami", **_bearer("not-the-token"))
        assert access.open_posture(request) is False
        with pytest.raises(HTTPException) as raised:
            access.caller(request)
        assert raised.value.status_code == 401

    def test_the_file_token_is_not_honoured_once_an_env_token_is_configured(
        self, fresh_install: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first enrollment has exactly one expectation; the open posture must have the same one."""
        auth._first_enrollment_proof()
        from_file = _local_token(fresh_install)
        monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "configured-by-the-operator")
        assert access.open_posture(connection(path="/api/whoami", **_bearer(from_file))) is False
        assert access.open_posture(connection(path="/api/whoami", **_bearer("configured-by-the-operator"))) is True


class TestTheProofAdmitsOnlyFromThisMachinesOwnBrowser:
    """The token is necessary; the topology checks are still necessary too."""

    @pytest.mark.parametrize("peer", ["127.0.0.1", "::1"])
    def test_the_local_token_admits_a_local_peer(self, fresh_install: Path, peer: str) -> None:
        auth._first_enrollment_proof()
        request = connection(peer=peer, path="/api/whoami", **_bearer(_local_token(fresh_install)))
        assert access.caller(request) == {"via": "loopback"}

    def test_the_configured_token_admits_a_local_peer(self, fresh_install: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "configured-by-the-operator")
        request = connection(path="/api/whoami", **_bearer("configured-by-the-operator"))
        assert access.caller(request) == {"via": "loopback"}

    def test_the_token_from_a_stranger_is_refused(self, fresh_install: Path) -> None:
        """A leaked token is still not presence at the machine."""
        auth._first_enrollment_proof()
        request = connection(peer=STRANGER, path="/api/whoami", **_bearer(_local_token(fresh_install)))
        assert access.open_posture(request) is False

    @pytest.mark.parametrize("header", ["x-forwarded-for", "x-real-ip", "forwarded"])
    def test_the_token_through_a_proxy_is_refused(self, fresh_install: Path, header: str) -> None:
        auth._first_enrollment_proof()
        request = connection(path="/api/whoami", **_bearer(_local_token(fresh_install)), **{header: "203.0.113.9"})
        assert access.open_posture(request) is False

    @pytest.mark.parametrize("origin", ["http://evil.example", "null"])
    def test_the_token_from_another_pages_origin_is_refused(self, fresh_install: Path, origin: str) -> None:
        auth._first_enrollment_proof()
        request = connection(path="/api/whoami", origin=origin, **_bearer(_local_token(fresh_install)))
        assert access.open_posture(request) is False

    def test_a_rebound_host_with_the_token_is_refused(self, fresh_install: Path) -> None:
        auth._first_enrollment_proof()
        request = connection(host="evil.example:8090", path="/api/whoami", **_bearer(_local_token(fresh_install)))
        assert access.open_posture(request) is False


class TestThePostureClosesOnItsOwn:
    """Once auth is on, the bootstrap token admits nobody: it guarded the window, not the dashboard."""

    def test_an_env_override_closes_it(self, fresh_install: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "configured-by-the-operator")
        monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "1")
        request = connection(path="/api/whoami", **_bearer("configured-by-the-operator"))
        assert access.open_posture(request) is False

    def test_a_static_token_closes_it(self, fresh_install: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "configured-by-the-operator")
        settings.override("security", "auth_token", "the-static-token")
        request = connection(path="/api/whoami", **_bearer("configured-by-the-operator"))
        assert access.open_posture(request) is False
        # The static token itself still works, through its own door.
        assert access.caller(connection(path="/api/whoami", **_bearer("the-static-token"))) == {"via": "token"}

    def test_a_token_less_caller_never_mints_a_file_it_could_not_read_anyway(self, fresh_install: Path) -> None:
        """Deciding a header-free request must not touch the disk: the answer is no before the file matters."""
        request = connection(path="/api/whoami")
        assert access.open_posture(request) is False
        assert not (fresh_install / "enroll_token").exists()


class TestTheRoutesFollow:
    """Every route takes ``require_session``; the server-level observable is 401 without the proof."""

    @pytest.fixture()
    def client(self, fresh_install: Path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from strands_robots.dashboard.server import create_app

        return TestClient(create_app())

    def test_whoami_needs_the_proof(self, client, fresh_install: Path) -> None:
        assert client.get("/api/whoami").status_code == 401
        auth._first_enrollment_proof()
        response = client.get("/api/whoami", headers=_bearer(_local_token(fresh_install)))
        assert response.status_code == 200
        assert response.json()["via"] == "loopback"

    def test_a_consent_grant_needs_the_proof(self, client, fresh_install: Path) -> None:
        response = client.post("/api/consent", json={"kind": "agent_physical_motion"})
        assert response.status_code == 401

    def test_the_login_screen_still_gets_its_fields(self, client) -> None:
        body = client.get("/api/auth/status").json()
        assert body["setup_required"] is True
        assert body["bootstrap_required"] is True
        assert body["open_posture"] is False
        assert client.get("/api/health").json()["ok"] is True
