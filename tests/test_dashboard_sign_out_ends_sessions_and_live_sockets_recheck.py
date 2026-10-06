# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Signing out ends a session everywhere, and a live socket notices when its session ends.

Removing a passkey already refused its tokens on HTTP routes. Four doors stayed
open: ``/api/auth/logout`` only deleted the cookie, so a copy of the token taken
before kept working and renewing; ``/ws/agent`` and ``/ws/voice`` checked the
credential once at the handshake, so an open socket kept driving the agent (and
answering its consent prompts) after the passkey was gone; ``register/begin``
let any session enrol another passkey, which then outlived the removal of the
owner's; and the last passkey could not be removed at all.

Now every token carries its passkey's session epoch (``ep``), sign-out advances
it, every socket re-checks its caller while it lives, and adding a passkey or
removing the last one needs the bootstrap proof.
"""

from __future__ import annotations

import json
import sys
import time
import types
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import jwt
import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from strands_robots.dashboard import access, agent_console, auth, settings, voice  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402
from tests._dashboard_bootstrap import BOOTSTRAP, configure_bootstrap  # noqa: E402


def _bearer(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


@pytest.fixture()
def sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """A dashboard with two passkeys enrolled, the bootstrap proof configured, and its app."""
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.clear_overrides()
    settings.load(refresh=True)
    configure_bootstrap(monkeypatch)
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "jwt_secret": "s" * 64,
                "created": 1789500000,
                "credentials": [
                    {"id": cid, "public_key": "AA", "sign_count": 0, "name": cid, "created": 1789500000}
                    for cid in ("cred-phone", "cred-laptop")
                ],
            }
        ),
        encoding="utf-8",
    )
    auth._load()
    yield create_app()
    settings.clear_overrides()
    settings.load(refresh=True)


class TestSigningOutEndsTheSessionOnTheServer:
    def test_a_copy_taken_before_sign_out_is_refused_everywhere(self, sealed: Any) -> None:
        now = time.time()
        ttl = auth._token_ttl()
        # Past its half-life, so renewal would hand out a fresh token if the session still stood.
        token = auth.issue_token("cred-phone", "phone", iat0=int(now - 0.6 * ttl), exp=int(now + 0.4 * ttl))
        with TestClient(sealed) as client:
            assert client.get("/api/auth/credentials", headers=_bearer(token)).status_code == 200

            out = client.post("/api/auth/logout", headers=_bearer(token))
            assert out.status_code == 200 and out.json()["sessions_ended"] is True

            assert client.get("/api/auth/credentials", headers=_bearer(token)).status_code == 401
            assert client.post("/api/auth/renew", headers=_bearer(token)).status_code == 401
        assert auth.renew_if_due(token, now=now) is None
        with pytest.raises(HTTPException) as raised:
            auth.verify_token(token)
        assert "signed out" in str(raised.value.detail)

    def test_another_passkeys_session_survives_and_a_new_sign_in_works(self, sealed: Any) -> None:
        phone, laptop = auth.issue_token("cred-phone"), auth.issue_token("cred-laptop")
        with TestClient(sealed) as client:
            client.post("/api/auth/logout", headers=_bearer(phone))
            assert client.get("/api/auth/credentials", headers=_bearer(laptop)).status_code == 200
            assert (
                client.get("/api/auth/credentials", headers=_bearer(auth.issue_token("cred-phone"))).status_code == 200
            )

    @pytest.mark.parametrize(
        "claims_ep,record_epoch",
        [(None, None), ("0", None), (True, 1), (0, "1")],
        ids=["token-without-epoch", "epoch-not-an-int", "bool-is-not-an-epoch", "store-epoch-unreadable"],
    )
    def test_an_epoch_that_cannot_be_checked_is_refused(self, sealed: Any, claims_ep: Any, record_epoch: Any) -> None:
        store = auth._load()
        if record_epoch is not None:
            store["credentials"][0]["epoch"] = record_epoch
            auth._save(store)
        payload: dict[str, Any] = {"sub": "cred-phone", "exp": int(time.time()) + 600}
        if claims_ep is not None:
            payload["ep"] = claims_ep
        token = jwt.encode(payload, auth._jwt_secret(), algorithm="HS256")
        assert auth.session_is_valid(token) is False


class RecordingConsole:
    """A console that records what it was asked and answers with one done event."""

    def __init__(self) -> None:
        self.prompts: list[Any] = []

    async def run(self, prompt: Any) -> AsyncIterator[dict[str, Any]]:
        self.prompts.append(prompt)
        yield {"type": "done", "stop_reason": "end_turn"}

    resume = staticmethod(agent_console.Console.resume)


async def _echo(ws: Any, *, bridge: Any = None) -> None:
    """A voice session that answers every frame, so a socket left open reads as an answer, not a hang."""
    while True:
        await ws.receive_text()
        await ws.send_json({"type": "transcript", "text": "heard"})


def _revoke(how: str) -> None:
    if how == "removed":
        auth.delete_credential("cred-phone")
    else:
        auth.end_sessions("cred-phone")


class TestALiveSocketReChecksItsSession:
    @pytest.fixture(autouse=True)
    def _wire(self, sealed: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self.console = RecordingConsole()
        sealed.state.console_factory = lambda: self.console
        monkeypatch.setitem(sys.modules, "strands.bidi", types.ModuleType("strands.bidi"))
        monkeypatch.setattr(voice, "run_voice_session", _echo)

    @pytest.mark.parametrize("how", ["removed", "signed-out"])
    @pytest.mark.parametrize("path", ["/ws/agent", "/ws/voice"])
    def test_an_idle_socket_is_closed_4401_once_its_session_ends(
        self, sealed: Any, monkeypatch: pytest.MonkeyPatch, path: str, how: str
    ) -> None:
        monkeypatch.setattr(access, "SOCKET_RECHECK_SECONDS", 0.05, raising=False)
        token = auth.issue_token("cred-phone")
        with TestClient(sealed) as client, client.websocket_connect(path, headers=_bearer(token)) as ws:
            _revoke(how)
            time.sleep(0.3)  # several re-check intervals with nothing said
            ws.send_text(json.dumps({"type": "say", "text": "anyone there?"}))
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
        assert closed.value.code == 4401

    def test_a_consent_answer_after_the_session_ended_is_never_delivered(
        self, sealed: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even inside one re-check interval: the frame is checked before it is acted on."""
        monkeypatch.setattr(access, "SOCKET_RECHECK_SECONDS", 3600.0, raising=False)
        token = auth.issue_token("cred-phone")
        with TestClient(sealed) as client, client.websocket_connect("/ws/agent", headers=_bearer(token)) as ws:
            ws.send_json({"type": "say", "text": "hello"})
            assert ws.receive_json()["type"] == "done"
            auth.end_sessions("cred-phone")
            ws.send_json({"type": "resume", "id": "i1", "approve": True})
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
        assert closed.value.code == 4401
        assert self.console.prompts == ["hello"]

    def test_a_socket_whose_session_stands_keeps_working(self, sealed: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(access, "SOCKET_RECHECK_SECONDS", 0.05, raising=False)
        token = auth.issue_token("cred-phone")
        with TestClient(sealed) as client, client.websocket_connect("/ws/agent", headers=_bearer(token)) as ws:
            auth.end_sessions("cred-laptop")
            time.sleep(0.2)
            ws.send_json({"type": "say", "text": "still here"})
            assert ws.receive_json()["type"] == "done"


class TestOwnershipChangesNeedTheBootstrapProof:
    @pytest.fixture(autouse=True)
    def _ceremony(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(auth, "begin_registration", lambda request, label, bootstrap: {"challenge_id": "c"})

    @pytest.mark.parametrize("bootstrap,status", [("", 401), ("a-guess", 401), (BOOTSTRAP, 200)])
    def test_a_session_alone_cannot_enrol_another_passkey(self, sealed: Any, bootstrap: str, status: int) -> None:
        token = auth.issue_token("cred-phone")
        with TestClient(sealed) as client:
            out = client.post("/api/auth/register/begin", headers=_bearer(token), json={"bootstrap": bootstrap})
        assert out.status_code == status

    def test_the_last_passkey_is_removed_only_with_the_proof(self, sealed: Any) -> None:
        auth.delete_credential("cred-laptop")
        token = auth.issue_token("cred-phone")
        with TestClient(sealed) as client:
            kept = client.request("DELETE", "/api/auth/credentials/cred-phone", headers=_bearer(token))
            assert kept.status_code == 409
            gone = client.request(
                "DELETE", "/api/auth/credentials/cred-phone", headers=_bearer(token), json={"bootstrap": BOOTSTRAP}
            )
            assert gone.status_code == 200 and gone.json()["remaining"] == 0
            assert client.get("/api/auth/credentials", headers=_bearer(token)).status_code == 401
        assert auth.status()["setup_required"] is True
