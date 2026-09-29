# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A WebSocket handshake is admitted by its Origin first, whatever credential it carries.

Finding f022 (CWE-1385, cross-site WebSocket hijacking). ``access.caller`` read the
``Origin`` header in one branch only, the fresh-install open posture. A handshake
carrying a valid passkey session cookie was admitted at the first branch with the
header never read. WebSockets are exempt from CORS and the cross-origin write
middleware never sees a ``websocket`` scope, and ``SameSite=Strict`` is scoped to
the site, which ignores the port, so a page on ``http://localhost:3000`` opened
``ws://localhost:8090/ws/agent`` with the operator's cookie attached and drove
the agent, the fleet feed and the camera tiles from another tab.

Now every socket goes through ``access.admit_socket``: a foreign ``Origin`` is a
handshake rejection before any credential is read, and a handshake with no
``Origin`` at all (browsers always send one) is admitted only when it presents an
explicit bearer, never on the cookie a browser would have attached. The
credential check follows, unchanged.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from strands_robots.dashboard import access, auth, settings  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402
from tests._dashboard_connection import websocket  # noqa: E402

HOST = "localhost:8090"
SELF = "http://localhost:8090"
SIBLING_PORT = "http://localhost:3000"
SOCKETS = ["/ws/mesh", "/ws/camera/peer/cam", "/ws/agent", "/ws/voice", "/ws/telemetry/nosuch"]


@pytest.fixture()
def sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    """An enrolled dashboard and the cookie its owner's browser holds."""
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.clear_overrides()
    settings.load(refresh=True)
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "jwt_secret": "s" * 64,
                "created": 1789500000,
                "credentials": [
                    {"id": "cred-owner", "public_key": "AA", "sign_count": 0, "name": "owner", "created": 1789500000}
                ],
            }
        ),
        encoding="utf-8",
    )
    auth._load()
    token = auth.issue_token("cred-owner", name="owner")
    yield {"cookie": f"{access.COOKIE}={token}", "token": token}
    settings.clear_overrides()
    settings.load(refresh=True)


def handshake(app: Any, path: str, headers: dict[str, str]) -> list[dict[str, Any]]:
    """Every ASGI message *app* sends while deciding a socket handshake at *path*."""
    messages: list[dict[str, Any]] = []
    scope: dict[str, Any] = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "ws",
        "server": ("127.0.0.1", 8090),
        "client": ("127.0.0.1", 54321),
        "root_path": "",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "subprotocols": [],
        "state": {},
    }

    inbound = iter([{"type": "websocket.connect"}])

    async def receive() -> dict[str, Any]:
        # The handshake, then the client leaves: an accepted stream ends instead of parking.
        return next(inbound, {"type": "websocket.disconnect", "code": 1000})

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    return messages


def rejected_at_the_handshake(messages: list[dict[str, Any]]) -> bool:
    return [m["type"] for m in messages] == ["websocket.close"]


class TestTheDecisionIsMadeOnTheConnection:
    """``access.socket_origin_is_self`` on the real ``WebSocket`` object."""

    def test_a_sibling_port_is_another_origin(self) -> None:
        assert access.socket_origin_is_self(websocket(host=HOST, origin=SIBLING_PORT)) is False

    def test_this_host_is_self(self) -> None:
        assert access.socket_origin_is_self(websocket(host=HOST, origin=SELF)) is True

    def test_no_origin_and_a_cookie_is_not_a_browser_page(self) -> None:
        assert access.socket_origin_is_self(websocket(host=HOST, cookie=f"{access.COOKIE}=x")) is False

    def test_no_origin_with_an_explicit_bearer_is_a_script_and_passes_here(self) -> None:
        assert access.socket_origin_is_self(websocket(host=HOST, authorization="Bearer x")) is True

    def test_null_is_another_origin(self) -> None:
        assert access.socket_origin_is_self(websocket(host=HOST, origin="null")) is False


class TestACredentialedCrossOriginHandshakeIsRejected:
    """The regression: the owner's cookie, a page from elsewhere. Nothing is accepted."""

    @pytest.mark.parametrize("path", SOCKETS)
    def test_a_sibling_port_with_the_cookie_is_rejected_before_accept(self, sealed, path: str) -> None:
        app = create_app()
        messages = handshake(app, path, {"host": HOST, "origin": SIBLING_PORT, "cookie": sealed["cookie"]})
        assert rejected_at_the_handshake(messages), messages

    @pytest.mark.parametrize("path", SOCKETS)
    def test_a_foreign_site_with_the_cookie_is_rejected_before_accept(self, sealed, path: str) -> None:
        app = create_app()
        messages = handshake(app, path, {"host": HOST, "origin": "http://evil.example", "cookie": sealed["cookie"]})
        assert rejected_at_the_handshake(messages), messages

    @pytest.mark.parametrize("path", SOCKETS)
    def test_the_cookie_with_no_origin_is_rejected_before_accept(self, sealed, path: str) -> None:
        app = create_app()
        messages = handshake(app, path, {"host": HOST, "cookie": sealed["cookie"]})
        assert rejected_at_the_handshake(messages), messages


class TestTheOwnersOwnPageAndScriptsStillGetIn:
    def test_the_owners_page_is_accepted(self, sealed) -> None:
        app = create_app()
        messages = handshake(app, "/ws/mesh", {"host": HOST, "origin": SELF, "cookie": sealed["cookie"]})
        assert messages[0]["type"] == "websocket.accept", messages

    def test_a_script_with_the_bearer_and_no_origin_is_accepted(self, sealed) -> None:
        app = create_app()
        messages = handshake(app, "/ws/mesh", {"host": HOST, "authorization": f"Bearer {sealed['token']}"})
        assert messages[0]["type"] == "websocket.accept", messages

    def test_the_owners_page_without_a_session_still_reads_4401(self, sealed) -> None:
        """The page shows the login screen on 4401; that path is unchanged for a same-origin page."""
        app = create_app()
        messages = handshake(app, "/ws/agent", {"host": HOST, "origin": SELF})
        assert [m["type"] for m in messages] == ["websocket.accept", "websocket.close"], messages
        assert messages[-1]["code"] == 4401
