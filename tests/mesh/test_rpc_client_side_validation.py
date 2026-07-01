"""Client-side validation contract for outbound RPC (``send`` / ``broadcast``).

The mesh documents that commands are validated on BOTH the client and the
server: a peer that issues a malformed command or addresses an illegal target
must be rejected locally, before anything touches the wire, rather than
publishing a bad sample and relying on the receiver to drop it. These tests lock
that contract for the two outbound RPC entry points:

* :meth:`Mesh.send` - single-target request/response.
* :meth:`Mesh.broadcast` - fan-out with a list-of-responses return.

The key security property is not merely the returned error shape but that a
rejected command is NEVER published: ``put`` (the transport chokepoint) must not
be called. All tests run against a mocked Zenoh session - no broker required.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh import Mesh
from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import session as mesh_session
from strands_robots.mesh.core import BROADCAST_RESPONDER


def _cmd_topics(puts: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Keys that carry an outbound RPC command (vs. presence/health/stream).

    ``Mesh.start`` publishes presence + periodic health on background topics;
    those are legitimate and unrelated to the RPC-validation contract. The
    security property under test is narrower: a rejected command must never
    reach its command topic - ``strands/<target>/cmd`` for send, or
    ``strands/broadcast`` for broadcast.
    """
    return [k for k, _ in puts if k.endswith("/cmd") or k == "strands/broadcast"]


class _FakeRobot:
    """Minimal duck-typed robot; RPC validation never reaches dispatch."""

    tool_name_str = "fakebot"

    def get_task_status(self) -> dict[str, Any]:
        return {"status": "idle"}


@pytest.fixture
def fake_session() -> Iterator[MagicMock]:
    sess = MagicMock()
    with (
        patch.object(mesh_session, "get_session", return_value=sess),
        patch.object(mesh_session, "current_session", return_value=sess),
        patch.object(mesh_core, "get_session", return_value=sess),
        patch.object(mesh_core, "current_session", return_value=sess),
        patch.object(mesh_core, "release_session"),
    ):
        yield sess


@pytest.fixture
def captured_puts(fake_session: MagicMock) -> Iterator[list[tuple[str, dict[str, Any]]]]:
    """Record every ``put(key, payload)`` the mesh performs on the wire."""
    seen: list[tuple[str, dict[str, Any]]] = []
    with patch.object(mesh_core, "put", side_effect=lambda k, d: seen.append((k, d))):
        yield seen


@pytest.fixture
def started_mesh(captured_puts: list[tuple[str, dict[str, Any]]]) -> Iterator[Mesh]:
    m = Mesh(_FakeRobot(), peer_id="peer-a", peer_type="robot")
    m.start()
    try:
        yield m
    finally:
        m.stop()


# --- send(): illegal target rejected before publish -----------------------


@pytest.mark.parametrize("target", ["", None, 123])
def test_send_rejects_non_string_or_empty_target(
    started_mesh: Mesh, captured_puts: list[tuple[str, dict[str, Any]]], target: Any
) -> None:
    out = started_mesh.send(target, {"action": "status"})
    assert out["status"] == "error"
    assert "non-empty string" in out["error"]
    assert _cmd_topics(captured_puts) == []  # no command reached the wire


@pytest.mark.parametrize("target", ["peer\x00b", BROADCAST_RESPONDER])
def test_send_rejects_nul_or_broadcast_sentinel_target(
    started_mesh: Mesh, captured_puts: list[tuple[str, dict[str, Any]]], target: str
) -> None:
    out = started_mesh.send(target, {"action": "status"})
    assert out["status"] == "error"
    assert "NUL" in out["error"] or "BROADCAST_RESPONDER" in out["error"]
    assert _cmd_topics(captured_puts) == []


# --- send(): malformed command rejected client-side -----------------------


def test_send_rejects_invalid_command_without_publishing(
    started_mesh: Mesh, captured_puts: list[tuple[str, dict[str, Any]]]
) -> None:
    out = started_mesh.send("peer-b", {"action": "warp"})  # not in ALLOWED_ACTIONS
    assert out["status"] == "error"
    assert out["error"].startswith("validation:")
    assert _cmd_topics(captured_puts) == []  # rejected before the command topic was published


# --- broadcast(): malformed command rejected client-side ------------------


def test_broadcast_rejects_invalid_command_and_returns_empty(
    started_mesh: Mesh, captured_puts: list[tuple[str, dict[str, Any]]]
) -> None:
    out = started_mesh.broadcast({"action": "warp"}, timeout=0.05)
    assert out == []  # list return type -> no structured error slot
    assert _cmd_topics(captured_puts) == []  # nothing published to strands/broadcast
