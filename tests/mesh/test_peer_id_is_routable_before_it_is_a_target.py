"""Regression tests: a peer id becomes a command key expression only if it is routable.

``Mesh.send`` interpolated ``target`` straight into ``strands/{target}/cmd``
after checking only for emptiness, a NUL byte and the broadcast sentinel, and
both presence paths (``Mesh._on_presence`` and the dashboard's
``MeshBridge._on_presence``) stored whatever ``robot_id`` a peer announced.
A peer announcing itself as ``*`` or ``**`` therefore became a fleet-wide
target for a command the operator aimed at one robot.

The inbound path already refused such identifiers with
``validate_mesh_identifier``; now the outbound path and both presence
registries apply the same rule, before publish and before the registry.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.mesh import session as mesh_session
from strands_robots.mesh.core import Mesh

UNROUTABLE = ["*", "**", "$*", "arm/extra", "arm b", "arm\n", "", "a" * 129, "arm?"]


class _Robot:
    tool_name_str = "arm"


def _presence(peer: str) -> Any:
    payload = {"robot_id": peer, "robot_type": "robot", "hostname": "h", "timestamp": time.time()}
    return SimpleNamespace(
        payload=SimpleNamespace(to_bytes=lambda: json.dumps(payload).encode()),
        source_info=None,
        key_expr=f"strands/{peer}/presence",
    )


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    with mesh_session._PEERS_LOCK:
        mesh_session._PEERS.clear()
    yield
    with mesh_session._PEERS_LOCK:
        mesh_session._PEERS.clear()


@pytest.fixture
def mesh(monkeypatch: pytest.MonkeyPatch) -> Mesh:
    m = Mesh(_Robot(), peer_id="op")
    m._running = True
    published: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(m, "publish", lambda key, payload, **kw: published.append((key, payload)))
    m._published = published  # type: ignore[attr-defined]
    return m


class TestSendRefusesAnUnroutableTarget:
    @pytest.mark.parametrize("target", [t for t in UNROUTABLE if t])
    def test_nothing_is_published(self, mesh: Mesh, target: str) -> None:
        out = mesh.send(target, {"action": "status"}, timeout=0.01)

        assert out["status"] == "error", out
        assert "target" in out["error"]
        assert mesh._published == []  # type: ignore[attr-defined]
        with mesh._rpc_lock:
            assert mesh._pending == {}

    def test_a_routable_target_is_published_to_its_own_key(self, mesh: Mesh) -> None:
        mesh.send("arm-b.1", {"action": "status"}, timeout=0.01)

        assert [k for k, _ in mesh._published] == ["strands/arm-b.1/cmd"]  # type: ignore[attr-defined]

    def test_ping_refuses_the_same_way(self, mesh: Mesh) -> None:
        out = mesh.ping("**", timeout=0.01)

        assert out["status"] == "error"
        assert mesh._published == []  # type: ignore[attr-defined]


class TestPresenceRefusesAnUnroutablePeerId:
    @pytest.mark.parametrize("peer", [p for p in UNROUTABLE if p])
    def test_core_registry_never_learns_it(self, mesh: Mesh, peer: str) -> None:
        mesh._on_presence(_presence(peer))

        assert mesh_session.get_peer(peer) is None
        assert mesh.peers == []

    def test_core_registry_learns_a_routable_one(self, mesh: Mesh) -> None:
        mesh._on_presence(_presence("arm-b"))

        assert [p["peer_id"] for p in mesh.peers] == ["arm-b"]

    @pytest.mark.parametrize("peer", [p for p in UNROUTABLE if p])
    def test_dashboard_bridge_never_learns_it(self, peer: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.dashboard.mesh_bridge import MeshBridge

        bridge = MeshBridge.__new__(MeshBridge)
        bridge.peers = {}
        bridge.peer_id = "dashboard"
        bridge._peers_lock = threading.Lock()
        emitted: list[dict[str, Any]] = []
        monkeypatch.setattr(bridge, "_emit", emitted.append)

        bridge._on_presence(_presence(peer))
        bridge._on_state(_presence(peer))

        assert bridge.peers == {}
        assert emitted == []


def test_the_outbound_rule_is_the_inbound_rule() -> None:
    """One charset, so a peer the receive side accepts is one the send side can address."""
    from strands_robots.mesh import core, security

    assert core._routable_target_error("arm-b") is None
    with pytest.raises(security.ValidationError):
        security.validate_mesh_identifier("**", "x")
    assert core._routable_target_error("**") is not None
