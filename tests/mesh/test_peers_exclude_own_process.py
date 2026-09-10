"""``Mesh.peers`` and ``robot_mesh peers`` must not count this process's own peers.

Measured on a lone ``Robot("so100", mesh=True)`` with no other process on
the network: ``robot.mesh.peers`` held the sim's own child peer
(``so100_sim-XXXX__so100``, age 0.0) and ``robot_mesh(action="peers")``
printed ``2 local, 2 remote`` -- zenoh delivers local presence to local
subscribers, and neither call subtracted what this process itself published.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import core, session


@pytest.fixture
def presence(monkeypatch):
    session.clear_peers()
    for pid in ("so100_sim-aaaa", "so100_sim-aaaa__so100", "so100_sim-bbbb__so100", "arm-remote"):
        session.update_peer(pid, "sim", "host", {})
    yield
    session.clear_peers()


def _mesh(peer_id: str) -> core.Mesh:
    m = core.Mesh.__new__(core.Mesh)
    m.peer_id = peer_id
    return m


def test_parent_peers_exclude_itself_and_its_own_children(presence):
    ids = {p["peer_id"] for p in _mesh("so100_sim-aaaa").peers}
    assert ids == {"so100_sim-bbbb__so100", "arm-remote"}


def test_another_process_child_is_still_a_peer(presence):
    # Only OWN children are hidden: a sibling process's child keeps its row.
    ids = {p["peer_id"] for p in _mesh("so100_sim-bbbb").peers}
    assert "so100_sim-aaaa__so100" in ids and "so100_sim-aaaa" in ids


def test_robot_mesh_peers_counts_only_other_processes_as_remote(presence, monkeypatch):
    from unittest.mock import MagicMock

    import strands_robots.mesh as mesh_pkg
    from strands_robots.tools.robot_mesh import robot_mesh

    locals_ = {"so100_sim-aaaa": _mesh("so100_sim-aaaa"), "so100_sim-aaaa__so100": _mesh("so100_sim-aaaa__so100")}
    for m in locals_.values():
        m.peer_type = "sim"
    monkeypatch.setattr(mesh_pkg, "get_local_robots", lambda: locals_)
    text = robot_mesh(action="peers", tool_context=MagicMock())["content"][0]["text"]
    assert text.startswith("[mesh] 2 local, 2 remote"), text
    discovered = text.split("Discovered peers:")[1]
    assert "so100_sim-aaaa" not in discovered
    assert "arm-remote" in discovered and "so100_sim-bbbb__so100" in discovered
