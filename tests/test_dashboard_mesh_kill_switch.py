"""STRANDS_MESH=false must stop the DASHBOARD from joining the fleet too.

Q48. The switch is documented as a hard kill switch, and MeshBridge.start() called
get_session() without ever asking - the same defect Q32 fixed one caller down, in
robot_mesh._gateway_mesh(), while leaving the biggest session-opener in the tree
unasked. Found by rehearsing cagatay's restart with the switch set: it surfaced as a
startup crash (mTLS is the default auth mode, so building a config for a session
nobody asked for raises), which is why nobody had noticed the quieter half - with
certs present, the dashboard would simply have joined the live fleet.
"""
from __future__ import annotations

import pytest

from strands_robots.dashboard.mesh_bridge import MeshBridge


class _Boom:
    """get_session() must not even be REACHED once the switch is set."""

    def __call__(self, *a, **k):  # pragma: no cover - the point is that it is not called
        raise AssertionError("get_session() called with STRANDS_MESH=false")


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", " false "])
def test_kill_switch_stops_the_bridge_before_it_opens_a_session(monkeypatch, value):
    monkeypatch.setenv("STRANDS_MESH", value)
    monkeypatch.setattr("strands_robots.mesh.session.get_session", _Boom())
    bridge = MeshBridge()
    assert bridge.start(loop=None) is False
    # "offline" must be a real state, not a half-started bridge.
    assert bridge._running is False
    assert bridge._session is None


def test_switch_off_means_business_as_usual(monkeypatch):
    """Absent/true must NOT be turned into a refusal - the switch only forces OFF."""
    calls = []

    def _fake_get_session():
        calls.append(1)
        return None  # zenoh missing: the pre-existing "runs offline" path

    monkeypatch.delenv("STRANDS_MESH", raising=False)
    monkeypatch.setattr("strands_robots.mesh.session.get_session", _fake_get_session)
    assert MeshBridge().start(loop=None) is False
    assert calls == [1], "with no switch set, the bridge must still try to open a session"

    monkeypatch.setenv("STRANDS_MESH", "true")
    assert MeshBridge().start(loop=None) is False
    assert calls == [1, 1]
