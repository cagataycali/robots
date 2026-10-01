"""The dashboard's signed safety rail speaks as the IoT Thing it connects with.

The bridge's robot-less :class:`~strands_robots.mesh.core.Mesh` (the rail that
signs e-stops and carries every command) announced itself as ``<dashboard>-safety``
on every backend. Over AWS IoT Core that peer id is the MQTT topic root
(``strands/<peer>/presence``), and the ``strands-operator`` policy lets a Thing
publish only under its own name, so the rail's first presence publish had the
broker drop the shared session: the dashboard's IoT leg died the moment the
operator sent the first command, and any reply addressed to
``strands/<dashboard>-safety/response/...`` was one the operator could never
subscribe to. On a backend with an IoT leg the rail's peer id is therefore the
Thing name; on plain Zenoh it stays ``<dashboard>-safety``.
"""

from __future__ import annotations

import inspect

import pytest

from strands_robots.dashboard import mesh_bridge
from strands_robots.dashboard.mesh_bridge import MeshBridge, safety_rail_peer_id


@pytest.mark.parametrize(
    ("backend", "thing", "expected"),
    [
        ("zenoh", "", "dash-safety"),
        ("zenoh", "dashiot-op", "dash-safety"),
        ("bridge", "dashiot-op", "dashiot-op"),
        ("iot", "dashiot-op", "dashiot-op"),
        ("bridge", "", "dash-safety"),
        ("", "dashiot-op", "dash-safety"),
    ],
)
def test_the_rail_speaks_as_the_thing_where_the_broker_binds_the_topic_root(
    monkeypatch: pytest.MonkeyPatch, backend: str, thing: str, expected: str
) -> None:
    monkeypatch.setenv("STRANDS_MESH_BACKEND", backend)
    monkeypatch.setenv("STRANDS_IOT_THING_NAME", thing)
    assert safety_rail_peer_id("dash") == expected


def test_the_rail_is_constructed_with_that_id() -> None:
    src = inspect.getsource(MeshBridge._safety_mesh)
    assert "peer_id=safety_rail_peer_id(self.peer_id)" in src
    assert '-safety"' not in src, "the suffix is spelled once, in safety_rail_peer_id"


def test_the_rail_is_never_an_estop_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever it is called, the dashboard's own rail is not a host to stop.

    The e-stop fan-out used to skip peers by the ``-safety`` suffix; a rail
    named after the Thing has no suffix and must still be skipped.
    """
    monkeypatch.setenv("STRANDS_MESH_BACKEND", "bridge")
    monkeypatch.setenv("STRANDS_IOT_THING_NAME", "dashiot-op")
    b = MeshBridge(peer_id="dash")
    assert b.rail_peer_id == "dashiot-op"
    assert b.is_own_rail("dashiot-op") is True
    assert b.is_own_rail("dash-safety") is True, "an older rail id is still the dashboard's own"
    assert b.is_own_rail("dashiot-so101") is False
    src = inspect.getsource(mesh_bridge.MeshBridge)
    assert 'pid.endswith("-safety")' not in src, "the fan-out filter goes through is_own_rail"
    assert "self.is_own_rail(pid)" in src
