"""An operator or gateway peer gets no Device Shadow mirror on the IoT backends.

Reproduced live 2026-09-30 (account 947951559549): ``init_mesh(console,
peer_id="ac-ops-01", peer_type="operator")`` on ``STRANDS_MESH_BACKEND=iot``
published ``$aws/things/ac-ops-01/shadow/name/presence/update`` on every
heartbeat. The shipped ``strands-operator`` policy grants no MQTT publish on
``$aws/things/...`` (its shadow statement is the REST pair on
``thing/strands-*``), so AWS ended the operator's session with reason 135
about once a second and every direct reply, which needs the ``response/#``
subscription, was lost: 5/5 ``status`` commands timed out at 10 s; with the
mirror off the first reply took 450 ms and the next four 226 ms.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh.iot.shadow import NON_ROBOT_PEER_TYPES, enable_for_mesh


def _mesh(peer_type: str) -> MagicMock:
    mesh = MagicMock()
    mesh.peer_id = "ac-ops-01"
    mesh.peer_type = peer_type
    mesh._build_presence = MagicMock(return_value={"robot_id": "ac-ops-01"})
    return mesh


def _on_iot():
    transport = MagicMock(is_alive=MagicMock(return_value=True))
    return (
        patch("strands_robots.mesh.transport.factory.current_backend", return_value="iot"),
        patch("strands_robots.mesh.transport.factory.current_transport", return_value=transport),
        transport,
    )


@pytest.mark.parametrize("peer_type", sorted(NON_ROBOT_PEER_TYPES))
def test_an_operator_side_peer_gets_no_mirror_and_its_heartbeat_publishes_no_shadow(peer_type, caplog):
    mesh = _mesh(peer_type)
    original = mesh._build_presence
    backend, transport_patch, transport = _on_iot()
    with backend, transport_patch, caplog.at_level(logging.INFO, logger="strands_robots.mesh.iot.shadow"):
        assert enable_for_mesh(mesh) is None
    assert mesh._build_presence is original
    mesh._build_presence()
    transport.put.assert_not_called()
    assert f"{peer_type} peer" in caplog.text and "no shadow mirror" in caplog.text


def test_the_two_operator_side_types_are_the_ones_the_shipped_policies_name():
    assert NON_ROBOT_PEER_TYPES == frozenset({"operator", "gateway"})


def test_a_robot_peer_keeps_its_mirror():
    mesh = _mesh("robot")
    backend, transport_patch, transport = _on_iot()
    with backend, transport_patch:
        assert enable_for_mesh(mesh) is not None
    mesh._build_presence()
    topic, _payload = transport.put.call_args.args
    assert topic == "$aws/things/ac-ops-01/shadow/name/presence/update"


def test_a_sim_peer_keeps_its_mirror_too():
    mesh = _mesh("sim")
    backend, transport_patch, transport = _on_iot()
    with backend, transport_patch:
        assert enable_for_mesh(mesh) is not None
