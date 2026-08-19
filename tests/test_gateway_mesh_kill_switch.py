"""Q32: STRANDS_MESH=false must stop the robot-less GATEWAY mesh too.

`init_mesh` has always honoured the kill switch, but `robot_mesh._gateway_mesh()`
built a `Mesh` directly, so a robot-less process joined the live fleet on the
first `robot_mesh` call — publishing presence and declaring subscribers whose
non-daemon pyo3 callback threads then hung interpreter shutdown. That is how a
test run became a live `gateway-*` peer on the operator's fleet screen.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from strands_robots.mesh.core import mesh_kill_switch_engaged
import strands_robots.tools.robot_mesh as rmt


@pytest.mark.parametrize("value", ["false", "FALSE", " no ", "0"])
def test_switch_engaged_for_every_documented_off_value(value: str) -> None:
    assert mesh_kill_switch_engaged({"STRANDS_MESH": value}) is True


@pytest.mark.parametrize("value", ["true", "1", "yes", "", "maybe"])
def test_switch_only_ever_forces_off(value: str) -> None:
    # An unset or affirmative value must not be read as "disabled" — the switch
    # is one-directional by design.
    assert mesh_kill_switch_engaged({"STRANDS_MESH": value}) is False
    assert mesh_kill_switch_engaged({}) is False


def test_gateway_mesh_refuses_to_construct_when_switch_is_engaged(monkeypatch) -> None:
    monkeypatch.setenv("STRANDS_MESH", "false")
    rmt._GATEWAY.pop("mesh", None)
    with patch("strands_robots.mesh.core.Mesh") as MeshCls:
        assert rmt._gateway_mesh() is None
    # The point is not just the None: no Mesh may be BUILT, because building one
    # is what opens the session and declares the subscribers.
    MeshCls.assert_not_called()


def test_gateway_mesh_still_builds_when_mesh_is_allowed(monkeypatch) -> None:
    monkeypatch.delenv("STRANDS_MESH", raising=False)
    rmt._GATEWAY.pop("mesh", None)
    with patch("strands_robots.mesh.core.Mesh") as MeshCls:
        MeshCls.return_value.alive = True
        MeshCls.return_value.peer_id = "gateway-test-0000"
        rmt._gateway_mesh()
    MeshCls.assert_called_once()
    rmt._GATEWAY.pop("mesh", None)
