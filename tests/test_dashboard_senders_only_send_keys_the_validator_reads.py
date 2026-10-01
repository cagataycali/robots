"""Every command an in-tree dashboard sender builds passes ``validate_command``.

The validator refuses a key it does not read (so a misspelt option can no
longer be dropped silently). The dashboard has two senders that build commands
for a peer without going through a dispatcher: ``route_task_target`` injects
``robot_name`` into a command aimed at a child sim peer, and the sim proxy tool's
``_SIM_FIELDS`` forwards the model's fields per action. Both used to send
``robot_name`` on ``state`` and ``reset``, which no dispatcher reads; the suites
stayed green because they faked the bridge above ``Mesh.send`` while the live
dashboard reset button returned ``ok=False`` and reset nothing. This grader
runs the real validator on what the senders build.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.dashboard import peer_tools
from strands_robots.dashboard.mesh_bridge import route_task_target
from strands_robots.mesh import security

_SAMPLE_INPUT: dict[str, Any] = {
    "robot_name": "so101",
    "target_joints": {"1": 0.1},
    "hold": True,
    "steps": 3,
    "instruction": "wave",
    "policy_provider": "mock",
    "duration": 2.0,
}


@pytest.mark.parametrize("action", sorted(peer_tools._SIM_FIELDS))
@pytest.mark.parametrize("names_the_robot", [False, True], ids=["routed-to-parent", "robot-named"])
def test_a_sim_proxy_command_aimed_at_a_child_peer_passes_the_validator(action: str, names_the_robot: bool) -> None:
    cmd: dict[str, Any] = {"action": action}
    for field in peer_tools._SIM_FIELDS[action]:
        if field in _SAMPLE_INPUT and (names_the_robot or field != "robot_name"):
            cmd[field] = _SAMPLE_INPUT[field]
    if action in ("execute", "start"):
        cmd.update({"instruction": "wave", "policy_provider": "mock", "duration": 2.0})
    _target, routed = route_task_target("lane__so101", cmd)
    security.validate_command(routed)  # raises ValidationError on a key no dispatcher reads


def test_the_child_peer_reset_is_routed_to_its_parent_without_a_key_no_dispatcher_reads() -> None:
    target, cmd = route_task_target("lane__so101", {"action": "reset"})
    assert (target, cmd) == ("lane", {"action": "reset"})
    security.validate_command(cmd)


def test_a_child_peer_set_joints_still_names_its_robot() -> None:
    target, cmd = route_task_target("lane__so101", {"action": "set_joints", "target_joints": {"1": 0.1}})
    assert target == "lane" and cmd["robot_name"] == "so101"
    security.validate_command(cmd)


def test_the_sim_fields_table_only_forwards_keys_the_validator_reads() -> None:
    for action, fields in peer_tools._SIM_FIELDS.items():
        admitted = security.COMMAND_KEYS.get(action, frozenset())
        assert set(fields) <= admitted, f"{action}: {sorted(set(fields) - admitted)} would be refused on the wire"
