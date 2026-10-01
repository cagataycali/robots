"""Regression for GH #4180: the policy's unit frame travels with the wire command.

A degrees trained SO-101 checkpoint named over the mesh drove the sim in radians:
``embodiment`` is the ``lerobot_local`` constructor kwarg that carries the
state/action unit conversion and the observation renames, and neither
:func:`strands_robots.mesh.security.validate_command` nor the dispatcher's
``extra`` tuple admitted it, so the wire dropped it silently. In process the same
checkpoint with ``embodiment="so101"`` stayed inside every joint range; over the
wire the arm pinned at five of six limits and the peer reported success.

Pinned here, end to end but without weights: the validator carries a registry
name and refuses anything else (an inline map is a constructor object the wire
never carries), both dispatch branches hand it to the policy constructor, and
every dashboard roster that mirrors the wire schema names it too.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.mesh import Mesh, security


class _FakeSim:
    """The SimEngine surface ``Mesh._dispatch`` keys the sim branch off."""

    def __init__(self) -> None:
        self._world: Any = object()
        self.run_policy_calls: list[dict[str, Any]] = []
        self.tool_name_str = "fakesim"

    def list_robots(self) -> list[str]:
        return ["so101"]

    def run_policy(self, robot_name: str, **kwargs: Any) -> dict[str, Any]:
        self.run_policy_calls.append(kwargs)
        return {"status": "success", "content": [{"text": f"ran {robot_name}"}]}


class _FakeArm:
    """The HardwareRobot entry point the dispatcher binds by keyword."""

    tool_name_str = "arm"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _execute_task_sync(self, instruction: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"instruction": instruction, **kwargs})
        return {"status": "success"}


def _cmd(**over: Any) -> dict[str, Any]:
    cmd: dict[str, Any] = {
        "action": "execute",
        "instruction": "pick up the red cube",
        "policy_provider": "lerobot_local",
        "pretrained_name_or_path": "lerobot/smolvla_base",
        "embodiment": "so101",
    }
    cmd.update(over)
    return cmd


def test_validate_command_carries_a_registry_embodiment_name() -> None:
    out = security.validate_command(_cmd())
    assert out["embodiment"] == "so101"


@pytest.mark.parametrize(
    "bad",
    [
        {"state_unit": "deg", "action_unit": "deg"},
        "",
        "so101 real",
        "so101/../x",
        7,
        "x" * (security.MAX_PEER_ID_LEN + 1),
    ],
    ids=["inline-map", "empty", "space", "slash", "int", "too-long"],
)
def test_validate_command_refuses_an_embodiment_that_is_not_a_name(bad: Any) -> None:
    with pytest.raises(security.ValidationError, match="embodiment"):
        security.validate_command(_cmd(embodiment=bad))


def test_sim_dispatch_hands_embodiment_to_the_policy_constructor() -> None:
    sim = _FakeSim()
    Mesh(sim, peer_id="sim-a")._dispatch(_cmd())
    kwargs = sim.run_policy_calls[0]
    assert kwargs["policy_config"]["embodiment"] == "so101"
    # A constructor extra, not a per call goal.
    assert "embodiment" not in kwargs["policy_kwargs"]


def test_hardware_dispatch_hands_embodiment_to_the_policy_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    # A wire execute on a real robot passes the operator gate on the robot host
    # first; pre-approve this one verb so the test reaches the dispatcher.
    monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "execute")
    arm = _FakeArm()
    Mesh(arm, peer_id="arm-a")._dispatch(_cmd())
    assert arm.calls[0]["embodiment"] == "so101"


def test_every_dashboard_wire_roster_names_embodiment() -> None:
    """The dashboard mirrors the wire schema in four places; each must carry the key."""
    from strands_robots.dashboard import config_api, peer_tools, policy_fit, routes_mesh, routes_record

    assert "embodiment" in routes_mesh.WIRE_CMD_KEYS
    assert "embodiment" in policy_fit.WIRE_CMD_KEYS
    assert "embodiment" in config_api.WIRE_CMD_KEYS
    assert config_api.WIRE_KEY_TYPES["embodiment"] == "string"
    assert "embodiment" in routes_record.WIRE_POLICY_CONFIG_KEYS
    for verb in ("execute", "start"):
        assert "embodiment" in peer_tools._SIM_FIELDS[verb]
    assert "embodiment" in peer_tools.sim_input_schema()["properties"]
