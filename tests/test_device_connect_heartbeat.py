"""Heartbeat payload contract for the Device Connect runtime.

``_build_heartbeat`` is the provider wired into every DeviceRuntime via
``set_heartbeat_provider``. Its output is what peers see on the mesh, so the
per-``peer_type`` schema is a contract: a ``robot`` peer advertises its current
task (status / instruction / step count) and a ``sim`` peer advertises world
telemetry (sim time / step count / robot roster). These tests pin that schema
so the branches cannot silently drift.
"""

from __future__ import annotations

from strands_robots.device_connect import _build_heartbeat


class _Status:
    def __init__(self, value: str) -> None:
        self.value = value


class _Task:
    def __init__(self, status: str, instruction: str | None, step_count: int) -> None:
        self.status = _Status(status)
        self.instruction = instruction
        self.step_count = step_count


class _Robot:
    """Minimal robot stand-in carrying the attributes the provider reads."""

    def __init__(self, tool_name=None, task=None):
        if tool_name is not None:
            self.tool_name_str = tool_name
        if task is not None:
            self._task_state = task


class _World:
    def __init__(self, sim_time, step_count, robots):
        self.sim_time = sim_time
        self.step_count = step_count
        self.robots = robots


class _Sim:
    def __init__(self, tool_name=None, world=None):
        if tool_name is not None:
            self.tool_name_str = tool_name
        if world is not None:
            self._world = world


def test_base_payload_always_carries_peer_type_and_tool_name():
    data = _build_heartbeat(_Robot(tool_name="my_arm"), "robot")
    assert data["peer_type"] == "robot"
    assert data["tool_name"] == "my_arm"


def test_missing_tool_name_falls_back_to_unknown():
    # An object with neither tool_name_str nor task/world state.
    data = _build_heartbeat(object(), "operator")
    assert data == {"peer_type": "operator", "tool_name": "unknown"}


def test_robot_peer_advertises_active_task():
    robot = _Robot(tool_name="picker", task=_Task("running", "stack the cubes", 42))
    data = _build_heartbeat(robot, "robot")
    assert data["task_status"] == "running"
    assert data["instruction"] == "stack the cubes"
    assert data["step_count"] == 42


def test_robot_peer_empty_instruction_normalized_to_empty_string():
    robot = _Robot(tool_name="picker", task=_Task("idle", None, 0))
    data = _build_heartbeat(robot, "robot")
    assert data["instruction"] == ""


def test_robot_peer_without_task_omits_task_fields():
    data = _build_heartbeat(_Robot(tool_name="picker"), "robot")
    assert "task_status" not in data
    assert "instruction" not in data
    assert "step_count" not in data


def test_sim_peer_advertises_world_telemetry():
    sim = _Sim(tool_name="mujoco", world=_World(1.5, 75, {"so101": object(), "panda": object()}))
    data = _build_heartbeat(sim, "sim")
    assert data["sim_time"] == 1.5
    assert data["step_count"] == 75
    assert sorted(data["robots"]) == ["panda", "so101"]


def test_sim_peer_without_world_omits_world_fields():
    data = _build_heartbeat(_Sim(tool_name="mujoco"), "sim")
    assert "sim_time" not in data
    assert "robots" not in data
