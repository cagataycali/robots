"""Acceptance: every agent tool answers through a real Strands ``Agent``.

Each ``@tool`` the package ships is registered on one ``Agent`` and called the
way a model calls it, through ``agent.tool.<name>``. A refused call must come
back as a failed tool result with a reason in it - Strands reads ``status`` only
off a dict that also carries ``content``, so a flat ``{"status": "error"}`` dict
reaches the model as a success. Every tool that needs no robot, network or
vendor SDK must also run for real; the rest are named with what they wait on.
A tool added to the package without a row here fails the walk.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from typing import Any

import pytest

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

pytest.importorskip("mujoco")

from strands import Agent  # noqa: E402
from strands.tools.decorator import DecoratedFunctionTool  # noqa: E402

pytestmark = pytest.mark.timeout(300)

NO_DATASET = {"root": "/nonexistent/dataset", "episode": 0}

# One call per tool that the tool must refuse. A driver-bound verb called with
# no driver handle is the refusal an agent hits first.
REFUSED: dict[str, dict[str, Any]] = {
    "download_assets": {"action": "__nope__"},
    "harness_memory": {"action": "__nope__"},
    "lerobot_camera": {"action": "__nope__"},
    "lerobot_teleoperate": {"action": "__nope__"},
    "lerobot_train": {"dataset_root": "/nonexistent/dataset"},
    "load_episode": NO_DATASET,
    "pose_tool": {"action": "__nope__"},
    "read_predicate_verdict": NO_DATASET,
    "robot_mesh": {"action": "__nope__"},
    "run_policy": {"simulation": None},
    "sample_frames": NO_DATASET,
    "serial_tool": {"action": "__nope__"},
    "train_policy": {"action": "__nope__"},
    "use_lerobot": {"module": "__nope__", "method": "x"},
    "use_ros": {"action": "__nope__"},
    "use_rosbridge": {"action": "__nope__"},
    "use_rtps": {"action": "__nope__"},
    "write_label": {**NO_DATASET, "quality": "good"},
    "g1_joints": {"query": "__nope__"},
    "g1_motion_gates": {"scope": "__nope__"},
    "g1_arm_actions": {"query": ["clap"]},
    "g1_error_codes": {"code": "seven"},
    "g1_task": {"action": "__nope__"},
    "use_unitree": {"service_name": "__nope__", "operation_name": "x"},
    **{
        name: {}
        for name in (
            "g1_arm_action g1_balance_stand g1_get_state g1_move_velocity g1_release_arm g1_run_policy "
            "g1_safe_lie_to_stand g1_safe_squat_to_stand g1_safe_stand_to_squat g1_send_action g1_sensor "
            "g1_set_fsm g1_set_stand_height g1_set_swing_height g1_shake_hand_loco g1_stop_move "
            "g1_wave_hand_loco reachy_antennas reachy_body_turn reachy_camera reachy_express "
            "reachy_get_state reachy_home reachy_list_emotions reachy_look reachy_look_at reachy_motors "
            "reachy_play_sound reachy_stop reachy_volume reachy_wake"
        ).split()
    },
}

# One call per tool that must succeed on a machine with no robot attached.
RUNS: dict[str, dict[str, Any]] = {
    "download_assets": {"action": "list"},
    "lerobot_camera": {"action": "discover"},
    "lerobot_teleoperate": {"action": "list"},
    "train_policy": {"action": "list"},
    "use_lerobot": {"module": "__discovery__", "method": "list_modules"},
    "use_rtps": {"action": "status"},
    "use_unitree": {"service_name": "meta", "operation_name": "list_services"},
    "g1_joints": {"query": "left_knee"},
    "g1_motion_gates": {},
    "g1_arm_actions": {"query": "clap"},
    "g1_error_codes": {"code": 0},
    "stop_conversation": {},
}

# What the real-run half of every other tool waits on.
WAITS_ON = {
    "the live Robot handle (run below)": ["run_policy"],
    "a recorded dataset": ["load_episode", "sample_frames", "read_predicate_verdict", "write_label"],
    "a serial bus or arm": ["serial_tool", "pose_tool", "lerobot_train"],
    "a writable trace store": ["harness_memory"],
    "a mesh or ROS 2 graph": ["robot_mesh", "use_ros", "use_rosbridge"],
    "a G1 or Reachy Mini": [n for n in REFUSED if n.startswith(("g1_", "reachy_")) and n not in RUNS],
}


def _tools() -> dict[str, DecoratedFunctionTool]:
    import strands_robots.tools as package

    names = [m.name for m in pkgutil.walk_packages(package.__path__, f"{package.__name__}.")]
    found: dict[str, DecoratedFunctionTool] = {}
    for name in [*names, "strands_robots.dashboard.voice"]:
        module = importlib.import_module(name)
        for value in vars(module).values():
            if isinstance(value, DecoratedFunctionTool) and value.__module__ == module.__name__:
                found[value.tool_name] = value
    return found


TOOLS = _tools()
AGENT = Agent(tools=list(TOOLS.values()), callback_handler=None)


def _reason(result: dict[str, Any]) -> str:
    block = result["content"][0]
    return str(block.get("text") or block.get("json", {}).get("message", ""))


def test_every_shipped_tool_has_a_row() -> None:
    waiting = {name for names in WAITS_ON.values() for name in names}
    assert set(TOOLS) == set(REFUSED) | {"stop_conversation"}
    assert set(TOOLS) == set(RUNS) | waiting, sorted(set(TOOLS) ^ (set(RUNS) | waiting))


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_a_refused_call_reaches_the_agent_as_a_failed_call(name: str) -> None:
    result = getattr(AGENT.tool, name)(**REFUSED[name])
    assert result["status"] == "error", result
    assert _reason(result).strip(), result


@pytest.mark.parametrize("name", sorted(RUNS))
def test_a_tool_that_needs_no_robot_runs(name: str) -> None:
    result = getattr(AGENT.tool, name)(**RUNS[name])
    assert result["status"] == "success", result


def test_run_policy_drives_a_live_sim_robot() -> None:
    from strands_robots import Robot

    robot = Robot("so101", mode="sim")
    try:
        result = AGENT.tool.run_policy(simulation=robot, robot_name="so101", policy_provider="mock", n_steps=5)
        assert result["status"] == "success", result
    finally:
        robot.destroy()
