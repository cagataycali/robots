"""Measure what a hardware task does with an unusable ``policy_port``.

Run in two checkouts; each dump records its own tree so the compose step can
prove the two halves came from different code.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import strands_robots.hardware_robot as hardware_robot

TREE = str(pathlib.Path(hardware_robot.__file__).parents[1])
print("TREE:", TREE)

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState, TaskStatus


class _Arm:
    def __init__(self) -> None:
        self.name = "fake_arm"
        self.robot_type = "fake_arm"
        self.sent_actions: list[dict[str, Any]] = []
        self.config = type("Cfg", (), {"cameras": {}})()

    def get_observation(self) -> dict[str, Any]:
        return {"j0.pos": 0.0}

    def send_action(self, action: dict[str, Any]) -> None:
        self.sent_actions.append(action)


def mk(gate: threading.Event | None = None) -> Any:
    r = HwRobot.__new__(HwRobot)
    r.tool_name_str = "so101_follower"
    r.action_horizon = 1
    r.data_config = None
    r.control_frequency = 50.0
    r.action_sleep_time = 1 / 50
    r._task_state = RobotTaskState()
    r._executor = ThreadPoolExecutor(max_workers=2)
    r._shutdown_event = threading.Event()
    r._stop_requested = threading.Event()
    r._task_admission = threading.Lock()
    r._task_claimed = False
    r.mesh = None
    r.peer_id = None
    r.robot = _Arm()
    r.connects: list[float] = []

    async def _connected() -> tuple[bool, str]:
        r.connects.append(1.0)
        if gate is not None:
            gate.wait(timeout=10.0)
        return (True, "")

    async def _ready() -> bool:
        return True

    r._connect_robot = _connected
    r._initialize_policy = lambda policy: _ready()
    r._publish_ros_telemetry = lambda observation, *, skip_images=False: None
    return r


def text(res: dict[str, Any]) -> str:
    return " ".join(b.get("text", "") for b in res.get("content", []) if isinstance(b, dict))


PORTS: list[tuple[str, Any]] = [
    ("5555", 5555),
    ("0", 0),
    ("-1", -1),
    ("99999", 99999),
    ("nan", math.nan),
    ("True", True),
    ("'5555'", "5555"),
    ("None", None),
]

facts: dict[str, Any] = {"tree": TREE, "rows": {}, "denial": {}}

for label, port in PORTS:
    r = mk()
    res = r.start_task("pick up the cube", policy_port=port, policy_provider="groot", duration=0.2)
    fut = r._task_state.task_future
    submitted = fut is not None
    if fut is not None:
        try:
            fut.result(timeout=20)
        except BaseException:  # noqa: BLE001 - a probe records the outcome
            pass
    facts["rows"][label] = {
        "start_status": res.get("status"),
        "start_text": text(res)[:220],
        "submitted": submitted,
        "connects": len(r.connects),
        "commands": len(r.robot.sent_actions),
        "final_status": r._task_state.status.value,
        "final_error": r._task_state.error_message[:120],
        "refused_for_port": ("policy_port" in text(res)) or ("policy_port" in r._task_state.error_message),
    }
    r._executor.shutdown(wait=False)

# The bus-denial consequence: a task whose port can never build a policy holds
# the single command bus through its whole bring-up window, so a concurrent
# legitimate task is turned away.
gate = threading.Event()
r = mk(gate=gate)
first = r.start_task("bad port", policy_port=99999, policy_provider="groot", duration=0.2)
threading.Event().wait(0.35)  # let the executor thread reach the blocked connect
second = r.start_task("pick up the cube", policy_port=5555, policy_provider="groot", duration=0.2)
gate.set()
fut = r._task_state.task_future
if fut is not None:
    try:
        fut.result(timeout=20)
    except BaseException:  # noqa: BLE001 - a probe records the outcome
        pass
facts["denial"] = {
    "first_status": first.get("status"),
    "first_text": text(first)[:200],
    "second_status": second.get("status"),
    "second_text": text(second)[:200],
    "connects": len(r.connects),
    "commands": len(r.robot.sent_actions),
}
r._executor.shutdown(wait=False)

out = pathlib.Path(sys.argv[1])
out.write_text(json.dumps(facts, indent=2), encoding="utf-8")
print(json.dumps(facts["denial"], indent=2))
print("wrote", out)
