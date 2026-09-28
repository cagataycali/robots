"""Every command entry point of the hardware ``Robot`` refuses identically.

``start_task``, ``execute_task``, ``run_policy`` and the agent tool's
``execute`` / ``start`` (before the operator is asked) all run
``Robot._preflight``. One table of robot states, one row per state: every entry
point the state applies to must answer with the same refusal (after its own
``method:`` prefix), and none may connect the arm or keep the bus claim.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import pytest
from strands.types._events import ToolResultEvent

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState
from tests._daemon_executor import DaemonThreadExecutor


class _Arm:
    name = "so101"
    robot_type = "so_follower"
    is_connected = False
    config = type("Cfg", (), {"port": "/dev/null", "cameras": {}})()

    def __init__(self) -> None:
        self.connects = 0

    def connect(self, *a: Any, **k: Any) -> None:
        self.connects += 1
        raise RuntimeError("the arm was connected")


def _hw() -> HwRobot:
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "so101"
    hw.data_config = None
    hw._task_state = RobotTaskState()
    hw._executor = DaemonThreadExecutor(max_workers=1, thread_name_prefix="t")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = _Arm()
    # A call that passes every check reaches the operator; say so distinctly.
    hw._gate_motion = lambda *a, **k: "GATED"  # type: ignore[method-assign]
    return hw


def _tool(action: str) -> Callable[..., dict[str, Any]]:
    def call(hw: HwRobot, provider: str, port: Any, duration: Any, **kwargs: Any) -> dict[str, Any]:
        tool_use = {
            "toolUseId": "t",
            "input": {
                "action": action,
                "instruction": "pick",
                "policy_provider": provider,
                "policy_port": port,
                "duration": duration,
            },
        }

        async def first() -> dict[str, Any]:
            async for event in hw.stream(tool_use, {}):  # type: ignore[arg-type]
                assert isinstance(event, ToolResultEvent), event
                return dict(event.tool_result)
            raise AssertionError("no result")

        return asyncio.run(first())

    return call


# name -> (method prefix, call)
ENTRY_POINTS: dict[str, tuple[str, Callable[..., dict[str, Any]]]] = {
    "start_task": ("start_task", lambda hw, pr, po, d, **k: hw.start_task("pick", po, "localhost", pr, d, **k)),
    "execute_task": (
        "execute_task",
        lambda hw, pr, po, d, **k: hw._execute_task_sync("pick", po, "localhost", pr, d, **k),
    ),
    "tool execute": ("execute_task", _tool("execute")),
    "tool start": ("start_task", _tool("start")),
    "run_policy": ("run_policy", lambda hw, pr, po, d, **k: hw.run_policy(object(), "pick", d, **k)),  # type: ignore[arg-type]
}

# state -> (arrange, provider, port, duration, policy kwargs, applies to, refusal fragment)
_ROLLOUTS = ("start_task", "execute_task", "tool execute", "tool start", "run_policy")
_BUILDERS = ("start_task", "execute_task", "tool execute", "tool start")
STATES: dict[str, tuple[Callable[[HwRobot], None], str, Any, Any, dict[str, Any], tuple[str, ...], str]] = {
    "shut down": (lambda hw: hw._shutdown_event.set(), "mock", None, 1.0, {}, _ROLLOUTS, "shut down"),
    "zero budget": (lambda hw: None, "mock", None, 0, {}, _ROLLOUTS, "duration"),
    "unknown provider": (lambda hw: None, "grooot", None, 1.0, {}, _BUILDERS, "grooot"),
    "port out of range": (lambda hw: None, "groot", 70000, 1.0, {}, _BUILDERS, "70000"),
    "zero step cap": (lambda hw: None, "mock", None, 1.0, {"n_steps": 0}, ("execute_task", "run_policy"), "n_steps"),
    "checkpoint missing": (lambda hw: None, "lerobot_local", None, 1.0, {}, _BUILDERS, "pretrained_name_or_path"),
}


def _refusal(entry: str, result: dict[str, Any]) -> str:
    assert result["status"] == "error", f"{entry} admitted the call: {result}"
    method = ENTRY_POINTS[entry][0]
    text = result["content"][0]["text"]
    assert text.startswith(f"{method}: "), f"{entry} did not speak for {method}: {text!r}"
    return text.removeprefix(f"{method}: ")


@pytest.mark.parametrize("state", sorted(STATES))
def test_identical_states_get_identical_refusals(state: str) -> None:
    arrange, provider, port, duration, kwargs, applies, fragment = STATES[state]
    refusals: dict[str, str] = {}
    for entry in applies:
        hw = _hw()
        arrange(hw)
        refusals[entry] = _refusal(entry, ENTRY_POINTS[entry][1](hw, provider, port, duration, **kwargs))
        assert hw.robot.connects == 0, f"{entry} connected the arm"
        assert hw._task_claimed is False, f"{entry} kept the bus claim"
    assert fragment in next(iter(refusals.values()))
    assert len(set(refusals.values())) == 1, refusals


def test_a_second_rollout_is_refused_by_every_claiming_entry_point() -> None:
    for entry in ("start_task", "execute_task", "run_policy"):
        hw = _hw()
        hw._task_claimed = True
        result = ENTRY_POINTS[entry][1](hw, "mock", None, 1.0)
        assert result["content"][0]["text"].startswith("Task already running"), entry
        assert hw.robot.connects == 0


@pytest.mark.parametrize("entry", ["tool execute", "tool start"])
def test_a_sound_call_still_reaches_the_operator(entry: str) -> None:
    """The over-reach control: the preflight refuses nothing it should pass."""
    result = ENTRY_POINTS[entry][1](_hw(), "mock", None, 1.0)
    assert result["content"][0]["text"] == "so101: GATED"
