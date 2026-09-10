"""``Robot.stream`` stops for an operator before ``execute``/``start`` reach real actuators.

Finding F-011 (CWE-862): ``strands_robots.hardware_robot.Robot`` is the
``mode="real"`` half of ``strands_robots.Robot``, and its agent tool dispatched
``execute`` to ``_execute_task_sync`` and ``start`` to ``start_task`` with no
gate - while the same robot commanded through ``robot_mesh`` was gated. The
README's first path to metal, ``Agent(tools=[Robot("so100", mode="real")])``,
therefore moved servos unasked.

These tests grade the observable: whether the rollout dispatcher was called.
The gate runs before either dispatcher, so a denied call is measured as "no
dispatch", not as which branch was taken. A ``Robot`` is an ``AgentTool`` and
gets no ``tool_context`` argument, so the operator is reached the way the SDK
reaches them for a decorated tool: through ``invocation_state["agent"]``. The
fake agent here carries a real interrupt state so the path graded is the real
``ToolContext.interrupt`` one, not a stand-in. The audit row is graded beside
the other gates in ``tests/tools/test_hitl_operator_response_audit.py``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from strands.types._events import ToolInterruptEvent, ToolResultEvent
from strands.types.interrupt import Interrupt
from strands.types.tools import ToolUse

from strands_robots import hardware_robot as hardware_robot_module
from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState
from tests._daemon_executor import DaemonThreadExecutor

MOTION = {
    "execute": {"instruction": "lift the cube", "policy_port": 5555},
    "start": {"instruction": "wave", "policy_port": 5556},
}


class _Answering(dict):
    """An interrupt table whose every new question already carries the operator's reply."""

    def __init__(self, response: object) -> None:
        super().__init__()
        self._response = response

    def setdefault(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if key not in self:
            self[key] = Interrupt(default.id, default.name, default.reason, self._response)
        return self[key]


class _InterruptState:
    def __init__(self, interrupts: dict[str, Interrupt]) -> None:
        self.interrupts = interrupts


class _FakeAgent:
    """Just enough of an ``Agent`` for ``ToolContext.interrupt`` to consult it."""

    def __init__(self, response: object | None) -> None:
        self._interrupt_state = _InterruptState(_Answering(response) if response is not None else {})
        self.cancel_signal = threading.Event()


def _state(response: object | None) -> dict[str, Any]:
    """An invocation state as the SDK builds it: the agent answering (or not yet answering)."""
    return {"agent": _FakeAgent(response)}


def _drain(agen) -> list:
    async def _run() -> list:
        return [ev async for ev in agen]

    return asyncio.run(_run())


def _make_robot() -> HwRobot:
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "test_arm"
    hw.action_horizon = 8
    hw.data_config = None
    hw.control_frequency = 30.0
    hw.action_sleep_time = 1.0 / 30.0
    hw._task_state = RobotTaskState()
    hw._executor = DaemonThreadExecutor(max_workers=1, thread_name_prefix="test_arm_executor")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = object()
    return hw


@pytest.fixture
def dispatched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple[HwRobot, list[tuple[str, str]]]]:
    """A robot whose two dispatchers record instead of moving, in a clean gate environment."""
    hw = _make_robot()
    calls: list[tuple[str, str]] = []

    def _execute(instruction: str, port: int, host: str, provider: str, duration: float) -> dict[str, Any]:
        calls.append(("execute", instruction))
        return {"status": "success", "content": [{"text": "done"}]}

    def _start(instruction: str, port: int, host: str, provider: str, duration: float) -> dict[str, Any]:
        calls.append(("start", instruction))
        return {"status": "success", "content": [{"text": "started"}]}

    hw._execute_task_sync = _execute  # type: ignore[assignment]
    hw.start_task = _start  # type: ignore[assignment]
    for name in ("BYPASS_TOOL_CONSENT", hardware_robot_module.COMMAND_ALLOW_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    yield hw, calls
    hw.cleanup()


def _stream(hw: HwRobot, action: str, state: dict[str, Any], **extra: Any) -> list:
    tool_use = cast(ToolUse, {"toolUseId": f"tu-{action}", "input": {"action": action, **MOTION[action], **extra}})
    return _drain(hw.stream(tool_use, state))


def _text(events: list) -> str:
    return events[-1].tool_result["content"][0]["text"]


class TestWithoutApprovalNothingIsDispatched:
    @pytest.mark.parametrize("action", sorted(MOTION))
    def test_a_declined_command_never_reaches_the_dispatcher(self, dispatched, action: str) -> None:
        hw, calls = dispatched
        events = _stream(hw, action, _state("n - someone is holding the arm"))

        assert isinstance(events[-1], ToolResultEvent)
        assert events[-1].tool_result["status"] == "error"
        assert "declined by the operator" in _text(events)
        assert calls == [], f"dispatched after the operator said no: {calls}"

    @pytest.mark.parametrize("action", sorted(MOTION))
    def test_no_agent_refuses_and_names_both_escape_hatches(self, dispatched, action: str) -> None:
        """Headless with nothing pre-approved fails closed - a direct ``stream`` call."""
        hw, calls = dispatched
        events = _stream(hw, action, {})

        assert events[-1].tool_result["status"] == "error"
        text = _text(events)
        assert hardware_robot_module.COMMAND_ALLOW_ENV in text and "BYPASS_TOOL_CONSENT" in text
        assert calls == []

    @pytest.mark.parametrize("action", sorted(MOTION))
    def test_an_unanswered_question_pauses_the_agent_instead_of_moving(self, dispatched, action: str) -> None:
        """A live agent with no reply yet gets the interrupt event the SDK resumes from - and no motion."""
        hw, calls = dispatched
        events = _stream(hw, action, _state(None))

        assert len(events) == 1 and isinstance(events[0], ToolInterruptEvent), events
        (interrupt,) = events[0].interrupts
        assert interrupt.name == "robot-command-approval"
        assert interrupt.reason["action"] == action
        assert interrupt.reason["target"] == "test_arm"
        assert MOTION[action]["instruction"] in interrupt.reason["warning"]
        assert calls == []

    def test_a_malformed_command_is_refused_before_the_operator_is_asked(self, dispatched) -> None:
        """A human is not asked to approve a command the tool would refuse anyway."""
        hw, calls = dispatched
        agent = _FakeAgent(None)
        events = _drain(hw.stream({"toolUseId": "tu-x", "input": {"action": "execute"}}, {"agent": agent}))

        assert events[-1].tool_result["status"] == "error"
        assert "required" in _text(events)
        assert agent._interrupt_state.interrupts == {}
        assert calls == []


class TestAnApprovalDispatchesOnce:
    @pytest.mark.parametrize("action", sorted(MOTION))
    def test_an_approved_command_is_dispatched_exactly_once(self, dispatched, action: str) -> None:
        hw, calls = dispatched
        events = _stream(hw, action, _state("y"))

        assert events[-1].tool_result["status"] == "success", events
        assert calls == [(action, MOTION[action]["instruction"])]


class TestPreApprovalIsSilent:
    @pytest.mark.parametrize("action", sorted(MOTION))
    def test_an_allowlisted_action_asks_nobody(self, dispatched, monkeypatch, action: str) -> None:
        hw, calls = dispatched
        monkeypatch.setenv(hardware_robot_module.COMMAND_ALLOW_ENV, action)
        agent = _FakeAgent(None)
        events = _stream(hw, action, {"agent": agent})

        assert events[-1].tool_result["status"] == "success"
        assert agent._interrupt_state.interrupts == {}
        assert calls == [(action, MOTION[action]["instruction"])]

    def test_the_allowlist_is_per_action(self, dispatched, monkeypatch) -> None:
        hw, calls = dispatched
        monkeypatch.setenv(hardware_robot_module.COMMAND_ALLOW_ENV, "start")
        events = _stream(hw, "execute", {})

        assert events[-1].tool_result["status"] == "error"
        assert calls == []

    def test_star_pre_approves_everything(self, dispatched, monkeypatch) -> None:
        hw, calls = dispatched
        monkeypatch.setenv(hardware_robot_module.COMMAND_ALLOW_ENV, "*")
        for action in sorted(MOTION):
            assert _stream(hw, action, {})[-1].tool_result["status"] == "success"
        assert [c[0] for c in calls] == sorted(MOTION)

    def test_bypass_tool_consent_lifts_the_gate_with_a_warning(self, dispatched, monkeypatch, caplog) -> None:
        hw, calls = dispatched
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
        with caplog.at_level("WARNING", logger="strands_robots.tools._command_gate"):
            events = _stream(hw, "execute", {})

        assert events[-1].tool_result["status"] == "success"
        assert calls == [("execute", MOTION["execute"]["instruction"])]
        assert any("BYPASS_TOOL_CONSENT" in r.getMessage() for r in caplog.records)


class TestTheDashboardGrantIsSpentOnce:
    def test_a_grant_the_hook_deposited_is_spent_instead_of_asking_twice(self, dispatched) -> None:
        agent_hitl = pytest.importorskip("strands_robots.dashboard.agent_hitl")
        hw, calls = dispatched
        tool_input = {"action": "start", **MOTION["start"]}
        agent_hitl.deposit_grant("test_arm", tool_input)

        first = _drain(hw.stream({"toolUseId": "g1", "input": dict(tool_input)}, {}))
        second = _drain(hw.stream({"toolUseId": "g2", "input": dict(tool_input)}, {}))

        assert first[-1].tool_result["status"] == "success"
        assert second[-1].tool_result["status"] == "error", "the grant was spent by the first call"
        assert calls == [("start", "wave")]

    def test_a_missing_dashboard_extra_means_no_grant_not_a_crash(self, dispatched, monkeypatch) -> None:
        import builtins

        hw, calls = dispatched
        real_import = builtins.__import__

        def _no_dashboard(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("strands_robots.dashboard"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_dashboard)
        events = _stream(hw, "execute", _state("y"))

        assert events[-1].tool_result["status"] == "success"
        assert calls == [("execute", MOTION["execute"]["instruction"])]


class TestReadsAndStopsAreNeverGated:
    @pytest.mark.parametrize("action", ["status", "stop"])
    def test_status_and_stop_ask_nobody_headless(self, dispatched, action: str) -> None:
        hw, _ = dispatched
        events = _drain(hw.stream({"toolUseId": f"r-{action}", "input": {"action": action}}, {}))

        assert events[-1].tool_result["status"] == "success", events[-1].tool_result
