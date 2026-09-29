"""The hardware agent tool can name the checkpoint a ``lerobot_local`` rollout is built from.

``Robot.tool_spec`` (the ``mode="real"`` half of ``strands_robots.Robot``) exposed
``instruction`` / ``policy_provider`` / ``policy_host`` / ``policy_port`` /
``duration`` and nothing else, and ``stream`` dispatched exactly those five. The
``**policy_kwargs`` that ``start_task`` and ``_execute_task_sync`` accept - and
that the mesh ``execute`` RPC fills from the wire command - were unreachable from
the agent tool, while the tool's own description told the agent that
``lerobot_local`` needs ``pretrained_name_or_path``. So an agent could run a
Hugging Face checkpoint on a simulated arm (the sim tool takes ``policy_config``)
and on a real arm over the mesh, but not on the real arm in front of it.

These cells grade the closed gap: the property exists on the spec, an approved
call hands the bag to the dispatcher, the approval prompt names the checkpoint the
operator is approving, and the three refusals - wrong shape, a host/port smuggled
into the bag, a checkpoint provider with no checkpoint - happen before the gate and
before any dispatcher runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from strands.types._events import ToolInterruptEvent, ToolResultEvent
from strands.types.interrupt import Interrupt
from strands.types.tools import ToolUse

from strands_robots import hardware_robot as hardware_robot_module
from strands_robots.hardware_robot import Robot as HwRobot
from tests._hardware_robot import hardware_robot_on

pytestmark = pytest.mark.usefixtures("named_rpc_caller")

CHECKPOINT = "lerobot/act_so101_cubes"


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


class _Agent:
    def __init__(self, response: object | None) -> None:
        table: dict[str, Interrupt] = _Answering(response) if response is not None else {}
        self._interrupt_state = _InterruptState(table)


def _state(response: object | None) -> dict[str, Any]:
    return {"agent": _Agent(response)}


def _drain(agen: Any) -> list:
    async def _run() -> list:
        return [event async for event in agen]

    return asyncio.run(_run())


def _make_robot() -> HwRobot:
    hw = hardware_robot_on(object(), tool_name="test_arm", control_frequency=30.0)
    return hw


@pytest.fixture
def dispatched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple[HwRobot, list[dict[str, Any]]]]:
    """A robot whose two dispatchers record their keyword bag instead of moving."""
    hw = _make_robot()
    calls: list[dict[str, Any]] = []

    def _execute(instruction: str, port: Any, host: str, provider: str, duration: float, **kw: Any) -> dict[str, Any]:
        calls.append({"action": "execute", "provider": provider, "port": port, "host": host, **kw})
        return {"status": "success", "content": [{"text": "done"}]}

    def _start(instruction: str, port: Any, host: str, provider: str, duration: float, **kw: Any) -> dict[str, Any]:
        calls.append({"action": "start", "provider": provider, "port": port, "host": host, **kw})
        return {"status": "success", "content": [{"text": "started"}]}

    hw._execute_task_sync = _execute  # type: ignore[assignment]
    hw.start_task = _start  # type: ignore[assignment]
    for name in ("BYPASS_TOOL_CONSENT", hardware_robot_module.COMMAND_ALLOW_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    yield hw, calls
    hw.cleanup()


def _stream(hw: HwRobot, state: dict[str, Any], **tool_input: Any) -> list:
    tool_use = cast(ToolUse, {"toolUseId": "tu-1", "input": {"action": "execute", "instruction": "pick", **tool_input}})
    return _drain(hw.stream(tool_use, state))


def _text(events: list) -> str:
    return " ".join(
        block["text"]
        for event in events
        if isinstance(event, ToolResultEvent)
        for block in event.tool_result["content"]
        if "text" in block
    )


class TestTheSpecOffersTheBag:
    def test_policy_config_is_an_object_property_that_names_the_checkpoint_key(self) -> None:
        """The description already told the agent about ``pretrained_name_or_path``; now there is somewhere to put it."""
        props = _make_robot().tool_spec["inputSchema"]["json"]["properties"]
        assert props["policy_config"]["type"] == "object"
        assert props["policy_config"]["additionalProperties"] is True
        assert "pretrained_name_or_path" in props["policy_config"]["description"]
        assert "policy_host" in props["policy_config"]["description"], "the bag must say host/port live elsewhere"


class TestAnApprovedCallCarriesTheBag:
    @pytest.mark.parametrize("action", ["execute", "start"])
    def test_the_checkpoint_reaches_the_dispatcher(self, dispatched: Any, action: str) -> None:
        hw, calls = dispatched
        tool_use = cast(
            ToolUse,
            {
                "toolUseId": "tu-ok",
                "input": {
                    "action": action,
                    "instruction": "pick the cube",
                    "policy_provider": "lerobot_local",
                    "policy_config": {"pretrained_name_or_path": CHECKPOINT, "device": "cpu"},
                },
            },
        )
        events = _drain(hw.stream(tool_use, _state("y")))
        assert "error" not in _text(events).lower(), _text(events)
        assert calls == [
            {
                "action": action,
                "provider": "lerobot_local",
                "port": None,
                "host": "localhost",
                "pretrained_name_or_path": CHECKPOINT,
                "device": "cpu",
            }
        ]

    def test_no_bag_dispatches_exactly_as_before(self, dispatched: Any) -> None:
        """The five-argument call shape is unchanged for every existing caller."""
        hw, calls = dispatched
        events = _stream(hw, _state("y"), policy_provider="mock")
        assert "error" not in _text(events).lower(), _text(events)
        assert calls == [{"action": "execute", "provider": "mock", "port": None, "host": "localhost"}]


class TestTheOperatorReadsTheCheckpoint:
    def test_the_approval_prompt_names_the_model_about_to_drive_the_arm(self, dispatched: Any) -> None:
        hw, calls = dispatched
        events = _stream(
            hw,
            _state(None),
            policy_provider="lerobot_local",
            policy_config={"pretrained_name_or_path": CHECKPOINT},
        )
        pauses = [event for event in events if isinstance(event, ToolInterruptEvent)]
        assert pauses, "the agent was not paused for the operator"
        reason = str(pauses[0].interrupts[0].reason)
        assert CHECKPOINT in reason, reason
        assert "checkpoint" in reason, reason
        assert calls == [], "the arm was dispatched before the operator answered"

    def test_a_bag_that_names_no_model_adds_nothing_to_the_prompt(self, dispatched: Any) -> None:
        hw, _calls = dispatched
        events = _stream(hw, _state(None), policy_provider="mock", policy_config={"seed": 3})
        reason = str([e for e in events if isinstance(e, ToolInterruptEvent)][0].interrupts[0].reason)
        assert "checkpoint" not in reason, reason


class TestRefusalsPrecedeTheGateAndTheDispatcher:
    @pytest.mark.parametrize(
        "bag",
        ['{"pretrained_name_or_path": "x"}', ["pretrained_name_or_path", "x"], 7],
        ids=["json-string", "list", "int"],
    )
    def test_a_bag_of_the_wrong_shape_is_refused_by_name(self, dispatched: Any, bag: Any) -> None:
        hw, calls = dispatched
        events = _stream(hw, _state("y"), policy_provider="lerobot_local", policy_config=bag)
        text = _text(events)
        assert "policy_config" in text, text
        assert calls == []
        assert not any(isinstance(e, ToolInterruptEvent) for e in events), "the operator was asked about a doomed call"

    @pytest.mark.parametrize("key", ["host", "port"])
    def test_host_or_port_inside_the_bag_is_refused(self, dispatched: Any, key: str) -> None:
        """The prompt describes the server from policy_host/policy_port; the bag may not override it silently."""
        hw, calls = dispatched
        events = _stream(hw, _state("y"), policy_provider="groot", policy_port=5555, policy_config={key: "other"})
        text = _text(events)
        assert key in text and "policy_host" in text, text
        assert calls == []

    def test_a_checkpoint_provider_with_no_checkpoint_is_refused_before_the_arm(self, dispatched: Any) -> None:
        """``lerobot_local`` declares ``requires: [pretrained_name_or_path]``; the tool now reads it."""
        hw, calls = dispatched
        events = _stream(hw, _state("y"), policy_provider="lerobot_local", policy_config={"device": "cpu"})
        text = _text(events)
        assert "pretrained_name_or_path" in text, text
        assert calls == []
        assert not any(isinstance(e, ToolInterruptEvent) for e in events)
