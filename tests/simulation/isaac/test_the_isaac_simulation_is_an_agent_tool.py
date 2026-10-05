"""``Robot(..., backend="isaac")`` is a Strands ``AgentTool`` an agent can actually drive.

Before: ``Agent(tools=[sim])`` logged "unrecognized tool specification" and the
agent got zero tools - ``IsaacSimulation`` had no ``tool_spec`` and no
``stream``. Measured on one L40S (Isaac Sim 6.1) with a real Bedrock agent:
``registered_tools == []`` on ``main``; with this change the agent registered
``so101_sim`` and ran add_object, reset, step, add_camera, render and
get_body_state through ``sim.run_agent(agent, prompt)`` in 33 s.

Unit-level: the engine is built from its real constructor with no Kit.
"""

from __future__ import annotations

import asyncio
import threading
import time
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands.tools.registry import ToolRegistry  # noqa: E402
from strands.tools.tools import AgentTool  # noqa: E402

from strands_robots.simulation.isaac.agent_tool import _shared_schema  # noqa: E402
from strands_robots.simulation.isaac.simulation import IsaacConfig, IsaacSimulation, _RobotState  # noqa: E402
from tests.simulation._isaac_engine import isaac_engine  # noqa: E402


def _engine(**kwargs: Any) -> Any:
    engine: Any = isaac_engine(IsaacConfig(headless=True))
    for k, v in kwargs.items():
        setattr(engine, k, v)
    return engine


def _text(result: dict[str, Any]) -> str:
    return " ".join(b.get("text", "") for b in result.get("content", []))


class TestItIsATool:
    def test_it_is_an_agent_tool_and_the_registry_takes_it(self) -> None:
        engine = _engine()
        engine.tool_name = "so101_sim"
        assert isinstance(engine, AgentTool)
        registry = ToolRegistry()
        registry.process_tools([engine])
        assert "so101_sim" in registry.registry

    def test_the_spec_publishes_only_what_this_backend_implements(self) -> None:
        spec = _engine().tool_spec
        enum = spec["inputSchema"]["json"]["properties"]["action"]["enum"]
        assert enum and all(callable(getattr(IsaacSimulation, a, None)) for a in enum)
        shared = _shared_schema()["properties"]["action"]["enum"]
        missing = [a for a in shared if not callable(getattr(IsaacSimulation, a, None))]
        assert missing and not set(missing) & set(enum)
        assert "run_pump_forever" not in enum and "run_agent" not in enum

    def test_the_description_names_the_robots_and_their_joints(self) -> None:
        engine = _engine(_world_created=True)
        engine._robots = {"so101": _RobotState(name="so101", prim_path="/World/Robots/so101", joint_names=["1", "2"])}
        assert "'so101' (joints ['1', '2'])" in engine.tool_spec["description"]

    def test_the_description_names_the_objects_and_cameras_already_added(self) -> None:
        engine = _engine(_world_created=True)
        engine._robots = {"so101": _RobotState(name="so101", prim_path="/World/Robots/so101", joint_names=["1"])}
        engine._objects, engine._cameras = {"red_cube": object()}, {"default": object(), "front": object()}
        assert "holds 1 object(s) 'red_cube' and 1 camera(s) 'front';" in engine.tool_spec["description"]


class TestItDispatches:
    def test_an_action_reaches_its_method_with_the_arguments_it_takes(self) -> None:
        engine = _engine()
        seen: dict[str, Any] = {}

        def fake_step(n_steps: int = 1) -> dict[str, Any]:
            seen["n"] = n_steps
            return {"status": "success", "content": [{"text": "stepped"}]}

        engine.step = fake_step
        result = engine(action="step", n_steps=3, camera_name="unused")
        assert seen == {"n": 3} and result["status"] == "success"
        assert "ignored parameters Isaac step() does not take: ['camera_name']" in _text(result)

    def test_an_unimplemented_action_says_so(self) -> None:
        text = _text(_engine()(action="save_state"))
        assert "not implemented on the Isaac backend" in text

    def test_an_unknown_action_lists_the_available_ones(self) -> None:
        text = _text(_engine()(action="teleport"))
        assert "Unknown action 'teleport'" in text and "get_body_state" in text

    def test_stream_yields_one_result_with_the_tool_use_id(self) -> None:
        engine = _engine()
        engine.step = lambda n_steps=1: {"status": "success", "content": [{"text": "ok"}]}

        async def run() -> list[Any]:
            return [e async for e in engine.stream({"toolUseId": "t1", "input": {"action": "step"}}, {})]

        events = asyncio.run(run())
        assert len(events) == 1
        assert events[0]["tool_result"]["toolUseId"] == "t1" and events[0]["tool_result"]["status"] == "success"


class TestItRunsOnTheKitThread:
    def test_a_worker_call_without_a_pump_is_refused_with_the_recipe(self) -> None:
        engine = _engine()
        engine.step = lambda n_steps=1: {"status": "success", "content": [{"text": "ran"}]}
        box: dict[str, Any] = {}
        t = threading.Thread(target=lambda: box.update(r=engine(action="step")))
        t.start()
        t.join()
        assert box["r"]["status"] == "error" and "sim.run_agent(agent, prompt)" in _text(box["r"])

    def test_run_agent_serves_the_agents_calls_on_this_thread(self) -> None:
        engine = _engine(_world_created=True)
        engine._world = types.SimpleNamespace()
        engine.pump = lambda render=True: None
        main = threading.get_ident()
        ran_on: list[int] = []

        def fake_step(n_steps: int = 1) -> dict[str, Any]:
            ran_on.append(threading.get_ident())
            return {"status": "success", "content": [{"text": "ok"}]}

        engine.step = fake_step

        def agent(prompt: str) -> str:
            assert threading.get_ident() != main
            return _text(engine(action="step"))

        assert engine.run_agent(agent, "go") == "ok"
        assert ran_on == [main]

    def test_run_agent_reraises_the_agents_exception(self) -> None:
        engine = _engine(_world_created=True)
        engine._world = types.SimpleNamespace()
        engine.pump = lambda render=True: None

        def agent(prompt: str) -> str:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            engine.run_agent(agent, "go")


class TestThePumpPicksUpAWorkerCallPromptly:
    def test_a_marshalled_call_waits_a_slice_not_the_whole_idle_period(self) -> None:
        engine = _engine(_world_created=True)
        engine._world = types.SimpleNamespace()
        engine.pump = lambda render=True: None
        stop = threading.Event()
        latencies: list[float] = []

        def worker() -> None:
            time.sleep(0.2)
            for _ in range(20):
                t0 = time.perf_counter()
                engine.run_on_main(lambda: None)
                latencies.append(time.perf_counter() - t0)
            stop.set()

        threading.Thread(target=worker).start()
        engine.run_pump_forever(stop_event=stop)
        latencies.sort()
        assert latencies[len(latencies) // 2] < 0.02, latencies  # was ~0.05-0.08 s
