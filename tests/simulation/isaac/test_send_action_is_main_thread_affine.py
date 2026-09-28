"""``send_action`` is main-thread affine, like ``step`` - it never steps off the owner.

Isaac's kit runtime only pumps updates on the thread that created ``SimulationApp``.
``step`` and ``reset`` route through :meth:`IsaacSimulation._marshal_main_thread_affine`
so that off the owning thread they refuse (with no pump) or hop onto the pump (with
one). ``send_action`` did not: it took ``self._lock`` once and called
``self._world.step`` directly for every substep on whatever thread called it, so a
worker thread's ``send_action`` blocked forever inside PhysX's native ``step`` (#4082) -
the exact path an agent's ``run_policy`` -> ``PolicyRunner`` -> ``send_action`` takes,
since Strands runs a sync tool through ``asyncio.to_thread``.

This pins the three cases the marshal defines, with a stubbed World that records which
thread each ``step`` ran on (the Kit leaf is stood in, as the sibling isaac unit tests
stand it in - no Isaac Sim runtime):

* on the owning thread the write-and-step runs inline;
* off it with no pump the call is refused with a ``RuntimeError`` naming the recipe,
  and the world is never stepped on the worker (pre-fix it stepped, and wedged);
* off it with ``run_pump_forever`` engaged the step hops onto the owning thread.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.config import IsaacConfig  # noqa: E402
from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacSimulation,
    _RobotState,
)


class _World:
    """Records the thread every ``step`` ran on, so affinity is asserted not inferred."""

    def __init__(self) -> None:
        self.physics_sim_view = object()
        self.step_threads: list[int] = []

    def step(self, render: bool = False) -> None:
        self.step_threads.append(threading.get_ident())

    def reset(self) -> None:
        return None


def _engine(*, main_tid: int, pump_running: bool = False) -> Any:
    """A skeleton engine with a live world and one articulation-less robot.

    ``articulation=None`` so the substep loop - the thing under test - runs without
    reaching the isaacsim ``ArticulationAction`` import, which is absent here.
    """
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world = _World()
    engine._world_created = True
    engine._robots = {}
    engine._cameras = {}
    engine._objects = {}
    engine._action_controllers = {}
    engine._applied_wrenches = {}
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._physics_view_stale = False
    engine._main_tid = main_tid
    engine._pump_running = pump_running
    engine._action_q = queue.Queue()
    engine._main_jobs = queue.Queue()
    engine._idle_converge = 1
    engine._idle_render_period = 3600.0
    # The idle preview would otherwise step the world on the pump thread and
    # pollute the affinity assertions; only send_action's steps are under test.
    engine._converge_render = lambda n: None  # type: ignore[method-assign]
    robot = _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=["j0", "j1"])
    robot.articulation = None
    engine._robots = {"arm": robot}
    return engine


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []))


class TestOnTheOwningThread:
    def test_it_runs_inline_and_steps(self) -> None:
        engine = _engine(main_tid=threading.get_ident())

        result = engine.send_action({"j0": 0.5}, robot_name="arm", n_substeps=3)

        assert result["status"] == "success"
        assert engine._world.step_threads == [threading.get_ident()] * 3
        assert engine._step_count == 3


class TestOffTheOwningThreadWithNoPump:
    def test_it_is_refused_and_never_steps(self) -> None:
        engine = _engine(main_tid=threading.get_ident(), pump_running=False)
        box: dict[str, Any] = {}

        def _worker() -> None:
            try:
                engine.send_action({"j0": 0.5}, robot_name="arm", n_substeps=3)
            except RuntimeError as exc:
                box["exc"] = exc

        worker = threading.Thread(target=_worker)
        worker.start()
        worker.join(timeout=10)

        assert not worker.is_alive(), "send_action wedged the worker instead of refusing"
        assert isinstance(box.get("exc"), RuntimeError)
        assert "run_pump_forever" in str(box["exc"])
        assert engine._world.step_threads == [], "the world was stepped on the worker thread"

    def test_a_bad_argument_is_still_a_dict_before_the_hop(self) -> None:
        """Entry validation runs on the caller, so a domain error is the envelope the
        PolicyRunner counts - not a RuntimeError - even off the owning thread."""
        engine = _engine(main_tid=threading.get_ident(), pump_running=False)
        box: dict[str, Any] = {}

        def _worker() -> None:
            box["r"] = engine.send_action({"j0": 0.5}, robot_name="arm", n_substeps=0)

        worker = threading.Thread(target=_worker)
        worker.start()
        worker.join(timeout=10)

        assert box["r"]["status"] == "error"
        assert "n_substeps" in _text(box["r"])
        assert engine._world.step_threads == []


class TestOffTheOwningThreadWithAPump:
    def test_the_step_hops_onto_the_owning_thread(self) -> None:
        owner_tid = threading.get_ident()
        engine = _engine(main_tid=owner_tid, pump_running=True)
        stop = threading.Event()
        box: dict[str, Any] = {}

        def _worker() -> None:
            try:
                box["r"] = engine.send_action({"j0": 0.5}, robot_name="arm", n_substeps=3)
                box["worker_tid"] = threading.get_ident()
            finally:
                stop.set()

        worker = threading.Thread(target=_worker)
        worker.start()
        # This (owning) thread runs the pump, which drains the marshalled job and
        # executes the write-and-step inline here. It exits when the worker is done.
        engine.run_pump_forever(stop_event=stop)
        worker.join(timeout=10)

        assert not worker.is_alive()
        assert box["r"]["status"] == "success"
        assert box["worker_tid"] != owner_tid, "the test did not actually use a worker thread"
        assert engine._world.step_threads == [owner_tid] * 3, "the step did not hop onto the owner"
        assert engine._step_count == 3
