"""The one run loop: observe -> policy -> act. ``Robot.run``, ``Robot.start`` and ``Simulation.run`` call it."""

from __future__ import annotations

import threading
import time as _time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ...policy.base import PolicyFault
from ...types import Observation
from .bind import Bound, bind, decide, release
from .result import RunResult
from .task import Task

if TYPE_CHECKING:
    from ...policy import PolicyLike
    from ...simulation import Simulation
    from ..robot import Robot

__all__ = ["run", "start", "lockstep", "Task", "RunResult"]


def run(
    robot: Robot,
    policy: PolicyLike | Callable[[Observation], Any],
    *,
    instruction: str | None = None,
    hz: float | None = None,
    seconds: float | None = None,
    steps: int | None = None,
    stop_when: Callable[[Observation], bool] | None = None,
    units: str | None = "auto",
    realtime: bool = False,
    stop_event: threading.Event | None = None,
) -> RunResult:
    """observe -> policy -> act at ``hz`` (default: the policy's ``hz``, else 30) until a limit or ``stop()``."""
    if seconds is None and steps is None and stop_when is None:
        raise ValueError("run needs seconds=, steps= or stop_when=")
    if hz is not None and hz <= 0:
        raise ValueError(f"hz must be > 0, got {hz}")  # before bind: a bad hz loads no checkpoint
    bound = bind(robot, policy, units)
    try:
        return _loop(robot, bound, instruction, hz, seconds, steps, stop_when, realtime, stop_event)
    finally:
        release(bound)


def _rate(bound: Bound, hz: float | None) -> float:
    rate = hz if hz is not None else (getattr(bound.policy, "hz", None) or 30.0)
    if rate <= 0:
        raise ValueError(f"hz must be > 0, got {rate}")
    return float(rate)


def _loop(
    robot: Robot,
    bound: Bound,
    instruction: str | None,
    hz: float | None,
    seconds: float | None,
    steps: int | None,
    stop_when: Callable[[Observation], bool] | None,
    realtime: bool,
    stop_event: threading.Event | None,
) -> RunResult:
    hz = _rate(bound, hz)
    realtime = realtime or bool(getattr(bound.policy, "realtime", False))
    dt = 1.0 / hz
    sim_steps = max(1, round(dt / robot.sim.physics.timestep)) if robot.sim is not None else None
    max_steps = steps if steps is not None else (int(round(seconds * hz)) if seconds is not None else None)
    n = 0
    why: str = "steps"
    fault: str | None = None
    wall0 = _time.perf_counter()
    next_tick = wall0
    while True:
        if stop_event is not None and stop_event.is_set():
            why = "stopped"
            break
        if max_steps is not None and n >= max_steps:
            break
        obs = robot.observe()
        if stop_when is not None and stop_when(obs):
            why = "stop_when"
            break
        try:
            action = decide(robot, bound, obs, instruction)
        except PolicyFault as e:
            why, fault = "fault", str(e)
            break
        robot.act(action, steps=sim_steps)
        n += 1
        if robot.sim is None or realtime:  # hardware (or a watched sim): pace in wall time
            next_tick += dt
            sleep = next_tick - _time.perf_counter()
            if sleep > 0:
                _time.sleep(sleep)
    wall = _time.perf_counter() - wall0
    elapsed = n * dt if robot.sim is not None else wall
    final = robot.observe(images=False)
    return RunResult(n, elapsed, hz, (n / wall) if wall else 0.0, why, None, final, fault)  # type: ignore[arg-type]


def start(robot: Robot, policy: PolicyLike | Callable[[Observation], Any], **kw: Any) -> Task:
    """``run`` in a background thread; ``Task.stop()`` / ``.status`` / ``.result``."""
    if robot._task is not None and robot._task.running:
        raise RuntimeError(f"robot {robot.id!r} already has a running task; call stop() first")
    robot._task = Task(lambda ev: run(robot, policy, stop_event=ev, **kw))
    return robot._task


def lockstep(
    sim: Simulation,
    policies: Mapping[Robot, PolicyLike | Callable[[Observation], Any]],
    *,
    hz: float | None = None,
    seconds: float | None = None,
    steps: int | None = None,
    instruction: str | None = None,
    units: str | None = "auto",
) -> dict[str, RunResult]:
    """Drive N robots in lock-step: every robot observes, every policy acts, the world steps once."""
    if seconds is None and steps is None:
        raise ValueError("Simulation.run needs seconds= or steps=")
    for r in policies:
        if r.sim is not sim:
            raise ValueError(f"robot {r.id!r} is not in this world")
    bound: dict[Robot, Bound] = {}
    try:
        for r, p in policies.items():
            bound[r] = bind(r, p, units)
        return _lockstep(sim, bound, hz, seconds, steps, instruction)
    finally:
        for b in bound.values():
            release(b)


def _lockstep(
    sim: Simulation,
    bound: Mapping[Robot, Bound],
    hz: float | None,
    seconds: float | None,
    steps: int | None,
    instruction: str | None,
) -> dict[str, RunResult]:
    dt = sim.physics.control_dt if hz is None else 1.0 / hz
    per = max(1, round(dt / sim.physics.timestep))
    total = steps if steps is not None else int(round(float(seconds or 0.0) / dt))
    realtime = any(getattr(b.policy, "realtime", False) for b in bound.values())
    faults: dict[str, str] = {}
    t0, n, next_tick = sim.time, 0, _time.perf_counter()
    with sim.manual_time():
        for _ in range(total):
            for r, b in bound.items():
                try:
                    r.act(decide(r, b, r.observe(), instruction))
                except PolicyFault as e:
                    faults[r.id] = str(e)
            if faults:
                break
            sim.step(per)
            n += 1
            if realtime:
                next_tick += dt
                time_left = next_tick - _time.perf_counter()
                if time_left > 0:
                    _time.sleep(time_left)
    el = sim.time - t0
    why = "fault" if faults else "steps"
    return {
        r.id: RunResult(n, el, 1.0 / dt, n / el if el else 0.0, why, None, r.observe(), faults.get(r.id))  # type: ignore[arg-type]
        for r in bound
    }
