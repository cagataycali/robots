"""A Robot is a description plus a Body: observe() / act() / run().

The Body is the only thing that differs between a simulated robot (``SimBody``,
a handle into a ``Simulation``) and a real one (a driver on a bus, which lands
with the drivers). Every loop - policy runs, recording, tools, transports - is
written once against ``Robot`` and never knows which Body it has.
"""

from __future__ import annotations

import threading
import time as _time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from .description import Description
from .types import Action, JointState, Observation, Pose, RunResult

if TYPE_CHECKING:
    from .policy import PolicyLike
    from .simulation import Simulation


@runtime_checkable
class Body(Protocol):
    """Something with joints you can read and targets you can write."""

    names: tuple[str, ...]

    def read(self) -> JointState: ...

    def write(self, targets: np.ndarray) -> None: ...

    def pose(self) -> Pose | None: ...

    def images(self) -> Mapping[str, np.ndarray]: ...

    def extras(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class SimBody:
    """A handle into a Simulation: the body named ``id`` in ``sim``."""

    def __init__(self, sim: Simulation, id: str, names: tuple[str, ...]) -> None:
        self.sim = sim
        self.id = id
        self.names = names

    def read(self) -> JointState:
        return self.sim.backend.joint_state(self.id)

    def write(self, targets: np.ndarray) -> None:
        self.sim.backend.set_targets(self.id, targets)

    def pose(self) -> Pose | None:
        return self.sim.backend.body_pose(self.id)

    def images(self) -> Mapping[str, np.ndarray]:
        return {c.name: self.sim.backend.render(c.name) for c in self.sim.scene.cameras if c.attach == self.id}

    def extras(self) -> Mapping[str, Any]:
        return {}

    def close(self) -> None:
        pass


class _NoLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


_NOLOCK = _NoLock()


class Task:
    """A background ``Robot.run``: ``stop()``, ``status``, ``result``."""

    def __init__(self, target: Callable[[threading.Event], RunResult]) -> None:
        self._stop = threading.Event()
        self._result: RunResult | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._go, args=(target,), daemon=True)
        self._thread.start()

    def _go(self, target: Callable[[threading.Event], RunResult]) -> None:
        try:
            self._result = target(self._stop)
        except BaseException as exc:  # noqa: BLE001 - surfaced via .result
            self._error = exc

    def stop(self, timeout: float = 5.0) -> RunResult | None:
        self._stop.set()
        self._thread.join(timeout)
        return self.result

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    @property
    def status(self) -> str:
        if self.running:
            return "running"
        return "error" if self._error else "done"

    @property
    def result(self) -> RunResult | None:
        if self._error:
            raise self._error
        return self._result


class Robot:
    """A body in a world (``sim=``, or ``Simulation.default()``): observe, act, run."""

    def __init__(
        self,
        name: str,
        *,
        sim: Simulation | None = None,
        position: Any = None,
        orientation: Any = None,
        keyframe: str | None = None,
        id: str | None = None,
    ) -> None:
        from .simulation import Simulation

        self.description: Description = Description.load(name)
        self.name = self.description.name
        self.is_real = False
        self.twin: Robot | None = None  # on a hardware Robot: the simulated body that mirrors it
        self.twin_of: Robot | None = None  # on that twin: the hardware Robot it mirrors
        self._task: Task | None = None
        world = sim if sim is not None else Simulation.default()
        self.id = world._unique_id(id or self.name)
        self.sim: Simulation | None = world
        world.add(self, position=position, orientation=orientation, keyframe=keyframe)
        self._body: Body | None = SimBody(world, self.id, self.description.action_keys)

    # identity

    @property
    def body(self) -> Body:
        if self._body is None:
            raise RuntimeError(f"robot {self.id!r} has been removed from its world")
        return self._body

    def _detached(self) -> None:
        self._body = None
        self.sim = None

    @property
    def action_keys(self) -> tuple[str, ...]:
        return self.description.action_keys

    # placement

    @property
    def pose(self) -> Pose:
        p = self.body.pose()
        if p is None:
            raise RuntimeError(f"{self.id}: this body does not report a base pose")
        return p

    def set_pose(self, position: Any = None, orientation: Any = None) -> None:
        """Teleport (sim only). A hardware robot raises; its twin can still be teleported."""
        if self.sim is None:
            raise NotImplementedError(f"{self.id} is not in a world: only a simulated body can be teleported")
        self.sim.set_pose(self, position=position, orientation=orientation)

    def set_gains(self, kp: Any, kd: Any) -> None:
        """PD gains of this body's actuators, ``action_keys`` order (sim only; a servo bus keeps its own)."""
        if self.sim is None:
            raise NotImplementedError(f"{self.id} is not in a world: its servo gains live in the firmware")
        kp, kd = np.asarray(kp, dtype=float).reshape(-1), np.asarray(kd, dtype=float).reshape(-1)
        n = len(self.description.action_keys)
        if kp.shape != (n,) or kd.shape != (n,):
            raise ValueError(f"{self.id}: kp {kp.shape} / kd {kd.shape} for {n} actuators {list(self.action_keys)}")
        with self.sim.lock:
            self.sim.backend.set_gains(self.id, kp, kd)

    # the verbs

    def observe(self, *, images: bool = True) -> Observation:
        body = self.body
        with self.sim.lock if self.sim is not None else _NOLOCK:
            js = body.read()
            obs = Observation(
                t=self.sim.time if self.sim is not None else _time.time(),
                joints=js,
                pose=body.pose(),
                images=body.images() if images else {},
                extras=body.extras(),
            )
        if self.twin is not None and self.twin.sim is not None:  # the twin shows what the real body measured
            with self.twin.sim.lock:
                self.twin.body.write(self._full_targets(js.positions))
        return obs

    def gate(self, action: Action, *, confirmed: bool = False) -> np.ndarray:
        """The operator gate for commands that did not come from the developer's own code (agent tool, dashboard).

        A simulated body passes. A real body passes only when an operator already said yes (``confirmed=True``);
        otherwise it refuses with ``PermissionError``: a real body never moves because of a default.
        """
        targets = self.coerce(action)
        if not self.is_real or confirmed:
            return targets
        raise PermissionError(f"{self.id} is a real robot and this command came through an agent or dashboard")

    def confirm_act(self, action: Action, *, steps: int | None = None, confirmed: bool = False) -> Observation:
        """``act`` through the operator gate: what an agent tool and the dashboard call."""
        return self.act(self.gate(action, confirmed=confirmed), steps=steps)

    def act(self, action: Action, *, steps: int | None = None) -> Observation:
        """Write targets for THIS body and advance its world one control step (unless manual time)."""
        targets = self.coerce(action)
        with self.sim.lock if self.sim is not None else _NOLOCK:
            self.body.write(targets)
            if self.sim is not None:
                if steps is None:
                    self.sim.control_step()
                else:
                    self.sim.step(steps)
        if self.twin is not None and self.twin.sim is not None:
            with self.twin.sim.lock:
                self.twin.body.write(targets)
        return self.observe(images=False)

    def coerce(self, action: Action) -> np.ndarray:
        """Any accepted Action -> a full target vector in ``action_keys`` order, clipped to limits."""
        keys = self.description.action_keys
        if isinstance(action, JointState):
            action = action.as_dict()
        if isinstance(action, Mapping):
            unknown = [k for k in action if k not in keys]
            if unknown:
                raise KeyError(f"unknown joint(s) {unknown} for {self.id}; action_keys are {list(keys)}")
            targets = np.array(self.body.read().positions, dtype=float)
            for k, v in action.items():
                targets[keys.index(k)] = float(v)
        else:
            targets = np.asarray(action, dtype=float).reshape(-1)
            if targets.shape != (len(keys),):
                raise ValueError(
                    f"action has {targets.size} values; {self.id} has {len(keys)} action keys {list(keys)}"
                )
        return self._full_targets(targets)

    def _full_targets(self, targets: np.ndarray) -> np.ndarray:
        out = np.array(targets, dtype=float)
        for i, k in enumerate(self.description.action_keys):
            lim = self.description.limits.get(k)
            if lim is not None:
                out[i] = min(max(out[i], lim[0]), lim[1])
        return out

    # the loop, written once

    def run(
        self,
        policy: PolicyLike | Callable[[Observation], Any],
        *,
        instruction: str | None = None,
        hz: float = 30.0,
        seconds: float | None = None,
        steps: int | None = None,
        stop_when: Callable[[Observation], bool] | None = None,
        units: str | None = "auto",
        realtime: bool = False,
        _stop: threading.Event | None = None,
    ) -> RunResult:
        """observe -> policy -> act at ``hz`` until ``seconds``/``steps``/``stop_when``/``stop()``.

        In a world, ``hz`` sets how many physics steps each act() advances (``realtime=True`` also paces the loop to
        the wall clock, for a human watching); on hardware it always paces wall time.
        ``units="auto"`` converts joints between the robot's radians and the units the policy declares
        (``policy.units``); ``"rad"`` disables, a name (``"lerobot_so10x"``) forces. Before the first inference
        ``policy.requirements()`` is checked against a real observation, so a missing camera is refused by name.
        """
        from .policy import as_policy
        from .units import resolve_units

        if seconds is None and steps is None and stop_when is None:
            raise ValueError("run needs seconds=, steps= or stop_when=")
        if hz <= 0:
            raise ValueError(f"hz must be > 0, got {hz}")
        pol = as_policy(policy)
        gains = getattr(pol, "gains", None)
        if gains is not None and self.sim is not None:  # the policy's PD is part of the policy, not of the world
            self.set_gains(*gains)
        conv = resolve_units(pol, units, self.description)
        req = getattr(pol, "requirements", None)
        if req is not None:
            req().check(self.observe(), who=repr(pol))
        pol.reset()
        dt = 1.0 / hz
        sim_steps = max(1, round(dt / self.sim.physics.timestep)) if self.sim is not None else None
        max_steps = steps if steps is not None else (int(round(seconds * hz)) if seconds is not None else None)
        n = 0
        why = "steps"
        wall0 = _time.perf_counter()
        next_tick = wall0
        while True:
            if _stop is not None and _stop.is_set():
                why = "stopped"
                break
            if max_steps is not None and n >= max_steps:
                break
            obs = self.observe()
            if stop_when is not None and stop_when(obs):
                why = "stop_when"
                break
            if conv is not None:
                seen = obs.with_joints(conv.from_rad(obs.joints.positions))
                action = self.coerce(conv.to_rad(np.asarray(pol.act(seen, instruction=instruction), dtype=float)))
            else:
                action = self.coerce(pol.act(obs, instruction=instruction))
            self.act(action, steps=sim_steps)
            n += 1
            if self.sim is None or realtime:  # hardware (or a watched sim): pace in wall time
                next_tick += dt
                sleep = next_tick - _time.perf_counter()
                if sleep > 0:
                    _time.sleep(sleep)
        wall = _time.perf_counter() - wall0
        elapsed = n * dt if self.sim is not None else wall
        return RunResult(n, elapsed, hz, (n / wall) if wall else 0.0, why, None, self.observe(images=False))

    def start(self, policy: PolicyLike | Callable[[Observation], Any], **kw: Any) -> Task:
        """``run`` in a background thread; ``Task.stop()`` / ``.status`` / ``.result``."""
        if self._task is not None and self._task.running:
            raise RuntimeError(f"robot {self.id!r} already has a running task; call stop() first")
        self._task = Task(lambda ev: self.run(policy, _stop=ev, **kw))
        return self._task

    def stop(self) -> RunResult | None:
        return self._task.stop() if self._task is not None else None

    # agents

    def describe(self) -> dict[str, Any]:
        d = self.description.describe()
        d.update(
            {
                "id": self.id,
                "real": self.is_real,
                "world": None if self.sim is None else self.sim.backend_name,
                "twin": None if self.twin is None else self.twin.id,
                "twin_of": None if self.twin_of is None else self.twin_of.id,
                "task": None if self._task is None else self._task.status,
            }
        )
        return d

    def close(self) -> None:
        self.stop()
        if self.sim is not None:
            self.sim.remove(self)
        elif self._body is not None:
            self._body.close()
            self._body = None
        if self.twin is not None:
            self.twin.close()
            self.twin = None

    def __repr__(self) -> str:
        where = "hardware" if self.is_real else (self.sim.backend_name if self.sim else "detached")
        return f"Robot({self.name!r}, id={self.id!r}, {where}, dof={self.description.dof})"
