"""Any ``lerobot-policy-server``, spoken through LeRobot's own client, engine and ``dispatch_tick``."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ...types import Action, Observation
from ...units import Units
from ..base import PolicyFault
from ..requirements import Requirements
from .contract import Contract, lerobot

if TYPE_CHECKING:
    from ...description import Description


class ZenohPolicy:
    """``Policy("zenoh://host:7447/<deployment>")``: LeRobot remote inference as a strands policy.

    ``attach`` discovers the deployment, reads its features against the robot, refuses a missing joint,
    camera or unknown semantics BEFORE asking for a session, then admits and starts LeRobot's engine.
    ``act`` publishes the frame and runs one LeRobot motor tick; a held/starved tick returns the last
    applied action; a terminal fault raises ``PolicyFault`` with LeRobot's reason. Chunk behaviour uses
    LeRobot's settings unchanged (``mode``, ``chunk_merge``, ``refill_seconds``, ``blend_*`` ...).
    """

    realtime = True  # LeRobot ages observations and paces chunks on time.monotonic()
    requires_position_hold = True  # LeRobot admits only a robot that holds between motor ticks

    def __init__(self, *, endpoint: str, deployment: str, **settings: Any) -> None:
        self.endpoint, self.deployment, self.settings = endpoint, deployment, settings
        self.contract: Contract | None = None
        self.hz: float | None = None
        self._engine: Any = None
        self._client: Any = None
        self._last: np.ndarray | None = None
        self._task: str | None = None

    def describe(self) -> dict[str, Any]:
        """The server's descriptor (checkpoint identity, features, fps, modes, semantics), without a session."""
        client = self._connect()
        try:
            return dict(client.descriptor)
        finally:
            client.close()

    def _config(self, semantics: str = "?") -> Any:
        from lerobot.inference import RemoteInferenceConfig

        blend = self.settings.get("blend_components")
        kw = dict(self.settings, blend_components=[f"{k}.pos" if "." not in k else k for k in blend or []])
        return RemoteInferenceConfig(endpoint=self.endpoint, deployment=self.deployment, semantics=semantics, **kw)

    def _connect(self) -> Any:
        lerobot()
        from lerobot.remote_inference import RemoteClient

        return RemoteClient.connect(self._config())

    def attach(self, robot: Any) -> None:
        self.detach()
        client = self._connect()
        try:
            caps, sem = client.capabilities, client.descriptor["semantics"]
            contract = Contract(robot.description, tuple(caps.features), caps.action_feature, sem)
            contract.check()
            contract.units()  # unknown semantics -> refused by name, before a session exists
            contract.requirements().check(robot.observe(), who=repr(self))
            config = self._config(sem)
            client.config = config
            client.admit(
                features=caps.features,
                action_feature=caps.action_feature,
                semantics=sem,
                action_interval=caps.action_interval,
                mode=config.mode,
            )
        except BaseException:
            client.close()
            raise
        from lerobot.remote_inference import RemoteInferenceEngine
        from lerobot.utils.action_interpolator import ActionInterpolator

        self.contract, self._client, self.hz = contract, client, 1.0 / caps.action_interval
        self._engine = RemoteInferenceEngine(client, config, task="")
        self._interp = ActionInterpolator(multiplier=1)
        self._engine.start()
        self._engine.resume()

    def detach(self) -> None:
        if self._engine is not None:
            self._engine.stop()
        elif self._client is not None:
            self._client.close()
        self._engine = self._client = None
        self._last = self._task = None  # a new session starts with the engine's empty task

    def _need(self) -> Contract:
        if self.contract is None:
            raise RuntimeError(f"{self!r} is not attached to a robot; run it with robot.run(policy, ...)")
        return self.contract

    def units_for(self, description: Description) -> Units | None:
        return self._need().units()

    def requirements(self) -> Requirements:
        return self.contract.requirements() if self.contract is not None else Requirements()

    def reset(self) -> None:
        if self._engine is not None and self._last is not None:  # a fresh session needs no reset
            self._engine.reset()
            self._engine.resume()
        self._last = None

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        from lerobot.inference import dispatch_tick

        c, engine = self._need(), self._engine
        if instruction is not None and instruction != self._task:
            if error := engine.text_input_error(instruction, instruction=True):
                raise ValueError(f"{self!r} refused the instruction: {error}")
            engine.set_task(instruction)
            self._task = instruction
        engine.submit_observation(c.frame(obs), capture_time=time.monotonic())
        sent: list[Any] = []
        target = dispatch_tick(engine, self._interp, send=sent.append, hold=lambda: None)
        if engine.failed:
            raise PolicyFault(f"{self!r}: {engine.runtime.failure or engine.failure_traceback or 'LeRobot fault'}")
        if target is not None:
            self._last = c.to_keys(target.detach().cpu().numpy())
        if self._last is None:  # startup: hold where the body is (already in the policy's units)
            self._last = np.asarray(obs.joints.positions, dtype=float).copy()
        return self._last

    def __repr__(self) -> str:
        return f"Policy('zenoh://{self.endpoint.removeprefix('tcp/')}/{self.deployment}')"
