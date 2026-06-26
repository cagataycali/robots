"""The SIM re-queries an RTC policy at its ``execution_horizon``, not its chunk.

Regression for the policy/sim contract collision: the runner used to consume
``max(action_horizon, policy.actions_per_step)`` actions before re-querying. For
a Real-Time-Chunking policy trained on a long chunk (e.g. 50) but with a short
execution horizon (e.g. 10), that drained the entire chunk before re-querying,
so the policy never received its previous chunk's unexecuted tail and its
cross-chunk blending was dead code - identical output to plain open-loop replay.

The fix routes every consumer through ``policy.execution_horizon`` (via
``resolve_chunk_length``). These tests pin the consumer behaviour - the runner
re-queries every ``execution_horizon`` steps and a user-supplied
``action_horizon`` cannot stretch that interval - without a real VLA checkpoint,
using an instrumented counting sim and a weight-free chunk policy.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner


class _CountingSim(SimEngine):
    """Minimal ``SimEngine`` that counts ``send_action`` calls."""

    def __init__(self) -> None:
        self._joint_names = ["j0", "j1", "j2"]
        self._robots = {"arm": self._joint_names}
        self._lock = threading.Lock()
        self.send_count = 0

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        return {"status": "success"}

    def get_state(self):
        return {"sim_time": 0.0, "step_count": self.send_count}

    def add_robot(self, name, **kw):
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._robots.get(robot_name, []))

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {n: 0.0 for n in self._joint_names}

    def send_action(self, action, robot_name=None, n_substeps=1):
        with self._lock:
            self.send_count += 1

    def render(self, camera_name="default", width=None, height=None):
        return {"image": np.zeros((height or 48, width or 64, 3), dtype=np.uint8)}


class _RtcChunkPolicy(Policy):
    """Weight-free RTC stand-in: long trained chunk, short execution horizon."""

    requires_images = False

    def __init__(self, actions_per_step: int = 50, execution_horizon: int = 10) -> None:
        self.actions_per_step = actions_per_step
        self.supports_rtc = True
        self._execution_horizon = execution_horizon
        self.robot_state_keys: list[str] = []
        self.query_steps: list[int] = []
        self._sim: _CountingSim | None = None

    @property
    def provider_name(self) -> str:
        return "rtc-test"

    @property
    def execution_horizon(self) -> int:
        return self._execution_horizon

    def bind(self, sim: _CountingSim) -> None:
        self._sim = sim

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = robot_state_keys

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        assert self._sim is not None
        self.query_steps.append(self._sim.send_count)
        keys = self.robot_state_keys or ["j0", "j1", "j2"]
        return [{k: 0.0 for k in keys} for _ in range(self.actions_per_step)]


def _run(policy: _RtcChunkPolicy, *, n_steps: int, action_horizon: int) -> dict:
    sim = _CountingSim()
    policy.bind(sim)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    return PolicyRunner(sim).run(
        "arm",
        policy,
        duration=n_steps / 50.0,
        control_frequency=50.0,
        action_horizon=action_horizon,
        fast_mode=True,
    )


def test_rtc_policy_requeried_every_execution_horizon() -> None:
    """Re-query interval == execution_horizon (10), not the 50-step chunk."""
    policy = _RtcChunkPolicy(actions_per_step=50, execution_horizon=10)
    result = _run(policy, n_steps=30, action_horizon=8)
    assert result["status"] == "success"
    # Queries fired at sim steps 0, 10, 20 -> three inferences over 30 steps.
    assert policy.query_steps == [0, 10, 20], policy.query_steps


def test_user_action_horizon_cannot_stretch_rtc_interval() -> None:
    """A huge action_horizon must not collapse RTC back to one big chunk."""
    policy = _RtcChunkPolicy(actions_per_step=50, execution_horizon=10)
    result = _run(policy, n_steps=30, action_horizon=999)
    assert result["status"] == "success"
    assert policy.query_steps == [0, 10, 20], policy.query_steps
