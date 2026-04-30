"""Tests proving ``PolicyRunner`` is truly backend-agnostic.

The runner must work against any ``SimEngine`` using only public methods
(``get_observation``, ``send_action``, ``step``, ``reset``, ``render``,
``list_robots``, ``robot_joint_names``). These tests use a pure-Python
``FakeSim`` stub — no MuJoCo import, no physics.

If these pass, Isaac / Newton / any new backend gets ``run_policy`` /
``replay`` / ``evaluate`` for free the moment they implement ``SimEngine``
primitives.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import CooperativeStop, PolicyRunner


class FakeSim(SimEngine):
    """Minimal ``SimEngine`` implementation — no physics, records all calls."""

    def __init__(self, joint_names: tuple[str, ...] = ("j0", "j1", "j2")):
        self._joint_names = list(joint_names)
        self.calls: list[tuple] = []
        self._step_count = 0
        self._sim_time = 0.0
        self._robots = {"fake_robot": self._joint_names}

    # --- Implement abstract methods (bare minimum) ---
    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        self.calls.append(("reset",))
        self._step_count = 0
        self._sim_time = 0.0
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        self.calls.append(("step", n_steps))
        self._step_count += n_steps
        self._sim_time += 0.002 * n_steps
        return {"status": "success"}

    def get_state(self):
        return {"sim_time": self._sim_time, "step_count": self._step_count}

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

    def get_observation(self, robot_name=None, camera_name=None):
        self.calls.append(("get_observation", robot_name, camera_name))
        return {n: 0.0 for n in self._joint_names}

    def send_action(self, action, robot_name=None, n_substeps=1):
        self.calls.append(("send_action", dict(action), robot_name))
        self._step_count += 1
        self._sim_time += 0.002

    def render(self, camera_name="default", width=None, height=None):
        self.calls.append(("render", camera_name, width, height))
        return {
            "image": np.zeros((height or 48, width or 64, 3), dtype=np.uint8),
        }


# ---------------------------------------------------------------------------


def test_policy_runner_only_touches_public_api():
    """Fail if PolicyRunner reaches past the SimEngine public surface."""
    sim = FakeSim()
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    result = PolicyRunner(sim).run(
        "fake_robot",
        policy,
        duration=0.1,
        control_frequency=10.0,  # → 1 step total
        fast_mode=True,
    )

    assert result["status"] == "success"
    allowed = {"get_observation", "send_action", "step", "render", "reset"}
    for call in sim.calls:
        assert call[0] in allowed, f"PolicyRunner touched private API: {call}. Only {allowed} are allowed."


def test_policy_runner_import_does_not_pull_in_mujoco():
    """Importing policy_runner must not drag in mujoco."""
    import sys

    # Wipe any existing mujoco imports
    for mod in [m for m in list(sys.modules) if m.startswith("mujoco")]:
        del sys.modules[mod]

    # Force a fresh import of the runner module
    if "strands_robots.simulation.policy_runner" in sys.modules:
        del sys.modules["strands_robots.simulation.policy_runner"]

    import strands_robots.simulation.policy_runner  # noqa: F401

    leaked = [m for m in sys.modules if m.startswith("mujoco")]
    assert not leaked, (
        f"strands_robots.simulation.policy_runner pulled in MuJoCo modules: {leaked}. "
        "The runner must be backend-agnostic."
    )


def test_on_frame_hook_receives_step_obs_action():
    """The on_frame hook is called per step with (idx, observation, action)."""
    captured: list[tuple] = []
    sim = FakeSim()
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    def hook(step: int, obs: dict[str, Any], action: dict[str, Any]) -> None:
        captured.append((step, dict(obs), dict(action)))

    result = PolicyRunner(sim).run(
        "fake_robot",
        policy,
        duration=0.3,
        control_frequency=10.0,  # → 3 steps
        fast_mode=True,
        on_frame=hook,
    )

    assert result["status"] == "success"
    assert len(captured) >= 2
    # Each hook call carries the joint observation and a MockPolicy action
    for step_idx, obs, action in captured:
        assert "j0" in obs
        assert isinstance(action, dict)


def test_cooperative_stop_is_normal_success():
    """Raising ``CooperativeStop`` in the hook returns a success result."""
    sim = FakeSim()
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    def hook(step: int, obs, action) -> None:
        if step >= 2:
            raise CooperativeStop("user stopped")

    result = PolicyRunner(sim).run(
        "fake_robot",
        policy,
        duration=10.0,
        control_frequency=10.0,  # would be 100 steps normally
        fast_mode=True,
        on_frame=hook,
    )
    assert result["status"] == "success"
    assert "stopped" in result["content"][0]["text"].lower()


def test_evaluate_calls_reset_per_episode():
    """evaluate() resets before every episode."""
    sim = FakeSim()
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    result = PolicyRunner(sim).evaluate(
        "fake_robot",
        policy,
        n_episodes=3,
        max_steps=5,
    )
    assert result["status"] == "success"
    # One reset per episode
    reset_calls = [c for c in sim.calls if c[0] == "reset"]
    assert len(reset_calls) == 3


def test_evaluate_success_fn_callable():
    """evaluate() supports arbitrary callable success_fn."""
    sim = FakeSim()
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    # Always succeed
    result = PolicyRunner(sim).evaluate(
        "fake_robot",
        policy,
        n_episodes=2,
        max_steps=10,
        success_fn=lambda obs: True,
    )

    payload = next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)
    assert payload["success_rate"] == 1.0
    assert payload["n_success"] == 2


def test_simengine_run_policy_facade_works_with_fake_sim():
    """The SimEngine.run_policy facade delegates to PolicyRunner correctly."""
    sim = FakeSim()
    # MockPolicy is the default — no policy_config needed.
    result = sim.run_policy(
        "fake_robot",
        policy_provider="mock",
        duration=0.2,
        control_frequency=10.0,
        fast_mode=True,
    )
    assert result["status"] == "success"


def test_simengine_eval_policy_facade_works_with_fake_sim():
    """The SimEngine.eval_policy facade delegates to PolicyRunner correctly."""
    sim = FakeSim()
    result = sim.eval_policy(
        robot_name="fake_robot",
        policy_provider="mock",
        n_episodes=2,
        max_steps=3,
    )
    assert result["status"] == "success"


def test_simengine_run_policy_validates_robot_exists():
    """run_policy returns a friendly error if the robot isn't in the sim."""
    sim = FakeSim()
    result = sim.run_policy(
        "nonexistent_robot",
        policy_provider="mock",
        duration=0.1,
        control_frequency=10.0,
        fast_mode=True,
    )
    assert result["status"] == "error"
    assert "not found" in result["content"][0]["text"].lower()
