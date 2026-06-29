"""Multi-episode rollouts reuse a single policy instance.

``run_policy(n_episodes=N)`` and ``eval_policy(n_episodes=N)`` build the policy
ONCE and drive every episode with that same instance. The brittle alternative -
constructing a fresh policy per episode - is invisible in the result text yet
catastrophic for heavyweight checkpoints (e.g. a vision-language-action policy
that loads gigabytes of weight shards from the HuggingFace cache pays that cost
on every episode, turning an N-episode eval into N model reloads for zero
functional gain, and wipes the warm action buffer at each boundary so the arm
appears to restart mid-task).

These tests pin the instantiation count via a counting provider so a regression
that moves ``create_policy`` back into the per-episode loop fails loudly instead
of silently degrading wall-clock and motion quality.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.base import Policy
from strands_robots.policies.factory import register_policy
from strands_robots.simulation.mujoco.simulation import Simulation


class _CountingChunkPolicy(Policy):
    """Chunk-emitting policy that counts its own instantiations and inferences.

    Counters live on the class so a fresh instance per episode is detectable
    from the test even though the rollout discards the instance reference.
    """

    init_count = 0
    infer_count = 0

    def __init__(self, chunk: int = 8, **_kwargs) -> None:
        type(self).init_count += 1
        self._keys: list[str] = []
        self._chunk = chunk
        self._step = 0

    @classmethod
    def reset_counters(cls) -> None:
        cls.init_count = 0
        cls.infer_count = 0

    @property
    def provider_name(self) -> str:
        return "counting_reuse"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = robot_state_keys

    async def get_actions(self, observation_dict, instruction, **_kwargs):
        type(self).infer_count += 1
        if not self._keys:
            self._keys = [f"joint_{i}" for i in range(6)]
        chunk = []
        for i in range(self._chunk):
            chunk.append({k: 0.01 * ((self._step + i) % 10) for k in self._keys})
        self._step += self._chunk
        return chunk


@pytest.fixture(autouse=True)
def _register_counting_provider():
    register_policy("counting_reuse", loader=lambda: _CountingChunkPolicy)
    _CountingChunkPolicy.reset_counters()
    yield
    _CountingChunkPolicy.reset_counters()


@pytest.fixture
def sim():
    s = Simulation(tool_name="multi_episode_reuse", mesh=False)
    s.create_world()
    s.add_robot(name="so100", data_config="so100")
    yield s
    s.cleanup()


def _payload(result: dict) -> dict:
    for blk in result.get("content", []):
        if isinstance(blk, dict) and isinstance(blk.get("json"), dict):
            return blk["json"]
    raise AssertionError(f"no json content block in {result}")


def test_run_policy_multi_episode_instantiates_policy_once(sim):
    result = sim.run_policy(
        robot_name="so100",
        policy_provider="counting_reuse",
        n_episodes=3,
        n_steps=24,
        control_frequency=50.0,
        action_horizon=8,
    )
    assert result["status"] == "success", result
    payload = _payload(result)
    assert payload["n_episodes_completed"] == 3
    # The whole point: one construction, reused across every episode.
    assert _CountingChunkPolicy.init_count == 1
    # Action buffer is preserved across chunk boundaries within an episode:
    # 24 steps / horizon 8 = 3 chunks per episode, never one query per step.
    assert _CountingChunkPolicy.infer_count == 3 * 3


def test_eval_policy_multi_episode_instantiates_policy_once(sim):
    result = sim.eval_policy(
        robot_name="so100",
        policy_provider="counting_reuse",
        n_episodes=3,
        max_steps=16,
        control_frequency=50.0,
        action_horizon=8,
    )
    assert result["status"] == "success", result
    assert _CountingChunkPolicy.init_count == 1
