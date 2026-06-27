"""Legacy ``PolicyRunner.evaluate(success_fn=...)`` chunk + success-timing.

Regression for the eval-success-rate corruption on the legacy ``success_fn``
path of :meth:`PolicyRunner.evaluate`:

* **Chunk consumption** - the legacy path used to apply only ``actions[0]``
  and re-query the policy every step, silently ignoring ``action_horizon``.
  A chunk-predicting policy must instead have its full chunk consumed
  (``resolve_chunk_length`` actions) before the next ``get_actions`` call,
  identical to ``run()`` and the ``spec=`` benchmark path.
* **Post-action success check** - the legacy path used to evaluate the
  success predicate against the STALE pre-action observation, so success was
  detected one step late and success that lands on the final step was missed
  entirely (under-reporting ``success_rate``). The predicate must run against
  the live post-action observation, matching ``spec.is_success``.
* **Path parity** - the legacy ``success_fn`` path and the ``spec=`` benchmark
  path must agree on ``success_rate`` for a deterministic policy + predicate.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "glfw")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.benchmark import BenchmarkProtocol, StepInfo
from strands_robots.simulation.policy_runner import PolicyRunner
from tests.simulation.test_policy_runner import FakeSim


class _CountingSim(FakeSim):
    """FakeSim whose integer ``pos`` advances by one per applied action.

    ``get_observation`` exposes ``pos`` so a legacy ``success_fn`` and a
    benchmark ``is_success`` can key off the exact same monotone state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pos = 0

    def reset(self):
        self.pos = 0
        return super().reset()

    def send_action(self, action, robot_name=None, n_substeps=1):
        super().send_action(action, robot_name=robot_name, n_substeps=n_substeps)
        self.pos += 1

    def step(self, n_steps: int = 1):
        # Degenerate-policy path advances physics; keep pos in lockstep.
        self.pos += n_steps
        return super().step(n_steps=n_steps)

    def get_observation(self, robot_name=None, *, skip_images=False):
        obs = super().get_observation(robot_name=robot_name, skip_images=skip_images)
        obs["pos"] = self.pos
        return obs


class _ChunkPolicy(MockPolicy):
    """Returns ``chunk_size`` actions per call; records ``get_actions`` count."""

    def __init__(self, joint_names, chunk_size: int) -> None:
        super().__init__()
        self.set_robot_state_keys(list(joint_names))
        self.actions_per_step = chunk_size
        self._chunk_size = chunk_size
        self.get_actions_calls = 0
        self._joint_names = list(joint_names)

    def get_actions_sync(self, observation_dict, instruction, **kwargs):
        self.get_actions_calls += 1
        return [{n: 0.0 for n in self._joint_names} for _ in range(self._chunk_size)]

    async def get_actions(self, observation_dict, instruction, **kwargs):
        return self.get_actions_sync(observation_dict, instruction, **kwargs)


class _PosThresholdBenchmark(BenchmarkProtocol):
    """Sparse-success spec: succeeds once ``sim.pos`` reaches ``threshold``."""

    def __init__(self, threshold: int, max_steps: int) -> None:
        self.threshold = threshold
        self.max_steps = max_steps

    @property
    def supported_robots(self) -> list[str]:
        return []  # any robot

    @property
    def default_robot(self) -> str:
        return "fake_robot"

    def on_step(self, sim, obs, action) -> StepInfo:
        return StepInfo(reward=0.0, done=False)

    def is_success(self, sim) -> bool:
        return getattr(sim, "pos", 0) >= self.threshold


def test_legacy_eval_consumes_full_chunk_before_requery():
    """action_horizon=8, chunk=8, max_steps=24 -> 3 queries, 24 actions.

    Pre-fix this was 24 queries / 24 actions (``actions[0]`` only, re-query
    every step), so ``action_horizon`` had zero effect.
    """
    sim = _CountingSim()
    policy = _ChunkPolicy(sim.robot_joint_names("fake_robot"), chunk_size=8)

    result = PolicyRunner(sim).evaluate(
        "fake_robot",
        policy,
        n_episodes=1,
        max_steps=24,
        action_horizon=8,
        success_fn=lambda obs: False,  # never succeeds: full horizon runs
    )

    assert result["status"] == "success"
    send_actions = [c for c in sim.calls if c[0] == "send_action"]
    assert len(send_actions) == 24, f"expected 24 actions applied, got {len(send_actions)}"
    assert policy.get_actions_calls == 3, (
        f"expected 3 chunk queries (24/8), got {policy.get_actions_calls} - "
        "the legacy path is re-querying instead of consuming the chunk"
    )


def test_legacy_eval_success_on_final_step_is_recorded():
    """Predicate true exactly when the last action lands -> success=True.

    Threshold == max_steps: success only becomes true on the post-action
    observation of the final applied action. Pre-fix the loop exited without
    re-reading the observation, so this was reported as failure.
    """
    sim = _CountingSim()
    policy = _ChunkPolicy(sim.robot_joint_names("fake_robot"), chunk_size=8)

    result = PolicyRunner(sim).evaluate(
        "fake_robot",
        policy,
        n_episodes=1,
        max_steps=8,
        action_horizon=8,
        success_fn=lambda obs: obs["pos"] >= 8,
    )

    payload = result["content"][1]["json"]
    assert payload["success_rate"] == 1.0, "success on the final step was not recorded"
    # avg_steps must be exactly 8, not 9 (no stale one-step-late detection).
    assert payload["avg_steps"] == 8.0, f"expected avg_steps=8, got {payload['avg_steps']}"


def test_legacy_success_fn_and_spec_paths_agree_on_success_rate():
    """Same deterministic policy + predicate -> identical success_rate.

    The threshold lands on the final applied action so the stale-observation
    bug is observable: pre-fix the legacy path missed the final-step success
    (rate 0.0) while the spec path recorded it (rate 1.0). Post-fix both agree.
    """
    threshold, max_steps, chunk = 24, 24, 8

    legacy_sim = _CountingSim()
    legacy = PolicyRunner(legacy_sim).evaluate(
        "fake_robot",
        _ChunkPolicy(legacy_sim.robot_joint_names("fake_robot"), chunk_size=chunk),
        n_episodes=3,
        max_steps=max_steps,
        action_horizon=chunk,
        success_fn=lambda obs: obs["pos"] >= threshold,
    )

    spec_sim = _CountingSim()
    spec = PolicyRunner(spec_sim).evaluate(
        "fake_robot",
        _ChunkPolicy(spec_sim.robot_joint_names("fake_robot"), chunk_size=chunk),
        n_episodes=3,
        action_horizon=chunk,
        spec=_PosThresholdBenchmark(threshold=threshold, max_steps=max_steps),
    )

    legacy_rate = legacy["content"][1]["json"]["success_rate"]
    spec_rate = spec["content"][1]["json"]["success_rate"]
    assert legacy_rate == spec_rate == 1.0, (
        f"legacy success_fn and spec paths disagree: legacy={legacy_rate} spec={spec_rate}"
    )
