"""Regression: dataset frames must stay time-aligned across an action chunk.

A chunk-emitting policy returns H actions from ONE ``get_actions`` call and the
runner replays them open-loop (re-querying only every H steps). The recording
``on_frame`` hook writes one ``(observation, action)`` frame per step. The bug:
the runner handed the SAME chunk-start observation to ``on_frame`` for every one
of the H actions, so a recorded episode contained H identical frames (frozen
image + frozen proprioceptive state) paired with H DIFFERENT actions while the
robot visibly moved. That is temporally-misaligned behavioural-cloning data: the
recorded observation does not match the action taken from it.

This bites every chunk-emitting policy (ACT, diffusion, pi0/pi0.5, SmolVLA,
MolmoAct2) and is worst for long chunks (SmolVLA ``n_action_steps=50`` froze an
entire short rollout). It is independent of the chunk-acquisition strategy, so
it manifests on BOTH the synchronous loop and the async-RTC loop.

The fix refreshes the observation handed to ``on_frame`` per step while a
recording is active (inference still consumes the chunk-start observation, so
open-loop replay is unchanged). These tests pin that the recorded per-step
proprioceptive state actually advances within a single chunk.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.mujoco.simulation import Simulation


class _CapturingRecorder:
    """Stub recorder that records the observation handed to each ``add_frame``.

    Mirrors only the surface the MuJoCo ``on_frame`` recording hook touches:
    ``add_frame(observation=, action=, task=)``. The hook forwards the raw sim
    observation (per-joint scalar keys such as ``Rotation`` / ``Pitch`` plus
    their ``.vel`` companions - the real ``DatasetRecorder`` packs these into
    ``observation.state``). We snapshot the scalar proprioceptive vector per
    frame so the test can assert the recorded trajectory advances instead of
    being frozen across the chunk.
    """

    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[dict[str, Any]] = []
        self.episode_frame_count = 0
        self.frame_count = 0

    @staticmethod
    def _scalar_state(observation: dict[str, Any]) -> np.ndarray:
        # Pack every scalar (non-image) observation value into a stable vector.
        items = sorted(
            (k, float(v)) for k, v in observation.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        return np.array([v for _, v in items], dtype=float)

    def add_frame(self, observation: dict[str, Any], action: dict[str, Any], task: str = "") -> None:
        self.states.append(self._scalar_state(observation))
        self.actions.append(dict(action))
        self.episode_frame_count += 1
        self.frame_count += 1

    def save_episode(self) -> dict[str, Any]:  # not exercised for n_episodes=1
        self.episode_frame_count = 0
        return {"status": "success"}


def _make_recording_sim() -> tuple[Simulation, _CapturingRecorder]:
    sim = Simulation(tool_name="chunk_obs_align", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", data_config="so100")
    assert sim._world is not None
    rec = _CapturingRecorder()
    # Open a recording session WITHOUT a real LeRobotDataset: the on_frame hook
    # only needs ``recording`` truthy and a recorder with ``add_frame``. This is
    # the same opaque-injection seam used by the episode-boundary regression.
    sim._world._backend_state["recording"] = True
    sim._world._backend_state["trajectory"] = []
    sim._world._backend_state["dataset_recorder"] = rec
    return sim, rec


@pytest.mark.parametrize("async_rtc", [False, True], ids=["sync", "async_rtc"])
def test_recorded_state_advances_within_a_single_chunk(async_rtc: bool) -> None:
    """Every recorded frame in one chunk must reflect the live (moving) state.

    The rollout is capped BELOW the chunk length so the whole episode is served
    from a single ``get_actions`` call (one chunk, no re-query). Pre-fix all
    frames carry the frozen chunk-start state; post-fix they advance step by
    step. MockPolicy emits smooth sinusoidal targets that move the arm, so a
    correctly time-aligned recording has a strictly non-zero step-to-step delta.
    """
    sim, rec = _make_recording_sim()
    try:
        policy = MockPolicy()
        # MockPolicy emits an 8-action chunk; stay strictly inside it so the
        # only motion in the recording would come from per-step obs refresh,
        # never from a chunk boundary re-query.
        n_steps = 6
        result = sim.run_policy(
            robot_name="arm",
            policy_provider="mock",
            policy_object=policy,
            instruction="move",
            n_steps=n_steps,
            action_horizon=8,
            control_frequency=50.0,
            fast_mode=True,
            async_rtc=async_rtc,
        )
        assert result["status"] == "success", result

        assert len(rec.states) == n_steps, f"expected {n_steps} recorded frames, got {len(rec.states)}"
        states = np.stack(rec.states)
        # The recorded actions DO differ per step (this was never the bug).
        # The defect was the OBSERVATION being frozen against those actions.
        step_deltas = np.linalg.norm(np.diff(states, axis=0), axis=1)
        assert step_deltas.max() > 1e-6, (
            "recorded observation.state is frozen across the chunk - the dataset "
            f"would pair {n_steps} identical observations with {n_steps} distinct "
            f"actions (per-step deltas={step_deltas})"
        )
        # Stronger: most consecutive frames advance (a moving rollout), not just
        # a single non-frozen frame.
        assert (step_deltas > 1e-6).sum() >= n_steps - 2, (
            f"recorded state barely advances across the chunk: deltas={step_deltas}"
        )
    finally:
        sim.cleanup()
