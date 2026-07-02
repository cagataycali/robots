"""Truncation-bootstrap contract for the off-policy RL trainer (FastSAC).

``SimpleReplayBuffer`` bootstraps the SAC target through a stored transition as
``target = r + gamma * (1 - done) * q_next``. A time-out (the episode-length
limit) is NOT a terminal state, so its successor value must still be
bootstrapped: the buffer's ``done`` slot must carry ``info["terminated"]`` (the
genuine terminal), never the env's ``done`` (which is ``terminated OR
time_out``). Storing ``done`` would zero the bootstrap on every time-out -- the
FastTD3 truncation-bootstrap bug fixed upstream in Jun 2025 -- and silently
degrade training while every shape / smoke-train test stays green.

These pin that ``FastSacTrainer.collect_rollout`` stores the terminal flag. They
are ``torch``-only (no MuJoCo, no model downloads): a scripted fake env drives
the done/terminated bookkeeping directly, so the contract runs in the fast CI
lane rather than the GPU / physics lane where the smoke train lives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

torch = pytest.importorskip("torch")

if TYPE_CHECKING:
    from strands_robots.training.rl import SimEnv


class _FakeTermEnv:
    """Minimal ``SimEnv``-shaped fake whose ``step`` emits a scripted terminated/done.

    Exposes only what ``FastSacTrainer.setup`` / ``collect_rollout`` touch:
    device, obs/action dims, ``reset`` and ``step``. ``step`` returns the
    ``SimEnv`` 4-tuple ``(obs, reward, done, info)`` with ``info["terminated"]``
    and ``info["time_out"]`` controlled by ``terminated_flag`` so a test can
    drive a time-out (done=1, terminated=0) or a genuine terminal
    (done=1, terminated=1).
    """

    def __init__(self, terminated_flag: bool, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.num_actor_obs = 2
        self.num_critic_obs = 2
        self.num_actions = 1
        self._terminated = terminated_flag
        self.resets = 0

    def _obs(self) -> dict:
        return {
            "actor_obs": torch.zeros(1, self.num_actor_obs, device=self.device),
            "critic_obs": torch.zeros(1, self.num_critic_obs, device=self.device),
        }

    def reset(self) -> dict:
        self.resets += 1
        return self._obs()

    def step(self, action):  # type: ignore[no-untyped-def]
        # Every step ends the episode (done=1). Whether that end is a genuine
        # terminal or a truncation is exactly what the trainer must record.
        done = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        reward = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        info = {"time_out": (not self._terminated), "terminated": self._terminated}
        return self._obs(), reward, done, info


def _sac_trainer_on_fake(terminated_flag: bool):  # type: ignore[no-untyped-def]
    """Build a ``FastSacTrainer`` set up on a scripted fake env, warmup branch only."""
    from strands_robots.training.rl import RLTrainSpec
    from strands_robots.training.rl.fast_sac import FastSacTrainer

    trainer = FastSacTrainer()
    spec = RLTrainSpec(
        env_factory=lambda: cast("SimEnv", _FakeTermEnv(terminated_flag)),
        output_dir="/tmp/sac_truncation_contract",
        device="cpu",
        rollout_steps=1,
        batch_size=16,
        # Keep the buffer below learning_starts so collect_rollout takes the
        # random-action warmup branch and never calls actor_critic.sample -- the
        # test isolates the done/terminated bookkeeping, not the policy output.
        learning_starts=1_000,
        normalize_obs=False,
        seed=0,
    )
    trainer.setup(spec)
    return trainer


def test_collect_rollout_stores_terminal_not_done_on_timeout() -> None:
    """A time-out (done=1, terminated=0) must be stored as done=0 (bootstrappable)."""
    trainer = _sac_trainer_on_fake(terminated_flag=False)
    trainer.collect_rollout()
    assert trainer.buffer.size == 1
    # The env reported done=1 but terminated=0; the buffer must record the
    # terminal (0.0) so the SAC target still bootstraps through the truncation.
    assert float(trainer.buffer._dones[0].item()) == 0.0


def test_collect_rollout_stores_terminal_on_genuine_terminal() -> None:
    """A genuine terminal (done=1, terminated=1) must be stored as done=1 (no bootstrap)."""
    trainer = _sac_trainer_on_fake(terminated_flag=True)
    trainer.collect_rollout()
    assert trainer.buffer.size == 1
    assert float(trainer.buffer._dones[0].item()) == 1.0
