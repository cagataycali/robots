"""``SimpleReplayBuffer`` - the off-policy transition store for SAC.

On-policy PPO consumes each rollout once and discards it; off-policy SAC instead
keeps a large ring buffer of past transitions and samples minibatches from it
repeatedly, which is what makes it sample-efficient. This buffer pre-allocates
fixed tensors on the learner device and overwrites oldest-first once full, so
there is no per-step allocation at the 50 Hz control rate.

It stores the asymmetric actor / critic observation split the
:class:`~strands_robots.training.rl.env.SimEnv` produces (the actor sees only
deployable keys, the critic may see privileged ones), so the sampled batch can
feed the SAC actor and twin Q critics directly.

Adapted from the Amazon FAR Holosoma ``SimpleReplayBuffer`` (BSD-3-Clause,
https://github.com/amazon-far/holosoma); the storage layout is sim-agnostic and
ported directly, re-homed onto the strands-robots observation contract.
"""

from __future__ import annotations

import torch


class SimpleReplayBuffer:
    """Fixed-capacity ring buffer of ``(obs, action, reward, next_obs, done)``.

    All transitions live in pre-allocated tensors on ``device``; :meth:`add`
    writes at a circular pointer (overwriting the oldest transition once full)
    and :meth:`sample` draws a uniform-random minibatch with replacement.

    Args:
        capacity: Maximum transitions retained (``RLTrainSpec.buffer_size``).
        num_actor_obs: Actor observation dimension.
        num_critic_obs: Critic observation dimension.
        num_actions: Action dimension.
        device: Torch device the storage tensors live on.

    Raises:
        ValueError: If ``capacity`` is not positive.
    """

    def __init__(
        self,
        capacity: int,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self._actor_obs = torch.zeros((self.capacity, num_actor_obs), dtype=torch.float32, device=self.device)
        self._critic_obs = torch.zeros((self.capacity, num_critic_obs), dtype=torch.float32, device=self.device)
        self._actions = torch.zeros((self.capacity, num_actions), dtype=torch.float32, device=self.device)
        self._rewards = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._next_actor_obs = torch.zeros((self.capacity, num_actor_obs), dtype=torch.float32, device=self.device)
        self._next_critic_obs = torch.zeros((self.capacity, num_critic_obs), dtype=torch.float32, device=self.device)
        # ``done`` excludes time-out truncations: a time-out is NOT a terminal
        # state, so its successor value must still be bootstrapped (see add()).
        self._dones = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._ptr = 0
        self._size = 0

    @property
    def size(self) -> int:
        """Number of transitions currently stored (``<= capacity``)."""
        return self._size

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_actor_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Store one transition (each tensor ``(1, dim)`` or ``(dim,)``).

        ``done`` MUST be the *terminal* flag (a real episode termination), not a
        time-out truncation, so the SAC target bootstraps correctly through
        truncated episodes.
        """
        i = self._ptr
        self._actor_obs[i] = actor_obs.detach().reshape(-1).to(self.device)
        self._critic_obs[i] = critic_obs.detach().reshape(-1).to(self.device)
        self._actions[i] = action.detach().reshape(-1).to(self.device)
        self._rewards[i] = float(reward.reshape(-1)[0])
        self._next_actor_obs[i] = next_actor_obs.detach().reshape(-1).to(self.device)
        self._next_critic_obs[i] = next_critic_obs.detach().reshape(-1).to(self.device)
        self._dones[i] = float(done.reshape(-1)[0])
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Draw a uniform-random minibatch (with replacement) from stored data.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict of batched tensors: ``actor_obs``, ``critic_obs``, ``actions``,
            ``rewards`` ``(B, 1)``, ``next_actor_obs``, ``next_critic_obs``,
            ``dones`` ``(B, 1)``.

        Raises:
            ValueError: If the buffer is empty.
        """
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return {
            "actor_obs": self._actor_obs[idx],
            "critic_obs": self._critic_obs[idx],
            "actions": self._actions[idx],
            "rewards": self._rewards[idx],
            "next_actor_obs": self._next_actor_obs[idx],
            "next_critic_obs": self._next_critic_obs[idx],
            "dones": self._dones[idx],
        }
