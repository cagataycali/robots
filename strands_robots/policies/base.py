"""Abstract base class for VLA policies."""

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    """Abstract base class for VLA policies.

    All policies implement async get_actions().  For convenience, a
    synchronous wrapper get_actions_sync() is provided.
    """

    @abstractmethod
    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Get actions from policy given observation and instruction.

        Args:
            observation_dict: Robot observation (cameras + state).
            instruction: Natural language instruction.

        Returns:
            List of action dicts for robot execution.
        """
        pass

    def get_actions_sync(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Synchronous convenience wrapper around get_actions().

        Safe to call from sync code, event loops, or notebooks.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.get_actions(observation_dict, instruction, **kwargs),
                ).result()
        else:
            return asyncio.run(self.get_actions(observation_dict, instruction, **kwargs))

    @abstractmethod
    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Configure the policy with robot state keys."""
        pass

    def reset(self, seed: int | None = None) -> None:
        """Reset per-episode policy state.

        Default implementation is a no-op. Policies that hold per-episode
        state (e.g. diffusion sampler RNG, action chunk caches, KV-caches)
        should override to apply the reset.

        For SERVICE-mode policies (e.g. ``Gr00tPolicy(host=...)`` over
        ZMQ), the override forwards the call to the server so its
        per-episode RNG state can be re-initialised — without this,
        ``set_eval_seed`` only seeds the client-side process, leaving
        the server's diffusion sampler RNG drifting across calls and
        breaking reproducibility (#187).

        Args:
            seed: Optional master seed forwarded to the policy's
                random-number generators. When ``None``, implementations
                may apply a default seed or leave RNG state untouched.
        """
        # Default no-op. Concrete policies override to apply per-episode
        # state reset (RNG seeding, action-cache flush, server-side
        # reset endpoint call, etc.).
        return None

    @property
    def requires_images(self) -> bool:
        """Whether this policy needs camera frames in its observation.

        Default True (most VLA policies do). Subclasses that only consume
        joint state (e.g. ``MockPolicy``, pure-IK controllers, scripted
        trajectories) can return ``False`` to let the simulation skip
        expensive camera rendering - a ~10x throughput win at 500Hz when
        no cameras are needed.
        """
        return True

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name for identification."""
        pass
