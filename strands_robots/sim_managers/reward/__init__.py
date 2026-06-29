"""Reward manager package."""

from strands_robots.sim_managers.reward import terms  # noqa: F401  (registers terms)
from strands_robots.sim_managers.reward.manager import RewardManager

__all__ = ["RewardManager"]
