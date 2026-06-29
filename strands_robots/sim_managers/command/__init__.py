"""Command manager package."""

from strands_robots.sim_managers.command import terms  # noqa: F401  (registers terms)
from strands_robots.sim_managers.command.manager import CommandManager

__all__ = ["CommandManager"]
