"""Termination manager package."""

from strands_robots.sim_managers.termination import terms  # noqa: F401  (registers terms)
from strands_robots.sim_managers.termination.manager import TerminationManager, TerminationResult

__all__ = ["TerminationManager", "TerminationResult"]
