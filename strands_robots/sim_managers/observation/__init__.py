"""Observation manager package."""

from strands_robots.sim_managers.observation import terms  # noqa: F401  (registers terms)
from strands_robots.sim_managers.observation.manager import ObservationManager

__all__ = ["ObservationManager"]
