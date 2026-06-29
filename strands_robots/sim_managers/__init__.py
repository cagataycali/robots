"""Backend-agnostic, config-driven manager framework for sim environments.

Compose observation / reward / termination / command recipes declaratively from
atomic, reusable *terms* - the manager pattern used by Isaac Lab / RSL-RL. Every
term reads only a backend-agnostic :class:`EnvState`, so the same recipe runs on
MuJoCo, MJWarp, Isaac Gym, Isaac Sim, or Newton.

Example::

    from strands_robots.sim_managers import RewardManager, ObservationManager

    reward = RewardManager.from_config({
        "terms": [
            {"name": "track", "func": "track_lin_vel_xy_exp", "weight": 1.0,
             "params": {"std": 0.25}},
            {"name": "z_vel", "func": "lin_vel_z_l2", "weight": -2.0},
        ]
    })
    r = reward.compute(state)            # scalar reward
    breakdown = reward.term_values       # per-term contributions

Or build the whole set from one YAML file via :func:`load_managers_config`.
"""

from strands_robots.sim_managers.base import (
    EnvState,
    Manager,
    Term,
    TermSpec,
    get_term_class,
    list_terms,
    register_term,
)
from strands_robots.sim_managers.command import CommandManager
from strands_robots.sim_managers.config import ManagerSet, build_managers, load_managers_config
from strands_robots.sim_managers.observation import ObservationManager
from strands_robots.sim_managers.reward import RewardManager
from strands_robots.sim_managers.termination import TerminationManager, TerminationResult

__all__ = [
    "EnvState",
    "Term",
    "TermSpec",
    "Manager",
    "register_term",
    "get_term_class",
    "list_terms",
    "ObservationManager",
    "RewardManager",
    "TerminationManager",
    "TerminationResult",
    "CommandManager",
    "ManagerSet",
    "build_managers",
    "load_managers_config",
]
