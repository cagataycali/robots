"""Declarative config DSL: build a full manager set from a dict or YAML file.

A managers config is a mapping of manager keys to ``{"terms": [...]}`` blocks::

    command_manager:
      terms:
        - name: base_velocity
          func: uniform_velocity
          params: {lin_vel_x: [-1.0, 1.0], ang_vel_z: [-1.0, 1.0]}
    observation_manager:
      terms:
        - {func: base_lin_vel, scale: 2.0}
        - {func: velocity_commands}
    reward_manager:
      terms:
        - {name: track_lin_vel, func: track_lin_vel_xy_exp, weight: 1.0, params: {std: 0.25}}
        - {name: lin_vel_z, func: lin_vel_z_l2, weight: -2.0}
    termination_manager:
      terms:
        - {func: time_out}
        - {func: bad_orientation, params: {limit_angle: 1.0}}

Term ``func`` names are validated against the closed term registry; an unknown
name raises rather than executing arbitrary code, so a config is safe to parse
from untrusted / LLM-authored YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands_robots.sim_managers.command import CommandManager
from strands_robots.sim_managers.observation import ObservationManager
from strands_robots.sim_managers.reward import RewardManager
from strands_robots.sim_managers.termination import TerminationManager
from strands_robots.utils import require_optional

_MANAGER_KEYS = {
    "command_manager",
    "observation_manager",
    "reward_manager",
    "termination_manager",
}


@dataclass
class ManagerSet:
    """A bundle of managers built from one config.

    Managers are ``None`` when their config block is absent. The
    :class:`CommandManager` should be computed first each step so reward /
    observation terms can read the commands it publishes.

    Args:
        command: Command manager, if configured.
        observation: Observation manager, if configured.
        reward: Reward manager, if configured.
        termination: Termination manager, if configured.
    """

    command: CommandManager | None = None
    observation: ObservationManager | None = None
    reward: RewardManager | None = None
    termination: TerminationManager | None = None


def build_managers(config: dict[str, Any]) -> ManagerSet:
    """Build a :class:`ManagerSet` from a config mapping.

    Args:
        config: Mapping with any of ``command_manager``, ``observation_manager``,
            ``reward_manager``, ``termination_manager``.

    Returns:
        A :class:`ManagerSet` with the configured managers.

    Raises:
        ValueError: On an unknown top-level key or an unknown term ``func``.
    """
    unknown = set(config) - _MANAGER_KEYS
    if unknown:
        raise ValueError(f"unknown manager config keys {sorted(unknown)}; allowed: {sorted(_MANAGER_KEYS)}")
    return ManagerSet(
        command=CommandManager.from_config(config["command_manager"]) if "command_manager" in config else None,
        observation=(
            ObservationManager.from_config(config["observation_manager"]) if "observation_manager" in config else None
        ),
        reward=RewardManager.from_config(config["reward_manager"]) if "reward_manager" in config else None,
        termination=(
            TerminationManager.from_config(config["termination_manager"]) if "termination_manager" in config else None
        ),
    )


def load_managers_config(path: str | Path) -> ManagerSet:
    """Load a managers config from a YAML or JSON file and build the managers.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` / ``.json`` config file.

    Returns:
        The built :class:`ManagerSet`.

    Raises:
        ImportError: If a YAML file is given but PyYAML is not installed
            (raised by ``require_optional`` with an install hint).
        ValueError: If the file does not parse to a mapping, or on an unknown
            manager key / term ``func``.
    """
    text = Path(path).read_text(encoding="utf-8")
    suffix = Path(path).suffix.lower()
    if suffix in (".yaml", ".yml"):
        yaml = require_optional("yaml", pip_install="pyyaml", purpose="YAML manager config loading")
        data = yaml.safe_load(text)  # type: ignore[attr-defined]
    else:
        import json

        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"managers config must be a mapping, got {type(data).__name__}")
    return build_managers(data)
