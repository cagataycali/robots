"""Shared type declarations for MuJoCo simulation mixins.

Defines the SimulationProtocol that all mixins can reference instead of
duplicating TYPE_CHECKING stubs for cross-mixin method signatures.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Protocol, runtime_checkable

from strands_robots.simulation.models import SimWorld


@runtime_checkable
class SimulationProtocol(Protocol):
    """Protocol describing the shared state and methods available across all mixins.

    Each mixin operates on a Simulation instance that provides this interface.
    Using a Protocol avoids duplicating private method stubs in TYPE_CHECKING blocks.
    """

    _world: SimWorld | None
    _lock: threading.Lock
    _executor: ThreadPoolExecutor
    _policy_threads: dict[str, Future[Any]]
    _mj: Any  # The lazily-imported mujoco module
    _renderer_model: Any
    _renderers: dict[tuple[int, int], Any]
    default_width: int
    default_height: int

    def _get_renderer(self, width: int, height: int) -> Any: ...
    def _get_sim_observation(self, robot_name: str, cam_name: str | None = None) -> dict[str, Any]: ...
    def _apply_sim_action(self, robot_name: str, action_dict: dict[str, Any], n_substeps: int = 1) -> None: ...
