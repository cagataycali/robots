"""Robot mesh networking - peer-to-peer presence, state, RPC, and teleoperation.

This package provides the Zenoh-based mesh layer for strands-robots. Each robot
(hardware or simulated) owns a :class:`Mesh` component that broadcasts its
presence, publishes sensor streams, and responds to RPC commands from peers.

Typical usage::

    from strands_robots.mesh import init_mesh

    mesh = init_mesh(robot, peer_id="arm-001")
    if mesh is not None:
        print(mesh.alive)  # True
        print(mesh.peers)  # discovered peers
        mesh.stop()

Submodules
----------
- ``session`` - Shared Zenoh session singleton and peer registry
- ``core`` - The Mesh class (lifecycle, presence, state, RPC, subscribe)
- ``sensors`` - Extended sensor topic loops (pose, health, IMU, odom, lidar, hand, map)
- ``input`` - InputPublisher / InputReceiver for teleoperation over mesh

The append-only safety event log this package writes through is
:mod:`strands_robots.audit`, one layer down: three layers write to it, so it
sits under all of them rather than inside the first of them.

The in-process registry of mesh-enabled robots is reachable through
:func:`get_local_robots`, which returns a snapshot. Code that needs to
mutate the registry itself reaches ``strands_robots.mesh.core``, where it is
defined; this package re-exports the public surface only.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any

from strands_robots.audit import log_safety_event
from strands_robots.mesh.core import Mesh, get_local_robots, init_mesh
from strands_robots.mesh.input import InputPublisher, InputReceiver
from strands_robots.mesh.session import (
    clear_peers,
    current_session,
    get_peers,
    get_session,
    prune_peers,
    put,
    release_session,
    session_alive,
    update_peer,
)

#: The ROS 2 mobile-base drivers this package held before they moved to
#: :mod:`strands_robots.drivers.ros` under the same module stems, by public name
#: -> stem. The names and the old module paths both resolve, with a
#: :class:`DeprecationWarning`, until 0.7.
_MOVED_TO_DRIVERS: dict[str, str] = {
    "ActionCapable": "_mobile_base",
    "MobileBaseRobot": "_mobile_base",
    "ServiceCapable": "_mobile_base",
    "Transport": "_mobile_base",
    "AckermannRosRobot": "ackermann_robot",
    "RosBridgedRobot": "ros_bridge",
    "RosbridgeRobot": "rosbridge_robot",
    "RtpsRobot": "rtps_robot",
}
_DRIVERS_ROS = "strands_robots.drivers.ros"


def __getattr__(name: str) -> Any:
    """Resolve a moved ROS mobile-base name from :mod:`strands_robots.drivers.ros`, with a warning."""
    stem = _MOVED_TO_DRIVERS.get(name)
    if stem is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"strands_robots.mesh.{name} moved to {_DRIVERS_ROS} and is removed from strands_robots.mesh in 0.7",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(f"{_DRIVERS_ROS}.{stem}"), name)


class _MovedModule(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Answer ``import strands_robots.mesh.<stem>`` with the moved driver module itself."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> ModuleSpec | None:
        stem = fullname.removeprefix(f"{__name__}.")
        if stem == fullname or stem not in _MOVED_TO_DRIVERS.values():
            return None
        return importlib.util.spec_from_loader(fullname, self)

    #: The moved module's own spec, keyed by the alias name, kept across the
    #: create/exec pair because importlib rebinds ``__spec__`` in between.
    _real_specs: dict[str, ModuleSpec | None]

    def __init__(self) -> None:
        self._real_specs = {}

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        stem = spec.name.removeprefix(f"{__name__}.")
        warnings.warn(
            f"{spec.name} moved to {_DRIVERS_ROS}.{stem} and is removed in 0.7", DeprecationWarning, stacklevel=2
        )
        module = importlib.import_module(f"{_DRIVERS_ROS}.{stem}")
        # importlib's ``_init_module_attrs`` sets ``__spec__`` on whatever
        # ``create_module`` returns, which here is the real driver module; a
        # clobbered spec makes ``importlib.reload(<driver module>)`` a no-op
        # that renames the module to the alias path. Stash it to restore below.
        self._real_specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        """The module is the moved one, already executed: give it its own spec back."""
        alias = next(
            (name for name, spec in self._real_specs.items() if spec is not None and spec.name == module.__name__), None
        )
        if alias is not None:
            module.__spec__ = self._real_specs.pop(alias)


if not any(type(finder).__qualname__ == _MovedModule.__qualname__ for finder in sys.meta_path):  # a reload adds none
    sys.meta_path.append(_MovedModule())


__all__ = [
    # Core types
    "Mesh",
    "InputPublisher",
    "InputReceiver",
    # Factory & registry
    "init_mesh",
    "get_local_robots",
    # Session helpers (re-exported from .session for convenience)
    "put",
    "get_session",
    "release_session",
    "current_session",
    "session_alive",
    # Peer registry
    "get_peers",
    "update_peer",
    "clear_peers",
    "prune_peers",
    # Safety
    "log_safety_event",
]
