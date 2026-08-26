"""Device Connect integration for strands-robots.

Provides DeviceDriver adapters that wrap Robot and Simulation instances,
exposing them to Device Connect's device registry, RPC routing, and event system.

Usage:
    from strands_robots.device_connect import init_device_connect

    robot = Robot("so100")
    runtime = await init_device_connect(robot, peer_id="so100-lab-1")

    # Now discoverable via Device Connect tools:
    #   discover_devices(device_type="strands_robot")
    #   invoke_device("so100-lab-1", "execute", {"instruction": "pick up cube"})

Module-load discipline
----------------------

Every symbol this package exports is behind ``__getattr__`` (:pep:`562`). The
package imports on a stock ``pip install strands-robots`` -- with no extras --
and only reaches ``device_connect_edge`` when a caller *uses* a name that needs
it. The four Device Connect drivers, ``DeviceRuntime`` and the two
``init_device_connect`` entry points all sit behind that gate.

This is required, not a stylistic choice. The sibling module
``strands_robots.device_connect.reachy_transport`` is stdlib-only and is
imported by the native Reachy driver (:mod:`strands_robots.drivers.reachy`).
Importing that leaf executes this ``__init__``, so a package init that eagerly
imports ``device_connect_edge`` raises ``ModuleNotFoundError`` inside the
native driver's first daemon touch on any install without ``[device-connect]``
-- escaping the driver's own no-raise refusal contract and breaking three
:mod:`AGENTS.md` conventions at once (\"Return error dicts, never raise\",
``require_optional()`` for optional deps, the module-load discipline the
driver's docstring makes explicit).

The public names are unchanged: ``from strands_robots.device_connect import
init_device_connect`` still works, and only fails when the caller reaches for a
name whose implementation genuinely needs ``device_connect_edge``. Static tools
(mypy, IDE autocomplete) see the names through the ``TYPE_CHECKING`` guard.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing-only, never executed
    from device_connect_edge import DeviceRuntime  # noqa: F401

    from strands_robots.device_connect._impl import (  # noqa: F401
        init_device_connect,
        init_device_connect_sync,
        resolve_allow_insecure,
    )
    from strands_robots.device_connect.reachy_mini_driver import (  # noqa: F401
        ReachyMiniDriver,
    )
    from strands_robots.device_connect.robot_driver import (  # noqa: F401
        RobotDeviceDriver,
    )
    from strands_robots.device_connect.sim_driver import (  # noqa: F401
        SimulationDeviceDriver,
    )


__all__ = [
    "init_device_connect",
    "init_device_connect_sync",
    "resolve_allow_insecure",
    "RobotDeviceDriver",
    "SimulationDeviceDriver",
    "ReachyMiniDriver",
]


# Map each exported name to the module it lives in. Grouped by whether the
# module itself needs ``device_connect_edge`` so a reader can see at a glance
# which names cost the extra and which are pure Python.
#
# All three driver modules import ``device_connect_edge`` at module load, and
# the two ``init_device_connect*`` entry points construct a ``DeviceRuntime``
# from it. ``resolve_allow_insecure`` is stdlib-only, kept in the same private
# ``_impl`` module so its behavioural pin does not drag the extra in either.
_ATTR_TO_MODULE: dict[str, str] = {
    "init_device_connect": "strands_robots.device_connect._impl",
    "init_device_connect_sync": "strands_robots.device_connect._impl",
    "resolve_allow_insecure": "strands_robots.device_connect._impl",
    "RobotDeviceDriver": "strands_robots.device_connect.robot_driver",
    "SimulationDeviceDriver": "strands_robots.device_connect.sim_driver",
    "ReachyMiniDriver": "strands_robots.device_connect.reachy_mini_driver",
}


def __getattr__(name: str) -> Any:
    """Resolve a public name lazily.

    Follows :pep:`562`. Called by the Python attribute machinery only when
    ``name`` is not already in the module's namespace, so first access pays
    the import cost and every subsequent access is a plain attribute read.

    A name that needs ``device_connect_edge`` is not imported by this package
    on load. Reaching for one on an install without ``[device-connect]``
    raises the ``ImportError`` from the target module rather than during
    ``import strands_robots.device_connect``. That is what lets the stdlib-only
    leaf ``strands_robots.device_connect.reachy_transport`` be imported on a
    stock install: importing that leaf executes this ``__init__`` first, and
    with this contract that ``__init__`` succeeds.
    """
    # Public name from the export table -- resolve to a symbol its module carries.
    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is not None:
        module = importlib.import_module(module_name)
        value = getattr(module, name)
        # Cache on the package so subsequent lookups skip __getattr__ entirely
        # and ``getattr`` returns bit-identical references across calls.
        globals()[name] = value
        return value

    # Submodule lookup -- ``getattr(pkg, "reachy_transport")`` and
    # ``from strands_robots.device_connect import reachy_transport`` are the two
    # spellings a caller uses to reach the leaves this package ships, and only
    # the second one is served by Python's own import machinery before
    # ``__getattr__`` is consulted. Attempting the import here answers the first
    # spelling with the same module the second returns, without turning the
    # dispatcher into a wildcard: a non-existent submodule name still raises
    # ``AttributeError`` below, because ``import_module`` for a name that is
    # neither a module nor in the export table raises ``ModuleNotFoundError``
    # here rather than reaching the caller as an attribute miss.
    if not name.startswith("_"):
        try:
            submodule = importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError:
            pass
        else:
            globals()[name] = submodule
            return submodule

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Advertise the lazy names to ``dir()`` and IDE completion.

    Without this, ``dir(strands_robots.device_connect)`` reports only the
    module's own literals, hiding every public symbol from a reader listing
    the package's contents.
    """
    return sorted(set(__all__) | set(globals()))
