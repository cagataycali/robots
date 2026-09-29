"""The one place this backend reaches Isaac Sim APIs that 6.1 deprecated.

Isaac Sim 6.1.0 moved the "core API" extensions this backend is built on into
``isaacsim/extsDeprecated/``: ``isaacsim.core.api``, ``isaacsim.core.prims``,
``isaacsim.core.utils`` and ``isaacsim.sensors.camera`` (measured on the
6.1.0.0 wheel; 6.0.x ships them as regular extensions). They still load, so
nothing breaks today - but they are next in line for removal, and before this
module the imports were scattered over ~25 call sites in
:mod:`strands_robots.simulation.isaac.simulation` and
:mod:`strands_robots.simulation.isaac.motion_primitives`, each with a dead ``omni.isaac.*`` fallback for Isaac
Sim 4.x (removed in 5.0; ``_install.ISAAC_SIM_MIN_VERSION`` is 6.0).

Every name is resolved lazily through PEP 562 ``__getattr__``, so importing
this module never imports Isaac Sim, and a call site keeps its function-local
import shape::

    from strands_robots.simulation.isaac._deprecated_api import World

A unit test that fakes ``sys.modules["isaacsim.core.api"]`` still reaches its
fake, because the lookup happens at the moment of that import.

Migrating off a deprecated API (to ``isaacsim.core.experimental.*`` and the new
sensor APIs) now means changing one row of :data:`_SOURCES` - or replacing it
with an adapter here - instead of hunting call sites.
A unit test refuses a deprecated-module import anywhere else in the backend,
and any ``omni.isaac.*`` import at all.
"""

from __future__ import annotations

import importlib
from typing import Any

#: Modules Isaac Sim 6.1 ships under ``extsDeprecated/``.
DEPRECATED_MODULES: tuple[str, ...] = (
    "isaacsim.core.api",
    "isaacsim.core.prims",
    "isaacsim.core.utils",
    "isaacsim.sensors.camera",
)

#: name -> module it is imported from. One row per symbol the backend uses.
_SOURCES: dict[str, str] = {
    "World": "isaacsim.core.api",
    "DynamicCapsule": "isaacsim.core.api.objects",
    "DynamicCuboid": "isaacsim.core.api.objects",
    "DynamicCylinder": "isaacsim.core.api.objects",
    "DynamicSphere": "isaacsim.core.api.objects",
    "FixedCapsule": "isaacsim.core.api.objects",
    "FixedCuboid": "isaacsim.core.api.objects",
    "FixedCylinder": "isaacsim.core.api.objects",
    "FixedSphere": "isaacsim.core.api.objects",
    "SingleArticulation": "isaacsim.core.prims",
    "SingleGeometryPrim": "isaacsim.core.prims",
    "SingleRigidPrim": "isaacsim.core.prims",
    "SingleXFormPrim": "isaacsim.core.prims",
    "add_reference_to_stage": "isaacsim.core.utils.stage",
    "delete_prim": "isaacsim.core.utils.prims",
    "ArticulationAction": "isaacsim.core.utils.types",
    "set_camera_view": "isaacsim.core.utils.viewports",
    "Camera": "isaacsim.sensors.camera",
}


def articulation_cls() -> Any:
    """The single-prim articulation wrapper, probed where 6.x builds put it.

    ``isaacsim.core.api.articulations.Articulation`` first (kept by some 6.0
    builds, and by the unit fakes), then ``isaacsim.core.prims.SingleArticulation``
    (what 6.0.1 and 6.1.0 resolve), then an ``Articulation`` alias under
    ``isaacsim.core.prims``. Raises ``ImportError`` when none resolves.
    """
    last: ImportError | None = None
    for module_name, attr in (
        ("isaacsim.core.api.articulations", "Articulation"),
        ("isaacsim.core.prims", "SingleArticulation"),
        ("isaacsim.core.prims", "Articulation"),
    ):
        try:
            return getattr(importlib.import_module(module_name), attr)
        except ImportError as e:
            last = e
        except AttributeError as e:
            last = ImportError(f"cannot import name {attr!r} from {module_name!r}", name=module_name)
            last.__cause__ = e
    assert last is not None
    raise last


def __getattr__(name: str) -> Any:
    try:
        module_name = _SOURCES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    # ImportError propagates unchanged (with ``.name`` set by the import
    # system): every call site's cleanup clause already catches it, and a
    # ``from ... import`` surfaces an AttributeError raised here as ImportError.
    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted([*globals(), *_SOURCES])


__all__ = ["DEPRECATED_MODULES", "articulation_cls", *_SOURCES]
