"""Driver seam for ``Robot(..., mode="real")``.

:class:`~strands_robots.drivers.base.HardwareDriver` is the contract a real
robot satisfies; :mod:`strands_robots.drivers.registry` decides which
implementation a given robot gets. A driver package that is not lerobot-shaped
registers itself here::

    from strands_robots.drivers import register_native_driver

    register_native_driver("unitree_g1", G1Driver)

and ``Robot("unitree_g1", mode="real", driver="strands")`` then builds it. The
drivers shipped in this package register themselves from :data:`_SHIPPED_DRIVERS`
on import.
"""

import logging

from strands_robots.drivers.base import (
    DEFAULT_DRIVER,
    DRIVER_CHOICES,
    DRIVER_SURFACE,
    HardwareDriver,
    missing_driver_members,
)
from strands_robots.drivers.registry import (
    driver_choice_error,
    get_native_driver_class,
    list_native_drivers,
    register_native_driver,
    resolve_driver,
)

#: The drivers this package ships, as ``(module, class name, robot names)``.
#: A table rather than a block per driver so a second driver cannot arrive with
#: a subtly different guard than the first - the import guard, the
#: already-registered tolerance and the alias handling are written once. Robot
#: names are the *canonical* names; :func:`register_native_driver` resolves
#: aliases, so listing an alias as well is harmless but redundant.
_SHIPPED_DRIVERS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("strands_robots.drivers.g1", "G1Driver", ("g1", "unitree_g1")),
    ("strands_robots.drivers.reachy", "ReachyDriver", ("reachy_mini",)),
)

logger = logging.getLogger(__name__)


def _register_shipped_drivers() -> None:
    """Register the drivers shipped with the package.

    A driver package outside this repo calls
    :func:`register_native_driver` itself; the drivers we ship register here
    so ``Robot("g1", mode="real", driver="strands")`` works without a second
    import. The registration is guarded per driver: an import that fails (a bad
    SDK install, a broken deps subset) skips *that* driver rather than breaking
    every ``from strands_robots.drivers import ...`` statement or costing the
    other drivers their registration. A driver package that overrides a shipped
    registration wins, because ``register_native_driver`` refuses
    double-registration by default and the refusal is tolerated here.
    """
    import importlib

    for module_path, class_name, robot_names in _SHIPPED_DRIVERS:
        try:
            driver_cls = getattr(importlib.import_module(module_path), class_name)
        except Exception:  # noqa: BLE001 - a broken driver must not break the seam
            logger.debug("Shipped driver %s.%s did not import; skipping", module_path, class_name)
            continue
        for canonical in robot_names:
            try:
                register_native_driver(canonical, driver_cls)
            except ValueError:
                # Already registered - a caller registered it first, honour that.
                pass


_register_shipped_drivers()

__all__ = [
    "DEFAULT_DRIVER",
    "DRIVER_CHOICES",
    "DRIVER_SURFACE",
    "HardwareDriver",
    "driver_choice_error",
    "get_native_driver_class",
    "list_native_drivers",
    "missing_driver_members",
    "register_native_driver",
    "resolve_driver",
]
