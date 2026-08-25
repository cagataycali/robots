"""Driver seam for ``Robot(..., mode="real")``.

:class:`~strands_robots.drivers.base.HardwareDriver` is the contract a real
robot satisfies; :mod:`strands_robots.drivers.registry` decides which
implementation a given robot gets. A driver package that is not lerobot-shaped
registers itself here::

    from strands_robots.drivers import register_native_driver

    register_native_driver("unitree_g1", G1Driver)

and ``Robot("unitree_g1", mode="real", driver="strands")`` then builds it.
"""

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
