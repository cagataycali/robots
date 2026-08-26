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


def _register_shipped_drivers() -> None:
    """Register the drivers shipped with the package.

    A driver package outside this repo calls
    :func:`register_native_driver` itself; the drivers we ship register here
    so ``Robot("g1", mode="real", driver="strands")`` works without a second
    import. The registration is guarded: an import that fails (a bad SDK
    install, a broken deps subset) leaves the table empty rather than
    breaking every ``from strands_robots.drivers import ...`` statement, and
    a driver package that overrides the shipped registration wins because
    ``register_native_driver`` refuses double-registration by default.
    """
    try:
        from strands_robots.drivers.g1 import G1Driver
    except Exception:  # noqa: BLE001 - a broken driver must not break the seam
        return
    for canonical in ("g1", "unitree_g1"):
        try:
            register_native_driver(canonical, G1Driver)
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
