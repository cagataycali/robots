"""Choose which driver builds a ``mode="real"`` robot, and find it.

Two questions, kept apart because they have different answers:

* *Which driver?* :func:`resolve_driver` - a name, from the caller's ``driver=``
  or the robot's registry entry, defaulting to
  :data:`~strands_robots.drivers.base.DEFAULT_DRIVER`.
* *Which class?* :func:`get_native_driver_class` - the class a native driver
  package registered for this robot, or ``None``.

The native table starts empty: every robot in the package registry is driven
through lerobot, so nothing is registered until a driver package calls
:func:`register_native_driver`. That is the seam - the point where an
implementation that is not lerobot-shaped can be reached by
:func:`~strands_robots.robot.Robot` without the factory knowing anything about
it.
"""

from __future__ import annotations

import logging

from strands_robots.drivers.base import DEFAULT_DRIVER, DRIVER_CHOICES, missing_driver_members
from strands_robots.registry import get_driver, resolve_name

logger = logging.getLogger(__name__)

#: Canonical robot name -> the class that drives it natively. Written only by
#: :func:`register_native_driver`, so every entry has passed the surface check.
_NATIVE_DRIVERS: dict[str, type] = {}


def driver_choice_error(value: object, param: str, context: str) -> str | None:
    """Report why ``value`` is not a driver name, or ``None`` if it is one.

    Args:
        value: The candidate driver name.
        param: Parameter name to quote in the reason.
        context: Calling surface to quote in the reason.

    Returns:
        A reason naming the accepted values, or ``None`` when ``value`` is
        one of :data:`~strands_robots.drivers.base.DRIVER_CHOICES`.
    """
    if isinstance(value, str) and value in DRIVER_CHOICES:
        return None
    return f"{context}: {param} must be one of {', '.join(DRIVER_CHOICES)}, got {value!r}"


def resolve_driver(canonical: str, explicit: str | None = None) -> str:
    """Decide which driver name builds ``canonical``.

    Precedence, highest first: the caller's explicit choice, the robot's
    registry ``hardware.driver``, then
    :data:`~strands_robots.drivers.base.DEFAULT_DRIVER`. ``"auto"`` and ``None``
    both mean "no explicit choice", so they defer to the registry.

    Args:
        canonical: Canonical robot name (or any alias -
            :func:`~strands_robots.registry.resolve_name` is applied).
        explicit: The caller's ``driver=``, or ``None`` when unset.

    Returns:
        A concrete driver name - never ``"auto"``.

    Raises:
        ValueError: If ``explicit`` is not one of
            :data:`~strands_robots.drivers.base.DRIVER_CHOICES`.
    """
    reason = driver_choice_error(explicit, "driver", "resolve_driver") if explicit is not None else None
    if reason is not None:
        raise ValueError(reason)

    if explicit is not None and explicit != "auto":
        return explicit

    # The registry reports what a robot declares and applies no default, so the
    # two "no preference" spellings - an absent field and a declared "auto" -
    # both land here, on the same answer.
    declared = get_driver(canonical)
    if declared is None or declared == "auto":
        return DEFAULT_DRIVER
    return declared


def register_native_driver(canonical: str, driver_cls: type, overwrite: bool = False) -> None:
    """Register ``driver_cls`` as the native driver for ``canonical``.

    The surface is checked here rather than at build time, because here is where
    the mistake is made: a driver missing ``stream`` registers fine and then
    fails on the first agent call, one process and several minutes away from the
    line that is wrong.

    Args:
        canonical: Robot name the driver drives; resolved through
            :func:`~strands_robots.registry.resolve_name` so an alias and its
            canonical name cannot register two different drivers. The robot need
            not exist in the registry yet - a driver package may register before
            the entry it serves is merged, and refusing that would make
            registration order-dependent.
        driver_cls: The class to build. Must satisfy
            :class:`~strands_robots.drivers.base.HardwareDriver`.
        overwrite: Replace an existing registration instead of refusing it.

    Raises:
        TypeError: If ``driver_cls`` does not satisfy the driver surface. The
            missing members are named - a contract violation, so the same class
            of error :mod:`abc` raises for an unimplemented abstract method.
        ValueError: If ``canonical`` already has a driver and ``overwrite`` is
            ``False``.
    """
    missing = missing_driver_members(driver_cls)
    if missing:
        raise TypeError(
            f"register_native_driver: {driver_cls.__name__} cannot drive {canonical!r} - "
            f"it is missing {', '.join(missing)}. See strands_robots.drivers.HardwareDriver "
            "for what a driver must expose."
        )

    key = resolve_name(canonical)
    existing = _NATIVE_DRIVERS.get(key)
    if existing is not None and not overwrite:
        raise ValueError(
            f"register_native_driver: {key!r} is already driven by {existing.__name__}. "
            f"Pass overwrite=True to replace it with {driver_cls.__name__}."
        )

    _NATIVE_DRIVERS[key] = driver_cls
    logger.debug("Registered native driver %s for %r", driver_cls.__name__, key)


def get_native_driver_class(canonical: str) -> type | None:
    """Return the native driver class for ``canonical``, or ``None``.

    Args:
        canonical: Robot name or alias.

    Returns:
        The registered class, or ``None`` when no native driver serves this
        robot - which is every robot until a driver package registers one.
    """
    return _NATIVE_DRIVERS.get(resolve_name(canonical))


def list_native_drivers() -> dict[str, str]:
    """Report which robots have a native driver.

    The discovery surface behind the refusal
    :func:`~strands_robots.robot.Robot` raises for ``driver="strands"`` on a
    robot with no native driver: a caller who asked for one needs to see what
    is available, not only that their choice was not.

    Returns:
        Canonical robot name -> driver class name, sorted by robot name.
    """
    return {name: cls.__name__ for name, cls in sorted(_NATIVE_DRIVERS.items())}
