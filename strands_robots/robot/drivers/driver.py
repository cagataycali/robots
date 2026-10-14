"""A Driver is a hardware Body: a servo bus, a DDS link, a vendor SDK.

Hardware bodies in 1.0: feetech (SO-100/101), unitree (G1/Go2/H1 over DDS), microduck (robotd).
``make_driver`` is the only place that maps a description + port to a driver class; the driver is
read from ``description.hardware.driver`` (registry data), and drivers import their SDK lazily.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ...description import Description
from ...types import JointState, Pose
from .calibration import Calibration

_DRIVERS: dict[str, str] = {
    "feetech": "strands_robots.robot.drivers.feetech:FeetechDriver",
    "unitree": "strands_robots.robot.drivers.unitree:UnitreeDriver",
    "microduck": "strands_robots.robot.drivers.microduck:MicroduckDriver",
}


@runtime_checkable
class Driver(Protocol):
    """Hardware Body: what a bus must offer. ``names`` are the description's action_keys, in order."""

    names: tuple[str, ...]

    def read(self) -> JointState: ...

    def write(self, targets: np.ndarray) -> None: ...

    def pose(self) -> Pose | None: ...

    def images(self) -> Mapping[str, np.ndarray]: ...

    def extras(self) -> Mapping[str, Any]: ...

    def calibrate(self, *, interactive: bool = True) -> Calibration: ...

    def close(self) -> None: ...


def driver_kind(description: Description) -> str | None:
    """Which driver family a registry entry declares (``hardware.driver``), or None for no 1.0 driver."""
    driver = description.hardware.driver
    return driver if driver in _DRIVERS else None


def make_driver(
    description: Description,
    *,
    port: str | None,
    driver: str | Driver | None,
    cameras: Mapping[str, int | str] | None,
) -> Driver:
    if driver is not None and not isinstance(driver, str):
        if not isinstance(driver, Driver):
            raise TypeError(f"driver= must be a driver name or a Driver instance, got {type(driver).__name__}")
        return driver
    kind = driver or driver_kind(description)
    if kind is None:
        raise ValueError(
            f"no driver known for {description.name!r}; "
            "pass driver='feetech'|'unitree'|'microduck' or a Driver instance"
        )
    if kind not in _DRIVERS:
        raise ValueError(f"unknown driver {kind!r}; known drivers: {sorted(_DRIVERS)}")
    mod_name, cls_name = _DRIVERS[kind].split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(description, port=port, cameras=cameras or {})


__all__ = ["Driver", "Calibration", "make_driver", "driver_kind"]
