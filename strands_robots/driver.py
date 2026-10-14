"""A Driver is a hardware Body: a servo bus, a DDS link, a vendor SDK.

``make_driver`` is the only place that maps a description + port to a driver
class; drivers themselves live in ``drivers/`` and import their SDK lazily.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .description import Description
from .types import Calibration, JointState, Pose

_DRIVERS: dict[str, str] = {
    "feetech": "strands_robots.drivers.feetech:FeetechDriver",
    "unitree": "strands_robots.drivers.unitree:UnitreeDriver",
    "microduck": "strands_robots.drivers.microduck:MicroduckDriver",
}
_HARDWARE_DRIVER = {
    "so100": "feetech",
    "so101": "feetech",
    "unitree_g1": "unitree",
    "unitree_go2": "unitree",
    "unitree_h1": "unitree",
    "microduck": "microduck",
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
    """Which driver family a registry entry implies (from its ``hardware`` block or its name)."""
    hw = description.hardware
    if hw.get("driver") in _DRIVERS:
        return str(hw["driver"])
    lt = str(hw.get("lerobot_type", ""))
    for key, kind in _HARDWARE_DRIVER.items():
        if description.name.startswith(key) or lt.startswith(key):
            return kind
    return None


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
