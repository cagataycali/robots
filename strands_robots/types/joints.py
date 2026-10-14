"""Joint readings and the Action a policy returns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


def _ro(a: Any, dtype: Any = np.float64) -> np.ndarray:
    """A read-only float array (frozen dataclasses deserve frozen arrays)."""
    out = np.array(a, dtype=dtype)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class JointState:
    """Positions/velocities/efforts for a named set of joints (SI: rad, rad/s, N·m)."""

    names: tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray
    efforts: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.names)
        for f in ("positions", "velocities", "efforts"):
            arr = _ro(getattr(self, f))
            if arr.shape != (n,):
                raise ValueError(f"JointState.{f} has shape {arr.shape}, expected ({n},) for {n} joints")
            object.__setattr__(self, f, arr)

    def __getitem__(self, joint: str) -> float:
        try:
            return float(self.positions[self.names.index(joint)])
        except ValueError:
            raise KeyError(f"unknown joint {joint!r}; joints are {list(self.names)}") from None

    def __len__(self) -> int:
        return len(self.names)

    def as_dict(self) -> dict[str, float]:
        return {n: float(p) for n, p in zip(self.names, self.positions, strict=True)}

    @classmethod
    def zeros(cls, names: tuple[str, ...] | list[str]) -> JointState:
        z = np.zeros(len(names))
        return cls(tuple(names), z, z.copy(), z.copy())


Action = np.ndarray | Mapping[str, float] | JointState
"""Joint targets, either a vector in ``Description.action_keys`` order, a ``{joint: target}`` mapping
(missing joints hold their current position) or the ``JointState`` an observation measured (a hold)."""
