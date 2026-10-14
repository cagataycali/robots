"""Per-joint bus calibration, in LeRobot's calibration JSON format."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JointCalibration:
    """One joint's bus calibration — field names and meaning match LeRobot's calibration JSON."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclass(frozen=True)
class Calibration:
    """Per-joint calibration for a hardware body. ``to_json``/``from_json`` are LeRobot's file format,
    so a calibration written by ``lerobot-calibrate`` loads here unchanged."""

    joints: Mapping[str, JointCalibration]

    def to_json(self) -> dict[str, dict[str, int]]:
        return {name: dict(vars(jc)) for name, jc in self.joints.items()}

    @classmethod
    def from_json(cls, data: Mapping[str, Mapping[str, Any]]) -> Calibration:
        return cls({name: JointCalibration(**{k: int(v) for k, v in jc.items()}) for name, jc in data.items()})
