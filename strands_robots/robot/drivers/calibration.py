"""Per-joint bus calibration, in LeRobot's calibration JSON format."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
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


# LeRobot 0.5+ writes every SO-10x arm under one robot name (``so_follower`` / ``so_leader``)
_LEROBOT_RENAMED = {f"so10{v}_{r}": f"so_{r}" for v in "01" for r in ("follower", "leader")}


def lerobot_calibration_path(robot_type: str, robot_id: str) -> Path:
    """Where ``lerobot-calibrate --robot.type=<type> --robot.id=<id>`` writes (honours HF_LEROBOT_CALIBRATION)."""
    root = Path(os.environ.get("HF_LEROBOT_CALIBRATION", "~/.cache/huggingface/lerobot/calibration")).expanduser()
    return root / "robots" / robot_type / f"{robot_id}.json"


def lerobot_calibration_candidates(robot_type: str, robot_id: str | None) -> list[Path]:
    """Every file a LeRobot calibration of this arm may be, newest LeRobot naming first.

    Without ``robot_id``: ``<type>.json``, or the only ``*.json`` in the directory (``--robot.id=my_arm``)."""
    out: list[Path] = []
    for rtype in (_LEROBOT_RENAMED.get(robot_type), robot_type):
        if rtype is None:
            continue
        path = lerobot_calibration_path(rtype, robot_id or robot_type)
        only = sorted(path.parent.glob("*.json")) if robot_id is None and path.parent.is_dir() else []
        out += [path] + (only if len(only) == 1 and only[0] != path else [])
    return out
