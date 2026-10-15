"""LeRobot v3 through lerobot's own ``LeRobotDataset.create / resume``; imported only when a run records."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._extras import hint
from ..types import Observation
from .features import features_from, units_meta

UNITS_FILE = "units.json"

if TYPE_CHECKING:
    from .episode import Episode


def open_dataset(ep: Episode, first: Observation) -> tuple[Any, bool]:
    """``(dataset, created)``: resume the one already at the target (same features, fps, units) or create it."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        raise ImportError(f"record= writes a LeRobot dataset and needs lerobot: {hint('lerobot')} ({e})") from e
    logging.getLogger("lerobot").setLevel(logging.ERROR)
    features = features_from(first, ep.description.action_keys)
    said = units_meta(ep.units, ep.description)
    root = ep.root if ep.root is not None else _hub_cache(ep.repo_id)
    if (root / "meta" / "info.json").is_file():
        _same_units(ep, root, said)
        ds = LeRobotDataset.resume(ep.repo_id, root=root)
        _same(ep, ds, features)
        return ds, False
    robot_type = ep.description.hardware.lerobot_type or ep.description.name
    ds = LeRobotDataset.create(
        ep.repo_id, fps=ep.fps, features=features, root=root, robot_type=robot_type, use_videos=True
    )
    (Path(ds.root) / "meta" / UNITS_FILE).write_text(json.dumps(said, indent=2) + "\n")
    return ds, True


def _same_units(ep: Episode, root: Path, said: dict[str, Any]) -> None:
    path = root / "meta" / UNITS_FILE
    have = json.loads(path.read_text()) if path.is_file() else None
    if have is not None and have != said:
        raise ValueError(
            f"{ep.target}: the dataset there is stored in units {have['units']!r}; this run records {said['units']!r}"
        )


def _hub_cache(repo_id: str) -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME) / repo_id


def _same(ep: Episode, ds: Any, features: dict[str, dict[str, Any]]) -> None:
    have = {k: (tuple(v["shape"]), list(v.get("names") or [])) for k, v in ds.meta.features.items() if k in features}
    want = {k: (tuple(v["shape"]), list(v["names"])) for k, v in features.items()}
    if have != want:
        raise ValueError(f"{ep.target}: the dataset there has features {have}; this run records {want}")
    if int(ds.fps) != ep.fps:
        raise ValueError(f"{ep.target}: the dataset there is {ds.fps} fps; this run records at {ep.fps}")
