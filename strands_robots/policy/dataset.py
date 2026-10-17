"""``Policy("<dataset dir>[?episode=n]")``: replay one recorded episode's actions, in the units it was stored in."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .._extras import LEROBOT
from .sequence import SequencePolicy


class ReplayPolicy(SequencePolicy):
    """A recorded episode's ``action`` column played back at the dataset's fps, then held on its last row.

    ``units`` is what ``meta/units.json`` says the dataset stores (``record=`` writes it); a dataset without it
    (one written by another tool) needs ``units=`` from the caller, because guessing radians for degrees moves a
    real arm by a factor of 57.
    """

    def __init__(self, root: Path, episode: int = 0, *, units: str | None = None) -> None:
        info = json.loads((root / "meta" / "info.json").read_text())
        stored = root / "meta" / "units.json"
        units = units or (json.loads(stored.read_text())["units"] if stored.is_file() else None)
        if units is None:
            raise ValueError(f"{root}: no meta/units.json says what the actions are in; pass units='rad' or 'lerobot'")
        self.units, self.hz, self.root, self.episode = units, float(info["fps"]), root, episode
        self.names = [n.removesuffix(".pos") for n in info["features"]["action"]["names"]]
        self.recorded = _episode_actions(root, episode)
        super().__init__(list(self.recorded))

    def attach(self, robot: Any) -> None:
        """Order the recorded columns by this robot's action keys; a key the dataset lacks is refused by name."""
        missing = [k for k in robot.action_keys if k not in self.names]
        if missing:
            raise ValueError(f"{self!r} has no action for {', '.join(missing)} (it recorded {', '.join(self.names)})")
        order = [self.names.index(k) for k in robot.action_keys]
        self.actions = list(self.recorded[:, order])

    def __repr__(self) -> str:
        return f"Policy('{self.root}?episode={self.episode}')"


def _episode_actions(root: Path, episode: int) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(f"replaying a dataset reads its parquet files and needs LeRobot: {LEROBOT} ({e})") from e
    files = sorted((root / "data").glob("**/*.parquet"))
    rows = [pq.read_table(f, columns=["episode_index", "frame_index", "action"]).to_pydict() for f in files]
    found = sorted({e for r in rows for e in r["episode_index"]})
    frames = sorted((f, a) for r in rows for e, f, a in zip(*r.values(), strict=True) if e == episode)
    if not frames:
        raise ValueError(f"{root}: no episode {episode}; it has episodes {found}")
    return np.asarray([a for _, a in frames], dtype=float)
