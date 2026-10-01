"""Acceptance: scripted teleop, leader to follower, records a demonstration.

A scripted leader - an object with ``get_action()`` that speaks lerobot's
``<joint>.pos`` keys, as an SO-101 leader arm does - drives
``Robot("so101", mode="sim")`` for five seconds at 30 Hz through
``attach_teleop`` / ``teleoperate``, the loop and slew bound a physical leader
uses, while ``start_recording`` is open. A separate interpreter that never
imported ``strands_robots`` re-reads the dataset with lerobot's own
``LeRobotDataset``: one episode, one frame per teleop tick, the action column
carrying the leader's sweep and the follower's state following it. Real MuJoCo,
real lerobot; the only scripted part is the leader's hand.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

pytest.importorskip("mujoco")
pytest.importorskip("lerobot.datasets.lerobot_dataset")

HZ = 30
SECONDS = 5.0
AMPLITUDE = 0.4  # rad on shoulder_pan, actuator "1" of the SO-101 MJCF

_READER = """
import json, sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(sys.argv[1], root=sys.argv[2])
names = ds.meta.features["action"]["names"]
state_names = ds.meta.features["observation.state"]["names"]
pan, span = names.index("1"), state_names.index("1")
rows = [ds[i] for i in range(len(ds))]
print(json.dumps({
    "strands_loaded": any(m.startswith("strands_robots") for m in sys.modules),
    "episodes": ds.num_episodes,
    "frames": ds.num_frames,
    "action_pan": [float(r["action"][pan]) for r in rows],
    "state_pan": [float(r["observation.state"][span]) for r in rows],
}))
"""


class _ScriptedLeader:
    is_connected = False

    def connect(self) -> None:
        self.is_connected, self._t0 = True, time.monotonic()

    def disconnect(self) -> None:
        self.is_connected = False

    def get_action(self) -> dict[str, float]:
        return {"shoulder_pan.pos": AMPLITUDE * math.sin(2 * math.pi * 0.4 * (time.monotonic() - self._t0))}


def test_scripted_teleop_records_a_demonstration(tmp_path: Path) -> None:
    from strands_robots import Robot

    root = tmp_path / "dataset"
    follower = Robot("so101", mode="sim")
    try:
        follower.attach_teleop(_ScriptedLeader(), name="leader", map_fn=lambda a: {"1": a["shoulder_pan.pos"]})
        opened = follower.start_recording(
            repo_id="local/teleop",
            task="follow the leader",
            fps=HZ,
            root=str(root),
            overwrite=True,
            cameras=["default"],
        )
        assert opened["status"] == "success", opened
        ran = follower.teleoperate(hz=HZ, duration=SECONDS, block=True)
        assert ran["status"] == "success", ran
        stopped = follower.stop_recording()
        assert stopped["status"] == "success", stopped
    finally:
        follower.destroy()
    ticks = ran["content"][1]["json"]["frames"]

    read = subprocess.run(
        [sys.executable, "-c", _READER, "local/teleop", str(root)], capture_output=True, text=True, check=False
    )
    assert read.returncode == 0, read.stderr[-2000:]
    got = json.loads(read.stdout.strip().splitlines()[-1])

    assert got["strands_loaded"] is False
    assert (got["episodes"], got["frames"]) == (1, ticks)
    assert ticks >= 0.8 * HZ * SECONDS, ticks
    commanded, reached = got["action_pan"], got["state_pan"]
    assert max(commanded) > 0.8 * AMPLITUDE and min(commanded) < -0.8 * AMPLITUDE, commanded
    assert max(reached) > 0.5 * AMPLITUDE and min(reached) < -0.5 * AMPLITUDE, reached
