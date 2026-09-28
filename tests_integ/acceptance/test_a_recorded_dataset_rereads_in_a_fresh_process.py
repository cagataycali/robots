"""Acceptance: a recorded dataset is finalized and re-read in a fresh process.

``Robot("so101", mode="sim")`` records a 10-second rollout through
``start_recording`` / ``stop_recording``, and a separate interpreter - one that
never imported ``strands_robots`` - opens it with lerobot's own
``LeRobotDataset``. The reader, not the writer, grades the result: the episode
and frame counts, every declared camera as a decodable ``(480, 640, 3)`` video
feature, and a state column that moved. Real MuJoCo, real lerobot, no doubles.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

pytest.importorskip("mujoco")
pytest.importorskip("lerobot.datasets.lerobot_dataset")

FPS = 30
SECONDS = 10.0
CAMERAS = ["default"]

_READER = """
import json, sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(sys.argv[1], root=sys.argv[2])
first, last = ds[0], ds[len(ds) - 1]
states = [ds[i]["observation.state"].tolist() for i in (0, len(ds) // 2, len(ds) - 1)]
print(json.dumps({
    "strands_loaded": any(m.startswith("strands_robots") for m in sys.modules),
    "episodes": ds.num_episodes,
    "frames": ds.num_frames,
    "fps": ds.fps,
    "video_keys": list(ds.meta.video_keys),
    "shapes": {k: list(v["shape"]) for k, v in ds.meta.features.items() if k in ds.meta.video_keys},
    "decoded": {k: list(first[k].shape) for k in ds.meta.video_keys},
    "last_frame_index": int(last["frame_index"]),
    "states": states,
}))
"""


def test_a_recorded_dataset_rereads_in_a_fresh_process(tmp_path: Path) -> None:
    from strands_robots import Robot

    root = tmp_path / "dataset"
    robot = Robot("so101", mode="sim")
    try:
        opened = robot.start_recording(
            repo_id="local/acceptance", task="hold", fps=FPS, root=str(root), overwrite=True, cameras=CAMERAS
        )
        assert opened["status"] == "success", opened
        ran = robot.run_policy(
            robot_name="so101", policy_provider="mock", instruction="hold", duration=SECONDS, control_frequency=FPS
        )
        assert ran["status"] == "success", ran
        stopped = robot.stop_recording()
        assert stopped["status"] == "success", stopped
    finally:
        robot.destroy()

    read = subprocess.run(
        [sys.executable, "-c", _READER, "local/acceptance", str(root)], capture_output=True, text=True, check=False
    )
    assert read.returncode == 0, read.stderr[-2000:]
    got = json.loads(read.stdout.strip().splitlines()[-1])

    frames = int(FPS * SECONDS)
    keys = [f"observation.images.{c}" for c in CAMERAS]
    assert got["strands_loaded"] is False
    assert (got["episodes"], got["frames"], got["fps"], got["last_frame_index"]) == (1, frames, FPS, frames - 1)
    assert got["video_keys"] == keys
    assert got["shapes"] == {k: [480, 640, 3] for k in keys}
    assert got["decoded"] == {k: [3, 480, 640] for k in keys}
    assert len({tuple(s) for s in got["states"]}) > 1, got["states"]
