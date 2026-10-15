"""A second ``run(record=same_dir)`` resumes: the episode index continues and earlier episodes are untouched."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from strands_robots import Robot, Simulation
from strands_robots.policy import SequencePolicy

pytest.importorskip("lerobot.datasets.lerobot_dataset")

_RUN = """
import sys
from strands_robots import Robot
from strands_robots.policy import SequencePolicy
arm = Robot("so101")
arm.sim.camera("wrist", position=[0.3, -0.3, 0.3], target=[0, 0, 0.1], size=(64, 48), attach=arm)
r = arm.run(SequencePolicy([{"shoulder_pan": -0.3}]), hz=20, steps=7, record=sys.argv[1], instruction="back")
print(r.recorded)
"""

_DUMP = """
import json, sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/x", root=sys.argv[1])
rows = [ds[i] for i in range(ds.num_frames)]
print(json.dumps({
    "episodes": ds.num_episodes,
    "episode_index": [int(r["episode_index"]) for r in rows],
    "action": [[round(float(v), 6) for v in r["action"]] for r in rows],
    "state": [[round(float(v), 6) for v in r["observation.state"]] for r in rows],
    "pixels": [round(float(r["observation.images.wrist"].mean()), 3) for r in rows],
    "tasks": [r["task"] for r in rows],
}))
"""


def _py(code: str, *args) -> str:
    out = subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def _arm():
    arm = Robot("so101")
    arm.sim.camera("wrist", position=[0.3, -0.3, 0.3], target=[0, 0, 0.1], size=(64, 48), attach=arm)
    return arm


def test_a_second_run_in_a_new_process_resumes_the_same_directory(tmp_path):
    target = tmp_path / "ds"
    _arm().run(SequencePolicy([{"shoulder_pan": 0.3}]), hz=20, steps=9, record=target, instruction="out")
    before = json.loads(_py(_DUMP, target))
    assert before["episodes"] == 1 and before["episode_index"] == [0] * 9
    assert _py(_RUN, target) == str(target)  # a different interpreter appends to what this one wrote
    after = json.loads(_py(_DUMP, target))
    assert after["episodes"] == 2 and after["episode_index"] == [0] * 9 + [1] * 7
    for k in ("action", "state", "pixels", "tasks"):
        assert after[k][:9] == before[k], f"episode 0 {k} changed on resume"
    assert after["tasks"][9:] == ["back"] * 7 and after["action"][-1][0] == pytest.approx(-0.3)
    print(f"[measured] resume: episodes {before['episodes']} -> {after['episodes']}, frames 9 -> {len(after['tasks'])}")


def test_resume_refuses_a_different_fps_or_robot(tmp_path):
    target = tmp_path / "ds"
    arm = _arm()
    arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=20, steps=3, record=target)
    with pytest.raises(ValueError, match="fps"):
        arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=10, steps=3, record=target)
    g1 = Robot("g1", sim=Simulation("mujoco"), keyframe="stand")
    with pytest.raises(ValueError, match="features"):
        g1.run(lambda o: o.joints.positions, hz=20, steps=3, record=target)
    g1.sim.close()


def test_non_integer_hz_is_refused_before_step_one(tmp_path):
    with pytest.raises(ValueError, match="integer fps"):
        _arm().run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=12.5, steps=2, record=tmp_path / "x")
    assert not (tmp_path / "x").exists()
