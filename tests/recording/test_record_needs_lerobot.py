"""record= without LeRobot's dataset writer refuses by name before the robot moves (no LeRobot needed to run this)."""

from __future__ import annotations

import sys

import pytest

from strands_robots import Robot


def test_record_without_lerobot_refuses_before_the_first_step(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", None)  # the import fails as if not installed
    arm = Robot("so101")
    before = arm.observe(images=False)["shoulder_pan"]
    with pytest.raises(ImportError, match=r"lerobot\[dataset\]"):
        arm.run(lambda obs: {"shoulder_pan": 0.5}, steps=3, record=tmp_path / "ds")
    assert arm.observe(images=False)["shoulder_pan"] == before
    assert not (tmp_path / "ds").exists()
