"""The ``open_duck_mini`` registry entry names a model file its upstream tree has.

It declared ``open_duck_mini_v2.xml``; the fetched tree
(``apirrone/Open_Duck_Mini`` ``mini_bdx/robots/open_duck_mini_v2``) holds
``robot.xml``, ``robot_motors.xml``, ``scene.xml`` and ``scene_position.xml``, so
``download_assets(robots="open_duck_mini")`` reported "Failed: 1 ... fetched tree
has no open_duck_mini_v2.xml" - and the robot then loaded anyway from
``scene.xml``, which includes ``robot_motors.xml``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from strands_robots.registry import get_robot


def _asset() -> dict:
    robot = get_robot("open_duck_mini")
    assert robot is not None
    return robot["asset"]


def test_the_declared_model_is_the_one_the_scene_includes() -> None:
    asset = _asset()
    assert asset["model_xml"] == "robot_motors.xml" and asset["scene_xml"] == "scene.xml"


def test_a_fetched_tree_carries_the_declared_model() -> None:
    root = Path(os.path.expanduser("~/.strands_robots/assets/open_duck_mini_v2"))
    if not (root / "scene.xml").is_file():
        pytest.skip("open_duck_mini assets not fetched on this machine")
    asset = _asset()
    assert (root / asset["model_xml"]).is_file()
    includes = re.findall(r'^\s*<include\s+file="([^"]+)"', (root / "scene.xml").read_text(), re.M)
    assert asset["model_xml"] in includes
