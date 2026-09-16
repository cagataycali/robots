"""The quickstart's "what you need" line matches what the steps raise.

It read "Steps 1 and 3-real need hardware; step 2 needs a GPU. Everything
else runs in sim." Step 5 is ``Simulation(ros2_bridge=True)``, which raises
``ImportError`` on a machine with no sourced ROS 2 distro - ``rclpy`` is not
on PyPI. The line now says so, and this pins the two to each other.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

QUICKSTART = Path(__file__).resolve().parents[1] / "docs" / "getting-started" / "quickstart.md"


def _needs_line() -> str:
    text = QUICKSTART.read_text()
    m = re.search(r"Steps 1 and 3-real need hardware;.*?\n\n", text, re.DOTALL)
    assert m, "the quickstart lost its 'what you need' paragraph"
    return m.group(0)


def test_step_5_is_still_the_ros2_bridge():
    assert "Simulation(ros2_bridge=True)" in QUICKSTART.read_text()


def test_the_needs_line_names_ros2_for_step_5():
    line = _needs_line()
    assert "step 5" in line and "ROS 2" in line and "rclpy" in line
    assert "Everything else runs in sim" not in line


def test_the_bridge_raises_the_error_the_line_describes():
    pytest.importorskip("mujoco")
    if __import__("importlib").util.find_spec("rclpy") is not None:
        pytest.skip("rclpy present - the line describes the machine without it")
    from strands_robots.simulation import Simulation

    with pytest.raises(ImportError, match="rclpy") as ei:
        Simulation(ros2_bridge=True)
    assert "setup.bash" in str(ei.value)
