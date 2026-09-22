"""``Robot(ros2_bridge="false")`` is refused for the flag, not for a missing rclpy.

``docs/ros2/hardware-bridge.md`` promises that ``ros2_bridge`` and ``ros2_commands``
"are checked at construction, so a config that spells the flag ``"false"`` is
refused". ``Robot.__init__`` reads ``if ros2_bridge:`` to decide whether to probe
the transport dependency before the grading in ``_init_ros_bridge`` runs, and
``"false"`` is truthy - so on a box without a sourced ROS 2 distro a caller who
asked for *no* bridge was told ``'rclpy' is required for the ROS 2 telemetry
bridge (ros2_bridge=True)``. The refusal has to report identically whether or not
the transport dependency is installed, so both flags are graded ahead of it.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

import strands_robots.utils as utils_mod
from strands_robots.hardware_robot import Robot

TRUTHY_OFF_SPELLINGS = ("false", "False", "no", "off", "0")


@pytest.fixture
def no_ros2_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither transport dependency is importable, as on a plain pip install."""
    for name in ("rclpy", "rosidl_runtime_py", "cyclonedds"):
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(utils_mod._lazy_modules, name, raising=False)


def _construct(**kwargs: Any) -> Robot:
    """The documented route: ``Robot("so101", mode="real", ...)`` lands here."""
    return Robot(tool_name="arm", robot="so101", **kwargs)


@pytest.mark.parametrize("spelling", TRUTHY_OFF_SPELLINGS)
def test_ros2_bridge_off_spelling_is_refused_for_the_flag(spelling: Any, no_ros2_transport: None) -> None:
    with pytest.raises(ValueError, match="ros2_bridge"):
        _construct(ros2_bridge=spelling)


@pytest.mark.parametrize("spelling", TRUTHY_OFF_SPELLINGS)
def test_ros2_commands_off_spelling_is_refused_for_the_flag(spelling: Any, no_ros2_transport: None) -> None:
    with pytest.raises(ValueError, match="ros2_commands"):
        _construct(ros2_bridge=True, ros2_commands=spelling)


def test_a_real_bridge_request_still_names_the_missing_transport(no_ros2_transport: None) -> None:
    with pytest.raises(ImportError, match="rclpy"):
        _construct(ros2_bridge=True)
