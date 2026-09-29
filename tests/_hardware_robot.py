"""A :class:`strands_robots.hardware_robot.Robot` built by its own constructor, no bus.

``Robot.__init__`` touches hardware in one step, ``_initialize_robot``, which
builds the lerobot device from a config or type string. Everything else it does
is state: the task latch, the executor, the stop and shutdown events, the stream
gate, the ROS bridge switches. A test that drives one verb against a stand-in
device starts from that state and hands its stand-in in as ``robot`` - exactly
what a ``__new__`` skeleton did with ``hw.robot = ...``, minus the copy of
``__init__`` that falls behind the moment the constructor gains a field.
``tests/test_hardware_stand_ins_start_from_the_constructor.py`` refuses the copy.

The executor is a :class:`tests._daemon_executor.DaemonThreadExecutor`, so a work
item a test abandons costs that test its verdict and not a hung job.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest import mock

from tests._daemon_executor import DaemonThreadExecutor

if TYPE_CHECKING:
    from strands_robots.hardware_robot import Robot


def hardware_robot_on(robot: Any = None, *, tool_name: str = "arm", **kwargs: Any) -> Robot:
    """Return a constructed ``Robot`` driving ``robot``; ``kwargs`` go to ``__init__``."""
    from strands_robots.hardware_robot import Robot

    # The constructor logs the device's name; the placeholder answers that, so a
    # stand-in needs to model only what the test drives.
    placeholder = SimpleNamespace(name=tool_name)
    with mock.patch.object(Robot, "_initialize_robot", return_value=placeholder):
        hw = Robot(tool_name, placeholder, **kwargs)  # type: ignore[arg-type]
    hw.robot = robot
    hw._executor.shutdown(wait=False)
    hw._executor = DaemonThreadExecutor(max_workers=1, thread_name_prefix=f"{tool_name}_executor")
    return hw
