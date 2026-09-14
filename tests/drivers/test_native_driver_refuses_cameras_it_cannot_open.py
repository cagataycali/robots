"""A ``cameras=`` no native driver will open is refused, not dropped.

``Robot(mode="real")`` forwards ``cameras=`` verbatim to the native driver it
builds. No driver shipped here reads it - these robots address their cameras
through their own SDK - so the keyword was accepted, the build reported success,
and the caller was handed a robot with no cameras at all. Nothing said so: eight
shipped robots declare ``driver="strands"`` in the registry, so a plain
``Robot("unitree_go2", mode="real", cameras={...})`` reached that path without
the caller ever writing ``driver=``.

The two sibling doors already answer correctly, which is what makes this a
convergence rather than a new rule: ``mode="sim"`` refuses this same keyword and
names the door that does attach cameras, and the spawn arguments the real branch
cannot honour are at least reported. A dropped camera is the one a caller cannot
see - it surfaces as a recording with no image columns - so it is the one that
has to be refused.

The refusal is read off the driver class, not a list, because drivers grow: one
that really does open the cameras it is given declares ``reads_cameras = True``
and receives them verbatim.
"""

from __future__ import annotations

from typing import Any

import pytest

import strands_robots.robot as robot_mod
from strands_robots import Robot
from strands_robots.drivers import get_native_driver_class, list_native_drivers

#: One camera, spelled the way the factory's own docstring spells it.
_CAMERAS = {"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}

#: A robot with a native driver that constructs without a bus or an SDK, used
#: for the rows that must reach the driver rather than the refusal.
_ROBOT = "so101"


def _one_robot_per_unopening_driver() -> dict[str, str]:
    """Map each shipped driver class that does not open cameras to one robot.

    Derived from the live registry rather than written out, so a driver added
    later is graded without touching this file, and one that starts opening
    cameras leaves the table by declaring ``reads_cameras``.
    """
    by_class: dict[str, str] = {}
    for name in sorted(list_native_drivers()):
        cls = get_native_driver_class(name)
        if cls is None or getattr(cls, "reads_cameras", False):
            continue
        by_class.setdefault(cls.__name__, name)
    return by_class


_UNOPENING = _one_robot_per_unopening_driver()


class TestADriverThatWillNotOpenThemIsToldSo:
    """The refusal, once per shipped driver class that discards ``cameras``."""

    def test_the_table_is_not_empty(self) -> None:
        """Premise: without a row the parametrized test below grades nothing."""
        assert _UNOPENING, "no shipped native driver was found to grade"

    @pytest.mark.parametrize(("driver_name", "robot_name"), sorted(_UNOPENING.items()))
    def test_a_camera_it_will_not_open_is_refused(self, driver_name: str, robot_name: str) -> None:
        with pytest.raises(ValueError, match="does not open cameras") as excinfo:
            Robot(robot_name, mode="real", driver="strands", cameras=_CAMERAS, mesh=False)
        message = str(excinfo.value)
        assert driver_name in message, "the refusal must name the driver that will not open them"
        assert repr(robot_name) in message
        # A refusal whose remedy is unnamed sends the caller back to the same
        # call. The named remedy is graded for real below.
        assert "driver='lerobot'" in message

    def test_the_declared_driver_is_not_built(self) -> None:
        """Refused at the door: the driver is never constructed, so nothing to undo."""
        built: list[Any] = []

        class _Recording:
            reads_cameras = False

            def __init__(self, **kwargs: Any) -> None:  # pragma: no cover - must not run
                built.append(kwargs)

        with pytest.raises(ValueError, match="does not open cameras"):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(robot_mod, "get_native_driver_class", lambda _name: _Recording)
                Robot(_ROBOT, mode="real", driver="strands", cameras=_CAMERAS, mesh=False)
        assert built == [], "the driver was constructed before the camera dict was graded"


class TestTheHonourableCallsAreUntouched:
    """Declaring no camera is honoured by doing nothing, not by refusing."""

    @pytest.mark.parametrize("cameras", [None, {}], ids=["unset", "empty"])
    def test_a_call_declaring_no_camera_still_builds(self, cameras: dict[str, Any] | None) -> None:
        driver = Robot(_ROBOT, mode="real", driver="strands", cameras=cameras, mesh=False, port="/dev/null")
        assert driver.tool_name == _ROBOT

    def test_a_driver_that_declares_it_opens_them_receives_them(self) -> None:
        """The opt-in is the whole growth hook: one flag, forwarded verbatim."""
        seen: dict[str, Any] = {}

        class _Opens:
            reads_cameras = True
            tool_name = _ROBOT

            def __init__(self, **kwargs: Any) -> None:
                seen.update(kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(robot_mod, "get_native_driver_class", lambda _name: _Opens)
            Robot(_ROBOT, mode="real", driver="strands", cameras=_CAMERAS, mesh=False)
        assert seen["cameras"] == _CAMERAS


class TestTheRemedyTheRefusalNamesWorks:
    """A remedy that does not attach cameras either would be a dead end."""

    def test_the_lerobot_path_attaches_the_camera_the_refusal_points_at(self) -> None:
        robot = Robot(_ROBOT, mode="real", driver="lerobot", cameras=_CAMERAS, mesh=False, port="/dev/null")
        assert sorted(robot.robot.config.cameras) == ["front"]
