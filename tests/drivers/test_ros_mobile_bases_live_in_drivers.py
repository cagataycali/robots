"""The ROS 2 mobile bases are drivers: they live in ``strands_robots.drivers.ros``.

Each old ``strands_robots.mesh`` location still resolves for one minor, to the same
object, and says where the name went.
"""

from __future__ import annotations

import importlib
import sys

import pytest

#: (module stem, the public names it defines) - the stem is unchanged by the move.
_MOVES = (
    ("_mobile_base", ("ActionCapable", "MobileBaseRobot", "ServiceCapable", "Transport")),
    ("ackermann_robot", ("AckermannRosRobot",)),
    ("ros_bridge", ("RosBridgedRobot",)),
    ("rosbridge_robot", ("RosbridgeRobot",)),
    ("rtps_robot", ("RtpsRobot",)),
)


@pytest.mark.parametrize(("stem", "names"), _MOVES, ids=[move[0] for move in _MOVES])
def test_a_moved_mesh_path_resolves_to_the_driver_and_warns(stem: str, names: tuple[str, ...]) -> None:
    moved = importlib.import_module(f"strands_robots.drivers.ros.{stem}")
    package = importlib.import_module("strands_robots.drivers.ros")
    mesh = importlib.import_module("strands_robots.mesh")
    for name in names:
        assert getattr(package, name) is getattr(moved, name)
        assert getattr(moved, name).__module__ == moved.__name__
        with pytest.warns(DeprecationWarning, match=r"moved to strands_robots\.drivers\.ros"):
            assert getattr(mesh, name) is getattr(moved, name)

    sys.modules.pop(f"strands_robots.mesh.{stem}", None)
    with pytest.warns(DeprecationWarning, match=rf"strands_robots\.drivers\.ros\.{stem}\b"):
        assert importlib.import_module(f"strands_robots.mesh.{stem}") is moved
