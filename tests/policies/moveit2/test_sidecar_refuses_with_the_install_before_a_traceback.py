# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoveIt2 sidecar entry point refuses with the install that supplies the module.

``python -m strands_robots.policies.moveit2.server.zmq_node`` is the command
the docs give an operator. Run in a shell with no ROS 2 sourced it used to die
on a bare ``import rclpy`` - a traceback ending in ``No module named 'rclpy'``
and exit status 1, with the remedy (source a distro; the module is not on
PyPI) nowhere in it. The same for the ``[moveit2]`` extra's own pyzmq and
msgpack, and for ``moveit_py`` inside ``_build_moveit_py``.

Every absence is now a refusal through ``require_optional`` before any socket
is bound: exit status 2, and the message names the step that supplies the
module. Graded through ``main()``, which is what ``python -m`` runs.

Only an absence. A MoveIt 2 that is present but whose binding moved raises an
``ImportError`` from the construction the gate admitted, and for that one the
remedy is not the answer - the traceback is. So the gate raises
``MissingRosModuleError`` and the exit-1 path keeps every other ``ImportError``:
``ImportError.name`` cannot tell them apart, because ``from moveit.planning
import MoveItPy`` against a renamed binding carries ``name="moveit.planning"``
too. The last cell grades that boundary.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from strands_robots import utils
from strands_robots.policies.moveit2.server import zmq_node
from strands_robots.policies.moveit2.server.zmq_node import MissingRosModuleError
from tests._blocked_module import blocked


def _run_main(caplog: pytest.LogCaptureFixture) -> tuple[int, str]:
    caplog.set_level(logging.ERROR, logger="moveit2.zmq_node")
    code = zmq_node.main(["--port", "0"])
    return code, "\n".join(record.getMessage() for record in caplog.records)


def test_no_rclpy_names_the_distro_to_source_and_exits_2(caplog: pytest.LogCaptureFixture) -> None:
    with blocked("rclpy"):
        code, text = _run_main(caplog)

    assert code == 2
    assert "'rclpy' is required for the MoveIt2 ZMQ sidecar" in text, text
    assert "source /opt/ros/jazzy/setup.bash" in text, text
    assert "ros-jazzy-moveit-py" in text, text
    assert "pip install rclpy" not in text, text
    assert "Traceback" not in text, text


def test_no_msgpack_names_the_moveit2_extra(caplog: pytest.LogCaptureFixture) -> None:
    with blocked("msgpack"):
        code, text = _run_main(caplog)

    assert code == 2
    assert "pip install 'strands-robots[moveit2]'" in text, text


def test_no_moveit_py_after_rclpy_names_the_bindings_and_shuts_rclpy_down(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sourced distro without MoveIt 2's Python bindings is the other unsourced-shell shape."""
    pytest.importorskip("msgpack", reason="msgpack not installed - pip install 'strands-robots[moveit2]'")
    pytest.importorskip("zmq", reason="pyzmq not installed - pip install 'strands-robots[moveit2]'")
    calls: list[str] = []
    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy.init = lambda: calls.append("init")  # type: ignore[attr-defined]
    fake_rclpy.shutdown = lambda: calls.append("shutdown")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)

    with blocked("moveit.planning"), blocked("moveit"):
        code, text = _run_main(caplog)

    assert code == 2
    assert calls == ["init", "shutdown"], calls
    assert "'moveit.planning' is required for the MoveIt2 ZMQ sidecar" in text, text
    assert "ros-jazzy-moveit-py" in text, text
    assert "Failed to construct MoveItPy" not in text, text


def test_a_pose_request_without_geometry_msgs_names_the_distro(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_plan`` imports the last ROS message package the sidecar needs.

    It is reached after the sidecar reported itself listening, so a fork that
    supplies its own planner past the ``_build_moveit_py`` seam meets this one
    first. Unguarded it raised ``ModuleNotFoundError: No module named
    'geometry_msgs'`` - which the REP loop answers the peer with, remedy nowhere
    in it.
    """
    with blocked("geometry_msgs.msg"), pytest.raises(ImportError) as excinfo:
        zmq_node._plan(
            object(),
            planning_group="arm",
            joint_state=None,
            target_pose=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
            target_joints=None,
            world_update=None,
        )

    text = str(excinfo.value)
    assert "'geometry_msgs.msg' is required for the MoveIt2 ZMQ sidecar" in text, text
    assert "source /opt/ros/jazzy/setup.bash" in text, text
    assert "pip install geometry_msgs" not in text, text


def test_a_construction_importerror_is_not_a_missing_install(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A MoveIt 2 whose binding moved keeps the traceback and exit status 1.

    ``from moveit.planning import MoveItPy`` against a renamed binding raises
    ``ImportError(name="moveit.planning")`` - the same ``name`` the gate for
    that module carries, so nothing but the exception type separates them.
    Reported as a missing install it becomes one line, no traceback, and a
    remedy (source a distro, apt install the bindings) that is already done.
    """
    assert issubclass(MissingRosModuleError, ImportError), "forks catch ImportError around the seam"
    pytest.importorskip("msgpack", reason="msgpack not installed - pip install 'strands-robots[moveit2]'")
    pytest.importorskip("zmq", reason="pyzmq not installed - pip install 'strands-robots[moveit2]'")
    calls: list[str] = []
    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy.init = lambda: calls.append("init")  # type: ignore[attr-defined]
    fake_rclpy.shutdown = lambda: calls.append("shutdown")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)

    def moved_binding(args: object) -> object:
        raise ImportError("cannot import name 'MoveItPy' from 'moveit.planning'", name="moveit.planning")

    monkeypatch.setattr(zmq_node, "_build_moveit_py", moved_binding)
    caplog.set_level(logging.ERROR, logger="moveit2.zmq_node")

    code = zmq_node.main(["--port", "0"])

    assert code == 1
    assert calls == ["init", "shutdown"], calls
    assert "Failed to construct MoveItPy" in caplog.text, caplog.text
    assert [record.exc_info is not None for record in caplog.records] == [True], "the traceback is the diagnosis"
    assert "sudo apt install" not in caplog.text, caplog.text


def test_a_moved_binding_is_not_reported_as_a_missing_moveit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The from-imports sit outside the gate, so an installed MoveIt 2 is never refused.

    ``moveit.planning`` importing while ``MoveItPy`` is not in it is a MoveIt 2
    that is present, and CPython reports it as ``ImportError`` with
    ``name="moveit.planning"`` - indistinguishable by ``name`` from the gate's
    own refusal for that module. Only the raise site separates them: gate the
    import, then import.
    """
    planning_mod = types.ModuleType("moveit.planning")  # no MoveItPy attribute
    moveit_pkg = types.ModuleType("moveit")
    moveit_pkg.planning = planning_mod  # type: ignore[attr-defined]
    configs_mod = types.ModuleType("moveit_configs_utils")
    for name, module in (
        ("moveit", moveit_pkg),
        ("moveit.planning", planning_mod),
        ("moveit_configs_utils", configs_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)
        # require_optional memoises what it imports; restore that mapping too,
        # or a later cell requesting the module answers from this fake.
        monkeypatch.setitem(utils._lazy_modules, name, module)

    with pytest.raises(ImportError) as excinfo:
        zmq_node._build_moveit_py(zmq_node._parse_args([]))

    assert not isinstance(excinfo.value, MissingRosModuleError), str(excinfo.value)
    assert "MoveItPy" in str(excinfo.value), str(excinfo.value)
    assert "sudo apt install" not in str(excinfo.value), str(excinfo.value)
