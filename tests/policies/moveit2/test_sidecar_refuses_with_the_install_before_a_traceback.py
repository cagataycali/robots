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
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from strands_robots.policies.moveit2.server import zmq_node
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
