"""The Isaac IK model carries the registry's ``tool_frame`` site, as MuJoCo's does.

The MuJoCo backend adds a robot's registry-declared ``tool_frame`` (a TCP site on
a named body - so100: ``tcp`` on ``Fixed_Jaw``) at ``add_robot``, and
``discover_ee_frame`` ranks that site first. Isaac compiles its IK model from the
same MJCF but never added it, so discovery fell to the wrist BODY: measured on
6.1 (a live GPU probe run), ``move_to`` on so100 tracked
``Wrist_Pitch_Roll`` while MuJoCo tracked ``so100/tcp`` - the same target, a
different point of the arm.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation import tool_frame as tool_frame_module
from strands_robots.simulation.ik import discover_ee_frame
from strands_robots.simulation.isaac.simulation import _RobotState
from strands_robots.simulation.tool_frame import ToolFrame

from .test_move_to_ik import ARM_XML, _make_sim

_NO_SITE_XML = ARM_XML.replace('<site name="ee_site" pos="0.05 0 0"/>', "")


def _robot(tmp_path: Any, xml: str = _NO_SITE_XML, name: str = "so100.xml") -> tuple[Any, _RobotState]:
    path = tmp_path / name
    path.write_text(xml, encoding="utf-8")
    sim, _ = _make_sim()
    robot = sim._robots["arm"]
    robot.description_path = str(path)
    return sim, robot


def _declare(monkeypatch, frame: ToolFrame | None, err: str | None = None) -> list:
    seen = []

    def _lookup(key):
        seen.append(key)
        return frame, err

    monkeypatch.setattr(tool_frame_module, "registry_tool_frame", _lookup)
    return seen


def test_the_declared_tool_site_is_the_ik_frame(tmp_path, monkeypatch) -> None:
    sim, robot = _robot(tmp_path)
    seen = _declare(monkeypatch, ToolFrame(body="link4", pos=(0.08, 0.0, 0.0), site="tcp"))

    _mj, model, err = sim._load_ik_mjcf(robot)

    assert err is None
    assert seen == ["prim_arm"]  # data_config is the registry key, as on MuJoCo
    assert discover_ee_frame(model, None) == ("tcp", "site")


def test_without_a_declaration_discovery_is_unchanged(tmp_path, monkeypatch) -> None:
    sim, robot = _robot(tmp_path)
    _declare(monkeypatch, None)

    _mj, model, err = sim._load_ik_mjcf(robot)

    assert err is None
    frame = discover_ee_frame(model, None)
    assert frame is not None and frame[1] == "body"


def test_a_urdf_description_is_exempt_as_on_mujoco(tmp_path, monkeypatch) -> None:
    sim, robot = _robot(tmp_path)
    robot.description_path = str(robot.description_path).replace(".xml", ".urdf")
    (tmp_path / "so100.urdf").write_text(
        '<robot name="r"><link name="a"/><link name="b"/>'
        '<joint name="j" type="revolute"><parent link="a"/><child link="b"/>'
        '<limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>',
        encoding="utf-8",
    )
    seen = _declare(monkeypatch, ToolFrame(body="b", pos=(0.0, 0.0, 0.1), site="tcp"))

    sim._load_ik_mjcf(robot)

    assert seen == []


def test_a_declaration_that_does_not_fit_the_model_falls_back_loudly(tmp_path, monkeypatch, caplog) -> None:
    """A custom description under a registry key: the robot is already on stage,
    so solve on the model as it stands and say the declaration was not applied."""
    sim, robot = _robot(tmp_path)
    _declare(monkeypatch, ToolFrame(body="no_such_body", pos=(0.0, 0.0, 0.0), site="tcp"))

    with caplog.at_level("WARNING"):
        _mj, model, err = sim._load_ik_mjcf(robot)

    assert err is None
    frame = discover_ee_frame(model, None)
    assert frame is not None and frame[1] == "body"
    assert any("no_such_body" in r.getMessage() for r in caplog.records)
