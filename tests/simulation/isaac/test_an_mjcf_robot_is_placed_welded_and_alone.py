"""An MJCF robot on Isaac lands where it was asked, is a fixed-base articulation, and brings no floor.

Measured on one L40S (Isaac Sim 6.1), three robots - so100 at the origin, so100
at (0.6, 0.2, 0), panda at (-0.8, 0, 0):

* every robot spawned at [0, 0, 0] whatever ``position`` said, so the two
  so100s overlapped: driving arm_a dragged arm_b 1.45 rad;
* ``get_jacobian`` refused panda ("Only fixed-base articulations are supported"),
  the delta-EEF controller failed on every action, and ``set_robot_pose`` was
  undone within one step (or drove the arm to NaN): the converter puts the
  articulation root on the base rigid body and welds it with a separate joint,
  which PhysX reads as a floating base held by a weld;
* the stage had four ground planes: the world's and one ``Geometry/floor`` per
  robot, from the menagerie ``scene.xml``;
* ``get_body_state("arm_b/Base")`` answered arm_a's Base.

After: bases at (0, 0), (0.6, 0.2), (-0.8, 0); arm_b drift 0.01 rad; one plane;
panda's Jacobian (3, 9) within 0.03 of MuJoCo's; a floating go2 asked for
(1, 0.5, 0.3) lands at z 0.741 (MuJoCo 0.745).

Real in-memory ``pxr`` stage, faked ``omni.usd``.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("strands_robots.simulation.isaac")
Usd = pytest.importorskip("pxr.Usd")
from pxr import Sdf, UsdGeom, UsdPhysics  # type: ignore[import-not-found]  # noqa: E402

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacSimulation,
    _anchor_fixed_base_articulation,
    _deactivate_imported_ground_planes,
    _place_robot_container,
    _RobotState,
)


@pytest.fixture
def stage(monkeypatch):
    st = Usd.Stage.CreateInMemory()
    ctx = types.SimpleNamespace(get_stage=lambda: st)
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.get_context = lambda: ctx  # type: ignore[attr-defined]
    omni = types.ModuleType("omni")
    omni.usd = omni_usd  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd)
    return st


def _mjcf_robot(st, root: str) -> None:
    """The MJCF converter's 6.1 layout: root API on the base body, welded to the container."""
    st.DefinePrim(root, "Xform")
    st.DefinePrim(f"{root}/Geometry", "Scope")
    base = st.DefinePrim(f"{root}/Geometry/Base", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(base)
    UsdPhysics.ArticulationRootAPI.Apply(base)
    link = st.DefinePrim(f"{root}/Geometry/Base/Upper_Arm", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(link)
    weld = UsdPhysics.FixedJoint.Define(st, f"{root}/Geometry/Base/PhysicsFixedJoint")
    weld.CreateBody0Rel().SetTargets([Sdf.Path(root)])
    weld.CreateBody1Rel().SetTargets([Sdf.Path(f"{root}/Geometry/Base")])
    floor = UsdGeom.Plane.Define(st, f"{root}/Geometry/floor").GetPrim()
    UsdPhysics.CollisionAPI.Apply(floor)


def _roots(st) -> list[str]:
    return [str(p.GetPath()) for p in st.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]


class TestTheMjcfBaseIsFixed:
    def test_the_root_moves_to_the_world_weld(self, stage) -> None:
        _mjcf_robot(stage, "/World/Robots/so100")
        assert (
            _anchor_fixed_base_articulation("/World/Robots/so100")
            == "/World/Robots/so100/Geometry/Base/PhysicsFixedJoint"
        )
        assert _roots(stage) == ["/World/Robots/so100/Geometry/Base/PhysicsFixedJoint"]


class TestThePositionIsAuthoredOnTheContainer:
    def test_the_container_carries_the_translate(self, stage) -> None:
        _mjcf_robot(stage, "/World/Robots/arm_b")
        assert _place_robot_container("/World/Robots/arm_b", [0.6, 0.2, 0.0]) is True
        xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Robots/arm_b"))
        t = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        assert [round(v, 6) for v in t] == [0.6, 0.2, 0.0]
        base = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Robots/arm_b/Geometry/Base"))
        assert [
            round(v, 6) for v in base.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        ] == [
            0.6,
            0.2,
            0.0,
        ]

    def test_an_existing_translate_is_overwritten_not_stacked(self, stage) -> None:
        _mjcf_robot(stage, "/World/Robots/arm")
        _place_robot_container("/World/Robots/arm", [1.0, 0.0, 0.0])
        _place_robot_container("/World/Robots/arm", [0.0, 2.0, 0.0])
        ops = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Robots/arm")).GetOrderedXformOps()
        assert len(ops) == 1 and list(ops[0].Get()) == [0.0, 2.0, 0.0]

    @pytest.mark.parametrize("bad", [None, [0.1, 0.2], [float("nan"), 0, 0], "x"])
    def test_an_unusable_position_writes_nothing(self, stage, bad) -> None:
        _mjcf_robot(stage, "/World/Robots/arm")
        assert _place_robot_container("/World/Robots/arm", bad) is False
        assert UsdGeom.Xformable(stage.GetPrimAtPath("/World/Robots/arm")).GetOrderedXformOps() == []


class TestTheImportedFloorIsDropped:
    def test_a_world_plane_goes_and_the_robot_stays(self, stage) -> None:
        _mjcf_robot(stage, "/World/Robots/so100")
        link_plane = UsdGeom.Plane.Define(stage, "/World/Robots/so100/Geometry/Base/Upper_Arm/pad").GetPrim()
        assert _deactivate_imported_ground_planes("/World/Robots/so100") == ["/World/Robots/so100/Geometry/floor"]
        assert not stage.GetPrimAtPath("/World/Robots/so100/Geometry/floor").IsActive()
        assert link_plane.IsActive() and stage.GetPrimAtPath("/World/Robots/so100/Geometry/Base").IsActive()


class TestEachRobotAnswersForItsOwnLinks:
    def test_the_second_robots_link_is_its_own(self, stage) -> None:
        for name in ("arm_a", "arm_b"):
            _mjcf_robot(stage, f"/World/Robots/{name}")
        arm_b = _RobotState(name="arm_b", prim_path="/World/Robots/arm_b", joint_names=[])
        arm_b.actual_prim_path = "/World/Robots/arm_b"
        prim = IsaacSimulation._find_robot_link_prim(stage, arm_b, "Base", Sdf, Usd, UsdGeom)
        assert str(prim.GetPath()) == "/World/Robots/arm_b/Geometry/Base"

    def test_the_second_robots_gripper_frame_is_its_own(self, stage) -> None:
        from tests.simulation._isaac_engine import isaac_engine

        engine = isaac_engine()
        engine._robots = {}
        for name, x in (("arm_a", 0.0), ("arm_b", 0.6)):
            root = f"/World/Robots/{name}"
            _mjcf_robot(stage, root)
            _place_robot_container(root, [x, 0.0, 0.0])
            stage.DefinePrim(f"{root}/Geometry/Base/Upper_Arm/gripper_frame", "Xform")
            robot = _RobotState(name=name, prim_path=root, joint_names=[])
            robot.actual_prim_path = root
            engine._robots[name] = robot
        pose = IsaacSimulation.gripper_frame_pose(engine, "arm_b")
        assert pose is not None and round(pose[0][0], 6) == 0.6
