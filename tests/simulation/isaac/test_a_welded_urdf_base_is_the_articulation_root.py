"""A welded URDF robot's articulation root sits on its weld, not its base body.

Isaac Sim 6.1's URDF importer applies ``ArticulationRootAPI`` to the base link
rigid body and welds it to the world with a separate ``root_joint``
``FixedJoint``. PhysX reads a root on a rigid body as a floating-base
articulation, so the weld did not hold: measured on 6.1.0
(a live GPU probe run), a fixed-base arm's ``base_link`` rose from
z=0 to 0.0198 m (pushed out of the ground plane by its own 4 cm cylinder) and
``link1`` from its URDF joint origin 0.05 to 0.0698 m, while 6.0.1 - root on the
``Geometry`` scope - held both exactly. After moving the root to the weld, 6.1
holds both at 0.000 / 0.050 through 240 steps.

Real in-memory ``pxr`` stage, faked ``omni.usd``.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("strands_robots.simulation.isaac")
Usd = pytest.importorskip("pxr.Usd")
from pxr import UsdPhysics  # type: ignore[import-not-found]  # noqa: E402

from strands_robots.simulation.isaac.simulation import _anchor_fixed_base_articulation  # noqa: E402

_ROBOT = "/World/Robots/arm"


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


def _roots(st) -> list[str]:
    return [str(p.GetPath()) for p in st.Traverse() if "PhysicsArticulationRootAPI" in p.GetAppliedSchemas()]


def _arm(st, *, root_on: str, weld_to_world: bool = True) -> None:
    st.DefinePrim(_ROBOT, "Xform")
    geometry = st.DefinePrim(f"{_ROBOT}/Geometry", "Scope")
    base = st.DefinePrim(f"{_ROBOT}/Geometry/base_link", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(base)
    link1 = st.DefinePrim(f"{_ROBOT}/Geometry/base_link/link1", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(link1)
    UsdPhysics.ArticulationRootAPI.Apply(base if root_on == "base" else geometry)
    weld = UsdPhysics.FixedJoint.Define(st, f"{_ROBOT}/Physics/root_joint")
    weld.CreateBody0Rel().SetTargets([_ROBOT if weld_to_world else f"{_ROBOT}/Geometry/base_link/link1"])
    weld.CreateBody1Rel().SetTargets([f"{_ROBOT}/Geometry/base_link"])
    pan = UsdPhysics.RevoluteJoint.Define(st, f"{_ROBOT}/Physics/shoulder_pan")
    pan.CreateBody0Rel().SetTargets([f"{_ROBOT}/Geometry/base_link"])
    pan.CreateBody1Rel().SetTargets([f"{_ROBOT}/Geometry/base_link/link1"])


class TestThe61Layout:
    def test_the_root_moves_from_the_base_body_to_the_weld(self, stage) -> None:
        _arm(stage, root_on="base")

        moved = _anchor_fixed_base_articulation(_ROBOT)

        assert moved == f"{_ROBOT}/Physics/root_joint"
        assert _roots(stage) == [f"{_ROBOT}/Physics/root_joint"]


class TestNothingElseIsTouched:
    def test_the_60_layout_root_on_a_scope_is_kept(self, stage) -> None:
        _arm(stage, root_on="geometry")

        assert _anchor_fixed_base_articulation(_ROBOT) is None
        assert _roots(stage) == [f"{_ROBOT}/Geometry"]

    def test_a_weld_to_another_body_is_not_a_world_anchor(self, stage) -> None:
        _arm(stage, root_on="base", weld_to_world=False)

        assert _anchor_fixed_base_articulation(_ROBOT) is None
        assert _roots(stage) == [f"{_ROBOT}/Geometry/base_link"]

    def test_a_floating_robot_without_a_weld_is_kept(self, stage) -> None:
        _arm(stage, root_on="base")
        stage.RemovePrim(f"{_ROBOT}/Physics/root_joint")

        assert _anchor_fixed_base_articulation(_ROBOT) is None
        assert _roots(stage) == [f"{_ROBOT}/Geometry/base_link"]

    def test_a_missing_prim_is_a_no_op(self, stage) -> None:
        assert _anchor_fixed_base_articulation("/World/Robots/nope") is None
