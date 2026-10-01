"""A robot converted as several articulations (aloha) is one robot with every joint.

Measured on one L40S (Isaac Sim 6.1): aloha's MJCF converts to two
articulations, one per arm, each on its own welded base. ``add_robot`` wrapped a
single ``Articulation`` over the container, which bound the first root only:
8 of 16 joints, the right arm uncommandable, while MuJoCo exposes all 16.
After: 16 joints named as MuJoCo names them, both arms driven.

Unit-level: real in-memory ``pxr`` stage for the root discovery and anchoring;
stand-in articulation parts for the composite.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")
Usd = pytest.importorskip("pxr.Usd")
from pxr import Sdf, UsdPhysics  # type: ignore[import-not-found]  # noqa: E402

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    _anchor_fixed_base_articulation,
    _articulation_root_paths,
    _MultiArticulation,
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


def _two_arms(st) -> None:
    root = "/World/Robots/aloha"
    st.DefinePrim(root, "Xform")
    st.DefinePrim(f"{root}/Geometry", "Scope")
    for side in ("left", "right"):
        base = st.DefinePrim(f"{root}/Geometry/{side}_base", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(base)
        UsdPhysics.ArticulationRootAPI.Apply(base)
        weld = UsdPhysics.FixedJoint.Define(st, f"{root}/Geometry/{side}_base/PhysicsFixedJoint")
        weld.CreateBody0Rel().SetTargets([Sdf.Path(root)])
        weld.CreateBody1Rel().SetTargets([Sdf.Path(f"{root}/Geometry/{side}_base")])


class TestEveryArmIsFound:
    def test_both_welded_roots_move_to_their_welds(self, stage) -> None:
        _two_arms(stage)
        _anchor_fixed_base_articulation("/World/Robots/aloha")
        assert _articulation_root_paths("/World/Robots/aloha") == [
            "/World/Robots/aloha/Geometry/left_base/PhysicsFixedJoint",
            "/World/Robots/aloha/Geometry/right_base/PhysicsFixedJoint",
        ]

    def test_a_root_left_on_the_base_above_its_weld_is_not_a_second_articulation(self, stage) -> None:
        _two_arms(stage)
        for side in ("left", "right"):
            UsdPhysics.ArticulationRootAPI.Apply(
                stage.GetPrimAtPath(f"/World/Robots/aloha/Geometry/{side}_base/PhysicsFixedJoint")
            )
        roots = _articulation_root_paths("/World/Robots/aloha")
        assert roots == [
            "/World/Robots/aloha/Geometry/left_base/PhysicsFixedJoint",
            "/World/Robots/aloha/Geometry/right_base/PhysicsFixedJoint",
        ]


class _Part:
    def __init__(self, names: list[str], q: list[float]) -> None:
        self.dof_names = names
        self.q = np.array(q, dtype=np.float64)
        self.actions: list[Any] = []
        self.dof_properties = np.array([(-1.0, 1.0)] * len(names), dtype=[("lower", "f8"), ("upper", "f8")])

    def initialize(self) -> None:
        pass

    def get_joint_positions(self) -> np.ndarray:
        return self.q

    def get_joint_velocities(self) -> np.ndarray:
        return self.q * 0

    def set_joint_positions(self, vals: Any, joint_indices: Any = None) -> None:
        self.q[np.asarray(joint_indices)] = vals

    def apply_action(self, action: Any) -> None:
        self.actions.append(action)

    def get_dof_limits(self) -> np.ndarray:
        return np.array([[-1.0, 1.0]] * len(self.dof_names))

    def get_world_pose(self) -> Any:
        return np.zeros(3), np.array([1.0, 0, 0, 0])


class TestTheCompositeIsOneArticulation:
    def _multi(self) -> tuple[_MultiArticulation, _Part, _Part]:
        left = _Part(["left/waist", "left/elbow"], [0.1, 0.2])
        right = _Part(["right/waist", "right/elbow"], [0.3, 0.4])
        return _MultiArticulation([left, right]), left, right

    def test_names_and_state_are_concatenated_in_root_order(self) -> None:
        multi, _, _ = self._multi()
        assert multi.dof_names == ["left/waist", "left/elbow", "right/waist", "right/elbow"]
        assert multi.get_joint_positions().tolist() == [0.1, 0.2, 0.3, 0.4]
        assert multi.get_dof_limits().shape == (4, 2) and len(multi.dof_properties) == 4

    def test_an_indexed_write_reaches_the_arm_it_names(self) -> None:
        multi, left, right = self._multi()
        multi.set_joint_positions(np.array([0.9, -0.9]), joint_indices=np.array([1, 2]))
        assert left.q.tolist() == pytest.approx([0.1, 0.9]) and right.q.tolist() == pytest.approx([-0.9, 0.4])

    def test_an_action_is_split_by_arm_with_local_indices(self) -> None:
        class _Action:
            def __init__(self, **kw: Any) -> None:
                self.__dict__.update(kw)

        multi, left, right = self._multi()
        multi.apply_action(_Action(joint_positions=np.array([0.5, 0.6, 0.7]), joint_indices=np.array([0, 2, 3])))
        assert left.actions[0].joint_indices.tolist() == [0]
        assert left.actions[0].joint_positions.tolist() == pytest.approx([0.5])
        assert right.actions[0].joint_indices.tolist() == [0, 1]
        assert right.actions[0].joint_positions.tolist() == pytest.approx([0.6, 0.7])

    def test_moving_the_base_is_refused_not_applied_to_one_arm(self) -> None:
        multi, _, _ = self._multi()
        with pytest.raises(RuntimeError, match="add_robot\\(position"):
            multi.set_world_pose(position=np.zeros(3))
