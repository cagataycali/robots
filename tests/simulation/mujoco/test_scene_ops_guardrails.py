"""Guardrail behavior of the MuJoCo scene-mutation layer (``scene_ops``).

The happy paths of ``patch_scene_mjcf`` / ``replace_scene_mjcf`` are exercised
elsewhere (``test_patch_scene_mjcf``, ``test_replace_scene_mjcf``). This module
pins the *defensive* contract of
:mod:`strands_robots.simulation.mujoco.scene_ops`: the structured-op validators
that reject malformed agent input with an actionable message, the optional-field
branches of each op, and the "no compiled world yet" early returns that every
inject/eject helper makes before touching the spec. These are the boundaries an
autonomous agent hits first when it drives the scene API blind, so they must
fail loudly and predictably rather than crash mid-mutation.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.models import (  # noqa: E402
    SimCamera,
    SimObject,
    SimRobot,
    SimWorld,
)
from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_scene_guard", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


# A minimal single-joint arm so attach-based namespacing can be exercised.
_ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link0" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="50"/>
  </actuator>
</mujoco>
"""


class TestInjectEjectRequireCompiledWorld:
    """Every mutation helper returns a clean ``False`` (never crashes) when the
    world has no spec/model yet - the agent called a scene edit before
    ``create_world``."""

    def test_inject_robot_without_spec_returns_false(self) -> None:
        world = SimWorld()
        ok = scene_ops.inject_robot_into_scene(world, SimRobot(name="r", urdf_path="x.xml"), "x.xml")
        assert ok is False

    def test_inject_object_without_spec_returns_false(self) -> None:
        world = SimWorld()
        assert scene_ops.inject_object_into_scene(world, SimObject(name="o", shape="box")) is False

    def test_inject_camera_without_spec_returns_false(self) -> None:
        world = SimWorld()
        assert scene_ops.inject_camera_into_scene(world, SimCamera(name="c")) is False

    def test_eject_body_without_spec_returns_false(self) -> None:
        world = SimWorld()
        assert scene_ops.eject_body_from_scene(world, "foo") is False

    def test_eject_robot_without_spec_returns_false(self) -> None:
        world = SimWorld()
        assert scene_ops.eject_robot_from_scene(world, "r") is False


class TestInjectFailuresReturnFalse:
    """A spec mutation that raises (bad shape, unreadable URDF) is caught and
    surfaced as ``False`` so the caller can emit an error dict, leaving the
    already-compiled world intact."""

    def test_inject_object_with_unsupported_shape_returns_false(self, sim: Simulation) -> None:
        sim.create_world()
        world = sim._world
        assert world is not None
        nbody_before = world._model.nbody
        assert scene_ops.inject_object_into_scene(world, SimObject(name="bad", shape="not_a_shape")) is False
        # The failed add must not have grown the compiled model.
        assert world._model.nbody == nbody_before

    def test_inject_robot_with_unreadable_urdf_returns_false(self, sim: Simulation) -> None:
        sim.create_world()
        world = sim._world
        assert world is not None
        ok = scene_ops.inject_robot_into_scene(
            world, SimRobot(name="rr", urdf_path="/no/such/file.xml"), "/no/such/file.xml"
        )
        assert ok is False


class TestEjectMissingBodyIsConsistent:
    def test_eject_unknown_body_returns_true_without_changing_model(self, sim: Simulation) -> None:
        """Ejecting a body that isn't in the spec is a no-op that still reports
        success: the caller has already dropped the Python-side entry, so the
        scene stays consistent."""
        sim.create_world()
        world = sim._world
        assert world is not None
        nbody_before = world._model.nbody
        assert scene_ops.eject_body_from_scene(world, "does_not_exist") is True
        assert world._model.nbody == nbody_before


class TestSnapshotRestoreWithoutModel:
    def test_snapshot_empty_world_returns_empty_dict(self) -> None:
        assert scene_ops._snapshot_joint_state(SimWorld()) == {}

    def test_restore_empty_world_restores_nothing(self) -> None:
        assert scene_ops._restore_joint_state(SimWorld(), {}) == 0


class TestPatchOpValidation:
    """``_apply_patch_op`` rejects malformed ops through the public
    ``patch_scene_mjcf`` entry point with an actionable, op-specific message and
    rolls the whole batch back (atomic)."""

    @pytest.fixture
    def world_sim(self, sim: Simulation) -> Simulation:
        sim.create_world()
        return sim

    def test_non_dict_op_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([42])  # type: ignore[list-item]
        assert result["status"] == "error"
        assert "must be a dict" in result["content"][0]["text"]

    def test_add_body_unknown_parent_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "add_body", "parent": "ghost", "name": "x"}])
        assert result["status"] == "error"
        assert "parent 'ghost' not found" in result["content"][0]["text"]

    def test_add_geom_requires_body(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "add_geom", "type": "box"}])
        assert result["status"] == "error"
        assert "add_geom requires 'body'" in result["content"][0]["text"]

    def test_add_geom_unknown_body_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "add_geom", "body": "ghost", "type": "box"}])
        assert result["status"] == "error"
        assert "body 'ghost' not found" in result["content"][0]["text"]

    def test_add_site_requires_name(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "add_site", "body": "world"}])
        assert result["status"] == "error"
        assert "add_site requires 'name'" in result["content"][0]["text"]

    def test_add_site_unknown_body_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "add_site", "body": "ghost", "name": "s"}])
        assert result["status"] == "error"
        assert "body 'ghost' not found" in result["content"][0]["text"]

    def test_set_body_pos_requires_name(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "set_body_pos"}])
        assert result["status"] == "error"
        assert "set_body_pos requires 'name'" in result["content"][0]["text"]

    def test_set_body_pos_unknown_body_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "ghost", "pos": [0, 0, 1]}])
        assert result["status"] == "error"
        assert "body 'ghost' not found" in result["content"][0]["text"]

    def test_set_body_quat_requires_name(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "set_body_quat"}])
        assert result["status"] == "error"
        assert "set_body_quat requires 'name'" in result["content"][0]["text"]

    def test_set_body_quat_unknown_body_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "set_body_quat", "name": "ghost", "quat": [1, 0, 0, 0]}])
        assert result["status"] == "error"
        assert "body 'ghost' not found" in result["content"][0]["text"]

    def test_delete_body_requires_name(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "delete_body"}])
        assert result["status"] == "error"
        assert "delete_body requires 'name'" in result["content"][0]["text"]

    def test_delete_body_unknown_body_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "delete_body", "name": "ghost"}])
        assert result["status"] == "error"
        assert "body 'ghost' not found" in result["content"][0]["text"]


class TestPatchOpOptionalFields:
    """The optional-attribute branches of the add ops (geom name/pos/quat,
    site size/rgba, body quat) are honored and compile into the model."""

    def test_add_geom_with_name_pos_quat(self, sim: Simulation) -> None:
        sim.create_world()
        world = sim._world
        assert world is not None
        result = sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "host", "pos": [0, 0, 0.5]},
                {
                    "op": "add_geom",
                    "body": "host",
                    "name": "shell",
                    "type": "sphere",
                    "size": [0.05],
                    "pos": [0, 0, 0.1],
                    "quat": [1, 0, 0, 0],
                },
            ]
        )
        assert result["status"] == "success", result
        mj = sim._mj
        assert mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_GEOM, "shell") >= 0

    def test_add_site_with_size_and_rgba(self, sim: Simulation) -> None:
        sim.create_world()
        world = sim._world
        assert world is not None
        result = sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "anchor", "pos": [0, 0, 0.3]},
                {
                    "op": "add_site",
                    "body": "anchor",
                    "name": "tip",
                    "pos": [0, 0, 0.1],
                    "size": [0.02],
                    "rgba": [1, 0, 0, 1],
                },
            ]
        )
        assert result["status"] == "success", result
        mj = sim._mj
        assert mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_SITE, "tip") >= 0

    def test_set_body_quat_updates_orientation(self, sim: Simulation) -> None:
        sim.create_world()
        world = sim._world
        assert world is not None
        sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "spinner", "pos": [0, 0, 0.5]},
                {"op": "add_geom", "body": "spinner", "type": "box", "size": [0.05, 0.05, 0.05]},
            ]
        )
        result = sim.patch_scene_mjcf([{"op": "set_body_quat", "name": "spinner", "quat": [0, 1, 0, 0]}])
        assert result["status"] == "success", result
        mj = sim._mj
        bid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_BODY, "spinner")
        assert bid >= 0
        assert pytest.approx(list(world._model.body_quat[bid]), abs=1e-6) == [0.0, 1.0, 0.0, 0.0]


class TestFindBodyScansAttachedRobotBodies:
    def test_namespaced_attached_body_is_patchable(self, sim: Simulation, tmp_path) -> None:
        """A body introduced via ``spec.attach`` (namespaced under the robot
        name) is not visible through ``spec.body(name)`` until the next compile,
        so ``_find_body`` falls back to scanning ``spec.bodies``. Referencing the
        namespaced body in a patch op must resolve through that fallback."""
        arm_path = tmp_path / "arm.xml"
        arm_path.write_text(_ARM_XML)
        sim.create_world()
        sim.add_robot(name="arm1", urdf_path=str(arm_path))
        result = sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "arm1/link0", "pos": [0, 0, 0.2]}])
        assert result["status"] == "success", result
        world = sim._world
        assert world is not None
        mj = sim._mj
        bid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_BODY, "arm1/link0")
        assert bid >= 0
        assert pytest.approx(list(world._model.body_pos[bid]), abs=1e-6) == [0.0, 0.0, 0.2]
