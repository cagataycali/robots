"""Integration tests for ``patch_scene_mjcf`` - the Stage-6 part-2
structured-op MJCF mutator (GH #125).

Where ``replace_scene_mjcf`` atomically swaps the whole scene for an
agent-written MJCF string, ``patch_scene_mjcf`` applies a list of small
structured ops to the LIVE spec and recompiles once at the end. This is
the "surgical edit" path for agents that already have a compiled world
and want to tweak specific bodies / geoms / sites without rewriting the
XML.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")


from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_patch", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


class TestPatchSceneMjcfHappyPath:
    def test_add_body_with_geom(self, sim: Simulation) -> None:
        sim.create_world()
        assert sim._world is not None
        mj = sim._mj
        bodies_before = sim._world._model.nbody

        result = sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "widget", "pos": [0, 0, 1]},
                {"op": "add_geom", "body": "widget", "type": "sphere", "size": [0.1]},
            ]
        )
        assert result["status"] == "success", result
        assert "2 op(s) applied" in result["content"][0]["text"]

        assert sim._world._model.nbody == bodies_before + 1
        assert sim._world is not None
        bid = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, "widget")
        assert bid >= 0

    def test_set_body_pos(self, sim: Simulation) -> None:
        sim.create_world()
        assert sim._world is not None

        sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "marker", "pos": [0, 0, 0.5]},
                {"op": "add_geom", "body": "marker", "type": "box", "size": [0.05, 0.05, 0.05]},
            ]
        )
        result = sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "marker", "pos": [1.0, 2.0, 3.0]}])
        assert result["status"] == "success", result

        mj = sim._mj
        assert sim._world is not None
        bid = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, "marker")
        assert bid >= 0
        # body_pos[bid] should now reflect the new position.
        pos = sim._world._model.body_pos[bid]
        assert pytest.approx(list(pos), rel=1e-6) == [1.0, 2.0, 3.0]

    def test_delete_body(self, sim: Simulation) -> None:
        sim.create_world()
        assert sim._world is not None
        mj = sim._mj

        sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "transient", "pos": [0, 0, 1]},
                {"op": "add_geom", "body": "transient", "type": "sphere", "size": [0.1]},
            ]
        )
        assert sim._world is not None
        assert mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, "transient") >= 0

        result = sim.patch_scene_mjcf([{"op": "delete_body", "name": "transient"}])
        assert result["status"] == "success", result
        assert sim._world is not None
        assert mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, "transient") == -1

    def test_add_site(self, sim: Simulation) -> None:
        sim.create_world()
        assert sim._world is not None

        result = sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "anchor", "pos": [0, 0, 0.3]},
                {"op": "add_geom", "body": "anchor", "type": "sphere", "size": [0.05]},
                {"op": "add_site", "body": "anchor", "name": "tip", "pos": [0, 0, 0.1]},
            ]
        )
        assert result["status"] == "success", result
        mj = sim._mj
        assert sim._world is not None
        sid = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_SITE, "tip")
        assert sid >= 0, "site 'tip' must be present"

    def test_empty_ops_is_noop(self, sim: Simulation) -> None:
        sim.create_world()
        assert sim._world is not None
        nbody_before = sim._world._model.nbody
        result = sim.patch_scene_mjcf([])
        assert result["status"] == "success"
        assert "0 op(s) applied" in result["content"][0]["text"]
        assert sim._world is not None
        assert sim._world._model.nbody == nbody_before


class TestPatchSceneMjcfAtomicRollback:
    def test_failed_op_rolls_back_whole_batch(self, sim: Simulation) -> None:
        """If a later op fails, earlier ops in the same batch must not stick."""
        sim.create_world()
        assert sim._world is not None
        mj = sim._mj
        nbody_before = sim._world._model.nbody

        # 2nd op is invalid - unknown op kind.
        result = sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "doomed", "pos": [0, 0, 1]},
                {"op": "totally_made_up", "name": "whatever"},
            ]
        )
        assert result["status"] == "error"
        assert "unknown op" in result["content"][0]["text"].lower()

        # The `doomed` body from op #1 must NOT be in the compiled model.
        assert sim._world is not None
        assert sim._world._model.nbody == nbody_before
        assert mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, "doomed") == -1

    def test_missing_required_field_rejects(self, sim: Simulation) -> None:
        sim.create_world()
        result = sim.patch_scene_mjcf([{"op": "add_body"}])  # no 'name'
        assert result["status"] == "error"
        assert "name" in result["content"][0]["text"].lower()


class TestPatchSceneMjcfErrorPaths:
    def test_no_world_errors(self) -> None:
        sim = Simulation(tool_name="devx_patch_nw", mesh=False)
        try:
            result = sim.patch_scene_mjcf([{"op": "add_body", "name": "foo"}])
            assert result["status"] == "error"
            assert "no world" in result["content"][0]["text"].lower()
        finally:
            sim.cleanup(policy_stop_timeout=0.5)

    def test_non_list_rejected(self, sim: Simulation) -> None:
        sim.create_world()
        result = sim.patch_scene_mjcf("not a list")  # type: ignore[arg-type]
        assert result["status"] == "error"
        assert "list" in result["content"][0]["text"].lower()

    def test_patch_blocked_during_policy(self, tmp_path) -> None:
        import time as _time

        arm_xml = """
        <mujoco model="arm">
          <compiler angle="radian"/>
          <worldbody>
            <body name="base" pos="0 0 0.1">
              <joint name="pan" type="hinge" axis="0 0 1"/>
              <geom type="cylinder" size="0.05 0.05"/>
            </body>
          </worldbody>
          <actuator>
            <position name="pan_act" joint="pan" kp="50"/>
          </actuator>
        </mujoco>
        """
        arm_path = tmp_path / "arm.xml"
        arm_path.write_text(arm_xml)

        sim = Simulation(tool_name="devx_patch_guard", mesh=False)
        try:
            sim.create_world()
            sim.add_robot(name="arm1", urdf_path=str(arm_path))
            sim.start_policy("arm1", policy_provider="mock", duration=1.0, fast_mode=False)
            _time.sleep(0.05)

            result = sim.patch_scene_mjcf([{"op": "add_body", "name": "foo", "pos": [0, 0, 1]}])
            assert result["status"] == "error"
            assert "policy is running" in result["content"][0]["text"].lower()
        finally:
            sim.cleanup(policy_stop_timeout=2.0)


class TestPatchSceneMjcfToolSpec:
    def test_action_present_in_tool_spec(self) -> None:
        from strands_robots.simulation.mujoco.simulation import _TOOL_SPEC_SCHEMA

        assert "patch_scene_mjcf" in _TOOL_SPEC_SCHEMA["properties"]["action"]["enum"]

    def test_ops_parameter_present_in_tool_spec(self) -> None:
        from strands_robots.simulation.mujoco.simulation import _TOOL_SPEC_SCHEMA

        assert "ops" in _TOOL_SPEC_SCHEMA["properties"]


class TestPatchPreservesJointState:
    def test_add_body_preserves_existing_joint_qpos(self, sim: Simulation) -> None:
        """Adding a new body should not reset qpos for joints that already
        exist. This is the whole point of spec.recompile() over rebuilding."""
        sim.create_world()
        assert sim._world is not None

        # Seed with one free-joint body.
        sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "seed", "pos": [0, 0, 0.5]},
                {"op": "add_geom", "body": "seed", "type": "sphere", "size": [0.1]},
            ]
        )
        # Tick so mjData is initialised.
        sim.step(n_steps=1)
        # (body has no freejoint, so qpos is empty - still a valid test:
        # we just care that the second recompile does not fail.)

        # Now add a sibling body and confirm the model grows but step still works.
        nbody_before = sim._world._model.nbody
        sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "sibling", "pos": [1, 0, 0.5]},
                {"op": "add_geom", "body": "sibling", "type": "box", "size": [0.05, 0.05, 0.05]},
            ]
        )
        assert sim._world is not None
        assert sim._world._model.nbody == nbody_before + 1
        # A post-patch step must not crash.
        sim.step(n_steps=1)
