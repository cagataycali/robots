"""Integration tests for ``replace_scene_mjcf`` - the Stage-6 agent-authored
MJCF escape hatch (GH #125).

``replace_scene_mjcf`` lets an agent write raw MJCF that includes elements
``SimObject`` / ``SimCamera`` / ``SimRobot`` can't express - ``<tendon>``,
``<equality>``, ``<pair>``, custom contact friction, sites, etc. MuJoCo
validates by actually compiling the XML via ``mujoco.MjSpec.from_string``
+ ``spec.compile()``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")


from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Fixtures


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_replace", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


# Happy-path tests


class TestReplaceSceneMjcf:
    def test_replace_with_simple_scene(self, sim: Simulation) -> None:
        """A minimal valid MJCF string should be accepted and installed."""
        sim.create_world()

        xml = """
        <mujoco model="agent_scene">
          <option timestep="0.002" gravity="0 0 -9.81"/>
          <worldbody>
            <body name="widget" pos="0 0 0.2">
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """

        result = sim.replace_scene_mjcf(xml)
        assert result["status"] == "success", result
        text = result["content"][0]["text"]
        assert "Bodies:" in text
        assert "Scene replaced via raw MJCF" in text

        # Body 'widget' must now exist in the compiled model.
        mj = sim._mj
        assert sim._world is not None
        bid = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, "widget")
        assert bid >= 0

    def test_replace_with_tendon_element(self, sim: Simulation) -> None:
        """MJCF with a <tendon> element - unexpressible via SimObject - must
        compile cleanly. Proves the escape hatch delivers its value."""
        sim.create_world()

        xml = """
        <mujoco model="tendon_scene">
          <option timestep="0.002"/>
          <worldbody>
            <site name="s1" pos="0 0 0.5"/>
            <site name="s2" pos="1 0 0.5"/>
          </worldbody>
          <tendon>
            <spatial name="rope" width="0.01" rgba="1 0 0 1">
              <site site="s1"/>
              <site site="s2"/>
            </spatial>
          </tendon>
        </mujoco>
        """

        result = sim.replace_scene_mjcf(xml)
        assert result["status"] == "success", result
        mj = sim._mj
        assert sim._world is not None
        tid = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_TENDON, "rope")
        assert tid >= 0, "tendon must be present in compiled model"

    def test_replace_updates_backend_state_spec(self, sim: Simulation) -> None:
        sim.create_world()

        assert sim._world is not None
        old_spec = sim._world._backend_state.get("spec")
        assert old_spec is not None

        result = sim.replace_scene_mjcf("<mujoco><worldbody/></mujoco>")
        assert result["status"] == "success"

        assert sim._world is not None
        new_spec = sim._world._backend_state.get("spec")
        assert new_spec is not None
        assert new_spec is not old_spec, "spec should have been replaced"

    def test_replace_preserves_running_sim_time(self, sim: Simulation) -> None:
        """Replacing the scene does NOT keep qpos/qvel - this is documented
        in the method docstring. But sim_time / step_count are SimWorld-level
        Python state, and they reset naturally because a fresh MjData starts
        at t=0.
        """
        sim.create_world()
        # Step the default world so sim_time > 0.
        for _ in range(5):
            sim.step(n_steps=1)
        assert sim._world is not None
        assert sim._world.sim_time > 0

        result = sim.replace_scene_mjcf("<mujoco><worldbody/></mujoco>")
        assert result["status"] == "success"
        # Fresh MjData -> sim time for the NEW world is 0.
        assert sim._world is not None
        assert sim._world._data is not None
        assert sim._world._data.time == 0.0


# Error paths


class TestReplaceSceneMjcfErrors:
    def test_no_world_errors(self) -> None:
        """replace_scene_mjcf before create_world must error uniformly."""
        sim = Simulation(tool_name="devx_replace_nw", mesh=False)
        try:
            result = sim.replace_scene_mjcf("<mujoco><worldbody/></mujoco>")
            assert result["status"] == "error"
            assert "no world" in result["content"][0]["text"].lower()
        finally:
            sim.cleanup(policy_stop_timeout=0.5)

    def test_invalid_xml_returns_error(self, sim: Simulation) -> None:
        """MuJoCo's compiler error must flow back to the caller - not crash
        the process."""
        sim.create_world()
        result = sim.replace_scene_mjcf("not really xml at all")
        assert result["status"] == "error"
        assert "compile failed" in result["content"][0]["text"].lower()

    def test_semantically_invalid_mjcf_returns_error(self, sim: Simulation) -> None:
        """Well-formed XML that's NOT valid MJCF (e.g. geom with unknown type)
        must also return an error."""
        sim.create_world()
        xml = '<mujoco><worldbody><body name="bad"><geom type="NOT_A_VALID_TYPE" size="1"/></body></worldbody></mujoco>'
        result = sim.replace_scene_mjcf(xml)
        assert result["status"] == "error"

    def test_replace_scene_blocked_during_policy(self, tmp_path) -> None:
        """Global-scope mutation - replace_scene_mjcf must be blocked while
        any policy is running. Mirrors the load_scene / add_robot guard."""
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

        sim = Simulation(tool_name="devx_replace_guard", mesh=False)
        try:
            sim.create_world()
            sim.add_robot(name="arm1", urdf_path=str(arm_path))
            sim.start_policy("arm1", policy_provider="mock", duration=1.0, fast_mode=False)
            _time.sleep(0.05)

            result = sim.replace_scene_mjcf("<mujoco><worldbody/></mujoco>")
            assert result["status"] == "error"
            assert "policy is running" in result["content"][0]["text"].lower()
        finally:
            sim.cleanup(policy_stop_timeout=2.0)


# tool_spec integration


class TestReplaceSceneMjcfToolSpec:
    def test_action_present_in_tool_spec(self) -> None:
        """The tool_spec must advertise the new action so LLMs can discover it."""
        from strands_robots.simulation.mujoco.simulation import _TOOL_SPEC_SCHEMA

        enum = _TOOL_SPEC_SCHEMA["properties"]["action"]["enum"]
        assert "replace_scene_mjcf" in enum

    def test_xml_parameter_present_in_tool_spec(self) -> None:
        from strands_robots.simulation.mujoco.simulation import _TOOL_SPEC_SCHEMA

        assert "xml" in _TOOL_SPEC_SCHEMA["properties"]
