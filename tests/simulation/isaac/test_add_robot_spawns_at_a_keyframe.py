"""``add_robot(keyframe=...)`` on Isaac reads the pose from the MJCF ``<keyframe>``.

The MuJoCo backend spawns a robot at a named keyframe (the Franka ``home``
pose, a humanoid's standing pose). Isaac used to refuse every keyframe, so a
policy trained from that pose started from the zero configuration instead. The
pose is now resolved with MuJoCo's own ``mj_resetDataKeyframe`` and becomes the
robot's default joint state and drive targets; these cells pin the resolver.
"""

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.isaac.joint_names import mjcf_keyframe_joint_positions

_MJCF = """
<mujoco>
  <worldbody>
    <body name="base">
      <freejoint name="root"/>
      <geom type="box" size=".1 .1 .1"/>
      <body name="link1">
        <joint name="j1" type="hinge"/>
        <geom type="capsule" size=".02" fromto="0 0 0 0 0 .2"/>
        <body name="link2">
          <joint name="slide" type="slide"/>
          <geom type="box" size=".02 .02 .02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="0 0 1 1 0 0 0 0.5 0.03"/>
    <key name="bent" qpos="0 0 1 1 0 0 0 -1.2 0"/>
  </keyframe>
</mujoco>
"""


@pytest.fixture
def mjcf(tmp_path):
    path = tmp_path / "robot.xml"
    path.write_text(_MJCF)
    return str(path)


def test_a_named_keyframe_gives_each_hinge_and_slide_joint_its_position(mjcf):
    pose, error = mjcf_keyframe_joint_positions(mjcf, "home")
    assert error is None
    # The free joint is the base pose, not a DOF; it is left out.
    assert pose == pytest.approx({"j1": 0.5, "slide": 0.03})


def test_a_keyframe_index_resolves_like_mujoco(mjcf):
    pose, error = mjcf_keyframe_joint_positions(mjcf, 1)
    assert error is None
    assert pose["j1"] == pytest.approx(-1.2)


def test_an_unknown_keyframe_names_the_ones_the_model_declares(mjcf):
    pose, error = mjcf_keyframe_joint_positions(mjcf, "crouch")
    assert pose is None
    assert "'crouch'" in error and "home" in error and "bent" in error


def test_a_model_without_keyframes_says_so(tmp_path):
    path = tmp_path / "bare.xml"
    path.write_text('<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody></mujoco>')
    pose, error = mjcf_keyframe_joint_positions(str(path), "home")
    assert pose is None and "declares no <keyframe>" in error


@pytest.mark.parametrize("bad", [True, 1.5, None])
def test_a_keyframe_that_is_not_a_name_or_index_is_refused(mjcf, bad):
    pose, error = mjcf_keyframe_joint_positions(mjcf, bad)
    assert pose is None and "name (str) or index (int)" in error


class TestResetReturnsToTheKeyframe:
    """``reset()`` returns a keyframe-spawned robot to its keyframe, also on the first reset after
    a dynamic object was added: that ``add_object`` marks the tensor view stale, ``world.reset()``
    rebuilds the real view, and the restore must run once the bookkeeping flag agrees (the drive
    targets would otherwise pull the arm back toward zero, and ``reset()`` would still say success)."""

    @staticmethod
    def _engine_with_a_spawned_robot(applied: list) -> tuple[object, object]:
        import types

        pytest.importorskip("strands_robots.simulation.isaac")
        from strands_robots.simulation.isaac.simulation import _RobotState
        from tests.simulation._isaac_engine import isaac_engine

        engine = isaac_engine()
        engine._world_created = True
        engine._world = types.SimpleNamespace(
            reset=lambda: None, stop=lambda: None, clear_instance=lambda: None, step=lambda **k: None
        )
        engine._obs_noise = {}
        engine._revive_articulations_after_reset = lambda: None  # type: ignore[method-assign]
        engine._flush_open_episode_before_reset = lambda: None  # type: ignore[method-assign]
        robot = _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=["j0", "j1"])
        robot.spawn_joint_positions = {"j0": 0.5, "j1": -0.25}
        robot.articulation = types.SimpleNamespace(
            set_joint_positions=lambda v: applied.append(("pos", [float(x) for x in v])),
            set_joint_velocities=lambda v: applied.append(("vel", [float(x) for x in v])),
            apply_action=lambda action: applied.append(("target", action)),
        )
        engine._robots = {"arm": robot}
        return engine, robot

    def test_the_first_reset_after_a_dynamic_object_restores_the_keyframe(self):
        applied: list = []
        engine, _robot = self._engine_with_a_spawned_robot(applied)
        engine._physics_view_stale = True  # what add_object(is_static=False) leaves behind

        result = engine.reset()

        assert result["status"] == "success"
        assert ("pos", [0.5, -0.25]) in applied, f"the keyframe was not restored on this reset: {applied}"
        assert engine._physics_view_stale is False
