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
