"""``add_robot(keyframe=...)`` spawns a robot in a source ``<keyframe>`` pose.

MuJoCo Menagerie robots ship a canonical home pose in a ``<keyframe>`` (panda
``home``, ur5e/fr3/kuka ``home``, aloha ``neutral_pose``, quadrupeds/humanoids
a standing ``home``). ``add_robot`` historically ran ``mj_resetData`` (the
all-zero configuration) and ``reset()`` does the same, so that shipped home
pose was unreachable outside the LIBERO benchmark adapter. A policy trained
from the home pose then sees an out-of-distribution start (a folded/collapsed
arm), which measurably degrades its rollout.

``add_robot(keyframe=...)`` applies the named/indexed keyframe's qpos to the
robot's joints by name at spawn and stores it so ``reset()`` restores it (a
keyframe spawn is sticky across resets, matching how a benchmark restores its
canonical start each episode). ``keyframe=None`` (the default) keeps the
historical zero-pose spawn byte-for-byte.

A ``<key>`` declares ``qpos`` and ``ctrl`` as a matched pair -- the ctrl are the
position-servo setpoints authored to HOLD that pose against gravity, which is
why ``mj_resetDataKeyframe`` (MuJoCo's own way to apply a keyframe) sets both.
Applying the pose alone leaves the servos commanded to the zero configuration,
so the robot is placed at its home pose and then immediately driven off it; the
setpoints therefore travel with the pose at spawn and across ``reset()``. A
keyframe that declares no ctrl (or an all-zero row, which MuJoCo also produces
for a ``<key>`` that names no ctrl at all) leaves the setpoints untouched.

These tests use a tiny inline two-hinge MJCF with a ``<keyframe>`` so they run
offline and GL-free in CI (no mesh download, no render).
"""

from __future__ import annotations

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Two hinge joints; the ``home`` keyframe bends them to a non-zero pose that is
# distinct from the all-zero default and from what gravity/servos settle to.
_ARM_MJCF = """
<mujoco model="kf_arm">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="l1" pos="0 0 0.1">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03" mass="1"/>
      <body name="l2" pos="0 0 0.3">
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="s_act" joint="shoulder" kp="50"/>
    <position name="e_act" joint="elbow" kp="50"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0.5 -1.2"/>
  </keyframe>
</mujoco>
"""

_HOME = [0.5, -1.2]


def _servo_arm_mjcf(ctrl_attr: str) -> str:
    """A gravity-loaded two-hinge arm whose ``home`` key carries ``ctrl_attr``.

    The links are HORIZONTAL so gravity has full leverage on both joints: an
    unheld pose collapses by radians, which is what makes "the servos hold the
    keyed pose" a sharp assertion rather than a numerical squint. ``kp=200``
    with joint damping settles on the setpoint instead of ringing, so a failure
    means the setpoint is wrong -- not that the servo was too soft to arrive.
    """
    return f"""
<mujoco model="kf_servo_arm">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="l1" pos="0 0 0.4">
      <joint name="shoulder" type="hinge" axis="0 1 0" damping="1.0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
      <body name="l2" pos="0.3 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="1.0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="s_act" joint="shoulder" kp="200"/>
    <position name="e_act" joint="elbow" kp="200"/>
  </actuator>
  <keyframe><key name="home" qpos="0.6 -1.1"{ctrl_attr}/></keyframe>
</mujoco>
"""


# The keyed pose, and the setpoints the same key pairs with it.
_SERVO_HOME = [0.6, -1.1]


@pytest.fixture
def servo_arm_xml(tmp_path):
    """Arm whose ``home`` key declares ctrl matched to its qpos."""
    p = tmp_path / "kf_servo_arm.xml"
    p.write_text(_servo_arm_mjcf(' ctrl="0.6 -1.1"'))
    return str(p)


@pytest.fixture
def servo_arm_no_ctrl_xml(tmp_path):
    """Same arm, but the ``home`` key declares no ctrl at all."""
    p = tmp_path / "kf_servo_arm_no_ctrl.xml"
    p.write_text(_servo_arm_mjcf(""))
    return str(p)


@pytest.fixture
def servo_arm_zero_ctrl_xml(tmp_path):
    """Same arm, but the ``home`` key spells its ctrl as an all-zero row."""
    p = tmp_path / "kf_servo_arm_zero_ctrl.xml"
    p.write_text(_servo_arm_mjcf(' ctrl="0 0"'))
    return str(p)


@pytest.fixture
def arm_xml(tmp_path):
    p = tmp_path / "kf_arm.xml"
    p.write_text(_ARM_MJCF)
    return str(p)


# A structurally-valid arm with NO ``<keyframe>`` block, to exercise the
# "model declares no keyframe" error contract.
_NO_KEYFRAME_MJCF = """
<mujoco model="no_kf_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="l1" pos="0 0 0.1">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Malformed MJCF (unclosed element) so ``MjModel.from_xml_path`` raises while
# reading the source model, exercising the "cannot read keyframe" branch.
_MALFORMED_MJCF = "<mujoco><worldbody><body><joint name='j' type='hinge'/></body</mujoco>"


@pytest.fixture
def no_keyframe_xml(tmp_path):
    p = tmp_path / "no_kf_arm.xml"
    p.write_text(_NO_KEYFRAME_MJCF)
    return str(p)


@pytest.fixture
def malformed_xml(tmp_path):
    p = tmp_path / "malformed_arm.xml"
    p.write_text(_MALFORMED_MJCF)
    return str(p)


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_add_robot_keyframe", mesh=False)
    s.create_world()
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _qpos(sim):
    return sim._world._data.qpos.copy()


class TestAddRobotKeyframe:
    def test_default_spawn_is_zero_pose(self, sim, arm_xml):
        # No keyframe -> historical all-zero spawn, and no home pose captured.
        sim.add_robot(name="a", urdf_path=arm_xml)
        assert np.allclose(_qpos(sim), [0.0, 0.0])
        assert sim._world.robots["a"].home_qpos == {}

    def test_keyframe_by_name_applies_home_pose(self, sim, arm_xml):
        result = sim.add_robot(name="a", urdf_path=arm_xml, keyframe="home")
        assert result["status"] == "success"
        assert np.allclose(_qpos(sim), _HOME)
        # Home pose captured under the namespaced joint names for reset().
        assert sim._world.robots["a"].home_qpos == {
            "a/shoulder": [0.5],
            "a/elbow": [-1.2],
        }

    def test_keyframe_by_index_applies_home_pose(self, sim, arm_xml):
        result = sim.add_robot(name="a", urdf_path=arm_xml, keyframe=0)
        assert result["status"] == "success"
        assert np.allclose(_qpos(sim), _HOME)

    def test_reset_restores_keyframe_home_pose(self, sim, arm_xml):
        sim.add_robot(name="a", urdf_path=arm_xml, keyframe="home")
        # Drive the arm off the home pose, then reset.
        sim.step(40)
        assert not np.allclose(_qpos(sim), _HOME)
        reset_result = sim.reset()
        assert reset_result["status"] == "success"
        # reset() must restore the keyframe home pose, not collapse to zeros.
        assert np.allclose(_qpos(sim), _HOME)

    def test_reset_without_keyframe_stays_zero(self, sim, arm_xml):
        # Guard the no-regression path: a robot added without a keyframe must
        # reset to the zero configuration exactly as before.
        sim.add_robot(name="a", urdf_path=arm_xml)
        sim.step(40)
        sim.reset()
        assert np.allclose(_qpos(sim), [0.0, 0.0])

    def test_unknown_keyframe_errors_and_leaks_nothing(self, sim, arm_xml):
        result = sim.add_robot(name="a", urdf_path=arm_xml, keyframe="does_not_exist")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "does_not_exist" in text
        # Names the available keyframe so the caller can fix the call.
        assert "'home'" in text
        # No half-added robot left behind; the same name is reusable.
        assert "a" not in sim._world.robots
        ok = sim.add_robot(name="a", urdf_path=arm_xml, keyframe="home")
        assert ok["status"] == "success"

    def test_bool_keyframe_rejected(self, sim, arm_xml):
        # bool is an int subclass; True/False must not be taken as index 1/0.
        result = sim.add_robot(name="a", urdf_path=arm_xml, keyframe=True)
        assert result["status"] == "error"
        assert "bool" in result["content"][0]["text"]
        assert "a" not in sim._world.robots

    def test_keyframe_via_tool_router(self, sim, arm_xml):
        # The agent-facing dispatch path forwards the keyframe param.
        result = sim._dispatch_action(
            "add_robot",
            {"action": "add_robot", "name": "a", "urdf_path": arm_xml, "keyframe": "home"},
        )
        assert result["status"] == "success"
        assert np.allclose(_qpos(sim), _HOME)

    @pytest.mark.parametrize("bad_index", [5, -1])
    def test_out_of_range_index_errors_and_leaks_nothing(self, sim, arm_xml, bad_index):
        # An integer index outside [0, nkey) must fail cleanly, naming the
        # keyframe count and available names so the caller can correct it, and
        # must not leave a half-added robot behind. Negative indices are
        # rejected too (they are not Python-style "from the end" here).
        result = sim.add_robot(name="a", urdf_path=arm_xml, keyframe=bad_index)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert f"keyframe index {bad_index} out of range" in text
        # The single available keyframe is named to make the error actionable.
        assert "1 keyframe(s)" in text
        assert "'home'" in text
        assert "a" not in sim._world.robots
        # The name is reusable after the rejected add.
        assert sim.add_robot(name="a", urdf_path=arm_xml, keyframe="home")["status"] == "success"

    def test_model_without_keyframe_errors(self, sim, no_keyframe_xml):
        # Requesting a keyframe from a model that declares none must surface a
        # clear error (naming the requested keyframe) rather than silently
        # spawning at the zero pose.
        result = sim.add_robot(name="a", urdf_path=no_keyframe_xml, keyframe="home")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "declares no <keyframe>" in text
        assert "keyframe='home'" in text
        assert "a" not in sim._world.robots

    def test_unreadable_source_model_errors(self, sim, malformed_xml):
        # If the source model cannot even be compiled to read its keyframes,
        # the failure is surfaced (naming the file) instead of raising an
        # opaque exception up through add_robot.
        result = sim.add_robot(name="a", urdf_path=malformed_xml, keyframe="home")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "Cannot read keyframe from" in text
        assert "malformed_arm.xml" in text
        assert "a" not in sim._world.robots


# A floating-base robot whose qpos mixes all three MuJoCo joint widths in one
# model: a free root (7 = 3 translation + 4 quaternion), a ball joint (4 = a
# quaternion), and a hinge (1). Humanoids and quadrupeds (unitree g1/go2/h1,
# etc.) spawn exactly this way - a free base plus articulated joints - and ship
# their standing pose in a ``<keyframe>``. The keyframe home pose must be
# sliced out of the flat qpos vector using the correct per-joint width, or the
# base position/orientation and every downstream joint land in the wrong slot.
_FLOAT_MJCF = """
<mujoco model="kf_float">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <freejoint name="root"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="link1" pos="0.1 0 0">
        <joint name="ball1" type="ball"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02" mass="0.2"/>
        <body name="link2" pos="0.1 0 0">
          <joint name="hinge1" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="0 0 0.5 1 0 0 0  0.7071 0 0.7071 0  0.3"/>
  </keyframe>
</mujoco>
"""

# free(7): x y z + wxyz quat | ball(4): wxyz quat | hinge(1): angle
_FLOAT_HOME = [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.7071, 0.0, 0.7071, 0.0, 0.3]


@pytest.fixture
def float_xml(tmp_path):
    p = tmp_path / "kf_float.xml"
    p.write_text(_FLOAT_MJCF)
    return str(p)


class TestFloatingBaseKeyframe:
    """Keyframe spawn must slice the home pose with the correct per-joint qpos
    width for every joint type - free (7) and ball (4), not just hinge/slide
    (1). This is the floating-base humanoid/quadruped spawn path.
    """

    def test_free_and_ball_joint_home_pose_applied(self, sim, float_xml):
        result = sim.add_robot(name="fb", urdf_path=float_xml, keyframe="home")
        assert result["status"] == "success"
        # The flat qpos vector is the free base pose, the ball quaternion, and
        # the hinge angle laid out in joint order - proving each was written to
        # its own slice at the right width rather than bleeding into the next.
        assert np.allclose(_qpos(sim), _FLOAT_HOME)

    def test_home_pose_captured_with_correct_per_joint_widths(self, sim, float_xml):
        sim.add_robot(name="fb", urdf_path=float_xml, keyframe="home")
        home = sim._world.robots["fb"].home_qpos
        # Namespaced joint names, each carrying exactly its joint-type width:
        # a wrong width here (e.g. treating the free root as a 1-wide slide)
        # would truncate the base pose and shift every later joint.
        assert set(home) == {"fb/root", "fb/ball1", "fb/hinge1"}
        assert len(home["fb/root"]) == 7
        assert len(home["fb/ball1"]) == 4
        assert len(home["fb/hinge1"]) == 1
        assert np.allclose(home["fb/root"], [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
        assert np.allclose(home["fb/ball1"], [0.7071, 0.0, 0.7071, 0.0])
        assert np.allclose(home["fb/hinge1"], [0.3])

    def test_observation_reflects_free_base_keyframe_pose(self, sim, float_xml):
        sim.add_robot(name="fb", urdf_path=float_xml, keyframe="home")
        obs = sim.get_observation(robot_name="fb", skip_images=True)
        # The free base spawns at the keyframe height/orientation, not the
        # zero-pose origin.
        assert np.allclose(obs["base_pos"], [0.0, 0.0, 0.5])
        assert np.allclose(obs["base_quat"], [1.0, 0.0, 0.0, 0.0])

    def test_reset_restores_floating_base_home_pose(self, sim, float_xml):
        sim.add_robot(name="fb", urdf_path=float_xml, keyframe="home")
        # Gravity pulls the un-actuated free base off the home pose.
        sim.step(40)
        assert not np.allclose(_qpos(sim), _FLOAT_HOME)
        assert sim.reset()["status"] == "success"
        assert np.allclose(_qpos(sim), _FLOAT_HOME)


def _joint_qpos(sim, joint_name: str) -> float:
    """Read the scalar qpos of a named (namespaced) hinge joint from the live
    compiled model - proves the pose landed on the intended robot's joint."""
    model = sim._world._model
    data = sim._world._data
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    assert jid >= 0, f"joint {joint_name!r} not found in compiled model"
    adr = int(model.jnt_qposadr[jid])
    return float(data.qpos[adr])


class TestMultiRobotKeyframeSpawn:
    """Incrementally adding robots must not disturb an already-spawned robot's
    keyframe home pose.

    ``add_robot`` runs ``mj_resetData`` (which zeroes the whole model) before
    posing the freshly-added robot. When an earlier robot was spawned from a
    ``<keyframe>``, that reset would silently collapse it back to the zero
    configuration - only the most recently added robot kept its home pose until
    an unrelated ``reset()`` happened to restore everyone. Building a multi-arm
    scene one ``add_robot`` at a time is the common path (e.g. a leader/follower
    pair), so each robot must stay at its canonical home pose across subsequent
    additions.
    """

    def test_adding_second_robot_preserves_first_home_pose(self, sim, arm_xml):
        # Spawn A from its keyframe: it lands at the home pose.
        sim.add_robot(name="a", urdf_path=arm_xml, keyframe="home")
        assert np.isclose(_joint_qpos(sim, "a/shoulder"), _HOME[0])
        assert np.isclose(_joint_qpos(sim, "a/elbow"), _HOME[1])

        # Add a SECOND keyframed robot. A's home pose must survive - the
        # regression: pre-fix, mj_resetData zeroed A and only B was re-posed.
        result = sim.add_robot(name="b", urdf_path=arm_xml, keyframe="home")
        assert result["status"] == "success"
        assert np.isclose(_joint_qpos(sim, "a/shoulder"), _HOME[0])
        assert np.isclose(_joint_qpos(sim, "a/elbow"), _HOME[1])
        assert np.isclose(_joint_qpos(sim, "b/shoulder"), _HOME[0])
        assert np.isclose(_joint_qpos(sim, "b/elbow"), _HOME[1])

    def test_keyframe_spawn_is_scoped_to_its_own_robot(self, sim, arm_xml):
        # A added WITHOUT a keyframe stays at the zero configuration even when a
        # later robot B IS spawned from a keyframe with identical short joint
        # names ("shoulder"/"elbow"). The home pose must not bleed across the
        # namespace into A's identically-named joints.
        sim.add_robot(name="a", urdf_path=arm_xml)
        sim.add_robot(name="b", urdf_path=arm_xml, keyframe="home")
        assert np.isclose(_joint_qpos(sim, "a/shoulder"), 0.0)
        assert np.isclose(_joint_qpos(sim, "a/elbow"), 0.0)
        assert np.isclose(_joint_qpos(sim, "b/shoulder"), _HOME[0])
        assert np.isclose(_joint_qpos(sim, "b/elbow"), _HOME[1])


class TestApplyHomeQposWidthGuard:
    """``_apply_home_state_to_robot`` must skip a home entry whose value width
    disagrees with the target joint's qpos width instead of writing a
    wrong-length slice.

    ``add_robot(keyframe=...)`` reads the home pose from the SAME model it
    writes it back onto, so the per-joint widths always agree and the guard is
    unreachable through the public spawn path. It exists as a defensive backstop
    for callers that hand in an externally-sourced pose (e.g. a benchmark
    reusing one robot's home pose on a structurally different model): a
    mismatched entry must be dropped, never sliced into ``qpos``, because a
    wrong-length assignment would either raise or spill into the adjacent
    joint's slice and silently corrupt its state. This exercises that guard
    directly.
    """

    def test_width_mismatch_entry_skipped_without_corrupting_neighbor(self, sim, arm_xml):
        # A robot with no keyframe -> both joints start at the zero pose.
        sim.add_robot(name="a", urdf_path=arm_xml)
        robot = sim._world.robots["a"]
        model = sim._world._model
        data = sim._world._data

        sh_adr = int(model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "a/shoulder")])
        el_adr = int(model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "a/elbow")])

        # "shoulder" is a 1-wide hinge but the supplied home value has width 2;
        # "elbow" is correctly sized. Keys are the short (namespace-stripped)
        # joint names the method matches against.
        from strands_robots.simulation.mujoco.simulation import _KeyframeHome

        home = _KeyframeHome(qpos={"shoulder": [0.1, 0.2], "elbow": [0.9]}, ctrl={})
        sim._apply_home_state_to_robot(robot, home)

        # The mismatched shoulder entry is dropped: its qpos slice is untouched
        # (still the zero pose) and the extra value did NOT spill into the
        # adjacent elbow slice.
        assert data.qpos[sh_adr] == 0.0
        assert data.qpos[el_adr] == 0.9
        # Only the correctly-sized joint is recorded for reset() to restore.
        assert robot.home_qpos == {"a/elbow": [0.9]}


def _ctrl(sim):
    return sim._world._data.ctrl.copy()


def _step(sim, n):
    for _ in range(n):
        mj.mj_step(sim._world._model, sim._world._data)


class TestKeyframeCtrlHoldsTheKeyedPose:
    """A keyframe's ``ctrl`` travels with its ``qpos``, at spawn and on reset.

    The keyed setpoints are the whole reason a ``<key>`` is a usable home pose:
    they are what holds it up. Restoring the pose without them leaves the
    servos commanded to the zero configuration, so the arm is placed at home
    and then driven off it -- and for an eval loop that resets between
    episodes, that happens before the policy's first observation of every
    episode.
    """

    def test_spawn_applies_the_setpoints_paired_with_the_pose(self, sim, servo_arm_xml):
        assert sim.add_robot(name="a", urdf_path=servo_arm_xml, keyframe="home")["status"] == "success"
        assert np.allclose(_qpos(sim), _SERVO_HOME)
        assert np.allclose(_ctrl(sim), _SERVO_HOME), (
            f"spawn left ctrl at {_ctrl(sim).tolist()} while the keyframe pairs "
            f"{_SERVO_HOME} with the pose it just applied"
        )

    def test_the_servos_hold_the_keyed_pose_after_spawn(self, sim, servo_arm_xml):
        # The observable that matters: step and stay. With the setpoints dropped
        # the servos drive toward zero and this gravity-loaded arm collapses.
        sim.add_robot(name="a", urdf_path=servo_arm_xml, keyframe="home")
        home = _qpos(sim)
        assert np.allclose(home, _SERVO_HOME), "premise: spawn must be at the keyed pose"
        _step(sim, 400)
        drift = float(np.max(np.abs(_qpos(sim) - home)))
        assert drift < 0.05, (
            f"the arm moved {drift:.4f} rad off its keyed home pose "
            f"(now {_qpos(sim).tolist()}, home {home.tolist()}) - the keyframe's "
            f"setpoints are not holding it"
        )

    def test_reset_restores_the_setpoints_not_only_the_pose(self, sim, servo_arm_xml):
        sim.add_robot(name="a", urdf_path=servo_arm_xml, keyframe="home")
        # Command the arm somewhere else and let it get there, so the restore
        # has something real to undo.
        sim._world._data.ctrl[:] = [0.0, 0.0]
        _step(sim, 400)
        assert not np.allclose(_qpos(sim), _SERVO_HOME, atol=0.05), "premise: the arm must have left home"
        assert sim.reset()["status"] == "success"
        assert np.allclose(_qpos(sim), _SERVO_HOME)
        assert np.allclose(_ctrl(sim), _SERVO_HOME), (
            f"reset() restored the pose but left ctrl at {_ctrl(sim).tolist()}; "
            f"reset() documents that the keyframe home state is restored"
        )

    def test_the_servos_hold_the_keyed_pose_after_reset(self, sim, servo_arm_xml):
        sim.add_robot(name="a", urdf_path=servo_arm_xml, keyframe="home")
        sim.reset()
        home = _qpos(sim)
        assert np.allclose(home, _SERVO_HOME), "premise: reset must land on the keyed pose"
        _step(sim, 400)
        drift = float(np.max(np.abs(_qpos(sim) - home)))
        assert drift < 0.05, (
            f"after reset() the arm moved {drift:.4f} rad off home - every episode "
            f"of a reset-between-episodes eval loop starts already falling"
        )

    def test_setpoints_are_recorded_under_namespaced_actuator_names(self, sim, servo_arm_xml):
        # Keyed by the merged scene's names, which is what reset() looks up.
        sim.add_robot(name="a", urdf_path=servo_arm_xml, keyframe="home")
        assert sim._world.robots["a"].home_ctrl == {"a/s_act": 0.6, "a/e_act": -1.1}

    def test_a_second_robots_reset_does_not_disturb_the_first(self, sim, servo_arm_xml):
        # Restoring is scoped to the actuators the robot's own keyframe named,
        # so a keyframed arm and a plain one coexist without clobbering.
        sim.add_robot(name="a", urdf_path=servo_arm_xml, keyframe="home")
        sim.add_robot(name="b", urdf_path=servo_arm_xml, position=[1.0, 0.0, 0.0])
        sim.reset()
        model, data = sim._world._model, sim._world._data
        by_name = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i): float(data.ctrl[i]) for i in range(int(model.nu))}
        assert by_name["a/s_act"] == pytest.approx(0.6)
        assert by_name["a/e_act"] == pytest.approx(-1.1)
        # b spawned without a keyframe: mj_resetData zeroed it and nothing
        # re-asserted a setpoint it never asked for.
        assert by_name["b/s_act"] == pytest.approx(0.0)
        assert by_name["b/e_act"] == pytest.approx(0.0)


class TestAKeyframeWithoutCtrlIsLeftAlone:
    """A ``<key>`` that names no ctrl keeps the historical setpoint-free spawn.

    These pass both before and after the setpoint restore: they are the bound
    on it. Six of the registry's forty shipped keyframes carry an all-zero ctrl
    row -- which is also exactly what MuJoCo materializes for a ``<key>`` that
    names no ctrl at all -- so the two cases are indistinguishable in the model
    and must behave the same.
    """

    def test_a_key_declaring_no_ctrl_leaves_the_setpoints_at_zero(self, sim, servo_arm_no_ctrl_xml):
        sim.add_robot(name="a", urdf_path=servo_arm_no_ctrl_xml, keyframe="home")
        assert np.allclose(_qpos(sim), _SERVO_HOME)
        assert np.allclose(_ctrl(sim), [0.0, 0.0])

    def test_an_all_zero_ctrl_row_leaves_the_setpoints_at_zero(self, sim, servo_arm_zero_ctrl_xml):
        sim.add_robot(name="a", urdf_path=servo_arm_zero_ctrl_xml, keyframe="home")
        assert np.allclose(_qpos(sim), _SERVO_HOME)
        assert np.allclose(_ctrl(sim), [0.0, 0.0])

    def test_a_spawn_without_a_keyframe_leaves_the_setpoints_at_zero(self, sim, servo_arm_xml):
        sim.add_robot(name="a", urdf_path=servo_arm_xml)
        assert np.allclose(_ctrl(sim), [0.0, 0.0])

    def test_reset_leaves_a_manually_commanded_unkeyed_robot_to_mj_reset_data(self, sim, servo_arm_xml):
        # Without a keyframe there is no recorded setpoint to re-assert, so
        # reset() must not invent one: mj_resetData's zero stands.
        sim.add_robot(name="a", urdf_path=servo_arm_xml)
        sim._world._data.ctrl[:] = [0.4, -0.4]
        sim.reset()
        assert np.allclose(_ctrl(sim), [0.0, 0.0])
