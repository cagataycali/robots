"""The Newton backend keeps an MJCF's compiled servo damping and torque ceiling.

Newton's MJCF importer reads a ``<position>`` actuator's ``kp`` into
``joint_target_ke`` and drops the rest of the servo: the ``dampratio`` MuJoCo
compiles into a velocity gain (``-actuator_biasprm[2]``) and the ``forcerange``
that caps the torque. A P-only servo with a 1e6 ceiling does not track - on the
shipped SO-100 a constant ``Rotation = 0.5`` command oscillates between 0.05 and
0.96 rad on Newton where the MuJoCo backend settles on 0.4997 - so
:mod:`strands_robots.simulation.newton.actuator_gains` reads both off the
compiled model and writes them onto the builder before ``finalize``.

Newton and Warp are not needed: the reader is MuJoCo-only and the writer touches
nothing but builder lists, so a recording stub is enough and these pins run on
every CI job.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.newton.actuator_gains import (
    JointServo,
    apply_joint_servos,
    mjcf_joint_servos,
)
from strands_robots.simulation.newton.simulation import NewtonSimEngine

# Three actuator spellings of the same joint gain, one per contract:
# dampratio (compiled into kv), an explicit kv, and a plain motor (no servo).
_MJCF = """
<mujoco model="servos">
  <worldbody>
    <body name="link_a" pos="0 0 1">
      <joint name="ratio" type="hinge" axis="0 0 1" armature="0.1"/>
      <geom type="box" size="0.1 0.02 0.02" mass="0.5"/>
      <body name="link_b" pos="0.2 0 0">
        <joint name="explicit" type="hinge" axis="0 0 1" armature="0.1"/>
        <geom type="box" size="0.1 0.02 0.02" mass="0.5"/>
        <body name="link_c" pos="0.2 0 0">
          <joint name="plain" type="hinge" axis="0 0 1" armature="0.1"/>
          <geom type="box" size="0.1 0.02 0.02" mass="0.5"/>
          <body name="link_d" pos="0.2 0 0">
            <joint name="unactuated" type="hinge" axis="0 0 1" armature="0.1"/>
            <geom type="box" size="0.1 0.02 0.02" mass="0.5"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="ratio" joint="ratio" kp="50" dampratio="1" forcerange="-3.5 3.5"/>
    <position name="explicit" joint="explicit" kp="20" kv="7"/>
    <motor name="plain" joint="plain"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def servos(tmp_path):
    """The compiled servos of :data:`_MJCF`, keyed by joint name."""
    pytest.importorskip("mujoco")
    path = tmp_path / "servos.xml"
    path.write_text(_MJCF)
    return mjcf_joint_servos(str(path))


def test_only_actuated_joints_are_reported(servos):
    """An unactuated joint has no servo to report."""
    assert sorted(servos) == ["explicit", "plain", "ratio"]


@pytest.mark.parametrize(
    ("joint", "kp", "effort_limit"),
    [
        ("ratio", 50.0, 3.5),  # forcerange -> the usable symmetric magnitude
        ("explicit", 20.0, None),  # no forcerange -> no ceiling to state
        ("plain", 1.0, None),  # a motor's gainprm is its gear, not a servo
    ],
)
def test_gain_and_force_ceiling_are_read_per_actuator(servos, joint, kp, effort_limit):
    assert servos[joint].kp == pytest.approx(kp)
    assert servos[joint].effort_limit == effort_limit


def test_dampratio_is_compiled_into_a_positive_velocity_gain(servos):
    """``dampratio`` names no kv in the XML; MuJoCo expands it into one."""
    assert servos["ratio"].kd > 0.0


def test_an_explicit_kv_is_read_verbatim(servos):
    """The velocity gain is ``-biasprm[2]``, not the position gain's slot."""
    assert servos["explicit"].kd == pytest.approx(7.0)


def test_a_motor_has_no_velocity_gain(servos):
    """No bias means no servo: writing a kd for it would invent damping."""
    assert servos["plain"].kd == 0.0


class _RecordingBuilder:
    """Newton ``ModelBuilder`` stand-in holding just the arrays gains touch.

    The first joint is a 6-DoF free base, so a joint's DOF index is NOT its
    ordinal - the trap this stub exists to expose.
    """

    def __init__(self) -> None:
        self.joint_label = [
            "robot/floating_base_joint",
            "robot/base/ratio",
            "robot/base/link/explicit",
            "robot/base/link/unactuated",
        ]
        self.joint_qd_start = [0, 6, 7, 8]
        self.joint_target_ke = [0.0] * 6 + [50.0, 20.0, 0.0]
        self.joint_target_kd = [0.0] * 9
        self.joint_effort_limit = [1e6] * 9


def _short_name(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def test_gains_land_on_the_dof_index_not_the_joint_ordinal():
    builder = _RecordingBuilder()
    applied = apply_joint_servos(
        builder,
        {
            "ratio": JointServo(kp=50.0, kd=5.0, effort_limit=3.5),
            "explicit": JointServo(kp=20.0, kd=7.0, effort_limit=None),
        },
        builder.joint_label,
        first_joint=0,
        short_name=_short_name,
    )

    assert sorted(applied) == ["explicit", "ratio"]
    # DOFs 6 and 7 - the free base's six DOFs and the unactuated joint's are
    # left exactly as Newton imported them.
    assert builder.joint_target_kd == [0.0] * 6 + [5.0, 7.0, 0.0]
    assert builder.joint_effort_limit == [1e6] * 6 + [3.5, 1e6, 1e6]
    # kp is Newton's to carry; this writer must not restate it.
    assert builder.joint_target_ke == [0.0] * 6 + [50.0, 20.0, 0.0]


def test_a_second_robot_only_touches_its_own_joints():
    """``first_joint`` scopes the write to the robot just imported."""
    builder = _RecordingBuilder()
    applied = apply_joint_servos(
        builder,
        {"ratio": JointServo(kp=50.0, kd=5.0, effort_limit=3.5)},
        builder.joint_label,
        first_joint=2,
        short_name=_short_name,
    )

    assert applied == {}
    assert builder.joint_target_kd == [0.0] * 9


def test_engine_applies_the_models_own_gains(tmp_path):
    """The engine hook reads the MJCF it just imported and writes its gains."""
    pytest.importorskip("mujoco")
    path = tmp_path / "servos.xml"
    path.write_text(_MJCF)
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    builder = _RecordingBuilder()

    engine._apply_mjcf_servo_gains(builder, str(path), 0)

    assert builder.joint_target_kd[6] > 0.0
    assert builder.joint_target_kd[7] == pytest.approx(7.0)
    assert builder.joint_effort_limit[6] == pytest.approx(3.5)


def test_an_unreadable_model_keeps_the_imported_gains(tmp_path, caplog):
    """A model MuJoCo cannot compile is logged, not raised: the robot is in."""
    path = tmp_path / "broken.xml"
    path.write_text("<mujoco><worldbody><body/></worldbody>")
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    builder = _RecordingBuilder()

    engine._apply_mjcf_servo_gains(builder, str(path), 0)

    assert builder.joint_target_kd == [0.0] * 9
    assert "servo gains unread" in caplog.text
