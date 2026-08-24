"""A velocity write names the joints whose rate drive commands something else.

``set_joint_velocities`` writes ``qvel`` directly. On a joint driven by a
``<velocity>`` actuator that ``ctrl`` IS the joint's own rate, so the drive is
already commanding a velocity of its own and the next step resolves the
disagreement in the drive's favour: measured on the fixture below, a written
``+2.0`` against a drive commanding ``-3.0`` reads ``-2.39877`` fifty steps
later, sign reversed. The report said ``Set 1/1 joint velocities`` either way,
byte-identical to the case where the drive commands the written rate and it is
held -- one report for two opposite outcomes.

Its sibling ``set_joint_positions`` names exactly this class for a position
servo ("still commanded to a different setpoint ... the next step drives the
pose back"), so the shape of the answer was already settled one method over;
these tests pin the velocity half of it, and pin that the report stays silent
for every drive whose ``ctrl`` is not a rate in the joint's own units.
"""

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation import create_simulation

#: A single-hinge body per drive, sized so a written rate settles without
#: tripping the integrator (``badqacc`` stays 0 on every case below).
_BODY = """
    <body name="b_{tag}" pos="0 {y} 0.5">
      <joint name="j_{tag}" type="hinge" axis="0 0 1" damping="0.5" armature="0.01"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" mass="1.0"/>
    </body>"""

_SCENE = """<mujoco model="drives">
  <compiler angle="radian"/>
  <worldbody>{bodies}
  </worldbody>
  <actuator>{actuators}
  </actuator>
</mujoco>"""

#: ``(tag, actuator element)`` for every MuJoCo actuator shortcut that can drive
#: a hinge, plus a joint no actuator drives. Only ``<velocity>`` commands a rate
#: in the joint's own units.
_DRIVES: list[tuple[str, str]] = [
    ("vel", '<velocity name="a_vel" joint="j_vel" kv="2" ctrlrange="-10 10"/>'),
    ("servo", '<position name="a_servo" joint="j_servo" kp="20" ctrlrange="-3 3"/>'),
    ("motor", '<motor name="a_motor" joint="j_motor" ctrlrange="-5 5"/>'),
    ("intvel", '<intvelocity name="a_intvel" joint="j_intvel" kp="20" actrange="-3 3" ctrlrange="-2 2"/>'),
    ("damper", '<damper name="a_damper" joint="j_damper" kv="2" ctrlrange="0 1"/>'),
    ("cylinder", '<cylinder name="a_cylinder" joint="j_cylinder" ctrlrange="-1 1"/>'),
    ("general", '<general name="a_general" joint="j_general" ctrlrange="-1 1"/>'),
    # <velocity> written longhand: the shortcut is sugar for exactly these
    # compiled fields, so a model spelling it out must classify the same way.
    (
        "gen_vel",
        '<general name="a_gen_vel" joint="j_gen_vel" biastype="affine" '
        'biasprm="0 0 -2" gainprm="2" ctrlrange="-10 10"/>',
    ),
    # Positive velocity feedback is anti-damping rather than a rate command:
    # measured, a written rate against it diverges (mj_step reports
    # "Nan, Inf or huge value in QACC") instead of settling on any commanded
    # value, so slot 2 has to be negative and not merely non-zero.
    (
        "gen_posbias",
        '<general name="a_gen_posbias" joint="j_gen_posbias" biastype="affine" '
        'biasprm="0 0 2" gainprm="2" ctrlrange="-10 10"/>',
    ),
    ("undriven", ""),
]

#: The tags whose ``ctrl`` is a velocity in the joint's own units.
_RATE_TAGS = frozenset({"vel", "gen_vel"})
_RATE_TAG = "vel"


def _scene_xml(tags: list[str]) -> str:
    bodies = "".join(_BODY.format(tag=t, y=round(0.4 * i, 2)) for i, t in enumerate(tags))
    by_tag = dict(_DRIVES)
    actuators = "".join(f"\n    {by_tag[t]}" for t in tags if by_tag[t])
    return _SCENE.format(bodies=bodies, actuators=actuators)


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


def _ok(result: dict[str, Any], what: str) -> dict[str, Any]:
    if result.get("status") != "success":
        raise AssertionError(f"{what} refused: {_text(result)}")
    return result


@pytest.fixture
def sim(tmp_path: Any) -> Any:
    """A world holding one robot with one hinge per drive kind."""
    tags = [t for t, _ in _DRIVES]
    path = tmp_path / "drives.xml"
    path.write_text(_scene_xml(tags))
    engine = create_simulation(backend="mujoco", tool_name="rate_drive_probe")
    _ok(engine.create_world(gravity=[0, 0, 0]), "create_world")
    _ok(engine.add_robot(name="arm", urdf_path=str(path)), "add_robot")
    yield engine
    engine.destroy()


def _qvel(engine: Any, joint: str) -> float:
    import mujoco as mj

    model = engine.mj_model
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint)
    assert jid >= 0, f"premise: {joint!r} is a joint of the fixture"
    return float(engine._world._data.qvel[model.jnt_dofadr[jid]])


def _badqacc(engine: Any) -> int:
    import mujoco as mj

    return int(engine._world._data.warning[mj.mjtWarning.mjWARN_BADQACC].number)


class TestAConflictingRateDriveIsNamed:
    """The written joints whose rate drive commands something else are reported."""

    def test_the_report_names_the_joint_and_the_way_to_command_the_rate(self, sim: Any) -> None:
        _ok(sim.send_action({"a_vel": -3.0}), "send_action")

        report = _text(_ok(sim.set_joint_velocities({"arm/j_vel": 2.0}), "set_joint_velocities"))

        assert "arm/j_vel" in report, report
        assert "commanding a different rate" in report, report
        assert "send_action" in report, report

    def test_the_written_rate_is_driven_back_toward_the_drives_command(self, sim: Any) -> None:
        _ok(sim.send_action({"a_vel": -3.0}), "send_action")
        _ok(sim.set_joint_velocities({"arm/j_vel": 2.0}), "set_joint_velocities")
        assert _qvel(sim, "arm/j_vel") == pytest.approx(2.0), "premise: the write landed in qvel"

        _ok(sim.step(50), "step")

        settled = _qvel(sim, "arm/j_vel")
        assert _badqacc(sim) == 0, "premise: the fixture integrates without a bad-qacc warning"
        assert settled < 0.0, f"the written +2.0 should be driven toward the commanded -3.0, got {settled}"

    def test_a_conflicting_drive_reads_differently_from_one_commanding_the_written_rate(self, sim: Any) -> None:
        _ok(sim.send_action({"a_vel": 2.0}), "send_action")
        agreeing = _text(_ok(sim.set_joint_velocities({"arm/j_vel": 2.0}), "set_joint_velocities"))

        _ok(sim.send_action({"a_vel": -3.0}), "send_action")
        conflicting = _text(_ok(sim.set_joint_velocities({"arm/j_vel": 2.0}), "set_joint_velocities"))

        assert agreeing != conflicting, (
            "a drive commanding the written rate and one commanding another rate are opposite "
            f"outcomes and must not share one report: {agreeing!r}"
        )


class TestOnlyARateDriveIsNamed:
    """Every drive is commanded away from the written rate; only a rate drive reads."""

    @pytest.mark.parametrize("tag", [t for t, _ in _DRIVES])
    def test_a_drive_is_named_only_when_its_ctrl_is_a_rate_in_the_joints_units(self, sim: Any, tag: str) -> None:
        if tag != "undriven":
            # Command the drive to a value the write does not ask for. For a rate
            # drive that is a conflicting velocity; for every other kind ctrl is a
            # torque, a pose or a tendon length, so there is no rate to conflict.
            _ok(sim.send_action({f"a_{tag}": 1.0}), "send_action")

        report = _text(_ok(sim.set_joint_velocities({f"arm/j_{tag}": 2.0}), "set_joint_velocities"))

        named = f"arm/j_{tag}" in report
        assert named is (tag in _RATE_TAGS), (
            f"j_{tag} named in the report: {tag in _RATE_TAGS} expected, {named} measured -- {report!r}"
        )


class TestTheReportIsNotWidened:
    """A drive commanding the written rate is not a disagreement."""

    def test_a_drive_commanding_the_written_rate_keeps_the_plain_report(self, sim: Any) -> None:
        _ok(sim.send_action({"a_vel": 2.0}), "send_action")

        report = _text(_ok(sim.set_joint_velocities({"arm/j_vel": 2.0}), "set_joint_velocities"))

        assert report == "Set 1/1 joint velocities", report

    def test_a_rate_drive_commanding_the_written_rate_holds_it(self, sim: Any) -> None:
        _ok(sim.send_action({"a_vel": 2.0}), "send_action")
        _ok(sim.set_joint_velocities({"arm/j_vel": 2.0}), "set_joint_velocities")

        _ok(sim.step(50), "step")

        settled = _qvel(sim, "arm/j_vel")
        assert _badqacc(sim) == 0, "premise: the fixture integrates without a bad-qacc warning"
        assert settled > 1.0, f"the drive commands the written rate, so it should hold it, got {settled}"


class TestTheSiblingPositionReportIsUnchanged:
    """The pose half of this contract keeps its own wording and remedy."""

    def test_a_stale_position_servo_still_names_hold(self, sim: Any) -> None:
        _ok(sim.send_action({"a_servo": 0.0}), "send_action")

        report = _text(_ok(sim.set_joint_positions({"arm/j_servo": 1.0}), "set_joint_positions"))

        assert "arm/j_servo" in report, report
        assert "different setpoint" in report, report
        assert "hold=True" in report, report


class TestTheRateDriveClassifier:
    """Root-cause pins on the classifier the report is derived from.

    These name ``joint_rate_drive_map`` directly, so on a tree without it they
    can only report its absence; the behavioural evidence for the report itself
    is in the classes above.
    """

    def test_the_scan_finds_the_fixtures_one_rate_drive(self, sim: Any) -> None:
        import mujoco as mj

        from strands_robots.simulation.mujoco.scene_ops import joint_rate_drive_map

        rate_drives = joint_rate_drive_map(sim.mj_model, mj)

        assert len(rate_drives) == len(_RATE_TAGS), (
            f"the fixture declares {len(_RATE_TAGS)} rate drives (the <velocity> shortcut and its "
            f"longhand <general> spelling); a scan reporting {len(rate_drives)} is not measuring the "
            "discriminator"
        )

    def test_a_tendon_rate_drive_is_not_a_joint_rate_drive(self, tmp_path: Any) -> None:
        import mujoco as mj

        from strands_robots.simulation.mujoco.scene_ops import joint_rate_drive_map

        xml = """<mujoco model="tendon">
          <compiler angle="radian"/>
          <worldbody>
            <body name="b" pos="0 0 0.5">
              <joint name="j1" type="hinge" axis="0 0 1" damping="0.5" armature="0.01"/>
              <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/>
              <body name="c" pos="0.2 0 0">
                <joint name="j2" type="hinge" axis="0 0 1" damping="0.5" armature="0.01"/>
                <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/>
              </body>
            </body>
          </worldbody>
          <tendon>
            <fixed name="t"><joint joint="j1" coef="1"/><joint joint="j2" coef="1"/></fixed>
          </tendon>
          <actuator>
            <velocity name="a_t" tendon="t" kv="2" ctrlrange="-10 10"/>
          </actuator>
        </mujoco>"""
        path = tmp_path / "tendon.xml"
        path.write_text(xml)
        model = mj.MjModel.from_xml_path(str(path))
        assert int(model.nu) == 1, "premise: the fixture declares one tendon velocity actuator"

        rate_drives = joint_rate_drive_map(model, mj)

        assert rate_drives == {}, (
            "a tendon drive's ctrl is in the tendon's units and drives several joints, so no single "
            f"joint rate can be written into it: {rate_drives}"
        )
