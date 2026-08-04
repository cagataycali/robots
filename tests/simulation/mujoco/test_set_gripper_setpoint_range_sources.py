"""Regression tests for GH #1942: where ``set_gripper`` gets its set-points.

``set_gripper`` derived its open/close set-points from ``actuator_ctrlrange``
alone and refused any actuator whose range was not strictly increasing. MuJoCo
reports exactly ``(0, 0)`` with ``actuator_ctrllimited == 0`` for an actuator
the MJCF left **unlimited**, which is a different claim from "this actuator
accepts nothing" - so ``set_gripper`` refused on so101, a shipped robot whose
registry gripper metadata is correct and whose ``move_to`` / ``rotate_wrist``
both work:

    set_gripper: actuator 'a/6' has no usable ctrlrange (0.0, 0.0);
    cannot infer open/close set-points.

so101's MJCF declares neither ``ctrlrange`` nor ``inheritrange="1"`` on its
position servos; so100's sets ``inheritrange="1"`` on every actuator, which
compiles a real ctrlrange from the driven joint - the only reason so100 was
unaffected. Measured on the downloaded assets:

    robot  gripper act  ctrllimited  ctrlrange       jnt_limited  jnt_range
    so100  Jaw          True         (-0.174, 1.75)  True         (-0.174, 1.75)
    so101  6            False        (0.0, 0.0)      True         (-0.1745, 1.7453)

These tests are hermetic - one inline MJCF arm per case, no asset download -
and vary exactly one thing between the working and refusing shapes: the jaw
actuator's declaration. They pin, in order:

1. The MuJoCo premise the fix rests on (an unauthored ctrlrange compiles to
   ``(0, 0)`` / ``ctrllimited == 0``, and ``inheritrange="1"`` compiles a real
   one). If a future MuJoCo changes that encoding, this fails first and names
   the reason, rather than the behaviour tests failing obscurely.
2. The regression: the so101 shape drives to the driven joint's limits.
3. Precedence: an authored ctrlrange wins over the joint range, so the fix
   cannot silently widen a deliberately narrowed command range.
4. The three shapes that must KEEP refusing - an authored ``ctrllimited == 1``
   with a degenerate range, an unlimited driven joint, and a tendon actuator
   (whose ctrlrange is a normalised command space, not joint units, so a joint
   range would command the wrong quantity).
"""

from typing import Any

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Jaw joint limits shared by every variant below: what the driven-joint
# substitution must produce, and what an authored ctrlrange must override.
JAW_LO, JAW_HI = -0.2, 1.5

# A minimal pan + jaw arm. The jaw joint and its actuator are the only things
# that vary between cases, so each variant differs from the next by exactly the
# MuJoCo attribute under test. The gripper resolves via the name heuristic (no
# data_config -> no registry metadata), which keeps these tests about the
# set-point range rather than about actuator classification.
_ARM_TEMPLATE = """
<mujoco model="setpoint_range_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <default>
    <joint armature="0.05" damping="0.5"/>
    <geom density="2000"/>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.04 0.02"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
        <body name="jaw_body" pos="0 0 0.2">
          {jaw_joint}
          <geom type="box" size="0.01 0.01 0.02" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan" joint="shoulder_pan" kp="20" dampratio="1" ctrlrange="-3.14 3.14"/>
    {jaw_actuator}
  </actuator>
</mujoco>
"""

_LIMITED_JAW_JOINT = (
    f'<joint name="jaw" type="hinge" axis="0 0 1" range="{JAW_LO} {JAW_HI}" armature="0.01" damping="0.1"/>'
)
# No `range` attribute -> jnt_limited is False (nothing to substitute).
_UNLIMITED_JAW_JOINT = '<joint name="jaw" type="hinge" axis="0 0 1" armature="0.01" damping="0.1"/>'


def _arm(jaw_actuator: str, jaw_joint: str = _LIMITED_JAW_JOINT) -> str:
    return _ARM_TEMPLATE.format(jaw_joint=jaw_joint, jaw_actuator=jaw_actuator)


# so101's shape: a position servo with neither ctrlrange nor inheritrange.
UNSET_CTRLRANGE_XML = _arm('<position name="jaw" joint="jaw" kp="2" dampratio="1"/>')

# so100's shape: inheritrange="1" compiles the ctrlrange from the driven joint.
INHERITRANGE_XML = _arm('<position name="jaw" joint="jaw" kp="2" dampratio="1" inheritrange="1"/>')

# An authored ctrlrange deliberately NARROWER than the joint range: the
# substitution must not widen it back out.
AUTHORED_LO, AUTHORED_HI = -0.1, 0.9
AUTHORED_CTRLRANGE_XML = _arm(
    f'<position name="jaw" joint="jaw" kp="2" dampratio="1" ctrlrange="{AUTHORED_LO} {AUTHORED_HI}"/>'
)

# A degenerate range written out in the MJCF. This does NOT compile to
# ctrllimited=1 - MuJoCo treats a non-strictly-increasing ctrlrange as unset
# (see TestMujocoEncodingPremise) - so it takes the same fallback as so101.
DEGENERATE_UNLIMITED_XML = _arm('<position name="jaw" joint="jaw" kp="2" dampratio="1" ctrlrange="0.5 0.5"/>')

# Unset ctrlrange AND an unlimited driven joint: both sources exhausted.
UNLIMITED_JOINT_XML = _arm(
    '<position name="jaw" joint="jaw" kp="2" dampratio="1"/>',
    jaw_joint=_UNLIMITED_JAW_JOINT,
)

# A tendon gripper (the Franka/Panda shape) with an unset ctrlrange. A tendon
# actuator's ctrlrange is a normalised command space, so the driven joints'
# limits are the wrong quantity; it has no _joint_actuator_map entry by
# construction, which is what scopes the substitution away from it.
TENDON_XML = """
<mujoco model="setpoint_range_tendon">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <default>
    <joint armature="0.05" damping="0.5"/>
    <geom density="2000"/>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.04 0.02"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
        <body name="hand" pos="0 0 0.2">
          <geom type="box" size="0.03 0.03 0.01" contype="0" conaffinity="0"/>
          <body name="finger1" pos="0.03 0 0">
            <joint name="finger_joint1" type="slide" axis="1 0 0" range="0 0.04"/>
            <geom type="box" size="0.005 0.01 0.02" contype="0" conaffinity="0"/>
          </body>
          <body name="finger2" pos="-0.03 0 0">
            <joint name="finger_joint2" type="slide" axis="-1 0 0" range="0 0.04"/>
            <geom type="box" size="0.005 0.01 0.02" contype="0" conaffinity="0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="split">
      <joint joint="finger_joint1" coef="1"/>
      <joint joint="finger_joint2" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <position name="shoulder_pan" joint="shoulder_pan" kp="20" dampratio="1" ctrlrange="-3.14 3.14"/>
    <position name="gripper" tendon="split" kp="1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def make_sim(tmp_path):
    """Build a Simulation holding one robot loaded from *xml*, and clean it up."""
    sims: list[Simulation] = []

    def _make(xml: str, name: str = "arm") -> Simulation:
        path = tmp_path / f"{name}_{len(sims)}.xml"
        path.write_text(xml)
        s = Simulation(tool_name="test_set_gripper_setpoint_range_sources", mesh=False)
        sims.append(s)
        assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
        assert s.add_robot(name, urdf_path=str(path))["status"] == "success"
        return s

    yield _make
    for s in sims:
        s.cleanup(policy_stop_timeout=2.0)


def _json_block(result: dict[str, Any]) -> dict[str, Any]:
    for block in result["content"]:
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in result: {result}")


def _set_gripper(s: Simulation, state: str, robot_name: str = "arm", steps: int = 40) -> dict[str, Any]:
    return s._dispatch_action(
        "set_gripper", {"action": "set_gripper", "robot_name": robot_name, "state": state, "steps": steps}
    )


def _jaw_position(s: Simulation, robot_name: str = "arm", joint: str = "jaw") -> float:
    state = _json_block(s.get_robot_state(robot_name))["state"]
    return float(state[joint]["position"])


class TestMujocoEncodingPremise:
    """The fix reads ``ctrllimited == 0`` as "unset". Pin that encoding.

    Nothing in the behaviour tests below would say *why* they broke if MuJoCo
    ever compiled an unauthored ctrlrange differently, so this asserts the
    premise directly and separately.
    """

    def test_an_unauthored_ctrlrange_compiles_to_zero_zero_and_ctrllimited_false(self):
        model = mujoco.MjModel.from_xml_string(UNSET_CTRLRANGE_XML)
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw")
        jnt = int(model.actuator_trnid[act, 0])

        assert bool(model.actuator_ctrllimited[act]) is False
        assert tuple(float(v) for v in model.actuator_ctrlrange[act]) == (0.0, 0.0)
        # ... while the joint it drives carries perfectly good limits, and the
        # transmission is the one where ctrl IS the joint target.
        assert int(model.actuator_trntype[act]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        assert bool(model.jnt_limited[jnt]) is True
        assert tuple(float(v) for v in model.jnt_range[jnt]) == (JAW_LO, JAW_HI)

    def test_inheritrange_compiles_the_joint_range_into_the_ctrlrange(self):
        """so100's shape - and the substitution the fix performs by hand."""
        model = mujoco.MjModel.from_xml_string(INHERITRANGE_XML)
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw")

        assert bool(model.actuator_ctrllimited[act]) is True
        # approx: the compiler round-trips the joint range through float storage.
        assert tuple(float(v) for v in model.actuator_ctrlrange[act]) == pytest.approx((JAW_LO, JAW_HI))

    @pytest.mark.parametrize("ctrlrange", ["0 0", "0.5 0.5", "1 0"])
    def test_a_bare_degenerate_ctrlrange_compiles_to_unlimited(self, ctrlrange):
        """MuJoCo treats any non-strictly-increasing ctrlrange as UNSET.

        This is why ``ctrllimited`` is the right signal for "was a command range
        authored", and why ``(0, 0)`` cannot be told apart from an omitted
        attribute: the compiler collapses both to the same state.
        """
        model = mujoco.MjModel.from_xml_string(
            _arm(f'<position name="jaw" joint="jaw" kp="2" dampratio="1" ctrlrange="{ctrlrange}"/>')
        )
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw")

        assert bool(model.actuator_ctrllimited[act]) is False

    def test_the_compiler_refuses_an_explicit_limit_over_a_degenerate_range(self):
        """So ``ctrllimited == 1`` + degenerate range is unreachable from MJCF.

        The guard that respects such a claim is therefore reachable only on a
        model mutated after compilation - which is exactly how it is exercised
        in ``TestTheExhaustedSourcesStillRefuse``.
        """
        with pytest.raises(ValueError, match="invalid control range"):
            mujoco.MjModel.from_xml_string(
                _arm('<position name="jaw" joint="jaw" kp="2" ctrllimited="true" ctrlrange="0 0"/>')
            )

    def test_an_unclamped_actuator_ignores_its_stored_degenerate_range(self):
        """A stored range under ctrllimited=0 is inert, not restrictive.

        MuJoCo clamps ``ctrl`` only when ``ctrllimited == 1``, so substituting
        the joint range cannot widen a restriction that was being enforced -
        there was none.
        """
        model = mujoco.MjModel.from_xml_string(DEGENERATE_UNLIMITED_XML)
        data = mujoco.MjData(model)
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw")

        data.ctrl[act] = JAW_HI  # far above the stored (0.5, 0.5)
        mujoco.mj_step(model, data)

        assert float(data.ctrl[act]) == pytest.approx(JAW_HI)

    def test_a_tendon_actuator_has_no_driven_joint_to_substitute(self):
        """The scope guard is structural: trntype is TENDON, not JOINT."""
        model = mujoco.MjModel.from_xml_string(TENDON_XML)
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")

        assert int(model.actuator_trntype[act]) == int(mujoco.mjtTrn.mjTRN_TENDON)
        assert bool(model.actuator_ctrllimited[act]) is False


class TestUnsetCtrlrangeFallsBackToTheDrivenJoint:
    """GH #1942: the so101 shape is driveable, and drives to the joint limits."""

    @pytest.mark.parametrize(
        ("state", "expected"),
        [("close", JAW_LO), ("open", JAW_HI)],
    )
    def test_set_gripper_commands_the_driven_joint_range_end(self, make_sim, state, expected):
        s = make_sim(UNSET_CTRLRANGE_XML)

        result = _set_gripper(s, state)

        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["targets"]["jaw"] == pytest.approx(expected)

    def test_the_substituted_source_is_named_in_the_payload(self, make_sim):
        """A substitution that changes which quantity defines the set-point is
        reported, not silent: an operator reading ``targets`` can tell an
        authored command range from an inferred one."""
        s = make_sim(UNSET_CTRLRANGE_XML)

        payload = _json_block(_set_gripper(s, "close"))

        assert payload["setpoint_sources"] == {"jaw": "driven joint range"}

    def test_the_jaw_actually_travels_toward_the_commanded_end(self, make_sim):
        """Behaviour, not just target arithmetic: the fingers move."""
        s = make_sim(UNSET_CTRLRANGE_XML)

        assert _set_gripper(s, "open", steps=80)["status"] == "success"
        opened = _jaw_position(s)
        assert _set_gripper(s, "close", steps=80)["status"] == "success"
        closed = _jaw_position(s)

        assert closed < opened, f"jaw did not travel: open={opened}, close={closed}"


class TestAnAuthoredCtrlrangeStillWins:
    """The substitution is a fallback, not an override."""

    def test_inheritrange_uses_the_compiled_ctrlrange(self, make_sim):
        """so100's shape is unchanged - and reports the ctrlrange as its source."""
        s = make_sim(INHERITRANGE_XML)

        payload = _json_block(_set_gripper(s, "open"))

        assert payload["targets"]["jaw"] == pytest.approx(JAW_HI)
        assert payload["setpoint_sources"] == {"jaw": "actuator ctrlrange"}

    @pytest.mark.parametrize(
        ("state", "expected"),
        [("close", AUTHORED_LO), ("open", AUTHORED_HI)],
    )
    def test_a_narrower_authored_ctrlrange_is_not_widened_to_the_joint_range(self, make_sim, state, expected):
        """An MJCF that deliberately restricts the command range below the joint
        range keeps that restriction: the fallback never fires when the
        ctrlrange is usable."""
        s = make_sim(AUTHORED_CTRLRANGE_XML)

        payload = _json_block(_set_gripper(s, state))

        assert payload["targets"]["jaw"] == pytest.approx(expected)
        assert payload["setpoint_sources"] == {"jaw": "actuator ctrlrange"}


class TestAnInertDegenerateCtrlrangeAlsoFallsBack:
    """A ctrlrange the MJCF wrote but MuJoCo ignores is not a restriction."""

    def test_a_degenerate_authored_ctrlrange_uses_the_driven_joint_range(self, make_sim):
        s = make_sim(DEGENERATE_UNLIMITED_XML)

        payload = _json_block(_set_gripper(s, "open"))

        assert payload["targets"]["jaw"] == pytest.approx(JAW_HI)
        assert payload["setpoint_sources"] == {"jaw": "driven joint range"}


class TestTheExhaustedSourcesStillRefuse:
    """Three shapes where no source yields set-points; each names what it tried."""

    def test_a_declared_limit_over_a_degenerate_range_is_respected(self, make_sim):
        """``ctrllimited == 1`` is a claim about the actuator, so it is not
        second-guessed even though the driven joint has usable limits.

        The MJCF compiler refuses to emit this combination, so it is produced
        the only way it can occur in practice - by mutating the compiled model,
        as ``policies/wbc/sim_control.py`` does when it hands ``ctrlrange`` to a
        whole-body controller.
        """
        s = make_sim(UNSET_CTRLRANGE_XML)
        model = s._world._model
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm/jaw")
        assert act >= 0, "jaw actuator not found under the robot namespace"
        assert bool(model.actuator_ctrllimited[act]) is False  # fallback applies before the mutation
        model.actuator_ctrllimited[act] = 1  # now it claims (0, 0) as a real limit

        result = _set_gripper(s, "close")

        assert result["status"] == "error", result
        text = str(result["content"])
        assert "no usable open/close set-points" in text
        assert "ctrllimited=1" in text

    def test_an_unlimited_driven_joint_leaves_nothing_to_substitute(self, make_sim):
        s = make_sim(UNLIMITED_JOINT_XML)

        result = _set_gripper(s, "close")

        assert result["status"] == "error", result
        text = str(result["content"])
        assert "no usable open/close set-points" in text
        assert "itself unlimited" in text

    def test_a_tendon_gripper_refuses_rather_than_borrowing_joint_units(self, make_sim):
        """The scope limit from the issue: a tendon ctrlrange is a normalised
        command space, so substituting a joint range would command the wrong
        quantity. Refusing is correct."""
        s = make_sim(TENDON_XML)

        result = _set_gripper(s, "close")

        assert result["status"] == "error", result
        text = str(result["content"])
        assert "no usable open/close set-points" in text
        assert "normalised command space" in text

    def test_every_refusal_still_points_at_the_direct_escape_hatch(self, make_sim):
        """The error stays actionable - send_action remains the way through."""
        for xml in (UNLIMITED_JOINT_XML, TENDON_XML):
            s = make_sim(xml)
            result = _set_gripper(s, "close")
            assert result["status"] == "error", result
            assert "send_action" in str(result["content"])
