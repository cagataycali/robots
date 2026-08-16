"""Regression tests: send_action refuses a non-finite action value.

``_coerce_action`` validated that every action value coerces to a scalar
``float`` and that a vector's length matches the robot's actuator count. It did
not validate finiteness, and ``nan``/``inf`` are perfectly good ``float``
objects, so a non-finite value was admitted, written to ``data.ctrl`` and handed
to ``mj_step``. Nothing further down looked at it either: the ctrl-clamp warning
passes a ``nan`` straight through.

Two distinct consequences, both reported as ``status="success"``:

* ``nan`` is not clamped. MuJoCo finds the resulting non-finite ``qacc`` and
  resets the world to its initial pose - on every substep, so the whole call's
  physics is discarded, and for *every* robot in the scene rather than only the
  commanded one. Because any later finite command integrates normally, the
  teleport leaves no residue: a recording rollout reports success for each step
  and the dataset simply holds a trajectory no robot followed.
* ``inf`` *is* clamped into ``ctrlrange``, i.e. silently rewritten into a
  full-travel command - the fabricated-absolute-position hazard class.

The sibling state writers ``set_joint_positions`` / ``set_joint_velocities``
already refuse a non-finite value, and ``get_ground_height`` refuses one for a
read-only terrain query, so these pin the actuator-command path to the same rule
and pin that the two accepted action shapes (mapping and ordered vector) cannot
diverge on it. The accepted domain is finiteness alone: a numeric string stays an
accepted spelling of a scalar, and a finite magnitude outside ``ctrlrange``
remains a units question surfaced by the clamp warning.
"""

import math

import numpy as np
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation

# Two-hinge position-servo arm. Inline so the test needs no asset download, and
# mounted high enough that neither link can reach the ground plane - a link in
# contact would hold the arm off its commanded target and make the parked pose
# (the non-vacuity premise of the scene-reset test below) indeterminate.
ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.9">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.5 1.5" limited="true" damping="4"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.03"/>
      <body name="link2" pos="0.2 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-1.5 1.5" limited="true" damping="4"/>
        <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.025"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="50" ctrlrange="-1.5 1.5"/>
    <position name="a_elbow" joint="elbow" kp="50" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>
"""

PARKED = 0.8
"""Commanded parking angle, well away from the model's ``qpos0`` of zero."""

NON_FINITE = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="-inf"),
    pytest.param(np.float32("nan"), id="np.float32-nan"),
    pytest.param(np.float64("inf"), id="np.float64-inf"),
]


@pytest.fixture
def arm_xml(tmp_path):
    path = tmp_path / "arm.xml"
    path.write_text(ARM_XML)
    return str(path)


@pytest.fixture
def sim(arm_xml):
    """Two independent arms, both parked away from ``qpos0``.

    Gravity is off so the parked pose is held by the servos alone, which is what
    makes an uncommanded robot's pose a usable witness for a whole-scene reset.
    """
    s = Simulation()
    s.create_world(gravity=[0.0, 0.0, 0.0])
    # Offset in y so the two arms cannot touch each other - co-located arms
    # collide and hold each other off their commanded targets. Both are added
    # before either is parked: adding a robot recompiles the model, which does
    # not carry ``ctrl`` over, so an arm parked first would be driven back to
    # zero while the second arm parks.
    for name, y in (("alice", -0.6), ("bob", 0.6)):
        s.add_robot(name=name, urdf_path=arm_xml, position=[0.0, y, 0.0])
    for name in ("alice", "bob"):
        s.send_action(
            {"a_shoulder": PARKED, "a_elbow": PARKED},
            robot_name=name,
            n_substeps=1,
        )
    s.step(1200)
    yield s
    s.cleanup()


def _pose(sim, robot):
    obs = sim.get_observation(robot_name=robot, skip_images=True)
    return (float(obs["shoulder"]), float(obs["elbow"]))


def _ctrl(sim, actuator):
    return float(sim.mj_data.ctrl[sim.mj_model.actuator(actuator).id])


class TestNonFiniteMappingValueRefused:
    """A mapping value that is not finite is refused, naming the key."""

    @pytest.mark.parametrize("value", NON_FINITE)
    def test_non_finite_mapping_value_is_refused(self, sim, value):
        result = sim.send_action({"a_shoulder": value}, robot_name="alice")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "finite (no nan/inf)" in text
        assert "'a_shoulder'" in text

    def test_message_names_the_real_consequence_not_clamping(self, sim):
        """The clamp warning attributes a ``nan`` to clamping; it is a reset."""
        result = sim.send_action({"a_shoulder": float("nan")}, robot_name="alice")
        text = result["content"][0]["text"]
        assert "reset" in text
        assert "not clamped" in text

    def test_single_element_non_finite_value_is_refused(self, sim):
        """The length-1 unwrap must not smuggle a ``nan`` past the check."""
        result = sim.send_action({"a_shoulder": [float("nan")]}, robot_name="alice")
        assert result["status"] == "error"
        assert "finite (no nan/inf)" in result["content"][0]["text"]


class TestNonFiniteVectorEntryRefused:
    """The ordered-vector shape is held to the same rule as the mapping."""

    @pytest.mark.parametrize("value", NON_FINITE)
    def test_non_finite_vector_entry_is_refused(self, sim, value):
        result = sim.send_action([value, PARKED], robot_name="alice")
        assert result["status"] == "error"
        assert "finite (no nan/inf)" in result["content"][0]["text"]

    def test_message_names_the_position_and_the_actuator_it_binds_to(self, sim):
        """A vector has no key, so the message must locate the entry itself."""
        result = sim.send_action([PARKED, float("nan")], robot_name="alice")
        text = result["content"][0]["text"]
        assert "entry 1" in text
        assert "'a_elbow'" in text

    def test_numpy_action_vector_entry_is_refused(self, sim):
        result = sim.send_action(np.array([float("nan"), PARKED]), robot_name="alice")
        assert result["status"] == "error"
        assert "finite (no nan/inf)" in result["content"][0]["text"]

    def test_a_length_mismatch_is_still_reported_before_finiteness(self, sim):
        """A wrong-length vector reports the length error, which lists the keys.

        The length mismatch is the more structural error and its message carries
        the actuator ordering the caller needs, so it must not be displaced by a
        finiteness complaint about an entry that would never have been bound.
        """
        result = sim.send_action([float("nan")], robot_name="alice")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "does not match robot" in text
        assert "finite (no nan/inf)" not in text


class TestRefusalPrecedesAnyActuatorWrite:
    """A refused action writes no ctrl at all, for any key."""

    def test_a_good_key_alongside_a_non_finite_key_applies_neither(self, sim):
        before = _ctrl(sim, "alice/a_elbow")
        result = sim.send_action(
            {"a_elbow": 0.2, "a_shoulder": float("nan")},
            robot_name="alice",
        )
        assert result["status"] == "error"
        assert _ctrl(sim, "alice/a_elbow") == before

    def test_no_actuator_in_the_scene_holds_a_non_finite_command(self, sim):
        sim.send_action({"a_shoulder": float("inf")}, robot_name="alice")
        assert all(math.isfinite(float(c)) for c in sim.mj_data.ctrl)


class TestNonFiniteCommandNoLongerResetsTheScene:
    """The headline consequence: one ``nan`` discarded the whole scene's state."""

    def test_the_parked_pose_is_a_usable_witness(self, sim):
        """Non-vacuity premise: parked is far from ``qpos0``, so a reset shows."""
        for robot in ("alice", "bob"):
            assert min(abs(v) for v in _pose(sim, robot)) > 0.1

    def test_an_uncommanded_robot_keeps_its_pose(self, sim):
        """``bob`` is never addressed, so nothing may move it."""
        before = _pose(sim, "bob")
        result = sim.send_action(
            {"a_shoulder": float("nan")},
            robot_name="alice",
            n_substeps=200,
        )
        assert result["status"] == "error"
        assert _pose(sim, "bob") == pytest.approx(before, abs=1e-9)

    def test_the_commanded_robot_keeps_its_pose_too(self, sim):
        """The refusal is atomic, so not even the addressed arm is disturbed."""
        before = _pose(sim, "alice")
        sim.send_action({"a_shoulder": float("nan")}, robot_name="alice", n_substeps=200)
        assert _pose(sim, "alice") == pytest.approx(before, abs=1e-9)

    def test_an_inf_no_longer_becomes_a_full_travel_command(self, sim):
        """``inf`` clamped into ``ctrlrange`` drove the joint to its limit."""
        before = _pose(sim, "alice")
        sim.send_action({"a_shoulder": float("inf")}, robot_name="alice", n_substeps=400)
        assert _pose(sim, "alice") == pytest.approx(before, abs=1e-9)


class TestFiniteValuesStillAccepted:
    """The guard constrains finiteness only - nothing else narrows."""

    def test_ordinary_finite_command_moves_the_arm(self, sim):
        before = _pose(sim, "alice")
        result = sim.send_action({"a_shoulder": 0.2}, robot_name="alice", n_substeps=400)
        assert result["status"] == "success", result
        assert abs(_pose(sim, "alice")[0] - before[0]) > 1e-3

    def test_a_numeric_string_is_still_an_accepted_scalar(self, sim):
        """A documented tolerance: ``float("0.5")`` is finite and exact."""
        result = sim.send_action({"a_shoulder": "0.5"}, robot_name="alice", n_substeps=2)
        assert result["status"] == "success", result

    def test_a_finite_magnitude_outside_ctrlrange_is_still_accepted(self, sim):
        """Out-of-range-but-finite stays the clamp warning's business."""
        result = sim.send_action({"a_shoulder": 1e300}, robot_name="alice", n_substeps=2)
        assert result["status"] == "success", result

    def test_a_finite_action_vector_is_still_accepted(self, sim):
        result = sim.send_action([0.3, 0.2], robot_name="alice", n_substeps=2)
        assert result["status"] == "success", result

    def test_numpy_finite_scalars_are_still_accepted(self, sim):
        result = sim.send_action(
            {"a_shoulder": np.float64(0.3), "a_elbow": np.float32(0.2)},
            robot_name="alice",
            n_substeps=2,
        )
        assert result["status"] == "success", result


class TestParityWithSiblingStateWriters:
    """``send_action`` accepts a value iff the sibling state writer does.

    ``set_joint_positions`` refused a non-finite value while the actuator
    command path admitted it, so the same scene had two different domains for
    the same kind of number. Parametrizing both over one probe set keeps them
    from drifting apart again.
    """

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="inf"),
            pytest.param(float("-inf"), id="-inf"),
            pytest.param(0.3, id="finite"),
            pytest.param(1e300, id="large-finite"),
        ],
    )
    def test_the_two_writers_agree_on_the_value(self, sim, value):
        action = sim.send_action({"a_shoulder": value}, robot_name="alice", n_substeps=2)
        state = sim.set_joint_positions({"shoulder": value}, robot_name="alice")
        assert (action["status"] == "error") == (state["status"] == "error"), (
            f"verdicts differ for {value!r}: send_action={action['status']} set_joint_positions={state['status']}"
        )
