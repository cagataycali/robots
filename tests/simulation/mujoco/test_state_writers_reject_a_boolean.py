"""The runtime state writers refuse a boolean instead of writing it as 1.0 (#1838).

``set_joint_positions``, ``set_joint_velocities`` and ``apply_force`` each
coerced their inputs with a bare ``float()``. ``bool`` is an ``int`` subclass, so
``float(True)`` is ``1.0`` and every one of them reported ``status="success"``
having written 1 radian, 1 rad/s or 1 N::

    set_joint_positions({"joint1": True})  -> success  "Set 1/1 joint positions, FK updated"
    apply_force(body_name="base", force=[True, 0.0, 0.0])  -> success  "Force: [1.0, 0.0, 0.0] N"

while every scene-construction vector refused the same value. #1837 settled the
**actuator-command** path (``send_action``); these are the **state writers**, and
``apply_force`` additionally carried a comment deferring the question as "out of
scope for numeric-element validation". The decision this pins is that they refuse
it, for consistency with the domain
:func:`strands_robots.utils.finite_vector_error` already states - the full
argument is in ``physics._BOOLEAN_STATE_REASON``.

Every boolean assertion here fails on pre-fix code, where the call returned
``status="success"``. The ``StillAccepted`` / ``still`` tests are the over-reach
controls: a gate that also rejected ``1``, ``np.uint8(1)``, a 0-d numeric array or
the documented ``force=[0, 0, 0]`` would satisfy the rejections and break every
real caller.
"""

from __future__ import annotations

import importlib.util
import inspect
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.mujoco.physics import (
    _BOOLEAN_STATE_REASON,
    _coerce_finite_joint_map,
)
from strands_robots.utils import is_boolean

# Every spelling of a boolean that can reach a writer: a python bool, a numpy
# boolean scalar (what ``gripper > 0.5`` produces), and a 0-d boolean array (what
# ``np.array(True)`` or a reduction produces). numpy.bool_ is not a bool
# subclass, so an isinstance-only gate catches the first two and misses the rest.
_BOOLEANS = [True, False, np.True_, np.bool_(False), np.array(True)]
_BOOLEAN_IDS = ["True", "False", "np_true", "np_false", "zero_d_array"]

# Numeric values a caller legitimately passes. ``1`` and ``np.uint8(1)`` are the
# load-bearing entries: they coerce to the same 1.0 the boolean would have
# written, so they separate "refuses a boolean" from "refuses anything equal to 1".
_NUMBERS = [0, 1, -1, 0.5, -1.25, np.float64(0.3), np.int64(2), np.uint8(1), np.array(0.7)]
_NUMBER_IDS = ["int_0", "int_1", "int_neg1", "float", "neg_float", "np_float", "np_int", "np_uint8", "zero_d_array"]


class TestSharedJointMapCoercionRefusesABoolean:
    """``_coerce_finite_joint_map`` backs both joint writers, so one gate covers both.

    Exercised directly so the domain is pinned even where MuJoCo is unavailable;
    the sim-level classes below prove the writers actually route through it.
    """

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    @pytest.mark.parametrize("param", ["positions", "velocities"])
    def test_every_boolean_spelling_is_refused(self, value: Any, param: str) -> None:
        out, err = _coerce_finite_joint_map({"j0": value}, param, f"set_joint_{param}")
        assert err is not None, f"{value!r} was accepted as a {param} value"
        assert err["status"] == "error"
        assert out == {}, "a refused map must coerce nothing"

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_the_refusal_names_the_joint_the_parameter_and_the_units(self, value: Any) -> None:
        _, err = _coerce_finite_joint_map({"shoulder_pan": value}, "positions", "set_joint_positions")
        assert err is not None
        text = err["content"][0]["text"]
        assert "set_joint_positions" in text
        assert "shoulder_pan" in text, "the caller cannot fix the value without knowing which joint"
        assert "'positions'" in text
        assert "not a bool" in text, "the message must distinguish a bool from a plain non-number"
        assert _BOOLEAN_STATE_REASON in text, "the refusal must carry the reason, not just the rejection"

    def test_a_boolean_among_valid_values_refuses_the_whole_map(self) -> None:
        """The write is all-or-nothing: one bad value coerces none of them."""
        out, err = _coerce_finite_joint_map({"j0": 0.5, "j1": True, "j2": -0.25}, "positions", "set_joint_positions")
        assert err is not None
        assert "j1" in err["content"][0]["text"]
        assert out == {}, "a partial coercion would let a partial pose through"

    @pytest.mark.parametrize("value", _NUMBERS, ids=_NUMBER_IDS)
    def test_numbers_are_still_accepted(self, value: Any) -> None:
        out, err = _coerce_finite_joint_map({"j0": value}, "positions", "set_joint_positions")
        assert err is None, f"{value!r} must remain a usable position"
        assert out == {"j0": float(value)}

    def test_one_is_accepted_though_it_is_what_true_would_have_written(self) -> None:
        """The gate keys on the type, not the value - 1.0 rad is a real request."""
        out, err = _coerce_finite_joint_map({"j0": 1}, "positions", "set_joint_positions")
        assert err is None
        assert out == {"j0": 1.0}
        _, bool_err = _coerce_finite_joint_map({"j0": True}, "positions", "set_joint_positions")
        assert bool_err is not None, "True must not ride in on int's acceptance"

    @pytest.mark.parametrize(
        "value, expected",
        [("nope", "must be a number"), (float("nan"), "finite"), (float("inf"), "finite")],
        ids=["non_numeric", "nan", "inf"],
    )
    def test_the_pre_existing_domain_is_unchanged(self, value: Any, expected: str) -> None:
        """The bool gate is additive: non-numeric and nan/inf keep their own messages."""
        _, err = _coerce_finite_joint_map({"j0": value}, "positions", "set_joint_positions")
        assert err is not None
        text = err["content"][0]["text"]
        assert expected in text
        assert "not a bool" not in text, "a nan is not a bool and must not be described as one"


class TestOneBooleanPredicateNotThree:
    """The predicate is shared, so the writers and ``send_action`` cannot diverge.

    Before this change ``simulation/base.py`` and ``mesh/security.py`` each
    carried their own numpy-bool unwrap. A third one in ``physics.py`` would make
    "is this a boolean" a question with three answers, so the writers reuse the
    one in :mod:`strands_robots.utils`, next to the scalar domains that already
    reject a bool for the same stated reason.
    """

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_the_shared_predicate_reports_every_boolean_spelling(self, value: Any) -> None:
        assert is_boolean(value) is True

    @pytest.mark.parametrize("value", _NUMBERS, ids=_NUMBER_IDS)
    def test_the_shared_predicate_reports_no_number_as_boolean(self, value: Any) -> None:
        assert is_boolean(value) is False

    @pytest.mark.parametrize(
        "value",
        ["0.5", None, [1.0], np.array([1.0]), np.array([True, False])],
        ids=["str", "None", "list", "array", "bool_array"],
    )
    def test_a_non_scalar_value_is_left_to_the_other_checks(self, value: Any) -> None:
        """A multi-element array has no single item, so the predicate must not raise."""
        assert is_boolean(value) is False

    def test_the_send_action_gate_uses_the_same_predicate(self) -> None:
        from strands_robots.simulation import base

        assert base.is_boolean is is_boolean, "send_action must not shadow the shared predicate"
        assert "is_boolean(value)" in inspect.getsource(base._boolean_action_error)

    def test_the_joint_writers_and_apply_force_use_the_same_predicate(self) -> None:
        from strands_robots.simulation.mujoco import physics

        assert physics.is_boolean is is_boolean
        assert "is_boolean(value)" in inspect.getsource(physics._coerce_finite_joint_map)
        assert "is_boolean(_elem)" in inspect.getsource(physics.PhysicsMixin.apply_force)

    def test_the_retired_scope_note_is_gone_from_apply_force(self) -> None:
        """A comment claiming bool is intentionally accepted must not outlive the behaviour.

        A stale note is worse than none: the next reader takes it as the contract
        and re-opens a settled question.
        """
        from strands_robots.simulation.mujoco import physics

        source = inspect.getsource(physics.PhysicsMixin.apply_force)
        assert "bool is intentionally accepted" not in source
        assert "out of scope for numeric-element validation" not in source


requires_mujoco = pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="mujoco not installed",
)

# Inline robot XML - no network dependency on robot model repos, and the joint
# and body names below are the fixture's own rather than a registry robot's.
_ROBOT_XML = """
<mujoco model="bool_writer_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="elbow" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def arm_sim(tmp_path):
    """A simulation with one two-joint arm, reproducing #1838's surfaces."""
    from strands_robots.simulation import Simulation

    path = tmp_path / "bool_writer_arm.xml"
    path.write_text(_ROBOT_XML)

    sim = Simulation()
    sim.create_world(timestep=0.002)
    result = sim.add_robot("arm", urdf_path=str(path), position=[0.0, 0.0, 0.0])
    assert result["status"] == "success", f"add_robot failed: {result}"
    sim.step(n_steps=5)
    yield sim
    sim.destroy()


@pytest.fixture
def arm_joint(arm_sim):
    """The first addressable joint name, resolved rather than assumed."""
    joints = arm_sim.robot_joint_names("arm")
    assert joints, "fixture robot must have joints"
    return joints[0]


@requires_mujoco
class TestSetJointPositionsRefusesABoolean:
    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_a_boolean_position_is_refused(self, arm_sim, arm_joint: str, value: Any) -> None:
        res = arm_sim.set_joint_positions({arm_joint: value})
        assert res["status"] == "error", f"{value!r} was written as a position"
        assert "not a bool" in res["content"][0]["text"]

    def test_a_refused_position_leaves_qpos_untouched(self, arm_sim, arm_joint: str) -> None:
        """Pre-fix this wrote 1.0 rad and reported "Set 1/1 joint positions"."""
        before = arm_sim._world._data.qpos.copy()
        res = arm_sim.set_joint_positions({arm_joint: True})
        assert res["status"] == "error"
        np.testing.assert_array_equal(arm_sim._world._data.qpos, before)

    def test_a_numeric_position_is_still_written(self, arm_sim, arm_joint: str) -> None:
        res = arm_sim.set_joint_positions({arm_joint: 0.25})
        assert res["status"] == "success", res

    def test_a_one_radian_position_is_still_written(self, arm_sim, arm_joint: str) -> None:
        """1.0 is a legitimate target; only the boolean spelling of it is refused."""
        assert arm_sim.set_joint_positions({arm_joint: 1})["status"] == "success"

    def test_a_numpy_position_is_still_written(self, arm_sim, arm_joint: str) -> None:
        assert arm_sim.set_joint_positions({arm_joint: np.float64(0.4)})["status"] == "success"


@requires_mujoco
class TestSetJointVelocitiesRefusesABoolean:
    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_a_boolean_velocity_is_refused(self, arm_sim, arm_joint: str, value: Any) -> None:
        res = arm_sim.set_joint_velocities({arm_joint: value})
        assert res["status"] == "error", f"{value!r} was written as a velocity"
        assert "not a bool" in res["content"][0]["text"]

    def test_a_refused_velocity_leaves_qvel_untouched(self, arm_sim, arm_joint: str) -> None:
        before = arm_sim._world._data.qvel.copy()
        res = arm_sim.set_joint_velocities({arm_joint: np.True_})
        assert res["status"] == "error"
        np.testing.assert_array_equal(arm_sim._world._data.qvel, before)

    def test_a_numeric_velocity_is_still_written(self, arm_sim, arm_joint: str) -> None:
        assert arm_sim.set_joint_velocities({arm_joint: 0.5})["status"] == "success"


@requires_mujoco
class TestApplyForceRefusesABooleanComponent:
    """The surface whose comment deferred the question; the note retires with it."""

    @pytest.mark.parametrize("vector", ["force", "torque", "point"])
    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_a_boolean_component_is_refused_on_every_vector(self, arm_sim, vector: str, value: Any) -> None:
        # point alone is not a wrench, so it needs a force to be reached at all.
        kwargs: dict[str, Any] = {} if vector == "force" else {"force": [1.0, 0.0, 0.0]}
        kwargs[vector] = [value, 0.0, 0.0]
        res = arm_sim.apply_force(body_name="arm/base", **kwargs)
        assert res["status"] == "error", f"{value!r} was accepted as a {vector} component"
        text = res["content"][0]["text"]
        assert f"'{vector}'" in text
        assert "not a bool" in text
        assert _BOOLEAN_STATE_REASON in text

    def test_a_refused_wrench_latches_nothing(self, arm_sim) -> None:
        """Pre-fix this reported "Force: [1.0, 0.0, 0.0] N" and latched it."""
        before = arm_sim._world._data.xfrc_applied.copy()
        res = arm_sim.apply_force(body_name="arm/base", force=[True, 0.0, 0.0])
        assert res["status"] == "error"
        np.testing.assert_array_equal(arm_sim._world._data.xfrc_applied, before)

    def test_a_numeric_wrench_is_still_latched(self, arm_sim) -> None:
        res = arm_sim.apply_force(body_name="arm/base", force=[1.0, 0.0, 0.0])
        assert res["status"] == "success", res
        assert arm_sim._world._data.xfrc_applied.any()

    def test_a_numpy_wrench_is_still_latched(self, arm_sim) -> None:
        """A computed wrench arrives as an array of numpy floats, not python floats."""
        res = arm_sim.apply_force(body_name="arm/base", force=np.array([0.0, 0.0, 2.5]))
        assert res["status"] == "success", res

    def test_the_documented_way_to_clear_a_force_still_works(self, arm_sim) -> None:
        """force=[0, 0, 0] is the documented stop; the bool gate must not catch the zeros."""
        assert arm_sim.apply_force(body_name="arm/base", force=[10.0, 0.0, 0.0])["status"] == "success"
        assert arm_sim.apply_force(body_name="arm/base", force=[0, 0, 0])["status"] == "success"

    def test_a_torque_only_call_is_still_accepted(self, arm_sim) -> None:
        assert arm_sim.apply_force(body_name="arm/base", torque=[0.0, 0.0, 0.1])["status"] == "success"


@requires_mujoco
class TestTheStateWritersAndTheSceneVectorsNowAgree:
    """The asymmetry #1838 reported: the same value, the same answer.

    A boolean was refused by ``add_object(position=...)`` and accepted by
    ``set_joint_positions``. Whatever the domain is, one library should not answer
    the question two ways.
    """

    def test_both_surfaces_refuse_a_boolean_component(self, arm_sim, arm_joint: str) -> None:
        scene = arm_sim.add_object(name="b1", shape="box", position=[True, 0.0, 0.3])
        assert scene["status"] == "error"
        assert arm_sim.set_joint_positions({arm_joint: True})["status"] == "error"

    def test_both_surfaces_accept_the_numeric_form(self, arm_sim, arm_joint: str) -> None:
        scene = arm_sim.add_object(name="b2", shape="box", position=[1.0, 0.0, 0.3])
        assert scene["status"] == "success", scene
        assert arm_sim.set_joint_positions({arm_joint: 1.0})["status"] == "success"
