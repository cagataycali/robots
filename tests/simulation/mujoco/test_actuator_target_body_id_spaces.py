"""An actuator's target id is read from the space its own transmission selects.

:func:`~strands_robots.simulation.mujoco.scene_ops.actuator_target_body_ids`
answers "what part of the machine does this actuator move", and it resolves that
from ``actuator_trnid[act, 0]`` -- one integer whose MEANING is chosen by
``actuator_trntype``: a joint id, a site id, a body id or a tendon id. Those
spaces all start at 0 and have different lengths, so the same integer is a valid
id in one and past the end of another, and every branch therefore guards the
read before it indexes.

This module pins what those guards return, which is the same answer in every
arm: no body. That is the load-bearing part, because the two ways an unguarded
read can go wrong are both worse than an empty set and neither is a raised
error the caller could act on:

* ``jnt_bodyid[target]`` / ``site_bodyid[target]`` past the end raises
  ``IndexError`` from inside a derivation the caller did not ask about;
* a NEGATIVE target indexes from the end, so it names a real body of the scene
  -- the last one -- and a body transmission hands the id straight back, so an
  out-of-space id becomes a plausible-looking wrong answer with nothing to
  distinguish it from a right one.

The consumer is what makes the wrong answer expensive: the bodies an actuator
moves are the SECOND seed
:meth:`~strands_robots.simulation.mujoco.rendering.RenderingMixin._robot_base_free_joint`
walks up to find a floating base, and it is the only seed an aerial robot has --
its rotors are forces at sites on the airframe, so it declares no joint besides
the unnamed base. That docstring records what a base read as absent costs:
``get_observation`` returns no state for a robot that is in the scene and
moving, and ``start_recording`` declares a dataset with no
``observation.state`` column while still recording the actions, "so the episode
trains nothing and reports success".

No MJCF spells an out-of-space transmission id -- the compiler emits only ids it
resolved -- so, like the sibling unkeyable-state suite, the ids are written into
the compiled model, which is the state these guards exist to survive. The model
itself is real MuJoCo and the healthy resolutions are asserted first, so a row
below fails when the guard goes, not when the fixture stops compiling.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import mujoco as mj  # noqa: E402

from strands_robots.simulation.models import SimWorld  # noqa: E402
from strands_robots.simulation.mujoco.scene_ops import actuator_target_body_ids  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Resolved at module scope: mjTRN_SO3 arrived in mujoco 3.12; the manifest floor is 3.5.
_SO3_TRN = getattr(mj.mjtTrn, "mjTRN_SO3", None)

# A hinge arm whose second link carries a site driven by a site-transmission
# actuator. Deliberately three different space lengths (njnt=2, nsite=1,
# nbody=4), so an id can be past the end of one space while still naming an
# element of another - the collision the guards are placed against.
_ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.04 0.04 0.04"/>
      <body name="link1" pos="0 0 0.08">
        <joint name="shoulder" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.18 0 0" size="0.02"/>
        <body name="link2" pos="0.18 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2 2" damping="1"/>
          <geom type="capsule" fromto="0 0 0 0.14 0 0" size="0.018"/>
          <site name="tip" pos="0.14 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="30"/>
    <general name="tip_act" site="tip" gear="0 0 1 0 0 0"/>
  </actuator>
</mujoco>
"""

_JOINT_ACT = 0
_SITE_ACT = 1


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_actuator_target_spaces", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


@pytest.fixture
def world(sim: Simulation) -> SimWorld:
    if sim._world is None:
        assert sim.create_world()["status"] == "success"
    assert sim.replace_scene_mjcf(_ARM_XML)["status"] == "success"
    w = sim._world
    assert w is not None and w._model is not None
    return w


class TestATargetInsideItsOwnSpaceNamesTheBody:
    """Premises. Every refusal below has to be about the id, not about the fixture."""

    def test_each_transmission_resolves_the_body_it_moves(self, world: SimWorld) -> None:
        model = world._model
        link2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link2")
        assert actuator_target_body_ids(model, _JOINT_ACT, mj) == frozenset({link2 - 1}), (
            "the shoulder hinge belongs to link1"
        )
        assert actuator_target_body_ids(model, _SITE_ACT, mj) == frozenset({link2}), "the tip site is on link2"

    def test_the_spaces_have_the_differing_lengths_the_rows_assume(self, world: SimWorld) -> None:
        model = world._model
        assert (int(model.njnt), int(model.nsite), int(model.nbody)) == (2, 1, 4)
        # The premise that makes the guards load-bearing rather than decorative:
        # each out-of-space id used below is a real element of some OTHER space.
        assert int(model.nsite) < int(model.njnt) < int(model.nbody)


class TestATargetOutsideItsOwnSpaceNamesNoBody:
    """One integer, four meanings: past the end of the selected space, nothing is named."""

    @pytest.mark.parametrize(
        ("space", "trntype_name", "target"),
        [
            ("joint", "mjTRN_JOINT", 2),  # njnt == 2
            ("joint (in parent frame)", "mjTRN_JOINTINPARENT", 5),
            ("site", "mjTRN_SITE", 1),  # nsite == 1
            ("site (slider-crank)", "mjTRN_SLIDERCRANK", 3),
            ("body", "mjTRN_BODY", 4),  # nbody == 4
        ],
    )
    def test_an_id_past_the_end_of_its_space_names_no_body(
        self, world: SimWorld, space: str, trntype_name: str, target: int
    ) -> None:
        trntype = getattr(mj.mjtTrn, trntype_name, None)
        if trntype is None:
            pytest.skip(f"{trntype_name} is not defined by this mujoco build")
        model = world._model
        model.actuator_trntype[_JOINT_ACT] = int(trntype)
        model.actuator_trnid[_JOINT_ACT][0] = target
        assert actuator_target_body_ids(model, _JOINT_ACT, mj) == frozenset(), (
            f"a {space} id of {target} is past the end of that space"
        )

    def test_an_unguarded_read_would_raise_rather_than_return(self, world: SimWorld) -> None:
        """Why the joint and site arms guard: the read itself fails, mid-derivation."""
        model = world._model
        with pytest.raises(IndexError):
            model.jnt_bodyid[int(model.njnt)]
        with pytest.raises(IndexError):
            model.site_bodyid[int(model.nsite)]

    def test_a_body_id_past_the_end_would_be_handed_back_unread(self, world: SimWorld) -> None:
        """Why the body arm guards: it returns the id itself, so nothing reads it and nothing fails."""
        model = world._model
        past_end = int(model.nbody)
        assert mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, past_end) is None, "premise: no such body"
        model.actuator_trntype[_JOINT_ACT] = int(mj.mjtTrn.mjTRN_BODY)
        model.actuator_trnid[_JOINT_ACT][0] = past_end
        assert past_end not in actuator_target_body_ids(model, _JOINT_ACT, mj)


class TestANegativeTargetNamesNoBodyRatherThanTheLastOne:
    """The arm with no error to fall back on: a negative id indexes from the end.

    Checked before the transmission is even branched on, because it is wrong in
    every space, and every space would answer it with a real element.
    """

    @pytest.mark.parametrize("trntype_name", ["mjTRN_JOINT", "mjTRN_SITE", "mjTRN_BODY"])
    def test_a_negative_target_names_no_body(self, world: SimWorld, trntype_name: str) -> None:
        model = world._model
        model.actuator_trntype[_JOINT_ACT] = int(getattr(mj.mjtTrn, trntype_name))
        model.actuator_trnid[_JOINT_ACT][0] = -1
        assert actuator_target_body_ids(model, _JOINT_ACT, mj) == frozenset()

    def test_the_answer_a_negative_read_would_have_given_is_a_real_body(self, world: SimWorld) -> None:
        """The measurement that makes the guard load-bearing, not tidiness."""
        model = world._model
        would_be = {int(model.jnt_bodyid[-1]), int(model.site_bodyid[-1])}
        assert all(0 <= b < int(model.nbody) for b in would_be), (
            "an unguarded negative read names existing bodies, so the wrong answer is indistinguishable"
        )
        assert all(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in would_be)


class TestATransmissionWithNoRuleNamesNoBody:
    """The fall-through: a transmission type this function has no resolution for.

    ``mjTRN_SO3`` is the live example -- ``mujoco`` 3.12 added it for a torque on
    a relative orientation, and this function resolves the other six types. It
    cannot arrive through the backend, which refuses a multi-control actuator
    (an ``<orientation>`` occupies three control slots) when the scene compiles,
    so this is the belt to that brace: were that refusal ever to move, the
    answer is still "no body" and not the id read against the wrong space.
    """

    @pytest.mark.skipif(_SO3_TRN is None, reason="mjTRN_SO3 arrived in mujoco 3.12; the manifest floor is 3.5")
    def test_an_unresolved_transmission_type_names_no_body(self, world: SimWorld) -> None:
        so3 = _SO3_TRN
        assert so3 is not None, "the skipif above admits only a build that defines mjTRN_SO3"
        model = world._model
        model.actuator_trntype[_JOINT_ACT] = int(so3)
        # An id that IS a valid joint, site and body, so an empty answer is the
        # rule being absent and not the id being out of range.
        model.actuator_trnid[_JOINT_ACT][0] = 0
        assert actuator_target_body_ids(model, _JOINT_ACT, mj) == frozenset()
