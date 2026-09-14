"""State a scene rebuild cannot carry, because nothing in the new model answers to its key.

A rebuild compiles a fresh model and allocates a fresh ``MjData``, so the
dynamic state has to be carried across that gap BY NAME -- the indices all shift
(see :class:`~strands_robots.simulation.mujoco.scene_ops._SceneState`). The
happy half of that is well pinned: the per-type joint widths round-trip, an
unnamed joint is carried through its body, a latched wrench survives.

This module pins the other half, which is one question asked of six surfaces:
what happens to a value whose HANDLE does not survive? Every carry helper in
``scene_ops`` answers it separately, and they do not answer it the same way:

* nothing names it, so no key can be formed -- the entry is dropped, and
  ``_snapshot_body_wrenches`` and ``_snapshot_scene_state`` say so at ``DEBUG``,
  the level for "this was never carryable" (``_snapshot_joint_forces`` drops it
  without a line, so the same rebuild reports the same joint once rather than
  twice);
* the key was formed but resolves to nothing in the rebuilt model -- dropped
  SILENTLY, because a removed element's absence is the point of the rebuild
  that removed it;
* the key resolves but the element's WIDTH changed -- dropped at ``WARNING``,
  because a same-named joint that changed type or an actuator that lost its
  activation state is not an expected outcome of a rebuild, and writing the old
  slice into it would put one dof's force on another.

The distinction matters because the quiet cases must stay quiet -- an eject logs
a warning per removed element otherwise -- while the width cases must not be:
they are the only ones where a caller's value is discarded without the caller
having asked for that.

The models are compiled MuJoCo, not doubles, so the anonymity is the compiler's
own: an unnamed ``<body>`` and an unnamed ``<freejoint/>`` are the ordinary MJCF
spelling for a prop, and ``mj_id2name`` reports them as ``None``. Two arms have
no MJCF spelling at all -- an actuator driving through a transmission this build
cannot key -- so those are written into the compiled model, and the transmission
used is a real one ``mujoco`` ships (:class:`TestUnkeyedTransmissionsAreRefused`
measures why that one cannot arrive through a rebuild).
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("mujoco")

import mujoco as mj  # noqa: E402

from strands_robots.simulation.models import SimWorld  # noqa: E402
from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Resolve at module scope: mjTRN_SO3 arrived in mujoco 3.12; the manifest floor is 3.5.
_SO3_TRN = getattr(mj.mjtTrn, "mjTRN_SO3", None)

_LOGGER = "strands_robots.simulation.mujoco.scene_ops"

# One named body with a named freejoint, and one of each with no name at all.
# The named pair is the control: it proves a drop is about the missing name and
# not about the surface refusing every entry.
_ANONYMOUS_XML = """
<mujoco model="anonymous">
  <compiler angle="radian"/>
  <worldbody>
    <body name="kept" pos="0 0 1">
      <freejoint name="kept_j"/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
    <body pos="0.4 0 1">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

# An unnamed actuator on a named joint, plus an unnamed joint for its
# transmission to be re-pointed at.
_UNNAMED_ACTUATOR_XML = """
<mujoco model="unnamed_actuator">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hinger" pos="0 0 0.2">
      <joint name="h" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
    <body name="slider" pos="0.3 0 0.2">
      <joint type="slide" axis="1 0 0"/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="h"/>
  </actuator>
</mujoco>
"""

# The same bodies before and after a rebuild, carrying one handle of each kind:
# "k" is unchanged, "j" is a hinge (1 dof) that comes back a freejoint (6), and
# "floater"'s unnamed freejoint -- the element that made ("body", "floater") a
# key at all -- is gone, leaving the body static. One restore against this pair
# therefore exercises all three outcomes at once.
#
# The bodies are also declared in a different order on the two sides, which is
# what the name-keying exists for: every id shifts, and the free joint ends up
# LAST so that an unchecked negative body id would resolve to it.
_BEFORE_XML = """
<mujoco model="before">
  <compiler angle="radian"/>
  <worldbody>
    <body name="b" pos="0 0 0.5">
      <joint name="j" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
    <body name="floater" pos="0.5 0 1">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="kept" pos="-0.5 0 0.5">
      <joint name="k" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

_AFTER_XML = """
<mujoco model="after">
  <compiler angle="radian"/>
  <worldbody>
    <body name="floater" pos="0.5 0 1">
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="kept" pos="-0.5 0 0.5">
      <joint name="k" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
    <body name="b" pos="0 0 0.5">
      <freejoint name="j"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

# One named actuator that owns an activation state, and the same name coming
# back as a plain motor, which owns none.
_WITH_ACTIVATION_XML = """
<mujoco model="with_activation">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hinger" pos="0 0 0.2">
      <joint name="h" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <general name="a" joint="h" dyntype="filter" dynprm="0.1" gainprm="1"/>
  </actuator>
</mujoco>
"""

_WITHOUT_ACTIVATION_XML = _WITH_ACTIVATION_XML.replace(
    '<general name="a" joint="h" dyntype="filter" dynprm="0.1" gainprm="1"/>',
    '<motor name="a" joint="h"/>',
).replace("with_activation", "without_activation")


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_unkeyable_state", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _scene(sim: Simulation, xml: str) -> SimWorld:
    """Compile ``xml`` as ``sim``'s whole scene and return the live world."""
    if sim._world is None:
        assert sim.create_world()["status"] == "success"
    assert sim.replace_scene_mjcf(xml)["status"] == "success"
    world = sim._world
    assert world is not None and world._model is not None and world._data is not None
    return world


class TestAnonymousElementsAreNotCarried:
    """Nothing names them, so no key can be formed - reported at DEBUG, not lost silently."""

    @pytest.fixture
    def world(self, sim: Simulation) -> SimWorld:
        world = _scene(sim, _ANONYMOUS_XML)
        data = world._data
        for bid in range(1, int(world._model.nbody)):
            data.xfrc_applied[bid] = [1.0 + bid, 0.0, 0.0, 0.0, 0.0, 0.5]
        data.qfrc_applied[:] = [0.1 * (i + 1) for i in range(len(data.qfrc_applied))]
        return world

    def test_the_scene_really_holds_an_anonymous_body_and_joint(self, world: SimWorld) -> None:
        """Premise. Without this the drops below would be vacuous rather than measured."""
        model = world._model
        assert [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in range(int(model.nbody))] == [
            "world",
            "kept",
            None,
        ]
        assert [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(int(model.njnt))] == ["kept_j", None]
        assert scene_ops._joint_key(model, 1, mj) is None, "no name on the joint or on its body"

    def test_wrench_on_an_unnamed_body_is_not_carried(self, world: SimWorld, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            wrenches = scene_ops._snapshot_body_wrenches(world._model, world._data, mj)
        assert set(wrenches) == {"kept"}, "the named body's wrench is still carried"
        assert "body id 2 has no name, wrench not carried over" in caplog.text

    def test_force_on_an_unkeyable_joint_is_not_carried(self, world: SimWorld, caplog) -> None:
        """Dropped without a line: ``_snapshot_scene_state`` walks the same joints and
        already reports this one, so a line here would double every rebuild's report."""
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            forces = scene_ops._snapshot_joint_forces(world._model, world._data, mj)
        assert set(forces) == {("joint", "kept_j")}
        assert caplog.records == []

    def test_scene_snapshot_reports_the_joint_it_cannot_key(self, world: SimWorld, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            snapshot = scene_ops._snapshot_scene_state(world)
        assert set(snapshot.joints) == {("joint", "kept_j")}
        assert set(snapshot.body_wrenches) == {"kept"}
        assert "neither joint id 1 nor its body carries a name, state not carried over" in caplog.text


class TestActuatorsWithNoUsableHandle:
    """An unnamed actuator is keyed by what it drives, so both halves of that key can fail.

    Neither arm has an MJCF spelling - an actuator references its target by name,
    and every transmission a compiled model can carry that this build does not
    key is refused before a snapshot sees it (:class:`TestUnkeyedTransmissionsAreRefused`).
    Both are therefore written into the compiled model, which is the state the
    code guards against reaching.
    """

    @pytest.fixture
    def world(self, sim: Simulation) -> SimWorld:
        return _scene(sim, _UNNAMED_ACTUATOR_XML)

    def test_an_unnamed_actuator_is_normally_keyed_by_its_target(self, world: SimWorld) -> None:
        """Premise. The poked arms below are about the key failing, not about it never forming."""
        assert scene_ops._actuator_key(world._model, 0, mj) == ("target", int(mj.mjtTrn.mjTRN_JOINT), "h", 0)

    @pytest.mark.parametrize("break_key", ["unkeyed transmission", "unnamed target"])
    def test_an_unresolvable_target_yields_no_key(self, world: SimWorld, break_key: str) -> None:
        model = world._model
        if break_key == "unkeyed transmission":
            if _SO3_TRN is None:
                pytest.skip("mjTRN_SO3 arrived in mujoco 3.12; the manifest floor is 3.5")
            model.actuator_trntype[0] = int(_SO3_TRN)
        else:
            unnamed = [j for j in range(int(model.njnt)) if not mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)]
            assert unnamed, "premise: the scene carries an unnamed joint to re-point at"
            model.actuator_trnid[0][0] = unnamed[0]
        assert scene_ops._actuator_key(model, 0, mj) is None

    @pytest.mark.skipif(_SO3_TRN is None, reason="mjTRN_SO3 arrived in mujoco 3.12; the manifest floor is 3.5")
    def test_scene_snapshot_reports_the_actuator_it_cannot_key(self, world: SimWorld, caplog) -> None:
        so3 = _SO3_TRN
        assert so3 is not None, "the skipif above admits only a build that defines mjTRN_SO3"
        world._model.actuator_trntype[0] = int(so3)
        world._data.ctrl[0] = 0.42
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            snapshot = scene_ops._snapshot_scene_state(world)
        assert snapshot.actuators == {}, "a setpoint with no handle is dropped, not mis-keyed"
        assert "actuator id 0 drives through a transmission this build cannot key" in caplog.text


class TestRestoreSkipsWhatTheRebuiltModelCannotTake:
    """A key that resolves to nothing is dropped in silence; one whose width changed is not."""

    @pytest.fixture
    def rebuilt(self, sim: Simulation) -> tuple[SimWorld, dict, scene_ops._SceneState]:
        """A real rebuild in which one handle loses its width and one loses its element."""
        world = _scene(sim, _BEFORE_XML)
        world._data.qfrc_applied[:] = 0.7
        forces = scene_ops._snapshot_joint_forces(world._model, world._data, mj)
        snapshot = scene_ops._snapshot_scene_state(world)
        assert set(forces) == {("joint", "j"), ("joint", "k"), ("body", "floater")}
        assert [len(forces[key]) for key in (("joint", "j"), ("joint", "k"), ("body", "floater"))] == [1, 1, 6]
        _scene(sim, _AFTER_XML)
        world._data.qfrc_applied[:] = 0.0
        return world, forces, snapshot

    def test_a_key_the_new_model_lacks_is_dropped_without_a_warning(self, rebuilt, caplog) -> None:
        """The ejected robot's own elements come through here on every eject."""
        world, forces, _ = rebuilt
        for resolvable in (("joint", "j"), ("joint", "k")):
            forces.pop(resolvable)  # keep only keys that resolve to nothing
        forces[("joint", "never_existed")] = [0.5]
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            scene_ops._restore_joint_forces(world._model, world._data, forces, mj)
        assert max(abs(float(x)) for x in world._data.qfrc_applied) == 0.0
        assert caplog.records == [], "a removed element's absence is the point of the rebuild"

    def test_a_joint_whose_dof_width_changed_is_reported(self, rebuilt, caplog) -> None:
        """One call, three verdicts: "k" is written back, "j" is refused loudly, "floater" quietly."""
        world, forces, _ = rebuilt
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            scene_ops._restore_joint_forces(world._model, world._data, forces, mj)
        assert [r.getMessage() for r in caplog.records] == [
            "_restore_joint_forces: dof width mismatch for ('joint', 'j') (1!=6), skipping"
        ]
        carried = scene_ops._snapshot_joint_forces(world._model, world._data, mj)
        assert carried == {("joint", "k"): [0.7]}, "the surviving joint alone, at its own width"

    def test_a_body_key_whose_free_joint_is_gone_resolves_to_nothing(self, rebuilt) -> None:
        """``("body", name)`` names an unnamed free joint through its body, so the body
        surviving is not enough - it has to still carry that one joint."""
        world, _, _ = rebuilt
        model = world._model
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "floater")
        assert bid >= 0 and int(model.body_jntnum[bid]) == 0, "premise: body kept, joint gone"
        assert scene_ops._resolve_joint_key(model, ("body", "floater"), mj) == -1

    def test_a_body_key_naming_no_body_at_all_resolves_to_nothing(self, rebuilt) -> None:
        """A name the rebuilt model does not carry must not index it from the end.

        ``mj_name2id`` reports a miss as ``-1``, and every ``model.body_*`` array
        accepts that as a numpy index onto the LAST body - which here owns a
        single free joint, exactly the shape a ``("body", name)`` key looks for.
        So the miss has to be caught before the lookup, not after it.
        """
        world, _, _ = rebuilt
        model = world._model
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "never_existed") == -1
        last = int(model.nbody) - 1
        assert int(model.body_jntnum[last]) == 1, "premise: the last body would answer the lookup"
        assert int(model.jnt_type[int(model.body_jntadr[last])]) == int(mj.mjtJoint.mjJNT_FREE)
        assert scene_ops._resolve_joint_key(model, ("body", "never_existed"), mj) == -1

    def test_only_the_surviving_joint_is_restored_by_the_eject_surface(self, rebuilt) -> None:
        """The same three verdicts through the surface an eject actually calls."""
        world, _, snapshot = rebuilt
        assert set(snapshot.joints) == {("joint", "j"), ("joint", "k"), ("body", "floater")}
        assert scene_ops._restore_scene_state(world, snapshot) == 1


class TestActivationStateThatNoLongerFits:
    """An actuator that kept its name but lost its activation state keeps its setpoint."""

    def test_ctrl_is_restored_and_the_orphaned_activation_is_reported(self, sim: Simulation, caplog) -> None:
        world = _scene(sim, _WITH_ACTIVATION_XML)
        assert [int(x) for x in world._model.actuator_actnum] == [1], "premise: it owns one activation"
        world._data.ctrl[0] = 0.33
        world._data.act[:] = 0.44
        snapshot = scene_ops._snapshot_scene_state(world)
        assert snapshot.actuators == {("actuator", "a"): (0.33, [0.44])}

        _scene(sim, _WITHOUT_ACTIVATION_XML)
        assert int(world._model.na) == 0, "premise: the rebuilt actuator owns none"
        world._data.ctrl[0] = 0.0
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            scene_ops._restore_scene_state(world, snapshot)
        assert float(world._data.ctrl[0]) == pytest.approx(0.33), "the setpoint still fits, so it is carried"
        assert [r.getMessage() for r in caplog.records] == [
            "_restore_scene_state: act width mismatch for actuator ('actuator', 'a') (1!=0), skipping activation"
        ]


class TestUnkeyedTransmissionsAreRefused:
    """Why an actuator this build cannot key never reaches a snapshot through a rebuild.

    :func:`~strands_robots.simulation.mujoco.scene_ops._actuator_target_kind`
    maps six of the eight ``mjtTrn`` members ``mujoco`` ships. That is not a gap
    as long as the two it omits cannot arrive: ``mjTRN_UNDEFINED`` is not a
    transmission a compiled actuator carries, and ``mjTRN_SO3`` -- the
    ``<orientation>`` actuator added in mujoco 3.12 -- spans three control slots,
    which this backend refuses outright before the model is installed.

    A future single-control transmission would break that argument, so the
    exemption list is pinned rather than the omission being left implicit.
    """

    def test_only_the_accounted_for_transmissions_are_unmapped(self) -> None:
        members = {name: int(getattr(mj.mjtTrn, name)) for name in dir(mj.mjtTrn) if name.startswith("mjTRN")}
        assert len(members) >= 7, "premise: the enum was read, not missed"
        unmapped = {name for name, value in members.items() if scene_ops._actuator_target_kind(value, mj) is None}
        expected = {"mjTRN_UNDEFINED", "mjTRN_SO3"} if _SO3_TRN is not None else {"mjTRN_UNDEFINED"}
        assert unmapped == expected

    @pytest.mark.skipif(_SO3_TRN is None, reason="mjTRN_SO3 arrived in mujoco 3.12; the manifest floor is 3.5")
    def test_an_so3_actuator_is_refused_before_it_can_be_keyed(self) -> None:
        model = mj.MjModel.from_xml_string(
            """
            <mujoco model="so3">
              <worldbody>
                <site name="ref"/>
                <body name="b" pos="0 0 1">
                  <freejoint/>
                  <geom type="box" size="0.1 0.1 0.1"/>
                  <site name="s"/>
                </body>
              </worldbody>
              <actuator><orientation site="s" refsite="ref"/></actuator>
            </mujoco>
            """
        )
        assert (int(model.nactuator), int(model.nu)) == (1, 3), "premise: one actuator, three control slots"
        assert scene_ops._actuator_key(model, 0, mj) is None, "unnamed, and its transmission is unmapped"
        reason = scene_ops._unaddressable_actuator_reason(model, mj)
        assert reason is not None and "owns control slots [0, 1, 2]" in reason
