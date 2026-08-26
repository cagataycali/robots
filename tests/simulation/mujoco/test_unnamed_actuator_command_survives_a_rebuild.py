"""An unnamed actuator keeps its command when a scene rebuild renumbers the model.

``eject_robot_from_scene`` compiles a fresh model and allocates a fresh
``MjData``, which renumbers every actuator id, so the dynamic state is
snapshotted under a name-based key and written back afterwards. The joint half
of that snapshot learned to carry an unnamed element through its owning body
plus an ordinal; the actuator half still keyed on the actuator's own name alone
and skipped anything unnamed, on the grounds that "its transmission target may
drive several actuators, so it cannot be matched across the rebuild".

A skipped actuator's ``ctrl`` and ``act`` were dropped, so it came back at its
fresh-compile zero while the operation reported success -- the setpoints holding
a surviving robot's pose, and the activation of a stateful actuator, silently
reset because some *other* robot was removed.

The premise was sound and the conclusion too strong, the same way it was for
joints: the target alone does not single the actuator out, yet the target *plus
its position among the actuators driving that target* does. MuJoCo stores
actuators in declaration order and an eject removes a robot's actuators
wholesale, so the surviving order is preserved, and no scene op inserts an
actuator (the patch vocabulary is ``add_body`` / ``add_geom`` / ``add_site`` /
``set_body_pos`` / ``set_body_quat`` / ``delete_body``).

An unnamed ``<actuator>`` child is the ordinary MJCF spelling: of the 235
actuator-bearing models in the MuJoCo Menagerie tree, 7 leave every one of
theirs unnamed -- ``ufactory_lite6`` (6 of 6), ``google_robot`` (9 of 9) and
``iit_softfoot`` (1 of 1) -- so such a scene lost its whole command vector.

Three properties make the key load-bearing rather than decorative, and each has
its own cell below:

* Four actuators drive one hinge here -- one named, three not -- so the target
  alone cannot tell them apart and keying on it would collapse three commands
  into one. The ordinal counts the named one too, because the snapshot and the
  restore have to count the same population.
* Two of them name the same joint id through *different* transmissions
  (``mjTRN_JOINT`` and ``mjTRN_JOINTINPARENT``), so the transmission type has to
  be part of the key.
* An actuator that gains a name is carried by its own ``("actuator", name)``
  key, so the target handle must not also claim it.

Still unmatched, deliberately: an actuator driving through a transmission the
``mjtTrn`` mapping does not know. Nothing then identifies its target, so it is
reported rather than guessed at, and the mapping's coverage is derived from the
enum here so a future MuJoCo transmission fails this file instead of silently
falling into that branch.

GL-free: ``mesh=False`` and no rendering, so this runs without a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# An arm whose only actuator is named, so it is carried by the pre-existing
# ``("actuator", name)`` key. It is the robot the tests eject.
_ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.05">
      <geom type="box" size="0.04 0.04 0.05"/>
      <body name="link" pos="0 0 0.1">
        <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.14 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="10"/>
  </actuator>
</mujoco>
"""

# The furniture that must survive the eject untouched. Its actuator block is the
# whole point: a NAMED actuator on the hinge ahead of three unnamed ones, so the
# ordinal has to count named siblings too; a fourth unnamed one reaching the same
# hinge through a different transmission; a fifth on a body; and a second named
# one as the control. The stateful ``dyntype="integrator"`` entry also exercises
# the ``act`` activation slice.
_FURNITURE_XML = """
<mujoco model="furniture">
  <compiler angle="radian"/>
  <worldbody>
    <body name="door_frame" pos="0.5 0 0.2">
      <geom type="box" size="0.02 0.02 0.2"/>
      <body name="door_panel" pos="0 0 0">
        <joint name="door_hinge" type="hinge" axis="0 0 1" range="-3 3" damping="0.5"/>
        <geom type="box" size="0.01 0.15 0.18" pos="0 0.15 0"/>
      </body>
    </body>
    <body name="drawer" pos="-0.5 0 0.2">
      <joint name="drawer_slide" type="slide" axis="1 0 0" range="-1 1" damping="0.5"/>
      <geom type="box" size="0.06 0.06 0.06"/>
    </body>
  </worldbody>
  <actuator>
    <position name="door_brake" joint="door_hinge" kp="1"/>
    <position joint="door_hinge" kp="5"/>
    <velocity joint="door_hinge" kv="1"/>
    <general joint="door_hinge" dyntype="integrator" gainprm="1"/>
    <general jointinparent="door_hinge" gear="1"/>
    <adhesion body="drawer" ctrlrange="0 1" gain="1"/>
    <position name="drawer_slide_act" joint="drawer_slide" kp="1"/>
  </actuator>
</mujoco>
"""

#: One distinctive ``ctrl`` per unnamed actuator, in id order. They differ from
#: each other so a swap between two actuators sharing a target is visible, and
#: none is zero so a value lost to a fresh compile is visible too. The adhesion
#: actuator's ``ctrlrange`` is ``0 1``, so its value is positive.
_UNNAMED_CTRL = (0.63, -0.42, 0.19, -0.77, 0.31)

#: The activation the one stateful unnamed actuator is seeded with.
_ACTIVATION = 1.37

#: The named furniture actuator's setpoint, restored by the pre-existing key.
_NAMED_CTRL = 0.55


def _actuator_names(model: Any) -> list[str | None]:
    """Every actuator name the model carries, in id order (``None`` if unnamed)."""
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) for aid in range(int(model.nu))]


def _unnamed_actuator_ids(model: Any) -> list[int]:
    """Every actuator id the model carries no name for, in id order."""
    return [aid for aid, name in enumerate(_actuator_names(model)) if not name]


def _commands(model: Any, data: Any) -> dict[Any, float]:
    """``key -> ctrl`` for every actuator, keyed the way the snapshot keys them."""
    return {scene_ops._actuator_key(model, aid, mujoco): float(data.ctrl[aid]) for aid in range(int(model.nu))}


def _activations(model: Any, data: Any) -> dict[Any, list[float]]:
    """``key -> act`` slice for every actuator that carries activation state."""
    out: dict[Any, list[float]] = {}
    for aid in range(int(model.nu)):
        adr = int(model.actuator_actadr[aid])
        num = int(model.actuator_actnum[aid])
        if adr >= 0 and num:
            out[scene_ops._actuator_key(model, aid, mujoco)] = [float(x) for x in data.act[adr : adr + num]]
    return out


def _stateful_id(model: Any) -> int:
    """The id of the one actuator carrying an activation slice."""
    return next(aid for aid in range(int(model.nu)) if int(model.actuator_actadr[aid]) >= 0)


@pytest.fixture
def scene(tmp_path: Path) -> Any:
    """A world holding the arm plus the furniture, with every setpoint seeded.

    Yields the ``Simulation``, the ``{key: ctrl}`` map and the ``{key: act}`` map
    the scene was seeded with, read back through the compiled model so the
    expectations are the model's own view rather than a hand-kept list.
    """
    (tmp_path / "arm.xml").write_text(_ARM_XML)
    (tmp_path / "furniture.xml").write_text(_FURNITURE_XML)

    sim = Simulation(tool_name="test_unnamed_actuator_command_survives_a_rebuild", mesh=False)
    sim.create_world(gravity=[0, 0, -9.81])
    assert sim.add_robot(name="arm", urdf_path=str(tmp_path / "arm.xml"))["status"] == "success"
    assert sim.add_robot(name="fur", urdf_path=str(tmp_path / "furniture.xml"))["status"] == "success"

    world = sim._world
    assert world is not None
    model, data = world._model, world._data
    assert model is not None and data is not None

    unnamed = _unnamed_actuator_ids(model)
    # The premise: the furniture really did compile to five unnamed actuators, so
    # a rename upstream cannot quietly turn these tests into no-ops.
    assert len(unnamed) == len(_UNNAMED_CTRL), _actuator_names(model)

    for aid, value in zip(unnamed, _UNNAMED_CTRL, strict=True):
        data.ctrl[aid] = value
    named = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fur/drawer_slide_act")
    data.ctrl[named] = _NAMED_CTRL
    data.act[int(model.actuator_actadr[_stateful_id(model)])] = _ACTIVATION
    mujoco.mj_forward(model, data)

    yield sim, _commands(model, data), _activations(model, data)
    sim.cleanup()


def _survivors(sim: Any) -> dict[Any, float]:
    """The current ``ctrl`` of every actuator, keyed as seeded."""
    world = sim._world
    return _commands(world._model, world._data)


class TestAnUnnamedActuatorSurvivesTheEject:
    """The regression: removing one robot must not reset the rest of the scene."""

    def test_every_surviving_actuator_keeps_the_command_it_was_seeded_with(self, scene: Any) -> None:
        sim, seeded, _ = scene
        ejected = ("actuator", "arm/pan_act")
        assert sim.remove_robot("arm")["status"] == "success"
        assert _survivors(sim) == {key: value for key, value in seeded.items() if key != ejected}

    def test_the_three_actuators_on_one_hinge_keep_their_distinct_commands(self, scene: Any) -> None:
        """The case that makes the ordinal load-bearing: the target cannot tell them apart."""
        sim, seeded, _ = scene
        joint = int(mujoco.mjtTrn.mjTRN_JOINT)
        shared = {key: value for key, value in seeded.items() if key[:3] == ("target", joint, "fur/door_hinge")}
        assert sorted(key[3] for key in shared) == [1, 2, 3]
        assert len(set(shared.values())) == 3

        sim.remove_robot("arm")
        after = _survivors(sim)
        assert {key: after[key] for key in shared} == shared

    def test_the_actuator_reached_through_another_transmission_keeps_its_command(self, scene: Any) -> None:
        """The case that makes the transmission type load-bearing: same joint id, different key."""
        sim, seeded, _ = scene
        parent = ("target", int(mujoco.mjtTrn.mjTRN_JOINTINPARENT), "fur/door_hinge", 0)
        assert parent in seeded
        sim.remove_robot("arm")
        assert _survivors(sim)[parent] == seeded[parent]

    def test_an_unnamed_body_transmission_keeps_its_command(self, scene: Any) -> None:
        """A transmission whose target is a body, not a joint, is keyed the same way."""
        sim, seeded, _ = scene
        body = ("target", int(mujoco.mjtTrn.mjTRN_BODY), "fur/drawer", 0)
        assert body in seeded
        sim.remove_robot("arm")
        assert _survivors(sim)[body] == seeded[body]

    def test_the_activation_of_a_stateful_unnamed_actuator_survives(self, scene: Any) -> None:
        """``act`` is the effective command of a stateful actuator, so it is carried too."""
        sim, _, activations = scene
        assert list(activations.values()) == [[_ACTIVATION]]
        sim.remove_robot("arm")
        world = sim._world
        assert _activations(world._model, world._data) == activations


class TestThePremisesTheKeyRestsOn:
    """What has to be true of the scene for the cells above to mean anything."""

    def test_the_eject_renumbers_the_surviving_actuators(self, scene: Any) -> None:
        """Without a renumber a positional copy would work and the key would be moot."""
        sim, _, _ = scene
        before = _unnamed_actuator_ids(sim._world._model)
        sim.remove_robot("arm")
        assert _unnamed_actuator_ids(sim._world._model) != before

    def test_adding_a_robot_namespaces_a_named_actuator_and_leaves_an_unnamed_one_unnamed(self, scene: Any) -> None:
        """The namespace is what makes a named actuator resolvable; it cannot name a nameless one."""
        sim, _, _ = scene
        names = _actuator_names(sim._world._model)
        assert "arm/pan_act" in names
        assert "fur/drawer_slide_act" in names
        assert "fur/door_brake" in names
        assert names.count(None) == len(_UNNAMED_CTRL)

    def test_the_three_shared_actuators_really_name_one_target(self, scene: Any) -> None:
        """If they named three targets the ordinal would be decoration."""
        sim, _, _ = scene
        model = sim._world._model
        joint = int(mujoco.mjtTrn.mjTRN_JOINT)
        targets = {
            int(model.actuator_trnid[aid][0])
            for aid in _unnamed_actuator_ids(model)
            if int(model.actuator_trntype[aid]) == joint
        }
        assert len(targets) == 1

    def test_two_transmissions_name_the_same_joint_id(self, scene: Any) -> None:
        """If they named different ids the transmission type would be decoration."""
        sim, _, _ = scene
        model = sim._world._model
        by_type: dict[int, set[int]] = {}
        for aid in _unnamed_actuator_ids(model):
            by_type.setdefault(int(model.actuator_trntype[aid]), set()).add(int(model.actuator_trnid[aid][0]))
        joint = int(mujoco.mjtTrn.mjTRN_JOINT)
        parent = int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)
        assert by_type[joint] == by_type[parent]

    def test_a_named_actuator_shares_a_target_with_the_unnamed_ones(self, scene: Any) -> None:
        """So the ordinal has to count named siblings; skipping them would desynchronise.

        The snapshot and the restore count the same population. If the key
        counted only unnamed siblings the two sides would disagree the moment a
        named actuator shared a target, which this scene arranges.
        """
        sim, _, _ = scene
        model = sim._world._model
        brake = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fur/door_brake")
        assert brake >= 0
        shared = [
            aid
            for aid in _unnamed_actuator_ids(model)
            if int(model.actuator_trntype[aid]) == int(model.actuator_trntype[brake])
            and int(model.actuator_trnid[aid][0]) == int(model.actuator_trnid[brake][0])
        ]
        assert shared
        assert brake < min(shared)

    def test_the_seeded_commands_are_all_distinct_and_none_is_zero(self, scene: Any) -> None:
        """A repeated or zero setpoint could not show a swap or a loss."""
        assert len(set(_UNNAMED_CTRL)) == len(_UNNAMED_CTRL)
        assert 0.0 not in _UNNAMED_CTRL


class TestTheKeyVocabulary:
    """The two key forms, and the residual case that is reported rather than guessed."""

    def test_a_named_actuator_is_keyed_by_its_name(self, scene: Any) -> None:
        sim, _, _ = scene
        model = sim._world._model
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm/pan_act")
        assert scene_ops._actuator_key(model, aid, mujoco) == ("actuator", "arm/pan_act")

    def test_an_unnamed_actuator_is_keyed_by_target_transmission_and_ordinal(self, scene: Any) -> None:
        sim, _, _ = scene
        model = sim._world._model
        keys = [scene_ops._actuator_key(model, aid, mujoco) for aid in _unnamed_actuator_ids(model)]
        assert keys == [
            # Ordinal 0 on this target is the named ``fur/door_brake``, which is
            # carried by its own key: the ordinal counts every actuator sharing
            # the target, named or not, so the two sides agree.
            ("target", int(mujoco.mjtTrn.mjTRN_JOINT), "fur/door_hinge", 1),
            ("target", int(mujoco.mjtTrn.mjTRN_JOINT), "fur/door_hinge", 2),
            ("target", int(mujoco.mjtTrn.mjTRN_JOINT), "fur/door_hinge", 3),
            ("target", int(mujoco.mjtTrn.mjTRN_JOINTINPARENT), "fur/door_hinge", 0),
            ("target", int(mujoco.mjtTrn.mjTRN_BODY), "fur/drawer", 0),
        ]

    def test_every_key_resolves_back_to_the_actuator_it_was_built_from(self, scene: Any) -> None:
        """A key that did not round-trip would restore one actuator's command onto another."""
        sim, _, _ = scene
        model = sim._world._model
        for aid in range(int(model.nu)):
            key = scene_ops._actuator_key(model, aid, mujoco)
            assert key is not None
            assert scene_ops._resolve_actuator_key(model, key, mujoco) == aid

    def test_the_target_handle_does_not_claim_an_actuator_that_carries_a_name(self, scene: Any) -> None:
        """A named actuator is carried by its own key, so both handles must not resolve to it."""
        sim, _, _ = scene
        model = sim._world._model
        named = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fur/drawer_slide_act")
        target_handle = (
            "target",
            int(model.actuator_trntype[named]),
            "fur/drawer_slide",
            0,
        )
        assert scene_ops._resolve_actuator_key(model, target_handle, mujoco) == -1

    def test_the_transmission_mapping_covers_every_transmission_mujoco_defines(self) -> None:
        """Derived from ``mjtTrn`` so a transmission added upstream fails here.

        ``mjTRN_UNDEFINED`` is excluded: it is the sentinel for "no transmission
        resolved", not a kind of target.
        """
        defined = {
            int(getattr(mujoco.mjtTrn, name))
            for name in dir(mujoco.mjtTrn)
            if name.startswith("mjTRN_") and name != "mjTRN_UNDEFINED"
        }
        assert defined
        unmapped = {trn for trn in defined if scene_ops._actuator_target_kind(trn, mujoco) is None}
        assert unmapped == set()

    def test_an_unknown_transmission_is_reported_rather_than_guessed(self, scene: Any) -> None:
        """The residual case: nothing identifies the target, so no key is built."""
        sim, _, _ = scene
        model = sim._world._model
        assert scene_ops._actuator_target_kind(int(mujoco.mjtTrn.mjTRN_UNDEFINED), mujoco) is None
        key = ("target", int(mujoco.mjtTrn.mjTRN_UNDEFINED), "fur/door_hinge", 0)
        assert scene_ops._resolve_actuator_key(model, key, mujoco) == -1


class TestWhatMustNotChange:
    """Carrying an unnamed actuator must not widen what the restore claims."""

    def test_the_named_actuator_still_survives_the_eject(self, scene: Any) -> None:
        """The pre-existing guarantee, unchanged."""
        sim, _, _ = scene
        sim.remove_robot("arm")
        model = sim._world._model
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fur/drawer_slide_act")
        assert float(sim._world._data.ctrl[aid]) == _NAMED_CTRL

    def test_the_ejected_robot_s_actuator_is_gone_rather_than_restored(self, scene: Any) -> None:
        """Its absence is the point of the rebuild."""
        sim, _, _ = scene
        sim.remove_robot("arm")
        assert "arm/pan_act" not in _actuator_names(sim._world._model)
        assert int(sim._world._model.nu) == len(_UNNAMED_CTRL) + 2

    def test_a_key_whose_target_is_gone_resolves_to_nothing(self, scene: Any) -> None:
        """The ejected robot's elements no longer resolve, which is how they are skipped."""
        sim, _, _ = scene
        sim.remove_robot("arm")
        model = sim._world._model
        key = ("target", int(mujoco.mjtTrn.mjTRN_JOINT), "arm/pan", 0)
        assert scene_ops._resolve_actuator_key(model, key, mujoco) == -1

    def test_a_target_that_survives_without_that_many_actuators_resolves_to_nothing(self, scene: Any) -> None:
        """An ordinal past the end is skipped rather than wrapped onto a sibling."""
        sim, _, _ = scene
        model = sim._world._model
        key = ("target", int(mujoco.mjtTrn.mjTRN_JOINT), "fur/door_hinge", 99)
        assert scene_ops._resolve_actuator_key(model, key, mujoco) == -1

    def test_the_joint_state_the_rebuild_already_carried_still_arrives(self, scene: Any) -> None:
        """The actuator key change must not disturb the joint half of the snapshot."""
        sim, _, _ = scene
        world = sim._world
        adr = int(
            world._model.jnt_qposadr[mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_JOINT, "fur/door_hinge")]
        )
        world._data.qpos[adr] = 0.70
        mujoco.mj_forward(world._model, world._data)
        sim.remove_robot("arm")
        model = world._model
        after = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fur/door_hinge")])
        assert float(world._data.qpos[after]) == 0.70
