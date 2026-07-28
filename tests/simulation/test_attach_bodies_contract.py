"""Regression tests for the ``attach_bodies`` / ``detach_bodies`` grasp-assist contract.

Four defects, each of which reported success (or raised past the documented
tool-result contract) while leaving the world wrong:

1. **Equality-name collision bricked the world permanently.** The weld constraint
   was named ``attach_weld_{parent}__{child}``, but ``__`` is legal inside a body
   name - so ``("a__b", "c")`` and ``("a", "b__c")`` both produced
   ``attach_weld_a__b__c``. MuJoCo rejected the duplicate with a ``ValueError``
   that escaped the "returns {status, content}" contract, AND left a
   half-initialized equality on the live spec (``eq.type`` never assigned, so it
   stayed ``mjEQ_CONNECT``). Every later scene operation then failed forever:
   ``add_object``, ``export_xml``, and even ``detach_bodies`` itself.

2. **A failed ``detach_bodies`` kept the registry record**, so the object became
   permanently unattachable *and* unremovable: retrying detach hit the same
   failure, ``remove_object`` refused ("referenced by an active attachment"), and
   re-attaching refused ("already attached").

3. **Kinematic attachment left derived state stale on the policy path.** The
   teleport writes ``qpos``; ``step()`` re-forwarded afterwards but
   ``_apply_sim_action`` did not, so ``xpos`` / ``geom_xpos`` lagged the real pose
   by ~6 mm at 3 m/s - inherited by renders, contact reads and recorded datasets
   on exactly the path policies use.

4. **Welding a STATIC child pinned the parent to the world.** A body with no DOF
   cannot be carried, so the equality anchored the parent instead: a free-falling
   carrier stopped mid-air (1.0 cm of fall instead of 1.76 m) under a normal
   grasp message.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="attach_bodies_contract", mesh=False)
    s.create_world()
    yield s
    s.destroy()


def _add(sim, name, pos, *, static=False, size=0.03, mass=0.2):
    assert (
        sim.add_object(name=name, shape="box", size=[size] * 3, position=pos, mass=mass, is_static=static)["status"]
        == "success"
    )


def test_underscore_body_names_do_not_collide(sim) -> None:
    """``a__b``+``c`` and ``a``+``b__c`` must produce distinct constraint names."""
    _add(sim, "a__b", [0.0, 0, 0.3], mass=2.0)
    _add(sim, "c", [0.06, 0, 0.3])
    _add(sim, "a", [0.5, 0, 0.3], mass=2.0)
    _add(sim, "b__c", [0.56, 0, 0.3])

    assert sim.attach_bodies(parent="a__b", child="c", mode="weld")["status"] == "success"
    # Pre-fix this raised ValueError and bricked the spec.
    assert sim.attach_bodies(parent="a", child="b__c", mode="weld")["status"] == "success"

    names = [e.name for e in sim._world._backend_state["spec"].equalities]
    assert len(names) == len(set(names)), f"duplicate equality names: {names}"


def test_world_stays_mutable_after_two_underscore_attaches(sim) -> None:
    """The follow-on symptom: every later scene op used to fail forever."""
    _add(sim, "a__b", [0.0, 0, 0.3], mass=2.0)
    _add(sim, "c", [0.06, 0, 0.3])
    _add(sim, "a", [0.5, 0, 0.3], mass=2.0)
    _add(sim, "b__c", [0.56, 0, 0.3])
    sim.attach_bodies(parent="a__b", child="c", mode="weld")
    sim.attach_bodies(parent="a", child="b__c", mode="weld")

    assert sim.add_object(name="later", shape="box", size=[0.02] * 3, position=[1, 1, 0.05])["status"] == "success"
    assert sim.export_xml()["status"] == "success"
    assert sim.detach_bodies(parent="a__b", child="c")["status"] == "success"


def test_stale_constraint_detach_clears_the_registry(sim) -> None:
    """A record whose equality is already gone must not block recovery."""
    _add(sim, "carrier", [0.0, 0, 0.3], size=0.1, mass=2.0)
    _add(sim, "cube", [0.08, 0, 0.3])
    assert sim.attach_bodies(parent="carrier", child="cube", mode="weld")["status"] == "success"

    # Simulate an external spec edit that removed the equality.
    spec = sim._world._backend_state["spec"]
    for eq in list(spec.equalities):
        if eq.name.startswith("attach_weld"):
            spec.delete(eq)

    result = sim.detach_bodies(parent="carrier", child="cube")
    assert result["status"] == "success"
    assert "already absent" in result["content"][0]["text"]
    assert sim._world._backend_state.get("attachments", {}) == {}
    # Recovery routes that were all blocked pre-fix.
    assert sim.attach_bodies(parent="carrier", child="cube", mode="weld")["status"] == "success"
    assert sim.detach_bodies(parent="carrier", child="cube")["status"] == "success"


def test_weld_rejects_a_static_child(sim) -> None:
    """A DOF-less child would anchor the parent to the world, not be carried."""
    _add(sim, "carrier", [0.0, 0, 2.0], size=0.1, mass=2.0)
    _add(sim, "pillar", [0.08, 0, 2.0], static=True)

    result = sim.attach_bodies(parent="carrier", child="pillar", mode="weld")
    assert result["status"] == "error"
    assert "static" in result["content"][0]["text"]

    # And the parent must still be free to fall.
    model, data = sim.mj_model, sim.mj_data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "carrier")
    z0 = float(data.xpos[body][2])
    for _ in range(300):
        mujoco.mj_step(model, data)
    assert z0 - float(data.xpos[body][2]) > 1.0, "carrier was pinned in mid-air"


def test_dynamic_child_weld_still_works(sim) -> None:
    """The fix must not reject a legitimate grasp."""
    _add(sim, "carrier", [0.0, 0, 0.3], size=0.1, mass=2.0)
    _add(sim, "cube", [0.08, 0, 0.3])
    assert sim.attach_bodies(parent="carrier", child="cube", mode="weld")["status"] == "success"
    assert sim.mj_model.neq == 1


_KINEMATIC_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 .1"/>
    <body name="carrier" pos="0 0 0.5">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <inertial pos="0 0 0" mass="2" diaginertia=".1 .1 .1"/>
      <geom type="box" size=".05 .05 .05"/>
    </body>
    <body name="cube" pos="0.3 0 0.5">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>
      <geom type="box" size=".02 .02 .02"/>
    </body>
  </worldbody>
  <actuator><velocity joint="slide" kv="200" ctrlrange="-5 5"/></actuator>
</mujoco>
"""


def test_kinematic_carry_refreshes_pose_on_the_policy_path(tmp_path) -> None:
    """``xpos`` must match ``qpos`` after the policy path, as it does after step()."""
    from strands_robots.simulation.models import SimRobot

    path = tmp_path / "kin.xml"
    path.write_text(_KINEMATIC_XML)
    sim = Simulation(tool_name="attach_kinematic_pose", mesh=False)
    try:
        sim.load_scene(scene_path=str(path))
        sim._world.robots["bot"] = SimRobot(name="bot", urdf_path="", position=[0, 0, 0], joint_names=["slide"])  # type: ignore[union-attr]
        assert sim.attach_bodies(parent="carrier", child="cube", mode="kinematic")["status"] == "success"

        model, data = sim.mj_model, sim.mj_data
        cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        free_jid = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        qadr = int(model.jnt_qposadr[free_jid])

        data.ctrl[0] = 3.0
        sim._apply_sim_action("bot", {"slide": 3.0}, n_substeps=25)

        # Pre-fix: xpos lagged qpos by ~6 mm on this path.
        assert float(data.xpos[cube][0]) == pytest.approx(float(data.qpos[qadr]), abs=1e-9)
    finally:
        sim.destroy()


def test_weld_survives_removal_of_an_unrelated_robot() -> None:
    """A runtime weld must not be collateral damage of another robot's removal.

    ``eject_robot_from_scene`` rebuilds the scene via ``SpecBuilder.build``, which
    reconstructs only objects/cameras/lights/ground - so every weld added at
    runtime by ``attach_bodies`` was silently destroyed. Measured: removing robot
    'b' dropped the weld holding a cube to robot 'a's hand (neq 3 -> 1) and the
    still-registered "grasp" drifted 0.30 m away.
    """
    sim = Simulation(tool_name="weld_survives_robot_removal", mesh=False)
    try:
        sim.create_world()
        assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
        assert sim.add_robot(name="b", data_config="panda", position=[1.5, 0, 0])["status"] == "success"
        _add(sim, "cube", [0.4, 0, 0.3])
        assert sim.attach_bodies(parent="a/hand", child="cube", mode="weld")["status"] == "success"

        def _weld_names(model):
            return [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
                for i in range(model.neq)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) or "").startswith("attach_weld")
            ]

        assert _weld_names(sim.mj_model) == ["attach_weld_cube"]

        assert sim.remove_robot(name="b")["status"] == "success"

        # The weld on the SURVIVING robot must still be in the compiled model.
        assert _weld_names(sim.mj_model) == ["attach_weld_cube"]
        assert "cube" in sim._world._backend_state.get("attachments", {})  # type: ignore[union-attr]

        # And it must still actually hold: the cube tracks the hand.
        model, data = sim.mj_model, sim.mj_data
        hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a/hand")
        cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        offset = (data.xpos[cube] - data.xpos[hand]).copy()
        for _ in range(500):
            mujoco.mj_step(model, data)
        drift = float(((data.xpos[cube] - data.xpos[hand]) - offset).round(6).__abs__().max())
        assert drift < 0.15, f"weld stopped holding after the rebuild (drift {drift})"
    finally:
        sim.destroy()


def test_removal_is_refused_while_the_robot_is_attached() -> None:
    """The guard that makes the ejected-robot case unreachable via the tool."""
    sim = Simulation(tool_name="weld_blocks_robot_removal", mesh=False)
    try:
        sim.create_world()
        sim.add_robot(name="b", data_config="panda")
        _add(sim, "cube", [0.4, 0, 0.3])
        sim.attach_bodies(parent="b/hand", child="cube", mode="weld")

        result = sim.remove_robot(name="b")
        assert result["status"] == "error"
        assert "detach_bodies" in result["content"][0]["text"]

        # The documented recovery path works.
        assert sim.detach_bodies(parent="b/hand", child="cube")["status"] == "success"
        assert sim.remove_robot(name="b")["status"] == "success"
    finally:
        sim.destroy()


def test_kinematic_carry_survives_an_unrelated_robot_removal() -> None:
    """A kinematic attachment must keep carrying after a FULL scene rebuild.

    ``eject_robot_from_scene`` rebuilds the scene declaratively, so the weld path
    needed an explicit replay (see the weld test above). The kinematic path
    survives instead because its record lives in ``_backend_state`` and is
    re-resolved BY NAME every step - this pins that property rather than leaving
    it to luck.

    Note the invariant is a BODY-frame offset: the world-frame delta legitimately
    changes as the parent rotates, so a world-frame assertion here would report a
    0.285 m "drift" for a perfectly working carry.
    """
    sim = Simulation(tool_name="kinematic_survives_robot_removal", mesh=False)
    try:
        sim.create_world()
        assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
        assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
        _add(sim, "cube", [0.4, 0, 0.3])
        assert sim.attach_bodies(parent="a/hand", child="cube", mode="kinematic")["status"] == "success"
        recorded = list(sim._world._backend_state["kinematic_attachments"]["cube"]["relpos"])  # type: ignore[union-attr]

        assert sim.remove_robot(name="b")["status"] == "success"
        assert "cube" in sim._world._backend_state.get("kinematic_attachments", {})  # type: ignore[union-attr]

        # Drive the arm so the parent both translates and rotates.
        model, data = sim.mj_model, sim.mj_data
        data.ctrl[:7] = [0.3, 0.2, 0, -1.8, 0, 1.6, 0.7]
        for _ in range(600):
            sim.step(n_steps=1)

        model, data = sim.mj_model, sim.mj_data
        hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a/hand")
        cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        neg = np.zeros(4)
        mujoco.mju_negQuat(neg, data.xquat[hand])
        rel = np.zeros(3)
        mujoco.mju_rotVecQuat(rel, data.xpos[cube] - data.xpos[hand], neg)
        assert rel == pytest.approx(recorded, abs=1e-3), "kinematic carry lost after the rebuild"
    finally:
        sim.destroy()
