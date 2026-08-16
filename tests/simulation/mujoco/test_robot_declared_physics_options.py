"""A robot model's declared ``<option>`` must survive being added to a world.

``<option>`` is model-global, so it does not come across ``spec.attach()``. A
robot MJCF that declares the solver settings its own contacts and actuators were
tuned for used to lose every one of them when composed into a generated scene,
and the effect is physical: a Franka Panda declares
``integrator="implicitfast"``, and under the Euler integrator the scene fell
back to, its position servos diverge enough that a scripted top-down grasp
pushes the cube 32 mm away on approach and squeezes straight through it on the
lift instead of carrying it.

These tests pin the contract on the compiled model - the only place the setting
can be observed to take effect - rather than on the spec, and pin the precedence
rules that decide which value a model-global field ends up holding.

``integrator`` assertions deliberately use ``RK4``: ``actuate_robot`` flips the
integrator to ``implicitfast`` scene-wide when it adds position servos to a
model that ships none, so ``implicitfast`` alone cannot distinguish adoption
from that unrelated path. Nothing else in the codebase selects ``RK4``.
"""

from __future__ import annotations

import logging

import pytest

mujoco = pytest.importorskip("mujoco")

import strands_robots as sr  # noqa: E402
from strands_robots.simulation.models import SimRobot  # noqa: E402
from strands_robots.simulation.mujoco.spec_builder import SpecBuilder  # noqa: E402

ARM_BODY = """
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
      <body name="link" pos="0 0 0.2">
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="20"/>
    <position name="elbow_act" joint="elbow" kp="20"/>
  </actuator>
"""


def _arm_mjcf(tmp_path, name: str, option: str) -> str:
    """Write a minimal actuated 2-joint arm MJCF carrying ``option``."""
    path = tmp_path / f"{name}.xml"
    path.write_text(f'<mujoco model="{name}">\n  {option}\n{ARM_BODY}</mujoco>\n')
    return str(path)


@pytest.fixture
def sim():
    s = sr.Simulation(backend="mujoco", tool_name="declared_options", mesh=False)
    yield s
    s.destroy()


class TestDeclaredOptionsAreCompiled:
    """The values a robot model declares reach the compiled model."""

    def test_declared_integrator_is_compiled(self, sim, tmp_path):
        """A model that asks for RK4 is integrated with RK4."""
        sim.create_world()
        sim.add_robot(name="arm", urdf_path=_arm_mjcf(tmp_path, "arm", '<option integrator="RK4"/>'))

        assert sim.mj_model.opt.integrator == int(mujoco.mjtIntegrator.mjINT_RK4)

    def test_declared_friction_cone_and_impratio_are_compiled(self, sim, tmp_path):
        """The elliptic-cone/impratio recipe a load-bearing gripper needs is kept.

        ``so100``, ``so101``, ``aloha``, ``shadow_hand`` and ``robotiq_2f85`` all
        declare exactly this pair.
        """
        sim.create_world()
        sim.add_robot(
            name="arm",
            urdf_path=_arm_mjcf(tmp_path, "arm", '<option cone="elliptic" impratio="10"/>'),
        )

        assert sim.mj_model.opt.cone == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
        assert sim.mj_model.opt.impratio == pytest.approx(10.0)

    def test_declared_solver_iteration_budget_is_compiled(self, sim, tmp_path):
        """Iteration/tolerance budgets are part of the same declaration."""
        sim.create_world()
        sim.add_robot(
            name="arm",
            urdf_path=_arm_mjcf(tmp_path, "arm", '<option solver="PGS" iterations="50" noslip_iterations="3"/>'),
        )

        assert sim.mj_model.opt.solver == int(mujoco.mjtSolver.mjSOL_PGS)
        assert sim.mj_model.opt.iterations == 50
        assert sim.mj_model.opt.noslip_iterations == 3

    def test_model_declaring_nothing_keeps_mujoco_defaults(self, sim, tmp_path):
        """Adoption only moves fields the model actually declares."""
        sim.create_world()
        sim.add_robot(name="arm", urdf_path=_arm_mjcf(tmp_path, "arm", ""))

        defaults = mujoco.MjSpec().option
        assert sim.mj_model.opt.cone == defaults.cone
        assert sim.mj_model.opt.impratio == pytest.approx(defaults.impratio)
        assert sim.mj_model.opt.iterations == defaults.iterations

    def test_adopted_option_survives_a_later_scene_recompile(self, sim, tmp_path):
        """Adding an object rebuilds the scene; the declaration must not be lost."""
        sim.create_world()
        sim.add_robot(name="arm", urdf_path=_arm_mjcf(tmp_path, "arm", '<option integrator="RK4"/>'))
        sim.add_object(name="cube", shape="box", position=[0.4, 0.0, 0.05], size=[0.05, 0.05, 0.05])

        assert sim.mj_model.opt.integrator == int(mujoco.mjtIntegrator.mjINT_RK4)


class TestWorldOwnedFieldsWin:
    """``create_world`` owns the fields it exposes; a model cannot move them."""

    def test_model_declared_timestep_does_not_move_the_world_dt(self, sim, tmp_path):
        """The world's dt is the caller's, and the rollout math is built on it."""
        sim.create_world(timestep=0.002)
        sim.add_robot(name="arm", urdf_path=_arm_mjcf(tmp_path, "arm", '<option timestep="0.01"/>'))

        assert sim.mj_model.opt.timestep == pytest.approx(0.002)

    def test_model_declared_gravity_does_not_move_the_world_gravity(self, sim, tmp_path):
        """Same for gravity: ``create_world(gravity=...)`` is the source of truth."""
        sim.create_world(gravity=[0.0, 0.0, -9.81])
        sim.add_robot(name="arm", urdf_path=_arm_mjcf(tmp_path, "arm", '<option gravity="0 0 -1"/>'))

        assert sim.mj_model.opt.gravity[2] == pytest.approx(-9.81)

    def test_model_declared_wind_is_not_adopted(self, sim, tmp_path):
        """Vector environment fields describe the world, not the robot."""
        sim.create_world()
        sim.add_robot(name="arm", urdf_path=_arm_mjcf(tmp_path, "arm", '<option wind="5 0 0"/>'))

        assert sim.mj_model.opt.wind[0] == pytest.approx(0.0)


class TestConflictBetweenTwoRobots:
    """A model-global field holds one value, so a disagreement is arbitrated."""

    def test_first_declaration_wins_and_the_discarded_one_is_reported(self, sim, tmp_path, caplog):
        """The scene value is kept and the rejected request is named in full."""
        sim.create_world()
        sim.add_robot(name="first", urdf_path=_arm_mjcf(tmp_path, "first", '<option integrator="RK4"/>'))

        with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.spec_builder"):
            sim.add_robot(
                name="second",
                position=[1.0, 0.0, 0.0],
                urdf_path=_arm_mjcf(tmp_path, "second", '<option integrator="implicitfast"/>'),
            )

        assert sim.mj_model.opt.integrator == int(mujoco.mjtIntegrator.mjINT_RK4)
        message = caplog.text
        assert "second" in message
        assert "integrator" in message

    def test_agreeing_declarations_are_not_reported_as_a_conflict(self, sim, tmp_path, caplog):
        """Two robots asking for the same value is not a disagreement."""
        sim.create_world()
        sim.add_robot(name="first", urdf_path=_arm_mjcf(tmp_path, "first", '<option cone="elliptic"/>'))

        with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.spec_builder"):
            sim.add_robot(
                name="second",
                position=[1.0, 0.0, 0.0],
                urdf_path=_arm_mjcf(tmp_path, "second", '<option cone="elliptic"/>'),
            )

        assert sim.mj_model.opt.cone == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
        assert "cone" not in caplog.text

    def test_two_robots_declaring_different_fields_both_apply(self, sim, tmp_path):
        """Arbitration is per field, so disjoint declarations do not compete."""
        sim.create_world()
        sim.add_robot(name="first", urdf_path=_arm_mjcf(tmp_path, "first", '<option integrator="RK4"/>'))
        sim.add_robot(
            name="second",
            position=[1.0, 0.0, 0.0],
            urdf_path=_arm_mjcf(tmp_path, "second", '<option impratio="10"/>'),
        )

        assert sim.mj_model.opt.integrator == int(mujoco.mjtIntegrator.mjINT_RK4)
        assert sim.mj_model.opt.impratio == pytest.approx(10.0)


class TestReadingAndApplyingAreSeparate:
    """The two halves of the adoption report exactly what they handled."""

    def test_the_read_names_the_declared_fields_and_skips_world_owned_ones(self, tmp_path):
        """Reading a model yields only the fields adoption is allowed to move."""
        scene = mujoco.MjSpec()
        robot = mujoco.MjSpec.from_file(_arm_mjcf(tmp_path, "arm", '<option integrator="RK4" timestep="0.01"/>'))

        declared = SpecBuilder.declared_options(robot)

        assert declared == {"integrator": int(mujoco.mjtIntegrator.mjINT_RK4)}
        assert scene.option.timestep == pytest.approx(mujoco.MjSpec().option.timestep)

    def test_the_read_names_nothing_for_a_model_that_declares_nothing(self, tmp_path):
        """A model restating MuJoCo's defaults declares nothing to adopt."""
        robot = mujoco.MjSpec.from_file(_arm_mjcf(tmp_path, "arm", ""))

        assert SpecBuilder.declared_options(robot) == {}

    def test_applying_reports_what_it_wrote_onto_the_scene(self, tmp_path):
        """The write reports the subset the scene did not already own."""
        scene = mujoco.MjSpec()
        scene.option.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
        declared = {
            "integrator": int(mujoco.mjtIntegrator.mjINT_RK4),
            "cone": int(mujoco.mjtCone.mjCONE_PYRAMIDAL),
        }

        adopted = SpecBuilder.adopt_declared_options(scene, declared, "arm")

        assert adopted == {"integrator": int(mujoco.mjtIntegrator.mjINT_RK4)}
        assert scene.option.cone == int(mujoco.mjtCone.mjCONE_ELLIPTIC)


class TestAFailedAttachLeavesTheSceneAlone:
    """A robot that never entered the world must not rewrite its physics.

    ``attach_robot`` mutates the *live* spec, which outlives a failed
    ``add_robot`` - ``inject_robot_into_scene`` catches the exception, logs it
    and reports the add as failed, and ``Simulation.add_robot`` then rolls back
    only its ``world.robots`` registry entry. Anything written onto the spec on
    the way to that error has no undo path, so it is baked into ``mj_model`` by
    the next successful scene mutation with no signal to the caller. Adoption is
    therefore read before the attach and written only after it has succeeded.
    """

    def test_scene_options_survive_an_attach_that_raises(self, tmp_path):
        """The declaration is dropped with the robot, not left on the scene."""
        scene = mujoco.MjSpec()
        robot = SimRobot(
            name="arm",
            urdf_path=_arm_mjcf(tmp_path, "arm", '<option integrator="RK4" impratio="10"/>'),
            # A 3-element orientation is refused by ``worldbody.add_frame``,
            # which runs between the read and the attach.
            orientation=[1.0, 0.0, 0.0],
        )

        with pytest.raises(ValueError):
            SpecBuilder.attach_robot(scene, robot, robot.urdf_path)

        defaults = mujoco.MjSpec().option
        assert scene.option.integrator == defaults.integrator
        assert scene.option.impratio == pytest.approx(defaults.impratio)

    def test_a_refused_add_robot_does_not_move_the_worlds_solver_settings(self, sim, tmp_path, monkeypatch):
        """The compiled model keeps its settings when the add is reported failed."""
        sim.create_world()
        defaults = mujoco.MjSpec().option

        def refuse(self, *args, **kwargs):
            raise ValueError("attach refused")

        monkeypatch.setattr(mujoco.MjSpec, "attach", refuse)
        result = sim.add_robot(name="arm", urdf_path=_arm_mjcf(tmp_path, "arm", '<option integrator="RK4"/>'))

        assert result["status"] == "error"
        assert "arm" not in sim._world.robots

        monkeypatch.undo()
        # The next scene mutation recompiles: a leaked adoption would surface
        # here, attributed to a robot the world never accepted.
        sim.add_object(name="cube", shape="box", position=[0.4, 0.0, 0.05], size=[0.05, 0.05, 0.05])

        assert sim.mj_model.opt.integrator == defaults.integrator

    def test_a_robot_added_after_a_refused_one_still_gets_its_declaration(self, sim, tmp_path, monkeypatch):
        """Dropping the leak must not amount to dropping adoption itself."""
        sim.create_world()

        def refuse(self, *args, **kwargs):
            raise ValueError("attach refused")

        monkeypatch.setattr(mujoco.MjSpec, "attach", refuse)
        sim.add_robot(name="refused", urdf_path=_arm_mjcf(tmp_path, "refused", '<option cone="elliptic"/>'))
        monkeypatch.undo()

        sim.add_robot(name="arm", urdf_path=_arm_mjcf(tmp_path, "arm", '<option integrator="RK4"/>'))

        assert sim.mj_model.opt.integrator == int(mujoco.mjtIntegrator.mjINT_RK4)
        # The refused robot's declaration is gone, so it does not out-rank a
        # later robot asking for a different value on the same field.
        assert sim.mj_model.opt.cone == mujoco.MjSpec().option.cone
