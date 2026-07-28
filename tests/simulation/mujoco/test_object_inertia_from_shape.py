"""A dynamic object's inertia is integrated from its shape, not a constant.

``add_object`` used to declare an explicit body-level inertial block with a
hard-coded diagonal - ``body.inertia = [0.001, 0.001, 0.001]`` - alongside the
caller's mass. ``body_mass`` was therefore correct and the object fell exactly
as it should, which is precisely what kept the defect silent: only the
*rotational* dynamics were wrong, and they were wrong by orders of magnitude in
a direction that flipped with size.

For a 100 g cube the true diagonal is ``m/6 * a**2``:

* 1 cm cube: ``1.67e-6`` - the constant is **600x too large**, so the cube
  resisted rotation like a flywheel and would not tumble on impact or spin out
  of a gripper.
* 5 cm cube: ``4.17e-5`` - **24x too large**.
* 30 cm 1 kg crate: ``0.015`` - the constant is **15x too small**, so the crate
  spun up as if it were hollow.

A single constant also cannot represent an anisotropic body at all: every real
cylinder, capsule and non-cubic box has ``Izz != Ixx``, and forcing them equal
removes the preferred spin axis that decides how an object topples.

Declaring the mass on the GEOM instead lets MuJoCo's compiler integrate the
tensor over the shape the caller actually asked for. The tests below check the
compiled tensor against the closed-form rigid-body formulas (an oracle
independent of the code under test) and then confirm the physical consequence:
a torque impulse spins the body up at the rate its true inertia implies.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# The constant diagonal that used to be written for every dynamic object.
LEGACY_CONSTANT_INERTIA = 0.001


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_object_inertia", mesh=False)
    s.create_world()
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _compiled(sim: Simulation, name: str) -> tuple[float, np.ndarray]:
    """Return the compiled ``(body_mass, body_inertia)`` for a named body."""
    assert sim._world is not None and sim._world._model is not None
    model = sim._world._model
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert body_id >= 0, f"body {name!r} not in the compiled model"
    return float(model.body_mass[body_id]), np.array(model.body_inertia[body_id])


# Closed-form principal inertias for a uniform solid, in terms of the FULL
# extents ``add_object`` documents (not MuJoCo's half-extents). Each entry is
# (shape, size, mass, expected diagonal).
ANALYTIC_CASES = [
    pytest.param(
        "box",
        [0.01, 0.01, 0.01],
        0.1,
        [0.1 / 12.0 * (0.01**2 + 0.01**2)] * 3,
        id="1cm-cube-legacy-was-600x-too-large",
    ),
    pytest.param(
        "box",
        [0.05, 0.05, 0.05],
        0.1,
        [0.1 / 12.0 * (0.05**2 + 0.05**2)] * 3,
        id="5cm-cube-legacy-was-24x-too-large",
    ),
    pytest.param(
        "box",
        [0.30, 0.30, 0.30],
        1.0,
        [1.0 / 12.0 * (0.30**2 + 0.30**2)] * 3,
        id="30cm-crate-legacy-was-15x-too-small",
    ),
    pytest.param(
        "box",
        [0.40, 0.10, 0.02],
        2.0,
        [
            2.0 / 12.0 * (0.10**2 + 0.02**2),
            2.0 / 12.0 * (0.40**2 + 0.02**2),
            2.0 / 12.0 * (0.40**2 + 0.10**2),
        ],
        id="plank-anisotropic-no-constant-can-express-it",
    ),
    pytest.param(
        "sphere",
        [0.04, 0.04, 0.04],
        0.2,
        [2.0 / 5.0 * 0.2 * 0.02**2] * 3,
        id="sphere-two-fifths-m-r-squared",
    ),
    pytest.param(
        "cylinder",
        [0.06, 0.06, 0.10],
        0.5,
        [
            0.5 / 12.0 * (3.0 * 0.03**2 + 0.10**2),
            0.5 / 12.0 * (3.0 * 0.03**2 + 0.10**2),
            0.5 * 0.5 * 0.03**2,
        ],
        id="cylinder-distinct-spin-axis",
    ),
]


class TestInertiaIntegratedFromTheShape:
    @pytest.mark.parametrize(("shape", "size", "mass", "expected"), ANALYTIC_CASES)
    def test_compiled_inertia_matches_the_closed_form_solid(self, sim, shape, size, mass, expected):
        """The tensor is the shape's real one, and the mass is still honored."""
        result = sim.add_object("body", shape=shape, size=size, position=[0.0, 0.0, 0.5], mass=mass)
        assert result["status"] == "success", result

        compiled_mass, inertia = _compiled(sim, "body")
        # Translation was never broken; keep it pinned so the fix cannot trade
        # a correct tensor for a wrong mass.
        assert compiled_mass == pytest.approx(mass)
        assert inertia == pytest.approx(np.array(expected), rel=1e-6)

    def test_the_legacy_constant_erred_in_both_directions(self, sim):
        """One number cannot serve a 1 cm cube and a 30 cm crate.

        This is the whole reason a constant is unusable rather than merely
        imprecise: it is too large for a small object and too small for a large
        one, so no choice of constant is safe.
        """
        assert (
            sim.add_object("pebble", shape="box", size=[0.01] * 3, position=[0, 0, 0.5], mass=0.1)["status"]
            == "success"
        )
        assert (
            sim.add_object("crate", shape="box", size=[0.30] * 3, position=[1, 0, 0.5], mass=1.0)["status"] == "success"
        )

        _, pebble = _compiled(sim, "pebble")
        _, crate = _compiled(sim, "crate")

        assert LEGACY_CONSTANT_INERTIA / pebble[0] == pytest.approx(600.0, rel=1e-3)
        assert LEGACY_CONSTANT_INERTIA / crate[0] == pytest.approx(1.0 / 15.0, rel=1e-3)

    def test_a_static_object_still_takes_its_mass_from_the_geom_density(self, sim):
        """Static bodies are unchanged: no freejoint, no mass declaration.

        ``mass`` is documented as ignored for a static body, and its compiled
        mass comes from the geom's default density (1000 kg/m^3 x a 5 cm cube =
        0.125 kg). Pinned so the change stays confined to dynamic objects.
        """
        result = sim.add_object("tile", shape="box", size=[0.05] * 3, mass=9.0, is_static=True)
        assert result["status"] == "success", result
        compiled_mass, _ = _compiled(sim, "tile")
        assert compiled_mass == pytest.approx(0.125)

    def test_the_inertia_survives_a_later_scene_recompile(self, sim):
        """Every scene mutation recompiles the spec; the tensor is rebuilt from it.

        The mass lives on the geom in the spec rather than in a constant
        inertial block, so an unrelated later edit must reproduce the same
        integrated tensor instead of reverting to a default.
        """
        assert (
            sim.add_object("crate", shape="box", size=[0.05] * 3, position=[0, 0, 0.3], mass=0.1)["status"] == "success"
        )
        before = _compiled(sim, "crate")

        assert sim.add_camera("side", position=[1.0, 0.0, 0.5], target=[0.0, 0.0, 0.1])["status"] == "success"
        assert sim.add_object("marker", shape="sphere", size=[0.02] * 3, position=[0.4, 0, 0.1])["status"] == "success"

        after = _compiled(sim, "crate")
        assert after[0] == pytest.approx(before[0])
        assert after[1] == pytest.approx(before[1])


class TestInertiaChangesTheDynamics:
    def test_a_torque_spins_the_body_up_at_the_rate_its_true_inertia_implies(self):
        """The consequence, not just the number in the model.

        With gravity off, a latched torque about z gives ``omega = tau * t / Izz``
        for a free body. The 5 cm 100 g cube's true ``Izz`` is 24x smaller than
        the constant that used to be compiled, so the same impulse produced a
        24x slower spin - a cube that would not be flicked over by a nudge and
        would not rotate in a gripper's grasp.
        """
        sim = Simulation(tool_name="devx_object_inertia_spin", mesh=False)
        try:
            sim.create_world(gravity=[0.0, 0.0, 0.0])
            assert (
                sim.add_object("cube", shape="box", size=[0.05] * 3, position=[0, 0, 0.5], mass=0.1)["status"]
                == "success"
            )

            izz = _compiled(sim, "cube")[1][2]
            torque = 1e-4
            assert sim.apply_force("cube", torque=[0.0, 0.0, torque])["status"] == "success"

            model = sim._world._model
            assert model is not None
            duration = 0.2
            sim.step(int(round(duration / float(model.opt.timestep))))

            state = sim.get_body_state("cube")
            omega_z = [c["json"] for c in state["content"] if "json" in c][0]["angular_velocity"][2]

            assert omega_z == pytest.approx(torque * duration / izz, rel=0.02)
            # And the same impulse under the old constant tensor would have been
            # a small fraction of that - the observable size of the defect.
            legacy_omega = torque * duration / LEGACY_CONSTANT_INERTIA
            assert omega_z > 20.0 * legacy_omega
        finally:
            sim.cleanup(policy_stop_timeout=0.5)
