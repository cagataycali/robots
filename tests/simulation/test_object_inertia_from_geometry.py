"""Regression tests: an object's rotational inertia matches its shape and mass.

``SpecBuilder.add_object`` declared an explicit inertial block with a HARDCODED
``body.inertia = [0.001, 0.001, 0.001]`` and ``explicitinertial = True``, applied
to every object regardless of shape, size or mass. ``body_mass`` was correct, so
the model inspected fine and only the rotational dynamics were wrong - by orders
of magnitude, and worst for exactly the small objects a manipulation task uses
(analytic vs the constant, box, mass 0.1 kg):

    edge  2 cm ->  6.7e-06 vs 1e-03    150x too resistant to spin
    edge  5 cm ->  4.2e-05 vs 1e-03     24x
    edge 50 cm ->  4.2e-03 vs 1e-03   0.24x (too easy to spin)

and independent of mass, so a 10 g cube was off by 240x. It was also isotropic
for every shape: a cylinder got Ixx == Izz, which no real cylinder has.

Mass is now declared on the GEOM, so MuJoCo's compiler integrates the actual
shape. Note this also decides where a runtime mass change must be mirrored: with
no ``explicitinertial`` block, writing ``spec.body(name).mass`` is silently
dropped at compile time (see ``_sync_spec_body``).
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


def _box_inertia(mass: float, half: list[float]) -> list[float]:
    """Analytic diagonal inertia of a uniform box about its centre."""
    hx, hy, hz = half
    return [
        mass / 3.0 * (hy * hy + hz * hz),
        mass / 3.0 * (hx * hx + hz * hz),
        mass / 3.0 * (hx * hx + hy * hy),
    ]


def _sphere_inertia(mass: float, radius: float) -> list[float]:
    return [2.0 / 5.0 * mass * radius * radius] * 3


def _body(sim, name: str):
    model = sim.mj_model
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, name
    return float(model.body_mass[bid]), np.asarray(model.body_inertia[bid]).copy()


@pytest.fixture
def sim():
    s = Simulation(tool_name="object_inertia_from_geometry", mesh=False)
    s.create_world()
    yield s
    s.destroy()


# add_object's ``size`` is FULL EDGE LENGTHS; MuJoCo geom_size is half-extents.
@pytest.mark.parametrize("edge", [0.02, 0.05, 0.1, 0.3])
@pytest.mark.parametrize("mass", [0.01, 0.2, 5.0])
def test_box_inertia_matches_the_analytic_value(sim, edge, mass) -> None:
    assert sim.add_object(name="b", shape="box", size=[edge] * 3, position=[0, 0, 1], mass=mass)["status"] == "success"
    got_mass, got_inertia = _body(sim, "b")
    assert got_mass == pytest.approx(mass), "mass must be exactly what the caller asked for"
    assert list(got_inertia) == pytest.approx(_box_inertia(mass, [edge / 2] * 3), rel=1e-6)


@pytest.mark.parametrize("diameter", [0.04, 0.1])
def test_sphere_inertia_matches_the_analytic_value(sim, diameter) -> None:
    assert (
        sim.add_object(name="s", shape="sphere", size=[diameter], position=[0, 0, 1], mass=0.2)["status"] == "success"
    )
    got_mass, got_inertia = _body(sim, "s")
    assert got_mass == pytest.approx(0.2)
    assert list(got_inertia) == pytest.approx(_sphere_inertia(0.2, diameter / 2), rel=1e-6)


def test_inertia_scales_with_size(sim) -> None:
    """The core defect: the constant made inertia independent of size."""
    assert (
        sim.add_object(name="small", shape="box", size=[0.05] * 3, position=[0, 0, 1], mass=0.2)["status"] == "success"
    )
    assert sim.add_object(name="big", shape="box", size=[0.15] * 3, position=[1, 0, 1], mass=0.2)["status"] == "success"
    small = _body(sim, "small")[1]
    big = _body(sim, "big")[1]
    # I ~ h^2, so tripling the edge must give 9x the inertia.
    assert float(big[0] / small[0]) == pytest.approx(9.0, rel=1e-6)


def test_inertia_scales_with_mass(sim) -> None:
    """The constant also made inertia independent of mass."""
    assert (
        sim.add_object(name="light", shape="box", size=[0.05] * 3, position=[0, 0, 1], mass=0.1)["status"] == "success"
    )
    assert (
        sim.add_object(name="heavy", shape="box", size=[0.05] * 3, position=[1, 0, 1], mass=1.0)["status"] == "success"
    )
    assert float(_body(sim, "heavy")[1][0] / _body(sim, "light")[1][0]) == pytest.approx(10.0, rel=1e-6)


def test_cylinder_inertia_is_anisotropic(sim) -> None:
    """A cylinder's axial inertia differs from its transverse; the constant did not."""
    assert (
        sim.add_object(name="c", shape="cylinder", size=[0.05, 0.0, 0.2], position=[0, 0, 1], mass=0.2)["status"]
        == "success"
    )
    inertia = _body(sim, "c")[1]
    assert float(inertia[0]) == pytest.approx(float(inertia[1]), rel=1e-9), "transverse axes must match"
    assert float(inertia[2]) != pytest.approx(float(inertia[0]), rel=1e-3), "axial must differ from transverse"


def test_inertia_is_positive_for_every_shape(sim) -> None:
    """A zero/degenerate inertia tensor is unintegrable."""
    shapes = [
        ("box", [0.05, 0.05, 0.05]),
        ("sphere", [0.05]),
        ("cylinder", [0.05, 0.0, 0.1]),
        ("capsule", [0.05, 0.0, 0.1]),
    ]
    for i, (shape, size) in enumerate(shapes):
        name = f"o{i}"
        assert sim.add_object(name=name, shape=shape, size=size, position=[i, 0, 1], mass=0.2)["status"] == "success"
        mass, inertia = _body(sim, name)
        assert mass == pytest.approx(0.2), shape
        assert bool(np.all(inertia > 0)), f"{shape} got a non-positive inertia {inertia}"


def test_runtime_mass_change_rescales_inertia_and_survives_a_rebuild(sim) -> None:
    """Where mass lives in the spec decides whether a runtime change persists."""
    assert sim.add_object(name="b", shape="box", size=[0.1] * 3, position=[0, 0, 1], mass=0.2)["status"] == "success"
    _, before = _body(sim, "b")

    assert sim.set_body_properties(body_name="b", mass=5.0)["status"] == "success"
    mass, after = _body(sim, "b")
    assert mass == pytest.approx(5.0)
    assert float(after[0] / before[0]) == pytest.approx(25.0, rel=1e-6)

    # A rebuild recompiles from the spec; the mass mirror must land where the
    # compiler reads it.
    assert (
        sim.add_object(name="other", shape="box", size=[0.04] * 3, position=[1, 1, 1], mass=0.1)["status"] == "success"
    )
    mass_after, inertia_after = _body(sim, "b")
    assert mass_after == pytest.approx(5.0), "runtime mass reverted on rebuild"
    assert list(inertia_after) == pytest.approx(list(after), rel=1e-6)


def test_a_small_cube_actually_spins_up_under_a_torque(sim) -> None:
    """End-to-end: the 300x excess inertia changed how an object responds.

    A 2 cm / 50 g cube in free space under a 2 mNm torque for 0.1 s. With the
    hardcoded 1e-3 constant it reached ~0.2 rad/s; the geometry-derived
    3.33e-06 tensor gives 60 rad/s, which is what w = tau*t/I predicts.
    """
    assert (
        sim.add_object(name="b", shape="box", size=[0.02] * 3, position=[0, 0, 0.5], mass=0.05)["status"] == "success"
    )
    assert sim.set_gravity(gravity=[0.0, 0.0, 0.0])["status"] == "success"
    assert sim.apply_force(body_name="b", torque=[0.0, 0.0, 0.002])["status"] == "success"

    model = sim.mj_model
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")
    inertia_zz = float(model.body_inertia[bid][2])
    steps = 50
    assert sim.step(n_steps=steps)["status"] == "success"

    model, data = sim.mj_model, sim.mj_data
    joint = next(i for i in range(model.njnt) if int(model.jnt_bodyid[i]) == bid)
    dof = int(model.jnt_dofadr[joint])
    spin = abs(float(data.qvel[dof + 5]))

    # Rigid-body prediction: w = tau * t / Izz.
    expected = 0.002 * (steps * float(model.opt.timestep)) / inertia_zz
    assert spin == pytest.approx(expected, rel=1e-3), f"{spin} rad/s vs predicted {expected}"
    assert bool(np.all(np.isfinite(data.qvel)))
