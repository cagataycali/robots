# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""``add_object(mass=M)`` must give the body exactly ``M`` kg, and M's inertia.

``_add_object_to_builder`` did ``builder.add_body(xform=..., mass=obj.mass)`` then
``add_shape_*(body, ...)`` with no ``ShapeConfig``. Newton's ``add_shape_*``
defaults to ``ShapeConfig.density = 1000.0`` and ACCUMULATES the shape's
density-derived mass and inertia onto the parent body, so the final mass was
``requested + 1000 * volume`` - and the inertia tensor was that of the
accumulated mass. Measured pre-fix, with the extra matching ``1000 * volume``
exactly for every shape::

    box  [0.05]^3  requested 0.500 -> 1.5000  extra 1.0000  (1000*vol 1.0000)
    box  [0.03]^3  requested 0.100 -> 0.3160  extra 0.2160  (1000*vol 0.2160)
    sphere r=0.04  requested 0.200 -> 0.4681  extra 0.2681  (1000*vol 0.2681)
    cylinder       requested 0.300 -> 0.5827  extra 0.2827  (1000*vol 0.2827)
    capsule        requested 0.250 -> 0.3840  extra 0.1340  (1000*vol 0.1340)
    box inertia diag 0.00166667 vs analytic-for-0.5kg 0.00083333   (2x off)

``list_objects`` prints ``obj.mass`` from the ``SimObject``, so the tool reported
a mass the physics did not use. The MuJoCo backend, the parity reference, reports
0.5 for that same box.

The fix creates the body massless and expresses the requested mass as a density
over the shape's own volume, so Newton derives both the mass and a
geometry-consistent inertia tensor.

These tests use the real engine (not the fake-builder dispatch test) because the
defect lives in what the finalized ``Model`` ends up holding.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


def _engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    return NewtonSimEngine(solver="mujoco")


def _build(shape: str, size: list[float], mass: float):
    """Return ``(engine, body_mass, inertia_diagonal)`` for one spawned object."""
    sim = _engine()
    sim.create_world()
    result = sim.add_object("o", shape=shape, position=[0.3, 0.0, 0.2], size=size, mass=mass)
    assert result["status"] == "success", result
    body_mass = float(sim._model.body_mass.numpy()[-1])
    inertia = sim._model.body_inertia.numpy()[-1]
    diagonal = [float(inertia[i][i]) for i in range(3)]
    return sim, body_mass, diagonal


class TestRequestedMassIsTheActualMass:
    @pytest.mark.parametrize(
        "shape,size,mass",
        [
            ("box", [0.05, 0.05, 0.05], 0.5),
            ("box", [0.03, 0.03, 0.03], 0.1),
            ("sphere", [0.04], 0.2),
            ("cylinder", [0.03, 0.05], 0.3),
            ("capsule", [0.02, 0.04], 0.25),
        ],
    )
    def test_the_body_mass_equals_the_requested_mass(self, shape, size, mass):
        """Regression: the body carried requested + 1000 * volume."""
        sim, body_mass, _ = _build(shape, size, mass)
        try:
            assert body_mass == pytest.approx(mass, rel=1e-3), (
                f"{shape} requested {mass} but the model holds {body_mass} (extra {body_mass - mass:.4f})"
            )
        finally:
            sim.destroy()

    def test_a_heavy_and_a_light_object_of_one_size_differ(self):
        """The density path must still scale with mass, not clamp to one value."""
        light_sim, light, _ = _build("box", [0.04, 0.04, 0.04], 0.05)
        heavy_sim, heavy, _ = _build("box", [0.04, 0.04, 0.04], 5.0)
        try:
            assert light == pytest.approx(0.05, rel=1e-3)
            assert heavy == pytest.approx(5.0, rel=1e-3)
        finally:
            light_sim.destroy()
            heavy_sim.destroy()

    def test_the_reported_mass_matches_the_physical_mass(self):
        """list_objects reads obj.mass; that number must be the real one."""
        sim, body_mass, _ = _build("box", [0.05, 0.05, 0.05], 0.5)
        try:
            assert sim._world.objects["o"].mass == pytest.approx(body_mass, rel=1e-3)
        finally:
            sim.destroy()


class TestInertiaMatchesTheRequestedMass:
    def test_box_inertia_is_analytic(self):
        """I_xx = m/3 * (hy^2 + hz^2). Pre-fix this was 2x too large."""
        sim, mass, diagonal = _build("box", [0.05, 0.05, 0.05], 0.5)
        try:
            expected = 0.5 / 3.0 * (0.05**2 + 0.05**2)
            assert diagonal[0] == pytest.approx(expected, rel=1e-3), f"{diagonal[0]} vs analytic {expected}"
        finally:
            sim.destroy()

    def test_sphere_inertia_is_analytic(self):
        """I = 2/5 m r^2, isotropic."""
        sim, mass, diagonal = _build("sphere", [0.04], 0.2)
        try:
            expected = 0.4 * 0.2 * 0.04**2
            for axis, value in enumerate(diagonal):
                assert value == pytest.approx(expected, rel=1e-3), f"axis {axis}: {value} vs {expected}"
        finally:
            sim.destroy()

    def test_cylinder_axial_inertia_is_analytic(self):
        """About its own axis (z): I_zz = 1/2 m r^2."""
        sim, mass, diagonal = _build("cylinder", [0.03, 0.05], 0.3)
        try:
            expected = 0.5 * 0.3 * 0.03**2
            assert diagonal[2] == pytest.approx(expected, rel=1e-3), f"{diagonal[2]} vs analytic {expected}"
        finally:
            sim.destroy()

    def test_inertia_scales_linearly_with_mass(self):
        """A 10x heavier body of one shape must have 10x the inertia."""
        light_sim, _, light = _build("box", [0.04, 0.04, 0.04], 0.1)
        heavy_sim, _, heavy = _build("box", [0.04, 0.04, 0.04], 1.0)
        try:
            assert heavy[0] == pytest.approx(10.0 * light[0], rel=1e-3)
        finally:
            light_sim.destroy()
            heavy_sim.destroy()


class TestStaticAndMasslessObjectsAreUnaffected:
    def test_a_static_object_creates_no_body(self):
        sim = _engine()
        try:
            sim.create_world()
            assert (
                sim.add_object("s", shape="box", position=[0.0, 0.0, 0.05], size=[0.1, 0.1, 0.01], is_static=True)[
                    "status"
                ]
                == "success"
            )

            assert int(sim._model.body_count) == 0
        finally:
            sim.destroy()

    def test_a_zero_mass_object_creates_no_body(self):
        sim = _engine()
        try:
            sim.create_world()
            assert (
                sim.add_object("z", shape="box", position=[0.0, 0.0, 0.05], size=[0.02, 0.02, 0.02], mass=0.0)["status"]
                == "success"
            )

            assert int(sim._model.body_count) == 0
        finally:
            sim.destroy()


class TestDensityHelperFallsBackLoudly:
    """A volume that cannot be computed must not divide by zero silently."""

    def _obj(self):
        from strands_robots.simulation.models import SimObject

        return SimObject(name="d", shape="sphere", position=[0.0, 0.0, 0.3], size=[0.0], mass=0.2)

    def test_a_zero_volume_falls_back_to_the_default_density(self, caplog):
        sim = _engine()
        try:
            sim.create_world()
            with caplog.at_level("WARNING"):
                cfg = sim._shape_density_cfg(self._obj(), 0, 0.0)

            assert cfg is None
            warnings = [record.getMessage() for record in caplog.records if "non-positive" in record.getMessage()]
            assert warnings, [record.getMessage() for record in caplog.records]
            assert warnings[0].isascii()
        finally:
            sim.destroy()

    def test_a_non_finite_volume_falls_back(self):
        sim = _engine()
        try:
            sim.create_world()
            assert sim._shape_density_cfg(self._obj(), 0, float("nan")) is None
            assert sim._shape_density_cfg(self._obj(), 0, float("inf")) is None
        finally:
            sim.destroy()

    def test_a_static_shape_gets_no_config(self):
        sim = _engine()
        try:
            sim.create_world()
            assert sim._shape_density_cfg(self._obj(), -1, 1.0) is None
        finally:
            sim.destroy()

    def test_a_good_volume_yields_mass_over_volume(self):
        sim = _engine()
        try:
            sim.create_world()
            cfg = sim._shape_density_cfg(self._obj(), 0, 0.001)

            assert cfg is not None
            assert cfg.density == pytest.approx(200.0)
        finally:
            sim.destroy()


class TestMeshVolume:
    def test_a_unit_cube_mesh_has_volume_one(self):
        import numpy as np

        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        # Unit cube centred on the origin, 12 triangles, outward winding.
        verts = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.int32,
        ).reshape(-1)

        volume = NewtonSimEngine._mesh_volume(verts, faces, (1.0, 1.0, 1.0))

        assert volume == pytest.approx(1.0, rel=1e-5)

    def test_scale_multiplies_the_volume(self):
        import numpy as np

        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32).reshape(-1)

        unit = NewtonSimEngine._mesh_volume(verts, faces, (1.0, 1.0, 1.0))
        scaled = NewtonSimEngine._mesh_volume(verts, faces, (2.0, 3.0, 4.0))

        assert unit == pytest.approx(1.0 / 6.0, rel=1e-5)
        assert scaled == pytest.approx(unit * 24.0, rel=1e-5)

    def test_an_empty_mesh_has_zero_volume(self):
        import numpy as np

        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        empty = np.zeros((0, 3), dtype=np.float32)
        assert NewtonSimEngine._mesh_volume(empty, np.zeros(0, dtype=np.int32), (1.0, 1.0, 1.0)) == 0.0


class TestPhysicsStillBehaves:
    def test_a_dropped_cube_still_rests_on_the_ground(self):
        """The mass change must not break contact resolution."""
        sim = _engine()
        try:
            sim.create_world(gravity=[0.0, 0.0, -9.81])
            assert (
                sim.add_object("c", shape="box", position=[0.0, 0.0, 0.30], size=[0.02, 0.02, 0.02], mass=0.2)["status"]
                == "success"
            )

            sim.step(int(round(1.5 / sim.physics_timestep())))

            resting = float(sim._state_0.body_q.numpy()[0][2])
            assert resting == pytest.approx(0.02, abs=0.005), f"cube settled at z={resting:.4f}"
        finally:
            sim.destroy()

    def test_it_matches_the_mujoco_backend(self):
        """Cross-backend parity: the same request must give the same mass."""
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        newton_sim, newton_mass, _ = _build("box", [0.05, 0.05, 0.05], 0.5)
        mj_sim = MuJoCoSimEngine()
        try:
            mj_sim.create_world()
            assert (
                mj_sim.add_object("o", shape="box", position=[0.3, 0.0, 0.2], size=[0.05, 0.05, 0.05], mass=0.5)[
                    "status"
                ]
                == "success"
            )
            mj = mj_sim._mj
            body_id = mj.mj_name2id(mj_sim._world._model, mj.mjtObj.mjOBJ_BODY, "o")
            mj_mass = float(mj_sim._world._model.body_mass[body_id])

            assert newton_mass == pytest.approx(mj_mass, rel=1e-3), f"newton {newton_mass} vs mujoco {mj_mass}"
        finally:
            newton_sim.destroy()
            mj_sim.destroy()
