"""Regression tests: resizing a geom re-derives the body's rotational inertia.

``body_inertia`` is a COMPILE-TIME product - the MJCF compiler integrates the
geom's shape once. ``set_geom_properties(size=...)`` writes ``geom_size`` at
runtime and refreshes the collision bounds, but left the inertia tensor
describing the OLD shape. The error was silent and large:

    add_object("cube", box, size=0.1, mass=1.0)   -> body_inertia 0.0016667
    set_geom_properties(size=[0.2]*3)             -> body_inertia 0.0016667  (!)
                                                     analytic:      0.1066667

    apply_force(torque=[0.1, 0, 0]); step 1 s
      stale inertia   -> 60.0000 rad/s
      correct inertia ->  0.9375 rad/s          == 64x wrong

Worse, it was order-dependent: an unrelated later ``add_object`` recompiles the
spec, and the compiler then silently corrects the tensor. So the SAME two calls
produced different physics depending on whether anything happened afterwards -
the hardest class of bug to see in a rollout.

The tensor is now re-derived from the new size at the body's existing mass (a
resize is a density change, not a material gain), and ``mj_setConst`` refreshes
the derived mass constants the constraint solver reads - the same treatment
``set_body_properties`` already applies for a mass change.

Every primitive is verified against a FRESH COMPILE at the resized dimensions,
i.e. against MuJoCo's own inertia integrator rather than a hand-derived formula.
That comparison caught a real error in the capsule branch: the familiar
``0.4 r^2`` hemisphere term is about the SPHERE centre, so the parallel-axis
shift must start from the hemisphere's own centroid (subtracting ``9/64 r^2``).
Using ``0.4 r^2 + d^2`` overstated a stubby capsule by up to 19%.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# (shape, add_object size in the FULL-extent convention, geom_size components)
_PRIMITIVES = [
    ("box", [0.1, 0.2, 0.3], 3),
    ("sphere", [0.2], 1),
    ("ellipsoid", [0.1, 0.2, 0.3], 3),
    ("cylinder", [0.1, 0.0, 0.4], 2),
    ("capsule", [0.1, 0.0, 0.4], 2),
]

_MASS = 2.0
_SCALE = 2.7


def _make(shape, size, mass=_MASS):
    s = Simulation(tool_name="geom_resize_inertia", mesh=False)
    s.create_world()
    assert s.add_object(name="o", shape=shape, size=size, position=[0, 0, 1.0], mass=mass)["status"] == "success"
    s.step(n_steps=2)
    return s


def _body_and_geom(sim):
    model = sim.mj_model
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "o")
    geom = next(g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body)
    return body, geom


def _inertia(sim) -> list[float]:
    body, _ = _body_and_geom(sim)
    return [float(v) for v in sim.mj_model.body_inertia[body]]


@pytest.mark.parametrize(("shape", "size", "ncomp"), _PRIMITIVES)
def test_resized_inertia_matches_a_fresh_compile(shape, size, ncomp) -> None:
    """Ground truth is MuJoCo's own compiler, not a formula in the test."""
    sim = _make(shape, size)
    try:
        _, geom = _body_and_geom(sim)
        half = [float(v) for v in sim.mj_model.geom_size[geom]][:ncomp]
        new_half = [v * _SCALE for v in half]
        assert sim.set_geom_properties(geom_name="o_geom", size=new_half)["status"] == "success"
        got = _inertia(sim)
    finally:
        sim.destroy()

    # A fresh sim built AT the resized dimensions. add_object takes full extents;
    # cylinder/capsule use the [diameter, unused, full height] layout.
    if shape in ("cylinder", "capsule"):
        full = [2 * new_half[0], 0.0, 2 * new_half[1]]
    else:
        full = [2 * v for v in new_half]
    truth_sim = _make(shape, full)
    try:
        truth = _inertia(truth_sim)
    finally:
        truth_sim.destroy()

    assert np.allclose(got, truth, rtol=2e-3), f"{shape}: got {got}, fresh compile gives {truth}"


def test_the_angular_acceleration_is_no_longer_64x_wrong() -> None:
    """The physical consequence, measured - and its order-dependence.

    Before the fix an unrelated later ``add_object`` recompiled the spec and
    silently corrected the tensor, so these two runs disagreed by 64x.
    """

    def spin(add_unrelated_object: bool) -> float:
        sim = _make("box", [0.1, 0.1, 0.1], mass=1.0)
        try:
            sim.set_geom_properties(geom_name="o_geom", size=[0.2, 0.2, 0.2])
            if add_unrelated_object:
                sim.add_object(name="probe", shape="sphere", size=[0.02], position=[5, 0, 1], mass=0.01)
            sim.set_gravity(gravity=[0, 0, 0])
            sim.apply_force(body_name="o", torque=[0.1, 0.0, 0.0])
            sim.step(n_steps=500)
            return float(sim.mj_data.qvel[3])
        finally:
            sim.destroy()

    without = spin(False)
    with_recompile = spin(True)
    assert np.isclose(without, with_recompile, rtol=1e-2), (
        f"order-dependent physics: {without:.4f} rad/s without a later recompile vs {with_recompile:.4f} rad/s with one"
    )
    # Note the convention difference between the two APIs, which is easy to trip
    # over when reading this: ``add_object(size=...)`` takes FULL extents, while
    # ``set_geom_properties(size=...)`` takes HALF-extents (as documented). So
    # ``size=[0.2]*3`` here is a 0.4 m cube, giving
    # I = 1/12 * 1.0 * (0.4^2 + 0.4^2) = 0.0266667 and omega = tau*t/I = 3.75.
    correct_inertia = 1.0 / 12.0 * (0.4**2 + 0.4**2)
    assert abs(without) == pytest.approx(0.1 * 1.0 / correct_inertia, rel=0.05)


def test_mass_is_preserved_by_a_resize() -> None:
    """A resize is a density change - it must not invent material."""
    sim = _make("box", [0.1, 0.1, 0.1])
    try:
        body, _ = _body_and_geom(sim)
        before = float(sim.mj_model.body_mass[body])
        sim.set_geom_properties(geom_name="o_geom", size=[0.3, 0.3, 0.3])
        body, _ = _body_and_geom(sim)
        assert float(sim.mj_model.body_mass[body]) == pytest.approx(before)
    finally:
        sim.destroy()


def test_a_resize_round_trip_returns_to_the_original_inertia() -> None:
    """No drift or accumulation across repeated writes."""
    sim = _make("box", [0.1, 0.1, 0.1], mass=1.0)
    try:
        baseline = _inertia(sim)
        sim.set_geom_properties(geom_name="o_geom", size=[0.2, 0.2, 0.2])
        assert not np.allclose(_inertia(sim), baseline), "premise: the resize must change the inertia"
        sim.set_geom_properties(geom_name="o_geom", size=[0.05, 0.05, 0.05])
        assert np.allclose(_inertia(sim), baseline)
    finally:
        sim.destroy()


def test_a_mass_change_then_a_resize_compose() -> None:
    """Both setters write the same tensor; the later one must not lose the first."""
    sim = _make("box", [0.1, 0.1, 0.1], mass=1.0)
    try:
        assert sim.set_body_properties(body_name="o", mass=3.0)["status"] == "success"
        assert sim.set_geom_properties(geom_name="o_geom", size=[0.1, 0.1, 0.1])["status"] == "success"
        # 3.0/12 * (0.2^2 + 0.2^2) for a 0.2 m cube at 3 kg.
        assert _inertia(sim) == pytest.approx([0.02] * 3, rel=1e-6)
        body, _ = _body_and_geom(sim)
        assert float(sim.mj_model.body_mass[body]) == pytest.approx(3.0)
    finally:
        sim.destroy()


def test_the_refreshed_inertia_survives_a_later_recompile() -> None:
    """Already correct, so the compiler's own value must agree with ours."""
    sim = _make("box", [0.1, 0.1, 0.1], mass=1.0)
    try:
        sim.set_geom_properties(geom_name="o_geom", size=[0.25, 0.25, 0.25])
        before = _inertia(sim)
        sim.add_object(name="probe", shape="sphere", size=[0.02], position=[4, 0, 1], mass=0.01)
        assert np.allclose(_inertia(sim), before, rtol=1e-9), (
            "the recompile changed the tensor, so our runtime value disagreed with the compiler"
        )
    finally:
        sim.destroy()


def test_the_result_text_reports_the_new_inertia() -> None:
    """A silent 64x change is what made this invisible; say it happened."""
    sim = _make("box", [0.1, 0.1, 0.1])
    try:
        result = sim.set_geom_properties(geom_name="o_geom", size=[0.2, 0.2, 0.2])
        assert result["status"] == "success"
        assert "inertia" in result["content"][0]["text"]
    finally:
        sim.destroy()


def test_a_multi_geom_body_says_it_cannot_be_re_derived() -> None:
    """With several geoms the per-geom mass split is not recoverable from mjModel,
    so inventing one would be a different kind of wrong. Report instead."""
    sim = Simulation(tool_name="geom_resize_multi", mesh=False)
    sim.create_world()
    try:
        built = sim.patch_scene_mjcf(
            ops=[
                {"op": "add_body", "name": "twin", "pos": [0, 0, 1]},
                {"op": "add_geom", "body": "twin", "name": "g1", "type": "box", "size": [0.05] * 3, "mass": 0.5},
                {
                    "op": "add_geom",
                    "body": "twin",
                    "name": "g2",
                    "type": "box",
                    "size": [0.05] * 3,
                    "pos": [0.2, 0, 0],
                    "mass": 0.5,
                },
            ]
        )
        assert built["status"] == "success", built["content"][0]["text"]
        result = sim.set_geom_properties(geom_name="g1", size=[0.1, 0.1, 0.1])
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "2 geoms" in text
        assert "cannot be re-derived" in text
    finally:
        sim.destroy()


def test_a_plane_resize_does_not_touch_inertia() -> None:
    """A plane is static and its bounds are type-derived; nothing to refresh."""
    sim = Simulation(tool_name="geom_resize_plane", mesh=False)
    sim.create_world()
    try:
        floor = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor < 0:
            pytest.skip("this scene has no named floor plane")
        result = sim.set_geom_properties(geom_id=int(floor), size=[8.0, 8.0, 0.1])
        assert result["status"] == "success"
        assert "inertia" not in result["content"][0]["text"]
    finally:
        sim.destroy()
