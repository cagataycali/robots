"""Gravity honouring and physics-parameter setters for the Newton backend.

The Newton backend previously built its model without writing the configured
gravity vector onto the finalised model, so ``create_world(gravity=...)`` was
silently ignored and every world fell under Newton's built-in default. These
tests pin that a configured gravity vector actually drives the dynamics and
that ``set_gravity`` / ``set_timestep`` mirror the MuJoCo backend's contract.

Gated on Newton + a usable compute device: the dynamics assertions step the
real solver, so they are skipped when Newton/Warp are unavailable.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


def _make_engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    return NewtonSimEngine(solver="mujoco")


def _ball_z(engine) -> float:
    """Return the current world-z of the single free body."""
    return float(engine._state_0.body_q.numpy()[0][2])


def _ball_x(engine) -> float:
    return float(engine._state_0.body_q.numpy()[0][0])


class TestGravityHonoured:
    def test_zero_gravity_keeps_ball_static(self):
        """Regression: zero gravity must leave a free body at rest.

        Pre-fix the configured gravity was dropped, so the ball fell under the
        engine default even when zero gravity was requested.
        """
        sim = _make_engine()
        try:
            sim.create_world(gravity=[0.0, 0.0, 0.0])
            sim.add_object("ball", shape="sphere", position=[0.0, 0.0, 0.5], size=[0.05], mass=0.2)
            z0 = _ball_z(sim)
            sim.step(50)
            assert _ball_z(sim) == pytest.approx(z0, abs=1e-3)
        finally:
            sim.destroy()

    def test_negative_gravity_makes_ball_fall(self):
        sim = _make_engine()
        try:
            sim.create_world(gravity=[0.0, 0.0, -9.81])
            sim.add_object("ball", shape="sphere", position=[0.0, 0.0, 0.5], size=[0.05], mass=0.2)
            z0 = _ball_z(sim)
            sim.step(50)
            assert _ball_z(sim) < z0 - 1e-2
        finally:
            sim.destroy()

    def test_inverted_gravity_makes_ball_rise(self):
        sim = _make_engine()
        try:
            sim.create_world(gravity=[0.0, 0.0, 5.0])
            sim.add_object("ball", shape="sphere", position=[0.0, 0.0, 0.5], size=[0.05], mass=0.2)
            z0 = _ball_z(sim)
            sim.step(50)
            assert _ball_z(sim) > z0 + 1e-3
        finally:
            sim.destroy()

    def test_non_axis_aligned_gravity_drives_lateral_drift(self):
        """A gravity vector with an x-component must move the body in x.

        Newton's builder only expresses gravity as a scalar along its up-axis;
        the full vec3 is written onto the finalised model so off-axis
        components are not silently dropped.
        """
        sim = _make_engine()
        try:
            sim.create_world(gravity=[3.0, 0.0, -9.81])
            sim.add_object("ball", shape="sphere", position=[0.0, 0.0, 0.5], size=[0.05], mass=0.2)
            x0 = _ball_x(sim)
            sim.step(50)
            assert _ball_x(sim) > x0 + 1e-3
        finally:
            sim.destroy()


class TestSetGravity:
    def test_set_gravity_scalar_is_z_component(self):
        sim = _make_engine()
        try:
            sim.create_world()
            result = sim.set_gravity(0.0)
            assert result["status"] == "success"
            assert sim.describe()["gravity"] == [0.0, 0.0, 0.0]
            sim.add_object("ball", shape="sphere", position=[0.0, 0.0, 0.5], size=[0.05], mass=0.2)
            z0 = _ball_z(sim)
            sim.step(50)
            assert _ball_z(sim) == pytest.approx(z0, abs=1e-3)
        finally:
            sim.destroy()

    def test_set_gravity_rejects_wrong_length(self):
        sim = _make_engine()
        try:
            sim.create_world()
            result = sim.set_gravity([1.0, 2.0])
            assert result["status"] == "error"
            assert "3-element" in result["content"][0]["text"]
        finally:
            sim.destroy()

    def test_set_gravity_rejects_non_finite(self):
        sim = _make_engine()
        try:
            sim.create_world()
            result = sim.set_gravity([0.0, 0.0, float("inf")])
            assert result["status"] == "error"
            assert "finite" in result["content"][0]["text"]
        finally:
            sim.destroy()

    def test_set_gravity_without_world_errors(self):
        sim = _make_engine()
        result = sim.set_gravity([0.0, 0.0, -9.81])
        assert result["status"] == "error"
        assert "create_world" in result["content"][0]["text"]


class TestSetTimestep:
    def test_set_timestep_updates_world(self):
        sim = _make_engine()
        try:
            sim.create_world()
            result = sim.set_timestep(0.002)
            assert result["status"] == "success"
            assert sim.physics_timestep() == pytest.approx(0.002)
        finally:
            sim.destroy()

    def test_set_timestep_warns_on_large_value(self):
        sim = _make_engine()
        try:
            sim.create_world()
            result = sim.set_timestep(0.5)
            assert result["status"] == "success"
            assert "unusually large" in result["content"][0]["text"]
        finally:
            sim.destroy()

    def test_set_timestep_rejects_non_positive(self):
        sim = _make_engine()
        try:
            sim.create_world()
            result = sim.set_timestep(0.0)
            assert result["status"] == "error"
            assert "positive" in result["content"][0]["text"]
        finally:
            sim.destroy()
