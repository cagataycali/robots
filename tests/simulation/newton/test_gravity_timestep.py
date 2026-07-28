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


class TestDefaultTimestepIsTheFrameCadence:
    """``_DEFAULT_TIMESTEP`` must be the FRAME dt its own comment claims.

    The constant was ``1/600`` while documented as "60 Hz frames with 10
    substeps". ``_advance`` divides it by ``substeps``, so the solver ran at
    6000 Hz - twelve times the 500 Hz the so101 MJCF asks for via
    ``opt.timestep=0.002``. And because ``PolicyRunner._control_substeps``
    derives frames-per-action from ``physics_timestep()``, a 50 Hz control loop
    became ``round((1/50)/(1/600)) = 12`` frames of 10 substeps each: 120 solver
    steps per control action.

    Measured on an NVIDIA Thor, one 50 Hz control action on so101::

        dt=1/600 substeps=10 solver=6000Hz n_substeps=12  641.6 ms  0.031x realtime
        dt=1/60  substeps=10 solver= 600Hz n_substeps= 1   55.0 ms  0.364x realtime

    11.7x, and the physics that matters is unchanged (see
    ``TestCoarserTimestepKeepsPhysicsCorrect``).
    """

    def test_default_timestep_is_the_frame_rate_not_the_solver_rate(self):
        sim = _make_engine()
        try:
            sim.create_world()
            assert sim.physics_timestep() == pytest.approx(1.0 / 60.0)
        finally:
            sim.destroy()

    def test_the_solver_rate_stays_in_the_documented_range(self):
        """timestep/substeps is the solver dt; 600 Hz is the example cadence."""
        sim = _make_engine()
        try:
            sim.create_world()
            solver_hz = 1.0 / (sim.physics_timestep() / sim.substeps)
            assert solver_hz == pytest.approx(600.0)
        finally:
            sim.destroy()

    def test_a_50hz_control_loop_takes_one_frame_per_action(self):
        """Regression: this derived 12 frames (120 solver steps) per action."""
        from strands_robots.simulation.policy_runner import PolicyRunner

        sim = _make_engine()
        try:
            sim.create_world()
            assert sim.add_robot("so101")["status"] == "success"

            n_substeps = PolicyRunner(sim)._control_substeps(50.0)

            assert n_substeps <= 2, f"{n_substeps} frames per 50Hz action means {n_substeps * 10} solver steps"
        finally:
            sim.destroy()

    def test_the_old_rate_is_still_reachable_explicitly(self):
        """No silent lock-in: a caller wanting 6 kHz integration can ask."""
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        sim = NewtonSimEngine(solver="mujoco", default_timestep=1.0 / 600.0)
        try:
            sim.create_world()
            assert sim.physics_timestep() == pytest.approx(1.0 / 600.0)
        finally:
            sim.destroy()


class TestCoarserTimestepKeepsPhysicsCorrect:
    """The default must be fast AND right; these are the fidelity guards.

    Measured across the candidates (free-fall error over 0.5s / cube resting
    height on the ground plane / so101 position-servo tracking error)::

        6000 Hz : 0.03%  +0.0199  0.0010 rad
         600 Hz : 0.33%  +0.0199  0.0009 rad   <- the new default
        1200 Hz : 0.17%  +0.0199  0.0009 rad
    """

    def test_free_fall_still_matches_the_analytic_solution(self):
        sim = _make_engine()
        try:
            sim.create_world(gravity=[0.0, 0.0, -9.81])
            sim.add_object("ball", shape="sphere", position=[0.0, 0.0, 5.0], size=[0.05], mass=0.2)
            z0 = _ball_z(sim)
            t_sim = 0.5

            sim.step(int(round(t_sim / sim.physics_timestep())))

            dropped = z0 - _ball_z(sim)
            analytic = 0.5 * 9.81 * t_sim**2
            # 1% covers the 0.33% measured plus the frame-quantisation of t_sim.
            assert dropped == pytest.approx(analytic, rel=0.01), f"fell {dropped:.4f}m, expected {analytic:.4f}m"
        finally:
            sim.destroy()

    def test_a_dropped_cube_rests_on_the_ground_instead_of_tunnelling(self):
        """The classic coarse-timestep failure: penetrating the ground plane."""
        sim = _make_engine()
        try:
            sim.create_world(gravity=[0.0, 0.0, -9.81])
            sim.add_object("cube", shape="box", position=[0.0, 0.0, 0.30], size=[0.02, 0.02, 0.02], mass=0.2)

            sim.step(int(round(1.5 / sim.physics_timestep())))

            resting = _ball_z(sim)
            # Half-extent is 0.02, so a resting cube sits at ~+0.02.
            assert resting == pytest.approx(0.02, abs=0.005), f"cube settled at z={resting:.4f}"
        finally:
            sim.destroy()

    def test_a_position_servo_joint_still_reaches_its_target(self):
        """The reason the old comment gave for the high rate."""
        sim = _make_engine()
        try:
            sim.create_world()
            assert sim.add_robot("so101")["status"] == "success"
            joints = sim.robot_joint_names("so101")
            target = 0.6

            assert sim.send_action({joints[0]: target}, robot_name="so101")["status"] == "success"
            sim.step(int(round(2.0 / sim.physics_timestep())))

            reached = float(sim.get_observation("so101")[joints[0]])
            assert reached == pytest.approx(target, abs=0.02), f"servo reached {reached:.4f}, target {target}"
        finally:
            sim.destroy()
