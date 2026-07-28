"""Newton must emit per-joint velocities like the MuJoCo backend.

``get_observation`` returned joint POSITIONS only. The MuJoCo backend has long
paired each position with an additive ``"<joint>.vel"`` entry, and real consumers
read it: ``WBCPolicy`` closes its balance loop on ``dqj`` (falling back to zeros
with a one-time warning when the keys are absent), and ``training.rl.SimEnv``
takes ``"<joint>.vel"`` directly in ``actor_obs_keys``. So a velocity-feedback
controller silently lost its feedback term on Newton while working on MuJoCo -
the worst shape of backend divergence, since nothing errors.

Newton already tracked the per-joint DOF index needed for this
(``_joint_dof_index``, distinct from ``_joint_coord_index`` because a free joint
spans 7 coordinates but 6 DOFs); it simply never read ``joint_qd`` for the scalar
joints.

Gated on Newton + Warp: the physics assertions step the real solver.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


def _make_engine(**kwargs):
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    return NewtonSimEngine(solver="mujoco", **kwargs)


def _vel_keys(obs: dict) -> set[str]:
    return {k for k in obs if k.endswith(".vel")}


class TestVelocityKeysPresent:
    def test_every_joint_has_a_velocity_entry(self):
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=5)
            obs = sim.get_observation(skip_images=True)

            joints = [k for k in obs if not k.endswith(".vel")]
            assert joints, "no joint positions reported"
            assert _vel_keys(obs) == {f"{j}.vel" for j in joints}
        finally:
            sim.destroy()

    def test_velocity_entries_are_plain_floats(self):
        """The observation contract is scalar floats, not numpy scalars."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=5)
            obs = sim.get_observation(skip_images=True)

            for key in _vel_keys(obs):
                assert type(obs[key]) is float, f"{key} is {type(obs[key])}"
        finally:
            sim.destroy()

    def test_matches_the_mujoco_backend_key_set(self):
        """Backend parity: the same robot must report the same scalar keys."""
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        newton_sim = _make_engine()
        mujoco_sim = MuJoCoSimEngine()
        try:
            for sim in (newton_sim, mujoco_sim):
                sim.create_world()
                sim.add_robot("so101")
                sim.step(n_steps=2)
            newton_obs = newton_sim.get_observation(skip_images=True)
            mujoco_obs = mujoco_sim.get_observation(skip_images=True)

            scalar = lambda obs: {k for k, v in obs.items() if isinstance(v, float)}  # noqa: E731
            assert scalar(newton_obs) == scalar(mujoco_obs)
        finally:
            newton_sim.destroy()
            mujoco_sim.destroy()


class TestVelocitiesAreRealPhysics:
    def test_moving_joints_report_nonzero_velocity_that_then_decays(self):
        """A commanded joint must show real velocity, and ~zero once settled.

        Guards against a plausible-but-fake implementation that reports a
        constant or a stale buffer: the value has to track the actual motion.
        """
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=5)
            sim.send_action({"1": 0.6, "2": -0.4, "3": 0.4, "4": 0.0, "5": 0.0, "6": 0.0})

            sim.step(n_steps=30)
            moving = sim.get_observation(skip_images=True)
            peak = max(abs(moving[k]) for k in _vel_keys(moving))
            assert peak > 0.1, f"no joint moved: peak |vel| {peak}"

            # Let the position servos settle on the target.
            sim.step(n_steps=1500)
            settled = sim.get_observation(skip_images=True)
            rest = max(abs(settled[k]) for k in _vel_keys(settled))
            assert rest < peak / 10.0, f"velocity did not decay: {rest} vs peak {peak}"
        finally:
            sim.destroy()

    def test_velocity_sign_follows_the_commanded_direction(self):
        """A joint driven negative must report a negative velocity while moving."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=5)
            sim.send_action({"1": -0.5, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0})
            sim.step(n_steps=25)

            assert sim.get_observation(skip_images=True)["1.vel"] < 0
        finally:
            sim.destroy()


class TestVelocityNoiseUsesTheVelocityStd:
    def test_velocity_entries_take_joint_vel_std_not_joint_pos_std(self):
        """Sensor noise must be routed by quantity, as the MuJoCo backend does.

        Position and velocity are different quantities in different units and
        their stds are configured independently, so applying the position std to
        a velocity would silently model the wrong sensor.
        """
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=5)
            # Noise ONLY on velocity: positions must come through clean.
            sim.set_obs_noise(joint_pos_std=0.0, joint_vel_std=5.0)

            obs = sim.get_observation(skip_images=True)
            # A 5.0-std velocity noise must dominate the settled (~0) velocities.
            assert max(abs(obs[k]) for k in _vel_keys(obs)) > 0.5
        finally:
            sim.destroy()

    def test_position_noise_does_not_leak_into_velocities(self):
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=5)
            # Huge POSITION noise, zero velocity noise: the settled velocities
            # must stay ~0 rather than picking up the position std.
            sim.set_obs_noise(joint_pos_std=10.0, joint_vel_std=0.0)

            obs = sim.get_observation(skip_images=True)
            assert max(abs(obs[k]) for k in _vel_keys(obs)) < 0.5
        finally:
            sim.destroy()
