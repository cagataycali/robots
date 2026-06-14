"""Kimodo text-to-motion policy for humanoid robots (NVlabs/kimodo).

Generates full-body kinematic motion from natural language prompts using
NVIDIA's Kimodo diffusion model. Outputs MuJoCo-compatible qpos frames
that can be replayed directly or used as reference trajectories for a
downstream physics controller (e.g. WBC torque tracking).

Supported robots: Unitree G1 (via nvidia/Kimodo-G1-RP-v1 checkpoint).

Usage::

    from strands_robots.simulation.mujoco import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("unitree_g1", data_config="unitree_g1")
    sim.run_policy(
        robot_name="unitree_g1",
        policy_provider="kimodo",
        policy_config={
            "model": "nvidia/Kimodo-G1-RP-v1",
            "prompt": "walk forward then wave",
            "duration": 5.0,
        },
    )
"""

from strands_robots.policies.kimodo.kimodo_policy import KimodoPolicy

__all__ = ["KimodoPolicy"]
