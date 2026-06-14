"""NVIDIA GR00T Whole-Body Control (WBC) policy for humanoid locomotion.

Runs ONNX-based Balance / Walk policies from the
``nvidia/GR00T-WholeBodyControl`` checkpoint. Targets Unitree G1 (29-DOF)
in MuJoCo with torque-mode (motor) actuators and PD position tracking.

The policy owns the physics stepping loop (``owns_stepping = True``) via an
action controller, matching the upstream NVLabs sim2mujoco reference.

Usage::

    from strands_robots.simulation.mujoco import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("unitree_g1", data_config="unitree_g1")
    sim.run_policy(
        robot_name="unitree_g1",
        policy_provider="wbc",
        policy_config={"checkpoint": "nvidia/GR00T-WholeBodyControl"},
        target_velocity=[0.5, 0.0, 0.0],
        duration=10.0,
    )
"""

from strands_robots.policies.wbc.wbc_policy import WBCPolicy

__all__ = ["WBCPolicy"]
