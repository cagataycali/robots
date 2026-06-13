#!/usr/bin/env python3
"""Spawn a robot in MuJoCo and run a policy — zero hardware required.

Goal: Show that Robot("so100") gives you a fully configured MuJoCo simulation
with one call; add objects and cameras, run a policy, done.

Dependencies: pip install "strands-robots[sim-mujoco]"
Expected output: "run_policy status: success" after 50 steps of mock actions.
Runtime: ~2 seconds on CPU.
"""

from strands_robots import MockPolicy, Robot

# Robot() defaults to mode="sim" — safe, no hardware interaction.
sim = Robot("so100", mesh=False)

# Build a scene: table, red cube, front camera.
sim.create_world()
sim.add_robot(name="arm", data_config="so100")
sim.add_object(
    name="cube",
    shape="box",
    position=[0.2, 0.0, 0.05],
    size=[0.025, 0.025, 0.025],
    color=[1, 0, 0, 1],
    mass=0.05,
)
sim.add_camera(name="front", position=[0.5, 0.0, 0.4], target=[0.2, 0, 0.05])

# Run a policy — MockPolicy produces sinusoidal test actions.
result = sim.run_policy(
    robot_name="arm",
    policy_object=MockPolicy(),
    instruction="pick up the red cube",
    n_steps=50,
)

print(f"run_policy status: {result['status']}")
