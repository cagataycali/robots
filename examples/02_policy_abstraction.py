#!/usr/bin/env python3
"""One function to load any policy - mock, GR00T, ACT, MolmoAct2, Cosmos3.

Goal: Demonstrate create_policy() as the universal entry point. The provider
is auto-resolved from the string: "mock", an HF repo, or a ZMQ URL.

Dependencies: pip install "strands-robots[sim-mujoco]"
Expected output: Policy loaded and run for 20 steps; prints action keys.
Runtime: ~1 second (mock provider, no GPU needed).
"""

from strands_robots import Robot, create_policy

# Robot("so100") already builds the world and adds the "so100" robot.
sim = Robot("so100", mesh=False)
sim.add_camera(name="front", position=[0.5, 0.0, 0.4], target=[0.2, 0, 0.05])

# create_policy("mock") -> MockPolicy (sinusoidal test actions)
# create_policy("lerobot/act_aloha_sim") -> LerobotLocalPolicy (HF inference)
# create_policy("zmq://localhost:5555") -> Gr00tPolicy (ZMQ client)
# create_policy("allenai/MolmoAct2-SO100_101", embodiment="so_real") -> MolmoAct2
policy = create_policy("mock")

print(f"Policy: {type(policy).__name__}")
print(f"Requires images: {getattr(policy, 'requires_images', True)}")

result = sim.run_policy(
    robot_name="so100",
    policy_object=policy,
    instruction="pick up the red cube",
    n_steps=20,
)
print(f"Status: {result['status']}")
