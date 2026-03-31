#!/usr/bin/env python3
"""Strands Agent with full MuJoCo physics introspection.

Demonstrates the new Physics API — direct Python access to MuJoCo C functions:
  mj_ray, mj_jacBody, mj_applyFT, mj_fullM, mj_inverse, mj_contactForce,
  mj_getState/mj_setState, mj_energyPos/mj_energyVel, and more.

An agent that can reason about physics: cast rays, compute Jacobians,
apply forces, checkpoint/restore state, read sensors, and analyze contacts
— all through natural language.

Requirements:
    pip install strands-agents strands-robots[sim]

Usage:
    python examples/physics_agent.py
"""

from strands import Agent

from strands_robots import Robot

# Create a simulated SO-100 robot arm
sim = Robot("so100")

# Give the agent the simulation tool — all 50+ actions available via NL
agent = Agent(
    tools=[sim],
    system_prompt=(
        "You are a robotics physicist. You have a simulated SO-100 robot arm "
        "in MuJoCo. Use the simulation tool to explore its physics. "
        "Be concise and use real numbers from the simulation."
    ),
)

# ─── Example 1: Full physics analysis in natural language ────────────────────
print("=" * 70)
print("Example 1: Agent-driven physics analysis")
print("=" * 70)

result = agent(
    "Analyze the robot's physics: "
    "1) Get the total mass breakdown, "
    "2) Compute the mass matrix and tell me its condition number, "
    "3) Read all sensor values, "
    "4) Get the system energy. "
    "Summarize the physical properties."
)
print(result)

# ─── Example 2: Raycasting for obstacle detection ───────────────────────────
print("\n" + "=" * 70)
print("Example 2: Agent uses raycasting for spatial reasoning")
print("=" * 70)

result = agent(
    "Cast rays downward from 5 points above the robot (height=1m) at "
    "x=-0.2, -0.1, 0, 0.1, 0.2 (y=0) to map what's below. "
    "Use multi_raycast for efficiency. Report the distance map."
)
print(result)

# ─── Example 3: State checkpointing + force experiments ─────────────────────
print("\n" + "=" * 70)
print("Example 3: Save state → experiment → restore")
print("=" * 70)

result = agent(
    "I want to experiment with forces without breaking the sim: "
    "1) Save the current state as 'pristine', "
    "2) Step 200 times to let things settle, "
    "3) Get the energy, "
    "4) Apply a 50N upward force to the robot's end-effector body, "
    "5) Step 100 more times, "
    "6) Get the energy again and compare, "
    "7) Restore the 'pristine' state, "
    "8) Verify we're back by checking energy matches the original."
)
print(result)

# ─── Example 4: Jacobian + inverse dynamics ─────────────────────────────────
print("\n" + "=" * 70)
print("Example 4: Dynamics analysis")
print("=" * 70)

result = agent(
    "Compute the Jacobian for the end-effector and run inverse dynamics. "
    "What forces are needed at each joint to hold the current pose?"
)
print(result)

# ─── Example 5: Contact analysis ────────────────────────────────────────────
print("\n" + "=" * 70)
print("Example 5: Contact force analysis")
print("=" * 70)

result = agent(
    "Step the simulation 500 times to let everything settle, "
    "then get detailed contact forces. "
    "Which bodies are in contact and what are the normal forces?"
)
print(result)

# Clean up
sim.destroy()
print("\n✅ Done — all physics examples complete.")
