# Chapter 1: Your First Robot

**Time:** 10 minutes · **Hardware:** None needed · **Level:** Beginner

In this chapter, you'll install Strands Robots, create your first simulated robot, and see what it can do.

---

## Install

```bash
pip install "strands-robots[mujoco]"
```

This gives you the core library plus MuJoCo simulation — enough to get started with no hardware.

---

## Create a Robot

```python
from strands_robots import Robot

robot = Robot("so100")
print(robot)
```

That's it. `Robot("so100")` looks up "so100" in the registry, finds the SO-100 tabletop arm, and creates it. Since no USB hardware is detected, it automatically starts in simulation mode.

<figure markdown>
  ![SO-100 in simulation](../assets/so100_act_demo.gif){ width="480" }
  <figcaption>SO-100 running ACT policy in MuJoCo — what you get after <code>Robot("so100")</code></figcaption>
</figure>

!!! info "What is the SO-100?"
    The SO-100 is a 6-DOF (6 degrees of freedom) tabletop robot arm. It's affordable, open-source, and one of the most popular robots for learning robotics. Think of it as the "Hello World" of robot arms.

---

## Read an Observation

Every robot can tell you what it sees and feels:

```python
obs = robot.get_observation()
print(obs.keys())
# dict_keys(['joint_positions', 'joint_velocities', 'cameras'])
```

An **observation** is everything the robot knows right now:

- **`joint_positions`** — where each joint is (in radians)
- **`joint_velocities`** — how fast each joint is moving
- **`cameras`** — images from any attached cameras

---

## Send an Action

Tell the robot where to move its joints:

```python
import numpy as np

# Move to a specific joint configuration
action = np.array([0.0, -0.5, 0.3, 0.0, 0.2, 0.5])
robot.apply_action(action)
```

Each number is a target position (in radians) for one joint. The SO-100 has 6 joints, so we pass 6 numbers.

---

## The Robot Lifecycle

Every robot interaction follows the same loop:

```python
# 1. Create
robot = Robot("so100")

# 2. Observe
obs = robot.get_observation()

# 3. Decide (you, a policy, or an AI agent)
action = decide_what_to_do(obs)

# 4. Act
robot.apply_action(action)

# 5. Repeat from step 2
```

This **observe → decide → act** loop is the foundation of all robotics. Everything else in this tutorial builds on it.

---

## Try Other Robots

The registry has 38 robots. Try a few:

```python
# A humanoid
robot = Robot("unitree_g1")

# A quadruped
robot = Robot("spot")

# A bimanual system (two arms)
robot = Robot("aloha")
```

Each one has different joints, cameras, and capabilities — but the interface is always the same.

---

## What You Learned

- ✅ How to install Strands Robots
- ✅ `Robot("name")` creates a robot from the registry
- ✅ Auto-detection: no hardware → simulation
- ✅ Observations tell you what the robot sees and feels
- ✅ Actions tell the robot where to move
- ✅ The observe → decide → act loop

---

**Next:** [Chapter 2: Simulation →](02-simulation.md) — Learn how to use MuJoCo physics to build virtual robot worlds.
