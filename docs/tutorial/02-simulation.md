# Chapter 2: Simulation

**Time:** 15 minutes · **Hardware:** None needed · **Level:** Beginner

Simulation lets you test robot code without risking real hardware. Drop a robot in a virtual world, run physics, render images — all on your laptop.

---

## Why Simulate?

- **Safe** — crash all you want, nothing breaks
- **Fast** — run thousands of episodes in minutes
- **Free** — no $500 robot arm required
- **Reproducible** — same physics, same results, every time

---

## Create a Simulation

```python
from strands_robots import Simulation

sim = Simulation("so100")
sim.reset()
```

`Simulation("so100")` loads the SO-100's [MJCF](https://mujoco.readthedocs.io/en/latest/XMLreference.html) model into MuJoCo. The `reset()` puts everything back to the starting position.

---

## The Simulation Loop

```python
for step in range(100):
    # 1. Read what the robot sees
    obs = sim.get_observation()

    # 2. Decide on an action
    action = [0.1, -0.2, 0.0, 0.0, 0.0, 0.5]

    # 3. Apply it
    sim.apply_action(action)

    # 4. Step physics forward
    sim.step()
```

Each `step()` advances physics by one timestep (typically 2ms). The robot's joints move toward the target positions you set with `apply_action()`.

---

## Render an Image

```python
sim.render("frame.png")
```

This saves a rendered image of the current simulation state. Useful for debugging and visualization.

```python
# Render to a numpy array instead
image = sim.render()
print(image.shape)  # (480, 640, 3)
```

---

## Add Objects

A robot in an empty world isn't very useful. Add objects to interact with:

```python
sim = Simulation("so100", objects=["red_cube", "blue_cylinder"])
sim.reset()
```

The simulation includes common objects for manipulation tasks — cubes, cylinders, spheres — in various colors.

---

## Three Simulation Backends

Strands Robots supports three physics engines. They all use the same `Simulation()` interface:

| Backend | Best for | Platform |
|---|---|---|
| **MuJoCo** | Development, single robot | Mac / Linux / Windows |
| **Newton** | GPU-accelerated training | Linux + NVIDIA GPU |
| **Isaac Sim** | Massive parallel (1–100K+ envs) | Linux + NVIDIA GPU |

```python
# MuJoCo (default)
sim = Simulation("so100", backend="mujoco")

# Newton GPU
sim = Simulation("so100", backend="newton")

# Isaac Sim
sim = Simulation("so100", backend="isaac")
```

!!! tip "Start with MuJoCo"
    MuJoCo works everywhere and is perfect for learning. Switch to GPU backends when you need speed for training.

---

## What You Learned

- ✅ Simulation lets you test without hardware
- ✅ `Simulation("robot")` creates a virtual environment
- ✅ The loop: observe → act → step → repeat
- ✅ Render images for visualization
- ✅ Three backends: MuJoCo, Newton, Isaac Sim

---

**Next:** [Chapter 3: Policies →](03-policies.md) — Give your robot a brain.
