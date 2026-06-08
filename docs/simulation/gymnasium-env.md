---
description: Gymnasium adapter — status, design, when to expect it.
---

# Gymnasium env

A `gymnasium.Env` wrapper around `Simulation` is on the roadmap but not yet implemented
on `main`. This page documents what we plan to ship and why it's not there yet.

## Status

> **Not implemented yet.** Track this on the
> [issue tracker](https://github.com/strands-labs/robots/issues).

## What it will look like

```python
# Future API — subject to change
import gymnasium as gym
import strands_robots.gym  # registers strands-robots envs with gymnasium

env = gym.make("strands_robots/SO100Pick-v0", render_mode="rgb_array")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

The wrapper will be a thin layer over `Simulation` mapping:

- `env.reset()` → `sim.reset()` + `sim.randomize(...)` (configurable).
- `env.step(action)` → `sim.step()` after applying the action to the registered
  robot(s).
- Observation / action spaces derived from the registered `data_config` (so it lines
  up with the GR00T / LeRobot dataset schemas you're recording).

## Why not yet?

- `Simulation` already has the moving parts a Gymnasium env needs (`step`, `reset`,
  `render`, observation/action features). Wrapping is straightforward.
- The friction is **task definition**: which scene, which success predicate, which
  reward. We don't want to ship a half-dozen toy tasks that nobody actually trains on.
- The LIBERO benchmark adapter (in `strands_robots/benchmarks/libero/`) is a more
  realistic starting point — it already has hundreds of well-defined tasks.

## Workaround today

Use `Simulation` directly inside your training loop:

```python
sim = Robot("so100")
for episode in range(N):
    sim.reset()
    sim.randomize(randomize_colors=True, randomize_lighting=True)
    for step in range(T):
        action = my_policy(observation)
        # apply action via add_robot's controller
        sim.step(n_steps=1)
```

For benchmark suites, see the LIBERO adapter — it gives you a fixed task set, success
predicates, and a deterministic eval loop.

## See also

- [LIBERO benchmark source](https://github.com/strands-labs/robots/tree/main/strands_robots/benchmarks/libero) —
  current benchmark integration.
- [Simulation overview](overview.md) — what's already implemented.
- [Issue tracker](https://github.com/strands-labs/robots/issues) — track Gymnasium
  wrapper progress.
