---
description: Gymnasium adapter — status, design, when to expect it.
---

# Gymnasium env

> **Not implemented.** Track on the [issue tracker](https://github.com/strands-labs/robots/issues).

Planned API:

```python
# Future API — subject to change
import gymnasium as gym
import strands_robots.gym   # registers strands-robots envs with gymnasium

env = gym.make("strands_robots/SO100Pick-v0", render_mode="rgb_array")
obs, info = env.reset()

for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
```

## Workaround

Use `Simulation` directly:

```python
from strands_robots import Robot

sim = Robot("so100")
for episode in range(N):
    sim.reset()
    sim.randomize(randomize_colors=True, randomize_lighting=True)
    for step in range(T):
        sim.step(n_steps=1)
```

For a task suite with success predicates, see the LIBERO benchmark adapter.

## See also

- [Simulation overview](overview.md)
- [LIBERO benchmark source](https://github.com/strands-labs/robots/tree/main/strands_robots/benchmarks/libero)
- [Issue tracker](https://github.com/strands-labs/robots/issues)
