# Gymnasium Environment

Use Strands Robots simulations as standard [Gymnasium](https://gymnasium.farama.org/) environments for reinforcement learning.

---

## Quick Start

```python
import gymnasium as gym
from strands_robots.gymnasium import make_env

env = make_env("so100", task="reach")
obs, info = env.reset()

for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

---

## Standard Interface

The environment follows the Gymnasium API:

| Method | What it does |
|---|---|
| `reset()` | Reset environment, return initial observation |
| `step(action)` | Apply action, return (obs, reward, terminated, truncated, info) |
| `render()` | Render the current frame |
| `close()` | Clean up |

---

## Compatible With

Any RL library that supports Gymnasium:

- Stable Baselines 3
- CleanRL
- RLlib
- Custom training loops
