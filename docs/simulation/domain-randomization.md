---
description: What randomize() actually samples - colors, lighting, physics, positions.
---

# Domain randomization

```python
sim.randomize(
    randomize_colors=True,      # resample object/floor RGB from color_range
    randomize_lighting=True,    # perturb directional + ambient light
    randomize_physics=False,    # mass (mass_range) + friction (friction_range) + damping
    randomize_positions=False,  # add position_noise (m) to every object position
    position_noise=0.02,
    color_range=(0.1, 1.0),
    friction_range=(0.5, 1.5),
    mass_range=(0.5, 2.0),
    seed=42,                    # deterministic sequence
)
```

**Destructive** - writes into MuJoCo model arrays. To restore: `load_scene(...)` or recreate the sim.

`randomize()` leaves the sim in a forwarded, render-ready state: the next `render()` / `get_observation()` reflects the perturbation immediately, with no manual `step()` in between. This matters for lighting in particular - the renderer reads light positions from the derived `data.light_xpos`, not `model.light_pos`, so a light-position jitter only reaches a render after a forward.

## Categories

| Flag | What changes | Range param |
|------|-------------|-------------|
| `randomize_colors` | Object + floor RGB (alpha fixed at 1.0) | `color_range` |
| `randomize_lighting` | Directional direction, intensity, ambient | - |
| `randomize_physics` | Per-object mass (mult), per-geom friction (scale), joint damping | `mass_range`, `friction_range` |
| `randomize_positions` | Object position offsets (metres) | `position_noise` |

Defaults: `colors=True`, `lighting=True`; `physics` and `positions` default `False`.

## Use in an eval loop

```python
for episode in range(N):
    sim.reset()
    sim.randomize(randomize_colors=True, randomize_physics=True, seed=episode)
    # eval_policy has no randomize= kwarg - call sim.randomize() before each episode
    result = sim.eval_policy(robot_name="so100", n_episodes=1, max_steps=300,
                             success_fn=my_fn)
```

## Newton backend

The Newton (GPU) backend mirrors this `randomize` contract for the axes it
supports (colors, lighting, physics) and adds `set_obs_noise` for additive
sensor noise. See [Newton backend](newton.md#domain-randomization-and-sensor-noise).

## See also

- [Simulation overview](overview.md)
- [World building](world-building.md)
- [Recording](../recording.md)
- [Real hardware](../hardware/robot-control.md)
